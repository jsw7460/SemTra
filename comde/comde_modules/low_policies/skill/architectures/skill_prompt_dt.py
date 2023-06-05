from typing import Dict

import einops
import flax.linen as nn
import jax.numpy as jnp
import transformers

from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.low_policies.naive.architectures.customed.gpt_modules import FlaxGPT2ModuleWoTimePosEmb


EPS = 1e-6

class PrimSkillPromptDT(nn.Module):
	"""
		Prompt - Decision Transformer
	"""
	gpt2_config: Dict
	obs_dim: int
	act_dim: int
	hidden_size: int
	act_scale: float
	max_ep_len: int

	normalization_mean: float = 0.0
	normalization_std: float = 1.0

	emb_time = None
	emb_obs = None
	emb_act = None
	emb_skill = None
	emb_prompt = None
	emb_prm = None
	emb_ln = None

	transformer = None
	pred_obs = None
	pred_act = None

	def setup(self) -> None:
		gpt2_config = transformers.GPT2Config(**self.gpt2_config, n_embd=self.hidden_size)
		self.transformer = FlaxGPT2ModuleWoTimePosEmb(gpt2_config, dtype=jnp.float32)
		self.emb_time = nn.Embed(self.max_ep_len, self.hidden_size)
		self.emb_obs = nn.Dense(self.hidden_size)
		self.emb_skill = nn.Dense(self.hidden_size)
		self.emb_prompt = nn.Dense(self.hidden_size)
		self.emb_act = nn.Dense(self.hidden_size)
		self.emb_ln = nn.LayerNorm(self.hidden_size)

		self.pred_obs = nn.Dense(self.obs_dim)
		pred_act = create_mlp(
			output_dim=self.act_dim,
			net_arch=[],
			squash_output=True
		)
		self.pred_act = Scaler(base_model=pred_act, scale=jnp.array(self.act_scale))

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(self, *args, **kwargs):
		""" Predict only action """
		predictions = self.forward_with_all_components(*args, **kwargs)
		return predictions[0]

	def forward_with_all_components(
		self,
		observations: jnp.ndarray,  # [b, l, d]
		actions: jnp.ndarray,  # [b, l, d]
		skills: jnp.ndarray,	# [b, l, d]
		prompts: jnp.ndarray,  # [b, P, d]
		prompts_maskings: jnp.ndarray, # [b, P]
		timesteps: jnp.ndarray,  # [b, l]
		maskings: jnp.ndarray,  # [b, l]
		deterministic: bool = True,
	):

		observations = (observations - self.normalization_mean) / (self.normalization_std + 1E-12)

		batch_size = observations.shape[0]
		subseq_len = observations.shape[1]

		observations_emb = self.emb_obs(observations)
		actions_emb = self.emb_act(actions)
		skills_emb = self.emb_skill(skills)

		prompts_emb = self.emb_prompt(prompts)  # [b, M, d]
		timesteps_emb = self.emb_time(timesteps)

		observations_emb = observations_emb + timesteps_emb
		actions_emb = actions_emb + timesteps_emb
		skills_emb = skills_emb + timesteps_emb

		# this makes the sequence look like (sk_1, s_1, a_1, sk_2, s_2, a_2, ...)
		# which works nice in an autoregressive sense since observations predict actions
		stacked_inputs = jnp.stack((skills_emb, observations_emb, actions_emb), axis=1)  # [b, 3, l, d]

		stacked_inputs = einops.rearrange(stacked_inputs, "b c l d -> b l c d")  # [b, l, 3, d]
		stacked_inputs = stacked_inputs.reshape(batch_size, 3 * subseq_len, self.hidden_size)  # [b, 3 * l, d]
		stacked_inputs = jnp.concatenate((prompts_emb, stacked_inputs), axis=1)
		stacked_inputs = self.emb_ln(stacked_inputs)

		stacked_masks = jnp.stack((maskings, maskings, maskings), axis=1)  # [b, 3, l]
		stacked_masks = einops.rearrange(stacked_masks, "b c l -> b l c")
		stacked_masks = stacked_masks.reshape(batch_size, 3 * subseq_len)
		stacked_masks = jnp.concatenate((prompts_maskings, stacked_masks), axis=1)

		transformer_outputs = self.transformer(
			hidden_states=stacked_inputs,
			attention_mask=stacked_masks,
			deterministic=deterministic
		)
		x = transformer_outputs["last_hidden_state"]

		# reshape x so that the second dimension corresponds to the original
		# skills(0), observations(1), actions(2)

		# Truncate the prompt
		prompt_length = prompts_emb.shape[1]
		x = x[:, prompt_length:, ...]
		x = x.reshape(batch_size, subseq_len, 3, self.hidden_size)
		x = einops.rearrange(x, "b l c d -> b c l d")

		obs_preds = self.pred_obs(x[:, 2])
		action_preds = self.pred_act(x[:, 1])

		additional_info = {
			"observations_emb": observations_emb,
			"actions_emb": actions_emb,
			"prompts_emb": prompts_emb,
			"timesteps_emb": timesteps_emb,
		}

		return action_preds, obs_preds, additional_info


def get_prompt(
	attention_weights: jnp.ndarray,	# [b, n_target_skills, length_of_language (48)]
	language_guidance: jnp.ndarray,	# [b, length_of_language, lang_dim]
	target_skills_mask: jnp.ndarray,	# [b, n_target_skills]
	language_guidance_mask: jnp.ndarray	# [b, length_of_language]
):
	attention_weights = jnp.expand_dims(attention_weights, axis=-1)  # [b, n_tar_sk, len_lang, 1]
	language_guidance = jnp.expand_dims(language_guidance, axis=1)  # [b, 1, len_lang, lang_dim]
	attention_weights = jnp.where(attention_weights == 0.0, EPS, attention_weights)
	prompt = attention_weights * language_guidance  # [b, n_tar_sk, len_lang, lamg_dim]

	target_skills_mask = jnp.expand_dims(target_skills_mask, axis=-1)
	language_guidance_mask = jnp.expand_dims(language_guidance_mask, axis=1)
	prompt_mask = jnp.logical_and(target_skills_mask, language_guidance_mask)

	prompt = einops.rearrange(prompt, "b n_sk l d -> b l n_sk d")
	prompt_mask = einops.rearrange(prompt_mask, "b n_sk l -> b l n_sk")

	batch_size = prompt.shape[0]
	lang_dim = prompt.shape[-1]

	prompt = prompt.reshape(batch_size, -1, lang_dim)
	prompt_mask = prompt_mask.reshape(batch_size, -1,)

	return {"prompts": prompt, "prompts_maskings": prompt_mask}