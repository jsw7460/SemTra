from typing import Dict

import einops
import flax.linen as nn
import jax.numpy as jnp
import transformers

from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.low_policies.naive.architectures.customed.gpt_modules import FlaxGPT2ModuleWoTimePosEmb


class PrimSensorPromptDT(nn.Module):
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

	only_prompt: bool = False
	use_timestep: bool = True

	emb_time = None
	emb_obs = None
	emb_act = None
	emb_rtg = None
	emb_prompt = None
	emb_seq = None
	emb_nf = None
	emb_prm = None
	emb_ln = None

	emb_sensor_prompt = None
	emb_lang_prompt = None

	transformer = None
	pred_obs = None
	pred_act = None

	def setup(self) -> None:
		gpt2_config = transformers.GPT2Config(**self.gpt2_config, n_embd=self.hidden_size)
		self.transformer = FlaxGPT2ModuleWoTimePosEmb(gpt2_config, dtype=jnp.float32)
		self.emb_time = nn.Embed(self.max_ep_len, self.hidden_size)
		self.emb_obs = nn.Dense(self.hidden_size)
		self.emb_prompt = nn.Dense(self.hidden_size)
		self.emb_act = nn.Dense(self.hidden_size)
		self.emb_rtg = nn.Dense(self.hidden_size)
		self.emb_seq = nn.Dense(self.hidden_size)  # Embed sequential requirements
		self.emb_nf = nn.Dense(self.hidden_size)  # Embed non-functionality
		self.emb_prm = nn.Dense(self.hidden_size)  # Embed parameter
		self.emb_ln = nn.LayerNorm(self.hidden_size)

		self.emb_sensor_prompt = nn.Dense(self.hidden_size)
		self.emb_lang_prompt = nn.Dense(self.hidden_size)

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
		rtgs: jnp.ndarray,  # [b, l, 1]

		sensor_prompts: jnp.ndarray,
		sensor_prompts_maskings: jnp.ndarray,
		language_prompts: jnp.ndarray,  # [b, M, d]
		language_prompts_maskings: jnp.ndarray,  # [b, M]

		sequential_requirement: jnp.ndarray,  # [b, d]
		non_functionality: jnp.ndarray,  # [b, d]
		param_for_skills: jnp.ndarray,  # [b, M, d]
		timesteps: jnp.ndarray,  # [b, l]
		maskings: jnp.ndarray,  # [b, l]
		deterministic: bool = True,
	):
		observations = (observations - self.normalization_mean) / (self.normalization_std + 1E-12)

		batch_size = observations.shape[0]
		subseq_len = observations.shape[1]

		observations_emb = self.emb_obs(observations)
		actions_emb = self.emb_act(actions)
		rtgs_emb = self.emb_rtg(rtgs)

		seq_emb = jnp.expand_dims(self.emb_seq(sequential_requirement), axis=1)  # [b, 1, d]
		nf_emb = jnp.expand_dims(self.emb_nf(non_functionality), axis=1)  # [b, 1, d]
		prm_emb = self.emb_prm(param_for_skills)  # [b, M, d]

		sensor_prompts = self.emb_sensor_prompt(sensor_prompts)
		language_prompts = self.emb_lang_prompt(language_prompts)

		prompts = jnp.concatenate((sensor_prompts, language_prompts), axis=-2)
		prompts_maskings = jnp.concatenate((sensor_prompts_maskings, language_prompts_maskings), axis=-1)

		prompt_emb = self.emb_prompt(prompts)  # [b, M, d]
		prompt = jnp.concatenate((prompt_emb, seq_emb, nf_emb, prm_emb),
								 axis=1)  # [b, p, d] p: prompt length # 36 + 1 + 1 + 4
		if self.only_prompt:
			seq_nf_mask = jnp.zeros((batch_size, 2))
		else:
			seq_nf_mask = jnp.ones((batch_size, 2))

		l_prm = prm_emb.shape[1]
		prm_maskings = jnp.ones((batch_size, l_prm))

		prompt_maskings = jnp.concatenate((prompts_maskings, seq_nf_mask, prm_maskings), axis=1)
		if self.use_timestep:
			timesteps_emb = self.emb_time(timesteps)
		else:
			timesteps_emb = 0.0

		observations_emb = observations_emb + timesteps_emb
		actions_emb = actions_emb + timesteps_emb
		rtgs_emb = rtgs_emb + timesteps_emb

		# this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
		# which works nice in an autoregressive sense since observations predict actions

		stacked_inputs = jnp.stack((rtgs_emb, observations_emb, actions_emb), axis=1)  # [b, 3, l, d]

		stacked_inputs = einops.rearrange(stacked_inputs, "b c l d -> b l c d")  # [b, l, 3, d]
		stacked_inputs = stacked_inputs.reshape(batch_size, 3 * subseq_len, self.hidden_size)  # [b, 3 * l, d]	# 15
		stacked_inputs = jnp.concatenate((prompt, stacked_inputs), axis=1)
		stacked_inputs = self.emb_ln(stacked_inputs)

		stacked_masks = jnp.stack((maskings, maskings, maskings), axis=1)  # [b, 3, l]
		stacked_masks = einops.rearrange(stacked_masks, "b c l -> b l c")
		stacked_masks = stacked_masks.reshape(batch_size, 3 * subseq_len)
		stacked_masks = jnp.concatenate((prompt_maskings, stacked_masks), axis=1)

		transformer_outputs = self.transformer(
			hidden_states=stacked_inputs,
			attention_mask=stacked_masks,
			deterministic=deterministic
		)
		x = transformer_outputs["last_hidden_state"]

		# reshape x so that the second dimension corresponds to the original
		# rtgs(0), observations(1), actions(2)

		# Truncate the prompt
		prompt_length = prompt.shape[1]
		x = x[:, prompt_length:, ...]
		x = x.reshape(batch_size, subseq_len, 3, self.hidden_size)
		x = einops.rearrange(x, "b l c d -> b c l d")

		obs_preds = self.pred_obs(x[:, 2])
		action_preds = self.pred_act(x[:, 1])

		additional_info = {
			"observations_emb": observations_emb,
			"actions_emb": actions_emb,
			"prompts_emb": prompt_emb,
			"timesteps_emb": timesteps_emb,
		}

		return action_preds, obs_preds, additional_info
