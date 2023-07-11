from typing import Dict

import einops
import flax.linen as nn
import jax.numpy as jnp
import transformers

from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.low_policies.naive.architectures.customed.gpt_modules import FlaxGPT2ModuleWoTimePosEmb


class PrimFlaxVIMA(nn.Module):
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
	emb_prompt = None
	emb_lang = None
	emb_seq = None
	emb_nf = None
	emb_prm = None
	emb_ln = None

	transformer = None
	pred_obs = None
	pred_act = None

	def setup(self) -> None:
		gpt2_config = transformers.GPT2Config(**self.gpt2_config, n_embd=self.hidden_size)
		self.transformer = FlaxGPT2ModuleWoTimePosEmb(gpt2_config, dtype=jnp.float32)
		self.emb_obs = nn.Dense(self.hidden_size)
		self.emb_act = nn.Dense(self.hidden_size)
		self.emb_time = nn.Embed(self.max_ep_len, self.hidden_size)
		self.emb_prompt = nn.Dense(self.hidden_size)  # On top of the tokenizer (e.g., T5, BERT, ...)
		self.emb_prm = nn.Dense(self.hidden_size)  # Embed (optimal) parameter
		self.emb_ln = nn.LayerNorm(self.hidden_size)

		pred_act = nn.Dense(self.act_dim)
		self.pred_act = Scaler(base_model=pred_act, scale=jnp.array(self.act_scale))

	def __call__(self, *args, **kwargs):
		y = self.forward_with_all_components(*args, **kwargs)
		return y["action_preds"]

	def forward_with_all_components(
		self,
		observations: jnp.ndarray,  # [b, l, d] (Query)
		actions: jnp.ndarray,  # [b, l, d] (Query)
		maskings: jnp.ndarray,  # state-action mask	[b, l]
		timesteps: jnp.ndarray,
		param_for_skills: jnp.ndarray,  # [b, M, d]
		prompts: jnp.ndarray,  # [b, l', d] (Key-value)
		prompts_maskings: jnp.ndarray,  # [b, l']
		deterministic: bool = False
	):
		batch_size = observations.shape[0]
		subseq_len = observations.shape[1]
		n_tar_skills = param_for_skills.shape[1]

		timesteps_emb = self.emb_time(timesteps)
		observations_emb = self.emb_obs(observations)
		actions_emb = self.emb_act(actions)
		observations_emb = observations_emb + timesteps_emb
		actions_emb = actions_emb + timesteps_emb

		traj_tokens = jnp.stack((observations_emb, actions_emb), axis=1)  # [b, 2, l, d]
		traj_tokens = einops.rearrange(traj_tokens, "b c l d -> b l c d")
		traj_tokens = traj_tokens.reshape(batch_size, 2 * subseq_len, self.hidden_size)
		traj_tokens = self.emb_ln(traj_tokens)

		traj_masks = jnp.stack((maskings, maskings), axis=1)  # [b, 2, l]
		traj_masks = einops.rearrange(traj_masks, "b c l -> b l c")
		traj_masks = traj_masks.reshape(batch_size, 2 * subseq_len)

		prompts_emb = self.emb_prompt(prompts)
		params_emb = self.emb_prm(param_for_skills)

		prompts_tokens = jnp.concatenate((prompts_emb, params_emb), axis=-2)
		prompts_tokens = self.emb_ln(prompts_tokens)
		prompts_masks = jnp.concatenate((prompts_maskings, jnp.ones((batch_size, n_tar_skills))), axis=-1)

		transformer_outputs = self.transformer(
			hidden_states=traj_tokens,  # Query
			attention_mask=traj_masks,  # Query mask
			encoder_hidden_states=prompts_tokens,	# Key-value
			encoder_attention_mask=prompts_masks,	# Key-value mask
			deterministic=deterministic
		)
		x = transformer_outputs["last_hidden_state"]
		x = x.reshape(batch_size, subseq_len, 2, self.hidden_size)
		x = einops.rearrange(x, "b l c d -> b c l d")

		action_preds = self.pred_act(x[:, 0])
		ret_info = {"action_preds": action_preds}
		return ret_info
