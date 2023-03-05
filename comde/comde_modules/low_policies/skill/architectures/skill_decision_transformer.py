from typing import Dict

import einops
import flax.linen as nn
import jax.numpy as jnp
import transformers

from comde.comde_modules.low_policies.naive.architectures.customed.gpt_modules import FlaxGPT2ModuleWoTimePosEmb
from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.common.utils import create_mlp


class PrimSkillDecisionTransformer(nn.Module):
	"""
		Decision Transformer with skill
	"""
	gpt2_config: Dict
	obs_dim: int
	act_dim: int
	skill_dim: int
	hidden_size: int
	act_scale: float
	max_ep_len: int

	emb_time = None
	emb_obs = None
	emb_act = None
	emb_skill = None
	emb_ret = None
	emb_ln = None

	transformer = None
	pred_obs = None
	pred_act = None
	pred_skill = None
	pred_ret = None

	def setup(self) -> None:
		gpt2_config = transformers.GPT2Config(**self.gpt2_config)
		self.transformer = FlaxGPT2ModuleWoTimePosEmb(gpt2_config, dtype=jnp.float32)

		self.emb_time = nn.Embed(self.max_ep_len, self.hidden_size)
		self.emb_obs = nn.Dense(self.hidden_size)
		self.emb_skill = nn.Dense(self.hidden_size)
		self.emb_act = nn.Dense(self.hidden_size)
		self.emb_ret = nn.Dense(self.hidden_size)
		self.emb_ln = nn.LayerNorm(self.hidden_size)

		self.pred_obs = nn.Dense(self.obs_dim)
		pred_act = create_mlp(
			output_dim=self.act_dim,
			net_arch=[64, 64],
			squash_output=True
		)
		self.pred_act = Scaler(base_model=pred_act, scale=self.act_scale)
		self.pred_skill = nn.Dense(self.skill_dim)
		self.pred_ret = nn.Dense(1)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		observations: jnp.ndarray,  # [b, l, d]
		actions: jnp.ndarray,  # [b, l, d]
		skills: jnp.ndarray, 	# [b, l, d]
		timesteps: jnp.ndarray,  # [b, l]
		maskings: jnp.ndarray,  # [b, l]
		rtgs: jnp.ndarray,
		deterministic: bool = True
	):
		batch_size = observations.shape[0]
		subseq_len = observations.shape[1]

		observations_emb = self.emb_obs(observations)
		actions_emb = self.emb_act(actions)
		skills_emb = self.emb_skill(skills)
		rtgs_emb = self.emb_ret(rtgs)
		timesteps_emb = self.emb_time(timesteps)

		observations_emb = observations_emb + timesteps_emb
		actions_emb = actions_emb + timesteps_emb
		skills_emb = skills_emb + timesteps_emb
		rtgs_emb = rtgs_emb + timesteps_emb

		# this makes the sequence look like (R_1, s_1, sk_1, a_1, R_2, s_2, sk_2, a_2, ...)
		# which works nice in an autoregressive sense since observations predict actions
		stacked_inputs = jnp.stack((rtgs_emb, observations_emb, skills_emb, actions_emb), axis=1)  # [b, 4, l, d]
		stacked_inputs = einops.rearrange(stacked_inputs, "b c l d -> b l c d")  # [b, l, 4, d]
		stacked_inputs = stacked_inputs.reshape(batch_size, 4 * subseq_len, self.hidden_size)  # [b, 4 * l, d]
		stacked_inputs = self.emb_ln(stacked_inputs)

		stacked_masks = jnp.stack((maskings, maskings, maskings, maskings), axis=1)  # [b, 3, l]
		stacked_masks = einops.rearrange(stacked_masks, "b c l -> b l c")
		stacked_masks = stacked_masks.reshape(batch_size, 4 * subseq_len)

		transformer_outputs = self.transformer(
			hidden_states=stacked_inputs,
			attention_mask=stacked_masks,
			deterministic=deterministic
		)
		x = transformer_outputs["last_hidden_state"]

		# reshape x so that the second dimension corresponds to the original
		# rtgs (0), observations (1), skills(2), or actions (3); i.e. x[:,1,t] is the token for s_t
		x = x.reshape(batch_size, subseq_len, 4, self.hidden_size)
		x = einops.rearrange(x, "b l c d -> b c l d")

		obs_preds = self.pred_obs(x[:, 3])
		action_preds = self.pred_act(x[:, 2])
		ret_preds = self.pred_ret(x[:, 3])

		return obs_preds, action_preds, ret_preds
