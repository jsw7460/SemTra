from typing import Dict

import einops
import flax.linen as nn
import jax.numpy as jnp
import transformers

from architectures.customed.gpt_modules import FlaxGPT2ModuleWoTimePosEmb
from comde.arch_modules.common.scaler import Scaler
from comde.arch_modules.common.utils import create_mlp


class DecisionTransformer(nn.Module):
	"""
		Basic Decision transformer model (No skill)
	"""
	gpt2_config: Dict
	state_dim: int
	act_dim: int
	hidden_size: int
	act_scale: float
	max_ep_len: int

	emb_time = None
	emb_state = None
	emb_act = None
	emb_ret = None
	emb_ln = None

	transformer = None
	pred_state = None
	pred_act = None
	pred_ret = None

	def setup(self) -> None:
		gpt2_config = transformers.GPT2Config(**self.gpt2_config)
		self.transformer = FlaxGPT2ModuleWoTimePosEmb(gpt2_config, dtype=jnp.float32)

		self.emb_time = nn.Embed(self.max_ep_len, self.hidden_size)
		self.emb_state = nn.Dense(self.hidden_size)
		self.emb_act = nn.Dense(self.hidden_size)
		self.emb_ret = nn.Dense(self.hidden_size)
		self.emb_ln = nn.LayerNorm(self.hidden_size)

		self.pred_state = nn.Dense(self.state_dim)
		pred_act = create_mlp(
			output_dim=self.act_dim,
			net_arch=[64, 64],
			squash_output=True
		)
		self.pred_act = Scaler(base_model=pred_act, scale=self.act_scale)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		states: jnp.ndarray,  # [b, l, d]
		actions: jnp.ndarray,  # [b, l, d]
		timesteps: jnp.ndarray,  # [b, l]
		masks: jnp.ndarray,  # [b, l]
		returns: jnp.ndarray,
		deterministic: bool = True
	):
		batch_size = states.shape[0]
		subseq_len = states.shape[1]

		states_emb = self.emb_state(states)
		actions_emb = self.emb_act(actions)
		returns_emb = self.emb_ret(returns)
		timesteps_emb = self.emb_time(timesteps)

		states_emb = states_emb + timesteps_emb
		actions_emb = actions_emb + timesteps_emb
		returns_emb = returns_emb + timesteps_emb

		# this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
		# which works nice in an autoregressive sense since states predict actions
		stacked_inputs = jnp.stack((returns_emb, states_emb, actions_emb), dim=1)  # [b, 3, l, d]
		stacked_inputs = einops.rearrange(stacked_inputs, "b c l d -> b l c d")  # [b, l, 3, d]
		stacked_inputs = stacked_inputs.reshape(batch_size, 3 * subseq_len, self.hidden_size)  # [b, 3 * l, d]
		stacked_inputs = self.emb_ln(stacked_inputs)

		stacked_masks = jnp.stack((masks, masks, masks), dim=1)  # [b, 3, l]
		stacked_masks = einops.rearrange(stacked_masks, "b c l -> b l c")
		stacked_masks = stacked_masks.reshape(batch_size, 3 * subseq_len)

		transformer_outputs = self.transformer(
			hidden_states=stacked_inputs,
			attention_mask=stacked_masks,
			deterministic=deterministic
		)
		x = transformer_outputs["last_hidden_state"]

		# reshape x so that the second dimension corresponds to the original
		# returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
		x = x.reshape(batch_size, subseq_len, 3, self.hidden_size)
		x = einops.rearrange(x, "b l c d -> b c l d")

		pred_state = self.pred_state(x[:, 2])
		pred_act = self.pred_act(x[:, 1])
		pred_ret = self.pred_ret(x[:, 2])

		return pred_state, pred_act, pred_ret
