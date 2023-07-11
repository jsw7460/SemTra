from typing import Dict

import einops
import flax.linen as nn
import jax.numpy as jnp
import transformers

from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.low_policies.naive.architectures.customed.gpt_modules import FlaxGPT2ModuleWoTimePosEmb


class PrimSensorEncoder(nn.Module):
	"""
		Decision Transformer with skill
	"""
	gpt2_config: Dict
	n_skill: int
	n_param: int	# Discrete output !
	hidden_size: int
	max_ep_len: int

	normalization_mean: float = 0.0
	normalization_std: float = 1.0

	use_timestep: bool = True

	emb_obs = None
	emb_time = None
	emb_act = None
	emb_skill = None
	emb_ln = None

	transformer = None
	pred_skill = None
	pred_param = None

	def setup(self) -> None:
		gpt2_config = transformers.GPT2Config(**self.gpt2_config, n_embd=self.hidden_size)
		self.transformer = FlaxGPT2ModuleWoTimePosEmb(gpt2_config, dtype=jnp.float32)
		self.emb_time = nn.Embed(self.max_ep_len, self.hidden_size)
		self.emb_obs = nn.Dense(self.hidden_size)
		self.emb_skill = nn.Dense(self.hidden_size)
		self.emb_act = nn.Dense(self.hidden_size)
		self.emb_ln = nn.LayerNorm(self.hidden_size)

		self.pred_skill = nn.Dense(self.n_skill)
		self.pred_param = nn.Dense(self.n_param)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(self, *args, **kwargs):
		""" Predict only action """
		predictions = self.forward_with_all_components(*args, **kwargs)
		return predictions

	def forward_with_all_components(
		self,
		observations: jnp.ndarray,  # [b, l, d]
		actions: jnp.ndarray,  # [b, l, d]
		maskings: jnp.ndarray,  # [b, l]
		deterministic: bool = True,
	):

		observations = (observations - self.normalization_mean) / (self.normalization_std + 1E-12)

		batch_size = observations.shape[0]
		subseq_len = observations.shape[1]

		observations_emb = self.emb_obs(observations)
		actions_emb = self.emb_act(actions)

		stacked_inputs = jnp.stack((observations_emb, actions_emb), axis=1)  # [b, 3, l, d]

		stacked_inputs = einops.rearrange(stacked_inputs, "b c l d -> b l c d")  # [b, l, 3, d]
		stacked_inputs = stacked_inputs.reshape(batch_size, 2 * subseq_len, self.hidden_size)  # [b, 3 * l, d]
		stacked_inputs = self.emb_ln(stacked_inputs)

		stacked_masks = jnp.stack((maskings, maskings), axis=1)  # [b, 3, l]
		stacked_masks = einops.rearrange(stacked_masks, "b c l -> b l c")
		stacked_masks = stacked_masks.reshape(batch_size, 2 * subseq_len)

		transformer_outputs = self.transformer(
			hidden_states=stacked_inputs,
			attention_mask=stacked_masks,
			deterministic=deterministic
		)
		x = transformer_outputs["last_hidden_state"]

		# reshape x so that the second dimension corresponds to the original
		# observations (0), skills(1), or actions (2); i.e. x[:,1,t] is the token for s_t
		x = x.reshape(batch_size, subseq_len, 2, self.hidden_size)
		x = einops.rearrange(x, "b l c d -> b c l d")

		skill_preds = self.pred_skill(x[:, 2])
		param_preds = self.pred_param(x[:, 2])

		output = {
			"param_preds": param_preds,
			"skill_preds": skill_preds,
			"observations_emb": observations_emb,
			"actions_emb": actions_emb
		}

		return output
