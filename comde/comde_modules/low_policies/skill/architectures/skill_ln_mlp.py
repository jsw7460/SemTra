from typing import List, Callable

import flax.linen as nn
import jax.numpy as jnp

from comde.utils.jax_utils.type_aliases import (
	PRNGKey,
	Shape,
	Dtype,
	Array
)
from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.common.utils import create_mlp


class PrimSkillLnMLP(nn.Module):
	act_scale: float
	hidden_size: int
	output_dim: int
	net_arch: List
	activation_fn: nn.Module = nn.relu
	dropout: float = 0.0
	squash_output: bool = False
	layer_norm: bool = False
	batch_norm: bool = False
	use_bias: bool = True
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal()
	bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

	emb_obs = None
	emb_skill = None
	emb_nonfunc = None
	emb_param = None
	ln = None
	mlp = None

	def setup(self) -> None:
		self.emb_obs = nn.Dense(self.hidden_size)
		self.emb_skill = nn.Dense(self.hidden_size)
		self.emb_nonfunc = nn.Dense(self.hidden_size)
		self.emb_param = nn.Dense(self.hidden_size)
		self.ln = nn.LayerNorm(self.hidden_size * 4)	# 4 = num of input ingradient (obs, skill, nonfunc, param)
		mlp = create_mlp(
			output_dim=self.output_dim,
			net_arch=self.net_arch,
			activation_fn=self.activation_fn,
			dropout=self.dropout,
			squash_output=self.squash_output,
			layer_norm=self.layer_norm,
			batch_norm=self.batch_norm,
			use_bias=self.use_bias,
			kernel_init=self.kernel_init,
			bias_init=self.bias_init
		)
		self.mlp = Scaler(base_model=mlp, scale=jnp.array(self.act_scale))

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		observations: jnp.ndarray,	# observations
		skills: jnp.ndarray,  # [b, l, d]
		non_functionality: jnp.ndarray,	# [b, l, d]
		parameters: jnp.ndarray,	# [b, l, d]
		deterministic: bool = False,
		training: bool = True,
		*args, **kwargs		# Do not remove this
	):
		obs = self.emb_obs(observations)
		skills = self.emb_skill(skills)
		non_func = self.emb_nonfunc(non_functionality)
		param = self.emb_param(parameters)

		mlp_input = jnp.concatenate((obs, skills, non_func, param), axis=-1)
		mlp_input = self.ln(mlp_input)

		y = self.mlp(mlp_input, deterministic=deterministic, training=training)
		return y
