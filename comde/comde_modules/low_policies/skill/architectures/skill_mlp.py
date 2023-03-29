from typing import List, Callable

import flax.linen as nn
import jax.numpy as jnp

from comde.utils.jax_utils.type_aliases import (
	PRNGKey,
	Shape,
	Dtype,
	Array
)
<<<<<<< HEAD
from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.common.utils import create_mlp


class PrimSkillMLP(nn.Module):
	act_scale: float
	output_dim: int
	net_arch: List
	activation_fn: nn.Module
	dropout: float = 0.0
	squash_output: bool = False
	layer_norm: bool = False
	batch_norm: bool = False
	use_bias: bool = True
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal()
	bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

	mlp = None

	def setup(self) -> None:
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
		self.mlp = Scaler(base_model=mlp, scale=self.act_scale)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		observations: jnp.ndarray,
		skills: jnp.ndarray,  # [b, l, d]
		deterministic: bool = False,
		training: bool = True,
		*args, **kwargs		# Do not remove this
	):
		mlp_input = jnp.concatenate((observations, skills), axis=-1)
		y = self.mlp(mlp_input, deterministic=deterministic, training=training)
		return y

=======

class PrimSkillMLP(nn.Module):
    net_arch: List
    activation_fn: nn.Module
    dropout: float = 0.0
    squash_output: bool = False

    layer_norm: bool = False
    batch_norm: bool = False
    use_bias: bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @nn.compact
    def forward(
        self, 
        observations: jnp.ndarray, 
		actions: jnp.ndarray,  # [b, l, d]
		skills: jnp.ndarray, 	# [b, l, d]
		timesteps: jnp.ndarray,  # [b, l]
		maskings: jnp.ndarray,  # [b, l]
        deterministic: bool = False, 
        training: bool = True):
    
        x = jnp.concatenate((observations, skills),axis=-1)
        for features in self.net_arch[: -1]:
            x = nn.Dense(features=features, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
            x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)

        if len(self.net_arch) > 0:
            x = nn.Dense(features=self.net_arch[-1], kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        # if self.squash_output:
        #     return nn.tanh(x)
        # else:
        #     return x

        # to match transformer's output shape, use: "None, x, None"
        if self.squash_output:
            return None,nn.tanh(x),None
        else:
            return None,x,None
>>>>>>> be19079ebe55ea9a065abe791c5f0f5aa5467473
