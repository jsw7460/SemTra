from typing import List, Callable

import flax.linen as nn
import jax.numpy as jnp

from comde.jax_utils.type_aliases import (
	PRNGKey,
	Shape,
	Dtype,
	Array
)


class MLP(nn.Module):
	net_arch: List
	activation_fn: nn.Module
	dropout: float = 0.0
	squash_output: bool = False

	layer_norm: bool = False
	batch_norm: bool = False
	use_bias: bool = True
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal()
	bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

	@nn.compact
	def __call__(self, x: jnp.ndarray, deterministic: bool = False, training: bool = True):
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

		if self.squash_output:
			return nn.tanh(x)
		else:
			return x
