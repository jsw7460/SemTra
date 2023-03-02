from typing import List, Callable

import flax.linen as nn

from comde.jax_utils.type_aliases import *
from mm_sbrl.component_jax.architectures.mlp import MLP


class LeakyReLu:
	def __init__(self, negative_slope: float = 1e-2):
		self.negative_slope = negative_slope

	def __call__(self, *args, **kwargs):
		return nn.leaky_relu(*args, **kwargs, negative_slope=self.negative_slope)


def create_mlp(
	output_dim: int,
    net_arch: List[int],
    activation_fn: Callable = nn.relu,
    dropout: float = 0.0,
    squash_output: bool = False,
    layer_norm: bool = False,
    batch_norm: bool = False,
    use_bias: bool = True,
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal(),
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
):
	if output_dim > 0:
		net_arch = list(net_arch)
		net_arch.append(output_dim)

	return MLP(
		net_arch=net_arch,
		activation_fn=activation_fn,
		dropout=dropout,
		squash_output=squash_output,
		layer_norm=layer_norm,
		batch_norm=batch_norm,
		use_bias=use_bias,
		kernel_init=kernel_init,
		bias_init=bias_init
	)
