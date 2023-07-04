from typing import List, Callable, Tuple

from flax import linen as nn
from flax.linen.initializers import zeros
from jax import numpy as jnp

from comde.comde_modules.common.utils import create_mlp
from comde.utils.jax_utils.general import uniform


class PrimLSTM(nn.Module):
	max_iter_len: int
	embed_dim: int  # Embed whole sequences
	hidden_dim: int  # Dimension of output of LSTM
	dropout: float
	activation_fn: Callable
	embed_net_arch: List
	carry_initializer: Callable = zeros
	embed_net_layernorm: bool = False

	embed_net = None  # Input is embedded by embed_net
	lstmcell = None

	def setup(self) -> None:
		self.embed_net = create_mlp(
			output_dim=self.embed_dim,
			net_arch=self.embed_net_arch,
			activation_fn=self.activation_fn,
			dropout=self.dropout,
			squash_output=False,
			layer_norm=self.embed_net_layernorm,
			batch_norm=False,
			use_bias=True
		)
		self.lstmcell = nn.LSTMCell(kernel_init=uniform(jnp.sqrt(1 / self.hidden_dim)))

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def initialize_carry(self, rng: jnp.ndarray, batch_dim: int):
		carry = self.lstmcell.initialize_carry(rng, (batch_dim,), self.hidden_dim, init_fn=self.carry_initializer)
		return carry

	def forward(
		self,
		sequence: jnp.ndarray,
		batch_size: int,
		deterministic: bool = False
	) -> Tuple[jnp.ndarray, jnp.ndarray]:
		"""
		:param sequence: [batch_size, sequence_length, dimension] is expected
		:param batch_size:
		:param deterministic
		"""
		rng = self.make_rng("init_carry")
		carry = self.initialize_carry(rng, batch_dim=batch_size)  # [batch_size, dim]
		embedded_input = self.embed_net(sequence, deterministic=deterministic)
		stacked_outputs = []
		for timestep in range(self.max_iter_len):
			current_component = embedded_input[:, timestep, :]
			carry, output = self.lstmcell(carry, current_component)
			stacked_outputs.append(output[:, jnp.newaxis, ...])
		stacked_output = jnp.concatenate(stacked_outputs, axis=1)	# [b, max_iter_len, d]
		return stacked_output
