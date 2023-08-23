from typing import List, Callable, Tuple

from flax import linen as nn
from flax.linen.initializers import zeros
from jax import numpy as jnp

from comde.comde_modules.common.utils import create_mlp
from comde.utils.jax_utils.general import uniform


class LSTM(nn.Module):
	"""
	This model is for fixed length lstm
	"""
	subseq_len: int
	embed_dim: int
	hidden_dim: int
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
		deterministic: bool = False
	) -> Tuple[jnp.ndarray, jnp.ndarray]:
		"""
		:param sequence: [batch_size, sequence_length, dimension] is expected
		:param deterministic
		"""
		batch_size = sequence.shape[0]
		rng = self.make_rng("init_carry")
		carry = self.initialize_carry(rng, batch_dim=batch_size)  # [batch_size, dim]
		embedded_input = self.embed_net(sequence, deterministic=deterministic)
		stacked_output = jnp.zeros((batch_size, 1, self.hidden_dim))

		for timestep in range(self.subseq_len):
			current_component = embedded_input[:, timestep, :]
			carry, output = self.lstmcell(carry, current_component)
			stacked_output = jnp.concatenate((stacked_output, output[:, jnp.newaxis, ...]), axis=1)

		return stacked_output, carry
