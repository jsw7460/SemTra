from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from comde.comde_modules.seq2seq.transformer.architectures.decoder_block import DecoderBlock


class TransformerDecoder(nn.Module):
	num_layers: int
	input_dim: int
	num_heads: int
	ff_dim: int
	dropout_prob: float
	use_bias: bool
	activation_fn: Callable

	decoder_blocks = None

	def setup(self) -> None:
		self.decoder_blocks = [
			DecoderBlock(
				input_dim=self.input_dim,
				num_heads=self.num_heads,
				ff_dim=self.ff_dim,
				dropout_prob=self.dropout_prob,
				use_bias=self.use_bias,
				activation_fn=self.activation_fn
			)
		]

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(self, x: jnp.ndarray, kv: jnp.ndarray, mask: jnp.ndarray, deterministic: bool):
		"""
			x: [batch_size, target_seq_len]
		"""
		for block in self.decoder_blocks:
			x = block(q=x, kv=kv, mask=mask, deterministic=deterministic)
		return x
