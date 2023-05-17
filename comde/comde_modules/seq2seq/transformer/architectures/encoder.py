from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.seq2seq.transformer.architectures.encoder_block import EncoderBlock
from comde.comde_modules.seq2seq.transformer.architectures.positional_encoding import PositionalEncoding


class TransformerEncoder(nn.Module):
	num_layers: int
	input_dim: int
	num_heads: int
	ff_dim: int
	dropout_prob: float
	use_bias: bool
	max_len: int  # Maximum possible length of sequence
	activation_fn: Callable

	input_dropout = None
	input_layer = None
	pos_encoding = None
	encoder_blocks = None

	def setup(self) -> None:
		self.input_dropout = nn.Dropout(self.dropout_prob)
		self.input_layer = create_mlp(self.input_dim, [])
		self.pos_encoding = PositionalEncoding(d_model=self.input_dim, max_len=self.max_len + 50)
		self.encoder_blocks = [
			EncoderBlock(
				input_dim=self.input_dim,
				num_heads=self.num_heads,
				ff_dim=self.ff_dim,
				dropout_prob=self.dropout_prob,
				use_bias=self.use_bias,
				activation_fn=self.activation_fn
			) for _ in range(self.num_layers)
		]

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(self, q: jnp.ndarray, kv: jnp.ndarray, q_mask: jnp.ndarray, kv_mask: jnp.ndarray, deterministic: bool):

		q = self.input_dropout(q, deterministic=deterministic)
		kv = self.input_dropout(kv, deterministic=deterministic)
		q = self.pos_encoding(q)

		for block in self.encoder_blocks:  # Todo: Change into lax_fori
			q = block(q=q, kv=kv, q_mask=q_mask, kv_mask=kv_mask, deterministic=deterministic)

		return q

	def get_attention_maps(
		self,
		q: jnp.ndarray,
		kv: jnp.ndarray,
		q_mask: jnp.ndarray,
		kv_mask: jnp.ndarray,
		deterministic: bool
	):
		"""..."""
