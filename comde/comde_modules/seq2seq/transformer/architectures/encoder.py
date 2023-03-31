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
		self.pos_encoding = PositionalEncoding(d_model=self.input_dim, max_len=self.max_len + 1)
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

	def forward(self, x: jnp.ndarray, mask: jnp.ndarray, deterministic: bool):

		x = self.input_dropout(x, deterministic=deterministic)
		x = self.input_layer(x)
		x = self.pos_encoding(x)

		for block in self.encoder_blocks:  # Todo: Change into lax_fori
			x = block(x=x, mask=mask, deterministic=deterministic)

		return x

	def get_attention_maps(self, x: jnp.ndarray, mask: jnp.ndarray, deterministic: bool):
		# A function to return the attention maps within the model for a single application
		# Used for visualization purpose later
		attention_maps = []

		for block in self.encoder_blocks:
			_, attn_map = block.self_attention(x=x, mask=mask)
			attention_maps.append(attn_map)
			x = block(x=x, mask=mask, deterministic=deterministic)

		return attention_maps
