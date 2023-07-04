from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.seq2seq.transformer.architectures.decoder_block import DecoderBlock
from comde.comde_modules.seq2seq.transformer.architectures.positional_encoding import PositionalEncoding


class TransformerIndexInputDecoder(nn.Module):
	num_layers: int
	vocab_size: int
	input_dim: int
	num_heads: int
	ff_dim: int
	dropout_prob: float
	use_bias: bool
	max_len: int  # Maximum possible length of sequence
	activation_fn: Callable

	word_embedding = None
	input_dropout = None
	pos_encoding = None
	decoder_blocks = None

	def setup(self) -> None:
		self.input_dropout = nn.Dropout(self.dropout_prob)
		self.word_embedding = nn.Embed(self.vocab_size, self.input_dim)
		self.pos_encoding = PositionalEncoding(d_model=self.input_dim, max_len=self.max_len + 1)
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
			Note: Input for decoder is word idxs [b, l]
			Note: Input for encoder is word embeddings [b, l, d]
			x: [batch_size, target_seq_len]: Label
			kv: Context
		"""
		x = self.word_embedding(x)
		x = self.input_dropout(x, deterministic=deterministic)
		x = self.pos_encoding(x)
		attention_weights = []
		for block in self.decoder_blocks:
			x, attention_weight = block(q=x, kv=kv, mask=mask, deterministic=deterministic)
			attention_weights.append(attention_weight)
		return x, attention_weights
