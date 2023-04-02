from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.seq2seq.transformer.architectures.self_attention import SelfAttention


class EncoderBlock(nn.Module):
	input_dim: int
	num_heads: int
	ff_dim: int
	dropout_prob: float
	use_bias: bool
	activation_fn: Callable

	self_attention = None
	linear = None
	ln1 = None
	ln2 = None
	dropout = None

	def setup(self) -> None:
		self.self_attention = SelfAttention(
			embed_dim=self.input_dim,
			num_heads=self.num_heads,
			use_bias=True
		)
		self.linear = create_mlp(
			output_dim=self.input_dim,
			net_arch=[],
			activation_fn=self.activation_fn,
			dropout=self.dropout_prob
		)

		self.ln1 = nn.LayerNorm()
		self.ln2 = nn.LayerNorm()

		self.dropout = nn.Dropout(self.dropout_prob)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(self, x: jnp.ndarray, mask: jnp.ndarray, deterministic: bool):
		attn_out, _ = self.self_attention(x=x, mask=mask, deterministic=deterministic)
		x = x + self.dropout(attn_out, deterministic=deterministic)

		x = self.ln1(x)

		linear_out = self.linear(x, deterministic=deterministic)
		x = x + self.dropout(linear_out, deterministic=deterministic)
		x = self.ln2(x)

		return x
