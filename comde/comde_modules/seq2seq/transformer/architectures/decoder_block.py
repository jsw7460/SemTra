from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.seq2seq.transformer.architectures.causal_attention import CausalSelfAttention
from comde.comde_modules.seq2seq.transformer.architectures.cross_attention import CrossAttention


class DecoderBlock(nn.Module):
	input_dim: int
	num_heads: int
	ff_dim: int
	dropout_prob: float
	use_bias: bool
	activation_fn: Callable

	causal_attention = None
	cross_attention = None
	linear = None
	ln1 = None
	ln2 = None
	ln3 = None
	dropout = None

	def setup(self) -> None:
		self.causal_attention = CausalSelfAttention(
			embed_dim=self.input_dim,
			num_heads=self.num_heads,
			use_bias=self.use_bias
		)
		self.cross_attention = CrossAttention(
			embed_dim=self.input_dim,
			num_heads=self.num_heads,
			use_bias=self.use_bias
		)
		self.linear = create_mlp(
			output_dim=self.input_dim,
			net_arch=[],
			activation_fn=self.activation_fn,
			dropout=self.dropout_prob
		)

		self.ln1 = nn.LayerNorm()
		self.ln2 = nn.LayerNorm()
		self.ln3 = nn.LayerNorm()

		self.dropout = nn.Dropout(self.dropout_prob)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		q: jnp.ndarray,
		kv: jnp.ndarray,
		mask: jnp.ndarray,
		deterministic: bool
	):
		x, _ = self.causal_attention(x=q, deterministic=deterministic)
		x = q + self.dropout(x, deterministic=deterministic)
		x = self.ln1(x)

		mask = jnp.expand_dims(mask, axis=(-3, -2))
		x2, _ = self.cross_attention(q=x, kv=kv, mask=mask, deterministic=deterministic)
		x2 = x + self.dropout(x2, deterministic=deterministic)
		x2 = self.ln2(x2)

		x3 = self.linear(x2, deterministic=deterministic)
		x3 = x2 + self.dropout(x3, deterministic=deterministic)
		x3 = self.ln3(x3)

		return x3
