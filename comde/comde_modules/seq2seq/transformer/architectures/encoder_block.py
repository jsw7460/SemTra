from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.seq2seq.transformer.architectures.cross_attention import CrossAttention


class EncoderBlock(nn.Module):
	input_dim: int
	num_heads: int
	ff_dim: int
	dropout_prob: float
	use_bias: bool
	activation_fn: Callable

	attention = None
	linear = None
	ln1 = None
	ln2 = None
	dropout = None

	def setup(self) -> None:
		self.attention = CrossAttention(
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

	def forward(
		self,
		q: jnp.ndarray,
		kv: jnp.ndarray,
		q_mask: jnp.ndarray,	# [b, len(q)]
		kv_mask: jnp.ndarray,	# [b, len(kv)]
		deterministic: bool
	):
		mask = jnp.matmul(jnp.expand_dims(q_mask, axis=2), jnp.expand_dims(kv_mask, axis=1))
		mask = jnp.expand_dims(mask, axis=1)

		x, _ = self.attention(q=q, kv=kv, mask=mask, deterministic=deterministic)
		x = q + self.dropout(x, deterministic=deterministic)

		x = self.ln1(x)

		x2 = self.linear(x, deterministic=deterministic)
		x2 = x + self.dropout(x2, deterministic=deterministic)
		x2 = self.ln2(x2)

		return x2
