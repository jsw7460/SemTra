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

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		q: jnp.ndarray,
		kv: jnp.ndarray,
		mask: jnp.ndarray,
		deterministic: bool
	):
		x = self.causal_attention(x=q, attention_mask=mask, deterministic=deterministic)
		x = self.ln1(x)
		x = self.cross_attention(q=x, kv=kv, deterministic=deterministic)
		x = self.ln2(x)
		x = self.linear(x, deterministic=deterministic)
		x = self.ln3(x)

		return x