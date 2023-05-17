import flax.linen as nn
import jax.numpy as jnp

from comde.comde_modules.seq2seq.transformer.architectures.multihead_attention import MultiheadDotProductAttention


class CrossAttention(nn.Module):
	embed_dim: int
	num_heads: int
	use_bias: bool

	mha = None

	def setup(self) -> None:
		self.mha = MultiheadDotProductAttention(
			embed_dim=self.embed_dim,
			num_heads=self.num_heads,
			use_bias=self.use_bias
		)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(self, q: jnp.ndarray, kv: jnp.ndarray, mask: jnp.ndarray, deterministic: bool):
		return self.mha(input_q=q, input_k=kv, input_v=kv, mask=mask, deterministic=deterministic)
