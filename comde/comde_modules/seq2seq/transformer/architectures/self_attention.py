import flax.linen as nn
import jax.numpy as jnp

from comde.comde_modules.seq2seq.transformer.architectures.multihead_attention import MultiheadDotProductAttention


class SelfAttention(nn.Module):
	embed_dim: int
	num_heads: int
	use_bias: bool

	mha = None

	def setup(self) -> None:
		self.mha = MultiheadDotProductAttention(
			embed_dim=self.embed_dim,
			num_heads=self.num_heads,
			use_bias=self.use_bias,
		)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		x: jnp.ndarray,  # [b, l, d]
		mask: jnp.ndarray,  # [b, l]
		deterministic: bool
	):
		mask = jnp.expand_dims(mask, axis=(-3, -2))
		ret = self.mha(input_q=x, input_k=x, input_v=x, mask=mask, deterministic=deterministic)
		return ret
