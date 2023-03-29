import flax.linen as nn
import jax.numpy as jnp

from comde.comde_modules.seq2seq.transformer.architectures.multihead_attention import MultiheadDotProductAttention


class CausalSelfAttention(nn.Module):
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

	@staticmethod
	def get_causal_mask(
		x: jnp.ndarray,	# [b, l, d]
	) -> jnp.ndarray:
		batch_size = x.shape[0]
		seq_len = x.shape[1]
		tril = 1 - jnp.tril(jnp.ones((seq_len, seq_len))).reshape(-1, seq_len, seq_len)
		tril = jnp.repeat(tril, axis=0, repeats=batch_size)
		return tril	# [b, l, l]

	def forward(self, x: jnp.ndarray, attention_mask: jnp.ndarray, deterministic: bool):
		causal_mask = self.get_causal_mask(x)	# [b, l, l]
		causal_mask = jnp.expand_dims(causal_mask, axis=-3)	# [b, 1, l, l]
		attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)	# [b, 1, l, l]
		mask = jnp.logical_or(causal_mask, attention_mask)	# [b, 1, l, l]
		return self.mha(input_q=x, input_k=x, input_v=x, mask=mask, deterministic=deterministic)
