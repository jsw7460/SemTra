import einops
import flax.linen as nn
import jax.nn
import jax.numpy as jnp

from comde.comde_modules.common.utils import create_mlp

INFTY = 1e+9

"""
SKIN: Skill-Intent Transformer Encoder

Input: Source skills and Language
Output: Latent vector (will be used as input to Transformer decoder).
"""


def scaled_dot_product(
	q: jnp.ndarray,  # [b, h, len(q), d]	(h: Num head)
	k: jnp.ndarray,  # [b, h, len(k), d]	(h: Num head)
	v: jnp.ndarray,  # [b, h, len(v), d]	(h: Num head)
	mask: jnp.ndarray  # [b, 1, 1, len(k)]	# Should be broadcasted to [b, h, l(q), l(k)]
):
	d_k = q.shape[-1]

	attn_logits = jnp.matmul(q, einops.rearrange(k, "b h l d -> b h d l"))	# [b, h, l(q), l(k)]
	attn_logits = attn_logits / jnp.sqrt(d_k)  # [b, h, l(q), l(k)]
	attn_logits = jnp.where(mask == 0, -INFTY, attn_logits)	# [b, h, l(q), l(k)]
	attention = jax.nn.softmax(attn_logits, axis=-1)  # [b, h, l, l]
	values = jnp.matmul(attention, v)  # [b, h, l, d]

	return values, attention


class MultiheadDotProductAttention(nn.Module):
	embed_dim: int
	num_heads: int
	use_bias: bool

	qkv_proj = None
	o_proj = None

	def setup(self) -> None:
		# Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
		# Note that in many implementations you see "bias=False" which is optional
		self.qkv_proj = create_mlp(
			output_dim=3 * self.embed_dim,
			use_bias=self.use_bias,
			kernel_init=nn.initializers.xavier_uniform(),
			bias_init=nn.initializers.zeros,
		)
		self.o_proj = create_mlp(
			output_dim=self.embed_dim,
			kernel_init=nn.initializers.xavier_uniform(),
			bias_init=nn.initializers.zeros,
		)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		input_q: jnp.ndarray,  # [b, l, d]
		input_k: jnp.ndarray,  # [b, l, d]
		input_v: jnp.ndarray,  # [b, l, d]
		mask: jnp.ndarray,  # [b, l]
		deterministic: bool
	):
		x = jnp.concatenate((input_q, input_k, input_v), axis=-1)
		batch_size, seq_len, embed_dim = x.shape
		qkv = self.qkv_proj(x, deterministic=deterministic)

		# Separate Q, K, V from linear output
		qkv = qkv.reshape(batch_size, seq_len, self.num_heads, -1)  # [b, l, h, d]
		qkv = einops.rearrange(qkv, "b l h d -> b h l d")
		q, k, v = jnp.array_split(qkv, 3, axis=-1)

		mask = jnp.expand_dims(mask, axis=(-3, -2))		# [b, 1, 1, l]
		values, attention = scaled_dot_product(q=q, k=k, v=v, mask=mask)  # [b, h, l, d], [b, h, l, l]
		values = einops.rearrange(values, "b h l d -> b l h d")
		values = values.reshape(batch_size, seq_len, embed_dim)
		o = self.o_proj(values, deterministic=deterministic)

		return o, attention
