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
	embed_dim: int	# = d_model
	num_heads: int
	use_bias: bool

	depth = None
	q_proj = None
	k_proj = None
	v_proj = None
	o_proj = None

	def setup(self) -> None:
		self.depth = self.embed_dim // self.num_heads

		# Note that in many implementations you see "bias=False" which is optional
		self.q_proj = create_mlp(
			output_dim=self.embed_dim,
			net_arch=[],
			use_bias=self.use_bias,
			kernel_init=nn.initializers.xavier_uniform(),
			bias_init=nn.initializers.zeros,
		)
		self.k_proj = create_mlp(
			output_dim=self.embed_dim,
			net_arch=[],
			use_bias=self.use_bias,
			kernel_init=nn.initializers.xavier_uniform(),
			bias_init=nn.initializers.zeros,
		)
		self.v_proj = create_mlp(
			output_dim=self.embed_dim,
			net_arch=[],
			use_bias=self.use_bias,
			kernel_init=nn.initializers.xavier_uniform(),
			bias_init=nn.initializers.zeros,
		)
		self.o_proj = create_mlp(
			output_dim=self.embed_dim,
			net_arch=[],
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
		# x = jnp.concatenate((input_q, input_k, input_v), axis=-1)
		# batch_size, seq_len, embed_dim = x.shape
		# qkv = self.qkv_proj(x, deterministic=deterministic)	# [b, l, 180]
		#
		# # Separate Q, K, V from linear output
		# qkv = qkv.reshape(batch_size, seq_len, self.num_heads, -1)  # [b, l, h, d]	[b, l, 4, 45]
		# qkv = einops.rearrange(qkv, "b l h d -> b h l d")	# [b h 7 45]
		# q, k, v = jnp.array_split(qkv, 3, axis=-1)	# [b h 7 15]

		batch_size = input_q.shape[0]
		l_q = input_q.shape[1]
		l_k = input_k.shape[1]
		l_v = input_v.shape[1]
		embed_dim = input_v.shape[-1]

		q = self.q_proj(input_q)
		k = self.k_proj(input_k)
		v = self.v_proj(input_v)

		q = q.reshape(batch_size, l_q, self.num_heads, self.depth)
		k = k.reshape(batch_size, l_k, self.num_heads, self.depth)
		v = v.reshape(batch_size, l_v, self.num_heads, self.depth)

		q = einops.rearrange(q, "b l h d -> b h l d")	# [b 4 7 15]
		k = einops.rearrange(k, "b l h d -> b h l d")	# [b 4 7 15]
		v = einops.rearrange(v, "b l h d -> b h l d")	# [b 4 7 15]

		values, attention = scaled_dot_product(q=q, k=k, v=v, mask=mask)  # [b, h, l, d], [b, h, l, l]	[b 4 7 15], [b 4 7 7]
		values = einops.rearrange(values, "b h l d -> b l h d")	# [b 7 4 15]
		values = values.reshape(batch_size, -1, embed_dim)
		o = self.o_proj(values, deterministic=deterministic)

		return o, attention
