from typing import List

import flax.linen as nn
import jax.numpy as jnp

from comde.comde_modules.seq2seq.transformer.architectures.encoder_block import EncoderBlock


class TransformerEncoder(nn.Module):
	num_layers: int
	input_dim: int
	num_heads: int
	ff_dim: int
	dropout_prob: float
	use_bias: bool

	encoder_blocks = None

	def setup(self) -> None:
		self.encoder_blocks = [
			EncoderBlock(
				input_dim=self.input_dim,
				num_heads=self.num_heads,
				ff_dim=self.ff_dim,
				dropout_prob=self.dropout_prob,
				use_bias=self.use_bias
			) for _ in range(self.num_layers)
		]  # type: List[EncoderBlock]

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(self, x: jnp.ndarray, mask: jnp.ndarray, deterministic: bool):
		for block in self.encoder_blocks:  # Todo: Change into lax_fori
			x = block(x=x, mask=mask, deterministic=deterministic)

	def get_attention_maps(self, x: jnp.ndarray, mask: jnp.ndarray, deterministic: bool):
		# A function to return the attention maps within the model for a single application
		# Used for visualization purpose later
		attention_maps = []

		for block in self.encoder_blocks:
			_, attn_map = block.self_attention(x=x, mask=mask)
			attention_maps.append(attn_map)
			x = block(x=x, mask=mask, deterministic=deterministic)

		return attention_maps
