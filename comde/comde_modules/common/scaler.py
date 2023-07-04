from flax import linen as nn
from jax import numpy as jnp


class Scaler(nn.Module):
	"""
		Scaling the output of base model
	"""
	base_model: nn.Module
	scale: jnp.ndarray

	@nn.compact
	def __call__(self, *args, **kwargs):
		original_output = self.base_model(*args, **kwargs)
		ret = self.scale * original_output
		return ret
