from flax import linen as nn


class Scaler(nn.Module):
	"""
		Scaling the output of base model
	"""
	base_model: nn.Module
	scale: float

	@nn.compact
	def __call__(self, *args, **kwargs):
		original_output = self.base_model(*args, **kwargs)
		return self.scale * original_output
