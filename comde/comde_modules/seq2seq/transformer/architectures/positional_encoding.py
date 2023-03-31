import math

import flax.linen as nn
import jax
import numpy as np
import jax.numpy as jnp


class PositionalEncoding(nn.Module):
	d_model: int  # Hidden dimensionality of the input
	max_len: int  # Maximum length of a sequence to expect

	pe = None

	def setup(self) -> None:
		pe = np.zeros((self.max_len, self.d_model))
		position = np.arange(0, self.max_len, dtype=np.float32)[:, np.newaxis]
		div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
		pe[:, 0::2] = np.sin(position * div_term)
		pe[:, 1::2] = np.cos(position * div_term)
		pe = pe[np.newaxis]
		self.pe = jax.device_put(pe)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(self, x: jnp.ndarray):
		x = x + self.pe[:, :x.shape[1]]
		return x
