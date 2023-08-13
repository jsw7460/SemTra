from typing import Dict

from flax import linen as nn
from jax import numpy as jnp

from comde.comde_modules.environment_encoder.architectures.lstm import LSTM

EPSILON = 1E-9


class PrimVecQuantizedLSTM(nn.Module):
	lstm_cfg: Dict
	n_codebook: int

	lstm = None  # type: LSTM
	codebook = None  # type: nn.Embed

	def setup(self) -> None:
		self.lstm = LSTM(**self.lstm_cfg)
		self.codebook = nn.Embed(self.n_codebook, self.lstm_cfg["hidden_dim"])

	def __call__(self, *args, **kwargs) -> Dict:
		return self.forward(*args, **kwargs)

	def forward(self, sequence: jnp.ndarray, n_iter: jnp.ndarray, deterministic: bool = False) -> Dict:
		batch_size = sequence.shape[0]
		lstm_stack = self.get_stacked_output(sequence=sequence, deterministic=deterministic)

		unquantized = lstm_stack.at[jnp.arange(batch_size), n_iter].get()  # [b, d]
		ret_info = self.quantization(unquantized)
		ret_info.update({"unquantized": unquantized})

		return ret_info

	def get_current_codebook(self):
		return self.codebook.embedding

	def get_stacked_output(self, sequence: jnp.ndarray, deterministic: bool = False):
		stacked_output, _ = self.lstm(sequence=sequence, deterministic=deterministic)
		return stacked_output

	def quantization(self, unquantized: jnp.ndarray) -> Dict:
		unquantized = jnp.expand_dims(unquantized, axis=1)  # [b, 1, d]
		codebook = jnp.expand_dims(self.codebook.embedding, axis=0)  # [1, n_codebook, d]

		distance = jnp.linalg.norm(unquantized - codebook, axis=-1, keepdims=False)

		# Note: Gradients are vanishes due to argmin function here
		nearest_codebook_idxs = jnp.argmin(distance, axis=-1, keepdims=False)
		quantized = self.codebook(nearest_codebook_idxs)

		return {"nearest_codebook_idxs": nearest_codebook_idxs, "quantized": quantized}
