import collections.abc
from abc import abstractmethod
from types import MappingProxyType
from typing import Dict

import jax
import numpy as np

from comde.utils.jax_utils.general import str_to_flax_activation


class ComdeBaseModule:
	def __init__(self, seed: int, cfg: Dict, init_build_model: bool):
		self.seed = seed
		self.rng = jax.random.PRNGKey(seed)
		self.cfg = cfg
		if init_build_model:
			self._str_to_activation()

		self.cfg = MappingProxyType(cfg)  # Freeze
		self.n_update = 0

	def str_to_activation(self, activation_fn: str):
		raise NotImplementedError("Obsolete")

	def _str_to_activation(self) -> None:
		def str_to_activation(data: collections.abc.Mapping):
			for key, value in data.items():
				if isinstance(value, collections.abc.Mapping):
					str_to_activation(value)
				else:
					if key == "activation_fn":
						activation = str_to_flax_activation(value)
						data[key] = activation

		str_to_activation(data=self.cfg)

	@abstractmethod
	def predict(self, *args, **kwargs) -> np.ndarray:
		raise NotImplementedError()
