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
		if init_build_model:
			if isinstance(cfg.get("activation_fn", None), str):
				cfg["activation_fn"] = self.str_to_activation(cfg["activation_fn"])

		self.cfg = MappingProxyType(cfg)  # Freeze

	def str_to_activation(self, activation_fn: str):
		return str_to_flax_activation(activation_fn)

	@abstractmethod
	def predict(self, *args, **kwargs) -> np.ndarray:
		raise NotImplementedError()
