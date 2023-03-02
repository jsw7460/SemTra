from abc import abstractmethod
from types import MappingProxyType
from typing import Dict

import jax
import numpy as np

from comde.comde_modules.common.superclasses.loggable import Loggable


class BaseLowPolicy(Loggable):

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool
	):
		super(BaseLowPolicy, self).__init__()
		"""
			:param cfg: This contains a configuration of a model architecture
		"""
		if init_build_model:
			if isinstance(cfg.get("activation_fn", None), str):
				cfg["activation_fn"] = self._str_to_activation(cfg["activation_fn"])

			self.cfg = MappingProxyType(cfg)  # Freeze

		self.seed = seed
		self.rng = jax.random.PRNGKey(seed=self.seed)

		self.observation_dim = cfg["observation_dim"]
		self.action_dim = cfg["action_dim"]

	def _str_to_activation(self, activation_fn: str):
		raise NotImplementedError()

	@abstractmethod
	def predict(self, *args, **kwargs) -> np.ndarray:
		raise NotImplementedError()
