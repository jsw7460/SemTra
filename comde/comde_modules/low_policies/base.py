from typing import Dict

import numpy as np

from comde.comde_modules.base import ComdeBaseModule


class BaseLowPolicy(ComdeBaseModule):

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool
	):
		super(BaseLowPolicy, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)
		"""
			:param cfg: This contains a configuration of a model architecture
		"""

		self.observation_dim = cfg["observation_dim"]
		self.action_dim = cfg["action_dim"]

	def predict(self, *args, **kwargs) -> np.ndarray:
		raise NotImplementedError()
