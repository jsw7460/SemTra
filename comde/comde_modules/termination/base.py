from typing import Dict

import numpy as np

from comde.comde_modules.base import ComdeBaseModule


class BaseTermination(ComdeBaseModule):

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool
	):
		super(BaseTermination, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

		self.observation_dim = self.cfg["observation_dim"]
		self.first_observation_dim = self.cfg["first_observation_dim"]
		self.action_dim = cfg["action_dim"]

	def predict(self, *args, **kwargs) -> np.ndarray:
		raise NotImplementedError()
