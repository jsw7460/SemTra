from typing import Dict, List

import numpy as np

from comde.comde_modules.base import ComdeBaseModule
from comde.utils.interfaces import IJaxSavable, ITrainable
from comde.utils.jax_utils.type_aliases import Params


class BaseLowPolicy(ComdeBaseModule, IJaxSavable, ITrainable):

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

	def _excluded_save_params(self) -> List:
		pass

	def _get_save_params(self) -> Dict[str, Params]:
		pass

	def _get_load_params(self) -> List[str]:
		pass

	@property
	def model(self):
		return None

	def build_model(self):
		pass

	def update(self, *args, **kwargs) -> Dict:
		pass

	def evaluate(self, *args, **kwargs) -> Dict:
		pass
