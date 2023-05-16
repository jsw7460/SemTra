from typing import Dict, List

import numpy as np

from comde.comde_modules.base import ComdeBaseModule
from comde.utils.interfaces import IJaxSavable, ITrainable
from comde.utils.jax_utils.type_aliases import Params
from comde.utils.jax_utils.model import Model

EPS = 1E-12


class BaseHighPolicy(ComdeBaseModule, IJaxSavable, ITrainable):
	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool
	):
		super(BaseHighPolicy, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)
		"""
			:param cfg: This contains a configuration of a model architecture
		"""

		self.observation_dim = cfg["observation_dim"]
		self.action_dim = cfg["action_dim"]	# Higher action dim

		self.nonfunc_dim = cfg["non_functionality_dim"]
		self.param_dim = cfg["param_dim"]
		self.param_repeats = cfg.get("param_repeats", None)
		# self.param_repeats = cfg["param_repeats"]	# type: int

		# ============================================ DEBUG
		if self.param_repeats is None:	# For the deprecated case.
			raise NotImplementedError("Unwrap this code if you really want")
			# self.param_dim = 10
			# self.param_repeats = 1

		self.total_param_dim = self.param_dim * self.param_repeats

		if init_build_model:
			self.normalization_mean = cfg["obs_mean"]
			self.normalization_std = cfg["obs_std"]

	def predict(self, *args, **kwargs) -> np.ndarray:
		raise NotImplementedError()

	def _excluded_save_params(self) -> List:
		pass

	def _get_save_params(self) -> Dict[str, Params]:
		pass

	def _get_load_params(self) -> List[str]:
		pass

	@property
	def model(self) -> Model:
		raise NotImplementedError()

	def build_model(self):
		pass

	def update(self, *args, **kwargs) -> Dict:
		pass

	def evaluate(self, *args, **kwargs) -> Dict:
		pass

	def str_to_activation(self, activation_fn: str):
		pass