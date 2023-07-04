from typing import Dict, List

import numpy as np

from comde.comde_modules.base import ComdeBaseModule
from comde.utils.interfaces import ITrainable, IJaxSavable
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class BaseIntentEmbedding(ComdeBaseModule, ITrainable, IJaxSavable):

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool
	):
		super(BaseIntentEmbedding, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)
		raise NotImplementedError("Obsolete")
		self.seed = seed
		self.cfg = cfg
		self.lang_dim = self.cfg["lang_dim"]		# Input

	def predict(self, *args, **kwargs) -> np.ndarray:
		pass

	@property
	def model(self) -> Model:
		pass

	def build_model(self):
		pass

	def update(self, *args, **kwargs) -> Dict:
		pass

	def evaluate(self, *args, **kwargs) -> Dict:
		pass

	def _excluded_save_params(self) -> List:
		pass

	def _get_save_params(self) -> Dict[str, Params]:
		pass

	def _get_load_params(self) -> List[str]:
		pass
