from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List

import numpy as np

from comde.comde_modules.base import ComdeBaseModule
from comde.utils.interfaces import ITrainable, IJaxSavable
from comde.utils.jax_utils.type_aliases import Params


class BaseSeqToSeq(ComdeBaseModule, ITrainable, IJaxSavable):

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool
	):
		super(BaseSeqToSeq, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)
		self.seed = seed
		self.cfg = cfg
		self.inseq_dim = cfg["inseq_dim"]  # Input sequence's dim
		self.encoder_max_len = self.cfg["encoder_max_len"]
		self.decoder_max_len = self.cfg["decoder_max_len"]

		self.tokens = None
		self.bos_token = None
		self.eos_token = None
		self.vocabulary = None

		self.register_vocabulary()

	def update_tokens(self, new_tokens: Dict):
		pass

	def predict(self, *args, **kwargs) -> np.ndarray:
		raise NotImplementedError()

	@property
	def model(self):
		return None

	@abstractmethod
	def register_vocabulary(self):
		raise NotImplementedError()

	def build_model(self):
		pass

	def update(self, *args, **kwargs) -> Dict:
		pass

	def evaluate(self, *args, **kwargs) -> Dict:
		return defaultdict()

	def _excluded_save_params(self) -> List:
		pass

	def _get_save_params(self) -> Dict[str, Params]:
		pass

	def _get_load_params(self) -> List[str]:
		pass
