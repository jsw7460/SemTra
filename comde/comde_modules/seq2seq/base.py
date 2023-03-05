from typing import Dict

import numpy as np

from comde.comde_modules.base import ComdeBaseModule


class BaseSeqToSeq(ComdeBaseModule):

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

	def predict(self, *args, **kwargs) -> np.ndarray:
		raise NotImplementedError()
