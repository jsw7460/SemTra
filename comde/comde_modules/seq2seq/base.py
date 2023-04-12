import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np

from comde.comde_modules.base import ComdeBaseModule
from comde.utils.common.lang_representation import SkillRepresentation
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

		if init_build_model:
			with open(self.cfg["skill_infos_path"], "rb") as f:
				self.tokens = pickle.load(f)	# type: Dict[str, List[SkillRepresentation]]

			with open(self.cfg["startend_token_path"], "rb") as f:
				token_dict = pickle.load(f)

			assert len(token_dict["start"]) == len(token_dict["end"]) == 1
			start_token = token_dict["start"][0]	# type: SkillRepresentation
			end_token = token_dict["end"][0]	# type: SkillRepresentation

			max_skill_index = max([token[0].index for token in self.tokens.values()])
			self.start_token = start_token._replace(index=max_skill_index + 1)
			self.end_token = end_token._replace(index=max_skill_index + 2)

			# Why 'example'?: each skill can have language variations. But we use only one.
			example_vocabulary = [sk[0] for sk in list(self.tokens.values())]
			example_vocabulary.extend([self.start_token, self.end_token])
			example_vocabulary.sort(key=lambda sk: sk.index)
			self._example_vocabulary = example_vocabulary

	def update_tokens(self, new_tokens: Dict):
		pass

	def predict(self, *args, **kwargs) -> np.ndarray:
		raise NotImplementedError()

	@property
	def model(self):
		return None

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
