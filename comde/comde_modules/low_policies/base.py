from typing import Dict, List

import numpy as np

from comde.comde_modules.base import ComdeBaseModule
from comde.utils.interfaces import IJaxSavable, ITrainable
from comde.utils.jax_utils.type_aliases import Params
from comde.utils.jax_utils.model import Model
from comde.rl.buffers.type_aliases import ComDeBufferSample


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

		self.skill_dim = cfg["skill_dim"]
		self.intent_dim = cfg["intent_dim"]

	@staticmethod
	def get_intent_conditioned_skill(replay_data: ComDeBufferSample) -> np.ndarray:
		"""
			Concatenate intent and skill
		"""
		skills = replay_data.skills
		intents = replay_data.intents
		if intents is None:
			intents = np.expand_dims(replay_data.language_operators, axis=1)
			intents = np.repeat(intents, repeats=replay_data.skills.shape[1], axis=1)

		skills = np.concatenate((skills, intents), axis=-1)

		return skills

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
