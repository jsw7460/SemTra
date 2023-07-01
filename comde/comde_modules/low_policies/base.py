from typing import Dict, List

import numpy as np

from comde.comde_modules.base import ComdeBaseModule
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.interfaces import IJaxSavable, ITrainable
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params

EPS = 1E-12


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
		self.nonfunc_dim = cfg["non_functionality_dim"]
		self.param_dim = cfg["param_dim"]
		self.param_repeats = cfg.get("param_repeats", None)
		self.total_param_dim = self.param_dim * self.param_repeats

		if init_build_model:
			self.normalization_mean = cfg["obs_mean"]
			self.normalization_std = cfg["obs_std"]

	def get_parameterized_skills(self, replay_data: ComDeBufferSample) -> Dict[str, np.ndarray]:
		"""
			Return parameterized skills: [Skill @ Non-functionality @ Parameter]
		"""
		if replay_data.parameterized_skills is not None:
			return replay_data.parameterized_skills
		else:
			skills = replay_data.skills
			non_func = np.broadcast_to(replay_data.non_functionality[:, np.newaxis, ...], skills.shape)
			params = replay_data.params_for_skills
			params = np.repeat(params, repeats=self.param_repeats, axis=-1)

			parameterized_skills = np.concatenate((skills, non_func, params), axis=-1)
			return {
				"parameterized_skills": parameterized_skills,
				"repeated_params": params,
				"non_functionality": non_func
			}

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
