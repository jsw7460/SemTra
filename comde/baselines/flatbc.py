import warnings
from typing import Dict, List

import jax.random
import numpy as np
import optax

from comde.baselines.algos.forwards import flatbc_forward
from comde.baselines.algos.updates.flatbc import flatbc_update
from comde.baselines.utils.get_episode_skills import get_episodic_level_skills
from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class FlatBC(BaseLowPolicy):
	PARAM_COMPONENTS = ["_FlatBC__policy"]
	"""
		Input: 
			1. state
			2. target skills	# Not current skill. It is a set, e.g., [1, 3, 4, 6]. Not [1], ...
			3. target skill's order듈 (?)
			
		i.e., Assume the target skills are optimally obtained. 
		Comparison for ComDe's skill decoder. 
		(- Temporally biasing policy is better)
		We need to input a target skill's order, otherwise agent doesn't know 
		how execute the skills in what order. 
	"""

	def __str__(self):
		return "FlatBC"

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		super(FlatBC, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

		self.language_dim = cfg["language_dim"]
		self.n_source_video = cfg["n_source_video"]  # Should be fixed. This is a sort of advantage.
		assert self.n_source_video == 1, "For convenience, we fix the source video consist of one"
		self.n_target_skill = cfg["n_target_skill"]

		self.__policy = None
		if init_build_model:
			self.build_model()

	@property
	def model(self) -> Model:
		return self.__policy

	def build_model(self):
		policy = create_mlp(**self.cfg["policy_cfg"])
		policy = Scaler(base_model=policy, scale=self.cfg["act_scale"])
		init_obs = np.zeros((1, self.observation_dim))
		init_target_skills = np.zeros((1, (self.skill_dim + self.total_param_dim) * self.n_target_skill))
		init_skills_orders = np.zeros((1, 1))
		policy_input = np.concatenate((init_obs, init_target_skills, init_skills_orders), axis=-1)

		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.adam(self.cfg["lr"])
		self.__policy = Model.create(model_def=policy, inputs=[rngs, policy_input], tx=tx)

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		assert np.all(replay_data.n_target_skills == self.n_target_skill), \
			"FlatBC supports for fixed length of target skills."

		episode_skills = get_episodic_level_skills(
			replay_data=replay_data,
			n_target_skill=self.n_target_skill,
			param_repeats=self.param_repeats
		)
		target_skills = episode_skills["parameterized_target_skills"]

		new_policy, info = flatbc_update(
			rng=self.rng,
			policy=self.model,
			observations=replay_data.observations,
			target_skills=target_skills,
			target_skills_order=replay_data.skills_order,
			actions=replay_data.actions,
			maskings=replay_data.maskings
		)
		self.__policy = new_policy
		self.rng, _ = jax.random.split(self.rng)
		return info

	def evaluate(self, *args, **kwargs) -> Dict:
		"""..."""

	def predict(
		self,
		observations: np.ndarray,  # [b, d]
		target_parameterized_skills: np.ndarray,  # [b, M, d]
		cur_skill_pos: np.ndarray,  # [b, 1]
		to_np: bool = True,
		squeeze: bool = False,
		*args, **kwargs  # Do not remove these dummy parameters.
	) -> np.ndarray:
		input_skills_num = target_parameterized_skills.shape[1]
		if input_skills_num != self.n_target_skill:
			warnings.warn(
				f"Inconsistant number of target skills. Model should receive {self.n_target_skill} skills "
				f"but got {input_skills_num} skills. So we slice them."
			)

		target_parameterized_skills = target_parameterized_skills[:, :self.n_target_skill, ...]

		batch_size = observations.shape[0]
		target_parameterized_skills = target_parameterized_skills.reshape(batch_size, -1)
		policy_input = np.concatenate((observations, target_parameterized_skills, cur_skill_pos), axis=-1)
		self.rng, actions = flatbc_forward(
			rng=self.rng,
			model=self.model,
			policy_input=policy_input
		)
		if to_np:
			return np.array(actions)
		else:
			return actions

	def _excluded_save_params(self) -> List:
		return FlatBC.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in FlatBC.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return FlatBC.PARAM_COMPONENTS
