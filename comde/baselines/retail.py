from typing import Dict, List

import jax.random
import numpy as np
import optax

from comde.baselines.algos.updates.retail import (
	retail_policy_update as policy_update,
	retail_transfer_update as transfer_update
)
from comde.baselines.architectures.action_transformer import PrimActionTransformer
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.low_policies.skill.architectures.skill_mlp import PrimGoalMLP
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params
from comde.baselines.utils.get_episode_skills import get_episodic_level_skills


class Retail(BaseLowPolicy):
	PARAM_COMPONENTS = ["_Retail__policy", "_Retail__transfer"]

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		super(Retail, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

		self.language_dim = cfg["language_dim"]
		self.entity_dim = cfg["entity_dim"]
		self.obs_subseq_len = cfg["obs_subseq_len"]
		self.act_subseq_len = cfg["act_subseq_len"]
		self.n_target_skill = cfg["n_target_skill"]
		self.policy_warmup_step = cfg["policy_warmup_step"]

		self.parameterized_skill_dim = self.skill_dim + self.nonfunc_dim + self.param_dim

		self.__policy = None
		self.__transfer = None

		if init_build_model:
			self.build_model()

	@property
	def model(self) -> Model:
		return self.__policy

	def build_model(self):
		self.create_policy()
		self.create_transfer()

	def create_policy(self):
		policy = PrimGoalMLP(**self.cfg["policy_cfg"])
		init_obs = np.zeros((1, self.observation_dim))

		# Goal = n-Concatenation of parameterized skills + non_functionality
		init_goals = np.zeros((1, (self.skill_dim + self.param_dim) * self.n_target_skill + self.nonfunc_dim ))

		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.adam(self.cfg["policy_lr"])
		self.__policy = Model.create(model_def=policy, inputs=[rngs, init_obs, init_goals], tx=tx)

	def create_transfer(self):
		policy_transfer = PrimActionTransformer(**self.cfg["transfer_cfg"])
		init_obs = np.zeros((1, self.obs_subseq_len, self.entity_dim))  # This model inputs entity
		init_target_skill_sequence = np.zeros((1, self.n_target_skill, self.skill_dim + self.param_dim))	# Parameterized skill
		init_lang = np.zeros((1, self.language_dim + self.nonfunc_dim))  # Templatized language
		init_act = np.zeros((1, self.act_subseq_len, self.action_dim))
		init_obs_masking = np.zeros((1, self.obs_subseq_len))
		init_act_masking = np.zeros((1, self.act_subseq_len))

		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.adam(self.cfg["transfer_lr"])
		self.__transfer = Model.create(
			model_def=policy_transfer,
			inputs=[
				rngs,
				init_obs,
				init_act,
				init_target_skill_sequence,
				init_lang,
				init_obs_masking,
				init_act_masking
			],
			tx=tx
		)

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		info = get_episodic_level_skills(
			replay_data=replay_data,
			n_target_skill=self.n_target_skill,
			param_repeats=self.param_repeats
		)
		f_p_source_skills = info["flat_nonfunc_parameterized_source_skills"]
		f_p_target_skills = info["flat_nonfunc_parameterized_target_skills"]
		p_target_skills = info["parameterized_target_skills"]

		subseq_len = replay_data.actions.shape[1]
		f_p_source_skills = np.expand_dims(f_p_source_skills, axis=1)
		f_p_source_skills = np.repeat(f_p_source_skills, repeats=subseq_len, axis=1)
		f_p_target_skills = np.expand_dims(f_p_target_skills, axis=1)
		f_p_target_skills = np.repeat(f_p_target_skills, repeats=subseq_len, axis=1)

		new_policy, policy_info = policy_update(
			rng=self.rng,
			policy=self.__policy,
			observations=replay_data.observations,
			actions=replay_data.actions,
			skills=f_p_target_skills,
			maskings=replay_data.maskings
		)
		self.__policy = new_policy

		language = np.concatenate((replay_data.sequential_requirement, replay_data.non_functionality), axis=-1)
		transfer_info = dict()
		if self.policy_warmup_step < self.n_update:
			new_transfer, transfer_info = transfer_update(
				rng=self.rng,
				policy=self.__policy,
				transfer=self.__transfer,
				source_observations=replay_data.source_observations,
				flat_source_skills=f_p_source_skills,
				flat_target_skills=f_p_target_skills,
				target_skill_sequence=p_target_skills,
				language=language,
				obs_maskings=replay_data.source_maskings,
				act_maskings=replay_data.source_maskings
			)
			self.__transfer = new_transfer

		self.rng, _ = jax.random.split(self.rng)
		self.n_update += 1
		return {**policy_info, **transfer_info}

	def evaluate(self, *args, **kwargs) -> Dict:
		pass

	def predict(self, *args, **kwargs) -> np.ndarray:
		pass

	def _excluded_save_params(self) -> List:
		return Retail.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in Retail.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return Retail.PARAM_COMPONENTS
