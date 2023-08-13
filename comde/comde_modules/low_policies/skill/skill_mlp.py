from typing import Dict, List

import jax
import numpy as np
import optax

from comde.comde_modules.low_policies.algos.forwards import skill_mlp_forward as fwd
from comde.comde_modules.low_policies.algos.updates.skill_mlp import skill_mlp_updt
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.low_policies.skill.architectures.skill_mlp import PrimSkillMLP
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.general import str_to_flax_activation
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params

"""
 code is from:
	mm_sbrl/component/decoder/sensoric/mlp_skill_decoder.py
"""


class SkillMLP(BaseLowPolicy):
	PARAM_COMPONENTS = ["_SkillMLP__model"]

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool = True
	):

		super(SkillMLP, self).__init__(
			seed=seed,
			cfg=cfg,
			init_build_model=init_build_model
		)
		self.__model = None

		if init_build_model:
			self.build_model()

	@property
	def model(self) -> Model:
		return self.__model

	@model.setter
	def model(self, value):
		self.__model = value

	def str_to_activation(self, activation_fn: str):
		return str_to_flax_activation(activation_fn)

	def build_model(self):
		mlp = PrimSkillMLP(
			act_scale=self.cfg["act_scale"],
			output_dim=self.action_dim,
			net_arch=self.cfg["net_arch"],
			activation_fn=self.cfg["activation_fn"],
			dropout=self.cfg["dropout"],
			squash_output=True,
			layer_norm=self.cfg["layer_norm"]
		)

		init_obs = np.zeros((1, self.observation_dim))
		# skill_dim = semantic / nonfunc_dim = nonfunctionality
		# total_param_dim = parameter / online_context_dim = online context inference dim
		init_skills = np.zeros((1, self.skill_dim + self.nonfunc_dim + self.total_param_dim + self.online_context_diim))

		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.adam(self.cfg["lr"])
		self.model = Model.create(model_def=mlp, inputs=[rngs, init_obs, init_skills], tx=tx)

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		skills_dict = self.get_parameterized_skills(replay_data)
		skills = skills_dict["parameterized_skills"]

		# a = replay_data.actions[0, ..., 0]
		# print(a)

		if replay_data.online_context is not None:
			observations = replay_data.observations[:, -1, ...]
			actions = replay_data.actions[:, -1, ...]
			maskings = replay_data.maskings[:, -1]
			skills = skills[:, -1, ...]
		else:
			observations = replay_data.observations
			actions = replay_data.actions
			maskings = replay_data.maskings

		new_model, info = skill_mlp_updt(
			rng=self.rng,
			mlp=self.model,
			observations=observations,
			actions=actions,
			skills=skills,
			maskings=maskings
		)

		self.model = new_model
		self.rng, _ = jax.random.split(self.rng)
		self.n_update += 1

		return info

	def predict(
		self,
		observations: np.ndarray,
		skills: np.ndarray,  # [b, l, dim] or [b, dim]
		to_np: bool = True,
		squeeze: bool = False,
		*args, **kwargs  # Do not remove these dummy parameters.
	) -> np.ndarray:
		assert observations.ndim == skills.ndim
		# Transformer inputs are used at evaluation time -> Use only current
		if observations.ndim == 3:
			# They should have the same dimension
			assert (skills.ndim == 3) and (observations.ndim == 3)
			observations = observations[:, -1, ...]
			skills = skills[:, -1, ...]

		self.rng, prediction = fwd(
			rng=self.rng,
			model=self.model,
			observations=observations,
			skills=skills
		)

		if squeeze:
			prediction = np.squeeze(prediction, axis=0)

		if to_np:
			return np.array(prediction)
		else:
			return prediction

	def evaluate(
		self,
		replay_data: ComDeBufferSample
	) -> Dict:
		observations = replay_data.observations[:, -1, ...]
		actions = replay_data.actions[:, -1, ...]

		skills_dict = self.get_parameterized_skills(replay_data)
		skills = skills_dict["parameterized_skills"][:, -1, ...]

		maskings = replay_data.maskings[:, -1]
		maskings = maskings.reshape(-1, 1)
		pred_actions = self.predict(observations=observations, skills=skills)

		pred_actions = pred_actions.reshape(-1, self.action_dim) * maskings
		target_actions = actions.reshape(-1, self.action_dim) * maskings
		mse_error = np.sum(np.mean((pred_actions - target_actions) ** 2, axis=-1)) / np.sum(maskings)
		eval_info = {
			"decoder/mse_error": mse_error,
			"decoder/mse_error_scaled(x100)": mse_error * 100
		}
		return eval_info

	def _excluded_save_params(self) -> List:
		return SkillMLP.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in SkillMLP.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return SkillMLP.PARAM_COMPONENTS
