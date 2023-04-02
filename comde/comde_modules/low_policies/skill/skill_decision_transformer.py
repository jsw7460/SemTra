from typing import Dict, List, Union, Tuple

import jax
import numpy as np
import optax

from comde.comde_modules.low_policies.algos.forwards import skill_decisiontransformer_forward as fwd
from comde.comde_modules.low_policies.algos.updates.skill_decision_transformer import skill_dt_updt
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params
from .architectures.skill_decision_transformer import PrimSkillDecisionTransformer


class SkillDecisionTransformer(BaseLowPolicy):
	PARAM_COMPONENTS = ["_SkillDecisionTransformer__model"]

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool = True
	):
		super(SkillDecisionTransformer, self).__init__(
			seed=seed,
			cfg=cfg,
			init_build_model=init_build_model
		)
		self.__model = None
		self.max_ep_len = cfg["max_ep_len"]

		if init_build_model:
			self.build_model()

	@property
	def model(self) -> Model:
		return self.__model

	@model.setter
	def model(self, value):
		self.__model = value

	@staticmethod
	def check_shape(observations: np.ndarray, actions: np.ndarray, skills: np.ndarray, timesteps: np.ndarray):
		if len(actions) > 0:
			assert observations.ndim == actions.ndim == skills.ndim == 3, \
				[
					f"{comp} should be 3-dimensional but has {comp.ndim}"
					for comp in [observations, actions, skills] if comp.ndim != 3
				]
			assert timesteps.ndim == 2, f"timestep should be 2-dimensional but has {timesteps.ndim}"

	def str_to_activation(self, activation_fn: str):
		return activation_fn

	def build_model(self):

		transformer = PrimSkillDecisionTransformer(
			gpt2_config=self.cfg["gpt2_config"],
			obs_dim=self.observation_dim,
			act_dim=self.action_dim,
			skill_dim=self.skill_dim,
			hidden_size=self.cfg["hidden_size"],
			act_scale=self.cfg["act_scale"],
			max_ep_len=self.max_ep_len,
		)
		self.rng, rngs = get_basic_rngs(self.rng)
		self.rng, transformer_dropout = jax.random.split(self.rng)
		rngs.update({"transformer_dropout": transformer_dropout})
		subseq_len = self.cfg["subseq_len"]
		init_observations = np.random.normal(size=(1, subseq_len, self.observation_dim))
		init_actions = np.random.normal(size=(1, subseq_len, self.action_dim))
		init_skills = np.random.normal(size=(1, subseq_len, self.skill_dim + self.intent_dim))
		init_timesteps = np.zeros((1, self.cfg["subseq_len"]), dtype="i4")
		init_masks = np.ones((1, self.cfg["subseq_len"]))
		tx = optax.chain(
			optax.clip(1.0),
			optax.adamw(learning_rate=self.cfg["lr"])
		)
		self.model = Model.create(
			model_def=transformer,
			inputs=[
				rngs,
				init_observations,
				init_actions,
				init_skills,
				init_timesteps,
				init_masks,
			],
			tx=tx
		)

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		skills = BaseLowPolicy.get_intent_conditioned_skill(replay_data)
		new_model, info = skill_dt_updt(
			rng=self.rng,
			dt=self.model,
			observations=replay_data.observations,
			actions=replay_data.actions,
			skills=skills,
			timesteps=replay_data.timesteps.astype("i4"),
			maskings=replay_data.maskings,
			action_targets=np.copy(replay_data.actions),
		)

		self.model = new_model
		self.rng, _ = jax.random.split(self.rng)

		return info

	def get_padded_components(
		self,
		observations: np.ndarray,
		actions: np.ndarray,
		skills: np.ndarray,
		timesteps: np.ndarray,
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

		self.check_shape(observations, actions, skills, timesteps)
		batch_size = observations.shape[0]
		subseq_len = self.cfg["subseq_len"]

		history_len = observations.shape[1]
		observations = np.concatenate(
			(observations, np.zeros((batch_size, subseq_len - history_len, self.observation_dim))), axis=1
		)
		actions = np.concatenate(
			(actions, np.zeros((batch_size, subseq_len - actions.shape[1], self.action_dim))), axis=1
		)
		skills = np.concatenate(
			(skills, np.zeros((batch_size, subseq_len - skills.shape[1], self.skill_dim))), axis=1
		)
		timesteps = np.concatenate(
			(timesteps, np.zeros((batch_size, subseq_len - timesteps.shape[1]))), axis=1
		)
		maskings = np.concatenate(
			(np.ones((batch_size, history_len)), np.zeros((batch_size, subseq_len - history_len))), axis=1
		)
		return observations, actions, skills, timesteps, maskings

	def predict(self, *args, **kwargs) -> np.ndarray:
		historical_actions = self.predict_and_get_historical_actions(*args, **kwargs)
		return historical_actions[:, -1]

	def predict_and_get_historical_actions(
		self,
		observations: np.ndarray,
		actions: np.ndarray,
		skills: np.ndarray,
		timesteps: np.ndarray,
		maskings: Union[np.ndarray] = None,
		to_np: bool = True
	) -> np.ndarray:

		self.check_shape(observations, actions, skills, timesteps)

		subseq_len = self.cfg["subseq_len"]
		cur_subseq_len = observations.shape[1]

		# Longer than subseq_len -> Truncate
		# Shorter than subseq_len -> Set zero paddings and get maskings.
		if cur_subseq_len < subseq_len:
			observations, actions, skills, timesteps, maskings = self.get_padded_components(
				observations=observations,
				actions=actions,
				skills=skills,
				timesteps=timesteps
			)

		subseq_len = self.cfg["subseq_len"]
		observations = observations[:, :subseq_len, :]
		actions = actions[:, :subseq_len, :]
		skills = skills[:, :subseq_len, :]
		timesteps = timesteps[:, :subseq_len]
		if maskings is None:
			maskings = np.ones((1, subseq_len))

		self.rng, action_preds = fwd(
			rng=self.rng,
			model=self.model,
			observations=observations,
			actions=actions,
			skills=skills,
			timesteps=timesteps.astype("i4"),
			maskings=maskings,
		)

		if to_np:
			return np.array(action_preds)
		else:
			return action_preds

	def evaluate(
		self,
		replay_data: ComDeBufferSample
	) -> Dict:
		observations = replay_data.observations
		actions = replay_data.actions
		skills = BaseLowPolicy.get_intent_conditioned_skill(replay_data)
		timesteps = replay_data.timesteps
		maskings = replay_data.maskings

		action_preds = self.predict_and_get_historical_actions(
			observations=observations,
			actions=actions,
			skills=skills,
			timesteps=timesteps,
			maskings=maskings,
		)
		action_preds = action_preds.reshape(-1, self.action_dim) * maskings.reshape(-1, 1)
		action_targets = actions.reshape(-1, self.action_dim) * maskings.reshape(-1, 1)

		mse_error = np.sum((action_preds - action_targets) ** 2) / np.sum(maskings)

		eval_info = {
			"skill_decoder/mse_error": mse_error,
			"skill_decoder/mse_error_scaled(x100)": mse_error * 100
		}

		return eval_info

	def _excluded_save_params(self) -> List:
		return SkillDecisionTransformer.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in SkillDecisionTransformer.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return SkillDecisionTransformer.PARAM_COMPONENTS
