from typing import Dict, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax

from comde.utils.interfaces import IJaxSavable, ITrainable
from comde.comde_modules.low_policies.algos.forwards import skill_decisiontransformer_forward as fwd
from comde.comde_modules.low_policies.algos.updates.skill_decision_transformer import skill_dt_updt
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params
from .architectures.skill_decision_transformer import PrimSkillDecisionTransformer


class SkillDecisionTransformer(BaseLowPolicy, IJaxSavable, ITrainable):

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
		self.skill_dim = cfg["skill_dim"]
		self.max_ep_len = cfg["max_ep_len"]

		if init_build_model:
			self.build_model()

	@property
	def model(self) -> Model:
		return self.__model

	@model.setter
	def model(self, value):
		self.__model = value

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
		init_observations = np.zeros((1, subseq_len, self.observation_dim))
		init_actions = np.zeros((1, subseq_len, self.action_dim))
		init_skills = np.zeros((1, subseq_len, self.skill_dim))
		init_timesteps = np.zeros((1, self.cfg["subseq_len"]), dtype="i4")
		init_masks = np.zeros((1, self.cfg["subseq_len"]))
		init_rtgs = np.zeros((1, subseq_len, 1))

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
				init_rtgs,
			],
			tx=tx
		)

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		new_model, info = skill_dt_updt(
			rng=self.rng,
			dt=self.model,
			observations=replay_data.observations,
			actions=replay_data.actions,
			skills=replay_data.skills,
			rtgs=replay_data.rtgs[..., np.newaxis],
			timesteps=replay_data.timesteps.astype("i4"),
			maskings=replay_data.maskings,
			action_targets=jnp.copy(replay_data.actions),
		)

		self.model = new_model
		self.rng, _ = jax.random.split(self.rng)

		return info

	def predict(
		self,
		observations: jnp.ndarray,
		actions: jnp.ndarray,
		skills: jnp.ndarray,
		timesteps: jnp.ndarray,
		maskings: Union[jnp.ndarray],
		rtgs: jnp.ndarray,
	) -> jnp.ndarray:
		self.rng, prediction = fwd(
			rng=self.rng,
			model=self.model,
			observations=observations,
			actions=actions,
			skills=skills,
			timesteps=timesteps,
			maskings=maskings,
			rtgs=rtgs,
		)
		obs_preds, action_preds, return_preds = prediction
		return action_preds[:, -1]

	def evaluate(
		self,
		replay_data: ComDeBufferSample
	) -> Dict:
		observations = replay_data.observations
		actions = replay_data.actions
		skills = replay_data.skills
		timesteps = replay_data.timesteps
		maskings = replay_data.maskings
		rtgs = replay_data.rtgs

		action_preds = self.predict(
			observations=observations,
			actions=actions,
			skills=skills,
			timesteps=timesteps,
			maskings=maskings,
			rtgs=rtgs
		)
		action_preds = action_preds.reshape(-1, self.action_dim) * maskings.reshape(-1, 1)
		action_targets = actions.reshape(-1, self.action_dim) * maskings.reshape(-1, 1)

		mse_error = jnp.sum((action_preds - action_targets)) / jnp.sum(maskings)

		eval_info = {
			"decoder/mse_error": mse_error,
			"decoder/mse_error_scaled(x100)": mse_error * 100
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
