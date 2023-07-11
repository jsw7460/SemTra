from typing import Dict, List

import jax
import numpy as np
import optax

from .architectures.sensor_encoder_transformer import PrimSensorEncoder
from comde.comde_modules.sensor_encoder.algos.updates.sensor_encoder_transformer import sensor_encoder_transformer_updt
from comde.comde_modules.low_policies.algos.updates.skill_mlp import skill_mlp_updt
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.general import str_to_flax_activation
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class SensorEncoder(BaseLowPolicy):
	PARAM_COMPONENTS = ["_SensorEncoder__model"]

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool = True
	):
		super(SensorEncoder, self).__init__(
			seed=seed,
			cfg=cfg,
			init_build_model=init_build_model
		)
		self.__model = None
		self.predict_component = cfg["predict_component"]
		self.coef_skill_loss = cfg["coef_skill_loss"]
		self.coef_param_loss = cfg["coef_param_loss"]

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
		model = PrimSensorEncoder(**self.cfg["predictor_cfg"])
		init_obs = np.zeros((1, 7, self.observation_dim))
		init_act = np.zeros((1, 7, self.action_dim))
		init_mask = np.zeros((1, 7))
		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.adam(self.cfg["lr"])
		self.model = Model.create(model_def=model, inputs=[rngs, init_obs, init_act, init_mask], tx=tx)

	def update(self, replay_data: ComDeBufferSample) -> Dict:

		initial_few_observations = replay_data.initial_few_observations
		initial_few_actions = replay_data.initial_few_actions
		batch_size = initial_few_observations.shape[0]
		subseq_len = initial_few_observations.shape[1]
		maskings = np.ones((batch_size, subseq_len))

		new_model, info = sensor_encoder_transformer_updt(
			rng=self.rng,
			model=self.model,
			observations=initial_few_observations,
			actions=initial_few_actions,
			maskings=maskings,
			skills_idxs=replay_data.initial_few_skills_idxs[:, 0],
			params_idxs=replay_data.initial_few_parameters_idxs[:, 0],
			coef_skill_loss=self.coef_skill_loss,
			coef_param_loss=self.coef_param_loss,
		)

		# print("skill preds", info["__skill_preds"])
		# print("pred skill idxs", info["__skill_pred_idxs"][: 10])
		# print("target skill idxs", info["__skills_idxs"][: 10])

		# print("param preds", info["__param_preds"])
		# print("target param idxs", info["__params_idxs"][: 10])
		# print("pred param idxs", info["__param_pred_idxs"][: 10])

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
		return {}

	def _excluded_save_params(self) -> List:
		return SensorEncoder.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in SensorEncoder.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return SensorEncoder.PARAM_COMPONENTS
