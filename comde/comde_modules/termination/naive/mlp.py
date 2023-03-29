from typing import Dict, List
from comde.utils.common.timeit import timeit

import jax
import jax.numpy as jnp
import numpy as np
import optax

from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.termination.algos.forward import termination_forward
from comde.comde_modules.termination.algos.updates.mlp import mlp_termination_updt
from comde.comde_modules.termination.base import BaseTermination
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class MLPTermination(BaseTermination):
	PARAM_COMPONENTS = ["_MLPTermination__model"]

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		super(MLPTermination, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)
		self.__model = None
		self.skill_dim = cfg["skill_dim"]

		if init_build_model:
			self.build_model()

	@property
	def model(self):
		return self.__model

	@model.setter
	def model(self, model):
		self.__model = model

	def build_model(self):
		mlp = create_mlp(
			# output_dim=2: Score vector for [not_done, done].
			output_dim=2,
			net_arch=self.cfg["net_arch"],
			activation_fn=self.cfg["activation_fn"],
			dropout=self.cfg["dropout"],
			squash_output=False,
			layer_norm=self.cfg["layer_norm"],
			batch_norm=self.cfg["batch_norm"],
			use_bias=self.cfg["use_bias"]
		)
		init_obs = np.zeros((1, self.observation_dim + self.first_observation_dim))
		init_skills = np.zeros((1, self.skill_dim))
		init_input = np.concatenate((init_obs, init_skills), axis=-1)

		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.chain(
			optax.clip_by_global_norm(10.0),
			optax.adam(learning_rate=self.cfg["lr"])
		)
		self.model = Model.create(model_def=mlp, inputs=[rngs, init_input], tx=tx)

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		new_model, info = mlp_termination_updt(
			rng=self.rng,
			mlp=self.model,
			observations=replay_data.observations,
			first_observations=replay_data.first_observations,
			skills=replay_data.skills,
			skills_done=replay_data.skills_done
		)
		self.model = new_model
		self.rng, _ = jax.random.split(self.rng)
		return info

	def predict(
		self,
		observations: np.ndarray,
		first_observations: np.ndarray,
		skills: np.ndarray,
		binary: bool = False
	) -> np.ndarray:
		"""
		:param observations:
		:param first_observations:
		:param skills:
		:param binary: If true -> return only 0 or 1. If false -> return array of predicted probability
		:return:
		"""
		self.rng, prediction = termination_forward(self.rng, self.model, observations, first_observations, skills)
		if binary:
			return np.argmax(prediction, axis=-1)
		else:
			return prediction

	def evaluate(
		self,
		replay_data: ComDeBufferSample
	) -> Dict:
		observations = replay_data.observations
		first_observations = replay_data.first_observations
		skills = replay_data.skills
		done_label = replay_data.skills_done

		prediction_score = self.predict(observations, first_observations, skills)
		prediction_class = jnp.argmax(prediction_score, axis=-1)

		accuracy = jnp.mean(prediction_class == done_label)

		true_positive = np.sum((done_label == 1) & (prediction_class == 1))
		# true_negative = np.sum((done_label == 0) & (prediction_class == 0))	# Not used
		false_positive = np.sum((done_label == 0) & (prediction_class == 1))
		false_negative = np.sum((done_label == 1) & (prediction_class == 0))

		recall = true_positive / max([(true_positive + false_negative), 1])
		precision = true_positive / max([(true_positive + false_positive), 1])

		if np.sum((done_label == 1)) == 0 and false_positive == 0:  # No positive label and prediction is all negative
			accuracy = 1.0
			recall = 1.0
			precision = 1.0

		eval_info = {
			"termination_policy/accuracy": accuracy,
			"termination_policy/recall": recall,
			"termination_policy/precision": precision
		}
		return eval_info

	def _excluded_save_params(self) -> List:
		return MLPTermination.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in MLPTermination.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params

		return params_dict

	def _get_load_params(self) -> List[str]:
		return MLPTermination.PARAM_COMPONENTS
