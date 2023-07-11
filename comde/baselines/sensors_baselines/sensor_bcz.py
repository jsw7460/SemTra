import pickle
from typing import Dict, List

import jax
import jax.random
import numpy as np
import optax

from comde.baselines.algos.forwards import (
	demogen_gravity_forward as gravity_forward,
	demogen_policy_forward as policy_forward
)
from comde.baselines.algos.updates.bcz import (
	bcz_policy_update as policy_update,
	bcz_gravity_update as gravity_update
)
from comde.baselines.bcz import BCZ
from comde.baselines.utils.get_episode_emb import get_sensor_text_embeddings
from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.common.utils import create_mlp
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class SensorBCZ(BCZ):
	"""
	Same with BCZ, but instead of video, we use sensor data
	"""

	PARAM_COMPONENTS = [
		"_SensorBCZ__policy",
		"_SensorBCZ__gravity"
	]

	def __str__(self):
		return "SensorBCZ"

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		self.sensors_inst = None
		self.sensor_subseq_len = cfg["sensor_subseq_len"]
		self.n_target_skill = cfg["n_target_skill"]
		super(SensorBCZ, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

	def _set_dictionaries(self):
		with open(self.cfg["episodic_instruction_path"], "rb") as f:
			self.episodic_inst = pickle.load(f)

		with open(self.cfg["sensor_instruction_path"], "rb") as f:
			self.sensors_inst = pickle.load(f)

	def _build_gravity(self):
		gravity_cfg = self.cfg["gravity_cfg"].copy()
		lr = gravity_cfg.pop("lr")
		gravity = create_mlp(**gravity_cfg)
		sensor_seq_dim = (self.observation_dim + self.action_dim) * self.sensor_subseq_len * self.n_target_skill
		init_sensor = np.zeros((1, sensor_seq_dim))
		init_seq = np.zeros((1, self.language_dim))
		init_gravity_input = np.concatenate((init_sensor, init_seq), axis=-1)
		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.adam(learning_rate=lr)
		self.__gravity = Model.create(model_def=gravity, inputs=[rngs, init_gravity_input], tx=tx)

	def _build_policy(self):
		policy_cfg = self.cfg["policy_cfg"].copy()
		lr = policy_cfg.pop("lr")
		policy = create_mlp(**policy_cfg)
		policy = Scaler(base_model=policy, scale=np.array(self.cfg["act_scale"]))
		init_obs = np.zeros((1, self.observation_dim))
		init_inst = np.zeros((1, self.language_dim))
		init_params = np.zeros((1, self.nonfunc_dim + self.total_param_dim))
		policy_input = np.concatenate((init_obs, init_inst, init_params), axis=-1)
		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.adam(learning_rate=lr)
		self.__policy = Model.create(model_def=policy, inputs=[rngs, policy_input], tx=tx)

	def get_source_target_info(self, replay_data: ComDeBufferSample):
		info = get_sensor_text_embeddings(
			sensor_dict=self.sensors_inst,
			text_dict=self.episodic_inst,
			replay_data=replay_data
		)
		so = info["source_observations"]
		sa = info["source_actions"]
		batch_size = so.shape[0]

		source = np.concatenate((so, sa), axis=-1).reshape(batch_size, -1)
		available_idxs = [_ for _ in range(batch_size)]

		ret_info = {
			"source": source,
			"target": info["target_text_embs"],
			"available_idxs": available_idxs
		}
		return ret_info

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		source_target_info = self.get_source_target_info(replay_data)
		source_video_embeddings = source_target_info["source"]
		target_text_embs = source_target_info["target"]

		available_idxs = np.array(source_target_info["available_idxs"])

		new_gravity, gravity_info = gravity_update(
			rng=self.rng,
			gravity=self.__gravity,
			source_video_embeddings=source_video_embeddings[available_idxs],
			sequential_requirement=replay_data.sequential_requirement[available_idxs],
			episodic_instructions=target_text_embs[available_idxs]
		)
		self.rng, _ = jax.random.split(self.rng)

		# Expand the parameter
		params_for_skills = np.repeat(replay_data.params_for_skills, axis=-1, repeats=self.param_repeats)

		episodic_inst = gravity_info.pop("__pred_episode_semantic")

		new_policy, policy_info = policy_update(
			rng=self.rng,
			policy=self.__policy,
			observations=replay_data.observations[available_idxs],
			episodic_inst=episodic_inst,
			non_functionality=replay_data.non_functionality[available_idxs],
			parameters=params_for_skills[available_idxs],
			actions=replay_data.actions[available_idxs],
			maskings=replay_data.maskings[available_idxs]
		)
		self.__gravity = new_gravity
		self.__policy = new_policy

		return {**gravity_info, **policy_info}

	def predict_demo(
		self,
		source_video_embeddings: np.ndarray,  # [b, d]
		sequential_requirement: np.ndarray,
		to_np: bool = True
	):
		gravity_input = np.concatenate((source_video_embeddings, sequential_requirement), axis=-1)
		self.rng, episodic_instruction = gravity_forward(rng=self.rng, model=self.__gravity, model_input=gravity_input)
		if to_np:
			return np.array(episodic_instruction)
		else:
			return episodic_instruction

	def predict_action(
		self,
		observations: np.ndarray,
		non_functionality: np.ndarray,
		current_params_for_skills: np.ndarray,
		episodic_instruction: np.ndarray,
		to_np: bool = True
	):
		policy_input = np.concatenate(
			(observations, episodic_instruction, non_functionality, current_params_for_skills),
			axis=-1
		)
		self.rng, actions = policy_forward(rng=self.rng, model=self.__policy, policy_input=policy_input)

		if to_np:
			return np.array(actions)
		else:
			return actions

	def _excluded_save_params(self) -> List:
		return SensorBCZ.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in SensorBCZ.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return SensorBCZ.PARAM_COMPONENTS
