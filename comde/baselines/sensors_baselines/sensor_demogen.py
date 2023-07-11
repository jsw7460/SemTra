import pickle
from typing import Dict, List

import numpy as np
import optax

from comde.baselines.algos.forwards import (
	demogen_gravity_forward as gravity_forward,
	demogen_policy_forward as policy_forward
)
from comde.baselines.algos.updates.demogen import (
	demogen_policy_update as policy_update,
	demogen_gravity_update as gravity_update
)
from comde.baselines.utils.get_episode_emb import get_sensor_text_embeddings, get_video_text_embeddings
from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params
from comde.baselines.demogen import DemoGen

class SensorDemoGen(DemoGen):
	PARAM_COMPONENTS = [
		"_SensorDemoGen__policy",
		"_SensorDemoGen__gravity"
	]

	def __str__(self):
		return "SensorDemoGen"

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		self.sensors_inst = None
		self.sensor_subseq_len = cfg["sensor_subseq_len"]
		self.n_target_skill = cfg["n_target_skill"]
		super(SensorDemoGen, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

	def _set_dictionaries(self):
		super(SensorDemoGen, self)._set_dictionaries()
		with open(self.cfg["sensor_instruction_path"], "rb") as f:
			self.sensors_inst = pickle.load(f)

	def _build_policy(self):
		policy_cfg = self.cfg["policy_cfg"].copy()
		policy_lr = policy_cfg.pop("lr")
		policy = create_mlp(**policy_cfg)
		policy = Scaler(base_model=policy, scale=np.array(self.cfg["act_scale"]))
		init_obs = np.zeros((1, self.observation_dim))
		init_demo = np.zeros((1, self.video_dim))
		init_params = np.zeros((1, self.nonfunc_dim + self.total_param_dim))
		policy_input = np.concatenate((init_obs, init_demo, init_params), axis=-1)
		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.adam(policy_lr)
		self.__policy = Model.create(model_def=policy, inputs=[rngs, policy_input], tx=tx)

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

	def get_source_target_info(self, replay_data: ComDeBufferSample):
		sensor_info = get_sensor_text_embeddings(
			sensor_dict=self.sensors_inst,
			text_dict=self.episodic_inst,
			replay_data=replay_data
		)
		so = sensor_info["source_observations"]
		sa = sensor_info["source_actions"]
		batch_size = so.shape[0]

		source = np.concatenate((so, sa), axis=-1).reshape(batch_size, -1)

		video_info = get_video_text_embeddings(
			video_feature_dict=self.video_feature_dict,
			replay_data=replay_data,
			text_dict=self.episodic_inst
		)

		ret_info = {
			"source": source,
			"target": video_info["target_video_embeddings"]
		}

		return ret_info

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		info = self.get_source_target_info(replay_data)
		source_video_embeddings = info["source"]
		target_video_embedding = info["target"]

		new_gravity, gravity_info = gravity_update(
			rng=self.rng,
			gravity=self.__gravity,
			source_video_embeddings=source_video_embeddings,
			sequential_requirement=replay_data.sequential_requirement,
			target_video_embedding=target_video_embedding
		)
		# Expand the parameter
		params_for_skills = np.repeat(replay_data.params_for_skills, axis=-1, repeats=self.param_repeats)

		video_embedding = gravity_info.pop("__pred_target_video")
		new_policy, policy_info = policy_update(
			rng=self.rng,
			policy=self.__policy,
			observations=replay_data.observations,
			video_embedding=video_embedding,
			non_functionality=replay_data.non_functionality,
			parameters=params_for_skills,
			actions=replay_data.actions,
			maskings=replay_data.maskings
		)
		self.__gravity = new_gravity
		self.__policy = new_policy
		return {**gravity_info, **policy_info}