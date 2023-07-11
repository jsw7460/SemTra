import pickle
import random
from typing import Dict

import jax.numpy as jnp
import jax.random
import numpy as np
import optax

from comde.baselines.algos.updates.vima import flaxvima_update as policy_update
from comde.baselines.architectures.sensor_flaxvima import PrimSensorFlaxVIMA
from comde.baselines.flaxvima import FlaxVIMA
from comde.baselines.utils.get_episode_emb import get_sensor_text_embeddings
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.common.pretrained_forwards.t5_forward import t5_forward
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model


class SensorFlaxVIMA(FlaxVIMA):
	PARAM_COMPONENTS = ["_SensorFlaxVIMA__model"]

	def __str__(self):
		return "SensorFlaxVIMA"

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		self.sensors_inst = None
		self.sensor_subseq_len = cfg["sensor_subseq_len"]
		self.n_target_skill = cfg["n_target_skill"]
		super(SensorFlaxVIMA, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

	def build_model(self):
		with open(self.cfg["sensor_instruction_path"], "rb") as f:
			self.sensors_inst = pickle.load(f)

		b = 3
		l = 7
		init_obs = jnp.zeros((b, 20, self.observation_dim))
		init_act = jnp.zeros((b, 20, self.action_dim))
		init_maskings = jnp.ones((b, 20))
		init_timestep = jnp.zeros((b, 20), dtype="i4")

		init_params = jnp.zeros((b, 4, self.total_param_dim))

		init_sensor_prompt = jnp.zeros((b, 60, self.observation_dim + self.action_dim))
		init_sensor_prompt_maskings = jnp.ones((b, 60))

		init_lang_prompt = jnp.zeros((b, l, self.prompt_dim))
		init_lang_prompt_maskings = jnp.ones((b, l))

		tx = optax.adam(self.cfg["lr"])
		self.rng, rngs = get_basic_rngs(self.rng)
		self.rng, dist_key = jax.random.split(self.rng, 2)
		rngs = {**rngs, "dist": dist_key}
		_model = PrimSensorFlaxVIMA(**self.cfg["dt_cfg"])
		self.__model = Model.create(
			model_def=_model,
			inputs=[
				rngs,
				init_obs,
				init_act,
				init_maskings,
				init_timestep,
				init_params,
				init_sensor_prompt,
				init_sensor_prompt_maskings,
				init_lang_prompt,
				init_lang_prompt_maskings,
				False,
			],
			tx=tx
		)

	def get_prompts(self, replay_data: ComDeBufferSample):
		prompts = []
		for source_skills_idx in replay_data.source_skills_idxs:
			tmp_prompts = np.array([random.choice(self.firstimage_mapping[str(sk)]) for sk in source_skills_idx])
			prompts.append(tmp_prompts)

		tokens_dict = t5_forward(replay_data.language_guidance)

		lang_prompts = tokens_dict["language_embedding"]
		lang_masks = tokens_dict["attention_mask"]

		info = get_sensor_text_embeddings(replay_data=replay_data, sensor_dict=self.sensors_inst, text_dict=None)

		so = info["source_observations"]
		sa = info["source_actions"]
		sensor_prompts = np.concatenate((so, sa), axis=-1)
		sensor_masks = np.ones((sensor_prompts.shape[0], sensor_prompts.shape[1]))

		return {
			"language_prompts": lang_prompts,
			"language_prompts_maskings": lang_masks,
			"sensor_prompts": sensor_prompts,
			"sensor_prompts_maskings": sensor_masks
		}

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		params_for_skills = self.get_param_for_skills(replay_data)
		prompts_dict = self.get_prompts(replay_data)

		prompts = {
			"language_prompts": prompts_dict["language_prompts"],
			"sensor_prompts": prompts_dict["sensor_prompts"]
		}
		prompts_maskings = {
			"language_prompts_maskings": prompts_dict["language_prompts_maskings"],
			"sensor_prompts_maskings": prompts_dict["sensor_prompts_maskings"]
		}

		new_policy, info = policy_update(
			policy=self.__model,
			rng=self.rng,
			observations=replay_data.observations,
			actions=replay_data.actions,
			maskings=replay_data.maskings,
			timesteps=replay_data.timesteps,
			params_for_skills=params_for_skills,
			prompts=prompts,
			prompts_maskings=prompts_maskings,
		)
		self.rng, _ = jax.random.split(self.rng)
		self.__model = new_policy
		self.n_update += 1

		return info
