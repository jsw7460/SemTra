import pickle
import random
from typing import Dict, List

import jax.numpy as jnp
import jax.random
import numpy as np
import optax

from comde.baselines.algos.forwards import flaxvima_forward
from comde.baselines.algos.updates.vima import flaxvima_update as policy_update
from comde.baselines.architectures.flax_vima import PrimFlaxVIMA
from comde.baselines.utils.get_episode_skills import get_episodic_level_skills
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.common.pretrained_forwards.t5_forward import t5_forward
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class FlaxVIMA(BaseLowPolicy):
	"""
	VLPromptDT: Video-Language prompt DT
	Input of PromptDT
		1. State-action-reward history (Like DT)
		2. First image of skill & Language instructions as a prompt (Along 'sub-sequence' axis)
			Note that this (2) is the way how VIMA formulates visual imitation learning.
	"""
	PARAM_COMPONENTS = ["_FlaxVIMA__model"]
	_PREFIX = "Follow this video:"

	def __str__(self):
		return "FlaxVIMA"

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		super().__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

		self.prompt_dim = cfg["prompt_dim"]
		firstimage_path = cfg["firstimage_path"]
		with open(firstimage_path + "_vit", "rb") as f:		# Use ViT !!!!!!!! (Otherwise, dimension mismatch)
			firstimage_mapping = pickle.load(f)  # type: Dict[Union[str, int], List]

		if -1 in firstimage_mapping.keys() or "-1" in firstimage_mapping.keys():
			raise LookupError("-1 is for the padded mapping. Please modify the code here.")

		# We don't want integer key.
		self.firstimage_mapping = {str(k): v for k, v in firstimage_mapping.items()}
		self.firstimage_mapping["-1"] = [np.zeros((self.prompt_dim,))]

		self.prefix = None
		self.tokenizer = None
		self.__model = None
		self.video_parsing = False

		if init_build_model:
			self.build_model()

	def build_model(self):
		b = 3
		l = 7
		init_obs = jnp.zeros((b, 20, self.observation_dim))
		init_act = jnp.zeros((b, 20, self.action_dim))
		init_maskings = jnp.ones((b, 20))
		init_timestep = jnp.zeros((b, 20), dtype="i4")

		init_params = jnp.zeros((b, 4, self.total_param_dim))
		init_prompt = jnp.zeros((b, l, self.prompt_dim))
		init_prompt_maskings = jnp.ones((b, l))

		tx = optax.adam(self.cfg["lr"])
		self.rng, rngs = get_basic_rngs(self.rng)
		self.rng, dist_key = jax.random.split(self.rng, 2)
		rngs = {**rngs, "dist": dist_key}
		_model = PrimFlaxVIMA(**self.cfg["dt_cfg"])
		self.__model = Model.create(
			model_def=_model,
			inputs=[
				rngs,
				init_obs,
				init_act,
				init_maskings,
				init_timestep,
				init_params,
				init_prompt,
				init_prompt_maskings,
				False,
			],
			tx=tx
		)

	def get_param_for_skills(self, replay_data: ComDeBufferSample):
		skill_param_dict = get_episodic_level_skills(
			replay_data=replay_data,
			param_repeats=self.param_repeats,
			n_target_skill=self.cfg["n_target_skill"]
		)
		return skill_param_dict["param_for_source_skills"]

	def get_prompts_from_components(
		self,
		language_guidances: List[str],
		source_skills: np.ndarray,
		n_source_skills: np.ndarray,
		source_skills_idxs: np.ndarray
	):
		buffer_sample = ComDeBufferSample(
			language_guidance=language_guidances,
			source_skills=source_skills,
			n_source_skills=n_source_skills,
			source_skills_idxs=source_skills_idxs
		)
		return self.get_prompts(buffer_sample)

	def get_prompts(self, replay_data: ComDeBufferSample):
		source_skills = replay_data.source_skills
		n_source_skills = replay_data.n_source_skills.reshape(-1, 1)

		prompts = []
		for source_skills_idx in replay_data.source_skills_idxs:
			tmp_prompts = np.array([random.choice(self.firstimage_mapping[str(sk)]) for sk in source_skills_idx])
			prompts.append(tmp_prompts)
		image_prompts = np.array(prompts)

		tokens_dict = t5_forward(replay_data.language_guidance)

		text_prompts = tokens_dict["language_embedding"]
		text_masks = tokens_dict["attention_mask"]

		prompts = np.concatenate((text_prompts, image_prompts), axis=1)

		batch_size = source_skills.shape[0]
		image_masks = np.arange(source_skills.shape[1]).reshape(1, -1)  # [1, M]
		image_masks = np.repeat(image_masks, repeats=batch_size, axis=0)  # [b, M]
		image_masks = np.where(image_masks < n_source_skills, 1, 0)

		prompts_maskings = np.concatenate((text_masks, image_masks), axis=1)

		return prompts, prompts_maskings

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		params_for_skills = self.get_param_for_skills(replay_data)
		prompts, prompts_maskings = self.get_prompts(replay_data)

		new_policy, info = policy_update(
			policy=self.__model,
			rng=self.rng,
			observations=replay_data.observations,
			actions=replay_data.actions,
			maskings=replay_data.maskings,
			timesteps=replay_data.timesteps,
			params_for_skills=params_for_skills,
			prompts={"prompts": prompts},
			prompts_maskings={"prompts_maskings": prompts_maskings},
		)
		self.rng, _ = jax.random.split(self.rng)
		self.__model = new_policy
		self.n_update += 1

		return info

	def predict(
		self,
		observations: jnp.ndarray,  # d_o
		maskings: jnp.ndarray,
		actions: jnp.ndarray,  # d_a
		timesteps: jnp.ndarray,
		param_for_skills: jnp.ndarray,
		prompts: jnp.ndarray,
		prompts_maskings: jnp.ndarray,
		to_np: bool = True,
		**kwargs
	) -> np.ndarray:
		self.rng, _ = jax.random.split(self.rng)

		self.rng, pred_actions = flaxvima_forward(
			rng=self.rng,
			model=self.model,
			observations=observations,
			actions=actions,
			maskings=maskings,
			timesteps=timesteps,
			param_for_skills=param_for_skills,
			prompts=prompts,
			prompts_maskings=prompts_maskings,
		)

		pred_actions = pred_actions[:, -1, ...]
		if to_np:
			return np.array(pred_actions)
		else:
			return pred_actions

	def evaluate(self, *args, **kwargs) -> Dict:
		return dict()

	@property
	def model(self):
		return self.__model

	def _excluded_save_params(self) -> List:
		return FlaxVIMA.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in FlaxVIMA.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return FlaxVIMA.PARAM_COMPONENTS
