import pickle
import random
from typing import Dict, List

import jax.numpy as jnp
import jax.random
import numpy as np
import optax
# from transformers.models.t5.tokenization_t5 import T5Tokenizer as Tokenizer
from transformers import AutoTokenizer

from comde.baselines.algos.updates.vima import vima_update as policy_update
from comde.baselines.architectures.vima import PrimVIMA
from comde.baselines.utils.get_episode_skills import get_episodic_level_skills
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params
from comde.baselines.algos.forwards import vima_forward


class VIMA(BaseLowPolicy):
	"""
	VLPromptDT: Video-Language prompt DT
	Input of PromptDT
		1. State-action-reward history (Like DT)
		2. First image of skill & Language instructions as a prompt (Along 'sub-sequence' axis)
			Note that this (2) is the way how VIMA formulates visual imitation learning.
	"""
	PARAM_COMPONENTS = ["_VIMA__model"]
	_PREFIX = "Follow this video:"

	def __str__(self):
		return "VIMA"

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		super().__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

		self.embed_dim = cfg["embed_dim"]
		self.prompt_dim = cfg["prompt_dim"]
		self.xf_num_layers = cfg["xf_num_layers"]
		self.sattn_num_heads = cfg["sattn_num_heads"]
		self.xattn_num_heads = cfg["xattn_num_heads"]

		firstimage_path = cfg["firstimage_path"]
		with open(firstimage_path, "rb") as f:
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
		init_obs = jnp.zeros((b, l, self.observation_dim))
		init_act = jnp.zeros((b, l, self.action_dim))
		init_timestep = jnp.zeros((b, l), dtype="i4")
		init_prompt = jnp.zeros((b, 4))
		init_prompt_assets = jnp.zeros((b, 4, self.prompt_dim))
		init_prompt_maskings = jnp.ones((b, 4))
		init_prompt_assets_maskings = jnp.ones((b, 4))
		maskings = jnp.ones((b, l))

		tx = optax.adam(self.cfg["lr"])
		self.rng, rngs = get_basic_rngs(self.rng)
		self.rng, dist_key = jax.random.split(self.rng, 2)
		rngs = {**rngs, "dist": dist_key}
		_model = PrimVIMA(
			embed_dim=self.embed_dim,
			prompt_dim=self.prompt_dim,
			action_dim=self.action_dim,
			xf_num_layers=self.xf_num_layers,
			sattn_num_heads=self.sattn_num_heads,
			xattn_num_heads=self.xattn_num_heads
		)

		self.__model = Model.create(
			model_def=_model,
			inputs=[
				rngs,
				init_obs,
				maskings,
				init_act,
				init_timestep,
				init_prompt,
				init_prompt_assets,
				init_prompt_maskings,
				init_prompt_assets_maskings,
				False,
			],
			tx=tx
		)

		if self.tokenizer is None:
			self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

	def get_param_for_skills(self, replay_data: ComDeBufferSample):
		skill_param_dict = get_episodic_level_skills(replay_data, param_repeats=self.param_repeats)
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
		prompts = np.array(prompts)

		tokens = self.tokenizer(replay_data.language_guidance, return_tensors="np", padding=True)
		prefix = tokens["input_ids"]
		prefix_maskings = tokens["attention_mask"]

		batch_size = source_skills.shape[0]
		prompts_maskings = np.arange(source_skills.shape[1]).reshape(1, -1)  # [1, M]
		prompts_maskings = np.repeat(prompts_maskings, repeats=batch_size, axis=0)  # [b, M]
		prompts_maskings = np.where(prompts_maskings < n_source_skills, 1, 0)

		return prefix, prompts, prefix_maskings, prompts_maskings

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		prefix, prompts, prefix_maskings, prompts_maskings = self.get_prompts(replay_data)
		new_policy, info = policy_update(
			policy=self.__model,
			rng=self.rng,
			observations=replay_data.observations,
			maskings=replay_data.maskings,
			actions=replay_data.actions,
			timesteps=replay_data.timesteps,
			prompts=prefix,
			prompt_assets=prompts,
			prompts_maskings=prefix_maskings,
			prompt_assets_maskings=prompts_maskings,
		)
		self.rng, _ = jax.random.split(self.rng)
		self.__model = new_policy
		self.n_update += 1

		action_preds = info["action_preds"]
		target_actions = info["target_actions"]

		if self.n_update % 100 == 0:
			print("Action preds", action_preds[:5])
			print("Target actions", target_actions[:5])
			print("\n\n\n")
		return info

	def predict(
		self,
		observations: jnp.ndarray,  # d_o
		maskings: jnp.ndarray,
		actions: jnp.ndarray,  # d_a
		timesteps: jnp.ndarray,
		prompts: jnp.ndarray,
		prompt_assets: jnp.ndarray,
		prompts_maskings: jnp.ndarray,
		prompt_assets_mask: jnp.ndarray,  # [b, l]
		to_np: bool = True,
		**kwargs
	) -> np.ndarray:
		self.rng, _ = jax.random.split(self.rng)

		self.rng, pred_actions = vima_forward(
			rng=self.rng,
			model=self.model,
			observations=observations,
			observations_mask=maskings,
			actions=actions,
			timesteps=timesteps,
			prompt=prompts,
			prompt_assets=prompt_assets,
			prompt_mask=prompts_maskings,
			prompt_assets_mask=prompt_assets_mask
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
		return VIMA.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in VIMA.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return VIMA.PARAM_COMPONENTS
