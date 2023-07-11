import random
import pickle
from typing import Dict, List

import jax.random
import jax.random
import numpy as np
import optax

from comde.baselines.algos.forwards import promptdt_forward as forward
from comde.baselines.algos.updates.prompt_dt import promptdt_update
from comde.baselines.architectures.prompt_dt import PrimPromptDT
from comde.baselines.utils.get_episode_skills import get_episodic_level_skills
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params

from comde.utils.common.pretrained_forwards.jax_bert_base import bert_base_forward


class VLPromptDT(BaseLowPolicy):
	"""
	VLPromptDT: Video-Language prompt DT
	Input of PromptDT
		1. State-action-reward history (Like DT)
		2. First image of skill & Language instructions as a prompt (Along 'sub-sequence' axis)
			Note that this (2) is the way how VIMA formulates visual imitation learning.
	"""
	PARAM_COMPONENTS = ["policy"]

	def __str__(self):
		return "VLPromptDT"

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		self.video_parsing = None
		self.video_embedding_dict = None
		# Use default get function for the model trained earlier (before change the code)
		self.conditioning_mode = cfg.get("conditioning_mode", None)
		self.prompt_dim = cfg.get("prompt_dim", None)
		super(VLPromptDT, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)
		self.sequential_requirements_dim = cfg["sequential_requirements_dim"]
		self.policy = None

		if init_build_model:
			self._set_video_embedding_dict()
			self.build_model()

	def _set_video_embedding_dict(self):
		if self.conditioning_mode == "skill":
			self.video_parsing = False
			with open(self.cfg["skill_video_path"], "rb") as f:
				self.video_embedding_dict = pickle.load(f)

		elif self.conditioning_mode == "task":
			self.video_parsing = False
			with open(self.cfg["task_video_path"], "rb") as f:
				self.video_embedding_dict = pickle.load(f)
		else:
			raise NotImplementedError("Wrong conditioning mode")

	def build_model(self):
		policy = PrimPromptDT(**self.cfg["dt_cfg"])
		b = 3
		l = 7
		init_obs = np.zeros((b, l, self.observation_dim))
		init_act = np.zeros((b, l, self.action_dim))
		init_rtg = np.zeros((b, l, 1))
		init_prompt = np.zeros((b, 4, self.prompt_dim))
		init_prompt_maskings = np.ones((b, 4))
		init_seq = np.zeros((b, self.sequential_requirements_dim))
		init_nf = np.zeros((b, self.nonfunc_dim))
		init_prm = np.zeros((b, 4, self.total_param_dim))
		init_timesteps = np.zeros((b, l), dtype="i4")
		maskings = np.ones((b, l))

		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.adam(self.cfg["lr"])
		self.policy = Model.create(
			model_def=policy,
			inputs=[
				rngs,
				init_obs,
				init_act,
				init_rtg,
				init_prompt,
				init_prompt_maskings,
				init_seq,
				init_nf,
				init_prm,
				init_timesteps,
				maskings,
				False],
			tx=tx
		)

	def get_param_for_skills(self, replay_data: ComDeBufferSample):
		skill_param_dict = get_episodic_level_skills(
			replay_data=replay_data,
			param_repeats=self.param_repeats,
			n_target_skill=self.cfg["n_target_skill"]
		)
		return skill_param_dict["param_for_source_skills"]

	def get_prompts_from_components(self, source_skills_idxs: np.ndarray, language_guidance: List[str]):
		"""
		:param source_skills_idxs: [b, M]
		:param language_guidance:  Length b
		:return:
		"""
		_rep_data = ComDeBufferSample(language_guidance=language_guidance, source_skills_idxs=source_skills_idxs)
		return self.get_prompts(_rep_data)

	def get_prompts(self, replay_data: ComDeBufferSample):

		qkv_info = bert_base_forward(replay_data.language_guidance)

		text_prompts = qkv_info["language_embedding"]
		text_prompts_maskings = qkv_info["attention_mask"]

		source_skills_idxs = replay_data.source_skills_idxs

		if self.conditioning_mode == "skill":
			batch_video_prompts = []
			for source_skill_idx in source_skills_idxs:
				episodic_video_prompts = []
				for sk in source_skill_idx:
					if sk != -1:
						episodic_video_prompts.append(random.choice(self.video_embedding_dict[sk]))
				batch_video_prompts.append(np.array(episodic_video_prompts))
			batch_video_prompts = np.array(batch_video_prompts)

		elif self.conditioning_mode == "task":
			batch_video_prompts = []
			for source_skill_idx in source_skills_idxs:
				source_skill_idx = [str(idx) for idx in source_skill_idx]
				source_skill_idx = "".join(source_skill_idx).replace("-1", "")
				for task_seq in self.video_embedding_dict.keys():
					_task_seq = task_seq.replace(" ", "")
					if source_skill_idx in _task_seq:
						batch_video_prompts.append(random.choice(self.video_embedding_dict[task_seq]))
						break
			batch_video_prompts = np.array(batch_video_prompts)
			batch_video_prompts = batch_video_prompts[:, np.newaxis, ...]

		else:
			raise NotImplementedError()

		prompts = np.concatenate((text_prompts, batch_video_prompts), axis=1)

		batch_size = batch_video_prompts.shape[0]
		seq_len = batch_video_prompts.shape[1]
		video_prompts_maskings = np.ones((batch_size, seq_len))

		prompts_maskings = np.concatenate((text_prompts_maskings, video_prompts_maskings), axis=1)
		return prompts, prompts_maskings

	def update(self, replay_data: ComDeBufferSample) -> Dict:

		param_for_source_skills = self.get_param_for_skills(replay_data)
		prompts, prompts_maskings = self.get_prompts(replay_data)

		rtgs = replay_data.rtgs
		rtgs = rtgs.reshape((*rtgs.shape, 1))

		new_policy, info = promptdt_update(
			rng=self.rng,
			policy=self.policy,
			observations=replay_data.observations,
			actions=replay_data.actions,
			rtgs=rtgs,
			prompts={"prompts": prompts},
			prompts_maskings={"prompts_maskings": prompts_maskings},
			sequential_requirement=replay_data.sequential_requirement,
			non_functionality=replay_data.non_functionality,
			param_for_skills=param_for_source_skills,
			timesteps=replay_data.timesteps,
			maskings=replay_data.maskings
		)
		self.policy = new_policy
		self.rng, _ = jax.random.split(self.rng)
		return info

	def predict(
		self,
		observations: np.ndarray,  # [b, l, d]
		actions: np.ndarray,  # [b, l, d]
		rtgs: np.ndarray,  # [b, l]
		prompts: np.ndarray,  # [b, M, d]
		prompts_maskings: np.ndarray,  # [b, M]
		sequential_requirement: np.ndarray,  # [b, d]
		non_functionality: np.ndarray,  # [b, d]
		param_for_skills: np.ndarray,  # [b, M, total_prm_dim]
		timesteps: np.ndarray,  # [b, l]
		maskings: np.ndarray,  # [b, l]
		to_np: bool = True,
	) -> np.ndarray:
		rtgs = rtgs[..., np.newaxis]
		self.rng, actions = forward(
			rng=self.rng,
			model=self.model,
			observations=observations,
			actions=actions,
			rtgs=rtgs,
			prompts=prompts,
			prompts_maskings=prompts_maskings,
			sequential_requirement=sequential_requirement,
			non_functionality=non_functionality,
			param_for_skills=param_for_skills,
			timesteps=timesteps,
			maskings=maskings
		)
		actions = actions[:, -1, ...]
		if to_np:
			return np.array(actions)
		else:
			return actions

	def evaluate(self, *args, **kwargs) -> Dict:
		return dict()

	@property
	def model(self) -> Model:
		return self.policy

	def _excluded_save_params(self) -> List:
		return VLPromptDT.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in VLPromptDT.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return VLPromptDT.PARAM_COMPONENTS
