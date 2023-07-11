import pickle
import random
from typing import Dict, List

import jax.random
import jax.random
import numpy as np
import optax

from comde.baselines.algos.forwards import sensor_promptdt_forward as forward
from comde.baselines.algos.updates.prompt_dt import promptdt_update
from comde.baselines.architectures.sensor_promptdt import PrimSensorPromptDT
from comde.baselines.source_lang import SourceLanguagePromptDT
from comde.baselines.utils.get_episode_emb import get_sensor_text_embeddings
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.common.pretrained_forwards.jax_bert_base import bert_base_forward
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class SensorSourceLanguagePromptDT(SourceLanguagePromptDT):
	PARAM_COMPONENTS = ["policy"]

	def __str__(self):
		return "SensorSourceLanguagePromptDT"

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		self.sensors_inst = None
		self.sensor_subseq_len = cfg["sensor_subseq_len"]
		self.n_target_skill = cfg["n_target_skill"]
		super(SensorSourceLanguagePromptDT, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

	def build_model(self):
		with open(self.cfg["sensor_instruction_path"], "rb") as f:
			self.sensors_inst = pickle.load(f)

		policy = PrimSensorPromptDT(**self.cfg["dt_cfg"])
		b = 3
		l = 7
		init_obs = np.zeros((b, l, self.observation_dim))
		init_act = np.zeros((b, l, self.action_dim))
		init_rtg = np.zeros((b, l, 1))

		init_sensor_prompt = np.zeros((b, 60, self.observation_dim + self.action_dim))
		init_sensor_prompt_maskings = np.ones((b, 60))
		init_lang_prompt = np.zeros((b, l, self.prompt_dim))
		init_lang_prompt_maskings = np.ones((b, l))

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
				init_sensor_prompt,
				init_sensor_prompt_maskings,
				init_lang_prompt,
				init_lang_prompt_maskings,
				init_seq,
				init_nf,
				init_prm,
				init_timesteps,
				maskings,
				False],
			tx=tx
		)

	def get_prompts(self, replay_data: ComDeBufferSample):
		qkv_info = bert_base_forward(replay_data.language_guidance)

		language_prompts = qkv_info["language_embedding"]
		language_prompts_maskings = qkv_info["attention_mask"]

		info = get_sensor_text_embeddings(replay_data=replay_data, sensor_dict=self.sensors_inst, text_dict=None)

		so = info["source_observations"]
		sa = info["source_actions"]
		sensor_prompts = np.concatenate((so, sa), axis=-1)
		sensor_masks = np.ones((sensor_prompts.shape[0], sensor_prompts.shape[1]))

		prompts_dict = {
			"sensor_prompts": sensor_prompts,
			"sensor_prompts_maskings": sensor_masks,
			"language_prompts": language_prompts,
			"language_prompts_maskings": language_prompts_maskings
		}

		return prompts_dict

	def get_prompts_from_components(
		self,
		language_guidances: List[str] = None,
		source_skills: np.ndarray = None,
		n_source_skills: np.ndarray = None,
		source_skills_idxs: np.ndarray = None
	):
		qkv_info = bert_base_forward(language_guidances)
		language_prompts = qkv_info["language_embedding"]
		language_prompts_maskings = qkv_info["attention_mask"]

		source_obss = []
		source_acts = []
		for src_sk in source_skills_idxs:
			print("SRC SK", src_sk)
			src_obs = [random.choice(self.sensors_inst[sk])["observations"] for sk in src_sk]
			src_act = [random.choice(self.sensors_inst[sk])["actions"] for sk in src_sk]

			source_obss.append(np.concatenate(src_obs, axis=0))
			source_acts.append(np.concatenate(src_act, axis=0))

		so = np.stack(source_obss, axis=0)  # [b, l, d]
		sa = np.stack(source_acts, axis=0)  # [b, l, d]

		sensor_prompts = np.concatenate((so, sa), axis=-1)
		sensor_masks = np.ones((sensor_prompts.shape[0], sensor_prompts.shape[1]))

		return {
			"language_prompts": language_prompts,
			"language_prompts_maskings": language_prompts_maskings,
			"sensor_prompts": sensor_prompts,
			"sensor_prompts_maskings": sensor_masks
		}

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		param_for_source_skills = self.get_param_for_skills(replay_data)
		prompts_dict = self.get_prompts(replay_data)
		prompts = {
			"language_prompts": prompts_dict["language_prompts"],
			"sensor_prompts": prompts_dict["sensor_prompts"]
		}
		prompts_maskings = {
			"language_prompts_maskings": prompts_dict["language_prompts_maskings"],
			"sensor_prompts_maskings": prompts_dict["sensor_prompts_maskings"]
		}
		rtgs = replay_data.rtgs
		rtgs = rtgs.reshape((*rtgs.shape, 1))

		new_policy, info = promptdt_update(
			rng=self.rng,
			policy=self.policy,
			observations=replay_data.observations,
			actions=replay_data.actions,
			rtgs=rtgs,
			prompts=prompts,
			prompts_maskings=prompts_maskings,
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
		sequential_requirement: np.ndarray,  # [b, d]
		non_functionality: np.ndarray,  # [b, d]
		param_for_skills: np.ndarray,  # [b, M, total_prm_dim]
		timesteps: np.ndarray,  # [b, l]
		maskings: np.ndarray,  # [b, l]
		sensor_prompts: np.ndarray = None,
		sensor_prompts_maskings: np.ndarray = None,
		language_prompts: np.ndarray = None,  # [b, M, d]
		language_prompts_maskings: np.ndarray = None,  # [b, M]
		to_np: bool = True,
	) -> np.ndarray:
		rtgs = rtgs[..., np.newaxis]
		self.rng, actions = forward(
			rng=self.rng,
			model=self.model,
			observations=observations,
			actions=actions,
			rtgs=rtgs,
			sensor_prompts=sensor_prompts,
			sensor_prompts_maskings=sensor_prompts_maskings,
			language_prompts=language_prompts,
			language_prompts_maskings=language_prompts_maskings,
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

	def _excluded_save_params(self) -> List:
		return SensorSourceLanguagePromptDT.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in SensorSourceLanguagePromptDT.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return SensorSourceLanguagePromptDT.PARAM_COMPONENTS
