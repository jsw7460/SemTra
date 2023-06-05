from typing import Dict, List, Union, Tuple, Optional

import jax
import numpy as np
import optax

from comde.comde_modules.low_policies.algos.forwards import skill_promptdt_forward as fwd
from comde.comde_modules.low_policies.algos.updates.skill_promptdt import skill_promptdt_updt
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params
from .architectures.skill_prompt_dt import PrimSkillPromptDT, get_prompt


class SkillPromptDT(BaseLowPolicy):
	PARAM_COMPONENTS = ["_SkillPromptDT__model"]

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool = True
	):
		super(SkillPromptDT, self).__init__(
			seed=seed,
			cfg=cfg,
			init_build_model=init_build_model
		)
		self.__model = None
		self.max_ep_len = cfg["max_ep_len"]

		self.get_prompt = get_prompt

		if init_build_model:
			self.build_model()

	@property
	def model(self) -> Model:
		return self.__model

	@model.setter
	def model(self, value):
		self.__model = value

	@staticmethod
	def check_shape(observations: np.ndarray, actions: np.ndarray, skills: np.ndarray, timesteps: np.ndarray):
		if len(actions) > 0:
			assert observations.ndim == actions.ndim == skills.ndim == 3, \
				[
					f"{comp} should be 3-dimensional but has {comp.ndim}"
					for comp in [observations, actions, skills] if comp.ndim != 3
				]
			assert timesteps.ndim == 2, f"timestep should be 2-dimensional but has {timesteps.ndim}"

	def str_to_activation(self, activation_fn: str):
		return activation_fn

	def build_model(self):
		transformer = PrimSkillPromptDT(
			gpt2_config=self.cfg["gpt2_config"],
			obs_dim=self.observation_dim,
			act_dim=self.action_dim,
			hidden_size=self.cfg["hidden_size"],
			act_scale=self.cfg["act_scale"],
			max_ep_len=self.max_ep_len,
			normalization_mean=self.normalization_mean,
			normalization_std=self.normalization_std,
		)
		self.rng, rngs = get_basic_rngs(self.rng)
		self.rng, transformer_dropout = jax.random.split(self.rng)
		rngs.update({"transformer_dropout": transformer_dropout})
		subseq_len = self.cfg["subseq_len"]
		init_observations = np.random.normal(size=(1, subseq_len, self.observation_dim))
		init_actions = np.random.normal(size=(1, subseq_len, self.action_dim))
		init_skills = np.random.normal(size=(1, subseq_len, self.skill_dim))
		prompt_len = 48
		init_prompts = np.random.normal(size=(1, prompt_len, self.skill_dim))
		init_prompts_maskings = np.zeros((1, prompt_len))
		init_timesteps = np.zeros((1, self.cfg["subseq_len"]), dtype="i4")
		init_masks = np.ones((1, self.cfg["subseq_len"]))
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
				init_prompts,
				init_prompts_maskings,
				init_timesteps,
				init_masks,
			],
			tx=tx
		)

	def update(self, replay_data: ComDeBufferSample, prompt_dict: Dict) -> Dict:
		new_model, info = skill_promptdt_updt(
			rng=self.rng,
			dt=self.model,
			observations=replay_data.observations,
			actions=replay_data.actions,
			skills=replay_data.skills,
			prompts=prompt_dict["prompts"],
			prompts_maskings=prompt_dict["prompts_maskings"],
			timesteps=replay_data.timesteps.astype("i4"),
			maskings=replay_data.maskings,
			action_targets=np.copy(replay_data.actions),
		)

		self.model = new_model
		self.rng, _ = jax.random.split(self.rng)

		return info

	def get_padded_components(
		self,
		observations: np.ndarray,
		actions: np.ndarray,
		skills: np.ndarray,
		timesteps: np.ndarray,
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

		self.check_shape(observations, actions, skills, timesteps)
		batch_size = observations.shape[0]
		subseq_len = self.cfg["subseq_len"]

		history_len = observations.shape[1]
		observations = np.concatenate(
			(observations, np.zeros((batch_size, subseq_len - history_len, self.observation_dim))), axis=1
		)
		actions = np.concatenate(
			(actions, np.zeros((batch_size, subseq_len - actions.shape[1], self.action_dim))), axis=1
		)
		skills = np.concatenate(
			(skills, np.zeros((batch_size, subseq_len - skills.shape[1], self.skill_dim))), axis=1
		)
		timesteps = np.concatenate(
			(timesteps, np.zeros((batch_size, subseq_len - timesteps.shape[1]))), axis=1
		)
		maskings = np.concatenate(
			(np.ones((batch_size, history_len)), np.zeros((batch_size, subseq_len - history_len))), axis=1
		)
		return observations, actions, skills, timesteps, maskings

	def predict(self, *args, **kwargs) -> np.ndarray:
		historical_actions = self.predict_and_get_historical_actions(*args, **kwargs)
		return historical_actions[:, -1]

	def predict_and_get_historical_actions(
		self,
		observations: np.ndarray,
		actions: np.ndarray,
		skills: np.ndarray,
		timesteps: np.ndarray,
		maskings: Union[np.ndarray] = None,
		to_np: bool = True,
		prompts_dict: Optional[Dict] = None,
		seq2seq_info: Optional[Dict] = None
	) -> np.ndarray:

		assert (prompts_dict is not None) or (seq2seq_info is not None), \
			"One of prompts_dict or seq2seq_info should be given"

		self.check_shape(observations, actions, skills, timesteps)

		subseq_len = self.cfg["subseq_len"]
		cur_subseq_len = observations.shape[1]

		# Longer than subseq_len -> Truncate
		# Shorter than subseq_len -> Set zero paddings and get maskings.
		if cur_subseq_len < subseq_len:
			raise NotImplementedError("Are you sure this if loop required?")

		subseq_len = self.cfg["subseq_len"]
		observations = observations[:, :subseq_len, :]
		actions = actions[:, :subseq_len, :]
		skills = skills[:, :subseq_len, :]
		timesteps = timesteps[:, :subseq_len]
		if maskings is None:
			maskings = np.ones((1, subseq_len))

		if prompts_dict is None:
			prompts_dict = self.get_prompts_dict_from_seq2seq_info(seq2seq_info)

		prompts = prompts_dict["prompts"]
		prompts_maskings = prompts_dict["prompts_maskings"]

		self.rng, action_preds = fwd(
			rng=self.rng,
			model=self.model,
			observations=observations,
			actions=actions,
			skills=skills,
			prompts=prompts,
			prompts_maskings=prompts_maskings,
			timesteps=timesteps.astype("i4"),
			maskings=maskings,
		)

		if to_np:
			return np.array(action_preds)
		else:
			return action_preds

	def get_prompts_dict_from_seq2seq_info(self, seq2seq_info: Dict):
		prompt_ingradient = self.reshape_att_for_prompt(
			attention_weights=seq2seq_info["__decoder_attention_weights"][-1]
		)
		prompts_dict = get_prompt(
			attention_weights=prompt_ingradient["attention_weights"],
			language_guidance=seq2seq_info["__language_guidance"],
			target_skills_mask=seq2seq_info["__target_skills_masks"],
			language_guidance_mask=seq2seq_info["__language_guidance_mask"]
		)

		return prompts_dict

	def evaluate(
		self,
		replay_data: ComDeBufferSample,
		seq2seq_info: Dict
	) -> Dict:
		if self.cfg["use_optimal_lang"]:
			raise NotImplementedError("Obsolete")

		prompts_dict = self.get_prompts_dict_from_seq2seq_info(seq2seq_info)

		observations = replay_data.observations
		actions = replay_data.actions
		skills = replay_data.skills
		timesteps = replay_data.timesteps
		maskings = replay_data.maskings

		action_preds = self.predict_and_get_historical_actions(
			observations=observations,
			actions=actions,
			skills=skills,
			prompts_dict=prompts_dict,
			timesteps=timesteps,
			maskings=maskings,
		)
		action_preds = action_preds.reshape(-1, self.action_dim) * maskings.reshape(-1, 1)
		action_targets = actions.reshape(-1, self.action_dim) * maskings.reshape(-1, 1)

		mse_error = np.sum(np.mean((action_preds - action_targets) ** 2, axis=-1)) / np.sum(maskings)

		eval_info = {
			"skill_decoder/mse_error": mse_error,
			"skill_decoder/mse_error_scaled(x100)": mse_error * 100
		}

		return eval_info

	@staticmethod
	def reshape_att_for_prompt(attention_weights: np.ndarray) -> Dict:
		"""Output of this function is supposed to be used in low policy"""
		attention_weights = attention_weights[0]
		attention_weights = np.mean(attention_weights, axis=1)
		return {"attention_weights": attention_weights}

	def _excluded_save_params(self) -> List:
		return SkillPromptDT.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in SkillPromptDT.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return SkillPromptDT.PARAM_COMPONENTS
