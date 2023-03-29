from typing import List, Dict

import jax.random
import numpy as np
import optax

from comde.comde_modules.seq2seq.algos.forward import skilltoskill_model_forward as fwd
from comde.comde_modules.seq2seq.algos.updates.skilltoskill_lstm import skilltoskill_model_updt
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.seq2seq.lstm.architectures.rnn.lstm import PrimLSTM
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params

EPS = 1e-9


class SkillToSkillLSTM(BaseSeqToSeq):
	PARAM_COMPONENTS = [
		"_SkillToSkillLSTM__model",
	]

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool = True,
	):
		super(SkillToSkillLSTM, self).__init__(
			seed=seed,
			cfg=cfg,
			init_build_model=init_build_model
		)
		self.__model = None
		self.__task_decoder = None
		self.max_iter_len = cfg["max_iter_len"]
		self.coef_skill_loss = cfg["coef_skill_loss"]

		if init_build_model:
			self.build_model()

	@property
	def model(self):
		return self.__model

	@model.setter
	def model(self, model):
		self.__model = model

	@property
	def task_decoder(self) -> Model:
		return self.__task_decoder

	@task_decoder.setter
	def task_decoder(self, task_decoder):
		self.__task_decoder = task_decoder

	def build_model(self):
		lstm_model = PrimLSTM(
			max_iter_len=self.max_iter_len,
			embed_dim=self.cfg["embed_dim"],
			hidden_dim=self.cfg["hidden_dim"],
			dropout=self.cfg["dropout"],
			activation_fn=self.cfg["activation_fn"],
			embed_net_arch=self.cfg["embed_net_arch"]
		)

		# Input of task encoder: state-action
		init_sequence = np.zeros((1, self.max_iter_len, self.cfg["skill_dim"]))
		self.rng, rngs = get_basic_rngs(self.rng)
		encoder_scheduler = optax.exponential_decay(
			init_value=self.cfg["lr"],
			decay_rate=self.cfg["decay_rate"],
			transition_steps=self.cfg["transition_steps"],
			transition_begin=self.cfg["transition_begin"],
		)
		encoder_tx = optax.chain(
			optax.clip(1.0),
			optax.adam(learning_rate=encoder_scheduler)
		)
		rngs.update(
			{"init_carry": self.rng, "sampling": self.rng + 1}
		)
		self.model = Model.create(
			model_def=lstm_model,
			inputs=[rngs, init_sequence, 1],
			tx=encoder_tx
		)

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		new_model, info = skilltoskill_model_updt(
			rng=self.rng,
			model=self.model,
			source_skills=replay_data.source_skills,
			language_operators=replay_data.language_operators,
			target_skills=replay_data.target_skills,
			observations=replay_data.observations,
			n_target_skills=replay_data.n_target_skills,
			coef_skill_loss=self.coef_skill_loss,
		)
		self.model = new_model
		self.rng, _ = jax.random.split(self.rng)
		return info

	def predict(
		self,
		source_skills: np.ndarray,  # [b, M, d]
		language_operator: np.ndarray,  # [b, d]
	) -> np.ndarray:
		batch_size = source_skills.shape[0]
		language_operator = language_operator[:, np.newaxis, ...]
		input_seq = np.concatenate((source_skills, language_operator), axis=1)
		self.rng, prediction = fwd(
			rng=self.rng,
			model=self.model,
			sequence=input_seq,
			batch_size=batch_size,
		)
		return prediction

	def evaluate(
		self,
		replay_data: ComDeBufferSample,
		np_idx_to_skills: np.ndarray = None
	) -> Dict:
		batch_size = replay_data.source_skills.shape[0]
		source_skills = replay_data.source_skills
		target_skills = replay_data.target_skills
		language_operator = replay_data.language_operators
		n_target_skills = replay_data.n_target_skills
		max_possible_skills = replay_data.target_skills.shape[1]

		lstm_output = self.predict(
			source_skills=source_skills,
			language_operator=language_operator
		)  # [b, max_iter_len, d]
		lstm_maskings = np.arange(self.max_iter_len)[np.newaxis, ...]
		lstm_maskings = np.repeat(lstm_maskings, repeats=batch_size, axis=0)
		# [b, max_iter_len, 1]
		lstm_maskings = np.where(lstm_maskings < n_target_skills.reshape(-1, 1), 1, 0)[..., np.newaxis]

		lstm_output = lstm_output * lstm_maskings
		skills_loss \
			= self.coef_skill_loss * np.mean((lstm_output[:, :max_possible_skills, ...] - target_skills) ** 2, axis=-1)

		skills_loss = np.sum(skills_loss) / np.sum(n_target_skills)

		eval_info = {"skill_lstm/skill_error": skills_loss, "__seq2seq_output": lstm_output}

		if np_idx_to_skills is not None:
			if batch_size > 1:
				return eval_info
			n_target_skills = np.squeeze(n_target_skills, axis=0)
			trunc_lstm_output = lstm_output[:, :n_target_skills, ...]  # [1, 4, d]
			idx_to_skills = np_idx_to_skills[:, np.newaxis, ...]  # [9, 1, d]

			distances = (trunc_lstm_output - idx_to_skills) ** 2  # [9, 4, d]
			distances = np.sum(distances, axis=-1)  # [9, 4]

			nearest_idx = np.argmin(distances, axis=0)
			pred_target_skills = np_idx_to_skills[nearest_idx]  # [4, d]

			distance_to_gt = np.sum(
				(pred_target_skills - target_skills.squeeze(axis=0)[:n_target_skills]) ** 2,
				axis=-1
			)
			correct_pred = np.sum(distance_to_gt < EPS)

			eval_info.update({
				"skill_lstm/distance_to_target": np.mean(distance_to_gt),
				"skill_lstm/accuracy": correct_pred / n_target_skills
			})

		return eval_info

	def _excluded_save_params(self) -> List:
		return SkillToSkillLSTM.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in SkillToSkillLSTM.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params

		return params_dict

	def _get_load_params(self) -> List[str]:
		return SkillToSkillLSTM.PARAM_COMPONENTS
