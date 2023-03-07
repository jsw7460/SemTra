from typing import List, Dict

import jax.random
import numpy as np
import optax

from comde.comde_modules.seq2seq.algos.forward import skilltoskill_lstm_forward as fwd
from comde.comde_modules.seq2seq.algos.updates.skilltoskill_lstm import skilltoskill_lstm_updt
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.seq2seq.lstm.architectures.rnn.lstm import PrimLSTM
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class SkillToSkillLSTM(BaseSeqToSeq):
	PARAM_COMPONENTS = [
		"_LSTM__model",
		"_LSTM__task_decoder"
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
			optax.adamw(learning_rate=encoder_scheduler)
		)
		rngs.update(
			{"init_carry": self.rng, "sampling": self.rng + 1}
		)
		self.model = Model.create(
			model_def=lstm_model,
			inputs=[rngs, init_sequence, 1],
			tx=encoder_tx
		)

	def update(
		self,
		replay_data: ComDeBufferSample,
		*,
		low_policy: Model
	) -> Dict:
		new_model, info = skilltoskill_lstm_updt(
			rng=self.rng,
			lstm=self.model,
			low_policy=low_policy,
			source_skills=replay_data.source_skills,
			language_operators=replay_data.language_operators,
			target_skills=replay_data.target_skills,
			observations=replay_data.observations,
			actions=replay_data.actions,
			timesteps=replay_data.timesteps,
			maskings=replay_data.maskings,
			skills_order=replay_data.skills_order,
			n_target_skills=replay_data.n_target_skills,
			coef_skill_loss=self.cfg["coef_skill_loss"],
			coef_decoder_aid=self.cfg["coef_decoder_aid"]
		)
		self.model = new_model
		self.rng, _ = jax.random.split(self.rng)

		return info

	def predict(
		self,
		source_skills: np.ndarray,  # [b, M, d]
		language_operators: np.ndarray,  # [b, d]
	) -> np.ndarray:
		batch_size = source_skills.shape[0]
		language_operators = language_operators[:, np.newaxis, ...]
		input_seq = np.concatenate((source_skills, language_operators), axis=1)
		self.rng, prediction = fwd(
			rng=self.rng,
			model=self.model,
			sequence=input_seq,
			batch_size=batch_size,
			deterministic=True
		)
		return prediction

	def evaluate(
		self,
		replay_data: ComDeBufferSample
	) -> Dict:
		batch_size = replay_data.source_skills.shape[0]
		source_skills = replay_data.source_skills
		target_skills = replay_data.target_skills
		language_operators = replay_data.language_operators
		n_target_skills = replay_data.n_target_skills
		max_possible_skills = replay_data.target_skills.shape[1]

		lstm_output = self.predict(
			source_skills=source_skills,
			language_operators=language_operators
		)
		lstm_maskings = np.arange(self.max_iter_len)[np.newaxis, ...]
		lstm_maskings = np.repeat(lstm_maskings, repeats=batch_size, axis=0)
		lstm_maskings = np.where(lstm_maskings < n_target_skills, 1, 0)[..., np.newaxis]  # [b, max_iter_len, 1]

		lstm_output = lstm_output * lstm_maskings
		skills_loss = np.sum((lstm_output[:, :max_possible_skills, ...] - target_skills) ** 2) \
					  / np.sum(n_target_skills)

		eval_info = {"skill_lstm/skill_error": skills_loss}

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
