from typing import List, Dict

import jax.random
import numpy as np
import optax

from comde.comde_modules.seq2seq.algos.forward import skilltoskill_model_forward as fwd
from comde.comde_modules.seq2seq.algos.updates.skilltoskill_lstm import skilltoskill_model_updt
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.seq2seq.lstm.architectures.mlp import PrimSkilltoSkillMLP
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params

EPS = 1e-9


class SkillToSkillMLP(BaseSeqToSeq):
	PARAM_COMPONENTS = ["_SkillToSkillMLP__model",]

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool = True,
	):
		super(SkillToSkillMLP, self).__init__(
			seed=seed,
			cfg=cfg,
			init_build_model=init_build_model
		)
		self.__model = None
		self.max_source_skills = cfg["max_source_skills"]
		self.coef_skill_loss = cfg["coef_skill_loss"]

		if init_build_model:
			self.build_model()

	@property
	def model(self):
		return self.__model

	@model.setter
	def model(self, model):
		self.__model = model

	def build_model(self):
		mlp = PrimSkilltoSkillMLP(
			output_dim=self.cfg["skill_dim"],
			net_arch=self.cfg["net_arch"],
			activation_fn=self.cfg["activation_fn"],
			squash_output=False
		)
		# Input of task encoder: state-action
		init_sequence = np.zeros((1, self.max_source_skills + 1, self.inseq_dim))

		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.chain(
			optax.clip(1.0),
			optax.adam(learning_rate=self.cfg["lr"])
		)
		rngs.update(
			{"init_carry": self.rng, "sampling": self.rng + 1}
		)
		self.model = Model.create(
			model_def=mlp,
			inputs=[rngs, init_sequence, 1],
			tx=tx
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

		mlp_output = self.predict(
			source_skills=source_skills,
			language_operator=language_operator
		)  # [b, max_iter_len, d]
		maskings = np.arange(self.max_source_skills)[np.newaxis, ...]
		maskings = np.repeat(maskings, repeats=batch_size, axis=0)
		# [b, max_iter_len, 1]
		maskings = np.where(maskings < n_target_skills.reshape(-1, 1), 1, 0)[..., np.newaxis]
		mlp_output = mlp_output * maskings
		skills_loss \
			= self.coef_skill_loss * np.mean((mlp_output[:, :max_possible_skills, ...] - target_skills) ** 2, axis=-1)
		skills_loss = np.sum(skills_loss) / np.sum(n_target_skills)
		eval_info = {"skill_mlp/skill_error": skills_loss, "__seq2seq_output": mlp_output}

		return eval_info

	def _excluded_save_params(self) -> List:
		return SkillToSkillMLP.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in SkillToSkillMLP.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return SkillToSkillMLP.PARAM_COMPONENTS
