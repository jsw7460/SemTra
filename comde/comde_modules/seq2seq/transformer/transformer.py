from typing import Dict

import jax.random
import numpy as np
import optax
import pickle
from comde.comde_modules.seq2seq.algos.updates.skilltoskill_transformer import skilltoskill_transformer_updt
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.seq2seq.transformer.architectures.transformer import PrimSklToSklIntTransformer
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.common.lang_representation import LanguageRepresentation


class SklToSklIntTransformer(BaseSeqToSeq):
	PARAM_COMPONENTS = ["__SklToSklIntTransformer__model"]

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		super(SklToSklIntTransformer, self).__init__(
			seed=seed,
			cfg=cfg,
			init_build_model=init_build_model
		)
		# Source skills: [b, l, d], Target skills: [b, l, d]
		self.__model = None

		self.tokens = None # type: Dict[str, LanguageRepresentation]

		if init_build_model:
			self.build_model()

	@property
	def model(self):
		return self.__model

	@model.setter
	def model(self, model):
		self.__model = model

	def build_model(self):
		transformer_cfg = self.cfg["skill_intent_transformer_cfg"]
		transformer = PrimSklToSklIntTransformer(**transformer_cfg)
		init_x = np.zeros((1, transformer_cfg["decoder_cfg"]["max_len"], self.inseq_dim))  # 512 dim
		init_context = np.zeros((1, transformer_cfg["encoder_cfg"]["max_len"], self.inseq_dim))  # 512 dim
		init_mask = np.zeros((1, transformer_cfg["encoder_cfg"]["max_len"]))

		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.chain(optax.clip(1.0), optax.adam(learning_rate=self.cfg["lr"]))
		self.model = Model.create(
			model_def=transformer,
			inputs=[rngs, init_x, init_context, init_mask],
			tx=tx
		)

		with open(self.cfg["token_path"], "rb") as f:
			token_dict = pickle.load(f)

		self.tokens = token_dict

	@staticmethod
	def concat_skill_language(replay_data: ComDeBufferSample):
		"""
		Source skills: [b, M, d] (There are zero paddings for axis=1)
		Language operators: [b, d]

		Output:
			Source-skill concatenated vector [b, M + 1, d]
		"""
		b, m, d = replay_data.source_skills.shape
		n_source_skills = replay_data.n_source_skills
		source_skills = replay_data.source_skills

		padded_source_skills = np.concatenate((source_skills, np.zeros((b, 1, d))), axis=1)  # [b, M + 1, d]

		residue = np.zeros_like(padded_source_skills)  # [b, M + 1, d]
		residue[np.arange(b), n_source_skills, ...] = replay_data.language_operators

		return padded_source_skills + residue

	def update(self, replay_data: ComDeBufferSample, *, low_policy: Model) -> Dict:
		# Concatenate source skills and language first
		src_skill_language_concat = self.concat_skill_language(replay_data)

		new_model, info = skilltoskill_transformer_updt(
			rng=self.rng,
			tr=self.model,
			low_policy=low_policy,
			context=src_skill_language_concat,
			target_skills=replay_data.target_skills,
			observations=replay_data.observations,
			actions=replay_data.actions,
			skills=replay_data.skills,
			skills_order=replay_data.skills_order,
			timesteps=replay_data.timesteps,
			maskings=replay_data.maskings,
			n_source_skills=replay_data.n_source_skills,
			n_target_skills=replay_data.n_target_skills,
			coef_intent=self.cfg["coef_intent"],
			coef_skill=self.cfg["coef_skill"]
		)
		self.model = new_model
		self.rng, _ = jax.random.split(self.rng)

		return info

	def predict(self, *args, **kwargs) -> np.ndarray:
		pass

	def evaluate(self, replay_data: ComDeBufferSample) -> Dict:
