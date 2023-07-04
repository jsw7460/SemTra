from typing import Dict, List

import jax
import jax.random
import numpy as np
import optax

from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.intent_emb.algos.updates.vq import intent_vq_updt
from comde.comde_modules.intent_emb.base import BaseIntentEmbedding
from comde.comde_modules.intent_emb.naive.architectures.vq.vq import PrimIntentEmbeddingVQ
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class IntentEmbeddingVQ(BaseIntentEmbedding):
	PARAM_COMPONENTS = ["_IntentEmbeddingVQ__model", "_IntentEmbeddingVQ__vq_decoder"]

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		super(IntentEmbeddingVQ, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)
		self.__model = None
		self.__vq_decoder = None

		if init_build_model:
			self.build_model()

	@property
	def model(self):
		return self.__model

	@model.setter
	def model(self, model):
		self.__model = model

	@property
	def vq_decoder(self):
		return self.__vq_decoder

	@vq_decoder.setter
	def vq_decoder(self, model):
		self.__vq_decoder = model

	def build_model(self):
		vq = PrimIntentEmbeddingVQ(
			net_arch=self.cfg["net_arch"],
			n_codebook=self.cfg["n_codebook"],
			codebook_dim=self.cfg["codebook_dim"]  # == clip dim
		)
		init_skills = np.zeros((1, self.cfg["subseq_len"], self.cfg["skill_dim"]))
		# Note: lang_dim == skill_dim (because they're in clip space)
		init_lang = np.zeros((1, self.cfg["subseq_len"], self.cfg["lang_dim"]))

		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.chain(
			optax.clip(1.0),
			optax.adam(learning_rate=self.cfg["lr"])
		)
		rngs.update(
			{"init_carry": self.rng, "sampling": self.rng + 1}
		)
		self.model = Model.create(
			model_def=vq,
			inputs=[rngs, init_skills, init_lang],
			tx=tx,
			# model_cls=PrimIntentEmbeddingVQ
		)

		vq_decoder = create_mlp(
			output_dim=self.cfg["skill_dim"] + self.cfg["lang_dim"],
			net_arch=self.cfg["vq_decoder_net_arch"],
			activation_fn=self.cfg["vq_decoder_activation_fn"],
			squash_output=False
		)
		self.rng, rngs = get_basic_rngs(self.rng)
		decoder_tx = optax.adam(learning_rate=self.cfg["vq_decoder_lr"])
		init_decoder_input = np.zeros((1, self.cfg["codebook_dim"]))
		self.vq_decoder = Model.create(
			model_def=vq_decoder,
			inputs=[rngs, init_decoder_input],
			tx=decoder_tx
		)

	def update(self, replay_data: ComDeBufferSample, *, low_policy: Model) -> Dict:
		model, vq_decoder, info = intent_vq_updt(
			rng=self.rng,
			intent_emb=self.model,
			low_policy=low_policy,
			vq_decoder=self.vq_decoder,
			language_operators=replay_data.language_operators,
			observations=replay_data.observations,
			actions=replay_data.actions,
			skills=replay_data.skills,
			timesteps=replay_data.timesteps,
			maskings=replay_data.maskings,
			coef_reconstruction=self.cfg["coef_reconstruction"],
			coef_decoder_aid=self.cfg["coef_decoder_aid"],
			coef_commitment=self.cfg["coef_commitment"],
			coef_moving_avg=self.cfg["coef_moving_avg"]
		)
		self.model = model
		self.vq_decoder = vq_decoder
		self.rng, _ = jax.random.split(self.rng)

		return info

	def _excluded_save_params(self) -> List:
		return IntentEmbeddingVQ.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in IntentEmbeddingVQ.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return IntentEmbeddingVQ.PARAM_COMPONENTS
