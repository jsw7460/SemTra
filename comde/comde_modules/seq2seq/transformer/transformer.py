import pickle
from typing import Dict, List

import jax.random
import numpy as np
import optax

from comde.comde_modules.seq2seq.algos.forward import skilltoskill_transformer_forward as fwd
from comde.comde_modules.seq2seq.algos.updates.skilltoskill_transformer import skilltoskill_transformer_updt
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.seq2seq.transformer.architectures.transformer import PrimSklToSklIntTransformer
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class SklToSklIntTransformer(BaseSeqToSeq):
	PARAM_COMPONENTS = ["_SklToSklIntTransformer__model"]

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		super(SklToSklIntTransformer, self).__init__(
			seed=seed,
			cfg=cfg,
			init_build_model=init_build_model
		)
		# Source skills: [b, l, d], Target skills: [b, l, d]
		self.__model = None

		self.decoder_max_len = self.cfg["skill_intent_transformer_cfg"]["decoder_cfg"]["max_len"]
		if init_build_model:
			self.build_model()

	@property
	def model(self):
		return self.__model

	@model.setter
	def model(self, model):
		self.__model = model

	def update_tokens(self, new_tokens: Dict):
		self.tokens = {**self.tokens, **new_tokens}

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

		token_dict = {k: v.vec for k, v in token_dict.items()}
		self.update_tokens(token_dict)

	@staticmethod
	def concat_skill_language(source_skills: np.ndarray, n_source_skills: np.ndarray, language_operators: np.ndarray):
		"""
		Source skills: [b, M, d] (There are zero paddings for axis=1)
		Language operators: [b, d]

		Output:
			Source-skill concatenated vector [b, M + 1, d]
		"""
		b, m, d = source_skills.shape
		source_skills = source_skills
		n_source_skills = n_source_skills.reshape(b, 1)

		source_skills_padding_mask = np.arange(m)
		source_skills_padding_mask = np.broadcast_to(source_skills_padding_mask, (b, m))
		source_skills_padding_mask = np.where(source_skills_padding_mask < n_source_skills, 1, 0)

		source_skills = source_skills * source_skills_padding_mask.reshape(b, m, 1)
		padded_source_skills = np.concatenate((source_skills, np.zeros((b, 1, d))), axis=1)  # [b, M + 1, d]

		residue = np.zeros_like(padded_source_skills)  # [b, M + 1, d]
		residue[np.arange(b), n_source_skills, ...] = language_operators

		return padded_source_skills + residue

	def update(self, replay_data: ComDeBufferSample, *, low_policy: Model) -> Dict:
		# Concatenate source skills and language first
		src_skill_language_concat = self.concat_skill_language(
			source_skills=replay_data.source_skills,
			n_source_skills=replay_data.n_source_skills,
			language_operators=replay_data.language_operators
		)

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
			start_token=self.tokens["start_token"],
			end_token=self.tokens["end_token"],
			coef_intent=self.cfg["coef_intent"],
			coef_skill=self.cfg["coef_skill"]
		)
		self.maybe_done(pred_skills=info["__pred_skills"][:, 0, ...])

		self.model = new_model
		self.rng, _ = jax.random.split(self.rng)

		return info

	def maybe_done(
		self,
		pred_skills: np.ndarray,  # [b, d]
		get_nearest_token: bool = True
	):
		distance_dict = {k: None for k in self.tokens.keys()}

		end_idx = list(distance_dict.keys()).index("end_token")

		tokens_vec = np.array(list(self.tokens.values()))  # [M, d]

		_tokens_vec = tokens_vec[np.newaxis, ...]  # [1, M, d]
		pred_skills = pred_skills[:, np.newaxis, ...]  # [b, 1, d]

		distance = np.mean((pred_skills - _tokens_vec) ** 2, axis=-1)  # [b, M]

		min_distance_idx = np.argmin(distance, axis=-1)  # [b, ]

		maybe_done = np.where(min_distance_idx == end_idx, 1, 0)  # [b, ]

		if get_nearest_token:
			nearest_tokens = tokens_vec[min_distance_idx]  # [b, d]
			return maybe_done, nearest_tokens

		return maybe_done

	def predict(
		self,
		source_skills: np.ndarray,  # [b, M, d]
		language_operator: np.ndarray,  # [b, d]
		n_source_skills: np.ndarray,  # [b,]: Indication of true length of source_skills without zero padding
	) -> Dict:
		"""
			Given source skills and language operator, this function generate the target skills.
			This can be done autoregressively.
		"""
		batch_size = source_skills.shape[0]
		skill_dim = source_skills.shape[-1]
		x = self.tokens["start_token"]
		x = np.broadcast_to(x, (batch_size, 1, skill_dim))  # [b, 1, d]

		done = False

		context = self.concat_skill_language(
			source_skills=source_skills,
			n_source_skills=n_source_skills,
			language_operators=language_operator
		)

		# print("=" * 100)
		# for i in range(batch_size):
		# 	print(f"context {i}", np.mean(context[i], axis=-1))

		context_maxlen = context.shape[1]
		ctx_padding_mask = np.arange(context_maxlen)
		ctx_padding_mask = np.broadcast_to(ctx_padding_mask, (batch_size, context_maxlen))  # [b, l]
		ctx_padding_mask = np.where(ctx_padding_mask <= n_source_skills.reshape(-1, 1), 1, 0)  # [b, l]

		pred_skills_seq_raw = []
		pred_skills_seq_quantized = []
		pred_intents_seq = []

		t = 0

		while not done:  # Autoregression

			# print("Timestep", t)

			self.rng, predictions = fwd(
				rng=self.rng,
				model=self.model,
				x=x,
				context=context,
				mask=ctx_padding_mask,
			)
			pred_skills = predictions["pred_skills"][:, -1, ...]  # [b, d]
			maybe_done, nearest_tokens = self.maybe_done(pred_skills, get_nearest_token=True)

			pred_intents = predictions["pred_intents"][:, -1, ...]

			pred_skills_seq_raw.append(pred_skills)
			pred_skills_seq_quantized.append(nearest_tokens)
			pred_intents_seq.append(pred_intents)

			nearest_tokens = nearest_tokens.reshape(batch_size, 1, skill_dim)
			x = np.concatenate((x, nearest_tokens), axis=1)

			done = np.all(maybe_done) or (len(pred_skills_seq_raw) >= self.decoder_max_len)
			t += 1

		pred_skills_raw = np.stack(pred_skills_seq_raw, axis=1)
		pred_skills_quantized = np.stack(pred_skills_seq_quantized, axis=1)
		pred_intents_seq = np.stack(pred_intents_seq, axis=1)

		ret = {
			"pred_skills_raw": pred_skills_raw,
			"pred_skills_quantized": pred_skills_quantized,
			"pred_intents": pred_intents_seq
		}
		return ret

	def predict_w_teacher_forcing(
		self,
		source_skills: np.ndarray,
		language_operator: np.ndarray,
		n_source_skills: np.ndarray,
		target: np.ndarray
	):
		context = self.concat_skill_language(
			source_skills=source_skills,
			n_source_skills=n_source_skills,
			language_operators=language_operator
		)
		batch_size = source_skills.shape[0]
		context_maxlen = context.shape[1]
		ctx_padding_mask = np.arange(context_maxlen)
		ctx_padding_mask = np.broadcast_to(ctx_padding_mask, (batch_size, context_maxlen))  # [b, l]
		ctx_padding_mask = np.where(ctx_padding_mask <= n_source_skills.reshape(-1, 1), 1, 0)  # [b, l]

		self.rng, predictions = fwd(
			rng=self.rng,
			model=self.model,
			x=target,
			context=context,
			mask=ctx_padding_mask
		)
		return predictions

	def evaluate(self, replay_data: ComDeBufferSample) -> Dict:

		prediction = self.predict(
			source_skills=replay_data.source_skills,
			language_operator=replay_data.language_operators,
			n_source_skills=replay_data.n_source_skills
		)

		target_skills = replay_data.target_skills  # [b, M, d]
		n_target_skills = replay_data.n_target_skills  # [b, M]
		pred_skills_raw = prediction["pred_skills_raw"]  # [b, l, d]

		batch_size = target_skills.shape[0]
		skill_dim = target_skills.shape[-1]
		pad = np.zeros((batch_size, target_skills.shape[1] - pred_skills_raw.shape[1], skill_dim))
		pred_skills_raw = np.concatenate((pred_skills_raw, pad), axis=1)  # [b, M, d]

		tgt_mask = np.arange(self.decoder_max_len)
		tgt_mask = np.broadcast_to(tgt_mask, (batch_size, self.decoder_max_len))  # [b, M]
		tgt_mask = np.where(tgt_mask < n_target_skills[..., np.newaxis], 1, 0)[..., np.newaxis]  # [b, M, 1]

		pred_skills_raw = pred_skills_raw * tgt_mask
		target_skills = target_skills * tgt_mask
		skills_mse = np.sum(np.mean((pred_skills_raw - target_skills) ** 2, axis=-1)) / np.sum(tgt_mask)

		pred_w_target = self.predict_w_teacher_forcing(
			source_skills=replay_data.source_skills,
			language_operator=replay_data.language_operators,
			n_source_skills=replay_data.n_source_skills,
			target=replay_data.target_skills
		)
		pred_intents = pred_w_target["pred_intents"]
		pred_intents = np.take_along_axis(pred_intents, indices=replay_data.skills_order[..., np.newaxis], axis=1)
		return {"s2s/skill_loss": skills_mse, "__intents": pred_intents}

	def _excluded_save_params(self) -> List:
		return SklToSklIntTransformer.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in SklToSklIntTransformer.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return SklToSklIntTransformer.PARAM_COMPONENTS