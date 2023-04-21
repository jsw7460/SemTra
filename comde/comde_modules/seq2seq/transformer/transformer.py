from typing import Dict, List

import einops

import jax.random
import numpy as np
import optax

from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.seq2seq.algos.forward import skilltoskill_transformer_forward as fwd
from comde.comde_modules.seq2seq.algos.updates.skilltoskill_transformer import (
	skilltoskill_transformer_updt as continuous_update,
	skilltoskill_transformer_ce_updt as discrete_update
)
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
		self.skill_pred_type = "continuous"

		# self.start_token = None	# type: SkillRepresentation
		# self.end_token = None # type: SkillRepresentation

		self.module_working = (self.cfg["coef_intent"] > 0) or (self.cfg["coef_skill"] > 0)

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

		if self.cfg["skill_pred_type"] == "continuous":
			raise NotImplementedError("Are you sure that skill matching is done via MSE?")
			# pred_dim = self.inseq_dim	# CLIP dim
			# transformer_cfg.update({"skill_pred_dim": self.inseq_dim})
			# self.skill_pred_type = "continuous"

		elif self.cfg["skill_pred_type"] == "discrete":
			# Predict discrete token. +2 for start-end tokens.
			transformer_cfg.update({"skill_pred_dim": transformer_cfg["decoder_cfg"]["max_len"] + 2})
			self.skill_pred_type = "discrete"

		else:
			raise NotImplementedError("Undefined skill prediction type")

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

	def update(self, replay_data: ComDeBufferSample, *, low_policy: BaseLowPolicy) -> Dict:
		# Concatenate source skills and language first

		if self.module_working:
			raise NotImplementedError("Currently, we don't update skill to skill transformer.")
			src_skill_language_concat = self.concat_skill_language(
				source_skills=replay_data.source_skills,
				n_source_skills=replay_data.n_source_skills,
				language_operators=replay_data.language_operators
			)

			update_fn_inputs = {
				"rng": self.rng,
				"tr": self.model,
				"low_policy": low_policy.model,
				"context": src_skill_language_concat,
				"target_skills": replay_data.target_skills,
				"observations": replay_data.observations,
				"actions": replay_data.actions,
				"skills": replay_data.skills,
				"skills_order": replay_data.skills_order,
				"timesteps": replay_data.timesteps,
				"maskings": replay_data.maskings,
				"n_source_skills": replay_data.n_source_skills,
				"n_target_skills": replay_data.n_target_skills,
				"start_token": self.start_token.vec,
				"end_token": self.end_token.vec,
				"coef_intent": self.cfg["coef_intent"],
				"coef_skill": self.cfg["coef_skill"]
			}

			if self.skill_pred_type == "continuous":
				raise NotImplementedError("Continuous transformer is obsolete")
				# update_fn = continuous_update

			elif self.skill_pred_type == "discrete":
				update_fn = discrete_update
				update_fn_inputs.update({"target_skills_idxs": replay_data.target_skills_idxs})

			else:
				raise NotImplementedError(f"Undefined skill prediction type: {self.skill_pred_type}")

			new_model, info = update_fn(**update_fn_inputs)
			self.model = new_model
		else:
			info = {"__parameterized_skills": None}

		self.rng, _ = jax.random.split(self.rng)
		return info

	def maybe_done(
		self,
		pred_skills: np.ndarray,  # [b, d]
		get_nearest_token: bool = True
	):
		tokens_vec = np.array([vocab.vec for vocab in self._example_vocabulary])

		_tokens_vec = tokens_vec[np.newaxis, ...]  # [1, M, d]
		pred_skills = pred_skills[:, np.newaxis, ...]  # [b, 1, d]

		if self.skill_pred_type == "discrete":
			pred_skills = np.argmax(pred_skills, axis=-1, keepdims=True)	# [b, 1, n_skills]
			pred_skills = np.take_along_axis(_tokens_vec, pred_skills, axis=1)	# [b, 1, d]

		distance = np.mean((pred_skills - _tokens_vec) ** 2, axis=-1)  # [b, M]

		min_distance_idx = np.argmin(distance, axis=-1)  # [b, ]

		maybe_done = np.where(min_distance_idx == self.end_token.index, 1, 0)  # [b, ]

		if get_nearest_token:
			nearest_tokens = tokens_vec[min_distance_idx]  # [b, d]
			return maybe_done, nearest_tokens, min_distance_idx

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
		x = self.start_token.vec
		x = np.broadcast_to(x, (batch_size, 1, skill_dim))  # [b, 1, d]

		done = False

		context = self.concat_skill_language(
			source_skills=source_skills,
			n_source_skills=n_source_skills,
			language_operators=language_operator
		)

		context_maxlen = context.shape[1]
		ctx_padding_mask = np.arange(context_maxlen)
		ctx_padding_mask = np.broadcast_to(ctx_padding_mask, (batch_size, context_maxlen))  # [b, l]
		ctx_padding_mask = np.where(ctx_padding_mask <= n_source_skills.reshape(-1, 1), 1, 0)  # [b, l]

		pred_skills_seq_raw = []
		pred_skills_seq_quantized = []
		pred_intents_seq = []
		pred_nearest_idxs_seq = []

		t = 0

		while not done:  # Autoregression

			self.rng, predictions = fwd(
				rng=self.rng,
				model=self.model,
				x=x,
				context=context,
				mask=ctx_padding_mask,
			)
			pred_skills = predictions["pred_skills"][:, -1, ...]  # [b, d]
			maybe_done, nearest_tokens, nearest_idxs = self.maybe_done(pred_skills, get_nearest_token=True)

			pred_intents = predictions["pred_intents"][:, -1, ...]
			pred_skills_seq_raw.append(pred_skills)
			pred_skills_seq_quantized.append(nearest_tokens)
			pred_intents_seq.append(pred_intents)
			pred_nearest_idxs_seq.append(nearest_idxs)

			nearest_tokens = nearest_tokens.reshape(batch_size, 1, skill_dim)
			x = np.concatenate((x, nearest_tokens), axis=1)

			done = np.all(maybe_done) or (len(pred_skills_seq_raw) >= self.decoder_max_len)
			t += 1

		pred_skills_raw = np.stack(pred_skills_seq_raw, axis=1)
		pred_skills_quantized = np.stack(pred_skills_seq_quantized, axis=1)
		pred_intents_seq = predictions["pred_intents"].copy()	# Use only final one
		ret = {
			"pred_skills_raw": pred_skills_raw,
			"pred_skills_quantized": pred_skills_quantized,
			"pred_intents": pred_intents_seq,
			"pred_nearest_idxs": pred_nearest_idxs_seq
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

	def _evaluate_continuous(self, replay_data: ComDeBufferSample) -> Dict:
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

	def _evaluate_discrete(self, replay_data: ComDeBufferSample) -> Dict:
		prediction = self.predict(
			source_skills=replay_data.source_skills,
			language_operator=replay_data.language_operators,
			n_source_skills=replay_data.n_source_skills
		)

		pred_nearest_idxs = np.array(prediction["pred_nearest_idxs"])
		pred_nearest_idxs = einops.rearrange(pred_nearest_idxs, "l b -> b l")	# [b, M]

		target_skills_idxs = replay_data.target_skills_idxs  # [b, M, d]
		n_target_skills = replay_data.n_target_skills  # [b, M]

		batch_size = target_skills_idxs.shape[0]

		tgt_mask = np.arange(self.decoder_max_len)
		tgt_mask = np.broadcast_to(tgt_mask, (batch_size, self.decoder_max_len))  # [b, M]
		tgt_mask = np.where(tgt_mask < n_target_skills[..., np.newaxis], 1, 0)  # [b, M]

		pred_nearest_idxs = np.where(tgt_mask == 1, pred_nearest_idxs, -1)
		target_skills_idxs = np.where(tgt_mask == 1, target_skills_idxs, -2)

		match_ratio = np.sum(pred_nearest_idxs == target_skills_idxs) / np.sum(tgt_mask)

		pred_w_target = self.predict_w_teacher_forcing(
			source_skills=replay_data.source_skills,
			language_operator=replay_data.language_operators,
			n_source_skills=replay_data.n_source_skills,
			target=replay_data.target_skills
		)
		pred_intents = pred_w_target["pred_intents"]
		pred_intents = np.take_along_axis(pred_intents, indices=replay_data.skills_order[..., np.newaxis], axis=1)
		return {"s2s/skill_match_ratio": match_ratio, "__intents": pred_intents}

	def evaluate(self, replay_data: ComDeBufferSample) -> Dict:
		if self.module_working:
			if self.skill_pred_type == "continuous":
				return self._evaluate_continuous(replay_data=replay_data)

			elif self.skill_pred_type == "discrete":
				return self._evaluate_discrete(replay_data=replay_data)
		else:
			return {"__parameterized_skills": None}

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
