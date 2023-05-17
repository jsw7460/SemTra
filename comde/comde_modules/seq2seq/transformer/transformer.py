import pickle
import random
from typing import Dict, List, Tuple

import einops
import jax.random
import numpy as np
import optax

from comde.comde_modules.seq2seq.algos.forward import skilltoskill_transformer_forward as fwd
from comde.comde_modules.seq2seq.algos.updates.skilltoskill_transformer import (
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

		self.decoder_max_len = self.cfg["transformer_cfg"]["decoder_cfg"]["max_len"]

		sentence_embedding_path = cfg["sentence_embedding_path"]
		with open(sentence_embedding_path, "rb") as f:
			self.sentence_embedding = pickle.load(f)  # type: Dict[str, Dict[str, np.ndarray]]

		self.heldout_instructions = {key: [] for key in self.sentence_embedding.keys()}
		self.reset_heldout_instructions()

		# Compute the max length of sentence.
		sentence_max_len = 0
		for key in self.sentence_embedding.values():
			assert len(key) > 30, "Too little variations."
			for var in key.values():
				sentence_length = var.shape[1]
				if sentence_length > sentence_max_len:
					sentence_max_len = sentence_length
		self.sentence_max_len = sentence_max_len

		if init_build_model:
			self.build_model()

	@property
	def model(self):
		return self.__model

	@model.setter
	def model(self, model):
		self.__model = model

	def reset_heldout_instructions(self):
		for key, value in self.sentence_embedding.items():
			heldout_sentence = random.choices(list(value.keys()), k=3)
			self.heldout_instructions[key] = heldout_sentence

	def build_model(self):
		transformer_cfg = self.cfg["transformer_cfg"]
		decoder_maxlen = transformer_cfg["decoder_cfg"]["max_len"]
		encoder_maxlen = transformer_cfg["encoder_cfg"]["max_len"]

		# Predict discrete token. +2 for start-end tokens.
		vocab_size = len(self._example_vocabulary)
		# transformer_cfg.update({"skill_pred_dim": transformer_cfg["decoder_cfg"]["max_len"] + 2})
		transformer_cfg.update({"skill_pred_dim": vocab_size - 2})  # We don't predict start token.

		transformer = PrimSklToSklIntTransformer(**transformer_cfg)
		init_x = np.zeros((1, decoder_maxlen, self.inseq_dim))  # 512 dim
		init_q = np.zeros((1, 7, self.inseq_dim))  # 512 dim
		init_kv = np.zeros((1, 15, self.inseq_dim))
		init_q_mask = np.zeros((1, 7))
		init_kv_mask = np.zeros((1, 15))

		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.chain(optax.clip(1.0), optax.adam(learning_rate=self.cfg["lr"]))
		self.model = Model.create(
			model_def=transformer,
			inputs=[rngs, init_x, init_q, init_kv, init_q_mask, init_kv_mask],
			tx=tx
		)

	def get_qkv_and_mask(
		self,
		source_skills: np.ndarray,
		n_source_skills: np.ndarray,
		natural_languages: List[str],
		training: bool
	) -> Dict[str, np.ndarray]:
		"""
		Source skills: [b, M, d] (There are zero paddings for axis=1)

		1. From natural_languages, sample sentence tokens	([b, L, d])
		2. Concatenate with source skills, which is a context for the Transformer. Thus, shape [b, M + L, d]
		3. For M + L tokens in each batch, compute the context masking, which has the shape [b, M + L].
		"""
		b, m, d = source_skills.shape
		n_source_skills = n_source_skills.reshape(b, 1)

		sequential_requirements = []
		pad_lengths = []
		for str_seq_req in natural_languages:
			if training:
				seq_req = [
					self.sentence_embedding[str_seq_req][var]
					for var in self.sentence_embedding[str_seq_req].keys()
					if var not in self.heldout_instructions[str_seq_req]
				]
			else:
				seq_req = [
					self.sentence_embedding[str_seq_req][var]
					for var in self.sentence_embedding[str_seq_req].keys()
					if var in self.heldout_instructions[str_seq_req]
				]
			seq_req = random.choice(seq_req)
			seq_req = np.squeeze(seq_req, axis=0)

			# Apply the zero padding
			pad_length = self.sentence_max_len - seq_req.shape[0]
			pad_lengths.append(pad_length)
			padding = np.zeros((pad_length, self.inseq_dim))
			padded_seq_req = np.concatenate((seq_req, padding), axis=0)
			sequential_requirements.append(padded_seq_req)

		sequential_requirements = np.array(sequential_requirements)	 # [b, 20, 768]
		pad_lengths = np.array(pad_lengths).reshape(-1, 1)	# [b, 1]
		seq_req_padding_mask = np.arange(self.sentence_max_len)
		seq_req_padding_mask = np.broadcast_to(seq_req_padding_mask, (b, self.sentence_max_len))
		seq_req_padding_mask = np.where(seq_req_padding_mask < self.sentence_max_len - pad_lengths, 1, 0)

		source_skills_padding_mask = np.arange(m)
		source_skills_padding_mask = np.broadcast_to(source_skills_padding_mask, (b, m))
		source_skills_padding_mask = np.where(source_skills_padding_mask < n_source_skills, 1, 0)

		info = {
			"q": sequential_requirements,
			"kv": source_skills,
			"q_mask": seq_req_padding_mask,
			"kv_mask": source_skills_padding_mask
		}

		return info

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		# Concatenate source skills and language first
		# return {"__parameterized_skills": None}	# For the present, not used.

		info = self.get_qkv_and_mask(
			source_skills=replay_data.source_skills,
			n_source_skills=replay_data.n_source_skills,
			natural_languages=replay_data.str_sequential_requirement,
			training=True
		)

		q = info["q"]
		kv = info["kv"]
		q_mask = info["q_mask"]
		kv_mask = info["kv_mask"]

		update_fn_inputs = {
			"rng": self.rng,
			"tr": self.model,
			"encoder_q": q,
			"encoder_kv": kv,
			"q_mask": q_mask,
			"kv_mask": kv_mask,
			"target_skills": replay_data.target_skills,
			"target_skills_idxs": replay_data.target_skills_idxs,
			"observations": replay_data.observations,
			"actions": replay_data.actions,
			"skills": replay_data.skills,
			"maskings": replay_data.maskings,
			"n_source_skills": replay_data.n_source_skills,
			"n_target_skills": replay_data.n_target_skills,
			"start_token": self.start_token.vec,
		}

		new_model, info = discrete_update(**update_fn_inputs)

		self.model = new_model
		self.rng, _ = jax.random.split(self.rng)
		return info

	def maybe_done(
		self,
		pred_skills: np.ndarray,  # [b, d]
		get_nearest_token: bool = True,
		stochastic_sampling: bool = False,
	):
		tokens_vec = np.array([vocab.vec for vocab in self._example_vocabulary])

		_tokens_vec = tokens_vec[np.newaxis, ...]  # [1, M, d]
		pred_skills = pred_skills[:, np.newaxis, ...]  # [b, 1, d]

		if stochastic_sampling:
			pred_skills = jax.random.categorical(self.rng, logits=pred_skills, axis=-1)
			pred_skills = pred_skills[..., np.newaxis]
		else:
			pred_skills = np.argmax(pred_skills, axis=-1, keepdims=True)  # [b, 1, 1]
		pred_skills = np.take_along_axis(_tokens_vec, pred_skills, axis=1)  # [b, 1, d]

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
		natural_languages: List[str],
		n_source_skills: np.ndarray,  # [b,]: Indication of true length of source_skills without zero padding
		stochastic: bool = False
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

		info = self.get_qkv_and_mask(
			source_skills=source_skills,
			n_source_skills=n_source_skills,
			natural_languages=natural_languages,
			training=True
		)

		q = info["q"]
		kv = info["kv"]
		q_mask = info["q_mask"]
		kv_mask = info["kv_mask"]

		pred_skills_seq_raw = []
		pred_skills_seq_quantized = []
		pred_nearest_idxs_seq = []

		t = 0

		while not done:  # Autoregression

			self.rng, predictions = fwd(
				rng=self.rng,
				model=self.model,
				x=x,
				encoder_q=q,
				encoder_kv=kv,
				q_mask=q_mask,
				kv_mask=kv_mask,
			)
			pred_skills = predictions["pred_skills"][:, -1, ...]  # [b, d]
			maybe_done, nearest_tokens, nearest_idxs = self.maybe_done(
				pred_skills,
				get_nearest_token=True,
				stochastic_sampling=stochastic
			)

			pred_skills_seq_raw.append(pred_skills)
			pred_skills_seq_quantized.append(nearest_tokens)
			pred_nearest_idxs_seq.append(nearest_idxs)

			nearest_tokens = nearest_tokens.reshape(batch_size, 1, skill_dim)
			x = np.concatenate((x, nearest_tokens), axis=1)

			done = np.all(maybe_done) or (len(pred_skills_seq_raw) >= self.decoder_max_len)
			t += 1

		pred_skills_raw = np.stack(pred_skills_seq_raw, axis=1)
		pred_skills_quantized = np.stack(pred_skills_seq_quantized, axis=1)
		ret = {
			"pred_skills_raw": pred_skills_raw,
			"pred_skills_quantized": pred_skills_quantized,
			"pred_nearest_idxs": pred_nearest_idxs_seq
		}
		return ret

	def predict_w_teacher_forcing(
		self,
		source_skills: np.ndarray,
		natural_languages: List[str],
		n_source_skills: np.ndarray,
		target: np.ndarray
	):
		info = self.get_qkv_and_mask(
			source_skills=source_skills,
			n_source_skills=n_source_skills,
			natural_languages=natural_languages,
			training=True
		)

		q = info["q"]
		kv = info["kv"]
		q_mask = info["q_mask"]
		kv_mask = info["kv_mask"]

		self.rng, predictions = fwd(
			rng=self.rng,
			model=self.model,
			x=target,
			encoder_q=q,
			encoder_kv=kv,
			q_mask=q_mask,
			kv_mask=kv_mask,
			deterministic=True
		)
		return predictions

	def _evaluate_continuous(self, replay_data: ComDeBufferSample) -> Dict:
		raise NotImplementedError("Continuous mode is obsolete")

	def _evaluate_discrete(self, replay_data: ComDeBufferSample) -> Dict:
		prediction = self.predict(
			source_skills=replay_data.source_skills,
			natural_languages=replay_data.str_sequential_requirement,
			n_source_skills=replay_data.n_source_skills
		)

		pred_nearest_idxs = np.array(prediction["pred_nearest_idxs"])
		pred_nearest_idxs = einops.rearrange(pred_nearest_idxs, "l b -> b l")  # [b, M]

		target_skills_idxs = replay_data.target_skills_idxs  # [b, M, d]
		n_target_skills = replay_data.n_target_skills  # [b, M]

		batch_size = target_skills_idxs.shape[0]

		tgt_mask = np.arange(self.decoder_max_len)
		tgt_mask = np.broadcast_to(tgt_mask, (batch_size, self.decoder_max_len))  # [b, M]
		tgt_mask = np.where(tgt_mask < n_target_skills[..., np.newaxis], 1, 0)  # [b, M]

		pred_nearest_idxs = np.where(tgt_mask == 1, pred_nearest_idxs, -1)
		target_skills_idxs = np.where(tgt_mask == 1, target_skills_idxs, -2)

		match_ratio = np.sum(pred_nearest_idxs == target_skills_idxs) / np.sum(tgt_mask)

		return {"s2s/skill_match_ratio": match_ratio}

	def evaluate(self, replay_data: ComDeBufferSample) -> Dict:
		eval_info =  self._evaluate_discrete(replay_data=replay_data)
		self.reset_heldout_instructions()
		return eval_info

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

	def str_to_activation(self, activation_fn: str):
		pass
