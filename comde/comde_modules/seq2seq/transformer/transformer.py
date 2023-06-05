import pickle
import random
from typing import Dict, List, Union, Tuple, Optional

import einops
import jax.random
import numpy as np
import optax

from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.seq2seq.algos.forward import skilltoskill_transformer_forward as fwd
from comde.comde_modules.seq2seq.algos.updates.skilltoskill_transformer import (
	skilltoskill_transformer_ce_updt as discrete_update
)
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.seq2seq.transformer.architectures.transformer import PrimSklToSklIntTransformer
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.common.pretrained_forwards.jax_bert_base import bert_base_forward
from comde.utils.common.visualization import dump_attention_weights_images
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params
from comde.utils.save_utils.jax_saves import (
	load_from_zip_file
)


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
		self.save_suffix = self.cfg["save_suffix"]
		self.decoder_max_len = self.cfg["transformer_cfg"]["decoder_cfg"]["max_len"]

		word_embedding_path = cfg["word_embedding_path"]
		with open(word_embedding_path, "rb") as f:
			self.word_embedding = pickle.load(f)  # type: Dict[str, Dict[str, np.ndarray]]

		self.heldout_instructions = {key: [] for key in self.word_embedding.keys()}
		self.reset_heldout_instructions()

		# Compute the max length of sentence.
		sentence_max_len = 0
		for key in self.word_embedding.values():
			assert len(key) >= 5, "Too little variations."
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
		for key, value in self.word_embedding.items():
			heldout_sentence = random.choices(list(value.keys()), k=2)
			self.heldout_instructions[key] = heldout_sentence

	def build_model(self):
		transformer_cfg = self.cfg["transformer_cfg"]
		decoder_maxlen = transformer_cfg["decoder_cfg"]["max_len"]

		# Predict discrete token. +2 for start-end tokens.
		vocab_size = len(self._example_vocabulary)
		# transformer_cfg.update({"skill_pred_dim": transformer_cfg["decoder_cfg"]["max_len"] + 2})
		transformer_cfg.update({"skill_pred_dim": vocab_size})  # We don't predict start token.

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
		All inputs are from replay_data in update function.

		Source skills: [b, M, d] (There are zero paddings for axis=1)

		1. From natural_languages, sample sentence tokens	([b, L, d])
		2. Concatenate with source skills, which is a context for the Transformer. Thus, shape [b, M + L, d]
		3. For M + L tokens in each batch, compute the context masking, which has the shape [b, M + L].
		"""
		b, m, d = source_skills.shape
		n_source_skills = n_source_skills.reshape(b, 1)

		sequential_requirements = []
		pad_lengths = []
		variations = None
		for str_seq_req in natural_languages:
			if training:
				variations = [
					var for var in self.word_embedding[str_seq_req].keys()
					if var not in self.heldout_instructions[str_seq_req]
				]

			else:
				variations = [
					var for var in self.word_embedding[str_seq_req].keys()
					if var not in self.heldout_instructions[str_seq_req]
				]

			seq_req = [self.word_embedding[str_seq_req][var] for var in variations]
			seq_req = random.choice(seq_req)
			seq_req = np.squeeze(seq_req, axis=0)

			# Apply the zero padding
			pad_length = self.sentence_max_len - seq_req.shape[0]
			pad_lengths.append(pad_length)
			padding = np.zeros((pad_length, self.inseq_dim))
			padded_seq_req = np.concatenate((seq_req, padding), axis=0)
			sequential_requirements.append(padded_seq_req)

		sequential_requirements = np.array(sequential_requirements)  # [b, 20, 768]
		pad_lengths = np.array(pad_lengths).reshape(-1, 1)  # [b, 1]
		seq_req_padding_mask = np.arange(self.sentence_max_len)
		seq_req_padding_mask = np.broadcast_to(seq_req_padding_mask, (b, self.sentence_max_len))
		seq_req_padding_mask = np.where(seq_req_padding_mask < self.sentence_max_len - pad_lengths, 1, 0)

		source_skills_padding_mask = np.arange(m)
		source_skills_padding_mask = np.broadcast_to(source_skills_padding_mask, (b, m))
		source_skills_padding_mask = np.where(source_skills_padding_mask < n_source_skills, 1, 0)

		info = {
			"q": source_skills,
			"kv": sequential_requirements,
			"q_mask": source_skills_padding_mask,
			"kv_mask": seq_req_padding_mask,
			"natural_language_variations": variations,
			"natural_language_padding_mask": seq_req_padding_mask,
			"source_skills_padding_mask": source_skills_padding_mask
		}

		return info

	def update(self, replay_data: ComDeBufferSample, low_policy: Optional[BaseLowPolicy]) -> Dict:
		qkv_info = bert_base_forward(replay_data.language_guidance)

		q = qkv_info["language_embedding"]
		kv = qkv_info["language_embedding"]
		q_mask = qkv_info["attention_mask"]
		kv_mask = qkv_info["attention_mask"]

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
			"low_policy": low_policy.model,
			"coef_low_policy": self.cfg["coef_low_policy"],
			"prompting_fn": low_policy.get_prompt,
			"timesteps": replay_data.timesteps
		}

		new_model, info = discrete_update(**update_fn_inputs)

		self.n_update += 1
		self.model = new_model
		self.rng, _ = jax.random.split(self.rng)
		return info

	def visualize_attention(
		self,
		language_guidance: List[str],
		target: np.ndarray = None
	):
		if target is not None:
			prediction, info = self.predict_w_teacher_forcing(
				language_guidance=language_guidance,
				target=target,
				return_qkv_info=True
			)
		else:
			prediction, info = self.predict(
				language_guidance=language_guidance,
				stochastic=True,
				return_qkv_info=True
			)

		attentions = prediction["__decoder_attention_weights"][-1][0]

		# attentions = prediction["decoder_attention_weights"][0]  # First layer attention score
		nl_masks = info["attention_mask"]

		for t, (nl, nl_mask, att) in enumerate(zip(language_guidance, nl_masks, attentions)):
			nl_length = np.sum(nl_mask)
			# 1. Average over heads
			att = np.mean(att, axis=0)
			# 2. Remove start tokens and remove paddings
			att = att[:, 2: nl_length]

			dump_attention_weights_images(
				path=f"/home/jsw7460/comde/visualization/attention_weights/{self.save_suffix}_step{str(self.n_update)}_{t}",
				natural_language=nl,
				attention_matrix=att
			)

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
		language_guidance: List[str],
		stochastic: bool = False,
		return_qkv_info: bool = False,
	) -> Union[Tuple, Dict]:
		"""
			Given source skills and language operator, this function generate the target skills.
			This can be done autoregressively.
		"""

		done = False

		qkv_info = bert_base_forward(language_guidance)

		q = qkv_info["language_embedding"]
		kv = qkv_info["language_embedding"]
		q_mask = qkv_info["attention_mask"]
		kv_mask = qkv_info["attention_mask"]

		batch_size = q.shape[0]
		skill_dim = q.shape[-1]

		x = self.start_token.vec
		x = np.broadcast_to(x, (batch_size, 1, skill_dim))  # [b, 1, d]

		pred_skills_seq_raw = []
		pred_skills_seq_quantized = []
		pred_nearest_idxs_seq = []
		encoder_attention_weights = None
		decoder_attention_weights = []

		t = 0

		prev_maybe_done = np.zeros((batch_size,))
		target_skills_masks = []

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

			if encoder_attention_weights is None:
				encoder_attention_weights = predictions.pop("encoder_attention_weights")

			decoder_attention_weight = predictions.pop("decoder_attention_weights")
			decoder_attention_weights.append(decoder_attention_weight)

			pred_skills = predictions["pred_skills"][:, -1, ...]  # [b, d]
			maybe_done, nearest_tokens, nearest_idxs = self.maybe_done(
				pred_skills,
				get_nearest_token=True,
				stochastic_sampling=stochastic
			)

			# If you once predict done (end token), then we set done for all after that
			maybe_done = np.logical_or(maybe_done, prev_maybe_done)
			target_skills_masks.append(1 - maybe_done)

			prev_maybe_done = maybe_done

			pred_skills_seq_raw.append(pred_skills)
			pred_skills_seq_quantized.append(nearest_tokens)
			pred_nearest_idxs_seq.append(nearest_idxs)

			nearest_tokens = nearest_tokens.reshape(batch_size, 1, skill_dim)
			x = np.concatenate((x, nearest_tokens), axis=1)

			done = np.all(maybe_done) or (len(pred_skills_seq_raw) >= self.decoder_max_len)
			t += 1

		target_skills_masks = np.stack(target_skills_masks, axis=1)

		pred_skills_raw = np.stack(pred_skills_seq_raw, axis=1)
		pred_skills_quantized = np.stack(pred_skills_seq_quantized, axis=1)

		pred_nearest_idxs = np.array(pred_nearest_idxs_seq)
		pred_target_skills = einops.rearrange(pred_nearest_idxs, "l b -> b l")

		ret = {
			"__pred_skills_raw": pred_skills_raw,
			"__pred_skills_quantized": pred_skills_quantized,
			"__pred_nearest_idxs": pred_nearest_idxs_seq,
			"__pred_target_skills": pred_target_skills,
			"__encoder_attention_weights": encoder_attention_weights,
			"__decoder_attention_weights": decoder_attention_weights,
			"__target_skills_masks": target_skills_masks,
			"__language_guidance": q,
			"__language_guidance_mask": q_mask
		}
		if return_qkv_info:
			return ret, qkv_info
		else:
			return ret

	def predict_w_teacher_forcing(
		self,
		language_guidance: List[str],
		target: np.ndarray,
		return_qkv_info: bool = False
	):
		qkv_info = bert_base_forward(language_guidance)

		q = qkv_info["language_embedding"]
		kv = qkv_info["language_embedding"]
		q_mask = qkv_info["attention_mask"]
		kv_mask = qkv_info["attention_mask"]

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
		if return_qkv_info:
			return predictions, qkv_info
		else:
			return predictions

	def _evaluate_continuous(self, replay_data: ComDeBufferSample) -> Dict:
		raise NotImplementedError("Continuous mode is obsolete")

	def _evaluate_discrete(self, replay_data: ComDeBufferSample) -> Dict:
		prediction = self.predict(language_guidance=replay_data.language_guidance)

		pred_nearest_idxs = np.array(prediction["__pred_nearest_idxs"])
		pred_nearest_idxs = einops.rearrange(pred_nearest_idxs, "l b -> b l")  # [b, M]

		target_skills_idxs = replay_data.target_skills_idxs  # [b, M, d]
		n_target_skills = replay_data.n_target_skills  # [b, M]

		batch_size = target_skills_idxs.shape[0]

		tgt_mask = np.arange(self.decoder_max_len)
		tgt_mask = np.broadcast_to(tgt_mask, (batch_size, self.decoder_max_len))  # [b, M]
		tgt_mask = np.where(tgt_mask < n_target_skills[..., np.newaxis], 1, 0)  # [b, M]

		if pred_nearest_idxs.shape[-1] < self.decoder_max_len:
			# Predict skill done too early
			pad_len = self.decoder_max_len - pred_nearest_idxs.shape[-1]
			dummy_skills_idxs = np.zeros((batch_size, pad_len)) - 1
			pred_nearest_idxs = np.concatenate((pred_nearest_idxs, dummy_skills_idxs), axis=-1)

		pred_nearest_idxs = np.where(tgt_mask == 1, pred_nearest_idxs, -1)
		target_skills_idxs = np.where(tgt_mask == 1, target_skills_idxs, -2)

		match_ratio = np.sum(pred_nearest_idxs == target_skills_idxs) / np.sum(tgt_mask)
		return {"s2s/skill_match_ratio": match_ratio, **prediction}

	def evaluate(self, replay_data: ComDeBufferSample, visualization: bool = False) -> Dict:
		eval_info = self._evaluate_discrete(replay_data=replay_data)
		self.visualize_attention(language_guidance=replay_data.language_guidance)
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
