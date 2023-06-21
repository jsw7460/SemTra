from copy import deepcopy
from typing import Dict, List, Union, Tuple

import einops
import jax.random
import numpy as np
import optax

from comde.comde_modules.seq2seq.algos.forward import skilltoskill_transformer_forward as fwd
from comde.comde_modules.seq2seq.algos.updates.skilltoskill_transformer import (
	skilltoskill_transformer_ce_updt as discrete_update
)
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.seq2seq.transformer.architectures.skltoskl_transformer import PrimSkillCompositionTransformer
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.common.natural_languages.lang_representation import SkillRepresentation as LanguageRepresentation
from comde.utils.common.pretrained_forwards.jax_bert_base import bert_base_forward
from comde.utils.common.visualization import dump_attention_weights_images
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class SkillCompositionTransformer(BaseSeqToSeq):
	PARAM_COMPONENTS = ["_SkillCompositionTransformer__model"]

	def __init__(
		self, seed: int,
		cfg: Dict,
		custom_tokens: Dict[str, LanguageRepresentation],
		init_build_model: bool = True
	) -> None:

		self.custom_tokens = custom_tokens
		super(SkillCompositionTransformer, self).__init__(
			seed=seed,
			cfg=cfg,
			init_build_model=init_build_model
		)
		# Source skills: [b, l, d], Target skills: [b, l, d]
		self.__model = None
		self.save_suffix = self.cfg["save_suffix"]

		if init_build_model:
			self.build_model()

	def register_vocabulary(self):

		if self.custom_tokens is not None:
			self.tokens = deepcopy(self.custom_tokens)

		bos_indicator = "start semantic skills composition."
		eos_indicator = "end your composition of semantic skills."

		bos_token = bert_base_forward([bos_indicator])
		bos_token_vec = np.squeeze(bos_token["language_embedding"], axis=0)[0]  # Use [CLS] embedding

		eos_token = bert_base_forward([eos_indicator])
		eos_token_vec = np.squeeze(eos_token["language_embedding"], axis=0)[0]  # Use [CLS] embedding

		max_skill_index = max([token[0].index for token in self.tokens.values()])

		self.bos_token = LanguageRepresentation(
			title="begin",
			variation=bos_indicator,
			vec=bos_token_vec,
			index=max_skill_index + 1
		)
		self.eos_token = LanguageRepresentation(
			title="end",
			variation=eos_indicator,
			vec=eos_token_vec,
			index=max_skill_index + 2
		)

		# Why 'example'?: each skill can have language variations. But we use only one.
		example_vocabulary = [sk[0] for sk in list(self.tokens.values())]
		example_vocabulary.extend([self.bos_token, self.eos_token])
		example_vocabulary.sort(key=lambda sk: sk.index)
		self.vocabulary = example_vocabulary

	@property
	def model(self):
		return self.__model

	@model.setter
	def model(self, model):
		self.__model = model

	def build_model(self):
		transformer_cfg = self.cfg["transformer_cfg"]
		decoder_maxlen = transformer_cfg["decoder_cfg"]["max_len"]

		# Predict discrete token. +2 for start-end tokens.
		vocab_size = len(self.vocabulary)
		transformer_cfg.update({"n_skills": vocab_size})

		transformer = PrimSkillCompositionTransformer(**transformer_cfg)
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

	def update(self, replay_data: ComDeBufferSample, **kwargs) -> Dict:
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
			"n_source_skills": replay_data.n_source_skills,
			"n_target_skills": replay_data.n_target_skills,
			"bos_token": self.bos_token.vec,
			"eos_token_idx": self.eos_token.index
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
		get_prediction_results: bool = True,
		stochastic_sampling: bool = False,
	):
		tokens_vec = np.array([vocab.vec for vocab in self.vocabulary])

		_tokens_vec = tokens_vec[np.newaxis, ...]  # [1, M, d]
		pred_skills = pred_skills[:, np.newaxis, ...]  # [b, 1, d]

		if stochastic_sampling:
			pred_skills = jax.random.categorical(self.rng, logits=pred_skills, axis=-1)
			pred_skills = pred_skills[..., np.newaxis]
		else:
			pred_skills = np.argmax(pred_skills, axis=-1)  # [b, 1, 1]

		pred_skills = np.squeeze(pred_skills)
		maybe_done = np.where(pred_skills == self.eos_token.index, 1, 0)  # [b, ]

		if get_prediction_results:
			return maybe_done, pred_skills

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

		tokens_vec = np.array([vocab.vec for vocab in self.vocabulary])
		done = False

		qkv_info = bert_base_forward(language_guidance)

		q = qkv_info["language_embedding"]
		kv = qkv_info["language_embedding"]
		q_mask = qkv_info["attention_mask"]
		kv_mask = qkv_info["attention_mask"]

		batch_size = q.shape[0]
		skill_dim = q.shape[-1]

		x = self.bos_token.vec
		x = np.broadcast_to(x, (batch_size, 1, skill_dim))  # [b, 1, d]

		pred_skills_seq_raw = []
		encoder_attention_weights = None
		decoder_attention_weights = []
		predictions = []
		target_skills_masks = []

		t = 0

		prev_maybe_done = np.zeros((batch_size,))

		while not done:  # Autoregression
			self.rng, tr_predictions = fwd(
				rng=self.rng,
				model=self.model,
				x=x,
				encoder_q=q,
				encoder_kv=kv,
				q_mask=q_mask,
				kv_mask=kv_mask,
			)

			if encoder_attention_weights is None:
				encoder_attention_weights = tr_predictions.pop("encoder_attention_weights")

			decoder_attention_weight = tr_predictions.pop("decoder_attention_weights")
			decoder_attention_weights.append(decoder_attention_weight)

			pred_logits = tr_predictions["pred_logits"][:, -1, ...]  # [b, d]
			maybe_done, pred_skills = self.maybe_done(
				pred_logits,
				get_prediction_results=True,
				stochastic_sampling=stochastic
			)

			# If you once predict done (end token), then we set done for all after that
			maybe_done = np.logical_or(maybe_done, prev_maybe_done)
			target_skills_masks.append(1 - maybe_done)

			prev_maybe_done = maybe_done

			predictions.append(pred_skills)

			pred_skills_seq_raw.append(pred_skills)
			pred_skills_vec = tokens_vec[pred_skills]

			pred_skills_vec = np.expand_dims(pred_skills_vec, axis=1)
			x = np.concatenate((x, pred_skills_vec), axis=1)

			done = np.all(maybe_done) or (len(pred_skills_seq_raw) >= self.decoder_max_len)
			t += 1

		target_skills_masks = np.stack(target_skills_masks, axis=1)
		pred_skills_raw = np.stack(pred_skills_seq_raw, axis=1)

		predictions = einops.rearrange(np.array(predictions), "l b -> b l")  # [b, M]

		ret = {
			"__pred_skills": predictions,
			"__pred_skills_raw": pred_skills_raw,
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

		pred_skills = np.array(prediction["__pred_skills"])
		# pred_skills = einops.rearrange(pred_skills, "l b -> b l")  # [b, M]

		target_skills_idxs = replay_data.target_skills_idxs  # [b, M, d]
		n_target_skills = replay_data.n_target_skills  # [b, M]

		batch_size = target_skills_idxs.shape[0]

		tgt_mask = np.arange(self.decoder_max_len)
		tgt_mask = np.broadcast_to(tgt_mask, (batch_size, self.decoder_max_len))  # [b, M]
		tgt_mask = np.where(tgt_mask < n_target_skills[..., np.newaxis], 1, 0)  # [b, M]

		if pred_skills.shape[-1] < self.decoder_max_len:
			# Predict skill done too early
			pad_len = self.decoder_max_len - pred_skills.shape[-1]
			dummy_skills_idxs = np.zeros((batch_size, pad_len)) - 1
			pred_skills = np.concatenate((pred_skills, dummy_skills_idxs), axis=-1)

		pred_skills = np.where(tgt_mask == 1, pred_skills, -1)
		target_skills_idxs = np.where(tgt_mask == 1, target_skills_idxs, -2)

		match_ratio = np.sum(pred_skills == target_skills_idxs) / np.sum(tgt_mask)
		return {"s2s/match_ratio(%)": match_ratio * 100, **prediction}

	def evaluate(self, replay_data: ComDeBufferSample, visualization: bool = False) -> Dict:
		eval_info = self._evaluate_discrete(replay_data=replay_data)
		self.visualize_attention(language_guidance=replay_data.language_guidance)

		return eval_info

	def _excluded_save_params(self) -> List:
		return SkillCompositionTransformer.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in SkillCompositionTransformer.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return SkillCompositionTransformer.PARAM_COMPONENTS

	def str_to_activation(self, activation_fn: str):
		pass
