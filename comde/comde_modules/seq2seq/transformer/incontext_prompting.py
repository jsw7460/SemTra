import random
from collections import deque
from typing import Dict, List, Optional, Union

import jax.random
import numpy as np
import optax
from transformers import AutoTokenizer

from comde.comde_modules.seq2seq.algos.forward import skilltoskill_transformer_forward as prompt_transformer_forward
from comde.comde_modules.seq2seq.algos.updates.promptlearning_transformer import promptlearning_transformer_updt as updt
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.seq2seq.transformer.architectures.incontext_transformer import PrimIncontextTransformer
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.common import pretrained_forwards
from comde.utils.common.natural_languages.lang_representation import SkillRepresentation as LanguageRepresentation
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class IncontextTransformer(BaseSeqToSeq):
	PARAM_COMPONENTS = ["_IncontextTransformer__model"]

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		# self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
		self.language_space = cfg.get("language_space", "bert")
		self.tokenizer = getattr(pretrained_forwards, self.language_space + "_tokenizer")

		if "bert" in self.language_space:
			self.bos_token_id = self.tokenizer.cls_token_id
			self.eos_token_id = self.tokenizer.sep_token_id
		elif "t5" in self.language_space:
			self.bos_token_id = self.tokenizer.unk_token_id
			self.eos_token_id = self.tokenizer.eos_token_id

		elif "clip" in self.language_space:
			self.bos_token_id = self.tokenizer.bos_token_id
			self.eos_token_id = self.tokenizer.eos_token_id

		self._language_encoder_forward = getattr(pretrained_forwards, self.language_space)

		super(IncontextTransformer, self).__init__(
			seed=seed,
			cfg=cfg,
			init_build_model=init_build_model
		)
		# Source skills: [b, l, d], Target skills: [b, l, d]
		self.__model = None
		self.save_suffix = self.cfg["save_suffix"]
		self.example_input_conjunction = " Then, extract from this: "
		self.example_storage = {
			"speed": deque(maxlen=100),
			"wind": deque(maxlen=100),
			"weight": deque(maxlen=100)
		}

		if init_build_model:
			self.build_model()

	@property
	def model(self):
		return self.__model

	@model.setter
	def model(self, model):
		self.__model = model

	def register_vocabulary(self):

		bos_indicator = "extract the non-functionality and parameter"
		eos_indicator = "end"
		bos_token = self._language_encoder_forward([bos_indicator])
		bos_token_vec = np.squeeze(bos_token["language_embedding"], axis=0)[0]  # Use [CLS] embedding
		eos_token = self._language_encoder_forward([eos_indicator])
		eos_token_vec = np.squeeze(eos_token["language_embedding"], axis=0)[0]  # Use [CLS] embedding

		self.bos_token = LanguageRepresentation(
			title="begin",
			variation=bos_indicator,
			vec=bos_token_vec,
			index=self.bos_token_id
		)
		self.eos_token = LanguageRepresentation(
			title="end",
			variation=eos_indicator,
			vec=eos_token_vec,
			index=self.eos_token_id
		)

	def build_model(self):
		transformer_cfg = self.cfg["transformer_cfg"]
		decoder_maxlen = transformer_cfg["decoder_cfg"]["max_len"]

		# Predict discrete token. +2 for start-end tokens.
		vocab_size = self.tokenizer.vocab_size
		transformer_cfg.update({"vocab_size": vocab_size})  # We don't predict start token.

		transformer = PrimIncontextTransformer(**transformer_cfg)
		init_x = np.zeros((1, decoder_maxlen), dtype="i4")  # Decoder input index
		init_q = np.zeros((1, 7, self.inseq_dim))  # Encoder input is language embedding
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

	def update(
		self,
		examples: List[str],
		target_inputs: List[str],
		target_outputs: List[str]
	) -> Dict:
		# Shuffle the data

		for ex in examples:
			if "speed" in ex:
				self.example_storage["speed"].append(ex)
			elif "wind" in ex:
				self.example_storage["wind"].append(ex)
			elif "weight" in ex:
				self.example_storage["weight"].append(ex)
			else:
				raise NotImplementedError(f"{ex} has no appropriate non functionality")

		examples = []
		for inp in target_inputs:
			if "speed" in inp:
				example = random.choice(self.example_storage["speed"])
			elif "wind" in inp:
				example = random.choice(self.example_storage["wind"])
			elif "weight" in inp:
				example = random.choice(self.example_storage["weight"])
			else:
				raise NotImplementedError(f"{inp} has no non-functionality")
			examples.append(example)

		model_inputs = [ex + self.example_input_conjunction + ti for (ex, ti) in zip(examples, target_inputs)]
		combined = list(zip(model_inputs, target_outputs))
		random.shuffle(combined)

		model_inputs, target_outputs = zip(*combined)
		model_inputs = list(model_inputs)
		target_outputs = list(target_outputs)

		label = self.tokenizer(target_outputs, return_tensors='np', padding=True)
		decoder_idxs = label["input_ids"]
		decoder_masks = label["attention_mask"]

		qkv_info = self._language_encoder_forward(model_inputs)
		q = qkv_info["language_embedding"]
		kv = qkv_info["language_embedding"]
		q_mask = qkv_info["attention_mask"]
		kv_mask = qkv_info["attention_mask"]

		if self.n_update % 30 == 0:
			self.predict(target_inputs=target_inputs[:2])

		new_model, info = updt(
			rng=self.rng,
			tr=self.model,
			encoder_q=q,
			encoder_kv=kv,
			q_mask=q_mask,
			kv_mask=kv_mask,
			decoder_idxs=decoder_idxs,
			decoder_masks=decoder_masks
		)
		self.n_update += 1
		self.model = new_model
		self.rng, _ = jax.random.split(self.rng)

		return info

	def predict(
		self,
		target_inputs: List[str],
		shuffle: bool = True,
		examples: Optional[List[str]] = None,
		stochastic: bool = False,
		return_qkv_info: bool = False,
		skip_special_tokens: bool = False,
		parse: bool = False
	) -> Union[List[str], List[Dict]]:
		if examples is None:
			examples = []
			for inp in target_inputs:
				if "speed" in inp:
					example = random.choice(self.example_storage["speed"])
				elif "wind" in inp:
					example = random.choice(self.example_storage["wind"])
				elif "weight" in inp:
					example = random.choice(self.example_storage["weight"])
				else:
					raise NotImplementedError(f"{inp} has no non-functionality")
				examples.append(example)

		model_inputs = [ex + self.example_input_conjunction + ti for (ex, ti) in zip(examples, target_inputs)]

		if shuffle:
			random.shuffle(model_inputs)

		qkv_info = self._language_encoder_forward(model_inputs)

		q = qkv_info["language_embedding"]
		kv = qkv_info["language_embedding"]
		q_mask = qkv_info["attention_mask"]
		kv_mask = qkv_info["attention_mask"]

		batch_size = q.shape[0]
		x = self.bos_token_id
		x = np.broadcast_to(x, (batch_size, 1))  # [b, 1]

		predictions = []

		while len(predictions) < self.decoder_max_len:
			self.rng, pred = prompt_transformer_forward(
				rng=self.rng,
				model=self.model,
				x=x,
				encoder_q=q,
				encoder_kv=kv,
				q_mask=q_mask,
				kv_mask=kv_mask
			)  # [b, l, vocab_size]
			pred = pred["pred"]
			new_token = np.argmax(pred[:, -1, ...], axis=-1, keepdims=True)
			predictions.append(new_token)
			x = np.concatenate((x, new_token), axis=-1)

		predictions = np.concatenate(predictions, axis=-1)
		predictions = [pred.tolist() for pred in predictions]
		lang_gen = self.tokenizer.batch_decode(predictions, skip_special_tokens=skip_special_tokens)  # type: List[str]

		# for i in range(2):
		# 	print("Input", target_inputs[i])
		# 	print("Pred", lang_gen[i])

		if parse:
			return IncontextTransformer.batch_parse(lang_gen)

		return lang_gen

	@staticmethod
	def parse(lang: str) -> Dict:
		parts = lang.split(" ")
		parse_dict = {"non_functionality": None, "skill": None, "param": None}
		for key, part in zip(parse_dict.keys(), parts):
			parse_dict[key] = part
		return parse_dict

	@staticmethod
	def batch_parse(langs: List[str]) -> List[Dict]:
		return [IncontextTransformer.parse(lang) for lang in langs]

	def evaluate(self, replay_data: ComDeBufferSample, visualization: bool = False) -> Dict:
		"""..."""

	def _excluded_save_params(self) -> List:
		return IncontextTransformer.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in IncontextTransformer.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return IncontextTransformer.PARAM_COMPONENTS

	def str_to_activation(self, activation_fn: str):
		pass
