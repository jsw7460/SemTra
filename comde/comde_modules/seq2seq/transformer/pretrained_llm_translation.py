from typing import Dict, List, Union, Tuple
import numpy as np

import torch as th
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

from comde.comde_modules.seq2seq.transformer.semantic_skill_translation import SemanticSkillTranslator
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.common.natural_languages.lang_representation import SkillRepresentation as LanguageRepresentation
from comde.utils.save_utils.common import recursive_getattr


class GPT2SkillTranslator(SemanticSkillTranslator):
	PARAM_COMPONENTS = ["_GPT2SkillTranslator__model"]

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		custom_tokens: Dict[str, LanguageRepresentation],
		init_build_model: bool = True
	) -> None:


		self.tokenizer = None	# type: GPT2Tokenizer
		self.__model = None
		self.optimizer = None
		self.conjunction = " Then the skill sequence is: "
		super(GPT2SkillTranslator, self).__init__(
			seed=seed,
			cfg=cfg,
			custom_tokens=custom_tokens,
			init_build_model=init_build_model
		)

	@property
	def model(self) -> th.nn.Module:
		return self.__model

	def build_model(self):
		# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
		model_name = self.cfg["language_model"]
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		if tokenizer.pad_token_id is None:
			tokenizer.pad_token_id = tokenizer.eos_token_id
		self.tokenizer = tokenizer
		# self.__model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()  # type: th.nn.Module
		self.__model = AutoModelForCausalLM.from_pretrained(model_name).cuda()  # type: th.nn.Module
		self.optimizer = th.optim.Adam(self.__model.parameters(), lr=self.cfg["lr"])

		param_size = 0
		for param in self.__model.parameters():
			param_size += param.nelement() * param.element_size()
		buffer_size = 0
		for buffer in self.__model.buffers():
			buffer_size += buffer.nelement() * buffer.element_size()

		size_all_mb = (param_size + buffer_size) / 1024 ** 2

		n_param = sum(p.numel() for p in self.__model.parameters() if p.requires_grad)

		print("\n\n\n\n")
		print('model size: {:.3f}MB'.format(size_all_mb))
		print('N param:', n_param)
		print("\n\n\n\n")

	def update(self, replay_data: ComDeBufferSample, **kwargs) -> Dict:
		llm_x = replay_data.language_guidance
		llm_y = replay_data.target_skills_str

		llm_inputs = []
		for x, y in zip(llm_x, llm_y):
			llm_input = x + self.conjunction + y + "." + self.tokenizer.eos_token
			llm_inputs.append(llm_input)

		llm_inputs = self.tokenizer(llm_inputs, return_tensors="pt", padding=True).to("cuda:0")
		prompt_maskings = self.tokenizer(replay_data.language_guidance, return_tensors="pt", padding=True).to("cuda:0")
		prompt_maskings = prompt_maskings["attention_mask"]

		labels = th.clone(llm_inputs["input_ids"])
		attention_mask = llm_inputs["attention_mask"]

		labels[~attention_mask.bool()] = -100
		labels[prompt_maskings] = -100

		outputs = self.model(**llm_inputs, labels=labels)
		loss = outputs.loss

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.n_update += 1

		if self.n_update > 60000:
			exit()

		return {"loss": loss.item()}

	def evaluate(self, replay_data: ComDeBufferSample, visualization: bool = False) -> Dict:

		accuracy = 0.0

		for guidance, target in zip(replay_data.language_guidance, replay_data.target_skills_str):
			llm_x = guidance + self.conjunction.rstrip(" ")
			llm_x = self.tokenizer(llm_x, return_tensors="pt").to("cuda:0")

			llm_outputs = self.model.generate(
				llm_x["input_ids"],
				max_length=80,
				pad_token_id=self.tokenizer.eos_token_id
			)

			decoded = self.tokenizer.decode(llm_outputs[0], skip_special_tokens=True)
			decoded = decoded.replace(guidance + self.conjunction, "")
			decoded = decoded.rstrip(".")

			decoded_tokens = decoded.split(", ")
			target_tokens = target.split(", ")

			n_first_match = sum([d == t for (d, t) in zip(decoded_tokens, target_tokens)])
			first_match_ratio = (n_first_match / min([len(decoded_tokens), len(target_tokens)])) * 100

			accuracy += first_match_ratio

		accuracy /= len(replay_data.language_guidance)
		ret_info = {"accuracy": accuracy}
		print("Language model:", self.cfg["language_model"])

		return ret_info

	def predict(
		self,
		language_guidance: List[str],
		stochastic: bool = False,
		return_qkv_info: bool = False,
		fix_offset: bool = False,
	) -> Union[Tuple, Dict]:
		llm_x = language_guidance[0] + self.conjunction.rstrip(" ")
		llm_x = self.tokenizer(llm_x, return_tensors="pt").to("cuda:0")

		llm_outputs = self.model.generate(llm_x["input_ids"], max_length=80, pad_token_id=self.tokenizer.eos_token_id)
		ret = self.tokenizer.decode(llm_outputs[0], skip_special_tokens=True)
		return ret

	def _evaluate_continuous(self, replay_data: ComDeBufferSample) -> Dict:
		raise NotImplementedError()

	def _excluded_save_params(self) -> List:
		return GPT2SkillTranslator.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict:
		params_dict = {}
		for component_str in GPT2SkillTranslator.PARAM_COMPONENTS:
			attr = recursive_getattr(self, component_str)
			params_dict[component_str] = {"state_dict": attr.state_dict()}
		# params_dict[component_str] = attr.state_dict()
		return params_dict

	def _get_load_params(self) -> List[str]:
		return GPT2SkillTranslator.PARAM_COMPONENTS
