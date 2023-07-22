from typing import Dict, List

import torch as th
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from comde.comde_modules.seq2seq.transformer.incontext_prompting import IncontextTransformer
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.save_utils.common import recursive_getattr


class GPT2PromptingTransformer(IncontextTransformer):
	PARAM_COMPONENTS = ["_GPT2PromptingTransformer__model"]

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool = True
	) -> None:

		self.tokenizer = None
		self.__model = None
		self.optimizer = None
		self.conjunction = " Then the non-functionalities are: "
		super(GPT2PromptingTransformer, self).__init__(
			seed=seed,
			cfg=cfg,
			init_build_model=init_build_model
		)

	def register_vocabulary(self):
		pass

	@property
	def model(self) -> th.nn.Module:
		return self.__model

	def build_model(self):
		tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
		if tokenizer.pad_token_id is None:
			tokenizer.pad_token_id = tokenizer.eos_token_id
		self.tokenizer = tokenizer
		self.__model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()  # type: th.nn.Module
		self.optimizer = th.optim.Adam(self.__model.parameters(), lr=self.cfg["lr"])

	def update(
		self,
		examples: List[str] = None,
		target_inputs: List[str] = None,
		target_outputs: List[str] = None,
		**kwargs
	) -> Dict:
		llm_x = target_inputs
		llm_y = target_outputs

		llm_inputs = []
		for x, y in zip(llm_x, llm_y):
			llm_input = x + self.conjunction + y + "." + self.tokenizer.eos_token
			llm_inputs.append(llm_input)

		llm_inputs = self.tokenizer(llm_inputs, return_tensors="pt", padding=True).to("cuda:0")
		prompt_maskings = self.tokenizer(target_inputs, return_tensors="pt", padding=True).to("cuda:0")
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
		return {"loss": loss.item()}

	def evaluate(
		self,
		examples: List[str] = None,
		target_inputs: List[str] = None,
		target_outputs: List[str] = None,
		**kwargs
	) -> Dict:
		llm_x = target_inputs
		llm_x = llm_x[0] + self.conjunction.rstrip(" ")
		llm_x = self.tokenizer(llm_x, return_tensors="pt").to("cuda:0")

		llm_outputs = self.model.generate(llm_x["input_ids"], max_length=80, pad_token_id=self.tokenizer.eos_token_id)
		print("Output:\n" + 100 * '-')
		print(self.tokenizer.decode(llm_outputs[0], skip_special_tokens=True))

		return dict()

	def _evaluate_continuous(self, replay_data: ComDeBufferSample) -> Dict:
		raise NotImplementedError()

	def _excluded_save_params(self) -> List:
		return GPT2PromptingTransformer.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict:
		params_dict = {}
		for component_str in GPT2PromptingTransformer.PARAM_COMPONENTS:
			attr = recursive_getattr(self, component_str)
			params_dict[component_str] = {"state_dict": attr.state_dict()}
		# params_dict[component_str] = attr.state_dict()
		return params_dict

	def _get_load_params(self) -> List[str]:
		return GPT2PromptingTransformer.PARAM_COMPONENTS
