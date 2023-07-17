from typing import Dict

import torch as th
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from comde.comde_modules.seq2seq.transformer.semantic_skill_translation import SemanticSkillTranslator
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.common.natural_languages.lang_representation import SkillRepresentation as LanguageRepresentation


class GPT2SkillTranslator(SemanticSkillTranslator):

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		custom_tokens: Dict[str, LanguageRepresentation],
		init_build_model: bool = True
	) -> None:

		self.tokenizer = None
		self.__model = None
		self.optimizer = None
		super(GPT2SkillTranslator, self).__init__(
			seed=seed,
			cfg=cfg,
			custom_tokens=custom_tokens,
			init_build_model=init_build_model
		)

	@property
	def model(self):
		return self.__model

	def build_model(self):
		tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
		if tokenizer.pad_token_id is None:
			tokenizer.pad_token_id = tokenizer.eos_token_id
		self.tokenizer = tokenizer
		self.__model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()  # type: th.nn.Module
		self.optimizer = th.optim.Adam(self.__model.parameters(), lr=self.cfg["lr"])

	def update(self, replay_data: ComDeBufferSample, **kwargs) -> Dict:
		llm_x = replay_data.language_guidance
		llm_y = replay_data.target_skills_str

		llm_inputs = []
		for x, y in zip(llm_x, llm_y):
			llm_inputs.append(x + ": " + y)

		llm_inputs = self.tokenizer(llm_inputs, return_tensors="pt", padding=True).to("cuda:0")
		outputs = self.model(**llm_inputs, labels=llm_inputs["input_ids"])
		loss = outputs.loss

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.n_update += 1

		return {"loss": loss.item()}

	def evaluate(self, replay_data: ComDeBufferSample, visualization: bool = False) -> Dict:
		llm_x = replay_data.language_guidance
		llm_x = llm_x[0] + ": "
		llm_x = self.tokenizer(llm_x, return_tensors="pt", padding=True).to("cuda:0")

		llm_outputs = self.model.generate(llm_x["input_ids"], max_length=60, pad_token_id=self.tokenizer.eos_token_id)
		print("Output:\n" + 100 * '-')
		print(self.tokenizer.decode(llm_outputs[0], skip_special_tokens=True))

		return dict()

	def _evaluate_continuous(self, replay_data: ComDeBufferSample) -> Dict:
		raise NotImplementedError()
