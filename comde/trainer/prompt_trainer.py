import random
from typing import Dict, List, Type

from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.rl.envs.base import ComdeSkillEnv
from comde.trainer.base import BaseTrainer
from comde.utils.common.natural_languages.prompt_templates import prompt_templates
from comde.utils.common.natural_languages.language_processing import number_to_words


class PromptTrainer(BaseTrainer):

	def __init__(
		self,
		cfg: Dict,
		envs: List[Type[ComdeSkillEnv]],
		prompt_learner: BaseSeqToSeq
	):
		super(PromptTrainer, self).__init__(cfg=cfg, env=envs)
		self.envs = envs
		del self.env
		self.prompt_learner = prompt_learner
		self.info_records = {"info/suffix": self.cfg["save_suffix"]}
		self.n_examples = cfg["prompt_learner"]["cfg"]["n_example"]
		self.__last_onehot_skills = None

	def _make_batch_training_data(self):
		prompts = []
		target_inputs = []
		target_outputs = []

		for env in self.envs:
			it = len(env.non_functionalities)
			for i in range(16 * it):
				language_guidance, info = env.generate_random_language_guidance(video_parsing=True)
				nf = info["non_functionality"]
				skill = info["param_applied_skill"]
				param = info["parameter"]
				if param.lstrip('-').isdigit():
					param = number_to_words(int(param))
				prompt = random.choice(prompt_templates)
				prompt = prompt.format(nf=nf, skill=skill, param=param)
				target_output = ComdeSkillEnv.ingradients_to_target(
					non_functionality=nf,
					skill=skill,
					param=param
				)
				prompts.append(prompt)
				target_inputs.append(language_guidance)
				target_outputs.append(target_output)

		all_examples = [pr + ": " + ti for (pr, ti) in zip(prompts, target_inputs)]
		examples = []
		for _ in range(len(target_inputs)):
			example = random.choices(all_examples, k=self.n_examples)
			example = " ".join(example)
			examples.append(example)
		info = {
			"examples": examples,
			"target_inputs": target_inputs,
			"target_outputs": target_outputs
		}

		return info

	def run(self, *args, **kwargs):
		dataset = self._make_batch_training_data()
		examples = dataset["examples"]
		target_inputs = dataset["target_inputs"]
		target_outputs = dataset["target_outputs"]

		info = self.prompt_learner.update(
			examples=examples,
			target_inputs=target_inputs,
			target_outputs=target_outputs
		)
		self.record_from_dicts(info, mode="train")
		self.n_update += 1

		if (self.n_update % 100) == 0:
			self.evaluate()
		if (self.n_update % self.log_interval) == 0:
			self.dump_logs(step=self.n_update)
		if (self.n_update % self.save_interval) == 0:
			self.save()

	def _gpt2_evaluate(self):
		dataset = self._make_batch_training_data()
		examples = dataset["examples"]
		target_inputs = dataset["target_inputs"]
		target_outputs = dataset["target_outputs"]
		self.prompt_learner.evaluate(
			examples=examples,
			target_inputs=target_inputs,
			target_outputs=target_outputs
		)

	def evaluate(self) -> None:
		self._gpt2_evaluate()
		return None


	def dump_logs(self, step: int):
		self.record(self.info_records)
		super(PromptTrainer, self).dump_logs(step=step)

	def save(self):
		for key, save_path in self.cfg["save_paths"].items():
			cur_step = str(self.n_update)
			getattr(self, key).save(f"{save_path}_{cur_step}")

	def load(self, *args, **kwargs):
		raise NotImplementedError()
