import random
from typing import Dict, List, Type

from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.rl.envs.base import ComdeSkillEnv
from comde.trainer.base import BaseTrainer
from comde.utils.common.pretrained_forwards.jax_bert_base import bert_base_forward
from comde.utils.common.prompt_templates import prompt_templates


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
			for i in range(64):
				language_guidance, info = env.generate_random_language_guidance()
				nf = info["non_functionality"]
				skill = info["param_applied_skill"]
				param = info["parameter"]
				prompt = random.choice(prompt_templates)
				prompt = prompt.format(nf=nf, skill=skill, param=param)
				target_output = " ".join([nf, skill, param])
				prompts.append(prompt)
				target_inputs.append(language_guidance)
				target_outputs.append(target_output)

		all_examples = [pr + ": " + ti for (pr, ti) in zip(prompts, target_inputs)]
		examples = []
		for _ in range(len(target_inputs)):
			example = random.choices(all_examples, k=self.n_examples)
			example = " ".join(example)
			examples.append(example)

		model_inputs = [ex + " " + ti for (ex, ti) in zip(examples, target_inputs)]

		# Shuffle the data
		combined = list(zip(model_inputs, target_outputs))
		random.shuffle(combined)

		model_inputs, target_outputs = zip(*combined)
		model_inputs = list(model_inputs)
		target_outputs = list(target_outputs)

		label = bert_base_forward(target_outputs)
		dec_idxs = label["input_ids"]
		dec_masks = label["attention_mask"]

		info = {
			"model_inputs": model_inputs,
			"target_outputs": target_outputs,
			"dec_idxs": dec_idxs,
			"dec_masks": dec_masks
		}

		return info

	def run(self, *args, **kwargs):
		dataset = self._make_batch_training_data()
		model_inputs = dataset["model_inputs"]
		decoder_idxs = dataset["dec_idxs"]
		decoder_masks = dataset["dec_masks"]
		self.prompt_learner.update(
			model_inputs=model_inputs,
			decoder_idxs=decoder_idxs,
			decoder_masks=decoder_masks
		)

	def dump_logs(self, step: int):
		self.record(self.info_records)
		super(PromptTrainer, self).dump_logs(step=step)

	def save(self):
		for key, save_path in self.cfg["save_paths"].items():
			cur_step = str(self.n_update)
			getattr(self, key).save(f"{save_path}_{cur_step}")

	def load(self, *args, **kwargs):
		raise NotImplementedError()
