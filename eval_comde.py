import pickle
import random

import numpy as np

random.seed(2)

from copy import deepcopy
from pathlib import Path
from typing import Type, Union

import gym
import hydra
from hydra.utils import get_class
from omegaconf import DictConfig

from comde.evaluations import evaluate_comde_batch
from comde.rl.envs import get_batch_env
from comde.rl.envs.utils import get_batch_source_skills, get_optimal_semantic_skills
from comde.utils.interfaces.i_savable.i_savable import IJaxSavable


@hydra.main(version_base=None, config_path="config/eval", config_name="eval_base.yaml")
def program(cfg: DictConfig) -> None:
	with open(cfg.pretrained_path, "rb") as f:
		pretrained_cfg = pickle.load(f)

	pretrained_models = dict()
	for module in pretrained_cfg["modules"]:
		module_cls = get_class(pretrained_cfg[module]["_target_"])  # type: Union[type, Type[IJaxSavable]]
		module_instance = module_cls.load(f"{pretrained_cfg['save_paths'][module]}_{cfg.step}")
		pretrained_models[module] = module_instance

	# Target tasks
	with open(cfg.env.eval_tasks_path, "rb") as f:
		tasks_for_eval = pickle.load(f)

	# Task -> Predicted source skills (; Output of Semantic skill encoder)
	with open(cfg.env.source_skills_path, "rb") as f:
		task_to_source_skills = pickle.load(f)

	# skill to vector
	with open(pretrained_cfg["env"]["skill_infos_path"], "rb") as f:
		skill_infos = pickle.load(f)

	with open(pretrained_cfg["non_functionalities_path"], "rb") as f:
		non_functionalities = pickle.load(f)
		non_functionalities = random.choice(list(non_functionalities[cfg.non_functionality].values()))

	param_dim = pretrained_cfg["low_policy"]["cfg"]["param_dim"]
	subseq_len = pretrained_cfg["subseq_len"]
	skill_dim = pretrained_cfg["skill_dim"] \
				+ pretrained_cfg["non_functionality_dim"] \
				+ param_dim

	env_class = get_class(cfg.env.path)  # type: Union[type, Type[gym.Env]]
	print("Tasks for eval", tasks_for_eval)
	envs_candidate = get_batch_env(
		env_class=env_class,
		tasks=tasks_for_eval.copy(),
		cfg=pretrained_cfg["env"],
		skill_dim=skill_dim,
		time_limit=cfg.env.timelimit,
		history_len=subseq_len,
		seed=cfg.seed
	)

	source_skills_candidate = get_batch_source_skills(
		task_to_source_skills=task_to_source_skills,
		sequential_requirement=cfg.sequential_requirement,
		skill_infos=skill_infos,
		tasks=deepcopy([env.skill_list for env in envs_candidate])
	)

	envs = []
	source_skills = []
	for env, source_skill in zip(envs_candidate, source_skills_candidate):
		if source_skill is not None:
			envs.append(env)
			source_skills.append(source_skill)
	n_source_skills = [3 for _ in range(len(envs))]

	# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	# Templatizing language instruction is done here
	# .... (Assume optimally extracted for now)

	if cfg.use_optimal_target_skill:
		pretrained_models.pop("seq2seq")
		# optimal_semantic_skills: [n_env, n_skill, d = 1024]
		optimal_semantic_skills, optimal_idxs = get_optimal_semantic_skills(envs=envs, skill_infos=skill_infos)
		with open(cfg.env.template_path, "rb") as f:
			templates = pickle.load(f)

		templates = templates[cfg.sequential_requirement]
		# random.shuffle(templates)
		# templates = templates[:10]
		templates_for_eval = []
		for template in templates:
			if template.non_functionality["name"] == cfg.non_functionality:
				templates_for_eval.append(template)

		if cfg.non_functionality == "speed" or cfg.non_functionality == "wind":
			params_for_skills = []
			for template in templates:

				parameter_dict = template.parameter
				param_for_skill = np.zeros_like(optimal_idxs)

				param_to_check = {k: 0.0 for k in range(7)}	# Kitchen, No wind
				# param_to_check = {1: 25.0, 3: 25.0, 4: 15.0, 6: 3.0}
				if parameter_dict != param_to_check:
					continue

				for skill_idx, parameter in parameter_dict.items():
					param_for_skill = np.where(optimal_idxs == skill_idx, parameter, param_for_skill)

				param_for_skill = np.repeat(param_for_skill[..., np.newaxis], repeats=param_dim, axis=-1)
				params_for_skills.append(param_for_skill)

			params_for_skills = np.stack(params_for_skills, axis=0)
		else:
			raise NotImplementedError("Undefined non functionality")

		non_functionalities = np.expand_dims(non_functionalities, axis=(0, 1))
		non_functionalities = np.broadcast_to(non_functionalities, optimal_semantic_skills.shape)

		semantic_skills = optimal_semantic_skills

	# 1. semantic skills
	# 2. non functionality
	# 3. parameter
	# --- > target skills

	else:
		"""
		"""
		# Here, we have to predict three vectors
		# semantic_skills: [n_envs, n_skills(~4), d]
		# non_functionalities: [n_envs, n_skills(~4), d]
		# params for skills: [N_TEMPLATES, n_envs, n_skills(~4), param_dim(~1)]
		raise NotImplementedError("For now, we only use optimal template.")

	# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	for template, param_for_skill in zip(templates, params_for_skills):
		print("Curren param:", param_for_skill)
		target_skills = np.concatenate((semantic_skills, non_functionalities, param_for_skill), axis=-1)

		info = evaluate_comde_batch(
			envs=envs,
			target_skills=target_skills,
			save_results=cfg.save_results or cfg.save_text,
			use_optimal_next_skill=cfg.use_optimal_next_skill,
			**pretrained_models
		)

		save_path = Path(cfg.save_prefix) / Path(cfg.date) / Path(cfg.pretrained_suffix)
		if cfg.save_text:
			save_path.mkdir(parents=True, exist_ok=True)
			print("=" * 30)
			print(f"Result is saved at {save_path}")
			print(info)
			# with open(save_path / Path(str(cfg.save_suffix)), "wb") as f:

			print(info)

		elif cfg.save_results:
			# 학습 한 모델은 그날 평가할거니깐 학습시킨 날짜로 저장...........
			save_path.mkdir(parents=True, exist_ok=True)
			print("=" * 30)
			print(f"Result is saved at {save_path}")
			with open(save_path / Path(str(cfg.save_suffix)), "wb") as f:
				pickle.dump(info, f)
		exit()


if __name__ == "__main__":
	program()
