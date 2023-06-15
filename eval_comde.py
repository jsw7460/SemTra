import pickle
import random

random.seed(7)

from copy import deepcopy
from pathlib import Path
from typing import Type, Union

import gym
import hydra
from hydra.utils import get_class
from omegaconf import DictConfig
import numpy as np

from comde.evaluations.utils.optimal_template import get_optimal_template
from comde.evaluations.utils.get_arguments import get_evaluation_function
from comde.rl.envs import get_batch_env
from comde.rl.envs.utils.get_source import get_batch_source_skills
from comde.utils.interfaces.i_savable.i_savable import IJaxSavable
from comde.evaluations.utils.dump_evaluations import dump_eval_logs


@hydra.main(version_base=None, config_path="config/eval", config_name="eval_base.yaml")
def program(cfg: DictConfig) -> None:
	with open(cfg.pretrained_path, "rb") as f:
		pretrained_cfg = pickle.load(f)

	pretrained_models = dict()
	if "seq2seq" in pretrained_cfg["modules"]:
		pretrained_cfg["modules"].remove("seq2seq")

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

	param_dim = pretrained_cfg["low_policy"]["cfg"]["param_dim"]
	param_repeats = pretrained_cfg["low_policy"]["cfg"].get("param_repeats", 100)
	total_param_dim = param_dim * param_repeats
	subseq_len = pretrained_cfg["subseq_len"]

	env_class = get_class(cfg.env.path)  # type: Union[type, Type[gym.Env]]
	envs_candidate = get_batch_env(
		env_class=env_class,
		tasks=tasks_for_eval.copy(),
		n_target=cfg.env.n_target,
		cfg={**pretrained_cfg["env"], **cfg.env},
		skill_dim=pretrained_cfg["skill_dim"] + pretrained_cfg["non_functionality_dim"] + total_param_dim,
		time_limit=cfg.env.timelimit,
		history_len=subseq_len,
		seed=cfg.seed
	)

	non_functionalities = envs_candidate[0].non_functionalities_vector_mapping
	non_functionalities = random.choice(list(non_functionalities[cfg.non_functionality].values()))
	skill_infos = envs_candidate[0].skill_infos

	source_skills_dict = get_batch_source_skills(
		task_to_source_skills=task_to_source_skills,
		sequential_requirement=cfg.sequential_requirement,
		skill_infos=skill_infos,
		tasks=deepcopy([env.skill_list[:cfg.env.n_target] for env in envs_candidate]),
	)
	source_skills_vec_candidate = source_skills_dict["np_source_skills"]
	source_skills_idx_candidate = source_skills_dict["source_skill_idxs"]

	envs = []
	source_skills_vec = []
	source_skills_idx = []
	for env, vec, idx in zip(envs_candidate, source_skills_vec_candidate, source_skills_idx_candidate):
		if vec is not None:
			envs.append(env)
			source_skills_vec.append(vec)
			source_skills_idx.append(idx)

	"""
		Templatizing language instruction is done here
		.... (Assume optimally extracted for now)
	"""
	optimal_template = get_optimal_template(
		cfg=cfg,
		envs=envs,
		skill_infos=skill_infos,
		non_functionalities=non_functionalities,
		param_repeats=param_repeats
	)
	if cfg.use_optimal_target_skill:
		"""
			semantic_skills_sequence: [n_env, n_target_skills, skill_dim]
			non_functionalities: [n_env, n_target_skills, nonfunctionality_dim]
			params_for_skills: ['M', n_env, n_target_skills, param_dim] 
				'M': The number of parameters to be evaluated. (wind 0.2, 0.3, -0.2, -0.3, ...)  
		"""
		semantic_skills_sequence = optimal_template["semantic_skills_sequence"]
		non_functionalities = optimal_template["non_functionalities"]
		params_for_skills = optimal_template["params_for_skills"]
		str_skill_pred_accuracy = "100% (Optimal)"

	# 1. semantic skills
	# 2. non functionality
	# 3. parameter
	# --- > target skills

	else:
		"""
		"""
		language_guidances = []
		for t, env in enumerate(envs):
			language_guidance = env.get_language_guidance_from_template(
				sequential_requirement=cfg.sequential_requirement,
				non_functionality=cfg.non_functionality,
				source_skills_idx=source_skills_idx[t],
				parameter=None
			)
			language_guidances.append(language_guidance)

		seq2seq_info = pretrained_models["seq2seq"].predict(language_guidances)
		target_skills_idxs = seq2seq_info["__pred_target_skills"]
		target_skills = []
		for target_skill_idx, env in zip(target_skills_idxs, envs):
			target_skill_vec = env.get_skill_vectors_from_idx_list(target_skill_idx.tolist())
			target_skills.append(target_skill_vec)

		target_skills = np.array(target_skills)
		optimal_target_skills = optimal_template["optimal_target_skill_idxs"]

		_target_skills_idxs = target_skills_idxs[:, :optimal_target_skills.shape[-1]]

		skill_pred_accuracy = np.mean(optimal_target_skills == _target_skills_idxs)
		str_skill_pred_accuracy = f"{skill_pred_accuracy * 100} %"

	n_eval = cfg.n_eval

	text_path_dir = Path(cfg.text_save_prefix) / Path(cfg.date) / Path(cfg.pretrained_suffix)
	text_path_dir.mkdir(parents=True, exist_ok=True)
	text_path = text_path_dir / Path(f"{cfg.save_suffix}.txt")

	returns_mean = 0.0

	for n_trial, (_seed, param_for_skill) in enumerate(zip(range(n_eval), params_for_skills)):

		evaluation, _info = get_evaluation_function(locals(), custom_seed=_seed)
		info, eval_fmt = evaluation()
		info.update(**_info)

		eval_str = "\n" \
				   + "=" * 30 + "\n" \
				   + f"seq_req: {_info['sequential_requirement']}, seed: {_seed}, step: {cfg.step}\n" \
				   + f"skill prediction: {str_skill_pred_accuracy}\n" \
				   + eval_fmt

		dump_eval_logs(save_path=text_path, eval_str=eval_str)
		save_path_dir = Path(cfg.save_prefix) / Path(cfg.date) / Path(cfg.pretrained_suffix)

		if cfg.save_results:
			save_path = save_path_dir / Path(f"{cfg.save_suffix}_{n_trial}")
			# 학습 한 모델은 그날 평가할거니깐 학습시킨 날짜로 저장...........
			save_path_dir.mkdir(parents=True, exist_ok=True)
			print("=" * 30)
			print(f"Result is saved at {save_path}")
			with open(save_path, "wb") as f:
				pickle.dump(info, f)


if __name__ == "__main__":
	program()
