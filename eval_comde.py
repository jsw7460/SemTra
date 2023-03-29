import pickle
import random
from typing import Type, Union
from pathlib import Path

import gym
import hydra
from hydra.utils import get_class
from omegaconf import DictConfig

from comde.evaluations import evaluate_bc_batch
from comde.rl.envs import get_batch_env
from comde.rl.envs.utils import get_batch_source_skills

from comde.utils.interfaces.i_savable.i_savable import IJaxSavable


@hydra.main(version_base=None, config_path="config/eval", config_name="eval_base.yaml")
def program(cfg: DictConfig) -> None:
	with open(cfg.pretrained_path, "rb") as f:
		pretrained_cfg = pickle.load(f)

	pretrained_models = dict()
	for module in pretrained_cfg["modules"]:
		module_cls = get_class(pretrained_cfg[module]["_target_"])	# type: Union[type, Type[IJaxSavable]]
		module_instance = module_cls.load(pretrained_cfg["save_paths"][module])
		pretrained_models[module] = module_instance

	with open(cfg.env.eval_tasks_path, "rb") as f:
		tasks_for_eval = pickle.load(f)  # Target tasks
		tasks_for_eval = [t for t in tasks_for_eval if t[0] == "door"]	# 잠시

	with open(cfg.env.source_skills_path, "rb") as f:
		task_to_source_skills = pickle.load(f)  # Task -> Predicted source skills (; Output of Semantic skill encoder)

	with open(pretrained_cfg["env"]["skill_to_vec_path"], "rb") as f:
		skill_to_vec = pickle.load(f)  # skill to vector

	with open(pretrained_cfg["language_guidance_path"], "rb") as f:
		language_guidances = pickle.load(f)
		language_guidances = language_guidances[cfg["language_guidance"]]  # Dict of vectors

	subseq_len = pretrained_cfg["subseq_len"]
	skill_dim = pretrained_cfg["skill_dim"] + pretrained_cfg["intent_dim"]

	env_class = get_class(cfg.env.path)  # type: Union[type, Type[gym.Env]]
	envs = get_batch_env(
		env_class=env_class,
		tasks=tasks_for_eval,
		skill_dim=skill_dim,
		time_limit=cfg.env.timelimit,
		history_len=subseq_len,
		seed=cfg.seed
	)

	source_skills = get_batch_source_skills(
		task_to_source_skills=task_to_source_skills,
		skill_to_vec=skill_to_vec,
		tasks=tasks_for_eval
	)
	# Note: Among language variations (e.g, ["Do sequentially", "In order", ...], select one randomly)
	language_guidance = random.choices(list(language_guidances.values()), k=len(source_skills))

	info = evaluate_bc_batch(
		envs=envs,
		target_skills=source_skills,
		save_results=cfg.save_results,
		use_optimal_next_skill=cfg.use_optimal_next_skill,
		language_guidance=language_guidance,
		**pretrained_models
	)

	if cfg.save_results:
		# 학습 한 모델은 그날 평가할거니깐 학습시킨 날짜로 저장...........
		save_path = Path(cfg.save_prefix) / Path(cfg.date) / Path(cfg.pretrained_suffix)
		save_path.mkdir(parents=True, exist_ok=True)
		with open(save_path / cfg.save_suffix, "wb") as f:
			pickle.dump(info, f)


if __name__ == "__main__":
	program()
