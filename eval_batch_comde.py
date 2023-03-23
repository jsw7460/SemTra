import pickle
import random
from typing import Type
from pathlib import Path

import gym
import hydra
from hydra.utils import get_class
from omegaconf import DictConfig

from comde.evaluations.comde_eval import evaluate_comde_batch
from comde.rl.envs import get_batch_env
from comde.rl.envs.utils import get_batch_source_skills


@hydra.main(version_base=None, config_path="config/eval", config_name="eval_base.yaml")
def program(cfg: DictConfig) -> None:
	with open(cfg.pretrained_path, "rb") as f:
		pretrained_cfg = pickle.load(f)

	low_policy = get_class(pretrained_cfg["low_policy"]["_target_"])
	low_policy = low_policy.load(pretrained_cfg["save_paths"]["low_policy"])

	seq2seq = get_class(pretrained_cfg["seq2seq"]["_target_"])
	seq2seq = seq2seq.load(pretrained_cfg["save_paths"]["seq2seq"])

	termination = get_class(pretrained_cfg["termination"]["_target_"])
	termination = termination.load(pretrained_cfg["save_paths"]["termination"])

	with open(cfg.env.eval_tasks_path, "rb") as f:
		tasks_for_eval = pickle.load(f)  # Target tasks

	with open(cfg.env.source_skills_path, "rb") as f:
		task_to_source_skills = pickle.load(f)  # Task -> Predicted source skills

	with open(pretrained_cfg["env"]["skill_to_vec_path"], "rb") as f:
		skill_to_vec = pickle.load(f)  # skill to vector

	with open(pretrained_cfg["language_guidance_path"], "rb") as f:
		language_guidances = pickle.load(f)
		language_guidances = language_guidances[cfg["language_guidance"]]  # Dict of vectors

	subseq_len = low_policy.cfg["subseq_len"]
	skill_dim = low_policy.cfg["skill_dim"]

	env_class = get_class(cfg.env.path)  # type: Type[gym.Env]
	envs = get_batch_env(
		env_class=env_class,
		tasks=tasks_for_eval,
		skill_dim=skill_dim,
		time_limit=cfg.env.timelimit,
		history_len=subseq_len,
		seed=cfg.seed
	)

	source_skills = get_batch_source_skills(
		task_to_source_skills=task_to_source_skills[cfg["language_guidance"]],
		skill_to_vec=skill_to_vec,
		tasks=tasks_for_eval
	)
	language_guidance = random.choices(list(language_guidances.values()), k=len(source_skills))
	info = evaluate_comde_batch(
		envs=envs,
		low_policy=low_policy,
		seq2seq=seq2seq,
		termination=termination,
		source_skills=source_skills,
		language_guidance=language_guidance,
		save_results=cfg.save_results,
		use_optimal_next_skill=cfg.use_optimal_next_skill,
	)

	if cfg.save_results:
		# 학습 한 모델은 그날 평가할거니깐 학습시킨 날짜로 저장...........
		save_path = Path(cfg.save_prefix) / Path(cfg.date) / Path(cfg.pretrained_suffix)
		save_path.mkdir(parents=True, exist_ok=True)
		with open(save_path / "", "wb") as f:
			pickle.dump(info, f)


if __name__ == "__main__":
	program()
