import pickle

import hydra
import numpy as np
from hydra.utils import get_class
from omegaconf import DictConfig

from comde.rl.envs.utils import BatchEnv, TimeLimitEnv
from comde.evaluations.comde_eval import evaluate_comde


@hydra.main(version_base=None, config_path="config/eval", config_name="eval_base.yaml")
def program(cfg: DictConfig) -> None:
	with open(cfg.pretrained_path, "rb") as f:
		pretrained_cfg = pickle.load(f)

	# source_skills = tasks_for_eval["source_skills"]
	# language_guidence = tasks_for_eval["language_guidence"]

	source_skills = np.zeros((1, 7, 512))
	language_guidence = np.zeros((1, 512))

	low_policy = get_class(pretrained_cfg["low_policy"]["_target_"])
	low_policy = low_policy.load(pretrained_cfg["save_paths"]["low_policy"])

	seq2seq = get_class(pretrained_cfg["seq2seq"]["_target_"])
	seq2seq = seq2seq.load(pretrained_cfg["save_paths"]["seq2seq"])

	termination = get_class(pretrained_cfg["termination"]["_target_"])
	termination = termination.load(pretrained_cfg["save_paths"]["termination"])

	with open(cfg.env.eval_tasks_path, "rb") as f:
		tasks_for_eval = pickle.load(f)

	for task in tasks_for_eval:
		env = get_class(cfg.env.path)
		env = env(seed=cfg.seed, task=task)  # Set task
		env = TimeLimitEnv(env=env, limit=cfg.env.timelimit)
		env = BatchEnv(env)
		with env.batch_mode():  # expand first dimension
			evaluate_comde(
				env=env,
				low_policy=low_policy,
				seq2seq=seq2seq,
				termination=termination,
				source_skills=source_skills,
				language_guidence=language_guidence
			)

if __name__ == "__main__":
	program()
