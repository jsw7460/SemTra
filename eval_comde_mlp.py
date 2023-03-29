import pickle
import random

import hydra
from hydra.utils import get_class
from omegaconf import DictConfig
from pathlib import Path

from comde.evaluations.comde_eval_mlp import evaluate_comde
from comde.rl.envs.utils import BatchEnv, TimeLimitEnv, get_source_skills


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
		tasks_for_eval = pickle.load(f)		# Target tasks

	with open(cfg.env.source_skills_path, "rb") as f:
		task_to_source_skills = pickle.load(f)	# Task -> Predicted source skills

	with open(pretrained_cfg["env"]["skill_to_vec_path"], "rb") as f:
		skill_to_vec = pickle.load(f)	# skill to vector

	with open(pretrained_cfg["language_guidance_path"], "rb") as f:
		language_guidances = pickle.load(f)
		language_guidances = language_guidances[cfg["language_guidance"]]	# Dict of vectors

	for task in tasks_for_eval:

		env = get_class(cfg.env.path)
		env = env(seed=cfg.seed, task=task)  # Set task
		env = TimeLimitEnv(env=env, limit=cfg.env.timelimit)
		env = BatchEnv(env)

		source_skills = get_source_skills(
			task_to_source_skills=task_to_source_skills[cfg["language_guidance"]],
			skill_to_vec=skill_to_vec,
			task=task
		)
		language_guidance = random.choice(list(language_guidances.values()))

		with env.batch_mode():  # expand first dimension
			info = evaluate_comde(
				env=env,
				low_policy=low_policy,
				seq2seq=seq2seq,
				termination=termination,
				source_skills=source_skills,
				language_guidance=language_guidance
			)

		if cfg.save_results:
			save_path = Path(cfg.save_prefix) / Path(cfg.date) / Path(cfg.pretrained_suffix)
			save_path.mkdir(parents=True, exist_ok=True)
			with open(save_path / Path(env.get_short_str_for_save() + str(info["return"])), "wb") as f:
				pickle.dump(info, f)


if __name__ == "__main__":
	program()
