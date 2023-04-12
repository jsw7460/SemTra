import pickle
import random

random.seed(777)

from copy import deepcopy
from pathlib import Path
from typing import Type, Union

import gym
import hydra
from hydra.utils import get_class
from omegaconf import DictConfig

from comde.evaluations import evaluate_comde_batch
from comde.rl.envs import get_batch_env
from comde.rl.envs.utils import get_batch_source_skills
from comde.utils.interfaces.i_savable.i_savable import IJaxSavable


@hydra.main(version_base=None, config_path="config/eval", config_name="eval_base.yaml")
def program(cfg: DictConfig) -> None:
	with open(cfg.pretrained_path, "rb") as f:
		pretrained_cfg = pickle.load(f)

	pretrained_models = dict()
	for module in pretrained_cfg["modules"]:
		module_cls = get_class(pretrained_cfg[module]["_target_"])  # type: Union[type, Type[IJaxSavable]]
		module_instance = module_cls.load(pretrained_cfg["save_paths"][module])
		pretrained_models[module] = module_instance

	with open(cfg.env.eval_tasks_path, "rb") as f:
		tasks_for_eval = pickle.load(f)  # Target tasks

	with open(cfg.env.source_skills_path, "rb") as f:
		task_to_source_skills = pickle.load(f)  # Task -> Predicted source skills (; Output of Semantic skill encoder)

	with open(pretrained_cfg["env"]["skill_infos_path"], "rb") as f:
	# with open("/home/jsw7460/mnt/comde_datasets/language_embeddings/clip_mappings/metaworld/clip_mapping", "rb") as f:
		skill_infos = pickle.load(f)  # skill to vector

	with open(pretrained_cfg["language_guidance_path"], "rb") as f:
	# with open("/home/jsw7460/mnt/comde_datasets/language_embeddings/clip_mappings/language_guidance/sequential_reverse_50var_clip_mapping_ViT-B_32", "rb") as f:
		language_guidances = pickle.load(f)
		language_guidances = language_guidances[cfg["language_guidance"]]  # Dict of vectors

	subseq_len = pretrained_cfg["subseq_len"]
	skill_dim = pretrained_cfg["skill_dim"] + pretrained_cfg["intent_dim"]

	env_class = get_class(cfg.env.path)  # type: Union[type, Type[gym.Env]]
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
		language_guidance=cfg.language_guidance,
		skill_infos=skill_infos,
		tasks=deepcopy([env.skill_list for env in envs_candidate])
	)

	envs = []
	source_skills = []
	for env, source_skill in zip(envs_candidate, source_skills_candidate):
		if source_skill is not None:
			envs.append(env)
			source_skills.append(source_skill)

	# Note: Among language variations (e.g, ["Do sequentially", "In order", ...], select one randomly)
	# language_guidance = random.choices(list(language_guidances.values()), k=len(source_skills))

	n_source_skills = [3 for _ in range(len(envs))]

	for k in language_guidances.keys():
		l = language_guidances[k]
		print("Guidance", k)

		info = evaluate_comde_batch(
			envs=envs,
			source_skills=source_skills,
			n_source_skills=n_source_skills,
			save_results=cfg.save_results,
			use_optimal_target_skill=cfg.use_optimal_target_skill,
			use_optimal_next_skill=cfg.use_optimal_next_skill,
			language_guidance=l,
			skill_infos=skill_infos,
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
