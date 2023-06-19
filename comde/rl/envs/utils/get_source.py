import random
from typing import Dict, List, Union, Tuple

import numpy as np

from comde.rl.envs.utils import SkillHistoryEnv
from comde.utils.common.natural_languages.lang_representation import SkillRepresentation


def get_source_skills(
	task_to_source_skills: Dict[str, Dict[str, List]],
	sequential_requirement: str,
	skill_infos: Dict[str, List[SkillRepresentation]],
	task: Union[str, List[str]]
) -> Union[Dict, None]:
	# task = task[: 3]

	# if type(task) == list:
	task = [str(t) for t in task]
	task_str = "".join(task).replace(" ","")

	stripped_task_to_source_skills = {
		k.replace(" ", "").replace("_", ""): v for k, v in task_to_source_skills.items()
	}
	source_skills = stripped_task_to_source_skills[task_str][sequential_requirement]
	source_skill_vectors = []
	source_skill_idxs = []

	if source_skills is not None:
		for skill in source_skills:
			skill_rep = random.choice(skill_infos[skill])
			source_skill_vectors.append(skill_rep.vec)
			source_skill_idxs.append(skill_rep.index)

		np_source_skills = np.array(source_skill_vectors)
		return {"np_source_skills": np_source_skills, "source_skill_idxs": source_skill_idxs}

	else:
		return None


def get_batch_source_skills(  # Evaluation on batch environment
	task_to_source_skills: Dict[str, Dict[str, List]],
	sequential_requirement: str,
	skill_infos: Dict[str, List[SkillRepresentation]],
	tasks: Union[str, list]
) -> Dict:
	skill_dict_list \
		= [get_source_skills(task_to_source_skills, sequential_requirement, skill_infos, task) for task in tasks]
	np_source_skills = [skill_dict["np_source_skills"] for skill_dict in skill_dict_list if skill_dict is not None]
	source_skill_idxs = [skill_dict["source_skill_idxs"] for skill_dict in skill_dict_list if skill_dict is not None]
	return {"np_source_skills": np_source_skills, "source_skill_idxs": source_skill_idxs}


def get_optimal_semantic_skills(
	envs: List[SkillHistoryEnv],
	skill_infos,
) -> Tuple[np.ndarray, np.ndarray]:
	optimal_target_idxs = []
	optimal_target_skills = []
	for env in envs:
		skill_list = env.skill_list
		optimal_target_skill = np.array([skill_infos[skill][0].vec for skill in skill_list])

		n_target = env.n_target
		optimal_target_idx = env.idx_skill_list[:n_target]	# Truncate it !
		optimal_target_skill = optimal_target_skill[:n_target]	# Truncate it !

		optimal_target_idxs.append(optimal_target_idx)
		optimal_target_skills.append(optimal_target_skill)

	optimal_target_idxs = np.array(optimal_target_idxs)
	optimal_target_skills = np.array(optimal_target_skills)
	print("What is optimal target skills shape?", optimal_target_skills.shape)
	return optimal_target_skills, optimal_target_idxs
