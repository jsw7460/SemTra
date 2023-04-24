import random
from typing import Dict, List, Union, Tuple

import numpy as np

from comde.rl.envs.utils import SkillHistoryEnv
from comde.utils.common.lang_representation import SkillRepresentation


def get_source_skills(
	task_to_source_skills: Dict[str, Dict[str, List]],
	sequential_requirement: str,
	skill_infos: Dict[str, List[SkillRepresentation]],
	task: Union[str, List[str]]
) -> Union[np.ndarray, None]:
	task = task[: 3]

	# if type(task) == list:
	task = [str(t) for t in task]
	task_str = "".join(task).replace(" ","")

	stripped_task_to_source_skills = {
		k.replace(" ", "").replace("_", ""): v for k, v in task_to_source_skills.items()
	}

	source_skills = stripped_task_to_source_skills[task_str][sequential_requirement]
	skill_vectors = []

	if source_skills is not None:
		for skill in source_skills:
			skill_vectors.append(random.choice(skill_infos[skill]).vec)
		source_skills = np.array(skill_vectors)
		print("Source skills", source_skills)
		return source_skills

	else:
		return None


def get_batch_source_skills(  # Evaluation on batch environment
	task_to_source_skills: Dict[str, Dict[str, List]],
	sequential_requirement: str,
	skill_infos: Dict[str, List[SkillRepresentation]],
	tasks: Union[str, list]
) -> List[np.ndarray]:
	ret = [get_source_skills(task_to_source_skills, sequential_requirement, skill_infos, task) for task in tasks]
	return ret


def get_optimal_semantic_skills(
	envs: List[SkillHistoryEnv],
	skill_infos,
) -> Tuple[np.ndarray, np.ndarray]:
	optimal_target_idxs = []
	optimal_target_skills = []
	for env in envs:
		skill_list = env.skill_list
		optimal_target_idxs.append(env.idx_skill_list)
		optimal_target_skill = np.array([skill_infos[skill][0].vec for skill in skill_list])
		optimal_target_skills.append(optimal_target_skill)

	optimal_target_idxs = np.array(optimal_target_idxs)
	print("Optimal target idxs", optimal_target_idxs)
	optimal_target_skills = np.array(optimal_target_skills)
	return optimal_target_skills, optimal_target_idxs
