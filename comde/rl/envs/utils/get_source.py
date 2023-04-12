from typing import Dict, List, Union
import random

import numpy as np
from comde.utils.common.lang_representation import SkillRepresentation


def get_source_skills(
	task_to_source_skills: Dict[str, Dict[str, List]],
	language_guidance: str,
	skill_infos: Dict[str, List[SkillRepresentation]],
	task: Union[str, List[str]]
) -> Union[np.ndarray, None]:

	task = task[: 3]

	# if type(task) == list:
	task = [str(t) for t in task]
	task_str = " ".join(task)

	compositional_language_guidance = language_guidance.split("||")[-1].strip()

	source_skills = task_to_source_skills[task_str][compositional_language_guidance]
	skill_vectors = []

	if source_skills is not None:
		for skill in source_skills:
			skill_vectors.append(random.choice(skill_infos[skill]).vec)
		source_skills = np.array(skill_vectors)
		return source_skills

	else:
		return None


def get_batch_source_skills(		# Evaluation on batch environment
	task_to_source_skills: Dict[str, Dict[str, List]],
	language_guidance: str,
	skill_infos: Dict[str, List[SkillRepresentation]],
	tasks: Union[str, list]
) -> List[np.ndarray]:
	ret = [get_source_skills(task_to_source_skills, language_guidance, skill_infos, task) for task in tasks]
	return ret
