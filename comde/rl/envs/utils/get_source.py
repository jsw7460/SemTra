from typing import Dict, List, Union

import numpy as np


def get_source_skills(
	task_to_source_skills: Dict[str, List[str]],
	skill_to_vec: Dict[str, np.ndarray],
	task: Union[str, list]
) -> np.ndarray:
	if type(task) == list:
		task = " ".join(task)
	source_skills = task_to_source_skills[task]

	skill_vectors = []
	for skill in source_skills:

		skill_vectors.append(skill_to_vec[skill])

	source_skills = np.array(skill_vectors)
	return source_skills
