import pickle
from typing import List, Dict

import numpy as np
from omegaconf import DictConfig

from comde.rl.envs.utils.get_source import get_optimal_semantic_skills


def get_optimal_template(
	cfg: DictConfig,
	envs: List,
	skill_infos: Dict,
	non_functionalities: np.ndarray,
	param_repeats: int
) -> Dict[str, np.ndarray]:

	param_to_check = envs[0].get_default_parameter(cfg.non_functionality)

	semantic_skills_sequence, optimal_idxs = get_optimal_semantic_skills(envs=envs, skill_infos=skill_infos)
	with open(cfg.env.template_path, "rb") as f:
		templates = pickle.load(f)

	templates = templates[cfg.sequential_requirement]

	if cfg.non_functionality in ["speed", "wind", "weight", "vehicle"]:
		params_for_skills = []
		for template in templates:

			parameter_dict = template.parameter
			n_max_target_skills = optimal_idxs.shape[-1]

			raw_param_dim = np.array([list(parameter_dict.values())[0]]).shape[-1]
			param_for_skill = np.zeros((len(envs), n_max_target_skills, raw_param_dim))
			if parameter_dict.keys() != param_to_check.keys() or \
				np.any(np.array(list(parameter_dict.values())) != np.array(list(param_to_check.values()))):
				continue

			for skill_idx, parameter in parameter_dict.items():
				idxs = np.where(optimal_idxs == skill_idx)	# [9, 3]
				param_for_skill[idxs] = parameter

			param_for_skill = np.repeat(param_for_skill, repeats=param_repeats, axis=-1)
			params_for_skills.append(param_for_skill)
		params_for_skills = np.stack(params_for_skills, axis=0)	# [n_envs, target_skill_len, param_dim]

	else:
		raise NotImplementedError("Undefined non functionality")

	non_functionalities = np.expand_dims(non_functionalities, axis=(0, 1))
	non_functionalities = np.broadcast_to(non_functionalities, semantic_skills_sequence.shape)

	optimal_template = {
		"optimal_target_skill_idxs": optimal_idxs,
		"semantic_skills_sequence": semantic_skills_sequence,
		"non_functionalities": non_functionalities,
		"params_for_skills": params_for_skills,
	}
	return optimal_template
