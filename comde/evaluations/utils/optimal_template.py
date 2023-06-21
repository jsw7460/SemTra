import pickle
from typing import List, Dict

import numpy as np
from omegaconf import DictConfig

from comde.rl.envs.utils.get_source import get_optimal_semantic_skills
from comde.utils.common.misc import get_params_for_skills


def get_optimal_template(
	cfg: DictConfig,
	envs: List,
	skill_infos: Dict,
	non_functionalities: np.ndarray,
	param_repeats: int
) -> Dict[str, np.ndarray]:

	param_to_check = envs[0].get_default_parameter(cfg.non_functionality)
	param_to_check = {1: 1.5, 3: 25.0, 4: 15.0, 6: 25.0}

	semantic_skills_sequence, optimal_idxs = get_optimal_semantic_skills(envs=envs, skill_infos=skill_infos)
	with open(cfg.env.template_path, "rb") as f:
		templates = pickle.load(f)

	templates = templates[cfg.sequential_requirement]

	if cfg.non_functionality in ["speed", "wind", "weight", "vehicle"]:
		params_for_skills = []
		for template in templates:

			parameter_dict = param_to_check

			if parameter_dict.keys() != param_to_check.keys() or \
				np.any(np.array(list(parameter_dict.values())) != np.array(list(param_to_check.values()))):
				continue

			param_for_skill = []
			for optimal_idx in optimal_idxs:	# iteration for the number of envs
				param_for_skill.append(get_params_for_skills(optimal_idx, parameter_dict))
			param_for_skill = np.array(param_for_skill)
			# param_for_skill = get_param_for_skill(optimal_idxs, parameter_dict)
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
		"params_for_skills": params_for_skills,	# [n_eval (or some number), n_env, n_target_seq, param_dim]
	}
	return optimal_template


# def get_param_for_skill(skills_idxs: np.ndarray, parameter_dict: Dict):
# 	"""
# 	:param skills_idxs:	[n_envs, n_target_skills]
# 	:param parameter_dict:
# 	return: [n_envs, n_target_skills, param_dim]
# 	"""
# 	n_envs, n_target_skills = skills_idxs.shape
# 	raw_param_dim = np.array([list(parameter_dict.values())[0]]).shape[-1]
# 	param_for_skill = np.zeros((n_envs, n_target_skills, raw_param_dim))
# 	for skill_idx, parameter in parameter_dict.items():
# 		idxs = np.where(skills_idxs == skill_idx)  # [9, 3]
# 		param_for_skill[idxs] = parameter
#
# 	return param_for_skill
