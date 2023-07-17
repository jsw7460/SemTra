import pickle
from copy import deepcopy
from typing import List, Dict, Union

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
) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:

	semantic_skills_sequence, optimal_idxs = get_optimal_semantic_skills(envs=envs, skill_infos=skill_infos)

	if cfg.non_functionality in ["speed", "wind", "weight", "vehicle"]:
		params_for_skills = []
		params_to_check = []
		for optimal_idx, env in zip(optimal_idxs, envs):
			param_to_check = env.get_parameter_from_adjective(cfg.parameter)
			parameter_dict = deepcopy(param_to_check)
			params_to_check.append(parameter_dict)

			if parameter_dict.keys() != param_to_check.keys() or \
				np.any(np.array(list(parameter_dict.values())) != np.array(list(param_to_check.values()))):
				continue

			# for optimal_idx in optimal_idxs:  # iteration for the number of envs
			param_for_skill = get_params_for_skills(optimal_idx, parameter_dict)
			param_for_skill = env.get_buffer_parameter(param_for_skill)
			param_for_skill = np.repeat(param_for_skill, repeats=param_repeats, axis=-1)
			params_for_skills.append(param_for_skill)

		params_for_skills = np.stack(params_for_skills, axis=0)

	else:
		raise NotImplementedError("Undefined non functionality")

	non_functionalities = np.expand_dims(non_functionalities, axis=(0, 1))
	non_functionalities = np.broadcast_to(non_functionalities, semantic_skills_sequence.shape)

	optimal_template = {
		"params_to_check": params_to_check,
		"optimal_target_skill_idxs": optimal_idxs,
		"semantic_skills_sequence": semantic_skills_sequence,
		"non_functionalities": non_functionalities,
		"params_for_skills": params_for_skills,  # [n_eval (or some number), n_env, n_target_seq, param_dim]
	}
	return optimal_template
