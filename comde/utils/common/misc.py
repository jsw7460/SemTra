from typing import Dict

import numpy as np


def get_params_for_skills(
	skills_idxs: np.ndarray,  # [l, ]
	parameter: Dict,
	n_repeats: int = 1
) -> np.ndarray:
	"""
	:param skills_idxs:	# [sequence length, ]
	:param parameter:
	:param n_repeats:
	:return: [seq_len, param_dim]
	"""

	seq_len = skills_idxs.shape[0]
	raw_param_dim = np.array([list(parameter.values())[0]]).shape[-1]
	return_parameter = np.zeros((seq_len, raw_param_dim))
	for skill_idx, param in parameter.items():
		idxs = np.where(skills_idxs == skill_idx)
		return_parameter[idxs] = param

	return_parameter = np.repeat(return_parameter, repeats=n_repeats, axis=-1)
	return return_parameter
