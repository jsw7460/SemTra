import os
from os.path import join, isfile
from pathlib import Path
from typing import Dict, Tuple, List

import h5py
import numpy as np

EPS = 1e-12


def get_normalized_data(dataset: Dict, normalization_components: List) -> Tuple[Dict, Dict]:
	possible_normalizations = {"observation"}
	assert set(normalization_components) <= possible_normalizations, \
		f"Undefined normalization type: {possible_normalizations - set(normalization_components)}"
	obs_mean = 0.
	obs_std = 1.
	act_mean = 0.
	act_std = 1.

	if "observation" in normalization_components:
		observations = dataset["observations"]
		obs_mean = np.mean(observations, axis=0)
		obs_std = np.std(observations, axis=0) + EPS
		dataset["observations"] = (observations - obs_mean) / obs_std

	if "action" in normalization_components:
		actions = dataset["actions"]
		act_mean = np.mean(actions, axis=0)
		act_std = np.std(actions, axis=0) + EPS
		dataset["actions"] = (actions - act_mean) / act_std

	return dataset, {"obs_mean": obs_mean, "obs_std": obs_std, "act_mean": act_mean, "act_std": act_std}


def get_observation_statistics(dir_paths: List[Path]):
	"""
		Return mean and variance of observations
		!Not! responsible for other components (actions, skills, ...)
	"""

	observations_list = []

	for dir_path in dir_paths:
		traj_paths = [join(dir_path, f) for f in os.listdir(dir_path) if isfile(join(dir_path, f))]

		for traj in traj_paths:
			trajectory = h5py.File(traj, "r")
			observations_list.extend(trajectory["observations"])

	obs_mean = np.mean(np.array(observations_list), axis=0)
	obs_var = np.var(np.array(observations_list), axis=0)
	obs_std = np.std(np.array(observations_list), axis=0)
	return {"obs_mean": obs_mean, "obs_std": obs_std, "obs_var": obs_var}
