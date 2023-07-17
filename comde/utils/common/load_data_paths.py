import os
import pickle
import random
from os.path import join, isfile
from pathlib import Path
from typing import Dict, Union, List

import h5py

from comde.rl.envs.base import ComdeSkillEnv
from comde.rl.envs.utils.skill_to_vec import SkillInfoEnv


def load_data_paths(cfg: Dict, env: Union[ComdeSkillEnv, SkillInfoEnv], rm_eval_tasks: bool = True):
	data_dirs = [
		Path(cfg["dataset_path"]) / Path(name)
		for name in os.listdir(cfg["dataset_path"]) if name not in cfg["excluded_dirs"]
	]
	data_dirs = data_dirs[:cfg["datadir_limit"]]
	hdf_files = []
	for data_dir in data_dirs:
		hdf_files.extend([join(data_dir, f) for f in os.listdir(data_dir) if isfile(join(data_dir, f))])
	assert len(hdf_files) > 0, "Empty dataset"
	if cfg["dataset"]["shuffle_dataset"]:
		random.shuffle(hdf_files)

	eval_tasks_path = cfg["env"]["eval_tasks_path"]

	if (not rm_eval_tasks) or (cfg["mode"]["mode"] == "baseline") or (eval_tasks_path == "None"):
		return hdf_files, data_dirs

	with open(eval_tasks_path, "rb") as f:
		str_eval_tasks = pickle.load(f)  # type: List[List[str]]

	eval_tasks = []
	for eval_task in str_eval_tasks:
		eval_task = [et.replace("_", " ") for et in eval_task]
		idx_eval_task = env.str_to_idxs_skills(env.onehot_skills_mapping, eval_task, to_str=True)
		eval_tasks.append(idx_eval_task)

	file_wo_eval_tasks = []
	for path in hdf_files:
		trajectory = h5py.File(path, "r")

		target_skills = list(trajectory["target_skills"])
		_train_task = "".join([str(tar_sk) for tar_sk in target_skills])

		add = True
		for eval_task in eval_tasks:
			_eval_task = "".join(eval_task)
			if _train_task in _eval_task:
				add = False
				continue
		if add:
			file_wo_eval_tasks.append(path)
	del hdf_files
	return file_wo_eval_tasks, data_dirs
