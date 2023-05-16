from typing import Dict
import os
from os.path import join, isfile
from pathlib import Path
import random


def load_data_paths(cfg: Dict):
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

	return data_dirs, hdf_files