import h5py
from jax.config import config

config.update("jax_debug_nans", True)

import random

random.seed(0)

import os
from os.path import join, isfile
from pathlib import Path

dataset_path = "/home/jsw7460/mnt/comde_datasets/episodes/metaworld/3_target_skills/"

data_dirs = [Path(dataset_path) / Path(name) for name in os.listdir(dataset_path)]
hdf_files = []
for data_dir in data_dirs:
	hdf_files.extend([join(data_dir, f) for f in os.listdir(data_dir) if isfile(join(data_dir, f))])

for j, path in enumerate(hdf_files):
	# print()
	trajectory = h5py.File(path, "r+")
	# print(trajectory["actions"])
	# print(trajectory.keys())
	# trajectory["operator"] 	# replace 1 with 4
	# trajectory["operator"] = "replace asdf with xzcv"
	# print(trajectory.keys())

	operator = str(trajectory["operator"][()], "utf-8")
	# print(operator)
	if "replace" in operator:
		z = operator.split("replace")
		print("Before:", operator)

		src = z[1][1]
		tar = z[1][-1]
		rectified = f"replace {tar} with {src}"
		# trajectory.create_dataset("operator", data=rectified)
		del trajectory["operator"]
		trajectory["operator"] = f"replace {tar} with {src}"

		trajectory.close()
	else:
		trajectory.close()

	# print(str(trajectory["operator"]))

# print(x)
# trajectory["operator"][()]
