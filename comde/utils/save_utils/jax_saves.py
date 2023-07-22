"""
Save util taken from stable_baselines
used to serialize data (class parameters) of model classes
"""
import io
import os
import pathlib
import platform
import warnings
import zipfile
from typing import Dict, Optional, Tuple, Union, Any

import flax
import gym
import jax
import numpy as np
import torch as th

from comde.utils.jax_utils.type_aliases import TensorDict, Params
from comde.utils.save_utils.common import open_path, data_to_json, json_to_data


def get_system_info(print_info: bool = True) -> Tuple[Dict[str, str], str]:
	"""
	Retrieve system and python env info for the current system.

	:param print_info: Whether to print or not those infos
	:return: Dictionary summing up the version for each relevant package
		and a formatted string.
	"""
	env_info = {
		"OS": f"{platform.platform()} {platform.version()}",
		"Python": platform.python_version(),
		"JAX": jax.__version__,
		"Numpy": np.__version__,
		"Gym": gym.__version__,
	}
	env_info_str = ""
	for key, value in env_info.items():
		env_info_str += f"{key}: {value}\n"
	if print_info:
		print(env_info_str)
	return env_info, env_info_str


def save_to_zip_file(
	save_path: Union[str, pathlib.Path, io.BufferedIOBase],
	data: Optional[Dict[str, Any]] = None,
	params: Optional[Dict[str, Params]] = None,
) -> None:
	"""
	Save model data to a zip archive.

	:param save_path: Where to store the model.
		if save_path is a str or pathlib.Path ensures that the path actually exists.
	:param data: Class parameters being stored (non-PyTorch variables)
	:param params: Model parameters being stored expected to contain an entry for every
				   state_dict with its name and the state_dict.
	"""
	save_path = open_path(save_path, "w", verbose=0, suffix="zip")
	# data/params can be None, so do not
	# try to serialize them blindly
	serialized_data = None
	if data is not None:
		serialized_data = data_to_json(data)

	# Create a zip-archive and write our objects there.
	with zipfile.ZipFile(save_path, mode="w") as archive:
		# Do not try to save "None" elements
		if data is not None:
			archive.writestr("data", serialized_data)
		if params is not None:
			for file_name, dict_ in params.items():
				try:  # jax
					bytes_dict = flax.serialization.to_bytes(dict_)
					with archive.open(file_name + ".jax", mode="w") as param_file:
						param_file.write(bytes_dict)
				except:  # torch
					with archive.open(file_name + ".pth", mode="w", force_zip64=True) as param_file:
						th.save(dict_, param_file)

		# Save system info about the current python env
		archive.writestr("system_info.txt", get_system_info(print_info=False)[1])


def load_from_zip_file(
	load_path: Union[str, pathlib.Path, io.BufferedIOBase],
	load_data: bool = True,
	custom_objects: Optional[Dict[str, Any]] = None,
	verbose: int = 0,
	print_system_info: bool = False,
) -> (Tuple[Optional[Dict[str, Any]], Dict[str, Params], Optional[TensorDict]]):
	"""
	Load model data from a .zip archive

	:param load_path: Where to load the model from
	:param load_data: Whether we should load and return data
		(class parameters). Mainly used by 'load_parameters' to only load model parameters (weights)
	:param custom_objects: Dictionary of objects to replace
		upon loading. If a variable is present in this dictionary as a
		key, it will not be deserialized and the corresponding item
		will be used instead. Similar to custom_objects in
		``keras.models.load_model``. Useful when you have an object in
		file that can not be deserialized.
	:param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
	:param print_system_info: Whether to print or not the system info
		about the saved model.
	:return: Class parameters, model state_dicts (aka "params", dict of state_dict)
		and dict of pytorch variables
	"""
	load_path = open_path(load_path, "r", verbose=verbose, suffix="zip")
	# Open the zip archive and load data
	try:
		with zipfile.ZipFile(load_path) as archive:
			namelist = archive.namelist()
			# If data or parameters is not in the
			# zip archive, assume they were stored
			# as None (_save_to_file_zip allows this).
			data = None
			pytorch_variables = None

			# Debug system info first
			if print_system_info:
				if "system_info.txt" in namelist:
					print("== SAVED MODEL SYSTEM INFO ==")
					print(archive.read("system_info.txt").decode())
				else:
					warnings.warn(
						"The model was saved with SB3 <= 1.2.0 and thus cannot print system information.",
						UserWarning,
					)

			if "data" in namelist and load_data:
				# Load class parameters that are stored
				# with either JSON or pickle (not PyTorch variables).
				json_data = archive.read("data").decode()
				data = json_to_data(json_data, custom_objects=custom_objects)

			# Check for all .pth files and load them using th.load.
			# "pytorch_variables.pth" stores PyTorch variables, and any other .pth
			# files store state_dicts of variables with custom names (e.g. policy, policy.optimizer)
			pth_files = [file_name for file_name in namelist if os.path.splitext(file_name)[1] in [".jax", ".pth"]]
			params = dict()
			for file_path in pth_files:
				with archive.open(file_path, mode="r") as param_file:
					if ".jax" in file_path:	# Jax
						_params = param_file.read()
					else:	# Torch
						file_content = io.BytesIO()
						file_content.write(param_file.read())
						# go to start of file
						file_content.seek(0)
						# Load the parameters with the right ``map_location``.
						# Remove ".pth" ending with splitext
						_params = th.load(file_content, map_location="cuda:0")

					params[file_path.split('.')[0]] = _params


	except zipfile.BadZipFile:
		# load_path wasn't a zip file
		raise ValueError(f"Error: the file {load_path} wasn't a zip-file")

	return data, params, pytorch_variables
