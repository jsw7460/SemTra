import base64
import functools
import io
import json
import pathlib
import pickle
import warnings
from typing import Any
from typing import Dict, Optional, Union

import cloudpickle


def recursive_getattr(obj: Any, attr: str, *args) -> Any:
	"""
	Recursive version of getattr
	taken from https://stackoverflow.com/questions/31174295

	Ex:
	> MyObject.sub_object = SubObject(name='test')
	> recursive_getattr(MyObject, 'sub_object.name')  # return test
	:param obj:
	:param attr: Attribute to retrieve
	:return: The attribute
	"""

	def _getattr(_obj: Any, _attr: str) -> Any:
		return getattr(_obj, _attr, *args)

	return functools.reduce(_getattr, [obj] + attr.split("."))


def recursive_setattr(obj: Any, attr: str, val: Any) -> None:
	"""
	Recursive version of setattr
	taken from https://stackoverflow.com/questions/31174295

	Ex:
	> MyObject.sub_object = SubObject(name='test')
	> recursive_setattr(MyObject, 'sub_object.name', 'hello')
	:param obj:
	:param attr: Attribute to set
	:param val: New value of the attribute
	"""
	pre, _, post = attr.rpartition(".")
	return setattr(recursive_getattr(obj, pre) if pre else obj, post, val)


def is_json_serializable(item: Any) -> bool:
	"""
	Test if an object is serializable into JSON

	:param item: The object to be tested for JSON serialization.
	:return: True if object is JSON serializable, false otherwise.
	"""
	# Try with try-except struct.
	json_serializable = True
	try:
		_ = json.dumps(item)
	except TypeError:
		json_serializable = False
	return json_serializable


def save_to_pkl(path: Union[str, pathlib.Path, io.BufferedIOBase], obj: Any, verbose: int = 0) -> None:
	"""
	Save an object to path creating the necessary folders along the way.
	If the path exists and is a directory, it will raise a warning and rename the path.
	If a suffix is provided in the path, it will use that suffix, otherwise, it will use '.pkl'.
	:param path: the path to open.
		if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
		path actually exists. If path is a io.BufferedIOBase the path exists.
	:param obj: The object to save.
	:param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
	"""
	with open_path(path, "w", verbose=verbose, suffix="pkl") as file_handler:
		# Use protocol>=4 to support saving replay buffers >= 4Gb
		# See https://docs.python.org/3/library/pickle.html
		pickle.dump(obj, file_handler, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pkl(path: Union[str, pathlib.Path, io.BufferedIOBase], verbose: int = 0) -> Any:
	"""
	Load an object from the path. If a suffix is provided in the path, it will use that suffix.
	If the path does not exist, it will attempt to load using the .pkl suffix.
	:param path: the path to open.
		if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
		path actually exists. If path is a io.BufferedIOBase the path exists.
	:param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
	"""
	with open_path(path, "r", verbose=verbose, suffix="pkl") as file_handler:
		return pickle.load(file_handler)


@functools.singledispatch
def open_path(
	path: Union[str, pathlib.Path, io.BufferedIOBase],
	mode: str,
	verbose: int = 0,
 	suffix: Optional[str] = None
):
	"""
	Opens a path for reading or writing with a preferred suffix and raises debug information.
	If the provided path is a derivative of io.BufferedIOBase it ensures that the file
	matches the provided mode, i.e. If the mode is read ("r", "read") it checks that the path is readable.
	If the mode is write ("w", "write") it checks that the file is writable.

	If the provided path is a string or a pathlib.Path, it ensures that it exists. If the mode is "read"
	it checks that it exists, if it doesn't exist it attempts to read path.suffix if a suffix is provided.
	If the mode is "write" and the path does not exist, it creates all the parent folders. If the path
	points to a folder, it changes the path to path_2. If the path already exists and verbose == 2,
	it raises a warning.

	:param path: the path to open.
		if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
		path actually exists. If path is a io.BufferedIOBase the path exists.
	:param mode: how to open the file. "w"|"write" for writing, "r"|"read" for reading.
	:param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
	:param suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
		If mode is "r" then we attempt to open the path. If an error is raised and the suffix
		is not None, we attempt to open the path with the suffix.
	:return:
	"""
	if not isinstance(path, io.BufferedIOBase):
		raise TypeError("Path parameter has invalid type.", io.BufferedIOBase)
	if path.closed:
		raise ValueError("File stream is closed.")
	mode = mode.lower()
	try:
		mode = {"write": "w", "read": "r", "w": "w", "r": "r"}[mode]
	except KeyError as e:
		raise ValueError("Expected mode to be either 'w' or 'r'.") from e
	if ("w" == mode) and not path.writable() or ("r" == mode) and not path.readable():
		e1 = "writable" if "w" == mode else "readable"
		raise ValueError(f"Expected a {e1} file.")
	return path


def data_to_json(data: Dict[str, Any]) -> str:
	"""
	Turn data (class parameters) into a JSON string for storing

	:param data: Dictionary of class parameters to be
		stored. Items that are not JSON serializable will be
		pickled with Cloudpickle and stored as bytearray in
		the JSON file
	:return: JSON string of the data serialized.
	"""
	# First, check what elements can not be JSONfied,
	# and turn them into byte-strings
	serializable_data = {}
	for data_key, data_item in data.items():
		# See if object is JSON serializable
		if is_json_serializable(data_item):
			# All good, store as it is
			serializable_data[data_key] = data_item
		else:
			# Not serializable, cloudpickle it into
			# bytes and convert to base64 string for storing.
			# Also store type of the class for consumption
			# from other languages/humans, so we have an
			# idea what was being stored.
			base64_encoded = base64.b64encode(cloudpickle.dumps(data_item)).decode()

			# Use ":" to make sure we do
			# not override these keys
			# when we include variables of the object later
			cloudpickle_serialization = {
				":type:": str(type(data_item)),
				":serialized:": base64_encoded,
			}

			# Add first-level JSON-serializable items of the
			# object for further details (but not deeper than this to
			# avoid deep nesting).
			# First we check that object has attributes (not all do,
			# e.g. numpy scalars)
			if hasattr(data_item, "__dict__") or isinstance(data_item, dict):
				# Take elements from __dict__ for custom classes
				item_generator = data_item.items if isinstance(data_item, dict) else data_item.__dict__.items
				for variable_name, variable_item in item_generator():
					# Check if serializable. If not, just include the
					# string-representation of the object.
					if is_json_serializable(variable_item):
						cloudpickle_serialization[variable_name] = variable_item
					else:
						cloudpickle_serialization[variable_name] = str(variable_item)

			serializable_data[data_key] = cloudpickle_serialization
	json_string = json.dumps(serializable_data, indent=4)
	return json_string


def json_to_data(json_string: str, custom_objects: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
	"""
	Turn JSON serialization of class-parameters back into dictionary.

	:param json_string: JSON serialization of the class-parameters
		that should be loaded.
	:param custom_objects: Dictionary of objects to replace
		upon loading. If a variable is present in this dictionary as a
		key, it will not be deserialized and the corresponding item
		will be used instead. Similar to custom_objects in
		``keras.models.load_model``. Useful when you have an object in
		file that can not be deserialized.
	:return: Loaded class parameters.
	"""
	if custom_objects is not None and not isinstance(custom_objects, dict):
		raise ValueError("custom_objects argument must be a dict or None")

	json_dict = json.loads(json_string)
	# This will be filled with deserialized data
	return_data = {}
	for data_key, data_item in json_dict.items():
		if custom_objects is not None and data_key in custom_objects.keys():
			# If item is provided in custom_objects, replace
			# the one from JSON with the one in custom_objects
			return_data[data_key] = custom_objects[data_key]
		elif isinstance(data_item, dict) and ":serialized:" in data_item.keys():
			# If item is dictionary with ":serialized:"
			# key, this means it is serialized with cloudpickle.
			serialization = data_item[":serialized:"]
			# Try-except deserialization in case we run into
			# errors. If so, we can tell bit more information to
			# user.
			try:
				base64_object = base64.b64decode(serialization.encode())
				deserialized_object = cloudpickle.loads(base64_object)
			except (RuntimeError, TypeError):
				warnings.warn(
					f"Could not deserialize object {data_key}. "
					+ "Consider using `custom_objects` argument to replace "
					+ "this object."
				)
			return_data[data_key] = deserialized_object
		else:
			# Read as it is
			return_data[data_key] = data_item
	return return_data


@open_path.register(str)
def open_path_str(path: str, mode: str, verbose: int = 0, suffix: Optional[str] = None) -> io.BufferedIOBase:
	"""
	Open a path given by a string. If writing to the path, the function ensures
	that the path exists.

	:param path: the path to open. If mode is "w" then it ensures that the path exists
		by creating the necessary folders and renaming path if it points to a folder.
	:param mode: how to open the file. "w" for writing, "r" for reading.
	:param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
	:param suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
		If mode is "r" then we attempt to open the path. If an error is raised and the suffix
		is not None, we attempt to open the path with the suffix.
	:return:
	"""
	return open_path(pathlib.Path(path), mode, verbose, suffix)


@open_path.register(pathlib.Path)
def open_path_pathlib(path: pathlib.Path, mode: str, verbose: int = 0,
					  suffix: Optional[str] = None) -> io.BufferedIOBase:
	"""
	Open a path given by a string. If writing to the path, the function ensures
	that the path exists.

	:param path: the path to check. If mode is "w" then it
		ensures that the path exists by creating the necessary folders and
		renaming path if it points to a folder.
	:param mode: how to open the file. "w" for writing, "r" for reading.
	:param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
	:param suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
		If mode is "r" then we attempt to open the path. If an error is raised and the suffix
		is not None, we attempt to open the path with the suffix.
	:return:
	"""
	if mode not in ("w", "r"):
		raise ValueError("Expected mode to be either 'w' or 'r'.")

	if mode == "r":
		try:
			path = path.open("rb")
		except FileNotFoundError as error:
			if suffix is not None and suffix != "":
				newpath = pathlib.Path(f"{path}.{suffix}")
				if verbose == 2:
					warnings.warn(f"Path '{path}' not found. Attempting {newpath}.")
				path, suffix = newpath, None
			else:
				raise error
	else:
		try:
			if path.suffix == "" and suffix is not None and suffix != "":
				path = pathlib.Path(f"{path}.{suffix}")
			if path.exists() and path.is_file() and verbose == 2:
				warnings.warn(f"Path '{path}' exists, will overwrite it.")
			path = path.open("wb")
		except IsADirectoryError:
			warnings.warn(f"Path '{path}' is a folder. Will save instead to {path}_2")
			path = pathlib.Path(f"{path}_2")
		except FileNotFoundError:  # Occurs when the parent folder doesn't exist
			warnings.warn(f"Path '{path.parent}' does not exist. Will create it.")
			path.parent.mkdir(exist_ok=True, parents=True)

	# if opening was successful uses the identity function
	# if opening failed with IsADirectory|FileNotFound, calls open_path_pathlib
	#   with corrections
	# if reading failed with FileNotFoundError, calls open_path_pathlib with suffix

	return open_path(path, mode, verbose, suffix)
