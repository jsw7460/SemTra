import inspect
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union

from comde.utils.save_utils.common import (
	recursive_getattr,
	recursive_setattr
)
from comde.utils.save_utils.jax_saves import (
	save_to_zip_file,
	load_from_zip_file
)
from comde.utils.jax_utils.type_aliases import Params


class IJaxSavable(metaclass=ABCMeta):

	@abstractmethod
	def _excluded_save_params(self) -> List:
		"""
			Excluded component saving in pickle. In general, a tensor-like param object is saved separately.
			Such param object is given by _get_save_params method
		"""
		raise NotImplementedError()

	@abstractmethod
	def _get_save_params(self) -> Dict[str, Params]:
		raise NotImplementedError()

	@abstractmethod
	def _get_load_params(self) -> List[str]:
		raise NotImplementedError()

	def save(self, path: str) -> None:
		"""
			Required for adaptation step
			:param path: path to the file where the class
			Save all the attributes of the object and the model parameters in a zip-file.
		"""
		# Copy parameter list, so we don't mutate the original dict
		data = self.__dict__.copy()

		# Exclude is union of specified parameters (if any) and standard exclusions
		exclude = set(self._excluded_save_params())

		# Remove parameter entries of parameters which are to be excluded
		for param_name in exclude:
			data.pop(param_name, None)

		# Build dict of state_dicts
		params_to_save = self._get_save_params()
		save_to_zip_file(path, data=data, params=params_to_save)

	@classmethod
	def load(cls, path: str):
		data, params, *_ = load_from_zip_file(path)
		class_type_name = IJaxSavable.get_class_type_name(cls)

		raise NotImplementedError(f"Undefined Class for MMSBRL ({class_type_name})")

		# load parameters
		model.__dict__.update(data)
		# model.__dict__.update(kwargs)
		model.build_model()

		# put state_dicts back in place
		model.set_parameters(params, exact_match=True)
		return model

	@staticmethod
	def get_class_type_name(cls) -> str:
		raise NotImplementedError("See here")

	def set_parameters(
		self,
		load_path_or_dict: Union[str, Dict[str, Params]],
		exact_match: bool = True,
	) -> None:
		"""
		Load parameters from a given zip-file or a nested dictionary containing parameters for
		different modules (see ``get_parameters``).

		:param load_path_or_dict: Location of the saved data (path or file-like, see ``save``), or a nested
			dictionary containing nn.Module parameters used by the policy. The dictionary maps
			object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
		:param exact_match: If True, the given parameters should include parameters for each
			module and each of their parameters, otherwise raises an Exception. If set to False, this
			can be used to update only specific parameters.
		"""

		if isinstance(load_path_or_dict, dict):
			params = load_path_or_dict
		else:
			_, params, _ = load_from_zip_file(load_path_or_dict)

		objects_needing_update = set(self._get_load_params())
		updated_objects = set()

		for name in params.keys():
			try:
				attr = recursive_getattr(self, name)
			except Exception:
				# What errors recursive_getattr could throw? KeyError, but
				# possible something else too (e.g. if key is an int?).
				# Catch anything for now.
				raise ValueError(f"Key {name} is an invalid object name.")

			jax_model = attr.load_dict(params[name])
			recursive_setattr(self, name, jax_model)
			updated_objects.add(name)

		if exact_match and updated_objects != objects_needing_update:
			raise ValueError(
				"Names of parameters do not match agents' parameters: "
				f"expected {objects_needing_update}, got {updated_objects}"
			)
