from typing import NamedTuple, List, Dict, Union

import numpy as np


class LanguageRepresentation(NamedTuple):
	title: str
	variation: str
	vec: Union[np.ndarray, str, None]


class SkillRepresentation(NamedTuple):
	title: str
	variation: Union[str, List[str]]
	vec: Union[np.ndarray, str, None]
	index: int = -1	# Null skill


class Template(NamedTuple):
	sequential_requirement: Dict	# {name: str ("Sequential"), variation: str("Do these tasks in order"), vec: np.array}
	non_functionality: Dict	# {name: str, variation: str vec: np.array},
	parameter: Dict	# {1: 0.2, 3: 3.5, 6: xxx, ...}
