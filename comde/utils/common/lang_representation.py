from typing import NamedTuple, List

import numpy as np


class LanguageRepresentation(NamedTuple):
	title: str
	variation: str
	vec: np.ndarray


class SkillRepresentation(NamedTuple):
	title: str
	variation: List[str]
	vec: List[np.ndarray]
	index: int = -1	# Null skill

