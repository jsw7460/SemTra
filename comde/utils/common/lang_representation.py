from typing import NamedTuple

import numpy as np


class LanguageRepresentation(NamedTuple):
	title: str
	variation: str
	vec: np.ndarray