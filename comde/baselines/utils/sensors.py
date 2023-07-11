from typing import NamedTuple

import numpy as np


class ObservationsActions(NamedTuple):
	observations: np.ndarray
	actions: np.ndarray
