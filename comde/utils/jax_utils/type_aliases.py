from typing import Sequence, Any, Dict, Union

import flax
from jax import numpy as jnp

PRNGKey = Any
Shape = Sequence[int]
Dtype = Any
Array = Any
Params = flax.core.FrozenDict[str, Any]
TensorDict = Dict[Union[str, int], jnp.ndarray]
