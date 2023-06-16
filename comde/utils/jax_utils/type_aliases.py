from typing import Any, Callable, Dict, Sequence, Union

import flax
from jax import numpy as jnp

PRNGKey = Any
Shape = Sequence[int]
Dtype = Any
Array = Any
Params = flax.core.FrozenDict[str, Any]
TensorDict = Dict[Union[str, int], jnp.ndarray]
Activation = Callable[[jnp.ndarray], jnp.ndarray]
ModuleLike = Union[flax.linen.Module, Activation]
