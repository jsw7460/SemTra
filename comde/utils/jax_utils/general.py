from typing import Callable, Tuple, Dict

import jax
from jax import dtypes
from jax import numpy as jnp
from jax import random

from flax import linen as nn

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Dtype


def get_basic_rngs(rng: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
	rng, param_key, dropout_key, batch_key = jax.random.split(rng, 4)
	return rng, {"params": param_key, "dropout": dropout_key, "batch_stats": batch_key}


def uniform(scale=1e-2, dtype: Dtype = jnp.float_) -> Callable:
	"""Builds an initializer that returns real uniformly-distributed random arrays.

	Args:
	  scale: optional; the upper bound of the random distribution.
	  dtype: optional; the initializer's default dtype.

	Returns:
	  An initializer that returns arrays whose values are uniformly distributed in
	  the range ``[0, scale)``.

	>>> import jax, jax.numpy as jnp
	>>> initializer = jax.nn.initializers.uniform(10.0)
	>>> initializer(jax.random.PRNGKey(42), (2, 3), jnp.float32)  # doctest: +SKIP
	DeviceArray([[7.298188 , 8.691938 , 8.7230015],
				 [2.0818567, 1.8662417, 5.5022564]], dtype=float32)
	"""

	def init(key, shape, _dtype=dtype):
		_dtype = dtypes.canonicalize_dtype(_dtype)
		return (random.uniform(key, shape, dtype) * 2 - 1) * scale

	return init


def preprocess_images(
	images: jnp.ndarray
):
	assert images.dtype == jnp.int8  # Normalized
	assert (jnp.argmin(images.shape)) == (len(images.shape) - 1)  # Channel last
	images /= 255.0
	return images


def polyak_update(source: Model, target: Model, tau: float) -> Model:
	new_target_params = jax.tree_map(lambda p, tp: p * tau + tp * (1 - tau), source.params, target.params)
	return target.replace(params=new_target_params)


def jnp_polyak_update(source: jnp.ndarray, target: jnp.ndarray, tau: float) -> jnp.ndarray:
	return source * tau + target * (1 - tau)


def str_to_flax_activation(name: str):
	return getattr(nn, name)