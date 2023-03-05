from typing import Tuple

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import PRNGKey


@jax.jit
def termination_forward(
	rng: jnp.ndarray,
	model: Model,
	observations: jnp.ndarray,
	first_observations: jnp.ndarray,
	skills: jnp.ndarray
) -> Tuple[PRNGKey, jnp.ndarray]:
	model_input = jnp.concatenate((observations, first_observations, skills), axis=-1)
	rng, dropout_key = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
		model_input,
		rngs={"dropout": dropout_key},
		deterministic=True,
		training=False
	)
	return rng, prediction