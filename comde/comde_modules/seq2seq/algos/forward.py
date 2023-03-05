from typing import Tuple

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import PRNGKey


@jax.jit
def skilltoskill_lstm_forward(
	rng: jnp.ndarray,
	model: Model,
	sequence: jnp.ndarray,
	batch_size: int,
	deterministic: bool = True
) -> Tuple[PRNGKey, jnp.ndarray]:
	rng, dropout_key = jax.random.split(rng)
	stacked_output, _ = model.apply_fn(
		{"params": model.params},
		sequence,
		batch_size,
		rngs={"dropout": dropout_key},
		deterministic=deterministic
	)
	return rng, stacked_output
