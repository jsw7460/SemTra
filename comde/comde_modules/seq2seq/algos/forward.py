from typing import Tuple
from functools import partial

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import PRNGKey


@partial(jax.jit, static_argnames=("batch_size", ))
def skilltoskill_lstm_forward(
	rng: jnp.ndarray,
	model: Model,
	sequence: jnp.ndarray,
	batch_size: int,
	deterministic: bool = True
) -> Tuple[PRNGKey, jnp.ndarray]:
	rng, dropout_key, carry_key = jax.random.split(rng, 3)
	stacked_output, _ = model.apply_fn(
		{"params": model.params},
		sequence,
		batch_size,
		rngs={"dropout": dropout_key, "init_carry": carry_key},
		deterministic=deterministic
	)
	return rng, stacked_output
