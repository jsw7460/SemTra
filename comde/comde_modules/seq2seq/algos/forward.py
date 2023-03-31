from typing import Tuple
from functools import partial

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import PRNGKey


@partial(jax.jit, static_argnames=("batch_size", ))
def skilltoskill_model_forward(
	rng: jnp.ndarray,
	model: Model,
	sequence: jnp.ndarray,
	batch_size: int,
) -> Tuple[PRNGKey, jnp.ndarray]:
	rng, dropout_key, carry_key = jax.random.split(rng, 3)
	stacked_output = model.apply_fn(
		{"params": model.params},
		sequence,
		batch_size,
		rngs={"dropout": dropout_key, "init_carry": carry_key},
	)
	return rng, stacked_output


@jax.jit
def skilltoskill_transformer_forward(
	rng: jnp.ndarray,
	model: Model,
	start_token: jnp.ndarray
):
