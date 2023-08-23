from typing import Tuple, Dict

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import PRNGKey


@jax.jit
def dynamics_encoder_forward(
	rng: jnp.ndarray,
	encoder: Model,
	sequence: jnp.ndarray,
	n_iter: jnp.ndarray,
	deterministic: bool = True
) -> Tuple[PRNGKey, Dict]:
	rng, dropout_key, carry_key = jax.random.split(rng, 3)
	output = encoder.apply_fn(
		{"params": encoder.params},
		sequence=sequence,
		n_iter=n_iter,
		rngs={"dropout": dropout_key, "init_carry": carry_key},
		deterministic=deterministic
	)

	return rng, output


@jax.jit
def dynamics_decoder_forward(
	rng: jnp.ndarray,
	decoder: Model,
	context: jnp.ndarray,
	history_observations: jnp.ndarray,
	history_next_observations: jnp.ndarray,
	deterministic: bool = True
):
	rng, dropout_key = jax.random.split(rng)
	decoder_input = jnp.concatenate((context, history_observations, history_next_observations), axis=-1)
	output = decoder.apply_fn(
		{"params": decoder.params},
		decoder_input,
		rngs={"dropout": dropout_key},
		deterministic=deterministic
	)
	return output
