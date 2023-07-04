from typing import Dict, Tuple

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


@jax.jit
def skilltoskill_model_updt(
	rng: jnp.ndarray,
	model: Model,
	source_skills: jnp.ndarray,  # [b, M, d],
	language_operators: jnp.ndarray,  # [b, d]
	target_skills: jnp.ndarray,  # [b, M, d]
	observations: jnp.ndarray,  # [b, l, d]
	n_target_skills: jnp.ndarray,  # [b, l]
	coef_skill_loss: float,
) -> Tuple[Model, Dict]:
	"""
		M: Maximum of the number of skills in the trajectory
	:return:

		Two types of gradients are flowed into model:
			1. Output of model ~ target skills (via similarity? or MSE?)
			2. Output of model is conditioned on low policy. Then model is tuned to minimize the BC loss.
	"""

	rng, carry_key, dropout_key = jax.random.split(rng, 3)

	batch_size = observations.shape[0]
	language_operators = language_operators[:, jnp.newaxis, ...]
	input_seq = jnp.concatenate((source_skills, language_operators), axis=1)

	max_possible_skills = target_skills.shape[1]

	def loss_fn(params: Params, ) -> Tuple[jnp.ndarray, Dict]:
		# 1. [Source skill] • [Lang. operator] -> Target skills
		model_output = model.apply_fn(
			{"params": params},
			sequence=input_seq,
			batch_size=batch_size,
			rngs={"init_carry": carry_key}
		)

		max_iter_len = model_output.shape[1]

		# The loss function flows only as much as the target skill.
		model_maskings = jnp.arange(max_iter_len)[jnp.newaxis, ...]
		model_maskings = jnp.repeat(model_maskings, repeats=batch_size, axis=0)

		# [b, max_iter_len, 1]
		model_maskings = jnp.where(model_maskings < n_target_skills.reshape(-1, 1), 1, 0)[..., jnp.newaxis]
		model_output = model_output * model_maskings
		target_skills_loss \
			= coef_skill_loss * jnp.mean((model_output[:, :max_possible_skills, ...] - target_skills) ** 2, axis=-1)

		target_skills_loss = jnp.sum(target_skills_loss, axis=1) / n_target_skills
		target_skills_loss = jnp.mean(target_skills_loss)

		model_loss = target_skills_loss
		_infos = {
			"model_target_skill_loss": target_skills_loss,
		}
		return model_loss, _infos

	new_model, info = model.apply_gradient(loss_fn)
	return new_model, info
