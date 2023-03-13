from typing import Dict, Tuple

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


@jax.jit
def skilltoskill_lstm_updt(
	rng: jnp.ndarray,
	lstm: Model,
	low_policy: Model,
	source_skills: jnp.ndarray,  # [b, M, d],
	language_operators: jnp.ndarray,  # [b, d]
	target_skills: jnp.ndarray,  # [b, M, d]
	observations: jnp.ndarray,  # [b, l, d]
	actions: jnp.ndarray,  # [b, l, d]
	timesteps: jnp.ndarray,
	maskings: jnp.ndarray,
	skills_order: jnp.ndarray,  # [b, l]
	n_target_skills: jnp.ndarray,  # [b, l]
	coef_skill_loss: float,
	coef_decoder_aid: float,
) -> Tuple[Model, Dict]:
	"""
		M: Maximum of the number of skills in the trajectory
	:return:

		Two types of gradients are flowed into LSTM:
			1. Output of LSTM ~ target skills (via similarity? or MSE?)
			2. Output of LSTM is conditioned on low policy. Then LSTM is tuned to minimize the BC loss.
	"""

	rng, carry_key, dropout_key = jax.random.split(rng, 3)

	batch_size = observations.shape[0]
	action_dim = actions.shape[-1]
	language_operators = language_operators[:, jnp.newaxis, ...]
	input_seq = jnp.concatenate((source_skills, language_operators), axis=1)

	max_possible_skills = target_skills.shape[1]

	def loss_fn(params: Params, ) -> Tuple[jnp.ndarray, Dict]:
		# 1. [Source skill] • [Lang. operator] -> Target skills
		lstm_output, _ = lstm.apply_fn(
			{"params": params},
			sequence=input_seq,
			batch_size=batch_size,
			rngs={"init_carry": carry_key}
		)

		max_iter_len = lstm_output.shape[1]

		# The loss function flows only as much as the target skill.
		lstm_maskings = jnp.arange(max_iter_len)[jnp.newaxis, ...]
		lstm_maskings = jnp.repeat(lstm_maskings, repeats=batch_size, axis=0)

		# [b, max_iter_len, 1]
		lstm_maskings = jnp.where(lstm_maskings < n_target_skills.reshape(-1, 1), 1, 0)[..., jnp.newaxis]
		lstm_output = lstm_output * lstm_maskings
		target_skills_loss \
			= coef_skill_loss * jnp.mean((lstm_output[:, :max_possible_skills, ...] - target_skills) ** 2, axis=-1)

		target_skills_loss = jnp.sum(target_skills_loss, axis=1) / n_target_skills
		target_skills_loss = jnp.mean(target_skills_loss)

		# 2. Predicted target skills should aid the skill decoder.
		pred_target_skills = jnp.take_along_axis(lstm_output, skills_order[..., jnp.newaxis], axis=1)  # [b, l, d]

		predictions = low_policy.apply_fn(
			{"params": low_policy.params},
			observations=observations,
			actions=actions,
			skills=pred_target_skills,
			timesteps=timesteps,
			maskings=maskings,
			deterministic=False,
			rngs={"dropout": dropout_key}
		)
		_, action_preds, _ = predictions

		action_preds = action_preds.reshape(-1, action_dim) * maskings.reshape(-1, 1)
		action_targets = actions.reshape(-1, action_dim) * maskings.reshape(-1, 1)

		low_policy_loss = jnp.sum((action_preds - action_targets) ** 2) / jnp.sum(maskings)
		low_policy_loss = coef_decoder_aid * low_policy_loss

		lstm_loss = target_skills_loss + low_policy_loss
		_infos = {
			"lstm_target_skill_loss": target_skills_loss,
			"lstm_low_policy_loss": low_policy_loss,
			"__pred_target_skills": pred_target_skills,
		}
		return lstm_loss, _infos

	new_lstm, info = lstm.apply_gradient(loss_fn)
	return new_lstm, info
