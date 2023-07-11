from typing import Tuple, Dict
from functools import partial

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


@partial(jax.jit, static_argnames=("coef_skill_loss", "coef_param_loss"))
def sensor_encoder_transformer_updt(
	rng: jnp.ndarray,
	model: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	maskings: jnp.ndarray,
	skills_idxs: jnp.ndarray,
	params_idxs: jnp.ndarray,
	coef_skill_loss: float,
	coef_param_loss: float,
):
	rng, dropout_key = jax.random.split(rng)
	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
		predictions = model.apply_fn(
			{"params": params},
			observations=observations,
			actions=actions,
			maskings=maskings,
			deterministic=False,
			rngs={"dropout": dropout_key}
		)
		param_preds = predictions["param_preds"][:, -1]	# [b, n_param]
		skill_preds = predictions["skill_preds"][:, -1]	# [b, n_skill]

		param_preds = jnp.log(jax.nn.softmax(param_preds, axis=-1))
		skill_preds = jnp.log(jax.nn.softmax(skill_preds, axis=-1))

		param_pred_idxs = jnp.argmax(param_preds, axis=-1)
		skill_pred_idxs = jnp.argmax(skill_preds, axis=-1)

		param_loss = jnp.take_along_axis(param_preds, params_idxs[..., jnp.newaxis], axis=-1)
		skill_loss = jnp.take_along_axis(skill_preds, skills_idxs[..., jnp.newaxis], axis=-1)

		param_loss = - coef_param_loss * jnp.mean(param_loss)
		skill_loss = - coef_skill_loss * jnp.mean(skill_loss)

		param_match_ratio = jnp.mean(param_pred_idxs == params_idxs)
		skill_match_ratio = jnp.mean(skill_pred_idxs == skills_idxs)

		loss = param_loss + skill_loss

		_info = {
			"__param_preds": param_preds,
			"__skill_preds": skill_preds,
			"__skill_pred_idxs": skill_pred_idxs,
			"__param_pred_idxs": param_pred_idxs,
			"__skills_idxs": skills_idxs,
			"__params_idxs": params_idxs
		}
		if coef_skill_loss > 0:
			_info.update({
				"sensor_encoder/loss(skill)": skill_loss,
				"sensor_encoder/match(skill)%": skill_match_ratio * 100
			})

		if coef_param_loss > 0:
			_info.update({
			"sensor_encoder/loss(param)": param_loss,
			"sensor_encoder/match(param)%": param_match_ratio * 100
			})

		return loss, _info

	new_model, info = model.apply_gradient(loss_fn)
	return new_model, info