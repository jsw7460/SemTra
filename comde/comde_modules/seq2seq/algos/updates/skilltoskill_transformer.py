from typing import Dict, Tuple
from functools import partial

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


@jax.jit
def skilltoskill_transformer_updt(
	rng: jnp.ndarray,
	tr: Model,  # transformer
	low_policy: Model,
	context: jnp.ndarray,	# [b, M + 1, d] : Cancatenation of source skills and language operator (calculated outside)
	target_skills: jnp.ndarray,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	skills: jnp.ndarray,
	skills_order: jnp.ndarray,  # [b, l]
	timesteps: jnp.ndarray,
	maskings: jnp.ndarray,
	n_source_skills: jnp.ndarray,  # [b, ]
	n_target_skills: jnp.ndarray,  # [b, ]
	start_token: jnp.ndarray,	# [d, ]
	end_token: jnp.ndarray, 	# [d, ]
	coef_intent: float,
	coef_skill: float
) -> Tuple[Model, Dict]:
	raise NotImplementedError("Continuous type skill matching is obsolete.")
	# rng, dropout_key = jax.random.split(rng)
	# # Note: Target skill을 한 칸 밀어줘야 함 (Due to start token)
	#
	# action_dim = actions.shape[-1]
	# skill_dim = skills.shape[-1]
	# batch_size = n_source_skills.shape[0]
	# context_maxlen = context.shape[1]
	#
	# start_token = jnp.broadcast_to(start_token, shape=(batch_size, 1, skill_dim))
	# end_token = jnp.broadcast_to(end_token, shape=(batch_size, 1, skill_dim))
	#
	# input_skills = jnp.concatenate((start_token, target_skills), axis=1)
	# target_skills = jnp.concatenate((target_skills, end_token), axis=1)
	#
	# target_max = target_skills.shape[1]
	#
	# # NOTE !!!! Since context vector contains language operator as well as source skills,
	# # we include equality to yeild a padding mask. (NOT a strict inequality)
	# ctx_padding_mask = jnp.arange(context_maxlen)
	# ctx_padding_mask = jnp.broadcast_to(ctx_padding_mask, (batch_size, context_maxlen))  # [b, l]
	# ctx_padding_mask = jnp.where(ctx_padding_mask <= n_source_skills.reshape(-1, 1), 1, 0)  # [b, l]
	#
	# actions_target = actions.copy().reshape(-1, action_dim) * maskings.reshape(-1, 1)
	#
	# def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
	# 	# 1. Intent is trained by behavior cloning of lower policy
	# 	model_output = tr.apply_fn(
	# 		{"params": params},
	# 		x=input_skills,
	# 		context=context,
	# 		mask=ctx_padding_mask,	# This is padding mask, not a causal mask.
	# 		rngs={"dropout": dropout_key}
	# 	)
	# 	pred_skills = model_output["pred_skills"]
	# 	pred_intent = model_output["pred_intents"]  # [b, M, d]
	#
	# 	intent_for_skill = jnp.take_along_axis(pred_intent, indices=skills_order[..., jnp.newaxis], axis=1)
	# 	intent_for_skill = jax.lax.stop_gradient(intent_for_skill)
	# 	intent_cond_skill = jnp.concatenate((skills, intent_for_skill), axis=-1)
	#
	# 	pred_actions = low_policy.apply_fn(
	# 		{"params": low_policy.params},
	# 		observations=observations,
	# 		actions=actions,
	# 		skills=intent_cond_skill,
	# 		timesteps=timesteps,
	# 		maskings=maskings,
	# 		deterministic=False,
	# 		rngs={"dropout": dropout_key}
	# 	)
	#
	# 	pred_actions = pred_actions.reshape(-1, action_dim) * maskings.reshape(-1, 1)
	# 	action_loss = coef_intent * jnp.sum(jnp.mean((pred_actions - actions_target) ** 2, axis=-1)) / jnp.sum(maskings)
	#
	# 	# 2. Learn target skills.
	# 	tgt_mask = jnp.arange(target_max)
	# 	tgt_mask = jnp.broadcast_to(tgt_mask, (batch_size, target_max))  # [b, M]
	# 	tgt_mask = jnp.where(tgt_mask <= n_target_skills[..., jnp.newaxis], 1, 0)[..., jnp.newaxis]
	#
	# 	pred_skills = pred_skills * tgt_mask
	# 	tgt_skills = target_skills * tgt_mask
	# 	# TODO: Replace MSE with CE (?)
	# 	skills_loss = coef_skill * jnp.sum(jnp.mean((pred_skills - tgt_skills) ** 2, axis=-1)) / jnp.sum(tgt_mask)
	#
	# 	loss = action_loss + skills_loss
	#
	# 	_info = {
	# 		"s2s/action_loss": action_loss,
	# 		"s2s/skill_loss(mse)": skills_loss,
	# 		"__tgt_mask": tgt_mask,
	# 		"__pred_skills": pred_skills,
	# 		"__tgt_skills": tgt_skills,
	# 		"__intent_for_skill": intent_for_skill,
	# 		"__intent_cond_skill": intent_cond_skill,
	# 		"__s2s/obs": observations,
	# 		"__s2s/act": actions,
	# 		"__s2s/skills": intent_cond_skill,
	# 		"__s2s/maskings": maskings,
	# 		"__s2s/ctx_padding_mask": ctx_padding_mask,
	# 		"__s2s/context_maxlen": context_maxlen,
	# 		"__skills": skills,
	# 	}
	# 	return loss, _info
	#
	# new_tr, info = tr.apply_gradient(loss_fn=loss_fn)
	#
	# return new_tr, info



# TODO: Cross entropy loss function.


@partial(jax.jit, static_argnames=("coef_intent", "coef_skill"))
def skilltoskill_transformer_ce_updt(
	rng: jnp.ndarray,
	tr: Model,  # transformer
	low_policy: Model,
	context: jnp.ndarray,	# [b, M + 1, d] : Cancatenation of source skills and language operator (calculated outside)
	target_skills: jnp.ndarray,
	target_skills_idxs: jnp.ndarray,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	skills: jnp.ndarray,
	skills_order: jnp.ndarray,  # [b, l]
	timesteps: jnp.ndarray,
	maskings: jnp.ndarray,
	n_source_skills: jnp.ndarray,  # [b, ]
	n_target_skills: jnp.ndarray,  # [b, ]
	start_token: jnp.ndarray,	# [d, ]
	end_token: jnp.ndarray, 	# [d, ]
	coef_intent: float,
	coef_skill: float
) -> Tuple[Model, Dict]:
	rng, dropout_key = jax.random.split(rng)
	# Note: Target skill을 한 칸 밀어줘야 함 (Due to start token)

	action_dim = actions.shape[-1]
	skill_dim = skills.shape[-1]
	batch_size = n_source_skills.shape[0]
	context_maxlen = context.shape[1]

	start_token = jnp.broadcast_to(start_token, shape=(batch_size, 1, skill_dim))

	input_skills = jnp.concatenate((start_token, target_skills), axis=1)

	target_skills_idxs = jnp.concatenate((target_skills_idxs, -1 + jnp.zeros((batch_size, 1), dtype="i4")), axis=-1)
	target_max = target_skills_idxs.shape[1]

	# NOTE !!!! Since context vector contains language operator as well as source skills,
	# we include equality to yeild a padding mask. (NOT a strict inequality)
	ctx_padding_mask = jnp.arange(context_maxlen)
	ctx_padding_mask = jnp.broadcast_to(ctx_padding_mask, (batch_size, context_maxlen))  # [b, l]
	ctx_padding_mask = jnp.where(ctx_padding_mask <= n_source_skills.reshape(-1, 1), 1, 0)  # [b, l]

	actions_target = actions.copy().reshape(-1, action_dim) * maskings.reshape(-1, 1)

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
		# 1. Intent is trained by behavior cloning of lower policy
		model_output = tr.apply_fn(
			{"params": params},
			x=input_skills,
			context=context,
			mask=ctx_padding_mask,	# This is padding mask, not a causal mask.
			rngs={"dropout": dropout_key}
		)
		pred_skills = model_output["pred_skills"]	# [b, M, n_skills]
		pred_intent = model_output["pred_intents"]  # [b, M, d]

		intent_for_skill = jnp.take_along_axis(pred_intent, indices=skills_order[..., jnp.newaxis], axis=1)
		intent_cond_skill = jnp.concatenate((skills, intent_for_skill), axis=-1)

		if coef_intent > 0:
			pred_actions = low_policy.apply_fn(
				{"params": low_policy.params},
				observations=observations,
				actions=actions,
				skills=intent_cond_skill,
				timesteps=timesteps,
				maskings=maskings,
				deterministic=False,
				rngs={"dropout": dropout_key}
			)

			pred_actions = pred_actions.reshape(-1, action_dim) * maskings.reshape(-1, 1)
			action_loss = coef_intent * jnp.sum(jnp.mean((pred_actions - actions_target) ** 2, axis=-1)) / jnp.sum(maskings)
		else:
			action_loss = 0.0

		# 2. Learn target skills.

		pred_skills = jax.nn.softmax(pred_skills, axis=-1)	# [b, M, n_skills] (Probability vec)
		tgt_skills = target_skills_idxs[..., jnp.newaxis]	# [b, M, 1]

		likelihood = jnp.log(pred_skills)		# [b, M, n_skills]
		likelihood = jnp.take_along_axis(likelihood, tgt_skills, axis=2)	# [b, M, 1]
		likelihood = jnp.squeeze(likelihood, axis=-1)

		tgt_mask = jnp.arange(target_max)
		tgt_mask = jnp.broadcast_to(tgt_mask, (batch_size, target_max))  # [b, M]
		tgt_mask = jnp.where(tgt_mask < n_target_skills[..., jnp.newaxis], 1, 0)	# [b, M]

		# ce_loss = - likelihood
		ce_loss = - likelihood * tgt_mask
		skills_loss = coef_skill * (jnp.sum(ce_loss) / jnp.sum(tgt_mask))

		loss = action_loss + skills_loss

		pred_skills_idxs = jnp.argmax(pred_skills, axis=-1)	# [b, M]
		pred_skills_idxs = jnp.where(tgt_mask == 1, pred_skills_idxs, -1)
		target_skills_answer = jnp.where(tgt_mask == 1, target_skills_idxs, -2)

		match_ratio = jnp.sum(pred_skills_idxs == target_skills_answer) / jnp.sum(tgt_mask)

		_info = {
			"s2s/action_loss": action_loss,
			"s2s/skill_loss(ce)": skills_loss,
			"s2s/match_ratio": match_ratio,
			"__tgt_mask": tgt_mask,
			"__pred_skills": jnp.log(pred_skills),
			"__tgt_skills": tgt_skills,
			"__intent_for_skill": intent_for_skill,
			"__intent_cond_skill": intent_cond_skill,
			"__s2s/obs": observations,
			"__s2s/act": actions,
			"__s2s/skills": intent_cond_skill,
			"__s2s/maskings": maskings,
			"__s2s/ctx_padding_mask": ctx_padding_mask,
			"__s2s/context_maxlen": context_maxlen,
			"__skills": skills,
			"__likelihood": likelihood,
			"__pred_skills_idxs": pred_skills_idxs,
			"__target_skills_answer": target_skills_answer
		}
		return loss, _info

	new_tr, info = tr.apply_gradient(loss_fn=loss_fn)

	return new_tr, info