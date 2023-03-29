from functools import partial
from typing import Tuple, Dict

import jax
from jax import numpy as jnp

from comde.comde_modules.intent_emb.naive.architectures.vq.vq import PrimIntentEmbeddingVQ
from comde.utils.jax_utils.general import jnp_polyak_update
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params

EPSILON = 1E-8


@partial(jax.jit, static_argnames=("coef_reconstruction", "coef_decoder_aid", "coef_commitment", "coef_moving_avg"))
def intent_vq_updt(
	rng: jnp.ndarray,
	intent_emb: Model,
	low_policy: Model,
	vq_decoder: Model,
	language_operators: jnp.ndarray,  # [b, d]
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	skills: jnp.ndarray,  # [b, l, d]
	timesteps: jnp.ndarray,
	maskings: jnp.ndarray,
	coef_reconstruction: float,
	coef_decoder_aid: float,
	coef_commitment: float,
	coef_moving_avg: float
) -> Tuple[Model, Model, Dict]:

	rng, dropout_key = jax.random.split(rng)
	subseq_len = skills.shape[1]
	language_operators = jnp.expand_dims(language_operators, axis=1)
	language_operators = jnp.repeat(language_operators, axis=1, repeats=subseq_len)
	action_targets = actions * maskings[..., jnp.newaxis]

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
		"""
		VQ includes three loss:
			1. Reconstruction (Update encoder & decoder)
			2. Accelerate B.C of low policy (Update encoder)
			3. Moving Average (Update codebook): Update code by center of unquantized vector (by means of moving avg)
			4. Commitment (Update encoder): Move unquantized vector to its corresponding code

		Note:
			Third loss (moving avg) is not related with gradient. So it is applied outside the loss function
		"""
		uq_intent, q_intent = intent_emb.apply_fn(
			{"params": params},
			skills=skills,
			language_operators=language_operators,
			deterministic=False,
			training=True
		)  # [b, l, emb_dim]

		# Define auxiliary input
		_grad_flow_intent = uq_intent + jax.lax.stop_gradient(q_intent - uq_intent)  # [b, l, d]
		parameterized_skills = skills + _grad_flow_intent

		# 1. Reconstruction
		_decoder_input = _grad_flow_intent
		reconstructions = vq_decoder.apply_fn(
			{"params": vq_decoder.params},
			uq_intent,
			rngs={"dropout": dropout_key},
			deterministic=False,
			training=True
		)  # [b, l, d]
		reconstructions = reconstructions * maskings[..., jnp.newaxis]
		recon_target = jnp.concatenate((skills, language_operators), axis=-1) * maskings[..., jnp.newaxis]
		recon_loss = jnp.sum(jnp.mean((reconstructions - recon_target) ** 2, axis=-1)) / jnp.sum(maskings)
		recon_loss = coef_reconstruction * recon_loss

		# 2. Accelerate B.C
		action_preds = low_policy.apply_fn(
			{"params": low_policy.params},
			observations=observations,
			actions=actions,
			skills=parameterized_skills,
			timesteps=timesteps,
			maskings=maskings,
			rngs={"dropout": dropout_key},
			deterministic=False
		)
		bc_loss = coef_decoder_aid * (jnp.sum((action_preds - action_targets) ** 2) / jnp.sum(maskings))

		# 4. Commitment loss: Move unquantized to the center of cluster
		codebook = intent_emb.apply_fn(
			{"params": intent_emb.params},
			rngs={"dropout": dropout_key},
			method=PrimIntentEmbeddingVQ.get_current_embedding
		)
		codebook = jax.lax.stop_gradient(codebook)
		n_codebook = codebook.shape[0]
		codebook_dim = codebook.shape[-1]
		codebook_idxs, nearest_distance = intent_emb.apply_fn(
			{"params": intent_emb.params},
			vec=uq_intent,
			method=PrimIntentEmbeddingVQ.get_nearest_code_idxs
		)  # [b, l]
		codebook_idxs = codebook_idxs[..., jnp.newaxis]  # [b, l, 1]

		def body_fun(i: int, infos: Dict) -> Dict:
			# ith_idx = jnp.where(codebook_idxs == i, 1, 0)  # [b, l, 1]
			ith_idx = (codebook_idxs == i).astype("i4")  # [b, l, 1]
			# ith_mean = 'center' of the unquantized vectors which belong to 'i'th code
			ith_mean = jnp.sum(uq_intent * ith_idx, axis=(0, 1)) / (jnp.sum(ith_idx) + EPSILON)

			cur_loss = jnp.mean((ith_mean - codebook.at[i].get()) ** 2)
			infos["population"] = infos["population"].at[i].set(jnp.sum(ith_idx))
			infos["cluster_mean"] = infos["cluster_mean"].at[i].set(ith_mean)
			infos["cum_commit_loss"] = infos["cum_commit_loss"] + cur_loss
			return infos

		init_val = {
			"population": jnp.zeros((n_codebook,), dtype="i4"),
			"cluster_mean": jnp.zeros((n_codebook, codebook_dim)),
			"cum_commit_loss": 0.0
		}
		_fori_info = jax.lax.fori_loop(lower=0, upper=n_codebook, body_fun=body_fun, init_val=init_val)
		commitment_loss = _fori_info.pop("cum_commit_loss")
		commitment_loss = coef_commitment * commitment_loss

		loss = recon_loss + bc_loss + commitment_loss
		# loss = jnp.mean(uq_intent - jnp.zeros_like(uq_intent)) ** 2
		_info = {
			"intent/loss": loss,
			"intent/recon_loss": recon_loss,
			"intent/bc_loss": bc_loss,
			"intent/commitment_loss": commitment_loss,
			"intent/ith_idx": _fori_info["population"],
			"__nearest_distance": nearest_distance,
			# The followings are for moving average update
			"__codebook_idxs": codebook_idxs,
			"__uq_intent": uq_intent,
			"__decoder_input": _decoder_input,
			"__fori_info": _fori_info,
			"__parameterized_skills": parameterized_skills,
		}
		return loss, _info

	new_intent_emb, info1 = intent_emb.apply_gradient(loss_fn)

	updated_codebook = new_intent_emb.apply_fn(
		{"params": new_intent_emb.params},
		method=PrimIntentEmbeddingVQ.get_current_embedding
	)

	# 3. Update codebook by moving average
	# If there is no index corresponds to some code, it should be not updated.
	fori_info = info1.pop("__fori_info")
	maybe_pop_zero = fori_info["population"]
	maybe_pop_zero = jnp.where(maybe_pop_zero == 0, 1, 0)[..., jnp.newaxis]  # [n_codebook, 1]

	moving_avg_target = fori_info["cluster_mean"]
	moving_avg_target = maybe_pop_zero * updated_codebook + (1 - maybe_pop_zero) * moving_avg_target
	updated_codebook = jnp_polyak_update(source=updated_codebook, target=moving_avg_target, tau=coef_moving_avg)

	# Update codebook dictionary
	current_params = new_intent_emb.params
	current_params["codebook"].update({"embedding": updated_codebook})

	decoder_input = info1["__decoder_input"]
	decoder_target = jnp.concatenate((skills, language_operators), axis=-1) * maskings[..., jnp.newaxis]

	def decoder_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
		"""Update vq decoder"""
		reconstruction = vq_decoder.apply_fn(
			{"params": params},
			decoder_input,
			deterministic=False,
			training=True
		)  # [b, l, d]
		reconstruction = reconstruction * maskings[..., jnp.newaxis]

		loss = jnp.sum(jnp.mean((reconstruction - decoder_target) ** 2, axis=-1)) / jnp.sum(maskings)

		return loss, {"intent/decoder_loss": loss, "__decoder_target": decoder_target, "__recon": reconstruction, "__decoder_input": decoder_input}

	new_vq_decoder, info2 = vq_decoder.apply_gradient(decoder_loss_fn)
	return new_intent_emb, new_vq_decoder, {**info1, **info2}
