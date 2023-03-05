import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params

EPS = 1E-6


@jax.jit
def mlp_termination_updt(
	rng: jnp.ndarray,
	mlp: Model,
	observations: jnp.ndarray,
	first_observations: jnp.ndarray,
	skills: jnp.ndarray,
	skills_done: jnp.ndarray
):
	"""
		skills: 0 (not done), 1 (done)
		mlp outputs:
			[:, 0]: log_prob of not done
			[:, 1]: log_prob of done
	"""
	rng, dropout_key = jax.random.split(rng)
	mlp_input = jnp.concatenate((observations, first_observations, skills), axis=-1)

	def loss_fn(params: Params):
		pred_dones = mlp.apply_fn(  # [batch_size, (subseq_len), 2]
			{"params": params},
			mlp_input,
			rngs={"dropuout": dropout_key},
			deterministic=False,
			training=True
		)
		# Note: the predictor model does not apply the softmax layer. So its output is interpreted as log-probability
		prob_pred_dones = jax.nn.softmax(pred_dones, axis=-1)
		prob_pred_dones = jnp.clip(prob_pred_dones, EPS, 1.0 - EPS)  # For stability
		log_p = jnp.log(prob_pred_dones[..., 1])
		log_one_minus_p = jnp.log(prob_pred_dones[..., 0])

		cross_entropy_loss = -jnp.mean((skills_done * log_p + (1 - skills_done) * log_one_minus_p))

		_infos = {"termination_policy/ce_loss": cross_entropy_loss, "__termination_policy/coord_vector": pred_dones}
		return cross_entropy_loss, _infos

	new_mlp, infos = mlp.apply_gradient(loss_fn)
	return new_mlp, infos
