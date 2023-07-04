from typing import Tuple, Dict

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params
from comde.comde_modules.low_policies.skill.architectures.skill_prompt_dt import PrimSkillPromptDT


@jax.jit
def skill_promptdt_updt(
	rng: jnp.ndarray,
	dt: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	skills: jnp.ndarray,
	prompts: jnp.ndarray,
	prompts_maskings: jnp.ndarray,
	timesteps: jnp.ndarray,
	maskings: jnp.ndarray,
	action_targets: jnp.ndarray,
):
	rng, dropout_key = jax.random.split(rng)

	action_dim = actions.shape[-1]

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
		_action_preds = dt.apply_fn(
			{"params": params},
			observations=observations,
			actions=actions,
			skills=skills,
			prompts=prompts,
			prompts_maskings=prompts_maskings,
			timesteps=timesteps,
			maskings=maskings,
			deterministic=False,
			rngs={"dropout": dropout_key},
			method=PrimSkillPromptDT.forward_with_all_components
		)
		action_preds = _action_preds[0]
		additional_info = _action_preds[2]

		action_preds = action_preds.reshape(-1, action_dim) * maskings.reshape(-1, 1)
		target = action_targets.reshape(-1, action_dim) * maskings.reshape(-1, 1)

		action_loss = jnp.sum(jnp.mean((action_preds - target) ** 2, axis=-1)) / jnp.sum(maskings)

		loss = action_loss

		# can pass skill loss either
		_infos = {
			"skill_decoder/mse_loss": loss,
			"__additional_info": additional_info,
			"__action_preds": action_preds,
			"__target": target
		}
		return loss, _infos

	new_dt, infos = dt.apply_gradient(loss_fn)
	return new_dt, infos
