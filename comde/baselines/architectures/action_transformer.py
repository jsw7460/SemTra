from typing import Dict

import flax.linen as nn
import jax.numpy as jnp
import transformers

from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.low_policies.naive.architectures.customed.gpt_modules import FlaxGPT2ModuleWoTimePosEmb


class PrimActionTransformer(nn.Module):
	"""
		Basic Decision transformer model (No skill)
	"""
	gpt2_config: Dict
	action_dim: int
	hidden_size: int
	act_scale: float

	emb_obs = None
	emb_act = None
	emb_skill = None
	emb_lang = None
	emb_ln = None

	transformer = None
	pred_act = None

	def setup(self) -> None:
		gpt2_config = transformers.GPT2Config(**self.gpt2_config, n_embd=self.hidden_size)
		self.transformer = FlaxGPT2ModuleWoTimePosEmb(gpt2_config, dtype=jnp.float32)

		self.emb_obs = nn.Dense(self.hidden_size)
		self.emb_act = nn.Dense(self.hidden_size)
		self.emb_skill = nn.Dense(self.hidden_size)
		self.emb_lang = nn.Dense(self.hidden_size)
		self.emb_ln = nn.LayerNorm(self.hidden_size)

		pred_act = create_mlp(
			output_dim=self.action_dim,
			net_arch=[64, 64],
			squash_output=True
		)
		self.pred_act = Scaler(base_model=pred_act, scale=self.act_scale)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		observations: jnp.ndarray,  # [b, l, d]
		source_actions: jnp.ndarray,  # [b, 1, d]
		target_skill_sequence: jnp.ndarray,	# [b, M, d]: parameterized target skills
		language: jnp.ndarray,  # [b, d]
		obs_maskings: jnp.ndarray,  # [b, l] This is source masking
		act_maskings: jnp.ndarray,  # [b, l] This is source masking
		deterministic: bool = False
	):
		batch_size = target_skill_sequence.shape[0]
		n_target_skill = target_skill_sequence.shape[1]

		observations_emb = self.emb_obs(observations)  # [b, l, d]
		actions_emb = self.emb_act(source_actions)  # [b, 1, d]
		skill_emb = self.emb_skill(target_skill_sequence)	# [b, M, d]
		lang_emb = self.emb_lang(language)  # [b, d]
		lang_emb = jnp.expand_dims(lang_emb, axis=1)  # [b, 1, d]

		stacked_inputs = jnp.concatenate((observations_emb, skill_emb, lang_emb, actions_emb), axis=1)
		stacked_inputs = self.emb_ln(stacked_inputs)

		# obs: 5 skill: 3 lang: 1 action: 1

		skill_maskings = jnp.ones((batch_size, n_target_skill))
		lang_maskings = jnp.ones((batch_size, 1))

		stacked_masks = jnp.concatenate((obs_maskings, skill_maskings, lang_maskings, act_maskings), axis=1)

		transformer_outputs = self.transformer(
			hidden_states=stacked_inputs,
			attention_mask=stacked_masks,
			deterministic=deterministic
		)
		x = transformer_outputs["last_hidden_state"]	# [b, l + l' + 1, d]

		target_actions = self.pred_act(x[:, -1])	# [b, d]
		return target_actions
