import flax.linen as nn
import jax.numpy as jnp
from transformers import FlaxBertModel

import comde.baselines.architectures.vima.nn as vnn
from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.common.utils import create_mlp

# prompt_encoder = FlaxT5EncoderModel.from_pretrained("t5-base")
prompt_encoder = FlaxBertModel.from_pretrained("bert-base-uncased")


class PrimVIMA(nn.Module):
	embed_dim: int
	prompt_dim: int
	action_dim: int
	xf_num_layers: int
	sattn_num_heads: int
	xattn_num_heads: int

	xattn_gpt = None
	obs_fusion_layer = None
	action_encoder = None

	emb_asset = None
	emb_timestep = None
	action_decoder = None
	prompt_encoder = None
	prompt_encoder_post_layer = None

	def setup(self) -> None:
		self.xattn_gpt = vnn.XAttnGPT(
			embed_dim=self.embed_dim,
			num_layers=self.xf_num_layers,
			num_heads=self.sattn_num_heads,
			dropout_rate=0.1,
			xattn_num_heads=self.xattn_num_heads,
			xattn_ff_dim=512,
			xattn_num_positions=self.prompt_dim,
			use_geglu=False,
		)

		self.emb_timestep = nn.Embed(1024, self.embed_dim)
		self.obs_fusion_layer = nn.Dense(features=self.embed_dim)
		self.action_encoder = nn.Dense(features=self.embed_dim)
		self.emb_asset = nn.Dense(features=self.embed_dim)
		action_decoder = create_mlp(output_dim=self.action_dim, net_arch=[], squash_output=True)
		self.action_decoder = Scaler(base_model=action_decoder, scale=1.35)
		self.prompt_encoder_post_layer = nn.Dense(features=self.embed_dim)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		observations: jnp.ndarray,  # d_o
		observations_mask: jnp.ndarray,
		actions: jnp.ndarray,  # d_a
		timesteps: jnp.ndarray,
		prompt: jnp.ndarray,
		prompt_assets: jnp.ndarray,
		prompt_mask: jnp.ndarray,
		prompt_assets_mask: jnp.ndarray,
		deterministic: bool,
	) -> jnp.ndarray:
		b, l_obs = observations.shape[:2]
		l_act = actions.shape[1]
		l = max(l_obs, l_act) * 2

		obs_tokens = self.obs_fusion_layer(observations)
		act_tokens = self.action_encoder(actions)
		timesteps_emb = self.emb_timestep(timesteps)
		obs_tokens = obs_tokens + timesteps_emb
		act_tokens = act_tokens + timesteps_emb

		tokens = jnp.empty(shape=(b, l, self.embed_dim), dtype=jnp.float32)
		masks = jnp.ones(shape=(b, l), dtype=jnp.bool_)

		tokens = tokens.at[:, 0::2, :].set(obs_tokens)
		tokens = tokens.at[:, 1::2, :].set(act_tokens)

		masks = masks.at[:, 0::2].set(observations_mask)

		# prompt = self.prompt_encoder(prompt, attention_mask=prompt_mask, batch_first=True)
		prompt = prompt_encoder(input_ids=prompt, attention_mask=prompt_mask)["last_hidden_state"]
		prompt = self.prompt_encoder_post_layer(prompt)
		prompt_assets = self.emb_asset(prompt_assets)
		prompt = jnp.concatenate([prompt, prompt_assets], axis=-2)

		prompt_mask = jnp.concatenate([
			prompt_mask,
			prompt_assets_mask,
		], axis=-1)

		position_ids = jnp.cumsum(masks, axis=0) - 1
		position_ids = position_ids.astype(jnp.int64)
		prompt_position_ids = (jnp.cumsum(prompt_mask, axis=1) - 1).astype(jnp.int64)

		tokens_out = self.xattn_gpt(
			obs_action_tokens=tokens,
			prompt_tokens=prompt,
			prompt_mask=prompt_mask,
			obs_action_masks=masks,
			obs_action_position_ids=position_ids,
			prompt_position_ids=prompt_position_ids,
			batch_first=True,
			deterministic=deterministic,
		)

		predicted_action_tokens = tokens_out[:, 0::2, :]
		outputs = self.action_decoder(predicted_action_tokens)

		return outputs
