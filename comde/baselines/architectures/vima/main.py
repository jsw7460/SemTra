import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import comde.baselines.architectures.vima.nn as vnn
from comde.baselines.architectures.vima.nn.utils import identity


class VIMA(nn.Module):
    embed_dim: int
    prompt_dim: int
    xf_num_layers: int
    sattn_num_heads: int
    xattn_num_heads: int

    def setup(self) -> None:
        self.xattn_gpt = vnn.XAttnGPT(
            embed_dim=self.embed_dim,
            num_layers=self.xf_num_layers,
            num_heads=self.sattn_num_heads,
            dropout_rate=0.1,
            xattn_num_heads=self.xattn_num_heads,
            xattn_ff_expanding=4,
            xattn_num_positions=self.prompt_dim,
            use_geglu=True,
        )

        # self.object_encoder = vnn.ObjectEncoder(
        #     transformer_embed_dim=self.embed_dim,
        #     views=["front", "top"],
        #     vit_output_dim=768,
        #     vit_resolution=32,
        #     vit_patch_size=16,
        #     vit_width=768,
        #     vit_num_layers=4,
        #     vit_num_heads=24,
        #     bbox_mlp_hidden_dim=768,
        #     bbox_mlp_hidden_depth=2,
        # )

        # self.end_effector_encoder = vnn.Embedding(num_embeddings=2, features=2)

        self.obs_fusion_layer = nn.Dense(features=self.embed_dim)

        self.action_encoder = vnn.ActionEmbedding(
            output_dim=self.embed_dim,
            embed_dict={
                "pose0_position": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose0_rotation": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose1_position": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose1_rotation": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
            }
        )

        self.action_decoder = vnn.ActionDecoder(
            action_dims={
                "pose0_position": 1,
                "pose0_rotation": 1,
                "pose1_position": 1,
                "pose1_rotation": 1,
            },
            hidden_dim=512,
            hidden_depth=2,
            activation="relu",
            norm_type=None,
            last_layer_gain=0.01,
        )

        self.prompt_encoder = vnn.T5PromptEncoder()
        self.prompt_encoder_post_layer = (
            identity if self.embed_dim == self.prompt_encoder.output_dim
            else nn.Dense(features=self.embed_dim, use_bias=False)
        )

        self.prompt_object_post_layer = vnn.build_mlp(
            hidden_dim=768,
            hidden_depth=2,
            output_dim=768,
        )

        self._views = ["front", "top"]
        self._num_discrete_x_bins = 50
        self._num_discrete_y_bins = 100
        self._num_discrete_z_bins = 50
        self._num_discrete_rot_bins = 50

    def __call__(
        self,
        observations: jnp.ndarray,  # d_o
        observations_mask: jnp.ndarray,
        actions: jnp.ndarray,   # d_a
        prompt: jnp.ndarray,
        prompt_assets: jnp.ndarray,
        prompt_mask: jnp.ndarray,
        prompt_assets_mask: jnp.ndarray,
        deterministic: bool,
    ) -> jnp.ndarray:
        B, L_obs = observations.shape[:2]
        L_act = actions.shape[1] if actions is not None else 0
        L = max(L_obs, L_act)

        obs_tokens = self.obs_fusion_layer(observations)
        act_tokens = self.action_encoder({
            "pose0_position": actions[..., 0, jnp.newaxis],
            "pose0_rotation": actions[..., 1, jnp.newaxis],
            "pose1_position": actions[..., 2, jnp.newaxis],
            "pose1_rotation": actions[..., 3, jnp.newaxis],
        })

        obs_pad = L - L_obs
        act_pad = L - L_act
        indices = jnp.arange(L * 2).reshape(L, 2).T.reshape(-1)

        obs_tokens = jnp.pad(obs_tokens, ((0, 0), (0, obs_pad), (0, 0)))
        act_tokens = jnp.pad(act_tokens, ((0, 0), (0, act_pad), (0, 0)))
        tokens = jnp.concatenate((obs_tokens, act_tokens), axis=-2)[:, indices]

        obs_mask = jnp.pad(observations_mask, ((0, 0), (0, obs_pad)))
        act_mask = jnp.ones(shape=(B, L), dtype=jnp.bool_)
        masks = jnp.concatenate((obs_mask, act_mask), axis=-1)[:, indices]

        prompt = self.prompt_encoder(prompt, batch_first=True)
        prompt = self.prompt_encoder_post_layer(prompt)
        prompt = jnp.concatenate([
            prompt,
            prompt_assets,
        ], axis=-2)

        if prompt_mask.dtype != jnp.bool_:
            prompt_mask = prompt_mask > 0.5

        if prompt_assets_mask.dtype != jnp.bool_:
            prompt_assets_mask = prompt_assets_mask > 0.5

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
        return jnp.stack([
            outputs["pose0_position"],
            outputs["pose0_rotation"],
            outputs["pose1_position"],
            outputs["pose1_rotation"],
        ], axis=-1)
