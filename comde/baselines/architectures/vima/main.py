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
    rng: jax.random.KeyArray

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

        self.object_encoder = vnn.ObjectEncoder(
            transformer_embed_dim=self.embed_dim,
            views=["front", "top"],
            vit_output_dim=768,
            vit_resolution=32,
            vit_patch_size=16,
            vit_width=768,
            vit_num_layers=4,
            vit_num_heads=24,
            bbox_mlp_hidden_dim=768,
            bbox_mlp_hidden_depth=2,
            rng=self.rng,
        )

        self.end_effector_encoder = vnn.Embedding(num_embeddings=2, features=2)

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
                "pose0_position": [50, 100],
                "pose0_rotation": [50] * 4,
                "pose1_position": [50, 100],
                "pose1_rotation": [50] * 4,
            },
            hidden_dim=512,
            hidden_depth=2,
            activation="relu",
            norm_type=None,
            last_layer_gain=0.01,
            rng=self.rng,
        )

        self.prompt_embedding = vnn.WordEmbedding()
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

    def forward(
        self,
        observations: jnp.ndarray,
        observations_mask: jnp.ndarray,
        actions: jnp.ndarray,
        prompt: jnp.ndarray,
        prompt_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        L_obs, B = observations.shape[:2]
        L_action = 0 if actions is None else actions.shape[0]
        n_max_objs = observations.shape[-2]
        L = L_obs * n_max_objs + L_action

        tokens = jnp.empty(shape=(L, B, self.embed_dim), dtype=jnp.float32)
        tokens = jax.device_put(tokens, observations.device())

        masks = jnp.ones(shape=(L, B), dtype=jnp.bool_)
        masks = jax.device_put(masks, observations.device())

        obs_token = einops.rearrange(observations, "L B Q E -> B L Q E")
        obs_token = einops.rearrange(obs_token, "B L Q E -> B (L Q) E")
        obs_token = einops.rearrange(obs_token, "B L E -> L B E")

        obs_mask = einops.rearrange(observations_mask, "L B Q -> B L Q")
        obs_mask = einops.rearrange(obs_mask, "B L Q -> B (L Q)")
        obs_mask = einops.rearrange(obs_mask, "B L -> L B")

        for q in range(n_max_objs):
            tokens[q :: n_max_objs + 1] = obs_token[q::n_max_objs]
            masks[q :: n_max_objs + 1] = obs_mask[q::n_max_objs]
        if actions is not None:
            tokens[n_max_objs::(n_max_objs + 1)] = actions

        position_ids = jnp.cumsum(masks, axis=0) - 1
        position_ids = position_ids.astype(jnp.int64)
        prompt_position_ids = jnp.cumsum(prompt_mask, axis=1) - 1

        tokens_out = self.xattn_gpt(
            obs_action_tokens=tokens,
            prompt_tokens=prompt,
            prompt_mask=prompt_mask,
            obs_action_masks=masks.transpose(0, 1),
            obs_action_position_ids=position_ids.transpose(0, 1),
            prompt_position_ids=prompt_position_ids,
        )

        predicted_action_tokens = tokens_out[n_max_objs - 1 :: n_max_objs + 1]
        return predicted_action_tokens
