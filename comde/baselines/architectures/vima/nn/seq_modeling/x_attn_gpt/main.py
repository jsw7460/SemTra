from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from comde.baselines.architectures.vima.nn.seq_modeling.x_attn_gpt.components import (
    Block, XAttention)


class XAttnGPT(nn.Module):
    embed_dim: int = 768
    num_positions: int = 512
    num_layers: int = 12
    num_heads: int = 12
    use_bias: bool = True
    layer_norm_epsilon: float = 1e-5
    attn_scale: bool = False
    dropout_rate: float = 0.1
    activation: str = "gelu"
    xattn_num_heads: int = 8
    xattn_ff_expanding: int = 4
    xattn_detach_qk: bool = False
    xattn_num_positions: int = 512
    use_geglu: bool = False

    def setup(self) -> None:
        self.positions_embed = nn.Embed(self.num_positions, self.embed_dim)
        self.xattn_positions_embed = nn.Embed(self.xattn_num_positions, self.embed_dim)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.h = [
            Block(
                num_positions=self.num_positions,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                use_bias=self.use_bias,
                layer_norm_epsilon=self.layer_norm_epsilon,
                attn_scale=self.attn_scale,
                mlp_dropout_rate=self.dropout_rate,
                mlp_act=self.activation,
            )
            for _ in range(self.num_layers)
        ]
        self.xattns = [
            XAttention(
                dim=self.embed_dim,
                kv_n_positions=self.xattn_num_positions,
                num_heads=self.xattn_num_heads,
                ff_expanding=self.xattn_ff_expanding,
                detach_qk=self.xattn_detach_qk,
                auto_add_pos_embd=False,
                use_gelu=self.use_geglu,
            )
            for _ in range(self.num_layers)
        ]

        self.position_ids = jnp.arange(self.num_positions)
        self.xattn_position_ids = jnp.arange(self.xattn_num_positions)
        
        self._input_checked = False
    
    def forward(
        self,
        *,
        obs_action_tokens: jnp.ndarray,
        obs_action_position_ids: Optional[jnp.ndarray] = None,
        prompt_tokens: jnp.ndarray,
        prompt_mask: Optional[jnp.ndarray] = None,
        prompt_position_ids: Optional[jnp.ndarray] = None,
        batch_first: bool = False,
        obs_action_masks: Optional[jnp.ndarray] = None,
    ):
        if not self._input_checked:
            self._check_input(
                obs_action_tokens,
                prompt_tokens,
                prompt_mask,
                batch_first,
                obs_action_masks,
            )
            self._input_checked = True
        if batch_first:
            B_oa, L_oa, E_oa = obs_action_tokens.shape
        else:
            L_oa, B_oa, E_oa = obs_action_tokens.shape
            obs_action_tokens = obs_action_tokens.transpose(0, 1)
            prompt_tokens = prompt_tokens.transpose(0, 1)
        input_shape = obs_action_tokens.shape[:-1]

        if not obs_action_position_ids:
            obs_action_position_ids = self.position_ids[jnp.newaxis, :input_shape[-1]]
        position_embeds = self.positions_embed(obs_action_position_ids)

        obs_action_tokens = obs_action_tokens + position_embeds
        obs_action_tokens = self.drop(obs_action_tokens)

        output_shape = input_shape + (obs_action_tokens.shape[-1],)

        assert prompt_tokens.shape[1] <= self.xattn_position_ids.shape[0]
        if not prompt_position_ids:
            prompt_position_ids = self.xattn_position_ids[jnp.newaxis, :prompt_tokens.shape[1]]
        prompt_position_embds = self.xattn_positions_embed(prompt_position_ids)
        prompt_tokens = prompt_tokens + prompt_position_embds

        if obs_action_masks:
            obs_action_masks = obs_action_masks[:, jnp.newaxis, jnp.newaxis, ...]
            obs_action_masks = obs_action_masks.astype(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            obs_action_masks = (1.0 - obs_action_masks) * jnp.finfo(self.dtype).min

        for self_attn, xattn in zip(self.h, self.xattns):
            obs_action_tokens = xattn(
                q=obs_action_tokens,
                kv=prompt_tokens,
                attention_mask=prompt_mask,
                kv_position_ids=None,
            )
            obs_action_tokens = self_attn(
                obs_action_tokens, attention_mask=obs_action_masks
            )[0]

        obs_action_tokens = obs_action_tokens.view(*output_shape)
        assert obs_action_tokens.shape == (B_oa, L_oa, E_oa)
        if not batch_first:
            obs_action_tokens = obs_action_tokens.transpose(0, 1)

        return obs_action_tokens

    def _check_input(
        self,
        obs_action_tokens: jnp.ndarray,
        prompt_tokens: jnp.ndarray,
        prompt_mask: Optional[jnp.ndarray] = None,
        batch_first: bool = False,
        obs_action_masks: Optional[jnp.ndarray] = None,
    ):
        assert obs_action_tokens.ndim == 3
        assert obs_action_tokens.dtype == jnp.float32
        assert prompt_tokens.ndim == 3
        assert prompt_tokens.dtype == jnp.float32

        if batch_first:
            B_oa, L_oa, E_oa = obs_action_tokens.shape
            B_p, L_p, E_p = prompt_tokens.shape
        else:
            L_oa, B_oa, E_oa = obs_action_tokens.shape
            L_p, B_p, E_p = prompt_tokens.shape
        assert B_oa == B_p
        assert E_oa == E_p
        B = B_oa

        if prompt_mask is not None:
            # fmt: off
            assert prompt_mask.shape == (B, L_p) or prompt_mask.shape == (B, 1, L_p), \
                f"Expect `prompt_mask` to have shape of either ({B, 1, L_p}) or ({B, L_p}), but got {prompt_mask.shape}"
            # fmt: on
            # a simple sanity check on the mask
            assert jnp.all(
                prompt_mask.sum(axis=-1) > 0
            ), "each source token should attend to at least one target token"
            assert prompt_mask.dtype == jnp.bool_
        if obs_action_masks is not None:
            assert obs_action_masks.shape == (B, L_oa)
            assert jnp.all(obs_action_masks.sum(axis=-1) > 0)
            assert obs_action_masks.dtype == jnp.bool_
