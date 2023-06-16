import math
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp

from comde.comde_modules.low_policies.naive.architectures.base.gpt_modules import \
    FlaxConv1D
from comde.comde_modules.seq2seq.transformer.architectures.self_attention import \
    SelfAttention


class Block(nn.Module):
    num_positions: int
    embed_dim: int
    num_heads: int
    use_bias: bool
    layer_norm_epsilon: float
    attn_scale: bool = False
    mlp_dropout_rate: float = 0.1
    mlp_act: str = "gelu"

    def setup(self) -> None:
        self.attn = SelfAttention(
            self.embed_dim, self.num_heads, use_bias=self.use_bias
        )
        self.ln1 = nn.LayerNorm(epsilon=self.layer_norm_epsilon)
        self.mlp = MLP(self.embed_dim, 4 * self.embed_dim, dropout_rate=self.mlp_dropout_rate)
        self.ln2 = nn.LayerNorm(epsilon=self.layer_norm_epsilon)

    def forward(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        deterministic: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        attn_outputs = self.attn(
            x,
            mask=mask,
            deterministic=deterministic,
        )
        a = attn_outputs[0]

        n = self.ln1(x + a)
        m = self.mlp(n)
        h = self.ln2(n + m)

        return h, attn_outputs[1]


class MLP(nn.Module):
    embed_dim: int
    num_states: int
    dropout_rate: float = 0.1
    activate_function: str = "gelu"

    def setup(self) -> None:
        self.fc = FlaxConv1D(self.num_states)
        self.act = getattr(nn.activation, self.activate_function)
        if self.act == "gelu":
            self.gated_layer = nn.Dense(self.num_states, use_bias=False)
        else:
            self.gated_layer = None
        self.proj = FlaxConv1D(self.embed_dim)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.act(self.fc(x))
        if self.gated_layer:
            h *= self.gated_layer(x)
        h = self.proj(h)
        return self.dropout(h)


class XAttention(nn.Module):
    dim: int
    kv_n_positions: int
    num_heads: int = 1
    ff_expanding: int = 4
    detach_qk: bool = False
    fp32_logits: bool = True
    auto_add_pos_embd: bool = True
    use_gelu: bool = False

    def setup(self) -> None:
        self.layernorm = nn.LayerNorm(self.dim)
        self.query = nn.Dense(self.dim, use_bias=False)
        self.key_value = nn.Dense(self.dim * 2, use_bias=False)
        self.attention_out = nn.Dense(self.dim, use_bias=False)

        inner_dim = int(self.dim * self.ff_expanding)
        self.ln = nn.LayerNorm(self.dim)
        self.linear1 = nn.Dense(inner_dim, use_bias=False)
        self.act = nn.gelu
        self.linear2 = nn.Dense(self.dim, use_bias=False)
        if self.use_gelu:
            self.gated_layer = nn.Dense(self.dim, use_bias=False)
        else:
            self.gated_layer = None
        if self.auto_add_pos_embd:
            self.kv_positions_embed = nn.Embed(self.kv_n_positions, self.dim)
        else:
            self.kv_positions_embed = None

        self.kv_position_ids = jnp.arange(self.kv_n_positions)

    def transpose_for_scores(
        self,
        x: jnp.ndarray,
        channels_per_head: int,
    ) -> jnp.ndarray:
        new_x_shape = x.shape[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return jnp.transpose(x, (0, 2, 1, 3))

    def forward(
        self,
        q: jnp.ndarray,
        kv: jnp.ndarray,
        mask: jnp.ndarray,
        kv_position_ids: jnp.ndarray,
    ) -> jnp.ndarray:
        queries = self.layernorm(q)
        queries = self.query(queries)

        if kv_position_ids is None:
            assert kv.shape[1] <= self.kv_position_ids.shape[0]
            kv_position_ids = self.kv_position_ids[None, :kv.shape[1]]
        if self.kv_positions_embed is not None:
            kv_position_embeds = self.kv_positions_embed(kv_position_ids)
            kv = kv + kv_position_embeds
        keys, values = self.key_value(kv).chunk(2, dim=-1)

        # Reshape channels for multi-head attention.
        # We reshape from (batch_size, time, channels) to (batch_size, num_heads, time, channels per head)
        queries = self.transpose_for_scores(queries, self.dim_per_head)
        keys = self.transpose_for_scores(keys, self.dim_per_head)
        values = self.transpose_for_scores(values, self.dim_per_head)

        # Take the dot product between the queries and keys to get the raw attention scores.
        if self._fp32_logits:
            queries = queries.astype(jnp.float32)
            keys = keys.astype(jnp.float32)
        attention_scores = jnp.matmul(
            queries, keys.transpose(-1, -2)
        )  # (B, NH, T_q, T_k)

        batch_size, _, _, q_head_dim = queries.shape
        _, _, kv_len, v_head_dim = values.shape
        hiddens = self.num_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)

        if mask is not None:
            assert mask.shape == (batch_size, kv_len)
            assert mask.dtype == jnp.bool_
            mask = self.invert_attention_mask(mask)
            attention_mask = mask.astype(attention_scores.dtype)
            attention_scores = attention_scores + attention_mask

        if self._detach_qk:
            attention_scores = attention_scores.detach()
        # Normalize the attention scores to probabilities.
        attention_probs = nn.softmax(attention_scores, axis=-1)
        attention_probs = attention_probs.to(values.dtype)

        context_layer = jnp.matmul(attention_probs, values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output projection
        attention_output = self.attention_out(context_layer)
        attention_output = attention_output + q

        ff_output = self.ln(attention_output)
        ff_output = self.linear1(ff_output)
        ff_output = self.act(ff_output)
        if self.gated_layer is not None:
            ff_output = ff_output * self.gated_layer(attention_output)
        ff_output = self.linear2(ff_output)

        output = ff_output + attention_output
        return output
