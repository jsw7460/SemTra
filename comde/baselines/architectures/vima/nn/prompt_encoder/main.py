from typing import Optional, cast

import flax.linen as nn
import jax.numpy as jnp
from transformers.models.t5.modeling_flax_t5 import FlaxT5EncoderModel


class T5PromptEncoder(nn.Module):
    def setup(self) -> None:
        self.t5 = cast(FlaxT5EncoderModel, FlaxT5EncoderModel.from_pretrained("t5-base"))
        self.output_dim = self.t5.config.d_model
        self.input_dim = self.t5.config.d_model

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        attention_mask: Optional[jnp.ndarray] = None,
        batch_first: bool = False,
    ):
        """
        x: (L, B, E) if batch_first == False else (B, L, E)
        attention_mask: (B, L) or (B, 1, L) concurrently work with the causal mask
        """
        if not batch_first:
            x = x.transpose(0, 1)

        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.squeeze(axis=1)
            attention_mask = attention_mask.astype(jnp.float32)
        out = self.t5(
            input_ids=x,
            attention_mask=attention_mask,
        ).last_hidden_state
        if not batch_first:
            out = out.transpose((1, 0, 2))
        return out
