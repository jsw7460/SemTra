from typing import Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from comde.baselines.architectures.vima.nn.obj_encoder.vit.main import \
    ViTEncoder
from comde.baselines.architectures.vima.nn.utils import build_mlp


class ObjectEncoder(nn.Module):
    transformer_embed_dim: int
    views: List[str]
    vit_resolution: int
    vit_patch_size: int
    vit_width: int
    vit_num_layers: int
    vit_num_heads: int
    vit_output_dim: int
    bbox_mlp_hidden_dim: int
    bbox_mlp_hidden_depth: int
    rng: jax.random.KeyArray

    bbox_max_height: int = 128
    bbox_max_width: int = 256

    def setup(self) -> None:
        self.views = sorted(self.views)
        self.cropped_image_encoder = ViTEncoder(
            output_dim=self.vit_output_dim,
            resolution=self.vit_resolution,
            patch_size=self.vit_patch_size,
            width=self.vit_width,
            num_layers=self.vit_num_layers,
            num_heads=self.vit_num_heads,
            rng=self.rng,
        )
        self.bbox_mlp = {
            view: build_mlp(
                hidden_dim=self.bbox_mlp_hidden_dim,
                hidden_depth=self.bbox_mlp_hidden_depth,
                output_dim=self.bbox_mlp_hidden_dim,
            )
            for view in self.views
        }
        self.pre_transformer_layer = {
            view: nn.Dense(self.transformer_embed_dim)
            for view in self.views
        }

    def forward(self, cropped_images: jnp.ndarray, bboxes: jnp.ndarray) -> jnp.ndarray:
        image_features = {
            view: self.cropped_image_encoder(cropped_images[view])
            for view in self.views
        }
        bbox_features = {
            view: bboxes[view].astype(jnp.float32)
            for view in self.views
        }
        _normalizer = jnp.asarray(
            [self.bbox_max_width, self.bbox_max_height, self.bbox_max_height, self.bbox_max_width],
            dtype=bboxes[self.views[0]].dtype,
        )
        _normalizer = jax.device_put(_normalizer, device=bboxes[self.views[0]].device())
        bbox_features = {
            view: bbox_features[view] / _normalizer
            for view in self.views
        }
        bbox_features = {
            view: self.bbox_mlp[view](bbox_features[view])
            for view in self.views
        }

        in_features = {
            view: self.pre_transformer_layer[view](
                jnp.concatenate([image_features[view], bbox_features[view]], axis=-1)
            )
            for view in self.views
        }
        outputs = jnp.concatenate([in_features[view] for view in self.views], axis=-2)
        return outputs

    @property
    def output_dim(self):
        return self.transformer_embed_dim


class GatoMultiViewRGBEncoder(nn.Module):
    embed_dim: int
    views: List[str]
    img_size: Tuple[int, int]
    vit_patch_size: int
    vit_width: int
    vit_num_layers: int
    vit_num_heads: int
    rng: jax.random.KeyArray

    def setup(self) -> None:
        self.views = sorted(self.views)
        self.cropped_img_encoder = GatoViTEncoder(
            img_size=self.img_size,
            patch_size=self.vit_patch_size,
            width=self.vit_width,
            num_layers=self.vit_num_layers,
            num_heads=self.vit_num_heads,
            output_dim=self.embed_dim,
            rng=self.rng,
        )

    def forward(self, rgb: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        input: (..., 3, H, W)
        output: (..., L * n_views, E)
        """
        img_feats = {
            view: self.cropped_img_encoder(rgb[view]) for view in self._views
        }  # dict of (..., L, E)
        out = jnp.concatenate(
            [img_feats[view] for view in self._views], axis=-2
        )  # (..., L * n_views, E)
        return out

    @property
    def img_patch_len(self):
        return self.cropped_img_encoder.vit.img_patch_len * len(self._views)
