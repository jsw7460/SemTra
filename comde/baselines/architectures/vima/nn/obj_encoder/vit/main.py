from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp

from comde.baselines.architectures.vima.nn.obj_encoder.vit.preprocess import \
    basic_image_array_preprocess
from comde.comde_modules.seq2seq.transformer.architectures.multihead_attention import MultiheadDotProductAttention

VIMA_IMG_MEAN = (0.3471, 0.3429, 0.3383)
VIMA_IMG_STD = (0.3011, 0.2961, 0.2956)


class ViTEncoder(nn.Module):
    output_dim: int
    resolution: int
    patch_size: int
    width: int
    num_layers: int
    num_heads: int
    rng: jax.random.KeyArray

    def setup(self) -> None:
        self.vit = VisionTransformer(
            resolution=self.resolution,
            patch_size=self.patch_size,
            width=self.width,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            output_dim=self.output_dim,
            rng=self.rng,
        )

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (..., 3, H, W)
        """
        assert x.ndim >= 4
        leading_dim = x.shape[:-3]
        x = basic_image_array_preprocess(x, mean=VIMA_IMG_MEAN, std=VIMA_IMG_STD)
        x = x.reshape(-1, *x.shape[-3:])
        x = self.vit(x)
        x = x.reshape(*leading_dim, self.output_dim)
        return x


class VisionTransformer(nn.Module):
    resolution: int
    patch_size: int
    width: int
    num_layers: int
    num_heads: int
    output_dim: int
    rng: jax.random.KeyArray

    def setup(self) -> None:
        self.conv1 = nn.Conv(
            features=self.width,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            use_bias=False,
        )

        scale = self.width ** (-0.5)
        self.cls_token = self.param(
            "token_embed",
            lambda: jax.random.normal(self.rng, (self.width,)),
        )
        self.pos_embed = self.param(
            "pos_embed",
            lambda: jax.random.normal(
                self.rng, ((self.resolution // self.patch_size) ** 2 + 1, self.width)
            )
        )
        self.ln_pre = nn.LayerNorm()
        self.blocks = nn.Sequential([
            ResidualAttentionBlock(self.width, self.num_heads)
            for _ in range(self.num_layers)
        ])
        self.ln_post = nn.LayerNorm()
        self.projection = self.param(
            "projection",
            lambda: jax.random.normal(self.rng, (self.width, self.output_dim)),
        )

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B = x.shape[0]
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.transpose(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = jnp.concatenate(
            [self.cls_token.repeat((B, 1, 1)), x], axis=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.transpose(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.transpose(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.projection is not None:
            x = x @ self.projection

        return x


class QuickGELU(nn.Module):
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * nn.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    embed_dim: int
    num_heads: int
    attention_mask: Optional[jnp.ndarray] = None

    def setup(self) -> None:
        self.attn = MultiheadDotProductAttention(
            self.embed_dim, self.num_heads, use_bias=True
        )
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.mlp = nn.Sequential([
            nn.Dense(self.embed_dim * 4),
            QuickGELU(),
            nn.Dense(self.embed_dim),
        ])
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.attn_mask = self.attention_mask

    def attention(self, x: jnp.ndarray):
        original_dtype = x.dtype
        self.attn_mask = (
            jax.device_put(
                self.attn_mask.astype(dtype=x.dtype), x.device(),
            )
            if self.attn_mask is not None
            else None
        )
        out = self.attn(
            x.astype(jnp.float32),
            x.astype(jnp.float32),
            x,
            need_weights=False,
            attn_mask=self.attn_mask,
        )[0]
        return out.astype(original_dtype)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ViTEncoderRectangular(nn.Module):
    output_dim: int
    image_size: Tuple[int, int]
    patch_size: int
    width: int
    num_layers: int
    num_heads: int

    def setup(self) -> None:
        self.vit = VisionTransformerRectangular(
            img_size=self.image_size,
            patch_size=self.patch_size,
            width=self.width,
            num_layers=self.num_layers,
            num_heads=self.heads,
            output_dim=self.output_dim,
        )

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (..., 3, H, W)
        """
        assert x.ndim >= 4
        leading_dim = x.shape[:-3]
        x = basic_image_array_preprocess(x, mean=VIMA_IMG_MEAN, std=VIMA_IMG_STD)
        x = x.reshape(-1, *x.shape[-3:])
        x = self.vit(x)
        x = x.reshape(*leading_dim, self.output_dim)
        return x


class VisionTransformerRectangular(nn.Module):
    img_size: Tuple[int, int]
    patch_size: int
    width: int
    num_layers: int
    num_heads: int
    output_dim: int

    def setup(self) -> None:
        self.conv1 = nn.Conv(
            features=self.width,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            use_bias=False,
        )

        scale = self.width ** (-0.5)
        self.cls_token = self.param(
            "cls_token",
            lambda: jax.random.normal(self.rng, (self.width,)),
        )
        n_patches_height = self.img_size[0] // self.patch_size
        n_patches_width = self.img_size[1] // self.patch_size
        self.pos_embed = self.param(
            "pos_embed",
            lambda: jax.random.normal(
                self.rng, (n_patches_height * n_patches_width + 1, self.width)
            )
        )
        self.ln_pre = nn.LayerNorm()
        self.blocks = nn.Sequential([
            ResidualAttentionBlock(self.width, self.num_heads)
            for _ in range(self.num_layers)
        ])
        self.ln_post = nn.LayerNorm()
        self.projection = self.param(
            "projection",
            lambda: jax.random.normal(self.rng, (self.width, self.output_dim)),
        )

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B = x.shape[0]
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.transpose(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = jnp.concatenate(
            [self.cls_token.repeat((B, 1, 1)), x], axis=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.transpose(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.transpose(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.projection is not None:
            x = x @ self.projection

        return x
