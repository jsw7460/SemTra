import flax.linen as nn
import jax
import jax.numpy as jnp

from comde.baselines.architectures.vima.nn.obj_encoder.vit.preprocess import \
    basic_image_array_preprocess

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
