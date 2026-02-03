"""
Stream-VAE for Z-Image World Model

Low-latency VAE encoding/decoding optimized for streaming inference.
Features:
- Tiled encoding/decoding for memory efficiency
- Async pipeline support
- Frame buffer management
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class StreamVAE(nn.Module):
    """Streaming VAE wrapper for low-latency encode/decode.

    Wraps the Z-Image VAE with optimizations for real-time streaming:
    - Tiled processing to reduce memory usage
    - Caching for repeated encoding
    - Async-friendly interface
    """

    def __init__(
        self,
        vae: Optional[nn.Module] = None,
        tile_size: int = 512,
        tile_overlap: int = 64,
        scaling_factor: float = 0.13025,  # Z-Image default
        use_tiling: bool = True,
        latent_channels: int = 16,  # Z-Image default, can be overridden
    ):
        super().__init__()

        self.vae = vae
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.scaling_factor = scaling_factor
        self.use_tiling = use_tiling
        self.latent_channels = latent_channels

        # Frame cache for repeated access
        self._latent_cache: dict[int, torch.Tensor] = {}
        self._cache_size = 16

    def set_vae(self, vae: nn.Module):
        """Set the underlying VAE model."""
        self.vae = vae

    def encode(
        self,
        images: torch.Tensor,
        use_cache: bool = True,
        cache_key: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode images to latent space.

        Args:
            images: Input images (batch, channels, height, width)
                   Values should be in [-1, 1]
            use_cache: Whether to cache the result
            cache_key: Optional key for caching

        Returns:
            Latents (batch, latent_channels, h//8, w//8)
        """
        if self.vae is None:
            raise RuntimeError("VAE not initialized. Call set_vae() first.")

        # Check cache
        if use_cache and cache_key is not None and cache_key in self._latent_cache:
            return self._latent_cache[cache_key]

        # Use tiled encoding for large images
        if self.use_tiling and (images.shape[-1] > self.tile_size or images.shape[-2] > self.tile_size):
            latents = self._tiled_encode(images)
        else:
            latents = self._encode(images)

        # Scale latents
        latents = latents * self.scaling_factor

        # Cache result
        if use_cache and cache_key is not None:
            self._update_cache(cache_key, latents)

        return latents

    def decode(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latents to images.

        Args:
            latents: Latent codes (batch, latent_channels, h, w)

        Returns:
            Images (batch, channels, height, width) in [-1, 1]
        """
        if self.vae is None:
            raise RuntimeError("VAE not initialized. Call set_vae() first.")

        # Unscale latents
        latents = latents / self.scaling_factor

        # Use tiled decoding for large latents
        if self.use_tiling:
            images = self._tiled_decode(latents)
        else:
            images = self._decode(latents)

        return images

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Standard encoding without tiling."""
        with torch.no_grad():
            posterior = self.vae.encode(images)
            if hasattr(posterior, "latent_dist"):
                latents = posterior.latent_dist.sample()
            elif hasattr(posterior, "sample"):
                latents = posterior.sample()
            else:
                latents = posterior
        return latents

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Standard decoding without tiling."""
        with torch.no_grad():
            images = self.vae.decode(latents)
            if hasattr(images, "sample"):
                images = images.sample
        return images

    def _tiled_encode(self, images: torch.Tensor) -> torch.Tensor:
        """Tiled encoding for memory efficiency.

        Processes image in overlapping tiles and blends results.
        """
        batch, channels, height, width = images.shape
        tile_size = self.tile_size
        overlap = self.tile_overlap

        # Calculate output size (8x downsampling)
        latent_h = height // 8
        latent_w = width // 8
        latent_tile = tile_size // 8
        latent_overlap = overlap // 8

        # Effective tile size after removing overlap
        effective_tile = latent_tile - 2 * latent_overlap

        # Calculate number of tiles
        n_tiles_h = max(1, (latent_h - latent_overlap) // effective_tile)
        n_tiles_w = max(1, (latent_w - latent_overlap) // effective_tile)

        # Initialize output
        latents = torch.zeros(
            batch, self.latent_channels, latent_h, latent_w,
            device=images.device, dtype=images.dtype
        )
        weights = torch.zeros(
            batch, 1, latent_h, latent_w,
            device=images.device, dtype=images.dtype
        )

        # Process tiles
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile boundaries in image space
                y_start = i * (tile_size - 2 * overlap)
                x_start = j * (tile_size - 2 * overlap)
                y_end = min(y_start + tile_size, height)
                x_end = min(x_start + tile_size, width)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                # Extract and encode tile
                tile = images[:, :, y_start:y_end, x_start:x_end]
                tile_latent = self._encode(tile)

                # Calculate boundaries in latent space
                ly_start = y_start // 8
                lx_start = x_start // 8
                ly_end = y_end // 8
                lx_end = x_end // 8

                # Create blending weights (linear ramp at edges)
                tile_weight = self._create_blend_weights(
                    tile_latent.shape[-2], tile_latent.shape[-1],
                    overlap // 8, device=images.device
                )

                # Accumulate
                latents[:, :, ly_start:ly_end, lx_start:lx_end] += (
                    tile_latent * tile_weight
                )
                weights[:, :, ly_start:ly_end, lx_start:lx_end] += tile_weight

        # Normalize by weights
        latents = latents / weights.clamp(min=1e-8)

        return latents

    def _tiled_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Tiled decoding for memory efficiency."""
        batch, channels, latent_h, latent_w = latents.shape
        tile_size = self.tile_size // 8  # Tile size in latent space
        overlap = self.tile_overlap // 8

        # Output size (8x upsampling)
        height = latent_h * 8
        width = latent_w * 8

        # Effective tile size
        effective_tile = tile_size - 2 * overlap

        # Calculate number of tiles
        n_tiles_h = max(1, (latent_h - overlap) // effective_tile)
        n_tiles_w = max(1, (latent_w - overlap) // effective_tile)

        # Initialize output
        images = torch.zeros(
            batch, 3, height, width,
            device=latents.device, dtype=latents.dtype
        )
        weights = torch.zeros(
            batch, 1, height, width,
            device=latents.device, dtype=latents.dtype
        )

        # Process tiles
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile boundaries in latent space
                ly_start = i * effective_tile
                lx_start = j * effective_tile
                ly_end = min(ly_start + tile_size, latent_h)
                lx_end = min(lx_start + tile_size, latent_w)
                ly_start = max(0, ly_end - tile_size)
                lx_start = max(0, lx_end - tile_size)

                # Extract and decode tile
                tile_latent = latents[:, :, ly_start:ly_end, lx_start:lx_end]
                tile_image = self._decode(tile_latent)

                # Calculate boundaries in image space
                y_start = ly_start * 8
                x_start = lx_start * 8
                y_end = ly_end * 8
                x_end = lx_end * 8

                # Create blending weights
                tile_weight = self._create_blend_weights(
                    tile_image.shape[-2], tile_image.shape[-1],
                    self.tile_overlap, device=latents.device
                )

                # Accumulate
                images[:, :, y_start:y_end, x_start:x_end] += (
                    tile_image * tile_weight
                )
                weights[:, :, y_start:y_end, x_start:x_end] += tile_weight

        # Normalize by weights
        images = images / weights.clamp(min=1e-8)

        return images

    def _create_blend_weights(
        self,
        height: int,
        width: int,
        overlap: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create blending weights with linear ramps at edges."""
        # Create 1D ramps
        ramp = torch.linspace(0, 1, overlap, device=device)

        # Height weights
        h_weights = torch.ones(height, device=device)
        if overlap > 0 and height > 2 * overlap:
            h_weights[:overlap] = ramp
            h_weights[-overlap:] = ramp.flip(0)

        # Width weights
        w_weights = torch.ones(width, device=device)
        if overlap > 0 and width > 2 * overlap:
            w_weights[:overlap] = ramp
            w_weights[-overlap:] = ramp.flip(0)

        # Combine into 2D weights
        weights = h_weights[:, None] * w_weights[None, :]
        return weights.unsqueeze(0).unsqueeze(0)

    def _update_cache(self, key: int, value: torch.Tensor):
        """Update cache with LRU eviction."""
        if len(self._latent_cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._latent_cache))
            del self._latent_cache[oldest_key]

        self._latent_cache[key] = value

    def clear_cache(self):
        """Clear the latent cache."""
        self._latent_cache.clear()

    def encode_sequence(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a sequence of frames.

        Args:
            images: Frame sequence (batch, num_frames, channels, height, width)

        Returns:
            Latent sequence (batch, num_frames, latent_channels, h//8, w//8)
        """
        batch, num_frames, c, h, w = images.shape

        # Flatten batch and frames
        images_flat = rearrange(images, "b f c h w -> (b f) c h w")

        # Encode all frames
        latents_flat = self.encode(images_flat, use_cache=False)

        # Reshape back
        latents = rearrange(
            latents_flat,
            "(b f) c h w -> b f c h w",
            b=batch, f=num_frames
        )

        return latents

    def decode_sequence(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a sequence of latent frames.

        Args:
            latents: Latent sequence (batch, num_frames, channels, h, w)

        Returns:
            Frame sequence (batch, num_frames, 3, height, width)
        """
        batch, num_frames, c, h, w = latents.shape

        # Flatten batch and frames
        latents_flat = rearrange(latents, "b f c h w -> (b f) c h w")

        # Decode all frames
        images_flat = self.decode(latents_flat)

        # Reshape back
        images = rearrange(
            images_flat,
            "(b f) c h w -> b f c h w",
            b=batch, f=num_frames
        )

        return images
