"""
Spatial Feature Cache for Streaming ZImageWorldModel Inference

Caches the pre-temporal-attention Z-Image spatial hidden states for context
frames so they don't need to be recomputed on every denoising step.

Key insight: context frames are processed at t=0 (clean). Since Z-Image
spatial attention is per-frame (frames are independent in the B*F batch dim),
their spatial hidden states are deterministic given (pixel_content, t=0,
null_caption). Caching them avoids re-running ~30 transformer blocks per
context frame per denoising step.

Speedup: reduces cost from O((N_ctx+1)*N_steps) to O(N_steps + 1) Z-Image
passes per generated frame. For N_ctx=4, N_steps=2: 8→3 passes.
"""

from collections import deque
from typing import Optional

import torch


class SpatialFeatureCache:
    """Stores per-layer pre-temporal spatial hidden states for context frames.

    Each frame's features are stored as a list of tensors (one per Z-Image
    layer that has a TemporalAttention). Uses absolute global frame indices
    so that eviction doesn't corrupt position embeddings.

    Usage:
        cache = SpatialFeatureCache(num_temporal_layers=30, max_context=4)

        # Populate from initial/context frame:
        layer_feats = model._collect_spatial_features(latent)
        cache.add_frame(layer_feats)

        # Use in streaming forward:
        for layer_i in range(num_layers):
            ctx = cache.get_context_feats(layer_i)  # (B, T, N, D) or None
            new_global_idx = cache.next_frame_global_idx
    """

    def __init__(self, num_layers: int, max_context_frames: int = 4):
        self.num_layers = num_layers
        self.max_context_frames = max_context_frames

        # Per-layer circular buffer of (B, N, D) tensors
        # Each element of _feats[layer_i] is the pre-temporal img_tokens for one frame
        self._feats: list[deque] = [
            deque(maxlen=max_context_frames) for _ in range(num_layers)
        ]

        # Global frame counter (monotonically increasing, never resets on eviction)
        self._frames_added: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all cached features (call when set_initial_frame is called)."""
        for d in self._feats:
            d.clear()
        self._frames_added = 0

    def add_frame(self, layer_feats: list[torch.Tensor]):
        """Add a new frame's per-layer pre-temporal features to the cache.

        Args:
            layer_feats: List of (B, N, D) tensors, one per layer that has
                         TemporalAttention (length == num_layers).
        """
        assert len(layer_feats) == self.num_layers, (
            f"Expected {self.num_layers} layer features, got {len(layer_feats)}"
        )
        for layer_i, feat in enumerate(layer_feats):
            self._feats[layer_i].append(feat.detach())

        self._frames_added += 1

    def get_context_feats(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get stacked context features for a layer.

        Args:
            layer_idx: Z-Image layer index.

        Returns:
            (B, T, N, D) tensor with T context frames, or None if cache is empty.
        """
        feats = list(self._feats[layer_idx])
        if not feats:
            return None
        return torch.stack(feats, dim=1)  # (B, T, N, D)

    # ------------------------------------------------------------------
    # Frame position helpers (for absolute global indices)
    # ------------------------------------------------------------------

    @property
    def num_context_frames(self) -> int:
        """Number of frames currently in cache (limited by max_context_frames)."""
        return len(self._feats[0])

    @property
    def oldest_frame_global_idx(self) -> int:
        """Global index of the oldest frame in cache."""
        return max(0, self._frames_added - self.max_context_frames)

    @property
    def next_frame_global_idx(self) -> int:
        """Global index that the NEXT frame to be added will have."""
        return self._frames_added

    def context_global_indices(self) -> list[int]:
        """Global indices of the frames currently in cache, oldest first."""
        start = self.oldest_frame_global_idx
        return list(range(start, start + self.num_context_frames))

    @property
    def is_populated(self) -> bool:
        """True if at least one frame is cached."""
        return self._frames_added > 0
