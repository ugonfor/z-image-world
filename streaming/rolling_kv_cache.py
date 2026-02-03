"""
Rolling KV Cache with Sink Tokens for Streaming Inference

Implements the rolling KV cache mechanism from StreamingLLM/StreamDiffusion
with sink tokens to maintain attention stability during long sequences.

Key features:
- Fixed-size sliding window cache
- Sink tokens preserved at the start for attention stability
- Efficient memory management for real-time inference
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CacheConfig:
    """Configuration for rolling KV cache."""

    max_length: int = 4096  # Maximum cache length
    num_sink_tokens: int = 4  # Number of sink tokens to preserve
    num_layers: int = 28  # Number of transformer layers
    num_heads: int = 32  # Number of attention heads
    head_dim: int = 128  # Dimension per head
    dtype: torch.dtype = torch.bfloat16


class SinkTokenManager:
    """Manages sink tokens for attention stability.

    Sink tokens are special tokens at the beginning of the sequence
    that accumulate attention mass and help maintain stability
    during long sequence generation.
    """

    def __init__(
        self,
        num_sink_tokens: int = 4,
        hidden_dim: int = 4096,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        self.num_sink_tokens = num_sink_tokens
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Learnable sink token embeddings
        self.sink_tokens = torch.nn.Parameter(
            torch.randn(num_sink_tokens, hidden_dim, dtype=dtype, device=self.device) * 0.02
        )

    def get_sink_tokens(self, batch_size: int) -> torch.Tensor:
        """Get sink tokens expanded for batch.

        Args:
            batch_size: Batch size

        Returns:
            Sink tokens of shape (batch, num_sink_tokens, hidden_dim)
        """
        return self.sink_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def prepend_sink_tokens(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Prepend sink tokens to input sequence.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)

        Returns:
            Tensor with sink tokens prepended (batch, num_sink + seq_len, hidden_dim)
        """
        batch_size = x.shape[0]
        sink = self.get_sink_tokens(batch_size)
        return torch.cat([sink, x], dim=1)


class RollingKVCache:
    """Rolling KV cache with sink token preservation.

    Maintains a fixed-size sliding window of key-value pairs while
    preserving sink tokens at the beginning for attention stability.

    The cache structure:
    [sink_tokens | ... | recent_tokens]
    ^-- preserved --^  ^-- rolling --^
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.max_length = config.max_length
        self.num_sink_tokens = config.num_sink_tokens
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dtype = config.dtype

        # Window size (excluding sink tokens)
        self.window_size = self.max_length - self.num_sink_tokens

        # Cache storage: list of (key, value) per layer
        self._cache: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(self.num_layers)
        ]

        # Current length of each layer's cache
        self._lengths: list[int] = [0 for _ in range(self.num_layers)]

        # Device
        self._device: Optional[torch.device] = None

    def reset(self):
        """Clear the cache."""
        self._cache = [None for _ in range(self.num_layers)]
        self._lengths = [0 for _ in range(self.num_layers)]

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a layer and return full key-value tensors.

        Args:
            layer_idx: Index of the transformer layer
            key: New key tensor (batch, num_heads, seq_len, head_dim)
            value: New value tensor (batch, num_heads, seq_len, head_dim)

        Returns:
            Full (cached + new) key and value tensors
        """
        if self._device is None:
            self._device = key.device

        batch_size, num_heads, new_len, head_dim = key.shape

        if self._cache[layer_idx] is None:
            # First update: just store
            self._cache[layer_idx] = (key.clone(), value.clone())
            self._lengths[layer_idx] = new_len
            return key, value

        cached_key, cached_value = self._cache[layer_idx]
        current_len = self._lengths[layer_idx]

        # Concatenate new tokens
        full_key = torch.cat([cached_key, key], dim=2)
        full_value = torch.cat([cached_value, value], dim=2)

        total_len = current_len + new_len

        # Check if we need to evict
        if total_len > self.max_length:
            # Preserve sink tokens and keep recent tokens
            num_to_keep = self.max_length

            # Extract sink tokens
            sink_key = full_key[:, :, : self.num_sink_tokens]
            sink_value = full_value[:, :, : self.num_sink_tokens]

            # Extract recent tokens (excluding what we'll evict)
            recent_start = total_len - (num_to_keep - self.num_sink_tokens)
            recent_key = full_key[:, :, recent_start:]
            recent_value = full_value[:, :, recent_start:]

            # Recombine
            full_key = torch.cat([sink_key, recent_key], dim=2)
            full_value = torch.cat([sink_value, recent_value], dim=2)

            total_len = num_to_keep

        # Update cache
        self._cache[layer_idx] = (full_key.clone(), full_value.clone())
        self._lengths[layer_idx] = total_len

        return full_key, full_value

    def get(
        self,
        layer_idx: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Get cached key-value for a layer.

        Args:
            layer_idx: Index of the transformer layer

        Returns:
            Cached (key, value) tuple or None if not cached
        """
        return self._cache[layer_idx]

    def get_length(self, layer_idx: int) -> int:
        """Get current cache length for a layer."""
        return self._lengths[layer_idx]

    def get_all(self) -> list[Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Get all layer caches."""
        return self._cache.copy()

    @property
    def is_initialized(self) -> bool:
        """Check if cache has been initialized."""
        return any(c is not None for c in self._cache)

    def get_attention_mask(
        self,
        query_len: int,
        layer_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Get attention mask for causal attention with cache.

        Args:
            query_len: Length of query sequence
            layer_idx: Layer index
            device: Target device

        Returns:
            Attention mask of shape (1, 1, query_len, key_len)
        """
        key_len = self._lengths[layer_idx] + query_len

        # Create causal mask
        mask = torch.triu(
            torch.ones(query_len, key_len, device=device),
            diagonal=key_len - query_len + 1,
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))

        # Allow attention to sink tokens
        if self.num_sink_tokens > 0:
            mask[:, : self.num_sink_tokens] = 0

        return mask.unsqueeze(0).unsqueeze(0)


class MultiFrameKVCache:
    """KV cache optimized for multi-frame video generation.

    Extends RollingKVCache with frame-aware management:
    - Tracks which frames are in cache
    - Supports frame-level eviction
    - Maintains temporal consistency
    """

    def __init__(
        self,
        config: CacheConfig,
        tokens_per_frame: int = 1200,  # ~480x640 with patch_size=2
    ):
        self.config = config
        self.tokens_per_frame = tokens_per_frame

        # Calculate max frames in cache
        self.max_frames = (config.max_length - config.num_sink_tokens) // tokens_per_frame

        # Underlying KV cache
        self.kv_cache = RollingKVCache(config)

        # Frame tracking
        self._frame_count = 0
        self._oldest_frame_idx = 0

    def reset(self):
        """Clear cache and reset frame tracking."""
        self.kv_cache.reset()
        self._frame_count = 0
        self._oldest_frame_idx = 0

    def add_frame(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add a single frame to the cache.

        Args:
            layer_idx: Transformer layer index
            key: Key tensor for one frame (batch, heads, tokens, head_dim)
            value: Value tensor for one frame

        Returns:
            Full key and value tensors including history
        """
        full_key, full_value = self.kv_cache.update(layer_idx, key, value)

        # Update frame tracking (only once per frame, not per layer)
        if layer_idx == 0:
            self._frame_count += 1
            if self._frame_count > self.max_frames:
                self._oldest_frame_idx += 1

        return full_key, full_value

    @property
    def num_cached_frames(self) -> int:
        """Number of frames currently in cache."""
        return min(self._frame_count, self.max_frames)

    @property
    def oldest_frame_index(self) -> int:
        """Index of the oldest frame in cache."""
        return self._oldest_frame_idx

    def get_frame_positions(self) -> tuple[int, int]:
        """Get the range of frame positions in cache.

        Returns:
            (start_frame_idx, end_frame_idx) tuple
        """
        return self._oldest_frame_idx, self._oldest_frame_idx + self.num_cached_frames
