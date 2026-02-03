"""Tests for streaming components."""

import pytest
import torch

from streaming import RollingKVCache, CacheConfig, MotionAwareNoiseController


class TestRollingKVCache:
    """Tests for RollingKVCache."""

    @pytest.fixture
    def cache_config(self):
        """Create cache config for testing."""
        return CacheConfig(
            max_length=32,
            num_sink_tokens=4,
            num_layers=2,
            num_heads=4,
            head_dim=16,
        )

    def test_cache_initialization(self, cache_config):
        """Test cache initializes empty."""
        cache = RollingKVCache(cache_config)

        assert not cache.is_initialized
        assert cache.get(0) is None

    def test_cache_update(self, cache_config):
        """Test cache update adds key-values."""
        cache = RollingKVCache(cache_config)

        batch_size = 2
        seq_len = 8

        key = torch.randn(batch_size, cache_config.num_heads, seq_len, cache_config.head_dim)
        value = torch.randn(batch_size, cache_config.num_heads, seq_len, cache_config.head_dim)

        full_key, full_value = cache.update(0, key, value)

        assert cache.is_initialized
        assert cache.get_length(0) == seq_len
        assert full_key.shape == key.shape

    def test_cache_accumulation(self, cache_config):
        """Test cache accumulates across updates."""
        cache = RollingKVCache(cache_config)

        batch_size = 2
        seq_len = 8

        # First update
        key1 = torch.randn(batch_size, cache_config.num_heads, seq_len, cache_config.head_dim)
        value1 = torch.randn(batch_size, cache_config.num_heads, seq_len, cache_config.head_dim)
        cache.update(0, key1, value1)

        # Second update
        key2 = torch.randn(batch_size, cache_config.num_heads, seq_len, cache_config.head_dim)
        value2 = torch.randn(batch_size, cache_config.num_heads, seq_len, cache_config.head_dim)
        full_key, full_value = cache.update(0, key2, value2)

        assert cache.get_length(0) == 2 * seq_len
        assert full_key.shape[2] == 2 * seq_len

    def test_cache_eviction(self, cache_config):
        """Test cache evicts old entries when full."""
        cache = RollingKVCache(cache_config)

        batch_size = 1
        seq_len = 16  # Will need to evict after 2 updates

        # Fill cache beyond max_length
        for _ in range(3):
            key = torch.randn(batch_size, cache_config.num_heads, seq_len, cache_config.head_dim)
            value = torch.randn(batch_size, cache_config.num_heads, seq_len, cache_config.head_dim)
            full_key, full_value = cache.update(0, key, value)

        # Should be capped at max_length
        assert cache.get_length(0) == cache_config.max_length
        assert full_key.shape[2] == cache_config.max_length

    def test_cache_reset(self, cache_config):
        """Test cache reset clears all data."""
        cache = RollingKVCache(cache_config)

        # Add data
        key = torch.randn(1, cache_config.num_heads, 8, cache_config.head_dim)
        value = torch.randn(1, cache_config.num_heads, 8, cache_config.head_dim)
        cache.update(0, key, value)

        # Reset
        cache.reset()

        assert not cache.is_initialized
        assert cache.get_length(0) == 0


class TestMotionController:
    """Tests for MotionAwareNoiseController."""

    @pytest.fixture
    def controller(self):
        """Create controller for testing."""
        return MotionAwareNoiseController(
            base_noise_level=0.5,
            min_noise_level=0.1,
            max_noise_level=0.9,
            device=torch.device("cpu"),
        )

    def test_noise_level_no_prev_frame(self, controller):
        """Test noise level when no previous frame."""
        curr_frame = torch.randn(1, 3, 64, 64)
        noise_level = controller.compute_noise_level(None, curr_frame)

        assert noise_level == controller.base_noise_level

    def test_noise_level_range(self, controller):
        """Test noise level is within valid range."""
        prev_frame = torch.randn(1, 3, 64, 64)
        curr_frame = torch.randn(1, 3, 64, 64)

        noise_level = controller.compute_noise_level(prev_frame, curr_frame)

        assert controller.min_noise_level <= noise_level <= controller.max_noise_level

    def test_timestep_conversion(self, controller):
        """Test noise level to timestep conversion."""
        # Low noise
        timestep_low = controller.get_timestep_from_noise_level(0.0, num_inference_steps=4)
        assert timestep_low == 0

        # High noise
        timestep_high = controller.get_timestep_from_noise_level(1.0, num_inference_steps=4)
        assert timestep_high == 3

    def test_reset(self, controller):
        """Test controller reset."""
        # Change state
        prev_frame = torch.randn(1, 3, 64, 64)
        curr_frame = torch.randn(1, 3, 64, 64)
        controller.compute_noise_level(prev_frame, curr_frame)

        # Reset
        controller.reset()

        assert controller._prev_noise_level == controller.base_noise_level
        assert controller._prev_motion == 0.0
