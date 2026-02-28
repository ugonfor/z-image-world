"""Tests for streaming components."""

import pytest
import torch
import torch.nn as nn

from streaming import RollingKVCache, CacheConfig, MotionAwareNoiseController, SpatialFeatureCache


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


class TestSpatialFeatureCache:
    """Tests for SpatialFeatureCache."""

    NUM_LAYERS = 4
    B, N, D = 1, 16, 32  # Small dims for testing

    def _make_layer_feats(self):
        return [torch.randn(self.B, self.N, self.D) for _ in range(self.NUM_LAYERS)]

    def test_empty_cache(self):
        """New cache is empty."""
        cache = SpatialFeatureCache(num_layers=self.NUM_LAYERS, max_context_frames=3)
        assert not cache.is_populated
        assert cache.num_context_frames == 0
        assert cache.get_context_feats(0) is None

    def test_add_one_frame(self):
        """Adding a frame populates cache."""
        cache = SpatialFeatureCache(num_layers=self.NUM_LAYERS, max_context_frames=3)
        cache.add_frame(self._make_layer_feats())

        assert cache.is_populated
        assert cache.num_context_frames == 1
        ctx = cache.get_context_feats(0)
        assert ctx is not None
        assert ctx.shape == (self.B, 1, self.N, self.D)

    def test_multiple_frames_stacked(self):
        """Multiple frames stacked along dim=1."""
        cache = SpatialFeatureCache(num_layers=self.NUM_LAYERS, max_context_frames=3)
        for _ in range(3):
            cache.add_frame(self._make_layer_feats())

        ctx = cache.get_context_feats(1)
        assert ctx.shape == (self.B, 3, self.N, self.D)

    def test_eviction_at_max_frames(self):
        """Oldest frame is evicted when max exceeded."""
        max_ctx = 2
        cache = SpatialFeatureCache(num_layers=self.NUM_LAYERS, max_context_frames=max_ctx)
        for _ in range(4):
            cache.add_frame(self._make_layer_feats())

        assert cache.num_context_frames == max_ctx
        ctx = cache.get_context_feats(0)
        assert ctx.shape[1] == max_ctx

    def test_global_indices_monotone(self):
        """Global frame indices increase monotonically."""
        cache = SpatialFeatureCache(num_layers=self.NUM_LAYERS, max_context_frames=2)
        for _ in range(3):
            cache.add_frame(self._make_layer_feats())

        # oldest_frame_global_idx should be 1 (frame 0 evicted)
        assert cache.oldest_frame_global_idx == 1
        assert cache.next_frame_global_idx == 3

    def test_reset_clears_cache(self):
        """Reset clears all cached data."""
        cache = SpatialFeatureCache(num_layers=self.NUM_LAYERS, max_context_frames=3)
        cache.add_frame(self._make_layer_feats())
        cache.reset()

        assert not cache.is_populated
        assert cache.num_context_frames == 0
        assert cache._frames_added == 0

    def test_wrong_num_layers_raises(self):
        """Adding wrong number of layers raises assertion."""
        cache = SpatialFeatureCache(num_layers=self.NUM_LAYERS, max_context_frames=3)
        wrong_feats = [torch.randn(self.B, self.N, self.D) for _ in range(self.NUM_LAYERS - 1)]
        with pytest.raises(AssertionError):
            cache.add_frame(wrong_feats)


class TestTemporalAttentionWithContext:
    """Tests for TemporalAttention.forward_with_context."""

    B, N, D = 1, 8, 64
    NUM_HEADS = 4
    MAX_FRAMES = 16

    @pytest.fixture
    def temporal_attn(self):
        from models.zimage_world_model import TemporalAttention
        return TemporalAttention(
            hidden_dim=self.D,
            num_heads=self.NUM_HEADS,
            max_frames=self.MAX_FRAMES,
        )

    def test_no_context_output_shape(self, temporal_attn):
        """Single frame without context returns same shape."""
        x_new = torch.randn(self.B, self.N, self.D)
        out = temporal_attn.forward_with_context(x_new, None, new_frame_global_idx=0)
        assert out.shape == x_new.shape

    def test_with_context_output_shape(self, temporal_attn):
        """With context frames output matches new frame shape."""
        x_new = torch.randn(self.B, self.N, self.D)
        context_feats = torch.randn(self.B, 3, self.N, self.D)
        out = temporal_attn.forward_with_context(x_new, context_feats, new_frame_global_idx=3)
        assert out.shape == x_new.shape

    def test_zero_gamma_is_identity(self, temporal_attn):
        """With zero gamma, output equals input (skip connection only)."""
        # gamma is zero-initialized
        x_new = torch.randn(self.B, self.N, self.D)
        context_feats = torch.randn(self.B, 2, self.N, self.D)
        out = temporal_attn.forward_with_context(x_new, context_feats, new_frame_global_idx=2)
        # Output should be very close to input (gamma=0)
        assert torch.allclose(out, x_new, atol=1e-5), "With gamma=0, output must equal input"

    def test_causal_structure(self, temporal_attn):
        """Changing context affects output when weights are non-zero."""
        # Re-init all weights (normally zero-init for training stability)
        nn.init.constant_(temporal_attn.gamma, 1.0)
        nn.init.xavier_uniform_(temporal_attn.to_out.weight)
        nn.init.zeros_(temporal_attn.to_out.bias)

        x_new = torch.randn(self.B, self.N, self.D)

        # Two different contexts
        ctx_a = torch.randn(self.B, 2, self.N, self.D)
        ctx_b = ctx_a.clone()
        ctx_b[:, 0] = torch.randn(self.B, self.N, self.D)  # Modify oldest frame

        out_a = temporal_attn.forward_with_context(x_new, ctx_a, new_frame_global_idx=2)
        out_b = temporal_attn.forward_with_context(x_new, ctx_b, new_frame_global_idx=2)

        # Different context → different output (new frame attends to all context)
        assert not torch.allclose(out_a, out_b, atol=1e-5)

    def test_global_idx_beyond_max_frames_clamped(self, temporal_attn):
        """Global indices beyond max_frames are clamped, no IndexError."""
        x_new = torch.randn(self.B, self.N, self.D)
        context_feats = torch.randn(self.B, 2, self.N, self.D)
        # global idx > max_frames (16) should not crash
        out = temporal_attn.forward_with_context(x_new, context_feats, new_frame_global_idx=50)
        assert out.shape == x_new.shape


class TestSpatialCacheLifecycle:
    """Integration tests for SpatialFeatureCache lifecycle matching pipeline behavior."""

    NUM_LAYERS = 4
    B, N, D = 1, 8, 32

    def _make_layer_feats(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return [torch.randn(self.B, self.N, self.D) for _ in range(self.NUM_LAYERS)]

    def test_cache_advances_after_each_frame(self):
        """next_frame_global_idx increments correctly as frames are added."""
        cache = SpatialFeatureCache(num_layers=self.NUM_LAYERS, max_context_frames=4)
        assert cache.next_frame_global_idx == 0

        for i in range(5):
            cache.add_frame(self._make_layer_feats(seed=i))
            assert cache.next_frame_global_idx == i + 1

    def test_context_frames_count_caps_at_max(self):
        """num_context_frames never exceeds max_context_frames."""
        max_ctx = 3
        cache = SpatialFeatureCache(num_layers=self.NUM_LAYERS, max_context_frames=max_ctx)

        for i in range(6):
            cache.add_frame(self._make_layer_feats(seed=i))
            assert cache.num_context_frames <= max_ctx

    def test_get_context_feats_shape_after_eviction(self):
        """After eviction, get_context_feats still returns max_context frames."""
        max_ctx = 2
        cache = SpatialFeatureCache(num_layers=self.NUM_LAYERS, max_context_frames=max_ctx)

        for i in range(5):
            cache.add_frame(self._make_layer_feats(seed=i))

        ctx = cache.get_context_feats(0)
        assert ctx.shape == (self.B, max_ctx, self.N, self.D)

    def test_oldest_frame_global_idx_advances_on_eviction(self):
        """oldest_frame_global_idx advances as frames are evicted."""
        max_ctx = 2
        cache = SpatialFeatureCache(num_layers=self.NUM_LAYERS, max_context_frames=max_ctx)

        for i in range(5):
            cache.add_frame(self._make_layer_feats(seed=i))

        # After 5 frames with max 2: oldest = 5 - 2 = 3
        assert cache.oldest_frame_global_idx == 3
        assert cache.next_frame_global_idx == 5

    def test_position_formula_consistent(self):
        """Position indices computed in forward_with_context match cache state."""
        max_ctx = 3
        cache = SpatialFeatureCache(num_layers=self.NUM_LAYERS, max_context_frames=max_ctx)

        for i in range(4):  # Add 4 frames, cache holds 3
            cache.add_frame(self._make_layer_feats(seed=i))

        T = cache.num_context_frames
        new_idx = cache.next_frame_global_idx
        start_idx = new_idx - T  # Formula from forward_with_context
        expected_context_indices = list(range(start_idx, new_idx))

        assert expected_context_indices == cache.context_global_indices()

    def test_reset_restores_initial_state(self):
        """After reset, cache behaves identically to a fresh instance."""
        cache = SpatialFeatureCache(num_layers=self.NUM_LAYERS, max_context_frames=3)
        for i in range(5):
            cache.add_frame(self._make_layer_feats(seed=i))

        cache.reset()

        assert cache.next_frame_global_idx == 0
        assert cache.num_context_frames == 0
        assert not cache.is_populated
        for layer_i in range(self.NUM_LAYERS):
            assert cache.get_context_feats(layer_i) is None
