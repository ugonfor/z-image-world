"""Tests for model components."""

import pytest
import torch

from models import ActionEncoder, ActionSpace, CausalDiT


class TestActionEncoder:
    """Tests for ActionEncoder."""

    def test_action_space_num_actions(self):
        """Test action space has correct number of actions."""
        assert ActionSpace.num_actions() == 17

    def test_action_space_from_keyboard_idle(self):
        """Test idle action when no keys pressed."""
        action = ActionSpace.from_keyboard(set())
        assert action == ActionSpace.IDLE

    def test_action_space_from_keyboard_forward(self):
        """Test forward movement."""
        action = ActionSpace.from_keyboard({"w"})
        assert action == ActionSpace.MOVE_FORWARD

    def test_action_space_from_keyboard_diagonal(self):
        """Test diagonal movement."""
        action = ActionSpace.from_keyboard({"w", "a"})
        assert action == ActionSpace.MOVE_FORWARD_LEFT

    def test_action_encoder_forward(self):
        """Test action encoder forward pass."""
        encoder = ActionEncoder(
            num_actions=17,
            embedding_dim=64,
            hidden_dim=128,
        )

        # Single action
        actions = torch.tensor([[0]])  # (batch=1, frames=1)
        output = encoder(actions)

        assert output.shape == (1, 1, 128)  # (batch, frames, hidden_dim)

    def test_action_encoder_sequence(self):
        """Test action encoder with sequence of actions."""
        encoder = ActionEncoder(
            num_actions=17,
            embedding_dim=64,
            hidden_dim=128,
            num_frames=8,
        )

        # Sequence of actions
        actions = torch.randint(0, 17, (2, 8))  # (batch=2, frames=8)
        output = encoder(actions)

        assert output.shape == (2, 8, 128)

    def test_action_encoder_multi_action(self):
        """Test encoding multiple simultaneous actions."""
        encoder = ActionEncoder(
            num_actions=17,
            embedding_dim=64,
            hidden_dim=128,
        )

        primary = torch.tensor([0])  # forward
        secondary = torch.tensor([16])  # attack

        output = encoder.encode_multi_action(primary, secondary)
        assert output.shape == (1, 64)  # (batch, embedding_dim)


class TestCausalDiT:
    """Tests for CausalDiT model."""

    @pytest.fixture
    def small_dit(self):
        """Create a small DiT for testing."""
        return CausalDiT(
            in_channels=4,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            patch_size=2,
            num_frames=4,
            action_injection_layers=[1],
        )

    def test_dit_forward_single_frame(self, small_dit):
        """Test DiT forward pass with single frame."""
        batch_size = 2
        channels = 4
        height, width = 16, 16

        x = torch.randn(batch_size, channels, height, width)
        timesteps = torch.randint(0, 1000, (batch_size,))

        output, _ = small_dit(x, timesteps)

        assert output.shape == x.shape

    def test_dit_forward_sequence(self, small_dit):
        """Test DiT forward pass with frame sequence."""
        batch_size = 2
        num_frames = 4
        channels = 4
        height, width = 16, 16

        x = torch.randn(batch_size, num_frames, channels, height, width)
        timesteps = torch.randint(0, 1000, (batch_size,))

        output, _ = small_dit(x, timesteps)

        assert output.shape == x.shape

    def test_dit_with_action_conditioning(self, small_dit):
        """Test DiT forward pass with action conditioning."""
        batch_size = 2
        num_frames = 4
        channels = 4
        height, width = 16, 16
        hidden_dim = 64

        x = torch.randn(batch_size, num_frames, channels, height, width)
        timesteps = torch.randint(0, 1000, (batch_size,))
        action_cond = torch.randn(batch_size, num_frames, hidden_dim)

        output, _ = small_dit(x, timesteps, action_conditioning=action_cond)

        assert output.shape == x.shape

    def test_dit_with_kv_cache(self, small_dit):
        """Test DiT forward pass with KV cache."""
        batch_size = 1
        channels = 4
        height, width = 16, 16

        x = torch.randn(batch_size, channels, height, width)
        timesteps = torch.randint(0, 1000, (batch_size,))

        # First pass
        output1, cache1 = small_dit(x, timesteps, use_cache=True)

        # Second pass with cache
        output2, cache2 = small_dit(
            x, timesteps, kv_cache=cache1, use_cache=True
        )

        assert output1.shape == x.shape
        assert output2.shape == x.shape
        assert cache2 is not None


class TestCausalMask:
    """Tests for causal masking."""

    def test_causal_mask_shape(self):
        """Test causal mask has correct shape."""
        from models.causal_dit import get_causal_mask

        seq_len = 8
        mask = get_causal_mask(seq_len, torch.device("cpu"))

        assert mask.shape == (1, 1, seq_len, seq_len)

    def test_causal_mask_values(self):
        """Test causal mask has correct values."""
        from models.causal_dit import get_causal_mask

        seq_len = 4
        mask = get_causal_mask(seq_len, torch.device("cpu"))

        # Should be 0 on lower triangle, -inf on upper triangle
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert mask[0, 0, i, j] == float("-inf")
                else:
                    assert mask[0, 0, i, j] == 0
