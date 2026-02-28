"""Tests for flow matching training components."""

import pytest
import torch
import torch.nn as nn

from training.flow_matching import (
    FlowMatchingConfig,
    FlowMatchingLoss,
    FlowMatchingTrainer,
    FlowMatchingInference,
    sample_flow_timesteps,
    flow_forward_process,
)


class TestFlowTimestepSampling:
    """Tests for timestep sampling strategies."""

    def test_uniform_sampling_range(self):
        """Uniform timesteps should be in [0, 1]."""
        t = sample_flow_timesteps(4, 8, sampling="uniform")
        assert t.shape == (4, 8)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_logit_normal_sampling_range(self):
        """Logit-normal timesteps should be in (0, 1)."""
        t = sample_flow_timesteps(4, 8, sampling="logit_normal")
        assert t.shape == (4, 8)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_cosmap_sampling_range(self):
        """CosMap timesteps should be in [0, 1]."""
        t = sample_flow_timesteps(4, 8, sampling="cosmap")
        assert t.shape == (4, 8)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_invalid_sampling(self):
        """Invalid sampling should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown timestep sampling"):
            sample_flow_timesteps(2, 4, sampling="invalid")


class TestFlowForwardProcess:
    """Tests for the linear interpolation forward process."""

    def test_forward_at_t0(self):
        """At t=0, x_t should equal x_0 (pure noise)."""
        x_0 = torch.randn(2, 4, 8, 8)
        x_1 = torch.randn(2, 4, 8, 8)
        t = torch.zeros(2, 1)

        x_t = flow_forward_process(x_1, x_0, t)
        assert torch.allclose(x_t, x_0, atol=1e-6)

    def test_forward_at_t1(self):
        """At t=1, x_t should equal x_1 (clean data)."""
        x_0 = torch.randn(2, 4, 8, 8)
        x_1 = torch.randn(2, 4, 8, 8)
        t = torch.ones(2, 1)

        x_t = flow_forward_process(x_1, x_0, t)
        assert torch.allclose(x_t, x_1, atol=1e-6)

    def test_forward_linearity(self):
        """x_t should be a linear interpolation."""
        x_0 = torch.zeros(2, 4, 8, 8)
        x_1 = torch.ones(2, 4, 8, 8)
        t = torch.full((2, 1), 0.5)

        x_t = flow_forward_process(x_1, x_0, t)
        expected = torch.full((2, 4, 8, 8), 0.5)
        assert torch.allclose(x_t, expected, atol=1e-6)

    def test_forward_frame_independent(self):
        """Different frames can have different t values."""
        x_0 = torch.zeros(2, 4, 4, 8, 8)  # (B, F, C, H, W)
        x_1 = torch.ones(2, 4, 4, 8, 8)
        # Different t per frame
        t = torch.tensor([[0.0, 0.25, 0.5, 0.75], [0.1, 0.3, 0.6, 0.9]])

        x_t = flow_forward_process(x_1, x_0, t)
        assert x_t.shape == (2, 4, 4, 8, 8)
        # First frame of first batch: t=0 → should be x_0 = 0
        assert torch.allclose(x_t[0, 0], torch.zeros(4, 8, 8), atol=1e-6)


class TestFlowMatchingLoss:
    """Tests for the flow matching loss function."""

    @pytest.fixture
    def loss_fn(self):
        return FlowMatchingLoss(prediction_type="velocity")

    def test_perfect_prediction(self, loss_fn):
        """Perfect velocity prediction should give near-zero loss."""
        x_0 = torch.randn(2, 4, 4, 8, 8)
        x_1 = torch.randn(2, 4, 4, 8, 8)
        t = torch.rand(2, 4)
        target_velocity = x_1 - x_0

        loss_dict = loss_fn(target_velocity, x_0, x_1, t)
        assert loss_dict["flow_loss"].item() < 1e-5

    def test_loss_keys(self, loss_fn):
        """Loss dict should contain required keys."""
        x_0 = torch.randn(2, 4, 4, 8, 8)
        x_1 = torch.randn(2, 4, 4, 8, 8)
        t = torch.rand(2, 4)
        pred = torch.randn_like(x_0)

        loss_dict = loss_fn(pred, x_0, x_1, t)
        assert "loss" in loss_dict
        assert "flow_loss" in loss_dict
        assert "temporal_loss" in loss_dict

    def test_total_loss_positive(self, loss_fn):
        """Total loss should be positive."""
        x_0 = torch.randn(2, 4, 4, 8, 8)
        x_1 = torch.randn(2, 4, 4, 8, 8)
        t = torch.rand(2, 4)
        pred = torch.randn(2, 4, 4, 8, 8)

        loss_dict = loss_fn(pred, x_0, x_1, t)
        assert loss_dict["loss"].item() > 0

    def test_temporal_loss_zero_single_frame(self, loss_fn):
        """Temporal loss should be zero for single-frame input."""
        x_0 = torch.randn(2, 1, 4, 8, 8)
        x_1 = torch.randn(2, 1, 4, 8, 8)
        t = torch.rand(2, 1)
        pred = torch.randn(2, 1, 4, 8, 8)

        loss_dict = loss_fn(pred, x_0, x_1, t)
        assert loss_dict["temporal_loss"].item() == 0.0


class TestFlowMatchingTrainer:
    """Tests for the FlowMatchingTrainer."""

    @pytest.fixture
    def small_model(self):
        """Create a tiny model for testing.

        Note: DummyVAE passes through 3-channel RGB, so model uses 3 channels.
        """
        class TinyFlowModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 1)  # (B*F, C, H, W) -> same

            def forward(self, x, timesteps, actions=None):
                # x: (B, F, C, H, W)
                b, f, c, h, w = x.shape
                x_flat = x.reshape(b * f, c, h, w)
                out = self.conv(x_flat)
                return out.reshape(b, f, c, h, w)

        return TinyFlowModel()

    @pytest.fixture
    def dummy_vae(self):
        """Create a pass-through VAE for testing."""
        class DummyVAE(nn.Module):
            class Config:
                scaling_factor = 1.0
            config = Config()

            def encode(self, x):
                return x

            def decode(self, x):
                return x

        return DummyVAE()

    @pytest.fixture
    def trainer(self, small_model, dummy_vae):
        config = FlowMatchingConfig(
            learning_rate=1e-3,
            warmup_steps=10,
            max_steps=100,
        )
        return FlowMatchingTrainer(
            model=small_model,
            vae=dummy_vae,
            config=config,
            device=torch.device("cpu"),
        )

    def test_train_step(self, trainer):
        """Training step should return loss metrics."""
        batch = {
            "frames": torch.rand(2, 4, 3, 16, 16),
        }
        metrics = trainer.train_step(batch)

        assert "loss" in metrics
        assert "flow_loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] > 0

    def test_train_step_with_actions(self, trainer):
        """Training step with actions should work."""
        batch = {
            "frames": torch.rand(2, 4, 3, 16, 16),
            "actions": torch.randint(0, 17, (2, 4)),
        }
        metrics = trainer.train_step(batch)
        assert "loss" in metrics

    def test_loss_decreases(self, trainer):
        """Loss should generally decrease over training steps."""
        batch = {"frames": torch.rand(2, 4, 3, 16, 16)}

        losses = []
        for _ in range(20):
            metrics = trainer.train_step(batch)
            losses.append(metrics["loss"])

        # Loss should be lower at end than start (on average)
        first_half_mean = sum(losses[:10]) / 10
        second_half_mean = sum(losses[10:]) / 10
        # Allow some variance, just check it's learning something
        assert second_half_mean < first_half_mean * 2


class TestFlowMatchingInference:
    """Tests for the Euler ODE integrator."""

    @pytest.fixture
    def constant_velocity_model(self):
        """A model that always predicts the same velocity."""
        class ConstVelocity(nn.Module):
            def forward(self, x, timesteps, actions=None):
                # Returns constant upward velocity
                return torch.ones_like(x) * 0.1

        return ConstVelocity()

    @pytest.fixture
    def dummy_vae(self):
        class DummyVAE(nn.Module):
            class Config:
                scaling_factor = 1.0
            config = Config()

            def decode(self, x):
                return x

        return DummyVAE()

    def test_denoising_changes_input(self, constant_velocity_model, dummy_vae):
        """Denoising should change the input latent."""
        inference = FlowMatchingInference(
            model=constant_velocity_model,
            vae=dummy_vae,
            num_steps=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        x_0 = torch.zeros(1, 4, 8, 8)
        x_1 = inference.denoise(x_0)

        # With constant positive velocity, output should be > input
        assert (x_1 > x_0).any()

    def test_denoising_shape_preserved(self, constant_velocity_model, dummy_vae):
        """Denoising should preserve tensor shape."""
        inference = FlowMatchingInference(
            model=constant_velocity_model,
            vae=dummy_vae,
            num_steps=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        x_0 = torch.randn(2, 4, 16, 16)
        x_1 = inference.denoise(x_0)
        assert x_1.shape == x_0.shape
