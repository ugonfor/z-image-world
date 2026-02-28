"""Tests for offline evaluation metrics."""

import pytest
import torch

from scripts.evaluate import (
    compute_psnr,
    compute_ssim,
    compute_temporal_consistency,
)


class TestPSNR:
    """Tests for PSNR computation."""

    def test_identical_images_infinite_psnr(self):
        """Identical images should have infinite PSNR."""
        img = torch.rand(2, 3, 64, 64)
        psnr = compute_psnr(img, img)
        assert psnr == float("inf")

    def test_different_images_finite_psnr(self):
        """Different images should have finite PSNR."""
        img1 = torch.rand(2, 3, 64, 64)
        img2 = torch.rand(2, 3, 64, 64)
        psnr = compute_psnr(img1, img2)
        assert 0 < psnr < 100  # Typical range for image comparison

    def test_psnr_decreases_with_noise(self):
        """Higher noise → lower PSNR."""
        img = torch.rand(1, 3, 64, 64)
        noisy_low = img + 0.01 * torch.randn_like(img)
        noisy_high = img + 0.1 * torch.randn_like(img)
        noisy_low = noisy_low.clamp(0, 1)
        noisy_high = noisy_high.clamp(0, 1)

        psnr_low = compute_psnr(noisy_low, img)
        psnr_high = compute_psnr(noisy_high, img)
        assert psnr_low > psnr_high


class TestSSIM:
    """Tests for SSIM computation."""

    def test_identical_images_ssim_one(self):
        """Identical images should have SSIM of 1."""
        img = torch.rand(2, 3, 64, 64)
        ssim = compute_ssim(img, img)
        assert abs(ssim - 1.0) < 0.01

    def test_ssim_range(self):
        """SSIM should be in [-1, 1]."""
        img1 = torch.rand(2, 3, 64, 64)
        img2 = torch.rand(2, 3, 64, 64)
        ssim = compute_ssim(img1, img2)
        assert -1.0 <= ssim <= 1.0

    def test_ssim_similar_images_higher(self):
        """Similar images should have higher SSIM than dissimilar ones."""
        img = torch.rand(1, 3, 64, 64)
        similar = img + 0.05 * torch.randn_like(img)
        dissimilar = torch.rand_like(img)

        ssim_similar = compute_ssim(similar.clamp(0, 1), img)
        ssim_dissimilar = compute_ssim(dissimilar, img)
        assert ssim_similar > ssim_dissimilar

    def test_ssim_5d_input(self):
        """SSIM should work with (B, F, C, H, W) input."""
        img = torch.rand(2, 4, 3, 64, 64)
        ssim = compute_ssim(img, img)
        assert abs(ssim - 1.0) < 0.01


class TestTemporalConsistency:
    """Tests for temporal consistency metrics."""

    def test_static_video_low_consistency_value(self):
        """Static video (no motion) should have near-zero frame L2."""
        static_frame = torch.rand(1, 3, 64, 64)
        frames = static_frame.expand(10, -1, -1, -1)  # T=10 identical frames

        metrics = compute_temporal_consistency(frames)
        assert metrics["frame_l2"] < 1e-10

    def test_random_video_higher_consistency_value(self):
        """Random video should have higher frame L2 than static."""
        static_frames = torch.rand(1, 3, 64, 64).expand(10, -1, -1, -1)
        random_frames = torch.rand(10, 3, 64, 64)

        static_metrics = compute_temporal_consistency(static_frames)
        random_metrics = compute_temporal_consistency(random_frames)

        assert random_metrics["frame_l2"] > static_metrics["frame_l2"]

    def test_output_keys(self):
        """Metrics dict should contain required keys."""
        frames = torch.rand(8, 3, 32, 32)
        metrics = compute_temporal_consistency(frames)

        assert "frame_l2" in metrics
        assert "frame_lpips_proxy" in metrics

    def test_single_frame_returns_zero(self):
        """Single frame should return zero metrics."""
        frames = torch.rand(1, 3, 32, 32)
        metrics = compute_temporal_consistency(frames)
        assert metrics["frame_l2"] == 0.0

    def test_5d_input(self):
        """Should handle (B, T, C, H, W) input."""
        frames = torch.rand(2, 8, 3, 32, 32)
        metrics = compute_temporal_consistency(frames)
        assert "frame_l2" in metrics
        assert metrics["frame_l2"] >= 0.0
