"""
Motion-Aware Noise Controller for Streaming Video Diffusion

Controls noise levels based on motion estimation between frames,
following the approach from StreamDiffusion V2.

Key features:
- Motion estimation via optical flow
- Adaptive noise scheduling based on motion magnitude
- Temporal consistency preservation
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpticalFlowEstimator(nn.Module):
    """Lightweight optical flow estimation for motion detection.

    Uses a simple convolutional approach for real-time estimation.
    For production, consider using RAFT or other dedicated flow networks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
    ):
        super().__init__()

        # Feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Flow prediction
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 2, 3, padding=1),
        )

    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate optical flow from frame1 to frame2.

        Args:
            frame1: First frame (batch, 3, height, width)
            frame2: Second frame (batch, 3, height, width)

        Returns:
            Flow field (batch, 2, height//2, width//2)
        """
        # Concatenate frames
        x = torch.cat([frame1, frame2], dim=1)

        # Extract features and predict flow
        features = self.encoder(x)
        flow = self.flow_head(features)

        return flow

    def compute_motion_magnitude(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute motion magnitude between frames.

        Args:
            frame1: First frame (batch, 3, height, width)
            frame2: Second frame (batch, 3, height, width)

        Returns:
            Motion magnitude per sample (batch,)
        """
        flow = self.forward(frame1, frame2)
        magnitude = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
        return magnitude.mean(dim=(-2, -1))  # Average over spatial dims


class MotionAwareNoiseController:
    """Controls diffusion noise based on motion estimation.

    Adapts the noise schedule based on detected motion:
    - High motion: More noise needed for generation flexibility
    - Low motion: Less noise to preserve temporal consistency

    This helps balance quality vs responsiveness in streaming generation.
    """

    def __init__(
        self,
        base_noise_level: float = 0.5,
        min_noise_level: float = 0.1,
        max_noise_level: float = 0.9,
        motion_threshold_low: float = 0.05,
        motion_threshold_high: float = 0.3,
        smoothing_factor: float = 0.7,
        device: Optional[torch.device] = None,
    ):
        """Initialize noise controller.

        Args:
            base_noise_level: Default noise level when no motion info
            min_noise_level: Minimum noise level (low motion)
            max_noise_level: Maximum noise level (high motion)
            motion_threshold_low: Motion below this is considered static
            motion_threshold_high: Motion above this is considered high
            smoothing_factor: Temporal smoothing for noise level changes
            device: Target device
        """
        self.base_noise_level = base_noise_level
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.motion_threshold_low = motion_threshold_low
        self.motion_threshold_high = motion_threshold_high
        self.smoothing_factor = smoothing_factor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Flow estimator (lazy init to avoid loading if not used)
        self._flow_estimator: Optional[OpticalFlowEstimator] = None

        # State for temporal smoothing
        self._prev_noise_level = base_noise_level
        self._prev_motion = 0.0

    @property
    def flow_estimator(self) -> OpticalFlowEstimator:
        """Lazy-load flow estimator."""
        if self._flow_estimator is None:
            self._flow_estimator = OpticalFlowEstimator().to(self.device)
            self._flow_estimator.eval()
        return self._flow_estimator

    def compute_noise_level(
        self,
        prev_frame: Optional[torch.Tensor],
        curr_frame: torch.Tensor,
        action: Optional[int] = None,
    ) -> float:
        """Compute appropriate noise level based on motion.

        Args:
            prev_frame: Previous frame (batch, 3, H, W) or None
            curr_frame: Current frame (batch, 3, H, W)
            action: Optional action index for action-based adjustment

        Returns:
            Noise level in [min_noise_level, max_noise_level]
        """
        if prev_frame is None:
            return self.base_noise_level

        # Estimate motion
        with torch.no_grad():
            motion = self.flow_estimator.compute_motion_magnitude(
                prev_frame, curr_frame
            ).mean().item()

        # Apply temporal smoothing to motion
        smoothed_motion = (
            self.smoothing_factor * self._prev_motion +
            (1 - self.smoothing_factor) * motion
        )
        self._prev_motion = smoothed_motion

        # Map motion to noise level
        noise_level = self._motion_to_noise(smoothed_motion)

        # Optional: Adjust based on action
        if action is not None:
            noise_level = self._adjust_for_action(noise_level, action)

        # Temporal smoothing for noise level
        smoothed_noise = (
            self.smoothing_factor * self._prev_noise_level +
            (1 - self.smoothing_factor) * noise_level
        )
        self._prev_noise_level = smoothed_noise

        return smoothed_noise

    def _motion_to_noise(self, motion: float) -> float:
        """Map motion magnitude to noise level.

        Uses linear interpolation between thresholds.
        """
        if motion < self.motion_threshold_low:
            return self.min_noise_level
        elif motion > self.motion_threshold_high:
            return self.max_noise_level
        else:
            # Linear interpolation
            t = (motion - self.motion_threshold_low) / (
                self.motion_threshold_high - self.motion_threshold_low
            )
            return self.min_noise_level + t * (self.max_noise_level - self.min_noise_level)

    def _adjust_for_action(self, noise_level: float, action: int) -> float:
        """Adjust noise level based on action type.

        Some actions (like jump, attack) typically cause more motion
        and may need higher noise levels.
        """
        # High-motion actions
        high_motion_actions = {13, 15, 16}  # jump, interact, attack

        if action in high_motion_actions:
            # Increase noise for high-motion actions
            noise_level = min(
                self.max_noise_level,
                noise_level + 0.1
            )

        return noise_level

    def get_timestep_from_noise_level(
        self,
        noise_level: float,
        num_inference_steps: int = 4,
        scheduler_type: str = "ddpm",
    ) -> int:
        """Convert noise level to diffusion timestep.

        Args:
            noise_level: Noise level in [0, 1]
            num_inference_steps: Total number of denoising steps
            scheduler_type: Type of noise scheduler

        Returns:
            Starting timestep for denoising
        """
        # For most schedulers, higher noise = earlier timestep
        max_timestep = num_inference_steps - 1
        timestep = int(noise_level * max_timestep)
        return min(max_timestep, max(0, timestep))

    def reset(self):
        """Reset internal state."""
        self._prev_noise_level = self.base_noise_level
        self._prev_motion = 0.0


class AdaptiveNoiseScheduler:
    """Combines motion-aware control with diffusion scheduler.

    Wraps a standard diffusion scheduler to provide adaptive
    noise levels based on motion estimation.
    """

    def __init__(
        self,
        base_scheduler,  # Diffusion scheduler (e.g., DDPMScheduler)
        motion_controller: Optional[MotionAwareNoiseController] = None,
        num_inference_steps: int = 4,
    ):
        self.base_scheduler = base_scheduler
        self.motion_controller = motion_controller or MotionAwareNoiseController()
        self.num_inference_steps = num_inference_steps

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        prev_frame: Optional[torch.Tensor] = None,
        action: Optional[int] = None,
    ) -> torch.Tensor:
        """Add noise to samples with motion-aware adjustment.

        Args:
            original_samples: Clean samples
            noise: Noise tensor
            timesteps: Diffusion timesteps
            prev_frame: Previous frame for motion estimation
            action: Current action

        Returns:
            Noisy samples
        """
        # Get motion-based noise level
        noise_level = self.motion_controller.compute_noise_level(
            prev_frame=prev_frame,
            curr_frame=original_samples,
            action=action,
        )

        # Adjust timesteps based on noise level
        adjusted_timesteps = self.motion_controller.get_timestep_from_noise_level(
            noise_level, self.num_inference_steps
        )

        # Use base scheduler with adjusted timesteps
        if hasattr(self.base_scheduler, "add_noise"):
            return self.base_scheduler.add_noise(
                original_samples,
                noise,
                torch.tensor([adjusted_timesteps], device=timesteps.device),
            )
        else:
            # Fallback: simple linear noise addition
            alpha = 1 - noise_level
            return alpha * original_samples + (1 - alpha) * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Perform one denoising step.

        Delegates to base scheduler.
        """
        return self.base_scheduler.step(
            model_output, timestep, sample, **kwargs
        )
