"""
Diffusion Forcing Training for Causal Video Generation

Implements the Diffusion Forcing training objective where each frame
in a sequence can have an independent noise level. This enables:
- Flexible autoregressive generation
- Robustness to compounding errors
- Better temporal consistency

Reference: Diffusion Forcing (Chen et al., 2024)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange


@dataclass
class DiffusionForcingConfig:
    """Configuration for Diffusion Forcing training."""

    # Noise schedule
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "scaled_linear"

    # Forcing schedule
    independent_noise: bool = True  # Independent noise per frame
    noise_level_sampling: str = "uniform"  # uniform, pyramid, causal

    # Training
    prediction_type: str = "v_prediction"  # epsilon, v_prediction, sample
    snr_gamma: Optional[float] = 5.0  # SNR weighting (min-SNR-gamma)

    # Frame sampling
    num_frames: int = 8
    frame_dropout_prob: float = 0.1  # Randomly drop frames during training


class DiffusionForcingLoss(nn.Module):
    """Compute Diffusion Forcing loss with per-frame noise levels.

    The key insight is that each frame can have a different noise level,
    allowing the model to learn robust generation across noise conditions.
    """

    def __init__(self, config: DiffusionForcingConfig):
        super().__init__()
        self.config = config

        # Create noise schedule
        self.register_buffer(
            "betas",
            self._make_beta_schedule(
                config.beta_schedule,
                config.num_train_timesteps,
                config.beta_start,
                config.beta_end,
            ),
        )

        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(self.alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - self.alphas_cumprod),
        )

        # SNR for loss weighting
        snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)
        self.register_buffer("snr", snr)

    def _make_beta_schedule(
        self,
        schedule: str,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
    ) -> torch.Tensor:
        """Create beta schedule for noise."""
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "scaled_linear":
            # Scaled linear for better training stability
            return torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps
            ) ** 2
        elif schedule == "cosine":
            # Cosine schedule (better for high resolution)
            steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
            alpha_bar = torch.cos((steps / num_timesteps + 0.008) / 1.008 * torch.pi / 2) ** 2
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def sample_timesteps(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample timesteps for each frame in the batch.

        Args:
            batch_size: Batch size
            num_frames: Number of frames
            device: Target device

        Returns:
            Timesteps of shape (batch, num_frames)
        """
        if self.config.independent_noise:
            if self.config.noise_level_sampling == "uniform":
                # Uniform random timesteps per frame
                timesteps = torch.randint(
                    0, self.config.num_train_timesteps,
                    (batch_size, num_frames),
                    device=device,
                )
            elif self.config.noise_level_sampling == "pyramid":
                # Earlier frames get lower noise (more clean)
                base = torch.randint(
                    0, self.config.num_train_timesteps,
                    (batch_size, 1),
                    device=device,
                )
                # Linearly increase noise for later frames
                offsets = torch.linspace(0, 0.3, num_frames, device=device)
                offsets = (offsets * self.config.num_train_timesteps).long()
                timesteps = (base + offsets.unsqueeze(0)) % self.config.num_train_timesteps
            elif self.config.noise_level_sampling == "causal":
                # First frame clean, later frames noisier
                first_t = torch.randint(
                    0, self.config.num_train_timesteps // 2,
                    (batch_size, 1),
                    device=device,
                )
                rest_t = torch.randint(
                    self.config.num_train_timesteps // 2,
                    self.config.num_train_timesteps,
                    (batch_size, num_frames - 1),
                    device=device,
                )
                timesteps = torch.cat([first_t, rest_t], dim=1)
            else:
                raise ValueError(f"Unknown sampling: {self.config.noise_level_sampling}")
        else:
            # Same timestep for all frames
            t = torch.randint(
                0, self.config.num_train_timesteps,
                (batch_size,),
                device=device,
            )
            timesteps = t.unsqueeze(1).expand(-1, num_frames)

        return timesteps

    def add_noise(
        self,
        clean_frames: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to frames with per-frame timesteps.

        Args:
            clean_frames: Clean frames (batch, num_frames, channels, h, w)
            noise: Noise tensor (same shape)
            timesteps: Per-frame timesteps (batch, num_frames)

        Returns:
            Noisy frames
        """
        batch, num_frames = timesteps.shape

        # Get alpha values for each frame
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]  # (B, F)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting
        sqrt_alpha = sqrt_alpha[:, :, None, None, None]  # (B, F, 1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha[:, :, None, None, None]

        noisy_frames = sqrt_alpha * clean_frames + sqrt_one_minus_alpha * noise
        return noisy_frames

    def get_velocity(
        self,
        clean_frames: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity target (v-prediction).

        v = sqrt(alpha) * noise - sqrt(1-alpha) * sample
        """
        batch, num_frames = timesteps.shape

        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]

        sqrt_alpha = sqrt_alpha[:, :, None, None, None]
        sqrt_one_minus_alpha = sqrt_one_minus_alpha[:, :, None, None, None]

        velocity = sqrt_alpha * noise - sqrt_one_minus_alpha * clean_frames
        return velocity

    def forward(
        self,
        model_output: torch.Tensor,
        clean_frames: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute Diffusion Forcing loss.

        Args:
            model_output: Model prediction (batch, num_frames, channels, h, w)
            clean_frames: Clean target frames
            noise: Added noise
            timesteps: Per-frame timesteps (batch, num_frames)

        Returns:
            Dictionary with loss and metrics
        """
        # Get target based on prediction type
        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "v_prediction":
            target = self.get_velocity(clean_frames, noise, timesteps)
        elif self.config.prediction_type == "sample":
            target = clean_frames
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")

        # Compute per-frame MSE loss
        loss_per_frame = F.mse_loss(model_output, target, reduction="none")
        loss_per_frame = loss_per_frame.mean(dim=(-3, -2, -1))  # (B, F)

        # SNR weighting (min-SNR-gamma)
        if self.config.snr_gamma is not None:
            snr = self.snr[timesteps]  # (B, F)
            weight = torch.clamp(snr, max=self.config.snr_gamma) / snr
            loss_per_frame = loss_per_frame * weight

        # Average over frames and batch
        loss = loss_per_frame.mean()

        return {
            "loss": loss,
            "loss_per_frame": loss_per_frame.mean(dim=0),  # (F,)
            "mean_timestep": timesteps.float().mean(),
        }


class DiffusionForcingTrainer:
    """Trainer for Diffusion Forcing with video sequences.

    Handles:
    - Data loading and preprocessing
    - Training loop with gradient accumulation
    - Checkpointing and logging
    """

    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        config: DiffusionForcingConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cuda"),
        gradient_accumulation_steps: int = 4,
        mixed_precision: bool = True,
    ):
        self.model = model
        self.vae = vae
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision

        self.loss_fn = DiffusionForcingLoss(config).to(device)

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler("cuda") if mixed_precision else None

        # Training state
        self.global_step = 0

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to latent space.

        Args:
            frames: Video frames (batch, num_frames, 3, h, w) in [0, 1]

        Returns:
            Latents (batch, num_frames, latent_channels, h//8, w//8)
        """
        batch, num_frames, c, h, w = frames.shape

        # Normalize to [-1, 1]
        frames = 2 * frames - 1

        # Flatten for VAE
        frames_flat = rearrange(frames, "b f c h w -> (b f) c h w")

        # Encode
        with torch.no_grad():
            posterior = self.vae.encode(frames_flat)
            if hasattr(posterior, "latent_dist"):
                latents = posterior.latent_dist.sample()
            else:
                latents = posterior

        # Reshape back
        latents = rearrange(latents, "(b f) c h w -> b f c h w", b=batch, f=num_frames)

        # Scale latents
        latents = latents * 0.13025  # Z-Image scaling factor

        return latents

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Execute one training step.

        Args:
            batch: Dictionary with "frames" tensor

        Returns:
            Dictionary with loss and metrics
        """
        frames = batch["frames"].to(self.device)  # (B, F, 3, H, W)
        batch_size, num_frames = frames.shape[:2]

        # Encode to latents
        latents = self.encode_frames(frames)

        # Sample timesteps
        timesteps = self.loss_fn.sample_timesteps(batch_size, num_frames, self.device)

        # Sample noise
        noise = torch.randn_like(latents)

        # Add noise
        noisy_latents = self.loss_fn.add_noise(latents, noise, timesteps)

        # Forward pass
        with torch.amp.autocast("cuda", enabled=self.mixed_precision):
            model_output, _ = self.model(
                noisy_latents,
                timesteps,
            )
            loss_dict = self.loss_fn(model_output, latents, noise, timesteps)

        loss = loss_dict["loss"] / self.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def train_epoch(
        self,
        dataloader: DataLoader,
        log_fn=None,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            log_fn: Optional logging function

        Returns:
            Average metrics for the epoch
        """
        self.model.train()
        total_metrics = {}

        for i, batch in enumerate(dataloader):
            metrics = self.train_step(batch)

            # Accumulate metrics
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v

            # Gradient step
            if (i + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

                if log_fn is not None:
                    log_fn(metrics, self.global_step)

        # Average metrics
        num_batches = len(dataloader)
        return {k: v / num_batches for k, v in total_metrics.items()}

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint["global_step"]
