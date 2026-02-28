"""
Flow Matching Trainer for Z-Image World Model

Implements rectified flow (linear flow matching) as an alternative to DDPM.
Key advantages:
- Straight-line trajectories → fewer inference steps needed (1-4 vs 20+)
- Simpler training objective: predict (x_1 - x_0) velocity
- Compatible with ZImageWorldModel and CausalDiT unchanged

Based on: SD3, Flux, and Wan 2.1 (all use rectified flow)

Key difference from DDPM v-prediction:
  DDPM v: sqrt(alpha_t)*noise - sqrt(1-alpha_t)*clean
  Flow v: x_1 - x_0 = clean - noise   (constant along trajectory)

The forward process is linear interpolation:
  x_t = (1 - t) * x_0 + t * x_1
  where x_0 = noise (Gaussian), x_1 = clean data, t ∈ [0, 1]
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching training."""

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 100000

    # Flow matching
    # Timestep sampling strategy:
    #   "uniform": t ~ Uniform[0, 1]
    #   "logit_normal": t = sigmoid(N(0, 1)) - better coverage of high-SNR regime
    #   "cosmap": cosine map sampling (from CosMap paper)
    timestep_sampling: str = "logit_normal"

    # Training objective
    # "velocity": predict (x_1 - x_0) directly (standard rectified flow)
    # "x1": predict the clean data x_1 directly
    prediction_type: str = "velocity"

    # Mini-batch size for temporal sequences
    num_frames: int = 8

    # Mixed precision
    use_amp: bool = True


class FlowMatchingLoss(nn.Module):
    """Loss function for rectified flow matching.

    The model learns to predict the velocity field v = x_1 - x_0
    where x_0 ~ N(0,I) and x_1 is the clean data.
    """

    def __init__(
        self,
        prediction_type: str = "velocity",
        temporal_weight: float = 0.05,
    ):
        super().__init__()
        self.prediction_type = prediction_type
        self.temporal_weight = temporal_weight

    def forward(
        self,
        model_output: torch.Tensor,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        timesteps_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute flow matching loss.

        Args:
            model_output: Model prediction (B, F, C, H, W)
            x_0: Noise samples (B, F, C, H, W)
            x_1: Clean data samples (B, F, C, H, W)
            timesteps_t: Flow time t ∈ [0, 1] (B,) or (B, F)

        Returns:
            Dictionary with 'loss' and component losses
        """
        # Target velocity: constant along straight-line trajectory
        target_velocity = x_1 - x_0  # (B, F, C, H, W)

        if self.prediction_type == "velocity":
            pred = model_output
            target = target_velocity
        elif self.prediction_type == "x1":
            # Model predicts clean data directly
            pred = model_output
            target = x_1
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Main MSE loss
        flow_loss = F.mse_loss(pred, target)

        # Temporal smoothness on velocity predictions
        if model_output.shape[1] > 1:
            frame_diff = pred[:, 1:] - pred[:, :-1]
            temporal_loss = (frame_diff ** 2).mean()
        else:
            temporal_loss = torch.tensor(0.0, device=model_output.device)

        total_loss = flow_loss + self.temporal_weight * temporal_loss

        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "temporal_loss": temporal_loss,
        }


def sample_flow_timesteps(
    batch_size: int,
    num_frames: int,
    sampling: str = "logit_normal",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Sample flow timesteps t ∈ [0, 1] for training.

    Args:
        batch_size: Number of sequences in batch
        num_frames: Number of frames per sequence
        sampling: Sampling strategy ('uniform', 'logit_normal', 'cosmap')
        device: Target device

    Returns:
        Timesteps of shape (batch_size,) or (batch_size, num_frames)
        depending on whether independent per-frame sampling is used
    """
    if sampling == "uniform":
        # Simple uniform sampling - each frame gets independent t
        t = torch.rand(batch_size, num_frames, device=device)
    elif sampling == "logit_normal":
        # Logit-normal: t = sigmoid(N(0,1))
        # Better coverage of high-SNR (t≈1) regime where most perceptual detail lives
        # Used by SD3, Flux
        u = torch.randn(batch_size, num_frames, device=device)
        t = torch.sigmoid(u)
    elif sampling == "cosmap":
        # Cosine map: t = 1 - 1/(tan(pi/2 * u) + 1)
        # More uniform in SNR space
        u = torch.rand(batch_size, num_frames, device=device)
        t = 1.0 - 1.0 / (torch.tan(torch.pi / 2 * u) + 1.0)
        t = t.clamp(0.001, 0.999)
    else:
        raise ValueError(f"Unknown timestep sampling: {sampling}")

    return t


def flow_forward_process(
    x_1: torch.Tensor,
    x_0: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Linear interpolation forward process for flow matching.

    x_t = (1 - t) * x_0 + t * x_1

    Args:
        x_1: Clean data (B, F, C, H, W)
        x_0: Noise (B, F, C, H, W)
        t: Flow time (B,) or (B, F) in [0, 1]

    Returns:
        Interpolated sample x_t of same shape as x_1
    """
    # Broadcast t to match spatial dimensions
    while t.dim() < x_1.dim():
        t = t.unsqueeze(-1)

    return (1.0 - t) * x_0 + t * x_1


class FlowMatchingTrainer:
    """Trainer for rectified flow matching.

    This is a SEPARATE trainer from DiffusionForcingTrainer.
    Do NOT mix flow matching and DDPM components — the forward processes,
    targets, and inference ODE integrators are fundamentally different.

    Training stages:
    1. Train with flow matching objective (predict velocity field)
    2. Optionally fine-tune with consistency distillation for 1-step inference

    Usage:
        trainer = FlowMatchingTrainer(model, vae, config)
        for batch in dataloader:
            metrics = trainer.train_step(batch)
    """

    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        config: FlowMatchingConfig,
        action_encoder: Optional[nn.Module] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model
        self.vae = vae
        self.config = config
        self.action_encoder = action_encoder
        self.device = device

        # Move to device
        self.model = self.model.to(device)
        if self.action_encoder is not None:
            self.action_encoder = self.action_encoder.to(device)

        # Loss function
        self.loss_fn = FlowMatchingLoss(
            prediction_type=config.prediction_type,
        )

        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Cosine schedule with warmup
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._warmup_cosine_lambda,
        )

        # AMP autocast (for bfloat16 models, no GradScaler needed)
        # GradScaler is only for float16; bfloat16 has same exponent range as float32
        self._use_amp = config.use_amp and device.type == "cuda"

        # Training state
        self.global_step = 0

    def _warmup_cosine_lambda(self, step: int) -> float:
        """Learning rate schedule with linear warmup then cosine decay."""
        if step < self.config.warmup_steps:
            return step / max(1, self.config.warmup_steps)
        progress = (step - self.config.warmup_steps) / max(
            1, self.config.max_steps - self.config.warmup_steps
        )
        return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item()))

    @torch.no_grad()
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames to latent space.

        Args:
            frames: (B, F, 3, H, W) in [0, 1]

        Returns:
            Latents (B, F, C, H//8, W//8)
        """
        batch, num_frames, c, h, w = frames.shape
        frames_flat = rearrange(frames, "b f c h w -> (b f) c h w")

        # Normalize to [-1, 1]
        frames_flat = frames_flat * 2.0 - 1.0

        posterior = self.vae.encode(frames_flat)
        if hasattr(posterior, "latent_dist"):
            latents = posterior.latent_dist.sample()
        elif hasattr(posterior, "sample"):
            latents = posterior.sample()
        else:
            latents = posterior

        # Apply VAE scaling factor if available
        scaling_factor = getattr(getattr(self.vae, "config", None), "scaling_factor", 0.13025)
        latents = latents * scaling_factor

        return rearrange(latents, "(b f) c h w -> b f c h w", b=batch, f=num_frames)

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Execute one flow matching training step.

        Args:
            batch: Dictionary with "frames" (B, F, 3, H, W) in [0, 1]
                   and optionally "actions" (B, F)

        Returns:
            Dictionary with loss metrics
        """
        frames = batch["frames"].to(self.device)
        actions = batch.get("actions", None)
        if actions is not None:
            actions = actions.to(self.device)

        batch_size, num_frames = frames.shape[:2]

        # Encode to latent space
        x_1 = self._encode_frames(frames)  # Clean data

        # Sample noise (Gaussian)
        x_0 = torch.randn_like(x_1)

        # Sample flow timesteps t ∈ [0, 1] for each frame independently
        # This is the "Diffusion Forcing" approach: per-frame independent t
        t = sample_flow_timesteps(
            batch_size,
            num_frames,
            sampling=self.config.timestep_sampling,
            device=self.device,
        )  # (B, F)

        # Forward process: linear interpolation
        x_t = flow_forward_process(x_1, x_0, t)

        # Convert t ∈ [0, 1] to integer timestep for model
        # (model expects integer timesteps 0-999 for embedding)
        # t=0 corresponds to pure noise, t=1 to clean data
        t_int = (t * 999).long().clamp(0, 999)  # (B, F)

        # Encode actions to conditioning embeddings if action encoder available
        action_conditioning = None
        if actions is not None and self.action_encoder is not None:
            with torch.no_grad():
                action_conditioning = self.action_encoder(actions.to(self.device))

        # Forward pass
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            if action_conditioning is not None:
                # CausalDiT API: action_conditioning (pre-encoded embeddings)
                model_output = self.model(x_t, t_int, action_conditioning=action_conditioning)
            elif actions is not None and not hasattr(self.model, "action_encoder"):
                # ZImageWorldModel API: actions (raw integer indices, model encodes internally)
                model_output = self.model(x_t, t_int, actions=actions)
            else:
                model_output = self.model(x_t, t_int)

            # Handle tuple output (model may return (output, cache))
            if isinstance(model_output, tuple):
                model_output = model_output[0]

            # Compute flow matching loss
            loss_dict = self.loss_fn(model_output, x_0, x_1, t)
            loss = loss_dict["loss"]

        # Backward (no GradScaler needed; bfloat16 doesn't underflow like float16)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def train_epoch(
        self,
        dataloader: DataLoader,
        log_fn=None,
    ) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_metrics: dict[str, float] = {}

        for batch in dataloader:
            metrics = self.train_step(batch)

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v

            if log_fn is not None and self.global_step % 100 == 0:
                log_fn(metrics, self.global_step)

        num_batches = len(dataloader)
        return {k: v / num_batches for k, v in total_metrics.items()}

    def save_checkpoint(self, path: str):
        """Save flow matching checkpoint."""
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": {
                k: v for k, v in self.model.state_dict().items()
                if any(p.requires_grad for p in [self.model.state_dict()[k]]
                       if isinstance(v, torch.Tensor))
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "training_type": "flow_matching",
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load flow matching checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        assert checkpoint.get("training_type") == "flow_matching", (
            "Checkpoint was not saved by FlowMatchingTrainer. "
            "Do not mix DDPM and flow matching checkpoints."
        )
        self.global_step = checkpoint["global_step"]
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


class FlowMatchingInference:
    """ODE integrator for flow matching inference.

    Uses Euler method to solve dx/dt = v_θ(x_t, t) from t=0 to t=1.

    For a trained flow matching model, 1-4 Euler steps give good quality.
    This is much faster than 20-100 DDPM steps.

    IMPORTANT: Use this inference class for flow-matching trained models.
    Do NOT use RealtimePipeline._denoise_step() (DDPM-based) with a
    flow-matching trained model — the update formulas are incompatible.
    """

    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        num_steps: int = 4,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.vae = vae
        self.num_steps = num_steps
        self.device = device
        self.dtype = dtype

        # Euler timesteps: t from 0 (noise) to 1 (clean)
        self.t_schedule = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    @torch.inference_mode()
    def denoise(
        self,
        x_0: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        context_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise from pure noise to clean latent using Euler ODE.

        Args:
            x_0: Initial noise (B, C, H, W) or (B, F, C, H, W)
            actions: Optional action conditioning (B, F)
            context_latent: Optional conditioning context

        Returns:
            Clean latent x_1 of same shape as x_0
        """
        x = x_0.to(self.device, dtype=self.dtype)

        for i in range(self.num_steps):
            t_cur = self.t_schedule[i]
            dt = self.t_schedule[i + 1] - t_cur

            # Integer timestep for model embedding (0-999)
            t_int = (t_cur * 999).long().clamp(0, 999)
            t_batch = t_int.expand(x.shape[0])

            # Predict velocity (autocast only on CUDA)
            use_amp = self.device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                if actions is not None:
                    # Try action_conditioning first (CausalDiT API), then actions (ZImageWorldModel API)
                    try:
                        velocity = self.model(x, t_batch, action_conditioning=actions)
                    except TypeError:
                        velocity = self.model(x, t_batch, actions=actions)
                else:
                    velocity = self.model(x, t_batch)

                if isinstance(velocity, tuple):
                    velocity = velocity[0]

            # Euler step: x_{t+dt} = x_t + dt * v_θ(x_t, t)
            x = x + dt * velocity.to(x.dtype)

        return x

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to pixel space.

        Args:
            latent: (B, C, H, W) or (B, F, C, H, W)

        Returns:
            Images in [0, 1]
        """
        scaling_factor = getattr(getattr(self.vae, "config", None), "scaling_factor", 0.13025)

        if latent.dim() == 5:
            batch, num_frames = latent.shape[:2]
            latent_flat = rearrange(latent, "b f c h w -> (b f) c h w")
        else:
            batch, num_frames = latent.shape[0], 1
            latent_flat = latent

        latent_flat = latent_flat / scaling_factor
        images = self.vae.decode(latent_flat)
        if hasattr(images, "sample"):
            images = images.sample

        images = (images + 1.0) / 2.0
        images = images.clamp(0.0, 1.0)

        if num_frames > 1:
            images = rearrange(images, "(b f) c h w -> b f c h w", b=batch, f=num_frames)

        return images
