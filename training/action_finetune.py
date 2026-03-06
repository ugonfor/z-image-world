"""
Action-Conditioned Fine-tuning with LoRA

Stage 2 training: Fine-tune the causal DiT model to respond to actions.
Uses LoRA (Low-Rank Adaptation) for parameter-efficient training.

Features:
- LoRA injection into attention layers
- Action embedding training
- Curriculum learning for action responsiveness
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


@dataclass
class ActionFinetuneConfig:
    """Configuration for action fine-tuning."""

    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"]
    )

    # Training
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 50000

    # Diffusion
    num_train_timesteps: int = 1000
    prediction_type: str = "v_prediction"

    # Curriculum
    use_curriculum: bool = True
    curriculum_stages: list[dict] = field(
        default_factory=lambda: [
            {"steps": 10000, "action_weight": 0.5, "motion_actions_only": True},
            {"steps": 20000, "action_weight": 0.75, "motion_actions_only": False},
            {"steps": 50000, "action_weight": 1.0, "motion_actions_only": False},
        ]
    )


class ActionConditioningLoss(nn.Module):
    """Loss function for action-conditioned generation.

    Combines:
    - Standard diffusion loss
    - Action consistency loss (frames should reflect actions)
    - Temporal smoothness loss
    """

    def __init__(
        self,
        prediction_type: str = "v_prediction",
        action_consistency_weight: float = 0.1,
        temporal_weight: float = 0.05,
    ):
        super().__init__()
        self.prediction_type = prediction_type
        self.action_consistency_weight = action_consistency_weight
        self.temporal_weight = temporal_weight

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        actions: torch.Tensor,
        action_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            model_output: Model prediction (B, F, C, H, W)
            target: Target (noise, v, or sample) (B, F, C, H, W)
            actions: Action indices (B, F)
            action_embeddings: Action embedding tensor (for consistency)

        Returns:
            Dictionary with losses
        """
        # Main diffusion loss
        diffusion_loss = F.mse_loss(model_output, target)

        # Temporal smoothness: consecutive frames should be similar
        if model_output.shape[1] > 1:
            frame_diff = model_output[:, 1:] - model_output[:, :-1]
            temporal_loss = (frame_diff ** 2).mean()
        else:
            temporal_loss = torch.tensor(0.0, device=model_output.device)

        # Action consistency: different actions should produce different outputs
        # This is a contrastive-style loss
        action_loss = self._action_consistency_loss(
            model_output, actions, action_embeddings
        )

        # Combine losses
        total_loss = (
            diffusion_loss
            + self.action_consistency_weight * action_loss
            + self.temporal_weight * temporal_loss
        )

        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "action_loss": action_loss,
            "temporal_loss": temporal_loss,
        }

    def _action_consistency_loss(
        self,
        model_output: torch.Tensor,
        actions: torch.Tensor,
        action_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute action embedding distinctiveness loss.

        Encourages action embeddings to be more distinct for different actions.
        This is a proxy for action responsiveness that operates in embedding
        space rather than output space (which would require full inference).

        Note: A proper action consistency loss would require running inference
        twice with different actions and comparing decoded outputs. That is
        computed offline in scripts/evaluate.py instead.
        """
        batch_size = action_embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=action_embeddings.device)

        # action_embeddings: (B, F, D) - use first frame embedding
        emb = action_embeddings[:, 0]  # (B, D)
        emb_norm = F.normalize(emb, dim=-1)

        # Pairwise cosine similarity
        sim = emb_norm @ emb_norm.T  # (B, B)

        # Action match matrix: same action → target 1.0, different → target -1.0
        # Using {-1, 1} instead of {0, 1} ensures different-action pairs have
        # active gradients from initialization (sim≈0, target=-1 → push apart),
        # not just same-action pairs (which are rare with small batches).
        actions_first = actions[:, 0]
        match = 2.0 * (actions_first.unsqueeze(1) == actions_first.unsqueeze(0)).float() - 1.0

        # Exclude diagonal (self-similarity is trivially 1)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim.device)
        loss = F.mse_loss(sim[mask], match[mask])
        return loss


class ActionFinetuner:
    """Fine-tuner for action-conditioned generation.

    Adds LoRA adapters to the base model and trains them
    along with the action encoder.
    """

    def __init__(
        self,
        model: nn.Module,
        action_encoder: nn.Module,
        vae: nn.Module,
        config: ActionFinetuneConfig,
        device: torch.device = torch.device("cuda"),
    ):
        self.base_model = model
        self.action_encoder = action_encoder
        self.vae = vae
        self.config = config
        self.device = device

        # Apply LoRA
        if PEFT_AVAILABLE:
            self.model = self._apply_lora(model)
        else:
            print("Warning: PEFT not available, training full model")
            self.model = model

        # Move to device
        self.model = self.model.to(device)
        self.action_encoder = self.action_encoder.to(device)

        # Loss function
        self.loss_fn = ActionConditioningLoss(
            prediction_type=config.prediction_type
        )

        # Precompute noise schedule (fixed, no need to recompute each step)
        betas = torch.linspace(0.0001, 0.02, config.num_train_timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer = lambda name, tensor: setattr(self, name, tensor)
        self.alphas_cumprod = alphas_cumprod

        # Optimizer (only LoRA params + action encoder)
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1,
        )

        # Training state
        self.global_step = 0
        self.current_curriculum_stage = 0

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA adapters to the model."""
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )

        return get_peft_model(model, lora_config)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for trainable parameters."""
        # Collect trainable parameters
        trainable_params = []

        # LoRA parameters (if using PEFT)
        if PEFT_AVAILABLE:
            for name, param in self.model.named_parameters():
                if "lora" in name.lower():
                    trainable_params.append(param)
        else:
            trainable_params.extend(self.model.parameters())

        # Action encoder parameters
        trainable_params.extend(self.action_encoder.parameters())

        return torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def get_curriculum_settings(self) -> dict:
        """Get current curriculum stage settings."""
        if not self.config.use_curriculum:
            return {"action_weight": 1.0, "motion_actions_only": False}

        for stage in self.config.curriculum_stages:
            if self.global_step < stage["steps"]:
                return stage

        return self.config.curriculum_stages[-1]

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Execute one training step.

        Args:
            batch: Dictionary with "frames" and "actions" tensors

        Returns:
            Dictionary with loss and metrics
        """
        frames = batch["frames"].to(self.device)  # (B, F, 3, H, W)
        actions = batch["actions"].to(self.device)  # (B, F)

        batch_size, num_frames = frames.shape[:2]

        # Get curriculum settings
        curriculum = self.get_curriculum_settings()

        # Filter to motion actions only if in early curriculum
        if curriculum.get("motion_actions_only", False):
            # Motion actions are 0-8 (movement + idle)
            motion_mask = actions[:, 0] <= 8
            if motion_mask.sum() > 0:
                frames = frames[motion_mask]
                actions = actions[motion_mask]
                batch_size = frames.shape[0]

        if batch_size == 0:
            return {"loss": 0.0}

        # Encode frames to latents
        latents = self._encode_frames(frames)

        # Sample timesteps (same for all frames in this stage)
        timesteps = torch.randint(
            0, self.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        )

        # Sample noise
        noise = torch.randn_like(latents)

        # Add noise to latents
        noisy_latents = self._add_noise(latents, noise, timesteps)

        # Get action conditioning
        action_conditioning = self.action_encoder(actions)

        # Forward pass
        with torch.amp.autocast("cuda"):
            model_output, _ = self.model(
                noisy_latents,
                timesteps,
                action_conditioning=action_conditioning,
            )

            # Get target
            if self.config.prediction_type == "epsilon":
                target = noise
            elif self.config.prediction_type == "v_prediction":
                target = self._get_velocity(latents, noise, timesteps)
            else:
                target = latents

            # Compute loss with curriculum weighting
            loss_dict = self.loss_fn(
                model_output, target, actions, action_conditioning
            )
            loss = loss_dict["loss"]

            # Apply curriculum action weight
            action_weight = curriculum.get("action_weight", 1.0)
            loss = loss * action_weight

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )
        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        return {
            k: v.item() if torch.is_tensor(v) else v
            for k, v in loss_dict.items()
        }

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to latent space."""
        batch, num_frames, c, h, w = frames.shape
        frames = 2 * frames - 1  # [0,1] -> [-1,1]

        frames_flat = rearrange(frames, "b f c h w -> (b f) c h w")

        with torch.no_grad():
            posterior = self.vae.encode(frames_flat)
            if hasattr(posterior, "latent_dist"):
                latents = posterior.latent_dist.sample()
            else:
                latents = posterior

        latents = rearrange(latents, "(b f) c h w -> b f c h w", b=batch, f=num_frames)
        return latents * 0.13025

    def _add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to latents using precomputed schedule."""
        sqrt_alpha = torch.sqrt(self.alphas_cumprod[timesteps])[:, None, None, None, None]
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alphas_cumprod[timesteps])[:, None, None, None, None]
        return sqrt_alpha * latents + sqrt_one_minus_alpha * noise

    def _get_velocity(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DDPM v-prediction target using precomputed schedule."""
        sqrt_alpha = torch.sqrt(self.alphas_cumprod[timesteps])[:, None, None, None, None]
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alphas_cumprod[timesteps])[:, None, None, None, None]
        return sqrt_alpha * noise - sqrt_one_minus_alpha * latents

    def train_epoch(
        self,
        dataloader: DataLoader,
        log_fn=None,
    ) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.action_encoder.train()

        total_metrics = {}

        for batch in dataloader:
            metrics = self.train_step(batch)

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v

            if log_fn is not None and self.global_step % 100 == 0:
                log_fn(metrics, self.global_step)

        num_batches = len(dataloader)
        return {k: v / num_batches for k, v in total_metrics.items()}

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "global_step": self.global_step,
            "action_encoder_state_dict": self.action_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }

        if PEFT_AVAILABLE:
            # Save LoRA weights separately
            checkpoint["lora_state_dict"] = {
                k: v for k, v in self.model.state_dict().items()
                if "lora" in k.lower()
            }
        else:
            checkpoint["model_state_dict"] = self.model.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.global_step = checkpoint["global_step"]
        self.action_encoder.load_state_dict(checkpoint["action_encoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "lora_state_dict" in checkpoint:
            # Load LoRA weights
            model_dict = self.model.state_dict()
            model_dict.update(checkpoint["lora_state_dict"])
            self.model.load_state_dict(model_dict)
        elif "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])

    def export_lora(self, path: str):
        """Export trained LoRA weights."""
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available, saving full model")
            torch.save(self.model.state_dict(), path)
            return

        # Save only LoRA weights
        lora_state = {
            k: v for k, v in self.model.state_dict().items()
            if "lora" in k.lower()
        }
        torch.save(lora_state, path)
