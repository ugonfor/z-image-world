"""
Z-Image World Pipeline for Interactive Frame Generation

Wraps ZImageWorldModel with a step(action) -> frame interface
for real-time interactive use. Uses DDIM-style denoising with
configurable quality/speed tradeoffs.
"""

import time
from dataclasses import dataclass
from typing import Optional
from collections import deque

import torch
import torch.nn as nn
from einops import rearrange


@dataclass
class ZImageWorldConfig:
    """Configuration for interactive world model pipeline."""

    # Resolution (generation, display may be different)
    height: int = 256
    width: int = 256

    # Denoising
    num_inference_steps: int = 2
    noise_start: float = 0.8  # Starting noise level (0-1, lower = closer to context)

    # Context
    context_frames: int = 2  # Previous frames to condition on

    # Performance
    compile_model: bool = False
    device: str = "cuda"

    # Model
    temporal_every_n: int = 1
    checkpoint_path: Optional[str] = None


# Quality presets
QUALITY_PRESETS = {
    "fast": ZImageWorldConfig(height=128, width=128, num_inference_steps=1, context_frames=1),
    "balanced": ZImageWorldConfig(height=256, width=256, num_inference_steps=2, context_frames=2),
    "quality": ZImageWorldConfig(height=384, width=384, num_inference_steps=4, context_frames=3),
}


class ZImageWorldPipeline:
    """Interactive pipeline wrapping ZImageWorldModel.

    Provides the same step(action) -> frame interface as RealtimePipeline,
    but uses the full 8.2B Z-Image-based world model.

    Usage:
        pipeline = ZImageWorldPipeline.from_pretrained()
        pipeline.set_initial_frame(image_tensor)

        while running:
            action = get_keyboard_action()
            frame = pipeline.step(action)
            display(frame)
    """

    def __init__(self, model, config: Optional[ZImageWorldConfig] = None):
        self.model = model
        self.config = config or ZImageWorldConfig()

        self.device = torch.device(self.config.device)

        # Frame buffer
        self._latents: deque[torch.Tensor] = deque(maxlen=16)
        self._frames: deque[torch.Tensor] = deque(maxlen=16)
        self._actions: deque[int] = deque(maxlen=16)

        # Noise schedule
        self._setup_scheduler()

        # Stats
        self._frame_count = 0
        self._total_time = 0.0
        self._last_step_time = 0.0

    def _setup_scheduler(self):
        """Setup DDIM noise schedule for inference."""
        num_steps = self.config.num_inference_steps
        betas = torch.linspace(0.0001, 0.02, 1000, device=self.device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Timestep schedule (evenly spaced)
        step_ratio = 1000 // max(num_steps, 1)
        self.timesteps = (torch.arange(num_steps, device=self.device) * step_ratio).flip(0).long()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Tongyi-MAI/Z-Image-Turbo",
        config: Optional[ZImageWorldConfig] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ZImageWorldPipeline":
        """Load model and create pipeline."""
        if config is None:
            config = ZImageWorldConfig()

        from models.zimage_world_model import ZImageWorldModel

        model = ZImageWorldModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            temporal_every_n=config.temporal_every_n,
            freeze_spatial=True,
            device=config.device,
        )

        # Load trained weights if checkpoint provided
        if config.checkpoint_path:
            print(f"Loading checkpoint: {config.checkpoint_path}")
            ckpt = torch.load(config.checkpoint_path, map_location=config.device)
            model.temporal_layers.load_state_dict(ckpt["temporal_state_dict"])
            model.action_injections.load_state_dict(ckpt["action_injections_state_dict"])
            model.action_encoder.load_state_dict(ckpt["action_encoder_state_dict"])
            print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

        model.eval()

        # torch.compile for speed
        if config.compile_model and hasattr(torch, "compile"):
            print("Compiling model with torch.compile...")
            model.transformer = torch.compile(model.transformer, mode="reduce-overhead")

        pipeline = cls(model, config)
        return pipeline

    def set_initial_frame(self, image: torch.Tensor):
        """Set the starting frame for generation.

        Args:
            image: (1, 3, H, W) in [0, 1] or (H, W, 3) numpy array
        """
        if not torch.is_tensor(image):
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device, dtype=torch.bfloat16)
        if image.max() > 1.0:
            image = image / 255.0

        # Resize to generation resolution
        if image.shape[-2] != self.config.height or image.shape[-1] != self.config.width:
            image = torch.nn.functional.interpolate(
                image, size=(self.config.height, self.config.width),
                mode="bilinear", align_corners=False,
            )

        # Encode to latent
        with torch.no_grad():
            image_5d = image.unsqueeze(1)  # (1, 1, 3, H, W)
            latent = self.model.encode_frames(image_5d)[:, 0]  # (1, 16, H//8, W//8)

        self._latents.clear()
        self._frames.clear()
        self._actions.clear()

        self._latents.append(latent)
        self._frames.append(image)
        self._actions.append(8)  # IDLE

        self._frame_count = 0
        self._total_time = 0.0

    @torch.inference_mode()
    def step(self, action: int = 8) -> torch.Tensor:
        """Generate next frame given an action.

        Args:
            action: Action index (0-16), default 8 (IDLE)

        Returns:
            Generated frame (1, 3, H, W) in [0, 1]
        """
        t_start = time.time()

        if not self._latents:
            raise RuntimeError("Call set_initial_frame() first")

        self._actions.append(action)

        # Get context latents
        ctx_len = min(self.config.context_frames, len(self._latents))
        context = list(self._latents)[-ctx_len:]
        context_tensor = torch.stack(context, dim=1)  # (1, ctx_len, 16, H, W)

        latent_shape = context[0].shape  # (1, 16, H//8, W//8)

        # Start from noise
        noise = torch.randn(1, 1, *latent_shape[1:], device=self.device, dtype=torch.bfloat16)

        # Build action sequence for context + generation
        action_ids = list(self._actions)[-ctx_len:] + [action]
        action_tensor = torch.tensor([action_ids], device=self.device)

        # Denoising loop
        x_t = noise[:, 0]  # (1, 16, H//8, W//8)

        for step_idx, t in enumerate(self.timesteps):
            # Build full sequence: [context (clean) | current (noisy)]
            full_latents = torch.cat([context_tensor, x_t.unsqueeze(1)], dim=1)  # (1, ctx+1, 16, H, W)

            # Timesteps: context = 0 (clean), current = t
            t_context = torch.zeros(1, ctx_len, device=self.device)
            t_current = t.float().unsqueeze(0).unsqueeze(0)
            timesteps = torch.cat([t_context, t_current], dim=1)

            # Forward pass
            pred = self.model(full_latents, timesteps, actions=action_tensor)

            # Extract prediction for the new frame (last in sequence)
            if pred.dim() == 5:
                v_pred = pred[:, -1]  # (1, 16, H//8, W//8)
            else:
                v_pred = pred

            # DDIM step (v-prediction to sample)
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha = alpha_t.sqrt()
            sqrt_one_minus_alpha = (1 - alpha_t).sqrt()

            # v = sqrt(alpha) * noise - sqrt(1-alpha) * sample
            # => sample = (sqrt(alpha) * noise - v) / sqrt(1-alpha)  ... if v_pred
            # Simplified: predict x0 from v
            x0_pred = sqrt_alpha * x_t - sqrt_one_minus_alpha * v_pred

            if step_idx < len(self.timesteps) - 1:
                # Next timestep
                t_next = self.timesteps[step_idx + 1]
                alpha_next = self.alphas_cumprod[t_next]
                sqrt_alpha_next = alpha_next.sqrt()
                sqrt_one_minus_alpha_next = (1 - alpha_next).sqrt()

                # DDIM: x_{t-1} = sqrt(alpha_{t-1}) * x0_pred + sqrt(1-alpha_{t-1}) * noise_direction
                noise_direction = (x_t - sqrt_alpha * x0_pred) / sqrt_one_minus_alpha.clamp(min=1e-8)
                x_t = sqrt_alpha_next * x0_pred + sqrt_one_minus_alpha_next * noise_direction
            else:
                x_t = x0_pred

        # Decode to image
        decoded = self.model.decode_latents(x_t.unsqueeze(1))  # (1, 1, 3, H, W)
        if decoded.dim() == 5:
            frame = decoded[:, 0]  # (1, 3, H, W)
        else:
            frame = decoded

        # Update buffers
        self._latents.append(x_t)
        self._frames.append(frame)

        # Stats
        self._last_step_time = time.time() - t_start
        self._total_time += self._last_step_time
        self._frame_count += 1

        return frame

    @property
    def fps(self) -> float:
        if self._total_time > 0:
            return self._frame_count / self._total_time
        return 0.0

    @property
    def last_step_time(self) -> float:
        return self._last_step_time

    def warmup(self, num_iterations: int = 2):
        """Run warmup iterations for torch.compile / CUDA initialization."""
        if not self._latents:
            # Create a dummy initial frame
            dummy = torch.rand(1, 3, self.config.height, self.config.width,
                             device=self.device, dtype=torch.bfloat16)
            self.set_initial_frame(dummy)

        print(f"Warming up ({num_iterations} iterations)...")
        for i in range(num_iterations):
            _ = self.step(8)  # IDLE action
        print(f"  Warmup done. FPS: {self.fps:.1f}")

    def set_quality(self, preset: str):
        """Switch quality preset at runtime.

        Args:
            preset: "fast", "balanced", or "quality"
        """
        if preset in QUALITY_PRESETS:
            new_config = QUALITY_PRESETS[preset]
            self.config.height = new_config.height
            self.config.width = new_config.width
            self.config.num_inference_steps = new_config.num_inference_steps
            self.config.context_frames = new_config.context_frames
            self._setup_scheduler()
            print(f"Quality preset: {preset} ({self.config.height}x{self.config.width}, {self.config.num_inference_steps} steps)")
