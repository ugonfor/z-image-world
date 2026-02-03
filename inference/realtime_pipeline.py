"""
Real-time Inference Pipeline for Z-Image World Model

Orchestrates the complete inference pipeline:
1. Receive keyboard input
2. Encode action
3. Generate next frame conditioned on action
4. Decode and display

Optimized for low latency with:
- Rolling KV cache
- Reduced denoising steps
- Async VAE operations
- Frame pipelining
"""

import time
from dataclasses import dataclass
from typing import Optional, Callable
from collections import deque

import torch
import torch.nn as nn
from einops import rearrange

from models import CausalDiT, ActionEncoder, ActionSpace, StreamVAE
from streaming import RollingKVCache, CacheConfig, MotionAwareNoiseController


@dataclass
class PipelineConfig:
    """Configuration for real-time pipeline."""

    # Resolution
    height: int = 480
    width: int = 640

    # Inference settings
    num_inference_steps: int = 4
    guidance_scale: float = 2.0

    # KV cache
    use_kv_cache: bool = True
    max_cache_length: int = 4096
    num_sink_tokens: int = 4

    # Motion-aware noise
    use_motion_control: bool = True
    base_noise_level: float = 0.5

    # Performance
    compile_model: bool = True
    use_flash_attention: bool = True
    channels_last: bool = True

    # Frame buffer
    context_frames: int = 4  # Number of previous frames to condition on
    target_fps: float = 20.0

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"


class FrameBuffer:
    """Circular buffer for recent frames."""

    def __init__(self, max_size: int = 16):
        self.max_size = max_size
        self._frames: deque[torch.Tensor] = deque(maxlen=max_size)
        self._latents: deque[torch.Tensor] = deque(maxlen=max_size)

    def add_frame(self, frame: torch.Tensor, latent: torch.Tensor):
        """Add a frame and its latent to the buffer."""
        self._frames.append(frame)
        self._latents.append(latent)

    def get_recent_frames(self, n: int) -> Optional[torch.Tensor]:
        """Get the n most recent frames."""
        if len(self._frames) == 0:
            return None
        n = min(n, len(self._frames))
        frames = list(self._frames)[-n:]
        return torch.stack(frames, dim=1)  # (1, n, C, H, W)

    def get_recent_latents(self, n: int) -> Optional[torch.Tensor]:
        """Get the n most recent latents."""
        if len(self._latents) == 0:
            return None
        n = min(n, len(self._latents))
        latents = list(self._latents)[-n:]
        return torch.stack(latents, dim=1)  # (1, n, C, H, W)

    @property
    def last_frame(self) -> Optional[torch.Tensor]:
        """Get the most recent frame."""
        return self._frames[-1] if self._frames else None

    @property
    def last_latent(self) -> Optional[torch.Tensor]:
        """Get the most recent latent."""
        return self._latents[-1] if self._latents else None

    def clear(self):
        """Clear the buffer."""
        self._frames.clear()
        self._latents.clear()


class RealtimePipeline:
    """Real-time inference pipeline for interactive world model.

    Usage:
        pipeline = RealtimePipeline.from_pretrained("model_path")
        pipeline.set_initial_frame(image)

        while running:
            action = get_user_input()
            next_frame = pipeline.step(action)
            display(next_frame)
    """

    def __init__(
        self,
        dit: CausalDiT,
        action_encoder: ActionEncoder,
        vae: StreamVAE,
        config: PipelineConfig,
    ):
        self.dit = dit
        self.action_encoder = action_encoder
        self.vae = vae
        self.config = config

        # Setup device and dtype
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        # Move models to device
        self.dit = self.dit.to(self.device, dtype=self.dtype)
        self.action_encoder = self.action_encoder.to(self.device, dtype=self.dtype)

        # Compile model if requested
        if config.compile_model and hasattr(torch, "compile"):
            self.dit = torch.compile(self.dit, mode="reduce-overhead")

        # Initialize KV cache
        if config.use_kv_cache:
            cache_config = CacheConfig(
                max_length=config.max_cache_length,
                num_sink_tokens=config.num_sink_tokens,
                num_layers=self.dit.num_layers,
            )
            self.kv_cache = RollingKVCache(cache_config)
        else:
            self.kv_cache = None

        # Motion-aware noise controller
        if config.use_motion_control:
            self.motion_controller = MotionAwareNoiseController(
                base_noise_level=config.base_noise_level,
                device=self.device,
            )
        else:
            self.motion_controller = None

        # Frame buffer
        self.frame_buffer = FrameBuffer(max_size=config.context_frames * 2)

        # Noise schedule (simple linear for inference)
        self._setup_scheduler()

        # Stats
        self._frame_count = 0
        self._total_time = 0.0

    def _setup_scheduler(self):
        """Setup noise scheduler for inference."""
        num_steps = self.config.num_inference_steps

        # Linear timestep schedule
        self.timesteps = torch.linspace(
            999, 0, num_steps, device=self.device
        ).long()

        # Precompute alpha values
        betas = torch.linspace(0.0001, 0.02, 1000, device=self.device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[PipelineConfig] = None,
    ) -> "RealtimePipeline":
        """Load pipeline from pretrained weights.

        Args:
            model_path: Path to model checkpoint
            config: Optional pipeline configuration

        Returns:
            Initialized pipeline
        """
        if config is None:
            config = PipelineConfig()

        device = torch.device(config.device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Initialize models
        dit = CausalDiT(
            in_channels=16,
            hidden_dim=4096,
            num_heads=32,
            num_layers=28,
            action_injection_layers=[7, 14, 21],
        )

        action_encoder = ActionEncoder(
            num_actions=17,
            embedding_dim=512,
            hidden_dim=4096,
        )

        # Load weights
        if "model_state_dict" in checkpoint:
            dit.load_state_dict(checkpoint["model_state_dict"])
        if "action_encoder_state_dict" in checkpoint:
            action_encoder.load_state_dict(checkpoint["action_encoder_state_dict"])

        # VAE (placeholder - would load from diffusers)
        vae = StreamVAE(tile_size=512)

        return cls(dit, action_encoder, vae, config)

    def set_initial_frame(self, image: torch.Tensor):
        """Set the initial frame to start generation from.

        Args:
            image: Initial image (1, 3, H, W) in [0, 1] or (H, W, 3) numpy
        """
        # Handle numpy input
        if not torch.is_tensor(image):
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        # Ensure correct shape
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Move to device and normalize
        image = image.to(self.device, dtype=self.dtype)
        if image.max() > 1.0:
            image = image / 255.0

        # Normalize to [-1, 1]
        image_normalized = 2 * image - 1

        # Encode to latent
        latent = self.vae.encode(image_normalized, use_cache=False)

        # Add to buffer
        self.frame_buffer.add_frame(image, latent)

        # Reset KV cache
        if self.kv_cache is not None:
            self.kv_cache.reset()

        # Reset stats
        self._frame_count = 0
        self._total_time = 0.0

    @torch.inference_mode()
    def step(
        self,
        action: int | ActionSpace,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Generate the next frame given an action.

        Args:
            action: Action index or ActionSpace enum
            return_latent: Whether to also return the latent

        Returns:
            Generated frame (1, 3, H, W) in [0, 1], optionally with latent
        """
        start_time = time.perf_counter()

        # Convert action to tensor
        if isinstance(action, ActionSpace):
            action = action.value
        action_tensor = torch.tensor([[action]], device=self.device)

        # Get context latent
        context_latent = self.frame_buffer.last_latent
        if context_latent is None:
            raise RuntimeError("No initial frame set. Call set_initial_frame first.")

        # Get motion-aware noise level
        if self.motion_controller is not None:
            prev_frame = self.frame_buffer.get_recent_frames(2)
            if prev_frame is not None and prev_frame.shape[1] >= 2:
                noise_level = self.motion_controller.compute_noise_level(
                    prev_frame[:, -2], prev_frame[:, -1], action
                )
            else:
                noise_level = self.config.base_noise_level
        else:
            noise_level = self.config.base_noise_level

        # Encode action
        action_conditioning = self.action_encoder(action_tensor)

        # Initialize from context latent with noise
        latent = self._add_initial_noise(context_latent, noise_level)

        # Denoising loop
        for t in self.timesteps:
            latent = self._denoise_step(
                latent,
                t,
                action_conditioning,
            )

        # Decode to image
        image = self.vae.decode(latent)
        image = (image + 1) / 2  # [-1, 1] -> [0, 1]
        image = image.clamp(0, 1)

        # Update buffer
        self.frame_buffer.add_frame(image, latent)

        # Update stats
        self._frame_count += 1
        self._total_time += time.perf_counter() - start_time

        if return_latent:
            return image, latent
        return image

    def _add_initial_noise(
        self,
        latent: torch.Tensor,
        noise_level: float,
    ) -> torch.Tensor:
        """Add initial noise based on motion level."""
        # Map noise level to timestep
        timestep = int(noise_level * (len(self.timesteps) - 1))
        timestep = min(max(timestep, 0), len(self.timesteps) - 1)
        t = self.timesteps[timestep]

        # Get alpha values
        sqrt_alpha = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alphas_cumprod[t])

        # Add noise
        noise = torch.randn_like(latent)
        noisy_latent = sqrt_alpha * latent + sqrt_one_minus_alpha * noise

        return noisy_latent

    def _denoise_step(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        action_conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one denoising step."""
        # Prepare timestep
        t = timestep.unsqueeze(0) if timestep.dim() == 0 else timestep

        # Model prediction
        noise_pred, _ = self.dit(
            latent.unsqueeze(1),  # Add frame dimension
            t,
            action_conditioning=action_conditioning,
            kv_cache=self.kv_cache.get_all() if self.kv_cache else None,
            use_cache=self.kv_cache is not None,
        )
        noise_pred = noise_pred.squeeze(1)  # Remove frame dimension

        # Simple DDPM step
        alpha = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[max(t - 1000 // self.config.num_inference_steps, 0)]

        # Predict x0
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha)
        x0_pred = (latent - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

        # Compute x_{t-1}
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)

        # Direction pointing to x_t
        direction = sqrt_one_minus_alpha_prev * noise_pred

        # No noise for final step
        if t > 0:
            noise = torch.randn_like(latent) * 0.0  # Deterministic for consistency
        else:
            noise = 0

        latent = sqrt_alpha_prev * x0_pred + direction + noise

        return latent

    @property
    def fps(self) -> float:
        """Get average frames per second."""
        if self._total_time == 0:
            return 0.0
        return self._frame_count / self._total_time

    @property
    def latency_ms(self) -> float:
        """Get average latency in milliseconds."""
        if self._frame_count == 0:
            return 0.0
        return (self._total_time / self._frame_count) * 1000

    def reset(self):
        """Reset the pipeline state."""
        self.frame_buffer.clear()
        if self.kv_cache:
            self.kv_cache.reset()
        if self.motion_controller:
            self.motion_controller.reset()
        self._frame_count = 0
        self._total_time = 0.0

    def warmup(self, num_iterations: int = 3):
        """Warmup the pipeline for stable performance.

        Args:
            num_iterations: Number of warmup iterations
        """
        # Create dummy input
        dummy_frame = torch.randn(
            1, 3, self.config.height, self.config.width,
            device=self.device, dtype=self.dtype
        )
        self.set_initial_frame(dummy_frame)

        # Run warmup iterations
        for _ in range(num_iterations):
            _ = self.step(ActionSpace.IDLE)

        # Reset after warmup
        self.reset()

    def benchmark(
        self,
        num_frames: int = 100,
        action_sequence: Optional[list[int]] = None,
    ) -> dict[str, float]:
        """Benchmark the pipeline performance.

        Args:
            num_frames: Number of frames to generate
            action_sequence: Optional sequence of actions (loops if shorter)

        Returns:
            Dictionary with performance metrics
        """
        # Create initial frame
        dummy_frame = torch.randn(
            1, 3, self.config.height, self.config.width,
            device=self.device, dtype=self.dtype
        )
        self.set_initial_frame(dummy_frame)

        # Default action sequence
        if action_sequence is None:
            action_sequence = [ActionSpace.MOVE_FORWARD.value] * num_frames

        # Warmup
        for _ in range(5):
            _ = self.step(ActionSpace.IDLE)

        # Reset stats
        self._frame_count = 0
        self._total_time = 0.0

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        for i in range(num_frames):
            action = action_sequence[i % len(action_sequence)]
            _ = self.step(action)

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start

        return {
            "total_frames": num_frames,
            "total_time_s": total_time,
            "fps": num_frames / total_time,
            "latency_ms": (total_time / num_frames) * 1000,
        }
