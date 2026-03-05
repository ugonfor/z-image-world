"""
FIFO-Diffusion Inference for ZImageWorldModel

Converts Z-Image-Turbo into a video generator using the FIFO-Diffusion
technique (Kim et al., NeurIPS 2024, arxiv:2405.11473).

Core idea: maintain a queue of K frames at staggered noise levels
[sigma=0 (clean), ..., sigma=1 (pure noise)]. At each FIFO step:
  1. Process all K frames jointly through ZImageWorldModel (temporal attention)
  2. Apply one Euler step to advance each frame toward cleaner state
  3. Dequeue the head frame (now sigma≈0, fully denoised) → output
  4. Shift queue, append new pure-noise frame at tail

This leverages:
- Z-Image-Turbo's spatial quality (photorealistic generation per frame)
- Causal temporal attention (coherence from previous frames)
- Flow Matching Euler scheduler (Z-Image's native schedule)

Usage:
    pipeline = FIFOPipeline.from_pretrained(
        'weights/Z-Image-Turbo',
        checkpoint='checkpoints/zimage_stage1_v2/world_model_final.pt',
    )
    frames = pipeline.generate(
        prompt='a lush green forest path in afternoon sunlight',
        num_frames=24,
        queue_size=8,
        num_inference_steps=20,
    )
    pipeline.save_gif(frames, 'output.gif')
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline
from einops import rearrange
from PIL import Image


@dataclass
class FIFOConfig:
    queue_size: int = 8           # Frames in the FIFO queue
    num_inference_steps: int = 20  # Denoising steps per frame
    height: int = 256
    width: int = 256
    guidance_scale: float = 3.5
    seed_steps: int = 6           # Steps for seed image generation
    temporal_every_n: int = 3     # Temporal attention frequency (match training)
    # Anchor init: initialize new tail frames from the most recent clean (head) frame
    # plus noise, rather than pure random noise. Dramatically improves temporal
    # coherence after the initial queue fills. Recommended: True.
    anchor_init: bool = True
    # Fraction of σ_max to add as noise when anchor_init=True.
    # 1.0 = full noise (start from scratch each frame, high creativity, more drift)
    # 0.7 = 70% noise (biases new frame toward previous, less drift, less motion)
    # 0.5 = 50% noise (very close to previous frame, near-static video)
    # Recommended range: 0.6–0.85 for balanced coherence vs. motion.
    anchor_noise_frac: float = 0.75
    # Lookahead: freeze the front fraction as committed context (FIFO paper variant).
    # Requires front frames to already be clean from prior iterations.
    # Disabled by default; enable only after validating basic pipeline quality.
    use_lookahead: bool = False
    lookahead_fraction: float = 0.167  # fraction of queue frozen (1/K = just the head)
    # Classifier-Free Guidance: run two forward passes (null + text) and extrapolate.
    # Improves text adherence and reduces semantic drift at 2x compute cost per step.
    # use_cfg=False recommended for real-time use; True for highest quality.
    use_cfg: bool = False
    # Action conditioning: pass per-frame discrete action indices to the model.
    # Requires a Stage 2 checkpoint (action_encoder + action_injections trained).
    # Actions: 0=idle, 1=forward, 2=backward, 3=left, 4=right, 5=run, 6=jump, 7=interact
    use_actions: bool = False


class FIFOPipeline:
    """FIFO-Diffusion pipeline for temporally coherent video generation.

    Generates infinite video from a text prompt using ZImageWorldModel
    with FIFO-Diffusion scheduling.
    """

    def __init__(
        self,
        model,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae,
        config: Optional[FIFOConfig] = None,
    ):
        self.model = model
        self.scheduler = scheduler
        self.vae = vae
        self.config = config or FIFOConfig()
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "weights/Z-Image-Turbo",
        checkpoint: Optional[str] = None,
        config: Optional[FIFOConfig] = None,
        device: str = "cuda",
    ) -> "FIFOPipeline":
        """Load model + checkpoint and return pipeline.

        Loads ZImagePipeline ONCE to avoid duplicate weight loading, then
        extracts transformer/vae for ZImageWorldModel and keeps the text
        encoder alive for prompt conditioning during inference.
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.zimage_world_model import ZImageWorldModel
        import torch.nn as nn

        cfg = config or FIFOConfig()

        # Load the full Z-Image pipeline ONCE (avoids loading the 7B transformer twice).
        print(f"Loading Z-Image-Turbo from {model_path}...")
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        pipe = pipe.to(device)

        # Build ZImageWorldModel from the already-loaded transformer + vae.
        transformer = pipe.transformer
        vae = pipe.vae
        num_layers = len(transformer.layers)
        print(f"Transformer: {num_layers} layers, "
              f"{sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B params")

        model = ZImageWorldModel(
            transformer=transformer,
            vae=vae,
            num_layers=num_layers,
            temporal_every_n=cfg.temporal_every_n,
            freeze_spatial=True,
        )
        model = model.to(device=device, dtype=torch.bfloat16)
        print(f"World model: {model.num_total_params() / 1e9:.2f}B total, "
              f"{model.num_trainable_params() / 1e6:.1f}M trainable")

        if checkpoint and Path(checkpoint).exists():
            print(f"Loading checkpoint: {checkpoint}")
            ckpt = torch.load(checkpoint, map_location=device)
            model.temporal_layers.load_state_dict(ckpt["temporal_state_dict"])
            gamma0 = list(ckpt["temporal_state_dict"].values())[0].item()
            has_actions = (
                "action_injections_state_dict" in ckpt
                and "action_encoder_state_dict" in ckpt
            )
            if has_actions:
                model.action_injections.load_state_dict(
                    ckpt["action_injections_state_dict"]
                )
                model.action_encoder.load_state_dict(
                    ckpt["action_encoder_state_dict"]
                )
                print(f"  Stage 2 checkpoint (action layers loaded), "
                      f"epoch={ckpt.get('epoch','?')}, gamma[0]={gamma0:.4f}")
            else:
                print(f"  Stage 1 checkpoint, "
                      f"epoch={ckpt.get('epoch','?')}, gamma[0]={gamma0:.4f}")

        model.eval()

        # Load scheduler (Flow Matching, Z-Image native)
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
            use_dynamic_shifting=False,
        )

        instance = cls(model, scheduler, vae, config=cfg)
        # Keep the full pipeline alive so encode_prompt() can be called.
        # pipe.transformer and pipe.vae are the same objects already in model,
        # so no extra GPU memory is used for those.  Only the text encoder
        # (~few GB) is the incremental cost vs. not keeping pipe.
        instance._zimage_pipe = pipe
        instance._model_path = model_path
        return instance

    def _encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Encode PIL image to VAE latent. Returns (1, 16, H/8, W/8)."""
        arr = np.array(pil_image.convert("RGB").resize(
            (self.config.width, self.config.height), Image.BILINEAR
        ))
        tensor = torch.from_numpy(arr).float().div(255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
        # Normalize to [-1, 1] for VAE
        tensor = tensor * 2.0 - 1.0
        with torch.no_grad():
            latent = self.vae.encode(tensor).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent

    def _decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """Decode latent to PIL image."""
        with torch.no_grad():
            img = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        img = img.float().clamp(-1, 1)
        img = (img + 1.0) / 2.0  # [-1,1] → [0,1]
        img_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def generate(
        self,
        prompt: str,
        num_frames: int = 24,
        seed_image: Optional[Image.Image] = None,
        seed: int = 42,
        actions: Optional[list[int]] = None,
    ) -> list[Image.Image]:
        """Generate a video sequence using FIFO-Diffusion.

        Args:
            prompt: Text description of the scene
            num_frames: Total output frames to generate
            seed_image: Optional starting image (skips seed generation if provided)
            seed: Random seed for reproducibility
            actions: Optional per-frame action indices (length >= num_frames).
                     0=idle, 1=forward, 2=backward, 3=left, 4=right,
                     5=run, 6=jump, 7=interact. Requires Stage 2 checkpoint.
                     If shorter than num_frames, padded with ACTION_IDLE (0).

        Returns:
            List of PIL images (num_frames long)
        """
        cfg = self.config
        device = self.device
        dtype = self.dtype
        torch.manual_seed(seed)

        H = cfg.height // 8
        W = cfg.width // 8
        C = self.vae.config.latent_channels  # 16

        # ── 0. Encode text prompt for conditioning ───────────────────────────
        # Critical: pass real caption features to the Z-Image transformer so
        # frames are denoised with text guidance, not unconditioned (null caps).
        cap_feats = None
        null_cap_feats = None
        if hasattr(self, "_zimage_pipe"):
            print(f"Encoding prompt for text conditioning...")
            with torch.no_grad():
                cap_feats_list, _ = self._zimage_pipe.encode_prompt(
                    prompt,
                    device=device,
                    do_classifier_free_guidance=False,
                )
            cap_feats = cap_feats_list[0].to(dtype=dtype)  # (seq_len, cap_feat_dim)
            print(f"  Caption features: {cap_feats.shape}")
            if cfg.use_cfg:
                # Null caption: 32 zero-tokens (matches the model's internal null)
                _SEQ_MULTI_OF = 32
                _cap_feat_dim = getattr(
                    self.model.transformer.config, "cap_feat_dim", 2560
                )
                null_cap_feats = torch.zeros(
                    _SEQ_MULTI_OF, _cap_feat_dim, device=device, dtype=dtype
                )
                print(f"  CFG enabled (guidance_scale={cfg.guidance_scale})")

        # ── 1. Generate seed image with ZImagePipeline ──────────────────────
        if seed_image is None:
            print(f"Generating seed image: '{prompt}'")
            seed_image = self._generate_seed_image(prompt, seed)

        seed_latent = self._encode_image(seed_image)  # (1, 16, H, W)
        print(f"Seed image encoded. Shape: {seed_latent.shape}")

        # ── 2. Initialize FIFO queue ─────────────────────────────────────────
        # Set up scheduler sigmas for queue_size staggered levels
        self.scheduler.set_timesteps(cfg.num_inference_steps, device=device)
        sigmas = self.scheduler.sigmas  # shape (num_steps+1,), sigma[0]=max, sigma[-1]≈0

        # Queue positions: evenly spaced through the sigma schedule
        # index 0 (head): sigma≈0 (cleanest)
        # index K-1 (tail): sigma=max (noisiest)
        queue_sigma_indices = torch.linspace(
            len(sigmas) - 1, 0, cfg.queue_size, dtype=torch.long
        )
        queue_sigmas = sigmas[queue_sigma_indices]  # (K,)

        # Initialize queue:
        # - Head (index 0): seed latent (fully clean)
        # - Rest: gradually noisier versions of seed
        queue: list[torch.Tensor] = []
        for qi in range(cfg.queue_size):
            if qi == 0:
                queue.append(seed_latent.clone())
            else:
                sigma = queue_sigmas[qi].to(dtype)
                # Add noise proportional to sigma
                noise = torch.randn_like(seed_latent)
                noisy = seed_latent + sigma * noise
                queue.append(noisy)

        print(f"Queue initialized: {cfg.queue_size} frames, "
              f"sigmas {queue_sigmas[-1]:.3f}..{queue_sigmas[0]:.3f}")

        # ── 3. FIFO generation loop ──────────────────────────────────────────
        output_frames: list[Image.Image] = [seed_image]  # frame 0 = seed

        import math
        # Each frame must accumulate enough steps to reach sigma≈0.
        # A frame spends queue_size outer iterations in the queue.
        # steps_per_frame inner steps per outer iteration → total = queue_size * steps_per_frame.
        # We need total ≥ num_inference_steps so frames are fully denoised.
        steps_per_frame = max(1, math.ceil(cfg.num_inference_steps / cfg.queue_size))

        # Lookahead: only freeze the head frame as committed context; update the rest.
        # lookahead_start=1 means frame 0 (head) is context, frames 1..K-1 are denoised.
        lookahead_start = int(cfg.queue_size * cfg.lookahead_fraction) if cfg.use_lookahead else 0

        t_start = time.perf_counter()
        print(f"\nGenerating {num_frames - 1} frames with FIFO-Diffusion "
              f"(queue={cfg.queue_size}, steps_per_frame={steps_per_frame}, "
              f"lookahead_start={lookahead_start})...")

        # Sigmas for queue positions: tail is high sigma (noisy), head is low (clean)
        all_sigmas = self.scheduler.sigmas.to(device, dtype)  # (num_steps+1,)
        sigma_max = all_sigmas[0]  # maximum sigma (pure noise level)

        # Queue step indices: head = near-clean (high index), tail = noisy (index 0)
        # all_sigmas[0] = sigma_max (noisy), all_sigmas[-1] ≈ 0 (clean)
        queue_step_indices = torch.linspace(
            len(all_sigmas) - 2, 0, cfg.queue_size, dtype=torch.long, device=device
        )  # [clean_idx, ..., noisy_idx]

        for frame_out_idx in range(num_frames - 1):
            # Run steps_per_frame denoising steps to advance all queue frames
            for _ in range(steps_per_frame):
                # Stack queue as sequence: (1, K, C, H, W)
                latent_seq = torch.stack(queue, dim=1)

                # Per-frame timesteps: each frame has its own sigma level
                t_per_frame = (all_sigmas[queue_step_indices] * 1000.0).float()
                t_per_frame = t_per_frame.unsqueeze(0)  # (1, K)

                # Build per-frame action tensor for this queue window
                actions_tensor = None
                if actions is not None and cfg.use_actions:
                    # Queue holds frames [frame_out_idx, ..., frame_out_idx + K - 1]
                    K = cfg.queue_size
                    action_indices = [
                        actions[min(frame_out_idx + qi, len(actions) - 1)]
                        for qi in range(K)
                    ]
                    actions_tensor = torch.tensor(
                        [action_indices], dtype=torch.long, device=device
                    )  # (1, K)

                # Forward pass through ZImageWorldModel with text conditioning
                with torch.inference_mode():
                    if cfg.use_cfg and null_cap_feats is not None:
                        # Classifier-Free Guidance: run null + text, extrapolate
                        v_null = self.model(
                            latent_seq, t_per_frame,
                            actions=actions_tensor,
                            cap_feat_override=null_cap_feats,
                        )
                        v_text = self.model(
                            latent_seq, t_per_frame,
                            actions=actions_tensor,
                            cap_feat_override=cap_feats,
                        )
                        v_pred = v_null + cfg.guidance_scale * (v_text - v_null)
                    else:
                        v_pred = self.model(
                            latent_seq, t_per_frame,
                            actions=actions_tensor,
                            cap_feat_override=cap_feats,
                        )  # (1, K, C, H, W)

                # Euler step: update all frames (or only rear portion with lookahead)
                for qi in range(lookahead_start, cfg.queue_size):
                    si = queue_step_indices[qi].item()
                    if si + 1 < len(all_sigmas):
                        sigma_t = all_sigmas[si]
                        sigma_next = all_sigmas[si + 1]
                        dt = sigma_next - sigma_t  # negative (sigma decreases toward clean)
                        # Flow matching Euler: x_next = x + v * dt
                        queue[qi] = queue[qi] + dt * v_pred[0, qi]

                # Advance all step indices toward cleaner state
                queue_step_indices = torch.clamp(
                    queue_step_indices + 1, max=len(all_sigmas) - 2
                )

            # Decode head frame (most-denoised position) and collect as output
            frame_img = self._decode_latent(queue[0])
            output_frames.append(frame_img)

            # FIFO shift: dequeue head, enqueue new frame at tail
            last_clean = queue[0].clone()  # Save head (most recently decoded frame)
            queue.pop(0)
            noise = torch.randn(1, C, H, W, device=device, dtype=dtype)
            if cfg.anchor_init:
                # Partial anchor init: new tail = last_clean + frac * σ_max * noise.
                # anchor_noise_frac=1.0 → full noise (high creativity, more drift)
                # anchor_noise_frac=0.75 → 75% noise (balanced coherence / motion)
                frac = cfg.anchor_noise_frac
                new_sigma = sigma_max * frac
                new_frame = last_clean + new_sigma * noise
                # Find the sigma index corresponding to new_sigma so denoising
                # starts from the right noise level (not always from sigma_max).
                if frac < 0.99:
                    dist = torch.abs(all_sigmas - new_sigma)
                    new_tail_idx = int(dist.argmin().item())
                    new_tail_idx = min(new_tail_idx, len(all_sigmas) - 2)
                else:
                    new_tail_idx = 0  # sigma_max → start from beginning
            else:
                # Pure noise: fully independent generation (less coherent)
                new_frame = noise * sigma_max
                new_tail_idx = 0
            queue.append(new_frame)

            # Shift the sigma index array: drop head's index, append new tail's index.
            queue_step_indices = torch.cat([
                queue_step_indices[1:],
                torch.tensor([new_tail_idx], dtype=torch.long, device=device),
            ])

            elapsed = time.perf_counter() - t_start
            fps = (frame_out_idx + 1) / elapsed
            print(f"  Frame {frame_out_idx + 2}/{num_frames} "
                  f"[{elapsed:.1f}s, {fps:.2f} gen-fps]", flush=True)

        total = time.perf_counter() - t_start
        print(f"\nGenerated {len(output_frames)} frames in {total:.1f}s "
              f"({len(output_frames)/total:.2f} gen-fps)")

        return output_frames

    def _generate_seed_image(self, prompt: str, seed: int = 42) -> Image.Image:
        """Use ZImagePipeline to generate a high-quality seed frame.

        Reuses self._zimage_pipe if available (avoids loading the 7B model twice).
        """
        if hasattr(self, "_zimage_pipe"):
            pipe = self._zimage_pipe
            result = pipe(
                prompt=prompt,
                height=self.config.height,
                width=self.config.width,
                num_inference_steps=self.config.seed_steps,
                guidance_scale=self.config.guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            )
            return result.images[0]

        # Fallback: load a fresh pipeline if _zimage_pipe was not kept
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(
            getattr(self, "_model_path", "weights/Z-Image-Turbo"),
            torch_dtype=self.dtype,
        )
        pipe = pipe.to(self.device)
        result = pipe(
            prompt=prompt,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=self.config.seed_steps,
            guidance_scale=self.config.guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        img = result.images[0]
        del pipe
        torch.cuda.empty_cache()
        return img

    @staticmethod
    def save_gif(
        frames: list[Image.Image],
        path: Union[str, Path],
        fps: float = 6.0,
    ) -> None:
        path = Path(path)
        duration_ms = int(1000 / fps)
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"Saved GIF: {path} ({len(frames)} frames @ {fps} fps)")

    @staticmethod
    def save_video(
        frames: list[Image.Image],
        path: Union[str, Path],
        fps: float = 6.0,
    ) -> None:
        import imageio
        path = Path(path)
        with imageio.get_writer(str(path), fps=fps) as writer:
            for frame in frames:
                writer.append_data(np.array(frame))
        print(f"Saved MP4: {path} ({len(frames)} frames @ {fps} fps)")
