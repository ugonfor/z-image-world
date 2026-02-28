#!/usr/bin/env python3
"""
Quick baseline test: verify Z-Image-Turbo works standalone.
1. Text-to-image generation (proves the model works)
2. VAE encode/decode roundtrip (proves our pipeline works)
3. Single-frame denoising through ZImageWorldModel (proves our wrapper works)
"""

import sys
import os
from pathlib import Path

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image


def test_zimage_text_to_image():
    """Test 1: Pure Z-Image text-to-image."""
    print("=" * 60)
    print("TEST 1: Z-Image-Turbo text-to-image")
    print("=" * 60)

    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    prompt = "A beautiful landscape with mountains and a lake, photorealistic"
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=512,
        width=512,
    ).images[0]

    output_dir = Path("inference_output/baseline_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / "text_to_image.png")
    print(f"Saved: {output_dir / 'text_to_image.png'}")
    print(f"Image size: {image.size}")

    # Keep VAE for test 2
    vae = pipe.vae
    del pipe.transformer
    torch.cuda.empty_cache()
    return vae


def test_vae_roundtrip(vae):
    """Test 2: VAE encode → decode roundtrip."""
    print("\n" + "=" * 60)
    print("TEST 2: VAE encode/decode roundtrip")
    print("=" * 60)

    output_dir = Path("inference_output/baseline_test")

    # Load the image we just generated
    img = Image.open(output_dir / "text_to_image.png").convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    img_tensor = img_tensor.to("cuda", dtype=torch.bfloat16)

    # Normalize to [-1, 1]
    img_normalized = img_tensor * 2.0 - 1.0

    # Encode
    with torch.no_grad():
        posterior = vae.encode(img_normalized)
        if hasattr(posterior, "latent_dist"):
            latents = posterior.latent_dist.sample()
        else:
            latents = posterior

    print(f"Latent shape: {latents.shape}")
    print(f"Latent range: [{latents.min():.3f}, {latents.max():.3f}]")

    # Scale (Z-Image uses scaling_factor and shift_factor)
    scaling = vae.config.scaling_factor
    shift = vae.config.shift_factor
    print(f"VAE scaling_factor={scaling}, shift_factor={shift}")

    latents_scaled = (latents - shift) * scaling

    # Unscale
    latents_unscaled = latents_scaled / scaling + shift

    # Decode
    with torch.no_grad():
        decoded = vae.decode(latents_unscaled)
        if hasattr(decoded, "sample"):
            decoded = decoded.sample

    # Normalize to [0, 1]
    decoded = (decoded + 1.0) / 2.0
    decoded = decoded.clamp(0, 1)

    # Save
    decoded_np = decoded[0].float().cpu().permute(1, 2, 0).numpy()
    decoded_img = Image.fromarray((decoded_np * 255).astype(np.uint8))
    decoded_img.save(output_dir / "vae_roundtrip.png")
    print(f"Saved: {output_dir / 'vae_roundtrip.png'}")

    # Compute PSNR
    original_np = img_tensor[0].float().cpu().permute(1, 2, 0).numpy()
    mse = np.mean((original_np - decoded_np) ** 2)
    psnr = -10 * np.log10(mse + 1e-10)
    print(f"Roundtrip PSNR: {psnr:.1f} dB (higher is better, >30 is good)")

    del vae
    torch.cuda.empty_cache()


def test_world_model_single_frame():
    """Test 3: ZImageWorldModel single-frame generation (no temporal)."""
    print("\n" + "=" * 60)
    print("TEST 3: ZImageWorldModel single-frame denoising")
    print("=" * 60)

    from models.zimage_world_model import ZImageWorldModel

    model = ZImageWorldModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        temporal_every_n=1,
        freeze_spatial=True,
        device="cuda",
    )
    model.eval()

    output_dir = Path("inference_output/baseline_test")

    # Load test image, encode it
    img = Image.open(output_dir / "text_to_image.png").convert("RGB").resize((256, 256))
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)
    img_tensor = img_tensor.to("cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        latents = model.encode_frames(img_tensor)  # (1, 1, 16, H//8, W//8)

    print(f"Encoded latent shape: {latents.shape}")
    print(f"Encoded latent range: [{latents.min():.3f}, {latents.max():.3f}]")

    # Decode back (roundtrip through our wrapper)
    with torch.no_grad():
        decoded = model.decode_latents(latents)
    print(f"Decoded shape: {decoded.shape}")

    # Handle both (B, F, C, H, W) and (B, C, H, W) outputs
    if decoded.dim() == 5:
        decoded_np = decoded[0, 0].float().cpu().permute(1, 2, 0).numpy()
    else:
        decoded_np = decoded[0].float().cpu().permute(1, 2, 0).numpy()
    decoded_img = Image.fromarray((decoded_np * 255).clip(0, 255).astype(np.uint8))
    decoded_img.save(output_dir / "world_model_roundtrip.png")
    print(f"Saved: {output_dir / 'world_model_roundtrip.png'}")

    # Now test: can the model denoise from pure noise to a coherent frame?
    # This tests if the pretrained weights are working through our wrapper
    print("\nTesting denoising from noise...")
    latent_shape = latents.shape[2:]  # (16, H//8, W//8)
    noise = torch.randn(1, 1, *latent_shape, device="cuda", dtype=torch.bfloat16)

    # Flow matching denoising (matching Z-Image's scheduler)
    shift = 3.0
    num_steps = 4
    base_sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device="cuda")
    sigmas = shift * base_sigmas / (1 + (shift - 1) * base_sigmas)

    x_t = noise[:, 0]  # (1, 16, H, W)
    for step_idx in range(num_steps):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]
        dt = sigma_next - sigma

        t_val = (sigma * 1000).long().float()
        t_input = t_val.unsqueeze(0).unsqueeze(0)  # (1, 1)

        with torch.no_grad():
            velocity = model(x_t.unsqueeze(1), t_input)
            if velocity.dim() == 5:
                velocity = velocity[:, 0]

        x_t = x_t + dt * velocity
        print(f"  Step {step_idx+1}/{num_steps}: sigma={sigma:.4f}→{sigma_next:.4f}, x range=[{x_t.min():.3f}, {x_t.max():.3f}]")

    # Decode
    with torch.no_grad():
        generated = model.decode_latents(x_t.unsqueeze(1))
    print(f"Generated shape: {generated.shape}")

    if generated.dim() == 5:
        gen_np = generated[0, 0].float().cpu().permute(1, 2, 0).numpy()
    else:
        gen_np = generated[0].float().cpu().permute(1, 2, 0).numpy()
    gen_img = Image.fromarray((gen_np * 255).clip(0, 255).astype(np.uint8))
    gen_img.save(output_dir / "world_model_from_noise.png")
    print(f"Saved: {output_dir / 'world_model_from_noise.png'}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    vae = test_zimage_text_to_image()
    test_vae_roundtrip(vae)
    test_world_model_single_frame()
    print("\n" + "=" * 60)
    print("ALL BASELINE TESTS COMPLETE")
    print("=" * 60)
