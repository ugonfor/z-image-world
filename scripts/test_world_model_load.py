#!/usr/bin/env python3
"""Test loading Z-Image-Turbo as a world model on DGX Spark."""

import torch
import time


def main():
    print("=" * 60)
    print("Z-Image World Model Load Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU Memory: {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total")

    # Step 1: Load the model
    print("\n--- Step 1: Loading ZImageWorldModel ---")
    t0 = time.time()

    from models.zimage_world_model import ZImageWorldModel

    model = ZImageWorldModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        device=device,
    )
    print(f"Loaded in {time.time() - t0:.1f}s")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU Memory after load: {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total")

    # Step 2: Test forward pass with single frame
    print("\n--- Step 2: Single frame forward pass ---")
    with torch.no_grad():
        # Fake latent (1 frame, 16 channels, 64x64 spatial = 512x512 image)
        latent = torch.randn(1, 1, 16, 64, 64, device=device, dtype=torch.bfloat16)
        timestep = torch.tensor([500.0], device=device)

        t0 = time.time()
        output = model(latent, timestep)
        print(f"Single frame output shape: {output.shape}")
        print(f"Forward pass took: {time.time() - t0:.2f}s")

    # Step 3: Test forward with multiple frames + actions
    print("\n--- Step 3: Multi-frame forward with actions ---")
    with torch.no_grad():
        latent = torch.randn(1, 4, 16, 64, 64, device=device, dtype=torch.bfloat16)
        timestep = torch.tensor([500.0], device=device)
        actions = torch.randint(0, 17, (1, 4), device=device)

        t0 = time.time()
        output = model(latent, timestep, actions=actions)
        print(f"Multi-frame output shape: {output.shape}")
        print(f"Forward pass took: {time.time() - t0:.2f}s")

    # Step 4: Test VAE encode/decode
    print("\n--- Step 4: VAE encode/decode ---")
    with torch.no_grad():
        fake_images = torch.rand(1, 2, 3, 512, 512, device=device, dtype=torch.bfloat16)
        latents = model.encode_frames(fake_images)
        print(f"Encoded shape: {latents.shape}")

        decoded = model.decode_latents(latents)
        print(f"Decoded shape: {decoded.shape}")

    # Step 5: Check trainable params
    print("\n--- Step 5: Parameter summary ---")
    total = model.num_total_params()
    trainable = model.num_trainable_params()
    print(f"Total params: {total:,} ({total / 1e9:.2f}B)")
    print(f"Trainable params: {trainable:,} ({trainable / 1e6:.1f}M)")
    print(f"Frozen params: {total - trainable:,} ({(total - trainable) / 1e9:.2f}B)")

    if torch.cuda.is_available():
        free, total_mem = torch.cuda.mem_get_info(0)
        print(f"\nFinal GPU Memory: {free / 1e9:.1f}GB free / {total_mem / 1e9:.1f}GB total")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
