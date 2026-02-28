#!/usr/bin/env python3
"""Debug: replicate the Z-Image pipeline loop exactly."""
import torch
import numpy as np
from PIL import Image


def main():
    from diffusers import ZImagePipeline

    print("Loading Z-Image pipeline...")
    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    transformer = pipe.transformer
    vae = pipe.vae
    scheduler = pipe.scheduler

    # Get text embeddings
    prompt_embeds = pipe.encode_prompt("a red cube on white background")[0]
    print(f"Prompt embeds: {len(prompt_embeds)} items, shape: {prompt_embeds[0].shape}")

    # Replicate pipeline's denoising loop EXACTLY
    print("\n--- Replicating pipeline loop ---")
    scheduler.set_timesteps(8)
    scheduler.sigma_min = 0.0

    latents = torch.randn(1, 16, 32, 32, device="cuda", dtype=torch.bfloat16)

    for i, t in enumerate(scheduler.timesteps):
        # This is EXACTLY what the pipeline does
        timestep = t.expand(latents.shape[0]).to("cuda")
        timestep = (1000 - timestep) / 1000

        latent_model_input = latents.to(transformer.dtype)
        latent_model_input_list = list(latent_model_input.unsqueeze(2))  # List[(C, 1, H, W)]
        prompt_embeds_input = list(prompt_embeds)

        with torch.no_grad():
            model_out_list = transformer(
                latent_model_input_list, timestep, prompt_embeds_input, return_dict=False
            )[0]

        # Stack model output
        model_output = torch.stack(
            [o.squeeze(1) for o in model_out_list], dim=0
        )  # (B, C, H, W)

        # Use scheduler step
        latents = scheduler.step(model_output, t, latents, return_dict=False)[0]

        print(f"  Step {i}: t={t.item():.1f}, latents range=[{latents.min():.3f}, {latents.max():.3f}]")

    # Decode
    with torch.no_grad():
        decoded = vae.decode(latents / vae.config.scaling_factor).sample
        decoded = ((decoded + 1.0) / 2.0).clamp(0, 1)
        img = (decoded[0].float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save("/tmp/test_exact_loop.png")
    print("  Saved: /tmp/test_exact_loop.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
