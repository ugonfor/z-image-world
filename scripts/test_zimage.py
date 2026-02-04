#!/usr/bin/env python3
"""Test Z-Image model integration."""

import torch
from diffusers import ZImagePipeline

print("Loading Z-Image-Turbo model...")
print("This may take a while on first run (downloading ~12GB)...")

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

print("Model loaded! Generating test image...")

prompt = "A beautiful fantasy landscape with mountains and a crystal lake, sunset lighting, highly detailed, 8k"

image = pipe(
    prompt=prompt,
    height=512,
    width=512,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

output_path = "test_zimage_output.png"
image.save(output_path)
print(f"Image saved to {output_path}")
print("Success! Z-Image is working.")
