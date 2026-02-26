#!/usr/bin/env python3
"""
Inspect Z-Image-Turbo model architecture.

Downloads and analyzes the model to extract exact dimensions,
layer names, and weight shapes needed for world model adaptation.
"""

import json
import os
import sys
from pathlib import Path

import torch


def inspect_model():
    """Load Z-Image-Turbo and inspect its architecture."""
    print("=" * 60)
    print("Z-Image-Turbo Model Inspection")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        print(f"GPU Memory: {free_mem / 1e9:.1f}GB free / {total_mem / 1e9:.1f}GB total")
    else:
        print("WARNING: No GPU available, loading on CPU")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pipeline
    print("\n--- Loading Z-Image-Turbo pipeline ---")
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
    )

    # Output directory
    out_dir = Path("weight_inspection")
    out_dir.mkdir(exist_ok=True)

    # ==========================================
    # Inspect Transformer
    # ==========================================
    print("\n" + "=" * 60)
    print("TRANSFORMER ARCHITECTURE")
    print("=" * 60)

    transformer = pipe.transformer
    print(f"\nClass: {type(transformer).__name__}")
    print(f"Config: {transformer.config}")

    # Print module tree
    print("\n--- Module Tree ---")
    module_tree = []
    for name, module in transformer.named_modules():
        if name:  # Skip root
            depth = name.count(".")
            indent = "  " * depth
            module_tree.append(f"{indent}{name}: {type(module).__name__}")
            if depth < 2:  # Print top-level modules
                print(f"{indent}{name}: {type(module).__name__}")

    with open(out_dir / "transformer_module_tree.txt", "w") as f:
        f.write("\n".join(module_tree))
    print(f"  (Full tree saved to {out_dir}/transformer_module_tree.txt)")

    # Print parameter shapes
    print("\n--- Parameter Shapes ---")
    param_shapes = {}
    total_params = 0
    trainable_params = 0
    for name, param in transformer.named_parameters():
        param_shapes[name] = {
            "shape": list(param.shape),
            "numel": param.numel(),
            "dtype": str(param.dtype),
        }
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    with open(out_dir / "transformer_params.json", "w") as f:
        json.dump(param_shapes, f, indent=2)

    print(f"  Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Parameter count saved to {out_dir}/transformer_params.json")

    # Extract key dimensions
    print("\n--- Key Dimensions ---")
    # Find hidden_dim from first layer's attention
    first_param_name = list(param_shapes.keys())[0]
    print(f"  First param: {first_param_name} -> {param_shapes[first_param_name]['shape']}")

    # Try to find attention weight dimensions
    for name, info in param_shapes.items():
        if "to_q" in name and "weight" in name:
            shape = info["shape"]
            print(f"  Attention Q weight ({name}): {shape}")
            if len(shape) == 2:
                print(f"    -> hidden_dim = {shape[1]}, out_dim = {shape[0]}")
            break

    for name, info in param_shapes.items():
        if "to_k" in name and "weight" in name:
            shape = info["shape"]
            print(f"  Attention K weight ({name}): {shape}")
            break

    for name, info in param_shapes.items():
        if "to_v" in name and "weight" in name:
            shape = info["shape"]
            print(f"  Attention V weight ({name}): {shape}")
            break

    # Count layers
    layer_names = set()
    for name in param_shapes:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0:
                prefix = ".".join(parts[:i])
                layer_names.add((prefix, int(p)))

    layer_counts = {}
    for prefix, idx in layer_names:
        if prefix not in layer_counts:
            layer_counts[prefix] = 0
        layer_counts[prefix] = max(layer_counts[prefix], idx + 1)

    print("\n  Layer counts:")
    for prefix, count in sorted(layer_counts.items()):
        print(f"    {prefix}: {count} layers")

    # ==========================================
    # Inspect VAE
    # ==========================================
    print("\n" + "=" * 60)
    print("VAE ARCHITECTURE")
    print("=" * 60)

    vae = pipe.vae
    print(f"\nClass: {type(vae).__name__}")
    print(f"Config: {vae.config}")

    vae_params = sum(p.numel() for p in vae.parameters())
    print(f"Total parameters: {vae_params:,} ({vae_params / 1e6:.1f}M)")

    vae_param_shapes = {}
    for name, param in vae.named_parameters():
        vae_param_shapes[name] = {
            "shape": list(param.shape),
            "numel": param.numel(),
        }
    with open(out_dir / "vae_params.json", "w") as f:
        json.dump(vae_param_shapes, f, indent=2)

    # ==========================================
    # Inspect Text Encoder
    # ==========================================
    print("\n" + "=" * 60)
    print("TEXT ENCODER")
    print("=" * 60)

    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        text_enc = pipe.text_encoder
        print(f"Class: {type(text_enc).__name__}")
        te_params = sum(p.numel() for p in text_enc.parameters())
        print(f"Total parameters: {te_params:,} ({te_params / 1e9:.2f}B)")
    else:
        print("No text_encoder found in pipeline")

    # ==========================================
    # Test Forward Pass
    # ==========================================
    print("\n" + "=" * 60)
    print("TEST FORWARD PASS")
    print("=" * 60)

    pipe.to(device)

    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        print(f"GPU Memory after loading: {free_mem / 1e9:.1f}GB free / {total_mem / 1e9:.1f}GB total")
        print(f"Model memory usage: {(total_mem - free_mem) / 1e9:.1f}GB")

    print("\nGenerating test image (512x512)...")
    with torch.no_grad():
        result = pipe(
            prompt="a simple red cube on white background",
            height=512,
            width=512,
            num_inference_steps=8,
            guidance_scale=0.0,
        )
        image = result.images[0]
        image.save(str(out_dir / "test_generation.png"))
        print(f"Test image saved to {out_dir}/test_generation.png")

    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        print(f"GPU Memory after generation: {free_mem / 1e9:.1f}GB free / {total_mem / 1e9:.1f}GB total")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Transformer class: {type(transformer).__name__}")
    print(f"Transformer params: {total_params / 1e9:.2f}B")
    print(f"VAE class: {type(vae).__name__}")
    print(f"VAE params: {vae_params / 1e6:.1f}M")
    print(f"Files saved to: {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    inspect_model()
