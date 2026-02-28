"""
INT8 Dynamic Quantization for ZImageWorldModel

Applies INT8 dynamic quantization to the trainable modules (temporal
attention and action injection layers). The frozen Z-Image spatial
transformer keeps its original bfloat16 weights.

INT8 dynamic quantization:
- Weights quantized to INT8 offline (at quantize time)
- Activations quantized to INT8 dynamically at runtime
- Requires no calibration dataset
- Typically 2–4x memory reduction + speedup on CPU; GPU benefit is via
  INT8 GEMM (available on A100 via torch.ao.nn.quantized.dynamic)

Usage:
    from models.quantization import quantize_temporal_layers, QuantizationReport

    report = quantize_temporal_layers(world_model)
    print(report)
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class QuantizationReport:
    """Summary of quantization results."""

    modules_quantized: list[str] = field(default_factory=list)
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    latency_before_ms: Optional[float] = None
    latency_after_ms: Optional[float] = None

    @property
    def compression_ratio(self) -> float:
        if self.quantized_size_mb > 0:
            return self.original_size_mb / self.quantized_size_mb
        return 1.0

    @property
    def speedup(self) -> Optional[float]:
        if self.latency_before_ms and self.latency_after_ms and self.latency_after_ms > 0:
            return self.latency_before_ms / self.latency_after_ms
        return None

    def __str__(self) -> str:
        lines = ["Quantization Report", "=" * 40]
        lines.append(f"Modules quantized: {len(self.modules_quantized)}")
        for m in self.modules_quantized[:5]:
            lines.append(f"  {m}")
        if len(self.modules_quantized) > 5:
            lines.append(f"  ... and {len(self.modules_quantized) - 5} more")
        lines.append(f"Memory: {self.original_size_mb:.1f} MB → {self.quantized_size_mb:.1f} MB "
                     f"(compression: {self.compression_ratio:.2f}x)")
        if self.speedup is not None:
            lines.append(f"Latency: {self.latency_before_ms:.1f} ms → {self.latency_after_ms:.1f} ms "
                         f"(speedup: {self.speedup:.2f}x)")
        return "\n".join(lines)


def _module_size_mb(module: nn.Module) -> float:
    """Compute parameter + buffer size of a module in MB."""
    total = sum(p.numel() * p.element_size() for p in module.parameters())
    total += sum(b.numel() * b.element_size() for b in module.buffers())
    return total / (1024 * 1024)


def _benchmark_module(
    module: nn.Module,
    sample_input: torch.Tensor,
    num_runs: int = 20,
    warmup: int = 5,
) -> float:
    """Measure forward pass latency in milliseconds."""
    module.eval()
    device = next(module.parameters(), sample_input).device

    with torch.no_grad():
        for _ in range(warmup):
            _ = module(sample_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(num_runs):
            _ = module(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    return (t1 - t0) / num_runs * 1000  # ms


def quantize_temporal_layers(
    world_model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    benchmark: bool = False,
) -> QuantizationReport:
    """Quantize the trainable temporal and action layers to INT8.

    Only quantizes ZImageWorldModel's new layers (temporal_layers,
    action_injections, action_encoder). The frozen Z-Image spatial
    transformer is left as-is.

    Args:
        world_model: ZImageWorldModel instance.
        dtype: Quantization dtype (default: qint8).
        benchmark: If True, measure latency before/after (CPU only).

    Returns:
        QuantizationReport with details.
    """
    report = QuantizationReport()

    # Collect trainable modules to quantize
    modules_to_quantize: dict[str, nn.Module] = {}

    if hasattr(world_model, "temporal_layers"):
        modules_to_quantize["temporal_layers"] = world_model.temporal_layers

    if hasattr(world_model, "action_injections"):
        modules_to_quantize["action_injections"] = world_model.action_injections

    if hasattr(world_model, "action_encoder"):
        modules_to_quantize["action_encoder"] = world_model.action_encoder

    if not modules_to_quantize:
        return report

    # Measure sizes before quantization
    for name, module in modules_to_quantize.items():
        report.original_size_mb += _module_size_mb(module)

    # Apply INT8 dynamic quantization to each module
    for name, module in modules_to_quantize.items():
        quantized = torch.quantization.quantize_dynamic(
            module,
            qconfig_spec={nn.Linear},
            dtype=dtype,
        )
        # Replace module in-place in world_model
        parts = name.split(".")
        parent = world_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], quantized)

        # Record quantized module names
        for subname, _ in quantized.named_modules():
            full_name = f"{name}.{subname}" if subname else name
            report.modules_quantized.append(full_name)

    # Measure sizes after quantization
    for name in modules_to_quantize:
        module = getattr(world_model, name.split(".")[0])
        report.quantized_size_mb += _module_size_mb(module)

    # Only remove duplicates from report
    report.modules_quantized = [
        m for m in report.modules_quantized
        if not any(m.startswith(other + ".") for other in report.modules_quantized
                   if other != m and m.startswith(other))
    ]

    return report


def estimate_quantized_size(world_model: nn.Module) -> dict[str, float]:
    """Estimate memory usage of each component (MB).

    Args:
        world_model: ZImageWorldModel instance.

    Returns:
        Dict mapping component name → size in MB.
    """
    sizes = {}

    if hasattr(world_model, "transformer"):
        sizes["transformer (spatial, frozen)"] = _module_size_mb(world_model.transformer)

    if hasattr(world_model, "vae"):
        sizes["vae (frozen)"] = _module_size_mb(world_model.vae)

    if hasattr(world_model, "temporal_layers"):
        sizes["temporal_layers (trainable)"] = _module_size_mb(world_model.temporal_layers)

    if hasattr(world_model, "action_injections"):
        sizes["action_injections (trainable)"] = _module_size_mb(world_model.action_injections)

    if hasattr(world_model, "action_encoder"):
        sizes["action_encoder (trainable)"] = _module_size_mb(world_model.action_encoder)

    return sizes
