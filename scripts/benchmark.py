#!/usr/bin/env python3
"""
Benchmark Script for Z-Image World Model

Measures inference performance including:
- Frames per second (FPS)
- Latency per frame
- Memory usage

Usage:
    python scripts/benchmark.py --model checkpoints/final.pt
"""

import argparse
import time
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import RealtimePipeline, PipelineConfig
from models import ActionSpace


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Z-Image World Model")

    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames to generate")
    parser.add_argument("--warmup_frames", type=int, default=10, help="Warmup frames")
    parser.add_argument("--resolution", type=int, nargs=2, default=[480, 640], help="Resolution (H W)")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Denoising steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")

    return parser.parse_args()


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def main():
    args = parse_args()

    print("=" * 60)
    print("Z-Image World Model Benchmark")
    print("=" * 60)

    device = torch.device(args.device)

    # Configuration
    config = PipelineConfig(
        height=args.resolution[0],
        width=args.resolution[1],
        num_inference_steps=args.num_inference_steps,
        device=args.device,
        compile_model=args.compile,
    )

    print(f"\nConfiguration:")
    print(f"  Resolution: {args.resolution[0]}x{args.resolution[1]}")
    print(f"  Denoising steps: {args.num_inference_steps}")
    print(f"  Device: {args.device}")
    print(f"  Compile: {args.compile}")

    # Load or create pipeline
    if args.model:
        print(f"\nLoading model from {args.model}...")
        pipeline = RealtimePipeline.from_pretrained(args.model, config)
    else:
        print("\nUsing dummy model for benchmarking...")
        from demo.interactive_app import create_dummy_pipeline
        pipeline = create_dummy_pipeline(args.device)

    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Create initial frame
    initial_frame = torch.randn(
        1, 3, args.resolution[0], args.resolution[1],
        device=device, dtype=torch.bfloat16
    )
    pipeline.set_initial_frame(initial_frame)

    # Warmup
    print(f"\nWarming up ({args.warmup_frames} frames)...")
    for _ in range(args.warmup_frames):
        _ = pipeline.step(ActionSpace.IDLE)

    # Reset stats after warmup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Benchmark
    print(f"\nBenchmarking ({args.num_frames} frames)...")

    # Create varied action sequence
    actions = [
        ActionSpace.MOVE_FORWARD, ActionSpace.MOVE_FORWARD,
        ActionSpace.MOVE_RIGHT, ActionSpace.MOVE_FORWARD,
        ActionSpace.LOOK_LEFT, ActionSpace.MOVE_FORWARD,
        ActionSpace.JUMP, ActionSpace.MOVE_FORWARD,
    ]

    latencies = []
    start_time = time.perf_counter()

    for i in range(args.num_frames):
        action = actions[i % len(actions)]

        frame_start = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        _ = pipeline.step(action)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        frame_time = time.perf_counter() - frame_start
        latencies.append(frame_time * 1000)  # Convert to ms

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i + 1}/{args.num_frames} frames")

    total_time = time.perf_counter() - start_time

    # Calculate statistics
    latencies = np.array(latencies)
    avg_fps = args.num_frames / total_time
    avg_latency = latencies.mean()
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    min_latency = latencies.min()
    max_latency = latencies.max()

    peak_memory = get_gpu_memory_usage()

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nThroughput:")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Total time: {total_time:.2f}s for {args.num_frames} frames")

    print(f"\nLatency (ms):")
    print(f"  Average: {avg_latency:.2f}")
    print(f"  P50: {p50_latency:.2f}")
    print(f"  P95: {p95_latency:.2f}")
    print(f"  P99: {p99_latency:.2f}")
    print(f"  Min: {min_latency:.2f}")
    print(f"  Max: {max_latency:.2f}")

    print(f"\nMemory:")
    print(f"  Peak GPU memory: {peak_memory:.2f} MB")

    print("\n" + "=" * 60)

    # Check if we meet target performance
    if avg_fps >= 20:
        print("✓ Performance GOOD: Meets 20+ FPS target")
    elif avg_fps >= 10:
        print("△ Performance ACCEPTABLE: 10-20 FPS")
    else:
        print("✗ Performance NEEDS IMPROVEMENT: <10 FPS")

    return {
        "fps": avg_fps,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "peak_memory_mb": peak_memory,
    }


if __name__ == "__main__":
    main()
