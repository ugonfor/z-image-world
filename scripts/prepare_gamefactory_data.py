#!/usr/bin/env python3
"""
Prepare KlingTeam/GameFactory-Dataset (GF-Minecraft) for Stage 2 training.

Downloads mp4 + JSON pairs from HuggingFace and converts GameFactory's
ws/ad/scs action format to our discrete 8-action vocabulary, producing
mp4 + _actions.json pairs ready for scripts/train_zimage_stage2.py.

GameFactory action encoding (per-frame):
  ws:  0=none, 1=forward, 2=backward
  ad:  0=none, 1=left,    2=right
  scs: 0=none, 1=jump,    2=sneak/crouch, 3=sprint

Our action space (must match train_zimage_stage2.py):
  0=idle, 1=forward, 2=backward, 3=left, 4=right, 5=run, 6=jump, 7=interact

Data splits available:
  sample-10.zip   ~527 MB    10 clips   (good for pipeline validation)
  data_269.zip    ~14 GB    269 clips   (keyboard-only, good for quick training)
  data_2003/      ~104 GB  2003 clips   (full mouse+keyboard, best quality)

Usage:
    # Validate with sample (fast, 10 clips):
    python scripts/prepare_gamefactory_data.py --split sample

    # Download 269-clip keyboard subset (~14 GB):
    python scripts/prepare_gamefactory_data.py --split data_269

    # Convert already-downloaded files in a directory:
    python scripts/prepare_gamefactory_data.py \\
        --local_dir /tmp/gamefactory/sample_10/sample-10 \\
        --output_dir data/videos/gamefactory

    # Then train:
    python scripts/train_zimage_stage2.py \\
        --data_dir data/videos/gamefactory \\
        --stage1_checkpoint checkpoints/zimage_stage1_v3/world_model_final.pt \\
        --epochs 30
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

# ── Action mapping ─────────────────────────────────────────────────

ACTION_IDLE     = 0
ACTION_FORWARD  = 1
ACTION_BACKWARD = 2
ACTION_LEFT     = 3
ACTION_RIGHT    = 4
ACTION_RUN      = 5
ACTION_JUMP     = 6
ACTION_INTERACT = 7


def gamefactory_to_action(frame: dict) -> int:
    """Convert one GameFactory frame dict to our discrete action.

    Priority order:
      1. Jump (scs=1) — most visually distinctive
      2. Sprint (ws=1 + scs=3) → RUN
      3. Forward (ws=1)
      4. Backward (ws=2)
      5. Strafe left (ad=1)
      6. Strafe right (ad=2)
      7. Idle (default)

    Note: sneak (scs=2) and camera-only motion are both mapped to IDLE,
    as they lack strong visual velocity signals.
    """
    ws  = int(frame.get("ws",  0))
    ad  = int(frame.get("ad",  0))
    scs = int(frame.get("scs", 0))

    if scs == 1:
        return ACTION_JUMP
    if ws == 1 and scs == 3:
        return ACTION_RUN
    if ws == 1:
        return ACTION_FORWARD
    if ws == 2:
        return ACTION_BACKWARD
    if ad == 1:
        return ACTION_LEFT
    if ad == 2:
        return ACTION_RIGHT
    return ACTION_IDLE


def convert_json_to_actions(json_path: Path) -> list[int]:
    """Load a GameFactory JSON file and return List[int] actions."""
    with open(json_path) as f:
        data = json.load(f)
    frames_dict = data.get("actions", {})
    if not frames_dict:
        return []
    # Keys are string frame indices "0", "1", ...
    n_frames = max(int(k) for k in frames_dict) + 1
    actions = []
    for i in range(n_frames):
        frame = frames_dict.get(str(i), {})
        actions.append(gamefactory_to_action(frame))
    return actions


def process_clip(
    mp4_path: Path,
    json_path: Path,
    output_dir: Path,
    clip_name: str,
) -> bool:
    """Convert one GameFactory clip to our format. Returns True on success."""
    out_mp4  = output_dir / f"{clip_name}.mp4"
    out_json = output_dir / f"{clip_name}_actions.json"

    if out_mp4.exists() and out_json.exists():
        return True  # Resume

    try:
        actions = convert_json_to_actions(json_path)
        if not actions:
            return False
    except Exception as e:
        print(f"  WARN: action conversion failed for {json_path.name}: {e}")
        return False

    # Copy mp4 (or symlink)
    if not out_mp4.exists():
        shutil.copy2(str(mp4_path), str(out_mp4))

    with open(out_json, "w") as f:
        json.dump(actions, f)

    return True


def _find_json(mp4_path: Path, root_dir: Path) -> Path | None:
    """Find the action JSON for an mp4 file.

    Supports two GameFactory layouts:
      Layout A (sample-10): mp4 and json are siblings in the same dir
      Layout B (data_269):  mp4 in root/video/, json in root/metadata/
    """
    # Layout A: sibling JSON
    sibling = mp4_path.with_suffix(".json")
    if sibling.exists():
        return sibling
    # Layout B: metadata/<stem>.json relative to common root
    metadata_dir = root_dir / "metadata"
    if metadata_dir.is_dir():
        candidate = metadata_dir / f"{mp4_path.stem}.json"
        if candidate.exists():
            return candidate
    return None


def process_directory(src_dir: Path, output_dir: Path) -> int:
    """Convert all GameFactory clips in a directory. Returns clip count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_files = sorted(src_dir.rglob("*.mp4"))
    if not mp4_files:
        print(f"  No .mp4 files found in {src_dir}")
        return 0

    ok = 0
    skipped = 0
    missing_json = 0
    for mp4_path in mp4_files:
        json_path = _find_json(mp4_path, src_dir)
        if json_path is None:
            missing_json += 1
            continue

        clip_name = mp4_path.stem  # e.g. seed_1_part_1
        out_mp4 = output_dir / f"{clip_name}.mp4"
        out_json = output_dir / f"{clip_name}_actions.json"

        if out_mp4.exists() and out_json.exists():
            skipped += 1
            continue

        if process_clip(mp4_path, json_path, output_dir, clip_name):
            ok += 1
        else:
            print(f"  FAIL: {clip_name}")

    if missing_json:
        print(f"  Skipped {missing_json} mp4s with no matching JSON")
    return ok + skipped


def download_and_convert(split: str, output_dir: Path, cache_dir: Path) -> int:
    """Download a dataset split and convert all clips."""
    from huggingface_hub import hf_hub_download
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    print(f"\nDownloading {split}...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if split == "sample":
        local = hf_hub_download(
            repo_id="KlingTeam/GameFactory-Dataset",
            filename="GF-Minecraft/sample-10.zip",
            repo_type="dataset",
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        zip_path = Path(local)
        extract_dir = cache_dir / "sample_10_extract"
        if not extract_dir.exists():
            print(f"Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
        return process_directory(extract_dir, output_dir)

    elif split == "data_269":
        local = hf_hub_download(
            repo_id="KlingTeam/GameFactory-Dataset",
            filename="GF-Minecraft/data_269.zip",
            repo_type="dataset",
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        zip_path = Path(local)
        extract_dir = cache_dir / "data_269_extract"
        if not extract_dir.exists():
            print(f"Extracting {zip_path.name} (may take a minute)...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
        return process_directory(extract_dir, output_dir)

    else:
        raise ValueError(f"Unknown split '{split}'. Choose: sample, data_269")


def print_action_stats(output_dir: Path, sample_n: int = 20):
    """Print action distribution from converted clips."""
    from collections import Counter
    import random

    json_files = list(output_dir.glob("*_actions.json"))
    if not json_files:
        return

    sample = random.sample(json_files, min(sample_n, len(json_files)))
    counts = Counter()
    for p in sample:
        try:
            counts.update(json.loads(p.read_text()))
        except Exception:
            pass

    names = ["idle", "forward", "backward", "left", "right", "run", "jump", "interact"]
    total = sum(counts.values())
    if total == 0:
        return

    print(f"\nAction distribution ({len(sample)} sampled clips):")
    for i, name in enumerate(names):
        pct = counts.get(i, 0) / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {i} {name:<10} {pct:5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GameFactory-GF-Minecraft data for Stage 2 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", choices=["sample", "data_269"],
                        default="sample",
                        help="Dataset split to download (sample=10 clips, data_269=269 clips)")
    parser.add_argument("--output_dir", default="data/videos/gamefactory",
                        help="Output directory for mp4 + _actions.json pairs")
    parser.add_argument("--cache_dir", default="/tmp/gamefactory_cache",
                        help="Cache dir for downloaded zips")
    parser.add_argument("--local_dir", default=None,
                        help="Convert already-downloaded GameFactory directory instead of downloading")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== GameFactory Data Preparation ===")
    print(f"Output: {output_dir}")

    if args.local_dir:
        print(f"Source: {args.local_dir}")
        n = process_directory(Path(args.local_dir), output_dir)
    else:
        print(f"Split:  {args.split}")
        n = download_and_convert(args.split, output_dir, Path(args.cache_dir))

    print(f"\nDone: {n} clips in {output_dir}")
    print_action_stats(output_dir)

    print(f"\nTo start Stage 2 training:")
    print(f"  python scripts/train_zimage_stage2.py \\")
    print(f"      --data_dir {output_dir} \\")
    print(f"      --stage1_checkpoint checkpoints/zimage_stage1_v3/world_model_final.pt \\")
    print(f"      --epochs 30")


if __name__ == "__main__":
    main()
