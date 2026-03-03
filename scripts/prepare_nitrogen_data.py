#!/usr/bin/env python3
"""
Prepare NitroGen dataset for Stage 2 training.

Downloads NitroGen gameplay videos with action labels, converts gamepad
actions to our discrete action space, and outputs mp4 + _actions.json pairs
ready for scripts/train_zimage_stage2.py.

Pipeline:
  1. Download SHARD_XXXX.tar.gz from HuggingFace (action labels, ~1.7 GB each)
  2. Scan metadata + parquet files to find movement-rich clips
  3. Download selected video clips from YouTube via yt-dlp (trimmed to 20s)
  4. Convert gamepad actions → discrete indices (0-7)
  5. Output: data/videos/nitrogen/{video_id}_chunk_{n}.mp4
             data/videos/nitrogen/{video_id}_chunk_{n}_actions.json

Action space:
  0=idle, 1=forward, 2=backward, 3=left, 4=right,
  5=run, 6=jump, 7=interact

Usage:
    # Download 2000 clips from 5 spread shards (~10 GB video)
    python scripts/prepare_nitrogen_data.py \\
        --num_shards 5 --max_clips 2000 --workers 4

    # Dry-run: scan shards, print stats, no downloads
    python scripts/prepare_nitrogen_data.py --dry_run --num_shards 2

    # Resume after interruption
    python scripts/prepare_nitrogen_data.py \\
        --num_shards 5 --max_clips 2000 --resume

    # Then train:
    python scripts/train_zimage_stage2.py \\
        --data_dir data/videos/nitrogen \\
        --stage1_checkpoint checkpoints/zimage_stage1_v2/world_model_final.pt \\
        --epochs 30
"""

import argparse
import io
import json
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --- Action constants (must match train_zimage_stage2.py) --------
ACTION_IDLE     = 0
ACTION_FORWARD  = 1
ACTION_BACKWARD = 2
ACTION_LEFT     = 3
ACTION_RIGHT    = 4
ACTION_RUN      = 5
ACTION_JUMP     = 6
ACTION_INTERACT = 7

# Joystick dead zone — below this magnitude the stick is "at rest"
JOYSTICK_THRESH = 0.30

# Games with very high movement density (used for --prefer_games sorting)
PREFERRED_GAMES = {
    "celeste", "hollow_knight", "dead_cells", "hades", "hades_ii",
    "elden_ring", "sekiro__shadows_die_twice", "dark_souls_iii",
    "dark_souls__remastered", "dark_souls_ii__scholar_of_the_first_sin",
    "lies_of_p", "god_of_war", "god_of_war_ragnarök",
    "pseudoregalia", "tunic", "ori_and_the_will_of_the_wisps",
    "streets_of_rage_2", "street_fighter_6", "mortal_kombat_11",
    "rocket_league",
}

# Games to skip — not actual gameplay footage
SKIP_GAMES = {"other", "", "unknown"}


# ── Data structures ────────────────────────────────────────────────

@dataclass
class Candidate:
    video_id: str
    chunk_id: str
    game: str
    url: str
    start_time: float
    end_time: float
    resolution: list          # [H, W] original
    idle_fraction: float
    bbox_game_area: Optional[dict]   # {xtl,ytl,xbr,ybr} in [0,1], or None
    shard_idx: int
    parquet_bytes: bytes      # raw bytes of actions_raw.parquet


# ── Action conversion ──────────────────────────────────────────────

def _parse_joystick(val) -> tuple[float, float]:
    """Parse j_left/j_right value from parquet into (x, y) floats."""
    if val is None:
        return 0.0, 0.0
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return float(val[0]), float(val[1])
    # Some parquet encodings store as numpy array
    try:
        arr = list(val)
        return float(arr[0]), float(arr[1])
    except Exception:
        return 0.0, 0.0


def row_to_action(row) -> int:
    """Convert one row of NitroGen gamepad state to a discrete action.

    Joystick convention (NitroGen standard):
      j_left = [x, y], x in [-1, 1] (left=-1, right=+1)
              y in [-1, 1] (-1 = pushed up = FORWARD, +1 = down = BACKWARD)

    Priority order (higher = earlier check):
      1. Jump (south = A/X button)
      2. Interact (east = B/○ button — covers dodge/roll)
      3. Run (left_trigger held + joystick active in most 3D action games)
      4. Joystick — dominant axis: if |y| >= |x|, forward/backward; else left/right
      5. D-pad (covers 2D platformers like Celeste, Dead Cells)
      6. Idle
    """
    jx, jy = _parse_joystick(row.get("j_left"))
    speed = (jx ** 2 + jy ** 2) ** 0.5

    if row.get("south", False):
        return ACTION_JUMP

    if row.get("east", False):
        return ACTION_INTERACT

    if row.get("left_trigger", False) and speed > JOYSTICK_THRESH:
        return ACTION_RUN

    if speed > JOYSTICK_THRESH:
        if abs(jy) >= abs(jx):
            return ACTION_FORWARD if jy < 0 else ACTION_BACKWARD
        else:
            return ACTION_LEFT if jx < 0 else ACTION_RIGHT

    if row.get("dpad_up", False):    return ACTION_FORWARD
    if row.get("dpad_down", False):  return ACTION_BACKWARD
    if row.get("dpad_left", False):  return ACTION_LEFT
    if row.get("dpad_right", False): return ACTION_RIGHT

    return ACTION_IDLE


def parquet_to_actions(parquet_bytes: bytes, target_fps: int = 30) -> list[int]:
    """Convert parquet action table to List[int] at target_fps.

    NitroGen videos are recorded at ~30fps (chunk_size=600 frames for 20s).
    If source appears to be 60fps, we subsample by 2.
    """
    import pandas as pd
    df = pd.read_parquet(io.BytesIO(parquet_bytes))
    total_frames = len(df)
    duration_s = 20.0  # fixed by NitroGen chunking
    source_fps = round(total_frames / duration_s)  # typically 30 or 60

    stride = max(1, round(source_fps / target_fps))
    rows = df.iloc[::stride].to_dict("records")
    return [row_to_action(r) for r in rows]


def compute_idle_fraction(parquet_bytes: bytes) -> float:
    """Fast idle-fraction estimate without full conversion."""
    import pandas as pd
    df = pd.read_parquet(io.BytesIO(parquet_bytes))

    btn_cols = [c for c in df.columns if c not in ("j_left", "j_right")]
    any_button = df[btn_cols].any(axis=1)

    jx = df["j_left"].apply(lambda v: _parse_joystick(v)[0])
    jy = df["j_left"].apply(lambda v: _parse_joystick(v)[1])
    joystick_active = (jx.abs() > JOYSTICK_THRESH) | (jy.abs() > JOYSTICK_THRESH)

    active = any_button | joystick_active
    return float((~active).mean())


# ── Shard scanning ────────────────────────────────────────────────

def scan_shard(
    shard_idx: int,
    tmp_dir: Path,
    idle_threshold: float = 0.50,
    max_chunks_per_video: int = 3,
) -> list[Candidate]:
    """Download one shard tar.gz, scan it, return movement-rich candidates.

    Uses a single streaming pass through the tar.gz (never calls getmembers()).
    Files within each chunk directory are buffered in memory, processed once
    all three files (metadata.json + parquet) for that chunk are seen, then
    discarded. Peak memory: ~200 MB for buffered files of one shard.

    The shard tar.gz is deleted after scanning to reclaim disk space.
    """
    from huggingface_hub import hf_hub_download

    shard_name = f"SHARD_{shard_idx:04d}.tar.gz"
    local_path = tmp_dir / shard_name

    if not local_path.exists():
        logging.info(f"Downloading {shard_name} (~1.7 GB)...")
        hf_hub_download(
            repo_id="nvidia/NitroGen",
            filename=f"actions/{shard_name}",
            repo_type="dataset",
            local_dir=str(tmp_dir),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may nest under 'actions/' subdir
        nested = tmp_dir / "actions" / shard_name
        if nested.exists() and not local_path.exists():
            nested.rename(local_path)

    logging.info(f"Scanning {shard_name} (single-pass streaming)...")

    # Buffer: chunk_dir → {"meta": dict, "parquet": bytes}
    # We flush a chunk once we have both meta and at least one parquet.
    buffer: dict[str, dict] = {}
    by_video: dict[str, list[Candidate]] = {}
    n_chunks_seen = 0

    def _flush_chunk(chunk_dir: str, data: dict):
        """Process one complete chunk (meta + parquet) into a Candidate."""
        meta = data.get("meta")
        parquet_bytes = data.get("parquet")
        if not meta or not parquet_bytes:
            return

        game = meta.get("game", "")
        if game in SKIP_GAMES:
            return

        orig = meta.get("original_video", {})
        video_id = orig.get("video_id", "")
        chunk_id = str(meta.get("chunk_id", "0000"))
        url = orig.get("url", "")
        start_time = float(orig.get("start_time", 0.0))
        end_time = float(orig.get("end_time", 20.0))
        resolution = orig.get("resolution", [1080, 1920])

        if not url or not video_id:
            return

        idle_frac = compute_idle_fraction(parquet_bytes)
        if idle_frac >= idle_threshold:
            return

        candidate = Candidate(
            video_id=video_id,
            chunk_id=chunk_id,
            game=game,
            url=url,
            start_time=start_time,
            end_time=end_time,
            resolution=resolution,
            idle_fraction=idle_frac,
            bbox_game_area=meta.get("bbox_game_area"),
            shard_idx=shard_idx,
            parquet_bytes=parquet_bytes,
        )
        by_video.setdefault(video_id, []).append(candidate)

    with tarfile.open(str(local_path), "r:gz") as tf:
        for member in tf:           # ← streaming: reads one header at a time
            if not member.isfile():
                continue

            name = member.name
            parts = Path(name).parts
            if len(parts) < 2:
                continue

            fname = parts[-1]
            chunk_dir = "/".join(parts[:-1])

            # Only care about metadata.json and action parquets
            if fname not in (
                "metadata.json",
                "actions_raw.parquet",
                "actions_processed.parquet",
            ):
                continue

            try:
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                raw = fobj.read()
            except Exception:
                continue

            entry = buffer.setdefault(chunk_dir, {})

            if fname == "metadata.json":
                try:
                    entry["meta"] = json.loads(raw)
                except Exception:
                    pass
            elif fname == "actions_processed.parquet":
                # Prefer processed over raw
                entry["parquet"] = raw
                entry["has_processed"] = True
            elif fname == "actions_raw.parquet" and "parquet" not in entry:
                entry["parquet"] = raw

            # Flush as soon as we have both meta + parquet
            if "meta" in entry and "parquet" in entry:
                _flush_chunk(chunk_dir, entry)
                del buffer[chunk_dir]
                n_chunks_seen += 1

                if n_chunks_seen % 5000 == 0:
                    logging.info(
                        f"  ... {n_chunks_seen} chunks scanned, "
                        f"{sum(len(v) for v in by_video.values())} candidates so far"
                    )

    # Flush anything remaining (chunks where files weren't adjacent in archive)
    for chunk_dir, data in buffer.items():
        _flush_chunk(chunk_dir, data)
        n_chunks_seen += 1

    # Delete shard to free disk
    try:
        local_path.unlink()
        logging.info(f"  Deleted {shard_name}")
    except Exception:
        pass

    # Keep top-N chunks per video (lowest idle fraction first)
    selected: list[Candidate] = []
    for chunks in by_video.values():
        chunks.sort(key=lambda c: c.idle_fraction)
        selected.extend(chunks[:max_chunks_per_video])

    logging.info(
        f"  Shard {shard_idx:04d}: {n_chunks_seen} chunks scanned, "
        f"{len(selected)} candidates from {len(by_video)} videos"
    )
    return selected


# ── Video download ────────────────────────────────────────────────

def _ensure_ytdlp():
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        logging.info("Installing yt-dlp...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "yt-dlp", "-q"],
            check=True,
        )


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _count_frames(video_path: Path) -> int:
    """Count video frames via ffprobe."""
    ffmpeg = _ffmpeg_exe()
    ffprobe = str(Path(ffmpeg).parent / "ffprobe")
    if not Path(ffprobe).exists():
        ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error",
             "-select_streams", "v:0",
             "-count_packets",
             "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        return int(r.stdout.strip())
    except Exception:
        return 0


def _build_vf(bbox_game_area: Optional[dict], resolution: list) -> str:
    """Build ffmpeg -vf filter string for crop + scale + fps."""
    filters = []

    if bbox_game_area:
        H, W = resolution[0], resolution[1]
        xtl = bbox_game_area.get("xtl", 0.0)
        ytl = bbox_game_area.get("ytl", 0.0)
        xbr = bbox_game_area.get("xbr", 1.0)
        ybr = bbox_game_area.get("ybr", 1.0)
        cw = int((xbr - xtl) * W) & ~1  # force even
        ch = int((ybr - ytl) * H) & ~1
        cx = int(xtl * W)
        cy = int(ytl * H)
        filters.append(f"crop={cw}:{ch}:{cx}:{cy}")

    # Scale to max 480p, preserve aspect ratio, force even dimensions
    filters.append(
        "scale='min(854,iw)':'min(480,ih)':force_original_aspect_ratio=decrease"
    )
    filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    filters.append("fps=30")

    return ",".join(filters)


def download_clip(
    c: Candidate,
    output_dir: Path,
    max_retries: int = 3,
    cookies_file: Optional[str] = None,
) -> Optional[Path]:
    """Download and process one 20-second clip. Returns mp4 path or None.

    YouTube bot-detection workaround: tries iOS/Android clients first (no
    PO token required), then falls back to web client with cookies if provided.

    Twitch VODs older than ~60 days are permanently deleted; those always fail.
    """
    import yt_dlp

    clip_name = f"{c.video_id}_chunk_{c.chunk_id}"
    out_mp4 = output_dir / f"{clip_name}.mp4"

    if out_mp4.exists():
        return out_mp4  # Already downloaded (resume mode)

    duration = c.end_time - c.start_time

    # Client order: ios/android skip the PO token check that blocks cloud IPs.
    # If cookies_file is provided, try web client last (uses cookies for auth).
    client_configs = [
        {"extractor_args": {"youtube": {"player_client": ["ios"]}}},
        {"extractor_args": {"youtube": {"player_client": ["android"]}}},
        {"extractor_args": {"youtube": {"player_client": ["android_embedded"]}}},
    ]
    if cookies_file:
        client_configs.append({"cookiefile": cookies_file})  # web client + cookies

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_tmpl = str(Path(tmpdir) / "raw.%(ext)s")

        for client_cfg in client_configs:
            ydl_opts = {
                "format": (
                    "bestvideo[height<=480][ext=mp4]"
                    "/bestvideo[height<=480]"
                    "/best[height<=480]"
                ),
                "outtmpl": raw_tmpl,
                "quiet": True,
                "no_warnings": True,
                "download_ranges": yt_dlp.utils.download_range_func(
                    [], [[c.start_time, c.end_time]]
                ),
                "socket_timeout": 30,
                "retries": max_retries,
                "fragment_retries": max_retries,
                **client_cfg,
            }

            for attempt in range(max_retries):
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([c.url])

                    raw_files = list(Path(tmpdir).glob("raw.*"))
                    if raw_files:
                        raw_file = raw_files[0]

                        # Re-encode: crop + scale + 30fps
                        vf = _build_vf(c.bbox_game_area, c.resolution)
                        ffmpeg_cmd = [
                            _ffmpeg_exe(), "-y",
                            "-i", str(raw_file),
                            "-vf", vf,
                            "-t", str(duration),
                            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                            "-an",
                            "-movflags", "+faststart",
                            str(out_mp4),
                        ]
                        result = subprocess.run(
                            ffmpeg_cmd, capture_output=True, timeout=120,
                        )
                        if result.returncode == 0 and out_mp4.exists():
                            return out_mp4
                        if out_mp4.exists():
                            out_mp4.unlink()
                    break  # yt-dlp succeeded but no file — try next client

                except Exception as e:
                    err_str = str(e)
                    # Permanent failures: deleted video or unsupported site
                    if any(x in err_str for x in [
                        "does not exist", "unavailable", "private video",
                        "has been removed", "no longer available",
                    ]):
                        return None  # Skip permanently
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    # else: try next client config

            if out_mp4.exists():
                return out_mp4

    logging.debug(f"All clients failed for {clip_name}")
    return None


def write_actions_json(
    c: Candidate,
    video_path: Path,
    output_dir: Path,
    target_fps: int = 30,
) -> Path:
    """Convert parquet actions and write _actions.json aligned to video frames."""
    clip_name = f"{c.video_id}_chunk_{c.chunk_id}"
    action_path = output_dir / f"{clip_name}_actions.json"

    if action_path.exists():
        return action_path

    actions = parquet_to_actions(c.parquet_bytes, target_fps=target_fps)

    # Align to actual frame count (handles edge cases in video encoding)
    actual_frames = _count_frames(video_path)
    if actual_frames > 0:
        if len(actions) > actual_frames:
            actions = actions[:actual_frames]
        elif len(actions) < actual_frames:
            actions += [ACTION_IDLE] * (actual_frames - len(actions))

    with open(action_path, "w") as f:
        json.dump(actions, f)

    return action_path


# ── Worker ────────────────────────────────────────────────────────

def process_candidate(
    c: Candidate,
    output_dir: Path,
    cookies_file: Optional[str] = None,
) -> dict:
    """Full pipeline for one clip: download + action JSON. Returns status."""
    clip_name = f"{c.video_id}_chunk_{c.chunk_id}"
    mp4_path = output_dir / f"{clip_name}.mp4"
    json_path = output_dir / f"{clip_name}_actions.json"

    result = {
        "clip": clip_name,
        "game": c.game,
        "idle_frac": round(c.idle_fraction, 3),
        "success": False,
        "error": None,
        "skipped": False,
    }

    # Resume: both files exist
    if mp4_path.exists() and json_path.exists():
        result["success"] = True
        result["skipped"] = True
        return result

    # Download video
    video_path = download_clip(c, output_dir, cookies_file=cookies_file)
    if video_path is None:
        result["error"] = "download_failed"
        return result

    # Generate action labels
    try:
        write_actions_json(c, video_path, output_dir)
        result["success"] = True
    except Exception as e:
        result["error"] = f"actions_failed: {e}"
        # Keep the video — can regenerate actions later

    return result


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare NitroGen data for Stage 2 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", default="data/videos/nitrogen",
                        help="Output directory for mp4 + _actions.json pairs")
    parser.add_argument("--tmp_dir", default="/tmp/nitrogen_shards",
                        help="Temp dir for shard tar.gz (auto-deleted after scan)")
    parser.add_argument("--num_shards", type=int, default=5,
                        help="Shards to scan (1–100). Spread across [0..99] for diversity")
    parser.add_argument("--shard_offset", type=int, default=0,
                        help="Start shard index for distributed runs (e.g., 0, 10, 20)")
    parser.add_argument("--max_clips", type=int, default=2000,
                        help="Total clips to download (stop early if reached)")
    parser.add_argument("--max_per_video", type=int, default=3,
                        help="Max 20s chunks to keep per YouTube video")
    parser.add_argument("--idle_threshold", type=float, default=0.50,
                        help="Skip clips where idle > this fraction")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers")
    parser.add_argument("--target_fps", type=int, default=30,
                        help="Output video FPS (actions aligned to this)")
    parser.add_argument("--prefer_games", action="store_true",
                        help="Sort candidates: preferred movement-heavy games first")
    parser.add_argument("--dry_run", action="store_true",
                        help="Scan shards, print stats, skip downloads")
    parser.add_argument("--resume", action="store_true",
                        help="Skip clips that already have both mp4 and _actions.json")
    parser.add_argument("--cookies", type=str, default=None,
                        help="Path to cookies.txt for YouTube auth (Netscape format). "
                             "Needed on cloud servers if iOS/Android clients are blocked. "
                             "Export from browser via yt-dlp's --cookies-from-browser or "
                             "a browser extension like 'Get cookies.txt LOCALLY'.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.dry_run:
        _ensure_ytdlp()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Spread shard indices evenly across [0..99] for game diversity
    if args.num_shards <= 50:
        step = max(1, 100 // args.num_shards)
        shard_indices = [
            (args.shard_offset + i * step) % 100
            for i in range(args.num_shards)
        ]
    else:
        shard_indices = [
            (args.shard_offset + i) % 100
            for i in range(args.num_shards)
        ]

    print(f"\n=== NitroGen Data Preparation ===")
    print(f"Shards to scan:  {shard_indices}")
    print(f"Output:          {output_dir}")
    print(f"Max clips:       {args.max_clips}")
    print(f"Idle threshold:  {args.idle_threshold:.0%}")
    print(f"Workers:         {args.workers}")
    print()

    fail_log_path = output_dir / "download_failures.jsonl"
    success_total = 0
    skip_total = 0

    for shard_idx in shard_indices:
        if success_total >= args.max_clips:
            print(f"Reached --max_clips={args.max_clips}, stopping.")
            break

        print(f"\n── Shard {shard_idx:04d} ─────────────────────────────")

        # Phase 1: Scan shard
        try:
            candidates = scan_shard(
                shard_idx=shard_idx,
                tmp_dir=tmp_dir,
                idle_threshold=args.idle_threshold,
                max_chunks_per_video=args.max_per_video,
            )
        except Exception as e:
            logging.error(f"Failed to scan shard {shard_idx}: {e}")
            continue

        if not candidates:
            print(f"  No candidates found in shard {shard_idx}")
            continue

        # Sort: preferred games first, then by idle fraction (most active first)
        if args.prefer_games:
            candidates.sort(key=lambda c: (
                0 if c.game in PREFERRED_GAMES else 1,
                c.idle_fraction,
            ))
        else:
            candidates.sort(key=lambda c: c.idle_fraction)

        # Cap at remaining budget
        remaining = args.max_clips - success_total
        candidates = candidates[:remaining]

        # Print game distribution
        from collections import Counter
        game_counts = Counter(c.game for c in candidates)
        top_games = game_counts.most_common(5)
        print(f"  {len(candidates)} candidates from shard {shard_idx}")
        print(f"  Top games: " + ", ".join(f"{g}({n})" for g, n in top_games))

        if args.dry_run:
            print("  [DRY RUN] Skipping downloads.")
            for c in candidates[:5]:
                idle_pct = f"{c.idle_fraction:.0%}"
                print(f"    {c.game}: chunk {c.chunk_id} idle={idle_pct}")
            continue

        # Phase 2+3: Download + action JSON (parallel)
        print(f"  Downloading with {args.workers} workers...")
        shard_success = 0
        shard_fail = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_candidate, c, output_dir, args.cookies
                ): c
                for c in candidates
            }

            try:
                from tqdm import tqdm as _tqdm
                it = _tqdm(as_completed(futures), total=len(futures),
                           desc=f"Shard {shard_idx:04d}")
            except ImportError:
                it = as_completed(futures)

            for fut in it:
                r = fut.result()
                if r["success"]:
                    shard_success += 1
                    if r.get("skipped"):
                        skip_total += 1
                else:
                    shard_fail += 1
                    with open(fail_log_path, "a") as fl:
                        fl.write(json.dumps(r) + "\n")

        success_total += shard_success
        print(f"  Shard {shard_idx:04d}: {shard_success} ok, {shard_fail} failed")

    # Summary
    print(f"\n{'='*50}")
    print(f"DONE: {success_total} clips ({skip_total} resumed) in {output_dir}")
    if fail_log_path.exists():
        with open(fail_log_path) as f:
            n_fail = sum(1 for _ in f)
        print(f"Failures logged: {fail_log_path} ({n_fail} entries)")

    # Print action distribution from a sample
    _print_action_stats(output_dir)

    print(f"\nTo start Stage 2 training:")
    print(f"  python scripts/train_zimage_stage2.py \\")
    print(f"      --data_dir {output_dir} \\")
    print(f"      --stage1_checkpoint checkpoints/zimage_stage1_v2/world_model_final.pt \\")
    print(f"      --epochs 30")


def _print_action_stats(output_dir: Path, sample_n: int = 50):
    """Print action distribution from a sample of downloaded clips."""
    import random
    from collections import Counter

    json_files = list(output_dir.glob("*_actions.json"))
    if not json_files:
        return

    sample = random.sample(json_files, min(sample_n, len(json_files)))
    counts = Counter()
    for p in sample:
        try:
            actions = json.loads(p.read_text())
            counts.update(actions)
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


if __name__ == "__main__":
    main()
