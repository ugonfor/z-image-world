"""
Action-Video Dataset for Training

Handles loading and preprocessing of video-action pairs for training
the action-conditioned world model.

Supports:
- NVIDIA NitroGen format
- Custom gameplay recordings
- Frame-action alignment
"""

import json
import os
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


class ActionVideoDataset(Dataset):
    """Dataset for video clips with action annotations.

    Expected data format:
    data_root/
        video_001/
            frames/
                0000.png
                0001.png
                ...
            actions.json  # List of action indices per frame
        video_002/
            ...

    Or for NVIDIA NitroGen format:
        video_001.mp4
        video_001_actions.json
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 8,
        frame_skip: int = 1,
        resolution: tuple[int, int] = (480, 640),
        transform: Optional[Callable] = None,
        cache_frames: bool = False,
        max_samples: Optional[int] = None,
    ):
        """Initialize dataset.

        Args:
            data_root: Root directory containing video data
            num_frames: Number of frames per sample
            frame_skip: Skip every N frames (temporal stride)
            resolution: Output resolution (height, width)
            transform: Optional transform function
            cache_frames: Whether to cache frames in memory
            max_samples: Maximum number of samples (for debugging)
        """
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.resolution = resolution
        self.transform = transform
        self.cache_frames = cache_frames

        # Find all video samples
        self.samples = self._find_samples()

        if max_samples:
            self.samples = self.samples[:max_samples]

        # Frame cache
        self._cache: dict[str, torch.Tensor] = {} if cache_frames else None

    def _find_samples(self) -> list[dict]:
        """Find all valid video-action samples."""
        samples = []

        # Look for video directories
        for item in self.data_root.iterdir():
            if item.is_dir():
                # Directory format
                frames_dir = item / "frames"
                actions_file = item / "actions.json"

                if frames_dir.exists() and actions_file.exists():
                    frame_files = sorted(frames_dir.glob("*.png")) + \
                                  sorted(frames_dir.glob("*.jpg"))

                    if len(frame_files) >= self.num_frames * self.frame_skip:
                        samples.append({
                            "type": "directory",
                            "frames_dir": frames_dir,
                            "actions_file": actions_file,
                            "frame_files": frame_files,
                            "name": item.name,
                        })

            elif item.suffix in [".mp4", ".avi", ".mov"]:
                # Video file format (NitroGen style)
                actions_file = item.with_name(f"{item.stem}_actions.json")
                if actions_file.exists():
                    samples.append({
                        "type": "video",
                        "video_file": item,
                        "actions_file": actions_file,
                        "name": item.stem,
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a video-action sample.

        Returns:
            Dictionary with:
            - "frames": (num_frames, 3, H, W) tensor in [0, 1]
            - "actions": (num_frames,) tensor of action indices
            - "name": Sample name
        """
        sample = self.samples[idx]

        if sample["type"] == "directory":
            frames, actions = self._load_directory_sample(sample)
        else:
            frames, actions = self._load_video_sample(sample)

        # Apply transform
        if self.transform:
            frames = self.transform(frames)

        return {
            "frames": frames,
            "actions": actions,
            "name": sample["name"],
        }

    def _load_directory_sample(
        self,
        sample: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load sample from directory format."""
        # Load actions
        with open(sample["actions_file"]) as f:
            all_actions = json.load(f)

        # Random start point
        total_frames = len(sample["frame_files"])
        max_start = total_frames - self.num_frames * self.frame_skip
        start_idx = np.random.randint(0, max_start + 1)

        # Load frames
        frames = []
        actions = []

        for i in range(self.num_frames):
            frame_idx = start_idx + i * self.frame_skip

            # Load frame
            frame_path = str(sample["frame_files"][frame_idx])

            if self._cache and frame_path in self._cache:
                frame = self._cache[frame_path]
            else:
                frame = self._load_image(frame_path)
                if self._cache:
                    self._cache[frame_path] = frame

            frames.append(frame)

            # Get action
            if isinstance(all_actions, list):
                action = all_actions[frame_idx] if frame_idx < len(all_actions) else 8
            else:
                action = 8  # Idle

            actions.append(action)

        frames = torch.stack(frames)
        actions = torch.tensor(actions, dtype=torch.long)

        return frames, actions

    def _load_video_sample(
        self,
        sample: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load sample from video file."""
        if not IMAGEIO_AVAILABLE:
            raise ImportError("imageio required for video loading")

        # Load actions
        with open(sample["actions_file"]) as f:
            all_actions = json.load(f)

        # Open video
        reader = imageio.get_reader(str(sample["video_file"]))
        total_frames = reader.count_frames()

        # Random start point
        max_start = total_frames - self.num_frames * self.frame_skip
        start_idx = np.random.randint(0, max_start + 1)

        # Load frames
        frames = []
        actions = []

        for i in range(self.num_frames):
            frame_idx = start_idx + i * self.frame_skip

            # Load frame
            frame = reader.get_data(frame_idx)
            frame = self._process_frame(frame)
            frames.append(frame)

            # Get action
            action = all_actions[frame_idx] if frame_idx < len(all_actions) else 8
            actions.append(action)

        reader.close()

        frames = torch.stack(frames)
        actions = torch.tensor(actions, dtype=torch.long)

        return frames, actions

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess a single image."""
        if CV2_AVAILABLE:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif IMAGEIO_AVAILABLE:
            img = imageio.imread(path)
        else:
            raise ImportError("Either cv2 or imageio required")

        return self._process_frame(img)

    def _process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Process frame to tensor."""
        # Resize
        h, w = self.resolution
        if CV2_AVAILABLE:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Simple numpy resize (not as good)
            import skimage.transform as skt
            frame = skt.resize(frame, (h, w), preserve_range=True)

        # Convert to tensor [0, 1]
        frame = torch.from_numpy(frame).float() / 255.0

        # HWC -> CHW
        frame = frame.permute(2, 0, 1)

        return frame


class ActionVideoCollator:
    """Collator for batching video-action samples."""

    def __call__(
        self,
        samples: list[dict],
    ) -> dict[str, torch.Tensor]:
        """Collate samples into a batch.

        Args:
            samples: List of sample dictionaries

        Returns:
            Batched dictionary
        """
        frames = torch.stack([s["frames"] for s in samples])
        actions = torch.stack([s["actions"] for s in samples])

        return {
            "frames": frames,
            "actions": actions,
        }


class VideoOnlyDataset(Dataset):
    """Dataset for video clips without action annotations.

    Used for Stage 1 causal adaptation training.
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 8,
        frame_skip: int = 1,
        resolution: tuple[int, int] = (480, 640),
        transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.resolution = resolution
        self.transform = transform

        # Find all video files
        self.video_files = list(self.data_root.glob("**/*.mp4")) + \
                          list(self.data_root.glob("**/*.avi")) + \
                          list(self.data_root.glob("**/*.mov"))

    def __len__(self) -> int:
        return len(self.video_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a video sample."""
        if not IMAGEIO_AVAILABLE:
            raise ImportError("imageio required for video loading")

        video_path = self.video_files[idx]
        reader = imageio.get_reader(str(video_path))

        try:
            total_frames = reader.count_frames()
        except Exception:
            # Fallback: try to iterate
            total_frames = sum(1 for _ in reader)
            reader = imageio.get_reader(str(video_path))

        # Random start point
        required_frames = self.num_frames * self.frame_skip
        if total_frames < required_frames:
            # Repeat frames if video too short
            start_idx = 0
        else:
            max_start = total_frames - required_frames
            start_idx = np.random.randint(0, max_start + 1)

        # Load frames
        frames = []
        for i in range(self.num_frames):
            frame_idx = min(start_idx + i * self.frame_skip, total_frames - 1)
            frame = reader.get_data(frame_idx)
            frame = self._process_frame(frame)
            frames.append(frame)

        reader.close()

        frames = torch.stack(frames)

        if self.transform:
            frames = self.transform(frames)

        return {"frames": frames}

    def _process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Process frame to tensor."""
        h, w = self.resolution
        if CV2_AVAILABLE:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)
        return frame


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for training.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle
        pin_memory: Whether to pin memory

    Returns:
        DataLoader instance
    """
    collator = ActionVideoCollator() if hasattr(dataset, "actions") else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=True,
    )
