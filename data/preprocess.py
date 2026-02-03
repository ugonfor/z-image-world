"""
Preprocessing Utilities for Video-Action Data

Handles:
- Frame-action temporal alignment
- Video preprocessing and extraction
- Action logging and synchronization
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field
import threading
from collections import deque

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class ActionLog:
    """Log of actions with timestamps."""

    actions: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    start_time: float = 0.0

    def add(self, action: int, timestamp: Optional[float] = None):
        """Add an action to the log."""
        if timestamp is None:
            timestamp = time.time() - self.start_time
        self.actions.append(action)
        self.timestamps.append(timestamp)

    def save(self, path: str):
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump({
                "actions": self.actions,
                "timestamps": self.timestamps,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ActionLog":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        log = cls()
        log.actions = data["actions"]
        log.timestamps = data["timestamps"]
        return log


class FrameActionAligner:
    """Aligns action timestamps with video frames.

    Handles the common case where actions are logged at irregular
    intervals but we need per-frame action labels.
    """

    def __init__(
        self,
        fps: float = 30.0,
        default_action: int = 8,  # Idle
    ):
        """Initialize aligner.

        Args:
            fps: Video frame rate
            default_action: Action to use when no action is recorded
        """
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.default_action = default_action

    def align(
        self,
        action_log: ActionLog,
        num_frames: int,
        video_start_time: float = 0.0,
    ) -> list[int]:
        """Align actions to frames.

        Args:
            action_log: Log of actions with timestamps
            num_frames: Number of video frames
            video_start_time: Start time of video relative to action log

        Returns:
            List of action indices, one per frame
        """
        if len(action_log.actions) == 0:
            return [self.default_action] * num_frames

        frame_actions = []
        action_idx = 0

        for frame in range(num_frames):
            frame_time = video_start_time + frame * self.frame_duration

            # Find the action active at this frame time
            # (last action that started before or at this time)
            while (action_idx < len(action_log.timestamps) - 1 and
                   action_log.timestamps[action_idx + 1] <= frame_time):
                action_idx += 1

            if action_idx < len(action_log.actions):
                action = action_log.actions[action_idx]
            else:
                action = self.default_action

            frame_actions.append(action)

        return frame_actions

    def align_with_interpolation(
        self,
        action_log: ActionLog,
        num_frames: int,
        video_start_time: float = 0.0,
    ) -> list[int]:
        """Align actions with temporal interpolation for smooth transitions.

        For continuous actions (movement), interpolates between action changes.
        For discrete actions (jump, attack), uses the exact timing.
        """
        # Continuous action groups (can be interpolated)
        continuous_actions = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

        frame_actions = []
        action_idx = 0

        for frame in range(num_frames):
            frame_time = video_start_time + frame * self.frame_duration

            # Find surrounding actions
            while (action_idx < len(action_log.timestamps) - 1 and
                   action_log.timestamps[action_idx + 1] <= frame_time):
                action_idx += 1

            current_action = action_log.actions[action_idx] if action_idx < len(action_log.actions) else self.default_action

            # For discrete actions, always use exact value
            if current_action not in continuous_actions:
                frame_actions.append(current_action)
                continue

            # For continuous actions, check if we should interpolate
            if action_idx < len(action_log.actions) - 1:
                next_action = action_log.actions[action_idx + 1]
                next_time = action_log.timestamps[action_idx + 1]
                curr_time = action_log.timestamps[action_idx]

                # Calculate blend factor
                if next_time > curr_time:
                    t = (frame_time - curr_time) / (next_time - curr_time)
                    t = max(0, min(1, t))

                    # Use next action if we're closer to it
                    if t > 0.5 and next_action in continuous_actions:
                        frame_actions.append(next_action)
                    else:
                        frame_actions.append(current_action)
                else:
                    frame_actions.append(current_action)
            else:
                frame_actions.append(current_action)

        return frame_actions


class VideoPreprocessor:
    """Preprocesses videos for training.

    Extracts frames, applies augmentations, and prepares
    data in the expected format.
    """

    def __init__(
        self,
        output_root: str,
        resolution: tuple[int, int] = (480, 640),
        fps: Optional[float] = None,  # None = keep original
        frame_format: str = "png",
    ):
        """Initialize preprocessor.

        Args:
            output_root: Root directory for output
            resolution: Output resolution (height, width)
            fps: Target FPS (None to keep original)
            frame_format: Image format for frames
        """
        if not CV2_AVAILABLE:
            raise ImportError("cv2 required for video preprocessing")

        self.output_root = Path(output_root)
        self.resolution = resolution
        self.target_fps = fps
        self.frame_format = frame_format

    def process_video(
        self,
        video_path: str,
        action_log: Optional[ActionLog] = None,
        output_name: Optional[str] = None,
    ) -> dict:
        """Process a single video.

        Args:
            video_path: Path to video file
            action_log: Optional action log to align
            output_name: Output directory name (default: video filename)

        Returns:
            Dictionary with processing info
        """
        video_path = Path(video_path)
        if output_name is None:
            output_name = video_path.stem

        # Create output directory
        output_dir = self.output_root / output_name
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame skip for target FPS
        if self.target_fps and self.target_fps < original_fps:
            frame_skip = int(original_fps / self.target_fps)
        else:
            frame_skip = 1

        output_fps = original_fps / frame_skip

        # Extract frames
        frame_idx = 0
        output_idx = 0
        h, w = self.resolution

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Resize
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

                # Save frame
                frame_path = frames_dir / f"{output_idx:06d}.{self.frame_format}"
                cv2.imwrite(str(frame_path), frame)
                output_idx += 1

            frame_idx += 1

        cap.release()

        # Align actions if provided
        if action_log is not None:
            aligner = FrameActionAligner(fps=output_fps)
            aligned_actions = aligner.align(action_log, output_idx)

            # Save actions
            actions_path = output_dir / "actions.json"
            with open(actions_path, "w") as f:
                json.dump(aligned_actions, f)

        return {
            "output_dir": str(output_dir),
            "num_frames": output_idx,
            "fps": output_fps,
            "resolution": self.resolution,
        }

    def process_directory(
        self,
        input_dir: str,
        action_logs_dir: Optional[str] = None,
    ) -> list[dict]:
        """Process all videos in a directory.

        Args:
            input_dir: Directory containing videos
            action_logs_dir: Directory containing action logs (same names as videos)

        Returns:
            List of processing info dictionaries
        """
        input_dir = Path(input_dir)
        results = []

        video_files = list(input_dir.glob("*.mp4")) + \
                     list(input_dir.glob("*.avi")) + \
                     list(input_dir.glob("*.mov"))

        for video_path in video_files:
            # Find matching action log
            action_log = None
            if action_logs_dir:
                log_path = Path(action_logs_dir) / f"{video_path.stem}_actions.json"
                if log_path.exists():
                    action_log = ActionLog.load(str(log_path))

            result = self.process_video(video_path, action_log)
            results.append(result)

        return results


class ScreenRecorder:
    """Records screen with synchronized action logging.

    For collecting custom training data from gameplay.
    """

    def __init__(
        self,
        output_dir: str,
        fps: float = 30.0,
        resolution: tuple[int, int] = (480, 640),
        region: Optional[tuple[int, int, int, int]] = None,  # x, y, w, h
    ):
        """Initialize recorder.

        Args:
            output_dir: Output directory
            fps: Recording FPS
            resolution: Output resolution
            region: Screen region to capture (None = full screen)
        """
        if not CV2_AVAILABLE:
            raise ImportError("cv2 required for screen recording")

        try:
            import mss
            self.mss = mss.mss()
        except ImportError:
            raise ImportError("mss required for screen capture")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.resolution = resolution
        self.region = region

        self.action_log = ActionLog()
        self._recording = False
        self._frames: deque = deque()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start recording."""
        self._recording = True
        self.action_log = ActionLog()
        self.action_log.start_time = time.time()
        self._frames.clear()

        self._thread = threading.Thread(target=self._capture_loop)
        self._thread.start()

    def stop(self) -> str:
        """Stop recording and save.

        Returns:
            Path to output directory
        """
        self._recording = False
        if self._thread:
            self._thread.join()

        # Save video
        video_path = self.output_dir / "recording.mp4"
        self._save_video(str(video_path))

        # Save actions
        self.action_log.save(str(self.output_dir / "actions.json"))

        return str(self.output_dir)

    def log_action(self, action: int):
        """Log an action during recording."""
        if self._recording:
            self.action_log.add(action)

    def _capture_loop(self):
        """Capture loop running in separate thread."""
        frame_interval = 1.0 / self.fps
        next_frame_time = time.time()

        if self.region:
            monitor = {
                "left": self.region[0],
                "top": self.region[1],
                "width": self.region[2],
                "height": self.region[3],
            }
        else:
            monitor = self.mss.monitors[1]  # Primary monitor

        while self._recording:
            current_time = time.time()

            if current_time >= next_frame_time:
                # Capture frame
                screenshot = self.mss.grab(monitor)
                frame = np.array(screenshot)[:, :, :3]  # Remove alpha

                # Resize
                h, w = self.resolution
                frame = cv2.resize(frame, (w, h))

                self._frames.append(frame)
                next_frame_time += frame_interval

            # Small sleep to prevent busy waiting
            time.sleep(0.001)

    def _save_video(self, path: str):
        """Save captured frames to video."""
        if not self._frames:
            return

        h, w = self.resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))

        for frame in self._frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
