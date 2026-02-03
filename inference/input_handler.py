"""
Input Handler for Keyboard Controls

Handles keyboard input for interactive world model control.
Supports:
- Real-time key detection
- Multi-key combinations
- Action mapping
- Input smoothing/debouncing
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import deque
import threading

from models.action_encoder import ActionSpace


@dataclass
class KeyboardState:
    """Current state of keyboard inputs."""

    # Currently pressed keys
    pressed_keys: set[str] = field(default_factory=set)

    # Last update timestamp
    last_update: float = 0.0

    # Key press history (for combos)
    key_history: deque = field(default_factory=lambda: deque(maxlen=10))

    def is_pressed(self, key: str) -> bool:
        """Check if a key is currently pressed."""
        return key.lower() in self.pressed_keys

    def any_pressed(self, keys: list[str]) -> bool:
        """Check if any of the keys are pressed."""
        return any(k.lower() in self.pressed_keys for k in keys)

    def all_pressed(self, keys: list[str]) -> bool:
        """Check if all of the keys are pressed."""
        return all(k.lower() in self.pressed_keys for k in keys)


class InputHandler:
    """Handles keyboard input and converts to actions.

    Can operate in different modes:
    - pygame: Uses pygame for real-time input (requires display)
    - pynput: Uses pynput for global keyboard capture
    - manual: Accepts manual key state updates

    Usage:
        handler = InputHandler(mode="pygame")
        handler.start()

        while running:
            action = handler.get_action()
            # Use action...

        handler.stop()
    """

    # Key mappings for different input backends
    KEY_ALIASES = {
        # pygame keycodes
        "pygame": {
            "K_w": "w", "K_a": "a", "K_s": "s", "K_d": "d",
            "K_UP": "up", "K_DOWN": "down", "K_LEFT": "left", "K_RIGHT": "right",
            "K_SPACE": "space", "K_LCTRL": "ctrl", "K_RCTRL": "ctrl",
            "K_e": "e", "K_f": "f", "K_c": "c",
            "K_i": "i", "K_j": "j", "K_k": "k", "K_l": "l",
        },
        # pynput key names
        "pynput": {
            "Key.space": "space", "Key.ctrl_l": "ctrl", "Key.ctrl_r": "ctrl",
            "Key.up": "up", "Key.down": "down", "Key.left": "left", "Key.right": "right",
        },
    }

    def __init__(
        self,
        mode: str = "manual",
        action_repeat_delay: float = 0.0,  # No delay for real-time
        debounce_time: float = 0.016,  # ~60 Hz
    ):
        """Initialize input handler.

        Args:
            mode: Input mode ("pygame", "pynput", or "manual")
            action_repeat_delay: Delay between repeated actions (0 = always repeat)
            debounce_time: Minimum time between key state updates
        """
        self.mode = mode
        self.action_repeat_delay = action_repeat_delay
        self.debounce_time = debounce_time

        self.state = KeyboardState()
        self._last_action_time: dict[int, float] = {}
        self._running = False
        self._input_thread: Optional[threading.Thread] = None

        # Callbacks for key events
        self._on_key_down: Optional[Callable[[str], None]] = None
        self._on_key_up: Optional[Callable[[str], None]] = None

    def start(self):
        """Start listening for input."""
        if self._running:
            return

        self._running = True

        if self.mode == "pygame":
            self._start_pygame()
        elif self.mode == "pynput":
            self._start_pynput()
        # Manual mode doesn't need starting

    def stop(self):
        """Stop listening for input."""
        self._running = False
        if self._input_thread:
            self._input_thread.join(timeout=1.0)
            self._input_thread = None

    def _start_pygame(self):
        """Start pygame input handling (requires pygame.init())."""
        try:
            import pygame
        except ImportError:
            raise ImportError("pygame required for pygame input mode")

        # Pygame input is handled in the main thread
        # No separate thread needed

    def _start_pynput(self):
        """Start pynput global keyboard listener."""
        try:
            from pynput import keyboard
        except ImportError:
            raise ImportError("pynput required for pynput input mode")

        def on_press(key):
            if not self._running:
                return False
            key_name = self._normalize_key(str(key), "pynput")
            self._key_down(key_name)

        def on_release(key):
            if not self._running:
                return False
            key_name = self._normalize_key(str(key), "pynput")
            self._key_up(key_name)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

    def _normalize_key(self, key: str, backend: str) -> str:
        """Normalize key name to standard format."""
        aliases = self.KEY_ALIASES.get(backend, {})
        # Remove quotes from pynput keys
        key = key.strip("'")
        return aliases.get(key, key.lower())

    def _key_down(self, key: str):
        """Handle key press."""
        key = key.lower()
        if key not in self.state.pressed_keys:
            self.state.pressed_keys.add(key)
            self.state.key_history.append(("down", key, time.time()))
            if self._on_key_down:
                self._on_key_down(key)

    def _key_up(self, key: str):
        """Handle key release."""
        key = key.lower()
        if key in self.state.pressed_keys:
            self.state.pressed_keys.discard(key)
            self.state.key_history.append(("up", key, time.time()))
            if self._on_key_up:
                self._on_key_up(key)

    def update_pygame(self):
        """Update state from pygame events. Call this in your game loop."""
        if self.mode != "pygame":
            return

        try:
            import pygame
        except ImportError:
            return

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                self._key_down(key_name)
            elif event.type == pygame.KEYUP:
                key_name = pygame.key.name(event.key)
                self._key_up(key_name)

        self.state.last_update = time.time()

    def set_key(self, key: str, pressed: bool):
        """Manually set key state (for manual mode or testing).

        Args:
            key: Key name
            pressed: Whether the key is pressed
        """
        if pressed:
            self._key_down(key)
        else:
            self._key_up(key)
        self.state.last_update = time.time()

    def set_keys(self, keys: set[str]):
        """Set the complete key state.

        Args:
            keys: Set of currently pressed keys
        """
        # Find keys that were released
        released = self.state.pressed_keys - keys
        for key in released:
            self._key_up(key)

        # Find keys that were pressed
        pressed = keys - self.state.pressed_keys
        for key in pressed:
            self._key_down(key)

        self.state.last_update = time.time()

    def get_action(self) -> ActionSpace:
        """Get the current action based on pressed keys.

        Returns:
            ActionSpace enum value for the current action
        """
        # Check action repeat delay
        current_time = time.time()

        # Convert key state to action
        action = ActionSpace.from_keyboard(self.state.pressed_keys)

        # Apply action repeat delay
        if self.action_repeat_delay > 0:
            last_time = self._last_action_time.get(action.value, 0)
            if current_time - last_time < self.action_repeat_delay:
                return ActionSpace.IDLE

        self._last_action_time[action.value] = current_time
        return action

    def get_action_index(self) -> int:
        """Get the current action as an integer index.

        Returns:
            Action index (0-16)
        """
        return self.get_action().value

    def on_key_down(self, callback: Callable[[str], None]):
        """Register callback for key press events."""
        self._on_key_down = callback

    def on_key_up(self, callback: Callable[[str], None]):
        """Register callback for key release events."""
        self._on_key_up = callback

    def is_action_pressed(self, action: ActionSpace) -> bool:
        """Check if a specific action's keys are pressed.

        Args:
            action: ActionSpace to check

        Returns:
            True if the action's keys are pressed
        """
        return self.get_action() == action

    def get_movement_vector(self) -> tuple[float, float]:
        """Get normalized movement vector from WASD keys.

        Returns:
            (x, y) movement vector where:
            - x: -1 (left) to 1 (right)
            - y: -1 (backward) to 1 (forward)
        """
        x, y = 0.0, 0.0

        if self.state.is_pressed("w") or self.state.is_pressed("up"):
            y += 1
        if self.state.is_pressed("s") or self.state.is_pressed("down"):
            y -= 1
        if self.state.is_pressed("a") or self.state.is_pressed("left"):
            x -= 1
        if self.state.is_pressed("d") or self.state.is_pressed("right"):
            x += 1

        # Normalize diagonal movement
        if x != 0 and y != 0:
            magnitude = (x**2 + y**2) ** 0.5
            x /= magnitude
            y /= magnitude

        return x, y

    def get_look_vector(self) -> tuple[float, float]:
        """Get normalized look/camera vector from IJKL keys.

        Returns:
            (x, y) look vector where:
            - x: -1 (look left) to 1 (look right)
            - y: -1 (look down) to 1 (look up)
        """
        x, y = 0.0, 0.0

        if self.state.is_pressed("i"):
            y += 1
        if self.state.is_pressed("k"):
            y -= 1
        if self.state.is_pressed("j"):
            x -= 1
        if self.state.is_pressed("l"):
            x += 1

        return x, y

    def reset(self):
        """Reset input state."""
        self.state.pressed_keys.clear()
        self.state.key_history.clear()
        self._last_action_time.clear()


class ActionRecorder:
    """Records action sequences for replay or training data collection.

    Usage:
        recorder = ActionRecorder()
        recorder.start()

        for frame in frames:
            action = input_handler.get_action()
            recorder.record(action, frame_index)

        recorder.stop()
        sequence = recorder.get_sequence()
    """

    def __init__(self):
        self._recording = False
        self._sequence: list[tuple[int, int, float]] = []  # (action, frame_idx, timestamp)
        self._start_time = 0.0

    def start(self):
        """Start recording."""
        self._recording = True
        self._sequence.clear()
        self._start_time = time.time()

    def stop(self):
        """Stop recording."""
        self._recording = False

    def record(self, action: int | ActionSpace, frame_index: int):
        """Record an action.

        Args:
            action: Action index or ActionSpace
            frame_index: Current frame index
        """
        if not self._recording:
            return

        if isinstance(action, ActionSpace):
            action = action.value

        self._sequence.append((
            action,
            frame_index,
            time.time() - self._start_time,
        ))

    def get_sequence(self) -> list[tuple[int, int, float]]:
        """Get the recorded sequence.

        Returns:
            List of (action, frame_index, timestamp) tuples
        """
        return self._sequence.copy()

    def get_actions_only(self) -> list[int]:
        """Get just the action indices.

        Returns:
            List of action indices
        """
        return [a[0] for a in self._sequence]

    def save(self, path: str):
        """Save recording to file."""
        import json
        with open(path, "w") as f:
            json.dump({
                "sequence": self._sequence,
                "action_names": {a.value: a.name for a in ActionSpace},
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ActionRecorder":
        """Load recording from file."""
        import json
        recorder = cls()
        with open(path) as f:
            data = json.load(f)
        recorder._sequence = [tuple(s) for s in data["sequence"]]
        return recorder
