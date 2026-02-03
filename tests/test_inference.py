"""Tests for inference components."""

import pytest
import torch

from inference import InputHandler, KeyboardState
from models import ActionSpace


class TestKeyboardState:
    """Tests for KeyboardState."""

    def test_initial_state_empty(self):
        """Test initial state has no pressed keys."""
        state = KeyboardState()
        assert len(state.pressed_keys) == 0

    def test_is_pressed(self):
        """Test is_pressed method."""
        state = KeyboardState()
        state.pressed_keys.add("w")

        assert state.is_pressed("w")
        assert state.is_pressed("W")  # Should be case-insensitive
        assert not state.is_pressed("s")

    def test_any_pressed(self):
        """Test any_pressed method."""
        state = KeyboardState()
        state.pressed_keys.add("w")

        assert state.any_pressed(["w", "a", "s", "d"])
        assert not state.any_pressed(["a", "s", "d"])

    def test_all_pressed(self):
        """Test all_pressed method."""
        state = KeyboardState()
        state.pressed_keys.add("w")
        state.pressed_keys.add("a")

        assert state.all_pressed(["w", "a"])
        assert not state.all_pressed(["w", "a", "s"])


class TestInputHandler:
    """Tests for InputHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler in manual mode."""
        return InputHandler(mode="manual")

    def test_get_action_idle(self, handler):
        """Test idle action when no keys pressed."""
        action = handler.get_action()
        assert action == ActionSpace.IDLE

    def test_get_action_forward(self, handler):
        """Test forward action."""
        handler.set_key("w", True)
        action = handler.get_action()
        assert action == ActionSpace.MOVE_FORWARD

    def test_get_action_diagonal(self, handler):
        """Test diagonal movement."""
        handler.set_key("w", True)
        handler.set_key("d", True)
        action = handler.get_action()
        assert action == ActionSpace.MOVE_FORWARD_RIGHT

    def test_get_action_jump(self, handler):
        """Test jump action."""
        handler.set_key("space", True)
        action = handler.get_action()
        assert action == ActionSpace.JUMP

    def test_set_keys(self, handler):
        """Test setting multiple keys at once."""
        handler.set_keys({"w", "a"})
        action = handler.get_action()
        assert action == ActionSpace.MOVE_FORWARD_LEFT

        # Change keys
        handler.set_keys({"s"})
        action = handler.get_action()
        assert action == ActionSpace.MOVE_BACKWARD

    def test_get_movement_vector(self, handler):
        """Test movement vector calculation."""
        # Forward
        handler.set_key("w", True)
        x, y = handler.get_movement_vector()
        assert x == 0
        assert y == 1

        # Diagonal
        handler.set_key("d", True)
        x, y = handler.get_movement_vector()
        assert abs(x - 0.707) < 0.01  # ~1/sqrt(2)
        assert abs(y - 0.707) < 0.01

    def test_reset(self, handler):
        """Test handler reset."""
        handler.set_key("w", True)
        handler.reset()

        action = handler.get_action()
        assert action == ActionSpace.IDLE


class TestActionRecorder:
    """Tests for ActionRecorder."""

    def test_record_sequence(self):
        """Test recording action sequence."""
        from inference.input_handler import ActionRecorder

        recorder = ActionRecorder()
        recorder.start()

        recorder.record(ActionSpace.MOVE_FORWARD, 0)
        recorder.record(ActionSpace.MOVE_FORWARD, 1)
        recorder.record(ActionSpace.JUMP, 2)

        recorder.stop()

        sequence = recorder.get_sequence()
        assert len(sequence) == 3
        assert sequence[0][0] == ActionSpace.MOVE_FORWARD.value
        assert sequence[2][0] == ActionSpace.JUMP.value

    def test_get_actions_only(self):
        """Test getting just action indices."""
        from inference.input_handler import ActionRecorder

        recorder = ActionRecorder()
        recorder.start()

        recorder.record(0, 0)
        recorder.record(13, 1)  # jump
        recorder.record(16, 2)  # attack

        recorder.stop()

        actions = recorder.get_actions_only()
        assert actions == [0, 13, 16]
