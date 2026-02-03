"""
Interactive Demo Application for Z-Image World Model

Provides a real-time interactive environment where users can
control the generated world using keyboard inputs.

Features:
- Real-time frame generation
- Keyboard controls
- Performance overlay
- Recording capability
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass

import torch
import numpy as np

if TYPE_CHECKING:
    import pygame

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None  # type: ignore
    PYGAME_AVAILABLE = False

from inference import RealtimePipeline, PipelineConfig, InputHandler
from models import ActionSpace


@dataclass
class DemoConfig:
    """Configuration for interactive demo."""

    # Display
    window_width: int = 1280
    window_height: int = 720
    fullscreen: bool = False

    # Performance
    target_fps: float = 30.0
    show_performance: bool = True

    # Recording
    enable_recording: bool = False
    recording_dir: str = "recordings"

    # Model
    model_path: Optional[str] = None
    device: str = "cuda"


class PerformanceOverlay:
    """Displays performance metrics on screen."""

    def __init__(self, font_size: int = 20):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required for demo")

        self.font = pygame.font.Font(None, font_size)
        self.fps_history = []
        self.latency_history = []
        self.max_history = 60

    def update(self, fps: float, latency_ms: float):
        """Update metrics."""
        self.fps_history.append(fps)
        self.latency_history.append(latency_ms)

        if len(self.fps_history) > self.max_history:
            self.fps_history.pop(0)
            self.latency_history.pop(0)

    def render(self, surface: pygame.Surface):
        """Render overlay to surface."""
        if not self.fps_history:
            return

        avg_fps = sum(self.fps_history) / len(self.fps_history)
        avg_latency = sum(self.latency_history) / len(self.latency_history)

        lines = [
            f"FPS: {avg_fps:.1f}",
            f"Latency: {avg_latency:.1f}ms",
            f"Generation FPS: {1000/avg_latency:.1f}" if avg_latency > 0 else "N/A",
        ]

        y = 10
        for line in lines:
            text = self.font.render(line, True, (255, 255, 255))
            # Draw shadow
            shadow = self.font.render(line, True, (0, 0, 0))
            surface.blit(shadow, (12, y + 2))
            surface.blit(text, (10, y))
            y += 25


class ControlsOverlay:
    """Displays control instructions."""

    def __init__(self, font_size: int = 18):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required for demo")

        self.font = pygame.font.Font(None, font_size)
        self.visible = True

    def toggle(self):
        """Toggle visibility."""
        self.visible = not self.visible

    def render(self, surface: pygame.Surface):
        """Render controls to surface."""
        if not self.visible:
            return

        controls = [
            "Controls:",
            "  WASD - Move",
            "  IJKL - Look",
            "  Space - Jump",
            "  E - Interact",
            "  F - Attack",
            "  C - Crouch",
            "",
            "  H - Toggle this help",
            "  P - Toggle performance",
            "  R - Start/stop recording",
            "  ESC - Quit",
        ]

        # Semi-transparent background
        bg_width = 180
        bg_height = len(controls) * 20 + 10
        bg_surface = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 128))

        surface.blit(bg_surface, (surface.get_width() - bg_width - 10, 10))

        y = 15
        for line in controls:
            text = self.font.render(line, True, (255, 255, 255))
            surface.blit(text, (surface.get_width() - bg_width, y))
            y += 20


class InteractiveApp:
    """Main interactive demo application.

    Usage:
        app = InteractiveApp(config)
        app.set_initial_image(image)
        app.run()
    """

    def __init__(
        self,
        config: Optional[DemoConfig] = None,
        pipeline: Optional[RealtimePipeline] = None,
    ):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required for interactive demo")

        self.config = config or DemoConfig()
        self.pipeline = pipeline

        # Initialize pygame
        pygame.init()

        # Setup display
        flags = pygame.DOUBLEBUF | pygame.HWSURFACE
        if self.config.fullscreen:
            flags |= pygame.FULLSCREEN

        self.screen = pygame.display.set_mode(
            (self.config.window_width, self.config.window_height),
            flags,
        )
        pygame.display.set_caption("Z-Image World - Interactive Demo")

        # Input handler
        self.input_handler = InputHandler(mode="pygame")

        # UI overlays
        self.perf_overlay = PerformanceOverlay()
        self.controls_overlay = ControlsOverlay()
        self.show_performance = self.config.show_performance

        # State
        self._running = False
        self._recording = False
        self._recorded_frames = []

        # Clock for FPS control
        self.clock = pygame.time.Clock()

    def set_pipeline(self, pipeline: RealtimePipeline):
        """Set the inference pipeline."""
        self.pipeline = pipeline

    def set_initial_image(self, image: np.ndarray | torch.Tensor):
        """Set the initial image to start from.

        Args:
            image: Initial image (H, W, 3) numpy array or (1, 3, H, W) tensor
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not set")

        self.pipeline.set_initial_frame(image)

    def run(self):
        """Run the interactive demo loop."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not set")

        self._running = True

        # Warmup
        print("Warming up pipeline...")
        self.pipeline.warmup(num_iterations=3)
        print("Ready!")

        frame_time = 1.0 / self.config.target_fps
        last_time = time.time()

        while self._running:
            loop_start = time.time()

            # Handle events
            self._handle_events()

            # Get action from input
            action = self.input_handler.get_action()

            # Generate next frame
            gen_start = time.time()
            frame = self.pipeline.step(action)
            gen_time = time.time() - gen_start

            # Convert to displayable format
            display_frame = self._tensor_to_surface(frame)

            # Render
            self._render(display_frame)

            # Record if enabled
            if self._recording:
                self._recorded_frames.append(
                    (frame.cpu().numpy(), action.value)
                )

            # Update performance metrics
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if current_time > last_time else 0
            self.perf_overlay.update(fps, gen_time * 1000)
            last_time = current_time

            # Cap frame rate
            self.clock.tick(self.config.target_fps)

        # Cleanup
        self._cleanup()

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                elif event.key == pygame.K_h:
                    self.controls_overlay.toggle()
                elif event.key == pygame.K_p:
                    self.show_performance = not self.show_performance
                elif event.key == pygame.K_r:
                    self._toggle_recording()

                # Pass to input handler
                key_name = pygame.key.name(event.key)
                self.input_handler.set_key(key_name, True)

            elif event.type == pygame.KEYUP:
                key_name = pygame.key.name(event.key)
                self.input_handler.set_key(key_name, False)

    def _tensor_to_surface(self, tensor: torch.Tensor) -> pygame.Surface:
        """Convert tensor to pygame surface."""
        # tensor: (1, 3, H, W) in [0, 1]
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # (3, H, W) -> (H, W, 3)
        array = tensor.permute(1, 2, 0).cpu().numpy()
        array = (array * 255).astype(np.uint8)

        # Create surface
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        # Scale to window size
        surface = pygame.transform.scale(
            surface,
            (self.config.window_width, self.config.window_height),
        )

        return surface

    def _render(self, frame_surface: pygame.Surface):
        """Render frame and overlays."""
        # Draw frame
        self.screen.blit(frame_surface, (0, 0))

        # Draw overlays
        if self.show_performance:
            self.perf_overlay.render(self.screen)

        self.controls_overlay.render(self.screen)

        # Recording indicator
        if self._recording:
            pygame.draw.circle(self.screen, (255, 0, 0), (30, self.config.window_height - 30), 10)

        # Update display
        pygame.display.flip()

    def _toggle_recording(self):
        """Toggle recording state."""
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """Start recording."""
        self._recording = True
        self._recorded_frames = []
        print("Recording started...")

    def _stop_recording(self):
        """Stop recording and save."""
        self._recording = False

        if not self._recorded_frames:
            print("No frames recorded.")
            return

        # Save recording
        output_dir = Path(self.config.recording_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        recording_dir = output_dir / f"recording_{timestamp}"
        recording_dir.mkdir()

        print(f"Saving {len(self._recorded_frames)} frames...")

        # Save frames and actions
        frames_dir = recording_dir / "frames"
        frames_dir.mkdir()

        actions = []
        for i, (frame, action) in enumerate(self._recorded_frames):
            # Save frame
            frame_uint8 = (frame.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)
            frame_path = frames_dir / f"{i:06d}.png"

            try:
                import cv2
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))
            except ImportError:
                import imageio
                imageio.imwrite(str(frame_path), frame_uint8)

            actions.append(action)

        # Save actions
        import json
        with open(recording_dir / "actions.json", "w") as f:
            json.dump(actions, f)

        print(f"Recording saved to {recording_dir}")
        self._recorded_frames = []

    def _cleanup(self):
        """Cleanup resources."""
        if self._recording:
            self._stop_recording()

        pygame.quit()


def run_demo(
    model_path: Optional[str] = None,
    initial_image: Optional[str] = None,
    config: Optional[DemoConfig] = None,
):
    """Run the interactive demo.

    Args:
        model_path: Path to model checkpoint
        initial_image: Path to initial image
        config: Demo configuration
    """
    if config is None:
        config = DemoConfig()

    if model_path:
        config.model_path = model_path

    # Load pipeline
    if config.model_path:
        print(f"Loading model from {config.model_path}...")
        pipeline_config = PipelineConfig(device=config.device)
        pipeline = RealtimePipeline.from_pretrained(config.model_path, pipeline_config)
    else:
        print("No model path provided. Using dummy pipeline for testing...")
        pipeline = create_dummy_pipeline(config.device)

    # Create app
    app = InteractiveApp(config, pipeline)

    # Load initial image
    if initial_image:
        print(f"Loading initial image from {initial_image}...")
        try:
            import cv2
            img = cv2.imread(initial_image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except ImportError:
            import imageio
            img = imageio.imread(initial_image)

        img = img.astype(np.float32) / 255.0
        app.set_initial_image(img)
    else:
        # Use random initial image for testing
        print("Using random initial image...")
        img = np.random.rand(480, 640, 3).astype(np.float32)
        app.set_initial_image(img)

    # Run
    print("Starting demo...")
    app.run()


class MockVAE:
    """Mock VAE for testing without a real VAE model."""

    def __init__(self, latent_channels: int = 4, scale_factor: int = 8):
        self.latent_channels = latent_channels
        self.scale_factor = scale_factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Mock encode: return random latents of correct shape."""
        b, c, h, w = x.shape
        latent_h = h // self.scale_factor
        latent_w = w // self.scale_factor
        return torch.randn(b, self.latent_channels, latent_h, latent_w, device=x.device, dtype=x.dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Mock decode: return random images of correct shape."""
        b, c, h, w = z.shape
        img_h = h * self.scale_factor
        img_w = w * self.scale_factor
        return torch.randn(b, 3, img_h, img_w, device=z.device, dtype=z.dtype)


def create_dummy_pipeline(device: str = "cuda"):
    """Create a dummy pipeline for testing without a trained model."""
    from models import CausalDiT, ActionEncoder, StreamVAE

    # Create models with small dimensions for testing
    dit = CausalDiT(
        in_channels=4,  # Smaller for testing
        hidden_dim=256,
        num_heads=4,
        num_layers=4,
        action_injection_layers=[1, 2],
    )

    action_encoder = ActionEncoder(
        num_actions=17,
        embedding_dim=128,
        hidden_dim=256,
    )

    # Use mock VAE for testing
    vae = StreamVAE(tile_size=256, latent_channels=4)
    vae.vae = MockVAE(latent_channels=4)  # Set mock VAE

    config = PipelineConfig(
        height=480,
        width=640,
        num_inference_steps=2,
        use_kv_cache=False,
        use_motion_control=False,
        compile_model=False,
        device=device,
        dtype="float32",  # Use float32 for testing to avoid dtype issues
    )

    return RealtimePipeline(dit, action_encoder, vae, config)


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Z-Image World Interactive Demo")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Path to initial image")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument("--fullscreen", action="store_true", help="Fullscreen mode")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    config = DemoConfig(
        window_width=args.width,
        window_height=args.height,
        fullscreen=args.fullscreen,
        device=args.device,
    )

    run_demo(
        model_path=args.model,
        initial_image=args.image,
        config=config,
    )
