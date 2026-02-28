"""Pytest configuration and fixtures."""

import os

# Fix protobuf/onnx incompatibility with system torch installation
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
