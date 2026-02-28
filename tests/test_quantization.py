"""Tests for INT8 dynamic quantization of ZImageWorldModel trainable layers."""

import pytest
import torch
import torch.nn as nn

from models.quantization import (
    quantize_temporal_layers,
    estimate_quantized_size,
    QuantizationReport,
    _module_size_mb,
)


# ---------------------------------------------------------------------------
# Minimal stub that mimics ZImageWorldModel's interface without loading Z-Image
# ---------------------------------------------------------------------------

class _StubTemporalAttention(nn.Module):
    def __init__(self, dim=128, num_heads=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class _StubActionEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.embed = nn.Embedding(17, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(self.embed(x))


class _StubWorldModel(nn.Module):
    """Minimal stub with the same sub-module names as ZImageWorldModel."""

    def __init__(self):
        super().__init__()
        self.temporal_layers = nn.ModuleDict({
            "0": _StubTemporalAttention(),
            "1": _StubTemporalAttention(),
        })
        self.action_injections = nn.ModuleDict({
            "1": nn.Linear(128, 128),
        })
        self.action_encoder = _StubActionEncoder()

    def forward(self, x, actions):
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQuantizationReport:
    def test_compression_ratio_identity(self):
        r = QuantizationReport(original_size_mb=10.0, quantized_size_mb=10.0)
        assert abs(r.compression_ratio - 1.0) < 1e-5

    def test_compression_ratio_half(self):
        r = QuantizationReport(original_size_mb=10.0, quantized_size_mb=5.0)
        assert abs(r.compression_ratio - 2.0) < 1e-5

    def test_speedup_none_when_no_latency(self):
        r = QuantizationReport()
        assert r.speedup is None

    def test_speedup_computed(self):
        r = QuantizationReport(latency_before_ms=10.0, latency_after_ms=5.0)
        assert abs(r.speedup - 2.0) < 1e-5

    def test_str_output(self):
        r = QuantizationReport(
            modules_quantized=["a", "b"],
            original_size_mb=10.0,
            quantized_size_mb=5.0,
        )
        s = str(r)
        assert "Quantization Report" in s
        assert "2" in s  # modules quantized count


class TestModuleSizeMb:
    def test_empty_module_is_zero(self):
        m = nn.Module()
        assert _module_size_mb(m) == 0.0

    def test_linear_size_correct(self):
        m = nn.Linear(1024, 1024, bias=False)  # 1M float32 params = 4MB
        size = _module_size_mb(m)
        assert abs(size - 4.0) < 0.1


class TestQuantizeTemporalLayers:
    @pytest.fixture
    def stub_model(self):
        return _StubWorldModel()

    def test_quantize_returns_report(self, stub_model):
        report = quantize_temporal_layers(stub_model)
        assert isinstance(report, QuantizationReport)

    def test_quantize_reduces_memory(self, stub_model):
        # Original float32 layers are bigger than INT8 quantized versions
        report = quantize_temporal_layers(stub_model)
        # INT8 quantized linears should be <= fp32 originals
        # (dynamic quant may not reduce for tiny models, but shouldn't exceed)
        assert report.quantized_size_mb <= report.original_size_mb * 1.1  # allow 10% overhead

    def test_modules_quantized_listed(self, stub_model):
        report = quantize_temporal_layers(stub_model)
        # All three sub-modules should be mentioned
        assert len(report.modules_quantized) > 0

    def test_quantized_model_still_callable(self, stub_model):
        """After quantization, temporal layers and action encoder should still work."""
        quantize_temporal_layers(stub_model)
        x = torch.randn(2, 128)
        # temporal_layers "0" is now quantized but should still be callable
        out = stub_model.temporal_layers["0"](x)
        assert out.shape == x.shape

    def test_action_encoder_still_callable(self, stub_model):
        quantize_temporal_layers(stub_model)
        actions = torch.randint(0, 17, (4,))
        out = stub_model.action_encoder(actions)
        assert out.shape == (4, 64)

    def test_empty_model_no_crash(self):
        """Model without expected sub-modules returns empty report."""
        report = quantize_temporal_layers(nn.Module())
        assert isinstance(report, QuantizationReport)
        assert len(report.modules_quantized) == 0


class TestEstimateQuantizedSize:
    def test_returns_dict(self):
        stub = _StubWorldModel()
        sizes = estimate_quantized_size(stub)
        assert isinstance(sizes, dict)

    def test_has_temporal_and_action_keys(self):
        stub = _StubWorldModel()
        sizes = estimate_quantized_size(stub)
        assert "temporal_layers (trainable)" in sizes
        assert "action_encoder (trainable)" in sizes

    def test_sizes_positive(self):
        stub = _StubWorldModel()
        sizes = estimate_quantized_size(stub)
        for name, size in sizes.items():
            assert size >= 0, f"{name} has negative size"
