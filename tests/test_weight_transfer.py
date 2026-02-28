"""Tests for weight transfer utilities.

All tests use synthetic (randomly-initialized) checkpoints at small model
dimensions to avoid requiring real pretrained weights.

The _make_pretrained_sd() helper inverts build_default_key_map() so that
the synthetic checkpoint has Z-Image-style keys whose tensor shapes match
what the target CausalDiT expects.
"""

import os
import tempfile

import pytest
import torch

from models.causal_dit import CausalDiT
from models.weight_transfer import (
    TransferReport,
    WeightTransfer,
    build_default_key_map,
    load_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_model(**extra) -> CausalDiT:
    """CausalDiT at tiny dimensions for fast CPU tests."""
    defaults = dict(
        in_channels=4,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        patch_size=2,
        num_frames=4,
        action_injection_layers=[1],
    )
    defaults.update(extra)
    return CausalDiT(**defaults)


def _make_pretrained_sd(model: CausalDiT) -> dict[str, torch.Tensor]:
    """Create a synthetic Z-Image-style checkpoint by inverting the key map.

    Generates random tensors whose shapes match what the target model expects,
    keyed with Z-Image-Turbo format. This simulates a real checkpoint without
    requiring actual pretrained weights.
    """
    km = build_default_key_map(model.num_layers)
    inv = {causal_key: z_key for z_key, causal_key in km.items()}

    model_sd = model.state_dict()
    pretrained: dict[str, torch.Tensor] = {}
    for causal_key, z_key in inv.items():
        if causal_key in model_sd:
            pretrained[z_key] = torch.randn_like(model_sd[causal_key])
    return pretrained


def _save_and_get_path(state_dict: dict, suffix: str = ".pt") -> str:
    """Save a state dict to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        torch.save(state_dict, f.name)
        return f.name


# ---------------------------------------------------------------------------
# TestBuildDefaultKeyMap
# ---------------------------------------------------------------------------

class TestBuildDefaultKeyMap:
    """Tests for the pure build_default_key_map() function."""

    def test_returns_dict(self):
        km = build_default_key_map(num_layers=2)
        assert isinstance(km, dict)
        assert len(km) > 0

    def test_patch_embed_mapped(self):
        km = build_default_key_map(num_layers=2)
        assert "transformer.patch_embed.proj.weight" in km
        assert km["transformer.patch_embed.proj.weight"] == "patch_embed.weight"
        assert km["transformer.patch_embed.proj.bias"] == "patch_embed.bias"

    def test_spatial_attn_mapped_per_layer(self):
        km = build_default_key_map(num_layers=3)
        for i in range(3):
            assert km[f"transformer.blocks.{i}.attn.qkv.weight"] == \
                   f"blocks.{i}.spatial_attn.to_qkv.weight"
            assert km[f"transformer.blocks.{i}.attn.proj.weight"] == \
                   f"blocks.{i}.spatial_attn.to_out.weight"

    def test_mlp_index_offset(self):
        """ff.net.2 (Z-Image) → mlp.3 (CausalDiT) due to Dropout at index 2."""
        km = build_default_key_map(num_layers=1)
        assert km["transformer.blocks.0.ff.net.0.weight"] == "blocks.0.mlp.0.weight"
        assert km["transformer.blocks.0.ff.net.2.weight"] == "blocks.0.mlp.3.weight"

    def test_timestep_index_offset(self):
        """Z-Image mlp.0 → timestep_embed.1 (sinusoidal is at index 0, no params)."""
        km = build_default_key_map(num_layers=1)
        assert km["transformer.t_embedder.mlp.0.weight"] == "timestep_embed.1.weight"
        assert km["transformer.t_embedder.mlp.2.weight"] == "timestep_embed.3.weight"

    def test_final_layer_mapped(self):
        km = build_default_key_map(num_layers=1)
        assert km["transformer.final_layer.linear.weight"] == "final_linear.weight"
        assert km["transformer.final_layer.norm_final.weight"] == "final_norm.weight"

    def test_temporal_keys_not_in_values(self):
        """Temporal attention keys must NOT be targets of the key map."""
        km = build_default_key_map(num_layers=4)
        mapped_targets = set(km.values())
        assert not any("temporal_attn" in v for v in mapped_targets)

    def test_action_injection_keys_not_in_values(self):
        km = build_default_key_map(num_layers=4)
        mapped_targets = set(km.values())
        assert not any("action_injection" in v for v in mapped_targets)


# ---------------------------------------------------------------------------
# TestRemapStateDict
# ---------------------------------------------------------------------------

class TestRemapStateDict:
    """Tests for WeightTransfer.remap_state_dict() (pure function)."""

    def test_maps_known_key(self):
        km = {"z.attn.qkv.weight": "blocks.0.spatial_attn.to_qkv.weight"}
        transfer = WeightTransfer(num_layers=1, key_map=km)
        t = torch.randn(192, 64)
        remapped, unexpected = transfer.remap_state_dict({"z.attn.qkv.weight": t})
        assert "blocks.0.spatial_attn.to_qkv.weight" in remapped
        assert len(unexpected) == 0

    def test_unknown_key_goes_to_unexpected(self):
        km = {"known": "target"}
        transfer = WeightTransfer(num_layers=1, key_map=km)
        _, unexpected = transfer.remap_state_dict({"known": torch.zeros(1), "extra": torch.zeros(1)})
        assert "extra" in unexpected

    def test_empty_checkpoint(self):
        transfer = WeightTransfer(num_layers=2)
        remapped, unexpected = transfer.remap_state_dict({})
        assert remapped == {}
        assert unexpected == []


# ---------------------------------------------------------------------------
# TestLoadCheckpoint
# ---------------------------------------------------------------------------

class TestLoadCheckpoint:
    """Tests for the load_checkpoint() file I/O helper."""

    def test_load_pt_file(self):
        sd = {"w": torch.randn(3, 4)}
        path = _save_and_get_path(sd, suffix=".pt")
        try:
            loaded = load_checkpoint(path)
            assert "w" in loaded
            assert loaded["w"].shape == (3, 4)
        finally:
            os.unlink(path)

    def test_unwrap_nested_model_key(self):
        inner = {"weight": torch.randn(4)}
        path = _save_and_get_path({"model": inner, "optimizer": {}})
        try:
            loaded = load_checkpoint(path)
            assert "weight" in loaded
        finally:
            os.unlink(path)

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path/checkpoint.pt")

    def test_unrecognized_extension_raises_value_error(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unrecognized checkpoint format"):
                load_checkpoint(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestWeightTransferLoad
# ---------------------------------------------------------------------------

class TestWeightTransferLoad:
    """End-to-end tests for WeightTransfer.load()."""

    def test_spatial_keys_change_after_load(self):
        """Spatial parameters must change from their fresh-init values."""
        model = _small_model()
        original = {k: v.clone() for k, v in model.state_dict().items()}
        pretrained = _make_pretrained_sd(model)
        path = _save_and_get_path(pretrained)
        try:
            WeightTransfer(num_layers=model.num_layers).load(model, path)
            new_sd = model.state_dict()
            changed = [
                k for k in new_sd
                if not torch.allclose(new_sd[k], original[k])
                and "temporal_attn" not in k
                and "action_injection" not in k
            ]
            assert len(changed) > 0, "No spatial keys were updated"
        finally:
            os.unlink(path)

    def test_report_loaded_nonempty(self):
        model = _small_model()
        path = _save_and_get_path(_make_pretrained_sd(model))
        try:
            report = WeightTransfer(num_layers=model.num_layers).load(model, path)
            assert len(report.loaded) > 0
        finally:
            os.unlink(path)

    def test_temporal_keys_in_new_layers(self):
        """Missing temporal keys must appear in report.new_layers, not .missing."""
        model = _small_model()
        path = _save_and_get_path(_make_pretrained_sd(model))
        try:
            report = WeightTransfer(num_layers=model.num_layers).load(model, path)
            temporal_keys = [k for k in model.state_dict() if "temporal_attn" in k]
            for k in temporal_keys:
                assert k in report.new_layers, f"{k} not in report.new_layers"
            for k in report.missing:
                assert "temporal_attn" not in k, f"Temporal key {k} wrongly in report.missing"
        finally:
            os.unlink(path)

    def test_action_injection_keys_in_new_layers(self):
        model = _small_model()
        path = _save_and_get_path(_make_pretrained_sd(model))
        try:
            report = WeightTransfer(num_layers=model.num_layers).load(model, path)
            action_keys = [k for k in model.state_dict() if "action_injection" in k]
            for k in action_keys:
                assert k in report.new_layers, f"{k} not in report.new_layers"
        finally:
            os.unlink(path)

    def test_temporal_gamma_remains_zero(self):
        """temporal_attn.gamma must stay at 0.0 after weight transfer."""
        model = _small_model()
        path = _save_and_get_path(_make_pretrained_sd(model))
        try:
            WeightTransfer(num_layers=model.num_layers).load(model, path)
            sd = model.state_dict()
            for i in range(model.num_layers):
                key = f"blocks.{i}.temporal_attn.gamma"
                assert sd[key].item() == 0.0, f"{key} should be 0, got {sd[key].item()}"
        finally:
            os.unlink(path)

    def test_no_missing_spatial_keys_for_complete_checkpoint(self):
        """A complete synthetic checkpoint should produce no missing spatial keys."""
        model = _small_model()
        path = _save_and_get_path(_make_pretrained_sd(model))
        try:
            report = WeightTransfer(num_layers=model.num_layers).load(model, path)
            assert report.missing == [], f"Unexpected missing spatial keys: {report.missing}"
        finally:
            os.unlink(path)

    def test_custom_key_map_override(self):
        """A custom key_map dict should be used instead of the default."""
        model = _small_model()
        custom_map = {"patch_embed.weight": "patch_embed.weight"}
        pretrained = {"patch_embed.weight": torch.randn_like(model.patch_embed.weight)}
        path = _save_and_get_path(pretrained)
        try:
            transfer = WeightTransfer(num_layers=model.num_layers, key_map=custom_map)
            report = transfer.load(model, path)
            assert "patch_embed.weight" in report.loaded
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestTransferReportLog
# ---------------------------------------------------------------------------

class TestTransferReportLog:
    def test_log_does_not_raise(self, caplog):
        """report.log() must not raise even with partially filled lists."""
        import logging
        report = TransferReport(
            loaded=["blocks.0.spatial_attn.to_qkv.weight"],
            new_layers=["blocks.0.temporal_attn.gamma"],
            missing=[],
            unexpected=["some.extra.key"],
        )
        with caplog.at_level(logging.INFO):
            report.log()

    def test_log_custom_fn(self):
        """report.log(logger_fn=...) routes output to the custom function."""
        messages = []
        report = TransferReport(loaded=["w"], new_layers=[], missing=[], unexpected=[])
        report.log(logger_fn=messages.append)
        assert len(messages) == 1
        assert "1 loaded" in messages[0]


# ---------------------------------------------------------------------------
# TestFromPretrained (integration with CausalDiT)
# ---------------------------------------------------------------------------

class TestFromPretrained:
    """Integration tests for CausalDiT.from_pretrained()."""

    def test_returns_causal_dit_instance(self):
        model = _small_model()
        path = _save_and_get_path(_make_pretrained_sd(model))
        try:
            loaded = CausalDiT.from_pretrained(
                path,
                in_channels=4, hidden_dim=64, num_heads=4,
                num_layers=2, patch_size=2, num_frames=4,
                action_injection_layers=[1],
            )
            assert isinstance(loaded, CausalDiT)
        finally:
            os.unlink(path)

    def test_forward_pass_after_loading(self):
        """Model loaded via from_pretrained must produce finite outputs."""
        model = _small_model()
        path = _save_and_get_path(_make_pretrained_sd(model))
        try:
            loaded = CausalDiT.from_pretrained(
                path,
                in_channels=4, hidden_dim=64, num_heads=4,
                num_layers=2, patch_size=2, num_frames=4,
                action_injection_layers=[1],
            )
            x = torch.randn(1, 4, 16, 16)
            t = torch.tensor([0.5])
            out, _ = loaded(x, t)
            assert out.shape == x.shape
            assert torch.isfinite(out).all()
        finally:
            os.unlink(path)

    def test_strict_spatial_raises_on_empty_checkpoint(self):
        """strict_spatial=True + empty checkpoint → RuntimeError."""
        path = _save_and_get_path({})
        try:
            with pytest.raises(RuntimeError, match="strict_spatial"):
                CausalDiT.from_pretrained(
                    path,
                    in_channels=4, hidden_dim=64, num_heads=4,
                    num_layers=2, patch_size=2, num_frames=4,
                    strict_spatial=True,
                )
        finally:
            os.unlink(path)

    def test_verbose_false_suppresses_logging(self, caplog):
        """verbose=False should not emit any log messages."""
        import logging
        model = _small_model()
        path = _save_and_get_path(_make_pretrained_sd(model))
        try:
            with caplog.at_level(logging.DEBUG):
                CausalDiT.from_pretrained(
                    path,
                    in_channels=4, hidden_dim=64, num_heads=4,
                    num_layers=2, patch_size=2, num_frames=4,
                    verbose=False,
                )
            # caplog may capture unrelated messages; check none are from causal_dit
            causal_dit_logs = [r for r in caplog.records if "causal_dit" in r.name]
            assert len(causal_dit_logs) == 0
        finally:
            os.unlink(path)
