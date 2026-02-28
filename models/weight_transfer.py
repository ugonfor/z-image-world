"""
Weight transfer utilities for loading pretrained Z-Image-Turbo spatial
weights into CausalDiT.

The core naming gap:

  Z-Image-Turbo key                         CausalDiT key
  ──────────────────────────────────────    ────────────────────────────────────
  transformer.blocks.{i}.attn.qkv.weight   blocks.{i}.spatial_attn.to_qkv.weight
  transformer.blocks.{i}.ff.net.0.weight   blocks.{i}.mlp.0.weight
  transformer.blocks.{i}.ff.net.2.weight   blocks.{i}.mlp.3.weight  ← +1 for Dropout

Temporal and action-injection layers (temporal_attn.*, action_injection.*)
are intentionally absent from the key map. They stay at their zero-gamma
initialization and are trained from scratch on video data.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key map builder
# ---------------------------------------------------------------------------

def build_default_key_map(num_layers: int) -> dict[str, str]:
    """Build the default Z-Image-Turbo → CausalDiT key mapping.

    Args:
        num_layers: Number of transformer blocks in the target model.

    Returns:
        Dict of {pretrained_key: causal_dit_key} for all spatial parameters.
        Temporal and action-injection keys are NOT included (they must be
        trained from scratch).
    """
    key_map: dict[str, str] = {}

    # Patch embedding
    key_map["transformer.patch_embed.proj.weight"] = "patch_embed.weight"
    key_map["transformer.patch_embed.proj.bias"]   = "patch_embed.bias"

    # Timestep MLP
    # CausalDiT timestep_embed = Sequential(Sinusoidal[0], Linear[1], SiLU[2], Linear[3])
    # Z-Image t_embedder.mlp  = Sequential(Linear[0], SiLU[1], Linear[2])
    # → offset by 1 because index 0 in CausalDiT is the param-free sinusoidal layer
    key_map["transformer.t_embedder.mlp.0.weight"] = "timestep_embed.1.weight"
    key_map["transformer.t_embedder.mlp.0.bias"]   = "timestep_embed.1.bias"
    key_map["transformer.t_embedder.mlp.2.weight"] = "timestep_embed.3.weight"
    key_map["transformer.t_embedder.mlp.2.bias"]   = "timestep_embed.3.bias"

    # Final output layers
    key_map["transformer.final_layer.norm_final.weight"] = "final_norm.weight"
    key_map["transformer.final_layer.norm_final.bias"]   = "final_norm.bias"
    key_map["transformer.final_layer.linear.weight"]     = "final_linear.weight"
    key_map["transformer.final_layer.linear.bias"]       = "final_linear.bias"

    # Per-block mappings
    for i in range(num_layers):
        z = f"transformer.blocks.{i}"   # Z-Image prefix
        c = f"blocks.{i}"               # CausalDiT prefix

        # Spatial self-attention
        # Note: to_qkv has bias=False → no bias key
        key_map[f"{z}.attn.qkv.weight"]  = f"{c}.spatial_attn.to_qkv.weight"
        key_map[f"{z}.attn.proj.weight"] = f"{c}.spatial_attn.to_out.weight"
        key_map[f"{z}.attn.proj.bias"]   = f"{c}.spatial_attn.to_out.bias"

        # Feed-forward network
        # CausalDiT mlp = Sequential(Linear[0], GELU[1], Dropout[2], Linear[3], Dropout[4])
        # Z-Image ff.net = Sequential(Linear[0], GELU[1], Linear[2])
        # → second Linear is at index 3 in CausalDiT (not 2) because Dropout sits between
        key_map[f"{z}.ff.net.0.weight"] = f"{c}.mlp.0.weight"
        key_map[f"{z}.ff.net.0.bias"]   = f"{c}.mlp.0.bias"
        key_map[f"{z}.ff.net.2.weight"] = f"{c}.mlp.3.weight"
        key_map[f"{z}.ff.net.2.bias"]   = f"{c}.mlp.3.bias"

        # Layer norms
        key_map[f"{z}.norm1.weight"] = f"{c}.norm1.weight"
        key_map[f"{z}.norm1.bias"]   = f"{c}.norm1.bias"
        key_map[f"{z}.norm2.weight"] = f"{c}.norm2.weight"
        key_map[f"{z}.norm2.bias"]   = f"{c}.norm2.bias"

        # AdaLN modulation
        # CausalDiT adaLN_modulation = Sequential(SiLU[0], Linear[1])
        # → index 1 is the Linear layer with params
        key_map[f"{z}.adaLN_modulation.1.weight"] = f"{c}.adaLN_modulation.1.weight"
        key_map[f"{z}.adaLN_modulation.1.bias"]   = f"{c}.adaLN_modulation.1.bias"

    # NOT mapped (CausalDiT-specific, initialized by _init_weights):
    #   pos_embed          - learnable position embedding (trunc_normal init)
    #   frame_pos_embed    - frame index embedding (standard Embedding init)
    #
    # NOT mapped (new temporal/action layers, zero-gamma init, train from scratch):
    #   blocks.{i}.temporal_attn.*
    #   blocks.{i}.action_injection.*

    return key_map


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    """Load a checkpoint file, supporting .safetensors and .pt / .pth / .bin.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Flat state dict {key: tensor}.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is not recognized.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path)

    if ext in (".pt", ".pth", ".bin"):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        # Unwrap common nested formats: {"model": {...}, "optimizer": {...}}
        if isinstance(state_dict, dict):
            for nested_key in ("model", "state_dict", "model_state_dict"):
                if nested_key in state_dict and isinstance(state_dict[nested_key], dict):
                    return state_dict[nested_key]
        return state_dict

    raise ValueError(
        f"Unrecognized checkpoint format: {ext!r}. "
        "Supported extensions: .safetensors, .pt, .pth, .bin"
    )


# ---------------------------------------------------------------------------
# Transfer report
# ---------------------------------------------------------------------------

@dataclass
class TransferReport:
    """Summary of what happened during weight transfer.

    Attributes:
        loaded:     CausalDiT keys successfully updated from the checkpoint.
        new_layers: Temporal / action-injection keys expected to be missing
                    (they are zero-initialized and must be trained).
        missing:    Spatial keys absent from the checkpoint (unexpected — may
                    indicate a key-map bug or a mismatched checkpoint).
        unexpected: Checkpoint keys that were not in the key map (silently
                    ignored during load).
    """
    loaded: list[str] = field(default_factory=list)
    new_layers: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    def log(self, logger_fn=None):
        """Log a human-readable transfer summary."""
        if logger_fn is None:
            logger_fn = logger.info
        logger_fn(
            f"Weight transfer: {len(self.loaded)} loaded, "
            f"{len(self.new_layers)} new (temporal/action, zero-init), "
            f"{len(self.missing)} missing spatial, "
            f"{len(self.unexpected)} unexpected from checkpoint"
        )
        if self.missing:
            logger_fn(f"  Missing spatial keys: {self.missing}")
        if self.unexpected:
            logger_fn(f"  Unexpected checkpoint keys (not mapped): {self.unexpected[:10]}")


# ---------------------------------------------------------------------------
# WeightTransfer
# ---------------------------------------------------------------------------

_NEW_LAYER_PREFIXES = ("temporal_attn.", "action_injection.")

# CausalDiT-specific parameters that have no equivalent in Z-Image-Turbo.
# These are intentionally absent from the pretrained checkpoint and are
# initialised by CausalDiT._init_weights / Embedding default init.
_CAUSAL_DIT_SPECIFIC_KEYS = frozenset({"pos_embed", "frame_pos_embed.weight"})


class WeightTransfer:
    """Transfers pretrained Z-Image-Turbo spatial weights to CausalDiT.

    Usage::

        transfer = WeightTransfer(num_layers=28)
        report = transfer.load(model, "/path/to/zimage.safetensors")
        report.log()

    Args:
        num_layers: Number of transformer blocks in the target model.
        key_map: Custom {pretrained_key: causal_dit_key} mapping.  ``None``
                 uses the default Z-Image-Turbo mapping for *num_layers*.
    """

    def __init__(
        self,
        num_layers: int = 28,
        key_map: Optional[dict[str, str]] = None,
    ):
        self.num_layers = num_layers
        self.key_map = key_map if key_map is not None else build_default_key_map(num_layers)

    def remap_state_dict(
        self,
        pretrained_sd: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """Apply key_map to a pretrained state dict.

        Args:
            pretrained_sd: Raw checkpoint tensors keyed with Z-Image names.

        Returns:
            ``(remapped_sd, unexpected)`` where *remapped_sd* uses CausalDiT
            key names and *unexpected* lists pretrained keys not in the map.
        """
        remapped: dict[str, torch.Tensor] = {}
        unexpected: list[str] = []

        for pretrained_key, tensor in pretrained_sd.items():
            if pretrained_key in self.key_map:
                remapped[self.key_map[pretrained_key]] = tensor
            else:
                unexpected.append(pretrained_key)

        return remapped, unexpected

    def load(
        self,
        model: nn.Module,
        checkpoint_path: str,
    ) -> TransferReport:
        """Load pretrained weights into *model* in-place.

        Temporal and action-injection layers are left at their zero-gamma
        initialization — this is expected, not an error.

        Args:
            model: CausalDiT instance (modified in-place).
            checkpoint_path: Path to pretrained checkpoint.

        Returns:
            :class:`TransferReport` describing what was loaded and what was
            skipped.
        """
        # Step 1: Load checkpoint
        pretrained_sd = load_checkpoint(checkpoint_path)

        # Step 2: Remap keys from Z-Image → CausalDiT
        remapped_sd, unexpected = self.remap_state_dict(pretrained_sd)

        # Step 3: Load with strict=False (temporal/action keys are new)
        result = model.load_state_dict(remapped_sd, strict=False)

        # Step 4: Classify missing keys
        new_layers: list[str] = []
        missing_spatial: list[str] = []

        for key in result.missing_keys:
            # Strip "blocks.{i}." prefix to get the local module name
            parts = key.split(".")
            if len(parts) >= 3 and parts[0] == "blocks":
                local = ".".join(parts[2:])  # e.g. "temporal_attn.gamma"
            else:
                local = key

            if key in _CAUSAL_DIT_SPECIFIC_KEYS:
                # CausalDiT-specific params: new by design, initialised locally
                new_layers.append(key)
            elif any(local.startswith(p) for p in _NEW_LAYER_PREFIXES):
                new_layers.append(key)
            else:
                missing_spatial.append(key)

        # Keys in remapped_sd that were actually consumed (not in missing_keys)
        loaded_keys = [k for k in remapped_sd if k not in result.missing_keys]

        return TransferReport(
            loaded=loaded_keys,
            new_layers=new_layers,
            missing=missing_spatial,
            unexpected=unexpected,
        )
