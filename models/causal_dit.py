"""
Causal DiT (Diffusion Transformer) for Z-Image World Model

Modified Z-Image DiT with:
1. Causal attention masking for autoregressive frame generation
2. Temporal attention layers for frame-to-frame consistency
3. Rolling KV cache support for streaming inference
4. Action conditioning injection points
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .action_encoder import ActionInjectionLayer


def get_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask.

    Args:
        seq_len: Sequence length
        device: Target device

    Returns:
        Causal mask of shape (1, 1, seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timestep and frame position."""

    def __init__(self, dim: int, max_length: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_length = max_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal embeddings.

        Args:
            x: Position indices (batch,) or (batch, seq) - any dtype

        Returns:
            Embeddings of shape (batch, dim) or (batch, seq, dim)
            Output dtype matches input dtype.
        """
        device = x.device
        # Compute in float32 for numerical precision, then cast to input dtype
        x_float = x.float()
        half_dim = self.dim // 2
        emb = math.log(self.max_length) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)

        if x_float.dim() == 1:
            emb = x_float[:, None] * emb[None, :]
        else:
            emb = x_float.unsqueeze(-1) * emb[None, None, :]

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # Always float32; caller casts to model dtype


class CausalAttention(nn.Module):
    """Multi-head attention with causal masking and KV cache support.

    Supports:
    - Causal masking for autoregressive generation
    - Rolling KV cache with sink tokens
    - Flash attention when available
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_flash_attention = use_flash_attention

        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV cache.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            causal_mask: Optional causal attention mask
            kv_cache: Optional (key, value) cache from previous steps
            use_cache: Whether to return updated KV cache

        Returns:
            Output tensor and optionally updated KV cache
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (three h d) -> three b h n d", three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Handle KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Attention computation
        if self.use_flash_attention and hasattr(F, "scaled_dot_product_attention"):
            # Use PyTorch's flash attention
            is_causal = causal_mask is None and kv_cache is None
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal_mask if not is_causal else None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Standard attention
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if causal_mask is not None:
                attn = attn + causal_mask

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return out, new_cache


class TemporalAttention(nn.Module):
    """Attention across temporal (frame) dimension for consistency.

    Applied after spatial attention to maintain frame-to-frame coherence.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        num_frames: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_frames = num_frames

        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = CausalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attention=True,
        )

        # Zero-init for residual
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        causal: bool = True,
    ) -> torch.Tensor:
        """Apply temporal attention.

        Args:
            x: Input (batch * num_frames, num_patches, hidden_dim)
            num_frames: Number of frames in the sequence
            causal: Whether to use causal masking

        Returns:
            Output with temporal attention applied
        """
        bf, n, d = x.shape
        batch_size = bf // num_frames

        # Reshape to (batch * num_patches, num_frames, hidden_dim)
        x_temporal = rearrange(
            x,
            "(b f) n d -> (b n) f d",
            b=batch_size,
            f=num_frames,
        )

        # Apply temporal attention
        x_norm = self.norm(x_temporal)

        if causal:
            causal_mask = get_causal_mask(num_frames, x.device)
        else:
            causal_mask = None

        attn_out, _ = self.attention(x_norm, causal_mask=causal_mask)

        # Reshape back
        out = rearrange(
            attn_out,
            "(b n) f d -> (b f) n d",
            b=batch_size,
            n=n,
        )

        return x + self.gamma * out


class DiTBlock(nn.Module):
    """Single DiT transformer block with temporal attention.

    Structure:
    1. Self-attention (spatial, within frame)
    2. Temporal attention (across frames, causal)
    3. Feed-forward network
    4. Optional action injection
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_frames: int = 16,
        has_temporal: bool = True,
        has_action_injection: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.has_temporal = has_temporal
        self.has_action_injection = has_action_injection

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Spatial self-attention
        self.spatial_attn = CausalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Temporal attention (optional)
        if has_temporal:
            self.temporal_attn = TemporalAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_frames=num_frames,
                dropout=dropout,
            )

        # Feed-forward network
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Action injection (optional)
        if has_action_injection:
            self.action_injection = ActionInjectionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        # AdaLN modulation for timestep conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: torch.Tensor,
        num_frames: int = 1,
        action_conditioning: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        """Forward pass through DiT block.

        Args:
            x: Input tensor (batch * num_frames, seq_len, hidden_dim)
            timestep_emb: Timestep embedding (batch * num_frames, hidden_dim)
            num_frames: Number of frames for temporal attention
            action_conditioning: Optional action embeddings
            kv_cache: Optional KV cache
            use_cache: Whether to return updated cache

        Returns:
            Output tensor and optionally updated cache
        """
        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(timestep_emb).chunk(6, dim=-1)
        )

        # Spatial self-attention
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, new_cache = self.spatial_attn(x_norm, kv_cache=kv_cache, use_cache=use_cache)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # Temporal attention
        if self.has_temporal and num_frames > 1:
            x = self.temporal_attn(x, num_frames=num_frames, causal=True)

        # Action injection
        if self.has_action_injection and action_conditioning is not None:
            x = self.action_injection(x, action_conditioning)

        # Feed-forward
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x, new_cache


class CausalDiT(nn.Module):
    """Causal Diffusion Transformer for autoregressive frame generation.

    Modified Z-Image DiT architecture with:
    - Causal temporal attention for frame-by-frame generation
    - Action conditioning injection at specified layers
    - Rolling KV cache support for streaming
    - Diffusion Forcing training support
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        num_layers: int = 28,
        patch_size: int = 2,
        num_frames: int = 16,
        action_injection_layers: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_frames = num_frames

        if action_injection_layers is None:
            # Default: inject at 1/4, 1/2, 3/4 depth
            action_injection_layers = [7, 14, 21]
        self.action_injection_layers = set(action_injection_layers)

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Position embedding (learnable)
        self.max_seq_len = 4096  # Maximum sequence length
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, hidden_dim))

        # Frame position embedding
        self.frame_pos_embed = nn.Embedding(num_frames, hidden_dim)

        # Timestep embedding
        self.timestep_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_frames=num_frames,
                has_temporal=True,
                has_action_injection=(i in self.action_injection_layers),
            )
            for i in range(num_layers)
        ])

        # Final layers
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_linear = nn.Linear(hidden_dim, patch_size**2 * in_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Patch embedding
        w = self.patch_embed.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)

        # Final linear (zero-init for residual)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert images to patch embeddings.

        Args:
            x: Input images (batch, channels, height, width)

        Returns:
            Patch embeddings (batch, num_patches, hidden_dim)
        """
        x = self.patch_embed(x)  # (B, D, H/P, W/P)
        x = rearrange(x, "b d h w -> b (h w) d")
        return x

    def unpatchify(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Convert patch embeddings back to images.

        Args:
            x: Patch embeddings (batch, num_patches, patch_size^2 * channels)
            height: Original image height
            width: Original image width

        Returns:
            Images (batch, channels, height, width)
        """
        p = self.patch_size
        h = height // p
        w = width // p

        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h,
            w=w,
            p1=p,
            p2=p,
        )
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        action_conditioning: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        kv_cache: Optional[list[tuple]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[tuple]]]:
        """Forward pass for diffusion denoising.

        Args:
            x: Noisy latents (batch, num_frames, channels, height, width)
                or (batch, channels, height, width) for single frame
            timesteps: Diffusion timesteps (batch,) or (batch, num_frames)
            action_conditioning: Action embeddings (batch, num_frames, hidden_dim)
            frame_indices: Frame position indices
            kv_cache: List of KV caches per layer
            use_cache: Whether to return updated caches

        Returns:
            Predicted noise and optionally updated caches
        """
        # Handle single frame vs sequence
        if x.dim() == 4:
            x = x.unsqueeze(1)
            num_frames = 1
        else:
            num_frames = x.shape[1]

        # Cast input to model's working dtype (handles float32 input to bfloat16 model)
        x = x.to(self.patch_embed.weight.dtype)

        batch_size, num_frames, channels, height, width = x.shape

        # Flatten batch and frames
        x = rearrange(x, "b f c h w -> (b f) c h w")

        # Patchify
        x = self.patchify(x)
        seq_len = x.shape[1]

        # Add position embeddings
        x = x + self.pos_embed[:, :seq_len]

        # Add frame position embeddings
        if frame_indices is None:
            frame_indices = torch.arange(num_frames, device=x.device)
            frame_indices = repeat(frame_indices, "f -> (b f)", b=batch_size)
        else:
            frame_indices = rearrange(frame_indices, "b f -> (b f)")

        frame_pos = self.frame_pos_embed(frame_indices)  # (B*F, D)
        x = x + frame_pos.unsqueeze(1)

        # Timestep embedding
        if timesteps.dim() == 1:
            timesteps = repeat(timesteps, "b -> (b f)", f=num_frames)
        else:
            timesteps = rearrange(timesteps, "b f -> (b f)")

        # Sinusoidal embedding runs in float32; cast to model dtype before Linear layers
        # (timestep_embed is Sequential: sinusoidal → linear → silu → linear)
        t_emb = self.timestep_embed[0](timesteps).to(x.dtype)  # float32 → model dtype
        for layer in list(self.timestep_embed)[1:]:
            t_emb = layer(t_emb)  # (B*F, D)

        # Expand action conditioning to match (batch * num_frames) structure
        # Input: (batch, num_frames, hidden_dim) -> Output: (batch * num_frames, num_frames, hidden_dim)
        if action_conditioning is not None:
            # Repeat for each frame position so each frame can attend to all action embeddings
            action_conditioning = repeat(
                action_conditioning,
                "b f d -> (b repeat) f d",
                repeat=num_frames,
            )

        # Transform blocks
        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, cache = block(
                x,
                t_emb,
                num_frames=num_frames,
                action_conditioning=action_conditioning,
                kv_cache=layer_cache,
                use_cache=use_cache,
            )
            if use_cache:
                new_caches.append(cache)

        # Final layers
        x = self.final_norm(x)
        x = self.final_linear(x)

        # Unpatchify
        x = self.unpatchify(x, height, width)

        # Reshape back to sequence
        x = rearrange(x, "(b f) c h w -> b f c h w", b=batch_size, f=num_frames)

        # Squeeze if single frame
        if num_frames == 1:
            x = x.squeeze(1)

        return x, new_caches

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        action_injection_layers: list[int] | None = None,
        **kwargs,
    ) -> "CausalDiT":
        """Load from pretrained Z-Image checkpoint and add causal components.

        Args:
            pretrained_model_path: Path to pretrained model
            action_injection_layers: Layers for action injection
            **kwargs: Additional model arguments

        Returns:
            Initialized CausalDiT model
        """
        # This would load weights from Z-Image and initialize temporal layers
        # For now, just create the model
        model = cls(action_injection_layers=action_injection_layers, **kwargs)

        # TODO: Load pretrained weights and initialize temporal layers
        # using Vid2World weight transfer method

        return model
