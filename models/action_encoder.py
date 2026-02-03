"""
Action Encoder for Z-Image World Model

Translates discrete keyboard actions into embeddings that condition the DiT model.
"""

from enum import IntEnum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ActionSpace(IntEnum):
    """17 discrete actions for game-like environments."""

    # Movement (8 directions + idle)
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    MOVE_FORWARD_LEFT = 4
    MOVE_FORWARD_RIGHT = 5
    MOVE_BACKWARD_LEFT = 6
    MOVE_BACKWARD_RIGHT = 7
    IDLE = 8

    # Camera/View
    LOOK_UP = 9
    LOOK_DOWN = 10
    LOOK_LEFT = 11
    LOOK_RIGHT = 12

    # Interactions
    JUMP = 13
    CROUCH = 14
    INTERACT = 15
    ATTACK = 16

    @classmethod
    def num_actions(cls) -> int:
        return 17

    @classmethod
    def from_keyboard(cls, keys: set[str]) -> "ActionSpace":
        """Convert keyboard keys to action.

        Args:
            keys: Set of currently pressed keys (e.g., {"w", "a"})

        Returns:
            The corresponding action
        """
        # Movement mapping
        w = "w" in keys or "up" in keys
        s = "s" in keys or "down" in keys
        a = "a" in keys or "left" in keys
        d = "d" in keys or "right" in keys

        # Diagonal movement
        if w and a:
            return cls.MOVE_FORWARD_LEFT
        if w and d:
            return cls.MOVE_FORWARD_RIGHT
        if s and a:
            return cls.MOVE_BACKWARD_LEFT
        if s and d:
            return cls.MOVE_BACKWARD_RIGHT

        # Cardinal movement
        if w:
            return cls.MOVE_FORWARD
        if s:
            return cls.MOVE_BACKWARD
        if a:
            return cls.MOVE_LEFT
        if d:
            return cls.MOVE_RIGHT

        # Camera/View (using i/j/k/l or arrow keys with shift)
        if "i" in keys:
            return cls.LOOK_UP
        if "k" in keys:
            return cls.LOOK_DOWN
        if "j" in keys:
            return cls.LOOK_LEFT
        if "l" in keys:
            return cls.LOOK_RIGHT

        # Interactions
        if "space" in keys:
            return cls.JUMP
        if "c" in keys or "ctrl" in keys:
            return cls.CROUCH
        if "e" in keys:
            return cls.INTERACT
        if "mouse1" in keys or "f" in keys:
            return cls.ATTACK

        return cls.IDLE


class ActionEncoder(nn.Module):
    """Encodes discrete actions into continuous embeddings for DiT conditioning.

    The encoder uses:
    1. Learnable action embeddings
    2. MLP projection to match DiT hidden dimension
    3. Optional positional encoding for frame-level actions

    Actions are injected via cross-attention at specified DiT layers.
    """

    def __init__(
        self,
        num_actions: int = 17,
        embedding_dim: int = 512,
        hidden_dim: int = 4096,
        num_frames: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames

        # Action embedding table
        self.action_embedding = nn.Embedding(num_actions, embedding_dim)

        # Frame position encoding
        self.frame_pos_embedding = nn.Embedding(num_frames, embedding_dim)

        # MLP projection to match DiT hidden dimension
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Optional: Action combination for multi-action scenarios
        self.action_combiner = nn.Linear(embedding_dim * 2, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        nn.init.normal_(self.action_embedding.weight, std=0.02)
        nn.init.normal_(self.frame_pos_embedding.weight, std=0.02)

        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        actions: torch.Tensor,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode actions into conditioning embeddings.

        Args:
            actions: Action indices of shape (batch, num_frames) or (batch,)
            frame_indices: Optional frame indices for positional encoding

        Returns:
            Conditioning embeddings of shape (batch, num_frames, hidden_dim)
        """
        # Handle single action input
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)

        batch_size, num_frames = actions.shape
        device = actions.device

        # Get action embeddings
        action_emb = self.action_embedding(actions)  # (B, F, E)

        # Add frame positional encoding
        if frame_indices is None:
            frame_indices = torch.arange(num_frames, device=device)
            frame_indices = frame_indices.unsqueeze(0).expand(batch_size, -1)

        frame_pos_emb = self.frame_pos_embedding(frame_indices)  # (B, F, E)
        action_emb = action_emb + frame_pos_emb

        # Project to DiT hidden dimension
        conditioning = self.projection(action_emb)  # (B, F, H)

        return conditioning

    def encode_multi_action(
        self,
        primary_action: torch.Tensor,
        secondary_action: torch.Tensor,
    ) -> torch.Tensor:
        """Encode combination of two simultaneous actions.

        Useful for scenarios like "move forward while attacking".

        Args:
            primary_action: Primary action indices (batch,)
            secondary_action: Secondary action indices (batch,)

        Returns:
            Combined action embedding (batch, embedding_dim)
        """
        primary_emb = self.action_embedding(primary_action)
        secondary_emb = self.action_embedding(secondary_action)

        combined = torch.cat([primary_emb, secondary_emb], dim=-1)
        return self.action_combiner(combined)


class ActionInjectionLayer(nn.Module):
    """Cross-attention layer for injecting action conditioning into DiT.

    This layer is inserted at specified depths in the DiT transformer
    to condition generation on the action sequence.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Cross-attention components
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer norm for pre-norm architecture
        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_cond = nn.LayerNorm(hidden_dim)

        # Gating for gradual injection during training
        self.gate = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in [self.to_q, self.to_k, self.to_v]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)

        nn.init.xavier_uniform_(self.to_out[0].weight, gain=0.1)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(
        self,
        x: torch.Tensor,
        action_conditioning: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply action conditioning via cross-attention.

        Args:
            x: DiT hidden states (batch, seq_len, hidden_dim)
            action_conditioning: Action embeddings (batch, num_frames, hidden_dim)
            mask: Optional attention mask

        Returns:
            Conditioned hidden states (batch, seq_len, hidden_dim)
        """
        # Pre-norm
        x_norm = self.norm_x(x)
        cond_norm = self.norm_cond(action_conditioning)

        # Compute Q, K, V
        q = self.to_q(x_norm)
        k = self.to_k(cond_norm)
        v = self.to_v(cond_norm)

        # Reshape for multi-head attention
        batch_size = x.shape[0]
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        # Gated residual connection
        gate = torch.sigmoid(self.gate)
        return x + gate * out
