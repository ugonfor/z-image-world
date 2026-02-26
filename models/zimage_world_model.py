"""
Z-Image World Model

Wraps the pretrained Z-Image-Turbo transformer (6.15B S3-DiT) and extends it
with temporal attention and action conditioning for world model generation.

Architecture:
- Pretrained Z-Image transformer (frozen spatial attention)
- Inserted TemporalAttention layers (trainable, zero-init gamma)
- ActionInjectionLayer at specified depths (trainable, zero-init gate)

The model starts as a pure Z-Image image generator and gradually learns
temporal dynamics and action conditioning through training.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# Z-Image-Turbo architecture constants (from inspection)
ZIMAGE_HIDDEN_DIM = 3840
ZIMAGE_NUM_HEADS = 30
ZIMAGE_HEAD_DIM = 128
ZIMAGE_NUM_LAYERS = 30
ZIMAGE_NUM_NOISE_REFINER = 2
ZIMAGE_NUM_CONTEXT_REFINER = 2
ZIMAGE_TIMESTEP_DIM = 256
ZIMAGE_CAP_FEAT_DIM = 2560
ZIMAGE_PATCH_DIM = 64  # patch_size^2 * in_channels = 2^2 * 16


class TemporalAttention(nn.Module):
    """Causal temporal attention across frames.

    Applied after each Z-Image transformer block to add temporal coherence.
    Uses zero-initialized gamma so the model starts as pure Z-Image.
    """

    def __init__(
        self,
        hidden_dim: int = ZIMAGE_HIDDEN_DIM,
        num_heads: int = ZIMAGE_NUM_HEADS,
        max_frames: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm = nn.RMSNorm(hidden_dim)
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        # QK-Norm (matching Z-Image)
        self.norm_q = nn.RMSNorm(self.head_dim)
        self.norm_k = nn.RMSNorm(self.head_dim)

        self.dropout = nn.Dropout(dropout)

        # Zero-init gate: model starts as pure Z-Image
        self.gamma = nn.Parameter(torch.zeros(1))

        # Frame position embedding for temporal ordering
        self.frame_pos = nn.Embedding(max_frames, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.to_q.weight, gain=0.01)
        nn.init.xavier_uniform_(self.to_k.weight, gain=0.01)
        nn.init.xavier_uniform_(self.to_v.weight, gain=0.01)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)
        nn.init.normal_(self.frame_pos.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """Apply causal temporal attention.

        Args:
            x: Hidden states (batch * num_frames, num_patches, hidden_dim)
            num_frames: Number of frames in sequence

        Returns:
            Temporally attended hidden states, same shape as input
        """
        bf, n, d = x.shape
        batch_size = bf // num_frames

        # Reshape to (batch * num_patches, num_frames, hidden_dim)
        x_temporal = rearrange(x, "(b f) n d -> (b n) f d", b=batch_size, f=num_frames)

        # Add frame position
        frame_idx = torch.arange(num_frames, device=x.device)
        x_temporal = x_temporal + self.frame_pos(frame_idx).unsqueeze(0)

        # Normalize
        x_norm = self.norm(x_temporal)

        # QKV
        q = self.to_q(x_norm)
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)

        # Reshape for multi-head attention
        q = rearrange(q, "b f (h d) -> b h f d", h=self.num_heads)
        k = rearrange(k, "b f (h d) -> b h f d", h=self.num_heads)
        v = rearrange(v, "b f (h d) -> b h f d", h=self.num_heads)

        # QK-Norm
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Causal attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

        out = rearrange(out, "b h f d -> b f (h d)")
        out = self.to_out(out)

        # Reshape back
        out = rearrange(out, "(b n) f d -> (b f) n d", b=batch_size, n=n)

        return x + self.gamma * out


class ActionInjectionLayer(nn.Module):
    """Cross-attention for injecting action conditioning into Z-Image hidden states.

    Zero-initialized gate so the model starts without action influence.
    """

    def __init__(
        self,
        hidden_dim: int = ZIMAGE_HIDDEN_DIM,
        num_heads: int = ZIMAGE_NUM_HEADS,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm_x = nn.RMSNorm(hidden_dim)
        self.norm_cond = nn.RMSNorm(hidden_dim)

        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Zero-init gate
        self.gate = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for module in [self.to_q, self.to_k, self.to_v]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(
        self,
        x: torch.Tensor,
        action_conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Apply action conditioning via cross-attention.

        Args:
            x: Hidden states (batch, seq_len, hidden_dim)
            action_conditioning: Action embeddings (batch, num_actions, hidden_dim)

        Returns:
            Conditioned hidden states
        """
        x_norm = self.norm_x(x)
        cond_norm = self.norm_cond(action_conditioning)

        q = rearrange(self.to_q(x_norm), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.to_k(cond_norm), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.to_v(cond_norm), "b n (h d) -> b h n d", h=self.num_heads)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return x + torch.sigmoid(self.gate) * out


class ActionEncoder(nn.Module):
    """Encodes discrete actions for Z-Image world model conditioning.

    Projects 17 discrete actions into Z-Image's hidden space (3840-dim).
    """

    def __init__(
        self,
        num_actions: int = 17,
        embedding_dim: int = 512,
        hidden_dim: int = ZIMAGE_HIDDEN_DIM,
        max_frames: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.action_embedding = nn.Embedding(num_actions, embedding_dim)
        self.frame_pos_embedding = nn.Embedding(max_frames, embedding_dim)

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
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
            actions: Action indices (batch, num_frames) or (batch,)

        Returns:
            Conditioning embeddings (batch, num_frames, hidden_dim)
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)

        batch_size, num_frames = actions.shape
        device = actions.device

        action_emb = self.action_embedding(actions)

        if frame_indices is None:
            frame_indices = torch.arange(num_frames, device=device)
            frame_indices = frame_indices.unsqueeze(0).expand(batch_size, -1)

        action_emb = action_emb + self.frame_pos_embedding(frame_indices)

        return self.projection(action_emb)


class ZImageWorldModel(nn.Module):
    """World model built on top of pretrained Z-Image-Turbo.

    Wraps the pretrained Z-Image transformer and inserts:
    - TemporalAttention layers after each transformer block
    - ActionInjectionLayer at specified depths

    The pretrained spatial weights are frozen, while temporal and action
    layers are trainable. This follows the Vid2World transfer approach.
    """

    def __init__(
        self,
        transformer: nn.Module,
        vae: nn.Module,
        num_layers: int = ZIMAGE_NUM_LAYERS,
        hidden_dim: int = ZIMAGE_HIDDEN_DIM,
        num_heads: int = ZIMAGE_NUM_HEADS,
        max_frames: int = 16,
        action_injection_layers: list[int] | None = None,
        temporal_every_n: int = 1,
        freeze_spatial: bool = True,
    ):
        """Initialize ZImageWorldModel.

        Args:
            temporal_every_n: Apply temporal attention every N layers.
                1 = every layer (default, 1965M trainable params),
                2 = every other layer (~983M), 3 = every 3rd (~655M).
        """
        super().__init__()

        self.transformer = transformer
        self.vae = vae
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_frames = max_frames
        self.temporal_every_n = temporal_every_n

        if action_injection_layers is None:
            # Inject at 1/4, 1/2, 3/4 depth
            action_injection_layers = [
                num_layers // 4,
                num_layers // 2,
                3 * num_layers // 4,
            ]
        self.action_injection_layer_indices = set(action_injection_layers)

        # Temporal attention layers (at every Nth transformer block)
        self.temporal_layer_indices = set(range(0, num_layers, temporal_every_n))
        self.temporal_layers = nn.ModuleDict({
            str(i): TemporalAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                max_frames=max_frames,
            )
            for i in self.temporal_layer_indices
        })

        # Action injection layers (at specified depths)
        self.action_injections = nn.ModuleDict({
            str(i): ActionInjectionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            )
            for i in action_injection_layers
        })

        # Action encoder
        self.action_encoder = ActionEncoder(
            hidden_dim=hidden_dim,
            max_frames=max_frames,
        )

        # Freeze pretrained weights if requested
        if freeze_spatial:
            self._freeze_spatial()

        # Freeze VAE always
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    def _freeze_spatial(self):
        """Freeze all pretrained Z-Image transformer weights."""
        for p in self.transformer.parameters():
            p.requires_grad_(False)

    def unfreeze_spatial(self):
        """Unfreeze spatial weights for full fine-tuning."""
        for p in self.transformer.parameters():
            p.requires_grad_(True)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for temporal layers to save memory."""
        self._use_gradient_checkpointing = True
        if hasattr(self.transformer, "gradient_checkpointing_enable"):
            self.transformer.gradient_checkpointing_enable()

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only trainable parameters (temporal + action layers)."""
        params = []
        for p in self.temporal_layers.parameters():
            params.append(p)
        for p in self.action_injections.parameters():
            params.append(p)
        for p in self.action_encoder.parameters():
            params.append(p)
        return params

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space using frozen VAE.

        Args:
            images: (batch, num_frames, 3, H, W) in [0, 1]

        Returns:
            Latents (batch, num_frames, 16, H//8, W//8)
        """
        batch, num_frames, c, h, w = images.shape
        images_flat = rearrange(images, "b f c h w -> (b f) c h w")

        # Normalize to [-1, 1]
        images_flat = images_flat * 2.0 - 1.0

        # Encode
        posterior = self.vae.encode(images_flat)
        if hasattr(posterior, "latent_dist"):
            latents = posterior.latent_dist.sample()
        elif hasattr(posterior, "sample"):
            latents = posterior.sample()
        else:
            latents = posterior

        # Scale
        latents = latents * self.vae.config.scaling_factor

        return rearrange(latents, "(b f) c h w -> b f c h w", b=batch, f=num_frames)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images using frozen VAE.

        Args:
            latents: (batch, num_frames, 16, H//8, W//8)

        Returns:
            Images (batch, num_frames, 3, H, W) in [0, 1]
        """
        if latents.dim() == 5:
            batch, num_frames = latents.shape[:2]
            latents_flat = rearrange(latents, "b f c h w -> (b f) c h w")
        else:
            batch, num_frames = latents.shape[0], 1
            latents_flat = latents

        # Unscale
        latents_flat = latents_flat / self.vae.config.scaling_factor

        # Decode
        images = self.vae.decode(latents_flat)
        if hasattr(images, "sample"):
            images = images.sample

        # Normalize to [0, 1]
        images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)

        if num_frames > 1:
            images = rearrange(images, "(b f) c h w -> b f c h w", b=batch, f=num_frames)

        return images

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for diffusion denoising.

        This hooks into the Z-Image transformer's forward pass, intercepting
        after each transformer block to apply temporal attention and action
        conditioning.

        Args:
            latents: Noisy latents (batch, num_frames, 16, H//8, W//8)
            timesteps: Diffusion timesteps (batch,) or (batch, num_frames)
            actions: Action indices (batch, num_frames), optional

        Returns:
            Predicted noise (same shape as latents)
        """
        # Handle single frame
        if latents.dim() == 4:
            latents = latents.unsqueeze(1)
        batch_size, num_frames, channels, height, width = latents.shape

        # Encode actions
        action_cond = None
        if actions is not None:
            action_cond = self.action_encoder(actions)  # (B, F, hidden_dim)

        # Flatten frames into batch for Z-Image transformer
        latents_flat = rearrange(latents, "b f c h w -> (b f) c h w")

        # Handle per-frame timesteps
        if timesteps.dim() == 1:
            timesteps_flat = repeat(timesteps, "b -> (b f)", f=num_frames)
        else:
            timesteps_flat = rearrange(timesteps, "b f -> (b f)")

        # Run through Z-Image transformer with temporal injection
        noise_pred = self._forward_with_temporal(
            latents_flat,
            timesteps_flat,
            batch_size=batch_size,
            num_frames=num_frames,
            action_cond=action_cond,
            height=height,
            width=width,
        )

        # Reshape back
        noise_pred = rearrange(noise_pred, "(b f) c h w -> b f c h w", b=batch_size, f=num_frames)

        if num_frames == 1:
            noise_pred = noise_pred.squeeze(1)

        return noise_pred

    def _forward_with_temporal(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        batch_size: int,
        num_frames: int,
        action_cond: Optional[torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Run Z-Image transformer with temporal attention injected after each block.

        Replicates the Z-Image transformer's forward flow but adds temporal
        attention and action injection between the main transformer blocks.

        The Z-Image forward processes:
        1. Patchify + embed image tokens and caption tokens separately
        2. Noise refiner on image tokens
        3. Context refiner on caption tokens
        4. Concatenate into unified stream [image | caption]
        5. Main transformer blocks on unified stream
        6. Final layer + unpatchify
        """
        from torch.nn.utils.rnn import pad_sequence

        transformer = self.transformer
        bf = hidden_states.shape[0]  # batch_size * num_frames
        device = hidden_states.device
        patch_size = 2
        f_patch_size = 1

        # --- Timestep embedding ---
        t_scaled = timesteps * transformer.t_scale
        adaln_input = transformer.t_embedder(t_scaled)  # (B*F, 256)

        # --- Prepare inputs as lists (Z-Image expects List[Tensor]) ---
        # Each frame is one item in the list
        # Z-Image expects (C, F, H, W) per item where F=1 for images
        x_list = [img.unsqueeze(1) for img in hidden_states.unbind(0)]  # List of (C, 1, H, W)
        # Null caption: single zero-vector per frame
        cap_list = [
            torch.zeros(1, ZIMAGE_CAP_FEAT_DIM, device=device, dtype=hidden_states.dtype)
            for _ in range(bf)
        ]

        # --- Patchify and embed (reuse transformer's method) ---
        (
            x_patches, cap_feats,
            x_size, x_pos_ids, cap_pos_ids,
            x_inner_pad_mask, cap_inner_pad_mask,
        ) = transformer.patchify_and_embed(x_list, cap_list, patch_size, f_patch_size)

        # --- Image embed ---
        x_item_seqlens = [len(p) for p in x_patches]
        x_patches_cat = torch.cat(x_patches, dim=0)
        x_patches_cat = transformer.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_patches_cat)

        adaln_input = adaln_input.type_as(x_patches_cat)
        x_patches_cat[torch.cat(x_inner_pad_mask)] = transformer.x_pad_token
        x_patches_split = list(x_patches_cat.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(
            transformer.rope_embedder(torch.cat(x_pos_ids, dim=0))
            .split([len(p) for p in x_pos_ids], dim=0)
        )

        x_padded = pad_sequence(x_patches_split, batch_first=True, padding_value=0.0)
        x_freqs_padded = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_freqs_padded = x_freqs_padded[:, :x_padded.shape[1]]

        x_max_seqlen = max(x_item_seqlens)
        x_attn_mask = torch.zeros((bf, x_max_seqlen), dtype=torch.bool, device=device)
        for i, sl in enumerate(x_item_seqlens):
            x_attn_mask[i, :sl] = 1

        # --- Noise refiner ---
        for layer in transformer.noise_refiner:
            x_padded = layer(x_padded, x_attn_mask, x_freqs_padded, adaln_input)

        # --- Caption embed & refine ---
        cap_item_seqlens = [len(c) for c in cap_feats]
        cap_feats_cat = torch.cat(cap_feats, dim=0)
        cap_feats_cat = transformer.cap_embedder(cap_feats_cat)
        cap_feats_cat[torch.cat(cap_inner_pad_mask)] = transformer.cap_pad_token
        cap_feats_split = list(cap_feats_cat.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(
            transformer.rope_embedder(torch.cat(cap_pos_ids, dim=0))
            .split([len(p) for p in cap_pos_ids], dim=0)
        )

        cap_padded = pad_sequence(cap_feats_split, batch_first=True, padding_value=0.0)
        cap_freqs_padded = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_freqs_padded = cap_freqs_padded[:, :cap_padded.shape[1]]

        cap_max_seqlen = max(cap_item_seqlens)
        cap_attn_mask = torch.zeros((bf, cap_max_seqlen), dtype=torch.bool, device=device)
        for i, sl in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :sl] = 1

        for layer in transformer.context_refiner:
            cap_padded = layer(cap_padded, cap_attn_mask, cap_freqs_padded)

        # --- Build unified stream [image | caption] ---
        # Since all frames have same size/caption, we can build a single padded tensor
        unified_list = []
        unified_freqs_list = []
        for i in range(bf):
            xl = x_item_seqlens[i]
            cl = cap_item_seqlens[i]
            unified_list.append(torch.cat([x_padded[i, :xl], cap_padded[i, :cl]]))
            unified_freqs_list.append(torch.cat([x_freqs_padded[i, :xl], cap_freqs_padded[i, :cl]]))

        unified_seqlens = [x_item_seqlens[i] + cap_item_seqlens[i] for i in range(bf)]
        unified_max_seqlen = max(unified_seqlens)

        unified = pad_sequence(unified_list, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(unified_freqs_list, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bf, unified_max_seqlen), dtype=torch.bool, device=device)
        for i, sl in enumerate(unified_seqlens):
            unified_attn_mask[i, :sl] = 1

        # Image token count (same for all frames since same spatial size)
        # Assert all frames have the same number of image tokens
        assert all(s == x_item_seqlens[0] for s in x_item_seqlens), (
            f"All frames must have the same spatial size, got varying seqlens: {x_item_seqlens}"
        )
        img_len = x_item_seqlens[0]

        # --- Main transformer blocks with temporal + action injection ---
        for i, layer in enumerate(transformer.layers):
            # Z-Image spatial block
            unified = layer(unified, unified_attn_mask, unified_freqs, adaln_input)

            # Temporal attention (only on image tokens, at configured layers)
            if num_frames > 1 and str(i) in self.temporal_layers:
                img_tokens = unified[:, :img_len]  # (B*F, img_len, D)
                if getattr(self, "_use_gradient_checkpointing", False) and self.training:
                    img_tokens = torch.utils.checkpoint.checkpoint(
                        self.temporal_layers[str(i)], img_tokens, num_frames,
                        use_reentrant=False,
                    )
                else:
                    img_tokens = self.temporal_layers[str(i)](img_tokens, num_frames)
                unified = torch.cat([img_tokens.to(unified.dtype), unified[:, img_len:]], dim=1)

            # Action injection (per-frame: each frame sees only its own action)
            if str(i) in self.action_injections and action_cond is not None:
                img_tokens = unified[:, :img_len]
                # action_cond is (B, F, D). Reshape to (B*F, 1, D) so each
                # frame's image tokens cross-attend to only its own action.
                # This prevents future action leakage.
                action_per_frame = rearrange(action_cond, "b f d -> (b f) 1 d")
                img_tokens = self.action_injections[str(i)](img_tokens, action_per_frame)
                unified = torch.cat([img_tokens.to(unified.dtype), unified[:, img_len:]], dim=1)

        # --- Final layer ---
        final_layer = transformer.all_final_layer[f"{patch_size}-{f_patch_size}"]
        unified = final_layer(unified, adaln_input)

        # --- Extract image tokens and unpatchify ---
        unified_split = list(unified.unbind(dim=0))
        x_size_list = list(x_size) if isinstance(x_size, (list, tuple)) else x_size
        output = transformer.unpatchify(unified_split, x_size_list, patch_size, f_patch_size)

        # Stack back into tensor and squeeze the F=1 dimension from unpatchify
        # unpatchify returns List[(C, F=1, H, W)], we need (B*F, C, H, W)
        output = torch.stack(output, dim=0)  # (B*F, C, 1, H, W)
        output = output.squeeze(2)  # (B*F, C, H, W)

        return output

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype: torch.dtype = torch.bfloat16,
        action_injection_layers: list[int] | None = None,
        temporal_every_n: int = 1,
        freeze_spatial: bool = True,
        device: str = "cuda",
    ) -> "ZImageWorldModel":
        """Load pretrained Z-Image-Turbo and create world model.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            torch_dtype: Model precision
            action_injection_layers: Layers for action injection
            freeze_spatial: Whether to freeze pretrained weights
            device: Target device

        Returns:
            Initialized ZImageWorldModel
        """
        from diffusers import ZImagePipeline

        print(f"Loading Z-Image-Turbo from {model_name_or_path}...")
        pipe = ZImagePipeline.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        )

        transformer = pipe.transformer
        vae = pipe.vae

        # Count actual layers
        num_layers = len(transformer.layers)
        print(f"Transformer: {num_layers} layers, {sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B params")

        model = cls(
            transformer=transformer,
            vae=vae,
            num_layers=num_layers,
            action_injection_layers=action_injection_layers,
            temporal_every_n=temporal_every_n,
            freeze_spatial=freeze_spatial,
        )

        model = model.to(device=device, dtype=torch_dtype)

        trainable = model.num_trainable_params()
        total = model.num_total_params()
        print(f"World model: {total / 1e9:.2f}B total, {trainable / 1e6:.1f}M trainable")

        # Clean up pipeline (we only keep transformer + vae)
        del pipe
        torch.cuda.empty_cache()

        return model
