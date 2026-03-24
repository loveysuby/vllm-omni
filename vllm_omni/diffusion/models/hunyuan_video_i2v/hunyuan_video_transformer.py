# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.flux.flux_transformer import FeedForward

logger = init_logger(__name__)


class HunyuanVideoPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int, int] = 2,
        in_chans: int = 16,
        embed_dim: int = 3072,
    ) -> None:
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states


class HunyuanVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_size_t: int,
        rope_dim: list[int],
        theta: float = 256.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [
            num_frames // self.patch_size_t,
            height // self.patch_size,
            width // self.patch_size,
        ]

        axes_grids = []
        for i in range(3):
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")
        grid = torch.stack(grid, dim=0)

        freqs = []
        for i in range(3):
            freq_cis = get_1d_rotary_pos_embed(
                self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=False
            )
            freqs.append((freq_cis.real, freq_cis.imag))

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1).float()
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1).float()
        return freqs_cos, freqs_sin


class HunyuanVideoAdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: int | None = None) -> None:
        super().__init__()
        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        return gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)


class HunyuanVideoConditionEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        guidance_embeds: bool,
        image_condition_type: str | None = None,
    ):
        super().__init__()
        self.image_condition_type = image_condition_type
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")
        self.guidance_embedder = None
        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        pooled_projection: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)

        token_replace_emb = None
        if self.image_condition_type == "token_replace":
            tr_timestep = torch.zeros_like(timestep)
            tr_proj = self.time_proj(tr_timestep)
            token_replace_emb = self.timestep_embedder(tr_proj.to(dtype=pooled_projection.dtype))
            token_replace_emb = token_replace_emb + pooled_projections

        if self.guidance_embedder is not None and guidance is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
            conditioning = timesteps_emb + guidance_emb + pooled_projections
        else:
            conditioning = timesteps_emb + pooled_projections

        return conditioning, token_replace_emb


class HunyuanVideoTokenReplaceAdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        token_replace_emb: torch.Tensor,
        first_frame_num_tokens: int,
    ) -> tuple:
        emb = self.linear(self.silu(emb))
        token_replace_emb = self.linear(self.silu(token_replace_emb))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        tr_shift_msa, tr_scale_msa, tr_gate_msa, tr_shift_mlp, tr_scale_mlp, tr_gate_mlp = (
            token_replace_emb.chunk(6, dim=1)
        )

        norm_hidden_states = self.norm(hidden_states)
        hidden_states_zero = (
            norm_hidden_states[:, :first_frame_num_tokens] * (1 + tr_scale_msa[:, None]) + tr_shift_msa[:, None]
        )
        hidden_states_orig = (
            norm_hidden_states[:, first_frame_num_tokens:] * (1 + scale_msa[:, None]) + shift_msa[:, None]
        )
        hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)

        return (
            hidden_states,
            gate_msa, shift_mlp, scale_mlp, gate_mlp,
            tr_gate_msa, tr_shift_mlp, tr_scale_mlp, tr_gate_mlp,
        )


class HunyuanVideoTokenReplaceAdaLayerNormZeroSingle(nn.Module):
    """Adapted for [text, latent] concat order: applies token_replace to first-frame latents only."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        token_replace_emb: torch.Tensor,
        text_seq_len: int,
        first_frame_num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        token_replace_emb = self.linear(self.silu(token_replace_emb))

        shift, scale, gate = emb.chunk(3, dim=1)
        tr_shift, tr_scale, tr_gate = token_replace_emb.chunk(3, dim=1)

        normed = self.norm(hidden_states)
        ff_start = text_seq_len
        ff_end = text_seq_len + first_frame_num_tokens
        text_hs = normed[:, :ff_start] * (1 + scale[:, None]) + shift[:, None]
        first_frame_hs = normed[:, ff_start:ff_end] * (1 + tr_scale[:, None]) + tr_shift[:, None]
        rest_hs = normed[:, ff_end:] * (1 + scale[:, None]) + shift[:, None]
        hidden_states = torch.cat([text_hs, first_frame_hs, rest_hs], dim=1)

        return hidden_states, gate, tr_gate


class HunyuanVideoIndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)

        from diffusers.models.attention_processor import Attention as DiffusersAttention

        self.attn = DiffusersAttention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)

        from diffusers.models.attention import FeedForward as DiffusersFeedForward

        self.ff = DiffusersFeedForward(
            hidden_size, mult=mlp_width_ratio, activation_fn="linear-silu", dropout=mlp_drop_rate
        )

        self.norm_out = HunyuanVideoAdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp
        return hidden_states


class HunyuanVideoIndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                HunyuanVideoIndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size, seq_len = attention_mask.shape
            attention_mask = attention_mask.to(hidden_states.device).bool()
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).expand(-1, -1, seq_len, -1)
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)
        return hidden_states


class HunyuanVideoTokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanVideoIndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)
        return hidden_states


class HunyuanVideoDualAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 24,
        dim_head: int = 128,
        bias: bool = True,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-6,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            bias=bias,
        )

        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    self.inner_dim, self.out_dim, bias=out_bias,
                    input_is_parallel=True, return_bias=False,
                ),
                nn.Identity(),
            ]
        )

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=self.added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                bias=added_proj_bias,
            )

            self.to_add_out = RowParallelLinear(
                self.inner_dim, query_dim, bias=out_bias,
                input_is_parallel=True, return_bias=False,
            )

        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qkv, _ = self.to_qkv(hidden_states)
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (self.to_qkv.num_heads, -1))
        key = key.unflatten(-1, (self.to_qkv.num_kv_heads, -1))
        value = value.unflatten(-1, (self.to_qkv.num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos, sin = cos.to(query.dtype), sin.to(query.dtype)
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)

        if encoder_hidden_states is not None:
            encoder_qkv, _ = self.add_kv_proj(encoder_hidden_states)
            add_q_size = self.add_kv_proj.num_heads * self.head_dim
            add_kv_size = self.add_kv_proj.num_kv_heads * self.head_dim
            enc_q, enc_k, enc_v = encoder_qkv.split([add_q_size, add_kv_size, add_kv_size], dim=-1)

            enc_q = enc_q.unflatten(-1, (self.add_kv_proj.num_heads, -1))
            enc_k = enc_k.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))
            enc_v = enc_v.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))

            enc_q = self.norm_added_q(enc_q)
            enc_k = self.norm_added_k(enc_k)

            query = torch.cat([query, enc_q], dim=1)
            key = torch.cat([key, enc_k], dim=1)
            value = torch.cat([value, enc_v], dim=1)

        attn_metadata = None
        if attention_mask is not None:
            seq_len = query.shape[1]
            attention_mask = F.pad(
                attention_mask, (seq_len - attention_mask.shape[1], 0), value=True
            ).bool()
            attn_metadata = AttentionMetadata(attn_mask=attention_mask)

        hidden_states = self.attn(query, key, value, attn_metadata)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
                [hidden_states.shape[1] - encoder_hidden_states.shape[1],
                 encoder_hidden_states.shape[1]], dim=1,
            )
            hidden_states = self.to_out[0](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states


class HunyuanVideoSingleAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 24,
        dim_head: int = 128,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.heads = heads

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            bias=bias,
        )

        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_seq_len: int,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        qkv, _ = self.to_qkv(hidden_states)
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (self.to_qkv.num_heads, -1))
        key = key.unflatten(-1, (self.to_qkv.num_kv_heads, -1))
        value = value.unflatten(-1, (self.to_qkv.num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos, sin = cos.to(query.dtype), sin.to(query.dtype)
            query[:, text_seq_len:] = self.rope(query[:, text_seq_len:], cos, sin)
            key[:, text_seq_len:] = self.rope(key[:, text_seq_len:], cos, sin)

        hidden_states = self.attn(query, key, value, None)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        return hidden_states


class HunyuanVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = HunyuanVideoDualAttention(
            query_dim=hidden_size,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=hidden_size, dim_out=hidden_size, mult=mlp_ratio)

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=hidden_size, dim_out=hidden_size, mult=mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hs, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_enc, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        attn_out, ctx_attn_out = self.attn(
            hidden_states=norm_hs,
            encoder_hidden_states=norm_enc,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        hidden_states = hidden_states + attn_out * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + ctx_attn_out * c_gate_msa.unsqueeze(1)

        norm_hs = self.norm2(hidden_states)
        norm_enc = self.norm2_context(encoder_hidden_states)

        norm_hs = norm_hs * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_enc = norm_enc * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_hs)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_enc)

        return hidden_states, encoder_hidden_states


class HunyuanVideoSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = ReplicatedLinear(
            dim, self.mlp_hidden_dim, bias=True, return_bias=False,
        )
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = ReplicatedLinear(
            dim + self.mlp_hidden_dim, dim, bias=True, return_bias=False,
        )

        self.attn = HunyuanVideoSingleAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            bias=True,
            eps=1e-6,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            norm_hidden_states, text_seq_len=text_seq_len, image_rotary_emb=image_rotary_emb,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return hidden_states, encoder_hidden_states


class HunyuanVideoTokenReplaceTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = HunyuanVideoTokenReplaceAdaLayerNormZero(hidden_size)
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = HunyuanVideoDualAttention(
            query_dim=hidden_size,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=hidden_size, dim_out=hidden_size, mult=mlp_ratio)

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=hidden_size, dim_out=hidden_size, mult=mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
        token_replace_emb: torch.Tensor | None = None,
        first_frame_num_tokens: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = first_frame_num_tokens

        (
            norm_hs,
            gate_msa, shift_mlp, scale_mlp, gate_mlp,
            tr_gate_msa, tr_shift_mlp, tr_scale_mlp, tr_gate_mlp,
        ) = self.norm1(hidden_states, temb, token_replace_emb, n)
        norm_enc, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        attn_out, ctx_attn_out = self.attn(
            hidden_states=norm_hs,
            encoder_hidden_states=norm_enc,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        hidden_states_zero = hidden_states[:, :n] + attn_out[:, :n] * tr_gate_msa.unsqueeze(1)
        hidden_states_orig = hidden_states[:, n:] + attn_out[:, n:] * gate_msa.unsqueeze(1)
        hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
        encoder_hidden_states = encoder_hidden_states + ctx_attn_out * c_gate_msa.unsqueeze(1)

        norm_hs = self.norm2(hidden_states)
        norm_enc = self.norm2_context(encoder_hidden_states)

        hs_zero = norm_hs[:, :n] * (1 + tr_scale_mlp[:, None]) + tr_shift_mlp[:, None]
        hs_orig = norm_hs[:, n:] * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_hs = torch.cat([hs_zero, hs_orig], dim=1)
        norm_enc = norm_enc * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        ff_output = self.ff(norm_hs)
        ctx_ff_output = self.ff_context(norm_enc)

        hidden_states_zero = hidden_states[:, :n] + ff_output[:, :n] * tr_gate_mlp.unsqueeze(1)
        hidden_states_orig = hidden_states[:, n:] + ff_output[:, n:] * gate_mlp.unsqueeze(1)
        hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * ctx_ff_output

        return hidden_states, encoder_hidden_states


class HunyuanVideoTokenReplaceSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = HunyuanVideoTokenReplaceAdaLayerNormZeroSingle(dim)
        self.proj_mlp = ReplicatedLinear(
            dim, self.mlp_hidden_dim, bias=True, return_bias=False,
        )
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = ReplicatedLinear(
            dim + self.mlp_hidden_dim, dim, bias=True, return_bias=False,
        )

        self.attn = HunyuanVideoSingleAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            bias=True,
            eps=1e-6,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        token_replace_emb: torch.Tensor | None = None,
        first_frame_num_tokens: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate, tr_gate = self.norm(
            hidden_states, temb, token_replace_emb, text_seq_len, first_frame_num_tokens,
        )
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            norm_hidden_states, text_seq_len=text_seq_len, image_rotary_emb=image_rotary_emb,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        proj_output = self.proj_out(hidden_states)

        ff_start = text_seq_len
        ff_end = text_seq_len + first_frame_num_tokens
        text_out = proj_output[:, :ff_start] * gate.unsqueeze(1)
        first_frame_out = proj_output[:, ff_start:ff_end] * tr_gate.unsqueeze(1)
        rest_out = proj_output[:, ff_end:] * gate.unsqueeze(1)
        hidden_states = torch.cat([text_out, first_frame_out, rest_out], dim=1)
        hidden_states = residual + hidden_states

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformer3DModel(nn.Module):
    _repeated_blocks = [
        "HunyuanVideoTransformerBlock",
        "HunyuanVideoSingleTransformerBlock",
        "HunyuanVideoTokenReplaceTransformerBlock",
        "HunyuanVideoTokenReplaceSingleTransformerBlock",
    ]
    _layerwise_offload_blocks_attr = "transformer_blocks"
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
    }

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return ("transformer_blocks" in name or "single_transformer_blocks" in name) and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: tuple[int, ...] = (16, 56, 56),
        image_condition_type: str | None = "token_replace",
    ):
        super().__init__()

        model_config = od_config.tf_model_config
        num_layers = getattr(model_config, "num_layers", num_layers)
        num_single_layers = getattr(model_config, "num_single_layers", num_single_layers)

        self.parallel_config = od_config.parallel_config
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.image_condition_type = image_condition_type
        self.guidance_embeds = guidance_embeds

        self.x_embedder = HunyuanVideoPatchEmbed(
            (patch_size_t, patch_size, patch_size), in_channels, inner_dim,
        )
        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers,
        )

        if image_condition_type == "token_replace":
            self.time_text_embed = HunyuanVideoConditionEmbedding(
                embedding_dim=inner_dim,
                pooled_projection_dim=pooled_projection_dim,
                guidance_embeds=guidance_embeds,
                image_condition_type=image_condition_type,
            )
        elif guidance_embeds:
            self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
                embedding_dim=inner_dim, pooled_projection_dim=pooled_projection_dim,
            )
        else:
            self.time_text_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=inner_dim, pooled_projection_dim=pooled_projection_dim,
            )

        self.rope = HunyuanVideoRotaryPosEmbed(patch_size, patch_size_t, list(rope_axes_dim), rope_theta)

        if image_condition_type == "token_replace":
            self.transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoTokenReplaceTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.single_transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoTokenReplaceSingleTransformerBlock(
                        dim=inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(num_single_layers)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.single_transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoSingleTransformerBlock(
                        dim=inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(num_single_layers)
                ]
            )

        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * self.out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size_t, self.patch_size, self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        image_rotary_emb = self.rope(hidden_states)

        if self.image_condition_type == "token_replace":
            temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)
            first_frame_num_tokens = post_patch_height * post_patch_width
        else:
            if self.guidance_embeds and guidance is not None:
                temb = self.time_text_embed(timestep, guidance, pooled_projections)
            else:
                temb = self.time_text_embed(timestep, pooled_projections)
            token_replace_emb = None
            first_frame_num_tokens = 0

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states, timestep, encoder_attention_mask,
        )

        if self.image_condition_type == "token_replace":
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb,
                    encoder_attention_mask, image_rotary_emb,
                    token_replace_emb, first_frame_num_tokens,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, image_rotary_emb,
                    token_replace_emb, first_frame_num_tokens,
                )
        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb,
                    encoder_attention_mask, image_rotary_emb,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, image_rotary_emb,
                )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p_h, p_w,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                if lookup_name not in params_dict:
                    break
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                if lookup_name not in params_dict:
                    continue
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)
        return loaded_params
