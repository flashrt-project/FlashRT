#!/usr/bin/env python3
"""MiniMax-Remover full-frame mask removal -- FlashRT NVFP4 quickstart.

End-to-end demo of the FlashRT-accelerated MiniMax-Remover video mask
removal pipeline. Every frame (plus its binary mask) is fed to the model
at once -- no segmentation, no bbox cropping. Mask pixels above the
threshold mark the region to inpaint; the rest of the frame is preserved.

This is a FlashRT optimization of the upstream project:
    https://github.com/zibojia/MiniMax-Remover

It wraps a loaded diffusers MiniMax-Remover pipeline with
``flash_rt.models.minimax_remover.MiniMaxRemoverPipeline``, which rewrites
the transformer denoise path onto NVFP4 (W4A4) GEMMs, fused fp32-stat
Triton norm/RoPE, kernel attention and a graph-capturable manual
flow-matching loop.

------------------------------------------------------------------
Test data
------------------------------------------------------------------
Sample frames + masks (numeric filenames, aligned by frame number):

    git clone https://github.com/chenping9999/object_removal_data.git

------------------------------------------------------------------
Model weights
------------------------------------------------------------------
Download the VAE / transformer / scheduler once:

    huggingface-cli download zibojia/minimax-remover \
        --include vae transformer scheduler --local-dir ./minimax-remover

------------------------------------------------------------------
Build
------------------------------------------------------------------
FlashRT must be built for Blackwell so the generic SM120 NVFP4 kernels
are compiled in (GPU_ARCH=120 / 121 auto-enables them):

    cmake -S . -B build -DGPU_ARCH=120 -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j --target flash_rt_kernels
    pip install -e ".[torch,minimax-remover]"

------------------------------------------------------------------
Run
------------------------------------------------------------------
    python3 examples/minimax_remover_quickstart.py \
        --model-dir ./minimax-remover \
        --frames-dir ./object_removal_data/<frames> \
        --masks-dir  ./object_removal_data/<masks> \
        --output-dir ./out

------------------------------------------------------------------
Precision note
------------------------------------------------------------------
NVFP4 (W4A4) is calibrated for the model's large GEMMs. The fused
norm/RoPE kernels keep the precision-critical path fp32-stat. For
large full-frame latents, 4-bit weight/activation quantisation can
accumulate small per-block error; if you observe colour drift on a
high-resolution full-frame job, prefer the bbox-cropped regime (crop
to the mask region before inference) where the quantisation grid is
tighter. The provided test data is sized for this path.

The VAE temporal compression factor is 4, so the input frame count
should be 4k+1 for a lossless round trip; trailing frames that the VAE
does not decode are copied from the source so the output frame count
always matches the input. Width/height not divisible by 16 are padded
bottom/right and cropped back on output.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Make the flash_rt package importable when run directly from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# diffusers model plumbing (vendored here so this demo depends only on pip
# packages -- no MiniMax-Remover source is imported).
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import AutoencoderKLWan
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import (PixArtAlphaTextProjection,
                                         TimestepEmbedding, Timesteps,
                                         get_1d_rotary_pos_embed)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange

DEVICE = torch.device("cuda:0")
MASK_THRESHOLD = 127
NUM_INFERENCE_STEPS = 12
PIPE_ITERATIONS = 6
RANDOM_SEED = 42


# =====================================================================
# Reference model definition (self-contained).
#
# The Transformer3DModel below mirrors the upstream MiniMax-Remover
# architecture verbatim so the released checkpoint loads without pulling
# in the upstream source tree. FlashRT then patches its forward path.
# =====================================================================

class AttnProcessor2_0:
    """Reference SDPA attention processor (replaced by FlashRT at runtime)."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0+.")

    def __call__(self, attn: Attention, hidden_states: torch.Tensor,
                 rotary_emb: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 encoder_hidden_states: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        encoder_hidden_states = hidden_states
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(
                    hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, time_freq_dim: int, time_proj_dim: int):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim,
                                        flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim,
                                               time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)

    def forward(self, timestep: torch.Tensor):
        timestep = self.timesteps_proj(timestep)
        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(self.time_proj.weight.data)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb, timestep_proj


class RotaryPosEmbed(nn.Module):
    def __init__(self, attention_head_dim: int, patch_size: Tuple[int, int, int],
                 max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False,
                repeat_interleave_real=False, freqs_dtype=torch.float64)
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
             self.attention_head_dim // 6,
             self.attention_head_dim // 6], dim=1)

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(
            1, 1, ppf * pph * ppw, -1)
        return freqs


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int,
                 qk_norm: str = "rms_norm_across_heads", cross_attn_norm: bool = False,
                 eps: float = 1e-6, added_kv_proj_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim, heads=num_heads, kv_heads=num_heads,
            dim_head=dim // num_heads, qk_norm=qk_norm, eps=eps, bias=True,
            cross_attention_dim=None, out_bias=True,
            processor=AttnProcessor2_0())
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor,
                rotary_emb: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()).chunk(6, dim=1)
        norm_hidden_states = (self.norm1(hidden_states.float())
                              * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = (hidden_states.float()
                         + attn_output * gate_msa).type_as(hidden_states)
        norm_hidden_states = (self.norm2(hidden_states.float())
                              * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float()
                         + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class Transformer3DModel(ModelMixin, ConfigMixin):
    """MiniMax-Remover Transformer (patch-embed + N adaLN blocks + proj_out).

    Loaded from the released checkpoint; its forward is then patched by
    the FlashRT pipeline onto NVFP4 GEMMs + fused kernels.
    """

    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["TransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2"]

    @register_to_config
    def __init__(self, patch_size: Tuple[int] = (1, 2, 2),
                 num_attention_heads: int = 40, attention_head_dim: int = 128,
                 in_channels: int = 16, out_channels: int = 16,
                 freq_dim: int = 256, ffn_dim: int = 13824, num_layers: int = 40,
                 cross_attn_norm: bool = True,
                 qk_norm: Optional[str] = "rms_norm_across_heads",
                 eps: float = 1e-6, added_kv_proj_dim: Optional[int] = None,
                 rope_max_seq_len: int = 1024) -> None:
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.rope = RotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        self.condition_embedder = TimeEmbedding(
            dim=inner_dim, time_freq_dim=freq_dim, time_proj_dim=inner_dim * 6)

        self.blocks = nn.ModuleList([
            TransformerBlock(inner_dim, ffn_dim, num_attention_heads, qk_norm,
                             cross_attn_norm, eps, added_kv_proj_dim)
            for _ in range(num_layers)])

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim ** 0.5)

    def forward(self, hidden_states: torch.Tensor,
                timestep: torch.LongTensor) -> Union[torch.Tensor, dict]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj = self.condition_embedder(timestep)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        for block in self.blocks:
            hidden_states = block(hidden_states, timestep_proj, rotary_emb)

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float())
                         * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
            p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return Transformer2DModelOutput(sample=output)


# =====================================================================
# Reference diffusers pipeline (self-contained).
#
# Constructs the pipe (transformer + vae + scheduler). Its __call__ is
# the reference flow-matching loop; FlashRT replaces it at runtime with
# the NVFP4 manual denoise pipeline.
# =====================================================================

class MinimaxRemoverPipeline(DiffusionPipeline):
    """diffusers pipeline wrapping the MiniMax-Remover transformer + VAE.

    Uses FlowMatchEulerDiscreteScheduler: the FlashRT manual denoise loop
    advances the latent with sigma-based Euler steps, which matches
    flow-matching semantics exactly.
    """

    model_cpu_offload_seq = "transformer->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(self, transformer: Transformer3DModel, vae: AutoencoderKLWan,
                 scheduler: FlowMatchEulerDiscreteScheduler):
        super().__init__()
        self.register_modules(vae=vae, transformer=transformer, scheduler=scheduler)
        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4)
        self.vae_scale_factor_spatial = (
            2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def expand_masks(self, masks, iterations):
        masks = masks.cpu().detach().numpy()
        masks2 = []
        for i in range(len(masks)):
            mask = masks[i]
            mask = mask > 0
            mask = scipy.ndimage.binary_dilation(mask, iterations=iterations)
            masks2.append(mask)
        masks = np.array(masks2).astype(np.float32)
        masks = torch.from_numpy(masks)
        masks = masks.repeat(1, 1, 1, 3)
        masks = rearrange(masks, "f h w c -> c f h w")
        masks = masks[None, ...]
        return masks

    def resize(self, images, w, h):
        bsz, _, _, _, _ = images.shape
        images = rearrange(images, "b c f w h -> (b f) c w h")
        images = F.interpolate(images, (w, h), mode="bilinear")
        images = rearrange(images, "(b f) c w h -> b c f w h", b=bsz)
        return images

    @torch.no_grad()
    def __call__(self, height: int = 720, width: int = 1280, num_frames: int = 81,
                 num_inference_steps: int = 50, generator=None,
                 images: Optional[torch.Tensor] = None,
                 masks: Optional[torch.Tensor] = None,
                 latents: Optional[torch.Tensor] = None,
                 output_type: Optional[str] = "np", iterations: int = 16):
        device = self._execution_device
        batch_size = 1
        transformer_dtype = torch.float16

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_channels_latents = 16
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        shape = (batch_size, num_channels_latents, num_latent_frames,
                 int(height) // self.vae_scale_factor_spatial,
                 int(width) // self.vae_scale_factor_spatial)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float16)

        masks = self.expand_masks(masks, iterations)
        masks = self.resize(masks, height, width).to("cuda:0").half()
        masks[masks > 0] = 1
        images = rearrange(images, "f h w c -> c f h w")
        images = self.resize(images[None, ...], height, width).to("cuda:0").half()
        masked_images = images * (1 - masks)

        latents_mean = (torch.tensor(self.vae.config.latents_mean)
                        .view(1, self.vae.config.z_dim, 1, 1, 1)
                        .to(self.vae.device, torch.float16))
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1).to(self.vae.device, torch.float16)

        with torch.no_grad():
            masked_latents = self.vae.encode(masked_images.half()).latent_dist.mode()
            masks_latents = self.vae.encode(2 * masks.half() - 1.0).latent_dist.mode()
            masked_latents = (masked_latents - latents_mean) * latents_std
            masks_latents = (masks_latents - latents_mean) * latents_std

        self._num_timesteps = len(timesteps)
        for i, t in enumerate(timesteps):
            latent_model_input = latents.to(transformer_dtype)
            latent_model_input = torch.cat(
                [latent_model_input, masked_latents, masks_latents], dim=1)
            timestep = t.expand(latents.shape[0])
            noise_pred = self.transformer(
                hidden_states=latent_model_input.half(), timestep=timestep)[0]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = latents.half() / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return WanPipelineOutput(frames=video)


def build_pipeline(model_dir: Path) -> MinimaxRemoverPipeline:
    """Load VAE / transformer / scheduler and assemble the diffusers pipe."""
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float16)
    transformer = Transformer3DModel.from_pretrained(
        model_dir / "transformer", torch_dtype=torch.float16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir / "scheduler")
    pipe = MinimaxRemoverPipeline(transformer=transformer, vae=vae, scheduler=scheduler)
    pipe.to(DEVICE)
    return pipe


# =====================================================================
# Frame / mask IO helpers (self-contained).
# =====================================================================

def collect_frame_files(frames_dir: Path) -> List[Path]:
    """Collect numeric-named image files, sorted by frame number."""
    supported = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    frame_files: Dict[int, Path] = {}
    for pattern in supported:
        for file in frames_dir.glob(pattern):
            try:
                frame_files.setdefault(int(file.stem), file)
            except ValueError:
                continue
    if not frame_files:
        raise ValueError(
            f"No frames found in {frames_dir} "
            "(numeric-named png/jpg/jpeg expected).")
    return [p for _, p in sorted(frame_files.items(), key=lambda x: x[0])]


def build_frame_path_map(paths: List[Path]) -> Dict[int, Path]:
    return {int(p.stem): p for p in paths}


def load_frames(image_paths: List[Path], num_workers: int = 8) -> np.ndarray:
    img0 = Image.open(image_paths[0]).convert("RGB")
    base_width, base_height = img0.size
    img0_np = np.array(img0, dtype=np.uint8)
    if len(image_paths) == 1:
        return np.stack([img0_np], axis=0)
    results: List[Optional[np.ndarray]] = [img0_np] + [None] * (len(image_paths) - 1)
    max_workers = max(1, min(num_workers, (os.cpu_count() or 4), len(image_paths)))

    def _load_one(idx, path):
        img = Image.open(path).convert("RGB")
        if img.size != (base_width, base_height):
            img = img.resize((base_width, base_height), Image.Resampling.BILINEAR)
        return idx, np.array(img, dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_load_one, idx, path)
                   for idx, path in enumerate(image_paths[1:], start=1)]
        for fut in as_completed(futures):
            idx, arr = fut.result()
            results[idx] = arr
    return np.stack([r for r in results if r is not None], axis=0)


def _load_single_mask(path: Optional[Path], orig_width: int,
                      orig_height: int) -> Tuple[np.ndarray, bool]:
    if path is None:
        return np.zeros((orig_height, orig_width, 1), dtype=np.float32), False
    img = Image.open(path).convert("L")
    if img.size != (orig_width, orig_height):
        img = img.resize((orig_width, orig_height), Image.Resampling.NEAREST)
    arr = np.array(img, dtype=np.uint8)
    mask = (arr > MASK_THRESHOLD).astype(np.float32)[..., None]
    return mask, bool(mask.any())


def load_masks(seg_paths: List[Path], mask_path_map: Dict[int, Path],
               orig_width: int, orig_height: int,
               num_workers: int = 8) -> Tuple[np.ndarray, List[bool]]:
    n = len(seg_paths)
    masks = np.zeros((n, orig_height, orig_width, 1), dtype=np.float32)
    has_region: List[bool] = [False] * n
    if n == 0:
        return masks, has_region
    masks[0], has_region[0] = _load_single_mask(
        mask_path_map.get(int(seg_paths[0].stem)), orig_width, orig_height)
    if n == 1:
        return masks, has_region
    max_workers = max(1, min(num_workers, (os.cpu_count() or 4), n))

    def _load_one(local_i, frame_no):
        mask, has = _load_single_mask(mask_path_map.get(frame_no), orig_width, orig_height)
        return local_i, mask, has

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_load_one, i, int(seg_paths[i].stem)) for i in range(1, n)]
        for fut in as_completed(futures):
            local_i, mask, has = fut.result()
            masks[local_i] = mask
            has_region[local_i] = has
    return masks, has_region


def pad_hw_to_multiple_of_16(height: int, width: int) -> Tuple[int, int, int, int]:
    ph = (16 - height % 16) % 16
    pw = (16 - width % 16) % 16
    return ph, pw, height + ph, width + pw


def pad_frames_bottom_right(frames: np.ndarray, ph: int, pw: int) -> np.ndarray:
    if ph == 0 and pw == 0:
        return frames
    return np.pad(frames, ((0, 0), (0, ph), (0, pw), (0, 0)), mode="edge")


def pad_masks_bottom_right(masks: np.ndarray, ph: int, pw: int) -> np.ndarray:
    if ph == 0 and pw == 0:
        return masks
    return np.pad(masks, ((0, 0), (0, ph), (0, pw), (0, 0)),
                  mode="constant", constant_values=0.0)


# =====================================================================
# Demo entry point.
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MiniMax-Remover full-frame mask removal (FlashRT NVFP4).")
    p.add_argument("--model-dir", type=str, required=True,
                   help="MiniMax-Remover checkpoint dir (vae/ transformer/ scheduler/).")
    p.add_argument("--frames-dir", type=str, required=True,
                   help="Input video frames dir (numeric filenames).")
    p.add_argument("--masks-dir", type=str, required=True,
                   help="Mask frames dir (numeric filenames, aligned with frames).")
    p.add_argument("--output-dir", type=str, required=True, help="Output frames dir.")
    p.add_argument("--iterations", type=int, default=PIPE_ITERATIONS,
                   help="Mask dilation iterations inside the pipeline (default 6).")
    p.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS,
                   help="Denoise steps (default 12).")
    p.add_argument("--no-flashrt", action="store_true",
                   help="Run the reference diffusers path instead of FlashRT (for diffing).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    if not frames_dir.is_dir():
        raise ValueError(f"frames dir does not exist: {frames_dir}")
    if not masks_dir.is_dir():
        raise ValueError(f"masks dir does not exist: {masks_dir}")
    if not (model_dir / "transformer").is_dir():
        raise ValueError(
            f"model dir missing transformer/: {model_dir}\n"
            "Download with: huggingface-cli download zibojia/minimax-remover "
            "--include vae transformer scheduler --local-dir <model-dir>")

    print("=" * 60)
    print("MiniMax-Remover full-frame mask removal (FlashRT NVFP4 W4A4)")
    print("=" * 60)

    image_paths = collect_frame_files(frames_dir)
    mask_paths = collect_frame_files(masks_dir)
    mask_path_map = build_frame_path_map(mask_paths)

    first = Image.open(image_paths[0]).convert("RGB")
    orig_w, orig_h = first.size
    first.close()

    ph, pw, pad_h, pad_w = pad_hw_to_multiple_of_16(orig_h, orig_w)
    print(f"  resolution: {orig_w}x{orig_h} -> padded {pad_w}x{pad_h} "
          f"(bottom +{ph}, right +{pw})")

    n = len(image_paths)
    print(f"  frames: {n}; steps={args.num_inference_steps}, iterations={args.iterations}")

    frames = load_frames(image_paths)
    masks, has_region = load_masks(image_paths, mask_path_map, orig_w, orig_h)

    if not any(has_region):
        print("  no removal region detected (masks empty); copying all source frames.")
        output_dir.mkdir(parents=True, exist_ok=True)
        for src in image_paths:
            shutil.copy2(src, output_dir / src.name)
        return

    frames_padded = pad_frames_bottom_right(frames, ph, pw)
    masks_padded = pad_masks_bottom_right(masks, ph, pw)

    pipe = build_pipeline(model_dir)

    if args.no_flashrt:
        runner = pipe
        tag = "reference (diffusers fp16)"
    else:
        from flash_rt.models.minimax_remover import MiniMaxRemoverPipeline
        runner = MiniMaxRemoverPipeline(pipe)
        tag = "FlashRT NVFP4 W4A4"

    t, h, w, _ = frames_padded.shape
    images_tensor = torch.from_numpy(frames_padded).to(device=DEVICE, dtype=torch.float16)
    images_tensor = images_tensor / 127.5 - 1.0
    masks_infer = torch.from_numpy(masks_padded.astype(np.float32))

    print(f"  running inference [{tag}] (all frames at once)...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    t0 = time.time()
    out = runner(
        images=images_tensor, masks=masks_infer, num_frames=t,
        height=h, width=w, num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device=DEVICE).manual_seed(RANDOM_SEED),
        iterations=args.iterations).frames[0]
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"  inference wall time: {elapsed:.2f}s")

    out_u8 = (np.array(out, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    proc_len = out_u8.shape[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = inpainted = uncovered = 0
    for local_i, path in enumerate(image_paths):
        dst = output_dir / path.name
        if local_i < proc_len and has_region[local_i]:
            crop = out_u8[local_i, :orig_h, :orig_w, :]
            Image.fromarray(crop, mode="RGB").save(dst)
            inpainted += 1
        elif local_i >= proc_len and has_region[local_i]:
            Image.fromarray(frames[local_i], mode="RGB").save(dst)
            uncovered += 1
        else:
            Image.fromarray(frames[local_i], mode="RGB").save(dst)
        saved += 1

    peak_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
    print(f"  saved {saved}/{n} frames to {output_dir} ({inpainted} inpainted)")
    if uncovered > 0:
        print(f"  note: {uncovered} trailing frame(s) kept from source "
              f"(VAE temporal factor 4; pad input to 4k+1 frames to avoid).")
    print(f"  peak VRAM: allocated {peak_alloc:.2f} GB / reserved {peak_reserved:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
