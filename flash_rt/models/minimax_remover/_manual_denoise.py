"""FlashRT -- MiniMax-Remover manual pointer-based denoise pipeline.

Replaces the diffusers ``Minimax_Remover_Pipeline.__call__`` denoise
loop with a pointer-based, graph-capturable implementation that mirrors
the FlashRT frontend pattern:

1. Pre-compute ALL step time embeddings BEFORE graph capture (the
   condition embedder's ``Timesteps`` uses ``torch.arange`` -- a CPU op
   that breaks capture).
2. Pre-compute ALL modulation vectors (``scale_shift_table + temb``) for
   every step x every block + norm_out, eliminating all ``.float()``
   casts, additions and chunk ops from the hot path.
3. Pre-compute RoPE freqs (fixed for a given latent shape).
4. Pre-compute the Euler flow-matching ``dt`` values.
5. Manual block forward (``block_forward_fused``): only kernel calls --
   Triton adaLN, kernel attention, NVFP4 GEMM, Triton gate-mul, FlashRT
   gelu. QKV quantises the norm output ONCE and reuses it for all three
   projections; the FFN up GEMM fuses bias + gelu straight to FP4 output
   so the FFN-down projection skips re-quantisation.
6. Manual norm_out: Triton adaLN (replaces FP32LayerNorm + mul + add).
7. CUDA Graph captures the ENTIRE N-step x N-block denoise loop as ONE
   graph (``FLASHRT_MANUAL_GRAPH=1``).
8. Euler step via the fp32-accumulating Triton kernel.
9. VAE encode / decode stay outside the graph (one-shot per segment).

Inside the captured graph there are ZERO torch elementwise ops -- every
operation is a kernel launch.

No MiniMax-Remover imports: the loaded diffusers ``pipe`` is duck-typed
through ``pipe.transformer`` / ``pipe.vae`` / ``pipe.scheduler``. The
``flash_rt_kernels`` module is passed in (``kern``) so importing this
module never requires the compiled extension.
"""

import math
import os

import torch
from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor

from ._kernels import (ada_layernorm_fp16_io, gate_mul_residual_bcast,
                       rms_norm_fp32stat, rope_apply_bshd, freqs_to_cos_sin,
                       euler_step_inplace, mask_mul,
                       latent_normalize, latent_denormalize)
from ._attention import attention_forward, _attention_mode

_USE_FUSED_BLOCK = os.environ.get("FLASHRT_FUSED_BLOCK", "1") == "1"


def _fp4_gemm_raw(a_packed, a_sfa, linear, m, device, kern):
    """FP4 GEMM from a pre-quantised packed input (no re-quantisation)."""
    k, n = linear.in_features, linear.out_features
    out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    stream = torch.cuda.current_stream().cuda_stream
    linear._gemm(
        a_packed.data_ptr(), linear.weight_packed.data_ptr(), out.data_ptr(),
        m, n, k, a_sfa.data_ptr(), linear.weight_sfb.data_ptr(),
        linear.weight_alpha, stream)
    if linear.bias is not None:
        kern.add_bias_bf16(out.data_ptr(), linear.bias.data_ptr(), m, n, stream)
    return out


def _quant_to_nvfp4(x_bf16, kern):
    """Quantise bf16 [S, D] -> (packed [S, D/2], swizzled SF) for FP4 GEMM."""
    m, k = x_bf16.shape
    packed = torch.empty(m, k // 2, dtype=torch.uint8, device=x_bf16.device)
    sfa = torch.empty(kern.nvfp4_sf_swizzled_bytes(m, k), dtype=torch.uint8,
                      device=x_bf16.device)
    stream = torch.cuda.current_stream().cuda_stream
    kern.quantize_bf16_to_nvfp4_swizzled(
        x_bf16.data_ptr(), packed.data_ptr(), sfa.data_ptr(), m, k, stream)
    return packed, sfa


def block_forward_fused(block, hidden_states, mod, rotary_emb, eps, kern,
                        cos_sin=None):
    """Block forward with pre-computed fp32 modulation; fused QKV + FFN.

    ``mod`` is [6, D] fp32: (shift_msa, scale_msa, gate_msa,
    shift_ffn, scale_ffn, gate_ffn). Only kernel calls remain.
    """
    _, S, D = hidden_states.shape
    hs = hidden_states.contiguous().view(S, D)
    stream = torch.cuda.current_stream().cuda_stream
    device = hs.device

    # ── Self-attention sub-layer ──
    norm1 = ada_layernorm_fp16_io(hs, mod[1], mod[0], eps)
    packed1, sfa1 = _quant_to_nvfp4(norm1, kern)

    attn = block.attn1
    H = attn.heads
    Dd = D // H
    q = _fp4_gemm_raw(packed1, sfa1, attn.to_q, S, device, kern)
    k = _fp4_gemm_raw(packed1, sfa1, attn.to_k, S, device, kern)
    v = _fp4_gemm_raw(packed1, sfa1, attn.to_v, S, device, kern)

    if attn.norm_q is not None:
        q = rms_norm_fp32stat(q, attn.norm_q.weight, attn.norm_q.eps)
    if attn.norm_k is not None:
        k = rms_norm_fp32stat(k, attn.norm_k.weight, attn.norm_k.eps)

    q = q.view(1, S, H, Dd)
    k = k.view(1, S, H, Dd)
    v = v.view(1, S, H, Dd)

    if rotary_emb is not None:
        if cos_sin is None:
            cos_sin = freqs_to_cos_sin(rotary_emb)
        rope_apply_bshd(q, cos_sin[0], cos_sin[1])
        rope_apply_bshd(k, cos_sin[0], cos_sin[1])

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    scale = 1.0 / math.sqrt(float(Dd))
    out = attention_forward(q, k, v, scale, _attention_mode())
    attn_out = attn.to_out[0](out.view(1, S, D)).view(S, D)
    gate_mul_residual_bcast(hs, attn_out, mod[2])

    # ── FFN sub-layer ──
    norm2 = ada_layernorm_fp16_io(hs, mod[4], mod[3], eps)
    packed2, sfa2 = _quant_to_nvfp4(norm2, kern)

    ffn_up = block.ffn.net[0].proj
    n_inner = ffn_up.out_features
    up_packed = torch.empty(S, n_inner // 2, dtype=torch.uint8, device=device)
    up_sfd = torch.empty(kern.nvfp4_sf_swizzled_bytes(S, n_inner),
                         dtype=torch.uint8, device=device)
    bias_ptr = ffn_up.bias.data_ptr() if ffn_up.bias is not None else 0
    kern.fp4_w4a16_gemm_bias_gelu_fp4out_sm120(
        packed2.data_ptr(), ffn_up.weight_packed.data_ptr(),
        sfa2.data_ptr(), ffn_up.weight_sfb.data_ptr(),
        bias_ptr, up_packed.data_ptr(), up_sfd.data_ptr(),
        S, n_inner, D, ffn_up.weight_alpha, stream)

    ff_out = _fp4_gemm_raw(up_packed, up_sfd, block.ffn.net[2], S, device, kern)
    gate_mul_residual_bcast(hs, ff_out, mod[5])

    return hs.view(1, S, D)


def block_forward_mod(block, hidden_states, mod, rotary_emb, eps, kern,
                      cos_sin=None):
    """Non-fused block forward (debug fallback): per-projection re-quant."""
    _, S, D = hidden_states.shape
    hs = hidden_states.contiguous().view(S, D)

    norm1 = ada_layernorm_fp16_io(hs, mod[1], mod[0], eps)
    attn_out = block.attn1(
        hidden_states=norm1.view(1, S, D), rotary_emb=rotary_emb).view(S, D)
    gate_mul_residual_bcast(hs, attn_out, mod[2])

    norm2 = ada_layernorm_fp16_io(hs, mod[4], mod[3], eps)
    up = block.ffn.net[0].proj(norm2.view(1, S, D))
    inner = up.shape[-1]
    stream = torch.cuda.current_stream().cuda_stream
    _gelu_fn = kern.gelu_inplace if up.dtype == torch.bfloat16 else kern.gelu_inplace_fp16
    _gelu_fn(up.data_ptr(), S * inner, stream)
    ff_out = block.ffn.net[2](up).view(S, D)
    gate_mul_residual_bcast(hs, ff_out, mod[5])

    return hs.view(1, S, D)


def transformer_forward_mod(transformer, hidden_states, mod_blocks_step,
                            mod_out_step, rotary_emb, eps, kern):
    """Transformer forward with pre-computed modulation; no torch elementwise."""
    B, C_in, T, H, W = hidden_states.shape
    p_t, p_h, p_w = transformer.config.patch_size
    post_t, post_h, post_w = T // p_t, H // p_h, W // p_w

    hs = transformer.patch_embedding(hidden_states)
    hs = hs.flatten(2).transpose(1, 2)

    cos_sin = freqs_to_cos_sin(rotary_emb) if rotary_emb is not None else None
    block_fn = block_forward_fused if _USE_FUSED_BLOCK else block_forward_mod

    for blk_idx, block in enumerate(transformer.blocks):
        hs = block_fn(block, hs, mod_blocks_step[blk_idx], rotary_emb, eps, kern, cos_sin)

    S, D = hs.shape[1], hs.shape[2]
    hs_2d = hs.contiguous().view(S, D)
    hs_2d = ada_layernorm_fp16_io(hs_2d, mod_out_step[1], mod_out_step[0], eps)
    hs = transformer.proj_out(hs_2d.view(1, S, D))

    hs = hs.reshape(B, post_t, post_h, post_w, p_t, p_h, p_w, -1)
    hs = hs.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hs.flatten(6, 7).flatten(4, 5).flatten(2, 3)


class ManualRemoverPipeline:
    """Pointer-based, graph-capturable replacement for ``pipe.__call__``.

    Captures the full N-step denoise loop as one CUDA Graph (when
    ``FLASHRT_MANUAL_GRAPH=1``) and replays it on subsequent calls with
    the same latent shape. Inside the captured graph: zero torch
    elementwise ops -- every operation is a kernel launch.

    The class reads everything it needs off the loaded ``pipe`` (its
    transformer, vae, scheduler, video processor, expand_masks / resize
    helpers and VAE scale factors); it imports no MiniMax-Remover code.
    ``kern`` is the validated ``flash_rt_kernels`` module.
    """

    def __init__(self, pipe, kern):
        self.kern = kern
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler
        self.device = pipe._execution_device
        self.video_processor = pipe.video_processor
        self.eps = float(pipe.transformer.config.eps)
        self.num_blocks = len(pipe.transformer.blocks)
        self.inner_dim = (pipe.transformer.config.num_attention_heads *
                          pipe.transformer.config.attention_head_dim)
        self._dtype = next(pipe.transformer.parameters()).dtype
        self._vae_dtype = next(pipe.vae.parameters()).dtype

        self._graphs = {}
        self._mod_cache = None
        self._rope_cache = {}

    @property
    def vae_scale_factor_temporal(self):
        return self.pipe.vae_scale_factor_temporal

    @property
    def vae_scale_factor_spatial(self):
        return self.pipe.vae_scale_factor_spatial

    def expand_masks(self, masks, iterations):
        return self.pipe.expand_masks(masks, iterations)

    def resize(self, images, w, h):
        return self.pipe.resize(images, w, h)

    def _precompute_modulation(self, temb_all, tproj_all, num_steps):
        """Pre-compute all modulation vectors for steps x blocks + norm_out."""
        D = self.inner_dim
        nb = self.num_blocks
        tr = self.transformer
        device = self.device

        mod_blocks = torch.empty(num_steps, nb, 6, D, dtype=torch.float32, device=device)
        mod_out = torch.empty(num_steps, 2, D, dtype=torch.float32, device=device)

        with torch.no_grad():
            for step in range(num_steps):
                tproj = tproj_all[step].float()
                for blk_idx, block in enumerate(tr.blocks):
                    mod_blocks[step, blk_idx] = (block.scale_shift_table + tproj).squeeze(0)
                temb = temb_all[step].float().unsqueeze(1)
                mod_out[step] = (tr.scale_shift_table + temb).squeeze(0)

        self._mod_cache = (num_steps, mod_blocks, mod_out)
        return mod_blocks, mod_out

    def __call__(self, images, masks, num_frames, height, width,
                 num_inference_steps=12, generator=None, iterations=16,
                 output_type="np"):
        device = self.device
        kern = self.kern

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas
        dt_all = [float(sigmas[i + 1] - sigmas[i]) for i in range(num_inference_steps)]

        vae_t = self.vae_scale_factor_temporal
        vae_s = self.vae_scale_factor_spatial
        num_latent_frames = (num_frames - 1) // vae_t + 1
        lat_shape = (1, 16, num_latent_frames, height // vae_s, width // vae_s)
        latents = randn_tensor(lat_shape, generator=generator,
                               device=device, dtype=self._dtype)

        masks_t = self.expand_masks(masks, iterations)
        masks_t = self.resize(masks_t, height, width).to(device).to(self._vae_dtype)
        masks_t[masks_t > 0] = 1
        images_t = rearrange(images, "f h w c -> c f h w")
        images_t = self.resize(images_t[None, ...], height, width).to(device).to(self._vae_dtype)
        masked_images = mask_mul(images_t, masks_t)

        latents_mean = (torch.tensor(self.vae.config.latents_mean)
                        .view(1, self.vae.config.z_dim, 1, 1, 1)
                        .to(device, self._vae_dtype))
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1).to(device, self._vae_dtype)

        with torch.no_grad():
            masked_latents = self.vae.encode(masked_images.to(self._vae_dtype)).latent_dist.mode()
            masks_latents = self.vae.encode((2 * masks_t - 1.0).to(self._vae_dtype)).latent_dist.mode()
            masked_latents = latent_normalize(masked_latents, latents_mean, latents_std).to(self._dtype)
            masks_latents = latent_normalize(masks_latents, latents_mean, latents_std).to(self._dtype)

        if self._mod_cache is None or self._mod_cache[0] != num_inference_steps:
            with torch.no_grad():
                temb_all, tproj_all = [], []
                for i in range(num_inference_steps):
                    t_step = timesteps[i:i + 1]
                    temb_i, tproj_i = self.transformer.condition_embedder(t_step)
                    tproj_i = tproj_i.unflatten(1, (6, -1))
                    temb_all.append(temb_i)
                    tproj_all.append(tproj_i)
            mod_blocks, mod_out = self._precompute_modulation(
                temb_all, tproj_all, num_inference_steps)
        else:
            mod_blocks, mod_out = self._mod_cache[1], self._mod_cache[2]

        rope_key = tuple(latents.shape)
        rotary_emb = self._rope_cache.get(rope_key)
        if rotary_emb is None:
            with torch.no_grad():
                rotary_emb = self.transformer.rope(latents)
            self._rope_cache[rope_key] = rotary_emb

        use_graph = os.environ.get("FLASHRT_MANUAL_GRAPH", "0") == "1"
        shape_key = tuple(latents.shape)
        entry = self._graphs.get(shape_key)

        if use_graph:
            if entry is None:
                entry = self._capture_graph(
                    latents, masked_latents, masks_latents,
                    mod_blocks, mod_out, dt_all, rotary_emb, num_inference_steps)
                self._graphs[shape_key] = entry
            graph, lat_buf, masked_buf, masks_buf = entry
            lat_buf.copy_(latents)
            masked_buf.copy_(masked_latents)
            masks_buf.copy_(masks_latents)
            graph.replay()
            torch.cuda.synchronize()
            result_latents = lat_buf.clone()
        else:
            result_latents = self._denoise_eager(
                latents, masked_latents, masks_latents,
                mod_blocks, mod_out, dt_all, rotary_emb, num_inference_steps)

        result_latents = latent_denormalize(result_latents.to(self._vae_dtype),
                                            latents_mean, latents_std)
        with torch.no_grad():
            video = self.vae.decode(result_latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
        return WanPipelineOutput(frames=video)

    def _denoise_eager(self, latents, masked_latents, masks_latents,
                       mod_blocks, mod_out, dt_all, rotary_emb, num_steps):
        device = latents.device
        kern = self.kern
        C = latents.shape[1]
        concat_buf = torch.empty(latents.shape[0], 3 * C, *latents.shape[2:],
                                 dtype=self._dtype, device=device)
        lat_buf = latents.clone()
        tr = self.transformer
        eps = self.eps

        for step in range(num_steps):
            concat_buf[:, :C].copy_(lat_buf)
            concat_buf[:, C:2 * C].copy_(masked_latents)
            concat_buf[:, 2 * C:3 * C].copy_(masks_latents)
            noise_pred = transformer_forward_mod(
                tr, concat_buf, mod_blocks[step], mod_out[step], rotary_emb, eps, kern)
            euler_step_inplace(lat_buf, noise_pred, dt_all[step])

        return lat_buf

    def _capture_graph(self, latents, masked_latents, masks_latents,
                       mod_blocks, mod_out, dt_all, rotary_emb, num_steps):
        """Capture the entire N-step denoise loop as one CUDA Graph."""
        device = latents.device
        kern = self.kern
        C = latents.shape[1]
        B, _, T, H, W = latents.shape

        lat_buf = latents.clone()
        masked_buf = masked_latents.clone()
        masks_buf = masks_latents.clone()
        concat_buf = torch.empty(B, 3 * C, T, H, W, dtype=self._dtype, device=device)

        tr = self.transformer
        eps = self.eps

        def denoise():
            for step in range(num_steps):
                concat_buf[:, :C].copy_(lat_buf)
                concat_buf[:, C:2 * C].copy_(masked_buf)
                concat_buf[:, 2 * C:3 * C].copy_(masks_buf)
                noise_pred = transformer_forward_mod(
                    tr, concat_buf, mod_blocks[step], mod_out[step], rotary_emb, eps, kern)
                euler_step_inplace(lat_buf, noise_pred, dt_all[step])

        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            concat_buf[:, :C].copy_(lat_buf)
            concat_buf[:, C:2 * C].copy_(masked_buf)
            concat_buf[:, 2 * C:3 * C].copy_(masks_buf)
            noise_pred = transformer_forward_mod(
                tr, concat_buf, mod_blocks[0], mod_out[0], rotary_emb, eps, kern)
            euler_step_inplace(lat_buf, noise_pred, dt_all[0])
            torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        lat_buf.copy_(latents)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            denoise()

        return (graph, lat_buf, masked_buf, masks_buf)
