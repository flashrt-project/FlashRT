"""FlashRT — Motus pipeline for RTX (sm120, RTX 5090) — G2.

Orchestrates a single Motus inference step over 30 MoT layers, exposing
every distinct compute primitive as its own method so G3 can swap each
in turn (PyTorch BF16 → fvk BF16 → fvk FP8). At G2 the methods delegate
to the loaded ``Motus`` upstream module; this gives us a structurally
correct FlashRT path with cos = 1.0 vs the baseline_artifacts reference.

Companion frontend: ``flash_rt/frontends/torch/motus_rtx.py``.

Reference baseline (PyTorch BF16, 2026-05-07, baseline_artifacts/):
    P50 = 1226.9 ms, GPU peak 24.8 GB, cos = 1.000000.

────────────────────────────────────────────────────────────────────
Per-call flow (one ``infer()``):
────────────────────────────────────────────────────────────────────

    set_prompt(...) was already called with pre-encoded T5 + VLM inputs.

    1. encode_first_frame(image)           # Wan VAE encode (1 frame)
    2. preprocess_t5_context(t5_embeds)    # Wan text_embedding (4096->3072)
    3. extract_und_tokens(vlm_inputs)      # Qwen3-VL forward → 512-dim
    4. init_noisy_latents(...)             # randn noise + cond frame inject
    5. for step in range(num_inference_steps):  # 10 steps
         t, t_next, dt = ...
         video_tokens = video_module.prepare_input(video_latent)
         action_tokens = action_expert.input_encoder(...)
         und_tokens   = re-extract  (G5: cache once outside loop)
         video_t_emb, video_adaln = get_time_embeddings(t, ...)
         action_t_emb, action_adaln = ...
         for L in range(30):
             v_mod, a_mod = compute_modulations(L)
             video, action, und = denoise_layer_joint_attn(...)
             video = denoise_layer_cross_attn_to_t5(...)
             video = denoise_layer_ffn_video(...)
             action = denoise_layer_ffn_action(...)
             und   = denoise_layer_ffn_und(...)
         video_velocity, action_velocity = output_heads(...)
         video_latent  = euler_step(video_latent, video_velocity, dt)
         video_latent  = teacher_force_first_frame(video_latent, cond_latent)
         action_latent = euler_step(action_latent, action_velocity, dt)
    6. predicted_frames = decode_video(video_latent)   # Wan VAE decode
    7. predicted_actions = action_latent

Full-E2E contract: Motus optimization targets the upstream
``inference_step`` semantics from the upstream Motus implementation and the paper's
Video-Action Joint Prediction path. The steady-state output is always
``(predicted_frames, predicted_actions)``. VAE decode is part of the
runtime contract and must be optimized, not skipped.

────────────────────────────────────────────────────────────────────
G2 status / G3 refactor plan
────────────────────────────────────────────────────────────────────

G2 implements every method by DELEGATING to the loaded ``Motus`` module
(its own VideoModule / ActionModule / UndModule helpers). This gives us:
    * cos = 1.0 vs baseline (literally same numerics)
    * the right method shape for G3 to surgically replace each step

G3 substitutes (in order):
    G3a  Norms             rms_norm/AdaLN -> fvk.rms_norm_fp16 / ada_layer_norm_fp16
    G3b  GEMMs             nn.Linear      -> fvk.bf16_nn / gemm_fp16
    G3c  Attention         sdpa fallback  -> fvk.attention_mha_fp16 via attn.run
    G3d  FFN gate-act      F.silu*x * y  -> fvk.silu_mul_split_fp16 (or geglu, audit Wan)
    G3e  RoPE              rope_apply    -> fvk.qkv_split_rope_kvcache_fp16
    G3f  Patch embed       3D conv       -> fvk.patch_im2col + patch_embed_bias_pos

G4 then layers FP8 weight + activation quantization on top of G3.
G5 captures the inner 30-layer loop into a CUDA Graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, List, Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────
# Static geometry (sourced from Stage-2 ckpt config + Wan2.2-5B).
# Verified against baseline_artifacts/meta.json.
# ─────────────────────────────────────────────────────────────────
NUM_LAYERS = 30
WAN_DIM = 3072
WAN_NUM_HEADS = 24
WAN_HEAD_DIM = 128
ACTION_DIM = 1024
ACTION_NUM_HEADS = 8
UND_DIM = 512
UND_NUM_HEADS = 4
T5_CTX_LEN = 512
T5_CTX_DIM = 4096


@dataclass
class MotusPipelineDims:
    """Runtime dim record. Frontend fills this in once weights are loaded."""
    num_layers: int = NUM_LAYERS
    num_inference_steps: int = 10
    num_video_frames: int = 8
    video_height: int = 384
    video_width: int = 320
    action_chunk_size: int = 8
    action_dim: int = 14
    state_dim: int = 14
    vae_channels: int = 48
    is_pretrain: bool = True       # Stage-2 ckpt geometry (no state, no reg)
    num_registers: int = 0


_STATIC_MOD_FP8_STAGE_BY_SHAPE: dict[Tuple[int, int, int],
                                     Tuple[torch.Tensor, ...]] = {}


@dataclass
class _StaticModFp8:
    q: Tuple[torch.Tensor, ...]
    scale: Tuple[torch.Tensor, ...]
    shape: Tuple[int, int, int]

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            import flash_rt.flash_rt_kernels as fvk
            from flash_rt.models.motus._stream import cs
        except Exception as exc:
            raise RuntimeError("FP8 static modulation cache requires "
                               "flash_rt_kernels") from exc
        stage = _STATIC_MOD_FP8_STAGE_BY_SHAPE.get(self.shape)
        if stage is None or stage[0].device != self.q[0].device:
            stage = tuple(
                torch.empty(self.shape, dtype=torch.bfloat16,
                            device=self.q[0].device)
                for _ in range(6))
            _STATIC_MOD_FP8_STAGE_BY_SHAPE[self.shape] = stage
        out = stage[idx]
        fvk.dequantize_fp8_static_bf16(
            int(self.q[idx].data_ptr()), int(out.data_ptr()),
            int(self.scale[idx].data_ptr()), int(out.numel()), cs())
        return out


class MotusPipelineRtx:
    """Single-step + multi-step orchestration of Motus denoising on RTX.

    Constructed by ``MotusTorchFrontendRtx``. Holds:
      * ``model``      the loaded upstream ``Motus`` instance (G2 source
                       of all numerical truth; G3 will rip out gradually)
      * ``attn``       the AttentionBackend (G2 stub; not yet wired into
                       the kernel calls — kept for shape declarations and
                       so G3 can substitute fvk attention)
      * ``dims``       runtime dims
      * ``dtype``      compute dtype (bfloat16 for the whole G2 family)
      * ``device``     cuda
    """

    def __init__(
        self,
        model,                       # upstream Motus nn.Module
        attn_backend,                # AttentionBackend stub
        dims: MotusPipelineDims,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model
        self.attn = attn_backend
        self.dims = dims
        self.dtype = dtype
        self.device = device
        self._static_v_mod_stage = None
        self._static_a_mod_stage = None
        self._cached_dt_host = None
        self._cached_v_head_shift = None
        self._cached_v_head_scale = None
        self._cached_a_head_shift = None
        self._cached_a_head_scale = None
        self._current_step_idx = None
        self._ffn_multistream = (
            os.environ.get("FLASH_RT_MOTUS_FFN_MULTI_STREAM", "0") == "1")
        self._ffn_action_stream = None
        self._ffn_und_stream = None

    # ──────────────────────────────────────────────────────────────
    # One-shot stages (outside the denoise loop)
    # ──────────────────────────────────────────────────────────────

    def _ensure_ffn_streams(self):
        if not self._ffn_multistream:
            return None, None
        if self._ffn_action_stream is None:
            self._ffn_action_stream = torch.cuda.Stream(device=self.device)
            self._ffn_und_stream = torch.cuda.Stream(device=self.device)
        return self._ffn_action_stream, self._ffn_und_stream

    @torch.no_grad()
    def encode_first_frame(self, first_frame: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] in [0, 1] → cond_latent [B, 48, 1, H', W'] bf16.

        G3a candidate: VAE encoder is small and one-shot per call;
        likely stays PyTorch even at G6 (not in hot path).
        """
        x = first_frame.to(self.device).to(self.dtype)
        x = (x * 2.0 - 1.0).unsqueeze(2)                # [B, C, 1, H, W]
        return self.model.video_model.encode_video(x)

    def preprocess_t5_context(
        self, t5_embeds: List[torch.Tensor]
    ) -> torch.Tensor:
        """List of [seq, 4096] → padded ctx then projected to 3072.

        G5: returns cached version if ``precompute_t5_context()`` was
        called. Pinning a stable address is important for CUDA Graph
        capture (cross-attn KV reads from this buffer).
        """
        cached = getattr(self, "_cached_t5_ctx", None)
        if cached is not None:
            return cached
        return self.model.video_module.preprocess_t5_embeddings(t5_embeds)

    def extract_und_tokens(self, vlm_inputs: List[dict]) -> torch.Tensor:
        """VLM (frozen Qwen3-VL-2B) forward → adapted und tokens.

        G5: if ``self._cached_und_tokens`` is set (precomputed via
        ``precompute_und_tokens()``), return that cached device tensor
        instead of re-running the VLM. The VLM has CPU-side ops
        (torch.linspace) that fail inside torch.cuda.graph() capture,
        so we MUST precompute outside the graph.
        """
        cached = getattr(self, "_cached_und_tokens", None)
        if cached is not None:
            return cached
        return self.model.und_module.extract_und_features(vlm_inputs)

    def precompute_und_tokens(self, vlm_inputs: List[dict]) -> torch.Tensor:
        """Run VLM once and cache the und tokens on the pipeline.

        Called by the frontend before CUDA Graph capture. Subsequent
        calls to ``extract_und_tokens`` short-circuit to the cache.
        """
        with torch.no_grad():
            self._cached_und_tokens = (
                self.model.und_module.extract_und_features(vlm_inputs)
            )
        return self._cached_und_tokens

    def precompute_t5_context(self, t5_embeds) -> torch.Tensor:
        """Run T5 text_embedding once and cache the projected ctx.

        Same rationale as ``precompute_und_tokens`` — the projection
        itself is graph-safe but pinning a stable device buffer
        ensures the captured pipeline reads from a fixed address.
        """
        with torch.no_grad():
            self._cached_t5_ctx = (
                self.model.video_module.preprocess_t5_embeddings(t5_embeds)
            )
        return self._cached_t5_ctx

    def precompute_time_embeddings(
        self,
        seq_len_video: int,
        seq_len_action: int,
        num_inference_steps: int,
        batch: int = 1,
    ) -> None:
        """Run get_video/action_time_embeddings for ALL N steps and cache.

        The upstream Wan get_time_embedding / sinusoidal_embedding_1d
        chain calls torch.arange(...) — a CPU-side op forbidden inside
        torch.cuda.graph() capture. By computing all 10 timesteps'
        embeddings in advance, the captured forward only does indexed
        reads from the cached tensors.

        Cached layout (one entry per step):
            self._cached_v_t_emb[step]:  [batch, S_v, 3072]
            self._cached_v_adaln[step]:  [batch, S_v, 6, 3072]
            self._cached_a_t_emb[step]:  [batch, S_a, 1024]
            self._cached_a_adaln[step]:  [batch, S_a, 6, 1024]
        """
        timesteps = torch.linspace(
            1.0, 0.0, num_inference_steps + 1,
            device=self.device, dtype=self.dtype)
        v_t_embs, v_adalns = [], []
        a_t_embs, a_adalns = [], []
        with torch.no_grad():
            for i in range(num_inference_steps):
                t = timesteps[i]
                vt = (t * 1000).expand(batch).to(self.dtype)
                at = (t * 1000).expand(batch).to(self.dtype)
                v_t, v_a = self.model.video_module.get_time_embedding(
                    vt, seq_len_video)
                a_t, a_a = self.model.action_module.get_time_embedding(
                    at, seq_len_action)
                v_t_embs.append(v_t)
                v_adalns.append(v_a)
                a_t_embs.append(a_t)
                a_adalns.append(a_a)
        # Stack along step axis for indexed reads inside the graph.
        self._cached_v_t_emb = torch.stack(v_t_embs, dim=0)
        self._cached_v_adaln = torch.stack(v_adalns, dim=0)
        self._cached_a_t_emb = torch.stack(a_t_embs, dim=0)
        self._cached_a_adaln = torch.stack(a_adalns, dim=0)
        self._cached_time_steps = timesteps
        self._cached_dt_host = [
            float((timesteps[i + 1] - timesteps[i]).item())
            for i in range(num_inference_steps)
        ]
        self._precompute_output_head_modulations()
        self._precompute_adaln_modulations()

    def _precompute_output_head_modulations(self) -> None:
        self._cached_v_head_shift = None
        self._cached_v_head_scale = None
        self._cached_a_head_shift = None
        self._cached_a_head_scale = None
        if os.environ.get("FLASH_RT_MOTUS_NO_G7_50_HEAD_ADALN", "0") == "1":
            return
        if self._cached_v_t_emb is None or self._cached_a_t_emb is None:
            return
        try:
            v_mod = self.model.video_model.wan_model.head.modulation
            a_mod = self.model.action_expert.decoder.modulation
        except Exception:
            return

        def make_pair(t_emb: torch.Tensor, modulation: torch.Tensor):
            if t_emb.dim() != 4 and t_emb.dim() != 3:
                return None
            if t_emb.dim() == 4:
                if int(t_emb.shape[1]) != 1:
                    return None
                base = t_emb[:, 0, :, :].float()
            else:
                if int(t_emb.shape[1]) != 1:
                    return None
                base = t_emb[:, 0:1, :].float()
            m = modulation.detach().float()
            if m.dim() == 3:
                m = m[0]
            if m.shape[0] != 2 or m.shape[-1] != base.shape[-1]:
                return None
            shift = (base + m[0].view(1, 1, -1)).to(torch.bfloat16).contiguous()
            scale = (base + m[1].view(1, 1, -1)).to(torch.bfloat16).contiguous()
            if shift.shape[1] == 1:
                shift = shift[:, 0, :].contiguous()
                scale = scale[:, 0, :].contiguous()
            return shift, scale

        v_pair = make_pair(self._cached_v_t_emb, v_mod)
        a_pair = make_pair(self._cached_a_t_emb, a_mod)
        if v_pair is not None:
            self._cached_v_head_shift, self._cached_v_head_scale = v_pair
        if a_pair is not None:
            self._cached_a_head_shift, self._cached_a_head_scale = a_pair

    def _precompute_adaln_modulations(self) -> None:
        self._cached_v_mod = None
        self._cached_a_mod = None
        self._static_v_mod_stage = None
        self._static_a_mod_stage = None
        _STATIC_MOD_FP8_STAGE_BY_SHAPE.clear()
        if os.environ.get("FLASH_RT_MOTUS_NO_G7_38_STATIC_MOD", "0") == "1":
            return
        if self._cached_v_adaln is None or self._cached_a_adaln is None:
            return
        try:
            import flash_rt.flash_rt_kernels as fvk
            from flash_rt.models.motus._stream import cs
        except Exception:
            return
        if not hasattr(fvk, "adaln_modulation6_bf16"):
            return
        skip_last_n = int(os.environ.get(
            "FLASH_RT_MOTUS_STATIC_MOD_SKIP_LAST_N", "0") or "0")
        fp8_last_n = int(os.environ.get(
            "FLASH_RT_MOTUS_STATIC_MOD_FP8_LAST_N", "6") or "0")
        use_fp8_cache = (
            os.environ.get("FLASH_RT_MOTUS_NO_STATIC_MOD_FP8", "0") != "1"
            and os.environ.get("FLASH_RT_MOTUS_STATIC_MOD_FP8", "1") == "1"
            and fp8_last_n != 0)
        num_layers = int(self.dims.num_layers)
        skip_layers = set()
        if skip_last_n > 0:
            first_skip = max(0, num_layers - skip_last_n)
            skip_layers.update(range(first_skip, num_layers))
        self._cached_mod_partial = bool(skip_layers)

        def pack_fp8_mod(outs: Tuple[torch.Tensor, ...]) -> _StaticModFp8:
            q_chunks = []
            scales = []
            n = int(outs[0].numel())
            for out in outs:
                # Precompute happens outside graph capture; the captured hot
                # path only sees dequantize_fp8_static_bf16 kernel launches.
                amax = float(out.float().abs().amax().item())
                scale_value = max(amax / 448.0, 1e-12)
                scale = torch.tensor([scale_value], dtype=torch.float32,
                                     device=out.device)
                q = torch.empty(out.shape, dtype=torch.float8_e4m3fn,
                                device=out.device)
                fvk.quantize_fp8_static(
                    int(out.data_ptr()), int(q.data_ptr()),
                    int(scale.data_ptr()), n, cs())
                q_chunks.append(q)
                scales.append(scale)
            return _StaticModFp8(tuple(q_chunks), tuple(scales),
                                 tuple(int(x) for x in outs[0].shape))

        def make_mods(adaln_steps, layers, allow_skip: bool):
            mods = []
            for s in range(int(adaln_steps.shape[0])):
                step_mods = []
                p = adaln_steps[s]
                if p.dtype != torch.float32:
                    p = p.float()
                p = p if p.is_contiguous() else p.contiguous()
                B, S, K6, D = p.shape
                if K6 != 6:
                    raise RuntimeError(f"bad adaln shape {tuple(p.shape)}")
                for layer_idx, layer in enumerate(layers):
                    if allow_skip and layer_idx in skip_layers:
                        step_mods.append(None)
                        continue
                    m = layer.modulation
                    if m.dtype != torch.float32:
                        m = m.float()
                    m = m if m.is_contiguous() else m.contiguous()
                    outs = tuple(
                        torch.empty(B, S, D, dtype=torch.bfloat16,
                                    device=p.device)
                        for _ in range(6))
                    fvk.adaln_modulation6_bf16(
                        int(p.data_ptr()), int(m.data_ptr()),
                        int(outs[0].data_ptr()), int(outs[1].data_ptr()),
                        int(outs[2].data_ptr()), int(outs[3].data_ptr()),
                        int(outs[4].data_ptr()), int(outs[5].data_ptr()),
                        int(B), int(S), int(D), cs())
                    use_layer_fp8 = (
                        use_fp8_cache
                        and (fp8_last_n <= 0
                             or layer_idx >= max(0, num_layers - fp8_last_n))
                    )
                    if use_layer_fp8:
                        step_mods.append(pack_fp8_mod(outs))
                        del outs
                    else:
                        step_mods.append(outs)
                mods.append(step_mods)
            torch.cuda.synchronize()
            return mods

        self._cached_v_mod = make_mods(
            self._cached_v_adaln,
            self.model.video_model.wan_model.blocks,
            allow_skip=True)
        self._cached_a_mod = make_mods(
            self._cached_a_adaln,
            self.model.action_expert.blocks,
            allow_skip=False)
        # The per-layer BF16 modulation tables are now the graph inputs.
        # Drop the larger FP32 adaln stacks to keep enough headroom for
        # CUDA graph private pools and VAE decode allocations.
        if not self._cached_mod_partial:
            self._cached_v_adaln = None
            self._cached_a_adaln = None
            # Full static modulation is close to the 5090 memory limit.
            # Return the now-dead AdaLN staging blocks before graph capture;
            # otherwise the allocator can keep them reserved and VAE decode
            # may fail while capturing its large conv workspaces.
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def _materialize_static_mod(self, mod: tuple, kind: str) -> tuple:
        if not isinstance(mod, _StaticModFp8):
            return mod
        try:
            import flash_rt.flash_rt_kernels as fvk
            from flash_rt.models.motus._stream import cs
        except Exception as exc:
            raise RuntimeError("FP8 static modulation cache requires "
                               "flash_rt_kernels") from exc
        attr = "_static_v_mod_stage" if kind == "video" else "_static_a_mod_stage"
        stage = getattr(self, attr, None)
        if stage is None or tuple(stage[0].shape) != mod.shape:
            stage = tuple(
                torch.empty(mod.shape, dtype=torch.bfloat16,
                            device=mod.q[0].device)
                for _ in range(6))
            setattr(self, attr, stage)
        n = int(stage[0].numel())
        if hasattr(fvk, "dequantize_fp8_static_bf16_6"):
            fvk.dequantize_fp8_static_bf16_6(
                int(mod.q[0].data_ptr()), int(mod.q[1].data_ptr()),
                int(mod.q[2].data_ptr()), int(mod.q[3].data_ptr()),
                int(mod.q[4].data_ptr()), int(mod.q[5].data_ptr()),
                int(stage[0].data_ptr()), int(stage[1].data_ptr()),
                int(stage[2].data_ptr()), int(stage[3].data_ptr()),
                int(stage[4].data_ptr()), int(stage[5].data_ptr()),
                int(mod.scale[0].data_ptr()), int(mod.scale[1].data_ptr()),
                int(mod.scale[2].data_ptr()), int(mod.scale[3].data_ptr()),
                int(mod.scale[4].data_ptr()), int(mod.scale[5].data_ptr()),
                n, cs())
        else:
            for q, scale, out in zip(mod.q, mod.scale, stage):
                fvk.dequantize_fp8_static_bf16(
                    int(q.data_ptr()), int(out.data_ptr()),
                    int(scale.data_ptr()), n, cs())
        return stage

    def _maybe_materialize_static_mod(self, mod: tuple, kind: str) -> tuple:
        if (kind == "video"
                and os.environ.get("FLASH_RT_MOTUS_STATIC_MOD_FP8_BAKED",
                                   "1") == "1"
                and isinstance(mod, _StaticModFp8)):
            return mod
        return self._materialize_static_mod(mod, kind)

    def clear_caches(self) -> None:
        """Drop precomputed und/t5/time caches (e.g. after set_prompt change)."""
        self._cached_und_tokens = None
        self._cached_t5_ctx = None
        self._cached_v_t_emb = None
        self._cached_v_adaln = None
        self._cached_a_t_emb = None
        self._cached_a_adaln = None
        self._cached_time_steps = None
        self._cached_dt_host = None
        self._cached_v_mod = None
        self._cached_a_mod = None
        self._cached_mod_partial = False
        self._static_v_mod_stage = None
        self._static_a_mod_stage = None
        self._cached_v_head_shift = None
        self._cached_v_head_scale = None
        self._cached_a_head_shift = None
        self._cached_a_head_scale = None
        self._current_step_idx = None
        _STATIC_MOD_FP8_STAGE_BY_SHAPE.clear()

    @torch.no_grad()
    def decode_video(self, video_latent: torch.Tensor) -> torch.Tensor:
        """[B, 48, T_l, H', W'] → [B, 3, T, H, W] in [0, 1].

        Drops the cond frame slot (index 0), matching upstream Motus
        ``inference_step``. This decode is mandatory for the full-E2E
        FlashRT target; action-only shortcuts are not an acceptable
        latency path for Motus delivery.
        """
        decoded = self.model.video_model.decode_video(video_latent)
        if (
            os.environ.get("FLASH_RT_MOTUS_USE_FVK_DECODE_POST", "1") == "1"
            and decoded.is_cuda
            and decoded.dtype == torch.bfloat16
            and decoded.dim() == 5
            and decoded.shape[2] > 1
        ):
            try:
                import flash_rt.flash_rt_kernels as fvk
                from flash_rt.models.motus._stream import cs
                B, C, T, H, W = (int(v) for v in decoded.shape)
                out_f32 = torch.empty(
                    (B, C, T - 1, H, W),
                    dtype=torch.float32, device=decoded.device)
                fvk.motus_decode_postprocess_bf16_to_fp32(
                    int(decoded.data_ptr()), int(out_f32.data_ptr()),
                    B, C, T, H, W, cs())
                return out_f32
            except Exception:
                pass
        out = decoded[:, :, 1:]                          # drop cond frame
        out = ((out + 1.0) / 2.0).clamp(0, 1)
        return out.float()

    # ──────────────────────────────────────────────────────────────
    # Per-step token preparation
    # ──────────────────────────────────────────────────────────────

    def init_video_latent(self, cond_latent: torch.Tensor) -> torch.Tensor:
        """Allocate noise [B, 48, T_l, H', W'] and inject cond frame at slot 0."""
        B, C, _, H, W = cond_latent.shape
        T_l = 1 + self.dims.num_video_frames // 4
        video_latent = torch.randn(
            (B, C, T_l, H, W), device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = cond_latent
        return video_latent

    def init_action_latent(self, B: int) -> torch.Tensor:
        return torch.randn(
            (B, self.dims.action_chunk_size, self.dims.action_dim),
            device=self.device, dtype=self.dtype)

    def prepare_video_tokens(self, video_latent: torch.Tensor) -> torch.Tensor:
        """Wan patch_embedding: [B, 48, T_l, H', W'] → [B, S_v, 3072].

        G3f target: replace 3D conv + flatten/transpose with
        ``fvk.patch_im2col`` + ``patch_embed_bias_pos`` fusion (~1 ms
        savings, but also gets us a contiguous fp8 buffer for G4).
        """
        if (
            os.environ.get("FLASH_RT_MOTUS_USE_G7_45_VIDEO_TOKEN_CONTIG", "1") == "1"
            and os.environ.get("FLASH_RT_MOTUS_USE_G7_45_FVK_TRANSPOSE", "1") == "1"
        ):
            latent = video_latent.to(self.dtype)
            patched = self.model.video_module.video_model.wan_model.patch_embedding(latent)
            if patched.dtype == torch.bfloat16:
                try:
                    import flash_rt.flash_rt_kernels as fvk
                    from flash_rt.models.motus._stream import cs
                    B, C, T, H, W = (int(v) for v in patched.shape)
                    tokens = torch.empty(
                        (B, T * H * W, C), dtype=torch.bfloat16,
                        device=patched.device)
                    if hasattr(fvk, "ncdhw_to_blc_bf16"):
                        fvk.ncdhw_to_blc_bf16(
                            int(patched.data_ptr()), int(tokens.data_ptr()),
                            B, C, T, H, W, cs())
                        return tokens
                except Exception:
                    pass
        tokens = self.model.video_module.prepare_input(video_latent.to(self.dtype))
        if (
            os.environ.get("FLASH_RT_MOTUS_USE_G7_45_VIDEO_TOKEN_CONTIG", "1") == "1"
            and not tokens.is_contiguous()
        ):
            tokens = tokens.contiguous()
        return tokens

    def prepare_action_tokens(
        self,
        action_latent: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Action expert input_encoder. Pretrain mode (Stage-2 ckpt):
        no state token, no register tokens — pos_embedding length = 8.

        ActionEncoder.forward(state, action, registers) signature
        accepts state=None and ignores it.
        """
        if self.dims.num_registers > 0 and self.model.action_expert.registers is not None:
            B = action_latent.shape[0]
            registers = self.model.action_expert.registers.expand(B, -1, -1)
        else:
            registers = None

        if self.dims.is_pretrain:
            # Pretrain: no state token; ActionEncoder ignores arg 0.
            if (
                os.environ.get("FLASH_RT_MOTUS_NO_G7_54_ACTION_POS_ADD", "0") != "1"
                and registers is None
            ):
                try:
                    import flash_rt.flash_rt_kernels as fvk
                    from flash_rt.models.motus._stream import cs
                    enc = self.model.action_expert.input_encoder
                    action_encoded = enc.action_encoder(action_latent)
                    pos = enc.pos_embedding[:, :action_encoded.shape[1], :]
                    if (
                        action_encoded.is_cuda
                        and action_encoded.dtype == torch.bfloat16
                        and pos.dtype == torch.bfloat16
                        and action_encoded.is_contiguous()
                        and pos.is_contiguous()
                        and hasattr(fvk, "add_bf16_out")
                    ):
                        out = torch.empty_like(action_encoded)
                        fvk.add_bf16_out(
                            int(action_encoded.data_ptr()),
                            int(pos.data_ptr()),
                            int(out.data_ptr()),
                            int(out.numel()), cs())
                        return out
                except Exception:
                    pass
            return self.model.action_expert.input_encoder(
                None, action_latent, registers)

        # Finetune mode (not exercised at G2 with current ckpt):
        st = state.unsqueeze(1).to(self.dtype)
        return self.model.action_expert.input_encoder(st, action_latent, registers)

    # ──────────────────────────────────────────────────────────────
    # Time embeddings + AdaLN modulation
    # ──────────────────────────────────────────────────────────────

    def get_video_time_embeddings(
        self, t_scalar: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wan time_embedding + time_projection at scaled timestep [0,1000].

        Returns (head_time_emb [B,seq,3072], adaln_params [B,seq,6,3072]).
        """
        return self.model.video_module.get_time_embedding(t_scalar, seq_len)

    def get_action_time_embeddings(
        self, t_scalar: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Action expert time_embedding (own weights, dim=1024)."""
        return self.model.action_module.get_time_embedding(t_scalar, seq_len)

    def compute_video_adaln_modulation(
        self, video_adaln_params: torch.Tensor, layer_idx: int,
    ) -> tuple:
        """Wan layer α/β/γ × (attn, ffn) — 6 chunks of 3072.

        G5 candidate: pre-bake all 30×N modulation tables in set_prompt
        and indexed-read inside graph (Pi0.5 saved 5.5 ms this way).
        """
        return self.model.video_module.compute_adaln_modulation(
            video_adaln_params, layer_idx)

    def compute_action_adaln_modulation(
        self, action_adaln_params: torch.Tensor, layer_idx: int,
    ) -> tuple:
        return self.model.action_module.compute_adaln_modulation(
            action_adaln_params, layer_idx)

    # ──────────────────────────────────────────────────────────────
    # Per-layer compute primitives — the G3/G4 substitution surface
    # ──────────────────────────────────────────────────────────────

    def denoise_layer_joint_attention(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        und_tokens: torch.Tensor,
        v_mod: tuple,
        a_mod: tuple,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tri-model joint self-attention: cat Q/K/V across {v,a,u} →
        single MHA → split outputs → expert-specific O proj + residual.

        G3c target: replace upstream ``flash_attention(...)`` (currently
        SDPA shim) with ``self.attn.run("mot_joint", layer_idx, q_seq=...)``.
        The QKV projections are inside the upstream method and need
        their own G3a/b refactor; we keep them lumped here at G2.
        """
        return self.model.video_module.process_joint_attention(
            video_tokens, action_tokens,
            v_mod, a_mod, layer_idx,
            self.model.action_expert.blocks[layer_idx],
            und_tokens, self.model.und_expert.blocks[layer_idx],
        )

    def denoise_layer_cross_attn_to_t5(
        self,
        video_tokens: torch.Tensor,
        v_adaln_params: torch.Tensor,
        layer_idx: int,
        t5_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Wan video tokens cross-attend to T5 ctx (KV reused across steps).

        G5 target: pre-compute KV proj per layer in set_prompt → store
        in ``self._t5_kv[30, 512, 24, 128]`` device buffer; the inner
        loop reads cached KV instead of recomputing every step.
        """
        return self.model.video_module.process_cross_attention(
            video_tokens, v_adaln_params, layer_idx, t5_ctx)

    def denoise_layer_ffn_video(
        self,
        video_tokens: torch.Tensor,
        v_mod: tuple,
        layer_idx: int,
    ) -> torch.Tensor:
        """Video FFN (large: hidden 3072, intermediate 14336).

        G3d target: gate + activation + mul + quantize fused into
        ``fvk.gate_geglu_merged_fp8_fp16`` (Wan uses GELU-tanh approx
        per upstream; verify in G3a startup audit).
        """
        return self.model.video_module.process_ffn(
            video_tokens, v_mod, layer_idx)

    def denoise_layer_ffn_action(
        self,
        action_tokens: torch.Tensor,
        a_mod: tuple,
        layer_idx: int,
    ) -> torch.Tensor:
        """Action expert FFN (hidden 1024)."""
        return self.model.action_module.process_ffn(
            action_tokens, a_mod, layer_idx)

    def denoise_layer_ffn_und(
        self,
        und_tokens: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Und expert FFN (hidden 512). No AdaLN — uses plain LayerNorm."""
        return self.model.und_module.process_ffn(und_tokens, layer_idx)

    # ──────────────────────────────────────────────────────────────
    # Output heads
    # ──────────────────────────────────────────────────────────────

    def _head_adaln_bf16(
        self,
        x: torch.Tensor,
        shift: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        eps: float,
    ) -> Optional[torch.Tensor]:
        if (
            shift is None
            or scale is None
            or not hasattr(shift, "data_ptr")
            or not hasattr(scale, "data_ptr")
            or not x.is_cuda
            or x.dtype != torch.bfloat16
            or shift.dtype != torch.bfloat16
            or scale.dtype != torch.bfloat16
            or x.shape[-1] != shift.shape[-1]
            or x.shape[-1] != scale.shape[-1]
            or not x.is_contiguous()
            or not shift.is_contiguous()
            or not scale.is_contiguous()
            or os.environ.get("FLASH_RT_MOTUS_NO_G7_50_HEAD_ADALN", "0") == "1"
        ):
            return None
        try:
            import flash_rt.flash_rt_kernels as fvk
            from flash_rt.models.motus._stream import cs
            if not hasattr(fvk, "ada_layer_norm_bf16"):
                return None
            dim = int(x.shape[-1])
            flat = x.reshape(-1, dim)
            out = torch.empty_like(flat)
            rows = int(flat.shape[0])
            if shift.dim() == 1:
                fvk.ada_layer_norm_bf16(
                    int(flat.data_ptr()), int(scale.data_ptr()),
                    int(shift.data_ptr()), int(out.data_ptr()),
                    rows, dim, float(eps), cs())
            elif (
                shift.dim() == 2
                and scale.dim() == 2
                and int(shift.shape[0]) == rows
                and int(scale.shape[0]) == rows
                and hasattr(fvk, "ada_layer_norm_bf16_per_token")
            ):
                fvk.ada_layer_norm_bf16_per_token(
                    int(flat.data_ptr()), int(scale.data_ptr()),
                    int(shift.data_ptr()), int(out.data_ptr()),
                    rows, dim, float(eps), cs())
            else:
                return None
            return out.view_as(x)
        except Exception:
            return None

    def video_output_head(
        self,
        video_tokens: torch.Tensor,
        video_t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """[B, S_v, 3072] → unpatchify → [B, 48, T_l, H', W'] velocity."""
        if os.environ.get("FLASH_RT_MOTUS_USE_BF16_VIDEO_HEAD", "1") == "1":
            try:
                wan_model = self.model.video_model.wan_model
                step_idx = getattr(self, "_current_step_idx", None)
                x = None
                if step_idx is not None:
                    x = self._head_adaln_bf16(
                        video_tokens,
                        None if self._cached_v_head_shift is None else self._cached_v_head_shift[step_idx],
                        None if self._cached_v_head_scale is None else self._cached_v_head_scale[step_idx],
                        float(wan_model.head.eps))
                    if x is not None:
                        x = wan_model.head.head(x)
                if x is None:
                    x = wan_model.head(video_tokens, video_t_emb)
                chunks = wan_model.unpatchify(
                    x, self.model.video_module.grid_sizes)
                if len(chunks) == 1 and chunks[0].dtype == torch.bfloat16:
                    return chunks[0].unsqueeze(0)
                if all(u.dtype == torch.bfloat16 for u in chunks):
                    return torch.stack(chunks, dim=0)
            except Exception:
                pass
        return self.model.video_module.apply_output_head(
            video_tokens, video_t_emb)

    def action_output_head(
        self,
        action_tokens: torch.Tensor,
        action_t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """ActionDecoder → velocity [B, S_a_full, action_dim]; slice
        out the actual action chunk based on pretrain/finetune layout.
        """
        decoder = self.model.action_expert.decoder
        full = None
        step_idx = getattr(self, "_current_step_idx", None)
        if step_idx is not None:
            z = self._head_adaln_bf16(
                action_tokens,
                None if self._cached_a_head_shift is None else self._cached_a_head_shift[step_idx],
                None if self._cached_a_head_scale is None else self._cached_a_head_scale[step_idx],
                float(decoder.norm.eps))
            if z is not None:
                full = decoder.action_head(z)
        if full is None:
            full = decoder(action_tokens, action_t_emb)
        n_reg = self.dims.num_registers
        up_len = full.shape[1] - n_reg            # drop trailing registers
        if self.dims.is_pretrain:
            return full[:, :up_len, :]            # no leading state token
        return full[:, 1:up_len, :]               # drop leading state token

    # ──────────────────────────────────────────────────────────────
    # Scheduler step + teacher-forcing (Euler form, matches baseline)
    # ──────────────────────────────────────────────────────────────

    def euler_step(
        self,
        latent: torch.Tensor,
        velocity: torch.Tensor,
        dt: torch.Tensor,
        dt_host: Optional[float] = None,
    ) -> torch.Tensor:
        if (
            os.environ.get("FLASH_RT_MOTUS_USE_FVK_EULER", "1") == "1"
            and dt_host is not None
            and latent.is_cuda
            and velocity.is_cuda
            and latent.dtype == torch.bfloat16
            and velocity.dtype == torch.bfloat16
            and latent.is_contiguous()
            and velocity.is_contiguous()
            and latent.numel() == velocity.numel()
            and (latent.numel() & 1) == 0
        ):
            try:
                import flash_rt.flash_rt_kernels as fvk
                from flash_rt.models.motus._stream import cs
                out = torch.empty_like(latent)
                fvk.euler_step_bf16_out(
                    int(latent.data_ptr()), int(velocity.data_ptr()),
                    int(out.data_ptr()), float(dt_host),
                    int(latent.numel()), cs())
                return out
            except Exception:
                pass
        return latent + velocity * dt

    def teacher_force_first_frame(
        self,
        video_latent: torch.Tensor,
        cond_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Re-inject the (deterministic) cond frame latent at slot 0
        every step — keeps the conditioning signal honest under noise.
        """
        if (
            os.environ.get("FLASH_RT_MOTUS_USE_FVK_TEACHER_FORCE", "1") == "1"
            and video_latent.is_cuda
            and cond_latent.is_cuda
            and video_latent.dtype == torch.bfloat16
            and cond_latent.dtype == torch.bfloat16
            and video_latent.is_contiguous()
            and cond_latent.is_contiguous()
            and video_latent.dim() == 5
            and cond_latent.dim() == 5
            and cond_latent.shape[2] == 1
        ):
            try:
                import flash_rt.flash_rt_kernels as fvk
                from flash_rt.models.motus._stream import cs
                B, C, T, H, W = (int(v) for v in video_latent.shape)
                fvk.teacher_force_first_frame_bf16(
                    int(video_latent.data_ptr()), int(cond_latent.data_ptr()),
                    B, C, T, H, W, cs())
                return video_latent
            except Exception:
                pass
        video_latent[:, :, 0:1] = cond_latent
        return video_latent

    def cast_bf16_to_float(self, x: torch.Tensor) -> torch.Tensor:
        if (
            os.environ.get("FLASH_RT_MOTUS_USE_FVK_CAST_F32", "1") == "1"
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
        ):
            try:
                import flash_rt.flash_rt_kernels as fvk
                from flash_rt.models.motus._stream import cs
                out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
                fvk.cast_bf16_to_fp32(
                    int(x.data_ptr()), int(out.data_ptr()),
                    int(x.numel()), cs())
                return out
            except Exception:
                pass
        return x.float()

    # ──────────────────────────────────────────────────────────────
    # End-to-end denoise (replaces upstream `inference_step`)
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run(
        self,
        first_frame: torch.Tensor,        # [B, 3, H, W] in [0, 1]
        state: Optional[torch.Tensor],    # [B, state_dim] or None (pretrain)
        t5_embeds: List[torch.Tensor],    # one entry per batch element
        vlm_inputs: List[dict],           # one per batch element
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the full N-step denoise + decode pipeline.

        Returns (predicted_frames [B,3,T,H,W] float32 in [0,1],
                 predicted_actions [B, action_chunk, action_dim] float32).
        """
        # --- one-shots ---
        cond_latent = self.encode_first_frame(first_frame)   # [B,48,1,H',W']
        t5_ctx = self.preprocess_t5_context(t5_embeds)       # [B,512,3072]
        B = cond_latent.shape[0]

        # --- noise init + cond frame inject ---
        video_latent = self.init_video_latent(cond_latent)
        action_latent = self.init_action_latent(B)

        # --- denoise loop (Euler, matches baseline_artifacts) ---
        # G5: prefer pre-computed time embeddings (cached on the
        # pipeline) — they avoid CPU-side ops (torch.arange) inside
        # the captured graph. Cache miss falls back to per-step
        # compute (eager path / pre-G5 callers).
        use_cached_t = (
            getattr(self, "_cached_v_t_emb", None) is not None
            and getattr(self, "_cached_a_t_emb", None) is not None
            and getattr(self, "_cached_time_steps", None) is not None
        )
        use_cached_mod = (
            use_cached_t
            and getattr(self, "_cached_v_mod", None) is not None
            and getattr(self, "_cached_a_mod", None) is not None
        )
        if use_cached_t:
            timesteps = self._cached_time_steps
        else:
            timesteps = torch.linspace(
                1.0, 0.0, self.dims.num_inference_steps + 1,
                device=self.device, dtype=self.dtype)

        for i in range(self.dims.num_inference_steps):
            self._current_step_idx = i
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt_host = None
            if use_cached_t and getattr(self, "_cached_dt_host", None) is not None:
                dt_host = self._cached_dt_host[i]
                dt = dt_host
            else:
                dt = t_next - t

            video_tokens = self.prepare_video_tokens(video_latent)
            action_tokens = self.prepare_action_tokens(action_latent, state)
            und_tokens = self.extract_und_tokens(vlm_inputs)

            # autocast matches baseline (Wan ops live under bf16 amp)
            with torch.autocast(
                device_type="cuda", dtype=self.model.video_model.precision,
            ):
                if use_cached_t:
                    v_t_emb = self._cached_v_t_emb[i]
                    a_t_emb = self._cached_a_t_emb[i]
                    if use_cached_mod:
                        if getattr(self, "_cached_mod_partial", False):
                            v_adaln = self._cached_v_adaln[i]
                            a_adaln = self._cached_a_adaln[i]
                        else:
                            v_adaln = None
                            a_adaln = None
                    else:
                        v_adaln = self._cached_v_adaln[i]
                        a_adaln = self._cached_a_adaln[i]
                else:
                    video_t_scaled = (t * 1000).expand(B).to(self.dtype)
                    action_t_scaled = (t * 1000).expand(B).to(self.dtype)
                    v_t_emb, v_adaln = self.get_video_time_embeddings(
                        video_t_scaled, video_tokens.shape[1])
                    a_t_emb, a_adaln = self.get_action_time_embeddings(
                        action_t_scaled, action_tokens.shape[1])

                for L in range(self.dims.num_layers):
                    if use_cached_mod:
                        v_mod = self._cached_v_mod[i][L]
                        a_mod = self._cached_a_mod[i][L]
                        if v_mod is None:
                            v_mod = self.compute_video_adaln_modulation(
                                v_adaln, L)
                        if a_mod is None:
                            a_mod = self.compute_action_adaln_modulation(
                                a_adaln, L)
                        v_mod = self._maybe_materialize_static_mod(
                            v_mod, "video")
                        a_mod = self._maybe_materialize_static_mod(
                            a_mod, "action")
                    else:
                        v_mod = self.compute_video_adaln_modulation(v_adaln, L)
                        a_mod = self.compute_action_adaln_modulation(a_adaln, L)

                    video_tokens, action_tokens, und_tokens = (
                        self.denoise_layer_joint_attention(
                            video_tokens, action_tokens, und_tokens,
                            v_mod, a_mod, L))
                    video_tokens = self.denoise_layer_cross_attn_to_t5(
                        video_tokens, v_adaln, L, t5_ctx)
                    action_stream, und_stream = self._ensure_ffn_streams()
                    if action_stream is None:
                        video_tokens = self.denoise_layer_ffn_video(
                            video_tokens, v_mod, L)
                        action_tokens = self.denoise_layer_ffn_action(
                            action_tokens, a_mod, L)
                        und_tokens = self.denoise_layer_ffn_und(und_tokens, L)
                    else:
                        main_stream = torch.cuda.current_stream(self.device)
                        action_stream.wait_stream(main_stream)
                        und_stream.wait_stream(main_stream)
                        with torch.cuda.stream(action_stream):
                            action_tokens_next = self.denoise_layer_ffn_action(
                                action_tokens, a_mod, L)
                        with torch.cuda.stream(und_stream):
                            und_tokens_next = self.denoise_layer_ffn_und(
                                und_tokens, L)
                        video_tokens = self.denoise_layer_ffn_video(
                            video_tokens, v_mod, L)
                        main_stream.wait_stream(action_stream)
                        main_stream.wait_stream(und_stream)
                        action_tokens = action_tokens_next
                        und_tokens = und_tokens_next

                action_stream, _ = self._ensure_ffn_streams()
                if action_stream is None:
                    video_velocity = self.video_output_head(video_tokens, v_t_emb)
                    action_velocity = self.action_output_head(
                        action_tokens, a_t_emb)
                else:
                    main_stream = torch.cuda.current_stream(self.device)
                    action_stream.wait_stream(main_stream)
                    with torch.cuda.stream(action_stream):
                        action_velocity_next = self.action_output_head(
                            action_tokens, a_t_emb)
                    video_velocity = self.video_output_head(video_tokens, v_t_emb)
                    main_stream.wait_stream(action_stream)
                    action_velocity = action_velocity_next

                video_latent = self.euler_step(
                    video_latent, video_velocity, dt, dt_host)
                action_latent = self.euler_step(
                    action_latent, action_velocity, dt, dt_host)
                video_latent = self.teacher_force_first_frame(
                    video_latent, cond_latent)
        self._current_step_idx = None

        # --- decode + post ---
        predicted_frames = self.decode_video(video_latent)
        predicted_actions = self.cast_bf16_to_float(action_latent)
        return predicted_frames, predicted_actions
