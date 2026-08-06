"""FlashRT — Qwen3-VL BF16 frontend for Jetson Thor (SM110).

Correctness-first BF16 bring-up of Qwen3-VL on Thor, establishing numerical
parity with the HF BF16 reference. All language dims are read from
``config.json``, so the 2B/4B/8B variants load unchanged.

Why a dedicated Thor module rather than the SM87 ``Qwen3VlTorchFrontendRtxBF16``:
  * Thor does not build the vendored FA2 (``flash_rt_fa2``); attention runs
    through :class:`ThorAttnBackendQwen3` (SDPA today; the Thor CUTLASS causal
    FMHA can be swapped in later behind the same buffer surface).
  * Prefill drives the batched `qwen3_qk_norm_rope_kvwrite_batched_bf16` in one
    launch per layer, which is bit-identical to the per-row
    `qwen3_q_norm_rope_qstage_bf16` / `qwen3_k_norm_rope_kvwrite_bf16` kernels
    that decode uses. So prefill and decode math agree without re-implementing
    RoPE in torch, and prefill avoids ~2·S launches per layer.
  * bf16 linears use the Thor ``flash_rt_qwen3_vl_kernels`` cuBLASLt matmul.

v1 scope: text-only and single-image chat prompts, greedy generation, eager
prefill, graph-replay decode. Multi-image, video, an FP8 ViT tower and
speculative decoding are not supported yet.

See docs/qwen3_vl_thor.md.
"""
from __future__ import annotations

import collections
import json
import os
from typing import Any


# flash_rt_kernels (fvk) language ops required on Thor. embedding_lookup_bf16
# and bf16_matmul_bf16 are model-neutral and built for every arch;
# bf16_matmul_bf16 is uncalled here (GEMMs go through the vlk cuBLASLt/GEMV
# path) and is required purely as a build-inventory health check.
_THOR_CORE_FNS = (
    'rms_norm',
    'residual_add',
    'residual_add_rms_norm',
    'silu_mul_qwen36_bf16',
    'qwen3_q_norm_rope_qstage_bf16',
    'qwen3_k_norm_rope_kvwrite_bf16',
    'embedding_lookup_bf16',
    'bf16_matmul_bf16',
)
# flash_rt_qwen3_vl_kernels (vlk) ops required on Thor.
_THOR_VL_FNS = (
    'bf16_matmul_cublaslt_bf16',
    'qwen3_qk_norm_rope_kvwrite_batched_bf16',
)


def _require_thor_kernels():
    try:
        from flash_rt import flash_rt_kernels as fvk
        from flash_rt import flash_rt_qwen3_vl_kernels as vlk
    except ImportError as e:
        raise RuntimeError(
            'Qwen3VlTorchFrontendThor requires flash_rt_kernels and '
            'flash_rt_qwen3_vl_kernels. Build with '
            '-DGPU_ARCH=110 -DFLASHRT_BUILD_QWEN3_VL=ON '
            '(targets flash_rt_kernels flash_rt_qwen3_vl_kernels).') from e
    missing_core = [n for n in _THOR_CORE_FNS if not hasattr(fvk, n)]
    missing_vl = [n for n in _THOR_VL_FNS if not hasattr(vlk, n)]
    if missing_core or missing_vl:
        pieces = []
        if missing_core:
            pieces.append('flash_rt_kernels: ' + ', '.join(missing_core))
        if missing_vl:
            pieces.append('flash_rt_qwen3_vl_kernels: ' + ', '.join(missing_vl))
        raise RuntimeError(
            'Thor Qwen3-VL kernels are incomplete: ' + '; '.join(pieces))
    return fvk, vlk


# Valid decode weight-quant modes and the projections an override may target.
_WQ_MODES = ('bf16', 'w8', 'w4')
_WQ_PROJ_KEYS = ('qkv_proj', 'o_proj', 'gate_up', 'mlp_down', 'lm_head')


def _validate_wq_overrides(overrides: dict | None) -> dict:
    """Validate a per-projection quant-mode override map and return a copy.

    Keys are ``'{proj}'`` or ``'L{layer}.{proj}'``; values are in
    ``_WQ_MODES``. Unknown keys are rejected rather than ignored — a silently
    dropped override reads as "this projection doesn't matter" in a
    sensitivity sweep.
    """
    ov = dict(overrides or {})
    bad = {k: v for k, v in ov.items() if v not in _WQ_MODES}
    if bad:
        raise ValueError(f'wq_overrides values must be one of {_WQ_MODES}: '
                         f'{bad}')
    for key in ov:
        if key.startswith('L') and '.' in key:
            prefix, base = key.split('.', 1)
            if not prefix[1:].isdigit():
                raise ValueError(
                    f'wq_overrides key {key!r}: layer prefix must be '
                    f"'L<integer>.' (e.g. 'L12.gate_up')")
        else:
            base = key
        if base not in _WQ_PROJ_KEYS:
            raise ValueError(
                f'wq_overrides key {key!r} must be one of {_WQ_PROJ_KEYS} '
                f"(optionally 'L<layer>.' prefixed)")
        if base == 'lm_head' and base != key:
            raise ValueError(
                f"wq_overrides key {key!r}: lm_head is not per-layer; "
                "use 'lm_head'")
    return ov


def _wq_active(mode: str, overrides: dict) -> bool:
    """True when the global mode or any override requests quantization."""
    return mode != 'bf16' or any(v != 'bf16' for v in overrides.values())


class Qwen3VlTorchFrontendThor:
    """Batch-1 Qwen3-VL image+text inference (Thor SM110, BF16)."""

    def __init__(self, checkpoint_path: str, *, device: str = 'cuda:0',
                 max_seq: int = 4096, max_pixels: int | None = None,
                 weight_mode: str = 'bf16',
                 wq_overrides: dict[str, str] | None = None) -> None:
        if weight_mode not in _WQ_MODES:
            raise ValueError(f"weight_mode must be one of {_WQ_MODES}, got "
                             f"{weight_mode!r}")
        self._wq_overrides = _validate_wq_overrides(wq_overrides)
        self._wq_mode = weight_mode          # 'bf16' | 'w8' | 'w4'
        self._use_wq = _wq_active(weight_mode, self._wq_overrides)

        import torch
        from transformers import AutoProcessor

        from flash_rt.frontends.torch._qwen3_vl_bf16_weights import (
            assert_extraction_invariants_qwen3_vl_bf16,
            extract_weights_qwen3_vl_bf16,
        )
        from flash_rt.hardware.thor.attn_backend_qwen3 import (
            ThorAttnBackendQwen3,
        )

        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.max_seq = int(max_seq)
        self.max_pixels = max_pixels
        self._prompt: dict[str, Any] | None = None

        dev = torch.device(device)
        if dev.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Qwen3VlTorchFrontendThor requires CUDA')
        cap = torch.cuda.get_device_capability(dev)
        if cap != (11, 0):
            raise RuntimeError(
                'Qwen3VlTorchFrontendThor targets Jetson Thor (SM110 / '
                f'cc 11.0); got sm_{cap[0]}{cap[1]} on {device}. Use the '
                'RTX/SM89 Qwen3-VL frontends on those arches.')

        self._fvk, self._vlk = _require_thor_kernels()

        with open(os.path.join(self.checkpoint_path, 'config.json')) as f:
            cfg = json.load(f)
        self._cfg_raw = cfg
        text_cfg = cfg['text_config']
        self._cfg = {
            'rms_norm_eps': float(text_cfg.get('rms_norm_eps', 1e-6)),
            'head_dim': int(text_cfg.get('head_dim') or
                            text_cfg['hidden_size'] //
                            text_cfg['num_attention_heads']),
            'hidden_size': int(text_cfg['hidden_size']),
            'vocab_size': int(text_cfg['vocab_size']),
            'num_hidden_layers': int(text_cfg['num_hidden_layers']),
            'num_q_heads': int(text_cfg['num_attention_heads']),
            'num_kv_heads': int(text_cfg['num_key_value_heads']),
            'intermediate': int(text_cfg['intermediate_size']),
            'rope_theta': float(text_cfg.get('rope_theta')
                                or cfg.get('rope_theta') or 1_000_000.0),
        }
        self._head_dim = self._cfg['head_dim']
        if self._head_dim != 128:
            raise RuntimeError(
                f'Thor Qwen3 kernels require head_dim=128, got {self._head_dim}')
        self._image_token_id = int(cfg['image_token_id'])
        self._video_token_id = int(cfg['video_token_id'])
        self._vision_start_token_id = int(cfg['vision_start_token_id'])
        vc = cfg['vision_config']
        self._vision_cfg = vc
        self._merge = int(vc['spatial_merge_size'])
        self._vis_head_dim = int(vc['hidden_size']) // int(vc['num_heads'])
        self._num_grid_per_side = int(vc['num_position_embeddings'] ** 0.5)
        self._deepstack_layers = len(vc['deepstack_visual_indexes'])
        rope_scaling = text_cfg.get('rope_scaling') or cfg.get('rope_scaling')
        if not rope_scaling or 'mrope_section' not in rope_scaling:
            raise RuntimeError(
                'Qwen3-VL config missing rope_scaling.mrope_section')
        self._mrope_section = tuple(rope_scaling['mrope_section'])
        eos = cfg.get('eos_token_id', text_cfg.get('eos_token_id'))
        if eos is None:
            self._eos_token_ids: set[int] = set()
        else:
            self._eos_token_ids = set(eos if isinstance(eos, list) else [eos])

        self._weights = extract_weights_qwen3_vl_bf16(
            self.checkpoint_path, device=self.device)
        assert_extraction_invariants_qwen3_vl_bf16(self._weights)

        self._attn = ThorAttnBackendQwen3(
            max_seq=self.max_seq, max_q_seq=self.max_seq,
            dtype=torch.bfloat16,
            num_layers=self._cfg['num_hidden_layers'],
            num_q_heads=self._cfg['num_q_heads'],
            num_kv_heads=self._cfg['num_kv_heads'],
            head_dim=self._cfg['head_dim'], device=self.device)

        self.processor = AutoProcessor.from_pretrained(self.checkpoint_path)
        self._tokenizer = self.processor.tokenizer
        self._processor_kwargs: dict[str, Any] = {'device': self.device}
        if max_pixels is not None:
            size = getattr(getattr(self.processor, 'image_processor', None),
                           'size', {}) or {}
            self._processor_kwargs['size'] = {
                'shortest_edge': int(size.get('shortest_edge', 65536)),
                'longest_edge': int(max_pixels),
            }

        self._vision = None  # lazily constructed on the first image prompt
        self._alloc_buffers()
        self._build_mrope_caches()
        if self._use_wq:
            self._load_wq_weights()

    # ── setup ──

    def _alloc_buffers(self) -> None:
        import torch

        cfg = self._cfg
        d = torch.device(self.device)
        bf16 = torch.bfloat16
        S = self.max_seq
        H = cfg['hidden_size']
        I = cfg['intermediate']
        NQ = cfg['num_q_heads']
        NKV = cfg['num_kv_heads']
        HD = cfg['head_dim']
        self._qkv_N = (NQ + 2 * NKV) * HD
        self._h_a = torch.empty(1, S, H, device=d, dtype=bf16)
        self._h_b = torch.empty(1, S, H, device=d, dtype=bf16)
        self._qkv_out = torch.empty(S, self._qkv_N, device=d, dtype=bf16)
        self._gate_up = torch.empty(S, 2 * I, device=d, dtype=bf16)
        self._gate_tmp = torch.empty(S, I, device=d, dtype=bf16)
        self._up_tmp = torch.empty(S, I, device=d, dtype=bf16)
        self._mlp_act = torch.empty(S, I, device=d, dtype=bf16)
        self._tmp_hidden = torch.empty(S, H, device=d, dtype=bf16)
        self._norm_buf = torch.empty(S, H, device=d, dtype=bf16)
        self._norm_buf2 = torch.empty(S, H, device=d, dtype=bf16)
        self._attn_out = torch.empty(S, H, device=d, dtype=bf16)
        self._logits = torch.empty(1, cfg['vocab_size'], device=d, dtype=bf16)
        # CUDA-graph decode state (all Thor decode primitives are capturable).
        self._static_token_id = torch.zeros(1, 1, device=d, dtype=torch.long)
        self._graph_stream = torch.cuda.Stream(device=d)
        self._decode_graphs: "collections.OrderedDict[tuple[int, int], Any]" = (
            collections.OrderedDict())
        self._graph_warmed = False
        # One graph per decode position; capping below max_seq would thrash
        # (evict + recapture every token) on long generations.
        self._max_decode_graphs = self.max_seq

    def _build_mrope_caches(self) -> None:
        from flash_rt.frontends.torch import _qwen3_vl_geometry as geo

        self._mrope_cos_cache, self._mrope_sin_cache = geo.build_mrope_cache(
            max_pos=self.max_seq + self._num_grid_per_side,
            head_dim=self._head_dim, rope_theta=self._cfg['rope_theta'],
            device=self.device)
        self._vision_rope_cos_cache, self._vision_rope_sin_cache = (
            geo.build_vision_rope_cache(
                max_hw=self.max_seq * self._merge,
                head_dim=self._vis_head_dim, device=self.device))

    # ── weight-quantized (W8A16 / W4A16) decode weights ──

    def _quant_wq(self, w, mode: str | None = None):
        """bf16 [N,K] -> (packed uint8, bf16 per-16 block scales [N,K/16]) for
        the given weight-quant mode (defaults to the frontend's global mode).
        'w8' = e4m3 (1 byte/w, scale=amax/448); 'w4' = e2m1 (0.5 byte/w,
        scale=amax/6, nearest-magnitude rounding)."""
        import torch
        mode = mode or self._wq_mode
        N, K = w.shape
        g = w.float().view(N, K // 16, 16)
        if mode == 'w8':
            scale = (g.abs().amax(2, keepdim=True) / 448.0).clamp(min=1e-8)
            q = (g / scale).to(torch.float8_e4m3fn).view(N, K)
            packed = q.view(torch.uint8).contiguous()
        else:  # 'w4' — e2m1, nearest of 8 magnitudes + sign, 2 per byte
            mags = torch.tensor(
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device)
            scale = (g.abs().amax(2, keepdim=True) / 6.0).clamp(min=1e-8)
            s = g / scale
            idx = (s.abs().unsqueeze(-1)
                   - mags.view(1, 1, 1, 8)).abs().argmin(dim=-1)
            code = idx | (s < 0).to(torch.int64).mul(8)
            code = code.view(N, K)
            lo = code[:, 0::2]
            hi = code[:, 1::2]
            packed = (lo | (hi << 4)).to(torch.uint8).contiguous()
        scales = scale.view(N, K // 16).to(torch.bfloat16).contiguous()
        return packed, scales

    def _wq_mode_for(self, key: str, layer: int | None = None) -> str:
        """Per-projection quant mode: exact 'L{i}.{key}' override, then
        '{key}', then the global mode. 'bf16' = keep unquantized."""
        ov = self._wq_overrides
        if layer is not None:
            m = ov.get(f'L{layer}.{key}')
            if m is not None:
                return m
        return ov.get(key, self._wq_mode)

    def _load_wq_weights(self) -> None:
        """Quantize the language linears (+ tied lm_head) to the decode weight
        format (W8A16 / W4A16, per-projection overridable) for the
        bandwidth-bound M=1 decode GEMV. Quantizes from the resident device
        tensors the loader kept, so sharded checkpoints need no special
        handling; the bf16 weights stay resident for prefill."""
        for fn in ('qwen3_vl_w8_gemv_m1', 'qwen3_vl_w4_gemv_m1'):
            if not hasattr(self._vlk, fn):
                raise RuntimeError(
                    f'W8/W4 decode requires {fn} in '
                    'flash_rt_qwen3_vl_kernels (rebuild with -DGPU_ARCH=110 '
                    '-DFLASHRT_BUILD_QWEN3_VL=ON)')
        gemv = {'w8': self._vlk.qwen3_vl_w8_gemv_m1,
                'w4': self._vlk.qwen3_vl_w4_gemv_m1}
        # Captured graphs bake in quant-buffer pointers; requantizing would
        # leave them reading freed memory, and the new buffers need a fresh
        # 3-iteration warmup.
        self._decode_graphs.clear()
        self._graph_warmed = False
        self._wq_anchors: list = []
        layers = self._weights.ptrs['layers']

        def store(lw, key, L):
            mode = self._wq_mode_for(key, L)
            if mode == 'bf16':
                lw.pop(key + '_wq', None)
                return
            p, s = self._quant_wq(lw[key + '_t'], mode)
            self._wq_anchors += [p, s]
            lw[key + '_wq'] = (int(p.data_ptr()), int(s.data_ptr()),
                               gemv[mode])

        for L in range(self._cfg['num_hidden_layers']):
            for key in ('qkv_proj', 'o_proj', 'gate_up', 'mlp_down'):
                store(layers[L], key, L)

        mode = self._wq_mode_for('lm_head')
        if mode == 'bf16':
            self._lmhead_wq = None
        else:
            p, s = self._quant_wq(self._weights.ptrs['lm_head_t'], mode)
            self._wq_anchors += [p, s]
            self._lmhead_wq = (int(p.data_ptr()), int(s.data_ptr()),
                               gemv[mode])

    def _proj(self, lw, key, x, out, S, N, K, stream) -> None:
        """Projection GEMM. M=1 decode uses the W8/W4 GEMV when enabled;
        prefill (S>1) and bf16 mode use the bf16 GEMM path."""
        wq = lw.get(key + '_wq') if self._use_wq else None
        if S == 1 and wq is not None:
            wq[2](x.data_ptr(), wq[0], wq[1], out.data_ptr(), N, K, stream)
        else:
            self._bf16_gemm(x, int(lw[key + '_w']), out, S, N, K, stream)

    def _lm_head(self, xn, out, stream, *, decode: bool) -> None:
        V = self._cfg['vocab_size']
        H = self._cfg['hidden_size']
        if decode and self._use_wq and self._lmhead_wq is not None:
            wq = self._lmhead_wq
            wq[2](xn.data_ptr(), wq[0], wq[1], out.data_ptr(), V, H, stream)
        else:
            self._bf16_gemm(xn, int(self._weights.ptrs['lm_head_w']),
                            out, 1, V, H, stream)

    def _ensure_native_vision(self):
        """Lazily build the native Thor bf16 ViT (``Qwen3VlVisionRtx`` forced
        to its bf16 path — all its bf16 kernels are present on Thor, the
        FP8-block128 ones are not). Drops the HF ``AutoModelForImageTextToText``
        dependency for the image path. Same forward surface the HF path feeds:
        (image_embeds [n_merged, out_hidden], [deepstack features])."""
        if self._vision is None:
            from flash_rt.frontends.torch._qwen3_vl_vision_rtx import (
                Qwen3VlVisionRtx,
            )
            self._vision = Qwen3VlVisionRtx(
                self.checkpoint_path, device=self.device,
                config=self._vision_cfg, fp8=False,
                attention_backend='sdpa')
        return self._vision

    def reset_state(self) -> None:
        self._attn.reset_cache()

    # ── prompt ──

    def set_prompt(self, messages: list) -> None:
        """Preprocess a text-only or single-image Qwen3-VL chat prompt."""
        import torch
        from flash_rt.frontends.torch import _qwen3_vl_geometry as geo

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt',
            processor_kwargs=self._processor_kwargs).to(self.device)
        input_ids = inputs['input_ids'][0]
        S = int(input_ids.shape[0])
        if S > self.max_seq:
            raise ValueError(
                f'prompt length {S} exceeds max_seq {self.max_seq}')

        image_grid = inputs.get('image_grid_thw')
        video_grid = inputs.get('video_grid_thw')
        if video_grid is not None and len(video_grid):
            raise ValueError('Qwen3VlTorchFrontendThor v1 supports images, '
                             'not video')
        pix_img = inputs.get('pixel_values')
        has_image = pix_img is not None

        span = None
        pixel_values = None
        vcos = vsin = pos_embeds = None
        if has_image:
            pix_img = pix_img.to(torch.bfloat16)
            segs = geo.vision_segments(
                input_ids.cpu(), image_grid, None,
                image_token_id=self._image_token_id,
                video_token_id=self._video_token_id,
                spatial_merge_size=self._merge)
            if len(segs) != 1:
                raise ValueError(
                    'Qwen3VlTorchFrontendThor v1 supports exactly one image; '
                    f'got {len(segs)} vision segments')
            sg = segs[0]
            span = sg['span']
            import torch as _torch
            n_patch = int(sg['patches'])
            pixel_values = pix_img[:n_patch].contiguous()
            seg_grid = _torch.tensor([sg['grid']], dtype=_torch.long)
            vcos, vsin = geo.vision_rope_cos_sin_cached(
                seg_grid, self._vision_rope_cos_cache,
                self._vision_rope_sin_cache,
                spatial_merge_size=self._merge)
            pos_embeds = geo.vision_pos_embeds(
                seg_grid, self._ensure_native_vision().pos_embed,
                num_grid_per_side=self._num_grid_per_side,
                spatial_merge_size=self._merge, device=self.device)

        pos_ids = geo.mrope_position_ids(
            input_ids.cpu(),
            image_grid.cpu() if (has_image and image_grid is not None)
            else None,
            None,
            image_token_id=self._image_token_id,
            video_token_id=self._video_token_id,
            vision_start_token_id=self._vision_start_token_id,
            spatial_merge_size=self._merge)
        mcos, msin = geo.mrope_cos_sin_cached(
            pos_ids, self._mrope_cos_cache, self._mrope_sin_cache,
            mrope_section=self._mrope_section)

        self._prompt = {
            'input_ids': input_ids.contiguous(),
            'has_image': has_image,
            'pixel_values': pixel_values,
            'vcos': vcos, 'vsin': vsin, 'pos_embeds': pos_embeds,
            'span': span,
            'mcos': mcos, 'msin': msin,
            'S': S,
            'mrope_max': int(pos_ids.max()),
        }

    def set_prompt_text(self, text: str, *, system: str | None = None) -> None:
        """Convenience: text-only chat prompt (Phase-1 language validation)."""
        messages = []
        if system is not None:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': text})
        self.set_prompt(messages)

    # ── linear helper ──

    def _bf16_gemm(self, x, weight_ptr: int, out, M: int, N: int, K: int,
                   stream: int) -> None:
        # M=1 decode: the dedicated warp-per-row GEMV is bandwidth-bound and
        # beats cuBLASLt's poor M=1 tactics (covers K in {2048, 6144}, which is
        # every 2B decode GEMM including lm_head). Larger M (prefill) and other
        # K go through cuBLASLt.
        if (M == 1 and K in (2048, 6144)
                and hasattr(self._vlk, 'qwen3_vl_bf16_gemv_m1')):
            self._vlk.qwen3_vl_bf16_gemv_m1(
                x.data_ptr(), int(weight_ptr), out.data_ptr(), N, K, stream)
        else:
            self._vlk.bf16_matmul_cublaslt_bf16(
                x.data_ptr(), int(weight_ptr), out.data_ptr(), M, N, K, stream)

    # ── one decoder layer (prefill loops rows; decode is S=1) ──

    def _layer_forward(self, L: int, h, cos_S, sin_S, start_pos: int, S: int):
        import torch

        cfg = self._cfg
        fvk = self._fvk
        H = cfg['hidden_size']
        I = cfg['intermediate']
        NQ = cfg['num_q_heads']
        NKV = cfg['num_kv_heads']
        HD = cfg['head_dim']
        Nq = NQ * HD
        Nk = NKV * HD
        eps = cfg['rms_norm_eps']
        stream = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]

        # 1) input RMSNorm
        h2 = h.view(S, H)
        xn = self._norm_buf[:S]
        fvk.rms_norm(h2.data_ptr(), int(lw['input_norm_w']), xn.data_ptr(),
                     S, H, eps, stream)

        # 2) fused QKV projection
        qkv = self._qkv_out[:S]
        self._proj(lw, 'qkv_proj', xn, qkv, S, int(lw['qkv_proj_N']), H, stream)

        # 3) fused q/k RMSNorm + MRoPE + KV-cache write for ALL S rows in ONE
        #    launch. Bit-identical to the per-row decode kernels (verified), but
        #    replaces the ~2·S per-layer Python→kernel launches that dominated
        #    prefill/TTFT. cos_S/sin_S are (S,64) contiguous so the kernel's
        #    token*64 row indexing matches; start_pos=0 for prefill.
        q_row_stride = int(qkv.stride(0))
        Q_buf = self._attn.Q_buf
        q_dst_row = int(Q_buf.stride(1))
        kv_layer_stride = self._attn.kv_layer_stride_bytes
        kv_row_stride = self._attn.kv_row_stride_bytes
        qkv_elem = qkv.element_size()
        kv_dst_row_elems = kv_row_stride // qkv_elem
        qkv_base = qkv.data_ptr()
        k_base = self._attn.K_cache.data_ptr() + L * kv_layer_stride
        v_base = self._attn.V_cache.data_ptr() + L * kv_layer_stride
        self._vlk.qwen3_qk_norm_rope_kvwrite_batched_bf16(
            qkv_base, qkv_base + Nq * qkv_elem,
            qkv_base + (Nq + Nk) * qkv_elem,
            int(lw['q_norm_w']), int(lw['k_norm_w']),
            cos_S.data_ptr(), sin_S.data_ptr(),
            Q_buf.data_ptr(),
            k_base + start_pos * kv_row_stride,
            v_base + start_pos * kv_row_stride,
            S, q_row_stride, q_row_stride, q_row_stride,
            q_dst_row, kv_dst_row_elems,
            NQ, NKV, eps, stream)

        # 4) attention over [start_pos : start_pos+S]
        self._attn.run('full', layer_idx=L, q_seq=S,
                       kv_seq=start_pos + S, stream=stream, causal=True)
        attn_2d = self._attn.O_buf[:, :S].reshape(S, H)
        attn_out = self._attn_out[:S]
        self._proj(lw, 'o_proj', attn_2d.contiguous(), attn_out, S, H, H,
                   stream)

        # 5) residual + post-attn RMSNorm
        xn2 = self._norm_buf[:S]
        fvk.residual_add_rms_norm(
            h.data_ptr(), attn_out.view(1, S, H).data_ptr(),
            int(lw['post_attn_norm_w']), xn2.data_ptr(), S, H, eps, stream)

        # 6) gate/up + silu·mul + down
        gate_up = self._gate_up[:S]
        self._proj(lw, 'gate_up', xn2, gate_up, S, int(lw['gate_up_N']), H,
                   stream)
        gate = self._gate_tmp[:S]
        up = self._up_tmp[:S]
        gate.copy_(gate_up[:, :I])
        up.copy_(gate_up[:, I:])
        fvk.silu_mul_qwen36_bf16(
            gate.data_ptr(), up.data_ptr(), self._mlp_act[:S].data_ptr(),
            S * I, stream)
        down = self._tmp_hidden[:S]
        self._proj(lw, 'mlp_down', self._mlp_act[:S], down, S, H, I, stream)

        # 7) residual (into ping-pong; layer 0 reads _h_a so it must write
        #    _h_b — flip parity so h_out never aliases h).
        h_out = (self._h_b if (L % 2 == 0) else self._h_a)[:, :S]
        torch.add(h.view(1, S, H), down.view(1, S, H), out=h_out)
        return h_out

    def _decode_layer(self, L, h, xn, cos, sin, cache_pos, next_norm_w):
        """Aggressively-fused S=1 decode layer. ``h`` is the residual (updated
        in place by residual_add_rms_norm); ``xn`` is this layer's pre-normed
        input (consumed early by the QKV GEMV, then reused as the output
        buffer for the NEXT layer's input norm). Fuses: (a) no gate/up copies —
        silu_mul reads the fused gate_up buffer via strided pointers (S=1 slices
        are contiguous); (b) the MLP residual-add folds into ``next_norm_w`` so
        there is no separate residual add + input rms_norm per layer (the last
        layer passes final_norm_w, folding the final norm too). Returns ``xn``
        holding rms_norm(updated residual, next_norm_w)."""
        import torch

        cfg = self._cfg
        fvk = self._fvk
        H = cfg['hidden_size']
        I = cfg['intermediate']
        NQ = cfg['num_q_heads']
        NKV = cfg['num_kv_heads']
        HD = cfg['head_dim']
        Nq = NQ * HD
        Nk = NKV * HD
        eps = cfg['rms_norm_eps']
        stream = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]

        qkv = self._qkv_out[:1]
        self._proj(lw, 'qkv_proj', xn, qkv, 1, int(lw['qkv_proj_N']), H, stream)

        qe = qkv.element_size()
        qkv_ptr = qkv.data_ptr()
        Q_buf = self._attn.Q_buf
        kv_ls = self._attn.kv_layer_stride_bytes
        kv_rs = self._attn.kv_row_stride_bytes
        slot = cache_pos * kv_rs
        fvk.qwen3_q_norm_rope_qstage_bf16(
            qkv_ptr, int(lw['q_norm_w']), cos.data_ptr(), sin.data_ptr(),
            Q_buf.data_ptr(), NQ, eps, stream)
        fvk.qwen3_k_norm_rope_kvwrite_bf16(
            qkv_ptr + Nq * qe, qkv_ptr + (Nq + Nk) * qe, int(lw['k_norm_w']),
            cos.data_ptr(), sin.data_ptr(),
            self._attn.K_cache.data_ptr() + L * kv_ls + slot,
            self._attn.V_cache.data_ptr() + L * kv_ls + slot,
            NKV, eps, stream)

        self._attn.run('full', layer_idx=L, q_seq=1, kv_seq=cache_pos + 1,
                       stream=stream, causal=True)
        attn_out = self._attn_out[:1]
        self._proj(lw, 'o_proj',
                   self._attn.O_buf[:, :1].reshape(1, H).contiguous(),
                   attn_out, 1, H, H, stream)

        xn2 = self._norm_buf2[:1]
        fvk.residual_add_rms_norm(
            h.data_ptr(), attn_out.view(1, 1, H).data_ptr(),
            int(lw['post_attn_norm_w']), xn2.data_ptr(), 1, H, eps, stream)

        gate_up = self._gate_up[:1]
        self._proj(lw, 'gate_up', xn2, gate_up, 1, int(lw['gate_up_N']), H,
                   stream)
        gp = gate_up.data_ptr()
        es = gate_up.element_size()
        fvk.silu_mul_qwen36_bf16(
            gp, gp + I * es, self._mlp_act[:1].data_ptr(), I, stream)
        down = self._tmp_hidden[:1]
        self._proj(lw, 'mlp_down', self._mlp_act[:1], down, 1, H, I, stream)

        # MLP residual-add fused with the next layer's input RMSNorm.
        fvk.residual_add_rms_norm(
            h.data_ptr(), down.view(1, 1, H).data_ptr(), int(next_norm_w),
            xn.data_ptr(), 1, H, eps, stream)
        return xn

    # ── prefill ──

    def prefill(self):
        """Eager multimodal prefill; returns (1, vocab) bf16 next-token logits."""
        import torch

        if self._prompt is None:
            raise RuntimeError('call set_prompt() before prefill()')
        p = self._prompt
        self.reset_state()
        cfg = self._cfg
        fvk = self._fvk
        H = cfg['hidden_size']
        S = p['S']
        stream = torch.cuda.current_stream().cuda_stream

        h = self._h_a[:, :S]
        fvk.embedding_lookup_bf16(
            p['input_ids'].view(-1).data_ptr(),
            int(self._weights.ptrs['embed_w']), h.data_ptr(), S, H, stream)

        deepstack = None
        if p['has_image']:
            a, b = p['span']
            with torch.no_grad():
                emb, deepstack = self._ensure_native_vision().forward(
                    p['pixel_values'], p['pos_embeds'],
                    p['vcos'], p['vsin'])
            h[0, a:b].copy_(emb.to(torch.bfloat16))

        cur = h
        # Batched QK norm-rope kernel (_layer_forward) hardcodes a 64-elem
        # cos/sin row stride; enforce that coupling once (silent-wrong guard).
        assert all(t.is_contiguous() and t.shape[-1] == 64
                   for t in (p['mcos'], p['msin'])), (
            'batched prefill requires (S,64) row-contiguous mcos/msin')
        for L in range(cfg['num_hidden_layers']):
            cur = self._layer_forward(L, cur, p['mcos'], p['msin'], 0, S)
            if deepstack is not None and L < self._deepstack_layers:
                a, b = p['span']
                fvk.residual_add(
                    cur[0, a:b].data_ptr(),
                    deepstack[L].to(torch.bfloat16).data_ptr(),
                    (b - a) * H, stream)

        x = cur.view(S, H)[S - 1:S].contiguous()
        xn = self._norm_buf[:1]
        fvk.rms_norm(x.data_ptr(), int(self._weights.ptrs['final_norm_w']),
                     xn.data_ptr(), 1, H, cfg['rms_norm_eps'], stream)
        self._lm_head(xn, self._logits, stream, decode=False)
        torch.cuda.synchronize()
        return self._logits

    # ── decode ──

    def decode_step(self, token_id: int, *, cache_pos: int, rope_pos: int):
        """Eager single-token decode (correctness reference for graph replay).
        Same body as the captured path, run directly with a sync."""
        import torch

        self._static_token_id.fill_(int(token_id))
        self._decode_body(cache_pos, rope_pos)
        torch.cuda.synchronize()
        return self._logits

    # ── graph-captured decode ──

    def _decode_body(self, cache_pos: int, rope_pos: int) -> None:
        """Capture-safe decode body: reads self._static_token_id, writes
        self._logits. No per-call tensor allocation, no synchronize. Uses the
        fully-fused decode chain (one input norm before the loop; each layer's
        MLP residual-add folds into the next layer's input norm / final norm)."""
        import torch

        cfg = self._cfg
        fvk = self._fvk
        H = cfg['hidden_size']
        eps = cfg['rms_norm_eps']
        n = cfg['num_hidden_layers']
        stream = torch.cuda.current_stream().cuda_stream
        h = self._h_a[:, :1]
        fvk.embedding_lookup_bf16(
            self._static_token_id.view(-1).data_ptr(),
            int(self._weights.ptrs['embed_w']), h.data_ptr(), 1, H, stream)
        cos = self._mrope_cos_cache[rope_pos:rope_pos + 1]
        sin = self._mrope_sin_cache[rope_pos:rope_pos + 1]
        layers = self._weights.ptrs['layers']
        final_w = int(self._weights.ptrs['final_norm_w'])
        xn = self._norm_buf[:1]
        fvk.rms_norm(h.view(1, H).data_ptr(), int(layers[0]['input_norm_w']),
                     xn.data_ptr(), 1, H, eps, stream)
        for L in range(n):
            nw = int(layers[L + 1]['input_norm_w']) if L + 1 < n else final_w
            xn = self._decode_layer(L, h, xn, cos, sin, cache_pos, nw)
        self._lm_head(xn, self._logits, stream, decode=True)

    def _ensure_decode_graph(self, cache_pos: int, rope_pos: int):
        import torch

        key = (int(cache_pos), int(rope_pos))
        g = self._decode_graphs.get(key)
        if g is not None:
            self._decode_graphs.move_to_end(key)
            return g
        gs = self._graph_stream
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs):
            for _ in range(3 if not self._graph_warmed else 1):
                self._decode_body(cache_pos, rope_pos)
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)
        self._graph_warmed = True
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=gs):
            self._decode_body(cache_pos, rope_pos)
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)
        while len(self._decode_graphs) >= self._max_decode_graphs:
            self._decode_graphs.popitem(last=False)
        self._decode_graphs[key] = g
        return g

    def decode_step_graph(self, token_id: int, *, cache_pos: int,
                          rope_pos: int):
        """Decode one token via a captured CUDA Graph replay."""
        self._static_token_id.fill_(int(token_id))
        self._ensure_decode_graph(cache_pos, rope_pos).replay()
        return self._logits

    def warmup_decode_graphs(self, n_tokens: int) -> None:
        if self._prompt is None:
            raise RuntimeError('call set_prompt() before warmup_decode_graphs')
        base_slot = int(self._prompt['S'])
        base_rope = int(self._prompt['mrope_max']) + 1
        if base_slot + int(n_tokens) > self.max_seq:
            raise ValueError(
                f'warmup of {n_tokens} positions from slot {base_slot} '
                f'exceeds max_seq={self.max_seq}')
        for i in range(int(n_tokens)):
            self._ensure_decode_graph(base_slot + i, base_rope + i)

    def set_wq_overrides(self, overrides: dict | None) -> None:
        """Set per-projection quant-mode overrides and requantize the decode
        weights in place (validated)."""
        was = self._use_wq
        self._wq_overrides = _validate_wq_overrides(overrides)
        self._use_wq = _wq_active(self._wq_mode, self._wq_overrides)
        if was or self._use_wq:
            self._load_wq_weights()

    def _finish_greedy(self, out_ids: list) -> str:
        """Greedy tail: cut at the first EOS, then decode."""
        eos = next((i for i, t in enumerate(out_ids)
                    if t in self._eos_token_ids), None)
        if eos is not None:
            out_ids = out_ids[:eos]
        return self._tokenizer.decode(out_ids, skip_special_tokens=True)

    def generate(self, messages: list, *, max_new_tokens: int = 64,
                 use_graph: bool = True) -> str:
        """Greedy generation. Decode runs via captured CUDA-Graph replay by
        default (graph output is bit-identical to eager); ``use_graph=False``
        forces the eager per-step path (correctness reference)."""
        if max_new_tokens < 1:
            raise ValueError(
                f'max_new_tokens must be >= 1, got {max_new_tokens}')
        self.set_prompt(messages)
        p = self._prompt
        assert p is not None
        base_slot = int(p['S'])
        base_rope = int(p['mrope_max']) + 1
        if base_slot + max_new_tokens - 1 > self.max_seq:
            raise ValueError(
                f'prompt ({base_slot} tokens) + max_new_tokens='
                f'{max_new_tokens} needs '
                f'{base_slot + max_new_tokens - 1} KV slots, but '
                f'max_seq={self.max_seq}')
        logits = self.prefill()
        step = self.decode_step_graph if use_graph else self.decode_step
        tok = int(logits[0].argmax())
        out_ids = [tok]
        for i in range(max_new_tokens - 1):
            if tok in self._eos_token_ids:
                break
            logits = step(
                tok, cache_pos=base_slot + i, rope_pos=base_rope + i)
            tok = int(logits[0].argmax())
            out_ids.append(tok)
        return self._finish_greedy(out_ids)
