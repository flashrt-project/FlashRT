"""FlashRT — PyTorch frontend for Qwen3 (plain dense) on RTX SM120.

Class name ``Qwen3TorchFrontendRtx`` follows the
``docs/adding_new_model.md`` §0 naming rule.

Init scope: __init__ / _load_nvfp4_path / _alloc_buffers /
_build_rope_table + a tiny ``_g0_smoke_forward`` that proves the
loader + every flash_rt_kernels NVFP4 entry point this path needs
is wired correctly. The full forward
(``forward_own_decode_nvfp4``) is below.

Sibling reference: ``flash_rt.frontends.torch.qwen36_rtx`` (the 27B
hybrid path). Strict subset semantics — no MTP, no spec-decode, no
linear-attn, no TurboQuant. Activations are W4A4 NVFP4 (per-token
dynamic SF + precomputed input_global_scale) baked into the GEMM
``alpha`` at load time — see
``_qwen3_rtx_nvfp4_weights.extract_weights_qwen3_nvfp4``.
"""
from __future__ import annotations

from typing import Any


class Qwen3TorchFrontendRtx:
    """Qwen3-8B-class inference frontend (PyTorch + RTX SM120, NVFP4).

    Public surface (frozen against the qwen36 sibling for analogy):
      __init__(checkpoint_path, *, device='cuda:0', max_seq=2048,
               max_q_seq=1, alloc_own_forward_buffers=True)
      set_prompt(text)              -- tokenizes for the next infer()
      reset_state()                 -- clears KV cache + cur_pos cursor
      forward_own_decode_nvfp4(...) -- per-step S=1 NVFP4 decode
    """

    # NVFP4 (N, K) shapes used as scratch buckets. Sized for the M=1
    # decode hot path; the S=N prefill path uses its own
    # max_q_seq-sized scratch.
    _NVFP4_SHAPES: tuple[tuple[int, int], ...] = (
        (4096, 4096),     # q_proj  (32 heads × 128) AND o_proj (K=hidden)
        (1024, 4096),     # k_proj / v_proj (8 heads × 128)
        (12288, 4096),    # mlp gate / mlp up (intermediate × hidden)
        (4096, 12288),    # mlp down (hidden × intermediate)
    )

    # Buckets exposed to the prefill path; one CUDA Graph is captured
    # per bucket at startup. Sized as a power-of-2 ladder up to 1024
    # for the ≤1k-token TTFT target. Adjustable via __init__ kwarg.
    DEFAULT_PREFILL_BUCKETS: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)

    # ── Init ──

    def __init__(self, checkpoint_path: str, *,
                 device: str = 'cuda:0',
                 max_seq: int = 2048,
                 max_q_seq: int = 1,
                 alloc_own_forward_buffers: bool = True,
                 prefill_buckets: tuple[int, ...] | None = None) -> None:
        """Load NVFP4 ckpt, build attention backend, allocate scratch.

        Args:
          checkpoint_path: dir holding config.json + sharded
            model-*.safetensors (compressed-tensors nvfp4-pack-quantized).
          device: CUDA device string.
          max_seq: KV cache length in tokens (caps prompt + generated).
          max_q_seq: max Q rows passed to FA2 in one call. 1 = decode
            only; bump to >1 to enable S=N prefill.
          alloc_own_forward_buffers: pre-allocate the decode + prefill
            scratch dicts at construction. Set False only for memory
            introspection unit tests.
          prefill_buckets: override the default {32,64,128,256,512,1024}
            CUDA Graph capture ladder.
        """
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.max_seq = int(max_seq)
        self.max_q_seq = int(max_q_seq)
        self.prefill_buckets = tuple(
            prefill_buckets if prefill_buckets is not None
            else self.DEFAULT_PREFILL_BUCKETS
        )
        self._tokenizer: Any = None
        self._prompt_ids = None
        self._weights = None     # WeightHandles
        self._bufs: dict | None = None
        self._attn = None        # RtxFlashAttnBackendQwen3
        self._cfg: dict | None = None
        self._rope_cos_table = None
        self._rope_sin_table = None
        self.latency_records: list[float] = []

        # Logical decode cursor (the pipeline owns it; the attention
        # backend just exposes pointer math). Starts at 0; incremented
        # by every decode/prefill step.
        self._cur_pos = 0

        self._load_nvfp4_path()
        if alloc_own_forward_buffers:
            self._alloc_buffers_nvfp4()
            self._build_rope_table()

    # ── Load ──

    def _load_nvfp4_path(self) -> None:
        """Load weights via the raw safetensors loader.

        No HF AutoModel, no compressed_tensors runtime — single safe_open
        pass per shard, on-device SF swizzle, alpha pre-baked.
        """
        import torch  # noqa: F401  (kept import-symmetric with qwen36)

        from flash_rt import flash_rt_kernels as fvk
        from flash_rt.frontends.torch._qwen3_rtx_nvfp4_weights import (
            assert_extraction_invariants_qwen3,
            extract_weights_qwen3_nvfp4,
        )

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)

        handles = extract_weights_qwen3_nvfp4(
            self.checkpoint_path, fvk, device=self.device,
        )
        assert_extraction_invariants_qwen3(handles)
        self._weights = handles

        p = handles.ptrs
        # Source-agnostic config namespace mirroring the qwen36 sibling
        # (so future shared helpers can read off self._cfg).
        self._cfg = {
            'rms_norm_eps': float(p['rms_norm_eps']),
            'head_dim': int(p['head_dim']),
            'hidden_size': int(p['hidden']),
            'vocab_size': int(p['vocab_size']),
            'num_hidden_layers': int(p['num_layers']),
            'layer_types': list(p['layer_types']),
            'num_q_heads': int(p['num_q_heads']),
            'num_kv_heads': int(p['num_kv_heads']),
            'intermediate': int(p['intermediate']),
            'rope_theta': float(p['rope_theta']),
            # full RoPE for plain Qwen3 (rotary_dim == head_dim)
            'rotary_dim': int(p['head_dim']),
        }
        self._fvk = fvk
        # NVFP4 GEMM + quant entries live in the unified
        # flash_rt_kernels module (no separate FP4 shared lib).
        if not fvk.has_nvfp4():
            raise RuntimeError(
                'flash_rt_kernels.has_nvfp4()==False — built without '
                'CUTLASS NVFP4 SM120 support.'
            )

    def _alloc_buffers_nvfp4(self) -> None:
        """Pre-allocate every NVFP4 forward buffer at fixed pointers.

        Hidden ping-pong + per-(N,K) NVFP4 scratch (act packed + SF
        swizzled + bf16 output) + intra-layer attn scratch + final
        logits + KV cache via the attn backend.
        """
        import torch

        from flash_rt.hardware.rtx.attn_backend_qwen3 import (
            RtxFlashAttnBackendQwen3,
        )

        device = torch.device(self.device)
        bf16 = torch.bfloat16
        u8 = torch.uint8
        fp32 = torch.float32

        cfg = self._cfg
        assert cfg is not None
        hidden = cfg['hidden_size']
        vocab = cfg['vocab_size']
        n_q = cfg['num_q_heads']
        n_kv = cfg['num_kv_heads']
        hd = cfg['head_dim']
        inter = cfg['intermediate']
        Sq_max = max(self.max_q_seq, max(self.prefill_buckets, default=1))

        # Attention backend (owns KV cache + Q/O scratch).
        self._attn = RtxFlashAttnBackendQwen3(
            max_seq=self.max_seq, max_q_seq=Sq_max, dtype=bf16,
        )

        # Hidden ping-pong.
        self._h_a = torch.empty(Sq_max, hidden, device=device, dtype=bf16)
        self._h_b = torch.empty(Sq_max, hidden, device=device, dtype=bf16)

        # NVFP4 scratch per (N, K) shape. Sized for max_q_seq (M dim).
        def _swz_bytes(rows: int, cols: int) -> int:
            n_blocks = cols // 16
            n_row_super = (rows + 127) // 128
            n_col_super = (n_blocks + 3) // 4
            return n_row_super * n_col_super * 512

        self._nvfp4_scratch: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
        for N, K in self._NVFP4_SHAPES:
            ap = torch.empty(Sq_max, K // 2, device=device, dtype=u8)
            sf = torch.zeros(_swz_bytes(Sq_max, K), device=device, dtype=u8)
            out = torch.empty(Sq_max, N, device=device, dtype=bf16)
            self._nvfp4_scratch[(N, K)] = (ap, sf, out)

        # Intra-layer scratches for the full-attn path.
        # q_norm output: (n_q × Sq_max, head_dim) so per-head RMSNorm
        # can run on a flat (N, head_dim) view for any S.
        self._q_norm_out = torch.empty(
            n_q * Sq_max, hd, device=device, dtype=bf16,
        )
        self._k_norm_out = torch.empty(
            n_kv * Sq_max, hd, device=device, dtype=bf16,
        )
        # Q/K post-RoPE staging (FA2 reads from K_cache directly so K
        # only needs a tmp buffer at write time).
        self._q_rot = torch.empty(
            1, Sq_max, n_q, hd, device=device, dtype=bf16,
        )
        self._k_rot = torch.empty(
            1, Sq_max, n_kv, hd, device=device, dtype=bf16,
        )
        # RoPE rotate_half index: full rotary_dim. For rotary_dim=128:
        #   rotate_half(x)[..., :64]  = -x[..., 64:128]
        #   rotate_half(x)[..., 64:]  =  x[..., :64]
        rope_dim = cfg['rotary_dim']
        half = rope_dim // 2
        idx_lo = torch.arange(half, rope_dim, device=device, dtype=torch.long)
        idx_hi = torch.arange(0, half, device=device, dtype=torch.long)
        self._rope_rotate_idx = torch.cat([idx_lo, idx_hi]).contiguous()
        self._rope_tmp_q = torch.empty(
            1, Sq_max, n_q, rope_dim, device=device, dtype=bf16,
        )
        self._rope_tmp_k = torch.empty(
            1, Sq_max, n_kv, rope_dim, device=device, dtype=bf16,
        )

        # MLP scratches (silu(gate) * up).
        self._mlp_silu_mul_out = torch.empty(
            Sq_max, inter, device=device, dtype=bf16,
        )
        # The gate output buffer is reused from _nvfp4_scratch[(inter, hidden)],
        # but up_proj also writes (Sq_max, inter); take a dedicated buf so
        # both can coexist without aliasing.
        self._mlp_up_out = torch.empty(
            Sq_max, inter, device=device, dtype=bf16,
        )

        # Layer-output ping-pong + intermediate residual.
        self._layer_out_a = torch.empty(
            1, Sq_max, hidden, device=device, dtype=bf16,
        )
        self._layer_out_b = torch.empty(
            1, Sq_max, hidden, device=device, dtype=bf16,
        )
        self._res_mid = torch.empty(
            1, Sq_max, hidden, device=device, dtype=bf16,
        )

        # Logits + last hidden snapshot.
        self._logits_buf = torch.empty(
            Sq_max, vocab, device=device, dtype=bf16,
        )
        self._last_hidden_buf = torch.empty(
            1, Sq_max, hidden, device=device, dtype=bf16,
        )

        # Static decode-step scratch: a (1, 1) long tensor holding the
        # next token id, copy_'d in place every step so CUDA graph
        # capture sees a stable pointer.
        self._static_token_id = torch.zeros(
            1, 1, device=device, dtype=torch.long,
        )

        # Static prefill prompt buffer. (1, Sq_max) long. Each
        # captured prefill graph reads from a slice [:, :S_bucket] of
        # this buffer; the driver copy_'s the real prompt ids in (with
        # tail padding) before replay so capture sees a stable address.
        self._static_prompt_ids = torch.zeros(
            1, Sq_max, device=device, dtype=torch.long,
        )

        # Boundary fusion: collapse the layer-L → layer-L+1 transition
        # (residual_2 + next-layer input_norm + nvfp4 quant) into a
        # single launch via the existing
        # `residual_add_rms_norm_to_nvfp4_swizzled_bf16` kernel. Saves
        # one launch per layer boundary. Default ON; flip to False for
        # A/B comparison against the unfused chain.
        self._enable_boundary_fusion: bool = True

        # qkv post-processing fusion: collapse (q_norm + RoPE + Q_buf
        # write) and (k_norm + RoPE + K/V cache write) into two fused
        # launches via `qwen3_q_norm_rope_qstage_bf16` /
        # `qwen3_k_norm_rope_kvwrite_bf16`. Replaces ~14 small per-layer
        # ops (RMSNorm, multi-op RoPE, three KV/Q cache copies). Default
        # ON.
        self._enable_qkv_post_fusion: bool = True

        # silu_mul + nvfp4 quant fusion: collapse `silu(gate) * up`
        # and the subsequent nvfp4 swizzled quantization into one
        # launch (`silu_mul_to_nvfp4_swizzled_bf16`); also avoids a
        # bf16 round-trip through HBM. Default ON.
        self._enable_silu_mul_quant_fusion: bool = True

        # Fused QKV / gate-up output buffers (decode-path only, M=1).
        # Sized exactly to the fused N for the layer (Nq+Nk+Nv = 6144;
        # 2× intermediate = 24576). Tiny — saves 2 launches per layer
        # × 36 = 72 kernel launches per decoded token.
        layers0 = self._weights.ptrs['layers'][0]
        if layers0.get('qkv_homogeneous_alpha'):
            qkv_N = int(layers0['qkv_proj_N'])
            self._qkv_fused_out = torch.empty(
                1, qkv_N, device=device, dtype=bf16,
            )
        else:
            self._qkv_fused_out = None
        if layers0.get('gate_up_homogeneous_alpha'):
            gu_N = int(layers0['gate_up_N'])
            self._gate_up_fused_out = torch.empty(
                1, gu_N, device=device, dtype=bf16,
            )
        else:
            self._gate_up_fused_out = None

        # CUDA Graph capture state.
        #   _captured_decode_graphs  : dict[cur_pos    -> torch.cuda.CUDAGraph]
        #   _captured_prefill_graphs : dict[S_bucket   -> torch.cuda.CUDAGraph]
        #   _graph_stream            : shared dedicated capture stream
        # Decode: each cur_pos gets its own graph because FA2
        # kv_seq=cur_pos+1 is baked into the captured kernel's int args.
        # Prefill: each S bucket gets its own graph because every kernel's
        # M-axis size is baked in. KV cache writes at [start_pos=0, S].
        # The driver pads the prompt up to S_bucket with the last real
        # token id; causal masking guarantees padded rows can't pollute
        # real-row outputs, and lm_head is run eager on the real-last-row
        # post-replay so the captured graph is independent of real prompt
        # length within a bucket.
        self._captured_decode_graphs: dict = {}
        self._captured_prefill_graphs: dict = {}
        self._graph_stream = torch.cuda.Stream(device=device)

        # Bookkeeping.
        self._bufs = {
            'Sq_max': Sq_max,
            'fp32_scratch': torch.empty(1, device=device, dtype=fp32),
        }

    def _build_rope_table(self) -> None:
        """Precompute (cos, sin) tables of shape (max_seq, rotary_dim/2).

        Standard Qwen3 RoPE: theta = rope_theta, half-dim cos/sin
        precomputed on host then broadcast at apply-time. Shape choice
        mirrors the qwen36 sibling for code reuse later.
        """
        import math

        import torch

        device = torch.device(self.device)
        cfg = self._cfg
        assert cfg is not None
        rope_dim = cfg['rotary_dim']
        theta = cfg['rope_theta']
        half = rope_dim // 2

        # inv_freq (half,)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, rope_dim, 2, device=device,
                                    dtype=torch.float32) / rope_dim)
        )
        positions = torch.arange(
            self.max_seq, device=device, dtype=torch.float32,
        )
        # (max_seq, half)
        freqs = torch.outer(positions, inv_freq)
        self._rope_cos_table = freqs.cos().to(torch.bfloat16).contiguous()
        self._rope_sin_table = freqs.sin().to(torch.bfloat16).contiguous()
        # Convenience: math constant guard for assertion in callers.
        self._rope_meta = {'theta': theta, 'rope_dim': rope_dim}
        _ = math  # silence linter on unused-import in a tiny corner

    def _rope_cos_sin(self, cur_pos: int):
        """Slice (cos, sin) at cur_pos for a single-token decode call."""
        cos = self._rope_cos_table[cur_pos:cur_pos + 1]
        sin = self._rope_sin_table[cur_pos:cur_pos + 1]
        return cos, sin

    # ── State management ──

    def reset_state(self) -> None:
        """Clear KV cache + reset cur_pos.

        Call before set_prompt() of an unrelated request to make the
        decode loop independent of prior prompts.
        """
        if self._attn is not None:
            self._attn.reset_cache()
        self._cur_pos = 0

    def set_prompt(self, text: str) -> None:
        """Tokenize prompt; downstream forward consumes self._prompt_ids."""
        if self._tokenizer is None:
            raise RuntimeError('tokenizer not loaded')
        ids = self._tokenizer(text, return_tensors='pt').input_ids.to(
            self.device,
        )
        self._prompt_ids = ids

    # ── Load smoke: exercise loader + one NVFP4 GEMM end-to-end ──

    def _g0_smoke_forward(self) -> dict:
        """Single-token smoke: embed → input_norm → q_proj NVFP4 GEMM.

        Doesn't exercise FA2 / RoPE / MLP / lm_head. Just proves:
          * Weights load + invariants pass.
          * `quantize_bf16_to_nvfp4_swizzled` produces a valid (packed,
            sf_swz) pair against the activation scratch.
          * `fp4_w4a16_gemm_sm120_bf16out` runs with the loaded weight
            ptrs + alpha and writes finite output.
        Returns a dict with shapes + finite-stats so the test can gate.
        """
        import torch

        from flash_rt import flash_rt_kernels as fvk

        assert self._weights is not None
        assert self._cfg is not None
        s = torch.cuda.current_stream().cuda_stream
        hidden = self._cfg['hidden_size']
        n_q = self._cfg['num_q_heads']
        hd = self._cfg['head_dim']
        eps = float(self._cfg['rms_norm_eps'])

        # 1) Embedding lookup for token id 0 (first vocab entry).
        embed_t = self._weights.anchors[0]   # (vocab, hidden) bf16
        token_id = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        h = embed_t[token_id.view(-1)].view(1, hidden).to(
            torch.bfloat16,
        ).contiguous()

        # 2) input_layernorm via plain rms_norm (w * x_normed).
        lw = self._weights.ptrs['layers'][0]
        x_norm = self._h_b[:1].view(1, hidden)
        fvk.rms_norm(
            h.data_ptr(), int(lw['input_norm_w']),
            x_norm.data_ptr(), 1, hidden, eps, s,
        )

        # 3) NVFP4 quantize the BF16 normed input to packed+swizzled SF
        #    (M=1, K=hidden=4096). Reuse the (4096, 4096) scratch — its
        #    K dim matches.
        ap, sf, _out = self._nvfp4_scratch[(4096, 4096)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_norm.data_ptr(), ap.data_ptr(), sf.data_ptr(),
            1, hidden, s,
        )

        # 4) q_proj NVFP4 W4A4 GEMM: (1, 4096) @ (4096, 4096).T → (1, 4096).
        q_out = _out[:1].view(1, n_q * hd)
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap.data_ptr(), int(lw['q_proj_packed']),
            q_out.data_ptr(),
            1, n_q * hd, hidden,
            sf.data_ptr(), int(lw['q_proj_sf']),
            float(lw['q_proj_alpha']),
            s,
        )
        torch.cuda.synchronize()

        out_f32 = q_out.float().cpu()
        return {
            'hidden': hidden,
            'q_out_shape': tuple(q_out.shape),
            'q_out_dtype': str(q_out.dtype),
            'q_out_min': float(out_f32.min()),
            'q_out_max': float(out_f32.max()),
            'q_out_mean': float(out_f32.mean()),
            'q_out_std': float(out_f32.std()),
            'all_finite': bool(torch.isfinite(out_f32).all()),
            'nan_count': int((~torch.isfinite(out_f32)).sum()),
            'q_proj_alpha': float(lw['q_proj_alpha']),
            'gsa_layer0_q': float(lw['q_proj_gsa']),
            'gsw_layer0_q': float(lw['q_proj_gsw']),
        }

    # ── S=1 decode forward ──

    def _rope_apply_inline(self, x_in, x_out, tmp, cos4, sin4):
        """Apply full-RoPE (rotary_dim == head_dim) to a Q or K tensor.

        Shapes:
          x_in / x_out / tmp : (1, S, H, head_dim) bf16
          cos4 / sin4        : (1, S, 1, head_dim/2) bf16

        Math: x_out = x_in * [cos | cos] + rotate_half(x_in) * [sin | sin]
        with rotate_half(x)[..., :half] = -x[..., half:]
             rotate_half(x)[..., half:] =  x[..., :half]

        Implementation matches the qwen36 sibling's inline pattern but
        without the partial-rotary copy of unrotated tail (here
        rotary_dim == head_dim, so every element is rotated).
        """
        import torch
        rope_dim = self._cfg['rotary_dim']
        half = rope_dim // 2
        # tmp = index_select(x_in, last, [half..rope_dim, 0..half])
        torch.index_select(x_in, -1, self._rope_rotate_idx, out=tmp)
        tmp[..., :half].neg_()
        # tmp now equals rotate_half(x_in)
        # cos4/sin4 are (1, S, 1, half); broadcast to half then concatenate
        # implicitly by applying twice.
        # Result: x_out = x_in * cos4_full + tmp * sin4_full
        # We treat the full head_dim as two halves with the SAME cos/sin.
        # Step 1: x_out[..., :half] = x_in[..., :half]*cos + tmp[..., :half]*sin
        # Step 2: x_out[..., half:] = x_in[..., half:]*cos + tmp[..., half:]*sin
        x_out_lo = x_out[..., :half]
        x_out_hi = x_out[..., half:]
        x_in_lo = x_in[..., :half]
        x_in_hi = x_in[..., half:]
        tmp_lo = tmp[..., :half]
        tmp_hi = tmp[..., half:]
        torch.mul(x_in_lo, cos4, out=x_out_lo)
        x_out_lo.addcmul_(tmp_lo, sin4)
        torch.mul(x_in_hi, cos4, out=x_out_hi)
        x_out_hi.addcmul_(tmp_hi, sin4)

    def _layer_forward_full_nvfp4(self, L: int, h_in, cos, sin,
                                    cur_pos: int,
                                    *,
                                    prequant_ap=None, prequant_sf=None,
                                    next_input_norm_w: int = 0):
        """Single Qwen3 full-attention decoder layer (S=1, NVFP4 W4A4).

        Mirrors ``Qwen36TorchFrontendRtx._layer_forward_full_nvfp4`` with
        Qwen3-specific dims and (a) no Q-output-gate split, (b) full
        RoPE (rotary_dim=128 = head_dim), (c) plain RMSNorm (no 1+w).

        Boundary fusion (opt-in via kwargs; default off):
          * If `prequant_ap` / `prequant_sf` are passed, skip step 1+2
            (input_norm + nvfp4 quant) — caller already filled them.
          * If `next_input_norm_w != 0`, replace step 16's torch.add
            (residual_2) with the fused
            `residual_add_rms_norm_to_nvfp4_swizzled_bf16` op, which
            also writes the next-layer's pre-quant ap/sf into the
            buffer the caller passed at step 1+2 (the
            `_nvfp4_scratch[(n_q*hd, hidden)]` ap/sf slot is reused
            across layers — at step 16 the current layer is done
            reading from it).
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        s = torch.cuda.current_stream().cuda_stream
        cfg = self._cfg
        assert cfg is not None
        hidden = cfg['hidden_size']
        n_q = cfg['num_q_heads']
        n_kv = cfg['num_kv_heads']
        hd = cfg['head_dim']
        inter = cfg['intermediate']
        eps = float(cfg['rms_norm_eps'])

        lw = self._weights.ptrs['layers'][L]
        h2 = h_in.view(1, hidden).contiguous()

        # 1+2) Fused input layernorm + NVFP4 quantize:
        # rms_norm_to_nvfp4_swizzled_bf16 reads h2, applies rms_norm
        # with input_norm_w, quantizes the result to NVFP4 + writes
        # swizzled SF — one launch instead of two. Skipped when the
        # caller passes prequant_ap/sf (boundary fusion: the previous
        # layer's tail already wrote them).
        if prequant_ap is not None and prequant_sf is not None:
            ap_h, sf_h = prequant_ap, prequant_sf
        else:
            ap_h, sf_h, _ = self._nvfp4_scratch[(n_q * hd, hidden)]
            fvk.rms_norm_to_nvfp4_swizzled_bf16(
                h2.data_ptr(), int(lw['input_norm_w']),
                ap_h.data_ptr(), sf_h.data_ptr(),
                1, hidden, eps, s,
            )

        # 3+4) Fused QKV NVFP4 W4A4 MMA GEMM → (1, Nq+Nk+Nv) in one
        # launch. Three (q/k/v) projections share the same activation
        # and the same alpha at every layer (verified at load time).
        # Single M=1 N=6144 K=4096 W4A4 MMA replaces 3× separate
        # GEMMs; the hand-rolled SM120 tensor-core kernel
        # (cp.async double-buffered) is ~1.77× faster than CUTLASS
        # W4A16 at this shape.
        qkv_N = int(lw['qkv_proj_N'])
        Nq = n_q * hd      # 4096
        Nk = n_kv * hd     # 1024
        # Nv = qkv_N - Nq - Nk = 1024
        qkv_out = self._qkv_fused_out
        fvk.fp4_w4a4_mma_sm120_full_n_bf16out(
            ap_h.data_ptr(), int(lw['qkv_proj_packed']),
            qkv_out.data_ptr(),
            qkv_N, hidden,
            sf_h.data_ptr(), int(lw['qkv_proj_sf']),
            float(lw['qkv_proj_alpha']),
            s,
        )
        # Slice the (1, 6144) output into q (4096), k (1024), v (1024).
        q_pre = qkv_out[:, :Nq].view(1, 1, n_q, hd)
        k_pre = qkv_out[:, Nq:Nq + Nk].view(1, 1, n_kv, hd).contiguous()
        # v_pre staged below at step 8 (after attention) since v is
        # written into V_cache directly without an intermediate; we
        # still need its slice live till then.
        v_slice = qkv_out[:, Nq + Nk:].view(1, n_kv, hd)

        # 5+6+7+8) qkv post-processing.
        if self._enable_qkv_post_fusion:
            # Two fused kernel calls replace the 17-op chain
            # (q_norm, k_norm, 2× RoPE multi-op, 3× copies).
            # q_pre / k_pre / v_slice are contiguous slices of qkv_out
            # so .data_ptr() points to the right (head, head_dim) base.
            q_pre_ptr = qkv_out[:, :Nq].data_ptr()
            k_pre_ptr = qkv_out[:, Nq:Nq + Nk].data_ptr()
            v_pre_ptr = qkv_out[:, Nq + Nk:].data_ptr()
            kv_layer_stride = self._attn.kv_layer_stride_bytes
            kv_row_stride = self._attn.kv_row_stride_bytes
            kv_slot_off = L * kv_layer_stride + cur_pos * kv_row_stride
            k_cache_dst = (
                self._attn.K_cache.data_ptr() + kv_slot_off
            )
            v_cache_dst = (
                self._attn.V_cache.data_ptr() + kv_slot_off
            )
            # Q_buf shape (1, max_q_seq, n_q, head_dim); decode S=1
            # writes into [:, 0, :, :] = base of contiguous (n_q, hd).
            q_buf_dst = self._attn.Q_buf[:, :1].data_ptr()
            fvk.qwen3_q_norm_rope_qstage_bf16(
                q_pre=q_pre_ptr, q_norm_w=int(lw['q_norm_w']),
                cos=cos.data_ptr(), sin=sin.data_ptr(),
                q_buf_dst=q_buf_dst,
                n_q_heads=n_q, eps=eps, stream=s,
            )
            fvk.qwen3_k_norm_rope_kvwrite_bf16(
                k_pre=k_pre_ptr, v_pre=v_pre_ptr,
                k_norm_w=int(lw['k_norm_w']),
                cos=cos.data_ptr(), sin=sin.data_ptr(),
                k_cache_dst=k_cache_dst,
                v_cache_dst=v_cache_dst,
                n_kv_heads=n_kv, eps=eps, stream=s,
            )
        else:
            # 5) q_norm / k_norm — per-head RMSNorm on the head_dim axis.
            q_pre_2d = q_pre.contiguous().view(n_q, hd)
            fvk.rms_norm(
                q_pre_2d.data_ptr(), int(lw['q_norm_w']),
                self._q_norm_out[:n_q].data_ptr(), n_q, hd, eps, s,
            )
            k_pre_2d = k_pre.view(n_kv, hd)
            fvk.rms_norm(
                k_pre_2d.data_ptr(), int(lw['k_norm_w']),
                self._k_norm_out[:n_kv].data_ptr(), n_kv, hd, eps, s,
            )
            # 6) Inline full-RoPE on Q/K (rotary_dim = head_dim = 128).
            q_for_rope = self._q_norm_out[:n_q].view(1, 1, n_q, hd)
            k_for_rope = self._k_norm_out[:n_kv].view(1, 1, n_kv, hd)
            cos4 = cos.view(1, 1, 1, hd // 2)
            sin4 = sin.view(1, 1, 1, hd // 2)
            q_rot = self._q_rot[:, :1]
            k_rot = self._k_rot[:, :1]
            self._rope_apply_inline(
                q_for_rope, q_rot,
                self._rope_tmp_q[:, :1], cos4, sin4,
            )
            self._rope_apply_inline(
                k_for_rope, k_rot,
                self._rope_tmp_k[:, :1], cos4, sin4,
            )
            # 7) Stage Q + write K to KV cache.
            self._attn.Q_buf[:, :1].copy_(q_rot)
            self._attn.K_cache[L, cur_pos:cur_pos + 1].copy_(
                k_rot.view(1, n_kv, hd),
            )
            # 8) V — copy v_slice into V_cache.
            self._attn.V_cache[L, cur_pos:cur_pos + 1].copy_(v_slice)

        # 9) Run attention.
        kv_seq = cur_pos + 1
        self._attn.run(
            'full', layer_idx=L, q_seq=1, kv_seq=kv_seq,
            stream=s, causal=True,
        )
        attn_out = self._attn.O_buf[:, :1]    # (1, 1, n_q, hd)

        # 10) o_proj NVFP4: K = n_q*hd = hidden, N = hidden.
        attn_2d = attn_out.reshape(1, n_q * hd).contiguous()
        ap_h2, sf_h2, _ = self._nvfp4_scratch[(hidden, hidden)]   # same shape as q
        fvk.quantize_bf16_to_nvfp4_swizzled(
            attn_2d.data_ptr(), ap_h2.data_ptr(), sf_h2.data_ptr(),
            1, n_q * hd, s,
        )
        out_op_buf = self._nvfp4_scratch[(hidden, hidden)][2]
        fvk.fp4_w4a4_mma_sm120_full_n_bf16out(
            ap_h2.data_ptr(), int(lw['o_proj_packed']),
            out_op_buf.data_ptr(),
            hidden, n_q * hd,
            sf_h2.data_ptr(), int(lw['o_proj_sf']),
            float(lw['o_proj_alpha']),
            s,
        )

        # 11+12) Fused residual_1 + post_attn_norm + NVFP4 quantize.
        # `residual_add_rms_norm_to_nvfp4_swizzled_bf16` writes
        # h_post = h_in + attn_proj AND quantizes the rms-normed
        # result to NVFP4 in one launch — collapses 3 separate ops.
        attn_proj = out_op_buf[:1].view(1, 1, hidden)
        h_post = self._res_mid[:, :1]
        ap_mlp, sf_mlp, _ = self._nvfp4_scratch[(inter, hidden)]
        fvk.residual_add_rms_norm_to_nvfp4_swizzled_bf16(
            h_in.data_ptr(), attn_proj.data_ptr(), h_post.data_ptr(),
            int(lw['post_attn_norm_w']),
            ap_mlp.data_ptr(), sf_mlp.data_ptr(),
            1, hidden, eps, s,
        )

        # 13) Fused gate+up MLP NVFP4 GEMM at M=1 N=24576 K=hidden.
        # Single launch replaces 2× separate (gate, up) GEMMs at
        # N=12288 each. Use the WIDEN tile-shape variant which our
        # iso-N benches showed wins for very-large N (lm_head 88% peak
        # BW vs 64% with default tile).
        gu_N = int(lw['gate_up_N'])
        gu_out = self._gate_up_fused_out
        fvk.fp4_w4a4_mma_sm120_full_n_bf16out(
            ap_mlp.data_ptr(), int(lw['gate_up_packed']),
            gu_out.data_ptr(),
            gu_N, hidden,
            sf_mlp.data_ptr(), int(lw['gate_up_sf']),
            float(lw['gate_up_alpha']),
            s,
        )
        gate_v = gu_out[:, :inter].view(1, inter)
        up_v = gu_out[:, inter:].view(1, inter)

        # 14+15a) silu(gate) * up + nvfp4 swizzled quant.
        ap_dn, sf_dn, _ = self._nvfp4_scratch[(hidden, inter)]
        if self._enable_silu_mul_quant_fusion:
            # One fused launch produces packed FP4 + swizzled SF
            # directly from gate/up — the bf16 silu*up intermediate
            # is never materialized to HBM.
            fvk.silu_mul_to_nvfp4_swizzled_bf16(
                gate=gate_v.data_ptr(), up=up_v.data_ptr(),
                packed=ap_dn.data_ptr(), sf_swz=sf_dn.data_ptr(),
                rows=1, cols=inter, stream=s,
            )
        else:
            fvk.silu_mul_qwen36_bf16(
                gate_v.data_ptr(), up_v.data_ptr(),
                self._mlp_silu_mul_out[:1].data_ptr(), inter, s,
            )
            fvk.quantize_bf16_to_nvfp4_swizzled(
                self._mlp_silu_mul_out[:1].data_ptr(),
                ap_dn.data_ptr(), sf_dn.data_ptr(),
                1, inter, s,
            )

        # 15b) MLP down: K=intermediate → N=hidden.
        down_out_buf = self._nvfp4_scratch[(hidden, inter)][2]
        fvk.fp4_w4a4_mma_sm120_full_n_bf16out(
            ap_dn.data_ptr(), int(lw['mlp_down_packed']),
            down_out_buf.data_ptr(),
            hidden, inter,
            sf_dn.data_ptr(), int(lw['mlp_down_sf']),
            float(lw['mlp_down_alpha']),
            s,
        )
        mlp_out = down_out_buf[:1].view(1, 1, hidden)

        # 16) Residual 2 (+ optional boundary fusion).
        # Use ping-pong layer-out buffers to keep producer/consumer
        # tensors distinct across the next layer's `h_in`.
        h_out = self._layer_out_a if (L % 2 == 0) else self._layer_out_b
        h_out_v = h_out[:, :1]
        if next_input_norm_w:
            # Fused: h_out = h_post + mlp_out  AND
            #        next ap/sf = nvfp4_quant(rms_norm(h_out, next_norm_w))
            # Reuse the (n_q*hd, hidden) NVFP4 scratch ap/sf — the
            # qkv GEMM read from it at step 2, by step 16 it's
            # consumer-done so we can safely overwrite for L+1.
            next_ap, next_sf, _ = self._nvfp4_scratch[(n_q * hd, hidden)]
            fvk.residual_add_rms_norm_to_nvfp4_swizzled_bf16(
                h_post.data_ptr(),
                mlp_out.contiguous().data_ptr(),
                h_out_v.data_ptr(),
                int(next_input_norm_w),
                next_ap.data_ptr(), next_sf.data_ptr(),
                1, hidden, eps, s,
            )
        else:
            torch.add(h_post, mlp_out, out=h_out_v)
        return h_out_v

    def forward_own_decode_nvfp4(self, token_id, cos_pos, sin_pos,
                                  cur_pos: int):
        """NVFP4 own-forward decode: 36 layers + final norm + lm_head BF16.

        Args:
          token_id: (1, 1) long, the token to decode.
          cos_pos / sin_pos: (1, head_dim/2) bf16, cos/sin at cur_pos.
          cur_pos: int, the absolute position to write K/V at.

        Writes:
          self._logits_buf[:1]  ← (1, vocab) bf16 logits for next token
          self._last_hidden_buf[:, :1] ← (1, 1, hidden) post-final-norm
            hidden state (handy for downstream tools / debug).

        Returns:
          self._logits_buf[:1].
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        bf16 = torch.bfloat16
        s = torch.cuda.current_stream().cuda_stream
        cfg = self._cfg
        assert cfg is not None
        hidden = cfg['hidden_size']
        vocab = cfg['vocab_size']
        eps = float(cfg['rms_norm_eps'])

        # 0) Embedding lookup. Anchors[0] is embed_w (vocab, hidden).
        if not isinstance(token_id, torch.Tensor):
            token_id = torch.tensor(
                [token_id], device=self.device, dtype=torch.long,
            )
        if token_id.ndim == 1:
            token_id = token_id.view(1, 1)
        embed_t = self._weights.anchors[0]
        h = embed_t[token_id.view(-1)].view(1, 1, hidden).contiguous()
        if h.dtype != bf16:
            h = h.to(bf16)

        # 1) 36 decoder layers.
        n_layers = cfg['num_hidden_layers']
        layers_ptrs = self._weights.ptrs['layers']
        if self._enable_boundary_fusion:
            # Pre-quant the layer-0 input here, then have each layer's
            # tail produce the next layer's pre-quant via the fused
            # residual+norm+quant op. Saves 1 launch per layer-boundary
            # (35 boundaries on a 36-layer model).
            n_q = cfg['num_q_heads']
            hd = cfg['head_dim']
            ap0, sf0, _ = self._nvfp4_scratch[(n_q * hd, hidden)]
            h0_v = h.view(1, hidden).contiguous()
            fvk.rms_norm_to_nvfp4_swizzled_bf16(
                h0_v.data_ptr(),
                int(layers_ptrs[0]['input_norm_w']),
                ap0.data_ptr(), sf0.data_ptr(),
                1, hidden, eps, s,
            )
            for L in range(n_layers):
                next_w = (
                    int(layers_ptrs[L + 1]['input_norm_w'])
                    if L + 1 < n_layers else 0
                )
                h = self._layer_forward_full_nvfp4(
                    L, h, cos_pos, sin_pos, cur_pos,
                    prequant_ap=ap0, prequant_sf=sf0,
                    next_input_norm_w=next_w,
                )
        else:
            for L in range(n_layers):
                h = self._layer_forward_full_nvfp4(
                    L, h, cos_pos, sin_pos, cur_pos,
                )

        # 2) Final RMSNorm (plain w) → BF16 last hidden.
        h2 = h.view(1, hidden).contiguous()
        x_norm = self._h_b[:1].view(1, hidden)
        fvk.rms_norm(
            h2.data_ptr(), int(self._weights.ptrs['final_norm_w']),
            x_norm.data_ptr(), 1, hidden, eps, s,
        )
        self._last_hidden_buf[:, :1].copy_(x_norm.view(1, 1, hidden))

        # 3) lm_head BF16 mat-vec.
        # An earlier attempt to NVFP4-quantize lm_head (~450 MiB BW
        # saved per token) regressed greedy decode quality on the
        # 152K-class argmax: full-logits cosine vs HF dropped from
        # 0.987 to 0.978 and 32-token byte match dropped from 24 to 8.
        # The W4A4 noise compounds over decode steps faster than the
        # BW saving justifies. Reverted to BF16. A future NVFP4
        # lm_head with FP8 per-row calibration could potentially
        # recover the gap.
        fvk.bf16_matmul_qwen36_bf16(
            x_norm.data_ptr(),
            int(self._weights.ptrs['lm_head_w']),
            self._logits_buf[:1].data_ptr(),
            1, vocab, hidden, s,
        )
        return self._logits_buf[:1]

    # ── D4: S=N prefill ──

    def _layer_forward_full_nvfp4_prefill(self, L: int, h_in_S, cos_S,
                                            sin_S, start_pos: int, S: int):
        """Single Qwen3 layer at S=N prefill.

        Same kernel sequence as ``_layer_forward_full_nvfp4`` but with
        all M-axis sized to S. Writes K/V[L, start_pos:start_pos+S]
        in one shot and runs FA2 causal q_seq=S kv_seq=start_pos+S.

        Args:
          h_in_S : (1, S, hidden) bf16
          cos_S  : (S, head_dim/2) bf16
          sin_S  : (S, head_dim/2) bf16
          start_pos: absolute position of the FIRST row of S
          S      : number of rows (must be ≤ max_q_seq).

        Returns:
          h_out : (1, S, hidden) bf16 (ping-pong layer-out buffer view).
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        s = torch.cuda.current_stream().cuda_stream
        cfg = self._cfg
        assert cfg is not None
        hidden = cfg['hidden_size']
        n_q = cfg['num_q_heads']
        n_kv = cfg['num_kv_heads']
        hd = cfg['head_dim']
        inter = cfg['intermediate']
        eps = float(cfg['rms_norm_eps'])
        lw = self._weights.ptrs['layers'][L]

        h2 = h_in_S.view(S, hidden).contiguous()

        # 1) input layernorm (M=S).
        x_norm = self._h_b[:S].view(S, hidden)
        fvk.rms_norm(
            h2.data_ptr(), int(lw['input_norm_w']),
            x_norm.data_ptr(), S, hidden, eps, s,
        )

        # 2) NVFP4 quantize x_norm at M=S (K=hidden).
        ap_h, sf_h, _ = self._nvfp4_scratch[(n_q * hd, hidden)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_norm.data_ptr(), ap_h.data_ptr(), sf_h.data_ptr(),
            S, hidden, s,
        )

        # 3) q/k/v_proj NVFP4 GEMMs at M=S.
        q_proj_out_buf = self._nvfp4_scratch[(n_q * hd, hidden)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_h.data_ptr(), int(lw['q_proj_packed']),
            q_proj_out_buf.data_ptr(),
            S, n_q * hd, hidden,
            sf_h.data_ptr(), int(lw['q_proj_sf']),
            float(lw['q_proj_alpha']),
            s,
        )
        kv_proj_out_buf = self._nvfp4_scratch[(n_kv * hd, hidden)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_h.data_ptr(), int(lw['k_proj_packed']),
            kv_proj_out_buf.data_ptr(),
            S, n_kv * hd, hidden,
            sf_h.data_ptr(), int(lw['k_proj_sf']),
            float(lw['k_proj_alpha']),
            s,
        )
        q_pre = q_proj_out_buf[:S].view(1, S, n_q, hd)
        k_pre = kv_proj_out_buf[:S].view(1, S, n_kv, hd).contiguous()

        # 4) q/k_norm at (S*n_q, hd) and (S*n_kv, hd) flat views.
        q_pre_flat = q_pre.contiguous().view(S * n_q, hd)
        k_pre_flat = k_pre.view(S * n_kv, hd)
        fvk.rms_norm(
            q_pre_flat.data_ptr(), int(lw['q_norm_w']),
            self._q_norm_out[:S * n_q].data_ptr(), S * n_q, hd, eps, s,
        )
        fvk.rms_norm(
            k_pre_flat.data_ptr(), int(lw['k_norm_w']),
            self._k_norm_out[:S * n_kv].data_ptr(), S * n_kv, hd, eps, s,
        )

        # 5) Inline full-RoPE (rotary_dim = head_dim).
        q_for_rope = self._q_norm_out[:S * n_q].view(1, S, n_q, hd)
        k_for_rope = self._k_norm_out[:S * n_kv].view(1, S, n_kv, hd)
        # cos_S/sin_S are (S, hd/2). Broadcast to (1, S, 1, hd/2).
        cos4 = cos_S.view(1, S, 1, hd // 2)
        sin4 = sin_S.view(1, S, 1, hd // 2)
        q_rot = self._q_rot[:, :S]
        k_rot = self._k_rot[:, :S]
        self._rope_apply_inline(
            q_for_rope, q_rot, self._rope_tmp_q[:, :S], cos4, sin4,
        )
        self._rope_apply_inline(
            k_for_rope, k_rot, self._rope_tmp_k[:, :S], cos4, sin4,
        )

        # 6) Stage Q + write K to KV cache at [start_pos:start_pos+S].
        self._attn.Q_buf[:, :S].copy_(q_rot)
        self._attn.K_cache[L, start_pos:start_pos + S].copy_(
            k_rot.view(S, n_kv, hd),
        )

        # 7) v_proj — reuse kv_proj_out_buf, write into V_cache.
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_h.data_ptr(), int(lw['v_proj_packed']),
            kv_proj_out_buf.data_ptr(),
            S, n_kv * hd, hidden,
            sf_h.data_ptr(), int(lw['v_proj_sf']),
            float(lw['v_proj_alpha']),
            s,
        )
        v_new = kv_proj_out_buf[:S].view(S, n_kv, hd)
        self._attn.V_cache[L, start_pos:start_pos + S].copy_(v_new)

        # 8) Run FA2 causal: q_seq=S, kv_seq=start_pos+S.
        kv_seq = start_pos + S
        self._attn.run(
            'full', layer_idx=L, q_seq=S, kv_seq=kv_seq,
            stream=s, causal=True,
        )
        attn_out = self._attn.O_buf[:, :S]

        # 9) o_proj NVFP4 at M=S.
        attn_2d = attn_out.reshape(S, n_q * hd).contiguous()
        ap_h2, sf_h2, _ = self._nvfp4_scratch[(hidden, hidden)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            attn_2d.data_ptr(), ap_h2.data_ptr(), sf_h2.data_ptr(),
            S, n_q * hd, s,
        )
        out_op_buf = self._nvfp4_scratch[(hidden, hidden)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_h2.data_ptr(), int(lw['o_proj_packed']),
            out_op_buf.data_ptr(),
            S, hidden, n_q * hd,
            sf_h2.data_ptr(), int(lw['o_proj_sf']),
            float(lw['o_proj_alpha']),
            s,
        )

        # 10) Residual 1.
        attn_proj = out_op_buf[:S].view(1, S, hidden)
        torch.add(h_in_S, attn_proj, out=self._res_mid[:, :S])
        h_post = self._res_mid[:, :S]

        # 11) post-attn layernorm at M=S.
        h_post_view = h_post.view(S, hidden)
        x_mlp = self._h_b[:S].view(S, hidden)
        fvk.rms_norm(
            h_post_view.data_ptr(), int(lw['post_attn_norm_w']),
            x_mlp.data_ptr(), S, hidden, eps, s,
        )

        # 12) MLP gate / up at M=S, K=hidden, N=intermediate.
        ap_mlp, sf_mlp, _ = self._nvfp4_scratch[(inter, hidden)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_mlp.data_ptr(), ap_mlp.data_ptr(), sf_mlp.data_ptr(),
            S, hidden, s,
        )
        gate_out_buf = self._nvfp4_scratch[(inter, hidden)][2]
        up_out_buf = self._mlp_up_out
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_mlp.data_ptr(), int(lw['mlp_gate_packed']),
            gate_out_buf.data_ptr(),
            S, inter, hidden,
            sf_mlp.data_ptr(), int(lw['mlp_gate_sf']),
            float(lw['mlp_gate_alpha']),
            s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_mlp.data_ptr(), int(lw['mlp_up_packed']),
            up_out_buf.data_ptr(),
            S, inter, hidden,
            sf_mlp.data_ptr(), int(lw['mlp_up_sf']),
            float(lw['mlp_up_alpha']),
            s,
        )

        # 13) silu(gate)*up at flat n_elements = S*inter (kernel is dim-agnostic).
        gate_v = gate_out_buf[:S].view(S * inter)
        up_v = up_out_buf[:S].view(S * inter)
        out_v = self._mlp_silu_mul_out[:S].view(S * inter)
        fvk.silu_mul_qwen36_bf16(
            gate_v.data_ptr(), up_v.data_ptr(),
            out_v.data_ptr(), S * inter, s,
        )
        gate_silu_up = self._mlp_silu_mul_out[:S]

        # 14) MLP down at M=S, K=intermediate, N=hidden.
        ap_dn, sf_dn, _ = self._nvfp4_scratch[(hidden, inter)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            gate_silu_up.data_ptr(), ap_dn.data_ptr(), sf_dn.data_ptr(),
            S, inter, s,
        )
        down_out_buf = self._nvfp4_scratch[(hidden, inter)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_dn.data_ptr(), int(lw['mlp_down_packed']),
            down_out_buf.data_ptr(),
            S, hidden, inter,
            sf_dn.data_ptr(), int(lw['mlp_down_sf']),
            float(lw['mlp_down_alpha']),
            s,
        )
        mlp_out = down_out_buf[:S].view(1, S, hidden)

        # 15) Residual 2 → ping-pong.
        h_out = self._layer_out_a if (L % 2 == 0) else self._layer_out_b
        h_out_v = h_out[:, :S]
        torch.add(h_post, mlp_out, out=h_out_v)
        return h_out_v

    def forward_prefill_nvfp4(self, prompt_ids, start_pos: int = 0,
                                *, full_logits: bool = False):
        """S=N prefill: process the whole prompt in one batched forward.

        Writes K/V[layer, start_pos:start_pos+S] for every layer in one
        shot and runs FA2 in causal mode at q_seq=S, kv_seq=start_pos+S.
        ``cur_pos`` is advanced internally.

        Args:
          prompt_ids : (1, S) long  on device.
          start_pos : absolute position of the first prompt token. 0 for
            a fresh KV cache; >0 if continuing an existing context.
          full_logits: if False (default), lm_head runs only on the
            last row → ``self._logits_buf[:1]`` (1, vocab). If True,
            lm_head runs on every row → ``self._logits_buf[:S]``
            (S, vocab). The True branch is used by the lookup-spec
            verify path which needs argmax per row.

        Writes:
          self._logits_buf[:S if full_logits else 1] ← bf16 logits
          self._last_hidden_buf[:, :S] ← per-row post-final-norm hidden

        Returns:
          self._logits_buf slice (last row or all S rows).

        Note: full_logits=False saves ~(S-1) × vocab × hidden BF16
        reads per prefill (~1 ms saved at S=10, vocab=152K, K=4096).
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        bf16 = torch.bfloat16
        s = torch.cuda.current_stream().cuda_stream
        cfg = self._cfg
        assert cfg is not None
        hidden = cfg['hidden_size']
        vocab = cfg['vocab_size']
        eps = float(cfg['rms_norm_eps'])
        rope_dim = cfg['rotary_dim']

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.view(1, -1)
        S = int(prompt_ids.shape[1])
        if S < 1 or S > self.max_q_seq:
            raise ValueError(
                f'prefill S={S} out of [1, {self.max_q_seq}]; bump '
                f'max_q_seq at construction or use a smaller prompt.'
            )
        if start_pos + S > self.max_seq:
            raise ValueError(
                f'prefill end {start_pos + S} > max_seq {self.max_seq}'
            )

        # 0) Embed S tokens.
        embed_t = self._weights.anchors[0]
        h = embed_t[prompt_ids.view(-1)].view(1, S, hidden).contiguous()
        if h.dtype != bf16:
            h = h.to(bf16)

        # 1) cos/sin for absolute positions [start_pos, start_pos+S).
        cos_S = self._rope_cos_table[start_pos:start_pos + S]
        sin_S = self._rope_sin_table[start_pos:start_pos + S]
        # shape (S, rope_dim/2) bf16

        # 2) 36 layers.
        n_layers = cfg['num_hidden_layers']
        for L in range(n_layers):
            h = self._layer_forward_full_nvfp4_prefill(
                L, h, cos_S, sin_S, start_pos, S,
            )

        # 3) Final RMSNorm at M=S.
        h2 = h.view(S, hidden).contiguous()
        x_norm = self._h_b[:S].view(S, hidden)
        fvk.rms_norm(
            h2.data_ptr(), int(self._weights.ptrs['final_norm_w']),
            x_norm.data_ptr(), S, hidden, eps, s,
        )
        self._last_hidden_buf[:, :S].copy_(x_norm.view(1, S, hidden))

        # 4) lm_head BF16. M=1 (last row, default) or M=S (full_logits).
        if full_logits:
            fvk.bf16_matmul_qwen36_bf16(
                x_norm.contiguous().data_ptr(),
                int(self._weights.ptrs['lm_head_w']),
                self._logits_buf[:S].data_ptr(),
                S, vocab, hidden, s,
            )
            ret = self._logits_buf[:S]
        else:
            last_row = x_norm[S - 1:S].contiguous()
            fvk.bf16_matmul_qwen36_bf16(
                last_row.data_ptr(),
                int(self._weights.ptrs['lm_head_w']),
                self._logits_buf[:1].data_ptr(),
                1, vocab, hidden, s,
            )
            ret = self._logits_buf[:1]

        # Advance the logical decode cursor to right after the prompt.
        self._cur_pos = start_pos + S
        _ = rope_dim
        return ret

    # ── D5: CUDA Graph capture for decode S=1 ──

    def _ensure_decode_graph(self, cur_pos: int):
        """Lazy-capture a CUDA Graph for forward_own_decode_nvfp4 at cur_pos.

        The graph reads from ``self._static_token_id`` (1, 1) long and
        writes to fixed buffers (KV cache slot at cur_pos, Q/O scratch,
        nvfp4 scratch dict, logits). Each cur_pos gets its own graph
        because FA2 kv_seq = cur_pos+1 is baked into the captured
        kernel's int args; cos/sin slices are also cur_pos-specific.

        Side-effect: the warmup/capture writes to KV cache row
        [cur_pos:cur_pos+1] for every layer. Since the caller is about
        to write that row anyway (this IS the decode for cur_pos), no
        snap/restore is needed — the capture's writes are the writes
        the caller wants.
        """
        import torch

        g = self._captured_decode_graphs.get(cur_pos)
        if g is not None:
            return g

        gs = self._graph_stream
        cos, sin = self._rope_cos_sin(cur_pos)
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.no_grad():
            # Warmup runs (qwen36 sibling does 2). Some kernels (e.g.
            # cuBLAS) cache tactic on first call; warmup ensures the
            # capture sees the steady-state launch sequence.
            for _ in range(2):
                self.forward_own_decode_nvfp4(
                    self._static_token_id, cos, sin, cur_pos,
                )
        gs.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=gs), torch.no_grad():
            self.forward_own_decode_nvfp4(
                self._static_token_id, cos, sin, cur_pos,
            )
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)

        self._captured_decode_graphs[cur_pos] = g
        return g

    def decode_step_with_graph(self, token_id, cur_pos: int):
        """Single-step decode at cur_pos using a captured graph.

        ``token_id`` may be (1,1) long tensor or a Python int. The
        graph is lazily captured on first call at this cur_pos.
        Returns ``self._logits_buf[:1]`` (1, vocab) bf16.
        """
        import torch
        if not isinstance(token_id, torch.Tensor):
            token_id = torch.tensor(
                [[int(token_id)]], device=self.device, dtype=torch.long,
            )
        if token_id.ndim == 1:
            token_id = token_id.view(1, 1)
        # Stage the new token id into the static input buffer the
        # captured graph reads from.
        self._static_token_id.copy_(token_id)
        g = self._ensure_decode_graph(cur_pos)
        g.replay()
        return self._logits_buf[:1]

    def warmup_decode_graphs(self, cur_pos_lo: int, cur_pos_hi: int) -> None:
        """Pre-capture decode graphs for cur_pos in [lo, hi).

        Recommended at server startup: call once per (prompt_len,
        prompt_len + max_new_tokens) range so the first real request
        replays warm graphs all the way through. Captures sequentially;
        each capture takes ~3-10 ms on a warm allocator.
        """
        for p in range(cur_pos_lo, cur_pos_hi):
            self._ensure_decode_graph(p)

    # ── P1-c: CUDA Graph capture for prefill S=N ──

    def _prefill_layers_into_last_hidden(self, prompt_ids, S: int) -> None:
        """Body of forward_prefill_nvfp4 minus lm_head and minus _cur_pos
        update. Reads from ``prompt_ids`` (caller-owned, must be a (1, S)
        long tensor on device — the captured graph reads from the static
        slice ``self._static_prompt_ids[:, :S]``). Writes
        ``self._last_hidden_buf[:, :S]`` with the post-final-norm hidden.

        start_pos is hard-coded to 0: the prefill graph is only used for
        a fresh KV cache (the OAI server case). Continue-prefill (>0)
        keeps using the eager forward_prefill_nvfp4 path.

        Mirrors forward_prefill_nvfp4 lines 1010-1035 exactly so
        capture and eager paths produce bit-identical hidden state for
        the same prompt_ids contents. The lm_head split into eager is
        what lets one captured graph serve all real-S ≤ S_bucket calls
        within a bucket: only the last-real-row's logits depend on
        real_S, so we compute them eagerly post-replay against the real
        row of ``self._last_hidden_buf``.
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        bf16 = torch.bfloat16
        s = torch.cuda.current_stream().cuda_stream
        cfg = self._cfg
        assert cfg is not None
        hidden = cfg['hidden_size']
        eps = float(cfg['rms_norm_eps'])

        embed_t = self._weights.anchors[0]
        h = embed_t[prompt_ids.view(-1)].view(1, S, hidden).contiguous()
        if h.dtype != bf16:
            h = h.to(bf16)

        cos_S = self._rope_cos_table[0:S]
        sin_S = self._rope_sin_table[0:S]

        n_layers = cfg['num_hidden_layers']
        for L in range(n_layers):
            h = self._layer_forward_full_nvfp4_prefill(
                L, h, cos_S, sin_S, 0, S,
            )

        h2 = h.view(S, hidden).contiguous()
        x_norm = self._h_b[:S].view(S, hidden)
        fvk.rms_norm(
            h2.data_ptr(), int(self._weights.ptrs['final_norm_w']),
            x_norm.data_ptr(), S, hidden, eps, s,
        )
        self._last_hidden_buf[:, :S].copy_(x_norm.view(1, S, hidden))

    def _ensure_prefill_graph(self, S_bucket: int):
        """Lazy-capture a CUDA Graph for fresh-KV prefill at S=S_bucket.

        Reads from ``self._static_prompt_ids[:, :S_bucket]`` and writes
        ``self._last_hidden_buf[:, :S_bucket]``. Each bucket gets its own
        graph because every kernel's M-axis size is baked in. lm_head is
        intentionally NOT in the graph (see _prefill_layers_into_last_hidden
        docstring). KV cache rows [0, S_bucket) are written for every
        layer; rows beyond real_S are bogus but never read by the
        subsequent decode (decode at cur_pos=real_S+k overwrites K[real_S+k]
        before reading it).

        The 2-iter warmup matches the decode-graph pattern. Capture cost
        is ~5-15 ms per bucket on a warm allocator; the OAI server amortizes
        this via warmup_prefill_graphs at startup.
        """
        import torch

        g = self._captured_prefill_graphs.get(S_bucket)
        if g is not None:
            return g

        if S_bucket < 1 or S_bucket > self.max_q_seq:
            raise ValueError(
                f'S_bucket={S_bucket} out of [1, {self.max_q_seq}]'
            )

        gs = self._graph_stream
        static_in = self._static_prompt_ids[:, :S_bucket]
        gs.wait_stream(torch.cuda.current_stream())
        # inference_mode is entered BEFORE torch.cuda.graph because
        # cuda.graph's __enter__ calls capture_begin() which itself
        # touches buffers — if any pre-allocated buffer was flagged
        # as an inference tensor by a prior eager
        # forward_prefill_nvfp4 call (which is the typical caller
        # pattern: prefill once then warm graphs), capture_begin
        # would error with "Inplace update to inference tensor
        # outside InferenceMode". Decode warmup doesn't hit this in
        # the standard test path because decode buffers were
        # allocated and only ever written from inside the same
        # inference_mode block.
        with torch.inference_mode(), torch.cuda.stream(gs):
            for _ in range(2):
                self._prefill_layers_into_last_hidden(static_in, S_bucket)
        gs.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g, stream=gs):
            self._prefill_layers_into_last_hidden(static_in, S_bucket)
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)

        self._captured_prefill_graphs[S_bucket] = g
        return g

    def _select_prefill_bucket(self, real_S: int) -> int | None:
        """Smallest bucket >= real_S, or None if real_S exceeds the
        largest bucket (caller falls back to eager).
        """
        for b in self.prefill_buckets:
            if b >= real_S:
                return b
        return None

    def prefill_with_graph(self, prompt_ids):
        """Fresh-KV prefill via a captured CUDA Graph. Returns
        ``self._logits_buf[:1]`` (1, vocab) bf16.

        Routing rule:
          * real_S = prompt_ids length.
          * If real_S > max(prefill_buckets): fall back to eager
            forward_prefill_nvfp4(prompt_ids, start_pos=0).
          * Otherwise pick the smallest bucket >= real_S, copy
            prompt_ids into the static buffer, **pad the tail with
            prompt_ids[..., -1]** (last real token; safe under causal
            masking — padded rows can never affect real rows), replay
            the graph, then run lm_head eagerly on
            ``self._last_hidden_buf[:, real_S - 1]``.

        Sets ``self._cur_pos = real_S`` post-replay (matches eager
        forward_prefill_nvfp4 semantics — decode resumes at real_S).
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.view(1, -1)
        real_S = int(prompt_ids.shape[1])
        if real_S < 1:
            raise ValueError('prompt_ids must contain at least 1 token')

        bucket = self._select_prefill_bucket(real_S)
        if bucket is None:
            # Prompt longer than any captured bucket — eager fallback.
            # forward_prefill_nvfp4 sets _cur_pos and writes _logits_buf[:1].
            return self.forward_prefill_nvfp4(prompt_ids, start_pos=0)

        # Stage prompt_ids into the static buffer with tail padding.
        slot = self._static_prompt_ids[:, :bucket]
        slot[:, :real_S].copy_(prompt_ids)
        if real_S < bucket:
            # Pad with the last real token id. Causal mask makes the
            # actual value irrelevant for real-row outputs; we still
            # use a valid id so the embed lookup never sees an OOB
            # vocabulary index (which would produce a garbage embedding
            # vector and could trip downstream NaN guards).
            pad_id = prompt_ids[0, real_S - 1]
            slot[0, real_S:bucket].fill_(pad_id.item())

        g = self._ensure_prefill_graph(bucket)
        g.replay()

        # Eager lm_head on the real-last-row of the captured-out hidden.
        cfg = self._cfg
        assert cfg is not None
        hidden = cfg['hidden_size']
        vocab = cfg['vocab_size']
        s = torch.cuda.current_stream().cuda_stream
        last_row = self._last_hidden_buf[0, real_S - 1:real_S].contiguous()
        fvk.bf16_matmul_qwen36_bf16(
            last_row.data_ptr(),
            int(self._weights.ptrs['lm_head_w']),
            self._logits_buf[:1].data_ptr(),
            1, vocab, hidden, s,
        )
        _ = bucket  # silence unused warning in IDE diff view

        # Decode resumes at the real prompt's end, NOT the bucket end.
        self._cur_pos = real_S
        return self._logits_buf[:1]

    def warmup_prefill_graphs(
        self, buckets: tuple[int, ...] | None = None,
    ) -> None:
        """Pre-capture prefill graphs for the listed buckets (default:
        the full self.prefill_buckets ladder). Recommended at server
        startup so the first real prefill at any bucketed length hits
        a warm graph.
        """
        for b in (buckets if buckets is not None else self.prefill_buckets):
            if 1 <= b <= self.max_q_seq:
                self._ensure_prefill_graph(b)

    # ── Driver: greedy decode (used by the OAI server) ──

    # ── n-gram lookup speculative decode ───────────────────

    def _lookup_2gram(self, ids: list, key, K_spec: int):
        """Find a (a,b)-2gram in `ids` history; return next K_spec tokens.

        Most-recent occurrence wins (better local-context match for
        repetitive emissions like JSON/tool_call). Returns None if no
        match has K_spec following tokens.
        """
        n = len(ids)
        a, b = key
        for i in range(n - 2 - K_spec, -1, -1):
            if ids[i] == a and ids[i + 1] == b:
                return ids[i + 2:i + 2 + K_spec]
        return None

    def lookup_decode_from_prompt(self, prompt_ids, max_new_tokens: int,
                                    *, K_spec: int = 2,
                                    eos_token_id: int | None = None):
        """Greedy decode with n-gram-lookup speculative acceleration.

        Maintains a running 2-gram → next-K_spec-tokens lookup over
        the prompt + already-generated tokens. After each accepted
        token, looks up the trailing 2-gram; if found, runs a verify
        forward at S=1+K_spec via the prefill path with full per-row
        logits; accepts the longest matching prefix.

        Acceptance dynamics:
          * On HIT (all K_spec speculation correct): saves K_spec
            decode steps, costs 1 verify forward (~ S=1+K_spec
            prefill cost ≈ slightly more than 1 decode step in
            practice given graph-warm decode).
          * On MISS: cost = 1 verify forward (replaces 1 decode step
            but is heavier; net cost ~1.5× a decode step).

        Empirically wins on highly-repetitive traffic (tool calls,
        structured JSON, code completion, system-prompt echoing).
        On free-form prose acceptance rate is low and total wall
        time may regress slightly — keep this path opt-in.

        Args:
          prompt_ids: (1, P) long.
          max_new_tokens: cap on new tokens.
          K_spec: speculation chain length per attempt (1..4 typical).
          eos_token_id: if set, stop early when emitted.

        Returns: (1, P + new) long, full sequence on cuda.
        """
        import torch

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.view(1, -1)
        prompt_len = int(prompt_ids.shape[1])
        if prompt_len + max_new_tokens > self.max_seq:
            raise ValueError(
                f'prompt + max_new ({prompt_len} + {max_new_tokens}) '
                f'exceeds max_seq={self.max_seq}'
            )
        if K_spec < 1 or 1 + K_spec > self.max_q_seq:
            raise ValueError(
                f'K_spec={K_spec} requires max_q_seq ≥ {1 + K_spec} '
                f'(got max_q_seq={self.max_q_seq})'
            )

        self.reset_state()
        ids_list = prompt_ids[0].tolist()

        # Phase A: prefill prompt — populates KV[0:P) and writes the
        # next-token logits for position P into _logits_buf[0].
        with torch.inference_mode():
            self.forward_prefill_nvfp4(prompt_ids, start_pos=0)

        # Pull the first generated token from the prefill logits and
        # advance state via the warm decode graph (writes K/V[P]).
        cur_pos = prompt_len
        with torch.inference_mode():
            tok = int(self._logits_buf[:1].argmax(dim=-1).item())
        ids_list.append(tok)
        if eos_token_id is not None and tok == eos_token_id:
            return torch.tensor([ids_list], device=self.device)

        # Stats (returned for debug / bench).
        n_attempts = 0
        n_accepted_extra = 0  # spec tokens accepted beyond the always-1

        while len(ids_list) - prompt_len < max_new_tokens:
            # Last accepted token is at ids_list[-1]; KV[cur_pos] not
            # yet written. Need to either:
            #  (a) issue a verify forward S=1+K_spec at start_pos=cur_pos
            #      with input [tok, spec[0], ..., spec[K-1]], OR
            #  (b) fall back to a single decode_step_with_graph.
            # We try (a) iff a 2-gram match exists.
            spec = None
            if len(ids_list) >= 2 and K_spec >= 1:
                key = (ids_list[-2], ids_list[-1])
                spec = self._lookup_2gram(ids_list, key, K_spec)

            with torch.inference_mode():
                if spec is None:
                    # No spec: standard graph-warm decode step.
                    self.decode_step_with_graph(
                        torch.tensor([[ids_list[-1]]],
                                     device=self.device, dtype=torch.long),
                        cur_pos,
                    )
                    next_tok = int(
                        self._logits_buf[:1].argmax(dim=-1).item()
                    )
                    ids_list.append(next_tok)
                    cur_pos += 1
                    if eos_token_id is not None and next_tok == eos_token_id:
                        break
                    continue

                # Verify path:
                #   input = [last_accepted_token, spec[0], spec[1], ...]
                S_verify = 1 + K_spec
                verify_in = torch.tensor(
                    [[ids_list[-1]] + spec],
                    device=self.device, dtype=torch.long,
                )
                logits_S = self.forward_prefill_nvfp4(
                    verify_in, start_pos=cur_pos, full_logits=True,
                )
                # logits_S: (S_verify, vocab). argmax_S[r] is the
                # model's predicted token at position cur_pos+r+1
                # (given verify input was correct at cur_pos..cur_pos+r).
                argmax_S = logits_S.argmax(dim=-1).tolist()

                # Algorithm:
                #   * argmax_S[0] is the true next token at cur_pos+1
                #     (no speculation needed) — ALWAYS emit.
                #   * For r in 0..K_spec-1: spec[r] was OUR guess for
                #     position cur_pos+r+1. It is correct iff
                #     argmax_S[r] == spec[r] (the model would have
                #     emitted the same thing). If correct, the chain
                #     to position cur_pos+r+2 is intact, and
                #     argmax_S[r+1] is the (true) next token there;
                #     emit it. Otherwise break.
                new_tokens = [argmax_S[0]]
                for r in range(K_spec):
                    if argmax_S[r] == spec[r]:
                        new_tokens.append(argmax_S[r + 1])
                    else:
                        break
                ids_list.extend(new_tokens)
                accepted = len(new_tokens)
                cur_pos += accepted
                n_attempts += 1
                n_accepted_extra += (accepted - 1)
                if eos_token_id is not None:
                    if eos_token_id in ids_list[-accepted:]:
                        # truncate to first eos
                        for j in range(accepted):
                            tail = ids_list[-accepted + j]
                            if tail == eos_token_id:
                                ids_list = ids_list[
                                    :len(ids_list) - accepted + j + 1
                                ]
                                break
                        break

        self._last_lookup_stats = {
            'attempts': n_attempts,
            'accepted_extra': n_accepted_extra,
            'avg_accepted': (
                1 + n_accepted_extra / n_attempts if n_attempts else 1.0
            ),
            'max_spec': K_spec,
        }

        # Trim to max_new_tokens (we may overshoot by spec acceptance).
        new_count = len(ids_list) - prompt_len
        if new_count > max_new_tokens:
            ids_list = ids_list[:prompt_len + max_new_tokens]
        return torch.tensor([ids_list], device=self.device)

    def greedy_decode_from_prompt(self, prompt_ids, max_new_tokens: int,
                                    *, use_prefill: bool = True,
                                    use_graph: bool = True,
                                    use_prefill_graph: bool = False):
        """Greedy autoregressive decode over a prompt.

        Args:
          prompt_ids : (1, S) long.
          max_new_tokens : new tokens to generate.
          use_prefill : if True (default), use the S=N prefill path
            for prompt ingest; if False, fall back to a per-token
            S=1 ingest loop (used by the prefill-vs-S=1 cosine
            equivalence test).
          use_prefill_graph : if True, route prompt ingest through
            prefill_with_graph. Caller is responsible for having run
            warmup_prefill_graphs at startup so the first request
            doesn't pay capture cost. Defaults to False to preserve
            the eager-prefill greedy-decode behavior; opt-in once
            the caller's startup pre-captures the prefill graphs.
        """
        import torch

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.view(1, -1)
        assert prompt_ids.shape[0] == 1, 'batch=1 only'
        prompt_len = int(prompt_ids.shape[1])
        if prompt_len + max_new_tokens > self.max_seq:
            raise ValueError(
                f'prompt + max_new ({prompt_len} + {max_new_tokens}) '
                f'exceeds max_seq={self.max_seq}'
            )

        self.reset_state()
        ids_list = prompt_ids[0].tolist()

        with torch.inference_mode():
            # Phase A: ingest prompt.
            if use_prefill and prompt_len <= self.max_q_seq:
                if use_prefill_graph:
                    self.prefill_with_graph(prompt_ids)
                else:
                    self.forward_prefill_nvfp4(prompt_ids, start_pos=0)
            else:
                for p in range(prompt_len):
                    self._static_token_id.copy_(prompt_ids[:, p:p + 1])
                    cos, sin = self._rope_cos_sin(p)
                    self.forward_own_decode_nvfp4(
                        self._static_token_id, cos, sin, p,
                    )
            # Phase B: generate.
            for _ in range(max_new_tokens):
                tok = self._logits_buf[:1].argmax(
                    dim=-1, keepdim=True,
                ).view(1, 1)
                ids_list.append(int(tok.item()))
                cur = len(ids_list) - 1
                if use_graph:
                    self.decode_step_with_graph(tok, cur)
                else:
                    self._static_token_id.copy_(tok)
                    cos, sin = self._rope_cos_sin(cur)
                    self.forward_own_decode_nvfp4(
                        self._static_token_id, cos, sin, cur,
                    )
        return torch.tensor([ids_list], device=self.device)
