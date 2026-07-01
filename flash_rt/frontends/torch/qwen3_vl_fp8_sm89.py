"""Qwen3-VL official-FP8 text decode path for RTX SM89.

This frontend targets the language stack inside the official
Qwen3-VL-8B-Instruct-FP8 checkpoint. It is intentionally text-only for the
first SM89 bring-up: no vision tower, no multimodal scatter, and no fallback
route. The hot linears use block-scaled M=1 FP8 GEMV against the checkpoint's
``weight`` + ``weight_scale_inv`` tensors.
"""
from __future__ import annotations

import collections
import os
from typing import Any


class _MergedKernels:
    """Resolve kernel symbols from the dedicated flash_rt_qwen3_vl_kernels
    module first, then fall back to the generic flash_rt_kernels. Lets the SM89
    Qwen3-VL FP8 kernels live in their own .so (mirroring the SM120 split)
    while frontend call sites keep using a single ``fvk`` handle."""

    __slots__ = ('_vl', '_core')

    def __init__(self, vl: Any, core: Any) -> None:
        self._vl = vl
        self._core = core

    def __getattr__(self, name: str) -> Any:
        if self._vl is not None and hasattr(self._vl, name):
            return getattr(self._vl, name)
        return getattr(self._core, name)


def _import_fvk() -> Any:
    """flash_rt_kernels (generic) + flash_rt_qwen3_vl_kernels (SM89 FP8)."""
    from flash_rt import flash_rt_kernels as core
    try:
        from flash_rt import flash_rt_qwen3_vl_kernels as vl
    except ImportError:
        vl = None
    return _MergedKernels(vl, core)


def _resolve_max_prefill_seq(max_seq: int,
                             max_prefill_seq: int | None) -> int:
    return int(max_seq) if max_prefill_seq is None else int(max_prefill_seq)


class Qwen3VlFp8Sm89TextFrontend:
    """Batch-1 Qwen3-VL text-only decode on SM89 official FP8 weights.

    Supports any Qwen3-VL language stack meeting the block-128 FP8 kernel
    constraints (head_dim == 128, all GEMM dims a multiple of 128). Validated
    on the official Qwen3-VL-8B-Instruct-FP8 checkpoint and on a block-128
    quantized Qwen3-VL-2B checkpoint.
    """

    def __init__(self, checkpoint_path: str, *,
                 device: str = 'cuda:0', max_seq: int = 2048,
                 max_prefill_seq: int | None = None,
                 fuse_gate_up: bool = False,
                 fuse_qk_postproc: bool = True,
                 use_fp8_lm_head: bool = False,
                 max_decode_graphs: int | None = None) -> None:
        import json

        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.max_seq = int(max_seq)
        self.max_prefill_seq = _resolve_max_prefill_seq(
            self.max_seq, max_prefill_seq)
        self.fuse_gate_up = bool(fuse_gate_up)
        self.fuse_qk_postproc = bool(fuse_qk_postproc)
        self.use_fp8_lm_head = bool(use_fp8_lm_head)
        self._tokenizer: Any = None
        self._weights = None
        self._cfg: dict | None = None
        self._attn = None
        self._cur_pos = 0
        if max_decode_graphs is None:
            max_decode_graphs = int(os.environ.get(
                'FLASHRT_QWEN3_VL_DECODE_GRAPH_CACHE_MAX', '256'))
        self.max_decode_graphs = int(max_decode_graphs)
        self._decode_graphs: collections.OrderedDict[int, Any] = (
            collections.OrderedDict())

        import torch
        fvk = _import_fvk()

        device_obj = torch.device(self.device)
        if device_obj.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError(
                'Qwen3VlFp8Sm89TextFrontend requires an SM89 CUDA device')
        cap = torch.cuda.get_device_capability(device_obj)
        if cap != (8, 9):
            raise RuntimeError(
                'Qwen3VlFp8Sm89TextFrontend requires GPU_ARCH=89 / sm_89; '
                f'got sm_{cap[0]}{cap[1]} on {self.device}')
        required = (
            'fp8_block128_gemm_blockscaled_sm89_bf16out',
            'ht_gemv_fp8_block128_m1_w8',
            'ht_gemv_fp8_block128_m1_w16',
            'fp8_per_token_block128_quant_bf16',
            'rms_norm_to_fp8_block128_bf16',
            'residual_add_rms_norm_to_fp8_block128_bf16',
            'silu_mul_to_fp8_block128_bf16',
            'silu_mul_merged_to_fp8_block128_bf16',
            'qwen3_qk_norm_rope_kvwrite_bf16',
            'qwen3_qk_norm_rope_kvwrite_batched_bf16',
            'qwen3_q_norm_rope_qstage_bf16',
            'qwen3_k_norm_rope_kvwrite_bf16',
            'embedding_lookup_bf16',
            'bf16_matmul_bf16',
            'rms_norm',
            'residual_add',
        )
        missing = [name for name in required if not hasattr(fvk, name)]
        if missing:
            raise RuntimeError(
                'flash_rt_kernels / flash_rt_qwen3_vl_kernels are missing SM89 '
                'Qwen3-VL FP8 symbols: '
                + ', '.join(missing)
                + '. Rebuild with -DGPU_ARCH=89 -DFLASHRT_BUILD_QWEN3_VL=ON '
                'and build flash_rt_kernels flash_rt_fa2 '
                'flash_rt_qwen3_vl_kernels.')
        self._fvk = fvk

        cfg_path = os.path.join(self.checkpoint_path, 'config.json')
        cfg = json.load(open(cfg_path))
        text_cfg = cfg['text_config']
        # This path supports Qwen3-VL language stacks whose dimensions satisfy
        # the SM89 block-128 FP8 kernel constraints and whose checkpoint uses
        # the official/block-128 FP8 tensor layout loaded below. The config
        # requirements are: head_dim == 128 (the fused qk-norm/RoPE/KV-write
        # kernels hardcode it), every GEMM N/K dimension a multiple of 128
        # (block-128 act/weight scaling), and num_q_heads a multiple of
        # num_kv_heads (GQA). Other language geometry is read from config.
        n_q = int(text_cfg['num_attention_heads'])
        n_kv = int(text_cfg['num_key_value_heads'])
        head_dim = int(text_cfg.get('head_dim')
                       or text_cfg['hidden_size'] // n_q)
        hidden = int(text_cfg['hidden_size'])
        inter = int(text_cfg['intermediate_size'])
        vocab = int(text_cfg['vocab_size'])
        qkv_N = (n_q + 2 * n_kv) * head_dim
        problems = []
        if head_dim != 128:
            problems.append(f'head_dim={head_dim} (kernels require 128)')
        if n_kv == 0 or (n_q % n_kv) != 0:
            problems.append(
                f'num_q_heads={n_q} not a multiple of num_kv_heads={n_kv}')
        for name, dim in (('hidden_size', hidden), ('intermediate_size', inter),
                          ('vocab_size', vocab), ('qkv_proj_N', qkv_N)):
            if (dim % 128) != 0:
                problems.append(f'{name}={dim} not a multiple of 128')
        if problems:
            raise RuntimeError(
                'Qwen3VlFp8Sm89TextFrontend requires Qwen3-VL config '
                'dimensions compatible with the SM89 block-128 FP8 kernels: '
                + '; '.join(problems) + f' (from {cfg_path})')

        self._load_fp8_path()
        self._alloc_buffers()
        self._build_rope_table()

    def _graph_cache_get(self, cache, key):
        graph = cache.get(key)
        if graph is not None and isinstance(cache, collections.OrderedDict):
            cache.move_to_end(key)
        return graph

    def _trim_lru_graph_cache(self, cache, max_entries: int) -> None:
        if max_entries <= 0 or not isinstance(cache, collections.OrderedDict):
            return
        while len(cache) > max_entries:
            cache.popitem(last=False)

    def clear_graphs(self) -> None:
        if self._decode_graphs:
            self._decode_graphs.clear()

    def graph_cache_stats(self) -> dict[str, Any]:
        return {
            'decode': {
                'max_graphs': self.max_decode_graphs,
                'graph_count': len(self._decode_graphs),
                'graph_keys': list(self._decode_graphs.keys()),
            },
        }

    def _load_fp8_path(self) -> None:
        from transformers import AutoTokenizer

        from flash_rt.frontends.torch._qwen3_vl_fp8_weights import (
            assert_extraction_invariants_qwen3_vl_fp8,
            extract_weights_qwen3_vl_fp8,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        handles = extract_weights_qwen3_vl_fp8(
            self.checkpoint_path, device=self.device,
            quantize_lm_head=self.use_fp8_lm_head)
        assert_extraction_invariants_qwen3_vl_fp8(handles)
        self._weights = handles
        p = handles.ptrs
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
            'rotary_dim': int(p['head_dim']),
        }

    def _alloc_buffers(self) -> None:
        import torch

        from flash_rt.hardware.rtx.attn_backend_qwen3 import (
            RtxFlashAttnBackendQwen3,
        )

        cfg = self._cfg
        assert cfg is not None
        device = torch.device(self.device)
        bf16 = torch.bfloat16
        f8 = torch.float8_e4m3fn
        fp32 = torch.float32
        hidden = cfg['hidden_size']
        vocab = cfg['vocab_size']
        n_q = cfg['num_q_heads']
        n_kv = cfg['num_kv_heads']
        hd = cfg['head_dim']
        inter = cfg['intermediate']
        Sq_max = max(1, self.max_prefill_seq)
        qkv_N = n_q * hd + 2 * n_kv * hd
        self._qkv_N = qkv_N
        # Block-128 FP8 GEMM/GEMV scratch shapes (N, K), derived from config
        # so the 8B (qkv 6144) and 2B (qkv 4096) stacks share one code path.
        self._fp8_shapes = (
            (qkv_N, hidden),        # fused q/k/v
            (hidden, hidden),       # o_proj
            (inter, hidden),        # gate / up
            (2 * inter, hidden),    # fused gate/up
            (hidden, inter),        # down
        )

        self._attn = RtxFlashAttnBackendQwen3(
            max_seq=self.max_seq, max_q_seq=Sq_max, dtype=bf16,
            num_layers=cfg['num_hidden_layers'],
            num_q_heads=n_q, num_kv_heads=n_kv, head_dim=hd,
            device=self.device)
        self._h_a = torch.empty(Sq_max, hidden, device=device, dtype=bf16)
        self._h_b = torch.empty(Sq_max, hidden, device=device, dtype=bf16)
        self._res_mid = torch.empty(1, Sq_max, hidden, device=device, dtype=bf16)
        self._layer_out_a = torch.empty(
            1, Sq_max, hidden, device=device, dtype=bf16)
        self._layer_out_b = torch.empty(
            1, Sq_max, hidden, device=device, dtype=bf16)
        self._q_norm_out = torch.empty(
            Sq_max * n_q, hd, device=device, dtype=bf16)
        self._k_norm_out = torch.empty(
            Sq_max * n_kv, hd, device=device, dtype=bf16)
        self._q_pre_flat = torch.empty(
            Sq_max * n_q, hd, device=device, dtype=bf16)
        self._k_pre_flat = torch.empty(
            Sq_max * n_kv, hd, device=device, dtype=bf16)
        self._q_rot = torch.empty(
            1, Sq_max, n_q, hd, device=device, dtype=bf16)
        self._k_rot = torch.empty(
            1, Sq_max, n_kv, hd, device=device, dtype=bf16)
        rope_dim = cfg['rotary_dim']
        half = rope_dim // 2
        idx_lo = torch.arange(half, rope_dim, device=device, dtype=torch.long)
        idx_hi = torch.arange(0, half, device=device, dtype=torch.long)
        self._rope_rotate_idx = torch.cat([idx_lo, idx_hi]).contiguous()
        self._rope_tmp_q = torch.empty(
            1, Sq_max, n_q, rope_dim, device=device, dtype=bf16)
        self._rope_tmp_k = torch.empty(
            1, Sq_max, n_kv, rope_dim, device=device, dtype=bf16)
        self._qkv_out = torch.empty(1, n_q * hd + 2 * n_kv * hd,
                                    device=device, dtype=bf16)
        self._gate_out = torch.empty(1, inter, device=device, dtype=bf16)
        self._up_out = torch.empty(1, inter, device=device, dtype=bf16)
        self._gate_up_out = torch.empty(1, 2 * inter, device=device,
                                        dtype=bf16)
        self._mlp_act = torch.empty(1, inter, device=device, dtype=bf16)
        self._prefill_up_out = torch.empty(
            Sq_max, inter, device=device, dtype=bf16)
        self._prefill_mlp_act = torch.empty(
            Sq_max, inter, device=device, dtype=bf16)
        self._logits_buf = torch.empty(1, vocab, device=device, dtype=bf16)
        self._last_hidden_buf = torch.empty(1, Sq_max, hidden, device=device,
                                            dtype=bf16)
        self._static_token_id = torch.zeros(1, 1, device=device,
                                            dtype=torch.long)
        self._graph_stream = torch.cuda.Stream(device=device)

        self._fp8_scratch: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
        for N, K in self._fp8_shapes + ((vocab, hidden),):
            act = torch.empty(1, K, device=device, dtype=f8)
            scale = torch.empty(1, K // 128, device=device, dtype=fp32)
            out = torch.empty(1, N, device=device, dtype=bf16)
            self._fp8_scratch[(N, K)] = (act, scale, out)
        self._prefill_fp8_scratch: dict[
            tuple[int, int], tuple[torch.Tensor, ...]] = {}
        for N, K in (
            (n_q * hd + 2 * n_kv * hd, hidden),
            (hidden, hidden),
            (inter, hidden),
            (2 * inter, hidden),
            (hidden, inter),
        ):
            act = torch.empty(Sq_max, K, device=device, dtype=f8)
            scale = torch.empty(Sq_max, K // 128, device=device, dtype=fp32)
            out = torch.empty(Sq_max, N, device=device, dtype=bf16)
            self._prefill_fp8_scratch[(N, K)] = (act, scale, out)

    def _build_rope_table(self) -> None:
        import torch

        cfg = self._cfg
        assert cfg is not None
        device = torch.device(self.device)
        rope_dim = cfg['rotary_dim']
        theta = cfg['rope_theta']
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, rope_dim, 2, device=device,
                                    dtype=torch.float32) / rope_dim)
        )
        positions = torch.arange(self.max_seq, device=device,
                                 dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self._rope_cos_table = freqs.cos().to(torch.bfloat16).contiguous()
        self._rope_sin_table = freqs.sin().to(torch.bfloat16).contiguous()

    def _rope_cos_sin(self, cur_pos: int):
        return (self._rope_cos_table[cur_pos:cur_pos + 1],
                self._rope_sin_table[cur_pos:cur_pos + 1])

    def _rope_apply_inline(self, x_in, x_out, tmp, cos4, sin4) -> None:
        import torch

        rope_dim = self._cfg['rotary_dim']
        half = rope_dim // 2
        torch.index_select(x_in, -1, self._rope_rotate_idx, out=tmp)
        tmp[..., :half].neg_()
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

    def reset_state(self) -> None:
        if self._attn is not None:
            self._attn.reset_cache()
        self._cur_pos = 0

    def _quant(self, x, shape: tuple[int, int]):
        import torch
        s = torch.cuda.current_stream().cuda_stream
        act, scale, out = self._fp8_scratch[shape]
        self._fvk.fp8_per_token_block128_quant_bf16(
            x.data_ptr(), act.data_ptr(), scale.data_ptr(), 1, shape[1], s)
        return act, scale, out

    def _gemv(self, act, scale, weight_ptr: int, w_scale_ptr: int,
              out, N: int, K: int) -> None:
        import torch
        s = torch.cuda.current_stream().cuda_stream
        if N >= 4096:
            fn = self._fvk.ht_gemv_fp8_block128_m1_w16
        else:
            fn = self._fvk.ht_gemv_fp8_block128_m1_w8
        fn(act.data_ptr(), weight_ptr, out.data_ptr(), 1, N, K,
           scale.data_ptr(), w_scale_ptr, 1.0, s)

    def _prefill_gemm(self, act, scale, weight_ptr: int,
                      w_scale_ptr: int, out, N: int, K: int,
                      S: int) -> None:
        import torch

        s = torch.cuda.current_stream().cuda_stream
        # Native Ada FP8 block-128 GEMM: reads the FP8 weight directly and
        # applies per-token act scale * block-128 weight scale in the
        # mainloop (no dequant scratch). Same scale semantics as the M=1
        # decode GEMV; ~3-4x faster than the old dequant-scratch prefill path.
        self._fvk.fp8_block128_gemm_blockscaled_sm89_bf16out(
            act.data_ptr(), weight_ptr, out.data_ptr(), S, N, K,
            scale.data_ptr(), w_scale_ptr, s)

    def _write_logits_from_hidden(self, last_row):
        import torch

        cfg = self._cfg
        assert cfg is not None
        hidden = cfg['hidden_size']
        vocab = cfg['vocab_size']
        s = torch.cuda.current_stream().cuda_stream
        last_row = last_row.view(1, hidden).contiguous()
        if self.use_fp8_lm_head:
            ap, sc, _ = self._quant(last_row, (vocab, hidden))
            self._gemv(ap, sc, int(self._weights.ptrs['lm_head_fp8_w']),
                       int(self._weights.ptrs['lm_head_fp8_s']),
                       self._logits_buf, vocab, hidden)
        else:
            self._fvk.bf16_matmul_bf16(
                last_row.data_ptr(), int(self._weights.ptrs['lm_head_w']),
                self._logits_buf.data_ptr(), 1, vocab, hidden, s)
        return self._logits_buf

    def _layer_forward(self, L: int, h_in, cos, sin, cur_pos: int,
                       *,
                       prequant=None,
                       next_input_norm_w: int = 0):
        import torch
        fvk = _import_fvk()

        cfg = self._cfg
        assert cfg is not None
        hidden = cfg['hidden_size']
        n_q = cfg['num_q_heads']
        n_kv = cfg['num_kv_heads']
        hd = cfg['head_dim']
        inter = cfg['intermediate']
        eps = float(cfg['rms_norm_eps'])
        s = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        h2 = h_in.view(1, hidden).contiguous()

        if prequant is None:
            ap, sc, _ = self._fp8_scratch[(self._qkv_N, hidden)]
            fvk.rms_norm_to_fp8_block128_bf16(
                h2.data_ptr(), int(lw['input_norm_w']),
                ap.data_ptr(), sc.data_ptr(), 1, hidden, eps, s)
        else:
            ap, sc = prequant
        self._gemv(ap, sc, int(lw['qkv_proj_w']), int(lw['qkv_proj_s']),
                   self._qkv_out, int(lw['qkv_proj_N']), hidden)
        Nq = n_q * hd
        Nk = n_kv * hd
        qkv_out = self._qkv_out[:1]
        q_pre = qkv_out[:, :Nq]
        k_pre = qkv_out[:, Nq:Nq + Nk]
        v_pre = qkv_out[:, Nq + Nk:]
        kv_layer_stride = self._attn.kv_layer_stride_bytes
        kv_row_stride = self._attn.kv_row_stride_bytes
        kv_slot_off = L * kv_layer_stride + cur_pos * kv_row_stride
        if self.fuse_qk_postproc:
            fvk.qwen3_qk_norm_rope_kvwrite_bf16(
                q_pre.data_ptr(), k_pre.data_ptr(), v_pre.data_ptr(),
                int(lw['q_norm_w']), int(lw['k_norm_w']),
                cos.data_ptr(), sin.data_ptr(),
                self._attn.Q_buf[:, :1].data_ptr(),
                self._attn.K_cache.data_ptr() + kv_slot_off,
                self._attn.V_cache.data_ptr() + kv_slot_off,
                n_q, n_kv, eps, s)
        else:
            fvk.qwen3_q_norm_rope_qstage_bf16(
                q_pre.data_ptr(), int(lw['q_norm_w']), cos.data_ptr(),
                sin.data_ptr(), self._attn.Q_buf[:, :1].data_ptr(),
                n_q, eps, s)
            fvk.qwen3_k_norm_rope_kvwrite_bf16(
                k_pre.data_ptr(), v_pre.data_ptr(), int(lw['k_norm_w']),
                cos.data_ptr(), sin.data_ptr(),
                self._attn.K_cache.data_ptr() + kv_slot_off,
                self._attn.V_cache.data_ptr() + kv_slot_off,
                n_kv, eps, s)

        self._attn.run(
            'full', layer_idx=L, q_seq=1, kv_seq=cur_pos + 1,
            stream=s, causal=True)
        attn_2d = self._attn.O_buf[:, :1].reshape(1, hidden).contiguous()
        ap, sc, o_out = self._quant(attn_2d, (hidden, hidden))
        self._gemv(ap, sc, int(lw['o_proj_w']), int(lw['o_proj_s']),
                   o_out, hidden, hidden)

        attn_proj = o_out.view(1, 1, hidden)
        h_post = self._res_mid[:, :1]
        ap, sc, gate_out = self._fp8_scratch[(inter, hidden)]
        fvk.residual_add_rms_norm_to_fp8_block128_bf16(
            h_in.data_ptr(), attn_proj.data_ptr(), h_post.data_ptr(),
            int(lw['post_attn_norm_w']),
            ap.data_ptr(), sc.data_ptr(), 1, hidden, eps, s)
        if self.fuse_gate_up:
            self._gemv(ap, sc, int(lw['gate_up_w']), int(lw['gate_up_s']),
                       self._gate_up_out, int(lw['gate_up_N']), hidden)
            gate_out = self._gate_up_out[:, :inter]
            up_out = self._gate_up_out[:, inter:]
        else:
            self._gemv(ap, sc, int(lw['mlp_gate_w']), int(lw['mlp_gate_s']),
                       self._gate_out, inter, hidden)
            self._gemv(ap, sc, int(lw['mlp_up_w']), int(lw['mlp_up_s']),
                       self._up_out, inter, hidden)
            gate_out = self._gate_out
            up_out = self._up_out
        ap, sc, down_out = self._fp8_scratch[(hidden, inter)]
        fvk.silu_mul_to_fp8_block128_bf16(
            gate_out.data_ptr(), up_out.data_ptr(),
            ap.data_ptr(), sc.data_ptr(), 1, inter, s)
        self._gemv(ap, sc, int(lw['mlp_down_w']), int(lw['mlp_down_s']),
                   down_out, hidden, inter)

        h_out = (self._layer_out_a if (L % 2 == 0)
                 else self._layer_out_b)[:, :1]
        mlp_out = down_out.view(1, 1, hidden)
        if next_input_norm_w:
            next_ap, next_sc, _ = self._fp8_scratch[(self._qkv_N, hidden)]
            fvk.residual_add_rms_norm_to_fp8_block128_bf16(
                h_post.data_ptr(), mlp_out.data_ptr(), h_out.data_ptr(),
                int(next_input_norm_w),
                next_ap.data_ptr(), next_sc.data_ptr(), 1, hidden, eps, s)
            return h_out, (next_ap, next_sc)
        torch.add(h_post, mlp_out, out=h_out)
        return h_out, None

    def _layer_forward_prefill_fp8_blockscaled(self, L: int, h_in_S, cos_S,
                                           sin_S, start_pos: int, S: int):
        import torch

        fvk = _import_fvk()

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
        h2 = h_in_S.view(S, hidden)

        ap_h, sc_h, qkv_out = self._prefill_fp8_scratch[(n_q * hd + 2 * n_kv * hd, hidden)]
        fvk.rms_norm_to_fp8_block128_bf16(
            h2.data_ptr(), int(lw['input_norm_w']),
            ap_h.data_ptr(), sc_h.data_ptr(), S, hidden, eps, s)
        self._prefill_gemm(
            ap_h, sc_h, int(lw['qkv_proj_w']), int(lw['qkv_proj_s']),
            qkv_out, int(lw['qkv_proj_N']), hidden, S)

        qkv = qkv_out[:S]
        Nq = n_q * hd
        Nk = n_kv * hd
        fvk.qwen3_qk_norm_rope_kvwrite_batched_bf16(
            qkv.data_ptr(),
            qkv[:, Nq:Nq + Nk].data_ptr(),
            qkv[:, Nq + Nk:].data_ptr(),
            int(lw['q_norm_w']),
            int(lw['k_norm_w']),
            cos_S.data_ptr(),
            sin_S.data_ptr(),
            self._attn.Q_buf[:, :S].data_ptr(),
            self._attn.K_cache[L, start_pos:start_pos + S].data_ptr(),
            self._attn.V_cache[L, start_pos:start_pos + S].data_ptr(),
            S,
            int(qkv.stride(0)),
            int(qkv[:, Nq:Nq + Nk].stride(0)),
            int(qkv[:, Nq + Nk:].stride(0)),
            int(self._attn.Q_buf[:, :S].stride(1)),
            int(self._attn.K_cache[L, start_pos:start_pos + S].stride(0)),
            n_q,
            n_kv,
            eps,
            s)

        self._attn.run(
            'full', layer_idx=L, q_seq=S, kv_seq=start_pos + S,
            stream=s, causal=True)
        attn_2d = self._attn.O_buf[:, :S].view(S, hidden)
        ap_o, sc_o, o_out = self._prefill_fp8_scratch[(hidden, hidden)]
        fvk.fp8_per_token_block128_quant_bf16(
            attn_2d.data_ptr(), ap_o.data_ptr(), sc_o.data_ptr(),
            S, hidden, s)
        self._prefill_gemm(
            ap_o, sc_o, int(lw['o_proj_w']), int(lw['o_proj_s']),
            o_out, hidden, hidden, S)

        h_post = self._res_mid[:, :S]
        ap_mlp, sc_mlp, gate_out = self._prefill_fp8_scratch[(inter, hidden)]
        fvk.residual_add_rms_norm_to_fp8_block128_bf16(
            h_in_S.data_ptr(), o_out[:S].view(1, S, hidden).data_ptr(),
            h_post.data_ptr(), int(lw['post_attn_norm_w']),
            ap_mlp.data_ptr(), sc_mlp.data_ptr(), S, hidden, eps, s)
        up_out = self._prefill_up_out[:S]
        ap_dn, sc_dn, down_out = self._prefill_fp8_scratch[(hidden, inter)]
        if self.fuse_gate_up:
            _, _, gate_up_out = self._prefill_fp8_scratch[(2 * inter, hidden)]
            self._prefill_gemm(
                ap_mlp, sc_mlp, int(lw['gate_up_w']), int(lw['gate_up_s']),
                gate_up_out, int(lw['gate_up_N']), hidden, S)
            fvk.silu_mul_merged_to_fp8_block128_bf16(
                gate_up_out[:S].data_ptr(),
                ap_dn.data_ptr(), sc_dn.data_ptr(), S, inter, s)
        else:
            self._prefill_gemm(
                ap_mlp, sc_mlp, int(lw['mlp_gate_w']), int(lw['mlp_gate_s']),
                gate_out, inter, hidden, S)
            self._prefill_gemm(
                ap_mlp, sc_mlp, int(lw['mlp_up_w']), int(lw['mlp_up_s']),
                up_out, inter, hidden, S)
            fvk.silu_mul_to_fp8_block128_bf16(
                gate_out[:S].data_ptr(), up_out.data_ptr(),
                ap_dn.data_ptr(), sc_dn.data_ptr(), S, inter, s)
        self._prefill_gemm(
            ap_dn, sc_dn, int(lw['mlp_down_w']), int(lw['mlp_down_s']),
            down_out, hidden, inter, S)

        h_out = (self._layer_out_a if (L % 2 == 0)
                 else self._layer_out_b)[:, :S]
        torch.add(h_post, down_out[:S].view(1, S, hidden), out=h_out)
        return h_out

    def forward_hidden_prefill_fp8_blockscaled(self, h_S, cos_S, sin_S,
                                           start_pos: int = 0,
                                           deepstack_by_layer=None,
                                           *,
                                           run_lm_head: bool = True):
        import torch

        cfg = self._cfg
        assert cfg is not None
        hidden = cfg['hidden_size']
        eps = float(cfg['rms_norm_eps'])
        s = torch.cuda.current_stream().cuda_stream
        S = int(h_S.shape[-2] if h_S.ndim == 3 else h_S.shape[0])
        if S < 1 or S > self.max_prefill_seq:
            raise ValueError(
                f'prefill S={S} out of [1, {self.max_prefill_seq}]; '
                'construct with a larger max_prefill_seq')
        if start_pos + S > self.max_seq:
            raise ValueError(
                f'prefill end {start_pos + S} > max_seq {self.max_seq}')
        h = h_S.view(1, S, hidden).to(torch.bfloat16).contiguous()
        for L in range(cfg['num_hidden_layers']):
            h = self._layer_forward_prefill_fp8_blockscaled(
                L, h, cos_S, sin_S, start_pos, S)
            if deepstack_by_layer is not None and L in deepstack_by_layer:
                for a, b, ds in deepstack_by_layer[L]:
                    self._fvk.residual_add(
                        h[0, a:b].data_ptr(), ds.data_ptr(),
                        (b - a) * hidden, s)

        h2 = h.view(S, hidden).contiguous()
        x_norm = self._h_b[:S].view(S, hidden)
        self._fvk.rms_norm(
            h2.data_ptr(), int(self._weights.ptrs['final_norm_w']),
            x_norm.data_ptr(), S, hidden, eps, s)
        self._last_hidden_buf[:, :S].copy_(x_norm.view(1, S, hidden))
        self._cur_pos = start_pos + S
        if not run_lm_head:
            return self._last_hidden_buf[:, :S]
        return self._write_logits_from_hidden(
            self._last_hidden_buf[0, S - 1:S])

    def forward_hidden_decode_fp8(self, h, cos_pos, sin_pos, cur_pos: int,
                                  deepstack_by_layer=None):
        import torch

        cfg = self._cfg
        assert cfg is not None
        hidden = cfg['hidden_size']
        vocab = cfg['vocab_size']
        eps = float(cfg['rms_norm_eps'])
        s = torch.cuda.current_stream().cuda_stream
        h = h.view(1, 1, hidden).contiguous()
        prequant = None
        for L in range(cfg['num_hidden_layers']):
            next_norm = 0
            if L + 1 < cfg['num_hidden_layers']:
                next_norm = int(self._weights.ptrs['layers'][L + 1]
                                ['input_norm_w'])
            h, prequant = self._layer_forward(
                L, h, cos_pos, sin_pos, cur_pos,
                prequant=prequant, next_input_norm_w=next_norm)
            if deepstack_by_layer is not None and L in deepstack_by_layer:
                h = h.clone()
                torch.add(h, deepstack_by_layer[L].view(1, 1, hidden), out=h)
                prequant = None

        x_norm = self._h_b[:1].view(1, hidden)
        self._fvk.rms_norm(
            h.view(1, hidden).contiguous().data_ptr(),
            int(self._weights.ptrs['final_norm_w']), x_norm.data_ptr(),
            1, hidden, eps, s)
        self._last_hidden_buf[:, :1].copy_(x_norm.view(1, 1, hidden))
        return self._write_logits_from_hidden(x_norm)

    def forward_own_decode_fp8(self, token_id, cos_pos, sin_pos, cur_pos: int):
        import torch

        cfg = self._cfg
        assert cfg is not None
        hidden = cfg['hidden_size']
        s = torch.cuda.current_stream().cuda_stream
        if not isinstance(token_id, torch.Tensor):
            token_id = torch.tensor([token_id], device=self.device,
                                    dtype=torch.long)
        if token_id.ndim == 1:
            token_id = token_id.view(1, 1)
        h = self._h_a[:1]
        self._fvk.embedding_lookup_bf16(
            token_id.view(-1).data_ptr(),
            int(self._weights.ptrs['embed_w']),
            h.data_ptr(),
            1, hidden, s)
        return self.forward_hidden_decode_fp8(h, cos_pos, sin_pos, cur_pos)

    def decode_step(self, token_id, cur_pos: int):
        cos, sin = self._rope_cos_sin(cur_pos)
        return self.forward_own_decode_fp8(token_id, cos, sin, cur_pos)

    def _ensure_decode_graph(self, cur_pos: int):
        import torch

        if not isinstance(self._decode_graphs, collections.OrderedDict):
            self._decode_graphs = collections.OrderedDict(
                self._decode_graphs)
        graph = self._graph_cache_get(self._decode_graphs, cur_pos)
        if graph is not None:
            return graph
        cos, sin = self._rope_cos_sin(cur_pos)
        gs = self._graph_stream
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.inference_mode():
            for _ in range(2):
                self.forward_own_decode_fp8(
                    self._static_token_id, cos, sin, cur_pos)
        gs.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=gs), torch.inference_mode():
            self.forward_own_decode_fp8(
                self._static_token_id, cos, sin, cur_pos)
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)
        self._decode_graphs[cur_pos] = graph
        if isinstance(self._decode_graphs, collections.OrderedDict):
            self._decode_graphs.move_to_end(cur_pos)
        self._trim_lru_graph_cache(
            self._decode_graphs, self.max_decode_graphs)
        return graph

    def decode_step_with_graph(self, token_id, cur_pos: int):
        import torch

        if isinstance(token_id, torch.Tensor):
            if token_id.ndim == 1:
                token_id = token_id.view(1, 1)
            self._static_token_id.copy_(token_id)
        else:
            self._static_token_id.fill_(int(token_id))
        self._ensure_decode_graph(cur_pos).replay()
        return self._logits_buf
