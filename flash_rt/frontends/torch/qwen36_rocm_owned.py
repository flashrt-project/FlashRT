"""FlashRT -- owned FP8 Qwen3.6 ROCm graph frontend.

This module packages the verified Qwen3.6 ROCm single-token FP8 pipeline into
a reusable owned-buffer frontend. The current path provides graph-correct
single-token logits with owned buffers.
Production decode still needs persistent Gated-DeltaNet state and full-attn KV
cache lifecycle.
"""

from __future__ import annotations

import os
import time
from typing import Any

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


class Qwen36OwnedDecodeRunner:
    """Static-buffer Qwen3.6 single-token FP8 ROCm runner."""

    def __init__(
        self,
        handles,
        *,
        persistent_linear_state: bool = False,
        use_full_attn_kv: bool = False,
        max_seq: int = 2048,
    ) -> None:
        import torch
        import aiter
        import flash_attn
        import flash_rt.flash_rt_rocm_kernels as kernels

        self.aiter = aiter
        self.flash_attn = flash_attn
        self.kernels = kernels
        self.handles = handles
        self.persistent_linear_state = bool(persistent_linear_state)
        self.use_full_attn_kv = bool(use_full_attn_kv)
        self.max_seq = int(max_seq)
        self.enable_gdn_norm_quant_fusion = True
        self.enable_gdn_broadcast3_recurrent = True
        self.enable_gdn_broadcast3_norm_quant_fusion = True
        self.enable_gdn_broadcast3_fastout = True
        self.enable_full_attn_gate_quant_fusion = True
        self.enable_gdn_seq_norm_quant_fusion = False
        self.full_q_heads = 24
        self.full_kv_heads = 4
        self.full_head_dim = 256
        self.full_rotary_dim = 64
        self.by_ptr = {int(t.data_ptr()): t for t in handles.anchors}
        self.eps = float(handles.ptrs["rms_norm_eps"])
        self.device = torch.device("cuda")
        self.bf16: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
        self.fp8: dict[tuple[str, int, int], torch.Tensor] = {}
        self.scale: dict[tuple[str, int, int], torch.Tensor] = {}
        self.f32: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
        self.combined_weights: dict[tuple[int, tuple[str, ...]], tuple[Any, Any, list[int]]] = {}
        self.linear_states: dict[int, torch.Tensor] = {}
        self.conv_states: dict[int, torch.Tensor] = {}
        if self.use_full_attn_kv:
            self.full_k_cache = torch.empty(
                64,
                self.max_seq,
                self.full_kv_heads,
                self.full_head_dim,
                device=self.device,
                dtype=torch.bfloat16,
            )
            self.full_v_cache = torch.empty_like(self.full_k_cache)
            self.cos, self.sin = self._build_rope_cache(
                self.max_seq, self.full_rotary_dim, self.device
            )
        else:
            self.full_k_cache = None
            self.full_v_cache = None
            self.cos = None
            self.sin = None

    @staticmethod
    def _build_rope_cache(max_seq: int, rotary_dim: int, device):
        import torch

        pos = torch.arange(max_seq, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            10_000_000.0
            ** (
                torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
                / rotary_dim
            )
        )
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().contiguous(), emb.sin().contiguous()

    def top(self, name: str):
        return self.by_ptr[int(self.handles.ptrs[name])]

    def get(self, layer: dict, name: str):
        return self.by_ptr[int(layer[name])]

    def b(self, tag: str, shape: tuple[int, ...]):
        import torch

        shape = tuple(int(x) for x in shape)
        key = (tag, shape)
        if key not in self.bf16:
            self.bf16[key] = torch.empty(*shape, device=self.device, dtype=torch.bfloat16)
        return self.bf16[key]

    def q(self, tag: str, rows: int, hidden: int):
        import torch

        rows = int(rows)
        hidden = int(hidden)
        key = (tag, rows, hidden)
        if key not in self.fp8:
            self.fp8[key] = torch.empty(rows, hidden, device=self.device, dtype=torch.float8_e4m3fnuz)
            self.scale[key] = torch.empty(rows, hidden // 128, device=self.device, dtype=torch.float32)
        return self.fp8[key], self.scale[key]

    def f(self, tag: str, shape: tuple[int, ...]):
        import torch

        shape = tuple(int(x) for x in shape)
        key = (tag, shape)
        if key not in self.f32:
            self.f32[key] = torch.empty(*shape, device=self.device, dtype=torch.float32)
        return self.f32[key]

    def linear_state(self, layer_idx: int):
        if layer_idx not in self.linear_states:
            self.linear_states[layer_idx] = self.f(
                f"l{layer_idx}_state", (48, 128, 128)
            )
        return self.linear_states[layer_idx]

    def conv_state(self, layer_idx: int):
        if layer_idx not in self.conv_states:
            self.conv_states[layer_idx] = self.b(
                f"l{layer_idx}_conv_state", (3, 10240)
            )
            self.conv_states[layer_idx].zero_()
        return self.conv_states[layer_idx]

    def reset_linear_states(self) -> None:
        for state in self.linear_states.values():
            state.zero_()
        for state in self.conv_states.values():
            state.zero_()

    def norm_quant(self, h, weight, tag: str):
        rows, hidden = h.shape
        out = self.b(tag + "_bf16", (rows, hidden))
        out_q, out_s = self.q(tag + "_fp8", rows, hidden)
        self.kernels.qwen36_rms_norm_bf16_quant_fp8_fnuz_out(
            h, weight.contiguous(), out, out_q, out_s, rows, hidden, self.eps
        )
        return out, out_q, out_s

    def quant(self, x, tag: str):
        rows, hidden = x.shape
        out_q, out_s = self.q(tag, rows, hidden)
        self.kernels.qwen36_quant_fp8_fnuz_1x128_out(x.contiguous(), out_q, out_s, rows, hidden)
        return out_q, out_s

    def silu_mul_quant(self, gate, up, tag: str):
        rows, hidden = gate.shape
        out_q, out_s = self.q(tag, rows, hidden)
        self.kernels.qwen36_silu_mul_quant_fp8_fnuz_1x128_out(
            gate.contiguous(), up.contiguous(), out_q, out_s, rows, hidden
        )
        return out_q, out_s

    def sigmoid_mul_quant(self, x, gate, tag: str):
        rows, hidden = x.shape
        out_q, out_s = self.q(tag, rows, hidden)
        self.kernels.qwen36_sigmoid_mul_quant_fp8_fnuz_1x128_out(
            x.contiguous(), gate.contiguous(), out_q, out_s, rows, hidden
        )
        return out_q, out_s

    def head_norm_gated_silu_quant(self, attn, gate, weight, seq: int, tag: str):
        out_q, out_s = self.q(tag, seq, 6144)
        self.kernels.qwen36_rms_norm_gated_silu_quant_fp8_fnuz_1x128_out(
            attn.contiguous(),
            gate.contiguous(),
            weight.contiguous(),
            out_q,
            out_s,
            seq,
            48,
            128,
            self.eps,
        )
        return out_q, out_s

    def linear_q(self, layer: dict, xq, xs, name: str, tag: str):
        w = self.get(layer, name + "_w")
        out = self.b(tag, (int(xq.shape[0]), int(w.shape[0])))
        self.aiter.gemm_a8w8_blockscale_ck(xq, w, xs, self.get(layer, name + "_s"), out)
        return out

    def _combined_weight(self, layer: dict, names: tuple[str, ...]):
        import torch

        key = (id(layer), tuple(names))
        cached = self.combined_weights.get(key)
        if cached is not None:
            return cached
        weights = [self.get(layer, name + "_w") for name in names]
        scales = [self.get(layer, name + "_s") for name in names]
        if any(w.dtype != weights[0].dtype for w in weights):
            raise RuntimeError("combined FP8 weights must have the same dtype")
        if weights[0].dtype in {torch.float8_e4m3fn, torch.float8_e4m3fnuz}:
            w = torch.cat([t.contiguous().view(torch.uint8) for t in weights], dim=0)
            w = w.contiguous().view(weights[0].dtype)
        else:
            w = torch.cat(weights, dim=0).contiguous()
        s = torch.cat(scales, dim=0).contiguous()
        sizes = [int(t.shape[0]) for t in weights]
        cached = (w, s, sizes)
        self.combined_weights[key] = cached
        return cached

    def linear_q_multi(self, layer: dict, xq, xs, names: tuple[str, ...], tag: str):
        w, s, sizes = self._combined_weight(layer, names)
        rows = int(xq.shape[0])
        out = self.b(tag, (rows, int(w.shape[0])))
        self.aiter.gemm_a8w8_blockscale_ck(xq, w, xs, s, out)
        parts = []
        start = 0
        for size in sizes:
            parts.append(out[:, start : start + size])
            start += size
        return out, parts

    def small_linear(self, x, w, tag: str):
        rows = int(x.shape[0])
        out_features = int(w.shape[0])
        hidden = int(w.shape[1])
        out = self.b(tag, (rows, out_features))
        self.kernels.qwen36_small_linear_bf16_out(x.contiguous(), w, out, rows, out_features, hidden)
        return out

    def in_proj_ab_gating(self, x, layer: dict, seq: int, tag: str):
        g = self.b(tag + "_g", (seq, 48))
        beta = self.b(tag + "_beta", (seq, 48))
        self.kernels.qwen36_in_proj_ab_gating_bf16_out(
            x.contiguous(),
            self.get(layer, "in_proj_a_w"),
            self.get(layer, "in_proj_b_w"),
            self.get(layer, "A_log"),
            self.get(layer, "dt_bias"),
            g,
            beta,
            seq,
            48,
            int(x.shape[1]),
        )
        return g, beta

    def add(self, a, b, tag: str):
        out = self.b(tag, tuple(a.shape))
        self.kernels.qwen36_add_bf16_out(a.contiguous(), b.contiguous(), out, int(a.numel()))
        return out

    def add_norm_quant(self, a, b, weight, tag: str):
        rows, hidden = a.shape
        residual = self.b(tag + "_residual", (rows, hidden))
        out = self.b(tag + "_bf16", (rows, hidden))
        out_q, out_s = self.q(tag + "_fp8", rows, hidden)
        self.kernels.qwen36_add_rms_norm_bf16_quant_fp8_fnuz_out(
            a.contiguous(),
            b.contiguous(),
            weight.contiguous(),
            residual,
            out,
            out_q,
            out_s,
            rows,
            hidden,
            self.eps,
        )
        return residual, out, out_q, out_s

    def finish_layer(self, h_post, mlp_out, layer_idx: int, next_layer: dict | None, next_idx: int | None):
        if next_layer is None:
            return self.add(h_post, mlp_out, f"l{layer_idx}_out"), None
        h, x, xq, xs = self.add_norm_quant(
            h_post,
            mlp_out,
            self.get(next_layer, "input_norm_eff_w"),
            f"l{next_idx}_in_norm",
        )
        return h, (x, xq, xs)

    def linear_layer(
        self,
        h,
        layer: dict,
        layer_idx: int,
        *,
        input_norm=None,
        next_layer: dict | None = None,
        next_idx: int | None = None,
    ):
        seq = int(h.shape[0])
        if input_norm is None:
            x, xq, xs = self.norm_quant(h, self.get(layer, "input_norm_eff_w"), f"l{layer_idx}_in_norm")
        else:
            x, xq, xs = input_norm
        if seq == 1:
            _, (qkv, z_raw) = self.linear_q_multi(
                layer,
                xq,
                xs,
                ("in_proj_qkv", "in_proj_z"),
                f"l{layer_idx}_qkv_z",
            )
            z = z_raw.view(seq, 48, 128)
        else:
            qkv = self.linear_q(layer, xq, xs, "in_proj_qkv", f"l{layer_idx}_qkv")
            z = self.linear_q(layer, xq, xs, "in_proj_z", f"l{layer_idx}_z").view(seq, 48, 128)
        if seq == 1:
            g, beta = self.in_proj_ab_gating(x, layer, seq, f"l{layer_idx}_ab_gate")
        else:
            a = self.small_linear(x, self.get(layer, "in_proj_a_w"), f"l{layer_idx}_a")
            b = self.small_linear(x, self.get(layer, "in_proj_b_w"), f"l{layer_idx}_b")

        q = self.b(f"l{layer_idx}_lin_q", (seq, 48, 128))
        k = self.b(f"l{layer_idx}_lin_k", (seq, 48, 128))
        v = self.b(f"l{layer_idx}_lin_v", (seq, 48, 128))
        if self.persistent_linear_state:
            conv_state = self.conv_state(layer_idx)
            if seq == 1:
                self.kernels.qwen36_causal_conv1d_state_split_qkv_bf16_inplace_out(
                    qkv,
                    self.get(layer, "conv1d_w"),
                    conv_state,
                    q,
                    k,
                    v,
                    10240,
                    4,
                )
            else:
                next_conv_state = self.b(f"l{layer_idx}_conv_state_next", (3, 10240))
                self.kernels.qwen36_causal_conv1d_state_split_qkv_bf16_out(
                    qkv,
                    self.get(layer, "conv1d_w"),
                    conv_state,
                    next_conv_state,
                    q,
                    k,
                    v,
                    seq,
                    10240,
                    4,
                )
                self.kernels.qwen36_copy_bf16_out(
                    next_conv_state, conv_state, int(next_conv_state.numel())
                )
        else:
            qkv_conv = self.b(f"l{layer_idx}_qkv_conv", (seq, 10240))
            self.kernels.qwen36_causal_conv1d_bf16_out(qkv, self.get(layer, "conv1d_w"), qkv_conv, seq, 10240, 4, True)
            self.kernels.qwen36_lin_split_qkv_broadcast_bf16_out(qkv_conv, q, k, v, seq)
        state = self.linear_state(layer_idx)
        if not self.persistent_linear_state:
            state.zero_()
        if seq != 1:
            g = self.b(f"l{layer_idx}_g", (seq, 48))
            beta = self.b(f"l{layer_idx}_beta", (seq, 48))
            self.kernels.qwen36_gdn_gating_bf16_out(
                a,
                b,
                self.get(layer, "A_log"),
                self.get(layer, "dt_bias"),
                g,
                beta,
                seq,
                48,
            )
        if self.enable_gdn_norm_quant_fusion and seq == 1:
            normed_q, normed_s = self.q(f"l{layer_idx}_out_proj_q", seq, 6144)
            if self.enable_gdn_broadcast3_fastout:
                self.kernels.qwen36_gated_deltanet_recurrent_broadcast3_fastout_norm_quant_fp8_fnuz_out(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    z,
                    self.get(layer, "head_norm_w"),
                    state,
                    normed_q,
                    normed_s,
                    16,
                    128,
                    self.eps,
                )
            elif self.enable_gdn_broadcast3_norm_quant_fusion:
                self.kernels.qwen36_gated_deltanet_recurrent_broadcast3_norm_quant_fp8_fnuz_out(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    z,
                    self.get(layer, "head_norm_w"),
                    state,
                    normed_q,
                    normed_s,
                    16,
                    128,
                    self.eps,
                )
            else:
                self.kernels.qwen36_gated_deltanet_recurrent_norm_quant_fp8_fnuz_out(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    z,
                    self.get(layer, "head_norm_w"),
                    state,
                    normed_q,
                    normed_s,
                    48,
                    128,
                    self.eps,
                )
        elif self.enable_gdn_seq_norm_quant_fusion and seq != 1:
            normed_q, normed_s = self.q(f"l{layer_idx}_out_proj_q", seq, 6144)
            self.kernels.qwen36_gated_deltanet_recurrent_norm_quant_seq_fp8_fnuz_out(
                q,
                k,
                v,
                g,
                beta,
                z,
                self.get(layer, "head_norm_w"),
                state,
                normed_q,
                normed_s,
                seq,
                48,
                128,
                self.eps,
            )
        else:
            attn = self.b(f"l{layer_idx}_attn", (seq, 48, 128))
            if self.enable_gdn_broadcast3_recurrent and seq == 1:
                self.kernels.qwen36_gated_deltanet_recurrent_broadcast3_bf16_out(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    state,
                    attn,
                    16,
                    128,
                )
            else:
                self.kernels.qwen36_gated_deltanet_recurrent_bf16_out(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    state,
                    attn,
                    seq,
                    48,
                    128,
                )
            normed_q, normed_s = self.head_norm_gated_silu_quant(
                attn,
                z,
                self.get(layer, "head_norm_w"),
                seq,
                f"l{layer_idx}_out_proj_q",
            )
        attn_proj = self.linear_q(layer, normed_q, normed_s, "out_proj", f"l{layer_idx}_attn_proj")
        h_post, x_mlp, x_mlp_q, x_mlp_s = self.add_norm_quant(
            h,
            attn_proj,
            self.get(layer, "post_attn_norm_eff_w"),
            f"l{layer_idx}_h_post_mlp_norm",
        )
        if seq == 1:
            _, (gate, up) = self.linear_q_multi(
                layer,
                x_mlp_q,
                x_mlp_s,
                ("mlp_gate", "mlp_up"),
                f"l{layer_idx}_mlp_gate_up",
            )
        else:
            gate = self.linear_q(layer, x_mlp_q, x_mlp_s, "mlp_gate", f"l{layer_idx}_mlp_gate")
            up = self.linear_q(layer, x_mlp_q, x_mlp_s, "mlp_up", f"l{layer_idx}_mlp_up")
        act_q, act_s = self.silu_mul_quant(gate, up, f"l{layer_idx}_mlp_act_q")
        mlp_out = self.linear_q(layer, act_q, act_s, "mlp_down", f"l{layer_idx}_mlp_down")
        return self.finish_layer(h_post, mlp_out, layer_idx, next_layer, next_idx)

    def full_layer_decode(
        self,
        h,
        layer: dict,
        layer_idx: int,
        *,
        pos_start: int = 0,
        kv_seq: int = 1,
        input_norm=None,
        next_layer: dict | None = None,
        next_idx: int | None = None,
    ):
        seq = 1
        if input_norm is None:
            x, xq, xs = self.norm_quant(h, self.get(layer, "input_norm_eff_w"), f"l{layer_idx}_in_norm")
        else:
            x, xq, xs = input_norm
        if self.use_full_attn_kv:
            _, (q_gate, k_proj, v_proj) = self.linear_q_multi(
                layer,
                xq,
                xs,
                ("q_proj", "k_proj", "v_proj"),
                f"l{layer_idx}_qkv_gate",
            )
            gate = q_gate[:, 6144:].contiguous()
            if self.full_k_cache is None or self.full_v_cache is None:
                raise RuntimeError("full-attn KV buffers are not initialized")
            if self.cos is None or self.sin is None:
                raise RuntimeError("full-attn RoPE cache is not initialized")
            q = self.b(f"l{layer_idx}_full_q", (seq, 6144))
            self.kernels.qwen36_full_qk_norm_partial_rope_cache_bf16_out(
                q_gate,
                k_proj.view(seq, 4, 256),
                v_proj.view(seq, 4, 256),
                self.cos[pos_start : pos_start + seq],
                self.sin[pos_start : pos_start + seq],
                self.get(layer, "q_norm_eff_w"),
                self.get(layer, "k_norm_eff_w"),
                q.view(seq, 24, 256),
                self.full_k_cache,
                self.full_v_cache,
                layer_idx,
                self.max_seq,
                pos_start,
                seq,
                24,
                4,
                256,
                64,
                self.eps,
            )
            if self.enable_full_attn_gate_quant_fusion:
                gated_q, gated_s = self.q(f"l{layer_idx}_o_in_q", seq, 6144)
                self.kernels.qwen3_decode_attention_gate_quant_fp8_fnuz_out(
                    q,
                    self.full_k_cache,
                    self.full_v_cache,
                    gate,
                    gated_q,
                    gated_s,
                    layer_idx,
                    self.max_seq,
                    kv_seq,
                    24,
                    4,
                    256,
                )
            else:
                attn = self.b(f"l{layer_idx}_full_attn", (seq, 6144))
                self.kernels.qwen3_decode_attention_bf16_ptr(
                    q.data_ptr(),
                    self.full_k_cache.data_ptr(),
                    self.full_v_cache.data_ptr(),
                    attn.data_ptr(),
                    layer_idx,
                    self.max_seq,
                    kv_seq,
                    24,
                    4,
                    256,
                )
                gated_q, gated_s = self.sigmoid_mul_quant(attn, gate, f"l{layer_idx}_o_in_q")
        else:
            _, (q_gate, v_proj) = self.linear_q_multi(
                layer,
                xq,
                xs,
                ("q_proj", "v_proj"),
                f"l{layer_idx}_qv_gate",
            )
            gate = q_gate[:, 6144:].contiguous()
            v_bcast = self.b(f"l{layer_idx}_v_bcast", (seq, 6144))
            self.kernels.qwen36_full_v_broadcast_bf16_out(v_proj.view(seq, 4, 256), v_bcast, seq, 4, 24, 256)
            gated_q, gated_s = self.sigmoid_mul_quant(v_bcast, gate, f"l{layer_idx}_o_in_q")
        attn_proj = self.linear_q(layer, gated_q, gated_s, "o_proj", f"l{layer_idx}_attn_proj")
        h_post, x_mlp, x_mlp_q, x_mlp_s = self.add_norm_quant(
            h,
            attn_proj,
            self.get(layer, "post_attn_norm_eff_w"),
            f"l{layer_idx}_h_post_mlp_norm",
        )
        _, (gate_mlp, up) = self.linear_q_multi(
            layer,
            x_mlp_q,
            x_mlp_s,
            ("mlp_gate", "mlp_up"),
            f"l{layer_idx}_mlp_gate_up",
        )
        act_q, act_s = self.silu_mul_quant(gate_mlp, up, f"l{layer_idx}_mlp_act_q")
        mlp_out = self.linear_q(layer, act_q, act_s, "mlp_down", f"l{layer_idx}_mlp_down")
        return self.finish_layer(h_post, mlp_out, layer_idx, next_layer, next_idx)

    def full_layer_prefill(
        self,
        h,
        layer: dict,
        layer_idx: int,
        *,
        pos_start: int = 0,
        kv_seq: int | None = None,
        input_norm=None,
        next_layer: dict | None = None,
        next_idx: int | None = None,
    ):
        seq = int(h.shape[0])
        pos_start = int(pos_start)
        if kv_seq is None:
            kv_seq = pos_start + seq
        kv_seq = int(kv_seq)
        if seq <= 0:
            raise ValueError("prefill sequence must be positive")
        if kv_seq < pos_start + seq:
            raise ValueError("kv_seq must include the prefill positions")
        if pos_start + seq > self.max_seq or kv_seq > self.max_seq:
            raise ValueError("prefill sequence exceeds max_seq")
        if not self.use_full_attn_kv:
            raise RuntimeError("full-attention prefill requires use_full_attn_kv=True")
        if self.full_k_cache is None or self.full_v_cache is None:
            raise RuntimeError("full-attn KV buffers are not initialized")
        if self.cos is None or self.sin is None:
            raise RuntimeError("full-attn RoPE cache is not initialized")

        if input_norm is None:
            x, xq, xs = self.norm_quant(h, self.get(layer, "input_norm_eff_w"), f"l{layer_idx}_in_norm")
        else:
            x, xq, xs = input_norm
        _, (q_gate, k_proj, v_proj) = self.linear_q_multi(
            layer,
            xq,
            xs,
            ("q_proj", "k_proj", "v_proj"),
            f"l{layer_idx}_qkv_prefill",
        )
        gate = q_gate[:, 6144:].contiguous()
        q = self.b(f"l{layer_idx}_full_q", (seq, 6144))
        self.kernels.qwen36_full_qk_norm_partial_rope_cache_bf16_out(
            q_gate[:, :6144].contiguous(),
            k_proj.contiguous().view(seq, 4, 256),
            v_proj.contiguous().view(seq, 4, 256),
            self.cos[pos_start : pos_start + seq],
            self.sin[pos_start : pos_start + seq],
            self.get(layer, "q_norm_eff_w"),
            self.get(layer, "k_norm_eff_w"),
            q.view(seq, 24, 256),
            self.full_k_cache,
            self.full_v_cache,
            layer_idx,
            self.max_seq,
            pos_start,
            seq,
            24,
            4,
            256,
            64,
            self.eps,
        )
        attn = self.flash_attn.flash_attn_func(
            q.view(1, seq, 24, 256),
            self.full_k_cache[layer_idx, :kv_seq].unsqueeze(0),
            self.full_v_cache[layer_idx, :kv_seq].unsqueeze(0),
            dropout_p=0.0,
            softmax_scale=1.0 / (256 ** 0.5),
            causal=True,
        ).view(seq, 6144)
        gated_q, gated_s = self.sigmoid_mul_quant(attn, gate, f"l{layer_idx}_o_in_q")
        attn_proj = self.linear_q(layer, gated_q, gated_s, "o_proj", f"l{layer_idx}_attn_proj")
        h_post, x_mlp, x_mlp_q, x_mlp_s = self.add_norm_quant(
            h,
            attn_proj,
            self.get(layer, "post_attn_norm_eff_w"),
            f"l{layer_idx}_h_post_mlp_norm",
        )
        gate_mlp = self.linear_q(layer, x_mlp_q, x_mlp_s, "mlp_gate", f"l{layer_idx}_mlp_gate")
        up = self.linear_q(layer, x_mlp_q, x_mlp_s, "mlp_up", f"l{layer_idx}_mlp_up")
        act_q, act_s = self.silu_mul_quant(gate_mlp, up, f"l{layer_idx}_mlp_act_q")
        mlp_out = self.linear_q(layer, act_q, act_s, "mlp_down", f"l{layer_idx}_mlp_down")
        return self.finish_layer(h_post, mlp_out, layer_idx, next_layer, next_idx)

    def forward_hidden(self, h, *, pos_start: int = 0, kv_seq: int = 1):
        input_norm = None
        layers = self.handles.ptrs["layers"]
        for i, layer in enumerate(layers):
            next_layer = layers[i + 1] if i + 1 < len(layers) else None
            next_idx = i + 1 if next_layer is not None else None
            if layer["type"] == "linear_attention":
                h, input_norm = self.linear_layer(
                    h,
                    layer,
                    i,
                    input_norm=input_norm,
                    next_layer=next_layer,
                    next_idx=next_idx,
                )
            elif int(h.shape[0]) == 1:
                h, input_norm = self.full_layer_decode(
                    h,
                    layer,
                    i,
                    pos_start=pos_start,
                    kv_seq=kv_seq,
                    input_norm=input_norm,
                    next_layer=next_layer,
                    next_idx=next_idx,
                )
            else:
                h, input_norm = self.full_layer_prefill(
                    h,
                    layer,
                    i,
                    pos_start=pos_start,
                    kv_seq=kv_seq,
                    input_norm=input_norm,
                    next_layer=next_layer,
                    next_idx=next_idx,
                )
        return h

    def full_logits_fp8(self, h0, *, pos_start: int = 0, kv_seq: int = 1):
        h = self.forward_hidden(h0, pos_start=pos_start, kv_seq=kv_seq)
        _, final_q, final_s = self.norm_quant(h, self.top("final_norm_eff_w"), "final_norm")
        out = self.b("lm_head_fp8_logits", (int(final_q.shape[0]), int(self.top("lm_head_fp8_w").shape[0])))
        self.aiter.gemm_a8w8_blockscale_ck(
            final_q, self.top("lm_head_fp8_w"), final_s, self.top("lm_head_fp8_s"), out
        )
        return out


class Qwen36RocmOwnedFP8Frontend:
    """Qwen3.6-27B ROCm owned FP8 graph frontend."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cuda",
        weight_mode: str = "fp8_fnuz_cached",
        trust_remote_code: bool = True,
        warmup_graph: bool = True,
        persistent_linear_state: bool = False,
        use_full_attn_kv: bool = False,
        max_seq: int = 2048,
        **_: Any,
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        from flash_rt.frontends.torch.qwen36_rocm_weights import (
            extract_weights_qwen36_fp8_rocm,
            summarize_qwen36_rocm_weights,
        )

        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            raise RuntimeError("Qwen36RocmOwnedFP8Frontend requires ROCm PyTorch")

        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.weight_mode = str(weight_mode)
        self.warmup_graph = bool(warmup_graph)
        self.persistent_linear_state = bool(persistent_linear_state)
        self.use_full_attn_kv = bool(use_full_attn_kv)
        self.max_seq = int(max_seq)
        self.latency_records: list[float] = []
        self._decode_graph = None
        self._decode_graphs: dict[tuple[int, int], Any] = {}
        self._prefill_graphs: dict[int, Any] = {}
        self._prefill_ids_bufs: dict[int, Any] = {}
        self._prefill_logits: dict[int, Any] = {}
        self.prefill_graph_max_decode_steps = 12
        self._last_logits = None

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            trust_remote_code=trust_remote_code,
        )
        self.weights = extract_weights_qwen36_fp8_rocm(
            self.checkpoint_path,
            device=self.device,
            weight_mode=self.weight_mode,
        )
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0
        if "lm_head_fp8_w" not in self.weights.ptrs or "lm_head_fp8_s" not in self.weights.ptrs:
            raise RuntimeError(
                "Qwen36RocmOwnedFP8Frontend requires a baked cache with "
                "lm_head.weight_fp8_fnuz and lm_head.weight_fp8_fnuz_scale_inv"
            )

        self.runner = Qwen36OwnedDecodeRunner(
            self.weights,
            persistent_linear_state=self.persistent_linear_state,
            use_full_attn_kv=self.use_full_attn_kv,
            max_seq=self.max_seq,
        )
        self.input_ids_buf = torch.empty(1, device=self.device, dtype=torch.long)
        self.logits = self.runner.b(
            "lm_head_fp8_logits", (1, int(self.runner.top("lm_head_fp8_w").shape[0]))
        )
        self.config_summary = summarize_qwen36_rocm_weights(self.weights)
        if self.persistent_linear_state and self.use_full_attn_kv:
            graph_scope = "single_token_logits_with_full_attn_kv_and_persistent_linear_state"
        elif self.use_full_attn_kv:
            graph_scope = "single_token_logits_with_full_attn_kv_no_persistent_linear_state"
        elif self.persistent_linear_state:
            graph_scope = "single_token_logits_with_persistent_linear_state_no_full_attn_kv"
        else:
            graph_scope = "single_token_logits_no_persistent_state"
        self.config_summary.update(
            {
                "backend": "rocm_owned_fp8_graph",
                "load_s": self.load_s,
                "graph_scope": graph_scope,
                "warmup_graph": self.warmup_graph,
                "persistent_linear_state": self.persistent_linear_state,
                "use_full_attn_kv": self.use_full_attn_kv,
                "max_seq": self.max_seq,
            }
        )
        if self.warmup_graph:
            self.capture_decode_graph(0)

    def _embed_input(self):
        return self.runner.top("embed_w").index_select(0, self.input_ids_buf).contiguous()

    def _prefill_ids_buf(self, prompt_len: int):
        import torch

        prompt_len = int(prompt_len)
        if prompt_len not in self._prefill_ids_bufs:
            self._prefill_ids_bufs[prompt_len] = torch.empty(
                prompt_len, device=self.device, dtype=torch.long
            )
        return self._prefill_ids_bufs[prompt_len]

    def _embed_prefill(self, prompt_len: int):
        ids = self._prefill_ids_buf(prompt_len)
        return self.runner.top("embed_w").index_select(0, ids).contiguous()

    def reset_state(self) -> None:
        self.runner.reset_linear_states()

    def forward_token(
        self,
        token_id: int,
        *,
        pos_start: int = 0,
        kv_seq: int | None = None,
        sync: bool = True,
    ):
        import torch

        if kv_seq is None:
            kv_seq = int(pos_start) + 1
        self.input_ids_buf.fill_(int(token_id))
        out = self.runner.full_logits_fp8(
            self._embed_input(),
            pos_start=int(pos_start),
            kv_seq=int(kv_seq),
        )
        if sync:
            torch.cuda.synchronize()
        self._last_logits = out
        return out

    def capture_decode_graph(
        self,
        token_id: int = 0,
        *,
        pos_start: int = 0,
        kv_seq: int | None = None,
    ):
        import torch

        pos_start = int(pos_start)
        if kv_seq is None:
            kv_seq = pos_start + 1
        kv_seq = int(kv_seq)
        key = (pos_start, kv_seq)
        cached = self._decode_graphs.get(key)
        if cached is not None:
            self._decode_graph = cached
            return cached

        self.input_ids_buf.fill_(int(token_id))
        for _ in range(2):
            self.runner.full_logits_fp8(
                self._embed_input(),
                pos_start=pos_start,
                kv_seq=kv_seq,
            )
        if self.persistent_linear_state:
            self.reset_state()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = self.runner.full_logits_fp8(
                self._embed_input(),
                pos_start=pos_start,
                kv_seq=kv_seq,
            )
        torch.cuda.synchronize()
        self._decode_graph = graph
        self._decode_graphs[key] = graph
        self._last_logits = captured
        if self.persistent_linear_state:
            self.reset_state()
        return graph

    def capture_decode_graph_table(
        self,
        start_pos: int,
        count: int,
        *,
        token_id: int = 0,
    ) -> dict[tuple[int, int], Any]:
        if count <= 0:
            raise ValueError("count must be positive")
        graphs = {}
        for pos in range(int(start_pos), int(start_pos) + int(count)):
            kv_seq = pos + 1
            graphs[(pos, kv_seq)] = self.capture_decode_graph(
                token_id,
                pos_start=pos,
                kv_seq=kv_seq,
            )
        return graphs

    def replay_decode_graph(
        self,
        token_id: int,
        *,
        pos_start: int = 0,
        kv_seq: int | None = None,
        sync: bool = True,
    ):
        import torch

        pos_start = int(pos_start)
        if kv_seq is None:
            kv_seq = pos_start + 1
        kv_seq = int(kv_seq)
        graph = self._decode_graphs.get((pos_start, kv_seq))
        if graph is None:
            graph = self.capture_decode_graph(token_id, pos_start=pos_start, kv_seq=kv_seq)
        self.input_ids_buf.fill_(int(token_id))
        if sync:
            torch.cuda.synchronize()
        graph.replay()
        if sync:
            torch.cuda.synchronize()
        self._last_logits = self.logits
        return self.logits

    def capture_prefill_graph(self, prompt_len: int):
        import torch

        prompt_len = int(prompt_len)
        if prompt_len <= 1:
            raise ValueError("prefill graph requires prompt_len > 1")
        cached = self._prefill_graphs.get(prompt_len)
        if cached is not None:
            return cached
        if not (self.persistent_linear_state and self.use_full_attn_kv):
            raise RuntimeError(
                "prefill graph requires persistent_linear_state=True and use_full_attn_kv=True"
            )
        ids = self._prefill_ids_buf(prompt_len)
        ids.zero_()
        for _ in range(2):
            self.reset_state()
            self.runner.full_logits_fp8(
                self._embed_prefill(prompt_len), pos_start=0, kv_seq=prompt_len
            )
        self.reset_state()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = self.runner.full_logits_fp8(
                self._embed_prefill(prompt_len), pos_start=0, kv_seq=prompt_len
            )
        torch.cuda.synchronize()
        self._prefill_graphs[prompt_len] = graph
        self._prefill_logits[prompt_len] = captured
        self.reset_state()
        return graph

    def replay_prefill_graph(self, input_ids, *, sync: bool = True):
        import torch

        if input_ids.dim() == 2:
            if input_ids.shape[0] != 1:
                raise ValueError("only batch size 1 is supported")
            input_ids = input_ids[0]
        prompt_len = int(input_ids.numel())
        if prompt_len <= 1:
            raise ValueError("prefill graph requires prompt_len > 1")
        graph = self._prefill_graphs.get(prompt_len)
        if graph is None:
            graph = self.capture_prefill_graph(prompt_len)
        ids = self._prefill_ids_buf(prompt_len)
        ids.copy_(input_ids.to(device=self.device, dtype=torch.long))
        self.reset_state()
        if sync:
            torch.cuda.synchronize()
        graph.replay()
        if sync:
            torch.cuda.synchronize()
        logits = self._prefill_logits[prompt_len]
        self._last_logits = logits
        return logits

    def run_decode_eager(
        self,
        token_id: int,
        steps: int,
        *,
        start_pos: int = 0,
        reset_state: bool = True,
    ) -> tuple[list[int], list[Any]]:
        if reset_state:
            self.reset_state()
        token = int(token_id)
        tokens: list[int] = []
        logits_out = []
        for step in range(int(steps)):
            pos = int(start_pos) + step
            logits = self.forward_token(token, pos_start=pos, kv_seq=pos + 1)
            saved = logits.detach().clone()
            token = int(saved[0].float().argmax().item())
            tokens.append(token)
            logits_out.append(saved)
        return tokens, logits_out

    def run_decode_graph(
        self,
        token_id: int,
        steps: int,
        *,
        start_pos: int = 0,
        reset_state: bool = True,
    ) -> tuple[list[int], list[Any]]:
        if reset_state:
            self.reset_state()
        self.capture_decode_graph_table(start_pos, steps, token_id=token_id)
        token = int(token_id)
        tokens: list[int] = []
        logits_out = []
        for step in range(int(steps)):
            pos = int(start_pos) + step
            logits = self.replay_decode_graph(
                token, pos_start=pos, kv_seq=pos + 1, sync=False
            )
            saved = logits.detach().clone()
            token = int(saved[0].float().argmax().item())
            tokens.append(token)
            logits_out.append(saved)
        import torch
        torch.cuda.synchronize()
        return tokens, logits_out

    def generate_from_ids_eager(
        self,
        input_ids,
        *,
        max_new_tokens: int = 1,
        reset_state: bool = True,
    ) -> dict[str, Any]:
        import torch

        if input_ids.dim() == 2:
            if input_ids.shape[0] != 1:
                raise ValueError("only batch size 1 is supported")
            input_ids = input_ids[0]
        prompt_tokens = [int(x) for x in input_ids.tolist()]
        if not prompt_tokens:
            raise ValueError("input_ids must contain at least one token")
        if reset_state:
            self.reset_state()
        logits_trace = []
        next_token = prompt_tokens[-1]
        for pos, token in enumerate(prompt_tokens):
            logits = self.forward_token(token, pos_start=pos, kv_seq=pos + 1)
            saved = logits.detach().clone()
            logits_trace.append(saved)
            next_token = int(saved[0].float().argmax().item())
        generated = []
        pos = len(prompt_tokens)
        for _ in range(int(max_new_tokens)):
            logits = self.forward_token(next_token, pos_start=pos, kv_seq=pos + 1)
            saved = logits.detach().clone()
            logits_trace.append(saved)
            next_token = int(saved[0].float().argmax().item())
            generated.append(next_token)
            pos += 1
        torch.cuda.synchronize()
        return {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated,
            "logits": logits_trace,
        }

    def generate_from_ids_graph(
        self,
        input_ids,
        *,
        max_new_tokens: int = 1,
        reset_state: bool = True,
        collect_logits: bool = True,
    ) -> dict[str, Any]:
        import torch

        if input_ids.dim() == 2:
            if input_ids.shape[0] != 1:
                raise ValueError("only batch size 1 is supported")
            input_ids = input_ids[0]
        prompt_tokens = [int(x) for x in input_ids.tolist()]
        if not prompt_tokens:
            raise ValueError("input_ids must contain at least one token")
        prompt_len = len(prompt_tokens)
        max_new_tokens = int(max_new_tokens)
        if (
            self.persistent_linear_state
            and self.use_full_attn_kv
            and prompt_len > 1
        ):
            self.capture_decode_graph_table(prompt_len, max_new_tokens, token_id=0)
            use_prefill_graph = max_new_tokens <= self.prefill_graph_max_decode_steps
            if use_prefill_graph:
                prompt_logits = self.replay_prefill_graph(input_ids, sync=False)
            else:
                if reset_state:
                    self.reset_state()
                h_prompt = self.runner.top("embed_w").index_select(
                    0, input_ids.to(device=self.device, dtype=torch.long)
                ).contiguous()
                prompt_logits = self.runner.full_logits_fp8(
                    h_prompt, pos_start=0, kv_seq=prompt_len
                )
                logits_trace = (
                    [prompt_logits[i : i + 1].detach().clone() for i in range(prompt_len)]
                    if collect_logits
                    else []
                )
                next_token = int(prompt_logits[-1].float().argmax().item())
                generated = []
                pos = prompt_len
                for _ in range(max_new_tokens):
                    logits = self.replay_decode_graph(
                        next_token, pos_start=pos, kv_seq=pos + 1, sync=False
                    )
                    if collect_logits:
                        logits_trace.append(logits.detach().clone())
                    next_token = int(logits[0].float().argmax().item())
                    generated.append(next_token)
                    pos += 1
                torch.cuda.synchronize()
                return {
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated,
                    "logits": logits_trace,
                }
            logits_trace = (
                [prompt_logits[i : i + 1].detach().clone() for i in range(prompt_len)]
                if collect_logits
                else []
            )
            next_token_buf = torch.empty(1, device=self.device, dtype=torch.long)
            next_token_buf.copy_(prompt_logits[-1:].float().argmax(dim=1).to(torch.long))
            generated = []
            generated_buf = torch.empty(max_new_tokens, device=self.device, dtype=torch.long)
            pos = prompt_len
            for step in range(max_new_tokens):
                self.input_ids_buf.copy_(next_token_buf)
                graph = self._decode_graphs.get((pos, pos + 1))
                if graph is None:
                    graph = self.capture_decode_graph(0, pos_start=pos, kv_seq=pos + 1)
                graph.replay()
                logits = self.logits
                if collect_logits:
                    logits_trace.append(logits.detach().clone())
                next_token_buf.copy_(logits.float().argmax(dim=1).to(torch.long))
                generated_buf[step].copy_(next_token_buf[0])
                pos += 1
            torch.cuda.synchronize()
            generated = [int(x) for x in generated_buf.cpu().tolist()]
            return {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated,
                "logits": logits_trace,
            }

        total_steps = prompt_len + max_new_tokens
        self.capture_decode_graph_table(0, total_steps, token_id=prompt_tokens[0])
        if reset_state:
            self.reset_state()
        logits_trace = []
        next_token = prompt_tokens[-1]
        for pos, token in enumerate(prompt_tokens):
            logits = self.replay_decode_graph(token, pos_start=pos, kv_seq=pos + 1, sync=False)
            saved = logits.detach().clone()
            if collect_logits:
                logits_trace.append(saved)
            next_token = int(saved[0].float().argmax().item())
        generated = []
        pos = prompt_len
        for _ in range(max_new_tokens):
            logits = self.replay_decode_graph(next_token, pos_start=pos, kv_seq=pos + 1, sync=False)
            saved = logits.detach().clone()
            if collect_logits:
                logits_trace.append(saved)
            next_token = int(saved[0].float().argmax().item())
            generated.append(next_token)
            pos += 1
        torch.cuda.synchronize()
        return {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated,
            "logits": logits_trace,
        }

    def bench_graph(self, token_id: int = 0, repeat: int = 10) -> dict[str, float | int]:
        import torch
        import torch.nn.functional as F

        eager = self.forward_token(token_id).detach().clone()
        graph = self.capture_decode_graph(token_id)
        self.logits.zero_()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        graph_logits = self.logits.detach().clone()
        max_abs = float((graph_logits.float() - eager.float()).abs().max().item())
        cos = float(F.cosine_similarity(graph_logits.float().flatten(), eager.float().flatten(), dim=0).item())

        t0 = time.perf_counter()
        for _ in range(int(repeat)):
            graph.replay()
        torch.cuda.synchronize()
        replay_s = (time.perf_counter() - t0) / int(repeat)
        return {
            "token_id": int(token_id),
            "top_token": int(torch.argmax(graph_logits[0]).item()),
            "max_abs_vs_eager": max_abs,
            "cos_vs_eager": cos,
            "graph_replay_s": replay_s,
            "graph_replay_ms": replay_s * 1000.0,
        }

    def generate_token_ids(
        self,
        token_id: int,
        *,
        max_new_tokens: int = 1,
        use_graph: bool = True,
        reset_state: bool = True,
    ) -> dict[str, Any]:
        import torch

        steps = int(max_new_tokens)
        if steps <= 0:
            return {"input_token": int(token_id), "tokens": [], "decode_ms": 0.0, "tok_per_s": 0.0}
        if reset_state:
            self.reset_state()
        if use_graph:
            self.capture_decode_graph_table(0, steps, token_id=token_id)
        token = int(token_id)
        tokens: list[int] = []
        t0 = time.perf_counter()
        for step in range(steps):
            if use_graph:
                logits = self.replay_decode_graph(token, pos_start=step, kv_seq=step + 1, sync=False)
            else:
                logits = self.forward_token(token, pos_start=step, kv_seq=step + 1)
            token = int(logits[0].float().argmax().item())
            tokens.append(token)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.latency_records.append(elapsed_ms)
        return {
            "input_token": int(token_id),
            "tokens": tokens,
            "decode_ms": elapsed_ms,
            "tok_per_s": (1000.0 * steps / elapsed_ms) if elapsed_ms else 0.0,
            "use_graph": bool(use_graph),
            "scope": self.config_summary["graph_scope"],
        }

    def generate_text_stateful(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 1,
        use_graph: bool = True,
    ) -> dict[str, Any]:
        import torch

        ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.device)
        t0 = time.perf_counter()
        if use_graph:
            result = self.generate_from_ids_graph(
                ids, max_new_tokens=max_new_tokens, collect_logits=False
            )
        else:
            result = self.generate_from_ids_eager(ids, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        all_tokens = result["prompt_tokens"] + result["generated_tokens"]
        return {
            "prompt_tokens": len(result["prompt_tokens"]),
            "generated_tokens": result["generated_tokens"],
            "output_ids": all_tokens,
            "text": self.tokenizer.decode(all_tokens, skip_special_tokens=True),
            "decode_ms": elapsed_ms,
            "tok_per_s": (
                1000.0 * int(max_new_tokens) / elapsed_ms if elapsed_ms else 0.0
            ),
            "use_graph": bool(use_graph),
            "scope": self.config_summary["graph_scope"],
        }

    def set_prompt(self, text: str) -> None:
        self._prompt = str(text)

    def infer(self, request: dict | str, debug: bool = False) -> dict[str, Any]:
        import torch

        prompt = request if isinstance(request, str) else request.get("prompt") or request.get("text")
        max_new_tokens = 1 if isinstance(request, str) else int(request.get("max_new_tokens", 1))
        stateful = False if isinstance(request, str) else bool(request.get("stateful", False))
        if not prompt:
            token_id = 0
        else:
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
            token_id = int(ids[-1].item())
        if max_new_tokens > 1 or stateful:
            if not (self.persistent_linear_state and self.use_full_attn_kv):
                raise RuntimeError(
                    "stateful multi-token infer requires persistent_linear_state=True "
                    "and use_full_attn_kv=True"
                )
            if prompt:
                out = self.generate_text_stateful(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    use_graph=True,
                )
            else:
                out = self.generate_token_ids(token_id, max_new_tokens=max_new_tokens, use_graph=True)
            if debug:
                out["debug"] = {"config": self.config_summary}
            return out
        t0 = time.perf_counter()
        logits = self.replay_decode_graph(token_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.latency_records.append(elapsed_ms)
        out = {
            "token_id": token_id,
            "top_token": int(torch.argmax(logits[0]).item()),
            "latency_ms": elapsed_ms,
            "scope": self.config_summary["graph_scope"],
        }
        if debug:
            out["debug"] = {"config": self.config_summary}
        return out

    def get_latency_stats(self) -> dict[str, float]:
        if not self.latency_records:
            return {"count": 0}
        vals = list(self.latency_records)
        return {
            "count": len(vals),
            "mean_ms": sum(vals) / len(vals),
            "min_ms": min(vals),
            "max_ms": max(vals),
        }
