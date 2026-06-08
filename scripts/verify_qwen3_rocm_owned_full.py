#!/usr/bin/env python3
"""Verify full 36-layer owned Qwen3 ROCm BF16 against Torch reference."""

from __future__ import annotations
import os

import argparse

import torch
import torch.nn.functional as F

import flash_rt.flash_rt_rocm_kernels as kernels
from flash_rt.frontends.torch.qwen3_rocm_weights import extract_weights_qwen3_bf16_rocm
from flash_rt.hardware.rocm.attn_backend_qwen3 import RocmQwen3AttnBackend


def _by_ptr(handles):
    return {int(t.data_ptr()): t for t in handles.anchors}


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()
    )


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    inv = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() * inv * w.float()).to(torch.bfloat16)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope(seq: int, head_dim: int, device: str):
    pos = torch.arange(seq, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        1_000_000.0
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().contiguous(), emb.sin().contiguous()


def _attention_ref(q, k, v, causal: bool):
    q_t = q.unsqueeze(0).transpose(1, 2)
    k_t = k.unsqueeze(0).transpose(1, 2)
    v_t = v.unsqueeze(0).transpose(1, 2)
    repeat = q_t.shape[1] // k_t.shape[1]
    k_t = k_t.repeat_interleave(repeat, dim=1)
    v_t = v_t.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(
        q_t, k_t, v_t, dropout_p=0.0, is_causal=causal
    ).transpose(1, 2).squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--seq", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(43)
    handles = extract_weights_qwen3_bf16_rocm(args.checkpoint)
    tensors = _by_ptr(handles)
    ptrs = handles.ptrs

    hidden_size = int(ptrs["hidden"])
    intermediate = int(ptrs["intermediate"])
    q_heads = int(ptrs["num_q_heads"])
    kv_heads = int(ptrs["num_kv_heads"])
    head_dim = int(ptrs["head_dim"])
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    eps = float(ptrs["rms_norm_eps"])
    seq = int(args.seq)

    embed_w = tensors[ptrs["embed_w"]]
    final_norm_w = tensors[ptrs["final_norm_w"]]
    lm_head_w = tensors[ptrs["lm_head_w"]]
    cos, sin = _rope(seq, head_dim, "cuda")

    input_ids = torch.randint(0, int(ptrs["vocab_size"]), (seq,), device="cuda")
    hidden = embed_w.index_select(0, input_ids).contiguous()
    hidden_ref = hidden.clone()

    norm = torch.empty(seq, hidden_size, device="cuda", dtype=torch.bfloat16)
    qkv = torch.empty(seq, qkv_dim, device="cuda", dtype=torch.bfloat16)
    attn_proj = torch.empty_like(hidden)
    post_norm = torch.empty_like(hidden)
    gate_up = torch.empty(seq, 2 * intermediate, device="cuda", dtype=torch.bfloat16)
    act = torch.empty(seq, intermediate, device="cuda", dtype=torch.bfloat16)
    mlp_out = torch.empty_like(hidden)
    final_norm = torch.empty_like(hidden)
    logits = torch.empty(seq, int(ptrs["vocab_size"]), device="cuda", dtype=torch.bfloat16)
    backend = RocmQwen3AttnBackend(
        num_layers=int(ptrs["num_layers"]),
        max_seq=seq,
        max_q_seq=seq,
    )

    for layer_idx, layer in enumerate(ptrs["layers"]):
        input_norm_w = tensors[layer["input_norm_w"]]
        post_norm_w = tensors[layer["post_attn_norm_w"]]
        q_norm_w = tensors[layer["q_norm_w"]]
        k_norm_w = tensors[layer["k_norm_w"]]
        qkv_w = tensors[layer["qkv_w"]]
        o_w = tensors[layer["o_w"]]
        gate_up_w = tensors[layer["gate_up_w"]]
        down_w = tensors[layer["down_w"]]

        residual = hidden.clone()
        kernels.rms_norm_bf16_plain_ptr(
            hidden.data_ptr(),
            input_norm_w.data_ptr(),
            norm.data_ptr(),
            seq,
            hidden_size,
            eps,
        )
        kernels.hipblaslt_linear_bf16_ptr(
            norm.data_ptr(), qkv_w.data_ptr(), 0, qkv.data_ptr(), seq, qkv_dim, hidden_size
        )
        kernels.qwen3_qkv_norm_rope_cache_bf16_ptr(
            qkv.data_ptr(),
            cos.data_ptr(),
            sin.data_ptr(),
            q_norm_w.data_ptr(),
            k_norm_w.data_ptr(),
            backend.q.data_ptr(),
            backend.k_cache.data_ptr(),
            backend.v_cache.data_ptr(),
            layer_idx,
            seq,
            0,
            seq,
            q_heads,
            kv_heads,
            head_dim,
        )
        backend.run(layer_idx, seq, seq, causal=True)
        attn_o = backend.o[:seq].reshape(seq, hidden_size)
        kernels.hipblaslt_linear_bf16_ptr(
            attn_o.data_ptr(),
            o_w.data_ptr(),
            0,
            attn_proj.data_ptr(),
            seq,
            hidden_size,
            hidden_size,
        )
        kernels.residual_add_rms_norm_bf16_plain_ptr(
            residual.data_ptr(),
            attn_proj.data_ptr(),
            post_norm_w.data_ptr(),
            post_norm.data_ptr(),
            seq,
            hidden_size,
            eps,
        )
        kernels.hipblaslt_linear_bf16_ptr(
            post_norm.data_ptr(),
            gate_up_w.data_ptr(),
            0,
            gate_up.data_ptr(),
            seq,
            2 * intermediate,
            hidden_size,
        )
        kernels.silu_mul_merged_bf16_ptr(
            gate_up.data_ptr(), act.data_ptr(), seq, intermediate
        )
        kernels.hipblaslt_linear_bf16_ptr(
            act.data_ptr(),
            down_w.data_ptr(),
            0,
            mlp_out.data_ptr(),
            seq,
            hidden_size,
            intermediate,
        )
        kernels.residual_add_bf16_ptr(
            residual.data_ptr(), mlp_out.data_ptr(), residual.numel()
        )
        hidden = residual

        ref_norm = _rms(hidden_ref, input_norm_w, eps)
        ref_qkv = F.linear(ref_norm, qkv_w)
        ref_q_raw = ref_qkv[:, :q_dim].reshape(seq, q_heads, head_dim)
        ref_k_raw = ref_qkv[:, q_dim : q_dim + kv_dim].reshape(seq, kv_heads, head_dim)
        ref_v = ref_qkv[:, q_dim + kv_dim :].reshape(seq, kv_heads, head_dim)
        ref_qn = _rms(ref_q_raw, q_norm_w, eps)
        ref_kn = _rms(ref_k_raw, k_norm_w, eps)
        ref_q = (
            ref_qn.float() * cos[:, None, :]
            + _rotate_half(ref_qn).float() * sin[:, None, :]
        ).to(torch.bfloat16)
        ref_k = (
            ref_kn.float() * cos[:, None, :]
            + _rotate_half(ref_kn).float() * sin[:, None, :]
        ).to(torch.bfloat16)
        ref_attn = _attention_ref(ref_q, ref_k, ref_v, causal=True).reshape(
            seq, hidden_size
        )
        ref_attn_proj = F.linear(ref_attn, o_w)
        ref_residual = (hidden_ref.float() + ref_attn_proj.float()).to(torch.bfloat16)
        ref_post_norm = _rms(ref_residual, post_norm_w, eps)
        ref_gate_up = F.linear(ref_post_norm, gate_up_w)
        ref_act = (
            F.silu(ref_gate_up[:, :intermediate].float())
            * ref_gate_up[:, intermediate:].float()
        ).to(torch.bfloat16)
        ref_mlp = F.linear(ref_act, down_w)
        hidden_ref = (ref_residual.float() + ref_mlp.float()).to(torch.bfloat16)

    kernels.rms_norm_bf16_plain_ptr(
        hidden.data_ptr(),
        final_norm_w.data_ptr(),
        final_norm.data_ptr(),
        seq,
        hidden_size,
        eps,
    )
    kernels.hipblaslt_linear_bf16_ptr(
        final_norm.data_ptr(),
        lm_head_w.data_ptr(),
        0,
        logits.data_ptr(),
        seq,
        int(ptrs["vocab_size"]),
        hidden_size,
    )
    ref_final_norm = _rms(hidden_ref, final_norm_w, eps)
    ref_logits = F.linear(ref_final_norm, lm_head_w)
    torch.cuda.synchronize()

    for name, out, ref in (
        ("HIDDEN", hidden, hidden_ref),
        ("FINAL_NORM", final_norm, ref_final_norm),
        ("LOGITS", logits, ref_logits),
    ):
        cos_v = _cos(out, ref)
        max_abs = float((out.float() - ref.float()).abs().max().item())
        print(name, "cos", cos_v, "max_abs", max_abs)
        assert cos_v > 0.999
    print("OK qwen3_rocm_owned_full_bf16")


if __name__ == "__main__":
    main()
