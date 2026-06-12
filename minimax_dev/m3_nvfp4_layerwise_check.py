#!/usr/bin/env python3
"""P3a: NVFP4 end-to-end accumulation check vs the BF16 reference (P1).

Streams the quantized artifacts (/models/MiniMax-M3-NVFP4) layer by layer and
runs the SAME prompt as m3_ref_layerwise.py, with every quantized linear going
through the real sm_121 FP4 GEMM (`fp4_w4a16_gemm_sm120_bf16out`) and dynamic
activation quantization — i.e. exactly the hot-path numerics of the future
runtime. Compares per-layer last-token hidden states and final logits against
minimax_dev/ref_out/{trace.pt,prefill_logits.pt}.

This is the go/no-go gate for 60-layer 4-bit accumulation BEFORE we invest in
the expert-cache/streaming runtime. If cos collapses, we localize the first
diverging layer (we have per-layer reference hiddens) and adjust the precision
mix cheaply.

Note on attention: at ctx < 2048 tokens (= 16 blocks of 128), MSA's top-16
block selection covers every visible block, so sparse attention degenerates to
EXACT dense causal attention. This check therefore skips the indexer and uses
dense causal SDPA for all layers — mathematically identical at this length.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_ref_layerwise import (  # noqa: E402
    DENSE_LAYERS, HEAD_DIM, HIDDEN, KV_HEADS, KVState, LAYERS, N_EXPERTS,
    PFX, Q_HEADS, TOP_K, apply_rope, rms_norm, rope_cos_sin, swigluoai,
)
from m3_quant_nvfp4 import (  # noqa: E402
    BLOCK_BYTES, W1_PACKED, W1_SF, W2_PACKED, W2_SF, W3_PACKED, W3_SF,
    INTER, _sf_bytes,
)

OFF_W1P = 0
OFF_W1S = OFF_W1P + W1_PACKED
OFF_W3P = OFF_W1S + W1_SF
OFF_W3S = OFF_W3P + W3_PACKED
OFF_W2P = OFF_W3S + W3_SF
OFF_W2S = OFF_W2P + W2_PACKED


class Fp4Linear:
    """One quantized linear: packed weight + swizzled SF + alpha, by pointer."""

    def __init__(self, b_ptr: int, sfb_ptr: int, alpha: float, N: int, K: int):
        self.b_ptr, self.sfb_ptr, self.alpha, self.N, self.K = (
            b_ptr, sfb_ptr, alpha, N, K)


class Fp4Ctx:
    def __init__(self, fvk, device):
        self.fvk = fvk
        self.device = device

    def linear(self, x: torch.Tensor, lin) -> torch.Tensor:
        """x [M, K] bf16 -> [M, N] bf16. `lin` is either an Fp4Linear
        (dynamic-act-quant FP4 GEMM) or a BF16 weight tensor (ablation:
        that component bypasses quantization entirely)."""
        if isinstance(lin, torch.Tensor):
            return F.linear(x, lin)
        M, K = x.shape
        assert K == lin.K, (K, lin.K)
        x = x.contiguous()
        a_packed = torch.empty(M, K // 2, dtype=torch.uint8, device=self.device)
        a_sf = torch.zeros(_sf_bytes(M, K), dtype=torch.uint8, device=self.device)
        self.fvk.quantize_bf16_to_nvfp4_swizzled(
            int(x.data_ptr()), int(a_packed.data_ptr()), int(a_sf.data_ptr()),
            M, K, 0)
        out = torch.zeros(M, lin.N, dtype=torch.bfloat16, device=self.device)
        self.fvk.fp4_w4a16_gemm_sm120_bf16out(
            int(a_packed.data_ptr()), lin.b_ptr, int(out.data_ptr()),
            M, lin.N, K, int(a_sf.data_ptr()), lin.sfb_ptr, lin.alpha, 0)
        return out


def _gpu_lin(res: dict, prefix: str, handles: list, device) -> Fp4Linear:
    packed = res[prefix + "_packed"].to(device)
    sf = res[prefix + "_sf"].to(device)
    handles += [packed, sf]  # keep alive
    N, K2 = packed.shape
    return Fp4Linear(int(packed.data_ptr()), int(sf.data_ptr()),
                     float(res[prefix + "_alpha"]), N, K2 * 2)


class QuantLayer:
    """One decoder layer materialized on GPU from the NVFP4 artifacts."""

    def __init__(self, i: int, qdir: str, device: str,
                 bf16_parts=frozenset(), rd=None):
        self.i = i
        self.sparse = i not in DENSE_LAYERS
        self.bf16_parts = bf16_parts
        self._keep = []
        pre = f"{PFX}layers.{i}."
        res = torch.load(os.path.join(qdir, f"resident_layer_{i:02d}.pt"),
                         map_location="cpu", weights_only=False)
        g = lambda p: _gpu_lin(res, p, self._keep, device)  # noqa: E731

        def orig(name):
            t = rd.get(pre + name, device).to(torch.bfloat16)
            self._keep.append(t)
            return t

        if "attn" in bf16_parts:
            self.q_proj = orig("self_attn.q_proj.weight")
            self.k_proj = orig("self_attn.k_proj.weight")
            self.v_proj = orig("self_attn.v_proj.weight")
        else:
            self.q_proj, self.k_proj = g("q_proj"), g("k_proj")
            self.v_proj = g("v_proj")
        self.o_proj = (orig("self_attn.o_proj.weight")
                       if {"attn", "o"} & bf16_parts else g("o_proj"))
        self.q_norm = res["q_norm"].to(device)
        self.k_norm = res["k_norm"].to(device)
        self.input_ln = res["input_ln"].to(device)
        self.post_ln = res["post_ln"].to(device)
        if not self.sparse:
            if "dense" in bf16_parts:
                self.mlp_gate = orig("mlp.gate_proj.weight")
                self.mlp_up = orig("mlp.up_proj.weight")
                self.mlp_down = orig("mlp.down_proj.weight")
            else:
                self.mlp_gate, self.mlp_up = g("mlp_gate"), g("mlp_up")
                self.mlp_down = g("mlp_down")
        else:
            self.gate_w = res["gate_w"].to(device)
            self.gate_bias = res["gate_bias"].to(device)
            if "shared" in bf16_parts:
                b = "block_sparse_moe.shared_experts."
                self.sh_gate = orig(b + "gate_proj.weight")
                self.sh_up = orig(b + "up_proj.weight")
                self.sh_down = orig(b + "down_proj.weight")
            else:
                self.sh_gate, self.sh_up = g("sh_gate"), g("sh_up")
                self.sh_down = g("sh_down")
            if {"w13", "w2"} & bf16_parts:
                b = f"{pre}block_sparse_moe.experts."
                names, self._orig_experts = [], {}
                wanted = ([("w1", "w13"), ("w3", "w13"), ("w2", "w2")])
                for wn, part in wanted:
                    if part in bf16_parts:
                        ts = rd.get_many(
                            [f"{b}{e}.{wn}.weight" for e in range(N_EXPERTS)],
                            device="cpu")
                        st = torch.stack(ts).to(device)
                        self._keep.append(st)
                        self._orig_experts[wn] = st
            else:
                self._orig_experts = {}
            self.alphas = res["expert_alphas"]  # [128, 3] cpu f32
            # whole expert bin -> GPU u8 [128, BLOCK_BYTES]
            path = os.path.join(qdir, f"experts_layer_{i:02d}.bin")
            fd = os.open(path, os.O_RDONLY)
            buf = torch.empty(N_EXPERTS, BLOCK_BYTES, dtype=torch.uint8)
            mv = memoryview(buf.numpy().reshape(-1))
            off, n = 0, N_EXPERTS * BLOCK_BYTES
            while off < n:
                r = os.preadv(fd, [mv[off:off + (1 << 28)]], off)
                if r <= 0:
                    raise IOError(f"short read {path} @ {off}")
                off += r
            os.close(fd)
            self.experts = buf.to(device)
            self._keep.append(self.experts)

    def expert_lin(self, e: int, which: str):
        if which in self._orig_experts:
            return self._orig_experts[which][e]
        base = int(self.experts[e].data_ptr())
        if which == "w1":
            return Fp4Linear(base + OFF_W1P, base + OFF_W1S,
                             float(self.alphas[e, 0]), INTER, HIDDEN)
        if which == "w3":
            return Fp4Linear(base + OFF_W3P, base + OFF_W3S,
                             float(self.alphas[e, 1]), INTER, HIDDEN)
        return Fp4Linear(base + OFF_W2P, base + OFF_W2S,
                         float(self.alphas[e, 2]), HIDDEN, INTER)

    def free(self):
        self._keep = []
        self.__dict__ = {"i": self.i, "sparse": self.sparse}


def layer_forward_q(ctx: Fp4Ctx, w: QuantLayer, x, cos, sin, position_ids,
                    kv: KVState):
    """Quantized layer forward, batch=1. Dense causal attn (ctx<2048 exact)."""
    S = x.shape[1]
    res = x
    h = rms_norm(x, w.input_ln)
    flat = h.view(S, HIDDEN)

    q = ctx.linear(flat, w.q_proj).view(1, S, Q_HEADS, HEAD_DIM)
    k = ctx.linear(flat, w.k_proj).view(1, S, KV_HEADS, HEAD_DIM)
    v = ctx.linear(flat, w.v_proj).view(1, S, KV_HEADS, HEAD_DIM)
    q = rms_norm(q, w.q_norm).transpose(1, 2)
    k = rms_norm(k, w.k_norm).transpose(1, 2)
    v = v.transpose(1, 2)
    q, k = apply_rope(q, k, cos[None, None], sin[None, None])
    k, v = kv.append(w.i, k, v)
    k_len = k.shape[2]

    k_pos = torch.arange(k_len, device=x.device)
    causal = k_pos[None, :] <= position_ids[:, None]
    mask = torch.zeros((1, 1, S, k_len), dtype=q.dtype, device=x.device)
    mask = mask.masked_fill(~causal[None, None], float("-inf"))
    attn = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, enable_gqa=True, scale=HEAD_DIM**-0.5)
    attn = attn.transpose(1, 2).reshape(S, Q_HEADS * HEAD_DIM)
    x = res + ctx.linear(attn, w.o_proj).view(1, S, HIDDEN)

    res = x
    h = rms_norm(x, w.post_ln)
    flat = h.view(S, HIDDEN)
    if not w.sparse:
        out = ctx.linear(
            swigluoai(ctx.linear(flat, w.mlp_gate), ctx.linear(flat, w.mlp_up)),
            w.mlp_down)
    else:
        shared = ctx.linear(
            swigluoai(ctx.linear(flat, w.sh_gate), ctx.linear(flat, w.sh_up)),
            w.sh_down)
        logits = F.linear(flat.to(w.gate_w.dtype), w.gate_w)
        scores = torch.sigmoid(logits.float())
        _, sel = torch.topk(scores + w.gate_bias.float(), TOP_K, dim=-1,
                            sorted=False)
        weights = scores.gather(1, sel)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        routed = torch.zeros_like(flat)
        for e in sel.unique():
            tok, pos = torch.where(sel == e)
            xe = flat[tok]
            cur = swigluoai(ctx.linear(xe, w.expert_lin(int(e), "w1")),
                            ctx.linear(xe, w.expert_lin(int(e), "w3")))
            cur = ctx.linear(cur, w.expert_lin(int(e), "w2"))
            cur = cur * weights[tok, pos, None].to(cur.dtype)
            routed.index_add_(0, tok, cur)
        out = routed * 2.0 + shared
    return res + out.view(1, S, HIDDEN)


def full_forward_q(ctx, qdir, kv, ids, position_ids, embed_w, norm_w,
                   lm_head, hidden_ref, tag, bf16_parts=frozenset(), rd=None):
    device = ids.device
    x = F.embedding(ids, embed_w)[None]
    cos, sin = rope_cos_sin(position_ids, device, x.dtype)
    coss = []
    for i in range(LAYERS):
        t0 = time.time()
        w = QuantLayer(i, qdir, device, bf16_parts, rd)
        x = layer_forward_q(ctx, w, x, cos, sin, position_ids, kv)
        w.free()
        if hidden_ref is not None:
            ref = hidden_ref[i].to(device)  # [n_passes, H]; pass 0 = prefill
            c = F.cosine_similarity(x[0, -1].float(), ref[0].float(), dim=0)
            coss.append(float(c))
        if i % 8 == 0:
            torch.cuda.empty_cache()
            print(f"    [{tag}] layer {i}: {time.time() - t0:.1f}s"
                  + (f" hid_cos {coss[-1]:.5f}" if coss else ""), flush=True)
    x = rms_norm(x, norm_w)
    logits = ctx.linear(x.view(-1, HIDDEN), lm_head).float()
    return logits, coss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", default="/models/MiniMax-M3-NVFP4")
    ap.add_argument("--ref", default="minimax_dev/ref_out")
    ap.add_argument("--model", default="/models/MiniMax-M3")
    ap.add_argument("--max-new", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bf16-parts", default="",
                    help="comma list of components to run as original BF16 "
                         "(ablation): attn,o,dense,shared,w13,w2,lm_head")
    args = ap.parse_args()

    sys.path.insert(0, "/workspace/FlashRT/flash_rt")
    import flash_rt_kernels as fvk
    from raw_st_reader import RawShardReader

    device = args.device
    ctx = Fp4Ctx(fvk, device)
    bf16_parts = frozenset(p for p in args.bf16_parts.split(",") if p)
    rd = RawShardReader(args.model, device)
    print(f"bf16 ablation parts: {sorted(bf16_parts) or 'none'}", flush=True)

    trace = torch.load(os.path.join(args.ref, "trace.pt"),
                       map_location="cpu", weights_only=False)
    ref_logits = torch.load(os.path.join(args.ref, "prefill_logits.pt"),
                            map_location="cpu", weights_only=False)
    ids = trace["prompt_ids"].to(device)
    ref_gen = trace["generated_ids"]
    hidden_ref = trace["hidden_last"]
    print(f"prompt tokens: {ids.shape[0]}; ref first tokens: {ref_gen[:4]}",
          flush=True)

    top = torch.load(os.path.join(args.qdir, "resident_top.pt"),
                     map_location="cpu", weights_only=False)
    embed_w = top["embed_w"].to(device)
    norm_w = top["final_norm"].to(device)
    keep = []
    if "lm_head" in bf16_parts:
        lm_head = rd.get("language_model.lm_head.weight",
                         device).to(torch.bfloat16)
    else:
        lm_head = _gpu_lin(top, "lm_head", keep, device)

    kv = KVState()
    pos = torch.arange(ids.shape[0], device=device)
    t0 = time.time()
    logits, coss = full_forward_q(ctx, args.qdir, kv, ids, pos, embed_w,
                                  norm_w, lm_head, hidden_ref, "prefill",
                                  bf16_parts, rd)
    print(f"prefill {time.time() - t0:.1f}s", flush=True)

    print("\nper-layer last-token hidden cos vs BF16 ref:")
    print("  min {:.5f} @ layer {}  | last layer {:.5f}".format(
        min(coss), coss.index(min(coss)), coss[-1]), flush=True)
    lc = F.cosine_similarity(logits[-1], ref_logits[-1].to(device), dim=0)
    match = int(logits[-1].argmax()) == ref_gen[0]
    print(f"final logits cos (last tok): {float(lc):.5f}; "
          f"argmax {'MATCH' if match else 'MISMATCH'} "
          f"({int(logits[-1].argmax())} vs {ref_gen[0]})", flush=True)

    gen = [int(logits[-1].argmax())]
    for step in range(args.max_new - 1):
        cur = torch.tensor([gen[-1]], device=device)
        p = torch.tensor([ids.shape[0] + step], device=device)
        logits, _ = full_forward_q(ctx, args.qdir, kv, cur, p, embed_w,
                                   norm_w, lm_head, None, f"step{step}",
                                   bf16_parts, rd)
        gen.append(int(logits[-1].argmax()))
        ok = "MATCH" if gen[-1] == ref_gen[step + 1] else "MISMATCH"
        print(f"step {step}: {gen[-1]} vs ref {ref_gen[step + 1]} {ok}",
              flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
