#!/usr/bin/env python3
"""P3b bring-up runtime: MiniMax-M3 NVFP4 on a single DGX Spark.

Architecture (per minimax_dev/HANDOFF.md):
  - Resident on GPU: all non-routed-expert weights (NVFP4 packed + BF16
    norms/router/indexer/embed), ~15 GB.
  - Routed experts: layer-quota LRU cache of fixed 31.85 MB packed blocks
    (default ~75 GB ≈ 2300 slots), warmed from the P1 routing trace;
    misses stream from experts_layer_NN.bin via pinned-staging preadv.
  - Attention: ctx < 2048 -> dense causal SDPA (EXACTLY equal to MSA
    top-16 selection at this length); ctx >= 2048 -> torch-level MSA
    (indexer blockmax top-16 + gathered-block SDPA). Triton/CUDA sparse
    kernels replace this in P4.
  - All quantized linears run the real sm_121 FP4 path (dynamic act quant
    + fp4_w4a16_gemm_sm120_bf16out) — W4A4 numerics; the W4A16 GEMV
    kernel (quality ladder, P3c) will slot into Fp4Ctx.linear.

Outputs per run: TTFT, decode tok/s, cache hit/miss + streamed bytes per
token, and (with --check-ref) logits cos + greedy token match vs the P1
BF16 reference on the same prompt.
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_ref_layerwise import (  # noqa: E402
    DENSE_LAYERS, HEAD_DIM, HIDDEN, IDX_BLOCK, IDX_DIM, IDX_HEADS, IDX_LOCAL,
    IDX_TOPK, KV_HEADS, LAYERS, N_EXPERTS, Q_HEADS, TOP_K,
    apply_rope, rms_norm, rope_cos_sin, swigluoai,
)
from m3_nvfp4_layerwise_check import (  # noqa: E402
    Fp4Ctx, Fp4Linear, _gpu_lin, OFF_W1P, OFF_W1S, OFF_W2P, OFF_W2S,
    OFF_W3P, OFF_W3S,
)
from m3_quant_nvfp4 import BLOCK_BYTES, INTER  # noqa: E402


class ExpertCache:
    """Layer-quota LRU cache of packed expert blocks on GPU."""

    def __init__(self, qdir: str, device: str, budget_gb: float, workers=8):
        self.device = device
        self.n_slots = int(budget_gb * 1e9 // BLOCK_BYTES)
        self.quota = max(4, self.n_slots // (LAYERS - len(DENSE_LAYERS)))
        self.slots = torch.empty(self.n_slots, BLOCK_BYTES,
                                 dtype=torch.uint8, device=device)
        self.free = list(range(self.n_slots))
        self.lru = {i: OrderedDict() for i in range(LAYERS)}  # (e)->slot
        self.fds = {}
        self.qdir = qdir
        self.stage = [torch.empty(BLOCK_BYTES, dtype=torch.uint8).pin_memory()
                      for _ in range(workers)]
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.hits = self.misses = 0
        self.bytes_streamed = 0

    def _fd(self, layer):
        if layer not in self.fds:
            self.fds[layer] = os.open(
                os.path.join(self.qdir, f"experts_layer_{layer:02d}.bin"),
                os.O_RDONLY)
        return self.fds[layer]

    def _read_into_slot(self, layer, e, slot, stage_idx=0):
        st = self.stage[stage_idx]
        mv = memoryview(st.numpy())
        fd = self._fd(layer)
        base = e * BLOCK_BYTES
        off = 0
        while off < BLOCK_BYTES:
            r = os.preadv(fd, [mv[off:off + (1 << 28)]], base + off)
            if r <= 0:
                raise IOError(f"short read L{layer} E{e}")
            off += r
        self.slots[slot].copy_(st)
        self.bytes_streamed += BLOCK_BYTES

    def _evict_one(self, layer) -> int:
        lru = self.lru[layer]
        if lru and len(lru) >= self.quota:
            _, slot = lru.popitem(last=False)
            return slot
        if self.free:
            return self.free.pop()
        # steal from the globally fattest layer
        fat = max(self.lru, key=lambda i: len(self.lru[i]))
        _, slot = self.lru[fat].popitem(last=False)
        return slot

    def get(self, layer: int, e: int) -> int:
        """Return device base ptr of the packed block for (layer, e)."""
        lru = self.lru[layer]
        if e in lru:
            lru.move_to_end(e)
            self.hits += 1
            return int(self.slots[lru[e]].data_ptr())
        self.misses += 1
        slot = self._evict_one(layer)
        self._read_into_slot(layer, e, slot)
        lru[e] = slot
        return int(self.slots[slot].data_ptr())

    def get_many(self, layer: int, experts) -> dict:
        """Batch fetch (parallel reads for the misses). -> {e: base_ptr}."""
        lru = self.lru[layer]
        out, missing = {}, []
        for e in experts:
            if e in lru:
                lru.move_to_end(e)
                self.hits += 1
                out[e] = int(self.slots[lru[e]].data_ptr())
            else:
                missing.append(e)
        if missing:
            self.misses += len(missing)
            slots = [self._evict_one(layer) for _ in missing]
            futs = []
            for k, (e, slot) in enumerate(zip(missing, slots)):
                futs.append(self.pool.submit(
                    self._read_into_slot, layer, e, slot,
                    k % len(self.stage)))
            for f in futs:
                f.result()
            for e, slot in zip(missing, slots):
                lru[e] = slot
                out[e] = int(self.slots[slot].data_ptr())
        return out

    def stats(self):
        tot = self.hits + self.misses
        return (f"hit {self.hits}/{tot} ({100.0 * self.hits / max(tot, 1):.1f}%), "
                f"streamed {self.bytes_streamed / 1e9:.2f} GB")


def expert_lin(base: int, alphas, e_row, which: str) -> Fp4Linear:
    if which == "w1":
        return Fp4Linear(base + OFF_W1P, base + OFF_W1S,
                         float(e_row[0]), INTER, HIDDEN)
    if which == "w3":
        return Fp4Linear(base + OFF_W3P, base + OFF_W3S,
                         float(e_row[1]), INTER, HIDDEN)
    return Fp4Linear(base + OFF_W2P, base + OFF_W2S,
                     float(e_row[2]), HIDDEN, INTER)


class M3Runtime:
    def __init__(self, qdir, device, fvk, cache_gb=75.0, max_seq=16384,
                 w4a16=False):
        self.device = device
        self.ctx = Fp4Ctx(fvk, device, w4a16=w4a16)
        self.qdir = qdir
        self.max_seq = max_seq
        t0 = time.time()
        self.layers = []
        for i in range(LAYERS):
            keep = []
            res = torch.load(os.path.join(qdir, f"resident_layer_{i:02d}.pt"),
                             map_location="cpu", weights_only=False)
            g = lambda p: _gpu_lin(res, p, keep, device)  # noqa: E731
            d = {"keep": keep, "sparse": i not in DENSE_LAYERS,
                 "q_proj": g("q_proj"), "k_proj": g("k_proj"),
                 "v_proj": g("v_proj"), "o_proj": g("o_proj"),
                 "q_norm": res["q_norm"].to(device),
                 "k_norm": res["k_norm"].to(device),
                 "input_ln": res["input_ln"].to(device),
                 "post_ln": res["post_ln"].to(device)}
            if d["sparse"]:
                d.update(gate_w=res["gate_w"].to(device),
                         gate_bias=res["gate_bias"].to(device).float(),
                         sh_gate=g("sh_gate"), sh_up=g("sh_up"),
                         sh_down=g("sh_down"),
                         idx_q_proj=res["idx_q_proj"].to(device),
                         idx_k_proj=res["idx_k_proj"].to(device),
                         idx_q_norm=res["idx_q_norm"].to(device),
                         idx_k_norm=res["idx_k_norm"].to(device),
                         alphas=res["expert_alphas"])
            else:
                d.update(mlp_gate=g("mlp_gate"), mlp_up=g("mlp_up"),
                         mlp_down=g("mlp_down"))
            self.layers.append(d)
        top = torch.load(os.path.join(qdir, "resident_top.pt"),
                         map_location="cpu", weights_only=False)
        self._top_keep = []
        self.embed_w = top["embed_w"].to(device)
        self.final_norm = top["final_norm"].to(device)
        self.lm_head = _gpu_lin(top, "lm_head", self._top_keep, device)
        print(f"resident weights loaded in {time.time() - t0:.1f}s; "
              f"gpu mem {torch.cuda.memory_allocated() / 1e9:.1f} GB",
              flush=True)

        self.cache = ExpertCache(qdir, device, cache_gb)
        print(f"expert cache: {self.cache.n_slots} slots "
              f"({self.cache.n_slots * BLOCK_BYTES / 1e9:.1f} GB), "
              f"quota {self.cache.quota}/layer", flush=True)

        # KV caches
        self.k_cache = torch.zeros(LAYERS, KV_HEADS, max_seq, HEAD_DIM,
                                   dtype=torch.bfloat16, device=device)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.idx_k_cache = torch.zeros(LAYERS, max_seq, IDX_DIM,
                                       dtype=torch.bfloat16, device=device)
        self.seq_len = 0

    def warm_from_trace(self, trace_path):
        tr = torch.load(trace_path, map_location="cpu", weights_only=False)
        t0 = time.time()
        for i, sel in tr["experts"].items():
            counts = torch.bincount(sel.flatten(), minlength=N_EXPERTS)
            top = counts.argsort(descending=True)[: self.cache.quota]
            self.cache.get_many(int(i), [int(e) for e in top])
        print(f"cache warmed from trace in {time.time() - t0:.1f}s; "
              f"{self.cache.stats()}", flush=True)
        self.cache.hits = self.cache.misses = 0
        self.cache.bytes_streamed = 0

    # ---- forward ----

    def _indexer_decode(self, d, li, h_flat, cos, sin, pos):
        """Maintain idx_k cache; return selected block ids for ONE query."""
        idx_q = F.linear(h_flat, d["idx_q_proj"]).view(1, 1, IDX_HEADS, IDX_DIM)
        idx_q = rms_norm(idx_q, d["idx_q_norm"]).transpose(1, 2)
        idx_k = F.linear(h_flat, d["idx_k_proj"]).view(1, 1, 1, IDX_DIM)
        idx_k = rms_norm(idx_k, d["idx_k_norm"]).transpose(1, 2)
        idx_q, idx_k = apply_rope(idx_q, idx_k, cos[None, None], sin[None, None])
        self.idx_k_cache[li, pos] = idx_k[0, 0, 0]
        k_len = pos + 1
        hist = self.idx_k_cache[li, :k_len]  # [k_len, D]
        scores = (idx_q[0, :, 0].float() @ hist.float().T)  # [H, k_len]
        n_blocks = -(-k_len // IDX_BLOCK)
        pad = n_blocks * IDX_BLOCK - k_len
        if pad:
            scores = F.pad(scores, (0, pad), value=float("-inf"))
        block_scores = scores.view(IDX_HEADS, n_blocks, IDX_BLOCK)
        block_scores = block_scores.amax(-1).amax(0)  # [n_blocks]
        qb = pos // IDX_BLOCK
        for j in range(IDX_LOCAL):
            block_scores[max(qb - j, 0)] = float("inf")
        topk = min(IDX_TOPK, n_blocks)
        sc, bid = block_scores.topk(topk)
        return bid[sc > float("-inf")]

    def _attn(self, d, li, h_flat, cos, sin, pos, S):
        ctx = self.ctx
        q = ctx.linear(h_flat, d["q_proj"]).view(1, S, Q_HEADS, HEAD_DIM)
        k = ctx.linear(h_flat, d["k_proj"]).view(1, S, KV_HEADS, HEAD_DIM)
        v = ctx.linear(h_flat, d["v_proj"]).view(1, S, KV_HEADS, HEAD_DIM)
        q = rms_norm(q, d["q_norm"]).transpose(1, 2)
        k = rms_norm(k, d["k_norm"]).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos[None, None], sin[None, None])
        self.k_cache[li, :, pos:pos + S] = k[0]
        self.v_cache[li, :, pos:pos + S] = v[0]
        k_len = pos + S

        use_sparse = (d["sparse"] and S == 1
                      and k_len > IDX_TOPK * IDX_BLOCK)
        if d["sparse"]:
            if S == 1:
                bid = self._indexer_decode(d, li, h_flat, cos, sin, pos)
            else:  # prefill: maintain idx_k for the whole prompt (batch)
                idx_k = F.linear(h_flat, d["idx_k_proj"]).view(S, 1, IDX_DIM)
                idx_k = rms_norm(idx_k, d["idx_k_norm"]).view(1, 1, S, IDX_DIM)
                z = torch.zeros(1, 1, S, IDX_DIM, dtype=idx_k.dtype,
                                device=idx_k.device)
                _, idx_k = apply_rope(z, idx_k, cos[None, None], sin[None, None])
                self.idx_k_cache[li, pos:pos + S] = idx_k[0, 0]

        if use_sparse:
            kk = self.k_cache[li, :, :k_len]
            vv = self.v_cache[li, :, :k_len]
            cols = (bid[:, None] * IDX_BLOCK
                    + torch.arange(IDX_BLOCK, device=q.device)[None]).flatten()
            cols = cols[cols < k_len]
            attn = F.scaled_dot_product_attention(
                q, kk[None, :, cols], vv[None, :, cols],
                enable_gqa=True, scale=HEAD_DIM**-0.5)
        else:
            kk = self.k_cache[li, :, :k_len][None]
            vv = self.v_cache[li, :, :k_len][None]
            if S == 1:
                attn = F.scaled_dot_product_attention(
                    q, kk, vv, enable_gqa=True, scale=HEAD_DIM**-0.5)
            else:
                attn = F.scaled_dot_product_attention(
                    q, kk, vv, is_causal=True, enable_gqa=True,
                    scale=HEAD_DIM**-0.5)
        attn = attn.transpose(1, 2).reshape(S, Q_HEADS * HEAD_DIM)
        return self.ctx.linear(attn, d["o_proj"])

    def _moe(self, d, li, flat):
        ctx = self.ctx
        shared = ctx.linear(
            swigluoai(ctx.linear(flat, d["sh_gate"]),
                      ctx.linear(flat, d["sh_up"])), d["sh_down"])
        logits = F.linear(flat.to(d["gate_w"].dtype), d["gate_w"])
        scores = torch.sigmoid(logits.float())
        _, sel = torch.topk(scores + d["gate_bias"], TOP_K, dim=-1,
                            sorted=False)
        weights = scores.gather(1, sel)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        uniq = [int(e) for e in sel.unique()]
        routed = torch.zeros_like(flat)
        al = d["alphas"]
        # Per-expert fetch + immediate consume: a layer can route to more
        # unique experts than the cache quota (prefill), so batch-fetching
        # pointers would let later self-evictions corrupt earlier slots.
        for e in uniq:
            base = self.cache.get(li, e)
            tok, p = torch.where(sel == e)
            xe = flat[tok]
            cur = swigluoai(
                ctx.linear(xe, expert_lin(base, al, al[e], "w1")),
                ctx.linear(xe, expert_lin(base, al, al[e], "w3")))
            cur = ctx.linear(cur, expert_lin(base, al, al[e], "w2"))
            routed.index_add_(0, tok, cur * weights[tok, p, None].to(cur.dtype))
        return routed * 2.0 + shared

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [S] at positions seq_len..seq_len+S-1 -> logits [S, V]."""
        S = ids.shape[0]
        pos0 = self.seq_len
        position_ids = torch.arange(pos0, pos0 + S, device=self.device)
        cos, sin = rope_cos_sin(position_ids, self.device, torch.bfloat16)
        x = F.embedding(ids, self.embed_w)[None]
        for li, d in enumerate(self.layers):
            h = rms_norm(x, d["input_ln"]).view(S, HIDDEN)
            x = x + self._attn(d, li, h, cos, sin, pos0, S).view(1, S, HIDDEN)
            h = rms_norm(x, d["post_ln"]).view(S, HIDDEN)
            if d["sparse"]:
                out = self._moe(d, li, h)
            else:
                out = self.ctx.linear(
                    swigluoai(self.ctx.linear(h, d["mlp_gate"]),
                              self.ctx.linear(h, d["mlp_up"])), d["mlp_down"])
            x = x + out.view(1, S, HIDDEN)
        self.seq_len = pos0 + S
        x = rms_norm(x, self.final_norm)
        return self.ctx.linear(x.view(S, HIDDEN), self.lm_head).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", default="/models/MiniMax-M3-NVFP4")
    ap.add_argument("--model", default="/models/MiniMax-M3")
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--cache-gb", type=float, default=75.0)
    ap.add_argument("--warm-trace", default="minimax_dev/ref_out/trace.pt")
    ap.add_argument("--check-ref", default="",
                    help="ref_out dir: replay its prompt and compare")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--w4a16", action="store_true",
                    help="W4A16 quality path (dequant weights, BF16 act)")
    args = ap.parse_args()

    sys.path.insert(0, "/workspace/FlashRT/flash_rt")
    import flash_rt_kernels as fvk

    rt = M3Runtime(args.qdir, args.device, fvk, cache_gb=args.cache_gb,
                   w4a16=args.w4a16)
    if args.warm_trace and os.path.exists(args.warm_trace):
        rt.warm_from_trace(args.warm_trace)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.check_ref:
        tr = torch.load(os.path.join(args.check_ref, "trace.pt"),
                        map_location="cpu", weights_only=False)
        ids = tr["prompt_ids"].to(args.device)
        ref_gen = tr["generated_ids"]
        ref_logits = torch.load(
            os.path.join(args.check_ref, "prefill_logits.pt"),
            map_location="cpu", weights_only=False)
    else:
        base = ("Explain how unified memory on GB10 changes the design of "
                "an LLM inference engine, focusing on weight streaming. ")
        ids = tok(base * 8, return_tensors="pt",
                  add_special_tokens=False).input_ids[0]
        ids = ids[: args.prompt_tokens].to(args.device)
        ref_gen, ref_logits = None, None
    print(f"prompt: {ids.shape[0]} tokens", flush=True)

    t0 = time.time()
    logits = rt.forward(ids)
    torch.cuda.synchronize()
    ttft = time.time() - t0
    print(f"TTFT(prefill): {ttft:.2f}s; {rt.cache.stats()}", flush=True)
    if ref_logits is not None:
        c = F.cosine_similarity(logits[-1],
                                ref_logits[-1].to(args.device), dim=0)
        print(f"prefill logits cos vs ref: {float(c):.5f}", flush=True)

    rt.cache.hits = rt.cache.misses = 0
    rt.cache.bytes_streamed = 0
    gen = [int(logits[-1].argmax())]
    times = []
    for step in range(args.max_new - 1):
        t0 = time.time()
        logits = rt.forward(torch.tensor([gen[-1]], device=args.device))
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        gen.append(int(logits[-1].argmax()))
        if ref_gen is not None and step + 1 < len(ref_gen):
            ok = "=" if gen[-1] == ref_gen[step + 1] else "!"
            print(f"  step {step}: {times[-1] * 1000:.0f}ms {ok}", flush=True)

    times_t = torch.tensor(times)
    print(f"\ndecode: mean {times_t.mean() * 1000:.0f}ms/tok "
          f"({1.0 / times_t.mean():.2f} tok/s), "
          f"p50 {times_t.median() * 1000:.0f}ms; {rt.cache.stats()}",
          flush=True)
    print(f"streamed/token: "
          f"{rt.cache.bytes_streamed / 1e9 / max(len(times), 1):.2f} GB",
          flush=True)
    print("text:", tok.decode(gen), flush=True)


if __name__ == "__main__":
    main()
