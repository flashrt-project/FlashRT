#!/usr/bin/env python3
"""Route B runtime: MiniMax-M3 with FP8 experts + BF16 resident on one Spark.

Quality path the user chose (E2E cos ~0.99 vs ~0.91 for 4-bit). Differences
from m3_runtime.py (W4A16):
  - resident weights (q/k/v/o, dense MLP, shared experts, router, indexer,
    norms, embed, lm_head) loaded as BF16 directly from the ORIGINAL
    checkpoint (no precision loss; ~9 GB, one-time at startup).
  - routed experts streamed as FP8 block-128 from experts_fp8_layer_NN.bin;
    each expert dequantized W8A16 (fp8_block128_dequantize_to_bf16 -> BF16,
    then BF16 matmul). FP8 block is 56.6 MB (1.8x the FP4 block) -> ~1.8x
    expert streaming traffic, so single-Spark decode is ~half the FP4 rate.

Attention: dense-exact at ctx<2048, torch-level MSA top-16 beyond (P4 Triton
decode-sparse slots in later). Same expert cache + warm-from-trace as W4A16.
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_ref_layerwise import (  # noqa: E402
    DENSE_LAYERS, HEAD_DIM, HIDDEN, IDX_BLOCK, IDX_DIM, IDX_HEADS, IDX_LOCAL,
    IDX_TOPK, KV_HEADS, LAYERS, N_EXPERTS, Q_HEADS, TOP_K,
    apply_rope, rms_norm, rope_cos_sin, swigluoai,
)
from m3_quant_fp8_experts import (  # noqa: E402
    BLOCK_BYTES, INTER, W1_FP8, W1_SCALE, W2_FP8, W2_SCALE, W3_FP8, W3_SCALE,
)

OFF_W1_FP8 = 0
OFF_W1_SCALE = OFF_W1_FP8 + W1_FP8
OFF_W3_FP8 = OFF_W1_SCALE + W1_SCALE
OFF_W3_SCALE = OFF_W3_FP8 + W3_FP8
OFF_W2_FP8 = OFF_W3_SCALE + W3_SCALE
OFF_W2_SCALE = OFF_W2_FP8 + W2_FP8
PFX = "language_model.model."


class ExpertCacheFP8:
    """Two-tier FP8 expert cache (56.6 MB blocks):
      - PINNED tier: per-layer warm set (top experts from the prompt trace),
        loaded once, NEVER evicted. Decode hot experts live here -> hits.
      - STREAM tier: a small shared LRU for everything else (prefill touches
        all 128 experts/layer; cold decode misses). Prefill no longer evicts
        the warm set, so decode hit rate = warm coverage instead of ~prefill
        churn (the 41% -> ~warm% fix).
    Per-expert get() + immediate consume keeps it evict-safe."""

    def __init__(self, qdir, device, budget_gb, workers=8):
        self.device = device
        self.qdir = qdir
        n_total = int(budget_gb * 1e9 // BLOCK_BYTES)
        n_sparse = LAYERS - len(DENSE_LAYERS)
        # reserve most slots for the pinned warm set, keep a stream ring
        self.n_stream = max(32, n_total // 16)
        self.quota = max(4, (n_total - self.n_stream) // n_sparse)
        self.n_pinned = self.quota * n_sparse
        self.n_slots = self.n_pinned + self.n_stream
        self.slots = torch.empty(self.n_slots, BLOCK_BYTES,
                                 dtype=torch.uint8, device=device)
        self.pinned = {i: {} for i in range(LAYERS)}   # layer -> {e: slot}
        self.stream_lru = OrderedDict()                 # (layer,e) -> slot
        self.stream_slots = list(range(self.n_pinned, self.n_slots))
        self._pin_next = 0
        self.fds = {}
        self.stage = [torch.empty(BLOCK_BYTES, dtype=torch.uint8).pin_memory()
                      for _ in range(workers)]
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.hits = self.misses = 0
        self.bytes_streamed = 0

    def _fd(self, layer):
        if layer not in self.fds:
            self.fds[layer] = os.open(
                os.path.join(self.qdir, f"experts_fp8_layer_{layer:02d}.bin"),
                os.O_RDONLY)
        return self.fds[layer]

    def _read(self, layer, e, slot, si=0):
        st = self.stage[si]
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

    def get(self, layer, e):
        """One expert; consume before the next get() (stream slot may be
        reused). Pinned warm-set hits never stream."""
        p = self.pinned[layer]
        slot = p.get(e)
        if slot is not None:
            self.hits += 1
            return int(self.slots[slot].data_ptr())
        self.misses += 1
        key = (layer, e)
        slot = self.stream_lru.get(key)
        if slot is not None:
            self.stream_lru.move_to_end(key)
            return int(self.slots[slot].data_ptr())
        if len(self.stream_lru) >= self.n_stream:
            slot = self.stream_lru.popitem(last=False)[1]
        else:
            slot = self.stream_slots[len(self.stream_lru)]
        self._read(layer, e, slot)
        self.stream_lru[key] = slot
        return int(self.slots[slot].data_ptr())

    def warm(self, trace_path):
        """Pin the top-`quota` experts per layer from the prompt trace."""
        tr = torch.load(trace_path, map_location="cpu", weights_only=False)
        t0 = time.time()
        for i, sel in tr["experts"].items():
            i = int(i)
            counts = torch.bincount(sel.flatten(), minlength=N_EXPERTS)
            top = counts.argsort(descending=True)[: self.quota]
            futs = []
            for e in top:
                slot = self._pin_next
                self._pin_next += 1
                self.pinned[i][int(e)] = slot
                futs.append(self.pool.submit(self._read, i, int(e), slot,
                                             len(futs) % len(self.stage)))
                if len(futs) == len(self.stage):
                    for f in futs:
                        f.result()
                    futs = []
            for f in futs:
                f.result()
        self.bytes_streamed = 0
        print(f"cache warmed {time.time() - t0:.1f}s; pinned "
              f"{self._pin_next} experts, {self.n_stream} stream slots",
              flush=True)

    def stats(self):
        tot = self.hits + self.misses
        return (f"hit {self.hits}/{tot} ({100.0 * self.hits / max(tot, 1):.1f}%), "
                f"streamed {self.bytes_streamed / 1e9:.2f} GB")


class FP8Ctx:
    """W8A16: dequant FP8 block-128 weight -> BF16, matmul with BF16 act."""

    def __init__(self, fvk, device):
        self.fvk = fvk
        self.device = device
        self._deq = None

    def _scratch(self, n):
        if self._deq is None or self._deq.numel() < n:
            self._deq = torch.empty(n, dtype=torch.bfloat16, device=self.device)
        return self._deq[:n]

    def expert_linear(self, x, fp8_ptr, scale_ptr, N, K):
        w = self._scratch(N * K).view(N, K)
        self.fvk.fp8_block128_dequantize_to_bf16(
            fp8_ptr, scale_ptr, int(w.data_ptr()), N, K, 0)
        return F.linear(x, w)


class M3RuntimeFP8:
    def __init__(self, qdir, model_dir, device, fvk, cache_gb=70.0,
                 max_seq=16384):
        from raw_st_reader import RawShardReader
        self.device = device
        self.fvk = fvk
        self.ctx = FP8Ctx(fvk, device)
        self.qdir = qdir
        self.max_seq = max_seq
        rd = RawShardReader(model_dir, device)

        def g(name):
            return rd.get(PFX + name, device).to(torch.bfloat16)

        t0 = time.time()
        self.layers = []
        for i in range(LAYERS):
            p = f"layers.{i}."
            sparse = i not in DENSE_LAYERS
            d = {"sparse": sparse,
                 "q_proj": g(p + "self_attn.q_proj.weight"),
                 "k_proj": g(p + "self_attn.k_proj.weight"),
                 "v_proj": g(p + "self_attn.v_proj.weight"),
                 "o_proj": g(p + "self_attn.o_proj.weight"),
                 "q_norm": g(p + "self_attn.q_norm.weight"),
                 "k_norm": g(p + "self_attn.k_norm.weight"),
                 "input_ln": g(p + "input_layernorm.weight"),
                 "post_ln": g(p + "post_attention_layernorm.weight")}
            if sparse:
                bm = p + "block_sparse_moe."
                d.update(
                    gate_w=g(bm + "gate.weight"),
                    gate_bias=rd.get(PFX + bm + "e_score_correction_bias",
                                     device).float(),
                    sh_gate=g(bm + "shared_experts.gate_proj.weight"),
                    sh_up=g(bm + "shared_experts.up_proj.weight"),
                    sh_down=g(bm + "shared_experts.down_proj.weight"),
                    idx_q_proj=g(p + "self_attn.index_q_proj.weight"),
                    idx_k_proj=g(p + "self_attn.index_k_proj.weight"),
                    idx_q_norm=g(p + "self_attn.index_q_norm.weight"),
                    idx_k_norm=g(p + "self_attn.index_k_norm.weight"))
            else:
                d.update(mlp_gate=g(p + "mlp.gate_proj.weight"),
                         mlp_up=g(p + "mlp.up_proj.weight"),
                         mlp_down=g(p + "mlp.down_proj.weight"))
            self.layers.append(d)
            rd.drop_all()
        self.embed_w = g("embed_tokens.weight")
        self.final_norm = g("norm.weight")
        self.lm_head = rd.get("language_model.lm_head.weight",
                              device).to(torch.bfloat16)
        rd.drop_all()
        print(f"resident BF16 loaded {time.time() - t0:.1f}s; "
              f"gpu {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

        self.cache = ExpertCacheFP8(qdir, device, cache_gb)
        print(f"FP8 expert cache: {self.cache.n_slots} slots "
              f"({self.cache.n_slots * BLOCK_BYTES / 1e9:.1f} GB), "
              f"quota {self.cache.quota}/layer", flush=True)
        self.k_cache = torch.zeros(LAYERS, KV_HEADS, max_seq, HEAD_DIM,
                                   dtype=torch.bfloat16, device=device)
        self.v_cache = torch.zeros_like(self.k_cache)
        # Native tensor-core block-sparse decode: used automatically when the
        # kernel is built into flash_rt_kernels, otherwise SDPA is used. The
        # identity req_to_token (head-major cache -> slot-major paged) is cached.
        self._has_native_decode = hasattr(
            self.fvk, "msa_decode_sparse_attn_mma_paged")
        self._r2t_identity = torch.arange(
            self.k_cache.shape[2], dtype=torch.int32, device=device).view(1, -1)
        self.idx_k_cache = torch.zeros(LAYERS, max_seq, IDX_DIM,
                                       dtype=torch.bfloat16, device=device)
        self.seq_len = 0

    def _expert(self, base, which, x):
        if which == "w1":
            return self.ctx.expert_linear(x, base + OFF_W1_FP8,
                                          base + OFF_W1_SCALE, INTER, HIDDEN)
        if which == "w3":
            return self.ctx.expert_linear(x, base + OFF_W3_FP8,
                                          base + OFF_W3_SCALE, INTER, HIDDEN)
        return self.ctx.expert_linear(x, base + OFF_W2_FP8,
                                      base + OFF_W2_SCALE, HIDDEN, INTER)

    def _indexer(self, d, li, h, cos, sin, pos):
        idx_q = F.linear(h, d["idx_q_proj"]).view(1, 1, IDX_HEADS, IDX_DIM)
        idx_q = rms_norm(idx_q, d["idx_q_norm"]).transpose(1, 2)
        idx_k = F.linear(h, d["idx_k_proj"]).view(1, 1, 1, IDX_DIM)
        idx_k = rms_norm(idx_k, d["idx_k_norm"]).transpose(1, 2)
        idx_q, idx_k = apply_rope(idx_q, idx_k, cos[None, None], sin[None, None])
        self.idx_k_cache[li, pos] = idx_k[0, 0, 0]
        k_len = pos + 1
        hist = self.idx_k_cache[li, :k_len]
        scores = (idx_q[0, :, 0].float() @ hist.float().T)
        nb = -(-k_len // IDX_BLOCK)
        pad = nb * IDX_BLOCK - k_len
        if pad:
            scores = F.pad(scores, (0, pad), value=float("-inf"))
        bs = scores.view(IDX_HEADS, nb, IDX_BLOCK).amax(-1).amax(0)
        qb = pos // IDX_BLOCK
        for j in range(IDX_LOCAL):
            bs[max(qb - j, 0)] = float("inf")
        topk = min(IDX_TOPK, nb)
        sc, bid = bs.topk(topk)
        return bid[sc > float("-inf")]

    def _attn(self, d, li, h, cos, sin, pos, S):
        q = F.linear(h, d["q_proj"]).view(1, S, Q_HEADS, HEAD_DIM)
        k = F.linear(h, d["k_proj"]).view(1, S, KV_HEADS, HEAD_DIM)
        v = F.linear(h, d["v_proj"]).view(1, S, KV_HEADS, HEAD_DIM)
        q = rms_norm(q, d["q_norm"]).transpose(1, 2)
        k = rms_norm(k, d["k_norm"]).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos[None, None], sin[None, None])
        self.k_cache[li, :, pos:pos + S] = k[0]
        self.v_cache[li, :, pos:pos + S] = v[0]
        k_len = pos + S
        sparse = (d["sparse"] and S == 1 and k_len > IDX_TOPK * IDX_BLOCK)
        if sparse:
            bid = self._indexer(d, li, h, cos, sin, pos)
            attn = self._sparse_decode(li, q, bid, k_len)
        else:
            if d["sparse"] and S > 1:
                idx_k = F.linear(h, d["idx_k_proj"]).view(S, 1, IDX_DIM)
                idx_k = rms_norm(idx_k, d["idx_k_norm"]).view(1, 1, S, IDX_DIM)
                z = torch.zeros_like(idx_k)
                _, idx_k = apply_rope(z, idx_k, cos[None, None], sin[None, None])
                self.idx_k_cache[li, pos:pos + S] = idx_k[0, 0]
            kk = self.k_cache[li, :, :k_len][None]
            vv = self.v_cache[li, :, :k_len][None]
            attn = F.scaled_dot_product_attention(
                q, kk, vv, is_causal=(S > 1), enable_gqa=True,
                scale=HEAD_DIM**-0.5)
        attn = attn.transpose(1, 2).reshape(S, Q_HEADS * HEAD_DIM)
        return F.linear(attn, d["o_proj"])

    def _sparse_decode(self, li, q, bid, k_len):
        # Native tensor-core path (cos ~1.0 vs SDPA) when the kernel is built;
        # the head-major [Hkv, seq, D] cache is transposed to the kernel's
        # slot-major paged layout with an identity req_to_token. SDPA fallback.
        if self._has_native_decode:
            Hq, Hkv, D = Q_HEADS, KV_HEADS, HEAD_DIM
            dev = q.device
            qn = q[0, :, 0, :].unsqueeze(0).contiguous()
            kc = self.k_cache[li, :, :k_len].transpose(0, 1).contiguous()
            vc = self.v_cache[li, :, :k_len].transpose(0, 1).contiguous()
            ms = kc.shape[0]
            r2t = self._r2t_identity
            sl = torch.tensor([k_len], dtype=torch.int32, device=dev)
            sid = torch.zeros(1, dtype=torch.int64, device=dev)
            ti = torch.full((Hkv, 1, IDX_TOPK), -1, dtype=torch.int32, device=dev)
            bb = bid.to(torch.int32)
            n = min(IDX_TOPK, bb.numel())
            for kh in range(Hkv):
                ti[kh, 0, :n] = bb[:n]
            out = torch.empty(1, Hq, D, dtype=torch.bfloat16, device=dev)
            self.fvk.msa_decode_sparse_attn_mma_paged(
                qn.data_ptr(), kc.data_ptr(), vc.data_ptr(), r2t.data_ptr(),
                sl.data_ptr(), sid.data_ptr(), ti.data_ptr(), out.data_ptr(),
                1, Hq, Hkv, D, ms, r2t.shape[1], IDX_BLOCK, IDX_TOPK,
                float(HEAD_DIM ** -0.5), 0)
            return out.view(1, Hq, 1, D)
        cols = (bid[:, None] * IDX_BLOCK
                + torch.arange(IDX_BLOCK, device=q.device)[None]).flatten()
        cols = cols[cols < k_len]
        kk = self.k_cache[li, :, cols][None]
        vv = self.v_cache[li, :, cols][None]
        return F.scaled_dot_product_attention(
            q, kk, vv, enable_gqa=True, scale=HEAD_DIM ** -0.5)

    def _moe(self, d, li, flat):
        shared = F.linear(swigluoai(F.linear(flat, d["sh_gate"]),
                                    F.linear(flat, d["sh_up"])), d["sh_down"])
        logits = F.linear(flat, d["gate_w"])
        scores = torch.sigmoid(logits.float())
        _, sel = torch.topk(scores + d["gate_bias"], TOP_K, dim=-1, sorted=False)
        weights = scores.gather(1, sel)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        uniq = [int(e) for e in sel.unique()]
        routed = torch.zeros_like(flat)
        # Per-expert fetch: dequant each expert into scratch and consume it
        # before fetching the next. A layer can route to more unique experts
        # than the cache quota (prefill: up to 128 > quota), so batch-fetching
        # pointers would let later evictions corrupt earlier slots.
        for e in uniq:
            base = self.cache.get(li, e)
            tok, pos = torch.where(sel == e)
            xe = flat[tok]
            cur = swigluoai(self._expert(base, "w1", xe),
                            self._expert(base, "w3", xe))
            cur = self._expert(base, "w2", cur)
            routed.index_add_(0, tok, cur * weights[tok, pos, None].to(cur.dtype))
        return routed * 2.0 + shared

    def forward(self, ids):
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
                out = F.linear(swigluoai(F.linear(h, d["mlp_gate"]),
                                         F.linear(h, d["mlp_up"])), d["mlp_down"])
            x = x + out.view(1, S, HIDDEN)
        self.seq_len = pos0 + S
        x = rms_norm(x, self.final_norm)
        return F.linear(x.view(S, HIDDEN), self.lm_head).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", default="/models/MiniMax-M3-FP8E")
    ap.add_argument("--model", default="/models/MiniMax-M3")
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--cache-gb", type=float, default=70.0)
    ap.add_argument("--warm-trace", default="minimax_dev/ref_out/trace.pt")
    ap.add_argument("--check-ref", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    sys.path.insert(0, "/workspace/FlashRT/flash_rt")
    import flash_rt_kernels as fvk
    from transformers import AutoTokenizer

    rt = M3RuntimeFP8(args.qdir, args.model, args.device, fvk,
                      cache_gb=args.cache_gb)
    if args.warm_trace and os.path.exists(args.warm_trace):
        rt.cache.warm(args.warm_trace)
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.check_ref:
        tr = torch.load(os.path.join(args.check_ref, "trace.pt"),
                        map_location="cpu", weights_only=False)
        ids = tr["prompt_ids"].to(args.device)
        ref_gen = tr["generated_ids"]
        ref_logits = torch.load(os.path.join(args.check_ref,
                                "prefill_logits.pt"), map_location="cpu",
                                weights_only=False)
    else:
        base = ("Explain how unified memory on GB10 changes the design of an "
                "LLM inference engine, focusing on weight streaming. ")
        ids = tok(base * 8, return_tensors="pt",
                  add_special_tokens=False).input_ids[0][:args.prompt_tokens]
        ids = ids.to(args.device)
        ref_gen = ref_logits = None
    print(f"prompt {ids.shape[0]} tokens", flush=True)

    t0 = time.time()
    logits = rt.forward(ids)
    torch.cuda.synchronize()
    print(f"TTFT {time.time() - t0:.2f}s; {rt.cache.stats()}", flush=True)
    if ref_logits is not None:
        c = F.cosine_similarity(logits[-1], ref_logits[-1].to(args.device), dim=0)
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
    tt = torch.tensor(times)
    print(f"\ndecode: mean {tt.mean() * 1000:.0f}ms/tok "
          f"({1.0 / tt.mean():.2f} tok/s), p50 {tt.median() * 1000:.0f}ms; "
          f"{rt.cache.stats()}; "
          f"{rt.cache.bytes_streamed / 1e9 / max(len(times), 1):.2f} GB/tok",
          flush=True)
    print("text:", tok.decode(gen), flush=True)


if __name__ == "__main__":
    main()
