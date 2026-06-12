#!/usr/bin/env python3
"""Layerwise disk-streaming BF16 reference for MiniMax-M3 (text path).

P1 deliverables:
  1. Reference logits on a real prompt (the cos red-line anchor for all later
     NVFP4 / kernel work — this is the ORIGINAL-model reference).
  2. Expert routing trace per layer per token (feeds streaming-cache design).
  3. Per-layer wall/IO timing on this exact Spark NVMe (feeds P3 projections).

Runs the 60-layer model one layer at a time: load that layer's tensors from the
original safetensors shards -> GPU -> forward -> free. Peak GPU residency is one
MoE layer (~14.5 GB BF16) + embeddings + KV, so it fits the 121 GB UMA box with
huge margin. Each full forward streams ~850 GB from NVMe (~2-4 min).

Math follows transformers main `modular_minimax_m3_vl.py` exactly:
  - Gemma-style RMSNorm: fp32 normalize, scale by (1 + weight)
  - per-head QK-norm on head_dim for main attn AND indexer heads
  - partial RoPE: rotary_dim=64 of head_dim=128, half-half (non-interleaved)
  - router: fp32 sigmoid scores; top-4 chosen on (scores + e_score_correction_bias);
    weights = raw sigmoid scores of chosen, normalized to sum 1 (bias NOT in weights)
  - expert/dense/shared MLP: swigluoai with clamp(gate<=7, |up|<=7),
    glu = gate*sigmoid(1.702*gate), out = down((up+1)*glu)
  - MoE combine: routed * 2.0 + shared
  - sparse attention (layers 3..59): lightning indexer, fp32 scores, blockmax over
    block=128 then over 4 index heads, local block forced, top-16 blocks
Validation plan: coherent greedy generation is the first gate; a per-layer
cross-check against HF's MiniMaxM3VLDecoderLayer on real weights is TODO.
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.manual_seed(0)

# ---- static dims (verified against config.json of MiniMaxAI/MiniMax-M3) ----
HIDDEN = 6144
LAYERS = 60
Q_HEADS = 64
KV_HEADS = 4
HEAD_DIM = 128
ROTARY_DIM = 64
ROPE_THETA = 5_000_000.0
RMS_EPS = 1e-6
VOCAB = 200064
N_EXPERTS = 128
TOP_K = 4
INTER = 3072
DENSE_INTER = 12288
SHARED_INTER = 3072
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
ROUTED_SCALING = 2.0
IDX_HEADS = 4
IDX_DIM = 128
IDX_BLOCK = 128
IDX_TOPK = 16
IDX_LOCAL = 1
DENSE_LAYERS = {0, 1, 2}  # dense MLP + full attention; 3..59 are MoE + sparse attn

PFX = "language_model.model."


class ShardReader:
    """weight_map-driven tensor loader over the original safetensors shards."""

    def __init__(self, model_dir: str, device: str):
        self.model_dir = model_dir
        self.device = device
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.weight_map = idx["weight_map"]
        self._handles = {}
        self.bytes_read = 0

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                os.path.join(self.model_dir, shard), framework="pt", device="cpu"
            )
        return self._handles[shard]

    def get(self, name: str) -> torch.Tensor:
        t = self._handle(self.weight_map[name]).get_tensor(name)
        self.bytes_read += t.numel() * t.element_size()
        return t.to(self.device, non_blocking=False)


def rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Gemma-style: fp32 normalize over last dim, scale by (1 + weight)."""
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return (out * (1.0 + w.float())).to(x.dtype)


def rope_cos_sin(position_ids: torch.Tensor, device, dtype):
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32, device=device) / ROTARY_DIM)
    )
    freqs = position_ids[:, None].float() * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)  # [S, ROTARY_DIM]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Partial RoPE on the leading cos.shape[-1] dims; tail passes through."""
    rot = cos.shape[-1]
    q_rot, q_pass = q[..., :rot], q[..., rot:]
    k_rot, k_pass = k[..., :rot], k[..., rot:]
    q_emb = q_rot * cos + rotate_half(q_rot) * sin
    k_emb = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat([q_emb, q_pass], dim=-1), torch.cat([k_emb, k_pass], dim=-1)


def swigluoai(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate = gate.clamp(max=SWIGLU_LIMIT)
    up = up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    glu = gate * torch.sigmoid(gate * SWIGLU_ALPHA)
    return (up + 1.0) * glu


class LayerWeights:
    """Loads and owns one decoder layer's tensors on GPU."""

    def __init__(self, rd: ShardReader, i: int):
        p = f"{PFX}layers.{i}."
        g = rd.get
        self.i = i
        self.sparse = i not in DENSE_LAYERS
        self.input_ln = g(p + "input_layernorm.weight")
        self.post_ln = g(p + "post_attention_layernorm.weight")
        self.q_proj = g(p + "self_attn.q_proj.weight")
        self.k_proj = g(p + "self_attn.k_proj.weight")
        self.v_proj = g(p + "self_attn.v_proj.weight")
        self.o_proj = g(p + "self_attn.o_proj.weight")
        self.q_norm = g(p + "self_attn.q_norm.weight")
        self.k_norm = g(p + "self_attn.k_norm.weight")
        if self.sparse:
            self.idx_q_proj = g(p + "self_attn.index_q_proj.weight")
            self.idx_k_proj = g(p + "self_attn.index_k_proj.weight")
            self.idx_q_norm = g(p + "self_attn.index_q_norm.weight")
            self.idx_k_norm = g(p + "self_attn.index_k_norm.weight")
            b = p + "block_sparse_moe."
            self.gate_w = g(b + "gate.weight")
            self.gate_bias = g(b + "e_score_correction_bias")
            self.sh_gate = g(b + "shared_experts.gate_proj.weight")
            self.sh_up = g(b + "shared_experts.up_proj.weight")
            self.sh_down = g(b + "shared_experts.down_proj.weight")
            # grouped expert tensors: w1=gate [I,H], w3=up [I,H], w2=down [H,I]
            # preallocate + copy_ per expert: avoids the 2x transient of stack()
            self.w1 = torch.empty(N_EXPERTS, INTER, HIDDEN, dtype=torch.bfloat16, device=rd.device)
            self.w3 = torch.empty(N_EXPERTS, INTER, HIDDEN, dtype=torch.bfloat16, device=rd.device)
            self.w2 = torch.empty(N_EXPERTS, HIDDEN, INTER, dtype=torch.bfloat16, device=rd.device)
            for e in range(N_EXPERTS):
                self.w1[e].copy_(g(f"{b}experts.{e}.w1.weight"))
                self.w3[e].copy_(g(f"{b}experts.{e}.w3.weight"))
                self.w2[e].copy_(g(f"{b}experts.{e}.w2.weight"))
        else:
            self.mlp_gate = g(p + "mlp.gate_proj.weight")
            self.mlp_up = g(p + "mlp.up_proj.weight")
            self.mlp_down = g(p + "mlp.down_proj.weight")

    def free(self):
        self.__dict__ = {"i": self.i, "sparse": self.sparse}


class KVState:
    """Per-layer growing caches (batch=1): k/v [1,KV,S,D], idx_k [1,1,S,IDX_DIM]."""

    def __init__(self):
        self.k = [None] * LAYERS
        self.v = [None] * LAYERS
        self.idx_k = [None] * LAYERS

    def append(self, i, k_new, v_new):
        self.k[i] = k_new if self.k[i] is None else torch.cat([self.k[i], k_new], dim=2)
        self.v[i] = v_new if self.v[i] is None else torch.cat([self.v[i], v_new], dim=2)
        return self.k[i], self.v[i]

    def append_idx(self, i, idx_k_new):
        self.idx_k[i] = (
            idx_k_new if self.idx_k[i] is None else torch.cat([self.idx_k[i], idx_k_new], dim=2)
        )
        return self.idx_k[i]


def indexer_select(w: LayerWeights, x, cos, sin, position_ids, kv: KVState):
    """Lightning indexer -> [S_q, IDX_TOPK] block indices (-1 padded)."""
    S = x.shape[1]
    idx_q = F.linear(x, w.idx_q_proj).view(1, S, IDX_HEADS, IDX_DIM)
    idx_q = rms_norm(idx_q, w.idx_q_norm).transpose(1, 2)  # [1,H,S,D]
    idx_k = F.linear(x, w.idx_k_proj).view(1, S, 1, IDX_DIM)
    idx_k = rms_norm(idx_k, w.idx_k_norm).transpose(1, 2)  # [1,1,S,D]
    idx_q, idx_k = apply_rope(idx_q, idx_k, cos[None, None], sin[None, None])
    idx_k = kv.append_idx(w.i, idx_k)

    k_len = idx_k.shape[2]
    n_blocks = -(-k_len // IDX_BLOCK)
    pad = n_blocks * IDX_BLOCK - k_len

    scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))  # [1,H,S,K]
    k_pos = torch.arange(k_len, device=x.device)
    future = k_pos[None, None, None, :] > position_ids[None, None, :, None]
    scores = scores.masked_fill(future, float("-inf"))
    if pad:
        scores = F.pad(scores, (0, pad), value=float("-inf"))
    scores = scores.view(1, IDX_HEADS, S, n_blocks, IDX_BLOCK)
    block_scores = scores.amax(dim=-1).amax(dim=1)  # [1,S,n_blocks]

    q_block = position_ids // IDX_BLOCK  # [S]
    local = torch.arange(IDX_LOCAL, device=x.device)
    local_idx = (q_block[None, :, None] - local.view(1, 1, -1)).clamp(min=0)
    block_scores.scatter_(-1, local_idx, float("inf"))

    topk = min(IDX_TOPK, n_blocks)
    topk_scores, topk_idx = block_scores.topk(topk, dim=-1)
    return topk_idx.masked_fill(topk_scores == float("-inf"), -1)[0]  # [S, topk]


def block_mask_from_indices(block_idx, k_len, position_ids, device):
    """[S,topk] block indices -> additive float mask [1,1,S,k_len] (0 keep / -inf drop)."""
    S, topk = block_idx.shape
    n_blocks = -(-k_len // IDX_BLOCK)
    safe = block_idx.masked_fill(block_idx < 0, n_blocks)
    keep_blocks = torch.zeros((S, n_blocks + 1), dtype=torch.bool, device=device)
    keep_blocks.scatter_(-1, safe, True)
    keep = keep_blocks[:, :n_blocks].repeat_interleave(IDX_BLOCK, dim=-1)[:, :k_len]
    k_pos = torch.arange(k_len, device=device)
    keep &= k_pos[None, :] <= position_ids[:, None]
    mask = torch.zeros((1, 1, S, k_len), dtype=torch.float32, device=device)
    return mask.masked_fill(~keep[None, None], float("-inf"))


def layer_forward(w: LayerWeights, x, cos, sin, position_ids, kv: KVState, trace):
    """x: [1,S,H] bf16. Returns new hidden states."""
    S = x.shape[1]
    res = x
    h = rms_norm(x, w.input_ln)

    # ---- attention ----
    q = F.linear(h, w.q_proj).view(1, S, Q_HEADS, HEAD_DIM)
    k = F.linear(h, w.k_proj).view(1, S, KV_HEADS, HEAD_DIM)
    v = F.linear(h, w.v_proj).view(1, S, KV_HEADS, HEAD_DIM)
    q = rms_norm(q, w.q_norm).transpose(1, 2)
    k = rms_norm(k, w.k_norm).transpose(1, 2)
    v = v.transpose(1, 2)
    q, k = apply_rope(q, k, cos[None, None], sin[None, None])
    k, v = kv.append(w.i, k, v)
    k_len = k.shape[2]

    if w.sparse:
        block_idx = indexer_select(w, h, cos, sin, position_ids, kv)
        if trace is not None:
            trace["blocks"][w.i].append(block_idx.cpu())
        mask = block_mask_from_indices(block_idx, k_len, position_ids, x.device)
    else:
        k_pos = torch.arange(k_len, device=x.device)
        causal = k_pos[None, :] <= position_ids[:, None]
        mask = torch.zeros((1, 1, S, k_len), dtype=torch.float32, device=x.device)
        mask = mask.masked_fill(~causal[None, None], float("-inf"))

    attn = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask.to(q.dtype), enable_gqa=True, scale=HEAD_DIM**-0.5
    )
    attn = attn.transpose(1, 2).reshape(1, S, -1)
    x = res + F.linear(attn, w.o_proj)

    # ---- mlp ----
    res = x
    h = rms_norm(x, w.post_ln)
    if not w.sparse:
        out = F.linear(swigluoai(F.linear(h, w.mlp_gate), F.linear(h, w.mlp_up)), w.mlp_down)
    else:
        flat = h.view(-1, HIDDEN)
        shared = F.linear(
            swigluoai(F.linear(flat, w.sh_gate), F.linear(flat, w.sh_up)), w.sh_down
        )
        logits = F.linear(flat.to(w.gate_w.dtype), w.gate_w)
        scores = torch.sigmoid(logits.float())
        sel_scores, sel = torch.topk(scores + w.gate_bias.float(), TOP_K, dim=-1, sorted=False)
        weights = scores.gather(1, sel)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        if trace is not None:
            trace["experts"][w.i].append(sel.cpu())
            trace["expert_weights"][w.i].append(weights.cpu())
        routed = torch.zeros_like(flat)
        for e in sel.unique():
            tok, pos = torch.where(sel == e)
            cur = swigluoai(
                F.linear(flat[tok], w.w1[e]), F.linear(flat[tok], w.w3[e])
            )
            cur = F.linear(cur, w.w2[e]) * weights[tok, pos, None].to(flat.dtype)
            routed.index_add_(0, tok, cur)
        out = (routed * ROUTED_SCALING + shared).view(1, S, HIDDEN)
    return res + out


def full_forward(rd, kv, ids, position_ids, embed_w, norm_w, head_w, trace, tag):
    """One full pass over all layers for tokens `ids` at `position_ids`."""
    device = ids.device
    x = F.embedding(ids, embed_w)[None]  # [1,S,H]
    cos, sin = rope_cos_sin(position_ids, device, x.dtype)
    layer_times = []
    for i in range(LAYERS):
        t0 = time.time()
        w = LayerWeights(rd, i)
        t_load = time.time() - t0
        x = layer_forward(w, x, cos, sin, position_ids, kv, trace)
        if trace is not None:
            trace["hidden_last"][i].append(x[0, -1].float().cpu())
        w.free()
        torch.cuda.synchronize()
        layer_times.append((t_load, time.time() - t0 - t_load))
        if i % 8 == 0:
            torch.cuda.empty_cache()
            print(f"    [{tag}] layer {i}: load {t_load:.1f}s "
                  f"compute {layer_times[-1][1]:.1f}s", flush=True)
    x = rms_norm(x, norm_w)
    logits = F.linear(x.float(), head_w.float())  # fp32 head for a stable reference
    tl = sum(t for t, _ in layer_times)
    tc = sum(c for _, c in layer_times)
    print(f"  [{tag}] load {tl:.1f}s compute {tc:.1f}s", flush=True)
    return logits[0]  # [S, V]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/MiniMax-M3")
    ap.add_argument("--prompt", default=None, help="raw text; default: built-in 128-tok prompt")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--out", default="/workspace/FlashRT/minimax_dev/ref_out")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device
    torch.cuda.set_device(device)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    prompt = args.prompt or (
        "You are a senior systems engineer. Explain, step by step, how a CUDA kernel "
        "achieves high memory bandwidth on a GPU with unified memory, and write a short "
        "C function that sums an array of 1024 floats using shared memory reduction. "
        "Then briefly compare this approach with using cub::DeviceReduce, and state one "
        "case where the library version would be slower than the hand-written kernel."
    )
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    print(f"prompt tokens: {ids.shape[0]}", flush=True)

    rd = ShardReader(args.model, device)
    embed_w = rd.get(PFX + "embed_tokens.weight")
    norm_w = rd.get(PFX + "norm.weight")
    head_w = rd.get("language_model.lm_head.weight")

    trace = {
        "experts": defaultdict(list),
        "expert_weights": defaultdict(list),
        "blocks": defaultdict(list),
        "hidden_last": defaultdict(list),
    }
    kv = KVState()

    t0 = time.time()
    pos = torch.arange(ids.shape[0], device=device)
    logits = full_forward(rd, kv, ids, pos, embed_w, norm_w, head_w, trace, "prefill")
    torch.save(logits.cpu(), os.path.join(args.out, "prefill_logits.pt"))
    next_id = int(logits[-1].argmax())
    gen = [next_id]
    print(f"prefill done in {time.time() - t0:.1f}s; first token: {next_id!r} "
          f"{tok.decode([next_id])!r}", flush=True)

    for step in range(args.max_new - 1):
        t0 = time.time()
        cur = torch.tensor([gen[-1]], device=device)
        pos = torch.tensor([ids.shape[0] + step], device=device)
        logits = full_forward(rd, kv, cur, pos, embed_w, norm_w, head_w, trace, f"step{step}")
        next_id = int(logits[-1].argmax())
        gen.append(next_id)
        print(f"step {step}: {time.time() - t0:.1f}s -> {tok.decode([next_id])!r}", flush=True)

    text = tok.decode(gen)
    print("\n=== GENERATED ===\n" + text, flush=True)

    torch.save(
        {
            "prompt_ids": ids.cpu(),
            "generated_ids": gen,
            "experts": {i: torch.cat(v) for i, v in trace["experts"].items()},
            "expert_weights": {i: torch.cat(v) for i, v in trace["expert_weights"].items()},
            "blocks": {i: [b for b in v] for i, v in trace["blocks"].items()},
            "hidden_last": {i: torch.stack(v) for i, v in trace["hidden_last"].items()},
        },
        os.path.join(args.out, "trace.pt"),
    )

    # routing concentration summary (the go/no-go number for expert streaming)
    all_sel = torch.cat([trace["experts"][i][0] for i in sorted(trace["experts"])])
    print(f"\nrouting trace saved; tokens x layers selections: {all_sel.shape}")
    per_layer = {}
    for i in sorted(trace["experts"]):
        sel = torch.cat(trace["experts"][i])  # [T,4]
        counts = torch.bincount(sel.flatten(), minlength=N_EXPERTS).float()
        order = counts.sort(descending=True).values
        cum = order.cumsum(0) / order.sum()
        n80 = int((cum < 0.80).sum()) + 1
        per_layer[i] = n80
    n80s = torch.tensor(list(per_layer.values()), dtype=torch.float32)
    print(f"experts needed for 80% coverage per layer: "
          f"min {int(n80s.min())} / median {int(n80s.median())} / max {int(n80s.max())} (of {N_EXPERTS})")
    print(f"total NVMe bytes read: {rd.bytes_read / 1e9:.0f} GB")


if __name__ == "__main__":
    main()
