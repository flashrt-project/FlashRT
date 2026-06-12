#!/usr/bin/env python3
"""MiniMax-M3 BF16 -> NVFP4 streaming quantizer + packed expert layout (P2).

Reads the original 854 GB BF16 shards layer by layer, quantizes with FlashRT's
`bf16_weight_to_nvfp4_swizzled` (packed u8 + CUTLASS Sm1xx swizzled SF +
per-tensor global scale), and writes:

  <out>/manifest.json              dims, block layout, file inventory
  <out>/resident_top.pt            embed (bf16) + final norm + lm_head (nvfp4)
  <out>/resident_layer_NN.pt       per-layer non-routed weights:
                                     attn q/k/v/o nvfp4, qk norms bf16,
                                     indexer projs/norms bf16, layer norms bf16,
                                     router gate + bias bf16/f32,
                                     shared/dense MLP nvfp4, expert alphas [128,3]
  <out>/experts_layer_NN.bin       (sparse layers only) 128 fixed-size blocks:
                                     w1_packed | w1_sf | w3_packed | w3_sf |
                                     w2_packed | w2_sf
                                   block = 31,850,496 B (4096-aligned by luck)

Quantized:   q/k/v/o_proj, dense MLP, shared experts, routed experts, lm_head.
Kept BF16:   embed_tokens, all norms, router gate + e_score_correction_bias,
             indexer q/k projs + norms (selection quality is the whole game).

Self-check per layer (--selfcheck): for one sampled quantized weight, compare
fp4_w4a16_gemm_sm120_bf16out(quant(x), W_nvfp4) against x @ W_bf16.T in BF16,
report cos. This validates quantization AND the sm_121 GEMM kernel in one shot.

Resumable: layers whose outputs already exist (right sizes) are skipped.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_st_reader import RawShardReader  # noqa: E402

HIDDEN = 6144
LAYERS = 60
INTER = 3072
DENSE_INTER = 12288
SHARED_INTER = 3072
N_EXPERTS = 128
Q_OUT = 64 * 128
KV_OUT = 4 * 128
VOCAB = 200064
DENSE_LAYERS = {0, 1, 2}
PFX = "language_model.model."

# fixed packed-expert block layout (bytes)
W1_PACKED = INTER * HIDDEN // 2          # 9_437_184
W3_PACKED = W1_PACKED
W2_PACKED = HIDDEN * INTER // 2          # 9_437_184
def _sf_bytes(n, k):
    return ((n + 127) // 128) * (((k // 16) + 3) // 4) * 512
W1_SF = _sf_bytes(INTER, HIDDEN)         # 1_179_648
W3_SF = W1_SF
W2_SF = _sf_bytes(HIDDEN, INTER)         # 1_179_648
BLOCK_BYTES = W1_PACKED + W3_PACKED + W2_PACKED + W1_SF + W3_SF + W2_SF
assert BLOCK_BYTES % 4096 == 0, BLOCK_BYTES


class Quantizer:
    def __init__(self, fvk, device):
        self.fvk = fvk
        self.device = device
        self.scratch_amax = torch.zeros(1, dtype=torch.float32, device=device)
        self.out_gs = torch.zeros(1, dtype=torch.float32, device=device)

    def quant(self, w_bf16: torch.Tensor):
        """(N, K) bf16 on GPU -> (packed u8 (N,K/2), sf_swz u8, alpha float)."""
        N, K = w_bf16.shape
        assert K % 16 == 0
        packed = torch.empty(N, K // 2, dtype=torch.uint8, device=self.device)
        sf = torch.zeros(_sf_bytes(N, K), dtype=torch.uint8, device=self.device)
        self.out_gs.zero_()
        self.fvk.bf16_weight_to_nvfp4_swizzled(
            int(w_bf16.data_ptr()), int(packed.data_ptr()), int(sf.data_ptr()),
            int(self.scratch_amax.data_ptr()), int(self.out_gs.data_ptr()),
            N, K, 0)
        torch.cuda.synchronize()
        return packed, sf, float(self.out_gs.item())


def selfcheck_gemm(fvk, qz, w_bf16, packed, sf, alpha, device):
    """cos( fp4_gemm(quant(x), W) , x @ W.T ) for random x, M=8."""
    N, K = w_bf16.shape
    M = 8
    x = (torch.randn(M, K, device=device) * 0.5).to(torch.bfloat16)
    a_packed = torch.empty(M, K // 2, dtype=torch.uint8, device=device)
    a_sf = torch.zeros(_sf_bytes(M, K), dtype=torch.uint8, device=device)
    fvk.quantize_bf16_to_nvfp4_swizzled(
        int(x.data_ptr()), int(a_packed.data_ptr()), int(a_sf.data_ptr()),
        M, K, 0)
    out = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
    fvk.fp4_w4a16_gemm_sm120_bf16out(
        int(a_packed.data_ptr()), int(packed.data_ptr()), int(out.data_ptr()),
        M, N, K, int(a_sf.data_ptr()), int(sf.data_ptr()), alpha, 0)
    torch.cuda.synchronize()
    ref = (x.float() @ w_bf16.float().T)
    cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0)
    return float(cos)


Reader = RawShardReader  # pread-based; safetensors mmap is ~0.3 GB/s on GB10


def do_layer(i, rd, qz, fvk, out_dir, device, selfcheck):
    p = f"{PFX}layers.{i}."
    sparse = i not in DENSE_LAYERS
    res_path = os.path.join(out_dir, f"resident_layer_{i:02d}.pt")
    bin_path = os.path.join(out_dir, f"experts_layer_{i:02d}.bin")
    want_bin = sparse
    if os.path.exists(res_path) and (
            not want_bin or (os.path.exists(bin_path) and
                             os.path.getsize(bin_path) == N_EXPERTS * BLOCK_BYTES)):
        print(f"layer {i}: exists, skip", flush=True)
        return

    t0 = time.time()
    res = {}

    def q_to(prefix, name):
        w = rd.get(p + name, device).to(torch.bfloat16).contiguous()
        packed, sf, alpha = qz.quant(w)
        res[prefix + "_packed"] = packed.cpu()
        res[prefix + "_sf"] = sf.cpu()
        res[prefix + "_alpha"] = alpha
        res[prefix + "_shape"] = tuple(w.shape)
        if selfcheck and prefix == "o_proj":
            res[prefix + "_selfcheck_cos"] = selfcheck_gemm(
                fvk, qz, w, packed, sf, alpha, device)
        del w
        return prefix

    def keep_bf16(prefix, name):
        res[prefix] = rd.get(p + name, "cpu").to(torch.bfloat16)

    # attention
    for pref, nm in [("q_proj", "self_attn.q_proj.weight"),
                     ("k_proj", "self_attn.k_proj.weight"),
                     ("v_proj", "self_attn.v_proj.weight"),
                     ("o_proj", "self_attn.o_proj.weight")]:
        q_to(pref, nm)
    for pref, nm in [("q_norm", "self_attn.q_norm.weight"),
                     ("k_norm", "self_attn.k_norm.weight"),
                     ("input_ln", "input_layernorm.weight"),
                     ("post_ln", "post_attention_layernorm.weight")]:
        keep_bf16(pref, nm)

    if not sparse:
        for pref, nm in [("mlp_gate", "mlp.gate_proj.weight"),
                         ("mlp_up", "mlp.up_proj.weight"),
                         ("mlp_down", "mlp.down_proj.weight")]:
            q_to(pref, nm)
    else:
        for pref, nm in [("idx_q_proj", "self_attn.index_q_proj.weight"),
                         ("idx_k_proj", "self_attn.index_k_proj.weight"),
                         ("idx_q_norm", "self_attn.index_q_norm.weight"),
                         ("idx_k_norm", "self_attn.index_k_norm.weight")]:
            keep_bf16(pref, nm)
        b = p + "block_sparse_moe."
        res["gate_w"] = rd.get(b + "gate.weight", "cpu").to(torch.bfloat16)
        res["gate_bias"] = rd.get(b + "e_score_correction_bias", "cpu").float()
        for pref, nm in [("sh_gate", "block_sparse_moe.shared_experts.gate_proj.weight"),
                         ("sh_up", "block_sparse_moe.shared_experts.up_proj.weight"),
                         ("sh_down", "block_sparse_moe.shared_experts.down_proj.weight")]:
            q_to(pref, nm)

        # routed experts -> packed bin (parallel pread whole layer to CPU)
        alphas = torch.zeros(N_EXPERTS, 3, dtype=torch.float32)
        names = [f"{b}experts.{e}.{wn}.weight"
                 for e in range(N_EXPERTS) for wn in ("w1", "w3", "w2")]
        cpu = rd.get_many(names, device="cpu")
        tmp_path = bin_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for e in range(N_EXPERTS):
                blk = bytearray()
                for j, wn in enumerate(["w1", "w3", "w2"]):
                    w = cpu[3 * e + j].to(device)
                    w = w.to(torch.bfloat16).contiguous()
                    packed, sf, alpha = qz.quant(w)
                    alphas[e, j] = alpha
                    blk += packed.cpu().numpy().tobytes()
                    blk += sf.cpu().numpy().tobytes()
                    del w, packed, sf
                # layout: packeds then sfs? NO — keep per-matrix (packed+sf)
                # adjacent: w1_packed|w1_sf|w3_packed|w3_sf|w2_packed|w2_sf
                assert len(blk) == BLOCK_BYTES, (len(blk), BLOCK_BYTES)
                f.write(blk)
        os.replace(tmp_path, bin_path)
        res["expert_alphas"] = alphas
        if selfcheck:
            # re-check one routed expert end-to-end from the written file
            with open(bin_path, "rb") as f:
                f.seek(0)
                blk = f.read(BLOCK_BYTES)
            off = 0
            w1p = torch.frombuffer(blk, dtype=torch.uint8, count=W1_PACKED,
                                   offset=off).clone().to(device); off += W1_PACKED
            w1s = torch.frombuffer(blk, dtype=torch.uint8, count=W1_SF,
                                   offset=off).clone().to(device); off += W1_SF
            w_ref = rd.get(f"{b}experts.0.w1.weight", device).to(torch.bfloat16)
            res["expert0_w1_selfcheck_cos"] = selfcheck_gemm(
                fvk, qz, w_ref.contiguous(),
                w1p.view(INTER, HIDDEN // 2), w1s, float(alphas[0, 0]), device)
            del w_ref

    torch.save(res, res_path + ".tmp")
    os.replace(res_path + ".tmp", res_path)
    rd.drop_all()  # keep streamed bytes out of the cgroup page cache
    checks = {k: v for k, v in res.items() if k.endswith("_cos")}
    print(f"layer {i}: done in {time.time() - t0:.1f}s {checks}", flush=True)
    del res
    torch.cuda.empty_cache()


def do_top(rd, qz, fvk, out_dir, device, selfcheck):
    path = os.path.join(out_dir, "resident_top.pt")
    if os.path.exists(path):
        print("top: exists, skip", flush=True)
        return
    res = {}
    res["embed_w"] = rd.get(PFX + "embed_tokens.weight", "cpu").to(torch.bfloat16)
    res["final_norm"] = rd.get(PFX + "norm.weight", "cpu").to(torch.bfloat16)
    w = rd.get("language_model.lm_head.weight", device).to(torch.bfloat16).contiguous()
    packed, sf, alpha = qz.quant(w)
    res["lm_head_packed"] = packed.cpu()
    res["lm_head_sf"] = sf.cpu()
    res["lm_head_alpha"] = alpha
    if selfcheck:
        res["lm_head_selfcheck_cos"] = selfcheck_gemm(fvk, qz, w, packed, sf,
                                                      alpha, device)
    del w
    torch.save(res, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"top: done {res.get('lm_head_selfcheck_cos', '')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/MiniMax-M3")
    ap.add_argument("--out", default="/models/MiniMax-M3-NVFP4")
    ap.add_argument("--layers", default="0:60", help="half-open range a:b")
    ap.add_argument("--selfcheck", action="store_true", default=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    sys.path.insert(0, "/workspace/FlashRT/flash_rt")
    import flash_rt_kernels as fvk  # built .so (lands in the package dir)

    os.makedirs(args.out, exist_ok=True)
    device = args.device
    rd = Reader(args.model)
    qz = Quantizer(fvk, device)

    a, bnd = (int(x) for x in args.layers.split(":"))
    manifest = {
        "format": "flashrt-m3-nvfp4-v1",
        "block_bytes": BLOCK_BYTES,
        "block_layout": ["w1_packed", "w1_sf", "w3_packed", "w3_sf",
                         "w2_packed", "w2_sf"],
        "sizes": {"w1_packed": W1_PACKED, "w1_sf": W1_SF,
                  "w3_packed": W3_PACKED, "w3_sf": W3_SF,
                  "w2_packed": W2_PACKED, "w2_sf": W2_SF},
        "n_experts": N_EXPERTS, "layers": LAYERS,
        "dense_layers": sorted(DENSE_LAYERS),
        "dims": {"hidden": HIDDEN, "inter": INTER,
                 "dense_inter": DENSE_INTER, "shared_inter": SHARED_INTER},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    do_top(rd, qz, fvk, args.out, device, args.selfcheck)
    for i in range(a, bnd):
        do_layer(i, rd, qz, fvk, args.out, device, args.selfcheck)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
