"""M3 step 1: a per-denoise-step reference for the pi0.5 Thor FP4 decoder.

Runs the library encoder for a real observation, then the library decoder
(10 steps) from a fixed noise, reproduces the decoder step by step with the
production kernel sequence, checks bitwise identity of the final actions and
the suffix K/V rows, and writes every tensor a native step needs.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi05_pipeline as m1  # noqa: E402

fvk = m1.fvk
fvk_fp4 = m1.fvk_fp4


def ref_step(s, a, k, stream=0):
    ctx, _fvk, _fvk_fp4, bufs, w, dims = a[:6]
    attn = k["attn"]
    S, D, H, NH, HD = (int(dims[n]) for n in ("S", "D", "H", "NH", "HD"))
    layers, enc_seq, total_keys = int(dims["layers"]), int(dims["enc_seq"]), int(dims["total_keys"])
    v_qkv, v_o, v_gu, v_down = (int(v) for v in dims["fp4_variants"])
    D3 = 3 * D
    x, xn, gate, qkv, attn_out, fg = (bufs[n] for n in ("x", "xn", "gate", "qkv", "attn_out", "fg"))
    xn_fp4, xn_sfa, ctx_fp4, ctx_sfa = bufs["xn_fp4"], bufs["xn_sfa"], bufs["ctx_fp4"], bufs["ctx_sfa"]
    hid_fp4, hid_sfa = bufs["hid_fp4"], bufs["hid_sfa"]
    noise = bufs["noise"]

    fvk.gmm_fp16(ctx, noise, w["ain_w"], x, S, D, 32, 0.0, stream)
    fvk.add_bias_fp16(x, w["ain_b"], S, D, stream)
    for l in range(layers):
        si = (s * layers + l) * S * D3
        sa_ptr = w["sa"] + si * 2
        sf_ptr = w["sf"] + si * 2
        if l == 0:
            fvk_fp4.pi05_adarms_fp4_sfa_native_fp16(x, sa_ptr, xn_fp4, xn_sfa, gate, S, D, stream)
        m1.check(fvk_fp4.cutlass_fp4_gemm_variant(v_qkv, xn_fp4, xn_sfa, w["qw_fp4"][l], w["qw_sfb"][l],
                                                  qkv, S, 2560, D, 1.0, 0.0, stream), "qkv")
        m1.check(fvk.qkv_split_rope_kvcache_fp16_vec(qkv, w["rope"], attn_out, w["Kc"], w["Vc"], S, NH * HD, HD,
                                                     HD, 2560, l * total_keys * HD + enc_seq * HD, HD, stream),
                 "qkv_split")
        attn.run("decoder", l, q_seq=S, kv_seq=total_keys, stream=stream)
        m1.check(fvk_fp4.rowops_quantize_fp4_sfa_v2(attn_out, ctx_fp4, ctx_sfa, S, NH * HD, stream), "quant")
        m1.check(fvk_fp4.cutlass_fp4_gemm_variant(v_o, ctx_fp4, ctx_sfa, w["ow_fp4"][l], w["ow_sfb"][l], fg,
                                                  S, D, NH * HD, 1.0, 0.0, stream), "o")
        fvk_fp4.pi05_gate_res_adarms_fp4_sfa_native_fp16(fg, gate, x, sf_ptr, xn_fp4, xn_sfa, gate, S, D, stream)
        m1.check(fvk_fp4.cutlass_fp4_gemm_geglu_il_hw_v10(xn_fp4, xn_sfa, w["gwil_fp4"][l], w["gwil_sfb"][l],
                                                          w["gu_dummy"], hid_fp4, hid_sfa, S, H * 2, D, stream),
                 "geglu")
        m1.check(fvk_fp4.cutlass_fp4_gemm_variant(v_down, hid_fp4, hid_sfa, w["dw_fp4"][l], w["dw_sfb"][l], fg,
                                                  S, D, H, 1.0, 0.0, stream), "down")
        if l < layers - 1:
            sa_next = w["sa"] + ((s * layers + l + 1) * S * D3) * 2
            fvk_fp4.pi05_gate_res_adarms_fp4_sfa_native_fp16(fg, gate, x, sa_next, xn_fp4, xn_sfa, gate, S, D,
                                                             stream)
        else:
            fvk.gate_res_fp16(fg, gate, x, S * D, stream)
    fs_ptr = w["fs"] + (s * S * D3) * 2
    fvk.adarms_fp16(x, fs_ptr, xn, gate, S, D, stream)
    fvk.gmm_fp16_out_fp32(ctx, xn, w["aow"], bufs["action_f32"], S, 32, D, stream)
    fvk.action_update_from_fp32(bufs["action_f32"], w["aob"], noise, S, 32, float(w["dt"]), True, stream)


def main():
    p = argparse.ArgumentParser()
    m1.add_policy_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    enc_calls, dec_calls = [], []
    enc_orig = m1.fe_mod.encoder_forward_with_fp4_subset
    dec_orig = m1.fe_mod.decoder_forward_fp4

    def enc_rec(*a, **k):
        if k.get("stream", 0) == 0:
            enc_calls.append((a, k))
        return enc_orig(*a, **k)

    def dec_rec(*a, **k):
        if k.get("stream", 0) == 0:
            dec_calls.append((a, k))
        return dec_orig(*a, **k)

    m1.fe_mod.encoder_forward_with_fp4_subset = enc_rec
    m1.fe_mod.decoder_forward_fp4 = dec_rec
    pipe, obs = m1.build_pipeline(args)
    ea, ek = enc_calls[-1]
    da, dk = dec_calls[-1]
    dims = da[5]
    w = da[4]
    print("decoder dims:", {n: dims[n] for n in sorted(dims) if not isinstance(dims[n], (list, tuple)) or n == "fp4_variants"})
    for flag in ("fixed_shape", "attn_splitkv", "attn_mqa", "dec_seq", "phase_cta", "fused_geglu_swap",
                 "fused_geglu_earlyb", "fused_geglu_nod"):
        assert not dims.get(flag), f"unexpected production flag {flag}={dims.get(flag)}"
    assert dims.get("fused_geglu") and dims.get("dec_rowops_quant", True)
    assert dims.get("weight_format", "nvfp4") == "nvfp4" and dims.get("act_format", "nvfp4") == "nvfp4"
    assert w.get("dt") is not None and da[3].get("action_f32")
    S, D, HD = int(dims["S"]), int(dims["D"]), int(dims["HD"])
    layers, steps = int(dims["layers"]), int(dims["steps"])
    enc_seq, total_keys = int(dims["enc_seq"]), int(dims["total_keys"])

    # Prefix K/V from the library encoder on observation 0.
    m1.upload_views(pipe, obs[0], args.num_views)
    pipe._siglip_u8_graph.replay()
    pipe._Kc.zero_(); pipe._Vc.zero_()
    enc_orig(*ea, **ek)
    torch.cuda.synchronize()
    kc0 = pipe._Kc.clone(); vc0 = pipe._Vc.clone()
    g = torch.Generator().manual_seed(args.seed)
    noise_in = torch.randn(S, 32, generator=g).half().cuda()

    # Library decoder.
    pipe._g_noise.view(S, 32).copy_(noise_in)
    dec_orig(*da, **dk)
    torch.cuda.synchronize()
    noise_lib = pipe._g_noise.view(S, 32).clone(); kc_lib = pipe._Kc.clone(); vc_lib = pipe._Vc.clone()

    # Step-by-step reference, recording every step's output noise.
    pipe._Kc.copy_(kc0); pipe._Vc.copy_(vc0)
    pipe._g_noise.view(S, 32).copy_(noise_in)
    per_step = []
    for s in range(steps):
        ref_step(s, da, dk)
        torch.cuda.synchronize()
        per_step.append(pipe._g_noise.view(S, 32).clone())
    same = (torch.equal(pipe._g_noise.view(S, 32), noise_lib) and torch.equal(pipe._Kc, kc_lib)
            and torch.equal(pipe._Vc, vc_lib))
    print("step reference vs library decoder: actions+KV bitwise", same)
    if not same:
        print("max |d noise|", (pipe._g_noise.view(S, 32).float() - noise_lib.float()).abs().max().item())
        sys.exit(1)

    out = {"noise_in": noise_in, "noise_out": noise_lib}
    for s in range(steps):
        out[f"step{s}.noise_out"] = per_step[s]
    kflat, vflat = kc_lib.reshape(-1), vc_lib.reshape(-1)
    for l in range(layers):
        off = l * total_keys * HD
        out[f"L{l}.k_prefix"] = kc0.reshape(-1)[off:off + enc_seq * HD].clone()
        out[f"L{l}.v_prefix"] = vc0.reshape(-1)[off:off + enc_seq * HD].clone()
        out[f"L{l}.k_suffix"] = kflat[off + enc_seq * HD:off + total_keys * HD].clone()
        out[f"L{l}.v_suffix"] = vflat[off + enc_seq * HD:off + total_keys * HD].clone()
    # Weights, per kind concatenated over layers (every layer has the same shape).
    fw = pipe._decoder_fp4_weights
    for kind, src in (("qw", "qkv"), ("ow", "o"), ("gwil", "gu_il"), ("dw", "down")):
        for part, suffix in (("packed", "fp4"), ("sfb", "sfb")):
            parts = [fw[l][src][part].contiguous().view(torch.uint8).reshape(-1) for l in range(layers)]
            assert all(t.numel() == parts[0].numel() for t in parts)
            out[f"{kind}_{suffix}"] = torch.cat(parts)
    for name, attr in (("ain_w", "_ain_w"), ("ain_b", "_ain_b"), ("aow", "_aow"), ("aob", "_aob"),
                       ("rope", "_dec_rope"), ("sa", "_sa_all"), ("sf", "_sf_all"), ("fs", "_fs_all")):
        t = getattr(pipe, attr)
        print(f"{name}: {attr} shape {tuple(t.shape)} {t.dtype}")
        out[name] = t.contiguous()
    gd = pipe._decoder_fp4_gu_dummy
    print("gu_dummy", tuple(gd.shape), gd.dtype)
    print("scratch sizes: xn sfa", pipe._decoder_fp4_xn.sfa.numel(), "ctx sfa", pipe._decoder_fp4_ctx.sfa.numel(),
          "hid sfa", pipe._decoder_fp4_hid.sfa.numel(), "logits", tuple(pipe._ae_logits.shape))
    out["meta"] = torch.tensor([S, D, int(dims["H"]), int(dims["NH"]), HD, layers, steps, enc_seq, total_keys]
                               + [int(v) for v in dims["fp4_variants"]] + [gd.numel()], dtype=torch.int64)
    out["dt"] = torch.tensor([float(w["dt"])], dtype=torch.float32)
    torch.save({"weights_keys": sorted(w.keys()), "bufs_keys": sorted(da[3].keys())},
               args.out + ".keys.pt")
    out = {n: t.detach().contiguous().cpu() for n, t in out.items()}
    save_file(out, args.out)
    print("wrote", args.out, "tensors", len(out))
    print("weights keys:", sorted(w.keys()))


if __name__ == "__main__":
    main()
