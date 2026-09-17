"""M1 step 1: a single-layer reference for the pi0.5 Thor FP4 encoder.

Builds the production FP4 pipeline, reproduces one encoder layer with the
same kernel sequence the library uses, checks that running the per-layer
reference for all layers is bitwise identical to the library's encoder
forward, and writes one layer's inputs, packed weights, scales and outputs
to a safetensors file for the native implementation to match.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

import flash_rt.flash_rt_fp4 as fvk_fp4
import flash_rt.flash_rt_kernels as fvk
import flash_rt.frontends.torch.pi05_thor_fp4 as fe_mod
from flash_rt import load_model

PROMPT_TOKENS = [2, 18075, 908, 573, 3118, 3963, 578, 2040, 665, 575, 573, 24655, 108]

VIEW_KEYS = ("image", "wrist_image", "wrist_image_right")


def add_policy_args(p):
    p.add_argument("--checkpoint", default=os.environ.get("PI05_CHECKPOINT"))
    p.add_argument("--fixture", default=os.environ.get("PI05_FIXTURE"),
                   help="calibration observations (.npz from tools/make_libero_fixture.py)")
    p.add_argument("--prompt-len", type=int, default=208,
                   help="pad the prompt with token 0 to this length; 0 keeps the exact prompt")
    p.add_argument("--num-views", type=int, default=3)


def build_pipeline(args):
    """Production FP4 pipeline, calibrated on the fixture, after one inference."""
    if not args.checkpoint or not args.fixture:
        raise SystemExit("--checkpoint and --fixture are required (or PI05_CHECKPOINT / PI05_FIXTURE)")
    pipe = load_model(checkpoint=args.checkpoint, framework="torch", config="pi05",
                      hardware="thor", num_views=args.num_views, autotune=3, use_fa4=True,
                      use_fp4=True, use_fp4_decoder=True).pipeline
    data = np.load(args.fixture)
    obs = [{"image": data[f"img_{i}"], "state": data[f"state_{i}"],
            "wrist_image": data[f"wrist_{i}"], "wrist_image_right": data[f"wrist_right_{i}"]}
           for i in range(int(data["n"]))]
    tokens = PROMPT_TOKENS + [0] * max(0, args.prompt_len - len(PROMPT_TOKENS))
    pipe.set_prompt(tokens)
    pipe.calibrate(obs, percentile=99.9, verbose=False)
    pipe.infer(obs[0])
    torch.cuda.synchronize()
    return pipe, obs


def upload_views(pipe, o, num_views):
    for i, name in enumerate(VIEW_KEYS[:num_views]):
        np.copyto(pipe._infer_images_u8_np[i], o[name])
    pipe._img_u8_buf.upload(pipe._infer_images_u8_np)


def rowops_flag(name, default=True):
    value = os.environ.get(f"FLASHRT_ROWOPS_{name}")
    return default if value is None else value != "0"


def check(rc, what):
    if rc not in (None, 0):
        raise RuntimeError(f"{what} rc={rc}")


def ref_layer(l, a, k, stream=0):
    """One encoder layer, production branch only (rowops v2, residual
    epilogue, P1 epilogue_hw_nod, FP4 attention O, FP8 QKV)."""
    gemm, fvk_, fvk_fp4_, bufs, weights, dims = a
    attn = k["attn"]
    fp4_weights = k["fp4_weights"]
    sc = k["fp4_scratch"]
    attn_fp4 = k["fp4_attn_weights"]
    Se, D, H, NH, HD, L = (dims[n] for n in ("Se", "D", "H", "NH", "HD", "L"))
    total_keys = dims["total_keys"]
    last = l == L - 1
    x, x_fp8, qkv, attn_out = bufs["x"], bufs["x_fp8"], bufs["qkv"], bufs["attn_out"]
    act_scales = weights["act_scales"]
    alpha_host = weights["alpha_host"]
    as_qkv = act_scales + (l * 4 + 0) * 4

    assert sc["rowops_v2"] and rowops_flag("RMS")
    check(fvk_fp4.rowops_rms_fp8_v2(x, x_fp8, Se, D, as_qkv, stream), "rms_fp8")
    fvk.cutlass_fp8_sq(x_fp8, weights["qkv_w"][l], qkv, Se, 2560, D,
                       alpha_host[l * 4 + 0], 0.0, stream)
    check(fvk.qkv_split_rope_kvcache_fp16_vec(
        qkv, weights["rope"], attn_out, weights["Kc"], weights["Vc"],
        Se, NH * HD, HD, HD, 2560, l * total_keys * HD, HD, stream), "qkv_split")
    if last:
        return
    attn.run("encoder", l, q_seq=Se, stream=stream)

    aw = attn_fp4[l]
    assert "o" in aw and "qkv" not in aw
    sc_at = sc["attn_act"]
    check(fvk_fp4.rowops_quantize_fp4_sfa_v2(
        attn_out, sc_at.packed.data_ptr(), sc_at.sfa.data_ptr(), Se, D, stream), "quant_o")
    check(fvk_fp4.cutlass_fp4_gemm_variant(
        sc["attn_variant"], sc_at.packed.data_ptr(), sc_at.sfa.data_ptr(),
        aw["o"]["packed"].data_ptr(), aw["o"]["sfb"].data_ptr(),
        x, Se, D, D, 1.0, 1.0, stream), "o_gemm")

    assert sc["res_epilogue"] and sc["p1_combiner"] == "epilogue_hw_nod"
    sc_gu, sc_dn = sc["gu_act"], sc["down_act"]
    awq_gu = sc["awq_inv_s_gu"][l]
    check(fvk_fp4.rowops_rms_mul_fp4_sfa_v2(
        x, awq_gu, sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(), Se, D, stream), "rms_mul_fp4")
    w_il = fp4_weights[l]["gu_il"]
    check(fvk_fp4.cutlass_fp4_gemm_geglu_il_hw_nod(
        sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
        w_il["packed"].data_ptr(), w_il["sfb"].data_ptr(), sc["p1_dummy"],
        sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(), Se, 2 * H, D, stream), "geglu")
    w_dn = fp4_weights[l]["down"]
    check(fvk_fp4.cutlass_fp4_gemm_variant(
        sc["variant_dn"], sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
        w_dn["packed"].data_ptr(), w_dn["sfb"].data_ptr(),
        x, Se, D, H, 1.0, 1.0, stream), "down_gemm")
    # The library also writes next-layer FP8 input here; the next layer
    # recomputes the identical value at its start, so it is omitted.


def main():
    p = argparse.ArgumentParser()
    add_policy_args(p)
    p.add_argument("--dump-layer", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    calls = []
    orig = fe_mod.encoder_forward_with_fp4_subset

    def recorder(*a, **k):
        if k.get("stream", 0) == 0:
            calls.append((a, k))
        return orig(*a, **k)

    fe_mod.encoder_forward_with_fp4_subset = recorder

    pipe, obs = build_pipeline(args)
    a, k = calls[-1]
    dims = a[5]
    Se, D, L, total_keys, HD = dims["Se"], dims["D"], dims["L"], dims["total_keys"], dims["HD"]
    print("Kc", tuple(pipe._Kc.shape), pipe._Kc.dtype, "scales", pipe._enc_calib_scales.dtype, tuple(pipe._enc_calib_scales.shape))
    print("dims", dims, "fp4 layers", sorted(k["fp4_layers"]), "use_p1", k["use_p1_split_gu"])

    # Encoder input for observation 0: upload images and replay SigLIP only.
    upload_views(pipe, obs[0], args.num_views)
    pipe._siglip_u8_graph.replay()
    torch.cuda.synchronize()
    x_in = pipe._enc_x[:Se].clone()

    # Library encoder forward, eager, stream 0.
    pipe._Kc.zero_(); pipe._Vc.zero_()
    orig(*a, **k)
    torch.cuda.synchronize()
    x_lib = pipe._enc_x[:Se].clone(); kc_lib = pipe._Kc.clone(); vc_lib = pipe._Vc.clone()

    # Per-layer reference for all layers.
    pipe._enc_x[:Se].copy_(x_in); pipe._Kc.zero_(); pipe._Vc.zero_()
    dump = {}
    for l in range(L):
        if l == args.dump_layer:
            dump["x_in"] = pipe._enc_x[:Se].clone()
        ref_layer(l, a, k)
        torch.cuda.synchronize()
        if l == args.dump_layer:
            dump["x_out"] = pipe._enc_x[:Se].clone()
    x_ref = pipe._enc_x[:Se].clone()
    same_x = torch.equal(x_ref, x_lib)
    same_kv = torch.equal(pipe._Kc, kc_lib) and torch.equal(pipe._Vc, vc_lib)
    print("per-layer reference vs library: x bitwise", same_x, "| KV bitwise", same_kv)
    if not (same_x and same_kv):
        diff = (x_ref.float() - x_lib.float()).abs()
        print("max |dx|", diff.max().item())
        sys.exit(1)

    # Dump one layer: inputs, weights, scales, outputs.
    l = args.dump_layer
    sc = k["fp4_scratch"]
    aw = k["fp4_attn_weights"][l]
    fw = k["fp4_weights"][l]
    kv_off = l * total_keys * HD
    scales = pipe._enc_calib_scales.reshape(-1)
    dump.update({
        "qkv_w": pipe._enc_qkv_w[l].contiguous(),
        "rope": pipe._enc_rope[:Se].contiguous(),
        "o_packed": aw["o"]["packed"], "o_sfb": aw["o"]["sfb"],
        "awq_inv_s_gu": pipe._awq_inv_s_gu[l],
        "gu_il_packed": fw["gu_il"]["packed"], "gu_il_sfb": fw["gu_il"]["sfb"],
        "down_packed": fw["down"]["packed"], "down_sfb": fw["down"]["sfb"],
        "act_scale_qkv": scales[l * 4 + 0:l * 4 + 1].clone(),
        "alpha_qkv": torch.tensor([a[4]["alpha_host"][l * 4 + 0]], dtype=torch.float32),
        "k_out": pipe._Kc.reshape(-1)[kv_off:kv_off + Se * HD].clone(),
        "v_out": pipe._Vc.reshape(-1)[kv_off:kv_off + Se * HD].clone(),
        "meta": torch.tensor([Se, D, dims["H"], dims["NH"], HD, total_keys,
                              sc["attn_variant"], sc["variant_dn"]], dtype=torch.int64),
    })
    dump = {name: t.detach().contiguous().cpu() for name, t in dump.items()}
    save_file(dump, args.out)
    print("wrote", args.out, {n: (tuple(t.shape), str(t.dtype)) for n, t in dump.items()})


if __name__ == "__main__":
    main()
