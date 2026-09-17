"""SigLIP reference dump for the pi0.5 Thor FP4 vision stage.

Builds the production FP4 pipeline (FA4, NVFP4 SigLIP FFN, rowops v2, AWQ
calibrated on real observations), then for one observation:
  - runs patch embedding, a per-layer Python reproduction of
    siglip_forward_with_fp4_ffn, and the post-LayerNorm projection eagerly,
  - checks the per-layer reproduction against the library SigLIP forward and
    the eager result against the captured uint8 SigLIP CUDA graph,
and writes the images, every weight the stage consumes, each layer's output
and the projected image tokens to a safetensors file.
"""
import argparse
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi05_pipeline as m1  # noqa: E402
from flash_rt.hardware.thor import shared_primitives_fp4 as sp4  # noqa: E402

fvk, fvk_fp4 = m1.fvk, m1.fvk_fp4
check = m1.check


def ref_layer(pipe, l, stream=0):
    """One SigLIP layer, production branch only (rowops v2, FA4, NVFP4 FFN)."""
    bufs, weights, dims = pipe._sig_bufs, pipe._sig_weights, pipe._sig_dims
    sc = pipe._sig_fp4_scratch
    w = pipe._sig_fp4_weights[l]
    S, D, NH, HD = dims["S"], dims["D"], dims["NH"], dims["HD"]
    spv = dims["seq_per_view"]
    H_pad = sc["H_pad"]
    x, x_fp8, qkv, attn_out = bufs["x"], bufs["x_fp8"], bufs["qkv"], bufs["attn_out"]
    alpha = weights["alpha"]
    assert sc["rowops_v2"] and sp4._rowops_flag("LN8", True) and sp4._rowops_flag("LN4", True)
    assert sc["siglip_up_variant"] != 0 and sc["siglip_down_variant"] == 0
    check(fvk_fp4.rowops_layer_norm_fp8_v2(x, weights["ln_attn_w"][l], weights["ln_attn_b"][l],
                                           x_fp8, S, D, 1e-5, stream), "ln_fp8")
    pipe._gemm.fp8_nn_bias(x_fp8, weights["qkv_w"][l], qkv, weights["qkv_b"][l], S, 3 * D, D,
                           alpha[l * 4 + 0], stream)
    pipe._attn.run("siglip", 0, q_seq=spv, stream=stream)
    fvk.quantize_fp8_static_fp16(attn_out, x_fp8, weights["unit_scale"], S * D, stream)
    pipe._gemm.fp8_nn_bias_res(x_fp8, weights["o_w"][l], x, weights["o_b"][l], S, D, D,
                               alpha[l * 4 + 1], stream)
    ln, hid = sc["ln_act"], sc["hid_act"]
    check(fvk_fp4.rowops_layer_norm_mul_fp4_sfa_v2(
        x, weights["ln_ffn_w"][l], weights["ln_ffn_b"][l], w["ln_inv_s"],
        ln.packed.data_ptr(), ln.sfa.data_ptr(), S, D, 1e-5, stream), "ln_fp4")
    check(fvk_fp4.cutlass_fp4_gemm_bias_gelu_fp4out_v(
        sc["siglip_up_variant"], ln.packed.data_ptr(), ln.sfa.data_ptr(),
        w["up"]["packed"].data_ptr(), w["up"]["sfb"].data_ptr(), w["up_bias"],
        hid.packed.data_ptr(), hid.sfa.data_ptr(), S, H_pad, D, stream), "up")
    check(fvk_fp4.cutlass_fp4_gemm_bias_res_fp16(
        hid.packed.data_ptr(), hid.sfa.data_ptr(),
        w["down"]["packed"].data_ptr(), w["down"]["sfb"].data_ptr(),
        weights["down_b"][l], x, x, S, D, H_pad, stream), "down")


def by_ptr(tensors, ptr, what):
    for t in tensors:
        if t.data_ptr() == ptr:
            return t
    raise RuntimeError(f"{what}: no tensor at {ptr:#x}")


def main():
    p = argparse.ArgumentParser()
    m1.add_policy_args(p)
    p.add_argument("--obs-index", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pipe, obs = m1.build_pipeline(args)

    dims, weights, bufs = pipe._sig_dims, pipe._sig_weights, pipe._sig_bufs
    S, D, H, NH, HD, L = (dims[n] for n in ("S", "D", "H", "NH", "HD", "L"))
    nv, spv = dims["num_views"], dims["seq_per_view"]
    sc = pipe._sig_fp4_scratch
    H_pad, De = sc["H_pad"], pipe.De
    assert pipe._attn._use_fa4, "reference must use FA4"
    assert int(pipe._attn._slots["siglip"]["O"]) == bufs["attn_out"]
    assert int(pipe._attn._slots["siglip"]["qkv"]) == bufs["qkv"]

    o = obs[args.obs_index]
    m1.upload_views(pipe, o, args.num_views)
    torch.cuda.synchronize()

    # captured production graph
    pipe._siglip_u8_graph.replay()
    torch.cuda.synchronize()
    tokens_graph = pipe._enc_x[:S].clone()

    # eager library forward
    for s in (sc["ln_act"], sc["hid_act"]):
        s.sfa.zero_()
    pipe._patch_embed_ops(0, uint8_input=True)
    torch.cuda.synchronize()
    x_embed = pipe._sig_x.clone()
    sp4.siglip_forward_with_fp4_ffn(pipe._gemm, fvk, fvk_fp4, bufs, weights, dims, stream=0,
                                    attn=pipe._attn, fp4_weights=pipe._sig_fp4_weights,
                                    fp4_scratch=sc)
    torch.cuda.synchronize()
    x_lib = pipe._sig_x.clone()
    pipe._postln_project_ops(0)
    torch.cuda.synchronize()
    tokens_eager = pipe._enc_x[:S].clone()

    # per-layer reproduction
    pipe._sig_x.copy_(x_embed)
    layer_out = []
    for l in range(L):
        ref_layer(pipe, l)
        torch.cuda.synchronize()
        layer_out.append(pipe._sig_x.clone())
    ok_layers = torch.equal(layer_out[-1], x_lib)
    ok_graph = torch.equal(tokens_eager, tokens_graph)
    print(f"per-layer reference vs library: {ok_layers} | eager tokens vs captured graph: {ok_graph}")
    if not (ok_layers and ok_graph):
        sys.exit(1)

    out = {
        "images_u8": torch.from_numpy(pipe._infer_images_u8_np.copy()),
        "lut": torch.from_numpy(pipe._img_u8_lut.download_new((256,), np.float16)),
        "pe_w": torch.from_numpy(pipe._pe_w.download_new((588, D), np.float16)),
        "pe_b": torch.from_numpy(pipe._pe_b.download_new((D,), np.float16)),
        "pos_emb": torch.from_numpy(pipe._pos_emb.download_new((spv, D), np.float16)),
        "postln_w": pipe._postln_w, "postln_b": pipe._postln_b,
        "proj_w": pipe._proj_w, "proj_b": pipe._proj_b,
        "x_embed": x_embed, "x_sig": x_lib, "tokens": tokens_eager,
        "alpha": torch.tensor(list(weights["alpha"]), dtype=torch.float32),
    }
    for l in range(L):
        w = pipe._sig_fp4_weights[l]
        out[f"L{l}.x_out"] = layer_out[l]
        out[f"L{l}.ln_attn_w"] = pipe._sig_ln_attn_w[l]
        out[f"L{l}.ln_attn_b"] = pipe._sig_ln_attn_b[l]
        out[f"L{l}.qkv_w"] = pipe._sig_qkv_w[l]
        out[f"L{l}.qkv_b"] = pipe._sig_qkv_b[l]
        out[f"L{l}.o_w"] = pipe._sig_o_w[l]
        out[f"L{l}.o_b"] = pipe._sig_o_b[l]
        out[f"L{l}.ln_ffn_w"] = pipe._sig_ln_ffn_w[l]
        out[f"L{l}.ln_ffn_b"] = pipe._sig_ln_ffn_b[l]
        out[f"L{l}.awq_inv_s"] = by_ptr(pipe._sig_awq_inv_s.values(), w["ln_inv_s"], "ln_inv_s")
        out[f"L{l}.up_packed"] = w["up"]["packed"]
        out[f"L{l}.up_sfb"] = w["up"]["sfb"]
        out[f"L{l}.up_b"] = by_ptr(pipe._sig_fp4_up_bias, w["up_bias"], "up_bias")
        out[f"L{l}.down_packed"] = w["down"]["packed"]
        out[f"L{l}.down_sfb"] = w["down"]["sfb"]
        out[f"L{l}.down_b"] = pipe._sig_down_b[l]
    out["meta"] = torch.tensor([S, D, H, NH, HD, L, nv, spv, H_pad, De,
                                sc["siglip_up_variant"], sc["siglip_down_variant"]], dtype=torch.int64)
    for n in ("L0.qkv_w", "L0.o_w", "L0.qkv_b", "L0.up_b", "L0.awq_inv_s", "pe_w", "proj_w"):
        print(n, tuple(out[n].shape), out[n].dtype)
    out = {n: t.detach().contiguous().cpu() for n, t in out.items()}
    save_file(out, args.out)
    print("wrote", args.out, "tensors", len(out))


if __name__ == "__main__":
    main()
