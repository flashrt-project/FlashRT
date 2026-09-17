"""dump the whole pi0.5 Thor FP4 encoder (all layers) for the stage test.

Reuses the M1 pipeline setup and per-layer reference. Writes the encoder
input, every layer's packed weights and scales, the final residual stream and
every layer's K/V rows, after checking the per-layer reference against the
library encoder forward.
"""
import argparse
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi05_pipeline as m1  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    m1.add_policy_args(p)
    p.add_argument("--obs-index", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    calls = []
    orig = m1.fe_mod.encoder_forward_with_fp4_subset

    def recorder(*a, **k):
        if k.get("stream", 0) == 0:
            calls.append((a, k))
        return orig(*a, **k)

    m1.fe_mod.encoder_forward_with_fp4_subset = recorder
    pipe, obs = m1.build_pipeline(args)
    a, k = calls[-1]
    dims = a[5]
    Se, D, L, total_keys, HD = dims["Se"], dims["D"], dims["L"], dims["total_keys"], dims["HD"]

    o = obs[args.obs_index]
    m1.upload_views(pipe, o, args.num_views)
    pipe._siglip_u8_graph.replay()
    torch.cuda.synchronize()
    x_in = pipe._enc_x[:Se].clone()

    pipe._Kc.zero_(); pipe._Vc.zero_()
    orig(*a, **k)
    torch.cuda.synchronize()
    x_lib = pipe._enc_x[:Se].clone()
    kc_flat = pipe._Kc.reshape(-1).clone(); vc_flat = pipe._Vc.reshape(-1).clone()

    pipe._enc_x[:Se].copy_(x_in); pipe._Kc.zero_(); pipe._Vc.zero_()
    for l in range(L):
        m1.ref_layer(l, a, k)
    torch.cuda.synchronize()
    ok = torch.equal(pipe._enc_x[:Se], x_lib) and torch.equal(pipe._Kc.reshape(-1), kc_flat) \
        and torch.equal(pipe._Vc.reshape(-1), vc_flat)
    print("per-layer reference vs library (all layers):", ok)
    if not ok:
        sys.exit(1)

    sc = k["fp4_scratch"]
    scales = pipe._enc_calib_scales.reshape(-1)
    out = {"x_in": x_in, "x_out": x_lib, "rope": pipe._enc_rope[:Se].contiguous()}
    for l in range(L):
        off = l * total_keys * HD
        out[f"L{l}.k_out"] = kc_flat[off:off + Se * HD].clone()
        out[f"L{l}.v_out"] = vc_flat[off:off + Se * HD].clone()
        out[f"L{l}.qkv_w"] = pipe._enc_qkv_w[l].contiguous()
        out[f"L{l}.act_scale_qkv"] = scales[l * 4:l * 4 + 1].clone()
        out[f"L{l}.alpha_qkv"] = torch.tensor([a[4]["alpha_host"][l * 4]], dtype=torch.float32)
        if l == L - 1:
            continue
        aw = k["fp4_attn_weights"][l]
        fw = k["fp4_weights"][l]
        out[f"L{l}.o_packed"] = aw["o"]["packed"]
        out[f"L{l}.o_sfb"] = aw["o"]["sfb"]
        out[f"L{l}.awq_inv_s_gu"] = pipe._awq_inv_s_gu[l]
        out[f"L{l}.gu_il_packed"] = fw["gu_il"]["packed"]
        out[f"L{l}.gu_il_sfb"] = fw["gu_il"]["sfb"]
        out[f"L{l}.down_packed"] = fw["down"]["packed"]
        out[f"L{l}.down_sfb"] = fw["down"]["sfb"]
    out["meta"] = torch.tensor([Se, D, dims["H"], dims["NH"], HD, total_keys,
                                sc["attn_variant"], sc["variant_dn"], L], dtype=torch.int64)
    out = {n: t.detach().contiguous().cpu() for n, t in out.items()}
    save_file(out, args.out)
    print("wrote", args.out, "tensors", len(out))


if __name__ == "__main__":
    main()
