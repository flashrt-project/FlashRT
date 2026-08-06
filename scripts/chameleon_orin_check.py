#!/usr/bin/env python3
"""Gate-1 correctness harness for Chameleon-7B on Orin SM87.

Compares the FlashRT INT8/INT4 frontend against a **stock transformers 4.57.1**
``ChameleonForConditionalGeneration`` reference (bf16, eager attention) on the
*same* token ids, and runs the CUDA-Graph safety gate on the decode body.

Two non-obvious things this handles:

* **The checkpoint does not load into stock transformers as-is.** Its
  ``ChameleonLayerNorm`` builds ``(num_heads, head_dim) = (32,128)`` weights
  while this Lumina-mGPT export stores ``(1,128)`` — the shard is
  ``model_parallel_size`` x ``head_dim`` and upstream expands it with
  ``repeat_interleave`` at forward time. We expand it at load instead, which is
  exactly equivalent for the 7B (mp=1) layout. transformers 4.57 also rejects
  ``from_pretrained(..., state_dict=...)``, so the model is built with a naked
  constructor + ``load_state_dict``.

* **VQ-GAN index drift would poison every number.** FlashRT runs the encoder
  convs in fp16, so codebook indices can differ from an all-fp32 reference. The
  FlashRT side therefore *exports* the ids it computed and the reference is fed
  those verbatim, isolating LLM error from tokenizer error. Run with
  ``--vq-fp16-argmin`` to measure the drift itself instead.

Usage:
    PYTHONPATH=. python scripts/chameleon_orin_check.py \
        --checkpoint /path/to/Chameleon_7B_mGPT \
        --image FlashRT.png --prompt "Describe this image." --steps 16
    ... --int4          # QuaRot W4A4 tier
    ... --text-only     # skip the image (fast smoke)
"""

from __future__ import annotations

import argparse
import sys

import torch

PROBE_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]

# Chameleon suppresses the 8192 image-codebook ids at every forward, so they
# carry no information and must be excluded from any similarity metric —
# including them makes cosine NaN (finfo(bf16).min squared overflows fp32).
IMG_LO, IMG_HI = 4, 8196


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def text_slice(logits: torch.Tensor) -> torch.Tensor:
    """Drop the masked image-id band before comparing logits."""
    return torch.cat([logits[..., :IMG_LO], logits[..., IMG_HI:]], dim=-1)


# ======================================================================
# Reference
# ======================================================================

def load_reference(ckpt: str):
    """Stock transformers Chameleon, bf16, eager, with qk_norm expanded."""
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    from transformers import ChameleonConfig, ChameleonForConditionalGeneration

    ckpt_p = Path(ckpt)
    index = json.loads((ckpt_p / "model.safetensors.index.json").read_text())
    sd, n_exp = {}, 0
    for shard in sorted(set(index["weight_map"].values())):
        full = load_file(str(ckpt_p / shard))
        for k, t in full.items():
            if ((".q_norm." in k or ".k_norm." in k)
                    and t.dim() == 2 and t.shape[0] == 1):
                t = t.repeat_interleave(32, dim=0)
                n_exp += 1
            sd[k] = t
        del full
    if n_exp != 128:
        print(f"  [warn] expanded {n_exp} qk_norm tensors, expected 128")

    cfg = ChameleonConfig.from_pretrained(ckpt)
    cfg._attn_implementation = "eager"
    torch.set_default_dtype(torch.bfloat16)
    model = ChameleonForConditionalGeneration(cfg)
    torch.set_default_dtype(torch.float32)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    if missing or unexpected:
        raise RuntimeError(f"ref load: missing={missing[:4]} unexpected={unexpected[:4]}")
    del sd
    return model.eval().cuda()


@torch.no_grad()
def reference_forward(model, ids: list):
    """Teacher-forced forward. Returns (per-layer hidden states, logits).

    Hidden states come from forward hooks on the decoder layers, NOT from
    ``output_hidden_states=True``: HF's ``all_hidden_states`` replaces the last
    entry with the *post-final-norm* tensor, so comparing it against a
    pre-norm probe shows a false cosine collapse.
    """
    caught = {}
    handles = []
    for li in PROBE_LAYERS:
        def hook(_m, _inp, out, li=li):
            caught[li] = (out[0] if isinstance(out, tuple) else out)[0].float().cpu()
            return None          # a non-None hook return REPLACES the output
        handles.append(model.model.layers[li].register_forward_hook(hook))

    def norm_hook(_m, _inp, out):
        caught["final_norm"] = out[0].float().cpu()
        return None
    handles.append(model.model.norm.register_forward_hook(norm_hook))
    try:
        t = torch.tensor([ids], device="cuda")
        logits = model(input_ids=t).logits[0].float().cpu()
    finally:
        for h in handles:
            h.remove()
    return caught, logits


# ======================================================================
# Gates
# ======================================================================

def gate_graph_safety(front) -> bool:
    """Capture the decode body and prove it is not frozen (stale-value test)."""
    print("\n── graph-safety gate (decode body) ──")
    pos = front.S
    tok_a, tok_b = 16853, 40000
    try:
        front.decode_step(tok_a, pos=pos)          # warm every M=1 shape first
        front.decode_step(tok_a, pos=pos)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream=s):
                # The kernels MUST be launched on the capture stream; on stream 0
                # they are silently not recorded and the replay looks frozen.
                front.decode_step(front._tok_dev, pos=pos,
                                  stream=int(s.cuda_stream))
        print("  capture           : OK (no code=13)")
    except Exception as e:                                    # pragma: no cover
        print(f"  capture           : FAIL — {type(e).__name__}: {e}")
        return False

    front._tok_dev.fill_(tok_a); g.replay(); torch.cuda.synchronize()
    la = front._logits.clone()
    front._tok_dev.fill_(tok_b); g.replay(); torch.cuda.synchronize()
    lb = front._logits.clone()
    c = cosine(text_slice(la), text_slice(lb))
    frozen = torch.equal(la, lb)
    print(f"  stale-value        : {'FAIL (frozen)' if frozen else 'PASS'} "
          f"(cos between two seed tokens = {c:.4f})")
    return not frozen


def gate_overflow(front) -> bool:
    """FP16 residual health.

    The gate is **finiteness**, not an absolute magnitude. Chameleon's L31
    residual legitimately reaches ~2.6e5 in the bf16 reference (the massive
    activation), so any absolute threshold below that would fail by
    construction. What must not happen is inf/nan, which the
    ``ffn_down_clamp`` prevents by capping the down output just under FP16's
    65504. Saturation at the clamp is therefore *expected* at L31 and is
    reported for information only.
    """
    print("\n── fp16 residual health (finite + clamp saturation) ──")
    snaps = front.snapshot_probe()
    if not snaps:
        print("  (frontend built without probe_layers — skipped)")
        return True
    clamp = float(getattr(front, "ffn_down_clamp", 0.0) or 0.0)
    bad, sat = [], []
    worst, worst_k = 0.0, ""
    for k, v in snaps.items():
        if not bool(torch.isfinite(v).all()):
            bad.append(k)
        m = float(v[torch.isfinite(v)].abs().max()) if v.numel() else 0.0
        if clamp and m >= clamp * 0.98:
            sat.append(k)
        if m > worst:
            worst, worst_k = m, k
    print(f"  max finite |x|     : {worst:.0f} at {worst_k} "
          f"(fp16 max 65504, clamp {clamp:.0f})")
    print(f"  saturating at clamp: {sat if sat else 'none'} "
          f"{'(expected at L31)' if sat else ''}")
    ok = not bad
    print(f"  inf/nan            : {bad if bad else 'none'} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default="Describe this image.")
    ap.add_argument("--steps", type=int, default=16, help="greedy tokens to compare")
    ap.add_argument("--max-seq", type=int, default=1280)
    ap.add_argument("--int4", action="store_true")
    ap.add_argument("--int4-down", action="store_true")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--vq-fp16-argmin", action="store_true",
                    help="measure VQ index drift instead of avoiding it")
    ap.add_argument("--split-kv-bias", type=int, default=4)
    ap.add_argument("--skip-ref", action="store_true",
                    help="run only the FlashRT-side gates")
    args = ap.parse_args()

    from flash_rt.frontends.torch.chameleon_rtx_sm87 import ChameleonTorchFrontendRtxSm87

    print("=" * 68)
    print("Chameleon-7B Orin SM87 — Gate 1")
    print("=" * 68)

    front = ChameleonTorchFrontendRtxSm87(
        args.checkpoint, max_seq=args.max_seq,
        use_int4=args.int4, use_int4_down=args.int4_down,
        split_kv_bias=args.split_kv_bias,
        vq_argmin_fp32=not args.vq_fp16_argmin,
        probe_layers=PROBE_LAYERS)
    print(f"tier={front.precision_tier}  spec={front.precision_spec()}")

    # ---- prompt ----
    images = None
    text = args.prompt
    if not args.text_only:
        if args.image:
            from PIL import Image
            images = [Image.open(args.image).convert("RGB")]
        else:
            import numpy as np
            from PIL import Image
            images = [Image.fromarray(np.random.RandomState(0).randint(
                0, 256, (480, 640, 3), dtype=np.uint8))]
            print("  [note] no --image given; using deterministic noise")
        text = "<image>" + args.prompt
    front.set_prompt(text, images=images)
    ids = front.input_ids.tolist()
    print(f"\nISL={len(ids)} (1 BOS + n_img*1026 + text + 1 sep)  "
          f"images={front.timing['n_images']}  "
          f"prompt_ms={front.timing['prompt_ms']:.0f}")

    # ---- FlashRT teacher-forced logits + probes ----
    lg_frt = front.prefill(logits_all=True).float().cpu()
    probes = front.snapshot_probe()
    print(f"prefill_ms={front.timing['prefill_ms']:.0f}")

    ok_overflow = gate_overflow(front)
    ok_graph = gate_graph_safety(front)

    # ---- greedy text ----
    front.set_prompt(text, images=images)
    frt_ids = front.generate(max_new_tokens=args.steps, return_ids=True)
    frt_txt = front.processor.tokenizer.decode(frt_ids, skip_special_tokens=True)
    tm = front.timing
    print(f"\n── generation ──\n  FlashRT ids : {frt_ids}"
          f"\n  FlashRT text: {frt_txt!r}"
          f"\n  decode      : {tm['decode_ms_per_token']:.2f} ms/token "
          f"= {tm['decode_tok_s']:.2f} tok/s  (ISL={len(ids)}, OSL={args.steps})")

    if args.skip_ref:
        print("\n(--skip-ref: reference comparison not run)")
        return 0 if (ok_graph and ok_overflow) else 1

    # ---- reference ----
    print(f"\n── HF reference (bf16 eager) ──")
    ref = load_reference(args.checkpoint)
    ref_h, ref_lg = reference_forward(ref, ids)

    print("\n  layer   cosine   norm-ratio   FlashRT|max|   ref|max|")
    worst = 1.0
    for li in PROBE_LAYERS:
        a = probes[f"layer_{li}"].float().cpu()
        b = ref_h[li]
        c = cosine(a, b)
        worst = min(worst, c)
        print(f"  L{li:<5d} {c:.6f}   {float(a.norm()/b.norm()):.4f}      "
              f"{float(a.abs().max()):9.1f}   {float(b.abs().max()):9.1f}")
    c_fn = cosine(probes["final_norm"].float().cpu(), ref_h["final_norm"])
    print(f"  final  {c_fn:.6f}")

    # logits + argmax over the whole teacher-forced sequence
    a_lg = text_slice(lg_frt)
    b_lg = text_slice(ref_lg)
    c_last = cosine(a_lg[-1], b_lg[-1])
    am_frt = a_lg.argmax(-1)
    am_ref = b_lg.argmax(-1)
    exact = am_frt == am_ref

    # A BF16 tie is not a precision failure: when the reference's top-1 and
    # top-2 are within one BF16 ULP the winner is numerically arbitrary, and
    # any engine may legitimately pick either. Classify those separately
    # instead of scoring them as errors.
    top2 = b_lg.topk(2, dim=-1).values
    gap = (top2[:, 0] - top2[:, 1]).abs()
    ulp = top2[:, 0].abs() * 2 ** -8            # BF16 has 8 mantissa bits
    tied = gap <= ulp
    real_bad = (~exact) & (~tied)
    n = len(am_ref)
    match_exact = float(exact.float().mean())
    match_adj = float((exact | tied).float().mean())
    print(f"\n  last-row logit cosine : {c_last:.6f}")
    print(f"  argmax exact match    : {match_exact*100:.2f}% "
          f"({int(exact.sum())}/{n})")
    print(f"  of {int((~exact).sum())} mismatches: {int(((~exact) & tied).sum())} "
          f"are BF16 ties (gap <= 1 ulp), {int(real_bad.sum())} are real")
    print(f"  tie-adjusted match    : {match_adj*100:.2f}%")
    if int(real_bad.sum()):
        g = gap[real_bad]
        print(f"  real-mismatch ref gap : median={float(g.median()):.4f} "
              f"max={float(g.max()):.4f} (logit scale "
              f"~{float(top2[:, 0].abs().median()):.1f})")
    match = match_adj

    # Split by position class. At an *image* position the model predicts the
    # next token while all 8192 image ids are masked out of the logits, so the
    # winner is an arbitrary low-confidence text token — averaging over the 1024
    # image positions swamps the handful that actually drive generation. Gate on
    # the text positions only.
    ids_t = torch.tensor(ids)
    is_img = ((ids_t >= IMG_LO) & (ids_t < IMG_HI)) | (ids_t == 8197) | (ids_t == 8196)
    for label, sel in (("image", is_img), ("text ", ~is_img)):
        k = int(sel.sum())
        if not k:
            continue
        e = float(exact[sel].float().mean())
        a = float((exact | tied)[sel].float().mean())
        print(f"  {label} positions ({k:4d}) : exact {e*100:6.2f}%  "
              f"tie-adjusted {a*100:6.2f}%  "
              f"median ref gap {float(gap[sel].median()):.3f}")
    text_sel = ~is_img
    if int(text_sel.sum()):
        match = float((exact | tied)[text_sel].float().mean())

    # greedy text identity
    with torch.no_grad():
        gen = ref.generate(input_ids=torch.tensor([ids], device="cuda"),
                           max_new_tokens=args.steps, do_sample=False,
                           num_beams=1)
    ref_new = gen[0, len(ids):].tolist()
    ref_txt = front.processor.tokenizer.decode(ref_new, skip_special_tokens=True)
    n_pref = 0
    for x, y in zip(frt_ids, ref_new):
        if x != y:
            break
        n_pref += 1
    print(f"\n  reference ids : {ref_new}")
    print(f"  reference text: {ref_txt!r}")
    print(f"  identical prefix: {n_pref}/{min(len(frt_ids), len(ref_new))} tokens")

    # ---- verdict ----
    cos_gate = 0.99 if front.use_int4 else 0.97
    n_text = int(text_sel.sum())
    checks = [
        ("worst layer cosine", worst >= cos_gate, f"{worst:.4f} >= {cos_gate}"),
        ("last-row logit cosine", c_last >= 0.999, f"{c_last:.6f} >= 0.999"),
        ("greedy text identical", n_pref == len(ref_new),
         f"{n_pref}/{len(ref_new)}"),
        ("graph safety", ok_graph, ""),
        ("fp16 residual finite", ok_overflow, ""),
    ]
    # Only binding with a meaningful sample: an image-heavy prompt leaves a
    # handful of text positions, where one near-tie flip swings the rate by
    # >15 points. Greedy text identity above is the metric that actually
    # tracks generation quality.
    if n_text >= 32:
        checks.insert(2, ("argmax match (text)", match >= 0.99,
                          f"{match*100:.2f}% >= 99% (n={n_text})"))
    else:
        print(f"\n  note: only {n_text} text positions — the argmax gate is "
              f"reported as informational, not binding "
              f"({match*100:.2f}% tie-adjusted)")
    print("\n" + "=" * 68)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:24s} {detail}")
    allok = all(c[1] for c in checks)
    print(f"\nGATE 1: {'PASS' if allok else 'FAIL'}")
    print("=" * 68)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
