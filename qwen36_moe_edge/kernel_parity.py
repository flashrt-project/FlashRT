#!/usr/bin/env python3
"""Cross-architecture parity for the qwen3_5_moe core and W4A16 kernels.

The tier split was verified by compiling for sm_87 and sm_110, which proves
nothing about what the kernels compute there. This exercises each binding with
seeded inputs and records its output, so the same script run on another target
can be diffed against a reference produced on sm_120a.

Deliberately no model and no checkpoint: shapes come from the Qwen3.6 geometry
and values from a fixed generator, so any machine can run it.

Inputs are generated on the CPU and stored alongside the outputs. A comparison
run loads them from the reference rather than regenerating: CUDA RNG is not
bit-reproducible across architectures -- the Philox thread mapping follows
occupancy -- so regenerating on the target would compare kernels on different
data and read as a kernel failure. Divergence appears only past the first
launch block, which is why small tensors matched and large ones did not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

HIDDEN = 2048
INTERMEDIATE = 512
TOPK = 8
NUM_EXPERTS = 256
# Linear-attention geometry. The split kernel broadcasts q and k from the 16
# stored key heads to all 32 value heads, so every one of its three outputs is
# NV * HK wide -- not the 16-head width the stored layout suggests. Sizing
# q/k at 2048 makes the kernel write past them into whatever the allocator
# placed next, which shows up as a corrupted third output.
NV = 32
HK = 128
HV = 128


# Inputs recorded by the reference run and replayed by comparison runs.
_INPUTS: dict[str, torch.Tensor] = {}
_REPLAY: dict[str, torch.Tensor] | None = None
_CPU_GEN = torch.Generator().manual_seed(20260730)
_CASE = ""


def _record(name: str, tensor: torch.Tensor, device) -> torch.Tensor:
    """Return the replayed input if one exists, else keep what we generated."""
    key = f"{_CASE}.in.{name}"
    if _REPLAY is not None:
        if key not in _REPLAY:
            raise KeyError(f"reference has no input {key}")
        tensor = _REPLAY[key].to(dtype=tensor.dtype)
    _INPUTS[key] = tensor.detach().cpu()
    return tensor.to(device)


def _bf16(name, shape, device, scale=1.0):
    """A bfloat16 input, generated on the CPU so it is machine-independent."""
    values = torch.randn(*shape, generator=_CPU_GEN, dtype=torch.float32)
    return _record(name, (values * scale).to(torch.bfloat16), device)


def case_bf16_matvec(fvk, device):
    x = _bf16("x", (1, HIDDEN), device)
    w = _bf16("w", (HIDDEN, HIDDEN), device, 0.02)
    out = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)
    rc = fvk.bf16_matvec_sm120_bf16(
        x.data_ptr(), w.data_ptr(), out.data_ptr(), HIDDEN, HIDDEN, 0)
    torch.cuda.synchronize()
    return {"rc": rc, "out": out, "torch": (x.float() @ w.float().T)}


def case_router_topk(fvk, device):
    logits = _bf16("logits", (NUM_EXPERTS,), device, 4.0).contiguous()
    idx = torch.empty(TOPK, dtype=torch.int32, device=device)
    val = torch.empty(TOPK, dtype=torch.float32, device=device)
    rc = fvk.moe_router_topk_sm120_bf16(
        logits.data_ptr(), idx.data_ptr(), val.data_ptr(),
        NUM_EXPERTS, TOPK, 0)
    torch.cuda.synchronize()
    reference = torch.topk(logits.float(), TOPK)
    return {"rc": rc, "idx": idx, "val": val,
            "torch_idx": reference.indices.to(torch.int32),
            "torch_val": reference.values}


def case_silu_mul(fvk, device):
    n = 4096
    g = _bf16("g", (n,), device)
    u = _bf16("u", (n,), device)
    out = torch.zeros(n, dtype=torch.bfloat16, device=device)
    rc = fvk.silu_mul_sm120_bf16(
        g.data_ptr(), u.data_ptr(), out.data_ptr(), n, 0)
    torch.cuda.synchronize()
    return {"rc": rc, "out": out,
            "torch": torch.nn.functional.silu(g.float()) * u.float()}


def case_sigmoid_mul(fvk, device):
    n = 4096
    x = _bf16("x", (n,), device)
    gate = _bf16("gate", (n,), device)
    out = torch.zeros(n, dtype=torch.bfloat16, device=device)
    rc = fvk.sigmoid_mul_sm120_bf16(
        x.data_ptr(), gate.data_ptr(), out.data_ptr(), n, 0)
    torch.cuda.synchronize()
    return {"rc": rc, "out": out,
            "torch": x.float() * torch.sigmoid(gate.float())}


def case_weighted_sum(fvk, device):
    d_dn = _bf16("d_dn", (TOPK, HIDDEN), device)
    rows = torch.arange(TOPK, dtype=torch.int32, device=device)
    weights = _record("weights", torch.softmax(
        torch.randn(TOPK, generator=_CPU_GEN), -1), device)
    # The reducer writes float32 and expects a flat buffer: see
    # tests/test_qwen36_moe_gpu.py and the decode call site. Handing it a
    # bfloat16 destination yields NaN, not a wrong-but-plausible answer.
    out = torch.zeros(HIDDEN, dtype=torch.float32, device=device)
    rc = fvk.moe_weighted_sum_sm120_bf16(
        d_dn.data_ptr(), rows.data_ptr(), weights.contiguous().data_ptr(),
        out.data_ptr(), 1, TOPK, HIDDEN, HIDDEN, 0)
    torch.cuda.synchronize()
    return {"rc": rc, "out": out, "torch": weights.float() @ d_dn.float()}


def case_w16a16_gemm(fvk, device):
    m, n, k = 16, HIDDEN, HIDDEN
    x = _bf16("x", (m, k), device)
    w = _bf16("w", (n, k), device, 0.02)
    out = torch.zeros(m, n, dtype=torch.bfloat16, device=device)
    rc = fvk.w16a16_gemm_sm120_bf16(
        x.data_ptr(), w.data_ptr(), out.data_ptr(), m, n, k, 1.0, 0)
    torch.cuda.synchronize()
    return {"rc": rc, "out": out, "torch": x.float() @ w.float().T}


def case_lin_split_qkv(fvk, device):
    S = 4
    conv_out = _bf16("conv_out", (S, 8192), device).contiguous()
    q32 = torch.zeros(S, NV, HK, dtype=torch.bfloat16, device=device)
    k32 = torch.zeros(S, NV, HK, dtype=torch.bfloat16, device=device)
    v32 = torch.zeros(S, NV, HV, dtype=torch.bfloat16, device=device)
    fvk.qwen35moe_lin_split_qkv_broadcast_bf16(
        conv_out.data_ptr(), q32.data_ptr(), k32.data_ptr(), v32.data_ptr(),
        S, 0)
    torch.cuda.synchronize()
    return {"q32": q32, "k32": k32, "v32": v32}


def case_e0m3_dequant(fvk, device):
    """The streamed-block decode: sign-magnitude 4-bit plus a two-level scale."""
    from qwen36_moe_edge.quantize_experts import _int4_weight, dequantize_int4

    rows, cols, group = 2 * INTERMEDIATE, HIDDEN, 16
    weight = _bf16("weight", (rows, cols), "cpu", 0.02).float()
    packed, scale, global_scale = _int4_weight(weight, group)
    packed = _record("packed", packed, device).contiguous()
    scale = _record("scale", scale, device).contiguous()
    out = torch.zeros(rows, cols, dtype=torch.bfloat16, device=device)
    rc = fvk.qwen35moe_e0m3_dequant_bf16(
        packed.data_ptr(), scale.data_ptr(), out.data_ptr(),
        rows, cols, group, float(global_scale), 0)
    torch.cuda.synchronize()
    reference = dequantize_int4(
        packed.cpu(), scale.cpu(), cols, group, global_scale)
    return {"rc": rc, "out": out, "torch": reference.to(device)}


def case_split_q_gate(fvk, device):
    S = 4
    q_proj = _bf16("q_proj", (S, 8192), device).contiguous()
    q_pre = torch.zeros(S, 4096, dtype=torch.bfloat16, device=device)
    gate = torch.zeros(S, 4096, dtype=torch.bfloat16, device=device)
    fvk.qwen35moe_split_q_gate_bf16(
        q_proj.data_ptr(), q_pre.data_ptr(), gate.data_ptr(), S, 0)
    torch.cuda.synchronize()
    return {"q_pre": q_pre, "gate": gate}


CASES = {
    "bf16_matvec": case_bf16_matvec,
    "moe_router_topk": case_router_topk,
    "silu_mul": case_silu_mul,
    "sigmoid_mul": case_sigmoid_mul,
    "moe_weighted_sum": case_weighted_sum,
    "w16a16_gemm": case_w16a16_gemm,
    "lin_split_qkv": case_lin_split_qkv,
    "split_q_gate": case_split_q_gate,
    "e0m3_dequant": case_e0m3_dequant,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from flash_rt import flash_rt_kernels as fvk

    capability = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name(0)} sm_{capability[0]}"
          f"{capability[1]}")

    global _REPLAY, _CASE
    if args.reference and args.reference.with_suffix(".pt").is_file():
        _REPLAY = torch.load(
            args.reference.with_suffix(".pt"), weights_only=True)
        print(f"replaying inputs from {args.reference.with_suffix('.pt')}")

    outputs, report = {}, {}
    for name, case in CASES.items():
        _CASE = name
        if not hasattr(fvk, _binding_of(name)):
            report[name] = {"status": "binding absent"}
            print(f"{name:<20} binding absent")
            continue
        try:
            result = case(fvk, args.device)
        except Exception as error:                       # noqa: BLE001
            report[name] = {"status": f"raised {type(error).__name__}: {error}"}
            print(f"{name:<20} RAISED {error}")
            continue
        entry = {"status": "ok"}
        if "rc" in result:
            entry["rc"] = int(result.pop("rc"))
        for key in list(result):
            if key.startswith("torch"):
                continue
            outputs[f"{name}.{key}"] = result[key].detach().float().cpu()
        # Local agreement with torch, where a reference was computed.
        for key in ("out", "val"):
            if key in result and "torch" in result:
                entry["torch_cosine"] = _cosine(
                    result[key], result["torch"])
        if "torch_idx" in result:
            # Compare as a set: the top-8 of this input contains an exact tie
            # (two logits at 8.9375), and the kernel and torch.topk are free to
            # break it differently while both being right.
            entry["topk_index_set_match"] = int(
                set(result["idx"].tolist())
                == set(result["torch_idx"].tolist()))
            entry["topk_value_match"] = int(torch.allclose(
                result["val"].cpu(), result["torch_val"].cpu()))
        report[name] = entry
        print(f"{name:<20} {entry}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**_INPUTS, **outputs}, args.output.with_suffix(".pt"))
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"capability": list(capability), "cases": report}, f,
                  indent=2)
        f.write("\n")

    if args.reference and args.reference.with_suffix(".pt").is_file():
        print("\n--- against reference ---")
        expected = _REPLAY
        worst = 1.0
        for key in sorted(k for k in set(expected) & set(outputs)
                          if ".in." not in k):
            cosine = _cosine(outputs[key], expected[key])
            exact = torch.equal(outputs[key], expected[key])
            worst = min(worst, cosine)
            print(f"{key:<34} cos={cosine:.8f} bitwise={'yes' if exact else 'no'}")
        compared = {k for k in set(expected) | set(outputs) if ".in." not in k}
        missing = sorted(compared - (set(expected) & set(outputs)))
        if missing:
            print(f"outputs present on only one side: {missing}")
        print(f"worst cosine: {worst:.8f}")


def _binding_of(name: str) -> str:
    return {
        "lin_split_qkv": "qwen35moe_lin_split_qkv_broadcast_bf16",
        "split_q_gate": "qwen35moe_split_q_gate_bf16",
        "e0m3_dequant": "qwen35moe_e0m3_dequant_bf16",
    }.get(name, f"{name}_sm120_bf16")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().flatten().cpu()
    y = b.detach().float().flatten().cpu()
    return torch.nn.functional.cosine_similarity(x, y, dim=0).item()


if __name__ == "__main__":
    main()
