from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify_qwen36_rocm_64layer_owned_decode import OwnedDecodeRunner


def full_logits_fp8(runner: OwnedDecodeRunner, h0: torch.Tensor) -> torch.Tensor:
    h = runner.forward_hidden(h0, verbose=False)
    _, final_q, final_s = runner.norm_quant(h, runner.top("final_norm_eff_w"), "final_norm")
    out = runner.b("lm_head_fp8_logits", (1, int(runner.top("lm_head_fp8_w").shape[0])))
    runner.aiter.gemm_a8w8_blockscale_ck(
        final_q,
        runner.top("lm_head_fp8_w"),
        final_s,
        runner.top("lm_head_fp8_s"),
        out,
    )
    return out


def main() -> None:
    from flash_rt.frontends.torch.qwen36_rocm_weights import extract_weights_qwen36_fp8_rocm

    model = os.environ.get("QWEN36_MODEL", "Qwen/Qwen3.6-27B-FP8")
    weight_mode = os.environ.get("QWEN36_WEIGHT_MODE", "fp8_fnuz_cached")
    handles = extract_weights_qwen36_fp8_rocm(model, weight_mode=weight_mode)
    runner = OwnedDecodeRunner(handles)
    token_id = torch.tensor([0], device="cuda", dtype=torch.long)
    h0 = runner.top("embed_w").index_select(0, token_id).contiguous()
    if "lm_head_fp8_w" not in handles.ptrs or "lm_head_fp8_s" not in handles.ptrs:
        print("missing_lm_head_fp8_cache")
        return

    eager = None
    for _ in range(2):
        eager = full_logits_fp8(runner, h0)
    torch.cuda.synchronize()
    eager_mean = float(eager.float().abs().mean().item())
    eager_top = int(torch.argmax(eager[0]).item())
    eager_ref = eager.detach().clone()
    print("eager_logits_mean_abs", eager_mean)
    print("eager_logits_finite", bool(torch.isfinite(eager.float()).all().item()))
    print("eager_top_token", eager_top)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = full_logits_fp8(runner, h0)
        torch.cuda.synchronize()
    except Exception as exc:
        print("graph_capture_error", type(exc).__name__, str(exc))
        return

    captured.zero_()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    graph_mean = float(captured.float().abs().mean().item())
    graph_top = int(torch.argmax(captured[0]).item())
    print("graph_logits_mean_abs", graph_mean)
    print("graph_logits_finite", bool(torch.isfinite(captured.float()).all().item()))
    print("graph_top_token", graph_top)
    print("graph_max_abs_vs_eager", float((captured.float() - eager_ref.float()).abs().max().item()))
    print(
        "graph_cos_vs_eager",
        f"{F.cosine_similarity(captured.float().flatten(), eager_ref.float().flatten(), dim=0).item():.8f}",
    )

    iters = int(os.environ.get("GRAPH_ITERS", "10"))
    t0 = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    print("graph_logits_replay_s", (time.perf_counter() - t0) / iters)


if __name__ == "__main__":
    main()
