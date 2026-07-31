"""Equivalence test for the prefill routing kernel.

The kernel replaces a softmax, a top-k, a renormalising divide, a stable
argsort, two gathers, a bincount, a cumulative sum and a scatter. Each output
is checked against that chain, and the checks are not all the same kind:

- ``se``, ``stok`` and ``group_off`` are the sorted layout the grouped GEMM
  reads and must match exactly. A cosine over these would pass while a slot
  sat under the wrong expert's weight.
- The selected expert set per token must match exactly, for the same reason.
- The weights are compared by relative error, since the two reduce the softmax
  in different orders.
- The rank order *within* a token's top-k is deliberately not compared. bf16
  logits tie exactly at the top-k boundary and neither implementation defines
  which of two equal experts it ranks first; what must hold is that the expert
  in the row a slot points at is the expert that slot chose.

Logits are drawn peaked rather than uniform: top-k selection is decided by the
tail, and uniform draws produce a tie structure a router does not.
"""

import pytest
import torch
import torch.nn.functional as F

N_EXPERTS, TOPK = 256, 8


def _load_fvk():
    # Deliberately not the sibling WY suite's loader: that one skips unless
    # the 27B kernels are present, and this kernel does not depend on them.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the routing kernel test")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except Exception as exc:                # pragma: no cover - environmental
        pytest.skip(f"flash_rt_kernels is not built: {exc}")
    return fvk


def _ptr(x):
    return x.data_ptr()


def _tensor_route(logits, n_experts, topk):
    prob = F.softmax(logits.float(), -1)
    tw, ti = torch.topk(prob, topk, -1)
    tw = tw / tw.sum(-1, keepdim=True)
    exp_flat = ti.reshape(-1).to(torch.int32)
    order = exp_flat.argsort(stable=True)
    se = exp_flat[order].contiguous()
    tok = torch.arange(logits.shape[0],
                       device=logits.device).repeat_interleave(topk)
    stok = tok[order]
    counts = torch.bincount(se, minlength=n_experts)
    group_off = torch.zeros(n_experts + 1, dtype=torch.int32,
                            device=logits.device)
    group_off[1:] = counts.cumsum(0).to(torch.int32)
    return ti.to(torch.int32), tw, se, stok, group_off


@pytest.mark.parametrize("S", [1, 64, 256, 1000, 2048])
def test_route_matches_tensor_chain(S):
    fvk = _load_fvk()
    if not hasattr(fvk, "moe_route_prefill_bf16"):
        pytest.skip("moe_route_prefill_bf16 not in this build")

    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(7 + S)
    logits = (torch.randn(S, N_EXPERTS, generator=g, device=dev) * 2.5
              ).to(torch.bfloat16)
    ti_r, tw_r, se_r, stok_r, goff_r = _tensor_route(logits, N_EXPERTS, TOPK)

    slots = S * TOPK
    ti = torch.empty(S, TOPK, dtype=torch.int32, device=dev)
    tw = torch.empty(S, TOPK, dtype=torch.float32, device=dev)
    se = torch.empty(slots, dtype=torch.int32, device=dev)
    # int64: this is handed to the activation quantiser as its gather index and
    # that kernel reads a long.
    stok = torch.empty(slots, dtype=torch.int64, device=dev)
    inv = torch.empty(slots, dtype=torch.int32, device=dev)
    goff = torch.empty(N_EXPERTS + 1, dtype=torch.int32, device=dev)
    nbytes = int(fvk.moe_route_prefill_workspace_bytes(S, TOPK, N_EXPERTS))
    ws = torch.empty(nbytes, dtype=torch.uint8, device=dev)

    rc = fvk.moe_route_prefill_bf16(
        _ptr(logits), _ptr(ti), _ptr(tw), _ptr(se), _ptr(stok), _ptr(inv),
        _ptr(goff), _ptr(ws), nbytes, S, N_EXPERTS, TOPK,
        torch.cuda.current_stream().cuda_stream)
    assert rc == 0, f"routing kernel returned {rc}"
    torch.cuda.synchronize(dev)

    assert torch.equal(se, se_r), "sorted expert layout differs"
    assert torch.equal(stok, stok_r), "sorted token layout differs"
    assert torch.equal(goff, goff_r), "group boundaries differ"

    # The expert in the row a slot points at is the expert that slot chose.
    assert torch.equal(se[inv.long()].reshape(S, TOPK), ti)

    oi = ti.argsort(stable=True, dim=-1)
    oj = ti_r.argsort(stable=True, dim=-1)
    assert torch.equal(torch.gather(ti, 1, oi), torch.gather(ti_r, 1, oj)), \
        "selected expert sets differ"
    rel = ((torch.gather(tw, 1, oi) - torch.gather(tw_r, 1, oj)).abs()
           / torch.gather(tw_r, 1, oj).abs().clamp_min(1e-9)).max()
    assert rel < 1e-5, f"weights differ by {rel:.3e}"


@pytest.mark.parametrize("S", [64, 2048])
def test_route_is_reproducible(S):
    """Prefill seeds a decode that has to reproduce, so the permutation may
    not depend on the order blocks happen to arrive in."""
    fvk = _load_fvk()
    if not hasattr(fvk, "moe_route_prefill_bf16"):
        pytest.skip("moe_route_prefill_bf16 not in this build")

    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(11)
    logits = (torch.randn(S, N_EXPERTS, generator=g, device=dev) * 2.5
              ).to(torch.bfloat16)
    slots = S * TOPK
    nbytes = int(fvk.moe_route_prefill_workspace_bytes(S, TOPK, N_EXPERTS))

    def run():
        ti = torch.empty(S, TOPK, dtype=torch.int32, device=dev)
        tw = torch.empty(S, TOPK, dtype=torch.float32, device=dev)
        se = torch.empty(slots, dtype=torch.int32, device=dev)
        stok = torch.empty(slots, dtype=torch.int64, device=dev)
        inv = torch.empty(slots, dtype=torch.int32, device=dev)
        goff = torch.empty(N_EXPERTS + 1, dtype=torch.int32, device=dev)
        ws = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        rc = fvk.moe_route_prefill_bf16(
            _ptr(logits), _ptr(ti), _ptr(tw), _ptr(se), _ptr(stok), _ptr(inv),
            _ptr(goff), _ptr(ws), nbytes, S, N_EXPERTS, TOPK,
            torch.cuda.current_stream().cuda_stream)
        assert rc == 0
        torch.cuda.synchronize(dev)
        return se, stok, inv, goff, ti, tw

    a = run()
    for _ in range(3):
        b = run()
        for x, y in zip(a, b):
            assert torch.equal(x, y), "routing is not reproducible"


def test_route_rejects_shapes_it_cannot_hold():
    """The top-k spreads a logit row across one warp, so the expert count has
    to be 32 times a power of two. Say so rather than compute nonsense."""
    fvk = _load_fvk()
    if not hasattr(fvk, "moe_route_prefill_bf16"):
        pytest.skip("moe_route_prefill_bf16 not in this build")

    dev = "cuda:0"
    S, E = 16, 96                      # 96 = 32 * 3, not a power of two
    logits = torch.zeros(S, E, dtype=torch.bfloat16, device=dev)
    slots = S * TOPK
    buf = lambda n, dt: torch.empty(n, dtype=dt, device=dev)
    nbytes = int(fvk.moe_route_prefill_workspace_bytes(S, TOPK, E))
    rc = fvk.moe_route_prefill_bf16(
        _ptr(logits), _ptr(buf(S * TOPK, torch.int32)),
        _ptr(buf(S * TOPK, torch.float32)), _ptr(buf(slots, torch.int32)),
        _ptr(buf(slots, torch.int64)), _ptr(buf(slots, torch.int32)),
        _ptr(buf(E + 1, torch.int32)), _ptr(buf(nbytes, torch.uint8)),
        nbytes, S, E, TOPK, torch.cuda.current_stream().cuda_stream)
    assert rc != 0, "unsupported expert count was accepted"
