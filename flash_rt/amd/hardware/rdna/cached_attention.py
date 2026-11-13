"""Causal GQA against a persistent KV cache and a device-owned position."""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


# Sequence extent only controls masks/loop bounds, not the compiled tile shape.
@triton.jit(do_not_specialize=["M"])
def _attention(Q, K, V, POS, O, M, HQ: tl.constexpr,
               HK: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    kh = head // (HQ // HK)
    dim = tl.arange(0, D)
    col = tl.arange(0, BN)
    base = tl.load(POS)
    end = base + M
    q = tl.load(Q + (row[:, None] * HQ + head) * D + dim[None, :], row[:, None] < M, 0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.full((BM,), 0.0, tl.float32)
    acc = tl.full((BM, D), 0.0, tl.float32)
    for start in range(0, end, BN):
        key = start + col
        k = tl.load(K + (key[None, :] * HK + kh) * D + dim[:, None], key[None, :] < end, 0)
        scores = tl.dot(q, k).to(tl.float32) * (D ** -0.5 * 1.4426950408889634)
        valid = (key[None, :] < end) & (key[None, :] <= base + row[:, None])
        scores = tl.where(valid, scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        probability = tl.exp2(scores - new_max[:, None])
        correction = tl.exp2(maximum - new_max)
        denominator = denominator * correction + tl.sum(probability, 1)
        acc *= correction[:, None]
        v = tl.load(V + (key[:, None] * HK + kh) * D + dim[None, :], key[:, None] < end, 0)
        acc = tl.dot(probability.to(v.dtype), v, acc)
        maximum = new_max
    output = acc / denominator[:, None]
    tl.store(O + (row[:, None] * HQ + head) * D + dim[None, :], output, row[:, None] < M)


@triton.jit
def _decode_split(Q, K, V, POS, PART, MAX, DEN, HQ: tl.constexpr,
                  HK: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr,
                  BT: tl.constexpr, ROWS: tl.constexpr):
    kh, split = tl.program_id(0), tl.program_id(1)
    group = HQ // HK
    row, dim, col = tl.arange(0, ROWS), tl.arange(0, D), tl.arange(0, BT)
    q = tl.load(Q + (kh * group + row[:, None]) * D + dim[None, :], row[:, None] < group, 0)
    end = tl.load(POS) + 1
    maximum = tl.full((ROWS,), -float("inf"), tl.float32)
    denominator = tl.full((ROWS,), 0.0, tl.float32)
    acc = tl.full((ROWS, D), 0.0, tl.float32)
    # Share each KV tile across its query heads. The fixed split grid remains
    # valid as the device position changes under graph replay.
    for start in range(split * BT, end, SPLITS * BT):
        key = start + col
        k = tl.load(K + (key[None, :] * HK + kh) * D + dim[:, None], key[None, :] < end, 0)
        score = tl.dot(q, k).to(tl.float32) * (D ** -0.5 * 1.4426950408889634)
        score = tl.where(key[None, :] < end, score, -float("inf"))
        updated = tl.maximum(maximum, tl.max(score, 1))
        probability = tl.exp2(score - updated[:, None])
        correction = tl.exp2(maximum - updated)
        denominator = denominator * correction + tl.sum(probability, 1)
        acc *= correction[:, None]
        v = tl.load(V + (key[:, None] * HK + kh) * D + dim[None, :], key[:, None] < end, 0)
        acc = tl.dot(probability.to(v.dtype), v, acc)
        maximum = updated
    head = kh * group + row
    tl.store(PART + (head[:, None] * SPLITS + split) * D + dim[None, :], acc, row[:, None] < group)
    tl.store(MAX + head * SPLITS + split, maximum, row < group)
    tl.store(DEN + head * SPLITS + split, denominator, row < group)


@triton.jit
def _decode_combine(PART, MAX, DEN, O, D: tl.constexpr, SPLITS: tl.constexpr):
    head = tl.program_id(0)
    split, dim = tl.arange(0, SPLITS), tl.arange(0, D)
    maximum = tl.load(MAX + head * SPLITS + split)
    denominator = tl.load(DEN + head * SPLITS + split)
    weight = tl.where(denominator > 0, tl.exp2(maximum - tl.max(maximum, 0)), 0.0)
    partial = tl.load(PART + (head * SPLITS + split[:, None]) * D + dim[None, :])
    value = tl.sum(partial * weight[:, None], 0) / tl.sum(denominator * weight, 0)
    tl.store(O + head * D + dim, value)


@dataclass(frozen=True)
class DecodeWorkspace:
    partial: torch.Tensor
    maximum: torch.Tensor
    denominator: torch.Tensor


def allocate_decode_workspace(heads, head_dim, device):
    """Allocate the caller-owned scratch shared by sequential decoder layers."""
    partial = torch.empty((heads, 8, head_dim), device=device, dtype=torch.float32)
    maximum = torch.empty((heads, 8), device=device, dtype=torch.float32)
    return DecodeWorkspace(partial, maximum, torch.empty_like(maximum))


def cached_attention(q, k, v, position, out, *, workspace=None):
    """Causal BF16 GQA with an FP32 softmax denominator."""
    if (q.ndim != 3 or k.ndim != 3 or k.shape != v.shape
            or min(*q.shape, *k.shape) < 1 or q.shape[1] % k.shape[1] or q.shape[2] != k.shape[2]
            or q.shape[2] not in (64, 128) or out.numel() != q.numel()):
        raise ValueError("RDNA cached attention requires matching SHD GQA tensors with D64/D128")
    if (any(t.dtype != torch.bfloat16 or t.device != q.device or not t.is_contiguous()
            for t in (q, k, v, out)) or position.numel() != 1
            or position.dtype != torch.int32 or position.device != q.device):
        raise ValueError("RDNA cached attention requires contiguous BF16 tensors and an int32 device position")
    if workspace is not None:
        if (q.shape[0] != 1 or q.shape[1] // k.shape[1] > 16
                or workspace.partial.shape != (q.shape[1], 8, q.shape[2])
                or workspace.maximum.shape != (q.shape[1], 8)
                or workspace.denominator.shape != workspace.maximum.shape
                or any(t.dtype != torch.float32 or t.device != q.device or not t.is_contiguous()
                       for t in (workspace.partial, workspace.maximum, workspace.denominator))):
            raise ValueError("RDNA split decode attention requires a single query and matching FP32 scratch")
        args = (workspace.partial, workspace.maximum, workspace.denominator)
        _decode_split[(k.shape[1], 8)](
            q, k, v, position, *args, q.shape[1], k.shape[1], q.shape[2], 8, 64, 16,
            num_warps=4, num_stages=1)
        _decode_combine[(q.shape[1],)](*args, out, q.shape[2], 8, num_warps=4)
        return
    _attention[(triton.cdiv(q.shape[0], 16), q.shape[1])](
        q, k, v, position, out, q.shape[0], q.shape[1], k.shape[1], q.shape[2], 16, 64,
        num_warps=4, num_stages=1)
