"""NVFP4 whole-LLM implementation of ``decoder_llm`` — native build.

The native omnivoice engines (``flash_rt.models.omnivoice.pipeline_rtx``)
promoted into the structures layer: the whole decoder stack forward is
the fused FP4 engine (fp4 GEMMs + fused qk-norm+RoPE + FA2 + fused
residual/norm/quant), with whole-step CUDA graph capture. Kernel
resolution is the PR-175 tiering: hub artifact first, the local native
build second, the retained host stack always the floor. This impl
consumes the local build.

Dispatch inside the stack boundary mirrors the native schedule: the
CFG batch (B=2) rides the BF16 engine, single-stream (B=1) rides the
FP4 graph. The MaskGIT two-phase loop is exposed as ``maskgit_gen`` on
this module — the schedule structure for non-text generation.

The impl's profile envelope is the native engine's v1 contract
(D=1024, L=28, NH=16, NKV=8, HD=128, FFN=3072, RoPE theta 1e6); other
profiles refuse cleanly (the fp8-KV band precedent).
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam

#: the native engine's v1 profile (kernel constants are compiled in)
PROFILE = dict(D=1024, L=28, NH=16, NKV=8, HD=128, FFN=3072)


@lru_cache(maxsize=1)
def _native():
    """(FlashRTLlm, FlashRTLlmBF16) from the local build, or None."""
    try:
        from flash_rt.models.omnivoice.pipeline_rtx import (  # noqa: F401
            FlashRTLlm,
            FlashRTLlmBF16,
        )
        from flash_rt import flash_rt_kernels  # noqa: F401
        from flash_rt import flash_rt_omnivoice  # noqa: F401
        from flash_rt import flash_rt_fa2  # noqa: F401
        return FlashRTLlm, FlashRTLlmBF16
    except ImportError:
        return None


class DecoderLlmNvfp4(GuardedSeam, torch.nn.Module):
    """Whole-stack seam: fused FP4 engine with per-batch dispatch."""

    _frt_host_attr = "host_llm"
    _frt_can_fallback = True

    def __init__(self, host_llm, bf16, fp4, device):
        super().__init__()
        self.host_llm = host_llm
        self._bf16 = bf16
        self._fp4 = fp4
        self._device = device
        guard = self._frt_arm(dtypes=CAST_OK, device=device)
        guard.notes["backend"] = "nvfp4_llm"

    def forward(self, *a, **kw):
        e = kw.get("inputs_embeds")
        if e is None and len(a) >= 2 and isinstance(a[1], torch.Tensor):
            e = a[1]
        if e is None:
            return self._frt_host()(*a, **kw)
        admitted = self._frt_admit(e, *a, **kw)
        if admitted is not PROCEED:
            return admitted
        if e.shape[0] > 1:
            # CFG batch: BF16 fused engine (no quantization drift)
            return self._bf16.forward(e, attention_mask=kw.get(
                "attention_mask"))
        return self._fp4.forward_graph(e, attention_mask=kw.get(
            "attention_mask"))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_llm":
                raise
            return getattr(super().__getattr__("host_llm"), name)


def _profile_of(host_llm) -> dict:
    cfg = getattr(host_llm, "config", None)
    d = int(cfg.hidden_size) if cfg is not None else 0
    l = len(host_llm.layers) if hasattr(host_llm, "layers") else 0
    if cfg is not None:
        nh = int(cfg.num_attention_heads)
        nkv = int(getattr(cfg, "num_key_value_heads", nh))
        hd = int(getattr(cfg, "head_dim", d // nh))
        f = int(cfg.intermediate_size)
    else:
        nh = nkv = hd = f = 0
    return dict(D=d, L=l, NH=nh, NKV=nkv, HD=hd, FFN=f)


@torch.no_grad()
def bind_decoder_stack(host_llm, *, variant=None, original=None):
    """Bind the whole stack to the native NVFP4 engines.

    ``host_llm`` is the decoder stack module (layers / embed_tokens /
    norm / rotary_emb). Weights are read in place; the host stack is
    retained for fallback, so it stays whole.
    """
    eng = _native()
    if eng is None:
        raise ValueError(
            "refused: decoder_llm nvfp4 needs the locally built "
            "flash_rt_kernels + flash_rt_omnivoice + flash_rt_fa2; "
            "rebuild with -DFLASHRT_ENABLE_OMNIVOICE=ON -DGPU_ARCH=120")
    got = _profile_of(host_llm)
    if got != PROFILE:
        raise ValueError(
            f"refused: decoder_llm nvfp4 profile mismatch — the native "
            f"engine's v1 contract is {PROFILE}, this host is {got}")
    FlashRTLlm, FlashRTLlmBF16 = eng
    dev = str(host_llm.embed_tokens.weight.device)
    bf16 = FlashRTLlmBF16(host_llm, dev)
    fp4 = FlashRTLlm(host_llm, dev)
    # calibrate: packed weights + workspaces (native inject's shapes)
    c2 = torch.randn(2, 178, PROFILE["D"], device=dev,
                     dtype=torch.bfloat16) * 0.02
    bf16.calibrate(c2)
    fp4.calibrate(c2)
    fp4.WL_bf16 = None
    bf16.WL_fp4 = None
    bf16._fp4_act = None
    bf16._alphas = None
    c1 = torch.randn(1, 178, PROFILE["D"], device=dev,
                     dtype=torch.bfloat16) * 0.02
    for _ in range(3):
        fp4.forward(c1)
    torch.cuda.synchronize()
    fp4._capture_graph(c1)
    for _ in range(3):
        fp4.forward_graph(c1)
    torch.cuda.synchronize()
    bf16._graph = None

    seam = DecoderLlmNvfp4(host_llm, bf16, fp4, torch.device(dev))
    seam._fp4 = fp4  # maskgit_gen's schedule reads the engine here
    if original is not None:
        seam.host_llm = original

    # bind-time smoke: one real forward through the engines (AGENTS §2.8)
    probe = seam.forward(inputs_embeds=torch.randn(
        1, 16, PROFILE["D"], device=dev, dtype=torch.bfloat16))
    if isinstance(probe, tuple):
        probe = probe[0]
    if tuple(probe.shape) != (1, 16, PROFILE["D"]) or \
            not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: decoder_llm nvfp4 bind smoke produced "
            f"{tuple(probe.shape)}, "
            f"finite={bool(torch.isfinite(probe).all())}")
    return seam


class MaskgitLoop:
    """The MaskGIT schedule as a serving object (the decode_loop twin).

    ``structures.maskgit_loop(model)`` returns one of these; ``generate``
    runs the two-phase loop (BF16 CFG steps, then FP4 noCFG single-stream
    graph replays) over whatever ``decoder_llm`` seam is attached.
    """

    def __init__(self, model, *, cfg_ratio: float = 0.05,
                 bookend: bool = False):
        self._model = model
        self._cfg_ratio = cfg_ratio
        self._bookend = bookend

    def generate(self, task, gen_config):
        return maskgit_gen(self._model, task, gen_config,
                           cfg_ratio=self._cfg_ratio,
                           bookend=self._bookend)


def maskgit_gen(model, task, gen_config, cfg_ratio=0.05, bookend=False):
    """The MaskGIT two-phase schedule over an attached decoder_llm seam.

    Phase 1: BF16 CFG (B=2, cfg_ratio fraction of steps). Phase 2: FP4
    noCFG (B=1, graph replay). Uses the native ``_optimize_maskgit``
    loop; the seam's per-batch dispatch replaces the inject's forward
    monkeypatching (B=2 -> BF16 engine, B=1 -> FP4 graph).
    """
    from flash_rt.models.omnivoice import pipeline_rtx as prtx

    seam = model.llm
    fp4 = getattr(seam, "_fp4", None)
    if fp4 is None:
        raise ValueError(
            "maskgit_gen: no decoder_llm structure attached (auto_swaps "
            "with decoder_llm + scheme nvfp4_static)")
    orig = model._generate_iterative
    prtx._optimize_maskgit(model, fp4, cfg_ratio, bookend)
    try:
        return model._generate_iterative(task, gen_config)
    finally:
        model._generate_iterative = orig
