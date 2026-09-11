"""vLLM engine family: explicit assembly onto a serving host's model.

A vLLM model is an ``nn.Module`` tree living inside the engine process,
but its seams do not match the static module patterns discovery reads:
projections are merged/parallel classes whose ``forward`` returns
``(out, bias)``, expert weights live stacked on a routed-experts child,
and the LM head is consulted through ``quant_method.apply`` rather than
module forward. This adapter recognises those seams by structure —
2-D ``weight`` plus a ``quant_method`` slot for projections, a
``w13_weight``/``w2_weight`` pair for an expert bank, a ``lm_head``
whose vocabulary row count may exceed a quantize entry's grid limit —
and never by class or model name.

Three engine facts shape the assembly, each carried here so callers do
not rediscover them:

- **Seats must be installed after weights load and before the engine's
  first trace.** vLLM's compiled artifact resolves parameters by tree
  path; a post-compile swap either raises ``KeyError`` or is silently
  bypassed. ``install_load_hook`` patches the model runner's
  ``load_model`` for exactly this window.
- **A Python shape branch dies in the compiled form.** vLLM traces with
  guard evaluation off, so band dispatch (decode rows to the packed
  bank, prefill rows to the retained host) lives inside a custom op:
  its body re-runs at capture (decode sizes bake the seam branch into
  the graphs) and eagerly at prefill.
- **The head is intercepted at ``quant_method``**, and binds as row
  slabs when the vocabulary exceeds the quantize entry's row support.

Everything installed through :func:`attach_engine` goes through
``swap.attach``; the returned handle detaches bit-exactly. The expert
bank and head interceptions are host mutations recorded as ``revert``
callables on the same handle, so one ``detach`` restores all of it.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from .. import swap as _swap
from ..impls.linear_proj import nvfp4_dynamic as _linear
from ..impls.moe_experts import nvfp4_w4a16 as _experts_w4a16
from ..impls.moe_experts import nvfp4_dynamic as _experts_w4a4

#: dense projection seams, by dataflow position suffix. These are
#: positions in the qwen3_5 family's dataflow, not module identities;
#: a host that lacks one simply contributes no seat.
DENSE_SEAT_SUFFIXES = (
    "linear_attn.out_proj", "linear_attn.in_proj_qkvz",
    "self_attn.qkv_proj", "self_attn.o_proj",
    "shared_expert.gate_up_proj", "shared_expert.down_proj",
    "mlp.gate_up_proj", "mlp.down_proj",
)

_SEATS_BY_IDX: dict[int, Any] = {}


# Registered at import: the registration itself must never sit on a
# traced path — a lazy first call lands inside dynamo and the schema
# inference graph-breaks the host's compiled forward.
@torch.library.custom_op("flash_rt_structures::vllm_moe_seat",
                         mutates_args=())
def _vllm_moe_seat_op(hidden: torch.Tensor, router_logits: torch.Tensor,
                      top_idx: torch.Tensor, top_w: torch.Tensor,
                      idx: int) -> torch.Tensor:
    return _SEATS_BY_IDX[idx].run(hidden, router_logits, top_idx, top_w)


@_vllm_moe_seat_op.register_fake
def _(hidden, router_logits, top_idx, top_w, idx):
    return torch.empty_like(hidden)


class _ProjSeat(nn.Module):
    """Preserves the engine's ``(out, bias)`` projection contract."""

    def __init__(self, seam):
        super().__init__()
        self.seam = seam

    def forward(self, x, *args, **kwargs):
        return self.seam(x), None


class _MoESeat(nn.Module):
    """Stands where the fused-MoE module stood: routing here, bank in
    the seam, the host's own shared-expert module added back (it owned
    it too, and its projections may themselves carry seats), and a
    declared band — decode rows walk the packed bank, prefill rows go
    to the retained host module."""

    #: rows above which the retained host module serves the batch.
    #: Measured, not assumed. A routed-MoE decode does not amortise the
    #: way a dense one does — eight tokens pick their own top-8 experts,
    #: so expert traffic grows with the batch instead of being shared,
    #: and a packed bank keeps paying well past batch one. Measured on
    #: Thor against vLLM 0.26 (35B-A3B, 128-token generations): 2.10x at
    #: batch 1, 2.45x at 4, 2.51x at 8, 1.65x at 16. The earlier value
    #: of 8 handed batch-16 traffic back to the host and threw that
    #: 1.65x away — the arm measured 0.98x, because at that batch the
    #: dense seats alone are worth nothing while the MoE seat is worth
    #: everything. 16 is the largest batch measured to pay; sweep with
    #: ``FRT_MOE_BAND_T`` before raising it further.
    BAND_T = int(os.environ.get("FRT_MOE_BAND_T", "16"))

    def __init__(self, seam, top_k, renormalize, shared, host):
        super().__init__()
        self.seam = seam
        self.top_k = top_k
        self.renormalize = renormalize
        self.shared = shared
        self.host = host
        self.host_internal = bool(getattr(host, "is_internal_router", False))
        self.is_internal_router = False   # the host block branches on this
        self._frt_host_serving = True     # prefill band runs through host
        self.idx = len(_SEATS_BY_IDX)
        _SEATS_BY_IDX[self.idx] = self

    def run(self, hidden_states, router_logits, top_idx, top_w):
        if hidden_states.shape[0] > self.BAND_T and self.host is not None:
            logits = (hidden_states if self.host_internal else router_logits)
            return self.host(hidden_states=hidden_states,
                             router_logits=logits)
        out = self.seam(hidden_states, top_idx, top_w)
        if self.shared is not None:
            out = out + self.shared(hidden_states)
        return out.to(hidden_states.dtype)

    def forward(self, hidden_states, router_logits):
        # routing stays in the traced region so inductor fuses the
        # softmax/topk/renormalize chain; the opaque op keeps only what
        # tracing would freeze — the band branch and the bank walk
        w = torch.softmax(router_logits.float(), dim=-1)
        tw, ti = torch.topk(w, self.top_k, dim=-1)
        if self.renormalize:
            tw = tw / tw.sum(dim=-1, keepdim=True)
        return _vllm_moe_seat_op(hidden_states, router_logits, ti, tw,
                                 self.idx)


class _SlabbedHeadMethod:
    """Stands in for the LM head's quant method: the engine computes
    logits through ``quant_method.apply``, never module forward."""

    def __init__(self, seams, orig):
        self.seams = seams
        self.orig = orig

    def apply(self, layer, x, bias=None):
        xb = x.to(torch.bfloat16)
        y = torch.cat([s(xb) for s in self.seams], dim=-1)
        if bias is not None:
            y = y + bias
        return y.to(x.dtype)

    def __getattr__(self, name):
        return getattr(self.orig, name)


def _is_projection(module) -> bool:
    w = getattr(module, "weight", None)
    return (isinstance(w, torch.Tensor) and w.dim() == 2
            and hasattr(module, "quant_method"))


def _expert_holder(module):
    for _, child in module.named_modules():
        if torch.is_tensor(getattr(child, "w13_weight", None)):
            return child
    return None


class _NoSeats:
    """The handle shape for a host where nothing could be seated.

    Every seat refusing is a normal outcome, not an error: the hub may be
    unreachable, this architecture may have no build, the memory budget
    may leave no room. What must not happen is the engine dying because
    its accelerator was absent — a server that fails to start is worse
    than one that starts unaccelerated. So the refusals are reported and
    the host is handed back untouched, with a handle of the same shape so
    callers need no special case.
    """

    def __init__(self, refused, reverts):
        self.notes = {"refused": refused, "head_slabs": 0, "seated": 0}
        self._reverts = list(reverts)

    def detach(self):
        for fn in reversed(self._reverts):
            fn()
        self._reverts.clear()

    def report(self):
        return {}

    def summary(self):
        return {"seams": 0, "guarded_calls": 0, "fallbacks": 0,
                "seams_fell_back": [], "seams_self_detached": [],
                "seams_never_called": [], "clean": True,
                "refused": len(self.notes["refused"])}


def attach_engine(model, *, seats=DENSE_SEAT_SUFFIXES, experts=True,
                  head=True, use_gemv=None, verbose=True, strict=False):
    """Seat a vLLM model: dense projections, expert banks, LM head.

    Call between weight load and the engine's first trace (see
    :func:`install_load_hook`). Returns the ``swap.attach`` handle;
    ``handle.detach()`` restores the module tree, the expert modules
    and the head's quant method.
    """
    if use_gemv is None:
        cc = torch.cuda.get_device_capability()
        use_gemv = cc >= (12, 0)   # the warp-split GEMV entry's own arch
    if not use_gemv:
        orig_init = _linear.LinearProjNvfp4Dynamic.__init__

        def _init(self, *a, **kw):
            orig_init(self, *a, **kw)
            self._gemv = None
        _linear.LinearProjNvfp4Dynamic.__init__ = _init

    swaps: dict[str, nn.Module] = {}
    reverts: list = []
    refused: list = []
    modules = dict(model.named_modules())

    # dense projections, smallest first: on tight cards early frees
    # make room for the big binds
    targets = [(n, m) for n, m in modules.items()
               if any(n.endswith(s) for s in seats) and _is_projection(m)]
    targets.sort(key=lambda t: t[1].weight.numel())
    for name, mod in targets:
        try:
            seam, _ = _linear.bind_proj_seam({"w": mod.weight.data})
            swaps[name] = _ProjSeat(seam)
        except Exception as e:
            refused.append((name, repr(e)[:120]))

    if experts:
        impl = (_experts_w4a4 if use_gemv else _experts_w4a16)
        for name, mod in modules.items():
            if not name.endswith("mlp.experts"):
                continue
            holder = _expert_holder(mod)
            if holder is None:
                continue
            try:
                seam, _ = impl.bind_experts_seam(
                    {"gate_up_proj": holder.w13_weight.data,
                     "down_proj": holder.w2_weight.data},
                    act_fn=torch.nn.functional.silu)
                top_k = (getattr(mod, "top_k", None)
                         or getattr(getattr(mod, "moe_config", None),
                                    "experts_per_token", None) or 8)
                renorm = getattr(mod, "renormalize", None)
                parent = modules[name.rsplit(".experts", 1)[0]]
                swaps[name] = _MoESeat(
                    seam, int(top_k),
                    True if renorm is None else bool(renorm),
                    getattr(parent, "shared_expert", None), mod)
            except Exception as e:
                refused.append((name, repr(e)[:120]))

    head_slabs = 0
    if head:
        lm = next((m for n, m in modules.items()
                   if n.endswith("lm_head")
                   and isinstance(getattr(m, "weight", None), torch.Tensor)),
                  None)
        if lm is not None:
            try:
                rows = lm.weight.shape[0]
                slab = -(-rows // 4) // 64 * 64
                seams = [
                    _linear.bind_proj_seam(
                        {"w": lm.weight.data[lo:lo + slab]})[0]
                    for lo in range(0, rows, slab)]
                orig_method = lm.quant_method
                lm.quant_method = _SlabbedHeadMethod(seams, orig_method)
                reverts.append(
                    lambda lm=lm, m=orig_method: setattr(
                        lm, "quant_method", m))
                head_slabs = len(seams)
            except Exception as e:
                refused.append(("lm_head", repr(e)[:120]))

    model.eval()
    if not swaps:
        if strict:
            raise RuntimeError(
                "refused: no seat could be bound on this host (%d refusals; "
                "first: %s). Pass strict=False to let the engine start "
                "unaccelerated." % (len(refused),
                                    refused[0][1] if refused else "none"))
        if verbose:
            print(f"[structures.vllm] 0 seats, {len(refused)} refused — "
                  f"host runs unmodified", flush=True)
            for name, why in refused[:3]:
                print(f"[structures.vllm]   {name}: {why}", flush=True)
        handle = _NoSeats(refused, reverts)
        handle.notes["refused"] = refused
        return handle
    handle = _swap.attach(model, swaps, revert=reverts)
    if verbose:
        print(f"[structures.vllm] {len(swaps)} seats "
              f"({head_slabs} head slabs), {len(refused)} refused",
              flush=True)
    handle.notes = {"refused": refused, "head_slabs": head_slabs}
    return handle


#: every handle :func:`install_load_hook` has seated, in order. A caller
#: that wants certainty rather than a log line asserts on this after the
#: engine is up: empty means the hook never fired.
_ATTACHED: list = []


def attached() -> list:
    """The handles seated so far. Empty after an engine came up means the
    patch never reached the process that loaded the model — see
    :func:`install_load_hook` on the start method."""
    return list(_ATTACHED)


def _patch_would_not_survive() -> str | None:
    """Why a patch made here would not exist in the worker process.

    The engine starts its worker with ``fork`` by default, which
    inherits this patch, and that is why the four-line integration
    works at all. But it switches to ``spawn`` under conditions the
    caller can walk into without noticing — a spawned worker re-imports
    the engine from scratch and the patch is simply not there.

    The failure is silent: patching succeeds here, the hook never fires,
    and the run comes out at baseline speed with no error anywhere. That
    is the one outcome this layer refuses to produce, so it is checked
    before the caller builds an engine rather than discovered afterwards
    from a missing log line.
    """
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") == "0":
        return None                     # the engine runs in this process
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn":
        return "VLLM_WORKER_MULTIPROC_METHOD is set to 'spawn'"
    if torch.cuda.is_initialized():
        return ("CUDA is already initialized in this process, and the "
                "engine forces 'spawn' when it is")
    return None


def install_load_hook(*, on_attached=None, allow_spawn=False,
                      **attach_kwargs):
    """Patch every importable vLLM model-runner so :func:`attach_engine`
    runs after weights load and before the engine's first trace. Set
    ``VLLM_DISABLE_COMPILE_CACHE=1``: the engine's compile cache key
    does not see the module tree, and a stale artifact resolves
    parameters that the seats replaced.

    Call this before touching CUDA. The engine forks its worker by
    default, which is what carries this patch into the process that
    loads the model, but it switches to spawning one the moment CUDA is
    already initialized here — and a spawned worker re-imports the
    engine without the patch. Nothing raises in that case: the seats
    simply never go in and the run comes out at baseline. So the
    condition is refused here instead, with the two ways out. Pass
    ``allow_spawn=True`` to proceed anyway.
    """
    import importlib
    import os

    # Find the runners before changing anything: "there is no host here"
    # is the more fundamental refusal, and a caller without vLLM should
    # hear that rather than a lecture about start methods. Nothing is
    # mutated until both questions have been answered.
    found = []
    for modname in ("vllm.v1.worker.gpu.model_runner",
                    "vllm.v1.worker.gpu_model_runner",
                    "vllm.v2.worker.gpu_model_runner"):
        try:
            module = importlib.import_module(modname)
        except ImportError:
            continue
        runner = getattr(module, "GPUModelRunner", None)
        if runner is None or not hasattr(runner, "load_model"):
            continue
        found.append((modname, runner))
    if not found:
        raise RuntimeError(
            "refused: no vLLM model runner found to hook; the engine "
            "layout is outside this adapter's profile")

    lost = None if allow_spawn else _patch_would_not_survive()
    if lost is not None:
        raise RuntimeError(
            "refused: this patch would not reach the process that loads "
            "the model — " + lost + ".\n"
            "The engine would start its worker with 'spawn', which "
            "re-imports it from scratch, so the seats would never be "
            "installed and the run would come out at baseline speed "
            "with nothing raised anywhere.\n"
            "Either call install_load_hook() before anything touches "
            "CUDA, or run the engine in this process with "
            "VLLM_ENABLE_V1_MULTIPROCESSING=0. Pass allow_spawn=True to "
            "proceed regardless.")

    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    patched = []
    for modname, runner in found:
        orig = runner.load_model

        def load_model(self, *a, __orig=orig, **kw):
            __orig(self, *a, **kw)
            handle = attach_engine(self.model, **attach_kwargs)
            _ATTACHED.append(handle)
            if on_attached is not None:
                on_attached(handle)
        runner.load_model = load_model
        patched.append(modname)
    return patched
