"""Runtime seam contract: check it, fall back, and leave a receipt.

A bound structure is calibrated for one execution form — a device, an
input dtype, a width, and for the implementations that preallocate, a row
count. Hand it something else and the honest outcomes are two: refuse, or
run the host's own module instead. The outcome this module exists to
prevent is the third one, where the kernel accepts a buffer it should not
have and the host keeps going with a wrong answer.

Falling back is allowed. Falling back *quietly* is not, because a seam
that always falls back reads exactly like a seam that works: the plan
still lists it, the receipt still counts it, and the latency it was
supposed to buy is simply absent. So every fallback is counted, the first
one per seam says so out loud, and a seam that never stops falling back
takes itself out of the model rather than staying on as a lie.

Three layers, one per timescale:

  per call      the contract is checked; a violation runs the host module
                and increments the seam's ledger
  first time    that seam warns once, naming the path and the reason
  persistent    after ``SELF_DETACH_AFTER`` consecutive fallbacks the seam
                restores the host module permanently and says so

``attach`` collects every guard it swaps in, so ``handle.report()`` is the
one place to ask what actually ran. Tests assert on it: a probe that does
not check its ledger has not checked that the thing it measured was on.

The check runs in Python, so it costs a call in an eager host and nothing
in a captured one — a graph replays the kernel it traced and never
re-enters this code. That is the right way round: the eager host is the
one whose shapes and devices can still change under it.
"""

from __future__ import annotations

import threading
import warnings
from typing import Any, Callable, Iterable

import torch

GUARD_ATTR = "_frt_guard"
SELF_DETACH_AFTER = 32

#: returned by :meth:`GuardedSeam._frt_admit` when the call may proceed
PROCEED = object()

_FP8 = torch.float8_e4m3fn
#: what a bf16-entry implementation will cast for itself. FP8 is
#: deliberately absent: an fp8 tensor handed to a bf16 entry is a
#: negotiation that did not happen, not an input to convert.
CAST_OK = frozenset({torch.bfloat16, torch.float16, torch.float32})
#: what an fp8-entry implementation requires, exactly
FP8_ONLY = frozenset({_FP8})


class GuardRefused(RuntimeError):
    """A seam was called outside the form it was bound for."""


def _concrete(device: torch.device | str) -> torch.device:
    """Resolve a device to the indexed form a tensor will report.

    ``torch.device("cuda")`` carries no index while every tensor on it
    reports ``cuda:0``, so an index-less contract would refuse every call
    it was meant to admit — a guard that fails closed on correct input is
    worse than no guard, because it reverts a working seam and blames the
    input. Normalising once at bind time keeps the per-call test an exact
    comparison.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


class SeamGuard:
    """One seam's runtime contract, its ledger, and its own exit.

    Created at bind time by the implementation, which knows the form it
    was calibrated for. Given a site by :func:`~flash_rt.structures.swap.attach`,
    which knows where in the model it ended up.
    """

    __slots__ = ("dtypes", "device", "k", "rows", "row_capacity", "kind",
                 "can_fallback", "calls", "fallbacks", "consecutive",
                 "last_reason", "detached", "site", "mode", "notes",
                 "thread", "_restore", "_warned")

    def __init__(self, *, kind: str, dtypes: Iterable[torch.dtype] | None,
                 device: torch.device, k: int | None, rows: int | None,
                 row_capacity: int | None, can_fallback: bool) -> None:
        self.kind = kind
        #: structure-specific counters an implementation keeps about
        #: itself, reported alongside the contract ones
        self.notes: dict[str, Any] = {}
        self.dtypes = None if dtypes is None else frozenset(dtypes)
        self.device = _concrete(device)
        self.k = k
        self.rows = rows
        self.row_capacity = row_capacity
        self.can_fallback = can_fallback
        self.calls = 0
        self.fallbacks = 0
        self.consecutive = 0
        self.last_reason: str | None = None
        self.detached = False
        self.site: str | None = None
        self.mode = "fallback"
        #: the thread that first ran this seam. These implementations keep
        #: preallocated stash, quantize and attention scratch buffers whose
        #: whole economy is that one call consumes a result before the next
        #: one writes it; a second thread in the same seam interleaves the
        #: writes and the corruption is silent and shape-correct. Single
        #: stream per attachment is the contract, and this is where it is
        #: enforced rather than assumed.
        self.thread: int | None = None
        self._restore: Callable[[], None] | None = None
        self._warned: set[str] = set()

    # ---- site binding (attach time) --------------------------------

    def bind_site(self, site: str, *, restore: Callable[[], None] | None,
                  mode: str) -> None:
        self.site = site
        self._restore = restore
        self.mode = mode

    def release_site(self) -> None:
        self.site = None
        self._restore = None

    # ---- per call --------------------------------------------------

    def admit(self, x: torch.Tensor) -> str | None:
        """Return ``None`` to proceed, else why this call cannot.

        The fast path is the whole point: one dtype test, one width test,
        one device test, and a row test only where the implementation
        preallocated for a fixed row count.
        """
        self.calls += 1
        if self.dtypes is not None and x.dtype not in self.dtypes:
            return f"input dtype {x.dtype} (bound for {self._dtype_note()})"
        if self.k is not None and x.shape[-1] != self.k:
            return f"input width {x.shape[-1]} (bound for {self.k})"
        if x.device != self.device:
            return f"input on {x.device} (bound on {self.device})"
        if self.rows is not None:
            rows = x.numel() // x.shape[-1]
            if rows != self.rows:
                return f"{rows} row(s) (bound for {self.rows})"
        if self.row_capacity is not None:
            rows = x.numel() // x.shape[-1]
            if rows > self.row_capacity:
                return (f"{rows} row(s) (buffer capacity "
                        f"{self.row_capacity})")
        tid = threading.get_ident()
        if self.thread is None:
            self.thread = tid
        elif tid != self.thread:
            return ("called from a second thread; this seam's stash and "
                    "scratch buffers are shared across its calls, so "
                    "concurrent use would interleave writes into them")
        self.consecutive = 0
        return None

    def _dtype_note(self) -> str:
        if self.dtypes == FP8_ONLY:
            return "fp8 from a producer seam"
        return "/".join(sorted(str(d).replace("torch.", "")
                               for d in self.dtypes or ()))

    def refuse(self, reason: str) -> None:
        """Record a fallback; speak up the first time and at the end."""
        where = self.site or f"<unattached {self.kind}>"
        if self.mode == "raise":
            # before counting: in strict mode nothing fell back, the call
            # was refused, and a ledger claiming otherwise would be one
            # more thing that says something that did not happen
            self.last_reason = reason
            raise GuardRefused(f"{where}: {reason}")
        self.fallbacks += 1
        self.consecutive += 1
        self.last_reason = reason
        if reason not in self._warned:
            self._warned.add(reason)
            warnings.warn(
                f"structures: {where} fell back to the host module — "
                f"{reason}. Further fallbacks of this kind are counted in "
                f"handle.report() and not repeated here.",
                RuntimeWarning, stacklevel=4)
        if (self.consecutive >= SELF_DETACH_AFTER and not self.detached):
            self._self_detach()

    def _self_detach(self) -> None:
        """Stop pretending: put the host module back for good."""
        self.detached = True
        where = self.site or f"<unattached {self.kind}>"
        if self._restore is None:
            warnings.warn(
                f"structures: {where} has fallen back "
                f"{self.consecutive} times in a row and cannot restore "
                f"itself (it is held inside another structure, not swapped "
                f"in at a path). It is costing a check per call and buying "
                f"nothing — detach the attachment.",
                RuntimeWarning, stacklevel=5)
            return
        self._restore()
        warnings.warn(
            f"structures: {where} fell back {self.consecutive} times in a "
            f"row ({self.last_reason}) — the host module has been restored "
            f"at that path permanently. The rest of the attachment is "
            f"unaffected.",
            RuntimeWarning, stacklevel=5)

    # ---- receipt ---------------------------------------------------

    def entry(self) -> dict[str, Any]:
        entry = {"kind": self.kind, "calls": self.calls,
                 "fallbacks": self.fallbacks, "detached": self.detached,
                 "last_reason": self.last_reason,
                 "form": {"dtype": self._dtype_note(), "k": self.k,
                          "rows": self.rows,
                          "row_capacity": self.row_capacity,
                          "device": str(self.device)}}
        if self.notes:
            entry["notes"] = dict(self.notes)
        return entry


class GuardedSeam:
    """Mixin: contract check, host fallback, and lifecycle refusals.

    Mixed in *before* ``torch.nn.Module`` so its ``_apply`` and
    ``state_dict`` overrides win. An implementation arms itself at the end
    of ``__init__`` with :meth:`_frt_arm` and opens its ``forward`` with
    :meth:`_frt_admit`.
    """

    #: attribute holding the retained host module, if any
    _frt_host_attr: str | None = None
    #: whether calling that module with this ``forward``'s own arguments
    #: reproduces the host's behaviour. False where the replacement's
    #: boundary is not the retained module's boundary (a composed block,
    #: an attention core called by an adapter): those refuse instead.
    _frt_can_fallback: bool = False
    #: class default so attribute lookup never reaches a host-forwarding
    #: ``__getattr__`` for an implementation that has not armed yet
    _frt_guard: SeamGuard | None = None

    def _frt_arm(self, *, dtypes: Iterable[torch.dtype] | None,
                 device: torch.device, k: int | None = None,
                 rows: int | None = None,
                 row_capacity: int | None = None) -> SeamGuard:
        """Declare the form this instance was calibrated for."""
        if rows is not None and row_capacity is not None:
            raise ValueError("rows and row_capacity are mutually exclusive")
        guard = SeamGuard(
            kind=type(self).__name__, dtypes=dtypes, device=device, k=k,
            rows=rows, row_capacity=row_capacity,
            can_fallback=self._frt_can_fallback)
        object.__setattr__(self, GUARD_ATTR, guard)
        return guard

    def _frt_touch(self) -> None:
        """Count a call whose form another guard already checked.

        Two cases need this. A replacement that ignores its inputs (a
        frozen buffer on a slower cadence) has no contract to test but
        still has to be countable. And a structure reached through a wider
        entry than its own ``forward`` — a producer that a composed block
        calls directly — is checked by the block's contract and would
        otherwise report zero calls, which reads as "never ran" when it
        ran every time.
        """
        guard = self._frt_guard
        if guard is not None and not torch.compiler.is_compiling():
            guard.calls += 1

    def _frt_host(self) -> torch.nn.Module | None:
        attr = self._frt_host_attr
        if attr is None:
            return None
        # straight out of _modules: going through getattr would hit the
        # host-forwarding __getattr__ these implementations define
        return self.__dict__.get("_modules", {}).get(attr)

    def _frt_admit(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """``PROCEED``, or the host's own result for this call.

        Raises :class:`GuardRefused` when there is no host module to fall
        back to, or when the attachment asked for strict mode.

        Steps aside entirely while a compiler is tracing, for two reasons
        that point the same way. It has to: the ledger's counter is a
        Python-visible side effect in a hot forward, and dynamo either
        breaks the graph around it or re-derives the region's boundaries
        around it — the surrounding compiled region is worth more than the
        count. And it may: dynamo installs its own guards on exactly the
        dtype, shape and device this contract tests, so a traced region
        handed a different form recompiles rather than running the wrong
        kernel. Inside a compiled region this check is redundant; in an
        eager host it is the only thing there.
        """
        guard = self._frt_guard
        if guard is None or torch.compiler.is_compiling():
            return PROCEED
        reason = guard.admit(x)
        if reason is None:
            return PROCEED
        host = self._frt_host()
        if host is None or not guard.can_fallback:
            raise GuardRefused(
                f"{guard.site or guard.kind}: {reason}; this structure has "
                "no equivalent host module to fall back to")
        if any(p.is_meta or p.numel() == 0
               for p in host.parameters()) and not \
                getattr(host, "_frt_tickets", None):
            raise GuardRefused(
                f"{guard.site or guard.kind}: {reason}; the host module's "
                "weights were consumed and cannot be restored")
        if getattr(host, "_frt_tickets", None):
            # the host's weights were consumed (their truth lives in the
            # weight store); a live fallback restores them once — slower
            # than a resident copy, never wrong — and the ledger says so
            from .storage import restore_for_fallback
            restore_for_fallback(host)
            guard.notes["restored_for_fallback"] = (
                guard.notes.get("restored_for_fallback", 0) + 1)
        guard.refuse(reason)
        return self._frt_host_call(host, x, *args, **kwargs)

    def _frt_host_call(self, host: torch.nn.Module, x: torch.Tensor,
                       *args: Any, **kwargs: Any) -> Any:
        """How to reproduce this seam with the host module. Overridable."""
        return host(x, *args, **kwargs)

    # ---- lifecycle refusals ---------------------------------------

    def _apply(self, *args: Any, **kwargs: Any):
        """Refuse device/dtype migration while swapped into a model.

        The packed weights and static scales of these implementations are
        derived tensors held outside the parameter system, so a migration
        would move the host's copy and leave the kernel's behind. Before
        attach and after detach this is an ordinary module and migrates
        normally.
        """
        guard = self._frt_guard
        if guard is not None and guard.site is not None:
            raise GuardRefused(
                f"{guard.site}: cannot migrate device or dtype while "
                "structures are attached — detach() first, migrate, then "
                "attach again (the scales are calibrated per form and do "
                "not survive a dtype change anyway)")
        return super()._apply(*args, **kwargs)

    def state_dict(self, *args: Any, destination: Any = None,
                   prefix: str = "", keep_vars: bool = False):
        """Emit the host module's own state at this path.

        A checkpoint should not change shape because an optimisation is
        attached: the packed and quantised tensors here are derived from
        the host weights and belong to the bind, not to the model's
        state. Delegating keeps the schema and the values identical to the
        unattached model, so saving while attached is safe.
        """
        host = self._frt_host()
        if host is None or args:
            return super().state_dict(*args, destination=destination,
                                      prefix=prefix, keep_vars=keep_vars)
        return host.state_dict(destination=destination, prefix=prefix,
                               keep_vars=keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Refuse loading into a swapped model rather than half-doing it.

        Loading would have to re-derive every packed weight and re-run
        calibration; silently loading the host's copy and leaving the
        kernel's stale is the failure this refuses. ``detach()`` restores
        the real modules, so the honest order is detach, load, attach.
        """
        guard = self._frt_guard
        if guard is not None and guard.site is not None:
            error_msgs.append(
                f"{guard.site}: cannot load_state_dict while structures "
                "are attached — detach(), load, then attach again so the "
                "packed weights are rebuilt from the loaded ones")
            return
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)


def collect(module: torch.nn.Module) -> list[tuple[str, SeamGuard]]:
    """Every guard at or under ``module``, by relative module name."""
    found = []
    for name, child in module.named_modules():
        guard = child.__dict__.get(GUARD_ATTR)
        if isinstance(guard, SeamGuard):
            found.append((name, guard))
    return found
