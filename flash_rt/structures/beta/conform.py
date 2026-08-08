"""Check the declarations against the impls they claim to describe.

A spec file that nobody checks drifts away from the code and becomes
worse than no spec, because it is then confidently wrong. Writing these
ports down already corrected one of my own beliefs about what
``attention_core`` can emit; that only stays true if something keeps
checking.

This is deliberately shallow. It asks the impls what they support and
compares against what the ports claim — it does not run a model. A
declaration that cannot be checked this way is a declaration that should
not be made.
"""

from __future__ import annotations

from typing import Sequence

from .joins import DECLARED
from .ports import Port


def _offers(port: Port, attr: str) -> tuple[str, ...]:
    return tuple(port.offers.get(attr, ()))


def check() -> list[str]:
    """Return one line per disagreement; empty means the ports hold."""
    problems: list[str] = []

    from ..impls.attention_core import fa2_seqused
    from ..impls.linear_proj import fp8_static as proj
    from ..impls.qkv_pack import PackedLinear

    core_cls = fa2_seqused.PackedKVAttention

    # buffer=alias is only real if both sides expose the entries that
    # make it real, and if the consumer can be told to stop copying
    for name, (out_port, in_port) in DECLARED.items():
        if "alias" not in _offers(out_port, "buffer"):
            continue
        if out_port.structure == "qkv_pack" and not hasattr(
                PackedLinear, "alias_stash"):
            problems.append(
                f"{name}: qkv_pack declares buffer=alias but has no "
                "alias_stash entry")
        if in_port.structure == "attention_core":
            for entry in ("alias_suffix", "forward_suffix"):
                if not hasattr(core_cls, entry):
                    problems.append(
                        f"{name}: attention_core declares buffer=alias "
                        f"but has no {entry} entry")

    # layout=bshd on the core's kv port is what forward_suffix takes; the
    # host-layout entry is a different port and must not be confused
    kv_in = DECLARED["qkv_pack->attention_core"][1]
    if _offers(kv_in, "layout") != ("bshd",):
        problems.append(
            "qkv_pack->attention_core: the core's kv port takes the "
            "kernel layout only; forward() is the host-layout entry and "
            "belongs to a different port")

    # dtype=fp8_static has to be an entry the consumer really has
    for name, (_, in_port) in DECLARED.items():
        if "fp8_static" not in _offers(in_port, "dtype"):
            continue
        if in_port.structure == "linear_proj":
            if "fp8_in" not in proj._BAND:
                problems.append(
                    f"{name}: linear_proj declares dtype=fp8_static but "
                    "has no fp8_in form")

    # opacity=must_persist is a claim about a compiler contract, so it
    # has to point at something the compiler cannot see through
    style_out = DECLARED["style_broker->producer"][0]
    if "must_persist" in _offers(style_out, "opacity"):
        import torch
        if not hasattr(torch.ops, "flash_rt_structures") or not hasattr(
                torch.ops.flash_rt_structures, "style_broadcast"):
            problems.append(
                "style_broker->producer: opacity=must_persist is claimed "
                "but the fill is not behind an opaque op, so the "
                "compiler may inline it back into each reader")

    return problems


def report(problems: Sequence[str] | None = None) -> str:
    problems = check() if problems is None else problems
    if not problems:
        return (f"beta: {len(DECLARED)} declared join(s), all consistent "
                "with the impls")
    return "beta: declarations disagree with the impls\n  " + "\n  ".join(
        problems)
