"""The join vocabulary: what two adjacent structures have to agree on.

Every attribute below was extracted from a join that had already gone
wrong, and the incident is recorded with it. Nothing here is speculative
— an attribute nobody negotiates is meant to be deleted, not kept as
documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

# Attribute -> the values a port may declare, most-preferred first where
# order carries meaning. A port that declares nothing for an attribute
# leaves it unnegotiated, which is exactly today's behaviour.
ATTRIBUTES: dict[str, tuple[str, ...]] = {
    # Who holds the activation in what numeric type, and therefore who
    # pays for the conversion.
    #
    # Incident: an fp8 producer and the pack it feeds, bound
    # independently, can diverge — a pack expecting fp8 whose producer
    # did not bind leaves the host to insert a quantize, which the
    # compiler fuses into the upstream gated residual and turns into fp8
    # arithmetic the target cannot lower. Binding the pair atomically
    # fixed it. This is the one join attribute that was ever declared,
    # and the only one that stopped failing.
    "dtype": ("fp8_static", "bf16", "int8_static"),

    # How the tensor's axes are arranged when it crosses the join.
    #
    # Incident: the host lays q/k/v out as (B, H, S, D) because that is
    # what eager attention wants; the fused kernel wants (B, S, H, D)
    # and transposed back. Two transposes that cancel, plus the copies
    # that make each contiguous, per projection per layer per step —
    # and neither side could see the other, because each was correct on
    # its own.
    "layout": ("row_major", "bshd", "bhsd"),

    # Who allocates the tensor the consumer reads. ``alias`` is the
    # declarative form of what a hand-written runtime does when it hands
    # the next stage a pointer: the producer writes straight into the
    # region the consumer already owns.
    #
    # Incident: the packed projection writes its k/v into its own stash
    # buffers and the attention core then copies them into its packed
    # KV region — two buffers where one would do.
    "buffer": ("alias", "caller_provided", "fresh"),

    # Whether a sublayer result is still pending at the join, so the
    # consumer can absorb it instead of the host closing it first.
    #
    # Incident: the adaptive-norm kernel computes ``residual + x * gate``
    # before it norms. Bound at the norm boundary there is no residual to
    # hand it, so it was fed zeros and the host kept its own elementwise
    # add — measured +0.17 ms, refused. The same kernel inside a block
    # boundary replaces that add instead of adding to it, and turns
    # positive. Same kernel, opposite verdict, because the join was
    # declared at the wrong place.
    "carry": ("gated_residual", "none"),

    # How often the value on this join actually changes, as opposed to
    # how often it is read.
    #
    # Incident: a step's style is fixed for every producer on one
    # conditioning stream, but was materialised per call — 720 launches
    # for work that changes 10 times a tick.
    "cadence": ("per_observation", "per_step", "per_call"),

    # Whether the compiler may rewrite away the arrangement this join
    # depends on. Not a hardware or dataflow property: a contract with
    # the compiler underneath.
    #
    # Incident: a step-scoped shared buffer, written as plain tensor
    # work, was legally eliminated — the compiler saw a buffer whose
    # only consumers were slices of it and inlined the fill back into
    # each of them. Semantically identical, and the sharing was gone:
    # -0.164 ms instead of -0.419 ms, with the launch count unchanged.
    # A structure layer sitting on a compiler has to be able to say
    # which of its decisions may not be undone.
    "opacity": ("must_persist", "fusible"),
}


class PortError(ValueError):
    """A port declared something outside the vocabulary."""


@dataclass(frozen=True)
class Port:
    """One side of a join: what this structure can accept or emit.

    ``offers`` maps an attribute to the values this port supports, most
    preferred first. An attribute left out is not negotiated — the join
    keeps whatever the host already does, which is why declaring a port
    can only add.
    """

    structure: str
    name: str
    direction: str                       # "in" | "out"
    offers: Mapping[str, Sequence[str]] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("in", "out"):
            raise PortError(
                f"{self.structure}.{self.name}: direction must be in/out")
        for attr, values in self.offers.items():
            known = ATTRIBUTES.get(attr)
            if known is None:
                raise PortError(
                    f"{self.structure}.{self.name}: {attr!r} is not a join "
                    f"attribute. The vocabulary is closed on purpose — "
                    f"adding one means a new incident to record")
            unknown = [v for v in values if v not in known]
            if unknown:
                raise PortError(
                    f"{self.structure}.{self.name}: {attr}={unknown} not in "
                    f"{known}")
            if not values:
                raise PortError(
                    f"{self.structure}.{self.name}: {attr} offers nothing; "
                    "leave the attribute out to keep it unnegotiated")


@dataclass(frozen=True)
class Join:
    """A negotiated agreement between two ports."""

    producer: Port
    consumer: Port
    chosen: Mapping[str, str]
    unconstrained: Sequence[str] = ()     # declared by one side only
    note: str = ""

    def __str__(self) -> str:
        picks = ", ".join(f"{k}={v}" for k, v in sorted(self.chosen.items()))
        return (f"{self.producer.structure}.{self.producer.name} -> "
                f"{self.consumer.structure}.{self.consumer.name} [{picks}]")
