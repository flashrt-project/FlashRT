"""Negotiate one join: intersect what both sides offer, name what fails.

This generalises what the fp8 producer chain already does by hand — two
components agreeing on one attribute and binding atomically — to the
whole vocabulary. The parts that carried their weight there are kept:

- the intersection is computed before anything binds, so a join that
  cannot be agreed changes nothing;
- a refusal names the attribute that failed, not the structure, because
  "refused" must never read as "this structure cannot be joined";
- an attribute only one side declares stays unnegotiated and is reported
  as such, so a partial declaration is a partial gain rather than a
  silent override.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from .ports import Join, Port


class JoinRefused(ValueError):
    """The two ports have no common value for some attribute."""


def negotiate(
    producer: Port,
    consumer: Port,
    *,
    prefer: Mapping[str, Sequence[str]] | None = None,
    cost: Callable[[str, str], float] | None = None,
) -> Join:
    """Agree the attributes both ports declare.

    ``prefer`` overrides the producer's own preference order for an
    attribute — this is where a measured band belongs, so that a choice
    follows the shapes at hand rather than a fixed opinion baked into
    the port. ``cost`` picks by number instead when one is available;
    the lowest wins.
    """
    if producer.direction != "out" or consumer.direction != "in":
        raise JoinRefused(
            f"{producer.structure}.{producer.name} -> "
            f"{consumer.structure}.{consumer.name}: a join runs from an "
            "out port to an in port")

    prefer = dict(prefer or {})
    chosen: dict[str, str] = {}
    unconstrained: list[str] = []

    for attr in sorted(set(producer.offers) | set(consumer.offers)):
        out_side = producer.offers.get(attr)
        in_side = consumer.offers.get(attr)
        if out_side is None or in_side is None:
            # only one side has an opinion: leave the join as the host
            # already has it, and say so
            unconstrained.append(attr)
            continue
        common = [v for v in out_side if v in in_side]
        if not common:
            raise JoinRefused(
                f"{producer.structure}.{producer.name} -> "
                f"{consumer.structure}.{consumer.name}: no common {attr} "
                f"({list(out_side)} vs {list(in_side)}) — the join stays "
                "as the host has it; the structures themselves are fine")
        order = [v for v in prefer.get(attr, ()) if v in common] or common
        if cost is not None:
            chosen[attr] = min(order, key=lambda value: cost(attr, value))
        else:
            chosen[attr] = order[0]

    return Join(producer=producer, consumer=consumer, chosen=chosen,
                unconstrained=tuple(unconstrained))


def describe(joins: Sequence[Join]) -> str:
    """A readable account of what a set of joins agreed on."""
    lines = []
    for join in joins:
        lines.append(str(join))
        if join.unconstrained:
            lines.append(f"    unconstrained: {', '.join(join.unconstrained)}")
    return "\n".join(lines)
