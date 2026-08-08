"""Adapter for hosts exposing a factored two-way attention processor."""

from __future__ import annotations

from ..impls.attention_core import bind_two_way_attention


class FactoredTwoWayAttentionAdapter:
    """Route compatible factored attention processors through Hub FA2."""

    __name__ = "factored_two_way_attention"

    def __call__(self, model, forward, *, prefix_cadence: bool = False):
        del prefix_cadence
        sites = []
        for path, module in model.named_modules():
            processor = getattr(module, "dispatch_attention_fn", None)
            if callable(processor):
                sites.append((path, module, processor))
        if not sites:
            return None

        captures: list[list[dict]] = [[] for _ in sites]
        for index, (_, module, original) in enumerate(sites):
            def record(query, key, value, *, i=index, fn=original):
                captures[i].append(
                    {"query": query, "key": key, "value": value})
                return fn(query, key, value)

            module.dispatch_attention_fn = record
        try:
            with __import__("torch").no_grad():
                forward()
        finally:
            for _, module, original in sites:
                module.dispatch_attention_fn = original

        if not any(captures):
            return None
        if any(not rows for rows in captures):
            raise ValueError(
                "attention_core two_way: only some discovered processors "
                "were called")

        cores = []
        routes = []
        for (path, module, original), rows in zip(sites, captures):
            first = rows[0]

            def shape(pack):
                return (
                    tuple(pack["causal_seq"].shape),
                    tuple(pack["full_only_seq"].shape),
                )

            expected = (
                shape(first["query"]),
                shape(first["key"]),
                shape(first["value"]),
            )
            for row in rows[1:]:
                got = (
                    shape(row["query"]),
                    shape(row["key"]),
                    shape(row["value"]),
                )
                if got != expected:
                    raise ValueError(
                        "attention_core two_way: processor shapes move "
                        f"within one calibration call: {expected} -> {got}")
            core = bind_two_way_attention(first)

            def routed(query, key, value, *, bound=core):
                return bound(query, key, value)

            cores.append((path, core))
            routes.append((module, original, routed))

        def enable() -> None:
            for module, _, routed in routes:
                module.dispatch_attention_fn = routed

        def disable() -> None:
            for module, original, _ in routes:
                module.dispatch_attention_fn = original

        def revert() -> None:
            disable()

        enable()
        observed = {
            f"{path}.dispatch_attention_fn::fa2_core": core
            for path, core in cores
        }
        return {}, None, {
            "revert": [revert],
            "observed": observed,
            "toggle": (enable, disable),
        }
