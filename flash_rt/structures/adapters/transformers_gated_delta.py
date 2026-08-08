"""Capability adapter for Transformers-style Gated Delta call slots."""

from __future__ import annotations

import torch

from ..impls.gated_delta_core import bind_gated_delta_core


def _compatible(module) -> bool:
    return (
        callable(getattr(module, "recurrent_gated_delta_rule", None))
        and callable(getattr(module, "chunk_gated_delta_rule", None))
        and getattr(module, "num_v_heads", None) in (32, 48)
        and getattr(module, "head_k_dim", None) == 128
        and getattr(module, "head_v_dim", None) == 128
    )


class _Recorder:
    def __init__(self, original, rows, phase):
        self.original = original
        self.rows = rows
        self.phase = phase

    def __call__(self, query, key, value, g, beta, *args, **kwargs):
        state = kwargs.get("initial_state")
        row = {
            "phase": self.phase,
            "query": query.detach(),
            "key": key.detach(),
            "value": value.detach(),
            "g": g.detach(),
            "beta": beta.detach(),
            "state": state.detach() if state is not None else None,
            "output_final_state": kwargs.get("output_final_state", False),
            "use_qk_l2norm": kwargs.get(
                "use_qk_l2norm_in_kernel", False),
        }
        if not self.rows:
            self.rows.append(row)
        else:
            expected = tuple(self.rows[0][name].shape
                             for name in ("query", "key", "value", "g", "beta"))
            got = tuple(row[name].shape
                        for name in ("query", "key", "value", "g", "beta"))
            if got != expected:
                raise ValueError(
                    "gated_delta_core: shape moved inside one host call")
        return self.original(query, key, value, g, beta, *args, **kwargs)


class _Route:
    def __init__(self, core, original):
        self.core = core
        self.original = original

    def __call__(self, query, key, value, g, beta, *args, **kwargs):
        supported_keys = {
            "initial_state", "output_final_state",
            "use_qk_l2norm_in_kernel", "chunk_size", "cu_seqlens",
        }
        unsupported = set(kwargs).difference(supported_keys)
        packed = kwargs.get("cu_seqlens") is not None
        if args or unsupported or packed or query.shape[1] != 1:
            guard = getattr(self.core, "_frt_guard", None)
            if guard is not None and not torch.compiler.is_compiling():
                reason = "unsupported packed or extended GDN call contract"
                guard.refuse(reason)
            return self.original(
                query, key, value, g, beta, *args, **kwargs)
        initial_state = kwargs.get("initial_state")
        output_final_state = bool(kwargs.get("output_final_state", False))
        use_norm = bool(kwargs.get("use_qk_l2norm_in_kernel", False))
        return self.core(
            query, key, value, g, beta, initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm=use_norm,
        )


class TransformersGatedDeltaAdapter:
    """Route recurrent/chunk callable slots without matching class names."""

    __name__ = "transformers_gated_delta"

    def __call__(self, model, forward):
        sites = [(path, module) for path, module in model.named_modules()
                 if _compatible(module)]
        if not sites:
            return None
        captures = [[] for _ in sites]
        originals = []
        for (_, module), rows in zip(sites, captures):
            recurrent = module.recurrent_gated_delta_rule
            chunk = module.chunk_gated_delta_rule
            originals.append((module, recurrent, chunk))
            module.recurrent_gated_delta_rule = _Recorder(
                recurrent, rows, "decode_recurrent")
            module.chunk_gated_delta_rule = _Recorder(
                chunk, rows, "sequence")
        try:
            with torch.no_grad():
                forward()
        finally:
            for module, recurrent, chunk in originals:
                module.recurrent_gated_delta_rule = recurrent
                module.chunk_gated_delta_rule = chunk

        if not any(captures):
            return None
        routes = []
        observed = {}
        for (path, module), rows, (_, recurrent, chunk) in zip(
                sites, captures, originals):
            if not rows:
                continue
            row = rows[0]
            if row["phase"] != "decode_recurrent":
                raise ValueError(
                    "gated_delta_core: Hub v3 explicit-state executable "
                    "covers recurrent decode only; a sequence-inout artifact "
                    "is required for prefill")
            if not all(row[name].is_contiguous()
                       for name in ("query", "key", "value")):
                raise ValueError(
                    "gated_delta_core: formal Hub v3 artifact requires "
                    "contiguous Q/K/V, but this host exposes split views; "
                    "a stride-aware recurrence artifact is required")
            # Each site owns its output/state scratch. Sharing one core across
            # equal signatures would make later layers overwrite live buffers
            # from earlier layers during compiled or captured execution.
            core = bind_gated_delta_core(row)
            routes.append((module, recurrent, chunk, core))
            observed[f"{path}.gated_delta_core"] = core

        def enable():
            for module, recurrent, chunk, core in routes:
                module.recurrent_gated_delta_rule = _Route(core, recurrent)
                module.chunk_gated_delta_rule = _Route(core, chunk)

        def disable():
            for module, recurrent, chunk, _ in routes:
                module.recurrent_gated_delta_rule = recurrent
                module.chunk_gated_delta_rule = chunk

        enable()
        return {
            "observed": observed,
            "revert": [disable],
            "toggle": (enable, disable),
        }
