"""Q/K norm + RoPE adapter for factored two-way attention hosts.

The host capability is two independent sibling-QKV projection groups over a
factored causal/full pack.  Each group is consumed by per-head RMSNorm and the
same pre-expanded rotate-half position table before a factored attention
processor.  This is the Cosmos/MoT form, but the adapter deliberately matches
those slots and dataflow rather than a model or class name.
"""

from __future__ import annotations

import types

from ..guard import GuardRefused
from ..impls.qk_norm_rope import bind_per_head_gqa_qk_norm_rope
from ..impls.qkv_pack.fp8_static import PackedLinear, StashReader


_PATHS = (
    (
        ("to_q", "to_k", "to_v"),
        ("norm_q", "norm_k"),
        "causal_seq",
    ),
    (
        ("add_q_proj", "add_k_proj", "add_v_proj"),
        ("norm_added_q", "norm_added_k"),
        "full_only_seq",
    ),
)


def _pack_parts(plan, path: str, attrs: tuple[str, str, str]):
    head = plan.swaps.get(f"{path}.{attrs[0]}")
    key = plan.swaps.get(f"{path}.{attrs[1]}")
    value = plan.swaps.get(f"{path}.{attrs[2]}")
    if not (
        isinstance(head, PackedLinear)
        and isinstance(key, StashReader)
        and isinstance(value, StashReader)
        and key._packed[0] is head
        and value._packed[0] is head
    ):
        return None
    return head


def _epsilon(norm) -> float | None:
    value = getattr(norm, "variance_epsilon", getattr(norm, "eps", None))
    return None if value is None else float(value)


class FactoredQkNormRopeAdapter:
    """Compose two packed QKV groups with per-head Q/K norm and RoPE."""

    __name__ = "factored_qk_norm_rope"

    def __call__(self, model, plan):
        routes = []
        observed = {}
        refused = []

        for path, module in model.named_modules():
            packs = [_pack_parts(plan, path, attrs) for attrs, _, _ in _PATHS]
            if not any(pack is not None for pack in packs):
                continue
            site = f"{path}::factored_qk_norm_rope"

            def refuse(reason: str) -> None:
                refused.append((site, f"qk_norm_rope refused: {reason}"))

            if not all(pack is not None for pack in packs):
                refuse("both causal and full QKV groups must be packed")
                continue
            if module.training:
                refuse("training/dropout form is outside the inference seam")
                continue
            if not all(
                hasattr(module, attr)
                for attr in (
                    "head_dim",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "dispatch_attention_fn",
                    "to_out",
                    "to_add_out",
                )
            ):
                refuse("host lacks the complete factored two-way slots")
                continue
            if getattr(module, "cp_mesh", None) is not None:
                refuse("context-parallel packs are outside this single-device seam")
                continue

            head_dim = int(module.head_dim)
            q_heads = int(module.num_attention_heads)
            kv_heads = int(module.num_key_value_heads)
            if head_dim != 128:
                refuse("current Hub entry requires head_dim=128")
                continue

            bounds = []
            bad = None
            for pack, (_, norms, key) in zip(packs, _PATHS):
                q_norm = getattr(module, norms[0], None)
                k_norm = getattr(module, norms[1], None)
                q_weight = getattr(q_norm, "weight", None)
                k_weight = getattr(k_norm, "weight", None)
                eps = _epsilon(q_norm)
                if q_weight is None or k_weight is None or eps is None:
                    bad = f"{key} Q/K norm weights or epsilon are absent"
                    break
                expected = (q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim)
                if tuple(pack.splits[:3]) != expected:
                    bad = f"{key} packed widths {tuple(pack.splits[:3])} != {expected}"
                    break
                try:
                    bound = bind_per_head_gqa_qk_norm_rope(
                        q_weight,
                        k_weight,
                        row_capacity=pack.rows,
                        q_heads=q_heads,
                        kv_heads=kv_heads,
                        head_dim=head_dim,
                        eps=eps,
                    )
                except (ValueError, RuntimeError) as exc:
                    bad = str(exc)
                    break
                bounds.append(bound)
            if bad is not None:
                refuse(bad)
                continue

            original = module.forward
            had_instance_forward = "forward" in module.__dict__
            causal_pack, full_pack = packs
            causal_bound, full_bound = bounds

            def routed(
                self,
                pack,
                attention_mask,
                packed_position_embeddings,
                dual_kv_cache=None,
                natten_metadata=None,
                *,
                und_packed=causal_pack,
                gen_packed=full_pack,
                und_bound=causal_bound,
                gen_bound=full_bound,
            ):
                del attention_mask
                if dual_kv_cache is not None:
                    raise GuardRefused(
                        "qk_norm_rope: factored cache mutation is outside the bound seam"
                    )
                if natten_metadata is not None:
                    raise GuardRefused(
                        "qk_norm_rope: neighborhood attention is outside the bound seam"
                    )
                if not isinstance(pack, dict) or not all(
                    key in pack for key in ("causal_seq", "full_only_seq")
                ):
                    raise GuardRefused(
                        "qk_norm_rope: expected a causal/full factored pack"
                    )
                try:
                    cos_pack, sin_pack = packed_position_embeddings
                    und_cos = cos_pack["causal_seq"]
                    und_sin = sin_pack["causal_seq"]
                    gen_cos = cos_pack["full_only_seq"]
                    gen_sin = sin_pack["full_only_seq"]
                except (KeyError, TypeError, ValueError) as exc:
                    raise GuardRefused(
                        "qk_norm_rope: position tables do not share the factored layout"
                    ) from exc

                q_und, k_und, v_und = und_bound(
                    und_packed.joint(pack["causal_seq"]).unsqueeze(0),
                    und_cos.unsqueeze(0),
                    und_sin.unsqueeze(0),
                )
                q_gen, k_gen, v_gen = gen_bound(
                    gen_packed.joint(pack["full_only_seq"]).unsqueeze(0),
                    gen_cos.unsqueeze(0),
                    gen_sin.unsqueeze(0),
                )
                q_und, k_und, v_und = q_und[0], k_und[0], v_und[0]
                q_gen, k_gen, v_gen = q_gen[0], k_gen[0], v_gen[0]

                if bool(getattr(getattr(self, "config", None), "freeze_und", False)):
                    q_und = q_und.detach()
                    k_und = k_und.detach()
                    v_und = v_und.detach()

                query = dict(pack)
                key = dict(pack)
                value = dict(pack)
                query["causal_seq"], query["full_only_seq"] = q_und, q_gen
                key["causal_seq"], key["full_only_seq"] = k_und, k_gen
                value["causal_seq"], value["full_only_seq"] = v_und, v_gen
                attended = self.dispatch_attention_fn(query, key, value)

                out = dict(pack)
                out["causal_seq"] = self.to_out(attended["causal_seq"])
                out["full_only_seq"] = self.to_add_out(attended["full_only_seq"])
                return out

            routes.append(
                (
                    module,
                    packs,
                    types.MethodType(routed, module),
                    original,
                    had_instance_forward,
                )
            )
            observed[f"{path}.causal::per_head_qk_norm_rope"] = causal_bound
            observed[f"{path}.full::per_head_qk_norm_rope"] = full_bound

        if not routes:
            return {"refused": refused} if refused else None

        def enable() -> None:
            for module, packs, routed, _, _ in routes:
                for pack in packs:
                    pack.enable_joint(3)
                module.forward = routed

        def disable() -> None:
            for module, packs, _, original, _ in routes:
                module.forward = original
                for pack in packs:
                    pack.disable_joint()

        def revert() -> None:
            for module, packs, _, original, had_instance_forward in routes:
                for pack in packs:
                    pack.disable_joint()
                if had_instance_forward:
                    module.forward = original
                elif "forward" in module.__dict__:
                    del module.forward

        enable()
        return {
            "observed": observed,
            "revert": [revert],
            "toggle": (enable, disable),
            "refused": refused,
        }
