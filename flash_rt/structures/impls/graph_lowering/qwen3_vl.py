"""Whole-graph shape lowering for the Qwen3-VL backbone family.

The GR00T N1.7 hosts (official Isaac-GR00T and the LeRobot port) both
run a Qwen3-VL vision-language backbone that recomputes, on every call,
quantities that depend only on the request's shape and token placement:
multimodal position ids, rope tables, vision patch routing, sequence
cumsums, placeholder masks. Several of those recomputations synchronize
(``.item()`` / ``.tolist()`` / mask ``.all()``), which CUDA graph
capture forbids. For one fixed observation shape they are constants, so
this adapter records one real request, computes each constant once, and
pins the handful of host functions that would otherwise recompute them.

Two transformers generations of the vision contract are served, probed
by class presence rather than version: the older visual tower returns
``(merged, deepstack_list)``; the newer one wraps the same tensors in
``BaseModelOutputWithDeepstackFeatures`` and splits pooled embeddings
inside ``get_image_features``. Hosts that wrap the backbone in their
own per-call glue (a legacy position-id injector) are pinned through a
capability probe on the wrapper, never through a host table.

Every pin is a shape-derived constant of one fixed request; image and
state *values* flow through untouched, and ``undo()`` restores every
patched attribute.
"""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

from .protocol import GraphLowering, GraphLoweringRefused


def _find_qwen3_vl(model: Any):
    """The first Qwen3-VL-shaped conditional-generation module, or None."""
    for _, module in model.named_modules():
        inner = getattr(module, "model", None)
        visual = getattr(inner, "visual", None)
        if (visual is not None
                and hasattr(visual, "fast_pos_embed_interpolate")
                and hasattr(visual, "deepstack_visual_indexes")
                and hasattr(inner, "language_model")
                and hasattr(module, "lm_head")
                and getattr(getattr(module, "config", None),
                            "image_token_id", None) is not None):
            return module
    return None


def _find_glue_owner(model: Any, qwen: Any):
    """A host wrapper owning ``qwen`` with per-call position-id glue."""
    for _, module in model.named_modules():
        if (getattr(module, "model", None) is qwen
                and hasattr(module, "_ensure_legacy_qwen3_position_ids")):
            return module
    return None


def _first_output_tensor(value):
    """One deterministic tensor from a pipeline output, for proofs."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in sorted(value):
            found = _first_output_tensor(value[key])
            if found is not None:
                return found
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_output_tensor(item)
            if found is not None:
                return found
    return None


class Qwen3VLGraphLoweringAdapter:
    """Pin the Qwen3-VL backbone's shape glue for one fixed request."""

    family = "qwen3_vl_backbone"

    def lower(self, model: Any,
              forward: Callable[[], Any]) -> GraphLowering | None:
        qwen = _find_qwen3_vl(model)
        if qwen is None:
            return None

        # ---- record one real request ---------------------------------
        request: dict[str, torch.Tensor] = {}

        def grab(_module, args, kwargs):
            for key in ("input_ids", "attention_mask", "image_grid_thw"):
                value = kwargs.get(key)
                if value is None and args:
                    break
                if value is not None and key not in request:
                    request[key] = value
            return None

        handle = qwen.register_forward_pre_hook(grab, with_kwargs=True)
        try:
            # the recording pass is calibration-class work: it runs with
            # observation hooks on host modules, and a compiled callable
            # must execute eagerly here — otherwise its first trace
            # happens with the recording hooks inside the graph
            runner = forward
            dynamo = getattr(torch, "_dynamo", None)
            if dynamo is not None and hasattr(dynamo, "disable"):
                runner = dynamo.disable(forward)
            with torch.inference_mode():
                runner()
        finally:
            handle.remove()
        missing = [k for k in ("input_ids", "attention_mask",
                               "image_grid_thw") if k not in request]
        if missing:
            raise GraphLoweringRefused(
                f"{self.family}: could not observe {missing} on the "
                "recorded request — the host calls its backbone in a "
                "form this family does not recognize")

        input_ids = request["input_ids"]
        attention_mask = request["attention_mask"]
        grid_thw = request["image_grid_thw"]
        if not bool(torch.all(attention_mask == 1).item()):
            raise GraphLoweringRefused(
                f"{self.family}: fixed-shape capture requires an "
                "all-one attention mask")

        base = qwen.model
        visual = base.visual

        # ---- the constants of this request ---------------------------
        import importlib
        modeling = importlib.import_module(type(visual).__module__)
        output_cls = getattr(modeling,
                             "BaseModelOutputWithDeepstackFeatures", None)

        pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
        rotary = visual.rot_pos_emb(grid_thw)
        rotary_pair = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary_pair.cos(), rotary_pair.sin())
        fixed_cu_seqlens = torch.nn.functional.pad(
            torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                    grid_thw[:, 0]).cumsum(
                dim=0, dtype=torch.int32), (1, 0), value=0)
        vision_lengths = tuple(
            int(v) for v in
            (fixed_cu_seqlens[1:] - fixed_cu_seqlens[:-1]).cpu().tolist())
        split_sizes = tuple(int(v) for v in (
            grid_thw.prod(-1)
            // visual.spatial_merge_size ** 2).cpu().tolist())
        image_mask = (input_ids
                      == qwen.config.image_token_id).unsqueeze(-1)
        video_mask = (input_ids
                      == qwen.config.video_token_id).unsqueeze(-1)
        visual_indices = (input_ids.reshape(-1)
                          == qwen.config.image_token_id).nonzero().flatten()
        try:
            fixed_position_ids, fixed_rope_deltas = base.get_rope_index(
                input_ids, grid_thw, None, attention_mask=attention_mask)
            have_rope_index = True
        except (TypeError, IndexError):
            # newer transformers changed this helper's signature; hosts
            # that inject position_ids never call it — nothing to pin
            have_rope_index = False

        glue_owner = _find_glue_owner(model, qwen)
        glue_position_ids = None
        if glue_owner is not None:
            probe = {"input_ids": input_ids,
                     "attention_mask": attention_mask,
                     "image_grid_thw": grid_thw}
            if hasattr(glue_owner, "_ensure_mm_token_type_ids"):
                glue_owner._ensure_mm_token_type_ids(probe)
            glue_owner._ensure_legacy_qwen3_position_ids(probe)
            glue_position_ids = probe.get("position_ids")

        # ---- pin ------------------------------------------------------
        from transformers import masking_utils

        saved = {
            "visual_forward": visual.forward,
            "attn_forwards": [b.attn.forward for b in visual.blocks],
            "get_image_features": base.get_image_features,
            "get_placeholder_mask": base.get_placeholder_mask,
            "deepstack": base.language_model._deepstack_process,
            "use_cache": qwen.config.text_config.use_cache,
            "ignore_mask": masking_utils._ignore_causal_mask_sdpa,
        }
        if have_rope_index:
            saved["get_rope_index"] = base.get_rope_index
        if glue_owner is not None and glue_position_ids is not None:
            saved["glue"] = glue_owner._ensure_legacy_qwen3_position_ids

        def fixed_visual_forward(self, hidden_states, grid_thw=None,
                                 **kwargs):
            del grid_thw
            kwargs.pop("return_dict", None)
            hidden_states = self.patch_embed(hidden_states)
            hidden_states = hidden_states + pos_embeds
            seq, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(seq, -1)
            deepstack_features = []
            for index, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states, cu_seqlens=fixed_cu_seqlens,
                    position_embeddings=position_embeddings, **kwargs)
                if index in self.deepstack_visual_indexes:
                    merger = self.deepstack_visual_indexes.index(index)
                    deepstack_features.append(
                        self.deepstack_merger_list[merger](hidden_states))
            merged = self.merger(hidden_states)
            if output_cls is not None:
                return output_cls(last_hidden_state=hidden_states,
                                  pooler_output=merged,
                                  deepstack_features=deepstack_features)
            return merged, deepstack_features

        def fixed_vision_attention(self, hidden_states, cu_seqlens=None,
                                   rotary_pos_emb=None,
                                   position_embeddings_arg=None, **kwargs):
            del cu_seqlens, rotary_pos_emb, position_embeddings_arg
            seq = hidden_states.shape[0]
            query, key, value = (self.qkv(hidden_states)
                                 .reshape(seq, 3, self.num_heads, -1)
                                 .permute(1, 0, 2, 3).unbind(0))
            cos_t, sin_t = position_embeddings
            query, key = modeling.apply_rotary_pos_emb_vision(
                query, key, cos_t, sin_t)
            query = query.transpose(0, 1).unsqueeze(0)
            key = key.transpose(0, 1).unsqueeze(0)
            value = value.transpose(0, 1).unsqueeze(0)
            interface = modeling.eager_attention_forward
            if self.config._attn_implementation != "eager":
                interface = modeling.ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation]
            chunks = (torch.split(t, vision_lengths, dim=2)
                      for t in (query, key, value))
            outputs = [interface(self, q, k, v, attention_mask=None,
                                 scaling=self.scaling, dropout=0.0,
                                 is_causal=False, **kwargs)[0]
                       for q, k, v in zip(*chunks)]
            return self.proj(torch.cat(outputs, dim=1)
                             .reshape(seq, -1).contiguous())

        def fixed_get_image_features(self, pixel_values,
                                     image_grid_thw=None, **kwargs):
            del kwargs
            pixel_values = pixel_values.type(self.visual.dtype)
            vision_output = self.visual(pixel_values,
                                        grid_thw=image_grid_thw)
            if output_cls is not None:
                vision_output.pooler_output = torch.split(
                    vision_output.pooler_output, split_sizes)
                return vision_output
            embeds, deepstack = vision_output
            return torch.split(embeds, split_sizes), deepstack

        def fixed_get_placeholder_mask(self, input_ids_arg, inputs_embeds,
                                       image_features=None,
                                       video_features=None):
            del self, input_ids_arg, image_features, video_features
            return (image_mask.expand_as(inputs_embeds),
                    video_mask.expand_as(inputs_embeds))

        def fixed_deepstack(self, hidden_states, visual_pos_masks,
                            visual_embeds):
            del self, visual_pos_masks
            b, s, c = hidden_states.shape
            flat = hidden_states.reshape(b * s, c)
            updated = flat.index_select(0, visual_indices) + visual_embeds
            return flat.index_copy(0, visual_indices,
                                   updated).reshape(b, s, c)

        visual.forward = types.MethodType(fixed_visual_forward, visual)
        for block in visual.blocks:
            block.attn.forward = types.MethodType(fixed_vision_attention,
                                                  block.attn)
        base.get_image_features = types.MethodType(
            fixed_get_image_features, base)
        base.get_placeholder_mask = types.MethodType(
            fixed_get_placeholder_mask, base)
        base.language_model._deepstack_process = types.MethodType(
            fixed_deepstack, base.language_model)
        qwen.config.text_config.use_cache = False
        masking_utils._ignore_causal_mask_sdpa = lambda *a, **k: True
        if have_rope_index:
            base.get_rope_index = types.MethodType(
                lambda self, *a, **k: (fixed_position_ids,
                                       fixed_rope_deltas), base)
        if "glue" in saved:
            def pinned_glue(model_input, _pids=glue_position_ids):
                model_input["position_ids"] = _pids
            glue_owner._ensure_legacy_qwen3_position_ids = pinned_glue

        pins = ["visual_forward", "vision_attention",
                "get_image_features", "get_placeholder_mask",
                "deepstack", "use_cache", "causal_mask_decision"]
        if have_rope_index:
            pins.append("get_rope_index")
        if "glue" in saved:
            pins.append("legacy_position_ids_glue")

        # ---- dead-output pin, held only under proof ------------------
        # A feature-extraction pipeline never reads the vocabulary
        # logits, but whether a compiler can prove that depends on the
        # host's wrapper form — one host's graph dead-code-eliminates
        # the [tokens, vocab] projection, another's keeps it alive and
        # pays real milliseconds for it every call. The family settles
        # it with a receipt instead: skip the head, re-run the recorded
        # request, and keep the pin only if the pipeline output is
        # bit-identical. Any mismatch or error restores the head alone.
        lm_head = getattr(qwen, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "forward"):
            with torch.inference_mode():
                before = _first_output_tensor(runner())
            saved_head = lm_head.forward

            def skipped_head(self, hidden_states, *args, **kwargs):
                del args, kwargs
                return hidden_states[..., :0]

            lm_head.forward = types.MethodType(skipped_head, lm_head)
            proven = False
            if before is not None:
                try:
                    with torch.inference_mode():
                        after = _first_output_tensor(runner())
                    proven = after is not None and torch.equal(before,
                                                               after)
                except Exception:   # noqa: BLE001 — proof failed
                    proven = False
            if proven:
                saved["lm_head"] = saved_head
                pins.append("lm_head_dead_output")
            else:
                lm_head.forward = saved_head

        def undo():
            visual.forward = saved["visual_forward"]
            for block, fwd in zip(visual.blocks, saved["attn_forwards"]):
                block.attn.forward = fwd
            base.get_image_features = saved["get_image_features"]
            base.get_placeholder_mask = saved["get_placeholder_mask"]
            base.language_model._deepstack_process = saved["deepstack"]
            qwen.config.text_config.use_cache = saved["use_cache"]
            masking_utils._ignore_causal_mask_sdpa = saved["ignore_mask"]
            if "get_rope_index" in saved:
                base.get_rope_index = saved["get_rope_index"]
            if "glue" in saved:
                glue_owner._ensure_legacy_qwen3_position_ids = saved["glue"]
            if "lm_head" in saved:
                qwen.lm_head.forward = saved["lm_head"]

        return GraphLowering(
            undo=undo, family=self.family, pins=tuple(pins),
            details={"tokens": int(input_ids.shape[-1]),
                     "vision_patches": int(sum(vision_lengths)),
                     "vision_contract": ("output_class"
                                         if output_cls is not None
                                         else "tuple")})
