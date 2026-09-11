"""Structure discovery — find catalog structures inside a host model.

Walks the module tree and matches region-structure seams by shape, not
by model name: a gated gate/up/down MLP is a ``decoder_ffn`` seam, a
fc1/fc2 MLP with a sibling LayerNorm is a ``vision_ffn`` seam. The
result is the same information a hand-written binding file carries
(paths, dims, variant), derived from the model object itself; bindings
become generated receipts instead of required inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from torch import nn

_DECODER_PROJ = ("gate_proj", "up_proj", "down_proj")
_VISION_PROJ = (("fc1", "fc2"), ("linear_fc1", "linear_fc2"),
                ("0", "2"),
                ("net.0.proj", "net.2"))
_NORM_ATTRS = ("post_attention_layernorm", "layer_norm2", "norm2", "norm3")
_ATTN_PROJ = (("q_proj", "k_proj", "v_proj", "o_proj"),
              ("q_proj", "k_proj", "v_proj", "out_proj"),
              ("q", "k", "v", "o"),
              ("to_q", "to_k", "to_v", "to_out"),
              ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"))
# the HF decoder-layer shape: two sublayers, each a norm feeding a
# compute region. Matched by slots, not by class name, so every host
# built on that layout is the same seam.
_BLOCK_SLOTS = ("self_attn", "mlp", "input_layernorm",
                "post_attention_layernorm")
# sibling groups that qkv_pack packs into one GEMM: same input, fixed
# consumption order. The trailing o_proj/out_proj is not part of the
# pack (it consumes the attention output, not the shared input).
_QKV_PACK = (("q_proj", "k_proj", "v_proj"),
             ("q", "k", "v"),
             ("to_q", "to_k", "to_v"),
             ("add_q_proj", "add_k_proj", "add_v_proj"))
# adaptive-norm modules: a norm that also projects a conditioning
# vector. The child that produces the modulation is the tell.
_COND_PROJ_ATTRS = ("dense", "linear", "adaLN_modulation", "modulation")
_PROJ_WEIGHT_FLOOR = 262144  # candidacy filter only; impls add their own
                             # work-based qualification and gates decide


def _is_attn_block(module: nn.Module) -> bool:
    """A whole attention block, not just sibling projections.

    When the host exposes q/k/v/out plus head_dim and scale, the pack
    can replace the block itself and declare the attention compute
    dtype too — strictly more than packing the projections alone.
    """
    if not all(isinstance(getattr(module, a, None), nn.Linear)
               for a in ("q_proj", "k_proj", "v_proj", "out_proj")):
        return False
    if not (hasattr(module, "head_dim") and hasattr(module, "scale")):
        return False
    widths = {getattr(module, a).out_features
              for a in ("q_proj", "k_proj", "v_proj")}
    return len(widths) == 1


def _has_cond_forward(module: nn.Module) -> bool:
    """A norm takes a conditioning argument (adaptive norm) if its
    forward accepts a second positional / a ``cond``/``temb`` keyword."""
    import inspect
    try:
        params = list(inspect.signature(module.forward).parameters)
    except (TypeError, ValueError):
        return False
    if any(p in params for p in ("cond", "temb", "emb", "c")):
        return True
    # (self is bound out of module.forward already) x + one more positional
    positional = [p for p in params if p not in ("args", "kwargs")]
    return len(positional) >= 2


def _nested_module(module: nn.Module, path: str) -> nn.Module | None:
    """Resolve a relative child path, including Sequential indices."""
    node = module
    try:
        for part in path.split("."):
            node = node[int(part)] if part.isdigit() else getattr(node, part)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return node if isinstance(node, nn.Module) else None


def _is_modnorm_qkv_chain(
    module: nn.Module,
) -> tuple[int, int, str] | None:
    """Recognise a direct conditional-norm -> sibling-QKV data flow.

    The block is admitted only when no positional module sits between the
    modulated norm and the projections.  That is the property required by
    the shared FP8 wire; class and model names are deliberately irrelevant.
    """
    norm = getattr(module, "norm1", None)
    attn = getattr(module, "attn1", None)
    if not (isinstance(norm, nn.Module) and _has_cond_forward(norm)
            and isinstance(attn, nn.Module)):
        return None
    if getattr(module, "pos_embed", None) is not None:
        return None
    q_proj = getattr(attn, "to_q", None)
    if not isinstance(q_proj, nn.Linear):
        return None
    dim = q_proj.in_features
    k_proj, v_proj = getattr(attn, "to_k", None), getattr(attn, "to_v", None)
    fanout = "q_only"
    if (isinstance(k_proj, nn.Linear) and isinstance(v_proj, nn.Linear)
            and k_proj.in_features == dim and v_proj.in_features == dim):
        fanout = "qkv"
    cond = next((getattr(norm, attr, None) for attr in _COND_PROJ_ATTRS
                 if isinstance(getattr(norm, attr, None), nn.Linear)), None)
    if cond is None or cond.out_features != 2 * dim:
        return None
    return dim, cond.in_features, fanout


def _is_table_modnorm_chain(module: nn.Module) -> tuple[int, int] | None:
    """Recognise the per-token-table modulated block (video-DiT family).

    Shape, not class names: the block carries its own ``[1, chunks, D]``
    modulation parameter, a no-affine ``norm1`` and ``norm3`` pair, and
    sibling self/cross attentions plus an FFN whose modulation happens
    inline in the block's forward from a per-token timestep table. Only a
    block owner can reroute that inline math, which is why this seam is
    the whole block rather than a norm module.
    """
    table = getattr(module, "scale_shift_table", None)
    if not (isinstance(table, torch.nn.Parameter) and table.dim() == 3
            and table.shape[0] == 1 and table.shape[1] in (4, 6, 9)):
        return None
    attn = getattr(module, "attn1", None)
    q_proj = getattr(attn, "to_q", None) if attn is not None else None
    if not isinstance(q_proj, nn.Linear):
        return None
    dim = q_proj.in_features
    if table.shape[2] != dim:
        return None
    for attr in ("to_k", "to_v"):
        proj = getattr(attn, attr, None)
        if not (isinstance(proj, nn.Linear) and proj.in_features == dim):
            return None
    for norm_attr in ("norm1", "norm3"):
        norm = getattr(module, norm_attr, None)
        if norm is None or getattr(norm, "weight", None) is not None:
            return None
    if getattr(module, "attn2", None) is None:
        return None
    if getattr(module, "ffn", None) is None:
        return None
    return dim, int(table.shape[1])


def _projection_child(
    module: nn.Module, attr: str,
) -> tuple[str, nn.Linear] | None:
    """One attention projection, including Diffusers' ``to_out[0]``."""
    direct = getattr(module, attr, None)
    if isinstance(direct, nn.Linear):
        return attr, direct
    if attr not in ("to_out", "to_add_out"):
        return None
    try:
        first = direct[0]
    except (IndexError, KeyError, TypeError):
        return None
    return (attr + ".0", first) if isinstance(first, nn.Linear) else None


@dataclass
class Seam:
    """One replaceable site: a structure instance found in the host."""

    structure: str
    path: str                 # dotted path of the swappable module
    parent_path: str
    norm_attr: str | None
    dims: dict[str, int]
    variant: dict[str, str]
    fc_attrs: tuple[str, str] | None = None
    proj_attr: str | None = None      # linear_proj: attr name in parent
    pack_attrs: tuple[str, ...] | None = None   # qkv_pack: sibling attrs
    cond_attr: str | None = None      # adaln_producer: cond-proj child
    family: str = ""
    layer_index: int = -1
    m_profile: list[int] = field(default_factory=list)
    #: what discovery had to take on trust to describe this seam. Carried
    #: to the receipt: an assumption nobody can see is indistinguishable
    #: from a fact, and these are the ones the parity gate has to check.
    assumptions: tuple[str, ...] = ()


_ACT_ATTRS = ("act_fn", "activation_fn", "act", "activation")


def _activation_of(module: nn.Module) -> tuple[str | None, bool]:
    """``(name, declared)`` for this module's activation.

    Two different unknowns, and conflating them was a silent failure. A
    module with *no* activation attribute tells us nothing, and the family
    default is a reasonable assumption to record and let the parity gate
    check. A module that *declares* an activation we cannot classify tells
    us something specific: it is not one of the two this library
    implements, so assuming otherwise would substitute a different
    function and the seam is refused instead.

    Returns ``(None, True)`` for the second case — declared but not ours.
    """
    fn = None
    for attr in _ACT_ATTRS:
        fn = getattr(module, attr, None)
        if fn is not None:
            break
    if fn is None:
        return None, False
    label = " ".join(
        [getattr(fn, "__name__", ""), type(fn).__name__, repr(fn)]).lower()
    if "silu" in label or "swish" in label:
        return "silu", True
    if "gelu" in label:
        return "gelu", True
    return None, True


def _activation_or_default(module: nn.Module, default: str
                           ) -> tuple[str | None, tuple[str, ...]]:
    """Resolve the activation, or refuse; report what was assumed.

    ``(None, ())`` means refuse this seam. Discovery turns that into
    "skip"; the explicit door turns it into an error — see
    :func:`activation_for`, which is the same decision with the other
    outcome, so the two doors cannot drift apart.
    """
    name, declared = _activation_of(module)
    if name is not None:
        return name, ()
    if declared:
        return None, ()          # declared, and not one of ours: refuse
    return default, (f"activation assumed {default} (host declares none)",)


def activation_for(module: nn.Module, default: str) -> str:
    """The activation name for an explicitly bound module, or raise."""
    name, _ = _activation_or_default(module, default)
    if name is None:
        fn = next((getattr(module, a) for a in _ACT_ATTRS
                   if getattr(module, a, None) is not None), None)
        raise ValueError(
            f"activation {type(fn).__name__!r} is declared by this module "
            f"and is not one this library implements (silu or gelu)")
    return name


def _resolve(root: nn.Module, path: str) -> nn.Module:
    node = root
    for part in path.split("."):
        if part:
            node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node


def _family_key(path: str) -> tuple[str, int]:
    """Template the trailing layer index: a.layers.12.mlp -> a.layers.{i}.mlp."""
    matches = list(re.finditer(r"\.(\d+)\.", "." + path + "."))
    if not matches:
        return path, -1
    m = matches[-1]
    start, end = m.start(1) - 1, m.end(1) - 1  # offsets in original path
    return path[:start] + "{i}" + path[end:], int(m.group(1))


def _norm_attr_of(root: nn.Module, parent_path: str) -> str | None:
    try:
        parent = _resolve(root, parent_path)
    except (AttributeError, IndexError, KeyError):
        return None
    for attr in _NORM_ATTRS:
        if isinstance(getattr(parent, attr, None), nn.Module):
            return attr
    return None


def _vision_norm_variant(
    norm: nn.Module, dim: int,
) -> tuple[str | None, str]:
    """Classify the exact LayerNorm affine contract at a vision-FFN seam."""
    shape = getattr(norm, "normalized_shape", None)
    if shape is None:
        # a fused LayerNorm twin (host accelerator libraries swap these
        # in) often drops ``normalized_shape`` while keeping the affine
        # contract itself: a 1-D weight names the normalized width just
        # as authoritatively, and the parity gates certify the math
        weight = getattr(norm, "weight", None)
        if weight is not None and getattr(weight, "ndim", 0) == 1:
            shape = tuple(weight.shape)
    if isinstance(shape, int):
        shape = (shape,)
    try:
        shape = tuple(shape)
    except TypeError:
        return None, ("norm has no LayerNorm shape contract "
                      "(normalized_shape or 1-D affine weight)")
    if shape != (dim,) or not hasattr(norm, "eps"):
        return None, (
            f"norm shape/epsilon is not LayerNorm({dim}); got shape={shape}")
    weight, bias = getattr(norm, "weight", None), getattr(norm, "bias", None)
    if weight is None and bias is None:
        return "identity", ""
    if weight is None or bias is None:
        return None, (
            "norm exposes a one-sided affine contract (for example RMSNorm); "
            "vision_ffn requires LayerNorm with both affine tensors or neither")
    if tuple(weight.shape) != (dim,) or tuple(bias.shape) != (dim,):
        return None, "norm affine tensors do not match the vision width"
    return "learned", ""


def discover(
    model: nn.Module,
    structures: tuple[str, ...] = ("decoder_ffn", "vision_ffn"),
    *,
    refused: list[tuple[str, str]] | None = None,
) -> list[Seam]:
    """Find every region-structure seam in ``model``."""
    seams: list[Seam] = []
    for path, module in model.named_modules():
        if not path:
            continue
        parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        if "decoder_ffn" in structures and all(
            # The catalog exposes only gate/up/down weights. Accepting a
            # biased host here would silently drop parameters at the seam.
            isinstance(getattr(module, a, None), nn.Linear)
            and getattr(module, a).bias is None
            for a in _DECODER_PROJ
        ):
            gate = module.gate_proj
            act, assumed = _activation_or_default(module, "silu")
            if act is None:
                continue
            family, idx = _family_key(path)
            seams.append(Seam(
                structure="decoder_ffn", path=path, parent_path=parent_path,
                norm_attr=_norm_attr_of(model, parent_path),
                dims={"D": gate.in_features, "F": gate.out_features},
                variant={"activation": act, "norm_weight_mode": "direct"},
                family=family, layer_index=idx, assumptions=assumed))
            continue
        if "decoder_block" in structures and all(
            isinstance(getattr(module, a, None), nn.Module)
            for a in _BLOCK_SLOTS
        ):
            norm_in = module.input_layernorm
            gated = _has_cond_forward(norm_in)
            width = getattr(module, "hidden_size", None)
            if width is None:
                w = getattr(norm_in, "weight", None)
                width = (int(w.shape[-1]) if w is not None
                         else getattr(norm_in, "dim", 0))
            family, idx = _family_key(path)
            seams.append(Seam(
                structure="decoder_block", path=path,
                parent_path=parent_path, norm_attr="input_layernorm",
                dims={"D": int(width)},
                variant={"residual": "gated" if gated else "plain",
                         "norm": "adaln_rms" if gated else "rms",
                         "ffn_entry": "fp8_static"},
                family=family, layer_index=idx))
        if "modnorm_qkv_chain" in structures:
            chain_dims = _is_modnorm_qkv_chain(module)
            if chain_dims is not None:
                dim, cond_dim, fanout = chain_dims
                family, idx = _family_key(path)
                seams.append(Seam(
                    structure="modnorm_qkv_chain", path=path,
                    parent_path=parent_path, norm_attr="norm1",
                    dims={"D": dim, "C": cond_dim},
                    variant={"modulation": "scale_shift",
                             "wire_dtype": "fp8_static",
                             "fanout": fanout},
                    family=family, layer_index=idx))
            else:
                table_dims = _is_table_modnorm_chain(module)
                if table_dims is not None:
                    dim, chunks = table_dims
                    family, idx = _family_key(path)
                    seams.append(Seam(
                        structure="modnorm_qkv_chain", path=path,
                        parent_path=parent_path, norm_attr="norm1",
                        dims={"D": dim, "C": chunks},
                        variant={"modulation": "per_token_table",
                                 "wire_dtype": "fp8_static",
                                 "fanout": "qkv"},
                        family=family, layer_index=idx))
        if "qkv_pack" in structures:
            for group in _QKV_PACK:
                projs = [getattr(module, a, None) for a in group]
                if not all(isinstance(p, nn.Linear) for p in projs):
                    continue
                if len({p.in_features for p in projs}) != 1:
                    continue  # siblings must share the input dim
                if projs[0].weight.numel() < _PROJ_WEIGHT_FLOOR:
                    continue
                family, idx = _family_key(path)
                bind = "module" if _is_attn_block(module) else "leaf"
                seams.append(Seam(
                    structure="qkv_pack", path=path, parent_path=parent_path,
                    norm_attr=None, pack_attrs=group,
                    dims={"K": projs[0].in_features,
                          "N": sum(p.out_features for p in projs)},
                    variant={"bind": bind,
                             "in_dtype": "bf16_fused_quant"},
                    family=family, layer_index=idx))
        if "adaln_producer" in structures:
            cond_attr = next(
                (a for a in _COND_PROJ_ATTRS
                 if isinstance(getattr(module, a, None), nn.Linear)), None)
            if (cond_attr is not None and _has_cond_forward(module)
                    and getattr(module, cond_attr).out_features
                    % 2 == 0):
                cond_proj = getattr(module, cond_attr)
                family, idx = _family_key(path)
                # style width is a multiple of the model dim: 3x (scale,
                # shift, gate) for RMS AdaLN, 2x (scale, shift) for LN
                seams.append(Seam(
                    structure="adaln_producer", path=path,
                    parent_path=parent_path, norm_attr=None,
                    cond_attr=cond_attr,
                    dims={"C": cond_proj.in_features,
                          "S": cond_proj.out_features},
                    variant={"bind": "table_only", "out_dtype": "bf16"},
                    family=family, layer_index=idx))
        if "norm_fused" in structures and isinstance(module, nn.LayerNorm):
            if (getattr(module, "weight", None) is not None
                    and getattr(module, "bias", None) is not None):
                family, idx = _family_key(path)
                seams.append(Seam(
                    structure="norm_fused", path=path,
                    parent_path=parent_path, norm_attr=None,
                    # take the dim from the affine weight: subclasses
                    # (fused LayerNorm variants) may not carry
                    # normalized_shape
                    dims={"D": int(module.weight.shape[-1])},
                    variant={"norm": "layer", "compute_dtype": "bf16"},
                    family=family, layer_index=idx))
        if "linear_proj" in structures:
            for group in _ATTN_PROJ:
                resolved = [_projection_child(module, attr)
                            for attr in group]
                if not all(item is not None for item in resolved):
                    continue
                for attr, proj in resolved:
                    if proj.weight.numel() < _PROJ_WEIGHT_FLOOR:
                        continue
                    family, idx = _family_key(path)
                    seams.append(Seam(
                        structure="linear_proj",
                        path=path + "." + attr, parent_path=path,
                        norm_attr=None, proj_attr=attr,
                        dims={"K": proj.in_features,
                              "N": proj.out_features},
                        variant={"bias": ("add" if proj.bias is not None
                                          else "none"),
                                 "epilogue": "none", "in_dtype": "bf16"},
                        family=family + "." + attr, layer_index=idx))
        if "patch_projection" in structures:
            # Some vision processors already emit one flattened, complete
            # spatio-temporal patch per row. Their host module spells the
            # following projection as Conv3d, even though kernel=stride is
            # exactly that one patch and the convolution has no overlap,
            # padding, dilation or groups. Match this complete semantic
            # contract; an ordinary Conv3d must never be lowered here.
            proj = getattr(module, "proj", None)
            if isinstance(proj, nn.Conv3d):
                try:
                    temporal = int(module.temporal_patch_size)
                    spatial = int(module.patch_size)
                    in_channels = int(module.in_channels)
                    embed_dim = int(module.embed_dim)
                except (AttributeError, TypeError, ValueError):
                    pass
                else:
                    kernel = (temporal, spatial, spatial)
                    if (
                        tuple(proj.kernel_size) == kernel
                        and tuple(proj.stride) == kernel
                        and tuple(proj.padding) == (0, 0, 0)
                        and tuple(proj.dilation) == (1, 1, 1)
                        and proj.groups == 1
                        and proj.in_channels == in_channels
                        and proj.out_channels == embed_dim
                        and proj.weight.numel() >= _PROJ_WEIGHT_FLOOR
                    ):
                        family, idx = _family_key(path)
                        seams.append(Seam(
                            structure="patch_projection", path=path,
                            parent_path=parent_path, norm_attr=None,
                            dims={"K": in_channels * temporal * spatial * spatial,
                                  "N": embed_dim},
                            variant={"layout": "preflattened_full_patch",
                                     "bias": ("add" if proj.bias is not None
                                              else "none")},
                            family=family + ".patch_projection",
                            layer_index=idx,
                        ))
        if "vision_ffn" in structures:
            for fc1_attr, fc2_attr in _VISION_PROJ:
                fc1 = _nested_module(module, fc1_attr)
                fc2 = _nested_module(module, fc2_attr)
                if not (isinstance(fc1, nn.Linear)
                        and isinstance(fc2, nn.Linear)):
                    continue
                if (fc1.out_features != fc2.in_features
                        or fc1.in_features != fc2.out_features):
                    continue  # not an FFN pair: silence is right here
                # Past this point the seam has been recognised, and every
                # exit is a refusal against a declared boundary. Those
                # must reach the trail: a silent skip reads as "nothing
                # here" when the truth is "this shape, refused for this
                # reason", and the difference is a debugging session.
                if fc1.bias is None or fc2.bias is None:
                    if refused is not None:
                        refused.append((
                            path,
                            "vision_ffn refused: b_fc1/b_fc2 are required "
                            "slots and this host's projections carry no "
                            "bias",
                        ))
                    continue
                norm_attr = _norm_attr_of(model, parent_path)
                if norm_attr is None:
                    if refused is not None:
                        refused.append((
                            path,
                            "vision_ffn refused: the boundary includes a "
                            "norm and no norm attribute was found beside "
                            "this feed-forward",
                        ))
                    continue
                norm = _resolve(
                    model,
                    (parent_path + "." + norm_attr).lstrip("."),
                )
                norm_affine, reason = _vision_norm_variant(
                    norm, fc1.in_features)
                if norm_affine is None:
                    if refused is not None:
                        refused.append((
                            path,
                            f"vision_ffn refused: {reason}",
                        ))
                    continue
                act, assumed = _activation_or_default(module, "gelu")
                if act is None:
                    break
                family, idx = _family_key(path)
                seams.append(Seam(
                    structure="vision_ffn", path=path,
                    parent_path=parent_path, norm_attr=norm_attr,
                    dims={"D": fc1.in_features, "F": fc1.out_features},
                    variant={"activation": act,
                             "norm_affine": norm_affine},
                    fc_attrs=(fc1_attr, fc2_attr),
                    family=family, layer_index=idx, assumptions=assumed))
                break
    return seams


def group_families(seams: list[Seam]) -> dict[str, list[Seam]]:
    """Group seams into families (same template path), index-sorted."""
    families: dict[str, list[Seam]] = {}
    for seam in seams:
        families.setdefault(seam.family, []).append(seam)
    for members in families.values():
        members.sort(key=lambda s: s.layer_index)
    return families


def seam_weights(model: nn.Module, seam: Seam) -> dict[str, torch.Tensor]:
    """Extract the impl-facing weight dict for one seam."""
    module = _resolve(model, seam.path)
    norm = (_resolve(model, seam.parent_path + "." + seam.norm_attr)
            if seam.norm_attr else None)
    if seam.structure == "linear_proj":
        return {"w": module.weight.detach(),
                "b": (module.bias.detach()
                      if module.bias is not None else None)}
    if seam.structure == "patch_projection":
        proj = module.proj
        return {
            "w": proj.weight.detach().reshape(seam.dims["N"], -1),
            "b": (proj.bias.detach() if proj.bias is not None else None),
        }
    if seam.structure == "decoder_ffn":
        w_norm = (norm.weight.detach() if norm is not None
                  and getattr(norm, "weight", None) is not None
                  else torch.ones(seam.dims["D"]))
        return {
            "w_norm": w_norm,
            "w_gate": module.gate_proj.weight.detach().t().contiguous(),
            "w_up": module.up_proj.weight.detach().t().contiguous(),
            "w_down": module.down_proj.weight.detach().t().contiguous(),
        }
    fc1_attr, fc2_attr = seam.fc_attrs
    fc1, fc2 = _nested_module(module, fc1_attr), _nested_module(
        module, fc2_attr)
    norm_weight = getattr(norm, "weight", None)
    norm_bias = getattr(norm, "bias", None)
    return {
        "w_norm": (norm_weight.detach()
                   if norm_weight is not None else None),
        "b_norm": (norm_bias.detach()
                   if norm_bias is not None else None),
        "w_fc1": fc1.weight.detach(),
        "b_fc1": fc1.bias.detach(),
        "w_fc2": fc2.weight.detach(),
        "b_fc2": fc2.bias.detach(),
    }
