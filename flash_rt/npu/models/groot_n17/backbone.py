"""GR00T N1.7 backbone on Ascend: the Qwen3-VL tower, bound and captured.

The backbone is the half of the frame the denoise loop does *not* repeat, and
on this part it is the larger half: 59.4 ms of device time in 2682 eager
launches. It starts where the other ports start it -- after the patch embed and
the text embedding lookup, both of which the caller performs and hands over as
``aux`` -- and ends at the 2048-wide feature sequence the action chain consumes.

Three stages, in order:

* **The patch projection**, which the checkpoint stores as a ``Conv3d`` over
  ``(3, 2, 16, 16)`` voxels and which is a plain matmul once the patch is
  flattened -- the host hands over raw patches and the graph projects them.
  The interpolated position table depends only on the image grid, so it is a
  prompt constant the projection adds.

* **A 24-block vision tower** over the patch features, with 2-D rotary
  embeddings and attention that does *not* cross views: the reference splits
  the sequence by ``cu_seqlens`` per image, which here is one batched call with
  the views on the batch axis rather than a loop.
* **Three DeepStack taps and a merger.** Blocks 5, 11 and 17 each feed a merger
  that normalises after the 2x2 spatial shuffle; the final merger normalises
  before it. The two read the same shuffle and differ only in that order, which
  is visible in the checkpoint as a 4096-wide norm against a 1024-wide one.
* **A truncated 16-layer causal LLM**, GQA 16/8 over head width 128, with the
  DeepStack features added at the visual token positions after each of the
  first three layers. What it publishes is the **last layer's output, before
  the final norm** -- the host reads ``hidden_states[-1]``, which this
  transformers version fills from a layer hook rather than from the normed
  result, so the tower's own final RMSNorm never reaches the action head. The
  weight is in the checkpoint and is deliberately not applied; applying it
  costs cosine 0.14 against the reference and nothing about the shapes
  complains.

The same decisions that paid in the action chain apply here and for the same
reasons: weights in fractal NZ, attention in BSH so a projection feeds the
operator without a head permute, residuals joined to the norm that follows
them, and every per-prompt quantity -- rope tables, the visual position index,
the causal extent -- computed once and then only read.

Causality does cost a mask. The prompt flash-attention operator takes a token
window, and a zero forward window reads like a causal one -- but at this
geometry it is ignored and the operator returns the full attention, silently
and with no error. Only an explicit upper-triangular mask makes it causal, so
that mask is built once with the prompt and then only read.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_rt.npu.models.groot_n17.pipeline import _bias, _nz, add_norm

# Vision tower.
VIT_LAYERS = 24
VIT_DIM = 1024
VIT_HEADS = 16
VIT_HEAD_DIM = 64
VIT_FF_DIM = 4096
VIT_EPS = 1e-6
PATCH_DIM = 3 * 2 * 16 * 16                    # channels x temporal x 16 x 16
SPATIAL_MERGE = 2
MERGED_DIM = VIT_DIM * SPATIAL_MERGE ** 2      # 4096
DEEPSTACK_TAPS = (5, 11, 17)

# Truncated language model.
LLM_LAYERS = 16
LLM_DIM = 2048
LLM_HEADS = 16
LLM_KV_HEADS = 8
LLM_HEAD_DIM = 128
LLM_FF_DIM = 6144
LLM_EPS = 1e-6

_INT_MAX = 2147483647


def rope(x, cos, sin):
    """``x * cos + rotate_half(x) * sin`` over the head axis, in one launch.

    Written out in torch this is six kernels -- two slices, a negate, a concat,
    two multiplies and an add -- and it runs eighty times a frame. The vendor's
    fused rotary is the same expression: 18.5 us against 104.2 at the vision
    tower's shape, 14.5 against 67.8 at the language model's, agreeing with the
    FP32-promoted form to cosine 0.999996, which is inside BF16's own rounding.

    The operand must be contiguous: it arrives as a slice of a fused projection,
    and handing the operator a strided view costs a copy it then has to make.
    """
    import torch_npu
    return torch_npu.npu_rotary_mul(x.contiguous(), cos, sin)


def attention(query, key, value, heads, head_dim, kv_heads=None, mask=None):
    """One capturable attention call in the layout the projections produce.

    The backbone's sites are hundreds of tokens wide, and there the fused
    infer-attention entry point is the faster of the two: 125.7 us against
    147.4 at the vision tower's geometry and 90.9 against 110.3 at the language
    model's, with identical output. At the action head's 41-token geometries
    the ordering reverses, which is why the two halves of this port call
    different operators — routing is by measured shape, not by preference.
    """
    import torch_npu
    return torch_npu.npu_fused_infer_attention_score(
        query, key, value, atten_mask=mask, num_heads=heads,
        num_key_value_heads=kv_heads or heads, scale=head_dim ** -0.5,
        input_layout="BSH", softmax_lse_flag=False)[0]


def causal_mask(length: int, device) -> torch.Tensor:
    """Mask out every key above the diagonal: True is masked, so this is
    exactly ``j > i``. A prompt constant, built once and read thereafter."""
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), 1)


class BoundBackbone:
    """Every backbone weight, bound once for the device."""

    def __init__(self, weights, *, device="npu:0"):
        self.device = device
        dev = device

        # The Conv3d's weight is (out, channels, temporal, h, w); flattened over
        # everything but the output it is exactly the matmul's (N, K).
        self.patch_embed = (
            _nz(weights._patch_embed_w.reshape(VIT_DIM, PATCH_DIM).t().contiguous(), dev),
            _bias(weights._patch_embed_b, dev))

        self.vit = []
        for i in range(VIT_LAYERS):
            self.vit.append(dict(
                norm1=(_bias(weights._vit_ln1_w[i], dev), _bias(weights._vit_ln1_b[i], dev)),
                qkv=(_nz(weights._vit_qkv_w[i], dev), _bias(weights._vit_qkv_b[i], dev)),
                o=(_nz(weights._vit_o_w[i], dev), _bias(weights._vit_o_b[i], dev)),
                norm2=(_bias(weights._vit_ln2_w[i], dev), _bias(weights._vit_ln2_b[i], dev)),
                fc1=(_nz(weights._vit_fc1_w[i], dev), _bias(weights._vit_fc1_b[i], dev)),
                fc2=(_nz(weights._vit_fc2_w[i], dev), _bias(weights._vit_fc2_b[i], dev)),
            ))

        self.deepstack = []
        for k in range(len(DEEPSTACK_TAPS)):
            self.deepstack.append(dict(
                norm=(_bias(getattr(weights, f"_dsm{k}_norm_w"), dev),
                      _bias(getattr(weights, f"_dsm{k}_norm_b"), dev)),
                fc1=(_nz(getattr(weights, f"_dsm{k}_fc1_w"), dev),
                     _bias(getattr(weights, f"_dsm{k}_fc1_b"), dev)),
                fc2=(_nz(getattr(weights, f"_dsm{k}_fc2_w"), dev),
                     _bias(getattr(weights, f"_dsm{k}_fc2_b"), dev)),
            ))
        self.merger = dict(
            norm=(_bias(weights._merger_norm_w, dev), _bias(weights._merger_norm_b, dev)),
            fc1=(_nz(weights._merger_fc1_w, dev), _bias(weights._merger_fc1_b, dev)),
            fc2=(_nz(weights._merger_fc2_w, dev), _bias(weights._merger_fc2_b, dev)),
        )

        self.llm = []
        for i in range(LLM_LAYERS):
            self.llm.append(dict(
                input_norm=_bias(weights._llm_input_ln_w[i], dev),
                qkv=_nz(weights._llm_qkv_w[i], dev),
                q_norm=_bias(weights._llm_q_norm_w[i], dev),
                k_norm=_bias(weights._llm_k_norm_w[i], dev),
                o=_nz(weights._llm_o_w[i], dev),
                post_norm=_bias(weights._llm_post_ln_w[i], dev),
                gate=_nz(weights._llm_gate_w[i], dev),
                up=_nz(weights._llm_up_w[i], dev),
                down=_nz(weights._llm_down_w[i], dev),
            ))
        self.llm_norm = _bias(weights._llm_norm_w, dev)


# ══════════════════════════════════════════════════════════════════════
#  Vision tower
# ══════════════════════════════════════════════════════════════════════

def patch_project(bound: BoundBackbone, patches: torch.Tensor,
                  positions: torch.Tensor) -> torch.Tensor:
    """Raw flattened patches to vision-tower features."""
    return bound.patch_embed[0](patches, bound.patch_embed[1]) + positions


def vision(bound: BoundBackbone, features: torch.Tensor, cos: torch.Tensor,
           sin: torch.Tensor, views: int):
    """Patch features in, merged image tokens and three DeepStack taps out.

    ``features`` is ``(views * tokens, VIT_DIM)``. Attention does not cross
    views, which the reference expresses as a per-image split and this as the
    view axis of one batched call.
    """
    tokens = features.shape[0] // views
    x = features.reshape(views, tokens, VIT_DIM)
    taps = []
    h = F.layer_norm(x, (VIT_DIM,), bound.vit[0]["norm1"][0],
                     bound.vit[0]["norm1"][1], VIT_EPS)
    for index, layer in enumerate(bound.vit):
        qkv = layer["qkv"][0](h, layer["qkv"][1])
        query, key, value = qkv.split(VIT_DIM, dim=-1)
        query = rope(query.reshape(views, tokens, VIT_HEADS, VIT_HEAD_DIM), cos, sin)
        key = rope(key.reshape(views, tokens, VIT_HEADS, VIT_HEAD_DIM), cos, sin)
        a = attention(query.reshape(views, tokens, VIT_DIM),
                      key.reshape(views, tokens, VIT_DIM),
                      value.contiguous(), VIT_HEADS, VIT_HEAD_DIM)
        h, x = add_norm(x, layer["o"][0](a, layer["o"][1]),
                        layer["norm2"][0], layer["norm2"][1], VIT_EPS)
        h = F.gelu(layer["fc1"][0](h, layer["fc1"][1]), approximate="tanh")
        branch = layer["fc2"][0](h, layer["fc2"][1])
        if index + 1 < VIT_LAYERS:
            following = bound.vit[index + 1]["norm1"]
            h, x = add_norm(x, branch, following[0], following[1], VIT_EPS)
        else:
            x = x + branch
        if index in DEEPSTACK_TAPS:
            taps.append(deepstack_merge(bound.deepstack[DEEPSTACK_TAPS.index(index)], x))
    return final_merge(bound.merger, x), taps


def deepstack_merge(weights, x: torch.Tensor) -> torch.Tensor:
    """A DeepStack merger: shuffle first, then normalise the shuffled width."""
    shuffled = x.reshape(-1, MERGED_DIM)
    h = F.layer_norm(shuffled, (MERGED_DIM,), weights["norm"][0], weights["norm"][1],
                     VIT_EPS)
    h = F.gelu(weights["fc1"][0](h, weights["fc1"][1]))
    return weights["fc2"][0](h, weights["fc2"][1])


def final_merge(weights, x: torch.Tensor) -> torch.Tensor:
    """The final merger: normalise the token width, then shuffle."""
    h = F.layer_norm(x, (VIT_DIM,), weights["norm"][0], weights["norm"][1], VIT_EPS)
    h = F.gelu(weights["fc1"][0](h.reshape(-1, MERGED_DIM), weights["fc1"][1]))
    return weights["fc2"][0](h, weights["fc2"][1])


# ══════════════════════════════════════════════════════════════════════
#  Truncated language model
# ══════════════════════════════════════════════════════════════════════

def rms_norm(x, gamma, eps=LLM_EPS):
    import torch_npu
    return torch_npu.npu_rms_norm(x, gamma, eps)[0]


def add_rms_norm(x, branch, gamma, eps=LLM_EPS):
    """``branch + x`` and the RMS norm of the sum, in one launch."""
    import torch_npu
    normalised, _, total = torch_npu.npu_add_rms_norm(x, branch, gamma, eps)
    return normalised, total


def language_trace(bound, embeds, cos, sin, visual_index, taps, mask):
    """Every layer's output, for bisecting a divergence against the reference."""
    return language(bound, embeds, cos, sin, visual_index, taps, mask, collect=True)[1]


def language(bound: BoundBackbone, embeds: torch.Tensor, cos: torch.Tensor,
             sin: torch.Tensor, visual_index: torch.Tensor, taps,
             mask: torch.Tensor, collect: bool = False) -> torch.Tensor:
    """The truncated causal tower, with the DeepStack taps injected.

    The taps land at the visual token positions after each of the first three
    layers. The positions are a property of the prompt, so the scatter is an
    ``index_add_`` against an index computed once.
    """
    batch, length, _ = embeds.shape
    trace = []
    x = embeds
    h = rms_norm(x, bound.llm[0]["input_norm"])
    for index, layer in enumerate(bound.llm):
        qkv = layer["qkv"](h)
        query, key, value = qkv.split(
            [LLM_HEADS * LLM_HEAD_DIM, LLM_KV_HEADS * LLM_HEAD_DIM,
             LLM_KV_HEADS * LLM_HEAD_DIM], dim=-1)
        query = rms_norm(query.reshape(batch, length, LLM_HEADS, LLM_HEAD_DIM),
                         layer["q_norm"])
        key = rms_norm(key.reshape(batch, length, LLM_KV_HEADS, LLM_HEAD_DIM),
                       layer["k_norm"])
        query = rope(query, cos, sin).reshape(batch, length, -1)
        key = rope(key, cos, sin).reshape(batch, length, -1)
        a = attention(query, key, value.contiguous(), LLM_HEADS, LLM_HEAD_DIM,
                      kv_heads=LLM_KV_HEADS, mask=mask)
        h, x = add_rms_norm(x, layer["o"](a), layer["post_norm"])
        gate = F.silu(layer["gate"](h)) * layer["up"](h)
        branch = layer["down"](gate)
        if index + 1 == LLM_LAYERS:
            # The published feature is this sum, unnormalised.
            x = x + branch
        elif index < len(taps):
            # The residual has to carry the tap, so the sum is taken here and
            # the following norm reads it rather than the other way round.
            x = x + branch
            x.reshape(-1, LLM_DIM).index_add_(0, visual_index, taps[index])
            h = rms_norm(x, bound.llm[index + 1]["input_norm"])
        else:
            h, x = add_rms_norm(x, branch, bound.llm[index + 1]["input_norm"])
        trace.append(x)
    return (x, trace) if collect else x
