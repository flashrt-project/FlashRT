"""GR00T N1.7 action chain on Ascend: setup-time binding and the frame forward.

The chain is the half of the frame that starts at the backbone's 461 feature
tokens and ends at an action trajectory: the VL adapter, the state encoder, and
then four flow-matching steps over a 32-layer DiT. On this part it is the
larger half in both time and launches, and it is the half the denoise loop
repeats, so it is where a layout or a fused epilogue is worth four times what
it is worth in the backbone.

Four things here are setup-time decisions rather than per-frame work, and each
of them removes launches from the replayed graph:

* **Every projection is held in fractal NZ.** The cube reads an NZ operand with
  contiguous fractal loads rather than the strided gather an ND weight forces.

* **Attention runs in BSH.** ``npu_prompt_flash_attention`` reads ``(B, S,
  heads*head_dim)`` directly, so the projection output is already in the layout
  the operator wants and the reference's four per-attention head permutes do
  not exist on this path. The eager frame spends 11.9 ms of its 58.4 ms in
  Transpose; this is most of where that goes.

* **The AdaLN modulators are a constant table.** The denoise timesteps are
  fixed integers -- 0, 250, 500, 750 for four steps -- so every timestep
  embedding, every layer's ``(shift, scale)`` pair and the output head's pair
  are computable once. Better, ``norm(x) * (1 + scale) + shift`` over an
  affine-free LayerNorm is exactly ``layer_norm(x, weight=1+scale,
  bias=shift)``, so each of the 128 modulated norms is one kernel instead of
  three.

* **Cross-attention K/V is computed once per frame.** It is a function of the
  backbone features alone, and the reference recomputes all 32 of those
  projections on every one of the four steps.

The vendor constraint that shapes the rest: ``F.scaled_dot_product_attention``
routes to ``npu_fusion_attention`` and **a capture containing one cannot be
ended** -- CANN reports a stream that was never joined. The PFA family captures
at every geometry this model uses, and it also accepts the ``[B, N, Sq, Skv]``
mask that the SDPA path refuses to broadcast along the query axis.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from flash_rt.npu.core.linear import NzBf16Weight

# Geometry. These are the shipped GR00T-N1.7-3B configuration; the frontend
# checks the checkpoint's own config against them rather than assuming.
DIT_LAYERS = 32
DIT_HEADS = 32
DIT_HEAD_DIM = 48
DIT_DIM = DIT_HEADS * DIT_HEAD_DIM          # 1536
DIT_FF_DIM = 4 * DIT_DIM                    # 6144
DIT_OUT_DIM = 1024
BACKBONE_DIM = 2048
VLSA_LAYERS = 4
VLSA_HEADS = 32
VLSA_HEAD_DIM = 64
VLSA_DIM = VLSA_HEADS * VLSA_HEAD_DIM       # 2048
VLSA_FF_DIM = 4 * VLSA_DIM                  # 8192
STATE_DIM = 132
ACTION_DIM = 132
ACTION_HORIZON = 40
NUM_STEPS = 4
TIMESTEP_BUCKETS = 1000
ATTEND_TEXT_EVERY_N_BLOCKS = 2
EPS = 1e-5
NORM_OUT_EPS = 1e-6

_INT_MAX = 2147483647


def _nz(weight_kn: torch.Tensor, device) -> NzBf16Weight:
    """Bind an already ``(K, N)`` oriented weight into a fractal-NZ operand.

    The shared checkpoint spec transposes on load, so weights arrive in the
    orientation a GEMM consumes. ``NzBf16Weight.bind`` takes the stored
    ``(N, K)`` form, hence the transpose back: it costs one setup-time copy and
    keeps a single validated binder for both backends' conventions.
    """
    return NzBf16Weight.bind(weight_kn.t().contiguous().to(torch.bfloat16).to(device))


def _bias(tensor, device):
    return tensor.to(torch.bfloat16).to(device).contiguous()


def attention(query, key, value, heads, head_dim, mask=None):
    """One capturable attention call in the layout the projections produce."""
    import torch_npu
    return torch_npu.npu_prompt_flash_attention(
        query, key, value, atten_mask=mask, num_heads=heads,
        num_key_value_heads=heads, scale_value=head_dim ** -0.5,
        input_layout="BSH", pre_tokens=_INT_MAX, next_tokens=_INT_MAX,
        sparse_mode=0)


def _timestep_projection(steps: int, channels: int = 256,
                         downscale_freq_shift: float = 1.0) -> torch.Tensor:
    """The reference's sinusoidal timestep features, on the host in FP32.

    ``diffusers.Timesteps`` with ``flip_sin_to_cos=True``: cosine half first,
    then sine. Reproduced rather than imported so the constant table does not
    depend on a diffusers version at serving time.
    """
    half = channels // 2
    exponent = -math.log(10000.0) * torch.arange(half, dtype=torch.float32)
    exponent = exponent / (half - downscale_freq_shift)
    frequencies = torch.exp(exponent)
    values = torch.tensor([int(step / steps * TIMESTEP_BUCKETS) for step in range(steps)],
                          dtype=torch.float32)
    emb = values[:, None] * frequencies[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


def _tau_encoding(steps: int, dim: int) -> torch.Tensor:
    """``SinusoidalPositionalEncoding`` of the action encoder, sine half first."""
    half = dim // 2
    exponent = -torch.arange(half, dtype=torch.float32) * (math.log(10000.0) / half)
    values = torch.tensor([int(step / steps * TIMESTEP_BUCKETS) for step in range(steps)],
                          dtype=torch.float32)
    freqs = values[:, None] * exponent.exp()[None, :]
    return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class BoundChain:
    """Every weight of the action chain, bound for one embodiment.

    Binding is where the per-embodiment slice, the NZ cast and the modulator
    table happen. After ``__init__`` returns, nothing here allocates or decides
    anything: the forward below is a fixed sequence of operator calls, which is
    what makes it capturable.
    """

    def __init__(self, weights, embodiment_id: int, *, device="npu:0",
                 steps: int = NUM_STEPS, horizon: int = ACTION_HORIZON):
        self.device = device
        self.steps = int(steps)
        self.horizon = int(horizon)
        self.embodiment_id = int(embodiment_id)
        dev = device

        # ── VL adapter ────────────────────────────────────────────────
        self.vlln = (_bias(weights._vlln_w, dev), _bias(weights._vlln_b, dev))
        self.vlsa = []
        for i in range(VLSA_LAYERS):
            self.vlsa.append(dict(
                norm1=(_bias(weights._vlsa_norm1_w[i], dev), _bias(weights._vlsa_norm1_b[i], dev)),
                q=(_nz(weights._vlsa_q_w[i], dev), _bias(weights._vlsa_q_b[i], dev)),
                k=(_nz(weights._vlsa_k_w[i], dev), _bias(weights._vlsa_k_b[i], dev)),
                v=(_nz(weights._vlsa_v_w[i], dev), _bias(weights._vlsa_v_b[i], dev)),
                o=(_nz(weights._vlsa_o_w[i], dev), _bias(weights._vlsa_o_b[i], dev)),
                norm3=(_bias(weights._vlsa_norm3_w[i], dev), _bias(weights._vlsa_norm3_b[i], dev)),
                fc1=(_nz(weights._vlsa_fc1_w[i], dev), _bias(weights._vlsa_fc1_b[i], dev)),
                fc2=(_nz(weights._vlsa_fc2_w[i], dev), _bias(weights._vlsa_fc2_b[i], dev)),
            ))

        # ── per-embodiment encoders and decoder ───────────────────────
        slot = self.embodiment_id
        self.state_encoder = (
            (_nz(weights._st_enc_l1_W[slot], dev), _bias(weights._st_enc_l1_b[slot], dev)),
            (_nz(weights._st_enc_l2_W[slot], dev), _bias(weights._st_enc_l2_b[slot], dev)))
        self.action_encoder = (
            (_nz(weights._ac_enc_W1_W[slot], dev), _bias(weights._ac_enc_W1_b[slot], dev)),
            (_nz(weights._ac_enc_W2_W[slot], dev), _bias(weights._ac_enc_W2_b[slot], dev)),
            (_nz(weights._ac_enc_W3_W[slot], dev), _bias(weights._ac_enc_W3_b[slot], dev)))
        self.action_decoder = (
            (_nz(weights._ac_dec_l1_W[slot], dev), _bias(weights._ac_dec_l1_b[slot], dev)),
            (_nz(weights._ac_dec_l2_W[slot], dev), _bias(weights._ac_dec_l2_b[slot], dev)))

        # Action-token position embeddings, already summed over the horizon.
        self.position_embedding = _bias(
            weights._ah_pos_embed_w[:self.horizon], dev).unsqueeze(0)

        # ── DiT layers ────────────────────────────────────────────────
        self.layers = []
        for i in range(DIT_LAYERS):
            self.layers.append(dict(
                cross=(i % 2 == 0),
                text=(i % (2 * ATTEND_TEXT_EVERY_N_BLOCKS) == 0),
                q=(_nz(weights._dit_q_w[i], dev), _bias(weights._dit_q_b[i], dev)),
                k=(_nz(weights._dit_k_w[i], dev), _bias(weights._dit_k_b[i], dev)),
                v=(_nz(weights._dit_v_w[i], dev), _bias(weights._dit_v_b[i], dev)),
                o=(_nz(weights._dit_o_w[i], dev), _bias(weights._dit_o_b[i], dev)),
                ff1=(_nz(weights._dit_ff_proj_w[i], dev), _bias(weights._dit_ff_proj_b[i], dev)),
                ff2=(_nz(weights._dit_ff_down_w[i], dev), _bias(weights._dit_ff_down_b[i], dev)),
            ))

        self._build_modulators(weights)

    # ------------------------------------------------------------------
    def _build_modulators(self, weights):
        """Collapse everything that depends only on the denoise step.

        The timestep embedding, each layer's AdaLN ``(shift, scale)`` and the
        output head's pair are functions of the step index alone. Folding the
        pair into a LayerNorm's affine parameters is exact: the norm carries no
        affine of its own, so ``norm(x) * (1 + scale) + shift`` and
        ``layer_norm(x, weight=1+scale, bias=shift)`` are the same expression.
        """
        dev = self.device
        proj = _timestep_projection(self.steps).to(torch.float32)
        lin1_w = weights._ts_lin1_w.to(torch.float32)     # (256, 1536)
        lin2_w = weights._ts_lin2_w.to(torch.float32)     # (1536, 1536)
        temb = proj @ lin1_w + weights._ts_lin1_b.float()
        temb = F.silu(temb) @ lin2_w + weights._ts_lin2_b.float()
        temb = temb.to(torch.bfloat16)                    # (steps, 1536)

        conditioned = F.silu(temb.float())
        self.ada = []
        for i in range(DIT_LAYERS):
            mod = conditioned @ weights._dit_ada_w[i].float() + weights._dit_ada_b[i].float()
            scale, shift = mod.chunk(2, dim=1)
            self.ada.append((
                (1.0 + scale).to(torch.bfloat16).to(dev).contiguous(),
                shift.to(torch.bfloat16).to(dev).contiguous()))

        out = conditioned @ weights._proj_out_1_w.float() + weights._proj_out_1_b.float()
        shift, scale = out.chunk(2, dim=1)
        self.out_norm = ((1.0 + scale).to(torch.bfloat16).to(dev).contiguous(),
                         shift.to(torch.bfloat16).to(dev).contiguous())
        self.proj_out_2 = (_nz(weights._proj_out_2_w, dev),
                           _bias(weights._proj_out_2_b, dev))

        # The action encoder's tau features are a step constant too.
        self.tau = _tau_encoding(self.steps, DIT_DIM).to(torch.bfloat16).to(dev)


# ══════════════════════════════════════════════════════════════════════
#  Frame forward
# ══════════════════════════════════════════════════════════════════════

def encode_backbone_features(bound: BoundChain, features: torch.Tensor) -> torch.Tensor:
    """vlln + the 4-layer VL self-attention adapter, over 461 tokens."""
    x = F.layer_norm(features, (BACKBONE_DIM,), bound.vlln[0], bound.vlln[1], EPS)
    for layer in bound.vlsa:
        h = F.layer_norm(x, (BACKBONE_DIM,), layer["norm1"][0], layer["norm1"][1], EPS)
        q = layer["q"][0](h, layer["q"][1])
        k = layer["k"][0](h, layer["k"][1])
        v = layer["v"][0](h, layer["v"][1])
        a = attention(q, k, v, VLSA_HEADS, VLSA_HEAD_DIM)
        x = x + layer["o"][0](a, layer["o"][1])
        h = F.layer_norm(x, (BACKBONE_DIM,), layer["norm3"][0], layer["norm3"][1], EPS)
        h = F.gelu(layer["fc1"][0](h, layer["fc1"][1]), approximate="tanh")
        x = x + layer["fc2"][0](h, layer["fc2"][1])
    return x


def encode_state(bound: BoundChain, state: torch.Tensor) -> torch.Tensor:
    """(B, history, state_dim) -> (B, 1, dit_dim)."""
    flat = state.reshape(state.shape[0], 1, -1)
    (w1, b1), (w2, b2) = bound.state_encoder
    return w2(F.relu(w1(flat, b1)), b2)


def cross_key_values(bound: BoundChain, vl: torch.Tensor):
    """K/V for every cross-attention layer, once per frame.

    These depend on the backbone features alone. The reference recomputes all
    32 projections inside each of the four denoise steps; here the four steps
    read one table.
    """
    cache = []
    for layer in bound.layers:
        if not layer["cross"]:
            cache.append(None)
            continue
        cache.append((layer["k"][0](vl, layer["k"][1]),
                      layer["v"][0](vl, layer["v"][1])))
    return cache


def encode_actions(bound: BoundChain, actions: torch.Tensor, step: int) -> torch.Tensor:
    """The multi-embodiment action encoder with its tau features precomputed."""
    (w1, b1), (w2, b2), (w3, b3) = bound.action_encoder
    a = w1(actions, b1)
    tau = bound.tau[step].expand(a.shape[0], a.shape[1], DIT_DIM)
    x = F.silu(w2(torch.cat((a, tau), dim=-1), b2))
    return w3(x, b3)


def dit_layer(bound: BoundChain, index: int, step: int, x: torch.Tensor,
              cross_kv, masks) -> torch.Tensor:
    layer = bound.layers[index]
    gamma, beta = bound.ada[index]
    h = F.layer_norm(x, (DIT_DIM,), gamma[step], beta[step], EPS)
    q = layer["q"][0](h, layer["q"][1])
    if layer["cross"]:
        key, value = cross_kv[index]
        mask = masks[0] if layer["text"] else masks[1]
        a = attention(q, key, value, DIT_HEADS, DIT_HEAD_DIM, mask)
    else:
        k = layer["k"][0](h, layer["k"][1])
        v = layer["v"][0](h, layer["v"][1])
        a = attention(q, k, v, DIT_HEADS, DIT_HEAD_DIM)
    x = x + layer["o"][0](a, layer["o"][1])
    h = F.layer_norm(x, (DIT_DIM,), None, None, EPS)
    h = F.gelu(layer["ff1"][0](h, layer["ff1"][1]), approximate="tanh")
    return x + layer["ff2"][0](h, layer["ff2"][1])


def dit(bound: BoundChain, hidden: torch.Tensor, step: int, cross_kv, masks):
    for index in range(DIT_LAYERS):
        hidden = dit_layer(bound, index, step, hidden, cross_kv, masks)
    gamma, beta = bound.out_norm
    hidden = F.layer_norm(hidden, (DIT_DIM,), gamma[step], beta[step], NORM_OUT_EPS)
    return bound.proj_out_2[0](hidden, bound.proj_out_2[1])


def denoise(bound: BoundChain, vl: torch.Tensor, state_features: torch.Tensor,
            masks, noise: torch.Tensor) -> torch.Tensor:
    """Four Euler steps over the DiT, from a supplied initial noise draw."""
    cross_kv = cross_key_values(bound, vl)
    (dw1, db1), (dw2, db2) = bound.action_decoder
    actions = noise
    dt = 1.0 / bound.steps
    for step in range(bound.steps):
        features = encode_actions(bound, actions, step) + bound.position_embedding
        hidden = torch.cat((state_features, features), dim=1)
        out = dit(bound, hidden, step, cross_kv, masks)
        pred = dw2(F.relu(dw1(out, db1)), db2)
        actions = actions + dt * pred[:, -bound.horizon:]
    return actions


def attention_masks(image_mask: torch.Tensor, attention_mask: torch.Tensor,
                    queries: int):
    """The two key masks the cross-attention layers alternate between.

    PFA masks out where the entry is True, the opposite of the reference's
    keep-mask, and it wants the query axis materialised rather than broadcast.
    Both masks are frame constants, so both are built once.
    """
    keep_text = (~image_mask) & attention_mask
    keep_image = image_mask & attention_mask
    def expand(keep):
        return (~keep).reshape(1, -1).expand(queries, keep.shape[-1]).contiguous()
    return expand(keep_text), expand(keep_image)
