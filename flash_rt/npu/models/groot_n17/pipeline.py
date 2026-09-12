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

* **Each residual joins the norm that follows it.** ``npu_add_layer_norm``
  computes the sum and the normalisation of the sum in one launch and hands
  back both, and the sum it returns is bit-identical to the separate add. At
  the DiT's width that pair costs 7.66 us fused against 13.52 us apart, and it
  runs 256 times a frame -- twice per layer-step, once across each layer
  boundary. The loop is therefore carried as ``(residual, normalised)``: a
  layer is entered with its modulated norm already computed by its
  predecessor's residual.

* **Cross-attention K/V is computed once per frame, over one token class.**
  It is a function of the backbone features alone, and the reference
  recomputes all 32 of those projections on every one of the four steps. The
  reference also hands every cross layer all of the backbone tokens together
  with a mask that hides the ones it does not want; the token classes are a
  property of the prompt, so this path gathers them once and each layer
  projects and attends over only its own class. On the shipped observation
  geometry that is 13 keys for a text layer rather than 461 with 448 of them
  masked out.

Gathering the classes also means no attention mask is built or passed: every
key a layer is given is a key it attends. The vendor constraint that shapes
the rest: ``F.scaled_dot_product_attention``
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
from flash_rt.npu.models.groot_n17 import norm as fused
from flash_rt.npu.models.groot_n17.attention import DitAttention

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

    Row-major was measured against this at every shape the frame runs. It wins
    the backbone's wide projections by 3 to 9 percent in isolation and loses the
    whole frame by 2 percent, so the fractal layout serves every site.
    """
    return NzBf16Weight.bind(weight_kn.t().contiguous().to(torch.bfloat16).to(device))


def _bias(tensor, device):
    return tensor.to(torch.bfloat16).to(device).contiguous()


#: Input channels appended to a projection so its bias can be one more row of the
#: weight instead of an ``addmm`` bias. See ``_nz_folded``.
BIAS_CHANNEL = 16


def _nz_folded(weight_kn: torch.Tensor, bias: torch.Tensor, device) -> NzBf16Weight:
    """Bind a projection with its bias folded in as one more input channel.

    ``aclnn``'s biased matmul casts its bias on every call, as its own kernel
    launch: 512 of the frame's 711 ``Cast`` calls are the DiT's, for numbers that
    never change. Two ways out, and which one a site uses depends on what reads
    its result.

    Where the consumer is one of ours and already walks the row, it adds the bias
    -- that is ``o`` and ``ff2``, whose consumer is the fused add-and-normalise.
    Where the consumer is the cube or a vendor kernel, the bias folds in here: the
    activation's row ends in a constant one, so that channel's contribution *is*
    the bias. Identical to ``addmm`` at cosine 1.0000000, because both add it in
    FP32 and only the order differs, at the cost of one more 16-channel block of
    weight -- 1% of these shapes.
    """
    k, n = int(weight_kn.shape[0]), int(weight_kn.shape[1])
    padded = torch.zeros(k + BIAS_CHANNEL, n, dtype=torch.bfloat16)
    padded[:k] = weight_kn.to(torch.bfloat16)
    padded[k] = bias.to(torch.bfloat16)
    return _nz(padded, device)


def attention(query, key, value, heads, head_dim):
    """One capturable attention call in the layout the projections produce.

    No mask: every site on this path is given exactly the keys it attends.
    """
    import torch_npu
    return torch_npu.npu_prompt_flash_attention(
        query, key, value, num_heads=heads, num_key_value_heads=heads,
        scale_value=head_dim ** -0.5, input_layout="BSH",
        pre_tokens=_INT_MAX, next_tokens=_INT_MAX, sparse_mode=0)


class DitAttentionSite:
    """One DiT attention geometry: the native kernel plus its operand buffers.

    The kernel wants two things the projections do not produce on their own.
    Query rows are padded to the fractal, and key rows are padded *with zeros*
    -- a padded key has to score exactly zero for the kernel's closed-form
    padding correction to hold. Neither costs a kernel: the projections write
    into their padded buffers directly through ``addmm``'s ``out``, and the
    padding, written once at allocation, is never touched again.

    The value is the third operand and it goes in whichever way is free. A
    cross-attention layer's value is a frame constant, so it is transposed once
    a frame into the ``(heads * head_dim, keys)`` form the B operand takes
    directly. A self-attention layer's is not, and slicing and permuting it
    cost 17 us a layer for 147 KB, so it goes in as the projection wrote it and
    the kernel transposes it between L1 and L0B.
    """

    def __init__(self, queries: int, keys: int, device):
        self.kernel = DitAttention(DIT_HEADS, queries, keys, DIT_HEAD_DIM, device=device)
        self.queries, self.keys = int(queries), int(keys)
        self.rows, self.columns = self.kernel.rows, self.kernel.columns
        self.width, self.device = self.kernel.width, device
        self.query = torch.zeros(self.rows, self.width, dtype=torch.bfloat16,
                                 device=device)
        self.self_key, self.self_value = self.key_buffers()
        # Query, key and value of a self-attention site share one activation, so
        # they are one GEMM of three times the width. One call of 4608 columns
        # costs 26 us where three of 1536 cost 34, and the slices are read in
        # place rather than copied out -- which is what ate the same fusion on
        # the other model on this part.
        self.fused = torch.zeros(self.rows, 3 * self.width, dtype=torch.bfloat16,
                                 device=device)

    def key_buffers(self):
        """Zero-padded key and value buffers for one layer."""
        return tuple(torch.zeros(self.columns, self.width, dtype=torch.bfloat16,
                                 device=self.device) for _ in range(2))

    def query_into(self, weight, x, bias=None):
        """``bias`` is ``None`` wherever the projection folded it into its
        weight, which is every site whose activation is a padded normalised row."""
        flat = x.reshape(-1, x.shape[-1])
        out = self.query[:self.queries]
        if bias is None:
            torch.matmul(flat, weight.tensor, out=out)
        else:
            torch.addmm(bias, flat, weight.tensor, out=out)

    def key_into(self, weight, x, bias, key):
        torch.addmm(bias, x.reshape(-1, x.shape[-1]), weight.tensor,
                    out=key[:self.keys])

    def value_into(self, weight, x, bias, value):
        """The projection, padded, and then its transpose.

        Transposing the padded projection into a fresh contiguous tensor costs
        15 us; writing a permuted view into a padded destination -- which looks
        like the same thing and saves an allocation -- costs 45, because it
        lowers to a transpose followed by a scatter. The transpose itself is
        flat in the operand size, so this is a fixed cost either way.
        """
        torch.addmm(bias, x.reshape(-1, x.shape[-1]), weight.tensor,
                    out=value[:self.keys])
        return value.t().contiguous()

    def fused_into(self, weight, x, bias=None):
        """One projection for query, key and value, read back as three slices.

        All three slices stay in place. The value used to be permuted out of
        this buffer for the kernel's B operand, which is a slice and a transpose
        -- 3.6 and 13.8 us at 48 by 1536, both of them fixed cost -- and the
        kernel now transposes it on the way into L0B instead.
        """
        flat = x.reshape(-1, x.shape[-1])
        out = self.fused[:self.queries]
        if bias is None:
            torch.matmul(flat, weight.tensor, out=out)
        else:
            torch.addmm(bias, flat, weight.tensor, out=out)
        width = self.width
        return (self.fused[:, :width], self.fused[:, width:2 * width],
                self.fused[:, 2 * width:])

    def __call__(self, key, value, query=None, stride=None, value_stride=0):
        query = self.query if query is None else query
        return self.kernel(query, key, value, stride,
                           value_stride)[:self.queries].unsqueeze(0)


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
            cross = i % 2 == 0
            layer = dict(
                cross=cross,
                text=(i % (2 * ATTEND_TEXT_EVERY_N_BLOCKS) == 0),
                # The attention projections and the first feed-forward hand
                # their result to the cube or to the vendor's GELU, so they carry
                # their bias as an extra input channel. The output projection and
                # the second feed-forward hand theirs to the fused
                # add-and-normalise, which adds the bias itself for nothing.
                q=(_nz_folded(weights._dit_q_w[i], weights._dit_q_b[i], dev), None),
                o=(_nz(weights._dit_o_w[i], dev), _bias(weights._dit_o_b[i], dev)),
                ff1=(_nz_folded(weights._dit_ff_proj_w[i],
                                weights._dit_ff_proj_b[i], dev), None),
                ff2=(_nz(weights._dit_ff_down_w[i], dev),
                     _bias(weights._dit_ff_down_b[i], dev)),
            )
            if cross:
                # A cross layer's key and value are frame constants over the
                # backbone tokens, so they stay separate; only a self layer's
                # three projections share an activation.
                layer["k"] = (_nz(weights._dit_k_w[i], dev),
                              _bias(weights._dit_k_b[i], dev))
                layer["v"] = (_nz(weights._dit_v_w[i], dev),
                              _bias(weights._dit_v_b[i], dev))
            else:
                # The checkpoint stores these transposed, so the three
                # projections concatenate along the output axis, which is dim 1
                # here and dim 0 after _nz transposes. Only the fused form is
                # kept: holding both costs 226 MB for nothing.
                layer["qkv"] = (
                    _nz_folded(torch.cat((weights._dit_q_w[i], weights._dit_k_w[i],
                                          weights._dit_v_w[i]), dim=1),
                               torch.cat((weights._dit_q_b[i], weights._dit_k_b[i],
                                          weights._dit_v_b[i]), dim=0), dev),
                    None)
                del layer["q"]
            self.layers.append(layer)

        # An affine-free LayerNorm synthesises unit weight and zero bias on
        # every call -- two extra launches and 2.5 us a call, measured. These
        # hold them instead.
        self.unit = torch.ones(DIT_DIM, dtype=torch.bfloat16, device=dev)
        self.zero = torch.zeros(DIT_DIM, dtype=torch.bfloat16, device=dev)

        # Every normalised row in the DiT lands here, at a pitch, with the last
        # channel a constant one so that the projections reading it carry their
        # bias in the weight (`_nz_folded`). One buffer serves all of them: the
        # block's two norms and the entry norm are strictly sequential, and each
        # row is consumed before the next is written.
        self.norm_pitch = DIT_DIM + BIAS_CHANNEL

        # The DiT is entered with the state token in front of the action
        # horizon, so every attention site on this path has that many queries.
        self.action_tokens = self.horizon + 1
        self.normalised = torch.zeros(1, self.action_tokens, self.norm_pitch,
                                      dtype=torch.bfloat16, device=dev)
        self.normalised[..., DIT_DIM] = 1.0
        # The entry norm has no residual branch of its own.
        self.no_branch = torch.zeros(1, self.action_tokens, DIT_DIM,
                                     dtype=torch.bfloat16, device=dev)
        # Attention sites and each cross layer's key/value pair are built the
        # first time they are asked for, which is during warm-up: capture runs
        # after three eager passes, so the graph never sees an allocation.
        self._sites: dict[int, DitAttentionSite] = {}
        self._cross_buffers: dict[int, tuple] = {}

        self._build_modulators(weights)

    # ------------------------------------------------------------------
    def site(self, keys: int) -> DitAttentionSite:
        site = self._sites.get(int(keys))
        if site is None:
            site = DitAttentionSite(self.action_tokens, int(keys), self.device)
            self._sites[int(keys)] = site
        return site

    def cross_buffers(self, index: int, site: DitAttentionSite):
        entry = self._cross_buffers.get(int(index))
        if entry is None:
            entry = site.key_buffers()
            self._cross_buffers[int(index)] = entry
        return entry

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
        # The output head reads the last block's normalised row, which is padded
        # like every other one, so it folds its bias in too.
        self.proj_out_2 = (_nz_folded(weights._proj_out_2_w,
                                      weights._proj_out_2_b, dev), None)

        # The action encoder's tau features are a step constant too.
        self.tau = _tau_encoding(self.steps, DIT_DIM).to(torch.bfloat16).to(dev)


# ══════════════════════════════════════════════════════════════════════
#  Frame forward
# ══════════════════════════════════════════════════════════════════════

def add_norm(residual, branch, weight, bias, eps, branch_bias=None, out=None):
    """``branch + residual``, and the LayerNorm of the sum, in one launch.

    Returns ``(normalised, sum)``. The sum is bit-identical to the separate
    add; the normalisation rounds once in the kernel where the separate pair
    rounds twice, which the end-to-end cosine judges.

    ``branch_bias`` is the bias of the projection that produced ``branch``, which
    this adds instead of that projection -- the same number into the same FP32
    sum, and one launch fewer. ``out`` is the padded destination described in
    ``vector.add_layer_norm``.
    """
    if fused.serves(residual) and fused.serves(branch):
        return fused.add_layer_norm(residual, branch, weight, bias, eps, branch_bias, out)
    import torch_npu
    if branch_bias is not None:
        branch = branch + branch_bias
    normalised, _, _, total = torch_npu.npu_add_layer_norm(
        residual, branch, weight, bias, eps, True)
    if out is not None:
        out[..., :normalised.shape[-1]] = normalised
        normalised = out
    return normalised, total


def encode_backbone_features(bound: BoundChain, features: torch.Tensor) -> torch.Tensor:
    """vlln + the 4-layer VL self-attention adapter, over the whole sequence."""
    x = F.layer_norm(features, (BACKBONE_DIM,), bound.vlln[0], bound.vlln[1], EPS)
    h = F.layer_norm(x, (BACKBONE_DIM,), bound.vlsa[0]["norm1"][0],
                     bound.vlsa[0]["norm1"][1], EPS)
    for index, layer in enumerate(bound.vlsa):
        q = layer["q"][0](h, layer["q"][1])
        k = layer["k"][0](h, layer["k"][1])
        v = layer["v"][0](h, layer["v"][1])
        # 461 tokens puts this site in the backbone's regime, where the fused
        # infer-attention entry point measures 1.31x the prompt one.
        from flash_rt.npu.models.groot_n17.backbone import attention as wide_attention

        a = wide_attention(q, k, v, VLSA_HEADS, VLSA_HEAD_DIM)
        h, x = add_norm(x, layer["o"][0](a, layer["o"][1]),
                        layer["norm3"][0], layer["norm3"][1], EPS)
        h = F.gelu(layer["fc1"][0](h, layer["fc1"][1]), approximate="tanh")
        branch = layer["fc2"][0](h, layer["fc2"][1])
        if index + 1 < len(bound.vlsa):
            following = bound.vlsa[index + 1]["norm1"]
            h, x = add_norm(x, branch, following[0], following[1], EPS)
        else:
            x = x + branch
    return x


def encode_state(bound: BoundChain, state: torch.Tensor) -> torch.Tensor:
    """(B, history, state_dim) -> (B, 1, dit_dim)."""
    flat = state.reshape(state.shape[0], 1, -1)
    (w1, b1), (w2, b2) = bound.state_encoder
    return w2(F.relu(w1(flat, b1)), b2)


def cross_key_values(bound: BoundChain, text: torch.Tensor, image: torch.Tensor):
    """K/V for every cross-attention layer, once per frame and per class.

    These depend on the backbone features alone; the reference recomputes all
    32 projections inside each of the four denoise steps. Each layer is also
    given only the token class it attends, so a text layer's projection runs
    over the prompt's handful of language tokens rather than over the whole
    sequence with the image tokens masked away.
    """
    cache = []
    for index, layer in enumerate(bound.layers):
        if not layer["cross"]:
            cache.append(None)
            continue
        source = text if layer["text"] else image
        site = bound.site(source.shape[-2])
        key, value = bound.cross_buffers(index, site)
        site.key_into(layer["k"][0], source, layer["k"][1], key)
        value_t = site.value_into(layer["v"][0], source, layer["v"][1], value)
        cache.append((site, key, value_t))
    return cache


def encode_actions(bound: BoundChain, actions: torch.Tensor, step: int) -> torch.Tensor:
    """The multi-embodiment action encoder with its tau features precomputed."""
    (w1, b1), (w2, b2), (w3, b3) = bound.action_encoder
    a = w1(actions, b1)
    tau = bound.tau[step].expand(a.shape[0], a.shape[1], DIT_DIM)
    x = F.silu(w2(torch.cat((a, tau), dim=-1), b2))
    return w3(x, b3)


def dit_layer(bound: BoundChain, index: int, step: int, x: torch.Tensor,
              h: torch.Tensor, cross_kv):
    """One block, entered with its modulated norm ``h`` already computed.

    Returns the residual and the norm the *next* consumer needs, which is the
    following block's modulated norm or, after the last block, the output
    head's.
    """
    layer = bound.layers[index]
    if layer["cross"]:
        site, key, value_t = cross_kv[index]
        site.query_into(layer["q"][0], h, layer["q"][1])
        a = site(key, value_t)
    else:
        site = bound.site(h.shape[-2])
        query, key, value = site.fused_into(layer["qkv"][0], h, layer["qkv"][1])
        pitch = 3 * site.width
        a = site(key, value, query=query, stride=pitch, value_stride=pitch)
    h, x = add_norm(x, layer["o"][0](a), bound.unit, bound.zero, EPS,
                    branch_bias=layer["o"][1], out=bound.normalised)
    h = F.gelu(layer["ff1"][0](h), approximate="tanh")
    branch = layer["ff2"][0](h)
    if index + 1 < DIT_LAYERS:
        gamma, beta = bound.ada[index + 1]
        return add_norm(x, branch, gamma[step], beta[step], EPS,
                        branch_bias=layer["ff2"][1], out=bound.normalised)
    gamma, beta = bound.out_norm
    return add_norm(x, branch, gamma[step], beta[step], NORM_OUT_EPS,
                    branch_bias=layer["ff2"][1], out=bound.normalised)


def dit(bound: BoundChain, hidden: torch.Tensor, step: int, cross_kv):
    gamma, beta = bound.ada[0]
    # The same kernel every other norm in the block uses, with no branch: it
    # writes the padded row the first layer's projection wants, and using the
    # vendor's LayerNorm here instead would put a second norm implementation in
    # a loop that charges for every kernel type in it.
    h, _ = add_norm(hidden, bound.no_branch, gamma[step], beta[step], EPS,
                    out=bound.normalised)
    x = hidden
    for index in range(DIT_LAYERS):
        h, x = dit_layer(bound, index, step, x, h, cross_kv)
    return bound.proj_out_2[0](h)


def denoise(bound: BoundChain, text: torch.Tensor, image: torch.Tensor,
            state_features: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Four Euler steps over the DiT, from a supplied initial noise draw."""
    cross_kv = cross_key_values(bound, text, image)
    (dw1, db1), (dw2, db2) = bound.action_decoder
    actions = noise
    dt = 1.0 / bound.steps
    for step in range(bound.steps):
        features = encode_actions(bound, actions, step) + bound.position_embedding
        hidden = torch.cat((state_features, features), dim=1)
        out = dit(bound, hidden, step, cross_kv)
        pred = dw2(F.relu(dw1(out, db1)), db2)
        actions = actions + dt * pred[:, -bound.horizon:]
    return actions


def token_partition(image_mask: torch.Tensor, attention_mask: torch.Tensor):
    """Split the backbone sequence into the two classes the DiT cross-attends.

    A prompt's token layout does not change between frames, so this runs when
    the prompt is set and the captured graph reads the resulting index vectors.
    Padding is dropped here rather than masked later: a token outside
    ``attention_mask`` lands in neither class, so no site is ever handed a key
    it must then be told to ignore.
    """
    image = image_mask.reshape(-1).bool()
    attend = attention_mask.reshape(-1).bool()
    text_index = torch.nonzero((~image) & attend, as_tuple=False).reshape(-1)
    image_index = torch.nonzero(image & attend, as_tuple=False).reshape(-1)
    if text_index.numel() == 0 or image_index.numel() == 0:
        raise ValueError(
            "the DiT cross-attention alternates between language and image "
            f"tokens; this prompt has {text_index.numel()} language and "
            f"{image_index.numel()} image tokens")
    return text_index, image_index
