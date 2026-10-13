"""FlashRT -- DM05 (LIBERO VLA) compute pipeline, SM120.

DM05 is a Pi0.5-shaped model, so this is a variant of the pi05 pipeline rather
than a new model package: the encoder and decoder widths are the ones
``pipeline_rtx`` already declares. What differs, and what lives here, is the
SigLIP tower at 448 px with two views and a head width of 72, for which FA2 has
no tile -- the shared attention set instantiates head dimensions 64, 96, 128 and
256, so the 72 is padded to 96 and a quarter of the MMA work lands on the pad.
``flash_rt_dm05`` supplies a kernel that reduces over the real 72.

Layering, per the repository's frontend/pipeline split: this module owns the
compute topology only. It is given weights and buffers that a frontend already
allocated and owns, it allocates nothing, it reads nothing back to the host, and
every stage runs on buffers whose addresses are stable, so the whole thing
captures into one CUDA graph.

Every weight in ``weights`` is ``(N, K)``, which is the layout
``GemmRunner::bf16_nt_bias`` and ``bf16_run`` take, so no repacking happens
between a checkpoint and a run. For the ``nn.Linear`` modules that is the tensor
as stored; the projector's ``mm_input_projection_weight`` is the one exception,
held by Gemma3MultiModalProjector as ``(vision_hidden, text_hidden)`` i.e.
``(K, N)``, so the frontend transposes it once at load.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# SigLIP-L/14-448 vision tower, two views. These are properties of the
# checkpoint's vision config, asserted by the frontend against what it loaded.
VIS_P = 14            # patch size
VIS_IMG = 448         # input resolution
VIS_GRID = VIS_IMG // VIS_P          # 32
VIS_SEQ_PER_VIEW = VIS_GRID * VIS_GRID   # 1024
VIS_L = 27            # layers
VIS_D = 1152
VIS_FF = 4304
VIS_NH = 16
VIS_HD = VIS_D // VIS_NH             # 72
VIS_TOK_SIDE = 16     # after the 2x2 average pool
PATCH_FEAT = 3 * VIS_P * VIS_P       # 588


class Dm05VisionStage:
    """SigLIP tower + multimodal projector, reading ``pixels`` into ``out``.

    ``weights`` holds one dict per layer with the checkpoint's own tensors, plus
    the tower-level ones; ``bufs`` holds the preallocated scratch. Neither is
    copied or reshaped here.

    The attention path is fixed at the head-72 kernel. The alternative -- FA2's
    hdim-96 instantiation over the same heads -- is a different reduction order
    and was measured to move a 444-observation action-cosine minimum, so a
    caller that wants it needs its own numbers rather than a flag here.
    """

    def __init__(self, weights, bufs, *, num_views=2, eps=1e-6, proj_eps=1e-6,
                 fvk=None, dm05=None, gemm=None):
        from flash_rt import flash_rt_kernels as _fvk
        from flash_rt import flash_rt_dm05 as _dm05
        self.fvk = fvk or _fvk
        self.dm05 = dm05 or _dm05
        self.gemm = gemm or self.fvk.GemmRunner()

        self.nv = num_views
        self.L = VIS_L
        self.D = VIS_D
        self.FF = VIS_FF
        self.NH = VIS_NH
        self.HD = VIS_HD
        self.P = VIS_P
        self.img = VIS_IMG
        self.grid = VIS_GRID
        self.seq_per_view = VIS_SEQ_PER_VIEW
        self.S = self.nv * self.seq_per_view
        self.pool = self.grid // VIS_TOK_SIDE      # 2
        self.eps = eps
        self.proj_eps = proj_eps
        self.scale = self.HD ** -0.5
        self.weights = weights
        self.bufs = bufs
        self.out_dim = int(weights["mm_proj_w"].shape[0])   # (N, K) == (out_dim, D)
        self.tok_per_view = VIS_TOK_SIDE * VIS_TOK_SIDE
        self.T = self.nv * self.tok_per_view
        self._check()

    def _check(self) -> None:
        """Fail fast on a checkpoint that is not the shape this stage computes.

        Every one of these is load-bearing and none of them is checked by a
        kernel: a mismatch would either raise somewhere far from here or, for
        the sequence length, quietly produce a different number of image tokens
        than the prefix expects.
        """
        w = self.weights
        problems = []
        if w["w_patch"].shape != (VIS_D, PATCH_FEAT):
            problems.append(f"patch weight {tuple(w['w_patch'].shape)} != {(VIS_D, PATCH_FEAT)}")
        if len(w["layers"]) != VIS_L:
            problems.append(f"{len(w['layers'])} vision layers, expected {VIS_L}")
        for i, ly in enumerate(w["layers"]):
            if ly["qkv_w"].shape != (3 * VIS_D, VIS_D):
                problems.append(f"layer {i} qkv weight {tuple(ly['qkv_w'].shape)}")
                break
        if w["mm_proj_w"].shape != (self.out_dim, VIS_D):
            problems.append(f"projector weight {tuple(w['mm_proj_w'].shape)}")
        if self.S % 128 != 0:
            problems.append(f"sequence {self.S} is not a multiple of 128, which the "
                            "head-72 attention requires")
        if self.nv != 2:
            problems.append(f"{self.nv} views; this checkpoint's prefixes assume 2")
        if problems:
            raise ValueError("Dm05VisionStage: " + "; ".join(problems))

    def forward(self, stream: int = 0) -> None:
        """Run ``bufs['pixels']`` -> ``bufs['out']`` on the caller's stream."""
        fvk, d, g = self.fvk, self.dm05, self.gemm
        b, w = self.bufs, self.weights
        S, D, FF, P = self.S, self.D, self.FF, self.P
        p = lambda t: t.data_ptr()

        fvk.patch_im2col_nchw_bf16(p(b["pixels"]), p(b["patches"]), self.nv, 3,
                                   self.img, self.img, P, stream)
        g.bf16_nt_bias(p(b["patches"]), p(w["w_patch"]), p(b["x"]), p(w["b_patch"]),
                       S, D, PATCH_FEAT, stream)
        fvk.residual_add(p(b["x"]), p(w["pos"]), S * D, stream)

        for i, ly in enumerate(w["layers"]):
            fvk.layer_norm(p(b["x"]), p(ly["ln1_w"]), p(ly["ln1_b"]), p(b["xn"]),
                           S, D, self.eps, stream)
            g.bf16_nt_bias(p(b["xn"]), p(ly["qkv_w"]), p(b["qkv"]), p(ly["qkv_b"]),
                           S, 3 * D, D, stream)
            d.dm05_vit_attn_hd72(p(b["qkv"]), 3 * D, D, p(b["attn"]), D,
                            self.seq_per_view, self.NH, self.nv, self.scale, 2, stream, 0, 0)
            # The reference carries each projection's bias in the GEMM's own fp32
            # epilogue and then runs a plain residual add / GELU. Folding the bias
            # into that elementwise op instead is a different rounding -- it would
            # round the accumulator before adding the bias rather than after -- and
            # the difference survives all 27 layers, so the split is kept as is.
            g.bf16_nt_bias(p(b["attn"]), p(ly["o_w"]), p(b["tmp"]), p(ly["o_b"]),
                           S, D, D, stream)
            fvk.residual_add(p(b["x"]), p(b["tmp"]), S * D, stream)

            fvk.layer_norm(p(b["x"]), p(ly["ln2_w"]), p(ly["ln2_b"]), p(b["xn"]),
                           S, D, self.eps, stream)
            g.bf16_nt_bias(p(b["xn"]), p(ly["fc1_w"]), p(b["h"]), p(ly["fc1_b"]),
                           S, FF, D, stream)
            fvk.gelu_inplace(p(b["h"]), S * FF, stream)
            g.bf16_nt_bias(p(b["h"]), p(ly["fc2_w"]), p(b["tmp"]), p(ly["fc2_b"]),
                           S, D, FF, stream)
            fvk.residual_add(p(b["x"]), p(b["tmp"]), S * D, stream)

        fvk.layer_norm(p(b["x"]), p(w["post_w"]), p(w["post_b"]), p(b["xn"]),
                       S, D, self.eps, stream)
        fvk.avg_pool_vision_tokens(p(b["xn"]), p(b["pooled"]), self.nv,
                                   self.grid, self.grid, D, self.pool, stream)
        fvk.gemma3_rms_norm_bf16(p(b["pooled"]), D, p(w["mm_norm_w"]), p(b["pooled_n"]), D,
                                 0, D, self.T, self.T, 0, D, self.proj_eps, stream)
        g.bf16_run(p(b["pooled_n"]), p(w["mm_proj_w"]), p(b["out"]),
                   self.T, self.out_dim, D, stream)


# Gemma3 text tower, the VLM's language model -- the prefix the action expert
# reads. These are the checkpoint's own numbers (``vlm_config.text_config``); the
# frontend asserts them against what it loaded, because nothing here is checked by
# a kernel.
ENC_L = 34            # layers
ENC_D = 2560
ENC_H = 10240         # FFN hidden
ENC_NH = 8            # query heads (GQA)
ENC_NKV = 4           # kv heads
ENC_HD = 256          # head dim, and the head dim the qk-norm+RoPE kernel is written for
ENC_EPS = 1e-6


class Dm05PrefixStage:
    """Gemma3 decoder stack: merged prefix embeddings -> per-layer K/V cache.

    One ``forward`` fills the cache the action expert reads. The prompt is bound
    separately, by :meth:`set_prompt`, because its length and its token-type run
    structure are properties of the request rather than of the weights.

    This is a variant of the pi05 encoder rather than a copy of it: the pi05
    encoder is 18 layers of Gemma-2B at D=2048 with a single KV head and
    interleaved rope weights, where this is 34 layers of Gemma3 at D=2560 with
    four KV heads, per-head q/k RMSNorm, HF's rotate-half rope and a sliding/full
    layer-type split. Nothing in ``pipeline_rtx.py`` computes any of those.

    The attention is not ported, because it does not need to be: the reference
    already calls this repository's ``flash_rt_fa2`` for every one of its
    attention sites. What this stage reproduces is the scaffolding around it.

    Allocations: none. The K/V cache, the FA2 split scratch and the prompt
    mirrors all arrive in ``bufs``, so a captured graph replays on fixed
    addresses.
    """

    def __init__(self, weights, bufs, *, max_seq, eps=ENC_EPS, fvk=None, gemm=None,
                 fa2=None):
        from flash_rt import flash_rt_kernels as _fvk
        self.fvk = fvk or _fvk
        self.gemm = gemm or self.fvk.GemmRunner()
        if fa2 is None:
            from flash_rt import flash_rt_fa2 as fa2
        self.fa2 = fa2

        self.L = ENC_L
        self.D = ENC_D
        self.FF = ENC_H
        self.NH = ENC_NH
        self.NKV = ENC_NKV
        self.HD = ENC_HD
        self.eps = eps
        self.HQ = ENC_NH * ENC_HD
        self.HK = ENC_NKV * ENC_HD
        self.scale = float(weights["scale"])
        self.embed_scale = float(weights["embed_scale"])
        self.max_seq = int(max_seq)
        self.nsms = int(weights["nsms"])
        self.seq = 0
        self.segments = []
        self.weights = weights
        self.bufs = bufs
        self.fa2_split = bool(weights.get("fa2_prefix_split", True))
        self.splits = int(weights.get("fa2_splits", 0))
        self._check()

    def _check(self) -> None:
        w = self.weights
        problems = []
        if w["embed_w"].shape[1] != ENC_D:
            problems.append(f"embedding width {w['embed_w'].shape[1]} != {ENC_D}")
        if len(w["layers"]) != ENC_L:
            problems.append(f"{len(w['layers'])} prefix layers, expected {ENC_L}")
        for i, ly in enumerate(w["layers"]):
            if ly["q"].shape != (self.HQ, ENC_D):
                problems.append(f"layer {i} q weight {tuple(ly['q'].shape)}")
                break
            if ly["k"].shape != (self.HK, ENC_D):
                problems.append(f"layer {i} k weight {tuple(ly['k'].shape)}")
                break
            if ly["q_norm"].numel() != ENC_HD or ly["k_norm"].numel() != ENC_HD:
                problems.append(f"layer {i} q/k norm is not {ENC_HD} wide")
                break
        for t in ("sliding_attention", "full_attention"):
            if t not in w["rope"]:
                problems.append(f"no {t} rope table")
        if self.max_seq % 8:
            problems.append(f"max_seq {self.max_seq} is not a multiple of 8, which the "
                            "vectorized norms require")
        if problems:
            raise ValueError("Dm05PrefixStage: " + "; ".join(problems))

    def set_prompt(self, ids, img_index, key_limits, segments) -> None:
        """Bind one prompt: token ids, the image-row map, per-row key limits, runs.

        ``ids`` is (seq,) int64, ``img_index`` (seq,) int32 with -1 on text rows,
        ``key_limits`` (seq,) int32, and ``segments`` is the run table
        ``(start, end, key_limit, causal)`` the reference's attention splits on.
        Every array is copied into a preallocated device buffer, so a replay
        afterwards does no host work.
        """
        from torch import from_numpy
        s = int(len(ids))
        if s > self.max_seq:
            raise ValueError(f"prefix length {s} exceeds the preallocated {self.max_seq}")
        if len(img_index) != s or len(key_limits) != s:
            raise ValueError(f"prompt is {s} rows but the index and limit tables are "
                             f"{len(img_index)} and {len(key_limits)}")
        # Validate before writing anything: a rejected run table must leave the
        # stage on its previous prompt rather than half bound to this one.
        for s0, s1, kk, causal in segments:
            if not (0 <= s0 < s1 <= s):
                raise ValueError(f"segment {(s0, s1, kk, causal)} is outside 1..{s}")
            if not (s1 <= kk <= s):
                raise ValueError(f"segment {(s0, s1, kk, causal)} reads keys past {s}")
        b = self.bufs
        b["ids"][:s].copy_(from_numpy(ids).to(b["ids"].device))
        b["img_index"][:s].copy_(from_numpy(img_index).to(b["img_index"].device))
        b["klim"][:s].copy_(from_numpy(key_limits).to(b["klim"].device))
        self.seq = s
        self.segments = [(int(a), int(b_), int(c), bool(d)) for a, b_, c, d in segments]

    def forward(self, image_tokens_ptr: int, stream: int = 0) -> None:
        """``bufs['ids']``/``bufs['img_index']`` + the projector output -> the cache.

        ``image_tokens_ptr`` is the projector output, (n_img, ENC_D) bf16, owned by
        the caller (the vision stage's ``out``).
        """
        fvk, g, fa2 = self.fvk, self.gemm, self.fa2
        b, w = self.bufs, self.weights
        S, D, FF, HD = self.seq, self.D, self.FF, self.HD
        p = lambda t: t.data_ptr()

        fvk.embedding_lookup_merge_bf16(p(b["ids"]), p(b["img_index"]), p(w["embed_w"]),
                                        image_tokens_ptr, p(b["x"]), S, D,
                                        self.embed_scale, stream)
        for i, ly in enumerate(w["layers"]):
            kc, vc = b["k_cache"][i], b["v_cache"][i]
            cos, sin = w["rope"][w["layer_types"][i]]
            # (1 + w) RMSNorm, vectorized reduction order -- the order the fused
            # qk-norm+RoPE kernel uses, which is not the one the strided kernel uses
            self._norm(b["x"], ly["in_norm"], b["xn"], S, D, stream)
            g.bf16_run(p(b["xn"]), p(ly["q"]), p(b["q"]), S, self.HQ, D, stream)
            g.bf16_run(p(b["xn"]), p(ly["k"]), p(kc), S, self.HK, D, stream)
            g.bf16_run(p(b["xn"]), p(ly["v"]), p(vc), S, self.HK, D, stream)
            # per-head q/k norm in place, then HF rotate-half rope: the reference
            # fuses the two, and this two-kernel split reproduces it because the
            # norm's rounding is the fused kernel's rounding
            self._qk_norm(b["q"], ly["q_norm"], S * self.NH, self.NH, self.HQ, stream)
            self._qk_norm(kc, ly["k_norm"], S * self.NKV, self.NKV, self.HK, stream)
            fvk.rope_hf_bf16(p(b["q"]), self.HQ, self.NH, HD, p(cos), p(sin), S, stream)
            fvk.rope_hf_bf16(p(kc), self.HK, self.NKV, HD, p(cos), p(sin), S, stream)
            self._attention(kc, vc, S, stream)
            g.bf16_run(p(b["attn"]), p(ly["o"]), p(b["tmp"]), S, D, self.HQ, stream)
            # x += post_attention_layernorm(o); the kernel adds the rounded norm
            self._residual_norm(b["tmp"], ly["post_attn"], b["x"], S, D, stream)

            self._norm(b["x"], ly["pre_ff"], b["xn"], S, D, stream)
            g.bf16_run(p(b["xn"]), p(ly["gate"]), p(b["g"]), S, FF, D, stream)
            g.bf16_run(p(b["xn"]), p(ly["up"]), p(b["u"]), S, FF, D, stream)
            fvk.gelu_mul_bf16(p(b["g"]), p(b["u"]), p(b["g"]), S * FF, stream)
            g.bf16_run(p(b["g"]), p(ly["down"]), p(b["tmp"]), S, D, FF, stream)
            self._residual_norm(b["tmp"], ly["post_ff"], b["x"], S, D, stream)

        self._norm(b["x"], w["final_norm"], b["last_hidden"], S, D, stream)

    # ------------------------------------------------------------------ helpers
    def _norm(self, x, weight, out, rows, dim, stream) -> None:
        """One full-width Gemma3 RMSNorm, vectorized order (the flat group form)."""
        self.fvk.gemma3_rms_norm_v8_bf16(x.data_ptr(), dim, weight.data_ptr(),
                                         out.data_ptr(), dim, 0, dim,
                                         rows, rows, 0, dim, self.eps, stream)

    def _residual_norm(self, x, weight, residual, rows, dim, stream) -> None:
        """``residual += bf16(norm(x))`` in one pass, no norm output written."""
        self.fvk.gemma3_rms_norm_v8_bf16(x.data_ptr(), dim, weight.data_ptr(),
                                         0, dim, residual.data_ptr(), dim,
                                         rows, rows, 0, dim, self.eps, stream)

    def _qk_norm(self, buf, weight, rows, heads, row_stride, stream) -> None:
        """Per-head RMSNorm in place: row r is head (r % heads) of sequence row
        (r / heads), which is what the group addressing encodes."""
        self.fvk.gemma3_rms_norm_v8_bf16(buf.data_ptr(), self.HD, weight.data_ptr(),
                                         buf.data_ptr(), self.HD, 0, self.HD,
                                         rows, heads, row_stride, self.HD, self.eps, stream)

    def _attention(self, kc, vc, S, stream) -> None:
        """The reference's own segmentation: one FA2 call per token-type run, each
        seeing keys up to its own limit, causal on text runs and bidirectional on
        image runs. Splitting exactly where the reference splits keeps the
        summation order, so the result is the one the reference produces."""
        b = self.bufs
        q, attn = b["q"], b["attn"]
        # FA2 indexes (batch, row, head), so these are the strides of the
        # (B, S, H, D) view: head stride HD, row stride HQ, and a zero batch
        # stride because every segment is a single batch.
        qs = (S * self.HQ, self.HQ, self.HD)
        ks = (S * self.HK, self.HK, self.HD)
        os_ = (S * self.HQ, self.HQ, self.HD)
        qp, kp, vp, op = (q.data_ptr(), kc.data_ptr(), vc.data_ptr(), attn.data_ptr())
        lp, lap, oap = (b["lse"].data_ptr(), b["lse_accum"].data_ptr(),
                        b["o_accum"].data_ptr())
        esz = q.element_size()
        for s0, s1, kk, causal in self.segments:
            # each segment carries its own query rows; K and V stay at the cache
            # base and are bounded by the segment's key limit instead
            off = s0 * self.HQ * esz
            nq_ = s1 - s0
            if causal:
                split = self.fa2_split
                # the causal wrapper takes the SM count and picks its own split
                # count, and unlike the seqused entry it needs no -inf pre-fill
                self.fa2.fwd_bf16_causal(qp + off, kp, vp, op + off, lp, lap, oap,
                                         1, nq_, kk, self.NH, self.NKV, self.HD,
                                         qs, ks, ks, os_, self.scale,
                                         self.nsms if split else 0, stream)
            else:
                # a bidirectional run is one pass per layer: no split, no accumulators
                self.fa2.fwd_bf16(qp + off, kp, vp, op + off, lp, 0, 0,
                                  1, nq_, kk, self.NH, self.NKV, self.HD,
                                  qs, ks, ks, os_, self.scale, 0, stream)


# Action expert, a 34-layer flow-matching head at the decoder width. Read from
# ``action_config`` in the checkpoint; the frontend asserts it against the prefix
# tower, which shares the layer count, the head layout and the layer-type split.
DEC_L = 34
DEC_D = 1024
DEC_H = 4096          # FFN hidden
DEC_NH = 8            # query heads (GQA)
DEC_NKV = 4           # kv heads
DEC_HD = 256
DEC_EPS = 1e-6
ACTION_DIM = 32
NUM_STEPS = 10


class Dm05ActionStage:
    """Action expert: the prefix K/V cache + a noise sample -> an action chunk.

    Ten Euler steps of a flow-matching head. The whole action state stays in fp32
    across the steps -- step 0 upcasts the bf16 noise -- and the two projections
    that bracket the loop are fp32 in and out, so nothing rounds the state between
    steps.

    Each layer writes its K and V into the rows of the prefix cache just past the
    prompt, and attends non-causally over ``[prefix | chunk]``: the suffix sees the
    whole prefix and its own rows, with no mask. That visible set is the same one
    the pi05 decoder uses; what differs is everything around it. Per layer there
    are two plain Gemma3 ``(1 + w)`` RMSNorms applied to the sublayer *outputs*
    before gating, on top of the two adaLN modulations, and the sublayer output is
    gated in bf16 with its own rounding rather than in the fp32 residual.

    ``prefix`` is a :class:`Dm05PrefixStage`; this stage reads and extends its
    cache rather than owning one. Allocations: none.
    """

    def __init__(self, weights, bufs, prefix, *, steps=NUM_STEPS, eps=DEC_EPS,
                 fvk=None, gemm=None, fa2=None):
        from flash_rt import flash_rt_kernels as _fvk
        self.fvk = fvk or _fvk
        self.gemm = gemm or self.fvk.GemmRunner()
        if fa2 is None:
            from flash_rt import flash_rt_fa2 as fa2
        self.fa2 = fa2

        self.prefix = prefix
        self.steps = int(steps)
        self.dt = -1.0 / self.steps
        self.L = DEC_L
        self.D = DEC_D
        self.FF = DEC_H
        self.NH = DEC_NH
        self.NKV = DEC_NKV
        self.HD = DEC_HD
        self.eps = eps
        self.HQ = DEC_NH * DEC_HD
        self.HK = DEC_NKV * DEC_HD
        self.T = int(weights["horizon"])
        self.A = int(weights["action_dim"])
        self.scale = float(weights["scale"])
        self.nsms = int(weights["nsms"])
        self.splits = int(weights["fa2_splits"])
        self.layer_types = list(weights["layer_types"])
        self.weights = weights
        self.bufs = bufs
        self._check()

    def _check(self) -> None:
        w = self.weights
        p = self.prefix
        problems = []
        if p.NKV != DEC_NKV or p.HD != DEC_HD or p.L != DEC_L:
            problems.append(f"prefix cache is {p.L}x{p.NKV}x{p.HD}, this stage needs "
                            f"{DEC_L}x{DEC_NKV}x{DEC_HD}")
        if len(w["layers"]) != DEC_L:
            problems.append(f"{len(w['layers'])} action layers, expected {DEC_L}")
        for i, ly in enumerate(w["layers"]):
            if ly["q"].shape != (self.HQ, DEC_D):
                problems.append(f"layer {i} q weight {tuple(ly['q'].shape)}")
                break
        if w["in_mod"].shape != (self.steps, DEC_L, 3 * DEC_D):
            problems.append(f"in_mod {tuple(w['in_mod'].shape)}")
        if w["mlp_mod"].shape != (self.steps, DEC_L, 3 * DEC_D):
            problems.append(f"mlp_mod {tuple(w['mlp_mod'].shape)}")
        if w["fin_mod"].shape != (self.steps, 3 * DEC_D):
            problems.append(f"fin_mod {tuple(w['fin_mod'].shape)}")
        if w["act_mask"].numel() != self.A:
            problems.append(f"action mask is {w['act_mask'].numel()} wide, not {self.A}")
        for t in ("sliding_attention", "full_attention"):
            if t not in w["rope"]:
                problems.append(f"no {t} rope table")
        if problems:
            raise ValueError("Dm05ActionStage: " + "; ".join(problems))

    def set_noise(self, noise) -> None:
        """Bind the initial noise (bf16) and its exact fp32 upcast, as the reference does."""
        b = self.bufs
        b["noise"].copy_(noise)
        b["noise_f32"].copy_(b["noise"])

    def forward(self, stream: int = 0) -> None:
        """``bufs['noise']`` -> ``bufs['x_t']``, fp32 actions, reading the prefix cache."""
        fvk, g, fa2 = self.fvk, self.gemm, self.fa2
        b, w = self.bufs, self.weights
        p = self.prefix
        T, D, FF, HD = self.T, self.D, self.FF, self.HD
        A = self.A
        ptr = lambda t: t.data_ptr()
        cache = p.bufs["k_cache"], p.bufs["v_cache"]

        for step in range(self.steps):
            if step == 0:
                fvk.copy_scaled_cols_f32(ptr(b["x_t"]), ptr(b["noise_f32"]),
                                         ptr(w["act_mask"]), T, A, stream)
            else:
                fvk.mul_cols_f32(ptr(b["x_t"]), ptr(w["act_mask"]), T, A, stream)
            fvk.linear_f32_in_bf16_out(ptr(b["x_t"]), ptr(w["w_in"]), ptr(w["b_in"]),
                                       ptr(b["h"]), T, D, A, stream)
            for i, ly in enumerate(w["layers"]):
                kc, vc = cache[0][i], cache[1][i]
                cos, sin = w["rope"][self.layer_types[i]]
                mod = w["in_mod"][step, i]
                if i == 0:
                    fvk.ada_rms_norm(ptr(b["h"]), ptr(mod), ptr(b["n"]), T, D, self.eps, stream)
                # the chunk's K/V land in the cache rows right after the prompt
                krow, vrow = kc[p.seq:p.seq + T], vc[p.seq:p.seq + T]
                g.bf16_run(ptr(b["n"]), ptr(ly["q"]), ptr(b["q"]), T, self.HQ, D, stream)
                g.bf16_run(ptr(b["n"]), ptr(ly["k"]), ptr(krow), T, self.HK, D, stream)
                g.bf16_run(ptr(b["n"]), ptr(ly["v"]), ptr(vrow), T, self.HK, D, stream)
                self._qk_norm_rope(b["q"], ly["q_norm"], T * self.NH, self.NH, self.HQ,
                                   cos[p.seq:p.seq + T], sin[p.seq:p.seq + T], T, stream)
                self._qk_norm_rope(krow, ly["k_norm"], T * self.NKV, self.NKV, kc.stride(0),
                                   cos[p.seq:p.seq + T], sin[p.seq:p.seq + T], T, stream)
                self._attention(kc, vc, p.seq + T, stream)
                g.bf16_run(ptr(b["attn"]), ptr(ly["o"]), ptr(b["tmp"]), T, D, self.HQ, stream)
                # h += bf16(bf16(norm(tmp)) * gate);  n = adaLN(h, mlp_mod)
                mmod = w["mlp_mod"][step, i]
                self._gate_norm(b["tmp"], ly["post_attn"], mod[2 * D:], b["h"], mmod,
                                b["n"], T, D, stream)
                g.bf16_run(ptr(b["n"]), ptr(ly["gate"]), ptr(b["g"]), T, FF, D, stream)
                g.bf16_run(ptr(b["n"]), ptr(ly["up"]), ptr(b["u"]), T, FF, D, stream)
                fvk.gelu_mul_bf16(ptr(b["g"]), ptr(b["u"]), ptr(b["g"]), T * FF, stream)
                g.bf16_run(ptr(b["g"]), ptr(ly["down"]), ptr(b["tmp"]), T, D, FF, stream)
                nmod = w["in_mod"][step, i + 1] if i + 1 < self.L else w["fin_mod"][step]
                self._gate_norm(b["tmp"], ly["post_ff"], mmod[2 * D:], b["h"], nmod,
                                b["n"], T, D, stream)
            fvk.linear_bf16_in_f32_out(ptr(b["n"]), ptr(w["w_out"]), ptr(w["b_out"]),
                                       ptr(b["v_t"]), T, A, D, stream)
            fvk.axpy_f32(ptr(b["x_t"]), ptr(b["v_t"]), self.dt, T * A, stream)

    # ------------------------------------------------------------------ helpers
    def _gate_norm(self, x, weight, gate, residual, mod, out, rows, dim, stream) -> None:
        """``residual = bf16(residual + bf16(bf16(norm(x)) * gate))`` and then the
        adaLN modulation of the updated residual, in one pass. Fused rather than
        the gate-residual kernel plus ``ada_rms_norm``: the reference reduces the
        second mean with this kernel's block shape and the standalone entry
        reduces it in the strided order, which differs on a few elements in ten
        thousand -- enough to survive the whole ten-step loop."""
        self.fvk.gemma3_norm_gate_residual_ada_v8_bf16(
            x.data_ptr(), weight.data_ptr(), gate.data_ptr(), residual.data_ptr(),
            mod.data_ptr(), out.data_ptr(), rows, dim, self.eps, stream)

    def _qk_norm_rope(self, buf, weight, rows, heads, row_stride, cos, sin, seq, stream) -> None:
        """Per-head ``(1 + w)`` RMSNorm in place, then HF rotate-half rope. The
        reference fuses the two; the split reproduces it because the norm's
        rounding is the fused kernel's rounding."""
        self.fvk.gemma3_rms_norm_v8_bf16(buf.data_ptr(), self.HD, weight.data_ptr(),
                                         buf.data_ptr(), self.HD, 0, self.HD,
                                         rows, heads, row_stride, self.HD, self.eps, stream)
        self.fvk.rope_hf_bf16(buf.data_ptr(), row_stride, heads, self.HD,
                              cos.data_ptr(), sin.data_ptr(), seq, stream)

    def _attention(self, kc, vc, sk, stream) -> None:
        """Non-causal over ``[prefix | chunk]``: one split FA2 pass, no mask. The
        split count is fixed so the fp32 partial-sum count -- and therefore the
        reduction order -- is the reference's."""
        b = self.bufs
        q, attn, T = b["q"], b["attn"], self.T
        qs = (T * self.HQ, self.HQ, self.HD)
        ks = (sk * self.HK, self.HK, self.HD)
        os_ = (T * self.HQ, self.HQ, self.HD)
        # No -inf pre-fill of `lse_accum` here, unlike the seqused entry: `sk` is
        # the real key count with no device-side truncation, so no split is ever
        # empty and every split writes its own lse slot.
        self.fa2.fwd_bf16(q.data_ptr(), kc.data_ptr(), vc.data_ptr(), attn.data_ptr(),
                          b["lse"].data_ptr(), b["lse_accum"].data_ptr(),
                          b["o_accum"].data_ptr(),
                          1, T, sk, self.NH, self.NKV, self.HD,
                          qs, ks, ks, os_, self.scale, self.splits, stream)
