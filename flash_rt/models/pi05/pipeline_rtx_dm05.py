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
