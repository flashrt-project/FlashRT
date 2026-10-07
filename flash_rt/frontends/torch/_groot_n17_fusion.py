"""Shared, dtype-agnostic pieces of the GR00T N1.7 image→embeds fusion.

Why this module exists
----------------------
The fusion (patch embed → ViT → final merger → text-embed lookup → scatter the
image tokens into the LLM input) was first implemented inside
``GrootN17TorchFrontendThorFP8``. The Orin frontend cannot inherit any of it:
its MRO is ``Orin → RtxFP16 → Rtx → Thor → object`` and ``ThorFP8`` is **not**
an ancestor, so ``_fast_pos_embed_interpolate`` is simply unreachable from
there. Rather than copy the method a second time (it is ~40 lines of fiddly
index arithmetic where a silent transcription error costs an hour of debugging),
it lives here as a free function and both callers can use it.

This is additive: ``groot_n17_thor_fp8.py`` is untouched and keeps its own copy
of the method. Folding that copy into a call here is a separate, deliberate
refactor of a shipped Thor path and is not done as a side effect of the Orin
port.

The one function here is a pure-torch port of Qwen3-VL's
``Qwen3VLVisionModel.fast_pos_embed_interpolate``. It is verified
**bit-identical** against HF on real data, not merely close::

    patch_embed(pixel_values) + fast_pos_embed_interpolate(grid_thw)
        == pixel_features captured at visual.blocks[0]   →  max|d| = 0

That equality is the whole reason the fusion can be trusted: it says the
in-kernel chain reproduces HF's ViT input exactly, so any later divergence is
in the ViT/merger, not in the setup.
"""
from __future__ import annotations

__all__ = ["fast_pos_embed_interpolate"]


def fast_pos_embed_interpolate(pos_embed, grid_thw, *, device, merge: int = 2):
    """Bilinear interpolation of the ViT position-embedding table to a grid.

    Args:
        pos_embed: the checkpoint's ``visual.pos_embed`` table, ``(side**2, D)``
            (Qwen3-VL-2B: ``(2304, 1024)``, side = 48; bf16 on Orin, whose
            weight spec rewrites every ``ToFp16`` into ``ToBf16``). Any float
            dtype. The bilinear weights are formed in fp32 and rounded to
            ``pos_embed.dtype`` once, then the gather, the weighting and the
            4-way sum all run **in ``pos_embed.dtype``** — not in fp32 with a
            single final round, which is a measurably different answer. See the
            note in the body for why that matters.
        grid_thw: iterable of ``(t, h, w)`` ints, one row per image — the same
            tensor HF's vision tower receives.
        merge: ``spatial_merge_size``. The output is reordered into merge-block
            order (not raster order) so it lines up with the merger's later
            ``(Sv//merge**2, D*merge**2)`` view.

    Returns:
        ``(sum(t*h*w), D)`` on ``device``, dtype ``pos_embed.dtype``.

    Called **once per prompt**, not per frame: the grid is a property of the
    camera setup and resolution, so the result is a prompt constant that a CUDA
    graph can bake in as a pointer.
    """
    import torch

    out_dtype = pos_embed.dtype
    # Keep the table in its own dtype. HF builds its bilinear weight tensor
    # with ``dtype=self.pos_embed.weight.dtype`` and does the gather, multiply
    # and 4-way sum in that dtype (bf16 here); computing in fp32 and rounding
    # once at the end is NOT the same and measured max|d| = 0.125 (1 bf16 ULP
    # at the pos-embed's ~32 magnitude). That 1 ULP then rides through a
    # 24-layer residual tower: vit_block_17 went 0.998156 -> 0.995355 and
    # backbone_features 0.999729 -> 0.996150. Matching HF's arithmetic exactly
    # is what makes the fused ViT input bit-identical, which is the only reason
    # the fusion can be gated at all.
    pos_w = pos_embed.to(device)
    side = int(round(pos_w.shape[0] ** 0.5))
    if side * side != pos_w.shape[0]:
        raise ValueError(
            f"pos_embed has {pos_w.shape[0]} rows, which is not a perfect "
            f"square — cannot infer the table side length. Refusing to guess: "
            "an interpolated position embedding of the wrong shape would be "
            "added to every patch and only show up as a degraded cosine.")

    rows = [tuple(int(x) for x in r) for r in grid_thw]
    idx_list: list = [[] for _ in range(4)]
    wgt_list: list = [[] for _ in range(4)]
    for t, h, w in rows:
        # bilinear gather: 4 corners per output position
        h_i = torch.linspace(0, side - 1, h)
        w_i = torch.linspace(0, side - 1, w)
        hf, wf = h_i.int(), w_i.int()
        hc = (hf + 1).clip(max=side - 1)
        wc = (wf + 1).clip(max=side - 1)
        dh, dw = h_i - hf, w_i - wf
        bh, bhc = hf * side, hc * side
        inds = [(bh[None].T + wf[None]).flatten(),
                (bh[None].T + wc[None]).flatten(),
                (bhc[None].T + wf[None]).flatten(),
                (bhc[None].T + wc[None]).flatten()]
        wgts = [((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten()]
        for i in range(4):
            idx_list[i].extend(inds[i].tolist())
            wgt_list[i].extend(wgts[i].tolist())

    idx = torch.tensor(idx_list, dtype=torch.long, device=device)
    # dtype=out_dtype, not fp32: see the note above.
    wgt = torch.tensor(wgt_list, dtype=out_dtype, device=device)
    pe = pos_w[idx] * wgt[:, :, None]
    patch = pe[0] + pe[1] + pe[2] + pe[3]

    grids = [(h, w) for _, h, w in rows]
    patch = patch.split([h * w for h, w in grids])
    out = []
    for pe_g, (t, _, _), (h, w) in zip(patch, rows, grids):
        pe_g = pe_g.repeat(t, 1)
        # raster (h, w) -> merge-block order, matching the merger's later view
        pe_g = (pe_g.view(t, h // merge, merge, w // merge, merge, -1)
                .permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
        out.append(pe_g)
    return torch.cat(out)
