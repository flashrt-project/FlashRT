"""FlashRT — Qwen3-VL position geometry builders (set-prompt precompute).

Pure-torch reimplementations of the three deterministic geometry steps
that Qwen3-VL needs before the kernel pipeline runs. They are evaluated
once per prompt (not on the decode hot path) and are independent of the
HuggingFace model object:

  * ``mrope_position_ids``  — the 3D (t, h, w) MRoPE position ids over the
    interleaved text + image-grid sequence;
  * ``mrope_cos_sin``       — interleaved-MRoPE cos/sin for the language
    stack (mrope_section [t, h, w]);
  * ``vision_rope_cos_sin`` — rotate_half 2D-RoPE cos/sin for the ViT
    patch grid;
  * ``vision_pos_embeds``   — bilinear-interpolated learned position
    embedding for the patch grid, in 2×2-merge token order.

Image and video inputs; the MRoPE builder mirrors HF's text/image/video
span walk (videos use timestamp alignment: the grid is split per frame so
each frame's temporal index is 0 and the inter-frame timestamp text tokens
carry the temporal information).
"""
from __future__ import annotations

import torch


def vision_segments(input_ids, image_grid_thw=None, video_grid_thw=None, *,
                    image_token_id, video_token_id, spatial_merge_size):
    """Walk the image/video token runs of a sequence in order.

    Videos are split into per-frame rows (t -> 1) so each frame is one
    segment. Returns a list of dicts, one per vision segment, in sequence
    order:

      {'span': (start, end),     # token indices of the run
       'grid': (t, h, w),        # per-frame grid (t == 1 for video frames)
       'kind': 'image'|'video',
       'kind_index': k,          # index within that modality's pixel tensor
       'patches': t*h*w}         # patches the ViT consumes for this segment

    Raises ValueError if there is no vision segment (text-only prompt) or a
    run length does not match its grid (t*(h/m)*(w/m)).
    """
    if video_grid_thw is not None and len(video_grid_thw):
        vg = torch.as_tensor(video_grid_thw).clone()
        vg = torch.repeat_interleave(vg, vg[:, 0], dim=0)
        vg[:, 0] = 1
    else:
        vg = None

    ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(
        input_ids)
    n = len(ids)
    m = spatial_merge_size
    segs: list = []
    im = vi = 0
    i = 0
    while i < n:
        tok = ids[i]
        if tok not in (image_token_id, video_token_id):
            i += 1
            continue
        j = i
        while j < n and ids[j] == tok:
            j += 1
        if tok == image_token_id:
            t, h, w = (int(x) for x in image_grid_thw[im])
            kind, kind_index = 'image', im
            im += 1
        else:
            t, h, w = (int(x) for x in vg[vi])
            kind, kind_index = 'video', vi
            vi += 1
        if j - i != t * (h // m) * (w // m):
            raise ValueError(
                f'vision-token span [{i},{j}) does not match its grid '
                f'{(t, h, w)} (expected {t * (h // m) * (w // m)} tokens)')
        segs.append({'span': (i, j), 'grid': (t, h, w), 'kind': kind,
                     'kind_index': kind_index, 'patches': t * h * w})
        i = j

    if not segs:
        raise ValueError(
            'Qwen3-VL frontend requires at least one image or video '
            'segment; for text-only prompts use the Qwen3 text path '
            '(Qwen3TorchFrontendRtx).')
    return segs


def mrope_position_ids(input_ids, image_grid_thw=None, video_grid_thw=None, *,
                       image_token_id, video_token_id, vision_start_token_id,
                       spatial_merge_size):
    """3D MRoPE position ids for a single (unpadded) sequence.

    Mirrors HF ``Qwen3VLModel.get_rope_index``: walks the interleaved text /
    image / video spans in token order. A video grid (t, h, w) is split into
    ``t`` per-frame rows (t -> 1) so each frame is encoded like an image and
    the temporal position is carried by the timestamp text tokens between
    frames.

    Args:
      input_ids: (S,) long.
      image_grid_thw: (num_images, 3) long — (t, h, w) per image, or None.
      video_grid_thw: (num_videos, 3) long — (t, h, w) per video, or None.
    Returns:
      (3, S) long position ids (t, h, w rows).
    """
    if video_grid_thw is not None and len(video_grid_thw):
        vg = video_grid_thw.clone()
        vg = torch.repeat_interleave(vg, vg[:, 0], dim=0)
        vg[:, 0] = 1
    else:
        vg = None

    ids = input_ids.tolist()
    n = len(ids)
    m = spatial_merge_size
    starts = [i for i, tok in enumerate(ids) if tok == vision_start_token_id]
    image_nums = sum(1 for i in starts if ids[i + 1] == image_token_id)
    video_nums = sum(1 for i in starts if ids[i + 1] == video_token_id)

    parts: list = []
    st = 0
    im = vi = 0
    rem_i, rem_v = image_nums, video_nums
    for _ in range(image_nums + video_nums):
        ed_i = ids.index(image_token_id, st) if (
            image_token_id in ids and rem_i > 0) else n + 1
        ed_v = ids.index(video_token_id, st) if (
            video_token_id in ids and rem_v > 0) else n + 1
        if ed_i < ed_v:
            t, h, w = (int(x) for x in image_grid_thw[im])
            im += 1
            rem_i -= 1
            ed = ed_i
        else:
            t, h, w = (int(x) for x in vg[vi])
            vi += 1
            rem_v -= 1
            ed = ed_v
        gt, gh, gw = t, h // m, w // m
        text_len = ed - st
        base = int(parts[-1].max()) + 1 if parts else 0
        parts.append(
            torch.arange(text_len).view(1, -1).expand(3, -1) + base)
        t_idx = torch.arange(gt).view(-1, 1).expand(-1, gh * gw).flatten()
        h_idx = torch.arange(gh).view(1, -1, 1).expand(gt, -1, gw).flatten()
        w_idx = torch.arange(gw).view(1, 1, -1).expand(gt, gh, -1).flatten()
        parts.append(
            torch.stack([t_idx, h_idx, w_idx]) + text_len + base)
        st = ed + gt * gh * gw
    if st < n:
        base = int(parts[-1].max()) + 1 if parts else 0
        parts.append(torch.arange(n - st).view(1, -1).expand(3, -1) + base)
    return torch.cat(parts, dim=1).reshape(3, n)


def mrope_cos_sin(position_ids, *, head_dim, rope_theta, mrope_section,
                  device='cuda:0', dtype=torch.bfloat16):
    """Interleaved-MRoPE cos/sin tables for the language stack.

    Args:
      position_ids: (3, S) long.
    Returns:
      (cos, sin) each (S, head_dim/2) on ``device`` in ``dtype``.
    """
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = position_ids.to(torch.float32)
    freqs = pos[:, :, None] * inv_freq[None, None, :]            # (3, S, half)
    out = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        idx = slice(offset, mrope_section[axis] * 3, 3)
        out[:, idx] = freqs[axis][:, idx]
    cos = out.cos().to(device).to(dtype).contiguous()
    sin = out.sin().to(device).to(dtype).contiguous()
    return cos, sin


def build_mrope_cache(*, max_pos, head_dim, rope_theta,
                      device='cuda:0', dtype=torch.bfloat16):
    """Precompute per-axis MRoPE cos/sin lookup tables.

    Returns (cos_cache, sin_cache), each (max_pos, head_dim/2). The caller
    applies Qwen3-VL's interleaving policy for t/h/w axes.
    """
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, head_dim, 2, device=device,
                     dtype=torch.float32) / head_dim))
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return (freqs.cos().to(dtype).contiguous(),
            freqs.sin().to(dtype).contiguous())


def mrope_cos_sin_cached(position_ids, cos_cache, sin_cache, *,
                         mrope_section):
    """Gather interleaved-MRoPE cos/sin from precomputed lookup tables."""
    pos = position_ids.to(device=cos_cache.device, dtype=torch.long)
    cos = cos_cache[pos[0]].clone()
    sin = sin_cache[pos[0]].clone()
    for axis, offset in ((1, 1), (2, 2)):
        idx = slice(offset, mrope_section[axis] * 3, 3)
        cos[:, idx] = cos_cache[pos[axis]][:, idx]
        sin[:, idx] = sin_cache[pos[axis]][:, idx]
    return cos.contiguous(), sin.contiguous()


def vision_rope_cos_sin(grid_thw, *, head_dim, spatial_merge_size,
                        rope_theta=10000.0, device='cuda:0',
                        dtype=torch.bfloat16):
    """rotate_half 2D-RoPE cos/sin for the ViT patch grid.

    Returns (cos, sin) each (S, head_dim/2) — S = total patches.
    """
    dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, dim, 2, dtype=torch.float32) / dim))       # (dim/2,)
    max_hw = int(grid_thw[:, 1:].max())
    # freq_table: (max_hw, dim/2)
    freq_table = torch.outer(
        torch.arange(max_hw, dtype=torch.float32), inv_freq)

    coords = []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        mh, mw = h // spatial_merge_size, w // spatial_merge_size
        m = spatial_merge_size
        rows = (torch.arange(mh)[:, None, None, None] * m
                + torch.arange(m)[None, None, :, None])
        cols = (torch.arange(mw)[None, :, None, None] * m
                + torch.arange(m)[None, None, None, :])
        rows = rows.expand(mh, mw, m, m).reshape(-1)
        cols = cols.expand(mh, mw, m, m).reshape(-1)
        c = torch.stack((rows, cols), dim=-1)
        if t > 1:
            c = c.repeat(t, 1)
        coords.append(c)
    pos_ids = torch.cat(coords, dim=0)                             # (S, 2)
    freqs = freq_table[pos_ids].flatten(1)                         # (S, dim)
    cos = freqs.cos().to(device).to(dtype).contiguous()
    sin = freqs.sin().to(device).to(dtype).contiguous()
    return cos, sin


def build_vision_rope_cache(*, max_hw, head_dim, rope_theta=10000.0,
                            device='cuda:0', dtype=torch.bfloat16):
    """Precompute rotate-half 2D-RoPE cos/sin lookup tables per coordinate."""
    dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    coords = torch.arange(max_hw, dtype=torch.float32)
    freqs = torch.outer(coords, inv_freq)
    return (freqs.cos().to(device).to(dtype).contiguous(),
            freqs.sin().to(device).to(dtype).contiguous())


def vision_rope_cos_sin_cached(grid_thw, cos_cache, sin_cache, *,
                               spatial_merge_size):
    """Gather rotate-half 2D-RoPE cos/sin from precomputed coordinate tables."""
    coords = []
    device = cos_cache.device
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        mh, mw = h // spatial_merge_size, w // spatial_merge_size
        m = spatial_merge_size
        rows = (torch.arange(mh, device=device)[:, None, None, None] * m
                + torch.arange(m, device=device)[None, None, :, None])
        cols = (torch.arange(mw, device=device)[None, :, None, None] * m
                + torch.arange(m, device=device)[None, None, None, :])
        rows = rows.expand(mh, mw, m, m).reshape(-1)
        cols = cols.expand(mh, mw, m, m).reshape(-1)
        c = torch.stack((rows, cols), dim=-1)
        if t > 1:
            c = c.repeat(t, 1)
        coords.append(c)
    pos_ids = torch.cat(coords, dim=0)
    cos = torch.cat((cos_cache[pos_ids[:, 0]], cos_cache[pos_ids[:, 1]]),
                    dim=1)
    sin = torch.cat((sin_cache[pos_ids[:, 0]], sin_cache[pos_ids[:, 1]]),
                    dim=1)
    return cos.contiguous(), sin.contiguous()


def vision_pos_embeds(grid_thw, pos_embed_table, *, num_grid_per_side,
                      spatial_merge_size, device='cuda:0',
                      dtype=torch.bfloat16):
    """Bilinear-interpolated learned position embedding, in merge order.

    Args:
      pos_embed_table: (num_grid_per_side^2, hidden) cuda tensor.
    Returns:
      (S, hidden) bf16 cuda.
    """
    gs = num_grid_per_side
    table = pos_embed_table.float()
    out_parts = []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        hi = torch.linspace(0, gs - 1, h)
        wi = torch.linspace(0, gs - 1, w)
        hf, wf = hi.int(), wi.int()
        hc = (hf + 1).clip(max=gs - 1)
        wc = (wf + 1).clip(max=gs - 1)
        dh, dw = hi - hf, wi - wf
        base, base_c = hf * gs, hc * gs
        idx = [
            (base[:, None] + wf[None]).flatten(),
            (base[:, None] + wc[None]).flatten(),
            (base_c[:, None] + wf[None]).flatten(),
            (base_c[:, None] + wc[None]).flatten(),
        ]
        wt = [
            ((1 - dh)[:, None] * (1 - dw)[None]).flatten(),
            ((1 - dh)[:, None] * dw[None]).flatten(),
            (dh[:, None] * (1 - dw)[None]).flatten(),
            (dh[:, None] * dw[None]).flatten(),
        ]
        pe = sum(table[idx[i].to(table.device)]
                 * wt[i].to(table.device)[:, None] for i in range(4))
        m = spatial_merge_size
        pe = pe.repeat(t, 1)
        pe = (pe.view(t, h // m, m, w // m, m, -1)
              .permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
        out_parts.append(pe)
    return torch.cat(out_parts, dim=0).to(device).to(dtype).contiguous()
