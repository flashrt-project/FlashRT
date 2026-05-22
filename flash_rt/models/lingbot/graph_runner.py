"""LingBot-VLA — CUDA Graph capture + replay for ``sample_actions``.

Captures the full inference (prefix encode + 50× denoise) into a single
``torch.cuda.CUDAGraph``. Every replay reuses the recorded kernels with
no Python overhead and no allocator traffic for intermediate buffers
(they come from the graph's private mempool).

Prerequisites baked in (don't have to be re-established by the caller):
    * static FP8 calibration must be installed before capture so
      every ``linear_fp8`` takes the static path (the dynamic path's
      per-call ``torch.empty(1, fp32)`` for ``act_scale`` would also
      capture fine, but the static scales are persistent device tensors
      that PyTorch can address-stably across replays).
    * fused ``residual_add_rms_norm_fp8`` is gated on scales and
      activates automatically inside the captured region.
    * 's ViT capture cache (``_GRID_CACHE`` in :mod:`vit`) is
      populated by the warmup runs — they materialize the ``attn_mask``,
      ``cos``, ``sin``, and ``split_sizes`` for the input shape.
    * Every fvk binding called on the hot path now routes through
      ``torch.cuda.current_stream().cuda_stream`` (see ``kernel_ops.
      _current_stream``); without this, fvk launches fall onto the
      default stream which is NOT inside the captured graph and the
      replay produces NaN.

Usage::

    from flash_rt.models.lingbot import calibration as calib
    from flash_rt.models.lingbot.graph_runner import sample_actions_graph

    calib.set_static_scales(
        calib.load_calibration(JSON_PATH, device=torch.device("cuda")))
    replay = sample_actions_graph(
        target=target,
        images=images, img_masks=img_masks,
        lang_tokens=lang_tokens, lang_masks=lang_masks,
        state=state, noise=noise, num_steps=50,
    )
    actions = replay()       # repeatable, ~3× faster than eager
"""

from __future__ import annotations

from typing import Callable

import torch

from flash_rt.models.lingbot.sample_actions import sample_actions


def sample_actions_graph(
    *,
    target,
    images: torch.Tensor,
    img_masks: torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor,
    num_steps: int = 50,
    n_action_steps: int = 50,
    action_dim: int = 75,
    warmup_iters: int = 3,
) -> "tuple[Callable[[], torch.Tensor], torch.Tensor, torch.cuda.CUDAGraph]":
    """Capture a CUDAGraph of one full ``sample_actions`` and return
    a ``(replay_fn, captured_output, graph_handle)`` triple.

    Args:
        target: device-bound weight namespace (bf16, cuda).
        images, img_masks, lang_tokens, lang_masks, state, noise:
            fixed inputs. Their memory addresses are baked into the
            captured graph; mutate-in-place if you want to change the
            actual values across replays (the input tensor lives on,
            but the values it holds at replay time are what the graph
            reads).
        num_steps: number of Euler denoise steps.

    Returns:
        replay_fn: zero-arg callable that re-runs the graph and
            returns the same ``captured_output`` tensor (whose
            contents will be the latest inference result).
        captured_output: handle to the FINAL ``x_t`` tensor produced
            by the captured pipeline. After each ``replay_fn()`` call
            this tensor's contents are the new actions chunk.
        graph_handle: the underlying ``torch.cuda.CUDAGraph`` (in case
            the caller wants to inspect it).
    """
    # ── Warmup. Populates: FP8 weight cache + transpose cache,
    # cuBLAS heuristic state, ViT _GRID_CACHE (attn_mask/cos/sin),
    # static FP8 scale lookups. All allocations from warmup live in
    # the default mempool — they're STABLE addresses that the captured
    # graph will reference as external read-only state.
    for _ in range(warmup_iters):
        _ = sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            target=target, noise=noise, num_steps=num_steps,
            n_action_steps=n_action_steps, action_dim=action_dim,
        )
    torch.cuda.synchronize()

    # ── Capture.
    pool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(g, pool=pool):
            captured_output = sample_actions(
                images, img_masks, lang_tokens, lang_masks, state,
                target=target, noise=noise, num_steps=num_steps,
                n_action_steps=n_action_steps, action_dim=action_dim,
            )
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    def replay_fn() -> torch.Tensor:
        # Closure must hold a reference to ``pool`` so the CUDA mempool
        # that owns the captured tensors stays alive between replays.
        # Without this pin, ``graph_pool_handle()`` is GC'd at return
        # and PyTorch reclaims the mempool buffers — the next replay
        # then asserts on device-side. ``_pin`` is read at call time to
        # keep the Python ref alive (closure capture is by name, not
        # value, so dropping ``del pool`` would still defeat this).
        _pin = pool  # noqa: F841
        g.replay()
        return captured_output

    # Also attach to the graph object so callers who keep ``g`` but
    # accidentally drop ``replay_fn`` still keep the pool alive.
    g._lingbot_pool = pool  # noqa: SLF001
    return replay_fn, captured_output, g


def sample_actions_split_graph(
    *,
    target,
    images: torch.Tensor,
    img_masks: torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor,
    num_steps: int = 10,
    n_action_steps: int = 50,
    action_dim: int = 75,
    vlm_causal: bool = False,
    warmup_iters: int = 3,
):
    """: capture prefix and decoder as TWO separate CUDAGraphs sharing
    one mempool. Returns ``(prefix_replay, decoder_replay,
    captured_output, kv_cache, _pin)``.

    Both graphs share a single ``graph_pool_handle`` so the KV-cache
    tensors allocated inside the prefix-graph capture region remain at
    stable addresses and are read by the decoder graph. Lifetime is
    pinned by the closures + ``_pin`` tuple — drop ``_pin`` only when
    you no longer need either graph.

    Usage on a stable prompt::

        prefix_replay, decoder_replay, out, kv, _ = sample_actions_split_graph(...)
        prefix_replay()              # once per prompt
        decoder_replay()             # once per inference; out is reused
        # mutate state/noise IN PLACE between replays, NOT reassign

    Compared to :func:`sample_actions_graph` (single big capture), this
    avoids re-running the 32-block ViT + 36-layer VLM prefix encode on
    every inference. At deploy-realistic 10-step, the prefix portion is
    ~22% of total time (~50 ms in the graph); skipping it on calls 2..N
    drops the per-inference cost by ~50 ms.
    """
    from flash_rt.models.lingbot.sample_actions import (
        embed_prefix, predict_velocity, make_att_2d_masks, precompute_film,
    )
    from flash_rt.models.lingbot.forward import prefix_encode_36L
    from flash_rt.models.lingbot.mixed_attention import DEFAULT_ATTN_DIMS

    # Warmup populates every cache (FP8 weights, transpose, ViT capture
    # cache, cuBLAS heuristics, static FP8 scales).
    for _ in range(warmup_iters):
        _ = sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            target=target, noise=noise, num_steps=num_steps,
            n_action_steps=n_action_steps, action_dim=action_dim,
            vlm_causal=vlm_causal,
        )
    torch.cuda.synchronize()

    # Single shared mempool — KV-cache tensors allocated in the prefix
    # capture region get the same addresses the decoder graph records.
    pool = torch.cuda.graph_pool_handle()

    # --- 1. Prefix-encode graph ---
    prefix_graph = torch.cuda.CUDAGraph()
    cap1 = torch.cuda.Stream()
    cap1.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap1):
        with torch.cuda.graph(prefix_graph, pool=pool):
            prefix_embs, prefix_pad_masks, prefix_att_masks = embed_prefix(
                images, img_masks, lang_tokens, lang_masks,
                target=target, vlm_causal=vlm_causal,
            )
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, kv_cache = prefix_encode_36L(
                prefix_embs,
                position_ids=prefix_position_ids,
                attention_mask=prefix_att_2d,
                target=target, dims=DEFAULT_ATTN_DIMS,
                pad_mask=prefix_pad_masks,    # unmasked FA4 prefix attn
            )
    torch.cuda.current_stream().wait_stream(cap1)
    torch.cuda.synchronize()

    # --- 2. Decoder graph: Euler loop + velocity head ---
    decoder_graph = torch.cuda.CUDAGraph()
    cap2 = torch.cuda.Stream()
    cap2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap2):
        with torch.cuda.graph(decoder_graph, pool=pool):
            dt_f = -1.0 / num_steps
            x_t = noise
            time_f = 1.0
            # batched FiLM precompute (graph mode is calibrated).
            film = precompute_film(
                num_steps, target, DEFAULT_ATTN_DIMS,
                noise.device, noise.dtype)
            for i in range(num_steps):
                expanded_time = torch.full(
                    (noise.shape[0],), time_f,
                    dtype=noise.dtype, device=noise.device,
                )
                v_t = predict_velocity(
                    state, prefix_pad_masks, kv_cache,
                    x_t, expanded_time,
                    target=target, n_action_steps=n_action_steps,
                    dims=DEFAULT_ATTN_DIMS,
                    film=film, step_idx=i,
                    use_fused_attn=True,    # FA4 denoise (matches sample_actions;
                )                           # pad K/V zeroed in prefix → cuBLAS-free capture
                x_t = x_t + dt_f * v_t
                time_f += dt_f
            captured_output = x_t
    torch.cuda.current_stream().wait_stream(cap2)
    torch.cuda.synchronize()

    # Pin the pool + the kv_cache + prefix_pad_masks tensors so the
    # mempool isn't reclaimed at function return. ``_pin`` keeps them
    # alive for the lifetime of the returned closures.
    _pin = (pool, kv_cache, prefix_pad_masks, prefix_embs)
    prefix_graph._lingbot_pin = _pin       # noqa: SLF001
    decoder_graph._lingbot_pin = _pin      # noqa: SLF001

    def prefix_replay():
        _pin_ref = _pin                    # keep closure alive
        prefix_graph.replay()

    def decoder_replay() -> torch.Tensor:
        _pin_ref = _pin
        decoder_graph.replay()
        return captured_output

    return prefix_replay, decoder_replay, captured_output, kv_cache, _pin


__all__ = ["sample_actions_graph", "sample_actions_split_graph"]
