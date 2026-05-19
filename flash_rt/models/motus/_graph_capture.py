"""G5 — CUDA Graph capture of the Motus denoise pipeline.

After all G3a-G4 swaps are installed and FP8 calibration is done, the
hot path is fully fvk-driven (norm + GEMM + attention + FFN + bias),
but each forward issues ~6000 pybind crossings per call. Even though
each kernel itself is fast, the per-launch CPU work ends up dominating.

CUDA Graph capture collapses the entire denoise loop (10 steps × 30
layers × N kernels each) into a single graph that replays with one
launch. Per-call cost drops from ~1037 ms (G4) to whatever the GPU
needs to actually compute the work.

Caveats / approach:
    1. PyTorch's torch.cuda.graph(...) requires CUDA ops to run on
       the capture stream. The fvk wrappers were updated in this gate
       to pass torch.cuda.current_stream().cuda_stream via
       flash_rt.models.motus._stream.cs(), which returns the right
       stream both inside and outside capture.
    2. Per-call inputs (first_frame, state) are copied into FIXED
       device buffers BEFORE replay. The graph reads from these
       same addresses on every replay.
    3. T5 ctx + VLM und_tokens are computed ONCE per set_prompt and
       stored on the pipeline as cached device tensors; the captured
       graph's wan-cross/und-attention paths read from these fixed
       addresses (no re-VLM per step inside graph).
    4. Random noise: torch.randn(...) inside the captured forward
       runs ONCE at capture time and produces the same bytes on
       every replay. This means the seed used at capture is "frozen"
       for all subsequent calls. Fine for our deterministic-test use
       case; will revisit for true production noise later.
    5. The output tensors are aliases into the captured mempool —
       must be cloned before being returned to the user, otherwise
       the next replay overwrites them.

Lifecycle:
    set_prompt(prompt, t5, vlm) -> pre-encode T5 + VLM tokens, store
        on pipeline (calls before infer; safe to repeat).
    infer #1 -> calibration (G4) on uncaptured eager path.
    infer #2 -> capture pass: warmup + torch.cuda.graph(...).
    infer #3+ -> replay only.

Or with explicit warmup helper:
    frontend.warmup_graph(sample_first_frame, sample_state)  # does #1+#2
    frontend.infer(...)  # straight to replay
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class MotusGraphState:
    """Owns graph + fixed I/O buffers for one captured pipeline."""

    def __init__(self):
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.mempool = None
        # Fixed input buffers (graph reads these addresses every replay).
        self.in_first_frame: Optional[torch.Tensor] = None
        self.in_state: Optional[torch.Tensor] = None
        # Output aliases — point into the captured mempool. Must be
        # cloned() in the caller before next replay.
        self.out_frames: Optional[torch.Tensor] = None
        self.out_actions: Optional[torch.Tensor] = None


def capture_motus_graph(
    pipeline,
    first_frame: torch.Tensor,
    state: Optional[torch.Tensor],
    t5_embeds,
    vlm_inputs,
) -> MotusGraphState:
    """Run a warmup forward, then capture pipeline.run into a CUDA Graph.

    Pre-conditions:
        * frontend has installed G3a-G4 swaps and finished calibration
        * t5_embeds + vlm_inputs already pre-encoded
        * first_frame, state are sample tensors of the same dtype/shape
          as production inputs (their bytes are not used after copy).

    Post-conditions:
        * Returned MotusGraphState carries a captured graph and fixed
          in/out buffers.
    """
    device = pipeline.device
    dtype = pipeline.dtype

    # ─── Allocate fixed input buffers ───
    state_obj = MotusGraphState()
    state_obj.in_first_frame = torch.empty(
        first_frame.shape, dtype=dtype, device=device).contiguous()
    state_obj.in_first_frame.copy_(first_frame.to(device).to(dtype))

    if state is not None:
        state_obj.in_state = torch.empty(
            state.shape, dtype=dtype, device=device).contiguous()
        state_obj.in_state.copy_(state.to(device).to(dtype))

    # ─── Warmup pass (eager) — populates cuBLAS / FA2 internal caches ───
    # This avoids surprise allocations during capture.
    # Also: this populates token shapes we need for time-embedding precompute.
    logger.info("[g5] graph warmup forward (eager)...")
    with torch.no_grad():
        _ = pipeline.run(
            first_frame=state_obj.in_first_frame,
            state=state_obj.in_state,
            t5_embeds=t5_embeds,
            vlm_inputs=vlm_inputs,
        )
    torch.cuda.synchronize()

    # ─── Precompute time embeddings for all N steps OUTSIDE graph ───
    # The upstream Wan get_time_embedding chain calls torch.arange — a
    # CPU op forbidden in capture. Derive token-counts from a probe.
    cond_latent = pipeline.encode_first_frame(state_obj.in_first_frame)
    video_latent_probe = pipeline.init_video_latent(cond_latent)
    video_tokens_probe = pipeline.prepare_video_tokens(video_latent_probe)
    action_tokens_probe = pipeline.prepare_action_tokens(
        pipeline.init_action_latent(state_obj.in_first_frame.shape[0]),
        state_obj.in_state,
    )

    # Precompute Wan 3D RoPE freq_grid for this video latent geometry.
    # Wan's grid_sizes is [B, 3] = [[T_l, H', W']]; for Motus B=1 it's
    # constant across calls.
    from flash_rt.models.motus._rope_swap import precompute_freq_grid
    grid_sizes = pipeline.model.video_module.grid_sizes
    precompute_freq_grid(pipeline.model, grid_sizes)
    seq_v = int(video_tokens_probe.shape[1])
    seq_a = int(action_tokens_probe.shape[1])
    logger.info(
        f"[g5] precomputing time embeddings: N={pipeline.dims.num_inference_steps}, "
        f"S_v={seq_v}, S_a={seq_a}")
    pipeline.precompute_time_embeddings(
        seq_len_video=seq_v,
        seq_len_action=seq_a,
        num_inference_steps=pipeline.dims.num_inference_steps,
        batch=state_obj.in_first_frame.shape[0],
    )

    # Re-run one warmup forward — this time exercising the cached
    # path so cuBLAS / FA2 see the same kernel sequences as the
    # captured graph will.
    with torch.no_grad():
        _ = pipeline.run(
            first_frame=state_obj.in_first_frame,
            state=state_obj.in_state,
            t5_embeds=t5_embeds,
            vlm_inputs=vlm_inputs,
        )
    torch.cuda.synchronize()

    # ─── Capture ───
    logger.info("[g5] capturing CUDA graph...")
    state_obj.mempool = torch.cuda.graphs.graph_pool_handle()
    state_obj.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(state_obj.graph, pool=state_obj.mempool):
        # All work emitted here ends up inside the graph; the captured
        # forward reads inputs from in_first_frame / in_state buffers
        # and writes outputs into mempool-allocated tensors that we
        # alias as out_frames / out_actions.
        with torch.no_grad():
            frames, actions = pipeline.run(
                first_frame=state_obj.in_first_frame,
                state=state_obj.in_state,
                t5_embeds=t5_embeds,
                vlm_inputs=vlm_inputs,
            )
        state_obj.out_frames = frames
        state_obj.out_actions = actions
    torch.cuda.synchronize()
    logger.info(
        f"[g5] graph captured: out_frames={tuple(state_obj.out_frames.shape)}, "
        f"out_actions={tuple(state_obj.out_actions.shape)}")

    return state_obj


def replay_motus_graph(
    state_obj: MotusGraphState,
    first_frame: torch.Tensor,
    state: Optional[torch.Tensor],
):
    """Copy live inputs into the captured buffers, replay, return clones.

    Returns (frames, actions) — both clone()d so the next replay can
    overwrite the graph's mempool tensors safely.
    """
    state_obj.in_first_frame.copy_(
        first_frame.to(state_obj.in_first_frame.device).to(
            state_obj.in_first_frame.dtype))
    if state is not None and state_obj.in_state is not None:
        state_obj.in_state.copy_(
            state.to(state_obj.in_state.device).to(state_obj.in_state.dtype))

    state_obj.graph.replay()
    return state_obj.out_frames.clone(), state_obj.out_actions.clone()
