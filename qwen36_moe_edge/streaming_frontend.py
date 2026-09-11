#!/usr/bin/env python3
"""Qwen3.6-35B-A3B with the routed experts read from storage.

The shipped frontend holds every expert in memory, which is 16.9 GiB of a
21.4 GiB footprint. This one skips them at load and serves each token's top-k
from a bounded cache over a prepared bundle, so what stays resident is the
non-expert weights plus however many slots the budget affords.

It is the same pipeline otherwise: same attention, same recurrence, same
router, same reducer. Only where the expert weights come from changes, which is
why a token-level comparison against the ordinary frontend is meaningful.

Greedy decode only, and not for CUDA Graph capture: a miss issues host reads,
which a captured graph cannot replay.
"""

from __future__ import annotations

from pathlib import Path

from flash_rt.frontends.torch.qwen36_moe import Qwen36MoeTextFrontend

from qwen36_moe_edge.expert_cache import CacheConfig, ExpertCache


class Qwen36MoeStreamingFrontend(Qwen36MoeTextFrontend):
    """Routed experts streamed from a bundle rather than held in memory."""

    _MODEL_LABEL = "Qwen3.6-35B-A3B text, streamed experts"

    # The block-scaled 4-bit MMA kernels are absent from this list because this
    # path never calls them: they serve the batched prefill, and streaming runs
    # prefill through the per-token loop instead, since a miss issues host reads.
    # Demanding them would refuse a build that can run this perfectly well --
    # which is what happened on the first attempt, on a target where the tier is
    # correctly not built at all.
    _REQUIRED_KERNELS = tuple(
        name for name in Qwen36MoeTextFrontend._REQUIRED_KERNELS
        if not name.startswith(('moe_blocktile_mma', 'moe_m16_mma'))
    ) + ('qwen35moe_e0m3_dequant_bf16', 'bf16_matvec_sm120_bf16')

    # The attention backend probes its kernel and falls back, so this runs on a
    # target that builds no FA2. Thor is one: it uses FA4, whose SM100-class
    # kernel needs Blackwell tensor memory that Orin's SM87 does not have --
    # so the two targets take different attention paths by design.
    _REQUIRE_FA2 = False

    def __init__(self, checkpoint_path: str, bundle: str | Path, *,
                 slots_per_layer: int,
                 device: str = "cuda:0",
                 max_seq: int = 2048,
                 staging_buffers: int = 4,
                 budget_bytes: int = 0,
                 reserve_bytes: int = 0,
                 warm_frequency=None) -> None:
        # Read by the loader through the base class, before any weight is
        # touched, so the expert tensors are never built.
        self._stream_experts = True
        super().__init__(
            checkpoint_path, device=device, max_seq=max_seq,
            quant_scope="experts")

        resident = 0
        try:
            import torch

            resident = int(torch.cuda.memory_allocated(device))
        except Exception:                                    # noqa: BLE001
            pass
        self.cache = ExpertCache(CacheConfig(
            bundle=Path(bundle),
            slots_per_layer=slots_per_layer,
            staging_buffers=staging_buffers,
            budget_bytes=budget_bytes,
            reserve_bytes=reserve_bytes,
            # Measured, not assumed: what the weights actually took.
            resident_bytes=resident,
            device=device,
        ))
        if warm_frequency is not None:
            self.cache.warm(warm_frequency)

    def generate(self, max_new_tokens: int, *, do_sample: bool = False):
        if self._prompt_ids is None:
            raise ValueError("call set_prompt(...) before generate()")
        if do_sample:
            raise NotImplementedError("greedy decoding only")

        from flash_rt.frontends.torch._nexn2_rtx_decode import (
            Nexn2DecodeState,
            generate_greedy,
        )

        if self._decode_state is None:
            self._decode_state = Nexn2DecodeState(
                self._weights, self._user_max_seq, self.device)
        state = self._decode_state
        state.expert_cache = self.cache
        # A miss reads from storage on the host, which a captured graph cannot
        # replay, so this path stays eager.
        state.batched_prefill = False
        return generate_greedy(
            state, self._prompt_ids, max_new_tokens, self._fvk, self.device)

    def close(self) -> None:
        self.cache.close()
