"""Fused decode form of a whole gated-delta layer.

The transformers fallback runs this layer as ~75 launches of Python
glue per token; serving its pieces individually is measurably negative
on a launch-bound host (a quantized projection swap *lost* throughput
here — the receipts are on the record). This impl owns the layer's
cached-decode step as one short chain of Hub kernels:

    packed in_proj GEMV -> causal_conv1d_update -> broadcast QKV split
    -> gating -> gated-delta recurrent core -> gated RMSNorm -> out_proj

A scheme may route the two projection GEMVs through the dynamic NVFP4
band (``gdn_projection_format="nvfp4_dynamic"``); the BF16 weights are
retained for prefill and detach either way.

Everything else — prefill, uncached calls, masked batches — dispatches
to the retained host layer and is counted.

Cache contract (the host's, followed not replaced): the layer reads and
writes ``cache_params.conv_states[idx]`` and ``recurrent_states[idx]``.
The host keeps the last K raw inputs in the conv state; the Hub update
kernel keeps the previous K-1, so the impl feeds ``state[..., 1:]`` and
rolls the host slot forward itself. The recurrent state slot is
normalised to a stable BF16 tensor on the first decode step and never
re-pointed after that: the core writes a scratch buffer (it cannot
write the slot it is reading within the same step) and the result is
copied back into the slot, which is what graph replay requires.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam

GDA_DEP = {"provider": "hf", "repo": "flashrt/gated-delta-attention",
           "version": ">=3"}
CONV_DEP = {"provider": "hf", "repo": "flashrt/causal-conv1d-state",
            "version": ">=1"}
FUSED_DEP = {"provider": "hf", "repo": "flashrt/transformer-fused-ops",
             "version": ">=1"}


@lru_cache(maxsize=1)
def _packages():
    from flash_rt.structures.impls import hub_kernel

    gda = hub_kernel(GDA_DEP["repo"], GDA_DEP["version"])
    conv = hub_kernel(CONV_DEP["repo"], CONV_DEP["version"])
    fused = hub_kernel(FUSED_DEP["repo"], FUSED_DEP["version"])
    for pkg, name in ((gda, "lin_split_qkv_broadcast_bf16"),
                      (gda, "gdn_gating_bf16"),
                      (gda, "gated_delta_recurrent_inout_bf16"),
                      (conv, "causal_conv1d_update_bf16"),
                      (fused, "rms_norm_gated_silu_bf16")):
        if not hasattr(pkg, name):
            raise ValueError(
                f"refused: installed build lacks {name}; a release "
                "carrying the fused decode chain is required")
    return gda, conv, fused


class FusedGatedDeltaDecodeLayer(GuardedSeam, torch.nn.Module):
    """Drop-in replacement for one gated-delta layer module."""

    _frt_host_attr = "host_layer"
    _frt_can_fallback = True

    def __init__(self, host, layer_idx: int,
                 projection_format: str | None = None):
        super().__init__()
        gda, conv, fused = _packages()
        self._gda, self._conv, self._fused = gda, conv, fused
        self.host_layer = host
        self._idx = int(layer_idx)
        self._hv = int(host.num_v_heads)
        self._hk = int(host.num_k_heads)
        self._d = int(host.head_v_dim)
        if (self._d != 128 or int(host.head_k_dim) != 128
                or self._hv <= 0 or self._hk <= 0
                or self._hv % self._hk):
            raise ValueError(
                "fused decode chain serves D=128 profiles whose v-head "
                "count is a multiple of the k-head count; other "
                "profiles keep the host layer")
        # the original 48/16 profile keeps its dedicated entries,
        # byte-for-byte; every other profile routes the head-generic
        # entries, whose absence from an older build is a clean bind
        # refusal (the ladder falls back to the callable-slot rule)
        if (self._hv, self._hk) == (48, 16):
            self._split_fn = gda.lin_split_qkv_broadcast_bf16
            self._gate_fn = gda.gdn_gating_bf16
            self._chunk_name = "gdn_chunk_from_conv_smem_bf16"
        else:
            for name in ("lin_split_qkv_broadcast_h_bf16",
                         "gdn_gating_h_bf16"):
                if not hasattr(gda, name):
                    raise ValueError(
                        f"refused: installed build lacks {name}; the "
                        f"{self._hv}/{self._hk}-head profile needs the "
                        "head-generic chain entries")
            hv, hk, d = self._hv, self._hk, self._d

            def _split(conv_out):
                return gda.lin_split_qkv_broadcast_h_bf16(
                    conv_out, hv, hk, d)

            def _gate(a, b, neg_exp_a, dt_bias):
                return gda.gdn_gating_h_bf16(a, b, neg_exp_a, dt_bias,
                                             num_heads=hv)

            self._split_fn = _split
            self._gate_fn = _gate
            self._chunk_name = "gdn_chunk_from_conv_smem_h_bf16"
        dev = host.in_proj_qkv.weight.device
        # the four input projections read the same activation; packing
        # them row-wise turns four GEMV launches into one, bit-identical
        # per output row. The host projections are rebound onto views of
        # the packed rows so the layer still carries one copy of these
        # weights — prefill and detach see the exact same values.
        self._packed_w = torch.cat(
            [host.in_proj_qkv.weight.detach(),
             host.in_proj_z.weight.detach(),
             host.in_proj_b.weight.detach(),
             host.in_proj_a.weight.detach()], dim=0)
        self._splits = []
        off = 0
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b",
                     "in_proj_a"):
            lin = getattr(host, name)
            n = int(lin.weight.shape[0])
            lin.weight = torch.nn.Parameter(
                self._packed_w[off:off + n],
                requires_grad=lin.weight.requires_grad)
            self._splits.append((off, off + n))
            off += n
        self._conv_w = host.conv1d.weight.detach().squeeze(1).contiguous()
        self._conv_b = (host.conv1d.bias.detach().contiguous()
                        if host.conv1d.bias is not None else None)
        self._neg_exp_a = (-host.A_log.detach().float().exp()).contiguous()
        self._dt_bias = host.dt_bias.detach().float().contiguous()
        self._eps = float(getattr(host.norm, "variance_epsilon",
                                  getattr(host.norm, "eps", 1e-6)))
        d_model = int(host.in_proj_qkv.weight.shape[1])
        # optional W4A4 decode band on the two projection GEMVs — a
        # scheme decision, never a default. The BF16 weights (and the
        # host views into them) are retained: prefill and detach stay
        # exact, only the decode band changes representation. A refusal
        # (missing package, unqualified shape) degrades to the BF16
        # band for this layer and is counted, not raised.
        self._proj_in = self._proj_out = None
        if projection_format == "nvfp4_dynamic":
            from ..linear_proj import nvfp4_dynamic
            try:
                self._proj_in, rel_in = nvfp4_dynamic.bind_proj_seam(
                    {"w": self._packed_w})
                self._proj_out, rel_out = nvfp4_dynamic.bind_proj_seam(
                    {"w": host.out_proj.weight.detach()})
                self._proj_rel = (rel_in, rel_out)
            except ValueError:
                self._proj_in = self._proj_out = None
        elif projection_format is not None:
            raise ValueError(
                f"refused: unknown gdn projection format "
                f"{projection_format!r}")
        self._state_a = torch.empty(1, self._hv, self._d, self._d,
                                    device=dev, dtype=torch.bfloat16)
        self._core_out = torch.empty(1, self._hv, self._d, device=dev,
                                     dtype=torch.bfloat16)
        # prefill chain needs the chunk entries; their absence is not a
        # bind refusal — prompts simply keep the host form
        self._chunk_ok = (
            hasattr(conv, "causal_conv1d_update_chunk_parallel_bf16")
            and hasattr(gda, self._chunk_name))
        # the WY pipeline is the prompt-length core: one fused chain
        # over all chunks, state carried inside the kernels — the
        # serial per-chunk walk of the fallback core is the measured
        # long-prompt TTFT term. Its entries are 48/16-shaped; other
        # profiles keep the chunk walk until head-generic WY ships.
        self._wy_h = (self._hv, self._hk) != (48, 16)
        _wy_names = ((
            "gdn_wy_norm_cumsum_pack_qk_h_bf16",
            "gdn_wy_kkt_b64_h_bf16",
            "gdn_wy_cast_ai_h_f32_to_bf16",
            "gdn_wy_recompute_wu_b64_mma_fla_h_bf16",
            "gdn_wy_chunk_h_b64_mma_fla_h_bf16",
            "gdn_wy_output_o_b64_mma_fla_h_bf16",
        ) if self._wy_h else (
            "lin_split_qkv_gqa_bf16",
            "gdn_wy_norm_cumsum_pack_qk_bf16",
            "gdn_wy_kkt_b64_bf16",
            "gdn_wy_cast_ai_f32_to_bf16",
            "gdn_wy_recompute_wu_b64_mma_fla_bf16",
            "gdn_wy_chunk_h_b64_mma_fla_bf16",
            "gdn_wy_output_o_b64_mma_fla_bf16",
        ))
        self._wy_ok = (self._chunk_ok
                       and all(hasattr(gda, n) for n in _wy_names))
        guard = self._frt_arm(dtypes=CAST_OK, device=dev, k=d_model)
        guard.notes["host_form_calls"] = 0
        guard.notes["proj_band"] = ("nvfp4" if self._proj_in is not None
                                    else "bf16")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_layer":
                raise
            return getattr(super().__getattr__("host_layer"), name)

    def _host_form(self, *args, **kwargs):
        if getattr(self, "_released", False):
            raise ValueError(
                "refused: host projection weights were released "
                "(one-way band); this call shape has no host fallback")
        guard = self._frt_guard
        if guard is not None and not torch.compiler.is_compiling():
            guard.notes["host_form_calls"] += 1
        return self.host_layer(*args, **kwargs)

    def _prefill_chain(self, hidden_states, cache_params):
        """Whole-prompt form: conv chunk + fused gating/split/recurrent.

        Chunks of 64 carry the conv state and the recurrent state
        forward in place, so any prompt length runs through the same
        two kernels per chunk. Larger slabs are on the record as a
        negative: at S>64 the chunk kernel's internal combine is not
        run-to-run stable (the repeat gate caught it) and the latency
        win measured under three percent — the fixed-order 64 chunk is
        the contract. Both host cache slots are written with the
        host's own semantics (last-K raw inputs; final state).
        """
        host = self.host_layer
        S = hidden_states.shape[1]
        x = hidden_states.view(S, -1)
        allp = (self._proj_in(x) if self._proj_in is not None
                else torch.nn.functional.linear(x, self._packed_w))
        (q0, q1), (z0, z1), (b0, b1), (a0, a1) = self._splits
        mixed = allp[:, q0:q1].contiguous()
        a_all = allp[:, a0:a1].contiguous()
        b_all = allp[:, b0:b1].contiguous()
        kk = self._conv_w.shape[-1]
        # continuation (a verify batch mid-stream) seeds from the live
        # slots; a fresh prompt starts from zero. The signal is an
        # explicit attribute only loop-owned caches carry — host caches
        # lack it and always get prompt semantics.
        # three continuation sources, one rule: a filled conv slot means
        # mid-stream unless the cache explicitly says fresh. Hosts that
        # chunk long prompts re-enter this branch per chunk with their
        # own cache carrying state (the 2K receipts caught the zero-
        # reset); loop-owned caches say False around a fresh prompt and
        # True around a verify batch.
        flag = getattr(cache_params, "frt_continue", None)
        old_slot = cache_params.conv_states[self._idx]
        cont = torch.is_tensor(old_slot) and flag is not False
        if cont:
            conv_state = old_slot[:, :, 1:].contiguous().clone()
            state = cache_params.recurrent_states[self._idx] \
                .view(self._hv, self._d, self._d) \
                .to(torch.bfloat16).contiguous().clone()
        else:
            conv_state = torch.zeros(1, mixed.shape[1], kk - 1,
                                     device=mixed.device,
                                     dtype=mixed.dtype)
            state = torch.zeros(self._hv, self._d, self._d,
                                device=mixed.device, dtype=torch.bfloat16)
        if self._wy_ok and S > 64:
            # the WY pipeline packs the whole span up front — gigabytes
            # of transients at deep prompts. Slabs bound the working
            # set: conv_state and state carry in place across slab
            # calls exactly as they do across the 64-chunks inside, so
            # the chunk sequence (and the arithmetic) is unchanged.
            slab = 8192
            if S > slab:
                core_out = torch.empty(S, self._hv, self._d,
                                       device=mixed.device,
                                       dtype=torch.bfloat16)
                for s0 in range(0, S, slab):
                    s1 = min(s0 + slab, S)
                    core_out[s0:s1] = self._wy_core(
                        mixed[s0:s1], a_all[s0:s1], b_all[s0:s1],
                        conv_state, state, s1 - s0)
            else:
                core_out = self._wy_core(mixed, a_all, b_all,
                                         conv_state, state, S)
            return self._prefill_epilogue(
                hidden_states, cache_params, allp, mixed, core_out,
                state, cont, old_slot, S)
        core_out = torch.empty(S, self._hv, self._d,
                               device=mixed.device, dtype=torch.bfloat16)
        for s0 in range(0, S, 64):
            s1 = min(s0 + 64, S)
            conv_out = self._conv.causal_conv1d_update_chunk_parallel_bf16(
                mixed[s0:s1].view(1, s1 - s0, -1), self._conv_w,
                conv_state, self._conv_b, apply_silu=True)
            if self._chunk_name.endswith("_h_bf16"):
                self._gda.gdn_chunk_from_conv_smem_h_bf16(
                    conv_out.view(s1 - s0, -1), a_all[s0:s1],
                    b_all[s0:s1], self._neg_exp_a, self._dt_bias, state,
                    num_v_heads=self._hv, num_k_heads=self._hk,
                    head_dim=self._d, use_qk_l2norm=True,
                    out=core_out[s0:s1])
            else:
                self._gda.gdn_chunk_from_conv_smem_bf16(
                    conv_out.view(s1 - s0, -1), a_all[s0:s1],
                    b_all[s0:s1], self._neg_exp_a, self._dt_bias, state,
                    use_qk_l2norm=True, out=core_out[s0:s1])
        return self._prefill_epilogue(
            hidden_states, cache_params, allp, mixed, core_out, state,
            cont, old_slot, S)

    def _wy_core(self, mixed, a_all, b_all, conv_state, state, S):
        """Whole-prompt gated-delta core: the WY pipeline, one pass.

        The conv update runs the full prompt in one launch; the WY
        chain (norm/cumsum -> KKT -> triangular solve -> WU recompute
        -> chunk-state carry -> output) keeps its chunks inside the
        kernels, carrying ``state`` in place — no serial per-chunk walk
        on the host, which is the measured long-prompt TTFT term the
        fallback core pays.
        """
        gda = self._gda
        conv_out = self._conv.causal_conv1d_update_chunk_parallel_bf16(
            mixed.view(1, S, -1), self._conv_w, conv_state,
            self._conv_b, apply_silu=True)
        co = conv_out.view(S, -1)
        g, beta = self._gate_fn(
            a_all.view(S, self._hv), b_all.view(S, self._hv),
            self._neg_exp_a, self._dt_bias)
        if self._wy_h:
            return self._wy_core_h(gda, co, g, beta, state, S)
        q16, k16, v48 = gda.lin_split_qkv_gqa_bf16(co)
        q16_l2, k16_l2, q_pack_hv, _k_pack_hk, g_cumsum = \
            gda.gdn_wy_norm_cumsum_pack_qk_bf16(q16, k16, g)
        big_a = gda.gdn_wy_kkt_b64_bf16(k16_l2, beta, g_cumsum)
        # the packaged triangular solve walks its rows serially and is
        # the measured 82% of this chain; the same inverse — semantics
        # pinned numerically: inv(I + strict_tril(A)) — through the
        # batched cuBLAS solve runs ~40x faster and stays deterministic
        eye = torch.eye(64, device=big_a.device,
                        dtype=big_a.dtype).expand_as(big_a).contiguous()
        ai = torch.linalg.solve_triangular(
            eye + torch.tril(big_a, -1), eye, upper=False).contiguous()
        ai_pack = gda.gdn_wy_cast_ai_f32_to_bf16(ai, S)
        w_pack, u_pack = gda.gdn_wy_recompute_wu_b64_mma_fla_bf16(
            k16_l2, v48, beta, g_cumsum, ai_pack)
        h0, _v_new, v_new_pack, k_pack_hv = \
            gda.gdn_wy_chunk_h_b64_mma_fla_bf16(
                k16_l2, w_pack, u_pack, g_cumsum, state)
        return gda.gdn_wy_output_o_b64_mma_fla_bf16(
            q_pack_hv, k_pack_hv, v_new_pack, h0, g_cumsum)

    def _wy_core_h(self, gda, co, g, beta, state, S):
        """The head-generic arm of the WY pipeline (non-48/16 hosts).

        The GQA split is contiguous column slices of the conv output —
        pinned bit-equal to the dedicated split kernel on the record —
        so the head-generic arm slices instead of asking for a kernel.
        """
        kd = self._hk * self._d
        hp = {"num_v_heads": self._hv, "num_k_heads": self._hk,
              "head_dim": self._d}
        q = co[:, :kd].contiguous().view(S, self._hk, self._d)
        k = co[:, kd:2 * kd].contiguous().view(S, self._hk, self._d)
        v = co[:, 2 * kd:].contiguous().view(S, self._hv, self._d)
        q_l2, k_l2, q_pack_hv, _k_pack_hk, g_cumsum = \
            gda.gdn_wy_norm_cumsum_pack_qk_h_bf16(q, k, g, **hp)
        big_a = gda.gdn_wy_kkt_b64_h_bf16(k_l2, beta, g_cumsum, **hp)
        eye = torch.eye(64, device=big_a.device,
                        dtype=big_a.dtype).expand_as(big_a).contiguous()
        ai = torch.linalg.solve_triangular(
            eye + torch.tril(big_a, -1), eye, upper=False).contiguous()
        ai_pack = gda.gdn_wy_cast_ai_h_f32_to_bf16(
            ai, S, num_v_heads=self._hv)
        w_pack, u_pack = gda.gdn_wy_recompute_wu_b64_mma_fla_h_bf16(
            k_l2, v, beta, g_cumsum, ai_pack, **hp)
        h0, _v_new, v_new_pack, k_pack_hv = \
            gda.gdn_wy_chunk_h_b64_mma_fla_h_bf16(
                k_l2, w_pack, u_pack, g_cumsum, state, **hp)
        return gda.gdn_wy_output_o_b64_mma_fla_h_bf16(
            q_pack_hv, k_pack_hv, v_new_pack, h0, g_cumsum, **hp)

    def _prefill_epilogue(self, hidden_states, cache_params, allp,
                          mixed, core_out, state, cont, old_slot, S):
        host = self.host_layer
        (_q0, _q1), (z0, z1), _b, _a = self._splits
        kk = self._conv_w.shape[-1]
        normed = self._fused.rms_norm_gated_silu_bf16(
            core_out.reshape(S * self._hv, self._d),
            allp[:, z0:z1].contiguous().view(S * self._hv, self._d),
            host.norm.weight, eps=self._eps)
        flat_norm = normed.view(S, -1)
        out = (self._proj_out(flat_norm) if self._proj_out is not None
               else torch.nn.functional.linear(flat_norm,
                                               host.out_proj.weight))
        # write INTO existing slots when they match — a repoint here
        # would strand a captured graph on the old tensors
        state4 = state.view(1, self._hv, self._d, self._d)
        rec = cache_params.recurrent_states[self._idx]
        if (torch.is_tensor(rec) and rec.shape == state4.shape
                and rec.dtype == state4.dtype):
            rec.copy_(state4)
        else:
            cache_params.recurrent_states[self._idx] = state4
        # the host slot keeps the last K *raw* projected inputs
        take = min(kk, S)
        cslot = cache_params.conv_states[self._idx]
        if not (torch.is_tensor(cslot)
                and cslot.shape == (1, mixed.shape[1], kk)
                and cslot.dtype == mixed.dtype):
            cslot = mixed.new_zeros(1, mixed.shape[1], kk)
            cache_params.conv_states[self._idx] = cslot
        if cont and S < kk:
            # short continuation: the slot keeps the last kk raw inputs
            # across the old tail and the new tokens
            head = old_slot[:, :, S:].clone()
            cslot[:, :, :kk - S].copy_(head)
        else:
            cslot.zero_()
        cslot[0, :, kk - take:] = mixed[S - take:].t()
        return out.view(1, S, -1).to(hidden_states.dtype)

    def forward(self, hidden_states, cache_params=None,
                attention_mask=None):
        admitted = self._frt_admit(hidden_states)
        if admitted is not PROCEED:
            return admitted
        decode = (cache_params is not None
                  and getattr(cache_params, "has_previous_state", False)
                  and hidden_states.shape[0] == 1
                  and hidden_states.shape[1] == 1
                  and (attention_mask is None
                       or bool(attention_mask.all())))
        if not decode:
            if (self._chunk_ok and cache_params is not None
                    and hidden_states.shape[0] == 1
                    and hidden_states.shape[1] > 1
                    and (attention_mask is None
                         or bool(attention_mask.all()))):
                return self._prefill_chain(hidden_states, cache_params)
            return self._host_form(hidden_states, cache_params,
                                   attention_mask)

        return self._decode_one(hidden_states, cache_params)

    def _decode_one(self, hidden_states, cache_params):
        host = self.host_layer
        x = hidden_states.view(1, -1)
        allp = (self._proj_in(x) if self._proj_in is not None
                else torch.nn.functional.linear(x, self._packed_w))
        # column slices of a single-row output stay contiguous
        (q0, q1), (z0, z1), (b0, b1), (a0, a1) = self._splits
        mixed = allp[:, q0:q1]
        z = allp[:, z0:z1]
        b = allp[:, b0:b1]
        a = allp[:, a0:a1]

        conv_host = cache_params.conv_states[self._idx]
        hub_state = conv_host[:, :, 1:].contiguous()
        conv_out = self._conv.causal_conv1d_update_bf16(
            mixed, self._conv_w, hub_state, self._conv_b,
            apply_silu=True)
        # the host slot keeps the last K raw inputs; roll it forward.
        # hub_state is a snapshot, so the two writes never overlap reads.
        conv_host[:, :, :-1].copy_(hub_state)
        conv_host[:, :, -1:].copy_(mixed.view(1, -1, 1))

        q, k, v = self._split_fn(conv_out)
        g, beta = self._gate_fn(
            a.view(1, self._hv), b.view(1, self._hv),
            self._neg_exp_a, self._dt_bias)
        state_in = cache_params.recurrent_states[self._idx]
        if state_in.dtype != torch.bfloat16 or not state_in.is_contiguous():
            # normalise the cache slot to a contiguous BF16 tensor once
            # (first decode after prefill); after this the slot pointer
            # never changes, which is what graph replay requires
            state_in = state_in.to(torch.bfloat16).contiguous()
            cache_params.recurrent_states[self._idx] = state_in
        core_out, new_state = self._gda.gated_delta_recurrent_inout_bf16(
            q.view(1, self._hv, self._d), k.view(1, self._hv, self._d),
            v.view(1, self._hv, self._d), g, beta,
            state_in, use_qk_l2norm=True,
            state_out=self._state_a, out=self._core_out)
        # scratch -> slot copy keeps the slot pointer stable; the core
        # cannot write the slot it is reading within the same step
        state_in.copy_(new_state)

        normed = self._fused.rms_norm_gated_silu_bf16(
            core_out.view(self._hv, self._d), z.view(self._hv, self._d),
            host.norm.weight, eps=self._eps)
        flat_norm = normed.view(1, -1)
        out = (self._proj_out(flat_norm) if self._proj_out is not None
               else torch.nn.functional.linear(flat_norm,
                                               host.out_proj.weight))
        return out.view(1, 1, -1).to(hidden_states.dtype)


@torch.no_grad()
def bind_fused_decode_layer(host, layer_idx: int,
                            projection_format: str | None = None,
                            release_host_weights: bool = False):
    """Bind one layer; a smoke step runs on zeros before handing out."""
    bound = FusedGatedDeltaDecodeLayer(host, layer_idx,
                                       projection_format)

    class _Cache:
        pass

    cache = _Cache()
    d_model = int(host.in_proj_qkv.weight.shape[1])
    conv_k = int(host.conv1d.weight.shape[-1])
    conv_dim = int(host.conv1d.weight.shape[0])
    dev = host.in_proj_qkv.weight.device
    cache.conv_states = {layer_idx: torch.zeros(
        1, conv_dim, conv_k, device=dev, dtype=torch.bfloat16)}
    cache.recurrent_states = {layer_idx: torch.zeros(
        1, bound._hv, bound._d, bound._d, device=dev,
        dtype=torch.bfloat16)}
    cache.has_previous_state = True
    probe = bound(torch.zeros(1, 1, d_model, device=dev,
                              dtype=torch.bfloat16), cache, None)
    if probe.shape != (1, 1, d_model) or \
            not torch.isfinite(probe.float()).all():
        raise ValueError("refused: fused decode chain smoke failed")
    guard = bound._frt_guard
    if guard is not None:
        guard.calls = 0
    if release_host_weights and bound._proj_in is not None \
            and bound._proj_out is not None:
        # one-way: the FP4 band passed its smoke, the BF16 projection
        # weights go. From here the host form refuses instead of
        # falling back, and detach restores structure, not bytes.
        empty = torch.nn.Parameter(
            host.in_proj_qkv.weight.new_empty(0), requires_grad=False)
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b",
                     "in_proj_a", "out_proj"):
            getattr(host, name).weight = empty
        bound._packed_w = None
        bound._released = True
        if guard is not None:
            guard.notes["host_weights"] = "released (one-way)"
    return bound
