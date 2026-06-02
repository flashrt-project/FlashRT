"""GR00T N1.7 Thor forward pipeline (FP8) — pointer-only forward functions.

Lit up incrementally.

Stages (call order in ``infer``):

1. ``qwen3vl_vit_forward``           — 24-layer Qwen3-VL ViT (FP16)
2. ``deepstack_merge_forward``       — 3 mergers from ViT layers [5, 11, 17]
3. ``qwen3vl_llm_forward``           — 16-layer truncated LLM with M-RoPE
4. ``vlln_forward``                  — LayerNorm on backbone_features [B, S, 2048]
5. ``vl_self_attn_forward``          — 4-layer MHA (32 heads × 64 head_dim)
6. ``embodiment_state_encode``       — state [132] → [1536]
7. For step in range(num_inference_timesteps=4):
   a. ``embodiment_action_encode``   — action [40, 132] + timestep → [40, 1536]
   b. ``dit_forward``                — 32 AlternateVLDiT blocks
   c. ``embodiment_action_decode``   — [40, 1536] → [40, 132] velocity
   d. ``gpu_euler_step``             — Euler update

Each forward is **pointer-only**: every device tensor is provided as an int
data_ptr by the frontend. No allocation, no host↔device traffic. The frontend
pre-allocates every scratch buffer in ``_allocate_buffers`` so the pipeline
captures into a CUDA Graph cleanly.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────
# Stage 4: VLLN — LayerNorm on backbone_features
# ─────────────────────────────────────────────────────────────────────────


def vlln_forward(gemm, fvk, bufs, weights, dims,
                 scales_dev=None, *, attn=None, stream: int = 0) -> None:
    """LayerNorm(2048) on backbone features ``(B, S, 2048)``.

    The N1.7 action head's ``vlln`` is ``nn.LayerNorm(backbone_embedding_dim=2048)``
    with bias and PyTorch default ``eps=1e-5`` (gr00t_n1d7.py:84).

    The kernel reads a flat ``S × D`` row-major buffer regardless of leading
    batch dim, so passing ``S = B * seq_len`` is correct.

    Required:
        bufs["x"]            — backbone_features fp16 (S × D)
        bufs["out"]          — vlln_out fp16 (S × D)
        weights["vlln_w"]    — gamma fp16 (D,)
        weights["vlln_b"]    — beta  fp16 (D,)
        dims["S"], dims["D"] — flat sequence × hidden
    """
    fvk.layer_norm_fp16(
        int(bufs["x"]),
        int(weights["vlln_w"]),
        int(weights["vlln_b"]),
        int(bufs["out"]),
        int(dims["S"]),
        int(dims["D"]),
        1e-5,
        int(stream),
    )


# ─────────────────────────────────────────────────────────────────────────
# Stages still pending (Phase 3b.2)
# ─────────────────────────────────────────────────────────────────────────


def qwen3vl_vit_forward(gemm, fvk, bufs, weights, dims,
                        scales_dev, *, attn, stream: int = 0,
                        layers_subset=None,
                        deepstack_taps=(5, 11, 17),
                        deepstack_capture=None, use_fp8=True) -> None:
    """24-layer Qwen3-VL ViT (FP16-input + FP8 GEMMs, multi-view batched FMHA).

    Per layer (input ``h``, in-place residual update):

        xn   = LayerNorm(h, norm1_w, norm1_b, eps=1e-6)
        xn_q = quantize_fp8(xn, act_qkv_scale)
        Q    = fp8_nn_bias(xn_q, q_w, q_b, alpha_q)        — (S, D)
        K    = fp8_nn_bias(xn_q, k_w, k_b, alpha_k)
        V    = fp8_nn_bias(xn_q, v_w, v_b, alpha_v)
        rope_rotate_half(Q, cos, sin, S, NH, HD)            — split-half RoPE
        rope_rotate_half(K, cos, sin, S, NH, HD)
        O    = attn.run("vit", li, q_seq=Sper_view, kv_seq=Sper_view)
                                                            — multi-view FMHA;
                                                            stride=D (separated Q/K/V).
        O_q  = quantize_fp8(O, act_o_scale)
        o    = fp8_nn_bias(O_q, o_w, o_b, alpha_o)
        h   += o                                            (residual 1)

        xn   = LayerNorm(h, norm2_w, norm2_b, eps=1e-6)
        xn_q = quantize_fp8(xn, act_fc1_scale)
        h1   = fp8_nn_gelu_bias(xn_q, fc1_w, fc1_b, alpha_fc1)  — GELU(tanh-approx) fused
        h1_q = quantize_fp8(h1, act_fc2_scale)
        f2   = fp8_nn_bias(h1_q, fc2_w, fc2_b, alpha_fc2)
        h   += f2                                           (residual 2)

        if li in deepstack_taps and deepstack_capture is not None:
            deepstack_capture[deepstack_taps.index(li)](h)

    The fused QKV is split into 3 separate FP8 GEMMs (vs HF's single fused
    Linear) so RoPE can apply on contiguous Q/K with the existing split-half
    rope kernel — the FMHA kernel is invoked in separated mode (stride=D).
    Q/K/V/O slots come from ``attn.get_slot_ptrs("vit", li)``; per-layer
    indexing is irrelevant (vit slots are layer-shared).

    Q/K/V buffer layout is ``(S, NH*HD)`` row-major; attention is multi-view
    batched (each of the ``num_views`` image groups attends only within
    itself, like HF's ``cu_seqlens`` chunking).

    Args:
        bufs:
            h           — running hidden state, fp16 (S, D), in-place
            xn          — LayerNorm scratch, fp16 (S, D)
            xn_fp8      — FP8 quant scratch, fp8 (S, D)
            o_proj_out  — fp16 (S, D)
            fc1_out     — fp16 (S, ff_inner=4096)
            fc1_fp8     — fp8  (S, ff_inner)
        weights (length-num_layers lists, plus singletons):
            norm1_w/b, norm2_w/b
            q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b
            fc1_w, fc1_b, fc2_w, fc2_b
            alpha_q/k/v/o, alpha_fc1, alpha_fc2 (host floats)
            cos, sin (singleton fp16 device ptrs, shape (S, HD))
        scales_dev:
            act_qkv, act_o, act_fc1, act_fc2  (fp32 device scalar ptrs)
        dims:
            S (= total tokens, e.g. 1024), D (= 1024), NH (= 16), HD (= 64),
            ff_inner (= 4096), Sper_view (= S // num_views)
        deepstack_taps: layer indices to expose to the deepstack callback
            (default ``(5, 11, 17)`` per Qwen3-VL config).
        deepstack_capture: optional list[Callable[device-ptr-int, None]].
            Length must equal len(deepstack_taps). Each callback receives the
            ``h`` device ptr immediately after the corresponding tap layer's
            residual update; intended for tests that snapshot intermediate
            hidden states.
    """
    S  = int(dims["S"])
    D  = int(dims["D"])
    NH = int(dims["NH"])
    HD = int(dims["HD"])
    FF = int(dims["ff_inner"])

    h_ptr        = int(bufs["h"])
    xn_ptr       = int(bufs["xn"])
    xn_fp8_ptr   = int(bufs["xn_fp8"])
    o_proj_out   = int(bufs["o_proj_out"])
    fc1_out_ptr  = int(bufs["fc1_out"])
    fc1_fp8_ptr  = int(bufs["fc1_fp8"])
    cos_ptr      = int(weights["cos"])
    sin_ptr      = int(weights["sin"])

    layer_iter = range(24) if layers_subset is None else list(layers_subset)
    Sper = int(dims.get("Sper_view", S))

    for li in layer_iter:
        slots = attn.get_slot_ptrs("vit", li)
        Q_ptr, K_ptr, V_ptr, O_ptr = slots["Q"], slots["K"], slots["V"], slots["O"]

        # ── Pre-attn LayerNorm ──────────────────────────────────────────
        fvk.layer_norm_fp16(
            h_ptr, int(weights["norm1_w"][li]), int(weights["norm1_b"][li]),
            xn_ptr, S, D, 1e-6, int(stream),
        )

        # ── Q/K/V projections (single shared act-scale on xn for FP8) ───
        if use_fp8:
            fvk.quantize_fp8_static_fp16(
                xn_ptr, xn_fp8_ptr, int(scales_dev["act_qkv"][li]),
                S * D, int(stream),
            )
            gemm.fp8_nn_bias(
                xn_fp8_ptr, int(weights["q_w"][li]), Q_ptr, int(weights["q_b"][li]),
                S, D, D, float(weights["alpha_q"][li]), int(stream),
            )
            gemm.fp8_nn_bias(
                xn_fp8_ptr, int(weights["k_w"][li]), K_ptr, int(weights["k_b"][li]),
                S, D, D, float(weights["alpha_k"][li]), int(stream),
            )
            gemm.fp8_nn_bias(
                xn_fp8_ptr, int(weights["v_w"][li]), V_ptr, int(weights["v_b"][li]),
                S, D, D, float(weights["alpha_v"][li]), int(stream),
            )
        else:
            for wk, bk, out in (("q_w_fp16", "q_b", Q_ptr),
                                ("k_w_fp16", "k_b", K_ptr),
                                ("v_w_fp16", "v_b", V_ptr)):
                gemm.fp16_nn(xn_ptr, int(weights[wk][li]), out, S, D, D, int(stream))
                fvk.add_bias_fp16(out, int(weights[bk][li]), S, D, int(stream))

        # ── Split-half RoPE on Q and K (contiguous (S, NH, HD)) ─────────
        fvk.rope_rotate_half_fp16(Q_ptr, cos_ptr, sin_ptr, S, NH, HD, int(stream))
        fvk.rope_rotate_half_fp16(K_ptr, cos_ptr, sin_ptr, S, NH, HD, int(stream))

        # ── Multi-view batched FMHA (separated Q/K/V, stride=D) ─────────
        attn.run("vit", li, q_seq=Sper, kv_seq=Sper, stream=int(stream))

        # ── o_proj ──────────────────────────────────────────────────────
        if use_fp8:
            fvk.quantize_fp8_static_fp16(
                O_ptr, xn_fp8_ptr, int(scales_dev["act_o"][li]),
                S * D, int(stream),
            )
            gemm.fp8_nn_bias(
                xn_fp8_ptr, int(weights["o_w"][li]), o_proj_out,
                int(weights["o_b"][li]),
                S, D, D, float(weights["alpha_o"][li]), int(stream),
            )
        else:
            gemm.fp16_nn(O_ptr, int(weights["o_w_fp16"][li]), o_proj_out, S, D, D, int(stream))
            fvk.add_bias_fp16(o_proj_out, int(weights["o_b"][li]), S, D, int(stream))

        # ── Residual 1 ─────────────────────────────────────────────────
        fvk.residual_add_fp16(h_ptr, o_proj_out, S * D, int(stream))

        # ── Pre-FF LayerNorm ───────────────────────────────────────────
        fvk.layer_norm_fp16(
            h_ptr, int(weights["norm2_w"][li]), int(weights["norm2_b"][li]),
            xn_ptr, S, D, 1e-6, int(stream),
        )

        # ── FF: D → FF (GELU) → D ──────────────────────────────────────
        if use_fp8:
            fvk.quantize_fp8_static_fp16(
                xn_ptr, xn_fp8_ptr, int(scales_dev["act_fc1"][li]),
                S * D, int(stream),
            )
            gemm.fp8_nn_gelu_bias(
                xn_fp8_ptr, int(weights["fc1_w"][li]), fc1_out_ptr,
                int(weights["fc1_b"][li]),
                S, FF, D, float(weights["alpha_fc1"][li]), int(stream),
            )
            fvk.quantize_fp8_static_fp16(
                fc1_out_ptr, fc1_fp8_ptr, int(scales_dev["act_fc2"][li]),
                S * FF, int(stream),
            )
            gemm.fp8_nn_bias(
                fc1_fp8_ptr, int(weights["fc2_w"][li]), o_proj_out,
                int(weights["fc2_b"][li]),
                S, D, FF, float(weights["alpha_fc2"][li]), int(stream),
            )
        else:
            gemm.fp16_nn(xn_ptr, int(weights["fc1_w_fp16"][li]), fc1_out_ptr, S, FF, D, int(stream))
            fvk.add_bias_fp16(fc1_out_ptr, int(weights["fc1_b"][li]), S, FF, int(stream))
            fvk.gelu_inplace_fp16(fc1_out_ptr, S * FF, int(stream))
            gemm.fp16_nn(fc1_out_ptr, int(weights["fc2_w_fp16"][li]), o_proj_out, S, D, FF, int(stream))
            fvk.add_bias_fp16(o_proj_out, int(weights["fc2_b"][li]), S, D, int(stream))

        # ── Residual 2 ─────────────────────────────────────────────────
        fvk.residual_add_fp16(h_ptr, o_proj_out, S * D, int(stream))

        # ── DeepStack tap callback ─────────────────────────────────────
        if deepstack_capture is not None and li in deepstack_taps:
            deepstack_capture[deepstack_taps.index(li)](h_ptr)


def deepstack_merge_forward(gemm, fvk, bufs, weights, dims,
                            scales_dev, *, attn=None, stream: int = 0,
                            use_fp8=True) -> None:
    """3 deepstack mergers — taps ViT layers ``[5, 11, 17]`` and produces
    3 features for DeepStack injection into LLM layers ``[0, 1, 2]``.

    Per HF ``Qwen3VLVisionPatchMerger`` with ``use_postshuffle_norm=True``
    (modeling_qwen3_vl.py:586-598), each merger ``j ∈ {0,1,2}``:

        x = vit_block_{[5,11,17][j]}.view(N//4, 4*1024)   # spatial 4:1 merge
        x = LayerNorm(x, w=norm_w, b=norm_b, dim=4096, eps=1e-6)
        x = quantize_fp8(x, act_fc1_scale)
        x = fp8_nn_gelu_bias(x, fc1_w, fc1_b, alpha_fc1)  # GELU fused (tanh-approx; verify)
        x = quantize_fp8(x, act_fc2_scale)
        x = fp8_nn_bias(x, fc2_w, fc2_b, alpha_fc2)        # → (256, 2048)

    The spatial-merge reshape is a no-op pointer-wise: row-major
    ``(N=1024, 1024)`` and ``(N//4=256, 4096)`` share the same byte layout.

    Args:
        bufs (per-merger lists where noted):
            in            — list[3] device fp16 ptrs, each (Nin, Din) input
            ln_out        — shared device fp16 ptr,   (Nout, Dmid) scratch
            fp8_scratch   — shared device fp8  ptr,   (Nout, Dmid) scratch
            fc1_out       — shared device fp16 ptr,   (Nout, Dmid) scratch
            out           — list[3] device fp16 ptrs, each (Nout, Dout) output
        weights (per-merger lists):
            norm_w[j], norm_b[j]            — fp16 (Dmid,)
            fc1_w[j]                        — fp8  (Dmid, Dmid) (B in [K,N])
            fc1_b[j]                        — fp16 (Dmid,)
            fc2_w[j]                        — fp8  (Dmid, Dout)
            fc2_b[j]                        — fp16 (Dout,)
            alpha_fc1[j], alpha_fc2[j]      — host floats: act_scale × weight_scale
        scales_dev (per-merger lists):
            act_fc1[j], act_fc2[j]          — fp16 device scalar ptrs (amax/448)
        dims:
            Nin, Din, Nout, Dmid, Dout
    """
    Nin = int(dims["Nin"])     # 1024 — patches per tap (post-ViT, pre-merge)
    Din = int(dims["Din"])     # 1024 — Qwen3VL ViT hidden_size
    Nout = int(dims["Nout"])   # 256  — Nin // 4
    Dmid = int(dims["Dmid"])   # 4096 — Din * spatial_merge_size**2
    Dout = int(dims["Dout"])   # 2048 — backbone_embedding_dim

    ln_out      = int(bufs["ln_out"])
    fp8_scratch = int(bufs["fp8_scratch"])
    fc1_out     = int(bufs["fc1_out"])

    for j in range(3):
        in_ptr  = int(bufs["in"][j])
        out_ptr = int(bufs["out"][j])

        # 1. LayerNorm(4096, eps=1e-6) — reshape to (Nout, Dmid) is a no-op pointer-wise.
        fvk.layer_norm_fp16(
            in_ptr, int(weights["norm_w"][j]), int(weights["norm_b"][j]),
            ln_out, Nout, Dmid, 1e-6, int(stream),
        )

        # 2-5. fc1 (GELU) → fc2.
        if use_fp8:
            fvk.quantize_fp8_static_fp16(
                ln_out, fp8_scratch, int(scales_dev["act_fc1"][j]),
                Nout * Dmid, int(stream),
            )
            gemm.fp8_nn_gelu_bias(
                fp8_scratch, int(weights["fc1_w"][j]),
                fc1_out,     int(weights["fc1_b"][j]),
                Nout, Dmid, Dmid,
                float(weights["alpha_fc1"][j]), int(stream),
            )
            fvk.quantize_fp8_static_fp16(
                fc1_out, fp8_scratch, int(scales_dev["act_fc2"][j]),
                Nout * Dmid, int(stream),
            )
            gemm.fp8_nn_bias(
                fp8_scratch, int(weights["fc2_w"][j]),
                out_ptr,     int(weights["fc2_b"][j]),
                Nout, Dout, Dmid,
                float(weights["alpha_fc2"][j]), int(stream),
            )
        else:
            gemm.fp16_nn(ln_out, int(weights["fc1_w_fp16"][j]), fc1_out, Nout, Dmid, Dmid, int(stream))
            fvk.add_bias_fp16(fc1_out, int(weights["fc1_b"][j]), Nout, Dmid, int(stream))
            fvk.gelu_inplace_fp16(fc1_out, Nout * Dmid, int(stream))
            gemm.fp16_nn(fc1_out, int(weights["fc2_w_fp16"][j]), out_ptr, Nout, Dout, Dmid, int(stream))
            fvk.add_bias_fp16(out_ptr, int(weights["fc2_b"][j]), Nout, Dout, int(stream))


def qwen3vl_llm_forward(gemm, fvk, bufs, weights, dims,
                        scales_dev, *, attn, stream: int = 0,
                        layers_subset=None, fp16_layers=()) -> None:
    """16 truncated Qwen3-VL LLM decoder layers.

    Per layer (input ``h``, in-place residual update):

        xn = RMSNorm(h, in_ln_w, eps=1e-6)                  — fp16 (S, D)
        xn_fp8 = quantize_fp8(xn, act_qkv_scale)             — shared input fp8

        Q = fp8_nn_dev(xn_fp8, q_w, [d_act_qkv, d_w_q])     — (S, NHQ*HD = 2048)
        K = fp8_nn_dev(xn_fp8, k_w, [d_act_qkv, d_w_k])     — (S, NHKV*HD = 1024)
        V = fp8_nn_dev(xn_fp8, v_w, [d_act_qkv, d_w_v])

        # Per-head Q/K RMSNorm — view (S, NH*HD) as (S*NH, HD), apply with
        # weight (HD,). RMSNorm normalizes over last dim and applies elementwise
        # weight; per-head independence and weight sharing across heads make
        # the flat-and-norm equivalent to per-head loop.
        rms_norm_fp16(Q, q_norm_w, Q, S*NHQ,  HD, eps=1e-6)
        rms_norm_fp16(K, k_norm_w, K, S*NHKV, HD, eps=1e-6)

        # M-RoPE — split-half rotation with cos/sin built host-side from
        # 3-axis position_ids (frontend's set_prompt builds these).
        rope_rotate_half_fp16(Q, cos, sin, S, NHQ,  HD)
        rope_rotate_half_fp16(K, cos, sin, S, NHKV, HD)

        # GQA expand: K, V from NHKV=8 to NHQ=16 (factor=2)
        gpu_repeat_interleave_heads(K, K_exp, S, NHKV, HD, 2)
        gpu_repeat_interleave_heads(V, V_exp, S, NHKV, HD, 2)

        O = attn.run("llm", li, q_seq=S, kv_seq=S)            — MHA 16×128
        O_fp8 = quantize_fp8(O, act_o_scale)
        o = fp8_nn_dev(O_fp8, o_w, [d_act_o, d_w_o])
        h += o                                                 (residual 1)

        xn = RMSNorm(h, post_ln_w, eps=1e-6)
        xn_fp8 = quantize_fp8(xn, act_gateup_scale)
        gate = fp8_nn_dev(xn_fp8, gate_w, [d_act_gu, d_w_gate])  — (S, FF=6144)
        up   = fp8_nn_dev(xn_fp8, up_w,   [d_act_gu, d_w_up])
        gu_fp8 = silu_mul_split_fp8_fp16(gate, up, gu_fp8, S*FF, d_act_down)
        down = fp8_nn_dev(gu_fp8, down_w, [d_act_down, d_w_down])  — (S, D)
        h += down                                              (residual 2)

        if li ∈ deepstack_layers (default {0,1,2}):
            h += deepstack_inject[li]                          (S, D) pre-expanded

    DeepStack injection: HF Qwen3VL injects at LLM layers ``[0, 1, 2]``
    (post-residual-2). The injection adds ``deepstack_features[li]`` only at
    visual-token positions. We pre-expand to a full ``(S, D)`` buffer host-side
    (zeros at non-visual positions), so the kernel call is a uniform residual_add.

    Args:
        bufs:
            h, xn          — fp16 (S, D)
            xn_fp8         — fp8  (S, D)
            Q              — fp16 (S, NHQ*HD)
            K, V           — fp16 (S, NHKV*HD)
            K_exp, V_exp   — fp16 (S, NHQ*HD)  (GQA-expanded for MHA kernel)
            o_proj_out     — fp16 (S, D)
            gate_out, up_out — fp16 (S, FF)
            gu_fp8         — fp8  (S, FF)
        weights (length-num_layers lists):
            in_ln_w, post_ln_w        — fp16 (D,)
            q_norm_w, k_norm_w        — fp16 (HD,)
            q_w, k_w, v_w, o_w        — fp8  ([K, N])
            d_w_q/k/v/o               — fp32 dev scalar (per-tensor weight scale)
            gate_w, up_w, down_w      — fp8
            d_w_gate, d_w_up, d_w_down — fp32 dev scalar
            cos, sin                  — fp16 dev ptr (S, HD) shared across layers
        scales_dev (length-num_layers):
            act_qkv, act_o, act_gateup, act_down  — fp32 dev scalar ptrs
        dims:
            S, D, NHQ, NHKV, HD, FF
        deepstack (optional dict):
            inject_ptrs  — list[N=16] of int (0 = no inject); length-16 even if
                only 0/1/2 are non-zero. Default behaviour: read from
                ``weights["deepstack_inject"]`` if present, else no injection.
    """
    S    = int(dims["S"])
    D    = int(dims["D"])
    NHQ  = int(dims["NHQ"])
    NHKV = int(dims["NHKV"])
    HD   = int(dims["HD"])
    FF   = int(dims["FF"])
    GQA  = NHQ // NHKV

    h_ptr      = int(bufs["h"])
    xn_ptr     = int(bufs["xn"])
    xn_fp8_ptr = int(bufs["xn_fp8"])
    Q_ptr      = int(bufs["Q"])
    K_ptr      = int(bufs["K"])
    V_ptr      = int(bufs["V"])
    K_exp_ptr  = int(bufs["K_exp"])
    V_exp_ptr  = int(bufs["V_exp"])
    o_out_ptr  = int(bufs["o_proj_out"])
    gate_ptr   = int(bufs["gate_out"])
    up_ptr     = int(bufs["up_out"])
    gu_fp8_ptr = int(bufs["gu_fp8"])
    cos_ptr    = int(weights["cos"])
    sin_ptr    = int(weights["sin"])

    inject_ptrs = weights.get("deepstack_inject", [0] * 16)
    layer_iter = range(16) if layers_subset is None else list(layers_subset)
    # Per-layer precision protection: layers in ``fp16_layers`` run their
    # QKV / O / gate / up GEMMs in fp16 (no input fp8 quant) to protect the
    # large visual-token activation spikes that wreck per-tensor FP8 on the
    # first decoder layers. The down GEMM stays FP8 (its gate*up input is
    # already smoothed by SiLU). Requires fp16 weight ptrs in
    # ``weights["{q,k,v,o,gate,up}_w_fp16"]``.
    fp16_set = set(fp16_layers or ())

    for li in layer_iter:
        slots = attn.get_slot_ptrs("llm", li)
        # Backend's "llm" site uses single Q/K/V/O slots; we override K/V
        # with GQA-expanded versions before the MHA kernel.
        hi_prec = li in fp16_set

        # ── Pre-attn RMSNorm ───────────────────────────────────────────
        fvk.rms_norm_fp16(
            h_ptr, int(weights["in_ln_w"][li]), xn_ptr,
            S, D, 1e-6, int(stream),
        )

        if hi_prec:
            # fp16-protected QKV: GEMM straight off the fp16 RMSNorm output.
            gemm.fp16_nn(xn_ptr, int(weights["q_w_fp16"][li]), Q_ptr,
                         S, NHQ * HD, D, int(stream))
            gemm.fp16_nn(xn_ptr, int(weights["k_w_fp16"][li]), K_ptr,
                         S, NHKV * HD, D, int(stream))
            gemm.fp16_nn(xn_ptr, int(weights["v_w_fp16"][li]), V_ptr,
                         S, NHKV * HD, D, int(stream))
        else:
            # ── Quantize xn for Q/K/V (single shared act scale) ────────
            fvk.quantize_fp8_static_fp16(
                xn_ptr, xn_fp8_ptr, int(scales_dev["act_qkv"][li]),
                S * D, int(stream),
            )
            # ── 3 split FP8 GEMMs (per-tensor act+weight scales) ───────
            d_act_qkv = int(scales_dev["act_qkv"][li])
            gemm.fp8_nn_dev(
                xn_fp8_ptr, int(weights["q_w"][li]), Q_ptr,
                S, NHQ * HD, D,
                d_act_qkv, int(weights["d_w_q"][li]), int(stream),
            )
            gemm.fp8_nn_dev(
                xn_fp8_ptr, int(weights["k_w"][li]), K_ptr,
                S, NHKV * HD, D,
                d_act_qkv, int(weights["d_w_k"][li]), int(stream),
            )
            gemm.fp8_nn_dev(
                xn_fp8_ptr, int(weights["v_w"][li]), V_ptr,
                S, NHKV * HD, D,
                d_act_qkv, int(weights["d_w_v"][li]), int(stream),
            )

        # ── Per-head q_norm / k_norm (BEFORE M-RoPE — Qwen3 standard) ──
        fvk.rms_norm_fp16(
            Q_ptr, int(weights["q_norm_w"][li]), Q_ptr,
            S * NHQ, HD, 1e-6, int(stream),
        )
        fvk.rms_norm_fp16(
            K_ptr, int(weights["k_norm_w"][li]), K_ptr,
            S * NHKV, HD, 1e-6, int(stream),
        )

        # ── M-RoPE on Q and K ──────────────────────────────────────────
        fvk.rope_rotate_half_fp16(Q_ptr, cos_ptr, sin_ptr, S, NHQ,  HD, int(stream))
        fvk.rope_rotate_half_fp16(K_ptr, cos_ptr, sin_ptr, S, NHKV, HD, int(stream))

        # ── GQA expand: K, V from NHKV → NHQ heads ─────────────────────
        fvk.gpu_repeat_interleave_heads(K_ptr, K_exp_ptr, S, NHKV, HD, GQA, int(stream))
        fvk.gpu_repeat_interleave_heads(V_ptr, V_exp_ptr, S, NHKV, HD, GQA, int(stream))

        # ── MHA via attn backend ───────────────────────────────────────
        # Backend reads the LLM site's Q/K/V/O slots; we wrote the expanded
        # K_exp/V_exp into those slot ptrs at allocation time so the kernel
        # call is unchanged.
        attn.run("llm", li, q_seq=S, kv_seq=S, stream=int(stream))

        if hi_prec:
            gemm.fp16_nn(int(slots["O"]), int(weights["o_w_fp16"][li]),
                         o_out_ptr, S, D, NHQ * HD, int(stream))
        else:
            # ── Quantize O for o_proj ──────────────────────────────────
            fvk.quantize_fp8_static_fp16(
                int(slots["O"]), xn_fp8_ptr, int(scales_dev["act_o"][li]),
                S * NHQ * HD, int(stream),
            )
            gemm.fp8_nn_dev(
                xn_fp8_ptr, int(weights["o_w"][li]), o_out_ptr,
                S, D, NHQ * HD,
                int(scales_dev["act_o"][li]), int(weights["d_w_o"][li]),
                int(stream),
            )

        # ── Residual 1 ────────────────────────────────────────────────
        fvk.residual_add_fp16(h_ptr, o_out_ptr, S * D, int(stream))

        # ── Pre-FFN RMSNorm ───────────────────────────────────────────
        fvk.rms_norm_fp16(
            h_ptr, int(weights["post_ln_w"][li]), xn_ptr,
            S, D, 1e-6, int(stream),
        )

        if hi_prec:
            gemm.fp16_nn(xn_ptr, int(weights["gate_w_fp16"][li]), gate_ptr,
                         S, FF, D, int(stream))
            gemm.fp16_nn(xn_ptr, int(weights["up_w_fp16"][li]), up_ptr,
                         S, FF, D, int(stream))
        else:
            # ── Quantize xn for gate+up (shared act scale) ────────────
            fvk.quantize_fp8_static_fp16(
                xn_ptr, xn_fp8_ptr, int(scales_dev["act_gateup"][li]),
                S * D, int(stream),
            )
            d_act_gu = int(scales_dev["act_gateup"][li])
            # ── gate / up FP8 GEMMs → fp16 outputs ────────────────────
            gemm.fp8_nn_dev(
                xn_fp8_ptr, int(weights["gate_w"][li]), gate_ptr,
                S, FF, D,
                d_act_gu, int(weights["d_w_gate"][li]), int(stream),
            )
            gemm.fp8_nn_dev(
                xn_fp8_ptr, int(weights["up_w"][li]), up_ptr,
                S, FF, D,
                d_act_gu, int(weights["d_w_up"][li]), int(stream),
            )
        if hi_prec:
            # fp16-protected FFN out: SiLU(gate)*up in fp16, fp16 down GEMM.
            fvk.silu_inplace_fp16(gate_ptr, S * FF, int(stream))
            fvk.mul_fp16(gate_ptr, up_ptr, gate_ptr, S * FF, int(stream))
            gemm.fp16_nn(gate_ptr, int(weights["down_w_fp16"][li]), o_out_ptr,
                         S, D, FF, int(stream))
        else:
            # ── SiLU(gate) * up → directly FP8 (fused) ────────────────
            fvk.silu_mul_split_fp8_fp16(
                gate_ptr, up_ptr, gu_fp8_ptr, S * FF,
                int(scales_dev["act_down"][li]), int(stream),
            )
            # ── down GEMM → fp16 ──────────────────────────────────────
            gemm.fp8_nn_dev(
                gu_fp8_ptr, int(weights["down_w"][li]), o_out_ptr,
                S, D, FF,
                int(scales_dev["act_down"][li]), int(weights["d_w_down"][li]),
                int(stream),
            )

        # ── Residual 2 ────────────────────────────────────────────────
        fvk.residual_add_fp16(h_ptr, o_out_ptr, S * D, int(stream))

        # ── DeepStack injection (HF: layers 0, 1, 2) ──────────────────
        inject_ptr = int(inject_ptrs[li]) if li < len(inject_ptrs) else 0
        if inject_ptr != 0:
            fvk.residual_add_fp16(h_ptr, inject_ptr, S * D, int(stream))


def vl_self_attn_forward(gemm, fvk, bufs, weights, dims,
                         scales_dev, *, attn, stream: int = 0,
                         layers_subset=None, use_fp8=True) -> None:
    """4-layer ``SelfAttentionTransformer`` (diffusers ``BasicTransformerBlock``,
    ``norm_type="layer_norm"``, ``activation_fn="gelu-approximate"``,
    ``positional_embeddings=None`` per N1.7 config).

    Per layer (input ``h``, in-place residual update):

        xn   = LayerNorm(h, norm1_w, norm1_b, eps=1e-5)
        xn_q = quantize_fp8(xn, act_qkv_scale)
        Q    = fp8_nn_bias(xn_q, q_w, q_b, alpha_q)        — (T, 2048)
        K    = fp8_nn_bias(xn_q, k_w, k_b, alpha_k)        — (T, 2048)
        V    = fp8_nn_bias(xn_q, v_w, v_b, alpha_v)        — (T, 2048)
        O    = attn.run("vl_self_attn", layer, T, T)        — MHA 32×64
        O_q  = quantize_fp8(O, act_o_scale)
        o    = fp8_nn_bias(O_q, o_w, o_b, alpha_o)
        h   += o                                            (residual 1)

        xn   = LayerNorm(h, norm3_w, norm3_b, eps=1e-5)
        xn_q = quantize_fp8(xn, act_fc1_scale)
        h1   = fp8_nn_gelu_bias(xn_q, fc1_w, fc1_b, alpha_fc1)  — (T, 8192)
        h1_q = quantize_fp8(h1, act_fc2_scale)
        f2   = fp8_nn_bias(h1_q, fc2_w, fc2_b, alpha_fc2)
        h   += f2                                           (residual 2)

    Q/K/V/O/logits live in attn-backend ``vl_self_attn`` slots — single
    buffer set shared across all 4 layers (layer-sequential).

    Args:
        bufs:
            h            — running hidden state, fp16 (T, D), in-place
            xn           — LN scratch, fp16 (T, D)
            xn_fp8       — FP8 quant scratch, fp8 (T, D)
            o_proj_out   — fp16 (T, D) scratch for o_proj output
            fc1_out      — fp16 (T, ff_inner=8192) scratch
            fc1_fp8      — fp8  (T, ff_inner) scratch
        weights (each is a length-num_layers list/sequence):
            norm1_w/b, norm3_w/b
            q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b   (FP8 weights, fp16 biases)
            fc1_w, fc1_b, fc2_w, fc2_b
            alpha_q, alpha_k, alpha_v, alpha_o      (host floats)
            alpha_fc1, alpha_fc2                    (host floats)
        scales_dev (each length-num_layers):
            act_qkv, act_o, act_fc1, act_fc2        (fp32 device scalar ptrs)
        dims:
            T (= seq_len), D (= 2048), NH (= 32), HD (= 64), ff_inner (= 8192)
        attn: ThorGrootN17AttnBackend (provides Q/K/V/O slot ptrs + run dispatch)
        layers_subset: iterable of layer indices to run; default = all 4.
            (Used by Phase 3b.2 cosine tests to validate one layer in isolation.)
    """
    T  = int(dims["T"])
    D  = int(dims["D"])
    NH = int(dims["NH"])
    HD = int(dims["HD"])
    FF = int(dims["ff_inner"])

    h_ptr        = int(bufs["h"])
    xn_ptr       = int(bufs["xn"])
    xn_fp8_ptr   = int(bufs["xn_fp8"])
    o_proj_out   = int(bufs["o_proj_out"])
    fc1_out_ptr  = int(bufs["fc1_out"])
    fc1_fp8_ptr  = int(bufs["fc1_fp8"])

    layer_iter = range(4) if layers_subset is None else list(layers_subset)

    for li in layer_iter:
        slots = attn.get_slot_ptrs("vl_self_attn", li)
        Q_ptr, K_ptr, V_ptr, O_ptr = slots["Q"], slots["K"], slots["V"], slots["O"]

        # ── Pre-attn LayerNorm ──────────────────────────────────────────
        fvk.layer_norm_fp16(
            h_ptr, int(weights["norm1_w"][li]), int(weights["norm1_b"][li]),
            xn_ptr, T, D, 1e-5, int(stream),
        )

        # ── Q / K / V projections (single shared act scale for FP8) ──────
        if use_fp8:
            fvk.quantize_fp8_static_fp16(
                xn_ptr, xn_fp8_ptr, int(scales_dev["act_qkv"][li]),
                T * D, int(stream),
            )
            gemm.fp8_nn_bias(
                xn_fp8_ptr, int(weights["q_w"][li]), Q_ptr, int(weights["q_b"][li]),
                T, D, D, float(weights["alpha_q"][li]), int(stream),
            )
            gemm.fp8_nn_bias(
                xn_fp8_ptr, int(weights["k_w"][li]), K_ptr, int(weights["k_b"][li]),
                T, D, D, float(weights["alpha_k"][li]), int(stream),
            )
            gemm.fp8_nn_bias(
                xn_fp8_ptr, int(weights["v_w"][li]), V_ptr, int(weights["v_b"][li]),
                T, D, D, float(weights["alpha_v"][li]), int(stream),
            )
        else:
            for wk, bk, out in (("q_w_fp16", "q_b", Q_ptr),
                                ("k_w_fp16", "k_b", K_ptr),
                                ("v_w_fp16", "v_b", V_ptr)):
                gemm.fp16_nn(xn_ptr, int(weights[wk][li]), out, T, D, D, int(stream))
                fvk.add_bias_fp16(out, int(weights[bk][li]), T, D, int(stream))

        # ── MHA via attn backend (kernel pre-fills logits as needed) ────
        attn.run("vl_self_attn", li, q_seq=T, kv_seq=T, stream=int(stream))

        # ── o_proj ──────────────────────────────────────────────────────
        if use_fp8:
            fvk.quantize_fp8_static_fp16(
                O_ptr, xn_fp8_ptr, int(scales_dev["act_o"][li]),
                T * D, int(stream),
            )
            gemm.fp8_nn_bias(
                xn_fp8_ptr, int(weights["o_w"][li]), o_proj_out, int(weights["o_b"][li]),
                T, D, D, float(weights["alpha_o"][li]), int(stream),
            )
        else:
            gemm.fp16_nn(O_ptr, int(weights["o_w_fp16"][li]), o_proj_out, T, D, D, int(stream))
            fvk.add_bias_fp16(o_proj_out, int(weights["o_b"][li]), T, D, int(stream))

        # ── Residual 1: h += o_proj_out ────────────────────────────────
        fvk.residual_add_fp16(h_ptr, o_proj_out, T * D, int(stream))

        # ── Pre-FF LayerNorm ───────────────────────────────────────────
        fvk.layer_norm_fp16(
            h_ptr, int(weights["norm3_w"][li]), int(weights["norm3_b"][li]),
            xn_ptr, T, D, 1e-5, int(stream),
        )

        # ── FF: 2048 → 8192 (GELU) → 2048 ──────────────────────────────
        if use_fp8:
            fvk.quantize_fp8_static_fp16(
                xn_ptr, xn_fp8_ptr, int(scales_dev["act_fc1"][li]),
                T * D, int(stream),
            )
            gemm.fp8_nn_gelu_bias(
                xn_fp8_ptr, int(weights["fc1_w"][li]), fc1_out_ptr,
                int(weights["fc1_b"][li]),
                T, FF, D, float(weights["alpha_fc1"][li]), int(stream),
            )
            fvk.quantize_fp8_static_fp16(
                fc1_out_ptr, fc1_fp8_ptr, int(scales_dev["act_fc2"][li]),
                T * FF, int(stream),
            )
            gemm.fp8_nn_bias(
                fc1_fp8_ptr, int(weights["fc2_w"][li]), o_proj_out,
                int(weights["fc2_b"][li]),
                T, D, FF, float(weights["alpha_fc2"][li]), int(stream),
            )
        else:
            gemm.fp16_nn(xn_ptr, int(weights["fc1_w_fp16"][li]), fc1_out_ptr, T, FF, D, int(stream))
            fvk.add_bias_fp16(fc1_out_ptr, int(weights["fc1_b"][li]), T, FF, int(stream))
            fvk.gelu_inplace_fp16(fc1_out_ptr, T * FF, int(stream))
            gemm.fp16_nn(fc1_out_ptr, int(weights["fc2_w_fp16"][li]), o_proj_out, T, D, FF, int(stream))
            fvk.add_bias_fp16(o_proj_out, int(weights["fc2_b"][li]), T, D, int(stream))

        # ── Residual 2: h += fc2_out ───────────────────────────────────
        fvk.residual_add_fp16(h_ptr, o_proj_out, T * D, int(stream))


def dit_forward(gemm, fvk, bufs, weights, dims,
                *, attn, stream: int = 0, layers_subset=None) -> None:
    """32-layer ``AlternateVLDiT`` (interleave_self_attention=True,
    attend_text_every_n_blocks=2). All bf16 GEMMs (no FP8 — N1.6 pattern).

    Per HF ``AlternateVLDiT.forward`` (dit.py:339):
      * Odd layers (idx % 2 == 1) → SELF-attn over (B, Sa=41, D=1536)
      * Even layers (idx % 2 == 0) → CROSS-attn to encoder_hidden_states
        (post-vlsa backbone features). Cross-target alternates every 2
        cross-blocks (every 4 layers): idx ∈ {0, 4, 8, ...} → text
        (non-image positions); idx ∈ {2, 6, 10, ...} → image positions.

    Per BasicTransformerBlock with ``norm_type="ada_norm"`` and
    ``activation_fn="gelu-approximate"``:

        # AdaLN: norm1.linear(silu(temb)).chunk(2) → (shift, scale)
        # Done once per layer at set_prompt; pipeline reads pre-computed
        # (shift_msa[li], scale_msa[li]) from weights.
        ada_layer_norm_fp16(h, scale_msa[li], shift_msa[li], xn, Sa, D)

        # Attention dispatch via AttentionBackend
        if li % 2 == 1:        # self-attn
          Q = bf16_nn_bias(xn, q_w[li], q_b[li])  — (Sa, D)
          K = bf16_nn_bias(xn, k_w[li], k_b[li])
          V = bf16_nn_bias(xn, v_w[li], v_b[li])
          attn.run("dit_self", li, Sa, Sa)
        else:                  # cross-attn
          Q = bf16_nn_bias(xn, q_w[li], q_b[li])  — (Sa, D)
          # K, V pre-computed from encoder_hidden_states at set_prompt
          # and resident in dit_cross site's per-layer K/V slots.
          attn.run("dit_cross", li, Sa, Skv_target_li)
        o = bf16_nn_bias(O, o_w[li], o_b[li])
        h += o                                              # residual 1

        # Pre-FF LayerNorm (no affine — DiT default)
        layer_norm_no_affine_fp16(h, xn, Sa, D, eps=1e-5)

        # FFN: GELU(tanh-approx) (NOT GeGLU per ckpt shapes 1536→6144→1536)
        ff = bf16_nn_bias_gelu(xn, ff_proj_w[li], ff_proj_b[li])  — (Sa, 6144)
        out = bf16_nn_bias(ff, ff_down_w[li], ff_down_b[li])      — (Sa, D)
        h += out                                            # residual 2

    Args:
        bufs:
            h, xn          — bf16 (Sa, D)
            Q, K, V, O     — bf16 (Sa, D) for dit_self; (Sa, kv_dim_per_layer) per-layer for cross
            o_proj_out     — bf16 (Sa, D)
            ff_proj_out    — bf16 (Sa, 6144)
        weights (length 32 lists):
            scale_msa, shift_msa  — bf16 (D,) per layer (pre-computed at set_prompt)
            q_w, q_b              — bf16 ([D, D]) (Q always 1536→1536)
            k_w, k_b, v_w, v_b    — bf16; shape varies per layer (self: D, cross: 2048)
            o_w (1536, 1536), o_b — bf16
            ff_proj_w (6144, 1536), ff_proj_b
            ff_down_w (1536, 6144), ff_down_b
        dims:
            Sa (= 41), D (= 1536), NH (= 32), HD (= 48), FF (= 6144),
            Skv_text, Skv_image
    """
    Sa = int(dims["Sa"])
    D = int(dims["D"])
    FF = int(dims["FF"])
    Skv_text = int(dims.get("Skv_text", 0))
    Skv_image = int(dims.get("Skv_image", 0))

    h_ptr      = int(bufs["h"])
    xn_ptr     = int(bufs["xn"])
    o_out_ptr  = int(bufs["o_proj_out"])
    ff_out_ptr = int(bufs["ff_proj_out"])

    layer_iter = range(32) if layers_subset is None else list(layers_subset)

    for li in layer_iter:
        is_self = (li % 2 == 1)
        # Backend's ``dit_self`` and ``dit_cross`` sites are indexed
        # cross-only / self-only (16 entries each, NOT the full 0..31
        # layer index). Map here.
        j_attn = (li - 1) // 2 if is_self else li // 2

        # ── AdaLN modulated norm1 (bf16) ──────────────────────────────
        fvk.ada_layer_norm_bf16(
            h_ptr,
            int(weights["scale_msa"][li]), int(weights["shift_msa"][li]),
            xn_ptr, Sa, D, 1e-5, int(stream),
        )

        # ── Q always projects from D → D ──────────────────────────────
        if is_self:
            slots = attn.get_slot_ptrs("dit_self", j_attn)
        else:
            slots = attn.get_slot_ptrs("dit_cross", j_attn)
        Q_ptr, K_ptr, V_ptr, O_ptr = slots["Q"], slots["K"], slots["V"], slots["O"]

        # bf16_nn_bias / bf16_nn_bias_gelu epilogues hit
        # CUBLAS_STATUS_NOT_SUPPORTED at M=Sa=41 (M not 16-aligned). Match
        # N1.6's calibrate path: bf16_nn (no epilogue) + explicit
        # add_bias_bf16. Slightly more launches but works on every shape.
        gemm.bf16_nn(xn_ptr, int(weights["q_w"][li]),
                      Q_ptr, Sa, D, D, int(stream))
        fvk.add_bias_bf16(Q_ptr, int(weights["q_b"][li]),
                           Sa, D, int(stream))

        if is_self:
            gemm.bf16_nn(xn_ptr, int(weights["k_w"][li]),
                          K_ptr, Sa, D, D, int(stream))
            fvk.add_bias_bf16(K_ptr, int(weights["k_b"][li]),
                               Sa, D, int(stream))
            gemm.bf16_nn(xn_ptr, int(weights["v_w"][li]),
                          V_ptr, Sa, D, D, int(stream))
            fvk.add_bias_bf16(V_ptr, int(weights["v_b"][li]),
                               Sa, D, int(stream))
            attn.run("dit_self", j_attn, q_seq=Sa, kv_seq=Sa, stream=int(stream))
        else:
            target_text = (li % 4 == 0)
            kv_seq = Skv_text if target_text else Skv_image
            attn.run("dit_cross", j_attn, q_seq=Sa, kv_seq=kv_seq,
                     stream=int(stream))

        gemm.bf16_nn(O_ptr, int(weights["o_w"][li]),
                      o_out_ptr, Sa, D, D, int(stream))
        fvk.add_bias_bf16(o_out_ptr, int(weights["o_b"][li]),
                           Sa, D, int(stream))
        fvk.residual_add(h_ptr, o_out_ptr, Sa * D, int(stream))

        # ── Pre-FF LayerNorm (no affine — DiT default) ───────────────
        fvk.layer_norm_no_affine_bf16(
            h_ptr, xn_ptr, Sa, D, 1e-5, int(stream),
        )

        # ── FFN: GELU(tanh-approx) ────────────────────────────────────
        # The FFN GEMMs are the compute-bound part of the (M=41) DiT, so an
        # FP8 path here is a real win (≈1.8× on the up-projection) and fuses
        # the bias+GELU into the GEMM epilogue. Activated when calibrated FP8
        # FFN weights/scales are supplied; otherwise the bf16 path runs. The
        # attention GEMMs stay bf16 — at M=41 they are launch-bound, so FP8
        # gives no speedup there.
        if "ff_proj_w_fp8" in weights:
            fvk.quantize_fp8_static(
                xn_ptr, int(bufs["xn_fp8"]),
                int(weights["act_fc1_scale"][li]), Sa * D, int(stream))
            gemm.fp8_nn_gelu_bias(
                int(bufs["xn_fp8"]), int(weights["ff_proj_w_fp8"][li]),
                int(bufs["ff_fp16"]), int(weights["ff_proj_b"][li]),
                Sa, FF, D, float(weights["alpha_fc1"][li]), int(stream))
            fvk.quantize_fp8_static_fp16(
                int(bufs["ff_fp16"]), int(bufs["ff_fp8"]),
                int(weights["act_fc2_scale"][li]), Sa * FF, int(stream))
            gemm.fp8_nn_bias_bf16(
                int(bufs["ff_fp8"]), int(weights["ff_down_w_fp8"][li]),
                o_out_ptr, int(weights["ff_down_b"][li]),
                Sa, D, FF, float(weights["alpha_fc2"][li]), int(stream))
        else:
            gemm.bf16_nn(xn_ptr, int(weights["ff_proj_w"][li]),
                          ff_out_ptr, Sa, FF, D, int(stream))
            fvk.add_bias_bf16(ff_out_ptr, int(weights["ff_proj_b"][li]),
                               Sa, FF, int(stream))
            fvk.gelu_inplace(ff_out_ptr, Sa * FF, int(stream))
            gemm.bf16_nn(ff_out_ptr, int(weights["ff_down_w"][li]),
                          o_out_ptr, Sa, D, FF, int(stream))
            fvk.add_bias_bf16(o_out_ptr, int(weights["ff_down_b"][li]),
                               Sa, D, int(stream))
        fvk.residual_add(h_ptr, o_out_ptr, Sa * D, int(stream))


def embodiment_state_encode(gemm, fvk, bufs, weights, dims, *,
                            stream: int = 0) -> None:
    """state ``(1, 132)`` → state_features ``(1, 1536)``.

    Per-embodiment ``CategorySpecificMLP`` (gr00t_n1d7.py:203). The frontend's
    ``_load_weights`` slices the (32, ·, ·) tensors to the active embodiment
    slot at load time; this forward uses the pre-sliced weights directly.

      h1 = state @ l1_w + l1_b              # (1, 1024)
      h1 = ReLU(h1)
      out = h1 @ l2_w + l2_b                # (1, 1536)

    GEMMs use ``bf16_nn`` + ``add_bias_bf16`` rather than the fused
    ``bf16_nn_bias`` epilogue, which returns ``CUBLAS_NOT_SUPPORTED`` on Thor
    SM110 (same split every live Thor GEMM uses).

    Args:
        bufs: state_in (1, 132), h1 (1, 1024), out (1, 1536) — bf16
        weights: l1_w (132, 1024), l1_b (1024,), l2_w (1024, 1536), l2_b (1536,)
        dims: M (= 1, single state token after flatten)
    """
    M = int(dims["M"])
    in_dim = int(dims["state_dim"])    # 132
    h_dim = int(dims["h_dim"])         # 1024
    out_dim = int(dims["out_dim"])     # 1536
    gemm.bf16_nn(
        int(bufs["state_in"]), int(weights["l1_w"]), int(bufs["h1"]),
        M, h_dim, in_dim, int(stream),
    )
    fvk.add_bias_bf16(int(bufs["h1"]), int(weights["l1_b"]), M, h_dim, int(stream))
    fvk.relu_inplace_fp16(int(bufs["h1"]), M * h_dim, int(stream))
    gemm.bf16_nn(
        int(bufs["h1"]), int(weights["l2_w"]), int(bufs["out"]),
        M, out_dim, h_dim, int(stream),
    )
    fvk.add_bias_bf16(int(bufs["out"]), int(weights["l2_b"]), M, out_dim, int(stream))


def embodiment_action_encode(gemm, fvk, bufs, weights, dims, *,
                             stream: int = 0) -> None:
    """noisy_action ``(40, 132)`` + timestep_emb ``(1, 1536)`` →
    action_features ``(40, 1536)``.

    Per ``MultiEmbodimentActionEncoder`` (gr00t_n1d7.py:391). 3-layer MLP with
    timestep concat after the first hidden layer. Pre-sliced per-embodiment.
    NOTE the activation pattern — unlike the ReLU ``CategorySpecificMLP``
    used by state-encode/action-decode, this encoder is NO-act → SiLU → NO-act:

      a   = noisy @ W1 + b1                  # (40, 1536)  NO activation
      cat = concat([a, tau_emb], dim=-1)     # (40, 3072)  tau = timestep emb
      i2  = SiLU(cat @ W2 + b2)              # (40, 1536)  swish, NOT relu
      out = i2 @ W3 + b3                      # (40, 1536)  NO activation

    ``tau_emb`` is the timestep embedding broadcast across the T action rows
    (identical per row). GEMMs use ``bf16_nn`` + ``add_bias_bf16`` (the fused
    ``bf16_nn_bias`` epilogue is ``CUBLAS_NOT_SUPPORTED`` on Thor SM110); the
    concat uses ``concat2_bf16`` (a true two-input concat — ``gpu_strided_copy``
    is a narrowing *gather*, not a scatter); SiLU runs through an fp16 round
    trip since the kernel library exposes it only for fp16.

    Args:
        bufs:
            noisy        — bf16 (T=40, 132)
            i1           — bf16 (T, 1536)  W1 output (a)
            tau_emb      — bf16 (T, 1536)  timestep emb broadcast over T
            concat_buf   — bf16 (T, 3072)  concat([i1, tau_emb])
            i2           — bf16 (T, 1536)
            i2_fp16      — fp16 (T, 1536)  SiLU scratch
            out          — bf16 (T, 1536)
        weights: W1 (132, 1536), b1 (1536), W2 (3072, 1536), b2, W3 (1536, 1536), b3
        dims: T (= 40), action_dim (= 132), h_dim (= 1536), cat_dim (= 3072)
    """
    T = int(dims["T"])
    A = int(dims["action_dim"])
    H = int(dims["h_dim"])
    Cat = int(dims["cat_dim"])

    # Layer 1: (T, A) → (T, H), NO activation
    gemm.bf16_nn(
        int(bufs["noisy"]), int(weights["W1"]), int(bufs["i1"]),
        T, H, A, int(stream),
    )
    fvk.add_bias_bf16(int(bufs["i1"]), int(weights["b1"]), T, H, int(stream))

    # concat([i1, tau_emb]) → (T, 2H)
    fvk.concat2_bf16(
        int(bufs["i1"]), int(bufs["tau_emb"]), int(bufs["concat_buf"]),
        T, H, H, int(stream),
    )

    # Layer 2: (T, Cat) → (T, H), then SiLU (fp16 round trip)
    gemm.bf16_nn(
        int(bufs["concat_buf"]), int(weights["W2"]), int(bufs["i2"]),
        T, H, Cat, int(stream),
    )
    fvk.add_bias_bf16(int(bufs["i2"]), int(weights["b2"]), T, H, int(stream))
    fvk.cast_bf16_to_fp16(int(bufs["i2"]), int(bufs["i2_fp16"]), T * H, int(stream))
    fvk.silu_inplace_fp16(int(bufs["i2_fp16"]), T * H, int(stream))
    fvk.cast_fp16_to_bf16(int(bufs["i2_fp16"]), int(bufs["i2"]), T * H, int(stream))

    # Layer 3: (T, H) → (T, H), NO activation
    gemm.bf16_nn(
        int(bufs["i2"]), int(weights["W3"]), int(bufs["out"]),
        T, H, H, int(stream),
    )
    fvk.add_bias_bf16(int(bufs["out"]), int(weights["b3"]), T, H, int(stream))


def embodiment_action_decode(gemm, fvk, bufs, weights, dims, *,
                             stream: int = 0) -> None:
    """DiT proj_out_2 result ``(40, 1024)`` → velocity ``(40, 132)``.

    Per ``CategorySpecificMLP`` (gr00t_n1d7.py:416). 2-layer MLP with ReLU.
    Pre-sliced per-embodiment. GEMMs use ``bf16_nn`` + ``add_bias_bf16`` (the
    fused ``bf16_nn_bias`` epilogue is ``CUBLAS_NOT_SUPPORTED`` on Thor SM110).

      h = ReLU(in @ l1_w + l1_b)            # (40, 1024)
      out = h @ l2_w + l2_b                 # (40, 132)
    """
    T = int(dims["T"])
    in_dim = int(dims["in_dim"])       # 1024
    h_dim = int(dims["h_dim"])         # 1024
    out_dim = int(dims["out_dim"])     # 132

    gemm.bf16_nn(
        int(bufs["dit_out"]), int(weights["l1_w"]), int(bufs["h"]),
        T, h_dim, in_dim, int(stream),
    )
    fvk.add_bias_bf16(int(bufs["h"]), int(weights["l1_b"]), T, h_dim, int(stream))
    fvk.relu_inplace_fp16(int(bufs["h"]), T * h_dim, int(stream))
    gemm.bf16_nn(
        int(bufs["h"]), int(weights["l2_w"]), int(bufs["velocity"]),
        T, out_dim, h_dim, int(stream),
    )
    fvk.add_bias_bf16(int(bufs["velocity"]), int(weights["l2_b"]), T, out_dim, int(stream))
