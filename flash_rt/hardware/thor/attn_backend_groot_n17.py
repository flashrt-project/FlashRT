"""FlashRT — Thor attention backend for GROOT N1.7.

Mirrors the N1.6 ``ThorGrootAttnBackend`` shape (no Q/O aliasing, separate
QKV buffers, per-layer K/V only for cross-attention) but with N1.7-specific
sites and dimensions. The N1.6 backend module is left untouched.

Sites:

* ``vit``               — Qwen3-VL ViT, 24 layers, MHA 16h × 64hd, hidden 1024,
                          batch_axis=num_views (multi-view batched FMHA).
* ``llm``               — Qwen3-VL LLM truncated to 16 layers, MHA 16h × 128hd
                          (GQA pre-expanded to MHA at the kernel boundary —
                          same idiom as N1.6's qwen3 site).
* ``vl_self_attn``      — 4-layer BasicTransformerBlock self-attn,
                          MHA 32h × 64hd, hidden 2048.
* ``dit_self``          — 16 layers self-attn, MHA 32h × 48hd, hidden 1536.
* ``dit_cross``         — 16 cross-attn layers (the other half of the 32-block
                          AlternateVLDiT). KV target alternates between text
                          tokens and image tokens every 2 blocks per the
                          ``attend_text_every_n_blocks=2`` config; the spec
                          tracks the union (max of text-kv and image-kv) and
                          the pipeline passes the right ``kv_seq`` per call.

Pointer-stability rules from ``docs/extension/attention_backend.md`` §9 hold:
* ``get_slot_ptrs(site, layer_idx)`` is deterministic for the backend's lifetime.
* Output pointer (``run``) is the pre-allocated O slot for the site.
"""

from __future__ import annotations

from typing import Optional

from flash_rt.hardware.backend import AttentionBackendBase, AttentionSpec


# ════════════════════════════════════════════════════════════════════
#  Backend
# ════════════════════════════════════════════════════════════════════


class ThorGrootN17AttnBackend(AttentionBackendBase):
    """N1.7 attention backend on Thor (SM110)."""

    EXPECTED_SITES = ("vit", "llm", "vl_self_attn", "dit_self", "dit_cross")

    def __init__(self, spec: AttentionSpec, *,
                 vit_slots: dict, llm_slots: dict,
                 vl_self_attn_slots: dict,
                 dit_self_slots: dict, dit_cross_slots: dict) -> None:
        """
        Args:
            spec: AttentionSpec with exactly the sites in ``EXPECTED_SITES``.
            vit_slots: {qkv (int), O (int), D (int)} — same idiom as
                ``siglip`` in N1.6 (interleaved QKV in one buffer, hits
                ``fmha_strided_full``).
            llm_slots / vl_self_attn_slots / dit_self_slots: {
                ctx           — fvk.FvkContext or wrapper with .cpp;
                                each site keeps its own cuBLAS handle so
                                workspace state does not bleed across
                                sites (Myelin tactic stability).
                Q (int)       — Q buffer (shared across all layers in this site)
                K (int)       — K buffer
                V (int)       — V buffer
                O (int)       — O buffer (no Q/O aliasing).
                logits (int)  — P = QK^T scratch
                scale (float) — 1/sqrt(head_dim)
            }
            dit_cross_slots: same as dit_self but K/V replaced by per-layer
                lists ``K_layers`` / ``V_layers`` (precomputed by the
                DiT pipeline once per ``set_prompt`` — text and image KV
                are written into the right layer index based on which
                target that block attends to).
        """
        super().__init__(spec)

        got = set(spec.sites.keys())
        if got != set(self.EXPECTED_SITES):
            raise ValueError(
                f"ThorGrootN17AttnBackend expects sites {set(self.EXPECTED_SITES)}, "
                f"got {got}")

        self._slots = {
            "vit":           dict(vit_slots),
            "llm":           dict(llm_slots),
            "vl_self_attn":  dict(vl_self_attn_slots),
            "dit_self":      dict(dit_self_slots),
            "dit_cross":     dict(dit_cross_slots),
        }

        # Cache per-site cuBLAS handles (unwrapped from .cpp if wrapper).
        self._ctx_cpp: dict[str, object] = {}
        for name in ("llm", "vl_self_attn", "dit_self", "dit_cross"):
            c = self._slots[name].get("ctx")
            if c is None:
                raise ValueError(f"{name}_slots missing required key 'ctx'")
            self._ctx_cpp[name] = c.cpp if hasattr(c, "cpp") else c

        # ViT slots come in two flavours:
        #   * fused-QKV (interleaved) — one ``qkv`` ptr, fmha stride=3*D
        #   * separated — three ``Q``/``K``/``V`` ptrs, fmha stride=D
        #     (used by N1.7 ViT so 2D RoPE can apply on contiguous Q/K).
        self._require_keys("vit", ("O", "D"))
        vit_slots = self._slots["vit"]
        has_qkv = "qkv" in vit_slots and int(vit_slots["qkv"]) != 0
        has_split = all(
            k in vit_slots and int(vit_slots[k]) != 0 for k in ("Q", "K", "V"))
        if not (has_qkv ^ has_split):
            raise ValueError(
                "vit_slots needs exactly one of: 'qkv' (interleaved buffer) "
                "or all of 'Q'/'K'/'V' (separated buffers)")
        for name in ("llm", "vl_self_attn", "dit_self"):
            self._require_keys(name, ("Q", "K", "V", "O", "logits", "scale"))
        self._require_keys(
            "dit_cross", ("Q", "K_layers", "V_layers", "O", "logits", "scale")
        )

        # vit D must match spec
        vsite = spec.site("vit")
        exp_D = vsite.num_q_heads * vsite.head_dim
        if int(self._slots["vit"]["D"]) != exp_D:
            raise ValueError(
                f"vit_slots['D']={self._slots['vit']['D']} != num_q_heads*head_dim={exp_D}")

        # dit_cross K/V list lengths
        nL_cross = spec.site("dit_cross").num_layers
        kL = self._slots["dit_cross"]["K_layers"]
        vL = self._slots["dit_cross"]["V_layers"]
        if len(kL) != nL_cross or len(vL) != nL_cross:
            raise ValueError(
                f"dit_cross K_layers/V_layers length must be {nL_cross}, "
                f"got K={len(kL)} V={len(vL)}")
        for i, (kp, vp) in enumerate(zip(kL, vL)):
            if int(kp) == 0 or int(vp) == 0:
                raise ValueError(
                    f"dit_cross K_layers[{i}] or V_layers[{i}] is null")

        self._fvk = None

    def _require_keys(self, site: str, keys: tuple[str, ...]) -> None:
        slot = self._slots[site]
        for k in keys:
            if k not in slot:
                raise ValueError(f"{site}_slots missing required key {k!r}")
            if k in ("qkv", "O", "Q", "K", "V", "logits"):
                if int(slot[k]) == 0:
                    raise ValueError(
                        f"{site}_slots[{k!r}] is a null device pointer")

    def _fvk_mod(self):
        if self._fvk is None:
            import flash_rt.flash_rt_kernels as fvk
            self._fvk = fvk
        return self._fvk

    # ────────────────────────────────────────────────────────────────
    # Protocol: get_slot_ptrs
    # ────────────────────────────────────────────────────────────────
    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]:
        if site not in self._slots:
            raise KeyError(f"unknown site {site!r}")

        if site == "vit":
            s = self._slots[site]
            if "qkv" in s and int(s["qkv"]) != 0:
                D2 = int(s["D"]) * 2  # fp16 bytes
                base = int(s["qkv"])
                return {
                    "Q": base,
                    "K": base + D2,
                    "V": base + 2 * D2,
                    "O": int(s["O"]),
                }
            return {
                "Q": int(s["Q"]),
                "K": int(s["K"]),
                "V": int(s["V"]),
                "O": int(s["O"]),
            }

        nL = self._spec.site(site).num_layers
        if not (0 <= layer_idx < nL):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r} (num_layers={nL})")

        s = self._slots[site]
        if site == "dit_cross":
            return {
                "Q": int(s["Q"]),
                "K": int(s["K_layers"][layer_idx]),
                "V": int(s["V_layers"][layer_idx]),
                "O": int(s["O"]),
            }
        return {
            "Q": int(s["Q"]),
            "K": int(s["K"]),
            "V": int(s["V"]),
            "O": int(s["O"]),
        }

    # ────────────────────────────────────────────────────────────────
    # Protocol: run
    # ────────────────────────────────────────────────────────────────
    def run(self, site: str, layer_idx: int, q_seq: int,
            *, kv_seq: Optional[int] = None, stream: int = 0,
            state_nk: Optional[int] = None) -> int:
        """Dispatch fvk attention for (site, layer_idx).

        Kernel routing:
          * ``vit``        → ``fvk.fmha_strided_full`` (multi-view batched).
          * everything else → ``fvk.attention_mha_fp16``.
        """
        if site not in self._slots:
            raise KeyError(f"unknown site {site!r}")

        fvk = self._fvk_mod()
        site_spec = self._spec.site(site)

        if site == "vit":
            s = self._slots[site]
            D = int(s["D"])
            nv = site_spec.batch_axis
            NH = site_spec.num_q_heads
            HD = site_spec.head_dim
            if kv_seq is None:
                kv_seq = q_seq
            if "qkv" in s and int(s["qkv"]) != 0:
                # Fused-QKV interleaved: stride = 3*D elements/token
                stride = 3 * D
                Q = int(s["qkv"])
                K = Q + D * 2  # fp16 byte offset
                V = Q + 2 * D * 2
            else:
                # Separated Q/K/V (used by N1.7 ViT for RoPE-friendly layout)
                stride = D
                Q, K, V = int(s["Q"]), int(s["K"]), int(s["V"])
            fvk.fmha_strided_full(
                Q, K, V, int(s["O"]),
                nv, q_seq, kv_seq, NH, NH, HD,
                stride, stride, stream,
            )
            return int(s["O"])

        nL = site_spec.num_layers
        if not (0 <= layer_idx < nL):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r} (num_layers={nL})")

        kernel = site_spec.extra.get("kernel", "standard")
        if kernel != "mha":
            raise ValueError(
                f"site {site!r} has unexpected kernel {kernel!r}; "
                f"ThorGrootN17AttnBackend only supports 'mha' for non-vit sites")

        s = self._slots[site]
        if site == "dit_cross":
            K_ptr = int(s["K_layers"][layer_idx])
            V_ptr = int(s["V_layers"][layer_idx])
        else:
            K_ptr = int(s["K"])
            V_ptr = int(s["V"])

        if kv_seq is None:
            kv_seq = q_seq

        # ``attention_mha_*`` requires the logits buffer to be pre-filled
        # with -inf — the kernel writes the QK^T product but does not zero
        # uninitialized positions, so leftover bytes survive into softmax
        # and corrupt the result. (Empirically: cos drops from 1.0 to 0.51
        # without this pre-fill.) ``logits`` is a per-site scratch buffer
        # of shape ``(NH, max_q_seq, max_kv_seq)``; fill it in full.
        nh = site_spec.num_q_heads
        mq = site_spec.max_q_seq
        mkv = site_spec.max_kv_seq

        # Per-site dtype: dit_self / dit_cross run at bf16 (matches ckpt
        # native dtype); everything else stays fp16. Flag opt-in via
        # extra={"dtype": "bf16"} on the spec site.
        dtype = site_spec.extra.get("dtype", "fp16")
        is_causal = bool(site_spec.extra.get("causal", False))

        # Masked-softmax tier (opt-in): the masked MHA variants softmax only
        # the valid S_kv columns, so the full-buffer -inf pre-fill (a DRAM
        # sweep per layer) disappears. The causal fp16 kernel already masks
        # by position internally (including the padding columns), so its
        # pre-fill is skipped outright.
        masked = bool(getattr(self, "_use_masked_softmax", False))

        if dtype == "bf16":
            if is_causal:
                raise ValueError(
                    f"site {site!r}: bf16 attention has no causal variant yet")
            if masked:
                fvk.attention_mha_bf16_masked(
                    self._ctx_cpp[site],
                    int(s["Q"]), K_ptr, V_ptr,
                    int(s["logits"]), int(s["O"]),
                    q_seq, kv_seq,
                    site_spec.num_q_heads, site_spec.head_dim,
                    float(s["scale"]), int(mkv),
                    int(s.get("qkv_stride", 0)), stream,
                )
                return int(s["O"])
            fvk.gpu_fill_neginf_bf16(int(s["logits"]), nh * mq * mkv, stream)
            # logits buffer is allocated as (NH, max_q_seq, max_kv_seq)
            # row-major; pass the kv-axis stride so the GEMM head batching
            # matches the buffer layout (otherwise head N writes into the
            # tail of head N-1's slab → NaN cascade).
            fvk.attention_mha_bf16(
                self._ctx_cpp[site],
                int(s["Q"]), K_ptr, V_ptr,
                int(s["logits"]), int(s["O"]),
                q_seq, kv_seq,
                site_spec.num_q_heads, site_spec.head_dim,
                float(s["scale"]), int(mkv), stream,
            )
            return int(s["O"])

        if not masked:
            fvk.gpu_fill_neginf_fp16(int(s["logits"]), nh * mq * mkv, stream)
        if is_causal:
            fvk.attention_mha_causal_fp16(
                self._ctx_cpp[site],
                int(s["Q"]), K_ptr, V_ptr,
                int(s["logits"]), int(s["O"]),
                q_seq, kv_seq,
                site_spec.num_q_heads, site_spec.head_dim,
                float(s["scale"]), stream,
            )
        elif masked:
            fvk.attention_mha_fp16_masked(
                self._ctx_cpp[site],
                int(s["Q"]), K_ptr, V_ptr,
                int(s["logits"]), int(s["O"]),
                q_seq, kv_seq,
                site_spec.num_q_heads, site_spec.head_dim,
                float(s["scale"]), stream,
            )
        else:
            fvk.attention_mha_fp16(
                self._ctx_cpp[site],
                int(s["Q"]), K_ptr, V_ptr,
                int(s["logits"]), int(s["O"]),
                q_seq, kv_seq,
                site_spec.num_q_heads, site_spec.head_dim,
                float(s["scale"]), stream,
            )
        return int(s["O"])


# ════════════════════════════════════════════════════════════════════
#  Spec builder
# ════════════════════════════════════════════════════════════════════


def make_groot_n17_attention_spec(
    *,
    num_views: int,
    llm_seq_max: int,
    vl_self_attn_seq_max: int,
    sa: int,
    s_kv_text: int,
    s_kv_image: int,
) -> AttentionSpec:
    """Build the N1.7 AttentionSpec (5 sites).

    Args:
        num_views: number of camera views (1 or 2 typical).
        llm_seq_max: max LLM token sequence (text + (num_views * image_tokens)).
        vl_self_attn_seq_max: same as llm_seq_max usually — vlln output is
            consumed by the 4-layer self-attn over the same sequence.
        sa: DiT action-sequence length (1 state + action_horizon=40).
        s_kv_text: max text-kv length for DiT cross-attn-to-text blocks.
        s_kv_image: max image-kv length for DiT cross-attn-to-image blocks.
    """
    spec = AttentionSpec()
    # Qwen3-VL ViT — 24 layers, MHA 16x64, hidden 1024.
    # batch_axis=num_views is the count of camera views processed in one batch.
    spec.add_site(
        "vit",
        num_layers=24, num_q_heads=16, num_kv_heads=16, head_dim=64,
        max_q_seq=256, max_kv_seq=256, batch_axis=int(num_views),
    )
    # Qwen3-VL LLM — 16 layers (truncated). GQA 16Q/8KV × 128hd; pipeline
    # pre-expands K/V before the kernel call, so this advertises MHA shape.
    spec.add_site(
        "llm",
        num_layers=16, num_q_heads=16, num_kv_heads=16, head_dim=128,
        max_q_seq=int(llm_seq_max), max_kv_seq=int(llm_seq_max),
        extra={"kernel": "mha", "causal": True},
    )
    # vl_self_attention — 4 layers, MHA 32x64.
    spec.add_site(
        "vl_self_attn",
        num_layers=4, num_q_heads=32, num_kv_heads=32, head_dim=64,
        max_q_seq=int(vl_self_attn_seq_max), max_kv_seq=int(vl_self_attn_seq_max),
        extra={"kernel": "mha"},
    )
    # DiT self-attention — 16 layers, MHA 32x48 (1536 hidden), self-attn over sa.
    # ``attention_mha_*`` rounds S_kv up to a multiple of 8 internally for
    # the cuBLAS strided-batched GEMM. We advertise the rounded value as
    # max_kv_seq so the per-site logits buffer (sized off this spec) and
    # the gpu_fill_neginf_* call (looped over nh*mq*mkv) both cover the
    # full padded region.
    def _pad8(n: int) -> int:
        return ((int(n) + 7) // 8) * 8

    spec.add_site(
        "dit_self",
        num_layers=16, num_q_heads=32, num_kv_heads=32, head_dim=48,
        max_q_seq=int(sa), max_kv_seq=_pad8(sa),
        extra={"kernel": "mha", "dtype": "bf16"},
    )
    # DiT cross-attention — 16 layers; KV is per-layer (precomputed for both
    # text and image targets at set_prompt time). max_kv_seq is the larger
    # of the two; the pipeline passes the actual ``kv_seq`` per call.
    spec.add_site(
        "dit_cross",
        num_layers=16, num_q_heads=32, num_kv_heads=32, head_dim=48,
        max_q_seq=int(sa),
        max_kv_seq=_pad8(max(int(s_kv_text), int(s_kv_image))),
        extra={"kernel": "mha", "dtype": "bf16"},
    )
    return spec
