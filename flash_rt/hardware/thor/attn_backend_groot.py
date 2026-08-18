"""FlashRT — Thor attention backend for GROOT.

GROOT's internal buffer layout is sufficiently different from Pi0.5/Pi0
that it warrants its own backend class (see
``docs/v2/stage4_groot_backend.md`` §4 for the rationale):

  * No Q/O aliasing — Q and O are separate buffers at every site.
  * No per-layer KV cache for qwen3/dit_self (one scratch set shared
    across all layers); dit_cross does have per-layer K/V but those
    are *precomputed* projections (stable torch tensors), not a
    cache that is written every step.
  * QKV are in 3 independent buffers, not interleaved.
  * SigLIP2 uses the same layout as Pi0.5's SigLIP (interleaved QKV in
    a single qkv buffer) and reuses the same ``fmha_strided_full`` kernel.

Sites:
  * ``siglip``     — SigLIP2 vision (27 layers, MHA 16×72).
  * ``qwen3``      — Qwen3 backbone (16 layers, MHA-post-expansion 16×128,
                     even though the model is GQA 16Q/8KV the pipeline
                     repeat_interleave's K/V before invoking attention).
  * ``dit_self``   — DiT self-attention (16 layers at odd positions,
                     MHA 32×48).
  * ``dit_cross``  — DiT cross-attention (16 layers at even positions,
                     MHA 32×48). K/V ptrs are per-layer — they come
                     from ``CKernelDiTHead._precomp_k/_precomp_v`` which
                     are allocated once and written in-place by
                     ``precompute_cross_kv``.

Pi0.5/Pi0's ``ThorFlashAttnBackend`` is unaffected by this module.
"""

from __future__ import annotations

from typing import Optional

from flash_rt.hardware.backend import AttentionBackendBase, AttentionSpec


# ════════════════════════════════════════════════════════════════════
#  Backend
# ════════════════════════════════════════════════════════════════════


class ThorGrootAttnBackend(AttentionBackendBase):
    """GROOT attention backend on Thor (SM110).

    Constructed by the frontend after CKernelQwen3/CKernelDiTHead have
    finished allocating their pipeline-owned buffers. The two class
    instances then bind ``self.attn = <this_backend>`` so their
    ``forward`` / ``_run_step`` dispatch through the protocol.
    """

    def __init__(self, spec: AttentionSpec, *,
                 siglip_slots: dict, qwen3_slots: dict,
                 dit_self_slots: dict, dit_cross_slots: dict) -> None:
        """
        Args:
            spec: AttentionSpec with exactly the sites
                {"siglip", "qwen3", "dit_self", "dit_cross"}.
            siglip_slots: {
                qkv (int)  — interleaved QKV buffer (same layout as Pi0.5)
                O   (int)
                D   (int)  — hidden dim for pointer arithmetic
            }
                (no ctx — fmha_strided_full does not take an FvkContext.)
            qwen3_slots / dit_self_slots: {
                ctx           — fvk.FvkContext (raw) or wrapper with .cpp;
                                passed to fvk.attention_mha_fp16. Separate
                                per site because CKernelQwen3 and
                                CKernelDiTHead own distinct cuBLAS handles
                                — reusing one for both would change
                                cuBLASLt workspace state and therefore
                                tactic selection (±2 ms Myelin drift).
                Q (int)       — Q buffer (shared across all layers)
                K (int)       — K buffer
                V (int)       — V buffer
                O (int)       — O buffer (distinct from Q, no aliasing)
                logits (int)  — P = QK^T scratch
                scale (float) — 1/sqrt(head_dim)
            }
            dit_cross_slots: same as dit_self_slots, replacing K/V with:
                K_layers (list[int])  — per-layer K ptrs (len = num_layers)
                V_layers (list[int])  — per-layer V ptrs
        """
        super().__init__(spec)

        expected_sites = {"siglip", "qwen3", "dit_self", "dit_cross"}
        got = set(spec.sites.keys())
        if got != expected_sites:
            raise ValueError(
                f"ThorGrootAttnBackend expects sites {expected_sites}, got {got}")

        self._slots = {
            "siglip":    dict(siglip_slots),
            "qwen3":     dict(qwen3_slots),
            "dit_self":  dict(dit_self_slots),
            "dit_cross": dict(dit_cross_slots),
        }

        # Cache per-site cuBLAS handles (unwrapped from .cpp if wrapper).
        self._ctx_cpp = {}
        for name in ("qwen3", "dit_self", "dit_cross"):
            c = self._slots[name].get("ctx")
            if c is None:
                raise ValueError(f"{name}_slots missing required key 'ctx'")
            self._ctx_cpp[name] = c.cpp if hasattr(c, "cpp") else c

        # Validate required keys per site.
        self._require_keys("siglip", ("qkv", "O", "D"))
        for name in ("qwen3", "dit_self"):
            self._require_keys(name, ("Q", "K", "V", "O", "logits", "scale"))
        self._require_keys("dit_cross",
                            ("Q", "K_layers", "V_layers", "O", "logits", "scale"))

        # siglip D matches spec
        sig = spec.site("siglip")
        exp_D = sig.num_q_heads * sig.head_dim
        if int(self._slots["siglip"]["D"]) != exp_D:
            raise ValueError(
                f"siglip_slots['D']={self._slots['siglip']['D']} does not "
                f"match spec (num_q_heads*head_dim = {exp_D})")

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

        self._fvk = None  # lazy-import

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

        if site == "siglip":
            s = self._slots[site]
            D2 = int(s["D"]) * 2  # bytes (fp16)
            base = int(s["qkv"])
            return {"Q": base, "K": base + D2, "V": base + 2 * D2,
                    "O": int(s["O"])}

        nL = self._spec.site(site).num_layers
        if not (0 <= layer_idx < nL):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r} "
                f"(num_layers={nL})")

        s = self._slots[site]
        if site == "dit_cross":
            return {
                "Q": int(s["Q"]),
                "K": int(s["K_layers"][layer_idx]),
                "V": int(s["V_layers"][layer_idx]),
                "O": int(s["O"]),
            }
        # qwen3 / dit_self — single K/V shared across layers
        return {"Q": int(s["Q"]), "K": int(s["K"]), "V": int(s["V"]),
                "O": int(s["O"])}

    # ────────────────────────────────────────────────────────────────
    # Protocol: run
    # ────────────────────────────────────────────────────────────────
    def run(self, site: str, layer_idx: int, q_seq: int,
            *, kv_seq: Optional[int] = None, stream: int = 0,
            state_nk: Optional[int] = None) -> int:
        """Dispatch the fvk attention kernel for (site, layer_idx).

        Kernel selection by ``site_spec.extra["kernel"]``:
          * ``"mha"`` / absent-for-siglip → fvk.attention_mha_fp16
                                             (or fvk.fmha_strided_full for siglip)
          * other values raise — this backend does not carry Pi0.5/Pi0 kernels.
        """
        if site not in self._slots:
            raise KeyError(f"unknown site {site!r}")

        fvk = self._fvk_mod()
        site_spec = self._spec.site(site)

        # ── SigLIP (same as Pi0.5) ──
        if site == "siglip":
            s = self._slots[site]
            D = int(s["D"])
            nv = site_spec.batch_axis
            NH = site_spec.num_q_heads
            HD = site_spec.head_dim
            if kv_seq is None:
                kv_seq = q_seq
            # The strided FMHA kernel is only validated for power-of-two
            # sequences on real data (648-seq cross-view SigLIP at the 252
            # training resolution diverges silently). Use torch sdpa when
            # tensor views are provided (graph-capturable, exact).
            if "qkv_t" in s:
                import torch.nn.functional as F
                qkv_t = s["qkv_t"]
                O_t = s["O_t"]
                seq = int(q_seq)
                q = qkv_t[:seq, :D].view(seq, NH, HD).transpose(0, 1)
                k = qkv_t[:seq, D:2 * D].view(seq, NH, HD).transpose(0, 1)
                v = qkv_t[:seq, 2 * D:].view(seq, NH, HD).transpose(0, 1)
                o = F.scaled_dot_product_attention(q, k, v)
                O_t[:seq].copy_(o.transpose(0, 1).reshape(seq, D))
                return int(s["O"])
            stride = 3 * D
            Q = int(s["qkv"])
            K = Q + D * 2
            V = Q + 2 * D * 2
            fvk.fmha_strided_full(Q, K, V, int(s["O"]),
                                   nv, q_seq, kv_seq, NH, NH, HD,
                                   stride, stride, stream)
            return int(s["O"])

        # ── qwen3 / dit_self / dit_cross — attention_mha_fp16 ──
        nL = site_spec.num_layers
        if not (0 <= layer_idx < nL):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r} "
                f"(num_layers={nL})")

        kernel = site_spec.extra.get("kernel", "standard")
        if kernel != "mha":
            raise ValueError(
                f"site {site!r} has unexpected kernel {kernel!r}; "
                f"ThorGrootAttnBackend only supports 'mha' for non-siglip sites")

        s = self._slots[site]
        if site == "dit_cross":
            K_ptr = int(s["K_layers"][layer_idx])
            V_ptr = int(s["V_layers"][layer_idx])
        else:
            K_ptr = int(s["K"])
            V_ptr = int(s["V"])

        if kv_seq is None:
            kv_seq = q_seq  # self-attn default

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


def make_groot_attention_spec(*, num_views: int, qwen3_seq_max: int,
                               sa: int, s_kv: int,
                               siglip_cross_view: bool = False) -> AttentionSpec:
    """Build the GROOT AttentionSpec (4 sites).

    Args:
        num_views: SigLIP2 camera views (stored as batch_axis on siglip).
        qwen3_seq_max: max Qwen3 sequence length (prompt + vision tokens).
        sa: DiT action sequence length (hidden tokens = 1 state + T actions).
        s_kv: DiT cross-attention KV length (non_img + img backbone features).
        siglip_cross_view: N1.6 HF semantics — the live HF baseline runs
            SigLIP2 under sdpa, which ignores the NaFlex per-view
            ``seq_len_list`` segmentation, so all packed views attend
            jointly. Model the SigLIP site as one batch-1 sequence of
            ``num_views * 256`` tokens instead of ``num_views``
            independent 256-token views. Default False keeps the legacy
            per-view behaviour.
    """
    spec = AttentionSpec()
    if siglip_cross_view:
        spec.add_site(
            "siglip",
            num_layers=27, num_q_heads=16, num_kv_heads=16, head_dim=72,
            max_q_seq=256 * int(num_views), max_kv_seq=256 * int(num_views),
            batch_axis=1,
        )
    else:
        spec.add_site(
            "siglip",
            num_layers=27, num_q_heads=16, num_kv_heads=16, head_dim=72,
            max_q_seq=256, max_kv_seq=256, batch_axis=int(num_views),
        )
    spec.add_site(
        "qwen3",
        num_layers=16, num_q_heads=16, num_kv_heads=16, head_dim=128,
        max_q_seq=int(qwen3_seq_max), max_kv_seq=int(qwen3_seq_max),
        extra={"kernel": "mha"},
    )
    spec.add_site(
        "dit_self",
        num_layers=16, num_q_heads=32, num_kv_heads=32, head_dim=48,
        max_q_seq=int(sa), max_kv_seq=int(sa),
        extra={"kernel": "mha"},
    )
    spec.add_site(
        "dit_cross",
        num_layers=16, num_q_heads=32, num_kv_heads=32, head_dim=48,
        max_q_seq=int(sa), max_kv_seq=int(s_kv),
        extra={"kernel": "mha"},
    )
    return spec
