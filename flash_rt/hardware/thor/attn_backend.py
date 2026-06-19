"""FlashRT — Thor attention backend.

Implements the AttentionBackend protocol (``flash_rt/hardware/backend.py``)
on Thor SM110 using the fvk attention primitives. Pipeline-owned memory
model: the pipeline allocates all Q/K/V/O/logits buffers (and the layered
KV cache in weights) and passes pointers at construction. The backend is a
thin dispatch layer — it looks up (site, layer_idx) → fvk call + pointer
arguments and returns the output pointer.

Currently supports Pi0.5's three sites:

  * ``siglip``   — multi-view vision FMHA via ``fmha_strided_full``.
                   QKV is interleaved in a single buffer (stride=3*D);
                   backend derives Q/K/V ptrs via pointer arithmetic.
  * ``encoder``  — PaliGemma 18-layer self-attention via
                   ``attention_qkv_fp16``. Q/O alias the same buffer
                   (``attn_out`` from the pipeline).
  * ``decoder``  — cross-attention into the encoder KV cache via the
                   same ``attention_qkv_fp16`` primitive. Reuses the
                   encoder's Kc/Vc pointers (shared cache).

Pi0 / Pi0-FAST / GROOT are out of scope for Stage 1 — Pi0-FAST is
explicitly excluded by ``docs/stable_api.md``; Pi0 and GROOT will be
addressed in later stages.
"""

from __future__ import annotations

import math
from typing import Optional

from flash_rt.hardware.backend import AttentionBackendBase, AttentionSpec


class ThorFlashAttnBackend(AttentionBackendBase):
    """Pi0.5 attention backend on Thor (SM110).

    Constructed by the frontend after it has allocated all pipeline
    buffers and loaded weights; injected into ``siglip_forward`` /
    ``encoder_forward`` / ``decoder_forward`` via an optional ``attn=``
    kwarg. Legacy fallback path (attn=None) remains available during the
    staged rollout so frontends can opt in one at a time.
    """

    # ────────────────────────────────────────────────────────────────
    # Construction
    # ────────────────────────────────────────────────────────────────
    def __init__(self, spec: AttentionSpec, ctx, *,
                 siglip_slots: dict, encoder_slots: dict,
                 decoder_slots: dict,
                 fixed_control: Optional[dict] = None) -> None:
        """
        Args:
            spec: AttentionSpec with exactly the sites
                {"siglip", "encoder", "decoder"}.
            ctx:  the object passed as the first argument to
                ``fvk.attention_qkv_fp16``. Accepted forms:
                  * raw ``fvk.FvkContext`` (Pi0.5 frontends)
                  * ``flash_rt.core.context.FvkContext`` Python wrapper
                    (newer call sites)
                Backend passes it through unchanged; it auto-detects the
                wrapper form via a ``.cpp`` attribute at call time.
            siglip_slots: dict with keys:
                qkv (int)  — interleaved QKV buffer ptr, stride = 3*D
                O   (int)  — attn_out buffer ptr, [num_views*spv, D] fp16
                D   (int)  — hidden dim; used for pointer arithmetic
            encoder_slots / decoder_slots: dict with keys:
                Q_O          (int) — attn_out; Q input AND O output
                Kc, Vc       (int) — layered KV cache base pointers
                logits       (int) — scratch for P = QK^T
                layer_stride (int) — bytes between successive layers in Kc/Vc
                scale        (float) — 1/sqrt(head_dim)
            fixed_control: optional dict for fixed-shape state-prompt mode.
                When omitted and torch CUDA is available, the backend
                allocates the three int32 device scalars itself. JAX callers
                pass CudaBuffer-backed pointers plus a setter callback to keep
                this module independent of PyTorch.

        Invariants enforced:
            * spec has the three expected sites with correct layer counts.
            * All pointer slots are non-zero.
            * ``siglip_slots['D']`` matches spec's siglip num_q_heads *
              head_dim.
        """
        super().__init__(spec)

        expected_sites = {"siglip", "encoder", "decoder"}
        got = set(spec.sites.keys())
        if got != expected_sites:
            raise ValueError(
                f"ThorFlashAttnBackend expects sites {expected_sites}, "
                f"got {got}")

        # Unwrap Python FvkContext wrapper if present; otherwise pass raw
        # fvk.FvkContext through. Both Pi0.5 (raw) and Pi0/Groot (wrapper)
        # call sites work without further conditioning in run().
        self._ctx_cpp = ctx.cpp if hasattr(ctx, "cpp") else ctx
        self._slots = {
            "siglip":  dict(siglip_slots),
            "encoder": dict(encoder_slots),
            "decoder": dict(decoder_slots),
        }

        # Validate required keys + non-zero ptrs.
        self._require_keys("siglip",  ("qkv", "O", "D"))
        self._require_keys("encoder",
                           ("Q_O", "Kc", "Vc", "logits",
                            "layer_stride", "scale"))
        self._require_keys("decoder",
                           ("Q_O", "Kc", "Vc", "logits",
                            "layer_stride", "scale"))

        sig = spec.site("siglip")
        exp_D = sig.num_q_heads * sig.head_dim
        if int(self._slots["siglip"]["D"]) != exp_D:
            raise ValueError(
                f"siglip_slots['D']={self._slots['siglip']['D']} does not "
                f"match spec (num_q_heads*head_dim = {exp_D})")

        # Precompute per-layer KV pointers for encoder/decoder.
        # Both sites share the same Kc/Vc base and layer_stride
        # (decoder reuses encoder's cache by design).
        self._per_layer_kv: dict[str, list[tuple[int, int]]] = {}
        for site_name in ("encoder", "decoder"):
            s = self._slots[site_name]
            nL = spec.site(site_name).num_layers
            stride = int(s["layer_stride"])
            Kc = int(s["Kc"])
            Vc = int(s["Vc"])
            self._per_layer_kv[site_name] = [
                (Kc + l * stride, Vc + l * stride) for l in range(nL)
            ]

        # Lazy-import fvk — importing at class definition time would
        # couple this module to module-load order and break tests that
        # construct the backend without a fully-initialised fvk env.
        self._fvk = None

        self._fixed_shape = False
        self._fixed_set_valid_len = None
        self.enc_seqused = None
        self.dec_seqused = None
        self.dec_devpos = None
        if fixed_control is not None:
            self.enc_seqused = fixed_control.get("enc_seqused")
            self.dec_seqused = fixed_control.get("dec_seqused")
            self.dec_devpos = fixed_control.get("dec_devpos")
            self._fixed_set_valid_len = fixed_control.get("set_valid_len")
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    self.enc_seqused = torch.zeros(
                        1, dtype=torch.int32, device="cuda")
                    self.dec_seqused = torch.zeros(
                        1, dtype=torch.int32, device="cuda")
                    self.dec_devpos = torch.zeros(
                        1, dtype=torch.int32, device="cuda")
            except Exception:
                pass
        self._enc_seqused_ptr = self._device_ptr(self.enc_seqused)
        self._dec_seqused_ptr = self._device_ptr(self.dec_seqused)
        self._dec_devpos_ptr = self._device_ptr(self.dec_devpos)

    @staticmethod
    def _device_ptr(obj) -> int:
        if obj is None:
            return 0
        if isinstance(obj, int):
            return int(obj)
        if hasattr(obj, "data_ptr"):
            return int(obj.data_ptr())
        if hasattr(obj, "ptr"):
            return int(obj.ptr.value)
        return int(obj)

    def _require_keys(self, site: str, keys: tuple[str, ...]) -> None:
        slot = self._slots[site]
        for k in keys:
            if k not in slot:
                raise ValueError(f"{site}_slots missing required key {k!r}")
            # Pointer-typed keys must be non-zero; numeric-typed keys
            # (D, layer_stride, scale) can be non-zero but may be floats.
            if k in ("qkv", "O", "Q_O", "Kc", "Vc", "logits"):
                if int(slot[k]) == 0:
                    raise ValueError(
                        f"{site}_slots[{k!r}] is a null device pointer")

    def _fvk_mod(self):
        if self._fvk is None:
            import flash_rt.flash_rt_kernels as fvk
            self._fvk = fvk
        return self._fvk

    def set_fixed_shape(self, enabled: bool) -> None:
        """Enable/disable fixed-shape state-prompt masking.

        When enabled, encoder and decoder standard attention use
        ``attention_qkv_fp16_seqused``. The captured K/V max shape stays fixed;
        :meth:`set_fixed_valid_len` updates the device-side valid lengths.
        """
        if enabled and not (
            self._enc_seqused_ptr and self._dec_seqused_ptr
            and self._dec_devpos_ptr
        ):
            raise RuntimeError(
                "Thor fixed-shape attention requires enc_seqused, "
                "dec_seqused, and dec_devpos device scalars")
        self._fixed_shape = bool(enabled)

    def set_fixed_valid_len(self, valid_prefix_len: int) -> None:
        """Update device-side valid prefix length for fixed-shape replay."""
        v = int(valid_prefix_len)
        enc_max = self._spec.site("encoder").max_kv_seq
        dec_max = self._spec.site("decoder").max_kv_seq
        chunk = self._spec.site("decoder").max_q_seq
        if not (0 < v <= enc_max):
            raise ValueError(
                f"valid_prefix_len={v} out of range for encoder max {enc_max}")
        if v + chunk > dec_max:
            raise ValueError(
                f"decoder valid length {v + chunk} exceeds max {dec_max}")
        if self._fixed_set_valid_len is not None:
            self._fixed_set_valid_len(v, v + chunk, v)
            return
        if not (hasattr(self.enc_seqused, "fill_")
                and hasattr(self.dec_seqused, "fill_")
                and hasattr(self.dec_devpos, "fill_")):
            raise RuntimeError(
                "Thor fixed-shape scalars are pointer-only; caller must "
                "provide fixed_control['set_valid_len']")
        self.enc_seqused.fill_(v)
        self.dec_seqused.fill_(v + chunk)
        self.dec_devpos.fill_(v)
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────────
    # Protocol: get_slot_ptrs
    # ────────────────────────────────────────────────────────────────
    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]:
        """Return {Q, K, V, O} device pointer ints for (site, layer_idx).

        * siglip: Q/K/V derived from the single interleaved ``qkv``
          buffer by pointer arithmetic (byte offsets: 0, D*2, 2*D*2).
          ``layer_idx`` is ignored (all 27 layers share one scratch
          buffer — pipeline reuses it across layers).
        * encoder / decoder: Q_O is the shared attn_out buffer (Q
          input + O output alias); K/V come from the pre-computed
          per-layer KV cache offsets.
        """
        if site not in self._slots:
            raise KeyError(f"unknown site {site!r}")

        if site == "siglip":
            s = self._slots[site]
            D2 = int(s["D"]) * 2  # bytes (fp16 = 2 bytes per elem)
            base = int(s["qkv"])
            return {
                "Q": base,
                "K": base + D2,
                "V": base + 2 * D2,
                "O": int(s["O"]),
            }

        # encoder / decoder
        nL = self._spec.site(site).num_layers
        if not (0 <= layer_idx < nL):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r} "
                f"(num_layers={nL})")
        K_ptr, V_ptr = self._per_layer_kv[site][layer_idx]
        q_o = int(self._slots[site]["Q_O"])
        return {"Q": q_o, "K": K_ptr, "V": V_ptr, "O": q_o}

    # ────────────────────────────────────────────────────────────────
    # Protocol: run
    # ────────────────────────────────────────────────────────────────
    def run(self, site: str, layer_idx: int, q_seq: int,
            *, kv_seq: Optional[int] = None, stream: int = 0,
            state_nk: Optional[int] = None) -> int:
        """Dispatch the fvk attention kernel for (site, layer_idx).

        Returns the output device pointer int. For Thor the output
        always aliases a pipeline-owned buffer (siglip.O for vision,
        encoder/decoder Q_O for the language/decoder paths).

        Kernel selection (encoder/decoder sites) is driven by
        ``SiteSpec.extra["kernel"]``:
          * absent / ``"standard"`` → ``fvk.attention_qkv_fp16``
          * ``"state_masked"``      → ``fvk.attention_qkv_fp16_state_masked``
                                      (Pi0 decoder; requires ``state_nk``).
        """
        if site not in self._slots:
            raise KeyError(f"unknown site {site!r}")

        fvk = self._fvk_mod()
        site_spec = self._spec.site(site)

        if site == "siglip":
            s = self._slots[site]
            D = int(s["D"])
            nv = site_spec.batch_axis
            NH = site_spec.num_q_heads
            HD = site_spec.head_dim
            if kv_seq is None:
                kv_seq = q_seq  # self-attention
            stride = 3 * D
            Q = int(s["qkv"])
            K = Q + D * 2
            V = Q + 2 * D * 2
            fvk.fmha_strided_full(Q, K, V, int(s["O"]),
                                   nv, q_seq, kv_seq, NH, NH, HD,
                                   stride, stride, stream)
            return int(s["O"])

        # encoder / decoder — kernel chosen by site.extra
        nL = site_spec.num_layers
        if not (0 <= layer_idx < nL):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r} "
                f"(num_layers={nL})")
        s = self._slots[site]
        K_ptr, V_ptr = self._per_layer_kv[site][layer_idx]
        if kv_seq is None:
            kv_seq = q_seq  # default self-attention; decoder caller
                             # always supplies kv_seq explicitly.

        kernel = site_spec.extra.get("kernel", "standard")
        if kernel == "state_masked":
            if state_nk is None:
                raise ValueError(
                    f"site {site!r} uses state_masked kernel but no "
                    f"state_nk was provided to run()")
            if not (0 < int(state_nk) <= kv_seq):
                raise ValueError(
                    f"state_nk={state_nk} out of range (kv_seq={kv_seq})")
            fvk.attention_qkv_fp16_state_masked(
                self._ctx_cpp,
                int(s["Q_O"]), K_ptr, V_ptr,
                int(s["logits"]), int(s["Q_O"]),
                q_seq, kv_seq,
                site_spec.num_q_heads, site_spec.head_dim,
                int(state_nk),
                float(s["scale"]), stream,
            )
        elif kernel == "standard":
            if self._fixed_shape and site in ("encoder", "decoder"):
                seqused = (self._enc_seqused_ptr if site == "encoder"
                           else self._dec_seqused_ptr)
                fvk.attention_qkv_fp16_seqused(
                    self._ctx_cpp,
                    int(s["Q_O"]), K_ptr, V_ptr,
                    int(s["logits"]), int(s["Q_O"]),
                    q_seq, kv_seq,
                    site_spec.num_q_heads, site_spec.head_dim,
                    seqused,
                    float(s["scale"]), stream,
                )
            else:
                fvk.attention_qkv_fp16(
                    self._ctx_cpp,
                    int(s["Q_O"]), K_ptr, V_ptr,
                    int(s["logits"]), int(s["Q_O"]),
                    q_seq, kv_seq,
                    site_spec.num_q_heads, site_spec.head_dim,
                    float(s["scale"]), stream,
                )
        else:
            raise ValueError(
                f"unknown kernel {kernel!r} for site {site!r} "
                f"(supported: 'standard', 'state_masked')")
        return int(s["Q_O"])


# ════════════════════════════════════════════════════════════════════
# Spec builder for Pi0.5
# ════════════════════════════════════════════════════════════════════

def make_pi05_attention_spec(*, num_views: int, enc_seq_max: int,
                              chunk_size: int = 10) -> AttentionSpec:
    """Build the Pi0.5 AttentionSpec (3 sites: siglip/encoder/decoder).

    Args:
        num_views: number of camera views used by SigLIP (1, 2, or 3).
                   Stored as ``batch_axis`` on the siglip site.
        enc_seq_max: maximum encoder sequence length (prompt_len + vision
                     tokens). Depends on tokenizer max_len and view count.
        chunk_size: action-chunk length used by the decoder
                    (== decoder Q length == number of action tokens).

    The per-site dimensions are Pi0.5-specific (PaliGemma 2B + SigLIP-L):

        siglip  : 27 layers, 16 heads × 72 head_dim,  256 tokens/view
        encoder : 18 layers, 8 Q heads, 1 KV head, 256 head_dim (GQA 8)
        decoder : 18 layers, same GQA config, cross-attends encoder KV
                  cache extended by ``chunk_size`` action tokens.
    """
    spec = AttentionSpec()
    spec.add_site(
        "siglip",
        num_layers=27, num_q_heads=16, num_kv_heads=16, head_dim=72,
        max_q_seq=256, max_kv_seq=256, batch_axis=int(num_views),
    )
    spec.add_site(
        "encoder",
        num_layers=18, num_q_heads=8, num_kv_heads=1, head_dim=256,
        max_q_seq=int(enc_seq_max), max_kv_seq=int(enc_seq_max),
    )
    spec.add_site(
        "decoder",
        num_layers=18, num_q_heads=8, num_kv_heads=1, head_dim=256,
        max_q_seq=int(chunk_size),
        max_kv_seq=int(enc_seq_max) + int(chunk_size),
    )
    return spec


def make_pi0_attention_spec(*, num_views: int, enc_seq_max: int,
                             S_dec: int) -> AttentionSpec:
    """Build the Pi0 AttentionSpec (3 sites: siglip/encoder/decoder).

    Shares siglip + encoder site shapes with Pi0.5. The decoder site
    differs:
        * ``max_q_seq = S_dec = chunk_size + 1`` — Pi0 prepends a
          state token to the action chunk before running the decoder.
        * ``extra = {"kernel": "state_masked"}`` — Pi0 decoder uses
          ``fvk.attention_qkv_fp16_state_masked``; the state token
          (row 0) is forbidden from attending to action K/V
          (columns ``[enc_seq_max + 1:]``).

    Args:
        num_views: number of camera views used by SigLIP (1, 2, or 3).
        enc_seq_max: maximum encoder sequence length.
        S_dec: decoder Q length (``chunk_size + 1``; includes state token).
    """
    spec = AttentionSpec()
    spec.add_site(
        "siglip",
        num_layers=27, num_q_heads=16, num_kv_heads=16, head_dim=72,
        max_q_seq=256, max_kv_seq=256, batch_axis=int(num_views),
    )
    spec.add_site(
        "encoder",
        num_layers=18, num_q_heads=8, num_kv_heads=1, head_dim=256,
        max_q_seq=int(enc_seq_max), max_kv_seq=int(enc_seq_max),
    )
    spec.add_site(
        "decoder",
        num_layers=18, num_q_heads=8, num_kv_heads=1, head_dim=256,
        max_q_seq=int(S_dec),
        max_kv_seq=int(enc_seq_max) + int(S_dec),
        extra={"kernel": "state_masked"},
    )
    return spec
