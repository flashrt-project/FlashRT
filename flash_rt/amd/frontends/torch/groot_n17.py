"""FlashRT AMD -- GROOT N1.7 FP8 torch frontend for CDNA4 (MI350X, gfx950).

FP8 kernel backbone with an unquantized bf16 action head. The whole
VLM backbone (ViT / DeepStack / LLM / VL self-attn) runs through the AMD
FP8 kernel surface via :mod:`flash_rt.amd.models.groot_n17.pipeline`
(FUSED-EPILOGUE tier by default — bias / bias+GELU in the hipBLASLt FP8
epilogue with host alphas; ``FVK_AMD_FUSED_EPILOGUE=0`` falls back to the
decomposed descale form), and the DiT action head stays bf16
(``_DIT_USE_FP8 = False``). The same env picks the DiT driver: fused on →
the AMD-local ``dit_forward`` (``bf16_nn_bias`` / ``bf16_nn_bias_gelu``);
off → the hardware-independent
:func:`flash_rt.models.groot_n17.pipeline_thor.dit_forward` with the AMD
kernel module passed in — every binding name the bf16 DiT path touches
(``bf16_nn`` / ``add_bias_bf16`` / ``ada_layer_norm_bf16`` /
``layer_norm_no_affine_bf16`` / ``gelu_inplace`` / ``residual_add`` /
``concat2_bf16`` / ``relu_inplace_bf16`` / ``silu_inplace_fp16`` /
``cast_*`` / ``gpu_copy``) exists in ``flash_rt_amd_kernels``.

Reuse strategy: subclass ``_GrootN17FP8BackboneMixin`` +
``GrootN17TorchFrontendThor``. The Thor base resolves its kernel module
and GEMM runners lazily behind ``hasattr(self, "_fvk"/"_gemm")`` guards,
so this class seeds them with the AMD module at construction and the
~1900 validated base lines (weight loading via WEIGHT_SPEC, aux
contract, calibration bake/save/cache, diffusion modulator precompute,
graph replay ``infer``) run unmodified. torch on ROCm keeps the "cuda"
device string, ``torch.cuda.Stream`` and ``torch.cuda.CUDAGraph``
(hipGraph underneath), so the base's graph machinery is used as-is.

AMD-specific overrides:

  * ``_run_kernel_backbone_fp8`` — AMD pipeline stages + the CDNA4
    attention backend. The llm stage's Q/K/V GEMMs write straight into
    the backend slots (Q 16 heads, K/V the NATIVE 8 KV heads — aiter
    handles GQA internally), so the RTX K/V staging buffers and the
    ``gpu_repeat_interleave_heads`` expand step are dropped.
  * ``_build_dit_attn`` — one :class:`Cdna4GrootN17AttnBackend` serves
    all five sites; the backbone-time instance is reused for the DiT
    when the action-token count matches.
  * ``_setup_cross_kv_kernel`` / ``_cross_kv_fwd`` — the per-frame
    cross-KV refresh writes directly into the backend's per-block K/V
    slots (exact-size prefix views of the padded slots). The gather of
    text/image backbone rows uses ``torch.index_select`` with a
    preallocated ``out=`` (the AMD kernel module has no
    ``embedding_lookup_bf16`` binding); with fixed shapes and no
    allocation it is graph-capture-safe.
  * ``_capture_kernel_dit_graphs`` — same fully-kernelized single-graph
    chain as Thor with two deltas: the Euler update runs as an in-place
    ``torch.add_`` on the persistent action buffer (no
    ``euler_step_bf16_out`` binding on AMD), and — per the aiter
    capture-safety contract — the warmup iterations run ON the capture
    stream (inside its ``torch.cuda.stream`` context) so aiter's
    workspace allocations reach caching-allocator steady state before
    stream capture begins and every torch-side call lands on the same
    stream the raw-int kernels use.

The FP8 / FP4 DiT quantization tiers of the base are NOT ported;
``_capture_kernel_dit_graphs`` rejects them explicitly.
"""

from __future__ import annotations

import os

import torch

from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor
from flash_rt.frontends.torch.groot_n17_rtx_fp8 import _GrootN17FP8BackboneMixin

_FP16 = torch.float16
_U8 = torch.uint8


def _fused_epilogue_enabled() -> bool:
    """FVK_AMD_FUSED_EPILOGUE gate (default ON) for BOTH fusions.

    When on: the FP8 backbone forwards run with ``fused_epilogue=True``
    (bias / bias+GELU in the hipBLASLt FP8 epilogue, host alphas) and the
    DiT runs the AMD-local fused ``dit_forward`` (``bf16_nn_bias`` /
    ``bf16_nn_bias_gelu``). One env flips both for the A/B against the
    decomposed form. Read at setup time (set_prompt / graph build), never
    on the hot path — flipping the env after graphs are built has no
    effect on the built graphs.
    """
    return os.environ.get("FVK_AMD_FUSED_EPILOGUE", "1").strip().lower() \
        not in ("0", "off", "false", "no")


class GrootN17TorchFrontendAmd(_GrootN17FP8BackboneMixin,
                               GrootN17TorchFrontendThor):
    """N1.7 CDNA4 frontend: FP8 kernel backbone + bf16 DiT action head."""

    # The DiT runs unquantized bf16 on AMD. NOTE this is a weaker tier
    # than the Thor/RTX FP8 frontends, which inherit _DIT_USE_FP8 = True
    # and quantize the DiT FFN and fused self-attention QKV as well; that
    # calibration path is not ported to CDNA4 yet.
    _DIT_USE_FP8 = False

    # Default DiT token count (1 state + 40 action tokens) used to size
    # the shared attention backend at backbone-build time; a differing
    # action_horizon at infer time rebuilds a matching backend.
    _DEFAULT_SA = 41

    def __init__(
        self,
        checkpoint_path: str,
        *,
        num_views: int = 2,
        embodiment_tag: str = "oxe_droid_relative_eef_relative_joint",
        device: str = "cuda:0",
    ):
        # gfx950-only gate, FIRST: the MFMA tile shapes and FP8 paths in
        # the extension are CDNA4-specific — refuse before touching the
        # checkpoint unless BOTH the visible device arch (e.g.
        # "gfx950:sramecc+:xnack-") and the extension's compile-time
        # gpu_arch are gfx950. Anything else computes garbage, not a
        # fallback (a forced hardware="amd_cdna4" on gfx942 fails here).
        # This runs ahead of super().__init__, which loads weights.
        from flash_rt.amd import flash_rt_amd_kernels as _fvk_gate
        # Compare the base target only ("gfx950" from
        # "gfx950:sramecc+:xnack-"); a prefix test would also accept a
        # future "gfx9500".
        _dev_arch = str(_fvk_gate.device_arch())
        _build_arch = str(dict(_fvk_gate.build_info()).get("gpu_arch",
                                                           "unknown"))
        if not (_dev_arch.split(":", 1)[0] == "gfx950"
                and _build_arch.split(":", 1)[0] == "gfx950"):
            raise RuntimeError(
                "GrootN17TorchFrontendAmd is gfx950-only (CDNA4 / "
                f"MI350-series): running device arch is {_dev_arch!r} and "
                f"the extension was built for gpu_arch {_build_arch!r}. "
                "Rebuild with scripts/amd/build_amd.sh on a gfx950 "
                "machine; other AMD arches are not supported by this "
                "backend.")

        # The strided-FMHA side-load is a CUDA-only .so (Thor ViT path);
        # the AMD ViT attention runs through the CDNA4 backend instead.
        super().__init__(
            checkpoint_path,
            num_views=num_views,
            embodiment_tag=embodiment_tag,
            device=device,
            load_strided_fmha=False,
        )
        # Seed the kernel-module / GEMM-runner handles with the AMD
        # surface BEFORE any base method runs its lazy
        # ``import flash_rt.flash_rt_kernels`` fallback — every such
        # import in the base is guarded by ``hasattr(self, "_fvk")`` /
        # ``hasattr(self, "_gemm")`` / ``hasattr(self, "_mlp_gemm")``.
        from flash_rt.amd import flash_rt_amd_kernels as fvk
        # (gfx950 gate already ran at the top of __init__.)
        self._fvk = fvk
        self._gemm = fvk.GemmRunner()
        self._mlp_gemm = fvk.GemmRunner()
        # Timed hipBLASLt algorithm selection on the first eager call of
        # each GEMM shape (all first calls happen pre-capture: shadow
        # calibration / set_prompt backbone / graph warmup). gfx950
        # heuristics have known gaps; pi05 measured meaningful wins from
        # timed picks. FLASHRT_FP8_NT_AUTOTUNE=off disables;
        # FLASHRT_FP8_ALGO_POOL sets the candidate pool (default 16 —
        # deeper pools widen run-to-run pick variance).
        import os as _os
        if _os.environ.get("FLASHRT_FP8_NT_AUTOTUNE", "auto").lower() != "off":
            pool = int(_os.environ.get("FLASHRT_FP8_ALGO_POOL", "16"))
            self._gemm.enable_lazy_autotune(pool)
            self._mlp_gemm.enable_lazy_autotune(pool)

    # ────────────────────────────────────────────────────────────────
    # Attention backend (single instance, all 5 sites)
    # ────────────────────────────────────────────────────────────────

    def set_hf_processor(self, processor) -> None:
        """Supply the HF processor instead of letting it be loaded.

        ``denormalize_action`` needs the processor for the
        relative-to-absolute action decode. It is normally built on
        demand from the checkpoint, which reaches the Hub for the
        tokenizer's metadata. Callers that must avoid that lookup —
        offline deployments, or environments hitting the transformers
        4.57.x bug described in :meth:`_hf_processor` — can build the
        processor themselves and inject it here before inference.
        """
        self._hf_proc_cached = processor

    def _hf_processor(self):
        """Build the HF processor, with an actionable offline error.

        Loading the processor resolves its tokenizer by Hub repository
        id, and transformers 4.57.x looks that repository's metadata up
        through ``huggingface_hub.model_info`` without catching
        ``OfflineModeIsEnabled``. With ``HF_HUB_OFFLINE=1`` the load then
        fails even when every file is already cached locally.

        The library does not work around this: the only interception
        point is ``huggingface_hub``'s module-level function, and
        swapping that out — even temporarily — mutates state shared with
        every other thread in the process. Instead the failure is
        reported with the two remedies that do not.
        """
        if hasattr(self, "_hf_proc_cached"):
            return self._hf_proc_cached
        try:
            return super()._hf_processor()
        except Exception as exc:
            if type(exc).__name__ != "OfflineModeIsEnabled":
                raise
            raise RuntimeError(
                "loading the GROOT processor requires a Hub metadata "
                "lookup that offline mode blocks. This is a transformers "
                "4.57.x issue (the tokenizer loader calls "
                "huggingface_hub.model_info without handling offline "
                "mode), not a missing file. Either allow that one "
                "lookup by clearing HF_HUB_OFFLINE for the load, or "
                "build the processor in your own setup code and pass it "
                "to set_hf_processor() before inference."
            ) from exc

    def _dit_kv_split(self) -> tuple:
        """(num_text_tokens, num_image_tokens) from the prompt's mask."""
        mask = self._visual_pos_masks
        n_text = int((~mask).sum().item())
        n_image = int(mask.sum().item())
        return n_text, n_image

    def _build_dit_attn(self, Sa: int) -> None:
        """Bind the CDNA4 backend for the DiT sites.

        Reuses the backbone-time backend when its DiT token capacity
        matches ``Sa``; otherwise constructs a fresh full backend with
        the same backbone geometry. When the eager torch path has
        already produced exact-size cross K/V tensors
        (``_precompute_dit_cross_kv``), their contents are copied into
        the backend's padded per-block slots; once
        ``_setup_cross_kv_kernel`` rebinds ``_dit_cross_K/V`` to slot
        prefix views, source and destination alias and the copy is
        skipped.
        """
        from flash_rt.amd.hardware.cdna4.attn_backend_groot_n17 import (
            Cdna4GrootN17AttnBackend,
        )

        Sa = int(Sa)
        n_text, n_image = self._dit_kv_split()
        kv_max = max(n_text, n_image)

        attn = getattr(self, "_kbb_attn", None)
        if attn is None or int(getattr(self, "_kbb_attn_sa", -1)) != Sa:
            attn = Cdna4GrootN17AttnBackend(
                num_vit_views=int(getattr(self, "_num_vit_views",
                                          self.num_views)),
                vit_seq=int(self._S_vit),
                llm_seq=int(self.Se),
                vl_self_attn_seq=int(self.Se),
                sa=Sa,
                dit_kv_seq=kv_max,
                device=self.device,
            )
        if hasattr(self, "_dit_cross_K"):
            for j, (k_src, v_src) in enumerate(
                    zip(self._dit_cross_K, self._dit_cross_V)):
                rows = int(k_src.shape[0])
                k_dst = attn.dit_cross_K[j].view(kv_max, -1)[:rows]
                v_dst = attn.dit_cross_V[j].view(kv_max, -1)[:rows]
                if k_dst.data_ptr() != k_src.data_ptr():
                    k_dst.copy_(k_src)
                if v_dst.data_ptr() != v_src.data_ptr():
                    v_dst.copy_(v_src)
        self._dit_attn = attn

    # ────────────────────────────────────────────────────────────────
    # Kernelized DiT cross-KV (writes straight into backend slots)
    # ────────────────────────────────────────────────────────────────

    def _setup_cross_kv_kernel(self) -> None:
        """Persistent cross-KV buffers over the CDNA4 backend slots.

        Mirrors the Thor base with one structural delta: the per-block
        K/V destinations are exact-size prefix views of the backend's
        padded ``dit_cross_K/V[j]`` slots (row-major ``(kv, NH*HD)``
        prefix of ``(kv_max, NH, HD)``), so the projection GEMMs land
        their output exactly where ``attn.run("dit_cross", ...)`` reads.
        """
        if hasattr(self, "_ck_text_idx"):
            return
        if not hasattr(self, "_dit_attn"):
            raise RuntimeError(
                "_setup_cross_kv_kernel requires the DiT attention backend; "
                "call _build_dit_attn first")
        dev = self.device
        S = self.Se
        mask = self._visual_pos_masks
        self._ck_text_idx = torch.where(~mask)[0].to(torch.int64).contiguous()
        self._ck_image_idx = torch.where(mask)[0].to(torch.int64).contiguous()
        nt = int(self._ck_text_idx.numel())
        ni = int(self._ck_image_idx.numel())
        self._ck_nt, self._ck_ni = nt, ni
        bf = torch.bfloat16
        # Per-frame backbone input (fp16, copied in before replay) + bf16 cast
        self._ck_bb_src = torch.empty(S, 2048, dtype=torch.float16, device=dev)
        self._ck_bb = torch.empty(S, 2048, dtype=bf, device=dev)
        self._ck_text_src = torch.empty(nt, 2048, dtype=bf, device=dev)
        self._ck_image_src = torch.empty(ni, 2048, dtype=bf, device=dev)

        # Cross K/V destinations = exact-size prefix views of the backend
        # slots. Block j maps to full-layer index li = 2j; text-target
        # blocks (li % 4 == 0) are the even j.
        attn = self._dit_attn
        kv_max = int(attn.dit_cross_K[0].shape[0])
        D = 1536
        self._dit_cross_K = [
            attn.dit_cross_K[j].view(kv_max, D)[: (nt if j % 2 == 0 else ni)]
            for j in range(16)]
        self._dit_cross_V = [
            attn.dit_cross_V[j].view(kv_max, D)[: (nt if j % 2 == 0 else ni)]
            for j in range(16)]

        # Seed from the current backbone (eager) so a non-graph consumer
        # sees valid K/V immediately.
        self._ck_bb_src.copy_(self._backbone_features.reshape(S, 2048).half())
        self._cross_kv_fwd(0)

    def _cross_kv_fwd(self, s: int) -> None:
        """Cross-KV forward over the persistent buffers (graph-safe).

        Reads ``_ck_bb_src`` (current backbone, fp16), writes the
        backend-slot K/V prefixes. The text/image row gather runs as
        ``torch.index_select`` with preallocated ``out=`` buffers — the
        AMD kernel module has no ``embedding_lookup_bf16`` binding. The
        torch calls land on torch's current stream; callers keep it
        consistent with the raw ``s`` int (stream 0 eagerly, or the
        capture stream via its ``torch.cuda.stream`` context).
        """
        K = self._fvk
        mg = self._mlp_gemm
        S = self.Se
        nt, ni = self._ck_nt, self._ck_ni
        fused = _fused_epilogue_enabled()
        K.cast_fp16_to_bf16(
            self._ck_bb_src.data_ptr(), self._ck_bb.data_ptr(),
            S * 2048, int(s))
        torch.index_select(
            self._ck_bb, 0, self._ck_text_idx, out=self._ck_text_src)
        torch.index_select(
            self._ck_bb, 0, self._ck_image_idx, out=self._ck_image_src)
        for j in range(16):
            li = 2 * j
            text = (li % 4 == 0)
            N = nt if text else ni
            src = self._ck_text_src if text else self._ck_image_src
            k_dst = self._dit_cross_K[j]
            v_dst = self._dit_cross_V[j]
            k_w = self._dit_k_w[li]
            k_b = self._dit_k_b[li]
            v_w = self._dit_v_w[li]
            v_b = self._dit_v_b[li]
            if fused:
                # AMD FUSED: bf16_nn + add_bias_bf16 → bf16_nn_bias (cross K/V).
                mg.bf16_nn_bias(src.data_ptr(), k_w.data_ptr(),
                                k_dst.data_ptr(), k_b.data_ptr(),
                                N, 1536, 2048, int(s))
                mg.bf16_nn_bias(src.data_ptr(), v_w.data_ptr(),
                                v_dst.data_ptr(), v_b.data_ptr(),
                                N, 1536, 2048, int(s))
            else:
                mg.bf16_nn(src.data_ptr(), k_w.data_ptr(), k_dst.data_ptr(),
                           N, 1536, 2048, int(s))
                K.add_bias_bf16(k_dst.data_ptr(), k_b.data_ptr(), N, 1536,
                                int(s))
                mg.bf16_nn(src.data_ptr(), v_w.data_ptr(), v_dst.data_ptr(),
                           N, 1536, 2048, int(s))
                K.add_bias_bf16(v_dst.data_ptr(), v_b.data_ptr(), N, 1536,
                                int(s))

    # ────────────────────────────────────────────────────────────────
    # Fully-kernelized DiT graph (bf16, single combined graph)
    # ────────────────────────────────────────────────────────────────

    def _pack_smallm_dit_weights(self) -> None:
        """MFMA-pack the DiT (41, 1536, 1536) projection weights ONCE at
        graph-build time (never per frame) for the gfx950 small-M packed
        bf16 kernel (csrc/amd/gemm/smallm_mfma_bf16.h).

        Measured routing (standalone gate vs autotuned hipBLASLt): the
        packed kernel wins ONLY on the square D→D shape — bias 1.41x,
        bias_res 1.12x — and loses on ff1/ff2, so only the projection
        weights feeding those sites get packed copies: Q (all 32
        layers), K/V (the 16 self blocks — the cross blocks' K/V
        weights are the (2048, 1536) cross-KV projections, a different
        shape consumed by ``_setup_cross_kv_kernel``), O (all 32).

        Packed copies land in ``self._dit_smallm_packed`` keyed
        ``{q,k,v,o}_w_{li}`` (pointer ints, matching the pipeline
        ``weights`` lists); the tensors are kept alive in
        ``self._dit_smallm_store``. Originals are KEPT — they serve the
        FVK_AMD_DIT_GEMM=hipblaslt fallback and the cross-KV consumer —
        at ~432 MB extra VRAM for the 96 packed copies.
        FVK_AMD_DIT_GEMM=hipblaslt (read once here, setup time) skips
        the packing entirely; the AMD ``dit_forward`` reads the same
        env for routing.
        """
        if getattr(self, "_dit_smallm_packed", None) is not None:
            return
        self._dit_smallm_packed: dict = {}
        self._dit_smallm_store: list = []
        # FVK_AMD_DIT_GEMM: "smallm" (default) = pack + route the D→D
        # projections to the MFMA packed kernel; "hipblaslt" = library path.
        if os.environ.get("FVK_AMD_DIT_GEMM", "smallm").strip().lower() \
                != "smallm":
            return
        with torch.no_grad():
            for li in range(32):
                sites = [("q_w", self._dit_q_w), ("o_w", self._dit_o_w)]
                if li % 2 == 1:  # self blocks only (K/V)
                    sites += [("k_w", self._dit_k_w),
                              ("v_w", self._dit_v_w)]
                for key, wl in sites:
                    W = wl[li]  # (K, N) row-major bf16, both 1536
                    Kd, Nd = W.shape
                    # Per-lane consumption order — the documented
                    # one-liner from smallm_mfma_bf16.h.
                    wp = (W.view(Kd // 32, 4, 8, Nd // 16, 16)
                           .permute(3, 0, 1, 4, 2).contiguous())
                    self._dit_smallm_store.append(wp)
                    self._dit_smallm_packed[f"{key}_{li}"] = wp.data_ptr()

    def _capture_kernel_dit_graphs(self, num_inference_timesteps: int = 4,
                                   action_horizon: int = 40) -> None:
        """Capture the per-frame action-head chain as ONE HIP graph.

        AMD adaptation of the Thor base method: same buffer set, same
        per-step closures, same combined cross-KV + state-encode +
        4-step-DiT graph, with three deltas (see module docstring):
        bf16-only DiT (no FP8/FP4 splice), a torch in-place Euler add,
        and warmup ON the capture stream per the aiter capture-safety
        contract.
        """
        from flash_rt.models.groot_n17 import pipeline_thor

        # FVK_AMD_FUSED_EPILOGUE flips BOTH fusions (backbone + DiT) for
        # the A/B: on → the AMD-local dit_forward (bf16_nn_bias /
        # bf16_nn_bias_gelu fused epilogues); off → the byte-identical
        # decomposed pipeline_thor.dit_forward. Only dit_forward is
        # AMD-local; embodiment_* stages keep coming from pipeline_thor.
        fused_ep = _fused_epilogue_enabled()
        if fused_ep:
            from flash_rt.amd.models.groot_n17 import pipeline as _amd_pipeline
            dit_forward = _amd_pipeline.dit_forward
        else:
            dit_forward = pipeline_thor.dit_forward

        if getattr(self, "_DIT_USE_FP8", False) or \
                getattr(self, "_DIT_QUANT", "fp8") == "fp4":
            raise NotImplementedError(
                "the AMD CDNA4 N1.7 frontend runs the DiT bf16; the FP8/FP4 "
                "DiT quantization tiers are not ported")

        Sa = action_horizon + 1
        if not hasattr(self, "_dit_attn"):
            self._build_dit_attn(Sa)
        self._setup_cross_kv_kernel()
        if not hasattr(self, "_infer_bufs"):
            self._allocate_infer_buffers(action_horizon)
        self._prepare_kernel_dit(num_inference_timesteps)
        self._allocate_kernel_dit_buffers(action_horizon)
        if fused_ep:
            # Setup-time MFMA packing for the smallm D→D projection
            # routing in the AMD dit_forward (FVK_AMD_DIT_GEMM).
            self._pack_smallm_dit_weights()

        K = self._fvk
        mg = self._mlp_gemm
        w = self._kw
        bufs = self._infer_bufs
        dit_h = bufs["dit_h"].data_ptr()
        Skv_text = int(self._dit_cross_K[0].shape[0])
        Skv_image = int(self._dit_cross_K[1].shape[0])
        dims = {"Sa": Sa, "D": 1536, "FF": 6144,
                "Skv_text": Skv_text, "Skv_image": Skv_image}
        bp = {"h": dit_h, "xn": bufs["dit_xn"].data_ptr(),
              "o_proj_out": bufs["dit_o_proj_out"].data_ptr(),
              "ff_proj_out": bufs["dit_ff_proj_out"].data_ptr()}
        dt = 1.0 / num_inference_timesteps
        # decode reads the action rows (1..Sa) of the (Sa, 1024) output_proj
        hout_dec = self._k_hout.data_ptr() + 1024 * 2  # skip state row

        def _dit_weights(step):
            d = {"scale_msa": [t.data_ptr() for t in self._step_scales[step]],
                 "shift_msa": [t.data_ptr() for t in self._step_shifts[step]]}
            for key, attr in (("q_w", "_dit_q_w"), ("q_b", "_dit_q_b"),
                              ("k_w", "_dit_k_w"), ("k_b", "_dit_k_b"),
                              ("v_w", "_dit_v_w"), ("v_b", "_dit_v_b"),
                              ("o_w", "_dit_o_w"), ("o_b", "_dit_o_b"),
                              ("ff_proj_w", "_dit_ff_proj_w"),
                              ("ff_proj_b", "_dit_ff_proj_b"),
                              ("ff_down_w", "_dit_ff_down_w"),
                              ("ff_down_b", "_dit_ff_down_b")):
                d[key] = [t.data_ptr() for t in getattr(self, attr)]
            if fused_ep:
                # MFMA-packed q/k/v/o copies for the smallm routing in
                # the AMD dit_forward (same dict for every step —
                # weights are step-invariant). Empty dict when
                # FVK_AMD_DIT_GEMM=hipblaslt skipped the packing.
                d["smallm_packed"] = self._dit_smallm_packed
            return d

        step_weights = [_dit_weights(s) for s in range(num_inference_timesteps)]

        def _state_fwd(s):
            if fused_ep:
                # AMD FUSED: bf16_nn + add_bias_bf16 → bf16_nn_bias (state l1).
                mg.bf16_nn_bias(self._k_state_in.data_ptr(),
                                w["st_l1"].data_ptr(),
                                self._k_st_h1.data_ptr(),
                                w["st_l1b"].data_ptr(), 1, 1024, 132, s)
            else:
                mg.bf16_nn(self._k_state_in.data_ptr(), w["st_l1"].data_ptr(),
                           self._k_st_h1.data_ptr(), 1, 1024, 132, s)
                K.add_bias_bf16(self._k_st_h1.data_ptr(),
                                w["st_l1b"].data_ptr(), 1, 1024, s)
            K.relu_inplace_bf16(self._k_st_h1.data_ptr(), 1024, s)
            if fused_ep:
                # AMD FUSED: bf16_nn + add_bias_bf16 → bf16_nn_bias (state l2).
                mg.bf16_nn_bias(self._k_st_h1.data_ptr(),
                                w["st_l2"].data_ptr(),
                                self._k_state_feat.data_ptr(),
                                w["st_l2b"].data_ptr(), 1, 1536, 1024, s)
            else:
                mg.bf16_nn(self._k_st_h1.data_ptr(), w["st_l2"].data_ptr(),
                           self._k_state_feat.data_ptr(), 1, 1536, 1024, s)
                K.add_bias_bf16(self._k_state_feat.data_ptr(),
                                w["st_l2b"].data_ptr(), 1, 1536, s)

        def _ae_fwd(step, s):
            # action_encode: W1 (no act) → cat[a_emb, tau] → W2 → SiLU → W3,
            # add pos, then fill dit_h ([0]=state, [1:]=action features).
            T = action_horizon
            if fused_ep:
                # AMD FUSED: bf16_nn + add_bias_bf16 → bf16_nn_bias (ae W1).
                mg.bf16_nn_bias(self._k_actions.data_ptr(),
                                w["ae_W1"].data_ptr(),
                                self._k_ae_aemb.data_ptr(),
                                w["ae_b1"].data_ptr(), T, 1536, 132, s)
            else:
                mg.bf16_nn(self._k_actions.data_ptr(), w["ae_W1"].data_ptr(),
                           self._k_ae_aemb.data_ptr(), T, 1536, 132, s)
                K.add_bias_bf16(self._k_ae_aemb.data_ptr(),
                                w["ae_b1"].data_ptr(), T, 1536, s)
            K.concat2_bf16(self._k_ae_aemb.data_ptr(),
                           self._k_tau[step].data_ptr(),
                           self._k_ae_concat.data_ptr(), T, 1536, 1536, s)
            if fused_ep:
                # AMD FUSED: bf16_nn + add_bias_bf16 → bf16_nn_bias (ae W2).
                mg.bf16_nn_bias(self._k_ae_concat.data_ptr(),
                                w["ae_W2"].data_ptr(),
                                self._k_ae_i2.data_ptr(),
                                w["ae_b2"].data_ptr(), T, 1536, 3072, s)
            else:
                mg.bf16_nn(self._k_ae_concat.data_ptr(), w["ae_W2"].data_ptr(),
                           self._k_ae_i2.data_ptr(), T, 1536, 3072, s)
                K.add_bias_bf16(self._k_ae_i2.data_ptr(),
                                w["ae_b2"].data_ptr(), T, 1536, s)
            K.cast_bf16_to_fp16(self._k_ae_i2.data_ptr(),
                                self._k_ae_i2f.data_ptr(), T * 1536, s)
            K.silu_inplace_fp16(self._k_ae_i2f.data_ptr(), T * 1536, s)
            K.cast_fp16_to_bf16(self._k_ae_i2f.data_ptr(),
                                self._k_ae_i2.data_ptr(), T * 1536, s)
            if fused_ep:
                # AMD FUSED: bf16_nn + add_bias_bf16 → bf16_nn_bias (ae W3).
                mg.bf16_nn_bias(self._k_ae_i2.data_ptr(),
                                w["ae_W3"].data_ptr(),
                                self._k_ae_out.data_ptr(),
                                w["ae_b3"].data_ptr(), T, 1536, 1536, s)
            else:
                mg.bf16_nn(self._k_ae_i2.data_ptr(), w["ae_W3"].data_ptr(),
                           self._k_ae_out.data_ptr(), T, 1536, 1536, s)
                K.add_bias_bf16(self._k_ae_out.data_ptr(),
                                w["ae_b3"].data_ptr(), T, 1536, s)
            K.residual_add(self._k_ae_out.data_ptr(), self._k_pos.data_ptr(),
                           T * 1536, s)
            K.gpu_copy(dit_h, self._k_state_feat.data_ptr(), 1536 * 2, s)
            K.gpu_copy(dit_h + 1536 * 2, self._k_ae_out.data_ptr(),
                       T * 1536 * 2, s)

        def _post_fwd(step, s):
            # output projection (AdaLN → proj_out_2) + action_decode + Euler.
            T = action_horizon
            K.ada_layer_norm_bf16(dit_h, self._k_oproj_scale[step].data_ptr(),
                                  self._k_oproj_shift[step].data_ptr(),
                                  self._k_hmod.data_ptr(), Sa, 1536, 1e-5, s)
            if fused_ep:
                # AMD FUSED: bf16_nn + add_bias_bf16 → bf16_nn_bias (po2).
                mg.bf16_nn_bias(self._k_hmod.data_ptr(), w["po2"].data_ptr(),
                                self._k_hout.data_ptr(),
                                w["po2b"].data_ptr(), Sa, 1024, 1536, s)
                # AMD FUSED: bf16_nn + add_bias_bf16 → bf16_nn_bias (dec l1).
                mg.bf16_nn_bias(hout_dec, w["dec_l1"].data_ptr(),
                                self._k_dec_h.data_ptr(),
                                w["dec_l1b"].data_ptr(), T, 1024, 1024, s)
            else:
                mg.bf16_nn(self._k_hmod.data_ptr(), w["po2"].data_ptr(),
                           self._k_hout.data_ptr(), Sa, 1024, 1536, s)
                K.add_bias_bf16(self._k_hout.data_ptr(), w["po2b"].data_ptr(),
                                Sa, 1024, s)
                mg.bf16_nn(hout_dec, w["dec_l1"].data_ptr(),
                           self._k_dec_h.data_ptr(), T, 1024, 1024, s)
                K.add_bias_bf16(self._k_dec_h.data_ptr(),
                                w["dec_l1b"].data_ptr(), T, 1024, s)
            K.relu_inplace_bf16(self._k_dec_h.data_ptr(), T * 1024, s)
            if fused_ep:
                # AMD FUSED: bf16_nn + add_bias_bf16 → bf16_nn_bias (dec l2).
                mg.bf16_nn_bias(self._k_dec_h.data_ptr(),
                                w["dec_l2"].data_ptr(),
                                self._k_vel.data_ptr(),
                                w["dec_l2b"].data_ptr(), T, 132, 1024, s)
            else:
                mg.bf16_nn(self._k_dec_h.data_ptr(), w["dec_l2"].data_ptr(),
                           self._k_vel.data_ptr(), T, 132, 1024, s)
                K.add_bias_bf16(self._k_vel.data_ptr(),
                                w["dec_l2b"].data_ptr(), T, 132, s)
            # Euler update: actions += dt * velocity, in place on the
            # persistent buffers. The AMD module has no euler_step
            # binding; an in-place torch add is capture-safe (fixed
            # shapes, no allocation) and lands on the capture stream via
            # the surrounding torch.cuda.stream context.
            self._k_actions.add_(self._k_vel, alpha=dt)

        def _step_fwd(step, s):
            _ae_fwd(step, s)
            dit_forward(
                gemm=self._gemm, fvk=K, bufs=bp, weights=step_weights[step],
                dims=dims, attn=self._dit_attn, stream=s)
            _post_fwd(step, s)

        self._kdit_fwd = (_state_fwd, _step_fwd)
        self._k_nsteps = num_inference_timesteps

        def _dit_all(s):
            self._cross_kv_fwd(s)
            _state_fwd(s)
            for step in range(num_inference_timesteps):
                _step_fwd(step, s)

        # aiter capture-safety contract (see the CDNA4 backend docstring):
        # aiter may allocate LSE/workspace through torch's caching
        # allocator per call, so the warmup iterations must run ON the
        # capture stream (allocator steady state per stream) and every
        # torch-side call must land on that same stream. The warmup also
        # primes the hipBLASLt GemmRunner algo caches for every captured
        # GEMM shape.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            s_int = stream.cuda_stream
            for _ in range(3):
                _dit_all(s_int)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            graph.capture_begin()
            _dit_all(stream.cuda_stream)
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self._k_dit_graph = graph

    # ────────────────────────────────────────────────────────────────
    # FP8 kernel backbone (AMD pipeline, native-GQA llm slots)
    # ────────────────────────────────────────────────────────────────

    def _run_kernel_backbone_fp8(self, aux: dict) -> "torch.Tensor":
        """ViT → DeepStack → LLM → vlln → VL-self-attn on AMD FP8 kernels.

        Mirror of the RTX mixin method retargeted at
        :mod:`flash_rt.amd.models.groot_n17.pipeline` and the CDNA4
        attention backend. The llm stage's Q/K/V descale GEMMs write
        straight into the backend slots (Q 16 heads, K/V the native
        8 KV heads); no K/V staging buffers and no K_exp/V_exp scratch
        are allocated — aiter consumes the 8 KV heads natively.
        """
        from flash_rt.amd.models.groot_n17 import pipeline as P
        from flash_rt.amd.hardware.cdna4.attn_backend_groot_n17 import (
            Cdna4GrootN17AttnBackend,
        )

        fvkm, gemm = self._fvk, self._gemm
        dev = self.device
        Sv, nv, Se = self._S_vit, self._num_vit_views, self.Se

        keep: list = []
        self._kbb_keep = keep

        def K(t):
            keep.append(t)
            return t

        def buf(*shape):
            return K(torch.empty(*shape, dtype=_FP16, device=dev))

        def buf8(*shape):
            return K(torch.empty(*shape, dtype=_U8, device=dev))

        def wsc(val):
            """Upload a host weight scale to a device fp32 scalar; keep ref."""
            t = K(torch.tensor([float(val)], dtype=torch.float32, device=dev))
            return t.data_ptr()

        def adv(dev_list):
            """Device act-scale scalar tensors → list of int ptrs."""
            return [t.data_ptr() for t in dev_list]

        # ── FUSED-EPILOGUE tier (FVK_AMD_FUSED_EPILOGUE, default on) ──
        # Host alphas for the fused fp8_nn_bias / fp8_nn_gelu_bias calls.
        # _bake_calibration (already run via _ensure_act_scales) composes
        # them as python floats: alpha = act_scale × w_scale per site per
        # layer. Key names parallel the scales_dev dicts. The llm stage is
        # biasless (Qwen3) and always stays on descale GEMMs; its
        # fused_epilogue flag fuses the norm/residual+quantize chains
        # instead (rms_norm_fp8_fp16 / residual_add_rms_norm_fp8_fp16 —
        # no alphas needed).
        fused = _fused_epilogue_enabled()
        if fused:
            vit_alphas = {
                "act_qkv": [float(a) for a in self._vit_alpha_q],
                "act_o":   [float(a) for a in self._vit_alpha_o],
                "act_fc1": [float(a) for a in self._vit_alpha_fc1],
                "act_fc2": [float(a) for a in self._vit_alpha_fc2],
            }
            ds_alphas = {
                "act_fc1": [float(a) for a in self._dsm_alpha_fc1],
                "act_fc2": [float(a) for a in self._dsm_alpha_fc2],
            }
            # vlsa Q/K/V have separate weight scales → per-layer 3-tuples.
            vlsa_alphas = {
                "act_qkv": [
                    (float(q), float(k), float(v))
                    for q, k, v in zip(self._vlsa_alpha_q,
                                       self._vlsa_alpha_k,
                                       self._vlsa_alpha_v)],
                "act_o":   [float(a) for a in self._vlsa_alpha_o],
                "act_fc1": [float(a) for a in self._vlsa_alpha_fc1],
                "act_fc2": [float(a) for a in self._vlsa_alpha_fc2],
            }
        else:
            vit_alphas = ds_alphas = vlsa_alphas = None

        # One backend for the whole model: backbone sites now, DiT sites
        # at first infer (reused by _build_dit_attn when Sa matches).
        n_text, n_image = self._dit_kv_split()
        sa = int(self._DEFAULT_SA)
        attn = Cdna4GrootN17AttnBackend(
            num_vit_views=nv, vit_seq=Sv, llm_seq=Se, vl_self_attn_seq=Se,
            sa=sa, dit_kv_seq=max(n_text, n_image), device=dev)
        self._kbb_attn = attn
        self._kbb_attn_sa = sa

        # ═══ ViT (24L) ═══
        vit_h = buf(Sv, 1024)
        vit_h.copy_(aux["pixel_features"].to(dev).half().reshape(Sv, 1024))
        vit_bufs = {"h": vit_h.data_ptr(), "xn": buf(Sv, 1024).data_ptr(),
                    "xn_fp8": buf8(Sv, 1024).data_ptr(),
                    "o_proj_out": buf(Sv, 1024).data_ptr(),
                    "fc1_out": buf(Sv, 4096).data_ptr(),
                    "fc1_fp8": buf8(Sv, 4096).data_ptr()}
        vw = {k: [] for k in (
            "norm1_w", "norm1_b", "norm2_w", "norm2_b", "q_w", "q_b",
            "k_w", "k_b", "v_w", "v_b", "o_w", "o_b", "fc1_w", "fc1_b",
            "fc2_w", "fc2_b", "q_ws", "k_ws", "v_ws", "o_ws",
            "fc1_ws", "fc2_ws")}
        vw["cos"] = self._vit_cos.data_ptr()
        vw["sin"] = self._vit_sin.data_ptr()
        for li in range(24):
            qkv = self._vit_qkv_w[li]               # fp8 (1024, 3072) [K, 3N]
            b = self._vit_qkv_b[li]                 # (3072,)
            q = K(qkv[:, :1024].contiguous())
            kk = K(qkv[:, 1024:2048].contiguous())
            v = K(qkv[:, 2048:].contiguous())
            qb = K(b[:1024].contiguous())
            kb = K(b[1024:2048].contiguous())
            vb = K(b[2048:].contiguous())
            qkv_ws = wsc(self._vit_alpha[li * 4 + 0])
            vw["norm1_w"].append(self._vit_ln1_w[li].data_ptr())
            vw["norm1_b"].append(self._vit_ln1_b[li].data_ptr())
            vw["norm2_w"].append(self._vit_ln2_w[li].data_ptr())
            vw["norm2_b"].append(self._vit_ln2_b[li].data_ptr())
            vw["q_w"].append(q.data_ptr()); vw["q_b"].append(qb.data_ptr())
            vw["k_w"].append(kk.data_ptr()); vw["k_b"].append(kb.data_ptr())
            vw["v_w"].append(v.data_ptr()); vw["v_b"].append(vb.data_ptr())
            vw["q_ws"].append(qkv_ws)
            vw["k_ws"].append(qkv_ws)
            vw["v_ws"].append(qkv_ws)
            vw["o_w"].append(self._vit_o_w[li].data_ptr())
            vw["o_b"].append(self._vit_o_b[li].data_ptr())
            vw["o_ws"].append(wsc(self._vit_alpha[li * 4 + 1]))
            vw["fc1_w"].append(self._vit_fc1_w[li].data_ptr())
            vw["fc1_b"].append(self._vit_fc1_b[li].data_ptr())
            vw["fc1_ws"].append(wsc(self._vit_alpha[li * 4 + 2]))
            vw["fc2_w"].append(self._vit_fc2_w[li].data_ptr())
            vw["fc2_b"].append(self._vit_fc2_b[li].data_ptr())
            vw["fc2_ws"].append(wsc(self._vit_alpha[li * 4 + 3]))
        vit_scales = {
            "act_qkv": adv(self._vit_act_qkv_dev),
            "act_o": adv(self._vit_act_o_dev),
            "act_fc1": adv(self._vit_act_fc1_dev),
            "act_fc2": adv(self._vit_act_fc2_dev)}

        tap_layers = (5, 11, 17)
        tap_bufs = {l: buf(Sv, 1024) for l in tap_layers}
        scell = [0]
        self._kbb_scell = scell

        def mk_cb(l):
            def cb(h_ptr):
                fvkm.gpu_copy(
                    tap_bufs[l].data_ptr(), int(h_ptr), Sv * 1024 * 2,
                    scell[0])
            return cb
        dcap = [mk_cb(l) for l in tap_layers]

        P.qwen3vl_vit_forward(
            gemm=gemm, fvk=fvkm, bufs=vit_bufs, weights=vw,
            scales_dev=vit_scales,
            dims={"S": Sv, "D": 1024, "NH": 16, "HD": 64,
                  "ff_inner": 4096, "Sper_view": Sv // nv},
            attn=attn, deepstack_taps=tap_layers, deepstack_capture=dcap,
            fused_epilogue=fused, alphas=vit_alphas)

        # ═══ DeepStack (3 mergers) ═══
        Nout = Sv // 4
        ds_out = [buf(Nout, 2048) for _ in range(3)]
        dsw = {k: [] for k in ("norm_w", "norm_b", "fc1_w", "fc1_b",
                               "fc2_w", "fc2_b", "fc1_ws", "fc2_ws")}
        for j in range(3):
            dsw["norm_w"].append(getattr(self, f"_dsm{j}_norm_w").data_ptr())
            dsw["norm_b"].append(getattr(self, f"_dsm{j}_norm_b").data_ptr())
            dsw["fc1_w"].append(getattr(self, f"_dsm{j}_fc1_w").data_ptr())
            dsw["fc1_b"].append(getattr(self, f"_dsm{j}_fc1_b").data_ptr())
            dsw["fc1_ws"].append(wsc(self._dsm_alpha[j * 2 + 0]))
            dsw["fc2_w"].append(getattr(self, f"_dsm{j}_fc2_w").data_ptr())
            dsw["fc2_b"].append(getattr(self, f"_dsm{j}_fc2_b").data_ptr())
            dsw["fc2_ws"].append(wsc(self._dsm_alpha[j * 2 + 1]))
        ds_scales = {"act_fc1": adv(self._dsm_act_fc1_dev),
                     "act_fc2": adv(self._dsm_act_fc2_dev)}
        ds_bufs = {"in": [tap_bufs[l].data_ptr() for l in tap_layers],
                   "ln_out": buf(Nout, 4096).data_ptr(),
                   "fp8_scratch": buf8(Nout, 4096).data_ptr(),
                   "fc1_out": buf(Nout, 4096).data_ptr(),
                   "out": [t.data_ptr() for t in ds_out]}
        ds_dims = {"Nin": Sv, "Din": 1024, "Nout": Nout,
                   "Dmid": 4096, "Dout": 2048}
        P.deepstack_merge_forward(
            gemm=gemm, fvk=fvkm, bufs=ds_bufs,
            weights=dsw, scales_dev=ds_scales, dims=ds_dims,
            fused_epilogue=fused, alphas=ds_alphas)

        # DeepStack inject buffers (S, D) — zero except visual positions.
        mask = self._visual_pos_masks
        vis_idx = K(mask.reshape(-1).nonzero(as_tuple=True)[0].to(torch.long))
        inject = [0] * 16
        injb = []
        for j in range(3):
            ib = buf(Se, 2048)
            ib.zero_()
            ib.index_copy_(0, vis_idx, ds_out[j])
            inject[j] = ib.data_ptr()
            injb.append(ib)

        # ═══ LLM (16L, causal, native GQA) ═══
        llm_h = buf(Se, 2048)
        llm_h.copy_(aux["llm_input_embeds"].to(dev).half().reshape(Se, 2048))
        lw = {k: [] for k in (
            "in_ln_w", "post_ln_w", "q_norm_w", "k_norm_w", "q_w", "k_w",
            "v_w", "o_w", "gate_w", "up_w", "down_w",
            "q_ws", "k_ws", "v_ws", "o_ws", "gate_ws", "up_ws", "down_ws")}
        lw["cos"] = self._mrope_cos.data_ptr()
        lw["sin"] = self._mrope_sin.data_ptr()
        lw["deepstack_inject"] = inject
        for li in range(16):
            qkv = self._llm_qkv_w[li]    # fp8 (2048, 4096) [K, NHQ·HD+2·NHKV·HD]
            q = K(qkv[:, :2048].contiguous())
            kk = K(qkv[:, 2048:3072].contiguous())
            v = K(qkv[:, 3072:4096].contiguous())
            qkv_ws = wsc(self._llm_alpha[li * 5 + 0])
            lw["in_ln_w"].append(self._llm_input_ln_w[li].data_ptr())
            lw["post_ln_w"].append(self._llm_post_ln_w[li].data_ptr())
            lw["q_norm_w"].append(self._llm_q_norm_w[li].data_ptr())
            lw["k_norm_w"].append(self._llm_k_norm_w[li].data_ptr())
            lw["q_w"].append(q.data_ptr())
            lw["k_w"].append(kk.data_ptr())
            lw["v_w"].append(v.data_ptr())
            lw["q_ws"].append(qkv_ws)
            lw["k_ws"].append(qkv_ws)
            lw["v_ws"].append(qkv_ws)
            lw["o_w"].append(self._llm_o_w[li].data_ptr())
            lw["o_ws"].append(wsc(self._llm_alpha[li * 5 + 1]))
            lw["gate_w"].append(self._llm_gate_w[li].data_ptr())
            lw["gate_ws"].append(wsc(self._llm_alpha[li * 5 + 2]))
            lw["up_w"].append(self._llm_up_w[li].data_ptr())
            lw["up_ws"].append(wsc(self._llm_alpha[li * 5 + 3]))
            lw["down_w"].append(self._llm_down_w[li].data_ptr())
            lw["down_ws"].append(wsc(self._llm_alpha[li * 5 + 4]))
        llm_scales = {
            "act_qkv": adv(self._llm_act_qkv_dev),
            "act_o": adv(self._llm_act_o_dev),
            "act_gateup": adv(self._llm_act_gateup_dev),
            "act_down": adv(self._llm_act_down_dev)}
        # AMD delta vs the RTX mixin: the Q/K/V descale GEMMs land in the
        # backend slots (Q 16 heads, K/V native 8 heads) and aiter runs
        # GQA internally, so the RTX Q/K/V staging buffers and the
        # K_exp/V_exp expand scratch are not allocated at all.
        llm_bufs = {
            "h": llm_h.data_ptr(), "xn": buf(Se, 2048).data_ptr(),
            "xn_fp8": buf8(Se, 2048).data_ptr(),
            "o_proj_out": buf(Se, 2048).data_ptr(),
            "gate_out": buf(Se, 6144).data_ptr(),
            "up_out": buf(Se, 6144).data_ptr(),
            "gu_fp8": buf8(Se, 6144).data_ptr()}
        llm_dims = {"S": Se, "D": 2048, "NHQ": 16, "NHKV": 8,
                    "HD": 128, "FF": 6144}
        P.qwen3vl_llm_forward(
            gemm=gemm, fvk=fvkm, bufs=llm_bufs, weights=lw,
            scales_dev=llm_scales, dims=llm_dims, attn=attn,
            fused_epilogue=fused)

        # ═══ vlln + VL self-attn (4L) ═══
        vlsa_h = buf(Se, 2048)
        vlln_bufs = {"x": llm_h.data_ptr(), "out": vlsa_h.data_ptr()}
        vlln_weights = {"vlln_w": self._vlln_w.data_ptr(),
                        "vlln_b": self._vlln_b.data_ptr()}
        P.vlln_forward(
            gemm=gemm, fvk=fvkm, bufs=vlln_bufs, weights=vlln_weights,
            dims={"S": Se, "D": 2048})
        vsw = {k: [] for k in (
            "norm1_w", "norm1_b", "norm3_w", "norm3_b", "q_w", "q_b",
            "k_w", "k_b", "v_w", "v_b", "o_w", "o_b", "fc1_w", "fc1_b",
            "fc2_w", "fc2_b", "q_ws", "k_ws", "v_ws", "o_ws",
            "fc1_ws", "fc2_ws")}
        for li in range(4):
            vsw["norm1_w"].append(self._vlsa_norm1_w[li].data_ptr())
            vsw["norm1_b"].append(self._vlsa_norm1_b[li].data_ptr())
            vsw["norm3_w"].append(self._vlsa_norm3_w[li].data_ptr())
            vsw["norm3_b"].append(self._vlsa_norm3_b[li].data_ptr())
            vsw["q_w"].append(self._vlsa_q_w[li].data_ptr())
            vsw["q_b"].append(self._vlsa_q_b[li].data_ptr())
            vsw["q_ws"].append(wsc(self._vlsa_alpha[li * 6 + 0]))
            vsw["k_w"].append(self._vlsa_k_w[li].data_ptr())
            vsw["k_b"].append(self._vlsa_k_b[li].data_ptr())
            vsw["k_ws"].append(wsc(self._vlsa_alpha[li * 6 + 1]))
            vsw["v_w"].append(self._vlsa_v_w[li].data_ptr())
            vsw["v_b"].append(self._vlsa_v_b[li].data_ptr())
            vsw["v_ws"].append(wsc(self._vlsa_alpha[li * 6 + 2]))
            vsw["o_w"].append(self._vlsa_o_w[li].data_ptr())
            vsw["o_b"].append(self._vlsa_o_b[li].data_ptr())
            vsw["o_ws"].append(wsc(self._vlsa_alpha[li * 6 + 3]))
            vsw["fc1_w"].append(self._vlsa_fc1_w[li].data_ptr())
            vsw["fc1_b"].append(self._vlsa_fc1_b[li].data_ptr())
            vsw["fc1_ws"].append(wsc(self._vlsa_alpha[li * 6 + 4]))
            vsw["fc2_w"].append(self._vlsa_fc2_w[li].data_ptr())
            vsw["fc2_b"].append(self._vlsa_fc2_b[li].data_ptr())
            vsw["fc2_ws"].append(wsc(self._vlsa_alpha[li * 6 + 5]))
        vlsa_scales = {
            "act_qkv": adv(self._vlsa_act_qkv_dev),
            "act_o": adv(self._vlsa_act_o_dev),
            "act_fc1": adv(self._vlsa_act_fc1_dev),
            "act_fc2": adv(self._vlsa_act_fc2_dev)}
        vlsa_bufs = {"h": vlsa_h.data_ptr(), "xn": buf(Se, 2048).data_ptr(),
                     "xn_fp8": buf8(Se, 2048).data_ptr(),
                     "o_proj_out": buf(Se, 2048).data_ptr(),
                     "fc1_out": buf(Se, 8192).data_ptr(),
                     "fc1_fp8": buf8(Se, 8192).data_ptr()}
        vlsa_dims = {"T": Se, "D": 2048, "NH": 32, "HD": 64,
                     "ff_inner": 8192}
        P.vl_self_attn_forward(
            gemm=gemm, fvk=fvkm, bufs=vlsa_bufs,
            weights=vsw, scales_dev=vlsa_scales, dims=vlsa_dims, attn=attn,
            fused_epilogue=fused, alphas=vlsa_alphas)
        torch.cuda.synchronize()

        vit_dims = {"S": Sv, "D": 1024, "NH": 16, "HD": 64,
                    "ff_inner": 4096, "Sper_view": Sv // nv}
        vlln_dims = {"S": Se, "D": 2048}

        def _kbb_forward(stream=0):
            scell[0] = stream
            P.qwen3vl_vit_forward(
                gemm=gemm, fvk=fvkm, bufs=vit_bufs, weights=vw,
                scales_dev=vit_scales, dims=vit_dims, attn=attn,
                deepstack_taps=tap_layers, deepstack_capture=dcap,
                stream=stream, fused_epilogue=fused, alphas=vit_alphas)
            P.deepstack_merge_forward(
                gemm=gemm, fvk=fvkm, bufs=ds_bufs, weights=dsw,
                scales_dev=ds_scales, dims=ds_dims, stream=stream,
                fused_epilogue=fused, alphas=ds_alphas)
            for j in range(3):
                injb[j].zero_()
                injb[j].index_copy_(0, vis_idx, ds_out[j])
            P.qwen3vl_llm_forward(
                gemm=gemm, fvk=fvkm, bufs=llm_bufs, weights=lw,
                scales_dev=llm_scales, dims=llm_dims, attn=attn,
                stream=stream, fused_epilogue=fused)
            P.vlln_forward(
                gemm=gemm, fvk=fvkm, bufs=vlln_bufs,
                weights=vlln_weights, dims=vlln_dims, stream=stream)
            P.vl_self_attn_forward(
                gemm=gemm, fvk=fvkm, bufs=vlsa_bufs, weights=vsw,
                scales_dev=vlsa_scales, dims=vlsa_dims, attn=attn,
                stream=stream, fused_epilogue=fused, alphas=vlsa_alphas)
            return vlsa_h

        self._kbb_forward = _kbb_forward
        self._kbb_vit_h = vit_h
        self._kbb_llm_h = llm_h
        self._kbb_vlsa_h = vlsa_h
        return vlsa_h.unsqueeze(0)
