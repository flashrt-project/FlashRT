"""FlashRT -- GROOT N1.7 FP8 torch frontend for RTX (SM120 / SM89).

Framework-conforming FP8 path for GROOT N1.7 on RTX. The whole VLM backbone
(ViT / DeepStack / LLM / VL self-attn) runs through FlashRT FP8 kernels via the
SM120-safe descale pattern in :mod:`flash_rt.models.groot_n17.pipeline_rtx_fp8`
(``fp8_descale_fp16`` + separate bias/GELU — the fused cuBLAS FP8 epilogue is
unsupported on SM120). No PyTorch matmul touches the serving feature path.

Activation scales follow the FlashRT calibration convention (docs/calibration.md):
weight scales are baked at load; activation scales are calibrated once and
cached to disk (``~/.flash_rt/calibration/<hash>_n17_Se<N>.json``). On a warm
``set_prompt`` the cache is loaded and the backbone runs FP8 kernels only — the
torch reference shadow runs only on a cold cache miss (or an explicit
``calibrate()``), purely to extract activation amax, never as the inference
backbone.

The action head (state/action encoders, the 32-layer DiT, output proj, decoder)
is never FP8-quantized; its dtype is inherited from the base frontend. Two thin
classes pair the shared FP8 backbone with the two action-head dtypes:

  * :class:`GrootN17TorchFrontendRtxFP8`       — bf16 DiT (Thor-parity dtype)
  * :class:`GrootN17TorchFrontendRtxFP8FP16DiT` — fp16 DiT (RTX-native dtype)

Additive: this module only adds new classes; it does not modify the bf16 or
full-FP16 frontends, the calibration shadow, or any kernel.
"""

from __future__ import annotations

import warnings

import torch

from flash_rt.frontends.torch.groot_n17_rtx import GrootN17TorchFrontendRtx
from flash_rt.frontends.torch.groot_n17_rtx_fp16 import GrootN17TorchFrontendRtxFP16

_FP16 = torch.float16
_U8 = torch.uint8


class _GrootN17FP8BackboneMixin:
    """set_prompt + FP8 kernel backbone + disk-cached activation scales.

    Mixed in front of a DiT-bearing base frontend (bf16 or fp16). Overrides
    ``set_prompt`` to (1) resolve activation scales from the calibration cache
    or a one-time shadow calibration, then (2) produce ``_backbone_features``
    through FP8 kernels. Everything downstream (DiT cross-KV, graph capture,
    infer) is inherited from the base.
    """

    # ── Calibration cache (load side; save side is inherited from Thor) ──
    def _load_calibration_cache(self) -> "dict | None":
        import json
        from flash_rt.core.quant.calibrator import _checkpoint_hash, CACHE_DIR

        try:
            ckpt_hash = _checkpoint_hash(self.checkpoint_path)
        except Exception:
            return None
        cache_path = CACHE_DIR / f"{ckpt_hash}_n17_Se{self.Se}.json"
        if not cache_path.exists():
            return None
        try:
            with open(cache_path) as f:
                data = json.load(f)
        except Exception:
            return None
        if data.get("ckpt_hash") != ckpt_hash:
            return None
        if int(data.get("Se", -1)) != int(self.Se):
            return None
        if int(data.get("embodiment_id", -1)) != int(self._embodiment_id):
            return None
        return data

    @staticmethod
    def _cache_to_stage_dicts(data: dict):
        out_vit = {k: data[k] for k in
                   ("vit_act_qkv", "vit_act_o", "vit_act_fc1", "vit_act_fc2")}
        out_ds = {k: data[k] for k in
                  ("deepstack_act_fc1", "deepstack_act_fc2")}
        out_llm = {k: data[k] for k in
                   ("llm_act_qkv", "llm_act_o", "llm_act_gateup", "llm_act_down")}
        out_vlsa = {k: data[k] for k in
                    ("vlsa_act_qkv", "vlsa_act_o", "vlsa_act_fc1", "vlsa_act_fc2")}
        return out_vit, out_ds, out_llm, out_vlsa

    def _ensure_act_scales(self, aux: dict) -> None:
        """Populate ``self._<stage>_act_<point>_dev`` device scalars.

        Warm path (cache hit): bake from disk, no torch. Cold path (miss):
        run the torch shadow ONCE to extract amax, bake, and persist — this
        is one-time calibration, not the serving feature path.
        """
        cached = self._load_calibration_cache()
        if cached is not None:
            self._bake_calibration(*self._cache_to_stage_dicts(cached))
            # Warm path: the FP8 backbone runs on FP8 weights, so the fp16
            # shadow weights (loaded at construction for cold calibration)
            # are dead — free them.
            if hasattr(self, "_fp16_shadow_weights"):
                del self._fp16_shadow_weights
                torch.cuda.empty_cache()
            return

        from flash_rt.models.groot_n17 import calibration as cal

        if not hasattr(self, "_fp16_shadow_weights"):
            self._load_fp16_shadow_weights()
        device = self.device
        out_vit = cal.calibrate_vit(
            self, aux["pixel_features"].to(device).float(),
            self._vit_cos.float(), self._vit_sin.float(),
            num_views=self._num_vit_views)
        out_ds = cal.calibrate_deepstack(self, out_vit["deepstack_taps"])
        out_llm = cal.calibrate_llm(
            self, aux["llm_input_embeds"].to(device).float(),
            self._mrope_cos.float(), self._mrope_sin.float(),
            self._visual_pos_masks, out_ds["features"])
        out_vlsa = cal.calibrate_vlsa(self, out_llm["llm_final"])
        self._bake_calibration(out_vit, out_ds, out_llm, out_vlsa)
        self._save_calibration_cache(out_vit, out_ds, out_llm, out_vlsa)
        if hasattr(self, "_fp16_shadow_weights"):
            del self._fp16_shadow_weights
            torch.cuda.empty_cache()

    # ── set_prompt: activation-scale calibration + FP8 kernel backbone ──
    def set_prompt(self, *, aux: dict, prompt: str | None = None) -> None:
        from flash_rt.models.groot_n17.calibration import build_vit_rope_tables

        if hasattr(self, "_backbone_features"):
            raise RuntimeError(
                "set_prompt() after prompt init is not supported; construct a "
                "new frontend instance for a new prompt")

        device = self.device
        self._prompt = prompt
        self.Se = int(aux["llm_input_embeds"].shape[1])
        self._mrope_cos = aux["rope_cos"][0].to(device).half().contiguous()
        self._mrope_sin = aux["rope_sin"][0].to(device).half().contiguous()
        grid_thw = [tuple(int(x) for x in row) for row in aux["grid_thw"].tolist()]
        vit_cos, vit_sin = build_vit_rope_tables(
            grid_thw, head_dim=64, theta=10000.0, spatial_merge_size=2,
            device=device)
        self._vit_cos = vit_cos
        self._vit_sin = vit_sin
        self._num_vit_views = len(grid_thw)
        self._S_vit = sum(int(t * h * w) for t, h, w in grid_thw)
        self._visual_pos_masks = aux["visual_pos_masks"][0].to(device)

        # ── Activation scales: warm cache load (no torch) or one-time shadow ──
        self._ensure_act_scales(aux)

        # ── FP8 KERNEL backbone (no torch matmul on the feature path) ──
        self._backbone_features = self._run_kernel_backbone_fp8(aux).half()

        try:
            self._warmup_infer()
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"set_prompt warmup failed (non-fatal): {e!r}")

    # ── FP8 kernel backbone: ViT → DeepStack → LLM → vlln → VL-self-attn ──
    def _run_kernel_backbone_fp8(self, aux: dict) -> "torch.Tensor":
        import flash_rt.flash_rt_kernels as fvk
        from flash_rt.models.groot_n17 import pipeline_rtx_fp8 as P
        from flash_rt.hardware.rtx.attn_backend_groot_n17_backbone import (
            RtxGrootN17BackboneAttn,
        )

        if not hasattr(self, "_gemm"):
            self._fvk = fvk
            self._gemm = fvk.GemmRunner()
        gemm, fvkm = self._gemm, self._fvk
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

        attn = RtxGrootN17BackboneAttn(
            num_vit_views=nv, vit_seq=Sv, llm_seq=Se, vl_self_attn_seq=Se,
            device=dev)
        self._kbb_attn = attn

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
            "fc2_w", "fc2_b", "q_ws", "k_ws", "v_ws", "o_ws", "fc1_ws", "fc2_ws")}
        vw["cos"] = self._vit_cos.data_ptr()
        vw["sin"] = self._vit_sin.data_ptr()
        for li in range(24):
            qkv = self._vit_qkv_w[li]               # fp8 (1024, 3072) [K, 3N]
            b = self._vit_qkv_b[li]                  # (3072,)
            q = K(qkv[:, :1024].contiguous()); kk = K(qkv[:, 1024:2048].contiguous())
            v = K(qkv[:, 2048:].contiguous())
            qb = K(b[:1024].contiguous()); kb = K(b[1024:2048].contiguous())
            vb = K(b[2048:].contiguous())
            qkv_ws = wsc(self._vit_alpha[li * 4 + 0])
            vw["norm1_w"].append(self._vit_ln1_w[li].data_ptr())
            vw["norm1_b"].append(self._vit_ln1_b[li].data_ptr())
            vw["norm2_w"].append(self._vit_ln2_w[li].data_ptr())
            vw["norm2_b"].append(self._vit_ln2_b[li].data_ptr())
            vw["q_w"].append(q.data_ptr()); vw["q_b"].append(qb.data_ptr())
            vw["k_w"].append(kk.data_ptr()); vw["k_b"].append(kb.data_ptr())
            vw["v_w"].append(v.data_ptr()); vw["v_b"].append(vb.data_ptr())
            vw["q_ws"].append(qkv_ws); vw["k_ws"].append(qkv_ws); vw["v_ws"].append(qkv_ws)
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
            "act_qkv": adv(self._vit_act_qkv_dev), "act_o": adv(self._vit_act_o_dev),
            "act_fc1": adv(self._vit_act_fc1_dev), "act_fc2": adv(self._vit_act_fc2_dev)}

        tap_layers = (5, 11, 17)
        tap_bufs = {l: buf(Sv, 1024) for l in tap_layers}

        def mk_cb(l):
            def cb(h_ptr):
                fvkm.gpu_copy(tap_bufs[l].data_ptr(), int(h_ptr), Sv * 1024 * 2, 0)
            return cb
        dcap = [mk_cb(l) for l in tap_layers]

        P.qwen3vl_vit_forward(
            gemm=gemm, fvk=fvkm, bufs=vit_bufs, weights=vw, scales_dev=vit_scales,
            dims={"S": Sv, "D": 1024, "NH": 16, "HD": 64,
                  "ff_inner": 4096, "Sper_view": Sv // nv},
            attn=attn, deepstack_taps=tap_layers, deepstack_capture=dcap)

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
        P.deepstack_merge_forward(
            gemm=gemm, fvk=fvkm,
            bufs={"in": [tap_bufs[l].data_ptr() for l in tap_layers],
                  "ln_out": buf(Nout, 4096).data_ptr(),
                  "fp8_scratch": buf8(Nout, 4096).data_ptr(),
                  "fc1_out": buf(Nout, 4096).data_ptr(),
                  "out": [t.data_ptr() for t in ds_out]},
            weights=dsw, scales_dev=ds_scales,
            dims={"Nin": Sv, "Din": 1024, "Nout": Nout, "Dmid": 4096, "Dout": 2048})

        # DeepStack inject buffers (S, D) — zero except visual positions.
        mask = self._visual_pos_masks
        inject = [0] * 16
        for j in range(3):
            ib = K(torch.zeros(Se, 2048, dtype=_FP16, device=dev))
            ib[mask] = ds_out[j]
            inject[j] = ib.data_ptr()

        # ═══ LLM (16L, causal, GQA) ═══
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
            qkv = self._llm_qkv_w[li]               # fp8 (2048, 4096) [K, NHQ·HD+2·NHKV·HD]
            q = K(qkv[:, :2048].contiguous())
            kk = K(qkv[:, 2048:3072].contiguous())
            v = K(qkv[:, 3072:4096].contiguous())
            qkv_ws = wsc(self._llm_alpha[li * 5 + 0])
            lw["in_ln_w"].append(self._llm_input_ln_w[li].data_ptr())
            lw["post_ln_w"].append(self._llm_post_ln_w[li].data_ptr())
            lw["q_norm_w"].append(self._llm_q_norm_w[li].data_ptr())
            lw["k_norm_w"].append(self._llm_k_norm_w[li].data_ptr())
            lw["q_w"].append(q.data_ptr()); lw["k_w"].append(kk.data_ptr())
            lw["v_w"].append(v.data_ptr())
            lw["q_ws"].append(qkv_ws); lw["k_ws"].append(qkv_ws); lw["v_ws"].append(qkv_ws)
            lw["o_w"].append(self._llm_o_w[li].data_ptr())
            lw["o_ws"].append(wsc(self._llm_alpha[li * 5 + 1]))
            lw["gate_w"].append(self._llm_gate_w[li].data_ptr())
            lw["gate_ws"].append(wsc(self._llm_alpha[li * 5 + 2]))
            lw["up_w"].append(self._llm_up_w[li].data_ptr())
            lw["up_ws"].append(wsc(self._llm_alpha[li * 5 + 3]))
            lw["down_w"].append(self._llm_down_w[li].data_ptr())
            lw["down_ws"].append(wsc(self._llm_alpha[li * 5 + 4]))
        llm_scales = {
            "act_qkv": adv(self._llm_act_qkv_dev), "act_o": adv(self._llm_act_o_dev),
            "act_gateup": adv(self._llm_act_gateup_dev),
            "act_down": adv(self._llm_act_down_dev)}
        slots = attn.get_slot_ptrs("llm")
        llm_bufs = {
            "h": llm_h.data_ptr(), "xn": buf(Se, 2048).data_ptr(),
            "xn_fp8": buf8(Se, 2048).data_ptr(),
            "Q": slots["Q"], "K": buf(Se, 1024).data_ptr(),
            "V": buf(Se, 1024).data_ptr(),
            "K_exp": slots["K"], "V_exp": slots["V"],
            "o_proj_out": buf(Se, 2048).data_ptr(),
            "gate_out": buf(Se, 6144).data_ptr(),
            "up_out": buf(Se, 6144).data_ptr(),
            "gu_fp8": buf8(Se, 6144).data_ptr()}
        P.qwen3vl_llm_forward(
            gemm=gemm, fvk=fvkm, bufs=llm_bufs, weights=lw, scales_dev=llm_scales,
            dims={"S": Se, "D": 2048, "NHQ": 16, "NHKV": 8, "HD": 128, "FF": 6144},
            attn=attn)

        # ═══ vlln + VL self-attn (4L) ═══
        vlsa_h = buf(Se, 2048)
        P.vlln_forward(
            gemm=gemm, fvk=fvkm,
            bufs={"x": llm_h.data_ptr(), "out": vlsa_h.data_ptr()},
            weights={"vlln_w": self._vlln_w.data_ptr(),
                     "vlln_b": self._vlln_b.data_ptr()},
            dims={"S": Se, "D": 2048})
        vsw = {k: [] for k in (
            "norm1_w", "norm1_b", "norm3_w", "norm3_b", "q_w", "q_b",
            "k_w", "k_b", "v_w", "v_b", "o_w", "o_b", "fc1_w", "fc1_b",
            "fc2_w", "fc2_b", "q_ws", "k_ws", "v_ws", "o_ws", "fc1_ws", "fc2_ws")}
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
            "act_qkv": adv(self._vlsa_act_qkv_dev), "act_o": adv(self._vlsa_act_o_dev),
            "act_fc1": adv(self._vlsa_act_fc1_dev), "act_fc2": adv(self._vlsa_act_fc2_dev)}
        P.vl_self_attn_forward(
            gemm=gemm, fvk=fvkm,
            bufs={"h": vlsa_h.data_ptr(), "xn": buf(Se, 2048).data_ptr(),
                  "xn_fp8": buf8(Se, 2048).data_ptr(),
                  "o_proj_out": buf(Se, 2048).data_ptr(),
                  "fc1_out": buf(Se, 8192).data_ptr(),
                  "fc1_fp8": buf8(Se, 8192).data_ptr()},
            weights=vsw, scales_dev=vlsa_scales,
            dims={"T": Se, "D": 2048, "NH": 32, "HD": 64, "ff_inner": 8192},
            attn=attn)
        torch.cuda.synchronize()
        return vlsa_h.unsqueeze(0)


class GrootN17TorchFrontendRtxFP8(_GrootN17FP8BackboneMixin, GrootN17TorchFrontendRtx):
    """N1.7 RTX FP8 frontend with a bf16 action head (Thor-parity dtype)."""

    _DIT_FP8_IMPL = "sm120_safe"


class GrootN17TorchFrontendRtxFP8FP16DiT(
        _GrootN17FP8BackboneMixin, GrootN17TorchFrontendRtxFP16):
    """N1.7 RTX FP8 frontend with a full-FP16 action head (RTX-native dtype)."""
