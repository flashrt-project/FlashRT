"""Pi05TorchFrontendThor with NVFP4 encoder-FFN subset support (Phase 4.3).

STRICTLY ADDITIVE. Does not modify pi05_thor.py. Subclasses Pi05TorchFrontendThor
and overrides only the graph-capture method to route encoder forward through
shared_primitives_fp4.encoder_forward_with_fp4_subset when FP4 is enabled.

Usage:
    pipe = Pi05TorchFrontendThorFP4(
        "/path/to/checkpoint",
        num_views=2,
        use_fp4_encoder_ffn=True,    # default False → bit-identical to base
        fp4_layers=(7, 8, 9),         # precision-safe middle encoder FFN
    )
    pipe.set_prompt("pick up the red cup")
    actions = pipe.infer(obs)["actions"]
"""
from __future__ import annotations

import logging
from typing import Iterable

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
from flash_rt.hardware.thor.shared_primitives import encoder_forward
from flash_rt.models.pi05.pipeline_thor import decoder_forward
from flash_rt.hardware.thor.shared_primitives_fp4 import encoder_forward_with_fp4_subset
from flash_rt.executors.fp4_utils import (
    FP4ActScratch, pick_variant, quant_weight_nvfp4, quant_weight_nvfp4_inplace,
)

try:
    import flash_rt.flash_rt_fp4 as fvk_fp4
    _HAS_FP4 = fvk_fp4.has_nvfp4()
except Exception as _e:  # pragma: no cover
    fvk_fp4 = None
    _HAS_FP4 = False

logger = logging.getLogger(__name__)
fp16 = torch.float16


class Pi05TorchFrontendThorFP4(Pi05TorchFrontendThor):
    """Pi0.5 Thor frontend with optional NVFP4 encoder-FFN layers."""

    def __init__(self, checkpoint_dir, num_views: int = 2,
                 use_cuda_graph: bool = True, autotune: int = 3,
                 *,
                 use_fp4_encoder_ffn: bool = False,
                 fp4_layers: Iterable[int] = (7, 8, 9),
                 use_awq: bool = False,
                 awq_alpha: float = 0.5,
                 awq_calib_iters: int = 8,
                 use_p1_split_gu: bool = False,
                 use_fp8: bool = True):
        # Base init (loads weights, allocates all FP8 buffers, etc.)
        super().__init__(checkpoint_dir, num_views=num_views,
                         use_cuda_graph=use_cuda_graph, autotune=autotune,
                         use_fp8=use_fp8)

        self.use_fp4_encoder_ffn = bool(use_fp4_encoder_ffn)
        self._fp4_layers = frozenset(fp4_layers) if self.use_fp4_encoder_ffn else frozenset()
        self.use_awq = bool(use_awq) and self.use_fp4_encoder_ffn
        self.awq_alpha = float(awq_alpha)
        self.awq_calib_iters = int(awq_calib_iters)
        # P1: split-GU 2-GEMM path (eliminates F4 v2; -2.9ms expected)
        self.use_p1_split_gu = bool(use_p1_split_gu) and self.use_fp4_encoder_ffn

        if self._fp4_layers:
            if not _HAS_FP4:
                raise RuntimeError(
                    "use_fp4_encoder_ffn=True but flash_rt_fp4 not available. "
                    "Ensure flash_rt_fp4.so is built with NVFP4 support.")
            self._prepare_fp4_encoder()
            logger.info("Pi05 FP4 enabled on encoder layers: %s  (AWQ=%s)",
                        sorted(self._fp4_layers), self.use_awq)

    # -------------------------------------------------------------------
    # Calibration override — block multi-sample on active FP4 layers
    # -------------------------------------------------------------------

    def _calibrate_multi_frame(self, obs_list, *, percentile: float, verbose: bool):
        """Pi0.5 FP4 multi-sample (N>=2) calibration.

        FP4 has three independent scale storages that all need to be
        consistent with each other and with the captured CUDA graph:

          * FP8 act scales (``_enc_calib_scales[72]``)         - FP4 layers'
                                                                QKV/O path +
                                                                every non-FP4
                                                                layer.
          * NVFP4 block scales (baked into ``_fp4_weights[l]['*']['sfb']``)
          * AWQ per-input-channel inv_s (``_awq_inv_s_gu[l]``,
            ``_awq_inv_s_dn[l]``)                              - only when
                                                                AWQ is on.

        The single-frame ``_recalibrate_with_real_data`` updates all three
        in the right order; a naive super() call only updates the FP8
        side and the captured graph ends up referencing stale AWQ +
        stale NVFP4 weights (measured cos_vs_prod collapsed to 0.12).

        Fix: do it in two phases.

          Phase 1 (FP8 + graph recapture, delegated to the base class)
            super()._calibrate_multi_frame(...)
              - per-sample FP8 amax on encoder + decoder
              - percentile reduce, upload, _enc_alpha_host recompute
              - _capture_enc_ae_graph() via the FP4-aware override
            Graph is now captured with fresh FP8 scales but the AWQ
            inv_s + NVFP4 weight buffers still hold whatever values
            _prepare_fp4_encoder wrote at construction time.

          Phase 2 (AWQ + NVFP4 re-quant, in place)
            For each obs: upload images + SigLIP.replay() so _enc_x
              reflects this sample, then _collect_awq_activation_amax()
              returns per-channel Gate+Up and Down amax for every active
              FP4 layer using the now-fresh FP8 scales.
            Percentile-reduce across samples.
            _requant_fp4_weights_with_awq() writes new inv_s and re-
              quantizes FP4 weights in place — the captured graph's
              pointers do not change, so the next replay picks up the
              new values without a second graph capture.

        If AWQ is disabled (``self.use_awq == False``) there is no
        per-channel scale to refit and the function reduces to Phase 1.
        """
        # No FP4 layers active -> behave exactly like the base class.
        if not self._fp4_layers:
            return super()._calibrate_multi_frame(
                obs_list, percentile=percentile, verbose=verbose)

        import numpy as np
        from flash_rt.core.calibration import (
            accumulate_amax,
            check_scale_ceiling,
            format_summary,
            summarize_amax_dispersion,
        )

        n = len(obs_list)
        logger.info(
            "Pi0.5 FP4 Thor: calibrating FP8 + AWQ across %d real samples "
            "(percentile=%.2f)...", n, percentile)

        # === Phase 1: FP8 scale percentile reduction + graph recapture ===
        # Delegates to Pi05TorchFrontendThor._calibrate_multi_frame. That
        # method's final step calls self._capture_enc_ae_graph() which
        # dispatches to the FP4-aware override in this class — the graph
        # is captured with fresh FP8 scales and the current (stale) AWQ
        # inv_s + NVFP4 weight buffers. Phase 2 below will update those
        # buffers in place without re-capturing the graph.
        super()._calibrate_multi_frame(
            obs_list, percentile=percentile, verbose=verbose)

        if not self.use_awq:
            logger.info(
                "Pi0.5 FP4 Thor: AWQ disabled, skipping Phase 2 AWQ "
                "refit (N=%d multi-frame limited to FP8 scales only)", n)
            return

        # === Phase 2: per-sample AWQ amax collection ===
        # Runs with the FINAL (post-Phase-1) FP8 scales active, which is
        # important because _collect_awq_activation_amax runs a hand-
        # rolled FP8 encoder forward that reads self._enc_calib_scales
        # and self._enc_alpha_host. Using stale Phase-0 scales here would
        # bias the AWQ distribution.
        nv = self.num_views
        De = self.De; He = self.He
        per_sample_gu: dict = {l: [] for l in self._fp4_layers}
        per_sample_dn: dict = {l: [] for l in self._fp4_layers}

        for i, obs in enumerate(obs_list):
            if 'images' in obs:
                img_list = obs['images']
            else:
                img_list = [obs['image']]
                if nv >= 2:
                    img_list.append(obs.get('wrist_image', obs['image']))
                if nv >= 3:
                    img_list.append(obs.get('wrist_image_right',
                                            img_list[-1]))

            def _to_np16(im):
                if isinstance(im, torch.Tensor):
                    if im.dtype == torch.uint8 or (
                            im.is_floating_point() and
                            torch.max(im).item() > 1.5):
                        im = (im.float() / 127.5 - 1.0)
                    return im.to(dtype=fp16).cpu().numpy()
                if im.dtype == np.float16:
                    return im
                return (im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)

            images_np = np.stack([_to_np16(im) for im in img_list[:nv]])
            self._img_buf.upload(images_np)
            self._siglip_graph.replay()
            torch.cuda.synchronize()

            # _collect_awq_activation_amax reads self._enc_x (just
            # written by the SigLIP replay above) and returns per-
            # channel activation amax at each FP4 layer's Gate+Up input
            # (shape [De]) and Down input (shape [He]).
            act_gu, act_dn = self._collect_awq_activation_amax()
            for l in self._fp4_layers:
                per_sample_gu[l].append(
                    act_gu[l].detach().cpu().numpy().astype(np.float32))
                per_sample_dn[l].append(
                    act_dn[l].detach().cpu().numpy().astype(np.float32))

            if verbose and (i + 1) % max(1, n // 10) == 0:
                logger.info("  AWQ sample %d/%d", i + 1, n)

        # === Percentile reduce per FP4 layer ===
        final_gu: dict = {}
        final_dn: dict = {}
        for l in self._fp4_layers:
            final_gu[l] = accumulate_amax(per_sample_gu[l], percentile=percentile)
            final_dn[l] = accumulate_amax(per_sample_dn[l], percentile=percentile)
            if verbose:
                logger.info(
                    "  layer %d: GU %s  Down %s", l,
                    format_summary(summarize_amax_dispersion(
                        per_sample_gu[l], final_gu[l])),
                    format_summary(summarize_amax_dispersion(
                        per_sample_dn[l], final_dn[l])))

        # Optional outlier warning keyed off the Down-input scale (which
        # is usually the widest-spread across layers).
        check_scale_ceiling(
            {f"L{l}_dn_max": float(final_dn[l].max()) for l in self._fp4_layers},
            label=f"pi05_thor_fp4_awq_N{n}")

        # Hand the reduced per-channel amax dicts to the existing single-
        # frame AWQ requant routine. It updates inv_s + NVFP4 packed/sfb
        # IN PLACE at the same pointer addresses the captured graph
        # already references, so no graph recapture is needed.
        act_gu_dev = {l: torch.from_numpy(final_gu[l]).cuda()
                      for l in self._fp4_layers}
        act_dn_dev = {l: torch.from_numpy(final_dn[l]).cuda()
                      for l in self._fp4_layers}
        self._requant_fp4_weights_with_awq(act_gu_dev, act_dn_dev)
        self._awq_calibrated = True

        logger.info(
            "Pi0.5 FP4 Thor multi-frame calibration complete "
            "(N=%d, percentile=%.2f, AWQ refit done in place)",
            n, percentile)

    # -------------------------------------------------------------------
    # Weight quantization (Se-independent, done once at __init__)
    # -------------------------------------------------------------------
    def _prepare_fp4_encoder(self):
        De = self.De
        He = self.He

        # NVIDIA-style FP4-native weight path: load original fp16 weights from
        # safetensors and apply FusedGateUp (gate/up concat with norm_fuse)
        # in fp16, then NVFP4-quantize directly. Bypasses the FP8 dequant step
        # and its double-lossy precision loss.
        # Falls back to FP8→fp16 dequant if safetensors load fails.
        self._fp4_weights = {}
        try:
            self._load_fp4_weights_from_safetensors()
            logger.info("FP4 weights loaded via fp16-native path (no FP8 intermediate)")
        except Exception as e:
            logger.warning("FP4-native weight load failed (%s); falling back to "
                           "FP8→fp16 dequant path", e)
            self._load_fp4_weights_from_fp8_dequant()

        # Variant selection (empirically tuned @ Se=968, see Step B test).
        self._fp4_variant_gu = pick_variant(2 * He, De)
        self._fp4_variant_dn = pick_variant(De, He)

        self._fp4_scratch_dict = None
        self._fp4_scratch_Se = -1

    # ------------------------------------------------------------------
    # FP4-native weight loading (fp16 → NVFP4 directly, no FP8 intermediate)
    # ------------------------------------------------------------------
    def _load_fp4_weights_from_safetensors(self):
        """Load fp16 encoder weights directly from safetensors, apply
        FusedGateUp (with post_attention_layernorm fuse) and NVFP4-quantize.
        Mirrors paligemma_encoder_block's spec but stays in fp16 end-to-end.

        When self.use_awq=True, also applies AWQ-style per-input-channel
        weight scaling (W'[:, k] = W[:, k] * s[k]) before NVFP4 quantization,
        with matching inverse scale stored for runtime activation scaling.
        """
        from safetensors import safe_open

        De = self.De; He = self.He
        model_root = "paligemma_with_expert.paligemma.model.language_model.layers"

        self._awq_inv_s_gu = {} if self.use_awq else None
        self._awq_inv_s_dn = {} if self.use_awq else None

        from flash_rt.executors.torch_weights import _autodetect_strip_prefix
        with safe_open(self._checkpoint_path, framework='pt', device='cuda') as sf:
            _strip = _autodetect_strip_prefix(set(sf.keys()))
            def get(k):
                return sf.get_tensor((_strip + k) if _strip else k)

            for l in self._fp4_layers:
                p = f"{model_root}.{l}"
                # FusedGateUp: ff = 1 + post_attn_layernorm; gw/uw = gate/up * ff
                ff = 1.0 + get(f"{p}.post_attention_layernorm.weight").float()
                ff_u = ff.unsqueeze(0)
                gw = (get(f"{p}.mlp.gate_proj.weight").float() * ff_u).to(fp16)
                uw = (get(f"{p}.mlp.up_proj.weight").float()   * ff_u).to(fp16)
                gu_fp16 = torch.cat([gw, uw], dim=0).contiguous()  # [2H, D]

                # Down: ToFp16 only (no fuse)
                d_fp16 = get(f"{p}.mlp.down_proj.weight").to(fp16).contiguous()  # [D, H]

                assert gu_fp16.shape == (2 * He, De), \
                    f"gate_up shape {gu_fp16.shape} != {(2*He, De)}"
                assert d_fp16.shape == (De, He), \
                    f"down shape {d_fp16.shape} != {(De, He)}"

                if self.use_awq:
                    gu_fp16, inv_s_gu = self._awq_scale_weight(gu_fp16)  # [2H, D]
                    d_fp16,  inv_s_dn = self._awq_scale_weight(d_fp16)   # [D,  H]
                    self._awq_inv_s_gu[l] = inv_s_gu  # fp16 [D]
                    self._awq_inv_s_dn[l] = inv_s_dn  # fp16 [H]

                self._fp4_weights[l] = {
                    'gate_up': quant_weight_nvfp4(gu_fp16),
                    'down':    quant_weight_nvfp4(d_fp16),
                }

                # P1: also store gate / up separately for the 2-GEMM split path.
                # gu_fp16 shape is [2H, D] = [gate || up] concatenated along N.
                if self.use_p1_split_gu:
                    g_fp16 = gu_fp16[:He, :].contiguous()
                    u_fp16 = gu_fp16[He:, :].contiguous()
                    self._fp4_weights[l]['gate'] = quant_weight_nvfp4(g_fp16)
                    self._fp4_weights[l]['up']   = quant_weight_nvfp4(u_fp16)

    def _awq_scale_weight(self, W: torch.Tensor,
                           activation_amax: torch.Tensor | None = None,
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """AWQ per-input-channel (K axis) pre-scale.

        If ``activation_amax`` is provided (shape [K], fp32), uses AWQ proper:
            s[k] = (activation_amax[k] / activation_amax.mean())^alpha
        Otherwise falls back to weight-only amax.

        Returns (W' = W * s broadcast along N, inv_s = 1/s).
        Activation must be scaled by inv_s before the FP4 GEMM:
            Y = (X * inv_s) @ (W * s).T = X @ W.T          (math preserved)
        """
        if activation_amax is not None:
            a = activation_amax.float().clamp(min=1e-6)
        else:
            a = W.abs().amax(dim=0).float().clamp(min=1e-6)
        s = (a / a.mean()).pow(self.awq_alpha).clamp(min=0.25, max=4.0)
        inv_s = (1.0 / s).to(fp16).contiguous()  # [K]
        W_scaled = (W.float() * s.unsqueeze(0)).to(fp16).contiguous()  # [N, K]
        return W_scaled, inv_s

    def _requant_fp4_weights_with_awq(self, act_amax_gu: dict, act_amax_dn: dict):
        """Re-quantize FP4 weights using activation-aware AWQ scales.

        Called after the first real-data forward pass collected per-channel
        activation amax at each FP4 layer's Gate+Up and Down inputs.

        Updates the packed/sfb buffers of ``self._fp4_weights[l]`` IN-PLACE
        (same pointer addresses) and ``self._awq_inv_s_{gu,dn}[l]`` IN-PLACE.
        This keeps the captured CUDA Graph valid — no recapture needed.
        """
        from safetensors import safe_open
        from flash_rt.executors.torch_weights import _autodetect_strip_prefix
        De = self.De; He = self.He
        model_root = "paligemma_with_expert.paligemma.model.language_model.layers"
        with safe_open(self._checkpoint_path, framework='pt', device='cuda') as sf:
            _strip = _autodetect_strip_prefix(set(sf.keys()))
            def get(k): return sf.get_tensor((_strip + k) if _strip else k)
            for l in self._fp4_layers:
                p = f"{model_root}.{l}"
                ff = 1.0 + get(f"{p}.post_attention_layernorm.weight").float()
                ff_u = ff.unsqueeze(0)
                gw = (get(f"{p}.mlp.gate_proj.weight").float() * ff_u).to(fp16)
                uw = (get(f"{p}.mlp.up_proj.weight").float()   * ff_u).to(fp16)
                gu_fp16 = torch.cat([gw, uw], dim=0).contiguous()
                d_fp16 = get(f"{p}.mlp.down_proj.weight").to(fp16).contiguous()

                gu_scaled, inv_s_gu = self._awq_scale_weight(gu_fp16, act_amax_gu[l])
                d_scaled,  inv_s_dn = self._awq_scale_weight(d_fp16,  act_amax_dn[l])

                # Update inv_s and packed/sfb buffers in place — pointer
                # addresses unchanged, so the captured graph remains valid.
                self._awq_inv_s_gu[l].copy_(inv_s_gu)
                self._awq_inv_s_dn[l].copy_(inv_s_dn)
                quant_weight_nvfp4_inplace(gu_scaled, self._fp4_weights[l]['gate_up'])
                quant_weight_nvfp4_inplace(d_scaled,  self._fp4_weights[l]['down'])

    def _collect_awq_activation_amax(self):
        """Run calibration forward with hook that snapshots fp16 activations
        at FP4 layers' Gate+Up and Down inputs; return per-channel amax dicts."""
        Se = self.Se; De = self.De; He = self.He; NHe = self.NHe; HDe = self.HDe
        Le = self.Le; total_keys = self.total_keys
        act_gu = {l: torch.zeros(De, dtype=torch.float32, device='cuda') for l in self._fp4_layers}
        act_dn = {l: torch.zeros(He, dtype=torch.float32, device='cuda') for l in self._fp4_layers}

        # Snapshot scratch buffers (one-per-FP4-layer to avoid re-alloc)
        x_snap = {l: torch.empty(Se, De, dtype=fp16, device='cuda') for l in self._fp4_layers}
        h_snap = {l: torch.empty(Se, He, dtype=fp16, device='cuda') for l in self._fp4_layers}

        # Run one encoder pass hand-rolled: use existing kernels step by step.
        # Uses FP8 path for every layer (no FP4 applied here) so activation
        # statistics reflect the production distribution prior to FP4.
        import math
        Q_dim = NHe * HDe
        K_dim = HDe
        attn_scale = 1.0 / math.sqrt(float(HDe))

        x = self._enc_x            # fp16 [Se, De]   — current SigLIP output
        x_fp8 = self._enc_x_fp8
        qkv = self._enc_qkv_buf
        attn_out = self._enc_attn
        o_fp8 = self._enc_o_fp8
        gate = self._enc_gate
        hid_fp8 = self._enc_hid_fp8
        fg = self._enc_fg
        act_scales = self._enc_calib_scales.data_ptr()
        alpha_host = self._enc_alpha_host
        rope = self._enc_rope.data_ptr()
        Kc = self._Kc.reshape(-1).data_ptr()
        Vc = self._Vc.reshape(-1).data_ptr()
        self._Kc.zero_(); self._Vc.zero_()

        for l in range(Le):
            last = (l == Le - 1)
            as_qkv = act_scales + (l * 4 + 0) * 4
            as_o   = act_scales + (l * 4 + 1) * 4
            as_gu  = act_scales + (l * 4 + 2) * 4
            as_d   = act_scales + (l * 4 + 3) * 4

            fvk.rms_norm_fp8_noweight_fp16(
                x.data_ptr(), x_fp8.data_ptr(), Se, De, as_qkv, 0)
            fvk.cutlass_fp8_sq(
                x_fp8.data_ptr(), self._enc_qkv_w[l].data_ptr(), qkv.data_ptr(),
                Se, 2560, De, alpha_host[l*4+0], 0.0, 0)
            kv_elem_off = l * total_keys * HDe
            fvk.qkv_split_rope_kvcache_fp16(
                qkv.data_ptr(), rope, attn_out.data_ptr(), Kc, Vc,
                Se, Q_dim, K_dim, HDe, 2560, kv_elem_off, HDe, 0)
            if last:
                continue
            if self._attn is not None:
                self._attn.run("encoder", l, q_seq=Se, stream=0)
            else:
                fvk.attention_qkv_fp16(
                    self._ctx, attn_out.data_ptr(),
                    Kc + kv_elem_off * 2, Vc + kv_elem_off * 2,
                    self._enc_logits.data_ptr(), attn_out.data_ptr(),
                    Se, Se, NHe, HDe, attn_scale, 0)
            fvk.quantize_fp8_static_fp16(
                attn_out.data_ptr(), o_fp8.data_ptr(), as_o, Se * De, 0)
            fvk.cutlass_fp8_sq(
                o_fp8.data_ptr(), self._enc_o_w[l].data_ptr(), fg.data_ptr(),
                Se, De, De, alpha_host[l*4+1], 0.0, 0)

            if l in self._fp4_layers:
                # snapshot fp16 Gate+Up input
                fvk_fp4.residual_add_rms_norm_noweight_fp16(
                    x.data_ptr(), fg.data_ptr(),
                    x_snap[l].data_ptr(), Se, De, 0)
                # also update the production residual via fp8 path (for subsequent layers)
                fvk.residual_add_rms_norm_fp8_noweight_fp16(
                    x.data_ptr(), fg.data_ptr(), x_fp8.data_ptr(),
                    Se, De, as_gu, 0)
            else:
                fvk.residual_add_rms_norm_fp8_noweight_fp16(
                    x.data_ptr(), fg.data_ptr(), x_fp8.data_ptr(),
                    Se, De, as_gu, 0)

            fvk.cutlass_fp8_t1(
                x_fp8.data_ptr(), self._enc_gu_w[l].data_ptr(), gate.data_ptr(),
                Se, He * 2, De, alpha_host[l*4+2], 0.0, 0)

            if l in self._fp4_layers:
                fvk.gate_geglu_merged_fp16(
                    gate.data_ptr(), h_snap[l].data_ptr(), Se, He, 0)
                fvk.gate_geglu_merged_fp8_fp16(
                    gate.data_ptr(), hid_fp8.data_ptr(), Se, He, as_d, 0)
            else:
                fvk.gate_geglu_merged_fp8_fp16(
                    gate.data_ptr(), hid_fp8.data_ptr(), Se, He, as_d, 0)

            fvk.cutlass_fp8_wide(
                hid_fp8.data_ptr(), self._enc_d_w[l].data_ptr(), fg.data_ptr(),
                Se, De, He, alpha_host[l*4+3], 0.0, 0)
            as_next = act_scales + ((l+1)*4 + 0) * 4
            fvk.residual_add_rms_norm_fp8_noweight_fp16(
                x.data_ptr(), fg.data_ptr(), x_fp8.data_ptr(),
                Se, De, as_next, 0)

        torch.cuda.synchronize()
        # Compute per-channel amax from snapshots
        for l in self._fp4_layers:
            act_gu[l] = x_snap[l].abs().float().amax(dim=0)  # [De]
            act_dn[l] = h_snap[l].abs().float().amax(dim=0)  # [He]
        # Free snapshots
        del x_snap, h_snap
        torch.cuda.empty_cache()
        return act_gu, act_dn

    def _load_fp4_weights_from_fp8_dequant(self):
        """Fallback path: dequant existing FP8 weights to fp16, then NVFP4.
        Double-lossy but does not require safetensors re-read."""
        De = self.De; He = self.He
        scales = self._enc_w_scales  # list of 72 fp32 floats (q,o,gu,d * 18)
        for l in self._fp4_layers:
            gu_fp8 = self._enc_gu_w[l]
            d_fp8  = self._enc_d_w[l]
            gu_scale = float(scales[l * 4 + 2])
            d_scale  = float(scales[l * 4 + 3])
            gu_fp16 = (gu_fp8.view(torch.float8_e4m3fn).float() * gu_scale).to(fp16).contiguous()
            d_fp16  = (d_fp8.view(torch.float8_e4m3fn).float()  * d_scale ).to(fp16).contiguous()
            self._fp4_weights[l] = {
                'gate_up': quant_weight_nvfp4(gu_fp16),
                'down':    quant_weight_nvfp4(d_fp16),
            }

    def _alloc_fp4_scratch_for_Se(self, Se: int):
        """Allocate/resize FP4 scratch buffers for current encoder seq length."""
        if self._fp4_scratch_Se == Se and self._fp4_scratch_dict is not None:
            return
        De = self.De; He = self.He
        self._fp4_gu_scratch = FP4ActScratch(Se, De, device='cuda')
        self._fp4_dn_scratch = FP4ActScratch(Se, He, device='cuda')
        self._fp4_x_normed   = torch.empty(Se, De,     dtype=fp16, device='cuda')
        self._fp4_gate_out   = torch.empty(Se, 2 * He, dtype=fp16, device='cuda')
        self._fp4_hid_fp16   = torch.empty(Se, He,     dtype=fp16, device='cuda')
        self._fp4_fg_fp16    = torch.empty(Se, De,     dtype=fp16, device='cuda')
        # P1: split-GU intermediate FP4 buffers (gate / up between fp4out GEMMs
        # and geglu_two combiner). Each is [Se, He/2] packed + SFA.
        if self.use_p1_split_gu:
            from flash_rt.executors.fp4_utils import FP4Buffer
            self._fp4_p1_gate = FP4Buffer(Se, He, device='cuda')
            self._fp4_p1_up   = FP4Buffer(Se, He, device='cuda')
        self._fp4_scratch_dict = {
            'gu_act':     self._fp4_gu_scratch,
            'down_act':   self._fp4_dn_scratch,
            'x_normed':   self._fp4_x_normed.data_ptr(),
            'gate_out':   self._fp4_gate_out.data_ptr(),
            'hid_fp16':   self._fp4_hid_fp16.data_ptr(),
            'fg_fp16':    self._fp4_fg_fp16.data_ptr(),
            'variant_gu': self._fp4_variant_gu,
            'variant_dn': self._fp4_variant_dn,
            # AWQ per-layer inv-scale pointers (dicts of layer_idx → int pointer)
            'awq_inv_s_gu': (
                {l: self._awq_inv_s_gu[l].data_ptr() for l in self._fp4_layers}
                if self.use_awq else None),
            'awq_inv_s_dn': (
                {l: self._awq_inv_s_dn[l].data_ptr() for l in self._fp4_layers}
                if self.use_awq else None),
        }
        if self.use_p1_split_gu:
            self._fp4_scratch_dict['p1_gate_p4']  = self._fp4_p1_gate.packed.data_ptr()
            self._fp4_scratch_dict['p1_gate_sfa'] = self._fp4_p1_gate.sfa.data_ptr()
            self._fp4_scratch_dict['p1_up_p4']    = self._fp4_p1_up.packed.data_ptr()
            self._fp4_scratch_dict['p1_up_sfa']   = self._fp4_p1_up.sfa.data_ptr()
        self._fp4_scratch_Se = Se

    # -------------------------------------------------------------------
    # AWQ hook into real-data recalibration
    # -------------------------------------------------------------------
    def _recalibrate_with_real_data(self):
        # Run production FP8 calibration + non-AWQ FP4 graph capture.
        super()._recalibrate_with_real_data()

        if not self.use_awq or getattr(self, '_awq_calibrated', False):
            return

        logger.info("Running AWQ activation-aware calibration on %d FP4 layers...",
                    len(self._fp4_layers))
        act_gu, act_dn = self._collect_awq_activation_amax()
        self._requant_fp4_weights_with_awq(act_gu, act_dn)
        self._awq_calibrated = True

        # No graph recapture: _requant_fp4_weights_with_awq updates packed/sfb
        # and inv_s buffers in place, so the captured graph picks up the new
        # values on the next replay. Saves ~1 s of set_prompt time.
        logger.info("AWQ calibration complete (graph reused, in-place requant)")

    # -------------------------------------------------------------------
    # Graph capture — reroute encoder forward when FP4 enabled
    # -------------------------------------------------------------------
    def _capture_enc_ae_graph(self):
        if not self._fp4_layers:
            # No FP4 → defer to base class (zero behaviour change)
            return super()._capture_enc_ae_graph()

        # FP4 path — same capture logic as base, with encoder_forward swapped
        # for encoder_forward_with_fp4_subset. Duplication is intentional
        # (framework constraint: do not modify base class method).
        Se = self.Se
        self._alloc_fp4_scratch_for_Se(Se)
        total_keys = self.total_keys
        Le = self.Le; De = self.De; He = self.He
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha
        La = self.La

        enc_bufs = {
            'x':       self._enc_x.data_ptr(),
            'x_fp8':   self._enc_x_fp8.data_ptr(),
            'qkv':     self._enc_qkv_buf.data_ptr(),
            'logits':  self._enc_logits.data_ptr(),
            'attn_out': self._enc_attn.data_ptr(),
            'o_fp8':   self._enc_o_fp8.data_ptr(),
            'gate':    self._enc_gate.data_ptr(),
            'hidden':  self._enc_hidden.data_ptr(),
            'hid_fp8': self._enc_hid_fp8.data_ptr(),
            'fg':      self._enc_fg.data_ptr(),
            'ctx':     self._ctx,
        }
        enc_weights = {
            'qkv_w':     [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':       [w.data_ptr() for w in self._enc_o_w],
            'gate_w':    [w.data_ptr() for w in self._enc_gu_w],
            'down_w':    [w.data_ptr() for w in self._enc_d_w],
            'rope':      self._enc_rope.data_ptr(),
            'Kc':        self._Kc.reshape(-1).data_ptr(),
            'Vc':        self._Vc.reshape(-1).data_ptr(),
            'act_scales':  self._enc_calib_scales.data_ptr(),
            'alpha_host':  self._enc_alpha_host,
        }
        enc_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        ae_bufs = {
            'noise':   self._g_noise.data_ptr(),
            'x':       self._ae_x.data_ptr(),
            'xn':      self._ae_xn.data_ptr(),
            'gate':    self._ae_gate.data_ptr(),
            'qkv':     self._ae_qkv.data_ptr(),
            'logits':  self._ae_logits.data_ptr(),
            'attn_out': self._ae_attn.data_ptr(),
            'hid':     self._ae_hid.data_ptr(),
            'fg':      self._ae_fg.data_ptr(),
            'xn_fp8':  self._ae_xn_fp8.data_ptr(),
            'hid_fp8': self._ae_hid_fp8.data_ptr(),
            'ctx_fp8': self._ae_ctx_fp8.data_ptr(),
        }
        ae_weights = {
            'ain_w':      self._ain_w.data_ptr(),
            'ain_b':      self._ain_b.data_ptr(),
            'sa':         self._sa_all.data_ptr(),
            'qw':         self._dec_qkv_flat.data_ptr(),
            'Kc':         self._Kc.reshape(-1).data_ptr(),
            'Vc':         self._Vc.reshape(-1).data_ptr(),
            'ow':         self._dec_o_flat.data_ptr(),
            'sf':         self._sf_all.data_ptr(),
            'gw':         self._dec_gu_flat.data_ptr(),
            'dw':         self._dec_d_flat.data_ptr(),
            'aow':        self._aow.data_ptr(),
            'aob':        self._aob.data_ptr(),
            'fs':         self._fs_all.data_ptr(),
            'rope':       self._dec_rope.data_ptr(),
            'w_scales':   self._ae_w_dev.data_ptr(),
            'act_scales': self._ae_calib_scales.data_ptr(),
        }
        ae_dims = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        fp4_layers = self._fp4_layers
        fp4_weights = self._fp4_weights
        fp4_scratch = self._fp4_scratch_dict

        # Warmup
        for _ in range(3):
            self._Kc.zero_(); self._Vc.zero_()
            encoder_forward_with_fp4_subset(
                self._gemm, fvk, fvk_fp4, enc_bufs, enc_weights, enc_dims,
                stream=0, attn=self._attn,
                fp4_layers=fp4_layers, fp4_weights=fp4_weights,
                fp4_scratch=fp4_scratch,
                use_p1_split_gu=self.use_p1_split_gu)
            decoder_forward(self._ctx, fvk, ae_bufs, ae_weights,
                            ae_dims, stream=0, attn=self._attn)
        torch.cuda.synchronize()

        # Capture
        stream = torch.cuda.Stream()
        self._enc_ae_graph = torch.cuda.CUDAGraph()
        s_int = stream.cuda_stream
        with torch.cuda.stream(stream):
            self._enc_ae_graph.capture_begin()
            self._Kc.zero_(); self._Vc.zero_()
            encoder_forward_with_fp4_subset(
                self._gemm, fvk, fvk_fp4, enc_bufs, enc_weights, enc_dims,
                stream=s_int, attn=self._attn,
                fp4_layers=fp4_layers, fp4_weights=fp4_weights,
                fp4_scratch=fp4_scratch)
            decoder_forward(self._ctx, fvk, ae_bufs, ae_weights,
                            ae_dims, stream=s_int, attn=self._attn)
            self._enc_ae_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("Enc+AE CUDA graph captured with FP4 layers=%s P1=%s (Se=%d)",
                    sorted(self._fp4_layers), self.use_p1_split_gu, Se)
