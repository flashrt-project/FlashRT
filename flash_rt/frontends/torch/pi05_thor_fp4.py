"""Pi05TorchFrontendThor with explicit NVFP4 encoder and decoder paths.

STRICTLY ADDITIVE. Does not modify pi05_thor.py. Subclasses
Pi05TorchFrontendThor and routes explicitly selected encoder and decoder
projections through the NVFP4 extension.

Usage:
    pipe = Pi05TorchFrontendThorFP4(
        "/path/to/checkpoint",
        num_views=2,
        use_fp4_encoder_ffn=True,    # default False → bit-identical to base
        use_fp4_decoder=True,        # all four action-expert projections
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
from flash_rt.models.pi05.pipeline_thor import decoder_forward, decoder_forward_fp4
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


def _interleave_gu_rows(g_fp16: torch.Tensor, u_fp16: torch.Tensor,
                        inv_s_dn: torch.Tensor | None = None) -> torch.Tensor:
    """Pairwise interleave gate/up rows along N for the fused GeGLU
    epilogue GEMM (il[2j] = gate[j], il[2j+1] = up[j]).

    Any per-output-column scale must live in the weights because the
    epilogue applies no per-column vector; the down-projection AWQ inv_s
    is folded into the up rows (gelu(g) * u * inv_s == gelu(g) * (u * inv_s)).
    """
    if inv_s_dn is not None:
        u_fp16 = (u_fp16.float()
                  * inv_s_dn.float().unsqueeze(1)).to(fp16)
    il = torch.empty(2 * g_fp16.shape[0], g_fp16.shape[1],
                     dtype=fp16, device=g_fp16.device)
    il[0::2] = g_fp16
    il[1::2] = u_fp16
    return il.contiguous()


def _expand_down_k(d_fp16: torch.Tensor) -> torch.Tensor:
    """K-expand the down weight for the full-width interleaved GeGLU
    output: even columns hold the original weight, odd columns are zero
    (the duplicated odd input columns must contribute nothing)."""
    dx = torch.zeros(d_fp16.shape[0], 2 * d_fp16.shape[1],
                     dtype=fp16, device=d_fp16.device)
    dx[:, 0::2] = d_fp16
    return dx.contiguous()


class Pi05TorchFrontendThorFP4(Pi05TorchFrontendThor):
    """Pi0.5 Thor frontend with optional NVFP4 encoder and decoder paths."""

    def __init__(self, checkpoint_dir, num_views: int = 2,
                 use_cuda_graph: bool = True, autotune: int = 3,
                 *,
                 use_fp4_encoder_ffn: bool = False,
                 use_fp4_decoder: bool = False,
                 fp4_layers: Iterable[int] = (7, 8, 9),
                 use_awq: bool = False,
                 awq_alpha: float = 0.5,
                 awq_calib_iters: int = 8,
                 use_p1_split_gu: bool = False,
                 encoder_p1_combiner: str = "epilogue_hw",
                 encoder_down_variant: int = 7,
                 encoder_down_x_variant: int = 6,
                 decoder_qkv_variant: int = 10,
                 decoder_o_variant: int = 10,
                 decoder_gate_up_variant: int = 10,
                 decoder_down_variant: int = 10,
                 decoder_weight_format: str = "nvfp4",
                 decoder_act_format: str = "nvfp4",
                 decoder_fused_attn: bool = False,
                 decoder_fused_geglu: bool = True,
                 decoder_rht: bool = False,
                 use_fp8: bool = True,
                 state_prompt_mode: str = "exact",
                 state_prompt_fixed_max_len=None,
                 use_fa4: bool = False,
                 use_fp4_encoder_attn: bool = False,
                 use_fp4_encoder_attn_qkv: bool = False,
                 use_fp4_siglip_ffn: bool = False,
                 fp4_siglip_layers: Iterable[int] = None):
        if use_fp4_decoder and not use_fp8:
            raise ValueError(
                "use_fp4_decoder=True requires the FP8 Thor encoder path")
        if use_fp4_encoder_attn and not use_fp4_encoder_ffn:
            raise ValueError(
                "use_fp4_encoder_attn=True requires use_fp4_encoder_ffn=True")
        if use_fp4_encoder_attn_qkv and not use_fp4_encoder_attn:
            raise ValueError(
                "use_fp4_encoder_attn_qkv=True requires "
                "use_fp4_encoder_attn=True")
        # Base init (loads weights, allocates all FP8 buffers, etc.)
        super().__init__(checkpoint_dir, num_views=num_views,
                         use_cuda_graph=use_cuda_graph, autotune=autotune,
                         use_fp8=use_fp8,
                         state_prompt_mode=state_prompt_mode,
                         state_prompt_fixed_max_len=state_prompt_fixed_max_len,
                         use_fa4=use_fa4)

        self.use_fp4_encoder_ffn = bool(use_fp4_encoder_ffn)
        self.use_fp4_decoder = bool(use_fp4_decoder)
        self.use_fp4_encoder_attn = bool(use_fp4_encoder_attn)
        self.use_fp4_encoder_attn_qkv = bool(use_fp4_encoder_attn_qkv)
        self.use_fp4_siglip_ffn = bool(use_fp4_siglip_ffn)
        # Default: all 27 layers. Without AWQ, deep-layer FP4 FFNs push the
        # worst-sample raw cosine below the 0.995 gate (0.989 at 27 of 27);
        # the activation-aware Up-weight requant during multi-sample
        # calibration recovers it to 0.998.
        self._fp4_siglip_layers = (
            frozenset(fp4_siglip_layers) if fp4_siglip_layers is not None
            else frozenset(range(27)))
        self._fp4_layers = frozenset(fp4_layers) if self.use_fp4_encoder_ffn else frozenset()
        self.use_awq = bool(use_awq) and self.use_fp4_encoder_ffn
        self.awq_alpha = float(awq_alpha)
        self.awq_calib_iters = int(awq_calib_iters)
        # P1: split-GU 2-GEMM path (eliminates F4 v2; -2.9ms expected)
        self.use_p1_split_gu = bool(use_p1_split_gu) and self.use_fp4_encoder_ffn
        invalid_layers = sorted(
            l for l in self._fp4_layers if l < 0 or l >= self.Le - 1)
        if invalid_layers:
            raise ValueError(
                "fp4_layers contains non-live encoder FFN layers "
                f"{invalid_layers}; valid layers are [0, {self.Le - 2}]")
        if encoder_p1_combiner not in ("direct", "lut", "lut_native",
                                       "epilogue", "epilogue_hw"):
            raise ValueError(
                "encoder_p1_combiner must be 'direct', 'lut', "
                "'lut_native', 'epilogue', or 'epilogue_hw', got "
                f"{encoder_p1_combiner!r}")
        self.encoder_p1_combiner = encoder_p1_combiner
        self.encoder_down_variant = int(encoder_down_variant)
        # Tile variant for the K-expanded Down GEMM of the 'epilogue' P1
        # combiner (K doubles, so the best tile differs from the stock Down).
        self.encoder_down_x_variant = int(encoder_down_x_variant)
        self.decoder_qkv_variant = int(decoder_qkv_variant)
        self.decoder_o_variant = int(decoder_o_variant)
        self.decoder_gate_up_variant = int(decoder_gate_up_variant)
        self.decoder_down_variant = int(decoder_down_variant)
        if decoder_weight_format not in ("nvfp4", "e0m3"):
            raise ValueError(
                "decoder_weight_format must be 'nvfp4' or 'e0m3', got "
                f"{decoder_weight_format!r}")
        if decoder_act_format not in ("nvfp4", "e0m3"):
            raise ValueError(
                "decoder_act_format must be 'nvfp4' or 'e0m3', got "
                f"{decoder_act_format!r}")
        if decoder_act_format == "e0m3" and decoder_weight_format != "e0m3":
            raise ValueError(
                "decoder_act_format='e0m3' requires "
                "decoder_weight_format='e0m3'")
        if decoder_rht and decoder_act_format != "e0m3":
            raise ValueError(
                "decoder_rht=True requires decoder_act_format='e0m3' "
                "(the rotation must be applied to both operands)")
        self.decoder_weight_format = decoder_weight_format
        self.decoder_act_format = decoder_act_format
        self.decoder_rht = bool(decoder_rht)
        # Fold the decoder attention's seqused mask into the softmax
        # kernel: bit-identical output, one fewer launch. Only the
        # fixed-shape state-prompt path takes the seqused kernels, so this
        # is off by default and unmeasured by the FP4 E2E suite.
        self.decoder_fused_attn = bool(decoder_fused_attn)
        # Fused-epilogue decoder FFN: one interleaved GeGLU GEMM replaces
        # the gate_up GEMM + GeGLU-quantize kernel (nvfp4 weights only).
        self.decoder_fused_geglu = (bool(decoder_fused_geglu)
                                    and decoder_weight_format == 'nvfp4')

        if self._fp4_layers:
            if not _HAS_FP4:
                raise RuntimeError(
                    "use_fp4_encoder_ffn=True but flash_rt_fp4 not available. "
                    "Ensure flash_rt_fp4.so is built with NVFP4 support.")
            fp4_variant_count = int(fvk_fp4.cutlass_fp4_gemm_num_variants())
            if not 0 <= self.encoder_down_variant < fp4_variant_count:
                raise ValueError(
                    "encoder_down_variant must be in "
                    f"[0, {fp4_variant_count}), got "
                    f"{self.encoder_down_variant}")
            self._prepare_fp4_encoder()
            if self.use_fp4_encoder_attn:
                self._prepare_fp4_encoder_attn()
            if self.use_fp4_siglip_ffn:
                # Graph capture happens later through the overridden
                # _capture_siglip_graph once the attention backend exists.
                self._prepare_fp4_siglip_ffn()
            logger.info(
                "Pi05 FP4 enabled on encoder layers: %s  (AWQ=%s, attn=%s)",
                sorted(self._fp4_layers), self.use_awq,
                self.use_fp4_encoder_attn)

        if self.use_fp4_decoder:
            if not _HAS_FP4:
                raise RuntimeError(
                    "use_fp4_decoder=True but flash_rt_fp4 is unavailable or "
                    "was built without NVFP4 support")
            device_name = torch.cuda.get_device_name(0)
            capability = tuple(torch.cuda.get_device_capability(0))
            if device_name != "NVIDIA Thor" or capability != (11, 0):
                raise RuntimeError(
                    "Pi0.5 decoder FP4 requires NVIDIA Thor SM110; got "
                    f"{device_name} CC {capability}")

            variants = (self.decoder_qkv_variant, self.decoder_o_variant,
                        self.decoder_gate_up_variant,
                        self.decoder_down_variant)
            variant_count = int(fvk_fp4.cutlass_fp4_gemm_num_variants())
            for name, v in (("decoder_qkv_variant", variants[0]),
                            ("decoder_o_variant", variants[1]),
                            ("decoder_gate_up_variant", variants[2]),
                            ("decoder_down_variant", variants[3])):
                if not 0 <= v < variant_count:
                    raise ValueError(
                        f"{name} must be in [0, {variant_count}), got {v}")
            self._decoder_fp4_variants = variants
            self._decoder_fp4_weights = []
            if self.decoder_rht:
                h = torch.ones(1, 1, dtype=torch.float32, device='cuda')
                for _ in range(4):
                    h = torch.cat([torch.cat([h, h], 1),
                                   torch.cat([h, -h], 1)], 0)
                self._rht_h16 = h / 4.0  # symmetric orthonormal [16, 16]

            from safetensors import safe_open
            from flash_rt.executors.torch_weights import (
                _autodetect_strip_prefix,
                _interleave_qk_core,
            )
            root = "paligemma_with_expert.gemma_expert.model.layers"
            with safe_open(self._checkpoint_path, framework='pt', device='cuda') as sf:
                strip = _autodetect_strip_prefix(set(sf.keys()))

                def get(key):
                    return sf.get_tensor((strip + key) if strip else key)

                for layer in range(self.La):
                    prefix = f"{root}.{layer}"
                    q = _interleave_qk_core(
                        get(f"{prefix}.self_attn.q_proj.weight").float(), 8)
                    k = _interleave_qk_core(
                        get(f"{prefix}.self_attn.k_proj.weight").float(), 1)
                    v = get(f"{prefix}.self_attn.v_proj.weight").float()
                    qkv = torch.cat([q, k, v], dim=0).to(fp16).contiguous()
                    o = get(f"{prefix}.self_attn.o_proj.weight").to(
                        fp16).contiguous()
                    gate = get(f"{prefix}.mlp.gate_proj.weight").to(fp16)
                    up = get(f"{prefix}.mlp.up_proj.weight").to(fp16)
                    gate_up = torch.cat([gate, up], dim=0).contiguous()
                    down = get(f"{prefix}.mlp.down_proj.weight").to(
                        fp16).contiguous()
                    actual = (qkv.shape, o.shape, gate_up.shape, down.shape)
                    expected = (
                        (2560, self.Da), (self.Da, 2048),
                        (2 * self.Ha, self.Da), (self.Da, self.Ha),
                    )
                    if actual != expected:
                        raise ValueError(
                            f"decoder FP4 layer {layer} shapes {actual} != "
                            f"{expected}")
                    quantized = {}
                    for name, weight in (
                            ('qkv', qkv), ('o', o),
                            ('gate_up', gate_up), ('down', down)):
                        packed = quant_weight_nvfp4(weight)
                        if self.decoder_weight_format == "e0m3":
                            # Uniform-INT4 weights for the runtime-idesc
                            # GEMM; same packed/scale buffer layout. With
                            # RHT, rotate each 16-wide K block by the
                            # orthonormal Hadamard matrix before
                            # quantization (the activation kernels apply
                            # the same rotation at run time, so the GEMM
                            # is mathematically unchanged).
                            if self.decoder_rht:
                                Nw, Kw = weight.shape
                                weight = (
                                    weight.float().view(Nw, Kw // 16, 16)
                                    @ self._rht_h16
                                ).to(fp16).view(Nw, Kw).contiguous()
                            rc = fvk_fp4.quantize_e0m3_dynamic_sfa_fp16(
                                weight.data_ptr(),
                                packed['packed'].data_ptr(),
                                packed['sfb'].data_ptr(),
                                packed['N'], packed['K'], True, 0)
                        else:
                            rc = fvk_fp4.quantize_fp4_dynamic_sfa_mse_fp16(
                                weight.data_ptr(),
                                packed['packed'].data_ptr(),
                                packed['sfb'].data_ptr(),
                                packed['N'], packed['K'], True, 0)
                        if rc != 0:
                            raise RuntimeError(
                                f"decoder {name} {self.decoder_weight_format} "
                                f"quantization failed at layer {layer}: "
                                f"rc={rc}")
                        quantized[name] = packed
                    if self.decoder_fused_geglu:
                        # Fused-epilogue FFN: pairwise-interleaved gate/up
                        # rows, MSE-quantized like the other decoder weights.
                        il = _interleave_gu_rows(
                            gate_up[:self.Ha].contiguous(),
                            gate_up[self.Ha:].contiguous())
                        packed = quant_weight_nvfp4(il)
                        rc = fvk_fp4.quantize_fp4_dynamic_sfa_mse_fp16(
                            il.data_ptr(), packed['packed'].data_ptr(),
                            packed['sfb'].data_ptr(), packed['N'],
                            packed['K'], True, 0)
                        if rc != 0:
                            raise RuntimeError(
                                "decoder gu_il quantization failed at "
                                f"layer {layer}: rc={rc}")
                        quantized['gu_il'] = packed
                    torch.cuda.synchronize()
                    self._decoder_fp4_weights.append(quantized)

            self._decoder_fp4_xn = FP4ActScratch(
                self.Sa, self.Da, device='cuda')
            self._decoder_fp4_ctx = FP4ActScratch(
                self.Sa, 2048, device='cuda')
            self._decoder_fp4_hid = FP4ActScratch(
                self.Sa, self.Ha, device='cuda')
            if self.decoder_fused_geglu:
                # Dummy landing zone for the fused GeGLU GEMM's D output.
                self._decoder_fp4_gu_dummy = torch.zeros(
                    self.Sa, self.Ha, dtype=torch.uint8, device='cuda')
            logger.info(
                "Pi0.5 decoder NVFP4 enabled: variants=%s, layers=%d",
                self._decoder_fp4_variants, self.La)

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
        sig_layers = (sorted(self._sig_fp4_weights)
                      if self.use_fp4_siglip_ffn else [])
        per_sample_sig: dict = {l: [] for l in sig_layers}
        per_sample_qkv: dict = {}
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
            if self.use_fp4_encoder_attn_qkv:
                act_gu, act_dn, act_qkv = self._collect_awq_activation_amax(
                    collect_qkv=True)
                for l in act_qkv:
                    per_sample_qkv.setdefault(l, []).append(
                        act_qkv[l].detach().cpu().numpy().astype(np.float32))
            else:
                act_gu, act_dn = self._collect_awq_activation_amax()
            for l in self._fp4_layers:
                per_sample_gu[l].append(
                    act_gu[l].detach().cpu().numpy().astype(np.float32))
                per_sample_dn[l].append(
                    act_dn[l].detach().cpu().numpy().astype(np.float32))

            if sig_layers:
                # Runs after the encoder collection: the eager SigLIP pass
                # overwrites the SigLIP buffers, which the next sample's
                # graph replay rewrites anyway.
                sig_amax = self._collect_siglip_awq_amax()
                for l in sig_layers:
                    per_sample_sig[l].append(
                        sig_amax[l].detach().cpu().numpy().astype(
                            np.float32))

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

        if sig_layers:
            final_sig = {
                l: accumulate_amax(per_sample_sig[l], percentile=percentile)
                for l in sig_layers}
            self._requant_sig_fp4_weights_with_awq({
                l: torch.as_tensor(final_sig[l], device='cuda')
                for l in sig_layers})
            self._sig_awq_calibrated = True

        if per_sample_qkv:
            self._requant_enc_attn_qkv_with_awq({
                l: torch.as_tensor(
                    accumulate_amax(v, percentile=percentile), device='cuda')
                for l, v in per_sample_qkv.items()})

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
        self._fp4_weights = {}
        self._load_fp4_weights_from_safetensors()
        logger.info("FP4 weights loaded via fp16-native path (no FP8 intermediate)")

        # Variant selection (empirically tuned @ Se=968, see Step B test).
        self._fp4_variant_gu = pick_variant(2 * He, De)
        self._fp4_variant_dn = self.encoder_down_variant

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
                    if self.encoder_p1_combiner in ('epilogue',
                                                    'epilogue_hw'):
                        # Fused-epilogue paths: single interleaved gate/up
                        # weight (down AWQ inv_s folded into the up rows).
                        # Full-width 'epilogue' additionally needs the
                        # K-expanded down weight; 'epilogue_hw' keeps the
                        # stock down weight.
                        inv_dn = (self._awq_inv_s_dn[l]
                                  if self.use_awq else None)
                        self._fp4_weights[l]['gu_il'] = quant_weight_nvfp4(
                            _interleave_gu_rows(g_fp16, u_fp16, inv_dn))
                        if self.encoder_p1_combiner == 'epilogue':
                            self._fp4_weights[l]['down_x'] = (
                                quant_weight_nvfp4(_expand_down_k(d_fp16)))
                    else:
                        self._fp4_weights[l]['gate'] = quant_weight_nvfp4(
                            g_fp16)
                        self._fp4_weights[l]['up'] = quant_weight_nvfp4(
                            u_fp16)

    def _prepare_fp4_encoder_attn(self):
        """NVFP4-quantize the encoder attention projections.

        O projections (all layers that run attention) use the per-block
        MSE scale search. With ``use_fp4_encoder_attn_qkv`` the QKV
        projections are quantized as well (input_layernorm folded, GQA
        8/1 interleave, all layers); without AWQ their 4-bit Q/K weights
        shift the attention logits enough to break the raw-cosine gate
        (worst sample 0.97), so the QKV path relies on the
        activation-aware requant during multi-sample calibration.
        """
        from safetensors import safe_open
        from flash_rt.executors.torch_weights import (
            _autodetect_strip_prefix, _interleave_qk_core)

        De = self.De
        model_root = (
            "paligemma_with_expert.paligemma.model.language_model.layers")
        self._enc_attn_fp4_weights = {}
        self._enc_attn_qkv_fp16 = {}
        self._enc_attn_qkv_inv_s = {}
        with safe_open(self._checkpoint_path, framework='pt',
                       device='cuda') as sf:
            _strip = _autodetect_strip_prefix(set(sf.keys()))

            def get(k):
                return sf.get_tensor((_strip + k) if _strip else k)

            for l in range(self.Le):
                p = f"{model_root}.{l}"
                entry = {}
                if l < self.Le - 1:
                    o_fp16 = get(
                        f"{p}.self_attn.o_proj.weight").to(fp16).contiguous()
                    if o_fp16.shape != (De, De):
                        raise ValueError(
                            f"encoder o shape {tuple(o_fp16.shape)} != "
                            f"{(De, De)}")
                    packed = quant_weight_nvfp4(o_fp16)
                    rc = fvk_fp4.quantize_fp4_dynamic_sfa_mse_fp16(
                        o_fp16.data_ptr(), packed['packed'].data_ptr(),
                        packed['sfb'].data_ptr(), packed['N'], packed['K'],
                        True, 0)
                    if rc != 0:
                        raise RuntimeError(
                            "encoder attn O FP4 MSE quantization failed at "
                            f"layer {l}: rc={rc}")
                    entry['o'] = packed
                if self.use_fp4_encoder_attn_qkv:
                    q = _interleave_qk_core(
                        get(f"{p}.self_attn.q_proj.weight").float(), 8)
                    k = _interleave_qk_core(
                        get(f"{p}.self_attn.k_proj.weight").float(), 1)
                    v = get(f"{p}.self_attn.v_proj.weight").float()
                    fa = (1.0 + get(f"{p}.input_layernorm.weight").float()
                          ).unsqueeze(0)
                    qkv_fp16 = torch.cat(
                        [q * fa, k * fa, v * fa],
                        dim=0).to(fp16).contiguous()
                    if qkv_fp16.shape != (2560, De):
                        raise ValueError(
                            f"encoder qkv shape {tuple(qkv_fp16.shape)} != "
                            f"{(2560, De)}")
                    packed_q = quant_weight_nvfp4(qkv_fp16)
                    rc = fvk_fp4.quantize_fp4_dynamic_sfa_mse_fp16(
                        qkv_fp16.data_ptr(), packed_q['packed'].data_ptr(),
                        packed_q['sfb'].data_ptr(), packed_q['N'],
                        packed_q['K'], True, 0)
                    if rc != 0:
                        raise RuntimeError(
                            "encoder attn QKV FP4 MSE quantization failed "
                            f"at layer {l}: rc={rc}")
                    inv = torch.ones(De, dtype=fp16, device='cuda')
                    entry['qkv'] = packed_q
                    entry['qkv_inv_s'] = inv.data_ptr()
                    self._enc_attn_qkv_fp16[l] = qkv_fp16
                    self._enc_attn_qkv_inv_s[l] = inv
                if entry:
                    torch.cuda.synchronize()
                    self._enc_attn_fp4_weights[l] = entry
        logger.info("Pi05 FP4 encoder attention projections quantized "
                    "(%d layers, qkv=%s)",
                    len(self._enc_attn_fp4_weights),
                    self.use_fp4_encoder_attn_qkv)

    def _requant_enc_attn_qkv_with_awq(self, qkv_amax: dict):
        """In-place AWQ requant of the encoder attention QKV weights."""
        for l, a in qkv_amax.items():
            if l not in self._enc_attn_qkv_fp16:
                continue
            w_scaled, inv_s = self._awq_scale_weight(
                self._enc_attn_qkv_fp16[l], activation_amax=a)
            packed = self._enc_attn_fp4_weights[l]['qkv']
            rc = fvk_fp4.quantize_fp4_dynamic_sfa_mse_fp16(
                w_scaled.data_ptr(), packed['packed'].data_ptr(),
                packed['sfb'].data_ptr(), packed['N'], packed['K'], True, 0)
            if rc != 0:
                raise RuntimeError(
                    f"encoder attn QKV AWQ requant failed at layer {l}: "
                    f"rc={rc}")
            self._enc_attn_qkv_inv_s[l].copy_(inv_s)
        torch.cuda.synchronize()
        logger.info("Encoder attn QKV AWQ requant complete (%d layers)",
                    len(qkv_amax))

    def _prepare_fp4_siglip_ffn(self):
        """NVFP4-quantize the SigLIP FFN (fc1/fc2) with the hidden dim
        zero-padded to a multiple of 32 (fp4 TMA alignment). Pad rows and
        columns carry zero weights/biases, so padding is mathematically
        inert. Also allocates the FFN activation scratch and re-captures
        the SigLIP graph through siglip_forward_with_fp4_ffn.
        """
        from safetensors import safe_open
        from flash_rt.executors.torch_weights import _autodetect_strip_prefix

        D, H, L = self.sig_D, self.sig_H, self.sig_L
        H_pad = ((H + 31) // 32) * 32
        root = ("paligemma_with_expert.paligemma.model.vision_tower."
                "vision_model.encoder.layers")
        self._sig_fp4_weights = {}
        self._sig_fp4_up_bias = []       # keep padded bias tensors alive
        self._sig_fp4_up_fp16 = {}
        self._sig_awq_inv_s = {}
        with safe_open(self._checkpoint_path, framework='pt',
                       device='cuda') as sf:
            _strip = _autodetect_strip_prefix(set(sf.keys()))

            def get(k):
                return sf.get_tensor((_strip + k) if _strip else k)

            for l in sorted(self._fp4_siglip_layers):
                p = f"{root}.{l}"
                w_up = get(f"{p}.mlp.fc1.weight").to(fp16)
                b_up = get(f"{p}.mlp.fc1.bias").to(fp16)
                w_dn = get(f"{p}.mlp.fc2.weight").to(fp16)
                if w_up.shape != (H, D) or w_dn.shape != (D, H):
                    raise ValueError(
                        f"SigLIP FFN layer {l} shapes {tuple(w_up.shape)}/"
                        f"{tuple(w_dn.shape)} != {(H, D)}/{(D, H)}")
                w_up_p = torch.zeros(H_pad, D, dtype=fp16, device='cuda')
                w_up_p[:H] = w_up
                b_up_p = torch.zeros(H_pad, dtype=fp16, device='cuda')
                b_up_p[:H] = b_up
                w_dn_p = torch.zeros(D, H_pad, dtype=fp16, device='cuda')
                w_dn_p[:, :H] = w_dn
                entry = {}
                for name, w in (('up', w_up_p.contiguous()),
                                ('down', w_dn_p.contiguous())):
                    packed = quant_weight_nvfp4(w)
                    rc = fvk_fp4.quantize_fp4_dynamic_sfa_mse_fp16(
                        w.data_ptr(), packed['packed'].data_ptr(),
                        packed['sfb'].data_ptr(), packed['N'], packed['K'],
                        True, 0)
                    if rc != 0:
                        raise RuntimeError(
                            f"SigLIP FFN {name} FP4 MSE quantization "
                            f"failed at layer {l}: rc={rc}")
                    entry[name] = packed
                entry['up_bias'] = b_up_p.data_ptr()
                self._sig_fp4_up_bias.append(b_up_p)
                # Keep the fp16 padded Up weight for the AWQ requant and
                # allocate the per-channel inverse scale (identity until
                # real-data calibration).
                self._sig_fp4_up_fp16[l] = w_up_p.contiguous()
                inv = torch.ones(D, dtype=fp16, device='cuda')
                self._sig_awq_inv_s[l] = inv
                entry['ln_inv_s'] = inv.data_ptr()
                torch.cuda.synchronize()
                self._sig_fp4_weights[l] = entry
        self._sig_fp4_scratch = {
            'ln_act': FP4ActScratch(self.sig_S, D, device='cuda'),
            'hid_act': FP4ActScratch(self.sig_S, H_pad, device='cuda'),
            'H_pad': H_pad,
        }
        logger.info("Pi05 SigLIP FFN NVFP4 quantized (%d layers, H_pad=%d)",
                    len(self._sig_fp4_weights), H_pad)

    def _capture_siglip_graph(self):
        """Capture the SigLIP graphs; routes through the NVFP4 FFN when
        use_fp4_siglip_ffn is enabled (base FP8 path otherwise, including
        during base-class construction before the flag exists)."""
        if not getattr(self, 'use_fp4_siglip_ffn', False) or \
                not hasattr(self, '_sig_fp4_weights'):
            return super()._capture_siglip_graph()
        import numpy as np
        from flash_rt.hardware.thor.shared_primitives_fp4 import (
            siglip_forward_with_fp4_ffn)

        def _sig_fwd(stream_int):
            siglip_forward_with_fp4_ffn(
                self._gemm, fvk, fvk_fp4, self._sig_bufs,
                self._sig_weights, self._sig_dims, stream=stream_int,
                attn=self._attn,
                fp4_weights=self._sig_fp4_weights,
                fp4_scratch=self._sig_fp4_scratch)

        dummy_img = np.zeros((self.num_views, 224, 224, 3), dtype=np.float16)
        self._img_buf.upload(dummy_img)
        for _ in range(3):
            self._patch_embed_ops(0)
            self._sig_x.zero_()
            _sig_fwd(0)
            self._postln_project_ops(0)
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        self._siglip_graph = torch.cuda.CUDAGraph()
        s_int = stream.cuda_stream
        with torch.cuda.stream(stream):
            self._siglip_graph.capture_begin()
            self._patch_embed_ops(s_int)
            _sig_fwd(s_int)
            self._postln_project_ops(s_int)
            self._siglip_graph.capture_end()
        torch.cuda.synchronize()

        dummy_u8 = np.full(
            (self.num_views, 224, 224, 3), 128, dtype=np.uint8)
        self._img_u8_buf.upload(dummy_u8)
        for _ in range(3):
            self._patch_embed_ops(0, uint8_input=True)
            self._sig_x.zero_()
            _sig_fwd(0)
            self._postln_project_ops(0)
        torch.cuda.synchronize()

        uint8_stream = torch.cuda.Stream()
        self._siglip_u8_graph = torch.cuda.CUDAGraph()
        u8_int = uint8_stream.cuda_stream
        with torch.cuda.stream(uint8_stream):
            self._siglip_u8_graph.capture_begin()
            self._patch_embed_ops(u8_int, uint8_input=True)
            _sig_fwd(u8_int)
            self._postln_project_ops(u8_int)
            self._siglip_u8_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("SigLIP CUDA graph captured with NVFP4 FFN (S=%d)",
                    self.sig_S)

    def _awq_scale_weight(self, W: torch.Tensor,
                           activation_amax: torch.Tensor | None = None,
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """AWQ per-input-channel (K axis) pre-scale.

        If ``activation_amax`` is provided (shape [K], fp32), uses standard
        activation-aware weight quantization:
            s[k] = (activation_amax[k] / activation_amax.mean())^alpha
        Otherwise fall back to the weight-only amax estimate.

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
                quant_weight_nvfp4_inplace(
                    gu_scaled, self._fp4_weights[l]['gate_up'])
                quant_weight_nvfp4_inplace(
                    d_scaled, self._fp4_weights[l]['down'])
                if self.use_p1_split_gu and 'gate' in self._fp4_weights[l]:
                    quant_weight_nvfp4_inplace(
                        gu_scaled[:He, :].contiguous(), self._fp4_weights[l]['gate'])
                    quant_weight_nvfp4_inplace(
                        gu_scaled[He:, :].contiguous(), self._fp4_weights[l]['up'])
                if self.use_p1_split_gu and 'gu_il' in self._fp4_weights[l]:
                    quant_weight_nvfp4_inplace(
                        _interleave_gu_rows(
                            gu_scaled[:He, :].contiguous(),
                            gu_scaled[He:, :].contiguous(),
                            inv_s_dn),
                        self._fp4_weights[l]['gu_il'])
                    if 'down_x' in self._fp4_weights[l]:
                        quant_weight_nvfp4_inplace(
                            _expand_down_k(d_scaled),
                            self._fp4_weights[l]['down_x'])

    def _collect_awq_activation_amax(self, collect_qkv=False):
        """Run calibration forward with hook that snapshots fp16 activations
        at FP4 layers' Gate+Up and Down inputs; return per-channel amax dicts."""
        Se = self.Se; De = self.De; He = self.He; NHe = self.NHe; HDe = self.HDe
        Le = self.Le; total_keys = self.total_keys
        act_gu = {l: torch.zeros(De, dtype=torch.float32, device='cuda') for l in self._fp4_layers}
        act_dn = {l: torch.zeros(He, dtype=torch.float32, device='cuda') for l in self._fp4_layers}
        # Snapshot scratch buffers (one-per-FP4-layer to avoid re-alloc)
        x_snap = {l: torch.empty(Se, De, dtype=fp16, device='cuda') for l in self._fp4_layers}
        qkv_snap = {}
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
            if collect_qkv:
                torch.cuda.synchronize()
                qkv_snap[l] = x.clone()
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
        # Per-channel activation amax, matching the reference calibration.
        for l in self._fp4_layers:
            act_gu[l] = x_snap[l].abs().float().amax(dim=0)
            act_dn[l] = h_snap[l].abs().float().amax(dim=0)
        act_qkv = {}
        if collect_qkv:
            # QKV GEMM input = weightless RMSNorm of the layer-entry
            # residual (the input_layernorm gamma is folded into the
            # quantized weight).
            for l, xs in qkv_snap.items():
                xn = xs.float()
                xn = xn * torch.rsqrt(
                    xn.pow(2).mean(dim=1, keepdim=True) + 1e-6)
                act_qkv[l] = xn.abs().amax(dim=0)
        # Free snapshots
        del x_snap, h_snap, qkv_snap
        torch.cuda.empty_cache()
        if collect_qkv:
            return act_gu, act_dn, act_qkv
        return act_gu, act_dn

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
            if self.encoder_p1_combiner == 'epilogue':
                # Fused-epilogue path: one full-width [Se, 2H] FP4 buffer
                # between the interleaved GEMM and the K-expanded Down.
                self._fp4_p1_il = FP4Buffer(Se, 2 * He, device='cuda')
            elif self.encoder_p1_combiner == 'epilogue_hw':
                # Half-width path writes Down's stock input scratch; the
                # collective's D output lands in this reusable dummy.
                self._fp4_p1_dummy = torch.zeros(
                    Se, He, dtype=torch.uint8, device='cuda')
            else:
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
            self._fp4_scratch_dict['p1_combiner'] = self.encoder_p1_combiner
            if self.encoder_p1_combiner == 'epilogue':
                self._fp4_scratch_dict['p1_il_p4']  = self._fp4_p1_il.packed.data_ptr()
                self._fp4_scratch_dict['p1_il_sfa'] = self._fp4_p1_il.sfa.data_ptr()
                self._fp4_scratch_dict['variant_dn_x'] = self.encoder_down_x_variant
            elif self.encoder_p1_combiner == 'epilogue_hw':
                self._fp4_scratch_dict['p1_dummy'] = self._fp4_p1_dummy.data_ptr()
            else:
                self._fp4_scratch_dict['p1_gate_p4']  = self._fp4_p1_gate.packed.data_ptr()
                self._fp4_scratch_dict['p1_gate_sfa'] = self._fp4_p1_gate.sfa.data_ptr()
                self._fp4_scratch_dict['p1_up_p4']    = self._fp4_p1_up.packed.data_ptr()
                self._fp4_scratch_dict['p1_up_sfa']   = self._fp4_p1_up.sfa.data_ptr()
        if self.use_fp4_encoder_attn:
            self._fp4_attn_scratch = FP4ActScratch(Se, De, device='cuda')
            self._fp4_scratch_dict['attn_act'] = self._fp4_attn_scratch
            self._fp4_scratch_dict['attn_variant'] = 7
        self._fp4_scratch_Se = Se

    # -------------------------------------------------------------------
    # AWQ hook into real-data recalibration
    # -------------------------------------------------------------------
    def _recalibrate_with_real_data(self):
        # Run production FP8 calibration + non-AWQ FP4 graph capture.
        super()._recalibrate_with_real_data()

        if (self.use_fp4_siglip_ffn
                and not getattr(self, '_sig_awq_calibrated', False)):
            logger.info(
                "Running SigLIP AWQ calibration on %d FP4 FFN layers...",
                len(self._sig_fp4_weights))
            sig_amax = self._collect_siglip_awq_amax()
            self._requant_sig_fp4_weights_with_awq(sig_amax)
            self._sig_awq_calibrated = True

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

    def _collect_siglip_awq_amax(self):
        """One eager FP8 SigLIP pass on the current calibration image,
        snapshotting the FFN LayerNorm input at each NVFP4 layer; returns
        {layer: per-channel amax of the fp16 LayerNorm output} (the Up GEMM
        activation)."""
        import torch.nn.functional as F
        S, D, H = self.sig_S, self.sig_D, self.sig_H
        L = self.sig_L
        spv = 256
        w = self._sig_weights
        alpha = w['alpha']
        unit = w['unit_scale']

        x = self._sig_x
        x_fp8 = self._sig_x_fp8
        qkv = self._sig_qkv
        attn_out = self._sig_attn
        hidden = self._sig_hidden
        hid_fp8 = self._sig_hid_fp8

        amax = {}
        self._patch_embed_ops(0)
        for l in range(L):
            a_qkv = alpha[l * 4 + 0]
            a_o = alpha[l * 4 + 1]
            a_up = alpha[l * 4 + 2]
            a_down = alpha[l * 4 + 3]
            fvk.layer_norm_fp8(x.data_ptr(), x_fp8.data_ptr(),
                               w['ln_attn_w'][l], w['ln_attn_b'][l],
                               S, D, 1e-5, 0)
            self._gemm.fp8_nn_bias(x_fp8.data_ptr(), w['qkv_w'][l],
                                   qkv.data_ptr(), w['qkv_b'][l],
                                   S, 3 * D, D, a_qkv, 0)
            self._attn.run("siglip", 0, q_seq=spv, stream=0)
            fvk.quantize_fp8_static_fp16(attn_out.data_ptr(),
                                         x_fp8.data_ptr(), unit, S * D, 0)
            self._gemm.fp8_nn_bias_res(x_fp8.data_ptr(), w['o_w'][l],
                                       x.data_ptr(), w['o_b'][l],
                                       S, D, D, a_o, 0)
            if l in self._sig_fp4_weights:
                torch.cuda.synchronize()
                xn = F.layer_norm(
                    x.float(), (D,), self._sig_ln_ffn_w[l].float(),
                    self._sig_ln_ffn_b[l].float(), 1e-5)
                amax[l] = xn.abs().amax(dim=0)
            fvk.layer_norm_fp8(x.data_ptr(), x_fp8.data_ptr(),
                               w['ln_ffn_w'][l], w['ln_ffn_b'][l],
                               S, D, 1e-5, 0)
            self._gemm.fp8_nn_gelu_bias(x_fp8.data_ptr(), w['up_w'][l],
                                        hidden.data_ptr(), w['up_b'][l],
                                        S, H, D, a_up, 0)
            fvk.quantize_fp8_static_fp16(hidden.data_ptr(),
                                         hid_fp8.data_ptr(), unit, S * H, 0)
            self._gemm.fp8_nn_bias_res(hid_fp8.data_ptr(), w['down_w'][l],
                                       x.data_ptr(), w['down_b'][l],
                                       S, D, H, a_down, 0)
        torch.cuda.synchronize()
        return amax

    def _requant_sig_fp4_weights_with_awq(self, sig_amax: dict):
        """In-place AWQ requant of the SigLIP Up weights (packed/sfb keep
        their addresses, so the captured graphs pick up the new values)."""
        for l, a in sig_amax.items():
            w_fp16 = self._sig_fp4_up_fp16[l]
            w_scaled, inv_s = self._awq_scale_weight(w_fp16,
                                                     activation_amax=a)
            packed = self._sig_fp4_weights[l]['up']
            rc = fvk_fp4.quantize_fp4_dynamic_sfa_mse_fp16(
                w_scaled.data_ptr(), packed['packed'].data_ptr(),
                packed['sfb'].data_ptr(), packed['N'], packed['K'], True, 0)
            if rc != 0:
                raise RuntimeError(
                    f"SigLIP AWQ Up requant failed at layer {l}: rc={rc}")
            self._sig_awq_inv_s[l].copy_(inv_s)
        torch.cuda.synchronize()
        logger.info("SigLIP AWQ requant complete (%d layers, in place)",
                    len(sig_amax))

    # -------------------------------------------------------------------
    # Graph capture — reroute encoder forward when FP4 enabled
    # -------------------------------------------------------------------
    def _capture_enc_ae_graph(self):
        if not self._fp4_layers and not self.use_fp4_decoder:
            # No FP4 → defer to base class (zero behaviour change)
            return super()._capture_enc_ae_graph()

        # FP4 path. Duplication keeps the base production frontend unchanged.
        Se = self.Se
        if self._fp4_layers:
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
            'action_f32': self._ae_action_f32.data_ptr(),
            'xn_fp8':  self._ae_xn_fp8.data_ptr(),
            'hid_fp8': self._ae_hid_fp8.data_ptr(),
            'ctx_fp8': self._ae_ctx_fp8.data_ptr(),
        }
        if self.use_fp4_decoder:
            ae_bufs.update({
                'xn_fp4': self._decoder_fp4_xn.packed.data_ptr(),
                'xn_sfa': self._decoder_fp4_xn.sfa.data_ptr(),
                'ctx_fp4': self._decoder_fp4_ctx.packed.data_ptr(),
                'ctx_sfa': self._decoder_fp4_ctx.sfa.data_ptr(),
                'hid_fp4': self._decoder_fp4_hid.packed.data_ptr(),
                'hid_sfa': self._decoder_fp4_hid.sfa.data_ptr(),
            })
        ae_weights = {
            'ain_w':      self._ain_w.data_ptr(),
            'ain_b':      self._ain_b.data_ptr(),
            'sa':         self._sa_all.data_ptr(),
            'qw':         self._dec_qkv_flat.data_ptr(),
            'Kc':         self._Kc.reshape(-1).data_ptr(),
            'Vc':         self._Vc.reshape(-1).data_ptr(),
            'dec_devpos': self._attn.dec_devpos.data_ptr(),
            'ow':         self._dec_o_flat.data_ptr(),
            'sf':         self._sf_all.data_ptr(),
            'gw':         self._dec_gu_flat.data_ptr(),
            'dw':         self._dec_d_flat.data_ptr(),
            'aow':        self._aow.data_ptr(),
            'aob':        self._aob.data_ptr(),
            'aob_dt':     self._aob_dt.data_ptr(),
            'dt':         self._ae_dt,
            'fs':         self._fs_all.data_ptr(),
            'rope':       self._dec_rope.data_ptr(),
            'w_scales':   self._ae_w_dev.data_ptr(),
            'act_scales': self._ae_calib_scales.data_ptr(),
        }
        if self.use_fp4_decoder:
            ae_weights.update({
                'qw_fp4': [w['qkv']['packed'].data_ptr()
                           for w in self._decoder_fp4_weights],
                'qw_sfb': [w['qkv']['sfb'].data_ptr()
                           for w in self._decoder_fp4_weights],
                'ow_fp4': [w['o']['packed'].data_ptr()
                           for w in self._decoder_fp4_weights],
                'ow_sfb': [w['o']['sfb'].data_ptr()
                           for w in self._decoder_fp4_weights],
                'gw_fp4': [w['gate_up']['packed'].data_ptr()
                           for w in self._decoder_fp4_weights],
                'gw_sfb': [w['gate_up']['sfb'].data_ptr()
                           for w in self._decoder_fp4_weights],
                'dw_fp4': [w['down']['packed'].data_ptr()
                           for w in self._decoder_fp4_weights],
                'dw_sfb': [w['down']['sfb'].data_ptr()
                           for w in self._decoder_fp4_weights],
            })
            if self.decoder_fused_geglu:
                ae_weights.update({
                    'gwil_fp4': [w['gu_il']['packed'].data_ptr()
                                 for w in self._decoder_fp4_weights],
                    'gwil_sfb': [w['gu_il']['sfb'].data_ptr()
                                 for w in self._decoder_fp4_weights],
                    'gu_dummy': self._decoder_fp4_gu_dummy.data_ptr(),
                })
        ae_dims = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': La, 'enc_seq': Se,
            'total_keys': total_keys,
            'fixed_shape': self._fixed_shape_active,
        }
        if self.use_fp4_decoder:
            ae_dims['fp4_variants'] = self._decoder_fp4_variants
            ae_dims['weight_format'] = self.decoder_weight_format
            ae_dims['act_format'] = self.decoder_act_format
            ae_dims['rht'] = self.decoder_rht
            ae_dims['fused_geglu'] = self.decoder_fused_geglu
            if self._attn is not None:
                # Fold the decoder seqused mask into the softmax kernel.
                self._attn.use_fused_softmax = self.decoder_fused_attn

        fp4_layers = self._fp4_layers
        fp4_weights = self._fp4_weights if fp4_layers else None
        fp4_scratch = self._fp4_scratch_dict if fp4_layers else None

        # Warmup
        for _ in range(3):
            self._Kc.zero_(); self._Vc.zero_()
            if fp4_layers:
                encoder_forward_with_fp4_subset(
                    self._gemm, fvk, fvk_fp4, enc_bufs, enc_weights, enc_dims,
                    stream=0, attn=self._attn,
                    fp4_layers=fp4_layers, fp4_weights=fp4_weights,
                    fp4_scratch=fp4_scratch,
                    use_p1_split_gu=self.use_p1_split_gu,
                    fp4_attn_weights=(self._enc_attn_fp4_weights
                                      if self.use_fp4_encoder_attn else None))
            else:
                encoder_forward(
                    self._gemm, fvk, enc_bufs, enc_weights, enc_dims,
                    stream=0, attn=self._attn, use_fp8=self.use_fp8)
            if self.use_fp4_decoder:
                decoder_forward_fp4(
                    self._ctx, fvk, fvk_fp4, ae_bufs, ae_weights,
                    ae_dims, stream=0, attn=self._attn)
            else:
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
            if fp4_layers:
                encoder_forward_with_fp4_subset(
                    self._gemm, fvk, fvk_fp4, enc_bufs, enc_weights, enc_dims,
                    stream=s_int, attn=self._attn,
                    fp4_layers=fp4_layers, fp4_weights=fp4_weights,
                    fp4_scratch=fp4_scratch,
                    use_p1_split_gu=self.use_p1_split_gu,
                    fp4_attn_weights=(self._enc_attn_fp4_weights
                                      if self.use_fp4_encoder_attn else None))
            else:
                encoder_forward(
                    self._gemm, fvk, enc_bufs, enc_weights, enc_dims,
                    stream=s_int, attn=self._attn, use_fp8=self.use_fp8)
            if self.use_fp4_decoder:
                decoder_forward_fp4(
                    self._ctx, fvk, fvk_fp4, ae_bufs, ae_weights,
                    ae_dims, stream=s_int, attn=self._attn)
            else:
                decoder_forward(self._ctx, fvk, ae_bufs, ae_weights,
                                ae_dims, stream=s_int, attn=self._attn)
            self._enc_ae_graph.capture_end()
        torch.cuda.synchronize()
        self._model_runtime_full_graph = None
        self._model_runtime_context_graph = None
        self._model_runtime_decode_only_graph = None
        self._model_runtime_rtc_prefix_graphs = {}
        self.pipeline._decoder_only_graph = None
        logger.info(
            "Enc+AE CUDA graph captured with encoder FP4 layers=%s, "
            "decoder FP4=%s, P1=%s (Se=%d)",
            sorted(self._fp4_layers), self.use_fp4_decoder,
            self.use_p1_split_gu, Se)

    def set_rl_mode(self, *, cfg_enable: bool = True, cfg_beta: float = 1.5,
                    advantage_positive: bool = True) -> None:
        if self.use_fp4_decoder and cfg_enable:
            raise RuntimeError(
                "Pi0.5 decoder FP4 currently supports standard B=1 inference "
                "only; CFG is not implemented")
        return super().set_rl_mode(
            cfg_enable=cfg_enable, cfg_beta=cfg_beta,
            advantage_positive=advantage_positive)

    def set_batched_mode(self, *, enable: bool = True,
                         batch_size: int = 2) -> None:
        if self.use_fp4_decoder and enable:
            raise RuntimeError(
                "Pi0.5 decoder FP4 currently supports standard B=1 inference "
                "only; batched inference is not implemented")
        return super().set_batched_mode(enable=enable, batch_size=batch_size)

    def _ensure_model_runtime_export(self) -> None:
        if self.use_fp4_decoder:
            raise RuntimeError(
                "Pi0.5 decoder FP4 model-runtime export is not implemented")
        return super()._ensure_model_runtime_export()
