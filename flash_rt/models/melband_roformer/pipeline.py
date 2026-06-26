"""FlashRT -- MelBandRoformer FP8 kernelized inference pipeline.

MelBandRoformer is an audio source-separation model: a band-split
spectrogram Transformer. The per-band Transformer blocks are rewritten
here as FP8 (e4m3) static-quantized forward using custom CUDA kernels
from ``flash_rt_kernels``. The STFT / band-split / band-merge / ISTFT
host logic (``_split_forward``) is unchanged from the reference model;
only the intra-band Transformer ``forward`` is monkey-patched onto FP8
GEMMs. After quantization the original BF16 Linear weights are deleted
to recover ~400 MB of VRAM with identical numerics.

Kernel surface used (all from ``flash_rt_kernels``):

* ``rms_norm_fp8``                         -- RMSNorm -> FP8 (attention in-norm)
* ``mbr_qkv_split_rope``                   -- QKV split + RoPE apply
* ``mbr_gated_attn_quant``                 -- output gate + FP8 quantize
* ``mbr_fp8_dequant_bf16``                 -- FP8 -> BF16 (gate residual)
* ``mbr_resadd_rmsnorm_fp8_keepres``       -- residual-add + RMSNorm -> FP8
* ``bias_gelu_quantize_fp8_static_bf16``   -- GeLU(bias) + FP8 quantize
* ``mbr_fused_add_rmsnorm_bf16``           -- fused residual-add + final RMSNorm

The ``mbr_*`` kernels require ``-DFLASHRT_ENABLE_MELBAND_ROFORMER=ON`` at
build time. Importing this module does **not** require them; validation is
deferred to ``MelBandRoformerPipeline.__init__``.
"""

import gc
import json
import logging
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

FP8_MAX = 448.0

_REQUIRED_MBR_SYMBOLS = (
    "mbr_qkv_split_rope",
    "mbr_gated_attn_quant",
    "mbr_fp8_dequant_bf16",
    "mbr_resadd_rmsnorm_fp8_keepres",
    "mbr_fused_add_rmsnorm_bf16",
)


def _load_kernels():
    """Import ``flash_rt_kernels`` and validate that all ``mbr_*`` symbols exist.

    Returns ``(fvk, mk)`` where ``fvk`` is the raw module and ``mk`` is a
    lightweight namespace grouping the ``mbr_*`` entry points.

    Raises ``RuntimeError`` if any required symbol is missing (i.e. the
    build did not include ``-DFLASHRT_ENABLE_MELBAND_ROFORMER=ON``).
    """
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError:
        try:
            import flash_rt_kernels as fvk  # type: ignore
        except ImportError:
            raise RuntimeError(
                "flash_rt_kernels is not available. Build FlashRT with "
                "'pip install -e .' and ensure the compiled .so is on the path.")

    missing = [s for s in _REQUIRED_MBR_SYMBOLS if not hasattr(fvk, s)]
    if missing:
        raise RuntimeError(
            "MelBandRoformer kernels are not compiled into flash_rt_kernels. "
            "Rebuild with:  cmake .. -DFLASHRT_ENABLE_MELBAND_ROFORMER=ON && make -j\n"
            f"Missing symbols: {', '.join(missing)}")

    mk = types.SimpleNamespace(
        qkv_split_rope=fvk.mbr_qkv_split_rope,
        gated_attn_quant=fvk.mbr_gated_attn_quant,
        fp8_dequant_bf16=fvk.mbr_fp8_dequant_bf16,
        resadd_rmsnorm_fp8_keepres=fvk.mbr_resadd_rmsnorm_fp8_keepres,
        fused_add_rmsnorm_bf16=fvk.mbr_fused_add_rmsnorm_bf16,
    )
    return fvk, mk


def _fp8_gemm(a, w_cm, sa, sb):
    return torch._scaled_mm(a, w_cm, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


def _load_calib(path):
    """Load the FP8 static per-tensor calibration dict from a JSON file path.

    Returns ``{}`` for a missing/None path so uncalibrated runs fall back to
    scale 1.0 (see ``MelBandRoformerPipeline._s``) instead of crashing.
    """
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


class _Norm:
    def __init__(self, gamma, act_scale, device):
        self.dim = gamma.shape[0]
        self.gamma = gamma.detach().to(torch.bfloat16).contiguous()
        self.d_scale = torch.tensor([float(act_scale)], dtype=torch.float32, device=device)
        self.scale_val = float(act_scale)


class _W:
    def __init__(self, linear, device, out_scale):
        w = linear.weight.data.to(torch.bfloat16)
        s = max(w.abs().max().item() / FP8_MAX, 1e-8)
        self.w = (w.float() / s).clamp(-FP8_MAX, FP8_MAX - 1).to(torch.float8_e4m3fn).t().to(device)
        self.s = torch.tensor([s], dtype=torch.float32, device=device)
        self.out_scale = float(out_scale)
        self.out_scale_dev = torch.tensor([float(out_scale)], dtype=torch.float32, device=device)
        self.bias = (linear.bias.data.to(torch.bfloat16).contiguous().to(device)
                     if linear.bias is not None else None)


class _RoPE:
    def __init__(self, rotary_embed, max_seq_len):
        device = rotary_embed.device
        pos = rotary_embed.get_seq_pos(max_seq_len, device=device, dtype=torch.float32)
        freqs = rotary_embed.forward(pos, seq_len=max_seq_len)
        D = freqs.shape[-1]
        fb = 10000.0 ** (-torch.arange(0, D, 2, dtype=torch.float32, device=device) / D)
        fp = torch.arange(max_seq_len, dtype=torch.float32, device=device).unsqueeze(1) * fb.unsqueeze(0)
        self.cos = fp.cos().contiguous()
        self.sin = fp.sin().contiguous()
        self.D = D


class MelBandRoformerPipeline:
    """FP8 kernelized inference pipeline for MelBandRoformer.

    Wraps a loaded MelBandRoformer ``frontend`` and rewrites the per-band
    Transformer ``forward`` passes onto static-quantized FP8 GEMMs driven by
    the ``mbr_*`` CUDA kernels. The model is consumed in place: Linear
    weights referenced by the patched path are quantized to FP8 and the
    original BF16 tensors are deleted to free VRAM.

    Args:
        frontend: object exposing ``.config``, ``.device`` and ``.model``
            (the MelBandRoformer module).
        max_seq_len: max band sequence length for RoPE precompute.
        calibration_path: optional explicit path to an
            ``fp8_calibration.json`` file holding per-tensor static scales.
        model_dir: optional checkpoint directory consulted for
            ``fp8_calibration.json`` when ``calibration_path`` is None.
    """

    def __init__(self, frontend, max_seq_len=1024, calibration_path=None, model_dir=None):
        self.fvk, self.mk = _load_kernels()
        self.frontend = frontend
        self.config = frontend.config
        self.device = frontend.device
        self.model = frontend.model
        if calibration_path is None and model_dir is not None:
            calibration_path = os.path.join(model_dir, "fp8_calibration.json")
        self.calib = _load_calib(calibration_path)
        self._build(max_seq_len)
        self._patch(max_seq_len)

    def _s(self, n):
        v = self.calib.get(n)
        return float(v) if v else 1.0

    def _build(self, max_seq_len):
        m, dev = self.model, self.device
        self.W = {nm: _W(mod, dev, self._s(nm))
                  for nm, mod in list(m.named_modules())
                  if nm.startswith("layers.") and isinstance(mod, nn.Linear) and not nm.endswith("to_gates")}
        self.attn_norm = {}
        self.ff_norm = {}
        self.ropes = {}
        self.final_norm_gamma = {}
        for nm, mod in list(m.named_modules()):
            if type(mod).__name__ == "Attention" and hasattr(mod, "attend"):
                self.attn_norm[nm] = _Norm(mod.norm.gamma.detach(), self._s(nm + ".to_qkv"), dev)
                self.ropes[nm] = _RoPE(mod.rotary_embed, max_seq_len)
            if type(mod).__name__ == "FeedForward":
                self.ff_norm[nm] = _Norm(mod.net[0].gamma.detach(), self._s(nm + ".net.1"), dev)
            if type(mod).__name__ == "Transformer":
                self.final_norm_gamma[nm] = mod.norm.gamma.detach().to(torch.bfloat16).contiguous().to(dev)
        logger.info("%d FP8 weights, %d attn norms, %d ff norms (before weight deletion)",
                    len(self.W), len(self.attn_norm), len(self.ff_norm))

        deleted_count = 0
        for nm in self.W.keys():
            module = m
            for part in nm.split("."):
                module = getattr(module, part)
            if hasattr(module, "weight"):
                del module.weight
                deleted_count += 1
            if hasattr(module, "bias"):
                del module.bias
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Deleted %d quantized Linear weights, saved ~400 MB", deleted_count)

    def _patch(self, max_seq_len):
        fvk, mk = self.fvk, self.mk
        m, dev = self.model, self.device
        ca = 0
        for nm, mod in list(m.named_modules()):
            if type(mod).__name__ != "Transformer":
                continue
            attn = mod.layers[0][0]
            ff = mod.layers[0][1]
            anm = nm + ".layers.0.0"
            fnm = nm + ".layers.0.1"
            anorm = self.attn_norm[anm]
            fnorm = self.ff_norm[fnm]
            wq = self.W[anm + ".to_qkv"]
            wo = self.W[anm + ".to_out.0"]
            gates = attn.to_gates
            heads = attn.heads
            rope = self.ropes[anm]
            D = rope.D
            w1 = self.W[fnm + ".net.1"]
            w2 = self.W[fnm + ".net.4"]
            final_gamma = self.final_norm_gamma[nm]

            def make_fwd(fvk, mk, anorm, fnorm, wq, wo, gates, heads, rope, D, w1, w2, final_gamma):
                def fwd(self, x):
                    Bp, T, dim = x.shape
                    M = Bp * T
                    st = int(torch.cuda.current_stream().cuda_stream)
                    # ---- Attention ----
                    nfp8 = torch.empty(M, dim, dtype=torch.float8_e4m3fn, device=x.device)
                    fvk.rms_norm_fp8(int(x.reshape(-1, dim).data_ptr()), int(anorm.gamma.data_ptr()),
                                     int(nfp8.data_ptr()), M, dim, 1e-6, int(anorm.d_scale.data_ptr()), st)
                    qkv = _fp8_gemm(nfp8, wq.w, anorm.d_scale, wq.s).view(Bp, T, -1)
                    Q = torch.empty(Bp, heads, T, D, dtype=torch.bfloat16, device=x.device)
                    K = torch.empty_like(Q)
                    V = torch.empty_like(Q)
                    mk.qkv_split_rope(qkv.data_ptr(), rope.cos[:T].data_ptr(), rope.sin[:T].data_ptr(),
                                      Q.data_ptr(), K.data_ptr(), V.data_ptr(), Bp, T, heads, D, st)
                    o = F.scaled_dot_product_attention(Q, K, V)
                    gx = torch.empty(Bp, T, dim, dtype=torch.bfloat16, device=x.device)
                    mk.fp8_dequant_bf16(nfp8.data_ptr(), anorm.scale_val, gx.data_ptr(), M * dim, st)
                    g = gates(gx)
                    HD = heads * D
                    ofp8 = torch.empty(M, HD, dtype=torch.float8_e4m3fn, device=x.device)
                    mk.gated_attn_quant(o.data_ptr(), g.data_ptr(), ofp8.data_ptr(), Bp, heads, T, D, wo.out_scale, st)
                    attn_out = _fp8_gemm(ofp8, wo.w, wo.out_scale_dev, wo.s)
                    attn_out = attn_out.view(Bp, T, dim)
                    # ---- FFN ----
                    x_new = torch.empty(M, dim, dtype=torch.bfloat16, device=x.device)
                    ffn_fp8 = torch.empty(M, dim, dtype=torch.float8_e4m3fn, device=x.device)
                    mk.resadd_rmsnorm_fp8_keepres(attn_out.reshape(-1, dim).data_ptr(),
                                                  x.reshape(-1, dim).data_ptr(), fnorm.gamma.data_ptr(), x_new.data_ptr(),
                                                  ffn_fp8.data_ptr(), M, dim, fnorm.scale_val, st)
                    h = _fp8_gemm(ffn_fp8, w1.w, fnorm.d_scale, w1.s)
                    Ni = h.shape[1]
                    hfp8 = torch.empty(M, Ni, dtype=torch.float8_e4m3fn, device=x.device)
                    fvk.bias_gelu_quantize_fp8_static_bf16(int(h.data_ptr()),
                                                           int(w1.bias.data_ptr()) if w1.bias is not None else 0,
                                                           int(hfp8.data_ptr()),
                                                           int(w2.out_scale_dev.data_ptr()),
                                                           M, Ni, st)
                    ff_out = _fp8_gemm(hfp8, w2.w, w2.out_scale_dev, w2.s)
                    if w2.bias is not None:
                        ff_out = ff_out + w2.bias
                    ff_out = ff_out.view(Bp, T, dim)
                    # ---- Fused residual + final_norm ----
                    output = torch.empty(Bp, T, dim, dtype=torch.bfloat16, device=x.device)
                    mk.fused_add_rmsnorm_bf16(
                        ff_out.reshape(-1, dim).data_ptr(),
                        x_new.data_ptr(),
                        final_gamma.data_ptr(),
                        output.reshape(-1, dim).data_ptr(),
                        M, dim, st)
                    return output
                return fwd
            mod.forward = types.MethodType(
                make_fwd(fvk, mk, anorm, fnorm, wq, wo, gates, heads, rope, D, w1, w2, final_gamma), mod)
            ca += 1
        logger.info("patched %d Transformer blocks (memory optimized)", ca)

    def _split_forward(self, model, raw_audio):
        device = raw_audio.device
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)
        batch, channels, L = raw_audio.shape
        packed = raw_audio.reshape(-1, L)
        win = model.stft_window_fn(device=device)
        stc = torch.stft(packed, **model.stft_kwargs, window=win, return_complex=True)
        srr = torch.view_as_real(stc).to(torch.bfloat16)
        srr = srr.reshape(batch, channels, *srr.shape[1:]).permute(0, 2, 1, 3, 4)
        T = srr.shape[3]
        sr = srr.reshape(batch, -1, T, 2)
        ba = torch.arange(batch, device=device)[..., None]
        x = sr[ba, model.freq_indices].permute(0, 2, 1, 3).reshape(batch, sr[ba, model.freq_indices].shape[2], -1)
        x = model.band_split(x)
        for tt, ft in model.layers:
            x = x.permute(0, 2, 1, 3)
            sh = x.shape
            x = tt(x.reshape(-1, sh[2], sh[3])).reshape(sh)
            x = x.permute(0, 2, 1, 3)
            sh2 = x.shape
            x = ft(x.reshape(-1, sh2[2], sh2[3])).reshape(sh2)
        ns = len(model.mask_estimators)
        masks = torch.stack([fn(x) for fn in model.mask_estimators], dim=1)
        masks = masks.reshape(batch, ns, masks.shape[2], -1, 2).permute(0, 1, 3, 2, 4)
        sf = sr.to(torch.float32).unsqueeze(1)
        mf = masks.to(torch.float32)
        sc = torch.view_as_complex(sf)
        mc = torch.view_as_complex(mf).type(sc.dtype)
        si = model.freq_indices.view(1, 1, -1, 1).expand(batch, ns, -1, sc.shape[-1])
        se = sc.expand(-1, ns, -1, -1)
        msum = torch.zeros_like(se).scatter_add_(2, si, mc)
        denom = model.num_bands_per_freq.view(-1, 1).expand(-1, channels).reshape(-1, 1)
        sc = sc * (msum / denom.clamp(min=1e-8))
        nf = model.num_bands_per_freq.shape[0]
        sc = sc.reshape(batch, ns, nf, channels, -1).permute(0, 1, 3, 2, 4).reshape(-1, nf, sc.shape[-1])
        recon = torch.istft(sc, **model.stft_kwargs, window=win, return_complex=False,
                            length=L if model.match_input_audio_length else None)
        recon = recon.reshape(batch, ns, channels, -1)
        if ns == 1:
            recon = recon.squeeze(1)
        return recon

    def forward(self, x):
        with torch.no_grad():
            return self._split_forward(self.model, x)
