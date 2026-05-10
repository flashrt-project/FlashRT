"""FlashRT — Thor JAX frontend for Pi0 (SM110).

Loads JAX Orbax checkpoints → FP8 quantize (ml_dtypes) → CudaBuffer (cudaMalloc)
→ pipeline_pi0.py (flash_rt_kernels.so) on Jetson AGX Thor.

Adapted from thor_jax_pi05.py with Pi0-specific changes:
  - Standard RMSNorm (fused into QKV/GateUp) instead of AdaRMSNorm
  - action_time_mlp (split W into action/time parts) replaces time_mlp + AdaRMS
  - state_proj: continuous state → single suffix token
  - S_dec = Sa + 1 (1 state + Sa actions, no padding)
  - State-masked attention (state only attends to prefix + self)

Usage:
    pipe = Pi0JaxFrontendThor("/path/to/orbax/checkpoint", num_views=2)
    pipe.set_prompt("pick up the red cup")
    result = pipe.infer({"image": img1, "wrist_image": img2, "state": state_vec})
    actions = result["actions"]  # (10, 7) numpy
"""

import ctypes
import json
import logging
import math
import os
import pathlib
import time as _time

import numpy as np

from flash_rt.hardware.thor.shared_primitives import (
    siglip_forward,
    postln_project,
    encoder_forward,
    encoder_forward_calibrate,
)
from flash_rt.models.pi0.pipeline_thor import (
    decoder_forward_pi0,
    decoder_forward_calibrate_pi0,
)
from flash_rt.hardware.thor.attn_backend import (
    ThorFlashAttnBackend,
    make_pi0_attention_spec,
)

logger = logging.getLogger(__name__)

# XLA flags must be set before JAX import
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_triton_gemm" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        _xla_flags + " --xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"
    ).strip()

if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp


from flash_rt.core.thor_frontend_utils import embed_prompt_numpy as _embed_prompt  # noqa: E402


class Pi0JaxFrontendThor:
    """Thor SM110 — JAX (Orbax) frontend for Pi0. CudaBuffer bridge to flash_rt_kernels."""

    def __init__(self, checkpoint_dir, engine_path=None, fmha_path=None,
                 use_cuda_graph=True, num_views=2, autotune=3,
                 weight_cache=True, use_fp8=True, **kwargs):
        from flash_rt.core.cuda_buffer import CudaBuffer, sync
        from flash_rt.core.weights.transformer import quantize_fp8_e4m3, compute_time_embeddings
        self._CudaBuffer = CudaBuffer
        self._sync = sync
        self.use_fp8 = bool(use_fp8)

        checkpoint_dir = pathlib.Path(checkpoint_dir)

        # ── Load norm stats ──
        self.norm_stats = None
        for p in [checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
                  checkpoint_dir / "norm_stats.json"]:
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                self.norm_stats = data.get("norm_stats", data)
                break

        # ── FvkContext + GemmRunner + FMHA ──
        from flash_rt import flash_rt_kernels as _fvk
        self._fvk = _fvk
        self._ctx = _fvk.FvkContext()
        self._gemm = _fvk.GemmRunner()

        if fmha_path is None:
            # Search order matches the torch Thor frontends: ckpt-adjacent,
            # ``flash_rt/`` package dir (pip / editable install target),
            # fresh cmake ``build/`` output, docker ``/workspace/``.
            _here = pathlib.Path(__file__)
            for c in [str(checkpoint_dir.parent / 'libfmha_fp16_strided.so'),
                      str(_here.parent.parent.parent / 'libfmha_fp16_strided.so'),
                      str(_here.parent.parent.parent.parent / 'build'
                          / 'libfmha_fp16_strided.so'),
                      '/workspace/libfmha_fp16_strided.so']:
                if os.path.exists(c):
                    fmha_path = c
                    break
        if fmha_path:
            _fvk.load_fmha_strided_library(fmha_path)

        # ── Load weights ──
        import gc
        self.num_views = num_views
        cache_hit = False

        if weight_cache:
            from flash_rt.core.weights.weight_cache import load_weight_cache
            cached = load_weight_cache(str(checkpoint_dir), num_views)
            if cached is not None:
                self._load_from_cache(cached)
                cache_hit = True

        if not cache_hit:
            from flash_rt.core.weights.loader import load_weights, detect_format
            from flash_rt.core.weights.transformer import transform_jax_weights_pi0

            fmt = detect_format(str(checkpoint_dir))
            raw = load_weights(str(checkpoint_dir), format=fmt)
            engine_w = transform_jax_weights_pi0(raw)
            del raw; gc.collect()

            self._upload_weights(engine_w, quantize_fp8_e4m3, compute_time_embeddings)
            del engine_w; gc.collect()

            if weight_cache:
                self._save_to_cache(str(checkpoint_dir))

        self._prefetch_weights()

        # ── State ──
        self._checkpoint_path = str(checkpoint_dir)
        self.current_prompt = None
        self.calibrated = False
        self._real_data_calibrated = False
        self.autotune = int(autotune) if autotune is not True else 3
        if autotune is False:
            self.autotune = 0
        self._rng_key = jax.random.PRNGKey(0)
        logger.info("JAX Pi0 backend initialized (cache_hit=%s)", cache_hit)

    def _prefetch_weights(self):
        logger.info("Weights ready (managed memory)")

    @staticmethod
    def _gpu_quantize_fp8(w_np):
        """FP8 E4M3 quantization on GPU via JAX."""
        import ml_dtypes
        w_jax = jnp.array(w_np, dtype=jnp.float32)
        amax = float(jnp.abs(w_jax).max())
        scale = max(amax / 448.0, 1e-12)
        w_scaled = jnp.clip(w_jax / scale, -448.0, 448.0)
        fp8_jax = w_scaled.astype(ml_dtypes.float8_e4m3fn)
        return fp8_jax.view(jnp.uint8), scale

    # -----------------------------------------------------------------------
    # Weight upload
    # -----------------------------------------------------------------------

    def _upload_weights(self, engine_w, quantize_fp8, compute_time_embeddings):
        CB = self._CudaBuffer
        fp16 = np.float16
        qfp8 = self._gpu_quantize_fp8

        # ── Declarative weight-loader pass (stage 7.7) ──
        # Populates self._sig_*_cb (12), self._sig_scales, self._enc_*_cb,
        # self._enc_ws, self.dec_{qkv,o,gu,d}_flat, self._ae_ws. Mutates
        # self._cache_blobs with the per-slot cache keys.
        self._cache_blobs = {}

        from flash_rt.executors.jax_weights import OrbaxDictSource
        from flash_rt.executors.weight_loader import WeightLoader
        from flash_rt.frontends.jax._pi0_thor_spec import build_spec
        WeightLoader(source=OrbaxDictSource(engine_w),
                     target=self, spec=build_spec()).run()

        # ── Vision (SigLIP) — compose 12-slot list ──
        S, D, H, NH, HD, L = self.num_views * 256, 1152, 4304, 16, 72, 27
        self.sig_dims = (S, D, H, NH, HD, L)

        self.sig_wt_fp8 = [
            self._sig_ln1w_cb, self._sig_ln1b_cb,
            self._sig_qw_cb,   self._sig_qb_cb,
            self._sig_ow_cb,   self._sig_ob_cb,
            self._sig_ln2w_cb, self._sig_ln2b_cb,
            self._sig_uw_cb,   self._sig_ub_cb,
            self._sig_dw_cb,   self._sig_db_cb,
        ]
        self.sig_fp8_scratch = CB.device_zeros(S * D, np.uint8)
        self.sig_hid_fp8 = CB.device_zeros(S * H, np.uint8)
        self.sig_scales_c = (ctypes.c_float * len(self._sig_scales))(*self._sig_scales)
        self.sig_buf = [CB.device_empty(S * 3 * D, fp16),
                        CB.device_empty(S * D, fp16),
                        CB.device_empty(S * H, fp16)]
        self.sig_scratch = CB.device_empty(S * max(D, H), fp16)

        # Patch embedding
        pe_w_np = engine_w["vision.patch_embed.weight"].astype(fp16)
        pe_oihw = pe_w_np.transpose(3, 2, 0, 1)
        pe_w_2d_np = pe_oihw.transpose(0,2,3,1).reshape(pe_oihw.shape[0], -1).T.copy().astype(fp16)
        pe_b_np = engine_w["vision.patch_embed.bias"].astype(fp16).copy()
        pos_emb_np = engine_w["vision.pos_embed"][:256].astype(fp16).copy()
        self.pe_w_buf = CB.from_numpy(pe_w_2d_np)
        self.pe_b_buf = CB.from_numpy(pe_b_np)
        self.pos_emb_buf = CB.from_numpy(pos_emb_np)
        self._cache_blobs["pe_w_buf"] = pe_w_2d_np.tobytes()
        self._cache_blobs["pe_b_buf"] = pe_b_np.tobytes()
        self._cache_blobs["pos_emb_buf"] = pos_emb_np.tobytes()
        nv = self.num_views
        self.img_buf = CB.device_empty(nv * 224 * 224 * 3, fp16)
        self.patches_buf = CB.device_empty(S * 588, fp16)

        # PostLN
        _flnw = engine_w["vision.final_norm.weight"].astype(fp16)
        _flnb = engine_w["vision.final_norm.bias"].astype(fp16)
        _projw = engine_w["vision.projector.weight"].T.copy().astype(fp16)
        _projb = engine_w["vision.projector.bias"].astype(fp16)
        self.final_ln_w = CB.from_numpy(_flnw)
        self.final_ln_b = CB.from_numpy(_flnb)
        self.proj_w = CB.from_numpy(_projw)
        self.mm_b = CB.from_numpy(_projb)
        self._cache_blobs["final_ln_w"] = _flnw.tobytes()
        self._cache_blobs["final_ln_b"] = _flnb.tobytes()
        self._cache_blobs["proj_w"] = _projw.tobytes()
        self._cache_blobs["mm_b"] = _projb.tobytes()

        # ── Encoder — identical to Pi0.5 ──
        Se_max, De, He, Le = self.num_views * 256 + 256, 2048, 16384, 18
        NHe, HDe = 8, 256
        self.Se_max = Se_max; self.De = De; self.He = He; self.Le = Le
        self.NHe = NHe; self.HDe = HDe

        # Encoder per-layer flat CudaBuffers populated by loader.
        self.ew = [self._enc_qkv_cb, self._enc_o_cb, self._enc_gu_cb,
                   CB.device_zeros(1, np.uint8),
                   self._enc_d_cb]
        _enc_ws_np = np.array(self._enc_ws, dtype=np.float32)
        self.enc_w_dev = CB.from_numpy_managed(_enc_ws_np)
        self._cache_blobs["ew.3"] = b'\x00'
        self._cache_blobs["enc_w_dev"] = _enc_ws_np.tobytes()

        # RoPE
        inv_freq = 1.0 / (10000 ** (np.arange(0, 256, 2, dtype=np.float64) / 256))
        pos = np.arange(1200, dtype=np.float64)
        kp = pos[:, None] * inv_freq[None, :]
        self._kc_t = np.cos(kp).astype(fp16)
        self._ks_t = np.sin(kp).astype(fp16)

        # ── Decoder — Pi0-specific ──
        Sa, Da, Ha, La = 10, 1024, 4096, 18
        steps = 10
        S_dec = Sa + 1  # 1 state + Sa actions
        self.Sa = Sa; self.Da = Da; self.Ha = Ha; self.La = La
        self.S_dec = S_dec
        total_keys_max = Se_max + S_dec

        self.Kc = CB.device_zeros(Le * total_keys_max * HDe, fp16)
        self.Vc = CB.device_zeros(Le * total_keys_max * HDe, fp16)

        # Encoder buffers
        self.enc_buf = [
            CB.device_zeros(Se_max * De, np.uint8),
            CB.device_empty(Se_max * 2560, fp16),
            CB.device_empty(Se_max * NHe * Se_max, fp16),
            CB.device_empty(Se_max * NHe * HDe, fp16),
            CB.device_empty(Se_max * De, fp16),
            CB.device_zeros(Se_max * De, np.uint8),
            CB.device_empty(Se_max * 2 * He, fp16),
            CB.device_empty(Se_max * He, fp16),
            CB.device_empty(Se_max * He, fp16),
            CB.device_zeros(Se_max * He, np.uint8),
            CB.device_empty(Se_max * De, fp16),
        ]
        self.enc_act_amax = CB.device_zeros(1, np.float32)
        self.enc_norm_buf = CB.device_empty(Se_max * max(De, He), fp16)
        self.enc_x = CB.device_empty(Se_max * De, fp16)
        self.enc_rope = CB.device_empty(Se_max * 256, fp16)

        # Decoder weights (self.dec_{qkv,o,gu,d}_flat) populated by loader.

        self._embedding_np = engine_w["encoder.embedding"].astype(fp16)

        _ainw = engine_w["action.in_proj.weight"].astype(fp16)
        _ainb = engine_w["action.in_proj.bias"].astype(fp16)
        _aoww = engine_w["action.out_proj.weight"].astype(fp16)
        _aobb = engine_w["action.out_proj.bias"].astype(fp16)
        _ae_ws_np = np.array(self._ae_ws, dtype=np.float32)
        self.ain_w = CB.from_numpy(_ainw)
        self.ain_b = CB.from_numpy(_ainb)
        self.aow = CB.from_numpy(_aoww)
        self.aob = CB.from_numpy(_aobb)
        self.ae_w_dev = CB.from_numpy(_ae_ws_np)
        self._cache_blobs["ain_w"] = _ainw.tobytes()
        self._cache_blobs["ain_b"] = _ainb.tobytes()
        self._cache_blobs["aow"] = _aoww.tobytes()
        self._cache_blobs["aob"] = _aobb.tobytes()
        self._cache_blobs["ae_w_dev"] = _ae_ws_np.tobytes()

        # Pi0-specific: action_time_mlp weights
        _atmlp_wa = engine_w["action_time_mlp.wa"].astype(fp16)   # (Da, Da) for GEMM
        _atmlp_out_w = engine_w["action_time_mlp.out_w"].astype(fp16)  # (Da, Da)
        _atmlp_out_b = engine_w["action_time_mlp.out_bias"].astype(fp16)
        self.atmlp_in_wa = CB.from_numpy(_atmlp_wa)
        self.atmlp_out_w = CB.from_numpy(_atmlp_out_w)
        self.atmlp_out_b = CB.from_numpy(_atmlp_out_b)
        self._cache_blobs["atmlp_in_wa"] = _atmlp_wa.tobytes()
        self._cache_blobs["atmlp_out_w"] = _atmlp_out_w.tobytes()
        self._cache_blobs["atmlp_out_b"] = _atmlp_out_b.tobytes()

        # W_time + bias kept as numpy for set_prompt precompute
        self._atmlp_in_wt = engine_w["action_time_mlp.wt"].astype(fp16)  # (Da, Da)
        self._atmlp_in_b = engine_w["action_time_mlp.in_bias"].astype(fp16)

        # state_proj
        _sp_w = engine_w["state_proj.weight"].astype(fp16)  # (32, Da)
        _sp_b = engine_w["state_proj.bias"].astype(fp16)
        self.state_proj_w = CB.from_numpy(_sp_w)
        self.state_proj_b = CB.from_numpy(_sp_b)
        self._cache_blobs["state_proj_w"] = _sp_w.tobytes()
        self._cache_blobs["state_proj_b"] = _sp_b.tobytes()

        # Final norm: standard RMSNorm with weight (1+scale, pre-fused)
        _fnw = engine_w["decoder.final_norm.weight"].astype(fp16)
        self.final_norm_w = CB.from_numpy(_fnw)
        self._cache_blobs["final_norm_w"] = _fnw.tobytes()

        # dec_{qkv,o,gu,d}_flat already populated by loader.

        # Decoder buffers (S_dec = 11)
        self.ae_buf = [
            CB.device_empty(S_dec * Da, fp16),             # x
            CB.device_empty(S_dec * Da, fp16),             # xn
            CB.device_empty(S_dec * Da, fp16),             # gate
            CB.device_empty(S_dec * 2560, fp16),           # qkv
            CB.device_empty(S_dec * 8 * total_keys_max, fp16),  # logits
            CB.device_empty(S_dec * 8 * 256, fp16),        # attn_out
            CB.device_empty(S_dec * 2 * Ha, fp16),         # hid
            CB.device_empty(S_dec * 2 * Ha, fp16),         # fg
            CB.device_zeros(S_dec * Da, np.uint8),         # xn_fp8
            CB.device_zeros(S_dec * Ha, np.uint8),         # hid_fp8
            CB.device_zeros(S_dec * 8 * 256, np.uint8),    # ctx_fp8
        ]
        self.ae_temp = CB.device_empty(Sa * Da, fp16)      # action_time_mlp scratch
        self.state_token = CB.device_zeros(1 * Da, fp16)   # state_proj output
        self.state_buf = CB.empty(1 * 32, fp16)            # managed: host upload

        self.dec_rope = CB.device_empty(S_dec * 256, fp16)
        self.g_xs = CB.device_zeros(S * D, fp16)
        self.g_noise = CB.empty(Sa * 32, fp16)

        unit_scale_np = np.array([1.0], dtype=np.float32)
        self._unit_scale_buf = CB.from_numpy(unit_scale_np)
        self._unit_scale_ptr = self._unit_scale_buf.ptr.value

        logger.info("JAX Pi0 backend weights uploaded to CudaBuffer")

    # -----------------------------------------------------------------------
    # FP8 Weight Cache
    # -----------------------------------------------------------------------

    _CACHE_GPU_BUFFERS = [
        ("sig_wt_fp8", True, False),
        ("pe_w_buf", False, False),
        ("pe_b_buf", False, False),
        ("pos_emb_buf", False, False),
        ("final_ln_w", False, False),
        ("final_ln_b", False, False),
        ("proj_w", False, False),
        ("mm_b", False, False),
        ("ew", True, False),
        ("enc_w_dev", False, True),
        ("dec_qkv_flat", False, False),
        ("dec_o_flat", False, False),
        ("dec_gu_flat", False, False),
        ("dec_d_flat", False, False),
        ("ae_w_dev", False, False),
        ("ain_w", False, False),
        ("ain_b", False, False),
        ("aow", False, False),
        ("aob", False, False),
        # Pi0-specific
        ("atmlp_in_wa", False, False),
        ("atmlp_out_w", False, False),
        ("atmlp_out_b", False, False),
        ("state_proj_w", False, False),
        ("state_proj_b", False, False),
        ("final_norm_w", False, False),
    ]

    _CACHE_NUMPY = [
        "_embedding_np", "_atmlp_in_wt", "_atmlp_in_b", "_kc_t", "_ks_t",
    ]

    def _save_to_cache(self, checkpoint_path):
        from flash_rt.core.weights.weight_cache import save_weight_cache

        entries = []
        blobs = []

        for attr, is_list, _ in self._CACHE_GPU_BUFFERS:
            if is_list:
                i = 0
                while f"{attr}.{i}" in self._cache_blobs:
                    data = self._cache_blobs[f"{attr}.{i}"]
                    entries.append({"name": f"{attr}.{i}", "dtype": "uint8",
                                    "shape": [len(data)]})
                    blobs.append(data)
                    i += 1
            else:
                data = self._cache_blobs[attr]
                entries.append({"name": attr, "dtype": "uint8",
                                "shape": [len(data)]})
                blobs.append(data)

        sig_n = len(self.sig_scales_c)
        sig_arr = np.array([self.sig_scales_c[i] for i in range(sig_n)], dtype=np.float32)
        entries.append({"name": "sig_scales", "dtype": "float32", "shape": [sig_n]})
        blobs.append(sig_arr.tobytes())

        for attr in self._CACHE_NUMPY:
            arr = getattr(self, attr)
            entries.append({"name": attr, "dtype": str(arr.dtype),
                            "shape": list(arr.shape)})
            blobs.append(np.ascontiguousarray(arr).tobytes())

        dims = {"sig_dims": list(self.sig_dims), "Se_max": self.Se_max,
                "De": self.De, "He": self.He, "Le": self.Le,
                "NHe": self.NHe, "HDe": self.HDe,
                "Sa": self.Sa, "Da": self.Da, "Ha": self.Ha, "La": self.La,
                "S_dec": self.S_dec}
        dims_json = json.dumps(dims).encode("utf-8")
        entries.append({"name": "_dims", "dtype": "json", "shape": [len(dims_json)]})
        blobs.append(dims_json)

        save_weight_cache(checkpoint_path, self.num_views, entries, blobs)
        del self._cache_blobs

    def _load_from_cache(self, cached):
        header, body = cached
        CB = self._CudaBuffer
        fp16 = np.float16

        lookup = {}
        for e in header["entries"]:
            lookup[e["name"]] = (e["offset"], e["nbytes"], e["dtype"], e["shape"])

        def _get_bytes(name):
            off, nb, _, _ = lookup[name]
            return body[off:off+nb]

        def _get_numpy(name):
            off, nb, dt, sh = lookup[name]
            return np.frombuffer(body[off:off+nb], dtype=np.dtype(dt)).reshape(sh).copy()

        def _get_device_buf(name):
            data = _get_bytes(name)
            buf = CB.device_empty(len(data), np.uint8)
            buf.upload(np.frombuffer(data, dtype=np.uint8))
            return buf

        def _get_managed_buf(name):
            data = _get_bytes(name)
            buf = CB.empty(len(data), np.uint8, managed=True)
            buf.upload(np.frombuffer(data, dtype=np.uint8))
            return buf

        for attr, is_list, is_managed in self._CACHE_GPU_BUFFERS:
            load_fn = _get_managed_buf if is_managed else _get_device_buf
            if is_list:
                bufs = []
                i = 0
                while f"{attr}.{i}" in lookup:
                    bufs.append(load_fn(f"{attr}.{i}"))
                    i += 1
                setattr(self, attr, bufs)
            else:
                setattr(self, attr, load_fn(attr))

        sig_arr = _get_numpy("sig_scales")
        self.sig_scales_c = (ctypes.c_float * len(sig_arr))(*sig_arr.tolist())

        dims = json.loads(_get_bytes("_dims").decode("utf-8"))
        self.sig_dims = tuple(dims["sig_dims"])
        for k in ["Se_max", "De", "He", "Le", "NHe", "HDe",
                   "Sa", "Da", "Ha", "La", "S_dec"]:
            setattr(self, k, dims[k])

        for attr in self._CACHE_NUMPY:
            setattr(self, attr, _get_numpy(attr))

        # Allocate scratch/working buffers
        S, D, H, NH, HD, L = self.sig_dims
        nv = self.num_views
        Se_max = self.Se_max; De = self.De; He = self.He; Le = self.Le
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha; La = self.La
        S_dec = self.S_dec
        total_keys_max = Se_max + S_dec

        self.sig_fp8_scratch = CB.device_zeros(S * D, np.uint8)
        self.sig_hid_fp8 = CB.device_zeros(S * H, np.uint8)
        self.sig_buf = [CB.device_empty(S * 3 * D, fp16),
                        CB.device_empty(S * D, fp16),
                        CB.device_empty(S * H, fp16)]
        self.sig_scratch = CB.device_empty(S * max(D, H), fp16)
        self.img_buf = CB.device_empty(nv * 224 * 224 * 3, fp16)
        self.patches_buf = CB.device_empty(S * 588, fp16)

        self.Kc = CB.device_zeros(Le * total_keys_max * HDe, fp16)
        self.Vc = CB.device_zeros(Le * total_keys_max * HDe, fp16)
        self.enc_buf = [
            CB.device_zeros(Se_max * De, np.uint8),
            CB.device_empty(Se_max * 2560, fp16),
            CB.device_empty(Se_max * NHe * Se_max, fp16),
            CB.device_empty(Se_max * NHe * HDe, fp16),
            CB.device_empty(Se_max * De, fp16),
            CB.device_zeros(Se_max * De, np.uint8),
            CB.device_empty(Se_max * 2 * He, fp16),
            CB.device_empty(Se_max * He, fp16),
            CB.device_empty(Se_max * He, fp16),
            CB.device_zeros(Se_max * He, np.uint8),
            CB.device_empty(Se_max * De, fp16),
        ]
        self.enc_act_amax = CB.device_zeros(1, np.float32)
        self.enc_norm_buf = CB.device_empty(Se_max * max(De, He), fp16)
        self.enc_x = CB.device_empty(Se_max * De, fp16)
        self.enc_rope = CB.device_empty(Se_max * 256, fp16)

        self.ae_buf = [
            CB.device_empty(S_dec * Da, fp16),
            CB.device_empty(S_dec * Da, fp16),
            CB.device_empty(S_dec * Da, fp16),
            CB.device_empty(S_dec * 2560, fp16),
            CB.device_empty(S_dec * 8 * total_keys_max, fp16),
            CB.device_empty(S_dec * 8 * 256, fp16),
            CB.device_empty(S_dec * 2 * Ha, fp16),
            CB.device_empty(S_dec * 2 * Ha, fp16),
            CB.device_zeros(S_dec * Da, np.uint8),
            CB.device_zeros(S_dec * Ha, np.uint8),
            CB.device_zeros(S_dec * 8 * 256, np.uint8),
        ]
        self.ae_temp = CB.device_empty(Sa * Da, fp16)
        self.state_token = CB.device_zeros(1 * Da, fp16)
        self.state_buf = CB.empty(1 * 32, fp16)

        self.dec_rope = CB.device_empty(S_dec * 256, fp16)
        self.g_xs = CB.device_zeros(S * D, fp16)
        self.g_noise = CB.empty(Sa * 32, fp16)

        unit_scale_np = np.array([1.0], dtype=np.float32)
        self._unit_scale_buf = CB.from_numpy(unit_scale_np)
        self._unit_scale_ptr = self._unit_scale_buf.ptr.value

        logger.info("Weights restored from cache (Pi0)")

    # -----------------------------------------------------------------------
    # set_prompt
    # -----------------------------------------------------------------------

    def set_prompt(self, prompt_text):
        CB = self._CudaBuffer
        fp16 = np.float16

        S = self.sig_dims[0]
        if isinstance(prompt_text, (np.ndarray, list)):
            token_ids = np.asarray(prompt_text, dtype=np.int64)
            prompt_len = len(token_ids)
            embeds_np = self._embedding_np[token_ids]
            embeds_np = (embeds_np * float(embeds_np.shape[-1] ** 0.5)).astype(np.float16)
        else:
            embeds_np, prompt_len = _embed_prompt(prompt_text, self._embedding_np, max_len=48)
        Se = S + prompt_len
        if Se % 2 != 0:
            Se += 1
        self.Se = Se
        self.total_keys = Se + self.S_dec

        # Stage 3.4 — build AttentionBackend (mirrors Pi0 torch stage 3.3).
        # Pi0 decoder uses state-masked kernel via extra={"kernel":"state_masked"};
        # the backend dispatches fvk.attention_qkv_fp16_state_masked with
        # state_nk supplied by decoder_forward_pi0.
        attn_scale = 1.0 / math.sqrt(float(self.HDe))
        layer_stride = int(self.total_keys) * int(self.HDe) * 2  # fp16 bytes
        _sig_D = int(self.sig_dims[1])  # sig_dims = (S, D, H, NH, HD, L)
        self._attn = ThorFlashAttnBackend(
            make_pi0_attention_spec(
                num_views=self.num_views,
                enc_seq_max=self.Se,
                S_dec=self.S_dec,
            ),
            self._ctx,
            siglip_slots={
                "qkv": self.sig_buf[0].ptr.value,
                "O":   self.sig_buf[1].ptr.value,
                "D":   _sig_D,
            },
            encoder_slots={
                "Q_O":          self.enc_buf[3].ptr.value,
                "Kc":           self.Kc.ptr.value,
                "Vc":           self.Vc.ptr.value,
                "logits":       self.enc_buf[2].ptr.value,
                "layer_stride": layer_stride,
                "scale":        attn_scale,
            },
            decoder_slots={
                "Q_O":          self.ae_buf[5].ptr.value,
                "Kc":           self.Kc.ptr.value,
                "Vc":           self.Vc.ptr.value,
                "logits":       self.ae_buf[4].ptr.value,
                "layer_stride": layer_stride,
                "scale":        attn_scale,
            },
        )

        actual_lang = Se - S
        if actual_lang > prompt_len:
            embeds_np = np.concatenate([embeds_np, embeds_np[-1:]], axis=0)
        self.lang_emb = CB.from_numpy(embeds_np[:actual_lang].astype(fp16))
        self.S_lang = actual_lang

        # RoPE
        enc_rope_np = np.concatenate(
            [self._kc_t[:Se, :, None], self._ks_t[:Se, :, None]], 2).reshape(Se, 256)
        self.enc_rope = CB.from_numpy(enc_rope_np.astype(fp16))
        dec_start = Se
        dec_rope_np = np.concatenate(
            [self._kc_t[dec_start:dec_start+self.S_dec, :, None],
             self._ks_t[dec_start:dec_start+self.S_dec, :, None]], 2).reshape(self.S_dec, 256)
        self.dec_rope = CB.from_numpy(dec_rope_np.astype(fp16))

        # ── Pi0 time conditioning: precompute time_proj_all ──
        Sa, Da = self.Sa, self.Da
        time_proj_list = []
        for step in range(10):
            t_val = 1.0 - step / 10
            fraction = np.linspace(0, 1, Da // 2, dtype=np.float64)
            period = 4e-3 * (4.0 / 4e-3) ** fraction
            scaling = 1.0 / period * 2 * math.pi
            sin_input = scaling * t_val
            time_emb = np.concatenate([np.sin(sin_input), np.cos(sin_input)]).astype(np.float16)

            # time_proj = time_emb @ W_time.T + bias → [1, Da]
            tp = (time_emb[None, :] @ self._atmlp_in_wt.T.astype(np.float32)
                  + self._atmlp_in_b[None, :].astype(np.float32)).astype(np.float16)
            tp_expanded = np.broadcast_to(tp, (Sa, Da)).copy()
            time_proj_list.append(tp_expanded)

        time_proj_all_np = np.concatenate(time_proj_list, axis=0).astype(fp16)  # [100, 1024]
        self.time_proj_all = CB.from_numpy(time_proj_all_np)

        # Capture SigLIP graph
        self._capture_siglip_graph()
        import ctypes as _ct
        _ct.CDLL('libcudart.so').cudaDeviceSynchronize()

        # Calibrate
        self._calibrate()

        # Free XLA caches before graph capture
        jax.clear_caches()
        import gc; gc.collect()
        logger.info("XLA caches cleared before graph capture")

        # Capture Enc+AE graph
        if self.autotune > 0:
            self._autotune_enc_ae(n_trials=self.autotune, n_bench=10)
        else:
            self._capture_enc_ae_graph()

        self.current_prompt = prompt_text
        logger.info(f"Set prompt: '{prompt_text}' (Se={Se}, S_dec={self.S_dec})")

    # -----------------------------------------------------------------------
    # Pipeline dicts
    # -----------------------------------------------------------------------

    def _build_enc_dicts(self, stream_int=0):
        Se = self.Se; tk = self.total_keys
        return (
            {'x': self.enc_x.ptr.value, 'x_fp8': self.enc_buf[0].ptr.value,
             'qkv': self.enc_buf[1].ptr.value, 'logits': self.enc_buf[2].ptr.value,
             'attn_out': self.enc_buf[3].ptr.value, 'o_fp8': self.enc_buf[5].ptr.value,
             'gate': self.enc_buf[6].ptr.value, 'hidden': self.enc_buf[7].ptr.value,
             'hid_fp8': self.enc_buf[9].ptr.value, 'fg': self.enc_buf[10].ptr.value,
             'ctx': self._ctx},
            {'qkv_w': [self.ew[0].ptr.value + i * self.De * 2560 for i in range(self.Le)],
             'o_w': [self.ew[1].ptr.value + i * self.De * self.De for i in range(self.Le)],
             'gate_w': [self.ew[2].ptr.value + i * self.De * self.He * 2 for i in range(self.Le)],
             'down_w': [self.ew[4].ptr.value + i * self.He * self.De for i in range(self.Le)],
             'rope': self.enc_rope.ptr.value,
             'Kc': self.Kc.ptr.value, 'Vc': self.Vc.ptr.value,
             'act_scales': self.enc_calib_scales.ptr.value,
             'alpha_host': [float(self.enc_alpha_host[i]) for i in range(self.Le * 4)],
             'w_scales': self.enc_w_dev.ptr.value},
            {'Se': Se, 'D': self.De, 'H': self.He, 'NH': self.NHe, 'HD': self.HDe,
             'L': self.Le, 'total_keys': tk}
        )

    def _build_ae_dicts(self, stream_int=0):
        Da = self.Da; Ha = self.Ha; S_dec = self.S_dec
        return (
            {'noise': self.g_noise.ptr.value,
             'x': self.ae_buf[0].ptr.value,
             'xn': self.ae_buf[1].ptr.value,
             'temp': self.ae_temp.ptr.value,
             'action_buf': self.ae_buf[0].ptr.value + 1 * Da * 2,  # x[1:]
             'gate': self.ae_buf[2].ptr.value,
             'qkv': self.ae_buf[3].ptr.value,
             'logits': self.ae_buf[4].ptr.value,
             'attn_out': self.ae_buf[5].ptr.value,
             'hid': self.ae_buf[6].ptr.value,
             'fg': self.ae_buf[7].ptr.value,
             'xn_fp8': self.ae_buf[8].ptr.value,
             'hid_fp8': self.ae_buf[9].ptr.value,
             'ctx_fp8': self.ae_buf[10].ptr.value,
             'state_token': self.state_token.ptr.value},
            {'ain_w': self.ain_w.ptr.value,
             'ain_b': self.ain_b.ptr.value,
             'wa_w': self.atmlp_in_wa.ptr.value,
             'atmlp_out_w': self.atmlp_out_w.ptr.value,
             'atmlp_out_b': self.atmlp_out_b.ptr.value,
             'time_proj_all': self.time_proj_all.ptr.value,
             'qw': self.dec_qkv_flat.ptr.value,
             'Kc': self.Kc.ptr.value,
             'Vc': self.Vc.ptr.value,
             'ow': self.dec_o_flat.ptr.value,
             'gw': self.dec_gu_flat.ptr.value,
             'dw': self.dec_d_flat.ptr.value,
             'aow': self.aow.ptr.value,
             'aob': self.aob.ptr.value,
             'final_norm_w': self.final_norm_w.ptr.value,
             'rope': self.dec_rope.ptr.value,
             'w_scales': self.ae_w_dev.ptr.value,
             'act_scales': self.ae_calib_scales.ptr.value},
            {'Sa': self.Sa, 'S_dec': S_dec, 'D': Da, 'H': Ha,
             'NH': 8, 'HD': 256, 'steps': 10, 'layers': self.La,
             'enc_seq': self.Se, 'total_keys': self.total_keys}
        )

    # -----------------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------------

    def _calibrate(self):
        from flash_rt.core.quant.calibrator import load_calibration, save_calibration

        CB = self._CudaBuffer
        Se = self.Se; total_keys = self.total_keys
        Le = self.Le; La = self.La
        _cudart = self._cudart; stream = self._stream
        stream_int = stream.value or 0

        cached = load_calibration(self._checkpoint_path, Se)
        if cached is not None:
            enc_scales_np = np.array(cached["enc_scales"], dtype=np.float32)
            ae_scales_np = np.array(cached["ae_scales"], dtype=np.float32)
            self.enc_calib_scales = CB.from_numpy_managed(enc_scales_np)
            self.ae_calib_scales = CB.from_numpy_managed(ae_scales_np)
            enc_ws_np = self.enc_w_dev.download_new((Le * 4,), np.float32)
            self.enc_alpha_host = [
                float(np.float32(enc_scales_np[i]) * np.float32(enc_ws_np[i]))
                for i in range(Le * 4)]
            logger.info("Calibration loaded from cache (enc=%d, ae=%d scales)",
                        Le * 4, La * 4)
            self.calibrated = True
            return

        # ── Cache miss: run dynamic calibration ──
        De = self.De; He = self.He
        Da = self.Da; Ha = self.Ha; S_dec = self.S_dec

        # Encoder calibration
        self.Kc.zero_(stream); self.Vc.zero_(stream)
        _cudart.cudaStreamSynchronize(stream)

        _norm_scratch = CB.device_empty(Se * De, np.float16)
        _x_scratch = CB.device_empty(Se * De, np.float16)
        _enc_calib_buf = CB.zeros(Le * 4, np.float32)
        _d_scale = CB.zeros(1, np.float32)
        _fp8_scratch = CB.device_zeros(Se * max(De, He), np.uint8)
        _ones_buf = CB.from_numpy(np.ones(De, dtype=np.float16))

        enc_scales_buf = CB.zeros(Le * 4, np.float32)
        self.enc_calib_scales = enc_scales_buf

        enc_bufs = {
            'x': self.enc_x.ptr.value, 'x_fp8': self.enc_buf[0].ptr.value,
            'qkv': self.enc_buf[1].ptr.value, 'logits': self.enc_buf[2].ptr.value,
            'attn_out': self.enc_buf[3].ptr.value, 'o_fp8': self.enc_buf[5].ptr.value,
            'gate': self.enc_buf[6].ptr.value, 'hidden': self.enc_buf[7].ptr.value,
            'hid_fp8': self.enc_buf[9].ptr.value, 'fg': self.enc_buf[10].ptr.value,
            'ctx': self._ctx,
            'norm_scratch': _norm_scratch.ptr.value,
            'x_scratch': _x_scratch.ptr.value,
            'calib_buf': _enc_calib_buf.ptr.value,
            'd_scale': _d_scale.ptr.value,
            'fp8_scratch': _fp8_scratch.ptr.value,
            'ones': _ones_buf.ptr.value,
        }
        enc_weights = {
            'qkv_w': [self.ew[0].ptr.value + i * De * 2560 for i in range(Le)],
            'o_w': [self.ew[1].ptr.value + i * De * De for i in range(Le)],
            'gate_w': [self.ew[2].ptr.value + i * De * He * 2 for i in range(Le)],
            'down_w': [self.ew[4].ptr.value + i * He * De for i in range(Le)],
            'rope': self.enc_rope.ptr.value,
            'Kc': self.Kc.ptr.value, 'Vc': self.Vc.ptr.value,
            'w_scales': self.enc_w_dev.ptr.value,
        }
        enc_dims = {'Se': Se, 'D': De, 'H': He, 'NH': self.NHe, 'HD': self.HDe,
                    'L': Le, 'total_keys': total_keys}
        encoder_forward_calibrate(
            self._gemm, self._fvk, enc_bufs, enc_weights, enc_dims,
            enc_scales_buf.ptr.value, stream=stream_int)
        _cudart.cudaStreamSynchronize(stream)

        self.enc_calib_scales = enc_scales_buf
        enc_scales_np = enc_scales_buf.download_new((Le * 4,), np.float32)
        enc_ws_np = self.enc_w_dev.download_new((Le * 4,), np.float32)
        self.enc_alpha_host = [
            float(np.float32(enc_scales_np[i]) * np.float32(enc_ws_np[i]))
            for i in range(Le * 4)]
        logger.info(f"Encoder calibrated: {Le*4} scales")

        # Decoder calibration (Pi0 pattern)
        ae_scales_buf = CB.zeros(La * 4, np.float32)
        self.ae_calib_scales = ae_scales_buf
        _ae_calib_buf = CB.zeros(La * 4, np.float32)
        _ae_d_scale = CB.zeros(1, np.float32)
        _ae_hidden_scratch = CB.device_empty(S_dec * Ha, np.float16)
        _ae_fp8_scratch = CB.device_zeros(S_dec * max(Da, Ha), np.uint8)
        _ae_norm_scratch = CB.device_empty(S_dec * Da, np.float16)
        _ae_x_scratch = CB.device_empty(S_dec * Da, np.float16)
        _ae_ones = CB.from_numpy(np.ones(Da, dtype=np.float16))

        noise_np = np.random.randn(self.Sa, 32).astype(np.float16)
        self.g_noise.upload(noise_np)
        self.state_token.zero_(stream)

        ae_bufs, ae_weights, ae_dims = self._build_ae_dicts(stream_int)
        ae_bufs['calib_buf'] = _ae_calib_buf.ptr.value
        ae_bufs['d_scale'] = _ae_d_scale.ptr.value
        ae_bufs['hidden_scratch'] = _ae_hidden_scratch.ptr.value
        ae_bufs['fp8_scratch'] = _ae_fp8_scratch.ptr.value
        ae_bufs['norm_scratch'] = _ae_norm_scratch.ptr.value
        ae_bufs['x_scratch'] = _ae_x_scratch.ptr.value
        ae_bufs['ones'] = _ae_ones.ptr.value
        decoder_forward_calibrate_pi0(
            self._ctx, self._fvk, ae_bufs, ae_weights, ae_dims,
            ae_scales_buf.ptr.value, stream=stream_int)
        _cudart.cudaStreamSynchronize(stream)

        self.ae_calib_scales = ae_scales_buf
        logger.info(f"Decoder calibrated: {La*4} scales")
        self.calibrated = True

        try:
            save_calibration(
                checkpoint_path=self._checkpoint_path,
                Se=Se,
                enc_scales=enc_scales_np.tolist(),
                enc_alpha=self.enc_alpha_host,
                ae_scales=ae_scales_buf.download_new((La * 4,), np.float32).tolist(),
                enc_w_scales=enc_ws_np.tolist(),
            )
        except Exception as e:
            logger.warning("Failed to save calibration cache: %s", e)

    # -----------------------------------------------------------------------
    # Patch embed + SigLIP + PostLN
    # -----------------------------------------------------------------------

    def _patch_embed_ops(self, stream_int):
        fvk = self._fvk
        S, D = self.sig_dims[0], self.sig_dims[1]
        fvk.patch_im2col(self.img_buf.ptr.value, self.patches_buf.ptr.value,
                          self.num_views, stream_int)
        self._gemm.fp16_nn(self.patches_buf.ptr.value, self.pe_w_buf.ptr.value,
                            self.g_xs.ptr.value, S, D, 588, stream_int)
        fvk.patch_embed_bias_pos(self.g_xs.ptr.value, self.pe_b_buf.ptr.value,
                                  self.pos_emb_buf.ptr.value, S, D, 256, stream_int)

    def _capture_siglip_graph(self):
        from flash_rt.core.cuda_graph import CUDAGraph
        S, D, H, NH, HD, L = self.sig_dims

        _cudart = ctypes.CDLL("libcudart.so")
        stream = ctypes.c_void_p()
        _cudart.cudaStreamCreate(ctypes.byref(stream))
        self._stream = stream
        self._cudart = _cudart
        stream_int = stream.value or 0

        D3 = 3 * D
        sig_bufs = {
            'x': self.g_xs.ptr.value, 'x_fp8': self.sig_fp8_scratch.ptr.value,
            'qkv': self.sig_buf[0].ptr.value, 'attn_out': self.sig_buf[1].ptr.value,
            'hidden': self.sig_buf[2].ptr.value, 'hid_fp8': self.sig_hid_fp8.ptr.value,
        }
        p = [b.ptr.value for b in self.sig_wt_fp8]
        sig_weights = {
            'ln_attn_w': [p[0]  + i * D * 2     for i in range(L)],
            'ln_attn_b': [p[1]  + i * D * 2     for i in range(L)],
            'qkv_w':     [p[2]  + i * D * D3    for i in range(L)],
            'qkv_b':     [p[3]  + i * D3 * 2    for i in range(L)],
            'o_w':       [p[4]  + i * D * D     for i in range(L)],
            'o_b':       [p[5]  + i * D * 2     for i in range(L)],
            'ln_ffn_w':  [p[6]  + i * D * 2     for i in range(L)],
            'ln_ffn_b':  [p[7]  + i * D * 2     for i in range(L)],
            'up_w':      [p[8]  + i * D * H     for i in range(L)],
            'up_b':      [p[9]  + i * H * 2     for i in range(L)],
            'down_w':    [p[10] + i * H * D     for i in range(L)],
            'down_b':    [p[11] + i * D * 2     for i in range(L)],
            'alpha': [float(self.sig_scales_c[i]) for i in range(L * 4)],
            'unit_scale': self._unit_scale_ptr,
        }
        sig_dims = {'S': S, 'D': D, 'H': H, 'NH': NH, 'HD': HD, 'L': L,
                    'num_views': self.num_views, 'seq_per_view': 256}

        postln_bufs = {'x_sig': self.g_xs.ptr.value, 'enc_x': self.enc_x.ptr.value,
                       'scratch': self.sig_scratch.ptr.value}
        postln_weights = {'ln_w': self.final_ln_w.ptr.value, 'ln_b': self.final_ln_b.ptr.value,
                          'proj_w': self.proj_w.ptr.value, 'proj_b': self.mm_b.ptr.value,
                          'lang_emb': self.lang_emb.ptr.value}
        postln_dims = {'S_sig': S, 'D_sig': D, 'D_enc': self.De, 'S_lang': self.S_lang}

        def _run_siglip(st):
            self._patch_embed_ops(st)
            siglip_forward(self._gemm, self._fvk, sig_bufs, sig_weights, sig_dims, st,
                           attn=self._attn)
            postln_project(self._gemm, self._fvk, postln_bufs, postln_weights, postln_dims, st)

        dummy_img = np.random.randn(self.num_views, 224, 224, 3).astype(np.float16)
        self.img_buf.upload(dummy_img)
        for _ in range(3):
            _run_siglip(stream_int)
        _cudart.cudaStreamSynchronize(stream)
        logger.info("Patch+SigLIP warmup done")

        self.siglip_graph = CUDAGraph()
        self.siglip_graph.begin_capture(stream)
        _run_siglip(stream_int)
        self.siglip_graph.end_capture(stream)
        _cudart.cudaStreamSynchronize(stream)
        logger.info("Patch+SigLIP+PostLN CUDA Graph captured")

    # -----------------------------------------------------------------------
    # Enc+AE graph
    # -----------------------------------------------------------------------

    def _capture_enc_ae_graph(self):
        from flash_rt.core.cuda_graph import CUDAGraph
        stream = self._stream; _cudart = self._cudart
        stream_int = stream.value or 0

        enc_bufs, enc_weights, enc_dims = self._build_enc_dicts(stream_int)
        ae_bufs, ae_weights, ae_dims = self._build_ae_dicts(stream_int)

        Da = self.Da
        fvk = self._fvk

        def _run(st):
            self.Kc.zero_(self._stream); self.Vc.zero_(self._stream)
            # Pi0: state_proj GEMM (in graph)
            fvk.gmm_fp16(self._ctx, self.state_buf.ptr.value,
                         self.state_proj_w.ptr.value,
                         self.state_token.ptr.value,
                         1, Da, 32, 0.0, st)
            fvk.add_bias_fp16(self.state_token.ptr.value,
                              self.state_proj_b.ptr.value, 1, Da, st)
            encoder_forward(self._gemm, fvk, enc_bufs, enc_weights, enc_dims, st,
                            attn=self._attn)
            decoder_forward_pi0(self._ctx, fvk, ae_bufs, ae_weights, ae_dims, st,
                                attn=self._attn)

        for _ in range(3):
            _run(stream_int)
        _cudart.cudaStreamSynchronize(stream)

        self.enc_ae_graph = CUDAGraph()
        self.enc_ae_graph.begin_capture(stream)
        _run(stream_int)
        self.enc_ae_graph.end_capture(stream)
        _cudart.cudaStreamSynchronize(stream)
        logger.info(f"Enc+AE CUDA Graph captured (Se={self.Se}, S_dec={self.S_dec})")

    def calibrate(
        self,
        observations,
        *,
        percentile: float = 99.9,
        max_samples=None,
        verbose: bool = False,
    ) -> None:
        """Unified calibration API (JAX Thor).

        ``N=1`` falls back to the implicit single-frame path; ``N>=2``
        runs ``_calibrate_multi_frame``: per-sample shadow forwards
        through encoder + Pi0 decoder, percentile-reduces per-tensor
        amax along the sample axis, uploads the reduced FP8 scales,
        recomputes ``enc_alpha_host`` in float32, and recaptures the
        enc/ae CUDA graph.

        Mirrors :meth:`Pi05JaxFrontendThor._calibrate_multi_frame` with
        the Pi0-specific decoder kernel (``decoder_forward_calibrate_pi0``),
        per-sample state-token projection, and the additional decoder
        scratch buffers (``norm_scratch``, ``x_scratch``, ``ones``).
        """
        if isinstance(observations, dict):
            obs_list = [observations]
        elif isinstance(observations, list):
            obs_list = observations
        else:
            obs_list = list(observations)
        if max_samples is not None:
            obs_list = obs_list[:max_samples]
        n = len(obs_list)
        if n == 0:
            raise ValueError("observations must contain at least 1 sample")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(
                f"percentile must be in [0, 100], got {percentile}")

        if n == 1:
            from flash_rt.core.calibration_api import implicit_calibrate
            implicit_calibrate(
                self, obs_list,
                percentile=percentile, max_samples=None, verbose=verbose,
            )
        else:
            self._calibrate_multi_frame(
                obs_list, percentile=percentile, verbose=verbose)

    def _calibrate_multi_frame(
        self, obs_list, *, percentile: float, verbose: bool,
    ) -> None:
        """N>=2 path — per-sample percentile reduce of FP8 activation scales.

        Runs encoder + Pi0 decoder shadow forwards for each obs in
        ``obs_list``, downloads per-sample amax, percentile-reduces along
        the sample axis via :func:`flash_rt.core.calibration.accumulate_amax`,
        and recaptures the enc+ae CUDA Graph.
        """
        from flash_rt.core.calibration import accumulate_amax

        n = len(obs_list)
        logger.info(
            "Pi0 JAX Thor multi-frame calibrate: N=%d, percentile=%.2f",
            n, percentile)

        CB = self._CudaBuffer
        stream = self._stream
        stream_int = stream.value or 0
        Se = int(self.Se); De = int(self.De); He = int(self.He)
        NHe = int(self.NHe); HDe = int(self.HDe)
        Le = int(self.Le); La = int(self.La)
        Da = int(self.Da); Ha = int(self.Ha); S_dec = int(self.S_dec)
        total_keys = int(self.total_keys)
        nv = self.num_views

        # Encoder scratch buffers (reused across samples).
        _norm_scratch = CB.device_empty(Se * De, np.float16)
        _x_scratch = CB.device_empty(Se * De, np.float16)
        _enc_calib_buf = CB.zeros(Le * 4, np.float32)
        _d_scale = CB.zeros(1, np.float32)
        _fp8_scratch = CB.device_zeros(Se * max(De, He), np.uint8)
        _ones_buf = CB.from_numpy(np.ones(De, dtype=np.float16))

        # Decoder scratch buffers (Pi0-specific extra: norm/x/ones).
        _ae_calib_buf = CB.zeros(La * 4, np.float32)
        _ae_d_scale = CB.zeros(1, np.float32)
        _ae_hidden_scratch = CB.device_empty(S_dec * Ha, np.float16)
        _ae_fp8_scratch = CB.device_zeros(S_dec * max(Da, Ha), np.uint8)
        _ae_norm_scratch = CB.device_empty(S_dec * Da, np.float16)
        _ae_x_scratch = CB.device_empty(S_dec * Da, np.float16)
        _ae_ones = CB.from_numpy(np.ones(Da, dtype=np.float16))

        def _upload_and_siglip(obs):
            if 'images' in obs:
                imgs = obs['images']
            else:
                imgs = [obs['image']]
                if nv >= 2:
                    imgs.append(obs.get('wrist_image', obs['image']))
                if nv >= 3:
                    imgs.append(obs.get('wrist_image_right', imgs[-1]))

            def _to_fp16(im):
                if getattr(im, 'dtype', None) == np.float16:
                    return im
                return (np.asarray(im).astype(np.float32) / 127.5 - 1.0
                        ).astype(np.float16)
            images_np = np.stack([_to_fp16(im) for im in imgs[:nv]])
            self.img_buf.upload(images_np)
            self.siglip_graph.replay(stream)
            self._cudart.cudaStreamSynchronize(stream)

        def _upload_state_and_project(obs):
            # Project per-sample state through state_proj into state_token
            # (mirrors infer() lines that populate state_buf + state_token).
            # Pad/truncate to 32 since state_buf is sized for 32-dim;
            # LIBERO returns 8-dim state, Pi0 expects 32 zero-padded.
            state = np.asarray(
                obs.get('state', np.zeros(32, dtype=np.float32))
            ).astype(np.float32).flatten()
            state_padded = np.zeros(32, dtype=np.float16)
            state_padded[:min(32, state.shape[0])] = state[:32].astype(np.float16)
            self.state_buf.upload(state_padded)
            self._fvk.gmm_fp16(
                self._ctx, self.state_buf.ptr.value,
                self.state_proj_w.ptr.value,
                self.state_token.ptr.value,
                1, Da, 32, 0.0, stream_int)
            self._fvk.add_bias_fp16(
                self.state_token.ptr.value,
                self.state_proj_b.ptr.value, 1, Da, stream_int)

        per_sample_enc: list = []
        per_sample_ae: list = []
        for i, obs in enumerate(obs_list):
            _upload_and_siglip(obs)

            # ── Encoder shadow forward ──
            enc_scales_buf = CB.zeros(Le * 4, np.float32)
            enc_bufs = {
                'x': self.enc_x.ptr.value, 'x_fp8': self.enc_buf[0].ptr.value,
                'qkv': self.enc_buf[1].ptr.value,
                'logits': self.enc_buf[2].ptr.value,
                'attn_out': self.enc_buf[3].ptr.value,
                'o_fp8': self.enc_buf[5].ptr.value,
                'gate': self.enc_buf[6].ptr.value,
                'hidden': self.enc_buf[7].ptr.value,
                'hid_fp8': self.enc_buf[9].ptr.value,
                'fg': self.enc_buf[10].ptr.value,
                'ctx': self._ctx,
                'norm_scratch': _norm_scratch.ptr.value,
                'x_scratch': _x_scratch.ptr.value,
                'calib_buf': _enc_calib_buf.ptr.value,
                'd_scale': _d_scale.ptr.value,
                'fp8_scratch': _fp8_scratch.ptr.value,
                'ones': _ones_buf.ptr.value,
            }
            enc_weights = {
                'qkv_w': [self.ew[0].ptr.value + j * De * 2560
                          for j in range(Le)],
                'o_w': [self.ew[1].ptr.value + j * De * De
                        for j in range(Le)],
                'gate_w': [self.ew[2].ptr.value + j * De * He * 2
                           for j in range(Le)],
                'down_w': [self.ew[4].ptr.value + j * He * De
                           for j in range(Le)],
                'rope': self.enc_rope.ptr.value,
                'Kc': self.Kc.ptr.value, 'Vc': self.Vc.ptr.value,
                'w_scales': self.enc_w_dev.ptr.value,
            }
            enc_dims = {'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
                        'L': Le, 'total_keys': total_keys}
            self.Kc.zero_(stream); self.Vc.zero_(stream)
            encoder_forward_calibrate(
                self._gemm, self._fvk, enc_bufs, enc_weights, enc_dims,
                enc_scales_buf.ptr.value, stream=stream_int)
            self._cudart.cudaStreamSynchronize(stream)
            per_sample_enc.append(
                enc_scales_buf.download_new((Le * 4,), np.float32))

            # ── Decoder shadow forward (Pi0 kernel) ──
            ae_scales_buf = CB.zeros(La * 4, np.float32)
            noise_np = np.random.default_rng(i).standard_normal(
                (self.Sa, 32)).astype(np.float16)
            self.g_noise.upload(noise_np)
            _upload_state_and_project(obs)

            ae_bufs, ae_weights, ae_dims = self._build_ae_dicts(stream_int)
            ae_bufs['calib_buf'] = _ae_calib_buf.ptr.value
            ae_bufs['d_scale'] = _ae_d_scale.ptr.value
            ae_bufs['hidden_scratch'] = _ae_hidden_scratch.ptr.value
            ae_bufs['fp8_scratch'] = _ae_fp8_scratch.ptr.value
            ae_bufs['norm_scratch'] = _ae_norm_scratch.ptr.value
            ae_bufs['x_scratch'] = _ae_x_scratch.ptr.value
            ae_bufs['ones'] = _ae_ones.ptr.value
            decoder_forward_calibrate_pi0(
                self._ctx, self._fvk, ae_bufs, ae_weights, ae_dims,
                ae_scales_buf.ptr.value, stream=stream_int)
            self._cudart.cudaStreamSynchronize(stream)
            per_sample_ae.append(
                ae_scales_buf.download_new((La * 4,), np.float32))

            if verbose and (i + 1) % max(1, n // 10) == 0:
                logger.info("  sample %d/%d", i + 1, n)

        # Percentile-reduce along sample axis.
        enc_final = accumulate_amax(per_sample_enc, percentile=percentile)
        ae_final = accumulate_amax(per_sample_ae, percentile=percentile)

        # Upload reduced scales + recompute alpha in float32.
        self.enc_calib_scales = CB.from_numpy_managed(enc_final)
        self.ae_calib_scales = CB.from_numpy_managed(ae_final)
        enc_ws_np = self.enc_w_dev.download_new((Le * 4,), np.float32)
        self.enc_alpha_host = [
            float(np.float32(enc_final[i]) * np.float32(enc_ws_np[i]))
            for i in range(Le * 4)
        ]

        # Recapture graph with fresh scales baked in.
        self._capture_enc_ae_graph()
        self._real_data_calibrated = True
        logger.info(
            "Pi0 JAX Thor multi-frame calibrate complete (N=%d, "
            "percentile=%.2f)", n, percentile)

    def calibrate_with_real_data(self, sample_observations) -> None:
        """Legacy alias for :meth:`calibrate`."""
        self.calibrate(sample_observations)

    @property
    def precision_spec(self):
        """JAX Thor does not yet surface a structured PrecisionSpec."""
        return None

    def _recalibrate_with_real_data(self):
        CB = self._CudaBuffer
        Se = self.Se; Le = self.Le; La = self.La
        total_keys = self.total_keys
        _cudart = self._cudart; stream = self._stream
        stream_int = stream.value or 0
        Da = self.Da; Ha = self.Ha; S_dec = self.S_dec

        # Encoder recalibration
        De = self.De; He = self.He
        _norm_scratch = CB.device_empty(Se * De, np.float16)
        _x_scratch = CB.device_empty(Se * De, np.float16)
        _enc_calib_buf = CB.zeros(Le * 4, np.float32)
        _d_scale = CB.zeros(1, np.float32)
        _fp8_scratch = CB.device_zeros(Se * max(De, He), np.uint8)
        _ones_buf = CB.from_numpy(np.ones(De, dtype=np.float16))

        enc_scales_buf = CB.zeros(Le * 4, np.float32)

        enc_bufs = {
            'x': self.enc_x.ptr.value, 'x_fp8': self.enc_buf[0].ptr.value,
            'qkv': self.enc_buf[1].ptr.value, 'logits': self.enc_buf[2].ptr.value,
            'attn_out': self.enc_buf[3].ptr.value, 'o_fp8': self.enc_buf[5].ptr.value,
            'gate': self.enc_buf[6].ptr.value, 'hidden': self.enc_buf[7].ptr.value,
            'hid_fp8': self.enc_buf[9].ptr.value, 'fg': self.enc_buf[10].ptr.value,
            'ctx': self._ctx,
            'norm_scratch': _norm_scratch.ptr.value,
            'x_scratch': _x_scratch.ptr.value,
            'calib_buf': _enc_calib_buf.ptr.value,
            'd_scale': _d_scale.ptr.value,
            'fp8_scratch': _fp8_scratch.ptr.value,
            'ones': _ones_buf.ptr.value,
        }
        enc_weights = {
            'qkv_w': [self.ew[0].ptr.value + i * De * 2560 for i in range(Le)],
            'o_w': [self.ew[1].ptr.value + i * De * De for i in range(Le)],
            'gate_w': [self.ew[2].ptr.value + i * De * He * 2 for i in range(Le)],
            'down_w': [self.ew[4].ptr.value + i * He * De for i in range(Le)],
            'rope': self.enc_rope.ptr.value,
            'Kc': self.Kc.ptr.value, 'Vc': self.Vc.ptr.value,
            'w_scales': self.enc_w_dev.ptr.value,
        }
        enc_dims = {'Se': Se, 'D': De, 'H': He, 'NH': self.NHe, 'HD': self.HDe,
                    'L': Le, 'total_keys': total_keys}

        self.Kc.zero_(stream); self.Vc.zero_(stream)
        encoder_forward_calibrate(
            self._gemm, self._fvk, enc_bufs, enc_weights, enc_dims,
            enc_scales_buf.ptr.value, stream=stream_int)
        _cudart.cudaStreamSynchronize(stream)

        self.enc_calib_scales = enc_scales_buf
        enc_scales_np = enc_scales_buf.download_new((Le * 4,), np.float32)
        enc_ws_np = self.enc_w_dev.download_new((Le * 4,), np.float32)
        self.enc_alpha_host = [
            float(np.float32(enc_scales_np[i]) * np.float32(enc_ws_np[i]))
            for i in range(Le * 4)]

        # Decoder recalibration
        ae_scales_buf = CB.zeros(La * 4, np.float32)
        _ae_calib_buf = CB.zeros(La * 4, np.float32)
        _ae_d_scale = CB.zeros(1, np.float32)
        _ae_hidden_scratch = CB.device_empty(S_dec * Ha, np.float16)
        _ae_fp8_scratch = CB.device_zeros(S_dec * max(Da, Ha), np.uint8)
        _ae_norm_scratch = CB.device_empty(S_dec * Da, np.float16)
        _ae_x_scratch = CB.device_empty(S_dec * Da, np.float16)
        _ae_ones = CB.from_numpy(np.ones(Da, dtype=np.float16))

        noise_np = np.random.randn(self.Sa, 32).astype(np.float16)
        self.g_noise.upload(noise_np)

        # State proj with current real state
        fvk = self._fvk
        fvk.gmm_fp16(self._ctx, self.state_buf.ptr.value,
                     self.state_proj_w.ptr.value,
                     self.state_token.ptr.value,
                     1, Da, 32, 0.0, stream_int)
        fvk.add_bias_fp16(self.state_token.ptr.value,
                          self.state_proj_b.ptr.value, 1, Da, stream_int)

        ae_bufs, ae_weights, ae_dims = self._build_ae_dicts(stream_int)
        ae_bufs['calib_buf'] = _ae_calib_buf.ptr.value
        ae_bufs['d_scale'] = _ae_d_scale.ptr.value
        ae_bufs['hidden_scratch'] = _ae_hidden_scratch.ptr.value
        ae_bufs['fp8_scratch'] = _ae_fp8_scratch.ptr.value
        ae_bufs['norm_scratch'] = _ae_norm_scratch.ptr.value
        ae_bufs['x_scratch'] = _ae_x_scratch.ptr.value
        ae_bufs['ones'] = _ae_ones.ptr.value
        decoder_forward_calibrate_pi0(
            self._ctx, self._fvk, ae_bufs, ae_weights, ae_dims,
            ae_scales_buf.ptr.value, stream=stream_int)
        _cudart.cudaStreamSynchronize(stream)

        self.ae_calib_scales = ae_scales_buf

        # Recapture graph with updated scales
        self._capture_enc_ae_graph()
        logger.info("Recalibrated with real data + graph recaptured (Pi0)")

    # -----------------------------------------------------------------------
    # Autotune
    # -----------------------------------------------------------------------

    def _autotune_enc_ae(self, n_trials=5, n_bench=10):
        _crt = self._cudart
        stream = self._stream

        dummy_img = np.zeros((self.num_views, 224, 224, 3), dtype=np.float16)
        self.img_buf.upload(dummy_img)

        logger.info("Autotune: up to %d trials for best Enc+AE graph...", n_trials)

        for trial in range(n_trials):
            self._capture_enc_ae_graph()

            latencies = []
            for _ in range(n_bench):
                noise_np = np.random.randn(self.Sa, 32).astype(np.float16)
                self.g_noise.upload(noise_np)
                self.siglip_graph.replay(self._stream)
                _crt.cudaStreamSynchronize(stream)

                t0 = _time.perf_counter()
                self.enc_ae_graph.replay(self._stream)
                _crt.cudaStreamSynchronize(stream)
                latencies.append((_time.perf_counter() - t0) * 1000)

            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            logger.info("  Trial %d: %.2f ms", trial, p50)

            if p50 < 38.0:
                logger.info("Autotune done: Enc+AE = %.2f ms (trial %d)", p50, trial)
                return

        logger.info("Autotune done: Enc+AE = %.2f ms (best of %d)", p50, n_trials)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def infer(self, observation, debug=False):
        t0 = _time.perf_counter()

        # Collect images
        if 'images' in observation:
            img_list = observation['images']
        else:
            img_list = [observation['image'], observation['wrist_image']]
            if self.num_views >= 3 and 'wrist_image_right' in observation:
                img_list.append(observation['wrist_image_right'])

        def _normalize(im):
            if im.dtype == np.float16:
                return im
            return (im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)

        images_np = np.stack([_normalize(im) for im in img_list[:self.num_views]])
        self.img_buf.upload(images_np)

        # Pi0: upload state
        state = observation.get('state', None)
        if state is not None:
            state_fp16 = np.asarray(state, dtype=np.float16).reshape(1, -1)
            self.state_buf.upload(state_fp16)
        else:
            self.state_buf.zero_()

        # Noise
        noise_np = np.random.randn(self.Sa, 32).astype(np.float16)
        self.g_noise.upload(noise_np)

        # CUDA Graph replay
        self.siglip_graph.replay(self._stream)

        # Lazy real-data recalibration on first call
        if not self._real_data_calibrated:
            self._cudart.cudaStreamSynchronize(self._stream)
            self._recalibrate_with_real_data()
            self._real_data_calibrated = True

        self.enc_ae_graph.replay(self._stream)
        self._cudart.cudaStreamSynchronize(self._stream)

        # Output
        raw_actions = self.g_noise.download_new((self.Sa, 32), np.float16).astype(np.float32)

        if self.norm_stats:
            from flash_rt.core.utils.actions import unnormalize_actions, LIBERO_ACTION_DIM
            unnorm = unnormalize_actions(raw_actions, self.norm_stats)
            robot_actions = unnorm[:, :LIBERO_ACTION_DIM]
        else:
            robot_actions = raw_actions

        latency = (_time.perf_counter() - t0) * 1000
        if debug:
            logger.info(f"JAX Pi0 infer: {latency:.1f} ms")

        return {"actions": robot_actions}
