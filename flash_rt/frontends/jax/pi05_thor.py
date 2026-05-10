"""FlashRT — Thor JAX frontend (SM110).

Loads JAX Orbax checkpoints → FP8 quantize (ml_dtypes) → CudaBuffer (cudaMalloc)
→ pipeline.py (flash_rt_kernels.so) on Jetson AGX Thor.

No PyTorch dependency.
CudaBuffer (cudaMalloc) bridges JAX weight loading ↔ kernel execution.

Architecture:
  Orbax → numpy → JAX GPU FP8 quantize → CudaBuffer
  → pipeline.py (siglip_forward, encoder_forward, decoder_forward)
  → CudaBuffer → numpy output
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
from flash_rt.models.pi05.pipeline_thor import (
    decoder_forward,
    decoder_forward_calibrate,
)
from flash_rt.hardware.thor.attn_backend import (
    ThorFlashAttnBackend,
    make_pi05_attention_spec,
)

logger = logging.getLogger(__name__)

# XLA flags must be set before JAX import
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_triton_gemm" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        _xla_flags + " --xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"
    ).strip()

# Thor unified memory: disable JAX preallocating ~75% of system RAM
if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp


from flash_rt.core.thor_frontend_utils import embed_prompt_numpy as _embed_prompt  # noqa: E402


class Pi05JaxFrontendThor:
    """Thor SM110 — JAX (Orbax) frontend. CudaBuffer bridge to flash_rt_kernels."""

    def __init__(self, checkpoint_dir, engine_path=None, fmha_path=None,
                 use_cuda_graph=True, num_views=2, autotune=3,
                 weight_cache=True, use_fp8=True, **kwargs):
        """
        Args:
            autotune: CUDA Graph autotune trials per set_prompt().
                0 = off, 3 = default, 5+ = thorough.
            weight_cache: Cache FP8-quantized weights to disk after first load.
                Reduces JAX cold start from ~42s to ~5s on subsequent loads.
                Only affects JAX (Orbax is slow to load; safetensors is already fast).
                Set False to force re-quantize (e.g., after fine-tuning).
        """
        from flash_rt.core.cuda_buffer import CudaBuffer, sync
        from flash_rt.core.weights.transformer import quantize_fp8_e4m3, compute_time_embeddings
        self._CudaBuffer = CudaBuffer
        self._sync = sync

        checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.use_fp8 = bool(use_fp8)

        # ── Load norm stats (openpi or lerobot HF release) ──
        from flash_rt.core.utils.norm_stats import (
            load_norm_stats, lerobot_candidates,
        )
        self.norm_stats = load_norm_stats(
            [checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
             checkpoint_dir / "norm_stats.json",
             *lerobot_candidates(checkpoint_dir)],
            checkpoint_dir=checkpoint_dir,
            strict=False,
        )

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

        # ── Load weights (with FP8 weight cache for fast reload) ──
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
            from flash_rt.core.weights.transformer import transform_jax_weights

            fmt = detect_format(str(checkpoint_dir))
            raw = load_weights(str(checkpoint_dir), format=fmt)
            engine_w = transform_jax_weights(raw)
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

        # ---- RL CFG state (set via set_rl_mode) ----
        # Mirrors :class:`Pi05TorchFrontendThor`'s contract — when
        # ``_rl_config`` is non-None, ``set_prompt`` builds an
        # advantage-conditioned (cond) + raw (uncond) prompt pair and
        # ``infer`` runs the encoder + decoder twice (cond / uncond)
        # then combines via ``fvk.cfg_combine_into_residual_fp16``
        # (per-chunk CFG; collapses to cond-only at ``cfg_beta=1.0``).
        self._rl_config = None
        self._lang_emb_cond_np = None
        self._lang_emb_uncond_np = None
        # Device-side mirrors of the cond/uncond embeds, populated at
        # set_prompt time. Used by the fused-CFG fast path: the SigLIP
        # callbacks D2D-copy from these into the captured ``self.lang_emb``
        # slot instead of doing per-call host→device uploads.
        self._lang_emb_cond_dev = None
        self._lang_emb_uncond_dev = None
        self._noise_R_snapshot_np = None
        self._v_cond_buf = None
        self._v_uncond_buf = None
        self._rl_current_prompt_text = None

        # ── B=N batched mode state (JAX side, parallel to torch) ──
        # All ``_b2``-suffixed CudaBuffers + the captured B=N graph
        # are lazily created on first ``set_batched_mode(enable=True)``
        # — the existing B=1 path stays byte-equal until that fires.
        self._batched = False
        self.B = 1
        self._Kc_b2 = None
        self._Vc_b2 = None
        self._enc_x_b2 = None
        self._enc_buf_b2 = None    # list mirroring self.enc_buf at B*Se
        self._ae_buf_b2 = None     # list mirroring self.ae_buf at B*Sa
        self._g_noise_b2 = None
        self._v_b2 = None
        self._sa_all_b2 = None     # B-tiled style buffers (CudaBuffer)
        self._sf_all_b2 = None
        self._fs_all_b2 = None
        self._enc_ae_graph_b2 = None
        # Outer fused-CFG graph: lang swap (×2) + SigLIP (×2) +
        # enc_ae_b2 (with per-step CFG inside), one replay() per CFG
        # inference. Mirrors RTX
        # ``Pi05CFGBatchedPipeline.forward`` /
        # ``self._graph.replay()``.
        self._cfg_b2_outer_graph = None
        # When non-None, ``_capture_enc_ae_graph_b2`` bakes the per-step
        # CFG combine + noise mirror into the graph (paper-correct
        # per-step CFG; mirrors RTX
        # ``Pi05CFGBatchedPipeline.transformer_decoder_batched``).
        self._enc_ae_graph_b2_cfg_beta = None
        self.autotune = int(autotune) if autotune is not True else 3
        if autotune is False:
            self.autotune = 0
        self._rng_key = jax.random.PRNGKey(0)
        logger.info("JAX backend initialized (cache_hit=%s)", cache_hit)

    def _prefetch_weights(self):
        """Weights are already managed (cudaMallocManaged). No conversion needed.
        On Thor unified memory, managed and device have ~0.001ms GEMM performance difference."""
        logger.info("Weights ready (managed memory)")

    @staticmethod
    def _gpu_quantize_fp8(w_np):
        """FP8 E4M3 quantization on GPU via JAX. Returns (jax uint8 array, scale float)."""
        import ml_dtypes
        w_jax = jnp.array(w_np, dtype=jnp.float32)
        amax = float(jnp.abs(w_jax).max())
        scale = max(amax / 448.0, 1e-12)
        w_scaled = jnp.clip(w_jax / scale, -448.0, 448.0)
        fp8_jax = w_scaled.astype(ml_dtypes.float8_e4m3fn)
        return fp8_jax.view(jnp.uint8), scale

    def _upload_weights(self, engine_w, quantize_fp8, compute_time_embeddings):
        """Upload numpy engine weights to CudaBuffer (cudaMalloc).
        FP8 quantization runs on GPU (JAX), then copy_from_jax to CudaBuffer."""
        CB = self._CudaBuffer
        fp16 = np.float16
        qfp8 = self._gpu_quantize_fp8

        # ── Declarative weight-loader pass (stage 7.6) ──
        # Populates on self:
        #   12 × _sig_{ln1w,ln1b,qw,qb,ow,ob,ln2w,ln2b,uw,ub,dw,db}_cb CudaBuffers
        #   _sig_scales (108 fp32), _enc_{qkv,o,gu,d}_cb, _enc_ws (72 fp32),
        #   dec_{qkv,o,gu,d}_flat CudaBuffers, _ae_ws (72 fp32),
        #   _{attn,ffn}_mod_{w,b} numpy lists (per-layer).
        # Also mutates self._cache_blobs with the per-slot cache keys.
        self._cache_blobs = {}

        from flash_rt.executors.jax_weights import OrbaxDictSource
        from flash_rt.executors.weight_loader import WeightLoader
        from flash_rt.frontends.jax._pi05_thor_spec import build_spec
        WeightLoader(source=OrbaxDictSource(engine_w),
                     target=self, spec=build_spec()).run()

        # ── Vision (SigLIP) ──
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
        self.sig_buf = [CB.device_empty(S * 3 * D, fp16),  # qkv_buf
                        CB.device_empty(S * D, fp16),      # attn_out
                        CB.device_empty(S * H, fp16)]      # hidden
        self.sig_scratch = CB.device_empty(S * max(D, H), fp16)

        # Patch embedding weights → CudaBuffer (for unified pybind11 path)
        pe_w_np = engine_w["vision.patch_embed.weight"].astype(fp16)  # (14,14,3,1152)
        pe_oihw = pe_w_np.transpose(3, 2, 0, 1)  # (O,C,H,W) = (1152,3,14,14)
        # C++ patch_im2col extracts patches in (kH, kW, C) order → reorder weight to match
        pe_w_2d_np = pe_oihw.transpose(0,2,3,1).reshape(pe_oihw.shape[0], -1).T.copy().astype(fp16)  # (588_HWC, 1152)
        pe_b_np = engine_w["vision.patch_embed.bias"].astype(fp16).copy()
        pos_emb_np = engine_w["vision.pos_embed"][:256].astype(fp16).copy()
        self.pe_w_buf = CB.from_numpy(pe_w_2d_np)
        self.pe_b_buf = CB.from_numpy(pe_b_np)
        self.pos_emb_buf = CB.from_numpy(pos_emb_np)
        self._cache_blobs["pe_w_buf"] = pe_w_2d_np.tobytes()
        self._cache_blobs["pe_b_buf"] = pe_b_np.tobytes()
        self._cache_blobs["pos_emb_buf"] = pos_emb_np.tobytes()
        # Pre-allocate image + patches buffers (fixed address for graph capture)
        nv = self.num_views
        self.img_buf = CB.device_empty(nv * 224 * 224 * 3, fp16)
        self.patches_buf = CB.device_empty(S * 588, fp16)
        # GemmRunner for FP16 patch_embed GEMM
        from flash_rt import flash_rt_kernels as _fvk
        self._fvk = _fvk
        self._gemm = _fvk.GemmRunner()

        # Vision PostLN weights → CudaBuffer (used by engine)
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

        # ── Encoder ──
        Se_max, De, He, Le = self.num_views * 256 + 256, 2048, 16384, 18
        NHe, HDe = 8, 256
        self.Se_max = Se_max; self.De = De; self.He = He; self.Le = Le
        self.NHe = NHe; self.HDe = HDe

        # Encoder per-layer flat CudaBuffers produced by loader; compose
        # the 5-slot list consumed by shared_primitives.encoder_forward
        # (slot [3] is a dummy 1-byte buffer — legacy placeholder).
        self.ew = [self._enc_qkv_cb, self._enc_o_cb, self._enc_gu_cb,
                   CB.device_zeros(1, np.uint8),
                   self._enc_d_cb]
        _enc_ws_np = np.array(self._enc_ws, dtype=np.float32)
        self.enc_w_dev = CB.from_numpy_managed(_enc_ws_np)
        self._cache_blobs["ew.3"] = b'\x00'  # placeholder for device_zeros(1)
        self._cache_blobs["enc_w_dev"] = _enc_ws_np.tobytes()

        # RoPE: keep as numpy for set_prompt computation
        inv_freq = 1.0 / (10000 ** (np.arange(0, 256, 2, dtype=np.float64) / 256))
        pos = np.arange(1200, dtype=np.float64)
        kp = pos[:, None] * inv_freq[None, :]
        self._kc_t = np.cos(kp).astype(fp16)
        self._ks_t = np.sin(kp).astype(fp16)

        Sa, Da, Ha, La = 10, 1024, 4096, 18
        self.Sa = Sa; self.Da = Da; self.Ha = Ha; self.La = La
        total_keys_max = Se_max + Sa
        self.Kc = CB.device_zeros(Le * total_keys_max * HDe, fp16)
        self.Vc = CB.device_zeros(Le * total_keys_max * HDe, fp16)

        # Encoder buffers
        self.enc_buf = [
            CB.device_zeros(Se_max * De, np.uint8),         # x_fp8
            CB.device_empty(Se_max * 2560, fp16),       # qkv_buf
            CB.device_empty(Se_max * NHe * Se_max, fp16),  # attn_logits
            CB.device_empty(Se_max * NHe * HDe, fp16),  # attn_out
            CB.device_empty(Se_max * De, fp16),          # o_proj_out
            CB.device_zeros(Se_max * De, np.uint8),          # o_fp8
            CB.device_empty(Se_max * 2 * He, fp16),     # gate_out
            CB.device_empty(Se_max * He, fp16),          # up_out/hidden
            CB.device_empty(Se_max * He, fp16),          # hidden
            CB.device_zeros(Se_max * He, np.uint8),          # hidden_fp8
            CB.device_empty(Se_max * De, fp16),          # down_out
        ]
        self.enc_act_amax = CB.device_zeros(1, np.float32)
        self.enc_norm_buf = CB.device_empty(Se_max * max(De, He), fp16)
        self.enc_x = CB.device_empty(Se_max * De, fp16)
        self.enc_rope = CB.device_empty(Se_max * 256, fp16)

        # ── Decoder ──
        # Per-layer flat buffers self.dec_{qkv,o,gu,d}_flat produced by
        # loader; scales live in self._ae_ws.

        # Embedding: keep as numpy for _embed_prompt
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
        self.ae_act_scale = CB.device_zeros(1, np.float32)
        self.ae_silu_buf = CB.device_empty(Sa * Ha, fp16)
        self.ae_buf = [
            CB.device_empty(Sa * Da, fp16),          # x
            CB.device_empty(Sa * Da, fp16),          # xn
            CB.device_empty(Sa * Da, fp16),          # gate
            CB.device_empty(Sa * 2560, fp16),        # qkv
            CB.device_empty(Sa * 8 * total_keys_max, fp16),  # logits
            CB.device_empty(Sa * 8 * 256, fp16),    # ctx
            CB.device_empty(Sa * Ha, fp16),          # hid
            CB.device_empty(Sa * 2 * Ha, fp16),     # fg
            CB.device_zeros(Sa * Da, np.uint8),           # xn_fp8
            CB.device_zeros(Sa * Ha, np.uint8),           # hid_fp8
            CB.device_zeros(Sa * 8 * 256, np.uint8),     # ctx_fp8
        ]

        # Time conditioning: numpy for CPU computation in set_prompt
        self._time_mlp_in_w = engine_w["time.mlp_in.weight"].astype(fp16)
        self._time_mlp_in_b = engine_w["time.mlp_in.bias"].astype(fp16)
        self._time_mlp_out_w = engine_w["time.mlp_out.weight"].astype(fp16)
        self._time_mlp_out_b = engine_w["time.mlp_out.bias"].astype(fp16)

        # Per-layer modulation Dense (self._{attn,ffn}_mod_{w,b}) already
        # populated by loader.
        self._final_mod_w = engine_w["decoder.final_mod.weight"].astype(fp16)
        self._final_mod_b = engine_w["decoder.final_mod.bias"].astype(fp16)

        self._time_emb_np = compute_time_embeddings(10, Da)

        # dec_{qkv,o,gu,d}_flat already populated by loader.
        self.dec_rope = CB.device_empty(Sa * 256, fp16)
        self.g_xs = CB.device_zeros(S * D, fp16)     # device: faster graph replay
        self.g_noise = CB.empty(Sa * 32, fp16)        # managed: needs D2H download

        # Patch embedding is now in CUDA graph (GPU im2col + GEMM + bias_pos via pybind11)
        # Unit scale (1.0) for SigLIP FP8 casts
        unit_scale_np = np.array([1.0], dtype=np.float32)
        self._unit_scale_buf = CB.from_numpy(unit_scale_np)
        self._unit_scale_ptr = self._unit_scale_buf.ptr.value

        logger.info("JAX backend weights uploaded to CudaBuffer")

    # -----------------------------------------------------------------------
    # FP8 Weight Cache (save / load)
    # -----------------------------------------------------------------------

    # Names and attributes of all CudaBuffers that hold weight data.
    # Format: (attr_path, is_list, is_managed)
    _CACHE_GPU_BUFFERS = [
        # SigLIP (12 buffers)
        ("sig_wt_fp8", True, False),
        # Patch embed + PostLN
        ("pe_w_buf", False, False),
        ("pe_b_buf", False, False),
        ("pos_emb_buf", False, False),
        ("final_ln_w", False, False),
        ("final_ln_b", False, False),
        ("proj_w", False, False),
        ("mm_b", False, False),
        # Encoder (5 buffers list + w_scales)
        ("ew", True, False),
        ("enc_w_dev", False, True),
        # Decoder
        ("dec_qkv_flat", False, False),
        ("dec_o_flat", False, False),
        ("dec_gu_flat", False, False),
        ("dec_d_flat", False, False),
        ("ae_w_dev", False, False),
        ("ain_w", False, False),
        ("ain_b", False, False),
        ("aow", False, False),
        ("aob", False, False),
    ]

    # Names of numpy arrays to cache (for set_prompt time conditioning)
    _CACHE_NUMPY = [
        "_embedding_np", "_time_mlp_in_w", "_time_mlp_in_b",
        "_time_mlp_out_w", "_time_mlp_out_b", "_final_mod_w", "_final_mod_b",
        "_time_emb_np", "_kc_t", "_ks_t",
    ]
    # Per-layer numpy lists
    _CACHE_NUMPY_LISTS = [
        "_attn_mod_w", "_attn_mod_b", "_ffn_mod_w", "_ffn_mod_b",
    ]

    def _save_to_cache(self, checkpoint_path):
        """Save pre-collected weight blobs + numpy arrays to disk.

        Uses _cache_blobs collected during _upload_weights (avoids GPU→CPU download
        which segfaults in XLA context).
        """
        from flash_rt.core.weights.weight_cache import save_weight_cache

        entries = []
        blobs = []

        # GPU buffers (from _cache_blobs, collected during _upload_weights)
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

        # SigLIP scales (ctypes float array)
        sig_n = len(self.sig_scales_c)
        sig_arr = np.array([self.sig_scales_c[i] for i in range(sig_n)], dtype=np.float32)
        entries.append({"name": "sig_scales", "dtype": "float32", "shape": [sig_n]})
        blobs.append(sig_arr.tobytes())

        # Numpy arrays
        for attr in self._CACHE_NUMPY:
            arr = getattr(self, attr)
            entries.append({"name": attr, "dtype": str(arr.dtype),
                            "shape": list(arr.shape)})
            blobs.append(np.ascontiguousarray(arr).tobytes())

        # Per-layer numpy lists
        for attr in self._CACHE_NUMPY_LISTS:
            lst = getattr(self, attr)
            for i, arr in enumerate(lst):
                entries.append({"name": f"{attr}.{i}", "dtype": str(arr.dtype),
                                "shape": list(arr.shape)})
                blobs.append(np.ascontiguousarray(arr).tobytes())

        # Dims metadata
        dims = {"sig_dims": list(self.sig_dims), "Se_max": self.Se_max,
                "De": self.De, "He": self.He, "Le": self.Le,
                "NHe": self.NHe, "HDe": self.HDe,
                "Sa": self.Sa, "Da": self.Da, "Ha": self.Ha, "La": self.La}
        dims_json = json.dumps(dims).encode("utf-8")
        entries.append({"name": "_dims", "dtype": "json", "shape": [len(dims_json)]})
        blobs.append(dims_json)

        save_weight_cache(checkpoint_path, self.num_views, entries, blobs)
        del self._cache_blobs  # free memory

    def _load_from_cache(self, cached):
        """Restore all weights from cache: bytes → CudaBuffer / numpy."""
        header, body = cached
        CB = self._CudaBuffer
        fp16 = np.float16

        # Build name→(offset, nbytes, dtype, shape) lookup
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

        # GPU buffers
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

        # SigLIP scales
        sig_arr = _get_numpy("sig_scales")
        self.sig_scales_c = (ctypes.c_float * len(sig_arr))(*sig_arr.tolist())

        # Dims
        dims = json.loads(_get_bytes("_dims").decode("utf-8"))
        self.sig_dims = tuple(dims["sig_dims"])
        for k in ["Se_max", "De", "He", "Le", "NHe", "HDe", "Sa", "Da", "Ha", "La"]:
            setattr(self, k, dims[k])

        # Numpy arrays
        for attr in self._CACHE_NUMPY:
            setattr(self, attr, _get_numpy(attr))

        # Per-layer numpy lists
        for attr in self._CACHE_NUMPY_LISTS:
            lst = []
            i = 0
            while f"{attr}.{i}" in lookup:
                lst.append(_get_numpy(f"{attr}.{i}"))
                i += 1
            setattr(self, attr, lst)

        # Allocate scratch/working buffers (not cached — just zeros/empty)
        S, D, H, NH, HD, L = self.sig_dims
        nv = self.num_views
        Se_max = self.Se_max; De = self.De; He = self.He; Le = self.Le
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha; La = self.La
        total_keys_max = Se_max + Sa

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

        self.ae_act_scale = CB.device_zeros(1, np.float32)
        self.ae_silu_buf = CB.device_empty(Sa * Ha, fp16)
        self.ae_buf = [
            CB.device_empty(Sa * Da, fp16),
            CB.device_empty(Sa * Da, fp16),
            CB.device_empty(Sa * Da, fp16),
            CB.device_empty(Sa * 2560, fp16),
            CB.device_empty(Sa * 8 * total_keys_max, fp16),
            CB.device_empty(Sa * 8 * 256, fp16),
            CB.device_empty(Sa * Ha, fp16),
            CB.device_empty(Sa * 2 * Ha, fp16),
            CB.device_zeros(Sa * Da, np.uint8),
            CB.device_zeros(Sa * Ha, np.uint8),
            CB.device_zeros(Sa * 8 * 256, np.uint8),
        ]
        self.dec_rope = CB.device_empty(Sa * 256, fp16)
        self.g_xs = CB.device_zeros(S * D, fp16)
        self.g_noise = CB.empty(Sa * 32, fp16)

        unit_scale_np = np.array([1.0], dtype=np.float32)
        self._unit_scale_buf = CB.from_numpy(unit_scale_np)
        self._unit_scale_ptr = self._unit_scale_buf.ptr.value

        logger.info("Weights restored from cache")

    def set_prompt(self, prompt_text):
        """Set prompt: tokenize, time conditioning (numpy), RoPE, calibrate, capture graph.

        When :meth:`set_rl_mode` has activated CFG inference and a text
        prompt is supplied, this routes to :meth:`_set_prompt_rl` which
        rebuilds for the cond + uncond pair. The re-entry guard
        ``_in_rl_set_prompt`` prevents recursion when ``_set_prompt_rl``
        calls back here with the cond text to drive the standard
        capture path.
        """
        if (self._rl_config is not None
                and not getattr(self, "_in_rl_set_prompt", False)
                and isinstance(prompt_text, str)):
            self._in_rl_set_prompt = True
            try:
                self._set_prompt_rl(prompt_text)
            finally:
                self._in_rl_set_prompt = False
            return

        CB = self._CudaBuffer
        fp16 = np.float16

        S = self.sig_dims[0]  # num_views * 256
        if isinstance(prompt_text, (np.ndarray, list)):
            # Raw token IDs
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
        self.total_keys = Se + self.Sa

        # Stage 1.5 — build AttentionBackend. total_keys must be set first
        # (see sibling comment in frontends/torch/pi05_thor.py). Rebuilt
        # per set_prompt because total_keys depends on prompt length.
        attn_scale = 1.0 / math.sqrt(float(self.HDe))
        layer_stride = int(self.total_keys) * int(self.HDe) * 2  # fp16 bytes
        _sig_D = int(self.sig_dims[1])  # sig_dims = (S, D, H, NH, HD, L)
        self._attn = ThorFlashAttnBackend(
            make_pi05_attention_spec(
                num_views=self.num_views,
                enc_seq_max=self.Se,
                chunk_size=self.Sa,
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

        # RoPE (numpy → CudaBuffer)
        enc_rope_np = np.concatenate(
            [self._kc_t[:Se, :, None], self._ks_t[:Se, :, None]], 2).reshape(Se, 256)
        self.enc_rope = CB.from_numpy(enc_rope_np.astype(fp16))
        dec_start = Se
        dec_rope_np = np.concatenate(
            [self._kc_t[dec_start:dec_start+self.Sa, :, None],
             self._ks_t[dec_start:dec_start+self.Sa, :, None]], 2).reshape(self.Sa, 256)
        self.dec_rope = CB.from_numpy(dec_rope_np.astype(fp16))

        # Time conditioning (JAX GPU — same matmuls as Thor but on GPU)
        La, Sa, Da = self.La, self.Sa, self.Da

        # Upload time conditioning weights to JAX GPU
        t_mlp_in_w = jnp.array(self._time_mlp_in_w)
        t_mlp_in_b = jnp.array(self._time_mlp_in_b)
        t_mlp_out_w = jnp.array(self._time_mlp_out_w)
        t_mlp_out_b = jnp.array(self._time_mlp_out_b)
        attn_mod_w = [jnp.array(w) for w in self._attn_mod_w]
        attn_mod_b = [jnp.array(b) for b in self._attn_mod_b]
        ffn_mod_w = [jnp.array(w) for w in self._ffn_mod_w]
        ffn_mod_b = [jnp.array(b) for b in self._ffn_mod_b]
        final_mod_w = jnp.array(self._final_mod_w)
        final_mod_b = jnp.array(self._final_mod_b)

        sa_list, sf_list, fs_list = [], [], []
        for step in range(10):
            te = jnp.array(self._time_emb_np[step][None, :], dtype=jnp.float16)
            tmp = (te @ t_mlp_in_w.T + t_mlp_in_b[None, :]).astype(jnp.float32)
            tmp = (tmp * jax.nn.sigmoid(tmp)).astype(jnp.float16)
            tmp2 = (tmp @ t_mlp_out_w.T + t_mlp_out_b[None, :]).astype(jnp.float32)
            tmp2 = (tmp2 * jax.nn.sigmoid(tmp2)).astype(jnp.float16)
            time_emb = jnp.broadcast_to(tmp2, (Sa, Da))
            for layer in range(La):
                sa_list.append(time_emb @ attn_mod_w[layer] + attn_mod_b[layer][None, :])
                sf_list.append(time_emb @ ffn_mod_w[layer] + ffn_mod_b[layer][None, :])
            fs_list.append(time_emb @ final_mod_w + final_mod_b[None, :])

        sa_all = jnp.concatenate(sa_list, axis=0).astype(jnp.float16)
        sf_all = jnp.concatenate(sf_list, axis=0).astype(jnp.float16)
        fs_all = jnp.concatenate(fs_list, axis=0).astype(jnp.float16)
        jax.block_until_ready(sa_all); jax.block_until_ready(sf_all); jax.block_until_ready(fs_all)

        def _jax_to_cb_flat(jax_arr):
            flat = jax_arr.reshape(-1)
            jax.block_until_ready(flat)
            buf = CB.device_empty(flat.size, np.dtype(flat.dtype))
            buf.copy_from_jax(flat)
            return buf

        self.ae_w_static = [
            self.ain_w, self.ain_b, _jax_to_cb_flat(sa_all),
            self.dec_qkv_flat, self.Kc, self.Vc,
            self.dec_o_flat, _jax_to_cb_flat(sf_all), self.dec_gu_flat,
            CB.device_zeros(1, np.uint8), self.dec_d_flat,
            self.aow, self.aob, _jax_to_cb_flat(fs_all),
        ]

        # Capture SigLIP graph (warmup fills enc_x)
        self._capture_siglip_graph()
        # Full device sync before calibrate (catch any async errors from graph capture)
        import ctypes as _ct
        _ct.CDLL('libcudart.so').cudaDeviceSynchronize()

        # Calibrate
        self._calibrate()

        # Free XLA caches before graph capture — XLA's GPU memory allocations
        # degrade CUDA Graph instantiation scheduling (~2ms slower).
        # All weights are already in CudaBuffer, XLA caches are not needed.
        import jax as _jax_mod
        _jax_mod.clear_caches()
        import gc; gc.collect()
        logger.info("XLA caches cleared before graph capture")

        # Capture Enc+AE graph
        if self.autotune > 0:
            self._autotune_enc_ae(n_trials=self.autotune, n_bench=10)
        else:
            self._capture_enc_ae_graph()

        self.current_prompt = prompt_text
        logger.info(f"Set prompt: '{prompt_text}' (Se={Se})")

    def _build_enc_dicts(self, stream_int=0):
        """Build pipeline.py-format dicts for encoder_forward."""
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
        """Build pipeline.py-format dicts for decoder_forward."""
        return (
            {'noise': self.g_noise.ptr.value, 'x': self.ae_buf[0].ptr.value,
             'xn': self.ae_buf[1].ptr.value, 'gate': self.ae_buf[2].ptr.value,
             'qkv': self.ae_buf[3].ptr.value, 'logits': self.ae_buf[4].ptr.value,
             'attn_out': self.ae_buf[5].ptr.value, 'hid': self.ae_buf[6].ptr.value,
             'fg': self.ae_buf[7].ptr.value, 'xn_fp8': self.ae_buf[8].ptr.value,
             'hid_fp8': self.ae_buf[9].ptr.value, 'ctx_fp8': self.ae_buf[10].ptr.value},
            {'ain_w': self.ain_w.ptr.value, 'ain_b': self.ain_b.ptr.value,
             'sa': self.ae_w_static[2].ptr.value, 'qw': self.dec_qkv_flat.ptr.value,
             'Kc': self.Kc.ptr.value, 'Vc': self.Vc.ptr.value,
             'ow': self.dec_o_flat.ptr.value, 'sf': self.ae_w_static[7].ptr.value,
             'gw': self.dec_gu_flat.ptr.value, 'dw': self.dec_d_flat.ptr.value,
             'aow': self.aow.ptr.value, 'aob': self.aob.ptr.value,
             'fs': self.ae_w_static[13].ptr.value, 'rope': self.dec_rope.ptr.value,
             'w_scales': self.ae_w_dev.ptr.value,
             'act_scales': self.ae_calib_scales.ptr.value},
            {'S': self.Sa, 'D': self.Da, 'H': self.Ha, 'NH': 8, 'HD': 256,
             'steps': 10, 'layers': self.La, 'enc_seq': self.Se,
             'total_keys': self.total_keys}
        )

    def _calibrate(self):
        """Calibrate FP8 scales using pipeline.py (framework-agnostic).

        Checks calibration cache first. On miss, runs dynamic calibration
        and saves to cache for next startup.
        """
        from flash_rt.core.quant.calibrator import load_calibration, save_calibration

        CB = self._CudaBuffer
        Se = self.Se; total_keys = self.total_keys
        Le = self.Le; La = self.La
        _cudart = self._cudart; stream = self._stream
        stream_int = stream.value or 0

        # Try cache first
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

        # Encoder calibration — scratch buffers allocated via CudaBuffer
        self.Kc.zero_(stream); self.Vc.zero_(stream)
        _cudart.cudaStreamSynchronize(stream)

        De = self.De; He = self.He
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
            'qkv_w': [self.ew[0].ptr.value + i * self.De * 2560 for i in range(Le)],
            'o_w': [self.ew[1].ptr.value + i * self.De * self.De for i in range(Le)],
            'gate_w': [self.ew[2].ptr.value + i * self.De * self.He * 2 for i in range(Le)],
            'down_w': [self.ew[4].ptr.value + i * self.He * self.De for i in range(Le)],
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

        # Decoder calibration
        Sa = self.Sa; Da = self.Da; Ha = self.Ha
        ae_scales_buf = CB.zeros(La * 4, np.float32)
        self.ae_calib_scales = ae_scales_buf
        _ae_calib_buf = CB.zeros(La * 4, np.float32)
        _ae_d_scale = CB.zeros(1, np.float32)
        _ae_hidden_scratch = CB.device_empty(Sa * Ha, np.float16)
        _ae_fp8_scratch = CB.device_zeros(Sa * max(Da, Ha), np.uint8)

        noise_np = np.random.randn(Sa, 32).astype(np.float16)
        self.g_noise.upload(noise_np)

        ae_bufs, ae_weights, ae_dims = self._build_ae_dicts(stream_int)
        ae_bufs['calib_buf'] = _ae_calib_buf.ptr.value
        ae_bufs['d_scale'] = _ae_d_scale.ptr.value
        ae_bufs['hidden_scratch'] = _ae_hidden_scratch.ptr.value
        ae_bufs['fp8_scratch'] = _ae_fp8_scratch.ptr.value
        decoder_forward_calibrate(
            self._ctx, self._fvk, ae_bufs, ae_weights, ae_dims,
            ae_scales_buf.ptr.value, stream=stream_int)
        _cudart.cudaStreamSynchronize(stream)

        self.ae_calib_scales = ae_scales_buf
        logger.info(f"Decoder calibrated: {La*4} scales")
        self.calibrated = True

        # Save to cache
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

    def _patch_embed_ops(self, stream_int):
        """Patch embedding: im2col → GEMM → bias+pos. Output in g_xs."""
        fvk = self._fvk
        S, D = self.sig_dims[0], self.sig_dims[1]
        fvk.patch_im2col(self.img_buf.ptr.value, self.patches_buf.ptr.value,
                          self.num_views, stream_int)
        self._gemm.fp16_nn(self.patches_buf.ptr.value, self.pe_w_buf.ptr.value,
                            self.g_xs.ptr.value, S, D, 588, stream_int)
        fvk.patch_embed_bias_pos(self.g_xs.ptr.value, self.pe_b_buf.ptr.value,
                                  self.pos_emb_buf.ptr.value, S, D, 256, stream_int)

    def _build_siglip_call_specs(self):
        """Build the SigLIP call dicts (sig_bufs / sig_weights / sig_dims
        + postln dicts) so both ``_capture_siglip_graph`` and the
        outer-CFG capture in ``_capture_cfg_b2_outer_graph`` can issue
        the same kernel sequence inline (without nested cudaGraphLaunch,
        which Thor SM110's runtime returns
        ``cudaErrorStreamCaptureUnsupported`` for during outer capture).
        """
        S, D, H, NH, HD, L = self.sig_dims
        D3 = 3 * D
        sig_bufs = {
            'x': self.g_xs.ptr.value,
            'x_fp8': self.sig_fp8_scratch.ptr.value,
            'qkv': self.sig_buf[0].ptr.value,
            'attn_out': self.sig_buf[1].ptr.value,
            'hidden': self.sig_buf[2].ptr.value,
            'hid_fp8': self.sig_hid_fp8.ptr.value,
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
        postln_bufs = {
            'x_sig': self.g_xs.ptr.value,
            'enc_x': self.enc_x.ptr.value,
            'scratch': self.sig_scratch.ptr.value,
        }
        postln_weights = {
            'ln_w': self.final_ln_w.ptr.value,
            'ln_b': self.final_ln_b.ptr.value,
            'proj_w': self.proj_w.ptr.value,
            'proj_b': self.mm_b.ptr.value,
            'lang_emb': self.lang_emb.ptr.value,
        }
        postln_dims = {'S_sig': S, 'D_sig': D, 'D_enc': self.De,
                       'S_lang': self.S_lang}
        return (sig_bufs, sig_weights, sig_dims,
                postln_bufs, postln_weights, postln_dims)

    def _run_siglip(self, stream_int, sig_bufs, sig_weights, sig_dims,
                    postln_bufs, postln_weights, postln_dims):
        """Issue patch_embed + SigLIP + PostLN kernels on ``stream_int``.

        Common helper so the standalone siglip CUDA Graph and the outer
        fused-CFG graph can both inline the same kernel sequence.
        """
        self._patch_embed_ops(stream_int)
        siglip_forward(self._gemm, self._fvk, sig_bufs, sig_weights,
                       sig_dims, stream_int, attn=self._attn)
        postln_project(self._gemm, self._fvk, postln_bufs,
                       postln_weights, postln_dims, stream_int)

    def _capture_siglip_graph(self):
        """Capture patch_embed + SigLIP + PostLN as CUDA graph."""
        from flash_rt.core.cuda_graph import CUDAGraph
        S, D, H, NH, HD, L = self.sig_dims

        _cudart = ctypes.CDLL("libcudart.so")
        stream = ctypes.c_void_p()
        _cudart.cudaStreamCreate(ctypes.byref(stream))
        self._stream = stream
        self._cudart = _cudart
        stream_int = stream.value or 0

        # Build SigLIP dicts for pipeline.py
        # sig_wt_fp8 layout: 12 flat CudaBuffers, each containing L layers concatenated.
        # Index: [0]=ln_attn_w, [1]=ln_attn_b, [2]=qkv_w(fp8), [3]=qkv_b,
        #        [4]=o_w(fp8), [5]=o_b, [6]=ln_ffn_w, [7]=ln_ffn_b,
        #        [8]=up_w(fp8), [9]=up_b, [10]=down_w(fp8), [11]=down_b
        # Per-layer sizes (bytes):
        #   ln_w/b: D*2 (fp16)
        #   qkv_w: D*3D (fp8, 1 byte each)
        #   qkv_b: 3D*2 (fp16)
        #   o_w: D*D (fp8)
        #   o_b: D*2 (fp16)
        #   up_w: D*H (fp8)  — transform gave (H,D).T = (D,H), then quant → fp8 (D*H bytes)
        #   up_b: H*2 (fp16)
        #   down_w: H*D (fp8)
        #   down_b: D*2 (fp16)
        D3 = 3 * D
        sig_bufs = {
            'x': self.g_xs.ptr.value, 'x_fp8': self.sig_fp8_scratch.ptr.value,
            'qkv': self.sig_buf[0].ptr.value, 'attn_out': self.sig_buf[1].ptr.value,
            'hidden': self.sig_buf[2].ptr.value, 'hid_fp8': self.sig_hid_fp8.ptr.value,
        }
        p = [b.ptr.value for b in self.sig_wt_fp8]  # base pointers
        sig_weights = {
            'ln_attn_w': [p[0]  + i * D * 2     for i in range(L)],  # fp16
            'ln_attn_b': [p[1]  + i * D * 2     for i in range(L)],  # fp16
            'qkv_w':     [p[2]  + i * D * D3    for i in range(L)],  # fp8 (D*3D bytes)
            'qkv_b':     [p[3]  + i * D3 * 2    for i in range(L)],  # fp16
            'o_w':       [p[4]  + i * D * D     for i in range(L)],  # fp8
            'o_b':       [p[5]  + i * D * 2     for i in range(L)],  # fp16
            'ln_ffn_w':  [p[6]  + i * D * 2     for i in range(L)],  # fp16
            'ln_ffn_b':  [p[7]  + i * D * 2     for i in range(L)],  # fp16
            'up_w':      [p[8]  + i * D * H     for i in range(L)],  # fp8 (D*H bytes)
            'up_b':      [p[9]  + i * H * 2     for i in range(L)],  # fp16
            'down_w':    [p[10] + i * H * D     for i in range(L)],  # fp8
            'down_b':    [p[11] + i * D * 2     for i in range(L)],  # fp16
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

        # Warmup
        dummy_img = np.random.randn(self.num_views, 224, 224, 3).astype(np.float16)
        self.img_buf.upload(dummy_img)
        for _ in range(3):
            _run_siglip(stream_int)
        _cudart.cudaStreamSynchronize(stream)
        logger.info("Patch+SigLIP warmup done")

        # Capture graph
        self.siglip_graph = CUDAGraph()
        self.siglip_graph.begin_capture(stream)
        _run_siglip(stream_int)
        self.siglip_graph.end_capture(stream)
        _cudart.cudaStreamSynchronize(stream)
        logger.info("Patch+SigLIP+PostLN CUDA Graph captured")

    def _capture_enc_ae_graph(self):
        """Capture Encoder+AE as CUDA graph via pipeline.py."""
        from flash_rt.core.cuda_graph import CUDAGraph
        stream = self._stream; _cudart = self._cudart
        stream_int = stream.value or 0

        enc_bufs, enc_weights, enc_dims = self._build_enc_dicts(stream_int)
        ae_bufs, ae_weights, ae_dims = self._build_ae_dicts(stream_int)

        def _run(st):
            self.Kc.zero_(self._stream); self.Vc.zero_(self._stream)
            encoder_forward(self._gemm, self._fvk, enc_bufs, enc_weights, enc_dims, st,
                            attn=self._attn)
            decoder_forward(self._ctx, self._fvk, ae_bufs, ae_weights, ae_dims, st,
                            attn=self._attn)

        for _ in range(3):
            _run(stream_int)
        _cudart.cudaStreamSynchronize(stream)

        self.enc_ae_graph = CUDAGraph()
        self.enc_ae_graph.begin_capture(stream)
        _run(stream_int)
        self.enc_ae_graph.end_capture(stream)
        _cudart.cudaStreamSynchronize(stream)
        logger.info(f"Enc+AE CUDA Graph captured (Se={self.Se})")

    # -----------------------------------------------------------------------
    # B=N batched inference (JAX side — Stage 2/3 parallel to torch)
    # -----------------------------------------------------------------------

    def _alloc_b2_buffers(self, B: int = 2) -> None:
        """Allocate all B=N JAX inference buffers (CudaBuffer-flavoured).

        Mirrors :meth:`Pi05TorchFrontendThor._alloc_b2_buffers` but uses
        ``CudaBuffer.device_zeros`` / ``device_empty`` instead of
        ``torch.zeros`` / ``torch.empty``. The JAX frontend keeps
        encoder + decoder working buffers as lists (``self.enc_buf`` /
        ``self.ae_buf``); the b2 mirrors are independent lists at
        ``B * <per-sample size>``.
        """
        if B < 2:
            raise ValueError(
                f"_alloc_b2_buffers requires B >= 2; got {B}")
        CB = self._CudaBuffer
        Le = self.Le; De = self.De; He = self.He
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha
        Se_max = self.Se_max
        total_keys_max = Se_max + Sa
        fp16 = np.float16
        u8 = np.uint8

        # ── KV cache (B slabs × La × total_keys_max × HD) ──
        # Allocated as ``B`` separate flat slabs so each sample's KV
        # base pointer is a clean ``self._Kc_b2[b].ptr.value``.
        self._Kc_b2 = [
            CB.device_zeros(Le * total_keys_max * HDe, fp16) for _ in range(B)
        ]
        self._Vc_b2 = [
            CB.device_zeros(Le * total_keys_max * HDe, fp16) for _ in range(B)
        ]

        # ── Encoder b2 buffers (parallel layout to self.enc_buf) ──
        self._enc_x_b2 = CB.device_empty(B * Se_max * De, fp16)
        self._enc_buf_b2 = [
            CB.device_zeros(B * Se_max * De, u8),         # x_fp8
            CB.device_empty(B * Se_max * 2560, fp16),     # qkv_buf
            CB.device_empty(Se_max * NHe * Se_max, fp16), # logits scratch (per-sample reused)
            CB.device_empty(B * Se_max * NHe * HDe, fp16),# attn_out
            CB.device_empty(B * Se_max * De, fp16),       # o_proj_out
            CB.device_zeros(B * Se_max * De, u8),         # o_fp8
            CB.device_empty(B * Se_max * 2 * He, fp16),   # gate_out
            CB.device_empty(B * Se_max * He, fp16),       # up_out
            CB.device_empty(B * Se_max * He, fp16),       # hidden
            CB.device_zeros(B * Se_max * He, u8),         # hidden_fp8
            CB.device_empty(B * Se_max * De, fp16),       # down_out (fg)
        ]

        # ── Decoder b2 buffers (parallel layout to self.ae_buf) ──
        self._ae_buf_b2 = [
            CB.device_empty(B * Sa * Da, fp16),           # x
            CB.device_empty(B * Sa * Da, fp16),           # xn
            CB.device_empty(B * Sa * Da, fp16),           # gate
            CB.device_empty(B * Sa * 2560, fp16),         # qkv
            CB.device_empty(Sa * 8 * total_keys_max, fp16),  # logits scratch
            CB.device_empty(B * Sa * 8 * 256, fp16),     # ctx (attn_out)
            CB.device_empty(B * Sa * Ha, fp16),           # hid
            CB.device_empty(B * Sa * 2 * Ha, fp16),       # fg
            CB.device_zeros(B * Sa * Da, u8),             # xn_fp8
            CB.device_zeros(B * Sa * Ha, u8),             # hid_fp8
            CB.device_zeros(B * Sa * 8 * 256, u8),        # ctx_fp8
        ]

        self._g_noise_b2 = CB.empty(B * Sa * 32, fp16)  # managed
        # Per-step velocity scratch for the CFG-batched graph: each step
        # writes (v_cond, v_uncond) here before the in-graph cfg_combine
        # mixes them into ``_g_noise_b2`` slot 0. Always allocated; the
        # non-CFG b2 path simply doesn't reference it.
        self._v_b2 = CB.device_empty(B * Sa * 32, fp16)
        self.B = int(B)
        logger.info(
            "JAX: allocated B=%d buffers (Se_max=%d, Sa=%d, total_keys_max=%d)",
            B, Se_max, Sa, total_keys_max)

    def _build_b2_style_buffers(self, B: int = 2) -> None:
        """B-tile the (steps, layers, Sa, D3) AdaRMSNorm style buffers.

        Reads the existing B=1 ``ae_w_static[2/7/13]`` (sa / sf / fs)
        on device, downloads to numpy, repeat-interleaves along the Sa
        row dim, and re-uploads to fresh CudaBuffers. Same trick as
        ``Pi05TorchFrontendThor.set_prompt`` Stage 2.C edit (calls
        ``repeat_interleave`` on a torch tensor); JAX equivalent goes
        through numpy because the source CudaBuffer lacks an in-place
        repeat op.
        """
        CB = self._CudaBuffer
        Sa = self.Sa
        D3a = 3 * self.Da
        steps = 10
        La = self.La

        n_sa = steps * La * Sa * D3a
        n_fs = steps * Sa * D3a
        sa_np = self.ae_w_static[2].download_new((n_sa,), np.float16)
        sf_np = self.ae_w_static[7].download_new((n_sa,), np.float16)
        fs_np = self.ae_w_static[13].download_new((n_fs,), np.float16)

        sa_tiled = sa_np.reshape(steps, La, Sa, D3a).repeat(B, axis=2
                                  ).reshape(steps * La * B * Sa, D3a)
        sf_tiled = sf_np.reshape(steps, La, Sa, D3a).repeat(B, axis=2
                                  ).reshape(steps * La * B * Sa, D3a)
        fs_tiled = fs_np.reshape(steps, Sa, D3a).repeat(B, axis=1
                                  ).reshape(steps * B * Sa, D3a)

        self._sa_all_b2 = CB.from_numpy(np.ascontiguousarray(sa_tiled))
        self._sf_all_b2 = CB.from_numpy(np.ascontiguousarray(sf_tiled))
        self._fs_all_b2 = CB.from_numpy(np.ascontiguousarray(fs_tiled))

    def _capture_enc_ae_graph_b2(self):
        """Capture the B=N JAX encoder + decoder graph.

        Mirrors :meth:`Pi05TorchFrontendThor._capture_enc_ae_graph_b2`
        but uses :class:`flash_rt.core.cuda_graph.CUDAGraph` and
        explicit-stream replay (no ``torch.cuda.Stream`` here). Drives
        a 3x warmup followed by a single capture against
        :func:`flash_rt.hardware.thor.shared_primitives_batched.encoder_forward_b2`
        + :func:`flash_rt.models.pi05.pipeline_thor_batched.decoder_forward_b2`.
        """
        from flash_rt.core.cuda_graph import CUDAGraph
        from flash_rt.hardware.thor.shared_primitives_batched import (
            encoder_forward_b2)
        from flash_rt.models.pi05.pipeline_thor_batched import (
            decoder_forward_b2)

        if self._sa_all_b2 is None:
            self._build_b2_style_buffers(B=self.B)

        B = self.B
        Se = self.Se
        total_keys = self.total_keys
        Le = self.Le; La = self.La; De = self.De; He = self.He
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha

        kc_b2 = [self._Kc_b2[b].ptr.value for b in range(B)]
        vc_b2 = [self._Vc_b2[b].ptr.value for b in range(B)]

        enc_bufs_b2 = {
            'x':       self._enc_x_b2.ptr.value,
            'x_fp8':   self._enc_buf_b2[0].ptr.value,
            'qkv':     self._enc_buf_b2[1].ptr.value,
            'logits':  self._enc_buf_b2[2].ptr.value,
            'attn_out': self._enc_buf_b2[3].ptr.value,
            'o_fp8':   self._enc_buf_b2[5].ptr.value,
            'gate':    self._enc_buf_b2[6].ptr.value,
            'hid_fp8': self._enc_buf_b2[9].ptr.value,
            'fg':      self._enc_buf_b2[10].ptr.value,
            'ctx':     self._ctx,
        }
        enc_weights_b2 = {
            'qkv_w':     [self.ew[0].ptr.value + j * De * 2560 for j in range(Le)],
            'o_w':       [self.ew[1].ptr.value + j * De * De for j in range(Le)],
            'gate_w':    [self.ew[2].ptr.value + j * De * He * 2 for j in range(Le)],
            'down_w':    [self.ew[4].ptr.value + j * He * De for j in range(Le)],
            'rope':      self.enc_rope.ptr.value,
            'Kc_b2':     kc_b2,
            'Vc_b2':     vc_b2,
            'act_scales':  self.enc_calib_scales.ptr.value,
            'alpha_host':  self.enc_alpha_host,
        }
        enc_dims_b2 = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        ae_bufs_b2 = {
            'noise':   self._g_noise_b2.ptr.value,
            'x':       self._ae_buf_b2[0].ptr.value,
            'xn':      self._ae_buf_b2[1].ptr.value,
            'gate':    self._ae_buf_b2[2].ptr.value,
            'qkv':     self._ae_buf_b2[3].ptr.value,
            'logits':  self._ae_buf_b2[4].ptr.value,
            'attn_out': self._ae_buf_b2[5].ptr.value,
            'fg':      self._ae_buf_b2[7].ptr.value,
            'xn_fp8':  self._ae_buf_b2[8].ptr.value,
            'hid_fp8': self._ae_buf_b2[9].ptr.value,
            'ctx_fp8': self._ae_buf_b2[10].ptr.value,
            # Per-step velocity scratch, only consumed when cfg_beta is
            # set (CFG-batched capture path); harmless otherwise.
            'v_b2':    self._v_b2.ptr.value,
        }
        # Decoder weight ptrs come from ae_w_static (B=1 layout); we
        # swap in the B-tiled style buffers (sa / sf / fs).
        ae_weights_b2 = {
            'ain_w':      self.ae_w_static[0].ptr.value,
            'ain_b':      self.ae_w_static[1].ptr.value,
            'sa':         self._sa_all_b2.ptr.value,
            'qw':         self.ae_w_static[3].ptr.value,
            'Kc_b2':      kc_b2,
            'Vc_b2':      vc_b2,
            'ow':         self.ae_w_static[6].ptr.value,
            'sf':         self._sf_all_b2.ptr.value,
            'gw':         self.ae_w_static[8].ptr.value,
            'dw':         self.ae_w_static[10].ptr.value,
            'aow':        self.ae_w_static[11].ptr.value,
            'aob':        self.ae_w_static[12].ptr.value,
            'fs':         self._fs_all_b2.ptr.value,
            'rope':       self.dec_rope.ptr.value,
            'w_scales':   self.ae_w_dev.ptr.value,
            'act_scales': self.ae_calib_scales.ptr.value,
        }
        ae_dims_b2 = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        stream = self._stream
        stream_int = stream.value or 0
        _cudart = self._cudart

        # When the CFG-batched pipeline is active, bake the per-step
        # CFG combine + noise mirror into the captured graph (matches
        # RTX). Non-CFG batched paths leave this None.
        cfg_beta = self._enc_ae_graph_b2_cfg_beta

        def _b2_run(st):
            for b in range(B):
                self._Kc_b2[b].zero_(stream)
                self._Vc_b2[b].zero_(stream)
            encoder_forward_b2(
                self._gemm, self._fvk, enc_bufs_b2, enc_weights_b2,
                enc_dims_b2, st, B=B)
            decoder_forward_b2(
                self._ctx, self._fvk, ae_bufs_b2, ae_weights_b2,
                ae_dims_b2, st, B=B, cfg_beta=cfg_beta)

        for _ in range(3):
            _b2_run(stream_int)
        _cudart.cudaStreamSynchronize(stream)

        self._enc_ae_graph_b2 = CUDAGraph()
        self._enc_ae_graph_b2.begin_capture(stream)
        _b2_run(stream_int)
        self._enc_ae_graph_b2.end_capture(stream)
        _cudart.cudaStreamSynchronize(stream)
        logger.info(
            "JAX Enc+AE CUDA Graph captured at B=%d (Se=%d)", B, Se)

    def _capture_cfg_b2_outer_graph(self) -> None:
        """Capture the entire fused-CFG B=2 pipeline as one outer graph.

        Mirrors :meth:`Pi05TorchFrontendThor._capture_cfg_b2_outer_graph`
        and RTX
        :meth:`Pi05CFGBatchedPipeline.forward` /
        ``self._graph.replay()``.

        Inline-records (rather than nests via cudaGraphLaunch) the
        SigLIP and encoder/decoder kernel sequences so the outer graph
        is a single flat capture. Thor SM110's runtime returns
        ``cudaErrorStreamCaptureUnsupported`` (900) when launching an
        already-instantiated child graph during outer-stream capture,
        so the safe path is to re-record the kernels.

        The outer graph captures (in order, via
        ``cudaStreamBeginCapture`` in Relaxed mode):

          1. ``lang_emb := lang_emb_cond_dev``       (D2D)
          2. patch_embed + siglip + postln           (writes _enc_x cond)
          3. ``_enc_x → _enc_x_b2[0:Se]``            (D2D snapshot)
          4. ``lang_emb := lang_emb_uncond_dev``     (D2D)
          5. patch_embed + siglip + postln           (writes _enc_x uncond)
          6. ``_enc_x → _enc_x_b2[Se:2*Se]``         (D2D snapshot)
          7. encoder_forward_b2 + decoder_forward_b2(cfg_beta=...)
             (decoder carries the per-step cfg_combine + noise mirror).

        Per-call frontend work shrinks to: image upload + noise R upload
        + ONE ``outer_graph.replay()`` + final sync.
        """
        from flash_rt.core.cuda_graph import CUDAGraph
        from flash_rt.hardware.thor.shared_primitives_batched import (
            encoder_forward_b2)
        from flash_rt.models.pi05.pipeline_thor_batched import (
            decoder_forward_b2)
        cudart = self._cudart
        stream = self._stream
        stream_int = stream.value or 0
        B = self.B
        Se = self.Se
        De = self.De
        Sa = self.Sa; Da = self.Da; Ha = self.Ha
        Le = self.Le; La = self.La; He = self.He
        NHe = self.NHe; HDe = self.HDe
        total_keys = self.total_keys
        lang_nbytes = self.S_lang * De * 2  # fp16
        enc_x_slot_bytes = Se * De * 2
        cfg_beta = self._enc_ae_graph_b2_cfg_beta

        # SigLIP call specs (rebuilt each capture; the underlying buffer
        # ptrs are stable across the frontend's lifetime).
        (sig_bufs, sig_weights, sig_dims,
         postln_bufs, postln_weights, postln_dims) = (
            self._build_siglip_call_specs())

        # Encoder-b2 + Decoder-b2 specs (mirror _capture_enc_ae_graph_b2's
        # local dicts; same _enc_buf_b2 / ew indexing convention).
        kc_b2 = [self._Kc_b2[b].ptr.value for b in range(B)]
        vc_b2 = [self._Vc_b2[b].ptr.value for b in range(B)]
        enc_bufs_b2 = {
            'x':       self._enc_x_b2.ptr.value,
            'x_fp8':   self._enc_buf_b2[0].ptr.value,
            'qkv':     self._enc_buf_b2[1].ptr.value,
            'logits':  self._enc_buf_b2[2].ptr.value,
            'attn_out': self._enc_buf_b2[3].ptr.value,
            'o_fp8':   self._enc_buf_b2[5].ptr.value,
            'gate':    self._enc_buf_b2[6].ptr.value,
            'hid_fp8': self._enc_buf_b2[9].ptr.value,
            'fg':      self._enc_buf_b2[10].ptr.value,
            'ctx':     self._ctx,
        }
        enc_weights_b2 = {
            'qkv_w':     [self.ew[0].ptr.value + j * De * 2560 for j in range(Le)],
            'o_w':       [self.ew[1].ptr.value + j * De * De for j in range(Le)],
            'gate_w':    [self.ew[2].ptr.value + j * De * He * 2 for j in range(Le)],
            'down_w':    [self.ew[4].ptr.value + j * He * De for j in range(Le)],
            'rope':      self.enc_rope.ptr.value,
            'Kc_b2':     kc_b2,
            'Vc_b2':     vc_b2,
            'act_scales':  self.enc_calib_scales.ptr.value,
            'alpha_host':  self.enc_alpha_host,
        }
        enc_dims_b2 = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }
        ae_bufs_b2 = {
            'noise':   self._g_noise_b2.ptr.value,
            'x':       self._ae_buf_b2[0].ptr.value,
            'xn':      self._ae_buf_b2[1].ptr.value,
            'gate':    self._ae_buf_b2[2].ptr.value,
            'qkv':     self._ae_buf_b2[3].ptr.value,
            'logits':  self._ae_buf_b2[4].ptr.value,
            'attn_out': self._ae_buf_b2[5].ptr.value,
            'fg':      self._ae_buf_b2[7].ptr.value,
            'xn_fp8':  self._ae_buf_b2[8].ptr.value,
            'hid_fp8': self._ae_buf_b2[9].ptr.value,
            'ctx_fp8': self._ae_buf_b2[10].ptr.value,
            'v_b2':    self._v_b2.ptr.value,
        }
        ae_weights_b2 = {
            'ain_w':      self.ae_w_static[0].ptr.value,
            'ain_b':      self.ae_w_static[1].ptr.value,
            'sa':         self._sa_all_b2.ptr.value,
            'qw':         self.ae_w_static[3].ptr.value,
            'Kc_b2':      kc_b2,
            'Vc_b2':      vc_b2,
            'ow':         self.ae_w_static[6].ptr.value,
            'sf':         self._sf_all_b2.ptr.value,
            'gw':         self.ae_w_static[8].ptr.value,
            'dw':         self.ae_w_static[10].ptr.value,
            'aow':        self.ae_w_static[11].ptr.value,
            'aob':        self.ae_w_static[12].ptr.value,
            'fs':         self._fs_all_b2.ptr.value,
            'rope':       self.dec_rope.ptr.value,
            'w_scales':   self.ae_w_dev.ptr.value,
            'act_scales': self.ae_calib_scales.ptr.value,
        }
        ae_dims_b2 = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        def _outer_run(st):
            # Cond branch
            self._fvk.gpu_copy(
                self.lang_emb.ptr.value,
                self._lang_emb_cond_dev.ptr.value,
                lang_nbytes, st)
            self._run_siglip(st, sig_bufs, sig_weights, sig_dims,
                             postln_bufs, postln_weights, postln_dims)
            self._fvk.gpu_copy(
                self._enc_x_b2.ptr.value,
                self.enc_x.ptr.value,
                enc_x_slot_bytes, st)
            # Uncond branch (overwrites _g_xs / _enc_x; we already
            # snapshot'd cond into _enc_x_b2[0]).
            self._fvk.gpu_copy(
                self.lang_emb.ptr.value,
                self._lang_emb_uncond_dev.ptr.value,
                lang_nbytes, st)
            self._run_siglip(st, sig_bufs, sig_weights, sig_dims,
                             postln_bufs, postln_weights, postln_dims)
            self._fvk.gpu_copy(
                self._enc_x_b2.ptr.value + enc_x_slot_bytes,
                self.enc_x.ptr.value,
                enc_x_slot_bytes, st)
            # Encoder + Decoder at B=2 with per-step CFG inline.
            for b in range(B):
                self._Kc_b2[b].zero_(stream)
                self._Vc_b2[b].zero_(stream)
            encoder_forward_b2(
                self._gemm, self._fvk, enc_bufs_b2, enc_weights_b2,
                enc_dims_b2, st, B=B)
            decoder_forward_b2(
                self._ctx, self._fvk, ae_bufs_b2, ae_weights_b2,
                ae_dims_b2, st, B=B, cfg_beta=cfg_beta)

        # Warmup so cuBLAS / cuDNN stabilise before capture freezes
        # tactics.
        for _ in range(3):
            _outer_run(stream_int)
        cudart.cudaStreamSynchronize(stream)

        self._cfg_b2_outer_graph = CUDAGraph()
        self._cfg_b2_outer_graph.begin_capture(stream)
        _outer_run(stream_int)
        self._cfg_b2_outer_graph.end_capture(stream)
        cudart.cudaStreamSynchronize(stream)
        logger.info(
            "JAX CFG-B=2 outer CUDA graph captured (Se=%d, S_lang=%d)",
            Se, self.S_lang)

    def _autotune_cfg_b2_outer_graph(self, n_trials: int = 3,
                                       n_bench: int = 10) -> None:
        """Capture the fused-CFG outer graph N times, keep the fastest.

        Same rationale as :meth:`_autotune_enc_ae` (B=1 path): cuBLASLt
        heuristic state and CUDA graph instantiation are not
        deterministic across captures on Thor, especially under
        process-state variation between backends. Recapturing N times
        and benching each lets each backend converge on its
        locally-optimal schedule instead of being stuck with whatever
        the first heuristic call returned.

        Driven by the frontend's ``self.autotune`` knob, parameterised
        identically to the B=1 path.
        """
        import time as _time
        cudart = self._cudart
        stream = self._stream

        candidates = []
        for trial in range(n_trials):
            self._cfg_b2_outer_graph = None
            self._capture_cfg_b2_outer_graph()
            graph = self._cfg_b2_outer_graph

            latencies = []
            for _ in range(n_bench):
                t0 = _time.perf_counter()
                graph.replay(stream)
                cudart.cudaStreamSynchronize(stream)
                latencies.append((_time.perf_counter() - t0) * 1000)
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            candidates.append((p50, graph))
            logger.info("  [JAX B2 autotune] trial %d/%d: p50=%.2f ms",
                        trial + 1, n_trials, p50)

        best_p50, best_graph = min(candidates, key=lambda x: x[0])
        self._cfg_b2_outer_graph = best_graph
        for p50, g in candidates:
            if g is not best_graph:
                del g
        logger.info("  [JAX B2 autotune] kept best: p50=%.2f ms (of %d trials)",
                    best_p50, n_trials)

    def set_batched_mode(self, *, enable: bool = True,
                          batch_size: int = 2) -> None:
        """Switch JAX frontend to / from B=N batched inference.

        Mirrors :meth:`Pi05TorchFrontendThor.set_batched_mode`.
        """
        if not enable:
            self._batched = False
            self._enc_ae_graph_b2 = None
            return
        if batch_size < 2:
            raise ValueError(
                f"set_batched_mode requires batch_size >= 2; got {batch_size}")
        if self._Kc_b2 is None or self.B != batch_size:
            self._alloc_b2_buffers(B=batch_size)
        self._batched = True
        self.B = int(batch_size)
        self._enc_ae_graph_b2 = None
        # B-tiled styles depend on set_prompt-time data; rebuild on
        # next graph capture.
        self._sa_all_b2 = None
        self._sf_all_b2 = None
        self._fs_all_b2 = None
        logger.info(
            "JAX batched mode ENABLED (B=%d). Next infer_batch / CFG "
            "infer will lazily capture the b2 graph.", batch_size)

    def infer_batch(self, observations):
        """B=N JAX inference on a list of observations.

        Mirrors :meth:`Pi05TorchFrontendThor.infer_batch`. Stages SigLIP
        per slot via the B=1 graph, copies vision tokens into the
        appropriate slot of ``_enc_x_b2``, replays the B=N graph,
        unpacks per-slot actions.
        """
        from flash_rt.core.utils.actions import (
            unnormalize_actions, LIBERO_ACTION_DIM)
        if not self._batched:
            raise RuntimeError(
                "set_batched_mode(enable=True) must be called first")
        if self._enc_ae_graph_b2 is None:
            self._capture_enc_ae_graph_b2()

        if isinstance(observations, dict):
            observations = [observations] * self.B
        if len(observations) != self.B:
            raise ValueError(
                f"infer_batch expected {self.B} observations; "
                f"got {len(observations)}")

        nv = self.num_views
        Se = self.Se
        De = self.De
        Sa = self.Sa
        stream = self._stream
        stream_int = stream.value or 0

        t0 = _time.perf_counter()

        # SigLIP per slot (B=1 graph, B times) + copy into _enc_x_b2.
        # Each iteration writes _enc_x[:Se] then we D2D copy that
        # slice into _enc_x_b2[b*Se : (b+1)*Se]. CudaBuffer doesn't
        # have a slice-copy primitive; we round-trip via numpy
        # (Se*De*2 bytes per slot — sub-millisecond at Se ≤ 800).
        for b, obs in enumerate(observations):
            if 'images' in obs:
                img_list = obs['images']
            else:
                img_list = [obs['image']]
                if nv >= 2:
                    img_list.append(
                        obs.get('wrist_image', obs['image']))
                if nv >= 3:
                    img_list.append(
                        obs.get('wrist_image_right', img_list[-1]))

            def _to_np16(im):
                if im.dtype == np.float16:
                    return im
                return (np.asarray(im).astype(np.float32) / 127.5 - 1.0
                        ).astype(np.float16)
            images_np = np.stack([_to_np16(im) for im in img_list[:nv]])
            self.img_buf.upload(images_np)
            self.siglip_graph.replay(stream)
            self._cudart.cudaStreamSynchronize(stream)
            slot_np = self.enc_x.download_new((Se * De,), np.float16)
            # Upload into _enc_x_b2 at offset b*Se*De.
            # CudaBuffer.upload only supports start-of-buffer; we use
            # cudaMemcpy directly to write at an offset.
            import ctypes as _ct
            self._cudart.cudaMemcpy(
                _ct.c_void_p(self._enc_x_b2.ptr.value + b * Se * De * 2),
                _ct.c_void_p(slot_np.ctypes.data),
                slot_np.nbytes, 1)  # 1 = cudaMemcpyHostToDevice

        # Seed noise per slot (different RNG draws).
        noise_np = np.random.randn(self.B * Sa, 32).astype(np.float16)
        self._g_noise_b2.upload(noise_np)

        # Replay the B=N graph.
        self._enc_ae_graph_b2.replay(stream)
        self._cudart.cudaStreamSynchronize(stream)

        latency_ms = (_time.perf_counter() - t0) * 1000.0

        # Unpack per-slot actions.
        out_np = self._g_noise_b2.download_new((self.B * Sa, 32), np.float16)
        out_np = out_np.astype(np.float32)
        results = []
        for b in range(self.B):
            raw = out_np[b * Sa : (b + 1) * Sa]
            if self.norm_stats:
                unnorm = unnormalize_actions(raw, self.norm_stats)
                results.append({"actions": unnorm[:, :LIBERO_ACTION_DIM]})
            else:
                results.append({"actions": raw})
        return results

    # -----------------------------------------------------------------------
    # RL CFG inference (opt-in via set_rl_mode)
    # -----------------------------------------------------------------------

    def set_rl_mode(
        self,
        *,
        cfg_enable: bool = True,
        cfg_beta: float = 1.5,
        advantage_positive: bool = True,
    ) -> None:
        """Enable / configure advantage-conditioned RL inference (opt-in).

        Mirrors :meth:`Pi05TorchFrontendThor.set_rl_mode` and
        :meth:`Pi05TorchFrontendRtx.set_rl_mode`. The signature is
        byte-equal across the three frontends.
        """
        if not cfg_enable:
            self._rl_config = None
            self._lang_emb_cond_np = None
            self._lang_emb_uncond_np = None
            self._lang_emb_cond_dev = None
            self._lang_emb_uncond_dev = None
            # If a CFG-batched B=2 graph was previously captured (with
            # in-graph cfg_combine), invalidate it so the next batched
            # use recaptures without the CFG schedule baked in.
            if self._enc_ae_graph_b2_cfg_beta is not None:
                self._enc_ae_graph_b2_cfg_beta = None
                self._enc_ae_graph_b2 = None
            return
        if cfg_beta < 1.0:
            raise ValueError(
                f"cfg_beta must be >= 1.0 (1.0 disables CFG); got {cfg_beta}")
        new_config = {
            "cfg_beta": float(cfg_beta),
            "advantage_positive": bool(advantage_positive),
        }
        if self._rl_config != new_config:
            self._rl_config = new_config
        logger.info(
            "JAX RL mode enabled: cfg_beta=%.2f, advantage_positive=%s",
            new_config["cfg_beta"], new_config["advantage_positive"])

    def _ensure_cfg_buffers(self):
        CB = self._CudaBuffer
        n = self.Sa * 32
        if self._v_cond_buf is None:
            self._v_cond_buf = CB.device_empty(n, np.float16)
        if self._v_uncond_buf is None:
            self._v_uncond_buf = CB.device_empty(n, np.float16)

    def _ensure_lang_dev_buffers(self, target_len: int):
        """Allocate device-side cond / uncond ``lang_emb`` buffers.

        These shadow the captured-graph's ``self.lang_emb`` slot so the
        per-branch siglip replays can swap languages with a single
        graph-capturable D2D copy instead of a host→device upload of a
        numpy array (which is sync, not graph-capturable, and forces a
        host-block before the next op when the destination is managed
        memory on Thor's iGPU).
        """
        CB = self._CudaBuffer
        nbytes = target_len * self.De * 2  # fp16
        if (self._lang_emb_cond_dev is None
                or self._lang_emb_cond_dev.nbytes != nbytes):
            self._lang_emb_cond_dev = CB.device_empty(
                target_len * self.De, np.float16)
        if (self._lang_emb_uncond_dev is None
                or self._lang_emb_uncond_dev.nbytes != nbytes):
            self._lang_emb_uncond_dev = CB.device_empty(
                target_len * self.De, np.float16)

    def _set_prompt_rl(self, prompt_text):
        """Build cond + uncond embeds, drive standard set_prompt with
        cond_text (cond is always >= uncond in length), pad uncond to
        ``S_lang`` and stash both numpy embeds for runtime swap.

        The captured ``siglip_graph`` reads from ``self.lang_emb`` (a
        CudaBuffer); :meth:`_infer_cfg` ``upload``s either
        ``_lang_emb_cond_np`` or ``_lang_emb_uncond_np`` between the
        two siglip replays.
        """
        if isinstance(prompt_text, (np.ndarray, list)):
            raise ValueError(
                "set_rl_mode requires a text prompt (the ACP tag is "
                "appended at the string level); pass a str, not token IDs")
        from flash_rt.core.rl import build_acp_tagged_task

        cfg = self._rl_config
        cond_text = build_acp_tagged_task(
            prompt_text, is_positive=cfg["advantage_positive"])
        uncond_text = prompt_text

        # Drive the standard graph-capture path with cond_text.
        self.set_prompt(cond_text)
        target_len = self.S_lang

        uncond_np, uncond_len = _embed_prompt(
            uncond_text, self._embedding_np, max_len=48)
        if uncond_np.shape[0] > target_len:
            raise RuntimeError(
                f"uncond_len={uncond_np.shape[0]} > cond target_len="
                f"{target_len}; the ACP tag is supposed to make cond at "
                f"least as long as uncond")
        if uncond_np.shape[0] < target_len:
            pad = np.tile(uncond_np[-1:],
                          (target_len - uncond_np.shape[0], 1))
            uncond_np = np.concatenate([uncond_np, pad], axis=0)
        self._lang_emb_uncond_np = np.ascontiguousarray(
            uncond_np.astype(np.float16))
        # Snapshot cond contents the standard set_prompt populated.
        self._lang_emb_cond_np = self.lang_emb.download_new(
            (target_len, self.De), np.float16)
        self._rl_current_prompt_text = prompt_text

        self._ensure_cfg_buffers()
        # Pre-stage cond / uncond into dedicated device buffers so the
        # fused-CFG fast path can swap languages with one graph-capturable
        # D2D copy instead of a per-call host→device upload.
        self._ensure_lang_dev_buffers(target_len)
        self._lang_emb_cond_dev.upload(self._lang_emb_cond_np)
        self._lang_emb_uncond_dev.upload(self._lang_emb_uncond_np)

        # Per-call mutable noise scratch — captured by the snapshot /
        # restore closures so cond and uncond replays see identical R.
        # Allocated lazily on the first ``_infer_cfg`` call (see below).
        self._rl_noise_np_holder: list = [None]

        # Build the appropriate CFG pipeline based on whether batched
        # mode is active. Stage 0 (serial) when ``_batched`` is False;
        # Stage 3 (B=2 fused) when ``_batched`` is True.
        if self._batched:
            self._build_cfg_batched_pipeline_jax(cfg["cfg_beta"])
        else:
            self._build_cfg_serial_pipeline_jax(cfg["cfg_beta"])

        logger.info(
            "JAX RL prompt: '%s' (cond_len=%d, uncond_len=%d, padded=%d, "
            "cfg_beta=%.2f, batched=%s)",
            prompt_text, target_len, uncond_len, target_len,
            cfg["cfg_beta"], self._batched)

    def _build_cfg_serial_pipeline_jax(self, cfg_beta: float) -> None:
        """JAX serial CFG pipeline (Stage 0 — per-chunk, B=1 graphs)."""
        from flash_rt.models.pi05.pipeline_thor_cfg import (
            Pi05ThorCFGPipeline)
        stream = self._stream
        stream_int = stream.value or 0

        def _replay_siglip_jax():
            self.siglip_graph.replay(stream)
            self._cudart.cudaStreamSynchronize(stream)

        def _replay_enc_ae_jax():
            self.enc_ae_graph.replay(stream)
            self._cudart.cudaStreamSynchronize(stream)

        def _upload_cond_jax():
            self.lang_emb.upload(self._lang_emb_cond_np)

        def _upload_uncond_jax():
            self.lang_emb.upload(self._lang_emb_uncond_np)

        def _snapshot_noise_jax():
            self._rl_noise_np_holder[0] = self.g_noise.download_new(
                (self.Sa * 32,), np.float16)

        def _restore_noise_jax():
            if self._rl_noise_np_holder[0] is None:
                raise RuntimeError(
                    "snapshot_noise must precede restore_noise")
            self.g_noise.upload(self._rl_noise_np_holder[0])

        def _snap_to_v_cond_jax():
            arr = self.g_noise.download_new((self.Sa * 32,), np.float16)
            self._v_cond_buf.upload(arr)

        def _snap_to_v_uncond_jax():
            arr = self.g_noise.download_new((self.Sa * 32,), np.float16)
            self._v_uncond_buf.upload(arr)

        def _zero_g_noise_jax():
            self.g_noise.zero_(stream)
            self._cudart.cudaStreamSynchronize(stream)

        def _sync_jax():
            self._cudart.cudaStreamSynchronize(stream)

        from flash_rt import flash_rt_kernels as _fvk
        self._cfg_pipeline = Pi05ThorCFGPipeline(
            _fvk,
            cfg_beta=cfg_beta,
            Sa=int(self.Sa),
            replay_siglip=_replay_siglip_jax,
            replay_enc_ae=_replay_enc_ae_jax,
            upload_cond_lang_emb=_upload_cond_jax,
            upload_uncond_lang_emb=_upload_uncond_jax,
            snapshot_noise=_snapshot_noise_jax,
            restore_noise=_restore_noise_jax,
            snapshot_g_noise_to_v_cond=_snap_to_v_cond_jax,
            snapshot_g_noise_to_v_uncond=_snap_to_v_uncond_jax,
            zero_g_noise=_zero_g_noise_jax,
            g_noise_ptr=lambda: self.g_noise.ptr.value,
            v_cond_ptr=lambda: self._v_cond_buf.ptr.value,
            v_uncond_ptr=lambda: self._v_uncond_buf.ptr.value,
            sync=_sync_jax,
            stream_int=stream_int,
        )

    def _build_cfg_batched_pipeline_jax(self, cfg_beta: float) -> None:
        """JAX fused B=2 CFG pipeline (paper-correct per-step CFG, single graph).

        Recaptures the inner ``_enc_ae_graph_b2`` against ``cfg_beta``
        so the per-step ``cfg_combine_into_residual_fp16`` + noise
        mirror live INSIDE that graph. Then captures the OUTER fused
        pipeline (lang swap (×2) + SigLIP (×2) + enc_ae_b2) — at
        runtime the pipeline's ``forward()`` is one
        ``outer_graph.replay()`` + final sync, matching RTX
        :meth:`Pi05CFGBatchedPipeline.forward` /
        ``self._graph.replay(...)``.

        Changing ``cfg_beta`` requires rebuilding the pipeline (the
        beta is baked into the captured cfg_combine kernel calls).
        """
        from flash_rt.models.pi05.pipeline_thor_cfg_batched import (
            Pi05ThorCFGBatchedPipeline)
        # Recapture against the new cfg_beta. Any previous B=2 graph
        # was either non-CFG or captured against a stale beta.
        self._enc_ae_graph_b2_cfg_beta = float(cfg_beta)
        self._enc_ae_graph_b2 = None
        self._capture_enc_ae_graph_b2()
        # Capture the full fused-CFG pipeline as one outer graph. When
        # the frontend is constructed with ``autotune > 0``, recapture
        # N times and keep the fastest — same parameterisation as the
        # B=1 path (``self.autotune``).
        self._cfg_b2_outer_graph = None
        if self.autotune > 0:
            self._autotune_cfg_b2_outer_graph(
                n_trials=self.autotune, n_bench=10)
        else:
            self._capture_cfg_b2_outer_graph()

        stream = self._stream
        stream_int = stream.value or 0
        Se = self.Se
        Sa = self.Sa
        De = self.De
        import ctypes as _ct

        # Lang slot byte size for the captured siglip graph.
        lang_nbytes = self.S_lang * self.De * 2  # fp16

        def _siglip_for_cond():
            # D2D from pre-staged cond device buffer → captured lang_emb
            # slot, replay the B=1 SigLIP graph, then D2D enc_x[:Se]
            # → _enc_x_b2[0:Se]. All ops on the same stream.
            self._cudart.cudaMemcpy(
                _ct.c_void_p(self.lang_emb.ptr.value),
                _ct.c_void_p(self._lang_emb_cond_dev.ptr.value),
                lang_nbytes, 3)  # 3 = cudaMemcpyDeviceToDevice
            self.siglip_graph.replay(stream)
            self._cudart.cudaStreamSynchronize(stream)
            self._cudart.cudaMemcpy(
                _ct.c_void_p(self._enc_x_b2.ptr.value),
                _ct.c_void_p(self.enc_x.ptr.value),
                Se * De * 2, 3)

        def _siglip_for_uncond():
            self._cudart.cudaMemcpy(
                _ct.c_void_p(self.lang_emb.ptr.value),
                _ct.c_void_p(self._lang_emb_uncond_dev.ptr.value),
                lang_nbytes, 3)
            self.siglip_graph.replay(stream)
            self._cudart.cudaStreamSynchronize(stream)
            self._cudart.cudaMemcpy(
                _ct.c_void_p(self._enc_x_b2.ptr.value + Se * De * 2),
                _ct.c_void_p(self.enc_x.ptr.value),
                Se * De * 2, 3)

        def _seed_b2_noise_from_R():
            # Single noise R (Sa, 32) shared across both slots so the
            # per-step CFG mirror at end of step 0 starts from
            # identical x_t. Cross-backend cos contract requires the
            # numpy CPU RNG (matching the torch frontend).
            R = np.random.randn(Sa, 32).astype(np.float16)
            self._g_noise_b2.upload(R)  # slot 0 region
            self._cudart.cudaMemcpy(
                _ct.c_void_p(self._g_noise_b2.ptr.value + Sa * 32 * 2),
                _ct.c_void_p(self._g_noise_b2.ptr.value),
                Sa * 32 * 2, 3)

        def _replay_enc_ae_b2():
            self._enc_ae_graph_b2.replay(stream)
            self._cudart.cudaStreamSynchronize(stream)

        def _sync_jax():
            self._cudart.cudaStreamSynchronize(stream)

        from flash_rt import flash_rt_kernels as _fvk
        self._cfg_pipeline = Pi05ThorCFGBatchedPipeline(
            _fvk,
            cfg_beta=cfg_beta,
            Sa=int(self.Sa),
            replay_siglip_for_cond=_siglip_for_cond,
            replay_siglip_for_uncond=_siglip_for_uncond,
            replay_enc_ae_b2=_replay_enc_ae_b2,
            seed_b2_noise_from_R=_seed_b2_noise_from_R,
            sync=_sync_jax,
            stream_int=stream_int,
            outer_graph_replay=lambda: self._cfg_b2_outer_graph.replay(stream),
        )

    def _infer_cfg(self, observation, debug=False):
        """JAX CFG inference: 2x (siglip + enc_ae) + cfg_combine kernel.

        See :meth:`Pi05TorchFrontendThor._infer_cfg` for the algorithm.
        Differences from the torch version:
          * Buffers are CudaBuffers (``upload`` / ``download`` instead of
            ``copy_`` between torch tensors).
          * Stream-explicit replays: ``graph.replay(stream)`` followed by
            ``cudaStreamSynchronize``.
          * Initial random noise is generated on CPU via
            ``np.random.randn`` (matching the standard JAX infer path)
            so the same numpy array can be re-uploaded for the uncond
            branch.
        """
        from flash_rt.core.utils.actions import (
            unnormalize_actions, LIBERO_ACTION_DIM)
        nv = self.num_views
        stream = self._stream
        stream_int = stream.value or 0
        n = self.Sa * 32
        t0 = _time.perf_counter()

        if 'images' in observation:
            img_list = observation['images']
        else:
            img_list = [observation['image']]
            if nv >= 2:
                img_list.append(
                    observation.get('wrist_image', observation['image']))
            if nv >= 3:
                img_list.append(
                    observation.get('wrist_image_right', img_list[-1]))

        def _to_np16(im):
            if im.dtype == np.float16:
                return im
            return (np.asarray(im).astype(np.float32) / 127.5 - 1.0
                    ).astype(np.float16)
        images_np = np.stack([_to_np16(im) for im in img_list[:nv]])
        self.img_buf.upload(images_np)

        # SigLIP needs to run once to populate enc_x with real vision
        # tokens before the lazy-recal hook fires; the CFG pipeline
        # will run it again per branch. Hook only fires on the first
        # infer call, amortized.
        if not self._real_data_calibrated:
            self.lang_emb.upload(self._lang_emb_cond_np)
            self.siglip_graph.replay(stream)
            self._cudart.cudaStreamSynchronize(stream)
            self._recalibrate_with_real_data()
            self._real_data_calibrated = True
            # When batched, recalibrate also invalidates the B=2 graph:
            # ``_recalibrate_with_real_data`` reallocates ``enc_calib_scales``
            # / ``ae_calib_scales`` (new ``CudaBuffer`` instances), which
            # ``__del__`` cudaFrees the old ones the B=2 graph captured by
            # pointer. Recapture B=2 here against the fresh scales so the
            # next pipeline.run replay reads valid memory.
            if self._batched:
                self._enc_ae_graph_b2 = None
                self._capture_enc_ae_graph_b2()
                # The outer fused-CFG graph also captured the freed
                # scale buffers as constants; recapture against the
                # fresh ones, honouring autotune.
                if self._cfg_b2_outer_graph is not None:
                    self._cfg_b2_outer_graph = None
                    if self.autotune > 0:
                        self._autotune_cfg_b2_outer_graph(
                            n_trials=self.autotune, n_bench=10)
                    else:
                        self._capture_cfg_b2_outer_graph()

        if self._batched:
            # Stage 3 fused B=2 CFG (JAX). One outer-graph replay
            # (lang swap + SigLIP×2 + encoder_b2 + decoder_b2 with
            # per-step CFG) + final sync; result lives in
            # ``_g_noise_b2[0:Sa]``.
            self._cfg_pipeline.forward()
            raw_actions = self._g_noise_b2.download_new(
                (self.B * self.Sa, 32), np.float16
            )[:self.Sa].astype(np.float32)
        else:
            # Stage 0 serial CFG path — frontend pre-seeds noise R into
            # ``g_noise``; pipeline runs the dual-branch + combine into
            # ``g_noise``.
            noise_np = np.random.randn(self.Sa, 32).astype(np.float16)
            self.g_noise.upload(noise_np)
            self._cfg_pipeline.run_pipeline()
            raw_actions = self.g_noise.download_new(
                (self.Sa, 32), np.float16).astype(np.float32)

        latency_ms = (_time.perf_counter() - t0) * 1000
        if debug:
            logger.info(
                "JAX CFG infer: %.1f ms (beta=%.2f, batched=%s)",
                latency_ms, self._cfg_pipeline.cfg_beta, self._batched)
        if self.norm_stats:
            unnorm = unnormalize_actions(raw_actions, self.norm_stats)
            robot_actions = unnorm[:, :LIBERO_ACTION_DIM]
        else:
            robot_actions = raw_actions
        return {"actions": robot_actions}

    def calibrate(
        self,
        observations,
        *,
        percentile: float = 99.9,
        max_samples=None,
        verbose: bool = False,
    ) -> None:
        """Unified calibration API (JAX Thor).

        ``N=1`` falls back to the implicit single-frame path (bit-equal
        with the recalibration triggered on first ``infer()``).
        ``N>=2`` runs ``_calibrate_multi_frame``: per-sample shadow
        forwards through encoder + decoder, percentile-reduces per-tensor
        amax along the sample axis, uploads the reduced FP8 scales, and
        recaptures the enc/ae graph with the new scales baked in
        (mirrors ``flash_rt/frontends/torch/pi05_thor.py``).
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

        Drives encoder + decoder shadow forwards over each obs in
        ``obs_list``, downloads per-tensor amax, percentile-reduces along
        the sample axis via ``flash_rt.core.calibration.accumulate_amax``
        (framework rule C5), uploads the reduced scales, recomputes the
        enc-side alpha = act_scale * weight_scale in float32 (rule C6),
        and recaptures the enc+ae CUDA Graph (rule C7) so the new scales
        are baked into the captured kernels.

        Mirrors:
          * Pi0.5 torch Thor — ``flash_rt/frontends/torch/pi05_thor.py``
            ``_calibrate_multi_frame``
          * Pi0.5 JAX Thor FP4 — ``flash_rt/frontends/jax/pi05_thor_fp4.py``
            ``_calibrate_multi_frame`` (Phase 1 only; FP4 also runs an
            AWQ refit in Phase 2 which is FP4-specific).
        """
        from flash_rt.core.calibration import accumulate_amax

        n = len(obs_list)
        logger.info(
            "Pi0.5 JAX Thor multi-frame calibrate: N=%d, percentile=%.2f",
            n, percentile)

        CB = self._CudaBuffer
        stream = self._stream
        stream_int = stream.value or 0
        Se = int(self.Se); De = int(self.De); He = int(self.He)
        NHe = int(self.NHe); HDe = int(self.HDe)
        Le = int(self.Le); La = int(self.La)
        Sa = int(self.Sa); Da = int(self.Da); Ha = int(self.Ha)
        total_keys = int(self.total_keys)
        nv = self.num_views

        # Scratch buffers reused across samples.
        _norm_scratch = CB.device_empty(Se * De, np.float16)
        _x_scratch = CB.device_empty(Se * De, np.float16)
        _enc_calib_buf = CB.zeros(Le * 4, np.float32)
        _d_scale = CB.zeros(1, np.float32)
        _fp8_scratch = CB.device_zeros(Se * max(De, He), np.uint8)
        _ones_buf = CB.from_numpy(np.ones(De, dtype=np.float16))
        _ae_calib_buf = CB.zeros(La * 4, np.float32)
        _ae_d_scale = CB.zeros(1, np.float32)
        _ae_hidden_scratch = CB.device_empty(Sa * Ha, np.float16)
        _ae_fp8_scratch = CB.device_zeros(Sa * max(Da, Ha), np.uint8)

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

            # ── Decoder shadow forward (fixed seed per sample, rule C10) ──
            ae_scales_buf = CB.zeros(La * 4, np.float32)
            noise_np = np.random.default_rng(i).standard_normal(
                (Sa, 32)).astype(np.float16)
            self.g_noise.upload(noise_np)
            ae_bufs, ae_weights, ae_dims = self._build_ae_dicts(stream_int)
            ae_bufs['calib_buf'] = _ae_calib_buf.ptr.value
            ae_bufs['d_scale'] = _ae_d_scale.ptr.value
            ae_bufs['hidden_scratch'] = _ae_hidden_scratch.ptr.value
            ae_bufs['fp8_scratch'] = _ae_fp8_scratch.ptr.value
            decoder_forward_calibrate(
                self._ctx, self._fvk, ae_bufs, ae_weights, ae_dims,
                ae_scales_buf.ptr.value, stream=stream_int)
            self._cudart.cudaStreamSynchronize(stream)
            per_sample_ae.append(
                ae_scales_buf.download_new((La * 4,), np.float32))

            if verbose and (i + 1) % max(1, n // 10) == 0:
                logger.info("  sample %d/%d", i + 1, n)

        # Percentile-reduce along sample axis (rule C5).
        enc_final = accumulate_amax(per_sample_enc, percentile=percentile)
        ae_final = accumulate_amax(per_sample_ae, percentile=percentile)

        # Upload reduced scales + recompute alpha in float32 (rule C6).
        self.enc_calib_scales = CB.from_numpy_managed(enc_final)
        self.ae_calib_scales = CB.from_numpy_managed(ae_final)
        enc_ws_np = self.enc_w_dev.download_new((Le * 4,), np.float32)
        self.enc_alpha_host = [
            float(np.float32(enc_final[i]) * np.float32(enc_ws_np[i]))
            for i in range(Le * 4)
        ]

        # Recapture graph with fresh scales baked in (rule C7).
        self._capture_enc_ae_graph()
        self._real_data_calibrated = True
        logger.info(
            "Pi0.5 JAX Thor multi-frame calibrate complete (N=%d, "
            "percentile=%.2f)", n, percentile)

    def calibrate_with_real_data(self, sample_observations) -> None:
        """Legacy alias for :meth:`calibrate`."""
        self.calibrate(sample_observations)

    @property
    def precision_spec(self):
        """JAX Thor does not yet surface a structured PrecisionSpec."""
        return None

    def _recalibrate_with_real_data(self):
        """Recalibrate using real SigLIP output, then recapture enc+ae graph.

        Called once on first infer() — after SigLIP has processed a real image,
        so enc_x contains realistic activation distributions instead of zeros.
        """
        CB = self._CudaBuffer
        Se = self.Se; Le = self.Le; La = self.La
        total_keys = self.total_keys
        _cudart = self._cudart; stream = self._stream
        stream_int = stream.value or 0

        # Encoder recalibration with real enc_x
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
            'qkv_w': [self.ew[0].ptr.value + i * self.De * 2560 for i in range(Le)],
            'o_w': [self.ew[1].ptr.value + i * self.De * self.De for i in range(Le)],
            'gate_w': [self.ew[2].ptr.value + i * self.De * self.He * 2 for i in range(Le)],
            'down_w': [self.ew[4].ptr.value + i * self.He * self.De for i in range(Le)],
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
        Sa = self.Sa; Da = self.Da; Ha = self.Ha
        ae_scales_buf = CB.zeros(La * 4, np.float32)
        _ae_calib_buf = CB.zeros(La * 4, np.float32)
        _ae_d_scale = CB.zeros(1, np.float32)
        _ae_hidden_scratch = CB.device_empty(Sa * Ha, np.float16)
        _ae_fp8_scratch = CB.device_zeros(Sa * max(Da, Ha), np.uint8)

        noise_np = np.random.randn(Sa, 32).astype(np.float16)
        self.g_noise.upload(noise_np)

        ae_bufs, ae_weights, ae_dims = self._build_ae_dicts(stream_int)
        ae_bufs['calib_buf'] = _ae_calib_buf.ptr.value
        ae_bufs['d_scale'] = _ae_d_scale.ptr.value
        ae_bufs['hidden_scratch'] = _ae_hidden_scratch.ptr.value
        ae_bufs['fp8_scratch'] = _ae_fp8_scratch.ptr.value
        decoder_forward_calibrate(
            self._ctx, self._fvk, ae_bufs, ae_weights, ae_dims,
            ae_scales_buf.ptr.value, stream=stream_int)
        _cudart.cudaStreamSynchronize(stream)

        self.ae_calib_scales = ae_scales_buf

        # Recapture graph with updated scales
        self._capture_enc_ae_graph()
        logger.info("Recalibrated with real data + graph recaptured")

    def _autotune_enc_ae(self, n_trials=5, n_bench=10):
        """Autotune Enc+AE graph: recapture until fast schedule is found.

        CUDA Graph instantiation is non-deterministic on Thor. This recaptures
        the graph until a fast schedule is obtained or max trials are exhausted.
        The LAST captured graph is always used (no stale references).

        Benchmark replicates real infer() flow: SigLIP replay first (L2 cache),
        then Enc+AE. Uses wall-clock timing with stream sync.

        Called once per set_prompt(). Adds ~1-5s to startup.
        """
        _crt = self._cudart
        stream = self._stream

        # Prepare image for SigLIP (replicate real infer flow)
        dummy_img = np.zeros((self.num_views, 224, 224, 3), dtype=np.float16)
        self.img_buf.upload(dummy_img)

        logger.info("Autotune: up to %d trials for best Enc+AE graph...", n_trials)

        for trial in range(n_trials):
            self._capture_enc_ae_graph()

            # Benchmark with SigLIP in front (replicates real infer flow)
            latencies = []
            for _ in range(n_bench):
                noise_np = np.random.randn(self.Sa, 32).astype(np.float16)
                self.g_noise.upload(noise_np)
                self.siglip_graph.replay(stream)
                _crt.cudaStreamSynchronize(stream)

                t0 = _time.perf_counter()
                self.enc_ae_graph.replay(stream)
                _crt.cudaStreamSynchronize(stream)
                latencies.append((_time.perf_counter() - t0) * 1000)

            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            logger.info("  Trial %d: %.2f ms", trial, p50)

            # Accept if fast enough (< 38.5ms = within fast regime)
            if p50 < 38.5:
                logger.info("Autotune done: Enc+AE = %.2f ms (trial %d)", p50, trial)
                return

        logger.info("Autotune done: Enc+AE = %.2f ms (best of %d)", p50, n_trials)

    def infer(self, observation, debug=False):
        """Run inference: upload image → CUDA Graph replay (patch_embed in graph)."""
        if self._rl_config is not None:
            return self._infer_cfg(observation, debug)
        t0 = _time.perf_counter()

        # Collect images
        if 'images' in observation:
            img_list = observation['images']
        else:
            img_list = [observation['image'], observation['wrist_image']]
            if self.num_views >= 3 and 'wrist_image_right' in observation:
                img_list.append(observation['wrist_image_right'])

        # Normalize images to FP16 (CPU) and stack
        def _normalize(im):
            if im.dtype == np.float16:
                return im
            return (im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)

        images_np = np.stack([_normalize(im) for im in img_list[:self.num_views]])  # (nv,224,224,3)
        self.img_buf.upload(images_np)

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
            logger.info(f"JAX infer: {latency:.1f} ms")

        return {"actions": robot_actions}
