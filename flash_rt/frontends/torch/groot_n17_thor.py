"""GR00T N1.7 Thor torch frontend — ``GrootN17TorchFrontendThor``.

Public surface mirrors ``GrootTorchFrontendThor`` (N1.6):

* ``__init__(checkpoint_path, num_views, embodiment_tag, ...)``
* ``set_prompt(prompt: str, embodiment_tag: str | None = None)``
* ``infer(observation: dict) -> np.ndarray``  # (action_horizon=40, 132)
* ``predict(...)`` — alias kept for API parity
* ``get_latency_stats()``


This commit (Phase 3c.a) lands the foundation: ``__init__`` +
``_load_weights`` driven by ``MultiSafetensorsSource`` and the
declarative ``WEIGHT_SPEC`` (Phase 3a). ``set_prompt``/``infer`` are
still stubbed.
"""

from __future__ import annotations

import glob
import os
import pathlib
import warnings
from typing import Optional

import torch


class GrootN17TorchFrontendThor:
    """N1.7 Thor inference frontend.

    Phase 3c lands in 4 stages:
      a. ``_load_weights`` (this commit) — full WEIGHT_SPEC across 2 ckpt
         shards via MultiSafetensorsSource; per-embodiment slot slicing
         on the dense (32, ·, ·) tensors.
      b. ``set_prompt`` — Qwen3-VL processor, M-RoPE cos/sin (mrope_table),
         timestep emb, DiT cross-KV precompute, calibration cache.
      c. eager ``infer`` (no graphs).
      d. CUDA Graph capture for vit / llm / vl_self_attn / dit-per-step.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        num_views: int = 2,
        embodiment_tag: str = "oxe_droid_relative_eef_relative_joint",
        device: str = "cuda:0",
        load_strided_fmha: bool = True,
    ):
        from flash_rt.models.groot_n17.embodiments import (
            EMBODIMENT_TAG_TO_INDEX, EMBODIMENT_NUM_VIEWS,
        )

        self.checkpoint_path = str(checkpoint_path)
        self.num_views = int(num_views)
        self.embodiment_tag = str(embodiment_tag)
        self.device = device

        if embodiment_tag not in EMBODIMENT_TAG_TO_INDEX:
            raise ValueError(
                f"unknown embodiment_tag {embodiment_tag!r}; supported: "
                f"{sorted(EMBODIMENT_TAG_TO_INDEX)}")
        self._embodiment_id = EMBODIMENT_TAG_TO_INDEX[embodiment_tag]

        # Side-load the strided FMHA library (pi05_thor.py:126 production
        # pattern) — required for vit's multi-view fmha_strided_full.
        if load_strided_fmha:
            self._load_fmha_strided()

        self._load_weights()

    # ────────────────────────────────────────────────────────────────
    # Weight loading (Phase 3c.a)
    # ────────────────────────────────────────────────────────────────

    def _load_fmha_strided(self) -> None:
        import flash_rt.flash_rt_kernels as fvk
        candidates = [
            "/workspace/libfmha_fp16_strided.so",
            str(pathlib.Path(self.checkpoint_path).parent / "libfmha_fp16_strided.so"),
            str(pathlib.Path(__file__).parent.parent.parent / "libfmha_fp16_strided.so"),
            str(pathlib.Path(__file__).parent.parent.parent.parent / "build" / "libfmha_fp16_strided.so"),
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    fvk.load_fmha_strided_library(p)
                    return
                except Exception as e:
                    warnings.warn(f"load_fmha_strided_library({p}) failed: {e}")
        warnings.warn(
            "libfmha_fp16_strided.so not found; multi-view ViT FMHA will be a "
            "no-op (cos test will fail). Build it from source/3rd-party.")

    def _load_weights(self) -> None:
        """Run WEIGHT_SPEC against all ckpt shards; slice per-embodiment dense
        matrices on the host side post-load."""
        from flash_rt.executors.torch_weights import MultiSafetensorsSource
        from flash_rt.executors.weight_loader import WeightLoader
        from flash_rt.frontends.torch._groot_n17_thor_spec import WEIGHT_SPEC

        shards = sorted(
            glob.glob(os.path.join(self.checkpoint_path, "model-*.safetensors")))
        if not shards:
            raise FileNotFoundError(
                f"no model-*.safetensors shards in {self.checkpoint_path}")
        source = MultiSafetensorsSource(shards, device=self.device)
        WeightLoader(source=source, target=self, spec=WEIGHT_SPEC).run()

        # DiT weights are loaded FP8 per spec (Quant() in WEIGHT_SPEC). N1.7
        # ckpt is natively bfloat16, so we dequant directly to bf16 (not fp16):
        # w_bf16 = w_fp8.float() * weight_scale → bf16. Biases are likewise
        # cast to bf16 so bf16_nn_bias's epilogue can consume them in-place.
        for i in range(32):
            base = i * 7
            for attr_w, attr_b, scale_idx in [
                ("_dit_q_w",       "_dit_q_b",       base + 0),
                ("_dit_k_w",       "_dit_k_b",       base + 1),
                ("_dit_v_w",       "_dit_v_b",       base + 2),
                ("_dit_o_w",       "_dit_o_b",       base + 3),
                ("_dit_ada_w",     "_dit_ada_b",     base + 4),
                ("_dit_ff_proj_w", "_dit_ff_proj_b", base + 5),
                ("_dit_ff_down_w", "_dit_ff_down_b", base + 6),
            ]:
                w_list = getattr(self, attr_w)
                b_list = getattr(self, attr_b)
                w_list[i] = (w_list[i].float() * float(self._dit_alpha[scale_idx])).bfloat16().contiguous()
                b_list[i] = b_list[i].bfloat16().contiguous()

        # Per-embodiment slot slicing: WEIGHT_SPEC loads dense
        # (32, in, out) and (32, out) tensors; the pipeline's per-embodiment
        # encoders/decoder expect already-sliced (in, out) and (out,) ones.
        # Replace each ``_st_enc_*``, ``_ac_enc_*``, ``_ac_dec_*`` attr with
        # its slot-N slice, contiguous on device.
        slot = self._embodiment_id
        for name in (
            "_st_enc_l1_W", "_st_enc_l1_b", "_st_enc_l2_W", "_st_enc_l2_b",
            "_ac_enc_W1_W", "_ac_enc_W1_b", "_ac_enc_W2_W", "_ac_enc_W2_b",
            "_ac_enc_W3_W", "_ac_enc_W3_b",
            "_ac_dec_l1_W", "_ac_dec_l1_b", "_ac_dec_l2_W", "_ac_dec_l2_b",
        ):
            full = getattr(self, name)
            sliced = full[slot].contiguous()
            setattr(self, name, sliced)
            # Drop the (32, ...) tensor's reference so the unused slots
            # can be freed by the allocator.
            del full

        self._load_fp16_shadow_weights()

    def _load_fp16_shadow_weights(self) -> None:
        """Load real fp16 GEMM weights (not FP8-dequantized) for the
        calibration shadow. Skipping the FP8 round-trip removes the
        per-layer ~0.001 cosine drift from quant noise so the bake-time
        amax aggregation matches the production input distribution.

        Stored in ``self._fp16_shadow_weights`` keyed by
        ``(stage, layer_idx, name)``. Only ViT / DSM / LLM / VLSA stages
        are covered (DiT does not run in the shadow). Released by
        ``set_prompt`` immediately after the calibration chain finishes.
        """
        from safetensors import safe_open

        shards = sorted(
            glob.glob(os.path.join(self.checkpoint_path, "model-*.safetensors")))
        handles = [safe_open(p, framework="pt", device=self.device) for p in shards]
        index: dict = {}
        for h in handles:
            for k in h.keys():
                index[k] = h

        def load_w(key: str) -> torch.Tensor:
            # Mirror spec ops: ToFp16 → T (transpose dim 0/1).
            t = index[key].get_tensor(key).to(torch.float16)
            return t.t().contiguous()

        shadow: dict = {}

        # ── ViT 24L: qkv, o, fc1, fc2 ──────────────────────────────────
        vp = "backbone.model.model.visual.blocks.{i}"
        for i in range(24):
            shadow[("vit", i, "qkv")] = load_w(f"{vp.format(i=i)}.attn.qkv.weight")
            shadow[("vit", i, "o")]   = load_w(f"{vp.format(i=i)}.attn.proj.weight")
            shadow[("vit", i, "fc1")] = load_w(f"{vp.format(i=i)}.mlp.linear_fc1.weight")
            shadow[("vit", i, "fc2")] = load_w(f"{vp.format(i=i)}.mlp.linear_fc2.weight")

        # ── DSM 3 mergers: fc1, fc2 ────────────────────────────────────
        dsm = "backbone.model.model.visual.deepstack_merger_list.{j}"
        for j in range(3):
            shadow[("dsm", j, "fc1")] = load_w(f"{dsm.format(j=j)}.linear_fc1.weight")
            shadow[("dsm", j, "fc2")] = load_w(f"{dsm.format(j=j)}.linear_fc2.weight")

        # ── LLM 16L: qkv (Cat[q,k,v] then T), o, gate, up, down ────────
        # Cat is along dim=0 BEFORE transpose, matching spec FusedQKV.
        lp = "backbone.model.model.language_model.layers.{i}"
        for i in range(16):
            q = index[f"{lp.format(i=i)}.self_attn.q_proj.weight"].get_tensor(
                f"{lp.format(i=i)}.self_attn.q_proj.weight").to(torch.float16)
            k = index[f"{lp.format(i=i)}.self_attn.k_proj.weight"].get_tensor(
                f"{lp.format(i=i)}.self_attn.k_proj.weight").to(torch.float16)
            v = index[f"{lp.format(i=i)}.self_attn.v_proj.weight"].get_tensor(
                f"{lp.format(i=i)}.self_attn.v_proj.weight").to(torch.float16)
            shadow[("llm", i, "qkv")] = torch.cat([q, k, v], dim=0).t().contiguous()
            shadow[("llm", i, "o")]    = load_w(f"{lp.format(i=i)}.self_attn.o_proj.weight")
            shadow[("llm", i, "gate")] = load_w(f"{lp.format(i=i)}.mlp.gate_proj.weight")
            shadow[("llm", i, "up")]   = load_w(f"{lp.format(i=i)}.mlp.up_proj.weight")
            shadow[("llm", i, "down")] = load_w(f"{lp.format(i=i)}.mlp.down_proj.weight")

        # ── VLSA 4L: q, k, v, o, fc1, fc2 ──────────────────────────────
        vlp = "action_head.vl_self_attention.transformer_blocks.{i}"
        for i in range(4):
            shadow[("vlsa", i, "q")]   = load_w(f"{vlp.format(i=i)}.attn1.to_q.weight")
            shadow[("vlsa", i, "k")]   = load_w(f"{vlp.format(i=i)}.attn1.to_k.weight")
            shadow[("vlsa", i, "v")]   = load_w(f"{vlp.format(i=i)}.attn1.to_v.weight")
            shadow[("vlsa", i, "o")]   = load_w(f"{vlp.format(i=i)}.attn1.to_out.0.weight")
            shadow[("vlsa", i, "fc1")] = load_w(f"{vlp.format(i=i)}.ff.net.0.proj.weight")
            shadow[("vlsa", i, "fc2")] = load_w(f"{vlp.format(i=i)}.ff.net.2.weight")

        self._fp16_shadow_weights = shadow

    # ────────────────────────────────────────────────────────────────
    # Phase 3c.b/c/d — pending
    # ────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────
    # Phase 3c.b2 — set_prompt
    # ────────────────────────────────────────────────────────────────

    def set_prompt(
        self,
        *,
        aux: dict,
        prompt: str | None = None,
    ) -> None:
        """Run the calibration shadow + bake FP8 alphas + cache.

        For 3c.b2 ``aux`` is the bundle of HF-derived setup tensors (the
        same shape produced by ``tests/_helpers/groot_n17/capture_llm_aux.py``):

          * ``input_ids``           — (1, S)  int64
          * ``visual_pos_masks``    — (1, S)  bool
          * ``position_ids``        — (3, 1, S)  int64 (M-RoPE T/H/W)
          * ``rope_cos``, ``rope_sin`` — (1, S, HD=128) bf16 (HF rotary_emb output)
          * ``llm_input_embeds``    — (1, S, 2048) fp32 (input to truncated LLM)
          * ``pixel_features``      — (S_vit=1024, 1024) fp32 (post-patch_embed+pos_embed)
          * ``grid_thw``            — (num_views, 3) int64

        A future revision (3c.c+) will derive ``aux`` end-to-end from raw
        ``(prompt, sample_obs)`` via the HF Qwen3VL processor + vision
        model. For 3c.b2 we accept the pre-captured form so the calibration
        path can land independently of the production preprocessing path.

        After this call, the frontend has:

          * ``self._<stage>_act_scale_dev[i]``  — per-layer fp32 dev scalar ptrs
          * ``self._<stage>_alpha[i]``           — per-layer host floats
          * ``self._mrope_cos / _mrope_sin``     — fp16 device (S, HD)
          * ``self._vit_cos / _vit_sin``         — fp16 device (S_vit, HD)
          * ``self._backbone_features``          — fp16 device (1, S, 2048)
          * ``self._visual_pos_masks``           — bool device (S,)
          * ``self.Se``                          — int = S
        """
        if hasattr(self, "_backbone_features"):
            raise RuntimeError(
                "set_prompt() after GROOT N1.7 prompt runtime initialization "
                "is not supported; construct a new frontend instance for a new prompt")

        from flash_rt.models.groot_n17 import calibration as cal
        from flash_rt.models.groot_n17.calibration import build_vit_rope_tables

        self._prompt = prompt
        self.Se = int(aux["llm_input_embeds"].shape[1])
        device = self.device

        # ── M-RoPE cos/sin: re-use HF's captured tables (already correct) ──
        self._mrope_cos = aux["rope_cos"][0].to(device).half().contiguous()
        self._mrope_sin = aux["rope_sin"][0].to(device).half().contiguous()

        # ── ViT 2D rope cos/sin from grid_thw ──────────────────────────
        grid_thw = [tuple(int(x) for x in row) for row in aux["grid_thw"].tolist()]
        vit_cos, vit_sin = build_vit_rope_tables(
            grid_thw, head_dim=64, theta=10000.0, spatial_merge_size=2,
            device=device,
        )
        self._vit_cos = vit_cos
        self._vit_sin = vit_sin
        self._num_vit_views = len(grid_thw)
        self._S_vit = sum(int(t * h * w) for t, h, w in grid_thw)
        self._S_vit_per_view = self._S_vit // self._num_vit_views

        # ── visual mask + LLM input ────────────────────────────────────
        self._visual_pos_masks = aux["visual_pos_masks"][0].to(device)
        llm_input = aux["llm_input_embeds"].to(device).float()  # (1, S, 2048)

        # ── Calibration shadow chain ───────────────────────────────────
        pixel_features = aux["pixel_features"].to(device).float()
        out_vit = cal.calibrate_vit(
            self, pixel_features, vit_cos.float(), vit_sin.float(),
            num_views=self._num_vit_views,
        )
        out_ds = cal.calibrate_deepstack(self, out_vit["deepstack_taps"])
        out_llm = cal.calibrate_llm(
            self, llm_input,
            self._mrope_cos.float(), self._mrope_sin.float(),
            self._visual_pos_masks, out_ds["features"],
        )
        out_vlsa = cal.calibrate_vlsa(self, out_llm["llm_final"])

        # Release fp16 shadow weight refs now that the calibration chain
        # is done — no other code path consumes them.
        if hasattr(self, "_fp16_shadow_weights"):
            del self._fp16_shadow_weights
            torch.cuda.empty_cache()

        # ── Bake d_act_scale device tensors + host alphas ──────────────
        self._bake_calibration(out_vit, out_ds, out_llm, out_vlsa)

        # ── Stash backbone for infer; also stash deepstack injection bufs ──
        self._backbone_features = out_vlsa["backbone_features"].half()
        # Pre-build the (S, D) DeepStack injection buffers (zero except at
        # visual positions) — used by qwen3vl_llm_forward.
        self._deepstack_inject = []
        for j in range(3):
            buf = torch.zeros(self.Se, 2048, dtype=torch.float16, device=device)
            buf[self._visual_pos_masks] = out_ds["features"][j].half()
            self._deepstack_inject.append(buf)

        # ── Save cache (R3 4-line template) ────────────────────────────
        self._save_calibration_cache(out_vit, out_ds, out_llm, out_vlsa)

        # ── Warmup: prime cuBLAS workspace + DiT lazy init ─────────────
        # Cold-start NaN observed under specific noise distributions
        # (e.g. HF's captured initial_noise). One eager infer call with
        # safe seeded noise primes the cuBLAS heuristics so subsequent
        # production infer calls hit a steady state.
        try:
            self._warmup_infer()
        except Exception as e:
            warnings.warn(f"set_prompt warmup failed (non-fatal): {e!r}")

    def _warmup_infer(self) -> None:
        """Single dry-run infer to prime cuBLAS / lazy-init DiT attn."""
        warm_state = torch.zeros(1, 1, 132, dtype=torch.float32)
        torch.manual_seed(0)
        warm_noise = torch.randn(1, 40, 132, dtype=torch.bfloat16, device=self.device)
        _ = self.infer(warm_state, initial_noise=warm_noise)

    # ────────────────────────────────────────────────────────────────
    # Phase 3c.d — CUDA Graph capture of the 32-layer DiT inner loop
    # ────────────────────────────────────────────────────────────────

    def _precompute_diffusion_modulators(
        self, num_inference_timesteps: int = 4,
        num_timestep_buckets: int = 1000,
    ) -> None:
        """Pre-compute every step-dependent quantity feeding the DiT inner
        loop so the graph capture sees a stable pointer set per step.

        Outputs (stashed on self):
          * ``_step_temb``      — list[num_steps] of (1, D=1536) bf16
          * ``_step_shifts``    — list[num_steps] of list[32] of (D,) bf16
          * ``_step_scales``    — list[num_steps] of list[32] of (D,) bf16
        Each underlying tensor is contiguous, allocated once, and reused
        for the lifetime of this frontend (so .data_ptr() values baked
        into a captured CUDA graph remain valid).
        """
        self._step_temb: list = []
        self._step_shifts: list = []
        self._step_scales: list = []
        for step in range(num_inference_timesteps):
            t_disc = int(step / num_inference_timesteps * num_timestep_buckets)
            temb = self._compute_timestep_emb(t_disc)
            shifts, scales = self._compute_dit_adaln_modulators(temb)
            self._step_temb.append(temb)
            self._step_shifts.append(shifts)
            self._step_scales.append(scales)

    def _capture_dit_graphs(self, num_inference_timesteps: int = 4,
                              action_horizon: int = 40) -> None:
        """Capture one CUDA graph per diffusion step. Each graph bakes that
        step's per-layer (shift, scale) modulator pointer set; the dit_h
        buffer is shared so callers populate it (sa_embs.copy_) before
        replay and read it after.

        Strict capture rules (PyTorch CUDA graph):
          * Capture stream must be non-default; we use one fresh
            torch.cuda.Stream per graph.
          * Buffers referenced by the captured kernels must be
            pre-allocated and reused — modulator tensors go through
            ``_precompute_diffusion_modulators``; the DiT scratch
            buffers (dit_h / dit_xn / dit_o_proj_out / dit_ff_proj_out)
            and attention slots come from ``_allocate_infer_buffers`` /
            ``_build_dit_attn``.
          * cuBLAS workspace state must be primed first — we run 3
            eager dit_forward iterations on the same buffer set.
        """
        from flash_rt.models.groot_n17 import pipeline_thor

        Sa = action_horizon + 1
        if not hasattr(self, "_infer_bufs"):
            self._allocate_infer_buffers(action_horizon)
        if not hasattr(self, "_dit_attn"):
            self._build_dit_attn(Sa)
        if not hasattr(self, "_step_shifts"):
            self._precompute_diffusion_modulators(
                num_inference_timesteps=num_inference_timesteps)
        if not hasattr(self, "_gemm"):
            import flash_rt.flash_rt_kernels as _fvk
            self._fvk = _fvk
            self._gemm = _fvk.GemmRunner()

        bufs = self._infer_bufs
        Skv_text = int(self._dit_cross_K[0].shape[0])
        Skv_image = int(self._dit_cross_K[1].shape[0])
        dims = {"Sa": Sa, "D": 1536, "FF": 6144,
                "Skv_text": Skv_text, "Skv_image": Skv_image}
        bufs_ptrs = {
            "h": bufs["dit_h"].data_ptr(),
            "xn": bufs["dit_xn"].data_ptr(),
            "o_proj_out": bufs["dit_o_proj_out"].data_ptr(),
            "ff_proj_out": bufs["dit_ff_proj_out"].data_ptr(),
        }

        # Per-step weights dict (pointers baked into each graph).
        def _weights_for(step: int) -> dict:
            return {
                "scale_msa": [t.data_ptr() for t in self._step_scales[step]],
                "shift_msa": [t.data_ptr() for t in self._step_shifts[step]],
                "q_w": [w.data_ptr() for w in self._dit_q_w],
                "q_b": [b.data_ptr() for b in self._dit_q_b],
                "k_w": [w.data_ptr() for w in self._dit_k_w],
                "k_b": [b.data_ptr() for b in self._dit_k_b],
                "v_w": [w.data_ptr() for w in self._dit_v_w],
                "v_b": [b.data_ptr() for b in self._dit_v_b],
                "o_w": [w.data_ptr() for w in self._dit_o_w],
                "o_b": [b.data_ptr() for b in self._dit_o_b],
                "ff_proj_w": [w.data_ptr() for w in self._dit_ff_proj_w],
                "ff_proj_b": [b.data_ptr() for b in self._dit_ff_proj_b],
                "ff_down_w": [w.data_ptr() for w in self._dit_ff_down_w],
                "ff_down_b": [b.data_ptr() for b in self._dit_ff_down_b],
            }

        # Warmup: 3 eager step-0 dit_forwards prime cuBLAS heuristics.
        weights_warm = _weights_for(0)
        for _ in range(3):
            pipeline_thor.dit_forward(
                gemm=self._gemm, fvk=self._fvk,
                bufs=bufs_ptrs, weights=weights_warm, dims=dims,
                attn=self._dit_attn,
            )
        torch.cuda.synchronize()

        # Capture one graph per step on a fresh stream.
        self._dit_graphs: list = []
        for step in range(num_inference_timesteps):
            weights = _weights_for(step)
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            s_int = stream.cuda_stream
            with torch.cuda.stream(stream):
                graph.capture_begin()
                pipeline_thor.dit_forward(
                    gemm=self._gemm, fvk=self._fvk,
                    bufs=bufs_ptrs, weights=weights, dims=dims,
                    attn=self._dit_attn, stream=s_int,
                )
                graph.capture_end()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self._dit_graphs.append(graph)

    # ────────────────────────────────────────────────────────────────
    # Fully-kernelized DiT inner loop — the entire per-step chain
    # (action_encode + cat + DiT + output_proj + decode + Euler) plus
    # the once-per-frame state_encode run as CUDA graphs over persistent
    # kernel buffers. No PyTorch compute on the per-frame hot path; the
    # only host work between frames is two device copies (state, noise)
    # and the graph replays.
    #
    # All MLP weights are materialized once as bf16 [K, N]; the DiT main
    # path is bf16 throughout, matching the ckpt's native dtype. The
    # action-encoder's SiLU is evaluated in fp16 (cast in/out) since the
    # kernel library exposes SiLU only for fp16 — fp16 carries more
    # mantissa than bf16 so this is at least as accurate as a bf16 SiLU.
    # ────────────────────────────────────────────────────────────────

    def _prepare_kernel_dit(self, num_inference_timesteps: int = 4) -> None:
        """Materialize bf16 MLP weights + per-step constants (timestep
        concat halves and output-projection AdaLN shift/scale) used by the
        kernelized DiT graphs. Image-independent → computed once."""
        if hasattr(self, "_kw"):
            return
        import math
        dev = self.device
        def b16(t):
            return t.to(torch.bfloat16).contiguous()
        alpha = self._dit_misc_alpha
        self._kw = {
            "st_l1": b16(self._st_enc_l1_W), "st_l1b": b16(self._st_enc_l1_b),
            "st_l2": b16(self._st_enc_l2_W), "st_l2b": b16(self._st_enc_l2_b),
            "ae_W1": b16(self._ac_enc_W1_W), "ae_b1": b16(self._ac_enc_W1_b),
            "ae_W2": b16(self._ac_enc_W2_W), "ae_b2": b16(self._ac_enc_W2_b),
            "ae_W3": b16(self._ac_enc_W3_W), "ae_b3": b16(self._ac_enc_W3_b),
            # proj_out_2 is stored FP8; dequant via misc alpha[3] → bf16.
            "po2": b16(self._proj_out_2_w.float() * alpha[3]),
            "po2b": b16(self._proj_out_2_b),
            "dec_l1": b16(self._ac_dec_l1_W), "dec_l1b": b16(self._ac_dec_l1_b),
            "dec_l2": b16(self._ac_dec_l2_W), "dec_l2b": b16(self._ac_dec_l2_b),
        }
        self._k_pos = b16(self._ah_pos_embed_w[:40])

        if not hasattr(self, "_step_temb"):
            self._precompute_diffusion_modulators(
                num_inference_timesteps=num_inference_timesteps)

        # action-encoder timestep embedding (SinusoidalPositionalEncoding,
        # broadcast over the action sequence) — the right half of the
        # encoder's concat input. Matches ``_run_action_encode``.
        half = 768
        ex = (-torch.arange(half, dtype=torch.float32, device=dev)
              * (math.log(10000.0) / half)).exp()
        po1w = self._proj_out_1_w.float() * alpha[2]
        po1b = self._proj_out_1_b.float()
        self._k_tau, self._k_oproj_shift, self._k_oproj_scale = [], [], []
        for s in range(num_inference_timesteps):
            t_disc = int(s / num_inference_timesteps * 1000)
            ts = torch.full((40,), float(t_disc), dtype=torch.float32, device=dev)
            fr = ts.unsqueeze(-1) * ex
            self._k_tau.append(
                torch.cat([torch.sin(fr), torch.cos(fr)], dim=-1)
                .to(torch.bfloat16).contiguous())
            # output_proj AdaLN modulators: silu(temb) @ proj_out_1 → (shift, scale)
            x = torch.nn.functional.silu(self._step_temb[s].float())
            mod = x @ po1w + po1b
            shift, scale = mod.chunk(2, dim=-1)
            self._k_oproj_shift.append(b16(shift.squeeze(0)))
            self._k_oproj_scale.append(b16(scale.squeeze(0)))

    def _allocate_kernel_dit_buffers(self, action_horizon: int = 40) -> None:
        """Persistent bf16 scratch for the kernelized DiT graphs (allocated
        once so the data_ptr()s baked into the captured graphs stay valid)."""
        if hasattr(self, "_k_actions"):
            return
        dev = self.device
        bf = torch.bfloat16
        def e(*shape, dtype=bf):
            return torch.empty(*shape, dtype=dtype, device=dev)
        self._k_state_in = torch.zeros(1, 132, dtype=bf, device=dev)
        self._k_st_h1 = e(1, 1024)
        self._k_state_feat = e(1, 1536)
        self._k_actions = torch.zeros(action_horizon, 132, dtype=bf, device=dev)
        self._k_ae_aemb = e(action_horizon, 1536)
        self._k_ae_concat = e(action_horizon, 3072)
        self._k_ae_i2 = e(action_horizon, 1536)
        self._k_ae_i2f = e(action_horizon, 1536, dtype=torch.float16)
        self._k_ae_out = e(action_horizon, 1536)
        self._k_hmod = e(action_horizon + 1, 1536)
        self._k_hout = e(action_horizon + 1, 1024)
        self._k_dec_h = e(action_horizon, 1024)
        self._k_vel = e(action_horizon, 132)
        if not hasattr(self, "_mlp_gemm"):
            self._mlp_gemm = self._fvk.GemmRunner()

    def _capture_kernel_dit_graphs(self, num_inference_timesteps: int = 4,
                                    action_horizon: int = 40) -> None:
        """Capture the once-per-frame state encode and the full per-step DiT
        chain as CUDA graphs. Replaying them (after copying the normalized
        state and the initial noise into the persistent input buffers) yields
        the denoised actions with zero PyTorch compute in between."""
        from flash_rt.models.groot_n17 import pipeline_thor

        Sa = action_horizon + 1
        if not hasattr(self, "_gemm"):
            import flash_rt.flash_rt_kernels as _fvk
            self._fvk = _fvk
            self._gemm = _fvk.GemmRunner()
        # Kernelized cross-KV: allocates the persistent K/V buffers (fixed
        # data_ptrs) that the attention backend and the cross-KV graph share.
        # Must run before _build_dit_attn so the backend wires those ptrs.
        self._setup_cross_kv_kernel()
        if not hasattr(self, "_infer_bufs"):
            self._allocate_infer_buffers(action_horizon)
        if not hasattr(self, "_dit_attn"):
            self._build_dit_attn(Sa)
        self._prepare_kernel_dit(num_inference_timesteps)
        self._allocate_kernel_dit_buffers(action_horizon)

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
            return d
        step_weights = [_dit_weights(s) for s in range(num_inference_timesteps)]

        def _state_fwd(s):
            mg.bf16_nn(self._k_state_in.data_ptr(), w["st_l1"].data_ptr(),
                       self._k_st_h1.data_ptr(), 1, 1024, 132, s)
            K.add_bias_bf16(self._k_st_h1.data_ptr(), w["st_l1b"].data_ptr(), 1, 1024, s)
            K.relu_inplace_fp16(self._k_st_h1.data_ptr(), 1024, s)
            mg.bf16_nn(self._k_st_h1.data_ptr(), w["st_l2"].data_ptr(),
                       self._k_state_feat.data_ptr(), 1, 1536, 1024, s)
            K.add_bias_bf16(self._k_state_feat.data_ptr(), w["st_l2b"].data_ptr(), 1, 1536, s)

        def _step_fwd(step, s):
            T = action_horizon
            # action_encode: W1 (no act) → cat[a_emb, tau] → W2 → SiLU → W3
            mg.bf16_nn(self._k_actions.data_ptr(), w["ae_W1"].data_ptr(),
                       self._k_ae_aemb.data_ptr(), T, 1536, 132, s)
            K.add_bias_bf16(self._k_ae_aemb.data_ptr(), w["ae_b1"].data_ptr(), T, 1536, s)
            K.concat2_bf16(self._k_ae_aemb.data_ptr(), self._k_tau[step].data_ptr(),
                           self._k_ae_concat.data_ptr(), T, 1536, 1536, s)
            mg.bf16_nn(self._k_ae_concat.data_ptr(), w["ae_W2"].data_ptr(),
                       self._k_ae_i2.data_ptr(), T, 1536, 3072, s)
            K.add_bias_bf16(self._k_ae_i2.data_ptr(), w["ae_b2"].data_ptr(), T, 1536, s)
            K.cast_bf16_to_fp16(self._k_ae_i2.data_ptr(), self._k_ae_i2f.data_ptr(), T * 1536, s)
            K.silu_inplace_fp16(self._k_ae_i2f.data_ptr(), T * 1536, s)
            K.cast_fp16_to_bf16(self._k_ae_i2f.data_ptr(), self._k_ae_i2.data_ptr(), T * 1536, s)
            mg.bf16_nn(self._k_ae_i2.data_ptr(), w["ae_W3"].data_ptr(),
                       self._k_ae_out.data_ptr(), T, 1536, 1536, s)
            K.add_bias_bf16(self._k_ae_out.data_ptr(), w["ae_b3"].data_ptr(), T, 1536, s)
            K.residual_add(self._k_ae_out.data_ptr(), self._k_pos.data_ptr(), T * 1536, s)
            # DiT input: dit_h[0] = state, dit_h[1:] = action features
            K.gpu_copy(dit_h, self._k_state_feat.data_ptr(), 1536 * 2, s)
            K.gpu_copy(dit_h + 1536 * 2, self._k_ae_out.data_ptr(), T * 1536 * 2, s)
            pipeline_thor.dit_forward(
                gemm=self._gemm, fvk=K, bufs=bp, weights=step_weights[step],
                dims=dims, attn=self._dit_attn, stream=s)
            # output projection: AdaLN(dit_h) → proj_out_2
            K.ada_layer_norm_bf16(dit_h, self._k_oproj_scale[step].data_ptr(),
                                  self._k_oproj_shift[step].data_ptr(),
                                  self._k_hmod.data_ptr(), Sa, 1536, 1e-5, s)
            mg.bf16_nn(self._k_hmod.data_ptr(), w["po2"].data_ptr(),
                       self._k_hout.data_ptr(), Sa, 1024, 1536, s)
            K.add_bias_bf16(self._k_hout.data_ptr(), w["po2b"].data_ptr(), Sa, 1024, s)
            # action_decode on the action rows → velocity
            mg.bf16_nn(hout_dec, w["dec_l1"].data_ptr(),
                       self._k_dec_h.data_ptr(), T, 1024, 1024, s)
            K.add_bias_bf16(self._k_dec_h.data_ptr(), w["dec_l1b"].data_ptr(), T, 1024, s)
            K.relu_inplace_fp16(self._k_dec_h.data_ptr(), T * 1024, s)
            mg.bf16_nn(self._k_dec_h.data_ptr(), w["dec_l2"].data_ptr(),
                       self._k_vel.data_ptr(), T, 132, 1024, s)
            K.add_bias_bf16(self._k_vel.data_ptr(), w["dec_l2b"].data_ptr(), T, 132, s)
            # Euler flow update (in place)
            K.euler_step_bf16_out(self._k_actions.data_ptr(), self._k_vel.data_ptr(),
                                  self._k_actions.data_ptr(), dt, T * 132, s)

        self._kdit_fwd = (_state_fwd, _step_fwd)

        # Warmup on the default stream to prime both GemmRunner algo caches
        # (cross-KV, state and per-step shapes all selected here so the
        # captured graphs bake stable, working cuBLASLt algos).
        for _ in range(3):
            self._cross_kv_fwd(0)
            _state_fwd(0)
            for step in range(num_inference_timesteps):
                _step_fwd(step, 0)
        torch.cuda.synchronize()

        def _capture(fn):
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.capture_begin()
                fn(stream.cuda_stream)
                graph.capture_end()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            return graph

        # Single combined graph for the entire action-head side: cross-KV
        # refresh + state encode + all diffusion steps. The three per-frame
        # inputs (_ck_bb_src, _k_state_in, _k_actions) are written before the
        # one replay. Matches the Pi0.5 two-graph e2e structure (vision graph
        # + action graph). _nsteps records the captured step count.
        self._k_nsteps = num_inference_timesteps

        def _dit_all(s):
            self._cross_kv_fwd(s)
            _state_fwd(s)
            for step in range(num_inference_timesteps):
                _step_fwd(step, s)

        self._k_dit_graph = _capture(_dit_all)

    def _bake_calibration(self, out_vit, out_ds, out_llm, out_vlsa) -> None:
        """Convert per-stage amax dicts to per-layer device d_act_scale tensors
        and host alphas. After this, the production forwards have everything
        they need in ``self.<stage>_*`` attrs."""
        from flash_rt.models.groot_n17 import calibration as cal
        device = self.device

        def to_devs(amaxes):
            return [cal.amax_to_dev_scale(a, device=device) for a in amaxes]

        # ── ViT (24L × 4 quants per layer; alpha = act × weight) ───────
        self._vit_act_qkv_dev = to_devs(out_vit["vit_act_qkv"])
        self._vit_act_o_dev   = to_devs(out_vit["vit_act_o"])
        self._vit_act_fc1_dev = to_devs(out_vit["vit_act_fc1"])
        self._vit_act_fc2_dev = to_devs(out_vit["vit_act_fc2"])
        self._vit_alpha_q = [cal.alpha(out_vit["vit_act_qkv"][i], self._vit_alpha[i*4+0]) for i in range(24)]
        self._vit_alpha_o = [cal.alpha(out_vit["vit_act_o"][i],   self._vit_alpha[i*4+1]) for i in range(24)]
        self._vit_alpha_fc1 = [cal.alpha(out_vit["vit_act_fc1"][i], self._vit_alpha[i*4+2]) for i in range(24)]
        self._vit_alpha_fc2 = [cal.alpha(out_vit["vit_act_fc2"][i], self._vit_alpha[i*4+3]) for i in range(24)]

        # ── DeepStack (3 mergers × 2 quants) ───────────────────────────
        self._dsm_act_fc1_dev = to_devs(out_ds["deepstack_act_fc1"])
        self._dsm_act_fc2_dev = to_devs(out_ds["deepstack_act_fc2"])
        self._dsm_alpha_fc1 = [cal.alpha(out_ds["deepstack_act_fc1"][j], self._dsm_alpha[j*2+0]) for j in range(3)]
        self._dsm_alpha_fc2 = [cal.alpha(out_ds["deepstack_act_fc2"][j], self._dsm_alpha[j*2+1]) for j in range(3)]

        # ── LLM (16L × 4 distinct act scales; 5 alphas/layer for 5 GEMMs) ──
        self._llm_act_qkv_dev    = to_devs(out_llm["llm_act_qkv"])
        self._llm_act_o_dev      = to_devs(out_llm["llm_act_o"])
        self._llm_act_gateup_dev = to_devs(out_llm["llm_act_gateup"])
        self._llm_act_down_dev   = to_devs(out_llm["llm_act_down"])
        # _llm_alpha layout: [qkv, o, gate, up, down] per layer
        self._llm_alpha_qkv  = [cal.alpha(out_llm["llm_act_qkv"][i], self._llm_alpha[i*5+0]) for i in range(16)]
        self._llm_alpha_o    = [cal.alpha(out_llm["llm_act_o"][i],   self._llm_alpha[i*5+1]) for i in range(16)]
        self._llm_alpha_gate = [cal.alpha(out_llm["llm_act_gateup"][i], self._llm_alpha[i*5+2]) for i in range(16)]
        self._llm_alpha_up   = [cal.alpha(out_llm["llm_act_gateup"][i], self._llm_alpha[i*5+3]) for i in range(16)]
        self._llm_alpha_down = [cal.alpha(out_llm["llm_act_down"][i],   self._llm_alpha[i*5+4]) for i in range(16)]

        # ── VLSA (4L × 4 distinct act scales; 6 alphas/layer for 6 GEMMs) ──
        self._vlsa_act_qkv_dev = to_devs(out_vlsa["vlsa_act_qkv"])
        self._vlsa_act_o_dev   = to_devs(out_vlsa["vlsa_act_o"])
        self._vlsa_act_fc1_dev = to_devs(out_vlsa["vlsa_act_fc1"])
        self._vlsa_act_fc2_dev = to_devs(out_vlsa["vlsa_act_fc2"])
        # _vlsa_alpha layout: [q, k, v, o, fc1, fc2] per layer
        self._vlsa_alpha_q   = [cal.alpha(out_vlsa["vlsa_act_qkv"][i], self._vlsa_alpha[i*6+0]) for i in range(4)]
        self._vlsa_alpha_k   = [cal.alpha(out_vlsa["vlsa_act_qkv"][i], self._vlsa_alpha[i*6+1]) for i in range(4)]
        self._vlsa_alpha_v   = [cal.alpha(out_vlsa["vlsa_act_qkv"][i], self._vlsa_alpha[i*6+2]) for i in range(4)]
        self._vlsa_alpha_o   = [cal.alpha(out_vlsa["vlsa_act_o"][i],   self._vlsa_alpha[i*6+3]) for i in range(4)]
        self._vlsa_alpha_fc1 = [cal.alpha(out_vlsa["vlsa_act_fc1"][i], self._vlsa_alpha[i*6+4]) for i in range(4)]
        self._vlsa_alpha_fc2 = [cal.alpha(out_vlsa["vlsa_act_fc2"][i], self._vlsa_alpha[i*6+5]) for i in range(4)]

    def _save_calibration_cache(self, out_vit, out_ds, out_llm, out_vlsa) -> None:
        """JSON cache so subsequent set_prompt calls skip the shadow forward.

        Schema is N1.7-specific (Pi05 ``save_calibration`` layout doesn't fit
        — different stages). Stored at ``~/.cache/flash_rt/<ckpt_hash>_n17_Se<n>.json``.
        """
        import json
        from flash_rt.core.quant.calibrator import _checkpoint_hash, CACHE_DIR

        try:
            ckpt_hash = _checkpoint_hash(self.checkpoint_path)
        except Exception:
            return  # best-effort
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"{ckpt_hash}_n17_Se{self.Se}.json"

        payload = {
            "version": 1, "ckpt_hash": ckpt_hash, "Se": self.Se,
            "embodiment_id": self._embodiment_id,
            "vit_act_qkv": out_vit["vit_act_qkv"],
            "vit_act_o":   out_vit["vit_act_o"],
            "vit_act_fc1": out_vit["vit_act_fc1"],
            "vit_act_fc2": out_vit["vit_act_fc2"],
            "deepstack_act_fc1": out_ds["deepstack_act_fc1"],
            "deepstack_act_fc2": out_ds["deepstack_act_fc2"],
            "llm_act_qkv":    out_llm["llm_act_qkv"],
            "llm_act_o":      out_llm["llm_act_o"],
            "llm_act_gateup": out_llm["llm_act_gateup"],
            "llm_act_down":   out_llm["llm_act_down"],
            "vlsa_act_qkv": out_vlsa["vlsa_act_qkv"],
            "vlsa_act_o":   out_vlsa["vlsa_act_o"],
            "vlsa_act_fc1": out_vlsa["vlsa_act_fc1"],
            "vlsa_act_fc2": out_vlsa["vlsa_act_fc2"],
        }
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2)
        self._calibration_cache_path = str(cache_path)

    # ────────────────────────────────────────────────────────────────
    # Multi-frame FP8 calibration (Phase 5b) — N>=2 percentile reduce
    # ────────────────────────────────────────────────────────────────

    def calibrate(
        self,
        aux_list,
        *,
        percentile: float = 99.9,
        verbose: bool = False,
    ) -> None:
        """Refine FP8 act-scale alphas across N calibration samples.

        N1.7 does not currently expose an obs→aux production path
        (Qwen3-VL processor + vision encoder live outside the frontend),
        so the calibration "sample" abstraction is the same ``aux`` dict
        that ``set_prompt`` consumes. Each entry in ``aux_list`` must
        carry the same keys (``pixel_features``, ``llm_input_embeds``,
        ``rope_cos``/``sin``, ``visual_pos_masks``, ``grid_thw``).
        Generate via ``tests/_helpers/groot_n17/capture_aux_multi.py``.

        Behaviour:

          * ``len(aux_list) == 1`` → no-op (the alphas already baked by
            ``set_prompt`` from this aux are bit-equal to a 1-sample
            percentile reduce).
          * ``len(aux_list) >= 2`` → runs the 4-stage shadow forward
            (vit / dsm / llm / vlsa) per aux, percentile-reduces each
            per-quant-point amax along the sample axis, and re-bakes
            the per-stage ``_<stage>_act_*_dev`` device scalars + host
            alphas. Backbone, DeepStack inject buffers, DiT cross-KV
            and the captured DiT graphs are **NOT** touched — those
            stay tied to the ``set_prompt`` aux (the inference-time
            prompt), which is the right contract for production: the
            calibration set is for scale headroom, the deployment aux
            is the actual prompt.

        DiT alphas are absent because the DiT path runs bf16 native
        (see calibration.py module docstring); ``_dit_alpha`` is
        deliberately untouched here.

        Args:
            aux_list: list of aux dicts (or single dict / iterable).
            percentile: percentile along the sample axis. ``100.0`` ==
                traditional max. ``99.9`` (default) clips outliers.
            verbose: log per-stage amax dispersion summaries after
                reduction.
        """
        from flash_rt.core.calibration import (
            accumulate_amax, format_summary, summarize_amax_dispersion,
        )
        from flash_rt.models.groot_n17 import calibration as cal
        import logging
        import numpy as np

        if not hasattr(self, "_backbone_features"):
            raise RuntimeError("call set_prompt before calibrate")

        if isinstance(aux_list, dict):
            aux_list = [aux_list]
        else:
            aux_list = list(aux_list)
        n = len(aux_list)
        if n == 0:
            raise ValueError("aux_list must contain at least 1 sample")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")

        logger = logging.getLogger(__name__)

        if n == 1:
            logger.info(
                "GrootN17 calibrate(N=1): no-op (set_prompt already baked alphas)")
            self._precision_spec = self._snapshot_precision_spec(
                method="single_frame", n=1, percentile=None)
            return

        # Re-load fp16 shadow weights — set_prompt deletes them after
        # baking the single-aux alphas, but multi-frame calibration
        # needs to run the shadow again per-sample. ~1s for the
        # safetensors index walk; released again at the end of this
        # method.
        if not hasattr(self, "_fp16_shadow_weights"):
            self._load_fp16_shadow_weights()

        logger.info(
            "GrootN17 calibrate(N=%d, percentile=%.2f): running shadow per sample...",
            n, percentile)

        per_sample: list[dict] = []
        for idx, aux in enumerate(aux_list):
            row = cal.calibrate_pipeline_amax(self, aux)
            per_sample.append(row)
            if verbose:
                logger.info("  sample %d/%d done", idx + 1, n)

        # Percentile-reduce each per-stage amax list across N samples.
        reduced: dict[str, list[float]] = {}
        per_stage_rows: dict[str, list[np.ndarray]] = {}
        for key in cal.AMAX_KEYS:
            rows = [np.asarray(s[key], dtype=np.float32) for s in per_sample]
            per_stage_rows[key] = rows
            stacked = accumulate_amax(rows, percentile=percentile)
            reduced[key] = stacked.astype(np.float32).tolist()

        if verbose:
            for key in cal.AMAX_KEYS:
                final = np.asarray(reduced[key], dtype=np.float64)
                summary = summarize_amax_dispersion(per_stage_rows[key], final)
                logger.info("  %s: %s", key, format_summary(summary))

        # Re-bake stage outputs from the reduced amax. ``_bake_calibration``
        # expects per-stage dicts with the same keys ``calibrate_*`` produce,
        # so wrap reduced lists to match that schema.
        out_vit = {
            "vit_act_qkv": reduced["vit_act_qkv"], "vit_act_o": reduced["vit_act_o"],
            "vit_act_fc1": reduced["vit_act_fc1"], "vit_act_fc2": reduced["vit_act_fc2"],
        }
        out_ds = {
            "deepstack_act_fc1": reduced["deepstack_act_fc1"],
            "deepstack_act_fc2": reduced["deepstack_act_fc2"],
        }
        out_llm = {
            "llm_act_qkv":    reduced["llm_act_qkv"],
            "llm_act_o":      reduced["llm_act_o"],
            "llm_act_gateup": reduced["llm_act_gateup"],
            "llm_act_down":   reduced["llm_act_down"],
        }
        out_vlsa = {
            "vlsa_act_qkv": reduced["vlsa_act_qkv"],
            "vlsa_act_o":   reduced["vlsa_act_o"],
            "vlsa_act_fc1": reduced["vlsa_act_fc1"],
            "vlsa_act_fc2": reduced["vlsa_act_fc2"],
        }
        self._bake_calibration(out_vit, out_ds, out_llm, out_vlsa)
        self._save_calibration_cache(out_vit, out_ds, out_llm, out_vlsa)

        # Release shadow weight refs again.
        if hasattr(self, "_fp16_shadow_weights"):
            del self._fp16_shadow_weights
            torch.cuda.empty_cache()

        self._precision_spec = self._snapshot_precision_spec(
            method="percentile", n=n, percentile=percentile)

        logger.info(
            "GrootN17 calibrate complete (N=%d, percentile=%.2f)", n, percentile)

    @property
    def precision_spec(self):
        """``ModelPrecisionSpec`` snapshot from the last calibrate call,
        or ``None`` if calibrate has not run."""
        return getattr(self, "_precision_spec", None)

    def _snapshot_precision_spec(
        self, *, method: str, n: int, percentile: float | None,
    ):
        """Build a ``ModelPrecisionSpec`` from the current per-stage host alphas.

        N1.7 has 4 FP8 stages (vit / dsm / llm / vlsa) each with multiple
        per-layer / per-quant-point alphas. We flatten them into the
        ``encoder_layer_specs`` namespace with stage-prefixed keys so a
        single dict can hold all of them without collisions. DiT runs
        bf16 native, so ``decoder_layer_specs`` is left empty.
        """
        import numpy as np
        from flash_rt.core.precision_spec import (
            ModelPrecisionSpec, PrecisionSpec,
        )

        spec = ModelPrecisionSpec(source="calibration")

        def _entry(scale_val: float):
            entry = PrecisionSpec(
                dtype="fp8_e4m3",
                granularity="per_tensor",
                scheme="symmetric",
                scale_source="calibration",
                scale=np.array([float(scale_val)], dtype=np.float32),
                calibration_method=method,
                calibration_samples=n,
                calibration_percentile=percentile,
            )
            entry.validate()
            return entry

        # ── ViT 24L × 4 quant points ────────────────────────────────────
        for i in range(24):
            spec.encoder_layer_specs[f"vit_{i}_qkv"] = _entry(self._vit_alpha_q[i])
            spec.encoder_layer_specs[f"vit_{i}_o"]   = _entry(self._vit_alpha_o[i])
            spec.encoder_layer_specs[f"vit_{i}_fc1"] = _entry(self._vit_alpha_fc1[i])
            spec.encoder_layer_specs[f"vit_{i}_fc2"] = _entry(self._vit_alpha_fc2[i])

        # ── DeepStack 3 mergers × 2 quant points ───────────────────────
        for j in range(3):
            spec.encoder_layer_specs[f"dsm_{j}_fc1"] = _entry(self._dsm_alpha_fc1[j])
            spec.encoder_layer_specs[f"dsm_{j}_fc2"] = _entry(self._dsm_alpha_fc2[j])

        # ── LLM 16L × 5 alphas (qkv share an act scale; gate / up share) ──
        for i in range(16):
            spec.encoder_layer_specs[f"llm_{i}_qkv"]  = _entry(self._llm_alpha_qkv[i])
            spec.encoder_layer_specs[f"llm_{i}_o"]    = _entry(self._llm_alpha_o[i])
            spec.encoder_layer_specs[f"llm_{i}_gate"] = _entry(self._llm_alpha_gate[i])
            spec.encoder_layer_specs[f"llm_{i}_up"]   = _entry(self._llm_alpha_up[i])
            spec.encoder_layer_specs[f"llm_{i}_down"] = _entry(self._llm_alpha_down[i])

        # ── VLSA 4L × 6 alphas ─────────────────────────────────────────
        for i in range(4):
            spec.encoder_layer_specs[f"vlsa_{i}_q"]   = _entry(self._vlsa_alpha_q[i])
            spec.encoder_layer_specs[f"vlsa_{i}_k"]   = _entry(self._vlsa_alpha_k[i])
            spec.encoder_layer_specs[f"vlsa_{i}_v"]   = _entry(self._vlsa_alpha_v[i])
            spec.encoder_layer_specs[f"vlsa_{i}_o"]   = _entry(self._vlsa_alpha_o[i])
            spec.encoder_layer_specs[f"vlsa_{i}_fc1"] = _entry(self._vlsa_alpha_fc1[i])
            spec.encoder_layer_specs[f"vlsa_{i}_fc2"] = _entry(self._vlsa_alpha_fc2[i])

        return spec

    # ────────────────────────────────────────────────────────────────
    # Phase 3c.c — eager infer (4-step flow-matching diffusion loop)
    # ────────────────────────────────────────────────────────────────

    def infer(
        self,
        state_normalized: torch.Tensor,        # (1, 1, 132) fp32 already normalized
        *,
        initial_noise: Optional[torch.Tensor] = None,
        num_inference_timesteps: int = 4,
        action_horizon: int = 40,
        num_timestep_buckets: int = 1000,
        use_dit_graph: bool = True,
    ) -> torch.Tensor:
        """4-step flow-matching diffusion. Returns ``(1, action_horizon, 132)``
        fp32 normalized actions. Caller is responsible for state normalization
        and action denormalization (see ``normalize_state`` / ``denormalize_action``).

        Requires ``set_prompt`` to have been called first (provides backbone,
        baked alphas, M-RoPE / ViT cos/sin tables, DeepStack inject buffers).
        """
        if not hasattr(self, "_backbone_features"):
            raise RuntimeError("call set_prompt before infer")

        device = self.device
        D = 1536
        action_dim = 132
        Sa = action_horizon + 1   # 1 state + 40 action tokens

        # ── Fully-kernelized graph fast-path ──────────────────────────────
        # Per frame, end to end in kernels: refresh cross-KV from the current
        # backbone, then replay the state encode + the per-step DiT chain
        # (action_encode + DiT + output_proj + decode + Euler), all captured
        # as CUDA graphs over persistent kernel buffers. Zero PyTorch compute
        # on the hot path. Captured lazily on first use behind set_prompt's
        # warmup. Falls back to the eager path on a timestep-count mismatch.
        if use_dit_graph:
            if not hasattr(self, "_k_dit_graph"):
                self._capture_kernel_dit_graphs(
                    num_inference_timesteps=num_inference_timesteps,
                    action_horizon=action_horizon)
            if self._k_nsteps == num_inference_timesteps:
                # Write the three per-frame inputs, then one graph replay does
                # cross-KV refresh + state encode + all diffusion steps.
                self._ck_bb_src.copy_(
                    self._backbone_features.reshape(self.Se, 2048).half())
                self._k_state_in.copy_(
                    state_normalized.reshape(1, action_dim).to(device).bfloat16())
                if initial_noise is not None:
                    self._k_actions.copy_(
                        initial_noise.reshape(action_horizon, action_dim).to(device).bfloat16())
                else:
                    self._k_actions.copy_(torch.randn(
                        action_horizon, action_dim, dtype=torch.bfloat16, device=device))
                self._k_dit_graph.replay()
                return self._k_actions.float().view(1, action_horizon, action_dim)

        # Eager fallback: torch cross-KV from backbone (graph path refreshes
        # it via the kernel cross-KV graph instead).
        if not hasattr(self, "_dit_cross_K"):
            self._precompute_dit_cross_kv()

        # ── state encode (one-shot; doesn't change across steps) ──────────
        state_features = self._run_state_encode(state_normalized.to(device).bfloat16())  # (1, 1, 1536) bf16

        # Per-step buffers for action_encode + dit + decode (allocate once)
        if not hasattr(self, "_infer_bufs"):
            self._allocate_infer_buffers(action_horizon)
        bufs = self._infer_bufs

        # ── initial noise (bf16, matches ckpt native dtype) ─────────────
        if initial_noise is not None:
            actions = initial_noise.to(device).bfloat16().contiguous().clone()
        else:
            actions = torch.randn(
                1, action_horizon, action_dim, dtype=torch.bfloat16, device=device)

        dt = 1.0 / num_inference_timesteps

        # Pre-build action position embedding (constant across steps).
        # action_head.position_embedding is loaded as _ah_pos_embed_w (1024, 1536) fp16;
        # cast to bf16 to match the DiT main path.
        pos_embed = self._ah_pos_embed_w[:action_horizon].bfloat16()  # (40, 1536) bf16

        # Per-step shift/scale tensors must live until ALL DiT kernels
        # they're referenced by have completed. Stash the per-step lists
        # here so PyTorch GC doesn't free them under our feet between
        # iterations (data_ptr() refs in the weights dict don't keep the
        # tensors alive).
        self._infer_shift_lists = []
        self._infer_scale_lists = []
        self._infer_temb_list = []

        # Graph fast-path: requires the per-step modulators / DiT scratch
        # buffers / attn slots / GemmRunner to all be pre-allocated and
        # the graphs already captured. We auto-trigger capture on the
        # first call (lazy) so the cost is hidden behind set_prompt's
        # warmup cycle when the caller leaves use_dit_graph=True.
        graphs = None
        if use_dit_graph:
            if not hasattr(self, "_dit_graphs"):
                self._capture_dit_graphs(
                    num_inference_timesteps=num_inference_timesteps,
                    action_horizon=action_horizon)
            graphs = self._dit_graphs
            # Sanity: captured for the same number of timesteps the caller
            # is requesting now.
            if len(graphs) != num_inference_timesteps:
                graphs = None  # fall back to eager — shape mismatch

        for step in range(num_inference_timesteps):
            t_cont = step / num_inference_timesteps
            t_disc = int(t_cont * num_timestep_buckets)

            if graphs is not None:
                # Pre-computed at set_prompt; reuse instead of recomputing.
                temb = self._step_temb[step]
                shift_list = self._step_shifts[step]
                scale_list = self._step_scales[step]
            else:
                # ── timestep emb (1, 1536) ───────────────────────────────
                temb = self._compute_timestep_emb(t_disc)
                # ── per-layer DiT AdaLN modulators (32 × shift, scale) ──
                shift_list, scale_list = self._compute_dit_adaln_modulators(temb)
                # Stash on self so the underlying tensors live past the
                # loop iter scope — pipeline_thor.dit_forward reads
                # ``weights["scale_msa"][li]`` as raw data_ptr ints, which
                # do not keep the python tensor objects alive. Without
                # this stash, GC of the previous step's local lists can
                # free the underlying device memory before the next
                # dit_forward kernel actually runs (NaN propagation seen
                # empirically).
                self._infer_shift_lists.append(shift_list)
                self._infer_scale_lists.append(scale_list)
                self._infer_temb_list.append(temb)

            # ── action features (1, 40, 1536) ────────────────────────────
            action_features = self._run_action_encode(
                actions, t_disc, action_horizon)
            action_features = action_features + pos_embed.unsqueeze(0)

            # ── DiT input: cat(state, action) → (1, 41, 1536) bf16 ──────
            sa_embs = torch.cat([state_features, action_features], dim=1)
            bufs["dit_h"][:Sa].copy_(sa_embs.squeeze(0).contiguous())

            # ── DiT forward (32 layers) ──────────────────────────────────
            if graphs is not None:
                graphs[step].replay()
            else:
                self._run_dit(bufs, shift_list, scale_list, Sa)

            # ── Output projection: AdaLN(SiLU(temb)) + linear → (1, 41, 1024) ──
            h_out = self._run_dit_output_proj(bufs["dit_h"][:Sa].unsqueeze(0), temb)

            # ── action_decode on last 40 tokens → velocity (1, 40, 132) ──
            velocity = self._run_action_decode(h_out[:, -action_horizon:])

            # ── Euler step ──────────────────────────────────────────────
            actions = actions + (dt * velocity).to(actions.dtype)

        return actions.float()

    # ────────────────────────────────────────────────────────────────
    # Internal helpers (eager; will be CUDA-graph wrapped in 3c.d)
    # ────────────────────────────────────────────────────────────────

    def _precompute_dit_cross_kv(self) -> None:
        """For each cross-attn DiT layer (even idx 0,2,...,30), compute K and V
        from backbone_features filtered to text or image positions per the
        ``attend_text_every_n_blocks=2`` rule:
          idx ∈ {0,4,8,...} → text positions (~visual_pos_masks)
          idx ∈ {2,6,10,...} → image positions (visual_pos_masks)

        Storage: ``self._dit_cross_K[16]``, ``self._dit_cross_V[16]`` each
        a (kv_seq_li, D=1536) fp16 tensor. Layer i (cross) maps to slot j=i//2.
        """
        backbone = self._backbone_features.squeeze(0)   # (S, 2048)
        mask = self._visual_pos_masks
        text_kv_src = backbone[~mask]    # (text_count=21, 2048)
        image_kv_src = backbone[mask]    # (256, 2048)

        # Cross layers are at full-layer indices [0, 2, 4, ..., 30] — 16 of
        # them. Backend's dit_cross site uses cross-only indexing (j ∈ 0..15),
        # so this list is length-16, indexed by j = li // 2.
        K_list, V_list = [], []
        for j in range(16):
            li = 2 * j
            target_text = (li % 4 == 0)
            kv_src = text_kv_src if target_text else image_kv_src
            # DiT weights are bf16 (pre-dequantized in _load_weights); no
            # additional scale multiply.
            k_w = self._dit_k_w[li].float()
            v_w = self._dit_v_w[li].float()
            k_b = self._dit_k_b[li].float()
            v_b = self._dit_v_b[li].float()
            K = (kv_src.float() @ k_w + k_b).bfloat16().contiguous()
            V = (kv_src.float() @ v_w + v_b).bfloat16().contiguous()
            K_list.append(K)
            V_list.append(V)
        self._dit_cross_K = K_list
        self._dit_cross_V = V_list

    def _setup_cross_kv_kernel(self) -> None:
        """Kernelized DiT cross-KV: gather the backbone's text/image rows and
        project them to per-layer K/V entirely on-device, writing into the
        SAME persistent buffers the DiT attention backend reads. The backbone
        changes every frame (new image), so this must run per-frame — running
        it as a kernel graph removes the only remaining PyTorch on the hot
        path (the previous torch implementation cost ~5.4 ms/frame).

        Replaces the torch ``_precompute_dit_cross_kv`` buffers with persistent
        ones (fixed data_ptrs) so both ``_build_dit_attn`` and the captured
        cross-KV graph reference them.
        """
        if hasattr(self, "_ck_text_idx"):
            return
        dev = self.device
        S = self.Se
        mask = self._visual_pos_masks
        self._ck_text_idx = torch.where(~mask)[0].to(torch.int64).contiguous()
        self._ck_image_idx = torch.where(mask)[0].to(torch.int64).contiguous()
        nt = int(self._ck_text_idx.numel())
        ni = int(self._ck_image_idx.numel())
        self._ck_nt, self._ck_ni = nt, ni
        bf = torch.bfloat16
        # per-frame backbone input (fp16, copied in before replay) + bf16 cast
        self._ck_bb_src = torch.empty(S, 2048, dtype=torch.float16, device=dev)
        self._ck_bb = torch.empty(S, 2048, dtype=bf, device=dev)
        self._ck_text_src = torch.empty(nt, 2048, dtype=bf, device=dev)
        self._ck_image_src = torch.empty(ni, 2048, dtype=bf, device=dev)
        # persistent per-layer K/V buffers (overwrite torch list)
        self._dit_cross_K = [
            torch.empty((nt if (2 * j) % 4 == 0 else ni), 1536, dtype=bf, device=dev)
            for j in range(16)]
        self._dit_cross_V = [
            torch.empty((nt if (2 * j) % 4 == 0 else ni), 1536, dtype=bf, device=dev)
            for j in range(16)]
        if not hasattr(self, "_fvk"):
            import flash_rt.flash_rt_kernels as _fvk
            self._fvk = _fvk
        if not hasattr(self, "_mlp_gemm"):
            self._mlp_gemm = self._fvk.GemmRunner()
        # seed the buffers from the current backbone (eager) so a non-graph
        # consumer sees valid K/V immediately.
        self._ck_bb_src.copy_(self._backbone_features.reshape(S, 2048).half())
        self._cross_kv_fwd(0)

    def _cross_kv_fwd(self, s: int) -> None:
        """Pure-kernel cross-KV forward over the persistent buffers (graph-safe).
        Reads ``_ck_bb_src`` (current backbone), writes ``_dit_cross_K/V``."""
        K = self._fvk
        mg = self._mlp_gemm
        S = self.Se
        nt, ni = self._ck_nt, self._ck_ni
        K.cast_fp16_to_bf16(self._ck_bb_src.data_ptr(), self._ck_bb.data_ptr(), S * 2048, s)
        K.qwen36_embedding_lookup_bf16(
            self._ck_text_idx.data_ptr(), self._ck_bb.data_ptr(),
            self._ck_text_src.data_ptr(), nt, 2048, s)
        K.qwen36_embedding_lookup_bf16(
            self._ck_image_idx.data_ptr(), self._ck_bb.data_ptr(),
            self._ck_image_src.data_ptr(), ni, 2048, s)
        for j in range(16):
            li = 2 * j
            text = (li % 4 == 0)
            src = self._ck_text_src if text else self._ck_image_src
            N = nt if text else ni
            mg.bf16_nn(src.data_ptr(), self._dit_k_w[li].data_ptr(),
                       self._dit_cross_K[j].data_ptr(), N, 1536, 2048, s)
            K.add_bias_bf16(self._dit_cross_K[j].data_ptr(),
                            self._dit_k_b[li].data_ptr(), N, 1536, s)
            mg.bf16_nn(src.data_ptr(), self._dit_v_w[li].data_ptr(),
                       self._dit_cross_V[j].data_ptr(), N, 1536, 2048, s)
            K.add_bias_bf16(self._dit_cross_V[j].data_ptr(),
                            self._dit_v_b[li].data_ptr(), N, 1536, s)

    def _compute_timestep_emb(self, t_disc: int) -> torch.Tensor:
        """diffusers Timesteps + TimestepEmbedding → (1, 1536) fp16."""
        import math
        half_dim = 128
        # downscale_freq_shift=1, flip_sin_to_cos=True
        exponent = -math.log(10000) * torch.arange(
            0, half_dim, dtype=torch.float32, device=self.device) / (half_dim - 1)
        freqs = torch.exp(exponent)                              # (128,)
        emb = torch.tensor([t_disc], dtype=torch.float32, device=self.device)[:, None] * freqs[None, :]
        # diffusers ``get_timestep_embedding`` first builds ``cat([sin, cos])``,
        # then under flip_sin_to_cos=True (used by N1.7 TimestepEncoder)
        # swaps the halves so the final layout is ``cat([cos, sin])``.
        # (Previous code dropped the flip — see embeddings.py L72-74 — and
        # ended up with the unswapped sin-first layout, which gave a 2x
        # magnitude mismatch vs HF's TimestepEncoder output even though the
        # DiT INPUT cosine still showed 1.0 — INPUT does not depend on
        # temb; it only flows into AdaLN modulators downstream.)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)   # (1, 256)
        # TimestepEmbedding: linear_1 (256→1536) + SiLU + linear_2 (1536→1536)
        # Loaded weights are post-T()-Quant() FP8; dequant via _dit_misc_alpha[0/1].
        # Scales layout: ts_lin1 idx 0, ts_lin2 idx 1, proj_out_1 idx 2, proj_out_2 idx 3
        ts_lin1_w = (self._ts_lin1_w.float() * self._dit_misc_alpha[0])  # (256, 1536)
        ts_lin1_b = self._ts_lin1_b.float()
        ts_lin2_w = (self._ts_lin2_w.float() * self._dit_misc_alpha[1])  # (1536, 1536)
        ts_lin2_b = self._ts_lin2_b.float()
        h = emb @ ts_lin1_w + ts_lin1_b
        h = torch.nn.functional.silu(h)
        h = h @ ts_lin2_w + ts_lin2_b
        return h.bfloat16().contiguous()       # (1, 1536) bf16

    def _compute_dit_adaln_modulators(self, temb: torch.Tensor):
        """Per layer: (silu(temb)) @ ada_w[i] + ada_b[i] → chunk(2) → (scale, shift).

        Returns (list[32] of bf16 (D=1536,) shift tensors, list[32] of (D,) scale).

        IMPORTANT: HF ``class AdaLayerNorm`` (gr00t/model/modules/dit.py:95)
        unpacks ``scale, shift = temb.chunk(2)`` — scale comes FIRST, shift
        SECOND. This differs from the final ``proj_out_1`` AdaLN (dit.py:331)
        where the unpack order is ``shift, scale``. Getting this wrong swaps
        the affine and turns the per-layer modulation into garbage (cos
        single-step ≈ 0.88 instead of 0.99+).
        """
        x = torch.nn.functional.silu(temb.float())   # (1, 1536)
        shifts, scales = [], []
        for i in range(32):
            # DiT weights are bf16 (pre-dequant); no scale multiply.
            ada_w = self._dit_ada_w[i].float()                               # (1536, 3072)
            ada_b = self._dit_ada_b[i].float()
            mod = x @ ada_w + ada_b              # (1, 3072)
            scale, shift = mod.chunk(2, dim=-1)  # each (1, 1536) — HF order
            shifts.append(shift.squeeze(0).bfloat16().contiguous())
            scales.append(scale.squeeze(0).bfloat16().contiguous())
        return shifts, scales

    def _run_state_encode(self, state_flat: torch.Tensor) -> torch.Tensor:
        """state (1, 1, 132) → state_features (1, 1, 1536) bf16. Pure PyTorch
        for 3c.c eager mode (small MLP; perf-irrelevant). bf16 matches the
        ckpt's native dtype and the DiT main path."""
        x = state_flat.view(1, 132).float()
        h = x @ self._st_enc_l1_W.float() + self._st_enc_l1_b.float()
        h = torch.nn.functional.relu(h)
        out = h @ self._st_enc_l2_W.float() + self._st_enc_l2_b.float()
        return out.bfloat16().view(1, 1, 1536)

    def _run_action_encode(self, actions: torch.Tensor, t_disc: int,
                            action_horizon: int) -> torch.Tensor:
        """noisy (1, 40, 132) + scalar t_disc → features (1, 40, 1536).

        Per HF ``MultiEmbodimentActionEncoder`` (embodiment_conditioned_mlp.py:60):
          a_emb = W1(actions)                          # (T, 1536), NO activation
          tau_emb = SinusoidalPositionalEncoding(t)    # (T, 1536) — its OWN
                                                       #   sin/cos formula, different
                                                       #   from DiT's TimestepEncoder
          x = cat(a_emb, tau_emb)                      # (T, 3072)
          x = swish(W2(x))                             # SiLU, not ReLU
          out = W3(x)                                  # no activation
        """
        import math
        device = self.device
        H = 1536
        half_dim = H // 2     # 768

        # SinusoidalPositionalEncoding(timesteps=full(action_horizon, t_disc)).
        # Note: this is NOT the diffusers TimestepEncoder — it has neither
        # learned linear layers nor a downscale_freq_shift, and uses
        # exponent = -arange * log(10000)/half_dim.
        exponent = -torch.arange(
            half_dim, dtype=torch.float32, device=device
        ) * (math.log(10000.0) / half_dim)
        timesteps = torch.full(
            (action_horizon,), float(t_disc),
            dtype=torch.float32, device=device,
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()         # (T, half_dim)
        tau_emb = torch.cat(
            [torch.sin(freqs), torch.cos(freqs)], dim=-1)       # (T, H)

        x = actions.view(action_horizon, 132).float()
        a_emb = x @ self._ac_enc_W1_W.float() + self._ac_enc_W1_b.float()   # (T, H)
        cat = torch.cat([a_emb, tau_emb], dim=-1)                           # (T, 2H)
        h = cat @ self._ac_enc_W2_W.float() + self._ac_enc_W2_b.float()
        h = torch.nn.functional.silu(h)                                     # swish
        out = h @ self._ac_enc_W3_W.float() + self._ac_enc_W3_b.float()
        return out.bfloat16().view(1, action_horizon, H)

    def _run_action_decode(self, dit_out: torch.Tensor) -> torch.Tensor:
        """dit_output (1, 40, 1024) bf16 → velocity (1, 40, 132) bf16."""
        x = dit_out.view(-1, 1024).float()
        h = x @ self._ac_dec_l1_W.float() + self._ac_dec_l1_b.float()
        h = torch.nn.functional.relu(h)
        out = h @ self._ac_dec_l2_W.float() + self._ac_dec_l2_b.float()
        return out.bfloat16().view(1, dit_out.shape[1], 132)

    def _run_dit_output_proj(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """proj_out_1(SiLU(temb)) → (shift, scale) → AdaLN(h) → proj_out_2 → (1, 41, 1024)."""
        D = 1536
        x = torch.nn.functional.silu(temb.float())   # (1, 1536)
        po1_w = self._proj_out_1_w.float() * self._dit_misc_alpha[2]   # (1536, 3072)
        po1_b = self._proj_out_1_b.float()
        mod = x @ po1_w + po1_b                       # (1, 3072)
        shift, scale = mod.chunk(2, dim=-1)
        # LayerNorm(no_affine) on h
        h_norm = torch.nn.functional.layer_norm(
            h.float(), (D,), eps=1e-5)               # (1, 41, D)
        h_mod = h_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        po2_w = self._proj_out_2_w.float() * self._dit_misc_alpha[3]   # (1536, 1024)
        po2_b = self._proj_out_2_b.float()
        return (h_mod @ po2_w + po2_b).bfloat16().contiguous()

    def _run_dit(self, bufs: dict, shift_list, scale_list, Sa: int) -> None:
        """Eager DiT 32-layer forward via pipeline_thor.dit_forward.

        Wires per-layer AdaLN shift/scale device ptrs and per-layer cross-KV
        slots into the production attn backend (lazily constructed)."""
        from flash_rt.models.groot_n17 import pipeline_thor

        if not hasattr(self, "_dit_attn"):
            self._build_dit_attn(Sa)

        # Per-layer shift/scale ptr lists (zip into weights dict expected by
        # dit_forward).
        weights = {
            "scale_msa": [t.data_ptr() for t in scale_list],
            "shift_msa": [t.data_ptr() for t in shift_list],
            "q_w": [w.data_ptr() for w in self._dit_q_w],
            "q_b": [b.data_ptr() for b in self._dit_q_b],
            "k_w": [w.data_ptr() for w in self._dit_k_w],
            "k_b": [b.data_ptr() for b in self._dit_k_b],
            "v_w": [w.data_ptr() for w in self._dit_v_w],
            "v_b": [b.data_ptr() for b in self._dit_v_b],
            "o_w": [w.data_ptr() for w in self._dit_o_w],
            "o_b": [b.data_ptr() for b in self._dit_o_b],
            "ff_proj_w": [w.data_ptr() for w in self._dit_ff_proj_w],
            "ff_proj_b": [b.data_ptr() for b in self._dit_ff_proj_b],
            "ff_down_w": [w.data_ptr() for w in self._dit_ff_down_w],
            "ff_down_b": [b.data_ptr() for b in self._dit_ff_down_b],
        }
        # j=0 is layer 0 (text-target), j=1 is layer 2 (image-target).
        Skv_text = int(self._dit_cross_K[0].shape[0])     # text tokens
        Skv_image = int(self._dit_cross_K[1].shape[0])    # image tokens
        dims = {
            "Sa": Sa, "D": 1536, "FF": 6144,
            "Skv_text": Skv_text, "Skv_image": Skv_image,
        }
        # ATTN backend: update K/V layer ptrs to current cross-KV (no-op if
        # already set; cross-KV is constant across diffusion steps so we wire
        # pointers once).
        bufs_ptrs = {
            "h": bufs["dit_h"].data_ptr(),
            "xn": bufs["dit_xn"].data_ptr(),
            "o_proj_out": bufs["dit_o_proj_out"].data_ptr(),
            "ff_proj_out": bufs["dit_ff_proj_out"].data_ptr(),
        }

        # cuBLAS GemmRunner — fresh per call (cheap construct).
        if not hasattr(self, "_gemm"):
            import flash_rt.flash_rt_kernels as _fvk
            self._fvk = _fvk
            self._gemm = _fvk.GemmRunner()

        pipeline_thor.dit_forward(
            gemm=self._gemm, fvk=self._fvk,
            bufs=bufs_ptrs, weights=weights, dims=dims,
            attn=self._dit_attn,
        )

    def _allocate_infer_buffers(self, action_horizon: int) -> None:
        """Pre-allocate DiT scratch buffers (eager mode, reused per step).

        N1.7 ckpt is natively bfloat16; the DiT main path runs entirely
        in bf16 so all hidden-state buffers are allocated as bf16. The
        boundary back into PyTorch (output_proj, action_decode) handles
        the bf16→fp32 cast on its own.
        """
        Sa = 1 + action_horizon
        D, FF = 1536, 6144
        device = self.device
        self._infer_bufs = {
            "dit_h":         torch.empty((Sa, D), dtype=torch.bfloat16, device=device),
            "dit_xn":        torch.empty((Sa, D), dtype=torch.bfloat16, device=device),
            "dit_o_proj_out":torch.empty((Sa, D), dtype=torch.bfloat16, device=device),
            "dit_ff_proj_out": torch.empty((Sa, FF), dtype=torch.bfloat16, device=device),
        }

    def _build_dit_attn(self, Sa: int) -> None:
        """Construct ThorGrootN17AttnBackend with DiT slots wired to current
        cross-KV. self_attn slots get fresh per-step buffers."""
        from flash_rt.hardware.thor.attn_backend_groot_n17 import (
            ThorGrootN17AttnBackend, make_groot_n17_attention_spec,
        )
        import flash_rt.flash_rt_kernels as fvk

        D = 1536; NH = 32; HD = 48
        Skv_text = int(self._dit_cross_K[0].shape[0])
        Skv_image = int(self._dit_cross_K[1].shape[0])

        # Per-DiT-layer self-attn buffers (Q/K/V/O/logits) — all bf16
        # (matches dit_h main buffer + N1.7 ckpt native dtype).
        # IMPORTANT: ``attention_mha_*`` rounds S_kv up to a multiple of 8
        # for cuBLAS GEMM stride (S_kv_pad). The logits buffer must hold
        # the padded width, otherwise the GEMM write stomps adjacent
        # allocations (silent in production where the next layer overwrites
        # everything, fatal under per-layer bisect that re-injects state).
        def _pad8(n: int) -> int:
            return ((int(n) + 7) // 8) * 8

        Sa_pad = _pad8(Sa)
        Q_self = torch.empty((Sa, NH * HD), dtype=torch.bfloat16, device=self.device)
        K_self = torch.empty((Sa, NH * HD), dtype=torch.bfloat16, device=self.device)
        V_self = torch.empty((Sa, NH * HD), dtype=torch.bfloat16, device=self.device)
        O_self = torch.empty((Sa, NH * HD), dtype=torch.bfloat16, device=self.device)
        log_self = torch.empty((NH, Sa, Sa_pad), dtype=torch.bfloat16, device=self.device)
        self._dit_self_bufs = (Q_self, K_self, V_self, O_self, log_self)

        # Cross-attn buffers (kv length pad as well)
        kv_max = max(Skv_text, Skv_image)
        Q_cross = torch.empty((Sa, NH * HD), dtype=torch.bfloat16, device=self.device)
        O_cross = torch.empty((Sa, NH * HD), dtype=torch.bfloat16, device=self.device)
        log_cross = torch.empty(
            (NH, Sa, _pad8(kv_max)),
            dtype=torch.bfloat16, device=self.device)
        self._dit_cross_bufs = (Q_cross, O_cross, log_cross)

        spec = make_groot_n17_attention_spec(
            num_views=4, llm_seq_max=self.Se, vl_self_attn_seq_max=self.Se,
            sa=Sa, s_kv_text=Skv_text, s_kv_image=Skv_image,
        )
        ctx = fvk.FvkContext()
        self._dit_ctx = ctx

        self._dit_attn = ThorGrootN17AttnBackend(
            spec,
            vit_slots={"qkv": 1, "O": 2, "D": 16 * 64},
            llm_slots={"ctx": ctx, "Q": 1, "K": 2, "V": 3, "O": 4,
                       "logits": 5, "scale": 1.0 / (128 ** 0.5)},
            vl_self_attn_slots={"ctx": ctx, "Q": 1, "K": 2, "V": 3, "O": 4,
                                "logits": 5, "scale": 1.0 / (64 ** 0.5)},
            dit_self_slots={
                "ctx": ctx,
                "Q": Q_self.data_ptr(), "K": K_self.data_ptr(),
                "V": V_self.data_ptr(), "O": O_self.data_ptr(),
                "logits": log_self.data_ptr(),
                "scale": 1.0 / (HD ** 0.5),
            },
            dit_cross_slots={
                "ctx": ctx,
                "Q": Q_cross.data_ptr(),
                "K_layers": [t.data_ptr() for t in self._dit_cross_K],
                "V_layers": [t.data_ptr() for t in self._dit_cross_V],
                "O": O_cross.data_ptr(),
                "logits": log_cross.data_ptr(),
                "scale": 1.0 / (HD ** 0.5),
            },
        )

    # ────────────────────────────────────────────────────────────────
    # Normalize / Denormalize helpers (statistics.json, q01/q99)
    # ────────────────────────────────────────────────────────────────

    def _read_statistics(self) -> dict:
        if hasattr(self, "_statistics"):
            return self._statistics
        import json
        stats_path = os.path.join(self.checkpoint_path, "statistics.json")
        with open(stats_path) as f:
            stats = json.load(f)
        self._statistics = stats[self.embodiment_tag]
        return self._statistics

    def normalize_state(self, state_dict: dict) -> torch.Tensor:
        """Build (1, 1, 132) state tensor per ``use_percentiles=true``.

        ``state_dict`` keys per the embodiment's modality config; values are
        per-modality fp32 arrays of shape ``(1, 1, dim_modality)``. Final 132-d
        tensor concatenates modalities in the order they appear in the
        statistics file, then pads to 132.
        """
        stats = self._read_statistics()["state"]
        flat: list[float] = []
        eps = 1e-8
        for mod_key, mod_stats in stats.items():
            v = state_dict[f"state.{mod_key}"]
            v = torch.as_tensor(v).float().reshape(-1)
            q01 = torch.tensor(mod_stats["q01"]).float()
            q99 = torch.tensor(mod_stats["q99"]).float()
            normed = 2 * (v - q01) / (q99 - q01 + eps) - 1
            flat.extend(normed.tolist())
        # Pad to 132
        while len(flat) < 132:
            flat.append(0.0)
        return torch.tensor(flat, dtype=torch.float32).view(1, 1, 132)

    def denormalize_action(self, action_normed: torch.Tensor,
                            state_dict: dict | None = None) -> dict:
        """Inverse of the action processing — returns dict[mod_key] of shape
        ``(1, action_horizon, dim_modality)`` fp32, matching the fixture
        ``actions.{eef_9d, gripper_position, joint_position}`` layout.

        For RELATIVE-action embodiments (e.g. ``oxe_droid_relative_eef_*``)
        the per-modality unnormalize step (q01/q99) is not enough — HF's
        ``Gr00tN1d7Processor.decode_action`` then adds the reference state
        back via SE3 pose composition (eef) or vector add (joint). Re-using
        the HF processor here is far simpler than re-implementing SE3
        chunking; it also automatically picks up future embodiment tags
        without code changes.

        Args:
            action_normed: ``(1, action_horizon, 132)`` torch tensor in the
                model's normalized output space (any float dtype).
            state_dict: ``{"state.<mod>": (1, 1, dim) ndarray}`` matching the
                ``normalize_state`` input. **Required** when the embodiment
                contains RELATIVE actions; optional otherwise (HF processor
                will raise if it actually needs it).
        """
        proc = self._hf_processor()
        emb_tag = self._hf_embodiment_tag()
        # decode_action expects np.ndarray (B, T, D) for action and
        # ``{key_without_state_prefix: (B, T_state, D)}`` for state.
        action_np = action_normed.detach().float().cpu().numpy()
        if action_np.ndim == 2:
            action_np = action_np[None]                # (1, T, D)
        batched_states: dict = {}
        if state_dict is not None:
            import numpy as _np
            for k, v in state_dict.items():
                key = k[len("state."):] if k.startswith("state.") else k
                arr = _np.asarray(v).astype(_np.float32)
                if arr.ndim == 2:                      # (T, D) → (1, T, D)
                    arr = arr[None]
                batched_states[key] = arr
        decoded = proc.decode_action(action_np, emb_tag, batched_states)
        # Cast everything to torch tensors (float32) to mirror the previous
        # API contract of this method.
        out = {k: torch.as_tensor(v).float() for k, v in decoded.items()}
        return out

    def _hf_processor(self):
        """Lazily build the HF Gr00tN1d7Processor. Used by
        ``denormalize_action`` for the relative→absolute step."""
        if not hasattr(self, "_hf_proc_cached"):
            # Side-effect import: registers Gr00tN1d7Config / processor under
            # AutoConfig / AutoProcessor when running outside the gr00t pkg
            # entry-points.
            import gr00t.model.gr00t_n1d7.gr00t_n1d7  # noqa: F401
            from transformers import AutoProcessor
            self._hf_proc_cached = AutoProcessor.from_pretrained(
                self.checkpoint_path, trust_remote_code=True)
        return self._hf_proc_cached

    def _hf_embodiment_tag(self):
        """Resolve our embodiment-tag string into the HF EmbodimentTag enum."""
        if not hasattr(self, "_hf_emb_tag_cached"):
            from gr00t.data.embodiment_tags import EmbodimentTag
            self._hf_emb_tag_cached = EmbodimentTag.resolve(
                self.embodiment_tag.upper())
        return self._hf_emb_tag_cached

    def predict(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def get_latency_stats(self):
        raise NotImplementedError("Phase 3c.d: latency stats")
