"""FlashRT -- GROOT N1.7 full-FP16 torch frontend for RTX.

Full-FP16 baseline variant of ``groot_n17_rtx.py``. The N1.7 RTX inference
path runs the action head (state/action encoders, the 32-layer DiT, output
proj, decoder) through FlashRT kernels; the production path runs those in
bf16. This variant runs them in FP16 instead, so the action head is a
uniform full-FP16 path — an A/B precision reference against the bf16 path.
The backbone (ViT/LLM/VLSA) is produced once at ``set_prompt`` by the shared
fp32 calibration shadow and is unchanged.

Additive: subclasses :class:`GrootN17TorchFrontendRtx` and overrides only the
dtype-bearing action-head methods. The bf16 path is left untouched.
"""

from __future__ import annotations

import math
import os
import glob

import torch

from flash_rt.frontends.torch.groot_n17_rtx import GrootN17TorchFrontendRtx

_FP16 = torch.float16


class GrootN17TorchFrontendRtxFP16(GrootN17TorchFrontendRtx):
    """N1.7 RTX frontend with a full-FP16 action-head path."""

    # ── Weight loading: dequant DiT FP8 weights to FP16 (bf16 in base) ──
    def _load_weights(self) -> None:
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

        # DiT weights are loaded FP8 per spec; dequant to FP16 (the bf16 path
        # dequants to bf16). w_fp16 = (w_fp8.float() * weight_scale).half().
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
                w_list[i] = (w_list[i].float()
                             * float(self._dit_alpha[scale_idx])).half().contiguous()
                b_list[i] = b_list[i].half().contiguous()

        slot = self._embodiment_id
        for name in (
            "_st_enc_l1_W", "_st_enc_l1_b", "_st_enc_l2_W", "_st_enc_l2_b",
            "_ac_enc_W1_W", "_ac_enc_W1_b", "_ac_enc_W2_W", "_ac_enc_W2_b",
            "_ac_enc_W3_W", "_ac_enc_W3_b",
            "_ac_dec_l1_W", "_ac_dec_l1_b", "_ac_dec_l2_W", "_ac_dec_l2_b",
        ):
            full = getattr(self, name)
            setattr(self, name, full[slot].contiguous())
            del full

        self._load_fp16_shadow_weights()

    # ── Cross-attn K/V precompute (FP16 output) ──
    def _precompute_dit_cross_kv(self) -> None:
        backbone = self._backbone_features.squeeze(0)
        mask = self._visual_pos_masks
        text_kv_src = backbone[~mask]
        image_kv_src = backbone[mask]
        K_list, V_list = [], []
        for j in range(16):
            li = 2 * j
            target_text = (li % 4 == 0)
            kv_src = text_kv_src if target_text else image_kv_src
            k_w = self._dit_k_w[li].float()
            v_w = self._dit_v_w[li].float()
            k_b = self._dit_k_b[li].float()
            v_b = self._dit_v_b[li].float()
            K_list.append((kv_src.float() @ k_w + k_b).half().contiguous())
            V_list.append((kv_src.float() @ v_w + v_b).half().contiguous())
        self._dit_cross_K = K_list
        self._dit_cross_V = V_list

    # ── Timestep embedding (FP16) ──
    def _compute_timestep_emb(self, t_disc: int) -> torch.Tensor:
        half_dim = 128
        exponent = -math.log(10000) * torch.arange(
            0, half_dim, dtype=torch.float32, device=self.device) / (half_dim - 1)
        freqs = torch.exp(exponent)
        emb = torch.tensor(
            [t_disc], dtype=torch.float32, device=self.device)[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        ts_lin1_w = (self._ts_lin1_w.float() * self._dit_misc_alpha[0])
        ts_lin1_b = self._ts_lin1_b.float()
        ts_lin2_w = (self._ts_lin2_w.float() * self._dit_misc_alpha[1])
        ts_lin2_b = self._ts_lin2_b.float()
        h = emb @ ts_lin1_w + ts_lin1_b
        h = torch.nn.functional.silu(h)
        h = h @ ts_lin2_w + ts_lin2_b
        return h.half().contiguous()

    # ── Per-layer AdaLN modulators (FP16) ──
    def _compute_dit_adaln_modulators(self, temb: torch.Tensor):
        x = torch.nn.functional.silu(temb.float())
        shifts, scales = [], []
        for i in range(32):
            ada_w = self._dit_ada_w[i].float()
            ada_b = self._dit_ada_b[i].float()
            mod = x @ ada_w + ada_b
            scale, shift = mod.chunk(2, dim=-1)   # HF order: scale, shift
            shifts.append(shift.squeeze(0).half().contiguous())
            scales.append(scale.squeeze(0).half().contiguous())
        return shifts, scales

    # ── Embodiment encoders / decoder (FP16 output) ──
    def _run_state_encode(self, state_flat: torch.Tensor) -> torch.Tensor:
        x = state_flat.view(1, 132).float()
        h = x @ self._st_enc_l1_W.float() + self._st_enc_l1_b.float()
        h = torch.nn.functional.relu(h)
        out = h @ self._st_enc_l2_W.float() + self._st_enc_l2_b.float()
        return out.half().view(1, 1, 1536)

    def _run_action_encode(self, actions: torch.Tensor, t_disc: int,
                           action_horizon: int) -> torch.Tensor:
        device = self.device
        H = 1536
        half_dim = H // 2
        exponent = -torch.arange(
            half_dim, dtype=torch.float32, device=device
        ) * (math.log(10000.0) / half_dim)
        timesteps = torch.full(
            (action_horizon,), float(t_disc), dtype=torch.float32, device=device)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        tau_emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)

        x = actions.view(action_horizon, 132).float()
        a_emb = x @ self._ac_enc_W1_W.float() + self._ac_enc_W1_b.float()
        cat = torch.cat([a_emb, tau_emb], dim=-1)
        h = cat @ self._ac_enc_W2_W.float() + self._ac_enc_W2_b.float()
        h = torch.nn.functional.silu(h)
        out = h @ self._ac_enc_W3_W.float() + self._ac_enc_W3_b.float()
        return out.half().view(1, action_horizon, H)

    def _run_action_decode(self, dit_out: torch.Tensor) -> torch.Tensor:
        x = dit_out.view(-1, 1024).float()
        h = x @ self._ac_dec_l1_W.float() + self._ac_dec_l1_b.float()
        h = torch.nn.functional.relu(h)
        out = h @ self._ac_dec_l2_W.float() + self._ac_dec_l2_b.float()
        return out.half().view(1, dit_out.shape[1], 132)

    def _run_dit_output_proj(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        D = 1536
        x = torch.nn.functional.silu(temb.float())
        po1_w = self._proj_out_1_w.float() * self._dit_misc_alpha[2]
        po1_b = self._proj_out_1_b.float()
        mod = x @ po1_w + po1_b
        shift, scale = mod.chunk(2, dim=-1)
        h_norm = torch.nn.functional.layer_norm(h.float(), (D,), eps=1e-5)
        h_mod = h_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        po2_w = self._proj_out_2_w.float() * self._dit_misc_alpha[3]
        po2_b = self._proj_out_2_b.float()
        return (h_mod @ po2_w + po2_b).half().contiguous()

    # ── DiT scratch buffers (FP16) ──
    def _allocate_infer_buffers(self, action_horizon: int) -> None:
        Sa = 1 + action_horizon
        D, FF = 1536, 6144
        device = self.device
        self._infer_bufs = {
            "dit_h":          torch.empty((Sa, D), dtype=_FP16, device=device),
            "dit_xn":         torch.empty((Sa, D), dtype=_FP16, device=device),
            "dit_o_proj_out": torch.empty((Sa, D), dtype=_FP16, device=device),
            "dit_ff_proj_out": torch.empty((Sa, FF), dtype=_FP16, device=device),
        }

    # ── RTX DiT attention backend with FP16 slots ──
    def _build_dit_attn(self, Sa: int) -> None:
        from flash_rt.hardware.rtx.attn_backend_groot_n17 import (
            RtxFlashAttnBackendGrootN17,
        )

        Skv_text = int(self._dit_cross_K[0].shape[0])
        Skv_image = int(self._dit_cross_K[1].shape[0])
        attn = RtxFlashAttnBackendGrootN17(
            num_vit_groups=int(getattr(self, "_num_vit_views", 4)),
            llm_seq_max=int(self.Se),
            vl_self_attn_seq_max=int(self.Se),
            sa=int(Sa),
            s_kv_text=Skv_text,
            s_kv_image=Skv_image,
            device=self.device,
            slot_dtype=_FP16,
        )
        for j, (k_src, v_src) in enumerate(zip(self._dit_cross_K, self._dit_cross_V)):
            attn.dit_cross_K[j].view(attn.dit_cross_K[j].shape[0], -1)[
                : k_src.shape[0]
            ].copy_(k_src)
            attn.dit_cross_V[j].view(attn.dit_cross_V[j].shape[0], -1)[
                : v_src.shape[0]
            ].copy_(v_src)
        self._dit_attn = attn

    # ── DiT forward via the full-FP16 pipeline ──
    def _run_dit(self, bufs: dict, shift_list, scale_list, Sa: int) -> None:
        from flash_rt.models.groot_n17 import pipeline_rtx_fp16

        if not hasattr(self, "_dit_attn"):
            self._build_dit_attn(Sa)

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
        Skv_text = int(self._dit_cross_K[0].shape[0])
        Skv_image = int(self._dit_cross_K[1].shape[0])
        dims = {
            "Sa": int(Sa), "D": 1536, "FF": 6144,
            "Skv_text": Skv_text, "Skv_image": Skv_image,
        }
        bufs_ptrs = {
            "h": bufs["dit_h"].data_ptr(),
            "xn": bufs["dit_xn"].data_ptr(),
            "o_proj_out": bufs["dit_o_proj_out"].data_ptr(),
            "ff_proj_out": bufs["dit_ff_proj_out"].data_ptr(),
        }
        if not hasattr(self, "_gemm"):
            import flash_rt.flash_rt_kernels as _fvk
            self._fvk = _fvk
            self._gemm = _fvk.GemmRunner()

        pipeline_rtx_fp16.dit_forward(
            gemm=self._gemm, fvk=self._fvk,
            bufs=bufs_ptrs, weights=weights, dims=dims,
            attn=self._dit_attn,
        )

    # ── CUDA graph capture over the full-FP16 DiT ──
    def _capture_dit_graphs(self, num_inference_timesteps: int = 4,
                            action_horizon: int = 40) -> None:
        from flash_rt.models.groot_n17 import pipeline_rtx_fp16

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

        weights_warm = _weights_for(0)
        for _ in range(3):
            pipeline_rtx_fp16.dit_forward(
                gemm=self._gemm, fvk=self._fvk,
                bufs=bufs_ptrs, weights=weights_warm, dims=dims,
                attn=self._dit_attn,
            )
        torch.cuda.synchronize()

        self._dit_graphs = []
        for step in range(num_inference_timesteps):
            weights = _weights_for(step)
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            s_int = stream.cuda_stream
            with torch.cuda.stream(stream):
                graph.capture_begin()
                pipeline_rtx_fp16.dit_forward(
                    gemm=self._gemm, fvk=self._fvk,
                    bufs=bufs_ptrs, weights=weights, dims=dims,
                    attn=self._dit_attn, stream=s_int,
                )
                graph.capture_end()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self._dit_graphs.append(graph)

    # ── set_prompt: produce backbone_features via FP16 kernels (no torch) ──
    def set_prompt(self, *, aux: dict, prompt: str | None = None) -> None:
        import warnings
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

        # ── Full-FP16 KERNEL backbone (replaces the torch calibration shadow) ──
        self._backbone_features = self._run_kernel_backbone(aux).half()

        try:
            self._warmup_infer()
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"set_prompt warmup failed (non-fatal): {e!r}")

    def _run_kernel_backbone(self, aux: dict) -> "torch.Tensor":
        """Run ViT → DeepStack → LLM → vlln → VL-self-attn through FP16 kernels.

        Returns backbone_features (1, Se, 2048). No PyTorch matmul on the
        feature path — the entire VLM backbone runs through FlashRT kernels.
        """
        import flash_rt.flash_rt_kernels as fvk
        from flash_rt.models.groot_n17 import pipeline_rtx_fp16 as P
        from flash_rt.hardware.rtx.attn_backend_groot_n17_backbone import (
            RtxGrootN17BackboneAttn,
        )

        if not hasattr(self, "_gemm"):
            self._fvk = fvk
            self._gemm = fvk.GemmRunner()
        gemm, fvkm = self._gemm, self._fvk
        dev = self.device
        Sv, nv, Se = self._S_vit, self._num_vit_views, self.Se
        sh = self._fp16_shadow_weights

        keep: list = []

        def K(t):
            keep.append(t)
            return t

        attn = RtxGrootN17BackboneAttn(
            num_vit_views=nv, vit_seq=Sv, llm_seq=Se, vl_self_attn_seq=Se,
            device=dev)
        self._kbb_keep = keep
        self._kbb_attn = attn

        def buf(*shape):
            return K(torch.empty(*shape, dtype=_FP16, device=dev))

        # ═══ ViT (24L) ═══
        vit_h = buf(Sv, 1024)
        vit_h.copy_(aux["pixel_features"].to(dev).half().reshape(Sv, 1024))
        vit_bufs = {"h": vit_h.data_ptr(), "xn": buf(Sv, 1024).data_ptr(),
                    "o_proj_out": buf(Sv, 1024).data_ptr(),
                    "fc1_out": buf(Sv, 4096).data_ptr()}
        vw = {k: [] for k in (
            "norm1_w", "norm1_b", "norm2_w", "norm2_b", "q_w", "q_b",
            "k_w", "k_b", "v_w", "v_b", "o_w", "o_b", "fc1_w", "fc1_b",
            "fc2_w", "fc2_b")}
        vw["cos"] = self._vit_cos.data_ptr()
        vw["sin"] = self._vit_sin.data_ptr()
        for li in range(24):
            qkv = sh[("vit", li, "qkv")]            # (1024, 3072) fp16
            b = self._vit_qkv_b[li]                 # (3072,)
            q = K(qkv[:, :1024].contiguous()); kk = K(qkv[:, 1024:2048].contiguous())
            v = K(qkv[:, 2048:].contiguous())
            qb = K(b[:1024].contiguous()); kb = K(b[1024:2048].contiguous())
            vb = K(b[2048:].contiguous())
            vw["norm1_w"].append(self._vit_ln1_w[li].data_ptr())
            vw["norm1_b"].append(self._vit_ln1_b[li].data_ptr())
            vw["norm2_w"].append(self._vit_ln2_w[li].data_ptr())
            vw["norm2_b"].append(self._vit_ln2_b[li].data_ptr())
            vw["q_w"].append(q.data_ptr()); vw["q_b"].append(qb.data_ptr())
            vw["k_w"].append(kk.data_ptr()); vw["k_b"].append(kb.data_ptr())
            vw["v_w"].append(v.data_ptr()); vw["v_b"].append(vb.data_ptr())
            vw["o_w"].append(sh[("vit", li, "o")].data_ptr())
            vw["o_b"].append(self._vit_o_b[li].data_ptr())
            vw["fc1_w"].append(sh[("vit", li, "fc1")].data_ptr())
            vw["fc1_b"].append(self._vit_fc1_b[li].data_ptr())
            vw["fc2_w"].append(sh[("vit", li, "fc2")].data_ptr())
            vw["fc2_b"].append(self._vit_fc2_b[li].data_ptr())

        tap_layers = (5, 11, 17)
        tap_bufs = {l: buf(Sv, 1024) for l in tap_layers}

        def mk_cb(l):
            def cb(h_ptr):
                fvkm.gpu_copy(tap_bufs[l].data_ptr(), int(h_ptr), Sv * 1024 * 2, 0)
            return cb
        dcap = [mk_cb(l) for l in tap_layers]

        P.qwen3vl_vit_forward(
            gemm=gemm, fvk=fvkm, bufs=vit_bufs, weights=vw,
            dims={"S": Sv, "D": 1024, "NH": 16, "HD": 64,
                  "ff_inner": 4096, "Sper_view": Sv // nv},
            attn=attn, deepstack_taps=tap_layers, deepstack_capture=dcap)

        # ═══ DeepStack (3 mergers) ═══
        Nout = Sv // 4
        ds_out = [buf(Nout, 2048) for _ in range(3)]
        dsw = {k: [] for k in ("norm_w", "norm_b", "fc1_w", "fc1_b",
                                "fc2_w", "fc2_b")}
        for j in range(3):
            dsw["norm_w"].append(getattr(self, f"_dsm{j}_norm_w").data_ptr())
            dsw["norm_b"].append(getattr(self, f"_dsm{j}_norm_b").data_ptr())
            dsw["fc1_w"].append(sh[("dsm", j, "fc1")].data_ptr())
            dsw["fc1_b"].append(getattr(self, f"_dsm{j}_fc1_b").data_ptr())
            dsw["fc2_w"].append(sh[("dsm", j, "fc2")].data_ptr())
            dsw["fc2_b"].append(getattr(self, f"_dsm{j}_fc2_b").data_ptr())
        P.deepstack_merge_forward(
            gemm=gemm, fvk=fvkm,
            bufs={"in": [tap_bufs[l].data_ptr() for l in tap_layers],
                  "ln_out": buf(Nout, 4096).data_ptr(),
                  "fc1_out": buf(Nout, 4096).data_ptr(),
                  "out": [t.data_ptr() for t in ds_out]},
            weights=dsw,
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
        # bufs["Q"]/["K_exp"]/["V_exp"] must alias the attn llm slots so run() reads them.
        lw = {k: [] for k in (
            "in_ln_w", "post_ln_w", "q_norm_w", "k_norm_w", "q_w", "k_w",
            "v_w", "o_w", "gate_w", "up_w", "down_w")}
        lw["cos"] = self._mrope_cos.data_ptr()
        lw["sin"] = self._mrope_sin.data_ptr()
        lw["deepstack_inject"] = inject
        for li in range(16):
            qkv = sh[("llm", li, "qkv")]            # (2048, 4096) fp16
            q = K(qkv[:, :2048].contiguous())
            kk = K(qkv[:, 2048:3072].contiguous())
            v = K(qkv[:, 3072:4096].contiguous())
            lw["in_ln_w"].append(self._llm_input_ln_w[li].data_ptr())
            lw["post_ln_w"].append(self._llm_post_ln_w[li].data_ptr())
            lw["q_norm_w"].append(self._llm_q_norm_w[li].data_ptr())
            lw["k_norm_w"].append(self._llm_k_norm_w[li].data_ptr())
            lw["q_w"].append(q.data_ptr()); lw["k_w"].append(kk.data_ptr())
            lw["v_w"].append(v.data_ptr())
            lw["o_w"].append(sh[("llm", li, "o")].data_ptr())
            lw["gate_w"].append(sh[("llm", li, "gate")].data_ptr())
            lw["up_w"].append(sh[("llm", li, "up")].data_ptr())
            lw["down_w"].append(sh[("llm", li, "down")].data_ptr())
        slots = attn.get_slot_ptrs("llm")
        llm_bufs = {
            "h": llm_h.data_ptr(), "xn": buf(Se, 2048).data_ptr(),
            "Q": slots["Q"], "K": buf(Se, 1024).data_ptr(),
            "V": buf(Se, 1024).data_ptr(),
            "K_exp": slots["K"], "V_exp": slots["V"],
            "o_proj_out": buf(Se, 2048).data_ptr(),
            "gate_out": buf(Se, 6144).data_ptr(),
            "up_out": buf(Se, 6144).data_ptr()}
        P.qwen3vl_llm_forward(
            gemm=gemm, fvk=fvkm, bufs=llm_bufs, weights=lw,
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
            "fc2_w", "fc2_b")}
        for li in range(4):
            vsw["norm1_w"].append(self._vlsa_norm1_w[li].data_ptr())
            vsw["norm1_b"].append(self._vlsa_norm1_b[li].data_ptr())
            vsw["norm3_w"].append(self._vlsa_norm3_w[li].data_ptr())
            vsw["norm3_b"].append(self._vlsa_norm3_b[li].data_ptr())
            vsw["q_w"].append(sh[("vlsa", li, "q")].data_ptr())
            vsw["q_b"].append(self._vlsa_q_b[li].data_ptr())
            vsw["k_w"].append(sh[("vlsa", li, "k")].data_ptr())
            vsw["k_b"].append(self._vlsa_k_b[li].data_ptr())
            vsw["v_w"].append(sh[("vlsa", li, "v")].data_ptr())
            vsw["v_b"].append(self._vlsa_v_b[li].data_ptr())
            vsw["o_w"].append(sh[("vlsa", li, "o")].data_ptr())
            vsw["o_b"].append(self._vlsa_o_b[li].data_ptr())
            vsw["fc1_w"].append(sh[("vlsa", li, "fc1")].data_ptr())
            vsw["fc1_b"].append(self._vlsa_fc1_b[li].data_ptr())
            vsw["fc2_w"].append(sh[("vlsa", li, "fc2")].data_ptr())
            vsw["fc2_b"].append(self._vlsa_fc2_b[li].data_ptr())
        P.vl_self_attn_forward(
            gemm=gemm, fvk=fvkm,
            bufs={"h": vlsa_h.data_ptr(), "xn": buf(Se, 2048).data_ptr(),
                  "o_proj_out": buf(Se, 2048).data_ptr(),
                  "fc1_out": buf(Se, 8192).data_ptr()},
            weights=vsw,
            dims={"T": Se, "D": 2048, "NH": 32, "HD": 64, "ff_inner": 8192},
            attn=attn)
        torch.cuda.synchronize()
        return vlsa_h.unsqueeze(0)

    # ── infer: same as base but action-head tensors in FP16 ──
    def infer(
        self,
        state_normalized: torch.Tensor,
        *,
        initial_noise=None,
        num_inference_timesteps: int = 4,
        action_horizon: int = 40,
        num_timestep_buckets: int = 1000,
        use_dit_graph: bool = True,
    ) -> torch.Tensor:
        if not hasattr(self, "_backbone_features"):
            raise RuntimeError("call set_prompt before infer")
        if not hasattr(self, "_dit_cross_K"):
            self._precompute_dit_cross_kv()

        device = self.device
        action_dim = 132
        Sa = action_horizon + 1

        state_features = self._run_state_encode(
            state_normalized.to(device).half())

        if not hasattr(self, "_infer_bufs"):
            self._allocate_infer_buffers(action_horizon)
        bufs = self._infer_bufs

        if initial_noise is not None:
            actions = initial_noise.to(device).half().contiguous().clone()
        else:
            actions = torch.randn(
                1, action_horizon, action_dim, dtype=_FP16, device=device)

        dt = 1.0 / num_inference_timesteps
        pos_embed = self._ah_pos_embed_w[:action_horizon].half()

        self._infer_shift_lists = []
        self._infer_scale_lists = []
        self._infer_temb_list = []

        graphs = None
        if use_dit_graph:
            if not hasattr(self, "_dit_graphs"):
                self._capture_dit_graphs(
                    num_inference_timesteps=num_inference_timesteps,
                    action_horizon=action_horizon)
            graphs = self._dit_graphs
            if len(graphs) != num_inference_timesteps:
                graphs = None

        for step in range(num_inference_timesteps):
            t_cont = step / num_inference_timesteps
            t_disc = int(t_cont * num_timestep_buckets)

            if graphs is not None:
                temb = self._step_temb[step]
                shift_list = self._step_shifts[step]
                scale_list = self._step_scales[step]
            else:
                temb = self._compute_timestep_emb(t_disc)
                shift_list, scale_list = self._compute_dit_adaln_modulators(temb)
                self._infer_shift_lists.append(shift_list)
                self._infer_scale_lists.append(scale_list)
                self._infer_temb_list.append(temb)

            action_features = self._run_action_encode(
                actions, t_disc, action_horizon)
            action_features = action_features + pos_embed.unsqueeze(0)

            sa_embs = torch.cat([state_features, action_features], dim=1)
            bufs["dit_h"][:Sa].copy_(sa_embs.squeeze(0).contiguous())

            if graphs is not None:
                graphs[step].replay()
            else:
                self._run_dit(bufs, shift_list, scale_list, Sa)

            h_out = self._run_dit_output_proj(bufs["dit_h"][:Sa].unsqueeze(0), temb)
            velocity = self._run_action_decode(h_out[:, -action_horizon:])
            actions = actions + (dt * velocity).to(actions.dtype)

        return actions.float()

    def _warmup_infer(self) -> None:
        warm_state = torch.zeros(1, 1, 132, dtype=torch.float32)
        torch.manual_seed(0)
        warm_noise = torch.randn(1, 40, 132, dtype=_FP16, device=self.device)
        _ = self.infer(warm_state, initial_noise=warm_noise, use_dit_graph=False)
