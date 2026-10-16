"""GROOT N1.7 RTX frontend with an NVFP4 action head (SM120).

``GrootN17TorchFrontendRtxFP4`` (``use_fp4=True`` on ``rtx_sm120``) keeps the
FP8 backbone of :class:`GrootN17TorchFrontendRtxFP8` (fused FP8 producers,
calibrated activation scales) and runs the action head on block-scaled W4A4
GEMMs (see ``flash_rt.models.groot_n17.pipeline_rtx_nvfp4``):

* DiT (M = 41): the skinny NVFP4 family. The consumers of its K-split
  partials apply bias, GELU, residual and (Ada)LayerNorm and emit the next
  NVFP4 operand; self / cross attention reads the Q / QKV projection
  partials directly.
* Cross-attention K/V: two CUTLASS SM120 NVFP4 GEMMs per frame (text rows,
  image rows) over the concatenated K|V weights of their blocks, with the
  bias in the epilogue; activation global scale from the set_prompt frame.

DiT and cross-K/V weights are quantized from the checkpoint's BF16 tensors
(not from the FP8-rounded copies the shared loader keeps) with a
squared-error block-scale search.

``FLASHRT_GROOT_N17_BB_NVFP4`` additionally builds NVFP4 backbone stages
(``vit``, ``llm``, ``vlsa``) for experiments only: W4A4 errors compound
through the ViT and the LLM, so the default backbone stays FP8.
"""

from __future__ import annotations

import glob
import os

import torch

from flash_rt.frontends.torch.groot_n17_rtx_fp8 import GrootN17TorchFrontendRtxFP8

_BLOCK = "action_head.model.transformer_blocks.{}.{}"


class _Runner:
    def __init__(self, vit_h, fn):
        self.vit_h, self._fn = vit_h, fn

    def __call__(self, stream):
        self._fn(stream)


def K_(t: torch.Tensor, keep: list) -> int:
    """Keep a contiguous device tensor alive and return its pointer."""
    t = t.contiguous()
    keep.append(t)
    return t.data_ptr()


class GrootN17TorchFrontendRtxFP4(GrootN17TorchFrontendRtxFP8):
    """N1.7 RTX inference frontend: FP8 backbone + NVFP4 DiT."""

    _DIT_QUANT = "nvfp4_skinny"
    _DIT_WEIGHT_MSE_STEPS = int(os.environ.get("FLASHRT_GROOT_N17_DIT_FP4_MSE", "4"))

    def _ckpt_bf16_tensor(self, key: str) -> torch.Tensor:
        from safetensors import safe_open
        if not hasattr(self, "_ckpt_key_index"):
            index = {}
            for shard in sorted(glob.glob(os.path.join(self.checkpoint_path, "model-*.safetensors"))):
                with safe_open(shard, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        index[k] = shard
            self._ckpt_key_index = index
        with safe_open(self._ckpt_key_index[key], framework="pt", device=str(self.device)) as f:
            return f.get_tensor(key).to(torch.bfloat16)

    def _quantize_nvfp4_site(self, w_nk: torch.Tensor, bias: torch.Tensor) -> tuple:
        """BF16 [N, K] weight -> packed NVFP4 + scales + global scale; keeps
        the tensors alive and returns the pointer tuple the DiT consumes."""
        fvk = self._fvk
        N, K = w_nk.shape
        w_nk = w_nk.contiguous()
        gs = (w_nk.float().abs().amax() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1).contiguous()
        q = torch.empty(N, K // 2, dtype=torch.uint8, device=self.device)
        sf = torch.empty(N, K // 16, dtype=torch.uint8, device=self.device)
        packed = torch.empty_like(q)
        sf_packed = torch.empty_like(sf)
        for rc, what in (
                (fvk.nvfp4_skinny_quantize_weight(w_nk.data_ptr(), gs.data_ptr(), q.data_ptr(), sf.data_ptr(),
                                                  N, K, 0, self._DIT_WEIGHT_MSE_STEPS), "quantize"),
                (fvk.nvfp4_skinny_pack_weight(q.data_ptr(), sf.data_ptr(), packed.data_ptr(),
                                              sf_packed.data_ptr(), N, K, 0), "pack")):
            if rc != 0:
                raise RuntimeError(f"N1.7 NVFP4 DiT weight {what} failed for [{N}, {K}]: cudaError {rc}")
        bias = bias.to(torch.bfloat16).contiguous()
        self._dit_nvfp4_keep += [packed, sf_packed, gs, bias]
        return (packed.data_ptr(), sf_packed.data_ptr(), gs.data_ptr(), bias.data_ptr())

    # ── Backbone ──────────────────────────────────────────────────────

    # Backbone stages on NVFP4 (opt-in; the default backbone is the FP8 one).
    # Measured against the HF activations on real DROID frames, W4A4 error
    # compounds through these stages: the ViT's DeepStack features drop to
    # cos ~0.85 (FP8 ~0.98) and the LLM's errors reach the VL self-attention
    # output (cos ~0.75 vs ~0.96), so the production tier keeps them FP8.
    _BB_NVFP4_STAGES = tuple(s for s in os.environ.get(
        "FLASHRT_GROOT_N17_BB_NVFP4", "").split(",") if s)

    def _ensure_act_scales(self, aux: dict) -> None:
        if "vit" not in self._BB_NVFP4_STAGES:
            super()._ensure_act_scales(aux)

    def _nvfp4_linear(self, w_nk: torch.Tensor, bias: "torch.Tensor | None"):
        """BF16 [N, K] -> NVFP4 weight with swizzled scales for the CUTLASS GEMMs."""
        from flash_rt.models.groot_n17 import pipeline_rtx_nvfp4 as P
        fvk = self._fvk
        N, K = w_nk.shape
        w_nk = w_nk.to(torch.bfloat16).contiguous()
        gs = (w_nk.float().abs().amax() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1).contiguous()
        q = torch.empty(N, K // 2, dtype=torch.uint8, device=self.device)
        sf = torch.empty(N, K // 16, dtype=torch.uint8, device=self.device)
        rc = fvk.nvfp4_skinny_quantize_weight(w_nk.data_ptr(), gs.data_ptr(), q.data_ptr(), sf.data_ptr(),
                                              N, K, 0, self._DIT_WEIGHT_MSE_STEPS)
        if rc != 0:
            raise RuntimeError(f"N1.7 NVFP4 backbone weight quantize failed for [{N}, {K}]: cudaError {rc}")
        swz = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(N, K), dtype=torch.uint8, device=self.device)
        fvk.nvfp4_sf_linear_to_swizzled(sf.data_ptr(), swz.data_ptr(), N, K, True, 0)
        if bias is None:
            bias = self._kbb_zero_bias[:N]
        bias = bias.to(torch.bfloat16).contiguous()
        self._kbb_keep += [q, swz, bias, gs]
        return P.NvFp4Linear(q.data_ptr(), swz.data_ptr(), float(gs.item()), gs.data_ptr(), bias.data_ptr(), N, K)

    def _nvfp4_swiglu_linear(self, gate_nk: torch.Tensor, up_nk: torch.Tensor):
        """Gate / up weights quantized separately, then row-interleaved
        (gate row i -> column 2i, up row i -> 2i + 1) for the SwiGLU-fold GEMM
        whose epilogue applies silu(alpha_g * gate) * alpha_u * up."""
        from types import SimpleNamespace
        fvk = self._fvk
        parts = []
        for w in (gate_nk, up_nk):
            w = w.to(torch.bfloat16).contiguous()
            N, K = w.shape
            gs = (w.float().abs().amax() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1).contiguous()
            q = torch.empty(N, K // 2, dtype=torch.uint8, device=self.device)
            sf = torch.empty(N, K // 16, dtype=torch.uint8, device=self.device)
            rc = fvk.nvfp4_skinny_quantize_weight(w.data_ptr(), gs.data_ptr(), q.data_ptr(), sf.data_ptr(),
                                                  N, K, 0, self._DIT_WEIGHT_MSE_STEPS)
            if rc != 0:
                raise RuntimeError(f"N1.7 NVFP4 SwiGLU weight quantize failed: cudaError {rc}")
            parts.append((q, sf, float(gs.item())))
        (gq, gsf, ga), (uq, usf, ua) = parts
        N, K = gate_nk.shape
        il = torch.empty(2 * N, K // 2, dtype=torch.uint8, device=self.device)
        il[0::2], il[1::2] = gq, uq
        il_sf = torch.empty(2 * N, K // 16, dtype=torch.uint8, device=self.device)
        il_sf[0::2], il_sf[1::2] = gsf, usf
        swz = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(2 * N, K), dtype=torch.uint8, device=self.device)
        fvk.nvfp4_sf_linear_to_swizzled(il_sf.data_ptr(), swz.data_ptr(), 2 * N, K, False, 0)
        self._kbb_keep += [il, swz]
        return SimpleNamespace(w=il.data_ptr(), sf=swz.data_ptr(), alpha_gate=ga, alpha_up=ua, n=N, k=K)

    def _run_kernel_backbone_fp8(self, aux: dict) -> "torch.Tensor":
        """Build (weights, buffers) and run the backbone once; returns the
        FP16 VL self-attention output (the DiT cross-KV source). Without
        NVFP4 stages this is the FP8 backbone."""
        if not self._BB_NVFP4_STAGES:
            return super()._run_kernel_backbone_fp8(aux)
        import flash_rt.flash_rt_kernels as fvk
        from flash_rt import flash_rt_fa2 as fa2
        from flash_rt.models.groot_n17 import pipeline_rtx_nvfp4 as P
        if not hasattr(fvk, "nvfp4_gemm_bias_res_bf16out"):
            raise RuntimeError("the N1.7 NVFP4 backbone needs a GPU_ARCH=120 kernel build")
        if not hasattr(self, "_gemm"):
            self._gemm = fvk.GemmRunner()
        self._fvk = fvk
        dev, bf = self.device, torch.bfloat16
        Sv, nv, Se = self._S_vit, self._num_vit_views, self.Se
        Nout = Sv // 4
        self._kbb_keep = []
        self._kbb_zero_bias = torch.zeros(16384, dtype=bf, device=dev)
        keep = self._kbb_keep

        def T(*shape, dtype=bf, zero=False):
            t = (torch.zeros if zero else torch.empty)(*shape, dtype=dtype, device=dev)
            keep.append(t)
            return t

        def fp4(rows, cols):
            # rows padded to the 128-row tile: the NVFP4-output epilogues run on
            # whole tiles (padding rows stay zero in the operands)
            rows = (rows + 127) // 128 * 128
            return (T(rows, cols // 2, dtype=torch.uint8, zero=True).data_ptr(),
                    T(fvk.nvfp4_sf_bytes(rows, cols), dtype=torch.uint8, zero=True).data_ptr())

        t = self._ckpt_bf16_tensor
        lin = self._nvfp4_linear
        VB = "backbone.model.model.visual.blocks.{}.{}"
        DS = "backbone.model.model.visual.deepstack_merger_list.{}.{}"
        LM = "backbone.model.model.language_model.layers.{}.{}"
        SA = "action_head.vl_self_attention.transformer_blocks.{}.{}"
        W = {"vit": [], "ds": [], "llm": [], "vlsa": []}
        with torch.no_grad():
            for li in range(P.VIT_TAPS[-1] + 1 if "vit" in self._BB_NVFP4_STAGES else 0):
                g = lambda n: t(VB.format(li, n))   # noqa: E731
                W["vit"].append({
                    "ln1_w": K_(g("norm1.weight"), keep), "ln1_b": K_(g("norm1.bias"), keep),
                    "ln2_w": K_(g("norm2.weight"), keep), "ln2_b": K_(g("norm2.bias"), keep),
                    "qkv": lin(g("attn.qkv.weight"), g("attn.qkv.bias")),
                    "o": lin(g("attn.proj.weight"), g("attn.proj.bias")),
                    "fc1": lin(g("mlp.linear_fc1.weight"), g("mlp.linear_fc1.bias")),
                    "fc2": lin(g("mlp.linear_fc2.weight"), g("mlp.linear_fc2.bias"))})
            for j in range(3 if "vit" in self._BB_NVFP4_STAGES else 0):
                g = lambda n: t(DS.format(j, n))   # noqa: E731
                W["ds"].append({"ln_w": K_(g("norm.weight"), keep), "ln_b": K_(g("norm.bias"), keep),
                                "fc1": lin(g("linear_fc1.weight"), g("linear_fc1.bias")),
                                "fc2": lin(g("linear_fc2.weight"), g("linear_fc2.bias"))})
            for li in range(16):
                g = lambda n: t(LM.format(li, n))   # noqa: E731
                W["llm"].append({
                    "in_ln_w": K_(g("input_layernorm.weight"), keep),
                    "post_ln_w": K_(g("post_attention_layernorm.weight"), keep),
                    "q_norm_w": K_(g("self_attn.q_norm.weight"), keep),
                    "k_norm_w": K_(g("self_attn.k_norm.weight"), keep),
                    "qkv": lin(torch.cat([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"),
                                          g("self_attn.v_proj.weight")], 0), None),
                    "o": lin(g("self_attn.o_proj.weight"), None),
                    "gate_up": self._nvfp4_swiglu_linear(g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")),
                    "down": lin(g("mlp.down_proj.weight"), None)})
            for li in range(4):
                g = lambda n: t(SA.format(li, n))   # noqa: E731
                W["vlsa"].append({
                    "n1_w": K_(g("norm1.weight"), keep), "n1_b": K_(g("norm1.bias"), keep),
                    "n3_w": K_(g("norm3.weight"), keep), "n3_b": K_(g("norm3.bias"), keep),
                    "qkv": lin(torch.cat([g("attn1.to_q.weight"), g("attn1.to_k.weight"),
                                          g("attn1.to_v.weight")], 0),
                               torch.cat([g("attn1.to_q.bias"), g("attn1.to_k.bias"), g("attn1.to_v.bias")])),
                    "o": lin(g("attn1.to_out.0.weight"), g("attn1.to_out.0.bias")),
                    "fc1": lin(g("ff.net.0.proj.weight"), g("ff.net.0.proj.bias")),
                    "fc2": lin(g("ff.net.2.weight"), g("ff.net.2.bias"))})
            vlln_w = t("action_head.vlln.weight")
            W["vlln_wm1"] = K_((vlln_w.float() - 1.0).to(bf), keep)
            W["vlln_b"] = K_(t("action_head.vlln.bias"), keep)
            W["vit_cos"] = K_(self._vit_cos.float().reshape(Sv, 64), keep)
            W["vit_sin"] = K_(self._vit_sin.float().reshape(Sv, 64), keep)
            W["llm_cos"] = K_(aux["rope_cos"][0].to(dev).float().reshape(Se, 128), keep)
            W["llm_sin"] = K_(aux["rope_sin"][0].to(dev).float().reshape(Se, 128), keep)

        vis_idx = self._visual_pos_masks.reshape(-1).nonzero(as_tuple=True)[0].to(torch.int64).contiguous()
        keep.append(vis_idx)
        if int(vis_idx.numel()) != Nout:
            raise ValueError(f"N1.7 NVFP4 backbone: {vis_idx.numel()} visual tokens, DeepStack emits {Nout}")
        lse_rows = lambda b, hh, s: T(b * hh * ((s + 127) // 128) * 128, dtype=torch.float32).data_ptr()  # noqa: E731
        vit_h, llm_h = T(Sv, 1024), T(Se, 2048)
        out_fp16 = T(Se, 2048, dtype=torch.float16)
        B = {"one": T(1, dtype=torch.float32).fill_(1.0).data_ptr(),
             "vit_h": vit_h.data_ptr(), "vit_qkv": T(Sv, 3072).data_ptr(), "vit_o": T(Sv, 1024).data_ptr(),
             "vit_lse": lse_rows(nv, 16, Sv // nv),
             "ds_out": [T(Nout, 2048).data_ptr() for _ in range(3)],
             "vis_idx": vis_idx.data_ptr(),
             "llm_h": llm_h.data_ptr(), "llm_qkv": T(Se, 4096).data_ptr(), "llm_o": T(Se, 2048).data_ptr(),
             "llm_lse": lse_rows(1, 16, Se),
             "llm_fold": T((Se + 127) // 128 * 128, 6144, dtype=torch.uint8, zero=True).data_ptr(),
             "vlsa_h": T(Se, 2048).data_ptr(), "vlsa_qkv": T(Se, 6144).data_ptr(),
             "vlsa_o": T(Se, 2048).data_ptr(), "vlsa_lse": lse_rows(1, 32, Se),
             "out_fp16": out_fp16.data_ptr()}
        for key, rows, cols in (("vit_x", Sv, 1024), ("vit_h", Sv, 4096), ("ds_x", Nout, 4096),
                                ("ds_h", Nout, 4096), ("llm_x", Se, 2048), ("llm_g", Se, 6144),
                                ("vlsa_x", Se, 2048), ("vlsa_h", Se, 8192)):
            q, sf = fp4(rows, cols)
            B[key + "q"], B[key + "sf"] = q, sf
        num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
        vit_fp8 = None
        if "vit" not in self._BB_NVFP4_STAGES:
            vit_fp8 = self._build_vit_fp8(aux, B, keep)
        vit_dims = {"S": Sv, "views": nv}
        Se_pad = (Se + 127) // 128 * 128
        llm_dims = {"S": Se, "S_pad": Se_pad, "n_vis": Nout}
        sa_dims = {"S": Se, "S_pad": Se_pad}

        def _kbb_forward(stream=0):
            if vit_fp8 is not None:
                vit_fp8(stream)
            else:
                P.vit_forward_nvfp4(fvk, fa2, B, W, vit_dims, stream=stream, num_sms=num_sms)
            P.llm_forward_nvfp4(fvk, fa2, B, W, llm_dims, stream=stream, num_sms=num_sms)
            P.vlsa_forward_nvfp4(fvk, fa2, B, W, sa_dims, stream=stream, num_sms=num_sms)
            return out_fp16

        self._kbb_forward = _kbb_forward
        self._kbb_nvfp4 = (W, B)
        self._kbb_vit_h_in = vit_fp8.vit_h if vit_fp8 is not None else vit_h
        self._kbb_llm_h = llm_h
        self._kbb_vlsa_h = out_fp16
        self._kbb_scell = [0]
        self._kbb_vit_h = self._kbb_vit_h_in
        self._kbb_vit_h.copy_(aux["pixel_features"].to(dev).reshape(Sv, 1024))
        llm_h.copy_(aux["llm_input_embeds"].to(dev).reshape(Se, 2048))
        _kbb_forward(0)
        torch.cuda.synchronize()
        return out_fp16.unsqueeze(0)

    def _build_vit_fp8(self, aux: dict, B: dict, keep: list):
        """FP8 ViT layers 0..17 + DeepStack mergers (the production FP8 kernels
        on FP16 buffers); the merger outputs are cast into B["ds_out"] (BF16)."""
        from flash_rt.models.groot_n17 import pipeline_rtx_fp8 as P8
        from flash_rt.models.groot_n17 import pipeline_rtx_nvfp4 as P
        from flash_rt.hardware.rtx.attn_backend_groot_n17_backbone import RtxGrootN17BackboneAttn
        fvk, gemm, dev = self._fvk, self._gemm, self.device
        Sv, nv = self._S_vit, self._num_vit_views
        Nout = Sv // 4
        f16 = torch.float16

        def buf(*shape, dtype=f16):
            t = torch.empty(*shape, dtype=dtype, device=dev)
            keep.append(t)
            return t

        def wsc(val):
            t = torch.tensor([float(val)], dtype=torch.float32, device=dev)
            keep.append(t)
            return t.data_ptr()

        attn = RtxGrootN17BackboneAttn(num_vit_views=nv, vit_seq=Sv, llm_seq=self.Se,
                                       vl_self_attn_seq=self.Se, device=dev)
        keep.append(attn)
        vit_h = buf(Sv, 1024)
        vb = {"h": vit_h.data_ptr(), "xn": buf(Sv, 1024).data_ptr(),
              "xn_fp8": buf(Sv, 1024, dtype=torch.uint8).data_ptr(),
              "o_proj_out": buf(Sv, 1024).data_ptr(), "fc1_out": buf(Sv, 4096).data_ptr(),
              "fc1_fp8": buf(Sv, 4096, dtype=torch.uint8).data_ptr()}
        vw = {k: [] for k in ("norm1_w", "norm1_b", "norm2_w", "norm2_b", "q_w", "q_b", "k_w", "k_b", "v_w",
                              "v_b", "o_w", "o_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b", "q_ws", "k_ws", "v_ws",
                              "o_ws", "fc1_ws", "fc2_ws")}
        vw["cos"], vw["sin"] = self._vit_cos.data_ptr(), self._vit_sin.data_ptr()
        for li in range(P.VIT_TAPS[-1] + 1):
            qkv, b = self._vit_qkv_w[li], self._vit_qkv_b[li]
            parts = [qkv[:, i * 1024:(i + 1) * 1024].contiguous() for i in range(3)]
            bparts = [b[i * 1024:(i + 1) * 1024].contiguous() for i in range(3)]
            keep += parts + bparts
            ws = wsc(self._vit_alpha[li * 4 + 0])
            for n, w_, b_ in zip("qkv", parts, bparts):
                vw[f"{n}_w"].append(w_.data_ptr()); vw[f"{n}_b"].append(b_.data_ptr()); vw[f"{n}_ws"].append(ws)
            for n, attr in (("norm1_w", "_vit_ln1_w"), ("norm1_b", "_vit_ln1_b"), ("norm2_w", "_vit_ln2_w"),
                            ("norm2_b", "_vit_ln2_b"), ("o_w", "_vit_o_w"), ("o_b", "_vit_o_b"),
                            ("fc1_w", "_vit_fc1_w"), ("fc1_b", "_vit_fc1_b"), ("fc2_w", "_vit_fc2_w"),
                            ("fc2_b", "_vit_fc2_b")):
                vw[n].append(getattr(self, attr)[li].data_ptr())
            vw["o_ws"].append(wsc(self._vit_alpha[li * 4 + 1]))
            vw["fc1_ws"].append(wsc(self._vit_alpha[li * 4 + 2]))
            vw["fc2_ws"].append(wsc(self._vit_alpha[li * 4 + 3]))
        vs = {k: [t.data_ptr() for t in getattr(self, f"_vit_act_{k}_dev")] for k in ("qkv", "o", "fc1", "fc2")}
        vs = {f"act_{k}": v for k, v in vs.items()}
        ds_in = [buf(Sv, 1024) for _ in range(3)]
        ds_out16 = [buf(Nout, 2048) for _ in range(3)]
        dw = {k: [] for k in ("norm_w", "norm_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b", "fc1_ws", "fc2_ws")}
        for j in range(3):
            for k in ("norm_w", "norm_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b"):
                dw[k].append(getattr(self, f"_dsm{j}_{k}").data_ptr())
            dw["fc1_ws"].append(wsc(self._dsm_alpha[j * 2 + 0]))
            dw["fc2_ws"].append(wsc(self._dsm_alpha[j * 2 + 1]))
        ds_scales = {"act_fc1": [t.data_ptr() for t in self._dsm_act_fc1_dev],
                     "act_fc2": [t.data_ptr() for t in self._dsm_act_fc2_dev]}
        db = {"in": [t.data_ptr() for t in ds_in], "ln_out": buf(Nout, 4096).data_ptr(),
              "fp8_scratch": buf(Nout, 4096, dtype=torch.uint8).data_ptr(),
              "fc1_out": buf(Nout, 4096).data_ptr(), "out": [t.data_ptr() for t in ds_out16]}
        vdims = {"S": Sv, "D": 1024, "NH": 16, "HD": 64, "ff_inner": 4096, "Sper_view": Sv // nv}
        ddims = {"Nin": Sv, "Din": 1024, "Nout": Nout, "Dmid": 4096, "Dout": 2048}
        scell = [0]
        cap = [lambda h, j=j: fvk.gpu_copy(ds_in[j].data_ptr(), int(h), Sv * 1024 * 2, scell[0]) for j in range(3)]

        def run(stream):
            scell[0] = stream
            P8.qwen3vl_vit_forward(gemm=gemm, fvk=fvk, bufs=vb, weights=vw, scales_dev=vs, dims=vdims, attn=attn,
                                   stream=stream, layers_subset=range(P.VIT_TAPS[-1] + 1),
                                   deepstack_taps=P.VIT_TAPS, deepstack_capture=cap)
            P8.deepstack_merge_forward(gemm=gemm, fvk=fvk, bufs=db, weights=dw, scales_dev=ds_scales,
                                       dims=ddims, stream=stream)
            for j in range(3):
                fvk.cast_fp16_to_bf16(ds_out16[j].data_ptr(), B["ds_out"][j], Nout * 2048, stream)

        return _Runner(vit_h, run)

    def run_backbone_graph(self, aux: dict) -> "torch.Tensor":
        """Replay the captured backbone graph with a fresh observation."""
        if not self._BB_NVFP4_STAGES:
            return super().run_backbone_graph(aux)
        self._validate_backbone_graph_contract(aux)
        self._kbb_vit_h.copy_(aux["pixel_features"].to(self.device).reshape(self._S_vit, 1024))
        self._kbb_llm_h.copy_(aux["llm_input_embeds"].to(self.device).reshape(self.Se, 2048))
        self._kbb_graph.replay()
        return self._kbb_vlsa_h.unsqueeze(0)

    # ── DiT ───────────────────────────────────────────────────────────

    def _prepare_dit_nvfp4_skinny(self, step_weights, bp, dims, num_steps: int, Sa: int) -> None:
        from flash_rt.models.groot_n17 import pipeline_rtx_nvfp4 as P
        fvk = self._fvk
        if not hasattr(fvk, "nvfp4_skinny_gemm"):
            raise RuntimeError("the N1.7 NVFP4 DiT needs a GPU_ARCH=120 kernel build (skinny NVFP4 family)")
        D, FF = P.D, P.FF
        dev = self.device
        if not hasattr(self, "_dit_nvfp4_sites"):
            self._dit_nvfp4_keep = []
            qkv, q, o, up, down = [], [], [], [], []
            with torch.no_grad():
                for li in range(P.LAYERS):
                    t = lambda name: self._ckpt_bf16_tensor(_BLOCK.format(li, name))   # noqa: E731
                    if li % 2 == 1:
                        w = torch.cat([t("attn1.to_q.weight"), t("attn1.to_k.weight"), t("attn1.to_v.weight")], 0)
                        b = torch.cat([self._dit_q_b[li], self._dit_k_b[li], self._dit_v_b[li]])
                        qkv.append(self._quantize_nvfp4_site(w, b))
                    else:
                        q.append(self._quantize_nvfp4_site(t("attn1.to_q.weight"), self._dit_q_b[li]))
                    o.append(self._quantize_nvfp4_site(t("attn1.to_out.0.weight"), self._dit_o_b[li]))
                    up.append(self._quantize_nvfp4_site(t("ff.net.0.proj.weight"), self._dit_ff_proj_b[li]))
                    down.append(self._quantize_nvfp4_site(t("ff.net.2.weight"), self._dit_ff_down_b[li]))
            self._dit_nvfp4_sites = {"qkv": qkv, "q": q, "o": o, "ff_up": up, "ff_down": down}
            self._k_dit_xn_fp4 = torch.empty(Sa, D // 2, dtype=torch.uint8, device=dev)
            self._k_dit_xn_sf = torch.zeros(Sa, D // 16, dtype=torch.uint8, device=dev)
            self._k_dit_hid_fp4 = torch.empty(Sa, FF // 2, dtype=torch.uint8, device=dev)
            self._k_dit_hid_sf = torch.zeros(Sa, FF // 16, dtype=torch.uint8, device=dev)
            self._k_dit_partials = torch.empty(P.partial_floats(fvk, Sa), dtype=torch.float32, device=dev)
            self._k_dit_one = torch.ones(1, dtype=torch.float32, device=dev)
        self._dit_nvfp4_bufs = {
            "h": bp["h"],
            "xn": self._k_dit_xn_fp4.data_ptr(), "xn_sf": self._k_dit_xn_sf.data_ptr(),
            "hid": self._k_dit_hid_fp4.data_ptr(), "hid_sf": self._k_dit_hid_sf.data_ptr(),
            "partials": self._k_dit_partials.data_ptr(), "one": self._k_dit_one.data_ptr(),
            "hmod": self._k_hmod.data_ptr(),
        }
        for step in range(num_steps):
            step_weights[step].update(self._dit_nvfp4_sites)
            step_weights[step]["out_scale"] = self._k_oproj_scale[step].data_ptr()
            step_weights[step]["out_shift"] = self._k_oproj_shift[step].data_ptr()

    def _dit_nvfp4_skinny_step(self, step: int, weights: dict, dims: dict, stream: int) -> None:
        from flash_rt.models.groot_n17 import pipeline_rtx_nvfp4 as P
        if hasattr(self, "_ck_nvfp4"):
            dims = dict(dims, kv_ld=self._ck_nvfp4["kv_ld"])
        P.dit_forward_nvfp4(self._fvk, self._dit_nvfp4_bufs, weights, dims, attn=self._dit_attn,
                            stream=int(stream), pdl=os.environ.get("FLASHRT_GROOT_N17_DIT_PDL", "1") != "0")

    # ── Cross K/V ─────────────────────────────────────────────────────
    #
    # The 16 cross-attention blocks project the same two per-frame sources
    # (the backbone's text rows for the even cross blocks, its image rows for
    # the odd ones) to K and V. On NVFP4 they run as two GEMMs, one per source,
    # over the concatenated K|V weights of its 8 blocks (N = 8 * 2 * 1536),
    # with the bias in the epilogue; each block's K / V is a column slice of
    # the output, which the DiT attention reads with its row stride.

    _CK_NVFP4 = os.environ.get("FLASHRT_GROOT_N17_CK_NVFP4", "1") != "0"

    def _setup_cross_kv_kernel(self) -> None:
        fresh = not hasattr(self, "_ck_text_idx")
        super()._setup_cross_kv_kernel()
        if fresh and self._CK_NVFP4 and hasattr(self._fvk, "nvfp4_gemm_bias_res_bf16out"):
            self._setup_cross_kv_nvfp4()

    def _setup_cross_kv_nvfp4(self) -> None:
        fvk = self._fvk
        dev = self.device
        D, Kd = 1536, 2048
        keep: list = []
        groups = {}
        with torch.no_grad():
            for name, src, rows in (("text", self._ck_text_src, self._ck_nt),
                                    ("image", self._ck_image_src, self._ck_ni)):
                blocks = [j for j in range(16) if ((2 * j) % 4 == 0) == (name == "text")]
                w = torch.cat([self._ckpt_bf16_tensor(_BLOCK.format(2 * j, f"attn1.to_{kv}.weight"))
                               for j in blocks for kv in ("k", "v")], 0).contiguous()        # [N, K]
                b = torch.cat([getattr(self, f"_dit_{kv}_b")[2 * j].to(torch.bfloat16)
                               for j in blocks for kv in ("k", "v")]).contiguous()
                N = w.shape[0]
                w_gs = (w.float().abs().amax() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1).contiguous()
                q = torch.empty(N, Kd // 2, dtype=torch.uint8, device=dev)
                sf_lin = torch.empty(N, Kd // 16, dtype=torch.uint8, device=dev)
                rc = fvk.nvfp4_skinny_quantize_weight(w.data_ptr(), w_gs.data_ptr(), q.data_ptr(), sf_lin.data_ptr(),
                                                      N, Kd, 0, self._DIT_WEIGHT_MSE_STEPS)
                if rc != 0:
                    raise RuntimeError(f"N1.7 cross-KV NVFP4 weight quantization failed: cudaError {rc}")
                sfb = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(N, Kd), dtype=torch.uint8, device=dev)
                fvk.nvfp4_sf_linear_to_swizzled(sf_lin.data_ptr(), sfb.data_ptr(), N, Kd, True, 0)
                # static activation global scale from the set_prompt frame
                # (amax / 448: 6x headroom before the block scales saturate)
                a_gs = (src.float().abs().amax() / 448.0).clamp_min(1e-12).reshape(1).contiguous()
                a = torch.empty(rows, Kd // 2, dtype=torch.uint8, device=dev)
                sfa = torch.zeros(fvk.nvfp4_sf_bytes(rows, Kd), dtype=torch.uint8, device=dev)
                out = torch.empty(rows, N, dtype=torch.bfloat16, device=dev)
                alpha = (a_gs * w_gs).contiguous()          # device scalar the GEMM epilogue reads
                keep += [q, sfb, b, w_gs, a_gs, alpha, a, sfa, out]
                groups[name] = {"src": src, "rows": rows, "N": N, "w": q, "sfb": sfb, "b": b, "a": a, "sfa": sfa,
                                "a_gs": a_gs, "alpha": alpha, "out": out, "blocks": blocks}
                del w
        for g in groups.values():
            for t, j in enumerate(g["blocks"]):
                self._dit_cross_K[j] = g["out"][:, 2 * t * D:(2 * t + 1) * D]
                self._dit_cross_V[j] = g["out"][:, (2 * t + 1) * D:(2 * t + 2) * D]
        self._ck_nvfp4 = {"groups": groups, "keep": keep, "kv_ld": groups["text"]["N"]}
        if hasattr(self, "_dit_attn"):
            delattr(self, "_dit_attn")
        self._cross_kv_fwd(0)

    def _cross_kv_fwd(self, s: int) -> None:
        if not hasattr(self, "_ck_nvfp4"):
            return super()._cross_kv_fwd(s)
        K = self._fvk
        S = self.Se
        K.cast_fp16_to_bf16(self._ck_bb_src.data_ptr(), self._ck_bb.data_ptr(), S * 2048, s)
        K.embedding_lookup_bf16(self._ck_text_idx.data_ptr(), self._ck_bb.data_ptr(),
                                self._ck_text_src.data_ptr(), self._ck_nt, 2048, s)
        K.embedding_lookup_bf16(self._ck_image_idx.data_ptr(), self._ck_bb.data_ptr(),
                                self._ck_image_src.data_ptr(), self._ck_ni, 2048, s)
        for name, g in self._ck_nvfp4["groups"].items():
            rc = K.nvfp4_quant(g["src"].data_ptr(), 2048, g["a"].data_ptr(), g["sfa"].data_ptr(),
                               g["a_gs"].data_ptr(), g["rows"], 2048, s)
            if rc == 0:
                rc = K.nvfp4_gemm_bias_res_bf16out(g["a"].data_ptr(), g["w"].data_ptr(), g["sfa"].data_ptr(),
                                                   g["sfb"].data_ptr(), g["b"].data_ptr(), 0, g["out"].data_ptr(),
                                                   g["rows"], g["N"], 2048, g["alpha"].data_ptr(), 0.0, 0, s)
            if rc != 0:
                raise RuntimeError(f"N1.7 cross-KV NVFP4 {name} failed: status {rc}")
