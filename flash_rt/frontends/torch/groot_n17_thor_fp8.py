"""FlashRT -- GROOT N1.7 Thor FP8 frontend (conforming serving path).

The base :class:`GrootN17TorchFrontendThor` produces ``_backbone_features``
from the PyTorch calibration shadow and re-runs that shadow on every
``set_prompt``. This subclass makes the serving feature path conform to the
FlashRT convention used by every other model:

  * Activation scales come from a one-time shadow calibration that is then
    persisted; subsequent prompts load the disk cache and skip the shadow
    entirely (warm path — no torch matmul).
  * ``_backbone_features`` is produced by the FP8 kernels
    (``pipeline_thor.qwen3vl_*``), not the shadow — so the serving feature
    path runs no torch matmul.

DiT, cross-KV precompute, graph capture and ``infer`` are inherited unchanged
from the base.
"""

from __future__ import annotations

import warnings

import torch

from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor

_FP16 = torch.float16
_FP8 = torch.float8_e4m3fn


class GrootN17TorchFrontendThorFP8(GrootN17TorchFrontendThor):
    """N1.7 Thor frontend with an FP8-kernel serving backbone + cached scales."""

    # LLM decoder layers that run their GEMMs in fp16 instead of per-tensor
    # FP8. The Qwen3-VL text decoder carries large visual-token activation
    # spikes whose per-tensor FP8 error — though only ~0.99 per isolated
    # layer — accumulates across the 16-layer stack into a per-position
    # backbone that the action head cannot use (post-vlln cosine collapses
    # to ~0.1; end-to-end action cosine 0.99 → 1.00 once protected). Running
    # the whole decoder in fp16 (still kernel-based, no torch on the feature
    # path) restores it; ViT / DeepStack / VL-self-attn stay FP8. This mirrors
    # the N1.6 precedent of fp16 for the precision-sensitive sub-modules.
    PROTECT_LLM_FP16 = tuple(range(16))

    # ViT / DeepStack / VL-self-attn GEMM precision. The FP8 production
    # frontend keeps these stages in FP8; the full-FP16 reference subclass
    # (:class:`GrootN17TorchFrontendThorFP16`) flips this to run them through
    # the fp16_nn path on the shadow weights, with no activation calibration.
    _KBB_USE_FP8 = True

    def _load_llm_protect_fp16(self) -> None:
        """Load fp16 q/k/v/o/gate/up weights for the protected LLM layers.

        Independent of the calibration shadow (which is freed) and valid on
        both warm and cold paths. Stored split-per-projection in
        ``self._llm_protect_fp16`` keyed by ``(li, name)``."""
        import glob as _glob
        import os as _os
        from safetensors import safe_open

        shards = sorted(_glob.glob(
            _os.path.join(self.checkpoint_path, "model-*.safetensors")))
        handles = [safe_open(p, framework="pt", device=self.device) for p in shards]
        index = {}
        for h in handles:
            for k in h.keys():
                index[k] = h

        def load_w(key):
            return index[key].get_tensor(key).to(_FP16).t().contiguous()

        lp = "backbone.model.model.language_model.layers.{i}"
        store = {}
        for li in self.PROTECT_LLM_FP16:
            p = lp.format(i=li)
            q = index[f"{p}.self_attn.q_proj.weight"].get_tensor(
                f"{p}.self_attn.q_proj.weight").to(_FP16)
            k = index[f"{p}.self_attn.k_proj.weight"].get_tensor(
                f"{p}.self_attn.k_proj.weight").to(_FP16)
            v = index[f"{p}.self_attn.v_proj.weight"].get_tensor(
                f"{p}.self_attn.v_proj.weight").to(_FP16)
            store[(li, "q")] = q.t().contiguous()
            store[(li, "k")] = k.t().contiguous()
            store[(li, "v")] = v.t().contiguous()
            store[(li, "o")] = load_w(f"{p}.self_attn.o_proj.weight")
            store[(li, "gate")] = load_w(f"{p}.mlp.gate_proj.weight")
            store[(li, "up")] = load_w(f"{p}.mlp.up_proj.weight")
            store[(li, "down")] = load_w(f"{p}.mlp.down_proj.weight")
        self._llm_protect_fp16 = store

    # ── Calibration cache (load side; save side inherited from base) ──
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
        """Populate the per-stage ``_<stage>_act_*_dev`` device scalars and
        host alphas. Warm path (cache hit) bakes from disk with no torch;
        cold path runs the shadow ONCE, bakes, and persists."""
        cached = self._load_calibration_cache()
        if cached is not None:
            self._bake_calibration(*self._cache_to_stage_dicts(cached))
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

    # ── set_prompt: cached scales + FP8 kernel backbone ──
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
        self._S_vit_per_view = self._S_vit // self._num_vit_views
        self._visual_pos_masks = aux["visual_pos_masks"][0].to(device)

        # Activation scales: warm cache load (no torch) or one-time shadow.
        self._ensure_act_scales(aux)

        # FP8 KERNEL backbone — built once, then captured as a single CUDA
        # graph so the per-observation hot path is one graph replay with zero
        # Python launch overhead. The graph is the producer of the backbone
        # features (and is reusable for a new observation via
        # ``run_backbone_graph``); no torch matmul touches the feature path.
        self._run_kernel_backbone(aux)
        self._capture_backbone_graph()
        self._backbone_features = self.run_backbone_graph(aux).clone().half()

        try:
            self._warmup_infer()
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"set_prompt warmup failed (non-fatal): {e!r}")

    # ── Patch embed (image → ViT input), kernelized ───────────────────────
    def _fast_pos_embed_interpolate(self, grid_thw) -> "torch.Tensor":
        """Bilinear interpolation of the ViT position-embedding table to the
        per-image grid — a pure-torch port of Qwen3-VL
        ``fast_pos_embed_interpolate`` (one-time, per prompt). Returns
        ``(S_vit, 1024)``."""
        dev = self.device
        pos_w = self._vit_pos_embed.float()           # (2304, 1024)
        side = int(round(pos_w.shape[0] ** 0.5))      # 48
        merge = 2
        idx_list = [[] for _ in range(4)]
        wgt_list = [[] for _ in range(4)]
        for t, h, w in [tuple(int(x) for x in row) for row in grid_thw]:
            h_i = torch.linspace(0, side - 1, h)
            w_i = torch.linspace(0, side - 1, w)
            hf, wf = h_i.int(), w_i.int()
            hc = (hf + 1).clip(max=side - 1)
            wc = (wf + 1).clip(max=side - 1)
            dh, dw = h_i - hf, w_i - wf
            bh, bhc = hf * side, hc * side
            inds = [(bh[None].T + wf[None]).flatten(), (bh[None].T + wc[None]).flatten(),
                    (bhc[None].T + wf[None]).flatten(), (bhc[None].T + wc[None]).flatten()]
            wgts = [((1 - dh)[None].T * (1 - dw)[None]).flatten(), ((1 - dh)[None].T * dw[None]).flatten(),
                    (dh[None].T * (1 - dw)[None]).flatten(), (dh[None].T * dw[None]).flatten()]
            for i in range(4):
                idx_list[i].extend(inds[i].tolist())
                wgt_list[i].extend(wgts[i].tolist())
        idx = torch.tensor(idx_list, dtype=torch.long, device=dev)
        wgt = torch.tensor(wgt_list, dtype=torch.float32, device=dev)
        pe = pos_w[idx] * wgt[:, :, None]
        patch = pe[0] + pe[1] + pe[2] + pe[3]
        grids = [(int(h), int(w)) for _, h, w in
                 [tuple(int(x) for x in row) for row in grid_thw]]
        ts = [int(t) for t, _, _ in [tuple(int(x) for x in row) for row in grid_thw]]
        patch = patch.split([h * w for h, w in grids])
        out = []
        for pe_g, t, (h, w) in zip(patch, ts, grids):
            pe_g = pe_g.repeat(t, 1)
            pe_g = (pe_g.view(t, h // merge, merge, w // merge, merge, -1)
                    .permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
            out.append(pe_g)
        return torch.cat(out)

    # ── FP8 kernel backbone: ViT → DeepStack → LLM → vlln → VL self-attn ──
    def _run_kernel_backbone(self, aux: dict) -> "torch.Tensor":
        import flash_rt.flash_rt_kernels as fvk
        from flash_rt.models.groot_n17 import pipeline_thor as P
        from flash_rt.hardware.thor.attn_backend_groot_n17 import (
            ThorGrootN17AttnBackend, make_groot_n17_attention_spec,
        )

        if not hasattr(self, "_gemm"):
            self._fvk = fvk
            self._gemm = fvk.GemmRunner()
        gemm, fvkm = self._gemm, self._fvk
        dev = self.device
        Sv, nv, Se = self._S_vit, self._num_vit_views, self.Se

        # Full-FP16 reference: run ViT / DeepStack / VL-self-attn through
        # fp16_nn on the shadow weights instead of FP8. The shadow holds [K, N]
        # fp16 GEMM weights for every stage; biases/norms are shared with the
        # FP8 path. No activation scales are needed in this mode.
        fp16_ref = not self._KBB_USE_FP8
        sh = self._fp16_shadow_weights if fp16_ref else None

        keep: list = []
        self._kbb_keep = keep

        def K(t):
            keep.append(t)
            return t

        def buf(*shape):
            return K(torch.empty(*shape, dtype=_FP16, device=dev))

        def buf8(*shape):
            return K(torch.empty(*shape, dtype=_FP8, device=dev))

        def wsc(val):
            """Upload a host weight scale to a device fp32 scalar; keep ref."""
            t = K(torch.tensor([float(val)], dtype=torch.float32, device=dev))
            return t.data_ptr()

        def adv(dev_list):
            return [t.data_ptr() for t in dev_list]

        def _pad8(n: int) -> int:
            return ((int(n) + 7) // 8) * 8

        # ── Backbone attention backend (real vit/llm/vlsa slots) ──
        vitQ = buf(Sv, 1024); vitKk = buf(Sv, 1024)
        vitV = buf(Sv, 1024); vitO = buf(Sv, 1024)
        ctx = K(fvk.FvkContext())
        llmQ = buf(Se, 2048); llmKx = buf(Se, 2048)
        llmVx = buf(Se, 2048); llmO = buf(Se, 2048)
        llmLog = buf(16, Se, _pad8(Se))
        vsaQ = buf(Se, 2048); vsaK = buf(Se, 2048)
        vsaV = buf(Se, 2048); vsaO = buf(Se, 2048)
        vsaLog = buf(32, Se, _pad8(Se))

        spec = make_groot_n17_attention_spec(
            num_views=nv, llm_seq_max=Se, vl_self_attn_seq_max=Se,
            sa=41, s_kv_text=128, s_kv_image=512)
        nL_cross = spec.site("dit_cross").num_layers
        attn = ThorGrootN17AttnBackend(
            spec,
            vit_slots={"qkv": 0, "Q": vitQ.data_ptr(), "K": vitKk.data_ptr(),
                       "V": vitV.data_ptr(), "O": vitO.data_ptr(), "D": 16 * 64},
            llm_slots={"ctx": ctx, "Q": llmQ.data_ptr(), "K": llmKx.data_ptr(),
                       "V": llmVx.data_ptr(), "O": llmO.data_ptr(),
                       "logits": llmLog.data_ptr(), "scale": 1.0 / (128 ** 0.5)},
            vl_self_attn_slots={"ctx": ctx, "Q": vsaQ.data_ptr(),
                                "K": vsaK.data_ptr(), "V": vsaV.data_ptr(),
                                "O": vsaO.data_ptr(), "logits": vsaLog.data_ptr(),
                                "scale": 1.0 / (64 ** 0.5)},
            dit_self_slots={"ctx": ctx, "Q": 1, "K": 2, "V": 3, "O": 4,
                            "logits": 5, "scale": 1.0 / (48 ** 0.5)},
            dit_cross_slots={"ctx": ctx, "Q": 1,
                             "K_layers": [10 + i for i in range(nL_cross)],
                             "V_layers": [20 + i for i in range(nL_cross)],
                             "O": 4, "logits": 5, "scale": 1.0 / (48 ** 0.5)},
        )
        self._kbb_attn = attn

        # ═══ ViT (24L) ═══
        vit_h = buf(Sv, 1024)
        # When the caller supplies raw ``pixel_values`` the patch embed runs
        # in-kernel as the graph's first op (fp16_nn + bias + interpolated pos
        # embed → vit_h); otherwise the pre-embedded ``pixel_features`` are
        # copied straight in (legacy path).
        use_pe = "pixel_values" in aux
        if use_pe:
            pv_buf = buf(Sv, 1536)
            pv_buf.copy_(aux["pixel_values"].to(dev).half().reshape(Sv, 1536))
            pe_W = K(self._patch_embed_w.reshape(1024, 1536).t().contiguous().to(_FP16))
            pe_b = K(self._patch_embed_b.to(_FP16).contiguous())
            pe_pos = K(self._fast_pos_embed_interpolate(aux["grid_thw"].tolist())
                       .to(_FP16).contiguous())
            self._kbb_pv = pv_buf
        else:
            vit_h.copy_(aux["pixel_features"].to(dev).half().reshape(Sv, 1024))
        vit_bufs = {"h": vit_h.data_ptr(), "xn": buf(Sv, 1024).data_ptr(),
                    "xn_fp8": buf8(Sv, 1024).data_ptr(),
                    "o_proj_out": buf(Sv, 1024).data_ptr(),
                    "fc1_out": buf(Sv, 4096).data_ptr(),
                    "fc1_fp8": buf8(Sv, 4096).data_ptr()}
        vw = {k: [] for k in (
            "norm1_w", "norm1_b", "norm2_w", "norm2_b", "q_w", "q_b",
            "k_w", "k_b", "v_w", "v_b", "o_w", "o_b", "fc1_w", "fc1_b",
            "fc2_w", "fc2_b", "alpha_q", "alpha_k", "alpha_v", "alpha_o",
            "alpha_fc1", "alpha_fc2")}
        vw["cos"] = self._vit_cos.data_ptr()
        vw["sin"] = self._vit_sin.data_ptr()
        for li in range(24):
            qkv = self._vit_qkv_w[li]               # fp8 (1024, 3072) [K, 3N]
            b = self._vit_qkv_b[li]                  # (3072,)
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
            vw["o_w"].append(self._vit_o_w[li].data_ptr())
            vw["o_b"].append(self._vit_o_b[li].data_ptr())
            vw["fc1_w"].append(self._vit_fc1_w[li].data_ptr())
            vw["fc1_b"].append(self._vit_fc1_b[li].data_ptr())
            vw["fc2_w"].append(self._vit_fc2_w[li].data_ptr())
            vw["fc2_b"].append(self._vit_fc2_b[li].data_ptr())
            if fp16_ref:
                # Split the fused [K, 3N] shadow qkv into per-projection [K, N].
                qkv16 = sh[("vit", li, "qkv")]
                vw.setdefault("q_w_fp16", []).append(K(qkv16[:, :1024].contiguous()).data_ptr())
                vw.setdefault("k_w_fp16", []).append(K(qkv16[:, 1024:2048].contiguous()).data_ptr())
                vw.setdefault("v_w_fp16", []).append(K(qkv16[:, 2048:].contiguous()).data_ptr())
                vw.setdefault("o_w_fp16", []).append(sh[("vit", li, "o")].data_ptr())
                vw.setdefault("fc1_w_fp16", []).append(sh[("vit", li, "fc1")].data_ptr())
                vw.setdefault("fc2_w_fp16", []).append(sh[("vit", li, "fc2")].data_ptr())
                continue
            # qkv is fused → single weight scale shared across q/k/v.
            vw["alpha_q"].append(self._vit_alpha_q[li])
            vw["alpha_k"].append(self._vit_alpha_q[li])
            vw["alpha_v"].append(self._vit_alpha_q[li])
            vw["alpha_o"].append(self._vit_alpha_o[li])
            vw["alpha_fc1"].append(self._vit_alpha_fc1[li])
            vw["alpha_fc2"].append(self._vit_alpha_fc2[li])
        vit_scales = None if fp16_ref else {
            "act_qkv": adv(self._vit_act_qkv_dev), "act_o": adv(self._vit_act_o_dev),
            "act_fc1": adv(self._vit_act_fc1_dev), "act_fc2": adv(self._vit_act_fc2_dev)}

        tap_layers = (5, 11, 17)
        tap_bufs = {l: buf(Sv, 1024) for l in tap_layers}

        # Mutable stream cell so the DeepStack tap copies land on the same
        # stream as the rest of the backbone (required under graph capture).
        scell = [0]
        self._kbb_scell = scell

        def mk_cb(l):
            def cb(h_ptr):
                fvkm.gpu_copy(tap_bufs[l].data_ptr(), int(h_ptr), Sv * 1024 * 2, scell[0])
            return cb
        dcap = [mk_cb(l) for l in tap_layers]

        P.qwen3vl_vit_forward(
            gemm=gemm, fvk=fvkm, bufs=vit_bufs, weights=vw, scales_dev=vit_scales,
            dims={"S": Sv, "D": 1024, "NH": 16, "HD": 64,
                  "ff_inner": 4096, "Sper_view": Sv // nv},
            attn=attn, deepstack_taps=tap_layers, deepstack_capture=dcap,
            use_fp8=self._KBB_USE_FP8)

        # ═══ DeepStack (3 mergers) ═══
        Nout = Sv // 4
        ds_out = [buf(Nout, 2048) for _ in range(3)]
        dsw = {k: [] for k in ("norm_w", "norm_b", "fc1_w", "fc1_b",
                                "fc2_w", "fc2_b", "alpha_fc1", "alpha_fc2")}
        for j in range(3):
            dsw["norm_w"].append(getattr(self, f"_dsm{j}_norm_w").data_ptr())
            dsw["norm_b"].append(getattr(self, f"_dsm{j}_norm_b").data_ptr())
            dsw["fc1_w"].append(getattr(self, f"_dsm{j}_fc1_w").data_ptr())
            dsw["fc1_b"].append(getattr(self, f"_dsm{j}_fc1_b").data_ptr())
            dsw["fc2_w"].append(getattr(self, f"_dsm{j}_fc2_w").data_ptr())
            dsw["fc2_b"].append(getattr(self, f"_dsm{j}_fc2_b").data_ptr())
            if fp16_ref:
                dsw.setdefault("fc1_w_fp16", []).append(sh[("dsm", j, "fc1")].data_ptr())
                dsw.setdefault("fc2_w_fp16", []).append(sh[("dsm", j, "fc2")].data_ptr())
                continue
            dsw["alpha_fc1"].append(self._dsm_alpha_fc1[j])
            dsw["alpha_fc2"].append(self._dsm_alpha_fc2[j])
        ds_scales = None if fp16_ref else {
            "act_fc1": adv(self._dsm_act_fc1_dev),
            "act_fc2": adv(self._dsm_act_fc2_dev)}
        ds_bufs = {"in": [tap_bufs[l].data_ptr() for l in tap_layers],
                   "ln_out": buf(Nout, 4096).data_ptr(),
                   "fp8_scratch": buf8(Nout, 4096).data_ptr(),
                   "fc1_out": buf(Nout, 4096).data_ptr(),
                   "out": [t.data_ptr() for t in ds_out]}
        P.deepstack_merge_forward(
            gemm=gemm, fvk=fvkm, bufs=ds_bufs,
            weights=dsw, scales_dev=ds_scales,
            dims={"Nin": Sv, "Din": 1024, "Nout": Nout, "Dmid": 4096, "Dout": 2048},
            use_fp8=self._KBB_USE_FP8)

        # DeepStack inject buffers (Se, 2048) — zero except visual positions.
        # Scatter via a fixed index_copy (capturable; bit-identical between the
        # eager and the graph paths).
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

        # ═══ LLM (16L, causal, GQA) ═══
        llm_h = buf(Se, 2048)
        llm_h.copy_(aux["llm_input_embeds"].to(dev).half().reshape(Se, 2048))
        if not hasattr(self, "_llm_protect_fp16"):
            self._load_llm_protect_fp16()
        lw = {k: [] for k in (
            "in_ln_w", "post_ln_w", "q_norm_w", "k_norm_w", "q_w", "k_w",
            "v_w", "o_w", "gate_w", "up_w", "down_w",
            "d_w_q", "d_w_k", "d_w_v", "d_w_o", "d_w_gate", "d_w_up", "d_w_down",
            "q_w_fp16", "k_w_fp16", "v_w_fp16", "o_w_fp16",
            "gate_w_fp16", "up_w_fp16", "down_w_fp16")}
        lw["cos"] = self._mrope_cos.data_ptr()
        lw["sin"] = self._mrope_sin.data_ptr()
        lw["deepstack_inject"] = inject
        prot = self._llm_protect_fp16
        for li in range(16):
            qkv = self._llm_qkv_w[li]                # fp8 (2048, 4096) [K, N]
            q = K(qkv[:, :2048].contiguous())
            kk = K(qkv[:, 2048:3072].contiguous())
            v = K(qkv[:, 3072:4096].contiguous())
            lw["in_ln_w"].append(self._llm_input_ln_w[li].data_ptr())
            lw["post_ln_w"].append(self._llm_post_ln_w[li].data_ptr())
            lw["q_norm_w"].append(self._llm_q_norm_w[li].data_ptr())
            lw["k_norm_w"].append(self._llm_k_norm_w[li].data_ptr())
            lw["q_w"].append(q.data_ptr()); lw["k_w"].append(kk.data_ptr())
            lw["v_w"].append(v.data_ptr())
            lw["o_w"].append(self._llm_o_w[li].data_ptr())
            lw["gate_w"].append(self._llm_gate_w[li].data_ptr())
            lw["up_w"].append(self._llm_up_w[li].data_ptr())
            lw["down_w"].append(self._llm_down_w[li].data_ptr())
            qkv_ws = wsc(self._llm_alpha[li * 5 + 0])   # fused qkv weight scale
            lw["d_w_q"].append(qkv_ws); lw["d_w_k"].append(qkv_ws); lw["d_w_v"].append(qkv_ws)
            lw["d_w_o"].append(wsc(self._llm_alpha[li * 5 + 1]))
            lw["d_w_gate"].append(wsc(self._llm_alpha[li * 5 + 2]))
            lw["d_w_up"].append(wsc(self._llm_alpha[li * 5 + 3]))
            lw["d_w_down"].append(wsc(self._llm_alpha[li * 5 + 4]))
            # fp16-protect weight ptrs (only valid for protected layers; 0
            # elsewhere — qwen3vl_llm_forward only reads them for fp16_layers).
            if (li, "q") in prot:
                lw["q_w_fp16"].append(prot[(li, "q")].data_ptr())
                lw["k_w_fp16"].append(prot[(li, "k")].data_ptr())
                lw["v_w_fp16"].append(prot[(li, "v")].data_ptr())
                lw["o_w_fp16"].append(prot[(li, "o")].data_ptr())
                lw["gate_w_fp16"].append(prot[(li, "gate")].data_ptr())
                lw["up_w_fp16"].append(prot[(li, "up")].data_ptr())
                lw["down_w_fp16"].append(prot[(li, "down")].data_ptr())
            else:
                for kk2 in ("q_w_fp16", "k_w_fp16", "v_w_fp16", "o_w_fp16",
                            "gate_w_fp16", "up_w_fp16", "down_w_fp16"):
                    lw[kk2].append(0)
        llm_scales = None if fp16_ref else {
            "act_qkv": adv(self._llm_act_qkv_dev), "act_o": adv(self._llm_act_o_dev),
            "act_gateup": adv(self._llm_act_gateup_dev),
            "act_down": adv(self._llm_act_down_dev)}
        llm_bufs = {
            "h": llm_h.data_ptr(), "xn": buf(Se, 2048).data_ptr(),
            "xn_fp8": buf8(Se, 2048).data_ptr(),
            "Q": llmQ.data_ptr(), "K": buf(Se, 1024).data_ptr(),
            "V": buf(Se, 1024).data_ptr(),
            "K_exp": llmKx.data_ptr(), "V_exp": llmVx.data_ptr(),
            "o_proj_out": buf(Se, 2048).data_ptr(),
            "gate_out": buf(Se, 6144).data_ptr(),
            "up_out": buf(Se, 6144).data_ptr(),
            "gu_fp8": buf8(Se, 6144).data_ptr()}
        P.qwen3vl_llm_forward(
            gemm=gemm, fvk=fvkm, bufs=llm_bufs, weights=lw, scales_dev=llm_scales,
            dims={"S": Se, "D": 2048, "NHQ": 16, "NHKV": 8, "HD": 128, "FF": 6144},
            attn=attn, fp16_layers=self.PROTECT_LLM_FP16)

        # ═══ vlln + VL self-attn (4L) ═══
        vlsa_h = buf(Se, 2048)
        P.vlln_forward(
            gemm=None, fvk=fvkm,
            bufs={"x": llm_h.data_ptr(), "out": vlsa_h.data_ptr()},
            weights={"vlln_w": self._vlln_w.data_ptr(),
                     "vlln_b": self._vlln_b.data_ptr()},
            dims={"S": Se, "D": 2048})
        vsw = {k: [] for k in (
            "norm1_w", "norm1_b", "norm3_w", "norm3_b", "q_w", "q_b",
            "k_w", "k_b", "v_w", "v_b", "o_w", "o_b", "fc1_w", "fc1_b",
            "fc2_w", "fc2_b", "alpha_q", "alpha_k", "alpha_v", "alpha_o",
            "alpha_fc1", "alpha_fc2")}
        for li in range(4):
            vsw["norm1_w"].append(self._vlsa_norm1_w[li].data_ptr())
            vsw["norm1_b"].append(self._vlsa_norm1_b[li].data_ptr())
            vsw["norm3_w"].append(self._vlsa_norm3_w[li].data_ptr())
            vsw["norm3_b"].append(self._vlsa_norm3_b[li].data_ptr())
            vsw["q_w"].append(self._vlsa_q_w[li].data_ptr())
            vsw["q_b"].append(self._vlsa_q_b[li].data_ptr())
            vsw["k_w"].append(self._vlsa_k_w[li].data_ptr())
            vsw["k_b"].append(self._vlsa_k_b[li].data_ptr())
            vsw["v_w"].append(self._vlsa_v_w[li].data_ptr())
            vsw["v_b"].append(self._vlsa_v_b[li].data_ptr())
            vsw["o_w"].append(self._vlsa_o_w[li].data_ptr())
            vsw["o_b"].append(self._vlsa_o_b[li].data_ptr())
            vsw["fc1_w"].append(self._vlsa_fc1_w[li].data_ptr())
            vsw["fc1_b"].append(self._vlsa_fc1_b[li].data_ptr())
            vsw["fc2_w"].append(self._vlsa_fc2_w[li].data_ptr())
            vsw["fc2_b"].append(self._vlsa_fc2_b[li].data_ptr())
            if fp16_ref:
                for nm in ("q", "k", "v", "o", "fc1", "fc2"):
                    vsw.setdefault(f"{nm}_w_fp16", []).append(sh[("vlsa", li, nm)].data_ptr())
                continue
            vsw["alpha_q"].append(self._vlsa_alpha_q[li])
            vsw["alpha_k"].append(self._vlsa_alpha_k[li])
            vsw["alpha_v"].append(self._vlsa_alpha_v[li])
            vsw["alpha_o"].append(self._vlsa_alpha_o[li])
            vsw["alpha_fc1"].append(self._vlsa_alpha_fc1[li])
            vsw["alpha_fc2"].append(self._vlsa_alpha_fc2[li])
        vsa_scales = None if fp16_ref else {
            "act_qkv": adv(self._vlsa_act_qkv_dev), "act_o": adv(self._vlsa_act_o_dev),
            "act_fc1": adv(self._vlsa_act_fc1_dev), "act_fc2": adv(self._vlsa_act_fc2_dev)}
        vsa_bufs = {"h": vlsa_h.data_ptr(), "xn": buf(Se, 2048).data_ptr(),
                    "xn_fp8": buf8(Se, 2048).data_ptr(),
                    "o_proj_out": buf(Se, 2048).data_ptr(),
                    "fc1_out": buf(Se, 8192).data_ptr(),
                    "fc1_fp8": buf8(Se, 8192).data_ptr()}
        P.vl_self_attn_forward(
            gemm=gemm, fvk=fvkm, bufs=vsa_bufs,
            weights=vsw, scales_dev=vsa_scales,
            dims={"T": Se, "D": 2048, "NH": 32, "HD": 64, "ff_inner": 8192},
            attn=attn, use_fp8=self._KBB_USE_FP8)
        torch.cuda.synchronize()

        # ── Stash a graph-capturable pure-kernel forward over the persistent
        # buffers above (no Python dict rebuild, no torch input prep). Inputs
        # (pixel_features / llm_input_embeds) are written into vit_h / llm_h by
        # the caller before replay; the DeepStack inject scatter uses a fixed
        # index_copy (capturable) instead of boolean-mask assignment. ──
        vit_dims = {"S": Sv, "D": 1024, "NH": 16, "HD": 64,
                    "ff_inner": 4096, "Sper_view": Sv // nv}
        ds_dims = {"Nin": Sv, "Din": 1024, "Nout": Nout, "Dmid": 4096, "Dout": 2048}
        llm_dims = {"S": Se, "D": 2048, "NHQ": 16, "NHKV": 8, "HD": 128, "FF": 6144}
        vlln_bufs = {"x": llm_h.data_ptr(), "out": vlsa_h.data_ptr()}
        vlln_w = {"vlln_w": self._vlln_w.data_ptr(), "vlln_b": self._vlln_b.data_ptr()}
        vsa_dims = {"T": Se, "D": 2048, "NH": 32, "HD": 64, "ff_inner": 8192}
        use_fp8 = self._KBB_USE_FP8

        def _kbb_forward(s=0):
            scell[0] = s
            if use_pe:
                gemm.fp16_nn(pv_buf.data_ptr(), pe_W.data_ptr(), vit_h.data_ptr(),
                             Sv, 1024, 1536, s)
                fvkm.add_bias_fp16(vit_h.data_ptr(), pe_b.data_ptr(), Sv, 1024, s)
                fvkm.residual_add_fp16(vit_h.data_ptr(), pe_pos.data_ptr(), Sv * 1024, s)
            P.qwen3vl_vit_forward(gemm=gemm, fvk=fvkm, bufs=vit_bufs, weights=vw,
                                  scales_dev=vit_scales, dims=vit_dims, attn=attn,
                                  deepstack_taps=tap_layers, deepstack_capture=dcap,
                                  use_fp8=use_fp8, stream=s)
            P.deepstack_merge_forward(gemm=gemm, fvk=fvkm, bufs=ds_bufs, weights=dsw,
                                      scales_dev=ds_scales, dims=ds_dims,
                                      use_fp8=use_fp8, stream=s)
            for j in range(3):
                injb[j].zero_()
                injb[j].index_copy_(0, vis_idx, ds_out[j])
            P.qwen3vl_llm_forward(gemm=gemm, fvk=fvkm, bufs=llm_bufs, weights=lw,
                                  scales_dev=llm_scales, dims=llm_dims, attn=attn,
                                  fp16_layers=self.PROTECT_LLM_FP16, stream=s)
            P.vlln_forward(gemm=None, fvk=fvkm, bufs=vlln_bufs, weights=vlln_w,
                           dims={"S": Se, "D": 2048}, stream=s)
            P.vl_self_attn_forward(gemm=gemm, fvk=fvkm, bufs=vsa_bufs, weights=vsw,
                                   scales_dev=vsa_scales, dims=vsa_dims, attn=attn,
                                   use_fp8=use_fp8, stream=s)
            return vlsa_h
        self._kbb_forward = _kbb_forward
        self._kbb_vit_h = vit_h
        self._kbb_llm_h = llm_h
        self._kbb_vlsa_h = vlsa_h
        return vlsa_h.unsqueeze(0)

    # ── Backbone CUDA graph: capture the full ViT→DeepStack→LLM→VLSA kernel
    # chain once so the per-observation hot path is a single graph replay with
    # zero Python launch overhead (the inputs are copied into the persistent
    # vit_h / llm_h buffers before replay). Requires _run_kernel_backbone to
    # have run once (it stashes the pure-kernel forward closure). ──
    def _capture_backbone_graph(self) -> None:
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                self._kbb_forward(s.cuda_stream)
        torch.cuda.synchronize()
        self._kbb_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            self._kbb_graph.capture_begin()
            self._kbb_forward(s.cuda_stream)
            self._kbb_graph.capture_end()
        torch.cuda.synchronize()
        self._kbb_scell[0] = 0  # replay runs on the default stream

    def run_backbone_graph(self, aux: dict) -> "torch.Tensor":
        """Per-observation hot path: write fresh inputs into the persistent
        buffers and replay the captured backbone graph."""
        dev = self.device
        if hasattr(self, "_kbb_pv") and "pixel_values" in aux:
            self._kbb_pv.copy_(
                aux["pixel_values"].to(dev).half().reshape(self._S_vit, 1536))
        else:
            self._kbb_vit_h.copy_(
                aux["pixel_features"].to(dev).half().reshape(self._S_vit, 1024))
        self._kbb_llm_h.copy_(
            aux["llm_input_embeds"].to(dev).half().reshape(self.Se, 2048))
        self._kbb_graph.replay()
        return self._kbb_vlsa_h.unsqueeze(0)
