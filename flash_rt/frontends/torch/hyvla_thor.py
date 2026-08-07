"""HyVLATorchFrontendThor — native FlashRT frontend for Hy-Embodied-0.5-VLA
on Thor SM110 (BF16 baseline).

Reachable via ``flash_rt.load_model(ckpt, config="hyvla", framework="torch")``.
Owns the full inference IO path with **no reference training-code import at
runtime**: tokenizer (checkpoint ``AutoTokenizer`` + hard-coded hy special
token ids), image preprocessing (resize-with-pad + ``*2-1``), prefix assembly
(BOS / hy_User / per-camera vision blocks / language / hy_Assistant), the
segmented prefix mask + prefix-LM suffix mask, the NTK-alpha RoPE tables, and
the flow-matching time embeddings. The heavy math lives in
``flash_rt.models.hyvla.pipeline_thor`` (ViT+merger, MoT prefill, expert
denoise), validated at cosine ≥ 0.9998 vs the HF reference eager path.

The BF16 baseline uses plain torch matmuls and SDPA attention; the
optimized path swaps GEMMs/norms for ``fvk`` pointer kernels + a single
CUDA graph + FP8.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.executors.torch_weights import SafetensorsSource, WeightLoader
from flash_rt.frontends.torch._hyvla_thor_spec import build_spec
from flash_rt.models.hyvla.pipeline_thor import HyVLAThorBF16Pipeline

_BF16 = torch.bfloat16

# hy special token ids (verified from the checkpoint tokenizer_config).
_TOK_BOS = 120000
_TOK_HY_USER = 120006
_TOK_VISION_START = 120684   # <｜hy_place▁holder▁no▁666｜>
_TOK_VISION_END = 120685     # <｜hy_place▁holder▁no▁667｜>
_TOK_VISION_SPLIT = 120689   # <｜hy_place▁holder▁no▁671｜>
_TOK_HY_ASSISTANT = "<｜hy_Assistant｜>"


def _resize_with_pad(img, height=224, width=224, pad_value=-1.0, mode="bilinear"):
    """Pi0-style resize with aspect-preserving center pad. (B,C,H,W)."""
    ch, cw = img.shape[2:]
    if (ch, cw) == (height, width):
        return img
    ratio = max(cw / width, ch / height)
    rh, rw = int(ch / ratio), int(cw / ratio)
    resized = F.interpolate(img, size=(rh, rw), mode=mode, align_corners=False)
    ph, pw = max(0, height - rh), max(0, width - rw)
    t, l = ph // 2, pw // 2
    return F.pad(resized, (l, pw - l, t, ph - t), value=pad_value)


def _camera_tensor(image):
    t = torch.as_tensor(np.asarray(image))
    scale_uint8 = t.dtype == torch.uint8
    if t.ndim == 3:
        if t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        elif t.shape[0] != 3:
            raise ValueError(f"camera image must have 3 channels, got shape {tuple(t.shape)}")
        t = t.unsqueeze(0)
    elif t.ndim == 4:
        if t.shape[-1] == 3:
            t = t.permute(0, 3, 1, 2)
        elif t.shape[1] != 3:
            raise ValueError(f"camera frames must have 3 channels, got shape {tuple(t.shape)}")
    else:
        raise ValueError(f"camera image must be rank 3 or 4, got shape {tuple(t.shape)}")
    t = t.contiguous().float()
    if scale_uint8:
        t = t / 255.0
    return t


class HyVLATorchFrontendThor:
    #: Development override for the hardware gate (e.g. running the weight
    #: loader on a non-Thor box). Any non-empty value skips the capability
    #: probe; kernels still require the real hardware at runtime. Not a
    #: supported production path.
    _FORCE_ARCH_ENV = "FLASHRT_HYVLA_FORCE_ARCH"
    _REQUIRED_CAPABILITY = (11, 0)
    _ARCH_NAME = "Jetson Thor SM110"

    def _require_arch(self):
        import os

        if os.environ.get(self._FORCE_ARCH_ENV, ""):
            return  # explicit documented dev override: skip the probe
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"HyVLA frontend requires a CUDA device ({self._ARCH_NAME}); "
                "CUDA is not available.")
        cap = torch.cuda.get_device_capability()
        if cap != self._REQUIRED_CAPABILITY:
            raise RuntimeError(
                f"HyVLA frontend requires {self._ARCH_NAME} (capability "
                f"{self._REQUIRED_CAPABILITY}), found capability {cap}. Set "
                f"{self._FORCE_ARCH_ENV}=1 to bypass this check for "
                "development only.")

    def __init__(self, checkpoint_dir: str, *, hardware: str = "thor",
                 use_fp8: bool = False, use_fp8_vit: bool = False,
                 use_fused: bool = False, use_fp4: bool = False,
                 use_fused_quant: bool = False, use_autotune: bool = False,
                 use_ffn_mega: bool = False, **kwargs):
        self.checkpoint_dir = str(checkpoint_dir)
        self.device = "cuda"
        self._require_arch()
        self.use_fp8 = bool(use_fp8)
        self.use_fp8_vit = bool(use_fp8_vit)
        self.use_fused = bool(use_fused)
        self.use_fp4 = bool(use_fp4)
        self.use_fused_quant = bool(use_fused_quant)
        self.use_autotune = bool(use_autotune)
        self.use_ffn_mega = bool(use_ffn_mega)
        cfg_path = pathlib.Path(self.checkpoint_dir) / "config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.cfg = cfg
        self.num_steps = int(cfg.get("num_steps", 10))
        self.chunk = int(cfg.get("n_action_steps", cfg.get("chunk_size", 40)))
        self.max_action_dim = int(cfg.get("max_action_dim", 32))
        self.max_state_dim = int(cfg.get("max_state_dim", 32))
        self.tok_max_len = int(cfg.get("tokenizer_max_length", 64))
        self.proj_width = int(cfg.get("proj_width", 1024))
        # Camera key order (must match the prefix assembly order).
        self.image_keys = list(cfg.get("image_features", {}).keys())

        txt = cfg.get("vlm_config_dict", {}).get("text_config", {})
        self.rope_theta = float(txt.get("rope_theta", 10000.0))
        self.head_dim = int(txt.get("head_dim", 128))
        alpha = float(txt.get("rope_scaling", {}).get("alpha", 1000.0))
        self._rope_base = self.rope_theta * alpha ** (self.head_dim / (self.head_dim - 2))
        self.n_kv = int(txt.get("num_key_value_heads", 4))
        self.n_heads = int(txt.get("num_attention_heads", 16))
        self.d_vlm = int(txt.get("hidden_size", 2048))

        self._load_weights()
        self.pipe = HyVLAThorBF16Pipeline(self)
        self._graph_cache = {}          # (S_p, n_vis) -> {"graph", "buf"}
        self._vit_graph_cache = {}      # (num_cam, K) -> {"graph", "img", "out"}
        self._tokenizer = None
        self._prompt = None
        self._lang_tokens = None
        self._lang_masks = None
        self._precompute_time_embs()
        if self.use_fp8:
            self._quantize_fp8()
            self.pipe.enable_fp8()
            if self.use_fused_quant:
                # denoise expert tower runs at M=41; collapse the 4-node
                # quantize_fp8_device into the single-CTA fused quant there.
                self.pipe._small_quant_m = 64
            if self.use_ffn_mega:
                self._quantize_ffn_mega()
                self.pipe._ffn_mega = True
        if self.use_fused:
            self.pipe._fused_attn = True
        if self.use_fp4:
            self._quantize_fp4()
            self.pipe.enable_fp4()

    # ------------------------------------------------------------------
    def _load_weights(self):
        sf = pathlib.Path(self.checkpoint_dir) / "model.safetensors"
        src = SafetensorsSource(str(sf), device=self.device, strip_prefix="model.")
        WeightLoader(source=src, target=self, spec=build_spec()).run()

    def _quantize_fp8(self):
        """Quantize expert-tower AND VLM-tower (text+vision) GEMM weights to
        graph-safe FP8: fp8 (K,N) tensor + precomputed device fp32 per-tensor
        scale, consumed by pipeline._fp8_gemm (dynamic-activation FP8)."""
        import flash_rt.flash_rt_kernels as fvk
        st = torch.cuda.current_stream().cuda_stream

        def q(w_bf16):
            wkn = w_bf16.t().contiguous()               # (N,K) -> (K,N)
            K, N = wkn.shape
            w8 = torch.empty(K, N, dtype=torch.uint8, device=self.device)
            ws = torch.empty(1, dtype=torch.float32, device=self.device)
            fvk.quantize_fp8_device(wkn.data_ptr(), w8.data_ptr(), ws.data_ptr(), K * N, st)
            return w8, ws

        def q_list(src):
            w8s, wss = [], []
            for w in src:
                w8, ws = q(w)
                w8s.append(w8); wss.append(ws)
            return w8s, wss

        # Expert tower (uniform _v)
        self._exp_qkv8, self._exp_qkv_ws = q_list(self._exp_qkv_v)
        self._exp_o8, self._exp_o_ws = q_list(self._exp_o_v)
        self._exp_gu8, self._exp_gu_ws = q_list(self._exp_gu_v)
        self._exp_d8, self._exp_d_ws = q_list(self._exp_d_v)
        self._exp_fp8_ready = True

        # VLM tower (vision + text branches)
        self._vlm_qkv_v8, self._vlm_qkv_v_ws = q_list(self._vlm_qkv_v)
        self._vlm_o_v8, self._vlm_o_v_ws = q_list(self._vlm_o_v)
        self._vlm_gu_v8, self._vlm_gu_v_ws = q_list(self._vlm_gu_v)
        self._vlm_d_v8, self._vlm_d_v_ws = q_list(self._vlm_d_v)
        self._vlm_qkv_t8, self._vlm_qkv_t_ws = q_list(self._vlm_qkv_t)
        self._vlm_o_t8, self._vlm_o_t_ws = q_list(self._vlm_o_t)
        self._vlm_gu_t8, self._vlm_gu_t_ws = q_list(self._vlm_gu_t)
        self._vlm_d_t8, self._vlm_d_t_ws = q_list(self._vlm_d_t)
        self._vlm_fp8_ready = True

        # ViT tower (27 blocks) — measured NET LOSS on Thor (quant+bias passes
        # outweigh large-M GEMM savings; confirms the FP8-ViT dead-end).
        # Opt-in only.
        if self.use_fp8_vit:
            self._vit_qkv_w8, self._vit_qkv_ws = q_list(self._vit_qkv_w)
            self._vit_proj_w8, self._vit_proj_ws = q_list(self._vit_proj_w)
            self._vit_fc1_w8, self._vit_fc1_ws = q_list(self._vit_fc1_w)
            self._vit_fc2_w8, self._vit_fc2_ws = q_list(self._vit_fc2_w)
            self._vit_fp8_ready = True
        torch.cuda.synchronize()

    def _quantize_ffn_mega(self):
        """Quantize the expert-tower FFN weights (gu, dn) to FP8 in (N,K) layout
        for the denoise FFN megakernel (hyvla_ffn_gu_silu_bf16 / _dn_res_bf16).

        The megakernel reads weight rows K-contiguous (N,K) — the ORIGINAL
        orientation — unlike _fp8_gemm's (K,N). Weight scale is read to a host
        float (constant across forwards); the activation scale stays dynamic
        (device pointer). Done once at load (the .item() sync is fine here)."""
        import flash_rt.flash_rt_kernels as fvk
        st = torch.cuda.current_stream().cuda_stream

        def q_nk(w_bf16):  # (N,K) bf16 -> (N,K) fp8 uint8 + host float scale
            N, K = w_bf16.shape
            wc = w_bf16.contiguous()
            w8 = torch.empty(N, K, dtype=torch.uint8, device=self.device)
            ws = torch.empty(1, dtype=torch.float32, device=self.device)
            fvk.quantize_fp8_device(wc.data_ptr(), w8.data_ptr(), ws.data_ptr(), N * K, st)
            return w8, float(ws.item())

        gu8, gus, dn8, dns = [], [], [], []
        for w in self._exp_gu_v:
            a, b = q_nk(w); gu8.append(a); gus.append(b)
        for w in self._exp_d_v:
            a, b = q_nk(w); dn8.append(a); dns.append(b)
        self._exp_gu_mk, self._exp_gu_mk_s = gu8, gus
        self._exp_d_mk, self._exp_d_mk_s = dn8, dns
        self._exp_inter = self._exp_gu_v[0].shape[0] // 2   # 2*inter -> inter
        self._exp_ffn_mega_ready = True
        torch.cuda.synchronize()

    def _quantize_fp4(self):
        """Quantize VLM prefill FFN weights (gu, down; text + vision branches)
        to NVFP4 via the flash_rt_fp4 family (packed 4-bit + swizzled UE4M3 SF),
        consumed by cutlass_fp4_sq_fp16. Prefill runs at M=240 where FP4 wins
        (gu 2.67x) and once (no Euler compounding), so it is the right FP4 target.
        Activation quant uses the SAME F4 family (see pipeline._fp4_gemm_f4) —
        mixing F4 weights with fvk fused-quant SF is the swizzle-mismatch trap."""
        import flash_rt.flash_rt_fp4 as F4
        st = torch.cuda.current_stream().cuda_stream

        def q_nvfp4(w_bf16):
            N, K = w_bf16.shape
            if K % 64 != 0:
                raise ValueError(
                    f"cutlass_fp4_sq_fp16 needs K%64==0, got K={K}")
            w16 = w_bf16.to(torch.float16).contiguous()
            packed = torch.empty(N, K // 2, dtype=torch.uint8, device=self.device)
            sf = torch.empty(F4.sfa_size_bytes(N, K, True), dtype=torch.uint8, device=self.device)
            F4.quantize_fp4_dynamic_sfa_fp16(w16.data_ptr(), packed.data_ptr(),
                                             sf.data_ptr(), N, K, True, st)
            return packed, sf

        def q_list(src):
            ps, ss = [], []
            for w in src:
                p, s = q_nvfp4(w); ps.append(p); ss.append(s)
            return ps, ss

        self._vlm_gu_v4, self._vlm_gu_v4sf = q_list(self._vlm_gu_v)
        self._vlm_d_v4, self._vlm_d_v4sf = q_list(self._vlm_d_v)
        self._vlm_gu_t4, self._vlm_gu_t4sf = q_list(self._vlm_gu_t)
        self._vlm_d_t4, self._vlm_d_t4sf = q_list(self._vlm_d_t)
        self._vlm_gu_N = self._vlm_gu_v[0].shape[0]     # 2*inter (gate+up merged)
        self._vlm_D = self._vlm_gu_v[0].shape[1]        # VLM hidden (gu K)
        self._vlm_inter = self._vlm_d_v[0].shape[1]     # inter (down K)
        self._vlm_fp4_ready = True
        torch.cuda.synchronize()

    def _precompute_time_embs(self):
        embs = []
        for s in range(self.num_steps):
            t = 1.0 + s * (-1.0 / self.num_steps)
            embs.append(self._sinusoidal_time(t))
        self._time_embs = torch.stack(embs).to(self.device, _BF16)  # (steps,1,proj_width)

    def _sinusoidal_time(self, t, min_period=4e-3, max_period=4.0):
        dim = self.proj_width
        frac = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
        period = min_period * (max_period / min_period) ** frac
        scaling = 1.0 / period * 2 * math.pi
        tt = torch.tensor([t], dtype=torch.float64)
        sin_in = scaling[None, :] * tt[:, None]
        return torch.cat([torch.sin(sin_in), torch.cos(sin_in)], dim=1)  # (1,dim)

    # ------------------------------------------------------------------
    def _rope_cos_sin(self, positions):
        """positions (1,S) long -> cos,sin (1,1,S,head_dim) bf16 (rotate_half).

        The original model is cast to bf16, so its ``rotary_emb.inv_freq``
        buffer is bf16-quantized; that rounding compounds over positions, so
        we must round our inv_freq through bf16 to match the reference tables
        exactly.
        """
        hd = self.head_dim
        inv_freq = 1.0 / (self._rope_base ** (
            torch.arange(0, hd, 2, dtype=torch.float64, device=self.device) / hd))
        inv_freq = inv_freq.to(_BF16).to(torch.float64)   # match reference bf16 storage
        freqs = positions.to(torch.float64)[..., None] * inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)              # (1,S,hd)
        return emb.cos()[:, None].to(_BF16), emb.sin()[:, None].to(_BF16)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            # Standard tokenizer path only — never execute checkpoint-provided
            # Python code (no trust_remote_code).
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint_dir)
        return self._tokenizer

    def _tokenize(self, prompt):
        task = prompt.strip().replace("_", " ").replace("\n", " ")
        if not task.endswith(_TOK_HY_ASSISTANT):
            task = f"{task}{_TOK_HY_ASSISTANT}"
        out = self.tokenizer([task], padding="max_length", padding_side="right",
                             truncation=True, max_length=self.tok_max_len,
                             return_tensors="pt", add_special_tokens=False)
        return (out["input_ids"].to(self.device),
                out["attention_mask"].to(self.device).bool())

    def set_prompt(self, prompt, state=None):
        self._prompt = prompt
        self._lang_tokens, self._lang_masks = self._tokenize(prompt)

    # ------------------------------------------------------------------
    def _embed_ids(self, ids):
        return F.embedding(ids, self._embed_weight)

    def _preprocess_images(self, images):
        """Normalize public inputs to (num_cam,K,3,H,W) in [0,1]."""
        if isinstance(images, dict):
            keys = [k for k in self.image_keys if k in images]
            if not keys:
                keys = [k for k in ("image", "wrist_image", "wrist_image_right") if k in images]
            if not keys:
                raise ValueError("images dict does not contain configured camera keys")
            images = [images[k] for k in keys]

        if isinstance(images, (list, tuple)):
            if not images:
                raise ValueError("images list must have at least one camera")
            images = torch.stack([_camera_tensor(im) for im in images], 0)
        else:
            images = torch.as_tensor(np.asarray(images))
            scale_uint8 = images.dtype == torch.uint8
            if images.ndim == 5:
                if images.shape[-1] == 3:
                    images = images.permute(0, 1, 4, 2, 3)
                elif images.shape[2] != 3:
                    raise ValueError(
                        f"images must have channel dimension of size 3, got {tuple(images.shape)}")
            elif images.ndim == 4:
                images = torch.stack([_camera_tensor(im) for im in images], 0)
            else:
                raise ValueError(
                    f"images must be list/dict or rank 4/5 tensor, got {tuple(images.shape)}")
            images = images.contiguous().float()
            if scale_uint8:
                images = images / 255.0

        out = []
        for cam in range(images.shape[0]):
            im = images[cam].to(self.device, _BF16)
            im = _resize_with_pad(im, 224, 224, pad_value=0.0)
            im = im * 2.0 - 1.0
            out.append(im[None])
        return out

    @torch.no_grad()
    def _assemble_prefix(self, merged):
        """merged: (num_cam, 49, 2048) merged vision tokens. Returns prefix tensors."""
        dev = self.device

        embs = [self._embed_ids(torch.tensor([[_TOK_BOS]], device=dev)),
                self._embed_ids(torch.tensor([[_TOK_HY_USER]], device=dev))]
        att = [1, 1]
        mm = [False, False]
        pad = [torch.ones((1, 2), dtype=torch.bool, device=dev)]
        idx_ranges, full_ranges = [], []

        vstart = self._embed_ids(torch.tensor([[_TOK_VISION_START]], device=dev))
        vend = self._embed_ids(torch.tensor([[_TOK_VISION_END]], device=dev))
        vsplit = self._embed_ids(torch.tensor([[_TOK_VISION_SPLIT]], device=dev))

        for ci in range(merged.shape[0]):
            img_emb = merged[ci][None]                        # (1,49,2048)
            g = int(img_emb.shape[1] ** 0.5)                  # 7
            embs.append(vstart); att.append(1); mm.append(False)
            pad.append(torch.ones((1, 1), dtype=torch.bool, device=dev))
            grid = img_emb.view(1, g, g, -1)
            split_exp = vsplit.unsqueeze(1).expand(1, g, 1, grid.shape[-1])
            with_split = torch.cat([grid, split_exp], dim=2).reshape(1, -1, grid.shape[-1])
            embs.append(with_split)
            row_len = g + 1
            total = g * row_len
            start = len(att)
            idx_ranges.extend([(start + r * row_len, start + r * row_len + g) for r in range(g)])
            full_ranges.append((start, start + total))
            att.extend([1] * total)
            mm.extend(([True] * g + [False]) * g)
            pad.append(torch.ones((1, total), dtype=torch.bool, device=dev))
            embs.append(vend); att.append(1); mm.append(False)
            pad.append(torch.ones((1, 1), dtype=torch.bool, device=dev))

        lang_emb = self._embed_ids(self._lang_tokens)         # (1,64,2048)
        embs.append(lang_emb)
        pad.append(self._lang_masks)
        n_lang = lang_emb.shape[1]
        att.extend([1] * n_lang)
        mm.extend([False] * n_lang)

        prefix_embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad, dim=1).bool()
        att_masks = torch.tensor(att, dtype=torch.bool, device=dev)[None]
        mm_prefix = torch.tensor(mm, dtype=torch.bool, device=dev)[None]
        return prefix_embs, pad_masks, att_masks, mm_prefix, idx_ranges, full_ranges

    @staticmethod
    def _make_att_2d(pad_masks, att_masks):
        cumsum = torch.cumsum(att_masks.long(), dim=1)
        att2d = cumsum[:, None, :] <= cumsum[:, :, None]
        pad2d = pad_masks[:, None, :] * pad_masks[:, :, None]
        return att2d & pad2d

    def _apply_segment_mask(self, att2d, idx_ranges, full_ranges):
        dev = att2d.device
        all_idx = [i for (s, e) in idx_ranges for i in range(s, e)]
        if all_idx:
            idx = torch.tensor(all_idx, device=dev)
            att2d[:, idx[:, None], idx[None, :]] = False
        for fs, fe in full_ranges:
            img_idx = [i for (s, e) in idx_ranges if s >= fs and e <= fe for i in range(s, e)]
            if img_idx:
                idx = torch.tensor(img_idx, device=dev)
                att2d[:, idx[:, None], idx[None, :]] = True
        return att2d

    # ------------------------------------------------------------------
    #  ViT + merger CUDA-graph (BF16, unchanged precision). Keyed on
    #  (num_cam, K); removes per-block launch overhead of the 27-block ViT.
    # ------------------------------------------------------------------
    def _vit_merge(self, imgs5, use_graph=True):
        """imgs5 (num_cam,K,3,224,224) bf16 [-1,1] -> merged (num_cam,49,2048)."""
        if not use_graph:
            return self.pipe.merger_forward(self.pipe.vit_forward(imgs5))
        key = tuple(imgs5.shape[:2])
        g = self._vit_graph_cache.get(key)
        if g is None:
            g = self._build_vit_graph(imgs5.shape)
            self._vit_graph_cache[key] = g
        g["img"].copy_(imgs5)
        g["graph"].replay()
        return g["out"].clone()

    def _build_vit_graph(self, shape):
        dev = self.device
        img = torch.zeros(shape, dtype=_BF16, device=dev)
        out = self.pipe.merger_forward(self.pipe.vit_forward(img)).clone()  # shape probe

        def body():
            out.copy_(self.pipe.merger_forward(self.pipe.vit_forward(img)))

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        return {"graph": graph, "img": img, "out": out}

    # ------------------------------------------------------------------
    #  CUDA-graph capture of prefill + 10-step denoise (BF16, unchanged
    #  precision). Keyed on (S_p, n_vis); ViT + prefix assembly stay eager
    #  and feed the static input buffers before each replay.
    # ------------------------------------------------------------------
    def _captured_body(self, b, n_vis, S_p):
        self.pipe.prefill(b["pe"], n_vis, b["pmask"], b["pcos"], b["psin"],
                          b["kbuf"], b["vbuf"])
        self.pipe.denoise(b["state"], b["x"], self._time_embs,
                          b["smask"], b["scos"], b["ssin"],
                          b["kbuf"], b["vbuf"], S_p, num_steps=self.num_steps)

    def _build_graph(self, key):
        S_p, n_vis = key
        dev = self.device
        L, nkv, hd = 32, self.n_kv, self.head_dim
        # KV cache is stored PRE-EXPANDED to all query heads: the megakernel
        # replicates each KV head kv_rep times, so attention reads it directly
        # and skips the per-call repeat_interleave over the whole cache
        # (measured ~36us/call x 640 calls on the 281-row cache).
        n_kvc = self.n_heads
        D, S_s = self.d_vlm, 1 + self.chunk
        z = lambda *s, dt=_BF16: torch.zeros(*s, dtype=dt, device=dev)
        b = {
            "pe": z(1, S_p, D),
            "pmask": z(1, 1, S_p, S_p, dt=torch.bool),
            "pcos": z(1, 1, S_p, hd), "psin": z(1, 1, S_p, hd),
            "smask": z(1, 1, S_s, S_p + S_s, dt=torch.bool),
            "scos": z(1, 1, S_s, hd), "ssin": z(1, 1, S_s, hd),
            "state": z(1, self.max_state_dim),
            "x": z(1, self.chunk, self.max_action_dim, dt=torch.float32),
            "kbuf": z(L, 1, n_kvc, S_p + S_s, hd),
            "vbuf": z(L, 1, n_kvc, S_p + S_s, hd),
        }
        # Per-shape FP8 GEMM autotune BEFORE capture (graph-safe: mutates only
        # the GemmRunner algo cache). A dry eager body records the exact (M,N,K)
        # set the captured path will hit, then we tune each on self.pipe.gemm.
        if self.use_autotune and self.use_fp8:
            self.pipe._gemm_shapes = set()
            self._captured_body(b, n_vis, S_p)
            shapes = self.pipe._gemm_shapes
            self.pipe._gemm_shapes = None
            self.pipe.autotune_gemms(shapes)
            torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._captured_body(b, n_vis, S_p)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._captured_body(b, n_vis, S_p)
        return {"graph": graph, "buf": b}

    def _graph_forward(self, S_p, n_vis, prefix_embs, pmask, pcos, psin,
                       smask, scos, ssin, state_t, noise_t, use_graph=True):
        dev = self.device
        if not use_graph:
            L, nkv, hd = 32, self.n_kv, self.head_dim
            kbuf = torch.zeros(L, 1, self.n_heads, S_p + 1 + self.chunk, hd, dtype=_BF16, device=dev)
            vbuf = torch.zeros_like(kbuf)
            x = noise_t.clone().float()
            self.pipe.prefill(prefix_embs.clone(), n_vis, pmask, pcos, psin, kbuf, vbuf)
            return self.pipe.denoise(state_t, x, self._time_embs, smask, scos, ssin,
                                     kbuf, vbuf, S_p, num_steps=self.num_steps)
        g = self._graph_cache.get((S_p, n_vis))
        if g is None:
            g = self._build_graph((S_p, n_vis))
            self._graph_cache[(S_p, n_vis)] = g
        b = g["buf"]
        b["pe"].copy_(prefix_embs); b["pmask"].copy_(pmask)
        b["pcos"].copy_(pcos); b["psin"].copy_(psin)
        b["smask"].copy_(smask); b["scos"].copy_(scos); b["ssin"].copy_(ssin)
        b["state"].copy_(state_t); b["x"].copy_(noise_t)
        g["graph"].replay()
        return b["x"].clone()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_actions(self, images, prompt=None, state=None, noise=None,
                        use_graph=True):
        """Full native forward. images: (num_cam,K,3,H,W) [0,1] array/tensor.
        Returns raw action chunk (1, chunk, max_action_dim) as numpy fp32."""
        if prompt is not None and prompt != self._prompt:
            self.set_prompt(prompt)
        if self._lang_tokens is None:
            raise RuntimeError("call set_prompt() before predict_actions()")
        dev = self.device

        if not torch.is_tensor(images):
            images = torch.as_tensor(np.asarray(images))
        cam_imgs = self._preprocess_images(images)
        imgs5 = torch.cat(cam_imgs, 0)                       # (num_cam,K,3,224,224)
        merged = self._vit_merge(imgs5, use_graph=use_graph)

        # state -> (1, max_state_dim)
        if state is None:
            state_t = torch.zeros(1, self.max_state_dim, device=dev, dtype=_BF16)
        else:
            st = torch.as_tensor(np.asarray(state), device=dev, dtype=_BF16).reshape(1, -1)
            if st.shape[1] > self.max_state_dim:
                raise ValueError(
                    f"state has {st.shape[1]} dims, max_state_dim is "
                    f"{self.max_state_dim}")
            if st.shape[1] < self.max_state_dim:
                st = F.pad(st, (0, self.max_state_dim - st.shape[1]))
            state_t = st

        (prefix_embs, pad_masks, att_masks, mm_prefix,
         idx_ranges, full_ranges) = self._assemble_prefix(merged)
        att2d = self._make_att_2d(pad_masks, att_masks)
        att2d = self._apply_segment_mask(att2d, idx_ranges, full_ranges)
        prefix_pos = torch.cumsum(pad_masks.long(), dim=1) - 1

        mm = mm_prefix[0]
        perm = torch.cat([torch.nonzero(mm).squeeze(-1), torch.nonzero(~mm).squeeze(-1)])
        n_vis = int(mm.sum().item())
        prefix_embs = prefix_embs[:, perm]
        att2d = att2d[:, perm][:, :, perm]
        prefix_pos = prefix_pos[:, perm]

        pcos, psin = self._rope_cos_sin(prefix_pos)
        pmask = att2d[:, None]

        S_p = pad_masks.shape[1]
        S_s = 1 + self.chunk
        suffix_pad = torch.ones(1, S_s, dtype=torch.bool, device=dev)
        suffix_att = torch.tensor([1, 1] + [0] * (self.chunk - 1),
                                  dtype=torch.bool, device=dev)[None]
        suffix_att2d = self._make_att_2d(suffix_pad, suffix_att)
        prefix_pad_2d = pad_masks[:, perm][:, None, :].expand(1, S_s, S_p)
        smask = torch.cat([prefix_pad_2d, suffix_att2d], 2)[:, None]
        suffix_pos = pad_masks.long().sum(-1)[:, None] + torch.cumsum(suffix_pad.long(), 1) - 1
        scos, ssin = self._rope_cos_sin(suffix_pos)

        if noise is None:
            noise_t = torch.randn(1, self.chunk, self.max_action_dim,
                                  dtype=torch.float32, device=dev)
        else:
            noise_t = torch.as_tensor(np.asarray(noise), device=dev, dtype=torch.float32)
            want = self.chunk * self.max_action_dim
            if noise_t.numel() != want:
                raise ValueError(
                    f"noise must have {want} elements "
                    f"(chunk={self.chunk} x max_action_dim={self.max_action_dim}), "
                    f"got {noise_t.numel()}")
            noise_t = noise_t.reshape(1, self.chunk, self.max_action_dim)

        x_t = self._graph_forward(S_p, n_vis, prefix_embs, pmask, pcos, psin,
                                  smask, scos, ssin, state_t, noise_t,
                                  use_graph=use_graph)
        return x_t.float().cpu().numpy()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def infer(self, obs):
        images = obs.get("images")
        if images is None:
            images = {k: obs[k] for k in self.image_keys if k in obs}
            if not images:
                images = [obs[k] for k in ("image", "wrist_image", "wrist_image_right") if k in obs]
        state = obs.get("state")
        noise = obs.get("noise")
        actions = self.predict_actions(images, prompt=obs.get("prompt"),
                                       state=state, noise=noise)
        return {"actions": actions[0]}


__all__ = ["HyVLATorchFrontendThor"]
