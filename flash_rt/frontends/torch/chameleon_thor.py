"""Standalone Chameleon-7B frontend for Jetson Thor (SM110).

Direct-use VLM/LLM frontend: text + real images -> Chameleon prefill logits
and incremental KV-cache greedy generation. It uses the Chameleon
Thor dynamic-FP8 pipeline and causal FMHA attention backend.
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import pathlib
from typing import List, Optional

import numpy as np
import PIL
from PIL import Image as _PILImage
import torch

import flash_rt.flash_rt_kernels as fvk
try:
    import flash_rt.flash_rt_fp4 as fvk_fp4
except Exception:
    fvk_fp4 = None
from flash_rt.hardware.thor.attn_backend_chameleon import (
    ThorChameleonAttnBackend,
    make_chameleon_attention_spec,
)
from flash_rt.models.chameleon.pipeline_thor import (
    chameleon_decode_step,
    chameleon_forward,
    chameleon_forward_calibrate,
    chameleon_forward_fp16,
)

logger = logging.getLogger(__name__)

fp16 = torch.float16
fp8 = torch.float8_e4m3fn
_cudart = ctypes.CDLL("libcudart.so")

D_LLM = 4096
NH_LLM = 32
HD_LLM = 128
L_LLM = 32
DFF_LLM = 11008
VOCAB_SIZE = 65536
ROPE_THETA = 10000.0

# Chameleon image/text special ids from the shipped vocabulary_map.
PAD_ID = 1
EOS_ID = 2
IMG_START_ID = 8197      # <racm3:break>
IMG_END_ID = 8196        # <eoss>
NEWLINE_ID = 8803
GRID_TOK_BASE = 8804
PATCH_SIZE = 32


class ChameleonTorchFrontendThor:
    """Standalone Chameleon-7B Thor prefill frontend."""

    #: Required CUDA capability (Jetson Thor SM110) and the documented
    #: dev override that skips the probe. Mirrors the Orin frontend's gate.
    _REQUIRED_CAPABILITY = (11, 0)
    _FORCE_ARCH_ENV = "FLASHRT_CHAMELEON_THOR_FORCE"

    def _require_arch(self) -> None:
        if os.environ.get(self._FORCE_ARCH_ENV) == "1":
            return  # explicit documented dev override: skip the probe
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ChameleonTorchFrontendThor requires a Jetson Thor SM110 CUDA "
                "device; CUDA is not available.")
        cc = torch.cuda.get_device_capability(0)
        if cc != self._REQUIRED_CAPABILITY:
            raise RuntimeError(
                f"ChameleonTorchFrontendThor targets SM110 (Thor); found "
                f"SM{cc[0]}{cc[1]}. The FP8 path needs the Thor kernel set. "
                f"Set {self._FORCE_ARCH_ENV}=1 to override for development.")

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        use_fp8: bool = True,
        use_cuda_graph: bool = True,
        max_seq: int = 4096,
        target_size: int = 512,
        tokenizer_path: Optional[str] = None,
        vqgan_path: Optional[str] = None,
        use_trt_vqgan: bool = False,
        trt_vqgan_engine_dir: Optional[str] = None,
        use_autotune: bool = True,
        ffn_clamp_layers: Optional[List[int]] = None,
        fp4_ffn_layers: Optional[List[int]] = None,
        use_fa4_attn: Optional[bool] = None,
    ) -> None:
        self._require_arch()
        self.checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser().resolve()
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"checkpoint_dir not found: {self.checkpoint_dir}")
        self._use_fp8 = bool(use_fp8)
        self._use_cuda_graph = bool(use_cuda_graph)
        self._max_pos = int(max_seq)
        self.target_size = int(target_size)
        self.tokenizer_path = tokenizer_path
        self.vqgan_path = vqgan_path
        self._use_trt_vqgan = bool(use_trt_vqgan)
        self._trt_vqgan_engine_dir = trt_vqgan_engine_dir
        self._trt_vqgan_backend = None
        self._vqgan_backend = "eager"
        self._trt_stream = torch.cuda.Stream() if self._use_trt_vqgan else None
        self._infer_graph = None
        self._captured_Se = None
        self._last_input_ids: Optional[list[int]] = None
        self.Se: Optional[int] = None
        self._real_len: int = 0
        self._use_autotune = bool(use_autotune)
        self._autotuned_se: set = set()
        self._ffn_clamp_layers = self._resolve_ffn_clamp_layers(ffn_clamp_layers)
        self._fp4_ffn_layers = self._resolve_fp4_ffn_layers(fp4_ffn_layers)
        if use_fa4_attn is None:
            use_fa4_attn = os.environ.get("FLASHRT_CHAMELEON_FA4_ATTN", "0") in ("1", "true", "on")
        self._use_fa4_attn = bool(use_fa4_attn)
        self._stream = torch.cuda.current_stream().cuda_stream

        with open(self.checkpoint_dir / "config.json") as f:
            self.config_json = json.load(f)
        self._validate_config()
        self._build_image_token_mask()
        with open(self.checkpoint_dir / "model.safetensors.index.json") as f:
            self.weight_index = json.load(f)

        self._load_weights()
        self._build_rope_tables()
        self._load_tokenizer()
        self._load_vqgan()
        self._allocate_buffers()

        from flash_rt.core.context import FvkContext
        self._ctx = FvkContext()
        self._gemm = self._ctx.gemm
        self._build_attention_backend()
        if self._use_fa4_attn:
            self._attn.set_fa4_attn(self._bufs["xn"], self._kv_cache)
        self._logits_buf.zero_()

        logger.info(
            "ChameleonTorchFrontendThor init: max_seq=%d target_size=%d fp8=%s graph=%s",
            self._Se_max, self.target_size, self._use_fp8, self._use_cuda_graph,
        )

    def _resolve_ffn_clamp_layers(self, layers):
        spec = os.environ.get("FLASHRT_CHAMELEON_FFN_CLAMP_LAYERS") \
            if layers is None else layers
        if spec is None:
            return frozenset({31})
        if isinstance(spec, str):
            text = spec.strip().lower()
            if text == "all":
                return None
            if text in ("", "none", "off", "false"):
                return frozenset()
            out = set()
            for chunk in text.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if "-" in chunk:
                    a, b = chunk.split("-", 1)
                    out.update(range(int(a), int(b) + 1))
                else:
                    out.add(int(chunk))
            return frozenset(i for i in out if 0 <= i < L_LLM)
        return frozenset(int(i) for i in spec if 0 <= int(i) < L_LLM)

    def _resolve_fp4_ffn_layers(self, layers):
        spec = os.environ.get("FLASHRT_CHAMELEON_FP4_LAYERS") \
            if layers is None else layers
        if spec is None:
            return frozenset()
        parsed = self._parse_layer_spec(spec)
        return frozenset() if parsed is None else parsed

    def _parse_layer_spec(self, spec):
        if isinstance(spec, str):
            text = spec.strip().lower()
            if text == "all":
                return frozenset(range(L_LLM))
            if text in ("", "none", "off", "false"):
                return frozenset()
            out = set()
            for chunk in text.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if "-" in chunk:
                    a, b = chunk.split("-", 1)
                    out.update(range(int(a), int(b) + 1))
                else:
                    out.add(int(chunk))
            return frozenset(i for i in out if 0 <= i < L_LLM)
        return frozenset(int(i) for i in spec if 0 <= int(i) < L_LLM)

    def _validate_config(self) -> None:
        c = self.config_json
        for k, want in (("hidden_size", D_LLM),
                        ("num_attention_heads", NH_LLM),
                        ("num_hidden_layers", L_LLM),
                        ("intermediate_size", DFF_LLM),
                        ("vocab_size", VOCAB_SIZE)):
            if int(c[k]) != want:
                raise ValueError(f"config {k}={c[k]}, expected {want}")
        if bool(c.get("attention_bias", False)):
            raise ValueError("standard Chameleon Thor path expects attention_bias=false")
        if bool(c.get("mlp_bias", False)):
            raise ValueError("standard Chameleon Thor path expects mlp_bias=false")

    def _build_image_token_mask(self) -> None:
        """Image-codebook vocab ids to suppress for text generation.

        Mirrors HF ``ChameleonForConditionalGeneration``'s
        ``mask_image_logits``: without this, greedy decode on a text
        prompt can emit VQGAN codebook ids (garbage BPE decode) because
        those ids are heavily represented in training and the raw
        logit distribution favors them for many contexts.
        """
        vocab_map = self.config_json.get("vocabulary_map", {})
        image_tokens = sorted(
            v for k, v in vocab_map.items() if k.startswith("IMGIMG"))
        self._mask_image_logits = bool(self.config_json.get("mask_image_logits", False))
        if image_tokens:
            self._image_token_ids = torch.tensor(
                image_tokens, dtype=torch.long, device="cuda")
        else:
            self._image_token_ids = None
            self._mask_image_logits = False

    def _load_weights(self) -> None:
        from flash_rt.executors.torch_weights import MultiSafetensorsSource, WeightLoader
        from flash_rt.frontends.torch._chameleon_thor_spec import build_spec

        shard_paths = sorted(self.checkpoint_dir.glob("model-*-of-*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(f"No safetensors shards in {self.checkpoint_dir}")
        src = MultiSafetensorsSource([str(p) for p in shard_paths], device="cuda")
        WeightLoader(source=src, target=self, spec=build_spec(use_fp8=self._use_fp8)).run()

        self._split_fused_llm_weights()
        self._lm_head_w_t = self._llm_lm_head_w.t().contiguous()
        if self._use_fp8:
            self._setup_fp8_weight_scales()
        self._init_fp4_weight_lists()
        if self._use_fp8 and self._fp4_ffn_layers:
            self._quantize_fp4_weights()

    def _split_fused_llm_weights(self) -> None:
        self._q_w, self._k_w, self._v_w = [], [], []
        self._gate_w, self._up_w = [], []
        for li in range(L_LLM):
            qkv = self._llm_qkv_w[li]
            self._q_w.append(qkv[:, :D_LLM].contiguous())
            self._k_w.append(qkv[:, D_LLM:2 * D_LLM].contiguous())
            self._v_w.append(qkv[:, 2 * D_LLM:].contiguous())

            gu = self._llm_gu_w[li]
            self._gate_w.append(gu[:, :DFF_LLM].contiguous())
            self._up_w.append(gu[:, DFF_LLM:].contiguous())

            self._llm_q_norm_w[li] = self._llm_q_norm_w[li].reshape(-1).contiguous()
            self._llm_q_norm_b[li] = self._llm_q_norm_b[li].reshape(-1).contiguous()
            self._llm_k_norm_w[li] = self._llm_k_norm_w[li].reshape(-1).contiguous()
            self._llm_k_norm_b[li] = self._llm_k_norm_b[li].reshape(-1).contiguous()

    def _setup_fp8_weight_scales(self) -> None:
        self._llm_w_dev = torch.tensor(self._llm_w_scales, dtype=torch.float32, device="cuda")
        base = self._llm_w_dev.data_ptr()
        self._d_w_qkv_ptrs = [(base + (li * 4 + 0) * 4) for li in range(L_LLM)]
        self._d_w_o_ptrs = [(base + (li * 4 + 1) * 4) for li in range(L_LLM)]
        self._d_w_gu_ptrs = [(base + (li * 4 + 2) * 4) for li in range(L_LLM)]
        self._d_w_d_ptrs = [(base + (li * 4 + 3) * 4) for li in range(L_LLM)]

    def _init_fp4_weight_lists(self) -> None:
        self._gu_w_fp4 = [None] * L_LLM
        self._gu_sfb = [None] * L_LLM
        self._d_w_fp4 = [None] * L_LLM
        self._d_sfb = [None] * L_LLM

    def _quantize_fp4_weights(self) -> None:
        if fvk_fp4 is None:
            logger.warning("flash_rt_fp4 unavailable; FP4 FFN disabled")
            self._fp4_ffn_layers = frozenset()
            return
        try:
            if not fvk_fp4.has_nvfp4():
                logger.warning("NVFP4 unavailable on this device; FP4 FFN disabled")
                self._fp4_ffn_layers = frozenset()
                return
        except AttributeError:
            pass

        from flash_rt.executors.torch_weights import MultiSafetensorsSource

        shard_paths = sorted(self.checkpoint_dir.glob("model-*-of-*.safetensors"))
        src = MultiSafetensorsSource([str(p) for p in shard_paths], device="cuda")

        def _quant(w_fp16: torch.Tensor):
            N, K = w_fp16.shape
            packed = torch.empty(N, K // 2, dtype=torch.uint8, device="cuda")
            sfb_bytes = fvk_fp4.sfa_size_bytes(N, K, True)
            sfb = torch.empty(sfb_bytes, dtype=torch.uint8, device="cuda")
            rc = fvk_fp4.quantize_fp4_dynamic_sfa_fp16(
                w_fp16.data_ptr(), packed.data_ptr(), sfb.data_ptr(),
                N, K, True, 0)
            if rc != 0:
                raise RuntimeError(f"quantize_fp4_dynamic_sfa_fp16 failed rc={rc}")
            return packed, sfb

        packed_layers = set()
        for li in range(L_LLM):
            if li not in self._fp4_ffn_layers:
                continue
            gate_key = f"model.layers.{li}.mlp.gate_proj.weight"
            up_key = f"model.layers.{li}.mlp.up_proj.weight"
            down_key = f"model.layers.{li}.mlp.down_proj.weight"
            gate_w = src.get(gate_key).to(fp16).contiguous()
            up_w = src.get(up_key).to(fp16).contiguous()
            down_w = src.get(down_key).to(fp16).contiguous()
            gu_w = torch.cat([gate_w, up_w], dim=0).contiguous()
            self._gu_w_fp4[li], self._gu_sfb[li] = _quant(gu_w)
            self._d_w_fp4[li], self._d_sfb[li] = _quant(down_w)
            packed_layers.add(li)
            del gate_w, up_w, down_w, gu_w
        torch.cuda.synchronize()
        self._fp4_ffn_layers = frozenset(packed_layers)
        logger.info("FP4 FFN weights quantized for standalone Chameleon layers: %s",
                    sorted(self._fp4_ffn_layers))

    def _build_rope_tables(self) -> None:
        inv_freq = 1.0 / (ROPE_THETA ** (
            torch.arange(0, HD_LLM, 2, dtype=torch.float32) / HD_LLM))
        pos = torch.arange(self._max_pos, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._rope_cos = emb.cos().to(fp16).cuda().contiguous()
        self._rope_sin = emb.sin().to(fp16).cuda().contiguous()

    def _load_tokenizer(self) -> None:
        from transformers import AutoTokenizer

        candidates: list[pathlib.Path] = []
        if self.tokenizer_path:
            candidates.append(pathlib.Path(self.tokenizer_path))
        env = os.environ.get("FLASHRT_CHAMELEON_TOKENIZER_DIR")
        if env:
            candidates.append(pathlib.Path(env))
        candidates.append(self.checkpoint_dir)

        for p in candidates:
            if p.exists() and (p / "tokenizer.json").exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(p), use_fast=True)
                self.tokenizer_dir = p
                logger.info("Tokenizer loaded from %s", p)
                return
        raise FileNotFoundError("Chameleon tokenizer not found")

    def _load_vqgan(self) -> None:
        roots: list[pathlib.Path] = []
        if self.vqgan_path:
            roots.append(pathlib.Path(self.vqgan_path))
        env = os.environ.get("FLASHRT_VQGAN_DIR")
        if env:
            roots.append(pathlib.Path(env))
        roots.extend([
            self.checkpoint_dir / "original_tokenizers",
            self.checkpoint_dir,
        ])

        yaml_path = ckpt_path = None
        for root in roots:
            for d in (root, root / "original_tokenizers", root / "tokenizer"):
                if (d / "vqgan.yaml").exists() and (d / "vqgan.ckpt").exists():
                    yaml_path = d / "vqgan.yaml"
                    ckpt_path = d / "vqgan.ckpt"
                    break
            if yaml_path is not None:
                break
        if yaml_path is None or ckpt_path is None:
            raise FileNotFoundError("Chameleon VQGAN assets not found")

        # Chameleon-7B VQGAN reference implementation (Meta Chameleon
        # license), vendored into the repo at flash_rt/models/chameleon/vqgan.
        from flash_rt.models.chameleon import vqgan as chameleon_vae_ori  # type: ignore

        self._vqgan_tokenizer = chameleon_vae_ori.ImageTokenizer(
            cfg_path=str(yaml_path), ckpt_path=str(ckpt_path), device="cuda")

        text_tok = pathlib.Path(yaml_path).parent / "text_tokenizer.json"
        if not text_tok.exists():
            text_tok = self.checkpoint_dir / "original_tokenizers" / "text_tokenizer.json"
        with open(text_tok, encoding="utf8") as f:
            vocab_json = json.load(f)
        vocab_info = chameleon_vae_ori.VocabInfo(vocab_json["model"]["vocab"])
        self._vqgan_translation = chameleon_vae_ori.VocabTranslation(vocab_info, device="cuda")
        logger.info("VQGAN loaded: yaml=%s ckpt=%s", yaml_path, ckpt_path)

    @property
    def vqgan_backend(self) -> str:
        return self._vqgan_backend

    @property
    def fa4_attn_active(self) -> bool:
        return bool(getattr(self._attn, "_fa4_enabled", False))

    def _ensure_trt_vqgan_loaded(self) -> bool:
        """Lazy-instantiate optional TensorRT VQGAN acceleration.

        Generic standalone Chameleon defaults to eager Chameleon VQGAN
        tokenization. TensorRT is an explicit opt-in framework acceleration
        path for deployments that have compatible engines under the FlashRT
        engine cache (or a caller-provided engine directory). If unavailable,
        the frontend falls back to eager tokenization.
        """
        if not self._use_trt_vqgan:
            return False
        if self._trt_vqgan_backend is not None:
            return self._trt_vqgan_backend.is_available()
        try:
            from flash_rt.hardware.thor.vqgan_trt_backend import VQGANTRTBackend
        except ImportError as e:
            logger.warning("TRT VQGAN import failed: %s", e)
            self._use_trt_vqgan = False
            self._vqgan_backend = "eager"
            return False
        if torch.backends.cudnn.allow_tf32:
            logger.info("TRT VQGAN: disabling cuDNN TF32 globally.")
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        engine_dir = pathlib.Path(self._trt_vqgan_engine_dir) \
            if self._trt_vqgan_engine_dir else None
        self._trt_vqgan_backend = VQGANTRTBackend(engine_dir=engine_dir)
        available = self._trt_vqgan_backend.is_available()
        if not available:
            logger.warning(
                "TRT VQGAN backend unavailable at %s; falling back to "
                "eager PyTorch encode.",
                engine_dir or VQGANTRTBackend.ENGINE_DIR)
            self._use_trt_vqgan = False
            self._vqgan_backend = "eager"
        else:
            self._vqgan_backend = "trt"
        return available

    def _preprocess_image_for_trt(self, pil_image, out_hw):
        """PIL uint8 -> CUDA [1,3,H,W] float32 [-1,1] (matches the HF reference preprocessing)."""
        H_out, W_out = out_hw
        if pil_image.size != (W_out, H_out):
            pil_image = pil_image.resize((W_out, H_out), resample=PIL.Image.BICUBIC)
        np_img = np.array(pil_image.convert("RGB")) / 255.0
        np_img = np_img * 2.0 - 1.0
        t = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)
        return t.to(dtype=torch.float32, device="cuda").contiguous()

    def _vqgan_encode(self, image) -> list[int]:
        if isinstance(image, np.ndarray):
            image = _PILImage.fromarray(image.astype(np.uint8)).convert("RGB")
        elif not isinstance(image, PIL.Image.Image):
            raise TypeError(f"image must be PIL.Image or np.ndarray, got {type(image)!r}")

        # ── TRT fast path ──
        if self._ensure_trt_vqgan_loaded():
            H_eng = W_eng = int(self.target_size)
            if self._trt_vqgan_backend.supports_resolution(H_eng, W_eng):
                with torch.cuda.stream(self._trt_stream):
                    img = self._preprocess_image_for_trt(image, (H_eng, W_eng))
                    indices = self._trt_vqgan_backend.encode(img)
                    if indices is not None:
                        latent_ids = indices.view(-1)
                        global_ids = self._vqgan_translation.convert_img2bp2(
                            latent_ids).view(-1)
                        h_lat = H_eng // 16
                        w_lat = W_eng // 16
                        h_grids = H_eng // PATCH_SIZE
                        w_grids = W_eng // PATCH_SIZE
                        grid = global_ids.view(h_lat, w_lat)
                        newline_col = torch.full(
                            (h_lat, 1), NEWLINE_ID, dtype=grid.dtype, device=grid.device)
                        with_nl = torch.cat([grid, newline_col], dim=1).flatten().tolist()
                        return [IMG_START_ID, GRID_TOK_BASE + h_grids, GRID_TOK_BASE + w_grids,
                                *with_nl, IMG_END_ID]

        # ── Eager PyTorch fallback ──
        # data_lerobot's aspect-preserving center crop when importable;
        # otherwise a self-contained re-implementation of the same
        # selection (max-coverage crop from the aspect-varied size list),
        # so the eager path never depends on external packages.
        try:
            from data_lerobot.item_processor import (  # type: ignore
                var_center_crop, generate_crop_size_list)
        except ImportError:
            def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
                crop_size_list = []
                wp, hp = num_patches, 1
                while wp > 0:
                    if max(wp, hp) / min(wp, hp) <= max_ratio:
                        crop_size_list.append((wp * patch_size, hp * patch_size))
                    if (hp + 1) * wp <= num_patches:
                        hp += 1
                    else:
                        wp -= 1
                return crop_size_list

            def var_center_crop(image, crop_size_list=None):
                crop_size_list = crop_size_list or [(image.size[0], image.size[1])]
                w, h = image.size
                rem = [min(cw / w, ch / h) / max(cw / w, ch / h)
                       for cw, ch in crop_size_list]
                best = sorted(zip(rem, crop_size_list), reverse=True)[0][1]
                left = max(0, (w - best[0]) // 2)
                top = max(0, (h - best[1]) // 2)
                return image.crop((left, top, left + best[0], top + best[1]))

        crop_size_list = generate_crop_size_list(
            (self.target_size // PATCH_SIZE) ** 2, PATCH_SIZE)
        cropped = var_center_crop(image, crop_size_list=crop_size_list)
        latent_ids = self._vqgan_tokenizer.img_tokens_from_pil(cropped)
        global_ids = self._vqgan_translation.convert_img2bp2(latent_ids).view(-1)

        w_grids = cropped.size[0] // PATCH_SIZE
        h_grids = cropped.size[1] // PATCH_SIZE
        w_lat = cropped.size[0] // 16
        h_lat = cropped.size[1] // 16
        grid = global_ids.view(h_lat, w_lat)
        newline_col = torch.full((h_lat, 1), NEWLINE_ID, dtype=grid.dtype, device=grid.device)
        with_nl = torch.cat([grid, newline_col], dim=1).flatten().tolist()
        return [IMG_START_ID, GRID_TOK_BASE + h_grids, GRID_TOK_BASE + w_grids,
                *with_nl, IMG_END_ID]

    def _allocate_buffers(self) -> None:
        # Capacity is floored to a multiple of 16 so the pad-to-16 in
        # set_prompt can never overshoot the allocated buffers/KV cache when
        # max_seq itself is not a multiple of 16.
        Se = (self._max_pos // 16) * 16
        D = D_LLM
        Dff = DFF_LLM
        self._Se_max = Se
        self._bufs: dict[str, torch.Tensor] = {}

        def _alloc(name, shape, dtype=fp16):
            self._bufs[name] = torch.zeros(shape, dtype=dtype, device="cuda")

        _alloc("x", (Se, D))
        _alloc("xn", (Se, D))
        _alloc("xn_fp8", (Se, D), fp8)
        _alloc("o_proj_out", (Se, D))
        _alloc("hidden_all", (Se, D))
        _alloc("gate_out", (Se, Dff))
        _alloc("up_out", (Se, Dff))
        _alloc("gu_fp8", (Se, Dff), fp8)
        _alloc("zero_bias_d", (D,))
        _alloc("zero_bias_dff", (Dff,))
        _alloc("act_fp4", (Se * D // 2,), torch.uint8)
        _alloc("act_sfa", (Se * D // 16 * 2,), torch.uint8)
        _alloc("ffn_act_fp4", (Se * Dff // 2,), torch.uint8)
        _alloc("ffn_act_sfa", (Se * Dff // 16 * 2,), torch.uint8)
        _alloc("gu_merged", (Se, 2 * Dff))
        _alloc("dyn_act_scales", (L_LLM * 4,), torch.float32)
        _alloc("last_logits", (VOCAB_SIZE,))

        self._llm_calib_scales = torch.ones(L_LLM * 4, dtype=torch.float32, device="cuda")
        self._kv_cache = torch.zeros(L_LLM, 2, Se, D, dtype=fp16, device="cuda")
        self._kv_layer_stride = 2 * Se * D * 2
        logits_sz = max(NH_LLM * Se * Se, 4)
        self._logits_buf = torch.zeros(logits_sz, dtype=fp16, device="cuda")

    def _build_attention_backend(self) -> None:
        spec = make_chameleon_attention_spec(seq_max=self._Se_max)
        kv_base = self._kv_cache.data_ptr()
        se_d_bytes = self._Se_max * D_LLM * 2
        chameleon_slots = {
            "Q_O": self._bufs["xn"].data_ptr(),
            "Kc": kv_base,
            "Vc": kv_base + se_d_bytes,
            "logits": self._logits_buf.data_ptr(),
            "layer_stride": self._kv_layer_stride,
            "scale": 1.0 / math.sqrt(HD_LLM),
        }
        self._attn = ThorChameleonAttnBackend(
            spec, self._ctx, chameleon_slots=chameleon_slots)

    def _build_llm_weights(self) -> dict:
        w = {
            "input_ln_w": [x.data_ptr() for x in self._llm_input_ln_w],
            "post_ln_w": [x.data_ptr() for x in self._llm_post_ln_w],
            "q_w": [x.data_ptr() for x in self._q_w],
            "k_w": [x.data_ptr() for x in self._k_w],
            "v_w": [x.data_ptr() for x in self._v_w],
            "o_w": [x.data_ptr() for x in self._llm_o_w],
            "gate_w": [x.data_ptr() for x in self._gate_w],
            "up_w": [x.data_ptr() for x in self._up_w],
            "d_w": [x.data_ptr() for x in self._llm_d_w],
            "q_norm_w": [x.data_ptr() for x in self._llm_q_norm_w],
            "q_norm_b": [x.data_ptr() for x in self._llm_q_norm_b],
            "k_norm_w": [x.data_ptr() for x in self._llm_k_norm_w],
            "k_norm_b": [x.data_ptr() for x in self._llm_k_norm_b],
            "o_b": [self._bufs["zero_bias_d"].data_ptr()] * L_LLM,
            "final_norm_w": self._llm_norm_w.data_ptr(),
            "rope_cos": self._rope_cos.data_ptr(),
            "rope_sin": self._rope_sin.data_ptr(),
        }
        if self._use_fp8:
            w.update({
                "w_scales_flat": self._llm_w_dev.data_ptr(),
                "d_w_qkv": self._d_w_qkv_ptrs,
                "d_w_o": self._d_w_o_ptrs,
                "d_w_gu": self._d_w_gu_ptrs,
                "d_w_d": self._d_w_d_ptrs,
                "alpha_host": [1.0] * (L_LLM * 4),
                "gu_w_fp4": [
                    x.data_ptr() if x is not None else 0 for x in self._gu_w_fp4],
                "gu_sfb": [
                    x.data_ptr() if x is not None else 0 for x in self._gu_sfb],
                "d_w_fp4": [
                    x.data_ptr() if x is not None else 0 for x in self._d_w_fp4],
                "d_sfb": [
                    x.data_ptr() if x is not None else 0 for x in self._d_sfb],
            })
        return w

    def _build_llm_bufs(self) -> dict:
        return {k: self._bufs[k].data_ptr() for k in (
            "x", "xn", "xn_fp8", "o_proj_out", "hidden_all",
            "gate_out", "up_out", "gu_fp8", "zero_bias_d", "zero_bias_dff",
            "act_fp4", "act_sfa", "ffn_act_fp4", "ffn_act_sfa", "gu_merged",
            "dyn_act_scales",
        )}

    def _build_llm_scales_dev(self) -> dict:
        calib_base = self._llm_calib_scales.data_ptr()
        dyn_base = self._bufs["dyn_act_scales"].data_ptr()
        return {
            "act_qkv": [(calib_base + (li * 4 + 0) * 4) for li in range(L_LLM)],
            "act_o": [(calib_base + (li * 4 + 1) * 4) for li in range(L_LLM)],
            "act_gu": [(calib_base + (li * 4 + 2) * 4) for li in range(L_LLM)],
            "act_down": [(calib_base + (li * 4 + 3) * 4) for li in range(L_LLM)],
            "dyn_act_qkv": [(dyn_base + (li * 4 + 0) * 4) for li in range(L_LLM)],
            "dyn_act_o": [(dyn_base + (li * 4 + 1) * 4) for li in range(L_LLM)],
            "dyn_act_gu": [(dyn_base + (li * 4 + 2) * 4) for li in range(L_LLM)],
            "dyn_act_down": [(dyn_base + (li * 4 + 3) * 4) for li in range(L_LLM)],
        }

    def encode_prompt(self, text: str, images: Optional[list] = None) -> list[int]:
        images = images or []
        chunks = text.split("<image>")
        if len(chunks) - 1 not in (0, len(images)):
            raise ValueError("number of <image> placeholders must be 0 or match images")
        ids: list[int] = []
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is not None:
            ids.append(int(bos))
        for i, chunk in enumerate(chunks):
            if chunk:
                ids.extend(self.tokenizer.encode(chunk, add_special_tokens=False))
            if i < len(chunks) - 1:
                ids.extend(self._vqgan_encode(images[i]))
        if len(chunks) == 1:
            for image in images:
                ids.extend(self._vqgan_encode(image))
        return ids

    def _embed_ids(self, input_ids: list[int]) -> None:
        Se = len(input_ids)
        ids_t = torch.tensor(input_ids, dtype=torch.long, device="cuda")
        emb = torch.nn.functional.embedding(ids_t, self._llm_embed_w)
        self._bufs["x"].zero_()
        _cudart.cudaMemcpyAsync(
            ctypes.c_void_p(self._bufs["x"].data_ptr()),
            ctypes.c_void_p(emb.data_ptr()),
            Se * D_LLM * 2,
            3,
            ctypes.c_void_p(self._stream),
        )

    def set_prompt(self, text: str, images: Optional[list] = None) -> list[int]:
        input_ids = self.encode_prompt(text, images)
        self._real_len = len(input_ids)
        rem = len(input_ids) % 16
        padded_len = len(input_ids) + ((16 - rem) if rem else 0)
        if padded_len > self._Se_max:
            raise ValueError(
                f"padded sequence length {padded_len} exceeds max_seq="
                f"{self._Se_max} (prompt has {len(input_ids)} tokens)")
        if rem:
            input_ids.extend([PAD_ID] * (16 - rem))
        self.Se = len(input_ids)
        self._last_input_ids = input_ids
        if self._use_autotune:
            self._autotune_gemms(self.Se)
        self._embed_ids(input_ids)
        if self._use_cuda_graph:
            self._capture_graph(self.Se)
        return input_ids

    def _autotune_gemms(self, Se: int, num_algos: int = 16) -> None:
        """Per-shape cuBLASLt algo autotune (motus/hyvla pattern).

        Chameleon's per-layer GEMMs collapse to 3 distinct (M,N,K) shapes
        at a given Se (q/k/v/o share one shape, gate/up share another,
        down is the third) plus the M=1 lm_head projection. Tuning each
        shape once mutates the GemmRunner's internal algo cache (keyed on
        (M,N,K)); every subsequent real call with that shape — including
        inside a captured CUDA graph — picks up the tuned algo for free.
        Dummy buffers are only used for timing, not correctness.
        """
        if Se in self._autotuned_se:
            return
        D, Dff = D_LLM, DFF_LLM
        dev = "cuda"
        if self._use_fp8 and hasattr(self._gemm, "autotune_fp8_nn_dev_fp16"):
            shapes = [(Se, D, D), (Se, Dff, D), (Se, D, Dff)]
            shapes += [(1, D, D), (1, Dff, D), (1, D, Dff)]  # decode
            for (M, N, K) in dict.fromkeys(shapes):
                A = torch.empty(M, K, dtype=torch.uint8, device=dev)
                B = torch.empty(K, N, dtype=torch.uint8, device=dev)
                Dbuf = torch.empty(M, N, dtype=fp16, device=dev)
                sa = torch.ones(1, dtype=torch.float32, device=dev)
                sb = torch.ones(1, dtype=torch.float32, device=dev)
                self._gemm.autotune_fp8_nn_dev_fp16(
                    A.data_ptr(), B.data_ptr(), Dbuf.data_ptr(),
                    M, N, K, sa.data_ptr(), sb.data_ptr(), num_algos)
        elif hasattr(self._gemm, "autotune_fp16_nn"):
            shapes = [(Se, D, D), (Se, Dff, D), (Se, D, Dff)]
            shapes += [(1, D, D), (1, Dff, D), (1, D, Dff)]  # decode
            for (M, N, K) in dict.fromkeys(shapes):
                A = torch.empty(M, K, dtype=fp16, device=dev)
                B = torch.empty(K, N, dtype=fp16, device=dev)
                Dbuf = torch.empty(M, N, dtype=fp16, device=dev)
                self._gemm.autotune_fp16_nn(
                    A.data_ptr(), B.data_ptr(), Dbuf.data_ptr(), M, N, K, num_algos)
        if hasattr(self._gemm, "autotune_fp16_nn"):
            A = torch.empty(1, D, dtype=fp16, device=dev)
            B = torch.empty(D, VOCAB_SIZE, dtype=fp16, device=dev)
            Dbuf = torch.empty(1, VOCAB_SIZE, dtype=fp16, device=dev)
            self._gemm.autotune_fp16_nn(
                A.data_ptr(), B.data_ptr(), Dbuf.data_ptr(), 1, VOCAB_SIZE, D, num_algos)
        torch.cuda.synchronize()
        self._autotuned_se.add(Se)

    def _capture_graph(self, Se: int) -> None:
        if self._infer_graph is not None and self._captured_Se == Se:
            return
        capture_stream = torch.cuda.Stream()
        prev_stream_id = self._stream
        with torch.cuda.stream(capture_stream):
            self._stream = capture_stream.cuda_stream
            self._run_backbone(Se)
        torch.cuda.synchronize()
        # The warmup forward mutates x into the final residual stream; the
        # capture pass (and every later replay) must start from clean
        # embeddings — see _replay_backbone.
        self._embed_ids(self._last_input_ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            self._run_backbone(Se)
        self._stream = prev_stream_id
        self._infer_graph = graph
        self._captured_Se = Se

    def _run_backbone(self, Se: int) -> None:
        dims = {"Se": Se, "D": D_LLM, "Dff": DFF_LLM,
                "L": L_LLM, "H": NH_LLM, "Hd": HD_LLM}
        if self._use_fp8:
            chameleon_forward(
                self._gemm, fvk, self._build_llm_bufs(), self._build_llm_weights(),
                dims, self._build_llm_scales_dev(), attn=self._attn,
                stream=self._stream, dynamic_fp8_layers=frozenset(range(L_LLM)),
                fp4_ffn_layers=self._fp4_ffn_layers, ffn_down_clamp_value=60000.0,
                ffn_clamp_layers=self._ffn_clamp_layers,
            )
        else:
            chameleon_forward_fp16(
                self._gemm, fvk, self._build_llm_bufs(), self._build_llm_weights(),
                dims, attn=self._attn, stream=self._stream,
            )

    def _project_last(self, last_idx: Optional[int] = None) -> None:
        if last_idx is None:
            last_idx = self._real_len - 1
        last_hidden_ptr = self._bufs["hidden_all"].data_ptr() + last_idx * D_LLM * 2
        self._gemm.fp16_nn(
            last_hidden_ptr, self._lm_head_w_t.data_ptr(),
            self._bufs["last_logits"].data_ptr(), 1, VOCAB_SIZE, D_LLM,
            int(self._stream),
        )
        if self._mask_image_logits and self._image_token_ids is not None:
            self._bufs["last_logits"].index_fill_(0, self._image_token_ids, -65504.0)

    def _run_forward(self, Se: int) -> None:
        self._run_backbone(Se)
        self._project_last()

    def _replay_backbone(self) -> None:
        """Replay the captured prefill graph over clean embeddings.

        Every backbone run mutates ``x`` in place into the final residual
        stream, so a replay must re-embed first or it recomputes over the
        previous run's residuals.
        """
        self._embed_ids(self._last_input_ids)
        self._infer_graph.replay()

    def prefill(self, text: str, images: Optional[list] = None) -> dict:
        input_ids = self.set_prompt(text, images)
        if self._use_cuda_graph and self._infer_graph is not None:
            self._replay_backbone()
            self._project_last()
        else:
            self._run_forward(self.Se)
        torch.cuda.synchronize()
        return {
            "input_ids": input_ids,
            "Se": self.Se,
            "vqgan_backend": self._vqgan_backend,
            "fa4_attn": self.fa4_attn_active,
            "logits": self._bufs["last_logits"].detach().float().cpu(),
            "hidden": self._bufs["hidden_all"][:self.Se].detach().float().cpu(),
        }

    def generate_greedy(self, text: str, images: Optional[list] = None,
                        max_new_tokens: int = 16,
                        eos_token_id: Optional[int] = None) -> dict:
        """Greedy generation with incremental KV-cache decode.

        One prefill over the prompt, then single-token decode steps
        (``chameleon_decode_step``) — O(n) instead of the historical O(n^2)
        full recompute. Decode runs eagerly (no CUDA graph) because the
        position is a host scalar baked into RoPE offsets and cache rows.
        """
        if not self._use_fp8:
            raise NotImplementedError(
                "incremental decode requires the dynamic-FP8 path "
                "(use_fp8=True)")
        if self._fp4_ffn_layers:
            raise NotImplementedError(
                "incremental decode does not model the NVFP4 FFN tiers "
                "(fp4_ffn_layers); use the prefill-only path")

        self.set_prompt(text, images)
        generated = list(self._last_input_ids[:self._real_len])
        eos = EOS_ID if eos_token_id is None else int(eos_token_id)
        budget = int(max_new_tokens)
        if budget < 0:
            raise ValueError(
                f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if budget == 0:
            return {"input_ids": generated,
                    "text": self.tokenizer.decode(generated)}

        # Prefill (graph or eager) + first token from the last prompt row.
        if self._use_cuda_graph and self._infer_graph is not None:
            self._replay_backbone()
        else:
            self._run_backbone(self.Se)
        self._project_last()
        torch.cuda.synchronize()
        next_id = int(torch.argmax(self._bufs["last_logits"]).item())
        generated.append(next_id)

        # Decode-step plumbing: all pointers are stable across steps, so
        # build the dicts once; only `pos` changes per token.
        dims = {"Se": 1, "D": D_LLM, "Dff": DFF_LLM,
                "L": L_LLM, "H": NH_LLM, "Hd": HD_LLM}
        bufs = self._build_llm_bufs()
        weights = self._build_llm_weights()
        scales_dev = self._build_llm_scales_dev()
        pos = self._real_len
        for _ in range(budget - 1):
            if next_id == eos or pos >= self._Se_max:
                break
            # Decode state: single-token embedding in residual row 0.
            self._bufs["x"][:1].copy_(self._llm_embed_w[next_id])
            chameleon_decode_step(
                self._gemm, fvk, bufs, weights, dims, scales_dev,
                attn=self._attn, pos=pos, stream=int(self._stream),
                ffn_down_clamp_value=60000.0,
                ffn_clamp_layers=self._ffn_clamp_layers,
            )
            self._project_last(last_idx=0)
            torch.cuda.synchronize()
            next_id = int(torch.argmax(self._bufs["last_logits"]).item())
            generated.append(next_id)
            pos += 1
        return {"input_ids": generated, "text": self.tokenizer.decode(generated)}

    def _generate_greedy_recompute(self, text: str,
                                   images: Optional[list] = None,
                                   max_new_tokens: int = 16) -> dict:
        """Legacy O(n^2) full-recompute greedy path (oracle for decode).

        Always eager: it shares the KV cache with the incremental path,
        and graph capture/replay of every growing-Se forward would
        overwrite cache rows for pad-16 filler positions.
        """
        ids = self.encode_prompt(text, images)
        generated = list(ids)
        for _ in range(int(max_new_tokens)):
            if len(generated) >= self._Se_max:
                break
            self._real_len = len(generated)
            padded = list(generated)
            rem = len(padded) % 16
            if rem:
                padded.extend([PAD_ID] * (16 - rem))
            self.Se = len(padded)
            self._last_input_ids = padded
            if self._use_autotune:
                self._autotune_gemms(self.Se)
            self._embed_ids(padded)
            self._run_forward(self.Se)
            torch.cuda.synchronize()
            next_id = int(torch.argmax(self._bufs["last_logits"]).item())
            generated.append(next_id)
        return {"input_ids": generated, "text": self.tokenizer.decode(generated)}


__all__ = ["ChameleonTorchFrontendThor"]
