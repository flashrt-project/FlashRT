"""X-VLA (``lerobot/xvla-libero``) — RTX / SM120 torch frontend.

Owns everything around the compute path: checkpoint mapping, buffer
allocation, static prompt assembly and CUDA Graph capture. The inference
boundary is raw production input — two 256x256 float camera frames, the
8-D proprio state and a pre-tokenised instruction — producing the
postprocessed action chunk.

Usage::

    fe = XVLATorchFrontendRtx("/path/to/xvla-libero")
    fe.set_prompt(input_ids)          # once per episode; static thereafter
    actions = fe.predict_actions(images, state)

``predict_actions`` writes the new inputs into the preallocated staging
buffers and replays the captured graph; no tensor is allocated and no kernel
is dispatched from Python during steady-state replay.

Precision (`precision="bf16" | "fp8" | "nvfp4"`) selects the weight/activation
format for the four action-transformer GEMMs. All three were measured on real
LIBERO frames through the same equal-scope harness; on RTX 5060 Ti the winner
is NVFP4, so that is the default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from flash_rt.core.context import FvkContext
from flash_rt.models.xvla.pipeline_rtx import (
    ACTION_DIM,
    CHUNK,
    DIM_PROPRIO,
    N_SOFT,
    PREFIX,
    SEQ,
    VISION_TOKENS,
    VIS_TOK_DIM,
    XVLARTXPipeline,
)

# ── Checkpoint layout of lerobot/xvla-libero ──
# The checkpoint keeps the original vendored Florence-2 module names
# (``window_attn.fn.qkv``, ``ffn.fn.net.fc1``, ``convs.N.proj``), so the
# mapping below reads them directly instead of round-tripping through a
# rename table.
_VENDOR = "model.vlm."
_LM = "model.vlm.language_model.model.encoder."
_VT = "model.vlm.vision_tower."
_AT = "model.transformer."

_REQUIRED_CORE = (
    'FvkContext',
    'GemmRunner',
    'add_bias_bf16_pdl',
    'bf16_weight_to_nvfp4_swizzled',
    'fp4_w4a16_gemm_bias_gelu_bf16out_sm120',
    'fp4_w4a16_gemm_bias_gelu_fp4out_sm120',
    'fp4_w4a16_gemm_sm120_bf16out_pingpong_bias',
    'fp8_block128_gemm_cutlass_sm120_bf16out_bias',
    'fp8_per_token_block128_quant_bf16_v2',
    'gelu_erf_bf16',
    'gelu_inplace',
    'layer_norm_pdl',
    'nvfp4_sf_swizzled_bytes',
    'quantize_bf16_to_nvfp4_swizzled_v3',
    'residual_add_pdl',
    'splitk_b128_fp8_gemm_32x64x128_w4',
    'tq_cutlass_bf16_gemm',
    'xvla_action_decoder_gemv',
    'xvla_add2_bf16',
    'xvla_add_ln_quant_bf16',
    'xvla_add_ln_quant_norm_bf16',
    'xvla_copy_bf16',
    'xvla_finalize_bf16',
    'xvla_flow_step_bf16',
    'xvla_fp4_gemm_bias_bf16out',
    'xvla_fp4_gemm_bias_gelu_fp4out',
    'xvla_ln_bf16',
    'xvla_ln_quant_bf16',
    'xvla_ln_quant_bf16_nt256',
    'xvla_pdl_scope',
    'xvla_proprio_bf16',
    'xvla_vision_channel_attn',
    'xvla_vision_dwconv3x3_res',
    'xvla_vision_gelu_erf_to_nvfp4',
    'xvla_vision_im2col',
    'xvla_vision_projector_pack',
    'xvla_vision_resize',
    'xvla_vision_window_out_unpad_to_nvfp4',
    'xvla_vision_window_qkv_pad',
    'xvla_vision_window_qkv_pad_fill',
)
_REQUIRED_FA2 = ("fwd_bf16_tile", "fwd_bf16_tile64", "fwd_bf16_tile64m32w2_qnvfp4",
                 "fwd_bf16_tile64_qnvfp4", "has_xvla_tile64")


def _require_kernels():
    from flash_rt import _extensions
    core = _extensions.require(config="xvla")
    attention = _extensions.require("flash_rt_fa2", config="xvla")
    missing = ["flash_rt_kernels." + name for name in _REQUIRED_CORE if not hasattr(core, name)]
    missing += ["flash_rt_fa2." + name for name in _REQUIRED_FA2 if not hasattr(attention, name)]
    if missing or not attention.has_xvla_tile64():
        detail = ", ".join(missing) if missing else "FA2 X-VLA BF16 tile64 profile"
        raise RuntimeError(
            "X-VLA kernels are not built: " + detail + ". Rebuild with "
            "-DFLASHRT_BUILD_XVLA=ON -DFA2_DTYPES=bf16 -DFA2_HDIMS='64;96;128;256'.")
    return core, attention


MIN_TORCH_BF16 = True


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).cuda().contiguous()


_FP8_MAX = 448.0
_FP8_BLOCK = 128


# The DaViT tower's four block GEMM families run NVFP4 W4A4. The route was
# rejected once on a tower-output cosine proxy -- 0.9766 for both operands
# against block-128 FP8's 0.9988 in the same harness -- and that proxy turned
# out not to predict the gate: on the real 96-frame corpus the 4-bit tower
# measures mean 0.999289 / min 0.992583 against the gate's 0.99, and 29.936 ms
# becomes 28.051 ms. The block-128 FP8 tower tier was slower and is not part of
# this path; `xvla_precision` still selects FP8 for the action/encoder.


def _block128_quant_weight(w: torch.Tensor):
    """(N,K) bf16 -> (N,K) e4m3 uint8 + (N/128,K/128) fp32 scale.

    Same convention the repository's SM120 FP8 callers use:
    reconstruct = e4m3(w / scale) * scale with scale = amax/448 per 128x128
    block.
    """
    n, k = w.shape
    if n % _FP8_BLOCK or k % _FP8_BLOCK:
        raise ValueError(f"FP8 block-128 needs N,K multiples of 128, got {n},{k}")
    wr = w.float().reshape(n // _FP8_BLOCK, _FP8_BLOCK, k // _FP8_BLOCK, _FP8_BLOCK)
    amax = wr.abs().amax(dim=(1, 3))
    scale = (amax / _FP8_MAX).clamp(min=1e-12)
    q = (wr / scale[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX)
    q8 = q.to(torch.float8_e4m3fn).view(torch.uint8).reshape(n, k)
    return q8.contiguous(), scale.contiguous()


def _f16(t: torch.Tensor) -> torch.Tensor:
    """DaViT weights. The tower runs bf16 so that its block GEMMs can take the
    repository's SM120 FP8 path, whose A operand the quantizer emits as bf16."""
    return t.detach().to(torch.bfloat16).cuda().contiguous()


def _f32_to_linear(w: torch.Tensor) -> torch.Tensor:
    """[in, out] row-major -> [out, in] for the GEMM runner's B operand."""
    return w.t().contiguous()


class XVLATorchFrontendRtx:
    """Static-shape, CUDA-Graph X-VLA inference on SM120."""

    def __init__(self, checkpoint_dir: str, *, num_sms: int = 0, device: str = "cuda",
                 precision: str = "nvfp4", num_views: int = 2) -> None:
        if precision not in ("bf16", "fp8", "nvfp4"):
            raise ValueError(f"precision must be bf16/fp8/nvfp4, got {precision!r}")
        if num_views not in (2, 3):
            raise ValueError(
                "X-VLA has two real camera views plus one configured empty camera; "
                f"num_views must be 2 (or 3 when the caller includes the empty slot), got {num_views}"
            )
        self.device = self._resolve_device(device)
        self._require_arch(self.device)
        global fvk, fa2
        fvk, fa2 = _require_kernels()
        self.checkpoint_dir = str(checkpoint_dir)
        path = Path(self.checkpoint_dir) / "model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"model.safetensors not found in {self.checkpoint_dir}")
        self.num_views = num_views
        self.precision = precision
        self._tokenizer = None
        self._tokenizer_name = "facebook/bart-large"
        self._tokenizer_max_length = 50
        cfg_path = Path(self.checkpoint_dir) / "config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text())
            self._tokenizer_name = cfg.get("tokenizer_name", self._tokenizer_name)
        # The checkpoint policy preprocessor uses max_length=50; the config's
        # tokenizer_max_length (1024) is the training-time cap.
        self._prompt_text: str | None = None

        ctx = FvkContext()
        self.gemm = ctx.gemm
        self.num_sms = num_sms or torch.cuda.get_device_properties(self.device).multi_processor_count

        self.W = self._load_weights(path, precision)
        use_fp8, use_fp4 = precision == "fp8", precision == "nvfp4"
        if use_fp4:
            self._quantize_fp4()
        elif use_fp8:
            self._quantize_fp8()
        self.pipeline = XVLARTXPipeline(
            self.gemm, fvk, fa2, self.W, num_sms=self.num_sms, use_fp8=use_fp8,
            use_fp4=use_fp4,
        )
        self.B = self.pipeline.B
        self._graph: torch.cuda.CUDAGraph | None = None
        self._prompt_set = False

    @staticmethod
    def _require_arch(device) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("X-VLA RTX requires a CUDA device (SM120/SM121)")
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) not in ((12, 0), (12, 1)):
            raise RuntimeError(
                f"X-VLA RTX is validated on SM120/SM121; found sm_{major}{minor}. "
                "There is no fallback path for this model."
            )

    @staticmethod
    def _resolve_device(device):
        if not torch.cuda.is_available():
            raise RuntimeError("X-VLA RTX requires CUDA")
        selected = torch.device(device)
        current = torch.cuda.current_device()
        if selected.type != "cuda" or selected.index not in (None, current):
            raise ValueError("X-VLA device must be the current CUDA device; select it before constructing the frontend")
        return torch.device("cuda", current)

    def _require_current_device(self):
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError("X-VLA current CUDA device changed; restore the frontend's device before inference")

    # ── checkpoint mapping ───────────────────────────────────────────────

    def _load_weights(self, path: Path, precision: str = "bf16") -> dict:
        from safetensors.torch import load_file

        sd = load_file(str(path), device="cpu")
        W: dict = {}

        quantize = precision != "bf16"
        W["enc"] = []
        for li in range(12):
            p = f"{_LM}layers.{li}."
            enc_blk = {
                "ln1_w": _bf16(sd[f"{p}self_attn_layer_norm.weight"]),
                "ln1_b": _bf16(sd[f"{p}self_attn_layer_norm.bias"]),
                "qkv_w": _bf16(torch.cat([sd[f"{p}self_attn.{n}_proj.weight"] for n in "qkv"], 0)),
                "qkv_b": _bf16(torch.cat([sd[f"{p}self_attn.{n}_proj.bias"] for n in "qkv"], 0)),
                "proj_w": _bf16(sd[f"{p}self_attn.out_proj.weight"]),
                "proj_b": _bf16(sd[f"{p}self_attn.out_proj.bias"]),
                "ln2_w": _bf16(sd[f"{p}final_layer_norm.weight"]),
                "ln2_b": _bf16(sd[f"{p}final_layer_norm.bias"]),
                "fc1_w": _bf16(sd[f"{p}fc1.weight"]),
                "fc1_b": _bf16(sd[f"{p}fc1.bias"]),
                "fc2_w": _bf16(sd[f"{p}fc2.weight"]),
                "fc2_b": _bf16(sd[f"{p}fc2.bias"]),
            }
            # Same block-128 FP8 eligibility as the action transformer: all
            # four encoder GEMMs have N and K multiples of 128.
            enc_blk["fp8"] = {
                k: _block128_quant_weight(enc_blk[f"{k}_w"].to(torch.bfloat16))
                for k in ("qkv", "proj", "fc1", "fc2")
            } if quantize else {}
            W["enc"].append(enc_blk)
        W["enc_ln_embed_w"] = _bf16(sd[f"{_LM}layernorm_embedding.weight"])
        W["enc_ln_embed_b"] = _bf16(sd[f"{_LM}layernorm_embedding.bias"])
        # BART counts learned positions from an offset of 2; the encoder sees
        # positions 2..101 for a 100-token input.
        W["enc_pos"] = _bf16(sd[f"{_LM}embed_positions.weight"][2 : 2 + 100].contiguous())

        W["act"] = []
        for li in range(24):
            p = f"{_AT}blocks.{li}."
            blk = {
                "qkv_w": _bf16(sd[f"{p}attn.qkv.weight"]),
                "qkv_b": _bf16(sd[f"{p}attn.qkv.bias"]),
                "proj_w": _bf16(sd[f"{p}attn.proj.weight"]),
                "proj_b": _bf16(sd[f"{p}attn.proj.bias"]),
                "fc1_w": _bf16(sd[f"{p}mlp.fc1.weight"]),
                "fc1_b": _bf16(sd[f"{p}mlp.fc1.bias"]),
                "fc2_w": _bf16(sd[f"{p}mlp.fc2.weight"]),
                "fc2_b": _bf16(sd[f"{p}mlp.fc2.bias"]),
            }
            for src, dst in (("norm1", "n1"), ("norm2", "n2")):
                blk[f"{dst}w"] = _bf16(sd[f"{p}{src}.weight"])
                blk[f"{dst}b"] = _bf16(sd[f"{p}{src}.bias"])
            blk["fp8"] = {}
            blk["fp4"] = {}
            W["act"].append(blk)

        # DomainAwareLinear at the fixed LIBERO domain is a plain Linear over
        # one embedding row: fc.weight[d] is [out * in] laid out as [in, out].
        dom = 3
        ae = sd[f"{_AT}action_encoder.fc.weight"].float()[dom].view(72, 1024)
        W["action_encoder_w"] = _f32_to_linear(ae).to(torch.bfloat16).cuda().contiguous()
        W["action_encoder_b"] = _bf16(sd[f"{_AT}action_encoder.bias.weight"][dom])
        ad = sd[f"{_AT}action_decoder.fc.weight"].float()[dom].view(1024, ACTION_DIM)
        W["action_decoder_w"] = _f32_to_linear(ad).to(torch.bfloat16).cuda().contiguous()
        W["action_decoder_b"] = _bf16(sd[f"{_AT}action_decoder.bias.weight"][dom])
        # The CUTLASS bf16 tile rejects N=20; GemmRunner.bf16_nn wants B as a
        # contiguous (K, N) buffer, which is exactly the untransposed row.
        W["action_decoder_wt"] = ad.contiguous().to(torch.bfloat16).cuda()

        W["pos_emb"] = _bf16(sd[f"{_AT}pos_emb"][0, :PREFIX].contiguous())
        W["soft"] = _bf16(sd[f"{_AT}soft_prompt_hub.weight"].float()[dom].view(N_SOFT, 1024))
        W["norm_w"] = _bf16(sd[f"{_AT}norm.weight"])
        W["norm_b"] = _bf16(sd[f"{_AT}norm.bias"])
        W["vlm_proj_w"] = _bf16(sd[f"{_AT}vlm_proj.weight"])
        W["vlm_proj_b"] = _bf16(sd[f"{_AT}vlm_proj.bias"])
        W["aux_proj_w"] = _bf16(sd[f"{_AT}aux_visual_proj.weight"])
        W["aux_proj_b"] = _bf16(sd[f"{_AT}aux_visual_proj.bias"])

        W["vision"] = self._load_vision(sd)
        W["n_state"] = 8  # observation.state width for LIBERO
        W["text_emb"] = torch.zeros(50, 1024, dtype=torch.bfloat16, device="cuda")
        self._embed_tokens = sd[f"{_LM}embed_tokens.weight"]
        return W

    def _load_vision(self, sd: dict) -> dict:
        V: dict = {"convs": [], "blocks": []}
        for i in range(4):
            # Conv weight [Cout, Cin, k, k] -> [Cin*k*k, Cout] for D = A @ B.
            cw = sd[f"{_VT}convs.{i}.proj.weight"].float().flatten(1).t().contiguous()
            V["convs"].append({
                "w_t": _f16(cw),
                "b": _f16(sd[f"{_VT}convs.{i}.proj.bias"]),
                "nw": _f16(sd[f"{_VT}convs.{i}.norm.weight"]),
                "nb": _f16(sd[f"{_VT}convs.{i}.norm.bias"]),
            })
        for s, depth in enumerate((1, 1, 9, 1)):
            stage = []
            for b in range(depth):
                p = f"{_VT}blocks.{s}.{b}."
                blk = {}
                for sub, attn in (("spatial", "window_attn"), ("channel", "channel_attn")):
                    q = f"{p}{sub}_block."
                    lin = lambda w: _f16(w.float().t().contiguous())  # noqa: E731

                    def _w4(weight):
                        w = _f16(weight).contiguous()          # (N, K)
                        n, k = w.shape
                        packed = torch.empty(n, k // 2, dtype=torch.uint8,
                                             device=w.device)
                        sfb = torch.empty(fvk.nvfp4_sf_swizzled_bytes(n, k),
                                          dtype=torch.uint8, device=w.device)
                        scratch = torch.empty(1, dtype=torch.float32, device=w.device)
                        og = torch.empty(1, dtype=torch.float32, device=w.device)
                        fvk.bf16_weight_to_nvfp4_swizzled(
                            w.data_ptr(), packed.data_ptr(), sfb.data_ptr(),
                            scratch.data_ptr(), og.data_ptr(), n, k, 0)
                        torch.cuda.synchronize()
                        return (packed, sfb, float(og.item()))

                    blk[sub] = {
                        # Depthwise weights stay [C, 9]; the kernel indexes them directly.
                        "conv1_w": _f16(sd[f"{q}conv1.fn.dw.weight"].flatten(1)),
                        "conv1_b": _f16(sd[f"{q}conv1.fn.dw.bias"]),
                        "conv2_w": _f16(sd[f"{q}conv2.fn.dw.weight"].flatten(1)),
                        "conv2_b": _f16(sd[f"{q}conv2.fn.dw.bias"]),
                        "qkv4": _w4(sd[f"{q}{attn}.fn.qkv.weight"]),
                        "proj4": _w4(sd[f"{q}{attn}.fn.proj.weight"]),
                        "fc1_4": _w4(sd[f"{q}ffn.fn.net.fc1.weight"]),
                        "fc2_4": _w4(sd[f"{q}ffn.fn.net.fc2.weight"]),
                        "n1w": _f16(sd[f"{q}{attn}.norm.weight"]),
                        "n1b": _f16(sd[f"{q}{attn}.norm.bias"]),
                        "qkv_w_t": lin(sd[f"{q}{attn}.fn.qkv.weight"]),
                        "qkv_b": _f16(sd[f"{q}{attn}.fn.qkv.bias"]),
                        "proj_w_t": lin(sd[f"{q}{attn}.fn.proj.weight"]),
                        "proj_b": _f16(sd[f"{q}{attn}.fn.proj.bias"]),
                        "n2w": _f16(sd[f"{q}ffn.norm.weight"]),
                        "n2b": _f16(sd[f"{q}ffn.norm.bias"]),
                        "fc1_w_t": lin(sd[f"{q}ffn.fn.net.fc1.weight"]),
                        "fc1_b": _f16(sd[f"{q}ffn.fn.net.fc1.bias"]),
                        "fc2_w_t": lin(sd[f"{q}ffn.fn.net.fc2.weight"]),
                        "fc2_b": _f16(sd[f"{q}ffn.fn.net.fc2.bias"]),
                    }
                stage.append(blk)
            V["blocks"].append(stage)

        # The checkpoint stores `image_projection` in the vendored layout as
        # [2048, 1024] and applies it as x @ p; the native Linear equivalent is
        # its transpose. GEMM wants D[M,1024] = A[M,2048] @ W[2048,1024], which
        # is the vendored orientation, so no transpose is needed here.
        V["proj_w"] = _f16(sd["model.vlm.image_projection"].float().contiguous())
        V["proj_norm_w"] = _f16(sd["model.vlm.image_proj_norm.weight"])
        V["proj_norm_b"] = _f16(sd["model.vlm.image_proj_norm.bias"])

        # learned_abs_2d: columns indexed by w over the first half of the
        # channels, rows indexed by h over the second half, added to the NCHW
        # feature map before the 7x7 grid is flattened row-major.
        row = sd["model.vlm.image_pos_embed.row_embeddings.weight"].float()[:7]
        col = sd["model.vlm.image_pos_embed.column_embeddings.weight"].float()[:7]
        pos = torch.cat([col.unsqueeze(0).repeat(7, 1, 1), row.unsqueeze(1).repeat(1, 7, 1)], -1)
        V["pos_map"] = _f16(pos.reshape(49, 2048))
        V["temporal"] = _f16(sd["model.vlm.visual_temporal_embed.pos_idx_to_embed"][0])
        V["tokens"] = 49
        return V

    def _quantize_fp8(self) -> None:
        """Block-128 FP8 weights for the action transformer's four GEMMs.

        Static: the scales come from the checkpoint weights alone and never
        change at runtime. The N/K multiples of 128 that the CUTLASS tile needs
        hold for all four (3072x1024, 1024x1024, 4096x1024, 1024x4096); the
        narrow action encoder/decoder and the whole vision tower stay out.
        """
        for blk in self.W["act"]:
            for key in ("qkv", "proj", "fc1", "fc2"):
                w = blk[f"{key}_w"].to(torch.bfloat16)
                blk["fp8"][key] = _block128_quant_weight(w)

    def _quantize_fp4(self) -> None:
        """NVFP4 W4A4 weights for the transformer GEMMs.

        Static: packed weights and their per-16-element scales come from the
        checkpoint alone. `alpha` is the global scale the prep kernel emits.

        Built for the BART encoder's blocks too: its twelve layers are the same
        kind of linear+GELU MLP and attention as the action transformer's, and
        the measured 4-bit path is 2-3x the block-128 FP8 tile at the encoder's
        M=100 shapes. The route is kept or dropped by the delivery gate on the
        real corpus, not by assumption.
        """
        for blk in list(self.W["act"]) + list(self.W.get("enc", [])):
            blk["fp4"] = {}
            for key in ("qkv", "proj", "fc1", "fc2"):
                w = blk[f"{key}_w"].to(torch.bfloat16).contiguous()
                n, k = w.shape
                packed = torch.empty(n, k // 2, dtype=torch.uint8, device=w.device)
                sfb = torch.empty(fvk.nvfp4_sf_swizzled_bytes(n, k), dtype=torch.uint8,
                                  device=w.device)
                scratch = torch.empty(1, dtype=torch.float32, device=w.device)
                og = torch.empty(1, dtype=torch.float32, device=w.device)
                fvk.bf16_weight_to_nvfp4_swizzled(w.data_ptr(), packed.data_ptr(),
                                                  sfb.data_ptr(), scratch.data_ptr(),
                                                  og.data_ptr(), n, k, 0)
                torch.cuda.synchronize()
                blk["fp4"][key] = (packed, sfb, float(og.item()))

    # ── static prompt ────────────────────────────────────────────────────

    def _tokenize(self, prompt: str) -> torch.Tensor:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "X-VLA prompt tokenization requires the optional 'transformers' "
                    "dependency; install flash-rt[xvla]"
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        enc = self._tokenizer(
            prompt,
            max_length=self._tokenizer_max_length,
            padding="max_length",
            padding_side="right",
            truncation=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]
        if ids.numel() != self._tokenizer_max_length:
            pad = torch.full((self._tokenizer_max_length - ids.numel(),), 1, dtype=ids.dtype)
            ids = torch.cat([ids, pad])
        return ids

    def set_prompt(self, prompt: str | torch.Tensor, images=None, state=None) -> None:
        """Embed a fixed instruction (string or pre-tokenised 50-token ids)."""
        self._require_current_device()
        if isinstance(prompt, str):
            ids = self._tokenize(prompt)
            self._prompt_text = prompt
        else:
            ids = torch.as_tensor(prompt).reshape(-1).long()
            self._prompt_text = None
        if ids.numel() != 50:
            raise ValueError(f"expected a 50-token instruction, got {ids.numel()}")
        self.W["text_emb"].copy_(self._embed_tokens[ids.cpu()].to(torch.bfloat16).cuda())
        self._prompt_set = True

    # ── graph capture ────────────────────────────────────────────────────

    def capture(self, warmup: int = 3) -> None:
        self._require_current_device()
        if self._graph is not None:
            return
        if not self._prompt_set:
            raise RuntimeError("set_prompt() must be called before capture()")
        for _ in range(warmup):
            self.pipeline.run()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.pipeline.run()
        self._graph = g

    # ── inference ────────────────────────────────────────────────────────

    def _to_image_tensor(self, image) -> torch.Tensor:
        t = torch.as_tensor(image)
        if t.ndim != 3:
            raise ValueError(f"X-VLA images must be HxWxC or CxHxW, got shape {tuple(t.shape)}")
        if t.shape[-1] in (1, 3, 4) and t.shape[0] not in (1, 3, 4):
            t = t.permute(2, 0, 1)
        if t.shape[0] == 4:
            t = t[:3]
        if t.shape[0] != 3:
            raise ValueError(f"X-VLA images must have 3 channels, got shape {tuple(t.shape)}")
        t = t.float()
        if float(t.max()) > 1.5:
            t = t / 255.0
        # The reference raw LIBERO frames are 256x256; the tower applies its
        # own 256->224 resize-with-padding inside the captured graph.
        if t.shape[1] != 256 or t.shape[2] != 256:
            t = F.interpolate(
                t.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
            ).squeeze(0)
        return t

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        """Public ``load_model`` inference surface: raw observation -> actions.

        ``obs`` follows the repository convention: ``images`` (a list of
        HxWxC camera arrays) or ``image`` / ``wrist_image`` keys, plus an 8-D
        ``state``. A string ``prompt``/``task`` rebinds the instruction when
        it changes; otherwise ``set_prompt`` must have been called first.
        """
        self._require_current_device()
        task = obs.get("prompt") or obs.get("task")
        if isinstance(task, str) and task != self._prompt_text:
            self.set_prompt(task)
        if not self._prompt_set:
            raise RuntimeError("set_prompt() must be called before infer()")
        images = obs.get("images")
        if images is None:
            images = [obs[k] for k in ("image", "wrist_image", "wrist_image_right")
                      if obs.get(k) is not None]
        if isinstance(images, dict):
            images = [images[k] for k in ("image", "wrist_image", "wrist_image_right")
                      if images.get(k) is not None]
        if not isinstance(images, (list, tuple)) or len(images) < 2:
            raise ValueError(
                "X-VLA requires two real camera views (agentview + wrist); got "
                f"{type(images).__name__} with "
                f"{len(images) if isinstance(images, (list, tuple)) else '?'} view(s)"
            )
        img_t = torch.stack([self._to_image_tensor(im) for im in images[:2]], 0).to(self.device)
        state = obs.get("state")
        if state is None:
            raise ValueError("X-VLA requires observation.state (8-D)")
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(-1)
        n = self.W["n_state"]
        if st.numel() < n:
            st = F.pad(st, (0, n - st.numel()))
        if self._graph is None:
            self.capture()
        actions = self.predict_actions(img_t, st[:n])
        return {"actions": actions.numpy()}

    @torch.no_grad()
    def predict_actions(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """images [V,3,H,W] float in [0,1]; state [8] -> action chunk [30,20] on CPU.

        Only the staging copies run per call; all model math is inside the graph.
        """
        self._require_current_device()
        n = self.W["n_state"]
        self.B["images"].copy_(images)
        self.B["state_in"][:n].copy_(state.reshape(-1)[:n])
        self._graph.replay()
        torch.cuda.synchronize()
        return self.B["out"].cpu()
