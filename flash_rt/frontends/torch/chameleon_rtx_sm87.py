"""FlashRT — upstream Chameleon-7B VLM frontend for Jetson AGX Orin (SM87).

Image+text -> text. Constructed directly (**not** via ``flash_rt.load_model``):
this is a chat-style VLM exposing ``set_prompt()`` + ``generate()``, whereas
``VLAModel.predict`` unconditionally reads ``result['actions']``. Same precedent
as ``qwen3_vl`` — see the redirect in ``flash_rt/api.py``.

Production precision policy (all defaults)
------------------------------------------
* **Q/K/V/O, FFN gate/up** — INT8 W8A8 **+ Hadamard rotation** (QuaRot at 8
  bits): weights rotated offline, activations rotated inside the fused norm
  kernel; per-row dynamic activation scales.
* **FFN down** — INT8 W8A8 per-row dynamic. Not rotated: K=11008 is not a power
  of two and its input is the un-rotated BF16 SiLU output.
* **lm_head** (65536x4096) — INT8 W8A8. 268 MB/token = 3.7 % of the decode
  budget; FP16 would double that for no measured argmax benefit.
* **residual / QK-LayerNorm / RoPE / attention / KV cache** — FP16, with
  ``ffn_down_clamp`` applied on the last ``ffn_down_clamp_last_n`` layers.
* **attention** — FA2 fp16 causal; ``split_kv_bias=4`` on decode, because FA2's
  own heuristic returns ``num_splits=1`` at Chameleon's 32 Q heads.
* **VQ-GAN encoder** — FP16 convs + **fp32** codebook distance/argmin.

Measured (ISL=1032, OSL=16, warm): greedy output **bit-identical to the HF bf16
reference for 16/16 tokens**, worst per-layer cosine 0.9986, last-row logit
cosine 0.999968, **21.07 tok/s** decode, 273.8 ms prefill, 7.6 GB resident.

Two non-obvious *correctness* requirements
------------------------------------------
1. **The Hadamard rotation is not an optimization.** Chameleon's
   massive-activation channels (measured: L31 row-0 channel d632 at 2.4e4 against
   a row median of 1e4) pin the per-row INT8 amax and round that row's other
   ~4090 channels to zero. Plain per-row INT8 reproduces only 8/16 reference
   tokens; rotating fixes it. It is free because it preserves per-row scales, so
   the stock ``cutlass_int8_rowwise_*`` GEMMs are reused unchanged.
2. **``ffn_down_clamp`` is not a tuning knob.** L31's down output reaches 2.6e5,
   past FP16's 65504; without the clamp the residual stores ``inf`` and the final
   RMSNorm poisons that row. See ``models/chameleon/pipeline_thor.py``.

Dim policy: backbone dims (32 layers / 4096 / 32 heads / 11008 / vocab 65536) and
the special token ids are **hard-asserted** against ``config.json``, so this
frontend is Chameleon-7B-specific by construction rather than by convention.
Note ``config.json``'s ``bos_token_id`` is stale (says 1, which is ``<pad>``);
``tokenizer.json`` gives ``<s> = 0`` and that is what the processor emits.

Preprocessing specifics
-----------------------
Chameleon needs PIL LANCZOS / 512 / ``u8*0.0078-1.0`` -> ``[-1, +0.989]``
normalization and a bare 1024-token raster per image (no grid/newline
layout); the quantizers live in ``_chameleon_quant``.

See ``docs/chameleon7b_rtx_sm87.md`` for the roofline, the measured lever menu,
and the dead-ends.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.frontends.torch import _chameleon_quant as cq
from flash_rt.frontends.torch._chameleon_rtx_sm87_spec import build_spec
from flash_rt.hardware.rtx.attn_backend_chameleon import (
    ChameleonAttnBackend, make_chameleon_attention_spec)
from flash_rt.models.chameleon.pipeline_rtx import chameleon_forward

logger = logging.getLogger(__name__)

_FP16 = torch.float16
_BF16 = torch.bfloat16


class ChameleonTorchFrontendRtxSm87:
    """Chameleon-7B image+text → text on Orin SM87.

    Typical use::

        f = ChameleonTorchFrontendRtxSm87("/path/to/Chameleon_7B_mGPT")
        f.set_prompt("<image>Describe this image.", images=[pil_img])
        print(f.generate(max_new_tokens=32))
    """

    # Token ids for this checkpoint. Hardcoded so an accelerated deployment does
    # not depend on the training code being importable, but asserted against
    # config.json at load so a different checkpoint fails loudly (principle #10).
    BOS_ID = 0
    EOS_ID = 2
    IMG_PLACEHOLDER_ID = 8711      # <image>
    BOI_ID = 8197                  # <racm3:break>
    EOI_ID = 8196                  # <eoss>
    SEP_ID = 8710                  # <reserved08706>
    IMG_ID_OFFSET = 4              # token_id = codebook_index + 4
    N_IMG_CODES = 8192
    IMG_TOKENS_PER_VIEW = 1024     # 512/16 = 32 -> 32x32 raster

    # Chameleon-7B backbone dims — hard-asserted against config.json.
    D = 4096
    L = 32
    H = 32
    HKV = 32
    HD = 128
    DFF = 11008
    VOCAB = 65536

    def __init__(self, checkpoint_dir: str, *,
                 max_seq: int = 2048,
                 use_int4: bool = False,
                 use_int4_down: bool = False,
                 use_hadamard: bool = True,
                 split_kv_bias: int = 4,
                 ffn_down_clamp: Optional[float] = None,
                 vq_argmin_fp32: bool = True,
                 free_fp16_weights: bool = True,
                 probe_layers: Optional[Sequence[int]] = None,
                 **_ignored) -> None:
        cc = torch.cuda.get_device_capability(0)
        if cc != (8, 7) and os.environ.get("FLASHRT_CHAMELEON_SM87_FORCE") != "1":
            raise RuntimeError(
                f"ChameleonTorchFrontendRtxSm87 targets SM87 (Orin); found SM{cc[0]}{cc[1]}. "
                "The INT8/INT4 path needs ENABLE_SM80_INT8_CUTLASS=ON. Set "
                "FLASHRT_CHAMELEON_SM87_FORCE=1 to override.")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_seq = int(max_seq)
        self.use_int4_down = bool(use_int4_down)
        self.use_int4 = bool(use_int4) or self.use_int4_down
        # W8A8+QuaRot: Hadamard-rotate the six K=4096 projections and quantize
        # at 8 bits. Conditions the massive-activation channels (which plain
        # per-row INT8 cannot handle) without INT4's noise, and keeps per-row
        # scales so the stock CUTLASS GEMM is reused. Ignored on the INT4 tier,
        # which already rotates.
        self.use_hadamard = bool(use_hadamard) and not self.use_int4
        self.vq_argmin_fp32 = bool(vq_argmin_fp32)
        # Required for correctness: L31's down output reaches ~2.6e5, far past
        # FP16's 65504. See chameleon_forward's docstring for the measured
        # per-layer table and docs/chameleon_acceleration_methodology.md §2.1.
        self.ffn_down_clamp = float(
            os.environ.get("FLASHRT_CHAMELEON_DOWN_CLAMP", "60000.0")
            if ffn_down_clamp is None else ffn_down_clamp)
        self.ffn_down_clamp_last_n = int(
            os.environ.get("FLASHRT_CHAMELEON_DOWN_CLAMP_LAST_N", "4"))
        self._probe_layers = list(probe_layers) if probe_layers else []

        self._validate_config()
        t0 = time.perf_counter()
        self._load_weights(free_fp16_weights=free_fp16_weights)
        self._build_rope_tables()
        self._allocate_buffers()
        self._build_attn_backend(split_kv_bias=split_kv_bias)
        self._load_processor()
        self._load_vqgan()

        # Prompt state.
        self.input_ids: Optional[torch.Tensor] = None
        self.S = 0
        self._prompt_ready = False
        self._timing: dict = {}

        logger.info(
            "ChameleonTorchFrontendRtxSm87 ready in %.1fs — tier=%s max_seq=%d "
            "gpu_mem=%.2f GB",
            time.perf_counter() - t0, self.precision_tier, self.max_seq,
            torch.cuda.memory_allocated() / 2 ** 30)

    # ==================================================================
    # Load
    # ==================================================================

    @property
    def precision_tier(self) -> str:
        if self.use_int4_down:
            return "int4+down"
        if self.use_int4:
            return "int4"
        return "int8+hadamard" if self.use_hadamard else "int8"

    def _validate_config(self) -> None:
        cfg = json.loads((self.checkpoint_dir / "config.json").read_text())
        expect = {
            "hidden_size": self.D, "num_hidden_layers": self.L,
            "num_attention_heads": self.H, "num_key_value_heads": self.HKV,
            "intermediate_size": self.DFF, "vocab_size": self.VOCAB,
        }
        for k, want in expect.items():
            got = cfg.get(k)
            if got != want:
                raise RuntimeError(
                    f"config.json {k}={got}, this frontend hardcodes {want}. "
                    "Chameleon-7B only.")
        if cfg.get("attention_bias") or cfg.get("mlp_bias"):
            raise RuntimeError(
                "This frontend assumes attention_bias=false and mlp_bias=false "
                "(upstream Chameleon). A biased checkpoint needs the *_bias "
                "GEMM entries wired back in.")
        if cfg.get("model_parallel_size", 1) != 1:
            raise RuntimeError(
                f"model_parallel_size={cfg.get('model_parallel_size')}: the "
                "QK-Norm params would differ per head group, but "
                "qk_norm_rope_fused_fp16 broadcasts one [head_dim] vector "
                "across all heads. Only the 7B (mp=1) layout is supported.")
        if self.max_seq > cfg.get("max_position_embeddings", 4096):
            raise ValueError(
                f"max_seq={self.max_seq} exceeds max_position_embeddings="
                f"{cfg['max_position_embeddings']}")

        # Token-id contract. config.json's bos_token_id is stale (says 1, which
        # is <pad>); tokenizer.json is authoritative and gives <s> = 0.
        vm = cfg.get("vocabulary_map") or {}
        for tok, want in (("<image>", self.IMG_PLACEHOLDER_ID),
                          ("<racm3:break>", self.BOI_ID),
                          ("<eoss>", self.EOI_ID),
                          ("<reserved08706>", self.SEP_ID)):
            if vm.get(tok) != want:
                raise RuntimeError(
                    f"token id mismatch: {tok} is {vm.get(tok)} in config.json, "
                    f"expected {want}")
        img_ids = sorted(v for k, v in vm.items() if k.startswith("IMGIMG"))
        if (len(img_ids) != self.N_IMG_CODES
                or img_ids[0] != self.IMG_ID_OFFSET
                or img_ids[-1] != self.IMG_ID_OFFSET + self.N_IMG_CODES - 1):
            raise RuntimeError(
                f"expected {self.N_IMG_CODES} contiguous IMGIMG ids starting at "
                f"{self.IMG_ID_OFFSET}; got {len(img_ids)} spanning "
                f"[{img_ids[0]}, {img_ids[-1]}]")
        self._config = cfg

    def _load_weights(self, *, free_fp16_weights: bool) -> None:
        from flash_rt.executors.torch_weights import (
            MultiSafetensorsSource, WeightLoader)

        shards = sorted(self.checkpoint_dir.glob("model-*-of-*.safetensors"))
        if not shards:
            shards = sorted(self.checkpoint_dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors in {self.checkpoint_dir}")

        src = MultiSafetensorsSource([str(p) for p in shards], device="cuda")
        WeightLoader(source=src, target=self, spec=build_spec()).run()
        del src

        proj = cq.split_fused_projections(
            self._llm_qkv_w, self._llm_gu_w, self._llm_o_w, self._llm_d_w,
            D=self.D, Dff=self.DFF)
        # The fused tensors are dead once split; release before quantizing so
        # the peak is (fp16 split + int8) rather than (fused + split + int8).
        self._llm_qkv_w = []
        self._llm_gu_w = []
        torch.cuda.empty_cache()

        self.qw = cq.quantize_int8_all(proj, num_layers=self.L)
        if self.use_int4:
            cq.quantize_int4_quarot(self.qw, proj, num_layers=self.L, D=self.D,
                                    include_down=self.use_int4_down)
        elif self.use_hadamard:
            cq.quantize_int8_hadamard(self.qw, proj, num_layers=self.L, D=self.D)

        # lm_head: INT8 in both tiers (see pipeline docstring for the ROI).
        self._lm_head_q, self._lm_head_s = cq.quantize_per_row_int8(
            self._llm_lm_head_w.t().contiguous())      # [V, D] -> [K=D, N=V]

        if free_fp16_weights:
            for key in cq.PROJECTIONS:
                proj[key] = []
            self._llm_o_w = []
            self._llm_d_w = []
            self._llm_lm_head_w = None
            torch.cuda.empty_cache()

        logger.info("weights: %.2f GB quantized (%s) + %.2f GB lm_head int8 + "
                    "%.2f GB embed fp16",
                    self.qw.bytes() / 2 ** 30, self.precision_tier,
                    self._lm_head_q.numel() / 2 ** 30,
                    self._llm_embed_w.numel() * 2 / 2 ** 30)

    def _build_rope_tables(self) -> None:
        """cos/sin as ``[max_seq, HD]`` fp16, ``cat([f, f], -1)`` tiled.

        The kernel reads ``cos[s*HD + d]`` for d in ``[0, HD)``, so the
        half-frequencies must be duplicated across the two halves; a
        ``[max_seq, HD/2]`` table indexes out of stride and corrupts RoPE. Being
        row-major with stride exactly HD is also what lets a decode step select
        position ``pos`` by pointer arithmetic alone.
        """
        theta = float(self._config.get("rope_theta", 10000.0))
        inv = 1.0 / (theta ** (torch.arange(0, self.HD, 2, dtype=torch.float32,
                                            device="cuda") / self.HD))
        pos = torch.arange(self.max_seq, device="cuda", dtype=torch.float32)
        f = pos[:, None] * inv[None, :]
        full = torch.cat([f, f], dim=-1)
        self._rope_cos = torch.cos(full).to(_FP16).contiguous()
        self._rope_sin = torch.sin(full).to(_FP16).contiguous()

    def _allocate_buffers(self) -> None:
        MS, D, Dff, V = self.max_seq, self.D, self.DFF, self.VOCAB
        dev = "cuda"
        z = lambda *sh, dt: torch.zeros(*sh, dtype=dt, device=dev)  # noqa: E731

        self._x = z(MS, D, dt=_FP16)                  # residual stream
        self._xn = z(MS, D, dt=_FP16)                 # final-norm output
        self._o_proj_out = z(MS, D, dt=_FP16)
        self._int8_act_d = z(MS, D, dt=torch.int8)
        self._int8_act_ff = z(MS, Dff, dt=torch.int8)
        self._bf16_gate_ff = z(MS, Dff, dt=_BF16)
        self._bf16_xn_ff = z(MS, Dff, dt=_BF16)
        self._int4_act_d = z(MS, D // 2, dt=torch.uint8) if self.use_int4 else None
        self._int4_act_ff = (z(MS, Dff // 2, dt=torch.uint8)
                             if self.use_int4_down else None)

        # Dynamic per-row activation scales — one shared [MS] vector per quant
        # site, reused by every layer (decode never uses static calibration:
        # that was fitted at prefill M=Se and does not describe one decode row).
        self._act_scale = {k: z(MS, dt=torch.float32)
                           for k in ("qkv", "o", "gu", "down")}
        self._lm_act = z(MS, D, dt=torch.int8)
        self._lm_act_scale = z(MS, dt=torch.float32)
        self._logits = z(1, V, dt=_BF16)
        self._logits_all: Optional[torch.Tensor] = None
        self._bf16_min = torch.finfo(_BF16).min

        self._probe_bufs = [z(MS, D, dt=_FP16) for _ in self._probe_layers]
        self._probe_final = z(MS, D, dt=_FP16) if self._probe_layers else None

        self._tok_dev = z(1, dt=torch.long)

    def _build_attn_backend(self, *, split_kv_bias: int) -> None:
        spec = make_chameleon_attention_spec(
            num_layers=self.L, num_q_heads=self.H, num_kv_heads=self.HKV,
            head_dim=self.HD, max_seq=self.max_seq)
        self.attn = ChameleonAttnBackend(spec, max_seq=self.max_seq,
                                        split_kv_bias=split_kv_bias)

    def _load_processor(self) -> None:
        """Use the HF ChameleonProcessor verbatim.

        Reimplementing it is a correctness liability: the pipeline is
        blend-RGBA-on-white → PIL **LANCZOS** shortest-edge 512 → center-crop
        512 → ``float32(float64(u8) * 0.0078) - 1.0``, giving ``[-1, +0.989]``
        (note: *not* ``[-1, 1]``). ``ChameleonImageProcessorFast`` silently
        substitutes BICUBIC for LANCZOS, so the slow/PIL path is required — it
        is what ``preprocessor_config.json`` selects by default, and it runs
        once per image outside any graph.
        """
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(str(self.checkpoint_dir))
        if int(getattr(self.processor, "image_seq_length", 0)) != self.IMG_TOKENS_PER_VIEW:
            raise RuntimeError(
                f"processor image_seq_length="
                f"{self.processor.image_seq_length}, expected "
                f"{self.IMG_TOKENS_PER_VIEW}")

    def _load_vqgan(self) -> None:
        """Load the HF ``ChameleonVQVAE`` encoder from the checkpoint shards."""
        from safetensors.torch import load_file
        from transformers import ChameleonConfig, ChameleonVQVAE

        index = self.checkpoint_dir / "model.safetensors.index.json"
        prefix = "model.vqmodel."
        sd = {}
        if index.exists():
            wmap = json.loads(index.read_text())["weight_map"]
            keys = [k for k in wmap if k.startswith(prefix)]
            for shard in sorted({wmap[k] for k in keys}):
                full = load_file(str(self.checkpoint_dir / shard))
                for k in keys:
                    if k in full:
                        sd[k[len(prefix):]] = full[k]
                del full
        if not sd:
            raise FileNotFoundError(
                f"no {prefix}* weights found under {self.checkpoint_dir}")

        cfg = ChameleonConfig.from_pretrained(str(self.checkpoint_dir))
        vq = ChameleonVQVAE._from_config(cfg.vq_config)
        missing, unexpected = vq.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected vqmodel keys: {unexpected[:4]}")
        if missing:
            raise RuntimeError(f"missing vqmodel keys: {missing[:4]}")
        self.vqgan = vq.eval().to(device="cuda", dtype=_FP16)
        # fp32 codebook for the distance/argmin (see _vq_encode).
        self._vq_codebook_f32 = (
            self.vqgan.quantize.embedding.weight.detach().float().contiguous())
        logger.info("VQ-GAN encoder loaded (%d tensors, fp16 convs, "
                    "argmin in %s)", len(sd),
                    "fp32" if self.vq_argmin_fp32 else "fp16")

    # ==================================================================
    # Image tokenization
    # ==================================================================

    @torch.no_grad()
    def _vq_encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[N, 3, 512, 512]`` fp32 in ``[-1, 0.989]`` → ``[N, 1024]`` token ids.

        The convs run fp16 but the codebook distance and argmin run **fp32**:
        ``|z|^2 + |e|^2 - 2 z.e`` is cancellation-prone, and in fp16 it flips
        codebook indices (measured 98.14 % → 99.02 % index match vs an all-fp32
        reference for <0.1 ms; see docs §2.4). Tokens are a row-major raster of
        the 32x32 latent grid, and the id is simply ``code + 4``.
        """
        px = pixel_values.to(device="cuda", dtype=_FP16)
        h = self.vqgan.quant_conv(self.vqgan.encoder(px))
        if not self.vq_argmin_fp32:
            _, _, idx = self.vqgan.quantize(h)
            return idx.view(px.shape[0], -1) + self.IMG_ID_OFFSET
        e = self._vq_codebook_f32                                  # [n_emb, dim]
        z = h.permute(0, 2, 3, 1).contiguous().view(-1, e.shape[1]).float()
        d = z.pow(2).sum(1, keepdim=True) + e.pow(2).sum(1) - 2.0 * (z @ e.t())
        return d.argmin(1).view(px.shape[0], -1) + self.IMG_ID_OFFSET

    # ==================================================================
    # Prompt
    # ==================================================================

    def set_prompt(self, text: str, images=None, *,
                   input_ids: Optional[Sequence[int]] = None) -> None:
        """Tokenize, VQ-encode the images, and seed the residual stream.

        Args:
            text: prompt containing one ``<image>`` per image, e.g.
                ``"<image>Describe this image."``. The processor expands each
                placeholder to ``<racm3:break>`` + 1024x ``<image>`` +
                ``<eoss>``, prepends BOS and appends the sep token.
            images: list of PIL images (or a single image).
            input_ids: bypass tokenization + VQ entirely and use these ids
                verbatim. Used by the precision harness to feed the reference's
                exact ids, which isolates LLM error from VQ-GAN index drift.
        """
        t0 = time.perf_counter()
        if input_ids is not None:
            ids = torch.as_tensor(list(input_ids), dtype=torch.long, device="cuda")
            n_img = 0
        else:
            if images is not None and not isinstance(images, (list, tuple)):
                images = [images]
            n_text_img = text.count("<image>")
            if images and n_text_img != len(images):
                raise ValueError(
                    f"text has {n_text_img} '<image>' placeholders but "
                    f"{len(images)} images were given")
            enc = self.processor(text=text, images=images if images else None,
                                 return_tensors="pt")
            ids = enc["input_ids"][0].to("cuda")
            n_img = 0
            if images:
                img_ids = self._vq_encode(enc["pixel_values"])       # [N, 1024]
                n_img = img_ids.shape[0]
                # Upstream substitutes at the id level: masked_scatter over
                # input_ids == <image>. Order is row-major per image, images in
                # order, which matches the placeholder order in the string.
                slot = ids == self.IMG_PLACEHOLDER_ID
                want = n_img * self.IMG_TOKENS_PER_VIEW
                if int(slot.sum()) != want:
                    raise RuntimeError(
                        f"{int(slot.sum())} <image> placeholders vs {want} "
                        "VQ tokens")
                ids = ids.masked_scatter(slot, img_ids.reshape(-1).to(ids.dtype))

        S = int(ids.numel())
        if S > self.max_seq:
            raise ValueError(
                f"prompt is {S} tokens but max_seq={self.max_seq}. Note "
                f"S = 1 + n_img*1026 + n_text + 1.")
        if S + 1 > self.max_seq:
            raise ValueError(
                f"prompt {S} tokens leaves no room to decode within "
                f"max_seq={self.max_seq}")

        # NOTE: S is used exactly, never padded. Padding would put junk rows in
        # the KV cache that decode would then attend to (a prefill path without a KV cache would pad Se to even).
        self.input_ids = ids
        self.S = S
        self.attn.reset_cache()
        # Seed the residual stream. No sqrt(D) scaling — Chameleon feeds raw
        # embeddings, and image tokens are ordinary vocab rows (no projector).
        torch.index_select(self._llm_embed_w, 0, ids, out=self._x[:S])
        self._prompt_ready = True
        self._timing = {"prompt_ms": (time.perf_counter() - t0) * 1e3,
                        "S": S, "n_images": n_img}

    # ==================================================================
    # Forward
    # ==================================================================

    def _dims(self) -> dict:
        return {"D": self.D, "Dff": self.DFF, "L": self.L, "H": self.H,
                "Hd": self.HD, "vocab": self.VOCAB}

    def _bufs(self, *, logits_ptr: int) -> dict:
        return {
            "x": self._x.data_ptr(),
            "xn": self._xn.data_ptr(),
            "o_proj_out": self._o_proj_out.data_ptr(),
            "int8_act_d": self._int8_act_d.data_ptr(),
            "int8_act_ff": self._int8_act_ff.data_ptr(),
            "int4_act_d": self._int4_act_d.data_ptr() if self.use_int4 else 0,
            "int4_act_ff": (self._int4_act_ff.data_ptr()
                            if self.use_int4_down else 0),
            "bf16_gate_ff": self._bf16_gate_ff.data_ptr(),
            "bf16_xn_ff": self._bf16_xn_ff.data_ptr(),
            "logits": logits_ptr,
            "lm_act": self._lm_act.data_ptr(),
            "lm_act_scale": self._lm_act_scale.data_ptr(),
        }

    def _weights(self) -> dict:
        w = {
            "rope_cos": self._rope_cos.data_ptr(),
            "rope_sin": self._rope_sin.data_ptr(),
            "final_norm_w": self._llm_norm_w.data_ptr(),
            "lm_head_w": self._lm_head_q.data_ptr(),
            "lm_head_w_scale": self._lm_head_s.data_ptr(),
            "input_ln_w": [t.data_ptr() for t in self._llm_input_ln_w],
            "post_ln_w": [t.data_ptr() for t in self._llm_post_ln_w],
            "q_norm_w": [t.data_ptr() for t in self._llm_q_norm_w],
            "q_norm_b": [t.data_ptr() for t in self._llm_q_norm_b],
            "k_norm_w": [t.data_ptr() for t in self._llm_k_norm_w],
            "k_norm_b": [t.data_ptr() for t in self._llm_k_norm_b],
        }
        for key in cq.PROJECTIONS:
            w[f"{key}_w"] = self.qw.ptr[key]
            w[f"{key}_w_scale"] = self.qw.scale_ptr[key]
        return w

    def _scales_dev(self) -> dict:
        return {f"act_{k}": [v.data_ptr()] * self.L
                for k, v in self._act_scale.items()}

    def _probe(self) -> Optional[dict]:
        if not self._probe_layers:
            return None
        return {"layers": self._probe_layers,
                "bufs": [b.data_ptr() for b in self._probe_bufs],
                "final_buf": self._probe_final.data_ptr()}

    def _require_prompt(self) -> None:
        if not self._prompt_ready:
            raise RuntimeError("call set_prompt() before prefill()/generate()")

    @torch.no_grad()
    def prefill(self, *, logits_all: bool = False) -> torch.Tensor:
        """Run the prompt through the 32 layers, filling the KV cache.

        Returns the masked BF16 logits: ``[1, vocab]`` for the last position, or
        ``[S, vocab]`` when ``logits_all`` (teacher-forced comparison).
        """
        self._require_prompt()
        t0 = time.perf_counter()
        if logits_all:
            if self._logits_all is None or self._logits_all.shape[0] < self.S:
                self._logits_all = torch.zeros(self.max_seq, self.VOCAB,
                                               dtype=_BF16, device="cuda")
            out = self._logits_all[:self.S]
            logits_ptr = self._logits_all.data_ptr()
        else:
            out = self._logits
            logits_ptr = self._logits.data_ptr()

        chameleon_forward(
            fvk, self._bufs(logits_ptr=logits_ptr), self._weights(),
            self._dims(), self._scales_dev(),
            attn=self.attn, S=self.S, pos=None, stream=0,
            use_int4=self.use_int4, use_int4_down=self.use_int4_down,
            use_hadamard=self.use_hadamard,
            ffn_down_clamp_value=self.ffn_down_clamp,
            ffn_down_clamp_last_n=self.ffn_down_clamp_last_n,
            logits_all=logits_all, probe=self._probe())
        self._mask_image_logits(out)
        torch.cuda.synchronize()
        self._timing["prefill_ms"] = (time.perf_counter() - t0) * 1e3
        return out

    @torch.no_grad()
    def decode_step(self, token_id, *, pos: int, stream: int = 0) -> torch.Tensor:
        """One decode step: embed ``token_id``, attend keys ``[0, pos]``.

        ``pos`` is the absolute KV position this token occupies, i.e. ``S`` for
        the first generated token.

        ``stream`` must be the capture stream when this body is being recorded
        into a CUDA graph. Launching the kernels on the legacy default stream
        while another stream is capturing leaves them **out** of the graph
        without raising — the replay then only re-runs the torch ops and the
        logits never change, which looks exactly like a frozen/stale graph.
        """
        self._require_prompt()
        if isinstance(token_id, torch.Tensor):
            self._tok_dev.copy_(token_id.reshape(1).to(torch.long))
        else:
            self._tok_dev.fill_(int(token_id))
        torch.index_select(self._llm_embed_w, 0, self._tok_dev, out=self._x[:1])
        chameleon_forward(
            fvk, self._bufs(logits_ptr=self._logits.data_ptr()), self._weights(),
            self._dims(), self._scales_dev(),
            attn=self.attn, S=1, pos=int(pos), stream=int(stream),
            use_int4=self.use_int4, use_int4_down=self.use_int4_down,
            use_hadamard=self.use_hadamard,
            ffn_down_clamp_value=self.ffn_down_clamp,
            ffn_down_clamp_last_n=self.ffn_down_clamp_last_n)
        self._mask_image_logits(self._logits)
        return self._logits

    def _mask_image_logits(self, logits: torch.Tensor) -> None:
        """Suppress the 8192 image-codebook ids, as upstream does every forward.

        Upstream applies this inside ``forward`` at *all* positions, so it is
        not something a ``LogitsProcessor`` could be used for and it cannot be
        disabled through the generation config.
        """
        lo = self.IMG_ID_OFFSET
        logits[:, lo:lo + self.N_IMG_CODES].fill_(self._bf16_min)

    # ==================================================================
    # Generate
    # ==================================================================

    @torch.no_grad()
    def generate(self, text: Optional[str] = None, images=None, *,
                 max_new_tokens: int = 32, eos_token_id: Optional[int] = None,
                 return_ids: bool = False, skip_special_tokens: bool = True):
        """Greedy decode. Returns the decoded string (or the raw id list).

        Greedy only: the checkpoint's generation config is already
        ``do_sample=False``, and argmax over BF16 logits is order-preserving, so
        it matches an fp32 argmax except on exact BF16 ties.
        """
        if text is not None:
            self.set_prompt(text, images)
        self._require_prompt()
        if max_new_tokens < 0:
            raise ValueError(
                f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if max_new_tokens == 0:
            return [] if return_ids else ""
        eos = self.EOS_ID if eos_token_id is None else int(eos_token_id)
        budget = min(max_new_tokens, self.max_seq - self.S)
        if budget < max_new_tokens:
            logger.warning("max_new_tokens clipped %d -> %d by max_seq=%d",
                           max_new_tokens, budget, self.max_seq)

        logits = self.prefill()
        tok = int(torch.argmax(logits[0]).item())
        out: List[int] = [tok]

        t0 = time.perf_counter()
        steps = 0
        for i in range(budget - 1):
            if tok == eos:
                break
            logits = self.decode_step(tok, pos=self.S + i)
            tok = int(torch.argmax(logits[0]).item())
            out.append(tok)
            steps += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self._timing["decode_steps"] = steps
        self._timing["decode_ms_per_token"] = (dt / steps * 1e3) if steps else 0.0
        self._timing["decode_tok_s"] = (steps / dt) if dt > 0 else 0.0

        if out and out[-1] == eos:
            out = out[:-1]
        if return_ids:
            return out
        return self.processor.tokenizer.decode(
            out, skip_special_tokens=skip_special_tokens)

    # ==================================================================
    # Introspection
    # ==================================================================

    def snapshot_probe(self) -> dict:
        """Per-layer post-residual hidden states captured during the last forward."""
        if not self._probe_layers:
            return {}
        S = self.S
        out = {f"layer_{li}": b[:S].clone()
               for li, b in zip(self._probe_layers, self._probe_bufs)}
        out["final_norm"] = self._probe_final[:S].clone()
        return out

    def reset(self) -> None:
        self.attn.reset_cache()
        self._prompt_ready = False
        self.input_ids = None
        self.S = 0

    @property
    def timing(self) -> dict:
        return dict(self._timing)

    def precision_spec(self) -> dict:
        return {
            "tier": self.precision_tier,
            "llm_gemms": dict(self.qw.precision),
            "lm_head": "int8",
            "residual": "fp16",
            "attention": "fp16 FA2 causal (split_kv_bias="
                         f"{self.attn.split_kv_bias})",
            "kv_cache": "fp16",
            "ffn_down_clamp": f"{self.ffn_down_clamp:.0f} on last {self.ffn_down_clamp_last_n} layers",
            "vqgan": f"fp16 convs, argmin "
                     f"{'fp32' if self.vq_argmin_fp32 else 'fp16'}",
        }

    def get_model_info(self) -> dict:
        return {
            "model": "chameleon-7b", "arch": "rtx_sm87",
            "layers": self.L, "hidden": self.D, "ffn": self.DFF,
            "heads": f"{self.H}Q/{self.HKV}KV", "head_dim": self.HD,
            "vocab": self.VOCAB, "max_seq": self.max_seq,
            "precision": self.precision_spec(),
        }


__all__ = ["ChameleonTorchFrontendRtxSm87"]
