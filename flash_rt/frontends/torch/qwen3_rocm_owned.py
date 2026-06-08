"""FlashRT -- owned BF16 Qwen3 frontend for ROCm.

This module provides a reusable Qwen3 ROCm owned-buffer decoder with
preallocated KV cache, graph capture, and static FP8 entry points.
"""

from __future__ import annotations

import os
import time
from typing import Any

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


class Qwen3RocmOwnedBF16Frontend:
    """Qwen3-8B ROCm BF16 frontend with owned buffers and KV cache."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        max_seq: int = 2048,
        max_q_seq: int = 512,
        device: str = "cuda",
        trust_remote_code: bool = True,
        preferred_attn_backend: str = "flash_attn",
        use_fp8_lm_head: bool = False,
        use_fp8_layers: bool = False,
        **_: Any,
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        from flash_rt.frontends.torch.qwen3_rocm_weights import (
            extract_weights_qwen3_bf16_rocm,
        )
        from flash_rt.hardware.rocm.attn_backend_qwen3 import RocmQwen3AttnBackend

        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            raise RuntimeError("Qwen3RocmOwnedBF16Frontend requires ROCm PyTorch")

        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.max_seq = int(max_seq)
        self.max_q_seq = int(max_q_seq)
        self.latency_records: list[float] = []
        self._prompt_ids = None
        self._generated_ids = None
        self._decode_graphs = {}
        self._prefill_graphs = {}
        self.use_fp8_lm_head = bool(use_fp8_lm_head)
        self.use_fp8_layers = bool(use_fp8_layers)
        self.fp8_lm_head_input_scale = None

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            trust_remote_code=trust_remote_code,
        )
        self.weights = extract_weights_qwen3_bf16_rocm(
            self.checkpoint_path,
            device=device,
            include_fp8=self.use_fp8_lm_head or self.use_fp8_layers,
            fp8_modules=tuple(
                name
                for name, enabled in (
                    ("lm_head", self.use_fp8_lm_head),
                    ("layers", self.use_fp8_layers),
                )
                if enabled
            ),
        )
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0

        ptrs = self.weights.ptrs
        self.ptrs = ptrs
        self.hidden_size = int(ptrs["hidden"])
        self.intermediate = int(ptrs["intermediate"])
        self.vocab_size = int(ptrs["vocab_size"])
        self.num_layers = int(ptrs["num_layers"])
        self.q_heads = int(ptrs["num_q_heads"])
        self.kv_heads = int(ptrs["num_kv_heads"])
        self.head_dim = int(ptrs["head_dim"])
        self.q_dim = self.q_heads * self.head_dim
        self.kv_dim = self.kv_heads * self.head_dim
        self.qkv_dim = self.q_dim + 2 * self.kv_dim
        self.eps = float(ptrs["rms_norm_eps"])
        self._tensor_by_ptr = {int(t.data_ptr()): t for t in self.weights.anchors}

        bf16 = torch.bfloat16
        self.hidden = torch.empty(
            self.max_q_seq, self.hidden_size, device=device, dtype=bf16
        )
        self.residual = torch.empty_like(self.hidden)
        self.norm = torch.empty_like(self.hidden)
        self.qkv = torch.empty(self.max_q_seq, self.qkv_dim, device=device, dtype=bf16)
        self.attn_proj = torch.empty_like(self.hidden)
        self.post_norm = torch.empty_like(self.hidden)
        self.gate_up = torch.empty(
            self.max_q_seq, 2 * self.intermediate, device=device, dtype=bf16
        )
        self.act = torch.empty(
            self.max_q_seq, self.intermediate, device=device, dtype=bf16
        )
        self.mlp_out = torch.empty_like(self.hidden)
        self.final_norm = torch.empty_like(self.hidden)
        self.logits = torch.empty(
            self.max_q_seq, self.vocab_size, device=device, dtype=bf16
        )
        self.final_norm_fp8 = (
            torch.empty(self.max_q_seq, self.hidden_size, device=device, dtype=torch.float8_e4m3fnuz)
            if self.use_fp8_lm_head
            else None
        )
        self.norm_fp8 = (
            torch.empty(self.max_q_seq, self.hidden_size, device=device, dtype=torch.float8_e4m3fnuz)
            if self.use_fp8_layers
            else None
        )
        self.attn_o_fp8 = (
            torch.empty(self.max_q_seq, self.hidden_size, device=device, dtype=torch.float8_e4m3fnuz)
            if self.use_fp8_layers
            else None
        )
        self.post_norm_fp8 = (
            torch.empty(self.max_q_seq, self.hidden_size, device=device, dtype=torch.float8_e4m3fnuz)
            if self.use_fp8_layers
            else None
        )
        self.act_fp8 = (
            torch.empty(self.max_q_seq, self.intermediate, device=device, dtype=torch.float8_e4m3fnuz)
            if self.use_fp8_layers
            else None
        )
        self.fp8_lm_head_input_scale = (
            torch.ones(1, device=device, dtype=torch.float32)
            if self.use_fp8_lm_head
            else None
        )
        self.fp8_qkv_input_scales = (
            torch.ones(self.num_layers, 1, device=device, dtype=torch.float32)
            if self.use_fp8_layers
            else None
        )
        self.fp8_o_input_scales = (
            torch.ones(self.num_layers, 1, device=device, dtype=torch.float32)
            if self.use_fp8_layers
            else None
        )
        self.fp8_gate_up_input_scales = (
            torch.ones(self.num_layers, 1, device=device, dtype=torch.float32)
            if self.use_fp8_layers
            else None
        )
        self.fp8_down_input_scales = (
            torch.ones(self.num_layers, 1, device=device, dtype=torch.float32)
            if self.use_fp8_layers
            else None
        )
        self._fp8_qkv_scale_views = (
            [self.fp8_qkv_input_scales[i : i + 1] for i in range(self.num_layers)]
            if self.use_fp8_layers
            else []
        )
        self._fp8_o_scale_views = (
            [self.fp8_o_input_scales[i : i + 1] for i in range(self.num_layers)]
            if self.use_fp8_layers
            else []
        )
        self._fp8_gate_up_scale_views = (
            [self.fp8_gate_up_input_scales[i : i + 1] for i in range(self.num_layers)]
            if self.use_fp8_layers
            else []
        )
        self._fp8_down_scale_views = (
            [self.fp8_down_input_scales[i : i + 1] for i in range(self.num_layers)]
            if self.use_fp8_layers
            else []
        )
        self.input_ids_buf = torch.empty(self.max_q_seq, device=device, dtype=torch.long)

        self.attn = RocmQwen3AttnBackend(
            num_layers=self.num_layers,
            max_seq=self.max_seq,
            max_q_seq=self.max_q_seq,
            q_heads=self.q_heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            dtype=bf16,
            preferred_backend=preferred_attn_backend,
        )
        self.cos, self.sin = self._build_rope_cache(self.max_seq, self.head_dim, device)

        self.config_summary = {
            "backend": "rocm_owned_bf16",
            "model_type": ptrs.get("model_type"),
            "num_hidden_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate,
            "num_attention_heads": self.q_heads,
            "num_key_value_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "max_seq": self.max_seq,
            "max_q_seq": self.max_q_seq,
            "attn_backend": self.attn.active_backend_name,
            "use_fp8_lm_head": self.use_fp8_lm_head,
            "use_fp8_layers": self.use_fp8_layers,
        }

    @staticmethod
    def _build_rope_cache(max_seq: int, head_dim: int, device: str):
        import torch

        pos = torch.arange(max_seq, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            1_000_000.0
            ** (
                torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                / head_dim
            )
        )
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().contiguous(), emb.sin().contiguous()

    def _t(self, ptr: int):
        return self._tensor_by_ptr[int(ptr)]

    @staticmethod
    def _copy_fp8_scale(dst, src) -> None:
        import torch

        scale = torch.clamp(src.float().abs().max() / 240.0, min=1.0e-8)
        dst.copy_(scale.reshape(1))

    def reset_state(self) -> None:
        self._generated_ids = None

    def set_prompt(self, text: str) -> None:
        enc = self.tokenizer(text, return_tensors="pt")
        ids = enc.input_ids.to(self.device)
        if ids.shape[1] > self.max_seq:
            raise ValueError(f"prompt length {ids.shape[1]} exceeds max_seq={self.max_seq}")
        self._prompt_ids = ids

    def _forward_prepared(
        self,
        q_seq: int,
        *,
        pos_start: int = 0,
        causal: bool | None = None,
        sync: bool = True,
        collect_fp8_scales: bool = False,
    ):
        """Run with ``self.input_ids_buf[:q_seq]`` already populated."""
        import torch
        import flash_rt.flash_rt_rocm_kernels as kernels
        stream = torch.cuda.current_stream().cuda_stream

        if q_seq <= 0 or q_seq > self.max_q_seq:
            raise ValueError(f"q_seq={q_seq} outside 1..max_q_seq={self.max_q_seq}")
        if pos_start < 0 or pos_start + q_seq > self.max_seq:
            raise ValueError("position range exceeds max_seq")
        if causal is None:
            causal = q_seq > 1

        embed_w = self._t(self.ptrs["embed_w"])
        kernels.embedding_lookup_bf16_ptr(
            self.input_ids_buf.data_ptr(),
            embed_w.data_ptr(),
            self.hidden.data_ptr(),
            q_seq,
            self.hidden_size,
            stream,
        )
        cos = self.cos[pos_start : pos_start + q_seq]
        sin = self.sin[pos_start : pos_start + q_seq]

        qkv_input_fp8_ready = False
        final_norm_fp8_ready = False
        for layer_idx, layer in enumerate(self.ptrs["layers"]):
            if self.use_fp8_layers:
                if self.norm_fp8 is None:
                    raise RuntimeError("FP8 layer buffers are not initialized")
                fp8_layer = self.ptrs["fp8"]["layers"][layer_idx]
                if qkv_input_fp8_ready:
                    qkv_input_fp8_ready = False
                else:
                    kernels.rms_norm_fp8_e4m3fnuz_plain_ptr(
                        self.hidden.data_ptr(),
                        self._t(layer["input_norm_w"]).data_ptr(),
                        self.norm_fp8.data_ptr(),
                        self._fp8_qkv_scale_views[layer_idx].data_ptr(),
                        q_seq,
                        self.hidden_size,
                        self.eps,
                        stream,
                    )
                kernels.hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr(
                    self.norm_fp8.data_ptr(),
                    self._t(fp8_layer["qkv_w"]).data_ptr(),
                    self._fp8_qkv_scale_views[layer_idx].data_ptr(),
                    self._t(fp8_layer["qkv_w_scale"]).data_ptr(),
                    0,
                    self.qkv.data_ptr(),
                    q_seq,
                    self.qkv_dim,
                    self.hidden_size,
                    stream,
                )
            else:
                kernels.rms_norm_bf16_plain_ptr(
                    self.hidden.data_ptr(),
                    self._t(layer["input_norm_w"]).data_ptr(),
                    self.norm.data_ptr(),
                    q_seq,
                    self.hidden_size,
                    self.eps,
                    stream,
                )
                if collect_fp8_scales and self.fp8_qkv_input_scales is not None:
                    self._copy_fp8_scale(
                        self._fp8_qkv_scale_views[layer_idx], self.norm[:q_seq]
                    )
                kernels.hipblaslt_linear_bf16_ptr(
                    self.norm.data_ptr(),
                    self._t(layer["qkv_w"]).data_ptr(),
                    0,
                    self.qkv.data_ptr(),
                    q_seq,
                    self.qkv_dim,
                    self.hidden_size,
                    stream,
                )
            kernels.qwen3_qkv_norm_rope_cache_bf16_ptr(
                self.qkv.data_ptr(),
                cos.data_ptr(),
                sin.data_ptr(),
                self._t(layer["q_norm_w"]).data_ptr(),
                self._t(layer["k_norm_w"]).data_ptr(),
                self.attn.q.data_ptr(),
                self.attn.k_cache.data_ptr(),
                self.attn.v_cache.data_ptr(),
                layer_idx,
                self.max_seq,
                pos_start,
                q_seq,
                self.q_heads,
                self.kv_heads,
                self.head_dim,
                stream,
            )
            kv_seq = pos_start + q_seq
            self.attn.run(layer_idx, q_seq, kv_seq, causal=bool(causal))
            attn_o = self.attn.o[:q_seq].reshape(q_seq, self.hidden_size)
            if collect_fp8_scales and self.fp8_o_input_scales is not None:
                self._copy_fp8_scale(self._fp8_o_scale_views[layer_idx], attn_o)
            if self.use_fp8_layers:
                if self.attn_o_fp8 is None:
                    raise RuntimeError("FP8 layer buffers are not initialized")
                fp8_layer = self.ptrs["fp8"]["layers"][layer_idx]
                kernels.quantize_bf16_to_fp8_e4m3fnuz_ptr(
                    attn_o.data_ptr(),
                    self._fp8_o_scale_views[layer_idx].data_ptr(),
                    self.attn_o_fp8.data_ptr(),
                    q_seq * self.hidden_size,
                    stream,
                )
                kernels.hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr(
                    self.attn_o_fp8.data_ptr(),
                    self._t(fp8_layer["o_w"]).data_ptr(),
                    self._fp8_o_scale_views[layer_idx].data_ptr(),
                    self._t(fp8_layer["o_w_scale"]).data_ptr(),
                    0,
                    self.attn_proj.data_ptr(),
                    q_seq,
                    self.hidden_size,
                    self.hidden_size,
                    stream,
                )
            else:
                kernels.hipblaslt_linear_bf16_ptr(
                    attn_o.data_ptr(),
                    self._t(layer["o_w"]).data_ptr(),
                    0,
                    self.attn_proj.data_ptr(),
                    q_seq,
                    self.hidden_size,
                    self.hidden_size,
                    stream,
                )
            if self.use_fp8_layers:
                if self.post_norm_fp8 is None:
                    raise RuntimeError("FP8 layer buffers are not initialized")
                fp8_layer = self.ptrs["fp8"]["layers"][layer_idx]
                kernels.residual_add_rms_norm_fp8_e4m3fnuz_plain_ptr(
                    self.hidden.data_ptr(),
                    self.attn_proj.data_ptr(),
                    self._t(layer["post_attn_norm_w"]).data_ptr(),
                    self.post_norm_fp8.data_ptr(),
                    self._fp8_gate_up_scale_views[layer_idx].data_ptr(),
                    q_seq,
                    self.hidden_size,
                    self.eps,
                    stream,
                )
                kernels.hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr(
                    self.post_norm_fp8.data_ptr(),
                    self._t(fp8_layer["gate_up_w"]).data_ptr(),
                    self._fp8_gate_up_scale_views[layer_idx].data_ptr(),
                    self._t(fp8_layer["gate_up_w_scale"]).data_ptr(),
                    0,
                    self.gate_up.data_ptr(),
                    q_seq,
                    2 * self.intermediate,
                    self.hidden_size,
                    stream,
                )
            else:
                kernels.residual_add_rms_norm_bf16_plain_ptr(
                    self.hidden.data_ptr(),
                    self.attn_proj.data_ptr(),
                    self._t(layer["post_attn_norm_w"]).data_ptr(),
                    self.post_norm.data_ptr(),
                    q_seq,
                    self.hidden_size,
                    self.eps,
                    stream,
                )
                if collect_fp8_scales and self.fp8_gate_up_input_scales is not None:
                    self._copy_fp8_scale(
                        self._fp8_gate_up_scale_views[layer_idx], self.post_norm[:q_seq]
                    )
                kernels.hipblaslt_linear_bf16_ptr(
                    self.post_norm.data_ptr(),
                    self._t(layer["gate_up_w"]).data_ptr(),
                    0,
                    self.gate_up.data_ptr(),
                    q_seq,
                    2 * self.intermediate,
                    self.hidden_size,
                    stream,
                )
            if self.use_fp8_layers:
                if self.act_fp8 is None:
                    raise RuntimeError("FP8 layer buffers are not initialized")
                fp8_layer = self.ptrs["fp8"]["layers"][layer_idx]
                kernels.silu_mul_merged_quantize_fp8_e4m3fnuz_ptr(
                    self.gate_up.data_ptr(),
                    self._fp8_down_scale_views[layer_idx].data_ptr(),
                    self.act_fp8.data_ptr(),
                    q_seq,
                    self.intermediate,
                    stream,
                )
                kernels.hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr(
                    self.act_fp8.data_ptr(),
                    self._t(fp8_layer["down_w"]).data_ptr(),
                    self._fp8_down_scale_views[layer_idx].data_ptr(),
                    self._t(fp8_layer["down_w_scale"]).data_ptr(),
                    0,
                    self.mlp_out.data_ptr(),
                    q_seq,
                    self.hidden_size,
                    self.intermediate,
                    stream,
                )
            else:
                kernels.silu_mul_merged_bf16_ptr(
                    self.gate_up.data_ptr(),
                    self.act.data_ptr(),
                    q_seq,
                    self.intermediate,
                    stream,
                )
                if collect_fp8_scales and self.fp8_down_input_scales is not None:
                    self._copy_fp8_scale(
                        self._fp8_down_scale_views[layer_idx], self.act[:q_seq]
                    )
                kernels.hipblaslt_linear_bf16_ptr(
                    self.act.data_ptr(),
                    self._t(layer["down_w"]).data_ptr(),
                    0,
                    self.mlp_out.data_ptr(),
                    q_seq,
                    self.hidden_size,
                    self.intermediate,
                    stream,
                )
            if self.use_fp8_layers:
                if layer_idx + 1 < self.num_layers:
                    next_layer = self.ptrs["layers"][layer_idx + 1]
                    kernels.residual_add_rms_norm_fp8_e4m3fnuz_plain_ptr(
                        self.hidden.data_ptr(),
                        self.mlp_out.data_ptr(),
                        self._t(next_layer["input_norm_w"]).data_ptr(),
                        self.norm_fp8.data_ptr(),
                        self._fp8_qkv_scale_views[layer_idx + 1].data_ptr(),
                        q_seq,
                        self.hidden_size,
                        self.eps,
                        stream,
                    )
                    qkv_input_fp8_ready = True
                elif self.use_fp8_lm_head:
                    if self.fp8_lm_head_input_scale is None or self.final_norm_fp8 is None:
                        raise RuntimeError("FP8 lm_head buffers are not initialized")
                    kernels.residual_add_rms_norm_fp8_e4m3fnuz_plain_ptr(
                        self.hidden.data_ptr(),
                        self.mlp_out.data_ptr(),
                        self._t(self.ptrs["final_norm_w"]).data_ptr(),
                        self.final_norm_fp8.data_ptr(),
                        self.fp8_lm_head_input_scale.data_ptr(),
                        q_seq,
                        self.hidden_size,
                        self.eps,
                        stream,
                    )
                    final_norm_fp8_ready = True
                else:
                    kernels.residual_add_bf16_ptr(
                        self.hidden.data_ptr(),
                        self.mlp_out.data_ptr(),
                        q_seq * self.hidden_size,
                        stream,
                    )
            else:
                kernels.residual_add_bf16_ptr(
                    self.hidden.data_ptr(),
                    self.mlp_out.data_ptr(),
                    q_seq * self.hidden_size,
                    stream,
                )

        if self.use_fp8_lm_head:
            if self.fp8_lm_head_input_scale is None or self.final_norm_fp8 is None:
                raise RuntimeError("FP8 lm_head buffers are not initialized")
            fp8 = self.ptrs["fp8"]
            if not final_norm_fp8_ready:
                kernels.rms_norm_fp8_e4m3fnuz_plain_ptr(
                    self.hidden.data_ptr(),
                    self._t(self.ptrs["final_norm_w"]).data_ptr(),
                    self.final_norm_fp8.data_ptr(),
                    self.fp8_lm_head_input_scale.data_ptr(),
                    q_seq,
                    self.hidden_size,
                    self.eps,
                    stream,
                )
            kernels.hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr(
                self.final_norm_fp8.data_ptr(),
                self._t(fp8["lm_head_w"]).data_ptr(),
                self.fp8_lm_head_input_scale.data_ptr(),
                self._t(fp8["lm_head_scale"]).data_ptr(),
                0,
                self.logits.data_ptr(),
                q_seq,
                self.vocab_size,
                self.hidden_size,
                stream,
            )
        else:
            kernels.rms_norm_bf16_plain_ptr(
                self.hidden.data_ptr(),
                self._t(self.ptrs["final_norm_w"]).data_ptr(),
                self.final_norm.data_ptr(),
                q_seq,
                self.hidden_size,
                self.eps,
                stream,
            )
            kernels.hipblaslt_linear_bf16_ptr(
                self.final_norm.data_ptr(),
                self._t(self.ptrs["lm_head_w"]).data_ptr(),
                0,
                self.logits.data_ptr(),
                q_seq,
                self.vocab_size,
                self.hidden_size,
                stream,
            )
        if sync:
            torch.cuda.synchronize()
        return self.logits[:q_seq]

    def calibrate_fp8_lm_head(self, input_ids, *, pos_start: int = 0) -> float:
        """Calibrate static final-norm activation scale for FP8 lm_head."""
        import torch

        if not self.use_fp8_lm_head:
            raise RuntimeError("calibrate_fp8_lm_head requires use_fp8_lm_head=True")
        was_lm_enabled = self.use_fp8_lm_head
        was_layers_enabled = self.use_fp8_layers
        try:
            self.use_fp8_lm_head = False
            self.use_fp8_layers = False
            self.forward_ids(input_ids, pos_start=pos_start, causal=input_ids.numel() > 1)
            q_seq = int(input_ids.numel())
            amax = torch.clamp(self.final_norm[:q_seq].float().abs().max(), min=1.0e-8)
            assert self.fp8_lm_head_input_scale is not None
            self.fp8_lm_head_input_scale.copy_((amax / 240.0).reshape(1))
        finally:
            self.use_fp8_layers = was_layers_enabled
            self.use_fp8_lm_head = was_lm_enabled
        return float(self.fp8_lm_head_input_scale.item())

    def calibrate_fp8_layers(self, input_ids, *, pos_start: int = 0) -> dict[str, float]:
        """Calibrate static activation scales for FP8 layer GEMMs.

        This runs the BF16 layer path and records the exact GEMM input ranges:
        input norm -> qkv, attention output -> o_proj, post-attention norm ->
        gate/up, and SiLU product -> down_proj.
        """
        import torch

        if not self.use_fp8_layers:
            raise RuntimeError("calibrate_fp8_layers requires use_fp8_layers=True")
        if input_ids.dim() == 2:
            if input_ids.shape[0] != 1:
                raise ValueError("only batch size 1 is supported by the owned ROCm path")
            input_ids = input_ids[0]
        q_seq = int(input_ids.numel())
        if q_seq <= 0 or q_seq > self.max_q_seq:
            raise ValueError(f"q_seq={q_seq} outside 1..max_q_seq={self.max_q_seq}")

        was_layers = self.use_fp8_layers
        was_lm = self.use_fp8_lm_head
        try:
            self.use_fp8_layers = False
            self.use_fp8_lm_head = False
            self.input_ids_buf[:q_seq].copy_(
                input_ids.reshape(-1).to(dtype=self.input_ids_buf.dtype)
            )
            self._forward_prepared(
                q_seq,
                pos_start=pos_start,
                causal=q_seq > 1,
                collect_fp8_scales=True,
            )
        finally:
            self.use_fp8_layers = was_layers
            self.use_fp8_lm_head = was_lm

        assert self.fp8_qkv_input_scales is not None
        assert self.fp8_o_input_scales is not None
        assert self.fp8_gate_up_input_scales is not None
        assert self.fp8_down_input_scales is not None
        return {
            "qkv_mean": float(self.fp8_qkv_input_scales.mean().item()),
            "qkv_max": float(self.fp8_qkv_input_scales.max().item()),
            "o_mean": float(self.fp8_o_input_scales.mean().item()),
            "o_max": float(self.fp8_o_input_scales.max().item()),
            "gate_up_mean": float(self.fp8_gate_up_input_scales.mean().item()),
            "gate_up_max": float(self.fp8_gate_up_input_scales.max().item()),
            "down_mean": float(self.fp8_down_input_scales.mean().item()),
            "down_max": float(self.fp8_down_input_scales.max().item()),
        }

    def forward_ids(self, input_ids, *, pos_start: int = 0, causal: bool | None = None):
        """Return logits for ``input_ids`` and update the owned KV cache."""
        if input_ids.dim() == 2:
            if input_ids.shape[0] != 1:
                raise ValueError("only batch size 1 is supported by the owned ROCm path")
            input_ids = input_ids[0]
        q_seq = int(input_ids.numel())
        if q_seq <= 0 or q_seq > self.max_q_seq:
            raise ValueError(f"q_seq={q_seq} outside 1..max_q_seq={self.max_q_seq}")
        self.input_ids_buf[:q_seq].copy_(input_ids.reshape(-1).to(dtype=self.input_ids_buf.dtype))
        return self._forward_prepared(q_seq, pos_start=pos_start, causal=causal)

    def capture_decode_graph(self, pos_start: int):
        """Capture a q_seq=1 decode graph for a fixed position.

        The caller updates ``input_ids_buf[0]`` before replaying the returned
        graph. Production code should own a small graph
        table keyed by decode position or bucket.
        """
        import torch

        if pos_start < 0 or pos_start >= self.max_seq:
            raise ValueError("pos_start outside max_seq")

        self.input_ids_buf[:1].fill_(0)
        for _ in range(3):
            self._forward_prepared(1, pos_start=pos_start, causal=False)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            self._forward_prepared(1, pos_start=pos_start, causal=False, sync=False)
        torch.cuda.synchronize()
        return graph

    def capture_decode_graph_table(self, start_pos: int, count: int) -> dict[int, Any]:
        """Capture q_seq=1 decode graphs for ``start_pos..start_pos+count-1``."""
        if count <= 0:
            raise ValueError("count must be positive")
        graphs = {}
        for pos in range(int(start_pos), int(start_pos) + int(count)):
            graphs[pos] = self.capture_decode_graph(pos)
        self._decode_graphs.update(graphs)
        return graphs

    def replay_decode_graph(self, token_id: int, pos_start: int):
        """Replay a captured q_seq=1 decode graph for ``token_id`` at ``pos_start``."""
        import torch

        try:
            graph = self._decode_graphs[int(pos_start)]
        except KeyError as exc:
            raise KeyError(f"decode graph for pos_start={pos_start} is not captured") from exc
        self.input_ids_buf[:1].fill_(int(token_id))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        return self.logits[:1]

    def capture_prefill_graph(self, bucket: int):
        """Capture a fresh-KV prefill graph for a fixed prompt bucket.

        The graph reads ``input_ids_buf[:bucket]`` and writes the owned KV cache,
        hidden scratch, and logits for all bucket rows. For real prompts shorter
        than the bucket, callers may pad the static input buffer with the last
        real token, but the ROCm FP8 path is only parity-tested for exact
        buckets because hipBLASLt FP8 numerics can vary with the GEMM M axis.
        """
        import torch

        bucket = int(bucket)
        if bucket <= 0 or bucket > self.max_q_seq:
            raise ValueError(f"bucket={bucket} outside 1..max_q_seq={self.max_q_seq}")

        cached = self._prefill_graphs.get(bucket)
        if cached is not None:
            return cached

        self.input_ids_buf[:bucket].fill_(0)
        for _ in range(2):
            self._forward_prepared(bucket, pos_start=0, causal=True)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            self._forward_prepared(bucket, pos_start=0, causal=True, sync=False)
        torch.cuda.synchronize()
        self._prefill_graphs[bucket] = graph
        return graph

    def capture_prefill_graph_table(self, buckets=(32, 64, 128, 256, 512, 1024)):
        """Capture fresh-KV prefill graphs for the requested bucket sizes."""
        graphs = {}
        for bucket in buckets:
            b = int(bucket)
            if 1 <= b <= self.max_q_seq:
                graphs[b] = self.capture_prefill_graph(b)
        return graphs

    def replay_prefill_graph(self, input_ids, *, bucket: int | None = None):
        """Replay a captured fresh-KV prefill graph and return real-row logits."""
        import torch

        if input_ids.dim() == 2:
            if input_ids.shape[0] != 1:
                raise ValueError("only batch size 1 is supported by the owned ROCm path")
            input_ids = input_ids[0]
        real_seq = int(input_ids.numel())
        if real_seq <= 0 or real_seq > self.max_q_seq:
            raise ValueError(f"real_seq={real_seq} outside 1..max_q_seq={self.max_q_seq}")
        if bucket is None:
            bucket = real_seq
        bucket = int(bucket)
        if bucket < real_seq or bucket > self.max_q_seq:
            raise ValueError("prefill bucket must satisfy real_seq <= bucket <= max_q_seq")
        graph = self.capture_prefill_graph(bucket)
        self.input_ids_buf[:real_seq].copy_(
            input_ids.reshape(-1).to(dtype=self.input_ids_buf.dtype)
        )
        if real_seq < bucket:
            self.input_ids_buf[real_seq:bucket].fill_(int(input_ids[-1].item()))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        return self.logits[:real_seq]

    def generate(
        self,
        prompt: str | None = None,
        *,
        max_new_tokens: int = 1,
        return_ids: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        import torch

        if prompt is not None:
            self.set_prompt(prompt)
        if self._prompt_ids is None:
            raise ValueError("prompt is required before generate()")

        input_ids = self._prompt_ids[0].to(self.device)
        if input_ids.numel() > self.max_q_seq:
            raise ValueError("prompt prefill exceeds max_q_seq for this frontend")
        tokens = [int(x) for x in input_ids.tolist()]

        with torch.inference_mode():
            t0 = time.perf_counter()
            logits = self.forward_ids(input_ids, pos_start=0, causal=input_ids.numel() > 1)
            pos = int(input_ids.numel())
            for _ in range(int(max_new_tokens)):
                next_id = int(logits[-1].float().argmax().item())
                tokens.append(next_id)
                if len(tokens) - input_ids.numel() >= int(max_new_tokens):
                    break
                self.input_ids_buf[:1].fill_(next_id)
                logits = self._forward_prepared(1, pos_start=pos, causal=False)
                pos += 1
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        out_ids = torch.tensor(tokens, device=self.device, dtype=input_ids.dtype).unsqueeze(0)
        self._generated_ids = out_ids
        self.latency_records.append(elapsed_ms)
        new_tokens = max(0, len(tokens) - int(input_ids.numel()))
        result: dict[str, Any] = {
            "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
            "new_tokens": new_tokens,
            "generate_ms": elapsed_ms,
            "tok_per_s": (1000.0 * new_tokens / elapsed_ms) if elapsed_ms else 0.0,
            "prompt_tokens": int(input_ids.numel()),
        }
        if return_ids:
            result["output_ids"] = out_ids
        return result

    def infer(self, request: dict | str, debug: bool = False) -> dict[str, Any]:
        if isinstance(request, str):
            prompt = request
            max_new_tokens = 1
        else:
            prompt = request.get("prompt") or request.get("text")
            max_new_tokens = int(request.get("max_new_tokens", 1))
        if prompt is None and self._prompt_ids is None:
            raise ValueError("request must include 'prompt' on first call")
        result = self.generate(prompt, max_new_tokens=max_new_tokens)
        if debug:
            result["debug"] = {
                "backend": "rocm_owned_bf16",
                "checkpoint": self.checkpoint_path,
                "config": self.config_summary,
                "load_s": self.load_s,
            }
        return result

    def get_latency_stats(self) -> dict[str, float]:
        if not self.latency_records:
            return {"count": 0}
        vals = list(self.latency_records)
        return {
            "count": len(vals),
            "mean_ms": sum(vals) / len(vals),
            "min_ms": min(vals),
            "max_ms": max(vals),
        }
