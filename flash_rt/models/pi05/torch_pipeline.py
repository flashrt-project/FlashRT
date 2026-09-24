"""Pi0.5 BF16 tensor execution shared by torch operation providers.

This module owns vision, prefix encoding and flow-matching semantics.
Targets supply tensor operations, GEMM and attention providers; they never
supply a second model forward or layer/denoising traversal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


VIS_L, VIS_D, VIS_H, VIS_NH, VIS_HD = 27, 1152, 4304, 16, 72
ENC_L, ENC_D, ENC_H, ENC_NH, ENC_HD = 18, 2048, 16384, 8, 256
DEC_L, DEC_D, DEC_H, DEC_NH, DEC_HD = 18, 1024, 4096, 8, 256
PATCHES_PER_VIEW = 256
ACTION_DIM = 32


class Pi05TorchPipeline:
    def __init__(
        self,
        weights: dict,
        ops,
        gemm,
        attn,
        *,
        num_views: int = 3,
        max_prompt_len: int = 200,
        chunk_size: int = 10,
        num_steps: int = 10,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        precompute_modulation: bool = True,
        compact_encoder: bool = True,
    ):
        if dtype != torch.bfloat16:
            raise TypeError("The Pi0.5 tensor pipeline supports BF16 only")
        if (
            num_views not in (2, 3)
            or max_prompt_len < 1
            or chunk_size < 1
            or num_steps < 1
        ):
            raise ValueError(
                f"Unsupported π0.5 profile: views={num_views}, "
                f"max_prompt_len={max_prompt_len}, horizon={chunk_size}, "
                f"steps={num_steps}"
            )
        self.w, self.ops, self.gemm, self.attn = weights, ops, gemm, attn
        self.device = torch.device(device)
        self.num_views = num_views
        self.max_prompt_len = max_prompt_len
        self.chunk_size = chunk_size
        self.num_steps = num_steps
        self.dtype = dtype
        self.prefix_capacity = num_views * PATCHES_PER_VIEW + max_prompt_len
        self.precompute_modulation = precompute_modulation
        self.compact_encoder = compact_encoder
        self.buf = self._allocate_buffers()
        self._init_rope()
        self._decoder_modulations = None
        if self.precompute_modulation:
            self._decoder_modulations = torch.empty(
                num_steps,
                2 * DEC_L + 1,
                3 * DEC_D,
                dtype=dtype,
                device=self.device,
            )
            with torch.inference_mode():
                for step in range(num_steps):
                    self.gemm.linear(
                        self.buf["time_tmp"],
                        self.w["decoder_time_embeds"][step : step + 1],
                        self.w["decoder_time_mlp_in_w"],
                        self.w["decoder_time_mlp_in_b"],
                    )
                    self.ops.silu(self.buf["time_tmp"], self.buf["time_tmp"])
                    self.gemm.linear(
                        self.buf["time_cond"],
                        self.buf["time_tmp"],
                        self.w["decoder_time_mlp_out_w"],
                        self.w["decoder_time_mlp_out_b"],
                    )
                    self.ops.silu(self.buf["time_cond"], self.buf["time_cond"])
                    for index in range(DEC_L):
                        torch.addmm(
                            self.w["decoder_pre_attn_norm_mod_b"][index],
                            self.buf["time_cond"],
                            self.w["decoder_pre_attn_norm_mod_w"][index],
                            out=self._decoder_modulations[step, 2 * index : 2 * index + 1],
                        )
                        torch.addmm(
                            self.w["decoder_pre_ffn_norm_mod_b"][index],
                            self.buf["time_cond"],
                            self.w["decoder_pre_ffn_norm_mod_w"][index],
                            out=self._decoder_modulations[step, 2 * index + 1 : 2 * index + 2],
                        )
                    torch.addmm(
                        self.w["decoder_final_norm_mod_b"],
                        self.buf["time_cond"],
                        self.w["decoder_final_norm_mod_w"],
                        out=self._decoder_modulations[step, -1:],
                    )
        self.probes: dict[str, torch.Tensor] = {}
        self._capture_probes = False
        self._graph = None
        self._graph_prompt_len = None
        self._decoder_only_graph = None
        self._decoder_only_graph_prompt_len = None
        self._current_prompt_len = None

    @staticmethod
    def _patch_position(out, patch, position) -> None:
        out.copy_(patch.flatten(2).transpose(1, 2))
        out.add_(position.unsqueeze(0))

    def _allocate_buffers(self) -> dict[str, torch.Tensor]:
        empty = lambda *shape: torch.empty(*shape, dtype=self.dtype, device=self.device)
        prefix = self.prefix_capacity
        chunk = self.chunk_size
        buffers = {
            "input_images": empty(self.num_views, 224, 224, 3),
            "input_prompt": empty(self.max_prompt_len, ENC_D),
            "input_noise": empty(chunk, ACTION_DIM),
            "vision_x": empty(self.num_views, PATCHES_PER_VIEW, VIS_D),
            "vision_norm": empty(self.num_views, PATCHES_PER_VIEW, VIS_D),
            "vision_qkv": empty(self.num_views, PATCHES_PER_VIEW, 3 * VIS_D),
            "vision_attn": empty(self.num_views, PATCHES_PER_VIEW, VIS_D),
            "vision_hidden": empty(self.num_views, PATCHES_PER_VIEW, VIS_H),
            "vision_out": empty(self.num_views, PATCHES_PER_VIEW, VIS_D),
            "encoder_x": empty(prefix, ENC_D),
            "encoder_norm": empty(prefix, ENC_D),
            "encoder_qkv": empty(prefix, (ENC_NH + 2) * ENC_HD),
            "encoder_attn": empty(prefix, ENC_D),
            "encoder_hidden": empty(prefix, ENC_H),
            "encoder_out": empty(prefix, ENC_D),
            "encoder_k": empty(ENC_L, prefix + chunk, 1, ENC_HD),
            "encoder_v": empty(ENC_L, prefix + chunk, 1, ENC_HD),
            "decoder_x": empty(chunk, DEC_D),
            "decoder_norm": empty(chunk, DEC_D),
            "decoder_qkv": empty(chunk, (DEC_NH + 2) * DEC_HD),
            "decoder_attn": empty(chunk, ENC_D),
            "decoder_out": empty(chunk, DEC_D),
            "decoder_hidden": empty(chunk, DEC_H),
            "decoder_modulation": empty(1, 3 * DEC_D),
            "time_tmp": empty(1, DEC_D),
            "time_cond": empty(1, DEC_D),
            "action": empty(chunk, ACTION_DIM),
            "noise": empty(chunk, ACTION_DIM),
        }
        # Split/reference FFN GEMMs write directly into strided halves of the
        # merged workspace, so provider switching does not require duplicate
        # gate and up buffers.
        buffers["encoder_gate_up"] = empty(prefix, 2 * ENC_H)
        buffers["decoder_gate_up"] = empty(chunk, 2 * DEC_H)
        return buffers

    def _init_rope(self) -> None:
        max_position = self.prefix_capacity + self.chunk_size
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, ENC_HD, 2, dtype=torch.float32, device=self.device) / ENC_HD)
        )
        phase = torch.arange(max_position, dtype=torch.float32, device=self.device)[:, None] * inv_freq
        self._rope_cos = phase.cos().to(self.dtype)
        self._rope_sin = phase.sin().to(self.dtype)
        self._positions = torch.arange(max_position, device=self.device)

    def _save(self, key: str, value: torch.Tensor) -> None:
        if self._capture_probes:
            self.probes[key] = value.detach().clone()

    def _rope(self, value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        pairs = value.view(value.shape[0], value.shape[1], ENC_HD // 2, 2)
        real, imag = pairs.unbind(dim=-1)
        cos = self._rope_cos[positions].unsqueeze(1)
        sin = self._rope_sin[positions].unsqueeze(1)
        return torch.stack(
            (real * cos - imag * sin, imag * cos + real * sin), dim=-1
        ).reshape_as(value)

    def _vision(self, images_nhwc: torch.Tensor) -> None:
        b = self.buf
        patch = F.conv2d(
            images_nhwc.permute(0, 3, 1, 2),
            self.w["vision_patch_embedding_w"],
            self.w["vision_patch_embedding_b"],
            stride=14,
        )
        self._save("patch", patch)
        self._patch_position(
            b["vision_x"], patch, self.w["vision_position_embedding"]
        )

        for index in range(VIS_L):
            self.ops.layer_norm(
                b["vision_norm"],
                b["vision_x"],
                self.w["vision_pre_attn_norm_w"][index],
                self.w["vision_pre_attn_norm_b"][index],
            )
            self.gemm.linear(
                b["vision_qkv"],
                b["vision_norm"],
                self.w["vision_attn_qkv_w"][index],
                self.w["vision_attn_qkv_b"][index],
            )
            context = self.attn.vision(b["vision_qkv"], out=b["vision_norm"])
            self.gemm.linear(
                b["vision_attn"], context, self.w["vision_attn_o_w"][index],
                self.w["vision_attn_o_b"][index],
            )
            self.ops.residual(b["vision_x"], b["vision_attn"], b["vision_x"])
            self.ops.layer_norm(
                b["vision_norm"],
                b["vision_x"],
                self.w["vision_pre_ffn_norm_w"][index],
                self.w["vision_pre_ffn_norm_b"][index],
            )
            self.gemm.linear(
                b["vision_hidden"], b["vision_norm"],
                self.w["vision_ffn_up_w"][index],
                self.w["vision_ffn_up_b"][index],
            )
            self.ops.gelu(b["vision_hidden"], b["vision_hidden"])
            self.gemm.linear(
                b["vision_out"], b["vision_hidden"],
                self.w["vision_ffn_down_w"][index],
                self.w["vision_ffn_down_b"][index],
            )
            self.ops.residual(b["vision_x"], b["vision_out"], b["vision_x"])
            if index in (0, 13, 26):
                self._save(f"vision_l{index}", b["vision_x"])

    def _encoder(
        self,
        prompt_embeds: torch.Tensor,
        prompt_len: int,
        encoder_k: torch.Tensor | None = None,
        encoder_v: torch.Tensor | None = None,
    ) -> int:
        b = self.buf
        encoder_k = b["encoder_k"] if encoder_k is None else encoder_k
        encoder_v = b["encoder_v"] if encoder_v is None else encoder_v
        self.ops.layer_norm(
            b["vision_norm"],
            b["vision_x"],
            self.w["vision_final_norm_w"],
            self.w["vision_final_norm_b"],
            eps=1e-6,
        )
        projected = b["encoder_x"][: self.num_views * PATCHES_PER_VIEW].view(
            self.num_views, PATCHES_PER_VIEW, ENC_D
        )
        self.gemm.linear(
            projected,
            b["vision_norm"],
            self.w["encoder_multi_modal_projector_w"],
            self.w["encoder_multi_modal_projector_b"],
        )
        self._save("projector", projected)
        image_tokens = self.num_views * PATCHES_PER_VIEW
        if not self.compact_encoder:
            b["encoder_x"][image_tokens:].zero_()
        b["encoder_x"][image_tokens : image_tokens + prompt_len].copy_(prompt_embeds)
        valid_prefix = image_tokens + prompt_len
        encoder_rows = valid_prefix if self.compact_encoder else self.prefix_capacity
        encoder_x = b["encoder_x"][:encoder_rows]
        encoder_norm = b["encoder_norm"][:encoder_rows]
        encoder_qkv = b["encoder_qkv"][:encoder_rows]
        encoder_attn = b["encoder_attn"][:encoder_rows]
        encoder_gate_up = b["encoder_gate_up"][:encoder_rows]
        encoder_gate = encoder_gate_up[:, :ENC_H]
        encoder_up = encoder_gate_up[:, ENC_H:]
        encoder_hidden = b["encoder_hidden"][:encoder_rows]
        encoder_out = b["encoder_out"][:encoder_rows]
        positions = self._positions[:encoder_rows]
        attention_end = encoder_rows

        self.ops.rms_norm(encoder_norm, encoder_x)
        for index in range(ENC_L):
            self.gemm.linear(
                encoder_qkv, encoder_norm, self.w["encoder_attn_qkv_w"][index]
            )
            if self.ops.fused_rope:
                q = encoder_out.view(encoder_rows, ENC_NH, ENC_HD)
                k = encoder_k[index, :encoder_rows]
                self.ops.qkv_rope(
                    q,
                    k,
                    encoder_v[index, :encoder_rows],
                    encoder_qkv,
                    self._rope_cos,
                    self._rope_sin,
                    0,
                )
            else:
                q_flat, k_flat, v_flat = torch.split(
                    encoder_qkv, (ENC_NH * ENC_HD, ENC_HD, ENC_HD), dim=-1
                )
                q = self._rope(
                    q_flat.view(encoder_rows, ENC_NH, ENC_HD), positions)
                k = self._rope(k_flat.view(encoder_rows, 1, ENC_HD), positions)
                encoder_k[index, :encoder_rows].copy_(k)
                encoder_v[index, :encoder_rows].copy_(
                    v_flat.view(encoder_rows, 1, ENC_HD)
                )
            if index == ENC_L - 1:
                self._save("enc_l17_k", k)
                self._save("enc_l17_v", encoder_v[index, :encoder_rows])
                break
            context = self.attn.gqa(
                q[:attention_end],
                encoder_k[index, :attention_end],
                encoder_v[index, :attention_end],
                valid_prefix=valid_prefix,
                prefix_capacity=self.prefix_capacity,
                out=encoder_out,
            )
            self.gemm.linear(
                encoder_attn, context, self.w["encoder_attn_o_w"][index]
            )
            self.ops.residual_rms(
                encoder_x, encoder_norm, encoder_attn, encoder_x
            )
            gate_up_weight = self.w["encoder_ffn_gate_up_w"][index]
            if self.ops.merged_encoder_ffn:
                self.gemm.linear(
                    encoder_gate_up, encoder_norm, gate_up_weight)
                self.ops.gelu_mul_merged(encoder_hidden, encoder_gate_up)
            else:
                self.gemm.linear(
                    encoder_gate, encoder_norm, gate_up_weight[:, :ENC_H])
                self.gemm.linear(
                    encoder_up, encoder_norm, gate_up_weight[:, ENC_H:])
                self.ops.gelu_mul(encoder_hidden, encoder_gate, encoder_up)
            self.gemm.linear(
                encoder_out, encoder_hidden, self.w["encoder_ffn_down_w"][index]
            )
            self.ops.residual_rms(
                encoder_x, encoder_norm, encoder_out, encoder_x
            )
            if index in (0, 8):
                self._save(f"enc_l{index}", encoder_x)
        return valid_prefix

    def _decoder(
        self,
        valid_prefix: int,
        encoder_k: torch.Tensor | None = None,
        encoder_v: torch.Tensor | None = None,
    ) -> None:
        b = self.buf
        encoder_k = b["encoder_k"] if encoder_k is None else encoder_k
        encoder_v = b["encoder_v"] if encoder_v is None else encoder_v
        positions = self._positions[
            valid_prefix : valid_prefix + self.chunk_size]
        suffix_start = (
            valid_prefix if self.compact_encoder else self.prefix_capacity)
        kv_end = suffix_start + self.chunk_size
        for step in range(self.num_steps):
            modulations = None
            if self.precompute_modulation:
                modulations = self._decoder_modulations[step]
            else:
                self.gemm.linear(
                    b["time_tmp"],
                    self.w["decoder_time_embeds"][step : step + 1],
                    self.w["decoder_time_mlp_in_w"],
                    self.w["decoder_time_mlp_in_b"],
                )
                self.ops.silu(b["time_tmp"], b["time_tmp"])
                self.gemm.linear(
                    b["time_cond"],
                    b["time_tmp"],
                    self.w["decoder_time_mlp_out_w"],
                    self.w["decoder_time_mlp_out_b"],
                )
                self.ops.silu(b["time_cond"], b["time_cond"])
            self.gemm.linear(
                b["decoder_x"],
                b["noise"],
                self.w["decoder_action_in_proj_w"],
                self.w["decoder_action_in_proj_b"],
            )

            if modulations is None:
                gate = self.ops.adarms(
                    b["decoder_norm"],
                    b["decoder_x"],
                    b["time_cond"],
                    self.w["decoder_pre_attn_norm_mod_w"][0],
                    self.w["decoder_pre_attn_norm_mod_b"][0],
                    modulation=b["decoder_modulation"],
                )
            else:
                gate = self.ops.adarms(
                    b["decoder_norm"], b["decoder_x"], None, None, None,
                    modulation=modulations[0:1],
                )
            for index in range(DEC_L):
                self.gemm.linear(
                    b["decoder_qkv"],
                    b["decoder_norm"],
                    self.w["decoder_attn_qkv_w"][index],
                )
                suffix = slice(suffix_start, suffix_start + self.chunk_size)
                if self.ops.fused_rope:
                    q = b["decoder_attn"].view(
                        self.chunk_size, DEC_NH, DEC_HD)
                    k = encoder_k[index, suffix]
                    self.ops.qkv_rope(
                        q,
                        k,
                        encoder_v[index, suffix],
                        b["decoder_qkv"],
                        self._rope_cos,
                        self._rope_sin,
                        valid_prefix,
                    )
                else:
                    q_flat, k_flat, v_flat = torch.split(
                        b["decoder_qkv"],
                        (DEC_NH * DEC_HD, DEC_HD, DEC_HD),
                        dim=-1,
                    )
                    q = self._rope(
                        q_flat.view(self.chunk_size, DEC_NH, DEC_HD), positions)
                    k = self._rope(
                        k_flat.view(self.chunk_size, 1, DEC_HD), positions)
                    encoder_k[index, suffix].copy_(k)
                    encoder_v[index, suffix].copy_(
                        v_flat.view(self.chunk_size, 1, DEC_HD)
                    )
                context = self.attn.gqa(
                    q,
                    encoder_k[index, :kv_end],
                    encoder_v[index, :kv_end],
                    valid_prefix=valid_prefix,
                    prefix_capacity=self.prefix_capacity,
                    out=b["decoder_attn"],
                )
                self.gemm.linear(
                    b["decoder_out"], context, self.w["decoder_attn_o_w"][index]
                )
                if modulations is None:
                    gate = self.ops.residual_adarms(
                        b["decoder_x"],
                        b["decoder_norm"],
                        b["decoder_out"],
                        b["decoder_x"],
                        gate,
                        b["time_cond"],
                        self.w["decoder_pre_ffn_norm_mod_w"][index],
                        self.w["decoder_pre_ffn_norm_mod_b"][index],
                        modulation=b["decoder_modulation"],
                    )
                else:
                    gate = self.ops.residual_adarms(
                        b["decoder_x"], b["decoder_norm"], b["decoder_out"],
                        b["decoder_x"], gate, None, None, None,
                        modulation=modulations[2 * index + 1 : 2 * index + 2],
                )
                gate_up_weight = self.w["decoder_ffn_gate_up_w"][index]
                decoder_gate = b["decoder_gate_up"][:, :DEC_H]
                decoder_up = b["decoder_gate_up"][:, DEC_H:]
                if self.ops.merged_decoder_ffn:
                    self.gemm.linear(
                        b["decoder_gate_up"], b["decoder_norm"],
                        gate_up_weight)
                    self.ops.gelu_mul_merged(
                        b["decoder_hidden"], b["decoder_gate_up"])
                else:
                    self.gemm.linear(
                        decoder_gate, b["decoder_norm"],
                        gate_up_weight[:, :DEC_H]
                    )
                    self.gemm.linear(
                        decoder_up, b["decoder_norm"],
                        gate_up_weight[:, DEC_H:]
                    )
                    self.ops.gelu_mul(
                        b["decoder_hidden"], decoder_gate, decoder_up)
                self.gemm.linear(
                    b["decoder_out"],
                    b["decoder_hidden"],
                    self.w["decoder_ffn_down_w"][index],
                )
                if index + 1 < DEC_L:
                    next_modulation = (
                        None if modulations is None
                        else modulations[2 * index + 2 : 2 * index + 3]
                    )
                    next_weight = self.w["decoder_pre_attn_norm_mod_w"][index + 1]
                    next_bias = self.w["decoder_pre_attn_norm_mod_b"][index + 1]
                else:
                    next_modulation = None if modulations is None else modulations[-1:]
                    next_weight = self.w["decoder_final_norm_mod_w"]
                    next_bias = self.w["decoder_final_norm_mod_b"]
                gate = self.ops.residual_adarms(
                    b["decoder_x"],
                    b["decoder_norm"],
                    b["decoder_out"],
                    b["decoder_x"],
                    gate,
                    b["time_cond"] if modulations is None else None,
                    next_weight,
                    next_bias,
                    modulation=(
                        b["decoder_modulation"]
                        if modulations is None else next_modulation
                    ),
                )
                if step in (0, 4, 9) and index in (0, 8, 17):
                    self._save(f"dec_s{step}_l{index}", b["decoder_x"])

            if not self.gemm.linear_residual(
                b["noise"],
                b["decoder_norm"],
                self.w["decoder_action_out_proj_w"],
                self.w["decoder_action_out_proj_b"],
            ):
                self.gemm.linear(
                    b["action"],
                    b["decoder_norm"],
                    self.w["decoder_action_out_proj_w"],
                    self.w["decoder_action_out_proj_b"],
                )
                b["noise"].add_(b["action"])
            self._save(f"noise_s{step}", b["noise"])

    def _run_static(self, prompt_len: int) -> None:
        self.buf["noise"].copy_(self.buf["input_noise"])
        self._vision(self.buf["input_images"])
        valid_prefix = self._encoder(self.buf["input_prompt"][:prompt_len], prompt_len)
        self._decoder(valid_prefix)

    def _run_decoder_static(self, valid_prefix: int) -> None:
        """Run only denoising against the most recently encoded K/V prefix."""
        self.buf["noise"].copy_(self.buf["input_noise"])
        self._decoder(valid_prefix)

    def _record_decoder_only_graph(self, prompt_len: int) -> None:
        """Capture the decoder-only half of the current static-shape pipeline."""
        valid_prefix = self.num_views * PATCHES_PER_VIEW + prompt_len
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._run_decoder_static(valid_prefix)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_decoder_static(valid_prefix)
        torch.cuda.synchronize()
        self._decoder_only_graph = graph
        self._decoder_only_graph_prompt_len = prompt_len

    def record_graph(self, prompt_len: int) -> None:
        if self._capture_probes:
            raise RuntimeError("Cannot capture a graph while debug probes are enabled")
        if self.device.type != "cuda":
            raise RuntimeError("Graph capture requires a CUDA/ROCm device")
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._run_static(prompt_len)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_static(prompt_len)
        torch.cuda.synchronize()
        self._graph = graph
        self._graph_prompt_len = prompt_len

    def _validate_inputs(
        self,
        images_nhwc: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_len: int,
        noise: torch.Tensor,
    ) -> None:
        if images_nhwc.shape != (self.num_views, 224, 224, 3):
            raise ValueError(f"Unsupported image shape {tuple(images_nhwc.shape)}")
        if (
            not 0 < prompt_len <= self.max_prompt_len
            or prompt_embeds.shape != (prompt_len, ENC_D)
        ):
            raise ValueError(
                f"Unsupported prompt shape {tuple(prompt_embeds.shape)}, len={prompt_len}, "
                f"capacity={self.max_prompt_len}"
            )
        if noise.shape not in ((self.chunk_size, ACTION_DIM), (1, self.chunk_size, ACTION_DIM)):
            raise ValueError(f"Unsupported noise shape {tuple(noise.shape)}")
        for name, value in (("images", images_nhwc), ("prompt", prompt_embeds), ("noise", noise)):
            if value.device.type != self.device.type or value.dtype != self.dtype:
                raise TypeError(f"{name} must be a BF16 tensor on {self.device}")

    @torch.inference_mode()
    def forward_with_inputs(
        self,
        images_nhwc: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_len: int,
        noise: torch.Tensor,
        *,
        capture_probes: bool = False,
        use_graph: bool = False,
    ) -> torch.Tensor:
        self._validate_inputs(images_nhwc, prompt_embeds, prompt_len, noise)

        if capture_probes and use_graph:
            raise ValueError("Debug probes are only available in no-graph mode")
        if capture_probes:
            self.probes = {}
        self._capture_probes = capture_probes
        try:
            if use_graph:
                self.buf["input_images"].copy_(images_nhwc)
                self.buf["input_prompt"][:prompt_len].copy_(prompt_embeds)
                self.buf["input_noise"].copy_(
                    noise.reshape(self.chunk_size, ACTION_DIM))
                if self._graph_prompt_len != prompt_len:
                    self.record_graph(prompt_len)
                self._graph.replay()
            else:
                self.buf["noise"].copy_(
                    noise.reshape(self.chunk_size, ACTION_DIM))
                self._vision(images_nhwc)
                valid_prefix = self._encoder(prompt_embeds, prompt_len)
                self._decoder(valid_prefix)
            self._current_prompt_len = prompt_len
            self._save("final_raw_action", self.buf["noise"])
        finally:
            self._capture_probes = False
        return self.buf["noise"].unsqueeze(0)

    @torch.inference_mode()
    def forward_decode_only(
        self,
        noise: torch.Tensor,
        *,
        capture_probes: bool = False,
        use_graph: bool = False,
    ) -> torch.Tensor:
        """Denoise with the K/V prefix produced by the last full forward.

        This is the tensor-pipeline equivalent of the CDNA4
        ``forward_decode_only`` path. Images and prompt embeddings are not
        consumed: the method intentionally reuses the last full forward's
        per-layer encoder K/V cache. A full forward is therefore required
        before the first call and whenever the prompt/context is invalidated.
        """
        if self._current_prompt_len is None:
            raise RuntimeError(
                "forward_decode_only requires a preceding full forward")
        if noise.shape not in (
            (self.chunk_size, ACTION_DIM),
            (1, self.chunk_size, ACTION_DIM),
        ):
            raise ValueError(f"Unsupported noise shape {tuple(noise.shape)}")
        if noise.device.type != self.device.type or noise.dtype != self.dtype:
            raise TypeError("noise must be a BF16 tensor on the pipeline device")
        if capture_probes and use_graph:
            raise ValueError("Debug probes are only available in no-graph mode")
        if capture_probes:
            self.probes = {}
        self._capture_probes = capture_probes
        try:
            if use_graph:
                self.buf["input_noise"].copy_(
                    noise.reshape(self.chunk_size, ACTION_DIM))
                if (
                    self._decoder_only_graph_prompt_len
                    != self._current_prompt_len
                ):
                    self._record_decoder_only_graph(self._current_prompt_len)
                self._decoder_only_graph.replay()
            else:
                self.buf["noise"].copy_(
                    noise.reshape(self.chunk_size, ACTION_DIM))
                valid_prefix = (
                    self.num_views * PATCHES_PER_VIEW
                    + self._current_prompt_len
                )
                self._decoder(valid_prefix)
            self._save("final_raw_action", self.buf["noise"])
        finally:
            self._capture_probes = False
        return self.buf["noise"].unsqueeze(0)
