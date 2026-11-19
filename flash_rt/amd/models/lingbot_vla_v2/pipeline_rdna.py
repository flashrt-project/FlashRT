"""LingBot-VLA-v2 BF16 inference pipeline for AMD RDNA 3 / 3.5 / 4.

The batch-1 path runs Qwen3-VL prefill and MoE action denoising with FP32
prefix K/V. The frontend owns input preparation and HIP Graph capture.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from flash_rt.amd.hardware.rdna35.ops import Rdna35TensorOps
from flash_rt.frontends.torch._groot_n17_fusion import fast_pos_embed_interpolate
from flash_rt.models.groot_n17.mrope_table import (
    RopeConfig,
    apply_interleaved_mrope,
    build_position_ids_for_segments,
    compute_inv_freq,
)

# ════════════════════════════════════════════════════════════════════
#  Transformer operations
# ════════════════════════════════════════════════════════════════════


class _TransformerOps:
    """Bind RDNA providers with the official BF16 rounding and FP32 attention.

    Dense weights retain nn.Linear storage; vision scores and probabilities
    round to BF16. Language/action attention retains FP32 intermediates.
    """

    def __init__(self, kernels=None, *, gemm=None, attention=None, fused_rope=True):
        self.tensor_ops = Rdna35TensorOps(kernels)
        self.gemm = gemm
        self.attention_provider = attention
        self.fused_rope = fused_rope
        self.prefix_attention = self._reference_attention
        if kernels is not None:
            from flash_rt.amd.hardware.rdna35.attention import RdnaAttention

            self.prefix_attention = RdnaAttention(kernels)
            if attention is None:
                self.attention_provider = self.prefix_attention.vision

    @staticmethod
    def prepare_weight(weight):
        """Keep logical [in, out] shape with physical nn.Linear [out, in] storage."""
        return weight.t().contiguous().t()

    def linear(self, x, weight, bias=None, *, reference_rows=None):
        shape = (*x.shape[:-1], weight.shape[1])
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        if self.gemm is None:
            return F.linear(flat, weight.t(), bias).view(shape)
        out = torch.empty((flat.shape[0], weight.shape[1]), dtype=x.dtype, device=x.device)
        self.gemm.linear(
            out, flat, weight.t(), bias, weight_transposed=True, reference_rows=reference_rows
        )
        return out.view(shape)

    def layer_norm(self, x, weight, bias, eps=1e-6):
        return self.tensor_ops.layer_norm(None, x, weight, bias, eps)

    def linear_pair(self, x, weight, *, reference_rows=None):
        """Project adjacent weight slices with one launch and reference reduction."""
        if self.gemm is None:
            return tuple(self.linear(x, part) for part in weight.chunk(2, dim=1))
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        columns = weight.shape[1] // 2
        out = torch.empty((2, flat.shape[0], columns), dtype=x.dtype, device=x.device)
        self.gemm.linear_pair(out, flat, weight.t(), reference_rows=reference_rows)
        return out.view(2, *x.shape[:-1], columns).unbind(0)

    def rms_norm(self, x, weight, eps=1e-6):
        """Affine RMSNorm with BF16 rounding before the weight multiplication."""
        if self.tensor_ops.fvk is not None:
            return self.tensor_ops.rms_norm_affine(x, weight, eps)
        out = torch.empty_like(x)
        self.tensor_ops.rms_norm(
            out, x, weight, eps, weight_offset=0.0, round_affine=True, fused=False
        )
        return out

    def rope(self, x, cos, sin):
        if self.tensor_ops.fvk is not None and self.fused_rope:
            return self.tensor_ops.rotate_half_hip(x, cos, sin)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        self.tensor_ops.rotate_half(out, x, cos, sin, fused=self.fused_rope)
        return out

    def attention(self, q, k, v):
        """BSHD attention; preserve eager BF16 scores and probabilities by default."""
        if self.attention_provider is not None:
            return self.attention_provider(q, k, v)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
        probs = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
        return (probs @ v).transpose(1, 2).contiguous()

    def fp32_rope(self, x, cos, sin):
        """Rotate FP32 Q/K without the BF16 output store used by vision RoPE."""
        if self.tensor_ops.fvk is not None:
            return self.tensor_ops.rotate_half_hip(x, cos, sin, fp32=True)
        x = x.float()
        first, second = x.chunk(2, -1)
        return x * cos[:, :, None] + torch.cat((-second, first), -1) * sin[:, :, None]

    def silu(self, x, up=None):
        if self.tensor_ops.fvk is not None:
            return self.tensor_ops.silu_bf16(x, up)
        activated = F.silu(x)
        return activated if up is None else activated * up

    def kv_cache(self, k, v, heads, suffix_rows):
        if self.tensor_ops.fvk is not None:
            return self.prefix_attention.prepare_cache(k, v, heads, suffix_rows)
        return {"key_states": k, "value_states": v}

    def cached_attention(self, q, cache, k, v, mask, *, rotary=None):
        prefix_k, prefix_v = cache["key_states"], cache["value_states"]
        if self.tensor_ops.fvk is not None:
            return self.prefix_attention(
                q, prefix_k, prefix_v, mask, suffix=(k, v), cache=cache, rotary=rotary
            )
        if rotary is not None:
            q, k = (self.fp32_rope(value, *rotary) for value in (q, k))
        return self.prefix_attention(
            q, torch.cat((prefix_k, k), 1), torch.cat((prefix_v, v.float()), 1), mask
        )

    def route(self, hidden, weight, bias, top_k, scale, *, router_input=None):
        flat = hidden.flatten(0, 1).float() if router_input is None else router_input.flatten(0, 1)
        if self.gemm is None:
            logits = F.linear(flat, weight)
            scores = logits.sigmoid()
            selected = torch.topk(scores + bias, top_k, dim=-1).indices
            route = scores.gather(1, selected)
            route = (route / (route.sum(-1, keepdim=True) + 1e-20) * scale).to(hidden.dtype)
        else:
            logits = torch.empty(
                (flat.shape[0], weight.shape[0]), device=flat.device, dtype=torch.float32
            )
            route = torch.empty((flat.shape[0], top_k), device=flat.device, dtype=hidden.dtype)
            selected = torch.empty((flat.shape[0], top_k), device=flat.device, dtype=torch.int32)
            stream = self.tensor_ops._stream(flat)
            self.gemm._runner.fp32_nt(
                flat.data_ptr(), weight.data_ptr(), logits.data_ptr(), flat.shape[0],
                weight.shape[0], flat.shape[1], weight.stride(0), weight.shape[0], stream,
            )
            self.tensor_ops.fvk.moe_topk_rdna(
                logits.data_ptr(), bias.data_ptr(), route.data_ptr(), selected.data_ptr(),
                flat.shape[0], weight.shape[0], top_k, scale, stream,
            )
        return logits, route, selected

    @staticmethod
    def _reference_attention(q, k, v, mask, *, cache=None):
        """Official FP32 GQA attention, including finite masking of padded rows."""
        groups = q.shape[2] // k.shape[2]
        k, v = k.repeat_interleave(groups, dim=2), v.repeat_interleave(groups, dim=2)
        q, k, v = (torch.einsum("blhd->bhld", x) for x in (q, k, v))
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * q.shape[-1] ** -0.5
        probs = torch.where(mask[:, None], scores, -2.3819763e38).softmax(-1)
        attended = torch.einsum("bhqk,bhkv->bhqv", probs, v)
        return attended.permute(0, 2, 1, 3).contiguous().flatten(2)


# ════════════════════════════════════════════════════════════════════
#  Position tables
# ════════════════════════════════════════════════════════════════════


def vit_rope(grid, *, device, dtype, head_dim, merge=2):
    """Spatial rotate-half tables in Qwen3-VL merge-block token order.

    The reference initializes inverse frequencies in FP32 on the host, then
    casts the non-persistent buffer with the model. Angles and trig functions
    use that buffer's BF16 dtype; applying RoPE uses FP32 products.
    """
    rotary_dim = head_dim // 2
    inv = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    seq = torch.arange(max(max(h, w) for _, h, w in grid), device=device, dtype=dtype)
    frequencies = torch.outer(seq, inv.to(device=device, dtype=dtype))
    positions = []
    for t, h, w in grid:
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        xy = torch.stack((yy, xx), -1).reshape(h // merge, merge, w // merge, merge, 2)
        positions.append(xy.permute(0, 2, 1, 3, 4).reshape(-1, 2).repeat(t, 1))
    angles = frequencies[torch.cat(positions).to(device)].flatten(1)
    angles = torch.cat((angles, angles), -1)
    return angles.cos(), angles.sin()


def llm_rope(position_ids, config, *, weight_dtype):
    """FP32 M-RoPE tables from the model's dtype-rounded inverse frequencies."""
    rope = RopeConfig(
        head_dim=config["head_dim"], rope_theta=config["rope_theta"],
        mrope_section=tuple(config["rope_scaling"]["mrope_section"]),
    )
    inv = compute_inv_freq(rope).to(weight_dtype).to(position_ids.device).float()
    expanded = inv[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
    frequencies = (expanded @ position_ids[:, :, None].float()).transpose(2, 3)
    frequencies = apply_interleaved_mrope(frequencies, rope.mrope_section)
    angles = torch.cat((frequencies, frequencies), -1)
    return angles.cos(), angles.sin()


def bf16_time_schedule(num_steps):
    """Accumulate the official Euler times in BF16 outside graph capture."""
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.bfloat16)
    time = torch.tensor(1.0, dtype=torch.bfloat16)
    times = []
    while time >= -dt / 2:
        times.append(time.clone())
        time += dt
    return torch.stack(times), dt


# ════════════════════════════════════════════════════════════════════
#  Vision traversal
# ════════════════════════════════════════════════════════════════════


class LingbotVlaV2PipelineRdna:
    """BF16 vision, language prefill and cached-prefix MoE action denoising."""

    def __init__(self, weights, config, ops):
        self.w, self.cfg, self.ops = weights, config, ops
        self.heads = config["num_heads"]
        self.hidden = weights._vit_patch_w.shape[0]
        self.head_dim = self.hidden // self.heads
        self.merge = config["spatial_merge_size"]
        self.merged_hidden = self.hidden * self.merge**2
        self.deepstack_indexes = config["deepstack_visual_indexes"]
        self.trace = None
        self.compact_prefill = (
            self.ops.gemm is not None
            and dict(self.ops.tensor_ops.fvk.build_info())["gpu_arch"] == "gfx1151"
        )
        self.pair_prefix_projections = self.compact_prefill
        self.pair_kv_projections = self.compact_prefill
        self.compact_prefix_queries = self.compact_prefill
        self.fuse_prefix_layout = self.compact_prefill
        self._prepare_vision_weights()

    def _prepare_vision_weights(self):
        # Pack dense projections once, before graph capture. Norms, embeddings
        # and Conv3d weights retain their checkpoint layout.
        for stem in ("_vit_qkv", "_vit_o", "_vit_fc1", "_vit_fc2", "_ds_fc1", "_ds_fc2"):
            name = stem + "_w"
            setattr(self.w, name, [self.ops.prepare_weight(w) for w in getattr(self.w, name)])
        for stem in ("_merger_fc1", "_merger_fc2"):
            name = stem + "_w"
            setattr(self.w, name, self.ops.prepare_weight(getattr(self.w, name)))

    def bind_vision(self, grid_thw):
        """Prepare position tables and temporal attention lengths outside the graph."""
        self.grid = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
        self.view_lengths = [h * w for t, h, w in self.grid for _ in range(t)]
        weight = self.w._vit_pos_embed
        self.position = fast_pos_embed_interpolate(
            weight, self.grid, device=weight.device, merge=self.merge
        )
        self.vit_cos, self.vit_sin = vit_rope(
            self.grid, device=weight.device, dtype=weight.dtype, head_dim=self.head_dim,
            merge=self.merge,
        )

    def _project(self, x, stem, index=None):
        weight, bias = getattr(self.w, stem + "_w"), getattr(self.w, stem + "_b")
        if index is not None:
            weight, bias = weight[index], bias[index]
        return self.ops.linear(x, weight, bias)

    def _norm(self, x, stem, index=None):
        weight, bias = getattr(self.w, stem + "_w"), getattr(self.w, stem + "_b")
        if index is not None:
            weight, bias = weight[index], bias[index]
        return self.ops.layer_norm(x, weight, bias, eps=1e-6)

    def _save(self, key, value):
        if self.trace is not None:
            self.trace[key] = value.detach().float().cpu().clone()

    def patch_embed(self, pixels):
        """Project processor-packed [tokens, C*T*P*P] patches with Conv3d."""
        weight = self.w._vit_patch_w
        patches = pixels.reshape(-1, *weight.shape[1:]).to(weight.dtype)
        return F.conv3d(patches, weight, self.w._vit_patch_b, stride=weight.shape[2:]).reshape(
            -1, self.hidden
        )

    def _vision_attention(self, q, k, v):
        # Keep the reference's per-slice GEMM layout. Batching equal-sized views
        # changes ROCm BF16 attention rounding at the production 16x256x64 shape.
        parts, start = [], 0
        for size in self.view_lengths:
            shape = (1, size, self.heads, self.head_dim)
            parts.append(
                self.ops.attention(
                    q[start : start + size].reshape(shape), k[start : start + size].reshape(shape),
                    v[start : start + size].reshape(shape),
                ).reshape(size, self.hidden)
            )
            start += size
        return torch.cat(parts)

    def _merger(self, x, index=None):
        # Main merger normalizes patches; DeepStack normalizes merged tokens.
        if index is None:
            x = self._norm(x, "_merger_ln").reshape(-1, self.merged_hidden)
            fc1, fc2 = "_merger_fc1", "_merger_fc2"
        else:
            x = self._norm(x.reshape(-1, self.merged_hidden), "_ds_ln", index)
            fc1, fc2 = "_ds_fc1", "_ds_fc2"
        # Merger uses exact GELU; the ViT MLP below uses the tanh approximation.
        return self._project(F.gelu(self._project(x, fc1, index)), fc2, index)

    def vision(self, pixels):
        """Packed image patches → visual tokens + DeepStack feature list, all BF16."""
        x = self.patch_embed(pixels)
        self._save("patch_embed", x)
        x = x + self.position
        self._save("pixel_features", x)
        taps = []
        for i in range(len(self.w._vit_qkv_w)):
            y = self._norm(x, "_vit_ln1", i)
            q, k, v = (
                self._project(y, "_vit_qkv", i).reshape(-1, 3, self.heads, self.head_dim).unbind(1)
            )
            q = self.ops.rope(q, self.vit_cos, self.vit_sin)
            k = self.ops.rope(k, self.vit_cos, self.vit_sin)
            x = x + self._project(self._vision_attention(q, k, v), "_vit_o", i)
            y = self._norm(x, "_vit_ln2", i)
            y = F.gelu(self._project(y, "_vit_fc1", i), approximate="tanh")
            x = x + self._project(y, "_vit_fc2", i)
            self._save(f"vit_block_{i}", x)
            if i in self.deepstack_indexes:
                index = self.deepstack_indexes.index(i)
                taps.append(self._merger(x, index))
                self._save(f"deepstack_{index}", taps[-1])
        merged = self._merger(x)
        self._save("vision_features", merged)
        return merged, taps

    # ════════════════════════════════════════════════════════════════════
    #  Prefix setup
    # ════════════════════════════════════════════════════════════════════

    def _prepare_prefix_weights(self):
        # Retain Q/K/V and gate/up views of nn.Linear weight storage.
        # Paired gate/up launches keep each projection's original reduction tiles.
        for stem in ("_llm_qkv", "_llm_o", "_llm_gate_up", "_llm_down"):
            name = stem + "_w"
            setattr(self.w, name, [self.ops.prepare_weight(w) for w in getattr(self.w, name)])
        for stem in ("_current_shared_task_proj", "_future_shared_task_proj"):
            name = stem + "_w"
            setattr(self.w, name, self.ops.prepare_weight(getattr(self.w, name)))
        widths = (
            self.llm_heads * self.llm_head_dim, self.llm_kv_heads * self.llm_head_dim,
            self.llm_kv_heads * self.llm_head_dim,
        )
        self.llm_qkv = [weight.split(widths, dim=1) for weight in self.w._llm_qkv_w]
        self.llm_kv = [weight[:, widths[0] :] for weight in self.w._llm_qkv_w]
        self.llm_gate_up = [weight.chunk(2, dim=1) for weight in self.w._llm_gate_up_w]

    def _query_embeddings(self, policy):
        align = policy["align_params"]
        if not align:
            return self.w._embed_tokens_w.new_empty((0, self.w._embed_tokens_w.shape[1]))
        count = align["num_task_tokens"]

        def pooled(name):
            weights = getattr(self.w, name)
            return weights.reshape(count, -1, weights.shape[-1]).mean(1)

        current = pooled("_depth_align_embs")
        video = align.get("video", {})
        use_video = align.get("use_future_video", False)
        use_patch = video.get("use_patch_loss", True)
        current_patch = video.get("use_current_patch_loss", False)
        if use_video and current_patch and video.get("use_current_shared_task_proj", current_patch):
            current = self._project(
                torch.cat((current, pooled("_current_video_align_embs")), -1),
                "_current_shared_task_proj",
            )
        queries = [current]
        shared = video.get("share_future_depth_query", False)
        if use_video and use_patch and not shared:
            queries.append(pooled("_future_video_align_embs"))
        if align.get("depth", {}).get("use_future_depth", False):
            future = pooled("_future_depth_align_embs")
            if (
                use_video
                and use_patch
                and shared
                and video.get("use_shared_future_task_proj", False)
            ):
                future = self._project(
                    torch.cat((future, pooled("_future_video_align_embs")), -1),
                    "_future_shared_task_proj",
                )
            queries.append(future)
        return torch.cat(queries)

    def bind_prefix(self, lang_tokens, lang_masks, img_masks, *, qwen_config, policy_config):
        """Bind a batch-1 tokenized prompt and fixed image masks before graph capture."""
        self.text_cfg = qwen_config["text_config"]
        self.llm_heads = self.text_cfg["num_attention_heads"]
        self.llm_kv_heads = self.text_cfg["num_key_value_heads"]
        self.llm_head_dim = self.text_cfg["head_dim"]
        self.llm_eps = self.text_cfg["rms_norm_eps"]
        self._prepare_prefix_weights()
        self.query_embeddings = self._query_embeddings(policy_config)
        device = self.w._embed_tokens_w.device
        tokens, text_valid = lang_tokens.detach().cpu().reshape(
            -1
        ), lang_masks.detach().cpu().bool().reshape(-1)
        images_valid = img_masks.detach().cpu().bool().reshape(-1).tolist()
        boundaries = policy_config.get("qwen3vl_use_vision_boundaries", True)
        ids, masks = [], []
        image_positions, visual_positions, source_positions = [], [], []
        lengths, kinds, grids = [], [], []
        offset, source_offset = 0, 0
        for grid, valid in zip(self.grid, images_valid):
            size = grid[0] * grid[1] * grid[2] // self.merge**2
            chunk = [qwen_config["image_token_id"]] * size
            if boundaries:
                chunk = [
                    qwen_config["vision_start_token_id"], *chunk,
                    qwen_config["vision_end_token_id"],
                ]
            positions = list(range(offset + int(boundaries), offset + int(boundaries) + size))
            image_positions.extend(positions)
            if valid:
                visual_positions.extend(positions)
                source_positions.extend(range(source_offset, source_offset + size))
                if boundaries:
                    lengths.extend((1, size, 1))
                    kinds.extend(("text", "image", "text"))
                    grids.extend((None, grid, None))
                else:
                    lengths.append(size)
                    kinds.append("text")
                    grids.append(None)
            ids.extend(chunk)
            masks.extend([valid] * len(chunk))
            offset += len(chunk)
            source_offset += size
        ids.extend(tokens.tolist())
        masks.extend(text_valid.tolist())
        query_start = len(ids)
        query_count = self.query_embeddings.shape[0]
        ids.extend([self.text_cfg["eos_token_id"]] * query_count)
        masks.extend([True] * query_count)
        lengths.append(int(text_valid.sum()) + query_count)
        kinds.append("text")
        grids.append(None)
        rope = RopeConfig(spatial_merge_size=self.merge)
        valid_positions = build_position_ids_for_segments(
            segment_lengths=lengths, segment_kinds=kinds, segment_grids=grids, cfg=rope
        )
        padding = torch.tensor(masks, dtype=torch.bool).reshape(1, -1)
        positions = torch.ones((3, 1, len(ids)), dtype=torch.long)
        positions[:, :, padding[0]] = valid_positions
        self.prefix_input_ids = torch.tensor(ids, device=device).reshape(1, -1)
        self.prefix_pad_masks = padding.to(device)
        self.prefix_position_ids = positions.to(device)
        self.image_positions = torch.tensor(image_positions, device=device, dtype=torch.long)
        self.visual_positions = torch.tensor(visual_positions, device=device, dtype=torch.long)
        self.deepstack_source_positions = torch.tensor(
            source_positions, device=device, dtype=torch.long
        )
        self.query_positions = torch.arange(query_start, len(ids), device=device)
        size = len(ids)
        causal = policy_config["vlm_causal"]
        allowed = torch.ones((size, size), dtype=torch.bool, device=device)
        if causal:
            allowed = allowed.tril()
        self.prefix_attention_mask = (
            allowed[None] & self.prefix_pad_masks[:, None] & self.prefix_pad_masks[:, :, None]
        )
        self.llm_cos, self.llm_sin = llm_rope(
            self.prefix_position_ids, self.text_cfg, weight_dtype=self.w._embed_tokens_w.dtype
        )
        active = padding[0].nonzero().flatten()
        self.prefix_active_rows = active.to(device)
        row_map = torch.full((size,), -1, dtype=torch.long)
        row_map[active] = torch.arange(active.numel())
        self.prefix_row_map = row_map.to(device)
        self.compact_visual_positions = row_map[visual_positions].to(device)
        self.compact_llm_cos = self.llm_cos.index_select(1, self.prefix_active_rows)
        self.compact_llm_sin = self.llm_sin.index_select(1, self.prefix_active_rows)
        self.compact_prefix_attention_mask = self.prefix_attention_mask.index_select(
            1, self.prefix_active_rows
        )
        if hasattr(self, "action_cfg"):
            self._bind_suffix()

    def embed_prefix(self, image_embeds, deepstack_embeds):
        """Assemble image boundaries, padded text and learned task-query embeddings."""
        hidden = F.embedding(self.prefix_input_ids, self.w._embed_tokens_w).flatten(0, 1)
        hidden.index_copy_(0, self.image_positions, image_embeds)
        hidden.index_copy_(0, self.query_positions, self.query_embeddings)
        taps = [tap.index_select(0, self.deepstack_source_positions) for tap in deepstack_embeds]
        hidden = hidden.unsqueeze(0)
        self._save("prefix_embeddings", hidden)
        return hidden, taps

    # ════════════════════════════════════════════════════════════════════
    #  LLM prefill
    # ════════════════════════════════════════════════════════════════════

    def _expand_prefix(self, value):
        out = value.new_zeros((1, self.prefix_pad_masks.numel(), *value.shape[2:]))
        return out.index_copy_(1, self.prefix_active_rows, value)

    def _use_prefix_layout(self):
        return (
            self.fuse_prefix_layout
            and self.compact_prefill
            and self.compact_prefix_queries
            and self.trace is None
            and self.llm_head_dim == 128
            and 0 < self.prefix_active_rows.numel() < self.prefix_pad_masks.numel()
            and self.ops.prefix_attention._groups(self.llm_heads, self.llm_kv_heads, 128) > 1
        )

    def encode_prefix(self, image_embeds, deepstack_embeds, *, cache_only=False):
        """BF16 LLM prefill → FP32 KV cache and optional final hidden states."""
        cache_only = cache_only and self.trace is None
        hidden, taps = self.embed_prefix(image_embeds, deepstack_embeds)
        rows = self.prefix_pad_masks.numel()
        compact = (
            self.compact_prefill
            and self.trace is None
            and 0 < self.prefix_active_rows.numel() < rows
        )
        reference_rows = rows if compact else None
        compact_queries = (
            compact
            and self.compact_prefix_queries
            and self.ops.prefix_attention._groups(
                self.llm_heads, self.llm_kv_heads, self.llm_head_dim
            )
            > 1
        )
        fused_layout = self._use_prefix_layout()
        cos, sin = (
            (self.compact_llm_cos, self.compact_llm_sin)
            if compact
            else (self.llm_cos, self.llm_sin)
        )
        visual_positions = self.compact_visual_positions if compact else self.visual_positions
        if compact:
            hidden = hidden.index_select(1, self.prefix_active_rows)

        def linear(value, weight):
            return self.ops.linear(value, weight, reference_rows=reference_rows)

        cache = {}
        suffix_rows = self.horizon + 1 if hasattr(self, "horizon") else 0
        for i, (q_weight, k_weight, v_weight) in enumerate(self.llm_qkv):
            normalized = self.ops.rms_norm(hidden, self.w._llm_input_ln_w[i], self.llm_eps)
            last_cache = cache_only and i + 1 == len(self.llm_qkv)
            if self.pair_kv_projections and self.trace is None:
                k, v = self.ops.linear_pair(
                    normalized, self.llm_kv[i], reference_rows=reference_rows
                )
            else:
                k, v = linear(normalized, k_weight), linear(normalized, v_weight)
            k = k.reshape(1, -1, self.llm_kv_heads, self.llm_head_dim)
            v = v.reshape(1, -1, self.llm_kv_heads, self.llm_head_dim)
            if fused_layout:
                cache[i] = self.ops.prefix_attention.prepare_prefix_cache(
                    k, v, self.w._llm_k_norm_w[i], cos, sin, self.prefix_row_map, self.llm_heads,
                    suffix_rows, self.llm_eps,
                )
                k, v = cache[i]["key_states"], cache[i]["value_states"]
            else:
                v = v.float()
                k = self.ops.rms_norm(k, self.w._llm_k_norm_w[i], self.llm_eps)
                k = self.ops.fp32_rope(k, cos, sin)
                if compact:
                    # Keep the original key positions and masked reduction slots.
                    k, v = (self._expand_prefix(value) for value in (k, v))
                cache[i] = self.ops.kv_cache(k, v, self.llm_heads, suffix_rows)
            self._save(f"llm_key_{i}", k)
            self._save(f"llm_value_{i}", v)
            if last_cache:
                return None, cache
            q = linear(normalized, q_weight).reshape(1, -1, self.llm_heads, self.llm_head_dim)
            packed_q = None
            if fused_layout:
                packed_q = self.ops.prefix_attention.prepare_prefix_query(
                    q, self.w._llm_q_norm_w[i], cos, sin, cache[i].groups, self.llm_eps
                )
            else:
                q = self.ops.rms_norm(q, self.w._llm_q_norm_w[i], self.llm_eps)
                q = self.ops.fp32_rope(q, cos, sin)
            if compact and not compact_queries:
                q = self._expand_prefix(q)
            if compact_queries:
                attended = self.ops.prefix_attention(
                    q, k, v, self.compact_prefix_attention_mask, cache=cache[i],
                    reference_queries=rows, packed_query=packed_q,
                ).to(hidden.dtype)
            else:
                attended = self.ops.prefix_attention(
                    q, k, v, self.prefix_attention_mask, cache=cache[i]
                ).to(hidden.dtype)
            if compact and not compact_queries:
                attended = attended.index_select(1, self.prefix_active_rows)
            hidden = hidden + linear(attended, self.w._llm_o_w[i])
            normalized = self.ops.rms_norm(hidden, self.w._llm_post_ln_w[i], self.llm_eps)
            if self.pair_prefix_projections and self.trace is None:
                gate, up = self.ops.linear_pair(
                    normalized, self.w._llm_gate_up_w[i], reference_rows=reference_rows
                )
            else:
                gate_weight, up_weight = self.llm_gate_up[i]
                gate, up = linear(normalized, gate_weight), linear(normalized, up_weight)
            ff = self.ops.silu(gate, up)
            hidden = hidden + linear(ff, self.w._llm_down_w[i])
            if i < len(taps):
                flat = hidden.flatten(0, 1)
                flat.index_copy_(
                    0, visual_positions, flat.index_select(0, visual_positions) + taps[i]
                )
            self._save(f"llm_layer_{i}", hidden)
        hidden = self.ops.rms_norm(hidden, self.w._llm_final_ln_w, self.llm_eps)
        if compact:
            hidden = self._expand_prefix(hidden)
        self._save("llm_features", hidden)
        return hidden, cache

    def prefill(self, pixels, *, cache_only=False):
        """Packed image patches → LLM prefix output and action-expert KV context."""
        image_embeds, deepstack_embeds = self.vision(pixels)
        return self.encode_prefix(image_embeds, deepstack_embeds, cache_only=cache_only)

    # ════════════════════════════════════════════════════════════════════
    #  Action expert setup
    # ════════════════════════════════════════════════════════════════════

    def bind_action(self, policy_config, *, moe=None):
        """Prepare the BF16 action expert, fixed time modulation and MoE workspace."""
        self.action_cfg = policy_config
        self.action_heads = policy_config.get("action_num_attention_heads", 32)
        self.action_kv_heads = policy_config.get("action_num_key_value_heads", 8)
        self.action_head_dim = policy_config.get("action_head_dim", 128)
        self.action_eps = 1e-6
        self.horizon = policy_config["n_action_steps"]
        self.top_k = policy_config["token_top_k"]
        self.routing_scale = policy_config["routed_scaling_factor"]
        for stem in (
            "_expert_qkv", "_expert_o", "_expert_input_gamma", "_expert_input_beta",
            "_expert_post_gamma", "_expert_post_beta", "_moe_shared_gate_up", "_moe_shared_down",
        ):
            name = stem + "_w"
            setattr(self.w, name, [self.ops.prepare_weight(w) for w in getattr(self.w, name)])
        for stem in (
            "_state_proj", "_action_in_proj", "_action_out_proj", "_action_time_mlp_in",
            "_action_time_mlp_out",
        ):
            name = stem + "_w"
            setattr(self.w, name, self.ops.prepare_weight(getattr(self.w, name)))
        widths = (
            self.action_heads * self.action_head_dim, self.action_kv_heads * self.action_head_dim,
            self.action_kv_heads * self.action_head_dim,
        )
        self.action_qkv = [weight.split(widths, dim=1) for weight in self.w._expert_qkv_w]
        self.action_qkv_bias = [bias.split(widths) for bias in self.w._expert_qkv_b]
        self.shared_gate_up = [weight.chunk(2, dim=1) for weight in self.w._moe_shared_gate_up_w]
        weight = self.w._state_proj_w
        fraction = torch.linspace(0.0, 1.0, weight.shape[1] // 2, device=weight.device)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        self.time_frequency = 1.0 / period * 2 * math.pi
        times, dt = bf16_time_schedule(policy_config["num_steps"])
        self.dt = dt.to(weight.device)
        self.times = times.to(weight.device)
        self.time_embeddings = [self._time_embedding(t.reshape(1)) for t in self.times]
        self.time_modulations = [self._modulation(t) for t in self.time_embeddings]
        if moe is None:
            from flash_rt.amd.hardware.rdna.moe import RdnaMoe

            experts, intermediate, hidden = self.w._moe_gate_w[0].shape
            moe = RdnaMoe(
                self.horizon + 1, hidden, experts, self.top_k, intermediate, device=weight.device,
                dtype=weight.dtype, kernels=self.ops.tensor_ops.fvk,
            )
            for gate, up, down in zip(self.w._moe_gate_w, self.w._moe_up_w, self.w._moe_down_w):
                moe.prepare_weights(gate, up, down, compress=True)
        self.moe = moe
        self.shared_ops, self.shared_stream = self.ops, None
        kernels = self.ops.tensor_ops.fvk
        self.fuse_action_residual = (
            kernels is not None and dict(kernels.build_info())["gpu_arch"] == "gfx1151"
        )
        self.fuse_router_input = (
            self.fuse_action_residual
            and self.w._expert_post_ln_w[0].numel() == 768
            and self.horizon + 1 >= 16
        )
        self.fuse_action_rope = (
            self.fuse_action_residual
            and self.ops.prefix_attention._groups(
                self.action_heads, self.action_kv_heads, self.action_head_dim
            )
            > 1
        )
        if self.ops.gemm is not None and dict(kernels.build_info())["gpu_arch"] == "gfx1151":
            from flash_rt.amd.hardware.rdna35.gemm import Rdna35GemmBackend

            self.shared_ops = copy.copy(self.ops)
            self.shared_ops.gemm = Rdna35GemmBackend(kernels, autotune=False, native_nt=True)
            self.shared_stream = torch.cuda.Stream(device=weight.device)
        self._bind_suffix()

    def _bind_suffix(self):
        """Bind cached-prefix attention and M-RoPE to the current prompt."""
        size = self.horizon + 1
        device = self.prefix_pad_masks.device
        suffix_mask = torch.ones((size, size), dtype=torch.bool, device=device)
        suffix_mask[0, 1:] = False
        prefix_mask = self.prefix_pad_masks[:, None].expand(-1, size, -1).clone()
        align = self.action_cfg["align_params"]
        count = align.get("num_task_tokens", 0)
        depth, video = align.get("depth", {}), align.get("video", {})
        start = int(self.query_positions[0]) if self.query_positions.numel() else 0
        if align.get("use_future_video", False) and video.get("use_patch_loss", True):
            fv_start = start + count
            if not video.get("share_future_depth_query", False):
                start += count
            if video.get("block_suffix_to_future_video", False):
                prefix_mask[:, :, fv_start : fv_start + count] = False
        if depth.get("use_future_depth", False) and depth.get(
            "block_future_depth_to_action", False
        ):
            prefix_mask[:, :, start + count : start + 2 * count] = False
        self.suffix_attention_mask = torch.cat((prefix_mask, suffix_mask[None]), -1)
        valid_positions = self.prefix_position_ids.masked_fill(~self.prefix_pad_masks[None], 0)
        offset = valid_positions.amax(dim=(0, 2)) + 1
        positions = offset[:, None] + torch.arange(size, device=device)
        self.suffix_position_ids = positions[None].expand(3, -1, -1)
        self.action_cos, self.action_sin = llm_rope(
            self.suffix_position_ids, self.text_cfg, weight_dtype=self.w._state_proj_w.dtype
        )

    def _time_embedding(self, timestep):
        angles = self.time_frequency[None] * timestep[:, None]
        return torch.cat((angles.sin(), angles.cos()), -1).to(self.w._state_proj_w.dtype)

    def _modulation(self, time_emb):
        return [
            (
                self._project(time_emb, f"_expert_{tag}_gamma", i).unsqueeze(1).float(),
                self._project(time_emb, f"_expert_{tag}_beta", i).unsqueeze(1).float(),
            )
            for i in range(len(self.action_qkv))
            for tag in ("input", "post")
        ]

    def _ada_rms_norm(self, hidden, weight, modulation):
        # AdaRMSNorm applies both checkpoint affine and FiLM in FP32.
        if self.ops.tensor_ops.fvk is not None:
            return self.ops.tensor_ops.rms_norm_affine(
                hidden, weight, self.action_eps, modulation=modulation
            )
        value = hidden.float()
        value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.action_eps)
        value = weight * value
        gamma, beta = modulation
        return ((1 + gamma) * value + beta).to(hidden.dtype)

    def _action_embeddings(self, state_emb, noisy_actions, time_emb):
        action_emb = self._project(noisy_actions, "_action_in_proj")
        time = time_emb[:, None].expand(-1, noisy_actions.shape[1], -1)
        hidden = self._project(torch.cat((action_emb, time), -1), "_action_time_mlp_in")
        hidden = self._project(self.ops.silu(hidden), "_action_time_mlp_out")
        return torch.cat((state_emb[:, None], hidden), 1)

    def embed_suffix(self, state, noisy_actions, timestep):
        """Normalized state + noisy actions + scalar time → BF16 suffix and condition."""
        time_emb = self._time_embedding(timestep)
        return time_emb, self._action_embeddings(
            self._project(state, "_state_proj"), noisy_actions, time_emb
        )

    # ════════════════════════════════════════════════════════════════════
    #  Cached-prefix action traversal and Euler denoising
    # ════════════════════════════════════════════════════════════════════

    def _shared_expert(self, flat, index):
        gate, up = self.shared_gate_up[index]
        ops = self.shared_ops
        shared = ops.silu(ops.linear(flat, gate), ops.linear(flat, up))
        return ops.linear(shared, self.w._moe_shared_down_w[index])

    def _action_moe(self, hidden, index, *, separate=False, router_input=None):
        flat = hidden.flatten(0, 1)
        if self.shared_stream is not None:
            # Shared experts depend only on the normalized input, not routing.
            root = torch.cuda.current_stream(hidden.device)
            self.shared_stream.wait_stream(root)
            with torch.cuda.stream(self.shared_stream):
                shared = self._shared_expert(flat, index)
                shared.record_stream(root)
        logits, route, selected = self.ops.route(
            hidden, self.w._moe_router_w[index], self.w._moe_router_bias[index], self.top_k,
            self.routing_scale, router_input=router_input,
        )
        self._save(f"expert_router_{index}", logits)
        self._save(f"expert_selected_{index}", selected)
        routed = self.moe(
            flat, route, selected, self.w._moe_gate_w[index], self.w._moe_up_w[index],
            self.w._moe_down_w[index],
        )
        if not separate:
            routed = routed.to(hidden.dtype)
        if self.shared_stream is None:
            shared = self._shared_expert(flat, index)
        else:
            root.wait_stream(self.shared_stream)
        if separate:
            return routed, shared
        return (routed + shared).view_as(hidden)

    def decode_suffix(self, hidden, cache, modulation):
        """Action expert layers against immutable FP32 prefix K/V."""
        fused = self.fuse_action_residual and self.trace is None
        fused_router = self.fuse_router_input and self._use_prefix_layout()
        normalized = self._ada_rms_norm(hidden, self.w._expert_input_ln_w[0], modulation[0])
        for i, (weights, biases) in enumerate(zip(self.action_qkv, self.action_qkv_bias)):
            q, k, v = [
                self.ops.linear(normalized, weight, bias).reshape(
                    1, -1, heads, self.action_head_dim
                )
                for weight, bias, heads in zip(
                    weights, biases, (self.action_heads, self.action_kv_heads, self.action_kv_heads)
                )
            ]
            rotary = None
            if self.fuse_action_rope and self.trace is None:
                rotary = self.action_cos, self.action_sin
            else:
                q = self.ops.fp32_rope(q, self.action_cos, self.action_sin)
                k = self.ops.fp32_rope(k, self.action_cos, self.action_sin)
            attended = self.ops.cached_attention(
                q, cache[i], k, v, self.suffix_attention_mask, rotary=rotary
            ).to(hidden.dtype)
            projected = self.ops.linear(attended, self.w._expert_o_w[i])
            if fused:
                router_input = None
                if fused_router:
                    hidden, normalized, router_input = self.ops.tensor_ops.residual_rms_router(
                        projected, hidden, self.w._expert_post_ln_w[i], modulation[2 * i + 1],
                        self.action_eps,
                    )
                else:
                    hidden, normalized = self.ops.tensor_ops.residual_rms_affine(
                        projected, hidden, self.w._expert_post_ln_w[i], self.action_eps,
                        modulation=modulation[2 * i + 1],
                    )
                routed, shared = self._action_moe(
                    normalized, i, separate=True, router_input=router_input
                )
                # Fold the second residual into the next layer's input norm.
                weight = (
                    self.w._expert_input_ln_w[i + 1]
                    if i + 1 < len(self.action_qkv)
                    else self.w._expert_final_ln_w
                )
                condition = modulation[2 * i + 2] if i + 1 < len(self.action_qkv) else None
                hidden, normalized = self.ops.tensor_ops.residual_rms_affine(
                    routed, hidden, weight, self.action_eps, modulation=condition, shared=shared
                )
            else:
                hidden = hidden + projected
                normalized = self._ada_rms_norm(
                    hidden, self.w._expert_post_ln_w[i], modulation[2 * i + 1]
                )
                hidden = hidden + self._action_moe(normalized, i)
            self._save(f"expert_layer_{i}", hidden)
            if not fused:
                if i + 1 < len(self.action_qkv):
                    normalized = self._ada_rms_norm(
                        hidden, self.w._expert_input_ln_w[i + 1], modulation[2 * i + 2]
                    )
                else:
                    normalized = self.ops.rms_norm(
                        hidden, self.w._expert_final_ln_w, self.action_eps
                    )
        self._save("expert_features", normalized)
        return normalized

    def predict_velocity(self, state, cache, noisy_actions, timestep):
        """Predict BF16 velocity with the prefix cache at an arbitrary timestep."""
        time_emb, hidden = self.embed_suffix(state, noisy_actions, timestep)
        self._save("suffix_embeddings", hidden)
        hidden = self.decode_suffix(hidden, cache, self._modulation(time_emb))
        velocity = self._project(hidden[:, -self.horizon :], "_action_out_proj")
        self._save("velocity", velocity)
        return velocity

    def denoise(self, state, cache, noise):
        """Run the official BF16 Euler schedule with precomputed time modulation."""
        state_emb = self._project(state, "_state_proj")
        actions = noise.clone()
        for i, (time_emb, modulation) in enumerate(
            zip(self.time_embeddings, self.time_modulations)
        ):
            hidden = self._action_embeddings(state_emb, actions, time_emb)
            hidden = self.decode_suffix(hidden, cache, modulation)
            velocity = self._project(hidden[:, -self.horizon :], "_action_out_proj")
            actions += self.dt * velocity
            self._save(f"denoise_step_{i}", actions)
        return actions

    def infer(self, pixels, state, noise):
        """Packed image patches + normalized state + noise → normalized BF16 actions."""
        _, cache = self.prefill(pixels, cache_only=True)
        return self.denoise(state, cache, noise)
