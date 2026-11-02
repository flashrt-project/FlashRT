"""Unquantized GR00T N1.7 frame pipeline for RDNA 3 / 3.5 / 4.

The frontend owns input buffers and graph capture. This module owns model
semantics: ViT, DeepStack, the truncated LLM, VL adapter, and the DiT. Hardware
operations are injected once; there are no architecture forks in the loop.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from flash_rt.amd.hardware.rdna35.attention import sdpa
from flash_rt.amd.hardware.rdna35.gemm import Rdna35GemmBackend
from flash_rt.amd.hardware.rdna35.ops import Rdna35TensorOps
from flash_rt.frontends.torch._groot_n17_fusion import fast_pos_embed_interpolate


class _TransformerOps:
    """Allocate model temporaries and adapt shared RDNA providers to N1.7.

    Qwen RMSNorm and DiT modulation select explicit BF16 rounding contracts.
    Target-specific GEMM/attention policy is supplied by the frontend.
    """

    def __init__(self, kernels, *, library_gemm=True, long_attention=None,
                 gemm_autotune=None, weight_transposed=False):
        self.tensor_ops = Rdna35TensorOps(kernels)
        self.fused = True
        self.long_attention = long_attention
        self.weight_transposed = weight_transposed
        self.gemm = (Rdna35GemmBackend(kernels, autotune=gemm_autotune)
                     if library_gemm else None)

    def prepare_weight(self, weight):
        """Keep logical K-by-N shapes while selecting the physical GEMM layout.

        Replacing the setup tensor avoids retaining two copies of every weight.
        Embedding tables and convolution weights do not use this operation.
        """
        return weight.t().contiguous().t() if self.weight_transposed else weight

    def linear(self, x, weight, bias=None):
        shape = (*x.shape[:-1], weight.shape[1])
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        if self.gemm is None:
            return F.linear(flat, weight.t(), bias).view(shape)
        out = torch.empty((flat.shape[0], weight.shape[1]),
                          device=x.device, dtype=x.dtype)
        if self.weight_transposed:
            self.gemm.linear(out, flat, weight.t(), bias, weight_transposed=True)
        else:
            self.gemm.linear(out, flat, weight, bias)
        return out.view(shape)

    def layer_norm(self, x, weight=None, bias=None, eps=1e-5):
        return self.tensor_ops.layer_norm(None, x, weight, bias, eps)

    def rms_norm(self, x, weight, eps=1e-6):
        out = torch.empty_like(x)
        self.tensor_ops.rms_norm(out, x, weight, eps, weight_offset=0.0,
                                 round_affine=True, fused=self.fused)
        return out

    def ada_layer_norm(self, x, scale, shift, eps=1e-5):
        out = torch.empty_like(x)
        self.tensor_ops.ada_layer_norm(out, x, scale, shift, eps, fused=self.fused)
        return out

    def rope(self, x, cos, sin):
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        self.tensor_ops.rotate_half(out, x, cos, sin, fused=self.fused)
        return out

    def attention(self, q, k, v, *, causal=False):
        if self.long_attention is not None and q.shape[1] >= 128:
            return self.long_attention(q, k, v, causal=causal)
        return self.sdpa(q, k, v, causal=causal)

    @staticmethod
    def sdpa(q, k, v, *, causal=False):
        return sdpa(q, k, v, causal=causal).contiguous()

    def residual_layer_norm(self, update, residual, scale=None, shift=None, eps=1e-5):
        out, summed = torch.empty_like(update), torch.empty_like(update)
        self.tensor_ops.residual_layer_norm(
            out, summed, update, residual, scale, shift, eps, fused=self.fused)
        return out, summed


def vit_rope(grid, device):
    """Spatial RoPE with the official BF16 policy's buffer/angle rounding."""
    positions = []
    for t, h, w in grid:
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        xy = torch.stack((yy, xx), -1).reshape(h // 2, 2, w // 2, 2, 2)
        positions.append(xy.permute(0, 2, 1, 3, 4).reshape(-1, 2).repeat(t, 1))
    positions = torch.cat(positions).to(device)
    inv = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, device=device).float() / 32))
    # Gr00tPolicy.to(bfloat16) also casts the non-persistent rotary buffer.
    # Vision computes the outer product and trigonometry in that dtype.
    angles = (positions[..., None].bfloat16() * inv.bfloat16()).flatten(1)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


class GrootN17PipelineRdna:
    """One BF16 checkpoint bound to a prompt and fixed denoising schedule."""

    def __init__(self, weights, config, ops, *, device, horizon, steps):
        self.w, self.cfg, self.ops = weights, config, ops
        self.device, self.horizon, self.steps = device, horizon, steps
        self.num_timestep_buckets = config["num_timestep_buckets"]
        self.trace = None
        self.vlsa_qkv = [self._combine("_vlsa", ("q", "k", "v"), i) for i in range(4)]
        self.dit_qkv = {i: self._combine("_dit", ("q", "k", "v"), i)
                        for i in range(1, 32, 2)}
        self.dit_kv = {i: self._combine("_dit", ("k", "v"), i)
                       for i in range(0, 32, 2)}
        self._prepare_schedule()

    def _combine(self, prefix, names, index):
        combined = []
        for suffix in ("w", "b"):
            sources = [getattr(self.w, f"{prefix}_{name}_{suffix}") for name in names]
            value = torch.cat([values[index] for values in sources], dim=-1).contiguous()
            combined.append(self.ops.prepare_weight(value) if suffix == "w" else value)
            # The combined projection replaces these weights for this layer.
            for values in sources:
                values[index] = None
        return tuple(combined)

    def _project(self, x, stem, index=None):
        w = getattr(self.w, stem + "_w")
        b = getattr(self.w, stem + "_b", None)
        if index is not None:
            w = w[index]
            b = None if b is None else b[index]
        return self.ops.linear(x, w, b)

    def _norm(self, x, stem, index=None, eps=1e-5):
        w, b = getattr(self.w, stem + "_w"), getattr(self.w, stem + "_b")
        if index is not None:
            w, b = w[index], b[index]
        return self.ops.layer_norm(x, w, b, eps)

    def _save(self, key, value):
        if self.trace is not None:
            self.trace[key] = value.detach().float().cpu().clone()

    def _prepare_schedule(self):
        w, dev = self.w, self.device
        times = torch.tensor([int(i / self.steps * self.num_timestep_buckets)
                              for i in range(self.steps)], device=dev).float()
        inv = torch.exp(-math.log(10000.0) * torch.arange(128, device=dev).float() / 127)
        angle = times[:, None] * inv
        # TimestepEncoder operates in the checkpoint's BF16 dtype.
        emb = torch.cat((angle.cos(), angle.sin()), -1).bfloat16()
        emb = self._project(F.silu(self._project(emb, "_ts_lin1")), "_ts_lin2")
        conditioned = F.silu(emb)
        self.modulators = [self._project(conditioned, "_dit_ada", i).chunk(2, -1)
                           for i in range(32)]
        self.out_shift, self.out_scale = self._project(conditioned, "_proj_out_1").chunk(2, -1)
        dim = w._ac_enc_W1_W.shape[1]
        inv = torch.exp(-math.log(10000.0) * torch.arange(dim // 2, device=dev).float()
                        / (dim // 2))
        angle = times[:, None] * inv
        self.tau = torch.cat((angle.sin(), angle.cos()), -1).bfloat16()

    def bind_prompt(self, aux, *, fuse_image_embeds):
        self.grid = [tuple(int(v) for v in row) for row in aux["grid_thw"].tolist()]
        # Qwen3-VL attends independently within each temporal grid slice.
        self.view_lengths = [h * w for t, h, w in self.grid for _ in range(t)]
        self.uniform_views = len(set(self.view_lengths)) == 1
        self.vit_cos, self.vit_sin = vit_rope(self.grid, self.device)
        self.position = fast_pos_embed_interpolate(self.w._vit_pos_embed, self.grid,
                                                   device=self.device)
        self.ids = (aux["input_ids"].reshape(-1).to(self.device, torch.long).clone()
                    if fuse_image_embeds else None)
        mask = aux["visual_pos_masks"].reshape(-1).to(self.device, torch.bool)
        self.image_index = torch.where(mask)[0]
        self.text_index = torch.where(~mask)[0]
        self.llm_cos = aux["rope_cos"].reshape(-1, 128).to(self.device, torch.bfloat16).clone()
        self.llm_sin = aux["rope_sin"].reshape(-1, 128).to(self.device, torch.bfloat16).clone()

    def _attention(self, q, k, v, heads, dim, *, kv_heads=None, causal=False):
        q = q.reshape(1, -1, heads, dim)
        k = k.reshape(1, -1, kv_heads or heads, dim)
        v = v.reshape(1, -1, kv_heads or heads, dim)
        return self.ops.attention(q, k, v, causal=causal).reshape(-1, heads * dim)

    def _merger(self, x, prefix, *, postshuffle):
        x = x.reshape(-1, 4096) if postshuffle else x
        x = self._norm(x, prefix + "_norm", eps=1e-6).reshape(-1, 4096)
        return self._project(F.gelu(self._project(x, prefix + "_fc1")), prefix + "_fc2")

    def backbone(self, pixels, llm_input_embeds=None):
        w, ops = self.w, self.ops
        # Keep Conv3d's accumulation and bias semantics from the reference.
        if llm_input_embeds is None:
            x = F.conv3d(pixels.reshape(-1, 3, 2, 16, 16), w._patch_embed_w,
                         w._patch_embed_b, stride=(2, 16, 16)).reshape(-1, 1024)
            x = x + self.position
        else:
            # Existing CUDA/CDNA aux contains patch projection + position and
            # fused text/image embeddings. Recompute its DeepStack taps only.
            x = pixels
        self._save("pixel_features", x)
        taps = []
        for i in range(24 if llm_input_embeds is None else 18):
            y = self._norm(x, "_vit_ln1", i, eps=1e-6)
            q, k, v = self._project(y, "_vit_qkv", i).reshape(-1, 3, 16, 64).unbind(1)
            q, k = ops.rope(q, self.vit_cos, self.vit_sin), ops.rope(k, self.vit_cos, self.vit_sin)
            # Each image has independent attention, including different grids.
            if self.uniform_views:
                shape = (len(self.view_lengths), self.view_lengths[0], 16, 64)
                attended = ops.attention(q.reshape(shape), k.reshape(shape), v.reshape(shape))
                attended = attended.reshape(-1, 1024)
            else:
                parts, start = [], 0
                for size in self.view_lengths:
                    parts.append(self._attention(q[start:start + size], k[start:start + size],
                                                 v[start:start + size], 16, 64))
                    start += size
                attended = torch.cat(parts)
            x = x + self._project(attended, "_vit_o", i)
            y = self._norm(x, "_vit_ln2", i, eps=1e-6)
            y = F.gelu(self._project(y, "_vit_fc1", i), approximate="tanh")
            x = x + self._project(y, "_vit_fc2", i)
            self._save(f"vit_block_{i}", x)
            if i in (5, 11, 17):
                taps.append(self._merger(x, f"_dsm{len(taps)}", postshuffle=True))
        if llm_input_embeds is None:
            merged = self._merger(x, "_merger", postshuffle=False)
            h = F.embedding(self.ids, w._embed_tokens_w)
            h.index_copy_(0, self.image_index, merged)
        else:
            h = llm_input_embeds.reshape(-1, 2048).clone()
        self._save("llm_input_embeds", h)
        for i in range(16):
            y = ops.rms_norm(h, w._llm_input_ln_w[i])
            q, k, v = self._project(y, "_llm_qkv", i).split((2048, 1024, 1024), -1)
            q = ops.rms_norm(q.reshape(-1, 16, 128), w._llm_q_norm_w[i])
            k = ops.rms_norm(k.reshape(-1, 8, 128), w._llm_k_norm_w[i])
            q, k = ops.rope(q, self.llm_cos, self.llm_sin), ops.rope(k, self.llm_cos, self.llm_sin)
            y = self._attention(q, k, v, 16, 128, kv_heads=8, causal=True)
            h = h + self._project(y, "_llm_o", i)
            y = ops.rms_norm(h, w._llm_post_ln_w[i])
            y = F.silu(self._project(y, "_llm_gate", i)) * self._project(y, "_llm_up", i)
            h = h + self._project(y, "_llm_down", i)
            self._save(f"llm_layer_{i}", h)
            if i < len(taps):
                selected = h.index_select(0, self.image_index) + taps[i]
                h.index_copy_(0, self.image_index, selected)
        # GR00T selects hidden_states[16], before the truncated LLM's final norm.
        self._save("backbone_features", h)
        h = self._norm(h, "_vlln")
        for i in range(4):
            y = self._norm(h, "_vlsa_norm1", i)
            q, k, v = ops.linear(y, *self.vlsa_qkv[i]).chunk(3, -1)
            h = h + self._project(self._attention(q, k, v, 32, 64), "_vlsa_o", i)
            y = F.gelu(self._project(self._norm(h, "_vlsa_norm3", i), "_vlsa_fc1", i),
                       approximate="tanh")
            h = h + self._project(y, "_vlsa_fc2", i)
            self._save(f"vlsa_block_{i}", h)
        return h

    def _embodiment_linear(self, x, stem):
        return self.ops.linear(x, getattr(self.w, stem + "_W")) + getattr(self.w, stem + "_b")

    def denoise(self, features, state, noise):
        w, ops = self.w, self.ops
        state_h = self._embodiment_linear(F.relu(self._embodiment_linear(state, "_st_enc_l1")),
                                          "_st_enc_l2")
        text = features.index_select(0, self.text_index)
        image = features.index_select(0, self.image_index)
        # Recomputed inside each frame, reused across its denoising steps only.
        kv = {}
        for i in range(0, 32, 2):
            source = text if i % 4 == 0 else image
            kv[i] = ops.linear(source, *self.dit_kv[i]).chunk(2, -1)
        actions = noise
        for step in range(self.steps):
            encoded = self._embodiment_linear(actions, "_ac_enc_W1")
            encoded = torch.cat((encoded, self.tau[step].expand(self.horizon, -1)), -1)
            encoded = self._embodiment_linear(encoded, "_ac_enc_W2")
            encoded = encoded * torch.sigmoid(encoded)
            encoded = self._embodiment_linear(encoded, "_ac_enc_W3")
            h = torch.cat((state_h, encoded + w._ah_pos_embed_w[:self.horizon]), 0)
            scale, shift = self.modulators[0]
            y = ops.ada_layer_norm(h, scale[step], shift[step])
            for i in range(32):
                if i % 2:
                    q, k, v = ops.linear(y, *self.dit_qkv[i]).chunk(3, -1)
                else:
                    q = self._project(y, "_dit_q", i)
                    k, v = kv[i]
                update = self._project(self._attention(q, k, v, 32, 48), "_dit_o", i)
                y, h = ops.residual_layer_norm(update, h)
                y = F.gelu(self._project(y, "_dit_ff_proj", i), approximate="tanh")
                update = self._project(y, "_dit_ff_down", i)
                scale, shift = (self.modulators[i + 1] if i < 31
                                else (self.out_scale, self.out_shift))
                y, h = ops.residual_layer_norm(update, h, scale[step], shift[step],
                                               eps=1e-5 if i < 31 else 1e-6)
            h = self._project(y, "_proj_out_2")
            pred = self._embodiment_linear(F.relu(self._embodiment_linear(h, "_ac_dec_l1")),
                                          "_ac_dec_l2")
            actions = actions + (1.0 / self.steps) * pred[-self.horizon:]
            self._save(f"actions_step_{step}", actions)
        return actions.unsqueeze(0)

    def forward(self, patches, state, noise):
        return self.denoise(self.backbone(patches), state, noise)
