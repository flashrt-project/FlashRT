"""Native Thor Pi0.5 execution with OpenPI BF16/FP32 boundaries.

Runtime arithmetic: project CUDA/cuBLAS/cuDNN kernels plus FlashRT vendored FA2.
Torch is used for allocations, views, copies, graph orchestration, and setup
constants only. No OpenPI model execution and no ATen arithmetic in execute().
"""

import math

import numpy as np
import torch

from flash_rt.models.pi05.thor_bf16_weights import (
    EXPERT_ROOT,
    GEMMA,
    LANGUAGE_ROOT,
    PROJECTOR,
    VISION,
    VISION_ROOT,
)

try:
    from flash_rt import pi05_bf16_fa2_ext
    from flash_rt import pi05_bf16_native_ext as ext
except ImportError as error:
    raise ImportError("Build Pi0.5 Thor BF16 with -DFLASHRT_ENABLE_PI05_THOR_BF16=ON") from error


class Pi05ThorNativeBf16:
    def __init__(self, weights, tokens, token_mask):
        self.w = {n: t.detach().contiguous() for n, t in weights.items()}
        self.pool = []
        self.cursor = 0
        self.runner = ext.NativeRunner()
        self.vision = VISION_ROOT
        self.prefix = LANGUAGE_ROOT
        self.expert = EXPERT_ROOT
        self.projector = PROJECTOR
        self.vcfg = VISION
        self.pcfg = GEMMA
        self.ecfg = GEMMA
        self.vlayers = self.vcfg["num_hidden_layers"]
        self.veps = self.vcfg["layer_norm_eps"]
        self.hd = self.pcfg["head_dim"]
        self.fp32 = {n: w.float().contiguous() for n, w in self.w.items() if "norm" in n}
        self.patch_conv = ext.NativeConv()
        self.patch_weight_nhwc = (
            self.w[self.vision + ".embeddings.patch_embedding.weight"]
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        pad = np.concatenate([np.ones(256, dtype=bool), np.asarray(token_mask, dtype=bool)])
        pmask = np.where(pad[:, None] & pad[None, :], np.float32(0), np.float32(-2.3819763e38))[
            None, None
        ]
        smask = np.where(
            np.broadcast_to(np.r_[pad, np.ones(10, dtype=bool)], (10, len(pad) + 10)),
            np.float32(0),
            np.float32(-2.3819763e38),
        )[None, None]
        self.pmask = torch.from_numpy(pmask).cuda()
        self.smask = torch.from_numpy(smask).cuda()
        ppos = torch.from_numpy(np.cumsum(pad).astype(np.int64) - 1).cuda()[None, :]
        spos = torch.arange(10, device="cuda")[None, :] + int(pad.sum())
        self.prope = self._rope_constants(ppos, self.pcfg)
        self.erope = self._rope_constants(spos, self.ecfg)
        self.language = self.w[self.prefix + ".embed_tokens.weight"][
            torch.as_tensor(tokens, device="cuda", dtype=torch.long)
        ].contiguous()
        # Prompt embedding and rotary/time tables are setup constants, as in upstream.
        self.language = (self.language * math.sqrt(self.language.shape[-1])).contiguous()
        self.time_embeddings = []
        t = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        dt = torch.tensor(-0.1, device="cuda", dtype=torch.float32)
        dim = self.w["action_in_proj.weight"].shape[0]
        for i in range(10):
            fraction = torch.linspace(0.0, 1.0, dim // 2, device="cuda", dtype=torch.float64)
            period = 4e-3 * (4.0 / 4e-3) ** fraction
            factor = 1.0 / period * 2 * math.pi
            x = factor[None, :] * t.reshape(1, 1)
            self.time_embeddings.append(torch.cat([x.sin(), x.cos()], dim=1).float().contiguous())
            t += dt
        self.fa2 = pi05_bf16_fa2_ext

    @staticmethod
    def _rope_constants(pos, cfg):
        d = cfg["head_dim"]
        inv = 1.0 / (
            cfg.get("rope_theta", 10000.0)
            ** (torch.arange(0, d, 2, device="cuda", dtype=torch.int64).float() / d)
        )
        # OpenPI casts the complete Gemma module, including rotary inv_freq buffers,
        # to BF16 first. Its rotary forward later upcasts the rounded buffer to FP32.
        # Retaining the original full-FP32 inv_freq is a different numerical path.
        inv = inv.to(torch.bfloat16).float()
        freq = (inv[None, :, None].expand(1, -1, 1) @ pos[:, None, :].float()).transpose(1, 2)
        e = torch.cat([freq, freq], -1)
        return e.cos().to(torch.bfloat16).contiguous(), e.sin().to(torch.bfloat16).contiguous()

    def new(self, shape, dtype=torch.bfloat16):
        shape = tuple(shape)
        if self.cursor == len(self.pool):
            self.pool.append(torch.empty(shape, device="cuda", dtype=dtype))
        t = self.pool[self.cursor]
        self.cursor += 1
        if t.shape != shape or t.dtype != dtype:
            raise RuntimeError("Native execution geometry changed")
        return t

    @staticmethod
    def stream():
        return torch.cuda.current_stream().cuda_stream

    def cast(self, x, dtype):
        if x.dtype == dtype:
            return x
        y = self.new(x.shape, dtype)
        ext.cast(x.data_ptr(), y.data_ptr(), x.numel(), dtype == torch.bfloat16, self.stream())
        return y

    def binary(self, x, y, op=0):
        dtype = torch.float32 if torch.float32 in (x.dtype, y.dtype) else torch.bfloat16
        x = self.cast(x, dtype)
        y = self.cast(y, dtype)
        o = self.new(x.shape, dtype)
        width = 0 if x.numel() == y.numel() else y.numel()
        if width and width != x.shape[-1]:
            raise ValueError("Only last-dimension broadcast supported")
        ext.binary(
            x.data_ptr(),
            y.data_ptr(),
            o.data_ptr(),
            x.numel(),
            width,
            op,
            0 if dtype == torch.bfloat16 else 1,
            self.stream(),
        )
        return o

    def unary(self, x, kind, scale=1.0):
        y = self.new(x.shape, x.dtype)
        ext.unary(
            x.data_ptr(),
            y.data_ptr(),
            x.numel(),
            kind,
            float(scale),
            0 if x.dtype == torch.bfloat16 else 1,
            self.stream(),
        )
        return y

    def gemm(self, x, w, b=None):
        if x.dtype != w.dtype or b is not None and b.dtype != w.dtype:
            raise ValueError("Linear dtype differs")
        n, k = w.shape
        m = x.numel() // k
        o = self.new((*x.shape[:-1], n), x.dtype)
        self.runner.gemm(
            x.data_ptr(),
            w.data_ptr(),
            0 if b is None else b.data_ptr(),
            o.data_ptr(),
            m,
            n,
            k,
            0 if x.dtype == torch.bfloat16 else 1,
            self.stream(),
        )
        return o

    def linear(self, x, name):
        return self.gemm(x, self.w[name + ".weight"], self.w.get(name + ".bias"))

    def norm(self, x, name):
        o = self.new(x.shape)
        ext.norm(
            x.data_ptr(),
            self.fp32[name + ".weight"].data_ptr(),
            self.fp32[name + ".bias"].data_ptr(),
            o.data_ptr(),
            x.numel() // x.shape[-1],
            x.shape[-1],
            self.veps,
            self.stream(),
        )
        return o

    def rms(self, x, name, eps, cond=None):
        o = self.new(x.shape)
        gate = self.new(x.shape) if cond is not None else None
        style = self.linear(cond, name + ".dense") if cond is not None else None
        ext.rms(
            x.data_ptr(),
            0 if style is not None else self.fp32[name + ".weight"].data_ptr(),
            0 if style is None else style.data_ptr(),
            o.data_ptr(),
            0 if gate is None else gate.data_ptr(),
            x.numel() // x.shape[-1],
            x.shape[-1],
            float(eps),
            self.stream(),
        )
        return o, gate

    def rope(self, q, k, tables, heads, hk):
        oq = self.new(q.shape)
        ok = self.new(k.shape)
        cos, sin = tables
        s = q.shape[1]
        ext.rope(
            q.data_ptr(),
            k.data_ptr(),
            cos.data_ptr(),
            sin.data_ptr(),
            oq.data_ptr(),
            ok.data_ptr(),
            s,
            heads,
            hk,
            self.hd,
            self.stream(),
        )
        return oq, ok

    def cat(self, a, b):
        # Batch1 sequence-major tensors, exact byte copies into ordinary per-call KV.
        o = self.new((1, a.shape[1] + b.shape[1], *a.shape[2:]), a.dtype)
        nb = a.numel() * a.element_size()
        ext.copy(a.data_ptr(), o.data_ptr(), nb, self.stream())
        ext.copy(b.data_ptr(), o.data_ptr() + nb, b.numel() * b.element_size(), self.stream())
        return o

    def attention(self, q, k, v, mask, heads, hk, d):
        sq = q.shape[1]
        sk = k.shape[1]
        scores = self.new((heads, sq, sk))
        prob = self.new(scores.shape)
        o = self.new((1, sq, heads * d))
        self.runner.attention(
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            scores.data_ptr(),
            prob.data_ptr(),
            o.data_ptr(),
            mask.data_ptr(),
            sq,
            sk,
            heads,
            hk,
            d,
            d**-0.5,
            self.stream(),
        )
        return o

    def vision_attention(self, q, k, v, out):
        lse = self.new((1, 16, 256), torch.float32)
        self.fa2.vision(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), lse.data_ptr(), self.stream()
        )
        return out

    def image(self, image):
        base = self.vision
        emb = base + ".embeddings"
        nhwc = self.new((1, 224, 224, 3), torch.float32)
        ext.nchw_to_nhwc(image.data_ptr(), nhwc.data_ptr(), 3, 224 * 224, self.stream())
        x = self.new((1, 256, 1152), torch.float32)
        self.patch_conv.run(
            nhwc.data_ptr(), self.patch_weight_nhwc.data_ptr(), x.data_ptr(), self.stream()
        )
        x = self.binary(x, self.w[emb + ".patch_embedding.bias"])
        x = self.binary(x, self.w[emb + ".position_embedding.weight"].view(1, 256, 1152))
        x = self.cast(x, torch.bfloat16)
        for i in range(self.vlayers):
            l = base + f".encoder.layers.{i}"
            h = self.norm(x, l + ".layer_norm1")
            q = self.linear(h, l + ".self_attn.q_proj")
            k = self.linear(h, l + ".self_attn.k_proj")
            v = self.linear(h, l + ".self_attn.v_proj")
            o = self.new(q.shape)
            self.vision_attention(q, k, v, o)
            x = self.binary(x, self.linear(o, l + ".self_attn.out_proj"))
            h = self.norm(x, l + ".layer_norm2")
            h = self.linear(self.unary(self.linear(h, l + ".mlp.fc1"), 1), l + ".mlp.fc2")
            x = self.binary(x, h)
        x = self.norm(x, base + ".post_layernorm")
        return self.linear(x, self.projector)

    def blocks(self, h, *, expert=False, cond=None, past=None):
        base = self.expert if expert else self.prefix
        cfg = self.ecfg if expert else self.pcfg
        layers = cfg["num_hidden_layers"]
        heads = cfg["num_attention_heads"]
        hk = cfg["num_key_value_heads"]
        h = self.cast(h, torch.bfloat16)
        cache = []
        mask = self.smask if expert else self.pmask
        tables = self.erope if expert else self.prope
        for i in range(layers):
            l = base + f".layers.{i}"
            z, gate = self.rms(h, l + ".input_layernorm", cfg["rms_norm_eps"], cond)
            q = self.linear(z, l + ".self_attn.q_proj")
            k = self.linear(z, l + ".self_attn.k_proj")
            v = self.linear(z, l + ".self_attn.v_proj")
            q, k = self.rope(q, k, tables, heads, hk)
            if past is not None:
                k = self.cat(past[i][0], k)
                v = self.cat(past[i][1], v)
            else:
                cache.append((k, v))
            a = self.linear(
                self.attention(q, k, v, mask, heads, hk, self.hd), l + ".self_attn.o_proj"
            )
            h = self.binary(h, a if gate is None else self.binary(a, gate, 1))
            z, gate = self.rms(h, l + ".post_attention_layernorm", cfg["rms_norm_eps"], cond)
            z = self.linear(
                self.binary(
                    self.unary(self.linear(z, l + ".mlp.gate_proj"), 1),
                    self.linear(z, l + ".mlp.up_proj"),
                    1,
                ),
                l + ".mlp.down_proj",
            )
            h = self.binary(h, z if gate is None else self.binary(z, gate, 1))
        h, _ = self.rms(h, base + ".norm", cfg["rms_norm_eps"], cond)
        return h, cache

    @torch.no_grad()
    def execute(self, image, noise):
        if (
            image.shape != (1, 3, 224, 224)
            or not image.is_contiguous()
            or image.dtype != torch.float32
        ):
            raise ValueError("Native image contract differs")
        if noise.shape != (1, 10, 32) or noise.dtype != torch.float32:
            raise ValueError("Noise shape/dtype differs")
        self.cursor = 0
        x = self.image(image)
        prefix = self.cat(x, self.language.view(1, -1, 2048))
        _, cache = self.blocks(prefix)
        x = noise
        for te in self.time_embeddings:
            cond = self.unary(
                self.linear(self.unary(self.linear(te, "time_mlp_in"), 0), "time_mlp_out"), 0
            )
            h = self.linear(x, "action_in_proj")
            h, _ = self.blocks(h, expert=True, cond=cond, past=cache)
            v = self.linear(self.cast(h, torch.float32), "action_out_proj")
            o = self.new(x.shape, torch.float32)
            ext.euler(x.data_ptr(), v.data_ptr(), o.data_ptr(), x.numel(), -0.1, self.stream())
            x = o
        return x
