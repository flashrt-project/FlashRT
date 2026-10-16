"""Native Gemma3 text prefix for DM05 (34 layers) on preallocated buffers.

One ``forward`` fills the per-layer prefix K/V cache the action expert reads,
from the merged prefix embeddings (text rows + projected image tokens).
Per layer, following ``Gemma3DecoderLayer`` / ``Gemma3Attention``:

    xn = input_layernorm(x)
    q, k, v = xn Wq, xn Wk, xn Wv          k, v written straight into the cache
    q, k = q_norm(q), k_norm(k)            per head, in place
    q, k = rope(q), rope(k)                sliding (10k) or full (1e6, linear x8) table
    a = attention(q, K, V)                 FA2, block-diagonal prefix as causal/full segments
    x = x + post_attention_layernorm(a Wo)
    x = x + post_feedforward_layernorm(down(gelu_tanh(xn' Wg) * xn' Wu)),  xn' = pre_ff_norm(x)

Weights are used in place in nn.Linear's (N, K) layout (``gemm_bf16(b_nk=True)``),
so the pipeline holds no second copy of the ~8 GB text tower. Every elementwise
step rounds to bf16 where HF rounds, so the result can be compared layer by layer.
"""
from __future__ import annotations

import torch

from . import _ext
from ._release import release
from .attention import _bshd_strides, _fwd, _scratch, _PREFIX_SPLIT

bf16 = torch.bfloat16


def _rope_tables(cfg, layer_type: str, max_pos: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) bf16 tables of shape (max_pos, head_dim), computed as HF does.

    Gemma3RotaryEmbedding forms inv_freq in fp32, divides by the linear factor for
    `linear` rope, takes cos/sin of (pos * inv_freq) in fp32, multiplies by the
    attention scaling (1.0 for both types here) and casts to the activation dtype.
    """
    rp = cfg.rope_parameters[layer_type]
    base = float(rp["rope_theta"])
    dim = int(cfg.head_dim)
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
    if rp.get("rope_type", "default") == "linear":
        inv = inv / float(rp["factor"])
    pos = torch.arange(max_pos, device=device).float()
    freqs = pos[:, None] * inv[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(bf16).contiguous(), emb.sin().to(bf16).contiguous()


class NativePrefix:
    """Gemma3 text tower: merged embeddings -> per-layer K/V cache."""

    def __init__(self, language_model, max_seq: int, device="cuda"):
        cfg = language_model.config
        self.cfg = cfg
        self.L = int(cfg.num_hidden_layers)
        self.D = int(cfg.hidden_size)
        self.FF = int(cfg.intermediate_size)
        self.NH = int(cfg.num_attention_heads)
        self.NKV = int(cfg.num_key_value_heads)
        self.HD = int(cfg.head_dim)
        self.eps = float(cfg.rms_norm_eps)
        self.scale = float(cfg.query_pre_attn_scalar) ** -0.5
        self.max_seq = int(max_seq)
        self.layer_types = list(cfg.layer_types)
        d = torch.device(device)
        self.embed = language_model.embed_tokens
        # Gemma3TextScaledWordEmbedding multiplies by embed_scale cast to the weight
        # dtype, i.e. by the bf16 value of sqrt(hidden_size), not the fp32 one.
        self.embed_scale = float(self.embed.embed_scale.to(self.embed.weight.dtype).float())

        self.layers = []
        for ly in language_model.layers:
            a = ly.self_attn
            self.layers.append(dict(
                in_norm=ly.input_layernorm.weight, q=a.q_proj.weight, k=a.k_proj.weight,
                v=a.v_proj.weight, o=a.o_proj.weight, q_norm=a.q_norm.weight,
                k_norm=a.k_norm.weight, post_attn=ly.post_attention_layernorm.weight,
                pre_ff=ly.pre_feedforward_layernorm.weight, gate=ly.mlp.gate_proj.weight,
                up=ly.mlp.up_proj.weight, down=ly.mlp.down_proj.weight,
                post_ff=ly.post_feedforward_layernorm.weight))
        self.final_norm = language_model.norm.weight
        self.rope = {t: _rope_tables(cfg, t, self.max_seq, d)
                     for t in ("sliding_attention", "full_attention")}

        S, D, z = self.max_seq, self.D, lambda *s: torch.empty(*s, dtype=bf16, device=d)
        self.x = z(S, D)
        self.xn = z(S, D)
        self.q = z(S, self.NH * self.HD)
        self.attn = z(S, self.NH * self.HD)
        self.tmp = z(S, D)
        self.g = z(S, self.FF)
        self.u = z(S, self.FF)
        # The prefix cache the action expert attends to: [L, S, NKV, HD] each.
        # Per-layer views; a fused q|k|v layer points q and its cache rows into one buffer.
        self.k_cache = list(z(self.L, S, self.NKV * self.HD).unbind(0))
        self.v_cache = list(z(self.L, S, self.NKV * self.HD).unbind(0))
        self.q_rows = [self.q] * self.L
        self.qkv_fused: dict[int, "Fp8Linear"] = {}
        self.last_hidden = z(S, D)
        self.ids = torch.zeros(S, dtype=torch.long, device=d)
        # [image-row map | per-row attention key limit], one buffer so the staged path
        # uploads both with one copy
        self.meta = torch.full((2 * S,), -1, dtype=torch.int32, device=d)
        self.img_index, self.klim = self.meta[:S], self.meta[S:]
        # pinned host mirrors: on the staged serving path the graph copies them to the
        # device on every replay, so rebinding a prompt issues no eager transfer at all
        self.host_ids = torch.zeros(S, dtype=torch.long, pin_memory=True)
        self.host_meta = torch.full((2 * S,), -1, dtype=torch.int32, pin_memory=True)
        self.host_img_index, self.host_klim = self.host_meta[:S], self.host_meta[S:]
        self.staged = False
        self.ext = _ext.load()
        self.fp4: dict[tuple[int, str], "Nvfp4Linear"] = {}
        self.single_call_attn = False
        # FA2 with the FP8 attention-output epilogue (None = stock FA2 + quantize)
        self.attn_fp8_epilogue = None
        self._nsms = torch.cuda.get_device_properties(d).multi_processor_count
        self.seq = 0
        self.segments: list[tuple[int, int, int, bool]] = []

    def enable_nvfp4(self, sites: dict[str, range]) -> list:
        """Route {'gate'|'up'|'down': layer range} through static NVFP4. Returns the sites."""
        from .nvfp4 import Nvfp4Linear
        for kind, layers in sites.items():
            for i in layers:
                self.fp4[(i, kind)] = Nvfp4Linear(self.layers[i][kind], self.max_seq,
                                                  name=f"lm.{i}.{kind}")
                release(self.layers[i][kind])
        return list(self.fp4.values())

    def fuse_qkv_fp8(self) -> list[int]:
        """One FP8 GEMM for q|k|v on layers whose three sites are static FP8 (they already share
        one quantized input). The fused GEMM applies v's weight scale to all N columns, so V is
        exact and q, k come out scaled by s_v/s_q, s_v/s_k, which q_norm/k_norm remove (RMS
        norm is scale-invariant). The output rows are [q | k | v] and serve directly as q and
        as the layer's K/V cache, so the GEMM writes the cache in place."""
        from .fp8 import Fp8Linear
        S, HQ, HK = self.max_seq, self.NH * self.HD, self.NKV * self.HD
        for i in range(self.L):
            fq, fk, fv = (self.fp4.get((i, k)) for k in ("q", "k", "v"))
            if not all(type(s) is Fp8Linear for s in (fq, fk, fv)):
                continue
            f = Fp8Linear.__new__(Fp8Linear)
            f.name, f.N, f.K = f"lm.{i}.qkv", HQ + 2 * HK, fq.K
            f.w8 = torch.cat([fq.w8, fk.w8, fv.w8]).contiguous()
            f.w_scale, f.autotune = fv.w_scale, fq.autotune
            for s in (fq, fk, fv):
                s.w8 = None
            buf = torch.empty(S, f.N, dtype=bf16, device=self.q.device)
            self.q_rows[i], self.k_cache[i], self.v_cache[i] = buf[:, :HQ], buf[:, HQ:HQ + HK], buf[:, HQ + HK:]
            self.qkv_fused[i] = f
        return list(self.qkv_fused)

    def enable_fp8(self, sites: dict[str, range]) -> list:
        """Route {'q'|'k'|'v'|'o'|'gate'|'up'|'down': layer range} through static per-tensor FP8.
        Sites share the Nvfp4Linear quantize/gemm interface, so the forward treats both alike."""
        from .fp8 import Fp8Linear
        for kind, layers in sites.items():
            for i in layers:
                if (i, kind) in self.fp4:
                    raise ValueError(f"lm.{i}.{kind} is already NVFP4")
                self.fp4[(i, kind)] = Fp8Linear(self.layers[i][kind], self.max_seq, slots=1,
                                                name=f"lm.{i}.{kind}.fp8")
                release(self.layers[i][kind])
        return list(self.fp4.values())

    @staticmethod
    def _key_limits(segments, s):
        """Row i of the prefix attends keys j < lim[i]: i + 1 on causal (text) runs, the run
        end on bidirectional (image) runs."""
        import numpy as np
        lim = np.empty(s, dtype=np.int32)
        for s0, s1, kk, causal in segments:
            lim[s0:s1] = np.arange(s0 + 1, s1 + 1, dtype=np.int32) if causal else kk
        return lim

    def set_prompt(self, input_ids: torch.Tensor, image_token_id: int,
                   segments: list[tuple[int, int, int, bool]]) -> None:
        """Bind one prompt: token ids, the image-row map and the attention segments.

        Runs on the host once per prompt, outside the replayed graph. ``segments``
        is the validated (q_start, q_end, k_len, causal) table from
        ``flash_rt_attention._segments_from_mask``.
        """
        ids = input_ids.reshape(-1)
        s = int(ids.numel())
        if s > self.max_seq:
            raise ValueError(f"prefix length {s} exceeds the preallocated {self.max_seq}")
        # the image-row map is derived on the host when the ids are host tensors (the
        # staged serving path), so rebinding a prompt costs two copies and no device sync
        if self.staged and not ids.is_cuda:
            # host path in numpy: no tensor ops, no device work, written into the pinned
            # mirrors the graph uploads
            import numpy as np
            a = ids.numpy()
            idx = np.full(s, -1, dtype=np.int32)
            m = a == image_token_id
            idx[m] = np.arange(int(m.sum()), dtype=np.int32)
            self.host_ids.numpy()[:s] = a
            self.host_img_index.numpy()[:s] = idx
            self.host_klim.numpy()[:s] = self._key_limits(segments, s)
        else:
            is_img = ids == image_token_id
            idx = torch.full_like(ids, -1, dtype=torch.int32)
            idx[is_img] = torch.arange(int(is_img.sum()), device=ids.device, dtype=torch.int32)
            self.ids[:s].copy_(ids, non_blocking=True)
            self.img_index[:s].copy_(idx, non_blocking=True)
            self.klim[:s].copy_(torch.from_numpy(self._key_limits(segments, s)), non_blocking=False)
        self.seq = s
        self.segments = list(segments)

    def _norm(self, x, w, out, rows, dim, stream, residual=None, x_stride=None):
        xs = dim if x_stride is None else x_stride
        if xs == dim and dim % 8 == 0 and dim // 8 <= 1024 and dim >= 1024:
            # full hidden-size rows: vectorized kernel (one global read, 16-byte accesses)
            self.ext.gemma_rms_norm_v8(x.data_ptr(), w.data_ptr(), 0 if out is None else out.data_ptr(),
                                       0 if residual is None else residual.data_ptr(), rows, dim,
                                       self.eps, stream)
            return
        self.ext.gemma_rms_norm(x.data_ptr(), xs, w.data_ptr(),
                                0 if out is None else out.data_ptr(), xs,
                                0 if residual is None else residual.data_ptr(), dim,
                                rows, rows, 0, dim, self.eps, stream)

    def _norm_quant(self, x, w, site, rows, stream) -> bool:
        """Gemma RMSNorm straight into `site`'s static quantized input (FP8 or clipped
        NVFP4), bit-identical to _norm + site.quantize. False when the site is not frozen
        yet (calibration) or the row is not a full vectorizable hidden row."""
        from .fp8 import Fp8Linear
        from .nvfp4 import Nvfp4Linear
        D = self.D
        if not site.calibrated or site.calibrating or D % 16 or D // 8 > 1024:
            return False
        if isinstance(site, Fp8Linear):
            self.ext.gemma_rms_norm_quant_v8(x.data_ptr(), w.data_ptr(), site.a_scale.data_ptr(),
                                             site.a8.data_ptr(), 0, rows, D, self.eps, 0, stream)
            site._last_scale = site.a_scale[0:1]
            return True
        if isinstance(site, Nvfp4Linear):
            self.ext.gemma_rms_norm_quant_v8(x.data_ptr(), w.data_ptr(), site.group_amax.data_ptr(),
                                             site.a_packed.data_ptr(), site.a_sfa.data_ptr(), rows, D,
                                             self.eps, 1, stream)
            return True
        return False

    def set_prompt_host(self, ids, image_token_id: int, segments) -> None:
        """Staged-path prompt binding from a numpy id array (no tensor ops)."""
        import numpy as np
        s = int(ids.shape[0])
        if s > self.max_seq:
            raise ValueError(f"prefix length {s} exceeds the preallocated {self.max_seq}")
        idx = np.full(s, -1, dtype=np.int32)
        m = ids == image_token_id
        idx[m] = np.arange(int(m.sum()), dtype=np.int32)
        self._host_ids_np = getattr(self, "_host_ids_np", None)
        if self._host_ids_np is None:
            self._host_ids_np, self._host_img_np = self.host_ids.numpy(), self.host_img_index.numpy()
        self._host_ids_np[:s] = ids
        self._host_img_np[:s] = idx
        self.host_klim.numpy()[:s] = self._key_limits(segments, s)
        self.seq = s
        self.segments = list(segments)

    def upload_staged(self) -> None:
        """Graph-captured H2D of the pinned prompt mirrors (staged serving path)."""
        self.ids.copy_(self.host_ids, non_blocking=True)
        self.meta.copy_(self.host_meta, non_blocking=True)

    def forward(self, image_tokens: torch.Tensor, stream: int) -> None:
        """image_tokens: (n_img, D) projector output; fills k_cache/v_cache[:, :seq]."""
        e, S, D = self.ext, self.seq, self.D
        mm = e.gemm_bf16
        p = lambda t: t.data_ptr()
        HQ, HK = self.NH * self.HD, self.NKV * self.HD
        e.embed_merge(p(self.ids), p(self.img_index), p(self.embed.weight), p(image_tokens),
                      p(self.x), S, D, self.embed_scale, stream)
        for i, w in enumerate(self.layers):
            kc, vc = self.k_cache[i], self.v_cache[i]
            cos, sin = self.rope[self.layer_types[i]]
            fq, fk, fv = self.fp4.get((i, "q")), self.fp4.get((i, "k")), self.fp4.get((i, "v"))
            prod = fq or fk or fv
            if prod is not None and fq and fk and fv and self._norm_quant(self.x, w["in_norm"], prod, S, stream):
                pass                                   # q/k/v read only the quantized normed input
            else:
                self._norm(self.x, w["in_norm"], self.xn, S, D, stream)
                if prod is not None:
                    prod.quantize(self.xn, S, stream)  # q/k/v read the same normed input
            q, fused = self.q_rows[i], self.qkv_fused.get(i)
            if fused is not None:
                fused.gemm(q.as_strided((S, fused.N), (fused.N, 1)), S, stream, act=prod)
            else:
                for site, wt, dst, n in ((fq, w["q"], q, HQ), (fk, w["k"], kc, HK), (fv, w["v"], vc, HK)):
                    if site is not None:
                        site.gemm(dst, S, stream, act=prod if site is not prod else None)
                    else:
                        mm(p(self.xn), p(wt), p(dst), 0, S, n, D, stream, True)
            e.qk_norm_rope_v8(p(q), q.stride(0), self.NH, p(kc), kc.stride(0), self.NKV, p(w["q_norm"]),
                           p(w["k_norm"]), p(cos), p(sin), S, self.eps, stream)
            fo = self.fp4.get((i, "o"))
            quantized = self._attention(q, kc[:S], vc[:S], stream, o_site=fo)
            if fo is not None:
                if not quantized:
                    fo.quantize(self.attn, S, stream)
                fo.gemm(self.tmp, S, stream)
            else:
                mm(p(self.attn), p(w["o"]), p(self.tmp), 0, S, D, HQ, stream, True)
            self._norm(self.tmp, w["post_attn"], None, S, D, stream, residual=self.x)
            fg, fu, fd = self.fp4.get((i, "gate")), self.fp4.get((i, "up")), self.fp4.get((i, "down"))
            if fg is not None and fu is not None and self._norm_quant(self.x, w["pre_ff"], fg, S, stream):
                fg.gemm(self.g, S, stream)
            elif fg is not None:
                self._norm(self.x, w["pre_ff"], self.xn, S, D, stream)
                fg.quantize(self.xn, S, stream)
                fg.gemm(self.g, S, stream)
            else:
                self._norm(self.x, w["pre_ff"], self.xn, S, D, stream)
            if fg is None:
                mm(p(self.xn), p(w["gate"]), p(self.g), 0, S, self.FF, D, stream, True)
            if fu is not None:
                if fg is None:
                    fu.quantize(self.xn, S, stream)
                fu.gemm(self.u, S, stream, act=fg)
            else:
                mm(p(self.xn), p(w["up"]), p(self.u), 0, S, self.FF, D, stream, True)
            from .fp8 import Fp8Linear
            from .nvfp4 import Nvfp4Linear
            if isinstance(fd, Fp8Linear) and fd.calibrated and not fd.calibrating:
                # gelu(gate) * up straight into down_proj's static FP8 input (bit-identical)
                e.gelu_mul_quant_fp8_v8(p(self.g), p(self.u), p(fd.a_scale), p(fd.a8), S * self.FF, stream)
                fd.gemm(self.tmp, S, stream)
            elif isinstance(fd, Nvfp4Linear) and fd.calibrated and not fd.calibrating:
                # gelu(gate) * up straight into down_proj's clipped static NVFP4 input (bit-identical)
                e.gelu_mul_quant_nvfp4(p(self.g), p(self.u), p(fd.group_amax), p(fd.a_packed), p(fd.a_sfa),
                                       S, self.FF, stream)
                fd.gemm(self.tmp, S, stream)
            elif fd is not None:
                e.gelu_mul_bf16(p(self.g), p(self.u), p(self.g), S * self.FF, stream)
                fd.quantize(self.g, S, stream)
                fd.gemm(self.tmp, S, stream)
            else:
                e.gelu_mul_bf16(p(self.g), p(self.u), p(self.g), S * self.FF, stream)
                mm(p(self.g), p(w["down"]), p(self.tmp), 0, S, D, self.FF, stream, True)
            self._norm(self.tmp, w["post_ff"], None, S, D, stream, residual=self.x)
        self._norm(self.x, self.final_norm, self.last_hidden, S, D, stream)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, stream: int,
                   o_site=None) -> bool:
        """Run the prefix attention. Returns True when the kernel also wrote `o_site`'s
        static FP8 input, so the caller can skip the standalone quantize pass."""
        S = self.seq
        if self.single_call_attn:
            if q.stride(0) != q.shape[1] or k.stride(0) != k.shape[1]:
                raise RuntimeError("prefix_attn takes contiguous q/k/v rows; disable prefix_fuse_qkv")
            # one launch per layer: the segment mask as a per-row key limit
            self.ext.prefix_attn(q.data_ptr(), k.data_ptr(), v.data_ptr(), self.attn.data_ptr(),
                                 self.klim.data_ptr(), S, self.NH, self.NKV, self.scale, stream)
            return False
        q4 = q[:S].view(1, S, self.NH, self.HD).transpose(1, 2)
        k4 = k.view(1, S, self.NKV, self.HD).transpose(1, 2)
        v4 = v.view(1, S, self.NKV, self.HD).transpose(1, 2)
        o4 = self.attn[:S].view(1, S, self.NH, self.HD)
        qs, ks, vs = _bshd_strides(q4), _bshd_strides(k4), _bshd_strides(v4)
        os_ = (o4.stride(0), o4.stride(1), o4.stride(2))
        esz = 2
        q_row, o_row = q4.stride(2), o4.stride(1)
        ext = self.attn_fp8_epilogue
        from .fp8 import Fp8Linear
        fuse = (ext is not None and isinstance(o_site, Fp8Linear) and o_site.calibrated
                and not o_site.calibrating and self.HD == 256
                )
        if fuse:
            # The FP8 copy is o_proj's only consumer here, so the bf16 store is skipped.
            a8, sa = o_site.a8, o_site.a_scale[0:1]
            o8s = (a8.shape[0] * a8.shape[1], a8.shape[1], self.HD)
            for s0, s1, kk, causal in self.segments:
                # split exactly where _fwd splits, so the summation order - and the
                # result - is the one the stock path produces
                split = causal and _PREFIX_SPLIT
                lse, lse_a, o_a, _ = _scratch(q.device, 1, self.NH, s1 - s0, self.HD, split)
                ext.fwd_bf16(
                    q4.data_ptr() + s0 * q_row * esz, k4.data_ptr(), v4.data_ptr(),
                    o4.data_ptr() + s0 * o_row * esz, lse, lse_a, o_a,
                    1, s1 - s0, kk, self.NH, self.NKV, self.HD, qs, ks, vs, os_,
                    self.scale, self._nsms, stream,
                    a8.data_ptr() + s0 * a8.shape[1], sa.data_ptr(), o8s, True, causal)
            o_site._last_scale = sa
            return True
        for s0, s1, kk, causal in self.segments:
            _fwd(q4.data_ptr() + s0 * q_row * esz, k4.data_ptr(), v4.data_ptr(),
                 o4.data_ptr() + s0 * o_row * esz, qs, ks, vs, os_, 1, s1 - s0, kk,
                 self.NH, self.NKV, self.HD, q.device, self.scale, causal,
                 causal and _PREFIX_SPLIT)
        return False
