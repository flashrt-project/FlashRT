"""Native DM05 action expert: the 10-step flow-matching loop on the prefix cache.

Per step, following ``DM05ActionExpert._compute_suffix_layer`` and
``DM05ForConditionalGeneration`` sampling:

    h = bf16(linear_fp32(x_t, action_in_proj))
    34 x [ n, g = ada_norm(h, input_mod)       # (1+scale)*norm + shift, fp32, one rounding
           q, k, v = n Wq, n Wk, n Wv;  q, k = rope(q_norm(q)), rope(k_norm(k))
           k, v -> prefix cache rows [P, P+T)  # the suffix attends [prefix | itself]
           a = attention(q, K[:P+T], V[:P+T])  # no mask: every non-pad prefix row is visible
           h = h + bf16(post_attn_norm(a Wo) * g)
           n, g = ada_norm(h, mlp_mod);  h = h + bf16(post_ff_norm(down(gelu(n Wg) * n Wu)) * g) ]
    h = ada_norm(h, final_mod)
    v_t = linear_fp32(h, action_out_proj);  x_t = x_t + v_t * dt          (fp32)

The time modulations depend only on the fixed schedule and frozen weights, so they
are computed once with the repository's own ``precompute_suffix_modulations`` and
held as static per-step buffers.
"""
from __future__ import annotations

import torch

from . import _ext
from ._release import release
from .text import _rope_tables
from .attention import _bshd_strides, _fwd

bf16 = torch.bfloat16


class NativeActionExpert:
    def __init__(self, dm05, prefix, steps: int = 10, chunk: int = 10, device="cuda"):
        """dm05: DM05ForConditionalGeneration; prefix: NativePrefix (owns the KV cache)."""
        m = dm05.model
        ae = m.action_expert
        cfg = ae.config
        self.prefix = prefix
        self.steps = int(steps)
        self.dt = -1.0 / self.steps
        self.L = len(ae.layers)
        self.D = int(cfg.hidden_size)
        self.FF = int(cfg.intermediate_size)
        self.NH = int(cfg.num_attention_heads)
        self.NKV = int(cfg.num_key_value_heads)
        self.HD = int(cfg.head_dim)
        self.eps = float(cfg.rms_norm_eps)
        self.scale = float(cfg.query_pre_attn_scalar) ** -0.5
        self.T = int(chunk)   # served chunk length (the config default of 50 is the training length)
        self.A = int(m.config.action_dim)
        self.layer_types = list(cfg.layer_types)
        if prefix.NKV != self.NKV or prefix.HD != self.HD or prefix.L != self.L:
            raise ValueError("action expert and prefix cache geometry disagree")
        d = torch.device(device)

        self.layers = [dict(q=ly.self_attn.q_proj.weight, k=ly.self_attn.k_proj.weight,
                            v=ly.self_attn.v_proj.weight, o=ly.self_attn.o_proj.weight,
                            q_norm=ly.self_attn.q_norm.weight, k_norm=ly.self_attn.k_norm.weight,
                            post_attn=ly.post_attention_layernorm.weight,
                            gate=ly.mlp.gate_proj.weight, up=ly.mlp.up_proj.weight,
                            down=ly.mlp.down_proj.weight,
                            post_ff=ly.post_feedforward_layernorm.weight)
                       for ly in ae.layers]
        self.w_in = m.action_in_proj.weight.detach().float().contiguous()
        self.b_in = m.action_in_proj.bias.detach().float().contiguous()
        self.w_out = m.action_out_proj.weight.detach().float().contiguous()
        self.b_out = m.action_out_proj.bias.detach().float().contiguous()

        with torch.no_grad():
            _, in_mods, mlp_mods, fin_mods = dm05.precompute_suffix_modulations(self.steps, bf16)
        # [step][layer] -> (3D,) bf16, flattened into one static buffer per kind
        self.in_mod = torch.stack([torch.stack([x.reshape(-1) for x in s]) for s in in_mods]).contiguous()
        self.mlp_mod = torch.stack([torch.stack([x.reshape(-1) for x in s]) for s in mlp_mods]).contiguous()
        self.fin_mod = torch.stack([x.reshape(-1) for x in fin_mods]).contiguous()

        T, D, z = self.T, self.D, lambda *s, dt=bf16: torch.empty(*s, dtype=dt, device=d)
        self.x_t = z(T, self.A, dt=torch.float32)       # the action state, fp32 after step 0
        self.noise = z(T, self.A)                       # bf16 initial noise
        self.noise_f32 = z(T, self.A, dt=torch.float32)  # its exact fp32 upcast, refreshed by set_noise
        self.v_t = z(T, self.A, dt=torch.float32)
        self.h = z(T, D)
        self.n = z(T, D)
        self.q = z(T, self.NH * self.HD)
        self.attn = z(T, self.NH * self.HD)
        self.tmp = z(T, D)
        self.g = z(T, self.FF)
        self.u = z(T, self.FF)
        self.rope = {t: _rope_tables(cfg, t, prefix.max_seq + T, d)
                     for t in ("sliding_attention", "full_attention")}
        self.ext = _ext.load()
        self.act_mask = torch.ones(self.A, dtype=torch.float32, device=d)
        self.use_mask = False
        self.decode_splits = 0         # >0: native GQA decode attention with this many key splits
        self._da_ws = None
        self.int8w: dict = {}
        self.qkv8: dict = {}
        self.glu8: dict = {}

    def enable_int8w(self, kinds, layers=None) -> None:
        """Route the given projections through weight-only int8 (W8A16, tensor cores)."""
        from .int8w import Int8WLinear
        for i in (range(self.L) if layers is None else layers):
            for kind in kinds:
                self.int8w[(i, kind)] = Int8WLinear(self.layers[i][kind], name=f"ae.{i}.{kind}")
                release(self.layers[i][kind])

    def fuse_qkv_int8(self) -> None:
        """One W8A16 launch for q|k|v per layer-step (concatenated weights, three output
        segments: q buffer, and the K/V cache rows). Requires q, k, v to be int8w sites."""
        from .int8w import Int8WLinear
        self.qkv8 = {}
        for i in range(self.L):
            sq, sk, sv = (self.int8w.get((i, k)) for k in ("q", "k", "v"))
            if not (sq and sk and sv) or not all(type(x).__name__ == "Int8WLinear" for x in (sq, sk, sv)):
                continue
            f = Int8WLinear.__new__(Int8WLinear)
            f.name, f.N, f.K, f.ext = f"ae.{i}.qkv", sq.N + sk.N + sv.N, sq.K, sq.ext
            f.scale = torch.cat([sq.scale, sk.scale, sv.scale]).contiguous()
            # packed layout is [N/16][...] blocks, so concatenating along N concatenates blocks
            f.packed = torch.cat([sq.packed.reshape(-1), sk.packed.reshape(-1), sv.packed.reshape(-1)]).contiguous()
            f.warps = 8   # measured: 4 -> 91.64, 8 -> 90.27, 16 -> 92.27 ms end-to-end
            self.qkv8[i] = (f, sq.N, sk.N, sv.N)

    def fuse_glu_int8(self) -> None:
        """gate|up as one W8A16 launch with the gelu(gate) * up product in its epilogue.
        Rows are interleaved per 16-row block (8 gate channels, then the same 8 up
        channels), so each block owns matching gate/up outputs."""
        from .int8w import Int8WLinear
        self.glu8 = {}
        for i in range(self.L):
            sg, su = self.int8w.get((i, "gate")), self.int8w.get((i, "up"))
            if not (sg and su) or type(sg).__name__ != "Int8WLinear" or type(su).__name__ != "Int8WLinear":
                continue
            ff, kk = sg.N, sg.K
            f = Int8WLinear.__new__(Int8WLinear)
            f.name, f.N, f.K, f.ext = f"ae.{i}.glu", 2 * ff, kk, sg.ext
            q = torch.empty(2 * ff, kk, dtype=torch.int8, device=sg.packed.device)
            qg, qu = sg.q_rows, su.q_rows
            q.view(ff // 8, 2, 8, kk)[:, 0] = qg.view(ff // 8, 8, kk)
            q.view(ff // 8, 2, 8, kk)[:, 1] = qu.view(ff // 8, 8, kk)
            sc = torch.empty(2 * ff, dtype=torch.float32, device=q.device)
            sc.view(ff // 8, 2, 8)[:, 0] = sg.scale.view(ff // 8, 8)
            sc.view(ff // 8, 2, 8)[:, 1] = su.scale.view(ff // 8, 8)
            f.scale = sc.contiguous()
            f.packed = torch.empty_like(q)
            f.ext.w8a16_pack(q.data_ptr(), f.packed.data_ptr(), 2 * ff, kk, 0)
            f.warps = 8
            self.glu8[i] = (f, ff)
        torch.cuda.synchronize()

    def _lin(self, i, kind, x, out, n, k, step, stream):
        w8 = self.int8w.get((i, kind))
        if w8 is not None:
            w8.run(x, out, self.T, stream)
            return
        self.ext.gemm_bf16(x.data_ptr(), self.layers[i][kind].data_ptr(), out.data_ptr(), 0,
                           self.T, n, k, stream, True)

    def set_action_mask(self, mask: torch.Tensor | None) -> None:
        """Per-dimension action mask, applied to x_t before every step (as the reference).
        Copies only when the mask object changes, so a fixed serving mask costs nothing."""
        self.use_mask = mask is not None
        if mask is not None and mask is not getattr(self, "_mask_src", None):
            self.act_mask.copy_(mask)
            self._mask_src = mask

    def set_noise(self, noise: torch.Tensor) -> None:
        """Bind the initial noise (bf16, as drawn by the reference) and its fp32 upcast."""
        self.noise.copy_(noise)
        self.noise_f32.copy_(self.noise)

    def forward(self, stream: int) -> None:
        """self.noise (bf16) -> self.x_t (fp32 actions), using prefix.k/v_cache[:, :P]."""
        e, P, T, D = self.ext, self.prefix.seq, self.T, self.D
        mm, p = e.gemm_bf16, (lambda t: t.data_ptr())
        HQ, HK, n_act = self.NH * self.HD, self.NKV * self.HD, T * self.A
        grn = e.gate_residual_ada_norm_v8 if D % 256 == 0 and D // 8 <= 1024 else e.gate_residual_ada_norm
        for step in range(self.steps):
            if step == 0:
                e.init_rows_f32(p(self.x_t), p(self.noise_f32), p(self.act_mask) if self.use_mask else 0,
                                T, self.A, stream)
            elif self.use_mask:
                e.mul_rows_f32(p(self.x_t), p(self.act_mask), T, self.A, stream)
            e.lin_in_f32_bf16(p(self.x_t), p(self.w_in), p(self.b_in), p(self.h), T, D, self.A, stream)
            for i, w in enumerate(self.layers):
                kc = self.prefix.k_cache[i]
                vc = self.prefix.v_cache[i]
                cos, sin = self.rope[self.layer_types[i]]
                mod = self.in_mod[step, i]
                if i == 0:      # later layers get this norm fused into the previous residual
                    e.ada_rms_norm(p(self.h), p(mod), p(self.n), T, D, self.eps, stream)
                krow, vrow = kc[P:P + T], vc[P:P + T]
                fq = self.qkv8.get(i)
                if fq is not None:
                    f, nq, nk, nv = fq
                    e.w8a16_tc3(p(self.n), p(f.packed), p(f.scale), p(self.q), nq, HQ, p(krow), nk,
                                kc.stride(0), p(vrow), nv, vc.stride(0), T, D, f.warps, stream)
                else:
                    if kc.stride(0) != HK:
                        raise RuntimeError("strided prefix cache rows need fuse_action_qkv")
                    self._lin(i, "q", self.n, self.q, HQ, D, step, stream)
                    self._lin(i, "k", self.n, krow, HK, D, step, stream)
                    self._lin(i, "v", self.n, vrow, HK, D, step, stream)
                cos_p, sin_p = cos[P:P + T], sin[P:P + T]
                e.qk_norm_rope_v8(p(self.q), HQ, self.NH, p(krow), kc.stride(0), self.NKV, p(w["q_norm"]),
                               p(w["k_norm"]), p(cos_p), p(sin_p), T, self.eps, stream)
                self._attention(kc[:P + T], vc[:P + T], P + T)
                self._lin(i, "o", self.attn, self.tmp, D, HQ, step, stream)
                mmod = self.mlp_mod[step, i]
                grn(p(self.tmp), p(w["post_attn"]), p(mod[2 * D:]), p(self.h),
                                         p(mmod), p(self.n), T, D, self.eps, stream)
                fg = self.glu8.get(i)
                if fg is not None:
                    e.w8a16_tc_glu(p(self.n), p(fg[0].packed), p(fg[0].scale), p(self.g), fg[1], T, D,
                                   fg[0].warps, stream)
                else:
                    self._lin(i, "gate", self.n, self.g, self.FF, D, step, stream)
                    self._lin(i, "up", self.n, self.u, self.FF, D, step, stream)
                    e.gelu_mul_bf16(p(self.g), p(self.u), p(self.g), T * self.FF, stream)
                self._lin(i, "down", self.g, self.tmp, D, self.FF, step, stream)
                nmod = self.in_mod[step, i + 1] if i + 1 < self.L else self.fin_mod[step]
                grn(p(self.tmp), p(w["post_ff"]), p(mmod[2 * D:]), p(self.h),
                                         p(nmod), p(self.n), T, D, self.eps, stream)
            e.lin_out_bf16_f32(p(self.n), p(self.w_out), p(self.b_out), p(self.v_t), T, self.A, D, stream)
            e.axpy_f32(p(self.x_t), p(self.v_t), self.dt, n_act, stream)

    def _attention(self, k: torch.Tensor, v: torch.Tensor, sk: int) -> None:
        T = self.T
        if self.decode_splits:
            # GQA-packed decode attention: K/V read once per kv head, splits combined in-kernel
            if self._da_ws is None:
                d = self.q.device
                self._da_ws = (torch.empty(self.NKV * 32 * 32 * self.HD, device=d),
                               torch.empty(self.NKV * 32 * 32 * 2, device=d),
                               torch.zeros(self.NKV, dtype=torch.int32, device=d))
            wo, wml, cnt = self._da_ws
            assert k.stride(0) == v.stride(0)
            self.ext.ae_decode_attn(self.q.data_ptr(), self.NH * self.HD, k.data_ptr(), v.data_ptr(),
                                    k.stride(0), self.attn.data_ptr(), self.NH * self.HD,
                                    wo.data_ptr(), wml.data_ptr(), cnt.data_ptr(), T, self.NH // self.NKV,
                                    self.NKV, sk, self.decode_splits, self.scale,
                                    torch.cuda.current_stream().cuda_stream)
            return
        q4 = self.q.view(1, T, self.NH, self.HD).transpose(1, 2)
        k4 = k.view(1, sk, self.NKV, self.HD).transpose(1, 2)
        v4 = v.view(1, sk, self.NKV, self.HD).transpose(1, 2)
        o4 = self.attn.view(1, T, self.NH, self.HD)
        _fwd(q4.data_ptr(), k4.data_ptr(), v4.data_ptr(), o4.data_ptr(),
             _bshd_strides(q4), _bshd_strides(k4), _bshd_strides(v4),
             (o4.stride(0), o4.stride(1), o4.stride(2)),
             1, T, sk, self.NH, self.NKV, self.HD, self.q.device, self.scale, False, True)
