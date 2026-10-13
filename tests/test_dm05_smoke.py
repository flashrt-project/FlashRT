#!/usr/bin/env python3
"""DM05 kernel module smoke and numerical tests.

Covers the module built with -DFLASHRT_ENABLE_DM05=ON on GPU_ARCH=120:

  * the flash_rt_dm05 module imports and exposes every published symbol
  * each kernel matches an fp64 reference on its production shape
  * the head_dim-72 FP8 epilogue is byte-identical to quantize_fp8_static
  * unsupported shapes raise before launch instead of computing a result

The tests skip cleanly when the module is absent, which is the default build
(FLASHRT_ENABLE_DM05=OFF) and any non-sm120 target.

Usage:
  PYTHONPATH=. pytest -q tests/test_dm05_smoke.py
"""
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

HD72_NH = 16                  # SigLIP: 16 heads of width 72
HD72_D = HD72_NH * 72         # 1152
AHD = 256                     # action expert head width


def _dm05():
    """The DM05 module, or a skip when this build did not produce it."""
    try:
        from flash_rt import flash_rt_dm05 as m
    except ImportError as exc:
        pytest.skip(f"flash_rt_dm05 not built (FLASHRT_ENABLE_DM05=OFF?): {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    return m


def _stream():
    return torch.cuda.current_stream().cuda_stream


# ── import and symbol surface ────────────────────────────────────────────────

class TestModuleSurface:
    def test_default_build_imports_without_the_module(self):
        """flash_rt itself must import whether or not the DM05 module exists."""
        import flash_rt
        assert flash_rt is not None

    def test_published_symbols(self):
        m = _dm05()
        for sym in ("dm05_vit_attn_hd72", "dm05_vit_attn_hd72_smem_bytes",
                    "dm05_ae_decode_attn", "dm05_ae_decode_attn_supports_r",
                    "dm05_ae_decode_attn_smem_bytes",
                    "dm05_ae_decode_attn_ws_o_floats",
                    "dm05_ae_decode_attn_ws_ml_floats"):
            assert hasattr(m, sym), f"missing published symbol {sym}"

    def test_smem_fits_an_sm120_block(self):
        """Both kernels must opt in to a shared-memory size the device allows."""
        m = _dm05()
        limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
        assert 0 < m.dm05_vit_attn_hd72_smem_bytes() <= limit
        assert 0 < m.dm05_ae_decode_attn_smem_bytes() <= limit


# ── head_dim 72 SigLIP attention ─────────────────────────────────────────────

def _vit_reference(q, k, v, scale, nv, S):
    """Exact attention in fp64, per (view, head)."""
    q = q.double().reshape(nv, S, HD72_NH, 72).permute(0, 2, 1, 3)
    k = k.double().reshape(nv, S, HD72_NH, 72).permute(0, 2, 1, 3)
    v = v.double().reshape(nv, S, HD72_NH, 72).permute(0, 2, 1, 3)
    p = torch.softmax((q * scale) @ k.transpose(-1, -2), dim=-1)
    return (p @ v).permute(0, 2, 1, 3).reshape(nv * S, HD72_D).float()


def _run_vit(m, S, nv, mt, seed=0):
    torch.manual_seed(seed)
    qkv = torch.randn(nv * S, 3 * HD72_D, device="cuda", dtype=torch.bfloat16) * 0.5
    out = torch.zeros(nv * S, HD72_D, device="cuda", dtype=torch.bfloat16)
    scale = 72 ** -0.5
    m.dm05_vit_attn_hd72(qkv.data_ptr(), 3 * HD72_D, HD72_D, out.data_ptr(), HD72_D,
                         S, HD72_NH, nv, scale, mt, _stream(), 0, 0)
    torch.cuda.synchronize()
    ref = _vit_reference(qkv[:, :HD72_D], qkv[:, HD72_D:2 * HD72_D], qkv[:, 2 * HD72_D:],
                         scale, nv, S)
    return out, ref


class TestVitAttnHd72:
    @pytest.mark.parametrize("S,nv,mt", [
        (1024, 2, 2),     # the SigLIP production shape: two views of 1024 patches
        (256, 1, 2),
        (128, 1, 1),      # smallest legal sequence, query tile of one block
        (1024, 2, 1),
    ])
    def test_matches_fp64_reference(self, S, nv, mt):
        m = _dm05()
        out, ref = _run_vit(m, S, nv, mt)
        max_abs = (out.float() - ref).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            out.float().reshape(-1).double(), ref.reshape(-1).double(), dim=0).item()
        assert max_abs < 0.05, f"max abs error {max_abs}"
        assert cos > 0.9995, f"cosine {cos}"

    def test_query_tile_sizes_agree(self):
        """mt selects the query rows per CTA; it must not change the result."""
        m = _dm05()
        a, _ = _run_vit(m, 1024, 2, 2, seed=7)
        b, _ = _run_vit(m, 1024, 2, 1, seed=7)
        assert torch.equal(a, b), "mt=1 and mt=2 produced different output"

    def test_fp8_epilogue_matches_quantize_fp8_static(self):
        """The fused epilogue must reproduce the pass it replaces, bit for bit."""
        m = _dm05()
        fvk = pytest.importorskip("flash_rt.flash_rt_kernels")
        S, nv = 1024, 2
        torch.manual_seed(3)
        qkv = torch.randn(nv * S, 3 * HD72_D, device="cuda", dtype=torch.bfloat16) * 0.5
        out = torch.zeros(nv * S, HD72_D, device="cuda", dtype=torch.bfloat16)
        q8 = torch.zeros(nv * S, HD72_D, device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
        m.dm05_vit_attn_hd72(qkv.data_ptr(), 3 * HD72_D, HD72_D, out.data_ptr(), HD72_D,
                             S, HD72_NH, nv, 72 ** -0.5, 2, _stream(), 0, 0)
        torch.cuda.synchronize()
        bf16_out = out.clone()
        m.dm05_vit_attn_hd72(qkv.data_ptr(), 3 * HD72_D, HD72_D, out.data_ptr(), HD72_D,
                             S, HD72_NH, nv, 72 ** -0.5, 2, _stream(),
                             q8.data_ptr(), scale.data_ptr())
        torch.cuda.synchronize()
        # the FP8 mode replaces the bf16 store: `out` is left where the caller had it
        assert torch.equal(out, bf16_out), "FP8 mode must not write the bf16 output"
        ref8 = torch.zeros_like(q8)
        fvk.quantize_fp8_static(bf16_out.reshape(-1).data_ptr(), ref8.reshape(-1).data_ptr(),
                                scale.data_ptr(), nv * S * HD72_D, _stream())
        torch.cuda.synchronize()
        assert torch.equal(q8.view(torch.uint8), ref8.view(torch.uint8)), \
            "FP8 epilogue differs from quantize_fp8_static"

    @pytest.mark.parametrize("desc,S,NH,D,nv,mt", [
        # D is passed independently of NH so the D == NH * 72 guard is really hit
        ("D != NH*72", 1024, 15, 1152, 1, 2),
        ("S % 128 != 0", 1000, 16, HD72_D, 1, 2),
        ("mt out of range", 1024, 16, HD72_D, 1, 3),
    ])
    def test_shape_guards(self, desc, S, NH, D, nv, mt):
        m = _dm05()
        qkv = torch.zeros(nv * S, 3 * D, device="cuda", dtype=torch.bfloat16)
        out = torch.zeros(nv * S, D, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="dm05_vit_attn_hd72"):
            m.dm05_vit_attn_hd72(qkv.data_ptr(), 3 * D, D, out.data_ptr(), D,
                                 S, NH, nv, 72 ** -0.5, mt, _stream(), 0, 0)


# ── action-expert grouped-query decode attention ─────────────────────────────

T_CHUNK, G_QKV, NKV_HEADS = 10, 2, 4       # DM05 action expert: R = G * T = 20
NH_TOTAL = G_QKV * NKV_HEADS


def _ae_reference(q, k, v, scale, sk):
    """fp64 reference: query head hk*G+gg attends key/value head hk over [0, sk)."""
    qh = q.double().view(T_CHUNK, NH_TOTAL, AHD)
    kh = k.double().view(sk, NKV_HEADS, AHD)
    vh = v.double().view(sk, NKV_HEADS, AHD)
    out = torch.zeros(T_CHUNK, NH_TOTAL, AHD, dtype=torch.float64, device=q.device)
    for hk in range(NKV_HEADS):
        for gg in range(G_QKV):
            h = hk * G_QKV + gg
            p = torch.softmax((qh[:, h, :] * scale) @ kh[:, hk, :].T, dim=-1)
            out[:, h, :] = p @ vh[:, hk, :]
    return out.reshape(T_CHUNK, NH_TOTAL * AHD).float()


def _run_ae(m, sk, ns, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(T_CHUNK, NH_TOTAL * AHD, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(sk, NKV_HEADS * AHD, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn(sk, NKV_HEADS * AHD, device="cuda", dtype=torch.bfloat16) * 0.5
    out = torch.zeros(T_CHUNK, NH_TOTAL * AHD, device="cuda", dtype=torch.bfloat16)
    n_ws = int(m.dm05_ae_decode_attn_ws_o_floats(NKV_HEADS, 32))
    n_ml = int(m.dm05_ae_decode_attn_ws_ml_floats(NKV_HEADS, 32))
    ws_o = torch.zeros(n_ws, device="cuda", dtype=torch.float32)
    ws_ml = torch.zeros(n_ml, device="cuda", dtype=torch.float32)
    cnt = torch.zeros(NKV_HEADS, device="cuda", dtype=torch.int32)
    scale = AHD ** -0.5
    m.dm05_ae_decode_attn(q.data_ptr(), NH_TOTAL * AHD, k.data_ptr(), v.data_ptr(),
                          NKV_HEADS * AHD, out.data_ptr(), NH_TOTAL * AHD,
                          ws_o.data_ptr(), ws_ml.data_ptr(), cnt.data_ptr(),
                          T_CHUNK, G_QKV, NKV_HEADS, sk, ns, scale, _stream())
    torch.cuda.synchronize()
    return out, _ae_reference(q, k, v, scale, sk), cnt


class TestAeDecodeAttn:
    @pytest.mark.parametrize("sk,ns", [(568, 1), (568, 8), (128, 8), (33, 4), (1, 1)])
    def test_matches_fp64_reference(self, sk, ns):
        m = _dm05()
        out, ref, _ = _run_ae(m, sk, ns)
        max_abs = (out.float() - ref).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            out.float().reshape(-1).double(), ref.reshape(-1).double(), dim=0).item()
        assert max_abs < 0.05, f"max abs error {max_abs}"
        assert cos > 0.9995, f"cosine {cos}"

    def test_split_path_agrees_with_single_pass(self):
        m = _dm05()
        a, _, _ = _run_ae(m, 568, 1, seed=2)
        b, _, _ = _run_ae(m, 568, 8, seed=2)
        assert (a.float() - b.float()).abs().max().item() < 0.01

    def test_counters_are_reset_for_graph_replay(self):
        """The combine CTA zeroes its counter, so a replay needs no memset."""
        m = _dm05()
        _, _, cnt = _run_ae(m, 568, 8)
        assert int(cnt.abs().sum().item()) == 0

    def test_instantiated_row_counts(self):
        m = _dm05()
        supported = [r for r in range(1, 33) if m.dm05_ae_decode_attn_supports_r(r)]
        assert 20 in supported, "G*T = 20 (the DM05 action expert shape) must be instantiated"
        assert not m.dm05_ae_decode_attn_supports_r(33)

    def test_row_count_guard(self):
        """An R that is legal but not instantiated must fail fast, not mis-launch."""
        m = _dm05()
        q = torch.zeros(10, NH_TOTAL * AHD, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros(64, NKV_HEADS * AHD, device="cuda", dtype=torch.bfloat16)
        o = torch.zeros(10, NH_TOTAL * AHD, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="not an instantiated row count"):
            m.dm05_ae_decode_attn(q.data_ptr(), NH_TOTAL * AHD, k.data_ptr(), k.data_ptr(),
                                  NKV_HEADS * AHD, o.data_ptr(), NH_TOTAL * AHD, 0, 0, 0,
                                  T_CHUNK, 1, NKV_HEADS, 64, 1, AHD ** -0.5, _stream())

    def test_workspace_guard(self):
        m = _dm05()
        q = torch.zeros(T_CHUNK, NH_TOTAL * AHD, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros(64, NKV_HEADS * AHD, device="cuda", dtype=torch.bfloat16)
        o = torch.zeros(T_CHUNK, NH_TOTAL * AHD, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="workspaces"):
            m.dm05_ae_decode_attn(q.data_ptr(), NH_TOTAL * AHD, k.data_ptr(), k.data_ptr(),
                                  NKV_HEADS * AHD, o.data_ptr(), NH_TOTAL * AHD, 0, 0, 0,
                                  T_CHUNK, G_QKV, NKV_HEADS, 64, 4, AHD ** -0.5, _stream())


# ── Gemma3 prefix attention with a per-row key limit ─────────────────────────

PFX_HD, PFX_NH, PFX_NKV = 256, 8, 4


def _prefix_reference(q, k, v, lim, scale, S):
    """fp64 reference: row i attends to keys [0, lim[i]) on its key/value head."""
    g = PFX_NH // PFX_NKV
    qh = q.double().view(S, PFX_NH, PFX_HD)
    kh = k.double().view(S, PFX_NKV, PFX_HD)
    vh = v.double().view(S, PFX_NKV, PFX_HD)
    cols = torch.arange(S, device=q.device)[None, :]
    keep = cols < lim[:, None]                     # (S, S) per-row key limit
    out = torch.zeros(S, PFX_NH, PFX_HD, dtype=torch.float64, device=q.device)
    for hk in range(PFX_NKV):
        for gg in range(g):
            h = hk * g + gg
            sc = (qh[:, h, :] * scale) @ kh[:, hk, :].T
            sc = sc.masked_fill(~keep, float("-inf"))
            out[:, h, :] = torch.softmax(sc, dim=-1) @ vh[:, hk, :]
    return out.reshape(S, PFX_NH * PFX_HD).float()


def _prefix_limits(S, kind, seg=0):
    """causal (text), block-bidirectional (image), or mixed -- the real shape."""
    if kind == "causal":
        return torch.arange(1, S + 1, device="cuda", dtype=torch.int32)
    lim = torch.empty(S, dtype=torch.int32, device="cuda")
    if kind == "blocks":
        for a in range(0, S, seg):
            lim[a:min(a + seg, S)] = min(a + seg, S)
    else:                                          # mixed: text then one image block
        cut = S // 2
        lim[:cut] = torch.arange(1, cut + 1, device="cuda", dtype=torch.int32)
        lim[cut:] = S
    return lim


def _run_prefix(m, S, kind, seg=0, seed=0):
    torch.manual_seed(seed)
    lim = _prefix_limits(S, kind, seg)
    q = (torch.randn(S, PFX_NH * PFX_HD, device="cuda") * 0.5).to(torch.bfloat16)
    k = (torch.randn(S, PFX_NKV * PFX_HD, device="cuda") * 0.5).to(torch.bfloat16)
    v = (torch.randn(S, PFX_NKV * PFX_HD, device="cuda") * 0.5).to(torch.bfloat16)
    out = torch.zeros(S, PFX_NH * PFX_HD, device="cuda", dtype=torch.bfloat16)
    scale = PFX_HD ** -0.5
    m.dm05_prefix_attn(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
                       lim.data_ptr(), S, PFX_NH, PFX_NKV, scale, _stream())
    torch.cuda.synchronize()
    return out, _prefix_reference(q, k, v, lim, scale, S)


class TestPrefixAttn:
    @pytest.mark.parametrize("S,kind,seg", [
        (568, "mixed", 0),        # the served shape: text then an image block
        (568, "blocks", 256),
        (568, "causal", 0),
        (130, "blocks", 64),      # not a multiple of the 64-row tile
        (1, "causal", 0),
    ])
    def test_matches_fp64_reference(self, S, kind, seg):
        m = _dm05()
        out, ref = _run_prefix(m, S, kind, seg)
        max_abs = (out.float() - ref).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            out.float().reshape(-1).double(), ref.reshape(-1).double(), dim=0).item()
        assert max_abs < 0.05, f"max abs error {max_abs}"
        assert cos > 0.9995, f"cosine {cos}"

    def test_the_key_limit_is_actually_enforced(self):
        """A kernel ignoring lim would attend to every key and still look right."""
        m = _dm05()
        S = 256
        torch.manual_seed(4)
        q = (torch.randn(S, PFX_NH * PFX_HD, device="cuda") * 0.5).to(torch.bfloat16)
        k = (torch.randn(S, PFX_NKV * PFX_HD, device="cuda") * 0.5).to(torch.bfloat16)
        v = (torch.randn(S, PFX_NKV * PFX_HD, device="cuda") * 0.5).to(torch.bfloat16)
        scale = PFX_HD ** -0.5
        outs = []
        for lim_val in (S, S // 2):
            lim = torch.full((S,), lim_val, device="cuda", dtype=torch.int32)
            o = torch.zeros(S, PFX_NH * PFX_HD, device="cuda", dtype=torch.bfloat16)
            m.dm05_prefix_attn(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
                               lim.data_ptr(), S, PFX_NH, PFX_NKV, scale, _stream())
            torch.cuda.synchronize()
            outs.append(o)
        assert not torch.equal(outs[0], outs[1]), "the key limit changed nothing"

    def test_row_zero_attends_to_exactly_one_key(self):
        """With lim[0] == 1 the first row's output must equal its own value head."""
        m = _dm05()
        S = 128
        torch.manual_seed(5)
        lim = _prefix_limits(S, "causal")
        q = (torch.randn(S, PFX_NH * PFX_HD, device="cuda") * 0.5).to(torch.bfloat16)
        k = (torch.randn(S, PFX_NKV * PFX_HD, device="cuda") * 0.5).to(torch.bfloat16)
        v = (torch.randn(S, PFX_NKV * PFX_HD, device="cuda") * 0.5).to(torch.bfloat16)
        out = torch.zeros(S, PFX_NH * PFX_HD, device="cuda", dtype=torch.bfloat16)
        m.dm05_prefix_attn(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
                           lim.data_ptr(), S, PFX_NH, PFX_NKV, PFX_HD ** -0.5, _stream())
        torch.cuda.synchronize()
        g = PFX_NH // PFX_NKV
        for hk in range(PFX_NKV):
            for gg in range(g):
                h = hk * g + gg
                want = v[0, hk * PFX_HD:(hk + 1) * PFX_HD]
                assert torch.equal(out[0, h * PFX_HD:(h + 1) * PFX_HD], want), f"head {h}"

    def test_guards(self):
        m = _dm05()
        S = 64
        q = torch.zeros(S, PFX_NH * PFX_HD, device="cuda", dtype=torch.bfloat16)
        out = torch.zeros_like(q)
        lim = torch.zeros(S, device="cuda", dtype=torch.int32)
        with pytest.raises(RuntimeError, match="null pointer"):
            m.dm05_prefix_attn(q.data_ptr(), q.data_ptr(), q.data_ptr(), out.data_ptr(),
                               0, S, PFX_NH, PFX_NKV, 1.0, _stream())
        torch.cuda.synchronize()
        with pytest.raises(RuntimeError, match="multiple of NKV"):
            m.dm05_prefix_attn(q.data_ptr(), q.data_ptr(), q.data_ptr(), out.data_ptr(),
                               lim.data_ptr(), S, 6, 4, 1.0, _stream())


# ── weight-only int8 GEMM (W8A16) ────────────────────────────────────────────

W8_SEQ = [(10, 1024, 32, 4), (10, 4096, 1024, 8), (1, 32, 1024, 4), (16, 128, 64, 2)]


def _w8_case(m, N, K, seed=0):
    """An int8 weight, its per-channel scale, and the packed form the GEMMs read."""
    torch.manual_seed(seed)
    w8 = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8)
    sc = (torch.rand(N, device="cuda") * 0.01 + 0.001).to(torch.float32)
    packed = torch.zeros(N * K, device="cuda", dtype=torch.int8)
    m.dm05_w8a16_pack(w8.data_ptr(), packed.data_ptr(), N, K, _stream())
    torch.cuda.synchronize()
    return w8, sc, packed


def _w8_ref(x, w8, sc):
    """fp32 accumulate over the dequantized weight, scale applied after the reduction."""
    return ((x.float() @ w8.float().t()) * sc[None, :]).to(torch.bfloat16)


class TestW8A16:
    @pytest.mark.parametrize("M,N,K,warps", W8_SEQ)
    def test_gemm_matches_a_reference(self, M, N, K, warps):
        m = _dm05()
        w8, sc, packed = _w8_case(m, N, K, seed=1)
        torch.manual_seed(2)
        x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
        out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
        m.dm05_w8a16_tc(x.data_ptr(), packed.data_ptr(), sc.data_ptr(), out.data_ptr(),
                        M, N, K, warps, _stream())
        torch.cuda.synchronize()
        ref = _w8_ref(x, w8, sc)
        max_abs = (out.float() - ref.float()).abs().max().item()
        assert max_abs < 0.05, f"max abs error {max_abs}"

    def test_the_scale_is_applied_per_output_channel(self):
        """A kernel folding the scale in wrongly still passes a loose check."""
        m = _dm05()
        N, K, M = 64, 32, 8
        w8, sc, packed = _w8_case(m, N, K, seed=3)
        torch.manual_seed(4)
        x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
        out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
        m.dm05_w8a16_tc(x.data_ptr(), packed.data_ptr(), sc.data_ptr(), out.data_ptr(),
                        M, N, K, 4, _stream())
        torch.cuda.synchronize()
        unscaled = (x.float() @ w8.float().t()).to(torch.bfloat16)
        assert not torch.equal(out, unscaled)

    def test_pack_is_a_permutation_of_the_weight(self):
        """The packed layout must be the same bytes, reordered."""
        m = _dm05()
        N, K = 128, 64
        w8, _, packed = _w8_case(m, N, K, seed=5)
        assert sorted(packed.tolist()) == sorted(w8.reshape(-1).tolist())

    def test_segmented_output_writes_each_leading_dimension(self):
        """k and v must land in their own buffers with their own row stride."""
        m = _dm05()
        N, K, M = 4096, 1024, 10
        w8, sc, packed = _w8_case(m, N, K, seed=6)
        torch.manual_seed(7)
        x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
        o0 = torch.zeros(M, 2048, device="cuda", dtype=torch.bfloat16)
        o1 = torch.zeros(M, 4096, device="cuda", dtype=torch.bfloat16)
        o2 = torch.zeros(M, 4096, device="cuda", dtype=torch.bfloat16)
        m.dm05_w8a16_tc3(x.data_ptr(), packed.data_ptr(), sc.data_ptr(),
                         o0.data_ptr(), 2048, 2048, o1.data_ptr(), 1024, 4096,
                         o2.data_ptr(), 1024, 4096, M, K, 8, _stream())
        torch.cuda.synchronize()
        full = _w8_ref(x, w8, sc)
        assert (o0.float() - full[:, :2048].float()).abs().max().item() < 0.05
        assert (o1[:, :1024].float() - full[:, 2048:3072].float()).abs().max().item() < 0.05
        assert (o2[:, :1024].float() - full[:, 3072:].float()).abs().max().item() < 0.05
        # the leading dimension is honoured: the tail of o1 must be untouched
        assert torch.equal(o1[:, 1024:], torch.zeros_like(o1[:, 1024:]))

    @pytest.mark.parametrize("M,ff,K", [(10, 2048, 1024), (10, 512, 32), (1, 128, 64)])
    def test_glu_epilogue_matches_the_separate_roundings(self, M, ff, K):
        m = _dm05()
        N = 2 * ff
        w8, sc, packed = _w8_case(m, N, K, seed=8)
        torch.manual_seed(9)
        x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
        out = torch.zeros(M, ff, device="cuda", dtype=torch.bfloat16)
        m.dm05_w8a16_tc_glu(x.data_ptr(), packed.data_ptr(), sc.data_ptr(), out.data_ptr(),
                            ff, M, K, 8, _stream())
        torch.cuda.synchronize()
        # gate in block rows 0..7, up in 8..15, i.e. a permutation of the N blocks
        full = _w8_ref(x, w8, sc)
        gate = torch.empty(M, ff, device="cuda", dtype=torch.bfloat16)
        up = torch.empty_like(gate)
        for blk in range(ff // 8):
            gate[:, blk * 8:(blk + 1) * 8] = full[:, blk * 16:blk * 16 + 8]
            up[:, blk * 8:(blk + 1) * 8] = full[:, blk * 16 + 8:blk * 16 + 16]
        # the activation rounds to bf16 before the multiply, as the separate
        # gate GEMM, up GEMM and multiply in the unfused form do
        act = torch.nn.functional.gelu(gate.float(), approximate="tanh").to(torch.bfloat16)
        want = (act.float() * up.float()).to(torch.bfloat16)
        max_abs = (out.float() - want.float()).abs().max().item()
        assert max_abs < 0.05, f"max abs error {max_abs}"

    def test_shape_guards(self):
        m = _dm05()
        w = torch.zeros(16, 32, device="cuda", dtype=torch.int8)
        out = torch.zeros(16 * 32, device="cuda", dtype=torch.int8)
        with pytest.raises(RuntimeError, match="multiple of 16"):
            m.dm05_w8a16_pack(w.data_ptr(), out.data_ptr(), 15, 32, _stream())
        with pytest.raises(RuntimeError, match="multiple of 32"):
            m.dm05_w8a16_pack(w.data_ptr(), out.data_ptr(), 16, 31, _stream())
        torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
