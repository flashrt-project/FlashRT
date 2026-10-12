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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
