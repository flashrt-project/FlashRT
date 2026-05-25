import pytest


torch = pytest.importorskip("torch")


def _load_fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for qwen36 WY kernel tests")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"flash_rt_kernels is not built: {exc}")
    required = (
        "qwen36_gdn_wy_norm_cumsum_bf16",
        "qwen36_gdn_wy_kkt_b64_bf16",
        "qwen36_gdn_wy_solve_tril_b64_f32",
        "qwen36_gdn_wy_recompute_wu_b64_bf16",
        "qwen36_gdn_wy_chunk_h_b64_bf16",
        "qwen36_gdn_wy_output_o_b64_bf16",
        "linear_attn_gdn_wy_kkt_b64_bf16_cublaslt",
        "linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt",
        "linear_attn_gdn_wy_solve_tril_b64_f32_parallel",
        "linear_attn_gdn_wy_output_o_b64_bf16_cublaslt",
        "linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt",
        "linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_f32state",
        "linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm",
    )
    missing = [name for name in required if not hasattr(fvk, name)]
    if missing:
        pytest.skip(f"flash_rt_kernels missing WY symbols: {missing}")
    return fvk


def _ptr(x):
    return x.data_ptr()


def _local_cumsum_bf16(g, chunk=64):
    out = torch.empty_like(g)
    for start in range(0, g.shape[0], chunk):
        end = min(start + chunk, g.shape[0])
        out[start:end] = torch.cumsum(g[start:end].float(), dim=0).to(g.dtype)
    return out


def _kkt_ref(k_l2, beta, g_cumsum, chunk=64):
    S = k_l2.shape[0]
    chunks = (S + chunk - 1) // chunk
    A = torch.zeros(chunks, 48, chunk, chunk, device=k_l2.device,
                    dtype=torch.float32)
    for ci, start in enumerate(range(0, S, chunk)):
        end = min(start + chunk, S)
        T = end - start
        kk = k_l2[start:end].float()
        bb = beta[start:end].float()
        gg = g_cumsum[start:end].float()
        for vh in range(48):
            kh = vh // 3
            dots = kk[:, kh] @ kk[:, kh].T
            decay = torch.exp(gg[:, vh, None] - gg[None, :, vh])
            block = bb[:, vh, None] * dots * decay
            block = torch.tril(block, diagonal=-1)
            A[ci, vh, :T, :T] = block
    return A


def _solve_ref(A, S, chunk=64):
    chunks = (S + chunk - 1) // chunk
    Ai = torch.zeros_like(A)
    eye = torch.eye(chunk, device=A.device, dtype=torch.float32)
    for ci in range(chunks):
        T = min(chunk, S - ci * chunk)
        for vh in range(48):
            tri = eye[:T, :T] + torch.tril(A[ci, vh, :T, :T], diagonal=-1)
            inv = torch.linalg.solve_triangular(
                tri, eye[:T, :T], upper=False)
            Ai[ci, vh, :T, :T] = inv
    return Ai


def _recompute_wu_ref(k_l2, v, beta, g_cumsum, Ai, chunk=64):
    S = k_l2.shape[0]
    w = torch.empty(S, 48, 128, device=k_l2.device, dtype=torch.bfloat16)
    u = torch.empty_like(w)
    for ci, start in enumerate(range(0, S, chunk)):
        end = min(start + chunk, S)
        kk = k_l2[start:end].float()
        vv = v[start:end].float()
        bb = beta[start:end].float()
        gg = g_cumsum[start:end].float()
        for vh in range(48):
            kh = vh // 3
            Aih = Ai[ci, vh, :end - start, :end - start]
            u[start:end, vh] = (Aih @ (vv[:, vh] * bb[:, vh, None])
                                ).to(torch.bfloat16)
            w[start:end, vh] = (
                Aih @ (kk[:, kh] * bb[:, vh, None]
                       * torch.exp(gg[:, vh, None]))
            ).to(torch.bfloat16)
    return w, u


def _unpack_wy_pack(x_pack, S, chunk=64):
    x = torch.empty(S, 48, 128, device=x_pack.device, dtype=x_pack.dtype)
    for ci, start in enumerate(range(0, S, chunk)):
        end = min(start + chunk, S)
        x[start:end] = x_pack[ci, :, :end - start].transpose(0, 1)
    return x


def _chunk_h_ref(k_l2, u, w, g_cumsum, state, chunk=64):
    S = k_l2.shape[0]
    chunks = (S + chunk - 1) // chunk
    state_f = state.float().clone()
    h0 = torch.empty(chunks, 48, 128, 128, device=k_l2.device,
                     dtype=torch.bfloat16)
    v_new = torch.empty(S, 48, 128, device=k_l2.device,
                        dtype=torch.bfloat16)
    for ci, start in enumerate(range(0, S, chunk)):
        end = min(start + chunk, S)
        T = end - start
        h0[ci] = state_f.to(torch.bfloat16)
        for vh in range(48):
            kh = vh // 3
            v_new_f = u[start:end, vh].float() - (
                w[start:end, vh].float() @ state_f[vh])
            v_new[start:end, vh] = v_new_f.to(torch.bfloat16)
            g_last = g_cumsum[end - 1, vh].float()
            state_f[vh] *= torch.exp(g_last)
            decayed = (
                v_new_f
                * torch.exp(g_last - g_cumsum[start:end, vh].float())[:, None]
            )
            state_f[vh] += k_l2[start:end, kh].float().T @ decayed
    return h0, v_new, state_f.to(torch.bfloat16)


def _output_o_ref(q_l2, k_l2, v_new, h0, g_cumsum, chunk=64):
    S = q_l2.shape[0]
    out = torch.empty(S, 48, 128, device=q_l2.device, dtype=torch.bfloat16)
    scale = 128 ** -0.5
    for ci, start in enumerate(range(0, S, chunk)):
        end = min(start + chunk, S)
        for s in range(start, end):
            i = s - start
            for vh in range(48):
                kh = vh // 3
                q = q_l2[s, kh].float()
                gi = g_cumsum[s, vh].float()
                qh = (q @ h0[ci, vh].float()) * torch.exp(gi)
                kk = k_l2[start:s + 1, kh].float()
                dots = kk @ q
                decay = torch.exp(gi - g_cumsum[start:s + 1, vh].float())
                local = (
                    (dots * decay)[:, None]
                    * v_new[start:s + 1, vh].float()
                ).sum(dim=0)
                out[s, vh] = ((qh + local) * scale).to(torch.bfloat16)
    return out


@pytest.mark.parametrize("S", [6, 64, 65])
def test_qwen36_wy_norm_cumsum_and_kkt_match_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(1234 + S)
    q = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    # Keep gates moderately small to avoid meaningless exp overflow in
    # a unit test; real model gates are already numerically bounded.
    g = (torch.randn(S, 48, device="cuda") * 0.05).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(S, 48, device="cuda")).to(torch.bfloat16)

    q_l2 = torch.empty_like(q)
    k_l2 = torch.empty_like(k)
    g_cumsum = torch.empty_like(g)
    fvk.qwen36_gdn_wy_norm_cumsum_bf16(
        _ptr(q), _ptr(k), _ptr(g),
        _ptr(q_l2), _ptr(k_l2), _ptr(g_cumsum),
        S, 0)
    torch.cuda.synchronize()

    q_ref = (q.float() / torch.sqrt(
        torch.sum(q.float() * q.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    k_ref = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_ref = _local_cumsum_bf16(g)

    torch.testing.assert_close(q_l2, q_ref, rtol=0, atol=1e-3)
    torch.testing.assert_close(k_l2, k_ref, rtol=0, atol=1e-3)
    torch.testing.assert_close(g_cumsum, g_ref, rtol=0, atol=0)

    chunks = (S + 63) // 64
    A = torch.empty(chunks, 48, 64, 64, device="cuda", dtype=torch.float32)
    fvk.qwen36_gdn_wy_kkt_b64_bf16(
        _ptr(k_l2), _ptr(beta), _ptr(g_cumsum), _ptr(A), S, 0)
    torch.cuda.synchronize()

    A_ref = _kkt_ref(k_ref, beta, g_ref)
    torch.testing.assert_close(A, A_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("S", [6, 64, 65])
def test_qwen36_wy_solve_tril_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(5678 + S)
    chunks = (S + 63) // 64
    A = torch.zeros(chunks, 48, 64, 64, device="cuda", dtype=torch.float32)
    for ci in range(chunks):
        T = min(64, S - ci * 64)
        block = torch.randn(48, T, T, device="cuda") * 0.01
        A[ci, :, :T, :T] = torch.tril(block, diagonal=-1)

    Ai = torch.empty_like(A)
    fvk.qwen36_gdn_wy_solve_tril_b64_f32(_ptr(A), _ptr(Ai), S, 0)
    torch.cuda.synchronize()

    Ai_ref = _solve_ref(A, S)
    torch.testing.assert_close(Ai, Ai_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("S", [6, 64, 65, 128])
def test_linear_attn_gdn_wy_solve_tril_parallel_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(6789 + S)
    chunks = (S + 63) // 64
    A = torch.zeros(chunks, 48, 64, 64, device="cuda", dtype=torch.float32)
    for ci in range(chunks):
        T = min(64, S - ci * 64)
        block = torch.randn(48, T, T, device="cuda") * 0.01
        A[ci, :, :T, :T] = torch.tril(block, diagonal=-1)

    Ai = torch.empty_like(A)
    fvk.linear_attn_gdn_wy_solve_tril_b64_f32_parallel(
        _ptr(A), _ptr(Ai), S, 48, 0)
    torch.cuda.synchronize()

    Ai_ref = _solve_ref(A, S)
    torch.testing.assert_close(Ai, Ai_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("S", [6, 64, 65])
def test_qwen36_wy_recompute_wu_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(9012 + S)
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    g = (torch.randn(S, 48, device="cuda") * 0.05).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(S, 48, device="cuda")).to(torch.bfloat16)

    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)
    A = _kkt_ref(k_l2, beta, g_cumsum)
    Ai = _solve_ref(A, S)
    w = torch.empty(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    u = torch.empty_like(w)

    fvk.qwen36_gdn_wy_recompute_wu_b64_bf16(
        _ptr(k_l2), _ptr(v), _ptr(beta), _ptr(g_cumsum), _ptr(Ai),
        _ptr(w), _ptr(u), S, 0)
    torch.cuda.synchronize()

    w_ref, u_ref = _recompute_wu_ref(k_l2, v, beta, g_cumsum, Ai)
    torch.testing.assert_close(w, w_ref, rtol=0, atol=1e-2)
    torch.testing.assert_close(u, u_ref, rtol=0, atol=1e-2)


@pytest.mark.parametrize("S", [6, 64, 65, 128])
def test_linear_attn_gdn_wy_kkt_cublaslt_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(2345 + S)
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    g = (torch.randn(S, 48, device="cuda") * 0.05).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(S, 48, device="cuda")).to(torch.bfloat16)
    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)
    chunks = (S + 63) // 64
    k_pack = torch.empty(chunks, 16, 64, 128, device="cuda",
                         dtype=torch.bfloat16)
    kkt_base = torch.empty(chunks, 16, 64, 64, device="cuda",
                           dtype=torch.float32)
    A = torch.empty(chunks, 48, 64, 64, device="cuda", dtype=torch.float32)
    fvk.linear_attn_gdn_wy_kkt_b64_bf16_cublaslt(
        _ptr(k_l2), _ptr(beta), _ptr(g_cumsum),
        _ptr(k_pack), _ptr(kkt_base), _ptr(A),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()
    A_ref = _kkt_ref(k_l2, beta, g_cumsum)
    torch.testing.assert_close(A, A_ref, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("S", [6, 64, 65, 128])
def test_linear_attn_gdn_wy_recompute_wu_cublaslt_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(4567 + S)
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    g = (torch.randn(S, 48, device="cuda") * 0.05).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(S, 48, device="cuda")).to(torch.bfloat16)
    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)
    A = _kkt_ref(k_l2, beta, g_cumsum)
    Ai = _solve_ref(A, S)
    chunks = (S + 63) // 64
    ai_pack = torch.empty(chunks, 48, 64, 64, device="cuda",
                          dtype=torch.bfloat16)
    rhs_w = torch.empty(chunks, 48, 64, 128, device="cuda",
                        dtype=torch.bfloat16)
    rhs_u = torch.empty_like(rhs_w)
    w_pack = torch.empty_like(rhs_w)
    u_pack = torch.empty_like(rhs_w)
    w = torch.empty(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    u = torch.empty_like(w)

    fvk.linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt(
        _ptr(k_l2), _ptr(v), _ptr(beta), _ptr(g_cumsum), _ptr(Ai),
        _ptr(ai_pack), _ptr(rhs_w), _ptr(rhs_u),
        _ptr(w_pack), _ptr(u_pack),
        _ptr(w), _ptr(u),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    w_ref, u_ref = _recompute_wu_ref(k_l2, v, beta, g_cumsum, Ai)
    torch.testing.assert_close(w, w_ref, rtol=0, atol=2e-2)
    torch.testing.assert_close(u, u_ref, rtol=0, atol=2e-2)


@pytest.mark.parametrize("S", [6, 64, 65, 128])
def test_linear_attn_gdn_wy_recompute_wu_cublaslt_packed_matches_reference(S):
    fvk = _load_fvk()
    if not hasattr(
            fvk,
            "linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt_packed"):
        pytest.skip("packed WY recompute kernel is not built")
    torch.manual_seed(5567 + S)
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    g = (torch.randn(S, 48, device="cuda") * 0.05).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(S, 48, device="cuda")).to(torch.bfloat16)
    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)
    A = _kkt_ref(k_l2, beta, g_cumsum)
    Ai = _solve_ref(A, S)
    chunks = (S + 63) // 64
    ai_pack = torch.empty(chunks, 48, 64, 64, device="cuda",
                          dtype=torch.bfloat16)
    rhs_w = torch.empty(chunks, 48, 64, 128, device="cuda",
                        dtype=torch.bfloat16)
    rhs_u = torch.empty_like(rhs_w)
    w_pack = torch.empty_like(rhs_w)
    u_pack = torch.empty_like(rhs_w)

    fvk.linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt_packed(
        _ptr(k_l2), _ptr(v), _ptr(beta), _ptr(g_cumsum), _ptr(Ai),
        _ptr(ai_pack), _ptr(rhs_w), _ptr(rhs_u),
        _ptr(w_pack), _ptr(u_pack),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    w_ref, u_ref = _recompute_wu_ref(k_l2, v, beta, g_cumsum, Ai)
    torch.testing.assert_close(
        _unpack_wy_pack(w_pack, S), w_ref, rtol=0, atol=2e-2)
    torch.testing.assert_close(
        _unpack_wy_pack(u_pack, S), u_ref, rtol=0, atol=2e-2)


@pytest.mark.parametrize("S", [6, 64, 65])
def test_qwen36_wy_chunk_h_and_output_match_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(3456 + S)
    q = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    g = (torch.randn(S, 48, device="cuda") * 0.02).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(S, 48, device="cuda")).to(torch.bfloat16)
    state0 = (torch.randn(48, 128, 128, device="cuda") * 0.02
              ).to(torch.bfloat16)

    q_l2 = (q.float() / torch.sqrt(
        torch.sum(q.float() * q.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)
    A = _kkt_ref(k_l2, beta, g_cumsum)
    Ai = _solve_ref(A, S)
    w, u = _recompute_wu_ref(k_l2, v, beta, g_cumsum, Ai)

    state = state0.clone()
    chunks = (S + 63) // 64
    h0 = torch.empty(chunks, 48, 128, 128, device="cuda",
                     dtype=torch.bfloat16)
    v_new = torch.empty(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(v_new)
    fvk.qwen36_gdn_wy_chunk_h_b64_bf16(
        _ptr(k_l2), _ptr(u), _ptr(w), _ptr(g_cumsum), _ptr(state),
        _ptr(h0), _ptr(v_new), S, 0)
    fvk.qwen36_gdn_wy_output_o_b64_bf16(
        _ptr(q_l2), _ptr(k_l2), _ptr(v_new), _ptr(h0), _ptr(g_cumsum),
        _ptr(out), S, 0)
    torch.cuda.synchronize()

    h0_ref, v_new_ref, state_ref = _chunk_h_ref(
        k_l2, u, w, g_cumsum, state0)
    out_ref = _output_o_ref(q_l2, k_l2, v_new_ref, h0_ref, g_cumsum)
    torch.testing.assert_close(h0, h0_ref, rtol=0, atol=2e-2)
    torch.testing.assert_close(v_new, v_new_ref, rtol=0, atol=2e-2)
    torch.testing.assert_close(state, state_ref, rtol=0, atol=2e-2)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=2e-2)


@pytest.mark.parametrize("S", [6, 64, 65])
def test_linear_attn_gdn_wy_chunk_h_cublaslt_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(3654 + S)
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    u = (torch.randn(S, 48, 128, device="cuda") * 0.05).to(torch.bfloat16)
    w = (torch.randn(S, 48, 128, device="cuda") * 0.05).to(torch.bfloat16)
    g = (torch.randn(S, 48, device="cuda") * 0.02).to(torch.bfloat16)
    state0 = (torch.randn(48, 128, 128, device="cuda") * 0.02
              ).to(torch.bfloat16)

    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)
    state = state0.clone()
    chunks = (S + 63) // 64
    h0 = torch.empty(chunks, 48, 128, 128, device="cuda",
                     dtype=torch.bfloat16)
    v_new = torch.empty(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    k_pack_hv = torch.empty(chunks, 48, 64, 128, device="cuda",
                            dtype=torch.bfloat16)
    w_pack = torch.empty_like(k_pack_hv)
    u_pack = torch.empty_like(k_pack_hv)
    wh_pack = torch.empty_like(k_pack_hv)
    decayed_v_pack = torch.empty_like(k_pack_hv)

    fvk.linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt(
        _ptr(k_l2), _ptr(u), _ptr(w), _ptr(g_cumsum), _ptr(state),
        _ptr(h0), _ptr(v_new),
        _ptr(k_pack_hv), _ptr(w_pack), _ptr(u_pack),
        _ptr(wh_pack), _ptr(decayed_v_pack),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    h0_ref, v_new_ref, state_ref = _chunk_h_ref(
        k_l2, u, w, g_cumsum, state0)
    torch.testing.assert_close(h0, h0_ref, rtol=0, atol=5e-2)
    torch.testing.assert_close(v_new, v_new_ref, rtol=0, atol=5e-2)
    torch.testing.assert_close(state, state_ref, rtol=0, atol=5e-2)


@pytest.mark.parametrize("S", [6, 64, 65])
def test_linear_attn_gdn_wy_chunk_h_cublaslt_f32state_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(4654 + S)
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    g = (torch.randn(S, 48, device="cuda") * 0.02).to(torch.bfloat16)
    state0 = (torch.randn(48, 128, 128, device="cuda") * 0.02
              ).to(torch.bfloat16)

    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)
    state = state0.clone()
    chunks = (S + 63) // 64
    h0 = torch.empty(chunks, 48, 128, 128, device="cuda",
                     dtype=torch.bfloat16)
    v_new = torch.empty(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    k_pack_hv = torch.empty(chunks, 48, 64, 128, device="cuda",
                            dtype=torch.bfloat16)
    w_pack = torch.empty_like(k_pack_hv)
    u_pack = torch.empty_like(k_pack_hv)
    wh_pack = torch.empty_like(k_pack_hv)
    decayed_v_pack = torch.empty_like(k_pack_hv)
    state_f32 = torch.empty(48, 128, 128, device="cuda", dtype=torch.float32)
    delta_f32 = torch.empty_like(state_f32)

    fvk.linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_f32state(
        _ptr(k_l2), _ptr(u), _ptr(w), _ptr(g_cumsum), _ptr(state),
        _ptr(h0), _ptr(v_new),
        _ptr(k_pack_hv), _ptr(w_pack), _ptr(u_pack),
        _ptr(wh_pack), _ptr(decayed_v_pack),
        _ptr(state_f32), _ptr(delta_f32),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    h0_ref, v_new_ref, state_ref = _chunk_h_ref(
        k_l2, u, w, g_cumsum, state0)
    torch.testing.assert_close(h0, h0_ref, rtol=0, atol=5e-2)
    torch.testing.assert_close(v_new, v_new_ref, rtol=0, atol=6e-2)
    torch.testing.assert_close(state, state_ref, rtol=0, atol=5e-2)


@pytest.mark.parametrize("S", [6, 64, 65])
def test_linear_attn_gdn_wy_chunk_h_cublaslt_f32gemm_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(5654 + S)
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    g = (torch.randn(S, 48, device="cuda") * 0.02).to(torch.bfloat16)
    state0 = (torch.randn(48, 128, 128, device="cuda") * 0.02
              ).to(torch.bfloat16)

    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)
    state = state0.clone()
    chunks = (S + 63) // 64
    h0 = torch.empty(chunks, 48, 128, 128, device="cuda",
                     dtype=torch.bfloat16)
    v_new = torch.empty(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    k_pack_hv = torch.empty(chunks, 48, 64, 128, device="cuda",
                            dtype=torch.bfloat16)
    w_pack = torch.empty_like(k_pack_hv)
    u_pack = torch.empty_like(k_pack_hv)
    wh_pack = torch.empty_like(k_pack_hv)
    decayed_v_pack = torch.empty_like(k_pack_hv)
    state_f32 = torch.empty(48, 128, 128, device="cuda", dtype=torch.float32)
    chunk_f32 = torch.empty(48, 64, 128, device="cuda", dtype=torch.float32)
    acc_f32 = torch.empty_like(chunk_f32)

    fvk.linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm(
        _ptr(k_l2), _ptr(u), _ptr(w), _ptr(g_cumsum), _ptr(state),
        _ptr(h0), _ptr(v_new),
        _ptr(k_pack_hv), _ptr(w_pack), _ptr(u_pack),
        _ptr(wh_pack), _ptr(decayed_v_pack),
        _ptr(state_f32), _ptr(chunk_f32), _ptr(acc_f32),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    h0_ref, v_new_ref, state_ref = _chunk_h_ref(
        k_l2, u, w, g_cumsum, state0)
    torch.testing.assert_close(h0, h0_ref, rtol=0, atol=5e-2)
    torch.testing.assert_close(v_new, v_new_ref, rtol=0, atol=6e-2)
    torch.testing.assert_close(state, state_ref, rtol=0, atol=6e-2)


@pytest.mark.parametrize("S", [6, 64, 65])
def test_linear_attn_gdn_wy_chunk_h_cublaslt_f32gemm_packed_wu_matches_reference(S):
    fvk = _load_fvk()
    if not hasattr(
            fvk,
            "linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm_packed_wu"):
        pytest.skip("packed WY chunk_h kernel is not built")
    torch.manual_seed(6654 + S)
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    g = (torch.randn(S, 48, device="cuda") * 0.02).to(torch.bfloat16)
    state0 = (torch.randn(48, 128, 128, device="cuda") * 0.02
              ).to(torch.bfloat16)

    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)
    state = state0.clone()
    chunks = (S + 63) // 64
    h0 = torch.empty(chunks, 48, 128, 128, device="cuda",
                     dtype=torch.bfloat16)
    v_new = torch.empty(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    k_pack_hv = torch.empty(chunks, 48, 64, 128, device="cuda",
                            dtype=torch.bfloat16)
    w_pack = torch.empty_like(k_pack_hv)
    u_pack = torch.empty_like(k_pack_hv)
    for ci, start in enumerate(range(0, S, 64)):
        end = min(start + 64, S)
        w_pack[ci, :, :end - start] = w[start:end].transpose(0, 1)
        u_pack[ci, :, :end - start] = u[start:end].transpose(0, 1)
    decayed_v_pack = torch.empty_like(k_pack_hv)
    state_f32 = torch.empty(48, 128, 128, device="cuda", dtype=torch.float32)
    chunk_f32 = torch.empty(48, 64, 128, device="cuda", dtype=torch.float32)
    acc_f32 = torch.empty_like(chunk_f32)

    fvk.linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm_packed_wu(
        _ptr(k_l2), _ptr(w_pack), _ptr(u_pack), _ptr(g_cumsum), _ptr(state),
        _ptr(h0), _ptr(v_new), _ptr(k_pack_hv), _ptr(decayed_v_pack),
        _ptr(state_f32), _ptr(chunk_f32), _ptr(acc_f32),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    h0_ref, v_new_ref, state_ref = _chunk_h_ref(
        k_l2, u, w, g_cumsum, state0)
    torch.testing.assert_close(h0, h0_ref, rtol=0, atol=5e-2)
    torch.testing.assert_close(v_new, v_new_ref, rtol=0, atol=6e-2)
    torch.testing.assert_close(state, state_ref, rtol=0, atol=6e-2)


@pytest.mark.parametrize("S", [6, 64, 65])
def test_linear_attn_gdn_wy_output_o_cublaslt_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(4567 + S)
    q = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    v_new = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    h0 = (torch.randn((S + 63) // 64, 48, 128, 128, device="cuda")
          * 0.02).to(torch.bfloat16)
    g = (torch.randn(S, 48, device="cuda") * 0.02).to(torch.bfloat16)

    q_l2 = (q.float() / torch.sqrt(
        torch.sum(q.float() * q.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)
    chunks = (S + 63) // 64
    q_pack = torch.empty(chunks, 48, 64, 128, device="cuda",
                         dtype=torch.bfloat16)
    k_pack_hv = torch.empty_like(q_pack)
    v_pack = torch.empty_like(q_pack)
    qk_base = torch.empty(chunks, 48, 64, 64, device="cuda",
                          dtype=torch.float32)
    local_a_pack = torch.empty(chunks, 48, 64, 64, device="cuda",
                               dtype=torch.bfloat16)
    qh_pack = torch.empty_like(q_pack)
    local_pack = torch.empty_like(q_pack)
    out = torch.empty(S, 48, 128, device="cuda", dtype=torch.bfloat16)

    fvk.linear_attn_gdn_wy_output_o_b64_bf16_cublaslt(
        _ptr(q_l2), _ptr(k_l2), _ptr(v_new), _ptr(h0), _ptr(g_cumsum),
        _ptr(q_pack), _ptr(k_pack_hv), _ptr(v_pack),
        _ptr(qk_base), _ptr(local_a_pack), _ptr(qh_pack), _ptr(local_pack),
        _ptr(out), S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    out_ref = _output_o_ref(q_l2, k_l2, v_new, h0, g_cumsum)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=3e-2)


@pytest.mark.parametrize("S", [6, 64, 65])
def test_linear_attn_gdn_wy_output_o_cublaslt_packed_k_matches_reference(S):
    fvk = _load_fvk()
    if not hasattr(fvk, "linear_attn_gdn_wy_output_o_b64_bf16_cublaslt_packed_k"):
        pytest.skip("packed-k WY output kernel is not built")
    torch.manual_seed(7567 + S)
    q = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    v_new = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16)
    h0 = (torch.randn((S + 63) // 64, 48, 128, 128, device="cuda")
          * 0.02).to(torch.bfloat16)
    g = (torch.randn(S, 48, device="cuda") * 0.02).to(torch.bfloat16)

    q_l2 = (q.float() / torch.sqrt(
        torch.sum(q.float() * q.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)
    chunks = (S + 63) // 64
    k_pack_hv = torch.empty(chunks, 48, 64, 128, device="cuda",
                            dtype=torch.bfloat16)
    for ci, start in enumerate(range(0, S, 64)):
        end = min(start + 64, S)
        for vh in range(48):
            k_pack_hv[ci, vh, :end - start] = k_l2[start:end, vh // 3]
    q_pack = torch.empty_like(k_pack_hv)
    v_pack = torch.empty_like(k_pack_hv)
    qk_base = torch.empty(chunks, 48, 64, 64, device="cuda",
                          dtype=torch.float32)
    local_a_pack = torch.empty(chunks, 48, 64, 64, device="cuda",
                               dtype=torch.bfloat16)
    qh_pack = torch.empty_like(k_pack_hv)
    local_pack = torch.empty_like(k_pack_hv)
    out = torch.empty(S, 48, 128, device="cuda", dtype=torch.bfloat16)

    fvk.linear_attn_gdn_wy_output_o_b64_bf16_cublaslt_packed_k(
        _ptr(q_l2), _ptr(k_pack_hv), _ptr(v_new), _ptr(h0),
        _ptr(g_cumsum), _ptr(q_pack), _ptr(v_pack), _ptr(qk_base),
        _ptr(local_a_pack), _ptr(qh_pack), _ptr(local_pack), _ptr(out),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    out_ref = _output_o_ref(q_l2, k_l2, v_new, h0, g_cumsum)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=3e-2)
