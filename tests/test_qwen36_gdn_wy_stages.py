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
        "qwen36_gdn_wy_norm_cumsum_pack_q_bf16",
        "qwen36_gdn_wy_kkt_b64_bf16",
        "qwen36_gdn_wy_solve_tril_b64_f32",
        "qwen36_gdn_wy_recompute_wu_b64_bf16",
        "qwen36_gdn_wy_chunk_h_b64_bf16",
        "qwen36_gdn_wy_output_o_b64_bf16",
        "linear_attn_gdn_wy_kkt_b64_bf16_cublaslt",
        "linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt",
        "linear_attn_gdn_wy_solve_tril_b64_f32_parallel",
        "linear_attn_gdn_wy_solve_tril_b64_f32_parallel_pack",
        "linear_attn_gdn_wy_output_o_b64_bf16_cublaslt",
        "linear_attn_gdn_wy_output_o_b64_bf16_cublaslt_packed_kv",
        "linear_attn_gdn_wy_output_o_b64_bf16_cublaslt_packed_qkv",
        "linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt",
        "linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_packed_wu",
        "linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_f32state",
        "linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm",
        "linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs",
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


def _kkt_nogate_ref(k_l2, beta, chunk=64):
    S = k_l2.shape[0]
    chunks = (S + chunk - 1) // chunk
    A = torch.zeros(chunks, 48, chunk, chunk, device=k_l2.device,
                    dtype=torch.float32)
    for ci, start in enumerate(range(0, S, chunk)):
        end = min(start + chunk, S)
        T = end - start
        kk = k_l2[start:end].float()
        bb = beta[start:end].float()
        for vh in range(48):
            kh = vh // 3
            dots = kk[:, kh] @ kk[:, kh].T
            block = bb[:, vh, None] * dots
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


def _flashqla_fused_gdr_ref(q_l2, k_l2, v, beta, g_cumsum, Ai_nogate, state,
                            chunk=64):
    S = k_l2.shape[0]
    chunks = (S + chunk - 1) // chunk
    state_f = state.float().clone()
    out = torch.empty(S, 48, 128, device=k_l2.device, dtype=torch.bfloat16)
    scale = 128 ** -0.5
    for ci in range(chunks):
        start = ci * chunk
        end = min(start + chunk, S)
        T = end - start
        for vh in range(48):
            kh = vh // 3
            qh = q_l2[start:end, kh].float()
            kh_l2 = k_l2[start:end, kh].float()
            vv = v[start:end, vh].float()
            gg = g_cumsum[start:end, vh].float()
            bb = beta[start:end, vh].float()
            state_prev = state_f[vh].clone()

            # FlashQLA-style form:
            # W = V - exp(g_t) * (K @ S)
            # Vd = (exp(g_i-g_j) * Ai_no_gate[i,j] * beta[j]) @ W
            w0 = vv - torch.exp(gg)[:, None] * (kh_l2 @ state_prev)
            decay = torch.exp(gg[:, None] - gg[None, :])
            ag = Ai_nogate[ci, vh, :T, :T] * decay * bb[None, :]
            vd = ag @ w0

            q_state = torch.exp(gg)[:, None] * (qh @ state_prev)
            p = qh @ kh_l2.T
            causal_decay = torch.tril(decay)
            local = (p * causal_decay) @ vd
            out[start:end, vh] = ((q_state + local) * scale).to(torch.bfloat16)

            g_last = gg[-1]
            state_f[vh] *= torch.exp(g_last)
            v_prime = vd * torch.exp(g_last - gg)[:, None]
            state_f[vh] += kh_l2.T @ v_prime
    return out, state_f.to(torch.bfloat16)


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
def test_qwen36_wy_norm_cumsum_pack_q_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(2234 + S)
    q = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    g = (torch.randn(S, 48, device="cuda") * 0.05).to(torch.bfloat16)

    chunks = (S + 63) // 64
    q_l2 = torch.empty_like(q)
    k_l2 = torch.empty_like(k)
    q_pack = torch.empty(chunks, 48, 64, 128, device="cuda",
                         dtype=torch.bfloat16)
    g_cumsum = torch.empty_like(g)
    fvk.qwen36_gdn_wy_norm_cumsum_pack_q_bf16(
        _ptr(q), _ptr(k), _ptr(g), _ptr(q_l2), _ptr(k_l2),
        _ptr(q_pack), _ptr(g_cumsum), S, 0)
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
    for ci, start in enumerate(range(0, S, 64)):
        end = min(start + 64, S)
        for vh in range(48):
            torch.testing.assert_close(
                q_pack[ci, vh, :end - start],
                q_ref[start:end, vh // 3],
                rtol=0, atol=1e-3)


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


@pytest.mark.parametrize("S", [6, 64, 65, 128])
def test_linear_attn_gdn_wy_solve_tril_parallel_pack_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(7789 + S)
    chunks = (S + 63) // 64
    A = torch.zeros(chunks, 48, 64, 64, device="cuda", dtype=torch.float32)
    for ci in range(chunks):
        T = min(64, S - ci * 64)
        block = torch.randn(48, T, T, device="cuda") * 0.01
        A[ci, :, :T, :T] = torch.tril(block, diagonal=-1)

    Ai = torch.empty_like(A)
    Ai_pack = torch.empty(chunks, 48, 64, 64, device="cuda",
                          dtype=torch.bfloat16)
    fvk.linear_attn_gdn_wy_solve_tril_b64_f32_parallel_pack(
        _ptr(A), _ptr(Ai), _ptr(Ai_pack), S, 48, 0)
    torch.cuda.synchronize()

    Ai_ref = _solve_ref(A, S)
    torch.testing.assert_close(Ai, Ai_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        Ai_pack, Ai.to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("S", [6, 64, 65, 128])
def test_linear_attn_gdn_wy_solve_tril_fused_pack_matches_reference(S):
    fvk = _load_fvk()
    if not hasattr(fvk, "linear_attn_gdn_wy_solve_tril_b64_f32_fused_pack"):
        pytest.skip("fused solve_tril pack kernel is not built")
    torch.manual_seed(8789 + S)
    chunks = (S + 63) // 64
    A = torch.zeros(chunks, 48, 64, 64, device="cuda", dtype=torch.float32)
    for ci in range(chunks):
        T = min(64, S - ci * 64)
        block = torch.randn(48, T, T, device="cuda") * 0.01
        A[ci, :, :T, :T] = torch.tril(block, diagonal=-1)

    Ai = torch.empty_like(A)
    Ai_pack = torch.empty(chunks, 48, 64, 64, device="cuda",
                          dtype=torch.bfloat16)
    fvk.linear_attn_gdn_wy_solve_tril_b64_f32_fused_pack(
        _ptr(A), _ptr(Ai), _ptr(Ai_pack), S, 48, 0)
    torch.cuda.synchronize()

    Ai_ref = _solve_ref(A, S)
    torch.testing.assert_close(Ai, Ai_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        Ai_pack, Ai.to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("S", [6, 64, 65, 128])
def test_linear_attn_gdn_wy_solve_tril_fused_pack_only_matches_pack(S):
    fvk = _load_fvk()
    if not hasattr(fvk, "linear_attn_gdn_wy_solve_tril_b64_f32_fused_pack_only"):
        pytest.skip("fused solve_tril pack-only kernel is not built")
    torch.manual_seed(8797 + S)
    chunks = (S + 63) // 64
    A = torch.zeros(chunks, 48, 64, 64, device="cuda", dtype=torch.float32)
    for ci in range(chunks):
        T = min(64, S - ci * 64)
        block = torch.randn(48, T, T, device="cuda") * 0.01
        A[ci, :, :T, :T] = torch.tril(block, diagonal=-1)

    Ai = torch.empty_like(A)
    Ai_pack = torch.empty(chunks, 48, 64, 64, device="cuda",
                          dtype=torch.bfloat16)
    Ai_pack_only = torch.empty_like(Ai_pack)
    fvk.linear_attn_gdn_wy_solve_tril_b64_f32_fused_pack(
        _ptr(A), _ptr(Ai), _ptr(Ai_pack), S, 48, 0)
    fvk.linear_attn_gdn_wy_solve_tril_b64_f32_fused_pack_only(
        _ptr(A), _ptr(Ai_pack_only), S, 48, 0)
    torch.cuda.synchronize()

    torch.testing.assert_close(Ai_pack_only, Ai_pack, rtol=0, atol=0)


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


@pytest.mark.parametrize("S", [64, 128])
def test_qwen36_wy_norm_pack_qk_and_kkt_packed_k_match_existing_path(S):
    fvk = _load_fvk()
    if not hasattr(fvk, "qwen36_gdn_wy_norm_cumsum_pack_qk_bf16"):
        pytest.skip("qk norm/pack kernel is not built")
    if not hasattr(fvk, "linear_attn_gdn_wy_kkt_b64_bf16_cublaslt_packed_k"):
        pytest.skip("packed-k kkt kernel is not built")
    torch.manual_seed(20260530 + S)
    chunks = (S + 63) // 64
    q = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    g = (torch.randn(S, 48, device="cuda") * 0.02).to(torch.bfloat16)
    beta = torch.rand(S, 48, device="cuda", dtype=torch.bfloat16)

    q_l2 = torch.empty_like(q)
    k_l2 = torch.empty_like(k)
    q_pack = torch.empty(chunks, 48, 64, 128, device="cuda",
                         dtype=torch.bfloat16)
    g_cumsum = torch.empty(S, 48, device="cuda", dtype=torch.bfloat16)
    fvk.qwen36_gdn_wy_norm_cumsum_pack_q_bf16(
        _ptr(q), _ptr(k), _ptr(g), _ptr(q_l2), _ptr(k_l2),
        _ptr(q_pack), _ptr(g_cumsum), S, 0)

    q_l2_fast = torch.empty_like(q)
    k_l2_fast = torch.empty_like(k)
    q_pack_fast = torch.empty_like(q_pack)
    k_pack = torch.empty(chunks, 16, 64, 128, device="cuda",
                         dtype=torch.bfloat16)
    g_cumsum_fast = torch.empty_like(g_cumsum)
    fvk.qwen36_gdn_wy_norm_cumsum_pack_qk_bf16(
        _ptr(q), _ptr(k), _ptr(g), _ptr(q_l2_fast), _ptr(k_l2_fast),
        _ptr(q_pack_fast), _ptr(k_pack), _ptr(g_cumsum_fast), S, 0)
    torch.cuda.synchronize()

    torch.testing.assert_close(q_l2_fast, q_l2, rtol=0, atol=0)
    torch.testing.assert_close(k_l2_fast, k_l2, rtol=0, atol=0)
    torch.testing.assert_close(q_pack_fast, q_pack, rtol=0, atol=0)
    torch.testing.assert_close(g_cumsum_fast, g_cumsum, rtol=0, atol=0)

    k_pack_base = torch.empty_like(k_pack)
    kkt_base = torch.empty(chunks, 16, 64, 64, device="cuda",
                           dtype=torch.float32)
    kkt_fast = torch.empty_like(kkt_base)
    A = torch.empty(chunks, 48, 64, 64, device="cuda", dtype=torch.float32)
    A_fast = torch.empty_like(A)
    fvk.linear_attn_gdn_wy_kkt_b64_bf16_cublaslt(
        _ptr(k_l2), _ptr(beta), _ptr(g_cumsum),
        _ptr(k_pack_base), _ptr(kkt_base), _ptr(A),
        S, 16, 48, 128, 3, 0)
    fvk.linear_attn_gdn_wy_kkt_b64_bf16_cublaslt_packed_k(
        _ptr(k_pack), _ptr(beta), _ptr(g_cumsum_fast),
        _ptr(kkt_fast), _ptr(A_fast), S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()
    torch.testing.assert_close(A_fast, A, rtol=0, atol=0)


@pytest.mark.parametrize("S", [64, 128])
def test_linear_attn_gdn_wy_kkt_nogate_matches_reference_and_gated_factorization(S):
    fvk = _load_fvk()
    if not hasattr(fvk, "linear_attn_gdn_wy_kkt_b64_bf16_cublaslt_nogate"):
        pytest.skip("nogate kkt cublasLt kernel is not built")
    torch.manual_seed(20260527 + S)
    chunks = (S + 63) // 64
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16) * 0.25
    beta = torch.rand(S, 48, device="cuda", dtype=torch.bfloat16) * 0.8
    g = (torch.randn(S, 48, device="cuda") * 0.02).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)
    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)

    k_pack = torch.empty(chunks, 16, 64, 128, device="cuda",
                         dtype=torch.bfloat16)
    kkt_base = torch.empty(chunks, 16, 64, 64, device="cuda",
                           dtype=torch.float32)
    A_nogate = torch.empty(chunks, 48, 64, 64, device="cuda",
                           dtype=torch.float32)
    fvk.linear_attn_gdn_wy_kkt_b64_bf16_cublaslt_nogate(
        _ptr(k_l2), _ptr(beta), _ptr(k_pack), _ptr(kkt_base), _ptr(A_nogate),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()
    A_nogate_ref = _kkt_nogate_ref(k_l2, beta)
    torch.testing.assert_close(A_nogate, A_nogate_ref, rtol=2e-3, atol=2e-3)

    A_gated = torch.empty_like(A_nogate)
    fvk.linear_attn_gdn_wy_kkt_b64_bf16_cublaslt(
        _ptr(k_l2), _ptr(beta), _ptr(g_cumsum),
        _ptr(k_pack), _ptr(kkt_base), _ptr(A_gated),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()
    Ai_gated = _solve_ref(A_gated, S)
    Ai_nogate = _solve_ref(A_nogate, S)

    for ci, start in enumerate(range(0, S, 64)):
        T = min(64, S - start)
        gg = g_cumsum[start:start + T].float()
        bb = beta[start:start + T].float()
        lhs = Ai_gated[ci, :, :T, :T] * bb.T[:, None, :]
        decay = torch.exp(gg.T[:, :, None] - gg.T[:, None, :])
        rhs = Ai_nogate[ci, :, :T, :T] * decay * bb.T[:, None, :]
        torch.testing.assert_close(lhs, rhs, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("S", [64, 128])
def test_flashqla_fused_gdr_math_matches_existing_wy_reference(S):
    torch.manual_seed(20260528 + S)
    q = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    v = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    beta = torch.rand(S, 48, device="cuda", dtype=torch.bfloat16) * 0.8
    g = (torch.randn(S, 48, device="cuda") * 0.015).to(torch.bfloat16)
    state0 = (torch.randn(48, 128, 128, device="cuda") * 0.01
              ).to(torch.bfloat16)
    q_l2 = (q.float() / torch.sqrt(
        torch.sum(q.float() * q.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)

    A_gated = _kkt_ref(k_l2, beta, g_cumsum)
    Ai_gated = _solve_ref(A_gated, S)
    w, u = _recompute_wu_ref(k_l2, v, beta, g_cumsum, Ai_gated)
    h0, v_new, state_existing = _chunk_h_ref(k_l2, u, w, g_cumsum, state0)
    out_existing = _output_o_ref(q_l2, k_l2, v_new, h0, g_cumsum)

    A_nogate = _kkt_nogate_ref(k_l2, beta)
    Ai_nogate = _solve_ref(A_nogate, S)
    out_fused, state_fused = _flashqla_fused_gdr_ref(
        q_l2, k_l2, v, beta, g_cumsum, Ai_nogate, state0)

    torch.testing.assert_close(state_fused, state_existing, rtol=0, atol=0.08)
    torch.testing.assert_close(out_fused, out_existing, rtol=0, atol=0.08)


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


@pytest.mark.parametrize("S", [6, 64, 65, 128])
def test_linear_attn_gdn_wy_recompute_wu_cublaslt_packed_rhs_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(6567 + S)
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
    ai_pack = Ai.to(torch.bfloat16)
    rhs_w = torch.empty(chunks, 48, 64, 128, device="cuda",
                        dtype=torch.bfloat16)
    rhs_u = torch.empty_like(rhs_w)
    w_pack = torch.empty_like(rhs_w)
    u_pack = torch.empty_like(rhs_w)
    rhs_w_base = torch.empty_like(rhs_w)
    rhs_u_base = torch.empty_like(rhs_w)
    w_pack_base = torch.empty_like(rhs_w)
    u_pack_base = torch.empty_like(rhs_w)

    fvk.linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs(
        _ptr(k_l2), _ptr(v), _ptr(beta), _ptr(g_cumsum), _ptr(ai_pack),
        _ptr(rhs_w), _ptr(rhs_u), _ptr(w_pack), _ptr(u_pack),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    fvk.linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt_packed(
        _ptr(k_l2), _ptr(v), _ptr(beta), _ptr(g_cumsum), _ptr(Ai),
        _ptr(ai_pack), _ptr(rhs_w_base), _ptr(rhs_u_base),
        _ptr(w_pack_base), _ptr(u_pack_base),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    torch.testing.assert_close(w_pack, w_pack_base, rtol=0, atol=0)
    torch.testing.assert_close(u_pack, u_pack_base, rtol=0, atol=0)


@pytest.mark.parametrize("S", [64, 128])
def test_linear_attn_gdn_wy_recompute_wu_nogate_packed_rhs_matches_gated(S):
    fvk = _load_fvk()
    if not hasattr(
            fvk,
            "linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs_nogate"):
        pytest.skip("nogate packed_rhs recompute_wu kernel is not built")
    torch.manual_seed(20260529 + S)
    chunks = (S + 63) // 64
    k = torch.randn(S, 16, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    v = torch.randn(S, 48, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    beta = torch.rand(S, 48, device="cuda", dtype=torch.bfloat16) * 0.8
    g = (torch.randn(S, 48, device="cuda") * 0.015).to(torch.bfloat16)
    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)

    k_pack = torch.empty(chunks, 16, 64, 128, device="cuda",
                         dtype=torch.bfloat16)
    kkt_base = torch.empty(chunks, 16, 64, 64, device="cuda",
                           dtype=torch.float32)
    A_gated = torch.empty(chunks, 48, 64, 64, device="cuda",
                          dtype=torch.float32)
    A_nogate = torch.empty_like(A_gated)
    Ai_gated = torch.empty_like(A_gated)
    Ai_nogate = torch.empty_like(A_gated)
    Ai_pack_gated = torch.empty(chunks, 48, 64, 64, device="cuda",
                                dtype=torch.bfloat16)
    Ai_pack_nogate = torch.empty_like(Ai_pack_gated)

    fvk.linear_attn_gdn_wy_kkt_b64_bf16_cublaslt(
        _ptr(k_l2), _ptr(beta), _ptr(g_cumsum),
        _ptr(k_pack), _ptr(kkt_base), _ptr(A_gated),
        S, 16, 48, 128, 3, 0)
    fvk.linear_attn_gdn_wy_solve_tril_b64_f32_fused_pack(
        _ptr(A_gated), _ptr(Ai_gated), _ptr(Ai_pack_gated), S, 48, 0)
    fvk.linear_attn_gdn_wy_kkt_b64_bf16_cublaslt_nogate(
        _ptr(k_l2), _ptr(beta), _ptr(k_pack), _ptr(kkt_base),
        _ptr(A_nogate), S, 16, 48, 128, 3, 0)
    fvk.linear_attn_gdn_wy_solve_tril_b64_f32_fused_pack(
        _ptr(A_nogate), _ptr(Ai_nogate), _ptr(Ai_pack_nogate), S, 48, 0)

    rhs_w_g = torch.empty(chunks, 48, 64, 128, device="cuda",
                          dtype=torch.bfloat16)
    rhs_u_g = torch.empty_like(rhs_w_g)
    w_g = torch.empty_like(rhs_w_g)
    u_g = torch.empty_like(rhs_w_g)
    fvk.linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs(
        _ptr(k_l2), _ptr(v), _ptr(beta), _ptr(g_cumsum),
        _ptr(Ai_pack_gated), _ptr(rhs_w_g), _ptr(rhs_u_g),
        _ptr(w_g), _ptr(u_g), S, 16, 48, 128, 3, 0)

    rhs_w_n = torch.empty_like(rhs_w_g)
    rhs_u_n = torch.empty_like(rhs_w_g)
    w_n = torch.empty_like(rhs_w_g)
    u_n = torch.empty_like(rhs_w_g)
    fvk.linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs_nogate(
        _ptr(k_l2), _ptr(v), _ptr(beta), _ptr(g_cumsum),
        _ptr(Ai_pack_nogate), _ptr(rhs_w_n), _ptr(rhs_u_n),
        _ptr(w_n), _ptr(u_n), S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    torch.testing.assert_close(w_n, w_g, rtol=0, atol=0.035)
    torch.testing.assert_close(u_n, u_g, rtol=0, atol=0.035)


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
def test_linear_attn_gdn_wy_chunk_h_cublaslt_packed_wu_matches_reference(S):
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
    w_pack = torch.zeros_like(k_pack_hv)
    u_pack = torch.zeros_like(k_pack_hv)
    for ci, start in enumerate(range(0, S, 64)):
        end = min(start + 64, S)
        w_pack[ci, :, :end - start] = w[start:end].transpose(0, 1)
        u_pack[ci, :, :end - start] = u[start:end].transpose(0, 1)
    w_pack_input = w_pack.clone()
    wh_pack = torch.empty_like(k_pack_hv)

    fvk.linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_packed_wu(
        _ptr(k_l2), _ptr(w_pack), _ptr(u_pack), _ptr(g_cumsum), _ptr(state),
        _ptr(h0), _ptr(v_new), _ptr(k_pack_hv), _ptr(wh_pack),
        _ptr(w_pack), S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    state_base = state0.clone()
    h0_base = torch.empty_like(h0)
    v_new_base = torch.empty_like(v_new)
    k_pack_base = torch.empty_like(k_pack_hv)
    w_pack_base = torch.empty_like(k_pack_hv)
    u_pack_base = torch.empty_like(k_pack_hv)
    wh_pack_base = torch.empty_like(k_pack_hv)
    decayed_base = torch.empty_like(k_pack_hv)
    fvk.linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt(
        _ptr(k_l2), _ptr(u), _ptr(w), _ptr(g_cumsum), _ptr(state_base),
        _ptr(h0_base), _ptr(v_new_base), _ptr(k_pack_base),
        _ptr(w_pack_base), _ptr(u_pack_base), _ptr(wh_pack_base),
        _ptr(decayed_base), S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    torch.testing.assert_close(h0, h0_base, rtol=0, atol=0)
    torch.testing.assert_close(v_new, v_new_base, rtol=0, atol=0)
    torch.testing.assert_close(state, state_base, rtol=0, atol=0)

    v_pack_ref = torch.zeros_like(k_pack_hv)
    for ci, start in enumerate(range(0, S, 64)):
        end = min(start + 64, S)
        v_pack_ref[ci, :, :end - start] = v_new_base[start:end].transpose(0, 1)
    torch.testing.assert_close(wh_pack, v_pack_ref, rtol=0, atol=0)

    state_no_v = state0.clone()
    h0_no_v = torch.empty_like(h0)
    k_pack_no_v = torch.empty_like(k_pack_hv)
    v_pack_no_v = torch.empty_like(k_pack_hv)
    decayed_no_v = torch.empty_like(k_pack_hv)
    fvk.linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_packed_wu(
        _ptr(k_l2), _ptr(w_pack_input), _ptr(u_pack), _ptr(g_cumsum),
        _ptr(state_no_v), _ptr(h0_no_v), 0, _ptr(k_pack_no_v),
        _ptr(v_pack_no_v), _ptr(decayed_no_v),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()
    torch.testing.assert_close(h0_no_v, h0_base, rtol=0, atol=0)
    torch.testing.assert_close(state_no_v, state_base, rtol=0, atol=0)
    torch.testing.assert_close(v_pack_no_v, v_pack_ref, rtol=0, atol=0)


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


@pytest.mark.parametrize("S", [6, 64, 65])
def test_linear_attn_gdn_wy_output_o_cublaslt_packed_kv_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(8567 + S)
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
    k_pack_hv = torch.zeros(chunks, 48, 64, 128, device="cuda",
                            dtype=torch.bfloat16)
    v_pack = torch.zeros_like(k_pack_hv)
    for ci, start in enumerate(range(0, S, 64)):
        end = min(start + 64, S)
        for vh in range(48):
            k_pack_hv[ci, vh, :end - start] = k_l2[start:end, vh // 3]
        v_pack[ci, :, :end - start] = v_new[start:end].transpose(0, 1)
    q_pack = torch.empty_like(k_pack_hv)
    qk_base = torch.empty(chunks, 48, 64, 64, device="cuda",
                          dtype=torch.float32)
    local_a_pack = torch.empty(chunks, 48, 64, 64, device="cuda",
                               dtype=torch.bfloat16)
    qh_pack = torch.empty_like(k_pack_hv)
    local_pack = torch.empty_like(k_pack_hv)
    out = torch.empty(S, 48, 128, device="cuda", dtype=torch.bfloat16)

    fvk.linear_attn_gdn_wy_output_o_b64_bf16_cublaslt_packed_kv(
        _ptr(q_l2), _ptr(k_pack_hv), _ptr(v_pack), _ptr(h0),
        _ptr(g_cumsum), _ptr(q_pack), _ptr(qk_base), _ptr(local_a_pack),
        _ptr(qh_pack), _ptr(local_pack), _ptr(out),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    out_ref = _output_o_ref(q_l2, k_l2, v_new, h0, g_cumsum)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=3e-2)


@pytest.mark.parametrize("S", [6, 64, 65])
def test_linear_attn_gdn_wy_output_o_cublaslt_packed_qkv_matches_reference(S):
    fvk = _load_fvk()
    torch.manual_seed(8667 + S)
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
    q_pack = torch.zeros(chunks, 48, 64, 128, device="cuda",
                         dtype=torch.bfloat16)
    k_pack_hv = torch.zeros_like(q_pack)
    v_pack = torch.zeros_like(q_pack)
    for ci, start in enumerate(range(0, S, 64)):
        end = min(start + 64, S)
        for vh in range(48):
            q_pack[ci, vh, :end - start] = q_l2[start:end, vh // 3]
            k_pack_hv[ci, vh, :end - start] = k_l2[start:end, vh // 3]
        v_pack[ci, :, :end - start] = v_new[start:end].transpose(0, 1)
    qk_base = torch.empty(chunks, 48, 64, 64, device="cuda",
                          dtype=torch.float32)
    local_a_pack = torch.empty(chunks, 48, 64, 64, device="cuda",
                               dtype=torch.bfloat16)
    qh_pack = torch.empty_like(q_pack)
    local_pack = torch.empty_like(q_pack)
    out = torch.empty(S, 48, 128, device="cuda", dtype=torch.bfloat16)

    fvk.linear_attn_gdn_wy_output_o_b64_bf16_cublaslt_packed_qkv(
        _ptr(q_pack), _ptr(k_pack_hv), _ptr(v_pack), _ptr(h0),
        _ptr(g_cumsum), _ptr(qk_base), _ptr(local_a_pack),
        _ptr(qh_pack), _ptr(local_pack), _ptr(out),
        S, 16, 48, 128, 3, 0)
    torch.cuda.synchronize()

    out_ref = _output_o_ref(q_l2, k_l2, v_new, h0, g_cumsum)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=3e-2)
