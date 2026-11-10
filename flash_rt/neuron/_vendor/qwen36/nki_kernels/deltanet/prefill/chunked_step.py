"""NKI per-chunk DeltaNet kernel for CTE (context encoding / prefill).

Single-chunk kernel: processes one chunk (128 tokens) with a stable
triangular solve for intra-chunk correction. The caller loops over chunks in
PyTorch, passing state between calls.

Each kernel call:
  - Takes one chunk of data: q, k, v, beta, g_cumsum, g_last  (all 128x128)
  - Takes recurrent state_in (128x128)
  - Returns chunk output (128x128) and state_out (128x128)

No sequence-indexed DMA inside the kernel -- all inputs/outputs are full tiles.
This avoids the DMA OOB issue seen with nl.sequential_range + slice indexing
in the NxDI model compilation context.

NKI v3 (SDK 2.29, NKI 0.3.0). Uses nki.* namespace.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128

# Broadcast partition 0 to all partitions in a 32-wide group.
_BROADCAST_MASK = [0] * 32


@nki.jit
def deltanet_chunk_step(
    query,  # (128, 128) float32 -- one chunk, l2-normed+scaled
    key,  # (128, 128) float32 -- one chunk, l2-normed
    value,  # (128, 128) float32 -- one chunk
    beta_broadcast,  # (128, 128) float32 -- write gate broadcast to 128
    g_cumsum,  # (128, 128) float32 -- cumsum of g within chunk, broadcast
    g_last,  # (128, 128) float32 -- g_cumsum[-1], constant in chunk, broadcast
    state_in,  # (128, 128) float32 -- recurrent state from previous chunk
    lower_mask,  # (128, 128) float32 -- strict lower triangular
    identity,  # (128, 128) float32 -- identity matrix
    lower_mask_diag,  # (128, 128) float32 -- lower tri with diagonal
):
    """Process one chunk of DeltaNet.

    Returns:
        output:    (128, 128) float32 -- chunk output
        state_out: (128, 128) float32 -- updated recurrent state
    """
    C, dim = query.shape  # C = 128, dim = 128

    # Output tensors in HBM
    output = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load all inputs into SBUF
    q_c = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_c, src=query)

    k_c = nl.ndarray((P_MAX, dim), dtype=key.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_c, src=key)

    v_c = nl.ndarray((P_MAX, dim), dtype=value.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_c, src=value)

    beta_c = nl.ndarray((P_MAX, dim), dtype=beta_broadcast.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_c, src=beta_broadcast)

    gc_c = nl.ndarray((P_MAX, dim), dtype=g_cumsum.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc_c, src=g_cumsum)

    gl_c = nl.ndarray((P_MAX, dim), dtype=g_last.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gl_c, src=g_last)

    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=state, src=state_in)

    # Load masks
    eye = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eye, src=identity)

    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)

    Lmask_d = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask_d, src=lower_mask_diag)

    # ============================================================
    # k_beta = K * beta, v_beta = V * beta
    # ============================================================
    k_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=k_beta, data1=k_c, data2=beta_c, op=nl.multiply)

    v_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=v_beta, data1=v_c, data2=beta_c, op=nl.multiply)

    # ============================================================
    # Stable decay factors from cumulative log-decay
    #
    # The caller passes g_cumsum and g_last broadcast to (128, 128).  Extract
    # one column and build pairwise decays as exp(gc[i] - gc[j]) so no
    # individual exp(-gc[j]) term can overflow.
    # ============================================================
    gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gc_p[0:P_MAX, 0:1], src=gc_c[0:P_MAX, 0:1])

    gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gl_p[0:P_MAX, 0:1], src=gl_c[0:P_MAX, 0:1])

    exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gc_p[0:P_MAX, 0:1],
        op=nl.exp,
        data=gc_p[0:P_MAX, 0:1],
        bias=None,
        scale=1.0,
    )

    exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gl_p[0:P_MAX, 0:1],
        op=nl.exp,
        data=gl_p[0:P_MAX, 0:1],
        bias=None,
        scale=1.0,
    )

    gc_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=gc_padded, value=0.0)
    nisa.tensor_copy(dst=gc_padded[0:P_MAX, 0:1], src=gc_p[0:P_MAX, 0:1])

    gc_row_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=gc_row_psum, data=gc_padded)

    gc_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gc_row[0:1, 0:P_MAX], src=gc_row_psum[0:1, 0:P_MAX])

    gc_row_broadcast = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=gc_row[0:1, 0:P_MAX],
            dst=gc_row_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
            shuffle_mask=_BROADCAST_MASK,
        )

    gc_col_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=gc_col_strict,
        data=Lmask,
        op0=nl.multiply,
        operand0=gc_p,
        engine=nisa.vector_engine,
    )

    gc_row_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gc_row_strict, data1=gc_row_broadcast, data2=Lmask, op=nl.multiply
    )

    g_diff_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=g_diff_strict,
        data1=gc_col_strict,
        data2=gc_row_strict,
        op=nl.subtract,
    )

    decay_strict_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=decay_strict_raw,
        op=nl.exp,
        data=g_diff_strict,
        bias=None,
        scale=1.0,
    )

    decay_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=decay_strict, data1=decay_strict_raw, data2=Lmask, op=nl.multiply
    )

    gc_col_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=gc_col_diag,
        data=Lmask_d,
        op0=nl.multiply,
        operand0=gc_p,
        engine=nisa.vector_engine,
    )

    gc_row_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gc_row_diag, data1=gc_row_broadcast, data2=Lmask_d, op=nl.multiply
    )

    g_diff_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=g_diff_diag,
        data1=gc_col_diag,
        data2=gc_row_diag,
        op=nl.subtract,
    )

    decay_diag_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=decay_diag_raw,
        op=nl.exp,
        data=g_diff_diag,
        bias=None,
        scale=1.0,
    )

    decay_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=decay_diag, data1=decay_diag_raw, data2=Lmask_d, op=nl.multiply
    )

    # ============================================================
    # Build the A matrix (intra-chunk correction)
    # QK = k_beta @ k^T  -- contract over features
    # ============================================================
    kb_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kb_T_psum, stationary=k_beta, moving=eye)

    kb_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kb_T, src=kb_T_psum)

    k_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=k_T_psum, stationary=k_c, moving=eye)

    k_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=k_T, src=k_T_psum)

    QK_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=QK_psum, stationary=kb_T, moving=k_T)

    QK = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=QK, src=QK_psum)

    # QK_decay[i,j] = QK[i,j] * exp(gc[i] - gc[j]) for i > j.
    QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=QK_decay, data1=QK, data2=decay_strict, op=nl.multiply)

    # A = -QK_decay * lower_mask
    neg_QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=neg_QK_decay,
        data=QK_decay,
        op0=nl.multiply,
        operand0=-1.0,
        engine=nisa.vector_engine,
    )

    A_mat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=A_mat, data1=neg_QK_decay, data2=Lmask, op=nl.multiply)

    # ============================================================
    # Stable triangular solve: N = inv(I - A_mat)
    #
    # A_mat is strictly lower triangular. Solve two 64x64 diagonal blocks
    # row-by-row, then merge the lower-left block. This is equivalent to the
    # nilpotent Neumann series but avoids repeated squaring of A.
    # ============================================================
    P_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=P_acc, value=0.0)

    A_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=A_T_psum, data=A_mat)

    A_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=A_T, src=A_T_psum)

    col_mask_left_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=col_mask_left_row, value=0.0)
    nisa.memset(dst=col_mask_left_row[0:1, 0:64], value=1.0)

    col_mask_left = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=col_mask_left_row[0:1, 0:P_MAX],
            dst=col_mask_left[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
            shuffle_mask=_BROADCAST_MASK,
        )

    col_mask_right_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=col_mask_right_row, value=0.0)
    nisa.memset(dst=col_mask_right_row[0:1, 64:P_MAX], value=1.0)

    col_mask_right = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=col_mask_right_row[0:1, 0:P_MAX],
            dst=col_mask_right[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
            shuffle_mask=_BROADCAST_MASK,
        )

    block_row_mask_bottom = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=block_row_mask_bottom[0:P_MAX, 0:1],
        src=Lmask_d[0:P_MAX, 64:65],
    )

    for solve_i in nl.static_range(64):
        row_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=row_psum, stationary=A_T, moving=P_acc)

        row_prod = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=row_prod, src=row_psum)

        row_with_eye = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=row_with_eye, data1=row_prod, data2=eye, op=nl.add)

        row_col_masked = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=row_col_masked,
            data1=row_with_eye,
            data2=col_mask_left,
            op=nl.multiply,
        )

        row_mask = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=row_mask[0:P_MAX, 0:1],
            src=eye[0:P_MAX, solve_i : solve_i + 1],
        )

        row_update = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=row_update,
            data=row_col_masked,
            op0=nl.multiply,
            operand0=row_mask,
            engine=nisa.vector_engine,
        )

        P_next = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=P_next, data1=P_acc, data2=row_update, op=nl.add)
        nisa.tensor_copy(dst=P_acc, src=P_next)

    for solve_i in nl.static_range(64):
        row_idx = 64 + solve_i

        row_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=row_psum, stationary=A_T, moving=P_acc)

        row_prod = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=row_prod, src=row_psum)

        row_with_eye = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=row_with_eye, data1=row_prod, data2=eye, op=nl.add)

        row_col_masked = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=row_col_masked,
            data1=row_with_eye,
            data2=col_mask_right,
            op=nl.multiply,
        )

        row_mask = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=row_mask[0:P_MAX, 0:1],
            src=eye[0:P_MAX, row_idx : row_idx + 1],
        )

        row_update = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=row_update,
            data=row_col_masked,
            op0=nl.multiply,
            operand0=row_mask,
            engine=nisa.vector_engine,
        )

        P_next = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=P_next, data1=P_acc, data2=row_update, op=nl.add)
        nisa.tensor_copy(dst=P_acc, src=P_next)

    N_diag_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=N_diag_T_psum, data=P_acc)

    N_diag_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=N_diag_T, src=N_diag_T_psum)

    tmp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=tmp_psum, stationary=N_diag_T, moving=A_mat)

    tmp = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tmp, src=tmp_psum)

    tmp_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=tmp_T_psum, data=tmp)

    tmp_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tmp_T, src=tmp_T_psum)

    N21_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=N21_psum, stationary=tmp_T, moving=P_acc)

    N21 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=N21, src=N21_psum)

    N21_col_masked = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=N21_col_masked,
        data1=N21,
        data2=col_mask_left,
        op=nl.multiply,
    )

    N21_block = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=N21_block,
        data=N21_col_masked,
        op0=nl.multiply,
        operand0=block_row_mask_bottom,
        engine=nisa.vector_engine,
    )
    nisa.tensor_tensor(dst=P_acc, data1=P_acc, data2=N21_block, op=nl.add)

    # ============================================================
    # Apply N: value_corr = N @ v_beta, k_cumdecay = N @ (k_beta * exp_gc)
    # ============================================================
    N_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=N_T_psum, data=P_acc)

    N_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=N_T, src=N_T_psum)

    vc_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=vc_psum, stationary=N_T, moving=v_beta)

    value_corr = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=value_corr, src=vc_psum)

    kb_exp_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=kb_exp_gc,
        data=k_beta,
        op0=nl.multiply,
        operand0=exp_gc_p,
        engine=nisa.vector_engine,
    )

    kcd_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kcd_psum, stationary=N_T, moving=kb_exp_gc)

    k_cumdecay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=k_cumdecay, src=kcd_psum)

    # ============================================================
    # Inter-chunk state propagation
    # attn_intra = (q @ k^T) * decay_mask * lower_mask_diag
    # ============================================================
    q_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=q_T_psum, stationary=q_c, moving=eye)

    q_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_T, src=q_T_psum)

    qk_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qk_psum, stationary=q_T, moving=k_T)

    qk_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qk_raw, src=qk_psum)

    attn_intra = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=attn_intra, data1=qk_raw, data2=decay_diag, op=nl.multiply)

    # ============================================================
    # v_prime = k_cumdecay @ state
    # ============================================================
    kcd_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kcd_T_psum, stationary=k_cumdecay, moving=eye)

    kcd_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kcd_T, src=kcd_T_psum)

    vp_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=vp_psum, stationary=kcd_T, moving=state)

    v_prime = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=v_prime, src=vp_psum)

    v_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=v_new, data1=value_corr, data2=v_prime, op=nl.subtract)

    # ============================================================
    # attn_inter = (q * exp(g_cumsum)) @ state
    # ============================================================
    q_exp = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=q_exp,
        data=q_c,
        op0=nl.multiply,
        operand0=exp_gc_p,
        engine=nisa.vector_engine,
    )

    qe_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qe_T_psum, stationary=q_exp, moving=eye)

    qe_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qe_T, src=qe_T_psum)

    ai_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ai_psum, stationary=qe_T, moving=state)

    attn_inter = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=attn_inter, src=ai_psum)

    # ============================================================
    # attn_intra @ v_new
    # ============================================================
    ai_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ai_T_psum, stationary=attn_intra, moving=eye)

    ai_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ai_T, src=ai_T_psum)

    intra_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=intra_psum, stationary=ai_T, moving=v_new)

    intra_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=intra_out, src=intra_psum)

    # ============================================================
    # chunk_output = attn_inter + intra_out
    # ============================================================
    chunk_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=chunk_out, data1=attn_inter, data2=intra_out, op=nl.add)

    nisa.dma_copy(dst=output, src=chunk_out)

    # ============================================================
    # State update: state_new = state * exp(g_last)
    #                         + (k * exp(g_last - gc))^T @ v_new
    # ============================================================
    gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gl_minus_gc_p,
        data1=gl_p,
        data2=gc_p,
        op=nl.subtract,
    )

    exp_gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gl_minus_gc_p,
        op=nl.exp,
        data=gl_minus_gc_p,
        bias=None,
        scale=1.0,
    )

    k_raw_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=k_raw_decay,
        data=k_c,
        op0=nl.multiply,
        operand0=exp_gl_minus_gc_p,
        engine=nisa.vector_engine,
    )

    kv_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kv_psum, stationary=k_raw_decay, moving=v_new)

    kv_outer = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kv_outer, src=kv_psum)

    state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=state_decayed,
        data=state,
        op0=nl.multiply,
        operand0=exp_gl_p,
        engine=nisa.vector_engine,
    )

    state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=state_new, data1=state_decayed, data2=kv_outer, op=nl.add)

    nisa.dma_copy(dst=state_out, src=state_new)

    return output, state_out
