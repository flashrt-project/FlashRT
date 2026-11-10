"""Fused single-kernel DeltaNet chunked forward for CTE (context encoding).

Status: gated OFF for this checkpoint -- its split decay form exp(gc[i]) * exp(-gc[j]) overflows fp32
at this checkpoint's gating magnitude and yields NaN logits, so chunked_step.py is the default.

Processes all chunks for one (batch, head) pair in a single call, with the 128x128 state resident in
SBUF across chunks. Chunk size = k_dim = v_dim = 128 = P_MAX, one tile per chunk. Per chunk, a
Neumann-series power-doubling builds the intra-chunk correction:
    A          = -QK_decay * lower_mask
    N          = (I+A)(I+A^2)(I+A^4)...(I+A^64)     6 rounds
    value_corr = N @ v_beta
    k_cumdecay = N @ (k_beta * exp(gc))

Then the state propagates between chunks:
    v_new      = value_corr - k_cumdecay @ state
    attn_inter = (q * exp(gc)) @ state
    attn_intra = (q @ k^T) * decay_mask * lower_mask_diag
    output     = attn_inter + attn_intra @ v_new
    state      = exp(g_last) * (state + k_raw_decay^T @ v_new)
Optimizations over chunked_step.py: specs/deltanet_prefill.md.
"""

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128  # Partition dim = chunk_size = k_dim = v_dim
CHUNK_SIZE = 128

# Broadcast partition 0 to all partitions in a 32-wide group
_BROADCAST_MASK = [0] * 32


def _make_lower_mask():
    """Strict lower triangular (128x128) as numpy constant."""
    return np.tril(np.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32), k=-1)


def _make_lower_mask_diag():
    """Lower triangular with diagonal (128x128) as numpy constant."""
    return np.tril(np.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32), k=0)


def _make_identity():
    """Identity matrix (128x128) as numpy constant."""
    return np.eye(CHUNK_SIZE, dtype=np.float32)


@nki.jit
def deltanet_fused_chunked_fwd(
    query: nl.ndarray,  # (S, 128) float32 — l2-normed and scaled
    key: nl.ndarray,  # (S, 128) float32 — l2-normed
    value: nl.ndarray,  # (S, 128) float32
    g_in: nl.ndarray,  # (S, 1)   float32 — per-token log-decay (NOT cumsum)
    beta_in: nl.ndarray,  # (S, 1)   float32 — per-token write gate
    lower_mask: nl.ndarray,  # (128, 128) float32 — strict lower tri
    identity: nl.ndarray,  # (128, 128) float32 — identity
    lower_mask_diag: nl.ndarray,  # (128, 128) float32 — lower tri with diag
):
    """Fused chunked DeltaNet forward — single kernel call per (batch, head).

    Processes all chunks sequentially within the kernel, keeping the recurrent
    state (128x128) in SBUF across chunks.  Returns per-token output and
    final state.

    Input requirements:
      - S must be divisible by 128 (pad before calling)
      - query must be l2-normed and scaled by 1/sqrt(k_dim)
      - key must be l2-normed
      - g_in is RAW log-decay (cumsum computed in-kernel via tensor_tensor_scan)
      - beta_in is sigmoid(b) (write gate)

    Returns:
        output:      (S, 128) float32
        final_state: (128, 128) float32
    """
    seq_len = query.shape[0]
    dim = query.shape[1]  # 128
    num_chunks = seq_len // CHUNK_SIZE

    # Output tensors in HBM
    output = nl.ndarray((seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    final_state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # ================================================================
    # Load constant masks into SBUF once (reused across all chunks)
    # ================================================================
    eye = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eye, src=identity)

    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)

    Lmask_d = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask_d, src=lower_mask_diag)

    # Ones vector for cumsum scan: (1, CHUNK_SIZE)
    ones_1xC = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_1xC, value=1.0)

    # Zero initial for cumsum scan
    zero_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_11, value=0.0)

    # ================================================================
    # Initialize recurrent state in SBUF — persists across ALL chunks
    # ================================================================
    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=state, value=0.0)

    # ================================================================
    # Sequential chunk processing
    # ================================================================
    for i_chunk in nl.sequential_range(num_chunks):
        chunk_start = i_chunk * CHUNK_SIZE

        # ---- Load chunk data from HBM ----
        q_c = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_c,
            src=query[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
        )

        k_c = nl.ndarray((P_MAX, dim), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k_c,
            src=key[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
        )

        v_c = nl.ndarray((P_MAX, dim), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=v_c,
            src=value[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
        )

        # g: (CHUNK_SIZE, 1) — raw log-decay per token
        g_chunk_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=g_chunk_p[0:CHUNK_SIZE, 0:1],
            src=g_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        # beta: (CHUNK_SIZE, 1) — write gate scalar per token
        beta_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=beta_p[0:CHUNK_SIZE, 0:1],
            src=beta_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        # ---- In-kernel cumsum of g via tensor_tensor_scan ----
        # Need g as (1, CHUNK_SIZE) for scan along free dim.
        # Transpose: (CHUNK_SIZE, 1) -> (1, CHUNK_SIZE) via nc_transpose
        g_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=g_padded, value=0.0)
        nisa.tensor_copy(
            dst=g_padded[0:CHUNK_SIZE, 0:1],
            src=g_chunk_p[0:CHUNK_SIZE, 0:1],
        )

        g_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=g_tp_psum, data=g_padded)

        g_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=g_row[0:1, 0:CHUNK_SIZE],
            src=g_tp_psum[0:1, 0:CHUNK_SIZE],
        )

        # cumsum: gc_row[t] = 1.0 * gc_row[t-1] + g_row[t]
        gc_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor_scan(
            dst=gc_row[0:1, 0:CHUNK_SIZE],
            data0=ones_1xC[0:1, 0:CHUNK_SIZE],
            data1=g_row[0:1, 0:CHUNK_SIZE],
            initial=zero_11[0:1, 0:1],
            op0=nl.multiply,
            op1=nl.add,
        )

        # Transpose gc back to (CHUNK_SIZE, 1) partition layout
        gc_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=gc_padded, value=0.0)
        nisa.tensor_copy(
            dst=gc_padded[0:1, 0:CHUNK_SIZE],
            src=gc_row[0:1, 0:CHUNK_SIZE],
        )

        gc_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=gc_tp_psum, data=gc_padded)

        # gc_p: (P_MAX, 1) — cumulative sum of g per token in this chunk
        gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=gc_p[0:CHUNK_SIZE, 0:1],
            src=gc_tp_psum[0:CHUNK_SIZE, 0:1],
        )

        # g_last = gc[-1] (scalar) — needed for state decay
        gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=gl_11[0:1, 0:1],
            src=gc_row[0:1, CHUNK_SIZE - 1 : CHUNK_SIZE],
        )

        # ---- Compute exp(gc), exp(-gc), exp(g_last) as (P_MAX, 1) scalars ----
        # These (P_MAX, 1) tensors are used with tensor_scalar to broadcast
        # across the free dimension without explicit (P_MAX, dim) copies.

        exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gc_p[0:P_MAX, 0:1],
            op=nl.exp,
            data=gc_p[0:P_MAX, 0:1],
            bias=None,
            scale=1.0,
        )

        neg_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=neg_gc_p,
            data=gc_p,
            op0=nl.multiply,
            operand0=-1.0,
            engine=nisa.vector_engine,
        )

        exp_neg_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_neg_gc_p[0:P_MAX, 0:1],
            op=nl.exp,
            data=neg_gc_p[0:P_MAX, 0:1],
            bias=None,
            scale=1.0,
        )

        # exp(g_last): scalar, then broadcast to (P_MAX, 1)
        exp_gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gl_11,
            op=nl.exp,
            data=gl_11,
            bias=None,
            scale=1.0,
        )

        exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=exp_gl_11[0:1, 0:1],
                dst=exp_gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        # ============================================================
        # k_beta = K * beta, v_beta = V * beta
        # tensor_scalar broadcasts beta_p (P_MAX, 1) across free dim
        # ============================================================
        k_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_beta,
            data=k_c,
            op0=nl.multiply,
            operand0=beta_p,
            engine=nisa.vector_engine,
        )

        v_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=v_beta,
            data=v_c,
            op0=nl.multiply,
            operand0=beta_p,
            engine=nisa.vector_engine,
        )

        # ============================================================
        # Build the A matrix (intra-chunk correction)
        # Transpose K and K_beta for matmul
        # ============================================================
        kb_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kb_T_psum, data=k_beta)

        kb_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kb_T, src=kb_T_psum)

        k_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=k_T_psum, data=k_c)

        k_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_T, src=k_T_psum)

        # QK = k_beta^T @ k  (contract over features)
        QK_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=QK_psum, stationary=kb_T, moving=k_T)

        QK = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=QK, src=QK_psum)

        # ============================================================
        # Decay mask: QK_decay[i,j] = QK[i,j] * exp(gc[i]) * exp(-gc[j])
        #
        # Apply the strict causal mask before the split exp(gc) / exp(-gc)
        # scaling. Upper-triangular entries are mathematically unused, but
        # scaling them first can create very large finite values that poison
        # later matmuls before the mask is applied.
        # ============================================================
        QK_masked = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=QK_masked, data1=QK, data2=Lmask, op=nl.multiply)

        # Row scaling: QK_row[i,:] = QK[i,:] * exp(gc[i])
        # Then transpose, column scale, transpose back.
        # Uses tensor_scalar with (P_MAX,1) operand for row scaling.
        QK_row = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=QK_row,
            data=QK_masked,
            op0=nl.multiply,
            operand0=exp_gc_p,
            engine=nisa.vector_engine,
        )

        # Transpose to scale columns (now rows in transposed view)
        QK_r_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=QK_r_T_psum, data=QK_row)

        QK_r_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=QK_r_T, src=QK_r_T_psum)

        QK_r_T_col = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=QK_r_T_col,
            data=QK_r_T,
            op0=nl.multiply,
            operand0=exp_neg_gc_p,
            engine=nisa.vector_engine,
        )

        # Transpose back
        QK_d_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=QK_d_psum, data=QK_r_T_col)

        QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=QK_decay, src=QK_d_psum)

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
        # Neumann power-doubling: N = (I+A)(I+A^2)...(I+A^{64})
        # 6 rounds → resolves rank up to 2^6 = 64 (sufficient for chunk=128)
        # ============================================================
        P_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=P_acc, data1=eye, data2=A_mat, op=nl.add)

        A_pow = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=A_pow, src=A_mat)

        for _round in nl.sequential_range(6):
            # A_pow = A_pow^2: transpose A_pow, then matmul
            Ap_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=Ap_T_psum, data=A_pow)

            Ap_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=Ap_T, src=Ap_T_psum)

            Ap_sq_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=Ap_sq_psum, stationary=Ap_T, moving=A_pow)
            nisa.tensor_copy(dst=A_pow, src=Ap_sq_psum)

            # P_acc = (I + A_pow) @ P_acc: transpose IpA, then matmul
            IpA = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=IpA, data1=eye, data2=A_pow, op=nl.add)

            IpA_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=IpA_T_psum, data=IpA)

            IpA_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=IpA_T, src=IpA_T_psum)

            Pacc_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=Pacc_psum, stationary=IpA_T, moving=P_acc)
            nisa.tensor_copy(dst=P_acc, src=Pacc_psum)

        # ============================================================
        # Apply N: value_corr = N @ v_beta
        #          k_cumdecay = N @ (k_beta * exp(gc))
        # ============================================================
        N_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=N_T_psum, data=P_acc)

        N_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=N_T, src=N_T_psum)

        vc_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=vc_psum, stationary=N_T, moving=v_beta)

        value_corr = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=value_corr, src=vc_psum)

        # k_beta * exp(gc): row-scaled
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
        nisa.nc_transpose(dst=q_T_psum, data=q_c)

        q_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_T, src=q_T_psum)

        qk_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=qk_psum, stationary=q_T, moving=k_T)

        qk_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qk_raw, src=qk_psum)

        # Mask before split scaling for the same reason as the A matrix above:
        # upper-triangular decay factors are unused and can be numerically huge.
        qk_masked = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=qk_masked, data1=qk_raw, data2=Lmask_d, op=nl.multiply)

        # Row-scale by exp(gc)
        qk_row = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=qk_row,
            data=qk_masked,
            op0=nl.multiply,
            operand0=exp_gc_p,
            engine=nisa.vector_engine,
        )

        # Transpose, column-scale by exp(-gc), transpose back
        qk_r_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=qk_r_T_psum, data=qk_row)

        qk_r_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qk_r_T, src=qk_r_T_psum)

        qk_r_T_col = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=qk_r_T_col,
            data=qk_r_T,
            op0=nl.multiply,
            operand0=exp_neg_gc_p,
            engine=nisa.vector_engine,
        )

        qk_d_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=qk_d_psum, data=qk_r_T_col)

        qk_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qk_decay, src=qk_d_psum)

        attn_intra = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=attn_intra, data1=qk_decay, data2=Lmask_d, op=nl.multiply
        )

        # ============================================================
        # v_prime = k_cumdecay @ state   (state is in SBUF!)
        # ============================================================
        kcd_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kcd_T_psum, data=k_cumdecay)

        kcd_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kcd_T, src=kcd_T_psum)

        vp_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=vp_psum, stationary=kcd_T, moving=state)

        v_prime = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=v_prime, src=vp_psum)

        v_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=v_new, data1=value_corr, data2=v_prime, op=nl.subtract)

        # ============================================================
        # attn_inter = (q * exp(gc)) @ state   (state is in SBUF!)
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
        nisa.nc_transpose(dst=qe_T_psum, data=q_exp)

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
        nisa.nc_transpose(dst=ai_T_psum, data=attn_intra)

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

        # Store output chunk to HBM
        nisa.dma_copy(
            dst=output[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            src=chunk_out,
        )

        # ============================================================
        # State update: state = exp(g_last) * (state + k_raw_decay^T @ v_new)
        # state is updated IN-PLACE in SBUF — no HBM round-trip!
        # ============================================================

        # k_raw_decay contributes as exp(g_last) * (k * exp(-gc))^T @ v_new.
        # Compute the equivalent stable form k * exp(g_last - gc), so the
        # factor is always <= 1 for valid causal positions.
        exp_gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=exp_gl_minus_gc_p,
            data1=exp_gl_p,
            data2=exp_neg_gc_p,
            op=nl.multiply,
        )

        # k_raw_decay = k * exp(g_last - gc)
        k_raw_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_raw_decay,
            data=k_c,
            op0=nl.multiply,
            operand0=exp_gl_minus_gc_p,
            engine=nisa.vector_engine,
        )

        # k_raw_decay^T @ v_new → (dim, dim) outer product sum
        # nc_matmul: result[M,N] = sum_K stationary[K,M] * moving[K,N]
        # stationary=k_raw_decay (P_MAX, dim), moving=v_new (P_MAX, dim)
        # Result: sum over tokens -> (dim, dim)
        kv_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_psum, stationary=k_raw_decay, moving=v_new)

        kv_outer = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_outer, src=kv_psum)

        # state = state * exp(g_last) + kv_outer
        # tensor_scalar broadcasts exp_gl_p (P_MAX, 1) across free dim.
        state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_decayed,
            data=state,
            op0=nl.multiply,
            operand0=exp_gl_p,
            engine=nisa.vector_engine,
        )
        nisa.tensor_tensor(dst=state, data1=state_decayed, data2=kv_outer, op=nl.add)

    # ---- Write final state to HBM ----
    nisa.dma_copy(dst=final_state_out, src=state)

    return output, final_state_out
