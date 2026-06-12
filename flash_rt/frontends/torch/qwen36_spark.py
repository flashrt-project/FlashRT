"""FlashRT - Qwen3.6-27B NVFP4 Spark frontend (SM121).

Spark reuses the RTX Qwen3.6 compute path, but needs the NVFP4 MTP
prompt-tail K/V prefill that Thor already uses. The RTX parent only
enables the K/V-only tail helper when BF16 shadow MTP weights exist;
the paired public FP8 MTP checkpoint is converted to NVFP4 at load time
and therefore lacks those shadow pointers. Without this override,
``FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL=auto`` collapses to zero and the
long FP8-KV route starts the drafter with an unseeded MTP cache.
"""

from __future__ import annotations

import os

from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx


class Qwen36TorchFrontendSpark(Qwen36TorchFrontendRtx):
    """Qwen3.6 NVFP4 frontend for DGX Spark / GB10 (SM121).

    This class is intentionally narrow: it keeps the RTX attention backend
    and all SM120/SM121 GEMM dispatch, while adding the NVFP4 MTP tail
    prefill path needed for high speculative acceptance length.
    """

    def _long_mtp_prefill_tail_for_prompt(self, prompt_len: int) -> int:
        raw = os.environ.get(
            'FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL', 'auto') or 'auto'
        if raw.lower() != 'auto':
            return max(0, int(raw))
        mtp = self._weights.ptrs.get('mtp') if self._weights else None
        if not isinstance(mtp, dict):
            return 0
        prompt_len = int(prompt_len)
        if prompt_len >= 128 and prompt_len < 512:
            return min(128, prompt_len)
        if prompt_len < 512:
            return 0
        if prompt_len < 768:
            return 512
        if prompt_len < 3072:
            return 2048
        if prompt_len < 6144:
            return 512
        if prompt_len < 24576:
            return 4096
        return 2048

    def _long_tq_effective_k(
            self, prompt_len: int, K: int,
            max_new_tokens: int | None = None) -> int:
        target_k = super()._long_tq_effective_k(
            prompt_len, K, max_new_tokens)
        if os.environ.get('FLASHRT_QWEN36_TQ_SPEC_K', ''):
            return target_k
        prompt_len = int(prompt_len)
        if 6144 <= prompt_len < 12288:
            target_k = 7
        elif 12288 <= prompt_len < 24576:
            target_k = 6
        elif 24576 <= prompt_len < 49152:
            target_k = 7
        caller_k = int(K)
        if caller_k < 6:
            return min(caller_k, target_k)
        return target_k

    def _fp8_xqa_auto_bucket_enabled(
            self, q_seq: int | None, end_pos: int) -> bool:
        """Spark-measured FP8-KV XQA bucket policy.

        GB10/SM121 differs from the RTX 5090 policy inherited by the parent:
        XQA is faster for the 2K and 16K buckets, while the staged BF16 path is
        faster around 8K for the measured K/tail policy.
        """
        end_pos = int(end_pos)
        if end_pos < 256 and q_seq is not None and int(q_seq) <= 7:
            return True
        if end_pos < 6144:
            return True
        if end_pos < 12288:
            return False
        return True

    def _prefill_mtp_tail_kv_nvfp4(
            self, prev_h_rows, token_ids, pos_start: int,
            cache_base_pos: int) -> bool:
        """Populate MTP prompt-tail K/V cache without full MTP decode.

        The parent implementation supports BF16-shadow MTP projection
        weights. Spark's normal public path uses the official FP8 MTP
        checkpoint converted to NVFP4, so compute K/V with the packed
        NVFP4 k/v projections in one batched tail pass.
        """
        import torch

        from flash_rt import flash_rt_kernels as fvk

        mtp = self._weights.ptrs.get('mtp')
        if mtp is None:
            return False
        if 'k_proj_w_bf16' in mtp:
            return super()._prefill_mtp_tail_kv_nvfp4(
                prev_h_rows, token_ids, pos_start, cache_base_pos)

        rows = int(token_ids.numel())
        if rows <= 0:
            return True

        hidden = self._cfg['hidden_size']
        eps = float(self._cfg['rms_norm_eps'])
        s = torch.cuda.current_stream().cuda_stream
        self._ensure_mtp_tail_kv_buffers(rows)

        embed = self._mtp_tail_embed_buf[:rows]
        h_norm = self._mtp_tail_h_norm_buf[:rows]
        e_norm = self._mtp_tail_e_norm_buf[:rows]
        cat_buf = self._mtp_tail_cat_buf[:rows]
        fc_out = self._mtp_tail_fc_out_buf[:rows]
        x_norm = self._mtp_tail_x_norm_buf[:rows]
        k_proj = self._mtp_tail_k_proj_buf[:rows]
        v_proj = self._mtp_tail_v_proj_buf[:rows]
        k_norm = self._mtp_tail_k_norm_buf[:rows * 4]

        fvk.qwen36_embedding_lookup_bf16(
            token_ids.view(-1).data_ptr(),
            int(self._weights.ptrs['embed_w']),
            embed.data_ptr(), rows, hidden, s,
        )
        fvk.rms_norm(
            prev_h_rows.view(rows, hidden).data_ptr(),
            int(mtp['pre_fc_norm_hidden_eff_w']),
            h_norm.data_ptr(), rows, hidden, eps, s,
        )
        fvk.rms_norm(
            embed.data_ptr(), int(mtp['pre_fc_norm_embedding_eff_w']),
            e_norm.data_ptr(), rows, hidden, eps, s,
        )
        fvk.concat2_bf16(
            e_norm.data_ptr(), h_norm.data_ptr(),
            cat_buf.data_ptr(), rows, hidden, hidden, s,
        )
        self._mtp_tail_fc_matmul(
            cat_buf.data_ptr(), int(mtp['fc_w']),
            fc_out.data_ptr(), rows, hidden, s,
        )
        fvk.rms_norm(
            fc_out.data_ptr(), int(mtp['input_norm_eff_w']),
            x_norm.data_ptr(), rows, hidden, eps, s,
        )

        ap_5120, sf_5120, _ = self._nvfp4_scratch[(1024, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_norm.data_ptr(), ap_5120.data_ptr(),
            sf_5120.data_ptr(), rows, hidden, s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(mtp['k_proj_packed']),
            k_proj.data_ptr(),
            rows, 4 * 256, hidden,
            sf_5120.data_ptr(), int(mtp['k_proj_sf']),
            float(mtp['k_proj_alpha']),
            s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(mtp['v_proj_packed']),
            v_proj.data_ptr(),
            rows, 4 * 256, hidden,
            sf_5120.data_ptr(), int(mtp['v_proj_sf']),
            float(mtp['v_proj_alpha']),
            s,
        )
        fvk.rms_norm(
            k_proj.view(rows * 4, 256).data_ptr(),
            int(mtp['k_norm_eff_w']),
            k_norm.data_ptr(), rows * 4, 256, eps, s,
        )

        cache_base = int(cache_base_pos)
        cos = self._rope_cos_table[
            pos_start:pos_start + rows].view(rows, self._rope_dim)
        sin = self._rope_sin_table[
            pos_start:pos_start + rows].view(rows, self._rope_dim)
        q_dummy = self._mtp_tail_dummy_q_in[:rows]
        q_dummy_out = self._mtp_tail_dummy_q_out[:rows]
        fvk.qwen36_partial_rope_qk_bf16(
            q_dummy.data_ptr(), k_norm.data_ptr(),
            cos.data_ptr(), sin.data_ptr(),
            q_dummy_out.data_ptr(),
            self._mtp_K_cache[
                cache_base:cache_base + rows].data_ptr(),
            rows, 1, 4, 256, self._rope_dim, s,
        )
        fvk.gpu_copy(
            self._mtp_V_cache[
                cache_base:cache_base + rows].data_ptr(),
            v_proj.data_ptr(), rows * 4 * 256 * 2, s,
        )
        return True
