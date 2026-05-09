"""FlashRT -- PyTorch frontend for Qwen3.6-27B on RTX SM120.

Class name ``Qwen36TorchFrontendRtx`` follows the contract from
``docs/adding_new_model.md`` §0 rule 2.

Phase-1 scope: load the safetensors checkpoint via HF's
AutoModelForCausalLM (with the FP8-dispatch monkey-patch from
qwen35_patch.py applied at module import), wrap it in
``Qwen36Pipeline``, and expose ``set_prompt`` / ``infer`` /
``generate``. No fvk wiring yet -- this commit proves the file
layout, the class contract, and the ``flash_rt.hardware._PIPELINE_MAP``
resolution path.

Phase 2+ will replace ``_load_hf_model`` with a manual safetensors
loader driven by ``WEIGHT_SPEC`` (see ``_qwen36_rtx_spec.py``, added
in Phase 2) that quantizes weights into fvk-compatible layouts.
"""

from __future__ import annotations

import collections
import os
from typing import Any

# Optional HF FP8 dispatch monkey-patch (legacy FP8 path only).
# Set FLASHRT_QWEN36_HF_PATCH to the patch script path to enable; if
# unset or missing, the patch step is skipped silently. The pure-NVFP4
# path (which is the documented v1 surface) does not need this.
_PATCH_PATH = os.environ.get('FLASHRT_QWEN36_HF_PATCH', '')
if _PATCH_PATH and os.path.isfile(_PATCH_PATH):
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        '_flashrt_qwen36_hf_patch', _PATCH_PATH,
    )
    if _spec is not None and _spec.loader is not None:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)


from flash_rt.models.qwen36 import Qwen36Pipeline  # noqa: E402


class Qwen36TorchFrontendRtx:
    """Qwen3.6-27B inference frontend (PyTorch + RTX SM120).

    Phase-1 surface area:
        - ``__init__(checkpoint_path)`` -- loads ckpt + builds Pipeline
        - ``set_prompt(text)``           -- tokenizes for next infer()
        - ``infer()``                    -- single forward, returns logits
        - ``generate(max_new_tokens)``   -- delegates to HF .generate()

    Future surface (frozen to keep tests stable):
        - ``calibrate_with_real_data(prompts)`` -- Phase 2 FP8 calibration
        - ``set_rl_mode(...)``                  -- Phase 6+, mirrors Pi05
        - ``latency_records``                   -- list[float] populated by infer()
    """

    def __init__(self, checkpoint_path: str, *,
                 device: str = 'cuda:0',
                 max_seq: int = 2048,
                 alloc_own_forward_buffers: bool = True,
                 quant: str = 'fp8') -> None:
        """Construct the frontend.

        Args:
          checkpoint_path: HF-style ckpt directory.
          device: cuda device string.
          max_seq: maximum sequence length (KV + scratch sized to this).
          alloc_own_forward_buffers: pre-allocate every per-step buf
            the own-forward / spec decode path consumes. Default True;
            set False only for memory-introspection unit tests.
          quant: weight quantization format.
            * ``'fp8'`` (default): the existing Qwen3.6 FP8 path —
              loads via HF AutoModelForCausalLM, uses
              ``cutlass_sm120_block128_fp8_gemm`` + the fp8 per-token
              activation quantizer. **This is the FP8 60.83 tok/s
              baseline path; nothing about it is touched when this is
              the value.**
            * ``'nvfp4'``: NEW NVFP4 W4A16 path (compressed-tensors
              ``nvfp4-pack-quantized`` ckpt). Skips HF AutoModel
              entirely; loads weights directly from safetensors via
              ``extract_weights_nvfp4``. Uses
              ``fp4_w4a16_gemm_sm120_bf16out`` for quantized linears
              + cuBLAS BF16 matmul for the linear-attn projections
              that this ckpt leaves unquantized. Tokenizer still
              loaded from ``checkpoint_path`` (NVFP4 ckpts ship
              tokenizer files at the same path).
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._user_max_seq = int(max_seq)
        self._tokenizer = None
        self._prompt_ids = None
        self._pipeline: Qwen36Pipeline | None = None
        self.latency_records: list[float] = []

        if quant not in ('fp8', 'nvfp4'):
            raise ValueError(
                f"quant must be 'fp8' or 'nvfp4', got {quant!r}")
        self._quant_format = quant

        # Auto-route: above ``LONG_CTX_THRESHOLD`` the BF16 KV cache
        # alone is too big to fit a 32 GB card alongside the model
        # (16 layers × max_seq × 4 × 256 × 2 bytes for K + V = 32 KB
        # per token, so 32K context = 1 GB just for K, 2 GB for K+V,
        # on top of a ~30 GB baseline). The fix is to route to the
        # TurboQuant (TQ) packed-cache path: BF16 KV is shrunk to a
        # 64-row dummy and persistent KV lives in NVFP4-packed form
        # (~1.83x compression at 1-byte idx, ~5x at bit-pack).
        # Spec decode is *not* yet integrated with TQ (Phase 3D
        # follow-up), so long-ctx mode falls back to single-token
        # decode (~30-40 tok/s eager, vs ~100-130 tok/s spec on
        # short-ctx). This is the documented trade-off and matches
        # the perf grid in docs/qwen36_nvfp4.md §4.
        if quant == 'nvfp4' and self._user_max_seq > self.LONG_CTX_THRESHOLD:
            # Long-ctx mode: allocate BF16 buffers at a small floor
            # (the BF16 KV cache will be shrunk to a 64-row dummy
            # immediately after init, and the per-step bf16 scratches
            # like _h_a/_h_b only ever read [:1] or [:K] rows so they
            # don't need to scale with max_seq). The TQ packed cache
            # then grows KV coverage out to ``user_max_seq``.
            bf16_init_seq = 2048
            self._long_ctx_mode = True
        else:
            bf16_init_seq = self._user_max_seq
            self._long_ctx_mode = False
        # ``self.max_seq`` is the value used to size the BF16 init.
        # ``self._user_max_seq`` is what the user asked for — that's
        # what the TQ packed cache grows to in long-ctx mode, and
        # what gets reported back via ``buffer_summary`` etc.
        self.max_seq = int(bf16_init_seq)

        # Phase 2.3b4 own-forward state (populated by _alloc_buffers).
        self._weights = None         # WeightHandles
        self._bufs: dict | None = None
        self._attn = None            # RtxFlashAttnBackendQwen36

        if quant == 'fp8':
            if self._long_ctx_mode:
                raise NotImplementedError(
                    'long-context auto-route (max_seq > '
                    f'{self.LONG_CTX_THRESHOLD}) is implemented for '
                    "quant='nvfp4' only. The FP8 path has no "
                    'TurboQuant packed-cache integration; pass a '
                    "smaller max_seq or use quant='nvfp4'."
                )
            # Existing FP8 path — completely unchanged behavior.
            self._load_hf_model()
            self._load_mtp_weights()
            if alloc_own_forward_buffers:
                self._alloc_buffers()
        else:
            # NVFP4 path — raw safetensors loader, no HF AutoModel,
            # no MTP for now (MTP head reuse from FP8 ckpt is a
            # follow-up; spec decode requires MTP).
            self._load_nvfp4_path(alloc_own_forward_buffers)
            if self._long_ctx_mode and alloc_own_forward_buffers:
                self._enter_long_ctx_mode()
                # Buffers are sized at the BF16 init seq, but the
                # user-facing max_seq reflects what they asked for —
                # the TQ packed cache covers up to user_max_seq.
                self.max_seq = self._user_max_seq

    # ---------- Phase 1 weight loading (HF path) ----------

    def _load_hf_model(self) -> None:
        """Load the safetensors checkpoint via HF AutoModelForCausalLM.

        Phase 1 only. Phase 2 will replace with a WEIGHT_SPEC-driven
        loader.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # attn_implementation='flash_attention_2' routes the attention
        # path through the vendored FA2 (flash_rt_fa2.so) instead of
        # HF SDPA. Qwen3.6 GQA 6:1 + head_dim=256 + causal — Phase
        # 2.1.a verified FA2 cos = 1.000000 vs SDPA on SM120.
        mdl = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            torch_dtype='auto',
            device_map={'': self.device},
            low_cpu_mem_usage=True,
            attn_implementation='flash_attention_2',
        ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        self._pipeline = Qwen36Pipeline(mdl)

    # ---------- Phase 6 MTP head loader ----------

    def _load_mtp_weights(self) -> None:
        """Load mtp.safetensors directly (HF skips it, no MTP module).

        The MTP head is a 1-layer DeepSeek-V3-style speculative decode
        extension, used by ``forward_mtp_head`` to draft the next token
        from the main model's last hidden state. We populate
        ``self._mtp_tensors`` -- a flat dict keyed by the post-``mtp.``
        suffix (e.g. ``layers.0.self_attn.q_proj.weight``,
        ``norm.weight``, ``fc.weight``) -- with cuda tensors typed:

          * .weight         -> float8_e4m3fn (FP8 projections)
          * .weight_scale_inv -> float32     (block-128 scale)
          * .weight (norms / fc) -> bfloat16

        On checkpoints without an MTP head, sets to None and leaves
        speculative decode disabled.
        """
        import os

        mtp_path = os.path.join(self.checkpoint_path, 'mtp.safetensors')
        if not os.path.exists(mtp_path):
            self._mtp_tensors = None
            return

        import safetensors.torch
        import torch

        raw = safetensors.torch.load_file(mtp_path, device=self.device)
        mtp: dict = {}
        for k, t in raw.items():
            if not k.startswith('mtp.'):
                # Defensive: should always start with mtp. per the
                # checkpoint layout we inspected; raise loudly if not.
                raise RuntimeError(f'unexpected MTP key {k!r}')
            short = k[len('mtp.'):]
            if t.dtype == torch.float8_e4m3fn:
                mtp[short] = t.contiguous()
            elif 'weight_scale_inv' in short:
                mtp[short] = t.to(torch.float32).contiguous()
            else:
                mtp[short] = t.to(torch.bfloat16).contiguous()
        self._mtp_tensors = mtp

    # ---------- Phase 2.3b4 own-forward state ----------

    # Phase 6 D4: maximum query-seq length for S=K decode paths
    # (speculative verify, multi-token batched decode). Originally 8 to
    # support K=5 spec (verify q_seq = K+1 = 6). Bumped to 16 in N6-A4
    # to host the DFlash drafter's block_size=16 verify forward
    # (input layout [prev_token, draft_0..draft_14]). Per-K-row scratches
    # scale linearly: full set is ~12 MB at K=16 (the dominant
    # _K_logits_buf at vocab=248320 is 16*248320*2 = 7.6 MB), tiny vs
    # the 28-30 GB main weight footprint.
    MAX_Q_SEQ: int = 16

    # Per-cache LRU bound on lazily captured CUDA Graphs. Each
    # ``_ensure_*_graph_*`` method captures a graph keyed by cur_pos
    # (or (cur_pos, K), or eff_ctx) the first time that key is used,
    # then replays on every subsequent call with that key. Without a
    # bound, long-running servers that see many distinct shapes grow
    # the cache linearly with the number of distinct cur_pos values
    # observed and eventually OOM (GitHub issue: NVFP4 server VRAM
    # leak on varied prompt lengths). The bound is per-cache, so the
    # total captured-graph budget across all 4 NVFP4 caches is roughly
    # 4 × ``GRAPH_CACHE_MAX``. Tuning notes:
    #   - For chat-style traffic with ``max_seq <= 8K``, the default
    #     of 256 is comfortably above the working set (one bucket per
    #     observed cur_pos value, i.e. 8K positions worst-case but
    #     the hot set is much smaller).
    #   - All graphs allocate from a single shared mempool
    #     (``self._graph_mempool``), so eviction reclaims pool memory
    #     as soon as the evicted graph is GC'd.
    #   - Override at process start via env
    #     ``FLASHRT_QWEN36_GRAPH_CACHE_MAX``; setting it to ``0``
    #     disables the bound (legacy unbounded behaviour, only safe
    #     when the calling pattern caps cur_pos by construction).
    GRAPH_CACHE_MAX: int = int(
        os.environ.get('FLASHRT_QWEN36_GRAPH_CACHE_MAX', '256'))

    # Auto-route threshold (NVFP4 only). At ``max_seq`` above this,
    # the constructor switches into long-ctx mode: BF16 KV buffers
    # are allocated at this threshold (small) and persistent KV is
    # served by the TurboQuant packed cache, which compresses 1.83x
    # at 1-byte idx (~5x at bit-pack). Spec decode is currently only
    # available below the threshold; long-ctx mode does single-token
    # decode (~30-40 tok/s) but supports up to 256 K context on a
    # 32 GB card.
    #
    # Tuning notes:
    #   - 16384 is the default because at max_seq=16K the BF16 KV is
    #     ~1 GB, leaving comfortable headroom on a 32 GB card after
    #     the model and scratches load. At 32 GB BF16 KV is 2 GB and
    #     there is essentially no headroom for capture transients.
    #   - Override via env ``FLASHRT_QWEN36_LONG_CTX_THRESHOLD``.
    #   - Setting to 0 forces every NVFP4 instance into long-ctx
    #     mode (useful if you want consistent long-ctx behaviour
    #     across short and long requests).
    #   - Setting it very high (e.g. 1_000_000) effectively disables
    #     auto-route — useful only on cards with > 32 GB.
    LONG_CTX_THRESHOLD: int = int(
        os.environ.get('FLASHRT_QWEN36_LONG_CTX_THRESHOLD', '16384'))

    # ---- DFlash hidden-tap capture (N6 phase) -------------------------
    # When forward_own_decode_K_nvfp4 is called with tap_buf set, the
    # hidden state after these layer indices is copied into the buffer
    # for the drafter's fc.weight input. Layer ids come from the z-lab
    # Qwen3.6-27B-DFlash config (dflash_config.target_layer_ids).
    _DFLASH_TAP_LAYERS: tuple[int, ...] = (1, 16, 31, 46, 61)
    _DFLASH_TAP_INDEX: dict = {1: 0, 16: 1, 31: 2, 46: 3, 61: 4}

    # Distinct (N_out, K_in) FP8 GEMM shapes used by Qwen3.6 forward.
    # Each entry gets one (qinput, scale, out) scratch triple sized to
    # (max_seq, ...). Shapes deduplicated -- e.g. o_proj and out_proj
    # both consume (5120, 6144), so a single buf serves both.
    _FP8_SHAPES: tuple[tuple[int, int], ...] = (
        (12288, 5120),   # full-attn q_proj (Q + output_gate fused)
        (1024, 5120),    # full-attn k_proj / v_proj (4*256)
        (5120, 6144),    # full-attn o_proj  AND  lin-attn out_proj
        (10240, 5120),   # lin-attn in_proj_qkv (2*K_dim + V_dim)
        (6144, 5120),    # lin-attn in_proj_z (V_dim output gate)
        (17408, 5120),   # MLP gate_proj  AND  MLP up_proj
        (5120, 17408),   # MLP down_proj
    )

    def _alloc_buffers(self) -> None:
        """Allocate every buffer the own-forward path will read/write.

        Called once at __init__ (after HF weights are loaded). All
        per-step Python-side allocations are pushed to load-time so
        the forward path runs at fixed pointers (CUDA Graph eligible
        once Phase 4 lands).

        Total VRAM footprint (max_seq=2048, B=1):
          * hidden ping-pong:        2 * (2048,5120) bf16 = 40 MB
          * FP8 scratch (7 shapes):  ~250 MB worst-case
                                     (down_proj qinput dominates)
          * recurrent state (lin):   48 layers * 48*128*128 bf16 = 75 MB
          * conv state (lin):        48 layers * 10240*3 bf16 = 6 MB
          * KV cache (full-attn):    16 layers * 2048*4*256*2 bytes
                                     = 64 MB (lives in attn backend)
          * logits scratch:          1 * 248320 bf16 = 0.5 MB

        FP8 scratch dominates because down_proj has K=17408
        (qinput = max_seq * K * 1 byte = 35 MB; plus per-token-block
        scale 2048*136*4 = 1 MB; out = max_seq*5120*2 = 20 MB). Across
        7 shapes the worst-case sum is ~250 MB. Acceptable on 32 GB
        5090 (we currently use 28 GB for the FP8 weights).
        """
        if self._pipeline is None:
            raise RuntimeError('_load_hf_model must run before _alloc_buffers')

        import torch
        from flash_rt.frontends.torch._qwen36_rtx_weights import (
            assert_extraction_invariants, extract_mtp_weights,
            extract_weights,
        )

        # Decode is inference-only; mark all weights as not requiring grad
        # so torch.add(out=...), torch.sigmoid(out=...) and the upcoming
        # CUDA Graph capture don't trip on parameter grad tracking.
        for p in self._pipeline.hf.parameters():
            p.requires_grad_(False)
        from flash_rt.hardware.rtx.attn_backend_qwen36 import (
            RtxFlashAttnBackendQwen36,
        )

        # Weight pointers + anchors.
        handles = extract_weights(self._pipeline.hf)
        assert_extraction_invariants(handles)
        self._weights = handles

        # Phase 6: MTP head handles (speculative decode draft model).
        # Optional -- only present if the checkpoint shipped
        # mtp.safetensors and _load_mtp_weights succeeded.
        if self._mtp_tensors is not None:
            handles.ptrs['mtp'] = extract_mtp_weights(
                self._mtp_tensors, handles)
        else:
            handles.ptrs['mtp'] = None

        dims = self._pipeline.DIMS
        device = torch.device(self.device)
        bf16 = torch.bfloat16
        max_seq = self.max_seq
        hidden = dims.hidden
        vocab = dims.vocab_size

        # --- hidden ping-pong ---
        self._h_a = torch.empty(max_seq, hidden, device=device, dtype=bf16)
        self._h_b = torch.empty(max_seq, hidden, device=device, dtype=bf16)

        # --- FP8 scratch per (N, K) shape ---
        # Each shape gets (qinput e4m3, scale fp32, out bf16) sized to
        # max_seq rows. Re-used across the layers that consume that
        # shape -- safe because consecutive same-shape calls within a
        # forward sequence each consume their output before the next
        # call to that shape fires (residual add / norm reads ``out``
        # before the next layer's same-shape FP8 op).
        fp8 = torch.float8_e4m3fn
        fp32 = torch.float32
        self._fp8_scratch: dict[tuple[int, int],
                                 tuple[torch.Tensor, ...]] = {}
        for N, K in self._FP8_SHAPES:
            qinp = torch.empty(max_seq, K, device=device, dtype=fp8)
            sc = torch.empty(max_seq, K // 128, device=device, dtype=fp32)
            out = torch.empty(max_seq, N, device=device, dtype=bf16)
            self._fp8_scratch[(N, K)] = (qinp, sc, out)

        # --- linear-attn recurrent state cache ---
        # (NUM_LIN_LAYERS, B=1, V_HEADS=48, HD_K=128, HD_V=128) bf16.
        # Zero-init: empty cache before prompt.
        self._lin_state = torch.zeros(
            48, 1, 48, 128, 128, device=device, dtype=bf16,
        )
        # --- linear-attn conv state cache ---
        # (NUM_LIN_LAYERS, B=1, conv_dim=10240, k=4) bf16.
        # NOTE: HF cache_utils.LinearAttentionLayer stores conv_states
        # with the last dim equal to the FULL conv_kernel_size (4), not
        # k-1 (3) -- F.pad in modeling_qwen3_5 always produces 4 cols.
        # The kernel reads/writes the first 3 cols (interpreting them
        # as the last k-1 history tokens) and leaves col 3 untouched.
        # We must match HF's layout so the same memory works through
        # both the patched HF path and our own forward.
        self._lin_conv_state = torch.zeros(
            48, 1, 10240, 4, device=device, dtype=bf16,
        )

        # --- linear-attn intra-layer scratches (B=1, S=1) ---
        # conv1d_update writes to a separate buf because the kernel reads
        # x_new while writing out -- must not overlap. Sized to conv_dim.
        self._lin_conv_out = torch.empty(
            1, 10240, device=device, dtype=bf16,
        )
        # recurrent_gated_delta_rule output buffer, (1, 48, 128) bf16.
        self._lin_attn_out = torch.empty(
            1, 48, 128, device=device, dtype=bf16,
        )
        # rms_norm_gated_silu output buffer, (48, 128) bf16.
        self._lin_norm_out = torch.empty(
            48, 128, device=device, dtype=bf16,
        )

        # --- full-attn intra-layer scratches ---
        # q_norm output: HF layout (B*S*H_q, head_dim) -> (24, 256).
        # k_norm output: (B*S*H_kv, head_dim) -> (4, 256).
        self._full_q_norm_out = torch.empty(
            24, 256, device=device, dtype=bf16,
        )
        self._full_k_norm_out = torch.empty(
            4, 256, device=device, dtype=bf16,
        )

        # --- Phase 4.3: SiLU-gate multiply output buffer ---
        # silu(gate) * up writes to this pre-alloc buf via fvk kernel.
        # Replaces the F.silu(g) * up Python composite (2 allocs/call).
        self._mlp_silu_mul_out = torch.empty(
            1, 17408, device=device, dtype=bf16,
        )

        # --- MLP up_proj output (separate from gate_proj output) ---
        # gate_proj and up_proj have the SAME (N, K) = (17408, 5120) but
        # both outputs must be live concurrently for `silu(gate) * up`,
        # so we add one extra (max_seq, 17408) bf16 slot. The gate output
        # reuses _fp8_scratch[(17408, 5120)][2].
        self._mlp_up_out = torch.empty(
            max_seq, 17408, device=device, dtype=bf16,
        )

        # --- Phase 4.2: inline RoPE + index_select broadcast scratches ---
        # full-attn RoPE outputs (in-place rotation, write to these bufs).
        self._full_q_rot = torch.empty(
            1, 1, 24, 256, device=device, dtype=bf16,
        )
        self._full_k_rot = torch.empty(
            1, 1, 4, 256, device=device, dtype=bf16,
        )
        # lin-attn 16->48 broadcast bufs (pre-allocated targets for
        # torch.index_select that replaces repeat_interleave).
        self._lin_q48 = torch.empty(
            1, 48, 128, device=device, dtype=bf16,
        )
        self._lin_k48 = torch.empty(
            1, 48, 128, device=device, dtype=bf16,
        )
        # Pre-built broadcast index [0,0,0, 1,1,1, ..., 15,15,15].
        self._lin_broadcast_idx = (
            torch.arange(48, device=device, dtype=torch.long) // 3
        ).contiguous()
        # Pre-built rotate_half index for partial RoPE (rotary_dim=64):
        # rotate_half(x)[i] = -x[32+i] for i<32, x[i-32] for i>=32.
        # We compute via index_select with idx=[32..63, 0..31] then
        # negate the first half. _rope_rotate_idx covers the half-flip.
        rope_dim = 64
        half = rope_dim // 2
        idx_lo = torch.arange(half, rope_dim, device=device, dtype=torch.long)
        idx_hi = torch.arange(0, half, device=device, dtype=torch.long)
        self._rope_rotate_idx = torch.cat([idx_lo, idx_hi]).contiguous()
        # Pre-allocated rotate_half scratch for q (24 heads) and k (4 heads).
        self._full_rope_tmp_q = torch.empty(
            1, 1, 24, rope_dim, device=device, dtype=bf16,
        )
        self._full_rope_tmp_k = torch.empty(
            1, 1, 4, rope_dim, device=device, dtype=bf16,
        )

        # --- Phase 4.1: in-place residual + sigmoid + cast scratches ---
        # Layer-output ping-pong: layer L writes to _layer_out[L % 2];
        # next layer reads it. Pre-allocated so torch.add(out=...) and
        # torch.sigmoid(out=...) can run with zero per-step allocs.
        self._layer_out_a = torch.empty(
            1, 1, hidden, device=device, dtype=bf16,
        )
        self._layer_out_b = torch.empty(
            1, 1, hidden, device=device, dtype=bf16,
        )
        # Mid-layer residual buffer (h_post between attn and mlp residuals).
        self._res_mid = torch.empty(
            1, 1, hidden, device=device, dtype=bf16,
        )
        # Linear-attn intra-step small scratches (per call, zero alloc).
        # _lin_a_vec / _lin_b_vec: pre-alloc targets for the new
        # bf16_matvec_qwen36_bf16 kernel that replaced F.linear in
        # in_proj_a / in_proj_b (Phase 4.4 stream-invariant GEMM).
        # A2c-2: backing _lin_ab_vec is one contiguous (1, 96) buffer so
        # the NVFP4 lin layer can fuse in_proj_a + in_proj_b into one
        # bf16_matvec call (N=96, K=5120) using the concatenated
        # in_proj_ab_w. Saves ~6 us per lin layer (probe-validated).
        self._lin_ab_vec = torch.empty(1, 96, device=device, dtype=bf16)
        self._lin_a_vec = self._lin_ab_vec[:, :48]
        self._lin_b_vec = self._lin_ab_vec[:, 48:]
        self._lin_beta = torch.empty(1, 48, device=device, dtype=bf16)
        self._lin_a_f32 = torch.empty(1, 48, device=device, dtype=fp32)
        self._lin_g_f32 = torch.empty(1, 48, device=device, dtype=fp32)
        self._lin_g_bf = torch.empty(1, 48, device=device, dtype=bf16)
        # Phase 4.4 step 2: pre-alloc buf for manual softplus (exp+log1p)
        # that replaces F.softplus's per-call allocation.
        self._lin_sp_buf = torch.empty(1, 48, device=device, dtype=fp32)
        # Full-attn intra-step scratches (output gate path).
        self._full_gate_sig = torch.empty(
            1, 1, 24 * 256, device=device, dtype=bf16,
        )
        self._full_gated = torch.empty(
            1, 1, 24 * 256, device=device, dtype=bf16,
        )

        # --- logits scratch (decode -- one row at a time) ---
        self._logits_buf = torch.empty(1, vocab, device=device, dtype=bf16)

        # Phase 6 D2: stable buffer for the post-64-layer / pre-final-
        # norm hidden state so forward_mtp_head can consume it after
        # forward_own_decode (or its captured graph replay).
        self._last_hidden_buf = torch.empty(
            1, 1, hidden, device=device, dtype=bf16,
        )

        # --- attention backend (owns full-attn KV cache) ---
        # max_q_seq = MAX_Q_SEQ so the S=1 path slices [:, :1] and the
        # S=K decode (Phase 6 D4) slices [:, :K]. Q_buf / O_buf / lse
        # scratches are sized for the larger K — incremental memory is
        # ~ MAX_Q_SEQ * 24 heads * 256 head_dim * 2 bytes ≈ 50 KB.
        self._attn = RtxFlashAttnBackendQwen36(
            max_seq=max_seq, max_q_seq=self.MAX_Q_SEQ, dtype=bf16,
        )

        # --- Phase 4.4: dedicated CUDA stream for graph capture+replay ---
        # Required because:
        #   1. fvk kernels launched on stream=0 (legacy default) are
        #      EXCLUDED from CUDA Graph capture ("CUDA Graph is empty").
        #   2. cuBLASLt picks per-stream-optimal GEMM algorithms; if
        #      warmup, capture, and replay use different streams, the
        #      replayed bf16 reductions diverge from the reference.
        # Solution: pin a single non-default stream and use it for
        # warmup + capture + replay. Verified in
        # internal-tests/rtx_qwen36_graph_stream_test.py: rms_norm
        # captured this way produces cos=1.0 vs non-graph baseline.
        self._graph_stream = torch.cuda.Stream(device=device)

        # Phase 4.4 step 6: per-cur_pos lazy CUDA Graph cache for
        # forward_own_decode. Each captured graph baked the runtime
        # int kv_seq for FA2 + cur_pos-specific cos/sin slice address,
        # so different cur_pos values need different graphs. We capture
        # on first encounter and replay thereafter.
        self._static_token_id = torch.zeros(
            1, 1, device=device, dtype=torch.long,
        )
        # Shared mempool for every captured graph in this instance.
        # Without this, ``torch.cuda.graph(g, stream=gs)`` defaults to
        # a *private* mempool per graph; with N captured graphs the
        # per-graph workspace overhead is paid N times instead of once.
        self._graph_mempool = torch.cuda.graph_pool_handle()
        self._captured_graphs: collections.OrderedDict[
            int, torch.cuda.CUDAGraph] = collections.OrderedDict()

        # Phase 6 D4-9: per-cur_pos CUDA Graph cache for the spec
        # decode S=K+1 verify forward. Uses static input buffers so
        # the spec loop can copy_ token_ids / cos / sin into stable
        # pointers and replay. Graph captures the entire 64-layer
        # forward + lm_head for K_verify rows.
        self._verify_static_tokens = torch.zeros(
            1, self.MAX_Q_SEQ, device=device, dtype=torch.long,
        )
        self._verify_static_cos = torch.empty(
            1, self.MAX_Q_SEQ, 64, device=device, dtype=bf16,
        )
        self._verify_static_sin = torch.empty(
            1, self.MAX_Q_SEQ, 64, device=device, dtype=bf16,
        )
        self._captured_verify_graphs: collections.OrderedDict[
            tuple[int, int], torch.cuda.CUDAGraph,
        ] = collections.OrderedDict()

        # Phase 6 D4-10: dedicated stream for state snapshotting in
        # the spec loop. The clones (lin_state 75 MB + KV partial)
        # take ~1 ms; running them on _snap_stream lets them overlap
        # with the MTP chain on the default stream.
        self._snap_stream = torch.cuda.Stream(device=device)

        # Phase 6 D4-15: pre-allocated snap buffers for the spec
        # loop. Replaces per-cycle .clone() (which goes through the
        # caching allocator and adds CPU-side overhead) with
        # in-place .copy_() into stable pointers. ~80 MB extra one-
        # time VRAM but saves ~0.5-1 ms / cycle of allocator chatter.
        self._snap_lin_buf = torch.empty_like(self._lin_state)
        self._snap_conv_buf = torch.empty_like(self._lin_conv_state)
        # Partial K_cache snap: only sized for max_K_verify rows
        # (= MAX_Q_SEQ at the upper bound).
        self._snap_K_buf = torch.empty(
            16, self.MAX_Q_SEQ, 4, 256, device=device, dtype=bf16,
        )
        self._snap_V_buf = torch.empty_like(self._snap_K_buf)

        # Phase 6 D4-13: per-mtp_pos CUDA Graph cache for the spec
        # loop's MTP chain. Each chain step at position cur_pos+k is
        # captured as its own graph. Static input bufs let the chain
        # copy_ prev_h / prev_token before each replay.
        if self._weights.ptrs.get('mtp') is not None:
            self._mtp_static_prev_h = torch.empty(
                1, 1, hidden, device=device, dtype=bf16,
            )
            self._mtp_static_prev_token = torch.zeros(
                1, 1, device=device, dtype=torch.long,
            )
            self._captured_mtp_graphs: collections.OrderedDict[
                int, torch.cuda.CUDAGraph,
            ] = collections.OrderedDict()

        # ---------- Phase 6 D4: S=K decode scratches ----------
        # Mirror the S=1 buffers but sized for max_q_seq=MAX_Q_SEQ so
        # the S=K forward path can write K rows in parallel where the
        # underlying kernels support it. The S=1 forward continues to
        # use the original (1, 1, ...) buffers — bit-identical.
        Kmax = self.MAX_Q_SEQ
        # full-attn intra-layer S=K scratches
        self._K_full_q_norm_out = torch.empty(
            Kmax * 24, 256, device=device, dtype=bf16,
        )
        self._K_full_k_norm_out = torch.empty(
            Kmax * 4, 256, device=device, dtype=bf16,
        )
        self._K_full_q_rot = torch.empty(
            1, Kmax, 24, 256, device=device, dtype=bf16,
        )
        self._K_full_k_rot = torch.empty(
            1, Kmax, 4, 256, device=device, dtype=bf16,
        )
        self._K_full_rope_tmp_q = torch.empty(
            1, Kmax, 24, 64, device=device, dtype=bf16,
        )
        self._K_full_rope_tmp_k = torch.empty(
            1, Kmax, 4, 64, device=device, dtype=bf16,
        )
        self._K_full_gate_sig = torch.empty(
            1, Kmax, 24 * 256, device=device, dtype=bf16,
        )
        self._K_full_gated = torch.empty(
            1, Kmax, 24 * 256, device=device, dtype=bf16,
        )
        # layer-output ping-pong + residual
        self._K_layer_out_a = torch.empty(
            1, Kmax, hidden, device=device, dtype=bf16,
        )
        self._K_layer_out_b = torch.empty(
            1, Kmax, hidden, device=device, dtype=bf16,
        )
        self._K_res_mid = torch.empty(
            1, Kmax, hidden, device=device, dtype=bf16,
        )
        # linear-attn small per-row scratches (K rows × 48 heads)
        self._K_lin_a_vec = torch.empty(
            Kmax, 48, device=device, dtype=bf16,
        )
        self._K_lin_b_vec = torch.empty(
            Kmax, 48, device=device, dtype=bf16,
        )
        self._K_lin_beta = torch.empty(
            Kmax, 48, device=device, dtype=bf16,
        )
        self._K_lin_a_f32 = torch.empty(
            Kmax, 48, device=device, dtype=fp32,
        )
        self._K_lin_g_f32 = torch.empty(
            Kmax, 48, device=device, dtype=fp32,
        )
        self._K_lin_g_bf = torch.empty(
            Kmax, 48, device=device, dtype=bf16,
        )
        self._K_lin_sp_buf = torch.empty(
            Kmax, 48, device=device, dtype=fp32,
        )
        # MLP S=K scratches
        self._K_mlp_silu_mul_out = torch.empty(
            Kmax, 17408, device=device, dtype=bf16,
        )
        # Linear-attn S=K accumulators: K rows of recurrent attn out
        # and K rows of rms_norm_gated_silu out (each (48, 128) bf16
        # per row; flattened the same as (K, 6144)).
        self._K_lin_attn_out = torch.empty(
            Kmax, 48, 128, device=device, dtype=bf16,
        )
        self._K_lin_norm_out = torch.empty(
            Kmax, 48, 128, device=device, dtype=bf16,
        )
        # Conv1d update output staging: K rows of (1, 10240) bf16 so
        # the K conv1d_update kernel calls each write into a stable
        # row, and the downstream split+recurrent loop reads them.
        self._K_lin_conv_out = torch.empty(
            Kmax, 10240, device=device, dtype=bf16,
        )
        # logits at K rows
        self._K_logits_buf = torch.empty(
            Kmax, vocab, device=device, dtype=bf16,
        )
        # last hidden at K rows (for spec loop chaining)
        self._K_last_hidden_buf = torch.empty(
            1, Kmax, hidden, device=device, dtype=bf16,
        )

        # ---------- Phase 6 D2: MTP head buffers ----------
        # Only allocated if mtp.safetensors was loaded.
        if self._weights.ptrs.get('mtp') is not None:
            # MTP's own KV cache (1 attn layer, GQA 24Q/4KV identical
            # to main full-attn).
            self._mtp_K_cache = torch.empty(
                max_seq, 4, 256, device=device, dtype=bf16,
            )
            self._mtp_V_cache = torch.empty_like(self._mtp_K_cache)
            # FA2 Q/O buffers separate from main backend so we can
            # call _fa2_fwd directly with our own pointers.
            self._mtp_Q_buf = torch.empty(
                1, 1, 24, 256, device=device, dtype=bf16,
            )
            self._mtp_O_buf = torch.empty_like(self._mtp_Q_buf)
            # FA2 splitkv scratches (mirror attn_backend layout).
            sq_rounded = 128  # max_q_seq=1 rounded up to 128
            self._mtp_lse_buf = torch.empty(
                1, 24, sq_rounded, device=device, dtype=fp32,
            )
            n_splits = min(128, (max_seq + 63) // 64)
            self._mtp_n_splits = n_splits
            self._mtp_lse_accum = torch.empty(
                n_splits, 1, 24, 1, device=device, dtype=fp32,
            )
            self._mtp_o_accum = torch.empty(
                n_splits, 1, 24, 1, 256, device=device, dtype=fp32,
            )
            # MTP head intra-step scratches (pre-fc + concat + fc out
            # + final layer out + logits). The full-attn layer's own
            # internal scratches (_full_q_norm_out, _full_q_rot,
            # _full_gate_sig, _full_gated, _res_mid, _h_b, _mlp_*)
            # are reused — MTP runs sequentially after main forward,
            # so those buffers are free at MTP time.
            self._mtp_h_norm_buf = torch.empty(
                1, 1, 5120, device=device, dtype=bf16,
            )
            self._mtp_e_norm_buf = torch.empty(
                1, 1, 5120, device=device, dtype=bf16,
            )
            self._mtp_cat_buf = torch.empty(
                1, 1, 10240, device=device, dtype=bf16,
            )
            self._mtp_fc_out_buf = torch.empty(
                1, 1, 5120, device=device, dtype=bf16,
            )
            self._mtp_layer_out_buf = torch.empty(
                1, 1, 5120, device=device, dtype=bf16,
            )
            self._mtp_logits_buf = torch.empty(
                1, vocab, device=device, dtype=bf16,
            )

        # Bundle ptrs for forward consumption (forward sees only ints).
        self._bufs = {
            'h_a': int(self._h_a.data_ptr()),
            'h_b': int(self._h_b.data_ptr()),
            'logits': int(self._logits_buf.data_ptr()),
            'lin_state_base': int(self._lin_state.data_ptr()),
            'lin_state_layer_stride_bytes': int(
                self._lin_state.stride(0) * self._lin_state.element_size()
            ),
            'lin_conv_state_base': int(self._lin_conv_state.data_ptr()),
            'lin_conv_state_layer_stride_bytes': int(
                self._lin_conv_state.stride(0)
                * self._lin_conv_state.element_size()
            ),
            'h_row_stride_elts': int(self._h_a.stride(0)),
            'fp8_scratch': {
                shape: (
                    int(qinp.data_ptr()),
                    int(sc.data_ptr()),
                    int(out.data_ptr()),
                )
                for shape, (qinp, sc, out) in self._fp8_scratch.items()
            },
        }

    # ---------- NVFP4 path (N4) ----------

    # Distinct (N_out, K_in) NVFP4-quantized GEMM shapes used by the
    # Qwen3.6 NVFP4 ckpt (compressed-tensors nvfp4-pack-quantized).
    # Only full-attn projections + MLP are quantized in this ckpt;
    # linear-attn in_proj_qkv/z + out_proj stay BF16 and use a separate
    # cuBLAS BF16 path with no NVFP4 scratch needed.
    _NVFP4_SHAPES: tuple[tuple[int, int], ...] = (
        (12288, 5120),   # full-attn q_proj (Q + output_gate fused)
        (1024, 5120),    # full-attn k_proj / v_proj
        (5120, 6144),    # full-attn o_proj  AND  lin-attn out_proj
        (17408, 5120),   # MLP gate_proj  AND  MLP up_proj
        (5120, 17408),   # MLP down_proj
        # G7: lin-attn projections quantized at load time
        (10240, 5120),   # lin-attn in_proj_qkv (was BF16 in ckpt)
        (6144, 5120),    # lin-attn in_proj_z (was BF16 in ckpt)
    )

    def _load_nvfp4_path(self, alloc_own_forward_buffers: bool) -> None:
        """NVFP4 ckpt path: tokenizer + raw weights + NVFP4 scratch.

        Replaces the (HF AutoModel + MTP + FP8 alloc_buffers) sequence.
        Stays light per the framework principle — no transformers
        AutoModel, no compressed_tensors, no HF runtime model object.

        MTP head: optional. The NVFP4 ckpt itself has no MTP file, so
        we borrow mtp.safetensors from a separate FP8 ckpt directory
        (env ``FLASHRT_QWEN36_MTP_CKPT_DIR`` — unset disables spec
        decode) and convert FP8 → BF16 → NVFP4 swizzled at load time.
        Result: pure-NVFP4 spec decode (no FP8 mixing in the hot path).
        """
        import os

        import torch

        from flash_rt import flash_rt_kernels as fvk
        from flash_rt.frontends.torch._qwen36_rtx_nvfp4_weights import (
            assert_extraction_invariants_nvfp4,
            extract_mtp_weights_nvfp4,
            extract_weights_nvfp4,
        )

        # Tokenizer — NVFP4 ckpts ship tokenizer.json + chat_template
        # at the ckpt path (verified prithivMLmods/Qwen3.6-27B-NVFP4).
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)

        # Raw NVFP4 weights → handles (compatible schema with FP8 path
        # but with *_packed/_sf/_alpha keys for quantized linears).
        handles = extract_weights_nvfp4(
            self.checkpoint_path, fvk, device=self.device)
        assert_extraction_invariants_nvfp4(handles)
        self._weights = handles

        # MTP head from FP8 ckpt → NVFP4 swizzled (pure-NVFP4 spec).
        # The compressed-tensors NVFP4 ckpt has no MTP module; we pull
        # it from a paired FP8 ckpt (Qwen3.6-Next-27B-FP8) via the
        # FLASHRT_QWEN36_MTP_CKPT_DIR env var. If unset, MTP is None
        # and speculative decode is unavailable (pure-decode still works).
        mtp_dir = os.environ.get('FLASHRT_QWEN36_MTP_CKPT_DIR')
        mtp_path = (os.path.join(mtp_dir, 'mtp.safetensors')
                    if mtp_dir else None)
        if mtp_path is not None and os.path.exists(mtp_path):
            import safetensors.torch
            raw = safetensors.torch.load_file(mtp_path, device=self.device)
            mtp_dict: dict = {}
            for k, t in raw.items():
                if not k.startswith('mtp.'):
                    raise RuntimeError(f'unexpected MTP key {k!r}')
                short = k[len('mtp.'):]
                if t.dtype == torch.float8_e4m3fn:
                    mtp_dict[short] = t.contiguous()
                elif 'weight_scale_inv' in short:
                    mtp_dict[short] = t.to(torch.float32).contiguous()
                else:
                    mtp_dict[short] = t.to(torch.bfloat16).contiguous()
            self._mtp_tensors = mtp_dict
            handles.ptrs['mtp'] = extract_mtp_weights_nvfp4(
                mtp_dict, handles, fvk, device=self.device)
        else:
            self._mtp_tensors = None
            handles.ptrs['mtp'] = None
        # DFlash unsupported on this path.
        handles.ptrs['dflash'] = None

        # Source-agnostic config namespace (FP8 path reads from
        # self._pipeline.hf.config; NVFP4 path doesn't have HF model).
        # Mirror the fields the forward methods read.
        self._cfg = {
            'rms_norm_eps': 1e-6,           # Qwen3.5/3.6 standard
            'head_dim': 256,                # full-attn head_dim (target)
            'hidden_size': handles.ptrs['hidden'],
            'vocab_size': handles.ptrs['vocab_size'],
            'num_hidden_layers': handles.ptrs['num_layers'],
            'layer_types': handles.ptrs['layer_types'],
            'partial_rotary_factor': 0.25,  # Qwen3.5/3.6 standard
            'rope_theta': 10_000_000.0,     # Qwen3.6 standard
        }

        if alloc_own_forward_buffers:
            self._alloc_buffers_nvfp4()

    def _alloc_buffers_nvfp4(self) -> None:
        """Pre-allocate every NVFP4 forward buffer at fixed pointers.

        Mirrors the FP8 ``_alloc_buffers`` but with NVFP4-shaped
        scratches:
          * activation packed (M, K/2) u8 per (N, K) shape
          * activation SF swizzled per (M, K) shape
          * output BF16 (M, N) per (N, K) shape

        For the 3 BF16 (unquantized) linear-attn projections in this
        ckpt (in_proj_qkv / in_proj_z / out_proj) we just need a BF16
        output scratch; the input is already BF16 (no quantization).
        """
        import torch

        from flash_rt.hardware.rtx.attn_backend_qwen36 import (
            RtxFlashAttnBackendQwen36,
        )

        device = torch.device(self.device)
        bf16 = torch.bfloat16
        max_seq = self.max_seq
        hidden = self._cfg['hidden_size']
        vocab = self._cfg['vocab_size']

        # Hidden ping-pong (same shape as FP8 path).
        self._h_a = torch.empty(max_seq, hidden, device=device, dtype=bf16)
        self._h_b = torch.empty(max_seq, hidden, device=device, dtype=bf16)

        # NVFP4 scratch per (N, K) shape:
        #   act_packed  (max_seq, K/2)         u8
        #   act_sf_swz  big enough for (max_seq, K) swizzled SF
        #   out         (max_seq, N)           bf16
        u8 = torch.uint8

        def _swz_bytes(rows: int, cols: int) -> int:
            n_blocks = cols // 16
            n_row_super = (rows + 127) // 128
            n_col_super = (n_blocks + 3) // 4
            return n_row_super * n_col_super * 512

        self._nvfp4_scratch: dict[tuple[int, int],
                                  tuple[torch.Tensor, ...]] = {}
        for N, K in self._NVFP4_SHAPES:
            ap = torch.empty(max_seq, K // 2, device=device, dtype=u8)
            sf = torch.zeros(_swz_bytes(max_seq, K), device=device, dtype=u8)
            out = torch.empty(max_seq, N, device=device, dtype=bf16)
            self._nvfp4_scratch[(N, K)] = (ap, sf, out)

        # BF16 scratch for unquantized linear-attn projections. Inputs
        # are already BF16 (no quant); we just need a strided output.
        # in_proj_qkv: (max_seq, 10240), in_proj_z: (max_seq, 6144),
        # out_proj: (max_seq, 5120).
        self._nvfp4_bf16_out: dict[int, torch.Tensor] = {}
        for N in (10240, 6144, 5120):
            self._nvfp4_bf16_out[N] = torch.empty(
                max_seq, N, device=device, dtype=bf16)

        # Linear-attn recurrent state cache + conv state — same shapes
        # as FP8 path (kernel ABI identical).
        self._lin_state = torch.zeros(
            48, 1, 48, 128, 128, device=device, dtype=bf16)
        self._lin_conv_state = torch.zeros(
            48, 1, 10240, 4, device=device, dtype=bf16)
        self._lin_conv_out = torch.empty(1, 10240, device=device, dtype=bf16)
        self._lin_attn_out = torch.empty(
            1, 48, 128, device=device, dtype=bf16)
        self._lin_norm_out = torch.empty(48, 128, device=device, dtype=bf16)
        # A2c-2: contiguous (1, 96) backing buffer for fused dual matvec.
        self._lin_ab_vec = torch.empty(1, 96, device=device, dtype=bf16)
        self._lin_a_vec = self._lin_ab_vec[:, :48]
        self._lin_b_vec = self._lin_ab_vec[:, 48:]
        self._lin_beta = torch.empty(1, 48, device=device, dtype=bf16)
        fp32 = torch.float32
        self._lin_a_f32 = torch.empty(1, 48, device=device, dtype=fp32)
        self._lin_g_f32 = torch.empty(1, 48, device=device, dtype=fp32)
        self._lin_g_bf = torch.empty(1, 48, device=device, dtype=bf16)
        self._lin_sp_buf = torch.empty(1, 48, device=device, dtype=fp32)
        # Q/K head broadcast (16 → 48) via index_select.
        self._lin_q48 = torch.empty(
            1, 48, 128, device=device, dtype=bf16)
        self._lin_k48 = torch.empty(
            1, 48, 128, device=device, dtype=bf16)
        self._lin_broadcast_idx = torch.arange(
            16, device=device, dtype=torch.long).repeat_interleave(3)

        # MLP scratch (silu(gate)*up output).
        self._mlp_silu_mul_out = torch.empty(
            1, 17408, device=device, dtype=bf16)
        # Separate up output buf (gate scratch comes from NVFP4 scratch
        # of (17408, 5120)).
        self._mlp_up_out = torch.empty(
            self.max_seq, 17408, device=device, dtype=bf16)

        # Layer-output ping-pong + intermediate residual.
        self._layer_out_a = torch.empty(
            1, 1, hidden, device=device, dtype=bf16)
        self._layer_out_b = torch.empty(
            1, 1, hidden, device=device, dtype=bf16)
        self._res_mid = torch.empty(
            1, 1, hidden, device=device, dtype=bf16)

        # Full-attn intra-layer scratches (same as FP8).
        self._full_q_norm_out = torch.empty(
            24, 256, device=device, dtype=bf16)
        self._full_k_norm_out = torch.empty(4, 256, device=device, dtype=bf16)
        # Full-attn RoPE scratch + rotate-half index (Phase 4.2 pattern).
        self._full_q_rot = torch.empty(
            1, 1, 24, 256, device=device, dtype=bf16)
        self._full_k_rot = torch.empty(
            1, 1, 4, 256, device=device, dtype=bf16)
        rope_dim = 64
        half = rope_dim // 2
        idx_lo = torch.arange(half, rope_dim, device=device, dtype=torch.long)
        idx_hi = torch.arange(0, half, device=device, dtype=torch.long)
        self._rope_rotate_idx = torch.cat([idx_lo, idx_hi]).contiguous()
        self._full_rope_tmp_q = torch.empty(
            1, 1, 24, rope_dim, device=device, dtype=bf16)
        self._full_rope_tmp_k = torch.empty(
            1, 1, 4, rope_dim, device=device, dtype=bf16)
        # Full-attn output gate scratch.
        self._full_gate_sig = torch.empty(
            1, 1, 24 * 256, device=device, dtype=bf16)
        self._full_gated = torch.empty(
            1, 1, 24 * 256, device=device, dtype=bf16)

        # Logits + last hidden snapshot.
        self._logits_buf = torch.empty(1, vocab, device=device, dtype=bf16)
        self._last_hidden_buf = torch.empty(
            1, 1, hidden, device=device, dtype=bf16)

        # ---------- N5-stage3: S=K decode scratches (NVFP4 path) ----------
        # Mirror the FP8 path's _K_* buffers (lines ~520-609 of the FP8
        # _alloc_buffers). These are format-agnostic (BF16 / FP32) — same
        # shapes work for the NVFP4 verify path because only the GEMM
        # kernel changes; everything else (rms_norm, conv1d_update,
        # FLA chunk, gated-silu, residuals, layernorm) is identical ABI.
        Kmax = self.MAX_Q_SEQ
        fp32 = torch.float32
        # full-attn intra-layer S=K scratches
        self._K_full_q_norm_out = torch.empty(
            Kmax * 24, 256, device=device, dtype=bf16)
        self._K_full_k_norm_out = torch.empty(
            Kmax * 4, 256, device=device, dtype=bf16)
        self._K_full_q_rot = torch.empty(
            1, Kmax, 24, 256, device=device, dtype=bf16)
        self._K_full_k_rot = torch.empty(
            1, Kmax, 4, 256, device=device, dtype=bf16)
        self._K_full_rope_tmp_q = torch.empty(
            1, Kmax, 24, 64, device=device, dtype=bf16)
        self._K_full_rope_tmp_k = torch.empty(
            1, Kmax, 4, 64, device=device, dtype=bf16)
        self._K_full_gate_sig = torch.empty(
            1, Kmax, 24 * 256, device=device, dtype=bf16)
        self._K_full_gated = torch.empty(
            1, Kmax, 24 * 256, device=device, dtype=bf16)
        # layer-output ping-pong + residual at K rows
        self._K_layer_out_a = torch.empty(
            1, Kmax, hidden, device=device, dtype=bf16)
        self._K_layer_out_b = torch.empty(
            1, Kmax, hidden, device=device, dtype=bf16)
        self._K_res_mid = torch.empty(
            1, Kmax, hidden, device=device, dtype=bf16)
        # linear-attn small per-row scratches (K rows × 48 heads)
        self._K_lin_a_vec = torch.empty(Kmax, 48, device=device, dtype=bf16)
        self._K_lin_b_vec = torch.empty(Kmax, 48, device=device, dtype=bf16)
        self._K_lin_beta = torch.empty(Kmax, 48, device=device, dtype=bf16)
        self._K_lin_a_f32 = torch.empty(Kmax, 48, device=device, dtype=fp32)
        self._K_lin_g_f32 = torch.empty(Kmax, 48, device=device, dtype=fp32)
        self._K_lin_g_bf = torch.empty(Kmax, 48, device=device, dtype=bf16)
        self._K_lin_sp_buf = torch.empty(Kmax, 48, device=device, dtype=fp32)
        # MLP S=K scratch (silu(gate)*up).
        self._K_mlp_silu_mul_out = torch.empty(
            Kmax, 17408, device=device, dtype=bf16)
        # Linear-attn S=K accumulators (each (48, 128) per row).
        self._K_lin_attn_out = torch.empty(
            Kmax, 48, 128, device=device, dtype=bf16)
        self._K_lin_norm_out = torch.empty(
            Kmax, 48, 128, device=device, dtype=bf16)
        # Conv1d update output staging: K rows of (1, 10240).
        self._K_lin_conv_out = torch.empty(
            Kmax, 10240, device=device, dtype=bf16)
        # logits at K rows + last hidden at K rows (spec chaining).
        self._K_logits_buf = torch.empty(
            Kmax, vocab, device=device, dtype=bf16)
        self._K_last_hidden_buf = torch.empty(
            1, Kmax, hidden, device=device, dtype=bf16)

        # ---------- N5-stage6: spec-decode snap buffers (NVFP4) ----------
        # Pre-allocated snap buffers + dedicated stream so the partial-
        # accept state recovery does in-place .copy_() into stable
        # pointers (no per-cycle allocator). Mirrors FP8 path lines
        # ~485-504. Snap covers lin_state (full) + lin_conv_state (full)
        # + K/V cache rows [cur_pos:cur_pos+K+1] (partial, sized for
        # MAX_Q_SEQ which caps K+1).
        self._snap_stream = torch.cuda.Stream(device=device)
        self._snap_lin_buf = torch.empty_like(self._lin_state)
        self._snap_conv_buf = torch.empty_like(self._lin_conv_state)
        self._snap_K_buf = torch.empty(
            16, self.MAX_Q_SEQ, 4, 256, device=device, dtype=bf16)
        self._snap_V_buf = torch.empty_like(self._snap_K_buf)

        # ---------- A1'-S0 per-step state save (recovery elimination) ----
        # Replaces the post-partial-accept restore+recovery-forward (~21
        # ms) with a constant-time copy from a per-step state checkpoint
        # written DURING the verify K-iter recurrent loop. K_save_max
        # is the lin-function q_seq dim (= spec K + 1). Sized for
        # exploration up to spec K=7 (q_seq=8): buffer is one-time and
        # the per-element cost is small.
        K_save_max = 8
        # (K_save_max, 48 lin layers, 1, 48 v_heads, head_k=128, head_v=128)
        # 8 × 48 × 1 × 48 × 128 × 128 × 2 = 600 MB.
        self._K_save_max = K_save_max
        self._K_lin_state_per_step = torch.empty(
            K_save_max, *self._lin_state.shape,
            device=device, dtype=bf16)
        # (K_save_max, 48 lin layers, 1, conv_dim=10240, k=4)
        # 6 × 48 × 1 × 10240 × 4 × 2 = 24 MB.
        self._K_lin_conv_state_per_step = torch.empty(
            K_save_max, *self._lin_conv_state.shape,
            device=device, dtype=bf16)
        # K-iter recurrent broadcast + output scratch (K-batched analogues
        # of _lin_q48 / _lin_k48 / _lin_attn_out used in S=1 path).
        # Sized for MAX_Q_SEQ rows (= verify q_seq cap).
        self._K_lin_q48 = torch.empty(
            self.MAX_Q_SEQ, 48, 128, device=device, dtype=bf16)
        self._K_lin_k48 = torch.empty(
            self.MAX_Q_SEQ, 48, 128, device=device, dtype=bf16)
        self._K_lin_attn_out = torch.empty(
            self.MAX_Q_SEQ, 48, 128, device=device, dtype=bf16)

        # Per-position pre-final-norm hidden cache, populated during
        # NVFP4 prefill so MTP prefill (steps 1..prompt_len) can read
        # the correct h_main_{p-1} as input — replaces FP8 path's
        # output_hidden_states from HF AutoModel.
        self._prefill_h_cache = torch.empty(
            max_seq, hidden, device=device, dtype=bf16)

        # FA2 backend for full-attention layers (same ctor as FP8 path).
        self._attn = RtxFlashAttnBackendQwen36(
            max_seq=max_seq, max_q_seq=self.MAX_Q_SEQ, dtype=bf16,
        )

        # CUDA Graph capture infrastructure (mirrors FP8 path).
        # Dedicated non-default stream so warmup / capture / replay all
        # use the same GEMM-algorithm-selection context (bf16 matvec
        # + cuBLASLt picks per-stream algos that diverge across streams).
        self._graph_stream = torch.cuda.Stream(device=device)
        # Static input buffer for the captured graph (pointer fixed for
        # capture). Per-cur_pos graph cache.
        self._static_token_id = torch.zeros(
            1, 1, device=device, dtype=torch.long)
        # Shared mempool for every captured graph in this instance.
        # Without this, ``torch.cuda.graph(g, stream=gs)`` defaults to
        # a *private* mempool per graph; with N captured graphs the
        # per-graph workspace overhead is paid N times instead of once.
        self._graph_mempool = torch.cuda.graph_pool_handle()
        self._captured_graphs: collections.OrderedDict[
            int, torch.cuda.CUDAGraph] = collections.OrderedDict()

        # Stage-7 G1: verify-forward graph cache + static input buffers.
        # The spec loop copies token_ids / cos / sin into these pointers
        # and replays the captured graph for forward_own_decode_K_nvfp4
        # at fixed (cur_pos, K). Each (cur_pos, K) pair gets its own
        # graph because FA2 bakes kv_seq into the captured kernel calls.
        self._verify_static_tokens = torch.zeros(
            1, self.MAX_Q_SEQ, device=device, dtype=torch.long)
        self._verify_static_cos = torch.empty(
            1, self.MAX_Q_SEQ, 64, device=device, dtype=bf16)
        self._verify_static_sin = torch.empty(
            1, self.MAX_Q_SEQ, 64, device=device, dtype=bf16)
        self._captured_verify_graphs: collections.OrderedDict[
            tuple[int, int], torch.cuda.CUDAGraph,
        ] = collections.OrderedDict()

        # MTP scratches (only when MTP is loaded). Mirror FP8 path's
        # MTP buffers — these are BF16/FP32 scratches and independent
        # of quant format, but live next to NVFP4-specific scratch.
        if self._weights.ptrs.get('mtp') is not None:
            fp32 = torch.float32
            self._mtp_K_cache = torch.empty(
                max_seq, 4, 256, device=device, dtype=bf16)
            self._mtp_V_cache = torch.empty_like(self._mtp_K_cache)
            self._mtp_Q_buf = torch.empty(
                1, 1, 24, 256, device=device, dtype=bf16)
            self._mtp_O_buf = torch.empty_like(self._mtp_Q_buf)
            sq_rounded = 128
            self._mtp_lse_buf = torch.empty(
                1, 24, sq_rounded, device=device, dtype=fp32)
            n_splits = min(128, (max_seq + 63) // 64)
            self._mtp_n_splits = n_splits
            self._mtp_lse_accum = torch.empty(
                n_splits, 1, 24, 1, device=device, dtype=fp32)
            self._mtp_o_accum = torch.empty(
                n_splits, 1, 24, 1, 256, device=device, dtype=fp32)
            self._mtp_h_norm_buf = torch.empty(
                1, 1, hidden, device=device, dtype=bf16)
            self._mtp_e_norm_buf = torch.empty(
                1, 1, hidden, device=device, dtype=bf16)
            self._mtp_cat_buf = torch.empty(
                1, 1, 2 * hidden, device=device, dtype=bf16)
            self._mtp_fc_out_buf = torch.empty(
                1, 1, hidden, device=device, dtype=bf16)
            self._mtp_layer_out_buf = torch.empty(
                1, 1, hidden, device=device, dtype=bf16)
            self._mtp_logits_buf = torch.empty(
                1, vocab, device=device, dtype=bf16)
            # Captured MTP graphs (per mtp_pos), mirrors FP8 path.
            self._mtp_static_prev_h = torch.empty(
                1, 1, hidden, device=device, dtype=bf16)
            self._mtp_static_prev_token = torch.zeros(
                1, 1, device=device, dtype=torch.long)
            self._captured_mtp_graphs: collections.OrderedDict[
                int, torch.cuda.CUDAGraph,
            ] = collections.OrderedDict()
            # G9: per-(base_pos, K) chain graph that captures the
            # entire K-step MTP chain (forward + argmax + inter-step
            # state copies) in one graph. Eliminates K-1 Python-level
            # replay/copy/argmax launches per spec cycle.
            self._chain_drafts_buf = torch.zeros(
                self.MAX_Q_SEQ - 1, 1, device=device, dtype=torch.long)
            self._captured_chain_graphs: collections.OrderedDict[
                tuple[int, int], torch.cuda.CUDAGraph,
            ] = collections.OrderedDict()

        # Bundle the integer-pointer view forward consumers expect.
        self._bufs = {
            'h_a': int(self._h_a.data_ptr()),
            'h_b': int(self._h_b.data_ptr()),
            'logits': int(self._logits_buf.data_ptr()),
            'lin_state_base': int(self._lin_state.data_ptr()),
            'lin_state_layer_stride_bytes': int(
                self._lin_state.stride(0) * self._lin_state.element_size()),
            'lin_conv_state_base': int(self._lin_conv_state.data_ptr()),
            'lin_conv_state_layer_stride_bytes': int(
                self._lin_conv_state.stride(0)
                * self._lin_conv_state.element_size()),
            'h_row_stride_elts': int(self._h_a.stride(0)),
            'nvfp4_scratch': {
                shape: (
                    int(ap.data_ptr()),
                    int(sf.data_ptr()),
                    int(out.data_ptr()),
                )
                for shape, (ap, sf, out) in self._nvfp4_scratch.items()
            },
            'nvfp4_bf16_out': {
                N: int(t.data_ptr())
                for N, t in self._nvfp4_bf16_out.items()
            },
        }

    # ---------- NVFP4 own forward (N4 stage 2) ----------

    def _layer_forward_lin_nvfp4(self, L: int, h_in):
        """Linear-attention layer forward, NVFP4 main + BF16 in_proj.

        Math is identical to ``_layer_forward_lin`` (FP8 path); only
        the GEMM kernels differ:
          * in_proj_qkv / in_proj_z / out_proj are stored BF16 in
            this NVFP4 ckpt (compressed-tensors leaves linear-attn
            projections un-quantized) — we use the stream-invariant
            ``bf16_matvec_qwen36_bf16`` so M=1 decode stays graph-safe.
          * MLP gate/up/down are NVFP4: swizzled per-token act quant
            + ``fp4_w4a16_gemm_sm120_bf16out`` with alpha=1/global.
          * conv1d / gated_deltanet / rms_norm_gated_silu / sigmoid /
            softplus / index_select / RMSNorm steps are 100% shared
            with the FP8 path (kernel ABI identical).
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        bf16 = torch.bfloat16
        s = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        assert lw['type'] == 'linear_attention', (
            f'_layer_forward_lin_nvfp4 layer {L} type {lw["type"]!r}'
        )

        h2 = h_in.view(1, 5120).contiguous()
        eps = float(self._cfg['rms_norm_eps'])

        x_norm = self._h_b[:1]
        x_norm_view = x_norm.view(1, 5120)

        # 1) input layernorm + NVFP4 quant — fused (B1).
        # Output ap/sf is reused by the three NVFP4 lin projections.
        ap_5120, sf_5120, _ = self._nvfp4_scratch[(10240, 5120)]
        fvk.rms_norm_to_nvfp4_swizzled_bf16(
            h2.data_ptr(), int(lw['input_norm_eff_w']),
            ap_5120.data_ptr(), sf_5120.data_ptr(),
            1, 5120, eps, s,
        )
        # We also need x_norm BF16 for the tiny in_proj_a/b matvec.
        # Recompute it: cheap (5120 elements, single launch).
        fvk.rms_norm(
            h2.data_ptr(), int(lw['input_norm_eff_w']),
            x_norm_view.data_ptr(), 1, 5120, eps, s,
        )

        # 2) in_proj_qkv (G7: NVFP4 N=10240, K=5120).
        out_qkv_buf = self._nvfp4_scratch[(10240, 5120)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['in_proj_qkv_packed']),
            out_qkv_buf.data_ptr(),
            1, 10240, 5120,
            sf_5120.data_ptr(), int(lw['in_proj_qkv_sf']),
            float(lw['in_proj_qkv_alpha']),
            s,
        )
        # 3) in_proj_z (G7: NVFP4 N=6144).
        out_z_buf = self._nvfp4_scratch[(6144, 5120)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['in_proj_z_packed']),
            out_z_buf.data_ptr(),
            1, 6144, 5120,
            sf_5120.data_ptr(), int(lw['in_proj_z_sf']),
            float(lw['in_proj_z_alpha']),
            s,
        )
        # 4) in_proj_a / in_proj_b stay BF16 (tiny, N=48 each).
        # A2c-2: fused as one N=96 matvec via concatenated in_proj_ab_w.
        # _lin_a_vec / _lin_b_vec are views into _lin_ab_vec[:, :48] /
        # _lin_ab_vec[:, 48:] so downstream consumers see them unchanged.
        a_vec = self._lin_a_vec
        b_vec = self._lin_b_vec
        fvk.bf16_matvec_qwen36_bf16(
            x_norm.data_ptr(), int(lw['in_proj_ab_w']),
            self._lin_ab_vec.data_ptr(), 96, 5120, s,
        )

        # 5) causal_conv1d_update on the qkv stream.
        qkv_in = out_qkv_buf[:1].view(1, 10240)
        lin_rank = self._linear_layer_rank(L)
        conv_state = self._lin_conv_state[lin_rank]
        rec_state = self._lin_state[lin_rank]
        conv_out = self._lin_conv_out
        fvk.causal_conv1d_qwen36_update_bf16(
            qkv_in.data_ptr(), int(lw['conv1d_w']),
            int(lw['conv1d_b']),
            conv_out.data_ptr(), conv_state.data_ptr(),
            1, 10240, 4, True, s,
        )

        # 6) split conv_out -> q (2048), k (2048), v (6144).
        q_flat = conv_out[:, :2048]
        k_flat = conv_out[:, 2048:4096]
        v_flat = conv_out[:, 4096:10240]
        q = q_flat.view(1, 1, 16, 128)
        k = k_flat.view(1, 1, 16, 128)
        v = v_flat.view(1, 1, 48, 128)

        # 7) beta = sigmoid(b); g = -A_log.exp() * softplus(a + dt_bias).
        torch.sigmoid(b_vec, out=self._lin_beta)
        self._lin_a_f32.copy_(a_vec)
        self._lin_a_f32.add_(lw['dt_bias_fp32_t'])
        torch.exp(self._lin_a_f32, out=self._lin_sp_buf)
        torch.log1p(self._lin_sp_buf, out=self._lin_sp_buf)
        torch.mul(lw['neg_A_log_exp_fp32_t'], self._lin_sp_buf,
                  out=self._lin_g_f32)
        self._lin_g_bf.copy_(self._lin_g_f32)
        beta = self._lin_beta
        g_bf = self._lin_g_bf

        # 8) Broadcast Q,K from 16 to 48 heads.
        q_2d = q.view(1, 16, 128)
        k_2d = k.view(1, 16, 128)
        torch.index_select(q_2d, 1, self._lin_broadcast_idx, out=self._lin_q48)
        torch.index_select(k_2d, 1, self._lin_broadcast_idx, out=self._lin_k48)
        q3 = self._lin_q48
        k3 = self._lin_k48
        v3 = v.view(1, 48, 128).contiguous()
        attn_out_buf = self._lin_attn_out

        fvk.gated_deltanet_recurrent_qwen36_bf16(
            q3.data_ptr(), k3.data_ptr(), v3.data_ptr(),
            g_bf.data_ptr(), beta.data_ptr(),
            rec_state.data_ptr(), attn_out_buf.data_ptr(),
            1, 48, 128, 128, True, s,
        )

        # 9) rms_norm_gated_silu (M=48, dim=128).
        z_flat = out_z_buf[:1].view(48, 128)
        attn_out_flat = attn_out_buf.view(48, 128)
        norm_out = self._lin_norm_out
        fvk.rms_norm_gated_silu_qwen36_bf16(
            attn_out_flat.data_ptr(), z_flat.data_ptr(),
            int(lw['head_norm_w']),
            norm_out.data_ptr(), 48, 128, eps, s,
        )

        # 10) out_proj (G7: NVFP4 N=5120, K=6144).
        # Quant the BF16 attn output (norm_out is the rms_norm_gated_silu
        # output, BF16 (1, 6144)) to NVFP4 then run NVFP4 GEMM.
        ap_6144, sf_6144, _ = self._nvfp4_scratch[(5120, 6144)]
        norm_out_1x6144 = norm_out.view(1, 6144)
        fvk.quantize_bf16_to_nvfp4_swizzled(
            norm_out_1x6144.data_ptr(), ap_6144.data_ptr(),
            sf_6144.data_ptr(), 1, 6144, s,
        )
        out_op_buf = self._nvfp4_scratch[(5120, 6144)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_6144.data_ptr(), int(lw['out_proj_packed']),
            out_op_buf.data_ptr(),
            1, 5120, 6144,
            sf_6144.data_ptr(), int(lw['out_proj_sf']),
            float(lw['out_proj_alpha']),
            s,
        )

        # 11) residual.
        attn_proj = out_op_buf[:1].view(1, 1, 5120)
        torch.add(h_in, attn_proj, out=self._res_mid)
        h_post = self._res_mid

        # 12) post-attn layernorm.
        h_post_view = h_post.view(1, 5120)
        x_mlp = self._h_b[:1].view(1, 5120)
        # 12) post-attn layernorm.
        # NB: tried fusing rms_norm + nvfp4_swizzled_quant; at M=1 +
        # CUDA Graph it was 2% slower (graph already amortizes launch
        # overhead, and the fused kernel has higher smem pressure).
        # The fused primitive (rms_norm_to_nvfp4_swizzled_bf16) IS
        # bound and bit-equivalent — kept for the verify (S=K) and
        # prefill (S=large) paths where its benefit kicks in.
        x_mlp = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            h_post_view.data_ptr(), int(lw['post_attn_norm_eff_w']),
            x_mlp.data_ptr(), 1, 5120, eps, s,
        )
        ap_5120, sf_5120, _ = self._nvfp4_scratch[(17408, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_mlp.data_ptr(), ap_5120.data_ptr(),
            sf_5120.data_ptr(), 1, 5120, s,
        )

        gate_out_buf = self._nvfp4_scratch[(17408, 5120)][2]
        up_out_buf = self._mlp_up_out
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_5120.data_ptr(), int(lw['mlp_gate_packed']),
            gate_out_buf.data_ptr(),
            1, 17408, 5120,
            sf_5120.data_ptr(), int(lw['mlp_gate_sf']),
            float(lw['mlp_gate_alpha']),
            s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_5120.data_ptr(), int(lw['mlp_up_packed']),
            up_out_buf.data_ptr(),
            1, 17408, 5120,
            sf_5120.data_ptr(), int(lw['mlp_up_sf']),
            float(lw['mlp_up_alpha']),
            s,
        )

        # 14) silu(gate) * up via fvk fused kernel.
        gate_v = gate_out_buf[:1].view(1, 17408)
        up_v = up_out_buf[:1].view(1, 17408)
        fvk.silu_mul_qwen36_bf16(
            gate_v.data_ptr(), up_v.data_ptr(),
            self._mlp_silu_mul_out.data_ptr(), 17408, s,
        )
        gate_silu_up = self._mlp_silu_mul_out

        # 15) MLP down: NVFP4 quant act (M=1, K=17408), FP4 GEMM N=5120.
        ap_17408, sf_17408, _ = self._nvfp4_scratch[(5120, 17408)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            gate_silu_up.data_ptr(), ap_17408.data_ptr(),
            sf_17408.data_ptr(), 1, 17408, s,
        )
        down_out_buf = self._nvfp4_scratch[(5120, 17408)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_17408.data_ptr(), int(lw['mlp_down_packed']),
            down_out_buf.data_ptr(),
            1, 5120, 17408,
            sf_17408.data_ptr(), int(lw['mlp_down_sf']),
            float(lw['mlp_down_alpha']),
            s,
        )
        mlp_out = down_out_buf[:1].view(1, 1, 5120)

        # 16) final residual.
        h_out = self._layer_out_a if (L % 2 == 0) else self._layer_out_b
        torch.add(h_post, mlp_out, out=h_out)
        return h_out

    def _layer_forward_full_nvfp4(self, L: int, h_in, cos, sin, cur_pos: int):
        """Full-attention layer forward, NVFP4 main.

        All four self-attn projections (q/k/v/o) AND all three MLP
        projections in this NVFP4 ckpt are quantized — every GEMM
        here is the new fp4_w4a16 kernel. Math otherwise identical
        to the FP8 ``_layer_forward_full``.
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        s = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        assert lw['type'] == 'full_attention', (
            f'_layer_forward_full_nvfp4 layer {L} type {lw["type"]!r}'
        )

        h2 = h_in.view(1, 5120).contiguous()
        eps = float(self._cfg['rms_norm_eps'])
        full_rank = self._full_layer_rank(L)

        # 1) input layernorm.
        x_norm = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            h2.data_ptr(), int(lw['input_norm_eff_w']),
            x_norm.data_ptr(), 1, 5120, eps, s,
        )
        # 2) NVFP4 quantize x_norm once (M=1, K=5120) — reused for q/k/v.
        ap_5120, sf_5120, _ = self._nvfp4_scratch[(12288, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_norm.data_ptr(), ap_5120.data_ptr(),
            sf_5120.data_ptr(), 1, 5120, s,
        )

        # 3) q_proj fused (Q + output_gate) -> (1, 12288).
        q_proj_out_buf = self._nvfp4_scratch[(12288, 5120)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['q_proj_packed']),
            q_proj_out_buf.data_ptr(),
            1, 12288, 5120,
            sf_5120.data_ptr(), int(lw['q_proj_sf']),
            float(lw['q_proj_alpha']),
            s,
        )
        q_full = q_proj_out_buf[:1].view(1, 1, 24, 512)
        q_pre, gate = torch.chunk(q_full, 2, dim=-1)
        gate_flat = gate.reshape(1, 1, 24 * 256)

        # 4) k_proj -> (1, 1024).
        kv_proj_out_buf = self._nvfp4_scratch[(1024, 5120)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['k_proj_packed']),
            kv_proj_out_buf.data_ptr(),
            1, 1024, 5120,
            sf_5120.data_ptr(), int(lw['k_proj_sf']),
            float(lw['k_proj_alpha']),
            s,
        )
        k_pre = kv_proj_out_buf[:1].view(1, 1, 4, 256).contiguous()

        # 5) q_norm / k_norm.
        q_pre_2d = q_pre.contiguous().view(24, 256)
        fvk.rms_norm(
            q_pre_2d.data_ptr(), int(lw['q_norm_eff_w']),
            self._full_q_norm_out.data_ptr(), 24, 256, eps, s,
        )
        k_pre_2d = k_pre.view(4, 256)
        fvk.rms_norm(
            k_pre_2d.data_ptr(), int(lw['k_norm_eff_w']),
            self._full_k_norm_out.data_ptr(), 4, 256, eps, s,
        )

        # 6) Inline RoPE on partial (rotary_dim=64) Q/K.
        q_for_rope = self._full_q_norm_out.view(1, 1, 24, 256)
        k_for_rope = self._full_k_norm_out.view(1, 1, 4, 256)
        cos4 = cos.view(1, 1, 1, 64)
        sin4 = sin.view(1, 1, 1, 64)

        def _rope_inline(x_in, x_out, tmp):
            x_out[..., 64:].copy_(x_in[..., 64:])
            torch.index_select(
                x_in[..., :64], -1, self._rope_rotate_idx, out=tmp,
            )
            tmp[..., :32].neg_()
            tmp.mul_(sin4)
            tmp.addcmul_(x_in[..., :64], cos4)
            x_out[..., :64].copy_(tmp)

        _rope_inline(q_for_rope, self._full_q_rot, self._full_rope_tmp_q)
        _rope_inline(k_for_rope, self._full_k_rot, self._full_rope_tmp_k)
        q_rot = self._full_q_rot
        k_rot = self._full_k_rot

        # 7) Stage Q + write K, V to KV cache.
        self._attn.Q_buf[:, :1].copy_(q_rot)
        self._attn.K_cache[full_rank, cur_pos:cur_pos + 1].copy_(
            k_rot.view(1, 4, 256)
        )
        # v_proj — reuse kv_proj_out_buf (k already in cache).
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['v_proj_packed']),
            kv_proj_out_buf.data_ptr(),
            1, 1024, 5120,
            sf_5120.data_ptr(), int(lw['v_proj_sf']),
            float(lw['v_proj_alpha']),
            s,
        )
        v_new = kv_proj_out_buf[:1].view(1, 4, 256)
        self._attn.V_cache[full_rank, cur_pos:cur_pos + 1].copy_(v_new)

        # N7-B4: TurboQuant inject (no-op when disabled or graph-capturing).
        # Replaces just-written K/V in cache with their TQ round-tripped
        # versions, so downstream attention sees the same noise profile
        # it would see in a packed-cache deployment.
        self._tq_inject_kv(full_rank, cur_pos, count=1)

        # 8) Run attention.
        kv_seq = cur_pos + 1
        scaling = float(self._cfg['head_dim']) ** -0.5
        self._attn.run(
            'full', layer_idx=full_rank, q_seq=1, kv_seq=kv_seq,
            stream=s, softmax_scale=scaling,
        )
        attn_out = self._attn.O_buf[:, :1]

        # 9) Output gate: attn * sigmoid(gate).
        attn_flat = attn_out.reshape(1, 1, 24 * 256)
        torch.sigmoid(gate_flat, out=self._full_gate_sig)
        torch.mul(attn_flat, self._full_gate_sig, out=self._full_gated)
        gated = self._full_gated

        # 10) o_proj NVFP4: K=6144 -> N=5120.
        ap_6144, sf_6144, _ = self._nvfp4_scratch[(5120, 6144)]
        gated_2d = gated.view(1, 6144).contiguous()
        fvk.quantize_bf16_to_nvfp4_swizzled(
            gated_2d.data_ptr(), ap_6144.data_ptr(),
            sf_6144.data_ptr(), 1, 6144, s,
        )
        out_op_buf = self._nvfp4_scratch[(5120, 6144)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_6144.data_ptr(), int(lw['o_proj_packed']),
            out_op_buf.data_ptr(),
            1, 5120, 6144,
            sf_6144.data_ptr(), int(lw['o_proj_sf']),
            float(lw['o_proj_alpha']),
            s,
        )

        # 11) Residual.
        attn_proj = out_op_buf[:1].view(1, 1, 5120)
        torch.add(h_in, attn_proj, out=self._res_mid)
        h_post = self._res_mid

        # 12) post-attn layernorm.
        h_post_view = h_post.view(1, 5120)
        x_mlp = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            h_post_view.data_ptr(), int(lw['post_attn_norm_eff_w']),
            x_mlp.data_ptr(), 1, 5120, eps, s,
        )

        # 13) MLP gate / up: NVFP4.
        ap_mlp, sf_mlp, _ = self._nvfp4_scratch[(17408, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_mlp.data_ptr(), ap_mlp.data_ptr(),
            sf_mlp.data_ptr(), 1, 5120, s,
        )
        gate_out_buf = self._nvfp4_scratch[(17408, 5120)][2]
        up_out_buf = self._mlp_up_out
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_mlp.data_ptr(), int(lw['mlp_gate_packed']),
            gate_out_buf.data_ptr(),
            1, 17408, 5120,
            sf_mlp.data_ptr(), int(lw['mlp_gate_sf']),
            float(lw['mlp_gate_alpha']),
            s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_mlp.data_ptr(), int(lw['mlp_up_packed']),
            up_out_buf.data_ptr(),
            1, 17408, 5120,
            sf_mlp.data_ptr(), int(lw['mlp_up_sf']),
            float(lw['mlp_up_alpha']),
            s,
        )

        # 14) silu(gate) * up via fvk.
        gate_v = gate_out_buf[:1].view(1, 17408)
        up_v = up_out_buf[:1].view(1, 17408)
        fvk.silu_mul_qwen36_bf16(
            gate_v.data_ptr(), up_v.data_ptr(),
            self._mlp_silu_mul_out.data_ptr(), 17408, s,
        )
        gate_silu_up = self._mlp_silu_mul_out

        # 15) MLP down.
        ap_dn, sf_dn, _ = self._nvfp4_scratch[(5120, 17408)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            gate_silu_up.data_ptr(), ap_dn.data_ptr(),
            sf_dn.data_ptr(), 1, 17408, s,
        )
        down_out_buf = self._nvfp4_scratch[(5120, 17408)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_dn.data_ptr(), int(lw['mlp_down_packed']),
            down_out_buf.data_ptr(),
            1, 5120, 17408,
            sf_dn.data_ptr(), int(lw['mlp_down_sf']),
            float(lw['mlp_down_alpha']),
            s,
        )
        mlp_out = down_out_buf[:1].view(1, 1, 5120)

        # 16) Final residual.
        h_out = self._layer_out_a if (L % 2 == 0) else self._layer_out_b
        torch.add(h_post, mlp_out, out=h_out)
        return h_out

    # ---------- N5-stage3: NVFP4 S=K linear-attn layer ----------

    def _layer_forward_lin_K_nvfp4(self, L: int, h_in_K, K: int):
        """NVFP4 S=K linear-attention decoder layer (verify path).

        Mechanical mirror of the FP8 ``_layer_forward_lin_K``. Three
        differences vs. the FP8 path:
          1. in_proj_qkv / in_proj_z / out_proj are BF16 in this NVFP4
             ckpt (compressed-tensors leaves linear-attn projections
             unquantized). Use K-loop ``bf16_matvec_qwen36_bf16``
             matching the NVFP4 S=1 path's idiom.
          2. MLP gate / up / down are NVFP4: per-token NVFP4 quant of
             the M=K activation, then ``fp4_w4a16_gemm_sm120_bf16out``
             at M=K with alpha = 1/global_scale.
          3. The conv1d / FLA chunk / rms_norm_gated_silu / sigmoid /
             softplus / index_select sub-steps are 100% shared with
             the FP8 path — kernel ABIs are identical.
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        s = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        assert lw['type'] == 'linear_attention', (
            f'_layer_forward_lin_K_nvfp4 layer {L} type {lw["type"]!r}'
        )
        eps = float(self._cfg['rms_norm_eps'])

        h2 = h_in_K.view(K, 5120).contiguous()
        x_norm = self._h_b[:K].view(K, 5120)

        # 1) input layernorm M=K.
        fvk.rms_norm(
            h2.data_ptr(), int(lw['input_norm_eff_w']),
            x_norm.data_ptr(),
            K, 5120, eps, s,
        )

        # 2) NVFP4 quant act once (M=K, K=5120) — reused for qkv/z (G7).
        ap_5120, sf_5120, _ = self._nvfp4_scratch[(10240, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_norm.data_ptr(), ap_5120.data_ptr(),
            sf_5120.data_ptr(), K, 5120, s,
        )
        # in_proj_qkv (G7: NVFP4 N=10240, K=5120, M=K).
        out_qkv_buf = self._nvfp4_scratch[(10240, 5120)][2]
        out_qkv_K = out_qkv_buf[:K]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['in_proj_qkv_packed']),
            out_qkv_K.data_ptr(),
            K, 10240, 5120,
            sf_5120.data_ptr(), int(lw['in_proj_qkv_sf']),
            float(lw['in_proj_qkv_alpha']),
            s,
        )
        # 3) in_proj_z (G7: NVFP4 N=6144, K=5120, M=K) — reuse same act.
        out_z_buf = self._nvfp4_scratch[(6144, 5120)][2]
        out_z_K = out_z_buf[:K]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['in_proj_z_packed']),
            out_z_K.data_ptr(),
            K, 6144, 5120,
            sf_5120.data_ptr(), int(lw['in_proj_z_sf']),
            float(lw['in_proj_z_alpha']),
            s,
        )

        # 4) in_proj_a / in_proj_b stay BF16 (tiny N=48). Single M=K
        # matmul each.
        a_vec_K = self._K_lin_a_vec[:K]
        b_vec_K = self._K_lin_b_vec[:K]
        fvk.bf16_matmul_qwen36_bf16(
            x_norm.data_ptr(), int(lw['in_proj_a_w']),
            a_vec_K.data_ptr(), K, 48, 5120, s,
        )
        fvk.bf16_matmul_qwen36_bf16(
            x_norm.data_ptr(), int(lw['in_proj_b_w']),
            b_vec_K.data_ptr(), K, 48, 5120, s,
        )

        # 5) Per-token conv1d_update with chained per-step state save.
        # A2c-3: conv1d in/out variant chains state through the
        # _K_lin_conv_state_per_step slots (step k reads slot k-1 or
        # the live conv_state for k=0; writes slot k). One final copy
        # at end commits the post-step-K-1 state back to lin_conv_state
        # (vs K explicit saves in the loop).
        lin_rank = self._linear_layer_rank(L)
        conv_state = self._lin_conv_state[lin_rank]
        qkv_K_view = out_qkv_K  # (K, 10240)
        save_steps = K if K <= self._K_save_max else 0

        if save_steps > 0:
            for k in range(K):
                qkv_row = qkv_K_view[k:k + 1]
                conv_out_row = self._K_lin_conv_out[k:k + 1]
                state_in_ptr = (
                    conv_state.data_ptr() if k == 0
                    else self._K_lin_conv_state_per_step[
                        k - 1, lin_rank].data_ptr()
                )
                state_out_ptr = self._K_lin_conv_state_per_step[
                    k, lin_rank].data_ptr()
                fvk.causal_conv1d_qwen36_update_inout_bf16(
                    qkv_row.data_ptr(), int(lw['conv1d_w']),
                    int(lw['conv1d_b']),
                    conv_out_row.data_ptr(),
                    state_in_ptr, state_out_ptr,
                    1, 10240, 4, True, s,
                )
            conv_state.copy_(
                self._K_lin_conv_state_per_step[K - 1, lin_rank])
        else:
            for k in range(K):
                qkv_row = qkv_K_view[k:k + 1]
                conv_out_row = self._K_lin_conv_out[k:k + 1]
                fvk.causal_conv1d_qwen36_update_bf16(
                    qkv_row.data_ptr(), int(lw['conv1d_w']),
                    int(lw['conv1d_b']),
                    conv_out_row.data_ptr(), conv_state.data_ptr(),
                    1, 10240, 4, True, s,
                )

        # 5b) Split conv output into Q (K, 16, 128), K (K, 16, 128),
        # V (K, 48, 128). Layout: [Q_dim=2048, K_dim=2048, V_dim=6144].
        conv_K = self._K_lin_conv_out[:K]
        q_K_heads = conv_K[:, :2048].contiguous().view(1, K, 16, 128)
        k_K_heads = conv_K[:, 2048:4096].contiguous().view(1, K, 16, 128)
        v_K_heads = conv_K[:, 4096:10240].contiguous().view(1, K, 48, 128)

        # 5c) Compute g, beta for all K tokens (M=K vector ops).
        beta_K = self._K_lin_beta[:K]
        torch.sigmoid(b_vec_K, out=beta_K)
        a_f32_K = self._K_lin_a_f32[:K]
        a_f32_K.copy_(a_vec_K)
        a_f32_K.add_(lw['dt_bias_fp32_t'])
        sp_K = self._K_lin_sp_buf[:K]
        torch.exp(a_f32_K, out=sp_K)
        torch.log1p(sp_K, out=sp_K)
        g_f32_K = self._K_lin_g_f32[:K]
        torch.mul(lw['neg_A_log_exp_fp32_t'], sp_K, out=g_f32_K)
        g_bf_K = self._K_lin_g_bf[:K]
        g_bf_K.copy_(g_f32_K)

        # 5d) K-iter recurrent (replaces FLA chunk_gated_delta_rule).
        # A2c-3: chain state through per-step save slots via the in/out
        # variant — eliminates the in-loop .copy_(state_save, state)
        # launch per step. Step k reads state from slot k-1 (or
        # lin_state for k=0) and writes to slot k. After K steps,
        # lin_state ← per_step[K-1] (1 final copy per layer instead
        # of K copies inside the loop).
        q_K_2d = q_K_heads.view(K, 16, 128)
        k_K_2d = k_K_heads.view(K, 16, 128)
        q_K_48 = self._K_lin_q48[:K]
        k_K_48 = self._K_lin_k48[:K]
        torch.index_select(q_K_2d, 1, self._lin_broadcast_idx,
                           out=q_K_48)
        torch.index_select(k_K_2d, 1, self._lin_broadcast_idx,
                           out=k_K_48)
        rec_state_view = self._lin_state[lin_rank]  # (1, 48, 128, 128)
        v_K_3d = v_K_heads.view(K, 48, 128)
        attn_out_K_buf = self._K_lin_attn_out[:K]
        use_inout = K <= self._K_save_max
        if use_inout:
            for k in range(K):
                state_in_ptr = (
                    rec_state_view.data_ptr() if k == 0
                    else self._K_lin_state_per_step[k - 1, lin_rank].data_ptr()
                )
                state_out_ptr = self._K_lin_state_per_step[
                    k, lin_rank].data_ptr()
                fvk.gated_deltanet_recurrent_inout_qwen36_bf16(
                    q_K_48[k].data_ptr(), k_K_48[k].data_ptr(),
                    v_K_3d[k].data_ptr(),
                    g_bf_K[k].data_ptr(), beta_K[k].data_ptr(),
                    state_in_ptr, state_out_ptr,
                    attn_out_K_buf[k].data_ptr(),
                    1, 48, 128, 128, True, s,
                )
            # lin_state ← state at step K-1 (= final state for next cycle).
            rec_state_view.copy_(
                self._K_lin_state_per_step[K - 1, lin_rank])
        else:
            # Fallback for K beyond save buffer: original in-place + copy path.
            for k in range(K):
                fvk.gated_deltanet_recurrent_qwen36_bf16(
                    q_K_48[k].data_ptr(), k_K_48[k].data_ptr(),
                    v_K_3d[k].data_ptr(),
                    g_bf_K[k].data_ptr(), beta_K[k].data_ptr(),
                    rec_state_view.data_ptr(),
                    attn_out_K_buf[k].data_ptr(),
                    1, 48, 128, 128, True, s,
                )
        attn_out_K = attn_out_K_buf.view(1, K, 48, 128)

        # 5e) rms_norm_gated_silu over (K*48, 128) rows in one call.
        attn_out_flat = attn_out_K.contiguous().view(K * 48, 128)
        z_flat = out_z_K.view(K * 48, 128)
        norm_out_flat = self._K_lin_norm_out[:K].view(K * 48, 128)
        fvk.rms_norm_gated_silu_qwen36_bf16(
            attn_out_flat.data_ptr(), z_flat.data_ptr(),
            int(lw['head_norm_w']),
            norm_out_flat.data_ptr(),
            K * 48, 128, eps, s,
        )

        # 6) out_proj (G7: NVFP4 N=5120, K_in=6144, M=K).
        norm_out_K = self._K_lin_norm_out[:K].view(K, 6144)
        ap_6144, sf_6144, _ = self._nvfp4_scratch[(5120, 6144)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            norm_out_K.data_ptr(), ap_6144.data_ptr(),
            sf_6144.data_ptr(), K, 6144, s,
        )
        out_op_buf = self._nvfp4_scratch[(5120, 6144)][2]
        out_op_K = out_op_buf[:K]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_6144.data_ptr(), int(lw['out_proj_packed']),
            out_op_K.data_ptr(),
            K, 5120, 6144,
            sf_6144.data_ptr(), int(lw['out_proj_sf']),
            float(lw['out_proj_alpha']),
            s,
        )

        # 7-8) Residual + post-attn norm + MLP NVFP4 quant — fused (B2).
        # Replaces (torch.add + rms_norm + quantize_bf16_to_nvfp4_swizzled)
        # 3-launch sequence with one kernel. h_post (BF16) is preserved
        # in _K_res_mid because the post-MLP residual still needs it.
        attn_proj = out_op_K.view(1, K, 5120)
        res_mid_K = self._K_res_mid[:, :K]
        ap_5120, sf_5120, _ = self._nvfp4_scratch[(17408, 5120)]
        fvk.residual_add_rms_norm_to_nvfp4_swizzled_bf16(
            h_in_K.data_ptr(), attn_proj.data_ptr(),
            res_mid_K.data_ptr(),
            int(lw['post_attn_norm_eff_w']),
            ap_5120.data_ptr(), sf_5120.data_ptr(),
            K, 5120, eps, s,
        )
        h_post = res_mid_K
        gate_out_buf = self._nvfp4_scratch[(17408, 5120)][2]
        up_out_buf = self._mlp_up_out
        # 8b) MLP gate / up: NVFP4 GEMM at M=K.
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_5120.data_ptr(), int(lw['mlp_gate_packed']),
            gate_out_buf.data_ptr(),
            K, 17408, 5120,
            sf_5120.data_ptr(), int(lw['mlp_gate_sf']),
            float(lw['mlp_gate_alpha']),
            s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_5120.data_ptr(), int(lw['mlp_up_packed']),
            up_out_buf.data_ptr(),
            K, 17408, 5120,
            sf_5120.data_ptr(), int(lw['mlp_up_sf']),
            float(lw['mlp_up_alpha']),
            s,
        )
        # 8c) silu(gate) * up over K*17408 elements.
        gate_v = gate_out_buf[:K].view(K, 17408)
        up_v = up_out_buf[:K].view(K, 17408)
        silu_out = self._K_mlp_silu_mul_out[:K]
        fvk.silu_mul_qwen36_bf16(
            gate_v.data_ptr(), up_v.data_ptr(),
            silu_out.data_ptr(), K * 17408, s,
        )
        # 8d) MLP down: NVFP4 quant act (M=K, K_in=17408), NVFP4 GEMM.
        ap_17408, sf_17408, _ = self._nvfp4_scratch[(5120, 17408)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            silu_out.data_ptr(), ap_17408.data_ptr(),
            sf_17408.data_ptr(), K, 17408, s,
        )
        down_out_buf = self._nvfp4_scratch[(5120, 17408)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_17408.data_ptr(), int(lw['mlp_down_packed']),
            down_out_buf.data_ptr(),
            K, 5120, 17408,
            sf_17408.data_ptr(), int(lw['mlp_down_sf']),
            float(lw['mlp_down_alpha']),
            s,
        )
        mlp_out = down_out_buf[:K].view(1, K, 5120)

        # 9) Final residual: write to _K layer-out ping-pong.
        h_out = (self._K_layer_out_a if (L % 2 == 0)
                 else self._K_layer_out_b)
        h_out_K = h_out[:, :K]
        torch.add(h_post, mlp_out, out=h_out_K)
        return h_out_K

    # ---------- N5-stage4: NVFP4 S=K full-attn layer ----------

    def _layer_forward_full_K_nvfp4(self, L: int, h_in_K, cos_K, sin_K,
                                     cur_pos: int, K: int):
        """NVFP4 S=K full-attention decoder layer (verify path).

        Mechanical mirror of FP8 ``_layer_forward_full_K`` with these
        NVFP4 swaps (every projection in this NVFP4 ckpt's full-attn
        is quantized):
          * q_proj / k_proj / v_proj / o_proj: NVFP4 quant of M=K
            activation, then ``fp4_w4a16_gemm_sm120_bf16out`` at M=K.
          * MLP gate / up / down: same NVFP4 quant + GEMM pattern.

        The K serial q_seq=1 FA2 calls (workaround for fwd_bf16's
        missing causal flag) are identical to the FP8 path — kernel
        ABI is format-agnostic.
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        s = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        assert lw['type'] == 'full_attention', (
            f'_layer_forward_full_K_nvfp4 layer {L} type {lw["type"]!r}'
        )
        eps = float(self._cfg['rms_norm_eps'])
        full_rank = self._full_layer_rank(L)

        h2 = h_in_K.view(K, 5120).contiguous()

        # 1) input layernorm + NVFP4 quant — fused (B1).
        # Output (packed + swizzled SF) is reused by q/k/v projections;
        # no BF16 intermediate buf needed.
        ap_5120, sf_5120, _ = self._nvfp4_scratch[(12288, 5120)]
        fvk.rms_norm_to_nvfp4_swizzled_bf16(
            h2.data_ptr(), int(lw['input_norm_eff_w']),
            ap_5120.data_ptr(), sf_5120.data_ptr(),
            K, 5120, eps, s,
        )

        # 3) q_proj fused (Q + output_gate) — M=K, N=12288.
        q_proj_out_buf = self._nvfp4_scratch[(12288, 5120)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['q_proj_packed']),
            q_proj_out_buf.data_ptr(),
            K, 12288, 5120,
            sf_5120.data_ptr(), int(lw['q_proj_sf']),
            float(lw['q_proj_alpha']),
            s,
        )
        q_full = q_proj_out_buf[:K].view(1, K, 24, 512)
        q_pre, gate = torch.chunk(q_full, 2, dim=-1)
        gate_flat = gate.reshape(1, K, 24 * 256)

        # 4) k_proj — M=K, N=1024.
        kv_proj_out_buf = self._nvfp4_scratch[(1024, 5120)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['k_proj_packed']),
            kv_proj_out_buf.data_ptr(),
            K, 1024, 5120,
            sf_5120.data_ptr(), int(lw['k_proj_sf']),
            float(lw['k_proj_alpha']),
            s,
        )
        k_pre = kv_proj_out_buf[:K].view(1, K, 4, 256).contiguous()

        # 5) q_norm / k_norm — per-head RMSNorm (M = K * heads).
        q_pre_2d = q_pre.contiguous().view(K * 24, 256)
        q_norm_out = self._K_full_q_norm_out[:K * 24]
        fvk.rms_norm(
            q_pre_2d.data_ptr(), int(lw['q_norm_eff_w']),
            q_norm_out.data_ptr(),
            K * 24, 256, eps, s,
        )
        k_pre_2d = k_pre.view(K * 4, 256)
        k_norm_out = self._K_full_k_norm_out[:K * 4]
        fvk.rms_norm(
            k_pre_2d.data_ptr(), int(lw['k_norm_eff_w']),
            k_norm_out.data_ptr(),
            K * 4, 256, eps, s,
        )

        # 6) RoPE inline over K positions (cos_K / sin_K shape (1, K, 64)).
        q_for_rope = q_norm_out.view(1, K, 24, 256)
        k_for_rope = k_norm_out.view(1, K, 4, 256)
        cos4 = cos_K.view(1, K, 1, 64)
        sin4 = sin_K.view(1, K, 1, 64)
        q_rot_K = self._K_full_q_rot[:, :K]
        k_rot_K = self._K_full_k_rot[:, :K]
        tmp_q = self._K_full_rope_tmp_q[:, :K]
        tmp_k = self._K_full_rope_tmp_k[:, :K]

        def _rope_inline_K(x_in, x_out, tmp):
            x_out[..., 64:].copy_(x_in[..., 64:])
            torch.index_select(
                x_in[..., :64], -1, self._rope_rotate_idx, out=tmp,
            )
            tmp[..., :32].neg_()
            tmp.mul_(sin4)
            tmp.addcmul_(x_in[..., :64], cos4)
            x_out[..., :64].copy_(tmp)

        _rope_inline_K(q_for_rope, q_rot_K, tmp_q)
        _rope_inline_K(k_for_rope, k_rot_K, tmp_k)

        # 7) Stage Q in backend Q_buf [:, :K]; write K rows of K/V.
        self._attn.Q_buf[:, :K].copy_(q_rot_K)
        self._attn.K_cache[full_rank, cur_pos:cur_pos + K].copy_(
            k_rot_K.view(K, 4, 256))

        # v_proj — M=K (overwrite kv_proj_out_buf, K already in cache).
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['v_proj_packed']),
            kv_proj_out_buf.data_ptr(),
            K, 1024, 5120,
            sf_5120.data_ptr(), int(lw['v_proj_sf']),
            float(lw['v_proj_alpha']),
            s,
        )
        v_new_K = kv_proj_out_buf[:K].view(K, 4, 256)
        self._attn.V_cache[full_rank, cur_pos:cur_pos + K].copy_(v_new_K)

        # N7-B4: TurboQuant inject for the K rows just written.
        self._tq_inject_kv(full_rank, cur_pos, count=K)

        # 8) FA2 — K serial q_seq=1 calls (fwd_bf16 has no causal flag).
        # Each call's kv_seq is bound by Q's own position so K[j>i] is
        # not read. Mirror of FP8 path (commit 2e16c11).
        scaling = float(self._cfg['head_dim']) ** -0.5
        for k in range(K):
            q_view = self._attn.Q_buf[:, k:k + 1]
            kv_seq_k = cur_pos + k + 1
            k_view = self._attn.K_cache[
                full_rank:full_rank + 1, :kv_seq_k]
            v_view = self._attn.V_cache[
                full_rank:full_rank + 1, :kv_seq_k]
            o_view = self._attn.O_buf[:, k:k + 1]
            self._attn._fa2_fwd(
                Q=q_view.data_ptr(), K=k_view.data_ptr(),
                V=v_view.data_ptr(), O=o_view.data_ptr(),
                softmax_lse=self._attn.lse_buf.data_ptr(),
                softmax_lse_accum=self._attn.lse_accum.data_ptr(),
                o_accum=self._attn.o_accum.data_ptr(),
                batch=1, seqlen_q=1, seqlen_k=kv_seq_k,
                num_heads_q=24, num_heads_kv=4,
                head_dim=256,
                q_strides=(q_view.stride(0), q_view.stride(1),
                           q_view.stride(2)),
                k_strides=(k_view.stride(0), k_view.stride(1),
                           k_view.stride(2)),
                v_strides=(v_view.stride(0), v_view.stride(1),
                           v_view.stride(2)),
                o_strides=(o_view.stride(0), o_view.stride(1),
                           o_view.stride(2)),
                softmax_scale=scaling,
                num_sms=self._attn._num_sms,
                stream=s,
            )
        attn_out = self._attn.O_buf[:, :K]  # (1, K, 24, 256)

        # 9) Output gate: attn * sigmoid(gate). K rows.
        attn_flat = attn_out.reshape(1, K, 24 * 256)
        gate_sig = self._K_full_gate_sig[:, :K]
        gated = self._K_full_gated[:, :K]
        torch.sigmoid(gate_flat, out=gate_sig)
        torch.mul(attn_flat, gate_sig, out=gated)

        # 10) o_proj NVFP4 — M=K, N=5120, K_in=6144.
        ap_6144, sf_6144, _ = self._nvfp4_scratch[(5120, 6144)]
        gated_2d = gated.view(K, 6144).contiguous()
        fvk.quantize_bf16_to_nvfp4_swizzled(
            gated_2d.data_ptr(), ap_6144.data_ptr(),
            sf_6144.data_ptr(), K, 6144, s,
        )
        out_op_buf = self._nvfp4_scratch[(5120, 6144)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_6144.data_ptr(), int(lw['o_proj_packed']),
            out_op_buf.data_ptr(),
            K, 5120, 6144,
            sf_6144.data_ptr(), int(lw['o_proj_sf']),
            float(lw['o_proj_alpha']),
            s,
        )

        # 11-12) Residual + post-attn norm + MLP NVFP4 quant — fused (B2).
        attn_proj = out_op_buf[:K].view(1, K, 5120)
        res_mid_K = self._K_res_mid[:, :K]
        ap_mlp, sf_mlp, _ = self._nvfp4_scratch[(17408, 5120)]
        fvk.residual_add_rms_norm_to_nvfp4_swizzled_bf16(
            h_in_K.data_ptr(), attn_proj.data_ptr(),
            res_mid_K.data_ptr(),
            int(lw['post_attn_norm_eff_w']),
            ap_mlp.data_ptr(), sf_mlp.data_ptr(),
            K, 5120, eps, s,
        )
        h_post = res_mid_K
        gate_out_buf = self._nvfp4_scratch[(17408, 5120)][2]
        up_out_buf = self._mlp_up_out
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_mlp.data_ptr(), int(lw['mlp_gate_packed']),
            gate_out_buf.data_ptr(),
            K, 17408, 5120,
            sf_mlp.data_ptr(), int(lw['mlp_gate_sf']),
            float(lw['mlp_gate_alpha']),
            s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_mlp.data_ptr(), int(lw['mlp_up_packed']),
            up_out_buf.data_ptr(),
            K, 17408, 5120,
            sf_mlp.data_ptr(), int(lw['mlp_up_sf']),
            float(lw['mlp_up_alpha']),
            s,
        )

        # 14) silu(gate) * up over K*17408 elements.
        gate_v = gate_out_buf[:K].view(K, 17408)
        up_v = up_out_buf[:K].view(K, 17408)
        silu_out = self._K_mlp_silu_mul_out[:K]
        fvk.silu_mul_qwen36_bf16(
            gate_v.data_ptr(), up_v.data_ptr(),
            silu_out.data_ptr(), K * 17408, s,
        )

        # 15) MLP down: NVFP4 quant + GEMM at M=K.
        ap_dn, sf_dn, _ = self._nvfp4_scratch[(5120, 17408)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            silu_out.data_ptr(), ap_dn.data_ptr(),
            sf_dn.data_ptr(), K, 17408, s,
        )
        down_out_buf = self._nvfp4_scratch[(5120, 17408)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_dn.data_ptr(), int(lw['mlp_down_packed']),
            down_out_buf.data_ptr(),
            K, 5120, 17408,
            sf_dn.data_ptr(), int(lw['mlp_down_sf']),
            float(lw['mlp_down_alpha']),
            s,
        )
        mlp_out = down_out_buf[:K].view(1, K, 5120)

        # 16) Final residual.
        h_out = (self._K_layer_out_a if (L % 2 == 0)
                 else self._K_layer_out_b)
        h_out_K = h_out[:, :K]
        torch.add(h_post, mlp_out, out=h_out_K)
        return h_out_K

    def forward_own_decode_nvfp4(self, token_id, cos_pos, sin_pos,
                                  cur_pos: int):
        """NVFP4 own-forward decode: 64 layers + final norm + lm_head.

        Mirrors ``forward_own_decode`` (FP8) but routes layers through
        the NVFP4 layer methods and reads embed / lm_head from
        anchored handles (no HF model object on this path).
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        bf16 = torch.bfloat16
        s = torch.cuda.current_stream().cuda_stream
        types = self._cfg['layer_types']
        eps = float(self._cfg['rms_norm_eps'])
        hidden = self._cfg['hidden_size']
        vocab = self._cfg['vocab_size']

        # Embedding lookup via anchored embed_w table.
        if not isinstance(token_id, torch.Tensor):
            token_id = torch.tensor(
                [token_id], device=self.device, dtype=torch.long)
        if token_id.ndim == 1:
            token_id = token_id.view(1, 1)
        # embed_w (vocab, hidden) bf16; anchors[0] is embed_w
        # (per extract_weights_nvfp4 ordering: top-level first).
        embed_w_ptr = self._weights.ptrs['embed_w']
        # Reconstruct an indexable view by walking handles.anchors —
        # the embed weight is the first anchor pushed in our loader.
        embed_t = self._weights.anchors[0]  # (vocab, hidden) bf16
        h = embed_t[token_id.view(-1)].view(1, 1, hidden).contiguous()
        if h.dtype != bf16:
            h = h.to(bf16)

        for L in range(self._cfg['num_hidden_layers']):
            t = types[L]
            if t == 'linear_attention':
                h = self._layer_forward_lin_nvfp4(L, h)
            elif t == 'full_attention':
                h = self._layer_forward_full_nvfp4(
                    L, h, cos_pos, sin_pos, cur_pos)
            else:
                raise ValueError(f'unknown layer_type {t!r} at L={L}')

        self._last_hidden_buf.copy_(h)
        h2 = h.view(1, hidden).contiguous()
        x_norm = self._h_b[:1].view(1, hidden)
        fvk.rms_norm(
            h2.data_ptr(), int(self._weights.ptrs['final_norm_eff_w']),
            x_norm.data_ptr(), 1, hidden, eps, s,
        )

        # G8: lm_head NVFP4 GEMM (was BF16 matvec). Reuse the K=5120
        # activation scratch from any existing (*, 5120) entry — they
        # all share the same swizzled SF layout.
        ap, sf, _ = self._nvfp4_scratch[(10240, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_norm.data_ptr(), ap.data_ptr(), sf.data_ptr(),
            1, hidden, s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap.data_ptr(), int(self._weights.ptrs['lm_head_packed']),
            self._logits_buf.data_ptr(),
            1, vocab, hidden,
            sf.data_ptr(), int(self._weights.ptrs['lm_head_sf']),
            float(self._weights.ptrs['lm_head_alpha']),
            s,
        )
        return self._logits_buf

    # ---------- N5-stage5: NVFP4 S=K decode (verify) ----------

    def forward_own_decode_K_nvfp4(self, token_ids_K, cos_K, sin_K,
                                    cur_pos: int, K: int,
                                    tap_buf=None):
        """NVFP4 S=K decode: 64 layers + final norm + lm_head at K rows.

        Mirror of FP8 ``forward_own_decode_K`` but routes through the
        NVFP4 S=K layer methods and reads embed / lm_head from the
        anchored handles (no HF model on this path). Used by the spec
        verify pass — K consecutive tokens at positions
        [cur_pos, cur_pos+K) flow through the network in one batched
        call, producing K rows of logits.

        Optional ``tap_buf`` (DFlash N6-A4): when not None, after each
        layer in ``_DFLASH_TAP_LAYERS`` the K-row pre-final-norm hidden
        state is copied into ``tap_buf[i, :K]`` (i = tap-layer index).
        Caller passes a (5, K_max, hidden) bf16 buffer; orchestration
        slices the row at the accepted-prefix position to feed the
        drafter's fc projection. No-op when ``tap_buf`` is None — main
        path stays bit-identical.
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        bf16 = torch.bfloat16
        s = torch.cuda.current_stream().cuda_stream
        types = self._cfg['layer_types']
        eps = float(self._cfg['rms_norm_eps'])
        hidden = self._cfg['hidden_size']
        vocab = self._cfg['vocab_size']

        # 0) Embed K tokens via anchored embed table.
        if not isinstance(token_ids_K, torch.Tensor):
            token_ids_K = torch.tensor(
                token_ids_K, device=self.device, dtype=torch.long)
        token_ids_K = token_ids_K.view(K)
        embed_t = self._weights.anchors[0]  # (vocab, hidden) bf16
        h = embed_t[token_ids_K].view(1, K, hidden).contiguous()
        if h.dtype != bf16:
            h = h.to(bf16)

        # 1) 64 decoder layers at S=K.
        for L in range(self._cfg['num_hidden_layers']):
            t = types[L]
            if t == 'linear_attention':
                h = self._layer_forward_lin_K_nvfp4(L, h, K)
            elif t == 'full_attention':
                h = self._layer_forward_full_K_nvfp4(
                    L, h, cos_K, sin_K, cur_pos, K)
            else:
                raise ValueError(f'unknown layer_type {t!r} at L={L}')
            # DFlash hidden-tap capture (no-op when tap_buf is None).
            # Inside captured graphs this conditional is evaluated at
            # capture time; if tap_buf was set then, the .copy_ kernels
            # are baked into the graph and replay automatically.
            if tap_buf is not None:
                tap_idx = self._DFLASH_TAP_INDEX.get(L, -1)
                if tap_idx >= 0:
                    tap_buf[tap_idx, :K].copy_(h.view(K, hidden))

        # 2) Stash pre-final-norm hidden so MTP head / chained spec
        # can consume per-row hiddens.
        self._K_last_hidden_buf[:, :K].copy_(h)

        # 3) Final RMSNorm M=K.
        h2 = h.view(K, hidden).contiguous()
        x_norm = self._h_b[:K].view(K, hidden)
        fvk.rms_norm(
            h2.data_ptr(), int(self._weights.ptrs['final_norm_eff_w']),
            x_norm.data_ptr(),
            K, hidden, eps, s,
        )

        # 4) G8: lm_head NVFP4 GEMM (M=K). NVFP4 weight is 4× smaller
        # than BF16 (0.6 GB vs 2.5 GB). Reads weight ONCE for all K
        # rows. Saves ~(K-1) × NVFP4_weight_read per verify forward.
        ap, sf, _ = self._nvfp4_scratch[(10240, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_norm.data_ptr(), ap.data_ptr(), sf.data_ptr(),
            K, hidden, s,
        )
        out_K = self._K_logits_buf[:K]
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap.data_ptr(), int(self._weights.ptrs['lm_head_packed']),
            out_K.data_ptr(),
            K, vocab, hidden,
            sf.data_ptr(), int(self._weights.ptrs['lm_head_sf']),
            float(self._weights.ptrs['lm_head_alpha']),
            s,
        )
        return self._K_logits_buf[:K]

    def forward_mtp_head_nvfp4(self, prev_h, prev_token_id, cur_pos: int):
        """NVFP4 MTP head forward (1-layer DeepSeek-V3 draft).

        Same math as ``forward_mtp_head`` (FP8 path) but every FP8 GEMM
        is replaced with the NVFP4 W4A16 GEMM, and per-token activation
        quant is the NVFP4 swizzled-SF quantizer instead of FP8 block
        quant. fc / norms / lm_head stay BF16. Per the user's no-FP8-
        mixing mandate.
        """
        import torch

        from flash_rt import flash_rt_kernels as fvk

        s = torch.cuda.current_stream().cuda_stream
        mtp = self._weights.ptrs.get('mtp')
        if mtp is None:
            raise RuntimeError(
                'MTP head not loaded — set FLASHRT_QWEN36_MTP_CKPT_DIR')
        eps = float(self._cfg['rms_norm_eps'])
        vocab = self._cfg['vocab_size']

        # 0) Embed prev_token via anchored embed_w table.
        if prev_token_id.ndim == 1:
            prev_token_id = prev_token_id.view(1, 1)
        embed_t = self._weights.anchors[0]   # (vocab, hidden) bf16
        e = embed_t[prev_token_id.view(-1)].view(1, 1, 5120).contiguous()
        if e.dtype != torch.bfloat16:
            e = e.to(torch.bfloat16)

        # 1) pre-fc norms.
        prev_h_2d = prev_h.view(1, 5120).contiguous()
        e_2d = e.view(1, 5120).contiguous()
        h_norm = self._mtp_h_norm_buf.view(1, 5120)
        e_norm = self._mtp_e_norm_buf.view(1, 5120)
        fvk.rms_norm(
            prev_h_2d.data_ptr(), int(mtp['pre_fc_norm_hidden_eff_w']),
            h_norm.data_ptr(), 1, 5120, eps, s,
        )
        fvk.rms_norm(
            e_2d.data_ptr(), int(mtp['pre_fc_norm_embedding_eff_w']),
            e_norm.data_ptr(), 1, 5120, eps, s,
        )

        # 2) cat [e_norm, h_norm].
        cat_buf = self._mtp_cat_buf.view(1, 10240)
        cat_buf[:, :5120].copy_(e_norm)
        cat_buf[:, 5120:].copy_(h_norm)

        # 3) fc: BF16 matvec, M=1, K=10240, N=5120.
        fc_out_2d = self._mtp_fc_out_buf.view(1, 5120)
        fvk.bf16_matvec_qwen36_bf16(
            cat_buf.data_ptr(), int(mtp['fc_w']),
            fc_out_2d.data_ptr(), 5120, 10240, s,
        )

        # 4) Full-attn layer body (NVFP4) on MTP-private KV cache.
        h_in_full = self._mtp_fc_out_buf
        cos, sin = self._rope_cos_sin(cur_pos)

        # 4a) input layernorm.
        h2 = h_in_full.view(1, 5120)
        x_norm = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            h2.data_ptr(), int(mtp['input_norm_eff_w']),
            x_norm.data_ptr(), 1, 5120, eps, s,
        )

        # 4b) NVFP4 quantize x_norm — reused for q/k/v.
        ap_5120, sf_5120, _ = self._nvfp4_scratch[(12288, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_norm.data_ptr(), ap_5120.data_ptr(),
            sf_5120.data_ptr(), 1, 5120, s,
        )

        # 4c) q_proj fused.
        q_proj_out_buf = self._nvfp4_scratch[(12288, 5120)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(mtp['q_proj_packed']),
            q_proj_out_buf.data_ptr(),
            1, 12288, 5120,
            sf_5120.data_ptr(), int(mtp['q_proj_sf']),
            float(mtp['q_proj_alpha']),
            s,
        )
        q_full = q_proj_out_buf[:1].view(1, 1, 24, 512)
        q_pre, gate = torch.chunk(q_full, 2, dim=-1)
        gate_flat = gate.reshape(1, 1, 24 * 256)

        # 4d) k_proj.
        kv_proj_out_buf = self._nvfp4_scratch[(1024, 5120)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(mtp['k_proj_packed']),
            kv_proj_out_buf.data_ptr(),
            1, 1024, 5120,
            sf_5120.data_ptr(), int(mtp['k_proj_sf']),
            float(mtp['k_proj_alpha']),
            s,
        )
        k_pre = kv_proj_out_buf[:1].view(1, 1, 4, 256).contiguous()

        # 4e) q_norm / k_norm.
        q_pre_2d = q_pre.contiguous().view(24, 256)
        fvk.rms_norm(
            q_pre_2d.data_ptr(), int(mtp['q_norm_eff_w']),
            self._full_q_norm_out.data_ptr(), 24, 256, eps, s,
        )
        k_pre_2d = k_pre.view(4, 256)
        fvk.rms_norm(
            k_pre_2d.data_ptr(), int(mtp['k_norm_eff_w']),
            self._full_k_norm_out.data_ptr(), 4, 256, eps, s,
        )

        # 4f) RoPE inline.
        q_for_rope = self._full_q_norm_out.view(1, 1, 24, 256)
        k_for_rope = self._full_k_norm_out.view(1, 1, 4, 256)
        cos4 = cos.view(1, 1, 1, 64)
        sin4 = sin.view(1, 1, 1, 64)

        def _rope_inline(x_in, x_out, tmp):
            x_out[..., 64:].copy_(x_in[..., 64:])
            torch.index_select(
                x_in[..., :64], -1, self._rope_rotate_idx, out=tmp,
            )
            tmp[..., :32].neg_()
            tmp.mul_(sin4)
            tmp.addcmul_(x_in[..., :64], cos4)
            x_out[..., :64].copy_(tmp)

        _rope_inline(q_for_rope, self._full_q_rot, self._full_rope_tmp_q)
        _rope_inline(k_for_rope, self._full_k_rot, self._full_rope_tmp_k)
        q_rot = self._full_q_rot
        k_rot = self._full_k_rot

        # 4g) Stage Q + write K to MTP cache.
        self._mtp_Q_buf[:, :1].copy_(q_rot)
        self._mtp_K_cache[cur_pos:cur_pos + 1].copy_(
            k_rot.view(1, 4, 256))

        # v_proj (NVFP4).
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(mtp['v_proj_packed']),
            kv_proj_out_buf.data_ptr(),
            1, 1024, 5120,
            sf_5120.data_ptr(), int(mtp['v_proj_sf']),
            float(mtp['v_proj_alpha']),
            s,
        )
        v_new = kv_proj_out_buf[:1].view(1, 4, 256)
        self._mtp_V_cache[cur_pos:cur_pos + 1].copy_(v_new)

        # 4h) FA2 on MTP cache.
        kv_seq = cur_pos + 1
        scaling = float(self._cfg['head_dim']) ** -0.5
        q_view = self._mtp_Q_buf[:, :1]
        k_view = self._mtp_K_cache[:kv_seq].view(1, kv_seq, 4, 256)
        v_view = self._mtp_V_cache[:kv_seq].view(1, kv_seq, 4, 256)
        o_view = self._mtp_O_buf[:, :1]
        self._attn._fa2_fwd(
            Q=q_view.data_ptr(), K=k_view.data_ptr(),
            V=v_view.data_ptr(), O=o_view.data_ptr(),
            softmax_lse=self._mtp_lse_buf.data_ptr(),
            softmax_lse_accum=self._mtp_lse_accum.data_ptr(),
            o_accum=self._mtp_o_accum.data_ptr(),
            batch=1, seqlen_q=1, seqlen_k=kv_seq,
            num_heads_q=24, num_heads_kv=4,
            head_dim=256,
            q_strides=(q_view.stride(0), q_view.stride(1),
                       q_view.stride(2)),
            k_strides=(k_view.stride(0), k_view.stride(1),
                       k_view.stride(2)),
            v_strides=(v_view.stride(0), v_view.stride(1),
                       v_view.stride(2)),
            o_strides=(o_view.stride(0), o_view.stride(1),
                       o_view.stride(2)),
            softmax_scale=scaling,
            num_sms=self._attn._num_sms,
            stream=s,
        )
        attn_out = self._mtp_O_buf[:, :1]

        # 4i) output gate.
        attn_flat = attn_out.reshape(1, 1, 24 * 256)
        torch.sigmoid(gate_flat, out=self._full_gate_sig)
        torch.mul(attn_flat, self._full_gate_sig, out=self._full_gated)
        gated = self._full_gated

        # 4j) o_proj NVFP4.
        ap_6144, sf_6144, _ = self._nvfp4_scratch[(5120, 6144)]
        gated_2d = gated.view(1, 6144).contiguous()
        fvk.quantize_bf16_to_nvfp4_swizzled(
            gated_2d.data_ptr(), ap_6144.data_ptr(),
            sf_6144.data_ptr(), 1, 6144, s,
        )
        out_op_buf = self._nvfp4_scratch[(5120, 6144)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_6144.data_ptr(), int(mtp['o_proj_packed']),
            out_op_buf.data_ptr(),
            1, 5120, 6144,
            sf_6144.data_ptr(), int(mtp['o_proj_sf']),
            float(mtp['o_proj_alpha']),
            s,
        )

        # 4k) residual.
        attn_proj = out_op_buf[:1].view(1, 1, 5120)
        torch.add(h_in_full, attn_proj, out=self._res_mid)
        h_post = self._res_mid

        # 4l) post-attn norm + MLP NVFP4.
        h_post_view = h_post.view(1, 5120)
        x_mlp = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            h_post_view.data_ptr(), int(mtp['post_attn_norm_eff_w']),
            x_mlp.data_ptr(), 1, 5120, eps, s,
        )
        ap_mlp, sf_mlp, _ = self._nvfp4_scratch[(17408, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_mlp.data_ptr(), ap_mlp.data_ptr(),
            sf_mlp.data_ptr(), 1, 5120, s,
        )
        gate_out_buf = self._nvfp4_scratch[(17408, 5120)][2]
        up_out_buf = self._mlp_up_out
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_mlp.data_ptr(), int(mtp['mlp_gate_packed']),
            gate_out_buf.data_ptr(),
            1, 17408, 5120,
            sf_mlp.data_ptr(), int(mtp['mlp_gate_sf']),
            float(mtp['mlp_gate_alpha']),
            s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_mlp.data_ptr(), int(mtp['mlp_up_packed']),
            up_out_buf.data_ptr(),
            1, 17408, 5120,
            sf_mlp.data_ptr(), int(mtp['mlp_up_sf']),
            float(mtp['mlp_up_alpha']),
            s,
        )
        gate_v = gate_out_buf[:1].view(1, 17408)
        up_v = up_out_buf[:1].view(1, 17408)
        fvk.silu_mul_qwen36_bf16(
            gate_v.data_ptr(), up_v.data_ptr(),
            self._mlp_silu_mul_out.data_ptr(), 17408, s,
        )
        gate_silu_up = self._mlp_silu_mul_out

        ap_dn, sf_dn, _ = self._nvfp4_scratch[(5120, 17408)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            gate_silu_up.data_ptr(), ap_dn.data_ptr(),
            sf_dn.data_ptr(), 1, 17408, s,
        )
        down_out_buf = self._nvfp4_scratch[(5120, 17408)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_dn.data_ptr(), int(mtp['mlp_down_packed']),
            down_out_buf.data_ptr(),
            1, 5120, 17408,
            sf_dn.data_ptr(), int(mtp['mlp_down_sf']),
            float(mtp['mlp_down_alpha']),
            s,
        )
        mlp_out = down_out_buf[:1].view(1, 1, 5120)

        # 4m) final residual.
        next_h = self._mtp_layer_out_buf
        torch.add(h_post, mlp_out, out=next_h)

        # 5) MTP final norm + lm_head.
        next_h_view = next_h.view(1, 5120)
        x_final_norm = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            next_h_view.data_ptr(), int(mtp['final_norm_eff_w']),
            x_final_norm.data_ptr(), 1, 5120, eps, s,
        )
        # G8: lm_head NVFP4 GEMM. Reuses (10240, 5120) scratch for
        # the K=5120 activation packed/SF buffers.
        ap_lm, sf_lm, _ = self._nvfp4_scratch[(10240, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_final_norm.data_ptr(), ap_lm.data_ptr(), sf_lm.data_ptr(),
            1, 5120, s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_lm.data_ptr(), int(self._weights.ptrs['lm_head_packed']),
            self._mtp_logits_buf.data_ptr(),
            1, vocab, 5120,
            sf_lm.data_ptr(), int(self._weights.ptrs['lm_head_sf']),
            float(self._weights.ptrs['lm_head_alpha']),
            s,
        )
        return next_h, self._mtp_logits_buf

    # ---------- N5-stage6: NVFP4 speculative decode (K-generic) ----------

    def generate_own_speculative_KN_nvfp4(
            self, input_ids, *, max_new_tokens: int, K: int = 5):
        """K-generic speculative decode on the NVFP4 path.

        In long-ctx mode (auto-routed when ``max_seq`` exceeds
        ``LONG_CTX_THRESHOLD`` at construction time) this method
        falls back to single-token decode through the TurboQuant
        packed cache — spec decode on the TQ path is a Phase 3D
        follow-up. The K argument is silently treated as 1; the
        caller does not need to know which path is active.

        Mirror of FP8 generate_own_speculative_KN. Differences vs. FP8:
          1. Prefill is DIY: walk prompt tokens through
             forward_own_decode_nvfp4 (S=1) to populate KV cache + lin
             state, capturing per-position pre-final-norm hidden into
             _prefill_h_cache. NVFP4 path has no HF AutoModel.
          2. MTP prefill: forward_mtp_head_nvfp4 over positions
             [1..prompt_len] using the captured prefill hiddens.
          3. Verify forward routes through forward_own_decode_K_nvfp4.
          4. MTP chain in spec loop calls forward_mtp_head_nvfp4
             directly (no graph yet — graph capture is N5-stage7).

        Cycle (same as FP8 path):
          - Snap lin/conv state + KV cache rows [cur_pos:cur_pos+K+1]
            on _snap_stream (overlaps with MTP chain on default stream).
          - Chain MTP K times -> drafts d_1..d_K.
          - forward_own_decode_K_nvfp4(K=K+1) over [tok, d_1..d_K] at
            cur_pos.
          - Argmax + accept-prefix N (largest prefix where draft ==
            verify argmax).
          - N==K: emit K+1 tokens, state correct, advance K+1.
          - N<K: emit N+1 tokens, restore state, re-advance with N+1
            valid inputs.

        Args:
            input_ids: (1, prompt_len) cuda long.
            max_new_tokens: how many tokens to generate.
            K: MTP chain length per cycle. K+1 must be <= MAX_Q_SEQ.
                Default 5 (Qwen3-Next official spec).

        Returns:
            (1, prompt_len + N) cuda long, trimmed to max_new_tokens.
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        # Long-ctx auto-route: TQ packed cache supports any context up
        # to ``self._user_max_seq``, but spec decode is not yet wired
        # on top of TQ (Phase 3D). Fall back to single-token decode —
        # caller does not need to know the path changed.
        if getattr(self, '_long_ctx_mode', False):
            return self._generate_long_ctx_single_token(
                input_ids, max_new_tokens)

        if self._weights.ptrs.get('mtp') is None:
            raise RuntimeError(
                'MTP head not loaded — speculative decode unavailable')
        if K < 1 or K + 1 > self.MAX_Q_SEQ:
            raise ValueError(
                f'K={K} out of range — need 1<=K<={self.MAX_Q_SEQ - 1}')

        bf16 = torch.bfloat16
        prompt_len = int(input_ids.shape[1])
        hidden = self._cfg['hidden_size']
        eps = float(self._cfg['rms_norm_eps'])

        self.reset_state()
        self.reset_mtp_state()
        if not hasattr(self, '_rope_cos_table'):
            self._build_rope_table()

        with torch.no_grad():
            # 1) Prefill: walk prompt tokens S=1 through main forward
            # using PER-CUR_POS captured graphs. Each step writes
            # pre-final-norm hidden to _last_hidden_buf; we capture
            # into _prefill_h_cache so MTP prefill can read
            # h_main_{p-1} below.
            #
            # NB: we deliberately do NOT use the batched
            # forward_own_decode_K_nvfp4 here even though it would
            # cut prefill BW by prompt_len × — its lin-layer FLA
            # chunk_gated_delta_rule has a different bf16 reduction
            # order than the S=1 single-step gated_deltanet_recurrent
            # kernel used at decode time. The MTP head trained
            # against the latter; feeding it FLA-chunk hiddens drops
            # AL from p_ind 0.75 → 0.51 at K=3 (verified empirically).
            # Prefill cost amortizes over output length anyway.
            gs_pf = self._graph_stream
            for p in range(prompt_len):
                self._static_token_id.copy_(input_ids[:, p:p + 1])
                g_pf = self._ensure_graph_for_pos_nvfp4(p)
                gs_pf.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(gs_pf):
                    g_pf.replay()
                torch.cuda.current_stream().wait_stream(gs_pf)
                self._prefill_h_cache[p:p + 1].copy_(
                    self._last_hidden_buf.view(1, hidden))
            # First decoded token = argmax of last prompt step's logits.
            logits_p = self._logits_buf
            tok = logits_p.argmax(dim=-1, keepdim=True).view(1, 1)
            generated = [tok]
            cur_pos = prompt_len

            # 2) MTP prefill: positions [1..prompt_len-1] use the
            # per-position captured hiddens; position prompt_len uses
            # the last prompt hidden + the just-predicted tok. Mirror
            # of FP8 path's MTP prefill semantics.
            for p in range(1, prompt_len):
                prev_h_p = self._prefill_h_cache[
                    p - 1:p].view(1, 1, hidden).contiguous()
                prev_tok_p = input_ids[:, p:p + 1]
                self.forward_mtp_head_nvfp4(prev_h_p, prev_tok_p, p)
            h_last_prompt = self._prefill_h_cache[
                prompt_len - 1:prompt_len].view(1, 1, hidden).contiguous()
            self.forward_mtp_head_nvfp4(h_last_prompt, tok, prompt_len)
            h = h_last_prompt

            self._spec_attempts = 0
            self._spec_accepts = 0
            self._spec_full = 0

            # 3) Spec decode loop.
            while len(generated) < max_new_tokens:
                # Snapshot main state on snap_stream first; overlaps
                # with MTP chain on default stream (independent state).
                snap_stream = self._snap_stream
                snap_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(snap_stream):
                    self._snap_lin_buf.copy_(self._lin_state)
                    self._snap_conv_buf.copy_(self._lin_conv_state)
                    self._snap_K_buf[:, :K + 1].copy_(
                        self._attn.K_cache[
                            :, cur_pos:cur_pos + K + 1])
                    self._snap_V_buf[:, :K + 1].copy_(
                        self._attn.V_cache[
                            :, cur_pos:cur_pos + K + 1])

                # G9: MTP chain — entire K-step chain captured as ONE
                # graph (forward × K + argmax × K + state copies × K-1).
                # One Python replay() instead of K replay()s + K argmax
                # calls + 2(K-1) inter-step copy_ launches.
                gs = self._graph_stream
                cg = self._ensure_mtp_chain_graph_nvfp4(cur_pos, K)
                gs.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(gs):
                    self._mtp_static_prev_h.copy_(h)
                    self._mtp_static_prev_token.copy_(tok)
                    cg.replay()
                torch.cuda.current_stream().wait_stream(gs)
                # drafts now sit in _chain_drafts_buf[:K] as (K, 1) long.
                drafts_t = self._chain_drafts_buf[:K]  # (K, 1)

                # Wait for snap before verify writes state.
                torch.cuda.current_stream().wait_stream(snap_stream)

                # Main S=K+1 verify via captured graph.
                # drafts_t is (K, 1) long; build verify input directly.
                d = self._rope_dim
                cos_KN = self._rope_cos_table[
                    cur_pos:cur_pos + K + 1].view(1, K + 1, d)
                sin_KN = self._rope_sin_table[
                    cur_pos:cur_pos + K + 1].view(1, K + 1, d)

                Kv = K + 1
                # Tokens layout in verify_static_tokens[:, :K+1]:
                # [tok, drafts_t[0], drafts_t[1], ..., drafts_t[K-1]]
                self._verify_static_tokens[:, 0:1].copy_(tok)
                self._verify_static_tokens[:, 1:Kv].copy_(
                    drafts_t.view(1, K))
                self._verify_static_cos[:, :Kv].copy_(cos_KN)
                self._verify_static_sin[:, :Kv].copy_(sin_KN)
                vg = self._ensure_verify_graph_nvfp4(cur_pos, Kv)
                gs.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(gs):
                    vg.replay()
                torch.cuda.current_stream().wait_stream(gs)
                logits_KN = self._K_logits_buf[:Kv]

                # Argmax + accept-prefix N (single-sync via sentinel).
                all_argmax = logits_KN.argmax(dim=-1)  # (K+1,) long
                drafts_stack = drafts_t.view(-1)
                matches = (all_argmax[:K] == drafts_stack).long()
                matches_pad = torch.cat([
                    matches,
                    torch.zeros(1, device=matches.device,
                                dtype=matches.dtype),
                ])
                N = int(matches_pad.argmin().item())
                self._spec_attempts += 1
                self._spec_accepts += N

                argmax_at = (lambda j: all_argmax[j:j + 1].view(1, 1))

                if N == K:
                    self._spec_full += 1
                    for j in range(K + 1):
                        if len(generated) < max_new_tokens:
                            generated.append(argmax_at(j))
                    tok = argmax_at(K)
                    h = self._K_last_hidden_buf[
                        :, K:K + 1, :].contiguous()
                    cur_pos += K + 1
                else:
                    for j in range(N + 1):
                        if len(generated) < max_new_tokens:
                            generated.append(argmax_at(j))
                    # A1'-S0: restore lin/conv state from per-step saves
                    # written DURING the verify K-iter recurrent loop.
                    # save[N] = state after processing tokens [tok,
                    # d_1, ..., d_N] = exactly the state we need for
                    # cur_pos+N+1. KV cache positions cur_pos+N+1..
                    # cur_pos+K hold stale (rejected) entries that are
                    # harmlessly overwritten by the next cycle's writes
                    # before any read (FA2 writes BEFORE reading at each
                    # q_seq=1 step).
                    self._lin_state.copy_(
                        self._K_lin_state_per_step[N])
                    self._lin_conv_state.copy_(
                        self._K_lin_conv_state_per_step[N])
                    h = self._K_last_hidden_buf[
                        :, N:N + 1, :].contiguous()
                    tok = argmax_at(N)
                    cur_pos += N + 1

            if len(generated) > max_new_tokens:
                generated = generated[:max_new_tokens]

        return torch.cat([input_ids] + generated, dim=1)

    # ---------- own forward (Phase 2.3b4) ----------
    #
    # Forward path implemented method-by-method on the frontend, not on
    # the pipeline class -- the frontend owns both the weights handles
    # and the pre-allocated buffers, so keeping the forward here avoids
    # passing six pointer dicts around per call. The pipeline class will
    # gain a ``forward_own(input_ids)`` wrapper in step 6 that dispatches
    # to these methods.

    def _layer_forward_lin(self, L: int, h_in):
        """Run linear-attention decoder layer L on a single token.

        Replaces ``Qwen3_5DecoderLayer.forward`` for layers whose
        ``layer_type == "linear_attention"``. Decode-mode only (B=1, S=1).

        Math (matches HF byte-for-byte using the patched fvk kernels --
        see transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer
        / Qwen3_5GatedDeltaNet):

            residual = h_in
            x  = rms_norm(h_in, input_norm_eff_w)
            qkv = in_proj_qkv(x)            # FP8 GEMM, (1, 1, 10240)
            z   = in_proj_z(x)              # FP8 GEMM, (1, 1, 6144)
            b   = in_proj_b(x)              # bf16 GEMM,(1, 1, 48)
            a   = in_proj_a(x)              # bf16 GEMM,(1, 1, 48)
            qkv = causal_conv1d_update(qkv, conv_state, conv_w, conv_b, silu)
            q,k,v = split(qkv, [2048, 2048, 6144])
            q = q.view(1,1,16,128); k = q-shape; v = (1,1,48,128)
            beta = sigmoid(b)
            g = -A_log.exp() * softplus(a + dt_bias)        (fp32)
            q = q.repeat_interleave(3, dim=2)               (16->48 broadcast)
            k = k.repeat_interleave(3, dim=2)
            attn_out, new_state = recurrent_gated_delta_rule(
                q, k, v, g, beta, recurrent_state, l2norm=True)
            attn_out = rms_norm_gated_silu(
                attn_out.reshape(48,128), z.reshape(48,128), head_norm_w)
            attn_out = attn_out.view(1,1,6144)
            attn_out = out_proj(attn_out)   # FP8 GEMM, (1, 1, 5120)
            h_post = residual + attn_out

            residual = h_post
            x = rms_norm(h_post, post_attn_norm_eff_w)
            gate_o = mlp.gate_proj(x)       # FP8 GEMM, (1, 1, 17408)
            up_o   = mlp.up_proj(x)         # FP8 GEMM, (1, 1, 17408)
            mlp_o  = mlp.down_proj(silu(gate_o) * up_o)   # FP8 GEMM, (1, 1, 5120)
            return h_post + mlp_o

        Args:
            L: decoder layer index (0..63), must be a linear-attn layer.
            h_in: (1, 1, 5120) bf16 cuda tensor -- input to the layer.

        Returns:
            (1, 1, 5120) bf16 cuda tensor -- output of the layer.
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        bf16 = torch.bfloat16
        # Pass current torch stream to all fvk kernels so they launch
        # on the same stream as the (potential) graph-capture context.
        # See _alloc_buffers's _graph_stream comment for the why.
        s = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        assert lw['type'] == 'linear_attention', (
            f'_layer_forward_lin called on layer {L} of type {lw["type"]!r}'
        )

        h2 = h_in.view(1, 5120).contiguous()  # (1, 5120) bf16
        eps = float(self._pipeline.hf.config.rms_norm_eps)

        # Buffers / scratch.
        x_norm = self._h_b[:1]               # (1, 5120) bf16
        x_norm_view = x_norm.view(1, 5120)

        qinp_5120, scale_5120, _ = self._fp8_scratch[(10240, 5120)]
        # Reuse the K=5120 quant scratch across in_proj_qkv / in_proj_z /
        # in_proj_a / in_proj_b. The FP8 a/b paths are bf16 weight so we
        # don't need fp8 quant for them.
        out_qkv_buf = self._fp8_scratch[(10240, 5120)][2]
        out_z_buf = self._fp8_scratch[(6144, 5120)][2]
        out_op_buf = self._fp8_scratch[(5120, 6144)][2]

        # 1) input layernorm.
        fvk.rms_norm(
            h2.data_ptr(), int(lw['input_norm_eff_w']),
            x_norm_view.data_ptr(),
            1, 5120, eps, s,
        )

        # 2) FP8 quant (M=1, K=5120).
        fvk.fp8_per_token_block128_quant_bf16(
            x_norm_view.data_ptr(), qinp_5120.data_ptr(),
            scale_5120.data_ptr(), 1, 5120, s,
        )

        # 3) in_proj_qkv -> (1, 10240) bf16.
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['in_proj_qkv_w']),
            out_qkv_buf.data_ptr(),
            1, 10240, 5120,
            scale_5120.data_ptr(), int(lw['in_proj_qkv_s']),
            s,
        )
        # 4) in_proj_z -> (1, 6144) bf16.
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['in_proj_z_w']),
            out_z_buf.data_ptr(),
            1, 6144, 5120,
            scale_5120.data_ptr(), int(lw['in_proj_z_s']),
            s,
        )
        # 5) in_proj_a / in_proj_b: bf16 weights, M=1 N=48 K=5120.
        # Use the stream-invariant fvk bf16 matvec (NOT F.linear /
        # cuBLASLt) so the kernel choice is identical across eager,
        # different streams, and CUDA Graph capture context. F.linear
        # via cuBLASLt picks per-stream/per-context algorithms whose
        # bf16 reductions differ -- breaks graph capture correctness.
        la = self._pipeline.hf.model.layers[L].linear_attn
        a_vec = self._lin_a_vec
        b_vec = self._lin_b_vec
        fvk.bf16_matvec_qwen36_bf16(
            x_norm.data_ptr(),
            la.in_proj_a.weight.data_ptr(),
            a_vec.data_ptr(), 48, 5120, s,
        )
        fvk.bf16_matvec_qwen36_bf16(
            x_norm.data_ptr(),
            la.in_proj_b.weight.data_ptr(),
            b_vec.data_ptr(), 48, 5120, s,
        )

        # 6) causal_conv1d_update on the qkv stream (1, conv_dim=10240).
        qkv_in = out_qkv_buf[:1].view(1, 10240)         # (1, 10240) bf16
        # conv1d state buffer for layer L (slice by layer index of the
        # 48-layer linear-attn cache). Layer ordering in our cache is by
        # *linear-attn position*, not original model index -- we need
        # the linear-attn rank. config.layer_types[L] tells if linear,
        # and the rank = #linear-attn layers among layers[0..L-1].
        lin_rank = self._linear_layer_rank(L)
        conv_state = self._lin_conv_state[lin_rank]      # (1, 10240, 3) bf16
        rec_state = self._lin_state[lin_rank]            # (1, 48, 128, 128)

        conv_out = self._lin_conv_out   # (1, 10240) dedicated scratch

        fvk.causal_conv1d_qwen36_update_bf16(
            qkv_in.data_ptr(), int(lw['conv1d_w']),
            int(lw['conv1d_b']),  # 0 if no bias
            conv_out.data_ptr(), conv_state.data_ptr(),
            1, 10240, 4, True, s,
        )

        # 7) split conv_out -> q (2048), k (2048), v (6144).
        # Layout of conv_dim=10240 = [K_dim=2048, K_dim=2048, V_dim=6144].
        q_flat = conv_out[:, :2048]            # (1, 2048)
        k_flat = conv_out[:, 2048:4096]
        v_flat = conv_out[:, 4096:10240]
        q = q_flat.view(1, 1, 16, 128)
        k = k_flat.view(1, 1, 16, 128)
        v = v_flat.view(1, 1, 48, 128)

        # 8) beta = sigmoid(b); g = -A_log.exp() * softplus(a + dt_bias).
        # Phase 4.4 step 2: zero-alloc form. Uses extractor-precomputed
        # `dt_bias_fp32_t` and `neg_A_log_exp_fp32_t` tensor handles
        # (constant per layer, anchored at frontend load) and a manual
        # softplus = log1p(exp(x)) into a pre-alloc buf. Eliminates the
        # 3 PyTorch elementwise allocations (la.dt_bias.float(),
        # F.softplus(...), -la.A_log.float().exp()) that previously hit
        # the per-call allocator and made the captured kernel see
        # graph-private-pool addresses different from the eager pool.
        torch.sigmoid(b_vec, out=self._lin_beta)
        self._lin_a_f32.copy_(a_vec)              # bf16 -> fp32 widening
        self._lin_a_f32.add_(lw['dt_bias_fp32_t'])  # +dt_bias broadcast
        # softplus(x) = log1p(exp(x)) -- stable for the practical range
        # (a_vec + dt_bias is ~O(1..few) in trained models; if any
        # prompt drives x > 20 we must add the threshold fallback).
        torch.exp(self._lin_a_f32, out=self._lin_sp_buf)
        torch.log1p(self._lin_sp_buf, out=self._lin_sp_buf)
        torch.mul(lw['neg_A_log_exp_fp32_t'], self._lin_sp_buf,
                  out=self._lin_g_f32)
        self._lin_g_bf.copy_(self._lin_g_f32)
        beta = self._lin_beta
        g_bf = self._lin_g_bf

        # 9) Broadcast q, k from 16 heads to 48 (interleave 3x).
        # In-place via torch.index_select with the precomputed broadcast
        # index (no Python alloc; pre-allocated _lin_q48/_lin_k48 targets).
        q_2d = q.view(1, 16, 128)
        k_2d = k.view(1, 16, 128)
        torch.index_select(q_2d, 1, self._lin_broadcast_idx, out=self._lin_q48)
        torch.index_select(k_2d, 1, self._lin_broadcast_idx, out=self._lin_k48)
        q3 = self._lin_q48
        k3 = self._lin_k48
        v3 = v.view(1, 48, 128).contiguous()
        attn_out_buf = self._lin_attn_out  # (1, 48, 128) dedicated scratch

        fvk.gated_deltanet_recurrent_qwen36_bf16(
            q3.data_ptr(), k3.data_ptr(), v3.data_ptr(),
            g_bf.data_ptr(), beta.data_ptr(),
            rec_state.data_ptr(), attn_out_buf.data_ptr(),
            1, 48, 128, 128, True, s,
        )

        # 11) rms_norm_gated_silu over (M=48, dim=128).
        z_flat = out_z_buf[:1].view(48, 128)   # z is (1, 6144) -> (48, 128)
        attn_out_flat = attn_out_buf.view(48, 128)
        norm_out = self._lin_norm_out   # (48, 128) dedicated scratch

        fvk.rms_norm_gated_silu_qwen36_bf16(
            attn_out_flat.data_ptr(), z_flat.data_ptr(),
            int(lw['head_norm_w']),
            norm_out.data_ptr(),
            48, 128, eps, s,
        )

        # 12) out_proj FP8: quantize (M=1, K=6144), then GEMM N=5120.
        qinp_6144, scale_6144, _ = self._fp8_scratch[(5120, 6144)]
        norm_out_1x6144 = norm_out.view(1, 6144)
        fvk.fp8_per_token_block128_quant_bf16(
            norm_out_1x6144.data_ptr(),
            qinp_6144.data_ptr(),
            scale_6144.data_ptr(),
            1, 6144, s,
        )
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_6144.data_ptr(), int(lw['out_proj_w']),
            out_op_buf.data_ptr(),
            1, 5120, 6144,
            scale_6144.data_ptr(), int(lw['out_proj_s']),
            s,
        )

        # 13) residual: h_post = h_in + attn_out (in-place, write to _res_mid)
        attn_proj = out_op_buf[:1].view(1, 1, 5120)
        torch.add(h_in, attn_proj, out=self._res_mid)
        h_post = self._res_mid

        # 14) post-attn layernorm + MLP swiglu + residual.
        h_post_view = h_post.view(1, 5120)
        x_mlp = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            h_post_view.data_ptr(), int(lw['post_attn_norm_eff_w']),
            x_mlp.data_ptr(),
            1, 5120, eps, s,
        )

        # MLP gate / up: same K=5120, can share quant scratch.
        fvk.fp8_per_token_block128_quant_bf16(
            x_mlp.data_ptr(), qinp_5120.data_ptr(),
            scale_5120.data_ptr(), 1, 5120, s,
        )
        gate_out_buf = self._fp8_scratch[(17408, 5120)][2]
        up_out_buf = self._mlp_up_out
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['mlp_gate_w']),
            gate_out_buf.data_ptr(),
            1, 17408, 5120,
            scale_5120.data_ptr(), int(lw['mlp_gate_s']),
            s,
        )
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['mlp_up_w']),
            up_out_buf.data_ptr(),
            1, 17408, 5120,
            scale_5120.data_ptr(), int(lw['mlp_up_s']),
            s,
        )

        # silu(gate) * up via fvk kernel (one launch, zero allocs).
        gate_v = gate_out_buf[:1].view(1, 17408)
        up_v = up_out_buf[:1].view(1, 17408)
        fvk.silu_mul_qwen36_bf16(
            gate_v.data_ptr(), up_v.data_ptr(),
            self._mlp_silu_mul_out.data_ptr(), 17408, s,
        )
        gate_silu_up = self._mlp_silu_mul_out

        # MLP down: K=17408 -> N=5120.
        qinp_17408, scale_17408, _ = self._fp8_scratch[(5120, 17408)]
        fvk.fp8_per_token_block128_quant_bf16(
            gate_silu_up.data_ptr(), qinp_17408.data_ptr(),
            scale_17408.data_ptr(), 1, 17408, s,
        )
        down_out_buf = self._fp8_scratch[(5120, 17408)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_17408.data_ptr(), int(lw['mlp_down_w']),
            down_out_buf.data_ptr(),
            1, 5120, 17408,
            scale_17408.data_ptr(), int(lw['mlp_down_s']),
            s,
        )
        mlp_out = down_out_buf[:1].view(1, 1, 5120)

        # 15) final residual: write to ping-pong layer-output buf.
        h_out = self._layer_out_a if (L % 2 == 0) else self._layer_out_b
        torch.add(h_post, mlp_out, out=h_out)
        return h_out

    # ---------- Phase 6 D4: S=K linear-attn layer ----------

    def _layer_forward_lin_K(self, L: int, h_in_K, K: int):
        """S=K linear-attention decoder layer (Phase 6 D4 verify path).

        The recurrent state and conv-1d state mutate sequentially per
        token, so the inner per-token block (conv1d_update -> split ->
        broadcast -> softplus chain -> gated_deltanet_recurrent ->
        rms_norm_gated_silu) runs in a K-iter loop. Everything that
        is row-parallel — input layernorm, in_proj_qkv/z FP8 GEMMs,
        out_proj FP8 GEMM, MLP — runs once with M=K.

        Args:
            L: linear-attn layer index.
            h_in_K: (1, K, 5120) bf16.
            K: 1 <= K <= MAX_Q_SEQ.

        Returns:
            (1, K, 5120) bf16 — output, written to _K layer-out
            ping-pong (slice [:, :K]).
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        s = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        assert lw['type'] == 'linear_attention', (
            f'_layer_forward_lin_K called on layer {L} of type '
            f'{lw["type"]!r}'
        )
        eps = float(self._pipeline.hf.config.rms_norm_eps)

        h2 = h_in_K.view(K, 5120).contiguous()

        # Buffers / scratch.
        x_norm = self._h_b[:K].view(K, 5120)

        qinp_5120, scale_5120, _ = self._fp8_scratch[(10240, 5120)]
        out_qkv_buf = self._fp8_scratch[(10240, 5120)][2]
        out_z_buf = self._fp8_scratch[(6144, 5120)][2]
        out_op_buf = self._fp8_scratch[(5120, 6144)][2]

        # 1) input layernorm M=K.
        fvk.rms_norm(
            h2.data_ptr(), int(lw['input_norm_eff_w']),
            x_norm.data_ptr(),
            K, 5120, eps, s,
        )

        # 2) FP8 quant M=K.
        fvk.fp8_per_token_block128_quant_bf16(
            x_norm.data_ptr(), qinp_5120.data_ptr(),
            scale_5120.data_ptr(), K, 5120, s,
        )

        # 3) in_proj_qkv -> (K, 10240).
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['in_proj_qkv_w']),
            out_qkv_buf.data_ptr(),
            K, 10240, 5120,
            scale_5120.data_ptr(), int(lw['in_proj_qkv_s']),
            s,
        )
        # 4) in_proj_z -> (K, 6144).
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['in_proj_z_w']),
            out_z_buf.data_ptr(),
            K, 6144, 5120,
            scale_5120.data_ptr(), int(lw['in_proj_z_s']),
            s,
        )

        # 5) in_proj_a / in_proj_b: K matvec calls (M=1 each, K times).
        # in_proj_a/b weights are bf16 (48, 5120). bf16_matvec_qwen36
        # is M=1; loop K times into K rows of _K_lin_a_vec / _K_lin_b_vec.
        la = self._pipeline.hf.model.layers[L].linear_attn
        a_vec_K = self._K_lin_a_vec[:K]
        b_vec_K = self._K_lin_b_vec[:K]
        for k in range(K):
            x_row = x_norm[k:k + 1]
            fvk.bf16_matvec_qwen36_bf16(
                x_row.data_ptr(),
                la.in_proj_a.weight.data_ptr(),
                a_vec_K[k:k + 1].data_ptr(), 48, 5120, s,
            )
            fvk.bf16_matvec_qwen36_bf16(
                x_row.data_ptr(),
                la.in_proj_b.weight.data_ptr(),
                b_vec_K[k:k + 1].data_ptr(), 48, 5120, s,
            )

        # 6) Per-token conv1d_update (state evolves), then a SINGLE
        # FLA chunk_gated_delta_rule call processing all K tokens at
        # once (~K× faster than K sequential single-step recurrent
        # calls), then a SINGLE rms_norm_gated_silu over K*48 rows.
        # This replaces the K-iter inner loop that was the dominant
        # cost at S=K (~3.9 ms/row × K compute, see profile probe).
        lin_rank = self._linear_layer_rank(L)
        conv_state = self._lin_conv_state[lin_rank]
        qkv_K_view = out_qkv_buf[:K]      # (K, 10240)

        # 6a) Sequential conv1d_update for K tokens — kernel mutates
        # conv_state in place, so we still loop. Each call is fast
        # (~5 us GPU work, mostly launch overhead).
        for k in range(K):
            qkv_row = qkv_K_view[k:k + 1]
            conv_out_row = self._K_lin_conv_out[k:k + 1]
            fvk.causal_conv1d_qwen36_update_bf16(
                qkv_row.data_ptr(), int(lw['conv1d_w']),
                int(lw['conv1d_b']),
                conv_out_row.data_ptr(), conv_state.data_ptr(),
                1, 10240, 4, True, s,
            )

        # 6b) Split K-token conv output into Q (K, 16, 128), K (same),
        # V (K, 48, 128). Layout in conv_out: [Q_dim=2048, K_dim=2048,
        # V_dim=6144] per row.
        conv_K = self._K_lin_conv_out[:K]  # (K, 10240)
        q_K_heads = conv_K[:, :2048].contiguous().view(1, K, 16, 128)
        k_K_heads = conv_K[:, 2048:4096].contiguous().view(1, K, 16, 128)
        v_K_heads = conv_K[:, 4096:10240].contiguous().view(1, K, 48, 128)

        # 6c) Compute g, beta for all K tokens at once (M=K vector ops).
        beta_K = self._K_lin_beta[:K]
        torch.sigmoid(b_vec_K, out=beta_K)
        a_f32_K = self._K_lin_a_f32[:K]
        a_f32_K.copy_(a_vec_K)
        a_f32_K.add_(lw['dt_bias_fp32_t'])
        sp_K = self._K_lin_sp_buf[:K]
        torch.exp(a_f32_K, out=sp_K)
        torch.log1p(sp_K, out=sp_K)
        g_f32_K = self._K_lin_g_f32[:K]
        torch.mul(lw['neg_A_log_exp_fp32_t'], sp_K, out=g_f32_K)
        g_bf_K = self._K_lin_g_bf[:K]
        g_bf_K.copy_(g_f32_K)

        # 6d) FLA chunked recurrent — one call processes all K tokens
        # natively, with internal L2-norm on q/k. GVA (16 K-heads,
        # 48 V-heads) handled by the kernel without needing the
        # 16->48 explicit broadcast.
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        rec_state_view = self._lin_state[lin_rank]  # (1, 48, 128, 128)
        beta_FLA = beta_K.view(1, K, 48)
        g_FLA = g_bf_K.view(1, K, 48)
        attn_out_K, new_state = chunk_gated_delta_rule(
            q_K_heads, k_K_heads, v_K_heads,
            g_FLA, beta_FLA,
            initial_state=rec_state_view,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # attn_out_K: (1, K, 48, 128); new_state: (1, 48, 128, 128).
        rec_state_view.copy_(new_state)

        # 6e) rms_norm_gated_silu over (K*48, 128) rows in one call.
        attn_out_flat = attn_out_K.contiguous().view(K * 48, 128)
        z_flat = out_z_buf[:K].view(K * 48, 128)
        norm_out_flat = self._K_lin_norm_out[:K].view(K * 48, 128)
        fvk.rms_norm_gated_silu_qwen36_bf16(
            attn_out_flat.data_ptr(), z_flat.data_ptr(),
            int(lw['head_norm_w']),
            norm_out_flat.data_ptr(),
            K * 48, 128, eps, s,
        )

        # 7) out_proj: K rows of (48, 128) -> (K, 6144).
        norm_out_K = self._K_lin_norm_out[:K].view(K, 6144)
        qinp_6144, scale_6144, _ = self._fp8_scratch[(5120, 6144)]
        fvk.fp8_per_token_block128_quant_bf16(
            norm_out_K.data_ptr(), qinp_6144.data_ptr(),
            scale_6144.data_ptr(), K, 6144, s,
        )
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_6144.data_ptr(), int(lw['out_proj_w']),
            out_op_buf.data_ptr(),
            K, 5120, 6144,
            scale_6144.data_ptr(), int(lw['out_proj_s']),
            s,
        )

        # 8) residual: h_post = h_in_K + attn_proj.
        attn_proj = out_op_buf[:K].view(1, K, 5120)
        res_mid_K = self._K_res_mid[:, :K]
        torch.add(h_in_K, attn_proj, out=res_mid_K)
        h_post = res_mid_K

        # 9) post-attn layernorm + MLP swiglu + residual (M=K).
        h_post_view = h_post.view(K, 5120)
        x_mlp = self._h_b[:K].view(K, 5120)
        fvk.rms_norm(
            h_post_view.data_ptr(), int(lw['post_attn_norm_eff_w']),
            x_mlp.data_ptr(),
            K, 5120, eps, s,
        )
        fvk.fp8_per_token_block128_quant_bf16(
            x_mlp.data_ptr(), qinp_5120.data_ptr(),
            scale_5120.data_ptr(), K, 5120, s,
        )
        gate_out_buf = self._fp8_scratch[(17408, 5120)][2]
        up_out_buf = self._mlp_up_out
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['mlp_gate_w']),
            gate_out_buf.data_ptr(),
            K, 17408, 5120,
            scale_5120.data_ptr(), int(lw['mlp_gate_s']),
            s,
        )
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['mlp_up_w']),
            up_out_buf.data_ptr(),
            K, 17408, 5120,
            scale_5120.data_ptr(), int(lw['mlp_up_s']),
            s,
        )
        gate_v = gate_out_buf[:K].view(K, 17408)
        up_v = up_out_buf[:K].view(K, 17408)
        silu_out = self._K_mlp_silu_mul_out[:K]
        fvk.silu_mul_qwen36_bf16(
            gate_v.data_ptr(), up_v.data_ptr(),
            silu_out.data_ptr(), K * 17408, s,
        )
        gate_silu_up = silu_out
        qinp_17408, scale_17408, _ = self._fp8_scratch[(5120, 17408)]
        fvk.fp8_per_token_block128_quant_bf16(
            gate_silu_up.data_ptr(), qinp_17408.data_ptr(),
            scale_17408.data_ptr(), K, 17408, s,
        )
        down_out_buf = self._fp8_scratch[(5120, 17408)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_17408.data_ptr(), int(lw['mlp_down_w']),
            down_out_buf.data_ptr(),
            K, 5120, 17408,
            scale_17408.data_ptr(), int(lw['mlp_down_s']),
            s,
        )
        mlp_out = down_out_buf[:K].view(1, K, 5120)

        # 10) final residual: write to _K layer-out ping-pong.
        h_out = (self._K_layer_out_a if (L % 2 == 0)
                 else self._K_layer_out_b)
        h_out_K = h_out[:, :K]
        torch.add(h_post, mlp_out, out=h_out_K)
        return h_out_K

    def _layer_forward_full(self, L: int, h_in, cos, sin, cur_pos: int):
        """Run full-attention decoder layer L on a single new token.

        Replaces ``Qwen3_5DecoderLayer.forward`` for layers whose
        ``layer_type == "full_attention"``. Decode-mode only (B=1, S=1).

        Args:
            L: original layer index (must be a full_attention layer).
            h_in: (1, 1, 5120) bf16 cuda tensor -- input to the layer.
            cos: rotary cos for position cur_pos, shape (1, 1, 256) bf16.
            sin: rotary sin for position cur_pos, shape (1, 1, 256) bf16.
            cur_pos: integer position of the new token, used as the
                row index for KV cache write. Must equal the number of
                tokens already in K/V before this call.

        Returns:
            (1, 1, 5120) bf16 cuda tensor -- output of the layer.
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        bf16 = torch.bfloat16
        s = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        assert lw['type'] == 'full_attention', (
            f'_layer_forward_full called on layer {L} of type '
            f'{lw["type"]!r}'
        )

        h2 = h_in.view(1, 5120).contiguous()
        eps = float(self._pipeline.hf.config.rms_norm_eps)
        full_rank = self._full_layer_rank(L)

        # 1) input layernorm.
        x_norm = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            h2.data_ptr(), int(lw['input_norm_eff_w']),
            x_norm.data_ptr(),
            1, 5120, eps, s,
        )

        # 2) FP8 quant (M=1, K=5120). Reuse the (12288, 5120) tuple's
        # qinput/scale -- same K, same input.
        qinp_5120, scale_5120, _ = self._fp8_scratch[(12288, 5120)]
        fvk.fp8_per_token_block128_quant_bf16(
            x_norm.data_ptr(), qinp_5120.data_ptr(),
            scale_5120.data_ptr(), 1, 5120, s,
        )

        # 3) q_proj fused (Q + output_gate) -> (1, 12288).
        q_proj_out_buf = self._fp8_scratch[(12288, 5120)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['q_proj_w']),
            q_proj_out_buf.data_ptr(),
            1, 12288, 5120,
            scale_5120.data_ptr(), int(lw['q_proj_s']),
            s,
        )

        # 4) Split q_proj into Q + output_gate per HF:
        #     q_proj.view(B, S, num_q_heads, head_dim*2).chunk(2, dim=-1)
        q_full = q_proj_out_buf[:1].view(1, 1, 24, 512)
        q_pre, gate = torch.chunk(q_full, 2, dim=-1)
        # q_pre: (1, 1, 24, 256); gate: (1, 1, 24, 256)
        gate_flat = gate.reshape(1, 1, 24 * 256)         # (1, 1, 6144)

        # 5) k_proj -> (1, 1024). Same K=5120 quant, distinct N=1024.
        kv_proj_out_buf = self._fp8_scratch[(1024, 5120)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['k_proj_w']),
            kv_proj_out_buf.data_ptr(),
            1, 1024, 5120,
            scale_5120.data_ptr(), int(lw['k_proj_s']),
            s,
        )
        k_pre = kv_proj_out_buf[:1].view(1, 1, 4, 256).contiguous()

        # 6) q_norm / k_norm: head_dim RMSNorm (1+w precomputed).
        q_pre_2d = q_pre.contiguous().view(24, 256)
        fvk.rms_norm(
            q_pre_2d.data_ptr(), int(lw['q_norm_eff_w']),
            self._full_q_norm_out.data_ptr(),
            24, 256, eps, s,
        )
        k_pre_2d = k_pre.view(4, 256)
        fvk.rms_norm(
            k_pre_2d.data_ptr(), int(lw['k_norm_eff_w']),
            self._full_k_norm_out.data_ptr(),
            4, 256, eps, s,
        )

        # 7) RoPE inline on (B=1, S=1, H, D), partial_rotary_factor=0.25.
        # cos/sin shape (1, 1, 64); unsqueeze_dim=2 -> (1, 1, 1, 64) broadcasts.
        # rotated[i] = q[i]*cos[i] + rotate_half(q[i])*sin[i] for i in [0,64)
        # passthrough = q[..., 64:] unchanged.
        # rotate_half via torch.index_select(idx=[32..63, 0..31], out=tmp)
        # then negate tmp[..., :32] in-place.
        q_for_rope = self._full_q_norm_out.view(1, 1, 24, 256)
        k_for_rope = self._full_k_norm_out.view(1, 1, 4, 256)
        cos4 = cos.view(1, 1, 1, 64)
        sin4 = sin.view(1, 1, 1, 64)

        def _rope_inline(x_in, x_out, tmp):
            # passthrough cols 64..256
            x_out[..., 64:].copy_(x_in[..., 64:])
            # tmp = rotate_half(x_in[..., :64])
            torch.index_select(
                x_in[..., :64], -1, self._rope_rotate_idx, out=tmp,
            )
            tmp[..., :32].neg_()
            # Compute into the contiguous tmp buf, then copy to x_out's
            # strided last-dim slice. Avoids the non-contiguous addcmul_
            # path which appears to be lossy on prompt 2.
            tmp.mul_(sin4)
            tmp.addcmul_(x_in[..., :64], cos4)
            x_out[..., :64].copy_(tmp)

        _rope_inline(q_for_rope, self._full_q_rot, self._full_rope_tmp_q)
        _rope_inline(k_for_rope, self._full_k_rot, self._full_rope_tmp_k)
        q_rot = self._full_q_rot
        k_rot = self._full_k_rot

        # 8) Stage Q in backend Q_buf and write new K/V into KV cache.
        # Backend Q_buf shape: (1, max_q_seq=1, 24, 256). q_rot already
        # matches.
        self._attn.Q_buf[:, :1].copy_(q_rot)

        # KV cache shape (NUM_FULL=16, max_seq, 4, 256). Write row cur_pos.
        self._attn.K_cache[full_rank, cur_pos:cur_pos + 1].copy_(
            k_rot.view(1, 4, 256)
        )

        # v_proj -> (1, 1024). Reuse same kv_proj_out_buf scratch (k is
        # already committed to K_cache, safe to overwrite).
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['v_proj_w']),
            kv_proj_out_buf.data_ptr(),
            1, 1024, 5120,
            scale_5120.data_ptr(), int(lw['v_proj_s']),
            s,
        )
        v_new = kv_proj_out_buf[:1].view(1, 4, 256)
        self._attn.V_cache[full_rank, cur_pos:cur_pos + 1].copy_(v_new)

        # 9) Run attention: q_seq=1, kv_seq=cur_pos+1.
        kv_seq = cur_pos + 1
        scaling = float(self._pipeline.hf.config.head_dim) ** -0.5
        self._attn.run(
            'full', layer_idx=full_rank, q_seq=1, kv_seq=kv_seq,
            stream=s, softmax_scale=scaling,
        )
        attn_out = self._attn.O_buf[:, :1]               # (1, 1, 24, 256)

        # 10) Apply output gate: attn * sigmoid(gate). In-place.
        attn_flat = attn_out.reshape(1, 1, 24 * 256)
        torch.sigmoid(gate_flat, out=self._full_gate_sig)
        torch.mul(attn_flat, self._full_gate_sig, out=self._full_gated)
        gated = self._full_gated

        # 11) o_proj FP8 GEMM: K=6144 -> N=5120.
        qinp_6144, scale_6144, _ = self._fp8_scratch[(5120, 6144)]
        gated_2d = gated.view(1, 6144).contiguous()
        fvk.fp8_per_token_block128_quant_bf16(
            gated_2d.data_ptr(), qinp_6144.data_ptr(),
            scale_6144.data_ptr(), 1, 6144, s,
        )
        out_op_buf = self._fp8_scratch[(5120, 6144)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_6144.data_ptr(), int(lw['o_proj_w']),
            out_op_buf.data_ptr(),
            1, 5120, 6144,
            scale_6144.data_ptr(), int(lw['o_proj_s']),
            s,
        )

        # 12) Residual: h_post = h_in + o_proj_out (in-place).
        attn_proj = out_op_buf[:1].view(1, 1, 5120)
        torch.add(h_in, attn_proj, out=self._res_mid)
        h_post = self._res_mid

        # 13) post-attn layernorm + MLP swiglu + residual.
        h_post_view = h_post.view(1, 5120)
        x_mlp = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            h_post_view.data_ptr(), int(lw['post_attn_norm_eff_w']),
            x_mlp.data_ptr(),
            1, 5120, eps, s,
        )

        fvk.fp8_per_token_block128_quant_bf16(
            x_mlp.data_ptr(), qinp_5120.data_ptr(),
            scale_5120.data_ptr(), 1, 5120, s,
        )
        gate_out_buf = self._fp8_scratch[(17408, 5120)][2]
        up_out_buf = self._mlp_up_out
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['mlp_gate_w']),
            gate_out_buf.data_ptr(),
            1, 17408, 5120,
            scale_5120.data_ptr(), int(lw['mlp_gate_s']),
            s,
        )
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['mlp_up_w']),
            up_out_buf.data_ptr(),
            1, 17408, 5120,
            scale_5120.data_ptr(), int(lw['mlp_up_s']),
            s,
        )

        gate_v = gate_out_buf[:1].view(1, 17408)
        up_v = up_out_buf[:1].view(1, 17408)
        gate_silu_up = torch.nn.functional.silu(gate_v) * up_v

        qinp_17408, scale_17408, _ = self._fp8_scratch[(5120, 17408)]
        fvk.fp8_per_token_block128_quant_bf16(
            gate_silu_up.data_ptr(), qinp_17408.data_ptr(),
            scale_17408.data_ptr(), 1, 17408, s,
        )
        down_out_buf = self._fp8_scratch[(5120, 17408)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_17408.data_ptr(), int(lw['mlp_down_w']),
            down_out_buf.data_ptr(),
            1, 5120, 17408,
            scale_17408.data_ptr(), int(lw['mlp_down_s']),
            s,
        )
        mlp_out = down_out_buf[:1].view(1, 1, 5120)

        # final residual: write to ping-pong layer-output buf.
        h_out = self._layer_out_a if (L % 2 == 0) else self._layer_out_b
        torch.add(h_post, mlp_out, out=h_out)
        return h_out

    # ---------- Phase 6 D4: S=K full-attn layer ----------

    def _layer_forward_full_K(self, L: int, h_in_K, cos_K, sin_K,
                              cur_pos: int, K: int):
        """S=K full-attention decoder layer (Phase 6 D4 verify path).

        Args:
            L: full-attn layer index in [0, num_layers).
            h_in_K: (1, K, 5120) bf16 — K input tokens at consecutive
                positions [cur_pos, cur_pos+K).
            cos_K: (1, K, 64) bf16 — RoPE cos for the K positions.
            sin_K: (1, K, 64) bf16 — RoPE sin for the K positions.
            cur_pos: position of the FIRST input token.
            K: number of input tokens (1 <= K <= MAX_Q_SEQ).

        Returns:
            (1, K, 5120) bf16 — output for the K positions, written
            into the layer-output ping-pong _K buf.
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        s = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        assert lw['type'] == 'full_attention', (
            f'_layer_forward_full_K called on layer {L} of type '
            f'{lw["type"]!r}'
        )
        eps = float(self._pipeline.hf.config.rms_norm_eps)
        full_rank = self._full_layer_rank(L)

        h2 = h_in_K.view(K, 5120).contiguous()

        # 1) input layernorm — M=K rows.
        x_norm = self._h_b[:K].view(K, 5120)
        fvk.rms_norm(
            h2.data_ptr(), int(lw['input_norm_eff_w']),
            x_norm.data_ptr(),
            K, 5120, eps, s,
        )

        # 2) FP8 quant M=K (per-token, block-128 scale).
        qinp_5120, scale_5120, _ = self._fp8_scratch[(12288, 5120)]
        fvk.fp8_per_token_block128_quant_bf16(
            x_norm.data_ptr(), qinp_5120.data_ptr(),
            scale_5120.data_ptr(), K, 5120, s,
        )

        # 3) q_proj fused (Q + output_gate) — M=K, N=12288.
        q_proj_out_buf = self._fp8_scratch[(12288, 5120)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['q_proj_w']),
            q_proj_out_buf.data_ptr(),
            K, 12288, 5120,
            scale_5120.data_ptr(), int(lw['q_proj_s']),
            s,
        )
        q_full = q_proj_out_buf[:K].view(1, K, 24, 512)
        q_pre, gate = torch.chunk(q_full, 2, dim=-1)
        gate_flat = gate.reshape(1, K, 24 * 256)

        # 4) k_proj — M=K.
        kv_proj_out_buf = self._fp8_scratch[(1024, 5120)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['k_proj_w']),
            kv_proj_out_buf.data_ptr(),
            K, 1024, 5120,
            scale_5120.data_ptr(), int(lw['k_proj_s']),
            s,
        )
        k_pre = kv_proj_out_buf[:K].view(1, K, 4, 256).contiguous()

        # 5) q_norm / k_norm — per-head RMSNorm. M = K*heads.
        q_pre_2d = q_pre.contiguous().view(K * 24, 256)
        q_norm_out = self._K_full_q_norm_out[:K * 24]
        fvk.rms_norm(
            q_pre_2d.data_ptr(), int(lw['q_norm_eff_w']),
            q_norm_out.data_ptr(),
            K * 24, 256, eps, s,
        )
        k_pre_2d = k_pre.view(K * 4, 256)
        k_norm_out = self._K_full_k_norm_out[:K * 4]
        fvk.rms_norm(
            k_pre_2d.data_ptr(), int(lw['k_norm_eff_w']),
            k_norm_out.data_ptr(),
            K * 4, 256, eps, s,
        )

        # 6) RoPE inline over K positions. cos_K, sin_K shape (1, K, 64).
        q_for_rope = q_norm_out.view(1, K, 24, 256)
        k_for_rope = k_norm_out.view(1, K, 4, 256)
        cos4 = cos_K.view(1, K, 1, 64)
        sin4 = sin_K.view(1, K, 1, 64)
        q_rot_K = self._K_full_q_rot[:, :K]
        k_rot_K = self._K_full_k_rot[:, :K]
        tmp_q = self._K_full_rope_tmp_q[:, :K]
        tmp_k = self._K_full_rope_tmp_k[:, :K]

        def _rope_inline_K(x_in, x_out, tmp):
            x_out[..., 64:].copy_(x_in[..., 64:])
            torch.index_select(
                x_in[..., :64], -1, self._rope_rotate_idx, out=tmp,
            )
            tmp[..., :32].neg_()
            tmp.mul_(sin4)
            tmp.addcmul_(x_in[..., :64], cos4)
            x_out[..., :64].copy_(tmp)

        _rope_inline_K(q_for_rope, q_rot_K, tmp_q)
        _rope_inline_K(k_for_rope, k_rot_K, tmp_k)

        # 7) Stage Q in backend Q_buf [:, :K]; write K rows of K/V into
        # KV cache at rows [cur_pos, cur_pos+K).
        self._attn.Q_buf[:, :K].copy_(q_rot_K)
        self._attn.K_cache[full_rank, cur_pos:cur_pos + K].copy_(
            k_rot_K.view(K, 4, 256))

        # v_proj — M=K (overwrite kv_proj_out_buf, K already committed).
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['v_proj_w']),
            kv_proj_out_buf.data_ptr(),
            K, 1024, 5120,
            scale_5120.data_ptr(), int(lw['v_proj_s']),
            s,
        )
        v_new_K = kv_proj_out_buf[:K].view(K, 4, 256)
        self._attn.V_cache[full_rank, cur_pos:cur_pos + K].copy_(v_new_K)

        # 8) FA2 — vendored fvk fa2 fwd_bf16 has no causal flag, so a
        # single q_seq=K call would let Q[i] attend to K[j>i] (future)
        # because K[cur_pos+j] for j>i was just written by step 7
        # batched K_cache write. Predicting at position cur_pos+i+1
        # would then be corrupted by K[cur_pos+i+1..cur_pos+K-1].
        # Workaround: K serial q_seq=1 calls, each with
        # kv_seq=cur_pos+k+1 — bound by Q's own position. Each call's
        # FA2 only reads the valid causal prefix of K_cache.
        scaling = float(self._pipeline.hf.config.head_dim) ** -0.5
        for k in range(K):
            q_view = self._attn.Q_buf[:, k:k + 1]
            kv_seq_k = cur_pos + k + 1
            k_view = self._attn.K_cache[
                full_rank:full_rank + 1, :kv_seq_k]
            v_view = self._attn.V_cache[
                full_rank:full_rank + 1, :kv_seq_k]
            o_view = self._attn.O_buf[:, k:k + 1]
            self._attn._fa2_fwd(
                Q=q_view.data_ptr(), K=k_view.data_ptr(),
                V=v_view.data_ptr(), O=o_view.data_ptr(),
                softmax_lse=self._attn.lse_buf.data_ptr(),
                softmax_lse_accum=self._attn.lse_accum.data_ptr(),
                o_accum=self._attn.o_accum.data_ptr(),
                batch=1, seqlen_q=1, seqlen_k=kv_seq_k,
                num_heads_q=24, num_heads_kv=4,
                head_dim=256,
                q_strides=(q_view.stride(0), q_view.stride(1),
                           q_view.stride(2)),
                k_strides=(k_view.stride(0), k_view.stride(1),
                           k_view.stride(2)),
                v_strides=(v_view.stride(0), v_view.stride(1),
                           v_view.stride(2)),
                o_strides=(o_view.stride(0), o_view.stride(1),
                           o_view.stride(2)),
                softmax_scale=scaling,
                num_sms=self._attn._num_sms,
                stream=s,
            )
        attn_out = self._attn.O_buf[:, :K]  # (1, K, 24, 256)

        # 9) output gate: attn * sigmoid(gate). K rows.
        attn_flat = attn_out.reshape(1, K, 24 * 256)
        gate_sig = self._K_full_gate_sig[:, :K]
        gated = self._K_full_gated[:, :K]
        torch.sigmoid(gate_flat, out=gate_sig)
        torch.mul(attn_flat, gate_sig, out=gated)

        # 10) o_proj — M=K, N=5120, K_in=6144.
        qinp_6144, scale_6144, _ = self._fp8_scratch[(5120, 6144)]
        gated_2d = gated.view(K, 6144).contiguous()
        fvk.fp8_per_token_block128_quant_bf16(
            gated_2d.data_ptr(), qinp_6144.data_ptr(),
            scale_6144.data_ptr(), K, 6144, s,
        )
        out_op_buf = self._fp8_scratch[(5120, 6144)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_6144.data_ptr(), int(lw['o_proj_w']),
            out_op_buf.data_ptr(),
            K, 5120, 6144,
            scale_6144.data_ptr(), int(lw['o_proj_s']),
            s,
        )

        # 11) residual: h_post = h_in_K + o_proj_out (K rows).
        attn_proj = out_op_buf[:K].view(1, K, 5120)
        res_mid_K = self._K_res_mid[:, :K]
        torch.add(h_in_K, attn_proj, out=res_mid_K)
        h_post = res_mid_K

        # 12) post-attn layernorm + MLP swiglu + residual.
        h_post_view = h_post.view(K, 5120)
        x_mlp = self._h_b[:K].view(K, 5120)
        fvk.rms_norm(
            h_post_view.data_ptr(), int(lw['post_attn_norm_eff_w']),
            x_mlp.data_ptr(),
            K, 5120, eps, s,
        )
        fvk.fp8_per_token_block128_quant_bf16(
            x_mlp.data_ptr(), qinp_5120.data_ptr(),
            scale_5120.data_ptr(), K, 5120, s,
        )
        gate_out_buf = self._fp8_scratch[(17408, 5120)][2]
        up_out_buf = self._mlp_up_out
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['mlp_gate_w']),
            gate_out_buf.data_ptr(),
            K, 17408, 5120,
            scale_5120.data_ptr(), int(lw['mlp_gate_s']),
            s,
        )
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(lw['mlp_up_w']),
            up_out_buf.data_ptr(),
            K, 17408, 5120,
            scale_5120.data_ptr(), int(lw['mlp_up_s']),
            s,
        )
        gate_v = gate_out_buf[:K].view(K, 17408)
        up_v = up_out_buf[:K].view(K, 17408)
        silu_out = self._K_mlp_silu_mul_out[:K]
        fvk.silu_mul_qwen36_bf16(
            gate_v.data_ptr(), up_v.data_ptr(),
            silu_out.data_ptr(), K * 17408, s,
        )
        gate_silu_up = silu_out
        qinp_17408, scale_17408, _ = self._fp8_scratch[(5120, 17408)]
        fvk.fp8_per_token_block128_quant_bf16(
            gate_silu_up.data_ptr(), qinp_17408.data_ptr(),
            scale_17408.data_ptr(), K, 17408, s,
        )
        down_out_buf = self._fp8_scratch[(5120, 17408)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_17408.data_ptr(), int(lw['mlp_down_w']),
            down_out_buf.data_ptr(),
            K, 5120, 17408,
            scale_17408.data_ptr(), int(lw['mlp_down_s']),
            s,
        )
        mlp_out = down_out_buf[:K].view(1, K, 5120)

        # 13) final residual: write to _K ping-pong.
        h_out = (self._K_layer_out_a if (L % 2 == 0)
                 else self._K_layer_out_b)
        h_out_K = h_out[:, :K]
        torch.add(h_post, mlp_out, out=h_out_K)
        return h_out_K

    # ---------- Phase 6 D4: S=K full forward ----------

    def forward_own_decode_K(self, token_ids_K, cos_K, sin_K,
                             cur_pos: int, K: int):
        """Run all 64 decoder layers + final norm + lm_head at S=K.

        Used by the speculative-decode verify pass. K consecutive
        tokens at positions [cur_pos, cur_pos + K) flow through the
        full network in a single batched call, producing K rows of
        logits.

        Args:
            token_ids_K: (1, K) long — input token IDs.
            cos_K: (1, K, 64) bf16 — RoPE cos at the K positions.
            sin_K: (1, K, 64) bf16 — RoPE sin.
            cur_pos: starting position.
            K: number of tokens (1 <= K <= MAX_Q_SEQ).

        Returns:
            (K, vocab) bf16 — logits for each of the K input
            positions. Argmax gives the predicted NEXT token at each
            of [cur_pos+1, cur_pos+2, ..., cur_pos+K].
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        bf16 = torch.bfloat16
        s = torch.cuda.current_stream().cuda_stream
        types = self._pipeline.hf.config.layer_types
        eps = float(self._pipeline.hf.config.rms_norm_eps)

        # 0) Embed K tokens.
        h = self._pipeline.hf.model.embed_tokens(token_ids_K)
        # h shape (1, K, 5120)
        h = h.contiguous()
        if h.dtype != bf16:
            h = h.to(bf16)

        # 1) 64 decoder layers.
        for L in range(self._pipeline.DIMS.num_layers):
            t = types[L]
            if t == 'linear_attention':
                h = self._layer_forward_lin_K(L, h, K)
            elif t == 'full_attention':
                h = self._layer_forward_full_K(
                    L, h, cos_K, sin_K, cur_pos, K)
            else:
                raise ValueError(f'unknown layer_type {t!r} at L={L}')

        # 2) Stash pre-final-norm hidden so MTP head / chained spec can
        # consume per-row hiddens.
        self._K_last_hidden_buf[:, :K].copy_(h)

        # 3) Final RMSNorm M=K.
        h2 = h.view(K, 5120).contiguous()
        x_norm = self._h_b[:K].view(K, 5120)
        fvk.rms_norm(
            h2.data_ptr(), int(self._weights.ptrs['final_norm_eff_w']),
            x_norm.data_ptr(),
            K, 5120, eps, s,
        )

        # 4) lm_head: M=1 matvec K times -> K rows of logits.
        # Looping is fine because launches are tiny in graph and the
        # cost is dominated by the lm_head weight read (1.5 GB) per
        # call. For K up to 4 this is the practical implementation
        # without a real bf16_matmul kernel.
        lm_head_w = self._pipeline.hf.lm_head.weight
        vocab = lm_head_w.shape[0]
        for k in range(K):
            x_row = x_norm[k:k + 1]
            out_row = self._K_logits_buf[k:k + 1]
            fvk.bf16_matvec_qwen36_bf16(
                x_row.data_ptr(), lm_head_w.data_ptr(),
                out_row.data_ptr(), vocab, 5120, s,
            )
        return self._K_logits_buf[:K]

    def forward_own_decode(self, token_id, cos_pos, sin_pos, cur_pos: int):
        """Run a full 64-layer decode step on a single new token.

        Replaces ``mdl.forward(input_ids=..., past_key_values=...)`` for
        the decode hot path. Uses our own per-layer forward functions
        end-to-end -- HF's decoder is bypassed entirely for this call.

        Args:
            token_id: int or (1,) cuda long tensor -- the new token id.
            cos_pos: rotary cos for position cur_pos, shape (1, 1, 256).
            sin_pos: rotary sin for position cur_pos, shape (1, 1, 256).
            cur_pos: integer absolute position of the new token.
                For decode after a prefill of N tokens, cur_pos starts
                at N (the first generated token's position).

        Returns:
            (1, vocab) bf16 logits for the new token.
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        bf16 = torch.bfloat16
        s = torch.cuda.current_stream().cuda_stream
        types = self._pipeline.hf.config.layer_types
        eps = float(self._pipeline.hf.config.rms_norm_eps)

        # 0) Embed: gather embed_tokens row for the input token.
        #    embed_tokens.weight shape (vocab=248320, hidden=5120) bf16.
        #    For decode S=1 just index a single row -- use HF's
        #    embed_tokens module so the implementation matches its
        #    nn.Embedding semantics (handles vocab size, padding_idx).
        if not isinstance(token_id, torch.Tensor):
            token_id = torch.tensor(
                [token_id], device=self.device, dtype=torch.long,
            )
        if token_id.ndim == 1:
            token_id = token_id.view(1, 1)
        h = self._pipeline.hf.model.embed_tokens(token_id)  # (1, 1, 5120)
        h = h.contiguous()
        if h.dtype != bf16:
            h = h.to(bf16)

        # 1) 64 decoder layers.
        for L in range(self._pipeline.DIMS.num_layers):
            t = types[L]
            if t == 'linear_attention':
                h = self._layer_forward_lin(L, h)
            elif t == 'full_attention':
                h = self._layer_forward_full(L, h, cos_pos, sin_pos, cur_pos)
            else:
                raise ValueError(f'unknown layer_type {t!r} at L={L}')

        # 2) Final RMSNorm.
        # Stash the post-64-layer / pre-final-norm hidden so MTP head
        # (forward_mtp_head) can consume it. _last_hidden_buf has a
        # fixed pointer; the .copy_ is just a memcpy of (1,1,5120) bf16.
        self._last_hidden_buf.copy_(h)
        h2 = h.view(1, 5120).contiguous()
        x_norm = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            h2.data_ptr(), int(self._weights.ptrs['final_norm_eff_w']),
            x_norm.data_ptr(),
            1, 5120, eps, s,
        )

        # 3) lm_head: bf16 matvec (1, 5120) -> (vocab,). Use the
        # stream-invariant fvk kernel instead of torch.matmul/cuBLASLt
        # so the kernel choice is identical across eager / different
        # streams / CUDA Graph capture (cuBLASLt's heuristic picked
        # different bf16 GEMM algos per context, breaking graph
        # correctness; see Phase 4.4 root-cause notes).
        lm_head_w = self._pipeline.hf.lm_head.weight  # (vocab, 5120) bf16
        vocab = lm_head_w.shape[0]
        fvk.bf16_matvec_qwen36_bf16(
            x_norm.data_ptr(), lm_head_w.data_ptr(),
            self._logits_buf.data_ptr(), vocab, 5120, s,
        )
        return self._logits_buf

    # ---------- Phase 6 D2: MTP head forward ----------

    def forward_mtp_head(self, prev_h, prev_token_id, cur_pos: int):
        """Run the 1-layer MTP head once.

        Math (DeepSeek-V3 single-MTP-module):

            e        = embed_tokens(prev_token_id)            # (1,1,5120)
            h_norm   = pre_fc_norm_hidden(prev_h)
            e_norm   = pre_fc_norm_embedding(e)
            x_in     = fc(cat[h_norm, e_norm])                # bf16 GEMM
            h_layer  = full_attn_layer_0(x_in, cos, sin, cur_pos,
                                         own KV cache)
            x_final  = mtp.norm(h_layer)
            logits   = lm_head(x_final)                       # tied weight

        Returns ``(h_layer, logits)``: the MTP layer's hidden output
        (used to chain a second MTP iteration) and the logits over
        the vocab for argmax. ``cur_pos`` is the position of
        ``prev_token_id`` in the sequence — the new draft token will
        sit at cur_pos+1.

        Decode-mode only (B=1, S=1).
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        s = torch.cuda.current_stream().cuda_stream
        mtp = self._weights.ptrs.get('mtp')
        if mtp is None:
            raise RuntimeError(
                'MTP head not loaded — checkpoint missing mtp.safetensors')
        eps = float(self._pipeline.hf.config.rms_norm_eps)
        vocab = self._pipeline.DIMS.vocab_size

        # 0) Embed prev_token (use HF embed for consistency w/ main path).
        if prev_token_id.ndim == 1:
            prev_token_id = prev_token_id.view(1, 1)
        e = self._pipeline.hf.model.embed_tokens(prev_token_id)
        if e.dtype != torch.bfloat16:
            e = e.to(torch.bfloat16)

        # 1) pre-fc norms on (h, e) -> bf16 (1, 5120) each.
        prev_h_2d = prev_h.view(1, 5120).contiguous()
        e_2d = e.view(1, 5120).contiguous()
        h_norm = self._mtp_h_norm_buf.view(1, 5120)
        e_norm = self._mtp_e_norm_buf.view(1, 5120)
        fvk.rms_norm(
            prev_h_2d.data_ptr(), int(mtp['pre_fc_norm_hidden_eff_w']),
            h_norm.data_ptr(),
            1, 5120, eps, s,
        )
        fvk.rms_norm(
            e_2d.data_ptr(), int(mtp['pre_fc_norm_embedding_eff_w']),
            e_norm.data_ptr(),
            1, 5120, eps, s,
        )

        # 2) cat [e_norm, h_norm] -> (1, 10240) into pre-alloc buf.
        # Order: embedding FIRST, then hidden (matches DeepSeek-V3 MTP
        # reference: torch.cat([normed_emb, normed_hidden], dim=-1)).
        cat_buf = self._mtp_cat_buf.view(1, 10240)
        cat_buf[:, :5120].copy_(e_norm)
        cat_buf[:, 5120:].copy_(h_norm)

        # 3) fc: BF16 matvec, M=1, K=10240 (= 40*256), N=5120.
        # K%256 == 0 -> bf16_matvec_qwen36 fast path.
        fc_out_2d = self._mtp_fc_out_buf.view(1, 5120)
        fvk.bf16_matvec_qwen36_bf16(
            cat_buf.data_ptr(), int(mtp['fc_w']),
            fc_out_2d.data_ptr(), 5120, 10240, s,
        )

        # 4) Full-attn layer body — mirrors _layer_forward_full but
        # reads MTP layer weights and uses the MTP-private KV cache /
        # Q/O bufs. Inlined rather than refactored to avoid disturbing
        # the heavily-tested main path.
        h_in_full = self._mtp_fc_out_buf  # (1, 1, 5120)
        cos, sin = self._rope_cos_sin(cur_pos)

        # 4a) input layernorm
        h2 = h_in_full.view(1, 5120)
        x_norm = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            h2.data_ptr(), int(mtp['input_norm_eff_w']),
            x_norm.data_ptr(),
            1, 5120, eps, s,
        )

        # 4b) FP8 quant on x_norm (M=1, K=5120).
        qinp_5120, scale_5120, _ = self._fp8_scratch[(12288, 5120)]
        fvk.fp8_per_token_block128_quant_bf16(
            x_norm.data_ptr(), qinp_5120.data_ptr(),
            scale_5120.data_ptr(), 1, 5120, s,
        )

        # 4c) q_proj fused (Q + output_gate) -> (1, 12288).
        q_proj_out_buf = self._fp8_scratch[(12288, 5120)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(mtp['q_proj_w']),
            q_proj_out_buf.data_ptr(),
            1, 12288, 5120,
            scale_5120.data_ptr(), int(mtp['q_proj_s']),
            s,
        )
        q_full = q_proj_out_buf[:1].view(1, 1, 24, 512)
        q_pre, gate = torch.chunk(q_full, 2, dim=-1)
        gate_flat = gate.reshape(1, 1, 24 * 256)

        # 4d) k_proj.
        kv_proj_out_buf = self._fp8_scratch[(1024, 5120)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(mtp['k_proj_w']),
            kv_proj_out_buf.data_ptr(),
            1, 1024, 5120,
            scale_5120.data_ptr(), int(mtp['k_proj_s']),
            s,
        )
        k_pre = kv_proj_out_buf[:1].view(1, 1, 4, 256).contiguous()

        # 4e) q_norm / k_norm.
        q_pre_2d = q_pre.contiguous().view(24, 256)
        fvk.rms_norm(
            q_pre_2d.data_ptr(), int(mtp['q_norm_eff_w']),
            self._full_q_norm_out.data_ptr(),
            24, 256, eps, s,
        )
        k_pre_2d = k_pre.view(4, 256)
        fvk.rms_norm(
            k_pre_2d.data_ptr(), int(mtp['k_norm_eff_w']),
            self._full_k_norm_out.data_ptr(),
            4, 256, eps, s,
        )

        # 4f) RoPE inline (partial_rotary_factor=0.25 -> rope_dim=64).
        q_for_rope = self._full_q_norm_out.view(1, 1, 24, 256)
        k_for_rope = self._full_k_norm_out.view(1, 1, 4, 256)
        cos4 = cos.view(1, 1, 1, 64)
        sin4 = sin.view(1, 1, 1, 64)

        def _rope_inline(x_in, x_out, tmp):
            x_out[..., 64:].copy_(x_in[..., 64:])
            torch.index_select(
                x_in[..., :64], -1, self._rope_rotate_idx, out=tmp,
            )
            tmp[..., :32].neg_()
            tmp.mul_(sin4)
            tmp.addcmul_(x_in[..., :64], cos4)
            x_out[..., :64].copy_(tmp)

        _rope_inline(q_for_rope, self._full_q_rot, self._full_rope_tmp_q)
        _rope_inline(k_for_rope, self._full_k_rot, self._full_rope_tmp_k)
        q_rot = self._full_q_rot
        k_rot = self._full_k_rot

        # 4g) Stage Q in MTP Q_buf, write K/V into MTP cache row cur_pos.
        self._mtp_Q_buf[:, :1].copy_(q_rot)
        self._mtp_K_cache[cur_pos:cur_pos + 1].copy_(
            k_rot.view(1, 4, 256))

        # v_proj (overwrite kv_proj_out_buf, K already committed).
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(mtp['v_proj_w']),
            kv_proj_out_buf.data_ptr(),
            1, 1024, 5120,
            scale_5120.data_ptr(), int(mtp['v_proj_s']),
            s,
        )
        v_new = kv_proj_out_buf[:1].view(1, 4, 256)
        self._mtp_V_cache[cur_pos:cur_pos + 1].copy_(v_new)

        # 4h) FA2 — call directly with MTP buffers (q_seq=1,
        # kv_seq=cur_pos+1).
        kv_seq = cur_pos + 1
        scaling = float(self._pipeline.hf.config.head_dim) ** -0.5
        q_view = self._mtp_Q_buf[:, :1]
        # K/V cache is shape (max_seq, 4, 256); we need (1, kv_seq, 4, 256).
        k_view = self._mtp_K_cache[:kv_seq].view(1, kv_seq, 4, 256)
        v_view = self._mtp_V_cache[:kv_seq].view(1, kv_seq, 4, 256)
        o_view = self._mtp_O_buf[:, :1]
        self._attn._fa2_fwd(
            Q=q_view.data_ptr(), K=k_view.data_ptr(),
            V=v_view.data_ptr(), O=o_view.data_ptr(),
            softmax_lse=self._mtp_lse_buf.data_ptr(),
            softmax_lse_accum=self._mtp_lse_accum.data_ptr(),
            o_accum=self._mtp_o_accum.data_ptr(),
            batch=1, seqlen_q=1, seqlen_k=kv_seq,
            num_heads_q=24, num_heads_kv=4,
            head_dim=256,
            q_strides=(q_view.stride(0), q_view.stride(1),
                       q_view.stride(2)),
            k_strides=(k_view.stride(0), k_view.stride(1),
                       k_view.stride(2)),
            v_strides=(v_view.stride(0), v_view.stride(1),
                       v_view.stride(2)),
            o_strides=(o_view.stride(0), o_view.stride(1),
                       o_view.stride(2)),
            softmax_scale=scaling,
            num_sms=self._attn._num_sms,
            stream=s,
        )
        attn_out = self._mtp_O_buf[:, :1]

        # 4i) output gate: attn * sigmoid(gate).
        attn_flat = attn_out.reshape(1, 1, 24 * 256)
        torch.sigmoid(gate_flat, out=self._full_gate_sig)
        torch.mul(attn_flat, self._full_gate_sig, out=self._full_gated)
        gated = self._full_gated

        # 4j) o_proj FP8 GEMM: K=6144 -> N=5120.
        qinp_6144, scale_6144, _ = self._fp8_scratch[(5120, 6144)]
        gated_2d = gated.view(1, 6144).contiguous()
        fvk.fp8_per_token_block128_quant_bf16(
            gated_2d.data_ptr(), qinp_6144.data_ptr(),
            scale_6144.data_ptr(), 1, 6144, s,
        )
        out_op_buf = self._fp8_scratch[(5120, 6144)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_6144.data_ptr(), int(mtp['o_proj_w']),
            out_op_buf.data_ptr(),
            1, 5120, 6144,
            scale_6144.data_ptr(), int(mtp['o_proj_s']),
            s,
        )

        # 4k) residual: h_post = h_in + o_proj_out.
        attn_proj = out_op_buf[:1].view(1, 1, 5120)
        torch.add(h_in_full, attn_proj, out=self._res_mid)
        h_post = self._res_mid

        # 4l) post-attn norm + MLP swiglu + residual.
        h_post_view = h_post.view(1, 5120)
        x_mlp = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            h_post_view.data_ptr(), int(mtp['post_attn_norm_eff_w']),
            x_mlp.data_ptr(),
            1, 5120, eps, s,
        )
        fvk.fp8_per_token_block128_quant_bf16(
            x_mlp.data_ptr(), qinp_5120.data_ptr(),
            scale_5120.data_ptr(), 1, 5120, s,
        )
        gate_out_buf = self._fp8_scratch[(17408, 5120)][2]
        up_out_buf = self._mlp_up_out
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(mtp['mlp_gate_w']),
            gate_out_buf.data_ptr(),
            1, 17408, 5120,
            scale_5120.data_ptr(), int(mtp['mlp_gate_s']),
            s,
        )
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_5120.data_ptr(), int(mtp['mlp_up_w']),
            up_out_buf.data_ptr(),
            1, 17408, 5120,
            scale_5120.data_ptr(), int(mtp['mlp_up_s']),
            s,
        )
        gate_v = gate_out_buf[:1].view(1, 17408)
        up_v = up_out_buf[:1].view(1, 17408)
        fvk.silu_mul_qwen36_bf16(
            gate_v.data_ptr(), up_v.data_ptr(),
            self._mlp_silu_mul_out.data_ptr(), 17408, s,
        )
        gate_silu_up = self._mlp_silu_mul_out
        qinp_17408, scale_17408, _ = self._fp8_scratch[(5120, 17408)]
        fvk.fp8_per_token_block128_quant_bf16(
            gate_silu_up.data_ptr(), qinp_17408.data_ptr(),
            scale_17408.data_ptr(), 1, 17408, s,
        )
        down_out_buf = self._fp8_scratch[(5120, 17408)][2]
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            qinp_17408.data_ptr(), int(mtp['mlp_down_w']),
            down_out_buf.data_ptr(),
            1, 5120, 17408,
            scale_17408.data_ptr(), int(mtp['mlp_down_s']),
            s,
        )
        mlp_out = down_out_buf[:1].view(1, 1, 5120)

        # 4m) final residual into MTP layer-out buf.
        next_h = self._mtp_layer_out_buf
        torch.add(h_post, mlp_out, out=next_h)

        # 5) MTP final norm + lm_head.
        next_h_view = next_h.view(1, 5120)
        x_final_norm = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            next_h_view.data_ptr(), int(mtp['final_norm_eff_w']),
            x_final_norm.data_ptr(),
            1, 5120, eps, s,
        )
        # lm_head shared with main (tied weight).
        lm_head_w = self._pipeline.hf.lm_head.weight
        fvk.bf16_matvec_qwen36_bf16(
            x_final_norm.data_ptr(), lm_head_w.data_ptr(),
            self._mtp_logits_buf.data_ptr(), vocab, 5120, s,
        )

        return next_h, self._mtp_logits_buf

    def reset_mtp_state(self) -> None:
        """Zero MTP's KV cache (between independent generations)."""
        if hasattr(self, '_mtp_K_cache'):
            self._mtp_K_cache.zero_()
            self._mtp_V_cache.zero_()

    def _ingest_hf_state(self, past_key_values, prefill_len: int) -> None:
        """Copy every per-layer state from HF Cache into our buffers.

        Called once after prefill so subsequent decodes can run through
        forward_own_decode without HF's per-layer Python overhead.

        Args:
            past_key_values: HF Cache instance produced by model.forward
                with use_cache=True. Layers must be initialized.
            prefill_len: number of tokens already in K/V cache (= prompt
                length for the first ingest).
        """
        import torch

        types = self._pipeline.hf.config.layer_types
        for li in range(self._pipeline.DIMS.num_layers):
            cl = past_key_values.layers[li]
            t = types[li]
            if t == 'linear_attention':
                lin_rank = self._linear_layer_rank(li)
                self._lin_conv_state[lin_rank].copy_(cl.conv_states)
                self._lin_state[lin_rank].copy_(cl.recurrent_states)
            else:
                full_rank = self._full_layer_rank(li)
                k = cl.keys
                v = cl.values
                # HF (B, H_kv, S, D) -> ours (max_seq, H_kv, D), B=1
                self._attn.K_cache[full_rank, :prefill_len].copy_(
                    k.transpose(1, 2).squeeze(0).to(torch.bfloat16),
                )
                self._attn.V_cache[full_rank, :prefill_len].copy_(
                    v.transpose(1, 2).squeeze(0).to(torch.bfloat16),
                )

    def _build_rope_table(self) -> None:
        """Pre-compute (cos, sin) for every position in [0, max_seq).

        Stored as ``self._rope_cos_table`` / ``self._rope_sin_table``,
        each shape (max_seq, 1, head_dim=256) bf16 cuda. ``_rope_cos_sin``
        below slices a single (1, 1, 256) row in O(1) -- no Python call
        into HF's RotaryEmbedding.forward per decode step.
        """
        import torch

        # NVFP4 path has no HF model — compute RoPE table from config.
        if self._quant_format == 'nvfp4':
            head_dim = int(self._cfg['head_dim'])
            partial = float(self._cfg['partial_rotary_factor'])
            theta = float(self._cfg['rope_theta'])
            rope_dim = int(head_dim * partial)        # 256 * 0.25 = 64
            inv_freq = 1.0 / (theta ** (
                torch.arange(0, rope_dim, 2,
                             device=self.device, dtype=torch.float32)
                / rope_dim))                          # (rope_dim/2,)
            positions = torch.arange(
                self.max_seq, device=self.device,
                dtype=torch.float32)                  # (max_seq,)
            freqs = positions[:, None] * inv_freq[None, :]  # (max_seq, rope_dim/2)
            # HF convention: cos/sin emb = cat([freqs, freqs], -1)
            emb = torch.cat([freqs, freqs], dim=-1)   # (max_seq, rope_dim)
            self._rope_cos_table = emb.cos().to(torch.bfloat16).contiguous()
            self._rope_sin_table = emb.sin().to(torch.bfloat16).contiguous()
            self._rope_dim = rope_dim
            return

        rotary = self._pipeline.hf.model.rotary_emb
        pos_ids = torch.arange(
            self.max_seq, device=self.device, dtype=torch.long,
        ).view(1, -1)
        ref = self._h_a[:1]
        with torch.no_grad():
            cos_all, sin_all = rotary(ref, pos_ids)
        # Qwen3.6 uses partial_rotary_factor=0.25 so rope_dim=64 (the
        # first 64 of head_dim=256 are rotated, the rest pass through).
        self._rope_cos_table = cos_all.squeeze(0).contiguous()
        self._rope_sin_table = sin_all.squeeze(0).contiguous()
        self._rope_dim = self._rope_cos_table.shape[-1]

    def _rope_cos_sin(self, pos: int):
        """Return (cos, sin) for absolute position ``pos`` from precomputed table.

        Args:
            pos: absolute token position, 0 <= pos < max_seq.

        Returns:
            (cos, sin) -- each shape (1, 1, head_dim=256) bf16 cuda.
        """
        if not hasattr(self, '_rope_cos_table'):
            self._build_rope_table()
        d = self._rope_dim
        cos = self._rope_cos_table[pos:pos + 1].view(1, 1, d)
        sin = self._rope_sin_table[pos:pos + 1].view(1, 1, d)
        return cos, sin

    # ── CUDA Graph cache LRU + shared-mempool helpers ──
    #
    # All ``_ensure_*_graph_*`` methods route their (key → CUDAGraph)
    # bookkeeping through ``_graph_cache_get`` / ``_graph_cache_put``
    # so the cache stays bounded (see ``GRAPH_CACHE_MAX``) and so that
    # captured graphs share one mempool (``self._graph_mempool``)
    # instead of each owning a private one. The shared mempool keeps
    # per-graph footprint to the few buffers actually unique to that
    # capture (cos/sin slice, FA2 partial-LSE workspace, etc.) and lets
    # eviction reclaim that memory once the evicted graph is GC'd.

    def _graph_cache_get(self, cache, key):
        """Return the cached graph for ``key`` and mark it MRU.

        Returns ``None`` on miss. Safe to call on a plain ``dict``
        (legacy state from older pickled instances or tests that
        bypass ``_init_graph_cache``).
        """
        g = cache.get(key)
        if g is not None and isinstance(cache, collections.OrderedDict):
            cache.move_to_end(key)
        return g

    def _graph_cache_put(self, cache, key, g) -> None:
        """Insert ``g`` for ``key``; evict oldest if over the cap.

        Eviction is a single ``popitem(last=False)`` — i.e. drop the
        least-recently-used entry. ``GRAPH_CACHE_MAX <= 0`` disables
        the bound (legacy unbounded behaviour).
        """
        cache[key] = g
        cap = self.GRAPH_CACHE_MAX
        if (cap > 0
                and isinstance(cache, collections.OrderedDict)
                and len(cache) > cap):
            cache.popitem(last=False)

    def clear_graphs(self) -> None:
        """Drop every captured CUDA Graph (NVFP4 + FP8 + DFlash + TQ).

        Public hatch for long-running servers / agents that need to
        proactively reclaim VRAM (e.g. before instantiating a second
        model on the same GPU, or after a phase change such as moving
        from short-prompt chat to long-context summarisation). Cheap
        when there is nothing cached.

        After this call the next request at any ``cur_pos`` re-pays
        the one-time graph capture cost (see ``docs/qwen36_usage.md``
        §"Cold-start vs warm-state" for the magnitude).
        """
        for attr in (
                '_captured_graphs',
                '_captured_verify_graphs',
                '_captured_mtp_graphs',
                '_captured_chain_graphs',
                '_captured_graphs_tq',
                '_captured_verify_graphs_dflash',
                '_captured_drafter_graphs_dflash',
        ):
            cache = getattr(self, attr, None)
            if cache:
                cache.clear()

    def _ensure_graph_for_pos(self, cur_pos: int):
        """Lazy-capture a CUDA Graph for forward_own_decode at cur_pos.

        Caller must already be inside ``with torch.cuda.stream(gs):``
        — every operation here (snap, warmup, restore, capture, second
        restore) must run on gs so there is no cross-stream race
        between state writes and the subsequent g.replay().

        Each cur_pos needs its own captured graph because FA2 takes
        kv_seq as an int kernel arg (= cur_pos+1) and cos/sin are
        sliced from the precomputed table at a cur_pos-specific
        address. Both bake into the captured kernel call list.

        State integrity: the warmup + capture both mutate live
        recurrent / conv / KV cache state. We snapshot before and
        restore after so the actual decode flow's state mutations on
        replay start from the correct pre-step state (i.e. capture is
        side-effect-free on live state).

        Returns:
            torch.cuda.CUDAGraph -- replay produces self._logits_buf
            for input self._static_token_id at cur_pos.
        """
        import torch

        g = self._graph_cache_get(self._captured_graphs, cur_pos)
        if g is not None:
            return g

        gs = self._graph_stream
        cos, sin = self._rope_cos_sin(cur_pos)

        # Partial snap: forward_own_decode only writes
        # K_cache[full_rank, cur_pos:cur_pos+1] across the 16 full-attn
        # layers, so cloning the (16, 1, 4, 256) slice is sufficient.
        # Cloning the full cache used a transient ~2 GB at
        # max_seq=32768 which OOMed long-prompt prefill on 32 GB cards.
        self._snap_lin_buf.copy_(self._lin_state)
        self._snap_conv_buf.copy_(self._lin_conv_state)
        snap_K_row = self._attn.K_cache[
            :, cur_pos:cur_pos + 1].clone()
        snap_V_row = self._attn.V_cache[
            :, cur_pos:cur_pos + 1].clone()

        def _restore_on_gs():
            self._lin_state.copy_(self._snap_lin_buf)
            self._lin_conv_state.copy_(self._snap_conv_buf)
            self._attn.K_cache[
                :, cur_pos:cur_pos + 1].copy_(snap_K_row)
            self._attn.V_cache[
                :, cur_pos:cur_pos + 1].copy_(snap_V_row)

        # Warmup (2 iters) — settles allocator / kernel-chain order.
        with torch.no_grad():
            for _ in range(2):
                self.forward_own_decode(
                    self._static_token_id, cos, sin, cur_pos,
                )
            _restore_on_gs()

        # Capture. torch.cuda.graph entry/exit handles capture begin/end
        # synchronously; we are already on gs so the ``stream=gs`` arg
        # is consistent.
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
                g, stream=gs, pool=self._graph_mempool,
        ), torch.no_grad():
            self.forward_own_decode(
                self._static_token_id, cos, sin, cur_pos,
            )
        # Capture itself executed the kernels once on gs; roll state
        # back so the subsequent replay starts from the snapshot.
        with torch.no_grad():
            _restore_on_gs()

        self._graph_cache_put(self._captured_graphs, cur_pos, g)
        return g

    def _ensure_mtp_graph(self, mtp_pos: int):
        """Lazy-capture a CUDA Graph for forward_mtp_head at mtp_pos.

        The graph reads from the static input buffers
        (_mtp_static_prev_h, _mtp_static_prev_token) and writes to
        the fixed _mtp_layer_out_buf / _mtp_logits_buf. Each mtp_pos
        gets its own graph because FA2's kv_seq is baked into the
        captured kernel's int args.

        Side-effect on MTP K/V cache: the capture writes to
        _mtp_K_cache[mtp_pos] / _mtp_V_cache[mtp_pos]. We snap+restore
        those rows so the warmup/capture doesn't perturb the live
        MTP cache state used by the surrounding decode flow.
        """
        import torch

        g = self._graph_cache_get(self._captured_mtp_graphs, mtp_pos)
        if g is not None:
            return g

        gs = self._graph_stream

        # Snap MTP K/V row at this position so warmup/capture writes
        # don't affect the live cache.
        snap_K = self._mtp_K_cache[mtp_pos:mtp_pos + 1].clone()
        snap_V = self._mtp_V_cache[mtp_pos:mtp_pos + 1].clone()

        def _restore():
            self._mtp_K_cache[mtp_pos:mtp_pos + 1].copy_(snap_K)
            self._mtp_V_cache[mtp_pos:mtp_pos + 1].copy_(snap_V)

        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.no_grad():
            for _ in range(2):
                self.forward_mtp_head(
                    self._mtp_static_prev_h,
                    self._mtp_static_prev_token,
                    mtp_pos,
                )
                _restore()

        gs.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
                g, stream=gs, pool=self._graph_mempool,
        ), torch.no_grad():
            self.forward_mtp_head(
                self._mtp_static_prev_h,
                self._mtp_static_prev_token,
                mtp_pos,
            )
        with torch.cuda.stream(gs), torch.no_grad():
            _restore()
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)

        self._graph_cache_put(self._captured_mtp_graphs, mtp_pos, g)
        return g

    def _ensure_verify_graph(self, cur_pos: int, K: int):
        """Lazy-capture a CUDA Graph for the spec verify forward.

        Captures forward_own_decode_K(K=K) at this cur_pos with the
        static input buffers (_verify_static_tokens / _cos / _sin).
        Each (cur_pos, K) pair gets its own graph because FA2 takes
        kv_seq=cur_pos+K as an int kernel arg.

        State integrity: caller is expected to be inside the spec
        loop AFTER the snap (pre-verify) — we run warmup + capture +
        restore inside our own snap window so the cached graph's
        replay produces a clean N+1 step state mutation on each call.

        Returns the cached CUDAGraph; caller is responsible for
        copying token_ids/cos/sin into the static buffers and
        replaying.
        """
        import torch

        key = (cur_pos, K)
        g = self._graph_cache_get(self._captured_verify_graphs, key)
        if g is not None:
            return g

        gs = self._graph_stream

        # Snap state for capture-only side-effects.
        snap_lin = self._lin_state.clone()
        snap_conv = self._lin_conv_state.clone()
        snap_K = self._attn.K_cache[:, cur_pos:cur_pos + K].clone()
        snap_V = self._attn.V_cache[:, cur_pos:cur_pos + K].clone()

        def _restore():
            self._lin_state.copy_(snap_lin)
            self._lin_conv_state.copy_(snap_conv)
            self._attn.K_cache[
                :, cur_pos:cur_pos + K].copy_(snap_K)
            self._attn.V_cache[
                :, cur_pos:cur_pos + K].copy_(snap_V)

        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.no_grad():
            tokens_K = self._verify_static_tokens[:, :K]
            cos_K = self._verify_static_cos[:, :K]
            sin_K = self._verify_static_sin[:, :K]
            for _ in range(2):
                self.forward_own_decode_K(
                    tokens_K, cos_K, sin_K, cur_pos, K=K)
                _restore()

        gs.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
                g, stream=gs, pool=self._graph_mempool,
        ), torch.no_grad():
            self.forward_own_decode_K(
                tokens_K, cos_K, sin_K, cur_pos, K=K)
        with torch.cuda.stream(gs), torch.no_grad():
            _restore()
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)

        self._graph_cache_put(self._captured_verify_graphs, key, g)
        return g

    def generate_own(self, input_ids, *, max_new_tokens: int,
                     use_graph: bool = True):
        """Greedy decode using our own forward for every step after prefill.

        Prefill stays on HF (chunk_gated_delta_rule for linear-attn,
        FA2 for full-attn -- these paths haven't been ported yet and
        are one-shot anyway). After prefill we ingest the cache state
        once and then loop through forward_own_decode.

        When ``use_graph`` is True (default), each decode step replays
        a per-cur_pos captured CUDA Graph that wraps the entire 64-
        layer forward + lm_head. First encounter at a given cur_pos
        pays a one-time capture cost (~3 forward-passes worth);
        subsequent encounters skip ~1500 cudaLaunchKernel calls and
        run as a single replay.

        Args:
            input_ids: (1, prompt_len) cuda long tensor.
            max_new_tokens: how many tokens to generate after prefill.
            use_graph: if False, fall back to per-step eager forward
                (kept for diff testing / first-pass cos verification).

        Returns:
            (1, prompt_len + max_new_tokens) cuda long tensor.
        """
        import torch

        prompt_len = int(input_ids.shape[1])
        with torch.no_grad():
            out_pre = self._pipeline.hf(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )
            pkv = out_pre.past_key_values
            self._ingest_hf_state(pkv, prompt_len)
            next_id = out_pre.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = [next_id]

            if use_graph:
                gs = self._graph_stream
                # Hand off to gs: gs must see the prefill / ingest /
                # static_in writes that happened on the default stream.
                gs.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(gs):
                    self._static_token_id.copy_(next_id)
                    for step in range(max_new_tokens - 1):
                        cur_pos = prompt_len + step
                        # Both ensure-graph (snap/warmup/capture/restore)
                        # and replay run on gs — no cross-stream race
                        # between state mutations and replay reads.
                        g = self._ensure_graph_for_pos(cur_pos)
                        g.replay()
                        next_id_view = self._logits_buf.argmax(
                            dim=-1, keepdim=True,
                        ).view(1, 1)
                        self._static_token_id.copy_(next_id_view)
                        generated.append(next_id_view.clone())
                # Hand back to default stream — caller will read the
                # generated list and may call other ops on default.
                torch.cuda.current_stream().wait_stream(gs)
            else:
                for step in range(max_new_tokens - 1):
                    cur_pos = prompt_len + step
                    cos, sin = self._rope_cos_sin(cur_pos)
                    logits = self.forward_own_decode(
                        next_id, cos, sin, cur_pos,
                    )
                    next_id = logits.argmax(
                        dim=-1, keepdim=True,
                    ).view(1, 1)
                    generated.append(next_id)

        return torch.cat([input_ids] + generated, dim=1)

    # ---------- Phase 6 D4-8: K-generic spec decode (MTP chain) ----------

    def generate_own_speculative_KN(self, input_ids, *,
                                    max_new_tokens: int, K: int = 5,
                                    use_verify_graph: bool = True,
                                    use_mtp_graph: bool = True):
        """K-generic speculative decode (MTP chain + S=K+1 verify).

        Per Qwen3-Next / DeepSeek-V3 design, the single MTP module is
        applied RECURSIVELY K times: each chain step takes h_mtp from
        the previous step (or h_main on step 0) and the just-drafted
        token, and produces the next draft. K up to MAX_Q_SEQ-1.

        Cycle:
          1. Chain MTP K times -> drafts d_1..d_K. Each step writes
             MTP K-cache @ position cur_pos+k.
          2. Snapshot lin recurrent state + K_cache/V_cache rows
             [cur_pos:cur_pos+K+1] (partial-snap).
          3. forward_own_decode_K(K=K+1) over [tok, d_1, ..., d_K] at
             positions [cur_pos, ..., cur_pos+K].
          4. argmax each row: a_0..a_K. a_i = main pred at cur_pos+i+1.
          5. Accept-prefix N: largest N with d_{j+1} == a_j for j<N.
          6. If N==K (full): output [a_0..a_K] (K+1 tokens). State
             correctly post-(K+1) valid steps.
          7. If N<K (partial): output [a_0..a_N] (N+1 tokens). Restore
             state, run forward_own_decode_K(K=N+1) over [tok, d_1..d_N]
             to re-advance state by N+1 valid steps.

        Greedy correctness: every output token is main's argmax under
        a confirmed-correct prefix of inputs. Matches HF .generate()
        bit-for-bit.

        Args:
            input_ids: (1, prompt_len) cuda long.
            max_new_tokens: how many tokens to generate.
            K: number of MTP chain drafts per cycle. K+1 must be
                <= MAX_Q_SEQ. Default 5 (Qwen3-Next official spec).

        Returns:
            (1, prompt_len + N) cuda long, trimmed to max_new_tokens.
        """
        import torch

        if self._weights.ptrs.get('mtp') is None:
            raise RuntimeError(
                'MTP head not loaded — speculative decode unavailable')
        if K < 1 or K + 1 > self.MAX_Q_SEQ:
            raise ValueError(
                f'K={K} out of range — need 1<=K<={self.MAX_Q_SEQ - 1}')

        prompt_len = int(input_ids.shape[1])
        self.reset_state()
        self.reset_mtp_state()

        with torch.no_grad():
            out_pre = self._pipeline.hf(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            self._ingest_hf_state(out_pre.past_key_values, prompt_len)
            h_main_all = out_pre.hidden_states[-1].to(torch.bfloat16)
            tok = out_pre.logits[:, -1, :].argmax(
                dim=-1, keepdim=True).view(1, 1)
            generated = [tok]
            cur_pos = prompt_len

            # MTP prefill: positions [1..prompt_len-1] from input_ids,
            # plus position prompt_len with the just-predicted tok.
            for p in range(1, prompt_len):
                prev_h_p = h_main_all[:, p - 1:p, :].contiguous()
                prev_tok_p = input_ids[:, p:p + 1]
                self.forward_mtp_head(prev_h_p, prev_tok_p, p)
            h_last_prompt = h_main_all[
                :, prompt_len - 1:prompt_len, :].contiguous()
            self.forward_mtp_head(h_last_prompt, tok, prompt_len)
            h = h_last_prompt

            self._spec_attempts = 0
            self._spec_accepts = 0  # individual draft accepts
            self._spec_full = 0     # full-K-accept cycles

            while len(generated) < max_new_tokens:
                # 2) Snapshot main state on _snap_stream FIRST so it
                # overlaps with the MTP chain on the default stream
                # (MTP doesn't touch main lin_state / K_cache, so the
                # copies can run concurrently). In-place .copy_ into
                # pre-alloc snap buffers — no allocator overhead.
                snap_stream = self._snap_stream
                snap_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(snap_stream):
                    self._snap_lin_buf.copy_(self._lin_state)
                    self._snap_conv_buf.copy_(self._lin_conv_state)
                    self._snap_K_buf[:, :K + 1].copy_(
                        self._attn.K_cache[
                            :, cur_pos:cur_pos + K + 1])
                    self._snap_V_buf[:, :K + 1].copy_(
                        self._attn.V_cache[
                            :, cur_pos:cur_pos + K + 1])

                # 1) Chain K MTP forwards on default stream — overlaps
                # with snap (no shared mutable state).
                drafts: list = []
                if use_mtp_graph:
                    # Replay path: keep entire chain on gs to avoid
                    # 2*K stream-bounces. argmax + copy_ run on gs too.
                    gs = self._graph_stream
                    # Pre-capture all needed graphs (outside gs context
                    # because _ensure_mtp_graph itself uses gs).
                    chain_graphs = [
                        self._ensure_mtp_graph(cur_pos + k)
                        for k in range(K)
                    ]
                    gs.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(gs):
                        self._mtp_static_prev_h.copy_(h)
                        self._mtp_static_prev_token.copy_(tok)
                        for k in range(K):
                            chain_graphs[k].replay()
                            d_k = self._mtp_logits_buf.argmax(
                                dim=-1, keepdim=True).view(1, 1)
                            drafts.append(d_k)
                            if k < K - 1:
                                self._mtp_static_prev_h.copy_(
                                    self._mtp_layer_out_buf)
                                self._mtp_static_prev_token.copy_(d_k)
                    torch.cuda.current_stream().wait_stream(gs)
                else:
                    h_chain = h
                    prev_tok = tok
                    for k in range(K):
                        mtp_pos = cur_pos + k
                        h_mtp_k, dl_k = self.forward_mtp_head(
                            h_chain, prev_tok, mtp_pos)
                        d_k = dl_k.argmax(
                            dim=-1, keepdim=True).view(1, 1)
                        drafts.append(d_k)
                        h_chain = h_mtp_k.clone()
                        prev_tok = d_k

                # Wait for snap to finish before verify writes state.
                torch.cuda.current_stream().wait_stream(snap_stream)

                # 3) Main S=K+1 verify.
                # Build inputs. Batch the K+1 cos/sin rows in one
                # slice from the precomputed RoPE table — avoids K+1
                # Python calls + torch.cat (~1-2 ms/cycle).
                tokens_KN = torch.cat([tok] + drafts, dim=1)
                if not hasattr(self, '_rope_cos_table'):
                    self._build_rope_table()
                d = self._rope_dim
                cos_KN = self._rope_cos_table[
                    cur_pos:cur_pos + K + 1].view(1, K + 1, d)
                sin_KN = self._rope_sin_table[
                    cur_pos:cur_pos + K + 1].view(1, K + 1, d)

                if use_verify_graph:
                    # Copy data into static input bufs and replay the
                    # captured verify graph for (cur_pos, K+1).
                    Kv = K + 1
                    self._verify_static_tokens[:, :Kv].copy_(tokens_KN)
                    self._verify_static_cos[:, :Kv].copy_(cos_KN)
                    self._verify_static_sin[:, :Kv].copy_(sin_KN)
                    g = self._ensure_verify_graph(cur_pos, Kv)
                    gs = self._graph_stream
                    gs.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(gs):
                        g.replay()
                    torch.cuda.current_stream().wait_stream(gs)
                    logits_KN = self._K_logits_buf[:Kv]
                else:
                    logits_KN = self.forward_own_decode_K(
                        tokens_KN, cos_KN, sin_KN, cur_pos, K=K + 1)

                # 4) Batched argmax + accept-prefix on GPU.
                # all_argmax shape (K+1,), drafts_stack shape (K,).
                # Compute first-mismatch index in a single sync via:
                #   pad matches with a sentinel-mismatch at index K
                #   N = argmin(matches_with_sentinel) -> first 0 index
                # If all K matches are 1, sentinel at K becomes the
                # argmin and we read N=K (full accept).
                all_argmax = logits_KN.argmax(dim=-1)  # (K+1,) long
                # drafts list of (1,1) tensors -> (K,) long stacked
                drafts_stack = torch.cat(drafts, dim=1).view(-1)
                matches = (all_argmax[:K] == drafts_stack).long()
                # append a 0 sentinel so argmin returns K on full match
                matches_pad = torch.cat([
                    matches,
                    torch.zeros(1, device=matches.device,
                                dtype=matches.dtype),
                ])
                N = int(matches_pad.argmin().item())  # single sync
                self._spec_attempts += 1
                self._spec_accepts += N
                # Index helper: argmax_at(j) returns a (1,1) long view
                argmax_at = (lambda j: all_argmax[j:j + 1]
                             .view(1, 1))

                if N == K:
                    # Full accept: output K+1 argmaxes, state correct.
                    self._spec_full += 1
                    for j in range(K + 1):
                        if len(generated) < max_new_tokens:
                            generated.append(argmax_at(j))
                    tok = argmax_at(K)
                    h = self._K_last_hidden_buf[
                        :, K:K + 1, :].contiguous()
                    cur_pos += K + 1
                else:
                    # Partial: output N+1 argmaxes, recover state.
                    for j in range(N + 1):
                        if len(generated) < max_new_tokens:
                            generated.append(argmax_at(j))
                    # Restore main state to pre-verify (read from
                    # pre-alloc snap buffers, no per-call alloc).
                    self._lin_state.copy_(self._snap_lin_buf)
                    self._lin_conv_state.copy_(self._snap_conv_buf)
                    self._attn.K_cache[
                        :, cur_pos:cur_pos + K + 1].copy_(
                            self._snap_K_buf[:, :K + 1])
                    self._attn.V_cache[
                        :, cur_pos:cur_pos + K + 1].copy_(
                            self._snap_V_buf[:, :K + 1])

                    # Re-advance with N+1 valid inputs:
                    # [tok, d_1..d_N] at positions [cur_pos..cur_pos+N].
                    if N == 0:
                        # Fast path: single S=1 forward. Lazy-capture
                        # the per-cur_pos forward_own_decode graph
                        # (shares cache with generate_own's path).
                        self._static_token_id.copy_(tok)
                        g_recov = self._ensure_graph_for_pos(cur_pos)
                        gs = self._graph_stream
                        gs.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(gs):
                            g_recov.replay()
                        torch.cuda.current_stream().wait_stream(gs)
                        h = self._last_hidden_buf.contiguous()
                    else:
                        rec_tokens = torch.cat(
                            [tok] + drafts[:N], dim=1)
                        rec_cos = cos_KN[:, :N + 1]
                        rec_sin = sin_KN[:, :N + 1]
                        _ = self.forward_own_decode_K(
                            rec_tokens, rec_cos, rec_sin,
                            cur_pos, K=N + 1)
                        h = self._K_last_hidden_buf[
                            :, N:N + 1, :].contiguous()
                    tok = argmax_at(N)
                    cur_pos += N + 1

            if len(generated) > max_new_tokens:
                generated = generated[:max_new_tokens]

        return torch.cat([input_ids] + generated, dim=1)

    # ---------- Phase 6 D4-5: real spec decode (S=K verify) ----------

    def generate_own_speculative_K2(self, input_ids, *,
                                    max_new_tokens: int):
        """Real K=1-draft / S=2-verify speculative decode.

        Per cycle:
          1. MTP head drafts the token at cur_pos+1.
          2. Snapshot main recurrent / KV state.
          3. forward_own_decode_K(K=2) over [tok@cur_pos, draft@cur_pos+1]
             produces (verify=row0.argmax, bonus=row1.argmax).
          4. Compare draft vs verify:
             ACCEPT (draft == verify): keep state (it's correctly
               advanced by 2 steps). Append [verify, bonus]. Run an
               extra MTP forward at cur_pos+1 to fill the MTP-cache
               hole at the skipped position (otherwise next-iter
               MTP attention has a zero row at cur_pos+1).
             REJECT: restore state from snapshot, run a single S=1
               main forward(tok, cur_pos) to re-advance state by 1
               valid step. Append [verify].

        Greedy correctness: matches HF .generate() bit-for-bit
        because every output token is either main's row0 prediction
        (verify) or main's row1 prediction conditioned on a
        confirmed-correct row0 input (bonus on accept).

        Args:
            input_ids: (1, prompt_len) cuda long.
            max_new_tokens: how many tokens to generate.

        Returns:
            (1, prompt_len + max_new_tokens) cuda long. May overshoot
            by 1 if the last cycle accepted a bonus past the budget;
            output is trimmed to exactly max_new_tokens.
        """
        import torch

        if self._weights.ptrs.get('mtp') is None:
            raise RuntimeError(
                'MTP head not loaded — speculative decode unavailable')

        prompt_len = int(input_ids.shape[1])
        self.reset_state()
        self.reset_mtp_state()

        with torch.no_grad():
            out_pre = self._pipeline.hf(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            self._ingest_hf_state(out_pre.past_key_values, prompt_len)
            h_main_all = out_pre.hidden_states[-1].to(torch.bfloat16)
            tok = out_pre.logits[:, -1, :].argmax(
                dim=-1, keepdim=True).view(1, 1)
            generated = [tok]
            cur_pos = prompt_len

            # MTP prefill: positions [1..prompt_len-1] use input_ids
            # tokens. Position prompt_len uses the just-predicted tok
            # so MTP cache covers [1..prompt_len] before the loop.
            for p in range(1, prompt_len):
                prev_h_p = h_main_all[:, p - 1:p, :].contiguous()
                prev_tok_p = input_ids[:, p:p + 1]
                self.forward_mtp_head(prev_h_p, prev_tok_p, p)
            # Position prompt_len: tok lives here.
            h_last_prompt = h_main_all[
                :, prompt_len - 1:prompt_len, :].contiguous()
            self.forward_mtp_head(h_last_prompt, tok, prompt_len)

            # Initial h for spec loop: h_main@(cur_pos-1) =
            # h_main@(prompt_len-1).
            h = h_last_prompt

            self._spec_attempts = 0
            self._spec_accepts = 0

            while len(generated) < max_new_tokens:
                # 1) MTP draft at cur_pos -> predicts tok@(cur_pos+1).
                _, draft_logits = self.forward_mtp_head(h, tok, cur_pos)
                draft = draft_logits.argmax(
                    dim=-1, keepdim=True).view(1, 1)

                # 2) Snapshot main state for potential restore. Only
                # the K_cache / V_cache rows [cur_pos:cur_pos+2] get
                # mutated by the K=2 forward — partial-snap saves
                # ~128 MB clones per cycle (was clone-everything).
                snap_lin = self._lin_state.clone()
                snap_conv = self._lin_conv_state.clone()
                snap_K = self._attn.K_cache[
                    :, cur_pos:cur_pos + 2].clone()
                snap_V = self._attn.V_cache[
                    :, cur_pos:cur_pos + 2].clone()

                # 3) S=2 verify with input [tok, draft] at positions
                # [cur_pos, cur_pos+1].
                cos1, sin1 = self._rope_cos_sin(cur_pos)
                cos2, sin2 = self._rope_cos_sin(cur_pos + 1)
                tokens_K2 = torch.cat([tok, draft], dim=1)
                cos_K2 = torch.cat([cos1, cos2], dim=1)
                sin_K2 = torch.cat([sin1, sin2], dim=1)
                logits_K2 = self.forward_own_decode_K(
                    tokens_K2, cos_K2, sin_K2, cur_pos, K=2)
                verify = logits_K2[0:1].argmax(
                    dim=-1, keepdim=True).view(1, 1)
                bonus = logits_K2[1:2].argmax(
                    dim=-1, keepdim=True).view(1, 1)
                self._spec_attempts += 1

                if int(draft.item()) == int(verify.item()):
                    # ACCEPT: keep state, output [verify, bonus].
                    generated.append(verify)
                    if len(generated) < max_new_tokens:
                        generated.append(bonus)
                    self._spec_accepts += 1

                    # Fill MTP cache hole at cur_pos+1 (the skipped
                    # position) using (h_main@cur_pos, verify).
                    h_at_cur = self._K_last_hidden_buf[
                        :, 0:1, :].contiguous()
                    self.forward_mtp_head(h_at_cur, verify, cur_pos + 1)

                    # h for next iter: h_main@(cur_pos+1) = row 1.
                    h = self._K_last_hidden_buf[
                        :, 1:2, :].contiguous()
                    tok = bonus
                    cur_pos += 2
                else:
                    # REJECT: restore state, output [verify], run
                    # S=1(tok) to re-advance state by 1 valid step.
                    generated.append(verify)
                    self._lin_state.copy_(snap_lin)
                    self._lin_conv_state.copy_(snap_conv)
                    self._attn.K_cache[
                        :, cur_pos:cur_pos + 2].copy_(snap_K)
                    self._attn.V_cache[
                        :, cur_pos:cur_pos + 2].copy_(snap_V)

                    cos, sin = self._rope_cos_sin(cur_pos)
                    _ = self.forward_own_decode(tok, cos, sin, cur_pos)
                    h = self._last_hidden_buf.contiguous()
                    tok = verify
                    cur_pos += 1

            if len(generated) > max_new_tokens:
                generated = generated[:max_new_tokens]

        return torch.cat([input_ids] + generated, dim=1)

    # ---------- Phase 6 D5: spec decode WIP (needs S=K main) ----------

    def generate_own_speculative(self, input_ids, *,
                                 max_new_tokens: int):
        """**WIP / does not save time** — kept for architecture validation.

        Critical realization (verified via rtx_qwen36_spec_debug):
        MTP draft and main verify both predict the SAME position
        (cur_pos+1 given input tok@cur_pos). They are competing
        predictions, not complementary. Even with high accept rate
        (~0.85) the ``bonus`` token appended on accept is the SAME
        token main is about to predict next — duplicates the output
        and skips a position, breaking token correctness vs HF greedy.

        True speculative decoding requires S=K main forward (Phase 6
        D4) so that one main pass predicts K+1 tokens given a draft
        chain of K. The architecture is in place (forward_mtp_head
        works at smoke level, MTP prefill populates the KV cache,
        accept rate >0.85 once warmed) but the speedup will only
        materialize once D4 lands.

        Until D4: this method is preserved for sub-system regression
        testing — it confirms MTP draft predictions track main's
        predictions reliably, which is the prerequisite for D4 to
        give real throughput. Do NOT use for production decoding.

        Self-spec mechanics (chained MTP hidden, no extra main on accept):
            t = 0:  main(input=tok_T, cur_pos=T)  ->  h_main, tok_{T+1}
                    output: tok_{T+1}
            loop:
              draft, h_mtp = MTP(h_main, tok_{T+1}, cur_pos=T+1)
              h_main, tok_{T+2} = main(input=tok_{T+1}, cur_pos=T+1)
              output: tok_{T+2}
              if draft == tok_{T+2}:
                  output: draft
                  carry h <- h_mtp, tok <- draft, cur_pos += 2
              else:
                  carry h <- h_main, tok <- tok_{T+2}, cur_pos += 1

        IMPORTANT: this first cut does NOT prefill the MTP head's KV
        cache over the prompt. MTP attention at the first decode step
        only sees its own freshly-written K/V (positions [0, prompt_len)
        are zero). Accept rate may be degraded for early tokens; if
        observed p_accept < 0.4, add an MTP prefill pass.

        Args:
            input_ids: (1, prompt_len) cuda long tensor.
            max_new_tokens: how many tokens to generate.

        Returns:
            (1, prompt_len + N) cuda long tensor, where N may be up to
            max_new_tokens (we may overshoot by 1 when the last cycle
            accepts a bonus draft; trimmed to max_new_tokens).
        """
        import torch

        if self._weights.ptrs.get('mtp') is None:
            raise RuntimeError(
                'MTP head not loaded — speculative decode unavailable')

        prompt_len = int(input_ids.shape[1])
        # Reset MTP state (KV cache zero) before each generation.
        self.reset_mtp_state()

        with torch.no_grad():
            # Prefill on HF — also get pre-final-norm hidden states for
            # every prompt position so we can warm the MTP KV cache.
            out_pre = self._pipeline.hf(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            self._ingest_hf_state(out_pre.past_key_values, prompt_len)
            # hidden_states is a tuple of length num_layers+1; the last
            # entry is the output of the LAST decoder layer (pre final
            # norm) — exactly the prev_h that MTP expects.
            h_main_all = out_pre.hidden_states[-1].to(torch.bfloat16)
            tok = out_pre.logits[:, -1, :].argmax(
                dim=-1, keepdim=True).view(1, 1)
            generated = [tok]
            cur_pos = prompt_len  # tok sits at position prompt_len

            # MTP prefill: populate MTP's KV cache by running the head
            # over each prompt position p in [1, prompt_len). Each call
            # writes (K, V) at row p of the MTP cache. Position 0 is
            # left zero (no h_main@(-1) exists). For prompt_len=11 this
            # is ~10 calls × 3 ms = ~30 ms of one-time overhead per
            # generation. Reused across all decode steps; amortized.
            for p in range(1, prompt_len):
                prev_h_p = h_main_all[:, p - 1:p, :].contiguous()
                prev_tok_p = input_ids[:, p:p + 1]
                self.forward_mtp_head(prev_h_p, prev_tok_p, p)

            # Telemetry: track accept rate for debugging / tuning.
            self._spec_attempts = 0
            self._spec_accepts = 0

            # Seed loop with one main step so we have h_main.
            cos, sin = self._rope_cos_sin(cur_pos)
            logits = self.forward_own_decode(tok, cos, sin, cur_pos)
            tok = logits.argmax(dim=-1, keepdim=True).view(1, 1)
            generated.append(tok)
            h = self._last_hidden_buf.clone()
            cur_pos += 1  # tok now sits at cur_pos

            while len(generated) < max_new_tokens:
                # Draft: MTP predicts the token AFTER tok.
                h_mtp, draft_logits = self.forward_mtp_head(
                    h, tok, cur_pos)
                draft = draft_logits.argmax(
                    dim=-1, keepdim=True).view(1, 1)

                # Verify: main forward predicts the token AFTER tok.
                cos, sin = self._rope_cos_sin(cur_pos)
                verify_logits = self.forward_own_decode(
                    tok, cos, sin, cur_pos)
                verify = verify_logits.argmax(
                    dim=-1, keepdim=True).view(1, 1)
                generated.append(verify)
                h_main = self._last_hidden_buf.clone()
                self._spec_attempts += 1

                # Accept-check: bonus draft only if it matches verify
                # AND we still need more tokens.
                if (draft.item() == verify.item()
                        and len(generated) < max_new_tokens):
                    generated.append(draft)
                    self._spec_accepts += 1
                    # Chain MTP hidden as next prev_h. Note: this is
                    # an approximation — h_mtp was produced by the MTP
                    # transformer, not by main. Empirically it's good
                    # enough for nearby positions.
                    h = h_mtp.clone()
                    tok = draft
                    cur_pos += 2
                else:
                    h = h_main
                    tok = verify
                    cur_pos += 1

            # Trim if we overshot (last accept appended one too many).
            if len(generated) > max_new_tokens:
                generated = generated[:max_new_tokens]

        return torch.cat([input_ids] + generated, dim=1)

    def _ensure_graph_for_pos_nvfp4(self, cur_pos: int):
        """Lazy CUDA Graph capture for forward_own_decode_nvfp4 at cur_pos.

        Mirrors ``_ensure_graph_for_pos`` (FP8) but captures the NVFP4
        forward chain instead. Each cur_pos gets its own graph because
        FA2 bakes kv_seq=cur_pos+1 and cos/sin slice addresses into
        the captured kernel call list.

        State integrity: snapshot lin_state / lin_conv_state / the
        single KV row this step writes, restore post-capture so
        replay starts from the original pre-step state.

        Snap is *partial* — only the row at ``cur_pos`` (32 KB across
        all 16 full-attn layers) is cloned, not the entire KV cache
        slab. Cloning the whole cache used a transient ~2 GB at
        ``max_seq=32768`` and was the proximate OOM in the long-prompt
        bug report; sister methods (verify / mtp / chain / dflash)
        already snap partially. Lin-attn state is copied into
        pre-allocated ``_snap_lin_buf`` / ``_snap_conv_buf`` (no fresh
        allocation per capture).
        """
        import torch

        g = self._graph_cache_get(self._captured_graphs, cur_pos)
        if g is not None:
            return g

        gs = self._graph_stream
        cos, sin = self._rope_cos_sin(cur_pos)

        # Snap into pre-allocated lin buffers — zero alloc per capture.
        self._snap_lin_buf.copy_(self._lin_state)
        self._snap_conv_buf.copy_(self._lin_conv_state)
        # Partial KV snap: forward_own_decode_nvfp4 only writes
        # K_cache[full_rank, cur_pos:cur_pos+1] across the 16 full-attn
        # layers, so the (16, 1, 4, 256) slice is the entire footprint
        # we need to restore.
        snap_K_row = self._attn.K_cache[
            :, cur_pos:cur_pos + 1].clone()
        snap_V_row = self._attn.V_cache[
            :, cur_pos:cur_pos + 1].clone()

        def _restore_on_gs():
            self._lin_state.copy_(self._snap_lin_buf)
            self._lin_conv_state.copy_(self._snap_conv_buf)
            self._attn.K_cache[
                :, cur_pos:cur_pos + 1].copy_(snap_K_row)
            self._attn.V_cache[
                :, cur_pos:cur_pos + 1].copy_(snap_V_row)

        # Warmup (2 iters) to settle allocator + kernel-chain order.
        with torch.no_grad():
            for _ in range(2):
                self.forward_own_decode_nvfp4(
                    self._static_token_id, cos, sin, cur_pos,
                )
            _restore_on_gs()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
                g, stream=gs, pool=self._graph_mempool,
        ), torch.no_grad():
            self.forward_own_decode_nvfp4(
                self._static_token_id, cos, sin, cur_pos,
            )
        with torch.no_grad():
            _restore_on_gs()

        self._graph_cache_put(self._captured_graphs, cur_pos, g)
        return g

    # ---------- Stage 7 G1: NVFP4 verify-forward graph capture ----------

    def _ensure_verify_graph_nvfp4(self, cur_pos: int, K: int):
        """Lazy CUDA Graph capture for forward_own_decode_K_nvfp4.

        Mirror of FP8 ``_ensure_verify_graph`` for the NVFP4 verify
        path. Each (cur_pos, K) pair gets its own graph — FA2 bakes
        kv_seq=cur_pos+k+1 into the K serial q_seq=1 calls so the
        captured kernel call list is cur_pos-specific.

        Caller (spec loop) copies token_ids / cos / sin into
        _verify_static_tokens / _cos / _sin, then replays.

        State integrity: snap+restore lin_state / lin_conv_state and
        the K rows of K/V cache that the verify forward writes, so
        the captured graph's replay produces the same N+1-step state
        mutation each time it's called from the spec loop.
        """
        import torch

        key = (cur_pos, K)
        g = self._graph_cache_get(self._captured_verify_graphs, key)
        if g is not None:
            return g

        gs = self._graph_stream

        snap_lin = self._lin_state.clone()
        snap_conv = self._lin_conv_state.clone()
        snap_K = self._attn.K_cache[:, cur_pos:cur_pos + K].clone()
        snap_V = self._attn.V_cache[:, cur_pos:cur_pos + K].clone()

        def _restore():
            self._lin_state.copy_(snap_lin)
            self._lin_conv_state.copy_(snap_conv)
            self._attn.K_cache[
                :, cur_pos:cur_pos + K].copy_(snap_K)
            self._attn.V_cache[
                :, cur_pos:cur_pos + K].copy_(snap_V)

        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.no_grad():
            tokens_K = self._verify_static_tokens[:, :K]
            cos_K = self._verify_static_cos[:, :K]
            sin_K = self._verify_static_sin[:, :K]
            for _ in range(2):
                self.forward_own_decode_K_nvfp4(
                    tokens_K, cos_K, sin_K, cur_pos, K=K)
                _restore()

        gs.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
                g, stream=gs, pool=self._graph_mempool,
        ), torch.no_grad():
            self.forward_own_decode_K_nvfp4(
                tokens_K, cos_K, sin_K, cur_pos, K=K)
        with torch.cuda.stream(gs), torch.no_grad():
            _restore()
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)

        self._graph_cache_put(self._captured_verify_graphs, key, g)
        return g

    # ---------- Stage 7 G2: NVFP4 MTP chain graph capture ----------

    def _ensure_mtp_graph_nvfp4(self, mtp_pos: int):
        """Lazy CUDA Graph capture for forward_mtp_head_nvfp4 at mtp_pos.

        Mirror of FP8 ``_ensure_mtp_graph`` for NVFP4. Reads from
        _mtp_static_prev_h / _mtp_static_prev_token, writes to
        _mtp_layer_out_buf / _mtp_logits_buf. Each mtp_pos gets its
        own graph because FA2's kv_seq=mtp_pos+1 is baked in.

        Snap+restore the single MTP K/V cache row at mtp_pos so the
        warmup/capture writes don't perturb the live cache.
        """
        import torch

        g = self._graph_cache_get(self._captured_mtp_graphs, mtp_pos)
        if g is not None:
            return g

        gs = self._graph_stream

        snap_K = self._mtp_K_cache[mtp_pos:mtp_pos + 1].clone()
        snap_V = self._mtp_V_cache[mtp_pos:mtp_pos + 1].clone()

        def _restore():
            self._mtp_K_cache[mtp_pos:mtp_pos + 1].copy_(snap_K)
            self._mtp_V_cache[mtp_pos:mtp_pos + 1].copy_(snap_V)

        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.no_grad():
            for _ in range(2):
                self.forward_mtp_head_nvfp4(
                    self._mtp_static_prev_h,
                    self._mtp_static_prev_token,
                    mtp_pos)
                _restore()

        gs.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
                g, stream=gs, pool=self._graph_mempool,
        ), torch.no_grad():
            self.forward_mtp_head_nvfp4(
                self._mtp_static_prev_h,
                self._mtp_static_prev_token,
                mtp_pos)
        with torch.cuda.stream(gs), torch.no_grad():
            _restore()
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)

        self._graph_cache_put(self._captured_mtp_graphs, mtp_pos, g)
        return g

    # ---------- G9: NVFP4 MTP CHAIN graph (K steps in one graph) ------

    def _ensure_mtp_chain_graph_nvfp4(self, base_pos: int, K: int):
        """Capture the entire K-step MTP chain as ONE CUDA Graph.

        Inputs (filled by caller before replay):
          self._mtp_static_prev_h        — h_main at start of cycle
          self._mtp_static_prev_token    — last confirmed token

        Outputs (read by caller after replay):
          self._chain_drafts_buf[:K]     — argmax of each step's logits
          self._mtp_layer_out_buf        — final h_mtp (used to chain
                                           into next spec cycle)

        Internal: between steps, this graph copies _mtp_layer_out_buf
        into _mtp_static_prev_h and the previous draft into
        _mtp_static_prev_token. All copies are recorded as kernel
        calls inside the graph — no Python in the inner loop at replay
        time.

        State integrity: each MTP step writes to mtp_K_cache[base_pos+k]
        / mtp_V_cache[base_pos+k]. Snap+restore those K rows so the
        warmup/capture writes don't perturb the live cache.
        """
        import torch

        key = (base_pos, K)
        g = self._graph_cache_get(self._captured_chain_graphs, key)
        if g is not None:
            return g

        gs = self._graph_stream

        snap_K = self._mtp_K_cache[base_pos:base_pos + K].clone()
        snap_V = self._mtp_V_cache[base_pos:base_pos + K].clone()

        def _restore():
            self._mtp_K_cache[base_pos:base_pos + K].copy_(snap_K)
            self._mtp_V_cache[base_pos:base_pos + K].copy_(snap_V)

        def _run_chain():
            for k in range(K):
                self.forward_mtp_head_nvfp4(
                    self._mtp_static_prev_h,
                    self._mtp_static_prev_token,
                    base_pos + k)
                # argmax → drafts buf row k
                d_k = self._mtp_logits_buf.argmax(
                    dim=-1, keepdim=True).view(1, 1)
                self._chain_drafts_buf[k:k + 1].copy_(d_k.view(1, 1))
                if k < K - 1:
                    # Chain: prev_h ← layer_out, prev_token ← d_k
                    self._mtp_static_prev_h.copy_(
                        self._mtp_layer_out_buf)
                    self._mtp_static_prev_token.copy_(
                        self._chain_drafts_buf[k:k + 1])

        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.no_grad():
            for _ in range(2):
                _run_chain()
                _restore()

        gs.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
                g, stream=gs, pool=self._graph_mempool,
        ), torch.no_grad():
            _run_chain()
        with torch.cuda.stream(gs), torch.no_grad():
            _restore()
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)

        self._graph_cache_put(self._captured_chain_graphs, key, g)
        return g

    # ==================================================================
    # Long-context auto-route (NVFP4 only)
    # ==================================================================

    def _enter_long_ctx_mode(self) -> None:
        """Switch a freshly-initialised NVFP4 frontend onto the TQ path.

        Called from ``__init__`` when ``max_seq > LONG_CTX_THRESHOLD``.
        Buffers were sized at the threshold; this method extends KV
        coverage out to the user-requested max_seq via the TurboQuant
        packed cache and shrinks the now-unused BF16 KV cache to a
        64-row dummy.

        Sequence (matches the canonical pattern from the team's own
        long-ctx benches, but without exposing the private steps to
        callers):

          1. Drop the BF16 KV cache (its rows are now stand-ins; the
             TQ packed cache is the source of truth).
          2. Extend the precomputed RoPE table out to the user's
             max_seq + a small slack (the BF16 path's rope table was
             sized at the threshold).
          3. Allocate the TQ packed cache + BF16 single-layer staging
             at ``max_seq_tq = user_max_seq + 16``.

        After this returns, ``forward_own_decode_nvfp4_tq`` (eager) is
        the correct decode entry. Captured-graph TQ replay is not
        used here because its KV-cache write is skipped at capture
        time, so replay only produces correct attention if the slot
        was already populated for *this exact token* — fine for
        bench (synthetic-fill, fixed token) but wrong for serving.
        """
        # b_v=4, b_k_total=4, bit_packed=True matches the long-ctx
        # bench config (docs/qwen36_nvfp4.md §4): 1.83x compression
        # at 1-byte idx → ~5x at bit-pack = the documented profile.
        self._shrink_bf16_kv_cache(new_max_seq=64)
        self._extend_rope_table_to(self._user_max_seq + 16)
        self._load_turboquant_packed(
            max_seq_tq=self._user_max_seq + 16,
            b_v=4, b_k_total=4, bit_packed=True,
        )

    def _generate_long_ctx_single_token(
            self, input_ids, max_new_tokens: int):
        """Long-ctx fallback for ``generate_own_speculative_KN_nvfp4``.

        Single-token decode through the eager TQ forward — supports
        any prompt length up to ``self._user_max_seq`` and any output
        length up to the same bound. Slower than spec (~30-40 tok/s
        decode at 8-32 K ctx, dropping to ~20 tok/s at 256 K) but
        works at every context length the TQ packed cache covers.

        Spec decode on the TQ path is the Phase 3D follow-up; until
        that lands, calling
        ``generate_own_speculative_KN_nvfp4(..., K=N)`` in long-ctx
        mode silently uses K=1 and logs a one-time info line.
        """
        import torch

        prompt_len = int(input_ids.shape[1])
        max_pos = prompt_len + int(max_new_tokens)
        if max_pos > self._user_max_seq:
            raise ValueError(
                f'prompt_len ({prompt_len}) + max_new_tokens '
                f'({max_new_tokens}) = {max_pos} exceeds the '
                f'frontend max_seq ({self._user_max_seq})'
            )

        self.reset_state()
        if not hasattr(self, '_rope_cos_table'):
            self._build_rope_table()

        generated = list(input_ids[0].tolist())
        cur_pos = 0
        with torch.no_grad():
            # Prefill: one TQ forward per prompt token.
            for p in range(prompt_len):
                tok = input_ids[:, p:p + 1]
                cos, sin = self._rope_cos_sin(cur_pos)
                self.forward_own_decode_nvfp4_tq(
                    tok, cos, sin, cur_pos)
                cur_pos += 1
            # First decoded token = argmax of last prefill step.
            tok = self._logits_buf.argmax(
                dim=-1, keepdim=True).view(1, 1)
            generated.append(int(tok.item()))
            # Decode loop.
            for _ in range(int(max_new_tokens) - 1):
                cos, sin = self._rope_cos_sin(cur_pos)
                self.forward_own_decode_nvfp4_tq(
                    tok, cos, sin, cur_pos)
                tok = self._logits_buf.argmax(
                    dim=-1, keepdim=True).view(1, 1)
                generated.append(int(tok.item()))
                cur_pos += 1

        return torch.tensor(
            [generated], device=input_ids.device, dtype=input_ids.dtype)

    # ==================================================================
    # N7-B4: TurboQuant KV cache (Phase 2B — long context to 200K+)
    # ==================================================================

    def _load_turboquant_kv(self, b_v: int = 3, b_k_total: int = 3,
                             base_seed: int = 0xC0FFEE) -> None:
        """Initialize TurboQuant KV cache (opt-in).

        Reads only on the NVFP4 path for now; FP8 follow-up in B5.
        After this call, ``self._tq_inject_enabled = True`` activates
        the per-write TQ roundtrip injection that validates whether
        TQ-encoded K/V breaks downstream attention quality. The actual
        VRAM-saving packed-cache integration is B6 (replaces BF16 K_cache
        with packed storage + chunked dequant).
        """
        from flash_rt.frontends.torch._qwen36_rtx_turboquant import (
            TurboQuantSetup,
        )
        if not hasattr(self, '_tq_setup'):
            self._tq_setup = TurboQuantSetup(
                num_layers=16, head_dim=256, base_seed=base_seed,
                device=self.device,
                b_v=b_v, b_k_total=b_k_total,
            )
        self._tq_inject_enabled = True

    def _extend_rope_table_to(self, target_max_seq: int) -> None:
        """Rebuild _rope_cos_table / _rope_sin_table at target_max_seq.
        Used when the frontend was init'd at a smaller max_seq (to keep
        BF16 K_cache small) but TQ path needs longer ctx via packed
        cache."""
        import torch

        if self._quant_format != 'nvfp4':
            raise RuntimeError('_extend_rope_table_to is NVFP4-only')
        head_dim = int(self._cfg['head_dim'])
        partial = float(self._cfg['partial_rotary_factor'])
        theta = float(self._cfg['rope_theta'])
        rope_dim = int(head_dim * partial)
        inv_freq = 1.0 / (theta ** (
            torch.arange(0, rope_dim, 2,
                         device=self.device, dtype=torch.float32)
            / rope_dim))
        positions = torch.arange(
            target_max_seq, device=self.device, dtype=torch.float32)
        freqs = positions[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        self._rope_cos_table = emb.cos().to(torch.bfloat16).contiguous()
        self._rope_sin_table = emb.sin().to(torch.bfloat16).contiguous()
        self._rope_dim = rope_dim

    def _shrink_bf16_kv_cache(self, new_max_seq: int = 64) -> None:
        """B6 helper: shrink the BF16 _attn.K_cache/V_cache to a tiny
        size, freeing the (16 layers × max_seq × 4 × 256 × 2) BF16 KV
        bytes (= 32 KB × max_seq) which the TQ path no longer uses.

        At max_seq=64K this frees ~4 GB; at 256K it frees ~16 GB.
        Also frees the snap K/V buffers and per-cur_pos captured graphs
        if any (they reference K_cache pointers that get invalidated).

        Caller MUST be on the TQ path (forward_own_decode_nvfp4_tq).
        Existing forward_own_decode_nvfp4 will fail since it reads from
        K_cache/V_cache directly.
        """
        import torch

        # Free captured graphs (their kernels reference K_cache pointers)
        if hasattr(self, '_captured_graphs'):
            self._captured_graphs.clear()
        if hasattr(self, '_captured_verify_graphs'):
            self._captured_verify_graphs.clear()
        if hasattr(self, '_captured_verify_graphs_dflash'):
            self._captured_verify_graphs_dflash.clear()
        if hasattr(self, '_captured_graphs_tq'):
            self._captured_graphs_tq.clear()
        # β: per-layer staging is now stale (no slot was actually
        # rewritten, but the shrink call is the canonical "TQ
        # bookkeeping reset" hook).
        if hasattr(self, '_tq_cache_packed'):
            self._tq_cache_packed.invalidate_all()
        bf16 = torch.bfloat16
        d = self.device
        nl = self._attn.NUM_FULL_LAYERS
        nkv = self._attn.NUM_KV_HEADS
        hd = self._attn.HEAD_DIM
        # Replace with tiny dummies (just need valid pointers, never read).
        self._attn.K_cache = torch.empty(
            nl, new_max_seq, nkv, hd, dtype=bf16, device=d)
        self._attn.V_cache = torch.empty_like(self._attn.K_cache)
        # Snap buffers used by spec orchestration — also tiny since TQ
        # path doesn't snap BF16 KV (uses TQ packed instead).
        if hasattr(self, '_snap_K_buf'):
            del self._snap_K_buf, self._snap_V_buf
            self._snap_K_buf = torch.empty(
                nl, self.MAX_Q_SEQ, nkv, hd, dtype=bf16, device=d)
            self._snap_V_buf = torch.empty_like(self._snap_K_buf)
        # _attn._max_seq is referenced by FA2 — keep at staging size (max_seq_tq)
        # which is set later by _load_turboquant_packed. For now just track.
        self._attn._max_seq_orig = self._attn._max_seq

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _load_turboquant_packed(self, max_seq_tq: int = 65536,
                                  b_v: int = 4, b_k_total: int = 4,
                                  bit_packed: bool = False) -> None:
        """B6: allocate packed TurboQuant KV cache + BF16 dequant staging.

        Switches to TQ-only KV storage (drops BF16 K_cache for the
        TQ path). Persistent storage is TQ packed (smaller per-token);
        attention reads dequant a single-layer staging tensor.

        Sizes (1-byte idx, no bit-pack yet):
          per-token cache: 16 layers × 4 heads × 550 bytes ≈ 35 KB
          per-token BF16 baseline (16L × 4H × 2K = 32) × 2 (K+V) = 64KB
          1.83x compression. Bit-pack (B8) gets ~5x.

        Single-layer BF16 staging (reused across all 16 layers):
          max_seq_tq × 4 × 256 × 2 (K+V) bytes
          = 4 KB × max_seq_tq × 2 ≈ 8KB × max_seq_tq
          At max_seq_tq=64K: 512 MB; at 256K: 2 GB.

        VRAM accounting (NVFP4 main = 17 GB):
          max_seq_tq=64K  cache + stage: 2.2+0.5 = 2.7 GB → fits
          max_seq_tq=128K            : 4.5+1.0 = 5.5 GB → fits
          max_seq_tq=256K (1-byte)   : 9.0+2.0 = 11  GB → tight
          max_seq_tq=256K (bit-pack) : 2.5+2.0 = 4.5 GB → comfortable
        """
        import torch

        from flash_rt.frontends.torch._qwen36_rtx_turboquant import (
            TurboQuantKVCache,
            TurboQuantSetup,
        )
        if not hasattr(self, '_tq_setup'):
            self._tq_setup = TurboQuantSetup(
                num_layers=16, head_dim=256, device=self.device,
                b_v=b_v, b_k_total=b_k_total,
            )
        if not hasattr(self, '_tq_cache_packed'):
            self._tq_cache_packed = TurboQuantKVCache(
                self._tq_setup, max_seq=max_seq_tq, num_kv=4,
                device=self.device, packed=bit_packed,
            )
            # Single-layer BF16 staging, reused per attn call (fallback
            # path for ctx > BETA_MAX_SEQ).
            self._tq_k_stage = torch.empty(
                max_seq_tq, 4, 256,
                dtype=torch.bfloat16, device=self.device)
            self._tq_v_stage = torch.empty_like(self._tq_k_stage)
            self._tq_max_seq = max_seq_tq

            # Phase 3B-β: per-layer BF16 staging for incremental dequant.
            # Enabled only at ≤ BETA_MAX_SEQ (β-0.2 probe: 64K user ctx
            # = 65552 max_seq leaves 6.8 GB headroom — safe; 128K user
            # ctx leaves 1 GB and OOMs under forward overhead).  100000
            # is the cushion-safe upper bound for 64K user ctx.  Above
            # threshold, fallback to the shared single-layer stage above.
            BETA_MAX_SEQ = 100000
            self._tq_use_per_layer = (max_seq_tq <= BETA_MAX_SEQ)
            if self._tq_use_per_layer:
                self._tq_k_stage_per_layer = torch.empty(
                    16, max_seq_tq, 4, 256,
                    dtype=torch.bfloat16, device=self.device)
                self._tq_v_stage_per_layer = torch.empty_like(
                    self._tq_k_stage_per_layer)

    def _tq_dequant_into_stage(self, layer: int, end_pos: int) -> None:
        """Batched dequant TQ cache layer rows [0, end_pos) into BF16
        staging buffers _tq_k_stage / _tq_v_stage.

        Phase 3A B9: when the kernels are available and the cache is in
        packed (B8) layout, use the CUDA unpack + cuBLAS GEMM + combine
        fast path (≈ 0.68 ms @ 32K vs 3.0 ms Python).  Otherwise fall
        back to the Python read_kv path.

        Phase 3B α-S3 (CUTLASS): set FVK_QWEN36_TQ_CUTLASS=1 to route
        through CUTLASS EVT-fused dequant — eliminates the fp32
        K_pre/V_pre intermediate buffers, ~1.6-2× per-call speedup at
        production ctx.  B8 16/16 verified.  Same precision as B9
        (fp32 acc + bf16 cast at output) but no intermediate.
        """
        cache = self._tq_cache_packed
        try:
            from flash_rt import flash_rt_kernels as _fvk
            fast_ok = (cache.packed
                       and hasattr(_fvk, 'tq_unpack_packed_bf16')
                       and hasattr(_fvk, 'tq_combine_kv_bf16'))
            cutlass_ok = (fast_ok
                          and hasattr(_fvk, 'tq_cutlass_k_combine')
                          and hasattr(_fvk, 'tq_cutlass_v_combine'))
        except ImportError:
            fast_ok = False
            cutlass_ok = False
        if cutlass_ok and os.environ.get('FVK_QWEN36_TQ_CUTLASS', '1') == '1':
            self._tq_dequant_cutlass(layer, end_pos)
            return
        if fast_ok:
            cache.read_kv_fast(layer, end_pos,
                               self._tq_k_stage, self._tq_v_stage)
            return
        k_hat, v_hat = cache.read_kv(layer, end_pos)
        self._tq_k_stage[:end_pos].copy_(k_hat)
        self._tq_v_stage[:end_pos].copy_(v_hat)

    def _tq_dequant_cutlass(self, layer: int, end_pos: int) -> None:
        """Phase 3B α-S3: CUTLASS EVT-fused dequant.

        Pipeline:
          1. tq_unpack_packed_bf16: packed → bf16 yk, yv, qjl
          2. cuBLAS qjl @ jl_bf16 → bf16 Sr → cast fp32 (CUTLASS aux)
          3. CUTLASS K_combine: yk @ Π^T + Sr scaling → bf16 K_stage
          4. CUTLASS V_combine: yv @ Π^T + norm scaling → bf16 V_stage

        Pre-cast Π^T bf16 and jl bf16 cached on TurboQuantSetup (per
        layer, computed lazily on first call).  Sr GEMM still uses
        cuBLAS bf16 — folding it into CUTLASS is a follow-up.
        """
        import math
        import torch

        from flash_rt import flash_rt_kernels as fvk

        cache = self._tq_cache_packed
        setup = self._tq_setup
        d = setup.head_dim
        nkv = cache.num_kv
        M = end_pos * nkv
        coef = math.sqrt(math.pi / 2.0) / d
        s = torch.cuda.current_stream().cuda_stream

        # Pre-cast per-layer Π^T bf16 (input to CUTLASS K/V combine).
        # jl stays fp32 for the Sr matmul to preserve B9-grade precision
        # (bf16 jl drifts B8 to 2/16; fp32 keeps it bit-stable).
        if not hasattr(setup, '_cu_rot_T_bf16'):
            setup._cu_rot_T_bf16 = [
                R.t().contiguous().bfloat16().contiguous()
                for R in setup.rotations]

        # Pre-allocate per-cache scratch (capped to max_seq * num_kv).
        # Phase 3.5b: qjl is unpacked directly to fp32 via the mixed
        # kernel — saves the bf16→fp32 cast (~192 MB BW / call at 32K).
        if not hasattr(cache, '_cu_yk'):
            cap_M = cache.max_seq * nkv
            t_bf = lambda: torch.empty(   # noqa: E731
                cap_M, d, dtype=torch.bfloat16, device=cache.device)
            t_fp = lambda: torch.empty(   # noqa: E731
                cap_M, d, dtype=torch.float32, device=cache.device)
            cache._cu_yk = t_bf()
            cache._cu_yv = t_bf()
            cache._cu_qjl_fp32 = t_fp()    # written by mixed unpack
            cache._cu_sr_fp32 = t_fp()
            cache._cu_coef_rnorm_fp32 = torch.empty(
                cap_M, dtype=torch.float32, device=cache.device)
            cache._cu_norm_k_fp32 = torch.empty(
                cap_M, dtype=torch.float32, device=cache.device)
            cache._cu_norm_v_fp32 = torch.empty(
                cap_M, dtype=torch.float32, device=cache.device)

        yk = cache._cu_yk[:M]
        yv = cache._cu_yv[:M]
        qjl_fp = cache._cu_qjl_fp32[:M]
        sr_fp = cache._cu_sr_fp32[:M]
        coef_rnorm = cache._cu_coef_rnorm_fp32[:M]
        norm_k_fp32 = cache._cu_norm_k_fp32[:M]
        norm_v_fp32 = cache._cu_norm_v_fp32[:M]

        cb_k = setup.codebooks[setup.b_k_mse]
        cb_v = setup.codebooks[setup.b_v]

        # 1. Mixed unpack: bf16 yk, fp32 qjl, bf16 yv (single kernel,
        # cast fused — no separate qjl bf16→fp32 promote needed).
        fvk.tq_unpack_packed_mixed(
            cache.k_idx[layer, :end_pos].data_ptr(),
            cache.k_qjl[layer, :end_pos].data_ptr(),
            cache.v_idx[layer, :end_pos].data_ptr(),
            cb_k.data_ptr(), cb_v.data_ptr(),
            yk.data_ptr(), qjl_fp.data_ptr(), yv.data_ptr(),
            M, setup.b_k_mse, setup.b_v, s,
        )

        # 2. Compute per-row scalars (in fp32, on device).
        # cache.k_norm/k_rnorm are fp16; cast to fp32 for CUTLASS aux load.
        norm_k_fp32.copy_(cache.k_norm[layer, :end_pos].view(-1).float())
        norm_v_fp32.copy_(cache.v_norm[layer, :end_pos].view(-1).float())
        torch.mul(
            cache.k_rnorm[layer, :end_pos].view(-1).float(), coef,
            out=coef_rnorm)

        # 3. Sr = qjl_fp32 @ jl_fp32 → fp32 (B9 precision, no bf16 cast).
        jl_fp32_layer = setup.jl[layer]
        torch.matmul(qjl_fp, jl_fp32_layer, out=sr_fp)

        # 4. CUTLASS K combine: yk @ Π^T → bf16 K_stage with combine inline.
        rot_T = setup._cu_rot_T_bf16[layer]
        fvk.tq_cutlass_k_combine(
            yk.data_ptr(), rot_T.data_ptr(),
            sr_fp.data_ptr(),
            norm_k_fp32.data_ptr(), coef_rnorm.data_ptr(),
            self._tq_k_stage[:end_pos].data_ptr(),
            M, d, d, s,
        )
        # 5. CUTLASS V combine: yv @ Π^T → bf16 V_stage.
        fvk.tq_cutlass_v_combine(
            yv.data_ptr(), rot_T.data_ptr(),
            norm_v_fp32.data_ptr(),
            self._tq_v_stage[:end_pos].data_ptr(),
            M, d, d, s,
        )

    def _layer_forward_full_nvfp4_tq(self, L: int, h_in, cos, sin,
                                       cur_pos: int):
        """B6: full-attn layer forward using packed TurboQuant KV cache.

        Mirror of ``_layer_forward_full_nvfp4`` but:
          - K/V cache writes go to ``self._tq_cache_packed`` (TQ packed)
          - Attention reads dequantized K/V from single-layer BF16
            staging buffers; calls ``_fa2_fwd`` directly with staging
            pointers (not via ``self._attn.run('full', ...)`` since
            that path hardcodes reading from ``_attn.K_cache``).
        """
        import torch

        from flash_rt import flash_rt_kernels as fvk

        s = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        assert lw['type'] == 'full_attention'

        h2 = h_in.view(1, 5120).contiguous()
        eps = float(self._cfg['rms_norm_eps'])
        full_rank = self._full_layer_rank(L)

        # ---- 1) input layernorm + NVFP4 quantize ----
        x_norm = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            h2.data_ptr(), int(lw['input_norm_eff_w']),
            x_norm.data_ptr(), 1, 5120, eps, s,
        )
        ap_5120, sf_5120, _ = self._nvfp4_scratch[(12288, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_norm.data_ptr(), ap_5120.data_ptr(),
            sf_5120.data_ptr(), 1, 5120, s,
        )

        # ---- 2) q_proj fused -> (1, 12288) ----
        q_proj_out_buf = self._nvfp4_scratch[(12288, 5120)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['q_proj_packed']),
            q_proj_out_buf.data_ptr(),
            1, 12288, 5120,
            sf_5120.data_ptr(), int(lw['q_proj_sf']),
            float(lw['q_proj_alpha']),
            s,
        )
        q_full = q_proj_out_buf[:1].view(1, 1, 24, 512)
        q_pre, gate = torch.chunk(q_full, 2, dim=-1)
        gate_flat = gate.reshape(1, 1, 24 * 256)

        # ---- 3) k_proj -> (1, 1024) ----
        kv_proj_out_buf = self._nvfp4_scratch[(1024, 5120)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['k_proj_packed']),
            kv_proj_out_buf.data_ptr(),
            1, 1024, 5120,
            sf_5120.data_ptr(), int(lw['k_proj_sf']),
            float(lw['k_proj_alpha']),
            s,
        )
        k_pre = kv_proj_out_buf[:1].view(1, 1, 4, 256).contiguous()

        # ---- 4) q_norm / k_norm ----
        q_pre_2d = q_pre.contiguous().view(24, 256)
        fvk.rms_norm(
            q_pre_2d.data_ptr(), int(lw['q_norm_eff_w']),
            self._full_q_norm_out.data_ptr(), 24, 256, eps, s,
        )
        k_pre_2d = k_pre.view(4, 256)
        fvk.rms_norm(
            k_pre_2d.data_ptr(), int(lw['k_norm_eff_w']),
            self._full_k_norm_out.data_ptr(), 4, 256, eps, s,
        )

        # ---- 5) inline RoPE ----
        q_for_rope = self._full_q_norm_out.view(1, 1, 24, 256)
        k_for_rope = self._full_k_norm_out.view(1, 1, 4, 256)
        cos4 = cos.view(1, 1, 1, 64)
        sin4 = sin.view(1, 1, 1, 64)

        def _rope_inline(x_in, x_out, tmp):
            x_out[..., 64:].copy_(x_in[..., 64:])
            torch.index_select(
                x_in[..., :64], -1, self._rope_rotate_idx, out=tmp,
            )
            tmp[..., :32].neg_()
            tmp.mul_(sin4)
            tmp.addcmul_(x_in[..., :64], cos4)
            x_out[..., :64].copy_(tmp)

        _rope_inline(q_for_rope, self._full_q_rot, self._full_rope_tmp_q)
        _rope_inline(k_for_rope, self._full_k_rot, self._full_rope_tmp_k)
        q_rot = self._full_q_rot
        k_rot = self._full_k_rot

        # ---- 6) v_proj ----
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['v_proj_packed']),
            kv_proj_out_buf.data_ptr(),
            1, 1024, 5120,
            sf_5120.data_ptr(), int(lw['v_proj_sf']),
            float(lw['v_proj_alpha']),
            s,
        )
        v_new = kv_proj_out_buf[:1].view(1, 4, 256)

        # ---- 7) Write K/V to TQ packed cache (no BF16 cache write) ----
        # k_rot shape (1, 1, 4, 256) -> (1, 4, 256)
        k_for_cache = k_rot.view(1, 4, 256)
        v_for_cache = v_new
        cache = self._tq_cache_packed
        # B9-S10: bit-exact capture-safe write_kv_fast (4 small CUDA
        # kernels + 3 cuBLAS GEMMs via torch.matmul).  Auto-route when
        # available — read-back cosine 1.000000 vs Python ref.
        if cache.packed and hasattr(fvk, 'tq_write_k1_unit_norm'):
            cache.write_kv_fast(
                full_rank, cur_pos, cur_pos + 1, k_for_cache, v_for_cache)
        else:
            cache.write_kv(
                full_rank, cur_pos, cur_pos + 1, k_for_cache, v_for_cache)

        # ---- 8) Dequant TQ[0:cur_pos+1] into BF16 staging ----
        kv_seq = cur_pos + 1
        self._tq_dequant_into_stage(full_rank, kv_seq)

        # ---- 9) Stage Q + run FA2 directly (read from staging) ----
        self._attn.Q_buf[:, :1].copy_(q_rot)
        scaling = float(self._cfg['head_dim']) ** -0.5

        # Build views for _fa2_fwd: K, V shape (1, kv_seq, 4, 256).
        k_view = self._tq_k_stage[:kv_seq].view(1, kv_seq, 4, 256)
        v_view = self._tq_v_stage[:kv_seq].view(1, kv_seq, 4, 256)
        q_view = self._attn.Q_buf[:, :1]  # (1, 1, 24, 256)
        o_view = self._attn.O_buf[:, :1]
        self._attn._fa2_fwd(
            Q=q_view.data_ptr(), K=k_view.data_ptr(),
            V=v_view.data_ptr(), O=o_view.data_ptr(),
            softmax_lse=self._attn.lse_buf.data_ptr(),
            softmax_lse_accum=self._attn.lse_accum.data_ptr(),
            o_accum=self._attn.o_accum.data_ptr(),
            batch=1, seqlen_q=1, seqlen_k=kv_seq,
            num_heads_q=self._attn.NUM_Q_HEADS,
            num_heads_kv=self._attn.NUM_KV_HEADS,
            head_dim=self._attn.HEAD_DIM,
            q_strides=(q_view.stride(0), q_view.stride(1), q_view.stride(2)),
            k_strides=(k_view.stride(0), k_view.stride(1), k_view.stride(2)),
            v_strides=(v_view.stride(0), v_view.stride(1), v_view.stride(2)),
            o_strides=(o_view.stride(0), o_view.stride(1), o_view.stride(2)),
            softmax_scale=scaling,
            num_sms=self._attn._num_sms,
            stream=s,
        )
        attn_out = self._attn.O_buf[:, :1]

        # ---- 10) output gate + o_proj + residual + post-attn + MLP ----
        # (rest identical to _layer_forward_full_nvfp4)
        attn_flat = attn_out.reshape(1, 1, 24 * 256)
        torch.sigmoid(gate_flat, out=self._full_gate_sig)
        torch.mul(attn_flat, self._full_gate_sig, out=self._full_gated)
        gated = self._full_gated

        ap_6144, sf_6144, _ = self._nvfp4_scratch[(5120, 6144)]
        gated_2d = gated.view(1, 6144).contiguous()
        fvk.quantize_bf16_to_nvfp4_swizzled(
            gated_2d.data_ptr(), ap_6144.data_ptr(),
            sf_6144.data_ptr(), 1, 6144, s,
        )
        out_op_buf = self._nvfp4_scratch[(5120, 6144)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_6144.data_ptr(), int(lw['o_proj_packed']),
            out_op_buf.data_ptr(),
            1, 5120, 6144,
            sf_6144.data_ptr(), int(lw['o_proj_sf']),
            float(lw['o_proj_alpha']),
            s,
        )

        attn_proj = out_op_buf[:1].view(1, 1, 5120)
        torch.add(h_in, attn_proj, out=self._res_mid)
        h_post = self._res_mid

        h_post_view = h_post.view(1, 5120)
        x_mlp = self._h_b[:1].view(1, 5120)
        fvk.rms_norm(
            h_post_view.data_ptr(), int(lw['post_attn_norm_eff_w']),
            x_mlp.data_ptr(), 1, 5120, eps, s,
        )

        ap_mlp, sf_mlp, _ = self._nvfp4_scratch[(17408, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_mlp.data_ptr(), ap_mlp.data_ptr(),
            sf_mlp.data_ptr(), 1, 5120, s,
        )
        gate_out_buf = self._nvfp4_scratch[(17408, 5120)][2]
        up_out_buf = self._mlp_up_out
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_mlp.data_ptr(), int(lw['mlp_gate_packed']),
            gate_out_buf.data_ptr(),
            1, 17408, 5120,
            sf_mlp.data_ptr(), int(lw['mlp_gate_sf']),
            float(lw['mlp_gate_alpha']),
            s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_mlp.data_ptr(), int(lw['mlp_up_packed']),
            up_out_buf.data_ptr(),
            1, 17408, 5120,
            sf_mlp.data_ptr(), int(lw['mlp_up_sf']),
            float(lw['mlp_up_alpha']),
            s,
        )
        gate_v = gate_out_buf[:1].view(1, 17408)
        up_v = up_out_buf[:1].view(1, 17408)
        fvk.silu_mul_qwen36_bf16(
            gate_v.data_ptr(), up_v.data_ptr(),
            self._mlp_silu_mul_out.data_ptr(), 17408, s,
        )
        gate_silu_up = self._mlp_silu_mul_out

        ap_dn, sf_dn, _ = self._nvfp4_scratch[(5120, 17408)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            gate_silu_up.data_ptr(), ap_dn.data_ptr(),
            sf_dn.data_ptr(), 1, 17408, s,
        )
        down_out_buf = self._nvfp4_scratch[(5120, 17408)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_dn.data_ptr(), int(lw['mlp_down_packed']),
            down_out_buf.data_ptr(),
            1, 5120, 17408,
            sf_dn.data_ptr(), int(lw['mlp_down_sf']),
            float(lw['mlp_down_alpha']),
            s,
        )
        mlp_out = down_out_buf[:1].view(1, 1, 5120)

        h_out = self._layer_out_a if (L % 2 == 0) else self._layer_out_b
        torch.add(h_post, mlp_out, out=h_out)
        return h_out

    def forward_own_decode_nvfp4_tq(self, token_id, cos_pos, sin_pos,
                                      cur_pos: int):
        """B6: full forward decode using TQ packed cache for full-attn.

        Lin-attn layers stay on existing path (state-based, no per-token
        KV cache). Full-attn layers route through
        ``_layer_forward_full_nvfp4_tq``.
        """
        import torch

        from flash_rt import flash_rt_kernels as fvk

        bf16 = torch.bfloat16
        s = torch.cuda.current_stream().cuda_stream
        types = self._cfg['layer_types']
        eps = float(self._cfg['rms_norm_eps'])
        hidden = self._cfg['hidden_size']
        vocab = self._cfg['vocab_size']

        if not isinstance(token_id, torch.Tensor):
            token_id = torch.tensor(
                [token_id], device=self.device, dtype=torch.long)
        if token_id.ndim == 1:
            token_id = token_id.view(1, 1)
        embed_t = self._weights.anchors[0]
        h = embed_t[token_id.view(-1)].view(1, 1, hidden).contiguous()
        if h.dtype != bf16:
            h = h.to(bf16)

        for L in range(self._cfg['num_hidden_layers']):
            t = types[L]
            if t == 'linear_attention':
                h = self._layer_forward_lin_nvfp4(L, h)
            elif t == 'full_attention':
                h = self._layer_forward_full_nvfp4_tq(
                    L, h, cos_pos, sin_pos, cur_pos)
            else:
                raise ValueError(f'unknown layer_type {t!r} at L={L}')

        self._last_hidden_buf.copy_(h)
        h2 = h.view(1, hidden).contiguous()
        x_norm = self._h_b[:1].view(1, hidden)
        fvk.rms_norm(
            h2.data_ptr(), int(self._weights.ptrs['final_norm_eff_w']),
            x_norm.data_ptr(), 1, hidden, eps, s,
        )
        ap, sf, _ = self._nvfp4_scratch[(10240, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_norm.data_ptr(), ap.data_ptr(), sf.data_ptr(),
            1, hidden, s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap.data_ptr(), int(self._weights.ptrs['lm_head_packed']),
            self._logits_buf.data_ptr(),
            1, vocab, hidden,
            sf.data_ptr(), int(self._weights.ptrs['lm_head_sf']),
            float(self._weights.ptrs['lm_head_alpha']),
            s,
        )
        return self._logits_buf

    # ---------- B9-S9: NVFP4 TQ forward CUDA Graph capture ----------

    def _ensure_graph_nvfp4_tq(self, cur_pos: int):
        """Lazy CUDA-Graph capture of forward_own_decode_nvfp4_tq.

        Mirror of _ensure_graph_nvfp4 but for the long-ctx TQ path:
          - K/V writes go to _tq_cache_packed (not _attn.K_cache)
          - dequant uses the read_kv_fast path (pre-allocated scratch)
          - FA2 reads from _tq_k/v_stage staging buffers

        Each cur_pos gets its own graph because:
          1. FA2 bakes kv_seq=cur_pos+1 into the captured kernel call list
          2. read_kv_fast bakes pos_end=cur_pos+1 into pointer math
          3. cos/sin slice addresses are cur_pos-specific

        State integrity: lin-attn state + lin-conv state are mutated by
        each call (they're recurrent), so we snapshot+restore.  The TQ
        cache slot at cur_pos is overwritten with the same content on
        each warmup iter (deterministic given the same token), so no
        restore needed there.
        """
        import torch

        if not hasattr(self, '_captured_graphs_tq'):
            self._captured_graphs_tq: collections.OrderedDict[
                int, torch.cuda.CUDAGraph,
            ] = collections.OrderedDict()
        g = self._graph_cache_get(self._captured_graphs_tq, cur_pos)
        if g is not None:
            return g

        gs = self._graph_stream
        cos, sin = self._rope_cos_sin(cur_pos)

        state_snap = {
            'lin_state': self._lin_state.clone(),
            'lin_conv_state': self._lin_conv_state.clone(),
        }

        def _restore_on_gs():
            self._lin_state.copy_(state_snap['lin_state'])
            self._lin_conv_state.copy_(state_snap['lin_conv_state'])

        # Warmup (2 iters) on the capture stream — settles allocator +
        # kernel-chain order.  Calls forward_own_decode_nvfp4_tq directly
        # so read_kv_fast's lazy scratch is alloc'd once before capture.
        # During warmup, write_kv runs eagerly to populate the cache slot
        # at cur_pos.  During capture, _tq_skip_write_capture skips the
        # Python pack-op path (which creates fresh tensors via torch
        # ops — incompatible with capture).  Replay relies on the cache
        # slot already containing the right content from a prior eager
        # write_kv call (or from the warmup).
        with torch.no_grad():
            for _ in range(2):
                self.forward_own_decode_nvfp4_tq(
                    self._static_token_id, cos, sin, cur_pos,
                )
            _restore_on_gs()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
                g, stream=gs, pool=self._graph_mempool,
        ), torch.no_grad():
            self.forward_own_decode_nvfp4_tq(
                self._static_token_id, cos, sin, cur_pos,
            )
        with torch.no_grad():
            _restore_on_gs()

        self._graph_cache_put(self._captured_graphs_tq, cur_pos, g)
        return g

    def forward_own_decode_nvfp4_tq_captured(
            self, token_id, cos_pos, sin_pos, cur_pos: int):
        """Graph-captured TQ decode.  Replays a per-cur_pos CUDA Graph;
        captures lazily on first hit.  Returns _logits_buf (live tensor).

        cos_pos/sin_pos are required by signature parity with the
        eager variant but are NOT used during replay — the captured
        graph already encodes the cur_pos-specific cos/sin pointers.
        Caller is responsible for filling _static_token_id with the
        token to decode before calling.
        """
        import torch

        if not isinstance(token_id, torch.Tensor):
            token_id = torch.tensor(
                [token_id], device=self.device, dtype=torch.long)
        self._static_token_id.copy_(token_id.view(1, 1))
        g = self._ensure_graph_nvfp4_tq(cur_pos)
        g.replay()
        return self._logits_buf

    def _tq_inject_all_layers(self, pos_start: int, pos_end: int) -> None:
        """Force TQ roundtrip on K/V cache rows [pos_start, pos_end)
        across all 16 full-attn layers, bypassing the graph-capture
        guard. Use after prefill (which uses captured graphs and skips
        the per-write inject) to populate prompt positions with TQ
        noise so long-context bench is realistic.
        """
        import torch

        if not getattr(self, '_tq_inject_enabled', False):
            return
        kc = self._attn.K_cache
        vc = self._attn.V_cache
        for L in range(self._attn.NUM_FULL_LAYERS):
            k_orig = kc[L, pos_start:pos_end]
            v_orig = vc[L, pos_start:pos_end]
            idx_k, qjl_k, norm_k, rnorm_k = self._tq_setup.quant_k(
                k_orig, L)
            k_hat = self._tq_setup.dequant_k(
                idx_k, qjl_k, norm_k, rnorm_k, L)
            idx_v, norm_v = self._tq_setup.quant_v(v_orig, L)
            v_hat = self._tq_setup.dequant_v(idx_v, norm_v, L)
            kc[L, pos_start:pos_end].copy_(k_hat.to(torch.bfloat16))
            vc[L, pos_start:pos_end].copy_(v_hat.to(torch.bfloat16))

    def _tq_inject_kv(self, full_rank: int, cur_pos: int,
                       count: int = 1) -> None:
        """In-place TQ roundtrip on the just-written K/V cache rows.

        Reads K_cache[full_rank, cur_pos:cur_pos+count] and
        V_cache[full_rank, cur_pos:cur_pos+count], runs Q_prod (K) +
        Q_mse (V) round-trip, and writes the dequantized vectors back.
        This is equivalent to having the cache stored as TQ packed
        from the start — every vector takes one round-trip in its
        lifetime, identical to the long-term TQ cache behavior.

        Used for B3 quality validation. No VRAM saving (B6 will do
        that with a proper packed cache).
        """
        import torch

        if not getattr(self, '_tq_inject_enabled', False):
            return
        # Skip during CUDA Graph capture so existing graphs stay clean.
        if torch.cuda.is_current_stream_capturing():
            return
        kc = self._attn.K_cache
        vc = self._attn.V_cache
        k_orig = kc[full_rank, cur_pos:cur_pos + count]   # (count, 4, 256)
        v_orig = vc[full_rank, cur_pos:cur_pos + count]
        idx_k, qjl_k, norm_k, rnorm_k = self._tq_setup.quant_k(
            k_orig, full_rank)
        k_hat = self._tq_setup.dequant_k(
            idx_k, qjl_k, norm_k, rnorm_k, full_rank)
        idx_v, norm_v = self._tq_setup.quant_v(v_orig, full_rank)
        v_hat = self._tq_setup.dequant_v(idx_v, norm_v, full_rank)
        kc[full_rank, cur_pos:cur_pos + count].copy_(
            k_hat.to(torch.bfloat16))
        vc[full_rank, cur_pos:cur_pos + count].copy_(
            v_hat.to(torch.bfloat16))

    # ==================================================================
    # N6-A4: DFlash spec decode (block-diffusion drafter + chain verify)
    # ==================================================================

    def _load_dflash_drafter(self, ckpt_dir: str | None = None) -> None:
        """Load the z-lab/Qwen3.6-27B-DFlash drafter (NVFP4 W4A16).

        Reads the drafter's BF16 safetensors, quantizes every linear
        projection to NVFP4 swizzled at load via G7 kernel, attaches the
        drafter dict at ``self._weights.ptrs['dflash']``, allocates
        per-shape drafter scratch on ``self._dflash_buf``, and pre-
        allocates the verify-time hidden-tap capture buffer
        ``self._dflash_taps_buf`` (shape: 5 × MAX_Q_SEQ × hidden bf16).

        Set the drafter ckpt dir via the ``ckpt_dir`` argument or the
        ``FLASHRT_QWEN36_DFLASH_CKPT_DIR`` env var (raises if neither is
        provided). Idempotent.
        """
        import os

        import torch

        from flash_rt import flash_rt_kernels as fvk
        from flash_rt.frontends.torch._qwen36_rtx_dflash_weights import (
            assert_dflash_extraction_invariants,
            extract_dflash_weights_nvfp4,
        )
        from flash_rt.frontends.torch._qwen36_rtx_dflash_forward import (
            alloc_drafter_scratch,
        )

        if self._weights is None or self._cfg.get('layer_types') is None:
            raise RuntimeError(
                'DFlash requires the NVFP4 main path to be loaded first')

        if self._weights.ptrs.get('dflash') is None:
            if ckpt_dir is None:
                ckpt_dir = os.environ.get(
                    'FLASHRT_QWEN36_DFLASH_CKPT_DIR')
                if not ckpt_dir:
                    raise RuntimeError(
                        'DFlash drafter ckpt path is required: pass '
                        'ckpt_dir= or set '
                        'FLASHRT_QWEN36_DFLASH_CKPT_DIR')
            extract_dflash_weights_nvfp4(
                self._weights, ckpt_dir, fvk, device=self.device)
            assert_dflash_extraction_invariants(self._weights)

        # Per-shape S=16 drafter scratch (idempotent)
        alloc_drafter_scratch(self, device=self.device)

        # Hidden-tap capture buffer for the main verify forward
        # (5 tap layers × MAX_Q_SEQ rows × hidden, bf16). Lives outside
        # _dflash_buf so it's available even when drafter forward isn't
        # called (e.g. for scoring tap statistics in tests).
        if not hasattr(self, '_dflash_taps_buf'):
            hidden = self._cfg['hidden_size']
            self._dflash_taps_buf = torch.empty(
                len(self._DFLASH_TAP_LAYERS), self.MAX_Q_SEQ, hidden,
                device=self.device, dtype=torch.bfloat16)

        # Per-(cur_pos, K) graph cache for the tap-capturing verify
        # forward. Distinct from the no-tap graph cache so we never
        # replay a tap-writing graph against a None tap_buf.
        if not hasattr(self, '_captured_verify_graphs_dflash'):
            self._captured_verify_graphs_dflash: collections.OrderedDict[
                tuple[int, int], torch.cuda.CUDAGraph,
            ] = collections.OrderedDict()
        # P7: per-eff_ctx drafter forward graph cache. Each eff_ctx
        # value gets its own graph because shapes (target_feat_window
        # rows, kv_seq) are baked in.
        if not hasattr(self, '_captured_drafter_graphs_dflash'):
            self._captured_drafter_graphs_dflash: collections.OrderedDict[
                int, torch.cuda.CUDAGraph,
            ] = collections.OrderedDict()

    def _ensure_drafter_graph_dflash_nvfp4(self, eff_ctx: int):
        """P7: Lazy CUDA Graph capture for the entire drafter forward.

        Captures dflash_drafter_forward_capture (which reads from
        ids_static, hidden_taps_static, target_feat_window) at the
        given eff_ctx. The captured graph collapses ~220 per-call
        kernel launches into one replay launch.
        """
        import torch

        from flash_rt.frontends.torch._qwen36_rtx_dflash_forward import (
            alloc_drafter_capture_window,
            dflash_drafter_forward_capture,
        )

        g = self._graph_cache_get(
            self._captured_drafter_graphs_dflash, eff_ctx)
        if g is not None:
            return g

        alloc_drafter_capture_window(self, eff_ctx)
        gs = self._graph_stream

        # Snap state we'll mutate: target_feat_window contents.
        snap_window = self._dflash_buf['target_feat_window'].clone()

        def _restore():
            self._dflash_buf['target_feat_window'].copy_(snap_window)

        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.no_grad():
            for _ in range(2):
                dflash_drafter_forward_capture(self)
                _restore()
        gs.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
                g, stream=gs, pool=self._graph_mempool,
        ), torch.no_grad():
            dflash_drafter_forward_capture(self)
        with torch.cuda.stream(gs), torch.no_grad():
            _restore()
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)

        self._graph_cache_put(
            self._captured_drafter_graphs_dflash, eff_ctx, g)
        return g

    def _ensure_verify_graph_dflash_nvfp4(self, cur_pos: int, K: int):
        """Lazy CUDA Graph for forward_own_decode_K_nvfp4 WITH tap_buf.

        Mirror of ``_ensure_verify_graph_nvfp4`` but binds
        ``tap_buf=self._dflash_taps_buf`` at capture time so the 5
        tap-layer copies are baked into the graph. The replay reads from
        the same ``_verify_static_*`` buffers as the no-tap variant.

        Each (cur_pos, K) pair gets its own graph because FA2 bakes
        kv_seq=cur_pos+i into the captured kernel call list.
        """
        import torch

        key = (cur_pos, K)
        g = self._graph_cache_get(
            self._captured_verify_graphs_dflash, key)
        if g is not None:
            return g

        gs = self._graph_stream

        snap_lin = self._lin_state.clone()
        snap_conv = self._lin_conv_state.clone()
        snap_K = self._attn.K_cache[:, cur_pos:cur_pos + K].clone()
        snap_V = self._attn.V_cache[:, cur_pos:cur_pos + K].clone()

        def _restore():
            self._lin_state.copy_(snap_lin)
            self._lin_conv_state.copy_(snap_conv)
            self._attn.K_cache[
                :, cur_pos:cur_pos + K].copy_(snap_K)
            self._attn.V_cache[
                :, cur_pos:cur_pos + K].copy_(snap_V)

        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.no_grad():
            tokens_K = self._verify_static_tokens[:, :K]
            cos_K = self._verify_static_cos[:, :K]
            sin_K = self._verify_static_sin[:, :K]
            tap_buf = self._dflash_taps_buf
            for _ in range(2):
                self.forward_own_decode_K_nvfp4(
                    tokens_K, cos_K, sin_K, cur_pos, K=K,
                    tap_buf=tap_buf)
                _restore()

        gs.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
                g, stream=gs, pool=self._graph_mempool,
        ), torch.no_grad():
            self.forward_own_decode_K_nvfp4(
                tokens_K, cos_K, sin_K, cur_pos, K=K,
                tap_buf=tap_buf)
        with torch.cuda.stream(gs), torch.no_grad():
            _restore()
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)

        self._graph_cache_put(
            self._captured_verify_graphs_dflash, key, g)
        return g

    def generate_own_speculative_DFlash_nvfp4(
            self, input_ids, *, max_new_tokens: int, K: int = 15):
        """DFlash spec decode on the NVFP4 path.

        K = number of speculative tokens per cycle (= block_size - 1).
        Each cycle:
          1. Snap state.
          2. Drafter forward (one S=16 NVFP4 transformer pass) consumes
             [last_committed_token + MASK x 15] + 5 hidden taps from the
             previous main verify (or zeros on first cycle), produces 16
             candidate tokens via per-row argmax.
          3. Main verify forward at S=K+1=16 over
             [last_committed_token, draft_0..draft_{K-1}], with hidden
             taps captured into self._dflash_taps_buf.
          4. Argmax + accept-prefix N (largest N s.t. main argmax[i] ==
             draft[i] for i in 0..N-1). N in [0, K].
          5. Commit verify_argmax[0..N]; on partial accept (N<K) restore
             pre-verify state and re-advance with N+1 valid inputs.
          6. Move to next cycle with prev_token = verify_argmax[N] and
             taps = self._dflash_taps_buf[:, N].

        Returns:
            (1, prompt_len + N) cuda long, trimmed to max_new_tokens.
        """
        import torch

        from flash_rt.frontends.torch._qwen36_rtx_dflash_forward import (
            alloc_drafter_capture_window,
            reset_drafter_capture_state,
        )

        if self._weights.ptrs.get('dflash') is None:
            raise RuntimeError(
                'DFlash drafter not loaded — call _load_dflash_drafter '
                'first or set FLASHRT_QWEN36_DFLASH_CKPT_DIR')
        if K < 1 or K + 1 > self.MAX_Q_SEQ:
            raise ValueError(
                f'K={K} out of range — need 1<=K<={self.MAX_Q_SEQ - 1}')

        prompt_len = int(input_ids.shape[1])

        self.reset_state()
        if not hasattr(self, '_rope_cos_table'):
            self._build_rope_table()
        # P7: prepare capture window (eff_ctx-sized shift buffer) and
        # clear shift state. eff_ctx defaults to 16 (sweet spot per P5).
        eff_ctx = int(getattr(self, '_dflash_eff_ctx', 16))
        alloc_drafter_capture_window(self, eff_ctx)
        reset_drafter_capture_state(self)
        # Initialize taps to zero — first drafter call gets no real
        # signal; AL on cycle 0 will be lower than steady-state.
        self._dflash_taps_buf.zero_()

        with torch.no_grad():
            # 1) Prefill (same as MTP path) — sequential S=1 forwards
            # via the per-cur_pos captured S=1 graph.
            gs_pf = self._graph_stream
            for p in range(prompt_len):
                self._static_token_id.copy_(input_ids[:, p:p + 1])
                g_pf = self._ensure_graph_for_pos_nvfp4(p)
                gs_pf.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(gs_pf):
                    g_pf.replay()
                torch.cuda.current_stream().wait_stream(gs_pf)
            tok = self._logits_buf.argmax(
                dim=-1, keepdim=True).view(1, 1)
            generated = [tok]
            cur_pos = prompt_len

            self._spec_attempts = 0
            self._spec_accepts = 0
            self._spec_full = 0

            d = self._rope_dim
            Kv = K + 1

            # 2) Spec decode loop
            while len(generated) < max_new_tokens:
                # 2a) Snap main state (overlap with drafter on default).
                snap_stream = self._snap_stream
                snap_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(snap_stream):
                    self._snap_lin_buf.copy_(self._lin_state)
                    self._snap_conv_buf.copy_(self._lin_conv_state)
                    self._snap_K_buf[:, :Kv].copy_(
                        self._attn.K_cache[
                            :, cur_pos:cur_pos + Kv])
                    self._snap_V_buf[:, :Kv].copy_(
                        self._attn.V_cache[
                            :, cur_pos:cur_pos + Kv])

                # 2b) Drafter forward (P7).
                # Caller writes static inputs (prev_token + hidden_taps).
                # During ramp-up (first eff_ctx cycles) the window is
                # not yet fully populated -> use eager forward with the
                # actual valid_ctx so attention only sees real history
                # (avoids zero-dilution that hurts AL). Once the window
                # is full, replay the captured graph.
                self._dflash_buf['ids_static'][0:1].copy_(tok.view(1))
                self._dflash_buf['hidden_taps_static'].copy_(
                    self._dflash_taps_buf[:, 0])
                if self._spec_attempts < eff_ctx:
                    from flash_rt.frontends.torch._qwen36_rtx_dflash_forward import (  # noqa: E501
                        dflash_drafter_forward_capture_eager,
                    )
                    valid_ctx = self._spec_attempts + 1
                    dflash_drafter_forward_capture_eager(self, valid_ctx)
                else:
                    drafter_g = self._ensure_drafter_graph_dflash_nvfp4(
                        eff_ctx)
                    drafter_g.replay()
                draft_logits = self._dflash_buf['logits']
                draft_tokens = draft_logits.argmax(dim=-1)   # (16,)
                # block_size=16 layout: input[0]=prev_token (verify of
                # self, position cur_pos-1), input[1..15]=MASK (positions
                # cur_pos..cur_pos+14). draft output[i] predicts position
                # cur_pos-1+i; we want predictions of cur_pos..cur_pos+K-1
                # so take output[1:K+1].
                drafts = draft_tokens[1:K + 1]               # (K,)

                # Wait for snap before verify mutates state.
                torch.cuda.current_stream().wait_stream(snap_stream)

                # 2c) Main verify (S=K+1=16) WITH tap capture.
                cos_KN = self._rope_cos_table[
                    cur_pos:cur_pos + Kv].view(1, Kv, d)
                sin_KN = self._rope_sin_table[
                    cur_pos:cur_pos + Kv].view(1, Kv, d)
                self._verify_static_tokens[:, 0:1].copy_(tok)
                self._verify_static_tokens[:, 1:Kv].copy_(
                    drafts.view(1, K))
                self._verify_static_cos[:, :Kv].copy_(cos_KN)
                self._verify_static_sin[:, :Kv].copy_(sin_KN)
                vg = self._ensure_verify_graph_dflash_nvfp4(cur_pos, Kv)
                gs = self._graph_stream
                gs.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(gs):
                    vg.replay()
                torch.cuda.current_stream().wait_stream(gs)
                logits_KN = self._K_logits_buf[:Kv]

                # 2d) Argmax + accept-prefix
                all_argmax = logits_KN.argmax(dim=-1)        # (Kv,) long
                matches = (all_argmax[:K] == drafts).long()
                matches_pad = torch.cat([
                    matches,
                    torch.zeros(1, device=matches.device,
                                dtype=matches.dtype),
                ])
                N = int(matches_pad.argmin().item())
                self._spec_attempts += 1
                self._spec_accepts += N

                argmax_at = (lambda j: all_argmax[j:j + 1].view(1, 1))

                if N == K:
                    self._spec_full += 1
                    for j in range(Kv):
                        if len(generated) < max_new_tokens:
                            generated.append(argmax_at(j))
                    tok = argmax_at(K)
                    # Move taps[K] -> taps[0] for next cycle
                    self._dflash_taps_buf[:, 0].copy_(
                        self._dflash_taps_buf[:, K])
                    cur_pos += Kv
                else:
                    for j in range(N + 1):
                        if len(generated) < max_new_tokens:
                            generated.append(argmax_at(j))
                    # Restore pre-verify state.
                    self._lin_state.copy_(self._snap_lin_buf)
                    self._lin_conv_state.copy_(self._snap_conv_buf)
                    self._attn.K_cache[
                        :, cur_pos:cur_pos + Kv].copy_(
                            self._snap_K_buf[:, :Kv])
                    self._attn.V_cache[
                        :, cur_pos:cur_pos + Kv].copy_(
                            self._snap_V_buf[:, :Kv])

                    # Re-advance with N+1 valid inputs via tapped verify
                    # at K=N+1 (always — including N=0; same code path
                    # as N>0). Re-uses the dflash verify graph cache.
                    Kr = N + 1
                    rec_cos = cos_KN[:, :Kr]
                    rec_sin = sin_KN[:, :Kr]
                    self._verify_static_tokens[:, 0:1].copy_(tok)
                    if N > 0:
                        self._verify_static_tokens[:, 1:Kr].copy_(
                            drafts[:N].view(1, N))
                    self._verify_static_cos[:, :Kr].copy_(rec_cos)
                    self._verify_static_sin[:, :Kr].copy_(rec_sin)
                    rg = self._ensure_verify_graph_dflash_nvfp4(
                        cur_pos, Kr)
                    gs.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(gs):
                        rg.replay()
                    torch.cuda.current_stream().wait_stream(gs)
                    tok = argmax_at(N)
                    self._dflash_taps_buf[:, 0].copy_(
                        self._dflash_taps_buf[:, N])
                    cur_pos += Kr

            if len(generated) > max_new_tokens:
                generated = generated[:max_new_tokens]

        return torch.cat([input_ids] + generated, dim=1)

    def _layer_types(self) -> list:
        """Source-agnostic layer_types accessor (works for both quant paths)."""
        if self._quant_format == 'nvfp4':
            return self._cfg['layer_types']
        return self._pipeline.hf.config.layer_types

    def _full_layer_rank(self, L: int) -> int:
        """Return how many full-attn layers exist before index L."""
        cache = getattr(self, '_full_rank_cache', None)
        if cache is None:
            types = self._layer_types()
            cache = {}
            r = 0
            for i, t in enumerate(types):
                if t == 'full_attention':
                    cache[i] = r
                    r += 1
            self._full_rank_cache = cache
        return cache[L]

    def _linear_layer_rank(self, L: int) -> int:
        """Return how many linear-attn layers exist before index L.

        Used to slice into self._lin_state / self._lin_conv_state, which
        are sized to (48, ...) -- one entry per linear-attn layer in
        layer-index order, not absolute layer index.
        """
        cache = getattr(self, '_lin_rank_cache', None)
        if cache is None:
            types = self._layer_types()
            cache = {}
            r = 0
            for i, t in enumerate(types):
                if t == 'linear_attention':
                    cache[i] = r
                    r += 1
            self._lin_rank_cache = cache
        return cache[L]

    def reset_state(self) -> None:
        """Zero linear-attn state caches and full-attn KV cache.

        Call between independent prompts. Cheap (no allocation).
        """
        if self._bufs is None:
            return
        self._lin_state.zero_()
        self._lin_conv_state.zero_()
        if self._attn is not None:
            self._attn.reset_cache()
        # β: independent prompt → per-layer dequant stage is stale.
        if hasattr(self, '_tq_cache_packed'):
            self._tq_cache_packed.invalidate_all()

    def buffer_summary(self) -> dict:
        """Return a summary of pre-allocated buffer sizes (debug / tests)."""
        if self._bufs is None:
            return {}
        import torch
        total = 0
        items: list[tuple[str, int]] = []

        def _add(name: str, t: torch.Tensor) -> None:
            n = t.element_size() * t.numel()
            items.append((name, n))
            nonlocal total
            total += n

        _add('h_a', self._h_a)
        _add('h_b', self._h_b)
        _add('logits', self._logits_buf)
        _add('lin_state', self._lin_state)
        _add('lin_conv_state', self._lin_conv_state)
        for (N, K), (qinp, sc, out) in self._fp8_scratch.items():
            _add(f'fp8_{N}x{K}_qinp', qinp)
            _add(f'fp8_{N}x{K}_scale', sc)
            _add(f'fp8_{N}x{K}_out', out)
        # attn backend buffers
        _add('attn_K_cache', self._attn.K_cache)
        _add('attn_V_cache', self._attn.V_cache)
        _add('attn_Q_buf', self._attn.Q_buf)
        _add('attn_O_buf', self._attn.O_buf)
        _add('attn_lse_buf', self._attn.lse_buf)
        _add('attn_lse_accum', self._attn.lse_accum)
        _add('attn_o_accum', self._attn.o_accum)

        return {
            'total_bytes': total,
            'total_mb': total / (1024 * 1024),
            'items': items,
        }

    # ---------- public API ----------

    @property
    def pipeline(self) -> Qwen36Pipeline:
        """Underlying Qwen36Pipeline. Tests / advanced users only."""
        if self._pipeline is None:
            raise RuntimeError(
                'Qwen36TorchFrontendRtx not initialized; _load_hf_model '
                'was not called'
            )
        return self._pipeline

    def set_prompt(self, prompt: str) -> None:
        """Tokenize ``prompt`` and stash it for the next ``infer()`` call."""
        if self._tokenizer is None:
            raise RuntimeError('tokenizer not loaded')
        ids = self._tokenizer(prompt, return_tensors='pt').input_ids
        self._prompt_ids = ids.to(self.device)

    def infer(self, input_ids: Any | None = None) -> Any:
        """Run a single forward pass and return logits.

        Args:
            input_ids: optional override for the cached prompt ids.

        Returns:
            ``logits`` tensor of shape ``(B, S, vocab_size)`` bf16.
        """
        import time
        import torch

        ids = input_ids if input_ids is not None else self._prompt_ids
        if ids is None:
            raise RuntimeError(
                'No prompt set. Call set_prompt() first or pass input_ids.'
            )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = self.pipeline.forward(ids)
        torch.cuda.synchronize()
        self.latency_records.append((time.perf_counter() - t0) * 1000.0)
        return logits

    def generate(self, *, max_new_tokens: int = 64, do_sample: bool = False) -> Any:
        """Greedy/sampled autoregressive generate from cached prompt."""
        if self._prompt_ids is None:
            raise RuntimeError('No prompt set. Call set_prompt() first.')
        return self.pipeline.generate(
            self._prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    def decode(self, token_ids: Any) -> str:
        """Detokenize token-id tensor / list."""
        if self._tokenizer is None:
            raise RuntimeError('tokenizer not loaded')
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=False,
        )
