"""NxDI port of Qwen3.6-35B-A3B (text stack).

Architecture
------------
40 decoder layers in a repeating [DN, DN, DN, GQA] x 10 pattern:
  * 30 Gated DeltaNet layers -- linear recurrent attention, causal conv1d on QKV,
    delta-rule update, gated output
  * 10 standard GQA layers -- 16 Q heads, 2 KV heads, head_dim=256, partial RoPE
    on 64 of 256 dims, sigmoid output gate

Every layer's FFN is a MoE block: 256 routed experts (top-8) plus 1 always-on shared
expert, both with intermediate dim 512. Routing uses fp32 softmax over normalized
top-k affinities. An MTP head sits alongside the main stack -- one extra decoder
layer, RMSNorms and an LM head -- invoked off the main traced forward as a drafter.

State / caching
---------------
GQA layers use NxDI's KVCacheManager. DeltaNet layers carry their recurrent state and
1-D conv state as nn.Parameter buffers, and return dummy KV tuples from forward.
"""

import contextlib
import copy
import gc
import math
import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
    NeuronFusedSpecModel,
    mask_padded_logits,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronx_distributed.parallel_layers import mappings, parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.operators.argmax import argmax as nxd_argmax
from neuronx_distributed.utils import cpu_mode

try:
    from nki import jit as nki_jit  # NKI 0.3.0+ (SDK 2.29)
except ImportError:
    from torch_neuronx.xla_impl.ops import nki_jit  # NKI 0.2.x (SDK 2.28)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from flash_rt.neuron._vendor.qwen36.nki_kernels import (
    deltanet_recurrent_fwd as _deltanet_nki_kernel,
    deltanet_recurrent_fwd_state as _deltanet_nki_kernel_state,
    deltanet_chunk_step as _deltanet_nki_chunk_step,
    deltanet_fused_chunked_fwd as _deltanet_fused_kernel,
    _make_lower_mask,
    _make_lower_mask_diag,
    _make_identity,
    deltanet_fused_tkg_fwd as deltanet_fused_kernel,
    deltanet_fused_tkg_fwd_state as deltanet_fused_kernel_state,
    deltanet_attention_layer_state,
    gqa_fused_tkg_fwd,
    moe_layer_fwd,
)
from flash_rt.neuron._vendor.qwen36.nki_kernels.megakernel import (
    DN_FIELDS,
    GQA_FIELDS,
    MOE_FIELDS,
    build_draft_megakernel,
    build_round_megakernel,
    build_verify_megakernel,
    flatten_draft_args,
    flatten_megakernel_args,
    flatten_round_args,
    split_megakernel_returns,
    split_round_returns,
)

from neuronx_distributed_inference.models.config import (
    FusedSpecNeuronConfig,
    InferenceConfig,
    MoENeuronConfig,
)
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    FUSED_SPECULATION_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    DecoderModelInstance,
    ModelWrapper,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
    KVCacheManager,
)
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

logger = logging.getLogger(__name__)

try:
    _flash_fwd_call = nki_jit()(attention_isa_kernel)
except TypeError:
    from torch_neuronx.xla_impl.ops import nki_jit as _torch_xla_nki_jit

    _flash_fwd_call = _torch_xla_nki_jit()(attention_isa_kernel)

# head_dim=256 needs an nkilib flash-attention kernel. Two ways to obtain one:
# (1) load nkilib's attention_cte directly from a local fork (USE_NKILIB_KERNEL),
# or (2) detect a global monkeypatch of _pre_prod_kernels (NKILIB_PATCH_ACTIVE).
USE_NKILIB_KERNEL = os.environ.get("USE_NKILIB_KERNEL", "0") == "1"

_nkilib_flash_attn = None
if USE_NKILIB_KERNEL:
    try:
        import neuronxcc.nki as _nki
        from neuronx_distributed_inference.modules.attention.attention_base import (
            peel_decorations as _peel_decorations,
            get_platform_target as _get_platform_target,
        )
        from neuronxcc.nki.compiler import (
            skip_middle_end_transformations as _skip_middle_end,
            enable_stack_allocator as _enable_stack_allocator,
        )

        import importlib

        _fork_path = "/home/ubuntu/nki-library-fork/nkilib_src"
        if os.path.isdir(_fork_path) and _fork_path not in sys.path:
            sys.path.insert(0, _fork_path)
        _to_remove = [k for k in sys.modules if k.startswith("nkilib")]
        for k in _to_remove:
            del sys.modules[k]
        import nki.language as _stub_nl
        import neuronxcc.nki.language as _real_nl

        for _attr in [
            "NKIObject",
            "float8_e4m3fn",
            "float8_e4m3fn_x4",
            "float8_e5m2_x4",
            "float4_e2m1fn_x4",
        ]:
            if not hasattr(_real_nl, _attr) and hasattr(_stub_nl, _attr):
                setattr(_real_nl, _attr, getattr(_stub_nl, _attr))
        from nkilib.core.attention.attention_cte import (
            attention_cte as _attention_cte_raw,
            _MAX_HEAD_DIM,
        )

        assert _MAX_HEAD_DIM == 256, (
            f"nkilib fork has _MAX_HEAD_DIM={_MAX_HEAD_DIM}, expected 256. "
            f"System nkilib may have been loaded instead of fork."
        )
        logger.info(
            f"Loaded nkilib attention_cte from fork (_MAX_HEAD_DIM={_MAX_HEAD_DIM})"
        )

        _raw_fn = _peel_decorations(_attention_cte_raw)
        _platform = _get_platform_target()
        _nkilib_flash_attn = _nki.jit(
            _raw_fn,
            mode="torchxla",
            platform_target=_platform,
            show_compiler_tb=True,
            debug_kernel=True,
        )
        _nkilib_flash_attn = _skip_middle_end(_nkilib_flash_attn)
        _nkilib_flash_attn = _enable_stack_allocator(
            _nkilib_flash_attn, log_level=logging.INFO
        )
        logger.info("nkilib flash attention loaded for head_dim > 128")
    except Exception as e:
        logger.warning(f"Failed to load nkilib flash attention: {e}")
        import traceback as _tb

        _tb.print_exc()
        _nkilib_flash_attn = None

# Detect whether _pre_prod_kernels was globally patched with the nkilib kernel.
NKILIB_PATCH_ACTIVE = False
try:
    from importlib import import_module as _import_module

    _attn_mod = _import_module("neuronxcc.nki._pre_prod_kernels.attn_fwd")
    if hasattr(_attn_mod, "_original_attention_nki_kernel_adapter"):
        NKILIB_PATCH_ACTIVE = True
        logger.info("_pre_prod_kernels patched with nkilib kernel")
except Exception:
    pass


class TransposedColumnParallelLinear(ColumnParallelLinear):
    """ColumnParallelLinear whose weight is stored [in, out] instead of [out, in].

    The matmul consumes the weight directly (``x @ W``), so the contraction dim is
    already on the partition axis and no runtime weight transpose is emitted.
    """
    def set_weight_and_bias_config(self) -> None:
        self.weight_shape = (self.input_size, self.output_size_per_partition)
        self.weight_partition_dim = 1
        if self.add_bias:
            bias_size = (
                self.output_size
                if self.gather_output
                else self.output_size_per_partition
            )
            self.bias_shape = (bias_size,)
            self.bias_partition_dim = 0
        else:
            self.bias_shape = None

    def forward(self, input, slice_indices=None):
        input_parallel = self._cpl_maybe_input_copy_to_tp_region(input)
        weight = (
            self.weight[:, slice_indices] if slice_indices is not None else self.weight
        )
        output_parallel = torch.matmul(input_parallel, weight)
        output = self._cpl_maybe_gather_output(output_parallel)
        if self.skip_bias_add:
            return output, self.bias
        return (output + self.bias) if self.bias is not None else output


# ============================================================
# Newton-Raphson Refined RMSNorm
# ============================================================
USE_NEWTON_RMSNORM = os.environ.get("USE_NEWTON_RMSNORM") == "1"
USE_PYTHON_RMSNORM = os.environ.get("USE_PYTHON_RMSNORM") == "1"


class NewtonRMSNorm(nn.Module):
    """RMSNorm with Newton-Raphson refined rsqrt for improved numerical accuracy."""
    def __init__(self, hidden_size=None, eps=1e-6):
        super().__init__()
        self.weight = None
        if hidden_size is not None:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.hidden_size = hidden_size
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        original_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        y = torch.rsqrt(variance + self.variance_epsilon)
        y = y * (3.0 - (variance + self.variance_epsilon) * y * y) * 0.5
        result = x * y
        if self.weight is not None:
            result = result * self.weight.float()
        return result.to(original_dtype)


class OffsetRMSNorm(nn.Module):
    """Qwen3.5 normalization: add the weight offset and scale in FP32.

    Keep the original BF16 checkpoint weight. Pre-adding one at conversion
    time loses low bits when NxDI subsequently casts the checkpoint to BF16.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        z = x.float()
        z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (z * (1.0 + self.weight.float())).to(x.dtype)


def get_rmsnorm_cls(config=None):
    if getattr(config, 'flashrt_faithful_numerics', False):
        return OffsetRMSNorm
    if cpu_mode() or USE_PYTHON_RMSNORM:
        return Qwen3MoeRMSNorm
    return NewtonRMSNorm if USE_NEWTON_RMSNORM else CustomRMSNorm


def l2norm(x, dim=-1, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)


# ============================================================
# Gated DeltaNet Module (Linear Recurrent Attention)
# ============================================================


class NeuronGatedDeltaNet(nn.Module):
    """Gated DeltaNet linear-recurrent attention.

    HF weight layout (dimensions parameterised by config):
        in_proj_qkv : (2*key_dim + value_dim, hidden_size)
        in_proj_z   : (value_dim, hidden_size)
        in_proj_a   : (num_v_heads, hidden_size)
        in_proj_b   : (num_v_heads, hidden_size)
        conv1d      : (conv_dim, 1, conv_kernel_size)
        A_log       : (num_v_heads,)
        dt_bias     : (num_v_heads,)
        norm        : (head_v_dim,)
        out_proj    : (hidden_size, value_dim)
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        tc = config
        self.flashrt_faithful_numerics = getattr(config, 'flashrt_faithful_numerics', False)

        self.hidden_size = tc.hidden_size
        self.tp_degree = tc.neuron_config.tp_degree
        self.global_num_v_heads = tc.linear_num_value_heads
        self.global_num_k_heads = tc.linear_num_key_heads
        self.head_k_dim = tc.linear_key_head_dim  # 128
        self.head_v_dim = tc.linear_value_head_dim  # 128
        if self.global_num_v_heads % self.tp_degree != 0:
            raise ValueError(
                f"linear_num_value_heads={self.global_num_v_heads} must be divisible "
                f"by tp_degree={self.tp_degree}"
            )
        if self.global_num_k_heads % self.tp_degree != 0:
            raise ValueError(
                f"linear_num_key_heads={self.global_num_k_heads} must be divisible "
                f"by tp_degree={self.tp_degree}"
            )
        self.num_v_heads = self.global_num_v_heads // self.tp_degree
        self.num_k_heads = self.global_num_k_heads // self.tp_degree
        self.global_key_dim = self.head_k_dim * self.global_num_k_heads  # 2048
        self.global_value_dim = self.head_v_dim * self.global_num_v_heads
        self.key_dim = self.head_k_dim * self.num_k_heads  # 512 at TP=4
        self.value_dim = self.head_v_dim * self.num_v_heads  # 1536 at TP=4
        self.conv_kernel_size = tc.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.rms_norm_eps = tc.rms_norm_eps
        self.use_hybrid_cache_manager = getattr(tc, "use_hybrid_cache_manager", False)
        self.use_qwen_hybrid_chunked_prefill = getattr(
            tc, "use_qwen_hybrid_chunked_prefill", False
        )
        self.use_qwen_hybrid_chunked_prefill_nki = getattr(
            tc, "use_qwen_hybrid_chunked_prefill_nki", False
        )
        self.use_tkg_attention_kernel = getattr(tc, "use_tkg_attention_kernel", False)

        # KV cache dummy shape info
        self.head_dim = tc.head_dim  # 256
        # Dummy KV layout must follow the global cache layout (BHDS when transposed).
        self.k_cache_transposed = tc.neuron_config.k_cache_transposed
        tp_degree = tc.neuron_config.tp_degree
        raw_kv_heads = tc.num_key_value_heads
        if raw_kv_heads < tp_degree:
            replicated_kv_heads = tp_degree
        else:
            replicated_kv_heads = raw_kv_heads
        self.kv_heads_per_rank = replicated_kv_heads // tp_degree

        # Conv1d on concatenated QKV (NOT Z).  Store the depthwise kernel in a
        # ColumnParallelLinear parameter container so NxD's checkpoint sharder
        # can split it by output channel.  Forward still uses it as Conv1d
        # weight after unsqueezing the singleton input-channel dimension.
        self.global_conv_dim = self.global_key_dim * 2 + self.global_value_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim  # 2560 at TP=4
        self.conv1d_weight = ColumnParallelLinear(
            self.conv_kernel_size,
            self.global_conv_dim,
            bias=False,
            gather_output=False,
        )

        # The four input projections (qkv|z|a|b) are fused into one contraction-first
        # [hidden, I] weight, so the TKG kernel needs no runtime cat/transpose. A
        # RowParallelLinear container shards the output axis; the state-dict conversion
        # builds the global weight with per-rank column order [qkv_r | z_r | a_r | b_r].
        self.in_proj_fused = RowParallelLinear(
            self.global_conv_dim + self.global_value_dim + 2 * self.global_num_v_heads,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
        )

        # Same parameter-container pattern for per-value-head decay vectors.
        # These are used as vectors in forward but sharded by output dim during
        # checkpoint conversion/loading.
        self.dt_bias_weight = ColumnParallelLinear(
            1,
            self.global_num_v_heads,
            bias=False,
            gather_output=False,
        )
        self.A_log_weight = ColumnParallelLinear(
            1,
            self.global_num_v_heads,
            bias=False,
            gather_output=False,
        )

        # Output norm and projection
        self.norm = Qwen3MoeRMSNorm(self.head_v_dim, eps=self.rms_norm_eps)
        # o_proj weight stored transposed ([value_dim, hidden], sharded on
        # value_dim) so the TKG kernel needs no runtime .t(); _output_project()
        # supplies the TP all-reduce that RowParallelLinear would have done.
        self.out_proj = ColumnParallelLinear(
            self.hidden_size,
            self.global_value_dim,
            bias=False,
            gather_output=False,
        )

        # State buffers for CTE -> TKG carry-over
        alloc_batch_size = getattr(config.neuron_config, "max_batch_size", 1)
        self._phase_batch_size = getattr(config.neuron_config, "batch_size", 1)
        self.recurrent_state_buffer = nn.Parameter(
            torch.zeros(
                alloc_batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                dtype=torch.float32 if self.flashrt_faithful_numerics else config.neuron_config.torch_dtype,
            ),
            requires_grad=False,
        )
        self.conv_state_buffer = nn.Parameter(
            torch.zeros(
                alloc_batch_size,
                self.conv_dim,
                self.conv_kernel_size - 1,
                dtype=config.neuron_config.torch_dtype,
            ),
            requires_grad=False,
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if self.flashrt_faithful_numerics:
            # NxDI casts the model to its matrix precision after construction.
            # The reference retains the recurrent state in FP32 between tokens.
            self.recurrent_state_buffer.data = self.recurrent_state_buffer.data.float()
        return result

    def _dummy_kv(self, batch_size, seq_len, dtype, device):
        """Zero KV placeholders for the KVCacheManager (DeltaNet carries state in
        buffers, not KV). K follows the global cache layout: BHDS when transposed."""
        v = torch.zeros(
            batch_size,
            self.kv_heads_per_rank,
            seq_len,
            self.head_dim,
            dtype=dtype,
            device=device,
        )
        if self.k_cache_transposed:
            k = torch.zeros(
                batch_size,
                self.kv_heads_per_rank,
                self.head_dim,
                seq_len,
                dtype=dtype,
                device=device,
            )
        else:
            k = torch.zeros_like(v)
        return k, v

    def _conv1d_weight(self):
        return self.conv1d_weight.weight.unsqueeze(1)

    def _dt_bias(self):
        return self.dt_bias_weight.weight.squeeze(1)

    def _A_log(self):
        return self.A_log_weight.weight.squeeze(1)

    def _project_inputs(self, hidden_states):
        """Fused input projection: one [hidden, I] matmul, sliced into qkv/z/a/b.
        No transpose -- in_proj_fused.weight is stored contraction-first at load."""
        proj = torch.matmul(hidden_states, self.in_proj_fused.weight)
        z_end = self.conv_dim + self.value_dim
        a_end = z_end + self.num_v_heads
        return (
            proj[..., : self.conv_dim],
            proj[..., self.conv_dim : z_end],
            proj[..., z_end:a_end],
            proj[..., a_end:],
        )

    def _output_project(self, x):
        """o_proj: per-rank (x @ [value_dim, hidden]) then TP all-reduce.
        out_proj.weight is stored transposed at load, so no runtime .t()."""
        out = torch.matmul(x, self.out_proj.weight)
        return mappings.reduce_from_tensor_model_parallel_region(
            out, process_group=parallel_state.get_tensor_model_parallel_group()
        )

    def _recurrent_step(self, query, key, value, g, beta, recurrent_state):
        """Single-step recurrent update for token generation."""
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        scale = 1.0 / (query.shape[-1] ** 0.5)
        query = query * scale

        q_t = query[:, :, 0]
        k_t = key[:, :, 0]
        v_t = value[:, :, 0]
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, 0].unsqueeze(-1)

        new_state = recurrent_state * g_t
        kv_mem = (new_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        new_state = new_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)

        return output.unsqueeze(2), new_state

    def fused_attn_nki(self, qkv, conv_seed, a, b, init_state, decode):
        """Causal conv + gated delta-rule recurrence in one fused NKI launch.

        Collapses the conv and recurrence into a single LNC=2 kernel with q/k/v kept in SBUF.
        ``qkv`` [B, S, conv_dim] is the raw ``in_proj_qkv`` output (token-major); ``conv_seed``
        [B, conv_dim, K-1] is the carried conv window; ``a, b`` [B, S, num_v_heads] are the raw
        ``in_proj_a``/``in_proj_b`` outputs; ``init_state`` [B, num_v_heads, Kd, Vd] is the
        recurrent state seed. The kernel folds in the conv + silu, the per-head q/k/v split,
        l2norm/scale of q/k, GQA head replication, beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias).
        Returns the raw head-major recurrence output [B, S, value_dim] (caller applies the output
        RMSNorm + z-gate) plus -- per ``decode`` -- the new conv/recurrent state ([B, conv_dim, K-1],
        [B, num_v_heads, Kd, Vd]) or the per-position candidate stacks ([B, S, conv_dim, K-1],
        [B, S, num_v_heads, Kd, Vd], candidate axis = S). bs=1; [2] shards the value-heads across
        both LNC=2 cores.
        """
        w = self.conv1d_weight.weight.float()
        qkv2d = qkv[0].float()
        seed2d = conv_seed[0].float()
        A_log = self._A_log().float()
        dt_bias = self._dt_bias().float()
        a2d = a[0].float()
        b2d = b[0].float()
        init2d = init_state[0].float()
        if decode:
            attn, final_state, new_conv_state = deltanet_fused_kernel[2](
                qkv2d, seed2d, w, self.key_dim, a2d, b2d, A_log, dt_bias, init2d
            )
            return (
                attn.unsqueeze(0),
                final_state.unsqueeze(0),
                new_conv_state.unsqueeze(0),
            )
        attn, cand_states, conv_cand = deltanet_fused_kernel_state[2](
            qkv2d, seed2d, w, self.key_dim, a2d, b2d, A_log, dt_bias, init2d
        )
        return attn.unsqueeze(0), cand_states.unsqueeze(0), conv_cand.unsqueeze(0)

    def attn_output(self, attn_raw, z, dtype):
        """Output RMSNorm + z-gate + projection for the kernel attention path.

        ``attn_raw`` [B, S, value_dim] is the raw recurrence output (head-major); ``z``
        [B, S, value_dim] is the raw ``in_proj_z``. The kernel already emits head-major output,
        so the per-head norm is a free reshape -- no transpose.
        """
        batch_size, seq_len, _ = attn_raw.shape
        output = attn_raw.to(dtype)
        output = output.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = self.norm(output)
        z_gate = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = output * F.silu(z_gate)
        output = output.reshape(batch_size, seq_len, self.value_dim)
        return self._output_project(output)

    def _commit_conv_state(
        self, new_conv_state, batch_size, seq_ids, hybrid_cache_active
    ):
        """Cast the kernel's new conv window to the buffer dtype and wire the input_output_alias
        dependency (mirrors the decode forward commit)."""
        new_conv_state = new_conv_state.to(self.conv_state_buffer.dtype)
        alloc_bs = self.conv_state_buffer.shape[0]
        if hybrid_cache_active:
            return new_conv_state
        if seq_ids is not None:
            return new_conv_state + self.conv_state_buffer * 0
        if batch_size < alloc_bs:
            return torch.cat(
                [new_conv_state, self.conv_state_buffer[batch_size:] * 0], dim=0
            )
        return new_conv_state + self.conv_state_buffer * 0

    def _commit_rec_state(self, final_state, batch_size, seq_ids, hybrid_cache_active):
        """Cast the kernel's final recurrent state to the buffer dtype and wire the
        input_output_alias dependency (mirrors the decode forward commit)."""
        final_state = final_state.to(self.recurrent_state_buffer.dtype)
        alloc_bs = self.recurrent_state_buffer.shape[0]
        if hybrid_cache_active:
            return final_state
        if seq_ids is not None:
            return final_state + self.recurrent_state_buffer * 0
        if batch_size < alloc_bs:
            return torch.cat(
                [final_state, self.recurrent_state_buffer[batch_size:] * 0], dim=0
            )
        return final_state + self.recurrent_state_buffer * 0

    def _forward_decode_tkg(
        self,
        hidden_states,
        qkv,
        z,
        a,
        b,
        seq_ids,
        conv_state_cache,
        recurrent_state_cache,
        hybrid_cache_active,
        batch_size,
        seq_len,
    ):
        """Single-token decode (T=1) attention through the fused DeltaNet TKG NKI kernel.

        Glue-free counterpart to the decode branch of ``forward``: one fused conv + recurrence
        launch -> output projection, with no host-side transpose / reshape / head replication.
        Returns the same ``(output, past_kv, new_rec_state, new_conv_state)`` tuple ``forward``
        emits for the decode path.
        """
        if conv_state_cache is not None:
            conv_state = conv_state_cache[:batch_size]
        elif seq_ids is not None:
            conv_state = torch.index_select(self.conv_state_buffer, 0, seq_ids)
        else:
            conv_state = self.conv_state_buffer[:batch_size]
        if recurrent_state_cache is not None:
            init_state = recurrent_state_cache[:batch_size]
        elif seq_ids is not None:
            init_state = torch.index_select(self.recurrent_state_buffer, 0, seq_ids)
        else:
            init_state = self.recurrent_state_buffer[:batch_size]

        attn_raw, final_state, new_conv_state = self.fused_attn_nki(
            qkv, conv_state, a, b, init_state, decode=True
        )
        new_conv_state = self._commit_conv_state(
            new_conv_state, batch_size, seq_ids, hybrid_cache_active
        )
        new_rec_state = self._commit_rec_state(
            final_state, batch_size, seq_ids, hybrid_cache_active
        )

        output = self.attn_output(attn_raw, z, hidden_states.dtype)

        if hybrid_cache_active:
            return (
                output,
                (new_rec_state, new_conv_state),
                new_rec_state,
                new_conv_state,
            )
        # Dummy KV for the KVCacheManager (DeltaNet carries state in the buffers, not KV).
        dummy_k, dummy_v = self._dummy_kv(
            batch_size, seq_len, hidden_states.dtype, hidden_states.device
        )
        return output, (dummy_k, dummy_v), new_rec_state, new_conv_state

    def verify_block_candidates(self, query, key, value, g, beta, initial_state):
        """Run the verify block as exact single-step recurrences seeded from
        ``initial_state``, checkpointing the state after each token so the host
        can select the carried state by accept count.

        Inputs are the projected/conv'd/headed block tensors in the
        ``[B, H, S, dim]`` layout ``_recurrent_step`` expects (g/beta
        ``[B, H, S]``), sliced one position at a time. Returns
        ``(out_stack [B, H, S, Vd], S_stack [B, S, H, Kd, Vd])`` where ``S_i``
        (axis-1) is the recurrent state after consuming token ``i``.
        """
        num_positions = query.shape[2]
        recurrent_state = initial_state
        outputs = []
        states = []
        # Round the carried state to the buffer dtype between steps so it matches
        # sequential TKG decode bit-for-bit (TKG writes the state back as bf16
        # after each token); the next step's math still runs in float32.
        carry_dtype = self.recurrent_state_buffer.dtype
        for i in range(num_positions):
            out_i, recurrent_state = self._recurrent_step(
                query[:, :, i : i + 1],
                key[:, :, i : i + 1],
                value[:, :, i : i + 1],
                g[:, :, i : i + 1],
                beta[:, :, i : i + 1],
                recurrent_state,
            )
            recurrent_state = recurrent_state.to(carry_dtype).float()
            outputs.append(out_i)
            states.append(recurrent_state)

        # out_i is [B, H, 1, Vd]; concat along the position axis -> [B, H, S, Vd].
        out_stack = torch.cat(outputs, dim=2)
        # Each state is [B, H, Kd, Vd]; stack on a new candidate axis-1.
        S_stack = torch.stack(states, dim=1)
        return out_stack, S_stack

    def conv_window_candidates(self, conv_seed, mixed):
        """Per-position causal conv windows for the verify block.

        ``conv_seed`` is the carried conv state ``[B, conv_dim, K-1]`` (the window
        feeding the first block token) and ``mixed`` is the block's conv input
        ``[B, conv_dim, S]`` (channels-first, pre-activation, same tensor whose
        tail is gathered into ``new_conv_state`` in forward()). The conv window
        *after* token ``i`` is the ``K-1`` columns of ``cat([conv_seed, mixed])``
        ending at token ``i``'s column, i.e. ``conv_input[:, :, i+1 : i+K]``.

        Returns ``[B, S, conv_dim, K-1]`` -- one window per block position, so the
        host can carry the conv state matching its chosen accept count.
        """
        state_len = self.conv_kernel_size - 1
        conv_input = torch.cat([conv_seed, mixed], dim=-1)
        num_positions = mixed.shape[-1]
        windows = [
            conv_input[:, :, i + 1 : i + 1 + state_len] for i in range(num_positions)
        ]
        return torch.stack(windows, dim=1)

    def verify_block(self, hidden_states, seq_ids, input_layernorm):
        """Verify-mode forward over the block, seeded read-only from the committed buffers.

        ``hidden_states`` is pre-input-norm. Returns ``(output [B, S, H], S_stack [B, S, num_v_heads,
        head_k_dim, head_v_dim], conv_cand [B, S, conv_dim, K-1])`` -- the all-reduced layer output and
        the per-token candidate recurrent / conv states.
        """
        batch_size, seq_len, _ = hidden_states.shape

        if self.use_tkg_attention_kernel:
            # Whole layer in one launch: input RMSNorm + in_proj + conv + recurrence + gated norm + o_proj.
            conv_seed = torch.index_select(self.conv_state_buffer, 0, seq_ids)
            init_state = torch.index_select(self.recurrent_state_buffer, 0, seq_ids)
            gamma = input_layernorm.weight.reshape(1, self.hidden_size)
            o_out, S_stack, conv_cand = deltanet_attention_layer_state[2](
                hidden_states,
                self.in_proj_fused.weight,
                gamma,
                self.rms_norm_eps,
                conv_seed[0],
                self.conv1d_weight.weight,
                self.key_dim,
                self._A_log(),
                self._dt_bias(),
                init_state[0],
                self.norm.weight,
                self.out_proj.weight,
                self.rms_norm_eps,
            )
            # o_out is the per-rank output-projection partial; all-reduce across tensor-parallel ranks.
            output = o_out.unsqueeze(0).to(hidden_states.dtype)
            output = mappings.reduce_from_tensor_model_parallel_region(
                output,
                process_group=parallel_state.get_tensor_model_parallel_group(),
            )
            return output, S_stack.unsqueeze(0), conv_cand.unsqueeze(0)

        # PyTorch fallback (reference path): apply input norm, then explicit conv + per-token recurrences.
        hidden_states = input_layernorm(hidden_states)
        qkv, z, a, b = self._project_inputs(hidden_states)
        query = qkv[..., : self.key_dim]
        key = qkv[..., self.key_dim : self.key_dim * 2]
        value = qkv[..., self.key_dim * 2 :]

        # Causal conv over the block, seeded from the committed conv window.
        mixed = torch.cat([query, key, value], dim=-1).transpose(
            1, 2
        )  # [B, conv_dim, S]
        conv_seed = torch.index_select(self.conv_state_buffer, 0, seq_ids)
        conv_input = torch.cat([conv_seed, mixed], dim=-1)
        w = self._conv1d_weight().squeeze(1)  # [conv_dim, K]
        conv_out = torch.zeros_like(mixed)
        for k in range(self.conv_kernel_size):
            conv_out = (
                conv_out
                + w[:, k].unsqueeze(0).unsqueeze(-1) * conv_input[:, :, k : k + seq_len]
            )
        mixed_post_conv = F.silu(conv_out)
        # Per-position conv windows (candidate conv states).
        conv_cand = self.conv_window_candidates(conv_seed, mixed)

        mixed_post_conv = mixed_post_conv.transpose(1, 2)  # [B, S, conv_dim]
        query = mixed_post_conv[..., : self.key_dim]
        key = mixed_post_conv[..., self.key_dim : self.key_dim * 2]
        value = mixed_post_conv[..., self.key_dim * 2 :]

        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self._A_log().float().exp() * F.softplus(a.float() + self._dt_bias())

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = (
                query.unsqueeze(3)
                .expand(-1, -1, -1, rep, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )
            key = (
                key.unsqueeze(3)
                .expand(-1, -1, -1, rep, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )

        query = query.transpose(1, 2).contiguous().float()
        key = key.transpose(1, 2).contiguous().float()
        value = value.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()

        # Seed the recurrence from the committed recurrent state (read-only).
        initial_state = torch.index_select(
            self.recurrent_state_buffer, 0, seq_ids
        ).float()

        out_stack, S_stack = self.verify_block_candidates(
            query, key, value, g, beta, initial_state
        )

        # Output: norm, z-gate, project (mirrors forward()).
        output = out_stack.to(hidden_states.dtype)
        output = output.transpose(1, 2).contiguous()  # [B, S, H, Vd]
        output = output.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = self.norm(output)
        z_gate = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = output * F.silu(z_gate)
        output = output.reshape(batch_size, seq_len, self.value_dim)
        output = self._output_project(output)

        return output, S_stack, conv_cand

    def _nki_recurrent_forward(self, query, key, value, g, beta):
        """Full-sequence recurrent forward using NKI kernel for context encoding."""
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        BH = B * H
        query_flat = query.reshape(BH, S, k_dim).contiguous()
        key_flat = key.reshape(BH, S, k_dim).contiguous()
        value_flat = value.reshape(BH, S, v_dim).contiguous()

        g_flat = g.reshape(BH, S).unsqueeze(-1).expand(-1, -1, v_dim).contiguous()
        beta_flat = beta.reshape(BH, S).unsqueeze(-1).expand(-1, -1, v_dim).contiguous()

        outputs = []
        states = []
        for bh in range(BH):
            out_bh, state_bh = _deltanet_nki_kernel_state(
                query_flat[bh],
                key_flat[bh],
                value_flat[bh],
                g_flat[bh],
                beta_flat[bh],
            )
            outputs.append(out_bh)
            states.append(state_bh)

        output = torch.stack(outputs, dim=0)
        output = output.reshape(B, H, S, v_dim)

        final_state = torch.stack(states, dim=0)
        final_state = final_state.reshape(B, H, k_dim, v_dim)

        return output, final_state

    def _nki_chunked_forward(
        self, query, key, value, g, beta, output_final_state=False, initial_state=None
    ):
        """Chunked NKI kernel forward for context encoding (prefill)."""
        chunk_size = 128

        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        pad_size = (chunk_size - S % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))
        total_seq_len = S + pad_size

        num_chunks = total_seq_len // chunk_size
        g_reshaped = g.reshape(B, H, num_chunks, chunk_size)
        g_cs = g_reshaped.cumsum(dim=-1)
        g_last_per_chunk = g_cs[:, :, :, -1:]
        g_last_expanded = g_last_per_chunk.expand(-1, -1, -1, chunk_size)

        query_chunks = query.reshape(B, H, num_chunks, chunk_size, k_dim)
        key_chunks = key.reshape(B, H, num_chunks, chunk_size, k_dim)
        value_chunks = value.reshape(B, H, num_chunks, chunk_size, v_dim)

        beta_chunks = (
            beta.reshape(B, H, num_chunks, chunk_size)
            .unsqueeze(-1)
            .expand(-1, -1, -1, -1, v_dim)
        )
        gc_chunks = g_cs.unsqueeze(-1).expand(-1, -1, -1, -1, v_dim)
        gl_chunks = g_last_expanded.unsqueeze(-1).expand(-1, -1, -1, -1, v_dim)

        BH = B * H
        query_chunks = query_chunks.reshape(
            BH, num_chunks, chunk_size, k_dim
        ).contiguous()
        key_chunks = key_chunks.reshape(BH, num_chunks, chunk_size, k_dim).contiguous()
        value_chunks = value_chunks.reshape(
            BH, num_chunks, chunk_size, v_dim
        ).contiguous()
        beta_chunks = beta_chunks.reshape(
            BH, num_chunks, chunk_size, v_dim
        ).contiguous()
        gc_chunks = gc_chunks.reshape(BH, num_chunks, chunk_size, v_dim).contiguous()
        gl_chunks = gl_chunks.reshape(BH, num_chunks, chunk_size, v_dim).contiguous()

        device = query.device
        lower_mask = torch.tril(
            torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=device),
            diagonal=-1,
        )
        identity_mat = torch.eye(chunk_size, dtype=torch.float32, device=device)
        lower_mask_diag = torch.tril(
            torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=device),
            diagonal=0,
        )

        initial_state_flat = None
        if initial_state is not None:
            initial_state_flat = (
                initial_state.reshape(BH, k_dim, v_dim).float().contiguous()
            )

        all_outputs = []
        all_states = []
        for bh in range(BH):
            if initial_state_flat is None:
                state = torch.zeros(k_dim, v_dim, dtype=torch.float32, device=device)
            else:
                state = initial_state_flat[bh]

            head_chunks = []
            for c_idx in range(num_chunks):
                q_chunk = query_chunks[bh, c_idx].contiguous()
                k_chunk = key_chunks[bh, c_idx].contiguous()
                v_chunk = value_chunks[bh, c_idx].contiguous()
                beta_chunk = beta_chunks[bh, c_idx].contiguous()
                gc_chunk = gc_chunks[bh, c_idx].contiguous()
                gl_chunk = gl_chunks[bh, c_idx].contiguous()

                out_chunk, state = _deltanet_nki_chunk_step(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    beta_chunk,
                    gc_chunk,
                    gl_chunk,
                    state,
                    lower_mask,
                    identity_mat,
                    lower_mask_diag,
                )
                head_chunks.append(out_chunk)

            head_output = torch.cat(head_chunks, dim=0)
            all_outputs.append(head_output)
            all_states.append(state)

        output = torch.stack(all_outputs, dim=0)
        output = output.reshape(B, H, total_seq_len, v_dim)
        output = output[:, :, :S]

        if output_final_state:
            final_state = torch.stack(all_states, dim=0)
            last_recurrent_state = final_state.reshape(B, H, k_dim, v_dim)
        else:
            last_recurrent_state = None

        return output, last_recurrent_state

    def _fused_chunked_forward(
        self, query, key, value, g, beta, output_final_state=False
    ):
        """Fused single-kernel chunked forward for CTE — SSD-style.

        Processes all chunks in a single NKI kernel call per (B,H) pair.
        State persists in SBUF across chunks (no HBM round-trips).
        Cumsum of g computed in-kernel via tensor_tensor_scan.

        This is the optimized version of _nki_chunked_forward with:
          1. Single kernel call per (B,H) instead of B*H*num_chunks
          2. State in SBUF across all chunks (biggest perf win)
          3. In-kernel cumsum (avoids PyTorch cumsum overhead)
          4. tensor_scalar for broadcasts (no explicit loops)
        """
        chunk_size = 128

        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        # Pad sequence to multiple of chunk_size
        pad_size = (chunk_size - S % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))
        total_seq_len = S + pad_size
        # Pass raw per-token log-decay. The fused NKI kernel forms decay as
        # exp(cumsum(g)_i - cumsum(g)_j), so no pre-kernel clamp is needed.

        BH = B * H
        # Flatten to (BH, S, dim) for per-(b,h) kernel calls
        query_flat = query.reshape(BH, total_seq_len, k_dim).contiguous()
        key_flat = key.reshape(BH, total_seq_len, k_dim).contiguous()
        value_flat = value.reshape(BH, total_seq_len, v_dim).contiguous()

        # g and beta: (BH, S) -> (BH, S, 1) for the kernel's (S, 1) input layout
        g_flat = g.reshape(BH, total_seq_len).unsqueeze(-1).contiguous()
        beta_flat = beta.reshape(BH, total_seq_len).unsqueeze(-1).contiguous()

        # Create constant mask tensors (shared across all B*H calls)
        device = query.device
        lower_mask = torch.tensor(
            _make_lower_mask(), dtype=torch.float32, device=device
        )
        identity_mat = torch.tensor(
            _make_identity(), dtype=torch.float32, device=device
        )
        lower_mask_diag = torch.tensor(
            _make_lower_mask_diag(), dtype=torch.float32, device=device
        )

        all_outputs = []
        all_states = []
        for bh in range(BH):
            out_bh, state_bh = _deltanet_fused_kernel(
                query_flat[bh],  # (S, 128)
                key_flat[bh],  # (S, 128)
                value_flat[bh],  # (S, 128)
                g_flat[bh],  # (S, 1) — RAW g, not cumsum
                beta_flat[bh],  # (S, 1) — sigmoid(b)
                lower_mask,  # (128, 128)
                identity_mat,  # (128, 128)
                lower_mask_diag,  # (128, 128)
            )
            all_outputs.append(out_bh)
            all_states.append(state_bh)

        output = torch.stack(all_outputs, dim=0)
        output = output.reshape(B, H, total_seq_len, v_dim)
        output = output[:, :, :S]

        if output_final_state:
            final_state = torch.stack(all_states, dim=0)
            last_recurrent_state = final_state.reshape(B, H, k_dim, v_dim)
        else:
            last_recurrent_state = None

        return output, last_recurrent_state

    def _sequential_forward(self, query, key, value, g, beta, output_final_state=False):
        """Sequential full-sequence gated delta rule for CTE.

        Uses the same per-step recurrence as _recurrent_step but loops over the
        full sequence.  Avoids the slice-assignment loop in _chunk_forward that
        may compile incorrectly on Neuron/XLA.
        """
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        state = query.new_zeros(B, H, k_dim, v_dim)
        all_outputs = []
        for t in range(S):
            q_t = query[:, :, t]  # (B, H, K)
            k_t = key[:, :, t]  # (B, H, K)
            v_t = value[:, :, t]  # (B, H, V)
            beta_t = beta[:, :, t].unsqueeze(-1)  # (B, H, 1)
            g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

            # Gated delta rule
            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # (B, H, V)
            delta = (v_t - kv_mem) * beta_t  # (B, H, V)
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # (B, H, K, V)

            o_t = (state * q_t.unsqueeze(-1)).sum(dim=-2)  # (B, H, V)
            all_outputs.append(o_t.unsqueeze(2))

        output = torch.cat(all_outputs, dim=2)  # (B, H, S, V)
        final_state = state if output_final_state else None
        return output, final_state

    def _chunk_forward(
        self, query, key, value, g, beta, output_final_state=False, initial_state=None
    ):
        """Chunk-based forward for context encoding (prefill)."""
        chunk_size = 64

        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        pad_size = (chunk_size - S % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))
        total_seq_len = S + pad_size

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        num_chunks = total_seq_len // chunk_size
        query = query.reshape(B, H, num_chunks, chunk_size, k_dim)
        key = key.reshape(B, H, num_chunks, chunk_size, k_dim)
        value = value.reshape(B, H, num_chunks, chunk_size, v_dim)
        k_beta = k_beta.reshape(B, H, num_chunks, chunk_size, k_dim)
        v_beta = v_beta.reshape(B, H, num_chunks, chunk_size, v_dim)
        g = g.reshape(B, H, num_chunks, chunk_size)

        mask = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=0,
        )

        g = g.cumsum(dim=-1)
        decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()

        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        if initial_state is None:
            last_recurrent_state = torch.zeros(
                B, H, k_dim, v_dim, dtype=query.dtype, device=query.device
            )
        else:
            last_recurrent_state = initial_state.to(dtype=query.dtype)
        core_attn_out = torch.zeros_like(value)
        mask2 = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=1,
        )

        for i in range(num_chunks):
            q_i = query[:, :, i]
            k_i = key[:, :, i]
            v_i = value[:, :, i]

            attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(
                mask2, 0
            )

            v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
            v_new = v_i - v_prime

            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn_i @ v_new

            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (
                    k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
                ).transpose(-1, -2)
                @ v_new
            )

        core_attn_out = core_attn_out.reshape(B, H, -1, v_dim)
        core_attn_out = core_attn_out[:, :, :S]

        if not output_final_state:
            last_recurrent_state = None

        return core_attn_out, last_recurrent_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        **kwargs,
    ):
        """Forward pass compatible with NxDI decoder layer interface."""
        batch_size, seq_len, _ = hidden_states.shape

        seq_ids = kwargs.get("seq_ids", None)
        qwen_chunked_prefill_active = (
            self.use_qwen_hybrid_chunked_prefill
            and past_key_value is not None
            and seq_len > 1
        )
        is_decode = past_key_value is not None and not qwen_chunked_prefill_active

        # Padding mask for DeltaNet: [B, S, 1], 1.0 for real tokens and 0.0 for padding.
        # Embeddings are already zeroed for padding; this additionally zeros the decay gate
        # so the recurrent state passes through padding positions unchanged.
        valid_mask_1d = kwargs.get("deltanet_padding_mask", None)  # [B, S, 1] or None
        hybrid_cache_active = self.use_hybrid_cache_manager
        recurrent_state_cache = None
        conv_state_cache = None
        if hybrid_cache_active and past_key_value is not None:
            recurrent_state_cache, conv_state_cache = past_key_value

        # Project inputs
        qkv, z, a, b = self._project_inputs(hidden_states)

        if is_decode and self.use_tkg_attention_kernel:
            # Glue-free decode attention through the conv + DeltaNet TKG NKI kernels.
            return self._forward_decode_tkg(
                hidden_states,
                qkv,
                z,
                a,
                b,
                seq_ids,
                conv_state_cache,
                recurrent_state_cache,
                hybrid_cache_active,
                batch_size,
                seq_len,
            )

        # Split QKV
        query = qkv[..., : self.key_dim]
        key = qkv[..., self.key_dim : self.key_dim * 2]
        value = qkv[..., self.key_dim * 2 :]

        # Causal Conv1d on QKV
        mixed = torch.cat([query, key, value], dim=-1)
        mixed = mixed.transpose(1, 2)

        if is_decode:
            if conv_state_cache is not None:
                conv_state = conv_state_cache[:batch_size]
            elif seq_ids is not None:
                conv_state = torch.index_select(self.conv_state_buffer, 0, seq_ids)
            else:
                conv_state = self.conv_state_buffer[:batch_size]
            conv_input = torch.cat([conv_state, mixed], dim=-1)

            w = self._conv1d_weight().squeeze(1)
            conv_out = torch.zeros_like(mixed)
            for k in range(4):
                conv_out = (
                    conv_out
                    + w[:, k].unsqueeze(0).unsqueeze(-1) * conv_input[:, :, k : k + 1]
                )
            mixed_post_conv = F.silu(conv_out)
            new_conv_state = torch.cat([conv_state[:, :, 1:], mixed], dim=-1)
            alloc_bs = self.conv_state_buffer.shape[0]
            if hybrid_cache_active:
                new_conv_state = new_conv_state.to(self.conv_state_buffer.dtype)
            elif seq_ids is not None:
                # BS=1 optimization: scatter to index 0 of size-1 buffer = direct replacement
                # Add buffer dependency for input_output_alias
                new_conv_state = (
                    new_conv_state.to(self.conv_state_buffer.dtype)
                    + self.conv_state_buffer * 0
                )
            elif batch_size < alloc_bs:
                pad_size = alloc_bs - batch_size
                new_conv_state = torch.cat(
                    [
                        new_conv_state,
                        self.conv_state_buffer[batch_size:] * 0,
                    ],
                    dim=0,
                )
            else:
                new_conv_state = new_conv_state + self.conv_state_buffer * 0
        else:
            if qwen_chunked_prefill_active and conv_state_cache is not None:
                conv_state = conv_state_cache[:batch_size]
                if position_ids is not None:
                    reset_mask = (position_ids[:, :1].long() == 0).to(
                        dtype=conv_state.dtype, device=conv_state.device
                    )
                    conv_state = conv_state * (1.0 - reset_mask[:, None, :])
                conv_input = torch.cat([conv_state, mixed], dim=-1)
                w = self._conv1d_weight().squeeze(1)
                conv_out = torch.zeros_like(mixed)
                for k in range(self.conv_kernel_size):
                    conv_out = (
                        conv_out
                        + w[:, k].unsqueeze(0).unsqueeze(-1)
                        * conv_input[:, :, k : k + seq_len]
                    )
                mixed_post_conv = F.silu(conv_out)
                if valid_mask_1d is not None:
                    state_len = self.conv_kernel_size - 1
                    num_valid = (
                        valid_mask_1d.squeeze(-1).sum(dim=-1, keepdim=True).long()
                    )
                    idx_base = (state_len + num_valid - state_len).clamp(min=0)
                    offsets = torch.arange(state_len, device=mixed.device).unsqueeze(0)
                    gather_idx = idx_base + offsets
                    gather_idx = gather_idx.unsqueeze(1).expand(-1, self.conv_dim, -1)
                    new_conv_state = torch.gather(conv_input, 2, gather_idx)
                else:
                    new_conv_state = conv_input[
                        :, :, -self.conv_kernel_size + 1 :
                    ].contiguous()
            else:
                mixed_post_conv = F.silu(
                    F.conv1d(
                        mixed,
                        self._conv1d_weight(),
                        bias=None,
                        padding=self.conv_kernel_size - 1,
                        groups=self.conv_dim,
                    )[:, :, :seq_len]
                )

                if valid_mask_1d is not None:
                    # valid_mask_1d is [B, S, 1]; count valid tokens per batch
                    num_valid = (
                        valid_mask_1d.squeeze(-1).sum(dim=-1, keepdim=True).long()
                    )  # [B, 1]
                    idx_base = num_valid - 3
                    idx_base = idx_base.clamp(min=0)
                    offsets = torch.arange(3, device=mixed.device).unsqueeze(0)
                    gather_idx = idx_base + offsets  # [B, 3]
                    gather_idx = gather_idx.unsqueeze(1).expand(-1, self.conv_dim, -1)
                    new_conv_state = torch.gather(mixed, 2, gather_idx)
                else:
                    new_conv_state = mixed[:, :, -3:].contiguous()

            alloc_bs = self.conv_state_buffer.shape[0]
            if hybrid_cache_active:
                new_conv_state = new_conv_state.to(self.conv_state_buffer.dtype)
            elif seq_ids is not None:
                # BS=1 optimization: scatter to index 0 = direct replacement
                new_conv_state = (
                    new_conv_state.to(self.conv_state_buffer.dtype)
                    + self.conv_state_buffer * 0
                )
            elif batch_size < alloc_bs:
                pad_size = alloc_bs - batch_size
                new_conv_state = torch.cat(
                    [
                        new_conv_state,
                        torch.zeros(
                            pad_size,
                            self.conv_dim,
                            self.conv_kernel_size - 1,
                            dtype=new_conv_state.dtype,
                            device=new_conv_state.device,
                        ),
                    ],
                    dim=0,
                )
                new_conv_state = new_conv_state + self.conv_state_buffer * 0
            else:
                new_conv_state = new_conv_state + self.conv_state_buffer * 0

        mixed_post_conv = mixed_post_conv.transpose(1, 2)

        # Zero out conv1d output for padding positions.
        # Conv1d with kernel_size=4 leaks real token info into the first
        # few padding positions.  Zeroing here ensures Q, K, V are exactly
        # zero for all padding positions so the recurrence is unaffected.
        if valid_mask_1d is not None:
            mixed_post_conv = (
                mixed_post_conv * valid_mask_1d
            )  # [B, S, conv_dim] * [B, S, 1]

        query = mixed_post_conv[..., : self.key_dim]
        key = mixed_post_conv[..., self.key_dim : self.key_dim * 2]
        value = mixed_post_conv[..., self.key_dim * 2 :]

        # Reshape to heads
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # Compute gating
        beta = b.sigmoid()
        g = -self._A_log().float().exp() * F.softplus(a.float() + self._dt_bias())

        if valid_mask_1d is not None:
            # Zero g for padding → alpha=exp(0)=1 → state preserved through padding
            # Zero beta for padding → no state update from padding tokens
            mask_2d = valid_mask_1d.squeeze(-1).float()  # [B, S]
            g = g * mask_2d.unsqueeze(-1)
            beta = beta * mask_2d.unsqueeze(-1)

        # Expand K heads to match V heads (16 -> 48) using expand+reshape
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads  # 3
            query = (
                query.unsqueeze(3)
                .expand(-1, -1, -1, rep, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )
            key = (
                key.unsqueeze(3)
                .expand(-1, -1, -1, rep, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )

        # Transpose to (B, H, S, dim)
        query = query.transpose(1, 2).contiguous().float()
        key = key.transpose(1, 2).contiguous().float()
        value = value.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()

        if is_decode:
            # TKG: single-step recurrent update
            if recurrent_state_cache is not None:
                recurrent_state = recurrent_state_cache[:batch_size].float()
            elif seq_ids is not None:
                recurrent_state = torch.index_select(
                    self.recurrent_state_buffer, 0, seq_ids
                ).float()
            else:
                recurrent_state = self.recurrent_state_buffer[:batch_size].float()

            output, new_state = self._recurrent_step(
                query, key, value, g, beta, recurrent_state
            )
            new_state_bf16 = new_state.to(self.recurrent_state_buffer.dtype)
            alloc_bs = self.recurrent_state_buffer.shape[0]
            if hybrid_cache_active:
                new_rec_state = new_state_bf16
            elif seq_ids is not None:
                # BS=1 optimization: scatter to index 0 of size-1 buffer = direct replacement
                # Add buffer dependency for input_output_alias
                new_rec_state = new_state_bf16 + self.recurrent_state_buffer * 0
            elif batch_size < alloc_bs:
                new_rec_state = torch.cat(
                    [
                        new_state_bf16,
                        self.recurrent_state_buffer[batch_size:] * 0,
                    ],
                    dim=0,
                )
            else:
                new_rec_state = new_state_bf16 + self.recurrent_state_buffer * 0
        else:
            # CTE: chunked NKI kernel by default. It forms the within-chunk decay as
            # exp(gc[i] - gc[j]), always <= 1 and numerically stable; the fused kernel's split
            # form overflows fp32 under this checkpoint's large decays. Env vars override.
            use_nki_fused = os.environ.get("USE_NKI_FUSED") == "1"
            use_nki = os.environ.get("USE_NKI") == "1"
            use_sequential = os.environ.get("DELTANET_SEQUENTIAL") == "1"
            use_pytorch_chunk = os.environ.get("USE_PYTORCH_CHUNK") == "1"
            # Default-on unless another path is explicitly requested.
            use_nki_chunked = os.environ.get("USE_NKI_CHUNKED", "1") != "0" and not (
                use_nki_fused or use_nki or use_sequential or use_pytorch_chunk
            )

            if qwen_chunked_prefill_active and recurrent_state_cache is not None:
                initial_state = recurrent_state_cache[:batch_size].float()
                if position_ids is not None:
                    reset_mask = (position_ids[:, :1].long() == 0).to(
                        dtype=initial_state.dtype, device=initial_state.device
                    )
                    initial_state = initial_state * (1.0 - reset_mask[:, :, None, None])
                if self.use_qwen_hybrid_chunked_prefill_nki:
                    output, final_state = self._nki_chunked_forward(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        output_final_state=True,
                        initial_state=initial_state,
                    )
                else:
                    output, final_state = self._chunk_forward(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        output_final_state=True,
                        initial_state=initial_state,
                    )
            elif use_pytorch_chunk:
                output, final_state = self._chunk_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            elif use_nki_chunked:
                output, final_state = self._nki_chunked_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            elif use_nki:
                output, final_state = self._nki_recurrent_forward(
                    query, key, value, g, beta
                )
            elif use_sequential:
                output, final_state = self._sequential_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            elif use_nki_fused:
                output, final_state = self._fused_chunked_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            else:
                output, final_state = self._fused_chunked_forward(
                    query, key, value, g, beta, output_final_state=True
                )

            if final_state is not None:
                final_state_bf16 = final_state.to(self.recurrent_state_buffer.dtype)
                alloc_bs = self.recurrent_state_buffer.shape[0]
                if hybrid_cache_active:
                    new_rec_state = final_state_bf16
                elif seq_ids is not None:
                    # BS=1 optimization: scatter to index 0 of size-1 buffer = direct replacement
                    # Add buffer dependency for input_output_alias
                    new_rec_state = final_state_bf16 + self.recurrent_state_buffer * 0
                elif batch_size < alloc_bs:
                    new_rec_state = torch.cat(
                        [
                            final_state_bf16,
                            torch.zeros(
                                alloc_bs - batch_size,
                                self.num_v_heads,
                                self.head_k_dim,
                                self.head_v_dim,
                                dtype=final_state_bf16.dtype,
                                device=final_state_bf16.device,
                            ),
                        ],
                        dim=0,
                    )
                    new_rec_state = new_rec_state + self.recurrent_state_buffer * 0
                else:
                    new_rec_state = final_state_bf16 + self.recurrent_state_buffer * 0
            else:
                new_rec_state = self.recurrent_state_buffer * 1

        # Output: norm, gate, project
        output = output.to(hidden_states.dtype)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = self.norm(output)
        z_gate = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = output * F.silu(z_gate)
        output = output.reshape(batch_size, seq_len, self.value_dim)
        output = self._output_project(output)

        if hybrid_cache_active:
            return (
                output,
                (new_rec_state, new_conv_state),
                new_rec_state,
                new_conv_state,
            )

        # Return dummy KV for KVCacheManager
        dummy_k, dummy_v = self._dummy_kv(
            batch_size, seq_len, hidden_states.dtype, hidden_states.device
        )

        return output, (dummy_k, dummy_v), new_rec_state, new_conv_state


# ============================================================
# InferenceConfig (Dense -- no MoE)
# ============================================================


class Qwen36A3BInferenceConfig(InferenceConfig):
    """Hybrid DeltaNet + GQA decoder with 256-expert MoE and an MTP head."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("num_hidden_layers", 40)

        # Hybrid layer dispatch: [3 DeltaNet + 1 GQA] repeated.
        if "layer_types" not in kwargs and not any(
            hasattr(a, "layer_types") for a in args if hasattr(a, "__dict__")
        ):
            num_layers = kwargs["num_hidden_layers"]
            if num_layers % 4 != 0:
                raise ValueError(
                    f"hybrid layer count must be divisible by 4, got {num_layers}"
                )
            layer_types: list = []
            for _ in range(num_layers // 4):
                layer_types.extend(["linear_attention"] * 3 + ["full_attention"])
            kwargs.setdefault("layer_types", layer_types)

        # DeltaNet
        kwargs.setdefault("linear_num_value_heads", 32)
        kwargs.setdefault("linear_num_key_heads", 16)
        kwargs.setdefault("linear_key_head_dim", 128)
        kwargs.setdefault("linear_value_head_dim", 128)
        kwargs.setdefault("linear_conv_kernel_dim", 4)
        kwargs.setdefault("use_hybrid_cache_manager", False)
        kwargs.setdefault("use_qwen_hybrid_chunked_prefill", False)
        kwargs.setdefault("use_qwen_hybrid_chunked_prefill_nki", False)
        kwargs.setdefault("use_tkg_attention_kernel", False)
        kwargs.setdefault("use_moe_layer_kernel", False)
        kwargs.setdefault("use_verify_megakernel", False)
        kwargs.setdefault("use_draft_megakernel", False)
        kwargs.setdefault("use_speculation_megakernel", False)

        # MoE
        kwargs.setdefault("num_experts", 256)
        kwargs.setdefault("num_experts_per_tok", 8)
        kwargs.setdefault("moe_intermediate_size", 512)
        kwargs.setdefault("shared_expert_intermediate_size", 512)
        kwargs.setdefault("norm_topk_prob", True)
        kwargs.setdefault("router_aux_loss_coef", 0.001)
        # ExpertMLPsV2 reads config.intermediate_size for routed + shared experts.
        # Seed it before super().__init__ so the validator passes.
        kwargs.setdefault("intermediate_size", kwargs["moe_intermediate_size"])

        # MTP
        kwargs.setdefault("mtp_num_hidden_layers", 1)
        kwargs.setdefault("mtp_use_dedicated_embeddings", False)

        super().__init__(*args, **kwargs)

        # Attention output gate
        self.attn_output_gate = getattr(self, "attn_output_gate", True)

        # Partial RoPE
        self.partial_rotary_factor = getattr(self, "partial_rotary_factor", 0.25)
        self.rope_dim = int(self.head_dim * self.partial_rotary_factor)

        # mRoPE (multimodal RoPE) for any VL extension
        rope_params = getattr(self, "rope_parameters", {}) or {}
        self.mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        self.mrope_interleaved = rope_params.get("mrope_interleaved", True)

        # Both routed and shared expert dims must match (single intermediate_size).
        if self.moe_intermediate_size != self.shared_expert_intermediate_size:
            raise ValueError(
                "routed and shared expert intermediate sizes must match; got "
                f"{self.moe_intermediate_size} vs {self.shared_expert_intermediate_size}"
            )
        self.num_local_experts = self.num_experts  # EP=1
        # Shared expert + sigmoid output gate are handled by NeuronMoEBlock
        # directly (NxDI SharedExperts has no per-token gate).
        self.n_shared_experts = 0

        # MoE router: fp32 softmax with normalized affinities.
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "softmax"
        self.neuron_config.disable_numeric_cc_token = True
        self.neuron_config.normalize_top_k_affinities = bool(self.norm_topk_prob)

        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "vocab_size",
            "linear_num_value_heads",
            "linear_num_key_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "layer_types",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "mtp_num_hidden_layers",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


# ============================================================
# Attention (standard GQA for 16 of 64 layers)
# With output gate: q_proj is 2x sized, split into (query, gate)
# With partial RoPE: only first rope_dim dimensions get rotary
# ============================================================


class Qwen36A3BMRoPEEmbedding(nn.Module):
    """Multimodal Rotary Position Embedding (mRoPE).

    Handles 3D position information (temporal, height, width) for VL paths.
    Position IDs have shape (3, batch_size, seq_len) for T/H/W dimensions.
    For text-only (2D position_ids), broadcasts to 3D with identical positions.
    """
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim  # 256
        self.rope_dim = config.rope_dim
        self.mrope_section = config.mrope_section  # [11, 11, 10]
        self.mrope_interleaved = getattr(config, "mrope_interleaved", True)
        self.rope_theta = config.rope_theta

        # Validate mrope_section sums to rope_dim // 2 = 32
        assert sum(self.mrope_section) == self.rope_dim // 2, (
            f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)}, "
            f"expected {self.rope_dim // 2}"
        )

    def forward(self, x, position_ids_3d):
        """Compute cos/sin from 3D position IDs.

        Args:
            x: hidden_states (for device/dtype inference)
            position_ids_3d: (3, batch_size, seq_len) -- T, H, W positions

        Returns:
            cos: (batch_size, seq_len, rope_dim)
            sin: (batch_size, seq_len, rope_dim)
        """
        device = x.device
        dtype = torch.float32

        if position_ids_3d.ndim == 2:
            position_ids_3d = position_ids_3d[None, ...].expand(
                3, position_ids_3d.shape[0], -1
            )

        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.rope_dim, 2, dtype=dtype, device=device)
                / self.rope_dim
            )
        )
        inv_freq = inv_freq[None, None, :, None].expand(
            3, position_ids_3d.shape[1], -1, 1
        )
        positions = position_ids_3d[:, :, None, :].float()
        freqs = (inv_freq.float() @ positions).transpose(2, 3)

        # Match HF Qwen3.6 mRoPE layout exactly: start from the temporal
        # frequencies, then splice H/W frequencies into interleaved positions.
        freqs_t = freqs[0]
        if self.mrope_interleaved:
            for dim, offset in enumerate((1, 2), start=1):
                length = self.mrope_section[dim] * 3
                idx = slice(offset, length, 3)
                freqs_t[..., idx] = freqs[dim, ..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)

        return cos, sin


class NeuronQwen36A3BAttention(NeuronAttentionBase):
    """Standard GQA attention with output gate and partial RoPE.

    16 Q heads, 2 KV heads (8:1 GQA), head_dim=256.
    q_proj is doubled (query + gate), split at load time.
    Only the first rope_dim=64 of head_dim=256 gets rotary encoding.

    Uses NeuronAttentionBase infrastructure for QKV projection, KV cache,
    RoPE, and attention computation. Overrides forward() to insert the
    sigmoid output gate between attention output and o_proj.
    """
    def __init__(self, config):
        # Partial RoPE: create mRoPE embedding with rope_dim (64)
        self.rope_dim = config.rope_dim

        # Create QK norm modules (will be passed to base class)
        rms_norm_eps = config.rms_norm_eps
        q_ln = get_rmsnorm_cls(config)(config.head_dim, rms_norm_eps)
        k_ln = get_rmsnorm_cls(config)(config.head_dim, rms_norm_eps)

        # Partial RoPE: use standard RotaryEmbedding.
        # For VL with 3D mRoPE positions, cos/sin are pre-computed externally in
        # get_model_output() using Qwen36A3BMRoPEEmbedding and passed as cos_cache/sin_cache.
        rotary_emb = RotaryEmbedding(
            self.rope_dim,  # Only 64 dims get rotary embedding
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=rms_norm_eps,
            use_qk_norm=False,
            q_layernorm=q_ln,
            k_layernorm=k_ln,
        )

        # Separate mRoPE module for VL 3D position_ids
        self.mrope_emb = Qwen36A3BMRoPEEmbedding(config)

        # Output gate projection: hidden_size -> num_heads * head_dim
        # Populated from the second half of q_proj during state dict conversion.
        self.output_gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
            gather_output=False,
        )

        # Fused GQA TKG kernel weights (decode/verify only): contraction-first, head-major,
        # per-rank, mirroring the DeltaNet in_proj_fused / out_proj layout. Prefill keeps
        # the standard q/k/v/output_gate/o_proj modules above.
        self.use_tkg_attention_kernel = getattr(
            config, "use_tkg_attention_kernel", False
        )
        if self.use_tkg_attention_kernel:
            replicated_kv = max(
                config.num_key_value_heads, config.neuron_config.tp_degree
            )
            global_qkv_dim = (
                config.num_attention_heads + 2 * replicated_kv
            ) * config.head_dim
            global_gate_dim = config.num_attention_heads * config.head_dim
            global_value_dim = config.num_attention_heads * config.head_dim
            # [hidden, I] head-major [q..|k|v]; RowParallelLinear shards I (dim 1).
            self.gqa_qkv_w = RowParallelLinear(
                global_qkv_dim,
                config.hidden_size,
                bias=False,
                input_is_parallel=True,
            )
            # [hidden, G] head-major output-gate weight; shards G.
            self.gqa_gate_w = RowParallelLinear(
                global_gate_dim,
                config.hidden_size,
                bias=False,
                input_is_parallel=True,
            )
            # o_proj stored transposed [value_dim, hidden] (shards value_dim).
            self.gqa_o_proj_w = ColumnParallelLinear(
                config.hidden_size,
                global_value_dim,
                bias=False,
                gather_output=False,
            )

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        """Partial RoPE: only apply rotary embedding to first rope_dim dimensions.

        Q shape: (B, H, S, head_dim) where head_dim=256
        cos/sin shape: (B, S, rope_dim) where rope_dim=64 (from RotaryEmbedding(dim=64))

        Split Q/K along last dim into:
          q_rope (first 64 dims) -- apply RoPE
          q_pass (remaining 192 dims) -- pass through unchanged
        """
        from neuronx_distributed_inference.modules.attention.utils import (
            apply_rotary_pos_emb,
        )

        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

        # Split into rope and pass-through portions
        Q_orig_dtype = Q.dtype
        q_rope = Q[..., : self.rope_dim]  # (B, H, S, 64)
        q_pass = Q[..., self.rope_dim :]  # (B, H, S, 192)
        k_rope = K[..., : self.rope_dim]
        k_pass = K[..., self.rope_dim :]

        # Apply RoPE only to the rope portion
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos_cache, sin_cache)

        # Concatenate back (ensure bf16 is maintained)
        Q = torch.cat([q_rope, q_pass], dim=-1).to(Q_orig_dtype)
        K = torch.cat([k_rope, k_pass], dim=-1).to(Q_orig_dtype)

        return Q, K, cos_cache, sin_cache

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask=None):
        """Prefill path with NKI flash attention for head_dim=256."""
        head_dim = Q.shape[-1]

        # Fork-loaded nkilib flash attention (head_dim > 128).
        if _nkilib_flash_attn is not None:
            q_contig = Q.contiguous()
            k_contig = K.contiguous()
            v_contig = V.contiguous()
            scale = 1.0 / math.sqrt(head_dim)
            result = _nkilib_flash_attn(
                q_contig, k_contig, v_contig, scale=scale, use_causal_mask=True
            )
            return result, None

        # Globally-patched nkilib kernel.
        if NKILIB_PATCH_ACTIVE:
            return _flash_fwd_call(Q, K, V, use_causal_mask=True), None

        # Fallback: softmax path (use 3D tensors to avoid compiler ICE with 4D patterns)
        if head_dim > 128:
            # GQA: expand K/V heads to match Q heads
            num_q_heads = Q.shape[1]
            num_kv_heads = K.shape[1]
            if num_q_heads != num_kv_heads:
                kv_rep = num_q_heads // num_kv_heads
                K = (
                    K.unsqueeze(2)
                    .expand(-1, -1, kv_rep, -1, -1)
                    .reshape(bsz, num_q_heads, q_len, head_dim)
                )
                V = (
                    V.unsqueeze(2)
                    .expand(-1, -1, kv_rep, -1, -1)
                    .reshape(bsz, num_q_heads, q_len, head_dim)
                )
            # Reshape to 3D (B*H, S, d) to avoid neuronx-cc codegen ICE with 4D
            # attention weight tensors (NCC_INLA001: Expected 2D tensor but got 4D AP)
            Q_3d = Q.reshape(bsz * num_q_heads, q_len, head_dim)
            K_3d = K.reshape(bsz * num_q_heads, q_len, head_dim)
            V_3d = V.reshape(bsz * num_q_heads, q_len, head_dim)
            attn_weights = torch.bmm(Q_3d, K_3d.transpose(-1, -2)) / math.sqrt(head_dim)
            # Build causal mask for 3D: (1, S, S) broadcast over B*H
            causal_mask = torch.triu(
                torch.full(
                    (q_len, q_len),
                    -65504.0,
                    dtype=attn_weights.dtype,
                    device=attn_weights.device,
                ),
                diagonal=1,
            ).unsqueeze(0)
            attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                Q.dtype
            )
            attn_output = torch.bmm(attn_weights, V_3d)
            # Reshape back to 4D (B, H, S, d)
            return attn_output.reshape(bsz, num_q_heads, q_len, head_dim), None

        return _flash_fwd_call(Q, K, V, use_causal_mask=True), None

    def perform_qwen_chunked_prefill(self, Q, K, V, past_key_value, position_ids):
        """Exact chunked CTE over the full decode cache.

        The current chunk K/V tensors are scattered into the full cache at
        absolute position_ids, then attention for this chunk is computed over
        all cache positions up to the chunk end. This keeps full-attention
        layers correct when model-local chunked prefill feeds context in
        multiple CTE-bucket calls.
        """
        k_cache, v_cache = past_key_value
        B, q_heads, q_len, head_dim = Q.shape
        kv_heads = K.shape[1]
        cache_len = k_cache.shape[2]

        pos = position_ids.long()
        k_index = pos[:, None, :, None].expand(B, kv_heads, q_len, head_dim)
        k_cache = torch.scatter(k_cache, dim=2, index=k_index, src=K.to(k_cache.dtype))
        v_cache = torch.scatter(v_cache, dim=2, index=k_index, src=V.to(v_cache.dtype))

        if q_heads != kv_heads:
            kv_rep = q_heads // kv_heads
            K_full = (
                k_cache.unsqueeze(2)
                .expand(-1, -1, kv_rep, -1, -1)
                .reshape(B, q_heads, cache_len, head_dim)
            )
            V_full = (
                v_cache.unsqueeze(2)
                .expand(-1, -1, kv_rep, -1, -1)
                .reshape(B, q_heads, cache_len, head_dim)
            )
        else:
            K_full = k_cache
            V_full = v_cache

        attn_weights = torch.matmul(Q, K_full.transpose(-1, -2)) / math.sqrt(head_dim)
        cache_positions = torch.arange(cache_len, device=position_ids.device).view(
            1, 1, 1, -1
        )
        causal_mask = cache_positions <= pos[:, None, :, None]
        attn_weights = attn_weights.masked_fill(~causal_mask, -65504.0)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
        return torch.matmul(attn_weights, V_full)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        adapter_ids=None,
        active_mask=None,
        **kwargs,
    ):
        """Forward with output gate applied BEFORE o_proj.

        Override NeuronAttentionBase.forward() to insert the sigmoid gate
        between the attention output and o_proj, matching the HF reference:
          gate = sigmoid(gate_proj(pre_attn_hidden))
          attn_output = attn_output * gate
          attn_output = o_proj(attn_output)
        """
        bsz, q_len, _ = hidden_states.shape

        if self.uses_tkg_kernel(past_key_value, q_len):
            return self._forward_tkg_kernel(
                hidden_states,
                attention_mask,
                active_mask,
                position_ids,
                past_key_value,
                cos_cache,
                sin_cache,
                kv_write_idx=kwargs.get("kv_write_idx"),
                gamma_in=kwargs.get("gamma_in"),
            )

        # Use standard 2D position_ids for prep_qkv_tensors.
        rope_pos_ids = position_ids

        # Compute gate from input hidden states (before QKV projection)
        gate = self.output_gate_proj(hidden_states)  # (B, S, num_heads * head_dim)

        # Standard QKV prep (projections, QK norm, RoPE)
        Q, K, V, cos_cache, sin_cache, _residual = self.prep_qkv_tensors(
            rope_pos_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
        )

        qwen_chunked_prefill_active = (
            past_key_value is not None
            and q_len > 1
            and getattr(self.config, "use_qwen_hybrid_chunked_prefill", False)
        )

        if past_key_value is None:
            # Context encoding (prefill)
            attn_output, _flash_strategy = self.perform_prefill(
                Q, K, V, q_len, bsz, attention_mask
            )
        elif qwen_chunked_prefill_active:
            attn_output = self.perform_qwen_chunked_prefill(
                Q, K, V, past_key_value, position_ids
            )
        else:
            # Token generation (decode)
            tkg_mask = attention_mask
            if tkg_mask is not None and tkg_mask.ndim == 2:
                tkg_mask = tkg_mask.unsqueeze(1).unsqueeze(2)  # (B, S) -> (B, 1, 1, S)
            attn_output = self.compute_for_token_gen(
                Q, K, V, position_ids, past_key_value, tkg_mask, active_mask
            )

        # attn_output is (B, H, S, head_dim) -- transpose to (B, S, H*head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Apply sigmoid output gate BEFORE o_proj (matching HF reference)
        attn_output = attn_output * torch.sigmoid(gate)

        # Apply o_proj
        attn_output = self.get_o_proj()(attn_output, adapter_ids=adapter_ids)

        # Debug knob: zero the attention contribution (K/V/cos/sin stay valid).
        if os.environ.get("A3B_BYPASS_ATTN") == "1":
            attn_output = torch.zeros_like(attn_output)

        # Ensure K, V are in model dtype (bf16) for KV cache update
        # (prevents mixed-precision dynamic-update-slice in neuronx-cc)
        K = K.to(self.torch_dtype)
        V = V.to(self.torch_dtype)
        # Transposed K cache expects BHDS (the scatter dim is head_dim-major).
        if self.k_cache_transposed:
            K = K.permute(0, 1, 3, 2)
        past_key_value = (K, V)
        return attn_output, past_key_value, cos_cache, sin_cache

    def _build_kernel_mask(self, attention_mask, active_mask, bsz, T, L, device):
        """Build the fused kernel mask [L, B, q_heads, T] uint8 (1=keep), s_prior-major.

        Prior window = first L-T cache slots (gated by the NxDI prior mask); the active tokens
        occupy the last T slots (gated by intra-block causality, or all-keep for T=1 decode).
        """
        am = attention_mask
        if am.dim() == 2:  # [B, L]
            am = am[:, None, :].expand(bsz, T, L)
        else:  # [B, 1, *, L]
            am = am[:, 0]
            if am.shape[1] == 1:
                am = am.expand(bsz, T, L)
        prior = am.to(torch.bool)[:, :, : L - T]  # [B, T(query), L-T]
        if active_mask is not None:
            active = active_mask[:, 0].to(torch.bool)  # [B, T(query), T(active key)]
        else:
            active = torch.ones(bsz, T, T, dtype=torch.bool, device=device)
        full = torch.cat([prior, active], dim=2)  # [B, T(query), L]
        mask = full.permute(2, 0, 1)[:, :, None, :].expand(L, bsz, self.num_heads, T)
        return mask.to(torch.uint8).contiguous()

    def uses_tkg_kernel(self, past_key_value, q_len):
        """Whether this call takes the fused GQA kernel: cache present, not chunked prefill.

        The decoder layer asks too, so it knows to leave the input RMSNorm to the kernel.
        """
        return (
            self.use_tkg_attention_kernel
            and past_key_value is not None
            and not (
                q_len > 1
                and getattr(self.config, "use_qwen_hybrid_chunked_prefill", False)
            )
        )

    def _forward_tkg_kernel(
        self,
        hidden_states,
        attention_mask,
        active_mask,
        position_ids,
        past_key_value,
        cos_cache,
        sin_cache,
        kv_write_idx=None,
        gamma_in=None,
    ):
        """Fused GQA decode/verify: hidden in, all-reduced [B,S,H] out.

        hidden_states is raw when gamma_in [1,H] is given -- the input RMSNorm then runs in-kernel,
        SBUF-resident -- and pre-normed otherwise. Returns (output, kv, cos_cache, sin_cache), the
        same tuple shape the standard forward returns.

        kv_write_idx [B,1] int32 scatters the active K/V into the caches in place; the mutated cache
        handles then take the KV slot, so the caller skips update_cache.
        """
        bsz, T, _ = hidden_states.shape
        k_cache, v_cache = past_key_value  # [B,1,D,L] BHDS, [B,1,L,D] BHSD
        L = k_cache.shape[-1]
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.mrope_emb(hidden_states, position_ids)
        mask = self._build_kernel_mask(
            attention_mask, active_mask, bsz, T, L, hidden_states.device
        )
        rets = gqa_fused_tkg_fwd[2](
            hidden_states,
            self.gqa_qkv_w.weight,
            self.gqa_gate_w.weight,
            self.q_layernorm.weight,
            self.k_layernorm.weight,
            cos_cache[0],
            sin_cache[0],
            k_cache,
            v_cache,
            mask,
            self.gqa_o_proj_w.weight,
            self.rms_norm_eps,
            kv_write_idx=kv_write_idx,
            gamma_in=gamma_in,
        )
        # In-place writes append the mutated cache handles; those are then the updated KV.
        kv = tuple(rets[3:5]) if kv_write_idx is not None else tuple(rets[1:3])
        # rets[0] is the per-rank o_proj partial; all-reduce across tensor-parallel ranks.
        output = rets[0].unsqueeze(0).to(hidden_states.dtype)
        output = mappings.reduce_from_tensor_model_parallel_region(
            output,
            process_group=parallel_state.get_tensor_model_parallel_group(),
        )
        return output, kv, cos_cache, sin_cache


# ============================================================
# MoE FFN block (every layer)
# ============================================================


@contextlib.contextmanager
def _defer_moe_allreduce():
    """Make the MoE's trailing all-reduce a no-op so its routed local partial can
    be fused with the gated shared-expert partial and reduced once (3 -> 2
    collectives/layer). Targets the only reduce both MoE paths share, so it holds
    regardless of `init_tkg_module`; `RowParallelLinear` reduces via a separate
    import and is unaffected."""
    orig = mappings.reduce_from_tensor_model_parallel_region
    mappings.reduce_from_tensor_model_parallel_region = lambda input_, *a, **k: input_
    try:
        yield
    finally:
        mappings.reduce_from_tensor_model_parallel_region = orig


class NeuronMoEBlock(nn.Module):
    """Routed top-k experts + sigmoid-gated shared expert.

    NxDI's `SharedExperts` has no per-token output gate, so the shared
    expert lives in this module directly rather than inside the MoE call.
    Layout:
        moe                      : NxDI MoE module, routed-only (n_shared=0)
        shared_expert.gate_proj  : (H -> shared_I)  column-parallel
        shared_expert.up_proj    : (H -> shared_I)  column-parallel
        shared_expert.down_proj  : (shared_I -> H)  row-parallel, no reduce
        shared_expert_gate       : (H -> 1)         column-parallel
    Forward: routed and shared down-projections return local partials; their
    gated sum is reduced once. Valid because the gate is rank-replicated:
    AR(routed) + gate*AR(shared) == AR(routed + gate*shared).
    """
    def __init__(self, config: "Qwen36A3BInferenceConfig"):
        super().__init__()
        self.config = config
        self.use_moe_layer_kernel = getattr(config, "use_moe_layer_kernel", False)
        self.moe = initialize_moe_module(config=config)

        # Always-on shared expert, owned here so we can apply the sigmoid gate.
        si = config.shared_expert_intermediate_size
        h = config.hidden_size
        self.shared_expert = SharedExpertMLP(
            h, si, transposed=self.use_moe_layer_kernel
        )
        # Scalar gate per token. Output is 1, so a parallel linear can't
        # shard along the output dim -- replicate across ranks instead.
        self.shared_expert_gate = nn.Linear(h, 1, bias=False)

        # Fused MoE TKG kernel (verify path) reads router/sigma contraction-first. Router lives inside
        # NxDI's self.moe (can't restow), so keep small rank-replicated transposed copies; the shared
        # expert is stowed transposed in-place (SharedExpertMLP above), so it needs no copy.
        if self.use_moe_layer_kernel:
            self.moe_router_w_t = nn.Parameter(
                torch.empty(h, config.num_experts), requires_grad=False
            )  # [H, E]
            self.moe_sigma_gate_t = nn.Parameter(
                torch.empty(h, 1), requires_grad=False
            )  # [H, 1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Debug knob: zero the MoE FFN contribution.
        if os.environ.get("A3B_BYPASS_MOE") == "1":
            return torch.zeros_like(hidden_states)

        # Fused MoE TKG kernel path (verify): raw hidden + gamma, norm + routed + shared + gate in-kernel.
        if gamma is not None:
            return self._forward_moe_kernel(hidden_states, gamma)

        is_spec = (
            self.config.neuron_config.enable_fused_speculation
            and not self.config.neuron_config.is_prefill_stage
        )
        # Routed and shared down-projections both return un-reduced local
        # partials (MoE reduce deferred below; shared_expert.down_proj has
        # reduce_output=False); their gated sum is all-reduced once.
        with _defer_moe_allreduce():
            routed_local = self.moe(
                hidden_states,
                padding_mask,
                is_speculative_decoding=is_spec,
            )[0]
        shared_local = self.shared_expert(hidden_states)
        gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        combined_local = routed_local + gate * shared_local
        return mappings.reduce_from_tensor_model_parallel_region(
            combined_local,
            process_group=parallel_state.get_tensor_model_parallel_group(),
        )

    def _forward_moe_kernel(self, hidden_states, gamma):
        """Fused MoE layer (verify, T>1): one LNC2 launch -> per-rank partial -> single TP all-reduce.

        Weights are read in kernel-native contraction-first layout with no runtime transpose: experts
        straight from self.moe (gate_up a free [E,H,2,I] reshape), shared stowed transposed, router/sigma
        from the rank-replicated transposed copies. ``gamma`` [1,H] is post_attention_layernorm.weight.
        """
        _, _, hidden = hidden_states.shape
        tp = self.config.neuron_config.tp_degree
        i_rank = self.config.moe_intermediate_size // tp  # per-rank routed intermediate
        gate_up_w = self.moe.expert_mlps.mlp_op.gate_up_proj.weight  # [E, H, 2*I]
        down_w = self.moe.expert_mlps.mlp_op.down_proj.weight  # [E, I, H]
        partial = moe_layer_fwd[2](
            hidden_states,
            gamma,
            self.moe_router_w_t,  # [H, E]
            gate_up_w.reshape(
                self.config.num_experts, hidden, 2, i_rank
            ),  # [E, H, 2, I]
            down_w,  # [E, I, H]
            self.moe_sigma_gate_t,  # [H, 1]
            self.shared_expert.gate_proj.weight,  # [H, I_s]
            self.shared_expert.up_proj.weight,  # [H, I_s]
            self.shared_expert.down_proj.weight,  # [I_s, H]
            self.config.rms_norm_eps,
            self.config.num_experts_per_tok,
        )
        return mappings.reduce_from_tensor_model_parallel_region(
            partial,
            process_group=parallel_state.get_tensor_model_parallel_group(),
        )


class SharedExpertMLP(nn.Module):
    """Always-on shared expert: SwiGLU FFN sized by `shared_expert_intermediate_size`.

    ``transposed`` stores weights contraction-first (gate/up ``[H, I_s]``, down ``[I_s, H]``) so the
    fused MoE TKG kernel reads them with no runtime transpose; the PyTorch forward stays transpose-free
    via direct matmuls (mirrors the DeltaNet in_proj_fused/out_proj layout). Both paths return the
    local partial (no reduce) so NeuronMoEBlock fuses it with the routed partial under one all-reduce.
    """
    def __init__(
        self, hidden_size: int, intermediate_size: int, transposed: bool = False
    ):
        super().__init__()
        self.transposed = transposed
        if transposed:
            # gate/up: [H, I_s] (shard I_s on dim 1); down: [I_s, H] (shard I_s on dim 0), matmul'd raw.
            self.gate_proj = TransposedColumnParallelLinear(
                hidden_size, intermediate_size, bias=False, gather_output=False
            )
            self.up_proj = TransposedColumnParallelLinear(
                hidden_size, intermediate_size, bias=False, gather_output=False
            )
            self.down_proj = ColumnParallelLinear(
                hidden_size, intermediate_size, bias=False, gather_output=False
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                hidden_size, intermediate_size, bias=False, gather_output=False
            )
            self.up_proj = ColumnParallelLinear(
                hidden_size, intermediate_size, bias=False, gather_output=False
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                input_is_parallel=True,
                reduce_output=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transposed:
            act = F.silu(self.gate_proj(x)) * self.up_proj(x)  # [., I_s] per-rank
            return torch.matmul(
                act, self.down_proj.weight
            )  # [., I_s] @ [I_s, H] -> [., H] partial
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# MTP draft head (self-speculative decoding)
# ============================================================


class NeuronMTPHead(nn.Module):
    """One extra decoder layer that predicts token t+2 from (hidden_t, embed_{t+1}).

    Layout (matches the HF checkpoint):
        embed_norm    : RMSNorm over embed(t+1)
        hidden_norm   : RMSNorm over last main hidden at t
        eh_proj       : Linear(2H, H)
        decoder_layer : one Qwen36A3BDecoderLayer
        final_norm    : RMSNorm
        mtp_lm_head   : ColumnParallelLinear(H, vocab)

    Invoked off the traced main forward by the speculative-decoding driver.
    """
    def __init__(
        self,
        config: "Qwen36A3BInferenceConfig",
        layer_idx: Optional[int] = None,
        layer_type: str = "full_attention",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = (
            layer_idx if layer_idx is not None else config.num_hidden_layers
        )
        self.layer_type = layer_type

        rms_cls = get_rmsnorm_cls(config)
        h = config.hidden_size
        eps = config.rms_norm_eps

        self.embed_norm = rms_cls(h, eps=eps)
        self.hidden_norm = rms_cls(h, eps=eps)
        # Stored transposed [2H, H_rank] (contraction first) for the draft megakernel;
        # the transposed matmul is numerically identical on the XLA path.
        self.eh_proj = TransposedColumnParallelLinear(
            2 * h,
            h,
            bias=False,
            gather_output=True,
        )

        # NeuronQwen36A3BDecoderLayer reads config.layer_types[layer_idx] to select
        # DeltaNet vs GQA. The MTP slot sits past the main 40 layers, so build a
        # shallow copy of config with layer_types extended to include this slot.
        layer_config = config.__class__.__new__(config.__class__)
        layer_config.__dict__.update(config.__dict__)
        layer_types = list(config.layer_types)

        if len(layer_types) <= self.layer_idx:
            layer_types += [layer_type] * (self.layer_idx + 1 - len(layer_types))

        layer_types[self.layer_idx] = layer_type
        layer_config.layer_types = layer_types

        self.decoder_layer = NeuronQwen36A3BDecoderLayer(layer_config, self.layer_idx)

        self.final_norm = rms_cls(h, eps=eps)
        # Sharded: the draft token is a sharded vocab-parallel argmax, so the
        # full-vocab logits are never gathered.
        self.mtp_lm_head = TransposedColumnParallelLinear(
            h,
            config.vocab_size,
            bias=False,
            gather_output=False,
        )

    def draft_step(
        self,
        prev_hidden: torch.Tensor,
        next_input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        **layer_kwargs,
    ):
        """Draft the t+2 logits, draft-layer KV, and pre-final-norm hidden.

        Returns ``(draft_logits, present_key_value, hidden)``. ``hidden`` is the
        decoder layer's pre-final-norm output (the trunk hidden the next draft step
        consumes). ``layer_kwargs`` (active_mask, is_for_context_encoding, seq_ids,
        ...) pass through to the GQA decoder layer.
        """
        # Concat order is [embed | hidden], matching the checkpoint's mtp.fc transposed at
        # load with no column repacking. Each norm is applied to its own tensor before the join.
        combined = torch.cat(
            [self.embed_norm(next_input_embeds), self.hidden_norm(prev_hidden)],
            dim=-1,
        )
        h_in = self.eh_proj(combined)
        out = self.decoder_layer(
            h_in,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            **layer_kwargs,
        )
        hidden = out[0]
        draft_logits = self.mtp_lm_head(self.final_norm(hidden))
        return draft_logits, out[1], hidden


# ============================================================
# Decoder Layer (hybrid dispatch -- DeltaNet or GQA, then MoE)
# ============================================================


class NeuronQwen36A3BDecoderLayer(nn.Module):
    """Hybrid decoder layer: dispatches to DeltaNet or standard attention, then MoE FFN."""
    def __init__(self, config: Qwen36A3BInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.layer_idx = layer_idx
        self.config = config

        if self.layer_type == "linear_attention":
            self.linear_attn = NeuronGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = NeuronQwen36A3BAttention(config=config)

        self.mlp = NeuronMoEBlock(config)

        self.input_layernorm = get_rmsnorm_cls(config)(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls(config)(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        padding_mask=None,
        cos_cache=None,
        sin_cache=None,
        verify_mode=False,
        **kwargs,
    ):
        residual = hidden_states

        # Layer-boundary markers are XLA custom calls; skip them on the host so
        # the layer runs under cpu_mode (device tracing is unaffected).
        if not cpu_mode():
            hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        gqa_kernel = self.layer_type != "linear_attention" and (
            self.self_attn.uses_tkg_kernel(past_key_value, hidden_states.shape[1])
        )
        # The GQA and verify linear-attention kernels fuse the input RMSNorm; norm the rest here.
        if not (gqa_kernel or (verify_mode and self.layer_type == "linear_attention")):
            hidden_states = self.input_layernorm(hidden_states)

        if verify_mode and self.layer_type == "linear_attention":
            # Verify: run the 2-token block as an exact single-step recurrence, seeded read-only
            # from the live committed buffers. The per-position S-candidates and conv windows
            # come back for the in-graph accept/reject commit.
            seq_ids = kwargs.get("seq_ids", None)
            attn_out, S_stack, conv_cand = self.linear_attn.verify_block(
                hidden_states, seq_ids, self.input_layernorm
            )
            hidden_states = residual + attn_out
            # Dummy KV so the verify target's KVCacheManager keeps a uniform
            # per-layer KV list (DeltaNet layers carry no GQA cache).
            b, s, _ = attn_out.shape
            dummy_k, dummy_v = self.linear_attn._dummy_kv(
                b, s, attn_out.dtype, attn_out.device
            )
            present_key_value = (dummy_k, dummy_v)
            deltanet_states = (S_stack, conv_cand)
        elif self.layer_type == "linear_attention":
            # DeltaNet path
            attn_out, dummy_kv, new_rec_state, new_conv_state = self.linear_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                **kwargs,
            )
            hidden_states = residual + attn_out
            present_key_value = dummy_kv
            deltanet_states = (
                None
                if getattr(self.config, "use_hybrid_cache_manager", False)
                else (new_rec_state, new_conv_state)
            )
        else:
            deltanet_states = None
            # Standard attention path
            hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                gamma_in=(
                    self.input_layernorm.weight.reshape(1, self.hidden_size)
                    if gqa_kernel
                    else None
                ),
                **kwargs,
            )
            hidden_states = residual + hidden_states

        # Dense MLP FFN
        residual = hidden_states
        # Fused MoE kernel is decode-only: is_prefill_stage is False only on TKG graphs.
        if (
            self.mlp.use_moe_layer_kernel
            and self.config.neuron_config.is_prefill_stage is False
        ):
            # Fused MoE kernel applies post_attention_layernorm in-kernel (norm-once, SBUF-resident);
            # pass raw hidden + gamma so residual stays the pre-norm hidden.
            hidden_states = self.mlp(
                hidden_states,
                gamma=self.post_attention_layernorm.weight.reshape(1, self.hidden_size),
            )
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if not cpu_mode():
            hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (
            hidden_states,
            present_key_value,
            cos_cache,
            sin_cache,
            None,
            deltanet_states,
        )
        return outputs


# ============================================================
# Hybrid Cache Manager (opt-in)
# ============================================================


class HybridDeltaNetCacheManager(KVCacheManager):
    """Layer-type-aware cache manager: DeltaNet (recurrent+conv state) or GQA (K/V)."""
    def __init__(self, config: Qwen36A3BInferenceConfig, num_kv_head, **kwargs):
        self.layer_types = list(config.layer_types)
        self._validate_hybrid_config(config)
        super().__init__(config, num_kv_head=num_kv_head, **kwargs)

        dtype = (
            config.neuron_config.attention_dtype
            if config.neuron_config.attention_dtype is not None
            else config.neuron_config.torch_dtype
        )
        cache_dtype = getattr(self, "cache_dtype", dtype)
        max_batch_size = (
            config.neuron_config.kv_cache_batch_size
            + config.neuron_config.kv_cache_padding_size
        )
        tp_degree = config.neuron_config.tp_degree
        if config.linear_num_value_heads % tp_degree != 0:
            raise ValueError(
                f"linear_num_value_heads={config.linear_num_value_heads} must be divisible "
                f"by tp_degree={tp_degree}"
            )
        if config.linear_num_key_heads % tp_degree != 0:
            raise ValueError(
                f"linear_num_key_heads={config.linear_num_key_heads} must be divisible "
                f"by tp_degree={tp_degree}"
            )
        local_num_value_heads = config.linear_num_value_heads // tp_degree
        local_num_key_heads = config.linear_num_key_heads // tp_degree
        recurrent_shape = [
            max_batch_size,
            local_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        ]
        conv_dim = (
            2 * local_num_key_heads * config.linear_key_head_dim
            + local_num_value_heads * config.linear_value_head_dim
        )
        conv_shape = [
            max_batch_size,
            conv_dim,
            config.linear_conv_kernel_dim - 1,
        ]

        params = []
        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "linear_attention":
                params.append(
                    nn.Parameter(
                        torch.zeros(recurrent_shape, dtype=dtype), requires_grad=False
                    )
                )
                params.append(
                    nn.Parameter(
                        torch.zeros(conv_shape, dtype=dtype), requires_grad=False
                    )
                )
            else:
                k_shape = (
                    self.k_shapes[layer_idx]
                    if hasattr(self, "k_shapes")
                    else self.k_shape
                )
                v_shape = (
                    self.v_shapes[layer_idx]
                    if hasattr(self, "v_shapes")
                    else self.v_shape
                )
                params.append(
                    nn.Parameter(
                        torch.zeros(k_shape, dtype=cache_dtype), requires_grad=False
                    )
                )
                params.append(
                    nn.Parameter(
                        torch.zeros(v_shape, dtype=cache_dtype), requires_grad=False
                    )
                )

        self.past_key_values = nn.ParameterList(params)

    @staticmethod
    def _validate_hybrid_config(config: Qwen36A3BInferenceConfig):
        nc = config.neuron_config
        unsupported = []
        if nc.is_block_kv_layout:
            unsupported.append("block KV layout")
        if getattr(nc, "kv_quant_config", None) is not None or getattr(
            nc, "kv_cache_quant", False
        ):
            unsupported.append("KV cache quantization")
        if nc.enable_fused_speculation or nc.speculation_length > 0 or nc.is_medusa:
            unsupported.append("speculative decoding")
        if getattr(nc, "enable_eagle_speculation", False) or getattr(
            nc, "is_eagle_draft", False
        ):
            unsupported.append("EAGLE speculation")
        if nc.flash_decoding_enabled:
            unsupported.append("flash decoding")
        if nc.attention_dp_degree > 1:
            unsupported.append("attention data parallelism")
        if nc.kv_cache_tiling:
            unsupported.append("KV cache tiling")
        if nc.padding_side != "right":
            unsupported.append("left padding")
        if nc.is_continuous_batching:
            unsupported.append("continuous batching")
        if unsupported:
            raise ValueError(
                "HybridDeltaNetCacheManager v1 does not support: "
                + ", ".join(unsupported)
            )

    def _is_deltanet_layer(self, idx: int) -> bool:
        return self.layer_types[idx] == "linear_attention"

    def get_seq_length(self, past_key_values=None):
        for idx, layer_type in enumerate(self.layer_types):
            if layer_type != "linear_attention":
                if past_key_values is None:
                    _, v_cache = self._fetch_cache(idx)
                elif len(past_key_values) == len(self.past_key_values):
                    v_cache = past_key_values[2 * idx + 1]
                else:
                    v_cache = past_key_values[idx][1]
                return v_cache.shape[2]
        return 0

    def get_deltanet_state_by_layer_id(self, idx, kvcache_buffer=None, seq_ids=None):
        recurrent_state, conv_state = self._fetch_cache(idx, kvcache_buffer)
        if seq_ids is not None:
            cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids)
            recurrent_state = torch.index_select(
                recurrent_state, dim=0, index=cache_idx
            )
            conv_state = torch.index_select(conv_state, dim=0, index=cache_idx)
        elif self.kv_cache_padding_size > 0:
            recurrent_state = recurrent_state[: -self.kv_cache_padding_size]
            conv_state = conv_state[: -self.kv_cache_padding_size]
        return recurrent_state, conv_state

    def get_cache(
        self,
        seq_len: int,
        skip_slice=False,
        kvcache_buffer=None,
        seq_ids=None,
        windowed_context_encoding_window_idx=-1,
        **kwargs,
    ):
        past_key_values = []
        for idx in range(len(self.past_key_values) // 2):
            if self._is_deltanet_layer(idx):
                past_key_values.append(
                    list(
                        self.get_deltanet_state_by_layer_id(
                            idx, kvcache_buffer, seq_ids
                        )
                    )
                )
            else:
                past_key_values.append(
                    list(
                        self.get_kv_by_layer_id(
                            idx=idx,
                            skip_slice=skip_slice,
                            seq_len=seq_len,
                            kvcache_buffer=kvcache_buffer,
                            seq_ids=seq_ids,
                            windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                            **kwargs,
                        )
                    )
                )
        return past_key_values

    def update_cache(
        self,
        is_for_context_encoding: bool,
        seq_ids: torch.Tensor,
        position_ids: torch.Tensor,
        new_key_values: List[torch.Tensor],
        seq_len: int,
        scatter_index=None,
        kv_active_mask=None,
        kvcache_buffer=None,
        windowed_context_encoding_window_idx: int = -1,
        **kwargs,
    ):
        updated_cache = []
        for idx, kv_per_layer in enumerate(new_key_values):
            if self._is_deltanet_layer(idx):
                recurrent_state, conv_state = self.update_deltanet_state_by_layer_id(
                    idx=idx,
                    seq_ids=seq_ids,
                    state_per_layer=kv_per_layer,
                    kvcache_buffer=kvcache_buffer,
                )
            elif kwargs.get("qwen_chunked_prefill_update", False):
                recurrent_state, conv_state = self.update_qwen_chunked_kv_by_layer_id(
                    idx=idx,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    kv_per_layer=kv_per_layer,
                    kvcache_buffer=kvcache_buffer,
                    valid_mask=kwargs.get("qwen_chunked_valid_mask", None),
                )
            else:
                recurrent_state, conv_state = self.update_kv_by_layer_id(
                    idx=idx,
                    is_for_context_encoding=is_for_context_encoding,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    kv_per_layer=kv_per_layer,
                    seq_len=seq_len,
                    scatter_index=scatter_index,
                    kv_active_mask=kv_active_mask,
                    kvcache_buffer=kvcache_buffer,
                    windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                    **kwargs,
                )
            updated_cache.append(recurrent_state)
            updated_cache.append(conv_state)
        return updated_cache

    def update_qwen_chunked_kv_by_layer_id(
        self,
        idx: int,
        seq_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_per_layer: Tuple[torch.Tensor, torch.Tensor],
        kvcache_buffer=None,
        valid_mask=None,
    ):
        latest_k, latest_v = kv_per_layer
        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)
        latest_k = latest_k.to(k_cache.dtype)
        latest_v = latest_v.to(v_cache.dtype)

        if seq_ids is not None:
            cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids)
            selected_k = torch.index_select(k_cache, dim=0, index=cache_idx)
            selected_v = torch.index_select(v_cache, dim=0, index=cache_idx)
        else:
            cache_idx = None
            selected_k = k_cache[: latest_k.shape[0]]
            selected_v = v_cache[: latest_v.shape[0]]

        pos = position_ids.long()
        k_index = pos[:, None, :, None].expand_as(latest_k)
        v_index = pos[:, None, :, None].expand_as(latest_v)

        if valid_mask is not None:
            valid = valid_mask.to(torch.bool)[:, None, :, None]
            old_k = torch.gather(selected_k, dim=2, index=k_index)
            old_v = torch.gather(selected_v, dim=2, index=v_index)
            latest_k = torch.where(valid, latest_k, old_k)
            latest_v = torch.where(valid, latest_v, old_v)

        updated_k = torch.scatter(selected_k, dim=2, index=k_index, src=latest_k)
        updated_v = torch.scatter(selected_v, dim=2, index=v_index, src=latest_v)

        if cache_idx is not None:
            k_row_index = cache_idx.view(-1, 1, 1, 1).expand_as(updated_k)
            v_row_index = cache_idx.view(-1, 1, 1, 1).expand_as(updated_v)
            k_cache = torch.scatter(k_cache, dim=0, index=k_row_index, src=updated_k)
            v_cache = torch.scatter(v_cache, dim=0, index=v_row_index, src=updated_v)
            return k_cache, v_cache

        if updated_k.shape[0] == k_cache.shape[0]:
            return updated_k + k_cache * 0, updated_v + v_cache * 0

        pad_rows = k_cache.shape[0] - updated_k.shape[0]
        if pad_rows > 0:
            updated_k = torch.cat([updated_k, k_cache[updated_k.shape[0] :] * 0], dim=0)
            updated_v = torch.cat([updated_v, v_cache[updated_v.shape[0] :] * 0], dim=0)
        return updated_k + k_cache * 0, updated_v + v_cache * 0

    def update_deltanet_state_by_layer_id(
        self,
        idx: int,
        seq_ids: torch.Tensor,
        state_per_layer: Tuple[torch.Tensor, torch.Tensor],
        kvcache_buffer=None,
    ):
        latest_recurrent, latest_conv = state_per_layer
        recurrent_cache, conv_cache = self._fetch_cache(idx, kvcache_buffer)
        latest_recurrent = latest_recurrent.to(recurrent_cache.dtype)
        latest_conv = latest_conv.to(conv_cache.dtype)

        if latest_recurrent.shape[0] == recurrent_cache.shape[0] and seq_ids is None:
            return (
                latest_recurrent + recurrent_cache * 0,
                latest_conv + conv_cache * 0,
            )

        if seq_ids is not None:
            cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids)
            recurrent_index = cache_idx.view(-1, 1, 1, 1).expand_as(latest_recurrent)
            conv_index = cache_idx.view(-1, 1, 1).expand_as(latest_conv)
            recurrent_cache = torch.scatter(
                input=recurrent_cache,
                dim=0,
                index=recurrent_index,
                src=latest_recurrent,
            )
            conv_cache = torch.scatter(
                input=conv_cache,
                dim=0,
                index=conv_index,
                src=latest_conv,
            )
            return recurrent_cache, conv_cache

        pad_size = recurrent_cache.shape[0] - latest_recurrent.shape[0]
        if pad_size > 0:
            latest_recurrent = torch.cat(
                [latest_recurrent, recurrent_cache[latest_recurrent.shape[0] :] * 0],
                dim=0,
            )
            latest_conv = torch.cat(
                [latest_conv, conv_cache[latest_conv.shape[0] :] * 0],
                dim=0,
            )
        return latest_recurrent + recurrent_cache * 0, latest_conv + conv_cache * 0


# ============================================================
# Model
# ============================================================


class NeuronQwen36A3BModel(NeuronBaseModel):
    def setup_attr_for_model(self, config: Qwen36A3BInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Qwen36A3BInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [
                NeuronQwen36A3BDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls(config)(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = TransposedColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )

        # mRoPE embedding for VL paths.
        self.mrope_emb = Qwen36A3BMRoPEEmbedding(config)

        # MTP draft head, called off the traced main forward by the
        # speculative-decoding driver in inference_qwen36_a3b.py.
        self.mtp_head = (
            NeuronMTPHead(config) if config.mtp_num_hidden_layers > 0 else None
        )

    def init_inference_optimization(self, config: Qwen36A3BInferenceConfig):
        super().init_inference_optimization(config)
        if getattr(config, "use_hybrid_cache_manager", False):
            self.kv_mgr = HybridDeltaNetCacheManager(
                config,
                num_kv_head=self.num_key_value_heads,
                global_rank=self.rank_util,
                attention_chunk_size=self.attention_chunk_size,
                sliding_window=self.sliding_window,
                windowed_context_encoding_size=self.windowed_context_encoding_size,
                layer_to_cache_size_mapping=self.layer_to_cache_size_mapping,
            )

    @property
    def _deltanet_state_params(self):
        """Return DeltaNet state nn.Parameters in alias order."""
        params = []
        for layer in self.layers:
            if hasattr(layer, "linear_attn"):
                params.append(layer.linear_attn.recurrent_state_buffer)
                params.append(layer.linear_attn.conv_state_buffer)
        return params

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
        """Scatter vision embeddings into text input embeddings at image token positions."""
        _, max_positions, embedding_dim = inputs_embeds.shape
        h_new = inputs_embeds.clone()
        vision_flat = vision_embeddings.view(-1, embedding_dim)
        positions_flat = vision_mask.view(-1)
        h_new.view(-1, embedding_dim).index_put_(
            (positions_flat,), vision_flat, accumulate=False
        )
        return h_new

    def get_model_output(
        self,
        input_ids=None,
        seq_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        active_mask=None,
        inputs_embeds=None,
        prev_hidden=None,
        adapter_ids=None,
        rotary_position_ids=None,
        update_cache=False,
        is_for_context_encoding=False,
        vision_embeddings=None,
        vision_mask=None,
        local_attn_mask=None,
        windowed_context_encoding_window_idx=-1,
        padding_mask=None,
        **kwargs,
    ):
        """Override to collect DeltaNet state tensors from decoder layers."""
        batch_size, seq_length = input_ids.shape[:2]
        if self.config.neuron_config.layer_boundary_markers:
            input_ids = ModuleMarkerStartWrapper()(input_ids)

        past_key_values_length = 0
        if past_key_values is not None:
            if hasattr(self.kv_mgr, "get_seq_length"):
                past_key_values_length = self.kv_mgr.get_seq_length(past_key_values)
            else:
                past_key_values_length = past_key_values[0][1].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Zero out embeddings for padding tokens: DeltaNet has no attention mask -- it runs
        # a linear recurrence over every position -- so a padding token's real embedding
        # would corrupt the recurrence state.
        if (
            is_for_context_encoding
            and attention_mask is not None
            and attention_mask.ndim == 2
        ):
            deltanet_padding_mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        else:
            deltanet_padding_mask = (
                (input_ids != self.padding_idx).unsqueeze(-1).to(inputs_embeds.dtype)
            )
        if is_for_context_encoding:
            inputs_embeds = inputs_embeds * deltanet_padding_mask

        # Vision embedding injection. Text-only calls still pass dummy vision
        # tensors to keep the traced input signature stable; those tensors have
        # one dummy entry per text token and must not overwrite text embeddings.
        if (vision_embeddings is not None) and (vision_mask is not None):
            if vision_embeddings.dtype != self.config.neuron_config.torch_dtype:
                vision_embeddings = vision_embeddings.to(
                    self.config.neuron_config.torch_dtype
                )
            has_real_vision_inputs = (
                vision_embeddings.ndim == 3
                and vision_mask.ndim == 3
                and vision_embeddings.shape[1] != seq_length
            )
            if is_for_context_encoding and has_real_vision_inputs:
                inputs_embeds = self.encode_vision_to_input(
                    inputs_embeds, vision_embeddings, vision_mask
                )
            elif is_for_context_encoding and vision_embeddings.numel() > 0:
                inputs_embeds = inputs_embeds + vision_embeddings.sum() * 0
                inputs_embeds = (
                    inputs_embeds + vision_mask.sum().to(inputs_embeds.dtype) * 0
                )

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        hidden_states = inputs_embeds

        # Get KV cache for TKG and for model-local chunked CTE.
        use_qwen_chunked_prefill = is_for_context_encoding and getattr(
            self.config, "use_qwen_hybrid_chunked_prefill", False
        )
        cache_size = (
            self.config.neuron_config.seq_len
            if use_qwen_chunked_prefill
            else self.n_positions
        )
        if (not is_for_context_encoding) or use_qwen_chunked_prefill:
            if self.kv_mgr is not None:
                past_key_values = self.kv_mgr.get_cache(
                    seq_ids=seq_ids,
                    seq_len=cache_size,
                    is_for_context_encoding=is_for_context_encoding,
                    windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                    **kwargs,
                )

        # Decoder layers
        next_decoder_cache = ()
        deltanet_state_tensors = []
        cos_cache = None
        sin_cache = None

        # Convert 2D attention_mask to 4D causal mask for CTE
        if (
            attention_mask is not None
            and attention_mask.ndim == 2
            and is_for_context_encoding
        ):
            causal = torch.ones(
                (seq_length, seq_length),
                dtype=torch.bool,
                device=attention_mask.device,
            ).tril()
            padding_4d = attention_mask[:, None, None, :].to(torch.bool)
            attention_mask = (causal[None, None, :, :] & padding_4d).to(
                attention_mask.dtype
            )

        # Pre-compute MRoPE-interleaved cos/sin. Qwen3.6 always uses the MRoPE form, even
        # for text-only inputs. Plain RoPE on the first 64 dims is NOT equivalent and
        # produces garbage logits.
        if rotary_position_ids is not None and rotary_position_ids.ndim == 3:
            cos_cache, sin_cache = self.mrope_emb(inputs_embeds, rotary_position_ids)
        elif cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.mrope_emb(inputs_embeds, position_ids)

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                seq_ids=seq_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                adapter_ids=adapter_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                rotary_position_ids=rotary_position_ids,
                kv_mgr=self.kv_mgr,
                get_kv_per_layer=False,
                update_kv_per_layer=False,
                idx=idx,
                is_for_context_encoding=is_for_context_encoding,
                seq_len=cache_size,
                residual=None,
                local_mask=local_attn_mask,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                padding_mask=padding_mask,
                deltanet_padding_mask=deltanet_padding_mask,
                qwen_chunked_prefill_update=use_qwen_chunked_prefill,
                qwen_chunked_valid_mask=deltanet_padding_mask.squeeze(-1)
                if use_qwen_chunked_prefill
                else None,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            kv = layer_outputs[1]
            next_decoder_cache += (kv,)
            cos_cache, sin_cache = layer_outputs[2:4]

            # Collect DeltaNet state tensors
            deltanet_states = layer_outputs[5] if len(layer_outputs) > 5 else None
            if deltanet_states is not None:
                deltanet_state_tensors.append(deltanet_states[0])
                deltanet_state_tensors.append(deltanet_states[1])

        # Update KV cache
        if update_cache:
            next_decoder_cache = self.kv_mgr.update_cache(
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=next_decoder_cache,
                seq_len=cache_size,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                qwen_chunked_prefill_update=use_qwen_chunked_prefill,
                qwen_chunked_valid_mask=deltanet_padding_mask.squeeze(-1)
                if use_qwen_chunked_prefill
                else None,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        self._deltanet_updated_states = deltanet_state_tensors

        return (hidden_states, next_decoder_cache)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden=None,
        adapter_ids=None,
        accepted_indices=None,
        current_length=None,
        medusa_mask=None,
        scatter_index=None,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
        tile_q_indices=None,
        tile_block_tables=None,
        tile_masks=None,
        inputs_embeds=None,
        kv_cache=None,
        active_mask=None,
        rotary_position_id=None,
        vision_embeddings=None,
        vision_mask=None,
    ):
        """Override base forward to append DeltaNet state tensors to output."""
        prev_hidden = self.set_none_if_empty(prev_hidden)
        adapter_ids = self.set_none_if_empty(adapter_ids)
        accepted_indices = self.set_none_if_empty(accepted_indices)
        current_length = self.set_none_if_empty(current_length)
        medusa_mask = self.set_none_if_empty(medusa_mask)
        scatter_index = self.set_none_if_empty(scatter_index)
        slot_mapping = self.set_none_if_empty(slot_mapping)
        active_block_table = self.set_none_if_empty(active_block_table)
        num_queries = self.set_none_if_empty(num_queries)
        computed_context_lens = self.set_none_if_empty(computed_context_lens)
        tile_q_indices = self.set_none_if_empty(tile_q_indices)
        tile_block_tables = self.set_none_if_empty(tile_block_tables)
        tile_masks = self.set_none_if_empty(tile_masks)
        inputs_embeds = self.set_none_if_empty(inputs_embeds)
        kv_cache = self.set_none_if_empty(kv_cache)
        active_mask = self.set_none_if_empty(active_mask)
        rotary_position_id = self.set_none_if_empty(rotary_position_id)
        vision_embeddings = self.set_none_if_empty(vision_embeddings)
        vision_mask = self.set_none_if_empty(vision_mask)

        is_for_context_encoding = position_ids.shape[-1] != 1 and not (
            hasattr(self.neuron_config, "speculation_length")
            and position_ids.shape[-1] == self.neuron_config.speculation_length
        )

        seq_ids = seq_ids.to(torch.int32)
        attn_mask = attention_mask

        hidden_states, updated_kv_cache = self.get_model_output(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,
            adapter_ids=adapter_ids,
            rotary_position_ids=rotary_position_id,
            update_cache=True,
            is_for_context_encoding=is_for_context_encoding,
            padding_mask=None,
            active_block_table=active_block_table,
            scatter_index=slot_mapping
            if getattr(self, "is_block_kv_layout", False)
            else scatter_index,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )

        batch_size = input_ids.shape[0]
        if not getattr(self, "sliced_hidden", False):
            if not is_for_context_encoding:
                pass
            else:
                if getattr(self.config, "use_qwen_hybrid_chunked_prefill", False):
                    if attention_mask is not None and attention_mask.ndim == 2:
                        index = (
                            attention_mask.to(torch.long).sum(dim=1, keepdim=True) - 1
                        ).clamp(min=0)
                    else:
                        index = (
                            (input_ids != self.padding_idx)
                            .sum(dim=1, keepdim=True)
                            .long()
                            - 1
                        ).clamp(min=0)
                else:
                    index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if hasattr(self.lm_head, "pad_size"):
            if self.lm_head.gather_output:
                rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
                world_size = 1
            else:
                from neuronx_distributed.parallel_layers import parallel_state

                rank_id = self.rank_util.get_rank()
                world_size = torch.distributed.get_world_size(
                    group=self.lm_head.tensor_parallel_group
                )
            from neuronx_distributed_inference.models.model_base import (
                mask_padded_logits,
            )

            logits = mask_padded_logits(
                logits, rank_id, world_size, pad_size=self.lm_head.pad_size
            )

        if self.on_device_sampling:
            res = self._sample_on_device(
                logits, sampling_params, False, is_for_context_encoding
            )
        else:
            res = logits

        outputs = [res]
        if self.neuron_config.output_logits:
            outputs += [logits]
        outputs += updated_kv_cache

        # Append DeltaNet state tensors for input_output_aliases.
        if not getattr(self.config, "use_hybrid_cache_manager", False) and hasattr(
            self, "_deltanet_updated_states"
        ):
            outputs += self._deltanet_updated_states

        return outputs


# ============================================================
# MTP draft config helper
# ============================================================


def _config_for_mtp_draft(config: Qwen36A3BInferenceConfig):
    """Deep-copy `config` collapsed to a single full-attention layer.

    The MTP draft graph owns a one-layer KV cache (the draft decoder layer is a
    standard full-attention layer), so the cache manager is built from a config
    whose `num_hidden_layers == 1` and `layer_types == ["full_attention"]`.
    """
    import copy as _copy

    draft_config = _copy.deepcopy(config)
    draft_config.num_hidden_layers = 1
    draft_config.layer_types = ["full_attention"]
    return draft_config


# ============================================================
# State Dict Converter
# ============================================================


def reorder_deltanet_qkv_for_tp(
    qkv_weight: torch.Tensor,
    tp_degree: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> torch.Tensor:
    """Pack [Q_all | K_all | V_all] into per-rank Q/K/V blocks.

    ColumnParallelLinear slices the first dimension contiguously. DeltaNet
    needs each rank to receive its local query, key, and value heads
    together, so the full HF tensor is repacked as:
    [rank0 Q | rank0 K | rank0 V | rank1 Q | rank1 K | rank1 V | ...].
    """
    if num_k_heads % tp_degree != 0:
        raise ValueError(
            f"linear_num_key_heads={num_k_heads} must be divisible by tp_degree={tp_degree}"
        )
    if num_v_heads % tp_degree != 0:
        raise ValueError(
            f"linear_num_value_heads={num_v_heads} must be divisible by tp_degree={tp_degree}"
        )

    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    q_weight = qkv_weight[:key_dim].reshape(num_k_heads, head_k_dim, -1)
    k_weight = qkv_weight[key_dim : 2 * key_dim].reshape(num_k_heads, head_k_dim, -1)
    v_weight = qkv_weight[2 * key_dim : 2 * key_dim + value_dim].reshape(
        num_v_heads, head_v_dim, -1
    )
    local_k_heads = num_k_heads // tp_degree
    local_v_heads = num_v_heads // tp_degree
    blocks = []
    for rank in range(tp_degree):
        blocks.append(
            q_weight[rank * local_k_heads : (rank + 1) * local_k_heads].reshape(
                -1, qkv_weight.shape[1]
            )
        )
        blocks.append(
            k_weight[rank * local_k_heads : (rank + 1) * local_k_heads].reshape(
                -1, qkv_weight.shape[1]
            )
        )
        blocks.append(
            v_weight[rank * local_v_heads : (rank + 1) * local_v_heads].reshape(
                -1, qkv_weight.shape[1]
            )
        )
    return torch.cat(blocks, dim=0).contiguous()


def reorder_deltanet_qkv_channels_for_tp(
    channel_tensor: torch.Tensor,
    tp_degree: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> torch.Tensor:
    """Per-rank repack of a first-dimension Q/K/V channel tensor (e.g. conv1d weight)."""
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    q_tensor = channel_tensor[:key_dim]
    k_tensor = channel_tensor[key_dim : 2 * key_dim]
    v_tensor = channel_tensor[2 * key_dim : 2 * key_dim + value_dim]
    local_key_dim = key_dim // tp_degree
    local_value_dim = value_dim // tp_degree
    blocks = []
    for rank in range(tp_degree):
        blocks.append(q_tensor[rank * local_key_dim : (rank + 1) * local_key_dim])
        blocks.append(k_tensor[rank * local_key_dim : (rank + 1) * local_key_dim])
        blocks.append(v_tensor[rank * local_value_dim : (rank + 1) * local_value_dim])
    return torch.cat(blocks, dim=0).contiguous()


def build_deltanet_in_proj_fused(
    qkv: torch.Tensor,
    z: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tp_degree: int,
) -> torch.Tensor:
    """Fuse the four DeltaNet input projections into one [hidden, I] weight.

    ``qkv`` is already per-rank-reordered ([rank0 Q|K|V | rank1 Q|K|V | ...]);
    z/a/b are plain output-sharded. Interleave them so each rank's contiguous
    slice of the fused output axis is [qkv_r | z_r | a_r | b_r] (the layout the
    TKG kernel slices), then transpose to contraction-first [hidden, I] so the
    kernel needs no runtime cat/transpose.
    """
    cd = qkv.shape[0] // tp_degree
    vd = z.shape[0] // tp_degree
    nv = a.shape[0] // tp_degree
    blocks = []
    for r in range(tp_degree):
        blocks.append(qkv[r * cd : (r + 1) * cd])
        blocks.append(z[r * vd : (r + 1) * vd])
        blocks.append(a[r * nv : (r + 1) * nv])
        blocks.append(b[r * nv : (r + 1) * nv])
    return torch.cat(blocks, dim=0).t().contiguous()


def build_gqa_qkv_fused(q, k, v, num_q_heads, num_kv_heads, head_dim, tp_degree):
    """Fuse GQA q/k/v projections into one contraction-first [hidden, I] weight.

    KV heads are first replicated to tp_degree (repeat_interleave, matching NxDI's
    GroupQueryAttention preshard), then each rank's contiguous output slice is
    head-major [q_4r..q_4r+3 | k_r | v_r] -- the layout the fused TKG kernel slices.
    Returns [hidden, I] (RowParallelLinear shards I).
    """
    reps = tp_degree // num_kv_heads
    k = k.reshape(num_kv_heads, head_dim, -1).repeat_interleave(reps, dim=0)
    v = v.reshape(num_kv_heads, head_dim, -1).repeat_interleave(reps, dim=0)
    q = q.reshape(num_q_heads, head_dim, -1)
    qh = num_q_heads // tp_degree  # q heads per rank
    blocks = []
    for r in range(tp_degree):
        blocks.append(q[r * qh : (r + 1) * qh].reshape(qh * head_dim, -1))
        blocks.append(k[r].reshape(head_dim, -1))
        blocks.append(v[r].reshape(head_dim, -1))
    return torch.cat(blocks, dim=0).t().contiguous()


def _convert_full_attention_block(sd, prefix, config):
    """GQA per-layer conversions on `{prefix}.self_attn.*`, in place.

    Splits the gated q_proj into query + sigmoid output gate, renames the q/k
    norms, and seeds rank_util. Shared by the main layers and the MTP head's
    decoder layer.
    """
    sd[f"{prefix}.self_attn.rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    for src, dst in (("q_norm", "q_layernorm"), ("k_norm", "k_layernorm")):
        key = f"{prefix}.self_attn.{src}.weight"
        if key in sd:
            sd[f"{prefix}.self_attn.{dst}.weight"] = sd.pop(key).detach().clone()

    # q_proj is doubled and interleaved per head: [head_i query | head_i gate].
    q_proj_key = f"{prefix}.self_attn.q_proj.weight"
    if q_proj_key in sd:
        num_heads, head_dim = config.num_attention_heads, config.head_dim
        w = sd.pop(q_proj_key).reshape(num_heads, head_dim * 2, config.hidden_size)
        sd[q_proj_key] = w[:, :head_dim, :].reshape(
            num_heads * head_dim, config.hidden_size
        )
        sd[f"{prefix}.self_attn.output_gate_proj.weight"] = w[:, head_dim:, :].reshape(
            num_heads * head_dim, config.hidden_size
        )

    # Fused GQA TKG kernel weights (decode/verify): head-major contraction-first
    # qkv + gate, and o_proj stored transposed. Built as copies; prefill keeps the
    # standard q/k/v/output_gate/o_proj. Mirrors DeltaNet in_proj_fused / out_proj.
    if getattr(config, "use_tkg_attention_kernel", False):
        tp = config.neuron_config.tp_degree
        num_heads, head_dim = config.num_attention_heads, config.head_dim
        q_w = sd[f"{prefix}.self_attn.q_proj.weight"]
        k_w = sd[f"{prefix}.self_attn.k_proj.weight"]
        v_w = sd[f"{prefix}.self_attn.v_proj.weight"]
        sd[f"{prefix}.self_attn.gqa_qkv_w.weight"] = build_gqa_qkv_fused(
            q_w, k_w, v_w, num_heads, config.num_key_value_heads, head_dim, tp
        )
        sd[f"{prefix}.self_attn.gqa_gate_w.weight"] = (
            sd[f"{prefix}.self_attn.output_gate_proj.weight"].t().contiguous()
        )
        sd[f"{prefix}.self_attn.gqa_o_proj_w.weight"] = (
            sd[f"{prefix}.self_attn.o_proj.weight"].t().contiguous()
        )

    if config.neuron_config.fused_qkv:
        q, k, v = (f"{prefix}.self_attn.{p}_proj.weight" for p in ("q", "k", "v"))
        if q in sd:
            sd[f"{prefix}.self_attn.Wqkv.weight"] = torch.cat([sd[q], sd[k], sd[v]])
            for key in (q, k, v):
                del sd[key]


def _convert_moe_block(sd, prefix, config):
    """MoE FFN conversions on `{prefix}.mlp.*`, in place.

    Renames the router and transposes the fused expert tensors into the layout
    ExpertMLPsV2 expects: gate_up_proj (E, H, 2I) and down_proj (E, I, H). The
    sigmoid-gated shared expert passes through under its HF names.
    """
    gate_key = f"{prefix}.mlp.gate.weight"
    if gate_key in sd:
        sd[f"{prefix}.mlp.moe.router.linear_router.weight"] = (
            sd.pop(gate_key).detach().clone()
        )

    for hf_suffix, nxdi_suffix in (
        ("experts.gate_up_proj", "moe.expert_mlps.mlp_op.gate_up_proj.weight"),
        ("experts.down_proj", "moe.expert_mlps.mlp_op.down_proj.weight"),
    ):
        src = f"{prefix}.mlp.{hf_suffix}"
        if src in sd:
            sd[f"{prefix}.mlp.{nxdi_suffix}"] = sd.pop(src).transpose(1, 2).contiguous()

    # Fused MoE layer kernel: stow the shared expert contraction-first in place so the
    # kernel reads it with no runtime transpose; SharedExpertMLP(transposed=True) consumes
    # the same layout. Router/sigma stay standard for the decode/prefill path and get small
    # rank-replicated transposed copies, since NxDI owns the router. Experts already match.
    if getattr(config, "use_moe_layer_kernel", False):
        for name in ("gate_proj", "up_proj", "down_proj"):
            skey = f"{prefix}.mlp.shared_expert.{name}.weight"
            if skey in sd:
                sd[skey] = sd[skey].t().contiguous()
        # .clone(): the copies must NOT share storage with the originals we keep for the PyTorch path
        # (a [1,H].t() -> [H,1] is trivially "contiguous" so .contiguous() alone won't copy; safetensors
        # rejects shared storage).
        rkey = f"{prefix}.mlp.moe.router.linear_router.weight"
        if rkey in sd:
            sd[f"{prefix}.mlp.moe_router_w_t"] = sd[rkey].t().contiguous().clone()
        sgkey = f"{prefix}.mlp.shared_expert_gate.weight"
        if sgkey in sd:
            sd[f"{prefix}.mlp.moe_sigma_gate_t"] = sd[sgkey].t().contiguous().clone()


def convert_qwen36_a3b_hf_to_neuron_state_dict(neuron_state_dict, config):
    """Convert HF Qwen3.6-A3B weights to NxDI keys / shapes.

    Per-layer mappings:

    DeltaNet (linear_attention):
        in_proj_qkv is repacked across TP ranks; conv1d, A_log, dt_bias
        are folded into ColumnParallelLinear parameter containers
        (conv1d_weight.weight, A_log_weight.weight, dt_bias_weight.weight).

    GQA (full_attention):
        q_proj is doubled (Q + output gate, interleaved per head): split
        into q_proj + output_gate_proj. q_norm/k_norm renamed to
        q_layernorm/k_layernorm. Optionally fused into Wqkv.

    MoE FFN (all layers):
        mlp.gate -> mlp.moe.router.linear_router
        mlp.experts.{0..N-1}.{gate,up,down}_proj are stacked + transposed
        into mlp.moe.expert_mlps.mlp_op.{gate_up_proj,down_proj}.
        mlp.shared_expert.{gate,up,down}_proj : pass-through (owned by
        NeuronMoEBlock, applied with a sigmoid gate over the output).
        mlp.shared_expert_gate.weight : pass-through.

    MTP (single layer, tied embeddings/LM head):
        mtp.fc                                    -> mtp_head.eh_proj
        mtp.pre_fc_norm_embedding                 -> mtp_head.embed_norm
        mtp.pre_fc_norm_hidden                    -> mtp_head.hidden_norm
        mtp.norm                                  -> mtp_head.final_norm
        mtp.layers.0.input_layernorm              -> mtp_head.decoder_layer.input_layernorm
        mtp.layers.0.self_attn.{q,k,v,o}_proj     -> mtp_head.decoder_layer.self_attn.{q,k,v,o}_proj
        mtp.layers.0.self_attn.{q,k}_norm         -> mtp_head.decoder_layer.self_attn.{q,k}_norm
        mtp.layers.0.post_attention_layernorm     -> mtp_head.decoder_layer.post_attention_layernorm
        mtp.layers.0.mlp.<rest>                   -> mtp_head.decoder_layer.mlp.<rest>
        lm_head.weight (main)                     -> mtp_head.mtp_lm_head.weight (tied copy, no mtp.lm_head in ckpt)
        (then _convert_full_attention_block + _convert_moe_block run on mtp_head.decoder_layer)

    All Qwen RMSNorm weights are converted from `(1 + w)` to standard form.
    """
    # Add rank_util
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0,
        config.neuron_config.tp_degree,
        dtype=torch.int32,
    )

    # HF norms use `output = norm(x) * (1 + weight)` with weight ~ 0; NxDI
    # RMSNorm uses `output = norm(x) * weight` with weight ~ 1. Add 1 to convert.
    norm_keys_to_convert = []
    for l in range(config.num_hidden_layers):
        norm_keys_to_convert.append(f"layers.{l}.input_layernorm.weight")
        norm_keys_to_convert.append(f"layers.{l}.post_attention_layernorm.weight")
        if config.layer_types[l] == "full_attention":
            norm_keys_to_convert.append(f"layers.{l}.self_attn.q_norm.weight")
            norm_keys_to_convert.append(f"layers.{l}.self_attn.k_norm.weight")
    norm_keys_to_convert.append("norm.weight")

    for nk in norm_keys_to_convert:
        if getattr(config, 'flashrt_faithful_numerics', False):
            continue
        if nk in neuron_state_dict:
            old_val = neuron_state_dict[nk]
            neuron_state_dict[nk] = old_val.float() + 1.0
            if "layers.0." in nk or nk == "norm.weight":
                logger.debug(
                    f"[NORM FIX] {nk}: mean {old_val.float().mean():.4f} -> {neuron_state_dict[nk].mean():.4f}"
                )
        else:
            if "layers.0." in nk or nk == "norm.weight":
                logger.warning(f"[NORM FIX] key not found: {nk}")

    for l in range(config.num_hidden_layers):
        layer_type = config.layer_types[l]

        # === DeltaNet layers ===
        if layer_type == "linear_attention":
            tp = config.neuron_config.tp_degree
            reorder_args = (
                tp,
                config.linear_num_key_heads,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            )
            # Fuse the four input projections into one contraction-first
            # [hidden, I] weight (no runtime cat/transpose in the TKG kernel).
            qkv_key = f"layers.{l}.linear_attn.in_proj_qkv.weight"
            if qkv_key in neuron_state_dict:
                qkv = neuron_state_dict.pop(qkv_key)
                if tp > 1:
                    qkv = reorder_deltanet_qkv_for_tp(qkv, *reorder_args)
                z = neuron_state_dict.pop(f"layers.{l}.linear_attn.in_proj_z.weight")
                a = neuron_state_dict.pop(f"layers.{l}.linear_attn.in_proj_a.weight")
                b = neuron_state_dict.pop(f"layers.{l}.linear_attn.in_proj_b.weight")
                neuron_state_dict[f"layers.{l}.linear_attn.in_proj_fused.weight"] = (
                    build_deltanet_in_proj_fused(qkv, z, a, b, tp)
                )

            # Store o_proj transposed ([value_dim, hidden]) for the kernel / matmul.
            out_proj_key = f"layers.{l}.linear_attn.out_proj.weight"
            if out_proj_key in neuron_state_dict:
                neuron_state_dict[out_proj_key] = (
                    neuron_state_dict[out_proj_key].t().contiguous()
                )

            conv_key = f"layers.{l}.linear_attn.conv1d.weight"
            conv_weight_key = f"layers.{l}.linear_attn.conv1d_weight.weight"
            if conv_key in neuron_state_dict:
                conv_weight = neuron_state_dict.pop(conv_key)
                if tp > 1:
                    conv_weight = reorder_deltanet_qkv_channels_for_tp(
                        conv_weight,
                        *reorder_args,
                    )
                neuron_state_dict[conv_weight_key] = conv_weight.squeeze(1).contiguous()

            for vector_name in ("A_log", "dt_bias"):
                vector_key = f"layers.{l}.linear_attn.{vector_name}"
                vector_weight_key = (
                    f"layers.{l}.linear_attn.{vector_name}_weight.weight"
                )
                if vector_key in neuron_state_dict:
                    neuron_state_dict[vector_weight_key] = (
                        neuron_state_dict.pop(vector_key).reshape(-1, 1).contiguous()
                    )

        # === Attention layers ===  (GQA; DeltaNet handled above)
        if layer_type == "full_attention":
            _convert_full_attention_block(neuron_state_dict, f"layers.{l}", config)

        # MoE FFN (every layer). The sigmoid-gated shared expert passes through
        # under its HF names (owned by NeuronMoEBlock).
        _convert_moe_block(neuron_state_dict, f"layers.{l}", config)

        gc.collect()

    # MTP head (Qwen3-Next layout). The checkpoint ships no draft embedding and no draft
    # LM head, so both are tied to the main model. When MTP is disabled, drop the keys
    # so they don't trip load_state_dict.

    # Store lm_head transposed ([hidden, vocab]) for TransposedColumnParallelLinear.
    if "lm_head.weight" in neuron_state_dict:
        neuron_state_dict["lm_head.weight"] = (
            neuron_state_dict["lm_head.weight"].t().contiguous()
        )

    if config.mtp_num_hidden_layers == 0:
        for k in [k for k in neuron_state_dict if k.startswith("mtp.")]:
            neuron_state_dict.pop(k)
    else:
        top_rename = {
            "mtp.fc.weight": "mtp_head.eh_proj.weight",
            "mtp.pre_fc_norm_embedding.weight": "mtp_head.embed_norm.weight",
            "mtp.pre_fc_norm_hidden.weight": "mtp_head.hidden_norm.weight",
            "mtp.norm.weight": "mtp_head.final_norm.weight",
        }
        for old, new in top_rename.items():
            if old in neuron_state_dict:
                neuron_state_dict[new] = neuron_state_dict.pop(old).detach().clone()

        # Store eh_proj transposed ([2H, hidden]) for TransposedColumnParallelLinear.
        ehk = "mtp_head.eh_proj.weight"
        if ehk in neuron_state_dict:
            neuron_state_dict[ehk] = neuron_state_dict[ehk].t().contiguous()

        layer_pfx = "mtp.layers.0."
        for k in [k for k in neuron_state_dict if k.startswith(layer_pfx)]:
            neuron_state_dict[f"mtp_head.decoder_layer.{k[len(layer_pfx) :]}"] = (
                neuron_state_dict.pop(k).detach().clone()
            )
        for k in [k for k in neuron_state_dict if k.startswith("mtp.")]:
            neuron_state_dict.pop(k)  # drop any stray mtp.* (none expected)

        # (1 + w) for every MTP RMSNorm (head norms + the draft layer's norms).
        for nk in (
            "mtp_head.embed_norm.weight",
            "mtp_head.hidden_norm.weight",
            "mtp_head.final_norm.weight",
            "mtp_head.decoder_layer.input_layernorm.weight",
            "mtp_head.decoder_layer.post_attention_layernorm.weight",
            "mtp_head.decoder_layer.self_attn.q_norm.weight",
            "mtp_head.decoder_layer.self_attn.k_norm.weight",
        ):
            if nk in neuron_state_dict:
                neuron_state_dict[nk] = neuron_state_dict[nk].float() + 1.0

        # The draft layer is a standard full-attention + MoE layer.
        _convert_full_attention_block(
            neuron_state_dict, "mtp_head.decoder_layer", config
        )
        _convert_moe_block(neuron_state_dict, "mtp_head.decoder_layer", config)

        # Tie the draft LM head to the main model's (no mtp.lm_head in checkpoint).
        neuron_state_dict["mtp_head.mtp_lm_head.weight"] = (
            neuron_state_dict["lm_head.weight"].detach().clone()
        )

    return neuron_state_dict


# ============================================================
# Custom ModelWrapper and DecoderModelInstance for DeltaNet state aliasing
# ============================================================


class Qwen36A3BDecoderModelInstance(DecoderModelInstance):
    """Custom DecoderModelInstance that adds DeltaNet state buffers to input_output_aliases."""
    def get(self, bucket_rank, **kwargs):
        """Override to add DeltaNet state aliases after KV cache aliases."""
        module, input_output_aliases = super().get(bucket_rank, **kwargs)

        num_output_from_trace = 1 if not self.neuron_config.output_logits else 2

        if module.kv_mgr is not None:
            num_kv = len(module.kv_mgr.past_key_values)
        else:
            num_kv = 0

        state_start_idx = num_output_from_trace + num_kv

        if not getattr(module.config, "use_hybrid_cache_manager", False) and hasattr(
            module, "_deltanet_state_params"
        ):
            seed_params = list(module._deltanet_state_params)
            for i, param in enumerate(seed_params):
                input_output_aliases[param] = state_start_idx + i

        return module, input_output_aliases


class Qwen36A3BModelWrapper(ModelWrapper):
    """Custom ModelWrapper for VL support with mRoPE and vision inputs."""
    def get_model_instance(self):
        return Qwen36A3BDecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )

    def input_generator(self):
        """Generate inputs including mrope_position_ids, vision_embeddings, and vision_mask."""
        base_inputs = super().input_generator()
        extended_inputs = []

        for bucket_inputs in base_inputs:
            input_ids = bucket_inputs[0]
            batch_size = input_ids.shape[0]
            n_active_tokens = input_ids.shape[1]

            is_cte = n_active_tokens > 1

            if is_cte:
                mrope_position_ids = (
                    torch.arange(0, n_active_tokens, dtype=torch.int32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(3, batch_size, -1)
                    .contiguous()
                )

                vision_embeddings = torch.zeros(
                    (batch_size, n_active_tokens, self.config.hidden_size),
                    dtype=self.config.neuron_config.torch_dtype,
                )
                vision_mask = torch.full(
                    (batch_size, n_active_tokens, 1),
                    fill_value=n_active_tokens - 1,
                    dtype=torch.int32,
                )
            else:
                mrope_position_ids = torch.zeros((0,), dtype=torch.int32)
                vision_embeddings = torch.zeros(
                    (0,), dtype=self.config.neuron_config.torch_dtype
                )
                vision_mask = torch.zeros((0,), dtype=torch.int32)

            padded = list(bucket_inputs)
            while len(padded) < 21:
                padded.append(torch.zeros((0,), dtype=torch.int32))
            padded.append(mrope_position_ids)  # position 21
            padded.append(vision_embeddings)  # position 22
            padded.append(vision_mask)  # position 23

            extended_inputs.append(tuple(padded))

        return extended_inputs

    def pad_inputs(self, *args, pad_type="first_fit"):
        """Override to pad mrope_position_ids and vision inputs to bucket size."""
        orig_mrope = args[21] if len(args) >= 22 else None
        orig_vis_emb = args[22] if len(args) >= 23 else None
        orig_vis_mask = args[23] if len(args) >= 24 else None

        padded_args = super().pad_inputs(*args, pad_type=pad_type)

        if len(padded_args) >= 24 and orig_mrope is not None:
            padded_seq_len = padded_args[0].shape[1]
            batch_size = padded_args[0].shape[0]
            is_cte = padded_seq_len > 1

            if is_cte:
                current_mrope = orig_mrope
                current_vis_emb = orig_vis_emb
                current_vis_mask = orig_vis_mask

                if (
                    current_mrope.ndim == 3
                    and current_mrope.shape[-1] != padded_seq_len
                ):
                    orig_len = current_mrope.shape[-1]
                    pad_size = padded_seq_len - orig_len
                    last_pos = current_mrope[:, :, -1:]
                    pad_offsets = torch.arange(
                        1, pad_size + 1, dtype=current_mrope.dtype
                    )
                    pad_offsets = (
                        pad_offsets.unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
                    )
                    mrope_pad = last_pos + pad_offsets
                    mrope_position_ids = torch.cat([current_mrope, mrope_pad], dim=-1)
                elif current_mrope.ndim == 3:
                    mrope_position_ids = current_mrope
                else:
                    mrope_position_ids = (
                        torch.arange(0, padded_seq_len, dtype=torch.int32)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand(3, batch_size, -1)
                        .contiguous()
                    )

                if (
                    current_vis_emb is not None
                    and current_vis_emb.ndim == 3
                    and current_vis_emb.shape[1] < padded_seq_len
                ):
                    pad_emb = torch.zeros(
                        (
                            batch_size,
                            padded_seq_len - current_vis_emb.shape[1],
                            current_vis_emb.shape[2],
                        ),
                        dtype=current_vis_emb.dtype,
                    )
                    vision_embeddings = torch.cat([current_vis_emb, pad_emb], dim=1)
                elif current_vis_emb is not None and current_vis_emb.ndim == 3:
                    vision_embeddings = current_vis_emb[:, :padded_seq_len]
                else:
                    vision_embeddings = torch.zeros(
                        (batch_size, padded_seq_len, self.config.hidden_size),
                        dtype=self.config.neuron_config.torch_dtype,
                    )

                if (
                    current_vis_mask is not None
                    and current_vis_mask.ndim == 3
                    and current_vis_mask.shape[1] < padded_seq_len
                ):
                    pad_mask = torch.full(
                        (batch_size, padded_seq_len - current_vis_mask.shape[1], 1),
                        fill_value=padded_seq_len - 1,
                        dtype=torch.int32,
                    )
                    vision_mask = torch.cat([current_vis_mask, pad_mask], dim=1)
                elif current_vis_mask is not None and current_vis_mask.ndim == 3:
                    vision_mask = current_vis_mask[:, :padded_seq_len]
                else:
                    vision_mask = torch.full(
                        (batch_size, padded_seq_len, 1),
                        fill_value=padded_seq_len - 1,
                        dtype=torch.int32,
                    )

                padded_args = (
                    *padded_args[:21],
                    mrope_position_ids,
                    vision_embeddings,
                    vision_mask,
                )

                padded_args = list(padded_args)
                padded_args[23] = padded_args[23].clamp(max=padded_seq_len - 1)
                padded_args = tuple(padded_args)

        return padded_args


# ============================================================
# Fused speculative decoding (NxDI NeuronFusedSpecModel, EAGLE path)
# ============================================================
#
# The MTP head drafts; the 40-layer backbone is the verify-mode target.
# Qwen36FusedSpecModel overrides the two EAGLE sub-forwards to run DeltaNet verify
# and commit the accepted recurrent/conv state in-graph. One CTE graph plus one
# FUSED_SPECULATION_MODEL_TAG graph, with the draft->verify->accept loop device-side.


def _greedy_argmax(lm_head, logits, rank_util, disable_argmax_kernel=False):
    """Per-position greedy token id over a vocab head, masking padding columns
    first so a pad column can't win. For a sharded head (gather_output=False)
    the argmax is a vocab-parallel reduction (NxD ``nxd_argmax`` gathers only the
    per-rank max value+index), so the full-vocab logits are never materialized."""
    logits = logits.float()
    if hasattr(lm_head, "pad_size"):
        if lm_head.gather_output:
            rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
            world_size = 1
        else:
            rank_id = rank_util.get_rank()
            world_size = lm_head.tensor_parallel_group.size()
        logits = mask_padded_logits(
            logits, rank_id, world_size, pad_size=lm_head.pad_size
        )
    if lm_head.gather_output:
        return torch.argmax(logits, dim=-1).to(torch.int32)
    return nxd_argmax(
        tensor=logits,
        dim=-1,
        gather_dim=-1,
        keepdim=False,
        process_group=lm_head.tensor_parallel_group,
        disable_argmax_kernel=disable_argmax_kernel,
    ).to(torch.int32)


def _decode_kernel_mask(sa, committed_len, bsz, q, L, n_positions, device):
    """The fused-kernel GQA mask for a decode block of q tokens at ``committed_len``.

    Prior over the committed cache, intra-block causal over the active tokens. ``committed_len``
    is [B, 1]; slots below it are the block's attendable history, which is how a stage sees a slot
    an earlier stage in the same launch has just written.
    """
    cache_positions = torch.arange(n_positions, device=device).view(1, 1, 1, -1)
    prior = (cache_positions < committed_len.view(bsz, 1, 1, 1)).expand(
        bsz, 1, q, n_positions
    )
    block_idx = torch.arange(q, device=device)
    active = (block_idx.view(1, 1, q, 1) >= block_idx.view(1, 1, 1, q)).expand(
        bsz, 1, q, q
    )
    return sa._build_kernel_mask(prior, active, bsz, q, L, device)


class Qwen36MTPDraft(NeuronBaseModel):
    """MTP head as an EAGLE draft worker.

    Drafts x_{t+2} from the trunk hidden h_t (rolling buffer) and embed(x_{t+1}),
    does on-device greedy argmax, and returns its one-layer KV + hidden so the
    parent loop can chain it. Called at n_active in {1 (draft step), spec_len
    (final KV repopulation), prompt_len (prefill)}. Manages its own aliased KV
    (kv_mgr), so the threaded ``kv_cache`` arg is ignored (k=1: no within-trace
    chaining needed). Return: [sampled_tokens, *kv, hidden].
    """
    def setup_attr_for_model(self, config: "Qwen36A3BInferenceConfig"):
        self.on_device_sampling = False
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: "Qwen36A3BInferenceConfig"):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.mtp_head = NeuronMTPHead(config)
        self.mrope_emb = Qwen36A3BMRoPEEmbedding(config)

    def init_inference_optimization(self, config: "Qwen36A3BInferenceConfig"):
        # One-layer KV for the draft's full-attention layer. Named ``kv_mgr`` so
        # the fused instance aliases module.draft_model.kv_mgr.past_key_values.
        self.kv_mgr = KVCacheManager(
            _config_for_mtp_draft(config),
            num_kv_head=self.num_key_value_heads,
            global_rank=self.rank_util,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden=None,
        *args,
        kv_cache=None,
        **kwargs,
    ):
        prev_hidden = self.set_none_if_empty(prev_hidden)
        seq_ids = seq_ids.to(torch.int32)
        bsz, q = input_ids.shape[:2]
        position_ids = position_ids.view(-1, q).long()
        spec_len = self.neuron_config.speculation_length

        # Draft step (q == 1) and replay (q == spec_len) each as one NKI launch;
        # prefill keeps the XLA path.
        if getattr(self.config, "use_draft_megakernel", False) and q in (1, spec_len):
            return self._forward_draft_megakernel(
                input_ids, position_ids, seq_ids, prev_hidden
            )

        next_embeds = self.embed_tokens(input_ids)
        cos_cache, sin_cache = self.mrope_emb(next_embeds, position_ids)

        # Prefill when the width is neither a single decode token nor the verify
        # block (mirrors the fused model's own CTE/TKG dispatch).
        is_cte = q > 1 and q != spec_len

        # Prefill attends via the causal mask, not a cache: the KV is fetched only
        # for decode so the GQA attention takes its prefill path (mirrors
        # get_model_output, where CTE leaves past_key_value None).
        if is_cte:
            past_key_value = None
        else:
            past_key_values = self.kv_mgr.get_cache(
                seq_ids=seq_ids,
                seq_len=self.n_positions,
                is_for_context_encoding=False,
            )
            past_key_value = past_key_values[0] if past_key_values is not None else None

        if is_cte:
            # 4D causal + padding mask over the block (mirrors get_model_output).
            causal = torch.ones(
                (q, q), dtype=torch.bool, device=input_ids.device
            ).tril()
            pad = (input_ids != self.padding_idx)[:, None, None, :]
            layer_attn_mask = (causal[None, None, :, :] & pad).to(torch.bool)
            active_mask = None
        else:
            # Decode: prior over the committed cache + intra-block causal active
            # mask (verify-style; handles q == 1 and q == spec_len).
            cache_positions = torch.arange(
                self.n_positions, device=input_ids.device
            ).view(1, 1, 1, -1)
            committed_len = position_ids[:, 0:1].view(bsz, 1, 1, 1)
            layer_attn_mask = (cache_positions < committed_len).expand(
                bsz, 1, q, self.n_positions
            )
            block_idx = torch.arange(q, device=input_ids.device)
            active_mask = (
                block_idx.view(1, 1, q, 1) >= block_idx.view(1, 1, 1, q)
            ).expand(bsz, 1, q, q)

        # The fused GQA kernel writes the draft KV in place, so the mutated cache handles are the
        # updated KV and the host scatter is skipped (mirrors the draft megakernel).
        inplace_kv = not is_cte and getattr(
            self.config, "use_tkg_attention_kernel", False
        )
        draft_logits, present_key_value, hidden = self.mtp_head.draft_step(
            prev_hidden,
            next_embeds,
            attention_mask=layer_attn_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            active_mask=active_mask,
            seq_ids=seq_ids,
            is_for_context_encoding=is_cte,
            kv_mgr=self.kv_mgr,
            get_kv_per_layer=False,
            update_kv_per_layer=False,
            idx=0,
            seq_len=self.n_positions,
            kv_write_idx=(
                position_ids[:, 0:1].to(torch.int32) if inplace_kv else None
            ),
        )

        if inplace_kv:
            updated_kv = list(present_key_value)
        else:
            updated_kv = self.kv_mgr.update_cache(
                is_for_context_encoding=is_cte,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=[present_key_value],
                seq_len=self.n_positions,
            )

        sampled = _greedy_argmax(  # [B, q]
            self.mtp_head.mtp_lm_head,
            draft_logits,
            self.rank_util,
            disable_argmax_kernel=self.neuron_config.disable_argmax_kernel,
        )
        return [sampled, *updated_kv, hidden]

    def _collect_mtp_weights(self, k_cache, v_cache):
        """The MTP head's kernel-layout weights, keyed by the megakernels' own field names.

        Shared by the standalone draft launch and the fused round, so the two cannot pick up
        different tensors for the same field.
        """
        H = self.hidden_size
        cfg = self.config
        head = self.mtp_head
        layer = head.decoder_layer
        sa = layer.self_attn
        mlp = layer.mlp
        E = cfg.num_experts
        i_rank = cfg.moe_intermediate_size // cfg.neuron_config.tp_degree

        front = {
            "embed_w": self.embed_tokens.weight,
            "gamma_e": head.embed_norm.weight.reshape(1, H),
            "gamma_h": head.hidden_norm.weight.reshape(1, H),
            "eh_w": head.eh_proj.weight,
        }
        gqa = {
            "in_gamma": layer.input_layernorm.weight.reshape(1, H),
            "qkv_w": sa.gqa_qkv_w.weight,
            "gate_w": sa.gqa_gate_w.weight,
            "gamma_q": sa.q_layernorm.weight,
            "gamma_k": sa.k_layernorm.weight,
            "o_proj_w": sa.gqa_o_proj_w.weight,
            "k_cache": k_cache,
            "v_cache": v_cache,
        }
        moe = {
            "gamma": layer.post_attention_layernorm.weight.reshape(1, H),
            "router_w": mlp.moe_router_w_t,
            "gate_up_w": mlp.moe.expert_mlps.mlp_op.gate_up_proj.weight.reshape(
                E, H, 2, i_rank
            ),
            "down_w": mlp.moe.expert_mlps.mlp_op.down_proj.weight,
            "sigma_gate_w": mlp.moe_sigma_gate_t,
            "shared_gate_w": mlp.shared_expert.gate_proj.weight,
            "shared_up_w": mlp.shared_expert.up_proj.weight,
            "shared_down_w": mlp.shared_expert.down_proj.weight,
        }
        mtp_head_w = {
            "final_gamma": head.final_norm.weight.reshape(1, H),
            "lm_head_w": head.mtp_lm_head.weight,
        }
        return front, gqa, moe, mtp_head_w

    def _forward_draft_megakernel(self, input_ids, position_ids, seq_ids, prev_hidden):
        """Draft step or replay as one NKI launch: ids + hidden -> [token, kv, hidden].

        The kernel embeds, runs the eh_proj front end, the GQA+MoE draft layer and
        the fused vocab head, and scatters the active K/V into the cache in place
        at position_ids -- so ``kv_mgr.update_cache`` is skipped and the mutated
        cache handles are the updated KV. ``hidden`` is pre-final-norm.

        The replay (q == spec_len) rewrites the whole block, so its logits are dead:
        it takes the headless build and returns None in the token slot.
        """
        bsz, q = input_ids.shape[:2]
        cfg = self.config

        cos_cache, sin_cache = self.mrope_emb(self.embed_tokens.weight, position_ids)
        past_key_values = self.kv_mgr.get_cache(
            seq_ids=seq_ids,
            seq_len=self.n_positions,
            is_for_context_encoding=False,
        )
        k_cache, v_cache = past_key_values[0]

        mask = _decode_kernel_mask(
            self.mtp_head.decoder_layer.self_attn,
            position_ids[:, 0:1],
            bsz,
            q,
            k_cache.shape[-1],
            self.n_positions,
            input_ids.device,
        )

        front, gqa, moe, mtp_head_w = self._collect_mtp_weights(k_cache, v_cache)

        replica_groups = (list(range(cfg.neuron_config.tp_degree)),)
        with_lm_head = q == 1
        megakernel = build_draft_megakernel(with_lm_head=with_lm_head)
        flat = flatten_draft_args(
            input_ids.to(torch.int32),
            prev_hidden,
            position_ids[:, 0:1].to(torch.int32),
            cos_cache[0],
            sin_cache[0],
            mask,
            front["embed_w"],
            front["gamma_e"],
            front["gamma_h"],
            front["eh_w"],
            gqa,
            moe,
            cfg.rms_norm_eps,
            replica_groups,
            final_gamma=mtp_head_w["final_gamma"] if with_lm_head else None,
            lm_head_w=mtp_head_w["lm_head_w"] if with_lm_head else None,
        )
        if with_lm_head:
            tokens, hidden, k_cache, v_cache, _, _ = megakernel[2](*flat)
        else:
            tokens = None
            hidden, k_cache, v_cache, _, _ = megakernel[2](*flat)
        hidden = hidden.to(cfg.neuron_config.torch_dtype)
        return [tokens, k_cache, v_cache, hidden]


class Qwen36SpecTarget(NeuronQwen36A3BModel):
    """Verify-mode target for fused speculation.

    Prefill (n_active != spec_len): the normal backbone forward, which writes the
    live DeltaNet + GQA state. Verify block (n_active == spec_len): the spec block
    with DeltaNet layers in verify_mode (read-only seed + per-position S/conv
    candidates) and GQA against the live KV; the candidates are stashed on
    ``self._verify_candidates`` for the in-graph commit. The per-position greedy
    argmax is a sharded vocab-parallel reduction (lm_head stays sharded).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._verify_candidates = []
        # Capture the full pre-final-norm hidden (the draft's input / rolling
        # buffer seed) via a forward pre-hook on the final norm.
        self._captured_prenorm = None
        self.norm.register_forward_pre_hook(self._capture_prenorm)

    def _capture_prenorm(self, module, args):
        self._captured_prenorm = args[0]

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        *args,
        **kwargs,
    ):
        if input_ids.shape[1] == self.neuron_config.speculation_length:
            return self._verify_forward(
                input_ids, attention_mask, position_ids, seq_ids, sampling_params
            )
        return super().forward(
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            *args,
            **kwargs,
        )

    def _verify_prologue(self, batch_size, seq_len, position_ids, seq_ids, device):
        """Shared entry work for every verify path: rope, cache handles, masks.

        Takes the block shape rather than the ids: no path needs the ids here, and the fused round
        does not have them on the host at all -- its speculative token never leaves the kernel.
        ``mrope_emb`` reads its first argument for device AND dtype -- an integer tensor there
        truncates cos/sin to {0, 1} and destroys RoPE.
        """
        position_ids = position_ids.view(-1, seq_len).long()

        cos_cache, sin_cache = self.mrope_emb(self.embed_tokens.weight, position_ids)
        past_key_values = self.kv_mgr.get_cache(
            seq_ids=seq_ids,
            seq_len=self.n_positions,
            is_for_context_encoding=False,
        )

        # Prior over the committed cache (shared by both block tokens) + intra-
        # block causal active mask. The in-block tokens' K/V is not cached yet.
        cache_positions = torch.arange(self.n_positions, device=device).view(
            1, 1, 1, -1
        )
        committed_len = position_ids[:, 0:1].view(batch_size, 1, 1, 1)
        prior_mask = (cache_positions < committed_len).expand(
            batch_size, 1, seq_len, self.n_positions
        )
        block_idx = torch.arange(seq_len, device=device)
        verify_active_mask = (
            block_idx.view(1, 1, seq_len, 1) >= block_idx.view(1, 1, 1, seq_len)
        ).expand(batch_size, 1, seq_len, seq_len)

        return (
            position_ids,
            cos_cache,
            sin_cache,
            past_key_values,
            prior_mask,
            verify_active_mask,
        )

    def _verify_epilogue(
        self,
        hidden_states,
        next_decoder_cache,
        layer_candidates,
        position_ids,
        seq_ids,
        tokens=None,
    ):
        """Shared exit work for both verify paths: cache commit, final norm, lm_head, argmax.

        ``tokens`` short-circuits the head: the megakernel already fused the final norm, vocab
        matmul and greedy argmax, and ``hidden_states`` is pre-final-norm either way.
        """
        updated_kv = self.kv_mgr.update_cache(
            is_for_context_encoding=False,
            seq_ids=seq_ids,
            position_ids=position_ids,
            new_key_values=next_decoder_cache,
            seq_len=self.n_positions,
        )

        verify_trunk_hidden = hidden_states  # pre-final-norm, both positions
        if tokens is None:
            logits = self.lm_head(self.norm(hidden_states)).float()
            tokens = _greedy_argmax(  # [B, spec_len]
                self.lm_head,
                logits,
                self.rank_util,
                disable_argmax_kernel=self.neuron_config.disable_argmax_kernel,
            )

        self._verify_candidates = layer_candidates
        return [tokens, *updated_kv, verify_trunk_hidden]

    def _verify_forward(
        self, input_ids, attention_mask, position_ids, seq_ids, sampling_params
    ):
        if getattr(self.config, "use_verify_megakernel", False):
            return self._verify_forward_megakernel(
                input_ids, attention_mask, position_ids, seq_ids, sampling_params
            )
        seq_ids = seq_ids.to(torch.int32)
        batch_size, seq_len = input_ids.shape[:2]
        (
            position_ids,
            cos_cache,
            sin_cache,
            past_key_values,
            prior_mask,
            verify_active_mask,
        ) = self._verify_prologue(
            batch_size, seq_len, position_ids, seq_ids, input_ids.device
        )
        hidden_states = self.embed_tokens(input_ids)

        next_decoder_cache = ()
        layer_candidates = []
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            layer_outputs = decoder_layer(
                hidden_states,
                seq_ids=seq_ids,
                attention_mask=prior_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=verify_active_mask,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                verify_mode=True,
            )
            hidden_states = layer_outputs[0]
            next_decoder_cache += (layer_outputs[1],)
            cos_cache, sin_cache = layer_outputs[2:4]
            if layer_outputs[5] is not None:  # (S_stack, conv_cand) for DeltaNet
                layer_candidates.append(layer_outputs[5])

        return self._verify_epilogue(
            hidden_states, next_decoder_cache, layer_candidates, position_ids, seq_ids
        )

    def _collect_verify_weights(
        self,
        past_key_values,
        seq_ids,
        prior_mask,
        verify_active_mask,
        batch_size,
        seq_len,
        device,
    ):
        """Per-field weight lists for the trunk, keyed by the megakernels' own field names.

        Shared by the standalone verify launch and the fused round, so the flat positional order
        stays owned by the flatten_* helpers and the two paths cannot pick up different tensors.
        Returns ``(layer_is_gqa, dn, gqa, moe, key_dim, gqa_mask)``.
        """
        H = self.config.hidden_size
        E = self.config.num_experts
        i_rank = (
            self.config.moe_intermediate_size // self.config.neuron_config.tp_degree
        )
        layer_is_gqa = [l.layer_type != "linear_attention" for l in self.layers]

        dn = {f: [] for f in DN_FIELDS}
        gqa = {f: [] for f in GQA_FIELDS}
        moe = {f: [] for f in MOE_FIELDS}
        key_dim = None
        gqa_mask = None

        for idx, layer in enumerate(self.layers):
            mlp = layer.mlp
            moe["gamma"].append(layer.post_attention_layernorm.weight.reshape(1, H))
            moe["router_w"].append(mlp.moe_router_w_t)
            moe["gate_up_w"].append(
                mlp.moe.expert_mlps.mlp_op.gate_up_proj.weight.reshape(E, H, 2, i_rank)
            )
            moe["down_w"].append(mlp.moe.expert_mlps.mlp_op.down_proj.weight)
            moe["sigma_gate_w"].append(mlp.moe_sigma_gate_t)
            moe["shared_gate_w"].append(mlp.shared_expert.gate_proj.weight)
            moe["shared_up_w"].append(mlp.shared_expert.up_proj.weight)
            moe["shared_down_w"].append(mlp.shared_expert.down_proj.weight)

            if layer_is_gqa[idx]:
                sa = layer.self_attn
                k_cache, v_cache = past_key_values[idx]
                gqa["qkv_w"].append(sa.gqa_qkv_w.weight)
                gqa["gate_w"].append(sa.gqa_gate_w.weight)
                gqa["gamma_q"].append(sa.q_layernorm.weight)
                gqa["gamma_k"].append(sa.k_layernorm.weight)
                gqa["in_gamma"].append(layer.input_layernorm.weight.reshape(1, H))
                gqa["o_proj_w"].append(sa.gqa_o_proj_w.weight)
                gqa["k_cache"].append(k_cache)
                gqa["v_cache"].append(v_cache)
                if gqa_mask is None:
                    gqa_mask = sa._build_kernel_mask(
                        prior_mask,
                        verify_active_mask,
                        batch_size,
                        seq_len,
                        k_cache.shape[-1],
                        device,
                    )
            else:
                la = layer.linear_attn
                if key_dim is None:
                    key_dim = la.key_dim
                dn["proj_w"].append(la.in_proj_fused.weight)
                dn["in_gamma"].append(layer.input_layernorm.weight.reshape(1, H))
                dn["conv_state"].append(
                    torch.index_select(la.conv_state_buffer, 0, seq_ids)[0]
                )
                dn["conv_weight"].append(la.conv1d_weight.weight)
                dn["A_log"].append(la._A_log())
                dn["dt_bias"].append(la._dt_bias())
                dn["init_state"].append(
                    torch.index_select(la.recurrent_state_buffer, 0, seq_ids)[0]
                )
                dn["out_w"].append(la.out_proj.weight)
                dn["z_gamma"].append(la.norm.weight)

        return layer_is_gqa, dn, gqa, moe, key_dim, gqa_mask

    def _verify_forward_megakernel(
        self, input_ids, attention_mask, position_ids, seq_ids, sampling_params
    ):
        seq_ids = seq_ids.to(torch.int32)
        batch_size, seq_len = input_ids.shape[:2]
        (
            position_ids,
            cos_cache,
            sin_cache,
            past_key_values,
            prior_mask,
            verify_active_mask,
        ) = self._verify_prologue(
            batch_size, seq_len, position_ids, seq_ids, input_ids.device
        )

        H = self.config.hidden_size
        eps = self.config.rms_norm_eps
        layer_is_gqa, dn, gqa, moe, key_dim, gqa_mask = self._collect_verify_weights(
            past_key_values,
            seq_ids,
            prior_mask,
            verify_active_mask,
            batch_size,
            seq_len,
            input_ids.device,
        )

        replica_groups = (list(range(self.config.neuron_config.tp_degree)),)
        megakernel = build_verify_megakernel(layer_is_gqa)
        flat = flatten_megakernel_args(
            self.embed_tokens(input_ids),
            dn,
            gqa,
            moe,
            cos_cache[0],
            sin_cache[0],
            gqa_mask,
            key_dim,
            eps,
            replica_groups,
            self.norm.weight.reshape(1, H),
            self.lm_head.weight,
        )
        n_gqa = sum(layer_is_gqa)
        n_dn = len(layer_is_gqa) - n_gqa
        tokens, output, gqa_active_kv, dn_cand = split_megakernel_returns(
            megakernel[2](*flat), n_gqa, n_dn
        )
        hidden_states = output.to(self.config.neuron_config.torch_dtype)

        next_decoder_cache = ()
        layer_candidates = []
        gi, di = 0, 0
        for idx, layer in enumerate(self.layers):
            if layer_is_gqa[idx]:
                next_decoder_cache += ((gqa_active_kv[gi][0], gqa_active_kv[gi][1]),)
                gi += 1
            else:
                dummy_k, dummy_v = layer.linear_attn._dummy_kv(
                    batch_size, seq_len, hidden_states.dtype, hidden_states.device
                )
                next_decoder_cache += ((dummy_k, dummy_v),)
                s_stack, conv_cand = dn_cand[di]
                layer_candidates.append((s_stack.unsqueeze(0), conv_cand.unsqueeze(0)))
                di += 1

        return self._verify_epilogue(
            hidden_states,
            next_decoder_cache,
            layer_candidates,
            position_ids,
            seq_ids,
            tokens=tokens,
        )


class Qwen36FusedSpecModel(NeuronFusedSpecModel):
    """Greedy EAGLE fused speculation with in-graph DeltaNet state commit.

    Overrides the two EAGLE sub-forwards: drop the non-greedy / prefix-caching /
    token-tree branches, and after the greedy accept, select S_{accept}/conv_{accept}
    per DeltaNet layer and ride them back into the live recurrent/conv buffers via
    the aliased state region (Qwen36FusedSpecModelInstance). spec_len=2 == k=1.
    """
    def _commit_deltanet(self, candidates, index):
        """Select S/conv by accept count (spec_len=2: 0 -> S1, 1 -> S2).

        Returns flat [rec0, conv0, rec1, conv1, ...] in target ``_deltanet_state_params``
        order, cast to the buffer dtype. In-graph torch.where (XLA lowers select
        correctly; never the eager offset-strided ``[:, idx]`` view).
        """
        dtype = self.neuron_config.torch_dtype
        sel_s = (index == 1).view(self.batch_size, 1, 1, 1)
        sel_c = (index == 1).view(self.batch_size, 1, 1)
        out = []
        for S_stack, conv_cand in candidates:
            out.append(torch.where(sel_s, S_stack[:, 1], S_stack[:, 0]).to(dtype))
            out.append(torch.where(sel_c, conv_cand[:, 1], conv_cand[:, 0]).to(dtype))
        return out

    def _round_epilogue(
        self, candidate_input_ids, target_tokens, hidden_state, candidates
    ):
        """The greedy accept and the state commit, shared by the 3-launch and 1-launch rounds.

        Returns ``(index, committed, hidden_state)`` -- the accept count, the flat selected
        DeltaNet states and the rolling-buffer hidden gathered at the last accepted position.
        """
        index = (
            (
                (~(candidate_input_ids[:, 1:] == target_tokens[:, :-1])).cumsum(dim=-1)
                < 1
            )
            .sum(dim=-1, keepdim=True, dtype=torch.int32)
            .view(self.batch_size, -1)
        )
        committed = self._commit_deltanet(candidates, index)
        hidx = index.reshape(self.batch_size, -1, 1).expand(
            self.batch_size, 1, self.rolling_buffer_hidden_size
        )
        return index, committed, torch.gather(hidden_state, dim=1, index=hidx)

    def _round_token_gen_forward(self, input_ids, position_ids, seq_ids):
        """The whole round -- draft, verify, replay -- as one NKI launch.

        The three stages read at three different positions, so rope, mask and cache-write index are
        per stage: D1 at t, the verify block at [t+1, t+2], D2 at t+1. D2's mask counts MTP slot t
        as committed prior, which is the slot D1 wrote earlier in the same launch.
        """
        spec_len = self.neuron_config.speculation_length
        assert spec_len == 2, "the round megakernel is derived for spec_len == 2 (k == 1)"
        draft, target = self.draft_model, self.target_model
        cfg = target.config
        H = cfg.hidden_size
        dtype = self.neuron_config.torch_dtype
        device = input_ids.device
        seq_ids = seq_ids.to(torch.int32)
        bsz = input_ids.shape[0]

        prev_hidden = self.hidden_state_rolling_buffer.get_state(seq_ids, position_ids)
        pos = position_ids.view(bsz, 1).long()  # t+1, the first candidate's position
        d1_pos, d2_pos = pos - 1, pos
        v_pos = torch.cat([pos, pos + 1], dim=1)

        mtp_k, mtp_v = draft.kv_mgr.get_cache(
            seq_ids=seq_ids, seq_len=draft.n_positions, is_for_context_encoding=False
        )[0]
        mtp_sa = draft.mtp_head.decoder_layer.self_attn
        mtp_L = mtp_k.shape[-1]
        stage = []
        for p, q in ((d1_pos, 1), (d2_pos, 1)):
            cos, sin = draft.mrope_emb(draft.embed_tokens.weight, p)
            stage.append(
                {
                    "kv_write_idx": p.to(torch.int32),
                    "cos": cos[0],
                    "sin": sin[0],
                    "mask": _decode_kernel_mask(
                        mtp_sa, p, bsz, q, mtp_L, draft.n_positions, device
                    ),
                }
            )
        front, mtp_gqa, mtp_moe, mtp_head_w = draft._collect_mtp_weights(mtp_k, mtp_v)

        (
            v_position_ids,
            cos_cache,
            sin_cache,
            past_key_values,
            prior_mask,
            verify_active_mask,
        ) = target._verify_prologue(bsz, spec_len, v_pos, seq_ids, device)
        layer_is_gqa, dn, gqa, moe, key_dim, gqa_mask = target._collect_verify_weights(
            past_key_values,
            seq_ids,
            prior_mask,
            verify_active_mask,
            bsz,
            spec_len,
            device,
        )

        replica_groups = (list(range(cfg.neuron_config.tp_degree)),)
        flat = flatten_round_args(
            input_ids.to(torch.int32),
            prev_hidden,
            front,
            mtp_gqa,
            mtp_moe,
            mtp_head_w,
            stage[0],
            stage[1],
            dn,
            gqa,
            moe,
            cos_cache[0],
            sin_cache[0],
            gqa_mask,
            pos.to(torch.int32),
            target.norm.weight.reshape(1, H),
            target.lm_head.weight,
            key_dim,
            cfg.rms_norm_eps,
            replica_groups,
        )
        n_gqa = sum(layer_is_gqa)
        n_dn = len(layer_is_gqa) - n_gqa
        cand_token, target_tokens, hidden, gqa_kv, dn_cand, mtp_kv = split_round_returns(
            build_round_megakernel(layer_is_gqa)[2](*flat), n_gqa, n_dn
        )
        hidden_state = hidden.to(dtype)

        # The trunk wrote its own GQA K/V in place, so those cache handles ARE the update. A
        # DeltaNet layer's kv_mgr slots are placeholders nothing reads -- its live state rides the
        # aliased state region -- but each layer still owes the graph its two cache outputs.
        target_cache = []
        gi = 0
        for idx, layer in enumerate(target.layers):
            if layer_is_gqa[idx]:
                target_cache += list(gqa_kv[gi])
                gi += 1
            else:
                target_cache += list(
                    target.kv_mgr.update_kv_by_layer_id(
                        idx=idx,
                        is_for_context_encoding=False,
                        seq_ids=seq_ids,
                        position_ids=v_position_ids,
                        kv_per_layer=layer.linear_attn._dummy_kv(
                            bsz, spec_len, dtype, device
                        ),
                        seq_len=target.n_positions,
                    )
                )
        candidates = [(s.unsqueeze(0), c.unsqueeze(0)) for s, c in dn_cand]

        candidate_input_ids = torch.cat(
            (input_ids, cand_token.view(bsz, -1)), dim=-1
        )
        _, committed, hidden_state = self._round_epilogue(
            candidate_input_ids, target_tokens, hidden_state, candidates
        )
        return (
            [candidate_input_ids, target_tokens]
            + list(mtp_kv)
            + target_cache
            + committed
            + [hidden_state]
        )

    def _eagle_token_gen_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
    ):
        if getattr(self.config, "use_speculation_megakernel", False):
            return self._round_token_gen_forward(input_ids, position_ids, seq_ids)

        spec_len = self.neuron_config.speculation_length
        bs = input_ids.shape[0]
        hidden_state = self.hidden_state_rolling_buffer.get_state(seq_ids, position_ids)

        draft_position_ids = position_ids.expand(bs, spec_len) - 1
        candidate_input_ids = input_ids
        target_position_ids = position_ids
        draft_attention_mask = copy.deepcopy(attention_mask)
        scatter_index = torch.sum(draft_attention_mask, dim=1, keepdim=True) - 1
        zeros = torch.zeros(
            scatter_index.shape,
            dtype=draft_attention_mask.dtype,
            device=draft_attention_mask.device,
        )
        draft_attention_mask = torch.scatter(
            draft_attention_mask, 1, scatter_index, zeros
        )
        orig_hidden = hidden_state

        # 1. Draft spec_len-1 candidate tokens.
        for i in range(spec_len - 1):
            draft_position_id = draft_position_ids[:, i : i + 1] + i
            draft_input_ids = candidate_input_ids[:, -1:]
            target_position_id = draft_position_ids[:, i : i + 1] + i + 2
            target_position_ids = torch.cat(
                [target_position_ids, target_position_id], dim=1
            )
            model_output = self.draft_model(
                draft_input_ids,
                draft_attention_mask,
                draft_position_id,
                seq_ids,
                sampling_params,
                prev_hidden=hidden_state,
            )
            hidden_state = model_output[-1]
            ones = torch.ones(
                draft_position_id.shape,
                dtype=draft_attention_mask.dtype,
                device=draft_attention_mask.device,
            )
            draft_attention_mask = torch.scatter(
                draft_attention_mask, 1, draft_position_id, ones
            )
            candidate_input_ids = torch.cat(
                (candidate_input_ids, model_output[0].view(bs, -1)), dim=-1
            )

        # 2. Verify the spec block with the target (DeltaNet read-only seed).
        outputs = self.target_model(
            candidate_input_ids,
            attention_mask,
            target_position_ids,
            seq_ids,
            sampling_params,
        )
        target_tokens = outputs[0]
        target_cache = list(outputs[1:-1])
        hidden_state = outputs[-1]
        candidates = self.target_model._verify_candidates

        prev_hidden = torch.cat(
            [orig_hidden, hidden_state[:, : spec_len - 1, :]], dim=1
        )

        # 3. Final draft run to repopulate the draft KV with the verified block.
        model_output = self.draft_model(
            candidate_input_ids,
            attention_mask,
            target_position_ids - 1,
            seq_ids,
            sampling_params,
            prev_hidden=prev_hidden,
        )
        flat_draft_cache = list(model_output[1:-1])

        # Greedy accept count - 1 (0 reject -> S1, 1 accept -> S2).
        _, committed, hidden_state = self._round_epilogue(
            candidate_input_ids, target_tokens, hidden_state, candidates
        )

        # Layout: [cand, target_tokens, *draft_kv, *target_kv, *committed, hidden].
        # token_gen_outs[-1] (hidden) is consumed by the parent forward's rolling
        # buffer set_state; committed states ride the aliased state region.
        return (
            [candidate_input_ids, target_tokens]
            + flat_draft_cache
            + target_cache
            + committed
            + [hidden_state]
        )

    def _eagle_context_encoding_forward(
        self, input_ids, attention_mask, position_ids, seq_ids, sampling_params
    ):
        self.draft_model.n_positions = self.n_positions
        self.target_model.n_positions = self.n_positions

        # Target prefill writes live DeltaNet + GQA state (exact; no commit needed).
        target_outputs = self.target_model(
            input_ids, attention_mask, position_ids, seq_ids, sampling_params
        )
        target_res = target_outputs[0]
        nt = len(self.target_model.kv_mgr.past_key_values)
        target_cache = list(target_outputs[1 : 1 + nt])
        deltanet_states = list(target_outputs[1 + nt :])
        full_hidden = self.target_model._captured_prenorm  # [B, S, H] pre-final-norm

        # Draft runs one position behind the target (BCDE for target ABCDE).
        gather_index = torch.arange(0, input_ids.shape[1], device=input_ids.device) + 1
        gather_index[-1] = 0
        gather_index = gather_index.expand(input_ids.shape)
        draft_input_ids = torch.gather(input_ids, 1, gather_index)
        draft_position_ids = copy.deepcopy(position_ids)
        scatter_index = torch.sum(attention_mask, dim=1, keepdim=True) - 1
        zeros = torch.zeros(
            scatter_index.shape,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        draft_position_ids = torch.scatter(draft_position_ids, 1, scatter_index, zeros)

        draft_outputs = self.draft_model(
            draft_input_ids,
            attention_mask,
            draft_position_ids,
            seq_ids,
            sampling_params,
            full_hidden,
        )
        draft_cache = list(draft_outputs[1:-1])

        # Last valid position's pre-norm hidden seeds the rolling buffer for decode.
        last_idx = (torch.sum(attention_mask, dim=1, keepdim=True) - 1).clamp(min=0)
        last_idx = last_idx.unsqueeze(1).expand(input_ids.shape[0], 1, self.hidden_size)
        last_hidden = torch.gather(full_hidden, 1, last_idx)

        # Same layout as the token-gen path.
        return (
            [draft_outputs[0], target_res]
            + draft_cache
            + target_cache
            + deltanet_states
            + [last_hidden]
        )


class Qwen36FusedSpecModelInstance(DecoderModelInstance):
    """Fused-spec aliasing + the target's DeltaNet recurrent/conv buffers.

    NxDI's DecoderModelInstance.get aliases draft/target KV then the rolling
    buffer at base = num_out*2 + nd + nt. We insert ns committed DeltaNet states
    at [base, base+ns) (matching the graph layout) and shift the rolling-buffer
    hidden to base+ns.
    """
    def get(self, bucket_rank, **kwargs):
        module, aliases = super().get(bucket_rank, **kwargs)

        num_output_from_trace = 1 if not self.neuron_config.output_logits else 2
        if self.neuron_config.tensor_capture_config:
            num_output_from_trace += (
                self.neuron_config.tensor_capture_config.get_offset()
            )
        num_output_from_trace += 1  # fused-spec acceptance output

        nd = len(module.draft_model.kv_mgr.past_key_values)
        nt = len(module.target_model.kv_mgr.past_key_values)
        base = num_output_from_trace * 2 + nd + nt

        state_params = list(module.target_model._deltanet_state_params)
        for i, param in enumerate(state_params):
            aliases[param] = base + i
        if self.neuron_config.enable_eagle_speculation:
            aliases[module.hidden_state_rolling_buffer.hidden_states] = base + len(
                state_params
            )

        return module, aliases


class Qwen36A3BFusedSpecModelWrapper(ModelWrapper):
    """Base ModelWrapper (standard fused-spec input signature) wired to the
    DeltaNet-aware fused instance."""
    def get_model_instance(self):
        return Qwen36FusedSpecModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )


# ============================================================
# Top-Level Model
# ============================================================


class NeuronQwen36A3BForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronQwen36A3BModel

    def get_model_wrapper_cls(self):
        """Return the ModelWrapper: fused-spec instance when speculating, else the
        DeltaNet-aliasing VL wrapper."""
        if self.neuron_config.enable_fused_speculation:
            return Qwen36A3BFusedSpecModelWrapper
        return Qwen36A3BModelWrapper

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load HF model weights.

        The model is a VL model (Qwen3_5ForConditionalGeneration) but we
        only need the text backbone.
        """
        from transformers import AutoModelForCausalLM

        kwargs.setdefault("trust_remote_code", True)
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return Qwen36A3BInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """Strip VL wrapper prefix and convert to NxDI format."""
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("language_model."):
                new_k = k.replace("language_model.", "", 1)
                new_sd[new_k] = v
            elif k.startswith("model.language_model."):
                new_k = k.replace("model.language_model.", "", 1)
                new_sd[new_k] = v
            elif k.startswith("model.visual") or k.startswith("visual"):
                continue  # Skip vision encoder (text-only port)
            elif k.startswith("model."):
                new_sd[k.replace("model.", "", 1)] = v
            else:
                # lm_head.*, mtp.*, and anything else passes through.
                new_sd[k] = v

        return convert_qwen36_a3b_hf_to_neuron_state_dict(new_sd, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        # Fused spec traces our Qwen36FusedSpecModel (the base swaps in the stock
        # NeuronFusedSpecModel before calling this).
        if self.neuron_config.enable_fused_speculation:
            self._model_cls = Qwen36FusedSpecModel
        super().enable_context_encoding()

    def enable_fused_spec(self):
        self._model_cls = Qwen36FusedSpecModel
        super().enable_fused_spec()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def _copy_past_key_values(self, outputs):
        """Override to also copy DeltaNet state buffers on CPU."""
        super()._copy_past_key_values(outputs)
        if getattr(self.config, "use_hybrid_cache_manager", False):
            return

        num_output_from_trace = 1
        if (
            self.neuron_config.output_logits
            and self.neuron_config.on_device_sampling_config
        ):
            num_output_from_trace = 2

        if (
            hasattr(self, "token_generation_model")
            and self.token_generation_model is not None
        ):
            tkg_model = self.token_generation_model.model
            cte_model = self.context_encoding_model.model
        else:
            return

        if tkg_model.kv_mgr is not None:
            num_kv = len(tkg_model.kv_mgr.past_key_values)
        else:
            num_kv = 0

        state_start = num_output_from_trace + num_kv

        tkg_params = getattr(tkg_model, "_deltanet_state_params", [])
        cte_params = getattr(cte_model, "_deltanet_state_params", [])

        if len(tkg_params) > 0 and state_start + len(tkg_params) <= len(outputs):
            for i, (tkg_param, cte_param) in enumerate(zip(tkg_params, cte_params)):
                new_state = outputs[state_start + i]
                tkg_param.data = new_state
                cte_param.data = new_state

    def get_required_kwargs(self):
        """Return extra kwargs for HF generation loop."""
        return ["llava_args"]

    def _get_model_outputs(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden,
        adapter_ids,
        medusa_args,
        llava_args,
        slot_mapping=None,
        block_table=None,
        full_context_lens=None,
        computed_context_lens=None,
        tf_args=None,
    ):
        """Override to pass all 24 positional args explicitly (text+VL path).

        Fused speculation uses NxDI's standard dispatch (the fused graph has the
        plain fused-spec input signature, not the VL one)."""
        if self.neuron_config.enable_fused_speculation:
            # The base unpacks medusa_args/llava_args/tf_args (*args); pass [] not None.
            return super()._get_model_outputs(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
                medusa_args or [],
                llava_args or [],
                slot_mapping,
                block_table,
                full_context_lens,
                computed_context_lens,
                tf_args or [],
            )

        is_prefill = self._is_prefill(position_ids) or (
            getattr(self.config, "use_qwen_hybrid_chunked_prefill", False)
            and input_ids.shape[-1] > 1
        )

        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        if llava_args and len(llava_args) >= 2:
            vision_embeddings = llava_args[0]
            vision_mask = llava_args[1]
            if len(llava_args) >= 3:
                mrope_position_ids = llava_args[2]
            else:
                mrope_position_ids = None
        elif is_prefill:
            vision_embeddings = torch.zeros(
                (batch_size, seq_len, self.config.hidden_size),
                dtype=self.config.neuron_config.torch_dtype,
            )
            vision_mask = torch.full(
                (batch_size, seq_len, 1),
                fill_value=seq_len - 1,
                dtype=torch.int32,
            )
            mrope_position_ids = None
        else:
            vision_embeddings = torch.zeros((0,), dtype=torch.float32)
            vision_mask = torch.zeros((0,), dtype=torch.int32)
            mrope_position_ids = None

        if is_prefill:
            if mrope_position_ids is None:
                mrope_position_ids = (
                    torch.arange(0, seq_len, dtype=torch.int32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(3, batch_size, -1)
                    .contiguous()
                )
        else:
            mrope_position_ids = torch.zeros((0,), dtype=torch.int32)

        empties = [torch.empty(0) for _ in range(14)]

        if is_prefill:
            ctx_bs = self.context_encoding_model.neuron_config.batch_size
            output_logits = []

            for cb in range(0, batch_size, ctx_bs):
                cb_end = min(cb + ctx_bs, batch_size)
                actual_chunk = cb_end - cb

                chunk_input_ids = input_ids[cb:cb_end]
                chunk_attn_mask = attention_mask[cb:cb_end]
                chunk_pos_ids = position_ids[cb:cb_end]
                chunk_seq_ids = seq_ids[cb:cb_end]
                chunk_sampling = sampling_params[cb:cb_end]
                chunk_prev_hidden = (
                    prev_hidden[cb:cb_end]
                    if prev_hidden is not None
                    and hasattr(prev_hidden, "ndim")
                    and prev_hidden.ndim > 0
                    and prev_hidden.shape[0] > 0
                    else prev_hidden
                )
                chunk_adapter_ids = (
                    adapter_ids[cb:cb_end]
                    if adapter_ids is not None
                    and hasattr(adapter_ids, "ndim")
                    and adapter_ids.ndim > 0
                    and adapter_ids.shape[0] > 0
                    else adapter_ids
                )

                if mrope_position_ids.ndim == 3:
                    chunk_mrope = mrope_position_ids[:, cb:cb_end, :]
                else:
                    chunk_mrope = mrope_position_ids

                if vision_embeddings.ndim == 3:
                    chunk_vis_emb = vision_embeddings[cb:cb_end]
                    chunk_vis_mask = vision_mask[cb:cb_end]
                else:
                    chunk_vis_emb = vision_embeddings
                    chunk_vis_mask = vision_mask

                if actual_chunk < ctx_bs:
                    pad_n = ctx_bs - actual_chunk
                    chunk_input_ids = torch.cat(
                        [chunk_input_ids, chunk_input_ids[:1].expand(pad_n, -1)], dim=0
                    )
                    chunk_attn_mask = torch.cat(
                        [chunk_attn_mask, chunk_attn_mask[:1].expand(pad_n, -1)], dim=0
                    )
                    chunk_pos_ids = torch.cat(
                        [chunk_pos_ids, chunk_pos_ids[:1].expand(pad_n, -1)], dim=0
                    )
                    pad_seq = torch.arange(
                        batch_size, batch_size + pad_n, dtype=chunk_seq_ids.dtype
                    )
                    chunk_seq_ids = torch.cat([chunk_seq_ids, pad_seq], dim=0)
                    chunk_sampling = torch.cat(
                        [chunk_sampling, chunk_sampling[:1].expand(pad_n, -1)], dim=0
                    )
                    if (
                        chunk_prev_hidden is not None
                        and hasattr(chunk_prev_hidden, "ndim")
                        and chunk_prev_hidden.ndim > 0
                        and chunk_prev_hidden.shape[0] > 0
                    ):
                        chunk_prev_hidden = torch.cat(
                            [
                                chunk_prev_hidden,
                                chunk_prev_hidden[:1].expand(pad_n, -1),
                            ],
                            dim=0,
                        )
                    if (
                        chunk_adapter_ids is not None
                        and hasattr(chunk_adapter_ids, "ndim")
                        and chunk_adapter_ids.ndim > 0
                        and chunk_adapter_ids.shape[0] > 0
                    ):
                        chunk_adapter_ids = torch.cat(
                            [
                                chunk_adapter_ids,
                                chunk_adapter_ids[:1].expand(pad_n, -1),
                            ],
                            dim=0,
                        )
                    if chunk_mrope.ndim == 3:
                        chunk_mrope = torch.cat(
                            [chunk_mrope, chunk_mrope[:, :1, :].expand(-1, pad_n, -1)],
                            dim=1,
                        )
                    if chunk_vis_emb.ndim == 3:
                        chunk_vis_emb = torch.cat(
                            [
                                chunk_vis_emb,
                                torch.zeros(
                                    (pad_n,) + chunk_vis_emb.shape[1:],
                                    dtype=chunk_vis_emb.dtype,
                                ),
                            ],
                            dim=0,
                        )
                        chunk_vis_mask = torch.cat(
                            [
                                chunk_vis_mask,
                                torch.full(
                                    (pad_n,) + chunk_vis_mask.shape[1:],
                                    fill_value=seq_len - 1,
                                    dtype=chunk_vis_mask.dtype,
                                ),
                            ],
                            dim=0,
                        )

                chunk_out = self.context_encoding_model(
                    chunk_input_ids,
                    chunk_attn_mask,
                    chunk_pos_ids,
                    chunk_seq_ids,
                    chunk_sampling,
                    chunk_prev_hidden,
                    chunk_adapter_ids,
                    *empties,
                    chunk_mrope,
                    chunk_vis_emb,
                    chunk_vis_mask,
                )
                if actual_chunk < ctx_bs:
                    chunk_out = chunk_out[:actual_chunk]
                output_logits.append(chunk_out)

            outputs = (
                torch.cat(output_logits, dim=0)
                if len(output_logits) > 1
                else output_logits[0]
            )
            self.kv_cache_populated = True
            is_run_on_neuron = self.context_encoding_model.is_neuron()
        else:
            outputs = self.token_generation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
                *empties,
                mrope_position_ids,
                vision_embeddings,
                vision_mask,
            )
            is_run_on_neuron = self.token_generation_model.is_neuron()

        return outputs, is_run_on_neuron

    def get_compiler_args(self):
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        else:
            optimization_level = "-O1"

        compiler_args = (
            "--enable-saturate-infinity "
            "--enable-mixed-precision-accumulation "
            f"--model-type transformer {optimization_level} "
            "--auto-cast=none "
        )
        return compiler_args


class NeuronQwen36MTPDraftForCausalLM(NeuronQwen36A3BForCausalLM):
    """Draft holder for fused speculation: exposes Qwen36MTPDraft as _model_cls and
    a converter that keeps only the embedding + MTP head (the backbone lives in the
    target). Self-speculative: loads the same checkpoint as the target."""
    _model_cls = Qwen36MTPDraft

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        sd = NeuronQwen36A3BForCausalLM.convert_hf_to_neuron_state_dict(
            state_dict, config
        )
        # Keep only the draft's params: embedding, MTP head (incl. its layer's own
        # rank_util), and the model-level rank_util. Drop the backbone layers/norm/
        # lm_head (the target owns those).
        return {
            k: v
            for k, v in sd.items()
            if k.startswith("embed_tokens.")
            or k.startswith("mtp_head.")
            or k == "rank_util.rank"
        }
