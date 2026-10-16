"""Alpamayo 1.5 inference on Jetson AGX Thor (SM110), BF16.

The engine wraps one official ``alpamayo1_5.models.alpamayo1_5.Alpamayo1_5``
instance (its weights are shared, never copied) and executes the request path
of the official ``sample_trajectories_from_data_with_vlm_rollout`` with
static buffers and CUDA graphs:

1. vision + prefill   batched vision tower and a 36-layer prefill on fused
                      kernels that write K/V straight into a static cache. The
                      prompt is encoded once and its K/V shared by every
                      trajectory sample (the official path repeats the whole
                      prompt, images included, per sample).
2. reasoning decode   one CUDA graph replayed per token for all samples: the
                      decode step over the static cache plus the Hugging Face
                      sampling chain (trajectory-token mask -> temperature ->
                      top-p -> softmax -> multinomial, finished rows padded).
                      The host only reads the sampled tokens to evaluate the
                      official stop criteria.
3. diffusion expert   every Euler step of the flow-matching expert in one CUDA
                      graph. Expert attention reads each sample's cache with a
                      device-side length, so one graph serves every reasoning
                      length.
4. trajectory         official unicycle integration.

Static cache layout per layer and sample: ``[prompt | generated | action]``.
During diffusion each sample's 64 action-token K/V rows are written right after
its visible prefix (later rows are masked for that sample anyway), so for one
sample expert attention is a single non-causal pass over a contiguous prefix.
With several samples the prompt K/V lives once (sample 0) and attention is a
shared-prompt pass for all samples plus a per-sample suffix pass, merged by
their log-sum-exp.

Numerics follow the official module graph under ``torch.autocast(bfloat16)``:
the VLM runs in BF16; the expert keeps an FP32 residual stream, applies RoPE in
FP32 and feeds BF16 operands to attention (autocast casts SDPA inputs).

``decode_weights="fp8"`` is an opt-in precision tier, not part of that
contract: the reasoning-decode projections (q/k/v, o, gate/up, down and the LM
head) read per-row FP8 E4M3 copies of their weights (BF16 activations, FP32
accumulation). Decode is weight-streaming bound, so this halves its bytes;
vision, prefill, the expert and the sampler keep BF16 arithmetic. The expert
still attends over the reasoning-token K/V written by the FP8 projections, so
trajectories can move even when the sampled tokens are identical.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import LogitsProcessorList
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)

BF16 = torch.bfloat16
DEV = "cuda"
logger = logging.getLogger(__name__)

ALPAMAYO15_KERNEL_SYMBOLS = (
    "alpamayo15_expert_qk_prep_bf16",
    "alpamayo15_res_rms_f32_bf16",
    "alpamayo15_rms_bf16",
    "alpamayo15_vision_rope_qk_bf16",
    "alpamayo15_vlm_qk_prep_bf16",
    "alpamayo15_w8a16_gemm_bf16",
)


def require_alpamayo15_kernels(fvk: object) -> None:
    """Fail fast (before any model setup) when the Thor kernel group is missing."""
    missing = [name for name in ALPAMAYO15_KERNEL_SYMBOLS if not hasattr(fvk, name)]
    if missing:
        raise RuntimeError(
            "FlashRT was built without the Alpamayo 1.5 kernels "
            f"({', '.join(missing)}). Rebuild with -DGPU_ARCH=110 -DFLASHRT_ENABLE_ALPAMAYO15=ON.")


def update_stop_state(last: list[int], traj_found: list[bool], unfinished: list[bool], n_generated: int, *,
                      traj_start_id: int, max_new_tokens: int, eos_ids: set[int]) -> None:
    """Official stop criteria after appending ``last`` (one token per sample).

    Mirrors ``generate``: the token budget and the generation-config EOS finish
    individual rows, and ``StopAfterEOS`` finishes every row one token after all
    rows have produced ``<|traj_future_start|>`` (so its K/V reaches the cache).
    ``traj_found`` and ``unfinished`` are updated in place.
    """
    stop_all = all(traj_found)
    for b, t in enumerate(last):
        traj_found[b] = traj_found[b] or t == traj_start_id
        if stop_all or n_generated >= max_new_tokens or t in eos_ids:
            unfinished[b] = False


def expert_attention_spans(tokens: list[list[int]], prompt_len: int, traj_start_id: int) -> tuple[list[int], list[int]]:
    """Per-sample ``(offset, visible)`` for the expert, as the official rollout derives them.

    ``offset`` is one past the first ``<|traj_future_start|>`` (``_find_eos_offset``; the
    sequence length when absent) and anchors the action-token positions. The expert
    sees cache rows ``[0, min(offset, cache_len))`` plus its own action tokens, where
    ``cache_len = prompt_len + n_generated - 1`` (the last sampled token is never
    forwarded); ``_build_expert_pos_ids_and_attn_mask`` masks the rest.
    """
    cache_len = prompt_len + len(tokens[0]) - 1
    offsets = []
    for row in tokens:
        if traj_start_id in row:
            offsets.append(prompt_len + row.index(traj_start_id) + 1)
        else:
            logger.warning("No <|traj_future_start|> in %d generated tokens", len(row))
            offsets.append(prompt_len + len(row))
    return offsets, [min(o, cache_len) for o in offsets]


def merge_attention(o1: torch.Tensor, lse1: torch.Tensor, o2: torch.Tensor, lse2: torch.Tensor) -> torch.Tensor:
    """Combine two attention passes over disjoint key sets by their log-sum-exp.

    ``o*`` are [rows, heads, D], ``lse*`` [heads, rows] (natural log). Partial
    outputs arrive in BF16, so this adds one rounding relative to a single pass.
    """
    m = torch.maximum(lse1, lse2)
    w1, w2 = torch.exp(lse1 - m), torch.exp(lse2 - m)
    den = w1 + w2
    w1, w2 = (w1 / den).t().unsqueeze(-1), (w2 / den).t().unsqueeze(-1)
    return (o1.float() * w1 + o2.float() * w2).to(BF16)


FP8_E4M3_MAX = 448.0


@torch.no_grad()
def quantize_fp8_rows(w: torch.Tensor, chunk_rows: int = 16384) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row FP8 E4M3 quantization of a [N, K] weight.

    ``scale = absmax(row) / 448`` (1 for an all-zero row) and the codes are
    ``round_to_nearest_even(w / scale)`` as ``torch.float8_e4m3fn``. Returns
    (codes as uint8 [N, K], scale fp32 [N]); rows are processed in chunks to
    bound the FP32 temporary. Runs without autograd: model parameters usually
    require grad, and a recorded graph would keep every FP32 temporary alive.
    """
    N, K = w.shape
    codes = torch.empty(N, K, device=w.device, dtype=torch.uint8)
    scale = torch.empty(N, device=w.device, dtype=torch.float32)
    for r0 in range(0, N, chunk_rows):
        wf = w[r0:r0 + chunk_rows].float()
        amax = wf.abs().amax(dim=1)
        s = torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax))
        q = (wf / s[:, None]).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        codes[r0:r0 + chunk_rows] = q.view(torch.uint8)
        scale[r0:r0 + chunk_rows] = s
    return codes, scale


@torch.no_grad()
def fp8_fragment_layout(codes: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder [N, K] e4m3 codes into the ``alpamayo15_w8a16_gemm_bf16`` fragment order.

    Rows are zero-padded to a multiple of 16 (scale 0); per 16-row tile and
    32-column chunk, lane ``4g + c`` holds the 16 bytes of its two
    ``mma.m16n8k16`` A fragments (see ``alpamayo15_kernels.cuh``).
    """
    N, K = codes.shape
    if K % 32:
        raise ValueError(f"K={K} must be a multiple of 32")
    T = (N + 15) // 16
    if T * 16 != N:
        codes = torch.cat([codes, codes.new_zeros(T * 16 - N, K)])
        scale = torch.cat([scale, scale.new_zeros(T * 16 - N)])
    # [t, rh, g, kc, kt, kh, c, e] -> [t, kc, g, c, kt, kh, rh, e]
    frag = codes.view(T, 2, 8, K // 32, 2, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).contiguous()
    return frag.view(-1), scale.contiguous()


_SUPPORTED_SAMPLING = {"repetition_penalty": 1.0, "typical_p": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
                       "no_repeat_ngram_size": 0, "encoder_repetition_penalty": 1.0, "renormalize_logits": False}


def check_generation_config(gc) -> None:
    """The decode graph implements mask -> temperature -> top-p; reject configs that add other HF processors.

    Fields the official rollout sets on every call (top_p, temperature, top_k,
    do_sample, max_new_tokens, ...) are not inspected.
    """
    bad = [k for k, v in _SUPPORTED_SAMPLING.items() if getattr(gc, k, v) not in (v, None)]
    bad += [k for k in ("min_p", "bad_words_ids", "suppress_tokens", "begin_suppress_tokens",
                        "sequence_bias", "forced_bos_token_id", "forced_eos_token_id", "exponential_decay_length_penalty",
                        "guidance_scale", "watermarking_config") if getattr(gc, k, None)]
    if getattr(gc, "min_new_tokens", None) or getattr(gc, "min_length", 0):
        bad.append("min_new_tokens/min_length")
    if bad:
        raise ValueError(f"Alpamayo15Thor does not implement these generation settings: {sorted(bad)}")


def _autocast():
    return torch.autocast("cuda", dtype=BF16)


class Alpamayo15Thor:
    """Static-buffer CUDA-graph engine for one official Alpamayo 1.5 model.

    Args:
        model: official ``Alpamayo1_5`` on CUDA in BF16, eval mode.
        max_prompt: largest supported prompt length in tokens.
        max_new_tokens: generation budget (official ``max_generation_length``).
        max_samples: largest ``num_traj_samples`` served (cache is sized for it).
        top_p, temperature: sampling parameters, baked into the decode graphs.
        decode_splits, expert_splits: FA4 KV splits for decode and expert attention.
        vision_attention: ``"fa4"`` (strided views, no copies), ``"cudnn"`` or ``"sdpa"``.
        prefill: ``"native"`` (fused FlashRT kernels writing the static cache)
            or ``"official"`` (official Qwen3-VL forward; reference path).
        decode_weights: ``"bf16"`` (official weights) or ``"fp8"`` (opt-in
            per-row FP8 E4M3 copies for the reasoning-decode projections and the
            LM head; about +7.6 GB, changes decode numerics - see module notes).
    """

    def __init__(
        self,
        model,
        *,
        max_prompt: int = 4096,
        max_new_tokens: int = 256,
        max_samples: int = 1,
        top_p: float = 0.98,
        temperature: float = 0.6,
        decode_splits: int = 4,
        expert_splits: int = 1,
        vision_attention: str = "fa4",
        prefill: str = "native",
        decode_weights: str = "bf16",
    ) -> None:
        if prefill not in ("native", "official"):
            raise ValueError(f"unknown prefill {prefill!r}")
        if decode_weights not in ("bf16", "fp8"):
            raise ValueError(f"unknown decode_weights {decode_weights!r}")
        if decode_weights == "fp8" and max_samples > 16:
            raise ValueError("decode_weights='fp8' supports max_samples <= 16 (the W8A16 kernel serves up to 16 rows)")
        if vision_attention not in ("fa4", "cudnn", "sdpa"):
            raise ValueError(f"unknown vision_attention {vision_attention!r}")
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        import flash_rt.flash_rt_kernels as fvk
        require_alpamayo15_kernels(fvk)
        if model.training:
            raise ValueError("Alpamayo15Thor requires an eval-mode model")
        from flash_rt.hardware.thor import fa4_backend
        self._fa4 = fa4_backend.fa4_fwd()
        if self._fa4 is None:
            raise RuntimeError(f"FA4 unavailable: {fa4_backend.status()}")
        self._fvk = fvk
        self.m = model
        self.vlm = model.vlm
        self.lm = model.vlm.model.language_model
        self.expert = model.expert
        cfg = model.vlm.config.text_config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.hidden = cfg.hidden_size
        self.ffn = cfg.intermediate_size
        self.eps = cfg.rms_norm_eps
        ecfg = model.expert.config
        if (ecfg.num_hidden_layers != self.n_layers or ecfg.num_key_value_heads != self.n_kv
                or ecfg.head_dim != self.head_dim):
            raise ValueError("expert attention geometry must match the VLM cache")
        if self.head_dim != 128:
            raise ValueError(f"Alpamayo 1.5 kernels require head_dim 128, got {self.head_dim}")
        check_generation_config(model.vlm.generation_config)
        self.e_heads = ecfg.num_attention_heads
        self.e_hidden = ecfg.hidden_size
        self.e_eps = ecfg.rms_norm_eps
        self.n_act, self.act_dim = model.action_space.get_action_space_dims()
        self.max_prompt = int(max_prompt)
        self.max_new = int(max_new_tokens)
        self.max_samples = int(max_samples)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.decode_splits = int(decode_splits)
        self.expert_splits = int(expert_splits)
        self.vision_attention = vision_attention
        self.prefill_impl = prefill
        self.decode_weights = decode_weights
        self.max_seq = self.max_prompt + self.max_new + self.n_act
        self.scale = self.head_dim ** -0.5

        tok = model.tokenizer
        self.traj_start_id = tok.convert_tokens_to_ids("<|traj_future_start|>")
        gen_eos = model.vlm.generation_config.eos_token_id
        self.eos_ids = set([gen_eos] if isinstance(gen_eos, int) else (gen_eos or []))
        self.pad_id = tok.pad_token_id

        B, L, S, KV, D = self.max_samples, self.n_layers, self.max_seq, self.n_kv, self.head_dim
        self.k_cache = torch.zeros(L, B, S, KV, D, device=DEV, dtype=BF16)
        self.v_cache = torch.zeros(L, B, S, KV, D, device=DEV, dtype=BF16)

        # Decode state: device scalars advanced inside the graph (all samples
        # sit at the same cache position), per-sample token / length / unfinished.
        self.d_tok = torch.zeros(B, dtype=torch.long, device=DEV)
        self.d_unfinished = torch.ones(B, dtype=torch.long, device=DEV)
        self.d_slot = torch.zeros(1, dtype=torch.long, device=DEV)     # cache row to write
        self.d_len = torch.zeros(B, dtype=torch.int32, device=DEV)     # attended prompt+generated rows
        self.d_glen = torch.zeros(B, dtype=torch.int32, device=DEV)    # attended generated rows
        self.d_plen = torch.zeros(1, dtype=torch.int32, device=DEV)    # prompt rows (sample 0)
        self.d_cuk = torch.zeros(B + 1, dtype=torch.int32, device=DEV) # per-sample suffix starts
        self.d_rpos = torch.zeros(1, dtype=torch.long, device=DEV)     # rope position
        # Rope tables via the official rotary module (same fp32 math, BF16 cast).
        max_rope = self.max_prompt + self.max_new + 8
        pos = torch.arange(max_rope, device=DEV).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = self.lm.rotary_emb(torch.zeros(1, device=DEV, dtype=BF16), pos)
        self.cos_table = cos[0].contiguous()   # [max_rope, D] bf16
        self.sin_table = sin[0].contiguous()
        self.logits_processor = LogitsProcessorList([
            _TrajTokenMask(model.config.traj_token_start_idx, model.config.traj_vocab_size),
            TemperatureLogitsWarper(self.temperature),
            TopPLogitsWarper(top_p=self.top_p, min_tokens_to_keep=1),
        ])
        self._decode_graphs: dict[int, torch.cuda.CUDAGraph] = {}

        # Expert state (per sample).
        self.x = torch.zeros(B, self.n_act, self.act_dim, device=DEV, dtype=torch.float32)
        self.e_cos = torch.zeros(B, self.n_act, self.head_dim, device=DEV, dtype=torch.float32)
        self.e_sin = torch.zeros(B, self.n_act, self.head_dim, device=DEV, dtype=torch.float32)
        self.d_eseq = torch.zeros(B, dtype=torch.int32, device=DEV)    # single-sample expert key rows
        self.d_esuf = torch.zeros(B, dtype=torch.int32, device=DEV)    # per-sample suffix key rows
        self.d_aslot = torch.zeros(B, dtype=torch.long, device=DEV)    # action-row start per sample
        self._cuq = {}
        self.n_steps = model.diffusion.num_inference_steps
        self.time_steps = torch.linspace(0.0, 1.0, self.n_steps + 1, device=DEV)
        self._expert_graphs: dict[int, torch.cuda.CUDAGraph] = {}

        self._fuse_weights()
        self._q8 = self._quantize_decode_weights() if decode_weights == "fp8" else None
        self._pf = None
        self._db = None
        self._eb = None
        self._vis_tables: dict = {}
        self._mark = None

    def _fuse_weights(self) -> None:
        """Fuse q/k/v (VLM and expert) and VLM gate/up projections into single tensors.

        The official modules keep working: their parameters become row views of
        the fused tensors (identical values, no extra memory). The engine keeps
        references to the fused tensors, so the model must not be moved or have
        these parameters replaced after construction.
        """
        def fuse(layers):
            out = []
            for layer in layers:
                a = layer.self_attn
                fused = torch.cat([a.q_proj.weight.data, a.k_proj.weight.data, a.v_proj.weight.data], 0).contiguous()
                nq, nk = a.q_proj.weight.shape[0], a.k_proj.weight.shape[0]
                a.q_proj.weight.data = fused[:nq]
                a.k_proj.weight.data = fused[nq:nq + nk]
                a.v_proj.weight.data = fused[nq + nk:]
                out.append(fused)
            return out
        self.w_qkv = fuse(self.lm.layers)
        self.e_w_qkv = fuse(self.expert.layers)
        self.w_gu = []
        for layer in self.lm.layers:
            mlp = layer.mlp
            fused = torch.cat([mlp.gate_proj.weight.data, mlp.up_proj.weight.data], 0).contiguous()
            ng = mlp.gate_proj.weight.shape[0]
            mlp.gate_proj.weight.data = fused[:ng]
            mlp.up_proj.weight.data = fused[ng:]
            self.w_gu.append(fused)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _quantize_decode_weights(self) -> dict:
        """FP8 fragment copies of the decode projections (the BF16 originals stay for prefill)."""
        q8 = {"qkv": [], "o": [], "gu": [], "down": []}
        for li, layer in enumerate(self.lm.layers):
            q8["qkv"].append(fp8_fragment_layout(*quantize_fp8_rows(self.w_qkv[li])))
            q8["o"].append(fp8_fragment_layout(*quantize_fp8_rows(layer.self_attn.o_proj.weight)))
            q8["gu"].append(fp8_fragment_layout(*quantize_fp8_rows(self.w_gu[li])))
            q8["down"].append(fp8_fragment_layout(*quantize_fp8_rows(layer.mlp.down_proj.weight)))
        q8["lm_head"] = fp8_fragment_layout(*quantize_fp8_rows(self.vlm.lm_head.weight))
        torch.cuda.empty_cache()
        return q8

    def _proj(self, x: torch.Tensor, w: torch.Tensor, key: str, li: int, out: torch.Tensor, stream: int) -> None:
        """``out = x @ w.T`` from the BF16 weight or, in the FP8 tier, its fragment copy."""
        if self._q8 is None:
            torch.mm(x, w.t(), out=out)
            return
        frag, scale = self._q8[key][li] if li >= 0 else self._q8[key]
        if not (x.is_contiguous() and out.is_contiguous() and x.shape[0] == out.shape[0]):
            raise ValueError("FP8 projection needs contiguous row-major x [n, K] and out [n, N]")
        self._fvk.alpamayo15_w8a16_gemm_bf16(frag.data_ptr(), scale.data_ptr(), x.data_ptr(), out.data_ptr(),
                                             x.shape[0], out.shape[1], x.shape[1], stream)

    # ------------------------------------------------------------------
    # vision
    # ------------------------------------------------------------------
    def _vision_tables(self, grid: torch.Tensor):
        """Per-grid position embeddings and rotary tables (official functions), cached."""
        key = tuple(grid.flatten().tolist())
        tab = self._vis_tables.pop(key, None)
        if tab is None:
            V = self.vlm.model.visual
            with _autocast():
                pos = V.fast_pos_embed_interpolate(grid)
            rot = V.rot_pos_emb(grid)
            emb = torch.cat((rot, rot), dim=-1)
            tab = (pos.to(BF16), emb.cos().float().contiguous(), emb.sin().float().contiguous())
            if len(self._vis_tables) >= 4:
                self._vis_tables.pop(next(iter(self._vis_tables)))
        self._vis_tables[key] = tab        # most recently used last
        return tab, key

    def _vision(self, pixel_values: torch.Tensor, grid: torch.Tensor):
        """Official Qwen3-VL vision tower semantics with batched per-image attention.

        LayerNorm runs on BF16 inputs with FP32 math and one rounding (as the
        autocast FP32 output feeding a BF16 GEMM); rotary is applied in FP32 and
        rounded once, as in the official module.
        """
        V = self.vlm.model.visual
        (pos, cos, sin), key = self._vision_tables(grid)
        pe = V.patch_embed.proj
        x = F.linear(pixel_values.to(BF16), pe.weight.view(pe.weight.shape[0], -1), pe.bias)
        x = x + pos
        S = x.shape[0]
        n_img = len(key) // 3
        t, gh, gw = key[0:3]
        equal = all(key[3 * i:3 * i + 3] == (t, gh, gw) for i in range(n_img))
        nh = V.blocks[0].attn.num_heads
        hd = x.shape[1] // nh
        stream = torch.cuda.current_stream().cuda_stream
        deep = []
        for li, blk in enumerate(V.blocks):
            a = blk.attn
            h = F.layer_norm(x, (x.shape[1],), blk.norm1.weight, blk.norm1.bias, blk.norm1.eps)
            qkv = F.linear(h, a.qkv.weight, a.qkv.bias)
            self._fvk.alpamayo15_vision_rope_qk_bf16(qkv.data_ptr(), cos.data_ptr(), sin.data_ptr(), S, nh, hd, stream)
            q, k, v = qkv.view(S, 3, nh, hd).unbind(1)
            if equal and self.vision_attention == "fa4":
                L = t * gh * gw
                o, _ = self._fa4(*(z.view(n_img, L, nh, hd) for z in (q, k, v)),
                                 softmax_scale=a.scaling, causal=False)
                o = o.reshape(S, -1)
            elif equal:
                L = t * gh * gw
                qb, kb, vb = (z.view(n_img, L, nh, hd).transpose(1, 2) for z in (q, k, v))
                if self.vision_attention == "cudnn":
                    qb, kb, vb = qb.contiguous(), kb.contiguous(), vb.contiguous()
                    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]):
                        o = F.scaled_dot_product_attention(qb, kb, vb, scale=a.scaling)
                else:
                    o = F.scaled_dot_product_attention(qb, kb, vb, scale=a.scaling)
                o = o.transpose(1, 2).reshape(S, -1)
            else:
                outs, off = [], 0
                for i in range(n_img):
                    L = key[3 * i] * key[3 * i + 1] * key[3 * i + 2]
                    qi, ki, vi = (z[off:off + L].transpose(0, 1).unsqueeze(0) for z in (q, k, v))
                    outs.append(F.scaled_dot_product_attention(qi, ki, vi, scale=a.scaling)[0]
                                .transpose(0, 1).reshape(L, -1))
                    off += L
                o = torch.cat(outs, 0)
            x = x + F.linear(o, a.proj.weight, a.proj.bias)
            m = blk.mlp
            h = F.layer_norm(x, (x.shape[1],), blk.norm2.weight, blk.norm2.bias, blk.norm2.eps)
            x = x + F.linear(m.act_fn(F.linear(h, m.linear_fc1.weight, m.linear_fc1.bias)),
                             m.linear_fc2.weight, m.linear_fc2.bias)
            if li in V.deepstack_visual_indexes:
                with _autocast():
                    deep.append(V.deepstack_merger_list[V.deepstack_visual_indexes.index(li)](x).to(BF16))
        with _autocast():
            img = V.merger(x).to(BF16)
        return img, deep

    # ------------------------------------------------------------------
    # prefill (writes sample 0 of the cache)
    # ------------------------------------------------------------------
    def _prefill_buffers(self, P: int) -> dict:
        if self._pf is None or self._pf["P"] < P:
            n = max(P, 1024)
            H, Hq, Hkv, F_ = self.hidden, self.n_heads * self.head_dim, self.n_kv * self.head_dim, self.ffn
            mk = lambda *shape: torch.empty(*shape, device=DEV, dtype=BF16)
            self._pf = {"P": n, "x": mk(n, H), "h": mk(n, H), "qkv": mk(n, Hq + 2 * Hkv), "q": mk(n, Hq),
                        "attn": mk(n, Hq), "o": mk(n, H), "g": mk(n, F_), "u": mk(n, F_), "d": mk(n, H)}
        return self._pf

    def _prefill_official(self, input_ids, tokenized):
        """Reference path: official forward, K/V copied into the static cache."""
        with _autocast():
            out = self.vlm(input_ids=input_ids, use_cache=True, logits_to_keep=1, return_dict=True, **tokenized)
        P = input_ids.shape[1]
        for li, layer in enumerate(out.past_key_values.layers):
            self.k_cache[li, 0, :P].copy_(layer.keys[0].transpose(0, 1))
            self.v_cache[li, 0, :P].copy_(layer.values[0].transpose(0, 1))
        return out.logits, int(self.vlm.model.rope_deltas.view(-1)[0].item())

    def _prefill_native(self, input_ids, tokenized):
        """Vision + 36-layer LM prefill on fused FlashRT kernels.

        K/V are normalized, rotated (interleaved M-RoPE) and written straight
        into the static cache; only the last row is normalized for the logits.
        """
        vm, lm, fvk = self.vlm.model, self.lm, self._fvk
        s = torch.cuda.current_stream().cuda_stream
        P = input_ids.shape[1]
        H, D, NH, NKV, F_ = self.hidden, self.head_dim, self.n_heads, self.n_kv, self.ffn
        b = self._prefill_buffers(P)
        x, h, qkv, q, attn, o, g, u, d = (b[k][:P] for k in ("x", "h", "qkv", "q", "attn", "o", "g", "u", "d"))
        grid = tokenized["image_grid_thw"]
        img, deepstack = self._vision(tokenized["pixel_values"], grid)
        if self._mark is not None:
            self._mark("vision_end")
        vis_idx = (input_ids[0] == self.vlm.config.image_token_id).nonzero().squeeze(1)
        if vis_idx.numel() != img.shape[0]:
            raise ValueError(f"image tokens {vis_idx.numel()} != image features {img.shape[0]}")
        torch.index_select(lm.embed_tokens.weight, 0, input_ids[0], out=x)
        x.index_copy_(0, vis_idx, img)
        position_ids, rope_deltas = vm.get_rope_index(input_ids, grid, None, attention_mask=tokenized.get("attention_mask"))
        vm.rope_deltas = rope_deltas
        cos, sin = lm.rotary_emb(x[:1].unsqueeze(0), position_ids)       # [1, P, D] bf16
        cos_f, sin_f = cos[0].contiguous(), sin[0].contiguous()
        layers = lm.layers
        fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, layers[0].input_layernorm.weight.data_ptr(), h.data_ptr(), P, H, self.eps, s)
        for li, layer in enumerate(layers):
            a = layer.self_attn
            torch.mm(h, self.w_qkv[li].t(), out=qkv)
            kc, vc = self.k_cache[li, 0, :P], self.v_cache[li, 0, :P]
            fvk.alpamayo15_vlm_qk_prep_bf16(
                qkv.data_ptr(), a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(),
                cos_f.data_ptr(), sin_f.data_ptr(), q.data_ptr(), kc.data_ptr(), vc.data_ptr(), 0,
                P, NH, NKV, NKV * D, P, 0, self.eps, s)
            self._fa4(q.view(1, P, NH, D), kc.unsqueeze(0), vc.unsqueeze(0), softmax_scale=self.scale,
                      causal=True, pack_gqa=True, out=attn.view(1, P, NH, D))
            torch.mm(attn, a.o_proj.weight.t(), out=o)
            fvk.alpamayo15_rms_bf16(x.data_ptr(), o.data_ptr(), layer.post_attention_layernorm.weight.data_ptr(),
                                    h.data_ptr(), P, H, self.eps, s)
            mlp = layer.mlp
            torch.mm(h, mlp.gate_proj.weight.t(), out=g)
            torch.mm(h, mlp.up_proj.weight.t(), out=u)
            fvk.silu_mul_qwen36_bf16(g.data_ptr(), u.data_ptr(), g.data_ptr(), P * F_, s)
            torch.mm(g, mlp.down_proj.weight.t(), out=d)
            if li < len(deepstack):
                fvk.residual_add(x.data_ptr(), d.data_ptr(), P * H, s)
                x[vis_idx] = x[vis_idx] + deepstack[li].to(BF16)
                if li + 1 < len(layers):
                    fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, layers[li + 1].input_layernorm.weight.data_ptr(),
                                            h.data_ptr(), P, H, self.eps, s)
            elif li + 1 < len(layers):
                fvk.alpamayo15_rms_bf16(x.data_ptr(), d.data_ptr(), layers[li + 1].input_layernorm.weight.data_ptr(),
                                        h.data_ptr(), P, H, self.eps, s)
            else:
                fvk.residual_add(x.data_ptr(), d.data_ptr(), P * H, s)
        with _autocast():
            logits = self.vlm.lm_head(lm.norm(x[P - 1:P].unsqueeze(0)))
        return logits, int(rope_deltas.view(-1)[0].item())

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    def _decode_buffers(self):
        if self._db is None:
            B, H, D, F_ = self.max_samples, self.hidden, self.head_dim, self.ffn
            mk = lambda *shape: torch.empty(*shape, device=DEV, dtype=BF16)
            self._db = {"x": mk(B, H), "h": mk(B, H), "qkv": mk(B, (self.n_heads + 2 * self.n_kv) * D),
                        "q": mk(B, self.n_heads * D), "o": mk(B, H), "gu": mk(B, 2 * F_), "a": mk(B, F_),
                        "d": mk(B, H), "cos": mk(1, D), "sin": mk(1, D),
                        "logits": mk(B, self.vlm.lm_head.weight.shape[0])}
        return self._db

    def _cu_q(self, n: int, rows: int) -> torch.Tensor:
        key = (n, rows)
        if key not in self._cuq:
            self._cuq[key] = (torch.arange(n + 1, device=DEV, dtype=torch.int32) * rows).contiguous()
        return self._cuq[key]

    def _attend(self, li: int, q: torch.Tensor, n: int, rows: int, single_len: torch.Tensor,
                suffix_len: torch.Tensor, splits: int) -> torch.Tensor:
        """Non-causal attention of ``q`` [n * rows, heads, D] over the cache.

        One sample: a single pass over rows [0, single_len). Several samples:
        the shared prompt (sample 0, rows [0, P)) once for all queries plus each
        sample's suffix (rows [P, P + suffix_len)), merged by log-sum-exp.
        """
        kc, vc = self.k_cache[li], self.v_cache[li]
        heads, D = q.shape[1], q.shape[2]
        if n == 1:
            out, _ = self._fa4(q.view(1, rows, heads, D), kc[:1], vc[:1], seqused_k=single_len[:1],
                               softmax_scale=self.scale, causal=False, pack_gqa=True, num_splits=splits)
            return out.view(rows, -1)
        o1, l1 = self._fa4(q.view(1, n * rows, heads, D), kc[:1], vc[:1], seqused_k=self.d_plen,
                           softmax_scale=self.scale, causal=False, pack_gqa=True, return_lse=True)
        S, KV = self.max_seq, self.n_kv
        o2, l2 = self._fa4(q, kc[:n].view(n * S, KV, D), vc[:n].view(n * S, KV, D),
                           cu_seqlens_q=self._cu_q(n, rows), cu_seqlens_k=self.d_cuk[:n + 1],
                           seqused_k=suffix_len[:n], max_seqlen_q=rows, max_seqlen_k=self.max_new + self.n_act,
                           softmax_scale=self.scale, causal=False, pack_gqa=True, return_lse=True)
        return merge_attention(o1.view(n * rows, heads, D), l1.view(heads, n * rows),
                               o2, l2.view(heads, n * rows)).view(n * rows, -1)

    def _decode_step(self, n: int) -> None:
        """One reasoning token for ``n`` samples: 36 layers (BF16, or FP8 projection weights) + on-device sampling."""
        lm, fvk = self.lm, self._fvk
        s = torch.cuda.current_stream().cuda_stream
        H, D, NH, NKV, F_, S = self.hidden, self.head_dim, self.n_heads, self.n_kv, self.ffn, self.max_seq
        b = self._decode_buffers()
        x, h, qkv, q, o, gu, act, d, logits = (b[k][:n] for k in ("x", "h", "qkv", "q", "o", "gu", "a", "d", "logits"))
        torch.index_select(lm.embed_tokens.weight, 0, self.d_tok[:n], out=x)
        torch.index_select(self.cos_table, 0, self.d_rpos, out=b["cos"])
        torch.index_select(self.sin_table, 0, self.d_rpos, out=b["sin"])
        cos = b["cos"].expand(n, D).contiguous() if n > 1 else b["cos"]
        sin = b["sin"].expand(n, D).contiguous() if n > 1 else b["sin"]
        layers = lm.layers
        fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, layers[0].input_layernorm.weight.data_ptr(), h.data_ptr(), n, H, self.eps, s)
        for li, layer in enumerate(layers):
            a = layer.self_attn
            self._proj(h, self.w_qkv[li], "qkv", li, qkv, s)
            fvk.alpamayo15_vlm_qk_prep_bf16(
                qkv.data_ptr(), a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(),
                cos.data_ptr(), sin.data_ptr(), q.data_ptr(),
                self.k_cache[li].data_ptr(), self.v_cache[li].data_ptr(), self.d_slot.data_ptr(),
                n, NH, NKV, NKV * D, 1, S, self.eps, s)
            attn = self._attend(li, q.view(n, NH, D), n, 1, self.d_len, self.d_glen, self.decode_splits)
            self._proj(attn, a.o_proj.weight, "o", li, o, s)
            fvk.alpamayo15_rms_bf16(x.data_ptr(), o.data_ptr(), layer.post_attention_layernorm.weight.data_ptr(),
                                    h.data_ptr(), n, H, self.eps, s)
            self._proj(h, self.w_gu[li], "gu", li, gu, s)
            if n == 1:
                fvk.silu_mul_qwen36_bf16(gu.data_ptr(), gu[:, F_:].data_ptr(), act.data_ptr(), F_, s)
            else:
                act.copy_(F.silu(gu[:, :F_]) * gu[:, F_:])
            self._proj(act, layer.mlp.down_proj.weight, "down", li, d, s)
            nxt = layers[li + 1].input_layernorm.weight if li + 1 < len(layers) else lm.norm.weight
            fvk.alpamayo15_rms_bf16(x.data_ptr(), d.data_ptr(), nxt.data_ptr(), h.data_ptr(), n, H, self.eps, s)
        if n == 1 or self._q8 is not None:
            self._proj(h, self.vlm.lm_head.weight, "lm_head", -1, logits, s)
        else:  # vocab-major product keeps the skinny-M lm_head at full weight-streaming bandwidth
            key = f"logits_t{n}"
            if key not in b:
                b[key] = torch.empty(logits.shape[1], n, device=DEV, dtype=BF16)
            logits_t = b[key]
            torch.mm(self.vlm.lm_head.weight, h.t(), out=logits_t)
            logits = logits_t.t()
        nxt = self._sample(logits)
        unf = self.d_unfinished[:n]
        self.d_tok[:n].copy_(nxt * unf + self.pad_id * (1 - unf))
        self.d_slot.add_(1)
        self.d_len.add_(1)
        self.d_glen.add_(1)
        self.d_rpos.add_(1)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Hugging Face ``_sample`` token selection for a [n, vocab] logits block."""
        scores = logits.to(copy=True, dtype=torch.float32)
        scores = self.logits_processor(None, scores)
        probs = F.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    def _ensure_decode_graph(self, n: int) -> torch.cuda.CUDAGraph:
        g = self._decode_graphs.get(n)
        if g is not None:
            return g
        # Warm up and capture on scratch state; restore state and the RNG so that
        # capture never consumes the caller's random stream or touches live rows.
        state = (self.d_tok, self.d_unfinished, self.d_slot, self.d_len, self.d_glen, self.d_plen, self.d_cuk,
                 self.d_rpos)
        saved = [t.clone() for t in state]
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        rng = gen.get_state()

        def scratch():
            self.d_slot.fill_(self.max_seq - 1)
            self.d_len.fill_(1)
            self.d_glen.fill_(1)
            self.d_plen.fill_(1)
            self.d_cuk.copy_(torch.arange(self.max_samples + 1, device=DEV, dtype=torch.int32) * self.max_seq)
            self.d_rpos.zero_()
            self.d_unfinished.fill_(1)
        scratch()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st), torch.no_grad(), _autocast():
            for _ in range(2):
                self._decode_step(n)
                scratch()
        torch.cuda.current_stream().wait_stream(st)
        scratch()
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), _autocast(), torch.cuda.graph(g):
            self._decode_step(n)
        torch.cuda.synchronize()
        for t, sv in zip(state, saved):
            t.copy_(sv)
        gen.set_state(rng)
        self._decode_graphs[n] = g
        return g

    # ------------------------------------------------------------------
    # expert
    # ------------------------------------------------------------------
    def _expert_buffers(self):
        if self._eb is None:
            rows, E, D = self.max_samples * self.n_act, self.e_hidden, self.head_dim
            mk = lambda *shape, dt=BF16: torch.empty(*shape, device=DEV, dtype=dt)
            ff = self.expert.config.intermediate_size
            self._eb = {"res": mk(rows, E, dt=torch.float32), "hn": mk(rows, E),
                        "qkv": mk(rows, (self.e_heads + 2 * self.n_kv) * D), "q": mk(rows, self.e_heads * D),
                        "attn": mk(rows, self.e_heads * D), "o": mk(rows, E), "g": mk(rows, ff), "u": mk(rows, ff),
                        "d": mk(rows, E)}
        return self._eb

    def _expert_velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Expert forward for one Euler step over ``x`` [n, n_act, act_dim], official autocast numerics."""
        m, fvk = self.m, self._fvk
        s = torch.cuda.current_stream().cuda_stream
        n = x.shape[0]
        A, E, D, HQ, KV, S = self.n_act, self.e_hidden, self.head_dim, self.e_heads, self.n_kv, self.max_seq
        rows = n * A
        b = self._expert_buffers()
        res, hn, qkv, q, attn, o, g, u, d = (b[k][:rows] for k in ("res", "hn", "qkv", "q", "attn", "o", "g", "u", "d"))
        ff = g.shape[1]
        eps = self.e_eps
        layers = self.expert.layers
        res.copy_(m.action_in_proj(x, t).view(rows, E))                 # fp32 residual stream
        fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), 0, layers[0].input_layernorm.weight.data_ptr(),
                                        hn.data_ptr(), rows, E, eps, s)
        for li, layer in enumerate(layers):
            a = layer.self_attn
            torch.mm(hn, self.e_w_qkv[li].t(), out=qkv)
            fvk.alpamayo15_expert_qk_prep_bf16(
                qkv.data_ptr(), a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(),
                self.e_cos.data_ptr(), self.e_sin.data_ptr(), q.data_ptr(),
                self.k_cache[li].data_ptr(), self.v_cache[li].data_ptr(), self.d_aslot.data_ptr(),
                rows, HQ, KV, KV * D, A, S, eps, s)
            attn = self._attend(li, q.view(rows, HQ, D), n, A, self.d_eseq, self.d_esuf, self.expert_splits)
            torch.mm(attn, a.o_proj.weight.t(), out=o)
            fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), o.data_ptr(),
                                            layer.post_attention_layernorm.weight.data_ptr(),
                                            hn.data_ptr(), rows, E, eps, s)
            mlp = layer.mlp
            torch.mm(hn, mlp.gate_proj.weight.t(), out=g)
            torch.mm(hn, mlp.up_proj.weight.t(), out=u)
            fvk.silu_mul_qwen36_bf16(g.data_ptr(), u.data_ptr(), g.data_ptr(), rows * ff, s)
            torch.mm(g, mlp.down_proj.weight.t(), out=d)
            nxt = layers[li + 1].input_layernorm.weight if li + 1 < len(layers) else self.expert.norm.weight
            fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), d.data_ptr(), nxt.data_ptr(), hn.data_ptr(), rows, E, eps, s)
        return F.linear(hn, m.action_out_proj.weight, m.action_out_proj.bias).view(-1, A, self.act_dim)

    def _denoise(self, n: int) -> None:
        ts, x = self.time_steps, self.x[:n]
        for i in range(self.n_steps):
            dt = (ts[i + 1] - ts[i]).view(1, 1, 1).expand(n, 1, 1)
            t_start = ts[i].view(1, 1, 1).expand(n, 1, 1)
            v = self._expert_velocity(x, t_start)
            x.copy_(x + dt * v)

    def _ensure_expert_graph(self, n: int) -> torch.cuda.CUDAGraph:
        g = self._expert_graphs.get(n)
        if g is not None:
            return g
        saved = [t.clone() for t in (self.x, self.d_eseq, self.d_esuf, self.d_aslot, self.d_plen, self.d_cuk)]
        self.d_eseq.fill_(self.n_act + 1)
        self.d_esuf.fill_(self.n_act)
        self.d_aslot.fill_(self.max_seq - self.n_act)   # scratch rows past any live prefix
        self.d_plen.fill_(1)
        self.d_cuk.copy_(torch.arange(self.max_samples + 1, device=DEV, dtype=torch.int32) * self.max_seq)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st), torch.no_grad(), _autocast():
            for _ in range(2):
                self._denoise(n)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), _autocast(), torch.cuda.graph(g):
            self._denoise(n)
        torch.cuda.synchronize()
        for t, sv in zip((self.x, self.d_eseq, self.d_esuf, self.d_aslot, self.d_plen, self.d_cuk), saved):
            t.copy_(sv)
        self._expert_graphs[n] = g
        return g

    def capture(self, num_samples: tuple[int, ...] = (1,)) -> None:
        """Capture the decode and diffusion graphs for the given sample counts and
        JIT-compile the prefill and vision attention kernels."""
        for n in num_samples:
            if not 1 <= n <= self.max_samples:
                raise ValueError(f"num_samples {n} outside [1, max_samples={self.max_samples}]")
            self._ensure_decode_graph(n)
            self._ensure_expert_graph(n)
        self._warm_attention()

    def _warm_attention(self) -> None:
        if getattr(self, "_attention_warm", False):
            return
        # FA4 compiles per configuration, which includes length-dependent tiling
        # choices: warm with prompt- and frame-sized sequences, not toy shapes.
        D, NH, NKV = self.head_dim, self.n_heads, self.n_kv
        q = torch.zeros(1, 1024, NH, D, device=DEV, dtype=BF16)
        kv = torch.zeros(1, 1024, NKV, D, device=DEV, dtype=BF16)
        self._fa4(q, kv, kv, softmax_scale=self.scale, causal=True, pack_gqa=True)
        if self.vision_attention == "fa4":
            V = self.vlm.model.visual
            nh = V.blocks[0].attn.num_heads
            hd = V.blocks[0].attn.qkv.weight.shape[1] // nh
            qv = torch.zeros(2, 3, 720, nh, hd, device=DEV, dtype=BF16)   # interleaved q/k/v views
            self._fa4(qv[:, 0], qv[:, 1], qv[:, 2], softmax_scale=V.blocks[0].attn.scaling, causal=False)
        torch.cuda.synchronize()
        self._attention_warm = True

    # ------------------------------------------------------------------
    # request
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_trajectories(self, data: dict, *, num_traj_samples: int = 1, return_extra: bool = True,
                            stage_events: dict | None = None):
        """Official request semantics (``num_traj_sets=1``).

        ``data`` is the official model input dict (``tokenized_data`` plus
        ``ego_history_xyz``/``ego_history_rot`` on CUDA). The caller seeds the
        CUDA generator exactly as for the official entry point. Returns
        ``pred_xyz`` [1, 1, N, T, 3], ``pred_rot`` [1, 1, N, T, 3, 3] and the
        extracted text fields shaped [1, 1, N].
        """
        def mark(name):
            if stage_events is not None:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                stage_events[name] = e

        m, N = self.m, int(num_traj_samples)
        self.capture((N,))
        self._mark = mark if stage_events is not None else None
        tokenized = dict(data["tokenized_data"])
        input_ids = tokenized.pop("input_ids")
        if tokenized.get("pixel_values_videos") is not None:
            raise ValueError("Alpamayo15Thor takes image frames; video inputs are not supported")
        hist_xyz, hist_rot = data["ego_history_xyz"], data["ego_history_rot"]
        if input_ids.shape[0] != 1 or hist_xyz.shape[1] != 1:
            raise ValueError("Alpamayo15Thor supports batch 1 / one trajectory group")
        input_ids = m.fuse_traj_tokens(input_ids, {"ego_history_xyz": hist_xyz, "ego_history_rot": hist_rot})
        P = input_ids.shape[1]
        if P > self.max_prompt:
            raise ValueError(f"prompt length {P} exceeds max_prompt {self.max_prompt}")

        # 1) vision + prefill once (sample 0 holds the shared prompt K/V); first tokens.
        mark("prefill_start")
        if self.prefill_impl == "native":
            logits, rope_delta = self._prefill_native(input_ids, tokenized)
        else:
            logits, rope_delta = self._prefill_official(input_ids, tokenized)
        first = self._sample(logits[:, -1].expand(N, -1))
        mark("prefill_end")

        # 2) decode: graph replays, host evaluates the official stop criteria.
        tokens = [[t] for t in first.tolist()]
        traj_found, unfinished = [False] * N, [True] * N
        update_stop_state([t[-1] for t in tokens], traj_found, unfinished, 1,
                          traj_start_id=self.traj_start_id, max_new_tokens=self.max_new, eos_ids=self.eos_ids)
        graph = self._decode_graphs[N]
        self.d_tok[:N].copy_(first)
        self.d_unfinished[:N].copy_(torch.tensor(unfinished, dtype=torch.long))
        self.d_slot.fill_(P)
        self.d_len.fill_(P + 1)
        self.d_glen.fill_(1)
        self.d_plen.fill_(P)
        cuk = torch.arange(N + 1, dtype=torch.int32) * self.max_seq
        cuk[:N] += P
        self.d_cuk[:N + 1].copy_(cuk)
        self.d_rpos.fill_(P + rope_delta)
        while any(unfinished):
            graph.replay()
            last = self.d_tok[:N].tolist()
            for row, t in zip(tokens, last):
                row.append(t)
            before = list(unfinished)
            update_stop_state(last, traj_found, unfinished, len(tokens[0]),
                              traj_start_id=self.traj_start_id, max_new_tokens=self.max_new, eos_ids=self.eos_ids)
            if unfinished != before and any(unfinished):
                self.d_unfinished[:N].copy_(torch.tensor(unfinished, dtype=torch.long))
        mark("decode_end")

        # 3) diffusion expert over each sample's cached prefix.
        offsets, visible = expert_attention_spans(tokens, P, self.traj_start_id)
        off = torch.tensor(offsets, device=DEV)
        pos = torch.arange(self.n_act, device=DEV)[None] + rope_delta + off[:, None]          # [N, n_act]
        cos, sin = self.expert.rotary_emb(torch.zeros(1, device=DEV, dtype=torch.float32),
                                          pos[None].expand(3, -1, -1))
        self.e_cos[:N].copy_(cos)
        self.e_sin[:N].copy_(sin)
        self.d_aslot[:N].copy_(torch.tensor(visible, dtype=torch.long))
        self.d_eseq[:N].copy_(torch.tensor([v + self.n_act for v in visible], dtype=torch.int32))
        self.d_esuf[:N].copy_(torch.tensor([v - P + self.n_act for v in visible], dtype=torch.int32))
        noise = torch.randn(N, self.n_act, self.act_dim, device=DEV)   # official draw point
        self.x[:N].copy_(noise)
        self._expert_graphs[N].replay()
        mark("diffusion_end")

        # 4) trajectory (official).
        sampled = self.x[:N].clone()
        hx = hist_xyz[:, -1].repeat_interleave(N, dim=0)
        hr = hist_rot[:, -1].repeat_interleave(N, dim=0)
        pred_xyz, pred_rot = m.action_space.action_to_traj(sampled, hx, hr)
        pred_xyz = pred_xyz.view(1, 1, N, *pred_xyz.shape[1:])
        pred_rot = pred_rot.view(1, 1, N, *pred_rot.shape[1:])
        mark("end")
        if not return_extra:
            return pred_xyz, pred_rot
        from alpamayo1_5.models.token_utils import extract_text_tokens, replace_padding_after_eos
        seq = torch.cat([input_ids.expand(N, -1), torch.tensor(tokens, device=DEV)], dim=1)
        seq = replace_padding_after_eos(seq, self.traj_start_id, self.pad_id)
        extra = extract_text_tokens(m.tokenizer, seq)
        for key in extra:
            extra[key] = np.array(extra[key]).reshape([1, 1, N])
        extra["generated_tokens"] = tokens[0] if N == 1 else tokens
        extra["noise"] = noise
        extra["rope_delta"] = rope_delta
        extra["prompt_len"] = P
        return pred_xyz, pred_rot, extra


class _TrajTokenMask:
    """Official ``ExpertLogitsProcessor``: masks discrete trajectory tokens."""

    def __init__(self, offset: int, size: int) -> None:
        self.offset, self.size = offset, size

    def __call__(self, input_ids, scores):
        scores[:, self.offset:self.offset + self.size] = float("-inf")
        return scores
