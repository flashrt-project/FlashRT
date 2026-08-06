"""FlashRT -- Nex-N2-mini inference frontend (PyTorch + RTX SM120).

LLM frontend for the qwen3_5_moe 35B-A3B model. Surface:
    - ``__init__(checkpoint_path, *, kernelized=, quant_scope=, ...)``
    - ``set_prompt(text)``          -- tokenizes for the next call
    - ``infer()``                   -- single forward, returns logits
    - ``generate(max_new_tokens)``  -- greedy decode
    - ``latency_records``           -- list[float] populated by infer()

Two backends share this surface:
  * ``kernelized=True`` (production): NVFP4 weights loaded directly via the
    fvk loader and run through the SM120 kernel forward / decode -- fits the
    32 GB RTX 5090 (the BF16 model does not). Requires the gated kernel build
    (-DFLASHRT_ENABLE_QWEN35MOE=ON). See docs/nexn2_usage.md.
  * ``kernelized=False`` (reference): the BF16 HF model, used to lock the
    golden cosine fixture. Large (35B total params) -- loads with HF device
    mapping and may offload to host RAM.
"""

from __future__ import annotations

import time

from flash_rt.models.nexn2.pipeline_rtx import Nexn2Pipeline

# Kernels the kernelized path calls; checked up front so a build without
# -DFLASHRT_ENABLE_QWEN35MOE=ON fails clearly instead of after loading the 35B
# checkpoint and crashing mid-forward on a missing symbol.
_REQUIRED_FVK = (
    'w16a16_gemm_sm120_bf16', 'moe_blocktile_mma_sm120_bf16',
    'moe_weighted_sum_sm120_bf16', 'moe_router_topk_sm120_bf16',
    'qwen36_partial_rope_qk_bf16', 'causal_conv1d_qwen36_bf16',
    'gdn_recurrent_seq_sm120_bf16',
)


# The tier combination each target can build. FLASHRT_ENABLE_QWEN35MOE turns on
# all three tiers including the block-scaled 4-bit MMA one, which needs
# sm_120a/sm_121a; recommending it on a target whose toolchain refuses it sends
# the reader to a configure error. So the advice is keyed by the device in
# front of them.
_TIER_ADVICE = {
    (11, 0): ("-DGPU_ARCH=110 -DFLASHRT_ENABLE_QWEN35MOE_CORE=ON "
              "-DFLASHRT_ENABLE_QWEN35MOE_W4A16=ON"),
    (12, 0): "-DGPU_ARCH=120 -DFLASHRT_ENABLE_QWEN35MOE=ON",
    (12, 1): "-DGPU_ARCH=121 -DFLASHRT_ENABLE_QWEN35MOE=ON",
}


def _build_advice() -> str:
    """The configure flags for the device this process is actually running."""
    try:
        import torch

        cap = torch.cuda.get_device_capability()
    except Exception:                                       # pragma: no cover
        return ("-DFLASHRT_ENABLE_QWEN35MOE=ON on sm_120a/sm_121a, or "
                "-DFLASHRT_ENABLE_QWEN35MOE_CORE=ON "
                "-DFLASHRT_ENABLE_QWEN35MOE_W4A16=ON elsewhere")
    return _TIER_ADVICE.get(
        cap,
        f"the tiers sm_{cap[0]}{cap[1]} can compile "
        "(FLASHRT_ENABLE_QWEN35MOE_CORE / _W4A16; _W4A4 needs sm_120a)")


def _require_kernels(
        fvk, *, model_label: str = "Nex-N2",
        usage_doc: str = "docs/nexn2_usage.md",
        required=None, require_fa2: bool = True) -> None:
    """Raise a clear RuntimeError if the gated qwen3_5_moe kernels or the FA2
    module are missing (the build did not enable the qwen3_5_moe tiers, or
    flash_rt_fa2 is absent).

    ``required`` lets a configuration that calls fewer kernels say so. A list
    demanding more than a path uses turns a working build into a refusal; one
    demanding less lets a missing symbol surface mid-forward. Both are wrong,
    so the list belongs with whatever decides which kernels get called.
    """
    missing = [s for s in (required or _REQUIRED_FVK) if not hasattr(fvk, s)]
    if missing:
        raise RuntimeError(
            f"{model_label} kernelized path needs the gated qwen3_5_moe "
            "kernels, which are absent from flash_rt_kernels (missing: "
            f"{', '.join(missing)}). Reconfigure with {_build_advice()}. "
            f"See {usage_doc}.")
    if not require_fa2:
        # The attention backend probes its kernel and falls back to a
        # reference implementation, so a target that builds no FA2 still runs
        # -- more slowly on a long prompt, and never differently.
        return
    try:
        from flash_rt import flash_rt_fa2 as _fa2
    except Exception as e:                                  # pragma: no cover
        raise RuntimeError(
            f"{model_label} full attention needs the vendored FA2 module "
            "(flash_rt_fa2), which failed to import. Build with FA2 enabled "
            "(automatic on sm_80/86/87/89/120/121; on Thor sm_110 it is "
            "opt-in with -DFLASHRT_ENABLE_THOR_FA2=ON).") from e
    fa2_missing = [s for s in ('fwd_bf16', 'fwd_bf16_causal')
                   if not hasattr(_fa2, s)]
    if fa2_missing:                                         # pragma: no cover
        raise RuntimeError(
            "flash_rt_fa2 is present but lacks "
            f"{', '.join(fa2_missing)} (decode uses fwd_bf16, prefill uses "
            "fwd_bf16_causal); rebuild the FA2 module.")


class Nexn2TorchFrontendRtx:
    """Nex-N2-mini inference frontend (PyTorch + RTX SM120)."""

    # Kernels this configuration calls; a subclass whose path calls fewer
    # narrows it. See _require_kernels.
    _REQUIRED_KERNELS = _REQUIRED_FVK

    # Whether the vendored FA2 module must be present. A subclass whose
    # attention backend can fall back sets this False.
    _REQUIRE_FA2 = True

    _MODEL_LABEL = "Nex-N2"
    _USAGE_DOC = "docs/nexn2_usage.md"

    def __init__(self, checkpoint_path: str, *,
                 device: str = 'cuda:0',
                 max_seq: int = 2048,
                 quant: str = 'nvfp4',
                 kernelized: bool = False,
                 quant_scope: str = 'experts') -> None:
        """Construct the frontend.

        Args:
          checkpoint_path: HF-style checkpoint directory.
          device: cuda device string for the kernelized path.
          max_seq: maximum sequence length (KV + scratch sized to this).
          quant: weight quantization format for the kernelized path. Only
            ``'nvfp4'`` is implemented (NVFP4 W4A16 for full-attn + MoE GEMM;
            GDN in_proj kept BF16); any other value raises NotImplementedError.
          kernelized: when False (default) load the BF16 HF reference model
            (correctness baseline; the 35B-A3B weights do not fit the 32 GB
            card). When True load the NVFP4-quantized weights directly via the
            fvk loader and run the kernel forward/decode -- the production path.
          quant_scope: kernelized-only. ``'experts'`` (default) = only the
            routed experts are NVFP4; the dense projections run on the
            deterministic BF16-weight w16a16 GEMM, so prefill cos vs the BF16
            golden is ~0.99 and bit-reproducible. ``'full'`` additionally
            NVFP4-quantises the non-red-line dense projections (q/k/v/o /
            out_proj / shared) for a smaller footprint at lower cos.
        """
        if quant != 'nvfp4':
            raise NotImplementedError(
                f"quant={quant!r} is not implemented; only 'nvfp4' is "
                "supported (the kernelized path quantizes via "
                "extract_weights_nexn2_nvfp4).")

        self.checkpoint_path = checkpoint_path
        self.device = device
        self._user_max_seq = int(max_seq)
        self._quant_format = quant
        self._kernelized = bool(kernelized)
        self._quant_scope = quant_scope
        # Set by a subclass that streams the routed experts from a bundle
        # instead of holding them; see _nexn2_rtx_decode._moe_experts_streamed.
        self._stream_experts = getattr(self, '_stream_experts', False)
        # Read by the loader before any weight is touched, like the above:
        # the draft head is another layer's worth of weights and is only
        # useful with a verifier, so nothing loads it unless asked.
        self._load_mtp = getattr(self, '_load_mtp', False)
        self._tokenizer = None
        self._prompt_ids = None
        self._pipeline: Nexn2Pipeline | None = None
        self._weights = None
        self._fvk = None
        self._decode_state = None
        self.latency_records: list[float] = []

        if self._kernelized:
            self._build_kernelized_nvfp4()
        else:
            self._build_phase1_reference()

    def _build_phase1_reference(self) -> None:
        """Load tokenizer + the BF16 HF reference model (kernelized=False).

        This is the correctness baseline only; the production path is the
        kernelized forward/decode (kernelized=True).
        """
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        hf_model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint_path,
            dtype=torch.bfloat16,
            device_map='auto',
        )
        hf_model.eval()
        self._pipeline = Nexn2Pipeline(hf_model)

    def _build_kernelized_nvfp4(self) -> None:
        """Load NVFP4 weights via the fvk loader and arm the kernel forward.

        No HF model: the 35B-A3B checkpoint does not fit a 32 GB RTX 5090
        in BF16. The loader streams each shard, quantizes the large GEMMs
        to NVFP4 (GDN in_proj / norms / router kept BF16) and frees the
        BF16 source as it goes, fitting in ~22 GB.
        """
        from flash_rt import flash_rt_kernels as fvk
        from flash_rt.frontends.torch._nexn2_rtx_nvfp4_weights import (
            extract_weights_nexn2_nvfp4,
        )

        # Fail before loading the 35B checkpoint.
        _require_kernels(
            fvk,
            model_label=self._MODEL_LABEL,
            usage_doc=self._USAGE_DOC,
            required=self._REQUIRED_KERNELS,
            require_fa2=self._REQUIRE_FA2,
        )

        self._fvk = fvk
        self._weights = extract_weights_nexn2_nvfp4(
            self.checkpoint_path, fvk, device=self.device,
            quant_scope=self._quant_scope,
            stream_experts=self._stream_experts,
            load_mtp=self._load_mtp)

    @property
    def tokenizer(self):
        """The checkpoint's tokenizer, loaded when something asks for it.

        Loading it eagerly would make ``transformers`` a hard requirement of
        the runtime, which it is not: a caller that supplies token ids through
        :meth:`set_prompt_ids` never needs one. That matters for a deployment
        target where the dependency may be absent or unwelcome, and it keeps
        the kernel and weight paths testable without it.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint_path)
        return self._tokenizer

    def set_prompt_ids(self, token_ids) -> None:
        """Set the prompt from token ids, requiring no tokenizer."""
        import torch

        ids = torch.as_tensor(
            token_ids, dtype=torch.long, device=self.device).reshape(1, -1)
        if ids.shape[1] == 0:
            raise ValueError('token_ids is empty')
        # Matches set_prompt: the decode state is not discarded, because
        # seed_prefill resets the recurrent and KV caches itself and
        # reallocating them per prompt would be waste.
        self._prompt_ids = ids

    def set_prompt(self, text: str) -> None:
        """Tokenize ``text`` for the next ``infer()`` / ``generate()`` call."""
        enc = self.tokenizer(text, return_tensors='pt')
        self._prompt_ids = enc['input_ids'].to(self.device)

    def infer(self):
        """Single forward pass over the current prompt; returns logits.

        Returns:
            logits: (B, S, vocab_size) tensor.
        """
        if self._prompt_ids is None:
            raise ValueError('call set_prompt(...) before infer()')
        if self._kernelized:
            import torch

            from flash_rt.frontends.torch._nexn2_rtx_forward import (
                nexn2_forward_nvfp4,
            )
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = nexn2_forward_nvfp4(
                    self._weights, self._prompt_ids, self._fvk, self.device)
            torch.cuda.synchronize()
            self.latency_records.append(time.perf_counter() - t0)
            return logits.unsqueeze(0)        # (1, S, vocab)
        t0 = time.perf_counter()
        logits = self._pipeline.forward(self._prompt_ids)
        self.latency_records.append(time.perf_counter() - t0)
        return logits

    def generate(self, max_new_tokens: int, *, do_sample: bool = False):
        """Autoregressive generate over the current prompt.

        Kernelized path: greedy M=1 decode over the fvk kernels (KV cache +
        GDN recurrent/conv state). Reference path: HF .generate().
        """
        if self._prompt_ids is None:
            raise ValueError('call set_prompt(...) before generate()')
        if self._kernelized:
            from flash_rt.frontends.torch._nexn2_rtx_decode import (
                Nexn2DecodeState, generate_greedy,
            )
            if self._decode_state is None:
                self._decode_state = Nexn2DecodeState(
                    self._weights, self._user_max_seq, self.device)
            return generate_greedy(
                self._decode_state, self._prompt_ids, max_new_tokens,
                self._fvk, self.device)
        return self._pipeline.generate(
            self._prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
