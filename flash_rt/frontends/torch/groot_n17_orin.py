"""FlashRT -- GROOT N1.7 torch frontend for Jetson Orin (SM87).

SM87 is Ampere: no FP8 and no FP4 tensor cores. Thor's FP8/FP4 DiT tiers and
RTX SM120's NVFP4 path therefore have nothing to run on here, and the
production weight spec's FP8 round-trip is a pure precision loss. This
frontend keeps the N1.7 public contract (``set_prompt`` / ``infer`` /
``normalize_state`` / ``denormalize_action``) and the RTX DiT attention
backend, and swaps the dtype-bearing pieces for BF16:

* weights load through :data:`ORIN_WEIGHT_SPEC` (BF16, no ``Quant``), so
  there are no FP8 alphas to multiply back in;
* the backbone (ViT / DeepStack / LLM / vlln / VL-self-attn) and the DiT run
  through :mod:`flash_rt.models.groot_n17.pipeline_orin`;
* LLM attention moves to FA2 ``fwd_bf16_causal`` with native GQA, dropping
  the FP16-only ``gpu_repeat_interleave_heads`` expansion.

BF16 (not FP16) is the right Orin dtype on three independent grounds: the
checkpoint is natively bf16 and the HF eager reference this port is gated
against runs in bf16; every INT8 kernel in the SM87 build is bf16-in /
bf16-out (there is no ``quantize_int8_rowwise_fp16`` here), and INT8 is the
one measured DiT lever on Orin; and the LLM residual stream carries a
15296-magnitude outlier channel, which bf16's fp32-width exponent handles
without headroom arithmetic.

Additive: the weight spec, the pipelines and the Thor/RTX frontends are left
untouched. What "subclasses the RTX full-FP16 frontend" means here is worth
stating plainly, because it reads as though FP16 behaviour is inherited and
none is. All 16 methods that class defines are overridden below and it defines
no ``__init__``, so it is reachable only as an MRO waypoint -- which is also
why its own header calling it "an A/B precision reference against the bf16
path" says nothing about this frontend. The reuse is two classes up:
``GrootN17TorchFrontendRtx.__init__`` (reached via ``super().__init__``) and
the 19 methods inherited from ``GrootN17TorchFrontendThor`` -- calibration, the
kernel-DiT graphs, ``normalize_state`` / ``denormalize_action`` / ``predict``
and the HF processor accessors. On top of those this frontend adds 28 of its
own (the SM87 arch gate, the INT8 DiT tier, the backbone CUDA graph, the
per-observation contract, the GPU image path). Subclassing Thor directly would
give the same 19 and none of the 16 replacements; changing the base of a
shipped chain is a deliberate refactor, not a side effect of this port.
"""

from __future__ import annotations

import glob
import logging
import math
import os
import time
from typing import Sequence

import torch

from flash_rt.frontends.torch.groot_n17_rtx_fp16 import (
    GrootN17TorchFrontendRtxFP16,
)

logger = logging.getLogger(__name__)

_BF16 = torch.bfloat16

#: Set once an inference tensor has been seen on the per-observation path, so the
#: explanation is logged once per process instead of once per frame.
_warned_inference_tensor = False

#: Set once the DiT graphs have been bypassed because this call's denoising
#: parameters differ from the ones they were captured with.
_warned_dit_graph_params = False


def _mutation_version(t) -> int | None:
    """PyTorch's in-place mutation counter, or ``None`` if it cannot report one.

    ``getattr(t, "_version", None)`` does **not** do this. ``_version`` is a
    property that *raises* ``RuntimeError("Inference tensors do not track
    version counter.")``, and ``getattr`` only substitutes the default for a
    missing attribute — an attribute that exists and throws propagates. Tensors
    captured under ``torch.inference_mode()`` are inference tensors, and that is
    the recommended way to run the official ``Gr00tPolicy.get_action``, so a
    live integration handing us its captures hits this on the first frame.

    ``None`` means "mutation is undetectable for this tensor", and every caller
    must treat it as *never* matching a cached version: the identity fast paths
    below exist to skip a host-side compare or an H2D copy, and taking one on a
    tensor we cannot version-check would serve silently stale bytes. Losing the
    fast path costs time; taking it wrongly costs correctness.
    """
    try:
        return t._version
    except (AttributeError, RuntimeError):
        return None


def _note_inference_tensor(where: str) -> None:
    """Explain, once, why a live aux bundle just got slower."""
    global _warned_inference_tensor
    if _warned_inference_tensor:
        return
    _warned_inference_tensor = True
    logger.warning(
        "%s received an inference tensor (produced under "
        "torch.inference_mode()). Its mutation counter is unreadable, so the "
        "identity fast path is disabled for it and every observation re-checks "
        "and re-copies instead. Correct, but ~1.5 ms/observation slower on the "
        "backbone load alone. Pass ordinary tensors (``t.clone()`` outside the "
        "inference_mode block) to keep the fast path.", where)


def _note_dit_graph_bypass(captured, want) -> None:
    """Explain, once, why the DiT graphs were bypassed and what it costs.

    The graphs bake ``Sa = action_horizon + 1`` into every DiT kernel's
    dimensions and the diffusion modulators into their pointers, so replaying
    them at a different horizon or timestep bucket does not fail — it processes
    the wrong number of token rows and returns a well-formed action. Measured on
    a real frame (docs §6.24): capturing at ``action_horizon=40`` and then asking
    for 20 moved the decoded action **0.0586 rad = 3.357 deg** from the eager
    arm, and the reverse order moved it **0.375 rad = 21.486 deg**, against a
    shipped worst case of 0.4935 deg. A non-default ``num_timestep_buckets``
    moved it 3.581 deg the same way, because the modulators were precomputed
    with the captured bucket while the action encoder used the requested one.

    Falling back to the eager arm is correct and not cheap: it recomputes
    ``_compute_dit_adaln_modulators`` per step, which is 32 layers x
    (1536, 3072) cast to fp32 = **576.38 MiB per step, 2305.50 MiB per
    inference** (48.02-48.07 ms of it, measured), and the whole arm measured
    **+64.120 / +67.222 ms per observation (1.6564x / 1.6855x)** over two runs.
    """
    global _warned_dit_graph_params
    if _warned_dit_graph_params:
        return
    _warned_dit_graph_params = True
    logger.warning(
        "the DiT CUDA graphs were captured for (num_inference_timesteps, "
        "action_horizon, num_timestep_buckets)=%s but this call asked for %s. "
        "Replaying them anyway would denoise the wrong number of action tokens "
        "and still return a well-formed action -- measured 3.36 to 21.49 deg of "
        "decoded-action error -- so the eager arm is used instead. That is "
        "correct but costs +64 to +67 ms per observation (1.66-1.69x), because "
        "the eager arm recomputes the AdaLN modulators per step (2305.50 MiB of "
        "fp32 casts per inference). To get the graphs back, keep one "
        "(action_horizon, num_timestep_buckets) per frontend, or pass "
        "use_dit_graph=False to make the slower arm explicit and silence this.",
        captured, want)


class GrootN17TorchFrontendOrin(GrootN17TorchFrontendRtxFP16):
    """N1.7 Orin SM87 frontend — BF16 backbone + rowwise-INT8 DiT.

    The DiT defaults to INT8 (``use_int8_dit=True``) because it is the measured
    winner and it cleared every gate the bf16 tier is held to, with no
    threshold relaxed: DiT x4 58.5 -> 41.5 ms (1.39-1.42x, two independent
    timing calipers agreeing to 2.2%), boundary 124.9 -> 108.0 ms, decoded
    action cos 0.999999 on both real frames. The cost is a worst-case decoded
    error of 0.364 deg against a joint range of roughly [-133, 152] deg; see
    docs/groot_n17_orin_sm87.md §6.9 for the full gate table and §6.8 for the
    fake-quant experiment that admitted the tier. Pass ``use_int8_dit=False``
    for the bf16 tier — both stay resident-capable so they can be A/B'd in one
    process.

    The INT8 tier **exempts the K/V projections** (``dit_bf16_families=
    ("k", "v")``). Two separate things, worth keeping apart:

    * the DiT's KV *cache* — the cross-attention K/V precomputed once per
      prompt and read by all 16 cross blocks in all 4 steps — was never
      quantized. ``_precompute_dit_cross_kv`` does the projection in fp32 from
      the bf16 weights and stores bf16, and cross blocks run no K/V GEMM at
      all, so there is no INT8 anywhere on that path;
    * the 32 per-step *self-attention* K/V projections were INT8, and exempting
      them costs **1.00 ms of a 33.23 ms** DiT loop (0.95% of the boundary)
      while moving frame 300's decoded error from 0.364 deg to **0.194 deg**
      against the bf16 tier's own 0.187 deg. Frame 0 is unchanged at 0.4935 deg
      in *all three* tiers, which locates that error in the bf16 backbone
      rather than in the DiT (§6.14).

    Pass ``dit_bf16_families=()`` for the all-six-family INT8 tier.
    """

    _REQUIRED_CAPABILITY = (8, 7)
    _ARCH_NAME = "Jetson Orin SM87"
    #: Set to "1" to skip the capability probe (same escape hatch as
    #: hyvla_orin's FLASHRT_HYVLA_FORCE_ARCH).
    _FORCE_ARCH_ENV = "FLASHRT_GROOT_N17_FORCE_ARCH"

    #: ``vision_config.deepstack_visual_indexes`` for GR00T-N1.7-3B: the ViT
    #: layers whose residual output feeds the 3 DeepStack mergers, which in
    #: turn inject into LLM layers 0/1/2.
    _DEEPSTACK_TAPS = (5, 11, 17)

    # The DiT never runs in FP8 here; keep the inherited class flags honest so
    # anything inspecting them does not believe an FP8 path is live. The
    # instance overrides _DIT_QUANT from the two tier kwargs, so this advertises
    # the shipped default: INT8 with the K/V projections exempt.
    _DIT_USE_FP8 = False
    _DIT_FP8_IMPL = "none"
    _DIT_QUANT = "int8_rowwise(k,v=bf16)"
    #: DiT weight families the INT8 tier leaves in bf16 when
    #: ``dit_bf16_families`` is not given. Measured trade, docs §6.14.
    _DIT_BF16_FAMILIES = ("k", "v")

    def __init__(
        self,
        checkpoint_path: str,
        *,
        num_views: int = 2,
        embodiment_tag: str = "new_embodiment",
        device: str = "cuda:0",
        use_fp8: bool = False,
        use_fp4: bool = False,
        load_strided_fmha: bool = False,
        fuse_image_embeds: bool = True,
        use_int8_dit: bool = True,
        dit_bf16_families: Sequence[str] | None = None,
        use_backbone_graph: bool = True,
    ):
        if use_fp4:
            raise RuntimeError(
                "GrootN17TorchFrontendOrin does not support FP4: SM87 has no "
                "native FP4 tensor cores (ENABLE_NVFP4 is disabled for "
                "GPU_ARCH=87). Use the default BF16 path.")
        if use_fp8:
            raise RuntimeError(
                "GrootN17TorchFrontendOrin does not support FP8: SM87 has no "
                "FP8 tensor cores, so an FP8 GEMM would be emulated slower "
                "than BF16. Use the default BF16 path.")

        self._require_arch(device)

        # Must be set before super().__init__(): that runs _load_weights, which
        # reads it to decide how many ViT layers to materialize. The fusion
        # needs the tower's *last* block output for the final merger, so it
        # costs the ViT-truncation lever (layers past the last DeepStack tap).
        self._fuse_image_embeds = bool(fuse_image_embeds)

        #: Only the ``infer(aux=...)`` path consults this; the one-shot path
        #: never captures. Capturing costs 262.1 ms once and saves 4.54 ms per
        #: observation (§6.13.2), so the break-even is 58 frames — a caller that
        #: serves only a handful of observations through ``aux`` should pass
        #: ``use_backbone_graph=False`` and keep the eager arm.
        self._use_backbone_graph = bool(use_backbone_graph)

        super().__init__(
            checkpoint_path,
            num_views=num_views,
            embodiment_tag=embodiment_tag,
            device=device,
            load_strided_fmha=load_strided_fmha,
        )

        # INT8 needs the loaded bf16 weights first, so unlike
        # fuse_image_embeds this runs *after* the parent init.
        self._use_int8_dit = bool(use_int8_dit)
        # None means "the shipped default for this tier": exempt K/V under
        # INT8, nothing to exempt under bf16. Spelled as a sentinel rather than
        # a ("k", "v") default so that use_int8_dit=False stays constructible
        # without the caller having to know to cancel the exemption.
        if dit_bf16_families is None:
            dit_bf16_families = (self._DIT_BF16_FAMILIES if self._use_int8_dit
                                 else ())
        self._dit_bf16_families = tuple(dit_bf16_families)
        bad = [f for f in self._dit_bf16_families
               if f not in self._DIT_INT8_FAMILIES]
        if bad:
            raise ValueError(
                f"dit_bf16_families names {bad}, which are not DiT weight "
                f"families {list(self._DIT_INT8_FAMILIES)}")
        if self._dit_bf16_families and not self._use_int8_dit:
            raise ValueError(
                f"dit_bf16_families={list(self._dit_bf16_families)} needs "
                "use_int8_dit=True: the bf16 tier already runs every family "
                "in bf16, so exempting some is a no-op that would silently "
                "misdescribe the tier")
        self._dit_int8_families = tuple(
            f for f in self._DIT_INT8_FAMILIES
            if f not in self._dit_bf16_families)
        if self._use_int8_dit:
            self._quantize_dit_weights()
        # Set in both directions: the class attribute advertises the default
        # (int8_rowwise), so an instance asked for bf16 must not inherit it.
        self._DIT_QUANT = "bf16" if not self._use_int8_dit else (
            "int8_rowwise" if not self._dit_bf16_families else
            "int8_rowwise(" + ",".join(self._dit_bf16_families) + "=bf16)")

    def _require_arch(self, device: str = "cuda:0") -> None:
        """Fail fast unless this really is an SM87 device.

        A wrong-arch frontend here does not fail at construction — it fails at
        the first kernel launch inside ``set_prompt``, with a CUDA error that
        does not name the cause.
        """
        if os.environ.get(self._FORCE_ARCH_ENV, "") == "1":
            return
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"GROOT N1.7 Orin frontend requires a CUDA device "
                f"({self._ARCH_NAME}); CUDA is not available.")
        spec = str(device)
        idx = int(spec.rsplit(":", 1)[1]) if ":" in spec else 0
        cap = torch.cuda.get_device_capability(idx)
        if cap != self._REQUIRED_CAPABILITY:
            raise RuntimeError(
                f"GROOT N1.7 Orin frontend requires {self._ARCH_NAME} "
                f"(capability {self._REQUIRED_CAPABILITY}), found capability "
                f"{cap}. Use the Thor/RTX/SM89 frontend for that device, or "
                f"set {self._FORCE_ARCH_ENV}=1 to bypass this probe.")

    # ────────────────────────────────────────────────────────────────
    # Weight loading
    # ────────────────────────────────────────────────────────────────

    def _shards(self) -> list:
        shards = sorted(
            glob.glob(os.path.join(self.checkpoint_path, "model-*.safetensors")))
        if not shards:
            raise FileNotFoundError(
                f"no model-*.safetensors shards in {self.checkpoint_path}")
        return shards

    def _load_weights(self) -> None:
        """Load every weight as BF16 (no FP8 round-trip) and slice the
        per-embodiment dense tables down to the active slot."""
        from flash_rt.executors.torch_weights import MultiSafetensorsSource
        from flash_rt.executors.weight_loader import WeightLoader
        from flash_rt.models.groot_n17.weight_spec_orin import ORIN_WEIGHT_SPEC

        source = MultiSafetensorsSource(self._shards(), device=self.device)
        WeightLoader(source=source, target=self, spec=ORIN_WEIGHT_SPEC).run()

        # ORIN_WEIGHT_SPEC drops Quant(), so weights arrive already bf16 and
        # no alpha list exists. Biases too. Just make them contiguous.
        for i in range(32):
            for attr in ("_dit_q_w", "_dit_k_w", "_dit_v_w", "_dit_o_w",
                         "_dit_ada_w", "_dit_ff_proj_w", "_dit_ff_down_w"):
                getattr(self, attr)[i] = getattr(self, attr)[i].contiguous()

        # Per-embodiment slot slicing: the spec loads the dense (32, ...)
        # tables; the encoders/decoder expect the already-sliced 2-D/1-D form.
        slot = self._embodiment_id
        for name in (
            "_st_enc_l1_W", "_st_enc_l1_b", "_st_enc_l2_W", "_st_enc_l2_b",
            "_ac_enc_W1_W", "_ac_enc_W1_b", "_ac_enc_W2_W", "_ac_enc_W2_b",
            "_ac_enc_W3_W", "_ac_enc_W3_b",
            "_ac_dec_l1_W", "_ac_dec_l1_b", "_ac_dec_l2_W", "_ac_dec_l2_b",
        ):
            full = getattr(self, name)
            setattr(self, name, full[slot].contiguous())
            del full

        # Weight files are ground truth for shapes; config.json is ground truth
        # for the configured horizon. Note position_embedding is (1024, D) —
        # a capacity, not the horizon — so it can only bound it from above.
        self._action_dim = int(self._ac_enc_W1_W.shape[0])
        self._dit_dim = int(self._ac_enc_W1_W.shape[1])
        if int(self._ac_enc_W3_W.shape[0]) != self._dit_dim:
            raise RuntimeError(
                f"action_encoder W3 input dim {self._ac_enc_W3_W.shape[0]} != "
                f"W1 output dim {self._dit_dim}")
        if int(self._ac_dec_l2_W.shape[1]) != self._action_dim:
            raise RuntimeError(
                f"action_decoder output dim {self._ac_dec_l2_W.shape[1]} != "
                f"action_dim {self._action_dim}")
        if self._dit_dim != int(self._dit_q_w[0].shape[0]):
            raise RuntimeError(
                f"DiT hidden dim {self._dit_q_w[0].shape[0]} != action_encoder "
                f"output dim {self._dit_dim}")

        cfg = self._read_config()
        self._action_horizon = int(cfg["action_horizon"])
        self._num_inference_timesteps = int(cfg["num_inference_timesteps"])
        self._num_timestep_buckets = int(cfg["num_timestep_buckets"])
        if int(cfg["max_action_dim"]) != self._action_dim:
            raise RuntimeError(
                f"config max_action_dim={cfg['max_action_dim']} disagrees with "
                f"the weight-derived action_dim={self._action_dim}; the "
                f"checkpoint and its config.json do not describe the same model")
        pos_capacity = int(self._ah_pos_embed_w.shape[0])
        if pos_capacity < self._action_horizon:
            raise RuntimeError(
                f"position_embedding capacity {pos_capacity} < configured "
                f"action_horizon {self._action_horizon}")
        if int(cfg["backbone_embedding_dim"]) != int(self._vlln_w.shape[0]):
            raise RuntimeError(
                f"config backbone_embedding_dim={cfg['backbone_embedding_dim']} "
                f"!= vlln weight dim {self._vlln_w.shape[0]}")

        # With the fusion OFF the vision tower is run only to produce the
        # DeepStack taps: the LLM's fused text+image input embeddings arrive in
        # ``aux`` (computed by the caller's HF forward), so the ViT's own final
        # output and the patch merger are never consumed. Layers past the last
        # tap are then dead compute — 6 of 24 here, ~8 ms of a 77 ms backbone.
        # With the fusion ON that lever is unavailable: the final merger
        # consumes the last block's output, so all 24 layers are live. The
        # trade is still strongly positive — the fusion removes HF's entire
        # 58.07 ms vision tower from the standalone path, which also means the
        # truncated mode was paying for the ViT twice (once in HF to build
        # llm_input_embeds, once here for the taps).
        self._vit_layers = (len(self._vit_ln1_w) if self._fuse_image_embeds
                            else max(self._DEEPSTACK_TAPS) + 1)

        self._load_bf16_shadow_weights()

    def _read_config(self) -> dict:
        if hasattr(self, "_cfg"):
            return self._cfg
        import json

        path = os.path.join(self.checkpoint_path, "config.json")
        with open(path) as f:
            cfg = json.load(f)
        required = ("action_horizon", "max_action_dim", "num_inference_timesteps",
                    "num_timestep_buckets", "backbone_embedding_dim")
        missing = [k for k in required if k not in cfg]
        if missing:
            raise RuntimeError(
                f"{path} is missing {missing}; refusing to guess inference "
                f"hyperparameters")
        self._cfg = cfg
        return cfg

    def _read_processor_geometry(self) -> dict:
        """The image-transform parameters this checkpoint actually ships.

        Read rather than defaulted, from ``processor_config.json`` in the
        checkpoint directory — the same file the vendor's processor reads. Two
        numbers decide the whole transform chain and neither can be guessed:

        * ``shortest_image_edge`` (**256** here) sets the patch grid, so a wrong
          value changes ``pixel_values``'s row count and is caught by the row
          check in :meth:`_pixel_values_from_frames`.
        * ``crop_fraction`` is **0.95** here while the vendor's *code* default is
          0.9. Nothing downstream would notice the difference except a quietly
          different image, which is precisely the failure worth refusing over.
          (``image_crop_size=[230,230]`` sits in the same file and is **not**
          what runs: the processor's ``fraction_to_use`` prefers
          ``crop_fraction`` whenever it is not None.)

        ``use_albumentations`` is checked as well, because the chain
        :mod:`flash_rt.frontends.torch._groot_n17_preprocess` reproduces *is* the
        albumentations eval transform; a checkpoint built with it off goes down a
        different resize path in the vendor code and would need a different
        reproduction, not this one.
        """
        if hasattr(self, "_proc_geom"):
            return self._proc_geom
        import json

        path = os.path.join(self.checkpoint_path, "processor_config.json")
        try:
            with open(path) as f:
                cfg = json.load(f)
        except OSError as exc:
            raise RuntimeError(
                f"the frames= image path needs {path} to learn this checkpoint's "
                f"shortest_image_edge and crop_fraction, and it is not readable "
                f"({exc!r}). Either hand infer() an already-processed aux bundle "
                "instead of frames=, or point checkpoint_path at a directory "
                "that carries its processor_config.json.") from exc
        kw = cfg.get("processor_kwargs", {})
        missing = [k for k in ("shortest_image_edge", "crop_fraction",
                               "use_albumentations") if k not in kw]
        if missing:
            raise RuntimeError(
                f"{path} processor_kwargs is missing {missing}; refusing to "
                "guess the image transform. A wrong crop_fraction yields a "
                "plausible image and a quietly degraded action.")
        if not kw["use_albumentations"]:
            raise NotImplementedError(
                f"{path} sets use_albumentations=False, so the vendor's "
                "evaluation transform is not the letterbox -> INTER_AREA -> "
                "centre-crop -> INTER_AREA chain that "
                "flash_rt.frontends.torch._groot_n17_preprocess reproduces. "
                "Pass an already-processed aux bundle for this checkpoint.")
        self._proc_geom = {
            "shortest": int(kw["shortest_image_edge"]),
            "crop_fraction": float(kw["crop_fraction"]),
        }
        return self._proc_geom

    def _load_bf16_shadow_weights(self) -> None:
        """Real bf16 GEMM weights for the kernel backbone, read straight from
        safetensors and materialized **per projection**.

        Every entry is ``[K, N]`` row-major (already transposed for
        ``gemm.bf16_nn``), keyed ``(stage, layer_idx, name)``. The per-layer
        weight dicts that ``_run_kernel_backbone`` hands to the pipeline are
        built from these once, not per call: the FP16 frontend's inline
        ``qkv[:, :1024].contiguous()`` slicing re-materializes ~120 large
        matrices on every backbone call, which measured 15.65 ms of a 77 ms
        backbone.

        The LLM checkpoint stores ``q_proj`` / ``k_proj`` / ``v_proj``
        separately, so they are loaded directly. Only the ViT's fused
        ``attn.qkv`` needs splitting.
        """
        from safetensors import safe_open

        handles = [safe_open(p, framework="pt", device=self.device)
                   for p in self._shards()]
        index: dict = {}
        for h in handles:
            for k in h.keys():
                index[k] = h

        def load_w(key: str) -> torch.Tensor:
            return index[key].get_tensor(key).to(_BF16).t().contiguous()

        def split3(key: str) -> tuple:
            """Split a fused ``[K, 3N]`` qkv into three contiguous ``[K, N]``."""
            w = load_w(key)
            n = w.shape[1] // 3
            if w.shape[1] != 3 * n:
                raise RuntimeError(
                    f"{key} has output dim {w.shape[1]}, not divisible by 3; "
                    f"not a fused qkv projection")
            return tuple(w[:, i * n:(i + 1) * n].contiguous() for i in range(3))

        shadow: dict = {}

        vp = "backbone.model.model.visual.blocks.{i}"
        for i in range(self._vit_layers):
            p = vp.format(i=i)
            q, k, v = split3(f"{p}.attn.qkv.weight")
            shadow[("vit", i, "q")], shadow[("vit", i, "k")] = q, k
            shadow[("vit", i, "v")] = v
            shadow[("vit", i, "o")] = load_w(f"{p}.attn.proj.weight")
            shadow[("vit", i, "fc1")] = load_w(f"{p}.mlp.linear_fc1.weight")
            shadow[("vit", i, "fc2")] = load_w(f"{p}.mlp.linear_fc2.weight")

        dsm = "backbone.model.model.visual.deepstack_merger_list.{j}"
        for j in range(3):
            shadow[("dsm", j, "fc1")] = load_w(
                f"{dsm.format(j=j)}.linear_fc1.weight")
            shadow[("dsm", j, "fc2")] = load_w(
                f"{dsm.format(j=j)}.linear_fc2.weight")

        lp = "backbone.model.model.language_model.layers.{i}"
        for i in range(16):
            p = lp.format(i=i)
            shadow[("llm", i, "q")] = load_w(f"{p}.self_attn.q_proj.weight")
            shadow[("llm", i, "k")] = load_w(f"{p}.self_attn.k_proj.weight")
            shadow[("llm", i, "v")] = load_w(f"{p}.self_attn.v_proj.weight")
            shadow[("llm", i, "o")] = load_w(f"{p}.self_attn.o_proj.weight")
            shadow[("llm", i, "gate")] = load_w(f"{p}.mlp.gate_proj.weight")
            shadow[("llm", i, "up")] = load_w(f"{p}.mlp.up_proj.weight")
            shadow[("llm", i, "down")] = load_w(f"{p}.mlp.down_proj.weight")

        vlp = "action_head.vl_self_attention.transformer_blocks.{i}"
        for i in range(4):
            p = vlp.format(i=i)
            shadow[("vlsa", i, "q")] = load_w(f"{p}.attn1.to_q.weight")
            shadow[("vlsa", i, "k")] = load_w(f"{p}.attn1.to_k.weight")
            shadow[("vlsa", i, "v")] = load_w(f"{p}.attn1.to_v.weight")
            shadow[("vlsa", i, "o")] = load_w(f"{p}.attn1.to_out.0.weight")
            shadow[("vlsa", i, "fc1")] = load_w(f"{p}.ff.net.0.proj.weight")
            shadow[("vlsa", i, "fc2")] = load_w(f"{p}.ff.net.2.weight")

        self._bf16_shadow_weights = shadow

    # ────────────────────────────────────────────────────────────────
    # Prompt setup
    # ────────────────────────────────────────────────────────────────

    def set_prompt(self, *, aux: dict, prompt: str | None = None) -> None:
        """Run the BF16 kernel backbone once for this (prompt, image) pair.

        ``aux`` carries the HF-derived setup tensors — the same bundle
        ``tests/_helpers/groot_n17/capture_llm_aux.py`` produces:
        ``llm_input_embeds``, ``rope_cos``, ``rope_sin``, ``grid_thw``,
        ``visual_pos_masks``, ``pixel_features``.
        """
        import warnings

        from flash_rt.models.groot_n17.calibration import build_vit_rope_tables
        from flash_rt.models.groot_n17.pipeline_orin import _rope_half_table

        if hasattr(self, "_backbone_features"):
            raise RuntimeError(
                "set_prompt() after prompt init is not supported; construct a "
                "new frontend instance for a new prompt")

        device = self.device
        self._prompt = prompt

        self._mrope_cos = aux["rope_cos"][0].to(device).to(_BF16).contiguous()
        self._mrope_sin = aux["rope_sin"][0].to(device).to(_BF16).contiguous()

        grid_thw = [tuple(int(x) for x in row) for row in aux["grid_thw"].tolist()]
        vit_cos, vit_sin = build_vit_rope_tables(
            grid_thw, head_dim=64, theta=10000.0, spatial_merge_size=2,
            device=device)
        # build_vit_rope_tables returns fp16 (its only consumer until now was
        # the FP16 kernel); the bf16 rope shim needs bf16 tables.
        self._vit_cos = vit_cos.to(_BF16).contiguous()
        self._vit_sin = vit_sin.to(_BF16).contiguous()

        # Half-width rope tables for the fused bf16 rotate-half kernel
        # (pipeline_orin._rope_qk). Built here, once: they are a property of the
        # prompt, not of the frame, and the backbone graph bakes their pointers.
        # _rope_half_table verifies the duplicated-halves invariant the kernel
        # depends on and returns None if it does not hold, which routes that
        # site back to the torch shim.
        self._mrope_cos_half = _rope_half_table(self._mrope_cos, "rope_cos")
        self._mrope_sin_half = _rope_half_table(self._mrope_sin, "rope_sin")
        self._vit_cos_half = _rope_half_table(self._vit_cos, "vit rope cos")
        self._vit_sin_half = _rope_half_table(self._vit_sin, "vit rope sin")
        self._num_vit_views = len(grid_thw)
        self._S_vit = sum(int(t * h * w) for t, h, w in grid_thw)
        self._visual_pos_masks = aux["visual_pos_masks"][0].to(device)

        # ── per-prompt fusion constants ──
        # Everything here is a property of the (prompt, camera setup), not of
        # the frame, so it is built once and baked into the graph as pointers.
        if self._fuse_image_embeds:
            missing = [k for k in ("pixel_values", "input_ids", "grid_thw",
                                   "visual_pos_masks") if k not in aux]
            if missing:
                raise KeyError(
                    f"fuse_image_embeds=True needs {missing} in aux: the "
                    "frontend builds the LLM's fused text+image input embeds "
                    "itself, so it needs the raw patch matrix and the token "
                    "ids. Either supply them or construct with "
                    "fuse_image_embeds=False to take them from an HF forward.")
            from flash_rt.frontends.torch._groot_n17_fusion import (
                fast_pos_embed_interpolate,
            )
            self.Se = int(aux["input_ids"].reshape(-1).shape[0])
            # read by pointer into embedding_lookup_bf16, so it must be
            # contiguous; the pixel matrix itself is copied per observation into
            # the backbone runtime's persistent buffer (_kbb_load_inputs).
            self._fus_ids = aux["input_ids"].reshape(-1).to(
                torch.int64).to(device).contiguous()
            self._fus_pos = fast_pos_embed_interpolate(
                self._vit_pos_embed, grid_thw, device=device).to(
                    _BF16).contiguous()
            # Patch embed is a Conv3d(3, 1024, (2,16,16), stride (2,16,16)),
            # which on an already-flattened (Sv, 1536) patch matrix is exactly
            # a (Sv,1536) @ (1536,1024) GEMM + bias. The checkpoint stores the
            # weight as (1024,3,2,16,16); flatten and transpose to [K, N] for
            # gemm.bf16_nn. Merger fc1/fc2 already arrive as [K, N] from the
            # spec (verified: both equal raw.T, and fc1 is square so a wrong
            # guess there would be silent).
            self._fus_pe_w = self._patch_embed_w.reshape(1024, 1536).t().contiguous()
            self._fus_pe_b = self._patch_embed_b.contiguous()
            self._fus_mg_fc1_w = self._merger_fc1_w.contiguous()
            self._fus_mg_fc2_w = self._merger_fc2_w.contiguous()
            self._fus_emb = self._embed_tokens_w.contiguous()
        else:
            if "llm_input_embeds" not in aux:
                raise KeyError(
                    "aux['llm_input_embeds'] is required when "
                    "fuse_image_embeds=False: in that mode the vision tower is "
                    "run only up to the last DeepStack tap and the LLM's fused "
                    "text+image input embeddings must come from the caller.")
            self.Se = int(aux["llm_input_embeds"].shape[1])

        # .clone(), not .to(_BF16): the backbone now returns a view of its
        # persistent vlsa_h buffer, and a dtype-preserving .to() is a no-op that
        # would leave _backbone_features aliasing a buffer the next backbone
        # call overwrites in place.
        self._backbone_features = self._run_kernel_backbone(aux).clone()

        try:
            self._warmup_infer()
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"set_prompt warmup failed (non-fatal): {e!r}")
        self.latency_records.clear()
        self._observation_contract = self._snapshot_observation_contract(aux)

    # ────────────────────────────────────────────────────────────────
    # Per-observation contract (continuous inference)
    # ────────────────────────────────────────────────────────────────

    #: aux keys whose *values* are baked into the backbone runtime and the
    #: captured DiT graphs. A new observation may change what the cameras saw;
    #: it may not change the prompt, the token layout, or the rope tables.
    _OBS_METADATA_KEYS = ("grid_thw", "visual_pos_masks", "rope_cos", "rope_sin")

    def _observation_slots(self) -> tuple:
        """The aux keys whose *contents* legitimately change every frame."""
        return (("pixel_values",) if self._fuse_image_embeds
                else ("pixel_features", "llm_input_embeds"))

    def _snapshot_observation_contract(self, aux: dict) -> dict:
        """Record the metadata this frontend's graphs are keyed to.

        ``input_ids`` joins the pinned set only in fused mode: ``set_prompt``
        copies it into ``_fus_ids`` once and the fusion reads that buffer by
        pointer forever after, so a changed token layout would be ignored
        rather than rejected.
        """
        keys = list(self._OBS_METADATA_KEYS)
        if self._fuse_image_embeds:
            keys.append("input_ids")
        missing = [k for k in keys if k not in aux]
        if missing:
            raise ValueError(
                "observation aux is missing required keys: "
                + ", ".join(missing))

        def snapshot(name):
            source = torch.as_tensor(aux[name])
            return {
                "value": source.detach().cpu().clone(),
                "source": source,
                "version": _mutation_version(source),
                "validated_source": None,
                "validated_version": None,
            }

        contract = {k: snapshot(k) for k in keys}
        for name in self._observation_slots():
            if name not in aux:
                raise ValueError(
                    f"observation aux is missing required key: {name}")
            contract[name + "_shape"] = tuple(aux[name].shape)
        return contract

    def _validate_observation_contract(self, aux: dict) -> None:
        """Reject a fresh observation whose graph-owned metadata changed.

        Shape drift in the observation slots is checked here for the message;
        the backbone runtime re-checks it against its own key and raises too.
        """
        expected = self._observation_contract
        for name in self._observation_slots():
            if name not in aux:
                raise ValueError(f"observation aux is missing key: {name}")
            shape_key = name + "_shape"
            actual = tuple(aux[name].shape)
            if actual != expected[shape_key]:
                raise ValueError(
                    f"this frontend's persistent buffers require {name} shape "
                    f"{expected[shape_key]}, got {actual}; construct a new "
                    "frontend for a changed image or sequence length")

        for name, entry in expected.items():
            if name.endswith("_shape"):
                continue
            if name not in aux:
                raise ValueError(f"observation aux is missing key: {name}")
            source = torch.as_tensor(aux[name])
            version = _mutation_version(source)
            # Identity + PyTorch's mutation version keeps the steady-state path
            # off the host: the rope tables alone are ~2.4 MB per observation,
            # and comparing them on Orin's ARM CPU would show up in the
            # per-frame latency this path exists to reduce. ``version is None``
            # (an inference tensor) must not take it — identity alone cannot
            # prove the caller did not mutate the tensor in place, and the
            # memoization below would then cache that unverdict forever.
            if version is not None and (
                    (source is entry["source"]
                     and version == entry["version"])
                    or (source is entry["validated_source"]
                        and version == entry["validated_version"])):
                continue
            if version is None:
                _note_inference_tensor("_validate_observation_contract")
            actual = source.detach().cpu()
            reference = entry["value"]
            if actual.dtype != reference.dtype or not torch.equal(
                    actual, reference):
                raise ValueError(
                    f"continuous inference requires {name} to match the "
                    "set_prompt() metadata; construct a new frontend for a "
                    "changed prompt or camera setup")
            entry["validated_source"] = source
            entry["validated_version"] = version

    # ────────────────────────────────────────────────────────────────
    # Raw camera frames → pixel_values (GPU image path)
    # ────────────────────────────────────────────────────────────────

    def _image_plan(self, height: int, width: int):
        """The cached image plan for one camera geometry, built on first use.

        A plan holds two dense fp32 operator matrices (256x640 and 256x243 at the
        shipped geometry) plus the letterbox padding — all a property of the
        camera setup, not of the frame. So it is keyed on the geometry and
        reused, the same lifecycle as the rope half-tables ``set_prompt`` builds.

        A *changed* geometry gets a new plan rather than an error: re-deriving is
        cheap next to one observation and cannot silently produce a wrong image.
        What the new plan implies for the row count **is** checked, in
        :meth:`_pixel_values_from_frames`, and that is where a camera setup this
        prompt was not built for gets refused.
        """
        from flash_rt.frontends.torch._groot_n17_preprocess import (
            build_image_plan,
        )

        geom = self._read_processor_geometry()
        key = (int(height), int(width), geom["shortest"], geom["crop_fraction"])
        cache = getattr(self, "_image_plans", None)
        if cache is None:
            cache = self._image_plans = {}
        plan = cache.get(key)
        if plan is None:
            plan = build_image_plan(height, width, device=self.device,
                                    shortest=geom["shortest"],
                                    crop_fraction=geom["crop_fraction"])
            cache[key] = plan
        return plan

    def _frames_to_device(self, frames) -> torch.Tensor:
        """Normalize a raw ``frames=`` argument to ``(V, H, W, 3)`` uint8 on the
        device, staging a host upload through a persistent pinned buffer.

        Accepts the three forms a caller actually has: a ``(V,H,W,3)`` tensor,
        the same as an ndarray, or a sequence of per-view ``(H,W,3)`` arrays —
        what a two-camera loop naturally accumulates and what the vendor's own
        ``video.<view>`` observation holds.

        A *dict* of views is refused rather than iterated. ``pixel_values`` rows
        are ordered by view, and a dict's key order is not a camera order this
        frontend can check against the prompt's; guessing it would serve the
        wrist camera where the front camera belongs and still produce a
        well-formed action.

        The host path goes through a pinned buffer (``hyvla_rtx.py``'s staging
        pattern) rather than a pageable ``.cuda()``: measured ~0.5 ms cheaper per
        observation here, which is a third of this whole path's cost. Both the
        pinned and the device buffer persist, so the steady state is one
        ``copy_`` and one DMA per frame with no allocator traffic.
        """
        import numpy as np

        if isinstance(frames, dict):
            raise ValueError(
                "frames= takes a (views,H,W,3) tensor/ndarray or a *sequence* of "
                "per-view (H,W,3) arrays, not a dict: pixel_values rows are "
                "ordered by view and a dict's key order is not a camera order "
                "this frontend can verify against the prompt. Pass the views in "
                "the order set_prompt's pixel_values used (the prompt's "
                "grid_thw order).")
        if isinstance(frames, np.ndarray):
            t = torch.from_numpy(frames)
        elif isinstance(frames, torch.Tensor):
            t = frames
        elif isinstance(frames, (list, tuple)):
            t = torch.stack([torch.from_numpy(f) if isinstance(f, np.ndarray)
                             else f for f in frames])
        else:
            raise ValueError(
                f"frames= must be a (views,H,W,3) tensor/ndarray or a sequence "
                f"of (H,W,3) arrays, got {type(frames).__name__}")
        if t.dim() != 4 or t.shape[-1] != 3:
            raise ValueError(
                f"frames= must be (views, H, W, 3), got {tuple(t.shape)}")
        if t.dtype != torch.uint8:
            raise ValueError(
                f"frames= must be uint8 camera bytes, got {t.dtype}: the vendor "
                "chain resizes integers and only then rescales, so a "
                "pre-normalized float frame would be transformed twice")
        t = t.contiguous()
        if t.device.type != "cpu":
            return t if t.device == self.device else t.to(self.device)
        if torch.device(self.device).type == "cpu":
            # No transfer to stage, and pin_memory is not allocatable without a
            # CUDA context — so the CPU contract tests can exercise this path.
            return t

        pin = getattr(self, "_frames_pin", None)
        if pin is None or tuple(pin.shape) != tuple(t.shape):
            pin = torch.empty(tuple(t.shape), dtype=torch.uint8, pin_memory=True)
            self._frames_pin = pin
        buf = getattr(self, "_frames_dev", None)
        if buf is None or tuple(buf.shape) != tuple(t.shape):
            buf = torch.empty(tuple(t.shape), dtype=torch.uint8,
                              device=self.device)
            self._frames_dev = buf
        pin.copy_(t)
        buf.copy_(pin, non_blocking=True)
        return buf

    def _pixel_values_from_frames(self, frames) -> torch.Tensor:
        """Raw camera frames → this prompt's ``pixel_values``, on the GPU.

        The whole point of the ``frames=`` arm: **12.916-14.027 ms** of
        per-observation host work on Orin's ARM CPU (the vendor's own chain,
        measured on 4 real frames across both datasets, clock-locked, median of
        11) becomes **1.448-1.485 ms** of device work. That host time is
        serialized *before* the model runs, so removing it removes wall clock
        directly — this is not a submission-overlap win, and §6.13.3 withdrew
        the "CPU submission bound" reading that would have made it one.

        Runs **outside** every CUDA graph. The plan and its operators are cached,
        but the frames are per-observation, and §6.17's correction is exactly
        that a per-observation source must not be baked into a capture.

        Returns fp32; the bf16 cast happens in ``_kbb_load_inputs``, which is
        also what makes the result match the HF processor bit for bit (see
        ``_groot_n17_preprocess.patchify``).
        """
        from flash_rt.frontends.torch._groot_n17_preprocess import (
            frames_to_pixel_values,
        )

        if not self._fuse_image_embeds:
            raise ValueError(
                "frames= needs fuse_image_embeds=True: without the fusion the "
                "backbone consumes pixel_features *and* llm_input_embeds, and "
                "the latter is an LLM forward over the prompt's text embeddings "
                "— not something an image path can produce. Either pass an aux "
                "bundle, or construct with fuse_image_embeds=True (the "
                "default).")

        frames = self._frames_to_device(frames)
        views, height, width, _ = frames.shape
        plan = self._image_plan(height, width)
        pv = frames_to_pixel_values(frames, plan)

        expected = self._observation_contract.get("pixel_values_shape")
        if expected is not None and tuple(pv.shape) != tuple(expected):
            raise ValueError(
                f"{views} frames of {height}x{width} produce pixel_values "
                f"{tuple(pv.shape)}, but this prompt was set with "
                f"{tuple(expected)}. The row count is views * "
                "(shortest_image_edge/16)**2, so this is a different camera "
                "count or resolution than the prompt was built for; construct a "
                "new frontend for it.")
        return pv

    def _observation_aux(self, frames, aux: dict | None) -> dict:
        """Build the per-observation aux bundle for the ``frames=`` arm.

        The prompt-scoped keys (``grid_thw``, ``visual_pos_masks``,
        ``rope_cos``/``rope_sin``, plus ``input_ids`` in fused mode) are
        re-presented **by identity** — the same tensor objects ``set_prompt``
        recorded — so ``_validate_observation_contract``'s identity+version fast
        path is taken instead of a host compare per key per frame.

        Measured on this checkpoint (Se=148), counting the validator's
        ``torch.equal`` calls: the ``frames=`` arm makes **0** and compares **0**
        bytes, on every observation including the first. The ``aux=`` arm makes
        **5** and compares **77156** bytes *every* frame, because a caller that
        rebuilds its bundle per observation — which is what an
        HF-processor-per-frame flow does — hands over fresh tensor objects that
        cannot match on identity. This arm cannot miss the fast path, since it
        sources the tensors from the contract rather than from the caller.

        Handing the validator copies would be numerically identical and would put
        that CPU cost straight back, which is the cost this path exists to
        remove.

        ``aux`` may still be passed alongside ``frames`` to override the other
        prompt-scoped keys, but not to carry ``pixel_values``: two sources for
        the one tensor that genuinely changes per observation is ambiguous, and
        silently picking one is how a stale frame gets served.
        """
        if aux is not None and "pixel_values" in aux:
            raise ValueError(
                "frames= and aux['pixel_values'] were both given. They are two "
                "answers to the same question and this frontend will not choose "
                "between them: pass frames= to run the GPU image path, or pass "
                "an aux bundle whose pixel_values is already processed.")
        base = dict(aux) if aux is not None else {}
        for name, entry in self._observation_contract.items():
            if name.endswith("_shape") or name in base:
                continue
            base[name] = entry["source"]
        base["pixel_values"] = self._pixel_values_from_frames(frames)
        return base

    def _backbone_weight_dicts(self):
        """Per-layer weight-pointer dicts for the kernel backbone, built once.

        Assembling these inline on every ``_run_kernel_backbone`` call (as the
        FP16 frontend does) re-slices the fused ViT ``qkv`` weight and bias
        into contiguous per-projection tensors each time: ~120 large copies per
        call, measured at 15.65 ms of a 77 ms backbone. Nothing here depends on
        the prompt, so it is cached for the frontend's lifetime and the
        intermediates are kept alive in ``self._bb_keep`` — a data_ptr() whose
        tensor has been freed is a silent use-after-free, not an error.

        ``lw`` deliberately omits ``deepstack_inject``, which is per-prompt.
        """
        if hasattr(self, "_bb_weights"):
            return self._bb_weights
        sh = self._bf16_shadow_weights
        keep: list = []
        self._bb_keep = keep

        def split3(b: torch.Tensor) -> list:
            n = b.shape[0] // 3
            if b.shape[0] != 3 * n:
                raise RuntimeError(
                    f"fused qkv bias has {b.shape[0]} entries, not divisible "
                    f"by 3")
            out = [b[i * n:(i + 1) * n].contiguous() for i in range(3)]
            keep.extend(out)
            return out

        # ── ViT (only the layers up to the last DeepStack tap run) ──
        vw = {k: [] for k in (
            "norm1_w", "norm1_b", "norm2_w", "norm2_b", "q_w", "q_b",
            "k_w", "k_b", "v_w", "v_b", "o_w", "o_b", "fc1_w", "fc1_b",
            "fc2_w", "fc2_b")}
        for li in range(self._vit_layers):
            qb, kb, vb = split3(self._vit_qkv_b[li])
            for key, val in (
                ("norm1_w", self._vit_ln1_w[li]), ("norm1_b", self._vit_ln1_b[li]),
                ("norm2_w", self._vit_ln2_w[li]), ("norm2_b", self._vit_ln2_b[li]),
                ("q_w", sh[("vit", li, "q")]), ("q_b", qb),
                ("k_w", sh[("vit", li, "k")]), ("k_b", kb),
                ("v_w", sh[("vit", li, "v")]), ("v_b", vb),
                ("o_w", sh[("vit", li, "o")]), ("o_b", self._vit_o_b[li]),
                ("fc1_w", sh[("vit", li, "fc1")]), ("fc1_b", self._vit_fc1_b[li]),
                ("fc2_w", sh[("vit", li, "fc2")]), ("fc2_b", self._vit_fc2_b[li]),
            ):
                vw[key].append(val.data_ptr())

        # ── DeepStack mergers ──
        dsw = {k: [] for k in ("norm_w", "norm_b", "fc1_w", "fc1_b",
                               "fc2_w", "fc2_b")}
        for j in range(3):
            dsw["norm_w"].append(getattr(self, f"_dsm{j}_norm_w").data_ptr())
            dsw["norm_b"].append(getattr(self, f"_dsm{j}_norm_b").data_ptr())
            dsw["fc1_w"].append(sh[("dsm", j, "fc1")].data_ptr())
            dsw["fc1_b"].append(getattr(self, f"_dsm{j}_fc1_b").data_ptr())
            dsw["fc2_w"].append(sh[("dsm", j, "fc2")].data_ptr())
            dsw["fc2_b"].append(getattr(self, f"_dsm{j}_fc2_b").data_ptr())

        # ── Truncated LLM (16L) ──
        lw = {k: [] for k in (
            "in_ln_w", "post_ln_w", "q_norm_w", "k_norm_w", "q_w", "k_w",
            "v_w", "o_w", "gate_w", "up_w", "down_w")}
        for li in range(16):
            for key, val in (
                ("in_ln_w", self._llm_input_ln_w[li]),
                ("post_ln_w", self._llm_post_ln_w[li]),
                ("q_norm_w", self._llm_q_norm_w[li]),
                ("k_norm_w", self._llm_k_norm_w[li]),
            ):
                lw[key].append(val.data_ptr())
            for nm in ("q", "k", "v", "o", "gate", "up", "down"):
                lw[f"{nm}_w"].append(sh[("llm", li, nm)].data_ptr())

        # ── VL self-attention (4L) ──
        vsw = {k: [] for k in (
            "norm1_w", "norm1_b", "norm3_w", "norm3_b", "q_w", "q_b",
            "k_w", "k_b", "v_w", "v_b", "o_w", "o_b", "fc1_w", "fc1_b",
            "fc2_w", "fc2_b")}
        for li in range(4):
            for key, val in (
                ("norm1_w", self._vlsa_norm1_w[li]),
                ("norm1_b", self._vlsa_norm1_b[li]),
                ("norm3_w", self._vlsa_norm3_w[li]),
                ("norm3_b", self._vlsa_norm3_b[li]),
            ):
                vsw[key].append(val.data_ptr())
            for nm in ("q", "k", "v", "o", "fc1", "fc2"):
                vsw[f"{nm}_w"].append(sh[("vlsa", li, nm)].data_ptr())
                vsw[f"{nm}_b"].append(
                    getattr(self, f"_vlsa_{nm}_b")[li].data_ptr())

        self._bb_weights = (vw, lw, vsw, dsw)
        return self._bb_weights

    # ────────────────────────────────────────────────────────────────
    # Kernel backbone: persistent runtime + pure kernel forward
    # ────────────────────────────────────────────────────────────────

    def _build_backbone_runtime(self) -> dict:
        """Allocate every backbone buffer once and hoist its attention backend.

        ``_run_kernel_backbone`` used to rebuild all of this on every call:
        ~30 ``torch.empty``, a fresh :class:`OrinGrootN17BackboneAttn` (which
        allocates the LLM K/V slots and the LSE slab), and five kernel-argument
        dicts. That is pure CPU work on a stage measured at 91-92% CPU
        submission (docs/groot_n17_orin_sm87.md §6.11.5), and it is also what
        made the stage uncapturable — a CUDA graph replays fixed pointers, so
        every buffer has to exist before ``capture_begin``.

        Two pointer classes are baked in here:

        * **weights** — checkpoint-level and immutable, kept alive by
          ``_bf16_shadow_weights`` / ``_bb_keep``;
        * **prompt tensors** — the interpolated positional embeds, token ids,
          embedding table and both RoPE tables. ``set_prompt`` refuses a second
          call, so these cannot change within an instance's life; a *shape*
          change raises instead of silently replaying stale pointers.

        Only the observation is copied per call, by :meth:`_kbb_load_inputs`,
        into buffers owned here.
        """
        key = (self._S_vit, self.Se, self._num_vit_views,
               bool(self._fuse_image_embeds), self._vit_layers)
        rt = getattr(self, "_kbb_rt", None)
        if rt is not None:
            if rt["key"] != key:
                raise RuntimeError(
                    "the backbone runtime was built for (S_vit, Se, views, "
                    f"fuse, vit_layers)={rt['key']} but the frontend now "
                    f"describes {key}; the persistent buffers bake these shapes "
                    "in, so construct a new frontend for a new shape")
            return rt
        if self._fuse_image_embeds and not hasattr(self, "_fus_ids"):
            raise RuntimeError(
                "fusion constants are missing; set_prompt() must run before "
                "_run_kernel_backbone() when fuse_image_embeds=True")

        import flash_rt.flash_rt_kernels as fvk
        from flash_rt.hardware.rtx.attn_backend_groot_n17_orin import (
            OrinGrootN17BackboneAttn,
        )

        if not hasattr(self, "_gemm"):
            self._fvk = fvk
            self._gemm = fvk.GemmRunner()
        dev = self.device
        Sv, nv, Se = self._S_vit, self._num_vit_views, self.Se
        Nout = Sv // 4
        fuse = bool(self._fuse_image_embeds)
        vw, lw, vsw, dsw = self._backbone_weight_dicts()

        keep: list = []

        def buf(*shape, dtype=_BF16):
            t = torch.empty(*shape, dtype=dtype, device=dev)
            keep.append(t)
            return t

        attn = OrinGrootN17BackboneAttn(
            num_vit_views=nv, vit_seq=Sv, llm_seq=Se, vl_self_attn_seq=Se,
            device=dev)
        slots = attn.get_slot_ptrs("llm")

        vit_h, llm_h, vlsa_h = buf(Sv, 1024), buf(Se, 2048), buf(Se, 2048)
        gate_out, up_out = buf(Se, 6144), buf(Se, 6144)
        inject = [buf(Se, 2048) for _ in range(3)]
        ds_out = [buf(Nout, 2048) for _ in range(3)]
        # Allocated for every layer that runs, but only the layers named in
        # ``deepstack_taps`` get a callback, so the production path pays for 3
        # gpu_copy calls and the per-layer debug path for ``_vit_layers``.
        tap_bufs = {layer: buf(Sv, 1024) for layer in range(self._vit_layers)}
        vis_idx = self._visual_pos_masks.reshape(-1).nonzero(
            as_tuple=True)[0].to(torch.long).contiguous()

        self._kbb_rt = rt = {
            "key": key, "keep": keep, "attn": attn, "vis_idx": vis_idx,
            "Sv": Sv, "Se": Se, "Nout": Nout, "fuse": fuse,
            "gemm": self._gemm, "fvk": self._fvk,
            "vit_h": vit_h, "llm_h": llm_h, "vlsa_h": vlsa_h,
            "tap_bufs": tap_bufs, "ds_out": ds_out, "inject": inject,
            "mg_norm": buf(Sv, 1024), "mg_fc1": buf(Nout, 4096),
            "mg_img": buf(Nout, 2048),
            # the only per-observation inputs; rewritten by _kbb_load_inputs.
            # Non-fused mode needs its own embeds buffer because ``llm_h`` is
            # the LLM's residual stream and is overwritten in place by the
            # forward; ``pf``/``pv`` are likewise distinct from ``vit_h``.
            "pv": buf(Sv, 1536) if fuse else None,
            "pf": None if fuse else buf(Sv, 1024),
            "llm_in": None if fuse else buf(Se, 2048),
            # (source tensor, mutation version) per slot, for the skip-if-
            # unchanged fast path in _kbb_load_inputs
            "loaded": {},
            "vit": dict(
                bufs={"h": vit_h.data_ptr(), "xn": buf(Sv, 1024).data_ptr(),
                      "o_proj_out": buf(Sv, 1024).data_ptr(),
                      "fc1_out": buf(Sv, 4096).data_ptr()},
                tbufs={"Q": attn.vit_Q, "K": attn.vit_K,
                       "cos": self._vit_cos, "sin": self._vit_sin,
                       "cos_half": self._vit_cos_half,
                       "sin_half": self._vit_sin_half},
                weights=vw,
                dims={"S": Sv, "D": 1024, "NH": 16, "HD": 64,
                      "ff_inner": 4096, "Sper_view": Sv // nv}),
            "deepstack": dict(
                bufs={"in": [tap_bufs[layer].data_ptr()
                             for layer in self._DEEPSTACK_TAPS],
                      "ln_out": buf(Nout, 4096).data_ptr(),
                      "fc1_out": buf(Nout, 4096).data_ptr(),
                      "out": [t.data_ptr() for t in ds_out]},
                weights=dsw,
                dims={"Nin": Sv, "Din": 1024, "Nout": Nout, "Dmid": 4096,
                      "Dout": 2048}),
            "llm": dict(
                bufs={"h": llm_h.data_ptr(), "xn": buf(Se, 2048).data_ptr(),
                      "Q": slots["Q"], "K": slots["K"], "V": slots["V"],
                      "o_proj_out": buf(Se, 2048).data_ptr(),
                      "gate_out": gate_out.data_ptr(),
                      "up_out": up_out.data_ptr()},
                # shallow copy: the cached weight dict must never hold the
                # prompt-dependent inject pointers
                weights=dict(lw, deepstack_inject=[t.data_ptr()
                                                   for t in inject]),
                tbufs={"Q": attn.llm_Q, "K": attn.llm_K,
                       "cos": self._mrope_cos, "sin": self._mrope_sin,
                       "cos_half": self._mrope_cos_half,
                       "sin_half": self._mrope_sin_half,
                       "gate": gate_out, "up": up_out},
                dims={"S": Se, "D": 2048, "NHQ": 16, "NHKV": 8, "HD": 128,
                      "FF": 6144}),
            "vlln": dict(
                bufs={"x": llm_h.data_ptr(), "out": vlsa_h.data_ptr()},
                weights={"vlln_w": self._vlln_w.data_ptr(),
                         "vlln_b": self._vlln_b.data_ptr()},
                dims={"S": Se, "D": 2048}),
            "vlsa": dict(
                bufs={"h": vlsa_h.data_ptr(), "xn": buf(Se, 2048).data_ptr(),
                      "o_proj_out": buf(Se, 2048).data_ptr(),
                      "fc1_out": buf(Se, 8192).data_ptr()},
                weights=vsw,
                dims={"T": Se, "D": 2048, "NH": 32, "HD": 64,
                      "ff_inner": 8192}),
        }
        return rt

    def _kbb_load_inputs(self, aux: dict) -> None:
        """Copy this observation into the runtime's persistent input buffers.

        Kept out of :meth:`_kbb_forward` on purpose: a CUDA graph replays fixed
        kernels over fixed pointers, so anything that varies per observation has
        to be written into a buffer the forward already reads.

        A source tensor that is the *same object*, unmutated since the last
        call, is skipped: ``aux`` normally arrives from the HF processor on the
        host, so re-copying it costs a host-to-device transfer plus a bf16
        conversion (~1.5 ms measured on a 512-patch frame) for bytes that are
        already resident. Identity plus PyTorch's mutation counter is the fast
        path the RTX FP8 backbone-graph contract uses for the same reason; a
        different tensor object, or an in-place edit of this one, still copies.
        An inference tensor has no readable counter, so it always copies: the
        skip would otherwise be decided by object identity alone, and a caller
        that reuses one buffer across observations would be served the first
        frame forever.
        """
        rt = self._build_backbone_runtime()
        dev = self.device
        loaded = rt["loaded"]

        def load(key, slot, shape, why):
            if key not in aux:
                raise KeyError(f"aux['{key}'] is required: {why}")
            src = aux[key]
            version = _mutation_version(src)
            prev = loaded.get(slot)
            if (version is not None and prev is not None
                    and prev[0] is src and prev[1] == version):
                return
            if version is None:
                _note_inference_tensor("_kbb_load_inputs")
            rt[slot].copy_(src.to(dev).to(_BF16).reshape(*shape))
            loaded[slot] = (src, version)

        if rt["fuse"]:
            load("pixel_values", "pv", (rt["Sv"], 1536),
                 "with fuse_image_embeds=True the frontend builds the LLM's "
                 "fused text+image input embeds itself, so it needs the raw "
                 "patch matrix and the token ids")
            return
        why = ("with fuse_image_embeds=False the vision tower runs only up to "
               "the last DeepStack tap and the LLM's fused text+image input "
               "embeddings must come from the caller")
        load("pixel_features", "pf", (rt["Sv"], 1024), why)
        load("llm_input_embeds", "llm_in", (rt["Se"], 2048), why)

    def _kbb_forward(self, stream: int = 0, snap_into: dict | None = None,
                     vit_layers: bool = False) -> None:
        """ViT -> DeepStack -> LLM -> vlln -> VL-self-attn. Pure kernels.

        Returns ``backbone_features`` in ``rt["vlsa_h"]``, shape ``(Se, 2048)``.
        No PyTorch matmul on the feature path; the only torch ops are the SM87
        kernel gaps documented in
        :mod:`flash_rt.models.groot_n17.pipeline_orin` (rotate-half RoPE and the
        SiLU-gate multiply), the fusion's ``conv3d`` patch embed, and two
        ``index_copy_`` scatters.

        This body is what a CUDA graph would record, so it allocates nothing,
        rebuilds no argument dict, reads nothing back to the host and does not
        synchronize; anything added here has to hold the same contract.

        ``snap_into`` (debug only — a host readback is not capturable) is filled
        with fp32 CPU clones of the stage-boundary tensors (``vit_h``,
        ``llm_h``, ``vlln_out``, ``vlsa_h``, ``llm_input_embeds``,
        ``merger_out``, ``deepstack_out_j``) so a precision failure can be
        localized to one stage instead of only showing up as a whole-backbone
        cosine. It costs nothing when None. ``vit_layers`` widens the ViT tap
        set to every layer that runs and adds ``vit_block_i`` for each.
        """
        from flash_rt.models.groot_n17 import pipeline_orin as P

        rt = self._build_backbone_runtime()
        gemm, fvkm, attn = rt["gemm"], rt["fvk"], rt["attn"]
        Sv, Se, Nout, fuse = rt["Sv"], rt["Se"], rt["Nout"], rt["fuse"]

        def snap(name, t):
            if snap_into is not None:
                snap_into[name] = t.detach().float().cpu().clone()

        # ═══ ViT ═══
        vit_h = rt["vit_h"]
        if fuse:
            # patch embed: torch's conv3d with the checkpoint's own
            # (1024,3,2,16,16) weight, in the checkpoint's own dtype (bf16).
            #
            # HF runs exactly this op, so both sides land on the same cuDNN
            # path and the result is bit-identical. That matters more than it
            # looks: this tensor feeds a 24-layer residual tower that
            # measurably amplifies ~1 ULP/layer (see
            # docs/groot_n17_orin_sm87.md 6.6.5). Three attempts, measured:
            #   bf16 GEMM kernel     -> vlsa_h 0.995323
            #   fp32 matmul, rounded -> vlsa_h 0.996077, vit_block_17 0.991470
            #   F.conv3d (this)      -> vlsa_h 0.996951, vit_block_17 0.998156,
            #                           patch embed max|d| == 0
            # A flattened GEMM is mathematically equivalent but accumulates in a
            # different order, so "more precise" is NOT "closer to HF" here --
            # and HF is what the gates compare against.
            #
            # torch logs "Plan failed with a cudnnException ... NOT_SUPPORTED"
            # on the first call and falls back to a legacy algorithm. Benign,
            # and the evidence is the measurement: the result is bit-identical
            # to HF's captured pixel_features (max|d| == 0), so both sides
            # resolve to the same algorithm -- they make the identical call in
            # the identical torch build. Cost is one op per observation on
            # (512,3,2,16,16).
            pe = torch.nn.functional.conv3d(
                rt["pv"].view(Sv, 3, 2, 16, 16),
                self._patch_embed_w, self._patch_embed_b,
                stride=(2, 16, 16)).view(Sv, 1024)
            vit_h.copy_(pe + self._fus_pos)
        else:
            vit_h.copy_(rt["pf"])

        tap_layers = self._DEEPSTACK_TAPS
        # Per-layer capture widens the tap set to every layer that runs; the
        # merger inputs are then just the tap-layer snapshots. pipeline_orin is
        # unchanged either way - it only calls back on the taps it is given.
        snap_layers = (tuple(range(self._vit_layers)) if vit_layers
                       else tap_layers)
        tap_bufs = rt["tap_bufs"]

        def mk_cb(layer):
            def cb(h_ptr):
                fvkm.gpu_copy(tap_bufs[layer].data_ptr(), int(h_ptr),
                              Sv * 1024 * 2, stream)
                if vit_layers:
                    snap_into[f"vit_block_{layer}"] = (
                        tap_bufs[layer].detach().float().cpu().clone())
            return cb

        P.qwen3vl_vit_forward(
            gemm=gemm, fvk=fvkm, attn=attn, stream=stream,
            deepstack_taps=snap_layers,
            deepstack_capture=[mk_cb(layer) for layer in snap_layers],
            layers_subset=range(self._vit_layers), **rt["vit"])
        snap("vit_h", vit_h)

        # ═══ DeepStack (3 mergers) ═══
        P.deepstack_merge_forward(gemm=gemm, fvk=fvkm, stream=stream,
                                  **rt["deepstack"])
        for j in range(3):
            snap(f"deepstack_out_{j}", rt["ds_out"][j])

        # DeepStack inject buffers (Se, D) - zero except at visual positions.
        # ``index_copy_`` over a precomputed index rather than the boolean-mask
        # assignment this replaced: ``ib[mask] = x`` is a data-dependent shape,
        # so it runs a ``nonzero()`` that synchronizes with the host - three
        # pipeline drains per call, and uncapturable. Same values either way.
        for j in range(3):
            ib = rt["inject"][j]
            ib.zero_()
            ib.index_copy_(0, rt["vis_idx"], rt["ds_out"][j])

        # ═══ LLM (16L, causal GQA, M-RoPE) ═══
        llm_h = rt["llm_h"]
        if fuse:
            # Text tokens: in-kernel embed lookup over the prompt's token ids.
            # Image positions hold placeholder embeds that the merger output
            # overwrites below. ``llm_h`` itself is the persistent buffer the
            # caller loaded (non-fused mode) or that this branch fills.
            fvkm.embedding_lookup_bf16(
                self._fus_ids.data_ptr(), self._fus_emb.data_ptr(),
                llm_h.data_ptr(), Se, 2048, stream)
            # ViT final merger: LayerNorm(1024) -> (Nout,4096) -> fc1 + GELU
            # -> fc2 -> (Nout,2048), then scatter over the visual positions.
            # The LayerNorm runs on the (Sv,1024) layout BEFORE the 2x2
            # shuffle, because Qwen3VLVisionPatchMerger is built with
            # use_postshuffle_norm=False (its norm is (1024,), not (4096,)).
            # act_fn is nn.GELU() == exact erf, hardcoded -- it does NOT read
            # vision_config.hidden_act ("gelu_pytorch_tanh"), which applies
            # only to the vision blocks. Using the tanh kernel here costs
            # ~1e-3 cos per merger and compounds into every image token.
            mg_norm, mg_fc1, mg_img = rt["mg_norm"], rt["mg_fc1"], rt["mg_img"]
            fvkm.layer_norm(vit_h.data_ptr(), self._merger_norm_w.data_ptr(),
                            self._merger_norm_b.data_ptr(), mg_norm.data_ptr(),
                            Sv, 1024, 1e-6, stream)
            gemm.bf16_nn(mg_norm.data_ptr(), self._fus_mg_fc1_w.data_ptr(),
                         mg_fc1.data_ptr(), Nout, 4096, 4096, stream)
            fvkm.add_bias_bf16(mg_fc1.data_ptr(),
                               self._merger_fc1_b.data_ptr(), Nout, 4096,
                               stream)
            fvkm.gelu_erf_bf16(mg_fc1.data_ptr(), Nout * 4096, stream)
            gemm.bf16_nn(mg_fc1.data_ptr(), self._fus_mg_fc2_w.data_ptr(),
                         mg_img.data_ptr(), Nout, 2048, 4096, stream)
            fvkm.add_bias_bf16(mg_img.data_ptr(),
                               self._merger_fc2_b.data_ptr(), Nout, 2048,
                               stream)
            llm_h.index_copy_(0, rt["vis_idx"], mg_img)
            snap("llm_input_embeds", llm_h)
            snap("merger_out", mg_img)
        else:
            # ``llm_h`` is the residual stream and the LLM overwrites it in
            # place, so the caller's embeds live in their own buffer and are
            # re-seeded here on every run.
            llm_h.copy_(rt["llm_in"])
        P.qwen3vl_llm_forward(gemm=gemm, fvk=fvkm, attn=attn, stream=stream,
                              **rt["llm"])
        snap("llm_h", llm_h)

        # ═══ vlln + VL self-attn (4L) ═══
        vlsa_h = rt["vlsa_h"]
        P.vlln_forward(gemm=gemm, fvk=fvkm, stream=stream, **rt["vlln"])
        snap("vlln_out", vlsa_h)
        P.vl_self_attn_forward(gemm=gemm, fvk=fvkm, attn=attn, stream=stream,
                               **rt["vlsa"])
        snap("vlsa_h", vlsa_h)

    def _run_kernel_backbone(self, aux: dict,
                             capture: dict | None = None) -> torch.Tensor:
        """One observation through the BF16 kernel backbone.

        Returns ``backbone_features`` ``(1, Se, 2048)``. The returned tensor is
        the runtime's persistent ``vlsa_h``, so a caller that keeps it across
        another call must clone it — ``set_prompt`` does.

        ``capture`` is passed through to :meth:`_kbb_forward`; see there for the
        snapshot keys.
        """
        self._kbb_load_inputs(aux)
        self._kbb_forward(
            0, snap_into=capture,
            vit_layers=bool(capture is not None and capture.get("vit_layers")))
        torch.cuda.synchronize()
        return self._kbb_rt["vlsa_h"].unsqueeze(0)

    # ────────────────────────────────────────────────────────────────
    # Backbone CUDA graph (lever #11)
    # ────────────────────────────────────────────────────────────────

    def _capture_backbone_graph(self) -> None:
        """Record :meth:`_kbb_forward` once, on a side stream, after warmup.

        Measured cost and benefit (docs §6.13.2, paired alternating A/B,
        median of 5×7, clocks locked):

        ===========  =============  =====================
        arm          wall           pure CPU submit
        ===========  =============  =====================
        eager        62.47 ms       51.97 ms
        graph        **57.93 ms**   **0.46 ms** (0.9%)
        ===========  =============  =====================

        So −4.54 ms per observation against a one-time warmup 198.7 ms +
        capture 63.4 ms = **262.1 ms**, i.e. **57.7 observations to break
        even**. That is why this is tied to the per-observation entry rather
        than to ``set_prompt``: under the one-shot contract (one frontend, one
        frame) the capture is a pure loss, and it was correctly left unlanded
        until ``infer(aux=...)`` existed (§6.15.6).

        The three warmup runs are not optional. Without them the first
        execution of each kernel resolves its cuDNN/cuBLAS workspace lazily
        *inside* the capture, which either fails or bakes a first-call
        allocation into the graph.

        Capture happens on a non-default stream and every launch inside
        ``_kbb_forward`` is handed ``stream.cuda_stream`` explicitly — the
        default stream cannot be captured, and a kernel launched on stream 0
        while capturing records nothing (docs §6.11.5 trap 3).
        """
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._kbb_forward(stream.cuda_stream)
        torch.cuda.synchronize()

        self._kbb_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            self._kbb_graph.capture_begin()
            self._kbb_forward(stream.cuda_stream)
            self._kbb_graph.capture_end()
        torch.cuda.synchronize()

    def run_backbone_graph(self, aux: dict) -> torch.Tensor:
        """Replay the captured backbone for a fresh observation.

        Public for parity with ``GrootN17TorchFrontendThorFP8`` and the RTX FP8
        mixin, which expose the same method. ``infer(aux=...)`` routes here when
        the graph arm is enabled.

        The order of the three steps is load-bearing:

        1. **validate** — the graph bakes the token layout, the rope tables and
           ``grid_thw``, so a changed prompt structure must be refused rather
           than replayed (§6.15.1);
        2. **load** — ``_kbb_load_inputs`` copies this observation into the
           runtime's *persistent* buffers, which are the addresses the graph
           recorded. Loading before the (lazy) capture also means the warmup
           runs on real values rather than whatever the buffers held;
        3. **replay** — then synchronize, matching ``_run_kernel_backbone``'s
           contract so the two arms are interchangeable for the caller and the
           A/B measures the same wall-clock boundary.

        The returned tensor is the runtime's persistent ``vlsa_h``, exactly like
        the eager arm, so a caller that keeps it across another observation must
        clone it. ``infer`` does.
        """
        self._validate_observation_contract(aux)
        self._kbb_load_inputs(aux)
        if not hasattr(self, "_kbb_graph"):
            self._capture_backbone_graph()
        self._kbb_graph.replay()
        torch.cuda.synchronize()
        return self._kbb_rt["vlsa_h"].unsqueeze(0)

    # ────────────────────────────────────────────────────────────────
    # DiT setup: cross K/V, timestep embedding, AdaLN modulators
    # ────────────────────────────────────────────────────────────────

    def _project_dit_cross_kv(self):
        """The 16 cross-attention K/V projections for the current observation.

        Shared by the one-shot and the continuous paths so they cannot drift.

        bf16 tensor-core GEMM (``gemm.bf16_nn``) plus ``fvk.add_bias_bf16``, from
        operands that are **already bf16**: the checkpoint's K/V weights load as
        ``(2048, 1536)`` bf16 row-major — exactly ``bf16_nn``'s ``[K, N]``
        convention — and ``backbone_features`` is bf16. So this is **not** a
        quantization of the KV cache; nothing is reduced in precision at the
        input and the stored dtype is unchanged (docs §6.18). What changes
        against the previous fp32 SGEMM arm is only the multiply/accumulate path.
        Measured on the real tensors, paired alternating, clocks locked:
        worst cos **0.999995467** against that arm (≈1 bf16 ULP) for
        **7.918 → 1.763 ms** per observation (**4.49×**).

        The bias is a separate kernel because ``bf16_nn_bias``'s cuBLASLt epilogue
        returns ``code=15`` (NOT_SUPPORTED) on SM87 — measured at **both** M=20
        and M=128, so it is not the M-alignment case ``pipeline_orin.py``
        documents; the epilogue is simply absent on this arch. This is the same
        workaround the DiT pipeline already uses.

        Allocates its 32 outputs, so it must never run inside a CUDA-graph
        capture. Neither caller does: the DiT graphs record ``_dit_call`` only and
        the backbone graph records ``_kbb_forward`` only.
        """
        backbone = self._backbone_features.squeeze(0)
        mask = self._visual_pos_masks
        # Boolean-mask indexing already yields a fresh contiguous tensor; the
        # explicit call is a no-op that makes the raw-pointer contract local.
        text_kv_src = backbone[~mask].contiguous()
        image_kv_src = backbone[mask].contiguous()
        gemm, fvk = self._gemm, self._fvk
        stream = torch.cuda.current_stream().cuda_stream

        K_list, V_list = [], []
        for j in range(16):
            li = 2 * j
            kv_src = text_kv_src if (li % 4 == 0) else image_kv_src
            M = kv_src.shape[0]
            for w, b, out_list in (
                    (self._dit_k_w[li], self._dit_k_b[li], K_list),
                    (self._dit_v_w[li], self._dit_v_b[li], V_list)):
                # Shapes come from the weights, not from a constant: they are the
                # ground truth and they are what the kernel is handed.
                Kin, N = w.shape
                if Kin != kv_src.shape[1]:
                    raise RuntimeError(
                        f"cross-K/V weight is [{Kin}, {N}] but this observation's "
                        f"backbone features are {kv_src.shape[1]}-wide; the "
                        "checkpoint and the backbone do not match")
                out = torch.empty(M, N, dtype=_BF16, device=backbone.device)
                gemm.bf16_nn(kv_src.data_ptr(), w.data_ptr(), out.data_ptr(),
                             M, N, Kin, stream)
                fvk.add_bias_bf16(out.data_ptr(), b.data_ptr(), M, N, stream)
                out_list.append(out)
        return K_list, V_list

    def _project_dit_cross_kv_fp32(self):
        """The fp32 SGEMM arm that :meth:`_project_dit_cross_kv` replaced.

        **Reference only — not on the serving path.** It exists because the bf16
        arm's entire claim is "same inputs, different multiply path", and that
        claim means nothing unless the previous arm is still callable to compare
        against. Promotes the weights per call, so it is deliberately slow: the
        403 MB fp32 cache the serving path used to keep
        (``_dit_cross_kv_weights``, §6.15.5) is gone, because a bf16 GEMM
        consumes the loaded bf16 weights directly.
        """
        backbone = self._backbone_features.squeeze(0)
        mask = self._visual_pos_masks
        text_kv_src = backbone[~mask]
        image_kv_src = backbone[mask]
        K_list, V_list = [], []
        for j in range(16):
            li = 2 * j
            kv_src = text_kv_src if (li % 4 == 0) else image_kv_src
            for w, b, out_list in (
                    (self._dit_k_w[li], self._dit_k_b[li], K_list),
                    (self._dit_v_w[li], self._dit_v_b[li], V_list)):
                out_list.append((kv_src.float() @ w.float()
                                 + b.float()).to(_BF16).contiguous())
        return K_list, V_list

    def _precompute_dit_cross_kv(self) -> None:
        K_list, V_list = self._project_dit_cross_kv()
        self._dit_cross_K = K_list
        self._dit_cross_V = V_list
        for name in ("_dit_attn", "_dit_graphs"):
            if hasattr(self, name):
                delattr(self, name)

    @staticmethod
    def _kv_slot_copy(slot: torch.Tensor, src: torch.Tensor) -> None:
        """Write ``src`` into the first rows of a ``[dit_kv_seq, 32, 48]`` slot.

        The slot is sized to ``max(Skv_text, Skv_image)``, so the shorter of
        the two families leaves a tail of stale rows; FA2 is handed the real
        ``kv_seq`` and never reads them.
        """
        flat = slot.view(slot.shape[0], -1)
        if src.shape[0] > flat.shape[0] or src.shape[1] != flat.shape[1]:
            raise RuntimeError(
                f"cross-K/V of shape {tuple(src.shape)} does not fit the "
                f"attention backend's slot of shape {tuple(flat.shape)}")
        flat[: src.shape[0]].copy_(src)

    def _refresh_dit_cross_kv(self) -> None:
        """Re-point the DiT's cross K/V at a new observation, in place.

        Writes into the attention backend's existing slots instead of
        rebuilding them. The four DiT CUDA graphs captured those slots'
        ``data_ptr()``s, so allocating fresh tensors would strand every graph
        on the previous observation's K/V while still replaying successfully —
        the failure is silent and the actions simply stop tracking the camera.
        ``_dit_attn``/``_dit_graphs`` are therefore *not* invalidated here; that
        is what distinguishes this from ``_precompute_dit_cross_kv``.
        """
        if not hasattr(self, "_dit_attn"):
            # Nothing captured yet, so there are no pointers to preserve and
            # the one-shot path is both correct and cheaper.
            self._precompute_dit_cross_kv()
            return

        K_list, V_list = self._project_dit_cross_kv()
        old = [tuple(t.shape) for t in self._dit_cross_K]
        new = [tuple(t.shape) for t in K_list]
        if new != old:
            raise RuntimeError(
                f"a new observation changed the DiT cross-K/V shapes {old} -> "
                f"{new}; the captured DiT graphs bake Skv_text/Skv_image, so "
                "this frontend cannot serve it. Construct a new frontend for "
                "a changed prompt or camera setup.")
        for j, (k, v) in enumerate(zip(K_list, V_list)):
            self._kv_slot_copy(self._dit_attn.dit_cross_K[j], k)
            self._kv_slot_copy(self._dit_attn.dit_cross_V[j], v)
        self._dit_cross_K = K_list
        self._dit_cross_V = V_list

    def _compute_timestep_emb(self, t_disc: int) -> torch.Tensor:
        half_dim = 128
        exponent = -math.log(10000) * torch.arange(
            0, half_dim, dtype=torch.float32, device=self.device) / (half_dim - 1)
        freqs = torch.exp(exponent)
        emb = torch.tensor(
            [t_disc], dtype=torch.float32, device=self.device)[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        h = emb @ self._ts_lin1_w.float() + self._ts_lin1_b.float()
        h = torch.nn.functional.silu(h)
        h = h @ self._ts_lin2_w.float() + self._ts_lin2_b.float()
        return h.to(_BF16).contiguous()

    def _compute_dit_adaln_modulators(self, temb: torch.Tensor):
        x = torch.nn.functional.silu(temb.float())
        shifts, scales = [], []
        for i in range(32):
            mod = x @ self._dit_ada_w[i].float() + self._dit_ada_b[i].float()
            scale, shift = mod.chunk(2, dim=-1)   # HF order: scale, shift
            shifts.append(shift.squeeze(0).to(_BF16).contiguous())
            scales.append(scale.squeeze(0).to(_BF16).contiguous())
        return shifts, scales

    # ────────────────────────────────────────────────────────────────
    # Embodiment encoders / decoder / output projection
    # ────────────────────────────────────────────────────────────────

    def _run_state_encode(self, state_flat: torch.Tensor) -> torch.Tensor:
        A, D = self._action_dim, self._dit_dim
        x = state_flat.view(1, A).float()
        h = x @ self._st_enc_l1_W.float() + self._st_enc_l1_b.float()
        h = torch.nn.functional.relu(h)
        out = h @ self._st_enc_l2_W.float() + self._st_enc_l2_b.float()
        return out.to(_BF16).view(1, 1, D)

    def _run_action_encode(self, actions: torch.Tensor, t_disc: int,
                           action_horizon: int) -> torch.Tensor:
        A, D = self._action_dim, self._dit_dim
        device = self.device
        half_dim = D // 2
        exponent = -torch.arange(
            half_dim, dtype=torch.float32, device=device
        ) * (math.log(10000.0) / half_dim)
        timesteps = torch.full(
            (action_horizon,), float(t_disc), dtype=torch.float32, device=device)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        tau_emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)

        x = actions.view(action_horizon, A).float()
        a_emb = x @ self._ac_enc_W1_W.float() + self._ac_enc_W1_b.float()
        cat = torch.cat([a_emb, tau_emb], dim=-1)
        h = cat @ self._ac_enc_W2_W.float() + self._ac_enc_W2_b.float()
        h = torch.nn.functional.silu(h)
        out = h @ self._ac_enc_W3_W.float() + self._ac_enc_W3_b.float()
        return out.to(_BF16).view(1, action_horizon, D)

    def _run_action_decode(self, dit_out: torch.Tensor) -> torch.Tensor:
        A = self._action_dim
        h_in = int(self._ac_dec_l1_W.shape[0])
        x = dit_out.view(-1, h_in).float()
        h = x @ self._ac_dec_l1_W.float() + self._ac_dec_l1_b.float()
        h = torch.nn.functional.relu(h)
        out = h @ self._ac_dec_l2_W.float() + self._ac_dec_l2_b.float()
        return out.to(_BF16).view(1, dit_out.shape[1], A)

    def _run_dit_output_proj(self, h: torch.Tensor,
                             temb: torch.Tensor) -> torch.Tensor:
        D = self._dit_dim
        x = torch.nn.functional.silu(temb.float())
        mod = x @ self._proj_out_1_w.float() + self._proj_out_1_b.float()
        shift, scale = mod.chunk(2, dim=-1)
        h_norm = torch.nn.functional.layer_norm(h.float(), (D,), eps=1e-5)
        h_mod = h_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        out = h_mod @ self._proj_out_2_w.float() + self._proj_out_2_b.float()
        return out.to(_BF16).contiguous()

    # ────────────────────────────────────────────────────────────────
    # DiT buffers, attention slots, forward, CUDA graph capture
    # ────────────────────────────────────────────────────────────────

    def _allocate_infer_buffers(self, action_horizon: int) -> None:
        Sa = 1 + action_horizon
        D, FF = self._dit_dim, 4 * self._dit_dim
        device = self.device
        self._infer_bufs = {
            "dit_h": torch.empty((Sa, D), dtype=_BF16, device=device),
            "dit_xn": torch.empty((Sa, D), dtype=_BF16, device=device),
            "dit_o_proj_out": torch.empty((Sa, D), dtype=_BF16, device=device),
            "dit_ff_proj_out": torch.empty((Sa, FF), dtype=_BF16, device=device),
        }
        if getattr(self, "_use_int8_dit", False):
            # Pre-allocated, never re-allocated: the pointers are baked into the
            # captured graph. The per-token scales live on the device and are
            # never read back -- a host-side scale would make the tier
            # uncapturable, and §6.3 measured the INT8 DiT at 2.1x SLOWER than
            # bf16 without the graph.
            self._infer_bufs.update({
                "dit_xn1_i8": torch.empty((Sa, D), dtype=torch.int8,
                                          device=device),
                "dit_xn2_i8": torch.empty((Sa, D), dtype=torch.int8,
                                          device=device),
                "dit_o_i8": torch.empty((Sa, D), dtype=torch.int8,
                                        device=device),
                "dit_ff_i8": torch.empty((Sa, FF), dtype=torch.int8,
                                         device=device),
                "dit_xn1_s": torch.empty((Sa,), dtype=torch.float32,
                                         device=device),
                "dit_xn2_s": torch.empty((Sa,), dtype=torch.float32,
                                         device=device),
                "dit_o_s": torch.empty((Sa,), dtype=torch.float32,
                                       device=device),
                "dit_ff_s": torch.empty((Sa,), dtype=torch.float32,
                                        device=device),
            })

    def _build_dit_attn(self, Sa: int) -> None:
        from flash_rt.hardware.rtx.attn_backend_groot_n17 import (
            RtxFlashAttnBackendGrootN17,
        )

        Skv_text = int(self._dit_cross_K[0].shape[0])
        Skv_image = int(self._dit_cross_K[1].shape[0])
        attn = RtxFlashAttnBackendGrootN17(
            num_vit_groups=int(getattr(self, "_num_vit_views", 4)),
            llm_seq_max=int(self.Se),
            vl_self_attn_seq_max=int(self.Se),
            sa=int(Sa),
            s_kv_text=Skv_text,
            s_kv_image=Skv_image,
            device=self.device,
            slot_dtype=_BF16,
        )
        for j, (k_src, v_src) in enumerate(zip(self._dit_cross_K,
                                               self._dit_cross_V)):
            self._kv_slot_copy(attn.dit_cross_K[j], k_src)
            self._kv_slot_copy(attn.dit_cross_V[j], v_src)
        self._dit_attn = attn

    def _dit_call(self, pipeline, bufs, weights, dims, *, stream: int = 0):
        if not hasattr(self, "_gemm"):
            import flash_rt.flash_rt_kernels as _fvk
            self._fvk = _fvk
            self._gemm = _fvk.GemmRunner()
        # _allocate_infer_buffers names the scratch by frontend convention;
        # the pipeline names it by role.
        bufs_ptrs = {
            "h": bufs["dit_h"].data_ptr(),
            "xn": bufs["dit_xn"].data_ptr(),
            "o_proj_out": bufs["dit_o_proj_out"].data_ptr(),
            "ff_proj_out": bufs["dit_ff_proj_out"].data_ptr(),
        }
        if getattr(self, "_use_int8_dit", False):
            for role in ("xn1_i8", "xn1_s", "xn2_i8", "xn2_s",
                         "o_i8", "o_s", "ff_i8", "ff_s"):
                bufs_ptrs[role] = bufs["dit_" + role].data_ptr()
        pipeline.dit_forward(
            gemm=self._gemm, fvk=self._fvk,
            bufs=bufs_ptrs,
            weights=weights, dims=dims, attn=self._dit_attn, stream=int(stream))

    #: DiT GEMM weight families that go INT8, with their (N, K) output/input
    #: dims as a function of D. ``ada`` is deliberately absent: it runs in
    #: torch fp32 in ``_compute_dit_adaln_modulators``, outside the kernel loop,
    #: so quantizing it would change a site the §6.8 gate measured but the
    #: shipped path does not use. (The gate quantized it anyway, i.e. it gated
    #: 228 Linears where this ships 192 — conservative in the right direction.)
    _DIT_INT8_FAMILIES = ("q", "k", "v", "o", "ff_proj", "ff_down")

    def _quantize_dit_weights(self) -> None:
        """Build the int8 (N,K) + fp32 (N,) scale pairs the CUTLASS kernel wants.

        Layout convention (written down because getting it wrong is silent for
        the square families): the weight spec applies ``T()``, so
        ``_dit_<F>_w[i]`` is **(K, N)** for ``gemm.bf16_nn``. The INT8 kernel
        wants the **untransposed** ``nn.Linear`` layout **(N, K)** with one
        scale per output channel — the opposite convention. Both are kept
        resident so the two tiers can be A/B'd inside one process (paired
        alternating timing); the cost is ~1.19 GB of int8 on top of the bf16
        2.18 GB, tracked as ``docs/groot_n17_orin_sm87.md`` §8, item
        「清理冗余权重内存」 (redundant weight memory).

        ``(N, K)`` is read **per layer from the tensor**, not from a table: K
        and V are ``(2048, 1536)`` on cross layers and ``(1536, 1536)`` on self
        layers, because a cross block projects from the 2048-dim backbone
        features while a self block projects from the 1536-dim action stream.
        A single table silently mis-quantizes half of them.
        """
        import flash_rt.flash_rt_kernels as _fvk

        dev = self.device
        self._fvk = _fvk
        self._dit_int8_nk = {}

        for fam in self._dit_int8_families:
            src = getattr(self, f"_dit_{fam}_w")
            w8_list, s_list, nk_list = [], [], []
            for w_kn in src:
                K, N = int(w_kn.shape[0]), int(w_kn.shape[1])
                nk_list.append((N, K))
                w_f32 = w_kn.t().contiguous().float()   # (N, K)
                scale = torch.clamp(
                    w_f32.abs().amax(dim=1) / 127.0, min=1e-12
                ).to(device=dev, dtype=torch.float32).contiguous()
                q = torch.clamp(torch.round(w_f32 / scale[:, None]),
                                -127, 127).to(torch.int8).contiguous()
                w8_list.append(q)
                s_list.append(scale)
            setattr(self, f"_dit_{fam}_w8", w8_list)
            setattr(self, f"_dit_{fam}_s", s_list)
            self._dit_int8_nk[fam] = nk_list

        # dit_forward hardcodes (M=Sa, N=D, K=D) for k/v and only ever calls
        # them on self layers, so those layers must actually be D x D. Check it
        # here rather than letting a mismatch reach the kernel. Families exempt
        # from INT8 have no entry and no INT8 call site to be wrong about.
        D = self._dit_dim
        for fam in ("k", "v"):
            if fam not in self._dit_int8_nk:
                continue
            for li in range(1, 32, 2):
                if self._dit_int8_nk[fam][li] != (D, D):
                    raise RuntimeError(
                        f"_dit_{fam}_w[{li}] (a self-attn layer) quantized to "
                        f"(N,K)={self._dit_int8_nk[fam][li]}, but dit_forward "
                        f"calls it as (D,D)=({D},{D})")

        self._smoke_int8_dit_layout()

    def _smoke_int8_dit_layout(self) -> None:
        """One real INT8 launch per family, at the pipeline's own shape.

        A bind-time smoke, and specifically a **transpose** smoke: ``q``/``k``/
        ``v``/``o`` are square on the layers that use them, so a transposed
        weight passes every dimension assertion and silently produces garbage.
        The reference is ``x @ w_kn`` — i.e. what ``gemm.bf16_nn`` computes in
        the other tier — so the smoke validates the whole (K,N)->(N,K)
        decision, not just the kernel's own convention. This is the same trap
        ``_merger_fc1_w`` (4096, 4096) presented during the fusion work.

        The input is synthetic on purpose: only the layout is under test here.
        Fidelity is gated separately, end to end, on the real-frame fixtures.
        """
        fvk = self._fvk
        D = self._dit_dim
        Sa = 41
        st = torch.cuda.current_stream().cuda_stream
        g = torch.Generator(device="cpu").manual_seed(0)

        for fam in self._dit_int8_families:
            # k/v are only consumed on self layers, whose K is D; use layer 1
            # so the smoke exercises the shape the pipeline really launches.
            li = 1 if fam in ("k", "v") else 0
            N, K = self._dit_int8_nk[fam][li]
            x = (torch.randn(Sa, K, generator=g) * 0.5).to(
                device=self.device, dtype=_BF16)
            a8 = torch.empty(Sa, K, dtype=torch.int8, device=self.device)
            a_s = torch.empty(Sa, dtype=torch.float32, device=self.device)
            fvk.quantize_int8_rowwise(x.data_ptr(), a8.data_ptr(),
                                      a_s.data_ptr(), Sa, K, st)
            out = torch.empty(Sa, N, dtype=_BF16, device=self.device)
            status = fvk.cutlass_int8_rowwise_bf16out(
                a8.data_ptr(),
                int(getattr(self, f"_dit_{fam}_w8")[li].data_ptr()),
                a_s.data_ptr(),
                int(getattr(self, f"_dit_{fam}_s")[li].data_ptr()),
                out.data_ptr(), Sa, N, K, st)
            if status != 0:
                raise RuntimeError(
                    f"INT8 DiT bind-time smoke failed for {fam!r}: "
                    f"cutlass_int8_rowwise_bf16out status={status} "
                    f"shape=({Sa},{N},{K}). The SM87 build may lack this tile.")
            ref = x.double() @ getattr(self, f"_dit_{fam}_w")[li].double()
            got = out.double()
            c = float((got.flatten() @ ref.flatten())
                      / (got.norm() * ref.norm() + 1e-30))
            if c < 0.99:
                raise RuntimeError(
                    f"INT8 DiT layout smoke for {fam!r} layer {li}: cos "
                    f"{c:.6f} vs x @ w_kn. The (N,K) int8 weight is almost "
                    f"certainly transposed -- the spec stores (K,N) and the "
                    f"kernel wants (N,K).")

    def _dit_weights(self, shift_list, scale_list) -> dict:
        w = {
            "scale_msa": [t.data_ptr() for t in scale_list],
            "shift_msa": [t.data_ptr() for t in shift_list],
            "q_w": [w.data_ptr() for w in self._dit_q_w],
            "q_b": [b.data_ptr() for b in self._dit_q_b],
            "k_w": [w.data_ptr() for w in self._dit_k_w],
            "k_b": [b.data_ptr() for b in self._dit_k_b],
            "v_w": [w.data_ptr() for w in self._dit_v_w],
            "v_b": [b.data_ptr() for b in self._dit_v_b],
            "o_w": [w.data_ptr() for w in self._dit_o_w],
            "o_b": [b.data_ptr() for b in self._dit_o_b],
            "ff_proj_w": [w.data_ptr() for w in self._dit_ff_proj_w],
            "ff_proj_b": [b.data_ptr() for b in self._dit_ff_proj_b],
            "ff_down_w": [w.data_ptr() for w in self._dit_ff_down_w],
            "ff_down_b": [b.data_ptr() for b in self._dit_ff_down_b],
        }
        if getattr(self, "_use_int8_dit", False):
            # The presence of "q_w8" is what selects the INT8 tier inside
            # dit_forward; the bf16 keys stay so the two tiers share one dict
            # shape and only the pointer set differs. bf16_families declares
            # the exemptions explicitly, so the pipeline can cross-check the
            # list against the pointers instead of inferring a tier per GEMM.
            for fam in self._dit_int8_families:
                w[fam + "_w8"] = [t.data_ptr()
                                  for t in getattr(self, f"_dit_{fam}_w8")]
                w[fam + "_s"] = [t.data_ptr()
                                 for t in getattr(self, f"_dit_{fam}_s")]
            w["bf16_families"] = list(self._dit_bf16_families)
        return w

    def _dit_dims(self, Sa: int) -> dict:
        D = self._dit_dim
        return {"Sa": int(Sa), "D": D, "FF": 4 * D,
                "Skv_text": int(self._dit_cross_K[0].shape[0]),
                "Skv_image": int(self._dit_cross_K[1].shape[0])}

    def _run_dit(self, bufs: dict, shift_list, scale_list, Sa: int) -> None:
        from flash_rt.models.groot_n17 import pipeline_orin

        if not hasattr(self, "_dit_attn"):
            self._build_dit_attn(Sa)
        self._dit_call(pipeline_orin, bufs,
                       self._dit_weights(shift_list, scale_list),
                       self._dit_dims(Sa))

    def _capture_dit_graphs(self, num_inference_timesteps: int = 4,
                            action_horizon: int = 40,
                            num_timestep_buckets: int = 1000) -> None:
        from flash_rt.models.groot_n17 import pipeline_orin

        Sa = action_horizon + 1
        if not hasattr(self, "_infer_bufs"):
            self._allocate_infer_buffers(action_horizon)
        if not hasattr(self, "_dit_attn"):
            self._build_dit_attn(Sa)
        # Keyed on (steps, buckets), not on presence. The modulators depend on
        # the bucket count and the graphs bake their pointers, so they have to
        # be rebuilt *before* capture whenever either changes. Omitting the
        # bucket here is what let the graph arm precompute with the base-class
        # default 1000 while ``infer`` fed the action encoder the caller's
        # value -- 3.581 deg of decoded-action error at buckets=200 (§6.24).
        mods = (int(num_inference_timesteps), int(num_timestep_buckets))
        if getattr(self, "_modulator_params", None) != mods:
            self._precompute_diffusion_modulators(
                num_inference_timesteps=num_inference_timesteps,
                num_timestep_buckets=num_timestep_buckets)
            self._modulator_params = mods
        if not hasattr(self, "_gemm"):
            import flash_rt.flash_rt_kernels as _fvk
            self._fvk = _fvk
            self._gemm = _fvk.GemmRunner()

        bufs = self._infer_bufs
        dims = self._dit_dims(Sa)

        for _ in range(3):
            self._dit_call(pipeline_orin, bufs,
                           self._dit_weights(self._step_shifts[0],
                                             self._step_scales[0]), dims)
        torch.cuda.synchronize()

        self._dit_graphs = []
        for step in range(num_inference_timesteps):
            weights = self._dit_weights(self._step_shifts[step],
                                        self._step_scales[step])
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.capture_begin()
                self._dit_call(pipeline_orin, bufs, weights, dims,
                               stream=stream.cuda_stream)
                graph.capture_end()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self._dit_graphs.append(graph)
        #: What these graphs are valid for. ``infer`` compares this against the
        #: requested triple before replaying; a graph list on its own does not
        #: record the horizon it was captured at, which is how a mismatch used
        #: to replay silently (§6.24).
        self._dit_graph_params = (int(num_inference_timesteps),
                                  int(action_horizon),
                                  int(num_timestep_buckets))

    # ────────────────────────────────────────────────────────────────
    # Calibration: not applicable on SM87, refused rather than inherited
    # ────────────────────────────────────────────────────────────────

    def calibrate(self, *args, **kwargs):
        """Refuse, with the reason. There is nothing to calibrate here.

        The inherited Thor implementation refines **FP8 act-scale alphas**
        across samples and then snapshots them via ``_snapshot_precision_spec``,
        which reads ``_vit_alpha_q`` / ``_dsm_alpha_*`` / ``_llm_alpha_*`` /
        ``_vlsa_alpha_*`` unconditionally. This frontend creates **none** of
        them, so inheriting it produced a bare
        ``AttributeError: 'GrootN17TorchFrontendOrin' object has no attribute
        '_vit_alpha_q'`` — an unexplained crash where the public API advertises
        a capability the platform does not have (AGENTS.md red line #4).

        Why there is nothing to calibrate:

        * SM87 has no FP8/FP4 tensor cores, so the whole alpha machinery the
          inherited method drives does not exist on this path.
        * The shipped low-bit tier is ``int8_rowwise`` on the DiT, and its
          scales are **dynamic**: ``quantize_int8_rowwise`` computes the amax
          on device every forward. A calibrated static scale would be strictly
          worse — the static-scale arm is what §6.10 measured and rejected.
        * The backbone is bf16, and bf16 needs no scale at all.

        ``precision_spec`` is inherited unchanged and keeps returning ``None``,
        which is now honest rather than merely unreachable: no calibration can
        ever run here.

        ``NotImplementedError`` is the exception the unified door asks for —
        ``flash_rt/api.py``'s ``VLAModel.calibrate`` documents that
        "unsupported frontends raise a clear NotImplementedError from their
        calibrate() method".
        """
        raise NotImplementedError(
            "GrootN17TorchFrontendOrin has nothing to calibrate: SM87 has no "
            "FP8/FP4 path, the INT8 DiT tier uses dynamic per-row device-side "
            "scales recomputed every forward, and the backbone is bf16. The "
            "inherited FP8-alpha calibration would crash on the absent "
            "_vit_alpha_*/_llm_alpha_* attributes. If you need a static-scale "
            "tier on Orin, that is a new precision arm, not a calibrate() call "
            "(docs/groot_n17_orin_sm87.md §6.10).")

    # ────────────────────────────────────────────────────────────────
    # Inference
    # ────────────────────────────────────────────────────────────────

    def infer(
        self,
        state_normalized: torch.Tensor,
        *,
        aux: dict | None = None,
        frames=None,
        initial_noise=None,
        num_inference_timesteps: int | None = None,
        action_horizon: int | None = None,
        num_timestep_buckets: int | None = None,
        use_dit_graph: bool = True,
        capture: dict | None = None,
    ) -> torch.Tensor:
        """Denoise an action chunk; optionally for a fresh observation.

        Passing ``aux`` runs the whole chain — backbone, DiT cross K/V,
        denoising loop — on that observation, so one frontend serves a stream
        of frames instead of one prompt. The four DiT CUDA graphs are *reused*,
        not re-captured: ``_refresh_dit_cross_kv`` writes the new K/V into the
        slots whose pointers the graphs already hold. Omitting ``aux`` keeps
        the original one-shot behaviour on ``set_prompt``'s observation.

        ``frames`` is the raw-input arm: ``(views, H, W, 3)`` uint8 camera bytes
        (a tensor, an ndarray, or a sequence of per-view arrays), host or device.
        The image path runs on the GPU here instead of in the vendor's processor
        on Orin's ARM CPU — 12.916-14.027 ms of host work becomes 1.448-1.485 ms
        of device work, measured end to end at 11.945-15.410 ms saved per
        observation (1.1189-1.1540x) — and the prompt-scoped tensors are
        re-presented by identity, so the caller never has to build an ``aux``
        bundle per observation. It needs ``fuse_image_embeds=True`` (the default)
        and cannot be combined with ``aux['pixel_values']``. See
        :mod:`flash_rt.frontends.torch._groot_n17_preprocess` for what is
        bit-exact against the vendor chain and what carries ≤1 LSB.

        The backbone also runs from a captured CUDA graph on this path
        (:meth:`run_backbone_graph`), captured lazily on the first ``aux`` call
        and replayed thereafter — worth 4.00 ms per observation at the deployed
        caliper against a 259.7 ms one-time capture (docs §6.17), which is why it
        is tied to ``aux`` and not to ``set_prompt``. Pass
        ``use_backbone_graph=False`` at construction to keep
        the eager arm. The one-shot path is unaffected either way. The image
        path stays outside the graph in both arms.
        """
        if not hasattr(self, "_backbone_features"):
            raise RuntimeError("call set_prompt before infer")
        if frames is not None:
            aux = self._observation_aux(frames, aux)
        if aux is not None:
            if self._use_backbone_graph:
                # run_backbone_graph validates the observation contract itself,
                # so validating again here would be a second (cheap, but
                # redundant) pass over the same metadata.
                backbone = self.run_backbone_graph(aux)
            else:
                self._validate_observation_contract(aux)
                backbone = self._run_kernel_backbone(aux)
            self._backbone_features = backbone.clone()
            self._refresh_dit_cross_kv()
        action_horizon = (self._action_horizon if action_horizon is None
                          else int(action_horizon))
        num_inference_timesteps = (
            self._num_inference_timesteps if num_inference_timesteps is None
            else int(num_inference_timesteps))
        num_timestep_buckets = (
            self._num_timestep_buckets if num_timestep_buckets is None
            else int(num_timestep_buckets))
        if action_horizon > self._action_horizon:
            raise ValueError(
                f"action_horizon={action_horizon} exceeds this checkpoint's "
                f"configured horizon {self._action_horizon}")

        t0 = time.perf_counter()
        if not hasattr(self, "_dit_cross_K"):
            self._precompute_dit_cross_kv()

        device = self.device
        A = self._action_dim
        Sa = action_horizon + 1

        state_features = self._run_state_encode(
            state_normalized.to(device).to(_BF16))

        if not hasattr(self, "_infer_bufs"):
            self._allocate_infer_buffers(action_horizon)
        bufs = self._infer_bufs

        if initial_noise is not None:
            actions = initial_noise.to(device).to(_BF16).contiguous().clone()
        else:
            actions = torch.randn(
                1, action_horizon, A, dtype=_BF16, device=device)

        dt = 1.0 / num_inference_timesteps
        pos_embed = self._ah_pos_embed_w[:action_horizon].to(_BF16)

        self._infer_shift_lists = []
        self._infer_scale_lists = []
        self._infer_temb_list = []

        graphs = None
        if use_dit_graph:
            #: The triple the graphs must have been captured with. All three are
            #: baked in: ``action_horizon`` sets ``Sa`` and therefore every DiT
            #: kernel's token-row count, ``num_inference_timesteps`` sets how
            #: many graphs exist, and ``num_timestep_buckets`` sets the AdaLN
            #: modulators the graphs read. Checking only the graph count is what
            #: let a horizon mismatch replay silently (§6.24).
            want = (int(num_inference_timesteps), int(action_horizon),
                    int(num_timestep_buckets))
            if not hasattr(self, "_dit_graphs"):
                self._capture_dit_graphs(
                    num_inference_timesteps=num_inference_timesteps,
                    action_horizon=action_horizon,
                    num_timestep_buckets=num_timestep_buckets)
            graphs = self._dit_graphs
            captured = getattr(self, "_dit_graph_params", None)
            if captured != want or len(graphs) != num_inference_timesteps:
                _note_dit_graph_bypass(captured, want)
                graphs = None

        # ``capture`` (optional) is filled with the per-step DiT input token
        # stack, the per-step decoded velocity, and the final normalized
        # action, so a precision failure can be pinned to a denoising step.
        if capture is not None:
            capture["dit_step_input"] = []
            capture["velocity_per_step"] = []

        for step in range(num_inference_timesteps):
            t_cont = step / num_inference_timesteps
            t_disc = int(t_cont * num_timestep_buckets)

            if graphs is not None:
                temb = self._step_temb[step]
                shift_list = self._step_shifts[step]
                scale_list = self._step_scales[step]
            else:
                temb = self._compute_timestep_emb(t_disc)
                shift_list, scale_list = self._compute_dit_adaln_modulators(temb)
                self._infer_shift_lists.append(shift_list)
                self._infer_scale_lists.append(scale_list)
                self._infer_temb_list.append(temb)

            action_features = self._run_action_encode(
                actions, t_disc, action_horizon)
            action_features = action_features + pos_embed.unsqueeze(0)

            sa_embs = torch.cat([state_features, action_features], dim=1)
            bufs["dit_h"][:Sa].copy_(sa_embs.squeeze(0).contiguous())
            if capture is not None:
                capture["dit_step_input"].append(
                    sa_embs.detach().float().cpu().clone())

            if graphs is not None:
                graphs[step].replay()
            else:
                self._run_dit(bufs, shift_list, scale_list, Sa)

            h_out = self._run_dit_output_proj(
                bufs["dit_h"][:Sa].unsqueeze(0), temb)
            velocity = self._run_action_decode(h_out[:, -action_horizon:])
            if capture is not None:
                capture["velocity_per_step"].append(
                    velocity.detach().float().cpu().clone())
            actions = actions + (dt * velocity).to(actions.dtype)

        if capture is not None:
            capture["final_actions_norm"] = actions.detach().float().cpu().clone()

        torch.cuda.synchronize()
        self.latency_records.append((time.perf_counter() - t0) * 1000)
        return actions.float()

    def _warmup_infer(self) -> None:
        warm_state = torch.zeros(
            1, 1, self._action_dim, dtype=torch.float32)
        torch.manual_seed(0)
        warm_noise = torch.randn(
            1, self._action_horizon, self._action_dim,
            dtype=_BF16, device=self.device)
        _ = self.infer(warm_state, initial_noise=warm_noise,
                       use_dit_graph=False)
