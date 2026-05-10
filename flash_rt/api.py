"""
FlashRT — Public API.

3 lines of code to run VLA inference:

    import flash_rt

    model = flash_rt.load_model(
        checkpoint="/path/to/checkpoint",
        framework="torch",
        autotune=3,
    )

    actions = model.predict(images=[base_img, wrist_img],
                            prompt="pick up the red block")
    # actions: np.ndarray (10, 7)
"""

import logging
import os

# Silence ``torch_xla``'s "Defaulting to PJRT_DEVICE=CPU" warning that
# fires when openpi (pulled in by the Pi0.5 torch frontend for the
# PaligemmaTokenizer) drags transformers→accelerate→torch_xla. We don't
# use XLA on the torch path, so the warning is pure noise. ``setdefault``
# preserves any value the user has already configured.
os.environ.setdefault("PJRT_DEVICE", "CUDA")

import numpy as np

logger = logging.getLogger(__name__)


class VLAModel:
    """Unified VLA inference model. Wraps ThorPipelineTorch or ThorPipelineJax."""

    def __init__(self, pipe, framework: str):
        self._pipe = pipe
        self._framework = framework
        self._current_prompt = None
        # rtx Pi0.5 (RtxTorchPi05) requires an explicit
        # ``calibrate_with_real_data([obs])`` call before the first
        # ``infer()``; Thor / rtx GROOT lazy-calibrate inside ``infer()``.
        # Track whether we still need to bootstrap calibration so that
        # first predict() can call it exactly once.
        self._needs_real_data_calibration = hasattr(
            pipe, "calibrate_with_real_data"
        )

    def predict(self, images, prompt=None, state=None):
        """Run inference.

        Args:
            images: list of numpy arrays (224,224,3) uint8 or float16.
                    Or a dict with 'image'/'wrist_image' keys.
            prompt: text prompt. Only needed on first call or when changing prompt.
                    If None, reuses the last prompt.
            state: robot state array (Pi0/Pi0-FAST only). Passed to set_prompt().
                   Pi0 uses continuous state projection; Pi0-FAST discretizes to text.

        Returns:
            np.ndarray: actions
        """
        if prompt is not None and prompt != self._current_prompt:
            if hasattr(self._pipe, 'set_prompt'):
                import inspect
                sig = inspect.signature(self._pipe.set_prompt)
                if 'state' in sig.parameters:
                    self._pipe.set_prompt(prompt, state=state)
                else:
                    self._pipe.set_prompt(prompt)
            self._current_prompt = prompt
        elif self._current_prompt is None:
            raise ValueError("prompt is required on first call")

        if isinstance(images, dict):
            obs = images
        elif isinstance(images, (list, tuple)):
            if len(images) == 0:
                raise ValueError("images list must have at least one frame")
            # Use the "images" list form so backends that support
            # variable num_views (rtx Pi0.5, etc.) don't choke on the
            # 1-view case. Also populate the legacy image / wrist_image
            # / wrist_image_right keys so Thor-style backends that only
            # read those still see the right frames.
            obs = {'images': list(images), 'image': images[0]}
            if len(images) >= 2:
                obs['wrist_image'] = images[1]
            if len(images) >= 3:
                obs['wrist_image_right'] = images[2]
        else:
            raise ValueError("images must be a list of numpy arrays or a dict")

        # rtx Pi0.5 expects an explicit calibration bootstrap before the
        # first infer(); fire it lazily here so user code stays "3 lines".
        if self._needs_real_data_calibration:
            self._pipe.calibrate_with_real_data([obs])
            self._needs_real_data_calibration = False

        result = self._pipe.infer(obs)
        return result['actions']

    def calibrate(
        self,
        observations,
        *,
        percentile: float = 99.9,
        max_samples=None,
        verbose: bool = False,
    ) -> None:
        """Unified calibration entry point.

        Args:
            observations: single dict or iterable of dicts. N=1 triggers
                the single-frame calibration path (back-compatible); N>=2
                engages dataset calibration with percentile-clipped amax
                reduction (RTX frontends only today).
            percentile: percentile for multi-sample amax reduction. 99.9
                by default; 100.0 == traditional max.
            max_samples: optional cap.
            verbose: log dispersion summary after reduction.

        See ``docs/calibration.md`` for full guidance.
        """
        if not hasattr(self._pipe, "calibrate"):
            raise NotImplementedError(
                "This frontend does not expose a public calibrate() API. "
                "Upgrade to a recent version of FlashRT that includes "
                "the unified calibration interface.")
        self._pipe.calibrate(
            observations,
            percentile=percentile,
            max_samples=max_samples,
            verbose=verbose,
        )
        # Any lazy-bootstrap was just handled explicitly — prevent
        # predict() from double-triggering it.
        self._needs_real_data_calibration = False

    @property
    def precision_spec(self):
        """Return the :class:`ModelPrecisionSpec` captured at calibration
        time, or None if the frontend does not surface it yet."""
        return getattr(self._pipe, "precision_spec", None)

    def recalibrate(self):
        """Force recalibration on next set_prompt().

        Use after fine-tuning or switching deployment domains.
        Clears calibration cache (and weight cache for JAX).
        """
        from flash_rt.core.quant.calibrator import clear_calibration
        clear_calibration(self._pipe._checkpoint_path)
        if self._framework == "jax":
            from flash_rt.core.weights.weight_cache import clear_weight_cache
            clear_weight_cache(self._pipe._checkpoint_path)
        self._pipe.calibrated = False
        self._pipe._real_data_calibrated = False
        self._current_prompt = None  # force re-set_prompt
        logger.info("Caches cleared. Next predict() will recalibrate.")

    @property
    def framework(self):
        return self._framework

    @property
    def prompt(self):
        return self._current_prompt


def load_model(checkpoint, framework="torch", num_views=2, autotune=3,
               recalibrate=False, weight_cache=True, config="pi05", device=None,
               decode_cuda_graph=False, decode_graph_steps=80,
               max_decode_steps=256,
               hardware="auto",
               embodiment_tag=None,
               action_horizon=None,
               use_fp4=False,
               fp4_layers=None,
               use_awq=None,
               awq_alpha=0.5,
               use_p1_split_gu=None,
               use_fp8=True):
    """Load a FlashRT model.

    Args:
        checkpoint: path to checkpoint directory.
            - torch: safetensors directory
            - jax: Orbax checkpoint directory
        framework: "torch" or "jax"
        num_views: number of camera views (default 2)
        autotune: CUDA Graph autotune intensity.
            0 or False = off (fastest startup, ~2ms slower inference risk)
            3 = default (Torch finds fast graph on trial 0-1)
            5+ = thorough (JAX may need more trials for fast graph)
            True = same as 3
        recalibrate: if True, ignore cached calibration (and weight cache for JAX)
            and force fresh FP8 quantization + calibration.
        weight_cache: if True (default), cache FP8-quantized weights to disk
            after first load. Only affects JAX.
        config: model config name: "pi05", "pi0", "groot", "pi0fast"
        device: ignored (auto-detects GPU). Reserved for future multi-GPU.
        decode_cuda_graph: Pi0-FAST only. Capture action-phase decode as CUDA
            Graph for max throughput (trades startup time for per-token speed).
        decode_graph_steps: Pi0-FAST only. Number of action tokens to capture
            in the decode graph (default 80).
        hardware: GPU backend selection. ``"auto"`` (default) detects the
            current CUDA device via compute capability and picks the
            best-matching backend:
              SM110 (Jetson Thor)  → ``flash_rt.hardware.thor.*``
              SM120 (RTX 5090)     → ``flash_rt.hardware.rtx.*``
                                     (falls back to Thor classes for models
                                      without an rtx-specific implementation —
                                      those classes have SM120 runtime forks
                                      where needed, e.g. Pi0-FAST.)
              SM89  (RTX 4090)     → ``flash_rt.hardware.rtx.*``
            Pass ``"thor"`` / ``"rtx_sm120"`` / ``"rtx_sm89"`` explicitly to
            force a specific backend (useful for cross-hardware debugging).
        embodiment_tag: GROOT only. Per-embodiment MLP slot to load. Passing
            ``None`` uses the backend default (``"new_embodiment"`` — unfit
            for the base 3B checkpoint demo; see below). The GR00T-N1.6-3B
            base checkpoint is only actually trained on a subset of its 32
            slots. For a working demo pick one of ``"gr1"``,
            ``"robocasa_panda_omron"``, or ``"behavior_r1_pro"``. Any other
            tag prints a warning and emits noise-like actions.
        action_horizon: GROOT only. Number of action steps to generate per
            inference (default = ``ACTION_HORIZON_MAX`` = 50). Set to a
            smaller value (e.g. 16 for LIBERO) to reduce DiT compute.
        use_fp4: Pi0.5 torch only. If True, enable NVFP4 quantization on the
            selected encoder FFN layers (Gate+Up + Down GEMMs). Requires
            SM100+ GPU (Thor SM110) and the flash_rt_fp4 extension. Falls
            back to FP8 with a warning if the extension is unavailable.
            Default False (production FP8 baseline).
            Validated on LIBERO Spatial: 491/500 = 98.2% (matches baseline).
        fp4_layers: Tuple of encoder layer indices to FP4-quantize (only
            applies when use_fp4=True). Default (7, 8, 9) = middle FFN
            subset, LIBERO-validated. Other subsets untested at task level.
        use_fp8: Enable FP8 execution where the selected frontend supports
            an FP8/BF16 switch. Defaults to True to preserve existing
            performance-oriented behavior.

    Returns:
        VLAModel instance with .predict() method.
    """
    if config not in ("pi05", "groot", "pi0", "pi0fast"):
        raise ValueError(
            f"Unknown config: {config}. Supported: pi05, groot, pi0, pi0fast")
    if framework not in ("torch", "jax"):
        raise ValueError(
            f"Unknown framework: {framework}. Supported: torch, jax")

    # When use_fp4=True, the default resolves to the best-known production
    # FP4 config (full 18 encoder FFN layers + AWQ + P1 split-GU). Passing
    # any sub-flag explicitly overrides the preset; None means "use preset".
    if use_fp4:
        if fp4_layers is None:
            fp4_layers = tuple(range(18))
        if use_awq is None:
            use_awq = True
        if use_p1_split_gu is None:
            use_p1_split_gu = True
    else:
        if fp4_layers is None:
            fp4_layers = (7, 8, 9)
        if use_awq is None:
            use_awq = False
        if use_p1_split_gu is None:
            use_p1_split_gu = False

    from flash_rt.hardware import detect_arch, resolve_pipeline_class
    arch = detect_arch() if hardware == "auto" else hardware

    if recalibrate:
        from flash_rt.core.quant.calibrator import clear_calibration
        try:
            clear_calibration(checkpoint)
        except FileNotFoundError:
            pass
        if framework == "jax":
            from flash_rt.core.weights.weight_cache import clear_weight_cache
            try:
                clear_weight_cache(checkpoint)
            except FileNotFoundError:
                pass
        logger.info("Caches cleared for %s", checkpoint)

    if framework == "jax":
        os.environ.setdefault(
            "XLA_FLAGS",
            "--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    pipe_cls = resolve_pipeline_class(config, framework, arch)

    # ── FP4 routing (Pi0.5 torch + Pi0.5 JAX on Thor) ──
    if use_fp4:
        if config != "pi05" or framework not in ("torch", "jax"):
            logger.warning(
                "use_fp4=True is only supported for config='pi05' with "
                "framework in ('torch', 'jax'); got config='%s' framework='%s'. "
                "Falling back to FP8.", config, framework)
            use_fp4 = False
        else:
            try:
                import flash_rt.flash_rt_fp4 as _fvk_fp4
                if not _fvk_fp4.has_nvfp4():
                    logger.warning(
                        "flash_rt_fp4 loaded but has_nvfp4()=False (SM100+ required). "
                        "Falling back to FP8.")
                    use_fp4 = False
            except ImportError:
                logger.warning(
                    "flash_rt_fp4 extension not available. Falling back to FP8.")
                use_fp4 = False

            if use_fp4:
                if framework == "torch":
                    from flash_rt.frontends.torch.pi05_thor_fp4 import (
                        Pi05TorchFrontendThorFP4,
                    )
                    pipe_cls = Pi05TorchFrontendThorFP4
                else:  # framework == "jax"
                    from flash_rt.frontends.jax.pi05_thor_fp4 import (
                        Pi05JaxFrontendThorFP4,
                    )
                    pipe_cls = Pi05JaxFrontendThorFP4
                logger.info(
                    "FP4 enabled (framework=%s): encoder FFN layers %s",
                    framework, sorted(fp4_layers))

    # Build the kwarg set per-model so we only pass args the target class
    # actually accepts. Keeps the dispatch table simple while still letting
    # users specify groot/pi0fast knobs.
    import inspect
    sig = inspect.signature(pipe_cls)
    kwargs: dict = {"num_views": num_views}
    if "use_fp8" in sig.parameters:
        kwargs["use_fp8"] = use_fp8
    if config == "pi0fast":
        kwargs.update(
            autotune=autotune,
            decode_cuda_graph=decode_cuda_graph,
            decode_graph_steps=decode_graph_steps,
            max_decode_steps=max_decode_steps,
        )
    elif config == "groot":
        # rtx-side GROOT accepts embodiment_tag + action_horizon; Thor-side
        # GROOT accepts embodiment_tag + autotune. Feature-detect via the
        # concrete class signature so one call site works for both.
        if "autotune" in sig.parameters:
            kwargs["autotune"] = autotune
        if "embodiment_tag" in sig.parameters and embodiment_tag is not None:
            kwargs["embodiment_tag"] = embodiment_tag
        if "action_horizon" in sig.parameters and action_horizon is not None:
            kwargs["action_horizon"] = action_horizon
    else:
        # pi05, pi0 — both Thor and rtx variants take (checkpoint, num_views, autotune)
        # or (checkpoint, num_views). Feature-detect.
        if "autotune" in sig.parameters:
            kwargs["autotune"] = autotune
        if "weight_cache" in sig.parameters:
            kwargs["weight_cache"] = weight_cache
        # FP4 frontend accepts these extra kwargs (only set when the class
        # actually accepts them — base class ignores, FP4 subclass uses).
        if use_fp4 and "use_fp4_encoder_ffn" in sig.parameters:
            kwargs["use_fp4_encoder_ffn"] = True
            kwargs["fp4_layers"] = fp4_layers
            if "use_awq" in sig.parameters:
                kwargs["use_awq"] = bool(use_awq)
                kwargs["awq_alpha"] = float(awq_alpha)
            if "use_p1_split_gu" in sig.parameters:
                kwargs["use_p1_split_gu"] = bool(use_p1_split_gu)

    pipe = pipe_cls(checkpoint, **kwargs)

    logger.info(
        "Model loaded: config=%s, framework=%s, arch=%s, class=%s",
        config, framework, arch, pipe_cls.__name__)
    return VLAModel(pipe, framework)
