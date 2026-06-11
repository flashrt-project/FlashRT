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
        self._current_prompt_state = None
        # rtx Pi0.5 (RtxTorchPi05) requires an explicit
        # ``calibrate_with_real_data([obs])`` call before the first
        # ``infer()``; Thor / rtx GROOT lazy-calibrate inside ``infer()``.
        # Track whether we still need to bootstrap calibration so that
        # first predict() can call it exactly once.
        self._needs_real_data_calibration = (
            hasattr(pipe, "calibrate_with_real_data")
            and hasattr(pipe, "calibrated")
        )

    @staticmethod
    def _snapshot_prompt_state(state):
        if state is None:
            return None
        try:
            return np.asarray(state).copy()
        except Exception:
            return state

    @staticmethod
    def _prompt_state_equal(a, b) -> bool:
        if a is None or b is None:
            return a is b
        try:
            return np.array_equal(np.asarray(a), np.asarray(b))
        except Exception:
            return a is b

    def predict(self, images, prompt=None, state=None):
        """Run inference.

        Args:
            images: list of numpy arrays (224,224,3) uint8 or float16.
                    Or a dict with 'image'/'wrist_image' keys.
            prompt: text prompt. Only needed on first call or when changing prompt.
                    If None, reuses the last prompt.
            state: optional robot state array. It is forwarded to
                   set_prompt() for frontends that encode state in prompt
                   tokens, and attached to the observation for frontends that
                   consume state during infer().

        Returns:
            np.ndarray: actions
        """
        if prompt is None and self._current_prompt is None:
            raise ValueError("prompt is required on first call")

        prompt_for_call = self._current_prompt if prompt is None else prompt
        prompt_changed = prompt is not None and prompt != self._current_prompt
        prompt_state_changed = False

        if hasattr(self._pipe, 'set_prompt'):
            import inspect
            sig = inspect.signature(self._pipe.set_prompt)
            prompt_accepts_state = 'state' in sig.parameters
            if prompt_accepts_state:
                prompt_state_changed = not self._prompt_state_equal(
                    self._current_prompt_state, state)
        else:
            sig = None
            prompt_accepts_state = False

        if prompt_changed or prompt_state_changed:
            if hasattr(self._pipe, 'set_prompt'):
                if prompt_accepts_state:
                    self._pipe.set_prompt(prompt_for_call, state=state)
                else:
                    self._pipe.set_prompt(prompt_for_call)
            self._current_prompt = prompt_for_call
            self._current_prompt_state = self._snapshot_prompt_state(state)

        if isinstance(images, dict):
            obs = dict(images)
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

        if state is not None and "state" not in obs:
            obs["state"] = state

        # RTX Pi0.5 can swap in a different cached pipeline when a changing
        # state prompt hits a new token length. Re-check that frontend's
        # calibration flag instead of relying only on the first-call latch.
        needs_real_data_calibration = self._needs_real_data_calibration
        if (hasattr(self._pipe, "_prompt_pipeline_cache")
                and not getattr(self._pipe, "calibrated", False)):
            needs_real_data_calibration = True
        if (needs_real_data_calibration
                and hasattr(self._pipe, "calibrate_with_real_data")):
            self._pipe.calibrate_with_real_data([obs])
            self._needs_real_data_calibration = False

        result = self._pipe.infer(obs)
        return result['actions']

    def warm_state_prompt_buckets(self, images, prompt, states):
        """Pre-build Pi0.5 state-prompt runtime buckets.

        Pi0.5 encodes robot state in the text prompt. Different state
        values can tokenize to different lengths; warming representative
        states up front prevents the control loop from paying graph
        capture/autotune the first time each length appears.
        """
        if not hasattr(self._pipe, "warm_state_prompt_buckets"):
            raise NotImplementedError(
                "This frontend does not expose state prompt bucket warmup.")

        if isinstance(images, dict):
            obs = dict(images)
        elif isinstance(images, (list, tuple)):
            if len(images) == 0:
                raise ValueError("images list must have at least one frame")
            obs = {"images": list(images), "image": images[0]}
            if len(images) >= 2:
                obs["wrist_image"] = images[1]
            if len(images) >= 3:
                obs["wrist_image_right"] = images[2]
        else:
            raise ValueError("images must be a list of numpy arrays or a dict")

        lengths = self._pipe.warm_state_prompt_buckets(prompt, states, obs)
        self._needs_real_data_calibration = False
        self._current_prompt = None
        self._current_prompt_state = None
        return lengths

    def set_prompt(self, *args, **kwargs):
        """Delegate prompt setup to the selected frontend."""
        if not hasattr(self._pipe, "set_prompt"):
            raise NotImplementedError(
                "This frontend does not expose set_prompt().")
        result = self._pipe.set_prompt(*args, **kwargs)
        if "prompt" in kwargs:
            self._current_prompt = kwargs["prompt"]
        elif args and isinstance(args[0], str):
            self._current_prompt = args[0]
        try:
            import inspect
            sig = inspect.signature(self._pipe.set_prompt)
            params = list(sig.parameters)
            if "state" in sig.parameters:
                state_pos = params.index("state")
                if "state" in kwargs:
                    state = kwargs["state"]
                elif len(args) > state_pos:
                    state = args[state_pos]
                else:
                    state = None
                self._current_prompt_state = self._snapshot_prompt_state(state)
        except (TypeError, ValueError):
            pass
        return result

    def infer(self, *args, **kwargs):
        """Delegate inference to the selected frontend."""
        if not hasattr(self._pipe, "infer"):
            raise NotImplementedError(
                "This frontend does not expose infer().")
        return self._pipe.infer(*args, **kwargs)

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
               num_steps=None,
               vision_pool_factor=None,
               vision_num_layers=None,
               cache_frames=None,
               use_fp16=False,
               use_fp8=True,
               state_prompt_mode="exact"):
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
        config: model config name: "pi05", "pi0", "groot", "groot_n17",
            "pi0fast", "motus", "wan22_ti2v_5b"
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
              SM87  (Jetson Orin)  → ``flash_rt.hardware.rtx.*`` (experimental,
                                     Pi0.5 torch only; BF16 default, INT8
                                     via Orin env flags)
            Pass ``"thor"`` / ``"rtx_sm120"`` / ``"rtx_sm89"`` /
            ``"rtx_sm87"`` explicitly to
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
        use_fp16: Opt-in non-quantized full-FP16 path for Pi0.5, GROOT N1.6,
            and GROOT N1.7 (torch, RTX SM120/SM89). Only valid with
            ``use_fp8=False``; an A/B reference against the quantized default.
            On GROOT N1.7 the default is FP8 (FP8 backbone + bf16 DiT), so
            ``use_fp8=False`` without ``use_fp16=True`` raises.
        num_steps: Pi0/Pi0.5 torch only when supported. Number of
            flow-matching ODE steps. ``None`` uses the frontend default.
        vision_pool_factor: Pi0.5 torch RTX/Orin only. Spatial pooling factor
            for vision tokens; valid values are 1, 2, or 4. ``None`` keeps
            the frontend default.
        vision_num_layers: Pi0.5 torch RTX/Orin only. Number of SigLIP vision
            layers to execute; valid range is 1-27. ``None`` keeps the
            frontend default.
        cache_frames: Pi0.5 torch RTX/Orin only. Temporal K/V reuse period.
            1 runs the full vision+encoder+decoder path on every frame; 2
            alternates full and decoder-only frames. ``None`` keeps the
            frontend default.
        state_prompt_mode: Pi0.5 torch RTX only. How the variable-length
            state-in-prompt is mapped to CUDA graphs:
              ``"exact"`` (default) — one captured graph per exact prompt
                length, cached; pair with ``warm_state_prompt_buckets()`` to
                front-load the lengths you expect. Unchanged legacy behavior.
              ``"fixed"`` — ONE graph at the max prompt length serves every
                length (padded prefix masked via FA2 ``seqused_k`` + decoder
                K/V appended at the valid offset); a changing length never
                re-captures and no warmup is needed. Requires the vendored
                bf16 FA2 path (``FVK_RTX_FA2=1``, encoder+decoder sites on).
            Env override: ``FLASHRT_PI05_STATE_PROMPT_MODE``.

    Returns:
        VLAModel instance with .predict() method.
    """
    if config not in ("pi05", "groot", "groot_n17", "pi0", "pi0fast",
                      "motus", "wan22_ti2v_5b"):
        raise ValueError(
            f"Unknown config: {config}. "
            f"Supported: pi05, groot, groot_n17, pi0, pi0fast, motus, "
            f"wan22_ti2v_5b")
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

    # GROOT N1.7 on RTX: default to the framework-conforming FP8 frontend. The
    # whole ViT/LLM/VL-self-attn backbone runs FP8 kernels via the SM120-safe
    # descale pattern (no torch matmul on the serving feature path; activation
    # scales come from the on-disk calibration cache, the torch shadow runs only
    # on a cold cache miss). The DiT stays bf16 for Thor parity. ``use_fp16=True``
    # opts into the full-FP16 frontend below (no FP8, non-quantized reference).
    if config == "groot_n17" and framework == "torch" \
            and arch in ("rtx_sm120", "rtx_sm89") and not use_fp16:
        if not use_fp8:
            raise ValueError(
                "GROOT N1.7 on RTX defaults to FP8; there is no separate "
                "non-FP16 BF16 fallback. For the non-quantized full-FP16 "
                "reference pass use_fp16=True, use_fp8=False.")
        from flash_rt.frontends.torch.groot_n17_rtx_fp8 import (
            GrootN17TorchFrontendRtxFP8,
        )
        pipe_cls = GrootN17TorchFrontendRtxFP8

    # GROOT N1.7 on Thor (SM110) runs the FP8 backbone (+ bf16 DiT) by
    # default. There is no BF16-only fallback; the non-quantized reference is
    # the explicit full-FP16 path (use_fp16=True with use_fp8=False), so a
    # bare use_fp8=False is rejected rather than silently ignored.
    if config == "groot_n17" and framework == "torch" and arch == "thor" \
            and not use_fp16 and not use_fp8:
        raise ValueError(
            "GROOT N1.7 on Thor defaults to FP8; there is no BF16-only "
            "fallback. For the non-quantized full-FP16 reference pass "
            "use_fp16=True together with use_fp8=False.")

    if use_fp16:
        if use_fp8:
            raise ValueError("use_fp16=True requires use_fp8=False")
        # GROOT N1.6 Thor full-FP16 reference: the same fully-kernelized,
        # CUDA-graph pipeline as the FP8 production frontend, with the GEMMs run
        # in FP16 instead of per-tensor FP8 (an A/B accuracy baseline).
        if config == "groot" and framework == "torch" and arch == "thor":
            from flash_rt.frontends.torch.groot_thor_fp16 import (
                GrootTorchFrontendThorFP16,
            )
            pipe_cls = GrootTorchFrontendThorFP16
        elif config == "groot_n17" and framework == "torch" and arch == "thor":
            # N1.7 Thor full-FP16 reference (no FP8): ViT / DeepStack / LLM /
            # VL-self-attn run fp16_nn on the shadow weights, and the DiT
            # action head runs the bf16 (non-FP8) graph (_DIT_USE_FP8=False).
            from flash_rt.frontends.torch.groot_n17_thor_fp16 import (
                GrootN17TorchFrontendThorFP16,
            )
            pipe_cls = GrootN17TorchFrontendThorFP16
        else:
            fp16_arches = ("rtx_sm120", "rtx_sm89")
            if config not in ("pi05", "groot", "groot_n17") or framework != "torch" \
                    or arch not in fp16_arches:
                raise ValueError(
                    "use_fp16=True is currently experimental and only supports "
                    "config in {'pi05', 'groot', 'groot_n17'}, framework='torch', "
                    "hardware in {'thor' (groot/groot_n17 only), 'rtx_sm120', "
                    "'rtx_sm89'}")
            if config == "pi05":
                from flash_rt.frontends.torch.pi05_rtx_fp16 import (
                    Pi05TorchFrontendRtxFP16,
                )
                pipe_cls = Pi05TorchFrontendRtxFP16
            elif config == "groot":
                from flash_rt.frontends.torch.groot_rtx_fp16 import (
                    GrootTorchFrontendRtxFP16,
                )
                pipe_cls = GrootTorchFrontendRtxFP16
            else:  # config == "groot_n17"
                from flash_rt.frontends.torch.groot_n17_rtx_fp16 import (
                    GrootN17TorchFrontendRtxFP16,
                )
                pipe_cls = GrootN17TorchFrontendRtxFP16

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
    if "hardware" in sig.parameters:
        kwargs["hardware"] = arch
    if "use_fp8" in sig.parameters:
        kwargs["use_fp8"] = use_fp8
    if config == "pi0fast":
        kwargs.update(
            autotune=autotune,
            decode_cuda_graph=decode_cuda_graph,
            decode_graph_steps=decode_graph_steps,
            max_decode_steps=max_decode_steps,
        )
    elif config in ("groot", "groot_n17"):
        # rtx-side GROOT accepts embodiment_tag + action_horizon; Thor-side
        # GROOT accepts embodiment_tag + autotune. Feature-detect via the
        # concrete class signature so one call site works for both.
        if "autotune" in sig.parameters:
            kwargs["autotune"] = autotune
        if "embodiment_tag" in sig.parameters and embodiment_tag is not None:
            kwargs["embodiment_tag"] = embodiment_tag
        if "action_horizon" in sig.parameters and action_horizon is not None:
            kwargs["action_horizon"] = action_horizon
    elif config == "wan22_ti2v_5b":
        if "autotune" in sig.parameters:
            kwargs["autotune"] = autotune
    else:
        # pi05, pi0 — both Thor and rtx variants take (checkpoint, num_views, autotune)
        # or (checkpoint, num_views). Feature-detect.
        if "autotune" in sig.parameters:
            kwargs["autotune"] = autotune
        if "weight_cache" in sig.parameters:
            kwargs["weight_cache"] = weight_cache
        # Orin-specific performance parameters (passed only when accepted and set).
        if num_steps is not None and "num_steps" in sig.parameters:
            kwargs["num_steps"] = num_steps
        if vision_pool_factor is not None and "vision_pool_factor" in sig.parameters:
            kwargs["vision_pool_factor"] = vision_pool_factor
        if vision_num_layers is not None and "vision_num_layers" in sig.parameters:
            kwargs["vision_num_layers"] = vision_num_layers
        if cache_frames is not None and "cache_frames" in sig.parameters:
            kwargs["cache_frames"] = cache_frames
        # Pi0.5 state-in-prompt graph strategy: "exact" (default, per-length
        # capture) / "fixed" (opt-in, one graph). Forwarded only if accepted.
        if "state_prompt_mode" in sig.parameters:
            kwargs["state_prompt_mode"] = state_prompt_mode
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
