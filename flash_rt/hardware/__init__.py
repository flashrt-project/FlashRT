"""FlashRT hardware-dispatch layer.

Detects the current GPU's compute capability and maps
``(config, framework, arch)`` triples to concrete frontend classes in
``flash_rt.frontends.*``.

``flash_rt.api.load_model`` calls ``resolve_pipeline_class`` so user
code doesn't need to know whether it's running on Jetson Thor (SM110),
an RTX 5090 (SM120), or an RTX 4090 (SM89).

Adding a new model
-------------------
External packages can register new models by mutating ``_PIPELINE_MAP``
at import time::

    from flash_rt.hardware import _PIPELINE_MAP
    _PIPELINE_MAP[("mymodel", "torch", "rtx_sm120")] = (
        "mymodel_plugin.frontend", "MyModelTorchFrontend"
    )

See ``docs/plugin_model_template.md`` for the full worked example.

Adding a new hardware target
-----------------------------
Extend ``detect_arch`` to return a new arch string, then add entries
to ``_PIPELINE_MAP`` for each (config, framework, new_arch) combination.
"""

from __future__ import annotations


def detect_arch() -> str:
    """Return a short string identifier for the current CUDA device.

    Supported:
        ``"thor"``      — Jetson AGX Thor, SM110 (cc 11.0)
        ``"rtx_sm120"`` — RTX 5090 / Blackwell consumer, SM120 (cc 12.0)
        ``"rtx_sm89"``  — RTX 4090 / Ada, SM89 (cc 8.9)
        ``"rtx_sm87"``  — Jetson Orin via RTX consumer backend, SM87 (cc 8.7)

    Raises RuntimeError if CUDA is unavailable or the card has an
    unsupported SM level. Deliberately strict: silently falling back to
    the wrong backend would hide latency/correctness regressions.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "FlashRT requires PyTorch for GPU detection") from e
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FlashRT requires a CUDA-capable GPU "
            "(torch.cuda.is_available()==False)")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) == (11, 0):
        return "thor"
    if (major, minor) == (12, 0):
        return "rtx_sm120"
    if (major, minor) == (8, 7):
        return "rtx_sm87"
    if (major, minor) == (8, 9):
        return "rtx_sm89"
    raise RuntimeError(
        f"FlashRT: unsupported GPU SM {major}.{minor}. "
        f"Supported architectures: SM110 (Thor), SM120 (RTX 5090), "
        f"SM89 (RTX 4090), SM87 (Jetson Orin experimental)."
    )


# Dispatch table: (config, framework, arch) → (module_path, class_name).
# Resolved lazily at load_model time so importing ``flash_rt`` does not
# drag in every backend. External plugins may add entries to this dict
# to register new models — see ``docs/plugin_model_template.md``.
_PIPELINE_MAP: dict[tuple[str, str, str], tuple[str, str]] = {
    # ── Pi0.5 ──
    ("pi05", "torch", "thor"):
        ("flash_rt.frontends.torch.pi05_thor", "Pi05TorchFrontendThor"),
    ("pi05", "torch", "rtx_sm120"):
        ("flash_rt.frontends.torch.pi05_rtx", "Pi05TorchFrontendRtx"),
    ("pi05", "torch", "rtx_sm87"):
        ("flash_rt.frontends.torch.pi05_rtx", "Pi05TorchFrontendRtx"),
    ("pi05", "torch", "rtx_sm89"):
        ("flash_rt.frontends.torch.pi05_rtx", "Pi05TorchFrontendRtx"),
    ("pi05", "jax", "thor"):
        ("flash_rt.frontends.jax.pi05_thor", "Pi05JaxFrontendThor"),
    ("pi05", "jax", "rtx_sm120"):
        ("flash_rt.frontends.jax.pi05_rtx", "Pi05JaxFrontendRtx"),
    ("pi05", "jax", "rtx_sm89"):
        ("flash_rt.frontends.jax.pi05_rtx", "Pi05JaxFrontendRtx"),

    # ── Pi0 ── (Thor native + RTX consumer via pipeline_rtx.py.)
    ("pi0", "torch", "thor"):
        ("flash_rt.frontends.torch.pi0_thor", "Pi0TorchFrontendThor"),
    ("pi0", "torch", "rtx_sm120"):
        ("flash_rt.frontends.torch.pi0_rtx", "Pi0TorchFrontendRtx"),
    ("pi0", "torch", "rtx_sm89"):
        ("flash_rt.frontends.torch.pi0_rtx", "Pi0TorchFrontendRtx"),
    ("pi0", "jax", "thor"):
        ("flash_rt.frontends.jax.pi0_thor", "Pi0JaxFrontendThor"),
    ("pi0", "jax", "rtx_sm120"):
        ("flash_rt.frontends.jax.pi0_rtx", "Pi0JaxFrontendRtx"),
    ("pi0", "jax", "rtx_sm89"):
        ("flash_rt.frontends.jax.pi0_rtx", "Pi0JaxFrontendRtx"),

    # ── GROOT N1.6 ──
    ("groot", "torch", "thor"):
        ("flash_rt.frontends.torch.groot_thor", "GrootTorchFrontendThor"),
    ("groot", "torch", "rtx_sm120"):
        ("flash_rt.frontends.torch.groot_rtx", "GrootTorchFrontendRtx"),

    # ── Motus (Wan2.2 + Qwen-VL + action/understanding experts) ──
    # RTX 5090 path only for now. Motus uses a bundle-based E2E contract
    # rather than the image-list VLA API used by Pi0/Pi0.5/GROOT.
    ("motus", "torch", "rtx_sm120"):
        ("flash_rt.frontends.torch.motus_rtx", "MotusTorchFrontendRtx"),

    # ── Pi0-FAST ── (SM120 runtime fork inside pipeline, no AttentionBackend protocol.)
    ("pi0fast", "torch", "thor"):
        ("flash_rt.frontends.torch.pi0fast", "Pi0FastTorchFrontend"),
    ("pi0fast", "torch", "rtx_sm120"):
        ("flash_rt.frontends.torch.pi0fast", "Pi0FastTorchFrontend"),
    ("pi0fast", "jax", "thor"):
        ("flash_rt.frontends.jax.pi0fast", "Pi0FastJaxFrontend"),
    ("pi0fast", "jax", "rtx_sm120"):
        ("flash_rt.frontends.jax.pi0fast", "Pi0FastJaxFrontend"),
}


def resolve_pipeline_class(config: str, framework: str, arch: str):
    """Resolve (config, framework, arch) to a pipeline class object.

    Lazily imports the backend module — touching ``flash_rt.hardware``
    does not pull in torch/jax/rtx code until a load happens.
    """
    key = (config, framework, arch)
    if arch == "rtx_sm87" and key != ("pi05", "torch", "rtx_sm87"):
        raise RuntimeError(
            "FlashRT: Jetson Orin SM87 currently supports only "
            "config='pi05' with framework='torch'. "
            f"config={config!r} framework={framework!r} is not supported yet."
        )
    if key not in _PIPELINE_MAP:
        supported = sorted(
            (c, f, a) for (c, f, a) in _PIPELINE_MAP
            if c == config and f == framework
        )
        if supported:
            hint = (f"This model/framework combo is built for: "
                    f"{[a for (_, _, a) in supported]}")
        else:
            hint = (f"No backend for config={config!r} "
                    f"framework={framework!r} in any supported architecture.")
        raise RuntimeError(
            f"FlashRT: no pipeline for "
            f"config={config!r} framework={framework!r} arch={arch!r}. "
            f"{hint}"
        )
    module_path, cls_name = _PIPELINE_MAP[key]
    module = __import__(module_path, fromlist=[cls_name])
    return getattr(module, cls_name)
