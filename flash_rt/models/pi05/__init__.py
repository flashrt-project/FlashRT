"""FlashRT -- Pi0.5 model pipelines.

The RTX pipeline class and its dimension constants are re-exported
here so that frontends and tests can
``from flash_rt.models.pi05 import ...`` without knowing the
internal file layout.

There are two RTX pipelines here for two different checkpoints, and they are
separate implementations on purpose:

    pipeline_rtx.py   - RTX SM120/SM89 Pi05Pipeline class: Gemma-2B encoder at
                        18 layers of D=2048 with one KV head and interleaved rope
    pipeline.py       - NativeDM05: the DM05-LIBERO checkpoint, Gemma3 at 34 layers
                        of D=2560 with four KV heads, per-head q/k RMSNorm, HF
                        rotate-half rope and a sliding/full layer-type split

Everything below ``pipeline.py`` -- vision.py, text.py, action.py and the site
modules -- is the DM05 implementation, ported from the opendm repository. It
reaches its kernels through ``_ext``, an adapter over this repository's kernels.

Per the unified pipeline_<hw>.py contract:
    pipeline_thor.py  - Thor SM110 decoder forward fns
    pipeline_rtx.py   - RTX SM120/SM89 Pi05Pipeline class
"""

__all__ = [
    "Pi05Pipeline",
    "VIS_L", "VIS_D", "VIS_H", "VIS_NH", "VIS_HD",
    "VIS_SEQ_PER_VIEW", "VIS_PATCH_FLAT",
    "ENC_L", "ENC_D", "ENC_H", "ENC_NH", "ENC_NKV", "ENC_HD",
    "DEC_L", "DEC_D", "DEC_H", "DEC_NH", "DEC_NKV", "DEC_HD",
    "ACTION_DIM", "NUM_STEPS_DEFAULT",
    # DM05-LIBERO: the ported pipeline and the three towers it owns
    "NativeDM05", "NativeConfig", "segments_from_token_types",
    "NativeVision", "NativePrefix", "NativeActionExpert",
]

# Where each exported name lives. The DM05 modules import ``_ext``, which imports
# the DM05 kernel module, so they stay behind __getattr__ and are only loaded when
# something actually asks for them -- a build without FLASHRT_ENABLE_DM05 still
# imports this package.
_SOURCE = {"NativeDM05": "pipeline", "NativeConfig": "pipeline",
           "segments_from_token_types": "pipeline", "NativeVision": "vision",
           "NativePrefix": "text", "NativeActionExpert": "action"}


def __getattr__(name):
    if name in __all__:
        from importlib import import_module
        mod = _SOURCE.get(name, "pipeline_rtx")
        value = getattr(import_module(f"flash_rt.models.pi05.{mod}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
