"""FlashRT -- Pi0.5 model pipelines.

The RTX pipeline class and its dimension constants are re-exported
here so that frontends and tests can
``from flash_rt.models.pi05 import ...`` without knowing the
internal file layout.

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
]


def __getattr__(name):
    if name in __all__:
        from importlib import import_module
        value = getattr(import_module("flash_rt.models.pi05.pipeline_rtx"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
