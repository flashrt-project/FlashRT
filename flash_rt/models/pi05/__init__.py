"""FlashRT Pi0.5 model pipelines."""

VIS_L = 27
VIS_D = 1152
VIS_H = 4304
VIS_NH = 16
VIS_HD = 72
VIS_SEQ_PER_VIEW = 256
VIS_PATCH_FLAT = 14 * 14 * 3

ENC_L = 18
ENC_D = 2048
ENC_H = 16384
ENC_NH = 8
ENC_NKV = 1
ENC_HD = 256

DEC_L = 18
DEC_D = 1024
DEC_H = 4096
DEC_NH = 8
DEC_NKV = 1
DEC_HD = 256

ACTION_DIM = 32
NUM_STEPS_DEFAULT = 10


def __getattr__(name: str):
    if name == "Pi05Pipeline":
        from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline

        return Pi05Pipeline
    if name == "Pi05PipelineRocm":
        from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

        return Pi05PipelineRocm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Pi05Pipeline",
    "Pi05PipelineRocm",
    "VIS_L",
    "VIS_D",
    "VIS_H",
    "VIS_NH",
    "VIS_HD",
    "VIS_SEQ_PER_VIEW",
    "VIS_PATCH_FLAT",
    "ENC_L",
    "ENC_D",
    "ENC_H",
    "ENC_NH",
    "ENC_NKV",
    "ENC_HD",
    "DEC_L",
    "DEC_D",
    "DEC_H",
    "DEC_NH",
    "DEC_NKV",
    "DEC_HD",
    "ACTION_DIM",
    "NUM_STEPS_DEFAULT",
]
