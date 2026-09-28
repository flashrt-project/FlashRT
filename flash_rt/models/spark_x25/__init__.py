"""Spark-X2.5-4B (SM120): hybrid sliding/full attention, NVFP4 weights.

The pipeline module holds the decode/prefill kernel sequence; the frontend
(``flash_rt.frontends.torch.spark_x25_rtx``) wires it to a checkpoint and a
generation loop. See ``docs/spark_x25_usage.md``.
"""

from flash_rt.models.spark_x25.config import SparkX25Config, load_config

__all__ = ["SparkX25Config", "load_config", "SparkX25Runtime"]


def __getattr__(name: str):
    """Import ``SparkX25Runtime`` on first use (PEP 562).

    The runtime module pulls in the SM120-gated extensions, so binding it
    eagerly would make ``import flash_rt.models.spark_x25`` -- and with it the
    config parsing and checkpoint validation the frontend shares -- fail on any
    machine that has not built ``flash_rt_sparkx25``. The runtime constructor
    raises the clear refusal instead; see ``_require_kernels``.
    """
    if name == "SparkX25Runtime":
        from flash_rt.models.spark_x25.pipeline_rtx import SparkX25Runtime
        return SparkX25Runtime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
