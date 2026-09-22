"""Spark-X2.5-4B (SM120): hybrid sliding/full attention, NVFP4 weights.

The pipeline module holds the decode/prefill kernel sequence; the frontend
(``flash_rt.frontends.torch.spark_x25_rtx``) wires it to a checkpoint and a
generation loop. See ``docs/spark_x25_usage.md``.
"""

from flash_rt.models.spark_x25.config import SparkX25Config, load_config
from flash_rt.models.spark_x25.pipeline_rtx import SparkX25Runtime

__all__ = ["SparkX25Config", "load_config", "SparkX25Runtime"]
