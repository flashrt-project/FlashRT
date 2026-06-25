"""FlashRT -- MelBandRoformer audio source-separation pipeline.

MelBandRoformer is a band-split spectrogram Transformer for audio source
separation. This package ships the FP8 (e4m3) kernelized inference
pipeline (``MelBandRoformerPipeline``): the per-band Transformer blocks
are rewritten as static-quantized FP8 GEMMs over custom CUDA kernels in
``flash_rt_kernels``, while the STFT / band-split / band-merge / ISTFT
host logic runs unchanged from the reference model.
"""

from flash_rt.models.melband_roformer.pipeline import MelBandRoformerPipeline

__all__ = ["MelBandRoformerPipeline"]
