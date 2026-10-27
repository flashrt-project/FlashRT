"""LaWAM (Latent World Action Model) policy on FlashRT.

LaWAM = Qwen3-VL-2B (first 16 LLM layers) + a VLM-to-latent QFormer + a
latent action model (DINOv3 ViT-B encoder and a 12-block AdaLN decoder that
predicts the next-frame features) + a 16-block alternate VL DiT flow-matching
action head (GR00T-style), released as the LeRobot checkpoint
``jialei02/lawam-pretrain-lerobot``.

Compute lives in :mod:`flash_rt.models.lawam.pipeline_rtx`; the torch
frontend is :mod:`flash_rt.frontends.torch.lawam_rtx`.
"""
