from __future__ import annotations
import os

import pathlib

import ml_dtypes
import numpy as np
import torch

from flash_rt import flash_rt_rocm_kernels as rocm
from flash_rt.frontends.torch.pi05_rocm import _load_openpi_model
from flash_rt.frontends.torch.pi05_rocm_weights import (
    build_rocm_vision_weights_from_openpi_model,
)
from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm


CHECKPOINT = pathlib.Path(
    os.environ.get("PI05_CHECKPOINT", "lerobot/pi05_libero_finetuned")
)


def reset_inputs(pipe: Pi05PipelineRocm, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    images = rng.normal(0.0, 0.02, size=(3, 224, 224, 3)).astype(np.float32)
    noise = rng.normal(0.0, 0.2, size=(pipe.chunk_size, 32)).astype(np.float32)
    encoder_x = rng.normal(
        0.0, 0.01, size=(pipe.encoder_seq_len, 2048)
    ).astype(np.float32)
    pipe.input_images_buf.upload(images.astype(ml_dtypes.bfloat16))
    pipe.input_noise_buf.upload(noise.astype(ml_dtypes.bfloat16))
    pipe.bufs["encoder_x"].upload(encoder_x.astype(ml_dtypes.bfloat16))


def read_noise(pipe: Pi05PipelineRocm) -> torch.Tensor:
    out = pipe.input_noise_buf.download_new((pipe.chunk_size, 32), np.uint16)
    return torch.from_numpy(out).view(torch.bfloat16).float().cuda()


def summarize(name: str, ref: torch.Tensor, got: torch.Tensor) -> None:
    ref_f = ref.flatten().float()
    got_f = got.flatten().float()
    diff = got_f - ref_f
    cos = torch.nn.functional.cosine_similarity(ref_f, got_f, dim=0).item()
    rel_l2 = (diff.norm() / ref_f.norm().clamp_min(1e-12)).item()
    print(
        name,
        {
            "cosine": cos,
            "max_abs": diff.abs().max().item(),
            "mean_abs": diff.abs().mean().item(),
            "rel_l2": rel_l2,
            "ref_finite": torch.isfinite(ref_f).all().item(),
            "got_finite": torch.isfinite(got_f).all().item(),
            "ref_sum": ref_f.sum().item(),
            "got_sum": got_f.sum().item(),
        },
    )


def main() -> None:
    print("torch", torch.__version__, "hip", torch.version.hip)
    model, _cfg, path = _load_openpi_model(CHECKPOINT)
    print("loaded", path)
    weights = build_rocm_vision_weights_from_openpi_model(model, include_fp8=True)

    bf16_pipe = Pi05PipelineRocm.with_sdpa_attention(
        num_views=3,
        max_prompt_len=48,
        chunk_size=10,
        num_steps=10,
    )
    fp8_pipe = Pi05PipelineRocm.with_sdpa_attention(
        num_views=3,
        max_prompt_len=48,
        chunk_size=10,
        num_steps=10,
    )

    reset_inputs(bf16_pipe, seed=123)
    bf16_pipe.bake_bf16_gemms(rocm)
    bf16_pipe.run_bf16_pipeline(rocm, weights)
    rocm.hip_sync()
    bf16_out = read_noise(bf16_pipe)

    reset_inputs(fp8_pipe, seed=123)
    fp8_pipe.bake_fp8_gemms(rocm)
    cal = fp8_pipe.calibrate_fp8(rocm, weights)
    torch.cuda.synchronize()
    print("scale_count", cal["scale_count"])

    reset_inputs(fp8_pipe, seed=123)
    fp8_pipe.run_fp8_pipeline(rocm, weights)
    torch.cuda.synchronize()
    fp8_eager_out = read_noise(fp8_pipe)
    summarize("BF16_vs_FP8_EAGER", bf16_out, fp8_eager_out)

    reset_inputs(fp8_pipe, seed=123)
    fp8_pipe.capture_fp8_graph(rocm, weights)
    torch.cuda.synchronize()
    reset_inputs(fp8_pipe, seed=123)
    fp8_pipe.replay_fp8_graph()
    torch.cuda.synchronize()
    fp8_graph_out = read_noise(fp8_pipe)
    summarize("BF16_vs_FP8_GRAPH", bf16_out, fp8_graph_out)
    summarize("FP8_EAGER_vs_FP8_GRAPH", fp8_eager_out, fp8_graph_out)


if __name__ == "__main__":
    main()
