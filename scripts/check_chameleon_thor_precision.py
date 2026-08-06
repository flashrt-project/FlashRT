#!/usr/bin/env python3
"""Real-image precision gate for standalone Chameleon-7B on Thor.

Compares:
  * FlashRT FP16 vs HF BF16 last-token logits (cosine, top-k overlap,
    greedy next-token equality)
  * FlashRT dynamic FP8 vs FlashRT FP16 last-token logits (same metrics)
  * optional final-hidden cosine

Inputs are always real images (from a user-supplied directory), never
synthetic token ids, per the standalone Chameleon-7B optimization plan.

Usage
-----
    PYTHONPATH=. python scripts/check_chameleon_thor_precision.py \\
        --checkpoint /path/to/Chameleon_7B_mGPT \\
        --image-dir /path/to/images \\
        --prompt "Describe the image." \\
        --output /tmp/chameleon_thor_precision.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def _load_hf_bf16(checkpoint_dir: pathlib.Path):
    """Load the plain HF ChameleonForConditionalGeneration at bf16.

    HF ``from_pretrained`` silently mis-loads q/k norm weights on this
    checkpoint's old [1,128] shape; use a naked model + manual
    ``load_state_dict`` instead. Requires the model's reference
    ``modeling_chameleon`` implementation (optional; ``--skip-hf``
    skips this comparison).
    """
    import torch
    from transformers import AutoConfig
    from transformers import ChameleonForConditionalGeneration as _Cls
    from safetensors.torch import load_file

    cfg = AutoConfig.from_pretrained(str(checkpoint_dir))
    cfg.rope_scaling = None
    if not hasattr(cfg, "rope_theta") or cfg.rope_theta is None:
        cfg.rope_theta = 10000.0

    model = _Cls(cfg)
    sd = {}
    for shard in sorted(checkpoint_dir.glob("model-*-of-*.safetensors")):
        sd.update(load_file(str(shard)))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=False)
    non_vq_missing = [k for k in missing if "vqmodel" not in k]
    if non_vq_missing:
        print(f"[_load_hf_bf16] WARNING: {len(non_vq_missing)} "
              f"non-VQVAE keys missing from ckpt")
    if unexpected:
        print(f"[_load_hf_bf16] WARNING: {len(unexpected)} unexpected keys")
    model = model.to(torch.bfloat16).cuda().eval()
    return model


def _hf_last_logits(model, input_ids: list[int]) -> np.ndarray:
    import torch
    ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    return logits


def _load_real_images(image_dir: pathlib.Path, max_images: int):
    from PIL import Image
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No real images found under {image_dir}")
    paths = paths[:max_images]
    images = [Image.open(p).convert("RGB") for p in paths]
    return images, [str(p) for p in paths]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    top_a = set(np.argsort(-a)[:k].tolist())
    top_b = set(np.argsort(-b)[:k].tolist())
    return len(top_a & top_b) / float(k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-dir", required=True,
                    help="Directory of real input images")
    ap.add_argument("--prompt", default="Describe the image.")
    ap.add_argument("--max-images", type=int, default=1)
    ap.add_argument("--target-size", type=int, default=512)
    ap.add_argument("--use-trt-vqgan", action="store_true",
                    help="Use TensorRT VQGAN if compatible engines exist "
                         "(recommended when available; default is eager VQGAN)")
    ap.add_argument("--trt-vqgan-engine-dir", default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--output", default="/tmp/chameleon_thor_precision.json")
    ap.add_argument("--skip-hf", action="store_true",
                     help="Skip HF BF16 comparison (FP16 vs FP8 only)")
    args = ap.parse_args()

    checkpoint_dir = pathlib.Path(args.checkpoint)
    image_dir = pathlib.Path(args.image_dir)
    images, image_paths = _load_real_images(image_dir, args.max_images)
    print(f"[check] loaded {len(images)} real image(s) from {image_dir}: "
          f"{image_paths}")

    from flash_rt.frontends.torch.chameleon_thor import ChameleonTorchFrontendThor

    result: dict = {
        "checkpoint": str(checkpoint_dir),
        "image_dir": str(image_dir),
        "image_paths": image_paths,
        "prompt": args.prompt,
        "target_size": args.target_size,
        "use_trt_vqgan": bool(args.use_trt_vqgan),
        "trt_vqgan_engine_dir": args.trt_vqgan_engine_dir,
        "vqgan_backend_requested": "trt" if args.use_trt_vqgan else "eager",
    }

    print("[check] running FlashRT FP16 reference path...")
    fe_fp16 = ChameleonTorchFrontendThor(
        str(checkpoint_dir), use_fp8=False, use_cuda_graph=False,
        target_size=args.target_size,
        use_trt_vqgan=args.use_trt_vqgan,
        trt_vqgan_engine_dir=args.trt_vqgan_engine_dir)
    out_fp16 = fe_fp16.prefill(args.prompt, images)
    logits_fp16 = out_fp16["logits"].numpy().ravel()
    hidden_fp16 = out_fp16["hidden"].numpy()
    ids_fp16 = out_fp16["input_ids"]
    result["vqgan_backend_actual_fp16"] = out_fp16.get("vqgan_backend")
    del fe_fp16
    import torch
    torch.cuda.empty_cache()

    print("[check] running FlashRT dynamic-FP8 path...")
    fe_fp8 = ChameleonTorchFrontendThor(
        str(checkpoint_dir), use_fp8=True, use_cuda_graph=False,
        target_size=args.target_size,
        use_trt_vqgan=args.use_trt_vqgan,
        trt_vqgan_engine_dir=args.trt_vqgan_engine_dir)
    out_fp8 = fe_fp8.prefill(args.prompt, images)
    logits_fp8 = out_fp8["logits"].numpy().ravel()
    hidden_fp8 = out_fp8["hidden"].numpy()
    result["vqgan_backend_actual_fp8"] = out_fp8.get("vqgan_backend")
    del fe_fp8
    torch.cuda.empty_cache()

    fp8_vs_fp16 = {
        "logits_cosine": _cosine(logits_fp8, logits_fp16),
        "topk_overlap": _topk_overlap(logits_fp8, logits_fp16, args.topk),
        "greedy_token_match": bool(
            int(np.argmax(logits_fp8)) == int(np.argmax(logits_fp16))),
        "hidden_cosine": _cosine(hidden_fp8, hidden_fp16),
    }
    result["flashrt_fp8_vs_flashrt_fp16"] = fp8_vs_fp16
    print(f"[check] FlashRT FP8 vs FP16: {fp8_vs_fp16}")

    if not args.skip_hf:
        print("[check] loading HF BF16 model (this may take a while)...")
        try:
            hf_model = _load_hf_bf16(checkpoint_dir)
        except (ImportError, ModuleNotFoundError) as e:
            print(f"[check] HF BF16 reference unavailable ({e}); "
                  f"rerun with --skip-hf for the FP16-vs-FP8 check only")
            hf_model = None
        if hf_model is not None:
            logits_hf = _hf_last_logits(hf_model, ids_fp16)
            del hf_model
            torch.cuda.empty_cache()

            fp16_vs_hf = {
                "logits_cosine": _cosine(logits_fp16, logits_hf),
                "topk_overlap": _topk_overlap(logits_fp16, logits_hf, args.topk),
                "greedy_token_match": bool(
                    int(np.argmax(logits_fp16)) == int(np.argmax(logits_hf))),
            }
            result["flashrt_fp16_vs_hf_bf16"] = fp16_vs_hf
            print(f"[check] FlashRT FP16 vs HF BF16: {fp16_vs_hf}")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[check] wrote {args.output}")


if __name__ == "__main__":
    main()
