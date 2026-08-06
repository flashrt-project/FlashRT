#!/usr/bin/env python3
"""Build TensorRT FP16 engines for the Chameleon VQ-GAN encoder.

Exports fixed-shape ONNX per resolution, then compiles TRT engines.
Engines are cached at ~/.flash_rt/trt_engines/vqgan/ (or --output_dir).

Must be run on the TARGET hardware (engines are not portable across GPUs).

Usage
-----
python scripts/build_vqgan_trt.py \
    --cfg_path /path/to/chameleon/tokenizer/vqgan.yaml \
    --ckpt_path /path/to/chameleon/tokenizer/vqgan.ckpt \
    --resolutions 384x512 384x384 384x672 512x512 \
    --verify
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt

from flash_rt.models.chameleon.vqgan import ImageTokenizer


class VQGANEncoderWrapper(nn.Module):
    """Image tensor -> codebook indices.

    Input  : x        of shape (B, 3, H, W), float32 in [-1, 1]
    Output : indices  of shape (B, H/16, W/16), int64
    """

    def __init__(self, vq_model: nn.Module):
        super().__init__()
        self.encoder = vq_model.encoder
        self.quant_conv = vq_model.quant_conv
        # Re-host the codebook as a self-contained nn.Embedding so this
        # wrapper is a fully standard nn.Module (no external parameter sharing).
        n_e, e_dim = vq_model.quantize.embedding.weight.shape
        self.codebook = nn.Embedding(n_e, e_dim)
        with torch.no_grad():
            self.codebook.weight.copy_(vq_model.quantize.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)              # (B, 2*z_channels, H/16, W/16) when double_z=True
        h = self.quant_conv(h)           # (B, e_dim, H/16, W/16)

        b, c, hh, ww = h.shape
        # (B, e_dim, H/16, W/16) -> (B, H/16, W/16, e_dim) -> (B*N, e_dim)
        z_flat = h.permute(0, 2, 3, 1).contiguous().view(-1, c)
        e = self.codebook.weight         # (n_e, e_dim)

        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        d = (
            (z_flat * z_flat).sum(dim=1, keepdim=True)
            + (e * e).sum(dim=1)
            - 2.0 * torch.matmul(z_flat, e.t())
        )                                # (B*N, n_e)
        idx = torch.argmin(d, dim=1)     # (B*N,)
        idx = idx.view(b, hh, ww).to(torch.int64)
        return idx


def build_vqmodel(cfg_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    tokenizer = ImageTokenizer(cfg_path=cfg_path, ckpt_path=ckpt_path, device=device)
    vq_model = tokenizer._vq_model.eval()
    for p in vq_model.parameters():
        p.requires_grad_(False)
    return vq_model


def parse_resolution(s: str) -> tuple:
    h, w = s.lower().split("x")
    return int(h), int(w)


def export_onnx(vq_model, height, width, batch, opset, output_path, device):
    wrapper = VQGANEncoderWrapper(vq_model).to(device).eval()
    dummy = torch.randn(batch, 3, height, width, device=device, dtype=torch.float32)
    export_kwargs = dict(
        input_names=["image"],
        output_names=["indices"],
        dynamic_axes=None,
        opset_version=opset,
        do_constant_folding=True,
    )
    # PyTorch 2.5+ defaults to dynamo=True which emits TRT-incompatible
    # IR-10/opset-18 nodes. Force legacy exporter on those versions.
    # On older PyTorch (< 2.5) the kwarg doesn't exist and isn't needed.
    _torch_ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if _torch_ver >= (2, 5):
        export_kwargs["dynamo"] = False
    torch.onnx.export(wrapper, dummy, output_path, **export_kwargs)
    print(f"  [onnx] exported {output_path} (shape=[{batch},3,{height},{width}])")
    return wrapper


def build_engine(onnx_path, engine_path, workspace_gb, opt_level, fp16=True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # Choose network creation flags. TRT 11+ supports STRONGLY_TYPED which
    # makes the engine honor the ONNX's native dtypes verbatim (so FP16
    # weights stay FP16). Older TRT uses EXPLICIT_BATCH + BuilderFlag.FP16
    # which silently demotes to FP32 for some ops on Ada (we observed this
    # on TRT 11.1 — the engine ran ~2× slower in FP32 by default).
    use_strong = (fp16 and
                  hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"))

    if use_strong:
        # Strongly typed mode requires the ONNX itself to be FP16.
        # Auto-convert from the FP32 ONNX (idempotent: skips if already FP16).
        from onnxconverter_common import float16
        import onnx as _onnx_mod

        onnx_path_obj = Path(onnx_path)
        fp16_onnx = onnx_path_obj.with_suffix(".fp16.onnx")
        if not fp16_onnx.exists():
            print(f"  [onnx-fp16] converting {onnx_path_obj.name} → "
                  f"{fp16_onnx.name}")
            mdl = _onnx_mod.load(str(onnx_path_obj))
            mdl16 = float16.convert_float_to_float16(mdl, keep_io_types=True)
            _onnx_mod.save(mdl16, str(fp16_onnx))
        else:
            print(f"  [onnx-fp16] reusing existing {fp16_onnx.name}")
        parse_path = str(fp16_onnx)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    elif hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        # Legacy path (TRT < 10).
        parse_path = str(onnx_path)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    else:
        # Very old TRT — default network.
        parse_path = str(onnx_path)
        network = builder.create_network(0)

    parser = trt.OnnxParser(network, logger)
    with open(parse_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  [trt] parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse ONNX: {parse_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                  int(workspace_gb * (1 << 30)))
    if fp16 and not use_strong and hasattr(trt.BuilderFlag, "FP16"):
        # Old TRT path that needs the explicit FP16 flag.
        config.set_flag(trt.BuilderFlag.FP16)
    config.builder_optimization_level = opt_level

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.time() - t0

    if serialized is None:
        raise RuntimeError(f"TRT engine build failed for {onnx_path}")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  [trt] built {engine_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")
    return engine_path


def verify_engine(engine_path, torch_wrapper, height, width, batch, device):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print("  [verify] FAILED: could not deserialize engine")
        return False

    context = engine.create_execution_context()
    stream = torch.cuda.current_stream()

    h_lat, w_lat = height // 16, width // 16
    inp = torch.randn(batch, 3, height, width, device=device, dtype=torch.float32)
    out = torch.zeros(batch, h_lat, w_lat, device=device, dtype=torch.int64)

    # TRT may use int32 for output; detect binding dtype
    out_name = "indices"
    out_dtype_trt = engine.get_tensor_dtype(out_name)
    if out_dtype_trt == trt.DataType.INT32:
        out_buf = torch.zeros(batch, h_lat, w_lat, device=device, dtype=torch.int32)
    else:
        out_buf = out

    context.set_tensor_address("image", inp.data_ptr())
    context.set_tensor_address("indices", out_buf.data_ptr())
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    if out_dtype_trt == trt.DataType.INT32:
        trt_indices = out_buf.to(torch.int64)
    else:
        trt_indices = out_buf

    with torch.no_grad():
        pt_indices = torch_wrapper(inp)

    match = (trt_indices == pt_indices).sum().item()
    total = trt_indices.numel()
    mismatch_pct = 100.0 * (1.0 - match / total)
    ok = mismatch_pct < 0.5
    print(f"  [verify] match={match}/{total} ({100*match/total:.2f}%), "
          f"mismatch={mismatch_pct:.3f}% {'OK' if ok else 'WARN'}")
    return ok


def compute_ckpt_hash(ckpt_path: str) -> str:
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Build TRT engines for VQ-GAN encoder")
    parser.add_argument("--cfg_path", type=str, required=True,
                        help="Path to the Chameleon VQ-GAN vqgan.yaml config")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the Chameleon VQ-GAN vqgan.ckpt checkpoint")
    parser.add_argument("--resolutions", nargs="+", default=["384x512", "384x384", "384x672", "512x512"],
                        help="HxW resolutions to build engines for")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output_dir", type=str,
                        default=str(Path.home() / ".flash_rt" / "trt_engines" / "vqgan"))
    parser.add_argument("--workspace_gb", type=float, default=2.0)
    parser.add_argument("--opt_level", type=int, default=5,
                        help="TRT builder optimization level (0-5)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--keep_onnx", action="store_true",
                        help="Keep intermediate ONNX files in output_dir")
    parser.add_argument("--verify", action="store_true",
                        help="Run parity check TRT vs PyTorch after build")
    args = parser.parse_args()

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"TensorRT {trt.__version__}")
    print(f"Platform: {platform.machine()}")
    print(f"Output dir: {output_dir}")
    print(f"Resolutions: {args.resolutions}")
    print()

    vq_model = build_vqmodel(args.cfg_path, args.ckpt_path, device)
    ckpt_hash = compute_ckpt_hash(args.ckpt_path)

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["build_date"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        manifest = {
            "trt_version": trt.__version__,
            "platform": platform.machine(),
            "ckpt_hash": ckpt_hash,
            "build_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "batch": args.batch,
            "precision": "fp16",
            "engines": {},
        }

    for res_str in args.resolutions:
        height, width = parse_resolution(res_str)
        assert height % 16 == 0 and width % 16 == 0, f"H/W must be multiples of 16, got {res_str}"
        h_lat, w_lat = height // 16, width // 16

        print(f"── {res_str} ({height}×{width} → {h_lat}×{w_lat} latent) ──")

        engine_name = f"vqgan_encoder_b{args.batch}_{height}x{width}_fp16.engine"
        engine_path = output_dir / engine_name

        # Export ONNX
        onnx_path = output_dir / f"vqgan_encoder_{height}x{width}.onnx"
        wrapper = export_onnx(vq_model, height, width, args.batch, args.opset,
                              str(onnx_path), device)

        # Build TRT engine
        build_engine(str(onnx_path), str(engine_path), args.workspace_gb, args.opt_level)

        # Verify
        if args.verify:
            verify_engine(str(engine_path), wrapper.to(device), height, width, args.batch, device)

        # Clean ONNX
        if not args.keep_onnx:
            onnx_path.unlink(missing_ok=True)

        manifest["engines"][res_str] = {
            "file": engine_name,
            "height": height,
            "width": width,
            "input_shape": [args.batch, 3, height, width],
            "output_shape": [args.batch, h_lat, w_lat],
        }
        print()

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
