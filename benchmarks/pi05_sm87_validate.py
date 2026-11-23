"""Reproduce Pi0.5 SM87 output checks without external adapters or downloads.

Use the same checkpoint, options and saved reference across tile OFF/ON builds.
--compare-strict additionally compares both consumer paths in one instance.
This generates offline actions only and never drives a robot.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--save-reference", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--compare-strict", action="store_true")
    args = parser.parse_args()
    if args.reps < 1 or args.warmup < 1:
        parser.error("reps and warmup must be positive")
    os.environ["FVK_PI05_RTX_FORCE_INT8"] = "1"
    os.environ.pop("FLASHRT_PI05_STRICT_VISUAL", None)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.get_device_capability() != (8,7):
        raise RuntimeError("This reproduction targets the validated SM87 path")
    from flash_rt import flash_rt_kernels as kernels
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline
    if args.compare_strict and not (
        hasattr(kernels,"pi05_strict_visual_available") and kernels.pi05_strict_visual_available()
    ):
        raise RuntimeError("Build with FLASHRT_ENABLE_PI05_STRICT_VISUAL_SM87=ON")
    # Keep cross-process BF16 algorithm selection consistent. Timings below
    # validate this harness, not the peak throughput of autotuned GEMMs.
    original_tune = Pi05Pipeline.autotune_gemms
    Pi05Pipeline.autotune_gemms = lambda self: None
    try:
        model = Pi05TorchFrontendRtx(
            args.checkpoint,num_views=2,chunk_size=15,num_steps=10,
            vision_pool_factor=1,vision_num_layers=27,cache_frames=1,
            hardware="rtx_sm87",use_fp8=False,state_prompt_mode="exact",
            use_cuda_graph=True)
        model.set_prompt("pick up the object")
        rng = np.random.default_rng(73)
        image_sets = [
            {"image":rng.integers(0,256,(224,224,3),dtype=np.uint8),
             "wrist_image":rng.integers(0,256,(224,224,3),dtype=np.uint8)}
            for _ in range(2)]
        noises = [rng.normal(size=(15,32)).astype(np.float32) for _ in range(2)]
        model.calibrate_with_real_data([image_sets[0]])
        p = model.pipeline
        stream = torch.cuda.Stream()
        graphs = {}
        for arm in ([0,1] if args.compare_strict else [0]):
            if args.compare_strict:
                p._strict_visual_fusion = bool(arm)
            with torch.cuda.stream(stream):
                p.record_infer_graph(external_stream_int=stream.cuda_stream)
            stream.synchronize()
            graphs[arm] = (p._graph,p._decoder_only_graph,p._graph_stream)
        model._graph_torch_stream = stream
        def infer(arm,images,noise):
            if args.compare_strict:
                p._strict_visual_fusion = bool(arm)
            p._graph,p._decoder_only_graph,p._graph_stream = graphs[arm]
            out = model.infer(images,noise=noise)
            raw = model._noise_out.float().cpu().numpy().copy()
            assert np.isfinite(raw).all()
            return raw,np.asarray(out["actions"]).copy()
        raw_outputs,actions = [],[]
        for images in image_sets:
            for noise in noises:
                outputs = [infer(arm,images,noise) for arm in graphs]
                if args.compare_strict:
                    assert np.array_equal(outputs[0][0],outputs[1][0])
                    assert np.array_equal(outputs[0][1],outputs[1][1])
                raw_outputs.append(outputs[0][0])
                actions.append(outputs[0][1])
        raw_outputs,actions = np.stack(raw_outputs),np.stack(actions)
        if args.reference:
            with np.load(args.reference,allow_pickle=False) as ref:
                assert np.array_equal(raw_outputs,ref["raw"])
                assert np.array_equal(actions,ref["actions"])
        if args.save_reference:
            np.savez_compressed(args.save_reference,raw=raw_outputs,actions=actions)
        for i in range(args.warmup):
            for arm in graphs:
                infer(arm,image_sets[i%2],noises[i%2])
        times = {arm:[] for arm in graphs}
        for rep in range(args.reps):
            order = ([0,1,1,0] if rep%2==0 else [1,0,0,1]) if args.compare_strict else [0]
            for arm in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                infer(arm,image_sets[rep%2],noises[rep%2])
                torch.cuda.synchronize()
                times[arm].append((time.perf_counter()-start)*1000)
        result = {
            "extension":kernels.__file__,"correctness":"bit-equal on all compared outputs",
            "scope":"resident RGB/noise -> model.infer and CPU raw-action readback",
            "inputs":"two deterministic synthetic views, two observations, two noises",
            "settings":{"layers":27,"chunk":15,"steps":10,"pool":1,"cache_frames":1},
            "warmup":args.warmup,"p50_ms":{k:float(np.median(v)) for k,v in times.items()},
            "p95_ms":{k:float(np.percentile(v,95)) for k,v in times.items()},
            "samples":{k:len(v) for k,v in times.items()},
            "limitations":"synthetic-input smoke, heuristic BF16 algorithms; not closed-loop or cross-day validation"}
        print(json.dumps(result,indent=2))
    finally:
        Pi05Pipeline.autotune_gemms = original_tune

if __name__ == "__main__":
    main()
