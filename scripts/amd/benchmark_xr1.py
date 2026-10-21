"""XR-1 smoke/latency check; does not measure robot task success."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import flash_rt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--attention', choices=['sdpa', 'eager'], default='sdpa')
    parser.add_argument('--num-steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10, help='Measured calls after warmup')
    parser.add_argument('--warmup', type=int, default=1, help='Warmup calls, including first call')
    parser.add_argument('--native', action='store_true', help='Use FlashRT HIP DiT ops')
    parser.add_argument('--observations', help='Optional NPZ from validate_xr1_demo.py (5B only)')
    graph = parser.add_mutually_exclusive_group()
    graph.add_argument('--graph', action='store_true', help='Capture full-model HIP Graph')
    graph.add_argument('--compare-graph', action='store_true',
                       help='Paired eager/graph A/B on one loaded model, alternating call order')
    parser.add_argument('--output', default='results/xr1/smoke.json')
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 1:
        parser.error('--runs and --warmup must be positive')
    start = time.perf_counter()
    model = flash_rt.load_model(
        checkpoint=args.checkpoint, config='xr1', framework='torch',
        hardware='amd_rdna35', use_fp8=False, num_views=3,
        num_steps=args.num_steps, attention=args.attention, fuse=args.native,
        use_graph=args.graph, autotune=0)
    load_ms = (time.perf_counter() - start) * 1000
    rng = np.random.default_rng(42)
    images = [rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(3)]
    state = np.zeros(60 if model.pipeline.foundation else 8, np.float32)
    expected_shape = (30, 60) if model.pipeline.foundation else (10, 7)
    observations = [dict(images=images, state=state)]
    prompt = 'pick up the red block'
    if args.observations:
        if not model.pipeline.foundation:
            parser.error('--observations requires the 5B checkpoint with demo normalization')
        with np.load(args.observations, allow_pickle=False) as data:
            observations = [dict(images=[data[f'image_{i}_{j}'] for j in range(3)],
                                 state=data[f'state_{i}']) for i in range(len(data['frames']))]
        if not observations:
            parser.error('--observations contains no frames')
        prompt = 'Load washer.'

    def predict(observation):
        torch.cuda.synchronize()
        start = time.perf_counter()
        actions = model.predict(observation, prompt)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if actions.shape != expected_shape or not np.isfinite(actions).all():
            raise RuntimeError(f'Invalid XR-1 actions: {actions.shape}')
        return actions, elapsed

    modes = ['eager', 'graph'] if args.compare_graph else ['graph' if args.graph else 'eager']
    measurements, outputs, expected = {}, {}, {}
    for mode in modes:
        model.pipeline.use_graph = mode == 'graph'
        torch.cuda.reset_peak_memory_stats()
        first, first_ms = predict(observations[0])
        outputs[mode] = first
        expected[mode, 0] = first
        for _ in range(args.warmup - 1):
            actions, _ = predict(observations[0])
            np.testing.assert_array_equal(actions, first)
        measurements[mode] = dict(
            first_call_ms=first_ms, latency_ms=[], repeat_max_abs=0.0,
            allocated_after_warmup_bytes=torch.cuda.memory_allocated(),
            reserved_after_warmup_bytes=torch.cuda.memory_reserved(),
            initialization_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    if args.compare_graph:
        np.testing.assert_array_equal(outputs['eager'], outputs['graph'])
    torch.cuda.reset_peak_memory_stats()
    for index in range(args.runs):
        obs_index = index % len(observations)
        for mode in (modes if index % 2 == 0 else modes[::-1]):
            model.pipeline.use_graph = mode == 'graph'
            actions, elapsed = predict(observations[obs_index])
            measurements[mode]['latency_ms'].append(elapsed)
            np.testing.assert_array_equal(actions, expected.setdefault((mode, obs_index), actions))
        if args.compare_graph:
            np.testing.assert_array_equal(expected['eager', obs_index], expected['graph', obs_index])
    for mode, stats in measurements.items():
        stats['median_ms'] = float(np.median(stats['latency_ms']))
        stats['p95_ms'] = float(np.percentile(stats['latency_ms'], 95))
    result = dict(
        torch=torch.__version__, hip=torch.version.hip, native=args.native,
        arch=torch.cuda.get_device_properties(0).gcnArchName,
        checkpoint=str(Path(args.checkpoint).resolve()),
        attention=args.attention, num_steps=args.num_steps, shape=list(expected_shape),
        finite=True, load_ms=load_ms, graph=args.graph, compare_graph=args.compare_graph,
        runs_per_mode=args.runs, warmup_calls_per_mode=args.warmup,
        torch_threads=torch.get_num_threads(),
        timing_scope='synchronized public predict: preprocess/upload/full forward/decode/download',
        measurement_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        aotriton_experimental=os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '0'),
        synthetic_observations=not bool(args.observations),
        observation_count=len(observations),
        observation_indices=[i % len(observations) for i in range(args.runs)])
    if args.observations:
        result['observations'] = str(Path(args.observations).resolve())
        outputs.update({f'{mode}_{index}': value for (mode, index), value in expected.items()})
    if args.compare_graph:
        result['modes'] = measurements
        eager, graph = (measurements[name]['median_ms'] for name in modes)
        result['speedup'] = eager / graph
        result['latency_reduction_percent'] = (1 - graph / eager) * 100
        result['eager_graph_max_abs'] = 0.0
    else:
        result.update(measurements[modes[0]])
    if model.pipeline.graph_runner is not None:
        runner = model.pipeline.graph_runner
        result['graph_stats'] = dict(captures=runner.capture_count,
                                    predictions=runner.replay_count,
                                    last_capture_ms=runner.last_capture_ms)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    np.savez(output.with_suffix('.npz'), **outputs)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
