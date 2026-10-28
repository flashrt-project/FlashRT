"""Run one backend per process against an immutable shared observation manifest."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from .common import file_sha256, shared_noise, validate_manifest, write_json
from .dataset import load_observation
from .metrics import action_array, error_metrics


def prompt_key(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def hardware_snapshot():
    out = {}
    for name, args in {
        'gpu': ['nvidia-smi', '--query-gpu=name,driver_version,utilization.gpu,temperature.gpu,clocks.current.graphics,clocks.current.memory', '--format=csv,noheader'],
        'compute_processes': ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
    }.items():
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=10)
            out[name] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            out[name] = str(exc)
    return out


def _backend(args, capture):
    if args.backend == 'flashrt':
        from .backends.flashrt import FlashRTBackend
        return FlashRTBackend(args.checkpoint, str(capture), cpu_threads=args.cpu_threads)
    from .backends.jal import TorchBackend, TensorRTBackend
    kw = {'checkpoint': args.checkpoint, 'cpu_threads': args.cpu_threads}
    if args.source_dir:
        kw['source_dir'] = args.source_dir
    if args.backend == 'trt':
        return TensorRTBackend(engine_dir=args.engine_dir, **kw)
    return TorchBackend(**kw)


def capture(args):
    from .backends.flashrt import capture_calibration
    manifest_path = Path(args.manifest).resolve()
    manifest = validate_manifest(manifest_path)
    calibration = {}
    for record in manifest['records']:
        if record['split'] == 'calibration':
            calibration.setdefault(record['prompt'], record)
    prompts = set(r['prompt'] for r in manifest['records'] if r['split'] == 'evaluation')
    missing = prompts - set(calibration)
    if missing:
        raise ValueError(f'No separate calibration example for prompts: {sorted(missing)}')
    directory = Path(args.capture_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for prompt in sorted(prompts):
        record = calibration[prompt]
        out = directory / (prompt_key(prompt) + '.pt')
        if out.exists():
            sidecar = out.with_suffix('.json')
            metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
            expected = {'manifest_sha256': file_sha256(manifest_path), 'calibration_record_id': record['id'], 'calibration_observation_sha256': file_sha256(manifest_path.parent / record['observation']), 'capture_sha256': file_sha256(out), 'prompt': prompt}
            if not all(metadata.get(key) == value for key, value in expected.items()):
                raise ValueError(f'Existing capture cannot be verified: {out}')
            print(json.dumps({'reused_verified_capture': str(out)}), flush=True)
            continue
        start = time.perf_counter()
        capture_calibration(args.checkpoint, load_observation(manifest_path.parent / record['observation']), str(out), cpu_threads=args.cpu_threads)
        write_json(out.with_suffix('.json'), {'manifest_sha256': file_sha256(manifest_path), 'calibration_record_id': record['id'], 'calibration_observation_sha256': file_sha256(manifest_path.parent / record['observation']), 'capture_sha256': file_sha256(out), 'prompt': prompt, 'setup_seconds': time.perf_counter() - start})
        print(json.dumps({'captured': str(out), 'record_id': record['id']}), flush=True)


def run(args):
    if args.phase in ('timing', 'both') and not args.exclusive_gpu:
        raise ValueError('Timing requires --exclusive-gpu after checking/stopping competing GPU jobs and fixing clocks. Accuracy-only is allowed while shared.')
    manifest_path = Path(args.manifest).resolve()
    manifest = validate_manifest(manifest_path)
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    records = [r for r in manifest['records'] if r['split'] == 'evaluation']
    groups = defaultdict(list)
    for record in records:
        groups[record['prompt']].append(record)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot = hardware_snapshot()
    report = {'format_version': 1, 'backend': args.backend, 'phase': args.phase,
              'manifest_sha256': file_sha256(manifest_path), 'protocol_sha256': file_sha256(protocol_path),
              'checkpoint_config_sha256': file_sha256(Path(args.checkpoint) / 'config.json'),
              'processor_config_sha256': file_sha256(Path(args.checkpoint) / 'processor_config.json'),
              'weight_manifest_sha256': file_sha256(args.weight_manifest),
              'input_contract': protocol['input'], 'noise_seed': args.seed,
              'warmup': args.warmup, 'iters_per_prompt': args.iters, 'cpu_threads': args.cpu_threads,
              'exclusive_gpu_asserted': args.exclusive_gpu, 'hardware_before': snapshot,
              'closed_loop_task_success_measured': False, 'accuracy_acceptance': 'not_claimed; report raw errors',
              'groups': [], 'records': [], 'timing_samples': []}
    write_json(out / 'run.json', report)
    for prompt in sorted(groups):
        current = groups[prompt]
        if args.max_records:
            current = current[:args.max_records]
        prepared = [(r, load_observation(manifest_path.parent / r['observation']), shared_noise(r['id'], args.seed)) for r in current]
        capture_path = Path(args.capture_dir) / (prompt_key(prompt) + '.pt') if args.capture_dir else None
        if args.backend == 'flashrt':
            meta = json.loads(capture_path.with_suffix('.json').read_text())
            if meta['manifest_sha256'] != report['manifest_sha256'] or meta['prompt'] != prompt or meta['capture_sha256'] != file_sha256(capture_path):
                raise ValueError('Calibration capture is not bound to this manifest/prompt')
        start = time.perf_counter()
        backend = _backend(args, capture_path)
        try:
            backend.synchronize()
            if tuple(backend.noise_shape) != tuple(protocol['input']['noise_shape']):
                raise ValueError(f'Noise shape differs from protocol: {backend.noise_shape}')
            group_report = {'prompt': prompt, 'constructor_seconds': time.perf_counter() - start, 'metadata': backend.metadata}
            report['groups'].append(group_report)
            if args.phase in ('accuracy', 'both'):
                for record, obs, noise in prepared:
                    physical = action_array(backend.get_action(obs, noise))
                    backend.synchronize()
                    if physical.shape != (16, 7):
                        raise ValueError(f'Physical action shape differs from protocol: {physical.shape}')
                    with np.load(manifest_path.parent / record['ground_truth'], allow_pickle=False) as gt:
                        expected = gt['actions']
                    predicted = physical[:len(expected)]
                    key = hashlib.sha256(record['id'].encode()).hexdigest()[:24]
                    target = out / (key + '.npz')
                    np.savez_compressed(target, actions=physical, executed_actions=predicted, ground_truth=expected, noise=noise)
                    report['records'].append({'id': record['id'], 'trajectory_id': record['trajectory_id'], 'frame_index': record['frame_index'], 'prompt': prompt, 'file': target.name, 'observation_sha256': file_sha256(manifest_path.parent / record['observation']), 'noise_sha256': hashlib.sha256(noise.tobytes()).hexdigest(), 'processed_inputs': backend.input_fingerprints(obs), 'ground_truth_metrics': error_metrics(predicted, expected)})
                    write_json(out / 'run.json', report)
                    print(json.dumps({'backend': args.backend, 'record': record['id'], 'accuracy_complete': True}), flush=True)
            if args.phase == 'timing':
                for record, obs, noise in prepared:
                    report['records'].append({'id': record['id'], 'observation_sha256': file_sha256(manifest_path.parent / record['observation']), 'noise_sha256': hashlib.sha256(noise.tobytes()).hexdigest(), 'processed_inputs': backend.input_fingerprints(obs)})
            if args.phase in ('timing', 'both'):
                for index in range(args.warmup):
                    _, obs, noise = prepared[index % len(prepared)]
                    backend.get_action(obs, noise)
                backend.synchronize()
                for index in range(args.iters):
                    record, obs, noise = prepared[index % len(prepared)]
                    backend.synchronize()
                    start = time.perf_counter()
                    actions = backend.get_action(obs, noise)
                    backend.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    action_array(actions)  # Shape/value validation outside timing for all backends.
                    report['timing_samples'].append({'prompt': prompt, 'record_id': record['id'], 'iteration': index, 'full_call_ms': elapsed})
                values = [x['full_call_ms'] for x in report['timing_samples'] if x['prompt'] == prompt]
                group_report['latency_ms'] = {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)), 'min': float(np.min(values)), 'max': float(np.max(values))}
        finally:
            backend.close()
            write_json(out / 'run.json', report)
    report['hardware_after'] = hardware_snapshot()
    report['completed'] = True
    write_json(out / 'run.json', report)
    print(json.dumps({'completed': str(out / 'run.json')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('capture', 'run'):
        q = sub.add_parser(name)
        q.add_argument('--manifest', required=True)
        q.add_argument('--checkpoint', required=True)
        q.add_argument('--capture-dir', required=name == 'capture')
        q.add_argument('--cpu-threads', type=int, default=2)
        if name == 'run':
            q.add_argument('--backend', choices=['torch', 'trt', 'flashrt'], required=True)
            q.add_argument('--engine-dir')
            q.add_argument('--source-dir')
            q.add_argument('--protocol', default=str(Path(__file__).resolve().parents[1] / 'configs/protocol.json'))
            q.add_argument('--weight-manifest', default=str(Path(__file__).resolve().parents[1] / 'artifacts/weight-hashes.json'))
            q.add_argument('--phase', choices=['accuracy', 'timing', 'both'], default='accuracy')
            q.add_argument('--output', required=True)
            q.add_argument('--warmup', type=int, default=20)
            q.add_argument('--iters', type=int, default=100)
            q.add_argument('--seed', type=int, default=20261026)
            q.add_argument('--max-records', type=int)
            q.add_argument('--exclusive-gpu', action='store_true')
    args = p.parse_args()
    if args.cpu_threads < 1:
        p.error('cpu-threads must be positive')
    if args.command == 'capture':
        capture(args)
    else:
        if args.backend == 'flashrt' and not args.capture_dir:
            p.error('FlashRT needs --capture-dir')
        if args.backend == 'trt' and not args.engine_dir:
            p.error('TensorRT needs --engine-dir')
        if args.warmup < 0 or args.iters < 1 or (args.max_records is not None and args.max_records < 1):
            p.error('Invalid iteration counts')
        run(args)

if __name__ == '__main__':
    main()
