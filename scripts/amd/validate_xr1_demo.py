"""Compare XR-1 reference/native actions on the upstream load-washer video.

Requires ffmpeg and the pinned json1 demo from download_xr1.sh demo. This
checks numerical agreement on real observations, not robot task success.
The checkpoint must use the matching load_washer normalization statistics.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

import flash_rt
from flash_rt.amd.models.xr1.pipeline_rdna35 import XR1PipelineRdna35


def read_observations(root, frames):
    data = json.loads((root / 'json/json1.json').read_text())
    cases = []
    for frame in frames:
        images = []
        for key in ('ego', 'wrist_left', 'wrist_right'):
            observation = data['observations'][key][0]
            video = root / 'videos' / Path(observation['path']).name
            encoded = subprocess.check_output([
                'ffmpeg', '-v', 'error', '-i', str(video), '-vf',
                f'select=eq(n\\,{observation["start"] + frame})',
                '-vsync', '0', '-frames:v', '1', '-f', 'image2pipe',
                '-vcodec', 'png', 'pipe:1'])
            image = Image.open(BytesIO(encoded)).convert('RGB')
            top, left, height, width = observation['crop_bbox']
            images.append(np.asarray(image.crop((left, top, left + width, top + height))))
        state = np.zeros(60, np.float32)
        for start, key in ((0, 'left_arm_joint'), (8, 'right_arm_joint')):
            values = data['proprios'][key][frame]
            state[start:start + len(values)] = values
        state[7] = data['proprios']['left_gripper_pos'][frame][0]
        state[15] = data['proprios']['right_gripper_pos'][frame][0]
        cases.append((images, state))
    return cases


def compare(reference, actual):
    if actual.shape != (30, 60) or not np.isfinite(actual).all():
        raise AssertionError('Expected finite (30,60) actions')
    cosine = float(np.dot(actual.ravel(), reference.ravel()) / (
        np.linalg.norm(actual) * np.linalg.norm(reference)))
    max_abs = float(np.max(np.abs(actual - reference)))
    return dict(cosine=cosine, max_abs=max_abs, passed=cosine > .999 and max_abs < .04)


@torch.inference_mode()
def upstream_actions(frontend, images, state, checkout):
    """Exercise the original runtime's messages/resize/state/decode utilities.

    Generation uses the loaded ROCm policy and SDPA in both cases. This checks
    the frontend contract; it does not compare CUDA/Liger against ROCm kernels.
    """
    sys.path.insert(0, str(checkout.resolve() / 'xr1'))
    try:
        from mibot.utils.io import (build_action_mask, compose_state,
                                   denormalize_action, resize_image)
        from mibot.server.runtime.client import Client
    finally:
        sys.path.pop(0)
    views = [resize_image(Image.fromarray(image)) for image in images]
    messages = Client._messages('Load washer.', *views)
    batch = frontend.processor.processor.apply_chat_template(
        [messages], tokenize=True, return_dict=True, return_tensors='pt',
        padding=True, images_kwargs={'do_resize': False})
    batch['state'] = torch.from_numpy(compose_state(
        state[7:8], state[:7], state[15:16], state[8:15]))[None]
    batch = {key: value.to('cuda') for key, value in batch.items()}
    stats = {key: value.to('cuda') for key, value in frontend.processor.stats.items()}
    q01, q99 = stats['q01'], stats['q99']
    valid = q99 > q01
    normalized = torch.zeros_like(batch['state'])
    normalized[..., valid[0]] = (
        2 * (batch['state'][..., valid[0]] - q01[..., valid[0]])
        / (q99[..., valid[0]] - q01[..., valid[0]] + 1e-6) - 1)
    batch['state'] = normalized.clamp(-1, 1)
    mask = torch.from_numpy(build_action_mask(30)).to('cuda')[None]
    batch['action_mask'] = mask
    batch['action'] = torch.zeros((1, 30, 60), device='cuda', dtype=torch.bfloat16)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        raw = frontend.model.generate(batch)
    decoded = denormalize_action(raw * mask, stats['mean'], stats['std']) * mask
    return decoded[0].float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='.cache/xr1/5b')
    parser.add_argument('--demo-dir', type=Path, default=Path('.cache/xr1/demo'))
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 160, 320])
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--upstream', type=Path, help='Optional pinned XR-1 source checkout')
    parser.add_argument('--output', type=Path, default=Path('results/xr1/demo_validation.json'))
    args = parser.parse_args()
    if args.runs < 1 or any(frame < 0 or frame >= 474 for frame in args.frames):
        parser.error('runs must be positive; json1 frames must be in [0,474)')
    cases = read_observations(args.demo_dir, args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep extracted observations for optional real-camera pytest regression.
    fixture = args.output.with_suffix('.npz')
    np.savez(fixture, frames=args.frames,
             **{f'image_{i}_{j}': image for i, (images, _) in enumerate(cases)
                for j, image in enumerate(images)},
             **{f'state_{i}': state for i, (_, state) in enumerate(cases)})
    model = flash_rt.load_model(args.checkpoint, config='xr1', num_views=3,
                               hardware='amd_rdna35', fuse=False, autotune=0)
    if not model.pipeline.foundation:
        parser.error('this demo requires the 5B model_states.pt checkpoint')
    expected = [model.predict(images, 'Load washer.', state) for images, state in cases]
    results = [dict(frame=frame, shape=list(reference.shape))
               for frame, reference in zip(args.frames, expected)]
    if args.upstream:
        for (images, state), reference, result in zip(cases, expected, results):
            upstream = upstream_actions(model.pipeline, images, state, args.upstream)
            np.testing.assert_array_equal(reference, upstream)
            result['upstream_max_abs'] = float(np.max(np.abs(reference - upstream)))
    for native in (False, True):
        if native:
            model.pipeline.pipeline = XR1PipelineRdna35(model.pipeline.model, native=True)
        for (images, state), reference, result in zip(cases, expected, results):
            first = model.predict(images, state=state)  # warmup / lazy tuning
            latencies = []
            for _ in range(args.runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                actual = model.predict(images, state=state)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)
                np.testing.assert_array_equal(actual, first)
            mode = 'native' if native else 'reference'
            result[mode] = dict(compare(reference, first), latency_ms=latencies,
                                median_ms=float(np.median(latencies)),
                                p95_ms=float(np.percentile(latencies, 95)))
            print(json.dumps(dict(frame=result['frame'], mode=mode, **result[mode])), flush=True)
    report = dict(source='XiaomiRobotics/xr1_post_train_demo/json1',
                  checkpoint=str(Path(args.checkpoint).resolve()),
                  embodiment=model.pipeline.embodiment_tag,
                  torch=torch.__version__, hip=torch.version.hip,
                  arch=torch.cuda.get_device_properties(0).gcnArchName,
                  task_success_measured=False, observations=str(fixture), results=results)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not all(result['native']['passed'] for result in results):
        raise SystemExit('Native action parity failed; see ' + str(args.output))


if __name__ == '__main__':
    main()
