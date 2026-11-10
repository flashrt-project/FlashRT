"""Frozen FP32 reference and synchronized full-model π inference benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def inputs(fixture, tokenizer, horizon, index=0):
    import sentencepiece as spm

    data = np.load(fixture)
    images = (
        np.stack([data[f"img_{index}"], data[f"wrist_{index}"]])
        .transpose(0, 3, 1, 2)
        .astype(np.float32)
        / 127.5
        - 1
    )
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    tokens = np.asarray(
        [sp.bos_id()] + sp.Encode("pick up the cup") + [108], dtype=np.int32
    )
    state = np.zeros(32, dtype=np.float32)
    # Tensor-boundary fixture: normalized state, not raw robot coordinates.
    state[:8] = np.random.default_rng(100 + index).uniform(-1, 1, 8)
    noise = (
        np.random.default_rng(200 + index)
        .standard_normal((horizon, 32))
        .astype(np.float32)
    )
    return images, tokens, state, noise


def reference(checkpoint, images, tokens, state, noise, pi05, steps):
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from flash_rt.npu.models.pi05 import pipeline as ref

    torch.set_num_threads(32)
    with safe_open(
        Path(checkpoint) / "model.safetensors", framework="pt", device="cpu"
    ) as f:
        w = {k.removeprefix("model."): f.get_tensor(k).float() for k in f.keys()}
    images, tokens, state, noise = [
        torch.from_numpy(x) for x in (images, tokens, state, noise)
    ]
    with torch.inference_mode():
        if pi05:
            return ref.sample(images, tokens, noise, w, steps).numpy()
        cache = ref.encoder_pass(
            torch.cat([ref.vision_tower(images, w), ref.embed_language(tokens, w)]), w
        )
        plen = cache[0][0].shape[0]
        seq = noise.shape[0] + 1
        mask = torch.zeros(seq, plen + seq)
        mask[0, plen + 1 :] = torch.finfo(torch.float32).min
        pos = torch.arange(plen, plen + seq)
        inv = ref._rope_inv_freqs(256)
        x = noise
        for s in range(steps):
            te = ref.time_embedding(1 - s / steps, 1024)[None].expand(x.shape[0], -1)
            act = F.linear(x, w["action_in_proj.weight"], w["action_in_proj.bias"])
            act = F.silu(
                F.linear(
                    torch.cat([act, te], -1),
                    w["action_time_mlp_in.weight"],
                    w["action_time_mlp_in.bias"],
                )
            )
            act = F.linear(
                act, w["action_time_mlp_out.weight"], w["action_time_mlp_out.bias"]
            )
            z = torch.cat(
                [
                    F.linear(state[None], w["state_proj.weight"], w["state_proj.bias"]),
                    act,
                ]
            )
            for i in range(18):
                base = ref._DP + f".{i}"
                xn = ref.rms_norm(z, w[base + ".input_layernorm.weight"])
                q, k, v = [
                    F.linear(xn, w[base + ".self_attn." + p + "_proj.weight"])
                    for p in ("q", "k", "v")
                ]
                q, k = [ref.rope_half_split(t, pos, inv, 256) for t in (q, k)]
                k, v = torch.cat([cache[i][0], k]), torch.cat([cache[i][1], v])
                o = ref.attention(
                    q.reshape(seq, 8, 256).transpose(0, 1),
                    k[None].expand(8, -1, -1),
                    v[None].expand(8, -1, -1),
                    mask,
                )
                z = z + F.linear(o, w[base + ".self_attn.o_proj.weight"])
                xn = ref.rms_norm(z, w[base + ".post_attention_layernorm.weight"])
                h = F.gelu(
                    F.linear(xn, w[base + ".mlp.gate_proj.weight"]), approximate="tanh"
                ) * F.linear(xn, w[base + ".mlp.up_proj.weight"])
                z = z + F.linear(h, w[base + ".mlp.down_proj.weight"])
            z = ref.rms_norm(
                z, w["paligemma_with_expert.gemma_expert.model.norm.weight"]
            )[1:]
            x = (
                x
                - F.linear(z, w["action_out_proj.weight"], w["action_out_proj.bias"])
                / steps
            )
        return x.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--model", required=True, choices=["pi0", "pi05"])
    p.add_argument("--fixture", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--reference-only", action="store_true")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--fixture-index", type=int, default=0)
    args = p.parse_args()
    if args.runs < 2 or args.warmup < 1 or args.fixture_index < 0:
        p.error("Use at least two runs, one warmup, and a nonnegative fixture index")
    data = inputs(args.fixture, args.tokenizer, args.horizon, args.fixture_index)
    input_digest = hashlib.sha256(b"".join(x.tobytes() for x in data)).hexdigest()
    if args.reference_only:
        started = time.monotonic()
        raw = reference(args.checkpoint, *data, args.model == "pi05", args.steps)
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite reference")
        np.savez(
            args.reference,
            raw=raw,
            input_sha256=input_digest,
            steps=args.steps,
            model=args.model,
        )
        args.output.write_text(
            json.dumps(
                {
                    "reference_seconds": time.monotonic() - started,
                    "input_sha256": input_digest,
                    "shape": list(raw.shape),
                }
            )
            + "\n"
        )
        return
    import jax
    import flash_rt
    from flash_rt.neuron.pi import sample

    print("Devices:", jax.devices(), flush=True)
    before = time.monotonic()
    model = flash_rt.load_model(
        str(args.checkpoint),
        framework="jax",
        hardware="neuron",
        config=args.model,
        num_views=2,
        num_steps=args.steps,
        action_horizon=args.horizon,
        precision="bf16",
    )
    fe = model.pipeline
    load_seconds = time.monotonic() - before
    device_data = [jax.device_put(x, fe.device) for x in data]
    before = time.monotonic()
    out = sample(fe.weights, *device_data, fe.times, pi05=fe.pi05).block_until_ready()
    first_seconds = time.monotonic() - before
    print("First inference", first_seconds, "seconds", flush=True)
    gold = np.load(args.reference)
    if (
        str(gold["input_sha256"]) != input_digest
        or int(gold["steps"]) != args.steps
        or str(gold["model"]) != args.model
    ):
        raise ValueError("Reference fixture mismatch")
    actual = np.asarray(out, dtype=np.float64)
    expected = np.asarray(gold["raw"], dtype=np.float64)
    cosine = float(
        np.dot(actual.ravel(), expected.ravel())
        / (np.linalg.norm(actual) * np.linalg.norm(expected))
    )
    nrmse = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    max_abs = float(np.max(np.abs(actual - expected)))
    correct = bool(np.isfinite(actual).all() and cosine >= 0.999 and nrmse <= 0.05)
    result = {
        "model": args.model,
        "device": str(fe.device),
        "platform": fe.device.platform,
        "precision": "bf16",
        "views": 2,
        "horizon": args.horizon,
        "steps": args.steps,
        "prompt_tokens": len(data[1]),
        "fixture_index": args.fixture_index,
        "input_sha256": input_digest,
        "load_seconds": load_seconds,
        "first_inference_seconds": first_seconds,
        "correctness": {
            "passed": correct,
            "cosine": cosine,
            "relative_l2": nrmse,
            "max_abs": max_abs,
            "cosine_min": 0.999,
            "relative_l2_max": 0.05,
        },
        "status": "qualified" if correct else "numerical_failure",
    }
    np.save(args.output.with_suffix(".actions.npy"), actual)
    # Failed correctness never produces accepted timing measurements.
    if correct:
        for _ in range(args.warmup):
            sample(fe.weights, *device_data, fe.times, pi05=fe.pi05).block_until_ready()
        times = []
        for _ in range(args.runs):
            start = time.perf_counter_ns()
            sample(fe.weights, *device_data, fe.times, pi05=fe.pi05).block_until_ready()
            times.append((time.perf_counter_ns() - start) / 1e6)
        result["device_resident_latency_ms"] = {
            "samples": times,
            "p50": float(np.median(times)),
            "p90": float(np.percentile(times, 90)),
            "p99": float(np.percentile(times, 99)),
        }
        times = []
        for _ in range(args.runs):
            start = time.perf_counter_ns()
            np.asarray(fe.infer_normalized(*data))
            times.append((time.perf_counter_ns() - start) / 1e6)
        result["tensor_boundary_latency_ms"] = {
            "samples": times,
            "p50": float(np.median(times)),
            "p90": float(np.percentile(times, 90)),
            "p99": float(np.percentile(times, 99)),
        }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)
    if not correct:
        raise RuntimeError("Numerical correctness gate failed")


if __name__ == "__main__":
    main()
