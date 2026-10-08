"""Long-horizon A/B for GR00T N1.7 on Orin: bf16 cross-KV arm vs fp32 arm vs HF eager.

Why this exists separately from ``test_orin_groot_n17_precision.py``
--------------------------------------------------------------------
The pytest suite gates a handful of frames. That is enough to prove an arm
*clears* a threshold, not enough to prove it holds over a task: the bf16
tensor-core cross-KV projection (docs/groot_n17_orin_sm87.md §6.19) moves the
decoded action by ~1 bf16 ULP at the projection, amplified by 4 Euler steps x
32 DiT blocks. A per-frame error that is flat is a rounding difference; one
that *grows with distance from the prompt* is a state bug (a stale slot, a
refresh that stopped refreshing). Only a long continuous run separates them.

So this drives the deployment mode end to end over a whole episode:

  ``set_prompt`` once, then ``infer(aux=...)`` for every subsequent frame,
  which is the path that refreshes ``_backbone_features`` and re-projects the
  16 cross-attention K/V pairs *per observation* — the exact code the bf16 arm
  changed. Both arms run on ONE frontend with ONE pinned ``initial_noise`` per
  frame, so the only variable is the projection.

Data is real (AGENTS.md §3.7): frames are decoded from the simulation-collected
LeRobot set the user supplied (``green_to_blue_block_sim``, 50 episodes /
31166 frames / 30 fps), never synthesized.

Streaming, not fixture-based: a 593-frame episode would be ~6.8 GB of aux
bundles on disk. This keeps one frame's capture in memory, compares, and drops
it — HF eager, both FlashRT arms and the comparison all live in one process.

Usage:
  PYTHONPATH=/mnt/Isaac-GR00T:/mnt/FlashRT \
    /mnt/venvs/groot_n17/bin/python \
    tests/_helpers/groot_orin/long_horizon_ab.py \
    --start 0 --count 593 --stride 1 --out /tmp/n17_long_horizon.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
HELPER = Path(__file__).resolve().parent
sys.path.insert(0, str(HELPER))

# Reuse the hook installers rather than re-deriving them: they encode the
# transformers-4.57 call-signature bindings and the "refuse to write a partial
# aux" checks, and a second copy would drift (AGENTS.md red line #2).
import capture_aux  # noqa: E402

#: This machine's paths, used only as defaults -- override with --ckpt /
#: --dataset. The run is not reproducible for anyone else without that, and a
#: hardcoded absolute path in a committed reproducer is how a script that was
#: verified once becomes a script nobody can run again.
DEFAULT_CKPT = "/mnt/GR00T/so101_sim_rynnbot/checkpoint-89-1.000"
DEFAULT_DATASET = "/mnt/groot_realdata/green_to_blue_block_sim"
SEED = 0
RAD2DEG = 180.0 / 3.141592653589793
DEVFREQ = Path("/sys/class/devfreq/17000000.gpu")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_CKPT,
                   help="GR00T N1.7 checkpoint directory")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="video-backed LeRobot root to stream frames from")
    p.add_argument("--start", type=int, default=0,
                   help="global frame index of the prompt (= first observation)")
    p.add_argument("--count", type=int, default=593,
                   help="how many frames to observe, prompt frame included")
    p.add_argument("--stride", type=int, default=1,
                   help="frame step; 1 = every frame of the episode")
    p.add_argument("--out", default="/tmp/n17_long_horizon.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-clock-gate", action="store_true",
                   help="skip the DVFS lock check (only for a smoke run; a "
                        "long-horizon number taken unlocked is not reportable "
                        "-- benchmarking-hygiene, and it invented a -1.7 ms "
                        "win on this exact board once)")
    return p.parse_args()


def check_clocks():
    cur = int((DEVFREQ / "cur_freq").read_text())
    lo = int((DEVFREQ / "min_freq").read_text())
    hi = int((DEVFREQ / "max_freq").read_text())
    if not (cur == lo == hi):
        raise SystemExit(
            f"GPU clocks are not locked ({cur}/{lo}/{hi} Hz). Lock them "
            "host-side first; an unlocked long run mixes precision drift with "
            "DVFS noise. --no-clock-gate overrides for smoke runs only.")
    return cur


def cos(a, b):
    a = a.double().flatten()
    b = b.double().flatten().to(a.device)
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def main():
    args = parse_args()
    cur = None if args.no_clock_gate else check_clocks()
    print(f"[clocks] {'unlocked (SMOKE ONLY)' if cur is None else f'locked at {cur/1e6:.1f} MHz'}")
    print(f"[cfg] ckpt={args.ckpt}")
    print(f"[cfg] dataset={args.dataset}")
    frames = list(range(args.start, args.start + args.count * args.stride,
                        args.stride))
    print(f"[cfg] prompt frame={frames[0]}  observations={len(frames)}  "
          f"stride={args.stride}  last={frames[-1]}")

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy import Gr00tPolicy

    from flash_rt.datasets.lerobot_video import LeRobotVideoDataset
    from flash_rt.frontends.torch.groot_n17_orin import (
        GrootN17TorchFrontendOrin,
    )

    t0 = time.perf_counter()
    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                         model_path=args.ckpt, device=args.device, strict=True)
    print(f"[load] HF {type(policy.model).__name__} in "
          f"{time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter()
    fe = GrootN17TorchFrontendOrin(args.ckpt, embodiment_tag="new_embodiment",
                                   device=args.device)
    print(f"[load] FlashRT {type(fe).__name__} in {time.perf_counter()-t0:.1f}s")

    CLS = type(fe)
    bf16_impl = CLS._project_dit_cross_kv
    fp32_impl = CLS._project_dit_cross_kv_fp32

    def run_arm(impl, aux, st, state_dict):
        """One observation through one projection arm."""
        CLS._project_dit_cross_kv = impl
        try:
            cap = {}
            out = fe.infer(fe.normalize_state(state_dict), aux=aux,
                           initial_noise=aux["initial_noise"],
                           use_dit_graph=True, capture=cap).cpu()
            dec = fe.denormalize_action(out, state_dict)
            K = torch.cat([t.flatten() for t in fe._project_dit_cross_kv()[0]])
            return out, dec, cap, K
        finally:
            CLS._project_dit_cross_kv = bf16_impl

    prompted = False
    recs = []
    chan_names = None
    t_hf = t_frt = 0.0

    with LeRobotVideoDataset(args.dataset) as ds, \
            open(args.out, "w") as sink:
        for i, fi in enumerate(frames):
            cap: dict = {}
            restore = capture_aux.install_hooks_n17(policy, cap)
            # N1.7's checkpoint expects RADIANS while the dataset stores
            # DEGREES (capture_aux/gen_reference's STATE_UNIT evidence); both
            # HF and FlashRT get the identical converted state, so the
            # implementation gate is unaffected either way.
            obs, raw = capture_aux.build_obs(ds, fi, state_to_radians=True)
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            t = time.perf_counter()
            with torch.inference_mode():
                action_dict, _info = policy.get_action(obs)
            torch.cuda.synchronize()
            t_hf += time.perf_counter() - t
            restore()

            missing = [k for k in capture_aux.REQUIRED_N17 if k not in cap]
            if missing:
                raise SystemExit(f"frame {fi}: hooks missed {missing}")
            hf_dec = {k: torch.as_tensor(v).float()
                      for k, v in action_dict.items()}

            st = torch.as_tensor(obs["state"]["state"]).reshape(1, 1, -1)
            state_dict = {"state.state": st}
            if not prompted:
                fe.set_prompt(aux=cap, prompt=raw["task"])
                prompted = True
                print(f"[prompt] frame {fi} task={raw['task']!r}")
                print(f"[prompt] decoded modalities: "
                      f"{ {k: tuple(v.shape) for k, v in hf_dec.items()} }")
                chan_names = list(hf_dec.keys())

            t = time.perf_counter()
            n32, d32, c32, K32 = run_arm(fp32_impl, cap, st, state_dict)
            nbf, dbf, cbf, Kbf = run_arm(bf16_impl, cap, st, state_dict)
            torch.cuda.synchronize()
            t_frt += time.perf_counter() - t

            rec = {"frame": fi, "episode": int(raw["episode_index"]),
                   "frame_in_episode": int(raw["frame_index"]),
                   "dist": i, "t1_cos": cos(Kbf, K32),
                   "t1_max": float((Kbf.double() - K32.double()).abs().max()),
                   "norm_cos_bf": cos(nbf, cap["final_actions_norm"]),
                   "norm_cos_32": cos(n32, cap["final_actions_norm"])}

            def vcos(cap_):
                ref = cap["velocity_per_step"]
                return min(cos(v, ref[j][:, -v.shape[1]:])
                           for j, v in enumerate(cap_["velocity_per_step"]))
            rec["t2_bf"], rec["t2_32"] = vcos(cbf), vcos(c32)

            for mod in hf_dec:
                want = hf_dec[mod]
                got_bf, got_32 = dbf[mod], d32[mod]
                sig = float(want.double().abs().max()) * RAD2DEG
                e_bf = (got_bf.double() - want.double()).abs() * RAD2DEG
                e_32 = (got_32.double() - want.double()).abs() * RAD2DEG
                e_ab = (got_bf.double() - got_32.double()).abs() * RAD2DEG
                rec[f"sig_{mod}"] = sig
                rec[f"cos_bf_{mod}"] = cos(got_bf, want)
                rec[f"cos_32_{mod}"] = cos(got_32, want)
                rec[f"max_bf_{mod}"] = float(e_bf.max())
                rec[f"max_32_{mod}"] = float(e_32.max())
                rec[f"mae_bf_{mod}"] = float(e_bf.mean())
                rec[f"mae_32_{mod}"] = float(e_32.mean())
                rec[f"max_ab_{mod}"] = float(e_ab.max())
                # Per-channel worst, so one bad channel cannot hide behind a
                # mean over the whole (horizon, dim) block.
                rec[f"perchan_bf_{mod}"] = [float(x) for x in e_bf[0].amax(0)]
                rec[f"perchan_32_{mod}"] = [float(x) for x in e_32[0].amax(0)]

            recs.append(rec)
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            if i % 25 == 0 or i == len(frames) - 1:
                mods = chan_names
                print(f"  [{i:>4}/{len(frames)}] f={fi} d={rec['dist']} | "
                      + " | ".join(
                          f"{m}: bf {rec[f'max_bf_{m}']:.4f}° "
                          f"32 {rec[f'max_32_{m}']:.4f}° "
                          f"cos {rec[f'cos_bf_{m}']:.7f}"
                          for m in mods)
                      + f" | t1 {rec['t1_cos']:.8f} t2 {rec['t2_bf']:.6f}")

    print(f"\n[time] HF eager {t_hf/len(frames)*1e3:.1f} ms/frame  "
          f"FlashRT both arms {t_frt/len(frames)*1e3:.1f} ms/frame")
    print(f"[out] {args.out} ({len(recs)} records)")

    summarize(recs, chan_names)


def summarize(recs, mods):
    n = len(recs)
    d = np.array([r["dist"] for r in recs], dtype=np.float64)

    def col(k):
        return np.array([r[k] for r in recs], dtype=np.float64)

    print("\n" + "=" * 78)
    print(f"LONG-HORIZON SUMMARY  ({n} observations, dist 0..{int(d.max())})")
    print("=" * 78)

    print("\n-- tier 1: stored cross-K, bf16 arm vs fp32 arm --")
    t1 = col("t1_cos")
    print(f"   cos   min {t1.min():.9f}  median {np.median(t1):.9f}")
    print(f"   max|d| worst {col('t1_max').max():.3e}")

    print("\n-- tier 2: velocity_per_step vs HF eager --")
    for arm in ("bf", "32"):
        v = col(f"t2_{arm}")
        print(f"   {arm:>2} arm  cos min {v.min():.7f}  median "
              f"{np.median(v):.7f}")

    print("\n-- tier 3: decoded action vs HF eager, degrees --")
    hdr = f"   {'modality':<20}{'arm':>4}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}{'signal':>9}"
    print(hdr)
    for m in mods:
        sig = col(f"sig_{m}").max()
        for arm, lab in (("bf", "bf16"), ("32", "fp32")):
            e = col(f"max_{arm}_{m}")
            print(f"   {m:<20}{lab:>4}{e.mean():>9.4f}{np.median(e):>9.4f}"
                  f"{np.percentile(e, 95):>9.4f}{e.max():>9.4f}{sig:>9.2f}")
        print(f"   {'':<20}{'A-B':>4}"
              f"{col(f'max_ab_{m}').mean():>9.4f}"
              f"{np.median(col(f'max_ab_{m}')):>9.4f}"
              f"{np.percentile(col(f'max_ab_{m}'), 95):>9.4f}"
              f"{col(f'max_ab_{m}').max():>9.4f}")
        for arm, lab in (("bf", "bf16"), ("32", "fp32")):
            c = col(f"cos_{arm}_{m}")
            print(f"   {m:<20}{lab:>4} cos min {c.min():.9f} median "
                  f"{np.median(c):.9f}")

    print("\n-- drift vs distance from the prompt (bf16 arm) --")
    for m in mods:
        e = col(f"max_bf_{m}")
        # Least-squares slope in degrees per observed frame, plus a bucketed
        # view so a non-linear trend is visible rather than averaged away.
        slope, intercept = np.polyfit(d, e, 1)
        nb = min(8, n)
        edges = np.linspace(0, n, nb + 1).astype(int)
        buck = " ".join(f"{e[edges[i]:edges[i+1]].mean():.3f}"
                        for i in range(nb) if edges[i + 1] > edges[i])
        first = e[:max(1, n // 10)].mean()
        last = e[-max(1, n // 10):].mean()
        print(f"   {m}: slope {slope:+.3e} deg/frame   first-10% {first:.4f}"
              f"  last-10% {last:.4f}  ratio {last/first if first else float('nan'):.2f}x")
        print(f"      bucket means ({nb}): {buck}")

    print("\n-- normalized-space (pre-decode) cos vs HF --")
    for arm in ("bf", "32"):
        c = col(f"norm_cos_{arm}")
        print(f"   {arm:>2} arm  min {c.min():.9f}  median {np.median(c):.9f}")


if __name__ == "__main__":
    main()
