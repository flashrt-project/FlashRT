#!/usr/bin/env python3
"""GR00T N1.5 action parity: FlashRT (BF16 / FP8) vs the unmodified upstream policy.

Observations come from LeRobot-format episodes (``meta/modality.json``,
``meta/tasks.jsonl``, ``data/chunk-000/episode_000000.parquet``,
``videos/chunk-000/<video key>/episode_000000.mp4``), e.g. the
``single_panda_gripper.*`` tasks of nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
for the RoboCasa ``panda_omron`` data config. ``--video-map`` maps the data
config's video keys to the dataset's (the default is panda_omron's). Frames
at 0, 1/3 and 2/3 of each episode are used, each with every seed in --seeds.

For every call the upstream ``Gr00tPolicy`` runs first; its normalized
inputs and the initial noise it draws (read by a forward pre-hook, nothing is
patched) are then given to FlashRT. Reported per tier: action cosine over all
dims and over the real (unpadded) dims, the max abs error on the real dims,
and whether the binary-normalized (discrete) outputs are equal.
``--noise-floor`` adds the upstream policy against itself with its attention
switched from flash-attn to SDPA (needs to replace torch.randn while the
head draws its noise, so it runs last).

    GROOT_N15_ROOT=<Isaac-GR00T> python scripts/groot_n15_parity.py \\
        --checkpoint <checkpoint> --data-config panda_omron \\
        --episodes <dataset>/single_panda_gripper.OpenDrawer ...
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

PANDA_OMRON_VIDEOS = ("robot0_agentview_left=left_view,robot0_agentview_right=right_view,"
                      "robot0_eye_in_hand=wrist_view")


def _frames(path: pathlib.Path, idxs: list[int]) -> list[np.ndarray]:
    import av
    want, out = set(idxs), {}
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = f.to_ndarray(format="rgb24")
            if len(out) == len(want):
                break
    return [out[i] for i in idxs]


def observations(episodes, policy, video_map):
    import pandas as pd
    lang = policy.modality_config["language"].modality_keys[0]
    vkeys = policy.modality_config["video"].modality_keys
    skeys = policy.modality_config["state"].modality_keys
    for ep in map(pathlib.Path, episodes):
        modality = json.load(open(ep / "meta/modality.json"))
        tasks = {}
        for line in open(ep / "meta/tasks.jsonl"):
            t = json.loads(line)
            tasks[t["task_index"]] = t["task"]
        df = pd.read_parquet(ep / "data/chunk-000/episode_000000.parquet")
        idxs = [0, len(df) // 3, 2 * len(df) // 3]
        views = {k: _frames(ep / f"videos/chunk-000/observation.images.{video_map[k]}/"
                                 "episode_000000.mp4", idxs) for k in vkeys}
        for j, fi in enumerate(idxs):
            row = df.iloc[fi]
            st = np.asarray(row["observation.state"], dtype=np.float64)
            obs = {k: views[k][j][None] for k in vkeys}
            for k in skeys:
                s = modality["state"][k.split(".", 1)[1]]
                obs[k] = st[s["start"]:s["end"]][None]
            obs[lang] = [tasks[int(row["annotation.human.action.task_description"])]]
            yield f"{ep.name}@{fi}", obs


def reference(policy, obs):
    seen = {}

    def grab_noise(module, args):   # returns None: the inputs pass through unchanged
        seen.setdefault("noise", args[0].detach().clone())

    hook = policy.model.action_head.action_encoder.register_forward_pre_hook(grab_noise)
    orig = policy._get_action_from_normalized_input

    def grab(normalized):
        seen["inputs"] = normalized
        seen["actions"] = orig(normalized)
        return seen["actions"]

    policy._get_action_from_normalized_input = grab
    try:
        out = policy.get_action(obs)
    finally:
        policy._get_action_from_normalized_input = orig
        hook.remove()
    return seen["inputs"], seen["noise"], seen["actions"].float().cpu(), out


def score(name, rows):
    cos = min(r["cos"] for r in rows)
    real = min(r["real_cos"] for r in rows)
    err = max(r["real_err"] for r in rows)
    disc = sum(r["discrete"] for r in rows)
    print(f"{name:28s} cos min {cos:.6f}  real-dim cos min {real:.6f}  "
          f"real-dim max|err| {err:.3e}  discrete equal {disc}/{len(rows)}")


def compare(out, ref, out_dict, ref_dict, real, binary):
    a, b = out.flatten(), ref.flatten()
    ar, br = out[..., :real].flatten(), ref[..., :real].flatten()
    return {"cos": F.cosine_similarity(a, b, dim=0).item(),
            "real_cos": F.cosine_similarity(ar, br, dim=0).item(),
            "real_err": (ar - br).abs().max().item(),
            "discrete": all(np.array_equal(out_dict[k], ref_dict[k]) for k in binary)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-config", default=os.environ.get("GROOT_N15_DATA_CONFIG"))
    ap.add_argument("--groot-root", default=os.environ.get("GROOT_N15_ROOT"))
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--video-map", default=PANDA_OMRON_VIDEOS,
                    help="data-config video key = dataset video key, comma-separated")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--precision", default="bf16,fp8")
    ap.add_argument("--noise-floor", action="store_true")
    args = ap.parse_args()
    if args.groot_root:
        sys.path.insert(0, args.groot_root)
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    from flash_rt.hardware import resolve_pipeline_class

    cfg = DATA_CONFIG_MAP[args.data_config]
    policy = Gr00tPolicy(model_path=args.checkpoint, embodiment_tag="new_embodiment",
                         modality_config=cfg.modality_config(),
                         modality_transform=cfg.transform(), denoising_steps=4)
    video_map = {"video." + k: v for k, v in (p.split("=") for p in args.video_map.split(","))}
    meta = json.load(open(os.path.join(args.checkpoint, "experiment_cfg",
                                       "metadata.json")))["new_embodiment"]
    real = sum(meta["modalities"]["action"][k.split(".", 1)[1]]["shape"][0]
               for k in policy.modality_config["action"].modality_keys)
    binary = [k for k, m in getattr(cfg, "action_normalization_modes", {}).items()
              if m == "binary"]
    calls = []
    for name, obs in observations(args.episodes, policy, video_map):
        for seed in (int(s) for s in args.seeds.split(",")):
            torch.manual_seed(seed)
            inputs, noise, ref, ref_dict = reference(policy, obs)
            calls.append((name, obs, inputs, noise, ref, ref_dict))
    print(f"{len(calls)} calls, {real} real action dims, discrete keys {binary}")
    frontend = resolve_pipeline_class("groot_n15", "torch", "rtx_sm89")
    for tier in args.precision.split(","):
        fe = frontend(args.checkpoint, num_views=len(video_map), data_config=args.data_config,
                      groot_root=args.groot_root, use_fp8=tier == "fp8")
        rows = []
        for _, _, inputs, noise, ref, ref_dict in calls:
            out = fe.predict_normalized(inputs, noise).cpu()
            rows.append(compare(out, ref, fe.io.unnormalize(out), ref_dict, real, binary))
        score(f"FlashRT {tier}", rows)
        del fe
        torch.cuda.empty_cache()
    if args.noise_floor:
        eagle = policy.model.backbone.eagle_model
        for module in (eagle.vision_model, eagle.language_model):
            for c in {id(m.config): m.config for m in module.modules()
                      if hasattr(m, "config")}.values():
                if hasattr(c, "_attn_implementation"):
                    c._attn_implementation = "sdpa"
        real_randn, rows = torch.randn, []
        for _, obs, _, noise, ref, ref_dict in calls:
            def fixed(*a, _n=noise.cuda(), **k):
                size = tuple(k.get("size", a[0] if a else ()))
                return _n.clone() if size == tuple(_n.shape) else real_randn(*a, **k)
            torch.randn = fixed
            try:
                _, _, out, out_dict = reference(policy, obs)
            finally:
                torch.randn = real_randn
            rows.append(compare(out, ref, out_dict, ref_dict, real, binary))
        score("upstream SDPA vs flash-attn", rows)


if __name__ == "__main__":
    main()
