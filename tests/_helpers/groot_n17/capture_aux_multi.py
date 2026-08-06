"""Capture N aux dicts for multi-frame FP8 calibration.

Each ``(traj, step)`` pair runs through the official PyTorch policy under
the same hook set used by ``capture_llm_aux.py`` (LLM forward, ViT block 0
pre-hook, rotary_emb, action-head DiT/decoder hooks, torch.randn capture).
The N captured aux dicts are stored as a python list in a single ``.pt``
fixture so a downstream test loads them with one ``torch.load`` call:

    aux_list = torch.load("tests/fixtures/gr00t_n17_aux_list_n8.pt",
                          weights_only=False)
    pipe.calibrate(aux_list, percentile=99.9)

Each entry has the same schema as the single-aux fixture written by
``capture_llm_aux.py`` (``input_ids``, ``visual_pos_masks``, ``rope_cos``,
``rope_sin``, ``llm_input_embeds``, ``pixel_features``, ``grid_thw``,
``initial_noise``, ...).

Stratification: the default ``--pairs`` argument (8 samples) walks two
trajectories at four distinct step indices each, which gives reasonable
coverage of pose / view / language variation in the demo dataset
without spending too long on policy.get_action (~30 s × N).

Usage::

    python tests/_helpers/groot_n17/capture_aux_multi.py \\
        --tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \\
        --pairs "1:0,1:50,1:100,1:150,2:0,2:50,2:100,2:150" \\
        --out tests/fixtures/gr00t_n17_aux_list_n8.pt
"""
from __future__ import annotations

import argparse
import functools
import glob
import os
import warnings
from pathlib import Path
from typing import Any

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

import numpy as np
import torch


def _autodetect_ckpt() -> str:
    cands = sorted(glob.glob(
        "/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/*"))
    if not cands:
        raise FileNotFoundError("no N1.7 snapshot under HF cache")
    return cands[0]


def _install_hooks(policy, captured: dict) -> list:
    """Install the same hook set as ``capture_llm_aux.py`` and return
    handles plus the original forwards so the caller can reset them
    between samples."""
    model = policy.model

    lm = model.backbone.model.model.language_model
    visual = model.backbone.model.model.visual
    block0 = visual.blocks[0]
    rot = lm.rotary_emb
    ah = model.action_head
    dit = ah.model
    ad = ah.action_decoder

    state = {
        "lm_orig": lm.forward,
        "rot_orig": rot.forward,
        "ah_gawf_orig": ah.get_action_with_features,
        "ad_orig": ad.forward,
        "dit_orig": dit.forward,
        "visual_orig": visual.forward,
        "handles": [],
    }

    def lm_hook(self, *fargs, **fkwargs):
        if "inputs_embeds" in fkwargs:
            captured["llm_input_embeds"] = (
                fkwargs["inputs_embeds"].detach().to(torch.float32).cpu())
            captured["inputs_embeds_shape"] = tuple(fkwargs["inputs_embeds"].shape)
        if "input_ids" in fkwargs and fkwargs["input_ids"] is not None:
            captured["input_ids"] = fkwargs["input_ids"].detach().cpu()
        if "visual_pos_masks" in fkwargs:
            vpm = fkwargs["visual_pos_masks"]
            captured["visual_pos_masks"] = (
                vpm.detach().cpu() if vpm is not None else None)
        if "position_ids" in fkwargs:
            pid = fkwargs["position_ids"]
            captured["position_ids"] = (
                pid.detach().cpu() if pid is not None else None)
        return state["lm_orig"](*fargs, **fkwargs)

    def gawf_hook(self, *args, **kwargs):
        orig_randn = torch.randn

        def patched_randn(*ra, **rkw):
            t = orig_randn(*ra, **rkw)
            if "initial_noise" not in captured:
                captured["initial_noise"] = (
                    t.detach().to(torch.float32).cpu().clone())
            return t
        torch.randn = patched_randn
        try:
            return state["ah_gawf_orig"](*args, **kwargs)
        finally:
            torch.randn = orig_randn

    def ad_hook(self, *args, **kwargs):
        out = state["ad_orig"](*args, **kwargs)
        captured["last_action_decoder_out"] = (
            out.detach().to(torch.float32).cpu())
        return out

    captured["dit_step_input"] = []
    captured["dit_step_output"] = []
    captured["dit_step_temb"] = []

    def dit_hook(self, hidden_states, *fa, **fk):
        captured["dit_step_input"].append(
            hidden_states.detach().to(torch.float32).cpu())
        if "timestep" in fk:
            captured["dit_step_temb"].append(fk["timestep"].detach().cpu().clone())
        out = state["dit_orig"](hidden_states, *fa, **fk)
        t0 = out[0] if isinstance(out, tuple) else out
        captured["dit_step_output"].append(t0.detach().to(torch.float32).cpu())
        return out

    def block0_pre_hook(module, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        captured["pixel_features"] = h.detach().to(torch.float32).cpu()
        return None

    def visual_hook(self, hidden_states, grid_thw, **kw):
        captured["grid_thw"] = grid_thw.detach().cpu()
        return state["visual_orig"](hidden_states, grid_thw, **kw)

    def rot_hook(self, x, position_ids):
        cos, sin = state["rot_orig"](x, position_ids)
        captured["rope_cos"] = cos.detach().cpu()
        captured["rope_sin"] = sin.detach().cpu()
        return cos, sin

    lm.forward = functools.partial(lm_hook, lm)
    rot.forward = functools.partial(rot_hook, rot)
    ah.get_action_with_features = functools.partial(gawf_hook, ah)
    ad.forward = functools.partial(ad_hook, ad)
    dit.forward = functools.partial(dit_hook, dit)
    visual.forward = functools.partial(visual_hook, visual)
    state["handles"].append(
        block0.register_forward_pre_hook(block0_pre_hook, with_kwargs=True))

    state["_objs"] = (lm, visual, block0, rot, ah, dit, ad)
    return state


def _restore_hooks(state: dict) -> None:
    lm, visual, block0, rot, ah, dit, ad = state["_objs"]
    lm.forward = state["lm_orig"]
    rot.forward = state["rot_orig"]
    ah.get_action_with_features = state["ah_gawf_orig"]
    ad.forward = state["ad_orig"]
    dit.forward = state["dit_orig"]
    visual.forward = state["visual_orig"]
    for h in state["handles"]:
        h.remove()


def _build_parsed(loader, policy, traj_idx: int, step_idx: int) -> dict:
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.eval.open_loop_eval import parse_observation_gr00t

    traj = loader[traj_idx]
    data_point = extract_step_data(
        traj, step_idx, policy.modality_configs, policy.embodiment_tag)
    obs: dict[str, Any] = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for lang_key in loader.modality_configs["language"].modality_keys:
        obs[lang_key] = data_point.text
    return parse_observation_gr00t(obs, loader.modality_configs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
    ap.add_argument("--ckpt", default=None,
                    help="N1.7 snapshot dir (auto-detected from HF cache if None)")
    ap.add_argument("--dataset",
                    default="/gr00t/Isaac-GR00T/demo_data/droid_sample")
    ap.add_argument("--pairs",
                    default="1:0,1:50,1:100,1:150,2:0,2:50,2:100,2:150",
                    help="comma-separated traj:step pairs")
    ap.add_argument("--seed", type=int, default=0,
                    help="deterministic noise seed (re-applied per sample)")
    ap.add_argument("--views", type=int, default=None,
                    help="keep only the first N camera views of the "
                         "embodiment's video modality (default: all). Use "
                         "1 to capture a single-camera fixture.")
    ap.add_argument("--out", required=True,
                    help="output .pt path (list of aux dicts)")
    args = ap.parse_args()

    pairs = []
    for tok in args.pairs.split(","):
        t, s = tok.strip().split(":")
        pairs.append((int(t), int(s)))
    if not pairs:
        raise ValueError("--pairs is empty")

    ckpt = args.ckpt or _autodetect_ckpt()
    print(f"ckpt:    {ckpt}")
    print(f"dataset: {args.dataset}")
    print(f"tag:     {args.tag}")
    print(f"pairs:   {pairs}")

    import gr00t.model  # noqa: F401
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    print("loading policy ...", flush=True)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.tag),
        model_path=ckpt, device="cuda:0",
    )
    if args.views is not None:
        # Restrict the video modality to the first N camera keys. Both the
        # policy transforms and the episode loader must agree, so trim the
        # shared ModalityConfig before the loader is built.
        vid = policy.modality_configs["video"]
        if args.views < 1 or args.views > len(vid.modality_keys):
            raise ValueError(
                f"--views must be in [1, {len(vid.modality_keys)}] for tag "
                f"{args.tag}")
        vid.modality_keys = list(vid.modality_keys[:args.views])
        print(f"views:   {vid.modality_keys}")

    print("loading dataset ...", flush=True)
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset,
        modality_configs=policy.modality_configs,
        video_backend="ffmpeg",
    )

    aux_list: list[dict] = []
    for i, (traj_idx, step_idx) in enumerate(pairs):
        print(f"\n── sample {i + 1}/{len(pairs)}: traj={traj_idx} step={step_idx} ──",
              flush=True)
        parsed = _build_parsed(loader, policy, traj_idx, step_idx)

        captured: dict = {}
        state = _install_hooks(policy, captured)
        try:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            with torch.inference_mode():
                policy.get_action(parsed)
        finally:
            _restore_hooks(state)

        # Project to canonical aux schema (mirror capture_llm_aux.py output).
        keep = (
            "input_ids", "visual_pos_masks", "position_ids",
            "rope_cos", "rope_sin", "inputs_embeds_shape",
            "llm_input_embeds", "pixel_features", "grid_thw",
            "initial_noise", "last_action_decoder_out",
            "dit_step_input", "dit_step_output", "dit_step_temb",
        )
        entry = {k: captured[k] for k in keep if k in captured}
        entry["_meta"] = {
            "traj": traj_idx, "step": step_idx, "seed": args.seed,
            "ckpt": ckpt, "tag": args.tag,
            "video_keys": list(policy.modality_configs["video"].modality_keys),
        }
        for k, v in entry.items():
            if hasattr(v, "shape"):
                print(f"    {k}: {tuple(v.shape)} {v.dtype}")
        aux_list.append(entry)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(aux_list, out_path)
    sz_mb = out_path.stat().st_size / 1e6
    print(f"\nwrote {out_path} ({sz_mb:.1f} MB, N={len(aux_list)})")


if __name__ == "__main__":
    main()
