"""Generate HF-eager reference fixtures for GR00T N1.6 / N1.7 from REAL robot data.

Generalizes tests/_helpers/groot_n17/gen_reference.py to both model versions and
to video-backed LeRobot datasets, and replaces the official ffmpeg frame backend
(which is O(N^2) -- one subprocess per frame, each decoding from the start, and
the only official backend that can read AV1 here) with the PyAV reader in
flash_rt.datasets.lerobot_video. That substitution is validated bit-for-bit
against the official backend; see docs/groot_n17_orin_sm87.md §5.3.

Data must be real (AGENTS.md §3.7 / skill principle: calibration and fidelity
inputs come from the host's real inference distribution). Synthetic tensors
mismeasure activation outliers and therefore pick the wrong quantization recipe
-- measured here: real frames have image std 42.1 vs 73.9 for uniform random, and
real state is in degrees (+-100) vs +-0.3 for randn*0.1.

Output fixture (torch.save dict):
  meta         version, ckpt, dataset, frame indices, seed, versions, shapes
  inputs       the obs dict handed to Gr00tPolicy.get_action (real frames/state/text)
  activations  per-block hidden states for every stage
  actions      decoded (unnormalized) action output

Usage:
  PYTHONPATH=/mnt/Isaac-GR00T:/mnt/FlashRT python \
    tests/_helpers/groot_orin/gen_reference.py --ver n17 --frames 0,100,300
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]

CKPTS = {
    "n17": "/mnt/GR00T/so101_sim_rynnbot/checkpoint-89-1.000",
    "n16": "/mnt/GR00T/lerobot_so100/checkpoint-10-1.000",
}
DATASETS = {
    "n17": "/mnt/groot_realdata/cube_to_bowl_5",
    "n16": "/mnt/groot_realdata/cube_to_bowl_5",
}
# expected block counts per version -- asserted so a silently truncated or
# differently-shaped backbone fails loudly instead of producing a bad reference.
# ``vlln`` is the action head's LayerNorm. N1.6 needs it captured because its
# output IS the context the DiT consumes and no later stage brackets it; N1.7's
# is bracketed by vlsa_block_0..3, and the N1.7 fixtures already on disk predate
# this hook (79 tensors, no ``vlln_out``) -- regenerating N1.7 adds it.
EXPECT = {
    "n17": {"vit": 24, "deepstack": 3, "llm": 16, "vlln": 1, "vlsa": 4, "dit": 32},
    "n16": {"vit": 27, "deepstack": 0, "llm": 16, "vlln": 1, "vlsa": 0, "dit": 32},
}
SEED = 0

# State units. ``cube_to_bowl_5`` stores SO101 joint positions in DEGREES
# (``shoulder_pan.pos`` ... ``gripper.pos``, range -99.4..100.0), but the two
# checkpoints disagree about what they expect:
#
#   n17 statistics.json q01/q99  = [-0.76, -1.74, -0.80, 0.80, -0.72, 0.00]
#                                  [ 0.88,  0.70,  1.57, 1.66,  0.76, 0.75]   -> RADIANS
#   n16 statistics.json q01/q99  = [-59.0, 27.7, 18.6, 46.1, -157.4, 0.6]
#                                  [ 62.8, 188.6, 175.1, 99.9, 109.4, 44.4]  -> DEGREES
#
# Converting the demo data to radians reproduces n17's percentiles closely
# (dim1 q01 -1.731 vs -1.741, dim3 q99 1.745 vs 1.658) while degrees are off by
# 57.3x, so n17 needs the conversion and n16 does not. Skipping it leaves the
# n17 normalized state at +-78..+-179 instead of +-1: out of distribution, which
# would corrupt the activation-outlier profile that decides the precision tier
# (AGENTS.md 3.7). Both the HF reference and FlashRT get the identical converted
# input, so the implementation gate is unaffected either way.
STATE_UNIT = {"n17": "deg2rad", "n16": "raw"}

#: Default meta.provenance. Names the real-robot SO101 set this helper was
#: first used on; a capture from any other dataset must pass --provenance,
#: because a fixture whose provenance mislabels its own data cannot support
#: the "real input distribution" claim the gates depend on (AGENTS.md §3.7).
DEFAULT_PROVENANCE = (
    "real robot frames (SO101 cube_to_bowl_5, AV1), PyAV decode validated "
    "bit-identical to the official ffmpeg backend")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ver", choices=list(CKPTS), required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--frames", default="0,100,300",
                   help="comma-separated GLOBAL frame indices into the dataset")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=str(REPO / "tests" / "fixtures"))
    p.add_argument("--time", action="store_true", help="also record E2E latency")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--tag", default="",
                   help="inserted into the fixture name before _frame, so a "
                        "second dataset can be captured without overwriting "
                        "the first (e.g. --tag _sim). Must match the tag "
                        "capture_aux.py and the tests use.")
    p.add_argument("--state-unit", choices=("raw", "deg2rad"), default=None,
                   help="default: per-version STATE_UNIT table (n17=deg2rad, "
                        "n16=raw) — see the comment above it for the evidence")
    p.add_argument("--provenance", default=DEFAULT_PROVENANCE,
                   help="recorded verbatim in meta.provenance. Must describe "
                        "the dataset actually being captured: the default "
                        "names the real-robot SO101 set, so a simulation "
                        "capture that omits this flag ships fixtures that "
                        "mislabel their own data")
    return p.parse_args()


def find_stage_modules(model, ver):
    """Locate the per-stage block lists by structure, not by a single hardcoded path.

    N1.6 backbone is Eagle (``backbone.model.vision_model`` / ``.language_model``),
    N1.7 is Qwen3-VL (``backbone.model.model.visual`` / ``.language_model``).
    Returns {stage: list_of_modules}; missing stages map to [].
    """
    out = {"vit": [], "deepstack": [], "llm": [], "vlln": [], "vlsa": [],
           "dit": [], "projector": []}
    bb = model.backbone

    visual = None
    for path in (("model", "model", "visual"), ("model", "vision_model")):
        cur = bb
        for a in path:
            cur = getattr(cur, a, None)
            if cur is None:
                break
        if cur is not None:
            visual = cur
            break
    if visual is None:
        raise AttributeError("could not locate the vision tower on the backbone")
    blocks = getattr(visual, "blocks", None)
    if blocks is None:  # SigLIP2 nests one level deeper
        blocks = visual.vision_model.encoder.layers
    out["vit"] = list(blocks)
    out["deepstack"] = list(getattr(visual, "deepstack_merger_list", []) or [])

    lm = None
    for path in (("model", "model", "language_model"), ("model", "language_model")):
        cur = bb
        for a in path:
            cur = getattr(cur, a, None)
            if cur is None:
                break
        if cur is not None:
            lm = cur
            break
    if lm is None:
        raise AttributeError("could not locate the language model on the backbone")
    layers = getattr(lm, "layers", None)
    if layers is None:
        layers = lm.model.layers
    out["llm"] = list(layers)
    if hasattr(bb.model, "mlp1"):
        out["projector"] = [bb.model.mlp1]

    ah = model.action_head
    vlln = getattr(ah, "vlln", None)
    if vlln is not None:
        out["vlln"] = [vlln]
    vlsa = getattr(ah, "vl_self_attention", None)
    if vlsa is not None and hasattr(vlsa, "transformer_blocks"):
        out["vlsa"] = list(vlsa.transformer_blocks)
    dit = getattr(ah, "model", None)
    if dit is not None and hasattr(dit, "transformer_blocks"):
        out["dit"] = list(dit.transformer_blocks)

    exp = EXPECT[ver]
    for stage, want in exp.items():
        got = len(out[stage])
        if got != want:
            raise AssertionError(
                f"{ver}: expected {want} {stage} blocks, found {got} -- the "
                f"backbone is not the shape this reference assumes (refusing to "
                f"write a fixture that would silently mis-gate the port)")
    return out


def register_hooks(mods, store):
    handles = []

    def make(key):
        def _h(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                store[key] = t.detach().to(torch.float32).cpu().clone()
        return _h

    naming = {
        "vit": "vit_block_{}", "deepstack": "deepstack_merger_{}",
        "llm": "llm_layer_{}", "vlsa": "vlsa_block_{}", "dit": "dit_block_{}",
        # single-module stage, but a format string with no placeholder is
        # returned unchanged by .format(), so it needs no special case
        "vlln": "vlln_out",
    }
    for stage, blocks in mods.items():
        if stage in naming:
            for i, b in enumerate(blocks):
                handles.append(b.register_forward_hook(make(naming[stage].format(i))))
        elif stage == "projector" and blocks:
            handles.append(blocks[0].register_forward_hook(make("projector_out")))
    return handles


def build_obs(ds, frame_index, *, state_to_radians: bool = False):
    """Real frame -> the obs dict Gr00tPolicy.get_action expects.

    ``state_to_radians`` converts the state and the ground-truth action from
    degrees to radians (see ``STATE_UNIT``); both live in the same units.
    """
    o = ds.load_frame(frame_index)
    scale = np.pi / 180.0 if state_to_radians else 1.0
    video = {}
    for k, img in o["images"].items():
        name = k.split(".")[-1]                     # observation.images.front -> front
        video[name] = img[None, None, ...]           # (B=1, T=1, H, W, 3) uint8
    state = {"state": (o["state"] * scale).astype(np.float32)[None, None, :]}
    return {
        "video": video,
        "state": state,
        "language": {"annotation.prompt": [[o["task"]]]},
    }, o, ("deg2rad" if state_to_radians else "raw")


def main():
    args = parse_args()
    ckpt = args.ckpt or CKPTS[args.ver]
    dataset = args.dataset or DATASETS[args.ver]
    frames = [int(x) for x in args.frames.split(",") if x.strip() != ""]
    state_unit = args.state_unit or STATE_UNIT[args.ver]

    import transformers
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy import Gr00tPolicy

    from flash_rt.datasets.lerobot_video import LeRobotVideoDataset

    print(f"[cfg] ver={args.ver} ckpt={ckpt}")
    print(f"[cfg] dataset={dataset}  frames={frames}  seed={args.seed}"
          f"  tag={args.tag or '(none)'}")
    print(f"[cfg] state_unit={state_unit}"
          f"{'  (degrees -> radians)' if state_unit == 'deg2rad' else ''}")
    print(f"[cfg] provenance={args.provenance}")
    print(f"[cfg] torch={torch.__version__} transformers={transformers.__version__}")
    print(f"[cfg] gpu cur_freq="
          f"{open('/sys/class/devfreq/17000000.gpu/cur_freq').read().strip()}")

    t0 = time.perf_counter()
    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                         model_path=ckpt, device=args.device, strict=True)
    model = policy.model.eval()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[load] {type(model).__name__} {nparam/1e6:.2f}M in {time.perf_counter()-t0:.1f}s "
          f"dtype={next(model.parameters()).dtype}")
    print(f"[load] attn_implementation={getattr(model.config, 'attn_implementation', '?')}")

    mods = find_stage_modules(model, args.ver)
    print("[stages] " + "  ".join(f"{k}={len(v)}" for k, v in mods.items() if v))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with LeRobotVideoDataset(dataset) as ds:
        for fi in frames:
            store: dict[str, torch.Tensor] = {}
            handles = register_hooks(mods, store)
            obs, raw, applied_unit = build_obs(
                ds, fi, state_to_radians=(state_unit == "deg2rad"))
            if applied_unit != state_unit:
                raise AssertionError(
                    f"build_obs applied {applied_unit!r}, expected {state_unit!r}")
            state_fed = obs["state"]["state"][0, 0]

            img = next(iter(obs["video"].values()))[0, 0]
            print(f"\n[frame {fi}] ep={raw['episode_index']} fi={raw['frame_index']} "
                  f"task={raw['task']!r}")
            print(f"  real image: shape={img.shape} mean={img.mean():.1f} std={img.std():.1f} "
                  f"min={img.min()} max={img.max()}")
            print(f"  state as stored ({len(raw['state'])}d): {np.round(raw['state'], 4)}")
            print(f"  state fed to model [{state_unit}]: {np.round(state_fed, 4)}")
            if "action" in raw:
                print(f"  real action (ground truth): {np.round(raw['action'], 3)}")

            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            t0 = time.perf_counter()
            action_dict, _info = policy.get_action(obs)
            torch.cuda.synchronize()
            t_first = (time.perf_counter() - t0) * 1e3

            for h in handles:
                h.remove()

            acts = {k: v for k, v in action_dict.items()}
            for k, v in acts.items():
                print(f"  predicted action[{k}]: shape={v.shape} "
                      f"min={v.min():.4f} max={v.max():.4f}")
            print(f"  captured {len(store)} activation tensors")

            lat = None
            if args.time:
                for _ in range(2):
                    torch.manual_seed(args.seed)
                    policy.get_action(obs)
                torch.cuda.synchronize()
                ts = []
                for _ in range(args.iters):
                    torch.manual_seed(args.seed)
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    policy.get_action(obs)
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t) * 1e3)
                lat = np.array(ts)
                print(f"  HF eager E2E: median={np.median(lat):.1f} ms "
                      f"min={lat.min():.1f} max={lat.max():.1f} "
                      f"-> {1000/np.median(lat):.2f} Hz  (first call {t_first:.1f} ms)")

            nviews = len(obs["video"])
            name = (f"gr00t_{args.ver}_ref_new_embodiment_{nviews}v"
                    f"{args.tag}_frame{fi}_seed{args.seed}.pt")
            path = out_dir / name
            torch.save({
                "meta": {
                    "version": args.ver, "ckpt": ckpt, "dataset": dataset,
                    "frame_index": fi, "episode_index": raw["episode_index"],
                    "frame_in_episode": raw["frame_index"],
                    "task": raw["task"], "seed": args.seed, "num_views": nviews,
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "model_class": type(model).__name__, "params_M": nparam / 1e6,
                    "attn_implementation": getattr(
                        model.config, "attn_implementation", None),
                    "n_activation_tensors": len(store),
                    "e2e_median_ms": float(np.median(lat)) if lat is not None else None,
                    "state_unit": state_unit,
                    "provenance": args.provenance,
                },
                "inputs": obs,
                "raw": {"state": raw["state"], "action": raw.get("action"),
                        "state_fed": np.asarray(state_fed, dtype=np.float32),
                        "action_fed": (
                            np.asarray(raw["action"], dtype=np.float32)
                            * (np.pi / 180.0 if state_unit == "deg2rad" else 1.0)
                            if raw.get("action") is not None else None),
                        "images": raw["images"]},
                "activations": store,
                "actions": {k: torch.as_tensor(v) for k, v in acts.items()},
            }, path)
            print(f"  wrote {path}  ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
