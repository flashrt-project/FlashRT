"""Capture the ``aux`` bundle the FlashRT N1.7 frontends need, from REAL robot data.

``Gr00tPolicy.get_action`` hides everything the FlashRT kernel path has to
reproduce: the post-patch-embed ViT input, the vision grid, the fused
text+image embeddings that enter the truncated LLM, the visual-token mask that
drives both DeepStack injection and the DiT's text/image cross-attention split,
the M-RoPE tables, and the diffusion noise. This script hooks the official model
once per frame and writes those tensors next to the reference fixture produced
by ``gen_reference.py``.

Generalizes ``tests/_helpers/groot_n17/capture_llm_aux.py`` to this repo's
Orin checkpoints and to the real SO101 dataset (the original hardcodes an
HF-cache checkpoint and the ``oxe_droid`` embodiment). Two differences that
matter:

* it also saves the DeepStack visual embeds themselves, not just their shapes,
  so ``deepstack_merge_forward`` can be gated directly;
* hooks bind arguments through ``inspect.Signature.bind`` rather than assuming
  a kwargs-only call style, because transformers 4.57 calls the text model with
  a mix of positional and keyword arguments.

Data must be real (AGENTS.md §3.7): synthetic inputs mismeasure activation
outliers, and the outlier profile is what decides the precision tier.

Usage:
  PYTHONPATH=/mnt/Isaac-GR00T:/mnt/FlashRT \
    /mnt/venvs/groot_n17/bin/python \
    tests/_helpers/groot_orin/capture_aux.py --ver n17 --frames 0,300
"""
from __future__ import annotations

import argparse
import functools
import inspect
import os
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
SEED = 0

# Per-version aux keys; see REQUIRED_BY_VER below for why these are enforced.
# ``pixel_values`` is the raw patch matrix the image->embeds fusion path starts
# from; aux bundles written before it was captured lack the key and can only
# drive the legacy pixel_features/llm_input_embeds path.
REQUIRED_N17 = (
    "pixel_values", "pixel_features", "input_ids", "grid_thw",
    "llm_input_embeds", "visual_pos_masks", "rope_cos", "rope_sin",
    "initial_noise",
)


def build_obs(ds, frame_index, *, state_to_radians: bool = False):
    """Real frame -> the obs dict ``Gr00tPolicy.get_action`` expects.

    Mirrors ``gen_reference.py``'s ``build_obs``; duplicated rather than
    imported because the helper directories are not packages and the two
    scripts must stay runnable standalone under their own venv. The state unit
    is not decided here — it is read back from the paired reference fixture's
    meta so the two can never disagree (see ``STATE_UNIT`` in
    ``gen_reference.py`` for the degrees-vs-radians evidence).
    """
    o = ds.load_frame(frame_index)
    scale = np.pi / 180.0 if state_to_radians else 1.0
    video = {}
    for k, img in o["images"].items():
        name = k.split(".")[-1]              # observation.images.front -> front
        video[name] = img[None, None, ...]   # (B=1, T=1, H, W, 3) uint8
    return {
        "video": video,
        "state": {"state": (o["state"] * scale).astype(np.float32)[None, None, :]},
        "language": {"annotation.prompt": [[o["task"]]]},
    }, o


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ver", choices=list(CKPTS), default="n17")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--frames", default="0,300")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tag", default="",
                   help="must match gen_reference.py's --tag: it selects which "
                        "reference fixture this aux bundle is paired with")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fixture-dir", default=str(REPO / "tests" / "fixtures"))
    return p.parse_args()


def _bind(orig, args, kwargs):
    """Normalize a hooked call into kwargs regardless of call style."""
    try:
        sig = inspect.signature(orig)
    except (TypeError, ValueError):
        return dict(kwargs)
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _hook_action_head(ah, cap: dict):
    """Hook the action head. Returns a ``restore()`` closure.

    Version-agnostic: N1.6 and N1.7 both use ``AlternateVLDiT`` as
    ``action_head.model`` plus a ``CategorySpecificMLP`` action_decoder, so
    the denoise-trajectory capture is identical and must not be duplicated.
    """
    orig_gawf = ah.get_action_with_features

    def gawf_hook(self, *args, **kwargs):
        orig_randn = torch.randn

        def patched_randn(*ra, **rkw):
            t = orig_randn(*ra, **rkw)
            if "initial_noise" not in cap:
                cap["initial_noise"] = t.detach().to(torch.float32).cpu().clone()
            return t

        torch.randn = patched_randn
        try:
            out = orig_gawf(*args, **kwargs)
        finally:
            torch.randn = orig_randn
        # The final post-Euler normalized action chunk — the direct E2E
        # reference for a port's infer(). (The action_decoder hook below only
        # ever sees per-step velocities.)
        for key in ("action_pred", "action", "actions", "noise_action"):
            if hasattr(out, "get") and out.get(key) is not None:
                cap["final_actions_norm"] = (
                    out[key].detach().to(torch.float32).cpu().clone())
                cap["final_actions_norm_key"] = key
                break
        else:
            cap["final_actions_norm_keys"] = list(out.keys()) if hasattr(
                out, "keys") else type(out).__name__
        return out

    ah.get_action_with_features = functools.partial(gawf_hook, ah)

    # ── per-denoise-step DiT input / output / timestep embedding ──
    dit = ah.model
    orig_dit = dit.forward
    cap["dit_step_input"], cap["dit_step_output"], cap["dit_step_temb"] = [], [], []

    def dit_hook(self, hidden_states, *args, **kwargs):
        cap["dit_step_input"].append(
            hidden_states.detach().to(torch.float32).cpu().clone())
        d = _bind(orig_dit, (hidden_states,) + args, kwargs)
        if d.get("timestep") is not None:
            cap["dit_step_temb"].append(
                d["timestep"].detach().to(torch.float32).cpu().clone())
        out = orig_dit(hidden_states, *args, **kwargs)
        t0 = out[0] if isinstance(out, tuple) else out
        cap["dit_step_output"].append(t0.detach().to(torch.float32).cpu().clone())
        return out

    dit.forward = functools.partial(dit_hook, dit)

    # ── action decoder output (pre-denormalization) ──
    ad = ah.action_decoder
    orig_ad = ad.forward
    cap["velocity_per_step"] = []

    def ad_hook(self, *args, **kwargs):
        out = orig_ad(*args, **kwargs)
        cap["velocity_per_step"].append(
            out.detach().to(torch.float32).cpu().clone())
        return out

    ad.forward = functools.partial(ad_hook, ad)

    def restore():
        ah.get_action_with_features = orig_gawf
        dit.forward = orig_dit
        ad.forward = orig_ad
    return restore


def install_hooks_n17(policy, cap: dict):
    """Hook the official N1.7 model to fill ``cap``. Returns the hook handles."""
    model = policy.model
    visual = model.backbone.model.model.visual
    lm = model.backbone.model.model.language_model

    # ── ViT input (post patch_embed + pos_embed) and the vision grid ──
    block0 = visual.blocks[0]

    def block0_pre(_m, args, kwargs):
        d = _bind(block0.forward, args, kwargs)
        h = d.get("hidden_states", args[0] if args else None)
        cap["pixel_features"] = h.detach().to(torch.float32).cpu().clone()

    h1 = block0.register_forward_pre_hook(block0_pre, with_kwargs=True)

    orig_visual = visual.forward

    def visual_hook(self, *args, **kwargs):
        d = _bind(orig_visual, args, kwargs)
        cap["grid_thw"] = d["grid_thw"].detach().cpu().clone()
        # raw patch matrix (num_patches, 1536) -- the input an image->embeds
        # fusion path needs. pixel_features above is *post* patch-embed, so a
        # port starting there still depends on HF for the fused LLM embeds.
        # transformers 4.57 names this parameter ``hidden_states`` and calls
        # ``self.patch_embed(hidden_states)`` on its first line; older versions
        # called it ``pixel_values``. Accept either.
        pv = d.get("hidden_states", d.get("pixel_values"))
        if pv is None:
            raise SystemExit(
                "visual.forward got no raw patch tensor under either "
                "'hidden_states' or 'pixel_values' -- the hook signature "
                "stopped matching; refusing to write an aux bundle that would "
                "silently force the fusion path off.")
        cap["pixel_values"] = pv.detach().to(torch.float32).cpu().clone()
        return orig_visual(*args, **kwargs)

    visual.forward = functools.partial(visual_hook, visual)

    # ── token ids, from the OUTER multimodal model ──
    # The text-model hook below cannot see them: Qwen3VLModel calls
    # ``language_model(inputs_embeds=...)`` so ``input_ids`` is None there. An
    # image->embeds fusion path needs the ids for its embedding lookup; in a
    # real deployment they come from the tokenizer, not from a model forward.
    outer = model.backbone.model.model
    orig_outer = outer.forward

    def outer_hook(self, *args, **kwargs):
        d = _bind(orig_outer, args, kwargs)
        ids = d.get("input_ids")
        if ids is None:
            raise SystemExit(
                "Qwen3VLModel.forward got no input_ids -- the hook signature "
                "stopped matching; refusing to write an aux bundle that would "
                "silently force the fusion path off.")
        cap["input_ids"] = ids.detach().cpu().clone()
        return orig_outer(*args, **kwargs)

    outer.forward = functools.partial(outer_hook, outer)

    # ── LLM input embeds + visual mask + DeepStack embeds ──
    orig_lm = lm.forward

    def lm_hook(self, *args, **kwargs):
        d = _bind(orig_lm, args, kwargs)
        if "inputs_embeds" in d and d["inputs_embeds"] is not None:
            cap["llm_input_embeds"] = (
                d["inputs_embeds"].detach().to(torch.float32).cpu().clone())
        if d.get("input_ids") is not None:
            cap["input_ids"] = d["input_ids"].detach().cpu().clone()
        if "visual_pos_masks" in d and d["visual_pos_masks"] is not None:
            cap["visual_pos_masks"] = (
                d["visual_pos_masks"].detach().cpu().clone())
        ds = d.get("deepstack_visual_embeds")
        if ds is not None:
            cap["deepstack_visual_embeds"] = [
                t.detach().to(torch.float32).cpu().clone() for t in ds]
        return orig_lm(*args, **kwargs)

    lm.forward = functools.partial(lm_hook, lm)

    # ── M-RoPE tables (already merged across the 3 axes by
    #    apply_interleaved_mrope, and cat(freqs, freqs) so both halves match) ──
    rot = lm.rotary_emb
    orig_rot = rot.forward

    def rot_hook(self, *args, **kwargs):
        cos, sin = orig_rot(*args, **kwargs)
        cap["rope_cos"] = cos.detach().cpu().clone()
        cap["rope_sin"] = sin.detach().cpu().clone()
        return cos, sin

    rot.forward = functools.partial(rot_hook, rot)

    restore_ah = _hook_action_head(model.action_head, cap)

    def restore():
        h1.remove()
        visual.forward = orig_visual
        outer.forward = orig_outer
        lm.forward = orig_lm
        rot.forward = orig_rot
        restore_ah()
    return restore


def install_hooks_n16(policy, cap: dict):
    """Hook the official N1.6 (Eagle) model to fill ``cap``.

    The hook paths are structurally different from N1.7 and were derived from
    ``gr00t/model/modules/eagle_backbone.py`` +
    ``Eagle-Block2A-2B-v2/modeling_eagle3_vl.py``:

    * ``EagleBackbone.forward`` passes only ``input_ids`` / ``attention_mask`` /
      ``pixel_values`` to ``Eagle3_VLForConditionalGeneration``;
    * that module gathers ``embed_tokens(input_ids)``, computes
      ``extract_feature(pixel_values, image_flags)`` and **scatters** the vision
      tokens into the image-token slots, then calls
      ``language_model(inputs_embeds=...)`` -- so ``inputs_embeds`` here is the
      fused tensor a port must reproduce (same boundary as N1.7);
    * ``extract_feature`` branches on the Eagle3-VL module's **own**
      ``select_layer``, which is ``-1`` for this checkpoint -- so it consumes
      ``last_hidden_state`` (post_layernorm of **all 27** layers) and there is
      **no ViT-truncation lever** here, unlike N1.7 whose DeepStack taps
      ``[5,11,17]`` make layers 18..23 dead. The top-level Gr00tN1d6
      ``select_layer=16`` truncates the *LLM* (eagle_backbone.py:52), not the ViT;
    * the vendored config sets ``use_rope=False`` / ``use_windows_attn=False``,
      so the ViT is plain full attention with a **learned** position embedding
      (``embeddings.position_embedding.weight``), and only the Qwen3 LLM has
      RoPE (1-D, not N1.7's M-RoPE).
    """
    model = policy.model
    e3 = model.backbone.model
    vis = e3.vision_model
    lm = e3.language_model
    # ``select_layer`` here is the **Eagle3-VL module's own** config value, which
    # is -1 for this checkpoint: ``extract_feature`` then consumes
    # ``last_hidden_state`` (= post_layernorm of all 27 layers) and the whole
    # tower is live. Do not confuse it with the top-level Gr00tN1d6
    # ``select_layer=16``, which truncates the *LLM* (eagle_backbone.py:52).
    # Recording which branch ran is what stops a port from gating against the
    # wrong ViT tap -- and it is why N1.6 has no ViT-truncation lever while
    # N1.7 (DeepStack taps [5,11,17]) does.
    sel = int(getattr(e3, "select_layer", -1))
    cap["select_layer"] = sel
    cap["vit_consumes"] = "last_hidden_state" if sel < 0 else f"hidden_states[{sel}]"

    # ── what the backbone was handed ──
    orig_e3 = e3.forward

    def e3_hook(self, *args, **kwargs):
        d = _bind(orig_e3, args, kwargs)
        for key in ("input_ids", "attention_mask", "image_flags"):
            if d.get(key) is not None:
                cap[key] = torch.as_tensor(d[key]).detach().cpu().clone()
        pv = d.get("pixel_values")
        if pv is None:
            raise SystemExit("Eagle3_VL forward got no pixel_values -- the hook "
                             "signature stopped matching; refusing to continue.")
        # pixel_values may be a list of per-image tensors (len() == num_images)
        cap["pixel_values"] = [torch.as_tensor(t).detach().to(torch.float32).cpu().clone()
                               for t in pv] if isinstance(pv, (list, tuple)) else \
            torch.as_tensor(pv).detach().to(torch.float32).cpu().clone()
        return orig_e3(*args, **kwargs)

    e3.forward = functools.partial(e3_hook, e3)

    # ── the ViT tap the projector actually consumes ──
    orig_vis = vis.forward

    def vis_hook(self, *args, **kwargs):
        out = orig_vis(*args, **kwargs)

        def f32(t):
            return t.detach().to(torch.float32).cpu().clone()

        # mirror extract_feature's own branch so the captured tensor is exactly
        # the one mlp1 consumes
        hs = getattr(out, "hidden_states", None)
        lhs = getattr(out, "last_hidden_state", None)
        if sel < 0:
            if lhs is None:
                raise SystemExit(
                    "select_layer=-1 but vision_model returned no "
                    "last_hidden_state -- the hook stopped matching.")
            cap["vit_select_layer_out"] = f32(lhs)
        else:
            if hs is None:
                raise SystemExit(
                    f"select_layer={sel} but vision_model returned no "
                    "hidden_states -- the hook stopped matching.")
            cap["vit_select_layer_out"] = f32(hs[sel])
            cap["vit_n_hidden_states"] = len(hs)
        if lhs is not None:
            cap["vit_last_hidden_state"] = f32(lhs)
        ss = getattr(out, "spatial_shapes", None)
        if ss is not None:
            cap["spatial_shapes"] = torch.as_tensor(ss).detach().cpu().clone()
        return out

    vis.forward = functools.partial(vis_hook, vis)

    # ── the fused LLM input embeds (the port's start boundary) ──
    orig_lm = lm.forward

    def lm_hook(self, *args, **kwargs):
        d = _bind(orig_lm, args, kwargs)
        if d.get("inputs_embeds") is None:
            raise SystemExit("Qwen3 LLM was called without inputs_embeds -- the "
                             "Eagle fusion path changed; refusing to continue.")
        cap["llm_input_embeds"] = (
            d["inputs_embeds"].detach().to(torch.float32).cpu().clone())
        if d.get("position_ids") is not None:
            cap["position_ids"] = (
                torch.as_tensor(d["position_ids"]).detach().cpu().clone())
        return orig_lm(*args, **kwargs)

    lm.forward = functools.partial(lm_hook, lm)

    # ── 1-D RoPE tables (Qwen3 LLM; the ViT has none) ──
    rot = lm.model.rotary_emb
    orig_rot = rot.forward

    def rot_hook(self, *args, **kwargs):
        cos, sin = orig_rot(*args, **kwargs)
        cap["rope_cos"] = cos.detach().cpu().clone()
        cap["rope_sin"] = sin.detach().cpu().clone()
        return cos, sin

    rot.forward = functools.partial(rot_hook, rot)

    restore_ah = _hook_action_head(model.action_head, cap)

    def restore():
        e3.forward = orig_e3
        vis.forward = orig_vis
        lm.forward = orig_lm
        rot.forward = orig_rot
        restore_ah()
    return restore


INSTALLERS = {"n17": install_hooks_n17, "n16": install_hooks_n16}

# aux keys every version must produce; a missing one means the hooks stopped
# matching the installed transformers, which must fail loudly rather than
# silently yield an aux bundle the frontend fills with wrong defaults.
REQUIRED_N16 = (
    "pixel_values", "input_ids", "attention_mask", "llm_input_embeds",
    "vit_select_layer_out", "rope_cos", "rope_sin", "initial_noise",
)
REQUIRED_BY_VER = {"n17": REQUIRED_N17, "n16": REQUIRED_N16}




def main():
    args = parse_args()
    ckpt = args.ckpt or CKPTS[args.ver]
    dataset = args.dataset or DATASETS[args.ver]
    frames = [int(x) for x in args.frames.split(",") if x.strip() != ""]

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy import Gr00tPolicy

    from flash_rt.datasets.lerobot_video import LeRobotVideoDataset

    print(f"[cfg] ver={args.ver} ckpt={ckpt}")
    print(f"[cfg] dataset={dataset} frames={frames} seed={args.seed}")
    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                         model_path=ckpt, device=args.device, strict=True)
    print(f"[load] {type(policy.model).__name__}")

    out_dir = Path(args.fixture_dir)
    with LeRobotVideoDataset(dataset) as ds:
        nviews = len(ds.video_keys)
        for fi in frames:
            base = out_dir / (
                f"gr00t_{args.ver}_ref_new_embodiment_{nviews}v"
                f"{args.tag}_frame{fi}_seed{args.seed}.pt")
            if not base.exists():
                raise SystemExit(
                    f"missing reference fixture {base}; run gen_reference.py "
                    f"--ver {args.ver} --frames {fi} first (the aux bundle is "
                    f"only meaningful paired with the activations it gates against)")
            # weights_only=False is deliberate: the paired fixture is written
            # locally by tests/_helpers/groot_orin/gen_reference.py in this repo
            # and is never downloaded, and the weights-only unpickler rejects
            # the numpy arrays and nested dicts it stores. Do not point this at
            # a fixture of unknown provenance.
            ref_meta = torch.load(base, map_location="cpu",
                                  weights_only=False)["meta"]
            state_unit = ref_meta.get("state_unit", "raw")

            cap: dict = {}
            restore = INSTALLERS[args.ver](policy, cap)
            obs, raw = build_obs(ds, fi,
                                 state_to_radians=(state_unit == "deg2rad"))
            print(f"\n[frame {fi}] task={raw['task']!r} state_unit={state_unit}")
            print(f"  state fed: "
                  f"{np.round(obs['state']['state'][0, 0], 4).tolist()}")
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            with torch.inference_mode():
                policy.get_action(obs)
            restore()

            required = REQUIRED_BY_VER[args.ver]
            missing = [k for k in required if k not in cap]
            if missing:
                raise SystemExit(
                    f"frame {fi}: hooks failed to capture {missing}. The "
                    f"official model's call signature probably changed; fix "
                    f"the version installer rather than proceeding with a partial aux.")

            print(f"[frame {fi}] captured:")
            for k, v in cap.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: {tuple(v.shape)} {v.dtype}")
                elif isinstance(v, list):
                    print(f"  {k}: list[{len(v)}] of "
                          f"{tuple(v[0].shape) if v and hasattr(v[0], 'shape') else '?'}")

            out = dict(cap)
            out["meta"] = {
                "version": args.ver, "ckpt": ckpt, "dataset": dataset,
                "frame_index": fi, "seed": args.seed,
                "paired_fixture": base.name,
                "state_unit": state_unit,
                "provenance": "real robot frames (SO101 cube_to_bowl_5, AV1); "
                              "aux captured by hooking the official "
                              "Gr00tPolicy forward",
            }
            path = base.with_name(base.stem + "_aux.pt")
            torch.save(out, path)
            print(f"  wrote {path} ({path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
