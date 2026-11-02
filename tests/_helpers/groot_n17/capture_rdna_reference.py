"""Generate independent GR00T BF16 fixtures for the RDNA regression tests.

Runs the official Isaac-GR00T policy only. No FlashRT model or preprocessing
is used to produce expected outputs. Requires the pinned processor environment
and model resources described in docs/deployment_amd_groot_n17.md.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from capture_aux_multi import _build_parsed, _install_hooks, _restore_hooks


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def official_policy(checkpoint, processor):
    """Load all inference weights from GR00T, without a second Cosmos download.

    Only paths and redundant backbone loading are adapted. The official forward
    methods are unchanged; incomplete checkpoint loading is a hard error.
    """
    import gr00t.model  # noqa: F401 — registers the official AutoModel class
    import transformers
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from transformers import AutoConfig, AutoModel, Qwen3VLForConditionalGeneration

    checkpoint, processor = Path(checkpoint).resolve(), Path(processor).resolve()
    audit = {"torch": torch.__version__, "hip": torch.version.hip,
             "transformers": transformers.__version__,
             "device": str(torch.cuda.get_device_properties(0)),
             "backbone_loading": "official architecture from config; weights from complete GR00T checkpoint"}
    # The official loader dispatches on the Cosmos name in this local path.
    with tempfile.TemporaryDirectory(prefix="groot-reference-") as directory:
        root = Path(directory)
        cosmos = root / "nvidia" / "Cosmos-Reason2-2B"
        cosmos.parent.mkdir()
        cosmos.symlink_to(processor, target_is_directory=True)
        overlay = root / "checkpoint"
        overlay.mkdir()
        for source in checkpoint.iterdir():
            target = overlay / source.name
            if source.name in ("config.json", "processor_config.json"):
                config = json.loads(source.read_text())
                if source.name == "config.json":
                    config["model_name"] = str(cosmos)
                else:
                    config["processor_kwargs"]["model_name"] = str(cosmos)
                target.write_text(json.dumps(config))
            else:
                target.symlink_to(source, target_is_directory=source.is_dir())

        original_load = AutoModel.from_pretrained

        def backbone_from_config(model_name, **kwargs):
            if str(model_name) != str(cosmos):
                raise ValueError(f"unexpected backbone: {model_name}")
            config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            return Qwen3VLForConditionalGeneration._from_config(
                config, torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                attn_implementation=kwargs.get("attn_implementation", "sdpa"))

        def audited_load(*args, **kwargs):
            model, info = original_load(*args, **kwargs, output_loading_info=True)
            audit["loading_info"] = info
            if any(info.get(key) for key in (
                    "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise RuntimeError(f"incomplete official checkpoint loading: {info}")
            return model

        with patch.object(Qwen3VLForConditionalGeneration, "from_pretrained",
                          side_effect=backbone_from_config), \
                patch.object(AutoModel, "from_pretrained", side_effect=audited_load):
            policy = Gr00tPolicy("OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
                                 str(overlay), device="cuda:0")
        policy.modality_configs["video"].modality_keys = (
            policy.modality_configs["video"].modality_keys[:2])
        audit["attention"] = str(policy.model.backbone.model.config._attn_implementation)
        audit["source_sha256"] = {
            str(path): sha256(path) for path in {
                Path(inspect.getfile(cls)) for cls in (
                    type(policy), type(policy.model), type(policy.model.backbone),
                    type(policy.model.action_head))}}
        audit["helper_sha256"] = {path.name: sha256(path) for path in (
            Path(__file__), Path(__file__).with_name("capture_aux_multi.py"))}
        audit["checkpoint_sha256"] = {p.name: sha256(p) for p in checkpoint.iterdir()
                                      if p.suffix in (".json", ".safetensors")}
        audit["processor_sha256"] = {p.name: sha256(p) for p in processor.iterdir()
                                     if p.suffix in (".json", ".txt")}
        yield policy, audit


@contextmanager
def fixed_noise(noise):
    """Supply the same tensor, rather than assuming equal cross-device RNGs."""
    calls = 0

    def sample(*args, **kwargs):
        nonlocal calls
        shape = kwargs.get("size", args[0] if args else None)
        if shape is None or tuple(shape) != tuple(noise.shape):
            raise RuntimeError(f"unexpected official inference RNG shape: {shape}")
        calls += 1
        return noise.to(device=kwargs["device"], dtype=kwargs["dtype"]).clone()

    with patch.object(torch, "randn", side_effect=sample):
        yield
    if calls != 1:
        raise RuntimeError(f"expected one official noise draw, got {calls}")


def prepare_official(policy, observation):
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import _rec_to_dtype

    samples = []
    for obs in policy._unbatch_observation(observation):
        step = policy._to_vla_step_data(obs)
        samples.append(policy.processor([
            {"type": MessageType.EPISODE_STEP.value, "content": step}]))
    return _rec_to_dtype(policy.collate_fn(samples), torch.bfloat16)


def to_device(value):
    if isinstance(value, torch.Tensor):
        return value.to("cuda:0")
    if hasattr(value, "items"):
        return {key: to_device(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item) for item in value]
    return value


def capture(policy, observation, noise):
    captured = {}
    hooks = _install_hooks(policy, captured)
    action_forward = policy.model.action_head.get_action_with_features

    def actions(*args, **kwargs):
        captured["state_normalized"] = kwargs["action_input"]["state"].detach().cpu().clone()
        result = action_forward(*args, **kwargs)
        captured["action_pred"] = result["action_pred"].detach().cpu().clone()
        return result

    def pixels(_module, args, kwargs):
        captured["pixel_values"] = (args[0] if args else kwargs["hidden_states"]).detach().cpu().clone()

    def tokens(_module, args, kwargs):
        captured["input_ids"] = kwargs["input_ids"].detach().cpu().clone()
        if kwargs.get("attention_mask") is not None:
            captured["attention_mask"] = kwargs["attention_mask"].detach().cpu().clone()

    policy.model.action_head.get_action_with_features = actions
    visual = policy.model.backbone.model.model.visual
    extra = [visual.register_forward_pre_hook(pixels, with_kwargs=True),
             policy.model.backbone.model.register_forward_pre_hook(tokens, with_kwargs=True)]
    try:
        with torch.inference_mode(), fixed_noise(noise):
            decoded, _ = policy.get_action(observation)
    finally:
        _restore_hooks(hooks)
        for handle in extra:
            handle.remove()
    required = ("input_ids", "visual_pos_masks", "rope_cos", "rope_sin", "grid_thw",
                "pixel_values", "pixel_features", "llm_input_embeds", "state_normalized",
                "initial_noise", "action_pred")
    entry = {key: captured[key] for key in required}
    if "attention_mask" in captured:
        entry["attention_mask"] = captured["attention_mask"]
    return dict(entry, inputs=observation, actions=decoded)


def noise_for_seed(seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((1, 40, 132), generator=generator).bfloat16()


def dataset_observation(loader, policy, trajectory, step):
    """Use PyAV's bundled CPU decoder, independent of TorchCodec's Torch ABI.

    Only video I/O is adapted; official temporal sampling, observation parsing
    and model preprocessing are preserved.
    """
    import av
    import numpy as np

    def frames_by_index(video_path, indices, decoder_kwargs=None):
        wanted = set(map(int, indices))
        decoded = {}
        with av.open(str(video_path)) as video:
            for index, frame in enumerate(video.decode(video=0)):
                if index in wanted:
                    decoded[index] = frame.to_ndarray(format="rgb24")
                if len(decoded) == len(wanted):
                    break
        if not wanted or wanted - decoded.keys():
            raise ValueError(f"invalid frame indices for {video_path}: {indices}")
        return np.stack([decoded[int(index)] for index in indices])

    with patch("gr00t.data.dataset.lerobot_episode_loader.get_frames_by_indices",
               side_effect=frames_by_index):
        return _build_parsed(loader, policy, trajectory, step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--dataset", help="pinned Isaac-GR00T demo_data/droid_sample directory")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--inputs", help="reuse observations/noise from a trusted official aux fixture")
    parser.add_argument("--matrix-inputs", help="reuse cases/noise from a trusted expanded fixture")
    parser.add_argument("--cases", default="1:20:0,1:110:0,1:200:0,0:20:0,2:20:0,1:110:7",
                        help="trajectory:step:seed triples for full aux capture")
    parser.add_argument("--extra-pairs", default="1:38,1:74,1:146,1:218")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not args.dataset and not (args.inputs and args.matrix_inputs):
        parser.error("provide --dataset or both --inputs and --matrix-inputs")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    with official_policy(args.checkpoint, args.processor) as (policy, audit):
        loader = None
        if args.dataset:
            import av
            from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
            audit["video_decoder"] = {"implementation": "PyAV CPU RGB24", "version": av.__version__}
            loader = LeRobotEpisodeLoader(
                dataset_path=args.dataset, modality_configs=policy.modality_configs,
                decoder_kwargs={"num_ffmpeg_threads": args.threads})
        if args.inputs:
            inputs = torch.load(args.inputs, map_location="cpu", weights_only=False)
        else:
            inputs = []
            for case in args.cases.split(","):
                trajectory, step, seed = map(int, case.split(":"))
                inputs.append(dict(inputs=dataset_observation(loader, policy, trajectory, step),
                                   initial_noise=noise_for_seed(seed),
                                   _meta=dict(trajectory=trajectory, step=step, seed=seed)))
        entries = []
        for item in inputs:
            entry = capture(policy, item["inputs"], item["initial_noise"])
            entry["_meta"] = item.get("_meta", {})
            entries.append(entry)
            print(f"captured aux {len(entries)}/{len(inputs)}", flush=True)
        torch.save(entries, args.output / "official-reference.pt")

        if args.matrix_inputs:
            matrix = torch.load(args.matrix_inputs, map_location="cpu", weights_only=False)
        else:
            observations = [item["inputs"] for item in inputs]
            for pair in filter(None, args.extra_pairs.split(",")):
                trajectory, step = map(int, pair.split(":"))
                observations.append(dataset_observation(loader, policy, trajectory, step))
            cases = [dict(index=index, seed=seed, steps=steps, initial_noise=noise_for_seed(seed))
                     for index in range(len(observations))
                     for seed in (0, 7, 23) for steps in (2, 4, 8)]
            cases += [dict(index=index, seed="aux", steps=4, initial_noise=item["initial_noise"])
                      for index, item in enumerate(inputs)]
            matrix = dict(observations=observations, entries=cases)
        expanded = []
        with torch.inference_mode():
            for case in matrix["entries"]:
                obs = matrix["observations"][case["index"]]
                ready = to_device(prepare_official(policy, obs))
                policy.model.action_head.num_inference_timesteps = case["steps"]
                with fixed_noise(case["initial_noise"]):
                    predicted = policy.model.get_action(**ready)["action_pred"].float().cpu()
                decoded = policy.processor.decode_action(
                    predicted.numpy(), policy.embodiment_tag, obs["state"])
                expanded.append(dict(case, action_pred=predicted, actions=decoded))
        torch.save(dict(observations=matrix["observations"], entries=expanded),
                   args.output / "official-expanded.pt")
        audit["arguments"] = {key: str(value) for key, value in vars(args).items()}
        audit["input_sha256"] = {str(p): sha256(p) for p in (args.inputs, args.matrix_inputs) if p}
        audit["output_sha256"] = {p.name: sha256(p) for p in args.output.glob("*.pt")}
        audit.update(aux_cases=len(entries), matrix_cases=len(expanded))
        (args.output / "provenance.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(f"saved {len(entries)} aux and {len(expanded)} matrix cases to {args.output}")


if __name__ == "__main__":
    main()
