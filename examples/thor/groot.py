"""GR00T N1.7 LIBERO: local checkpoint setup, reference capture and FlashRT checks."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from _validation import load_groot_input, metrics, save_report


def prepare_checkpoint(checkpoint, cosmos, output):
    """Use a separate config overlay; leave downloaded checkpoint files untouched."""
    checkpoint, cosmos, output = map(lambda p: Path(p).resolve(), (checkpoint, cosmos, output))
    if output.exists():
        raise FileExistsError(f"Choose a new overlay directory: {output}")
    alias = output.parent / "nvidia/Cosmos-Reason2-2B"
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists():
        alias.symlink_to(cosmos, target_is_directory=True)
    if alias.resolve() != cosmos:
        raise ValueError("Existing Cosmos link points to another checkpoint")
    output.mkdir()
    for item in checkpoint.iterdir():
        if item.name in ("config.json", "processor_config.json"):
            config = json.loads(item.read_text())
            if item.name == "config.json":
                config["model_name"] = str(alias)
            else:
                config["processor_kwargs"]["model_name"] = str(alias)
                video = config["processor_kwargs"]["modality_configs"]["libero_sim"]["video"]
                if video["delta_indices"] != [0]:
                    raise ValueError("Expected LIBERO single-frame checkpoint")
                video["modality_keys"] = ["image"]
            (output / item.name).write_text(json.dumps(config, indent=2))
        else:
            (output / item.name).symlink_to(item, target_is_directory=item.is_dir())
    print(output)


def capture_reference(policy, inputs):
    """Capture setup tensors once. Timing below runs the policy without hooks."""
    import torch
    helper_dir = Path(__file__).resolve().parents[2] / "tests/_helpers/groot_n17"
    sys.path.insert(0, str(helper_dir))
    from capture_aux_multi import _install_hooks, _restore_hooks
    aux = {}
    hooks = _install_hooks(policy, aux)
    backbone = policy.model.backbone.model
    handle = backbone.register_forward_pre_hook(
        lambda module, args, kwargs: aux.update(input_ids=kwargs["input_ids"].detach().cpu()),
        with_kwargs=True)
    visual = backbone.model.visual
    original = visual.forward

    def visual_forward(hidden_states, grid_thw, **kwargs):
        aux["pixel_values"] = hidden_states.detach().cpu()
        return original(hidden_states, grid_thw, **kwargs)

    visual.forward = visual_forward
    try:
        torch.manual_seed(0)
        np.random.seed(0)
        with torch.inference_mode():
            result = policy.get_action(inputs)
    finally:
        handle.remove()
        visual.forward = original
        _restore_hooks(hooks)
    return aux, result[0] if isinstance(result, tuple) else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("prepare", "reference", "fp8", "fp4"), default="fp4")
    parser.add_argument("--cosmos")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "assets/groot_libero.npz")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/groot.json"))
    parser.add_argument("--cpu-threads", type=int, default=2,
                        help="PyTorch intra-op CPU threads for the image processor (default: 2)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.cosmos:
            parser.error("prepare requires --cosmos")
        prepare_checkpoint(args.checkpoint, args.cosmos, args.output)
        return 0
    if args.cpu_threads < 1 or args.iters < 1 or args.warmup < 0:
        parser.error("cpu-threads/iters must be positive and warmup nonnegative")
    import torch
    import gr00t.model  # Registers the official AutoProcessor.
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
    inputs = load_groot_input(args.input)
    tag = EmbodimentTag.resolve("LIBERO_PANDA")
    component_samples = []
    if args.mode == "reference":
        policy = Gr00tPolicy(embodiment_tag=tag, model_path=args.checkpoint, device="cuda:0")
        torch.set_num_threads(args.cpu_threads)
        aux, expected = capture_reference(policy, inputs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(aux=aux, actions=expected, inputs=inputs), args.output.with_suffix(".pt"))

        def run():
            torch.manual_seed(0)
            np.random.seed(0)
            with torch.inference_mode():
                result = policy.get_action(inputs)
            return result[0] if isinstance(result, tuple) else result

        actual = run()
        keys = sorted(expected)
        report = metrics(np.concatenate([actual[k] for k in keys], -1),
                         np.concatenate([expected[k] for k in keys], -1))
        report["passed"] = report["worst_cosine"] >= .995
    else:
        if not args.reference:
            parser.error("FlashRT modes require a fresh --reference .pt capture")
        reference = torch.load(args.reference, map_location="cpu", weights_only=False)
        for group in inputs:
            for key, value in inputs[group].items():
                if not np.array_equal(value, reference["inputs"][group][key]):
                    raise ValueError("Input differs from reference capture")
        import flash_rt
        model = flash_rt.load_model(args.checkpoint, config="groot_n17", framework="torch",
                                    hardware="thor", num_views=1, embodiment_tag="libero_sim",
                                    use_fp4=args.mode == "fp4")
        frontend = model.pipeline
        aux = reference["aux"]
        prompt = str(next(iter(inputs["language"].values()))[0][0])
        frontend.set_prompt(aux=aux, prompt=prompt)
        torch.set_num_threads(args.cpu_threads)
        from gr00t.data.types import MessageType
        processor = frontend._hf_processor()
        processor.eval()
        prep = Gr00tPolicy.__new__(Gr00tPolicy)
        prep.processor, prep.embodiment_tag = processor, tag
        prep.modality_configs = processor.get_modality_configs()[tag.value]
        prep.language_key = prep.modality_configs["language"].modality_keys[0]
        state = {"state." + key: value for key, value in inputs["state"].items()}
        noise = aux["initial_noise"].to("cuda").bfloat16().contiguous()

        def prepare():
            steps = [processor([{"type": MessageType.EPISODE_STEP.value,
                                 "content": prep._to_vla_step_data(obs)}])
                     for obs in prep._unbatch_observation(inputs)]
            return _rec_to_dtype(processor.collator(steps), torch.bfloat16)["inputs"]

        check = prepare()
        for key, captured in (("pixel_values", "pixel_values"), ("input_ids", "input_ids"),
                              ("image_grid_thw", "grid_thw")):
            if not torch.equal(check[key].cpu(), aux[captured].to(check[key].dtype).cpu()):
                raise ValueError(f"Processor input differs: {key}")

        def run():
            start = time.perf_counter()
            processed = prepare()
            preprocessing = (time.perf_counter() - start) * 1000
            torch.cuda.synchronize()
            start = time.perf_counter()
            frontend._backbone_features = frontend.run_backbone_graph(
                {"pixel_values": processed["pixel_values"].to("cuda")})
            output = frontend.infer(processed["state"].to("cuda"), initial_noise=noise,
                                    num_inference_timesteps=4, action_horizon=40)
            torch.cuda.synchronize()
            inference = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            decoded = frontend.denormalize_action(output, state_dict=state)
            torch.cuda.synchronize()
            decoding = (time.perf_counter() - start) * 1000
            component_samples.append(dict(preprocessing_ms=preprocessing, model_ms=inference,
                                          decoding_ms=decoding))
            return output, decoded

        first, actual = run()
        first = first.detach().cpu().clone()
        repeat, _ = run()
        repeat = repeat.detach().cpu().clone()
        repeat_metrics = metrics(first.float(), repeat.float())
        keys = sorted(reference["actions"])
        actual_arrays = {key: np.asarray(actual[key]) for key in keys}
        report = metrics(np.concatenate([actual_arrays[k] for k in keys], -1),
                         np.concatenate([reference["actions"][k] for k in keys], -1))
        report["repeat"] = repeat_metrics
        report["repeat_identical"] = bool(torch.equal(first, repeat))
        report["shape"] = list(np.concatenate([actual_arrays[k] for k in keys], -1).shape)
        groups = dict(position=("x", "y", "z"), rotation=("roll", "pitch", "yaw"), gripper=("gripper",))
        report["groups"] = {group: metrics(np.concatenate([actual_arrays[k] for k in members], -1),
                                           np.concatenate([reference["actions"][k] for k in members], -1))
                            for group, members in groups.items()}
        report["passed"] = report["mean_cosine"] >= .999 and report["worst_cosine"] >= .995 and repeat_metrics["worst_cosine"] >= .9999 and repeat_metrics["max_abs"] <= .05
        report["strict_group_diagnostic_passed"] = all(report["groups"][g]["worst_cosine"] >= .995 for g in ("position", "rotation")) and report["groups"]["gripper"]["max_abs"] <= .05
    for _ in range(args.warmup):
        run()
    samples = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    report.update(mode=args.mode, cpu_threads=torch.get_num_threads(), warmup=args.warmup,
                  iters=args.iters, full_call_ms=float(np.median(samples)),
                  p95_ms=float(np.percentile(samples, 95)))
    if component_samples:
        measured = component_samples[-args.iters:]
        report.update({key: float(np.median([sample[key] for sample in measured]))
                       for key in measured[0]})
    save_report(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
