#!/usr/bin/env python3
"""Run GR00T N1.7 BF16 on AMD RDNA with projected, raw aux or frames inputs."""

import argparse
import json
import time

import numpy as np
import torch

import flash_rt
from flash_rt.amd.frontends.torch.groot_n17_rdna import frames_from_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", required=True, help="Cosmos-Reason2 tokenizer/image processor directory")
    parser.add_argument("--observation", required=True,
                        help="NPZ with state.* and either projected features or video.* and instruction")
    parser.add_argument("--hardware", default="auto",
                        choices=("auto", "amd_rdna3", "amd_rdna35", "amd_rdna4"))
    parser.add_argument("--input", choices=("projected", "aux", "frames"), default="frames")
    parser.add_argument("--embodiment", default="oxe_droid_relative_eef_relative_joint")
    parser.add_argument("--benchmark", type=int, default=0, help="number of measured predictions after warmup")
    args = parser.parse_args()
    if args.benchmark < 0:
        parser.error("--benchmark must be nonnegative")
    if args.input == "projected" and args.benchmark:
        parser.error("--benchmark requires aux or frames; saved projected features exclude model computation")
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    with np.load(args.observation, allow_pickle=False) as data:
        observation = {"state": {key.removeprefix("state."): data[key] for key in data.files
                                 if key.startswith("state.")}}
        if args.input == "projected":
            aux = {key: torch.from_numpy(data[key]) for key in (
                "pixel_features", "llm_input_embeds", "grid_thw",
                "visual_pos_masks", "rope_cos", "rope_sin")}
        else:
            observation["video"] = {key.removeprefix("video."): data[key] for key in data.files
                                    if key.startswith("video.")}
            instruction = str(data["instruction"].item())
    processor = Gr00tN1d7Processor.from_pretrained(args.checkpoint, model_name=args.processor)
    camera_order = processor.modality_configs[args.embodiment]["video"].modality_keys
    model = flash_rt.load_model(args.checkpoint, config="groot_n17", framework="torch",
                               num_views=len(camera_order), use_fp8=False,
                               hardware=args.hardware, embodiment_tag=args.embodiment,
                               fuse_image_embeds=args.input != "projected")
    frontend = model.pipeline
    frontend.set_hf_processor(processor)
    if args.input != "projected":
        language_key = processor.modality_configs[frontend.embodiment_tag]["language"].modality_keys[0]
        observation["language"] = {language_key: [[instruction]]}
        _, aux = frontend.prepare_observation(observation)
    model.set_prompt(aux=aux)

    def run_observation():
        # For a live stream, refresh the observation (including both projected
        # features in that mode). Rebind the prompt if instruction/grid changes.
        if args.input == "aux":
            state, current_aux = frontend.prepare_observation(observation)
            normalized = model.infer(state, aux=current_aux)
        elif args.input == "frames":
            frames = frames_from_observation(observation, camera_order)
            state = frontend.normalize_state({"state." + k: v for k, v in observation["state"].items()})
            normalized = model.infer(state, frames=frames)
        else:
            state = frontend.normalize_state(observation["state"])
            normalized = model.infer(state, aux=aux)
        return frontend.denormalize_action(normalized, state_dict=observation["state"])

    actions = run_observation()
    report = {"hardware": getattr(frontend, "hardware", args.hardware), "input": args.input,
              "action_shapes": {k: list(v.shape) for k, v in actions.items()},
              "finite": all(bool(torch.isfinite(v).all()) for v in actions.values())}
    if args.benchmark:
        for _ in range(5):
            run_observation()
        wall = []
        for _ in range(args.benchmark):
            started = time.perf_counter()
            run_observation()
            wall.append((time.perf_counter() - started) * 1000)
        report.update(observation_p50_ms=float(np.median(wall)),
                      timing_scope="frame preparation, transfers, state normalization, model and physical decoding; excludes setup/warmup")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
