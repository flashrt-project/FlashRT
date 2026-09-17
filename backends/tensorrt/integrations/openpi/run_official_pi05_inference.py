#!/usr/bin/env python3
"""Run the openpi Thor tutorial's deployment_scripts/pi05_inference.py unchanged,
with its TensorRT hook pointed at a FlashRT engine.

All arguments go to pi05_inference.py (--inference-mode tensorrt|compare,
--engine-path, --golden-noise-path, ...). The plugin library comes from
$FLASHRT_TRT_PLUGIN; without it the tutorial engine runs through the
tutorial's own hook. $EXAMPLE_SEED seeds NumPy so the synthetic example (random
images and state) is the same across runs. Run from the openpi checkout inside
the tutorial container.
"""
import functools
import os
import runpy
import sys

import tensorrt as trt

import deployment_scripts.trt_model_forward as tmf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openpi_flashrt import flashrt_sample_actions  # noqa: E402

import numpy as np  # noqa: E402

if os.environ.get("EXAMPLE_SEED"):
    np.random.seed(int(os.environ["EXAMPLE_SEED"]))
plugin = os.environ.get("FLASHRT_TRT_PLUGIN")
tutorial_setup = tmf.setup_pi0_tensorrt_engine


def flashrt_setup(policy, engine_path):
    trt.get_plugin_registry().load_library(plugin)
    policy = tutorial_setup(policy, engine_path)
    model = policy._model
    sample = functools.partial(flashrt_sample_actions, model)
    model.sample_actions = sample
    policy._sample_actions = sample
    print("FlashRT engine hooked to policy sample_actions")
    return policy


if plugin:
    tmf.setup_pi0_tensorrt_engine = flashrt_setup
sys.argv[0] = "deployment_scripts/pi05_inference.py"
runpy.run_path("deployment_scripts/pi05_inference.py", run_name="__main__")
