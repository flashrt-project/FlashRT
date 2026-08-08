"""Export the official host's prepared model-level inputs once.

Both GR00T N1.7 hosts run the same checkpoint and the same backbone
contract (input_ids / attention_mask / pixel_values / image_grid_thw
plus state / embodiment_id). Saving the official host's prepared
tensors lets the LeRobot host be timed on identical inputs, so the two
measurements differ in host code and nothing else.
"""

import argparse
from pathlib import Path

import torch

from groot_n17 import clone_tree, load_policy

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--backbone-assets", type=Path, required=True)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
OUT = args.out

policy = load_policy(args.host, args.checkpoint, args.backbone_assets)
model = policy.model
fixture = torch.load(args.fixture, map_location="cpu",
                     weights_only=False)["inputs"]

captured = {}
original = model.get_action


def spy(inputs, options=None):
    captured["inputs"] = clone_tree(inputs)
    return original(inputs, options)


model.get_action = spy
with torch.inference_mode():
    policy.get_action(fixture)
model.get_action = original

backbone_inputs, action_inputs = model.prepare_input(dict(captured["inputs"]))


def to_cpu(tree):
    if torch.is_tensor(tree):
        return tree.detach().cpu()
    if isinstance(tree, dict):
        return {k: to_cpu(v) for k, v in tree.items()}
    return tree


payload = {"backbone_inputs": to_cpu(dict(backbone_inputs)),
           "action_inputs": to_cpu(dict(action_inputs))}
torch.save(payload, OUT)
for group, tree in payload.items():
    for key, value in tree.items():
        shape = tuple(value.shape) if torch.is_tensor(value) else value
        print(f"{group}.{key}: {shape}")
print("saved", OUT)
