"""openpi reference for the accuracy gate: raw pi05_libero actions for one
observation, prompt and pinned noise, from the PyTorch model or (4th argument)
a TensorRT engine through the tutorial hook. Runs in the tutorial container."""
import sys
import types

import numpy as np
import torch
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies import policy_config
from openpi.training import config as _config

inp = np.load(sys.argv[1])
ckpt, out_path = sys.argv[2], sys.argv[3]
tok = PaligemmaTokenizer(max_len=200)
tokens = inp["tokens"].tolist()
prompt = tok._tokenizer.decode(tokens[1:-1])
check, _ = tok.tokenize(prompt)
n = len(tokens)
print("prompt:", repr(prompt), "| retokenized matches:", check[:n].tolist() == tokens)

config = _config.get_config("pi05_libero")
policy = policy_config.create_trained_policy(config, ckpt)
from deployment_scripts.trt_model_forward import install_attention_mask_dtype_fix  # noqa: E402

model = policy._model
install_attention_mask_dtype_fix(model)
example = {"observation/image": inp["image"], "observation/wrist_image": inp["wrist_image"],
           "observation/state": inp["state"], "prompt": prompt}
x = policy._input_transform(dict(example))
x = {k: (torch.from_numpy(np.array(v)).to("cuda")[None, ...] if not isinstance(v, dict) else
         {kk: torch.from_numpy(np.array(vv)).to("cuda")[None, ...] for kk, vv in v.items()}) for k, v in x.items()}
images = x["image"]
for key, img in images.items():
    if img.dtype == torch.uint8:
        images[key] = img.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
obs = types.SimpleNamespace(images=images, image_masks=x["image_mask"], state=x["state"],
                            tokenized_prompt=x.get("tokenized_prompt"), tokenized_prompt_mask=x.get("tokenized_prompt_mask"),
                            token_ar_mask=x.get("token_ar_mask"), token_loss_mask=x.get("token_loss_mask"))
print("image masks:", {k: bool(v.item()) for k, v in x["image_mask"].items()},
      "| prompt tokens used:", int(x["tokenized_prompt_mask"].sum()))
noise = torch.from_numpy(inp["noise"].astype(np.float32))[None].cuda()
if len(sys.argv) > 4:  # TensorRT engine through the tutorial's own hook
    from deployment_scripts.trt_model_forward import setup_pi0_tensorrt_engine  # noqa: E402

    setup_pi0_tensorrt_engine(policy, sys.argv[4])
    model = policy._model
    print("engine:", sys.argv[4])
with torch.no_grad():
    actions = model.sample_actions("cuda", obs, noise=noise, num_steps=10)
np.savez(out_path, actions=actions[0].float().cpu().numpy())
print("wrote", out_path, tuple(actions.shape), actions.dtype)
