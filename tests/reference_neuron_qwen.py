"""Independent Hugging Face CPU reference for the published Qwen checkpoint."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.set_num_threads(32)
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    ).eval()
    loaded = time.monotonic()
    rows = []
    for prompt in [
        "The capital of France is",
        "Q: What color is the sky?\nA:",
        "def fibonacci(n):\n    ",
    ]:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=16,
                pad_token_id=tokenizer.eos_token_id,
            )[0].tolist()
        rows.append(
            {
                "prompt": prompt,
                "token_ids": ids,
                "text": tokenizer.decode(ids, skip_special_tokens=True),
            }
        )
        print(rows[-1], flush=True)
    result = {
        "checkpoint": str(args.checkpoint),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "dtype": args.dtype,
        "device": "cpu",
        "max_new_tokens": 16,
        "load_seconds": loaded - start,
        "reference_seconds": time.monotonic() - loaded,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
