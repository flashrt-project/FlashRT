# Adopt structures in 20 lines

Every number below is a measured receipt from this repo's evidence
runs, on a single RTX 5090, against the unmodified Hugging Face host.

## The 20 lines (Qwen3-VL-8B, measured 2.56x)

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from flash_rt import structures
from flash_rt.structures.swap import attach

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16).to("cuda").eval()

def calibration():                       # one real forward, that's all
    with torch.no_grad():
        model(**inputs)

plan = structures.auto_swaps(            # discover -> calibrate -> build
    model, calibration,
    structures=("decoder_ffn", "linear_proj"),
    scheme="w8a16_decode")               # precision is a named scheme
attach(model, plan.swaps, observe=plan.observed, revert=plan.revert)

loop = structures.decode_loop(model, max_len=512)   # serving form
out = loop.generate(input_ids, max_new_tokens=64)   # greedy, gated
```

What you get, in the order you should check it:

1. `print(structures.explain(plan))` — what was bound, what was routed
   to which precision format, what stayed at host precision and why,
   what refused and why. If this table surprises you, stop here.
2. Parity gates before speed: the protocol is teacher-forced token
   agreement for LLMs (free-running comparisons cascade and prove
   nothing), stepwise value bands for diffusion. The probes under
   `HF-kernels-collab/tests/` are the runnable precedents.
3. Speed, paired, same process: swapped vs detached. `attach` returns
   a handle whose `detach()` restores the host bit-for-bit — that
   restoration is itself a gate.

## Gallery — the shipped recipes and their receipts

| Host | Recipe | Receipt |
|---|---|---|
| Qwen3-VL-8B | W8 decode band + `decode_loop` | 168 tok/s, 2.56x host, teacher-forced 1.0 |
| Qwen3-VL-8B multi-image | same + `loop.generate_from(inputs)` (host prefill, loop decode) | 1/2/4 images, tf 1.0/0.97/1.0, 1.43-1.47x |
| Qwen3-8B | `decode_loop` alone, zero swaps | 1.42x host, tf 0.979 |
| Qwen3.6-27B NVFP4 | `adopt_prequantized` + fused gated-delta + `w4a4_decode` + loop | 77.7 tok/s plain; 129.7 tok/s with the checkpoint's MTP head (`enable_mtp`), tokens bit-identical |
| Wan2.2 TI2V-5B | `fp8_static` chain + `torch.compile` | 2.31x per step, stepwise band 0.998 |

Three habits the receipts keep proving:

- **Right-size the static window.** The same VL cases read 0.70x under
  a blanket 4096-token window and 1.45x under a 512 one. Buckets are
  not optional.
- **Let refusals happen.** A seam kept at host precision or an adapter
  that steps aside is the system working; `explain` names every one.
- **Keep the receipt.** Every gate can call
  `flash_rt.structures.gates.save_record` — records carry an
  environment lock and verify with `verify_record`/`check_env`, so a
  number you measured today is a number you can defend later.
