# GR00T N1.7 on Ascend (910B4)

Model-specific guide. For the backend layout, the tested environment and the
native build, read [deployment_npu.md](deployment_npu.md) first — its *Build and
run* section is shared by every model on this backend.

## Scope, stated first

This page describes a **native Ascend compute path that is not yet routed
through `load_model`.** There is no `("groot_n17", "torch", "npu")` entry in
`_PIPELINE_MAP`, so `flash_rt.load_model(..., hardware="npu")` will not select
it; the pipeline is driven directly, as *Usage* below shows. The frontend that
would close that gap has to reproduce the official processor's observation
bundle — prompt token partition, rope tables, patch positions — and that is a
separate change with its own on-device validation.

What is here is the compute: the ported backbone and action head, three native
kernels, the image path, and the tests. Everything below was measured on one
910B4 die.

## Performance

GR00T N1.7 (3 B: Qwen3-VL ViT + truncated Cosmos-Reason2 LLM + VL adapter +
32-layer DiT action head), 2 camera views, 4 flow-matching steps, action horizon
40. **Full frame, end to end**: one fresh observation in, denormalized
robot-space actions out — the same boundary the reference policy is timed at,
including the official transforms and the action decode.

| arm | median | vs eager |
|---|---|---|
| PyTorch eager, official policy | 396.67 ms | 1.00× |
| **this path** | **52.40 ms** (p90 52.61, min 52.27) | **7.6×** |

Three independent reference runs read 395.62, 396.67 and 397.29 ms, so treat the
baseline as ~396 ms. **`torch.compile` was not measured on this backend**; on
CDNA4 the same model compiled to 77.9 ms against eager's 67.9 — slower — so
compile is not assumed to be the free baseline here, and the judged baseline is
eager.

Of the 52.40 ms, **48.02 ms is the whole model in one captured graph** (1982
kernels) and the rest is host: the official transforms, the upload, and the
action decode. Device memory 6.09 GB.

## Accuracy

Combined denormalized-action cosine against the reference, with the initial
noise pinned to the reference's draw: **0.9999570**. Per modality —
end-effector 0.9998915, joint targets 0.9999829, gripper 0.9956549.

The 1-D gripper signal is near-constant over a trajectory, so its cosine is
naturally lower and is **not** a useful gate; end-effector and joint targets are
the ones to read. Judged against the original FP32 reference, never against an
intermediate arm.

Determinism: 300 replays of the captured action chain produce **one distinct
answer**, and the replay is bit-identical to the same graph run eagerly.

## Usage

The pipeline is assembled from three pieces: the weights, the two bound towers,
and the captured frame.

```python
import torch
from flash_rt.npu.models.groot_n17 import backbone as bb
from flash_rt.npu.models.groot_n17 import pipeline as pl
from flash_rt.npu.models.groot_n17 import preprocess as pre
from flash_rt.npu.models.groot_n17.actions import ActionDecoder, StateEncoder
from flash_rt.npu.models.groot_n17.captured import CapturedFrame
from flash_rt.npu.models.groot_n17.weights import load_frame

weights = load_frame(checkpoint_dir)
frame = CapturedFrame(bb.BoundBackbone(weights, device="npu:0"),
                      pl.BoundChain(weights, embodiment_id, device="npu:0"),
                      aux)
del weights

transform = pre.EvalImageTransform(height, width, device="npu:0", images=views)
encoder = StateEncoder(processor, embodiment_tag, action_dim, device="npu:0")
decoder = ActionDecoder(processor, embodiment_tag, device="npu:0")

frame.fill(pre.patch_rows(transform(raw_uint8_frames)),
           encoder(state_dict),
           torch.randn(1, horizon, action_dim, dtype=torch.bfloat16, device="npu:0"))
frame.capture()                       # once; it runs its own three warm-up passes
actions = decoder(frame.replay(), state_dict)
```

`aux` is the observation bundle the official processor produces — the image/text
token partition, the rope tables and the patch positions. `CapturedChain` is the
same thing for the action head alone, for deployments that reuse backbone
features across frames.

A captured graph may not allocate, so every buffer is sized at bind time and
`fill` writes into them. `capture()` runs the frame eagerly three times before
recording it, and `run_eager()` is there to compare a replay against the same
graph run without capture.

## What is native here, and why

Three kernels, each of which exists because the vendor operator at this model's
shapes is dominated by fixed cost rather than by work.

**The DiT's attention** (`csrc/npu/kernels/dit_attn_910b.cpp`). 41 query rows
against 41, 13 or 448 keys over 32 heads of 48 channels. The vendor's prompt
flash-attention charges 42–49 us for every one of those geometries — the cost
barely moves between 13 keys and 448 — while the 41×41 case moves 378 KB and
does 10 MFLOP. This drives `Mmad` directly, keeps a whole head in L0 untiled,
and does the row softmax over a whole score plane with no vector-to-scalar round
trips: **14.9 us against 42.0**, cosine 0.999996. The value operand is
transposed on the way from L1 to L0B rather than by a separate permute, which is
bit-exact and removes two launches a layer.

**The fused add-and-normalise** (`dit_vector_910b.cpp`). A draw on time against
the vendor's fused form, and it is here for its arithmetic: the residual sum is
rounded to BF16 once, because that value is what the next block carries forward,
and the normalisation runs in FP32 from it rather than rounding a second time.
It also adds the bias of the projection that produced its branch, which removes
that projection's per-call bias cast.

**The evaluation image transform's resize** (`area_resize_910b.cpp`). The
reference's transform is a smallest-edge resize, a fractional centre crop and a
second resize, all `cv2.INTER_AREA` on uint8 — 7.0 ms of host a frame. This
reproduces OpenCV's enlarging path **bit for bit**: 0 mismatched pixels over 28
million, for 0.30 ms of device. A torch version of the same arithmetic measures
8.9 ms, because an integer right shift on this part runs at 15 GB/s where a
multiply runs at 151, and the shifts cannot become float multiplies outside a
kernel — the vertical pass forms a 26-bit product and FP32 has 24 bits.

Everything else runs on `torch_npu` operators with the weights in Ascend
fractal-NZ layout.

## Precision

BF16 throughout. Ascend 910/A2 parts have no FP8 tensor hardware, so
`precision="fp8"` is refused by the backend before any frontend import.

INT8 on the DiT was built and measured and is **not** shipped. Its arithmetic is
fine — every one of the DiT's 192 projections quantised per row reads 0.9999571
end to end — but as separate kernels it is net negative on this part: the
quantised GEMM wins 3.05 us a call while the kernel types it adds to the block's
loop make the launches already in that loop slower. On this part a kernel's cost
includes what else is interleaved with it, measured at about 4.5 us a launch, so
a quantised path only pays if it rides in the epilogue of a kernel that was
going to be launched anyway.

## Tests

```bash
python -m pytest tests/test_npu_groot_n17_chain.py -q      # no device needed
python -m pytest tests/test_npu_contracts.py -q            # no device needed
```

The chain tests cover the weight specification, the geometry the attention
kernel accepts and refuses, the padded-operand contract, and the action decode's
shape and ordering, all against fakes so they run on any host. Kernel numerics
and latency need a 910B4 and the built shared objects.
