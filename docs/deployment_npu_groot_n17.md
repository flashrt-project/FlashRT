# GR00T N1.7 on Ascend (910B4)

Model-specific guide. For the backend layout, the tested environment and the
native build, read [deployment_npu.md](deployment_npu.md) first — its *Build and
run* section is shared by every model on this backend.

## Scope, stated first

This is **compute groundwork, not a finished model integration.** There is no
`("groot_n17", "torch", "npu")` entry in `_PIPELINE_MAP`, so
`flash_rt.load_model(..., hardware="npu")` does not select it and the documented
public entry point cannot reach it. What exists is the compute — the ported
backbone and action head, three native kernels, the image path, the action
decode and the state encode — assembled by the caller, as *Usage* below shows.

What is missing is one piece: a `GrootN17TorchFrontendNpu` that reproduces the
official processor's observation bundle (the prompt's token partition, the two
rope tables, the interpolated patch positions and the instruction's language
embeddings) and registers itself. Until that exists, this page's *Usage* is what
a caller has to write, and every deployment writes it again — which is the
argument for finishing it rather than leaving it here.

Everything below was measured on one 910B4 die through the assembly shown.

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

Setup, once. `processor` is the official policy's processor (it owns the
normalisation parameters and the modality layout); `embodiment_id` and
`embodiment_tag` identify the trained slot; `action_dim` and `horizon` come from
the action head's config; `height`, `width` and `views` describe the camera.

```python
import torch
from flash_rt.npu.models.groot_n17 import backbone as bb
from flash_rt.npu.models.groot_n17 import pipeline as pl
from flash_rt.npu.models.groot_n17 import preprocess as pre
from flash_rt.npu.models.groot_n17.actions import ActionDecoder, StateEncoder
from flash_rt.npu.models.groot_n17.captured import CapturedFrame
from flash_rt.npu.models.groot_n17.weights import load_frame

DEVICE = "npu:0"
weights = load_frame(checkpoint_dir)
frame = CapturedFrame(bb.BoundBackbone(weights, device=DEVICE),
                      pl.BoundChain(weights, embodiment_id, device=DEVICE),
                      aux)
del weights                                  # 6.5 GB of host tensors

transform = pre.EvalImageTransform(height, width, device=DEVICE, images=views)
encoder = StateEncoder(processor, embodiment_tag, action_dim, device=DEVICE)
decoder = ActionDecoder(processor, embodiment_tag, device=DEVICE)
frame.capture()                  # once; it runs its own three warm-up passes
```

Per frame. The upload is explicit and is part of the frame's cost: the transform
is handed `frames.data_ptr()`, so it requires a tensor already on the device and
refuses a host one rather than faulting inside the kernel.

```python
frames = torch.from_numpy(raw_uint8_frames).to(DEVICE)   # (views, H, W, 3) uint8
noise = torch.randn(1, horizon, action_dim, dtype=torch.bfloat16, device=DEVICE)
frame.fill(pre.patch_rows(transform(frames)), encoder(state_dict), noise)
actions = decoder(frame.replay(), state_dict)             # {modality: tensor}
```

`aux` is the bundle that depends on the prompt alone, and it is the piece a
frontend would build. `CapturedFrame` reads these keys from it:

| key | what it is |
|---|---|
| `views` | camera views in one frame |
| `image_mask`, `attention_mask` | which sequence positions are image tokens, and which are live |
| `vit_cos`, `vit_sin` | the ViT's rope tables for this image grid |
| `llm_cos`, `llm_sin` | the language model's rope tables for this sequence |
| `text_embeds` | the instruction's embeddings, zero where an image token goes |
| `patch_shape` | `(patch tokens, 588)` — the voxel width the patch projection consumes |
| `patch_positions` | the interpolated position table for this image grid |

Every one of them is a function of the prompt and the camera geometry, not of the
frame, which is why the graph has exactly three inputs: patches, state and noise.
`CapturedChain` is the same object for the action head alone, for deployments
that reuse backbone features across frames.

A captured graph may not allocate, so every buffer is sized at bind time and
`fill` writes into them. `run_eager()` runs the same frame without capture, which
is how a replay is compared against it.

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

**The fused add-and-normalise** (`dit_norm_910b.cpp`). A draw on time against
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

## Build

The three units are compiled only when this model is selected, because a
deployment that serves Pi0.5 has no reason to build them:

```bash
FLASHRT_ENABLE_NPU_GROOT_N17=ON bash scripts/npu/build.sh
```

That adds `libflashrt_npu_dit_attn.so`, `libflashrt_npu_dit_norm.so` and
`libflashrt_npu_image.so` to `flash_rt/npu/lib`, overridable individually with
`FLASHRT_NPU_DIT_ATTENTION_LIBRARY`, `FLASHRT_NPU_DIT_NORM_LIBRARY` and
`FLASHRT_NPU_IMAGE_LIBRARY`. `FLASHRT_ENABLE_NPU_PI05=OFF` leaves the Pi0.5 units
out; at least one model has to be selected.

## Tests

No device needed — these run on any host:

```bash
python -m pytest tests/test_npu_operands.py \
                tests/test_npu_groot_n17_units.py \
                tests/test_npu_groot_n17_actions.py \
                tests/test_npu_groot_n17_chain.py -q
```

They cover the operand checks that stand in front of every raw pointer, each of
the three units' identity checks and the refusal a missing build gives, the
action decode and state encode including every representation they refuse, the
weight specification, and — against OpenCV itself — the tap construction that
makes the image transform bit-exact.

On a 910B4 with the units built:

```bash
python -m pytest tests/test_npu_groot_n17_device.py -q
```

Three numerical smokes at the exact shapes the model runs: the attention against
the vendor operator at all three DiT geometries and the two value layouts against
each other bit-for-bit, the norm against a reference that rounds where it rounds,
and the whole image transform against OpenCV.
