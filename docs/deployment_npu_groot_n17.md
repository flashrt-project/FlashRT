# GR00T N1.7 on Ascend (910B4)

Model-specific guide. For the backend layout, the tested environment and the
native build, read [deployment_npu.md](deployment_npu.md) first — its *Build and
run* section is shared by every model on this backend.

## Scope, stated first

`("groot_n17", "torch", "npu")` is registered, so `flash_rt.load_model` routes
here and `model.pipeline` is the entry point. As on every other hardware this
model runs on, **the observation bundle is the caller's**: `VLAModel.predict`
takes a prompt string alone and this model needs the prompt's token partition and
its rope tables as well, so the four-call contract below is the path — the same
one Thor and CDNA4 expose.

Two things are deliberately not served, and both are refused rather than
approximated. There is **no INT8 tier** — it was built, measured and left out
(see *Precision*) — and a request for one raises instead of quietly running BF16,
because a downgrade nobody was told about gets measured as a regression. And
`infer` without a fresh observation, reusing the previous frame's backbone
features, raises too: this tier captures the whole frame in one graph, and a
features-only replay is a second graph that has not been measured.

The figures below were measured on one 910B4 die through the assembly the
frontend performs.

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

```python
import flash_rt

model = flash_rt.load_model(checkpoint_dir, config="groot_n17",
                            framework="torch", hardware="npu", num_views=2,
                            embodiment_tag=embodiment_tag)
fe = model.pipeline
fe.set_hf_processor(processor)        # the official policy's processor
fe.set_prompt(aux=aux, prompt=instruction)

state = fe.normalize_state(state_dict)
out = fe.infer(state, aux={"frames": frames})      # (views, H, W, 3) uint8, on the device
actions = fe.denormalize_action(out, state_dict=state_dict)
```

`frames` has to be on the device: the image transform is handed their address, so
a host tensor is refused rather than faulting inside the kernel, and the upload is
part of the frame's cost and belongs where the caller can see it. A caller that
already has the patch rows passes `aux={"patches": ...}` instead; exactly one of
the two is required.

`aux` also carries what depends on the prompt and the camera geometry rather than
on the frame, which is why the captured graph has exactly three inputs. Every key
is checked on the way into `set_prompt`, because a missing one would otherwise
surface as a shape error several calls later:

| key | what it is |
|---|---|
| `views` | camera views in one frame |
| `image_mask`, `attention_mask` | which sequence positions are image tokens, and which are live |
| `vit_cos`, `vit_sin` | the ViT's rope tables for this image grid |
| `llm_cos`, `llm_sin` | the language model's rope tables for this sequence |
| `text_embeds` | the instruction's embeddings, zero where an image token goes |
| `patch_shape` | `(patch tokens, 588)` — the voxel width the patch projection consumes |
| `patch_positions` | the interpolated position table for this image grid |

The official processor and the reference backbone's rotary embeddings are where
these come from, and none of it is Ascend-specific — it is the caller's on Thor
and CDNA4 for the same reason.

### Underneath

The frontend is an adapter; the compute is reachable on its own, which is how the
kernels are benchmarked and how a host that already owns its own scheduling can
drive them:

```python
import torch
from flash_rt.npu.models.groot_n17 import backbone as bb, pipeline as pl
from flash_rt.npu.models.groot_n17 import preprocess as pre
from flash_rt.npu.models.groot_n17.captured import CapturedFrame
from flash_rt.npu.models.groot_n17.weights import load_frame

weights = load_frame(checkpoint_dir)
frame = CapturedFrame(bb.BoundBackbone(weights, device="npu:0"),
                      pl.BoundChain(weights, embodiment_id, device="npu:0"), aux)
del weights                                  # 6.5 GB of host tensors
frame.capture()                              # runs its own three warm-up passes

transform = pre.EvalImageTransform(height, width, device="npu:0", images=views)
frame.fill(pre.patch_rows(transform(frames)), state, noise)
actions = frame.replay()
```

A captured graph may not allocate, so every buffer is sized at bind time and
`fill` writes into them. `run_eager()` runs the same frame without capture, which
is how a replay is compared against it. `CapturedChain` is the same object for the
action head alone.

## What is native here, and why

Three kernels, each of which exists because the vendor operator at this model's
shapes is dominated by fixed cost rather than by work.

**The DiT's attention** (`csrc/npu/kernels/groot_n17/dit_attn_910b.cpp`). 41 query rows
against 41, 13 or 448 keys over 32 heads of 48 channels. The vendor's prompt
flash-attention charges 42–49 us for every one of those geometries — the cost
barely moves between 13 keys and 448 — while the 41×41 case moves 378 KB and
does 10 MFLOP. This drives `Mmad` directly, keeps a whole head in L0 untiled,
and does the row softmax over a whole score plane with no vector-to-scalar round
trips: **14.9 us against 42.0**, cosine 0.999996. The value operand is
transposed on the way from L1 to L0B rather than by a separate permute, which is
bit-exact and removes two launches a layer.

**The fused add-and-normalise** (`groot_n17/dit_norm_910b.cpp`). A draw on time against
the vendor's fused form, and it is here for its arithmetic: the residual sum is
rounded to BF16 once, because that value is what the next block carries forward,
and the normalisation runs in FP32 from it rather than rounding a second time.
It also adds the bias of the projection that produced its branch, which removes
that projection's per-call bias cast.

**The evaluation image transform's resize** (`groot_n17/area_resize_910b.cpp`). The
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

That compiles `csrc/npu/kernels/groot_n17/` into three shared objects in
`flash_rt/npu/lib`, each overridable on its own:

| shared object | override |
|---|---|
| `libflashrt_npu_groot_n17_dit_attn.so` | `FLASHRT_NPU_GROOT_N17_DIT_ATTENTION_LIBRARY` |
| `libflashrt_npu_groot_n17_dit_norm.so` | `FLASHRT_NPU_GROOT_N17_DIT_NORM_LIBRARY` |
| `libflashrt_npu_groot_n17_image.so` | `FLASHRT_NPU_GROOT_N17_IMAGE_LIBRARY` |

`FLASHRT_ENABLE_NPU_PI05=OFF` leaves the Pi0.5 units out; at least one model has
to be selected, and a switch that is neither `ON` nor `OFF` is refused by name.

**A deselected model's libraries are removed, not merely skipped.** Building with
this model on and then building again with the default selection would otherwise
leave its three behind, so the directory would disagree with the selection that
produced it. Only the standard names are touched; an override lives wherever the
caller put it.

## Tests

No device needed — these run on any host:

```bash
python -m pytest tests/test_npu_*.py -q
```

They cover the frontend's routing and every argument it refuses, what a build
selection compiles and leaves behind (driven against a fake toolchain, so the
script carries no test hook), the operand checks that stand in front of every raw
pointer, each of the three units' identity checks and the refusal a missing build
gives, the action decode and state encode including every representation they
refuse, the weight specification, and — against OpenCV itself — the tap
construction that makes the image transform bit-exact.

On a 910B4 with the units built:

```bash
python -m pytest tests/test_npu_groot_n17_device.py -q
```

Three numerical smokes at the exact shapes the model runs: the attention against
the vendor operator at all three DiT geometries and the two value layouts against
each other bit-for-bit, the norm against a reference that rounds where it rounds,
and the whole image transform against OpenCV.
