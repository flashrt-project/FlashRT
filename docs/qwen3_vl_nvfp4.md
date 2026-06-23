# Qwen3-VL-8B on RTX 5090 (NVFP4 language stack + FP8 ViT)

FlashRT multimodal inference path for
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
on a single RTX 5090 (sm_120, 32 GB): the dense-Qwen3 language stack runs
NVFP4 W4A4 (reusing the [Qwen3-8B path](qwen3_8b_nvfp4.md) unchanged) and
the SigLIP-style ViT tower runs hand-controlled kernels (FP8 block-128
W8A8 GEMMs, BF16 attention).

For the framework intro see [`../README.md`](../README.md); for the
text-only Qwen3-8B path see [`qwen3_8b_nvfp4.md`](qwen3_8b_nvfp4.md).

---

## 1. Headline performance

```
RTX 5090 / sm_120 / 32 GB · NVFP4 W4A4 language stack · BF16 ViT tower
image + text prompt, 1581 LLM tokens (1564 vision + 17 text), FlashRT.png
```

| Metric | HF SDPA (bf16) | FlashRT |
|---|---:|---:|
| TTFT (full request, image+text) | 206 ms | **~100 ms** |
| Decode (warm CUDA Graph)   | ~48 tok/s | **~150 tok/s** |

Same 5090, FlashRT.png, full resolution (1581 prompt tokens). FlashRT TTFT
is GPU image preprocessing (~8 ms) + the **CUDA-graph prefill** (~93 ms);
decode is the NVFP4 W4A4 + graph path. Both prefill and decode replay as
pure-kernel CUDA graphs (no per-request Python/torch dispatch), so latency
is low and stable.

The prefill graph captures the whole single-image path (embed → ViT tower →
image-feature scatter → NVFP4 layers + DeepStack → final norm) into one
graph; lm_head runs eager on the last row. Multi-image / video fall back to
the eager prefill. Inside the graph the costs are the ViT full-attention
over 6256 patches (~31 ms, ~75 % bf16 roofline at head_dim 72), the FP8 +
NVFP4 GEMMs, and the attention; all compute-bound.

Prefill history on this path (5090, prefill-only ms): 621 → 121 (ViT FFN on
the fast GEMM) → 102 (FP8 block-128 ViT GEMMs) → 99 (FP8 quant fused into
LayerNorm/GELU) → 95 (merger FP8 + whole-prefill CUDA graph) → **93** (ViT
bias adds fused into adjacent kernels). Image preprocessing moved CPU → GPU
(24 → 8 ms), so the full request TTFT is ~100 ms.

### TTFT vs resolution (the dominant knob)

The ~100 ms above is the **full-resolution** benchmark: the 2172×724 image
patchifies to 6256 patches → 1564 of the 1581 LLM tokens are vision. The
patch count (≈ pixels / 16²) sets both the ViT cost and the LLM prefill
length, so capping resolution cuts both at once. Pass ``max_pixels`` (the
processor's smart_resize rounds to the patch grid); the description is
unchanged across this range (full-request TTFT = GPU preprocessing + graph
prefill):

| `max_pixels` | patches | LLM tokens | TTFT |
|---|---:|---:|---:|
| none (full) | 6256 | 1581 | 100 ms |
| 1.0 M | 3888 | 989 | 59 ms |
| 0.5 M | 1824 | 473 | **32 ms** |
| 0.25 M | 972 | 260 | 25 ms |

```python
fe = Qwen3VlTorchFrontendRtx(ckpt, max_pixels=1_000_000)  # or --max-pixels
```

Reproduce:

```bash
python examples/qwen3_vl_quickstart.py \
    --checkpoint /path/to/Qwen3-VL-8B-FlashRT-NVFP4 \
    --image FlashRT.png --benchmark 20
```

---

## 2. Quick start

```bash
# 1. Build the FlashRT checkpoint once (NVFP4 language linears +
#    BF16 vision tower + combined config), from a stock BF16 ckpt.
python tools/quantize_qwen3_vl_nvfp4.py \
    --src /path/to/Qwen3-VL-8B-Instruct \
    --dst /path/to/Qwen3-VL-8B-FlashRT-NVFP4

# 2. Describe an image.
python examples/qwen3_vl_quickstart.py \
    --checkpoint /path/to/Qwen3-VL-8B-FlashRT-NVFP4 \
    --image FlashRT.png \
    --prompt "Describe this image in one sentence."
```

```python
from PIL import Image
from flash_rt.frontends.torch.qwen3_vl_rtx import Qwen3VlTorchFrontendRtx

fe = Qwen3VlTorchFrontendRtx('/path/to/Qwen3-VL-8B-FlashRT-NVFP4')
messages = [{'role': 'user', 'content': [
    {'type': 'image', 'image': Image.open('FlashRT.png').convert('RGB')},
    {'type': 'text', 'text': 'Describe this image in one sentence.'},
]}]
print(fe.generate(messages, max_new_tokens=128))
```

The build needs the `flash_rt_qwen3_vl_kernels` module
(`cmake -B build -S . -DGPU_ARCH=120 -DFLASHRT_BUILD_QWEN3_VL=ON` then
build that target); the shared `flash_rt_kernels.so` is unchanged.

### Inputs (image / multi-image / video)

`messages` follows the processor's chat format. Mix any number of images
and/or one video; the frontend builds the per-segment geometry and scatter
automatically.

```python
# Multiple images
content = [
    {'type': 'image', 'image': img_a},
    {'type': 'image', 'image': img_b},
    {'type': 'text', 'text': 'Compare the two images.'},
]

# Video (numpy (T,H,W,C) array or a list of PIL frames; frame sampling
# follows the processor defaults)
content = [
    {'type': 'video', 'video': frames},
    {'type': 'text', 'text': 'What happens in this video?'},
]
```

### Resolution (`max_pixels`) — explicit, opt-in

`max_pixels` is **not** set by default (`None` = the checkpoint's full
resolution, so behaviour is unchanged and the output is bit-identical).
Because the patch count drives both the ViT and the LLM prefill length
(see §1), set it explicitly to trade visual detail for latency — it is the
single biggest TTFT lever:

```python
fe = Qwen3VlTorchFrontendRtx(ckpt, max_pixels=1_000_000)   # ~57 ms
fe = Qwen3VlTorchFrontendRtx(ckpt, max_pixels=500_000)     # ~30 ms
```

The same knob is exposed as `--max-pixels` on
`examples/qwen3_vl_quickstart.py` and `examples/qwen3_vl_openai_server.py`.
It applies the processor's smart_resize (rounds to the patch grid) to both
images and videos.

---

## 3. Inference architecture

```
  image pixels (patchified)
    └─ ViT tower (BF16, 27 blocks, hidden 1152, 16 heads × head_dim 72)
         patch_embed → +interpolated pos_embed
         per block: LayerNorm → qkv → 2D rotate_half RoPE → FA2 (full,
                    bidirectional) → proj → residual → LayerNorm →
                    fc1 → GELU(tanh) → fc2 → residual
         DeepStack taps at layers 8 / 16 / 24
         2×2 patch merger → image_embeds (out_hidden 4096)
    └─ scatter image_embeds into the embedding stream at the image span
  text tokens → embed_tokens (BF16)
    └─ language stack (NVFP4 W4A4, 36 layers, reused from Qwen3-8B)
         interleaved-MRoPE; DeepStack added at the first 3 layers
         final RMSNorm → lm_head (BF16)
  ↓
  logits → greedy decode (CUDA Graph replay)
```

Per-prompt geometry (3D MRoPE position ids, MRoPE / vision-RoPE tables,
the interpolated vision position embedding) is precomputed once in
`set_prompt` (with the image processor running on the GPU), which also
stages the captured-prefill static input buffers. The single-image prefill
then replays as one CUDA graph — pure kernel, no torch dispatch — and the
eager paths (`prefill`, multi-image / video) run only kernels. The vision
tower's
intermediate (4304) is zero-padded to 4352 at load so every FFN GEMM uses
the fast `w16a16` kernel (the unpadded fc2 fell back to an M=1-tuned
matmul that dominated the tower at ~19 ms/call).

New kernels (general, in the separate `flash_rt_qwen3_vl_kernels` module):
`rope_neox_qk_bf16`, a rotate_half RoPE for Q/K not covered by the
interleaved `rope_apply` or the norm-fused Qwen3 RoPE kernels; and
`layer_norm_to_fp8_block128_bf16` / `gelu_tanh_to_fp8_block128_bf16`,
which fuse the per-token / per-128-K-block FP8 activation quant into the
producing LayerNorm / GELU so the bf16 activation never round-trips to
HBM before the FP8 GEMM (≈2× over the unfused norm/gelu + quant chain).

---

## 4. Correctness

Validated against the stock HF bf16 reference on an image+text prompt:

| Check | Result |
|---|---|
| ViT block (cumulative, layer 8) cosine | 0.9998 |
| DeepStack features cosine | 0.9999 / 0.9986 / 0.9967 |
| image_embeds (merger) cosine | 0.984 bf16 / 0.971 FP8 |
| Next-token argmax vs HF | match |
| End-to-end image description | correct |

A single massive-activation channel in the last ViT blocks makes
image_embeds sensitive (a known bf16 effect, amplified by FP8): patch
embed, the mergers and the first 3 blocks stay bf16 to hold the cosine at
0.971. It does not change the generated token sequence (argmax matches HF,
the description is unchanged).

---

## 5. Notes & limitations

- **lm_head stays BF16** (same as the Qwen3-8B path).
- **Multi-image and video** are supported. The ViT runs once per vision
  segment — each image, or each video frame-group, is an independent
  attention window, reproducing HF's per-window cu_seqlens attention — and
  the features scatter into each segment's token span with per-segment
  DeepStack injection. Video uses Qwen3-VL's timestamp-aligned MRoPE (the
  grid is split per frame so each frame's temporal index is 0 and the
  inter-frame timestamp text tokens carry the temporal position). Frame
  sampling follows the processor defaults.
- **Compute core is at the kernel floor for this shape** (measured): the
  ViT attention (bf16 FA2, head_dim 72) ties the best community kernels
  (sm_120 has no FP8/INT8 attention that holds the accuracy — INT8-QK
  collapses image_embeds cosine), and the NVFP4 language prefill GEMM is
  already faster than an FP8 block-128 path at these shapes. Further TTFT
  comes from capping `max_pixels` (above) or an algorithm change (fewer
  patches / windowed attention) that would alter outputs.
- **No KV-cache offload**; prompt + generation must fit in `max_seq`.
