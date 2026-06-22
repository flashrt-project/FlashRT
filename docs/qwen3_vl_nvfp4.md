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
| TTFT (prefill, image+text) | 206 ms | **99 ms** |
| Decode (warm CUDA Graph)   | ~48 tok/s | **143 tok/s** |

TTFT is dominated by the ViT tower (~60 ms) plus the NVFP4 language
prefill (~39 ms). After the FP8 GEMMs, the two remaining ViT costs are
the full-attention over 6256 patches (~31 ms, ~75% of bf16 roofline at
head_dim 72) and the FP8 GEMMs (~20 ms); both are compute-bound. Decode
reuses the Qwen3-8B NVFP4 W4A4 + CUDA-Graph path and lands at the same
~143 tok/s ceiling; the MRoPE position continues past the image while the
KV-cache slot advances from the image-compressed prompt length.

TTFT history on this path (5090): 621 ms (initial eager) → 121 ms (ViT
FFN on the fast GEMM + bf16) → 102 ms (FP8 block-128 ViT GEMMs) → **99 ms**
(FP8 activation quant fused into the producing LayerNorm / GELU).

### TTFT vs resolution (the dominant knob)

The 99 ms above is the **full-resolution** benchmark: the 2172×724 image
patchifies to 6256 patches → 1564 of the 1581 LLM tokens are vision. The
patch count (≈ pixels / 16²) sets both the ViT cost and the LLM prefill
length, so capping resolution cuts both at once. Pass ``max_pixels`` (the
processor's smart_resize rounds to the patch grid); the description is
unchanged across this range:

| `max_pixels` | patches | LLM tokens | TTFT |
|---|---:|---:|---:|
| none (full) | 6256 | 1581 | 99 ms |
| 1.0 M | 3888 | 989 | 57 ms |
| 0.5 M | 1824 | 473 | **30 ms** |
| 0.25 M | 972 | 260 | 22 ms |

```python
fe = Qwen3VlTorchFrontendRtx(ckpt, max_pixels=1_000_000)  # or pass --max-pixels
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
`set_prompt`; the forward runs only kernels. The vision tower's
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
- **Next TTFT levers** (both need new kernels): an FP8 ViT attention
  (Sage/FA on sm_120 only ships head_dim 128/256, so head_dim 72 would
  need padding), and an FP8 language prefill (the stack is NVFP4, whose
  large-M GEMM dequantizes to bf16 compute).
- **No KV-cache offload**; prompt + generation must fit in `max_seq`.
