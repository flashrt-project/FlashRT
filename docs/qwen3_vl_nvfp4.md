# Qwen3-VL-8B on RTX 5090 (NVFP4 language stack + BF16 ViT)

FlashRT multimodal inference path for
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
on a single RTX 5090 (sm_120, 32 GB): the dense-Qwen3 language stack runs
NVFP4 W4A4 (reusing the [Qwen3-8B path](qwen3_8b_nvfp4.md) unchanged) and
the SigLIP-style ViT tower runs hand-controlled BF16 kernels.

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
| TTFT (prefill, image+text) | 206 ms | **121 ms** |
| Decode (warm CUDA Graph)   | ~48 tok/s | **143 tok/s** |

The TTFT is dominated by the ViT tower (~81 ms, FA2-bound at the
6256-patch full-attention) plus the NVFP4 language prefill (~39 ms).
Decode reuses the Qwen3-8B NVFP4 W4A4 + CUDA-Graph path and lands at the
same ~143 tok/s ceiling; the MRoPE position continues past the image
while the KV-cache slot advances from the image-compressed prompt length.

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

New kernel (general, in the separate `flash_rt_qwen3_vl_kernels` module):
`rope_neox_qk_bf16`, a rotate_half RoPE for Q/K not covered by the
interleaved `rope_apply` or the norm-fused Qwen3 RoPE kernels.

---

## 4. Correctness

Validated against the stock HF bf16 reference on an image+text prompt:

| Check | Result |
|---|---|
| ViT block (cumulative, layer 8) cosine | 0.9998 |
| DeepStack features cosine | 0.9999 / 0.9986 / 0.9967 |
| image_embeds (merger) cosine | 0.984 |
| Next-token argmax vs HF | match |
| End-to-end image description | correct |

The image_embeds 0.984 is a single massive-activation channel in the last
ViT block (a known bf16 sensitivity); it does not change the generated
token sequence.

---

## 5. Notes & limitations

- **lm_head stays BF16** (same as the Qwen3-8B path).
- **Image inputs.** Video and multi-image are not wired in this path.
- **Vision is BF16.** The ViT GEMMs (~52% of TTFT) are the next lever; an
  FP8 vision tower would roughly halve the GEMM time.
- **No KV-cache offload**; prompt + generation must fit in `max_seq`.
