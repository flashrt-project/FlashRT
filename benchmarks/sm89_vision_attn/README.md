# SM89 Vision-Attention FA2 Tile Micro-Bench

Standalone harness to pick the best FA2 forward tile for the Qwen3-VL
**vision tower** attention on Ada (sm_89), without paying the ~9-minute full
FA2 rebuild per experiment.

The vision tower runs **one non-causal full-attention call over all S=6256
patches per block** (bf16, 16 heads). Head-dim buckets:

- **2B vision: head_dim=64** → FA2 hd64 bucket (this branch adds it).
- **8B vision: head_dim=72** → padded to the hd96 bucket.

This harness links ONLY the FA2 forward template (no split-KV, no causal), so
each build compiles a small candidate tile table for one head-dim bucket in a
few minutes instead of the full matrix.

## Usage

```bash
./build.sh 64    # 2B vision (head_dim 64)
./build/bench_vision_attn_hdim64 --s 6256 --heads 16 --d 64

./build.sh 96    # 8B vision bucket (see faithfulness caveat below)
./build/bench_vision_attn_hdim96 --s 6256 --heads 16 --d 72
```

## Layout faithfulness caveat

The harness packs Q/K/V with `head_stride = BENCH_HDIM`. This is faithful only
when the model's real `head_dim == BENCH_HDIM`:

- **hd64 (2B, d=64): faithful.** Bench gives `128x64 = 1.185 ms` <
  `128x128 = 1.216` < `128x32 = 1.241`, and `1.185 × 24 layers = 28.4 ms`
  matches the end-to-end nsys vision-attention total (28.6 ms). Trustworthy.
- **hd96 (8B, d=72): NOT faithful.** The real model packs `head_stride=72`
  (hidden=16×72), not 96, so this harness's hd96 tile ranking disagrees with
  the faithful end-to-end nsys numbers (which show 128x64 > 128x32). For the
  hd96 tile choice, trust end-to-end, not this harness.

## Result (2026-06-26, RTX 4090)

hd64 vision regime (S=6256, 16 heads, d=64): **128x64 is the best tile**, so
`run_mha_fwd_hdim64` uses 128x64 for sm8x non-causal. This is the tile applied
in `flash_fwd_launch_template.h`.
