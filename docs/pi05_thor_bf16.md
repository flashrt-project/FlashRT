# Pi0.5 BF16 on Jetson AGX Thor

This opt-in PyTorch frontend runs a single RGB view and a text-only prompt on
SM110. It uses ten denoising steps, BF16 weights and activations where appropriate,
and FP32 boundaries for the OpenPI reference precision. The public output is a
`(10, 7)` LIBERO action chunk. Requests with state or additional views are rejected.

## Build and use

Follow the [CUDA build prerequisites](INSTALL.md). Install the cuDNN development
headers and libraries for your CUDA environment. Build the two BF16 targets
and use the editable Python install from the same checkout.

```bash
cmake -S . -B build -DGPU_ARCH=110 -DFLASHRT_ENABLE_PI05_THOR_BF16=ON
cmake --build build --target pi05_bf16_native_ext pi05_bf16_fa2_ext -j2
```

Prepare the OpenPI `pi05_libero` PyTorch checkpoint and tokenizer as described in
the [Thor examples](../examples/thor/README.md). Keep the checkpoint's own action
statistics at `assets/physical-intelligence/libero/norm_stats.json`.

```python
import flash_rt

model = flash_rt.load_model(
    checkpoint, config="pi05", framework="torch", hardware="thor",
    precision="bf16", num_views=1, use_fp8=False,
)
actions = model.predict(images=[rgb_uint8], prompt="pick up the black bowl and place it on the plate")
```

Prompt binding prepares a CUDA Graph. Reusing the same prompt replays the
graph with the current image and initial noise. A fixed noise array can be
passed as `{"image": rgb_uint8, "noise": fp32_noise}` with shape `(10, 32)`.

## Reproduce the single-view check

Use the same checkpoint in all three runs. The `reference` mode requires
OpenPI with its PyTorch backend. Use `--warmup 200 --iters 1000` for a longer
steady-state run. The script uses the primary camera of the public Thor fixture
with the other views and state omitted on both paths.

```bash
python examples/thor/pi05_bf16.py --checkpoint "$CKPT" --mode reference --output reference.npz
python examples/thor/pi05_bf16.py --checkpoint "$CKPT" --mode eager --reference reference.npz --output eager.npz
python examples/thor/pi05_bf16.py --checkpoint "$CKPT" --mode graph --reference reference.npz --output graph.npz
```

The reports compare raw and decoded actions for the same image, prompt and
FP32 initial noise. The timing includes input preparation and decoded output,
with model loading, prompt binding, and graph capture outside the timed loop.

The vision attention wrapper uses FlashRT's vendored FlashAttention 2. The
native reductions reference PyTorch's CUDA implementation at commit
[`5aff3928d8`](https://github.com/pytorch/pytorch/tree/5aff3928d8/aten/src/ATen/native/cuda).
See the [PyTorch license](../csrc/kernels/pi05/thor/LICENSE-pytorch).
