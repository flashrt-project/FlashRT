# Pi0.5 SM87 gated INT8 GEMM tile

The optional 64-row CTA tile applies only to the measured decoder up-projection
shape (M,N,K)=(15,4096,1024). Other shapes retain the 128-row tile. The INT8
inputs, INT32 accumulation, FP32 scale multiplication order, SiLU gate and
BF16 store are unchanged. The existing binding signature is unchanged.

Build in an editable checkout with CUTLASS v4.4.2:

```bash
cmake -S . -B build -DGPU_ARCH=87 -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DFA2_ARCH_NATIVE_ONLY=ON -DFA2_HDIMS="96;128;256" -DFA2_DTYPES=bf16 \
  -DFLASHRT_ENABLE_PI05_GATED_TILE_SM87=ON
cmake --build build -j4 --target flash_rt_kernels flash_rt_fa2
python -c "from flash_rt import flash_rt_kernels as k; print(k.__file__)"
python -m pytest tests/test_pi05_sm87_gated_tile.py -q
```

The option defaults OFF. Enabling it on another architecture or without the
SM80-family INT8 build is a configuration error.

## Preliminary measurements

On Jetson AGX Orin 64GB, CUDA 12.6, PyTorch 2.9.1, five independent-process
comparisons on the prior pinned producer measured a graph paired-median saving
of 0.930 ms (0.666%), with a session/cycle bootstrap interval of
[0.569,1.180] ms. Raw and decoded actions were bit-identical for the tested
fixtures. Workload: two cameras, 27 visual layers, pool=1, 15 action tokens,
10 denoising steps, cache_frames=1; eight observations and seeds 7/19.
These are historical experiment results, not a performance claim for a newly
built checkout. Report fresh build/model checks separately in the PR.

The paired full-request results had a nonzero negative-control median, so
the graph measurement is the primary repeatability evidence. No multi-day,
other-shape or other-hardware speedup is claimed.

For a reproducible full-model numerical comparison, save a reference with the
option OFF, rebuild with it ON, then compare the same synthetic RGB/noise:

```bash
python benchmarks/pi05_sm87_validate.py --checkpoint /path/to/checkpoint \
  --save-reference /tmp/pi05_gated_reference.npz
# Reconfigure/rebuild with FLASHRT_ENABLE_PI05_GATED_TILE_SM87=ON.
python benchmarks/pi05_sm87_validate.py --checkpoint /path/to/checkpoint \
  --reference /tmp/pi05_gated_reference.npz
```

The script freezes BF16 heuristic selection across processes to isolate the
INT8 tile change. Its timing is a smoke measurement with synthetic inputs,
not the historical DROID latency experiment above.
