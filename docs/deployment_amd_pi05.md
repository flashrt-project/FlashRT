# Pi0.5 on AMD Instinct (MI300X / CDNA3 and MI350X / CDNA4)

Model-specific guide. For hardware support, the build, routing, shared
environment knobs and the test suite, see
[deployment_amd.md](deployment_amd.md) first.

## Performance

The two architectures use different FP8 formats and kernel routes, so their
results are reported separately. MI300X uses E4M3 FNUZ; MI350X uses OCP
E4M3. Both use architecture-specific packed MFMA decoder kernels where they
beat hipBLASLt.

Pi0.5, 2 camera views, 10-step denoise, median end-to-end latency on
real observation frames:

### MI350X (gfx950)

| Configuration | Median latency | vs torch.compile |
|---|---|---|
| PyTorch eager | 116.9 ms | — |
| torch.compile (max-autotune) | 40.5 ms | 1.00× |
| **FlashRT AMD, FP8 (default)** | **16.4 ms** | **2.47×** |
| FlashRT AMD, BF16 (`use_fp8=False`) | ~30 ms | 1.34× |

For context, the same protocol on an RTX 5090 (FlashRT FP8, CUDA)
measures 19.7 ms.

### MI300X (gfx942)

Measured on this RunPod with ROCm 7.2.4 and AITER, using real frames from
LIBERO Spatial episode 0, 50 warmup replays and 100 measured calls:

| Configuration | Min | Median | p95 | Max |
|---|---:|---:|---:|---:|
| FlashRT AMD, FP8 FNUZ, packed decoder | 41.42 ms | 43.99 ms | 44.20 ms | 52.65 ms |
| FlashRT AMD, BF16, packed decoder + fused boundaries | 34.97 ms | 35.14 ms | 35.43 ms | 50.15 ms |

The timed boundary is the public frontend call: real uint8 camera frames and
pinned denoising noise enter `infer`, and the host action array has returned.
Prompt processing, calibration and graph capture are outside the timed region.
For this gfx942 workload FP8 is an accuracy/footprint tier; it is not a latency
win over BF16. The BF16 result uses gfx942 packed MFMA for decoder QKV, output,
and merged gate/up projections, plus a BF16 boundary kernel that fuses the
gated residual update with the following adaptive RMS normalization. The down
projection remains on hipBLASLt because it is faster at `(M,N,K)=(10,1024,4096)`.
With the same pinned noise, FP8 versus BF16 action cosine was **0.999979** and
the maximum absolute action difference was **0.01562**.

The process was bound to CPUs 48–63, a compact subset of the measured GPU's
NUMA node. Without affinity, container scheduling introduced large gaps
between graph launches.

These numbers are measured with aiter available. Without it the
attention sites fall back to torch SDPA and end-to-end latency rises by
roughly 6 ms — see [deployment_amd.md](deployment_amd.md#build).

Accuracy: output cosine against the FP32 reference sits in a
**0.9992–0.9994** band across processes with the denoise noise pinned
(see [Judging accuracy](#judging-accuracy)). Graph replay inside a
process is bit-identical.

## Quick start

```bash
python examples/pi05_amd_quickstart.py --checkpoint <pi05_checkpoint_dir>
```

Flags: `--num-views {2,3}` (default 2), `--bf16` for the unquantized
tier, `--iters` for the timing loop length. On gfx950 expect roughly
16–17 ms median FP8 after warmup. On the measured gfx942 system FP8 processes
measured about 44 ms median and the optimized BF16 tier measured 35.14 ms.

Library use:

```python
import flash_rt

model = flash_rt.load_model(checkpoint_dir, config="pi05",
                            framework="torch", num_views=2, use_fp8=True)
fe = model.pipeline
fe.set_prompt(prompt_text, state=state_vector)
fe.calibrate(observation)              # one-time, real data
result = fe.infer(observation)         # returns {"actions": ...}
```

## Precision tiers

- **FP8 (default)** — activations calibrated once from real data
  (`calibrate(...)`, percentile 99.9 by default, the same contract as
  [calibration.md](calibration.md)); weights quantized at load. The frontend
  selects FNUZ/240 on CDNA3 and OCP E4M3/448 on CDNA4 from extension metadata.
- **BF16** (`use_fp8=False`, or `FVK_PI05_AMD_FORCE_BF16=1`) — the
  unquantized baseline. Calling `calibrate` still prepares buffers and captures
  the runtime graph, but it collects no activation statistics in this tier.
- **FP4 (MXFP4, CDNA4 only)** — the GEMM kernels are unlocked and parity-verified
  (`GemmRunner.mxfp4_nt_dev`, hipBLASLt `HIP_R_4F_E2M1` with UE8M0 vec32
  block scales), but there is deliberately **no end-to-end FP4 tier**:
  on the current ROCm stack no available FP4 path beats FP8 at these
  shapes, so a user-facing knob would only select a slower path.
  `load_model(use_fp4=True)` logs a fallback to FP8;
  `use_fp4_decoder=True` remains Thor-only and raises.

Packed FNUZ is the default FP8 decoder route on gfx942. Packed BF16 is the
default for the gfx942 BF16 decoder. `FVK_AMD_CDNA3_BF16_SMALLM=0` restores
the BF16 hipBLASLt route; `FVK_AMD_DEC_GEMM=hipblaslt` restores the FP8
library route.

## Observation contract

The frontend validates observations strictly — a missing or wrong-shaped
view raises instead of leaving stale image buffers in the captured graph
inputs.

- `num_views` must be **2** (base + wrist camera, the LIBERO deployment)
  or **3** (+ right wrist camera).
- Provide either `observation["images"]` with exactly `num_views`
  entries, or the named keys `image`, `wrist_image`, `wrist_image_right`
  (the first `num_views` of them; supplying `wrist_image_right` with
  `num_views=2` is rejected as a view-count mismatch).
- Every image must be a `uint8` array of shape `(224, 224, 3)`. The
  normalization path is defined on the 0–255 range.

## Prompt-length strategies

Pi0.5 renders robot state into the prompt, so token length drifts with
the state values. Both strategies of the RTX frontend are supported with
the same semantics:

- `state_prompt_mode="exact"` (default) — one captured graph per exact
  prompt length, cached. Pair with `warm_state_prompt_buckets(...)` to
  front-load captures instead of paying them mid-episode.
- `state_prompt_mode="fixed"` — ONE padded graph serves every length
  (masked prefix plus runtime `devpos`/`seqused` K/V append), so no
  capture ever happens mid-loop. Latency follows the **padded** length,
  so right-size the capacity to the deployment's real prompt+state
  length with `state_prompt_fixed_max_len=<tokens>` (env
  `FLASHRT_PI05_STATE_PROMPT_FIXED_MAX_LEN`; the default is the
  200-token ceiling, and a prompt exceeding the capacity raises rather
  than silently recapturing).

  In fixed mode the decoder runs the same hand-written split-KV kernel
  as exact mode (seqused pointer, fused FP8-out epilogue included) and
  the encoder runs the seqused variant of the MFMA flash kernel. At a
  right-sized capacity the premium over exact mode is well under 1 ms
  and accuracy holds (cosine 0.9993, on par with exact).

## Environment knobs

Shared knobs are in [deployment_amd.md](deployment_amd.md). Pi0.5-specific:

| Env | Default | Meaning |
|---|---|---|
| `FVK_AMD_DEC_ATTN` | `custom` | decoder cross-attention: `custom` (hand-written split-KV flash, fastest) or the backend default |
| `FVK_AMD_ATTN_FP8OUT` | `1` | fuse the decoder attention output's FP8 quantize into the attention epilogue (bit-identical to the standalone quantize) |
| `FVK_AMD_FIXED_ENC_ATTN` | `flash` | fixed-mode encoder attention: `flash` (MFMA flash kernel, seqused pointer) or `sdpa` (masked torch fallback) |
| `FVK_AMD_CALIB_DET_ATTN` | `flash` | route the encoder site through the deterministic MFMA flash kernel during FP8 calibration so collected scales are run-to-run stable; `off` calibrates on the library path |
| `FVK_AMD_DEC_GEMM` | `mfma` | decoder small-M GEMMs: packed native-format MFMA on measured shapes (FNUZ on gfx942, OCP on gfx950); `hipblaslt` disables it |
| `FVK_AMD_CDNA3_BF16_SMALLM` | `1` | use packed gfx942 BF16 for decoder QKV, O and merged gate/up; `0` restores hipBLASLt |
| `FVK_AMD_CDNA3_BF16_FUSION` | `1` | fuse decoder gated-residual + following AdaRMSNorm boundaries in the BF16 tier; `0` restores the two-kernel chain |
| `FVK_PI05_AMD_FORCE_BF16` | `0` | force the BF16 baseline regardless of `use_fp8` |
| `FLASHRT_PI05_STATE_PROMPT_MODE` | — | overrides the `state_prompt_mode` constructor argument |
| `FLASHRT_PI05_STATE_PROMPT_FIXED_MAX_LEN` | `200` | fixed-mode padded capacity in tokens |

`FRT_ATTN_NSPLIT` / `FRT_ATTN_FUSED` / `FRT_ATTN_REDUCE_ALT` are
attention micro-benchmark knobs for A/B sweeps; leave them unset in
production.

## Feature matrix versus the RTX frontend

| Surface | AMD |
|---|---|
| `set_prompt` / `warm_state_prompt_buckets` | ✅ |
| `calibrate` / `calibrate_with_real_data` (single and multi-frame percentile) | ✅ |
| `infer` / `precision_spec` / `get_latency_stats` | ✅ |
| `state_prompt_mode` `"exact"` and `"fixed"` (devpos/seqused) | ✅ |
| Temporal K/V caching (`cache_frames`), vision pooling/truncation knobs | ✅ |
| `infer(noise=...)` pinned-noise judging | ✅ |
| `set_rl_mode` (advantage-conditioned RL) | ❌ raises `NotImplementedError` |
| Batched serving (`set_batched_mode`, `infer_batch`) | ❌ raises `NotImplementedError` |
| End-to-end FP4 tier | ❌ deliberately not exposed (see Precision tiers) |

## Judging accuracy

The denoise trajectory is conditioned on its starting noise, so a fresh
random draw moves the output cosine by about 1e-3 — the same magnitude
as a real numerical regression.

```python
noise = numpy.random.default_rng(0).standard_normal((10, 32)).astype("float32")
actions = fe.infer(observation, noise=noise)["actions"]
```

Pin the noise to the exact array the reference was generated with and
record the seed (or a hash of the array) alongside the reference. With
pinned noise the FP8 band is 0.9992–0.9994 across processes; the
residual spread comes from library-attention nondeterminism and timed
algorithm picks. Serving should keep the default random draw — pinning
is a judging protocol, not a deployment setting.

## Reproducing the numbers

1. Build for the installed architecture with ROCm 7.x and a ROCm PyTorch
   build (`bash scripts/amd/build_amd.sh gfx942` for MI300X).
2. On a multi-NUMA host, find the selected GPU's `numa_node` through its PCI
   device in `/sys/class/drm`, then bind to a compact subset of that node's
   `local_cpulist`. For this pod GPU 0 is on node 1 and the command was:

   ```bash
   taskset -c 48-63 python benchmarks/pi05_amd_latency.py \
     --ckpt <pi05_checkpoint_dir> \
     --data <real_libero_episode.npz> \
     --warmup 50 --iters 100 --json-out pi05-fp8.json
   ```

   Add `--bf16` for the aligned BF16 control. Do not copy `48-63` to a
   different host without checking its topology.
3. For a judged comparison, run the identical loop on the CUDA build
   (RTX frontend). The protocols are the same: 50-iteration median after
   5 warmup replays, real image observations, `calibrate` before timing.
4. For the model-level test gates, point the suite at a checkpoint:

   ```bash
   export FLASH_RT_PI05_AMD_CKPT=<pi05_checkpoint_dir>
   # optional: a saved reference-actions .npy for the cosine gate
   export FLASH_RT_PI05_AMD_REF_ACTIONS=<reference_actions.npy>
   python -m pytest tests/test_amd_pi05_model.py -v
   ```

Cross-run medians can move with host scheduling and hipBLASLt algorithm
selection; compare arms under the same affinity and data protocol.
