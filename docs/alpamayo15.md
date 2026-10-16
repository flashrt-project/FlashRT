# Alpamayo 1.5 on Jetson AGX Thor

This guide covers original NVIDIA full-request inference and FlashDrive streaming
inference, each with its reference execution and optional FlashRT backend.
The two algorithms are independently selectable; their speedups do not multiply.

## Execution contracts

| Arm | Weights | Procedure | Execution |
|---|---|---|---|
| official | NVIDIA Alpamayo | full 16-frame request, 10 expert calls | official SDPA |
| flashrt_full | same NVIDIA weights | same full-request procedure | FlashRT BF16; optional FP8 decode |
| flashdrive | z-lab base, PARO, DFlash | streaming KV reuse, speculative reasoning, W4A8 and velocity caching | author implementation |
| fd_flashrt | same z-lab weights | same FlashDrive procedure | FlashRT primitive backends |

Streaming supports one trajectory sample; the full-request frontend supports
multiple samples. The first streaming window returns no trajectory. Reset between
unrelated streams. Compare steady-state speed only within each matched pair.

## Platform and pinned dependencies

Validated: Jetson AGX Thor SM110, 20 SMs, NVIDIA PyTorch 26.05, Python 3.12.3,
Torch 2.12.0a0+5aff3928d8.nv26.05, CUDA 13.2, Transformers 4.57.1,
physical-ai-av 0.2.0 and PyAV 19.0.1.

| Source | Full-request environment | Streaming environment |
|---|---|---|
| NVlabs/alpamayo1.5 | 36aeb4c | a5bd40c |
| NVlabs/alpamayo | not required | 4cda35d |
| z-lab/flashdrive | not required | 2626517 |
| vLLM | not required | v0.16.0, source-built SM110 W4A8 Marlin |
| ParoQuant | not required | 0.1.15 |

Upstream generic Torch/Triton pins differ from the Thor platform build. Retain
NVIDIA Torch, torchvision and Triton; do not run an unrestricted upstream uv sync
or install an incompatible CUDA wheel. These model dependencies remain optional
and are not imported by the default FlashRT package.

## Isolated installation

Run inside a compatible ARM64 NVIDIA GPU environment with the CUDA toolkit.
Start in the FlashRT checkout. All directories below are relative and user-chosen.
For Docker on Jetson, configure the NVIDIA runtime for the platform.

```bash
python -m venv --system-site-packages .venv-official
python -m venv --system-site-packages .venv-flashdrive
mkdir -p dependencies
git clone https://github.com/NVlabs/alpamayo1.5 dependencies/alpamayo-official
git -C dependencies/alpamayo-official checkout 36aeb4c
git clone https://github.com/NVlabs/alpamayo1.5 dependencies/alpamayo-streaming
git -C dependencies/alpamayo-streaming checkout a5bd40c
git clone https://github.com/NVlabs/alpamayo dependencies/alpamayo-r1
git -C dependencies/alpamayo-r1 checkout 4cda35d
git clone https://github.com/z-lab/flashdrive dependencies/flashdrive
git -C dependencies/flashdrive checkout 2626517
git clone --branch v0.16.0 --depth 1 https://github.com/vllm-project/vllm dependencies/vllm
git clone --branch v4.4.2 --depth 1 https://github.com/NVIDIA/cutlass third_party/cutlass
```

Install explicit inference dependencies, retaining platform GPU packages:

```bash
for env_dir in .venv-official .venv-flashdrive; do
  "$env_dir/bin/python" -m pip install accelerate==1.15.0 av==19.0.1 einops==0.8.2 \
    hydra-colorlog==1.2.0 hydra-core==1.3.7 pandas==3.0.2 physical-ai-av==0.2.0 \
    pillow==12.2.0 transformers==4.57.1 huggingface_hub==0.36.2 tokenizers==0.22.2 \
    safetensors==0.7.0 matplotlib seaborn pytest pybind11 cmake ninja uv_build setuptools wheel
  "$env_dir/bin/python" -m pip install --no-deps -e .
done
.venv-official/bin/python -m pip install --no-deps dependencies/alpamayo-official
.venv-flashdrive/bin/python -m pip install --no-deps dependencies/alpamayo-streaming \
  dependencies/alpamayo-r1 dependencies/flashdrive paroquant==0.1.15
.venv-flashdrive/bin/python -m pip install cbor2 gguf rich
VLLM_TARGET_DEVICE=empty .venv-flashdrive/bin/python -m pip install \
  --no-deps --no-build-isolation dependencies/vllm
source .venv-flashdrive/bin/activate
PYTHON=python bash scripts/build_alpamayo15_marlin.sh dependencies/vllm build-marlin
```

The Marlin recipe requires a fresh build directory and builds only the original
vLLM W4A8 operator schema and compatible SM110 kernels. Other vLLM CUDA operations
are unavailable. Do not reuse a prebuilt extension from another GPU/Torch ABI.

Generate platform constraints before installing the FA4 runtime:

```bash
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
Path('dependencies/platform-constraints.txt').write_text(
    ''.join(f'{name}=={version(name)}\n' for name in ('torch', 'torchvision', 'triton')))
PY
python -m pip install -c dependencies/platform-constraints.txt apache-tvm-ffi==0.1.14.post1 torch-c-dlpack-ext==0.1.5
python -m pip install -c dependencies/platform-constraints.txt -e '.[thor-fa4]'
export CUTE_DSL_ARCH=sm_101a FLASH_ATTENTION_ARCH=sm_100a
cmake -S . -B build-alpamayo -DGPU_ARCH=110 -DFLASHRT_ENABLE_ALPAMAYO15=ON \
  -DFLASHRT_ENABLE_COSMOS3_EDGE=ON -DFLASHRT_ENABLE_COSMOS3_REASONER=ON
cmake --build build-alpamayo --target flash_rt_kernels -j4
python -c 'import flash_rt.flash_rt_kernels'
python -c 'import vllm._C, torch; assert hasattr(torch.ops._C, "marlin_gemm")'
python -m pip freeze > dependencies/flashdrive-environment.lock.txt
```

Install the same constrained FA4 runtime in the official environment before
running `flashrt_full`:

```bash
.venv-official/bin/python -m pip install -c dependencies/platform-constraints.txt \
  apache-tvm-ffi==0.1.14.post1 torch-c-dlpack-ext==0.1.5 -e '.[thor-fa4]'
```
 The current engine also uses gated Cosmos Edge/Reasoner GEMM helpers; enable
those groups as shown. This is a build dependency, not a change of model.
The Alpamayo kernels are opt-in and Thor-only; unsupported architectures
raise errors rather than silently switching execution.

## Checkpoints and real input data

Authenticate with `hf auth login` for gated resources. Do not place credentials
in source or command examples. Obtain access to auxiliary processor repositories
referenced by the official configuration, or prepopulate a verified offline cache.
Do not silently substitute a different tokenizer or image processor.

| Checkpoint | Revision |
|---|---|
| nvidia/Alpamayo-1.5-10B | 7aba8293c09993f2e125c6819df05d7fa3e873ea |
| z-lab/Alpamayo-1.5-10B | 03637f1f3acac01d8bf5a094147d6352570e6c46 |
| z-lab/Alpamayo-1.5-10B-PARO | d029319a |
| z-lab/Alpamayo-1.5-10B-DFlash | 2496c456 |
| nvidia/PhysicalAI-Autonomous-Vehicles | 33f9bf447ed3bcb7d545ce13f4226f824214fafb |

Use a dedicated cache and resolve pinned snapshots before benchmarking:

```bash
export HF_HOME="$PWD/.hf-cache"
python - <<'CACHE_PY'
from pathlib import Path
from huggingface_hub import snapshot_download
pins = {
    "nvidia/Alpamayo-1.5-10B": "7aba8293c09993f2e125c6819df05d7fa3e873ea",
    "z-lab/Alpamayo-1.5-10B": "03637f1f3acac01d8bf5a094147d6352570e6c46",
    "z-lab/Alpamayo-1.5-10B-PARO": "d029319a",
    "z-lab/Alpamayo-1.5-10B-DFlash": "2496c456",
}
for repo, revision in pins.items():
    snapshot = Path(snapshot_download(repo_id=repo, revision=revision))
    refs = snapshot.parent.parent / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(snapshot.name)
CACHE_PY
```

Cache auxiliary files before enabling HF_HUB_OFFLINE=1 and
TRANSFORMERS_OFFLINE=1. Leave offline mode unset when preparing new dataset
windows. The cache references above are deliberately pinned; do not update them
between comparator runs.  Pin auxiliary files
and record their hashes. The author loader selects optimized behavior from a
`z-lab/` repo ID; use its pinned regular HF cache for offline execution. The
FlashRT frontend accepts explicit local directories for the three companions.

Prepare consecutive real windows using the official loader. It streams needed
features rather than fetching the full dataset. Video chunk transport can exceed
the decoded frame volume; a 120-window serialized pack may consume several GB.

```bash
.venv-official/bin/python benchmarks/alpamayo15_thor.py --prepare \
  --clip 87147a1b-3eef-4c25-94d2-ec7718a49a7a --start-us 1700000 \
  --stride-us 100000 --windows 120 --output data/example.pt
```

## Four-arm reproduction

Set CUTE_DSL_ARCH=sm_101a and FLASH_ATTENTION_ARCH=sm_100a in the benchmark
shell as well as the build shell when using the validated CUDA 13.2 platform.

Use identical packs, locked GPU/memory/CPU clocks and no competing workload.
Acquire the site's GPU scheduler lock for benchmarks, builds and GPU tests.
Only load serialized packs you created or trust.

```bash
.venv-official/bin/python benchmarks/alpamayo15_thor.py --arm official \
  --inputs data/example.pt --seed 0 --output results/official.json
.venv-official/bin/python benchmarks/alpamayo15_thor.py --arm flashrt_full \
  --inputs data/example.pt --seed 0 --output results/flashrt-full.json
.venv-flashdrive/bin/python benchmarks/alpamayo15_thor.py --arm flashdrive \
  --inputs data/example.pt --seed 0 --output results/flashdrive.json
.venv-flashdrive/bin/python benchmarks/alpamayo15_thor.py --arm fd_flashrt \
  --inputs data/example.pt --seed 0 --output results/combined.json
```

Repeat seeds 0,1,2 and independent clips. Keep windows in temporal order and
reset between streams. The runner excludes windows 0-2 from steady statistics
and retains all window timings and ADE/FDE. `--official-expert-graph` enables
the official graph baseline; label that result separately.

`request_ms` includes preprocessing, model execution and CPU trajectory return,
with synchronization. Loading, downloads and graph construction at model
initialization are separate `load_s` costs. Lazy compilation/capture appears in
the early window timings. Historical tables below report narrower model-only
latency and must not be presented as request-inclusive results from this runner.

All four commands use 128 reasoning tokens and one trajectory sample. The
NVIDIA standalone example uses 256 tokens; do not mix budgets. Historical
streaming evaluation used nine clips (120 windows on the example, 60 on eight
others), three seeds, 1719 steady windows per arm. Full-request comparison used
the same clips with seed 0. A single example validates setup, not broad model
quality or closed-loop driving safety.

## Tests

```bash
python -m pytest -q -rs tests/test_alpamayo15_thor.py tests/test_alpamayo15_flashdrive_thor.py
python benchmarks/alpamayo15_thor.py --help
bash -n scripts/build_alpamayo15_marlin.sh
```

For optional streaming checkpoint integration, set `FLASHRT_FLASHDRIVE_CKPT`
to a local directory containing `Alpamayo-1.5-10B`, `Alpamayo-1.5-10B-PARO`
and `Alpamayo-1.5-10B-DFlash`. Set `FLASHRT_TEST_ALPAMAYO15_REQUIRED=1`
to make missing required full-request kernels fail rather than skip.

GPU tests require a compatible build and optional model dependencies; checkpoint
integration tests require local authorized assets. Inspect skip reasons and do
not report skipped model integration as a passing hardware test. The new public
runner should be smoke-tested on all four arms before publishing new metrics.

## API contracts and historical evidence

The following sections consolidate the reviewed implementation's interfaces,
numerical contracts, measurements and limitations.


## Full-request execution

FlashRT runs NVIDIA [Alpamayo 1.5](https://github.com/NVlabs/alpamayo1.5)
(`nvidia/Alpamayo-1.5-10B`) on Jetson AGX Thor (`sm_110`) in BF16, with an
opt-in FP8 decode-weight tier (see [FP8 decode tier](#fp8-decode-tier)). The request
semantics of the official `sample_trajectories_from_data_with_vlm_rollout`
are preserved: the same multi-camera prompt, chain-of-causation reasoning with
the official sampler (trajectory tokens masked, temperature, top-p, multinomial
draw), the same stop rule, all 10 flow-matching steps of the diffusion expert,
and the official noise draw and trajectory integration.

## What runs where

| Stage | Official (Hugging Face eager) | FlashRT Thor |
|---|---|---|
| Vision tower (27 layers, 16 frames) | per-frame attention calls, FP32 LayerNorm/RoPE round trips, CPU-built position tables | batched FA4 attention on the interleaved q/k/v, BF16 LayerNorm with FP32 math, fused in-place FP32 RoPE, per-grid cached tables, patch embedding as one GEMM |
| LM prefill (36 layers) | eager modules, `DynamicCache` | fused q/k/v GEMM, fused head RMSNorm + M-RoPE writing K/V straight into a static cache, FA4 causal attention, fused residual + RMSNorm, SiLU-mul |
| Reasoning decode | `generate()` | one CUDA graph per token: fused GEMVs, split-KV FA4 decode attention, sampler on device |
| Diffusion expert (10 Euler steps) | 10 eager expert calls (or bounded per-length graphs) | all 10 steps in one CUDA graph; one graph serves every reasoning length |
| `num_traj_samples = N` | prompt (images included) repeated N times through vision and prefill | vision and prefill once; the prompt K/V is shared by all samples |

The static cache holds `[prompt | reasoning | action]` per sample. Each
sample's 64 action-token K/V rows are written right after its visible prefix,
so a single sample attends in one non-causal pass with a device-side length.
With several samples, attention is one pass over the shared prompt for all
samples plus a per-sample variable-length suffix pass, merged by their
log-sum-exp.

## Numerics

The model-local kernels (`csrc/kernels/alpamayo15_kernels.cu`) reproduce the
rounding sequence of the official module graph under
`torch.autocast(bfloat16)`:

- VLM: BF16 activations; Qwen3 RMSNorm rounds the normalized value, then the
  weight product; RoPE rounds each product and the sum to BF16.
- Expert: FP32 residual stream (its input comes from an FP32 LayerNorm under
  autocast), FP32 RoPE, BF16 operands at attention.
- Vision: RoPE in FP32 with one BF16 rounding.

`tests/test_alpamayo15_thor.py` checks each kernel against these PyTorch op
sequences (vision RoPE bit-exact, the others within one BF16 ulp from FP32
reduction order; the module is built with flush-to-zero). GEMV/GEMM and attention kernels keep FP32 accumulation but
differ from the eager kernels in reduction order, so near-tie sampling steps
can choose a different token, exactly as switching the official model between
its SDPA and FlashAttention-2 backends does. With several samples, the
shared-prompt and per-sample attention passes are merged from BF16 partial
outputs (one extra rounding) and the reasoning lm_head runs as a vocab-major
GEMM, so per-sample numerics are not bit-identical to a single-sample run.

## Build

```bash
cmake -B build -S . -DGPU_ARCH=110 -DFLASHRT_ENABLE_ALPAMAYO15=ON \
  -DFLASHRT_ENABLE_COSMOS3_EDGE=ON -DFLASHRT_ENABLE_COSMOS3_REASONER=ON
cmake --build build -j$(nproc) --target flash_rt_kernels
pip install -e ".[thor-fa4]"     # FA4 (CuTe DSL) runtime
```

`FLASHRT_ENABLE_ALPAMAYO15` is Thor-only and off by default; the bindings are
gated with the sources. On CUDA 13.2 containers, compile FA4 for Thor with
`CUTE_DSL_ARCH=sm_101a` (and `FLASH_ATTENTION_ARCH=sm_100a`); `sm_110a` is
rejected by that libNVVM.

The official `alpamayo1_5` package (and `transformers==4.57.1`,
`physical_ai_av` for its data loader) must be importable. The official
`pyproject.toml` pins `torch==2.8.0`; on Thor use the platform PyTorch build
(validated: NGC PyTorch 26.05, torch 2.12, CUDA 13.2).

## Usage

```python
from flash_rt.frontends.torch.alpamayo15_thor import Alpamayo15TorchFrontendThor

fe = Alpamayo15TorchFrontendThor("nvidia/Alpamayo-1.5-10B")   # loads BF16 + SDPA, captures graphs
out = fe.infer(image_frames,      # (n_cameras, n_frames, 3, H, W) uint8
               camera_indices,    # (n_cameras,)
               ego_history_xyz,   # (1, 1, 16, 3) ego frame at t0
               ego_history_rot,   # (1, 1, 16, 3, 3)
               seed=42)
out["pred_xyz"], out["pred_rot"], out["cot"][0]
# several trajectory samples: construct with max_samples=16, then
# fe.infer(..., num_traj_samples=16) -> pred_xyz (1, 1, 16, 64, 3)
# opt-in FP8 decode weights: engine_kwargs={"decode_weights": "fp8"}
```

`load_model(config="alpamayo15")` raises a redirect: the model takes camera
frames plus ego history and returns a trajectory with reasoning text, not the
`VLAModel.predict(images)` surface. The resolver entry is
`("alpamayo15", "torch", "thor")`.

## Measured on Thor

Jetson AGX Thor, MAXN with locked clocks (GPC 1575 MHz, EMC 4266 MHz), NGC
PyTorch 26.05. Inputs are real clips of `nvidia/PhysicalAI-Autonomous-Vehicles`
(revision `33f9bf44`) loaded with the official
`load_physical_aiavdataset` at t0 = 5.1 s: the official example clip
`030c760c-ae38-49aa-9ad8-f5650a545d26` and seven more clips, 4 cameras x 4
frames (3086 prompt tokens). One request = decoded `uint8` frames in host
memory -> processor/tokenization -> model -> trajectory on the host, seeded as
the official example, in a warm process.

Official example clip, seed 42 (20 generated tokens, 19 decode steps). Request
times are medians of 5 requests (official N=6/16: 3 and 2); the prefill,
decode and diffusion columns come from one additional instrumented request.
FlashRT rows were measured at the commit that adds the FP8 tier fix (the BF16
path is unchanged by it; earlier commits measured within 1.3%):

| Path | N=1 request ms | Prefill ms (incl. vision) | Decode ms/token | Diffusion ms | N=6 request ms | N=16 request ms |
|---|---|---|---|---|---|---|
| Official, SDPA (eager expert) | 3461 | 1360 | 73.5 | 635 | 12855 | 29868 |
| Official, SDPA + official expert CUDA graph | 3405 | 1356 | 73.9 | 601 | | |
| Official, FlashAttention-2 | 3548 | 1354 | 78.7 | 636 | | |
| FlashRT Thor (BF16) | 2310 | 712 | 68.3 | 230 | 2457 | 2833 |
| FlashRT Thor, `decode_weights="fp8"` | 1658 | 703 | 35.4 | 230 | 1824 | 2336 |

BF16 decode is bounded by streaming the 15.2 GB of decoder weights per token
(the projections run at 230-255 GB/s), so request time grows by about 69 ms
per generated token. On these clips sampled reasoning is 11-67 tokens per
sample at N=1; on one clip (`8b5bcb48`) it reaches 89 tokens at N=6 and the
256-token budget at N=16, on both the official model and FlashRT.

Eight clips x seeds {42, 0, 1, 2, 3} at N=1 (40 requests), seeds {42, 0} at
N=6 (16 requests, 96 samples), seed 42 (official, 8 requests) and {42, 0}
(FlashRT) at N=16, compared request by request with the official SDPA model on
the same seed. These sweeps ran on the BF16 engine before the FP8 tier and on
the first FP8 commit; at the final commit, two clips x five seeds reproduce
both tiers' tokens and trajectories bit-exactly:

| Path | N=1 request ms (median) | N=1 requests token-identical | N=1 mean minADE | N=6 request ms | N=6 samples token-identical | N=16 request ms | N=16 samples token-identical |
|---|---|---|---|---|---|---|---|
| Official, SDPA | 3277 | reference | 3.21 m | 12821 | reference | 31675 | reference |
| Official, FlashAttention-2 | 3362 | 38/40 | 3.15 m | 12708 | 91/96 | | |
| FlashRT Thor (BF16) | 2155 | 38/40 | 3.14 m | 2448 | 89/96 | 2807 | 104/128 |
| FlashRT Thor, FP8 decode | 1579 | 34/40 | 3.18 m | 1815 | 72/96 | 2362 | 86/128 |

The BF16 engine generates the same tokens as the official FlashAttention-2
backend on all 40 N=1 requests; both differ from the SDPA backend on the same
two requests, at near-tie steps (tokens 17 and 41) of one clip (`8b5bcb48`).
At N=1, identical tokens mean identical diffusion noise; trajectories then
differ from the SDPA run by a median 3 cm at 6.4 s (max 32 cm), and from the
FlashAttention-2 run by a median 4 cm, except one 57-token request on
`8b5bcb48` that differs by 1.4 m. (With several samples, the sampler's random
stream follows the longest sample, so a divergence in one sample also changes
the noise of the others.) Teacher-forced on the official tokens, BF16 decode
logits match the official ones with cosine >= 0.99987 and identical argmax at
every step. Paired minADE differences to the official SDPA run: BF16 -7 cm
(95% CI -33..12 cm) at N=1, +0.01 cm (CI -0.5..0.5) at N=6, +0.2 cm
(CI -0.5..0.9, eight requests) at N=16; switching the official model to
FlashAttention-2 moves N=6 minADE by -0.5 cm (CI -1.0..-0.1). N=1 minADE is
dominated by sampling variance across seeds (0.2-2.9 m on the example clip),
so it is not a precise quality metric, and the bootstrap intervals treat
requests from the same clip as independent.

With several samples the official model repeats the whole prompt (images
included) per sample; FlashRT encodes it once, which is where the N=6/16
speedups come from.

## FP8 decode tier

`Alpamayo15Thor(model, decode_weights="fp8")` (frontend:
`engine_kwargs={"decode_weights": "fp8"}`) stores per-row FP8 E4M3 copies
(scale = row absmax / 448, round-to-nearest-even) of the reasoning-decode
projections (q/k/v, o, gate/up, down) and the LM head, and runs them through
`alpamayo15_w8a16_gemm_bf16`: BF16 activations, exact e4m3 -> bf16 widening,
FP32 tensor-core accumulation, up to 16 samples. The projections take about
half the time of their BF16 versions at the same 240-255 GB/s, and decode drops
from 68.3 to 35.4 ms per token. Vision, prefill, the diffusion expert and the
sampler keep BF16 arithmetic, but the expert attends over reasoning K/V
produced by the FP8 projections. The BF16 weights stay resident for prefill;
the FP8 copies add 7.6 GB (7.1 GiB).

This is a numerics change, not a reproduction of the official model.
Teacher-forced on official tokens, the sampling distribution moves by a mean
KL (temperature 0.6) of 8e-5 to 1e-4 on two clips and 2e-3 on a long,
ambiguous clip where 1 of 67 argmax decisions flips (BF16: 5e-6 to 2e-4, no
flips). Free-running, it changes the sampled reasoning on 6/40 N=1 requests,
24/96 N=6 samples and 42/128 N=16 samples (official FlashAttention-2 vs SDPA:
2/40 and 5/96). On these eight clips the paired minADE difference to official
SDPA is -3 cm at N=1 (95% CI -39..30), -7 cm at N=6 (-38..13) and +7 cm at
N=16 (-5..27, eight requests); these intervals are wide and do not show
equivalence. Single requests move by up to 5 m in either direction, all on the
long, ambiguous clip `8b5bcb48` (where one official SDPA run itself exhausts the
256-token budget); on the other clips minADE moves by at most 0.4 m. Against
the BF16 engine at N=16 (16 requests) the mean is +21 cm (-4..59), driven by
`8b5bcb48` (+0.7 m and +2.9 m). Validate on your own scenarios before enabling it; the default is
BF16.

## Limitations

- One scene per request (`num_traj_sets=1`); `num_traj_samples` up to the
  engine's `max_samples`. The navigation-guided CFG entry point is not
  accelerated.
- The engine rewires the official model's q/k/v and gate/up parameters into
  views of fused tensors; do not move the model or replace those parameters
  after constructing the frontend.
- The K/V cache is sized for `max_prompt + max_new_tokens + 64` rows per
  sample (about 0.6 GB per sample at the defaults).
- Offline trajectory metrics on eight clips are not evidence of closed-loop
  driving quality.
- Video frames are decoded by the official loader on the host CPU. With the
  same PyAV/FFmpeg versions, the decoded YUV planes are identical on Thor
  (aarch64) and x86, but the YUV -> RGB conversion differs (mean 0.8 levels,
  at most 2 levels in the sampled pixels), so pixel inputs and therefore
  sampled outputs can differ slightly between platforms.

## FlashDrive streaming execution

[FlashDrive](https://github.com/z-lab/flashdrive) (z-lab, MIT) accelerates
NVIDIA Alpamayo 1.5 by changing how it is run: the model is driven as a
stream of windows that reuses earlier frames' keys/values, its reasoning is
speculated with a DFlash block-diffusion draft, the language model runs as
W4A8 (ParoQuant), and the flow-matching expert is fine-tuned for that stream
and reuses cached velocities on 4 of its 8 Euler steps. FlashRT keeps that
algorithm -- FlashDrive's own `sample_trajectories_streaming` drives every
window -- and executes its primitives on FlashRT kernels and CUDA graphs
(`flash_rt/models/alpamayo15_flashdrive/pipeline_thor.py`), selected per
primitive group so each replacement can be checked against FlashDrive's own
execution.

This is a different inference procedure from the official model (and from
the [full-request engine](alpamayo15.md), which reproduces the
official request semantics). Its outputs, latency and accuracy are not
interchangeable with official full requests; compare it with FlashDrive's own
implementation.

## What FlashDrive changes (kept as is)

| | Official Alpamayo 1.5 | FlashDrive |
|---|---|---|
| Prompt per window | 4 cameras x 4 frames, full prefill | first window full; later windows prefill only each camera's newest frame plus trajectory/text, older frames' K/V shifted in a pre-RoPE cache |
| Reasoning | autoregressive sampling (`max_generation_length` default 128; the official example uses 256) | DFlash: 2-layer draft proposes 7 tokens per round, the target verifies them in one forward and samples; stops at `<cot_end>` or when a sampled bonus token is `.`; 128-token budget |
| Language model | BF16 | W4A8: learned pairwise rotations + INT4 weights (group 128, zero points) + per-token INT8 activations (Marlin) |
| Action expert | official weights, 10 Euler steps | fine-tuned weights, 8 steps, velocity reused on steps 3-6, BF16 state |

Checkpoints: `z-lab/Alpamayo-1.5-10B` carries the official VLM tensors
unchanged (byte-identical) and a fine-tuned action expert (every expert,
`action_in_proj` and `action_out_proj` tensor differs from
`nvidia/Alpamayo-1.5-10B`). `z-lab/Alpamayo-1.5-10B-PARO` is a full model
file: FlashDrive loads it over the base, so its copies of the vision tower,
LM norms, embeddings, `lm_head` and expert are the ones that run (they equal
the base up to an FP16 round trip of values below about 1e-5).
`z-lab/Alpamayo-1.5-10B-DFlash` holds the draft and its mask embedding. The
weights are NVIDIA's (non-commercial license); FlashDrive's code is MIT.

## FlashRT backends

`FlashDriveThor(model, expert=..., vision=..., prefill=..., decode=..., w4a8=..., w4a8_gemm=...)`
installs FlashRT implementations of FlashDrive primitives on a loaded
FlashDrive model (every group by default); groups set to `"flashdrive"` keep
FlashDrive's execution (eager or `torch.compile`). Calls must run under
`torch.autocast("cuda", dtype=torch.bfloat16)`, as FlashDrive's
`scripts/infer.py` does (the backend raises otherwise).

| Group | FlashDrive primitives | FlashRT implementation |
|---|---|---|
| `expert` | `_action` | the 8-step Euler loop with cached velocities in one CUDA graph: FP32 residual stream, fused head RMSNorm + FP32 RoPE, FA4 attention with a device-side key length, FlashDrive's fused q/k/v and gate/up weights |
| `vision` | `_encode` | the FlashRT Alpamayo 1.5 vision tower (one-GEMM patch embedding, FP32 RoPE kernel, batched FA4 attention), one CUDA graph per frame grid |
| `prefill` | `_dflash_prefill`, `_shift_streaming_kv_cache`, cache bookkeeping of `_first_prefill` | 36 decoder layers on FlashRT kernels (fused head norm + RoPE writing the cache, FA4, fused residual + RMSNorm, SiLU-mul) with FlashDrive's W4A8 projections; the window's rows attend by one FA4 varlen call; one CUDA graph per stream |
| `decode` (needs `prefill="flashrt"`) | `_dflash_block_step`, `_dflash_traj_forward` | draft proposal, 8-row verification, posterior sampling (FlashDrive's `sample_tokens`) and acceptance decisions in one CUDA graph, replayed once per round; the trajectory-start forward as a one-row graph |
| `w4a8="fused"` (default with `prefill="flashrt"`) | the W4A8 linears inside the above | one activation pass per shared input (FP16 cast, ParoQuant pairwise rotation, per-token INT8 quantization), vLLM's Marlin W4A8 GEMM on each linear's own buffers, and the FP16 GEMM outputs read directly by the next kernel (head norm + RoPE, residual + RMSNorm, SiLU-mul), which applies FlashDrive's cast to BF16 on load |
| `w4a8_gemm="tcgen05"` (default with `w4a8="fused"`) | the W4A8 GEMMs of streaming-window prefill (groups of at least 256 rows) | an INT8 tensor-core (tcgen05) GEMM on the unpacked `w - z` weights with Marlin's integer group scales; verification and trajectory-start rows stay on Marlin, which is faster at a few rows |

### Cache contract

FlashDrive's `StaticCache` stays the store of record: keys un-rotated and
values, `[1, Hk, S, 128]` per layer, with one position table per stream.
FlashDrive rotates the whole cache on every forward; the backend keeps a
rotated copy instead, equal to FlashDrive's eager rotation (BF16 tables, each
product and the sum rounded to BF16; `torch.compile` fuses that arithmetic and
rounds once) for every row the FlashRT primitives read, updated wherever a row
is written: the first-window prefill, streaming windows, the frame-block
shift, verification rows (including rejected draft rows), the
trajectory-start row and the expert's action rows. The expert rotates the
cache with its own FP32 tables once per window, as FlashDrive's expert does on
each of its forwards. FlashDrive's masks become FA4 lengths: a streaming
window's rows attend causally by cache position (FlashDrive's view-block mask
for Alpamayo 1.5 equals that), verification rows causally over
`[0, cur + 8)`, the one-row trajectory-start forward every cache row (FlashDrive
passes it no mask), the expert rows `[0, action_start + 64)`.

### Numerics

- The fused W4A8 activation pass is bit-exact with ParoQuant's rotation kernel
  and vLLM's per-token INT8 quantization (same INT8 codes and scales). With
  `w4a8_gemm="marlin"`, and for verification and trajectory-start rows, the
  GEMM is vLLM's Marlin kernel itself; the kernels that read the FP16 GEMM
  outputs produce exactly what FlashDrive's cast to BF16 followed by the BF16
  kernel produces.
- The tcgen05 W4A8 GEMM computes Marlin's integer arithmetic exactly: per
  128-channel group the INT8 x (`w - z`) sums times the integer group scale,
  accumulated in int32, converted to float once and multiplied by the
  per-token scale. Marlin converts partial sums over slices of K to float and
  adds them in FP32, so the two agree except where Marlin's FP32 sum rounds to
  the neighbouring FP16 value (on the checkpoint's layers about 0.001-0.01% of
  outputs, by one FP16 ulp in the cases observed). The int32 total (Marlin
  also accumulates int32 within each slice) wraps only above 2^31, at least
  16x above the largest total measured on the checkpoint's layers with
  uniformly random INT8 activations (not correlated with the weights). The
  weights are unpacked from Marlin's buffers through vLLM's own layout
  definitions and checked against Marlin when the backend is built.
- The other FlashRT primitives reproduce FlashDrive's eager operation sequence
  (rounding points, RNG consumption) with different kernels (FA4, FlashRT
  norms), so results differ from FlashDrive eager at the reduction-order
  level. On identical cache and random state, these differences are no larger
  than those between FlashDrive's own eager and `torch.compile` executions
  (see the last table below). Neither FlashDrive's compiled stack nor the
  backend is bit-reproducible against the other, so sampled reasoning can
  diverge at near-tie tokens.

## Requirements

- `GPU_ARCH=110` build with `-DFLASHRT_ENABLE_ALPAMAYO15=ON` and the FA4
  runtime (see [Alpamayo 1.5 on Thor](alpamayo15.md#build)). The option
  also compiles the tcgen05 W4A8 GEMM against the vendored CUTLASS (v4.4.2).
- `flashdrive`, its Alpamayo dependency (`alpamayo1_5`), `paroquant` (its
  rotation kernel is compiled on first use) and vLLM's Marlin W4A8 GEMM
  (`torch.ops._C.marlin_gemm` with INT8 activations), which FlashDrive's W4A8
  layers call. vLLM wheels target other PyTorch builds and GPUs; on Thor's
  platform PyTorch, build the Marlin kernels from the vLLM source for
  `sm_110` (only the INT8-activation / UINT4-weight instantiations are used).

## Usage

```python
from flash_rt.frontends.torch.alpamayo15_flashdrive_thor import (
    Alpamayo15FlashDriveTorchFrontendThor, FlashDriveCheckpoints)

fe = Alpamayo15FlashDriveTorchFrontendThor(
    FlashDriveCheckpoints(base="z-lab/Alpamayo-1.5-10B",          # local copies of the z-lab
                          paro="z-lab/Alpamayo-1.5-10B-PARO",     # checkpoints work too: the
                          dflash="z-lab/Alpamayo-1.5-10B-DFlash"))# companions are explicit
for window in windows:                 # consecutive windows of one clip, 0.1 s apart
    out = fe.step(window["image_frames"], window["camera_indices"],
                  window["ego_history_xyz"], window["ego_history_rot"])
    # out is None for the first window of a stream (prefill only)
fe.reset()                             # before an unrelated stream
```

The frontend's `torch_compile` (default `"max-autotune"`, FlashDrive's own
setting) applies only to primitive groups left on FlashDrive; with the default
backends FlashDrive executes only its first-window prefill, which it does not
compile.
`FlashDriveRollout` holds FlashDrive's controls (token budget 128,
temperature 0.6, top-p 0.98, 8 Euler steps, cached steps 3-6, one sample).
`reset()` ends a stream completely (FlashDrive's cache, shape caches,
streaming mask and compiled-step state); primitives that FlashDrive executes
with `torch.compile` are compiled again for the next stream.
FlashDrive's `scripts/infer.py` sets `torch.set_float32_matmul_precision("high")`
for the process; the frontend leaves this global setting to the caller.
`load_model(config="alpamayo15_flashdrive")` raises a redirect to this
frontend.

## Measured on Thor

Jetson AGX Thor (20 SMs), locked clocks (GPC 1575 MHz, EMC 4266 MHz), NGC
PyTorch 26.05. Nine clips of `nvidia/PhysicalAI-Autonomous-Vehicles`
(FlashDrive's default clip `87147a1b` with 120 windows and eight more with
60), windows from t0 = 1.7 s, 0.1 s apart, built by the official loader; 4
cameras x 4 frames; one trajectory sample. Latency is per window from inputs
on the host (tokenized) to the trajectory on the host, windows 3 onward; the
streaming arms use FlashDrive's `scripts/infer.py` settings and seeds 0-2
(1719 windows each). The two protocols below run different procedures and are
not comparable with each other: a streaming window reuses earlier frames'
keys/values and DFlash-speculates its reasoning; a full request re-encodes all
16 frames and decodes autoregressively. Bracketed ranges are 95% bootstrap
intervals over clips.

Streaming (FlashDrive's procedure):

| Path | mean ms | median ms | p90 ms | p99 ms | peak GB |
|---|---|---|---|---|---|
| FlashDrive (unmodified, `torch.compile` max-autotune) | 956 [856, 1106] | 856 | 1057 | 2888 | 13.3 |
| FlashDrive + FlashRT backend (defaults) | 514 [454, 610] | 461 | 571 | 1625 | 20.8 |

Per window the backend is 442 ms faster on average (paired by clip, seed and
window, [399, 504] ms). FlashDrive averages 4.64 DFlash rounds and 3.56
reasoning tokens per round; the backend 4.62 and 3.61. Cold costs (medians):
FlashDrive's first window of a stream takes 4.4 s and its second 148 s
(`torch.compile` with a warm compile cache; FlashDrive's `scripts/infer.py`
runs one clip per process). The backend ran each seed's nine clips in one
process with `reset()` between them: first window 2.5 s, second 1.2 s (CUDA
graph capture; 37 s for the first stream of a process, which also builds the
FA4 kernels). The backend's peak includes the INT8 weight copy of
`w4a8_gemm="tcgen05"`.

Trajectory error against the logged future (meters, xy, one sample):

| Path | ADE mean | ADE median | ADE p95 | ADE max | FDE mean |
|---|---|---|---|---|---|
| FlashDrive (unmodified) | 2.710 [1.74, 3.91] | 1.741 | 9.51 | 21.1 | 8.087 [5.38, 11.68] |
| FlashDrive + FlashRT backend | 2.714 [1.75, 4.05] | 1.778 | 9.83 | 19.9 | 8.114 [5.27, 11.70] |

Paired difference (backend - FlashDrive): ADE +0.004 m [-0.043, +0.059], FDE
+0.027 m [-0.125, +0.193]. Reasoning is sampled (temperature 0.6), so these
are distribution-level comparisons; FlashDrive's own compiled stack does not
reproduce its tokens across processes with the same seed. The backend itself
is reproducible: three processes with the same seed produced identical
reasoning tokens and trajectories on all 117 windows of the default clip.

Full requests (official request semantics, seed 0, the same clips and
windows):

| Path | mean ms | median ms | p90 ms | p99 ms | peak GB |
|---|---|---|---|---|---|
| Official Alpamayo 1.5 (BF16, SDPA) | 3320 [3102, 3612] | 3164 | 3745 | 5939 | 21.6 |
| FlashRT Alpamayo 1.5 engine (BF16) | 2159 [1956, 2444] | 2012 | 2555 | 4583 | 22.1 |

ADE 2.867 vs 2.852 m (paired difference -0.015 [-0.091, +0.024]).

Per primitive group (FlashDrive's default clip, seed 0; each row adds one
group to the previous):

| Configuration | mean ms | median ms | prefill | DFlash round | expert | encode |
|---|---|---|---|---|---|---|
| FlashDrive (unmodified) | 929 | 855 | 413 | 64.0 | 165 | 47 |
| expert | 863 | 789 | 413 | 64.9 | 95 | 47 |
| + vision | 922 | 790 | 413 | 64.8 | 95 | 46 |
| + streaming prefill / shift (FlashDrive W4A8 modules) | 849 | 713 | 339 | 64.3 | 95 | 47 |
| + DFlash verify / trajectory start | 690 | 667 | 339 | 37.8 | 95 | 46 |
| + fused W4A8 activations (Marlin GEMM) | 619 | 597 | 278 | 36.1 | 95 | 46 |
| + tcgen05 W4A8 prefill GEMM (default) | 491 | 497 | 179 | 36.1 | 95 | 46 |

Stage columns are CUDA-event means per window (ms; the DFlash column per
round). Groups left on FlashDrive run compiled. Window latency depends on the
number of DFlash rounds, which varies with the sampled reasoning, so compare
the stage columns between rows.

On identical cache and random state (12 windows, each FlashRT primitive
against FlashDrive eager; FlashDrive's compiled primitives against eager for
scale), minimum cosine similarity of outputs / written cache rows:

| Primitive | FlashRT vs eager | FlashDrive compiled vs eager |
|---|---|---|
| vision | 0.9991 / - | 0.9988 / - |
| streaming prefill | 0.9993 / 0.9974 | 0.9993 / 0.9973 |
| DFlash round | 0.9987 / 0.9970 | 0.9090 / 0.8164 |
| trajectory start | - / 0.9986 | - / 0.9986 |
| expert | 1.0000 / 0.9999 | 1.0000 / 0.9994 |

Random-number consumption matched FlashDrive on every call, and the rotated
cache copy equalled the rotation of FlashDrive's cache after every FlashRT
primitive.

## Limitations

- One trajectory sample per window; Alpamayo 1.5 checkpoints only.
- `w4a8_gemm="tcgen05"` keeps an INT8 copy of the language model's linear
  weights (about 7 GB) next to Marlin's INT4 buffers; `w4a8_gemm="marlin"`
  avoids it.
- Streams are reset explicitly (`reset()`); FlashDrive's first window of each
  stream is a full prefill that returns no trajectory.
- ADE over nine clips and three seeds compares the two implementations of the
  same procedure; it is not an accuracy evaluation of the model.
