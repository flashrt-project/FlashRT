# FlashRT — Docker

The fastest path to a working FlashRT install. One image, one
command, no CUTLASS clone, no `flash-attn` wheel-hunting, no manual
`cp *.so` step.

Two Dockerfiles ship with the repo:

| Hardware | Dockerfile | NGC base |
|----------|------------|----------|
| RTX 5090 / 4090 / 3090 / Ampere (x86_64)    | [`Dockerfile`](Dockerfile)         | `nvcr.io/nvidia/pytorch:25.10-py3` |
| Jetson AGX Thor (SM110, aarch64)            | [`Dockerfile.thor`](Dockerfile.thor) | `nvcr.io/nvidia/pytorch:25.09-py3` (arm64 manifest) |

Thor uses a hand-tuned cuBLAS-decomposed attention path
(`csrc/attention/fmha_dispatch.cu`) instead of the vendored
Flash-Attention 2, so its image deliberately does NOT produce
`flash_rt_fa2.so`. Everything else builds the same way. Skip to
[§4](#4-thor-jetson-agx-thor-sm110-aarch64) for the Thor flow.

---

## 1. Build the image locally (current default path)

> **Note on the prebuilt registry image.** Following the
> `flash_vla → flash_rt` package-rename refactor that landed in
> [#6](https://github.com/flashrt-project/FlashRT/pull/6), the
> `ghcr.io/liangsu8899/flashrt` image has not been re-pushed yet —
> we plan to push the new image once the post-rename surface is
> fully stable. Until then, build the image yourself with the
> commands below — it's a one-time `cmake/make` pass on top of the
> NGC base image, and the produced `.so` files match what the
> registry image will eventually ship.

Build the image yourself when you want to pin a specific commit,
target a different GPU than the build host, or modify the kernels:

```bash
# Default — auto-detects GPU arch via nvidia-smi (requires --gpus on build).
docker build -t flashrt:dev -f docker/Dockerfile .

# Pin to a specific arch (recommended for image distribution):
docker build -t flashrt:5090 \
    --build-arg GPU_ARCH=120 \
    -f docker/Dockerfile .

# Slim FA2 codegen for shipped models only (Pi0/Pi0.5/GROOT use 96 + 256):
docker build -t flashrt:slim \
    --build-arg GPU_ARCH=120 \
    --build-arg FA2_HDIMS="96;256" \
    -f docker/Dockerfile .
```

### Build args

| Arg | Default | When to set |
|---|---|---|
| `BASE_IMAGE` | `nvcr.io/nvidia/pytorch:25.10-py3` | Pin to an older NGC if your host CUDA driver is old. |
| `GPU_ARCH` | _(auto-detect)_ | Set when shipping the image to a different GPU than the build host. `120`=5090, `89`=4090, `86`=3090, `80`=A100. |
| `CUTLASS_REF` | `v4.4.2` | Bump if the upstream tag is yanked or you want to test a newer CUTLASS. |
| `FA2_HDIMS` | _(all of 96;128;256)_ | Drop unused head_dims to slim the image. Shipped models only need `96;256`. |

### Build time

Cold build dominated by two phases: pulling the NGC base image
(network-bound, depends on bandwidth and CDN warmth) and the FA2
template instantiation pass during `make -j`. Subsequent rebuilds
reuse the NGC layer and CUTLASS clone, leaving only the kernel
compile. `FA2_ARCH_NATIVE_ONLY=ON` plus a single-arch slim
materially shortens the kernel compile by skipping non-native AOT
passes — useful when iterating on the source.

### Pushing your build to a private registry (Modal / RunPod / cloud)

Until the public registry image is re-pushed, the standard cloud
flow is to push your local build to a registry you own and point
the cloud runtime at it:

```bash
docker tag flashrt:5090 <your-registry>/flashrt:0.2.0
docker push <your-registry>/flashrt:0.2.0
```

```python
# Modal example (mirrors the eventual public-image flow)
import modal

image = modal.Image.from_registry(
    "<your-registry>/flashrt:0.2.0"
).pip_install("your-app-deps")

app = modal.App("flashrt-app", image=image)

@app.function(gpu="L40S")  # or H100, A100, etc.
def infer():
    import flash_rt
    model = flash_rt.load_model(checkpoint="/path/to/ckpt", framework="torch")
    ...
```

Once `ghcr.io/liangsu8899/flashrt:<tag>` is re-pushed, swap in the
public URL — the rest of the pipeline stays identical.

---

## 2. Run

```bash
# Default: drops you in a Python REPL with `flash_rt` already imported.
docker run --rm --gpus all -it flashrt:dev

# Run the quickstart against a checkpoint mounted from the host:
docker run --rm --gpus all \
    -v /path/to/pi05_ckpt:/ckpt:ro \
    flashrt:dev \
    python3 examples/quickstart.py --checkpoint /ckpt --benchmark 20
```

---

## 3. What's inside

- Base: `nvcr.io/nvidia/pytorch:25.10-py3`
  (CUDA 13.0, PyTorch 2.9, cuBLASLt, nvcc, Python 3.12)
- CUTLASS 4.4.2 vendored at `/opt/cutlass`
- FlashRT source at `/workspace/FlashRT`, editable-installed
- Kernel `.so` files prebuilt directly into `flash_rt/`. The exact
  set depends on the target GPU arch (gating defined in
  [`CMakeLists.txt`](../CMakeLists.txt)):

  | Target | Always | FA2 (sm_80/86/89/120) | NVFP4 GEMM (sm_120) | SM100 FMHA (sm_100/110) |
  |---|---|---|---|---|
  | `flash_rt_kernels.so`        | ✓ | — | — | — |
  | `flash_rt_jax_ffi.so`        | ✓ | — | — | — |
  | `flash_rt_fp4.so`            | ✓ | — | NVFP4 paths active here | — |
  | `flash_rt_fa2.so`            | — | ✓ | — | skipped |
  | `libfmha_fp16_strided.so`    | — | skipped | — | ✓ |

  In the default x86 build (auto-detected sm_120 on RTX 5090) you
  get 4 `.so` files: `flash_rt_kernels`, `flash_rt_fa2`,
  `flash_rt_fp4`, `flash_rt_jax_ffi`. The Thor build (sm_110) also
  produces 4 but swaps `flash_rt_fa2` for `libfmha_fp16_strided`.
- An import smoke check runs at image-build time, so a broken image
  fails the `docker build` instead of the user's first pull.

The image deliberately does **not** include the upstream `flash-attn`
pip wheel — the default RTX path uses the vendored `flash_rt_fa2.so`
and works without it. If you need legacy upstream attention or run
GROOT, install it yourself:

```bash
docker run --rm --gpus all flashrt:dev \
    pip install flash-attn  # or build from source per upstream docs
```

---

## 4. Thor (Jetson AGX Thor, SM110, aarch64)

The Thor image uses a separate Dockerfile, [`Dockerfile.thor`](Dockerfile.thor),
because Thor pulls a different NGC manifest (`linux/arm64`) and skips
the FA2 build (Thor has its own attention path). Build on a Thor
host so `nvidia-smi` auto-detects `sm_110a`:

```bash
# On the Thor host
docker build -t flashrt:thor -f docker/Dockerfile.thor .

# Run (note --runtime=nvidia for Jetson — see below for why)
docker run --rm --gpus all -it --runtime=nvidia flashrt:thor
```

### Why `--runtime=nvidia` on Jetson

Unlike a discrete-GPU host (where `--gpus all` alone is enough — the
libnvidia-container shim auto-discovers `/dev/nvidia*` and the
matching driver libs), Jetson's iGPU stack is bound to host kernel
drivers and is exposed to containers through a **CSV-driven
mount mechanism** owned by `nvidia-container-runtime`:

```
/etc/nvidia-container-runtime/host-files-for-container.d/
├── devices.csv     # /dev/nvgpu, /dev/nvhost-*, /dev/nvmap, …
└── drivers.csv     # /usr/lib/aarch64-linux-gnu/tegra/libcuda.so.*, …
```

Passing `--runtime=nvidia` is what activates that runtime, which in
turn parses the two CSV files at container start and bind-mounts
every listed device node and driver library from the Tegra host
into the container. Without the flag the standard runc starts the
container without those mounts; the result is no `/dev/nvgpu`, no
`libcuda.so`, and `torch.cuda.is_available()` returns `False` even
though `nvidia-smi` works on the host.

`--gpus all` is left in the example for parity with the x86 docs and
because the libnvidia-container CLI hook ignores it gracefully on
Jetson, but the load-bearing flag here is `--runtime=nvidia`.

### What's different vs the x86 image

- **Base**: `nvcr.io/nvidia/pytorch:25.09-py3` (one minor older than the
  x86 image — 25.09 has the validated arm64 / Thor manifest, 25.10
  arm64 has not been smoke-tested on SM110 yet).
- **Build targets**: 4 `.so` files (`flash_rt_kernels`,
  `flash_rt_fp4`, `libfmha_fp16_strided`, `flash_rt_jax_ffi`).
  Same artifact count as the default x86 build but with
  `libfmha_fp16_strided` swapped in for `flash_rt_fa2`.
- **No `flash_rt_fa2.so`**: Thor's `csrc/attention/fmha_dispatch.cu`
  loads `libfmha_fp16_strided.so` at runtime via dlopen instead of
  going through the FA2 template instantiation pass — that's the
  largest cold-build saving on Thor vs x86.
- **`flash_rt_fp4.so` on Thor**: built for sm_110a, but NVFP4 GEMM
  paths gate to sm_120 only at runtime
  (see [`docs/kernel_catalog.md`](../docs/kernel_catalog.md), the
  `quantize_bf16_to_nvfp4` and `has_nvfp4()` entries). The kernel
  object compiles fine on Thor; calls into NVFP4-only entry points
  short-circuit when `has_nvfp4()` returns False.

### Build args

Same as the x86 image (`GPU_ARCH`, `CUTLASS_REF`), minus `FA2_HDIMS`
which is a no-op on Thor.

### Smoke check

The image-build smoke deliberately asserts `libfmha_fp16_strided.so`
is present and does NOT import `flash_rt_fa2`, so a future regression
that reintroduces FA2 onto Thor by accident gets caught at build
time.

