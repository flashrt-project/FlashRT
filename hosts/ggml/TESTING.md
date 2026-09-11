# Testing

Three layers of validation, from operator level to release gate.

## 1. Operator tests (`test-backend-ops`)

The host's `test-backend-ops` exercises the NVFP4 mul_mat path against the
CPU reference:

```bash
GGML_CUDA_FLASHRT_NO_CACHE=1 ./build/bin/test-backend-ops test -o MUL_MAT
```

`GGML_CUDA_FLASHRT_NO_CACHE=1` is **required**: the weight repack cache is
keyed by tensor data pointer under the assumption that weights are
immortal, which holds for models but not for the test harness's rapidly
recycled tensors. NVFP4 mismatches up to ~2e-2 are inherent W4A4
activation-quantization noise (upstream applies the same tolerance to
native FP4 backends), not failures.

## 2. Qualification gates (`qualification/`)

`qualification/run_qualification.py` gates a build the way a release
would:

- **manifest** — the pipeline binding
  (`bindings/jetson_pi_edge_pi05.yaml`) must map the host's complete hot
  path onto catalog structures.
- **pins** — the structure versions the binding names must match the
  catalog (`qualification/pins.yaml`).
- **e2e golden** (`--e2e`, on-device, needs a running server) — drives the
  fixed synthetic-input protocol and compares the steady-state action
  chunk against `qualification/goldens/pi05_thor_action.json` **exactly**.
  The adapter is bitwise deterministic across processes after warmup, so
  any bit difference is a real change.

```bash
python qualification/run_qualification.py           # offline gates
python qualification/run_qualification.py --e2e     # + on-device golden
python qualification/run_qualification.py --e2e --update-golden
```

Take the golden only after at least two warm-up inferences (the first
inference after cold start differs from steady state) and only for
changes whose numerics were judged (below).

## 3. Benchmark and parity methodology

Every performance change must pass this protocol on device:

- **Hot-regime A/B/A sandwich** — run the candidate, the fallback (via its
  runtime switch), and the candidate again as three separate server
  processes, ≥15 warm-up + ~20 measured inferences each, comparing P50.
  Thor drifts ±1–3 ms across long sessions, so only same-session
  back-to-back numbers are comparable; single measurements and
  cross-session comparisons are not accepted.
- **Bitwise parity** — save the action chunk from each leg. A change that
  claims numeric neutrality must be bit-identical to the previous
  accepted state. Note the converse trap: bit-identical output *plus*
  zero performance delta usually means the window never fired — verify
  the window triggers (kernel census, `GGML_FLASHRT_DEBUG`) before
  interpreting the A/B.
- **Real-observation judge** — for changes that move numerics, run a set
  of real robot observations (gripper-active frames) through the NVFP4
  build and an f16-weights build of the same tree, and compare per-dim
  cosine of the action chunks against the f16 reference. The distance to
  the reference must not systematically regress. Any bit-level change in
  the action path amplifies to ~2e-2 absolute wobble on final actions
  through the 10 denoise steps, so raw action diffs are meaningless —
  only the distance-to-reference comparison judges accuracy.
- **Kernel-level accounting** — attribute wins with an nsys census
  (`GGML_CUDA_DISABLE_GRAPHS=1`, full-lifetime `-t cuda` trace with a
  graceful server exit so buffers flush). CUDA-graph replays hide kernels
  from the profiler, and profiling on Tegra inflates kernel times, so the
  census attributes *where* time went while the non-profiled A/B decides
  *whether* the change lands.
