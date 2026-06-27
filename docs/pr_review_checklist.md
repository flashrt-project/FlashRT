# FlashRT PR Review Checklist

This checklist is the public maintenance standard for FlashRT pull requests.
It is intentionally strict because FlashRT is a multi-model, multi-framework,
multi-hardware runtime. Reviewers should optimize for long-term correctness,
clear ownership, and incremental development over short-term benchmark wins.

The core contract is:

- predictable low-latency inference
- explicit hardware and model routing
- stable public APIs
- incremental changes over broad rewrites
- thin Python glue composing C++/CUDA kernels
- clean frontend, model pipeline, hardware, and serving boundaries
- opt-in feature gates for model-specific kernels
- clear failure modes instead of silent fallback
- no private paths, local environment assumptions, generated traces, or stale
  benchmark claims

## 1. Review Verdicts

Use one of these outcomes:

| Verdict | Meaning | Action |
|---|---|---|
| Accept | Correct, isolated, documented, and tested enough for the risk | Merge |
| Accept after small cleanup | Only wording, comments, stale doc text, or harmless cleanup remains | Ask for cleanup or patch directly |
| Request changes | API mismatch, failing test, unclear build gate, wrong docs, or unsupported fallback | Block merge until fixed |
| Blocker | Existing path can break, math is wrong, symbols are missing, private data leaked, or behavior is unsafe | Must fix before merge |
| Needs hardware validation | Code is structurally right but touches hardware or precision paths not validated by reviewer | Require contributor or CI evidence |

Do not merge a PR with a known blocker because the diff is small.

## 2. Risk Classification

Classify the PR before reviewing implementation details.

| Risk | Typical PR | Minimum review posture |
|---|---|---|
| R0 | Docs typo, comment-only update, dead test cleanup | Check diff and links |
| R1 | Local bug fix in one non-hot file | Focused test and import check |
| R2 | New opt-in frontend/model path, isolated docs/tests | Build/import where relevant plus correctness evidence |
| R3 | CMake, binding, kernel, calibration, graph, cache, or dispatch change | Full affected-path review plus old-path regression evidence |
| R4 | Public API, shared runtime refactor, default behavior, multi-hardware routing, serving/exec contract | Require design clarity, tests, docs, and rollback path |

The stricter checklist applies if the PR touches any of:

- `CMakeLists.txt`
- `csrc/`
- `flash_rt/api.py`
- `flash_rt/hardware/`
- `flash_rt/frontends/`
- `flash_rt/models/`
- `serving/`
- calibration, cache, CUDA Graph, dispatch, or precision code

## 3. Basic Review Commands

Use a temporary worktree rather than a developer's active checkout.

```bash
PR=<number>
REPO=flashrt-project/FlashRT
ROOT=<repo-root>
WT=<temporary-worktree>

git -C "$ROOT" fetch origin main pull/${PR}/head:refs/remotes/origin/pr${PR}-head --force
rm -rf "$WT"
git -C "$ROOT" worktree prune
git -C "$ROOT" worktree add "$WT" origin/pr${PR}-head
cd "$WT"

gh pr view "$PR" --repo "$REPO" \
  --json title,author,headRefName,headRefOid,baseRefName,mergeable,isDraft,url,additions,deletions,changedFiles,body

git diff --stat origin/main...HEAD
git diff --name-status origin/main...HEAD
git log --oneline origin/main..HEAD
git diff --check origin/main...HEAD
```

Ask for a rebase when the PR changes the same ownership area as recent main:
CMake targets, binding lists, frontend/pipeline files, kernels, generated
artifacts, or performance-sensitive dispatch code.

## 4. Scope And Isolation

Required:

- The PR title and body must match the actual diff.
- A PR should have one primary reason to exist: one bug fix, one model path,
  one hardware path, one precision path, one serving host, or one refactor.
- New feature code should be additive unless the PR is explicitly a refactor.
- Existing behavior should remain unchanged when the new flag, route, model, or
  hardware target is not selected.
- Shared helper changes require a call-site inventory and old-path evidence.
- Hardware-specific changes must stay under explicit hardware routing.
- Model-specific changes must not be placed in generic helpers unless they are
  truly model-agnostic.

Blockers:

- A model PR changes unrelated models.
- A hardware PR changes shared dispatch without proving old hardware paths.
- A precision PR changes default dtype selection for unrelated paths.
- A "cleanup" PR changes launch order, dtype, shape semantics, cache semantics,
  or public behavior.
- A feature is enabled by default but validated on only one model or machine.
- The PR mixes unrelated model kernels into a bug fix or shared runtime change.

## 5. Repository Layering

Keep responsibilities in their owner layer.

| Layer | Owns | Must not own |
|---|---|---|
| `csrc/` | C++/CUDA kernels, launch wrappers, pybind ABI, vendor kernel adapters | model routing, prompts, checkpoint names, serving policy |
| `exec/` | C ABI replay mechanism: Buffer, Graph, Plan, Event, ShapeKey | sessions, schedulers, KV semantics, protocol fields, model policy |
| `flash_rt/hardware/` | arch detection, attention backend factories, hardware-generic primitives | model-specific decoder logic or checkpoint-specific shapes |
| `flash_rt/models/<model>/` | per-model compute pipeline and model-local helpers | public root exports, serving protocol, unrelated shared utilities |
| `flash_rt/frontends/<framework>/` | IO path, weight loading, calibration, buffer allocation, graph capture | low-level kernel implementations, cross-model policy |
| `serving/` | scenario hosts, sessions, protocols, request policy | kernel implementation, default core imports, common execution policy |
| `training/` | training and finetuning paths | inference hot-path dependencies unless explicitly shared and tested |

Blockers:

- A frontend imports a serving host or server-only dependency.
- A model pipeline imports FastAPI, datasets, training packages, notebooks, CLI
  parsing, or benchmark harness code.
- A generic hardware helper contains tensor names, layer counts, prompt rules,
  or checkpoint keys from one model.
- `exec/` gains a field or verb whose meaning is specific to one model family,
  protocol, session policy, scheduler policy, or KV-cache policy.

## 6. Public API And Import Boundaries

Required:

- Stable APIs must match `docs/stable_api.md`.
- Model-specific helper APIs should live under `flash_rt.models.<model>` or a
  framework-specific frontend module, not the `flash_rt` root.
- Optional model packages must be lazy imported.
- `import flash_rt` must not require optional model, server, eval, training, or
  checkpoint dependencies.
- Importing a model package should ideally succeed without the external model
  package; runtime entry points can fail clearly when called.
- New dependencies must land in the narrowest optional extra.
- Environment variables are allowed only for explicit opt-in behavior,
  diagnostics, or temporary compatibility, and must be documented.

Blockers:

- `import flash_rt` fails without an optional dependency.
- Core install starts requiring heavy optional packages.
- Tests assert an API path that implementation does not expose.
- Docs show a root import for a model-specific API that is not exported there.
- A missing optional `.so` triggers an unclear low-level error instead of a
  clear `RuntimeError` or equivalent fail-fast message.

Minimum checks:

```bash
PYTHONPATH=. python - <<'PY'
import flash_rt
print("flash_rt import ok")
PY

PYTHONPATH=. python - <<'PY'
import importlib
m = importlib.import_module("flash_rt.models.<model>")
print("model package import ok", m)
PY
```

## 7. CMake, Binding, And Module Ownership

Every pybind symbol must have matching build ownership.

Required:

- Feature-specific modules should be behind explicit CMake flags.
- Default `flash_rt_kernels` must not compile model-specific kernels unless
  those kernels are truly shared.
- Vendor-, architecture-, and precision-specific sources must be gated at the
  object-library or target level, not only at runtime.
- Gated `.cu` sources must not be referenced by unconditional pybind entries.
- Dedicated modules are preferred for large model-specific kernel groups:
  `flash_rt_<model>_kernels`, `flash_rt_<feature>`, or similar.
- A new CMake flag must have a clear default, status message, target ownership,
  and docs entry.
- Architecture labels and CMake feature flags must be consistent with runtime
  hardware detection and routing.

Blockers:

- Undefined symbol at import time.
- Duplicate symbol at link time.
- CMake flag says OFF by default but sources still enter the default target.
- Architecture-specific kernels compile into unsupported architecture targets.
- Vendor-specific compile options leak into generic targets without a gate.
- Binding list and required-symbol list are not updated together.

Build checks:

```bash
cmake -S . -B <build-dir> -DGPU_ARCH=<target_arch> -DCMAKE_BUILD_TYPE=Release
cmake --build <build-dir> --target flash_rt_kernels -j$(nproc)
```

For a gated module:

```bash
cmake -S . -B <build-dir> \
  -DGPU_ARCH=<target_arch> -DCMAKE_BUILD_TYPE=Release \
  -D<FEATURE_FLAG>=ON
cmake --build <build-dir> --target <target_module> -j$(nproc)
```

Import check:

```bash
PYTHONPATH=. python - <<'PY'
from flash_rt import flash_rt_kernels as fvk
print(fvk.__file__)
PY
```

## 8. Kernel Naming And Long-Term Ownership

Only truly model-agnostic kernels may use generic names. Model-, hardware-, or
shape-specialized kernels must include an ownership prefix in:

- file path
- `.cu/.cuh` file name
- C++ function name
- pybind symbol name
- Python required-symbol list
- docs and tests

Good:

```text
csrc/kernels/<model>/<model>_qk_norm_rope.cu
<model>_qk_norm_rope_bf16

csrc/kernels/<feature>/<feature>_matmul_sm120.cu
<feature>_matmul_sm120_bf16
```

Bad:

```text
csrc/kernels/fused_qk_norm_rope.cu
fused_qk_norm_rope_v4_bf16

csrc/kernels/cfg_combine.cu
cfg_combine_log_softmax_bf16
```

Required:

- No experimental suffixes such as `_v4`, `_new`, or `_fast` in public pybind
  names unless the version is a documented ABI.
- Use hardware or precision suffixes only when they are part of the contract.
- If the kernel is shape-specialized, document the shape constraints and guard
  unsupported shapes before launch.
- Dead or future kernels must not be exported or listed as required.
- If a kernel is exported, the runtime must call it or docs must mark it
  explicitly experimental and not required.

Blockers:

- A model-specific kernel uses a generic name.
- Docs claim a kernel is on the hot path but runtime does not call it.
- Required-symbol tests include unused symbols.
- A symbol is renamed but stale docs/tests still use the old name.
- Shape-specialized code accepts generic shape parameters and can silently
  compute wrong results.

## 9. Pybind ABI And Shape Contracts

Required:

- Every `m.def(...)` argument list must match the C++ launcher signature.
- Pybind names must match Python call sites and required-symbol lists.
- Raw pointer arguments should be typed consistently.
- Shape argument order must be documented when not obvious.
- Backward-compatible aliases must preserve old shape semantics.

Blockers:

- Binding accepts one shape order while caller passes another.
- Binding alias maps an old API to new indexing without proof.
- Device pointer is built from a temporary tensor that can be freed before use.
- `.data_ptr()` is taken from an unanchored tensor and stored for later.
- Binding signature mismatch compiles but fails at runtime.

Minimum tests:

- Import the module.
- Check every required symbol exists.
- Run a small numerical smoke for the new binding where possible.
- Test missing-symbol fail-fast behavior for required kernels.

## 10. Hot Path Cleanliness

The hot path includes forward, decode, CUDA Graph replay, and per-step sampling.

Forbidden in hot path unless explicitly justified:

- `.item()`
- `.cpu()`
- `.numpy()`
- host-to-device scalar readback
- `torch.cuda.synchronize()`
- debug `print()`
- dynamic tensor allocation
- changing Python containers that define capture-time launch order
- heavy imports inside repeated forward
- silent PyTorch fallback for operations claimed to be kernelized

Required:

- CUDA Graph replay must reuse stable buffers and pointers.
- Warmup and capture must execute the same kernel path.
- Prompt, state, or cache changes must invalidate captured state.
- Dynamic shapes must be bucketed, pre-warmed, or rejected clearly.

Blockers:

- Host sync in decode/prefill hot path without a correctness reason.
- Cache is reused across prompt/model-state changes without reset.
- A CUDA Graph captures one branch and replay uses another branch.
- A required kernel is missing and the runtime silently enters PyTorch.

## 11. Model And Hardware Routing

Required:

- Use one pipeline per `(model, hardware)` for graph-captured or VLA-style
  runtime paths.
- Use one frontend per `(model, framework, hardware)`.
- Routing map entries must be explicit.
- Hardware-specific path names should include hardware in file or class names.
- Cross-hardware sharing should go through small helpers, not large runtime
  `if arch` branches.
- Existing routing defaults must not change unless the PR is explicitly a
  routing migration with old-path evidence.
- Plugin or model registration must remain additive and explicit.

Blockers:

- Multiple hardware targets share one frontend with many runtime branches.
- A new path changes default routing for existing models.
- A framework path silently enters another framework's unvalidated route.
- Hardware behavior is selected by guesswork rather than explicit capability.
- A new hardware target reuses an existing frontend without documenting
  unsupported kernels, precision modes, and validation gaps.

## 12. Frontend And Pipeline Ownership

Frontend responsibilities:

- Validate user inputs and config.
- Load checkpoints through declarative specs or documented adapters.
- Own persistent tensors, buffers, scales, and pointer lifetimes.
- Build attention specs and choose hardware backends through documented
  factories.
- Run calibration, warmup, CUDA Graph capture, and fail-fast symbol checks.
- Expose the documented model surface.

Pipeline responsibilities:

- Compose already-owned buffers and weights through kernel calls.
- Keep launch order deterministic for capture.
- Accept raw pointers, primitive dims, backend handles, streams, and small
  immutable config objects.
- Keep hardware-specific compute in `pipeline_<hardware>.py` or model-local
  helpers imported only by that hardware path.

Pipeline must not:

- Import checkpoint loaders, tokenizers, web/server frameworks, datasets, CLI
  parsers, training loops, or benchmark harnesses.
- Allocate dynamic tensors in repeated forward/decode paths.
- Read host values from device tensors in hot path.
- Hide required kernels behind `hasattr(...)` unless the fallback is a tested
  optional fast path.
- Implement large chunks of model math in PyTorch/JAX after claiming the path
  is kernelized.

Blockers:

- A new model lands as one monolithic file containing IO, weights, calibration,
  graph capture, and hardware-specific forward branches.
- A pipeline imports a dependency that is not needed to launch kernels.
- A frontend stores `.data_ptr()` from a tensor it does not own persistently.

## 13. New Model Acceptance Contract

A new model PR is not complete with only a working script.

Required files or explicit non-applicability:

- Config or documented direct-instantiation path.
- Routing registration for each validated `(config, framework, arch)`.
- One frontend per `(model, framework, hardware)`.
- One compute pipeline per `(model, hardware)` when graph-captured.
- Weight spec or documented checkpoint adapter.
- Attention spec/backend selection when the model uses attention.
- Calibration/precision spec when FP8, FP4, NVFP4, INT8, or similar is used.
- Model-specific usage docs.
- Focused tests for import/routing, missing symbols, shape guards, and
  first-light correctness against a reference.

Blockers:

- The PR only adds an example script and private local paths.
- The model requires an undocumented checkpoint bundle format.
- The model changes shared kernel names or shared frontend APIs to fit one
  integration.
- First output has no reference comparison or finite-value smoke.

## 14. Precision, Calibration, Cache, And Correctness

Required:

- State dtype and precision mode must be explicit.
- Precision changes must state the reference path.
- Calibration caches must be invalidated when scale semantics change.
- Device scale buffers must be persistent if kernels read them later.
- Quantized paths need cosine, token-match, or domain-specific validation
  against the relevant reference.
- Cache reuse must document exactly what is cached and when it is invalidated.

Blockers:

- Activation scale writer and reader use different keys.
- Refit or packing updates one weight representation but hot graph reads
  another.
- Prompt/state changes do not reset temporal or KV cache state.
- Speed is reported without correctness.
- Low correctness is accepted without comparing to the right reference noise
  floor.

Minimum correctness evidence:

- VLA actions: action cosine or max error vs reference, plus task smoke.
- LLM/VLM: logits cosine, argmax match, or short generation sanity.
- TTS/audio: mel cosine, token match where applicable, subjective notes only as
  secondary evidence.
- Diffusion/video: latent cosine, output smoke, and finite checks.
- Kernel only: numerical unit test vs PyTorch or NumPy reference.

## 15. Tests

At least one relevant test class should exist for every non-doc PR:

- Import smoke test.
- Missing-symbol fail-fast test.
- Shape guard test.
- Config/routing guard test.
- Numerical unit test for new kernels or quantizers.
- Backward compatibility test for API changes.
- Build/import test for dedicated modules where possible.

Tests must not:

- Depend on private paths.
- Insert absolute local bundle paths by default.
- Require multi-GB checkpoints unless marked as local smoke scripts, not CI.
- Assert APIs that implementation does not expose.
- Pass by catching all exceptions without checking the message.
- Skip everything in the default environment without testing import/fail-fast.
- Depend on test ordering or global CUDA state from another test.
- Use network downloads in default tests.

Commands:

```bash
python -m compileall <changed_python_files>
PYTHONPATH=. pytest -q <focused_tests>
```

## 16. Documentation

Required:

- Public user-facing behavior must be documented.
- Build flags must match CMake exactly.
- Import paths must match implementation.
- Kernel/module ownership must match CMake.
- New model docs must state supported `(framework, hardware, precision)` tuples
  and unsupported combinations.
- New hardware docs must state exact GPU/SM, toolkit assumptions, build flags,
  and validation status.
- New precision docs must state scale granularity, cache key semantics,
  reference path, and invalidation rules.
- Performance tables must state hardware, toolkit, checkpoint, precision,
  shape, and metric definition.
- Docs must distinguish wall-clock latency from replay-only latency.
- Experimental modes must be labeled as experimental.

Blockers:

- Docs tell users to import a non-existent API.
- Docs mention stale kernels that were removed.
- Docs claim fallback behavior but implementation fail-fasts, or the reverse.
- Docs show private paths, hostnames, local usernames, or internal checkpoint
  locations.
- Benchmark claims have no command or reproducibility note.
- Docs say a feature is default when it is gated, or gated when it is default.

## 17. Privacy And Repository Hygiene

Run a diff-scoped scan. Avoid broad scans that report unrelated existing files.

```bash
changed="$(git diff --name-only origin/main...HEAD)"

rg -n "<absolute-local-path>|<internal-host>|ssh://|private-path|private-host" $changed
rg -n "secret|password|apikey|api_key|access_token|auth_token|bearer |ChatGPT|Claude|Co-authored-by:|Generated by|pdb|breakpoint|TODO|FIXME|XXX" $changed
```

Not every hit is a blocker:

- Placeholder paths such as `<path-to-checkpoint>` are fine.
- `127.0.0.1` in server docs is fine.
- `TODO` is only a blocker if it is introduced in maintained code without a
  tracked plan.

Blockers:

- Real private paths.
- Secrets or tokens.
- Internal hostnames.
- Local usernames.
- AI-generated traces or comments.
- Debug print in library code.
- Temporary benchmark output, local notebooks, or cache files.

## 18. Performance Claims

Required:

- State the exact command.
- State hardware, toolkit, driver if relevant, checkpoint, precision, shape,
  batch, sequence length, warmup count, and iteration count.
- Compare against the correct baseline.
- Report correctness alongside speed.

Blockers:

- Speedup claim without correctness.
- Comparing replay-only latency to wall-clock baseline.
- Using different prompt/image/shape between baseline and optimized path.
- Reporting a cherry-picked median without iteration count or distribution.
- Claiming a kernel is used while runtime still calls PyTorch for that op.

## 19. Serving And Execution Contract

Required:

- `exec/` stays mechanism-only.
- Serving policy stays in `serving/` or user hosts.
- GPU/model state remains owned by frontend/pipeline.
- Serving may own request metadata, sessions, protocol, and serialization.
- Server-only dependencies must be optional and documented.

Blockers:

- Server dependencies are imported by core `flash_rt` on import.
- Protocol/session/cache policy is added to `exec/`.
- Serving example mutates model internals without using a documented model API.

## 20. Future Hardware And Platform Readiness

Required:

- Keep generic runtime names vendor-neutral unless the API is truly vendor-only.
- Vendor-specific code belongs under explicit hardware/backend paths, CMake
  flags, or dedicated modules.
- Do not put vendor-only assumptions into model-independent Python APIs unless
  the API name or docs say so.
- Hardware capability checks must be explicit and fail fast.
- Platform-specific build behavior must be isolated in CMake or platform docs.
- Future backend placeholders are acceptable only in docs or disabled stubs
  that raise clear "not implemented" errors.

Blockers:

- A generic module depends on vendor-specific headers, compiler flags, shared
  libraries, or Python packages without a gate.
- A hardware abstraction is widened by adding vendor-specific optional
  parameters to common APIs.
- Unsupported hardware silently falls back to a slower or numerically different
  path.
- Platform-specific shell commands or library names are baked into public docs
  without alternatives.

## 21. Merge Checklist

Before merge, verify:

- [ ] PR is based on a recent `origin/main` or conflict risk is understood.
- [ ] Diff scope matches PR title/body.
- [ ] No unrelated files, generated artifacts, or local outputs.
- [ ] No private paths, secrets, AI traces, or debug code.
- [ ] Existing behavior is preserved or the intentional behavior change is
      documented and tested.
- [ ] Shared helper changes include call-site inventory and old-path evidence.
- [ ] Default build path still works.
- [ ] Gated build path works if a gated feature is added.
- [ ] Required symbols and bindings match CMake ownership.
- [ ] New kernels are named and located according to ownership rules.
- [ ] New model code follows frontend/pipeline split.
- [ ] Optional dependencies are lazy imported.
- [ ] Core install/import does not gain server, eval, training, or model-only
      dependencies.
- [ ] Fail-fast errors are clear and early.
- [ ] Tests pass or skip cleanly in default environment.
- [ ] Hardware-specific behavior is isolated.
- [ ] Unsupported hardware/platform combinations fail clearly.
- [ ] Precision/cache/graph changes have correctness evidence.
- [ ] Docs match actual API, flags, modules, and behavior.
- [ ] Performance claims include reproducible commands and correctness.

## 22. Reviewer Comment Template

Use this shape for PR comments:

```markdown
Review result: <Accept / Request changes / Blocker / Needs hardware validation>.
Risk class: <R0/R1/R2/R3/R4>.

Before merge, these items need to be fixed:

- [ ] <blocking item with file/symbol and why it matters>
- [ ] <expected fix>
- [ ] <doc/test/build item>

What I verified:

- `<command>`: passed
- `<command>`: failed with <short reason>

Hardware or fixtures I could not validate:

- <target or fixture>: <reason>
```

Avoid vague comments such as "please make it cleaner". Name the file, symbol,
expected behavior, and whether the item blocks merge.

## 23. Follow-Up Versus Blocking

Usually acceptable as follow-up:

- Wider hardware benchmark when structure is isolated and target hardware has
  contributor or CI evidence.
- Extra performance tuning after correctness passes.
- More polished docs when existing docs are accurate enough to run.
- Renaming non-public internal helper variables.
- Consolidating similar model-specific helpers after both are correct.
- Adding another hardware/framework tuple not claimed by the PR.

Must block:

- Failing tests in the default environment.
- Import failure for `flash_rt`.
- Undefined or duplicate symbols.
- Wrong public docs.
- Wrong API path in docs/tests.
- Host sync in claimed hot path.
- Private paths or secrets.
- Unvalidated route for an existing hardware path.
- Incorrect precision/cache behavior.
- New default behavior without old-path evidence.
- New dependency leaking into core import or default install.
- New model path without routing/import/correctness evidence.
- Kernel names, paths, or pybind symbols that hide model/hardware ownership.
- `exec/` or common runtime changes that include scenario policy.
