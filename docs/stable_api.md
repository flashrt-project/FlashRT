# FlashRT — Stable Public API

This document enumerates every symbol that is part of FlashRT's public
stability contract. Symbols listed here will not be removed or have their
signatures changed without a major version bump.

Symbols **not** listed here (internal modules, private functions, class
internals) may change between minor releases.

---

## Top-level (`flash_rt`)

```python
import flash_rt

flash_rt.__version__   # str, e.g. "2.2.0"
flash_rt.load_model    # → VLAModel
flash_rt.VLAModel      # inference wrapper
```

### `flash_rt.load_model(...)`

```python
def load_model(
    checkpoint: str,
    framework: str = "torch",       # "torch" | "jax"
    num_views: int = 2,             # 1, 2, or 3
    autotune: int = 3,              # 0=off, 3=default, 5+=thorough
    recalibrate: bool = False,
    weight_cache: bool = True,      # JAX only
    config: str = "pi05",           # "pi05" | "pi0" | "groot" | "groot_n17" | "pi0fast" | "motus" | "wan22_ti2v_5b"
    device=None,                    # reserved
    # Pi0-FAST-specific:
    decode_cuda_graph: bool = False,
    decode_graph_steps: int = 80,
    max_decode_steps: int = 256,
    hardware: str = "auto",         # "auto" | "thor" | "rtx_sm120" | "rtx_sm89" | "rtx_sm87"
    # GROOT-specific:
    embodiment_tag: str | None = None,
    action_horizon: int | None = None,
    # Pi0.5 torch-specific:
    use_fp4: bool = False,
    fp4_layers: tuple[int, ...] | None = None,
    use_awq: bool | None = None,
    awq_alpha: float = 0.5,
    use_p1_split_gu: bool | None = None,
    num_steps: int | None = None,
    vision_pool_factor: int | None = None,
    vision_num_layers: int | None = None,
    cache_frames: int | None = None,
    # Frontends with an FP8/BF16 switch:
    use_fp8: bool = True,
    # Pi0.5 torch RTX SM120/SM89 opt-in:
    use_fp16: bool = False,
) -> VLAModel
```

Returns a `VLAModel` wrapping the appropriate frontend for the detected
(or explicitly specified) GPU architecture.

- `decode_cuda_graph`, `decode_graph_steps`, `max_decode_steps` apply to
  Pi0-FAST.
- `embodiment_tag` and `action_horizon` apply to GROOT.
- `use_fp4`, `fp4_layers`, `use_awq`, `awq_alpha`, and
  `use_p1_split_gu` apply to the Pi0.5 torch NVFP4 encoder path.
- `num_steps`, `vision_pool_factor`, `vision_num_layers`, and
  `cache_frames` apply only to frontends that expose those constructor
  parameters today. The Pi0.5 torch RTX/Orin frontend validates
  `vision_pool_factor in {1, 2, 4}`, `vision_num_layers in [1, 27]`, and
  `cache_frames >= 1`.
- `use_fp8=False` disables FP8 where the selected frontend exposes a
  BF16 fallback; unsupported frontends ignore it.
- `use_fp16=True` selects the opt-in Pi0.5 torch RTX SM120/SM89 full-FP16
  CUDA Graph path. It requires `use_fp8=False` and is currently valid
  only for `config="pi05"`, `framework="torch"`, and
  `hardware in {"rtx_sm120", "rtx_sm89"}`.
- `config="motus"` is a beta RTX SM120 frontend. It expects a Motus
  checkpoint plus Wan and VLM checkpoint paths supplied to the Motus
  quickstart/frontend; see `docs/motus_usage_beta.md`.
- `config="wan22_ti2v_5b"` is an RTX SM120 official-pipeline Wan2.2
  baseline. It exposes `set_prompt(prompt, negative_prompt=...)` and
  `infer(mode="t2v"|"i2v", width=..., height=..., frames=..., steps=...,
  shift=..., guide_scale=..., seed=..., teacache=False,
  teacache_threshold=..., teacache_start_step=...,
  teacache_end_step=..., teacache_cache_device=...)`; `predict()` is not
  part of this video-generation API. See `docs/wan22_usage.md`.
- `config="groot_n17"` is registered for `framework="torch"` and
  `hardware="rtx_sm120"`. This route validates the N1.7 DiT
  self/cross-attention path on the vendored FA2 backend.

### `flash_rt.VLAModel`

```python
class VLAModel:
    def set_prompt(self, *args, **kwargs): ...
    def infer(self, *args, **kwargs): ...
    def predict(self, images, prompt=None, state=None) -> np.ndarray: ...
    def warm_state_prompt_buckets(self, images, prompt, states) -> list[int]: ...
    def recalibrate(self) -> None: ...

    @property
    def framework(self) -> str: ...
    @property
    def prompt(self) -> str | None: ...
```

- `set_prompt(*args, **kwargs)` — delegate prompt setup to the selected
  frontend. GROOT N1.7 currently uses `set_prompt(aux=..., prompt=...)`,
  where `aux` contains the captured Qwen3-VL setup tensors consumed by
  the N1.7 calibration path.

- `infer(*args, **kwargs)` — delegate inference to the selected frontend.
  GROOT N1.7 currently uses
  `infer(state_normalized, initial_noise=..., use_dit_graph=...)` and
  returns normalized actions.

- `predict(images, prompt, state)` — run one inference step.
  `images`: list of `(224,224,3)` uint8 numpy arrays, or a dict with
  `"image"` / `"wrist_image"` / `"wrist_image_right"` keys.
  `prompt`: required on first call, reused on subsequent calls if `None`.
  `state`: optional robot state array. `predict()` attaches it to the
  observation as `"state"` when the caller passes image lists, while preserving
  an explicit `"state"` already present in a dict observation.
  For frontends whose `set_prompt()` accepts `state`, `predict()` refreshes the
  prompt prefix when either the prompt text or that prompt-state value changes.
  Returns `np.ndarray` of shape `(action_horizon, action_dim)`.

  `state` is part of the VLA observation schema; each model encodes it through
  its own reference contract:
  - Pi0 uses a continuous state token in the action-expert suffix.
  - Pi0.5 encodes state as openpi-compatible discretized prompt tokens:
    `Task: <prompt>, State: <openpi state bins>;\nAction: `. The bins use
    OpenPI's `np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1`
    convention: normalized in-range values usually become 0..255, while
    values below -1 become -1. RTX/Thor torch frontends and the JAX Thor
    Pi0.5 frontend accept `state` in `set_prompt()` and through `predict()`.
    Same-length state prompt updates reuse the captured graph; recurring
    RTX prompt lengths reuse the cached pipeline instead of rebuilding.
  - Pi0-FAST encodes state in the FAST token prefix.
  - GROOT N1.6 consumes proprioceptive state from `obs["state"]`; if omitted,
    the backend uses zeros.
  - GROOT N1.7 currently uses the lower-level
    `normalize_state(...)` + `infer(state_normalized, initial_noise=...)`
    contract rather than the image-list `predict()` contract.

- `warm_state_prompt_buckets(images, prompt, states)` — Pi0.5 RTX torch
  helper for realtime loops that pass changing state through the prompt.
  It calls the selected frontend's bucket warmup hook, using `images` as
  the calibration/capture observation and each item in `states` as a
  representative robot state. The return value is the sorted list of
  warmed prompt lengths. The method preserves the OpenPI state prompt
  text format; it does not zero-pad or otherwise rewrite state tokens.
  A recurring prompt token length reuses its cached runtime pipeline, but
  a previously unseen state-token length still pays one first-time bucket
  build/capture cost. Serving code should keep state serialization stable
  (fixed state dimension and numeric rounding/precision policy) and prewarm
  representative states from the deployment range, for example reset,
  mid-rollout, near-goal, and a few recorded rollout observations whose
  discretized values may tokenize to different lengths.

- `recalibrate()` — clear FP8 calibration cache and force re-calibration
  on the next `predict()` call.

---

## Hardware dispatch (`flash_rt.hardware`)

```python
from flash_rt.hardware import detect_arch, resolve_pipeline_class
```

### `detect_arch() -> str`

Returns `"thor"`, `"rtx_sm120"`, `"rtx_sm89"`, or `"rtx_sm87"` based on
the current CUDA device's compute capability. SM87 is currently supported
only for `config="pi05", framework="torch"`.

### `resolve_pipeline_class(config, framework, arch)`

Lazily imports and returns the concrete frontend class for the given
`(config, framework, arch)` triple. Used internally by `load_model`.
Motus beta is registered for `(config="motus", framework="torch",
arch="rtx_sm120")`.
GROOT N1.7 RTX is registered for `(config="groot_n17",
framework="torch", arch="rtx_sm120")`.
Wan2.2 TI2V-5B is registered for `(config="wan22_ti2v_5b",
framework="torch", arch="rtx_sm120")`.

### `_PIPELINE_MAP`

```python
flash_rt.hardware._PIPELINE_MAP: dict[tuple[str, str, str], tuple[str, str]]
```

The dispatch table mapping `(config, framework, arch)` to
`(module_path, class_name)`. **External plugins may mutate this dict**
at import time to register new models — see
[Plugin Model Template](plugin_model_template.md).

---

## AttentionBackend protocol (`flash_rt.hardware.backend`)

```python
from flash_rt.hardware.backend import (
    AttentionBackend,    # Protocol class
    AttentionBackendBase,  # Optional base with accessor defaults
    AttentionSpec,       # Full model attention specification
    SiteSpec,            # One attention site descriptor
)
```

### `SiteSpec`

```python
@dataclass
class SiteSpec:
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    max_q_seq: int
    max_kv_seq: int | None = None     # None → self-attention
    batch_axis: int = 1
    sliding_window: int | None = None  # reserved for SWA models
    causal: bool = False
    extra: dict = field(default_factory=dict)
```

### `AttentionSpec`

```python
@dataclass
class AttentionSpec:
    sites: dict[str, SiteSpec]
    def add_site(self, name: str, **kwargs) -> AttentionSpec: ...
    def site(self, name: str) -> SiteSpec: ...
```

### `AttentionBackend` (Protocol)

```python
class AttentionBackend(Protocol):
    def sites(self) -> tuple[str, ...]: ...
    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]: ...
    def run(self, site: str, layer_idx: int, q_seq: int, *,
            kv_seq: int | None = None, stream: int = 0) -> int: ...
    def head_dim(self, site: str) -> int: ...
    def num_q_heads(self, site: str) -> int: ...
    def num_kv_heads(self, site: str) -> int: ...
```

---

## Reference implementations (reusable, not stable)

These concrete backends are available for reuse by plugins. Their
**existence** is stable, but their internal signatures may change
between minor releases. Plugins that subclass or call into these
should pin a minor version.

```python
from flash_rt.hardware.rtx.attn_backend       import RtxFlashAttnBackend
from flash_rt.hardware.rtx.attn_backend_groot import RtxFlashAttnBackendGroot
```

Both classes are framework-neutral — used by torch and jax frontends
alike. The old names ``TorchFlashAttnBackend`` /
``TorchFlashAttnBackendGroot`` are kept as deprecated module-level
aliases and will be removed in the next major version.

---

## Core utilities (`flash_rt.core`)

```python
from flash_rt.core.cuda_buffer import CudaBuffer
from flash_rt.core.cuda_graph import CUDAGraph
from flash_rt.core.quant.calibrator import load_calibration, save_calibration
```

These are used by both pipelines and frontends. Their public API is
stable; internal helper functions are not.

---

## Native extension modules

FlashRT ships two pybind11 Python extension modules:

```python
from flash_rt import flash_rt_kernels   # always present
from flash_rt import flash_rt_fa2       # RTX (SM80/86/89/120) only
```

### `flash_rt.flash_rt_kernels`

The main kernel module — hand-written CUDA code plus cuBLASLt/
CUTLASS wrappers for memory-bound ops (norm, activation, fusion,
FP8 quant, residual, gate-geglu, true-silu, etc.) and Thor-specific attention
(`fvk.attention_qkv_fp16`). Binary name pattern:
`flash_rt_kernels.cpython-<abi>.so`, ~3 MB.

All `fvk.<symbol>(...)` calls seen in pipeline code live here.
Signatures are internal — plug-ins should go through the
`AttentionBackend` protocol, not call `fvk.*` directly.

Motus beta SM120 kernels are built into `flash_rt_kernels` when CMake is
configured with `-DFLASHRT_ENABLE_MOTUS=ON` (currently the default).
They do not require a separate runtime kernel library directory.

---

## Runtime utilities

```python
from flash_rt.runtime.rtc import (
    ActionChunkAdapter,
    CallablePolicyAdapter,
    AsyncChunkRunner,
    RTCStats,
)
```

RTC-lite is a beta inference scheduling utility for action-chunk
policies. It does not change model numerics or calibration; it only
serves action chunks at a fixed controller rate while a background worker
prepares the next chunk. See `docs/rtc_lite_design.md`.

### `flash_rt.flash_rt_fa2`

Vendored Flash-Attention 2 v2.7.4.post1 (forward only, fp16 + bf16,
SM80-family SASS). Binary name pattern:
`flash_rt_fa2.cpython-<abi>.so`, ~135 MB. Only built when
`GPU_ARCH ∈ {80, 86, 87, 89, 120}`. Exposes:

```python
flash_rt_fa2.fwd_fp16(
    Q, K, V, O, softmax_lse,
    softmax_lse_accum=0, o_accum=0,   # splitkv scratch ptrs; 0 disables splitkv
    *,
    batch, seqlen_q, seqlen_k,
    num_heads_q, num_heads_kv, head_dim,
    q_strides, k_strides, v_strides, o_strides,   # 3-tuples (batch, row, head) in elements
    softmax_scale=1.0,
    num_sms=0,                         # required for splitkv heuristic
    stream=0,
)
flash_rt_fa2.fwd_bf16(...)            # same signature, bfloat16 dtype
```

All pointer args are int device pointers (`tensor.data_ptr()`).
Pipeline code should go through `RtxFlashAttnBackend` (which calls
this module internally) rather than invoking it directly — direct
use is unstable and may change without notice. `RtxFlashAttnBackend`
selects between this module and the pip `flash-attn` wheel via the
`FVK_RTX_FA2` env var (default `"1"` = use vendored FA2; `"0"` =
fallback to `flash_attn.flash_attn_func`). The backend name reflects
the hardware family (RTX), not the frontend framework — the **same**
backend instance serves both torch and jax frontends.

Thor (SM110) builds do **not** produce this module — attention on
Thor uses `flash_rt_kernels.attention_qkv_fp16` (cuBLAS-decomposed)
because FA2's Ampere tile shapes aren't tuned for Thor's unified
LPDDR memory model. Code importing `flash_rt_fa2` must therefore
guard for `ImportError` on Thor deployments, or stay inside the
`AttentionBackend` protocol which handles the dispatch transparently.

---

## Directory structure (post-refactor)

```
flash_rt/
├── __init__.py              # load_model, VLAModel
├── api.py                   # load_model implementation
├── core/                    # shared utilities (CudaBuffer, CUDAGraph, calibrator)
├── hardware/
│   ├── __init__.py          # detect_arch, _PIPELINE_MAP, resolve_pipeline_class
│   ├── backend.py           # AttentionBackend protocol
│   ├── rtx/                 # rtx attention backends (hardware primitives only)
│   └── thor/
│       ├── attn_backend.py        # Thor FMHA wrapper
│       ├── attn_backend_groot.py  # GROOT-specific Thor attention
│       └── shared_primitives.py   # CLOSED SET: model-agnostic Thor helpers only
│                                  #   (_gpu_*, _measure_scale_gpu,
│                                  #    siglip_forward, encoder_forward,
│                                  #    encoder_forward_calibrate)
├── models/
│   ├── pi05/
│   │   ├── pipeline_thor.py       # Pi0.5 Thor compute (postln_project, decoder_*)
│   │   └── pipeline_rtx.py        # Pi0.5 RTX Pi05Pipeline class
│   ├── pi0/
│   │   ├── pipeline_thor.py       # Pi0 Thor decoder fns
│   │   └── pipeline_rtx.py        # Pi0 RTX Pi0Pipeline class
│   ├── pi0fast/
│   │   └── pipeline.py            # DEPRECATED PATTERN: Thor+SM120 runtime fork
│   │                              # do NOT copy this style for new models
│   └── groot/
│       ├── pipeline_thor.py       # GROOT Thor pipeline
│       ├── pipeline_rtx.py        # GROOT RTX pipeline
│       └── embodiments.py         # per-embodiment MLP slots
└── frontends/
    ├── torch/
    │   ├── pi05_thor.py    (Pi05TorchFrontendThor)
    │   ├── pi05_rtx.py     (Pi05TorchFrontendRtx)
    │   ├── pi0_thor.py     (Pi0TorchFrontendThor)
    │   ├── pi0_rtx.py      (Pi0TorchFrontendRtx)
    │   ├── pi0fast.py      (Pi0FastTorchFrontend)    DEPRECATED — Thor+RTX hybrid
    │   ├── groot_thor.py   (GrootTorchFrontendThor)
    │   └── groot_rtx.py    (GrootTorchFrontendRtx)
    └── jax/
        ├── pi05_thor.py    (Pi05JaxFrontendThor)
        ├── pi05_rtx.py     (Pi05JaxFrontendRtx)
        ├── pi0_thor.py     (Pi0JaxFrontendThor)
        ├── pi0_rtx.py      (Pi0JaxFrontendRtx)
        └── pi0fast.py      (Pi0FastJaxFrontend)      DEPRECATED — Thor+RTX hybrid
```

**Naming convention** (established 2026-04, stage 8 unified-pipeline-layout refactor):

* **Every (model, hardware) compute path is its own file**:
  `models/<m>/pipeline_<hw>.py` where `<hw>` ∈ {`thor`, `rtx`}.
  No `pipeline.py` (no-suffix default entry) is allowed.
* **Every (model, framework, hardware) IO path is its own frontend file**:
  `frontends/<fw>/<m>_<hw>.py` with class `<Model><Fw>Frontend<Hw>`
  (e.g. `Pi05TorchFrontendThor`, `Pi05TorchFrontendRtx`).
* **No runtime hardware forks** (`if self._has_sm100`, `hasattr(fvk, ...)`):
  if a model needs different code on Thor vs RTX, those are separate files.
* **`hardware/<hw>/shared_primitives.py` is a closed set** of
  model-agnostic helpers. Model-specific forwards/decoders go into
  `models/<m>/pipeline_<hw>.py`, never into `shared_primitives.py`.
* **`_PIPELINE_MAP` is one-to-one**: each `(model, framework, hw)` tuple
  routes to exactly one frontend file/class.

**Known historical exception** (do NOT copy for new models):
* `pi0fast` ships as a single `pipeline.py` with 14+ `if self._has_sm100`
  branches and a single multi-hw frontend file. This in-file SM-fork
  pattern is retained for the existing Pi0-FAST implementation only;
  new models should follow the standard `(model, framework, hw)` split.

---

## Declarative weight loading (stage 7)

Thor frontends' per-layer weight-loading loops are expressed as
`ModelWeightSpec` objects in private spec modules next to the frontend.
The public surface is three things:

```python
from flash_rt.executors.weight_loader import (
    Item, LayerBlock, ModelWeightSpec, WeightLoader,
)
from flash_rt.executors.torch_weights import (  # torch side
    SafetensorsSource, DictSource,
    Cat, FusedQKV, FusedGateUp,
    ToFp16, ToFp32, T, tT, InterleaveQK, Quant, Mul,
    Attr, TensorList, FlatCat,
)
from flash_rt.executors.jax_weights import (  # jax side
    OrbaxDictSource,
    Transpose, Astype, Contiguous, JaxQuant,
    NumpyAttr, NumpyList, CudaBufferAttr, CudaBufferFlat,
)
```

Stability contract: the classes above and their constructor signatures
are public. Adding new sink/transform/composite classes is
backwards-compatible; existing ones will not be removed or renamed
without a major version bump.

**Spec file naming**: `flash_rt/frontends/{torch,jax}/_<model>_thor_spec.py`,
each exporting a `build_spec() -> ModelWeightSpec`. Shared block
builders live in `_thor_spec_common.py` per framework.

**Convention for scale lists**: spec items that quant set
`scale_into="_<group>_scales"`; frontends wrap these into device
tensors after `loader.run()`:

```python
self._enc_w_dev = torch.tensor(self._enc_w_scales,
                               dtype=torch.float32, device='cuda')
```

See [`docs/adding_new_model.md`](adding_new_model.md) for the end-to-end
model adaptation walkthrough, and
[`docs/plugin_model_template.md`](plugin_model_template.md) for
registering an external-plugin model via `_PIPELINE_MAP`.

---

## Adaptation / extension guides

When adding a new model or kernel, read these in order:

1. [`docs/adding_new_model.md`](adding_new_model.md) — end-to-end
   walkthrough for wiring a new VLA model into FlashRT on Thor
   (AttentionSpec → WEIGHT_SPEC → pipeline forward → frontend →
   calibration → graph capture → registration → tests).
2. [`docs/calibration.md`](calibration.md) — FP8 weight/activation
   scale mechanics, `alpha = act_scale × weight_scale` invariants,
   calibration cache format, and the four historical bugs every new
   model's `_calibrate` should guard against.
3. [`docs/kernel_fusion.md`](kernel_fusion.md) — the 93 public
   `fvk` kernels grouped by purpose, current production fusion
   patterns, what does and does not fuse, and the catalog of failed
   optimizations (OPT-3 / OPT-5 / v1.5-B2 / v1.5-B4) to avoid.
