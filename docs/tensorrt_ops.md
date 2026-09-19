# FlashRT Operators for TensorRT (Jetson AGX Thor, SM110)

The [TensorRT backend](tensorrt_backend.md) ships pi0.5 as whole stages. This
document covers the layer underneath: FlashRT's Thor kernels as model-free
operators, so a host can place them in its own graph at whatever granularity it
wants — a TensorRT network, [TensorRT Edge-LLM](tensorrt_edgellm.md), or a
runtime of its own.

Everything here is measured on one Jetson AGX Thor, JetPack 7.2, MAXN, with the
openpi `pi05_libero` checkpoint and two cameras. Both arms of every comparison
are timed by `trtexec` with the same flags.

## 1. The C ABI — `backends/tensorrt/kernels/frt_ops.h`

Raw device pointers, no model semantics, one CUDA stream, `0` on success.

| call | what it runs |
|---|---|
| `frt_nvfp4_linear` | normalize → quantize to NVFP4 → block-scaled GEMM → epilogue |
| `frt_nvfp4_mlp` | normalize → quantize → gate/up GEMM with the activation fused into its epilogue → down GEMM → epilogue |
| `frt_fa4_gqa_hd256` | FlashAttention-4, head_dim 256, grouped queries over one KV head |
| `frt_fa4_mha_hd72` | FlashAttention-4, head_dim 72, multi-head |
| `frt_nvfp4_linear_workspace`, `frt_nvfp4_mlp_workspace` | scratch bytes the two need |
| `frt_nvfp4_num_variants`, `frt_nvfp4_variant_name` | the tile tables, for tuning at your own shape (§5) |
| `frt_fa4_load`, `frt_fa4_default_scale`, `frt_set_pdl` | load the FA4 modules, the `1/sqrt(head_dim)` FlashRT uses, programmatic dependent launch |

Modes are enums, not strings: `frt_norm_mode` (none / RMS / LayerNorm),
`frt_epilogue_mode` (none / accumulate / bias+residual), `frt_gate_mode`
(interleaved GeGLU / bias+GELU).

NVFP4 is e2m1 with a UE4M3 scale every 16 elements along K: packed weights are
`uint8 [N, K/2]`, block scales `uint8 [N, K/16]`.

The kernels are CUDA C++ with CUTLASS, except FlashAttention-4, which is CuTe
DSL compiled ahead of time into the library — no Python, no JIT at run time.

## 2. The plugins — `backends/tensorrt/plugins/ops/`

Three `IPluginV3` shells over exactly those calls, in ONNX domain `flashrt`,
version `1`. They are in the same library as the stage plugins and are
registered by the same `getCreators()` export, so `--dynamicPlugins` or
`EDGELLM_PLUGIN_PATH` gives a host both.

NVFP4 codes and block scales travel in `INT32` tensors of `ceil(bytes / 4)`
elements: TensorRT constants do not take `UINT8`, and `INT32` storage passes
bytes through unchanged.

### FlashrtNvfp4Linear

`out fp16 [M, N] = epilogue(quantize(norm(x)) · wᵀ)`

| | |
|---|---|
| required inputs | `x` fp16 `[M, K]`, `w_packed` blob `[N, K/2]`, `w_sfb` blob `[N, K/16]` |
| optional inputs | in this order, present when `opt_mask` has the bit: `gamma` (0), `beta` (1), `awq_inv_s` (2), `bias` (3), `residual` (4) |
| attributes | `N`, `K`, `norm_mode`, `epilogue`, `variant`, `opt_mask` (int32), `eps` (float32) |
| output | fp16 `[M, N]` |

### FlashrtNvfp4Mlp

`out fp16 [M, D] = epilogue(down(activation(gate_up(quantize(norm(x))))))`

| | |
|---|---|
| required inputs | `x` fp16 `[M, D]`, `gate_up_packed`, `gate_up_sfb`, `down_packed`, `down_sfb` (blobs) |
| optional inputs | `gamma` (0), `beta` (1), `awq_inv_s` (2), `gate_up_bias` (3), `down_bias` (4), `residual` (5) |
| attributes | `D`, `H`, `norm_mode`, `gate_mode`, `gate_variant`, `down_variant`, `epilogue`, `opt_mask`, `eps` |
| output | fp16 `[M, D]` |

The gate/up GEMM writes NVFP4 and its block scales straight from the epilogue,
so the hidden activation is never materialized in fp16. That is the operator's
reason to exist; §3 measures what it is worth.

### FlashrtFa4Attention

| | |
|---|---|
| inputs | `q` fp16 `[Sq, NH*HD]`, `k`, `v` fp16 `[Sk, HD]` (`mode` 0) or `[S, NH*HD]` (`mode` 1) |
| attributes | `mode` (0 = head_dim 256 GQA over one KV head, 1 = head_dim 72 MHA), `NH`, `head_dim`, `batch`, `scale` (float32; `0` means `1/sqrt(head_dim)`) |
| output | fp16, shaped like `q` |

The FA4 modules load on the first shape change, never during graph capture.

## 3. Measured against TensorRT

### 3.1 Method

Both arms are `trtexec --useCudaGraph --noDataTransfers --iterations=200
--avgRuns=10`, and the number is its median GPU compute time. The TensorRT arm
is a subgraph cut out of the openpi Thor tutorial's `model_fp8_nvfp4.onnx` with
`onnx.utils.Extractor` — its quantization nodes, its weights, its layouts,
only the boundaries are ours (`backends/tensorrt/tests/trt_native_ops.py` names
them). Boundaries match: both arms start at the already normalized activation
and neither adds the residual.

A process on Thor lands in a fast or a slow clock mode and stays there, so the
two arms alternate over several rounds and each is reported at its fastest
round. Timing in Python instead costs a variable few microseconds of host time
per replay, which is why the tuning sweep (§5) uses it for ordering and every
number below comes from `trtexec`.

### 3.2 Thor is bandwidth bound

Two reference points, same harness:

| | median |
|---|---|
| one elementwise kernel over 64 elements | 4.3 µs |
| one elementwise kernel over `[526, 2048]` fp16 (4.3 MB moved) | 12.8 µs |

So an operator at these shapes is priced in memory passes, and about 4 µs of
any number below is the harness.

### 3.3 Operators

| operator, at the shape pi0.5 runs it | FlashRT | TensorRT | ratio |
|---|---|---|---|
| encoder FFN, `M=526, D=2048, H=16384`, NVFP4 both sides | **309.8 µs** | 408.0 µs | **1.32×** |
| encoder output projection, `526×2048×2048`, quantize + GEMM | 30.8 µs | **28.9 µs** | 0.94× |
| SigLIP FFN, `M=512, D=1152, H=4320` (FlashRT NVFP4, export FP8) | **49.4 µs** | 52.2 µs | **1.06×** |
| SigLIP attention, head_dim 72, `2×16×256×72` | **39.6 µs** | 130.2 µs | **3.29×** |

The export writes attention as MatMul/Softmax/MatMul, so that last row only
says FlashAttention-4 beats *that graph*. Against TensorRT's own `Attention`
operator (`backends/tensorrt/tests/attention_arm.py` builds it, five rounds):

| attention | FlashRT FA4 | TensorRT `Attention` | ratio |
|---|---|---|---|
| SigLIP, head_dim 72, `2×16×256×72` | **39.6 µs** | 67.6 µs | **1.71×** |
| encoder, head_dim 256 GQA, `1×8×526×256`, one KV head | **91.6 µs** | 114.0 µs | **1.25×** |

### 3.4 Why the bare GEMM does not win

The output projection is the one operator TensorRT takes. Holding everything
else fixed and scaling `N`:

| `N` | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| FlashRT | 22.7 µs | 22.2 µs | 25.9 µs | 31.0 µs |

(The alternating run of §3.3 gives 30.8 µs for that last engine; both are the
same measurement, taken minutes apart.)

About 22 µs of it does not depend on `N`: that is the separate pass that reads
the fp16 activation and writes NVFP4 plus its block scales. Counting bytes at
this shape, FlashRT moves 7.9 MB — activation in, NVFP4 and its scales out,
both back in, weights, result — against TensorRT's 6.7 MB, because TensorRT
folds the quantization into its GEMM prologue and never lands the quantized
activation:

| | bytes moved | time | effective |
|---|---|---|---|
| FlashRT, quantize pass + GEMM | 7.9 MB | 31.0 µs | 254 GB/s |
| TensorRT, GEMM with a quantizing prologue | 6.7 MB | 28.9 µs | 231 GB/s |
| the elementwise probe of §3.2 | 4.3 MB | 12.8 µs | 336 GB/s |

Both arms are close to what the memory system delivers for what they move — the
difference between them is the extra round trip, not the arithmetic. It costs
2.1 µs rather than the 4.7 µs the bytes alone suggest, because the quantized
activation is written and read back immediately and much of it is still in L2.

The same accounting explains the FFN in the other direction. TensorRT's FFN
materializes the hidden activation in fp16 — 526×16384 written and read again,
34 MB per layer — while FlashRT's gate/up epilogue emits NVFP4 with its block
scales, about 9 MB. The 25 MB difference is roughly 100 µs at the rates above;
the measured difference is 98 µs. That is the 1.32×.

So the ordering is not an accident of tiles, and it does not move much with
them (§5): **an isolated GEMM is where a fused quantization prologue wins, and a
block is where keeping the activation in NVFP4 wins.** It is also why
FlashRT's plugins are cut at fusion boundaries.

## 4. Granularity: operator, layer, stage

The encoder is available at all three. Measured with the same weights, the same
`Se=526`, under an outer CUDA graph:

| granularity | per block | 18 layers | bitwise vs FlashRT |
|---|---|---|---|
| operators (`bench_granularity.py`) | attention + projection + FFN, **no QKV, no RoPE, no normalization** | 7.580 ms | — |
| layer plugins (`engine_encoder_layer_chain.py`) | the whole layer | 8.620 ms | yes |
| stage plugin (`engine_encoder_stage.py`) | all 18 layers in one node | 8.507 ms | yes |

Read it this way: the operator arm is a *lower bound*. What it leaves out — the
QKV projection, RoPE, two normalizations, the K/V writes — is about a fifth of
a layer's bytes, roughly 21 MB against the 87 MB the three operators move, so
of order 1.5 ms over 18 layers at the rates in §3.4. The arm already spends 89%
of what the stage spends while doing that much less, and adding the rest back
would put an operator-level encoder near 9 ms against the stage's 8.5.

What the layer boundary itself costs is not the same everywhere, and it is
worth knowing before picking one — all three are bitwise equal to the stage:

| part | layer or step plugins | one stage plugin | cost of cutting |
|---|---|---|---|
| SigLIP, 27 layers | 3.576 ms | 3.627 ms | none measurable |
| prefix encoder, 18 layers | 8.620 ms | 8.507 ms | +1.3% |
| action decoder, 10 denoising steps | 12.95 ms | 11.67 ms | +11% |

The decoder is where it shows: its stage keeps the next step's weights coming
in on a side stream and chains programmatic dependent launches through the
step, and a plugin boundary per step ends both. SigLIP's layers are large
enough relative to their boundaries that the difference disappears into the
run-to-run spread. So layer granularity is a real option for the vision tower
and the encoder, and an expensive one for the decoder.

Two things a host cannot get back at operator granularity:

- **NVFP4 between operators.** A plugin boundary is an fp16 tensor. FlashRT's
  row quantizer is built for rows up to 2048 wide, because the fused path never
  needs to quantize a hidden activation — the previous epilogue already emitted
  one. An FFN cut into two GEMM plugins cannot be expressed at all at
  `H=16384`, let alone at the same speed.
- **Scheduling across operators.** Programmatic dependent launch between the
  kernels of a block, and the side stream that pumps the next layer's weights
  into L2, both live inside a stage.

Mixing vendors in one graph — TensorRT's GEMM next to FlashRT's fused block —
is possible at graph level, and §3.4 bounds what it buys: about 2 µs per bare
projection, against the ≥1 ms per encoder pass that operator granularity costs.
Inside a kernel it is not possible in either direction; both sides' GEMMs are
closed.

## 5. Tuning tiles at your own shape

Every NVFP4 GEMM has a table of tile shapes. `frt_nvfp4_num_variants(kind)`
says how many, `frt_nvfp4_variant_name(kind, idx)` describes one, and the index
goes into `variant` / `gate_variant` / `down_variant`. Which table an index
means follows from the mode and the epilogue, which is why the kinds are named
after kernels (`FRT_VARIANT_GEMM`, `FRT_VARIANT_GATE_BIAS_GELU`,
`FRT_VARIANT_DOWN_BIAS_RES`).

`bench_ops.py --sweep TAG` walks them at that operator's own shape, with the
recorded weights and activations, and reports the best:

```bash
python backends/tensorrt/tests/bench_ops.py build/tensorrt/libflashrt_trt_pi05.so \
    encoder_all.safetensors siglip_all.safetensors \
    --only siglip_mlp --sweep siglip_mlp --json sweep.json
```

What it found for pi0.5 on Thor, against the defaults the end-to-end tuning
left behind. These are the operators at the boundary §3.3 measures (no
normalization, no residual), so the down GEMM is the plain one:

| operator | table | default | best at this shape | change |
|---|---|---|---|---|
| encoder FFN down GEMM | `GEMM` | 8 — `128x256x256` | **6** — `128x256x128 cluster1x1x1` | −3% |
| encoder output projection | `GEMM` | 1 — `128x256x128 cluster2x1x1` | **6** — `128x256x128 cluster1x1x1` | −2% |
| SigLIP up GEMM | `GATE_BIAS_GELU` | 2 — `128x128x128` | **6** — `256x128x256 cluster2x1x1` (2-SM) | −3% |
| SigLIP down GEMM | `GEMM` | 0 — `128x128x128 cluster2x1x1` | **38** — `256x256x256 cluster2x1x1` (2-SM) | −15% |

With the residual epilogue the SigLIP FFN uses the other down table
(`DOWN_BIAS_RES`), where the sweep picks 4 (`256x128x256 cluster2x1x1`) over
the base 0, and its up GEMM picks 4 over 2.

The SigLIP FFN is the one that moves the ordering: it was 0.88× of TensorRT
before the sweep and is 1.06× after. The others move by a couple of percent,
which is §3.4's point — at these shapes the tile is not what decides.

Tiles are per shape, so the stage path keeps the ones its own end-to-end tuning
picked; these are the operator path's.

## 6. Reproduce

Build the plugin library as in [tensorrt_usage.md](tensorrt_usage.md), then,
with the recordings from `backends/tensorrt/tools/reference/`:

```bash
# FlashRT arm: one engine per operator, plus the engines trtexec will re-time.
# The tiles are the ones §5 measured for these shapes.
python backends/tensorrt/tests/bench_ops.py build/tensorrt/libflashrt_trt_pi05.so \
    encoder_all.safetensors siglip_all.safetensors --save-engines ops/ \
    --set encoder_mlp_plain.down_variant=6 --set encoder_o_plain.variant=6 \
    --set siglip_mlp_plain.gate_variant=6 --set siglip_mlp_plain.down_variant=38

# TensorRT arm: subgraphs cut from the export, and the native Attention graphs
python backends/tensorrt/tests/extract_trt_ops.py --onnx .../model_fp8_nvfp4.onnx --out ops/
python backends/tensorrt/tests/attention_arm.py --out ops/

# both arms, alternating, same trtexec flags
python backends/tensorrt/tests/ab_ops.py --ops-dir ops/ \
    --plugin build/tensorrt/libflashrt_trt_pi05.so --rounds 3
python backends/tensorrt/tests/ab_ops.py --ops-dir ops/ \
    --plugin build/tensorrt/libflashrt_trt_pi05.so --native attention \
    --only siglip_attn,encoder_attn --rounds 5

# granularity
python backends/tensorrt/tests/bench_granularity.py build/tensorrt/libflashrt_trt_pi05.so \
    encoder_all.safetensors --depth 18
```

`extract_trt_ops.py` writes the subgraphs next to a link to the export's
external weight file, so its initializers keep resolving; the export directory
itself is usually read-only.

## 7. Numerical contract

The operators call the same kernels, in the same order, as the pi0.5 stages, so
an operator and the matching slice of a stage produce the same bits. What the
tests pin today:

- the stage and layer plugins are bitwise equal to FlashRT's library, eagerly
  and under an outer CUDA graph (`backends/tensorrt/tests/run_regression.sh`);
- the operator plugins are checked for finiteness at the pi0.5 shapes by
  `bench_ops.py`.

A bitwise check of an operator against the corresponding slice of a stage needs
the recording tools to export the intermediate tensors at those boundaries;
that is not in the dumps yet.

One place where operator granularity changes the arithmetic rather than the
speed: `FRT_EPI_ACCUM` adds into its output buffer, so a plugin whose residual
arrives as a separate tensor copies it in first. Inside a stage the residual is
already there.
