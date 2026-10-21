# Qwen3-4B BF16：FlashRT vs HuggingFace 精度修复纪要

日期：2026-08-04  
硬件：NVIDIA GeForce RTX 4090  
代码：`flashrt/flash_rt/frontends/torch/qwen3_4b_rtx.py`  
对照：`transformers.Qwen3ForCausalLM`，随机权重共享 `state_dict`（无 Hub 下载）

## 问题

全 FlashRT 内核路径（FA2 + 融合 norm/RoPE/QKV/SwiGLU）在短序列上可过阈值，长序列失败：

| seq | full-seq cosine（修复前） | ≥ 0.999 |
|---:|---:|:---:|
| 8 | ~0.99945 | 通过 |
| 1024 | **~0.961** | 失败 |

`seq=256` 在修完 RoPE 后仍稳定略低于阈值（多 seed 均值 ~0.99878）。

## 根因（按影响排序）

### 1. RoPE `inv_freq` 被 bf16 量化（主因，长序列）

HuggingFace 侧 `model.to(dtype=bfloat16)` 会把 `Qwen3RotaryEmbedding.inv_freq` **buffer 一并铸成 bf16**。  
`rotary_emb.forward` 里虽有 `.float()`，只能把**已经量化过**的频率抬回 fp32，无法恢复丢失的尾数。

FlashRT 原先用 **fp32 `inv_freq`** 建 cos/sin 表，数学上更“准”，但与 HF 实际推理不一致。位置越大，频率误差累积越明显 → `S=1024` cosine 掉到 ~0.96。

**修复：** 建表时对 `inv_freq` 做与 HF 相同的 bf16 round-trip：

```python
inv = (1.0 / (theta ** (arange(0, hd, 2) / hd))).to(bfloat16).float()
```

仅对齐 RoPE 表后，`S=1024` full cosine 可到 **1.000000**（其余路径不变时）。

### 2. 融合 QKV GEMM 的 bf16 归约结构（主因，中等序列）

HF 用三个独立 `nn.Linear`（q / k / v）。FlashRT 把权重 `cat` 后做一次融合 GEMM，tile/归约顺序不同，逐层 ULP 差会放大。

隔离实验（喂相同 HF 输入）：

| 路径 | 单层 attn 输出 | 端到端 `S=256` |
|---|---:|---:|
| 分离 q/k/v + `F.linear` | 1.000000 | **1.000000** |
| 融合 qkv + `F.linear` | — | ~0.9988 |
| 分离 q/k/v + FlashRT `_gemm` | — | ~0.99926 |

分层观测：Attention 输出全程仍高（~0.9997），整体 cosine 随层下降，融合 QKV 是端到端掉点的结构来源。

**修复：** 默认 `use_sep_qkv=True`（三次独立 GEMM，仍走 FlashRT `_gemm`）。

### 3. 其余“HF 语义”开关（leave-one-out）

在 **RoPE 表已 bf16 对齐** 的前提下，对五个开关做 leave-one-out（S=256 / 1024，阈值 ≥ 0.999）：

| 关掉的开关 | S=256 | S=1024 | 结论 |
|---|:---:|:---:|---|
| `use_sdpa` | 过 | 过 | **无实质影响**（可回 FA2） |
| `use_hf_norm` | 过 | 过 | **无实质影响**（可回 fvk norm） |
| `use_hf_qkn_rope` | 不过 | 不过 | **必需** |
| `use_sep_gateup` | 过 | 过 | **无实质影响**（可回 fused silu） |
| `use_sep_qkv` | 不过 | 过 | **S=256 必需**；S=1024  alone 可过 |
| 全部关掉 (`all_fvk`) | 不过 | 不过 | — |

最小充分集：`use_hf_qkn_rope=True` + `use_sep_qkv=True`（其余默认 False）即可在 8–2048 过阈值。

当前默认：

| 开关 | 默认 | 路径 |
|---|---|---|
| `use_sdpa` | False | FA2 |
| `use_hf_norm` | False | fvk rms / residual_add_rms_norm |
| `use_hf_qkn_rope` | **True** | HF q/k norm + rotate_half RoPE |
| `use_sep_gateup` | False | fused gu + silu_mul |
| `use_sep_qkv` | **True** | 分离 q/k/v GEMM |

**仍走 FlashRT：** embedding、各投影 GEMM（含分离后的 q/k/v）、FA2 attention、fvk norm / silu_mul。

#### 对精度“有明确影响”的

- **RoPE `inv_freq` bf16 对齐** — 不做则长 seq 直接崩（~0.96）。
- **`use_hf_qkn_rope`** — 关掉则 256/1024 都掉到阈值下。
- **`use_sep_qkv`** — 关掉则 `S=256` 不过 0.999。

#### 对精度无实质影响的（正确 RoPE 下）

- **`use_sdpa`**、**`use_hf_norm`**、**`use_sep_gateup`** — leave-one-out 仍 ≥ 0.999。

## 修复后精度（默认路径）

| seq | full cosine | 说明 |
|---:|---:|---|
| 8 | ≥ 0.999 | 通过 |
| 64 / 128 | ≥ 0.999 | 通过 |
| 256 | ≥ 0.999（10 seed 全过，min≈0.99921） | sep QKV 后 |
| 512 / 1024 / 2048 | ≥ 0.999（1024 可达 1.0） | RoPE 对齐后 |

测试入口：`flashrt/tests/test_qwen3_4b_hf_cosine.py`（含 `S=1024`）。

## 延迟（精简默认后，RTX 4090）

相对“五个开关全开”的精度路径（S=1024 e2e ≈ 70.3 ms），精简后重新快于/接近 HF：

| seq | HF e2e (ms) | FlashRT 默认 (ms) | HF/FlashRT |
|---:|---:|---:|---:|
| 512 | 36.4 | 34.9 | 1.04× |
| 1024 | 67.4 | 67.0 | 1.01× |

全 fvk（含关掉 `use_hf_qkn_rope` / `use_sep_qkv`）长/中序列 cosine 可能再次掉到阈值以下。

## 结论

1. 与 HF 比精度时，必须复现 HF 的 **bf16 `inv_freq` RoPE 表**，而不是“更精确的 fp32 频率”。  
2. **融合 QKV** 与 HF 三个 Linear 的 bf16 归约不同，是中等 seq（如 256）掉点的结构原因；默认改为分离 QKV。  
3. 默认路径 = 精度敏感步走 HF/torch 语义 + GEMM/embedding 留 FlashRT；速度与精度需显式用开关权衡。
