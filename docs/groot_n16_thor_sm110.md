# GR00T N1.6 × FlashRT（Jetson Thor SM110）— 权威适配与优化文档

> 自权重对齐开始的完整记录：HF 数值对齐 → 12 个真实 bug 修复 → parity 通路 →
> 逐轮性能优化（130 ms → 28.5 ms）。配套两份独立主题文档见 §10。
>
> 本文是 N1.6 的唯一权威文档（one authoritative doc per model/platform）。
> 提交 PR 时以本文 + §11 文件清单为准。

---

## 0. 状态总览

**完成并通过仿真验证。** FlashRT 与 HF eager 基线数值和动作行为
一致，仿真可完成任务。

| 指标 | 数值 | 备注 |
|---|---|---|
| GPU 推理（e2e，中位数） | **~28.5 ms** | 4 步 DiT、2 相机、252×252、T=50 |
| 服务侧 total | ~32 ms | 含预处理 2.8 ms + 解码 0.3 ms |
| HTTP 往返 / RCP 侧 | ~38 ms | 20 请求 median 39 / p95 40 / 全 finite |
| 精度 vs HF eager | cos **0.999933** / maxd **0.059** | 去归一化动作空间 |
| 起点（HF eager） | ~110–150 ms | NVIDIA 官方 eager ~126 ms |
| Thor 实测带宽 | **252–255 GB/s**（官方 ~273 GB/s，~93%） | 见 §8 |

推理参数与 HF 完全一致（4 步 flow-matching、252×252、T=50、bf16 数学），
**未为提速改动任何推理超参**；加速全部来自 kernel 化 / 量化 / 图融合。

---

## 1. 背景与模型架构

GR00T N1.6-3B = Eagle-Block2A-2B-v2 backbone + AlternateVLDiT action head：

| 组件 | 结构 | 说明 |
|---|---|---|
| 视觉塔 | SigLIP2 NaFlex，27 层，D=1152，HD=72，16 头 | 252×252 → 18×18=324 patch/视图，2 视图打包成 648 token 单序列 |
| LLM | Qwen3，16 层，D=2048，GQA 16Q/8KV，HD=128 | ★ checkpoint 只截断到 16 层（非 28） |
| mlp1 | pixel-unshuffle + LN + 2×Linear | 视觉 → LLM 维度投影 |
| Action head | AlternateVLDiT，32 层，D=1536，NH=32，HD=48，FF=6144 | Sa = T+1 = 51；4 步 flow-matching；奇数层 self-attn、偶数层 cross-attn（交替 text/image KV） |
| 动作 | action_dim=128（padding），so101 实际仅前 ~6 维有效 | T=50（padded max） |

关键架构事实（对性能分析至关重要）：

- **SigLIP 是 cross-view full attention**：HF(sdpa) 对打包的 648 token 做全注意力，
  **不是** per-view。NaFlex 代码里按 img_idx 的 `seq_len_list` 分段**只有
  flash_attention_2 后端消费，sdpa 静默忽略**；Eagle `extra_kwargs` 里的
  `attn_implementation="flash_attention_2"` 是死代码（未传入加载）。训练/部署实际都是 sdpa。
- **DiT 是权重带宽主导**：32 层 × 4 步，每步重读 ~415 MB fp4 权重；M=Sa=51 极小，
  GEMM 完全 weight-bandwidth-bound。
- **HF eval 图像链输出 252×252**（256→0.95 crop→Eagle smart_resize 到 14 的倍数），
  不是 224。

---

## 2. 适配路线总览（自权重对齐开始）

```
阶段一  权重对齐 & HF 基线      → 发现 transformers>=5 静默权重损坏，修复；
                                   建立 HF eager 数值基线
阶段二  12 个真实 bug 定位修复   → tokenization / 分辨率 / cross-view / patch 序 /
                                   FMHA 发散 / Qwen3 发散 / adaLN chunk 序 /
                                   图捕获野指针 / 校准 / prompt 切换 / 空闲重置 …
阶段三  parity 通路落地          → SigLIP/Qwen3/DiT 全 HF 原生 bf16 + CUDA graph，
                                   与 HF eager cos≈1.0（~111 ms）
阶段四  服务侧优化              → 预处理 11→2.5 ms（apply_state 直连 / 线程池 /
                                   GPU 归一化 / JPEG→cv2）
阶段五  FA4 + NVFP4 + 全 kernel 化 → 130 ms → 28.5 ms（§6 逐轮）
```

---

## 3. 阶段一：权重对齐与 HF 基线

### 3.1 transformers>=5 静默权重损坏（第一个核心问题）

详见配套文档 `docs/groot_transformers5_weight_corruption.md`。摘要：

- `from_pretrained` 加载本身正确（1106/1106，0 missing），但收尾阶段
  `_initialize_missing_keys` 因 `_auto_class` 为空 → `is_remote_code()==False`
  → 无视 `_is_hf_initialized` → 用 `_init_weights` **重新随机化整个 SigLIP2
  视觉塔（282 张量）+ mlp1.1/3**。
- 症状：策略"不看图"，只跟随 state；黑图 vs 真实图 Δ 被噪声淹没。
- 修复（一行）：`type(self)._auto_class = "AutoModel"`（`Gr00tN1d6.__init__`，
  `post_init()` 之前）。
- 预防：权重完整性校验（`verify_weight_integrity()`） 抽样 ~12 张量 live vs safetensors
  比对，不一致拒绝启动（日志 `[weight-check]`）。

### 3.2 HF eager 基线

HF eager 基线（独立部署仓库）：`Gr00tPolicy`（AutoModel/AutoProcessor）纯 eager，
作为数值/行为的 ground truth。所有 FlashRT 精度都对它度量。

---

## 4. 阶段二：真实 Bug 清单（12 项，已修）

上游 N1.6 前端把 **openpi 系（Pi0/Pi0.5）的视觉/核假设**直接套到 N1.6，而 N1.6 的
HF 实际行为不同；叠加若干实现 bug。全部修复集中在
`flash_rt/frontends/torch/groot_thor.py`。

| # | 问题 | 根因 | 修复 | 验证 |
|---|------|------|------|------|
| 1 | backbone 特征与 HF 正交 | **tokenization**：前端裸 `encode(prompt)`；HF 用 Eagle chat template（system/user 头 + formalize + 每视图图像块，168→196 tokens） | 前端 `build_input_ids` 逐 token 复现 HF | `torch.equal` |
| 2 | 视觉输入分布错 | **分辨率**：HF eval 链输出 252×252，前端固定 224 | `image_size=252`；aux 在 252 用 processor 默认链 + 复刻 smart_resize（PIL bicubic） | 像素 maxΔ 0.002 |
| 3 | SigLIP 深层发散（L26 cos 0.86） | **注意力范围**：HF(sdpa) 做 cross-view full attention，前端做 per-view（见 §1） | SigLIP 改 cross-view（batch=1 单序列） | 27 层逐层 cos≥0.999 |
| 4 | patch-embed cos 0.966 | **patch 展平序**：openpi 系 `(C,ph,pw)`；HF NaFlex `convert_images_to_patches` 是 `(ph,pw,C)` | permute 改 `(0,2,3,4,5,1)` | patch-embed cos 1.0 |
| 5 | 648-seq 注意力静默错 | **strided FMHA kernel 在非 2 幂 seq + 真实数据下发散**（256/512 与随机数据单测均正常，极隐蔽） | parity 模式 SigLIP attention 走 torch sdpa | kernel 单测 vs sdpa |
| 6 | Qwen3 输出与 HF 正交 | **CKernelQwen3 与 HF 在真实序列上发散**（同输入对拍 block 级逐步衰减） | parity 模式改跑 HF 原生 `Qwen3Model`（bf16、sdpa、图捕获） | vlln cos 0.9998 |
| 7 | 重捕获后 replay 出 NaN | **野指针**：Qwen3 图捕获时 vlln LN 引用的 `vlln_w/vlln_b` 是 `_capture_all_graphs` 局部张量，函数返回即被 allocator 复用 | 改持久属性 `_g_vlln_w/_g_vlln_b`；加 `_g_vlln_buf` 有限性自检 + 重捕获兜底 | 重捕获回归 |
| 8 | DiT chunk 震荡、动作失真 | **最终 adaLN chunk 顺序反**：HF `proj_out_1` 为 `(shift, scale)`，前端写成 `(scale, shift)` | 交换 | chunk 单调平滑，maxΔ 0.018 |
| 9 | 首帧后动作饱和/NaN | **单帧 FP8 校准过窄**：捕获由首帧触发，scale 只测该帧 | 服务侧「当前帧 + 7 合成帧」`calibrate(percentile=99.9)` 多帧校准 | 饱和消失 |
| 10 | prompt 切换报错/卡死 | prompt 烘焙进图，建图后 `set_prompt` 被拒 | 检测 prompt 变化 → `reset_graph_runtime()` + `set_prompt` + 重捕获 | 多 prompt A/B |
| 11 | 空闲后首帧垃圾 | Thor 空闲重置使已捕获图失效（见配套文档） | replay 后 `_g_vlln_buf` 有限性自检，非有限即重捕获重试 | 压测 |
| 12 | prompt 切换重捕获 device-side assert | `reset_graph_runtime` 未删 DiT 静态缓冲/索引（`_dit_txt_idx` 等），重捕获后按旧 Se 越界 | 加入 stale 列表，重捕获时重建 | 切换+空闲 live 全 finite |

---

## 5. 证伪/排除项（**非** bug，勿再追）

- **FP8/FP16 精度**：FP32/BF16 参考给出相同分叉形态，精度不是上述任何一项的根因
  （parity DiT 用 bf16，与 HF eager 完全一致）。
- **GELU 近似（exact/tanh）**：两种近似结果相同。
- **position_embed resize**：16×16→16×16 为恒等；252 的 18×18 resize 与 HF 同参一致。
- **每块 sinusoidal position embed「缺失」**：**假 bug**。本模型
  `diffusion_model_cfg.positional_embeddings=None`，每块本就不加 pe；按 diffusers
  默认补加反而引入大偏差（曾误修，已撤）。
- **「kernel DiT 独立不收敛」**：**归因错误**。噪声不收敛由 backbone 失真（#3/#4）+
  DiT 实现 bug（#8）共同造成；无独立 kernel DiT 问题。旧文档
  `groot_n16_dit_kernel_nonconvergence.md` 结论作废，已删除。
- **训练/部署注意力不一致（NVIDIA 侧）**：不存在。flash_attention_2 为死代码，
  训练与部署均为 sdpa cross-view。
- **DiT cross-KV fp4**：实测动作精度恶化 5×（cos 0.9992 / maxd 0.105）且无时延收益，
  cross 特征喂所有 cross 层 —— 已排除（commit b5afeecc）。

---

## 6. 阶段四/五：性能优化全记录（130 → 28.5 ms）

### 6.0 优化前服务侧预处理优化（阶段四）

- `processor()` 全调用（5.7 ms，含用不到的 VLM/tokenizer 路径）替换为
  `state_action_processor.apply_state` + 零填充 —— 与 processor 的 state 输出
  **bit 一致**（28 组真实/合成 state 验证）。
- 双视图 albumentations + PIL bicubic 变换改线程池并行（4.2→2.1 ms）。
- SigLIP 像素归一化移 GPU（uint8 H2D + fp32 除法，数学不变）；mlp1 权重布局缓存
  （去掉每帧 `.T.contiguous()`）。
- JPEG 解码 PIL→cv2（`pb_utils._jpeg_to_rgb`，带 PIL 回退）：2.13→0.94 ms/图，
  解码像素 bit 一致（同为 libjpeg）。
- 结果：预处理 11→2.5 ms，parity 整链 ~111→~73 ms（4 步）。

### 6.1 Roofline / 天花板分析（阶段五的指导）

- Thor 实测带宽 **252–255 GB/s**（官方 ~273 GB/s，~93%）；GPU GPC 1575 MHz /
  NVD 1692 MHz 均满载，无节流。
- DiT 32 层权重 fp4 ~415 MB/步 × 4 步 = 1.66 GB → 纯权重地板 ~6.6 ms；加注意力/
  norm/quant/激活，DiT ~15 ms 为该配置接近地板的水平。
- 结论：**DiT/Qwen3 fp4 GEMM 已带宽受限**（L2-cold 实测 170–239 GB/s），tile 重调
  无空间；减步数伤行为（见 §6.4）；**28.5 ms 已接近 2 相机 252px T=50 的实际下限**。

### 6.2 FA4 + NVFP4 移植（参考 Chameleon/HyVLA/N1.7 #163）

- **FA4（FlashAttention-4 CuTe-DSL，`flash_rt/hardware/thor/fa4_backend.py`）用于
  SigLIP**：SigLIP 本就是 cross-view full attention（648 token 单序列），FA4
  causal=False 为 sdpa 精确替换；encoder graph 10.3→9.0 ms，动作 cos 1.000000。
  开关 `FLASHRT_N16_FA4=1`（显式开启，缺失自动回退）。
- **FA4 不用于 DiT/Qwen3**：小 seq（51/208）下 FA4 0.042–0.394 ms ≫ sdpa 0.013–0.015 ms。
- **NVFP4（W4A4 CUTLASS）用于 DiT**：block GEMM 走
  `quantize_fp4_dynamic_sfa_fp16` + `cutlass_fp4_sq_fp16`（per-16 block scale、动态激活
  量化、无校准）。关键事实：**M=51 冷 L2 流式下 fp4 GEMM 比 bf16 cuBLAS 快 2–5×**
  （此前 L2-hot 微基准误判为无收益）。开关 `FLASHRT_N16_DIT_FP4`。

### 6.3 融合 epilogue kernel 移植（上游 N1.7 #163，本地重编译）

从上游移植（`csrc/`，加入 `fp4_kernels_obj` + `fp4_bindings`）：

- `gemm/fp4/cutlass_fp4_gemm_bias_bf16_sm100.cu` — bias / bias+residual /
  bias+tanh-GELU+fp4out 三种 epilogue。
- `fused_fp4/dit_norm_fp4_sfa.cu` — AdaLN norm 与 no-affine LN 直接输出 fp4+SFA。
- `quantize/quantize_fp4_sfa_bf16.cu` — bf16 向量化量化。

DiT 链每层仅 8 kernel（无层间逐元素流量）：**DiT 36.6→15.7 ms**，
精度不变（vs HF cos 0.999995）。

### 6.4 DiT 减步数实验（**已回退，勿用于生产**）

| 步数 | e2e | 精度/行为 |
|---|---|---|
| 4（生产） | ~28.5 ms | 基线 |
| 2（`FLASHRT_N16_DIT_STEPS=2`） | ~46 ms* | ⚠️ 仿真反馈动作犹豫、夹爪收起慢（Euler 大步长对速度场积分欠冲，夹取等速度突变阶段显现；离线录制帧诊断不可见）→ **回退** |
| 1（`FLASHRT_N16_DIT_STEPS=1`） | ~36 ms* | maxΔ 0.055，更激进 |

\* 减步数同时减少权重读，但**行为退化**，生产保持 4 步。该开关保留作实验用。

### 6.5 全 kernel 化三轮（37.5 → 28.5 ms）

- **第一轮 — Qwen3 融合**（`FLASHRT_N16_QWEN3_FP4=1`，显式 fast profile）：16 层全部
  q/k/v/o/gate/up/down 走 W4A4 融合 GEMM（residual 原地更新）。
  - 新写两个 bf16 输入 producer kernel（`csrc/fused_fp4/`，与 torch 两步链 **bit 一致**）：
    `rms_norm_weight_fp4_sfa_bf16`（加权 RMSNorm→fp4 直出）、
    `silu_mul_fp4_sfa_bf16`（silu·mul→fp4 直出）。
  - torch LN/quant/silu/mul 五趟 → 两个融合 kernel：10.9→9.2 ms。
  - **fused per-head RMSNorm + rotate-half RoPE**（`qk_norm_rope_rotate_half_bf16`）：
    −2.2 ms。
  - **fused GQA attention**：torch `enable_gqa` 会退回非融合 math 路径（每层 2 个
    fp16 GEMM + 显式 softmax）；改 `repeat_interleave` 展开 KV 走融合后端：−2.0 ms。
  - Qwen3 12.7→5.0 ms；vs HF cos 0.999986。
- **第二轮 — SigLIP encoder fp4**（`FLASHRT_N16_SIGLIP_FP4=1`，显式 fast profile）：LN producer
  用 scale=w−1/shift=b 表达 affine；q/k/v 分离 GEMM 写连续缓冲直喂 FA4；o/ffn 残差进
  epilogue；fc1 N 与 fc2 K 4304→4352 pad（pad 维全零端到端无贡献）。encoder 9.0→6.9 ms。
  全开时动作精度不降（vs HF cos 0.999988）。
- **第三轮 — SigLIP embeddings 入图**：推理 34→**28.5 ms**。NaFlex window split 对
  固定 image_size 是静态 gather、antialias 位置编码 resize 只依赖静态形状；setup 时
  预计算 gather 索引 + resize 后位置编码，per-frame 只做
  patchify+patch_embedding+pos_add+gather 并入 SigLIP 图。对 HF embeddings 前向
  **bit 一致（max diff 0.0）**，省去每帧 antialias interpolate + unfold/im2col。
  图捕获失败时自动回退「encoder-only 图 + eager embeddings」。

### 6.6 已排除的优化路径（dead-end，勿重复尝试）

- **torch.compile（DiT/Qwen3，max-autotune）**：无收益（43→41 ms），且曾出现图捕获
  数值异常，勿在生产启用。
- **torch fp8（`torch._scaled_mm`）**：Thor 仅支持 tensorwise scale + bf16 输出；
  DiT 权重 per-channel 量化精度好（cos 0.99993），但图捕获后量化/反量化开销吃掉权重
  减半收益（29.6 > bf16 21.5 ms），不采用。
- **legacy fp8 kernel 快路径（`--fp8`）**：实测 siglip 32.8 / qwen3 12.8 / dit 41.5
  = 91 ms，全面慢于 torch bf16，且含 #5/#6 数值问题。**N1.6 生产请用默认 `--no-fp8`**。
- **SigLIP FFN torch 级 fp4**：量化开销 47→53 ms，弃用；SigLIP down-proj K=4304 需 pad。
- **NVFP4 扩展到 SigLIP encoder（早期 torch 级）**：162 Linear fp4 后 9.0→20.7 ms
  （M=648 激活流量 12.6× 于 DiT，无融合 kernel 时量化开销主导）且 postln cos 降至 0.94
  → torch 级 fp4 只对 M 小、权重流量主导的 DiT 有效。（后被 §6.5 的融合 kernel 路线取代。）
- **DiT attention head_dim 48→64 padding**：注意力仅 1.28 ms，pad 浪费 ~0.3 ms，
  需自写 head_dim-48 kernel，性价比低，不做。
- **DiT GEMM tile 重调**：已带宽受限（§6.1），无空间。

---

## 7. 最终架构与配置

### 7.1 数据流（parity + 三层 NVFP4 + FA4，显式 fast profile）

```
obs ──> preprocess（apply_state 直连 + 线程池图像变换）           ~2.8 ms
      ──> SigLIP 图（embeddings 入图 + 27 层 fp4 encoder + FA4）      ~6.5 ms
      ──> mlp1 / pixel-unshuffle（bf16 torch）                      ~0.4 ms
      ──> Qwen3 图（16 层 fp4 + fused norm/rope/GQA）               ~5.0 ms
      ──> DiT 图（4 步 × 32 层 fp4 fused epilogue）                 ~15.2 ms
      ──> denormalize_actions（复刻 HF decode_action）              ~0.3 ms
```

全部 CUDA Graph 捕获；三层各自独立 tier 开关，可单独回退。

### 7.2 精度 tier 与开关

| 开关 | 默认 | 作用 |
|---|---|---|
| `FLASHRT_N16_DIT_FP4` | 关 | DiT NVFP4 融合链；设为 `1` 开启 |
| `FLASHRT_N16_QWEN3_FP4` | 关 | Qwen3 NVFP4 融合层（含 fused norm/rope/GQA）；设为 `1` 开启 |
| `FLASHRT_N16_SIGLIP_FP4` | 关 | SigLIP encoder fp4 层；设为 `1` 开启 |
| `FLASHRT_N16_FA4` | 关 | SigLIP FA4 注意力；设为 `1` 开启，缺失时回退 |
| `FLASHRT_N16_DIT_STEPS` | 4 | flow-matching 步数（**生产勿改**，减步伤行为） |
| `parity` | 关 | HF 原生 parity 通路（非 legacy kernel 快路径）；直接构造 frontend 时显式开启 |

精度汇总（去归一化动作 vs HF eager）：

| 配置 | cos | maxd |
|---|---|---|
| 全开（显式 fast profile） | 0.999933 | 0.059 |
| Qwen3 fp4 单独 | 0.999986 | 0.023 |
| DiT fp4 单独 | 0.999995 | 0.012 |

### 7.3 生产稳定性（继承 N1.7 三层防护）

见 `docs/thor_gpu_idle_reset_workaround.md`：GPU 心跳保活
（`FLASHRT_GPU_KEEPALIVE=0.15`）+ 空闲重捕获守卫（`FLASHRT_GRAPH_IDLE_REINIT_S=2`）
+ 有限性自检。对抗 Thor GPU 空闲 ~200–300 ms 后驱动重置破坏已捕获 graph 的缺陷。

---

## 8. 带宽与天花板分析

| 项 | 值 |
|---|---|
| Thor 实测带宽 | 252–255 GB/s（读/拷贝、fp16/fp32、512/1024MB 一致） |
| 官方标称 | ~273 GB/s（LPDDR5X），实测 ~93% |
| GPU 时钟 | GPC 1575 MHz / NVD 1692 MHz（满载，无节流） |
| DiT fp4 权重地板 | 1.66 GB / 253 GB/s ≈ 6.6 ms（4 步） |

> 注：早期手测曾报 63 GB/s，系把 float32 **元素数当成字节数**少算 4× 的统计错误；
> 修正后与 roofline.py 一致（~254 GB/s）。roofline 脚本：
> `.qoder/skills/flashrt-model-adaptation/scripts/roofline.py --measure-bw`。

**天花板结论**：DiT 15.2 ms（4 步权重重读，带宽主导）是剩余大头，无法在不改推理
超参（减步数）的前提下进一步压缩。**28.5 ms 为该任务配置的实际下限附近。**

---

## 9. 验证方法（可复现）

- **数值一致性**：离线 `cos / maxd` vs HF eager（`/tmp/hf_act_ref.npy` 参考），
  以及 live A/B（FlashRT vs HF eager 逐关节 Δ）。
- **bit 一致验证**：fast embeddings、fused norm/rope/silu-mul producer 均对
  torch 两步链 / HF 前向 bit 一致（max diff 0.0）。
- **稳定性**：20 请求 median/p95/全 finite；stress_gaps（间隙压测）+ selfcheck。
- **带宽**：`roofline.py --measure-bw`。
- 时延分解：`FLASHRT_GROOT_TIMING=1` 输出 `[timing] preprocess/infer/total`。

---

## 10. 相关文件与配套文档

**本 PR 改动的核心文件**（`git diff main..HEAD`）：

- 前端：`flash_rt/frontends/torch/groot_thor.py`（核心，+1526 行）
- attention backend：`flash_rt/hardware/thor/attn_backend_groot.py`
- 模型管线：`flash_rt/models/groot/pipeline_thor.py`
- 新 kernel：`csrc/fused_fp4/{dit_norm_fp4_sfa,silu_mul_fp4_sfa_bf16}.{cu,cuh}`、
  `csrc/gemm/fp4/cutlass_fp4_gemm_bias_bf16_sm100.{cu,cuh}`、
  `csrc/quantize/quantize_fp4_sfa_bf16.{cu,cuh}`、`csrc/fp4_bindings.cpp`
- 权重转换（开发调试工具，不含于本 PR）：前端已内置完整的 HF→FlashRT layout 变换（transpose、QKV fuse），直接加载 HF 原始 safetensors 即可推理。离线审计脚本 `convert_groot_n16_hf_checkpoint.py` 仅用于 parity 对拍时定位 weight mapping 问题。
- 文档：本文 + `docs/groot_transformers5_weight_corruption.md`

**配套文档（独立主题）**：

- `docs/thor_gpu_idle_reset_workaround.md` — Thor GPU 空闲重置缺陷与 CUDA Graph 防护。
- `docs/groot_transformers5_weight_corruption.md` — transformers>=5 权重静默损坏。

---

## 11. 遗留工作（猜想/未测）

- [ ] **N1.7 同清单巡检**：N1.7 视觉同为 Eagle/SigLIP2 NaFlex，猜想存在同款
      cross-view/patch 序问题；N1.7 仿真"可用"但未与其 HF 基线数值对拍。
- [ ] **`FLASHRT_N16_DIT_STEPS=2/1` 仿真验证**：离线数值已测，闭环任务成功率未测
      （N=2 已见夹爪犹豫，默认不用）。
- [ ] **parity 模式小时级长稳压测**：prompt 切换 + 空闲自愈已测，长稳未测。
- [ ] **SigLIP embeddings 之外的进一步 kernel 化**：已接近地板，收益有限。
