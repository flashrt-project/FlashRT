# MiniMax-M3 on DGX Spark (GB10) — FlashRT 适配 Handoff

更新: 2026-06-13。分支 `minimax-spark` @ ab9c6a9。Session 入口 memory:
`project_minimax_m3_spark_adaptation.md`。

## 当前状态 (2026-06-13 收尾)

DONE: M3 427B 单 Spark E2E 跑通。质量 Route B = FP8 experts + BF16 resident
**prefill cos 0.981** (达成 0.96+ 目标; W4A4 0.797 / W4A16 0.871)。
单 Spark 单流 FP8 decode **0.77 tok/s** (两层 cache 后), 纯流式带宽受限。

产物: /models/MiniMax-M3-NVFP4 (232GB W4A4) + /models/MiniMax-M3-FP8E
(385GB FP8 experts, BF16 resident from original)。
runtime: minimax_dev/m3_runtime_fp8.py (Route B 主) / m3_runtime.py (W4A16)。
kernels 新增: nvfp4_dequant_swizzled_to_bf16 (additive); FP8 用现成
fp8_block128_dequantize_to_bf16。MSA Triton: minimax_dev/msa_triton/
(decode-sparse sm_121 全验证, 未接 runtime)。

🔴 MTP 未发布: checkpoint 无 mtp/nextn 张量, spec decode 不可行。
单 Spark 单流提速杠杆只剩: batching (吞吐) / 2-Spark / prompt-lookup spec /
降 W4A16 (0.914, ~2×速度)。待用户 steer。

未做: 真 8K/32K/128K benchmark (decode 已确认流式受限~ctx无关; 128K 需 FP8 KV
cache, 当前 BF16 KV 放不下); P4-2 Triton decode-sparse 接 runtime (长 ctx
attention 加速, 但 decode 被 expert 流式主导, 收益次要)。

合成 prompt bug: m3_runtime_fp8.py main 的 `base*8` 只 ~177 token, 长 ctx
benchmark 要改 repeat 次数 (base*400 给 8K)。

---
## 历史 (2026-06-12)

## 任务与现实约束

目标（用户确认）: 单 Spark 跑通 M3 (先跑起来再优化) + MSA kernels 的 sm_121 版
+ ctx ∈ {128, 8K, 32K, 128K} 的 TTFT/decode 报告。不测 BF16 性能 baseline。

硬约束（验证过）:
- M3 = 427B/22B-act MoE: 60 层, hidden 6144, GQA 64Q/4KV hd128, partial RoPE 64,
  128 experts top-4 + 1 shared (inter 3072), 层 0-2 dense(12288)+full-attn,
  层 3-59 MoE+MSA sparse attn (block 128, top-16, 4 idx heads, blockmax),
  vocab 200064, 7 MTP modules (ckpt `mtp.*`)。
- 单机 121GB 放不下任何全精度: BF16 854 / MXFP8 444 / NVFP4 ~232 GB。
  → 方案: NVFP4 + 常驻热 expert cache (~83GB) + packed NVMe streaming。
- MiniMax-AI/MSA kernels SM100-only (tcgen05) — GB10 sm_121 跑不了。
  → 我们的 sm_121 实现就是生态贡献点。

## 环境（全部就绪）

- SSH: `sshpass -p '@leadtek' ssh -p 8879 leadtek@webchat.libaoguo.top`
- 容器: `flashrt-minimax`（TRT-LLM spark image, `--memory=110g` 护栏,
  挂 ~/minimax_spark_work:/workspace + ~/models:/models）
- 仓库: Spark `~/minimax_spark_work/FlashRT` (git clone, identity 已配);
  push 路径: Spark commit → 本地 `git fetch spark` (ssh remote) → push GitHub
- venv: `/workspace/venvs/m3ref` (transformers 5.12 + 容器 torch 2.8 sm_121)
- build: `cmake -B build-spark-sm121 -S . -DGPU_ARCH=121 && cmake --build ... -j6`
  (⚠️ -j16 会 OOM; ⚠️ 编译别和模型流式任务同容器并发)
  .so 落在 `flash_rt/` 包目录; FA2 在独立 `flash_rt_fa2` 模块
- 模型: `/models/MiniMax-M3` 59 shards 完整验证 (23416 tensors, 原始 MiniMax
  命名 per-expert w1/w3/w2, 无 HF modeling 文件; transformers 5.12 原生支持)

## 已测硬数据

- NVMe: dd direct 3.4GB/s, buffered(预读后) 11.8GB/s。
  safetensors mmap 路径 ~0.3GB/s (Spark 已知坑, readahead 8192 已设仍慢)。
  **preadv+16线程 直读 = ~4.3GB/s (14.5GB MoE 层 3.3s)** ← raw_st_reader.py
- NVFP4 selfcheck (FP4 GEMM vs BF16 matmul, M=8 随机激活):
  **每层 o_proj + expert w1 cos ≈ 0.990** — 量化管线 + sm_121 FP4 GEMM 双验证。
  ⚠️ 单层 0.99 ≠ 60 层 E2E 质量 (motus stage3 教训), E2E cos 对 reference 是裁判。
- pread 单次上限 2GB-4K (embed/lm_head 2.46GB 要分块)。

## kernel 绑定（fvk, 已验证存在 sm_121）

- 权重量化: `bf16_weight_to_nvfp4_swizzled(w,packed,sf_swz,scratch,out_gs,N,K,s)`
  alpha = out_gs 直接作 GEMM alpha
- 激活量化: `quantize_bf16_to_nvfp4_swizzled(in,fp4,sf,rows,cols,s)`
- GEMM: `fp4_w4a16_gemm_sm120_bf16out(A_packed,B_packed,D,M,N,K,SFA,SFB,alpha,s)`
  (+ `_widen` / `_pingpong` 变体; sm_121 下同名可用)
- SF swizzle: `nvfp4_sf_linear_to_swizzled`; SF bytes = ⌈N/128⌉×⌈(K/16)/4⌉×512

## M3 数学关键点（照 transformers modular_minimax_m3_vl.py 固化）

- RMSNorm 全部 Gemma 式: fp32 normalize × (1+w)
- 主注意力: per-head QK-norm (hd128), partial RoPE 前 64 维 (theta 5e6,
  非交错 half-half), scale = 128^-0.5
- indexer: q_proj 6144→4×128, k_proj 6144→128 (单共享头), 各自 RMSNorm,
  同款 partial RoPE; score = fp32 q·k^T, causal mask, pad → block 128,
  amax(block) → amax(4 heads), local block scatter +inf, top-16, -1 右补
- router: fp32 sigmoid(logits); 选择用 scores+e_score_correction_bias;
  权重用 raw sigmoid gather 后归一化 (bias 不进权重!); block 级 ×2.0 再加 shared
- expert/dense/shared MLP: swigluoai = clamp(gate≤7, |up|≤7),
  glu = gate·σ(1.702·gate), out = down((up+1)·glu)   ← 注意 (up+1)
- MTP 权重 `mtp.*`, HF 忽略加载; M3 推荐采样 temp=1.0 top_p=0.95 top_k=40

## 文件清单 (minimax_dev/)

- `raw_st_reader.py` — preadv 并行 shard 读 (get / get_many / drop_all)
- `m3_ref_layerwise.py` — P1: 逐层流式 BF16 reference。输出 ref_out/:
  prefill_logits.pt, trace.pt (experts/expert_weights/blocks/hidden_last
  per layer), 路由集中度统计打印 (80% 覆盖所需 expert 数)
- `m3_quant_nvfp4.py` — P2: 量化到 /models/MiniMax-M3-NVFP4/:
  resident_top.pt + resident_layer_NN.pt + experts_layer_NN.bin
  (128 × 31,850,496B 定长 block: w1_packed|w1_sf|w3_packed|w3_sf|w2_packed|w2_sf,
  4096 对齐; alphas 在 resident_layer 的 expert_alphas [128,3])
  量化范围: attn qkvo / dense / shared / experts / lm_head = NVFP4;
  embed / norms / router gate+bias / indexer projs = BF16。可断点续跑。

## P4 (MSA sm_121) 已定策略

社区扫描结论: vLLM PR #45381 与 SGLang PR #27944 已有 M3 精确同语义 Triton
kernel (indexer+bitonic topk / block-sparse GQA attend / split-K decode /
FP8-KV), Triton 在 sm_121 直接编。策略: vendor Triton 当 baseline 与正确性
参照 → profile → 只对热点 (decode split-K, 融合 indexer) 手写 CUDA。
SGLang 还有可借鉴的纯 CUDA `minimax_decode_topk.cuh`。ground truth 另有
MSA 仓库 `cute/test_sparse_atten.py::sparse_attention_ref`。

## P3 设计草案（下一个大块）

runtime = FlashRT 模板三件套 (models/minimax_m3/pipeline_spark.py +
frontends/torch/minimax_m3_spark.py + spec) + 新增:
1. 常驻区: 非 expert 权重 NVFP4 ~9GB + KV + embed BF16
2. expert cache: 分层配额 (bigs: layer-aware, 64 全局槽=0 命中教训),
   ~83GB ≈ 2600 slots × 31.85MB, P1 路由分布定每层配额
3. miss loader: packed bin pread (块内偏移=manifest), 16 线程池,
   预期单块 31.85MB @ ~4GB/s ≈ 8ms
4. 路由→gather→4+1 expert FP4 GEMM→scatter (decode M=1: 逐 expert GEMV)
5. attention: 层 0-2 FA2 (flash_rt_fa2), 层 3-59 先 Triton sparse (P4)
   或短 ctx 时 dense SDPA 起步
预期 decode: 路由命中 85-90% → 6-8 tok/s; +MTP spec ×~1.7。
TTFT: prefill 全 expert 触碰 ≈ 232GB/4.3GB/s ≈ 54s 流式垫底 (热缓存可减)。

## 当前在跑 (链式后台, 容器内)

`ref.log`: P1 reference (prefill 253 tok + 8 token 贪心, ~3.5min/pass)
→ `quant.log`: P2 量化续跑 (resume 跳过已完成层)。
完成判据: ref.log 出现 GENERATED 文本 (连贯=正确性初验) + 路由集中度统计;
quant.log 出现 ALL DONE + 各层 selfcheck cos。
