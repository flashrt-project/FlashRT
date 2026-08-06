# 标准 Chameleon-7B @ Thor SM110 — 权威文档

**平台**: Jetson AGX Thor (SM110, aarch64) · CUDA 13.0 · transformers 4.43+
**模型**: 标准/独立 Chameleon-7B(纯 LLM 主干 + VQGAN 图像 tokenizer,**无 ActionHead / ActionVAE**)
**生产方案**: 全 32 层**运行时动态 per-tensor FP8**(实现位于 Chameleon 专用的 `flash_rt/models/chameleon/pipeline_thor.py::chameleon_forward`,迁移自 derivative repo 的 RynnVLA-002 Thor 移植)+ 通用 eager Chameleon VQGAN 默认路径 + cuBLASLt 逐 shape autotune + L31 selective clamp;TensorRT VQGAN 仅显式 opt-in
**版本**: v1.4 (2026-08,新增 KV-cache 增量解码 `generate_greedy`:30.4 tok/s,逐 token 与全前缀重算 oracle 一致)

> 与 RynnVLA-002 Thor 的关系:本文档描述的是**独立/标准 Chameleon-7B**(纯文本+图像对话骨干,直接 frontend `encode_prompt`/`prefill`/`generate_greedy`,不是 VLA `predict()` 接口),没有 ActionHead/ActionVAE。其动态 FP8 Chameleon 主干实现与 Thor attention backend 迁移自 derivative repo 中 RynnVLA-002 的移植。

---

## 0. 结论先行

- **资产路径**:`/path/to/Chameleon_7B_mGPT`(注意实际目录名是 `mGPT` 不是 `mGP`)。包含权重 shards、tokenizer、`original_tokenizers/vqgan.{yaml,ckpt}`。
- **HF 直接加载会失败**:当前 `transformers` 的 `ChameleonForConditionalGeneration.from_pretrained` 在该 checkpoint 上因 `q_norm`/`k_norm` 形状为旧式 `[1,128]`(而不是新式 `[32,128]`)而报错/静默错载。**变通方案**:生产路径直接走 FlashRT 自己的声明式 `WeightLoader`(完全绕开 HF `from_pretrained`);或如 `scripts/check_chameleon_thor_precision.py` 那样,用裸 `ChameleonForConditionalGeneration` 构造器 + `load_state_dict(strict=False)` 加载 HF 参考模型(上游没有 vendor 目录,脚本直接从 `transformers` 导入 `ChameleonForConditionalGeneration`,可用 `--skip-hf` 跳过 HF 参考比对)。
- **精度已验证(真实图片,非合成 token id)**:
  - FlashRT FP16 vs HF BF16(last-token logits cosine,mask_image_logits 后):**0.9999997**,greedy next-token 完全一致。
  - FlashRT 动态 FP8 vs FlashRT FP16:**0.99999999**,greedy next-token 完全一致,top-10 overlap 1.0。
- **VQGAN backend policy(框架定位)**:FlashRT 面向**通用/标准 Chameleon**时保持框架通用性——VQGAN **默认走 eager** Chameleon tokenization(`use_trt_vqgan=False`),不默认依赖 RynnVLA/TensorRT engine,保证框架自身能力可独立运行。**若部署环境存在可用 TRT engine,建议显式开启**(`use_trt_vqgan=True` 或脚本 `--use-trt-vqgan`;实测 VQGAN 74.9→17.3ms,TRT E2E ~121ms vs eager ~190ms)。这与此前 RynnVLA 等**专用模型**的策略不同:专用模型以降低模型耗时为核心目标,可默认/直接使用 TRT 等加速路径;通用模型必须保持 FlashRT 框架能力为默认,TRT 只是显式 opt-in 的**建议加速项**。输出 JSON 记录实际 backend(`eager`/`trt`)。
- **最新端到端性能(真实图片 `hand_1.jpg`,prompt "Describe the image.",target_size=512,stage-aware benchmark,含 §4.11 融合 kernel + §4.12 FA4 之后)**:

  | 口径 | VQGAN backend | FlashRT FP8 p50/mean | 说明 |
  |---|---|--:|---|
  | 默认 E2E | eager | **~190 ms** | VQGAN 74.9ms 主导;eager 无 TRT 时瓶颈在 VQGAN |
  | 显式 opt-in E2E | TRT | **121.1 / 121.2 ms** | TRT VQGAN 17.5ms + transformer 103.5ms(含 FA4) |
  | transformer-prefill-only(FA4) | eager ids reused | **101.9 / 102.0 ms** | HF-comparable,不含 VQGAN,50 iter |

  > **2026-08-05 复测(单热窗口,20 iter,`benchmarks/chameleon_thor_latency.py`)**:
  > transformer-only FA4 off **111.2 ms** / FA4 on **104.2 ms**(−7.0);E2E eager+FA4 **177.3 ms**;
  > E2E TRT+FA4 **120.2 ms**。与下表历史值差异在热噪声(±5%)内;PR 面文档
  > (`docs/chameleon_usage.md`、`docs/benchmark_comparison.md`、USAGE.md)统一用复测值。

  Roofline 结论(详见 §4.10-4.12):Se=1056/1072 时理论工作量约 **14.3-14.5 TFLOP**;按 240 TFLOP/s 计,乐观 compute floor 约 **59-60 ms**。per-shape GEMM 微测(§4.11)证实 GEMM tactic 已接近 Thor 实测天花板(32 层 GEMM-only ≈61.9ms),因此与 floor 的差距主要来自非 GEMM 工作。§4.11 融合 RMSNorm/SwiGLU+amax(117.5→110.9ms)、§4.12 引入 FA4 attention(110.9→**101.9ms**,58.3% of 240TFLOP/s,**1.71× floor**)。剩余空间集中在 O-projection 量化(无自然融合点)与 KV-cache 增量解码(后者已在 §4.13 落地)。
- **FA4 attention(显式 opt-in)**:参考上游 PR [`flashrt-project/FlashRT#163`](https://github.com/flashrt-project/FlashRT/pull/163)(GROOT N1.7 Thor NVFP4+FA4,单图 51.6→29.9ms,1.70×)。Chameleon 形状(Se=1056,32 head,HD=128,causal)实测 FA4 比仓库 CUTLASS causal FMHA **快 2.75×**(450.5→163.9 µs/层),输出 cos=0.99999994;集成后 transformer-only FP8 **-8.4ms**。依赖 `pip install .[thor-fa4]`(nvidia-cutlass-dsl==4.5.1 + quack-kernels==0.4.1),通过 `FLASHRT_CHAMELEON_FA4_ATTN=1` 或构造参数 `use_fa4_attn=True` 开启,backend 不可用时自动回退 CUTLASS FMHA。
- **KV-cache 增量解码(2026-08 新增,详见 §4.13)**:`generate_greedy` 现为一次 prefill + M=1 增量 decode(`chameleon_decode_step`),稳态 **30.4 tok/s**(32.9 ms/token),墙钟约 **2.8×** 于全前缀重算;逐 token 与 eager 全前缀重算 oracle 完全一致(32-token 生成 38/38)。新增 bottom-right 对齐 causal FMHA 符号 `fmha_fp16_causal_br`(decode 时 SQ=1<SK);仅支持动态 FP8 路径(`use_fp8=True`)。
- **关键 bug 与修复**(本次适配过程中发现,已修复,详见 §3):
  1. **PAD 行被误当作"最后一个 token"**:`set_prompt()` 将 input_ids 补齐到 16 的倍数,但取 lm_head 输入时用了补齐后的 `Se-1`,取到的是 PAD 行而非真实最后一个 token 的隐藏状态。修复:单独跟踪 `self._real_len`(补齐前长度),用 `real_len - 1` 索引。
  2. **未做 `mask_image_logits`,导致比对时 cosine 严重失真(0.08~0.47)**:HF `ChameleonForConditionalGeneration` 在生成 logits 后,将全部 8192 个 VQGAN 图像码本 token 的 logit 置为 fp16 最小值(防止文本生成时输出乱码图像 token)。FlashRT 侧原始实现未做此掩码,导致直接比较 raw logits 时,占了 65536 词表 1/8 的图像 token 区间数值分布不同,cosine 相似度被严重拉低,一度误判为"模型前向出 bug"。逐层 probe(`chameleon_forward` 的 `probe` 参数,对比 layer 0/1/4/8/16/31 的 post-residual-2 隐藏状态)证实主干与 HF 参考 cosine ≥ 0.998,证明主干本身正确;真正的问题是输出层缺少与 HF 参考一致的图像 token 掩码。修复:`ChameleonTorchFrontendThor` 新增 `_build_image_token_mask()`(从 `config.json` 的 `vocabulary_map` 中收集 `IMGIMG*` 键对应的 8192 个 id),在 `_run_forward()` 的 lm_head GEMM 之后对 `last_logits` 做 `index_fill_(-65504.0)`。该操作是纯 tensor 原地操作,CUDA-Graph 安全,无需新 kernel。
- **已知限制(FP16 全量 hidden 输出,非 logits)**:纯 FP16 路径(`use_fp8=False`)在真实长图片序列(Se≈1056)上,`hidden_all` 完整序列输出中**部分非末位 token 行**会出现 NaN(逐行检查:1056 行中约 861 行受影响,均为图像 token 位置),根因是 Chameleon-7B 深层(尤其 L31)残差流的量级可达 ~9000(massive activation),在 fp16(max 65504)累加多层后局部溢出;`chameleon_forward_fp16` 现有的 `ffn_gate_clamp_value`(默认 10000)只 clamp 了 gate*up,未 clamp 残差流本身。**这不影响 last-token logits 的正确性**(已验证 argmax/cosine 均正确,因为最后一个 token 的隐藏状态本身没有溢出),仅影响调试时读取 `out["hidden"]` 全序列这一可选路径。动态 FP8 路径因为有 `ffn_down_clamp_value=60000` 护住深层溢出,未观察到此问题。**建议**:生产/精度验证默认用动态 FP8 路径;如需 FP16 全序列 hidden 用于调试,只信任最后几个真实 token 的行,不要假设中间图像 token 位置的 hidden 一定是有效数值。这是已知限制,本轮不引入新 kernel 修复(超出本次范围)。

---

## 1. 生产配置

```python
from flash_rt.frontends.torch.chameleon_thor import ChameleonTorchFrontendThor

fe = ChameleonTorchFrontendThor(
    checkpoint_dir="/path/to/Chameleon_7B_mGPT",
    use_fp8=True,           # 全 32 层运行时动态 per-tensor FP8(推荐默认)
    use_cuda_graph=True,    # graph 捕获默认开
    use_trt_vqgan=False,    # 通用 Chameleon 默认 eager VQGAN;若环境有 TRT engine 建议显式开启(True)
    use_autotune=True,      # 每个新 Se 只 autotune 一次 cuBLASLt tactic
    target_size=512,        # 质量优先;384 是推荐快速档;256 风险较高
    ffn_clamp_layers=[31],  # 默认只 clamp L31;"all" 可恢复旧全层 clamp
    fp4_ffn_layers=None,    # 默认关闭;可用 [0..7] 做 sweep
    max_seq=4096,
)

out = fe.prefill("Describe the image.", [pil_image])
# out["logits"]: (VOCAB_SIZE,) fp32,已应用 mask_image_logits
# out["hidden"]: (Se, D) fp32,padded 长度,FP8 路径全序列可信;FP16 路径中间行可能 NaN(见上)
# out["input_ids"]: 补齐后的 token id 列表

gen = fe.generate_greedy("Describe the image.", [pil_image], max_new_tokens=32)
# gen["text"]: 完整贪心解码文本
```

- `use_fp8=False` 仅用于对照/调试(动态 FP8 已验证与其 cosine 0.9999998,且更快),不建议作为生产默认。
- **VQGAN 生产建议**:通用 Chameleon 默认 `use_trt_vqgan=False`(eager,框架通用);**若部署环境存在可用 TRT engine,建议显式设为 `use_trt_vqgan=True`**(VQGAN 74.9→17.3ms,E2E ~190→~121ms)。专用模型(RynnVLA 等)以降低模型耗时为核心理念,可默认使用 TRT;通用模型保持 FlashRT 框架能力为默认,TRT 是显式 opt-in 的建议加速项。
- `generate_greedy` 已实现 KV-cache 增量解码(§4.13):一次 prefill + M=1 decode step,稳态 30.4 tok/s;仅动态 FP8 路径(`use_fp8=True`)支持,FP16 / NVFP4 FFN 配置会 fail-fast 报错。全前缀重算版本保留为 `_generate_greedy_recompute`(oracle/调试用,始终 eager)。

## 2. 权重加载

标准 Chameleon-7B 使用 32 层 Chameleon 主干布局(`attention_bias=false`,`mlp_bias=false`,per-head Q/K LayerNorm,SwiGLU FFN)。上游 FlashRT 中该布局由 Chameleon 专用、内联的 `_llm_block()` 声明式描述(见 `flash_rt/frontends/torch/_chameleon_thor_spec.py`;SM87 INT8 变体见 `flash_rt/frontends/torch/_chameleon_spec.py`),不依赖任何其他模型的 weight spec。`_chameleon_thor_spec.py::build_spec()` 在 `_llm_block()` 之外声明 `model.embed_tokens.weight` / `model.norm.weight` / `lm_head.weight`(标准 Chameleon 未 tie word embedding,`lm_head.weight` 是独立权重)。

Checkpoint 的 `self_attn.q_norm.weight` / `k_norm.weight`(以及对应 bias)是旧式 `[1, 128]` 形状(按 model_parallel_size=1、所有 head 共享同一份 LayerNorm 参数);加载后统一 `.reshape(-1).contiguous()` 成扁平 `(128,)` 张量再喂给现有的 `qk_norm_rope_fused_fp16` kernel,与 HF 的 `ChameleonLayerNorm`(每个 head 组共享一份权重,`model_parallel_size=1` 时等价于全 32 head 共享)语义一致。

## 3. 图像输入与真实数据验证

**所有精度/性能验证均使用真实图片**,而不是合成 token id(路径:`/path/to/images/*.jpg`,一个存放真实照片的目录;验证使用人手真实照片)。VQGAN 图像 tokenization 采用迁移自 derivative repo 的 `<IMG_START, h_grid_tok, w_grid_tok, [VQ tokens with NEWLINE per row], IMG_END>` 布局(`ChameleonTorchFrontendThor._vqgan_encode`),底层使用随仓库内建的 `flash_rt/models/chameleon/vqgan/`(`ImageTokenizer` + `VocabTranslation.convert_img2bp2`)。

### 3.1 验证脚本

```bash
# 精度门禁(FP16 vs HF BF16, FP8 vs FP16;真实图片;默认 eager VQGAN;
# HF 参考比对从 transformers 导入 ChameleonForConditionalGeneration,
# 只跑 FP16-vs-FP8 时可加 --skip-hf)
PYTHONPATH=. python scripts/check_chameleon_thor_precision.py \
  --checkpoint /path/to/Chameleon_7B_mGPT \
  --image-dir /path/to/images \
  --prompt "Describe the image." \
  --output /tmp/chameleon_thor_precision.json

# TensorRT VQGAN 显式 opt-in 加速测量时加 --use-trt-vqgan
# 延迟基准(HF BF16 eager / FlashRT FP16 / FlashRT 动态 FP8;真实图片)
PYTHONPATH=. python scripts/bench_chameleon_thor.py \
  --checkpoint /path/to/Chameleon_7B_mGPT \
  --image-dir /path/to/images \
  --prompt "Describe the image." \
  --use-trt-vqgan \
  --iters 10 --warmup 2 \
  --output /tmp/chameleon_thor_bench.json
```

### 3.2 实测结果(单图 `hand_1.jpg`,Se≈1056,已写入 §0 表格)

- 精度 JSON 关键字段:`flashrt_fp8_vs_flashrt_fp16.logits_cosine=0.9999999999`,`flashrt_fp16_vs_hf_bf16.logits_cosine=0.9999997`,两者 `greedy_token_match=true`。
- 逐层 probe(手工验证,未纳入脚本):layer 0/1/4/8/16/31 post-residual-2 hidden cosine 均 ≥ 0.9977(layer 31 隐藏状态量级 ~9000,与 HF 参考一致,证实 massive-activation 复现正确)。

## 4. 硬件利用率剖析与性能优化(TRT VQGAN / kernel 融合 / autotune / selective clamp / FP4 sweep)

初版实现(仅动态 FP8,无 TRT VQGAN/融合/autotune/selective-clamp)在真实图片上测得 246.2 ms;用 `nsys` 对 `_run_forward`(32 层 LLM)做 kernel 级 profiling 后,发现四个明确、低风险的优化点,逐一落地并复测;另补做 FP4 FFN sweep 与 target_size 速度/质量档位。

### 4.1 剖析方法与发现

- **理论 FLOPs vs 实测吞吐**:Se≈1072 时单次 prefill 理论 GEMM+attention ≈14.5 TFLOP;初版 182 ms 的 LLM 前向对应 ~80 TFLOPS,而 derivative repo 中 RynnVLA-002 Thor 移植的记录为:cuBLASLt FP8 GEMM 实测顶峰 ~240 TFLOPS——利用率约 33%,提示还有空间。
- **`nsys profile --capture-range=cudaProfilerApi` 对 3 次 `_run_forward` 取 kernel 汇总**:两种 FP8 GEMM tactic(`nvjet_qqhsh_*`)合计占 **61.3%**,CUTLASS causal FMHA(attention)占 8.2%,其余 ~30% 分散在多个小的 elementwise/量化 kernel 上:
  - `clamp_inplace_fp16`(FFN 溢出保护)6.4%
  - `quantize_fp8_kernel_generic` + `absmax_kernel`(动态 FP8 逐层测 amax + 量化)6.4%
  - `mul_fp16` + `silu_inplace`(SwiGLU 的两个独立 kernel)8.1%
  - 其余 rms_norm / residual_add / qk_norm_rope 等 ~10%
- **关键发现**:`flash_rt_kernels` 里已经有现成的融合 kernel `gate_geglu_fp16`(=`gate_silu_mul_fp16`,一次 kernel 完成 `SiLU(gate)*up`),但 `chameleon_forward`/`chameleon_forward_fp16` 里 SwiGLU 走的是两个独立调用(`silu_inplace_fp16` + `mul_fp16`)。这是**零新增 kernel、纯路由层面**的优化机会。

### 4.2 优化 1:TRT FP16 VQGAN(eager PyTorch → TensorRT)

延迟拆解发现 VQGAN 图像编码(eager PyTorch,`img_tokens_from_pil`)占单次 prefill 的 **35%**(86.7 ms / 246 ms)。FlashRT 提供的可选加速包装是 `flash_rt.hardware.thor.vqgan_trt_backend.VQGANTRTBackend`,读取 `~/.flash_rt/trt_engines/vqgan/manifest.json` 中的固定分辨率 TRT engine。标准 Chameleon 与 derivative repo 中 RynnVLA-002 使用的 frozen `vqgan.ckpt` 前 64KB hash 一致(`d18dedf91281e8b3`),因此在**显式 opt-in** 且 engine 兼容时可复用同一 FlashRT engine cache。

修改:`flash_rt/frontends/torch/chameleon_thor.py` 的 `_ensure_trt_vqgan_loaded()`/`_vqgan_encode()` 保留 TRT-fast-path + eager-fallback 逻辑,但 `use_trt_vqgan` 默认是 **False**;只有构造参数 `use_trt_vqgan=True` 或脚本 `--use-trt-vqgan` 才会尝试 TRT。TRT opt-in 时,VQGAN 编码延迟从 86.7 ms 降到 **19.8 ms**(~4.4×)。

**策略说明(框架定位)**:通用/标准 Chameleon 是 FlashRT 框架的通用能力,默认保持 eager VQGAN、不依赖 RynnVLA/TensorRT engine;**若部署环境存在可用 TRT engine,建议显式开启 `use_trt_vqgan=True`**。这与专用模型(如 RynnVLA,核心是降低该模型的耗时)可默认使用 TRT 的策略不同——通用模型的默认路径必须保持 FlashRT 框架自身能力可独立运行,TRT 只作为显式 opt-in 的建议加速项。

⚠️ 注意:TRT 路径固定用 `target_size×target_size` 方形 resize(bicubic),eager 路径走 `var_center_crop` 保持长宽比裁剪——两者对非方形图片产出的图像 token 数量/内容不完全相同(本例 Se 从 1056 变为 1072),这是预期行为差异,不是 bug,该设计继承自 derivative repo 的 RynnVLA-002 移植。

### 4.3 优化 2:融合 SwiGLU kernel(`silu_inplace_fp16`+`mul_fp16` → `gate_geglu_fp16`)

`flash_rt/models/chameleon/pipeline_thor.py` 里 4 处 `SiLU(gate)*up` 计算(`chameleon_forward` 的动态 FP8 分支、AWQ-D 静态分支、`chameleon_forward_fp16`、`chameleon_forward_calibrate`)统一从两个独立 kernel 调用改为单次 `fvk.gate_geglu_fp16(gate, up, out, n, stream)` 调用。数学上完全等价(同一 SiLU-mul 公式,`gate_geglu_fp16` 底层就是 `gate_silu_mul_fp16`),只是省了一次 kernel launch 和一次显存读写往返。

**注**:上游的 `flash_rt/models/chameleon/pipeline_thor.py` 是 Chameleon 专用文件;改动是纯数学等价替换,精度验证无回归(FP8 vs FP16 cosine 0.9999999996,与融合前 0.9999999997 基本一致)。

### 4.4 优化 3:cuBLASLt 逐 shape autotune

Chameleon 每层的 7 个 GEMM(q/k/v/o/gate/up/down)在给定 Se 下只有 3 种不同的 `(M,N,K)` 形状(q/k/v/o 共享 `(Se,4096,4096)`,gate/up 共享 `(Se,11008,4096)`,down 是 `(Se,4096,11008)`),外加 lm_head 的 `(1,65536,4096)`。新增 `ChameleonTorchFrontendThor._autotune_gemms(Se)`(复刻 FlashRT 内既有模型(如 motus)的 autotune 模式):用 dummy buffer 对每个形状跑 `gemm.autotune_fp8_nn_dev_fp16(...)`/`autotune_fp16_nn(...)`,cuBLASLt 内部按 `(M,N,K)` 缓存最优 tactic,后续(包括 CUDA Graph 内的)同形状调用自动复用。`set_prompt()` 里对每个新 Se 只跑一次(`self._autotuned_se` 去重),默认开启(`use_autotune=True`)。

单次 autotune 耗时 ~1 秒(8 个候选 tactic × 4 个形状),之后稳态延迟从 187.8 ms 降到 **140.3 ms**(~1.34×)。

### 4.5 优化 4:L31 selective clamp

优化后 profile 里 `clamp_inplace_fp16` 仍占 **8.3%**。代码注释和实测都说明真正需要保护的是深层 outlier(尤其 L31):完全关闭 clamp 会让真实图片 FP8 vs FP16 logits cosine 变成 NaN,但只保留 L31 clamp 仍保持精度无回归。

实现:在 `chameleon_forward(...)` 增加 `ffn_clamp_layers=None` 参数,**默认 None 表示全层 clamp(最保守行为)**;标准 Chameleon frontend 默认解析为 `frozenset({31})`,也可通过 `ffn_clamp_layers=[...]` 或 env `FLASHRT_CHAMELEON_FFN_CLAMP_LAYERS="24-31"/"all"/"off"` 覆盖。

结果:单图 512 档动态 FP8 从 **140.3 ms → 130.6 ms**,真实图片 FP8 vs FP16 logits cosine **0.99999999996**,两图真实输入也通过(FP8 vs FP16 cosine **0.99999999994**)。

### 4.6 FP4 FFN sweep(已实现,默认关闭)

标准 Chameleon frontend 已接入 decoupled FP4 FFN 机制(迁移自 derivative repo 的 RynnVLA-002 移植):从原始 safetensors 重新读取 `gate_proj/up_proj/down_proj` FP16 权重,打包 `gu_w_fp4/d_w_fp4` + scale-factor buffer,传入 `chameleon_forward(fp4_ffn_layers=...)`。默认保持关闭,通过 `fp4_ffn_layers=[...]` 或 env `FLASHRT_CHAMELEON_FP4_LAYERS="0-7"` 开启。

真实图片单图 512 档 sweep:

| FP4 FFN 层 | 延迟(mean) | greedy | 备注 |
|---|--:|---|---|
| 关闭 | 132.1 ms | EOS | 当前安全默认 |
| L0-3 | 133.4 ms | EOS | 反而略慢 |
| L0-7 | 130.4 ms | EOS | 只有 ~1-2 ms 收益 |

L0-7 的 FP8/FP4 vs FP16 logits cosine **0.9999999998**,top-k overlap 0.8,greedy match true。结论:FP4 路径可用,但在标准 Chameleon 单图 Se≈1072 上收益太小,**不建议默认开启**;保留给多图/长序列/更高分辨率 sweep。

### 4.7 target_size 速度/质量档位

`target_size` 是当前最大杠杆,因为它直接改变 VQGAN image token 数和 LLM Se。脚本 `check_chameleon_thor_precision.py` / `bench_chameleon_thor.py` 已新增 `--target-size` 参数。

真实图片单图当前优化栈下测得:

| target_size | Se | 延迟(mean) | greedy | 建议 |
|---|--:|--:|---|---|
| 256 | 288 | **64.8 ms** | 图像起始 token | 极速但质量风险高 |
| 384 | 624 | **93.2 ms** | EOS | 推荐快速档 |
| 512 | 1072 | **130.6 ms** | EOS | 默认质量档 |

结论:通用 Chameleon 默认保持 512,但服务端可暴露 384 作为低延迟模式;256 需要任务级质量验证后再使用。

### 4.8 graph 拆分与 TRT 非默认 stream 修复

后续检查发现一个 CUDA Graph 正确性隐患:旧实现把 lm_head 的 `last_hidden_ptr` 一起捕获进 graph,当相同 padded `Se` 但真实 `real_len` 不同时,graph 会复用旧 last-token 指针。已修复为 **backbone graph + eager lm_head 投影**:`_capture_graph()` 只捕获 32 层 Chameleon backbone,每次 replay 后用当前 `self._real_len` 调 `_project_last()`。这也让 `generate_greedy()` 在同一 padded-Se block 内复用 backbone graph;4-token 测试 target_size=384 时从 **78.4 ms/token → 72.8 ms/token**。

另外,TRT VQGAN 原先在 PyTorch default stream 上调用 `execute_async_v3`,TensorRT 会提示潜在同步开销。已新增专用 `self._trt_stream` 并在该非默认 stream 内完成 preprocess/TRT/translation/token list materialization,警告消失,VQGAN 编码约 **19.3 ms**。

### 4.9 优化汇总(单图 `hand_1.jpg`,动态 FP8,target_size=512)

| 阶段 | 延迟 | 累计相对 HF BF16 |
|---|--:|--:|
| 初版(仅动态 FP8) | 246.2 ms | 1.62× |
| + TRT VQGAN | 193.3 ms | 2.06× |
| + SwiGLU 融合 | 187.8 ms | 2.14× |
| + autotune | 140.3 ms | 2.87× |
| + L31 selective clamp | **130.6 ms** | **3.08×** |

### 4.10 stage-aware E2E 与理论上限评估(2026-08-05 复测)

为避免把 VQGAN-inclusive E2E 与 HF 参考 transformer-only 数字混淆,`scripts/bench_chameleon_thor.py` 已新增 stage-aware 输出、`--reuse-input-ids` transformer-only 模式、`--generate-greedy N`、以及 roofline 字段。

**target_size=512,单图 `hand_1.jpg`,FlashRT FP8:**

| 口径 | VQGAN | Se | p50/mean | stage split |
|---|---|--:|--:|---|
| 默认 E2E | eager | 1056 | **186.5 / 186.6 ms** | encode 74.7 ms + transformer 111.7 ms |
| opt-in E2E | TRT | 1072 | **129.7 / 129.7 ms** | encode 17.5 ms + transformer 112.1 ms |
| transformer-only | ids reused | 1056 | **117.5 / 117.5 ms** | embed 0.43 + backbone 114.8 + lm_head 2.26 ms |

**Roofline:**

- Estimated work at Se=1056: **14.26 TFLOP**.
- Measured Thor FP8 GEMM plateau used as roofline reference: **240 TFLOP/s**(derivative repo 中 RynnVLA-002 的 GEMM sweep 记录)。
- Optimistic compute floor: **59.4 ms**.
- Measured transformer-prefill-only p50: **117.5 ms**.
- Achieved throughput: **121.4 TFLOP/s**,约 **50.6%** of 240 TFLOP/s.
- Measured/floor: **1.98×**。

结论:当前实现已经显著快于 HF/模型 eager,但**未达到理论最优**。若按“1.25-1.50× optimistic floor”作为 near-optimum gate,当前 1.98× 仍有 kernel/backend headroom。继续优化优先级应是:1) LLM backbone 大 GEMM/FMHA profile + tactic/attention backend;2) FP8 causal FMHA(长 Se/多图更重要);3) 真正 KV-cache incremental decode(已在 §4.13 落地)。

**Nsight Systems eager-backbone profile(target_size=512,ids reused,no graph,3 iters)**:

| 热区 | 占比 | 说明 |
|---|--:|---|
| FP8 GEMM(`nvjet_qqhsh_*`) | **55.4%** | q/k/v/o/gate/up/down 主 GEMM,仍是最大项 |
| CUTLASS causal FMHA | **11.4%** | attention,Se≈1056 时已可见,长 Se 会更高 |
| `gate_silu_mul_kernel` | **10.9%** | 已从 silu+mul 两 kernel 融合到一个,但仍是一轮大 elementwise pass |
| dynamic FP8 quantize + absmax | **9.0%** | 每层动态 amax + quantize 的固定成本 |
| norm/residual/qk_rope | **~10.9%** | 多个小 kernel 聚合 |
| lm_head | **1.9%** | 非优先项 |
| clamp | **0.3%** | L31 selective clamp 后已基本消掉 |

检查过现有 `silu_mul_split_fp8_fp16` / `gate_geglu_merged_fp8_fp16`:它们都要求传入已知 `d_scale`,不能直接替换当前动态 FP8 的 `gate_geglu + amax + quantize`。若强行用静态 down scale,会回到 derivative repo 的 RynnVLA-002 移植中已证明会在长序列上失真的静态-scale 风险。因此下一步若要吃掉这 9-11% elementwise+dynamic-quant 开销,需要**新增/改造 graph-safe fused dynamic scale kernel**(SiLU×Up 同时 amax+quantize)或接受重新验证静态/半动态 scale,不能简单路由现有 kernel。

**generate_greedy(当时口径,已被 §4.13 增量解码取代):** target_size=384、ids reused、8 token full-prefix greedy:FP8 **88.7 ms/token**。这不是 decode 理论最优路径,因为每个 token 都重跑全前缀;要接近 decode 最优必须实现 KV append/M=1 decode(已实现,§4.13:32.9 ms/token)。

**FP8 causal FMHA 可行性评估(结论:不接入)**:FP8 causal FMHA 库(`libfmha_fp8_causal.so`,源码 `csrc/attention/fmha_fp8_causal.cu`)确实已编译存在于磁盘,并非文档旧称的"未构建"。但检查内核签名后发现:`extern "C" int fmha_fp8_causal(Q, K, V, O, ..., float scale_q, float scale_k, float scale_v, float inv_scale_o, stream)` 的 4 个反量化系数是**标定时刻固定的 host float 标量**(`ctypes.c_float`),不是设备指针——量化 Q/K/V 的前置步骤(`quantize_fp8_static_fp16`)虽然走设备指针、graph-safe,但 FMHA 内核自身的反量化数值一旦标定完成就冻结,不随后续真实输入的 amax 变化而更新。这与本文档 §0 已诊断并修复过的"静态 per-tensor scale 在长序列/多模态输入上失配"是同一失败模式(当时 cosine 0.738,靠 pivot 到 runtime dynamic FP8 才恢复到 0.9997)。标准 Chameleon 的 Se≈1056-1072 且每次输入图片不同,属于该失败模式的高风险场景;derivative repo 中 RynnVLA-001 移植的同款路径实测也是净负收益(约 **-3.9ms**),默认关闭。结论:**不将 FP8 causal FMHA 接入标准 Chameleon**,继续使用当前 dynamic-FP8 GEMM + CUTLASS FP16 causal FMHA 的组合作为默认 attention 路径。

### 4.11 修正 roofline + 融合动态 quantize kernel(2026-08-05 复测)

**§4.10 的"1.98× floor"框架存在误导**:用逐 shape GEMM 微测方法(迁移来源仓库中由 `scripts/bench_gemm_fp8_fp4_thor.py` 承担)单独微测标准 Chameleon 每层实际用到的 7 个 GEMM 形状(q/k/v/o: `(1056,4096,4096)`;gate/up: `(1056,11008,4096)`;down: `(1056,4096,11008)`),autotune 后逐个实测吞吐为 **193-260 TFLOPS**,累加 32 层的 GEMM-only 理论下限 ≈ **61.9 ms**——与朴素"240 TFLOPS peak"算出的 59.4 ms 几乎一致。这说明 **GEMM tactic 本身已经接近这些具体 shape 在 Thor 上能达到的实测天花板**,§4.10 的 1.98× 差距主要是把"含 40-47% 非 GEMM elementwise/attention/norm 开销的总墙钟时间"去除以"纯 GEMM peak"算出来的,并不是 GEMM 效率问题的证据。继续在 GEMM tactic/shape 上找空间已经没有意义。

**新增两个融合 kernel(零精度代价,实测生效)**:

- `rms_norm_quantize_dynamic_fp8_fp16`(`csrc/kernels/norm.cu` 新增 `rms_norm_amax_kernel` + `quantize.cu` 组合 host 函数):RMSNorm 写 xn 的同一次 pass 里用 `block_reduce_max` 顺带把 abs-max 原子归约进 scale buffer,省掉原来 `quantize_fp8_device_fp16` 内部单独的 `absmax_kernel` 整读一遍 xn 的开销。用于 pre-QKV 与 post-attn(gate/up 输入)两处 per-layer RMSNorm。
- `gate_geglu_quantize_dynamic_fp8_fp16`(`csrc/kernels/activation.cu` 新增 `gate_geglu_amax_kernel`):同样原理融合 SwiGLU 写 pass 与 amax 归约,用于 FFN down-proj 输入量化。**仅用于不需要 outlier clamp 的层**(默认只有 L31 需要 clamp);clamp 层继续用旧的 `gate_geglu_fp16` → `clamp_inplace_fp16` → `quantize_fp8_device_fp16` 三步序列,因为 clamp 必须在算 scale 之前生效。
- 接线位置:`flash_rt/models/chameleon/pipeline_thor.py::chameleon_forward` 的动态 FP8 分支;O-projection 的量化(无自然的写入者可以顺带做 amax)和 L31 clamp 层未改动。两个新 kernel 只在 `dynamic_fp8_layers` 分支使用,不影响同文件内 AWQ 静态分支 / FP16 分支 / FP4 分支的行为。

**实测结果(单图 `hand_1.jpg`,target_size=512)**:

| 指标 | 融合前 | 融合后 |
|---|--:|--:|
| FP8 vs FP16 logits cosine | 0.9999999996 | 0.9999999991(仍 10 个 9,greedy 完全一致) |
| transformer-only p50 | 117.5 ms | **110.9 ms**(-5.6ms,与预估 ~5.1ms 吻合) |
| 效率 vs 240 TFLOP/s | 50.6% | **53.6%** |
| measured/floor | 1.98× | **1.87×** |
| TRT opt-in E2E p50 | 129.7 ms | **~128.2 ms** |

回归测试:`tests/test_install_smoke.py`、`tests/test_chameleon_thor_vqgan_backend.py` 全部通过。

**结论(该轮优化到此为止)**:GEMM tactic 已确认接近其 shape-specific 天花板,不再是杠杆;新增的两个融合 kernel 吃掉了 dynamic-quantize 侧唯一还有明确 ROI 的部分。剩余的 O-projection 量化没有可顺带做 amax 的写入者(输入是 attention 输出,不经过我们控制的 elementwise kernel),继续在这里抠 kernel 收益递减。真正还有数量级空间的杠杆是 **KV-cache 增量解码(M=1 decode)**,属架构级改动,该轮不实现——已在下一轮落地,见 §4.13。

### 4.12 residual+norm+quantize 三合一与 FA4 attention(2026-08-05 复测)

**#3 融合:`residual_add_rms_norm_quantize_dynamic_fp8_fp16`**

post-attn 位置原为 `residual_add_fp16` + `rms_norm_quantize_dynamic_fp8_fp16` 两个 kernel。新 kernel(`csrc/kernels/norm.cu` 的 `residual_add_rms_norm_amax_kernel`,寄存器缓存 residual,ssq 用 fp16 舍入后的值,amax 折进 xn 写 pass;组合 host wrapper 在 `quantize.cu`)把两者合成一个 elementwise kernel。数值语义与旧序列逐位一致(与 GROOT N1.7 的 `ac975b6` 提交同一融合模式,但该提交用静态 scale,我们的是动态 amax 版)。实测:transformer-only 110.9→**110.3ms**(-0.6ms;x/xn 缓冲 L2 常驻,主要省的是 launch 与 L2 往返,比乐观估计小),FP8 vs FP16 cosine 0.99999999 无回归。FP4 分支保持原 `residual_add_fp16`+`rms_norm_fp16` 序列,不受影响。

**FA4 attention(参考上游 PR #163)**

- **上游证据**:[`flashrt-project/FlashRT#163`](https://github.com/flashrt-project/FlashRT/pull/163) "GROOT N1.7 update: Thor NVFP4 + FA4 performance tier":单相机 LIBERO 36.8→23.7ms,双视角 51.6→29.9ms(1.70×),action cosine 0.99994-0.99995,graph replay 确定性 1.0。
- **Chameleon 形状 A/B**(Se=1056,NH=32,HD=128,causal,fp16,随机 Q/K/V):CUTLASS causal FMHA 450.5µs vs FA4 **163.9µs(2.75×)**,输出 cosine **0.99999994**。
- **集成**:`ThorChameleonAttnBackend` 新增 `set_fa4_attn(q_tensor, kv_cache)` + run() 中 CUTLASS 之前的 FA4 分支——用 torch 元数据视图切片(无分配、capture-safe),`fa4(..., causal=True, pack_gqa=True)` 在 `torch.no_grad()` 下执行,输出写回 Q_O 槽(xn 缓冲),异常自动回退 CUTLASS。frontend 侧 `use_fa4_attn=True` / env `FLASHRT_CHAMELEON_FA4_ATTN=1` 显式开启;依赖 `pip install .[thor-fa4]`(nvidia-cutlass-dsl==4.5.1 + quack-kernels==0.4.1),未装则自动回退并告警。`prefill()` 输出与 bench JSON 新增 `fa4_attn` 字段。
- **实测**:transformer-only FP8 110.3→**101.9ms**(-8.4ms);TRT VQGAN E2E 128.2→**121.1ms**;FP8 vs FP16 logits cosine **0.9999999912**,greedy 完全一致。efficiency 58.3% of 240 TFLOP/s,measured/floor **1.71×**。回归测试全过。

**§4.9 优化阶梯更新(transformer-only / TRT E2E 口径)**:

| 阶段 | transformer-only | TRT VQGAN E2E |
|---|--:|--:|
| 初版动态 FP8 | ~182 ms | 246.2 ms |
| + TRT VQGAN | — | 193.3 ms |
| + SwiGLU 融合 | — | 187.8 ms |
| + autotune | — | 140.3 ms |
| + L31 selective clamp | ~117.5 ms | 130.6 ms |
| + §4.11 融合 quantize kernel | 110.9 ms | 128.2 ms |
| + #3 residual+norm+quantize | 110.3 ms | — |
| + FA4 attention | **101.9 ms** | **121.1 ms** |

**剩余空间**:O-projection 量化(无自然融合点)、`generate_greedy` 全前缀重算(KV-cache 增量解码,已在 §4.13 落地)、eager VQGAN(默认路径;Conv-heavy 子图按 skill 结论应走离线编译,TRT opt-in 已提供)。

**上游 PR 适配(按 FlashRT CONTRIBUTING / docs/adding_new_model.md 约定新增)**:

- `flash_rt/models/chameleon/pipeline_thor.py`——Chameleon 专用 compute-path 归属文件(rule 1),包含 `chameleon_forward` 族,frontend 从此处导入。
- `examples/thor/chameleon_quickstart.py`、`benchmarks/chameleon_thor_latency.py`——quickstart 与延迟基准(`<model>_thor_latency.py` 命名惯例,含 `--reuse-input-ids`/`--use-trt-vqgan`/FA4 记录)。
- `docs/chameleon_usage.md`(英文,lingbot_usage.md 风格,含 VQGAN backend policy)、`USAGE.md` Chameleon-7B 段落、`docs/benchmark_comparison.md` Chameleon 表格。
- `tests/test_chameleon_thor_fused_kernels.py`——融合 kernel vs 非融合参考路径的 **bitwise 相等**回归(无需 checkpoint),满足 CONTRIBUTING"fused replacements validated against unfused reference paths"。该测试还暴露并修正了一处 amax 语义:融合 kernel 原来统计未舍入 fp32 amax,与 `absmax_kernel` 读 fp16 存储值差 0.004%;修正为统计 fp16 舍入后值后 bitwise 完全一致(顺带让 E2E cosine 从 0.9999999912 升到 0.9999999955)。

### 4.13 KV-cache 增量解码(M=1 decode,2026-08-06)

**动机**:§4.10-4.12 之后 prefill 已接近 GEMM shape 天花板;生成场景最后的数量级杠杆是 M=1 增量解码(每 token 只算 1 行,不再重跑全前缀)。此前 `generate_greedy` 每 token 重算全前缀(88.7 ms/token,target_size=384)。

**实现(四个文件,零新增量化/norm kernel)**:

- `flash_rt/models/chameleon/pipeline_thor.py` 新增 `chameleon_decode_step(gemm, fvk, bufs, weights, dims, scales_dev, *, attn, pos, ...)`:Se=1 版动态 FP8 主干,K/V GEMM 直接写进 cache 第 `pos` 行(`attn.kv_row_ptrs`),RoPE 指针偏移 `pos*Hd*2`,`attn.run_decode(kv_len=pos+1)`;保留 L31 clamp 语义。`pos` 是 host 标量,故 decode 始终 eager(不进 CUDA graph)。
- `csrc/attention/fmha_fp16_causal.cu`:模板化 `FmhaCausalTraits<IsQBegin>`,新增 bottom-right 对齐符号 `fmha_fp16_causal_br`(SQ<SK 时 mask 对齐序列末尾)。prefill(SQ==SK)两种对齐等价,继续用原 `fmha_fp16_causal`(top-left)。同一文件编译,CMake 无改动。
- `flash_rt/hardware/thor/attn_backend_chameleon.py` 新增 `run_decode(site, layer_idx, kv_len, stream)`(FA4 → CUTLASS `_br` → cuBLAS `attention_mha_fp16` 三级回退;q_seq=1 时 causal mask 退化为恒等)与整合后的 `kv_row_ptrs`;`run()` 对 decode 形状(q_seq=1<kv_seq)fail-fast。
- `flash_rt/frontends/torch/chameleon_thor.py`:`generate_greedy` 重写为一次 prefill + 单 token decode 循环(`use_fp8=False` / `fp4_ffn_layers` fail-fast);decode 相关 autotune shape(M=1)并入 `_autotune_gemms`;全前缀重算版保留为 `_generate_greedy_recompute`(oracle,始终 eager)。

**实测(text prompt "The capital of France is",32 new tokens)**:

| 指标 | 值 |
|---|--:|
| 稳态 decode | **30.4 tok/s**(32.9 ms/token) |
| vs 全前缀重算(墙钟) | **2.83×** |
| 逐 token vs eager 重算 oracle | **38/38 一致** |
| prefill p50(graph replay) | 157.3 ms(E2E 口径) |

精度说明:图像 prompt 下个别 token 出现 decode vs oracle 的 argmax 翻转,根因是 M=1 与全序列动态 FP8 per-tensor scale 不同(动态量化的固有属性,非 bug);文本 prompt 完全一致。bottom-right FMHA 与 PyTorch SDPA 参考在 SQ∈{1,2,128,144}、SK≤256 上均在 fp16 舍入内一致。

**本轮修复的两个 graph/state bug(调试成本最高,值得沉淀)**:

1. **capture warmup 吃掉 `x` 残差流**:主干原地更新 `x`(`residual_add_fp16` = x += out),一次 forward 后 `x` 是最终残差流(amax ~15632)。`_capture_graph` 的 warmup 跑完后直接进 capture pass,导致 graph 录制的计算建立在陈旧残差上,动态 FP8 amax 被污染 → 生成乱 token。修复:capture 前与每次 replay 前都重新 embed(`_replay_backbone`)。此前多轮"graph 写坏 KV cache"的探针结论全部是该 bug 的假象(探针自己连跑两次 backbone 未重新 embed)。
2. **oracle 的 graph replay 污染共享 KV pad 行**:重算 oracle 若走 CUDA graph,每个增长的 pad-16 Se 都会覆写与增量路径共享的 pad 行 cache,交叉比对时互相踩。修复:oracle 强制 eager。

教训:**对原地更新输入 buffer 的 backbone,"capture/warmup/replay/再跑一次"之前必须恢复输入**;任何"graph 与 eager 结果不一致"的探针,先验证探针自身没有在脏输入上跑第二次 forward。

## 5. 硬件注册

`flash_rt/hardware/__init__.py::_PIPELINE_MAP` 新增:

```python
("chameleon", "torch", "thor"):
    ("flash_rt.frontends.torch.chameleon_thor", "ChameleonTorchFrontendThor"),
```

`resolve_pipeline_class("chameleon", "torch", "thor")` 已验证可正确解析到 `ChameleonTorchFrontendThor`。因为标准 Chameleon 是纯文本/图像对话接口(`encode_prompt`/`prefill`/`generate_greedy`),不是 VLA `predict(images)` 接口,所以走**直接实例化**而非 `flash_rt.load_model()` 的 `VLAModel` 包装(与 Qwen3-VL/Nex-N2 的模式一致)。

## 6. 本轮范围外(Out of Scope)

- FP4 FFN 默认生产启用——当前已实现可选 sweep 路径,但单图 512 档收益很小,默认仍关闭。增量解码与其互斥(`generate_greedy` fail-fast)。
- 新 CUDA kernel——decode 复用全部现有动态 FP8/norm/融合 kernel;唯一 kernel 侧改动是 `fmha_fp16_causal.cu` 模板化新增 bottom-right 对齐符号(§4.13),非新算子。
- 修复纯 FP16 路径的残差流溢出(§0 已知限制)——需要在 `chameleon_forward_fp16` 里加残差流 clamp,留待后续按需评估。
- decode 的 CUDA graph 化——`pos` 是 host 标量(RoPE/cache 行偏移),当前 eager;如需进一步压 decode 延迟,可考虑 graph-per-pos 或 device-side pos,本轮不做。
