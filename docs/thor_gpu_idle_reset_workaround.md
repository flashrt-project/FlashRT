# Thor 部署稳定性:GPU 空闲重置缺陷与 CUDA Graph 防护体系

## 状态

已修复并验证(2026-08-08/09)。本文记录 GR00T N1.7 FlashRT 服务
(`serving/groot_n17/`,端口 5558)排查"推理结果周期性震荡 + 偶发 4s 卡顿"
全过程的结论与修复方案,作为后续对 **所有 Thor(SM110)serving 代码**
做系统化修复的依据。

配套文档:
- `docs/groot_transformers5_weight_corruption.md`(transformers>=5 权重静默损坏,
  本轮修复的第一个核心问题,已单独记录)。
- `docs/groot_n16_thor_sm110.md`(N1.6×FlashRT Thor 权威文档,含 GPU 空闲重置
  防护在本服务上的适用说明)。

## 一、问题现象

1. **震荡**:服务稳定推理 30–50 帧后,action 输出突然剧烈震动;错误帧的
   耗时反而比正常帧短(50ms vs 正常 70ms)。
2. **卡顿**:偶发单帧推理卡住 3.5–4.5s(约每 20–30s 一次),输出本身正常。
3. 两者相关:卡顿帧之后若 graph 已失效,后续帧持续输出垃圾。

## 二、根因(两个独立但同源的缺陷)

### 根因 A:Jetson Thor CUDA 驱动空闲重置缺陷(平台级,无法从应用层修复)

- **GPU 连续空闲超过约 200–300ms** 后,驱动进入某种低功耗/重置状态;
  下一次任意 CUDA 调用(哪怕 2KB 的 H2D 拷贝或一个 kernel launch)
  会在**内核态自旋约 3.5 秒**(实测 3.3–3.9s,`stime` 主导、wchan=0,
  周期 22–30s 随空闲节奏出现)。
- 该重置过程**同时破坏已捕获的 CUDA graph**:之后的 `graph.replay()`
  跑得很快但输出垃圾——这就是现象 1 的根源。

证据链(可复现,探针脚本见"复现方法"):

| 实验 | 结果 |
|---|---|
| 纯 2KB H2D 拷贝 @1Hz,无任何模型 | 每 22–27s 卡一次,每次 ~3.49s |
| 纯 kernel launch + sync | 同样卡顿 → 与拷贝/分配无关 |
| `gc.disable()` | 无效 → 排除 Python GC |
| CPU 调频(schedutil,实测锁 2.6GHz)/内存压力 | 无异常 → 排除 |
| GPU 持续繁忙(matmul 循环)60s | **0 卡顿** |
| 心跳周期 100/200ms | 0 卡顿 |
| 心跳周期 300/500/1000ms | 复现卡顿 |

结论:**空闲阈值在 200–300ms 之间;保活算子大小无关,只要 GPU 不空闲
超过约 200ms 即可**。机器上无 `jetson_clocks`/`nvpmodel`,无法用官方
工具锁定电源状态;只能靠保活规避。

### 根因 B:CUDA graph 无空闲防护(代码级)

前端原有逻辑:graph 一旦捕获就永远 replay。在根因 A 触发后,
replay 的是已被驱动重置破坏的 graph → 垃圾输出。需要在 replay 前
判断"距上次使用是否超过安全窗口",超时则丢弃重捕获。

## 三、修复方案(三层防护,已在 N1.7 落地)

### 防护 1:GPU 心跳保活(主修复,规避根因 A)

位置:`serving/groot_n17/run_http_policy.py` lifespan。

- 每 **150ms**(环境变量 `FLASHRT_GPU_KEEPALIVE`,秒,默认 `0.15`)
  向 GPU 提交一个微小算子(`torch.ones(16)` 加法 + `synchronize`)。
- 心跳通过与推理相同的 `ThreadPoolExecutor` 提交,推理忙时自动排队让路。
- **关键约束:心跳绝不能更新前端的 `_last_graph_use` 时间戳**,
  否则会掩盖防护 2 的空闲判断(早期踩过这个坑:心跳更新时间戳后,
  40s 空闲后直接输出垃圾)。
- 设为 `0` 可关闭(不建议:机器人按 action chunk 消费时请求间隙
  常达 0.5–2s,必然触发)。

### 防护 2:空闲重捕获守卫(兜底,对抗根因 B)

空闲判断收敛在基类 `GrootN17TorchFrontendThor` 的共享助手:
`_graph_idle_limit_s`(读 `FLASHRT_GRAPH_IDLE_REINIT_S`)、
`_graph_idle_stale()`、`invalidate_graphs()`(外部强制重捕获入口)。
两处对称使用:

1. **backbone graph** — `flash_rt/frontends/torch/groot_n17_thor_fp8.py`
   `set_prompt()`:比较 `time.monotonic() - _last_graph_use` 与
   `FLASHRT_GRAPH_IDLE_REINIT_S`(默认 **2s**);超时或形状变化则
   `reset_prompt_runtime()` 重新捕获,否则走快路径(copy_ + replay)。
2. **DiT graph** — `flash_rt/frontends/torch/groot_n17_thor.py`
   `infer()`:`use_dit_graph` 路径同样检查空闲时间,超时则
   `del _k_dit_graph` 后重新 `_capture_kernel_dit_graphs()`。

阈值演进:10s → 5s(9s 间隙实测损坏)→ 2s(4–5s 间隙实测损坏,
且 5.0s 整的间隙因严格 `>` 比较漏网)。有心跳后正常不会触发,
只在心跳失效或极端长间隙时兜底;触发时代价 ~500ms/帧。

### 防护 3:graph vs eager 自检(保险,保证输出正确性)

`serving/groot_n17/run_http_policy.py`,`FLASHRT_SELFCHECK=N`(每 N 帧):

- 用**固定噪声**分别跑 graph 路径和 eager 路径(`use_dit_graph=False`),
  比较输出 maxΔ:健康基线 0.015–0.05;graph 损坏时 3.3–3.8。
- 阈值 0.5 超限 → `fe.invalidate_graphs()` 强制重捕获 → 重跑该帧,
  客户端拿到的仍是正确结果。
- 注意开销:每帧自检会让平均耗时从 ~70ms 升到 ~170ms,
  生产用 `N=10` 或更大。

## 四、验证结果(5558,2026-08-08)

- `stress_gaps`(60 帧 @1Hz + 每 10 帧 4.5s 间隙):**0 卡顿、0 坏帧**;
  普通帧 ~65ms,gap 后重捕获帧 ~500ms,selfcheck maxΔ=0.03。
- 修复前同样测试:必现 1 次 3.5–4.5s 卡顿,4–5s 间隙后 4 次 graph
  损坏(d=3.29–3.84)。

## 五、系统化修复清单(待执行)

上述三层防护目前只落在 GR00T N1.7 链路上,需推广:

- [ ] **N1.6 serving**(`serving/groot_n16/`):同样使用 captured graph,
      需移植空闲守卫 + 心跳 + 自检(先做离线数值一致性验证,任务 #10)。
- [ ] **其他 Thor 前端**:Chameleon-7B、HyVLA-0.5 等 SM110 前端的
      serving 入口,凡使用 CUDA graph 的都要加:
      1. 心跳保活(默认 0.15s);
      2. graph replay 前的空闲检查 + 重捕获;
      3. 可选自检。
- [ ] **守卫参数统一**:把 `FLASHRT_GRAPH_IDLE_REINIT_S`(默认 2s)、
      `FLASHRT_GPU_KEEPALIVE`(默认 0.15s)、`FLASHRT_SELFCHECK`
      三个环境变量的语义写进 `docs/serving_production.md`,
      避免各 serving 脚本各自实现走样。
- [x] **代码清理**(2026-08-09 已完成,dev→生产整理):
      `gc.freeze()` 已删(与本问题无关);debug 开关
      `FLASHRT_REPROMPT_DEBUG`/`FLASHRT_FORCE_RECAPTURE`/`FLASHRT_DUMP_OBS`/
      `FLASHRT_BUILD_STAGE_TIMING` 及 `[act]` 数组打印已删;
      `aux_builder.py` 的 `[builder-slow]`/`[preprocess-slow]` 慢帧日志保留
      (仅 >100ms 触发,零常态开销)。
- [ ] **提交**:改动文件(均未提交)——
      `flash_rt/frontends/torch/groot_n17_thor_fp8.py`、
      `flash_rt/frontends/torch/groot_n17_thor.py`、
      `serving/groot_n17/run_http_policy.py`、
      `serving/groot_n17/aux_builder.py`、
      `serving/groot_n17/eager_server.py`。建议 conventional commits 分拆。
- [ ] **AGENTS.md**:在架构规则处加一条 Thor 空闲重置缺陷提示,
      指向本文档。

## 六、复现与诊断方法(备查)

- 最小复现:`/tmp/h2d_probe.py`(2KB H2D @1Hz)、
  `/tmp/cuda_op_probe.py`(kernel/alloc/pinned 分模式)、
  `/tmp/keepalive_probe.py`(`BEAT_MS`/`BEAT_TINY` 扫周期)。
  典型输出:`wall=3494ms cpu=3485ms STALL`(内核态自旋特征)。
- 服务侧诊断:`FLASHRT_GROOT_TIMING=1` 输出每帧
  `[timing] preprocess+backbone=… set_prompt=… infer=… total=…`。
- 判别口诀:
  - 错误帧**更快**(50ms)+ 间隙后出现 → graph 损坏,查防护 2;
  - 单帧**更慢** 3.5s+ 且输出正常 → 驱动空闲重置,查防护 1;
  - 平均耗时整体抬升 → 检查自检频率(`FLASHRT_SELFCHECK`)。

## 七、关键事实速查

| 项 | 值 |
|---|---|
| 空闲触发阈值 | 200–300ms(200 安全,300 复现) |
| 驱动停顿时长 | ~3.5s(内核态自旋) |
| graph 损坏表现 | replay 快(50ms)但输出垃圾,maxΔ≈3.3–3.8 |
| 健康自检 maxΔ | 0.015–0.05,阈值 0.5 |
| 心跳默认周期 | 150ms(`FLASHRT_GPU_KEEPALIVE=0.15`) |
| 空闲守卫阈值 | 2s(`FLASHRT_GRAPH_IDLE_REINIT_S=2`) |
| 正常帧耗时 | ~65–70ms;重捕获帧 ~500ms;自检帧 +100ms |
