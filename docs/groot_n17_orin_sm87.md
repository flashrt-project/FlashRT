# GR00T N1.7 × FlashRT（Jetson Orin SM87）— 权威适配与优化文档

> Orin SM87 适配记录。参考实现是 Thor SM110 的 N1.7 通路
> （`docs/deployment_npu_groot_n17.md`、`flash_rt/models/groot_n17/pipeline_thor.py`）。
> SM87 无原生 FP8/FP4 tensor core，Thor 的 fp8/fp4 通路全部不可用，
> 精度档必须落在 BF16 + SM80 家族 INT8 W8A8。
>
> 本文是 N1.7 × Orin 的唯一权威文档（one authoritative doc per model/platform）。
> 姊妹文档：`docs/groot_n16_orin_sm87.md`（同平台、不同 backbone）。
> Thor 权威见 `docs/groot_n16_thor_sm110.md` 与 `docs/deployment_npu_groot_n17.md`。

---

## 0. 状态总览

**Phase 0（侦察）✅ / Phase 1（前端 + 正确性）✅ / Phase 2–7 进行中。**

### 0.1 结论先行（2 相机，真机 SO101 帧，锁频 1300.5 MHz，frame 0/300）

**两个口径必须并列，否则会高估本通路**（详见 §6.7）：融合前 FlashRT 从 aux 白拿
`llm_input_embeds`，而 HF 要跑完整视觉塔（58.07 ms）才能产出它。所以"同一边界"
这个口径把 HF 的 58.07 ms 排除了，一直在**高估**加速比；真实独立部署成本才是
可主张的数字。融合上线后两个口径重合。

| | HF eager | FlashRT（融合前） | FlashRT（融合 + bf16 DiT，`use_int8_dit=False`） | **FlashRT（融合 + INT8 DiT，k/v 留 bf16，出厂默认）** |
|---|---|---|---|---|
| backbone（ViT+DeepStack+LLM+vlln+VLSA） | 138.79 ms | 57.79–58.49 ms | 66.13–66.31 ms（ViT 跑满 24 层） | **51.08 ms**（DiT 档不动 backbone；含杠杆 #12 的 −2.35 ms 与杠杆 #9 的 **−11.95 ms**，§6.22.5） |
| action head（DiT ×4 + encoder/decoder） | 231.97 ms | 58.38–58.47 ms | 58.23–59.06 ms（graph） | **43.75 ms**（graph，**1.329×**；k/v 豁免使它比六族全量化的 41.58–41.90 贵 1.83 ms，§6.14.4） |
| HF 视觉塔（为产出 `llm_input_embeds`） | 含在 138.79 内 | **额外 58.07 ms** | **0**（不再需要） | **0** |
| **口径 A：同一边界** | **360.38 ms** | 116.18–116.79 ms → 3.09× | 124.42–125.28 ms → 2.88–2.90× | **94.35 ms → 3.82×** |
| **口径 B：真实独立部署成本** | 360.38 ms | **174.86 ms → 2.06×** | 124.42–125.28 ms → 2.88–2.90× | **94.35 ms → 3.82×** |
| 完整 `get_action` 等价 | 387.93 ms | ~143 ms | ~152 ms（**推算**：124.9 + 27.55 pre/post） | **不再推算**——杠杆 #14 把每观测图像通路替换掉之后，改用**实测**的口径 D（下一行）。⚠️ 那个"27.55 pre/post"本来就是**两个独立量测的总量相减**的产物，不可当账面用（§6.23.6 教训 3） |
| **口径 C：每观测墙钟**（连续推理，`infer(aux=...)`，§6.15 + §6.17 + §6.19 + §6.22） | 386.27 ms（完整 `get_action`，仿真档） | — | — | **97.28 ms → 3.97×**（vs 完整）/ **3.69×**（vs 同一边界 358.84）。演进：117.56（§6.15.4）→ 111.40（杠杆 #11，§6.17）→ 108.55（cross-KV 改 bf16 tensor core，§6.19）→ **97.28**（融合 bf16 rotate-half RoPE，§6.22；**逐位相同**）。比口径 A 多 **cross-KV 刷新 4.75 ms** 与 state/action encode；一次性路径把刷新摊在 warmup 里。⚠️ **本口径有一处边界不对称，§6.23.1 已补账**：97.28 ms 是**模型侧**（`aux` 来自磁盘 fixture，**不含**图像预处理），而 386.27 ms 是 HF **含**预处理的完整 `get_action` ⇒ **3.97× 偏乐观**，该主张的数字是下面口径 D 的 **3.85–3.90×** |
| **口径 D：每观测全链路**（**含图像预处理**，一帧 uint8 图进、一个动作出；三臂配对交替，中位数 of 11，4 帧真机数据 × 两个数据集，§6.23.1） | 387.93 ms（真机）/ 386.27 ms（仿真） | — | — | **改前（vendor 图像臂）112.374–115.499 ms → 3.344–3.444×**；**改后（`frames=` 臂）99.579–100.428 ms → 3.846–3.896×**。真机 f0/f300 **3.439/3.444 → 3.896/3.883**，仿真 f100/f107 **3.344/3.437 → 3.859/3.846**。vendor 臂**自证**：其 `pixel_values` 与 fixture **bf16 逐位相同（4/4）**。仍含 §6.15 已入账的 ~2.98 ms（HF 每观测付 `_apply_vlm_processing`+tokenizer 2.74 与 `decode_action` 0.24，FlashRT 把前者提进了 `set_prompt`），**不是杠杆 #14 的功劳** |
| 去噪后 action（G4，真正的 E2E 数字） | — | cos 1.000000；max&#124;d&#124; 0.0043 / 0.0033 rad（f0/f300） | cos 1.000000；max&#124;d&#124; 0.0086 / 0.0033 rad（**f0 的 0.4935° 三档逐位同值 ⇒ 是 bf16 backbone 的地板，不是 INT8 的代价**，§6.14.3；f300 0.1868°） | **cos 0.999999391**；max&#124;d&#124; 0.0086 / **0.0034** rad（f0 同 bf16；**f300 0.1940°**，贴回 bf16 的 0.1868°，六族全量化时是 0.3636°） |
| CUDA graph vs eager | — | bit-identical（max&#124;d&#124; = 0） | **bit-identical**；1.97×（115.0 → 58.3 ms） | **bit-identical**；连续 8 帧**捕获 1 次、复用 8 次**（§6.15.2 C5） |
| **独立数据集复验**（仿真 SO101，Se=**148**，§5.6） | 386.27 ms（完整）/ 358.84 ms（同一边界） | — | 同边界 124.4 ms 量级 | **同一边界 94.4 ms → 3.80×**（杠杆 #9 前是 106.89 → 3.36×）；配对交替 DiT **1.336×**（cube_to_bowl_5 上是 1.329×，差 0.5%）；17 个门全过 |
| **长任务复验**（仿真 episode 0 **全 593 帧连续观测**，`set_prompt` 一次 + 593 次 `infer(aux=)`，距 prompt 0..592，§6.20） | 401.2 ms/帧 | — | decoded max° mean 0.3032 / **max 0.5808**（fp32 cross-KV 臂） | decoded max° mean **0.3043** / **max 0.4935**；cos 最低 **0.999998845**；三档门 **0/593 违反**；相对误差 mean **0.186%**（量程 246.82°）；两臂均值差 **+0.0012°**、**36.1% 逐值相等**；**A−B 对 prompt 距离 r=+0.0076 ⇒ 无状态漂移** |
| 测试 | — | — | **218 passed**（真机档：precision **31** + dispatch **102** + preprocess **65**（CPU-only，3.1 s）+ `test_lerobot_video` **20**；两档共用同一套门，默认档由 `inspect.signature` 自动跟随，翻默认不需改测试）/**244 passed, 0 skipped**（仿真档另加 continuous **10** 与 N1.6 后端门 16，§6.15.2/§6.17/§6.21/§6.22/§6.23/§6.24） | 同左 |

> 上表默认档的两个口径来自两把独立的尺：**出厂延迟测试**
> `test_latency_at_the_hf_boundary`（backbone 63.16 / action head 43.75 / 合计 **106.91 → 3.37×**）
> 与**逐阶段插桩脚本** `/tmp/stage_breakdown.py`（六族全量化那次是 63.36 / 41.90 /
> **105.26 → 3.42×**，与当时的出厂测试相差 **0.45%**）。
> DiT 档的比值另有配对交替 A/B 一把尺：**1.329×**（bf16 58.21 / INT8+k/v豁免 43.81，
> 中位数 of 11）；六族全量化时是 **1.398×**（§6.14.2 的 A 臂）。

⇒ 融合本身的收益是 **174.86 → 124.56 ms = 1.40×**，且 ViT 不再被算两遍。
口径 A 从 3.09× "掉"到 2.89× 不是退步，是把此前白拿的 58.07 ms 计入了成本。
⇒ INT8 DiT 再省 **~14.5 ms**（DiT 1.33×，边界 ~1.15×），两个独立口径互相印证
（隔离量测 1.411× / 配对交替 1.329–1.422×，共 6 次独立跑，§6.9.3 + §6.14.2）。
**INT8 现为出厂默认档，且 k/v 两族留 bf16**（使用方拍板 §6.9.6 + 经验教训 §6.14）：
全门通过且**未放宽任何门**。代价与收益都要按最新一档读：
G4 max&#124;d&#124; 在 frame 300 是 **0.1940°**，而 bf16 档自己是 **0.1868°**
（六族全量化时是 0.364°，即 k/v 豁免把这项代价从 1.95× 缩到 **1.04×**）；
frame 0 的 0.4935° **三档逐位同值**，是 bf16 backbone 的地板而非 INT8 的代价（§6.14.3）。
两帧 decoded cos 仍 **0.999999**。退回六族全量化用 `dit_bf16_families=()`，
退回全 bf16 用 `use_int8_dit=False`。

> 边界口径：HF 的 `backbone.forward + action_head.get_action_with_features`
> 占其完整 `get_action` 的 **0.929**（两次独立插桩量测：359.21/387.09=0.9280、
> 370.76/399.01=0.9292；插桩本身因逐级 sync 抬高约 3%，故只用比值不用绝对值）。
> 详见 §6.6 / §6.7。厂商 Orin 最好成绩 216.5 ms 是 **1 相机**，不可直接比（§3.1）。

### 0.2 门禁状态

| 项目 | 状态 |
|---|---|
| 权重加载 + 完整性核对 | ✅ PASS（24/1030 抽样张量 rel=0.00e+00） |
| kernel 构建（`-DGPU_ARCH=87`） | ✅ 21m14s，0 error |
| 真实数据源（非合成） | ✅ **两个独立数据集**：`cube_to_bowl_5`（SO101 真机 AV1，Se=141，出厂门禁用）+ `green_to_blue_block_sim`（SO101 **仿真**采集 h264，Se=**148**，独立复验，§5.6）。后者的 PyAV 解码已在帧 100/101/107/15000/31000 上验证与 ffmpeg 后端**逐位相同** |
| 硬件锚点（锁频 1300.5 MHz） | ✅ 已测；所有量测脚本断言锁频，未锁频拒绝出数 |
| 厂商锚点 | ✅ 已找到（口径不同，见 §3） |
| HF eager 数值基线 | ✅ fixture 落地（N1.7 两帧 + aux 束；融合后另含 `pixel_values` / `input_ids`，见 §5.4；仿真集另 10 帧 + 10 aux，见 §5.6） |
| Orin frontend / pipeline | ✅ `frontends/torch/groot_n17_orin.py` + `models/groot_n17/pipeline_orin.py` |
| dispatch 注册 | ✅ `_PIPELINE_MAP` **与** `_SM87_ALLOWED` 双注册 |
| **image→embeds 融合** | ✅ 默认开（`fuse_image_embeds=True`）；`llm_input_embeds` 依赖已摘掉（§6.7） |
| **融合自身的算术门** | ✅ `test_fusion_reproduces_hf_embeds`：文本 embed **bit-identical**、kernel merger vs torch fp64 同输入 0.99999455 |
| **精度门（逐级 cos）** | ✅ G1–G4 通过；被消费张量 **≥0.995**（`backbone_features` 0.996951 / 0.998082，四条理由见 §6.7.2），其余 ≥0.999 |
| **图安全门（capture + stale-value）** | ✅ graph≡eager、replay≡replay 均 bit-identical；换输入后 replay 正确跟随 |
| **DiT 图参数护栏（§6.24）** | ✅ `_dit_graph_params` 三元组 `(num_inference_timesteps, action_horizon, num_timestep_buckets)`：与捕获时不符 ⇒ **一次性告警 + 退回 eager**（红线 #4），不再静默重放旧图。之所以敢回落而不是 raise：**V1c 实测 eager 臂与 horizon 顺序无关**（Sa=41 的前端跑 eager@20 与 Sa=21 的前端跑 eager@20，max&#124;d&#124; **0**）。修前实测偏 **3.3572°**（40→20）/ **21.4859°**（20→40）/ **3.5810°**（buckets→200）且**都不报错**；修后三条陈旧路径 **max&#124;d&#124; 0.000000e+00**，回退代价 **+64.120…+67.222 ms（1.6564…1.6855×）**已写进告警文案。**红线 #5**：新增 **C9** 用**重放计数**钉住图臂真的跑了——C5 只数**捕获**，一个静默回落到 eager 的回归它看不见 |
| **连续推理门（C1–C9，§6.15.2）** | ✅ **已进仓** `tests/test_orin_groot_n17_continuous.py`（**10 passed**）。8 个连续帧 + 1 个远端帧，一个前端：逐帧 decoded cos **0.999999232–0.999999628**、max&#124;d&#124; **0.194–0.387°**；repeat **逐位相同**；连续 == 一次性 **逐位相同**（两臂，§6.15.2）；DiT 图**捕获 1 次 / 复用 8 次**（对象与槽 `data_ptr` 身份均唯一，C5）；**C8** 用捕获计数证明 backbone 图臂真的跑了（它与 eager 臂逐位相同，数值门看不见绕过）；**C9** 用重放计数证明 DiT 图臂每观测都重放（§6.24）；负控制打断刷新后槽 cos **1.000000 → 0.809373**（C6，decoded action 另加 **10.0° 灾难界**，实测滞留值 **1.2337°**）；契约篡改 4/4 被拒 |
| **无 GPU 契约覆盖面（§6.24）** | ✅ 补掉了两块**零覆盖**：`flash_rt/datasets/lerobot_video.py` 的**索引→帧映射**（原 5 条测试全部只驱动 `_decode`；`load_frame`/`_episode_for`/`task_for_frame`/`_frame_index`/`_resolve_column`/`metadata` 一条没有）与 `attn_backend_groot_n17_orin.py`（**没有任何测试 import 过它**）。⚠️ 严重性如实记：映射错会把**同一个错帧**同时喂给 HF 参考和 FlashRT ⇒ **等价性主张不受影响**，坏掉的是**溯源**（"frame 100"、"593 连续帧"、"距 prompt 多远"这三根轴，以及 C6 远端臂 cos 0.900 的标定基准）。两块的负控各 4 / 7 个断点，**逐个跑红**（§6.24.4） |
| 测试 | ✅ `tests/test_orin_groot_n17_{dispatch,precision,preprocess,continuous}.py` + `tests/test_lerobot_video.py` — 真机档 **218 passed**（precision **31** + dispatch **102** + preprocess **65** + `test_lerobot_video` **20**）；仿真档 **244 passed, 0 skipped**（另加 continuous **10** 与 N1.6 后端门 16）。⚠️ 仿真档需**同时**给 `FLASHRT_GROOT_N17_FIXTURE_TAG=_sim` 与 `FLASHRT_GROOT_N17_FRAMES=100,107`：只给后者会让 precision **整模块 skip** 而总数看着仍像绿（§5.6、§6.24.7 教训 4）。两档共用同一套门，默认档由 `inspect.signature` 自动跟随。演进：63→67（§6.21）→80（§6.22.5）→179/204（§6.23）→**218/244**（§6.24，CPU-only 契约 **+37**、GPU 门 **+3**）。`test_orin_groot_n17_preprocess.py` 与 `test_lerobot_video.py` **不需 GPU、不需 checkpoint**（3.1 s / 1.7 s） |
| N1.6 通路 | ⏳ 侦察 ✅（fixture + aux + `docs/groot_n16_orin_sm87.md`）；frontend/pipeline ❌ 未开始 |
| DiT INT8（杠杆 #3） | ✅ **已交付且为出厂默认档，k/v 两族留 bf16**（§6.9 + §6.14）：精度门（§6.8）+ kernel 接入 + in-pipeline A/B + 三臂 A/B/C 全过。DiT **1.329×**（58.21→43.81 ms），边界 **124.9→106.91 ms（3.37×）**。退回六族全量化 `dit_bf16_families=()`，退回 bf16 `use_int8_dit=False` |
| LLM INT8（杠杆 #4） | ❌ **关闭，不落地**（§6.11，已接入实测后回退）：`backbone_features` cos **0.992077 < 0.995**（G4 看着无害 ⇒ 逐级门又一次拦住）；且 eager 只省 0.6–1.0 ms（GPU 层面省 3.93 ms，被 INT8 臂自己多出的 ~3.9 ms 不可重叠发射成本吃掉，§6.13.3） |
| ViT INT8（杠杆 #5） | ❌ **关闭**（§6.10.2 实测）：per-row 把 tap `vit_block_17` 打到 0.971（门 0.998）。⚠️ G4 看着无害（cos 0.999999）⇒ 逐级门不可省 |
| QuaRot（杠杆 #6） | ⏸️ **挂起**：三次真激活门后适用范围只剩 ViT，而 ViT 已按"尽量不用旋转"关闭 |
| backbone CUDA-graph 捕获（杠杆 #11） | ✅ **已交付**（§6.17）：`run_backbone_graph` 与 Thor FP8 / RTX FP8 mixin **同名同契约**。backbone **63.52 → 59.52 ms（−4.00，1.0673×）**、每观测 **115.44 → 111.40 ms（−4.03，1.0362×）**，`backbone_features` 与解码动作**逐位相同**（max&#124;d&#124; 0）。一次性 **259.7 ms ⇒ 64.9 帧回本**。默认开，只作用于 `infer(aux=...)`；关掉用 `use_backbone_graph=False`。⚠️ §6.13.2 那个 4.54 ms 是**裸 kernel 口径**，部署口径是 4.00 ms |
| 常驻 backbone runtime（杠杆 #12） | ✅ **已交付**（§6.13.1）：backbone **65.71→63.16 ms**，边界 **3.31–3.34× → 3.37×**，精度**逐位不变**，测试 **31→68**；顺带修掉一个 31 个 GPU 测试都拦不住的静默 bug（unfused 把观测载进 LLM 残差流） |
| **每观测入口（杠杆 #13）** | ✅ **已交付**（§6.15）：`infer(state, aux=...)` 让一个前端服务整条观测流；cross-KV **原地刷新**（重分配会让四张图读上一帧的 KV 且照样 replay 成功）。每观测 **117.56 ms → 3.29×**（vs 完整 `get_action`）/ **3.05×**（vs 同一边界）。顺带落地 fp32 权重缓存：刷新 **13.10→8.52 ms，逐位相同**。➡️ 杠杆 #11 落地后（§6.17）到 **111.40 ms**，cross-KV 改 bf16 tensor core 后（§6.19）到 **108.55 ms**，融合 RoPE 后（§6.22）到 **97.28 ms → 3.97× / 3.69×** |
| **融合 bf16 rotate-half RoPE（杠杆 #9）** | ✅ **已交付**（§6.22）：`rope_neox_qk_bf16` 把 ViT/LLM 的 Q+K RoPE 从 2× `_rope_rotate_half` torch shim（每次 ~7 个算子）压成 **1 次 launch**。原 `blocked_on_kernel` 判定是**空判**——kernel 早已编好，只是在 `flash_rt_qwen3_vl_kernels`（另一个扩展）里。与 shim **逐位相同**（3 个真实形状、两臂到 fp64 等距）⇒ 精度门数值一个没动，`backbone_features` / `infer` 输出 / decoded action 全 `torch.equal`；调用计数 **40 融合 vs 80 shim**（红线 #5）。每观测 **107.752 → 97.281 ms（1.1076×）**，边界 **106.47 → 94.35 ms（3.38× → 3.82×）**。缺 kernel 时**响亮告警**并回落 shim（红线 #4）。dispatch **67 → 80** |
| **每观测图像通路（杠杆 #14）** | ✅ **已交付**（§6.23）：`infer(state, frames=...)` 吃**裸 uint8 相机帧**，图像通路全在设备上跑（letterbox → 稠密 fp32 matmul 的连续覆盖 INTER_AREA → 中心裁剪 → 放大 → rescale/normalize → Qwen2-VL merge-block patchify），**无 HF、无 albumentations、无 PIL**。预处理 **12.916–14.027 → 1.448–1.485 ms**，每观测 **112.374–115.499 → 99.579–100.428 ms（省 11.945–15.410，1.1189–1.1540×）**。几何从 checkpoint 自己的 `processor_config.json` 读（`crop_fraction=0.95`，**不是** 0.9 的代码默认值），读不到就**拒绝**而不猜。精度：**12 帧真机数据 × 3 臂**，cv2 臂与 HF **九位小数全同**（⇒ T1 的移动全部归因于 GPU 链），T3/T4 **未放宽门**，只有 T1 新增 `THR_IMAGE_GPU_BACKBONE = 0.985`；负控（跳过裁剪+放大）T1 **0.9918 → 0.8989**，掉穿门 0.086。红线 #5：`frames=` 与 `aux=` 装同一份 `pixel_values` 时 `torch.equal=True` ⇒ 用**调用计数**钉（GPU 链 3 次 / cv2 参考 **0** 次 / plan 构建 **1** 次），并额外断言 GPU 臂的 `pixel_values` **与 fixture 不同**。顺带把每观测的 host 比较从 **5 次 / 77156 字节**降到 **0 次 / 0 字节**（契约项按**身份**重新呈现）。precision **17 → 29**、新增 **65** 条 CPU-only 契约 |
| ⚠️ **方法论纠正 1** | §6.11.5 曾从"CPU 提交 60.00 ms / 墙钟 65.77 ms（91.2%）"推出"backbone 是 CPU 提交受限"—— **该推论已撤回**（§6.13.3）。CPU 提交与 GPU 执行重叠，那个比值量的是 CPU 有多忙；launch 受限份额只能用 `(eager − replay)/eager` 量，实测 **7.3%** |
| ⚠️ **方法论纠正 2** | **decoded action 的 cos 门不住 stranded KV**（§6.15.3，负控制实测）：打断 cross-KV 刷新后，连续帧只退化 1.45×（cos 0.9999988）、远端帧 5.8×（cos **0.9999917**）—— **两种都仍过 0.999 的门**。层 1（槽内容 vs 该帧应有的 K）才是完全判别的：好 **1.000000000** / 打断 **0.809373** 且**精确等于**被滞留那帧的 K。⇒ stale-value 门必须打在内部张量或 velocity，不能打在 decoded action |
| ⚠️ **方法论纠正 3** | §6.22.3：只枚举 `flash_rt_kernels` 就下 `blocked_on_kernel` 判定，会让**已建好的 kernel 被记成缺口**（本例把 2.7 ms 记成实为 **11.902 ms** 的洞，偏低 **4.4×**）。本仓有 **三个**扩展模块，`csrc/qwen3_vl_bindings.cpp` 单独绑定 `flash_rt_qwen3_vl_kernels`。**规则：kernel 可用性审计必须枚举全部已建扩展，缺口判定必须附"在哪个模块里查过"。** 另一条：no-op 归因探针系统性**高估**（本例 11.478/11.902 = 96.4% 命中，差的 3.6% 正是融合 kernel 自己的 1.07 ms） |
| ⚠️ **方法论纠正 4** | §6.22.1：**关键字分桶对 C++ mangled 模板名不可靠，且同时朝两个方向错**。FA2 的 `fa2_vendor::flash_fwd_kernel<…cutlass::arch::Sm80…>` 被排在 attention **之前**的 GEMM 规则吃掉，而 attention 规则里的 `"flash"` 又匹配命名空间 `flash_rt::` ⇒ 注意力被记成 **0.312 ms / 0.3% / 21 launches**，实为 **2.335 ms / 2.28% / 172 launches**（偏低 **7.5×**），GEMM 同时虚高成 73.3%（真值 **64.1%**）。这是同一份普查**第二次**栽在分桶上（第一次是 `DefaultGemmWi…` 归进 copy/cast）。**规则：分桶按精确名字前缀、attention 排在 GEMM 之前、打印每个桶的成员名单而不只是总量**；更正后要能自证（本例：GEMM 在两臂之间只差 0.12 ms，而 rope 改动不该碰 GEMM）。⚠️ 本次结论没变（FA2 仍是小头），**但画像数字错了不会立刻现形，只在未来的排序决策里发作** |
| ⚠️ **方法论纠正 5** | §6.23.1：**口径 C 的分子分母边界不一致**，与 §6.7 当初把口径 A 拆成 A/B 的原因同类。口径 C 记的 **97.28 ms → 3.97×** 中，97.28 是**模型侧**每观测（`aux` 来自磁盘 fixture，**不含**图像预处理），而分母 386.27 ms 是 HF **含**自己图像预处理的完整 `get_action` ⇒ **3.97× 偏乐观**。补齐成**口径 D**（两边都含图像预处理）后是 **3.85–3.90×**。**这是本文档第二次因为"边界不对称"而下修一个已入账的倍率**（第一次是口径 A→B，把白拿的 58.07 ms 计入成本）。**规则：任何倍率入账前，逐边问一句"这一侧含不含预/后处理"。** 同一轮还纠正了一个账面数字：那个 **~27.55 ms pre/post** 是**两个独立量测的总量相减**得来的，而本次计划又从**分段中位数相加**推出 "~10.3 ms"，整链实测是 **12.916–14.027 ms**（偏低 1.25–1.36×）。**规则：总量只引用端到底端口径，分段只用于归因** |
| ⚠️ **方法论纠正 6** | §6.23.6 教训 5：**跑门禁套件时编辑被测源文件会造出与真失败一模一样的假失败**。仿真档首轮报 `test_the_eager_backbone_arm_stays_reachable` 失败，单独跑通过、三条断言手工验证也成立；根因是套件运行途中改了 `groot_n17_orin.py` 的 docstring——那条测试用 `inspect.getsource(CLS.infer)` 做源码钉子，`inspect` 走 `linecache` **会按 mtime 重新读盘**，而进程里的 code object 仍带**改动前**的 `co_firstlineno` ⇒ 抽出的源码块整体错位 3 行。⚠️ **本仓大量使用源码钉子**（§6.22 教训 3 的产物），所以这是一个**新增的失效面**：数值测试完全不受影响，只有源码钉子会红。**规则：门禁套件运行期间只许改 `.md`；改完 `.py` 必须重跑。**（重跑后仿真档 **204 passed**） |

---

## 1. 平台与构建

| 字段 | 值 |
|---|---|
| 设备 | Jetson AGX Orin 64GB，SM87（cc 8.7），**16 SMs** |
| L4T | R36.4.7（JetPack 6.2），Ubuntu 22.04.3，kernel 5.15.148-tegra |
| 原生 FP8 / FP4 | **无**（Ampere；`ENABLE_NVFP4` 在 SM87 上 DISABLED） |
| torch / CUDA | 2.3.0 / nvcc 12.2（`torch.version.cuda=12.2`） |
| Python | 3.10.15 |
| 构建 | `-DGPU_ARCH=87 -DENABLE_SM80_INT8_CUTLASS=ON -DFLASHRT_BUILD_QWEN3_VL=ON -DFA2_ARCH_NATIVE_ONLY=ON -DFA2_HDIMS='64;96;128;256' -DFA2_DTYPES='bf16;fp16'` |
| 产出 | `flash_rt_kernels.so`(6.6 MB) / `flash_rt_qwen3_vl_kernels.so`(0.6 MB) / `flash_rt_fa2.so`(93 MB) |
| 前置 | CUTLASS 需手动 vendor：`git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass`（仓库不自带，CMakeLists.txt:342 fail-fast） |

venv（`--system-site-packages` 共享系统 torch 2.3.0）：

| venv | transformers | 备注 |
|---|---|---|
| `/mnt/venvs/groot_n17` | 4.57.1（系统） | N1.7 官方 orin pin 是 4.57.6，4.57.1 实测可加载且权重完整 |
| `/mnt/venvs/groot_n16` | **4.51.3**（降级） | N1.6 官方 pin；需额外 `lmdb`（vendored Eagle remote code 导入它） |

> ⚠️ N1.6 与 N1.7 的 transformers 版本互斥，**必须两个 venv**，不能共用。

---

## 2. 硬件锚点（实测，锁频 1300.5 MHz）

`nvpmodel`/`jetson_clocks` 在容器内不存在，由宿主机执行 `nvpmodel -m 0 && jetson_clocks`；
容器内验证 `/sys/class/devfreq/17000000.gpu/{cur,min,max}_freq` 三者均为 1300500000。
**所有量测脚本都断言这一条，未锁频直接拒绝出数**（原则 #16）。

| 锚点 | 实测值 | 备注 |
|---|---|---|
| read-only 带宽（1 GB buffer，`torch.sum`） | **97.2 GB/s** | 与 `docs/hyvla05_orin_sm87.md` 的 97 GB/s 一致 |
| read-only 带宽（256 MB） | 154.9 GB/s | 小张量偏高，非权重流场景 |
| copy r+w（1 GB） | 158.0 GB/s | |
| bf16 cuBLAS 峰值（4096³） | **29.08 TFLOPS** | |
| fp16 cuBLAS 峰值（4096³） | 26.60 TFLOPS | |
| bf16 GEMM 有效带宽（M=41，18.9 MB 权重） | 157–160 GB/s | 大权重接近 copy 上限 |
| `torch._int_mm` INT8（4096³） | 10.42 TOPS | **不可用**：未走 tensor core，必须用 FlashRT CUTLASS kernel |

### 2.1 必须纠正的两处既有文档错误

1. `docs/deployment_orin.md` 硬件表写 **"5.3 TFLOPS BF16 / 60 TOPS INT8"**。
   实测 bf16 峰值 29.08 TFLOPS（4096³），理论峰值 16 SM × 2048 FLOP/clk × 1.3005 GHz
   = **42.6 TFLOPS**，INT8 理论 85.2 TOPS。该表的 BF16 数字错了约 4–5.5×，
   且 60 TOPS / 5.3 TFLOPS = 11.3× 的比值在 Ampere 上不可能（INT8 dense 恰为 BF16 的 2×）。
   以本文 §2 实测为准。
2. `flash_rt/configs/groot.yaml` 写 `ffn_activation: geglu`（并建议用
   `gate_geglu_merged_*` kernel）。**错误**。权重实测 DiT FFN 是
   `ff.net.0.proj (6144,1536)` → `ff.net.2 (1536,6144)`，即 inner=6144 的**单层 Linear**；
   GEGLU 的 proj 应为 (12288,1536)。官方 `gr00t/model/modules/dit.py:234,429` 默认
   `activation_fn="gelu-approximate"` 并在 :269/:451 透传给 block，与权重形状一致。
   FlashRT 自己在 `flash_rt/models/groot_n17/weight_spec.py:264` 也记了这一点。
   → **激活是 tanh 近似 GELU，对应 kernel 是 `bias_gelu_bf16_strict` [PRE]**。

> 这条恰好与 hyvla 相反：`docs/hyvla05_orin_sm87.md:156` 把 `bias_gelu_bf16_strict`
> 列为 dead end，理由是"它用 tanh 近似而 hyvla 参考用 exact erf"。GR00T 的参考
> **就是** tanh 近似，所以该 kernel 在 GR00T 上是精确匹配，不是妥协。

---

## 3. 厂商锚点与口径

NVIDIA 官方 `Isaac-GR00T` 仓库 `scripts/deployment/README.md:171-176`
（GR00T N1.7，**4 步去噪，1 相机**，模型 `GR00T-N1.7-LIBERO/libero_10`）：

| Orin | Data Proc | Backbone | Action Head | E2E | 频率 |
|---|---|---|---|---|---|
| PyTorch Eager | 9.45 ms | 127.6 ms | 205.39 ms | **342.8 ms** | 2.9 Hz |
| torch.compile | 9.45 ms | 128.59 ms | 78.94 ms | **217.0 ms** | 4.6 Hz |
| TensorRT (DiT-only) | 9.45 ms | 128.38 ms | 78.6 ms | **216.5 ms** | 4.6 Hz |

官方注：*"Orin uses DiT-only TensorRT because TRT 10.3 does not support the
backbone engine."* → **NVIDIA 在 Orin 上完全没有优化 backbone，128 ms 全留在 eager。**
这正是 FlashRT 的差异化空间（本仓库已有 SM87 验证过的 Qwen3-VL bf16 kernel）。

### 3.1 口径差异（必须声明，否则对比无效）

| 维度 | 厂商锚点 | 本适配 |
|---|---|---|
| 相机数 | **1** | **2**（front + wrist） |
| embodiment tag | `LIBERO_PANDA` | `NEW_EMBODIMENT` |
| Se（LLM 序列长） | 未知（1 相机 ⇒ 约 64 image + text） | **141**（实测） |
| Sa（DiT 序列长） | 41 | 41 |

1 相机 ⇒ image token 数减半 ⇒ ViT 与 LLM prefill 都更便宜。
**厂商的 216.5 ms 不能与本文 2 相机数字直接比较**；后续所有对比都必须带相机数与 Se。

### 3.2 与厂商数字的交叉校验（实测 vs 厂商）

用 32 层 DiT 的 GEMM-only 代理（M=41，224 个 GEMM，1082.1M 参数 = 2.164 GB bf16，
锁频，best-of-5 × 100 iters）：

| | 每步 | ×4 步 |
|---|---|---|
| 厂商 eager action head | 51.35 ms | 205.39 ms |
| 厂商 TRT action head | 19.73 ms | 78.94 ms |
| 本文 bf16 eager（GEMM-only） | 15.29 ms | 61.17 ms |
| 本文 bf16 CUDA graph（GEMM-only） | **14.33 ms** | **57.34 ms** |

推论：
- 厂商 TRT(19.73) − 本文 GEMM-only graph(14.33) = **~5.4 ms/步** 是 attention + norm + elementwise，
  这部分本文尚未计入，属待测项。
- 厂商 eager(51.35) − 厂商 TRT(19.73) = **31.6 ms/步** 纯 launch/dispatch 开销
  ⇒ Orin 上 eager PyTorch 严重 launch-bound，与 hyvla 的经验一致（其 eager 参考 3137 ms）。
- **CUDA graph 对 DiT GEMM 链本身只有 1.07×**（15.29→14.33 ms）。
  bf16 DiT 已达实测权重流上限的 **95.6%**（14.33 ms vs 158 GB/s 下的 13.70 ms 下限）。
  ⇒ *推翻*了"每 GEMM ~20 µs 固定开销 × 768 次 ≈ 15 ms"的初步假设；
  单独 µbench 里小 GEMM 只有 95 GB/s 是 cold-L2 假象（原则 #16 的 µbench 反转陷阱）。
- **结论：bf16 DiT 已无空间，唯一杠杆是减少字节数（INT8 / INT4）。**

---

## 4. 模型结构（实测，权重为准）

checkpoint：`/mnt/GR00T/so101_sim_rynnbot/checkpoint-89-1.000`
（`architectures: ["Gr00tN1d7"]`，`base_model_path: .../GR00T-N1.7-3B`，
微调 89 步只 tune projector；1030 个张量，与 `docs/groot_transformers5_weight_corruption.md`
记录的 N1.7「1030/1030」一致 ⇒ **与 Thor 适配用的是同一批权重**）。

`Gr00tN1d7` 总参数 **3144.02M**（加载时打印值与本文按 safetensors header 累加值完全一致）。

| 组件 | 参数量 | 结构（权重键实测） |
|---|---|---|
| Qwen3-VL ViT | 331.42M | `visual.blocks` 24 层，hidden 1024，16 头，intermediate 4096，patch 16，temporal_patch 2，spatial_merge 2 |
| DeepStack | 75.54M | `visual.deepstack_merger_list` 3 个，taps `[5,11,17]` |
| LLM | 805.38M | `language_model.layers` **16** 层（官方代码加载完整 28 层后 `pop` 到 `select_layer=16`，见 `qwen3_backbone.py:85`），hidden 2048，16Q/8KV，head_dim 128，FFN 6144，rope_theta 5e6，mrope_section `[24,20,20]` interleaved |
| embed_tokens | 311.16M | (151936, 2048)，**gather only**，不计带宽 |
| vlln | ~0 | LayerNorm(2048) |
| vl_self_attention | 201.43M | 4 层 `SelfAttentionTransformer`，dim 2048，32 头×64，FFN 2048→8192→2048 GELU |
| AlternateVLDiT | **1091.72M** | 32 层：偶数层 cross-attn（to_k/to_v 的 K=**2048**）、奇数层 self-attn（K=**1536**）；偶数层 34.620M / 奇数层 33.048M；FFN 单层 Linear inner 6144 + GELU(tanh)；`norm1.linear (3072,1536)` ada_norm |
| action/state encoder, decoder | 232.93M + 54.74M + 37.92M（表） | **per-embodiment 表，leading dim 32**；单次推理只取 1 行 ⇒ 活跃约 7M，**不是 325M** |

### 4.1 实测运行时形状（2 相机，真机数据，`new_embodiment`）

| 阶段 | 形状 |
|---|---|
| ViT 输入 → 输出 | `(512, 1536)` → `(128, 2048)` |
| LLM | `(1, **141**, 2048)` = 128 image + 13 text |
| vlln | `(1,141,2048)` |
| vl_self_attention | `(1,141,2048)` → `(1,141,2048)`，**DiT 之前跑 1 次** |
| DiT block | `(1, **41**, 1536)` → out `(1,41,1024)` |
| 最终 action | `(1, **16**, 6)` |

> DiT 按 `action_horizon=40` 算 41 个槽位，但 `new_embodiment` 的
> `action delta_indices` 只有 16 ⇒ 只解码 16 步、只有前 6 维有效。
> **把 horizon 改成 16 可省下 DiT 约 2.4× 的 M 维工作量，但那是改推理超参**，
> 与 `docs/groot_n16_thor_sm110.md` 的"未为提速改动任何推理超参"原则冲突，
> 只作为模型侧杠杆记录，不在本适配中启用。

⚠️ FlashRT 现有 N1.7 代码里 `ni=256 / Se≈277`（`groot_n17_thor.py:1397-1398` 注释）
对应的是**另一个 embodiment/分辨率**的 fixture，不是本文的 2 相机 SO101 场景。
照抄该 Se 会把 LLM prefill 成本估高约 2×（实测 Se=277 的 qkv GEMM 0.492 ms
vs Se=196 的 0.160 ms，非线性，疑似跨 128 行 tile 边界）。

---

## 5. 数据（真实分布，非合成）

按 AGENTS.md §3.7 与 skill 原则：**标定与精度评测输入必须来自宿主真实推理分布，
经宿主自己的预处理链构建**。合成/随机张量会测错激活离群值结构，进而选错量化配方。

实测证明这不是理论担忧：

| | 真机帧（`cube_to_bowl_5`） | 早期随机探针 |
|---|---|---|
| 图像 mean / std | 130.8 / **42.1** | 127.5 / **73.9** |
| state 取值 | `[5.74, -96.70, 94.25, 77.96, -1.14, 0.54]`（度） | `randn×0.1` ⇒ **±0.3** |

随机 state 的量级错了约 **300×**，会直接毁掉归一化、DiT 输入分布和所有 INT8 门限。

### 5.1 数据源

- 原训练集不在本机（`trainer_args.json` 指向 `/mnt/data/xueshengke/dataset/...`，`/mnt/data` 不存在）。
- 采用官方 `Isaac-GR00T` `demo_data/cube_to_bowl_5`：`robot_type: so101_follower`，
  5 episodes / 4148 帧 / 30 fps，任务 `cube into yellow bowl`、`cube into green bowl`。
  与 N1.7 checkpoint 的训练集 `so101/stack_color_cubes_24_rynnbot` 同机器人同任务族。
- 视频是 **AV1（libdav1d）**；`opencv-python` 在 aarch64 上无 AV1 软解，
  必须走 `video_backend="ffmpeg"`（系统 `/usr/bin/ffmpeg` 带 libdav1d）——
  与 FlashRT `tests/_helpers/groot_n17/gen_reference.py:141` 的注释同一结论。
- `demo_data` 的 mp4/parquet 是 **git-LFS 指针**（132 B），需
  `git lfs pull --include="demo_data/cube_to_bowl_5/**"` 才是真数据。

### 5.2 需要的元数据适配（仅改元数据，真实字节不动）

checkpoint 的 `new_embodiment` tag 要求 language 键为 `annotation.prompt`，
而 `cube_to_bowl_5/meta/modality.json` 声明的是 `human.task_description`
（两者 `original_key` 都是 `task_index`）。video（`front`/`wrist`）与
state（`observation.state` [6]）键**已完全匹配**。

做法：`/mnt/groot_realdata/cube_to_bowl_5/` 下放**真实 `meta/` 的副本（仅 annotation 键改名）**，
`data/` 与 `videos/` 用**符号链接**指向原数据集。已验证 video/state/action 三段
与源文件字节一致，源 `modality.json` sha256 `cc0229b172325386` 未被修改。

### 5.3 加载器

- 参考生成走**官方链路**：`LeRobotEpisodeLoader` → `extract_step_data` →
  `parse_observation_gr00t` → `Gr00tPolicy.get_action`（保证预处理链与宿主一致）。
- FlashRT 侧新增 `flash_rt/datasets/lerobot_video.py` [PHASE 0]：
  video-backed LeRobot v2.x 读取器，补齐 `flash_rt/datasets/libero.py` [PRE] 只支持
  parquet 内嵌 PNG 的空缺。契约与 `LiberoDataset` 对齐（`metadata` / `load_frame` /
  `load_calibration_obs`，复用 `flash_rt.core.calibration.stratified_sample_indices` [PRE]），
  且 **torch-free / gr00t-free**，因此加速路径可在没有 GR00T 训练代码的部署环境里取真实标定数据
  （skill 原则 #10）。已验证：跨 episode 读取、seek 回退、同帧重复读 bit-identical、
  分层抽样覆盖 episode 0–3。

### 5.4 HF eager 真机数据基线（实测）

`tests/_helpers/groot_orin/gen_reference.py` [PHASE 1]，2 相机，`new_embodiment`，
锁频 1300.5 MHz，seed=0，median of 7（warmup 2）：

| 版本 | HF eager E2E | state 单位 | 阶段块数（已断言） | 激活张量 | fixture |
|---|---|---|---|---|---|
| **N1.7** | **387.93 / 387.22 ms**（frame 0 / 300）⇒ 2.58 Hz | **deg2rad**（§5.5） | vit=24, deepstack=3, llm=16, vlsa=4, dit=32 | 79 | `tests/fixtures/gr00t_n17_ref_new_embodiment_2v_frame{0,300}_seed0.pt`（91 MB）+ `_aux.pt`（**11.44 MB**，为 §6.7 的融合重生成，新增 `pixel_values` (512,1536) 与 `input_ids` (1,141)） |
| N1.6 | **349.2 / 349.0 ms** ⇒ 2.86 / 2.87 Hz | raw（度，§5.5） | vit=27, llm=16, **vlln=1**, dit=32, projector=1 | **77** | 同上，`gr00t_n16_*`（125.6 MB）+ `_aux.pt`（11.46 MB）—— N1.6 权威见 `docs/groot_n16_orin_sm87.md` |

块数由脚本**硬断言**（数量不符直接抛错拒绝写 fixture），防止 backbone 被静默截断后
产出会误判门的参考。`state_unit` 写进 fixture meta，并由 aux 脚本回读
（**绝不在两处各存一份**）。

> ⚠️ **N1.7 的 379.9 / 380.4 ms 已作废**：那批 fixture 是在 §5.5 的单位错误下生成的
> （state 以「度」喂给一个 statistics 为「弧度」的 checkpoint）。修正后重测得
> 387.93 / 387.22 ms（+2.1%，同一量级，因为 state 只影响 DiT 的 token 0，
> 不影响 backbone 的工作量）。**所有引用旧数字的结论都需以本节为准。**
>
> ✅ **N1.6 的 fixture 与 aux 束已用正确的 checkout 重生成**（此前失败于
> `ValueError: model type 'Gr00tN1d6' but Transformers does not recognize this
> architecture` —— 误用了 N1.7 的 checkout）。**必须**
> `PYTHONPATH=/mnt/Isaac-GR00T-n16:/mnt/FlashRT` + `/mnt/venvs/groot_n16`
> （transformers 4.51.3）；两个 venv 的 transformers 版本互斥。
> 重生成时顺带给 `gen_reference.py` 补了 **`vlln` 阶段采集**：N1.6 没有
> `vl_self_attention`，`vlln` 的输出就是 DiT 消费的 context，不采集它就
> **没有任何张量能门住 LLM→DiT 这一段**（N1.7 因后面还有 `vlsa_block_0..3`
> 而被间接覆盖，其现存 79-张量 fixture 早于该 hook，重新生成即会带上）。
> N1.6 的完整侦察结论见 `docs/groot_n16_orin_sm87.md`。

**与厂商锚点交叉校验**：NVIDIA 官方 Orin eager 是 **342.8 ms @ 1 相机**（§3）；
本文 N1.7 **387.9 ms @ 2 相机**。多一路相机多出 ViT + LLM prefill 工作量，
342.8 → 388 的 +13% 与口径差一致 ⇒ **本文基线与厂商 eager 同源可比**，
harness 未系统性偏快或偏慢。

**预测有效性（不是只看数量级）**：frame 0 的真实数据上（弧度空间），
`pred[step0] = [-0.124, -1.789, 1.317, 1.401, -0.043, -0.276]` rad vs
数据集真实 action `[0.104, -1.650, 1.624, 1.566, -0.021, 0.007]` rad，
**cosine = 0.98312**，最大绝对误差 0.307 rad（17.6°）；frame 300 为
cos 0.98973 / 0.270 rad（15.4°）。16 步全 chunk 落在 **[-2.32, 2.65] rad**
（= [-133°, 152°]），在 SO101 关节限位内。

⇒ 两点结论：(a) **视觉确实进入了策略**，排除"权重被静默随机化 ⇒ 只跟随 state"
那类故障；(b) 15–18° 的 step-0 误差是**数据域差**而非 bug —— checkpoint 是
`so101_sim_rynnbot`（**仿真**训练），评测帧是 `cube_to_bowl_5`（**真机**）。
因此 **fixture 的角色是数值参考**（FlashRT vs HF eager，同权重同输入），
**不是任务性能基准**；pred-vs-truth 只用作 §5.5 单位正确性的 sanity check。

> ⚠️ N1.6 的 `checkpoint-10-1.000` 只微调了 **10 步**（`trainer_args.json: max_steps=10`），
> 本质是基座权重 + 几乎未训练的 projector，其预测 action 范围（-181.9 ~ 147.5）
> 超出真实物理范围。**因此 N1.6 不能用"action 贴近真值"当门**；
> 它作为**数值参考**完全有效（比的是 FlashRT vs HF eager 同权重同输入，不是 vs 真值）。

### 5.5 state 单位不对称：N1.7 是弧度，N1.6 是度（实测踩坑）

**症状**（正是"必须用真实数据"这条要求直接抓出来的 bug）：归一化后的 state 是
`[6.93, -78.71, 79.30, 178.50, -1.56, 0.45]`，而 `normalize_state` 按
`2(v-q01)/(q99-q01)-1` 的定义域应当落在 **≈[-1, 1]**。偏出约 **57×**。
随机 state 永远抓不到这一条 —— 它本来就没有单位。

**根因**：`cube_to_bowl_5` 的 `observation.state` 以**度**存储
（`[5.74, -96.70, 94.25, 77.96, -1.14, 0.54]`），而 **N1.7 checkpoint 的
`statistics.json` 的 q01/q99 是弧度**。

**证据**（百分位匹配，而非猜测）：把 demo 数据换算成弧度后取 q01/q99，
与 checkpoint 的 statistics 对齐；用度则完全对不上：

| | q01 | q99 |
|---|---|---|
| demo → **弧度** | `[-0.505, -1.731, -1.559, 0.896, -0.192, 0.006]` | `[0.190, 1.005, 1.710, 1.745, 0.044, 1.103]` |
| demo → 度（错） | `[-28.9, -99.2, -89.3, 51.3, -11.0, 0.34]` | `[10.9, 57.6, 98.0, 100.0, 2.5, 63.2]` |
| **checkpoint** | `[-0.760, -1.741, -0.796, 0.798, -0.724, 0.0016]` | `[0.879, 0.703, 1.571, 1.658, 0.757, 0.749]` |

弧度行的量级与逐维符号结构都对得上；度行差 57.3×（= 180/π）。

**N1.6 相反**：其 `statistics.json` 的 q01/q99 **就是度**，因此 **不换算**。
⇒ 修正是 **N1.7-only** 的，不能写成"GR00T 都要换算"。

**修法**（在数据侧，不改模型侧一行）：`gen_reference.py` 增加
`STATE_UNIT = {"n17": "deg2rad", "n16": "raw"}` 表与 `--state-unit` 开关，
`build_obs` 返回 `(obs, raw, unit)` 三元组，调用点断言
`applied_unit == state_unit`，并同时打印"state as stored"与"state fed to model"。
fixture meta 落 `state_unit`，`raw` 段落同时保存 `state_fed` / `action_fed`。

**修正后**：归一化 state = `[0.0498, -0.9567, 1.0627, 0.3080, -0.0488, -0.9789]` ✅
落在 [-1,1]；预测 action 从 N1.6 那种 -181.9~147.5 的单位混乱空间变成
N1.7 的 [-2.32, 2.65] rad。

**下游影响（必须记住的一条）**：§6.5 的激活离群值剖面里，**只有 DiT 各行受影响**
—— state 经 state encoder 成为 DiT 41-token 栈的 token 0，
而 ViT / DeepStack / LLM / VLSA 只由图像与文本驱动，与 state 无关。
所以 backbone 各行仍然有效，**DiT 各行必须在修正后的 fixture 上复测**（已在 §6.5 复测）。

### 5.6 第二个独立数据集：仿真采集的 `green_to_blue_block_sim`（复验用）

**来源**：使用方提供 `/mnt/3ccddd530eb34632a126596a640adbaa.zip`，解压到
`/mnt/groot_simdata/3ccddd530eb34632a126596a640adbaa/`（原始字节未改动）。
LeRobot v2.1 布局：**50 episodes / 31166 frames / 30 fps / 2 相机**
（`observation.images.front` + `.wrist`），h264 yuv420p 640×360，
state/action 是 6 维关节。任务是 "Pick up green block and put it on the blue block"。

**为什么值得单开一节**：出厂门禁全部建立在 `cube_to_bowl_5`（真机 AV1，Se=141）
的**两帧**上。两帧 + 一个数据集不足以支撑"这个通路对"的主张 —— 
尤其是 Se 是常驻 runtime 的 key 之一（杠杆 #12），
而 DiT 的 `Skv_text/Skv_image` 又随 Se 变。第二个数据集给出
**不同编码器（h264 vs AV1）、不同 Se（148 vs 141）、不同采集方式（仿真 vs 真机）**
的独立复验。

**元数据适配（沿用 §5.2 的先例，真实字节不动）**：
`/mnt/groot_realdata/green_to_blue_block_sim/` 只放 `meta/`（`annotation.prompt`
按 §5.2 改名），`data` 与 `videos` **符号链接**回原目录并逐字节校验；
源目录的 sha256 不变。

**加载器改动（`flash_rt/datasets/lerobot_video.py`，一处，通用）**：
该数据集的列名是 `observation.state.joint` / `action.joint`，
而加载器默认找 `observation.state` / `action`。新增 `_resolve_column`：
精确命中 → 用之；否则若**恰有一个** `requested + "."` 前缀的 feature → 用它并
`logger.warning` 记录替换；否则 raise 并列出可用列。
**不做模糊匹配**（红线 #4：宁可响亮失败也不静默近似）。
已验证在帧 0 / 5000 / 31000 上正确解析并解码。

**解码后端复验**：PyAV（加载器用的）与 ffmpeg CLI 在帧
100 / 101 / 107 / 15000 / 31000 × 2 相机 = **10 次解码全部逐位相同**。
这是 fixture provenance 里那句话的依据，不是照抄 §5.1 的措辞。

**fixture**：10 帧（100–107 连续 + 15000 + 31000）各一个参考
（~92 MB，80 个激活张量）+ 一个 aux 束（~11.5 MB）。
帧 100–107 连续是为了 §6.15 的连续推理门；31000 在**另一个 episode**，
是 §6.15.3 那个负控制的远端臂。

**HF eager 基线（锁频，median of 7）**：**384.4–387.6 ms**
（`cube_to_bowl_5` 上是 387.93 ms，差 <1%）；同一边界 **358.84 ms**（× 0.929）。

**门禁套件参数化，不复制测试**：`tests/test_orin_groot_n17_precision.py` 的
fixture tag 与帧列表改为环境变量驱动 ——

```bash
FLASHRT_GROOT_N17_FIXTURE_TAG=_sim \
FLASHRT_GROOT_N17_FRAMES=100,15000 \
PYTHONPATH=/mnt/Isaac-GR00T \
FLASHRT_GROOT_N17_CHECKPOINT=/mnt/GR00T/so101_sim_rynnbot/checkpoint-89-1.000 \
  pytest tests/test_orin_groot_n17_precision.py -q
```

同一套 **17 个门**在第二个数据集上**全过**，且延迟与配对 A/B 与第一个数据集
一致到 0.5% 以内：

| | `cube_to_bowl_5`（Se=141） | `green_to_blue_block_sim`（Se=148） |
|---|---|---|
| backbone | 63.16 ms | **63.23 ms** |
| DiT（graph） | 43.75 ms | **43.65 ms** |
| 同一边界合计 | 106.91 ms → **3.37×** | 106.89 ms → **3.36×** |
| 配对交替 DiT 档比值 | **1.329×** | **1.336×** |

⇒ **通路对 Se 与数据集都不敏感**，常驻 runtime 的 shape keying 在 Se=148 上正确重建。

**⚠️ 顺带发现并修掉的一个 provenance 缺陷**：`gen_reference.py` 的
`meta.provenance` 原是**硬编码**的 "real robot frames (SO101 cube_to_bowl_5, AV1)"。
加 `--tag _sim` 复用该脚本后，10 个**仿真** fixture 全都声称自己是真机 AV1 帧 —— 
数据溯源说谎，而 AGENTS.md §3.7 把 provenance 列为验收项。
修法遵循本文一贯的做法（**派生而非复述**，同 `_shipped_default_int8()` 从构造函数读默认值）：
加 `--provenance` 参数，默认值就是原来那句话（`DEFAULT_PROVENANCE`，
所以既有 cube_to_bowl_5 捕获不受影响），并在 `[cfg]` 启动行回显。
已落盘的 10 个 fixture 就地改写该字段，**其余 payload 逐张量 sha256 校验不变**。

---

## 6. Roofline 与杠杆树

### 6.1 每次推理的活跃字节与 FLOPs（2 相机，Se=141，Sa=41，4 步）

| 阶段 | FLOPs | 权重字节(bf16) | 主导机制 | 预测 |
|---|---|---|---|---|
| ViT（512 tok，24 blk） | ~309 G | 604 MB | 计算 | ~17 ms |
| DeepStack ×3 | ~77 G | 151 MB | 计算 | ~4 ms |
| LLM prefill（Se=141，16 层） | ~227 G | 1.61 GB | 计算/带宽均衡 | ~12 ms |
| vl_self_attention（Se=141，4 层） | ~57 G | 403 MB | 均衡 | ~3 ms |
| **DiT ×4 步（M=41）** | ~358 G | **8.73 GB** | **带宽** | **~79 ms**（GEMM 57.3 实测 + ~21.6 非 GEMM 推算） |
| encoder/decoder（1 embodiment 行 ×4） | 小 | ~56 MB | — | ~0.5 ms |
| **合计** | ~1.03 T | ~11.5 GB | | **~124 ms** |

DiT 占 bf16 总量的 **~64%**，且其中 75% 的字节来自"4 步各重读一遍全部 DiT 权重"。

### 6.2 杠杆树（按 ROI 排序）

| # | 杠杆 | 机制 | 预测收益 | 状态 |
|---|---|---|---|---|
| 1 | **CUDA Graph + 融合 elementwise（全链路）** | 消除 launch/dispatch 开销；厂商 eager→TRT 的 126 ms 差几乎全是这个 | 342.8 → ~124 ms（**~2.8×**） | ✅ **已交付**：DiT eager→graph **1.97×**（115.0→58.4 ms）；边界倍率见杠杆 #10（两个口径） |
| 2 | **跳过 lm_head 与 `output_hidden_states`** | 官方 `qwen3_backbone.py:143` 调 `self.model(..., output_hidden_states=True)` 取 `hidden_states[-1]`，**却算完并丢弃 (141,151936) logits**（~87.6 GFLOP + 43 MB 写），且保留全部 16 层 hidden states | 免费 ~4–5 ms | ✅ **已交付**：Orin pipeline 自算 16 层，从不构造 lm_head；并核实 `hidden_states[-1]` 是 **pre-final-norm**（§6.6.2） |
| 3 | **DiT INT8 W8A8** | 8.73 GB → 4.37 GB | µbench 预测 1.56×；**前提是 CUDA graph**，无图反而慢 2.1× | ✅ **已交付**（§6.8 精度门 + §6.9 接入）。**实测 1.39–1.42×**：4 步 58.5 → 41.5 ms（省 ~17 ms），边界 **124.9 → 108.0–108.8 ms（3.31–3.34×）**。比预测低 9.6%，因为 µbench 是合成的、没算不被加速的 attention/norm/bias/residual（§6.9.4）。G1–G5 全过、未放宽任何门；**per-row 是硬要求**（per-tensor 差 2.5×）。**现为出厂默认档**（`use_int8_dit=False` 退回 bf16） |
| 4 | LLM prefill INT8（动态 per-row） | LLM 现占 backbone 38.4%（§6.6.3）；权重流量减半 | ~22.5 → ~13 ms（**预测偏大 2.4×**） | ❌ **关闭，不落地**（§6.11 已接入并实测后回退）。两条独立的否决理由：**(a) 精度不过门** —— `backbone_features`（DiT 真正消费的张量）cos **0.992077 < 0.995**，而 G4 看着无害（0.329°，比 bf16 档还小）⇒ §6.10.1 的 fake-quant 门**漏量了 VLSA 之后那一级**；**(b) 收益拿不到** —— GPU 层面确实省 **3.93 ms**（捕获后 4.77 ms），但 eager 墙钟只省 **0.6–1.0 ms**：INT8 臂自己多带了 **~3.9 ms 藏不住的发射成本**（32 个额外 quantize kernel，每个 GPU 时间极短 ⇒ GPU 在它们之间饿着），见 §6.13.3 的反推表。⚠️ 另测得 **k/v 在 INT8 下更慢（0.851×）** ⇒ 即使复活也必须排除 k/v |
| 5 | ViT INT8 | hyvla 在此**失败**（E2E cos 0.997459，且只省 7 ms） | 不确定；ViT 现占 backbone 41.2% | ❌ **关闭**（§6.10.2 实测不过门）：per-row 动态量化把**被消费的 tap** `vit_block_17` 打到 **0.971083**（门 0.998），per-tensor 崩到 **0.5127**。⚠️ 但 G4 仍是 cos 0.999999/0.740° ⇒ **只看端到端会误判**，这是逐级门存在的理由 |
| 6 | QuaRot/Hadamard 旋转 | 若 INT8 因离群值不过门：旋转修的是离群值**条件数**，位宽修的是**噪声**，两轴独立（原则 #17）。Chameleon/Orin 上 W8A8+rotation 曾同时压倒 W8A8 与 W4A16 | 使 #5 过门 | ⏸️ **挂起**：三次真激活门把适用范围从"DiT+LLM+ViT"一路收窄到**只剩 ViT**（§6.5 末表），而 #5 已按使用方"尽量不用 QuaRot"的指示关闭 ⇒ **当前无任何区域需要旋转**。若将来重启 ViT INT8 才需要，且按原则 #13 先微基准 |
| **7** | **权重指针字典缓存**（Phase 1 实测发现，原树里没有） | `_backbone_weight_dicts()` 每次调用都把融合 ViT qkv 权重/偏置重新切成 per-projection 连续张量：~120 次大 copy | 未预测 | ✅ **−11.46 ms**（实测 15.65 ms/77 ms backbone）；缓存在 `self._bb_weights`，中间张量由 `self._bb_keep` 保活 |
| **8** | **ViT 截断到 18 层**（Phase 1 实测发现） | DeepStack taps 是 `[5,11,17]`，LLM 只消费 merger 输出 ⇒ **block 18–23 的输出无人消费**，但 HF 照算 | 未预测 | ⚠️ **−7.77 ms，但被杠杆 #10 换掉了**：融合开启（默认）时 final merger 需要 block 23 的输出，`_vit_layers` 随之在 **24 / 18** 间切换。净账仍是赚的（+7.81 ms ViT 换掉 −58.07 ms HF 视觉塔），但**不能把两个杠杆的收益相加** |
| **9** | ViT RoPE / FFN gate·up 的 kernel 化 | 二者目前走 torch shim（§6.6.4 的 SM87 bf16 kernel 缺口），`_rope_rotate_half` 每次调用 ~7 个 torch 算子 | ~~2.7 ms~~ → **RoPE 实为 11.902 ms**（归因探针实测，§6.22.2；原值**偏低 4.4×**）；gate·up 的 `mul_` 实测只值 **0.092 ms** | ✅ **RoPE 已交付**（§6.22）：`blocked_on_kernel` 是**空判**——`rope_neox_qk_bf16` 早已编好，只是在**另一个扩展** `flash_rt_qwen3_vl_kernels`（`csrc/qwen3_vl_bindings.cpp`）里，只查 `flash_rt_kernels` 必然漏。与 shim **逐位相同**（3 个真实形状，两臂到 fp64 等距）⇒ 精度门数值一个没动。80 次 shim → **40 次 launch**；每观测 **−10.471 ms（1.1076×，仿真）** / −8.833（真机）；边界 **106.47 → 94.35 ms（3.38× → 3.82×）**；口径 C **3.56× → 3.97×**。测试 67 → 80。❌ **gate·up 的 `mul_` 否决**：`silu_mul_qwen36_bf16` 逐位相同但只省 0.092 ms（0.08%），低于噪声地板（原则 #13）。⚠️ 两条旧标注仍成立但需按新数读：§6.11.5 那次"5–10 ms 升值"的**理由**（CPU 提交受限）已撤回，可**结论的数值区间反而被归因探针证实了**（11.9 ms）——省的是 GPU 时间。融合仍故意用 `F.conv3d` 做 patch embed（§6.7.1(a)：展平 GEMM 与 HF 差 ~1 ULP，会被 24 层残差塔放大）；cuDNN 回落算法比 roofline 慢 ~23×（~1.25 ms GPU，§6.13.2） |
| **10** | **image→embeds 融合**（§6.7） | 摘掉 `aux["llm_input_embeds"]`，自算 patch embed → ViT 24 层 → final merger → 文本 embed gather → 按视觉位 scatter | 未预测（原以为是"纯工程补齐"） | ✅ **已交付**：真实独立部署成本 **174.86 → 124.56 ms = 1.40×**（省掉 HF 视觉塔 58.07 ms，代价 ViT +7.81 ms）。融合前两个口径分歧（同一边界 3.09× / 真实成本 2.06×），**融合后重合于 2.89×** |
| **11** | **backbone CUDA-graph 捕获** | 补齐 Orin 因继承链漏掉的能力（Thor/RTX-FP8 早就有，§6.12.2）。原依据是"CPU 提交 60.00 ms / 墙钟 65.77 ms（91.2%）" | 原判"数毫秒到十几毫秒"（**依据是错的量法**） | ⏳ **复活条件已满足，未落地**（§6.13.2 实测 + §6.15.6 改判）。**可行且逐位相同**（max&#124;d&#124; = 0），值 **4.54 ms（1.078×）**：CPU 提交 51.97 → **0.46 ms**，墙钟 62.47 → **57.93 ms** ⇒ launch 受限份额只有 **7.3%**，不是 91.2%。一次性成本 **262.1 ms** ⇒ **回本需 57.7 帧**；原判"净亏"的唯一理由是 `set_prompt` 一实例只跑一帧，而**杠杆 #13 已经把那个契约换掉了** ⇒ 现为 ROI 最高的未落地项，预期每观测 117.56 → ~113 ms |
| **12** | **常驻 backbone runtime + 纯 kernel forward**（做 #11 的前置时发现，独立成立） | `_run_kernel_backbone` 每次都重建 ~30 个 buffer、一个 `OrinGrootN17BackboneAttn`、5 个参数字典，并对 3 个 DeepStack inject 做**布尔掩码赋值**（内含 `nonzero()` ⇒ **3 次 host 同步**） | 未预测 | ✅ **已交付**（§6.13.1）：backbone **65.71 → 63.36 ms（−2.35）**，CPU 提交 **−7.2 ms**，边界 **107.40 → 105.26 ms（3.36× → 3.42×）**，融合余量 **4.21 → 2.16 ms**。精度与落地前**逐位一致**（四级 cos 与 §6.6.1/§6.7.2 记过的数字逐位对上），测试 **31 → 36 passed**。⚠️ 过程中踩到并修掉一个 31 个 GPU 测试都拦不住的静默 bug（unfused 把观测载进了 LLM 残差流），已用 5 个 CPU 契约测试钉住 |
| **13** | **每观测入口 `infer(aux=...)`**（连续推理；使用方指定的验收项） | 原契约是"一个前端 = 一个 prompt + 一帧"，`set_prompt` 第二次调用直接 raise。改为：校验被图烘住的元数据 → 重跑 backbone → **原地**刷新 DiT cross-KV → **复用**已捕获的四张 DiT 图 | 未预测（是正确性项，不是性能项） | ✅ **已交付**（§6.15）：每观测 **117.56 ms → 3.29×**（vs HF 完整 `get_action` 386.27）/ **3.05×**（vs 同一边界 358.84）。七道门全过，其中**连续 == 一次性逐位相同**、**捕获 1 次 / 复用 8 次**。顺带诊断出 cross-KV 刷新每帧重做 403 MB 的 `.float()` 提升（占该步 40%），缓存后 **13.10 → 8.52 ms 且逐位相同**。⚠️ 负控制测出一条方法论：**decoded action 的 cos 门不住 stranded KV**（打断刷新后仍 0.9999917 > 0.999），必须门在槽内容或 velocity 上（§6.15.3） |
| **14** | **每观测图像通路：HF processor → 纯 torch GPU 链**（§6.23） | 本仓 **7 个** N1.7 CUDA 前端**没有一个**跑图像通路，全都要求调用方交已处理好的 `aux` 束 ⇒ 厂商 `Gr00tN1d7Processor` 每观测跑一遍，且是**纯标量 host 代码**跑在弱 ARM CPU 上。换成：letterbox → **稠密 fp32 matmul 的连续覆盖 INTER_AREA** → 中心裁剪 → 放大 → rescale/normalize → Qwen2-VL merge-block patchify，全在设备上 | 计划里写 ~8.6 ms（由 ~10.3 ms 的账面估计推） | ✅ **已交付**：整链端到底端实测 **12.916–14.027 ms → 1.448–1.485 ms**，每观测 **112.374–115.499 → 99.579–100.428 ms（省 11.945–15.410 ms，1.1189–1.1540×）**，**预测偏低 1.4–1.8×**（那个账面数是分段中位数相加，见 §6.23.6 教训 3）。**新增口径 D**（每观测含图像预处理）：真机 **3.439–3.444× → 3.883–3.896×**、仿真 **3.344–3.437× → 3.846–3.859×**；⚠️ 同时**纠正口径 C 的边界不对称**（97.28 ms 是模型侧、却对着 HF 含预处理的 386.27 ms 比 ⇒ 3.97× 偏乐观）。精度：12 帧 × 3 臂，cv2 臂与 HF **九位小数全同**，T3/T4 **未放宽门**，只有 T1 新增 `THR_IMAGE_GPU_BACKBONE = 0.985`（出处：最坏实测帧再往下一整个观测跨度）。**缩小步逐位精确且可证**（奇数分子的无平局论证 + 3350 万次穷举 0 分歧）；**放大步 max = 1 LSB**，4 种表述全部收敛 ⇒ **声明未知**而不是编说法。测试 precision **17 → 29**、新增 `test_orin_groot_n17_preprocess.py` **65 条 CPU-only**、真机档 **102 → 179** |

### 6.3 INT8 实测结论（task #9，已完成）

`docs/hyvla05_orin_sm87.md:198` 曾实测 INT8 在 denoise M=41 下只有 6–8 TOPS、
权重流 71–95 GB/s，比 bf16 cuBLAS 更差。用 `cutlass_int8_rowwise_bf16out`（128×128）
与 `_t64x128` 两个 tile 在 6 个真实 DiT GEMM 形状上、M=41/51 做了 in-pipeline A/B
（锁频，best-of-5 × 300 iters；权重经 `hyvla_orin._quantize_per_row_int8` [PRE] 量化，
激化为 `quantize_int8_rowwise` [PRE] 动态 per-row）：

**单独 µbench（每层 6 个 GEMM 之和，M=41）**

| | bf16 | INT8 GEMM-only | INT8 + 独立量化 | INT8 t64×128 |
|---|---|---|---|---|
| 每层合计 | 0.4537 ms | 0.4243 ms | **0.7778 ms** | 0.7766 ms |
| 相对 bf16 | 1.00× | 0.94× | **0.58×（更差）** | 0.58× |

**全 32 层 DiT 单步（1082.1M 参数，M=41）**

| 档位 | eager | CUDA graph | graph 权重流 | ×4 步 |
|---|---|---|---|---|
| bf16 | 15.60 ms | **14.60 ms** | 148.2 GB/s（94% of 158 上限） | 58.40 ms |
| INT8 | **33.39 ms** | **9.37 ms** | 115.5 GB/s（73% of 上限） | **37.48 ms** |

M=51（N1.6）结论相同：bf16 graph 14.82 ms / INT8 graph 9.57 ms，1.55×。
逐 GEMM 输出余弦 **0.99992–0.99993**。

**判定被反转了两次，这正是原则 #16 的 µbench 陷阱：**

| 视角 | INT8 vs bf16 |
|---|---|
| 单独 GEMM（含独立量化） | 0.58× — **INT8 更差** |
| 全链路 eager | 0.47× — **INT8 差得多** |
| 全链路 + CUDA graph | **1.56× — INT8 胜** |

机制：`quantize_int8_rowwise` 单独调用固定花 ~59 µs、INT8 GEMM 单独调用固定 ~70 µs，
**与权重大小几乎无关**（3072×1536 到 6144×1536 都是 ~0.070 ms）⇒ 单独测时完全是
launch/latency-bound，掩盖了真实带宽收益。图捕获把这些固定开销全部消掉：
INT8 从 eager 33.39 → graph 9.37 ms（**3.6×**），而 bf16 只有 15.60 → 14.60 ms（1.07×）。

**两条硬结论：**

1. **CUDA Graph 是 INT8 的前置条件，不是优化项。** 无图时 INT8 比 bf16 eager 慢 **2.1×**
   （33.39 vs 15.60 ms）。任何"先跑通 INT8 再加图"的顺序都会得到 INT8 无用的错误结论。
2. **INT8 只给 1.56×，不是 2×。** 因为 INT8 权重流只有 115.5 GB/s，而 bf16 有 148.2 GB/s
   （印证 hyvla 的 71–95 GB/s 观测）。字节减半的收益被更低的带宽效率吃掉一部分。
   → DiT 4 步：58.4 → 37.5 ms，**净省 20.9 ms**。

**剩余空间与缺口：**

- INT8 kernel 只到带宽上限的 73%（bf16 是 94%）⇒ **tile/流水线仍有调优空间**；
  若达到 148 GB/s，DiT 单步可到 ~7.3 ms、4 步 ~29 ms。t64×128 tile 实测无改善
  （0.1295 vs 0.1293 ms），不是答案。
- 独立量化 pass 占 INT8 单步的很大一部分。`gate_residual_ada_norm_int8` [PRE]
  可让 ada_norm 直接输出 int8，把量化融进 norm，省掉独立 pass ⇒ 应优先接上。
- **INT4 缺口**：仓库只有 `qwen3_vl_int4_gemv_m1` / `_w4_gemv_m1`（M=1 decode 专用），
  **没有 M=41 的 INT4 GEMM**。DiT 再减半字节需要新 kernel；按 AGENTS.md 红线 #8，
  这是 kernel owner 的交付物，本适配只记录缺口、不用 eager 链 emulation。

### 6.4 预期设定与实测对照（原则 #15：每个门都要报 measured vs predicted）

Thor N1.6 是 28.5 ms，靠 FP4：DiT 每步只读 ~415 MB，而 Orin bf16 每步读 2164 MB
—— **5.2× 字节差**，外加 Thor 带宽 252 GB/s vs Orin 实测 ~158 GB/s。
SM87 无 FP8/FP4，**Orin 不可能追平 Thor**。

以实测 eager 基线 387.93 ms（§5.4）为起点，边界 = backbone + action head：

| 档位 | 预测 | **实测** | 偏差 | 依据 |
|---|---|---|---|---|
| HF eager 完整 `get_action` | — | **387.93 ms** | — | §5.4 |
| HF eager 同一边界（×0.929） | — | **360.38 ms** | — | §6.6 插桩 |
| **BF16 + graph + elementwise 融合（Phase 1 交付，image→embeds 融合前）** | ~124 ms | **116.18 ms** | **−6.3%（预测偏保守）** | §6.1 roofline 求和 vs §6.6 |
| **+ image→embeds 融合（Phase 1.5，当前）** | ~132 ms（= 上面 ~124 **+ 融合多跑的 6 层 ViT +7.81**，杠杆 #8 被 #10 换掉） | **124.56 ms** | **−5.5%** | §6.7；口径 A/B 在此重合 |
| **+ INT8 DiT（Phase 2，已交付）** | ~103 ms | **107.8–108.8 ms** | **+4.7–5.6%** | §6.9；实测 DiT 1.411× 而非 µbench 的 1.56×，差在"不被加速的部分"（§6.9.4） |
| + INT8 LLM-FFN（动态 per-row，杠杆 #4） | ~90–95 ms（照搬 DiT 比例的旧推法） | ❌ **未交付，已回退**（边界仍 107.8–108.8 ms） | **预测偏大 ~4×**：实测 eager 只省 **0.6–1.0 ms**，不是 13–15 ms | §6.11：GPU 层面确实省 **3.93 ms**（捕获后 **4.77 ms**），但 INT8 臂自己多带 **~3.9 ms 藏不住的发射成本**（§6.13.3 反推），eager 墙钟只拿到 1/5；**且精度不过门**（`backbone_features` **0.992077 < 0.995**，而 G4 是 0.329° 看着无害）⇒ 两条独立否决 |
| + 常驻 backbone runtime（杠杆 #12，做 #11 前置时发现） | 未预测 | ✅ **已交付：104.79–105.26 ms（3.42–3.44×）** | 省 **2.35 ms** | hoist ~30 个分配 + attn backend 构造 + 5 个参数字典，并去掉 3 次布尔掩码赋值内含的 **host 同步**；精度逐位不变（§6.13.1） |
| + backbone CUDA-graph 捕获（杠杆 #11） | **未预测；原判"数毫秒到十几毫秒"** | ❌ **实测后不落地：只值 4.54 ms（1.078×）** | **预测偏大 ~3×，且依据的量法本身是错的** | CPU 提交 51.97 → 0.46 ms（−99%）而墙钟只 62.47 → 57.93 ⇒ launch 受限份额 **7.3%**，不是 91.2%。一次性成本 262.1 ms ⇒ 回本需 **57.7 帧**，现契约一实例一帧（§6.13.2/§6.13.3）。➡️ **后续改判并落地**：每观测入口（§6.15）推翻了"一实例一帧"这个唯一前提，§6.17 实测部署口径 **−4.00 ms**、回本 **64.9 帧**、逐位相同 |
| + INT8 全 LLM 塔（含 QKV/O，动态 per-row） | 未预测 | ❌ **关闭** | — | §6.10.1 的 G4 过门**不代表可做**：§6.11.1 实测 **k/v 在 INT8 下更慢（0.851×，N=1024 时 quantize 的固定开销吃掉 GEMM 收益）**，且 FFN-only 已因 `backbone_features` 0.992077 被拦 ⇒ 全塔是更差的版本 |
| + INT8 ViT（杠杆 #5） | — | ❌ **关闭** | — | §6.10.2：tap `vit_block_17` per-row **0.971083** vs 门 0.998，per-tensor **0.5127** ⇒ 动态量化救不了；按"尽量不用 QuaRot"保持 bf16 |
| 厂商 Orin 最好成绩（**1 相机**） | 216.5 ms | — | 口径不同 | §3 |

**预测有效性判定**：roofline 求和 124 ms vs 实测 116.18 ms，偏差 6.3%，
**瓶颈模型是对的**（不是"数量级对但机理错"）。但分项必须分开看，否则会错在同一处：

| 分项 | 预测 | 实测 | 差异机理 |
|---|---|---|---|
| DiT ×4 步 | ~79 ms（GEMM 57.3 + 非 GEMM ~21.6 推算） | **58.38 ms** | 非 GEMM 那 21.6 ms 高估了：CUDA graph 把 attention/norm/elementwise 的 launch 开销几乎全消掉，实测非 GEMM 只剩 ~1 ms |
| backbone | ~36 ms（ViT 17 + DS 4 + LLM 12 + VLSA 3） | **57.79 ms** | 低估了：ViT 的 RoPE 与 FFN gate·up 相乘走 torch shim 而非 kernel（§6.6.4），且 HF 侧 ViT 自己也 fallback 掉 FA2 |

两边部分抵消，所以总数看着准。**记：roofline 求和能定"值不值得做"，
但分项误差会互相掩盖，下一次预测必须分项校准。**

即相对厂商 **~1.87×**（216.5/116.18），但**厂商是 1 相机、本文是 2 相机，此比值不可用**；
同口径需按 1 相机复测本文通路才能给可比数字（§3.1）。

### 6.5 激活离群值剖面（真机数据实测）与精度档判定

> ⚠️ **本节整表已重测并作废旧表。** 旧表各行用了**不一致的度量**，其"离群通道"列
> （48/933、62、976、899/425、**1793**、**1999**、698→1383）在所声明的定义下
> **不可能成立** —— 若 1793/2048 个通道都 >10× 中位数，中位数本身就不可能是那个值。
> 该列不可信，已被下表替换。旧表的 `max|a|` 与 `rms` 两列与复测一致，仍然有效。

**统一度量（本表唯一口径，可复现）**：对激活 `A` 形状 `(tokens, C)`，
`p_c = max_tokens |A[:,c]|`；**逐通道比 = `max_c p_c / median_c p_c`**；
**离群通道数 = `#{c : p_c > 10 × median_c p_c}`**。
per-row/per-tensor 缩放能吸收整体幅度漂移，但吸收不了**逐通道**离群，
所以逐通道比才是选档的依据。数据源：§5.4 的 `state_unit=deg2rad` fixture（frame 0）。

| 阶段 | max&#124;a&#124; | rms | 逐通道比 | 离群通道 | INT8 可行性 |
|---|---|---|---|---|---|
| vit_block_0–9 | 9.5–22.4 | 0.36–0.78 | 11.3–15.7× | 2–8 / 1024 | 边缘 |
| **vit_block_10–23** | 210 → 2720 | 0.55–40.5 | **120.8–204.8×** | 4–8 / 1024 | ❌ 拒绝 |
| deepstack_merger_0–2 | 8.2–19.5 | 0.34–0.54 | **8.3–14.9×** | 0–1 / 2048 | ✅ 可（但 ROI 为零，见判定 4） |
| llm_layer_0–1 | 29.4 / 61.0 | 1.19 / 1.62 | 10.1 / 16.8× | 1–2 / 2048 | 边缘 |
| **llm_layer_2–15** | **15296.0**（恒定） | 33.7–33.9 | **2146.8–3546.9×** | 16–26 / 2048 | ❌ 拒绝 |
| vlsa_block_0–3 | 89–94.5 | 0.80–1.43 | 27.5–55.2× | 2 / 2048 | 边缘 |
| **dit_block_0–31** | 77.0 → 1152 | 3.18 → 42.8 | **18.4–142.3×**（中位 92×） | 9 → 41 / 1536 | ⚠️ 需真激活门 |

逐 block 明细（DiT，全 32 层 min 18.4× / median 92.0× / max 142.3×）：

| block | max&#124;a&#124; | rms | 逐通道比 | 离群通道 |
|---|---|---|---|---|
| dit_block_0 | 77.0 | 3.18 | 18.4× | 9 / 1536 |
| dit_block_1 | 105.0 | 3.46 | 23.5× | 11 / 1536 |
| dit_block_8 | 488.0 | 9.38 | 85.3× | 20 / 1536 |
| dit_block_15 | 612.0 | 9.52 | 91.9× | 22 / 1536 |
| dit_block_23 | 808.0 | 14.27 | 101.8× | 33 / 1536 |
| dit_block_31 | 1152.0 | 42.76 | 125.4× | 41 / 1536 |

**DiT 行为何必须复测（§5.5 的下游）**：旧 DiT 行测自 state 偏大 57× 的 fixture。
state 只经 state encoder 成为 DiT 的 token 0，**backbone 各行与 state 无关**，
所以只有 DiT 行受单位错误影响。复测后**两个指标朝相反方向动了**：

- 逐通道**比值变差**（旧记 52× → 实测 142×）：per-row 缩放的损害来自
  "某一行里含一个极端通道"，比值越大该行其余通道损失的有效位越多。**对 INT8 不利。**
- 离群**通道数极少**（41 / 1536 = 2.7%）：尾部集中且稀疏。**对 INT8/旋转有利。**

**判定（数据驱动，非照搬 hyvla）：**

1. **DiT INT8 W8A8 动态 per-row —— 已由 §6.8 的实测门改回 `✅`（过门）。**
   ~~⚠️ 由 `✅` 下调为"需先过真激活门"。~~ **下调已撤回**：本节要求的实验做完了，
   结果是 per-row W8A8 在两帧真机数据上 **cos ≥0.999999、最坏 0.255°**，
   比本通路 bf16 融合档已有的 0.49° 还小（§6.8）。
   原始推理保留如下 —— §6.3 的"逐 GEMM cos 0.99992–0.99993"来自 µbench，
   **其激活不是本 fixture 的真实 DiT 激活**；真实逐通道比比当初假设差 2.7×，
   因此按原则 #11/#12 必须先做真激活门再谈接入。**这个流程要求是对的，
   只是它对结果的预测偏悲观**：142× 的逐通道离群**没有**击穿 per-row 缩放。
   旁证也一致：NPU 通路 per-row INT8 DiT 免标定 e2e cos 0.9999571
   （`docs/deployment_npu_groot_n17.md:174-181`）。
   ⚠️ 但 per-row **是硬要求**：per-tensor 在 frame 300 差 2.5×（§6.8 判定 2）。
2. ~~**LLM 保持 BF16** ❌ INT8~~ —— **本判定已被 §6.10.1 的真激活门推翻，改为 ✅ 可 INT8。**
   原始推理保留：通道 1793 的 15296.0 对 rms 33.7 是 **453×**，逐通道比
   **2147–3547×**（比旧表记的 382–612× 严重得多）；per-tensor 会把 scale 钉死在
   离群值上，主体分布只剩不到 1 bit；**"per-row 也救不了，因为离群是逐通道的，
   每行都被同一个通道主导"**。
   ⚠️ **最后一句是错的**，而且错法很典型：前半句（每行被同一通道主导）成立，
   但**结论不跟着成立** —— 决定成败的是量化后**非离群通道的相对误差**，
   不是离群比本身。2147–3547× 描述的是**分布**，不是**误差**。
   实测：per-row 动态量化**全 112 个 Linear**（含 QKV/O）G4 cos **0.999999**、
   最坏 **0.573°**；FFN-only 更好（**0.247°**）。而 per-tensor 差 **2.9–9.0×**
   ⇒ 这确实是 **scale 问题**（原则 #12），**不需要 QuaRot**。
   旁证也在本节自己的记录里：该离群通道在 bf16 下**已冻结**（ULP=64），
   后 14 层几乎不携带信息 ⇒ 粗量化它损失不了什么。
3. **ViT 保持 BF16** ❌ INT8 —— **§6.10.2 的真激活门确认成立，杠杆 #5 关闭。**
   逐通道比 **120.8–204.8×**（block 10 起跳，与 `max|a|` 从 9.5 跳到 210 同一位置）；
   与 hyvla ViT INT8 失败（cos 0.997459）同源。
   实测 per-row 动态量化把**被消费的 tap** `vit_block_17` 打到 **0.971083**
   （门是 0.998），per-tensor 崩到 **0.5127**。
   ⚠️ 但**只看 G4 会误判**：同一档的 G4 是 cos 0.999999 / 0.740°，看着完全无害
   —— 逐级门与原则 #2 正是为此存在（§6.10.2）。
4. **DeepStack mergers 是全模型最干净的区域**（8.3–14.9×，0–1 个离群通道），
   但它们只占 1.38 ms（§6.6.3）⇒ **INT8 化 ROI 为零，不做**（原则 #13）。
5. **VLSA 暂保 BF16**（27.5–55.2×，边缘；只 2 个离群通道，可试 INT8 但必须单独验门）。

> **一个必须记下的 bf16 现象**：`llm_layer_2..15` 的 max&#124;a&#124; **完全相同（15296.0）**，
> 复测 16 层逐层为 `29.4, 61.0, 15296.0 ×14`。
> 不是巧合——bf16 只有 7 位尾数，在 15296 ≈ 1.867×2¹³ 处 **ULP = 2¹³⁻⁷ = 64**
> （15296/64 = 239，整除）。任何 <32 的残差更新都会被舍入掉，
> 所以该通道在 bf16 下**已经冻结**。推论：该离群通道在后 14 层几乎不携带信息，
> 但它仍然支配 INT8 的 scale —— 这正是"旋转（改条件数）而非降位宽（改噪声）"
> 适用的场景（原则 #17）。
> **实测印证**：FlashRT 的 `llm_h` 对该层参考的 max&#124;d&#124; 恰为 **128 = 2 ULP**，
> 而 cos 仍是 0.999998（§6.6.1）—— 误差几乎全落在那个已冻结的通道上。

~~**因此杠杆 #6（QuaRot/Hadamard 旋转）不是备选，而是拿到 backbone INT8 收益的唯一路径。**~~
**三次真激活门把这句话一路收窄到几乎不剩什么：**

| 区域 | 逐通道离群比 | 原判定 | **真激活门实测** | 需要旋转？ |
|---|---|---|---|---|
| DiT | 142× | ⚠️ 需先过门 | ✅ **过门**（cos ≥0.999999，§6.8） | **不需要** |
| LLM | 2147–3547× | ❌ per-row 救不了 | ✅ **过门**（全 112 Lin，最坏 0.573°，§6.10.1） | **不需要** |
| ViT | 120–205× | ❌ | ❌ **不过门**（tap 0.971 vs 门 0.998，§6.10.2） | 只有这里可能需要 |

⇒ **杠杆 #6 的适用范围只剩 ViT**，而使用方指示"尽量不用 QuaRot" ⇒
**当前结论是 ViT 保持 bf16，杠杆 #6 挂起**（若将来确实要吃 ViT 那 41.2% 的
backbone 占比，再按原则 #13 先微基准；旋转要融进 norm kernel（radix-16
寄存器 FHT）才近乎免费。Chameleon/Orin 的先例是 W8A8+rotation 同时压倒
plain W8A8 与 W4A16）。

> **这一串的核心教训（比任何单个数字都值钱）**：
> **"离群比大 ⇒ per-row 救不了 ⇒ 需要旋转"是一条看起来严谨、实则未经测量的推理链，
> 三个区域里错了两个。** 离群比描述的是**激活分布**，而门的判据是
> **量化后的输出误差**；两者之间隔着"离群通道是否还携带信息"
> （LLM 的那个已被 bf16 冻结）与"该区域还剩多少精度余量"
> （ViT 的 bf16 tap 已经只有 0.998156）。
> ⇒ 按原则 #11/#12：**精度档判定必须先做真激活门，再谈机制解释。**
> 机制解释是用来**理解**实测结果的，不是用来**代替**它的。

### 6.6 Phase 1 实测：精度门、延迟分解、已采纳的杠杆、kernel 缺口

> ⚠️ **本节是融合前的 Phase 1 快照，保留作历史记录。**
> 其中的延迟与倍率（116.18 ms / 3.10× / 21 passed）已被 **§6.7 取代**
> （当前 124.56 ms @ 边界 / 2.89× / 23 passed），门的**结构**仍成立但阈值变了
> （`backbone_features` 从 ≥0.999 改为 ≥`THR_FUSED_CONSUMED`=0.995，四条理由见 §6.7.2）。
> 引用当前状态请读 §0.1。

数据：§5.4 的 `state_unit=deg2rad` fixture（frame 0 与 300，真机 SO101 帧），
锁频 1300.5 MHz，`tests/test_orin_groot_n17_precision.py` **21 passed**
（含 `test_orin_groot_n17_dispatch.py` 的 11 项无 GPU 契约测试）。

#### 6.6.1 精度门 G1–G5（cos 均在 **float64** 下计算）

> 为什么必须 float64：LLM 残差流带一个 **15296** 量级的通道（§6.5），
> 在 2.9e5 个元素上用 fp32 累加两个范数会返回 **cos = 1.000167 > 1**（不可能值）。
> 换 float64 后真值是 0.999998。**任何在该张量上用 fp32 算 cos 的门都会误判。**

| 门 | 张量 | frame 0 | frame 300 | 阈值 |
|---|---|---|---|---|
| **G1** backbone 逐级 | `vit_h` vs `vit_block_17`（**tap，诊断用**） | 0.998156（max&#124;d&#124; 17.89） | 0.999503（max&#124;d&#124; 8） | 0.998 |
| | `deepstack_out_0/1/2`（**被 LLM 消费**） | 0.999960 / 0.999687 / 0.999224 | 0.999956 / 0.999787 / 0.999642 | 0.999 |
| | `llm_h` vs `llm_layer_15` | **0.999998**（max&#124;d&#124; 128 = 2 ULP） | 0.999998 | 0.999 |
| | `vlsa_h` vs `vlsa_block_3`（= backbone_features） | 0.999729 | 0.999757 | 0.999 |
| **G2** 每步 DiT 输入 | `dit_step_input[0..3]` | 0.999993 / 0.999993 / 0.999992 / 0.999992 | 0.999992 ×4 | 0.999 |
| **G2b** 每步速度 | `velocity[0..3]` | 0.999855 / 0.999944 / 0.999945 / 0.999899 | 0.999902 / 0.999950 / 0.999947 / 0.999902 | 0.999 |
| **G3** 归一化 action | `final_actions_norm` (1,40,132) | **0.999999**（max&#124;d&#124; 0.0156） | 0.999999（max&#124;d&#124; 0.0313） | 0.999 |
| **G4** 解码后 action（真 E2E） | (1,16,6) 弧度 | **1.000000**（max&#124;d&#124; **0.004307 rad** = 0.25°） | 1.000000（max&#124;d&#124; 0.00326 rad） | 0.999 |
| **G5** 图安全 | graph vs eager | **max&#124;d&#124; = 0（bit-identical）** | 同 | 恒等 |
| | replay vs replay | **max&#124;d&#124; = 0** | 同 | 恒等 |
| | 换输入后 replay（stale-value） | ✅ 正确跟随 `state_norm + 0.25` | — | 必须变 |

**G1 的 `vit_h` 用 0.998 而非 0.999，理由必须写清（否则就是放宽门）**：
`vit_block_17` 是 **DeepStack 的最后一个 tap，不是被 LLM 直接消费的张量**；
真正进 LLM 的是三个 merger 的输出（0.9992–0.99996，全部过 0.999）。
逐层曲线（§6.6.5）显示这是 bf16 平滑累积而非某一层出错，
且 HF 自己的 ViT 因一个子模块留在 fp32 而 **fallback 掉 FA2**
（日志：`current dtype in Qwen3VLVisionModel is torch.float32`）⇒ 参考侧本身也不是全 bf16。
**门是加在被消费张量上的，tap 只作深度诊断** —— 这是 §0.2 里"被消费张量全部 ≥0.999"的含义。

#### 6.6.2 抓到的两个真实 math bug（都不是精度档问题，是算错了）

| # | 症状 | 根因 | 修法 | 效果 |
|---|---|---|---|---|
| 1 | `vit_h` cos 只有 0.9917 | 自写的 torch RoPE shim **在 bf16 下算**，而 HF 的 `apply_rotary_pos_emb_vision` 与 FlashRT 的 `rope_rotate_half_fp16` **都升到 fp32 再算再降回** | shim 改 fp32 数学 / bf16 存储 | `vit_h` cos 0.9917 → **0.9939**（24 层口径；余下是 bf16 地板，见 §6.6.5） |
| 2 | merger cos 0.99994 / 0.99894，且误差会传进 LLM 的 image token | `deepstack_merge_forward` 用了 `gelu_inplace`（**tanh 近似**），但 `Qwen3VLVisionPatchMerger.act_fn = nn.GELU()` 是 **exact erf、硬编码**，且**不读** `vision_config.hidden_act`（那个 `"gelu_pytorch_tanh"` 只作用于 vision blocks） | merger 改用 `gelu_erf_bf16` | 0.99994→**0.999960**、0.99894→**0.999224** |

> 🔴 **bug #2 在既有 RTX FP16 通路里同样存在，且尚未修**：
> `flash_rt/models/groot_n17/pipeline_rtx_fp16.py` 的 `deepstack_merge_forward`（L211–276）
> **两条分支都用 tanh 近似** —— FP8 分支 `:222`（`fp8_nn_gelu_bias`，该行还留着作者自己的
> `(tanh-approx; verify)` 注释）与非 FP8 分支 `:271`（`gelu_inplace_fp16`）。
> SM87 有 `gelu_erf_bf16` 可用（本通路在 `pipeline_orin.py:231` 用它），
> **FP16 侧没有 `gelu_erf_fp16`**（已 `hasattr` 核实为 False），
> 所以修它需要先有 kernel（AGENTS.md 红线 #8：只记录缺口，不代 kernel owner 交付）。
> 按红线 #1（additive only）本次**未改动**该文件，只在此登记。
> 影响量级：每个 merger ~1e-3 cos，经 3 个 merger 复合进 LLM 的 128 个 image token。
>
> ⚠️ **不要顺手"一起修"**：同文件 `:199`（`qwen3vl_vit_forward`）、`:560`
> （`vl_self_attn_forward`）、`:704`（`dit_forward`）用 tanh 是**正确的** ——
> vision blocks 读 `vision_config.hidden_act = "gelu_pytorch_tanh"`，
> VLSA 与 DiT 的 `activation_fn="gelu-approximate"` 也是 tanh。
> **只有 merger 一处是 exact erf**，因为 `Qwen3VLVisionPatchMerger` 把它硬编码了。

**另一处必须记下的核实结论**：FlashRT 的 LLM 通路**不施加** `_llm_norm_w`（final norm），
这不是遗漏。transformers 4.57 的 `_can_record_outputs =
{"hidden_states": Qwen3VLTextDecoderLayer}` ⇒ `outputs.hidden_states[-1]` 是
**最后一个 decoder layer 的输出，即 pre-final-norm**。实测 `llm_h` cos 0.999998 印证。

#### 6.6.3 延迟分解与已采纳的杠杆

**HF eager 侧（插桩，每级一个 sync；absolute 抬高约 3%，只用比值）**

| 阶段 | 时间 | 占比 |
|---|---|---|
| `backbone.forward` | 138.79 ms | 34.8% |
| `action_head.get_action_with_features` | 231.97 ms | 58.1% |
| pre/post（processor、normalize、denormalize） | 28.24 ms | 7.1% |
| 插桩总 | 399.01 ms | → 边界占 **0.929** |
| 未插桩总（fixture meta） | **387.93 ms** | |

**FlashRT Orin 侧（同一边界，锁频，median of 10）**

| | 时间 | 相对 HF |
|---|---|---|
| backbone | **57.79–58.49 ms** | 2.37–2.40× |
| DiT ×4（**graph**） | **58.38 ms** | 3.97× |
| DiT ×4（eager，对照） | 115.01 ms | 2.01× |
| **边界合计** | **116.18 ms** | **3.10×** |
| pre/post（**未替换**，仍走 HF processor） | ~28 ms | — |
| 完整 `get_action` 等价 | ~145 ms | ~2.67× |

backbone 内部（`torch.cuda.Event` 逐阶段，wall 58.49 ms）：

| 阶段 | 时间 | 占 wall |
|---|---|---|
| `qwen3vl_vit_forward`（18 层） | **24.10 ms** | 41.2% |
| `qwen3vl_llm_forward`（16 层） | **22.46 ms** | 38.4% |
| `vl_self_attn_forward`（4 层） | 6.45 ms | 11.0% |
| `deepstack_merge_forward`（×3） | 1.38 ms | 2.4% |
| `vlln_forward` | 0.07 ms | 0.1% |
| 阶段间 gap | 4.03 ms | 6.9% |

⇒ **ViT 与 LLM 各占约 40%，是下一步唯一值得动的两块**；
DeepStack/vlln 合计 1.45 ms，任何在其上的优化 ROI 为零（原则 #13）。

**Phase 1 期间实际采纳的两个杠杆（都不在 §6.2 的原始预测树里）**

| 杠杆 | 机制 | 实测 | 精度影响 |
|---|---|---|---|
| **权重指针字典缓存** | `_backbone_weight_dicts()` 原先每次调用都把融合 ViT qkv 权重/偏置重切成 per-projection 连续张量（~120 次大 copy）。改为构建一次、缓存在 `self._bb_weights`，中间张量由 `self._bb_keep` 保活 | **77.29 → 58.68 ms 中的 −11.46 ms**（单独量到 15.65 ms/call） | 全部门数值**不变** |
| **ViT 截断到 18 层** | taps `[5,11,17]` ⇒ 只有 block 0–17 的输出被消费；HF 照算 24 层。设 `_vit_layers = max(_DEEPSTACK_TAPS)+1 = 18`，`layers_subset=range(18)`，shadow 权重也只加载 18 层 | **−7.77 ms** | 全部门数值**不变**（被截掉的 6 层输出本就无人读） |

两者合计把 backbone 从 77.29 降到 58.68 ms（**1.32×**），且**精度门数值一字未动** ——
这是"纯开销、非近似"的证据（原则 #5：不能只看输出相同，还要看它改的不是数值路径）。

⚠️ `deepstack_inject` 是 **per-prompt** 的（依赖 `visual_pos_masks`），
所以 `lw_run = dict(lw)` 每次浅拷贝后再写入，**缓存里绝不保存 prompt 相关的指针**。

#### 6.6.4 SM87 bf16 kernel 缺口（本次实测确认，全部有 workaround，未新写 kernel）

先记一条容易踩的契约：**SM87 上所有"generic"命名的 kernel 都是 bf16**
（`rms_norm`、`layer_norm`、`gelu_inplace`[tanh]、`residual_add`、`qkv_split`、
`bias_residual`）；`quantize_int8_rowwise` 与 `cutlass_int8_rowwise_bf16out` 也是 bf16-only。
`quantize_int8_rowwise_fp16` 在 `csrc/bindings.cpp:8919` **有声明但未为 SM87 编译**。
FA2 侧可用：`fwd_bf16{,_causal,_seqused,_seqused_splitkv,_tile,_window}`、`fwd_fp16`；
其中 **`fwd_bf16_causal` 分别收 `num_heads_q` 与 `num_heads_kv` ⇒ 原生 GQA**。

| 缺口 | 影响点 | 本次 workaround | 代价 |
|---|---|---|---|
| 无 bf16 `rope_rotate_half` | ViT 每层 + LLM 每层 | torch shim，**fp32 数学 / bf16 存储**（必须 fp32，见 §6.6.2 bug #1） | ViT/LLM 各若干 ms；是 §6.4 backbone 预测偏低的直接原因 |
| 无 bf16 `gpu_repeat_interleave_heads` | LLM 的 GQA 16Q/8KV | **不需要了**：FA2 `fwd_bf16_causal` 原生 GQA | 反而省掉 head 扩展与其 `gpu_fill_neginf_fp16` logits slab |
| 无 bf16 `attention_mha_causal` | LLM self-attn | FA2 `fwd_bf16_causal` | 无 |
| 无 bf16 elementwise `mul` | LLM FFN 的 `silu(gate) * up` | `fvk.silu_bf16` 后接 torch `gate_t.mul_(up_t)` | 一次额外 kernel launch（已被 CUDA graph 吸收） |
| `bf16_nn_bias*` 在 M=41 报 `CUBLAS_STATUS_NOT_SUPPORTED` | DiT 全部 GEMM（M=41 非 16 对齐） | `gemm.bf16_nn` + 独立 `add_bias_bf16` | 每 GEMM 多一次 launch（已被 graph 吸收） |

`flash_rt/hardware/rtx/attn_backend_groot_n17_orin.py` 里
**主动 `del self._llm_logits, self._llm_ctx`**：父类那套 FP16 cuBLAS-MHA 脚手架在本通路
永不使用，删掉它才能让"误落回该路径"变成**响亮失败**而不是把 bf16 当 fp16 读（红线 #4）。

#### 6.6.5 ViT 逐层累积（为什么 tap 门是 0.998，且为什么不是某一层出错）

frame 0，FlashRT vs HF 参考，逐层 `rel_l2 = ||d||₂ / ||ref||₂`：

| layer | cos | 1−cos | max&#124;d&#124; | ref rms | ref max&#124;a&#124; | rel_l2 | 层间倍增 |
|---|---|---|---|---|---|---|---|
| 0 | 0.999994 | 5.83e-06 | 0.19 | 0.78 | 22.4 | **0.003415** | — |
| 1 | 0.999965 | 3.55e-05 | 0.20 | 0.48 | 10.9 | 0.008421 | ×6.08 |
| 2 | 0.999934 | 6.64e-05 | 0.20 | 0.42 | 12.2 | 0.01152 | ×1.87 |
| 3 | 0.999906 | 9.36e-05 | 0.38 | 0.40 | 15.5 | 0.01368 | ×1.41 |
| 4–8 | 0.999899→0.999880 | ~1.1e-04 | 0.25–0.44 | 0.37–0.41 | 9.8–16.4 | 0.0142–0.0155 | ×0.98–1.10 |
| 9 | 0.999818 | 1.82e-04 | 0.71 | 0.36 | 9.5 | 0.01905 | ×1.51 |
| **10** | 0.999518 | 4.82e-04 | 10 | 0.55 | **210** | 0.03349 | ×2.65 |
| 11 | 0.999618 | 3.82e-04 | 12 | 0.74 | 310 | 0.02997 | ×0.79 |
| 12–14 | 0.999347→0.998966 | 6.5e-04→1.0e-03 | 11–12 | 0.77–0.92 | 312–328 | 0.0378–0.0470 | ×0.99–1.71 |
| 15 | 0.998700 | 1.30e-03 | 17.6 | 1.02 | 346 | 0.05149 | ×1.26 |
| 16 | 0.998431 | 1.57e-03 | 18.4 | 1.07 | 346 | 0.05644 | ×1.21 |
| **17（tap）** | **0.998156** | 1.84e-03 | 17.9 | 1.17 | 346 | **0.0611** | ×1.18 |

读法：
- **layer 0 的 rel_l2 = 0.003415 ≈ bf16 eps（2⁻⁸ = 0.0039）** ⇒ 起点就是 1 ULP，
  不是算法错。
- **无跳变点**：最大层间倍增只有 2.65×（layer 10），且该层正是 `max|a|` 从 9.5
  跳到 210 的位置（§6.5 里 ViT 离群比同时从 15× 跳到 195×）⇒ 是**幅度变大导致
  绝对误差变大**，不是该层实现有误。
- 曲线平滑单调 ⇒ 判定为 **bf16 累积**，不是某一层的 bug。这也解释了为什么
  修完 §6.6.2 的两个 bug 后 cos 从 0.9917 只升到 0.9939（24 层口径）——
  剩下的部分是 dtype 本身的地板。
- layer 18–23 不再计算（杠杆 #8）。在旧的 24 层口径下 layer 23 是 cos 0.9939 /
  rel_l2 0.111，同样无跳变点。

#### 6.6.6 边界诚实声明（哪些没被替换）

1. ~~**`aux["llm_input_embeds"]` 仍来自一次 HF forward。**~~
   **已于 §6.7 解决**：融合默认开启，该依赖已摘掉。历史记录保留如下 ——
   FlashRT 的 backbone 原先从"已经拼好的 LLM 输入 embeds"起跑，**没有替换
   image→embeds 的融合**（patch embed、merger、`embed_tokens` gather、image token
   scatter）。这与既有 Thor / RTX N1.7 通路是同一个边界，所以当时的倍率是
   "同边界可比"的，但**不是"完整替换 HF"**。
   ⚠️ 后来量测发现这条被**低估**了：HF 为产出该张量必须跑完整视觉塔
   **58.07 ms**，而 FlashRT 又重跑了 ViT 前 18 层去取 DeepStack taps —— 也就是
   ViT 被算了两遍。真实独立成本是 174.86 ms 而非 116.79 ms（§6.7）。
   **仍未摘掉**：`rope_cos` / `rope_sin`（M-RoPE 表）、`visual_pos_masks`
   （等价于 `input_ids == image_token_index`）、以及 `pixel_values` 本身
   （HF processor 的 resize/normalize/im2col 产物）。
2. **pre/post 的 ~28 ms 未替换**（HF processor、normalize/denormalize）。
   完整 `get_action` 等价约 **152 ms**（推算：124.56 边界 + 27.55 pre/post），
   **不是 124.56 ms**；报数时必须区分这两个口径。
3. **权重内存有冗余**：`weight_spec_orin` 仍加载融合的 `_llm_qkv_w` / `_vit_qkv_w`
   （~420 MB），而 Orin frontend 走 shadow 里的**分离**副本（另 ~420 MB）。
   不影响正确性与延迟（一次加载），但部署内存翻倍，属待清理项。
4. **推理超参一个都没改**：`action_horizon=40`、`num_inference_timesteps=4`、
   `num_timestep_buckets=1000`、2 相机、Se=141 全部照 checkpoint/HF。
   §4.1 记的"把 horizon 改成 16 可省 DiT ~2.4× M 维工作量"**仍然只是模型侧杠杆记录，未启用**。

#### 6.6.7 交付物

| 文件 | 角色 |
|---|---|
| `flash_rt/models/groot_n17/weight_spec_orin.py`（新增） | 由已验证的 N1.7 spec **逐项改写**派生（FP16 cast→BF16、丢弃 `Quant`），因此 checkpoint 键覆盖率不可能漂移；若残留任何 `Quant` 直接 `RuntimeError`。实测 998 项、op 直方图 `{'ToBf16': 105, 'T': 34}`、sink 列表与基线一致。**基线 `weight_spec.py` 未改一字** |
| `flash_rt/models/groot_n17/pipeline_orin.py`（新增） | `pipeline_rtx_fp16.py` 的 BF16 兄弟：**同样的阶段分解与算子顺序**，这样逐级 cos 隔离的是 dtype 而不是算法 |
| `flash_rt/hardware/rtx/attn_backend_groot_n17_orin.py`（新增） | bf16 slot + LLM 站点改走 FA2 `fwd_bf16_causal` 原生 GQA |
| `flash_rt/frontends/torch/groot_n17_orin.py`（新增） | 继承 `GrootN17TorchFrontendRtxFP16`，只覆写带 dtype 的方法；`use_fp8/use_fp4` 在**任何 CUDA 工作之前**拒绝 |
| `flash_rt/hardware/__init__.py`（改） | `("groot_n17","torch","rtx_sm87")` **双注册**：`_PIPELINE_MAP` + `_SM87_ALLOWED` |
| `tests/_helpers/groot_orin/capture_aux.py`（新增） | 官方模型一次 forward 抓齐 7 个必需 aux 张量；缺任何一个直接 `SystemExit`，不产出半份 aux |
| `tests/test_orin_groot_n17_dispatch.py`（新增，11 项） | 无 GPU/无 checkpoint 即可跑：dispatch、SM87 allowlist、其他 arch 不被抢、`_require_arch` 四态、低比特档在任何 CUDA/checkpoint 工作前被拒、Orin spec 无 `Quant`、**AST 遍历 pipeline 里每个 `gemm.*`/`fvk.*` 断言本次构建真有该符号** |
| `tests/test_orin_groot_n17_precision.py`（新增，10 项） | G1–G5 + stale-value + 锁频延迟；checkpoint/fixture 缺失时整体 skip 而非假绿 |

> 那条 AST 测试值得单独记：它把"kernel 在这个 arch 上没编译"从**运行期段错误/AttributeError**
> 变成**收集期失败**，且不需要 GPU。§6.6.4 的 `quantize_int8_rowwise_fp16`
> 有声明无编译正是这类陷阱。

### 6.7 image→embeds 融合（已实现，摘掉了对 HF forward 的依赖）

**动机不只是"少一个依赖"，而是 §6.6.6 第 1 条那个口径问题被低估了。** 实测：
HF 为产出 `llm_input_embeds` 必须跑**完整视觉塔** —— `visual(pixel_values, grid_thw)`
锁频 median **58.07 ms**（其中 `patch_embed` 1.40 ms）。而 FlashRT 又**重跑**了
ViT 前 18 层（24.10 ms）去取 DeepStack taps。也就是说融合开启前的真实独立部署成本是

| | 融合前 | 融合后 |
|---|---|---|
| HF 视觉塔（为产出 `llm_input_embeds`） | **58.07 ms** | **0**（不再需要） |
| FlashRT backbone | 58.32 ms | **66.13 ms**（ViT 24 层，+7.81） |
| FlashRT DiT（graph） | 58.47 ms | 58.43 ms |
| **真实独立总成本** | **174.86 ms** | **124.56 ms** |

⇒ **1.40× 改善**，且 ViT 不再被算两遍。
（"同一边界 3.09×→2.89×"是**变差**的，但那个边界本来就把 HF 的 58.07 ms 排除在外，
所以它一直在**高估**本通路。§0.1 的表已按两个口径并列。）

> **两次独立量测**（均锁频 1300.5 MHz、median of 10、`torch.cuda.Event`）：
> 第一次 backbone 66.13 / DiT 58.43 / **合计 124.56 ms → 2.89×**；
> 第二次 backbone 66.19 / DiT 58.23 / **合计 124.42 ms → 2.90×**。
> 逐项差 ≤0.20 ms（≤0.35%），故本文按 **124.4–124.6 ms / 2.89–2.90×** 记，
> 单点数字不主张到小数点后第二位的精度。

**实现（复用 Thor，不新写算法）**：Thor FP8 早已实现同一套融合，由
`use_pe = "pixel_values" in aux` 开关（`groot_n17_thor_fp8.py:374`）。本通路照搬其
5 步结构（patch embed → ViT → final merger → 文本 embed 查表 → 按视觉位 scatter），
只换 dtype 与 kernel 名。**未修改 Thor 文件**（红线 #1）。

| 步骤 | Thor（fp16） | Orin（bf16） |
|---|---|---|
| patch embed | `gemm.fp16_nn` + `add_bias_fp16` + `residual_add_fp16(pe_pos)` | **`torch.nn.functional.conv3d`**（见下） |
| pos embed | `_fast_pos_embed_interpolate`（该方法在 `ThorFP8` 上，**不在 Orin 的 MRO 里**） | 抽成共享自由函数 `_groot_n17_fusion.fast_pos_embed_interpolate` |
| 文本 embed | `fvk.embedding_lookup_bf16` | 同（SM87 上 **`embedding_lookup_bf16` 存在、`_fp16` 不存在**） |
| final merger | `layer_norm_fp16` + `fp16_nn` + **`gelu_inplace_fp16`（tanh）** | `layer_norm` + `bf16_nn` + **`gelu_erf_bf16`（exact erf）** |
| scatter | `llm_h.index_copy_(0, vis_idx, mg_img)` | 同 |

#### 6.7.1 两个必须记下的数值陷阱（都实测过，都会静默降精度）

**(a) patch embed 必须用 `conv3d`，不能用展平的 GEMM。**
`Qwen3VLVisionPatchEmbed` 是 `Conv3d(3,1024,(2,16,16),stride=(2,16,16))`；
在已展平的 `(512,1536)` 上它数学上等价于 `(512,1536)@(1536,1024)+bias`，
但**累加顺序不同**。三种写法的实测结果：

| patch embed 写法 | ViT 输入 vs HF | `vit_block_17` | `backbone_features` |
|---|---|---|---|
| `gemm.bf16_nn`（展平 GEMM，bf16） | max&#124;d&#124; ≠ 0 | — | 0.995323 |
| torch fp32 matmul 后降 bf16 | max&#124;d&#124; = 0.125 | 0.991470 | 0.996077 |
| **`F.conv3d`（与 HF 同一算子同一 dtype）** | **bit-identical，max&#124;d&#124; = 0** | **0.998156** | **0.996951** |

**(b) pos-embed 插值必须在表自己的 dtype 里算。**
HF 用 `torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype)` 构造双线性权重
（即 **bf16**），gather / 乘 / 四路求和全在 bf16。我最初按"更精确"写成 fp32 再一次性降回，
结果 **max&#124;d&#124; = 0.125（1 ULP）** —— 而这 1 ULP 会经过 24 层残差塔放大。
改成与 HF 同 dtype 后 **bit-identical**。

> 这两条合起来的教训：**在一个 24 层残差塔前面，"更精确"不等于"更接近参考"。**
> 目标是与 HF bit 对齐，不是与数学真值对齐 —— 因为门是拿 HF 当参考的。
> 修好之后两个模式的 layer 0/5/11/17 余弦**逐位相同**
> （0.999994 / 0.999892 / 0.999618 / 0.998156），证明差异只剩"多跑的 6 层"。

#### 6.7.2 融合后的门（`THR_FUSED_CONSUMED = 0.995`，含理由）

| 门 | 融合前 | **融合后** | 说明 |
|---|---|---|---|
| `vit_block_17`（tap） | 0.998156 | **0.998156** | 逐位相同 |
| `deepstack_out_0/1/2` | 0.999960 / 0.999687 / 0.999224 | **完全相同** | 证明 ViT 输入 bit 对齐 |
| `vit_block_23`（新增） | —（不跑） | 0.993874 | 与旧 24 层口径记录值一致，是 bf16 地板 |
| `llm_h` | 0.999998 | 0.999975 | |
| `backbone_features` | 0.999729 | **0.996951 / 0.998082**（frame 0/300） | ↓ 但见下 |
| G2 `dit_step_input` | 0.999992–3 | **0.999992–3** | 不变（backbone 只经 cross-attn 进 DiT） |
| G2b velocity | 0.999855–0.999945 | 0.999822–0.999936 | 变化 ~1e-5 |
| G3 `final_actions_norm` | 0.999999 | 0.999998 | |
| **G4 解码 action** | cos 1.000000；max&#124;d&#124; **0.004307**（f0）/ **0.003260**（f300）rad | cos 1.000000；max&#124;d&#124; **0.008613**（f0，0.49°）/ **0.003260**（f300，**不变**）rad | 四者 cos 均 1.000000 |
| G5 graph≡eager / replay≡replay | bit-identical | **bit-identical** | |

**为什么接受 `backbone_features` 从 0.9997 降到 0.9970**（必须写清，否则就是放宽门）：

1. **降幅的来源已定位，不是融合的算术错。** 分离实验（torch fp64 参考）：
   - `merger(HF 的 vit_block_23)` vs HF 的 image token：**cos 0.99999632**
   - `kernel merger` vs `torch merger`（**同一输入**）：**cos 0.99999455**
   ⇒ merger 的 norm-before-shuffle 顺序、exact-erf GELU、fc1/fc2 布局全部正确；
   残差**全部**来自 ViT 24 层的 bf16 累积（`vit_block_23` 0.993874）。
2. **参考侧本身更准，这不是可对等的目标。** HF 的视觉塔因一个子模块留在 fp32 而
   fallback 掉 FA2（日志 `current dtype in Qwen3VLVisionModel is torch.float32`）；
   融合前那 0.9997 是"拿 HF 自己算好的 image token"换来的，本质是把这部分误差
   **外包给了 HF**。自己算就必须自己承担 bf16 地板。
3. **它被强烈衰减，但并非完全到不了输出（诚实修正）。** DiT 只通过
   cross-attention（141 token）读 `backbone_features`，逐 token 误差被平均掉：
   velocity 只动 ~1e-5，两帧的 **cos 均仍为 1.000000**。max&#124;d&#124; 方面
   **frame 300 完全不变**（0.003260 rad），frame 0 从 0.004307 翻倍到
   **0.008613 rad（0.25°→0.49°）**，而关节量程是 [-133°, 152°]。
   ⇒ 不是"零影响"，是"影响落在 0.5° 以内且 cos 不动"。
4. **融合自身的算术另有严格门**（`test_fusion_reproduces_hf_embeds`）：
   文本 token embed 必须 **max&#124;d&#124; == 0**（纯 gather，任何非零都是索引 bug 而非舍入），
   kernel merger 对 torch fp64 参考必须 ≥0.9999，融合结果整体 ≥0.995。
   ⇒ 宽松门只覆盖"ViT 深度"，不覆盖"融合对不对"。

**新增/修改文件**：
- `flash_rt/frontends/torch/_groot_n17_fusion.py`（新增）：`fast_pos_embed_interpolate`
  共享自由函数。**为什么不能继承**：Orin 的 MRO 是
  `Orin → RtxFP16 → Rtx → Thor → object`，而该方法定义在 `GrootN17TorchFrontendThorFP8`
  （**不是祖先**）。抽成自由函数而不是复制第二份 —— 那 40 行索引算术抄错一次要查一小时。
  Thor 保持原样（红线 #1），把它折进来是另一次针对已上线通路的独立重构。
- `flash_rt/frontends/torch/groot_n17_orin.py`（改）：新增 `fuse_image_embeds=True`
  构造开关（默认开）；`_vit_layers` 随之在 24 / 18 间切换；`set_prompt` 构建 6 个
  per-prompt 常量（`_fus_pv` / `_fus_pos` / `_fus_ids` / `_fus_vis_idx` /
  `_fus_pe_w` / `_fus_emb`）；`_run_kernel_backbone` 双模式。
  **`fuse_image_embeds` 必须在 `super().__init__()` 之前赋值**，因为那次调用会跑
  `_load_weights`，而它要读这个标志决定加载多少层 ViT。
- `tests/_helpers/groot_orin/capture_aux.py`（改）：n17 新增采集 `pixel_values`
  （原始 patch 矩阵 `(512,1536)`）与 `input_ids`。
  ⚠️ `input_ids` **必须从外层 `Qwen3VLModel` 取**：文本模型是被
  `language_model(inputs_embeds=...)` 调用的，那一层 `input_ids is None`，
  原来的钩子因此静默采不到（已由 `REQUIRED_N17` 拦住）。
  ⚠️ transformers 4.57 里 `Qwen3VLVisionModel.forward` 的首参叫 **`hidden_states`**
  而非 `pixel_values`（第一行就 `self.patch_embed(hidden_states)`）；钩子两名都收。
- `tests/test_orin_groot_n17_precision.py`（改）：新增 `test_fusion_reproduces_hf_embeds`；
  `THR_FUSED_CONSUMED` 与逐条理由；修掉一处**真实的测试 bug** —— 原先拿
  `cap["vit_h"]` 去比 `vit_block_17`，融合开启后 `vit_h` 是 layer-23 输出，
  该比较测的是空气（cos 0.271）；改为比逐层快照 `cap["vit_block_17"]`。

**复跑命令**（少任何一项都会假失败，都是实测踩过的）：

```bash
source /mnt/venvs/groot_n17/bin/activate
FLASHRT_GROOT_N17_CHECKPOINT=/mnt/GR00T/so101_sim_rynnbot/checkpoint-89-1.000 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/mnt/Isaac-GR00T:. \
  python -m pytest tests/test_orin_groot_n17_{dispatch,precision}.py -q
# => 23 passed（precision 12 + dispatch 11）
```

- `PYTHONPATH` 里的 `/mnt/Isaac-GR00T` **不是可选的**：G4 调 `denormalize_action`，
  它会惰性 `import gr00t.model.gr00t_n1d7` 去建 HF processor。缺了就
  `ModuleNotFoundError: No module named 'gr00t'`（只挂 G4 那 2 项，其余 10 项照过
  ⇒ 极易被误读成"融合坏了"）。
- `HF_HUB_OFFLINE=1`：本机无外网，HF 会对 `Cosmos-Reason2-2B` 发 HEAD 请求并
  **每轮重试 ~30 s**，不加就白等几分钟。
- 未设 `FLASHRT_GROOT_N17_CHECKPOINT` 时 precision 文件**整模块 skip**
  （不是 fail）—— 所以"11 passed, 1 skipped"意味着门根本没跑，别当绿。
- 量测前必须确认锁频：`cat /sys/class/devfreq/17000000.gpu/{cur,min,max}_freq`
  三个值都应是 `1300500000`。

**仍未摘掉的 HF 依赖**（诚实声明）：`rope_cos` / `rope_sin`（M-RoPE 表）与
`visual_pos_masks` 仍来自 aux。前者可由 `grid_thw` + 位置推导，后者等价于
`input_ids == image_token_index`，都不难，但属另一件事；
`pixel_values` 本身也仍是 HF processor 的产物（resize/normalize/im2col）。

### 6.8 INT8 DiT 真激活 fake-quant 门（§6.5 判定 1 要求的实验，已做完）

**结论先行：过门。杠杆 #3 的 `⚠️` 撤回，改回 `✅`（精度侧）；
且 QuaRot（杠杆 #6）对 DiT 不需要。**

**方法**（纯精度实验，**不产生任何速度主张**）。没有改 `pipeline_orin.py`：
问题问的是"per-row INT8 W8A8 这个**数值方案**在真实激活上保不保得住输出"，
而不是"我们的 kernel 对不对"。所以直接给**厂商模型自己**的 DiT 打补丁 ——
同一份权重、同一批形状、同一份真机激活，把 `action_head.model`（`AlternateVLDiT`）
下**全部 228 个 `nn.Linear`** 的 forward 换成 quantize→dequantize 后照常算：

| 补丁范围 | 数量 |
|---|---|
| `transformer_blocks[0..31]`（每块 7 个 Linear：adaLN、q、k、v、o、ff.proj、ff.2） | 224 |
| `timestep_encoder` | 2 |
| `proj_out_1` / `proj_out_2` | 2 |
| **合计** | **228** |

粒度与 `cutlass_int8_rowwise_bf16out` 一致：权重 **per-output-channel**（`W` 的行）、
激活 **per-token**（`x` 的行），都是对称 int8（±127）。然后跑**厂商自己的
`policy.get_action(obs)`**，`obs` 取自 §5.4 的真机 fixture。

**实验有效性先自证**（否则后面全部无意义）：未打补丁的基线对 fixture 里存的
`actions` 是 **max&#124;d&#124; = 0.000000**（两帧皆是）⇒ 噪声种子、state 单位、
preprocessing 全部复现，脚本里的 `assert` 卡住这一点。

**结果**（G4 = 解码后 action，对 fixture 的 HF 参考；两帧真机 SO101）：

| 变体 | frame 0 cos | frame 0 max&#124;d&#124; | frame 300 cos | frame 300 max&#124;d&#124; |
|---|---|---|---|---|
| 未量化基线 | 1.000000 | 0（= fixture） | 1.000000 | 0（= fixture） |
| `w_only`（仅权重 per-row INT8） | **1.000000** | 0.004307 rad（0.247°） | **1.000000** | 0.002418 rad（0.139°） |
| `a_only`（仅激活 per-row INT8） | 0.999999 | 0.004307 rad（0.247°） | 0.999999 | 0.003831 rad（0.219°） |
| **`w8a8`（两者，per-row）= 杠杆 #3 要上的档** | **0.999999** | **0.004450 rad（0.255°）** | **0.999999** | **0.003831 rad（0.219°）** |
| `w8a8_pt`（两者，**per-tensor**） | 0.999999 | 0.004632 rad（0.265°） | **0.999996** | **0.011492 rad（0.658°）** |

**四条判定**：

1. **per-row W8A8 过门，且余量很大。** cos ≥ **0.999999**（两帧），
   最坏 max&#124;d&#124; **0.255°**。对照：本通路**现在已经在跑的 bf16 融合档**
   在 frame 0 的 G4 是 **0.49°**（§6.7.2）。
   ⇒ **DiT 降到 INT8 引入的误差，比 ViT 留在 bf16 已有的累积误差还小。**
   §6.5 因为"逐通道比 142× 而非 52×"下调判定，是**保守但方向错了**：
   142× 的离群**没有**击穿 per-row 缩放。
2. **这是 scale 问题，不是 noise 问题（原则 #12 的分类被实测坐实）。**
   per-tensor 在 frame 300 掉到 cos 0.999996 / **0.658°**，比 per-row 差 **2.5×**；
   而 per-row 几乎无损。⇒ 修它的是**缩放粒度**，不是 AWQ/GPTQ/百分位标定那些
   治噪声的手段。**推论：DiT 不需要 QuaRot（杠杆 #6）** —— 旋转是治
   "per-row 也救不了"的逐通道离群（LLM 的 2147–3547×、ViT 的 120–205×），
   DiT 不属于那一类。§6.5 判定 1 末尾那句"旋转对 DiT 的 142× 也可能有效"
   **可以撤回**：不需要。
3. **误差主要来自激活，不是权重。** frame 300：`w_only` 0.139° < `a_only` 0.219°，
   且 `w8a8`(0.219°) ≈ `a_only`(0.219°) ⇒ 权重那一轴叠加上去几乎不再加错。
   实操含义：如果将来要退一档，**退权重（W8A16）没有意义**，要退就退激活。
4. **权重流量与 §6.1 交叉核对通过**（不是引用旧数字，是重算的）：
   228 个 Linear 共 **1091.17 M** 权重参数 ⇒ bf16 **2.182 GB/步 × 4 步 = 8.729 GB**，
   INT8 **1.091 GB/步 × 4 步 = 4.365 GB**。与 §6.1 表里的 8.73 GB / §6.2 的
   "8.73 GB → 4.37 GB" 逐项吻合。

**这个门的边界（诚实声明，别把它当"INT8 已上线"）**：

- fake-quant 用 **fp32 算 quant/dequant、再走 bf16 `F.linear`**。真 kernel 是
  int32 累加 + epilogue 里应用 scale。int32 累加比 bf16 累加**更准**，
  所以本门在累加这一轴上是**偏保守**的；但它**没有**建模 epilogue 的
  scale 应用顺序，也没有建模 M=41 的 tile 边界效应。
- 只覆盖 `action_head.model`（DiT）。`state_encoder` / `action_encoder` /
  `action_decoder` / `vlln` / `vl_self_attention` **未量化**，与杠杆 #3 的范围一致。
- **不产生速度主张。** 1.56×（58.4→37.5 ms）来自 §6.3 的独立微基准；
  本节只回答"能不能上"，不回答"上了多快"。接上 `cutlass_int8_rowwise_bf16out`
  之后必须按原则 #16 重做 in-pipeline A/B（锁频、median、graph 已预捕获）。
- 脚本 `/tmp/dit_int8_fakequant.py` 是一次性实验件，**未进仓**；
  要复现按本节方法重写即可（228 个 Linear、per-row 对称 int8、厂商 `get_action`）。

**下一步**：接 `cutlass_int8_rowwise_bf16out`（128×128 tile，§6.3 实测优于 `_t64x128`）
到 DiT 的 224 个 block GEMM，权重量化离线做一次（per-output-channel），
激活用 `quantize_int8_rowwise` [PRE] 动态 per-row（**必须 device-side scale**，
否则破 CUDA graph，§6.3 已记 "无图反而慢 2.1×"）。目标：DiT 58.4→37.5 ms ⇒
边界 **124.5 → ~103.7 ms**。 ⇒ **已做完，见 §6.9**（实测 DiT 1.39–1.42×，边界 **107.8–108.8 ms**）。

### 6.9 INT8 DiT 接入（杠杆 #3 已交付：DiT 1.39–1.42×，边界 → 107.8–108.8 ms / 3.31–3.34×）

**结论先行**：`use_int8_dit=True` 打开后，DiT 4 步 **58.5 → 41.5 ms（1.39–1.42×）**，
边界 **124.9 → 108.0–108.8 ms**，对 HF 同边界 **2.89× → 3.31–3.34×**。
G1–G5 全过，**已由使用方拍板翻成出厂默认档**（见 §6.9.6）。

#### 6.9.1 实现（additive，bf16 通路一字未改语义）

| 位置 | 改动 |
|---|---|
| `models/groot_n17/pipeline_orin.py` | `dit_forward` 加**双档分发**：`"q_w8" in weights` 选 INT8。分发在循环**外**解析成三个闭包（`qz` 量化并返回该读的指针 / `mm` GEMM / `W` 解析权重族），所以 32 层 × 6 站点没有一个 `if` |
| 同上，新增 `_int8_quantize` / `_int8_nn` | 两个薄封装；`_int8_nn` 检查 `cutlass_int8_rowwise_bf16out` 的返回 status，非 0 直接 `RuntimeError` |
| `frontends/torch/groot_n17_orin.py` | `use_int8_dit=False` 构造开关；`_quantize_dit_weights()` 在 `super().__init__()` **之后**跑（要先有 bf16 权重）；`_allocate_infer_buffers` 多分 8 个 int8/fp32 scratch；`_dit_weights` 多吐 `F_w8`/`F_s` 指针表 |

**三个必须记下的实现要点**：

1. **量化次数是每层 4 次，不是 6 次。** adaLN 后的 `xn` 同时喂 Q、K、V，
   所以只量化一次复用（`qz` 返回指针就是为这个）。4 个站点 = `xn1`、
   attention 输出 `O`、`xn2`（no-affine LN 后）、`ff_out`（GELU 后）。
   ⇒ 32 层共 **128 次 quantize + 160 次 INT8 GEMM**（16 self×6 + 16 cross×4）。
   **这两个数被无 GPU 契约测试钉住**（`test_int8_dit_tier_selection_and_quantize_pass_count`），
   退化会立刻被抓到，不用等 profiler。
2. **`(N,K)` 必须逐层从权重形状推导，不能用一张表。** 我第一版写死了
   `nk = {"k": (D, D), ...}`，立刻炸在 `_dit_k_w is (2048, 1536)` ——
   **K/V 的形状随层奇偶变**：cross 层从 2048 维 backbone 特征投影（(2048,1536)），
   self 层从 1536 维 action 流投影（(1536,1536)）。实测确认：
   `q`/`o` 全 (1536,1536)、`ff_proj` (1536,6144)、`ff_down` (6144,1536)、
   `k`/`v` 偶数层 (2048,1536) 奇数层 (1536,1536)。
   `dit_forward` 对 k/v 硬编码 `(Sa, D, D)` 且只在 self 层调用 ⇒ 我加了一条
   断言：奇数层的 k/v 量化结果必须是 `(D, D)`，否则构造期就报错。
3. **布局约定是反的，而且方阵上转置错误是静默的。** spec 的 `T()` 让
   `_dit_<F>_w` 是 **(K,N)**（`gemm.bf16_nn` 的 B 操作数），而
   `cutlass_int8_rowwise_bf16out` 要 **nn.Linear 原生的 (N,K)**。
   `q`/`k`/`v`/`o` 在实际用到的层上都是方阵 ⇒ **维度断言抓不到转置错误**。
   所以加了 `_smoke_int8_dit_layout()`：每个权重族在 pipeline 真实形状上跑
   **一次真 INT8 launch**，与 `x @ w_kn`（即 bf16 档 `bf16_nn` 算的东西）比 cos，
   <0.99 就拒绝构造。这与融合期 `_merger_fc1_w` (4096,4096) 是同一个陷阱。
   （输入是合成的——**这里只验布局**；保真度另由真机 fixture 的 E2E 门管。）

**图安全**：激活 scale 全程 **device-side**（`quantize_int8_rowwise` 写进设备指针，
CUTLASS epilogue 在设备上读），scratch 在 capture **之前**分配好。
实测 graph≡eager、replay≡replay 均 **bit-identical**（两帧两档）。

#### 6.9.2 精度（两帧真机 SO101，与 bf16 档同一门 `THR_CONSUMED = 0.999`）

| 门 | bf16（frame 0 / 300） | **INT8（frame 0 / 300）** |
|---|---|---|
| G2 `dit_step_input` | 0.999992 | **0.999992**（不变，backbone 未动） |
| G2b velocity（4 步范围） | 0.999822–0.999936 / 0.999899–0.999953 | **0.999512–0.999697 / 0.999574–0.999684** |
| G3 `final_actions_norm` | 0.999998 / 0.999999 | **0.999997 / 0.999998** |
| G4 解码 action cos | 1.000000 / 1.000000 | **0.999999 / 0.999999** |
| G4 max&#124;d&#124; | 0.008613（0.493°）/ 0.003260（0.187°）rad | **0.008613（0.493°）/ 0.006345（0.364°）rad** |
| G5 graph≡eager、replay≡replay | bit-identical | **bit-identical** |
| INT8 vs bf16（归一化 action） | — | cos **0.999997**；max&#124;d&#124; 0.031250 / 0.023438 |

**没有为 INT8 放宽任何门**：velocity 从 ~0.9998 降到 ~0.9995，仍在
`THR_CONSUMED = 0.999` 之上；G4 最坏 0.364°，与 §6.8 fake-quant 预测的
0.255° 同量级（略高，因为 fake-quant 用 fp32 算 quant/dequant，真 kernel 走
int8→int32→bf16 的完整舍入链）。frame 0 的 max&#124;d&#124; **与 bf16 完全相同**
（0.008613）⇒ 那一帧的最坏误差元素由 backbone（ViT bf16 累积）决定，不是 DiT。

#### 6.9.3 延迟（两个独立口径互相印证，各 3–5 次重复）

**口径 A：单档隔离量测**（每次进程里只驻留一个 frontend，median of 10，锁频 1300.5 MHz）

| 档 | DiT ×4（graph），3 次独立跑 | **边界合计** | vs HF 360.38 ms |
|---|---|---|---|
| bf16 | 58.46 / 58.57 / 59.06 ms（另两次早于本次改动：58.23 / 58.43） | **124.62 / 124.88 / 125.28 ms**（早两次 124.42 / 124.56） | 2.877–2.892× |
| **INT8** | **41.49 / 41.50 / 41.74 ms** | **107.80 / 107.97 / 108.00 ms** | **3.337–3.343×** |

⇒ DiT 中位 **58.57 → 41.50 ms = 1.411×**；边界中位 **124.88 → 107.97 ms**，
净省 **~16.9 ms（1.157×）**。
bf16 的五次跨两个 session，极差 0.7%（124.42–125.28）；INT8 三次极差 0.2%。
**所以单点绝对值只主张到 ±0.7%，比值才是稳的。**

⚠️ 顺手排掉一个假警报：接入 INT8 后 bf16 边界从 124.42–124.56 变成
124.62–125.28，看着像 `dit_forward` 的闭包分发引入了回归。**复测三次证明是
session 间噪声**（0.3%，落在 bf16 自己的极差内），且机理上不该有影响 ——
延迟测的是 **graph replay**，replay 里没有 Python，闭包只在 capture 时跑一次。

**口径 B：配对交替 A/B**（AGENTS.md §3.8 要求的口径：两档交替、取中位数，
`test_int8_dit_paired_alternating_speedup`，median of 11）

| | bf16 | INT8 | 比值 |
|---|---|---|---|
| 第一次（`/tmp` 脚本） | 59.27 ms | 42.21 ms | 1.404× |
| 第二次（pytest 内） | 58.96 ms | 41.45 ms | **1.422×** |
| 第三次（默认档翻转后复跑） | 58.77 ms | 42.23 ms | **1.391×** |

⇒ **口径 A 的 1.411× 与口径 B 的 1.391–1.422× 一致**（三次极差 2.2%）。
两档同时驻留只让 bf16 从 58.5 抬到 58.77–59.27 ms（+0.5–0.8%），
所以交替量测的 L2 污染很小 —— 这一点本身值得记：**在这个规模上
配对交替的代价小于它的收益**。第三次是纯复跑（代码未变，只改了文档），
它落在前两次的区间下沿 ⇒ **1.39–1.42× 是 session 噪声带，不是趋势**。

**本文统一按 DiT 1.39–1.42×（中位 ~1.41×）/ 边界 107.8–108.8 ms / 3.31–3.34× 记。**

#### 6.9.4 measured vs predicted（原则 #15）

| | §6.3 预测 | **实测** | 偏差 | 机理 |
|---|---|---|---|---|
| DiT 4 步 | 58.4 → 37.5 ms（**1.56×**） | 58.5 → 41.5 ms（**1.411×**） | **−9.6%** | §6.3 的 µbench 是**合成的**：随机权重，且**没有**跑真实循环里的 attention / adaLN / no-affine LN / GELU / bias / residual —— 这些**不随 INT8 变快**，于是稀释了比值 |
| 边界合计 | ~103.7 ms | **107.8–108.8 ms** | **+4.0%** | 同上 |

**记：合成 µbench 能定"值不值得接"（1.56× 与 1.42× 是同一个结论），
但不能当交付数字。** 这与 §6.4 的教训同源（roofline 求和的分项误差会互相掩盖），
只是这次偏在"忽略了不被加速的部分"。

#### 6.9.5 新发现的 kernel 缺口（红线 #8：只报不做）

§8 原待办写着"用 `gate_residual_ada_norm_int8` 把量化融进 ada_norm，省掉独立 pass"。
**这条不成立，已撤回**：读了 `csrc/kernels/fusion.cu:157-225`，该 kernel 算的是

```
rms = rsqrtf(block_reduce_sum(local_sum) / dim + eps)     // 无减均值
```

即 **RMS** norm；而 GR00T DiT 用的是**减均值**的 `AdaLayerNorm`（eps=1e-5），
且它的残差结构是 `residual + x*gate`（带门），DiT 这里是普通 `h += o_out`（无门）。
pi05 用它时传的 `weight` 是 `_rms_ones_dec`（全 1），也印证是 RMS 语义。
**接上去会静默改变数学**，所以不接。

同理，SM87 上**没有** int8 输出的 `ada_layer_norm`，也**没有** int8 输出的
`layer_norm_no_affine`（有 `_fp8_static_bf16` 变体，但 SM87 无 FP8）。
⇒ **4 个量化 pass 必须独立存在**。它们的代价已包含在 41.50 ms 里：
每步额外流量 ~42 MB，对 2182 MB 的 bf16 权重流量是 ~2%。

**要报给 kernel owner 的缺口**：SM87 上 (a) int8 输出的 ada-**LayerNorm**
（减均值 + scale/shift 调制 + per-row 对称 int8 输出 + device-side scale），
(b) int8 输出的 `layer_norm_no_affine`，(c) int8 输出的 `gelu_tanh`。
三者各能省掉一个独立量化 pass；按 §6.3 的口径这可能把 DiT 从 41.50 ms
再往下推，但**未量测，不作为主张**。

#### 6.9.6 代价与默认档决策

**内存**：INT8 档**同时保留** bf16 副本（2.18 GB）与 int8 副本（**1.19 GB**，
比 §6.8 估的 1.09 GB 多，因为 cross 层 k/v 的 (2048,1536) 也量化了）。
保留 bf16 是**故意的**：配对交替 A/B 要求两档在同一进程里可比
（AGENTS.md §3.8）。若确定只跑 INT8，可释放 bf16 的 q/k/v/o/ff_proj/ff_down
（**但 cross 层 k/v 必须留** —— `_precompute_dit_cross_kv` 在 torch fp32 里用它们）。
这与 §6.6.6 第 3 条的冗余权重内存是同一笔账，一起清理。

**默认档：INT8（`use_int8_dit=True`）—— 已由使用方拍板。**

出厂配置即 `GrootN17TorchFrontendOrin(ckpt, embodiment_tag=...)`，无需传参；
退回 bf16 用 `use_int8_dit=False`。理由与代价都记在这里：

- **收益**：DiT 1.39–1.42×，边界 124.9 → **108.0–108.8 ms（3.31–3.34× vs HF）**。
- **代价**：G4 max&#124;d&#124; 在 frame 300 上翻倍（0.187°→**0.364°**），
  frame 0 不变；两帧 cos 均 **0.999999**。关节量程约 [-133°, 152°]（285°），
  所以 0.364° 是量程的 **0.13%**。
- **没有为它放宽任何门**：velocity 最低 0.999512，仍在 `THR_CONSUMED=0.999` 之上。
- **测试自动跟随默认档**：`DEFAULT_INT8` 由 `inspect.signature` 从构造函数读出，
  `prompted` fixture 与 `test_latency_at_the_hf_boundary` 都用它 ⇒
  **翻默认不需要改任何测试**，而且头条延迟数字永远描述的是真正出厂的那一档
  （一个硬编码默认值的测试会安静地一直验证没人跑的档）。
  另有 `test_int8_dit_is_the_shipped_default` 无 GPU 钉住这件事：
  误翻回去是**测试失败**，不是静默降速。
- **内存代价成为默认**：+1.19 GB int8 副本，且 bf16 DiT 权重仍保留（配对 A/B 需要）。
  若确定不再 A/B，可释放 bf16 的 q/o/ff_proj/ff_down 与 self 层 k/v
  （**cross 层 k/v 必须留** —— `_precompute_dit_cross_kv` 在 torch fp32 里用）。

**新增/修改文件**：
- `flash_rt/models/groot_n17/pipeline_orin.py`（改）：`_int8_quantize` / `_int8_nn` /
  `dit_forward` 双档。
- `flash_rt/frontends/torch/groot_n17_orin.py`（改）：`use_int8_dit`、
  `_DIT_INT8_FAMILIES`、`_quantize_dit_weights`、`_smoke_int8_dit_layout`、
  int8 scratch 分配、`_dit_weights` 扩展。
- `tests/test_orin_groot_n17_precision.py`（改）：`TIERS` 参数化 +
  `prompted_by_tier` 缓存 + `test_int8_dit_paired_alternating_speedup`。
  **12 → 17 项**。
- `tests/test_orin_groot_n17_dispatch.py`（改）：三个无 GPU 契约测试
  （缺 scratch 必须 `KeyError` 且不预先 launch；档位选择与 128/160 计数；
  `test_int8_dit_is_the_shipped_default` 钉住出厂默认）。
  **11 → 14 项**。
- **合计 31 passed**（原 23）。

> 那条"缺 scratch 必须报错"的测试值得单记：INT8 静默回落 bf16 在数值上
> **更准**，所以任何"比输出"的验证都抓不到它 —— 只会让所有 INT8 的精度与
> 延迟数字悄悄变成 bf16 的。这正是 AGENTS.md §3.6 说的
> "identical output alone proves nothing"。测试还断言**报错前一个 kernel 都没发**。

### 6.10 动态量化门：LLM 与 ViT（按"优先动态量化、尽量不用 QuaRot"执行）

**动机**：使用方指示"若 outlier 太大，则可以考虑采用动态量化，尽量不使用 QuaRot"。
DiT **本来就是**动态量化（`quantize_int8_rowwise` 每次 forward 在设备上重算
per-token amax，无标定、无静态 scale），所以这条指示对 DiT 无需动作。
真正 outlier 大的是 **LLM（2147–3547×）与 ViT（120–205×）**，而 §6.5 对它们
判了"per-row 也救不了 ⇒ 只能旋转"。**但那个判定和 DiT 的 142× 判定一样，
从未做过真激活门** —— 而 DiT 的判定被实测推翻了。所以两个都重测。

**方法**：与 §6.8 完全相同（给厂商模型打 quant→dequant 补丁，跑厂商自己的
`get_action`，真机 fixture，未量化基线必须先复现 fixture 到 max&#124;d&#124;=0）。
全部是**动态**量化：scale 每次从当次 forward 自己的 amax 算，无标定集、无静态 scale、无旋转。

#### 6.10.1 LLM（16 层 × 7 Linear = 112）—— **§6.5 判定 2 被推翻，过门**

| 变体 | 量化范围 | `llm_layer_15` cos（f0 / f300） | G4 cos | G4 max&#124;d&#124;（f0 / f300） |
|---|---|---|---|---|
| 未量化基线 | — | 1.000000 | 1.000000 | 0 / 0 |
| **`ffn_row`** | mlp 的 gate/up/down（48 Lin），per-row | **0.999884 / 0.999880** | 0.999999 | **0.247° / 0.242°** |
| `ffn_pt` | 同 48 Lin，per-tensor | 0.999081 / 0.999136 | 0.999980 / 0.999979 | 2.221° / 1.063° |
| **`all_row`** | + q/k/v/o（**全 112 Lin**），per-row | **0.999876 / 0.999859** | **0.999999 / 0.999997** | **0.439° / 0.573°** |
| `all_pt` | 全 112 Lin，per-tensor | 0.999069 / 0.999081 | 0.999986 / 0.999984 | 1.234° / 1.234° |

**三条结论**：

1. **per-row 动态量化能救 LLM，连 QKV/O 一起量化也能救。** `all_row` 的 G4 是
   cos **0.999999/0.999997**、最坏 **0.573°** —— 与我们**已经上线的 DiT INT8 档**
   （0.493°/0.364°）同一量级。§6.5 判定 2 的"❌ INT8、per-row 也救不了"
   **撤回**。`ffn_row`（杠杆 #4 的本体）更好：**0.247°/0.242°**。
2. **这确实是 scale 问题，不是 conditioning 问题 ⇒ 不需要 QuaRot。**
   per-tensor 比 per-row 差 **2.9×（all）到 9.0×（ffn，frame 0）**。
   原则 #12 的分类被第二次坐实（第一次是 DiT）。**杠杆 #6 对 LLM 也可以撤下。**
3. **§6.5 那条推理错在哪**（值得记，因为它是"看起来对"的推理）：
   它说"离群是逐通道的，每行都被同一个通道主导，所以 per-row 也救不了"。
   前半句对，**结论不跟着成立** —— 决定成败的是量化后**非离群通道的相对误差**，
   不是离群比本身。2147–3547× 描述的是**分布**，不是**误差**。
   而且 §6.5 自己记过一条关键事实：LLM 的离群通道 1793（幅度 15296.0）
   在 bf16 下**已经冻结**（该量级 ULP=64，任何 <32 的残差更新都被舍掉）⇒
   它在后 14 层几乎不携带信息，**把它量化得再粗也损失不了什么**。

#### 6.10.2 ViT（24 blocks × 4 Linear + 4 merger = 104）—— **§6.5 判定 3 成立，不过门**

| 变体 | 量化范围 | `vit_block_17`（tap，f0 / f300） | `vit_block_23`（f0 / f300） | G4 cos | G4 max&#124;d&#124;（f0 / f300） |
|---|---|---|---|---|---|
| 未量化基线 | — | 1.000000 | 1.000000 | 1.000000 | 0 / 0 |
| `blk_row` | 24 blocks（96 Lin），per-row | **0.971083 / 0.990567** | 0.964869 / 0.975181 | 0.999999 / 1.000000 | 0.740° / 0.247° |
| `blk_pt` | 同 96 Lin，per-tensor | **0.512692 / 0.758536** | 0.752110 / 0.729306 | 0.999990 / 0.999987 | 1.403° / 1.064° |
| `all_row` | + merger/DeepStack（104 Lin），per-row | 0.971083 / 0.990567 | 0.964869 / 0.975181 | 0.999998 / 1.000000 | 0.987° / 0.158° |

**不过门，而且不是勉强不过**：`vit_block_17` 是**被消费的 tap**，门是
`THR_VIT_TAP = 0.998`；per-row 实测 **0.971083**（frame 0）、**0.990567**（frame 300），
连宽松的 `THR_FUSED_CONSUMED = 0.995` 都不到。per-tensor 直接崩到 **0.5127**。
⇒ **§6.5 判定 3（ViT 保持 BF16）确认成立**，杠杆 #5 **关闭**。

> ⚠️ **一个必须记下的方法论教训：只看 G4 会误判。** `blk_row` 的 G4 是
> **cos 0.999999 / 0.740°** —— 单看这个数字会得出"ViT INT8 完全没问题"的结论，
> 而它的 tap 只有 **0.971**。原因与 §6.7.2 同一条：DiT 只通过 cross-attention
> 读 backbone 特征，逐 token 误差被平均掉。
> ⇒ **这正是原则 #2（生产指标 ≠ 内部精度）与逐级门存在的理由。**
> 如果本次只量 G4，就会把一个 tap cos 0.971 的档放上线。
> （顺带：`all_row` 在 frame 300 的 G4 是 0.158°，比 `blk_row` 的 0.247° 还"好"——
> 这是解码端的噪声级抖动，**不是**量化 merger 带来的改善，不可当结论用。）

#### 6.10.3 为什么离群比**小 10 倍**的 ViT 反而不过门（假说，与已记数据一致）

反直觉：LLM 逐通道比 2147–3547×，过门；ViT 只有 120–205×，不过门。
两条已记录的实测事实可以解释，**但这是假说不是证明**：

1. **LLM 的离群通道已经死掉了，ViT 的还活着。** §6.5 记过：LLM 通道 1793
   幅度 15296.0，在 bf16 下 ULP=64，`llm_layer_2..15` 的 max&#124;a&#124; **完全相同**
   ⇒ 该通道在后 14 层**已被 bf16 冻结**，粗量化它等于没损失。
   而 ViT 的离群（max&#124;a&#124; 从 block 10 起由 9.5 跳到 210）幅度只有 rms 的
   120–205×，**仍在 bf16 的分辨能力之内** ⇒ INT8 的 127 级是真的在毁信息。
2. **深度×宽度的错误复合方向相反。** ViT 是 24 层 × D=1024（深而窄），
   LLM 是 16 层 × D=2048。§6.6.5 已经量过 ViT 的 bf16 误差是
   **~1 ULP/层平滑累积**（layer 0 rel_l2 0.0034 → layer 17 的 0.061），
   本通路 bf16 在 tap 上就已经只剩 0.998156 的余量；INT8 再叠上去就穿底。
   LLM 那边 bf16 的 `llm_h` 是 0.999975，余量大得多。

**⇒ 结论（回答使用方的指示）**：
- **DiT**：动态 per-row，**已上线且为默认档**（§6.9）。
- **LLM**：动态 per-row **可行，不需要 QuaRot**；优先做 `ffn_row`（杠杆 #4，
  最坏 0.247°），all-tower 也过门（0.573°）但收益/风险比更差。
- **ViT**：动态量化**救不了**，这是全模型唯一"per-row 失效"的区域。
  按"尽量不用 QuaRot"的指示 ⇒ **ViT 保持 bf16**，杠杆 #5 关闭，
  杠杆 #6 降为"仅当将来确实要吃 ViT 那 41.2% 的 backbone 占比时才考虑"。

**实验件**：`/tmp/llm_int8_fakequant.py`、`/tmp/vit_int8_fakequant.py`，
均未进仓（方法已完整记录在本节，可重写）。两脚本都先断言未量化基线
复现 fixture 到 max&#124;d&#124;=0，否则实验作废。
⚠️ ViT 脚本踩到一个坑值得记：**vision block 的 forward 返回裸张量，
而 LLM decoder layer 返回 tuple** —— 一律写 `o[0]` 会静默取到**第一行**
（1024 元素而非 512×1024），cos 照算不误。已改成 `isinstance(o, tuple)` 判断
**并在比较处断言元素数**，因为这类错误不会抛异常，只会给出一个看着合理的假 cos。

### 6.11 LLM INT8 FFN（杠杆 #4）：接入 → 实测 → **不落地**，并意外量到真正的瓶颈

**结论先行**：杠杆 #4 **精度不过门，收益也拿不到**，代码已完整回退。
但为了搞清"为什么 GPU 明明省了 3.93 ms、墙钟只省 0.6–1.0 ms"，
量到了本通路一个**比任何精度档都大的结构事实**：

> **backbone 是 CPU 提交（launch）受限的，不是 GPU 算力或带宽受限的。**
> 一次 `_run_kernel_backbone` 的**纯 CPU 提交时间是 60.00 ms，墙钟 65.77 ms**。

⇒ 新**杠杆 #11（backbone CUDA-graph 捕获）成为 ROI 最高项，且不需要动任何精度档**。

#### 6.11.1 微基准（原则 #13：先量再写）

真实形状 M=Se=141、D=2048、FF=6144；配对交替取中位数。
**no-op 臂量出事件+同步地板 18.4 µs**，单算子行必须减掉它
（不减会把 quantize 读成 56 µs 而不是 37 µs）。

| shape (M,N,K) | bf16 µs | int8 µs（quant+gemm） | 比值 |
|---|---|---|---|
| gate/up (141,6144,2048) | 258.4 | 194.3 | 1.330 |
| down (141,2048,6144) | 257.5 | 198.8 | 1.295 |
| q/o (141,2048,2048) | 154.0 | 112.5 | 1.369 |
| **k/v (141,1024,2048)** | 79.4 | **93.3** | **0.851 ← 更慢** |

三条读数：

1. **k/v 在 INT8 下更慢**（N=1024 太小，quantize 的固定开销吃掉 GEMM 的收益）
   ⇒ all-tower 必须排除 k/v。**"过门"不等于"该做"**：§6.10.1 的 `all_row`
   精度过门，但其中两个 family 是**负收益**。
2. **`quantize_int8_rowwise` 是 ~37 µs 的固定开销**：cols=2048 与 cols=6144 一样。
   ⇒ 它**是延迟受限，不是带宽受限**。这条给 kernel 缺口定了价：
   把量化融进前一个 norm 的 epilogue，每处可回收 ~37 µs。
3. 换成**真实权重 + 真实 xn**（从 `llm_h` 做 RMSNorm 得到，逐通道比 30.8×）
   重跑同一个 FFN 块（gate/up 共享一次量化）：bf16 **704.4 µs/层 → int8 467.2 µs/层**，
   16 层 **省 3.80 ms（1.508×）**。与权重流量 roofline 的预测 **3.82 ms**
   几乎完全吻合 ⇒ **合成 µbench 在延迟上是可信的**。
   （§6.9.4 那条教训的准确表述应是：合成 µbench 不能当**交付数字**，
   但能定**值不值得做** —— 这次它两边都对。）

#### 6.11.2 接入（additive，已回退）

按 DiT 的同一套模式接完：`use_int8_llm_ffn`、
`_LLM_INT8_FAMILIES=("gate","up","down")`、`_quantize_llm_ffn_weights`
（(K,N)→(N,K) 逐层从张量推，并校验 gate/up 同形、down 是其转置）、
`_smoke_int8_llm_layout`（bind-time 真发射，参考 `x.double() @ w_kn.double()`，
cos<0.99 拒绝构造）、4 个 device-side scratch、`qwen3vl_llm_forward` 档位分发
（缺 scratch 必须 `KeyError` 且**一个 kernel 都不发**）。
每层 2 次量化（xn 被 gate/up 共享）+ 3 个 INT8 GEMM。

**红线 #5 的证据**：kernel 计数实测，每次 backbone
`quantize_int8_rowwise` **32** 次、`cutlass_int8_rowwise_bf16out` **48** 次，
bf16 档两者均为 **0**。

#### 6.11.3 精度：**不过门**（真机 frame 0）

| 张量 | bf16 档 | INT8 FFN 档 | 门 |
|---|---|---|---|
| `llm_layer_15` | 0.999975 | 过 0.999（fake-quant 侧测得 0.999884，§6.10.1） | `THR_CONSUMED` 0.999 ✅ |
| **`backbone_features`（= `vlsa_block_3`）** | **0.996951** | **0.992077** | `THR_FUSED_CONSUMED` **0.995 ❌** |
| G4 decoded cos | 0.999999 | 0.999999 | 0.999 ✅ |
| G4 max&#124;d&#124; | 0.008613（0.493°） | **0.005746（0.329°）** | — |

**⚠️ 又是同一条教训（第三次现形）：只看 G4 会放过它。**
G4 是 cos 0.999999、max&#124;d&#124; **0.329°，比 bf16 档自己的 0.493° 还小**。
抓住它的是 **`backbone_features` 这一级门** —— 也就是 DiT 真正消费的那个张量。
与 §6.7.2、§6.10.2 同一条机理。

机理也量到了：**vlln + 4 层 VLSA 把 LLM 的偏差放大约 40–120×（按 1−cos）**。
bf16：`llm_h` 2.5e-5 → `vlsa_block_3` 3.0e-3；INT8：2e-4 → 7.9e-3。

> **这条推出一条对 fake-quant 门的通用限制，比本节任何数字都值钱：**
> §6.10.1 的门**结构上就漏了这一级** —— 它量了 `llm_layer_15` 和 G4，
> 却没量 VLSA 之后的 `backbone_features`。
> **门必须覆盖到"被消费的那一级"，而不只是"被改的那一级"。**
> 一个过了 fake-quant 门的档，仍可能在接入后被中间级门拦下 ——
> 这不是门太严，是前一个门覆盖不全。

> 注：G4 比 bf16 更好**不是**"INT8 更准"。max&#124;d&#124; 的那个元素由 backbone 的
> bf16 累积误差决定（§6.9.2 已记：frame 0 的 max&#124;d&#124; 在 DiT 换档时**完全不变**），
> INT8 只是重新分布了误差。**不可当收益主张。**

#### 6.11.4 延迟：**GPU 省 3.93 ms，eager 墙钟只省 0.6–1.0 ms**

| 量法 | bf16 | INT8 FFN | 省 |
|---|---|---|---|
| 逐 kernel GPU 中位数（管线内事件插桩） | FFN 10.79 ms | FFN 6.86 ms | **3.93 ms** |
| LLM stage **CUDA-graph 捕获后**，配对交替 | **20.31 ms** | **15.54 ms** | **4.77 ms（1.307×）** |
| LLM stage **eager**，配对交替（3 个进程） | 22.37 / 22.42 / 22.46 | 21.35 / 21.13 / 20.55 | 1.02 / 1.29 / 1.92 ms |
| 整个 backbone **eager**，配对交替（3 个进程） | 66.48 / 65.93 / 65.67 | 66.04 / 64.70 / 64.65 | **0.44 / 0.85 / 0.97 ms** |

⇒ **GPU 层面的收益是真的**（捕获后 4.77 ms，比逐 kernel 预测的 3.93 还多，
因为捕获同时消掉了 INT8 臂多出的发射），**但 eager 墙钟只拿到约 1/5**。

#### 6.11.5 机理：**backbone 是 CPU 提交受限的**（本节最重要的产出）

> ⚠️ **本节的推论已被 §6.13.3 撤回，标题里的"受限"不成立。**
> 下面这张表量的是**CPU 在一次调用的墙钟里有多忙**，不是**墙钟里有多少由 CPU 造成**
> —— 二者差一个"重叠"。捕获后实测：CPU 提交从 51.97 ms 掉到 **0.46 ms**（−99%），
> 墙钟只从 62.47 掉到 **57.93 ms（−7.3%）**。
> launch 受限份额只能用 `(eager − replay)/eager` 量，**不能用 CPU/墙钟**。
> 本节其余内容（单算子 CPU 成本、三个量测陷阱、排除"INT8 发射更多所以更慢"）仍然有效；
> 表格保留作为"错在哪"的原始证据。

| 量 | bf16 档 | INT8 FFN 档 |
|---|---|---|
| 一次 backbone 的**纯 CPU 提交时间** | **60.00 ms** | 59.87 ms |
| 同一调用的墙钟 | 65.77 ms | 64.72 ms |
| CPU / 墙钟 | **91.2%** | 92.5% |

单算子 CPU 成本（`perf_counter`，不 sync，µs/次）：
`gemm.bf16_nn` **21.14**、`cutlass_int8_rowwise_bf16out` **10.88**、
`quantize_int8_rowwise` **9.76**、`rms_norm` **9.41**。
⇒ 每层 FFN 的 CPU 提交：bf16 **133.5 µs** vs int8 **75.6 µs**。

**所以"INT8 发射更多所以更慢"这条不成立，已被排除** —— INT8 臂的 CPU 提交
反而**更便宜 0.13 ms**。真实机理是：**eager backbone 的墙钟由提交节奏 +
每次发射的 GPU 前端间隙决定，而不是由 GPU 算术决定**；把 CPU 拿掉（捕获）后，
LLM stage 从 22.37 掉到 20.31 ms（−9.2%），同时 INT8 的 4.77 ms 才显现出来。

**既有旁证早就在文档里**：杠杆 #1 记的 **DiT eager 115.0 → 捕获 58.4 ms（1.97×）**。
DiT 已经告诉我们这一层是 launch 受限的；**backbone 从来没被捕获过**，
所以它 66 ms 里同样有一大块是提交开销 —— 只是直到这次没人量过。
§6.1 的 roofline 说 backbone 的 GPU 工作只有 ~36 ms，而墙钟是 66 ms；
§6.6.3 当时把这 30 ms 归给"ViT 的 RoPE 与 gate·up 走 torch shim"，
**方向对，但真正的量级来自 CPU dispatch，不是 GPU 时间**。

**一个意外副产品**：那次**写错的**捕获（用 `stream=0`，pybind kernel 全落在图外，
只捕到 torch 算子）**replay 只要 2.72 ms**。也就是说 LLM stage 里
`_rope_rotate_half` + `mul_` 这些 torch shim 的**GPU 时间**就有 ~2.7 ms
（杠杆 #9 的目标）；而它们的 **CPU dispatch 成本更值得怀疑**（每层 14 个 torch 算子）。

**三个量测陷阱（都踩过，都必须记）**：

1. **`_run_kernel_backbone` 末尾自带 `torch.cuda.synchronize()`。**
   不临时屏蔽它就量"CPU 提交时间"，量到的其实是墙钟，会得到
   **CPU/wall = 100%** 这个看着惊人、实为同义反复的结论（第一次就中招）。
2. **n>1 连发会把驱动的 launch 队列（~1024）打满，`cudaLaunchKernel` 随之阻塞**，
   于是量到的是 `max(CPU, GPU)` 而不是 CPU。**必须 n=1**，且 sync 放在计时之外。
3. **捕获必须显式把 capture stream 传进管线**（`stream=s.cuda_stream`）。
   用 `stream=0` 捕获得到**空图/半空图，而且不报错**：第一次量出
   "LLM stage 捕获后 2.72 ms"（对 22 ms 的阶段），差点当成 8× 收益写进文档。
   `torch.cuda.graph()` 会切当前流，但本管线的 kernel 用的是**传进去的 stream 参数**，
   两者不是同一个东西。

#### 6.11.6 判定与产出

- **杠杆 #4 关闭（不落地）**：精度不过 `THR_FUSED_CONSUMED`（**0.992077 < 0.995**），
  且 eager 收益只有 0.6–1.0 ms（边界的 **0.6–0.9%**），代价是 +0.60 GB 常驻权重
  和一个新精度档。**放宽门去换 0.6% 正是 AGENTS.md §6"smoke floors are
  load-bearing"要拦的事**，而 §6.10.2 刚写下"只看 G4 会误判"。
  代码已完整回退（pipeline / frontend / 两个测试文件），本节保留全部量测与机理。
- 这是原则 #11(b) 的标准结局，但比它更严一档：**不是"修复成本>收益 ⇒ 默认关"，
  而是"连门都没过 ⇒ 不落地"**。两条理由都记：精度门是主因，ROI 不足是次要原因。
- **捕获之后杠杆 #4 也救不回来**：4.77 ms 的 GPU 收益会显现，
  但 `backbone_features` 0.992077 与是否捕获**无关**。
  ⇒ **两件事独立：先做捕获（零精度让步），杠杆 #4 不因捕获而复活。**
- **新杠杆 #11：backbone CUDA-graph 捕获** —— 当时判为 ROI 最高项，
  **已实现并实测，结论是不落地**（§6.13）。
  下界估算的依据（"整个 backbone 的 CPU 提交 60.00 ms / 墙钟 65.71 ms（92.1%）"）
  **是错的量法**：捕获后 CPU 提交确实掉到 0.46 ms，墙钟只掉 **4.54 ms（1.078×）**，
  而一次性捕获成本 262.1 ms ⇒ **回本需 57.7 帧**，现契约一实例只跑一帧。
  ✅ "这不是新发明，是补齐 Orin 因继承链漏掉的能力"（红线 #2：先 grep）—— 这句仍然成立，
  Thor 的实现确实可以直接抄，§6.12.3 的七项前置改造也确实全部做完了；
  **只是抄过来之后量出它在这块硅的这个阶段不值钱**。
  前置重构（常驻 runtime、hoist attn backend、mask→`index_copy_`）**已单独落地并留下**，
  值 2.35 ms（§6.13.1）。

**实验件**：`/tmp/llm_int8_ubench.py`、`/tmp/llm_ffn_realdata.py`、
`/tmp/llm_decomp.py`、`/tmp/llm_graph_bisect.py`、`/tmp/cpu_launch_cost.py`、
`/tmp/backbone_launch_bound.py`，均未进仓（方法与数字已完整记录在本节）。

### 6.12 当前分段耗时实测 + CUDA Graph 的跨平台现状（杠杆 #11 的依据）

> **结局（§6.13）**：本节列的七项前置改造**全部做完并落地**（杠杆 #12，−2.35 ms），
> 捕获本身也做出来并验证**逐位相同**，但实测只值 **4.54 ms（1.078×）**、
> 回本需 **57.7 帧**，在"一实例一帧"的现契约下是净亏 ⇒ **捕获不落地**。
> 本节的跨平台对照（6.12.2）与 Thor 的四条设计（6.12.3）仍然是有效的参考资料：
> 它们是对的，只是这块硅的这个阶段不吃这一套。

#### 6.12.1 当前分段耗时（融合 + INT8 DiT 出厂默认档，锁频，真机 frame 0）

> ⚠️ **本表是杠杆 #12 重构前的快照，已被 §6.13.1 的表取代**（边界 107.40 → **105.26 ms**，
> 3.36× → **3.42×**；融合余量 4.21 → **2.16 ms**）。保留它是为了给出重构前的逐项对照。
> 末行"纯 CPU 提交时间 = 墙钟的 92.1%"**不是** launch 受限份额，见 §6.13.3。

§6.6.3 的 backbone 分解是**融合前**的快照（18 层 ViT、bf16 DiT、wall 58.49 ms），
已被本表取代。逐阶段用 CUDA event 对包住管线函数实测，中位数 of 7：

| 阶段 | ms | 占边界 | CUDA Graph |
|---|---|---|---|
| ViT（24 层，融合后跑满） | **31.18** | 29.0% | ❌ eager |
| **DiT ×4 步（graph replay）** | **33.08** | **30.8%** | ✅ **已捕获** |
| LLM（16 层） | **22.52** | 21.0% | ❌ eager |
| action encoder + decoder | 8.61 | 8.0% | ❌ eager |
| VLSA（4 层） | 6.47 | 6.0% | ❌ eager |
| 融合余量（conv3d patch embed / pos-embed 插值 / final merger / `index_copy_` scatter / ~30 次 `buf()` 分配 / 每次重建 attn backend） | 4.21 | 3.9% | ❌ eager |
| DeepStack mergers | 1.33 | 1.2% | ❌ eager |
| vlln | 0.02 | 0.0% | ❌ eager |
| **边界合计** | **107.40** | 100% | **31% 捕获 / 69% eager** |

自洽性核对：31.18+22.52+6.47+1.33+0.02+4.21 = **65.73** ≈ backbone 墙钟 **65.71 ms**；
65.71+33.08+8.61 = **107.40** ✅。

| 层 | 耗时 | vs HF eager |
|---|---|---|
| 边界（backbone + action head） | **107.40 ms** | **3.36×**（HF 360.38 ms） |
| 完整 `get_action` 等价（+ pre/post ~27.55 ms，仍走 HF processor） | **~135 ms** | **2.87×**（HF 387.93 ms） |
| 一次 backbone 的**纯 CPU 提交时间** | **60.52 ms** | = 墙钟的 **92.1%** |

**顺带修正一个既有条目的口径**：§0.1 / §6.9 里"DiT ×4 = 41.5 ms、1.39–1.42×"
量的是**整个 action head**（含 8.61 ms 的 encoder/decoder，这部分不随档位变）。
剥掉它：DiT 本体 **bf16 ≈ 49.6 ms → INT8 33.08 ms = 1.506×**
（bf16 侧由配对 A/B 的差值 16.54 ms 反推）。
⇒ **原报的 1.39–1.42× 是被稀释过的保守值**，不需要改结论，但口径要写清。

**⚠️ pre/post 的 ~27.55 ms 现在是完整 `get_action` 里最大的非 GPU 项（占 20%）**，
且完全没被本通路碰过（仍走 HF processor：图像 resize/normalize/im2col +
normalize/denormalize）。它和 §8 的"摘掉融合残留的三个 HF 依赖"是同一件事的两面。
> ⚠️ **本段已被 §6.23 部分兑现、并纠正了一个数字**：其中最大的一段
> （每观测图像通路）**已替换成纯 torch GPU 链**，实测 **12.916–14.027 → 1.448–1.485 ms**；
> 而 "~27.55" 这个总量本身是**两个独立量测相减**的产物，不可当账面用。
> 残留的每观测 pre/post 只剩 GPU 图像链 1.448–1.485 + `decode_action` 0.24
> **≈ 1.7–1.8 ms**（`_apply_vlm_processing`+tokenizer 那 2.74 ms 是 prompt-scoped，
> §6.15 已提进 `set_prompt`）。仍走 HF 的只剩 `visual_pos_masks` 与
> `rope_cos`/`rope_sin`，后者**被 `mrope_table.py` 阻塞**（§6.23.7 / §8）。

#### 6.12.2 CUDA Graph 的跨平台现状：**Thor 早就做了全图捕获，Orin 是唯一漏的**

| 通路 | backbone 图 | DiT 图 |
|---|---|---|
| **Thor N1.7** FP8 / FP16 / FP4 | ✅ **ViT→DeepStack→LLM→VLSA 一张图** | ✅ |
| **Thor N1.6**（`groot_thor.py`） | ✅ 更细，分阶段多张图：`_siglip_graph` / `_qwen3_graph` / `_qwen3_torch_graph` / `_dit_graph` | ✅ |
| RTX SM120 FP8 | ✅ 同一个 `_GrootN17FP8BackboneMixin` | ✅ |
| RTX FP16 / SM89 FP16 | ❌ | ✅ |
| AMD CDNA3/4 | ❌（backbone eager） | ✅ |
| **Orin SM87** | ❌ | ✅ |

Thor 的调用序列（`groot_n17_thor_fp8.py:202-204`）：

```python
self._run_kernel_backbone(aux)      # eager 跑一次，顺便 stash 可捕获闭包
self._capture_backbone_graph()      # 捕获整条 backbone
self._backbone_features = self.run_backbone_graph(aux).clone().half()
```

其注释原话：*"captured as a single CUDA graph so the per-observation hot path is
one graph replay with **zero Python launch overhead**"* —— 与 §6.11.5 量到的
"CPU 提交占墙钟 92.1%"正是同一件事的两种说法。

**Orin 为什么漏了**：继承链
`GrootN17TorchFrontendOrin → RtxFP16 → Rtx → Thor`（**基类**），
而 `_capture_backbone_graph` / `_kbb_forward` / `run_backbone_graph` 定义在
**`ThorFP8`** 里（FP16/FP4 因为是它的子类才白拿）。
⇒ Orin 从基类拿到了 DiT 图，**backbone 图从来没进过这条链**。
这不是设计取舍，是继承结构的副作用。

#### 6.12.3 Thor 的实现可以直接抄（红线 #2）

四条关键设计：

1. **`_run_kernel_backbone` 末尾 stash 一个 `_kbb_forward(stream)` 闭包**，
   跑在**常驻 buffer** 上。注释原话："no Python dict rebuild, no torch input prep"。
2. **每次观测把新输入 `copy_` 进常驻 buffer 再 replay**
   （`_kbb_pv.copy_(...)` / `_kbb_llm_h.copy_(...)`），图里没有 host 依赖。
3. **boolean-mask 赋值换成固定 `index_copy`**，因为掩码不可捕获。
   ✅ **Orin 的融合用的正好就是 `index_copy_`（`llm_h.index_copy_(0, self._fus_vis_idx, mg_img)`），
   这个坑天然躲过了。**
4. **一个可变 stream cell（`scell[0] = s`）**，让同一个闭包既能在 capture stream 上跑、
   replay 时又落回默认流；侧流上 warmup 3 次再 capture。

**Orin 还要多处理 Thor 没有的几处**：

| 项 | 处理 | 结果（§6.13） |
|---|---|---|
| 每次调用重建的 `OrinGrootN17BackboneAttn(...)` | 提为常驻（`_build_dit_attn` 是先例） | ✅ 已做，进 `_build_backbone_runtime()` |
| ~30 个 `buf()` 分配 | 提为常驻（`_allocate_infer_buffers` 是先例） | ✅ 已做；`keep` 列表保活，匿名中间 buffer 也在内 |
| `_fus_pv` / `_mrope_cos,sin` per-prompt 常量 | 走 Thor 的"copy 进常驻 buffer"那条路 | ✅ 改成 `pv` 常驻 buffer + identity/`_version` 快路径；RoPE 表按引用烘焙（`set_prompt` 拒绝二次调用，故不可能变） |
| `_fus_vis_idx` | ⚠️ **若形状随 prompt 变就不能进图**（Thor 的 `index_copy` 同样要求固定索引） | ✅ 由 runtime 从 `_visual_pos_masks` 自建常驻 `vis_idx`，**两种融合模式都有**（原来只有 fused 模式有）；形状变化 ⇒ key 指纹不符 ⇒ raise |
| `F.conv3d` patch embed | ⚠️ 首次调用会 cuDNN "Plan failed … NOT_SUPPORTED" 回落；**必须确认回落后算法稳定可捕获**，否则留在图外 | ✅ 单独验过：**可捕获且 replay 后逐位相同**。副产品：回落算法比 roofline 慢 ~23×（~1.25 ms GPU） |
| 末尾的 `torch.cuda.synchronize()` 与 `snap()` 的 D2H clone | **必须在捕获区之外** | ✅ 都移到 `_run_kernel_backbone`；`_kbb_forward` 内零 sync、零回读 |
| 捕获后 | ⚠️ **必须重跑 stale-value 门**（换一帧输入后 replay 要跟着变） | ✅ 探针里做了（`max&#124;d&#124; = 0` 对同帧；换观测必须变，已在两种融合模式上验）；但**捕获本身不落地**，所以这道门没有进仓 |

**实验件**：`/tmp/stage_breakdown.py`（未进仓；逐阶段 event 插桩 + CPU 提交量测，
方法与 §6.11.5 的三个陷阱规避一致）。


### 6.13 杠杆 #11 落地实测：重构留下，**捕获不落地**，并纠正 §6.11.5 的机理判断

按 §6.12.3 的清单分两步做。**Step 1（常驻 runtime 重构）已落地**；
**Step 2（整条 backbone 捕获）做完探针后判定不落地**。两步的数字都在下面，
但本节最重要的产出是 **6.13.3 的机理纠正**：§6.11.5 从"CPU 提交占墙钟 91.2%"
推出"backbone 是 CPU 提交受限"，**这个推论是错的**，而杠杆 #11 的预期收益
（"数毫秒到十几毫秒"）正是建立在这个错推论上的。

#### 6.13.1 Step 1（已落地）：常驻 runtime + 纯 kernel forward

`_run_kernel_backbone` 拆成三块（`groot_n17_orin.py`）：

| 新方法 | 职责 | 每次调用的成本 |
|---|---|---|
| `_build_backbone_runtime()` | 常驻 buffer（~30 个）、`OrinGrootN17BackboneAttn`、5 个 kernel 参数字典，**全部只建一次** | 0（形状指纹不符则 raise） |
| `_kbb_load_inputs(aux)` | 把**本次观测**拷进常驻输入 buffer（`pv`，或 unfused 的 `pf`+`llm_in`） | 同一 source 对象且未原地改动 ⇒ **跳过** |
| `_kbb_forward(stream, snap_into, vit_layers)` | 纯 kernel 链：不分配、不重建字典、不回读 host、不 sync | — |

顺带消掉的三件每次都发生的事：

1. **3 次 host 同步**：DeepStack inject 原来是 `ib[mask] = ds_out[j]`（布尔掩码赋值 ⇒
   内部 `nonzero()` ⇒ 与 host 同步），3 个 merger 就是 3 次流水线排空。
   改成 `ib.zero_()` + `ib.index_copy_(0, vis_idx, ...)`，**实测逐位相同**
   （`torch.equal` 在三个真机 inject buffer 上都为 True）。这也正是
   §6.12.3 第 3 条 Thor 早就做过的替换。
2. **每次重建 attn backend**（分配 LLM K/V 槽与 LSE slab）。
3. **每帧一次的 H2D + bf16 转换**：`aux["pixel_values"]` 从 host 搬到 device
   再转 bf16，实测 **~1.5 ms**。`set_prompt` 原来只做一次，重构后若不加
   identity+`_version` 快路径就变成每帧都做 ⇒ 已加（复用
   `_GrootN17FP8BackboneMixin._validate_backbone_graph_contract` 的同一套快路径，
   红线 #2），新对象或原地改动仍会重载。

**实测（同一口径 `/tmp/stage_breakdown.py`，锁频 1300.5 MHz，真机 frame 0）**：

| 量 | 重构前 | 重构后 | Δ |
|---|---|---|---|
| backbone 墙钟 | 65.71 ms | **63.36 ms** | **−2.35 ms（−3.6%）** |
| 一次 backbone 的纯 CPU 提交 | 60.00–60.52 ms | **52.80 ms** | **−7.2 ms（−12%）** |
| **边界合计** | 107.40 ms | **105.26 ms** | −2.14 ms |
| 边界 vs HF eager | 3.36× | **3.42×** | |
| 完整 `get_action` 等价 | ~135 ms / 2.87× | **~133 ms / 2.92×** | |
| 融合余量（conv3d + merger + 分配 + attn 构造 + inject） | 4.21 ms | **2.16 ms** | **−49%** |

逐阶段（中位数 of 7）：ViT **31.71** / DiT replay **33.33** / LLM **21.77** /
enc+dec **8.57** / VLSA **6.37** / 融合余量 **2.16** / DeepStack **1.32** /
vlln **0.02**。
噪声底：同一脚本 5 轮的极差 **±0.06 ms**（探针里 eager 62.44–62.50、
replay 57.90–57.95），所以 −2.35 ms 是噪声的 ~39 倍，不是漂移。

**精度：与落地前逐位一致**，证据不是"过了门"而是**与文档自己记过的数字逐位对上**：
`vit_block_17` **0.998156**（=§6.6.1/§6.7.1 表）、`deepstack_out_0/1/2`
**0.999960 / 0.999687 / 0.999224**（=§6.6.1）、`llm_h` **0.999975**（=§6.7.2）、
`backbone_features` **0.996951**（=§0.1/§6.7.2/§6.11.3 的 bf16 臂）。
外加：连续 3 次调用逐位相同；`_backbone_features` 不被后续调用改写
（`.to(_BF16)` 在 bf16 上是 no-op，会让它 alias 常驻 buffer ⇒ 已改 `.clone()`，
与 Thor 的 `run_backbone_graph(aux).clone().half()` 同一理由）。
**测试 31 → 36 passed。**

**⚠️ 重构过程中真踩到的一个 bug（31 个 GPU 测试全绿也没拦住）**：
unfused 模式把 `aux["llm_input_embeds"]` 直接载进了常驻的 `llm_h`，
而 `llm_h` 是 LLM 的残差流、会被 16 层原地覆写 ⇒ **第二次观测会从第一次的
layer-15 输出起跑**。不报错、cos 也"看着合理"，只是描述的是错误输入。
根因是**GPU 侧没有任何测试构造 `fuse_image_embeds=False` 的前端**。
修法：unfused 独立 `llm_in` buffer，forward 里每轮 `llm_h.copy_(llm_in)` 重新播种。
钉法：**5 个 CPU 契约测试**（`tests/test_orin_groot_n17_dispatch.py` 末节，
用 `object.__new__` + 假 runtime，无需 GPU/checkpoint），覆盖
"载入不得写残差流"、快路径两个方向、缺 key 必须 raise。
⇒ **与 AGENTS.md §1.1 第 6 条的 repeat-identical 门同一类**：
常驻 buffer 一旦兼作输入与工作区，只有"跑两遍比对"能发现。

#### 6.13.2 Step 2（探针完成，**不落地**）：整条 backbone 捕获

**可行性没问题**（这一步的价值是把设计风险彻底清零）：

- `F.conv3d` patch embed 在 SM87 上**可捕获且 replay 后逐位相同**
  （`torch.cuda.graph()` 1255.4 µs / 手工 `capture_begin|end` 1248.7 µs）。
  顺带量到：cuDNN "Plan failed … NOT_SUPPORTED" 回落后的算法
  **比它的 roofline（55 µs）慢 ~23×**（~1.25 ms GPU）。
- 侧流 warmup 3 次 + 显式传 `stream=s.cuda_stream` 捕获（§6.11.5 陷阱 3），
  **replay 与 eager 逐位相同，max&#124;d&#124; = 0**。
- 三个陷阱都躲过了：sync 与 `snap()` 的 D2H 在捕获区外；
  mask 赋值已换成 `index_copy_`；per-prompt 常量走"引用常驻对象"。

**但收益只有 4.54 ms（配对交替 A/B，中位数 of 5×7）**：

| 臂 | 墙钟 | 纯 CPU 提交 |
|---|---|---|
| eager | **62.47 ms**（62.44–62.50） | **51.97 ms** |
| graph replay | **57.93 ms**（57.90–57.95） | **0.46 ms（eager 的 0.9%）** |
| Δ | **−4.54 ms（1.078×）** | −51.5 ms |

一次性成本：warmup 198.7 ms + capture 63.4 ms = **262.1 ms**
⇒ **回本需要 57.7 帧观测**。而 `set_prompt` 明确拒绝第二次调用
（一个实例 = 一个 prompt + 一帧），**在出厂契约下捕获是净亏**。

**判定：不落地。** 代码不留半截（不留 `_capture_backbone_graph` 死方法、
不留 `use_backbone_graph` 开关）—— 一个在现契约下亏本、
且要复活必须先有另一个特性的开关，正是"不要为假想需求设计"要拦的东西。
Step 1 的重构**保留**：它自己有 2.35 ms 的正收益，且是任何未来捕获的硬前置。

**复活条件（写清，免得下次重新推一遍）**：只有当出现**每观测入口**
（RTX FP8 的 `infer(aux=...)` 那条路：载入新观测 → replay → 刷新
`_backbone_features`）时，捕获才开始赚钱，且 58 帧之后每帧净省 4.54 ms。
⚠️ 那条路在 RTX FP8 里**本身还不完整**：它刷新了 `_backbone_features`，
却没有任何地方失效 `_dit_cross_K/_V`（`infer` 用 `hasattr` 缓存），
而 `_precompute_dit_cross_kv` 是从 `_backbone_features` 算出来的
⇒ **action head 会继续用上一帧的 cross-KV**。要做每观测入口，
必须连这个一起修（并加"换一帧后解码动作必须变"的 stale-value 门）。

> ✅ **两条复活条件均已在 §6.15 满足 ⇒ 本杠杆已在 §6.17 落地。**
> 落地实测：backbone **63.52 → 59.52 ms（−4.00 ms，1.0673×）**、每观测
> **115.44 → 111.40 ms（−4.03 ms）**，`backbone_features` 与解码动作**逐位相同**，
> 一次性 259.7 ms ⇒ **64.9 帧回本**。本节那个 4.54 ms 是**裸 `_kbb_forward` 的口径**，
> 部署能拿到的是 4.00 ms，差在契约校验 + 输入载入不被图吃掉（§6.17 有解释）。
> 每观测入口已交付；cross-KV 的失效不是靠 `delattr` 而是**原地刷新**
> （`_refresh_dit_cross_kv` 写进图已捕获的那些槽），并由 §6.15.3 的层 1
> 负控制证明它真的写了。stale-value 门也加了 —— 但同时测出
> **"换一帧后解码动作必须变"这个口径本身太弱**（连续帧只差 0.895°），
> 真正能门的是槽内容对应有 K 的 cos。**上面那段警告仍然对 RTX FP8 成立**：
> 那条路至今没有 cross-KV 失效，属于它自己的待修项（§8）。
> 捕获本身的经济学：一次性 262.1 ms / 每帧 4.54 ms ⇒ **58 帧回本**，
> 连续部署跑千帧量级 ⇒ 预期每观测 117.56 → ~113 ms。

#### 6.13.3 机理纠正：**"CPU 提交 / 墙钟"不是 launch 受限比例**

§6.11.5 写下"纯 CPU 提交 60.00 ms / 墙钟 65.77 ms（91.2%）⇒ backbone 是
CPU 提交受限"。**这个推论错了**，而且它一路污染到 §6.2 杠杆 #11 的估值
（"数毫秒到十几毫秒"）、§6.6.3 的 30 ms 缺口归因、和杠杆 #9 的升值。

错在哪：**CPU 提交与 GPU 执行是重叠的**。那个比值量的是
"墙钟里 CPU 有多忙"，不是"墙钟里有多少是 CPU 造成的"。
launch 受限比例只有一个正确量法：

> **launch 受限份额 = (eager 墙钟 − replay 墙钟) / eager 墙钟**

代入实测：(62.47 − 57.93) / 62.47 = **7.3%**，不是 91.2%。
GPU kernel 地板 ≈ **57.93 ms**（replay 把所有 kernel 一次性提交，
间隙最小），eager 只有 4.54 ms 藏不住。

**为什么 DiT 与 backbone 差这么多**（原则 #16：瓶颈要按 kernel 粒度分类，
不能跨阶段外推）：

| | kernel 粒度 | 每次发射的 CPU | 每次发射的 GPU | 谁卡住 |
|---|---|---|---|---|
| DiT（M=41，32 blocks × 4 步） | ~1900 个 **~9.4 µs** 的小 kernel | ~21 µs | ~9.4 µs | **CPU**（⇒ 捕获 1.97×） |
| backbone（M=141/512） | 数百个 **27–400 µs** 的 GEMM/FA2 | ~21 µs | 27–400 µs | **GPU**（⇒ 捕获 1.078×） |

DiT 的 1.97× 是因为它每个 kernel 的 GPU 时间**小于**发射它的 CPU 时间；
backbone 的 GEMM 普遍比 21 µs 长，CPU 完全藏在 GPU 后面。
**"DiT 捕获赚 1.97×，所以 backbone 也会赚"是错的类比**，
而 §6.11.6 当时已经写了"不得照搬 DiT 的比值"—— 这句自我提醒是对的，
只是没人料到底下界是 1.078×。

**杠杆 #4 的旧账因此要重算**（§6.11.4 的"GPU 省 3.93 ms、eager 只省 0.6–1.0 ms"）：
当时的解释是"被 CPU 提交吃掉"，方向不对。用两次捕获量测反推：

| 臂 | eager 墙钟 | 捕获后墙钟 | 藏不住的 launch 成本 |
|---|---|---|---|
| bf16 | 62.47 | 57.93 | **4.54 ms** |
| INT8 FFN | ~61.7（= 62.47 − 0.8） | ~53.2（= 57.93 − 4.77） | **~8.5 ms** |

⇒ INT8 臂的 GPU 收益是真的（捕获后 4.77 ms 显现），
但它**自己多带了 ~3.9 ms 藏不住的发射成本**（32 个额外 quantize kernel，
每个 GPU 时间很短 ⇒ GPU 在它们之间饿着）。
**总 CPU 提交时间两臂几乎相同（59.87 vs 60.00 ms），差别在"能不能重叠"，
不在"提交了多少"** —— 这是同一个纠正的另一面。
⚠️ 这张表的两个臂来自不同量测场次（口径略有差异），是**算术重构**而非配对实测，
只用于说明量级；杠杆 #4 的否决理由是精度（0.992077 < 0.995），不依赖这张表。

#### 6.13.4 判定与产出

- **杠杆 #11 关闭（不落地）**：可行、逐位相同、但只值 **4.54 ms（1.078×）**，
  回本需 **57.7 帧**，而现契约一实例只跑一帧。原则 #11(b) 的标准结局：
  量出 ROI ⇒ 记档 ⇒ 默认关 ⇒ 往下走。
- **Step 1 落地**：−2.35 ms backbone / −2.14 ms 边界，3.36× → **3.42×**，
  精度逐位不变，测试 31 → **36**。它是纯收益，且是未来捕获的硬前置。
- **方法论产出（本节最值钱的一条）**：
  **"CPU 提交时间 / 墙钟"不能用来判断 launch 受限**；
  必须捕获一次、用 `(eager − replay)/eager` 量。
  这次差点照着 91.2% 去写一整条捕获通路 + 每观测入口 + 配套门禁。
- 连带纠正：§6.6.3 把 backbone 的 30 ms roofline 缺口归给
  "ViT 的 RoPE / gate·up 走 torch shim" —— §6.11.5 说"方向对但量级来自 CPU dispatch"，
  **现在看两者都不是**：GPU 地板 57.93 ms vs §6.1 roofline 的 ~36 ms，
  缺口是**真实 GPU 时间**（torch shim 的 GPU 开销 + kernel 效率），
  不是发射开销。杠杆 #9 因此**回到"只值 GPU 时间"的原估值**，
  §6.11.5 给它的升值要撤回（旁证：§6.11.5 自己量到那些 torch shim 的
  GPU 时间 ~2.7 ms，与"捕获只省 4.54 ms"是自洽的）。

**实验件**：`/tmp/bb_graph_probe.py`（整条 backbone 的捕获探针：
一次性成本、逐位比对、配对交替 A/B、CPU 提交、回本帧数）、
`/tmp/conv3d_capturable.py`、`/tmp/verify_refactor_identical.py`
（重构的逐位一致性 + 两种融合模式 + 快路径双向），均未进仓。


### 6.14 DiT 的 K/V 到底有没有被量化（使用方指定的经验教训，已审计 + 三臂实测）

使用方的要求是"**DiT 的 KV Cache 尽量别量化**"。先审计，再实测，最后改默认档。

#### 6.14.1 审计：这里有两个不同的"K/V"，只有一个曾被量化

| 对象 | 是什么 | INT8 档下的状态 |
|---|---|---|
| **cross-attention 的 K/V 缓存** | `_precompute_dit_cross_kv()` 每个 prompt 算一次，被 **16 个 cross block × 4 步**反复读 | **从来没有被量化过**：投影在 **fp32** 里做（`kv_src.float() @ k_w + k_b`，`k_w` 取的是 `self._dit_k_w`，不是 `_w8`），存成 **bf16**，FA2 也按 bf16 消费。cross block 在 `dit_forward` 里**根本不做 K/V GEMM**，只 `attn.run("dit_cross", ...)` 读槽位 |
| **self-attention 的 K/V 投影** | 16 个 self block 每步现算，M=41、N=K=1536，共 32 个 GEMM | **曾是 INT8**（`k_w8`/`v_w8`），本节处理的就是它 |

⇒ **"KV Cache 不量化"这条出厂就已满足**；需要决策的是第二个对象。
另外核实：`_precompute_dit_cross_kv` 末尾会 `del _dit_attn / _dit_graphs`，
所以 Orin 这条路在 backbone 特征刷新后**会**重建 cross-KV 与 DiT 图
（RTX FP8 的 mixin 没有这一步，见 §6.13.2 的复活条件）。

#### 6.14.2 三臂实测（真机 frame 0/300，锁频，同一 fixture）

为了能按"族"切换档位，`dit_forward` 新增 `weights["bf16_families"]`
（前端 `dit_bf16_families=`），把原来的 `qz/mm/W` 三件套换成一个按族查表的
`mmq()`。**逐族校验是双向的**：被豁免的族不得带 `_w8`，未豁免的族必须带 ——
两个方向都 raise，绝不猜某个 GEMM 跑在哪个档（红线 #4）。

| 臂 | 配置 | f0 decoded max&#124;d&#124; | f300 decoded max&#124;d&#124; | f0 velocity cos | f300 velocity cos | decoded cos（最差） | DiT ×4 |
|---|---|---|---|---|---|---|---|
| **A** | INT8，六族全量化（原出厂档） | 0.008613 rad = **0.4935°** | 0.006345 rad = **0.3636°** | 0.999512010 | 0.999573796 | 0.999999291 | **33.23 ms** |
| **B** | INT8，**k/v 留 bf16** | 0.008613 rad = **0.4935°** | 0.003386 rad = **0.1940°** | 0.999501654 | 0.999591404 | 0.999999391 | **34.23 ms** |
| **C** | 全 bf16（参考档） | 0.008613 rad = **0.4935°** | 0.003260 rad = **0.1868°** | 0.999821863 | 0.999899045 | 0.999999544 | （≈49.6 ms，§6.12.1 反推） |

延迟口径：A/B 两臂**配对交替**、各 5×11 取中位数，只放两个前端常驻（极差
A 33.08–33.33 / B 34.22–34.50 ⇒ ±0.15 ms，远小于 1.00 ms 的差）。

#### 6.14.3 两个从这张表里读出来的结论

1. **frame 0 的 0.4935° 与 DiT 档位无关** —— 三臂**逐位同值**（0.008613 rad）。
   ⇒ 这个最差误差来自 **bf16 backbone 本身**（`backbone_features` 0.996951 那一级），
   不是 INT8 的代价。此前 §0.1/§6.9 把 0.493° 记成"INT8 档的代价"，
   **归因错了**：INT8 档的真实代价只体现在 f300（0.3636° vs bf16 的 0.1868°）
   和 velocity cos（0.9995 vs 0.9998）。
2. **豁免 k/v 把 f300 的误差从 0.3636° 拉回 0.1940°**，几乎贴上 bf16 的 0.1868°
   （差距从 1.95× 缩到 1.04×），而 f0 不动（因为 f0 的误差本来不在 DiT）。
   decoded cos 也略好（0.999999291 → 0.999999391）；velocity floor 略差
   （0.999512 → 0.999502），两者都远高于 0.999 的门 ⇒ **精度是等或更好**。

**成本为什么这么小**：豁免 k/v 只把 **160 个 GEMM 里的 32 个**挪回 bf16，
而且**一个 quantize 都不省也一个都不多** —— 因为 adaLN 之后的激活是
q/k/v **共用**的，q 仍在 INT8，那次 quantize 就还得做。
这两条计数已由 CPU 契约测试钉住
（`test_int8_dit_exempt_families_really_run_bf16`：128 个 int8 GEMM /
32 个 `bf16_nn` / 128 个 quantize，与六族全量化时的 160/0/128 对照）。

⚠️ 写这段时踩到的一个真 bug：`mmq` 最初按**整档**分支定义
（`if use_int8:` 里定义 int8 版、`else:` 里定义 bf16 版），
于是 INT8 档下**所有**站点都走 int8，`bf16_families` 被静默忽略，
最后在第一个 self 层因为 `k_w8` 不存在而 KeyError。
**这正是"双向校验"要拦的那类错**，而它是在 A/B 脚本里才暴露的 ——
所以那个计数测试是必须的，不是装饰。

#### 6.14.4 判定：**k/v 豁免成为出厂默认档**

- 依据：使用方的经验教训（KV 不量化）+ 上表（精度等或更好，尤其 f300 的
  decoded 误差几乎回到 bf16 水平）。
- **代价（两把尺都记）**：
  DiT ×4 **+1.00 ms**（33.23 → 34.23，配对交替）；
  出厂延迟测试口径下 action head **+1.83 ms**（41.75 → 43.58），
  边界 **104.79 → 106.60 ms**，**3.44× → 3.38×**；
  配对交替的 DiT 档比值 **1.398× → 1.337×**（仍远高于 1.15× 的门）。
- 退回六族全量化：`dit_bf16_families=()`。退回全 bf16：`use_int8_dit=False`。
  三档可同进程共存，便于配对 A/B。
- 默认值写成 `None` 哨兵而非 `("k","v")`：否则 `use_int8_dit=False`
  会撞上"bf16 档还传豁免表"的校验而**构造失败**（第一次就这么写坏了 6 个测试）。
  `None` = "按本档的出厂豁免表"，类常量 `_DIT_BF16_FAMILIES` 是唯一事实来源，
  并由 `test_int8_dit_is_the_shipped_default` 钉住。
- **顺带的独立收获**：A 臂（未改行为的对照）复现了文档早先记下的每一个数字
  —— velocity floor 0.999512（bf16 0.999822）、f0 0.493°、f300 0.364°。
  这独立证明了 §6.14.2 里对 `dit_forward` 的按族重构**数值上与重构前一致**。
- 内存副作用：不再量化 k/v，少驻 ~100 MB int8 权重（§8 的冗余权重项）。

**实验件**：`/tmp/dit_kv_bf16_ab.py`（三臂精度 + A/B 配对延迟，未进仓）。
⚠️ 第一版一次性建 6 个前端（3 臂 × 2 帧）把 CUDA caching allocator 打到
`NVML_SUCCESS == r INTERNAL ASSERT FAILED`；改成**最多两个常驻**才跑通。


---

### 6.15 每观测入口（`infer(aux=...)`）：连续推理落地，并测出"外层 cos 门不住 stale KV"

**杠杆 #13，已交付。** §6.13.2 把 backbone 捕获判为净亏的唯一理由是
"`set_prompt` 一实例只跑一帧"；这一节把那个契约换掉了。

#### 6.15.1 契约变化与三处必须原地写的状态

出厂契约原是**一个前端 = 一个 prompt + 一帧**（`set_prompt` 第二次调用直接 raise）。
现在 `infer(state, aux=...)` 让**一个前端服务一整条观测流**：
校验契约 → 重跑 backbone → 刷新 cross-KV → 复用已捕获的 DiT 图。
不传 `aux` 时行为与从前逐位相同（C4 钉住）。

沿用 RTX FP8 mixin 的 `infer(aux=...)` + `_snapshot/_validate_backbone_graph_contract`
先例，但 Orin 有三处不同，都是"重分配 vs 原地写"的取舍：

| 状态 | 谁持有 | 换观测时 | 为什么 |
|---|---|---|---|
| backbone 输入槽（`pv` / `pf`+`llm_in`） | 常驻 runtime（杠杆 #12） | `copy_`，且**按 identity+`_version` 跳过未变的源** | 重分配会让 runtime 的 key 失效 |
| `_backbone_features` | 每次 `.clone()` | 重新赋值 | backbone 返回的是常驻 `vlsa_h` 的**视图**，不 clone 就被下一帧原地覆写 |
| **DiT cross-K/V** | `attn.dit_cross_K/V[j]` 槽 | **`_kv_slot_copy` 原地写** | 🔴 四个 DiT 图捕获的是这些槽的 `data_ptr()`；改成重分配 ⇒ 图**照样 replay 成功**，但读的是上一帧的 KV |

第三行是本节的全部风险所在，也是 `_precompute_dit_cross_kv`（一次性路径，
**必须**失效 `_dit_attn`/`_dit_graphs`）与 `_refresh_dit_cross_kv`（每观测路径，
**必须不**失效）不能互换的原因。两者共用 `_project_dit_cross_kv`，
所以 fp32-from-bf16 的投影数学只有一份，不会漂。

契约校验钉住的是**被图按指针烘进去的元数据**：`grid_thw` /
`visual_pos_masks` / `rope_cos` / `rope_sin`，融合档另加 `input_ids`
（`set_prompt` 把它拷进 `_fus_ids` 后一直按指针读）。
非融合档**不**钉 `input_ids`——那条路根本不消费它，钉了会拒掉合法观测。
稳态路径靠 identity + `_version` 跳过比较：rope 表每帧 ~2.4 MB，
在 Orin 的 ARM CPU 上逐帧比会直接吃进这条路径本来要省的延迟（实测 **0.031 ms**）。

#### 6.15.2 七道门（仿真数据集，Se=148，帧 100–107 连续 + 31000 远端）

出厂默认档 `int8_rowwise(k,v=bf16)`，锁频 1300.5 MHz，
仿真数据集（Se=148）。**七道门已全部进仓**：GPU 侧
`tests/test_orin_groot_n17_continuous.py`（**7 passed**，负控制 C6 因此坐在
验收集里而不是实验脚本里 —— AGENTS.md §3.6），CPU 契约侧
`tests/test_orin_groot_n17_dispatch.py` 末两节。原实验件
`/tmp/continuous_infer_check.py` 未进仓，其内容已被上述两个文件覆盖。

| 门 | 内容 | 实测 |
|---|---|---|
| **C1** 逐帧保真 | 8 个连续帧各自 vs 该帧的 HF 参考 | backbone cos **0.996034–0.997702**；decoded action cos **0.999999232–0.999999628**；max&#124;d&#124; **0.1940–0.3872°** |
| **C2** stale-value | 换一帧动作必须变 | 帧 100 vs 107 不相同，max&#124;d&#124; **0.015625 rad = 0.895°** |
| **C3** repeat-identical | 同一帧跑两遍 | **逐位相同** |
| **C4** 连续 == 一次性 | `infer(aux=帧100)` vs `set_prompt(帧100)+infer()` | **逐位相同**（max&#124;d&#124; = 0） |
| **C5** 图被复用而非重捕获 | 计数 + 对象/指针身份 | 整个前端生命周期 **捕获 1 次**（一次性调用付的），8 帧连续 **0 次**；`_dit_attn` / `_dit_graphs` / 槽 `data_ptr` 集合各 **1 个**；持图 **4** 张 |
| **C6** 负控制 | 故意打断刷新，指标必须可见退化 | 见 §6.15.3 |
| **C7** 契约拒绝 | 四个被烘的元数据各改一个元素 | `visual_pos_masks` / `rope_cos` / `grid_thw` / `input_ids` **全部 raise**，无一被静默服务 |

C1–C4 已进仓为 CPU 契约测试（`tests/test_orin_groot_n17_dispatch.py` 末两节，
**dispatch 21 → 51**，§6.16 的对齐审计再加 5 个 ⇒ **56**）：契约的正/负/重建三路、`_version` 那条快路径的**两个方向**、
`_kv_slot_copy` 只写给定行且拒绝装不下的源、
`_refresh` 保图而 `_precompute` 弃图、`_refresh` 在形状变化时**拒绝而非服务**
（`Skv_text/Skv_image` 经 `_dit_dims` 烘进了图）。

⚠️ **进仓时量到一个测试自身的顺序缺陷（不是前端缺陷）**：`fe` 是
module-scoped 且携带**可变的观测状态**，C3 把前端留在 NEIGHBOUR 上，于是 C4 的
一次性臂拿 **RUNUP 的 state** 去比 **NEIGHBOUR 的 context**，报
max&#124;d&#124; **3.1e-02** —— 正好是 C2 量到的普通帧间差，不是 bug。
修法是把 RUNUP 重新装回去，并顺手把 C4 加强成两臂：
(a) 重载后的 `_backbone_features` 与 `set_prompt` 自己那份**逐位相同**
（证明重载不是第二条数值路径），(b) 同一观测下 `infer()` 与 `infer(aux=)`
**逐位相同**。**通用教训**：共享的 module-scoped fixture 只要携带可变状态，
任何门都变成顺序相关，除非该门自己把前置条件重新装好。

#### 6.15.3 ⚠️ 本节最值钱的发现：**decoded action 的 cos 门不住 stranded KV**

第一版负控制写在**连续帧**上（把帧 100 的 KV 留在槽里，然后要帧 107），
结果 **没有退化**：0.3636° vs 0.2499°（1.45×），cos **0.999998752** —— 
**仍然远高于 0.999 的门**。换成远端帧（31000，另一集）才勉强到 5.8×，
cos 还是 **0.999991673**，**门照样过**。

也就是说：**照 G4 那个口径写的 stale-value 负控制是无效的**，
而 AGENTS.md §3.6 明确要求负控制"必须可见退化，以证明这个测得出来"。
分三层重测才定位到问题不在 bug 而在**量错了张量**：

| 层 | 张量 | 好 | 打断刷新 | 判别力 |
|---|---|---|---|---|
| 0 | 两帧**自己的** HF 参考 | cos(action₃₁₀₀₀, action₁₀₀) = **0.900110**；cos(cross-K) = **0.809373** | — | 上层天花板：参考之间差 1e-1，外层**本可以**动 |
| **1** | **cross-K 槽 vs 该帧应有的 K** | **1.000000000**（vs 31000）/ 0.809373（vs 100） | **0.809373**（vs 31000）/ **1.000000000**（vs 100） | ✅ **完全判别**：打断后槽**精确等于**帧 100 的 K |
| 2 | step-0 velocity vs 该帧 HF | **0.999590913** | **0.994225295** | ✅ 掉 5.4e-3，可门 |
| 3 | decoded action | 0.2129°（cos 0.999999582） | 1.2337°（cos **0.999991673**） | ❌ **过门**，不可用 |

**结论写进方法论**：层 0 说明"两帧的参考动作差 10%"，层 3 却只动 5.8×且 cos 不掉 —— 
⇒ **本 checkpoint 的 decoded action 被 state 输入主导，视觉 context 的贡献只有 ~1°**。
这与原则 #2（生产指标 ≠ 内部精度）同型，但方向相反：
那里是后处理**掩盖**了 >10° 的模型误差，这里是**动作本身对 context 不敏感**，
所以任何外层指标都放大不出 KV 的陈旧。
**stale-value 门必须打在层 1（槽内容 vs 应有内容）或层 2（velocity），不能打在 decoded action。**
C1 的 cos ≥0.999 保留为**保真**门（它本来就为此设计），但**不再兼作** stale 门。

#### 6.15.4 每观测成本分解（中位数 of 7，锁频）

| 段 | ms | 备注 |
|---|---|---|
| 契约校验 | **0.031** | identity+`_version` 快路径；逐帧比 rope 表会贵两个数量级 |
| backbone | **63.51** | = §6.13.1 的杠杆 #12 数字，未被本节改动 |
| **cross-K/V 刷新** | **8.52** | 一次性路径只在 warmup 付一次；连续路径**每帧**付 ⇒ 见 §6.15.5 |
| 其余（state/action encode、4 次 DiT replay、decode、D2H） | **43.65** | |
| **`infer(aux=)` 合计** | **115.70** | |
| **每观测墙钟**（帧 101–107） | **117.56**（min 117.35 / max 117.84） | 首个连续帧 122.69（仍在热） |

两把尺并列（口径规则同 §0.1）：
vs HF 完整 `get_action` **386.27 ms → 3.29×**；
vs HF 同一边界 **358.84 ms → 3.05×**。

#### 6.15.5 顺带落地的一个零风险优化：fp32 权重缓存（−4.58 ms，逐位相同）

刷新从 13.10 ms 起（首次量测），诊断而非猜测：
`_project_dit_cross_kv` 每次调用都对 **32 个 (2048,1536) 权重做 `.float()`** —— 
403 MB 分配 + ~604 MB 流量，实测 **5.10 ms / 12.68 ms = 40%**。
权重永不改变，且 bf16→fp32 是**精确**提升 ⇒ 缓存后 **逐位相同**（实测
`torch.equal` 全 32 项 True）。只提升**偶数层**（`li = 2*j` 从不取奇数）。

| | 刷新 | 每观测墙钟 | vs HF 完整 |
|---|---|---|---|
| 每次重新 `.float()` | 13.10 ms | 122.29 ms | 3.16× |
| **缓存 fp32 权重** | **8.52 ms** | **117.56 ms** | **3.29×** |

代价：常驻 **403 MB**（61 GB 统一内存的 0.7%）。
剩下的 8.52 ms 是 fp32 GEMM 本身（M=20/128、K=2048、N=1536，共 ~14.9 GFLOP）；
改走 bf16 tensor core 曾被记成"会动到 KV cache 的数值、按使用方 KV 不量化的要求未做"——
**这个说法是错的，已在 §6.18 纠正并实测**：两臂输入**逐位相同**（`.float()` 是精确提升），
所以那**不是量化**，只是乘加路径不同；实测存量结果 cos **0.999999997**（≈1 个 bf16 ULP），
而收益是每观测 **5.22 ms**。列为 §8 待办。

#### 6.15.6 对杠杆 #11 的影响：**复活条件已满足**

§6.13.2 的复活条件写的是"需先有每观测入口，且要先修 RTX FP8 那条路漏掉的
cross-KV 失效"。**两条现在都满足了**：每观测入口是本节，
cross-KV 失效由 `_refresh_dit_cross_kv` 原地完成（且 C6 层 1 证明它真的写了）。
⇒ 捕获的经济学反转：一次性 262.1 ms / 每帧 4.54 ms ⇒ **58 帧回本，此后每帧净省 4.54 ms**，
而连续部署跑的是千帧量级。**杠杆 #11 从"净亏关闭"改判为"当前 ROI 最高的未落地项"**，
预期 117.56 → ~113 ms。⇒ **已在 §6.17 落地并实测：每观测 115.44 → 111.40 ms（−4.03 ms）**，
与这里的 ~113 ms 预测同量级（预测用的是 §6.15.4 那次 117.56 ms 的分段基准）。

### 6.16 与 Thor / RTX 的功能对齐审计（逐符号核实，不靠印象）

继承链实测：`GrootN17TorchFrontendOrin` → `RtxFP16` → `Rtx` → `Thor`
（`__mro__` 打印确认）。所以 **Thor 的整个公共 API 在 Orin 上都可见**，
问题只在于哪些成员描述的是 SM87 没有的能力。

| 能力 | Thor FP8 | RTX sm120/sm89 FP8 | **Orin SM87** |
|---|---|---|---|
| `set_prompt` / `infer` / `predict` / `normalize_state` / `denormalize_action` / `get_latency_stats` | ✅ | ✅ | ✅（`set_prompt`/`infer` 覆写，其余继承） |
| `num_inference_timesteps` / `action_horizon` / `num_timestep_buckets` / `initial_noise` / `use_dit_graph` 覆写 | ✅ | ✅ | ✅ **但图臂有前提**（§6.24）：`initial_noise` / `use_dit_graph` / `num_inference_timesteps` 逐项生效；`action_horizon` 与 `num_timestep_buckets` 被 4 张 DiT 图**按值烘死**（`Sa = action_horizon + 1`、调制器按 bucket 数预计算），覆写它们时**图臂一次性告警并退回 eager**（**+64…+67 ms/观测，1.66–1.69×**），而 eager 臂本身**与调用顺序无关**（实测 max&#124;d&#124; **0**）⇒ 数值正确。**修前是静默重放旧图**：实测解码动作偏 **3.36°**（40→20）/ **21.49°**（20→40）/ **3.58°**（buckets 1000→200），三者**都不报错**。要拿回图：一个前端固定一组 `(action_horizon, num_timestep_buckets)`；或显式传 `use_dit_graph=False` 把慢臂变成调用方自己的选择、同时消掉告警 |
| `num_views` 构造参数（1 相机可传） | ✅ | ✅ | ✅（**未复测**，§8） |
| DiT CUDA graph（4 张，per-bucket） | ✅ | ✅ | ✅ |
| 低比特档 | FP8 backbone（vit/dsm/llm/vlsa）+ FP4 变体 | FP8 backbone | **INT8 DiT**（默认，k/v 留 bf16）；backbone bf16 —— LLM/ViT 的 INT8 **实测否决**（§6.10/§6.11），SM87 无 FP8/FP4 硬件 |
| `calibrate` / `precision_spec` | ✅ FP8 alpha 校准 | ✅ | ❌ **不适用**：动态 per-row 无需校准 ⇒ **改为显式 `NotImplementedError`**（见下） |
| **backbone CUDA graph**（`run_backbone_graph`） | ✅ | ✅（FP8 mixin） | ✅ **已落地**（§6.17，每观测 −4.00 ms，逐位相同） |
| **每观测入口** `infer(aux=...)` | ❌（只有公开的 `run_backbone_graph(aux)`，要调用方自己驱动） | ✅（FP8 mixin） | ✅ **且是唯一正确的那个**（见下） |
| 契约校验（被图烘住的元数据） | ❌ | ✅ `_validate_backbone_graph_contract` | ✅ `_validate_observation_contract`（同款 snapshot/`_version` 快路径，另钉融合档的 `input_ids`） |
| **DiT cross-KV 失效** | ❌ | 🔴 **❌ 缺陷** | ✅ **原地刷新** |

**两个从这张表里落地/纠正的东西**：

1. 🔴 **RTX FP8 mixin 的 `infer(aux=...)` 会让 DiT 读到上一帧的 cross-KV**（已核实到行）：
   `groot_n17_rtx_fp8.py:180-192` 只重算 `_backbone_features`，
   然后调 `super().infer(...)`；而 Thor 的 `infer` 是
   `if not hasattr(self, "_dit_cross_K"): self._precompute_dit_cross_kv()` —— 
   `set_prompt` 已经建过了，所以**永远走不到重算**。
   整个 mixin 源码里 `_dit_cross` **零命中**（已用 `inspect.getsource` 钉进测试）。
   这正是 §6.15.3 那个"外层 cos 门不住"的失效模式：在**本 checkpoint** 上
   decoded cos 还有 0.9999917，**过门**。按红线 #1/#8 **只报不改**（§8）。
   ⚠️ Thor FP8 同理（`run_backbone_graph` 是公开的，调用方驱动，也没有 cross-KV 失效）。
2. ✅ **`calibrate` 从"崩溃"改成"拒绝"**。继承来的 Thor 实现无条件读
   `_vit_alpha_q` / `_dsm_alpha_*` / `_llm_alpha_*` / `_vlsa_alpha_*`，
   而本前端**一个都没有**（grep 计数 0，且实测
   `_snapshot_precision_spec` 抛 `AttributeError: no attribute '_vit_alpha_q'`）。
   即公共 API 广告了一个平台没有的能力，且失败方式是**别人代码里的裸属性错误**。
   现覆写为 `NotImplementedError` 并写清理由（SM87 无 FP8/FP4；INT8 DiT 的 scale 是
   **动态 per-row、每次 forward 重算**，静态档正是 §6.10 实测否决的那条；backbone bf16 不需要 scale）。
   选 `NotImplementedError` 是因为 `api.py` 的统一门**明文要求**这个异常类型。
   `precision_spec` 保持继承、继续返回 `None` —— 现在这个 `None` 是**诚实**的而不是**够不着**的。
   5 个 CPU 测试钉住（dispatch **51 → 56**），其中一条是**反向绊线**：
   哪天本前端真有了 FP8 alpha，测试会失败并要求把这个拒绝撤掉。

**⚠️ 一处此前的记录是错的，已按源码纠正**：§8 曾写"公共 API 路由不到 Orin N1.7"。
实测 `resolve_pipeline_class("groot_n17","torch","rtx_sm87")` →
**`GrootN17TorchFrontendOrin`**（`api.py:875` 本来就走 `_PIPELINE_MAP`）。
真正成立的只有：`api.py:831` 那个 **`use_fp16=True` 实验性白名单**里没有本三元组，
所以在 Orin 上传 `use_fp16=True` 会拿到一条**没提到 sm87** 的 `ValueError` —— 
是**报错信息质量**问题，不是路由缺口。另核实 `use_fp8=True` + sm87 会正常路由到 Orin，
并在 `__init__` 里**任何 CUDA 工作之前**被响亮拒绝（`_require_arch` 与
`super().__init__` 都在拒绝之后）。红线 #7：写下来的 API 都要对过源码，这条没对过。

**结论**：与 Thor/RTX 相比，Orin N1.7 当时**只有一个真实功能缺口**（backbone CUDA graph，
杠杆 #11）—— **该缺口已在 §6.17 补上**（每观测 −4.00 ms，逐位相同，64.9 帧回本），
所以现在**功能面没有缺口**。低比特档少是**硬件 + 实测否决**的结果
而不是没做；每观测入口与 cross-KV 失效这两项 **Orin 是唯一正确的实现**。

### 6.17 ✅ 杠杆 #11 已落地：backbone CUDA graph（每观测 −4.00 ms，逐位相同）

§6.13.2 判"不落地"、§6.15.6 改判"ROI 最高的未落地项"，本节把它做完。

**实现**（`groot_n17_orin.py`，全部 additive）：

| 成员 | 作用 |
|---|---|
| `_capture_backbone_graph()` | 侧流 warmup **3 次** → `capture_begin` / `_kbb_forward(stream)` / `capture_end`。warmup 不是可选的：不 warmup 时每个 kernel 的首次执行会在**捕获区内**惰性解析 cuDNN/cuBLAS workspace |
| `run_backbone_graph(aux)` | 公开方法，**与 Thor FP8 / RTX FP8 mixin 同名同契约**。顺序是 load-bearing：**校验契约 → 载入观测 → 惰性捕获 → replay → sync** |
| `infer(aux=...)` | 按 `_use_backbone_graph` 分流；graph 臂由 `run_backbone_graph` 自己校验契约，eager 臂由 `infer` 校验 ⇒ **两条臂各校验一次，不重复** |
| `use_backbone_graph=True` | 构造参数（默认开）。**只影响 `infer(aux=...)`**；一次性路径永不捕获 |

之所以能这么小：杠杆 #12 已经把 backbone 拆成 `_kbb_load_inputs`（写常驻输入缓冲）
+ `_kbb_forward`（纯 kernel、不分配、不 sync、显式收 `stream`），
那正是"任何未来捕获的硬前置"（§6.13.1 原话）。**捕获本身没有新增任何数值路径。**

**量测**（配对交替 A/B，**同一个前端、同一次捕获**，只翻 `_use_backbone_graph`，
所以两臂不可能在权重/缓冲/槽指针/捕获状态上有别；n=**56**/臂，中位数，锁频 1300.5 MHz，
仿真数据集 Se=148，8 个连续帧循环）：

| 边界 | eager | graph replay | Δ | 倍率 |
|---|---|---|---|---|
| **backbone 单独** | 63.52 ms（63.44–63.73） | **59.52 ms**（59.29–59.75） | **−4.00 ms** | **1.0673×** |
| **每观测 `infer(aux=)`** | 115.44 ms（115.09–115.93） | **111.40 ms**（110.98–112.33） | **−4.03 ms** | **1.0362×** |

**逐位相同**（这是"graph 没有引入第二条数值路径"的唯一合格证据）：
`backbone_features` max&#124;d&#124; **0.000e+00**、解码后动作 max&#124;d&#124; **0.000e+00**。
仓库门里 C4 的第一臂（重载 vs `set_prompt` 自己那份）现在也正是**graph vs eager** 的比较。

**一次性成本 259.7 ms ⇒ 回本 64.9 帧观测。** 连续部署是千帧量级 ⇒ 净赚。

**实测 vs 预测**（SKILL #15）：§6.13.2 的探针量到 Δ **4.54 ms**、回本 **57.7 帧**；
落地后实测 Δ **4.00 ms**、回本 **64.9 帧**。**差 0.54 ms（12%），方向是变小**，
两个原因都能说清、都不是回归：

1. 探针量的是**裸 `_kbb_forward`**；本节量的是**每观测真实边界**，
   两臂都含 `_validate_observation_contract` + `_kbb_load_inputs`。
   这两步在 graph 臂里**不会被图吃掉**（载入是 H2D copy，校验是 host 逻辑），
   所以它们同时抬高两臂、压低差值。
2. 数据集不同：探针在真机 cube_to_bowl_5（**Se=141**），本节在仿真集（**Se=148**）。
   eager 臂本身也从 62.47 变成 63.52 ms。

⇒ **§6.13.2 那个 4.54 ms 是"kernel 层面的可省量"，本节的 4.00 ms 是"部署能拿到的量"**，
两个数字都对，口径不同，别混用。

**门（"graph 真的跑了"的证据，红线 #5）**：`run_backbone_graph` 与 eager 臂
**逐位相同是设计目标**，所以任何数值门都**证明不了 graph 臂没被静默绕过**。
因此另有三条非数值的钉子：

| 门 | 断言 |
|---|---|
| **C8**（`…_continuous.py`，GPU） | `_kbb_graph` 存在；`_capture_backbone_graph` 整个模块生命周期**恰好 1 次**（此时已服务过 C1/C3/C4/C5 的多次观测）；再观测一次后**图对象身份不变**、`vlsa_h` 的 `data_ptr` **不变** |
| eager 臂可达性 | `use_backbone_graph` 默认 `True`，且 `infer` 源码里两条臂都在（opt-out 不是装饰品） |
| 校验不重复 | graph 臂里 `infer` **不**校验（由 `run_backbone_graph` 校验），eager 臂里 `infer` 校验；且校验发生在 `replay()` **之前** |

`test_the_backbone_graph_entry_point_is_a_known_gap` 那条**反向绊线已按设计触发并被替换** —— 
它当初写的就是"哪天 `run_backbone_graph` 出现了，这个测试会失败并要求同步文档"。

**测试计数**：dispatch **56 → 58**，continuous **7 → 9**；
真机档 **75 passed**（precision 17 + dispatch 58，55.8 s），
仿真档 **100 passed**（+ continuous 9 + N1.6 后端门 16，66.8 s）。

**原则对号**：#13（先量构成再动手；捕获能这么小是因为 #12 已经拆好了缝）、
#14（可捕获性就是这条路的天花板；一个单次调用略慢但图安全的形态是最高 ROI）、
#15（实测 vs 预测都报，且解释 12% 的差从哪来）、
#16（配对交替 + 中位数 + 锁频；n=56/臂，臂间差 4.00 ms 远大于 ±0.3 ms 的散布）、
红线 #1（全部 additive，`_kbb_forward` 一行未改）、
红线 #5（逐位相同**不**当证据，另立 C8）。

**实验件**：`/tmp/backbone_graph_ab.py`（未进仓）。
⚠️ 第一版把等价性检查写错了：**没传 `initial_noise`**，两臂各抽了一份新噪声，
于是报 max&#124;d&#124; **4.96** 看着像 graph 破了数值 —— 其实是量错了对象。
与 §6.15.3、§2.5(N1.6) 是同一类错误：**比对之前先确认两臂的输入是同一个**。

### 6.18 KV cache 到底是什么精度，以及"bf16 是不是 Orin 上的软件量化"（实测回答）

使用方澄清：「KV Cache 别量化」指的是**不做 INT8 量化**。本节把当前精度逐个核实，
并回答 bf16 在 Orin 上是否属于软件量化。**结论：不是。**

#### 6.18.1 当前每一个 K/V 张量的精度（逐个核实，非记忆）

| K/V | 位置 | dtype | 出处 |
|---|---|---|---|
| DiT **cross** K/V（就是"KV cache"） | `_dit_cross_K/V` + 后端槽 `dit_cross_K/V[j]` | **bf16** | `groot_n17_orin.py` 的 `_project_dit_cross_kv`；`attn_backend_groot_n17.py:60,64` |
| DiT **self** K/V | `dit_self_K/V` | **bf16** | `attn_backend_groot_n17.py:55-56`（`empty_like(dit_self_Q)`，Q 是 bf16） |
| LLM K/V（截断编码器，单次前向） | `llm_K/V` | **bf16** | `attn_backend_groot_n17_orin.py:45-47`，且 `dt != bfloat16` **直接 raise** |
| ViT K/V | `vit_K/V` | **bf16** | 同上（`slot_dtype=bfloat16`） |

⇒ **Orin N1.7 通路上没有任何一个 K/V 是 INT8。** INT8 只覆盖 DiT 的其余 GEMM 族，
`k`/`v` 两族被 `_DIT_BF16_FAMILIES = ("k","v")` 显式豁免（§6.14.4）。

#### 6.18.2 而且它算得比参考**更准**，不是更差

- checkpoint 文件里权重是 **F32**（实测 `model.safetensors.index.json` 抽样，
  八个模块组全是 F32），但 **HF eager 把整个模型 cast 成 bf16 运行**：
  `Isaac-GR00T/gr00t/policy/gr00t_policy.py:102` `model.to(device=device, dtype=torch.bfloat16)`，
  `:404` 连输入也 cast 成 bf16。⇒ **参考自己的 cross-K/V 就是 bf16 GEMM → bf16**。
- 本通路是 `kv_src.float() @ k_w + k_b` 然后 `.to(bf16)`：
  **fp32 真 SGEMM**（实测 `torch.backends.cuda.matmul.allow_tf32 = False`，所以不是 TF32）
  → 存 bf16。
- ⇒ **存量精度与参考相同（bf16），累加精度高于参考（fp32 vs bf16 tensor core）。**
  这里没有"相对基线的损失"可量。

#### 6.18.3 bf16 在 Orin 上是**硬件**，不是软件量化

三条独立理由，其中两条是实测的：

1. **实测：bf16 比 fp32 快 4.42×。** 在 cross-KV 的真实形状上
   （32 个 GEMM/观测，K=2048、N=1536、M=20 文本 ×16 + M=128 图像 ×16，
   锁频，中位数 of 15）：

   | 臂 | 32 个 GEMM |
   |---|---|
   | fp32（现状） | **6.746 ms**（6.734–6.783） |
   | bf16 tensor core | **1.525 ms**（1.521–1.540） |
   | | **4.42×，每观测省 5.22 ms** |

   **软件模拟的格式只会比原生格式慢**，不会快 4.4 倍。SM87 是 Ampere，
   bf16 tensor core 是一等硬件。
   ⚠️ 平台映射与 Thor/RTX **正好相反**：在 Orin 上被软件模拟的是 **FP8/FP4**
   （无硬件）—— 这正是前端对 `use_fp8`/`use_fp4` 直接 raise 的原因。
2. **bf16 不是量化方案，是浮点格式。** 量化 = 带**缩放因子**地映射到另一种数值格式。
   INT8 是定点（max **127**），必须 amax/校准/裁剪/零点，且**丢掉指数动态范围**；
   bf16 保留 fp32 的 **8 位指数** ⇒ max **3.3895e+38** vs fp32 的 3.4028e+38，
   **动态范围相同**，只是尾数短（7 位 vs 23 位，eps 7.812e-03）。
   **无 scale、无校准、无 amax、无裁剪。**
   ⇒ §6.5 那个"100–1000× 离群通道打死 per-tensor INT8"的问题，对 bf16 **根本不存在**。
3. **它就是基线自己的精度**（§6.18.2），所以不存在"比参考低一档"。

#### 6.18.4 ⚠️ 纠正：§8 那条"cross-KV 改走 bf16"此前被我错误归类成"量化 KV"

原话是"改走 bf16 tensor core 会动到 KV cache 的数值，按 KV 尽量别量化**未做**"。
**这个归类是错的**，实测：

- **两臂输入逐位相同**：`torch.equal(w32.to(bfloat16), w16)` = **True**，
  M=20 / M=128 的激活同样 **True**。`.float()` 是**精确提升**
  （这也是 §6.15.5 缓存它能逐位相同的原因）。
  ⇒ 没有任何东西被降精度，**这不是量化**，只是乘加路径不同
  （fp32×fp32 累加 vs bf16×bf16 乘、fp32 累加）。
- **存量结果的差**（这才是会进 KV 槽的东西）：

  | M | fp32 臂 vs bf16 臂（都是 fp32 值） | fp32 臂 `.to(bf16)` 存量后 vs bf16 臂 |
  |---|---|---|
  | 20 | cos 0.999998634，max&#124;d&#124; 4.96e-01（rel 2.58e-03） | **cos 0.999999997**，max&#124;d&#124; 5.0e-01 |
  | 128 | cos 0.999998631，max&#124;d&#124; 5.00e-01（rel 2.36e-03） | **cos 0.999999994**，max&#124;d&#124; 5.0e-01 |

  max&#124;d&#124; 0.5 相对 ~194 的量级是 **≈1 个 bf16 ULP**（bf16 eps 7.8e-03），
  rel 2.4–2.6e-03 与之吻合。**存量 cos 到 7 个 9**，远高于 0.999/0.995 的门。
- **收益**：每观测 **5.22 ms** ⇒ 111.40 → **~106.2 ms**（口径 C 3.47× → **~3.64×**）。
  与 §6.15.4 的分段自洽：8.52 ms 刷新 = 6.75 ms GEMM + ~1.8 ms 的
  bias/`.to(bf16)`/`.contiguous()`/掩码索引。

⇒ **该项不再是"默认不做"，而是"精度门通过后就该做"**，且它**不违反**
"KV 不做 INT8 量化"的要求。门仍必须做（存量 cos / velocity floor / decoded max|d|
三档并列，§8），因为 1 ULP 的差要经过 4 步 Euler 与 32 层 DiT 放大后才看得出结论。

**实验件**：`/tmp/kv_bf16_native_check.py`（未进仓）。

**原则对号**：#6（权重/运行时为准：checkpoint 是 F32 但 HF 跑 bf16，两处都要实测）、
#12（先分类再动手：这不是"噪声 vs scale"，而是**根本不是量化问题**）、
#13（先微基准再写 kernel：4.42× 是量出来的，不是推出来的）、
#16（锁频 + 中位数 + 真实形状；散布 ±0.02 ms 远小于 5.22 ms 的差）、
红线 #7（每条 API/数值都对过源码或量过 —— 本节纠正的正是没对过就写下的那句话）。


### 6.19 cross-KV 投影改走 bf16 tensor core：三档门通过，但**收益比隔离微基准说的小一半**

§6.18.4 判定该项"不是量化、门通过后就该做"。本节是落地 + 三档门 + 配对量测。

**实现**：`_project_dit_cross_kv` 改用 `gemm.bf16_nn` + `fvk.add_bias_bf16`，
直接吃**已经是 bf16** 的 `_dit_k_w[li]`（实测 `(2048,1536)` bf16 row-major，
正好是 `bf16_nn` 的 `[K,N]` 约定）。**M/N/K 从权重形状读，不写常量**（红线 #6）。
旧的 fp32 臂保留为 `_project_dit_cross_kv_fp32`，**明确标注只作参考、不在服务路径上** —— 
它是三档门的对照臂，删了门就没法写。
**403 MB 的 `_dit_cross_kv_weights` fp32 权重缓存随之删除**（§6.15.5 那个），
bf16 GEMM 不需要它。

**为什么 bias 是单独一个 kernel**：`bf16_nn_bias` 的 cuBLASLt epilogue 在 SM87 上
返回 `code=15`（NOT_SUPPORTED），**实测 M=20 与 M=128 都失败** ⇒
不是 `pipeline_orin.py:520` 记的"M 未 16 对齐"那个原因，而是**这个 arch 上根本没有该 epilogue**。
（已列 §8 上报 kernel owner，并纠正 pipeline_orin 那条归因。）

#### 6.19.1 三档精度门（6 帧，两个数据集，同一前端同一 `initial_noise`）

唯一的变量是投影臂（临时把 `_project_dit_cross_kv` 绑到 fp32 参考臂）。

| 帧 | 存量 cross-K cos(bf16,fp32) | max&#124;d&#124; | velocity cos fp32→bf16 | decoded vs HF：fp32 臂 → **bf16 臂** |
|---|---|---|---|---|
| 真机 0 | 0.999995693 | 1.25e-01 | 0.9995017 → 0.9994165 | 0.4935° → **0.4935°**（不变） |
| 真机 300 | 0.999995672 | 1.25e-01 | 0.9995914 → 0.9995758 | 0.1940° → **0.2626°** |
| 仿真 100 | 0.999995876 | 1.25e-01 | 0.9994823 → 0.9994513 | 0.2467° → **0.2467°**（不变） |
| 仿真 101 | 0.999995894 | 6.25e-02 | 0.9995131 → 0.9994848 | 0.1940° → **0.2499°** |
| 仿真 107 | 0.999995896 | 1.25e-01 | 0.9995668 → 0.9995331 | 0.2499° → **0.3872°** |
| 仿真 31000 | 0.999995882 | 1.25e-01 | 0.9995048 → 0.9994292 | 0.2129° → **0.2467°** |

| 档 | 最差 | 门 | 判定 |
|---|---|---|---|
| 1 存量 cross-K | cos **0.999995672**（≈1 个 bf16 ULP，rel ~7e-03） | ≥0.999 | ✅ |
| 2 velocity vs HF | cos **0.9994165** | ≥0.999 | ✅ |
| 3 decoded vs HF | cos **0.999999073**；max **0.4935°** | ≥0.999 | ✅ |

**必须说清的代价**：6 帧里 **4 帧的 decoded 误差上升 +0.034…+0.137°**，
两帧不变（那两帧本来就被 bf16 backbone 的地板 0.4935° 顶住，本改动碰不到）。
**跨 6 帧的最坏值不变（0.4935°）**。所有门都过，余量在 4 个数量级以上。

> ⚠️ **一处需要如实指出的比较**：仿真 107 的 bf16 臂 0.3872° **略高于**
> §6.14.2 六族全量化档在真机 f300 上的 0.3636°。两者不是同一帧、不是同一数据集，
> 所以**不能直接比大小**；但也不能拿"远低于全量化档"当安全论据。
> 站得住的说法只有一条：**本档在自己的 6 帧上最坏 0.4935°，与改动前逐帧同值或更差
> ≤0.137°，且所有既有门全部通过**。

#### 6.19.2 延迟：**隔离微基准把收益高估了 2.3×**（原则 #16 的又一次实例）

三臂配对交替（同一前端、同一批图、n=**56**/臂、中位数、锁频、8 个连续仿真帧循环）：

| 臂 | refresh 单步 | 每观测 `infer(aux=)` |
|---|---|---|
| **bf16（新）** | **4.75 ms** | **108.55 ms**（108.05–109.14） |
| fp32 + 权重缓存（**改动前的出厂实现**） | 8.50 ms | 111.17 ms（111.00–111.74） |
| fp32 每次提升（参考臂，非出厂） | 13.36 ms | 116.16 ms |

**诚实的收益：每观测 −2.62 ms（1.0241×）**，refresh 单步 −3.75 ms，
外加**释放 403 MB**。口径 C：**3.47× → 3.56×**（vs 完整 `get_action`）/
**3.23× → 3.31×**（vs 同一边界）。

**为什么先前报的 5.22 / 6.15 ms 是错的**（两个都错，方向一致）：

1. `/tmp/kv_bf16_native_check.py` 量的是**合成张量上的裸 torch matmul**，没有 bias、没有 cast；
2. `/tmp/kv_bf16_kernel_probe.py` 量的是**真张量上的 32 个 GEMM 循环**（7.918 → 1.763 = 6.15 ms），
   但 refresh 这一步里还有 **~3 ms 的非 GEMM 开销**（布尔掩码索引内含 `nonzero()` ⇒ host sync、
   `.contiguous()`、32 次 `torch.empty`、32 次槽 copy），**这部分两臂相同**；
3. 进了流水线之后，fp32 臂那部分开销还有一部分**与其他工作重叠** ⇒
   refresh 省 3.75 ms 而每观测只省 2.62 ms。

⇒ 这正是 SKILL 原则 #16 说的"隔离 µbench 在**两个方向**上都会翻转结论"，
以及红线：**只有在流水线里的配对 A/B 才算数**。
✅ 量测自身的健康检查：fp32+缓存臂量到 **8.50 ms**，与 §6.15.4 记录的 **8.52 ms** 对上；
每观测 **111.17 ms** 与 §6.17 记录的 **111.40 ms** 对上 ⇒ 三臂是在同一条件下量的。

#### 6.19.3 判定与门

> **⚠️ 本小节的"取舍"定性已被 §6.20 撤回。** 下面"失"那一行是 **n=6** 的观测，
> §6.20 用**全 episode 593 连续观测**重测：两臂均值差 **+0.0012°**、**36.1% 的帧逐值相等**、
> 全程最坏值 bf16 **0.4935°** 反而**优于** fp32 的 **0.5808°**、三档门 **0/593 违反**。
> 即 bf16 臂在门内与参考臂**不可区分**，不构成精度换速度的取舍。回退方式仍然保留，
> 但现在是冗余保险而非待决事项。**教训记在 §6.20.5。**

**判定：保留 bf16 臂为服务路径**，但这是一个**使用方可以否决的取舍**，
因为它动的正是使用方点名关注的 KV 通路：

- **得**：每观测 −2.62 ms（2.4%）、口径 C 3.47× → **3.56×**、释放 **403 MB**；
- **失**：4/6 帧的 decoded 误差 +0.034…+0.137°（跨帧最坏值不变），
  velocity cos 0.99950 → 0.99943。
- **不是**失：KV 的**存储精度没变**（仍 bf16，与 HF eager 自己的一致，§6.18.2），
  **没有 INT8**，没有 scale/校准/裁剪。

**回退方式**（若使用方否决）：把 `_project_dit_cross_kv` 绑回
`_project_dit_cross_kv_fp32` 并恢复 `_dit_cross_kv_weights` 缓存即可，
参考臂与其文档都还在仓里，**不需要重新实现**。

**测试**：dispatch **58 → 63**（新增 5 条 CPU 契约钉子：
`bf16_nn` 的 **(M,N,K) 参数顺序**——注意是 N 在 K 前，换位在裸指针上不会报错只会算错；
输出 dtype/shape/连续性；权重 K 与特征宽度不符时**拒绝**；
不得使用 `bf16_nn_bias`；403 MB 缓存确实已删）。
CPU 桩改为把 `_project_dit_cross_kv` 影子绑定到 fp32 参考臂
（服务臂需要 CUDA + `GemmRunner`，而这些钉子测的是槽与图的**记账**，不是那段算术）。
仿真档 **100 → 105 passed**，真机档 **75 → 80 passed**。
G1–G5 与 C1–C8 **全部重跑通过**。

**实验件**：`/tmp/kv_bf16_kernel_probe.py`、`/tmp/kv_bf16_three_tier.py`、
`/tmp/kv_bf16_latency_ab.py`（均未进仓；结论已进本节，契约门已进 dispatch）。

**原则对号**：#6（M/N/K 从权重形状读）、#13（先探针后实现——探针也正是它
先给出了偏大的收益数字，随后被流水线 A/B 纠正）、
#16（**本节的主教训**：隔离微基准高估 2.3×；三臂里必须有"改动前的出厂实现"那一臂，
否则会把参考臂的额外开销算成收益）、
红线 #2（复用 `bf16_nn` + `add_bias_bf16`，与 DiT pipeline 同一套 workaround）、
红线 #4（形状不符直接 raise）、红线 #7（`bf16_nn_bias` 的失败原因实测纠正了既有归因）。


---

### 6.20 ✅ 长任务精度对比：全 episode **593 连续观测**，bf16 臂与 fp32 臂**不可区分**

使用方拍板"保留 bf16 臂"，并要求用其提供的仿真数据集做一次**长任务**对比。
§6.19.1 的三档门只跑了 **6 帧**——够判定"过门"，不够判定"代价的分布"。本节补上。

#### 6.20.1 口径：部署模式，不是逐帧独立跑

| 项 | 值 |
|---|---|
| 数据 | `green_to_blue_block_sim`（使用方提供，仿真采集 SO101：50 episodes / 31166 frames / 30 fps / 2 cameras h264 640×360） |
| 任务 | episode 0 = "Pick up green block and put it on the blue block"，**593 帧全跑**（19.7 s 操作） |
| 服务方式 | **`set_prompt` 一次（frame 0），随后 593 次 `infer(aux=...)`**；距 prompt **0..592** |
| 每观测实际走的代码 | 契约校验 → backbone 输入装载 → backbone graph replay → **重投影 16 组 cross-K/V** → 槽内就地刷新 → DiT graph replay |
| 三臂控制 | 同一 frontend、同一 pinned `initial_noise`，**唯一变量是投影臂**（类属性换绑） |
| 墙钟 | HF eager **401.2 ms/帧**；FlashRT 两臂合计 **264.6 ms/帧** |
| 时钟 | 全程锁定 1300500000（`cur == min == max`，跑前跑后各验一次） |
| harness | `tests/_helpers/groot_orin/long_horizon_ab.py`（**已进仓**）：流式，**不落盘 6.8 GB fixture**；hooks 直接复用 `capture_aux.py`（红线 #2） |

**harness 自校验**（否则长时程数字无从采信）：在 frame 100/101/102 上与 fixture 版
三档门 `/tmp/kv_bf16_three_tier.py` 对表——t1 cos **0.999995876 vs 0.99999588**、
bf16 decoded **0.2467° 两者同值**、t2 **0.9994513 两者同值**（frame 102 同样逐项吻合）。
⇒ "活 HF + FlashRT 同进程"与"磁盘 fixture"是同一口径，长时程跑的是**真实部署形状**。

#### 6.20.2 结果（n = 593，全部真实帧）

| 档 | 指标 | bf16 臂 | fp32 参考臂 | 门 | 违反 |
|---|---|---|---|---|---|
| 1 stored cross-K | cos min / median | **0.999995853** / 0.999995888 | （两臂之比） | ≥0.999 | **0/593** |
| 1 | max&#124;d&#124; worst | 1.250e-01 | — | — | — |
| 2 velocity vs HF | cos min / median | **0.9992429** / 0.9994851 | 0.9992084 / 0.9994835 | ≥0.995 | **0/593** |
| 3 decoded vs HF | cos min / median | **0.999998845** / 0.999999553 | 0.999998965 / 0.999999556 | ≥0.999 | **0/593** |
| 3 | max° mean / median / p95 / **max** | 0.3043 / 0.2802 / 0.4511 / **0.4935** | 0.3032 / 0.2763 / 0.4511 / **0.5808** | — | — |
| 3 | A−B（两臂之差）mean / p95 / max | 0.2470 / 0.3921 / 0.7402 | — | — | — |
| normalized（解码前） | cos min | 0.999996140 | 0.999996220 | — | — |

**动作量程 246.82°**（中位 150.23°）⇒ bf16 臂**相对**误差 mean **0.186%**、p95 0.299%、max **0.413%**。

#### 6.20.3 决定性的一条：两臂之差**不随 prompt 距离增长**

长时程要抓的是**状态漂移**（槽位陈旧、刷新漏刷），它的指纹是"误差随距离单调上升"。实测：

| 量 | r(距离) | 斜率 | r(signal 幅值) | 斜率 |
|---|---|---|---|---|
| **A−B（两臂之差）** | **+0.0076** | **+3.32e-06 °/帧** | −0.0077 | −1.29e-05 |
| bf16 臂 vs HF | +0.2107 | +9.47e-05 | +0.2119 | +3.67e-04 |
| fp32 臂 vs HF | **+0.2641** | **+1.20e-04** | +0.3012 | +5.29e-04 |

- **A−B 对距离的相关是 +0.0076，斜率 3.3e-06 °/帧**（592 帧累计 **+0.002°**）⇒ **排除状态漂移**。
- **fp32 参考臂自己的距离趋势（+0.2641）比 bf16 臂（+0.2107）更强**。参考臂里没有任何 bf16 改动，
  所以这个趋势是流水线/内容固有的，**不是本次改动带来的**——这是"用参考臂给趋势定基线"的标准做法。
- 趋势的主因是**内容**不是**距离**，三条独立证据：
  ① 两臂对 signal 幅值的相关与对距离的相关同量级；
  ② **相对**误差与距离 **负**相关 r = **−0.3198**（越到后面动作越大，相对误差反而降）；
  ③ 分桶表最后一桶**两臂同时**抬高，且该桶 signal 也最大：

| 桶 | 距离 | bf16 | fp32 | A−B | signal |
|---|---|---|---|---|---|
| 0 | 0–73 | 0.2965 | 0.2983 | 0.2529 | 187.9 |
| 1 | 74–147 | 0.2834 | 0.2675 | 0.2262 | 120.7 |
| 2 | 148–221 | 0.3005 | 0.2888 | 0.2452 | 138.6 |
| 3 | 222–295 | 0.2780 | 0.2909 | 0.2619 | 148.2 |
| 4 | 296–369 | 0.3058 | 0.3009 | 0.2544 | 149.6 |
| 5 | 370–443 | 0.3054 | 0.2978 | 0.2480 | 164.6 |
| 6 | 444–517 | 0.3083 | 0.3129 | 0.2448 | 218.0 |
| 7 | 518–592 | **0.3561** | **0.3672** | 0.2429 | **245.9** |

桶 3 比桶 0 低、桶 6 比桶 5 高、最后一桶**fp32 更高**——非单调，且 A−B 那一列基本平（0.226–0.262）。

#### 6.20.4 头对头：是噪声，不是偏置

| 项 | 值 |
|---|---|
| bf16 更好 / 更差 / **完全相等** | 182 (30.7%) / 197 (33.2%) / **214 (36.1%)** |
| mean(bf16 − fp32) | **+0.0012°**（median +0.0000） |
| p95 / 最坏 / 最好 | +0.1475 / **+0.3021** / **−0.3105** ⇒ **对称** |
| 两臂 worst-20 帧重合 | **3/20** ⇒ 各自的"最坏帧"不是同一批 |
| 逐通道最坏（6 个 SO101 关节） | shoulder_lift 0.4511 / elbow_flex 0.4935 / wrist_flex 0.3891 / wrist_roll 0.4646 **四通道两臂逐值相同**；shoulder_pan **+0.1046**（bf16 差）、gripper **−0.1432**（bf16 好） |

`0.4935°` 在全程反复出现（f58/f378/f380/f390/f460…）且在 4 个通道上两臂**逐值相同**
⇒ 它是 **bf16 输出网格的台阶**，与 §6.14.3 认定的"bf16 backbone 地板"同源，不是任何一帧的异常。

#### 6.20.5 结论：**撤回 §6.19.3 的"使用方可以否决的取舍"定性**

n=6 时测到的"4/6 帧 decoded 误差 +0.034…+0.137°"是**小样本噪声**。n=593 时：

- 两臂均值差 **+0.0012°**，**36.1% 的帧两臂逐值相等**；
- **全程最坏值 bf16 0.4935° 反而优于 fp32 的 0.5808°**；
- 三档门 **0/593 违反**。

**余量（按 1−cos 算，不用"几个数量级"这种口头话）**：

| 档 | 全程最差 cos | 门 | 1−cos | 余量 |
|---|---|---|---|---|
| 1 stored cross-K | 0.999995853 | ≥0.999 | 4.147e-06 | **241×**（2.38 个数量级） |
| 2 velocity | 0.9992429 | ≥0.995 | 7.571e-04 | **6.6×**（0.82 个数量级） |
| 3 decoded | 0.999998845 | ≥0.999 | 1.155e-06 | **866×**（2.94 个数量级） |

**档 2 是真正的约束**，余量只有 6.6×，而且两臂在这里几乎同值（bf16 0.9992429 / fp32 0.9992084，
bf16 还略好）⇒ 这条余量是**整条通路共有的**（bf16 backbone + INT8 DiT），不是 cross-KV 投影的开销。
写"余量 6 个数量级"会把它盖住——**这是本节自己差点犯的记录错误，按红线 #7 纠正。**

⇒ bf16 臂**不是"用精度换速度"的取舍**，而是**在门内与参考臂不可区分**，
同时白拿 **−2.62 ms/观测**、口径 C **3.56×**、释放 **403 MB**。§6.19.3 记的回退方式仍然保留
（参考臂与文档都在仓里），但它现在是**冗余保险**，不是待决事项。

**教训（写进原则对号）**：原则 #16 说"perf delta 只有在 hygiene + triage 之后才算数"，
**精度 delta 同理，而且对样本量的要求更高**——延迟可以"中位数 of N 次重复"自带降噪，
精度每帧只有一次采样，n=6 的均值差的置信区间宽到能改变结论的**符号**。
§6.19.1 的三档门判 PASS 判对了；错的是 §6.19.3 用 6 帧去描述代价的**分布**。
**规则：任何"取舍"定性都要有长时程或大样本支撑，否则只写"过门"，不写"代价"。**

---

### 6.21 长任务过程中实测到的两个真实缺陷（都已修 + 都有负控）

长时程跑法的价值不止在精度数字：它第一次把 FlashRT 放在**活体集成**（HF 与 FlashRT 同进程、
整段 episode 连续读）里，于是暴露出两个此前所有 A/B 都绕过的缺陷。

#### 6.21.1 inference tensor 打在每观测通路上（第一帧就炸）

**现象**：`RuntimeError: Inference tensors do not track version counter.`，`_kbb_load_inputs` 第一帧。

**根因**：`getattr(t, "_version", None)` **不会**返回默认值——`_version` 是个**会抛异常的 property**，
`getattr` 只对"属性不存在"兜底，属性存在但抛异常会**直接穿透**。
`torch.inference_mode()` 下产出的张量是 inference tensor，而 `inference_mode` 正是官方
`Gr00tPolicy.get_action` 的推荐跑法 ⇒ **活体集成在第一帧就不可用**。

**为什么之前没暴露**：既有全部验证都走**磁盘 fixture**，`torch.load` 出来的不是 inference tensor。
这是 §6.15 / §6.17 / §6.19 所有 A/B 的**共同盲区**。

**修法**（新增模块级 `_mutation_version` 助手 + 3 处调用点）：版本号不可读 ⇒ **绝不**走 identity 快路径，
退化成"每观测重新校验 + 重新拷贝"。丢的是时间（backbone load ~1.5 ms/观测），不是正确性；
反过来（把 `None` 当成可匹配）会**静默服务陈旧字节**，红线 #4。

**更深的一个洞（负控证实）**：`_validate_observation_contract` 会把 `(validated_source, validated_version)`
记下来做备忘，而 inference tensor 的版本是 `None`、`None == None` 成立 ⇒ **第二次就跳过值比较**。
调用方在自己的 `inference_mode()` 块里原地改了 rope 表，会被当成"已校验"放行——
这是静默正确性错误，比崩溃严重得多。去掉 `version is not None` 守卫后新钉子报
`DID NOT RAISE ValueError`；加回守卫则通过。

**不静默**：一次性 `logger.warning` 说明为什么变慢、以及怎么把快路径买回来
（在 `inference_mode` 块外 `t.clone()`）。

**测试**：dispatch **63 → 67**（`_mutation_version` 单测并**钉住 getattr 陷阱本身**、备忘不得缓存不可校验的张量、
inference tensor 观测必须重新拷贝、慢下来必须播报一次）。

**同款缺陷在别处（report-only）**：`groot_n17_rtx_fp8.py:211` 与 `:244` 是同一个
`getattr(..., "_version", None)` 写法 ⇒ **RTX FP8 / SM89 / Thor FP8 的 backbone-graph 契约有同样的洞**
（含那个"备忘缓存 `None` 版本"的静默放行）。改它要动三个平台，按红线 #1 只报告不代改，已进 §8。

#### 6.21.2 视频读取器丢掉整段 episode 的尾巴（长时程少 2%）

**现象**：593 帧的 episode 0 跑到 frame **581** 抛 `av.error.EOFError: 'avcodec_send_packet()'`，
进程死在 `summarize()` 之前。

**不是数据问题**：容器里两条相机流各 **593** 帧（`stream.frames` 与实际解码数一致）、
parquet **593** 行、`episodes.jsonl` length=**593**——四者完全一致。

**根因**：`_decode` 每次调用**重建** `container.decode(stream)` 生成器，并在循环体内 `return`
——每次都**弃置一个活着的生成器**。PyAV 在 finalize 被弃置的解码生成器时会 **flush 编解码器**，
flush 之后仍排在队列里的帧就再也取不出来；升序走到文件尾部时 demuxer 先耗尽 ⇒ EOFError。

**定位方式（原则 #13：先探针后动手）**：20 行独立探针，同一个文件、同一个 codec、
同一个 `thread_type="AUTO"`，**只改"生成器寿命"这一个变量**——弃置式在 target=**581** 失败，
持有式读完 **593**。再用合成 600 帧视频复现在 **587**（两者都在 ~98% 处，
与"flush 之后队列里的尾部帧不可达"的机制一致）。

**修法**：每个 decoder 持有一个**跨调用挂起**的迭代器；**seek 之前先把它置 `None`**
（流被挪动之后再恢复挂起的迭代器是未定义行为）。

**改完还要证明"取到的还是同一帧"**（少一帧不会报错，只会让参考悄悄错位）：
拿真数据集对**独立基准**逐像素比对——基准是"每帧新开一个容器、从头解码到目标"，
与读取器不共享任何状态。5 轴全过：

| 轴 | 内容 | 结果 |
|---|---|---|
| 1 | episode 0 整段升序 | **593/593**（修前 581 崩） |
| 2 | 逐像素一致，含**旧失败点之后**的 581/592（帧 0/1/17/100/301/580/581/592） | **0 处不符** |
| 3 | 随机访问 + 回退 `[500,10,499,0,300,299,592,1]` | **0 处不符** |
| 4 | 跨 episode 边界（全局 592/593/594/1124/1125，含 ep0 末帧与 ep1 首末帧） | **0 处不符** |
| 5 | 越过真实末尾 | `_episode_for` 抛 `KeyError`（既有的响亮守卫，非静默返回旧帧） |

轴 2–4 是一次性实测（不在仓内测试里，因为要真数据集）；轴 1/3/5 的等价物由下面的
合成视频测试常驻钉住。

**为什么之前没人踩到**：既有调用都是**散点**帧号（`--frames 0,300`），要么重开容器要么 seek，
几乎不走长升序。只有"整段 episode"这种读法才会踩——而它正是长任务验证的读法。
⚠️ **§5.3 记录的验证矩阵是"跨 episode 读取 / seek 回退 / 同帧重复读 bit-identical /
分层抽样覆盖 episode 0–3"——四条全是散点访问**，所以那次 bit-for-bit 校验**结构上不可能**发现这个洞。
本读取器是 FlashRT 侧新增（§5.3 标 `[PHASE 0]`），不是既有仓库代码，缺陷从一开始就在。
**补进验证矩阵的一条：长升序全程逐帧可达**（现已由 `test_every_frame_of_a_full_ascending_pass_is_reachable` 钉住）。

**测试**：新增 `tests/test_lerobot_video.py`（**5 条，CPU-only，0.78 s**）：
全长升序逐帧命中（**回归钉**）、容器确实复用（防止"每次都重开"这种也算修好、
但把 O(N²) 换回来的修法）、seek 回退仍逐帧正确、越界 raise IndexError、两路相机各自独立位置。
合成视频自带（600 帧 64×64 libx264，~1 s，帧 *i* 白带落在第 `i % 64` 行 ⇒ 校验的是**帧身份**不是帧数），
**不依赖仓外数据集**，无 libx264 时 skip。

**负控**：把 `for frame in state["it"]` 改回 `for frame in container.decode(stream)`，
`test_every_frame_of_a_full_ascending_pass_is_reachable` 以**生产环境同一条** `EOFError` 失败；
改回修复版 5 passed。

#### 6.21.3 一次测量卫生事故（记录以免重犯）

长时程后台进程刚结束（`ps` 显示 stat **`Z`**，尚未被回收）时立刻跑真机档闸门，出现 **4 failed**
（全在 **bf16 档**：`test_denoise_loop_and_decoded_action[bf16-0/300]`、
`test_dit_graph_is_bit_identical_to_eager_and_deterministic[bf16-0/300]`）。

逐项隔离：单个测试函数 **4 passed** → precision 模块单独 **17 passed** → dispatch+precision **84 passed**
→ **同一条三文件命令重跑 89 passed**。

⇒ 与本轮两处代码改动**无因果**，是前一个 GPU 进程的上下文尚未释放导致的瞬态；
bf16 档要额外构造 frontend（"四个活 frontend ~14 GB"），所以它是**第一个顶到内存压力**的。

**规则**：后台 GPU 任务报 zombie 之后，**先确认进程被回收再跑闸门**，否则会把瞬态读成回归。
这与 `references/case-studies/chameleon-vlm-orin.md` 那四类自造测量错误是同一类。

**最终闸门**（时钟全程 1300500000）：仿真档 **114 passed**（dispatch 67 + precision 17 + continuous 9
+ N1.6 后端 16 + lerobot_video 5，67.4 s）；真机档 **89 passed**（dispatch 67 + precision 17 + lerobot_video 5，56.5 s）。

**原则对号**：#1（先复现基线再谈结论——harness 先与 fixture 对表）、#13（20 行探针定位，单变量）、
#16（瞬态不等于回归；逐项隔离后才归因）、
红线 #1（`rtx_fp8` 同款洞只报告不代改）、#2（hooks 复用 `capture_aux.py`，未另写一份）、
#4（不可读版本号 ⇒ 退化到慢而正确的一边；性能退化也要播报；越界 raise 不静默）、
#7（`_version` 是 property 而非缺失属性，已按源码行为写进 docstring）、
AGENTS.md §3.6（两个修复**都有负控**：去掉守卫 `DID NOT RAISE`、还原生成器 `EOFError`）、
§3.7（593 帧全部真实采集，含 state 的 deg→rad 口径）。

---

### 6.22 ✅ 杠杆 #9 落地：融合 bf16 rotate-half RoPE（每观测 −10.47 ms，**逐位相同**）

使用方要求"继续完善 N1.7，检查是否还有可做算子融合的，或者引入 FA2"。按原则 #16
（每次收益之后瓶颈会移动）**重新做了画像**，而不是照旧杠杆树施工。结果分三条。

#### 6.22.1 FA2 这条杠杆早已花掉：注意力只占 GPU 时间 **2.3%**（原记 0.3%，是分桶错误，已更正）

出厂配置的 kernel 普查（`infer(aux=)`，锁频 1300.5 MHz，**shim 臂 = 改动前**，总计 102.257 ms）：

| 分组 | ms | 占比 | launches |
|---|---|---|---|
| GEMM 合计（bf16 ampere 44.138 + INT8 cutlass DiT 21.407） | **65.545** | **64.1%** | 1011 |
| aten elementwise | 13.216 | 12.9% | 870 |
| `add_bias_bf16_kernel` | 8.852 | 8.7% | 848 |
| norm | 3.111 | 3.0% | 385 |
| **attention（FA2，`fa2_vendor::flash_fwd_kernel`）** | **2.335** | **2.28%** | **172** |
| quantize（INT8 per-row） | 2.311 | 2.3% | 512 |
| activation（`gelu_kernel`） | 2.224 | 2.2% | 176 |
| `res_add_kernel` | 1.845 | 1.8% | 347 |
| aten cat | 1.639 | 1.6% | 92 |
| conv（cuDNN，patch embed） | 0.471 | 0.5% | 2 |

FA2 **本来就在每一个注意力位点**：backbone 用 `OrinGrootN17BackboneAttn` 的
`fwd_bf16` / `fwd_bf16_causal`，LLM 用**原生 GQA** 的 `fwd_bf16_causal`
（`num_heads_q=16` / `num_heads_kv=8`，不需要 `repeat_interleave`）。
172 次 launch **逐项对得上位点数**：**128** = DiT 的 32 个注意力层 × 4 步
（奇偶**交替** self/cross，不是叠加，§4）+ **24** = ViT + **16** = LLM + **4** = VLLN 的
4 个 `vl_self_attention`。模板实例也自洽：LLM 那条是 `Flash_fwd_kernel_traits<128, 64, …>`
（head_dim **128**），其余三条是 `<64, 128, …>`（head_dim **64**），与 §4.1 的实测形状一致。
**2.28% 就是它的账，且已是厂商 kernel ⇒ "引入 FA2" 无可再得**（把注意力整块清零也只值 2.3 ms）。

⚠️ **本节第一版把注意力记成了 0.312 ms / 0.3% / 21 launches，错了 7.5×；同时把 GEMM 记成
67.97 ms / 73.3%，也错了。两个错同源，都是关键字分桶撞上了 C++ mangled 模板名**：

| 撞车 | 后果 |
|---|---|
| FA2 的名字是 `fa2_vendor::flash_fwd_kernel<Flash_fwd_kernel_traits<…, cutlass::arch::Sm80, …>>`，**含 `cutlass`/`sm80`**，而 GEMM 规则排在 attention 规则**之前** | **全部 FA2 时间被算进 GEMM** ⇒ GEMM 虚高 2.335 ms，attention 只剩 0 |
| attention 规则里的 `"flash" in kl` **匹配命名空间 `flash_rt::kernels::`** | 那条 0.312 ms / 21 launches 根本不是注意力，是**漏网的 FlashRT 自有 kernel**（rope shim 时代的残余）|

更正后的交叉验证（这也是能确认更正正确的理由）：**GEMM 在两臂之间几乎不动**
（65.545 → 65.425 ms，差 0.12 ms），而 rope 改动本来就不该碰任何 GEMM；
两臂总差 9.818 ms 全部由 elementwise（13.216→4.268，−8.948）+ cat（1.639→0.058，−1.581）
+ 新增 rope kernel（+1.053）+ 其余零头构成，逐项可加。
上一版记的"GEMM 73.3%"= 65.545 真 GEMM + 2.335 FA2 + 0.09 零头 = 67.97，**误差来源可对账**。

⚙️ **规约（进 §8，与杠杆 #9 那条"枚举全部扩展"同源）**：kernel 普查的分桶
**必须按精确名字前缀分类，attention 排在 GEMM 之前**，并且**要打印每个桶的成员名单**而不只是总量
——本次两个错都是"总量看着合理"才活下来的。本节此前已因同一类错误纠正过一次
（`DefaultGemmWi…` 被归进 copy/cast），**这是第二次**。

#### 6.22.2 归因探针：rope shim 是 **11.902 ms**，不是杠杆树记的 2.7 ms（**4.4× 偏低**）

普查里 `aten elementwise 13.088 ms` + `aten cat 1.642 ms`，而 `CatArrayBatchedCopy`
恰好 **80 次** —— 正好是一次 backbone 里 `_rope_rotate_half` 的调用数
（24 ViT 层 ×2 + 16 LLM 层 ×2）。这个巧合太强，不能靠推断收尾，于是**直接归因**：
把 shim 猴补丁成 no-op，取 GPU 时间差。

| | shim 活着 | shim → no-op | **归因给 shim** |
|---|---|---|---|
| backbone GPU 总时间 | 60.972 ms | 49.070 ms | **+11.902 ms（19.5%）** |
| aten elementwise | 10.044 | 0.439 | +9.605 |
| aten cat | 1.746 | 0.000 | +1.746 |
| launches | 1683 | 883 | **+800**（10 kernel/次 × 80 次） |

⇒ **§6.2 杠杆 #9 与 §8 记的"~2.7 ms"错了 4.4×**，且 §8 那句"上限就是那 ~2.7 ms"
也一并作废。按 108.55 ms 的口径 C 算，这一项是 **11.0%**，是**当时最大的未落地杠杆**。
（§6.13.3 说"别按省 CPU dispatch 估这条"是对的——但它是 GPU 时间本身被低估了。）

#### 6.22.3 `blocked_on_kernel` 是**空判**：kernel 早就编好了，只是在**另一个扩展模块**里

杠杆 #9 标的是"需 bf16 `rope_rotate_half`"。实际枚举**全部已构建的扩展**
（不是只看 `flash_rt_kernels`）：

```
flash_rt/flash_rt_kernels*.so                  <- 一直只查这个
flash_rt/flash_rt_fa2*.so
flash_rt/flash_rt_qwen3_vl_kernels*.so         <- rope_neox_qk_bf16 在这里
```

`rope_neox_qk_bf16` 由 **`csrc/qwen3_vl_bindings.cpp`** 绑定（不是 `csrc/bindings.cpp`），
所以只查主模块的普查必然漏掉它。仓里已有 4 处 `from flash_rt import
flash_rt_qwen3_vl_kernels as vlk` 的先例（红线 #2）。

**约定核实**（N1.6 文档 §5.2 的警告：做错不报错，只会 cos 掉）——读 `csrc/kernels/rope_neox_qk_bf16.cu`：

| 项 | kernel | shim | 结论 |
|---|---|---|---|
| 配对方式 | `d` ↔ `d + head_dim/2`（rotate_half / NeoX） | 同 | ✅ 与 HF `apply_rotary_pos_emb` 一致 |
| 精度 | `to_f32` → fp32 运算 → `from_f32<bf16>` | 同 | ✅ **舍入点相同** |
| 张量布局 | `(rows, heads, head_dim)`，**允许原地** | `(S, NH, HD)` | ✅ 完全一致 |
| cos/sin | **半宽** `(rows, head_dim/2)` | 全宽 `(S, HD)`（两半重复） | ⚠️ 需切半 |
| Q/K | **一次 launch 同时转 Q 和 K**，`k_heads` 可 ≠ `q_heads` | 每层调 2 次 | ✅ 正好吃掉 GQA |

**微基准（3 个真实形状，先探针后施工）**：

| 位点 | 形状 | kernel vs shim | 两臂 vs fp64 参考 | shim 墙钟 | kernel 墙钟 |
|---|---|---|---|---|---|
| ViT（MHA） | S=512 NH=16 HD=64 | **bit-identical**，cos 1.000000000，max&#124;d&#124; 0 | 同为 0.999998625 | 863.2 µs | **62.9 µs** |
| LLM（GQA）Se=141 | 16Q/8KV HD=128 | **bit-identical**，max&#124;d&#124; 0 | 同为 0.999998632 | 853.3 µs | **33.3 µs** |
| LLM（GQA）Se=148 | 16Q/8KV HD=128 | **bit-identical**，max&#124;d&#124; 0 | 同为 0.999998632 | 852.1 µs | **34.0 µs** |

**逐位相同**，且两臂到 fp64 的距离**相等** ⇒ 不是"谁更准"，是**同一套舍入**。
所以本项**没有精度内容**，它的门是 `torch.equal` 而不是 cos 阈值。

#### 6.22.4 落地

`pipeline_orin.py`：`_rope_neox_qk_kernel()`（**一次性**解析 + 缓存，缺失时播报一次并退回 shim）、
`_rope_qk(Q_t, K_t, tbufs, rows, q_heads, k_heads, head_dim, stream)`（一次 launch 替两次 shim）、
`_rope_half_table(t, what)`（切半 + **校验两半重复**；不满足返回 `None` 退回 shim 并播报一次）。
两个位点分别传 `(S, NH, NH, HD)` 与 `(S, NHQ, NHKV, HD)`。
frontend 在 `set_prompt` 里一次性建 4 张半宽表（prompt 不变量，图会烘住指针），
经两处 `tbufs` 的 `cos_half`/`sin_half` 传入。
**80 次 shim 调用 → 40 次 kernel launch。** shim 保留为**回退路径 + 数值参考**
（删了就没有对照物了）。

#### 6.22.5 结果

**逐位相同（端到端）**，两个数据集各验一次：

| 比较项 | 真机 frame 0 | 仿真 frame 100 |
|---|---|---|
| `backbone_features` `torch.equal` | **True**（max&#124;d&#124; 0） | **True** |
| `infer` 输出 `torch.equal` | **True** | **True** |
| decoded action `torch.equal` | **True**（max&#124;d&#124; 0°） | **True** |
| vs HF fixture | cos **0.999999398** / max **0.4935°**（两臂同值） | cos **0.999999496** / max **0.2467°**（两臂同值） |

**红线 #5 的证据**（相同输出本身不算证据，必须配调用计数）：
kernel 臂一次 backbone **40** 次融合调用、shim 臂 **80** 次 shim 调用、彼此 0 次串台；
首次 `infer(aux=)` 分别 **160 / 320** 次（= 4 × 40 / 4 × 80，因为 `_capture_backbone_graph`
要 3 次 warmup + 1 次捕获）。另加一条**图冻结检查**：把 kernel 臂的前端放在
"解析器返回 None"下重放，结果仍等于它自己的捕获 ⇒ 两张图各自烘住了自己的臂。

**延迟**（配对交替，中位数 of 9，锁频）：

| 口径 | kernel 臂 | shim 臂 | 省下 | 比值 |
|---|---|---|---|---|
| 每观测 `infer(aux=)`（真机 f0） | **98.137 ms** | 106.970 ms | **−8.833 ms** | **1.0900×** |
| 每观测 `infer(aux=)`（仿真 f100） | **97.281 ms** | 107.752 ms | **−10.471 ms** | **1.1076×** |
| eager backbone（仿真） | 51.294 ms | 63.720 ms | −12.426 ms | 1.2423× |
| GPU 时间普查（仿真 backbone） | 50.309 ms | 61.787 ms | **−11.478 ms** | — |

健康检查：shim 臂 107.752 ms vs §6.19 记的 108.55 ms，差 **0.7%** ⇒ 基线没漂。

**分阶段**（`/tmp/stage_breakdown.py` 重跑，中位数 of 7）：

| 阶段 | 前 | 后 | Δ |
|---|---|---|---|
| ViT | 31.42 ms | **22.27 ms** | **−9.15**（48/80 次调用在此，且 S=512 张量最大） |
| LLM | 21.63 ms | **19.00 ms** | **−2.63**（32/80 次） |
| backbone（eager） | 63.03 ms | **51.08 ms** | **−11.95** |
| backbone 的 CPU 提交 | 54.54 ms | **20.82 ms** | **−33.72**（800 次 launch 没了；eager 臂才吃得到） |
| DiT ×4（图） | 34.96 ms | 34.81 ms | 不变（DiT 没有 rope） |
| action enc+dec | 8.48 ms | 8.45 ms | 不变 |
| **边界合计** | **106.47 ms** | **94.35 ms** | **−12.12** |

**倍率**（HF 分母不变：同一边界 360.38 ms、完整 `get_action` 387.93 ms、仿真档 386.27/358.84）：

| 口径 | 前 | 后 |
|---|---|---|
| 口径 A：同一边界 | 106.47 → 3.38× | **94.35 → 3.82×** |
| 口径 C：每观测（仿真，vs 完整 386.27） | 108.55 → 3.56× | **97.28 → 3.97×** |
| 口径 C：每观测（仿真，vs 同一边界 358.84） | → 3.31× | **→ 3.69×** |
| 完整 `get_action` 等价（含未替换的 ~27.55 ms pre/post） | ~134 ms → 2.89× | **~122 ms → 3.18×** |

> ⚠️ 上表最后一行**已被 §6.23 取代**：杠杆 #14 把每观测图像通路换成了设备链，
> 而那个 "~27.55 ms pre/post" 本身是两个独立总量相减的产物（§6.23.6 教训 3）。
> 现在改用**实测**的**口径 D**（两边都含图像预处理）：改前 112.374–115.499 ms →
> **3.344–3.444×**，改后 **99.579–100.428 ms → 3.846–3.896×**。
> 同时 §6.23.1 查明**口径 C 的 3.97× 偏乐观**（分子不含预处理、分母含）。

**measured vs predicted**：归因探针预测 GPU 省 11.902 ms，普查实测省 **11.478 ms（96.4%）**；
每观测实测省 **10.471 ms（88%）**。差的正是融合 kernel 自己的 **1.07 ms**
（no-op 探针把它算成了纯收益）——这是 no-op 归因法的**已知系统性偏高**，量级 3.6%，
比 §6.19.2 那个 2.3× 的隔离微基准偏差小两个量级，因为这次是**同一进程同一条流水线**里量的。

**测试**：dispatch **67 → 80**（+13）。钉的是**契约**而非数值——因为两臂逐位相同，
任何数值门都抓不到接线错误（接错了也是一路相同，直到某天突然不同）：
半宽表必须是**实体拷贝**（kernel 按 `row*(hd/2)+d` 平铺寻址，跨步视图会静默读错行）、
不可用的表退回 shim 且**只播报一次**、GQA 的 `(rows, q_heads, k_heads, head_dim)` 顺序
（16≠8，换位不报错只会让 K 按错步长旋转）、原地（in 指针 == out 指针）、
kernel 缺失/半表缺失都退回 shim 且 kernel 不得执行、解析器**只导入一次**、
`set_prompt` 建齐 4 张半表、两处 `tbufs` 都带上、两个 forward 里**不得再有**直调 shim。
**负控**：把 ViT 位点还原成两行 `_rope_rotate_half` ⇒ 2 条钉子立刻红。

闸门：仿真档 **114 → 127 passed**（66.4 s），真机档 **89 → 102 passed**（55.9 s），
时钟全程 1300500000。**逐位相同 ⇒ 所有精度门数值一个没动。**

#### 6.22.6 同一轮里量到但**否决**的（原则 #13：先微基准）

| 候选 | 实测 | 判定 |
|---|---|---|
| `silu_mul_qwen36_bf16` 替掉 LLM FFN 的 `silu_bf16` + `gate_t.mul_()`（管线里**最后一个** torch elementwise shim） | **逐位相同**，但每层 83.0 → 77.3 µs，×16 层 = **省 0.092 ms/backbone（0.08%）** | ❌ **否决**：低于噪声地板，不值得为它动一条已上线的数值路径 |
| `bf16_nn_bias` / `bf16_nn_bias_gelu` / `bf16_nn_bias_res` 融合掉 `add_bias_bf16`（848 次、**8.880 ms**） | 符号**存在**，但 SM87 上运行时 `code=15`，实测 M=**20 / 128 / 141 / 512** 全失败 | 🔴 **kernel owner 缺口**（红线 #8，只报不改）。这是**当前最大的非 GEMM 单项** |

#### 6.22.7 下一条杠杆：已量到，但需要自己的一道门

`bf16_matmul_cublaslt_bf16`（模块级函数，仓里已有先例：`_higgs_audio_v3_bf16.py`
拿它当 `fast_gemm`、`_qwen3_vl_vision_rtx.py` 也在用）在 backbone 的**真实 GEMM 形状**上
比现役 `bf16_nn` 快 **1.1778×**：

| 形状 | M | N | K | 次数 | `bf16_nn` | cublaslt | 比值 |
|---|---|---|---|---|---|---|---|
| ViT q/k/v/o | 512 | 1024 | 1024 | 96 | 88.5 µs | 73.9 µs | 1.198 |
| ViT fc1 | 512 | 4096 | 1024 | 24 | 189.6 | 187.9 | 1.009 |
| ViT fc2 | 512 | 1024 | 4096 | 24 | 202.0 | 165.5 | 1.221 |
| LLM q/o | 141 | 2048 | 2048 | 32 | 159.5 | 102.6 | **1.555** |
| LLM k/v | 141 | 1024 | 2048 | 32 | 80.1 | 76.8 | 1.043 |
| LLM gate/up | 141 | 6144 | 2048 | 32 | 262.1 | 237.4 | 1.104 |
| LLM down | 141 | 2048 | 6144 | 16 | 265.3 | 219.8 | 1.207 |
| **合计/backbone** | | | | **256** | **38.192 ms** | **32.425 ms** | **1.1778×（省 5.766 ms）** |

与普查对得上（出厂配置的 bf16 ampere GEMM 合计 39.57 ms）。省 5.766 ms ≈ 每观测 **5.9%**。

**为什么本轮没有顺手做掉**——两个未决风险，各自都需要一道门，与 rope 那种
"逐位相同所以零风险"完全不是一类：

1. **不是逐位相同**（7 个形状全部 `torch.equal == False`）：换算法就换累加顺序，
   是 bf16 级别的舍入差。要走完整的逐级 cos 门（含 §6.10.1 那次教训：
   fake-quant 门漏量 VLSA 之后一级）+ E2E decoded 门。
2. **图可捕获性未验**：`max_algos` 默认 0 意味着内部要做 cuBLASLt 启发式查询，
   而原则 #14 记过 cuBLASLt 的 FP8 GEMM **不能被流捕获**（`code=13`）并因此封过整条路的顶。
   bf16 不等于 FP8，但这条**必须先验再说**。

⇒ 记为**下一条杠杆**，不混进本轮改动（两件事一起动会毁掉归因）。

#### 6.22.8 剩余时间去哪了（本轮之后的画像）

出厂配置 GPU 普查 **102.12 → 92.73 ms**：`rope_neox_qk_kernel` 40 次 / **1.020 ms**，
`aten elementwise` **13.088 → 4.241 ms**，`aten cat` **1.642 → 0.059 ms**，
launches **4634 → 3874**。（§6.22.1 那次更正后的复测：**102.257 → 92.439 ms**、
rope 1.053、elementwise 13.216 → 4.268、cat 1.639 → 0.058、launches 4634 → **3798**，
各项差 <0.4%，launch 计数差 76 属 profiler 对图内 kernel 的聚合抖动。）剩下的结构
（**按更正后的分桶**，shipped 臂 92.439 ms）：

| 块 | ms | 占比 | 还能不能动 |
|---|---|---|---|
| GEMM 合计（bf16 ampere 43.953 + INT8 cutlass DiT 21.472） | **65.425** | **70.8%** | DiT 已 INT8；ViT/LLM 的 INT8 都被精度门实测关闭（§6.10.2 / §6.11）⇒ 只剩 §6.22.7 的换算法（**5.766 ms**） |
| `add_bias_bf16`（848 次） | 8.906 | 9.6% | 🔴 kernel owner（`code=15`，已复验 4 个 M） |
| aten elementwise（残余 230 次） | 4.268 | 4.6% | 最大单项是 78 次 × 43.8 µs 的 `unrolled_elementwise`；管线内只剩 `gate_t.mul_()`（16 次），已量到只值 0.092 ms ⇒ 其余在 DiT/frontend |
| norm（385 次） | 3.106 | 3.4% | 已是融合件 |
| activation（`gelu_kernel` 156 次） | 2.503 | 2.7% | 已是融合件 |
| quantize（INT8 per-row，512 次） | 2.305 | 2.5% | 已是融合件；要再省得靠 owner 的 int8-输出 norm/gelu epilogue（§8） |
| **attention（FA2，172 次）** | **2.137** | **2.3%** | **已到底**（⚠️ 原记 1.323 / 1.4%，§6.22.1 的分桶错误） |
| `res_add_kernel`（347 次） | 1.872 | 2.0% | 已是融合件 |
| `rope_neox_qk_kernel`（40 次） | 1.053 | 1.1% | 本轮新落地 |
| conv（cuDNN patch embed，2 次） | 0.467 | 0.5% | §6.7.1(a) 故意保留（展平 GEMM 与 HF 差 ~1 ULP） |
| HF processor pre/post | ~27.55 | — | **口径 C 之外**，但真实部署里是最大单块；§8 已记（`pixel_values` 最重）。⚠️ **§6.23 已把其中最大的一段（每观测图像通路 12.916–14.027 ms）替换成 1.448–1.485 ms 的设备链**，并查明这个 "~27.55" 本身是两个独立总量相减的产物（§6.23.6 教训 3）⇒ 本行只作 §6.22 当时的画像存档 |

**原则对号**：#13（**两个候选都先微基准再决定**：rope 做了，silu_mul 量到 0.09 ms 当场否决）、
#14（对每个复用依赖做"天花板检查"：cublaslt 的图可捕获性列为未决风险而不是假定可行）、
#15（**先重画像再施工**：一上来就发现 FA2 只占 **2.3%**（更正后）、GEMM 占 **64–71%**，
避免把力气花在已经到底的地方；并在结尾给出带预测值的剩余杠杆菜单）、
#16（**瓶颈会移动**：本轮之后 `add_bias_bf16` 升为最大非 GEMM 单项；no-op 归因法系统性偏高 3.6% 已量化）、
红线 #2（复用已构建的 kernel 与既有 `vlk` 导入先例，未写任何新 kernel）、
红线 #4（半表不可用/kernel 缺失 ⇒ 退回慢而正确的一边，且**播报**）、
红线 #7（`bf16_nn_bias` 的 `code=15` 在 4 个 M 上复验；**分桶误归因查出两处并已更正**）、
红线 #8（`add_bias_bf16` 只报不改）、
AGENTS.md §2.1「**先逛再建**」+ §3.6（负控：还原 ViT 位点 ⇒ 2 条钉子红）。

**方法论教训 1（本轮最重要的一条）**：**宣布 `blocked_on_kernel` 之前，必须枚举全部已构建的扩展模块。**
`rope_neox_qk_bf16` 在 `flash_rt_qwen3_vl_kernels` 里躺了整段时间，
而杠杆 #9 因为在 `flash_rt_kernels` 里找不到同名 bf16 条目就被判成缺口、
还顺带把收益低估了 4.4×（2.7 vs 11.9 ms）。**两个错误同源**：都来自"只查了一个模块"。
按 AGENTS.md §0.1 阶段 3 的措辞，`blocked_on_kernel` 是一个**要立刻上报并停手**的判定，
所以它的举证标准必须比一般结论更高——"最近的 Hub API 是什么、为什么不够"这一栏，
得在**所有**已交付工件里找过才填得动。

**方法论教训 2（收尾时自查出来的，§6.22.1）**：**关键字分桶对 C++ mangled 模板名不可靠，
而且会同时朝两个方向错。** FA2 的 `fa2_vendor::flash_fwd_kernel<…cutlass::arch::Sm80…>`
被 GEMM 规则先吃掉（⇒ 注意力记成 0.312 ms / 0.3%，实为 **2.335 ms / 2.28%**，偏低 **7.5×**），
而 attention 规则里的 `"flash"` 又匹配命名空间 `flash_rt::`（⇒ 那 0.312 ms 根本不是注意力）。
**这是同一份普查里第二次栽在分桶上**（第一次是 `DefaultGemmWi…` 被归进 copy/cast）。
**规则：分桶按精确名字前缀、attention 排在 GEMM 之前，并且必须打印每个桶的成员名单**——
两个错都是"总量看着合理"才活下来的。判据也要能自证：更正后 **GEMM 在两臂之间只差 0.12 ms**，
而 rope 改动本来就不该碰 GEMM，这才敢认。
⚠️ 注意**结论没变但余量变了**：FA2 从 0.3% 变成 2.3%，"无可再得"仍成立（清零也只值 2.3 ms），
但**如果当初按 0.3% 去否决一个值 1 ms 的注意力改动，就会错杀**。
**画像数字错了不会立刻现形，它只在未来的排序决策里发作。**

**实验件**：`/tmp/n17_remaining_time.py`、`/tmp/n17_kernel_census.py`（⚠️ 分桶有上述缺陷，
仅存档）、`/tmp/n17_census_corrected.py`（**更正版，两臂同进程**）、`/tmp/n17_attn_bucket_probe.py`、
`/tmp/n17_rope_attribution.py`、`/tmp/n17_rope_kernel_microbench.py`、
`/tmp/n17_rope_kernel_ab.py`（均未进仓；结论已进本节，契约门已进 dispatch）。

---

### 6.23 ✅ 杠杆 #14 落地：每观测图像通路改走纯 torch GPU 链（每观测 −11.95…−15.41 ms）

**触发**：使用方要求"处理 HF pre/post，这部分 Thor/RTX 是如何处理"，并指定两条边界——
"既然当前是 ORIN，就不应该看 NPU/AMD，而尽量参考 N1.6/Pi0.5 的方案，**走纯 torch 后端，
依靠 GPU 加速处理**"，入口形态是"**裸输入**，即输入就是图和 prompt"。

**第一件事仍是核实，而不是照做**。逐符号查了本仓 **7 个** N1.7 CUDA 前端
（Thor、Thor FP8、RTX、RTX FP16/FP8、SM89、AMD、Orin）：**没有一个跑图像通路**——
全都要求调用方交一个已处理好的 `aux` 束，也就是厂商的 `Gr00tN1d7Processor`
**每观测跑一遍**。唯一替换掉它的是 Ascend NPU 通路（本次按要求不看）。
⇒ 这不是"Orin 漏了一项"，而是**整条 CUDA 家族共同的空白**；而 Orin 是它代价最高的地方，
因为这段是纯标量 host 代码，跑在 §6.11.5 量过的那颗弱 ARM CPU 上。
可复用的先例是 **N1.6 `groot_rtx.py:1054/1191`** 与 **Pi0.5 `pi05_rtx.py:1498/2060`**
的裸输入房型（红线 #2）：预分配 buffer + `copy_` 原地写，而不是每帧 `.cuda()`。

#### 6.23.1 先补一个新口径：**口径 D（每观测含图像预处理）**，并纠正口径 C 的一处边界不对称

三臂配对交替（锁频 1300.5 MHz，中位数 of 11，4 帧真机数据 × 两个数据集），
**vendor 臂自证**：它产出的 `pixel_values` 与磁盘 fixture **bf16 逐位相同（4/4 帧）**
⇒ 这个臂就是 HF 的原路径，不是近似。

| 数据集/帧 | 预处理 vendor / cv2 / **gpu**（ms） | 每观测 vendor / cv2 / **gpu**（ms） | vendor→gpu 省 | cv2→gpu 省 |
|---|---|---|---|---|
| 真机 frame 0 | 13.072 / 6.185 / **1.483** | 112.814 / 104.664 / **99.579** | **13.235 ms（1.1329×）** | 5.085（1.0511×） |
| 真机 frame 300 | 13.509 / 6.310 / **1.485** | 112.644 / 105.431 / **99.912** | **12.732 ms（1.1274×）** | 5.519（1.0552×） |
| 仿真 frame 100 | 12.916 / 6.239 / **1.452** | 115.499 / 105.504 / **100.089** | **15.410 ms（1.1540×）** | 5.415（1.0541×） |
| 仿真 frame 107 | 14.027 / 6.219 / **1.448** | 112.374 / 105.289 / **100.428** | **11.945 ms（1.1189×）** | 4.861（1.0484×） |

由此得到**口径 D**（一帧原始 uint8 图像进、一个动作出，两边都含各自的图像预处理）：

| | HF eager 完整 `get_action` | FlashRT 改前（vendor 图像臂） | **FlashRT 改后（`frames=` 臂）** |
|---|---|---|---|
| 真机 frame 0 | 387.93 ms | 112.814 → 3.439× | **99.579 → 3.896×** |
| 真机 frame 300 | 387.93 | 112.644 → 3.444× | **99.912 → 3.883×** |
| 仿真 frame 100 | 386.27 | 115.499 → 3.344× | **100.089 → 3.859×** |
| 仿真 frame 107 | 386.27 | 112.374 → 3.437× | **100.428 → 3.846×** |

⚠️ **口径 C 有一处边界不对称，本节补账**（与 §6.7 当初把口径 A 拆成 A/B 的原因同类：
**分子分母的边界不一致**）。§6.22 记的 **97.28 ms → 3.97×** 是**模型侧**每观测
（`aux` 从磁盘 fixture 来，**不含**图像预处理），而它对比的 HF 386.27 ms 是
**完整 `get_action`，含 HF 自己的图像预处理**。口径 D 把两边都补齐后，
倍率是 **3.85–3.90×**（改后）——**这才是该主张的数字**，3.97× 是偏乐观的。
（口径 D 里仍留着 §6.15 已入账的一项 FlashRT 优势：HF 每观测付
`_apply_vlm_processing` + tokenizer **2.74 ms** 与 `decode_action` **0.24 ms**，
FlashRT 把前者提到了 `set_prompt`；合计 ~2.98 ms ≈ 3.0%。这一项**不是**本次的功劳。）

**每观测的完整拆分**（同进程、三臂在**同一次交替**里量、中位数 of 11、
4 帧真机数据 × 两个数据集）。`pre + model − whole` 的残差是
**+0.391 / +0.073 / +0.277 / −0.119 ms ≈ 0** ⇒ 这三段确实**相加**，不是彼此重叠：

| 段 | ms | 占每观测 |
|---|---|---|
| **前处理合计**（`_pixel_values_from_frames`） | **1.768–2.567** | **1.8–2.6%** |
| ├ pinned H2D staging（`_frames_to_device`，隔离量测） | 0.286–0.316 | 0.3% |
| └ 设备链（letterbox→AREA→裁剪→放大→归一化→patchify，隔离量测） | 1.455–1.496 | 1.5% |
| **模型合计**（`infer(aux=)`，喂**同一份** GPU `pixel_values`） | **97.375–97.773** | **97.4–98.0%** |
| **每观测墙钟**（`infer(frames=)`，口径 D） | **99.626–100.015** | 100% |

模型那 97.4–97.8 ms 再拆（自重时间探针，每条边一次 sync，**按每次调用**归一；
探针自身只抬高 **+0.2…+0.7%**，所以这个拆分可以直接读，不必只当比例）：

| 段 | ms | 占每观测 | 与既有账面 |
|---|---|---|---|
| `run_backbone_graph`（ViT 24 层 + DeepStack + LLM 16 层 + VLSA + 融合，**一张图**） | **48.614–49.319** | **48.6–49.1%** | 杠杆 #9 之后的图臂；与 §6.22.5 的 backbone **51.08**（eager 口径）同量级 |
| DiT ×4 步（graph replay）+ action encoder/decoder + 内联胶水 | **42.992–43.367** | **42.9–43.3%** | 与 §0.1 的 action head **43.75** 一致 |
| `_refresh_dit_cross_kv`（每观测**原地**刷新 cross-K/V） | **5.193–5.348** | **5.2–5.3%** | §6.19 记的 **4.75** 是隔离口径 |
| `_pixel_values_from_frames`（**自重**：设备链 + plan 查表 + 形状核对） | 1.567–1.633 | 1.6% | — |
| `_run_state_encode` | 0.570–0.609 | 0.6% | — |
| `_frames_to_device`（pinned H2D，**流水线内**） | 0.492–0.643 | 0.5–0.6% | ⚠️ 比隔离量测的 0.286–0.316 高 ~0.3 ms，又是原则 #16 的隔离/流水线之差 |
| `_observation_aux` + `_validate_observation_contract` | **0.139–0.160** | **0.2%** | 这就是**身份快路径**的全部代价（§6.23.4：0 次 `torch.equal` / 0 字节） |

⇒ **前处理 1.8–2.6%，推理 97.4–98.0%**。推理内部
**backbone 图 : action head ≈ 48.9 : 43.2**，与口径 A 的边界
（backbone 51.08 + action head 43.75 = **94.35**）对得上；
每观测比边界多出的 **~5.8 ms** 就是 cross-KV 刷新 5.2 + state encode 0.6，
正是 §6.15 记的那两项（口径 C 与口径 A 的差）。
📌 本节先前写的"图像通路占比没有单独拆、1.4–3.2 ms 只作量级参考"
是**跨进程相减**推出来的，**已由上面这张同进程实测表取代**。

#### 6.23.2 什么**逐位精确**、什么不精确（含一个**可证**的无平局条件）

厂商 eval 链（`use_albumentations=True`）是：LetterBoxPad 成正方形 →
`SmallestMaxSize(interpolation=3)` = **`cv2.INTER_AREA`** 到 `shortest_image_edge=256` →
`FractionalCenterCrop` 到 `int(256×0.95)=243`（偏移 `(256−243)//2=6`）→ INTER_AREA 回 256。
然后 HF 的 `Qwen2VLImageProcessorFast` 做 rescale(1/255) + normalize(mean=std=0.5) + patchify。
两个参数**都从 checkpoint 自己的 `processor_config.json` 读**，不用代码默认值：
`crop_fraction=0.95`（**不是** 0.9）、`shortest_image_edge=256`；
`image_crop_size=[230,230]` **未被使用**。这条链是**确定性**的
（`FractionalCenterCrop` 是中心裁剪；带 `np.random.randint` 的那个是
`FractionalRandomCrop`，train-only）——探针 40 次抽样只返回一个 `(6,6,249,249)`，
且对 `np.random.seed` 不变。

**第一步（640→256 缩小）逐位精确，而且是可证的**，证明对任意几何都成立，
所以它是**被检查**而不是被假设的。把尺度写成最简分数 `s = src/dst = p/q`，
连续覆盖给出的权重是 `m_i/p`（`m_i` 是求和为 `p` 的非负整数），
于是一个像素的精确输出是 `N/p`（`N` 为整数）。**出现四舍五入平局**需要
`N/p = k + 1/2`，即 `2N = p(2k+1)`；**`p` 为奇数**时这迫使 `p | N`，
使 `N/p` 是整数而永不是半整数 ⇒ **任何输入都没有平局**，
fp32 误差（~1e-4，对平局余量 `1/(2p)`）就不可能翻掉一次舍入。
出厂的 640→256 尺度是 **5/2**：权重恰为 {0.2, 0.4}、3 抽头、`out = (a + 2b + 2c)/5`。

| 检查 | 结果 |
|---|---|
| vs `cv2.INTER_AREA`，4 帧真机 × 2 数据集（各 393216 像素） | **max = 0**（numpy fp64 与 torch fp32/CUDA 各一遍） |
| 穷举：每个不同抽头模式的**全部** 256³ 输入（2 模式 × 16,777,216 = **3350 万**次） | **0 次舍入分歧** |
| 三个缩小尺度（640→256 = 5/2、320→256 与 40→32 = 5/4）× fp32/fp64 × 随机图/结构图 | **max = 0** |
| **负控**：把 `A1` 一个权重扰动 1e-3 | 穷举分歧 **> 0** ⇒ 上面那条门是活的 |

`build_image_plan` 把该条件**记录**成 `shrink_p` / `shrink_exact`，
`frames_to_resized` 在不成立时**拒绝执行**（红线 #4）——例如一台 512 宽的相机给出
`p = 2`，此时 `out = (a+b)/2` 在每一个奇数和上都平局，而 OpenCV 的平局破法
不是本实现所复现的那一种。**fp16/bf16 权重不精确**（max = 1）、
**TF32 也不精确**（10 位尾数给出 ~0.25 的误差，对 0.1 的平局余量），两者都被拒。
`_MAX_EXACT_NUMERATOR = 1000` 的来历：平局余量 `1/(2p)`，fp32 在 255 处一个 ULP 是
3.0e-5 ⇒ 需要 `p < 5000`；界取在其中**一个数量级以内**。

⚠️ 顺带量到：**torch 自己的 `interpolate(mode="area")` 偏 18 LSB**——
它用整数源边界加一个普通均值，而不是分数覆盖权重。**不能用它替代**。

**第三步（243→256 放大）不精确**，且**残差是算法性的、不是精度性的**：
**fp64 给出完全相同的残差**。真机帧上 **max = 1 LSB、mean 0.051–0.069、
5.1–6.9% 的像素**。试过 4 种表述——连续覆盖权重、OpenCV 的 1/2048 两抽头定点、
各自带与不带中间移位，配 round / trunc / round-half-even，fp32/fp64/int32——
**全部收敛到 max = 1**。出厂用的浮点覆盖形式**差异像素占比最低**
（合成图上 12.2% 随机 / 11.5% 渐变 / 13.8% 棋盘，定点两种形式是 21.0% 与 77.7%）。

🔴 **本节推翻了自己先前的一个归因**（教训见 §6.23.6）。先前写的是"残差在 OpenCV
放大分支的权重/抽头推导里"。**一个 one-hot float32 探针证伪了它**：
one-hot 输入能把权重单独隔离出来，实测 **OpenCV 的有效权重就是连续覆盖**，
在四个尺度上都吻合到 **7.2e-8**。所以残差在 OpenCV 的
**uint8 累加/舍入路径**上，与权重无关。这里选择**声明未知**而不是编一个说法。
（奇偶论证也救不了它：243/256 的分子是奇数 ⇒ 连续算子无平局，
OpenCV 只是在放大时**用了不同的舍入**。）

**patchify 在 bf16 下逐位精确**：Qwen2-VL 的 merge-block 洗牌
（patch 16 / merge 2 / temporal 2 → `(512,1536)`），`torch.equal` vs HF processor，
**0/786432 不同**，4 帧全对。fp32 下差 **5.9e-08**（`/255` 与 `*1/255` 的顺序），
在模型真正吃的那次 bf16 cast 下消失。`image_grid_thw = [[1,16,16],[1,16,16]]`；
`visual_pos_masks == (input_ids == 151655)` 在 4/4 fixture 上成立。

⚠️ **矩阵结合序是承重的**：`(A@x)@Aᵀ` 与 `A@(x@Aᵀ)` 的 fp32 舍入不同，
真机帧上实测移动 **393216 个 uint8 像素中的 2 个**（各 1 LSB，下游 786432 个 bf16
元素中的 4 个）。量级很小但**不是零**，所以出厂形式被写进 docstring 钉住，
不许顺手"整理"。

#### 6.23.3 三臂精度门（**12 帧真机数据** × 3 臂）与 `THR_IMAGE_GPU_BACKBONE = 0.985`

三臂 = HF fixture、一条 **G6.3 证明与 HF 逐位相同**的 cv2 链、GPU 链。
**cv2 臂在每一帧每一档都与 HF 臂九位小数全同** ⇒ 下表里 T1 的移动**全部是 GPU 链的**，
不是测试自身的抖动（这一点很要紧，否则无法归因）。

| 档 | HF fixture / cv2 臂 | **GPU 图像臂** | 门 |
|---|---|---|---|
| T1 `backbone_features` | 0.996034 … 0.998082 | **0.990967 … 0.996485** | `THR_IMAGE_GPU_BACKBONE = 0.985`（新） |
| T3 velocity | 0.999417 … 0.999576 | **0.999285 … 0.999538** | `THR_CONSUMED = 0.999`（**未放宽**） |
| T4 decoded cos | ≥ 0.999999073 | **≥ 0.999999095** | `THR_CONSUMED = 0.999`（**未放宽**） |
| T4 max 误差 | 0.1936 … 0.4935° | **0.1868 … 0.4935°** | — |

**只有 T1 被放宽，且放宽的量有出处**：那个 ≤1 LSB 要穿过一座 24 层 bf16 残差塔
才被人消费——这正是 `THR_FUSED_CONSUMED`（§6.7.2）已经记过的同一机理
（中间量动了、被消费的输出没动）。T3 的变化跨度是 **−0.000087 … +0.000153**，
**有时反而更好**；T4 的 max 误差 **4 帧更好 / 4 帧更差 / 4 帧相等**——**对称**，
即无系统性退化。这是 §6.20 那条教训（n=4 不足以做方向性判断）的**执行**：
**n 从 4 提到 12**，而且只主张"无系统性退化"，不主张"更好"。
门值取在**最坏实测帧再往下一整个观测跨度（0.0055）**，即 0.985。

**负控（G6.5）确实能红**：把链条换成"只缩小、跳过中心裁剪与放大"
（形状对、值域对、图像看着合理），T1 **0.9918 → 0.8989** 与 **0.9936 → 0.9002**
（好值 0.9909，门 0.985）⇒ 掉穿门 **0.086**。T3 同时退化。
⚠️ 这个负控**必须**打在内层档上：§6.15.3 已经量过，这个 checkpoint 上
decoded action 由 state 主导、视觉上下文只值 ~1°，
一个**滞留的 cross-K/V** 都还能拿到 cos 0.999991673 > 0.999。

**G6.3 另钉了一条容易漏的前提**：fixture 把 **bf16 量化过的值存在 fp32 张量里**
（测试先断言 `torch.equal(want, want.to(bfloat16).to(float32))`）。
所以对着它做 **fp32** 比较会额外背上**半个 bf16 ULP（1.930e-03）的 fixture 自身量化**。
第一版把预算写成 fp32 对 fixture，实测 9.773e-03 撞 7.843e-03 的界——
**差的正好是 1.930e-03**。改法：GPU-vs-cv2 在 fp32 比、GPU-vs-fixture 在 bf16 比。
**规则：对着参考设容差之前，先声明参考自己的精度。**

#### 6.23.4 落地（**additive**，红线 #1）与红线对号

新模块 `flash_rt/frontends/torch/_groot_n17_preprocess.py`，
`_groot_n17_fusion.py` 的兄弟，照抄它的房型（前导下划线文件名、
`from __future__ import annotations`、显式 `__all__`、长叙事 docstring 带实测数字、
惰性 `import torch`、**解释而不是猜**的 `ValueError`）。导出
`area_operator` / `ImagePlan` / `build_image_plan` / `patchify` /
`frames_to_resized` / `frames_to_pixel_values` / `host_reference_chain` /
`shrink_scale_is_exact`。

前端只加不改：`infer(state, *, frames=None, aux=None, …)` 新增一个**关键字-only** 参数。

| 机制 | 做法 | 出处 |
|---|---|---|
| 几何从 checkpoint 读，不硬编码 | `_read_processor_geometry()` 读 `processor_config.json`；读不到 / 缺 `shortest_image_edge`·`crop_fraction`·`use_albumentations` ⇒ **RuntimeError**；`use_albumentations=False` ⇒ **NotImplementedError**（不猜另一条链） | 原则 #6（权重/配置以文件为准） |
| plan 按几何缓存 | key = `(h, w, shortest, crop_fraction)`，一次构建，与 `set_prompt` 建 rope 半表同一生命周期 | 红线 #2 |
| **每观测 host 比较归零** | `_observation_aux` 把契约里每个非 `_shape` 项**按身份**重新呈现（`base[name] = entry["source"]`）⇒ 命中校验器的 identity 快路径 | 实测见下 |
| H2D 走 pinned 常驻 buffer | `_frames_pin` / `_frames_dev` 都 `copy_` 原地写 + `non_blocking=True` | `pi05_rtx.py:830-836/2584-2594` 房型 |
| 图像通路**在任何 CUDA graph 之外** | plan 是实例属性，但帧是每观测的 | §6.17 的纠正：每观测的**源**不得烘进捕获 |
| 歧义**拒绝** | `frames=` 与 `aux["pixel_values"]` 同时给 ⇒ `ValueError`（"两个答案，本前端不替你选"） | 红线 #4 |
| dict-of-views **拒绝** | `pixel_values` 的行序是**按视角**的，dict 的键序不是本前端能核对的相机顺序；猜错会把腕部相机喂到前视位置**且照样产出一个形状正常的动作** | 红线 #4 |
| uint8 **强制** | 传进一个已归一化的 float 帧会被**变换两次** | 红线 #4 |
| 未融合时**拒绝** `frames=` | 没有融合时 backbone 还要吃 `llm_input_embeds`，那是一次 LLM forward，图像通路产不出来 | 红线 #4 |

**每观测 host 比较实测**（数 `torch.equal` 调用次数与比较字节数）：

| 臂 | 每观测 `torch.equal` 调用 | 每观测比较字节 |
|---|---|---|
| `aux=`（调用方自己重建 bundle） | **5** | **77156** |
| **`frames=`** | **0** | **0** |

比预测的还好：调用方每观测重建 bundle 时**没法在身份上匹配**，
而 `frames=` 臂的这些张量**直接来自契约本身**，所以身份天然成立。
测试里用一个**记录型 stub** 把这条钉住，并且带一个对照（重建过的 bundle **确实**掉进慢路径），
房型是 `_RecordingRope`。

**红线 #5（相同输出不是证据）**：`frames=` 与 `aux=` 装同一份 GPU `pixel_values` 时
`torch.equal = True` ⇒ 数值上无法区分"GPU 链跑了"和"回落到 cv2 参考"。
所以 G6.6 用**调用计数**钉：3 次观测里 `frames_to_pixel_values` **3** 次、
`host_reference_chain` **0** 次、`_image_plan` 构建 **1** 次；
并额外断言 GPU 臂的 `pixel_values` **与 fixture 不同**（因为它带那 ≤1 LSB）——
静默回落到 cv2 参考会**同时**在计数和这条相等性上红。

**故意不做的**：把稠密 fp32 matmul 换成 gather/稀疏形式。`A1` 在 640 上是 **3-稀疏**的，
换成 gather 能砍掉 ~200× 的 FLOPs，但**精确性是对稠密 fp32 matmul 验证的**，
换一个累加顺序就可能动最后一个 ULP。在 1.5 ms 的量级上不值得冒这个险；
记为一条**必须先重验精确性**的后续可能项。
同理**没写任何新 kernel**（红线 #8）：缩小步是一次 fp32 matmul、patchify 是一次 reshape，
都不需要。

**天花板检查**（AGENTS.md §2.4 / 原则 #14）：`fvk.patch_im2col_uint8` 在 SM87 构建里**存在**
但**不可用**——`csrc/kernels/patch_embed.cu:64` 硬编码 `total = nv * 256 * 588`
（SigLIP 14×14×3）并经 LUT 输出 **fp16**；N1.7 要的是 **1536 = 3×2×16×16** 列、
Qwen2-VL 的 merge-block 行序、**bf16**。torch 版 0.44 ms 已经够快 ⇒ 只报不做（进 §8）。

#### 6.23.5 measured vs predicted（原则 #15）

| 项 | 预测 | 实测 | 判读 |
|---|---|---|---|
| 预处理省下的时间 | ~8.6 ms（由 ~10.3 ms 的账面估计推） | **11.945–15.410 ms** | **预测偏低 1.4–1.8×** |
| 偏低的根因 | — | 那个 ~10.3 ms 是**分段中位数相加**得来的；整链端到底端量是 **12.916–14.027 ms** | 见 §6.23.6 教训 3 |
| 分段之和 | 3.87 + 5.27 + (2.02…5.30) = **11.16–14.44 ms** | 整链 **12.916–14.027** | **落在区间内** ⇒ 分段归因可信，只是不能当总量用 |
| 设备链 | 1.48–1.53 ms（隔离）/ 1.59–1.84 ms（含 pinned H2D） | 隔离 **1.448–1.485**；配对交替 **1.803–1.854** | 隔离与配对差 **0.36 ms** |

⚠️ 最后那行是**原则 #16 的又一次实例**，而且这次两个方向都出现了：
同一个函数、同一种计时法（`perf_counter` + `synchronize`），
隔离重复量到 **1.448–1.485 ms**，而**与一条 6 ms 的 cv2 host 链交替**量到
**1.803–1.854 ms**（L2 变冷 + host 链的张量分配把 allocator 搅乱）。
**门里断言的是配对交替那个（较保守的）数**，因为 AGENTS.md §3.8 要求 A/B 必须配对交替；
隔离那个只用来做归因。

#### 6.23.6 教训（五条，其中前两条是两个探针 harness bug）

**教训 1（本轮最贵的一条）：精确性探针必须自带一个"答案已知"的用例。**
放大步的探针第一版写成 `A @ b.transpose(0,2,1)`，
而测试图是**方图** ⇒ 转置被静默吸收，在**已知逐位精确**的 640→256 缩小用例上
报出 **max = 155**。因为这个已知用例被当成 harness 自检跑了一遍，才当场抓住。
**规则：任何"我的实现 == 参考实现"的探针，先在一个已知答案的用例上跑，
否则你量的是 harness 而不是实现。**

**教训 2：探针必须调用出厂的那个函数，不能重新推导它的算术。**
第二版 harness 在**两次 matmul 之后各舍入一次**（双重舍入），
于是在 320→256 上报 **max = 1**，而出厂的单次舍入形式给 **max = 0**。
一个不存在的残差差点被写进文档当成"放大步在别的尺度上也差 1 LSB"。
**规则：探针 re-import 被测函数，而不是照抄一份。**

**教训 3：分段中位数相加不是量测。** 本次计划里把 ~10.3 ms 当成账面数字，
而整链实测 **12.916–14.027 ms**（偏低 **1.25–1.36×**）。
本文档早就有同一个毛病的另一面：那个"~27.55 ms pre/post"
是**两个独立量测的总量相减**得来的产物。
**规则：总量引用端到底端的口径，分段只用于归因。**
（分段之和 11.16–14.44 确实**包住**了整链实测值 ⇒ 分段本身没错，错在拿它当总量。）

**教训 4："为什么有残差"这种归因要探针，不要合理的故事。**
"残差在 OpenCV 放大分支的权重推导里"这句曾经**已经写进模块 docstring**，
读起来完全合理，而且是**错的**：one-hot 输入能单独隔离权重，
一跑就显示 OpenCV 的有效权重**就是**连续覆盖（吻合到 7.2e-8）。
真相是 uint8 累加/舍入路径，而且 **fp64 下残差完全相同 ⇒ 算法性而非精度性**。
**规则：写下"因为 X"之前，做一个只有 X 为真才会给出这个结果的实验。**
本节的处置是**声明未知**——4 种表述全部收敛到 max=1，
出厂那种的差异占比最低，但没有一个能复现 OpenCV 的舍入。

**教训 5（又一次测量卫生事故，与 §6.21.3 同类）：跑门禁套件时不许编辑被测源文件。**
仿真档首轮报 `test_the_eager_backbone_arm_stays_reachable` 失败，
而**单独跑通过**、三条断言逐条手工验证也都成立。根因是**我在套件运行途中改了
`groot_n17_orin.py` 的两处 docstring**：那条测试用 `inspect.getsource(CLS.infer)`
做源码钉子，而 `inspect` 走 `linecache`，**会按 mtime 重新读盘**；
进程里的 code object 仍带着**改动前**的 `co_firstlineno`，
docstring 多了 3 行 ⇒ 抽出来的源码块整体错位 ⇒ 钉子里的字符串不在块里。
**这不是缺陷，是我自己造的假失败**，但它长得和真失败一模一样，
而且**只在做源码钉子的测试上出现**（数值测试完全不受影响）——
正是本仓大量使用源码钉子（§6.22 教训 3）带来的一个新失效面。
**规则：门禁套件运行期间只许改 `.md`；改完 `.py` 必须重跑。**
（重跑后仿真档 **204 passed**，见 §7 第 17 条。）

**顺带记一条 torch 2.3 的坑**（都实测踩过）：
`Tensor._version` 是**会抛异常的 property**（inference tensor 上），
`np.abs` 作用在 uint8 差值上会**模 256 回绕**（一次"max=255"其实是比对 bug），
`np.arange(n, dtype=np.uint8) * 255` **溢出**（"渐变图"其实是回绕噪声——
这一条是被负控自己揭发的：扰动只让输出动了 0.12 而不是 2.5）。

#### 6.23.7 Stage 2（HF-free `set_prompt(str)`）：**已定位，被阻塞，且没有延迟收益**

使用方要的"输入就是图和 prompt"这个字面契约，还差 prompt 那一半。
**但 prompt 侧的一切都是 prompt-scoped，`set_prompt` 已经把它提走了**
（`input_ids` / `attention_mask` / `image_grid_thw` / `embodiment_id` / rope 表），
所以 Stage 2 买的是**部署独立性**（原则 #10：加速路径必须能在厂商训练代码不在场时跑起来），
**不是速度**。

🔴 **阻塞项（本次实测发现，不是推测）**：`flash_rt/models/groot_n17/mrope_table.py`
**既是休眠的、又是错的** —— 对着 4/4 fixture 捕获的 `rope_cos`/`rope_sin`，
maxdiff **2.2e-02 … 3.1e-02**（bf16，约 **5–10 ULP**）。
而 `tests/_helpers/groot_n17/mrope_ref.py` 实现的是**同一个**
`apply_interleaved_mrope`（`slice(offset, mrope_section[axis]*3, 3)`），
所以它那句"对 HF 验证过逐位精确"是在**旋转后的 Q/K** 上做的，**不是在表上**。
⇒ 表从来没被验证过。

**已定位的诊断路径**（下次接着做）：`mrope_section = [24,20,20]` 铺在
`head_dim/2 = 64` 个槽上时，T 写 `slice(0,72,3)`（**22** 个槽）、
H 写 `slice(1,60,3)`（20）、W 写 `slice(2,60,3)`（20）——
**64 个里覆盖了 62 个**，把槽 60/61/62 留在了初始 `clone()` 的 T 轴上。
**逐列**比对捕获到的表，就能立刻看出差异是不是这三个槽（或者轴分配）。
另需：`grid_for(H,W)` 产 `image_grid_thw`、
`visual_pos_masks = (input_ids == 151655)`（**已验证**）、
以及每 prompt 一次 tokenizer 调用（模板 + 确定性的 `<|fim_prefix|>` → 64 份展开）。

**本节产物**：`flash_rt/frontends/torch/_groot_n17_preprocess.py`（新）、
`groot_n17_orin.py`（`_read_processor_geometry` / `_image_plan` / `_frames_to_device` /
`_pixel_values_from_frames` / `_observation_aux` + `infer(frames=)`）、
`tests/test_orin_groot_n17_preprocess.py`（新，**65 条，CPU-only，2.96 s**）、
`tests/test_orin_groot_n17_precision.py` 的 **G6 段（+12 条）**。
测试：dispatch 不变 **80**、precision **17 → 29**、真机档 **102 → 179 passed（71.9 s）**。
未进仓的探针：`/tmp/n17_{vendor_arm_ab,enlarge_exact_probe,tier_table,frames_arm_smoke,slowpath_count}.py`、
`/tmp/verify_preprocess_module.py`。

### 6.24 ✅ 三方审阅的六项发现：**先逐项复验为真，再修**（全部复现，0 假阳性）

**触发**：使用方要求对已实现的 Orin × N1.7 代码做审阅（原仓风格 / 测试验证 / 推理准确性），
审阅产出 7 项；使用方的边界是"**1~6 需要实现/补充，但还需要再次确认 1~6 内的问题是真实存在的，
即再次验证后再开始修复。对于 7，咱不该动**"。
⇒ 本节的结构就是这条边界：**Phase V（复验，不改一行代码）→ Phase F（只修复现的项）**。
第 7 项（把继承来的同名缺陷上报给 kernel/前端 owner）**按要求完全没动**：
`groot_n17_thor.py` / `groot_n17_rtx_fp16.py` / `groot_n17_rtx_fp8.py` / `groot_n17_rtx_sm89.py`
与 AMD 前端一行未改，§8 也没有为它们加条目。

**为什么复验不是走过场**：7 项里有 2 项来自子代理读代码，而它给出的一个支撑数字**本身就是错的**——
它把 eager 回退的代价写成"每次推理 ~600 MB cast 流量"，但 `_compute_dit_adaln_modulators`
是**每去噪步跑一次**，所以那个数是**每步**值，真实值要 ×4。红线 #7 不允许把没量过的数字写进仓库。

#### 6.24.1 Phase V 复验结果：六项**全部复现**，其中一项的量级被审阅低估了 4×

| # | 发现 | 复验探针 | 复现？ | 实测 |
|---|---|---|---|---|
| **1** | DiT CUDA graph 把 `action_horizon` / `num_timestep_buckets` 按值烘死，覆写后**静默重放旧图** | V1a/V1b/V1c/V1d，真 fixture、锁频、一次一个前端 | ✅ **四条全中** | 见 §6.24.2 |
| **2** | `infer` 的图→eager 回落**一声不响**，且代价没人量过 | V2：数告警 + 配对交替计时 + 按真实权重形状重算 cast 流量 | ✅ | **0 条告警**；**+64.120…+67.222 ms（1.6564…1.6855×）** |
| **3** | 前端模块 docstring 自称"只覆写带 dtype 的方法" | V3：逐个列 MRO 上的 override，不靠印象 | ✅ **且比审阅说的更严重** | **16/16 全覆写、从直接父类继承 0 个**；另**新增 28 个** |
| **4** | `:1782` 指向一个不存在的"redundant-weight-memory TODO" | V4：全树 grep + 找 §8 里真实的条目 | ✅ | 真身是 §8 的 `- [ ] 清理冗余权重内存（§6.6.6 第 3 条 + §6.9.6）` |
| **5** | `lerobot_video.py` 的**索引→帧映射完全没测** | V5：列出被测面 vs 未测面 | ✅ | 5 条测试**全部只驱动 `_decode`**；`load_frame` / `_episode_for` / `task_for_frame` / `_frame_index` / `_resolve_column` / `metadata` **零覆盖** |
| **6** | 注意力后端与融合模块**没有 CPU 契约测试** | V6：grep 谁 import 了它们 | ✅ | **没有任何测试 import `attn_backend_groot_n17_orin`**；融合只被 `test_fusion_reproduces_hf_embeds` 门住，而那条要 **GPU + checkpoint** |

**审阅的那个 600 MB 数字，按真实权重形状重算**（红线 #7）：DiT 32 层 × `(1536, 3072)` bf16
⇒ **576.38 MiB/步**，而 `num_inference_timesteps=4` ⇒ **2305.50 MiB/次推理**。
审阅记的是**每步**值，**偏低 4×**。

#### 6.24.2 第 1 项（最关键）：静默重放旧图，实测偏到 **21.49°**，且不报错

`_capture_dit_graphs` 把 `Sa = action_horizon + 1` 烘进 `_dit_dims(Sa)` 与 `_build_dit_attn(Sa)`，
调制器按 `num_timestep_buckets` 预计算；而修复前**唯一的重放护栏**是
`if len(graphs) != num_inference_timesteps: graphs = None` —— 只查步数，不查另外两个。
`action_horizon` 在三个测试文件里出现 **0 次**，而 §6.16 的对齐表把这三个覆写都标成 ✅。

| 探针 | 场景 | 结果 |
|---|---|---|
| **V1a** | 捕获 @ `ah=20`（`Sa=21`），再请求 `ah=40`（`Sa=41`） | **max&#124;d&#124; 0.375 rad = 21.4859°**，**什么都没 raise** |
| **V1b** | 出厂顺序：捕获 @ `ah=40`，再请求 `ah=20` | **0.0586 rad = 3.3572°**，**不报错**；同 horizon 时 graph 与 eager **逐位相同**（G5 原有主张成立） |
| **V1c** | **决定修法的那一条**：eager 臂自己有没有 horizon 状态？ | **没有**——Sa=41 的前端跑 eager@20 与 Sa=21 的前端跑 eager@20，**max&#124;d&#124; = 0** ⇒ 顺序无关 ⇒ **warn + 回落是安全的**（`_rope_qk` 先例）。附带发现：`_run_dit` 的 `hasattr` 缓存让 Sa=21 的前端跑 eager@40 抛一个**难懂的 RuntimeError** |
| **V1d** | `num_timestep_buckets=200`（本 checkpoint 出厂是 1000，属**潜伏**） | **0.0625 rad = 3.5810°**；`_step_temb[0]` 偏 **1.093**。`_step_shifts` 是 `hasattr` 缓存的 ⇒ 后续调用**修不回来** |

**V1c 是这一项里唯一决定性的探针**：如果 eager 臂也带 horizon 状态，回落就只是把"错得静默"
换成"错得响亮"，修法必须是 raise；实测它与顺序无关，所以 warn + 回落既正确又保住了性能可解释性。

#### 6.24.3 第 2 项：回落代价**实测**，不继承审阅的估计

配对交替（锁频、中位数）：图臂 **98.068 ms** vs eager 臂 **165.290 ms**
⇒ **+67.222 ms（1.6855×）**；第二次跑 **+64.120 ms（1.6564×）**。
⇒ **告警文案里写的是区间 `+64…+67 ms（1.66-1.69x）`，不是单次跑出来的那个数**
（第一版硬写了 1.6855×，第二次跑就变成 1.6564×，属于把一次测量当常量，已改）。
归因：eager 臂每步重算 AdaLN 调制器，**48.021 ms/次推理（12.005 ms/步）**，
外加 **2305.50 MiB** 的 fp32 cast 流量。

#### 6.24.4 Phase F：改了什么，以及**哪道门现在钉住它**

全部改动都在 Orin 自己的 override 里；Thor / RTX-FP16 基类**一行未动**（红线 #1，且第 7 项在范围外）。

| 项 | 改动 | 钉住它的门 | 负控（已跑，已红） |
|---|---|---|---|
| **F1** | `_capture_dit_graphs` 收 `action_horizon` / `num_timestep_buckets` 并转发给 `_precompute_diffusion_modulators`；调制器按 `(steps, buckets)` **键控**而非"有没有"；捕获末尾记 `_dit_graph_params` 三元组；`infer` 的护栏**查整个三元组**，不符就 warn 一次 + `graphs = None` | precision 新增 `test_a_changed_denoising_parameter_bypasses_the_graphs`（对 `action_horizon` / `num_timestep_buckets` 参数化）：断言**重放 0 次**、输出与 eager **逐位相同**、`_dit_graph_params` **没被重捕获**、告警文案在；再做正向对照（三元组相符 ⇒ 重放 4 次、无告警） | 把护栏换回旧的"只查步数" ⇒ **2 failed**：`4 graph replays … the stale graphs were served`。修后 V1 的三条陈旧路径全部 **max&#124;d&#124; 0.000000e+00** |
| **F2** | `_note_dit_graph_bypass()` 一次性 `logger.warning`，文案带 §6.24.3 的**实测**数字与补救办法 | dispatch 新增 `test_the_dit_graph_bypass_is_announced_once_with_its_cost`（连叫三次只出 1 条，且三元组、`1.66-1.69x`、`2305.50 MiB`、`use_dit_graph=False` 都在文案里）；continuous 新增 **C9** `test_the_dit_graphs_are_replayed_on_every_observation`（**重放计数** = 每观测 4 × 3 帧，且捕获计数不动） | C9 的负控就是它自己存在的理由：**C5 只数捕获，看不见回落**。`torch.cuda.CUDAGraph` 是 pybind 对象、**没有 `__dict__`**，`monkeypatch.setattr(graph, "replay", …)` 挂不上去 ⇒ 用代理 list 计数 |
| **F5** | 只加测试，`lerobot_video.py` **一行未改**（400 行不变） | `test_lerobot_video.py` **5 → 20**：合成一个真的 2 episode LeRobot v2.1 录制（`meta/info.json` + `tasks.jsonl` + `episodes.jsonl`、parquet 只有 state/action + 四个索引列、每 (episode, camera) 一个 mp4），钉 `_resolve_column`（裸键 / `.joint` 后缀 / 歧义拒绝 / `None` 直通）、`_episode_for`（跨界 + 越界 `KeyError`）、`load_frame`（选对行、float32、两个相机各自的内容、task）、`_frame_index`（pts 路 + 无 pts/无 time_base 的回落）、`metadata`（采样器要的四列 + 缓存） | **4 个断点各跑一遍**：按行号选行 ⇒ 3 red；`_episode_for` 返回全局索引 ⇒ 4 red；`_resolve_column` 歧义时猜第一个 ⇒ 1 red；`_frame_index` 无视 pts ⇒ 2 red。另有**套件内**负控：只把 parquet 的 `index` 列 +1 ⇒ `load_frame(g)` 必须返回**邻帧**的 state/action/task/像素 |
| **F6** | 只加测试；`attn_backend_groot_n17_orin.py`（98 行）与 `_groot_n17_fusion.py` 的**代码一行未改** | dispatch **80 → 102**：15 条钉 SM87 注意力后端（原生 GQA 槽宽 `_LLM_NHKV=8`、`del _llm_logits/_llm_ctx` 的**响亮失败**契约、`_run_fa2` 按 `q.dtype` 选入口、causal 臂恒取 `fwd_bf16_causal`、`kv_seq≠q_seq` 拒绝、`_check_seq` 越界拒绝、`vit`/`vl_self_attn` 委托父类）+ 6 条钉融合的位置嵌入插值（`torch.equal` 对**独立转写**的双线性参考、merge-block 序 vs raster 序、`t` 轴重复、非平方行数拒绝） | **7 个断点各跑一遍**：GQA 改回 16 ⇒ 1 red；不 `del` ⇒ 1 red；`_run_fa2` 不再按 dtype 分派 ⇒ 2 red；不拒 cross-attn ⇒ 1 red；改成 fp32 算完再round ⇒ 3 red；merge-block 改成恒等 ⇒ 4 red；不拒非平方表 ⇒ 1 red |
| **F3/F4** | 三处注释/文档串纠正（见 §6.24.6） | 源码钉子 + 人工核对 | — |

#### 6.24.5 F6 的融合参考：**独立转写**，并在真表上自证

`fast_pos_embed_interpolate` 的主张是**bf16 逐位相同**（不是"接近"），所以 CPU 参考必须钉同一件事、
不能钉更强的。参考是**从定义写的**，不共享实现代码：四角 gather 用显式 `(row, col)` 索引，
merge-block 重排用显式四重循环——正好是两处可能藏转置的地方。

* 真实 `visual.pos_embed.weight` 实测 **`(2304, 1024)`**（side 48，checkpoint 里存 F32，
  Orin 的 weight spec 把每个 `ToFp16` 改写成 `ToBf16` ⇒ 运行时 bf16），
  **absmax 32.75 / std 0.6019**。
* 参考 vs 实现：**max&#124;d&#124; = 0.0**，`torch.equal = True`（真表与合成替身表**都是**）。
* fp32-then-round 变体 vs 实现：**max&#124;d&#124; = 0.125**（= 该量级下 1 bf16 ULP），
  **36.4% 的元素不同** —— 与模块注释里记的那个数**独立复现一致**。
* 合成替身表**按真表的量级分布造**（std 0.6 的主体 + 1% 行 ×12 的重尾）。
  这不是装饰：1 bf16 ULP 在 &#124;x&#124;≈32 是 0.125、在 &#124;x&#124;≈0.6 只有 ~0.004，
  **纯 `randn` 替身会让 fp32-vs-bf16 这个负控小到不足以当证据**。

#### 6.24.6 F3/F4：三处纠正（其中一处是**做 F6 时新发现的**）

1. `groot_n17_orin.py` 模块 docstring 原写"subclasses the RTX full-FP16 frontend and
   overrides only the dtype-bearing methods"。V3 逐符号核实：`RtxFP16` 定义 **16** 个方法，
   Orin **16/16 全覆盖**，且它**不定义 `__init__`** ⇒ 它在运行时**只是个 MRO 途经点**；
   真正被复用的是**上两级**：`GrootN17TorchFrontendRtx.__init__`（Orin 用 `super().__init__` 调它）
   与从 `GrootN17TorchFrontendThor` 继承的 **19** 个方法（校准、kernel-DiT 图、
   `normalize_state`/`denormalize_action`/`predict`、HF processor 访问器）；Orin 另**新增 28 个**。
   这也解释了审阅觉得困惑的那点：基类自称"an A/B precision reference against the bf16 path"，
   而**它的 FP16 行为一个都没被继承**。改基类属于改继承链，**在范围外，没动**。
2. `:1824` 的"tracked under the redundant-weight-memory TODO"指向一个**代码树里不存在**的 TODO；
   改成能解析的引用：`docs/groot_n17_orin_sm87.md` §8 的「清理冗余权重内存」。
3. 🆕 `_groot_n17_fusion.py` 的 Args 自相矛盾：它写"the arithmetic runs in fp32 and the result is
   cast back to `pos_embed.dtype`"，而**同文件下方 20 行的注释和代码说的正好相反**
   （`wgt` 按 `out_dtype` 建、gather/加权/四项求和都在 `out_dtype` 里跑），
   并且那条注释还记了"fp32 算完再 round 会得到 max&#124;d&#124; 0.125"。
   §6.24.5 的实测**站在代码这一边** ⇒ 改 Args，不改代码。
4. `tests/test_orin_groot_n17_continuous.py` 的 C6 里 `_err_deg(broken_dec, ref_far)`
   **算了就丢**，而注释写着"Recorded, not asserted"。按计划里"优先加强界"处理：
   改成**灾难界 10.0°**（实测滞留值 **1.2337°**，约 8× 余量；NaN/inf 两个比较都为 False，
   一条 assert 就覆盖）。负控：把界收紧到 **1.0°** ⇒ 报红并**打印出 1.2337°**，
   与 §6.15.3 记录的数字**精确一致**。

#### 6.24.7 教训（四条，前两条是**探针/夹具自身的 bug**）

1. **`pytest` 的 `caplog.records` 会在同一测试内的多个 `caplog.at_level` 块之间累积**。
   F1 的门先跑"不符 ⇒ 该告警"、再跑"相符 ⇒ 不该告警"，正向对照因此**看见了上一段的告警**而报红。
   修法是对照前 `caplog.clear()`。**这与 §6.23.6 教训 4 同源**：都是"跨段状态没清"。
2. **往 `sys.modules` 里塞一个 `SimpleNamespace` 冒充扩展模块，会让一个已经建好的扩展被报成"没建"**。
   `flash_rt/__init__.py` 用模块级 `__getattr__` → `_extensions.require` → `present`，
   而 `present` 调 `importlib.util.find_spec`；后者遇到 `sys.modules` 里**没有 `__spec__`** 的条目
   抛 `ValueError`，被 `present` 当成"not built"吞掉 ⇒ 父类那句
   `import flash_rt.flash_rt_kernels as fvk` 直接吐构建指南。
   而且它**只在单独跑这些测试时发作**：整文件跑时前面某条测试已经真导入过、包属性已存在，
   `IMPORT_FROM` 的 `getattr` 就成功了（`IMPORT_FROM` 只在 **AttributeError** 上回落到 `sys.modules`，
   **ImportError 不回**）。**规则：不要 stub 本仓的扩展模块；要 stub 就 stub 到包属性上，
   并且新增测试必须"单独跑 + 整文件跑 + 随机序跑"三种都绿**（本次三种都验过）。
3. **多文件负控脚本必须在每轮开始前还原*全部*文件**，只还原"这一轮要改的那个"会让上一轮的破坏
   漏到后面几轮，把红账记到错误的改动头上（本次第 4 个断点的破坏漏进了第 5/6/7 轮）。
4. ⚠️ **又踩了一次"skip 不是绿"**（§6.5 记过、§7 第 15 条也记过）：仿真档忘了
   `FLASHRT_GROOT_N17_FIXTURE_TAG=_sim`，只给了 `FRAMES=100,107` ⇒
   **`213 passed, 1 skipped`**，看着像绿，实际是 **precision 整个模块（31 道门）一道都没跑**。
   识别办法很机械：`102+65+10+16+20 = 213`，与总数相等 ⇒ precision 贡献 0。
   补上 tag 后是 **244 passed, 0 skipped**。**规则：门禁套件的验收标准是
   "N passed, 0 skipped"，任何 skipped 都要先解释再放行。**

**本节产物**：`groot_n17_orin.py`（`_warned_dit_graph_params` + `_note_dit_graph_bypass` +
`_capture_dit_graphs` 签名/键控/`_dit_graph_params` + `infer` 三元组护栏 + 两处注释纠正）、
`_groot_n17_fusion.py`（仅 Args 纠正）、
`tests/test_orin_groot_n17_{precision,dispatch,continuous}.py`、`tests/test_lerobot_video.py`。
测试：dispatch **80 → 102**、precision **29 → 31**、`test_lerobot_video` **5 → 20**、
continuous **9 → 10**、preprocess **65** 不变、N1.6 后端门 **16** 不变；
**真机档 179 → 218 passed（74.44 s）**、**仿真档 204 → 244 passed（86.13 s，0 skipped）**，
两档全程锁频 1300500000，均在干净进程里跑。
未进仓的探针：`/tmp/n17_{horizon_probe,fallthrough_cost,f5_negctl,f6_negctl,obs_split}.py`。

---

## 7. Phase 日志

### 1. Phase 0: 侦察（无性能数字产出）

**目标**: 确定版本、平台能力、真实形状、厂商锚点与杠杆树，避免在错误假设上开工。
**设计**: 以权重文件为 ground truth（非 config.json），锁频后实测硬件锚点，
用官方 demo_data 真机数据而非合成数据。

**新增/修改文件**:
- `flash_rt/datasets/lerobot_video.py`（新增）: video-backed LeRobot v2.x 读取器，
  契约对齐 `libero.py`，torch-free/gr00t-free
- `third_party/cutlass`（vendored v4.4.2，gitignored）
- `build_orin_sm87/` + 三个 `.so`（gitignored）

**关键发现**:
| 发现 | 影响 |
|---|---|
| `/mnt/GR00T` 含 **N1.5 / N1.6 / N1.7 三个版本** + Cosmos-Reason2-2B（N1.7 的 backbone 原模型） | 版本选型；N1.5 在 FlashRT 中零支持 |
| 两个 checkpoint 的张量数（1106 / 1030）与 `docs/groot_transformers5_weight_corruption.md` 记录一致 | **与 Thor 适配同一批权重**，Thor 精度数字可直接对照 |
| bf16 DiT 已达权重流上限 95.6% | bf16 无空间，唯一杠杆是减字节 |
| CUDA graph 对 DiT GEMM 链只有 1.07× | 推翻"launch 开销 15 ms"假设；graph 的价值在全链路 eager→图，不在 GEMM 本身 |
| 官方参考实现算完 lm_head logits 后丢弃 | 免费 ~4–5 ms |
| `deployment_orin.md` 的 5.3 TFLOPS、`configs/groot.yaml` 的 geglu 均为错 | 见 §2.1 |
| 随机数据 state 量级错 300× | 验证了必须用真机数据 |

**验证方法**: 权重完整性 = 抽样 24 个张量 live vs safetensors 逐值比对（rel=0.00e+00，
防的正是 transformers 大版本静默重新初始化视觉塔那一类故障）；
硬件锚点 = 锁频断言 + best-of-5 × 100 iters；
形状 = 官方 `Gr00tPolicy` + forward hook 实测。
**累计 E2E**: 尚无（Phase 0 不产出性能数字）。
**决策记录**: 起点选 `GrootN17TorchFrontendRtxFP16`（无 FP8/无标定、RTX FA2 后端、
SM87 本就走 `flash_rt.hardware.rtx.*`），而非 Thor 系（绑定 Thor 专属 attention 与 side-load）。

---

### 2. Phase 1: 前端 + 正确性（N1.7 已闭环）

**目标**: BF16 通路跑通并**同时**过精度门与图安全门，再谈优化（skill 阶段门）。
**设计**: 三条独立理由选 BF16 而非复用 RTX 的 FP16 pipeline ——
(a) `quantize_int8_rowwise` 是 **bf16-only**，而 INT8 是 DiT 唯一实测有效杠杆（§6.3）；
(b) HF eager 参考与 checkpoint **本身就是 bf16**；
(c) SM87 无 FP8 tensor core，FP8 权重往返**换不到任何吞吐**却损失 ~3 位尾数。
pipeline 刻意与 `pipeline_rtx_fp16.py` **同阶段分解、同算子顺序**，
这样逐级 cos 隔离的是 dtype 而非算法。

**新增/修改文件**: 见 §6.6.7（4 个新文件 + 1 处双注册 + 2 个测试 + 1 个 aux 采集器）。

**关键发现**:
| 发现 | 影响 |
|---|---|
| **state 单位不对称**：N1.7 statistics 是弧度、N1.6 是度，而数据集存度（§5.5） | 归一化 state 偏出 [-1,1] 达 57×；随机数据**永远抓不到**。修在数据侧，模型侧零改动 |
| `Qwen3VLVisionPatchMerger` 硬编码 `nn.GELU()`（exact erf），**不读** `hidden_act` | 用 tanh 近似每个 merger 损 ~1e-3 cos 并复合进 LLM 的 image token；**RTX FP16 通路同病未修**（§6.6.2） |
| RoPE 必须 fp32 数学 | HF 与 FlashRT 的 fp16 kernel 都升 fp32；bf16 shim 若不升，`vit_h` cos 只有 0.9917 |
| `_can_record_outputs` 使 `hidden_states[-1]` 为 **pre-final-norm** | 证实 FlashRT 省略 `_llm_norm_w` 是对的，不是遗漏 |
| `fwd_bf16_causal` 收 `num_heads_q`/`num_heads_kv` 两个参数 | **原生 GQA**，省掉 head 扩展与 `gpu_fill_neginf_fp16` logits slab |
| `position_embedding.weight` 是 `(1024,1536)` = **容量**不是 horizon | 从权重形状推 horizon 会得到 1024，warmup 直接炸；改为从 config 读 `action_horizon=40` 并与权重形状交叉断言 |
| `bf16_nn_bias*` 在 M=41 报 `CUBLAS_STATUS_NOT_SUPPORTED` | M 非 16 对齐；改 `bf16_nn` + 独立 `add_bias_bf16`（launch 被 graph 吸收） |
| 权重字典每次调用重切融合 qkv（~120 次大 copy） | **−11.46 ms**（杠杆 #7） |
| ViT block 18–23 的输出无人消费 | **−7.77 ms**（杠杆 #8） |
| fp32 算 cos 会返回 **1.000167 > 1** | 15296 通道 + 2.9e5 元素的范数累加溢出；所有 cos 一律 float64 |
| §6.5 旧离群表的"离群通道"列在自述定义下**不可能成立** | 整表以单一度量重测；LLM 逐通道比实为 **2147–3547×**（旧记 382–612×），DiT 实为 142×（旧记 52×）⇒ 当时**下调杠杆 #3 精度判定**（后被 §6.8 的实测门**撤回**） |

**验证方法**: G1–G5 全部对 §5.4 的**真机** fixture（frame 0 + 300 两帧，
`_aux.pt` 交叉校验 `paired_fixture` 字段防错配）；cos 一律 float64；
图安全 = graph≡eager **且** replay≡replay **且** 换输入后 replay 必须跟随（stale-value）；
延迟 = 锁频断言 + median of 10 + `torch.cuda.Event`，并与 HF 插桩的**同一边界**比。
**累计 E2E（Phase 1 快照，融合前）**: **116.18 ms @ 边界（3.10× vs HF eager 360.38 ms）**；
完整 `get_action` 等价 ~143 ms；G4 解码 action cos **1.000000**，
max&#124;d&#124; **0.0043 rad（0.25°）**。
⚠️ 这个倍率是**口径 A**（HF 的 58.07 ms 视觉塔未计入 FlashRT 成本），
真实独立部署成本是 174.86 ms（2.06×）—— 见下面的 Phase 1.5。
**决策记录**:
- 不修 `pipeline_rtx_fp16.py` 的 merger GELU（红线 #1 additive only；且 FP16 侧
  **没有** `gelu_erf_fp16` 可用，修它需要先有 kernel ⇒ 红线 #8 只登记缺口）。
- `attn_backend_groot_n17_orin.py` **主动 `del`** 父类的 FP16 MHA 脚手架，
  让误落回变成响亮失败而非把 bf16 当 fp16 读（红线 #4）。
- `lw_run = dict(lw)` 每次浅拷贝：`deepstack_inject` 是 per-prompt 的，
  **缓存里绝不保存 prompt 相关指针**。
- 杠杆 #3（INT8 DiT）**不在 Phase 1 内接入**：先把 §6.5 的真激活剖面复测出来，
  结果推翻了原判定，因此改为"先过 fake-quant 门再谈速度"（原则 #11/#12）。
  ⇒ **该门已在 Phase 2 做完并通过**（§6.8 / 下面第 4 条日志）。

### 3. Phase 1.5: image→embeds 融合（杠杆 #10，N1.7 已闭环）

**目标**: 摘掉 `aux["llm_input_embeds"]` 这个 HF forward 依赖，让本通路能独立部署。
用户指令是"尽量复用或参考其他平台"，因此**不新写算法**，照搬 Thor FP8 通路的
5 步结构，只换 dtype 与 kernel 名（详见 §6.7 的 Thor→Orin 逐行映射表）。

**设计**:
- `_fast_pos_embed_interpolate` 定义在 `GrootN17TorchFrontendThorFP8` 上，
  而 Orin 的 MRO 是 `Orin → RtxFP16 → Rtx → Thor → object` —— **它不是祖先**。
  抽成共享自由函数 `frontends/torch/_groot_n17_fusion.py`，而不是复制第二份
  （40 行索引算术，抄错一次要查一小时）。Thor 一字未改（红线 #1）。
- 融合常量全部在 `set_prompt` 里算好、以指针烘进图；per-frame 只做 copy。
- `fuse_image_embeds` **必须在 `super().__init__()` 之前赋值**：那次调用会跑
  `_load_weights`，而它要读这个标志决定加载多少层 ViT。
- `_vit_layers` 从**已加载的 per-layer 列表长度**推导，不硬编码 24。

**关键发现**:
| 发现 | 影响 |
|---|---|
| HF 为产出 `llm_input_embeds` 必须跑完整视觉塔 **58.07 ms**，而 FlashRT 又重跑了 ViT 前 18 层 | ViT 被算两遍；"同一边界"口径一直在**高估**本通路（真实成本 174.86 ms 而非 116.79 ms） |
| patch embed 必须用 `F.conv3d`，**不能**用展平 GEMM | 数学等价但累加顺序不同，差 ~1 ULP；被 24 层残差塔放大成 `backbone_features` 0.996951→0.995323 |
| HF 用 `dtype=pos_embed.weight.dtype`（**bf16**）构造双线性插值权重 | 我按"更精确"写成 fp32 再降回，得 max&#124;d&#124; = 0.125（1 ULP），`vit_block_17` 0.998156→0.995355。**在 24 层残差塔前面，"更精确"≠"更接近参考"** |
| `_merger_fc1_w` 是方阵 (4096,4096) | 转置 bug **不会报错**；上线前显式断言 `== raw.T` 才敢接 |
| merger 的 LayerNorm 在 2×2 shuffle **之前**（`use_postshuffle_norm=False`） | 顺序错了不会崩，只会静默降精度 |
| `input_ids` 必须从**外层** `Qwen3VLModel` 抓 | 文本模型是被 `language_model(inputs_embeds=...)` 调的，那一层 `input_ids is None`，旧钩子静默采不到（已由 `REQUIRED_N17` 拦住） |
| transformers 4.57 把 `Qwen3VLVisionModel.forward` 首参改名为 `hidden_states` | 钩子两名都收，否则升级即断 |
| **测试自身有 bug**：拿 `cap["vit_h"]` 比 `vit_block_17` | 融合开启后 `vit_h` 是 layer-23 输出，该比较测的是空气（cos 0.271）。改为比逐层快照 |

**验证方法**: 三段式严格门 `test_fusion_reproduces_hf_embeds` ——
(a) 文本 token embed 必须 **bit-identical**（纯 gather，任何非零都是索引 bug 而非舍入）；
(b) kernel merger vs torch **fp64** merger 在**同一输入**上 ≥0.9999
（实测 0.99999455）；(c) 融合结果整体 ≥`THR_FUSED_CONSUMED`。
另有分离实验证明 merger 无罪：`merger(HF 的 vit_block_23)` vs HF image token
cos **0.99999632** ⇒ 残差**全部**来自 ViT 24 层的 bf16 累积。
修好后两个模式的 layer 0/5/11/17 余弦**逐位相同**
（0.999994 / 0.999892 / 0.999618 / 0.998156）。

**累计 E2E（融合后，当前）**: backbone **66.13 ms** + DiT(graph) **58.43 ms** =
**124.56 ms @ 边界**。两个口径重合于 **2.89×**（vs HF 360.38 ms）；
融合自身 **174.86 → 124.56 ms = 1.40×**。
G4 解码 action **cos 1.000000（两帧均是）**；max&#124;d&#124; frame 300 **不变**
（0.003260 rad），frame 0 由 0.004307 → **0.008613 rad（0.25°→0.49°）**。
G5 graph≡eager / replay≡replay 仍 **bit-identical**。**23 passed**
（precision 12 + dispatch 11；复跑已确认）。

**决策记录**:
- **接受 `backbone_features` 0.999729 → 0.996951**，并新增
  `THR_FUSED_CONSUMED = 0.995` 而非沿用 `THR_CONSUMED = 0.999`。
  这是**被测量的取舍**，不是放宽门：四条理由写在 §6.7.2
  （误差来源已定位为 ViT bf16 地板 / 参考侧本身更准故不可对等 /
  经 cross-attn 强烈衰减 —— **不是零影响**，f0 的 max&#124;d&#124; 确实翻倍，
  但两帧 cos 均仍为 1.000000 且绝对误差 <0.5° / 融合自身另有严格门）。
- **不把杠杆 #8 的 −7.77 ms 与杠杆 #10 的收益相加**：融合需要 block 23，
  ViT 必须跑满 24 层（+7.81 ms）。净账仍赚，但两者互斥。
- **保留 `fuse_image_embeds=False` 分支**：它让"融合对不对"可被 A/B 隔离
  （本次两个数值陷阱都是靠它定位的），不是向后兼容 shim。
- 仍未摘掉的 HF 依赖（`rope_cos/sin`、`visual_pos_masks`、`pixel_values`）
  **不在本次范围内**，登记进 §8。

### 4. Phase 2（部分）: INT8 DiT 真激活 fake-quant 门（杠杆 #3 精度侧解锁）

**目标**: 执行 §6.5 判定 1 要求的实验 —— 在接任何 INT8 kernel **之前**，
先回答"per-row INT8 W8A8 这个数值方案在**真实**激活上保不保得住输出"。
**纯精度实验，不产生速度主张**（原则 #11/#12）。

**设计**: **没有改 `pipeline_orin.py`**。问题问的是数值方案而不是我们的 kernel，
所以直接给**厂商模型自己**的 DiT 打补丁：`action_head.model`（`AlternateVLDiT`）下
全部 **228 个 `nn.Linear`**（224 block + 2 timestep_encoder + 2 proj_out）的 forward
换成 quantize→dequantize 后照常算，粒度与 `cutlass_int8_rowwise_bf16out` 对齐
（权重 per-output-channel、激活 per-token，对称 int8 ±127），
然后跑厂商自己的 `policy.get_action(obs)`，`obs` 取自 §5.4 的真机 fixture。
四个变体（`w_only` / `a_only` / `w8a8` / `w8a8_pt`）以便**归因到轴**而不是只观察结果。

**关键发现**:
| 发现 | 影响 |
|---|---|
| 未打补丁的基线对 fixture `actions` 是 **max&#124;d&#124; = 0.000000** | 先自证实验有效（噪声种子 / state 单位 / preprocessing 全部复现），脚本里 `assert` 卡住；否则后面全部无意义 |
| **per-row W8A8 两帧 cos ≥0.999999、最坏 0.255°** | 比本通路 bf16 融合档**已经在跑的** 0.49°（§6.7.2）还小 ⇒ §6.5 的下调**撤回**，杠杆 #3 精度侧解锁 |
| **per-tensor 差 2.5×**（frame 300：0.658° / cos 0.999996） | 坐实这是 **scale 问题不是 noise 问题**（原则 #12）⇒ per-row 是**硬要求**；且 AWQ/GPTQ/百分位标定那些治噪声的手段在此**无用** |
| `w8a8` ≈ `a_only`（frame 300 均 0.219°），而 `w_only` 只 0.139° | 误差**主要来自激活**；将来若要退一档，**退权重（W8A16）没有意义** |
| DiT 的 142× 逐通道离群**没有**击穿 per-row 缩放 | **撤回** §6.5 末尾"旋转对 DiT 也可能有效"的猜测：DiT **不需要** QuaRot，杠杆 #6 范围收窄到 LLM/ViT |
| 228 个 Linear 共 1091.17 M 权重 ⇒ bf16 8.729 GB / INT8 4.365 GB（×4 步） | **重算并与 §6.1 的 8.73 GB、§6.2 的 "8.73→4.37 GB" 逐项吻合**（那是 4 步流量，不是权重大小） |

**验证方法**: 两帧真机 SO101（frame 0 / 300），厂商 `get_action` 全流程，
G4 = 解码后 action 对 fixture 的 HF 参考，cos 在 float64 下算。
**结论**: 过门。**没有产生任何延迟数字** —— 1.56×（58.4→37.5 ms）仍只来自
§6.3 的独立微基准，接入 kernel 后必须按原则 #16 重做 in-pipeline A/B。

**决策记录**:
- **在厂商模型上做，不在 FlashRT pipeline 上做**：不为一个精度实验去动已跑通的
  bf16 通路（红线 #1 的精神）；且厂商模型自带同权重、同形状、同真实激活。
- **fake-quant 的边界写清**：fp32 算 quant/dequant + bf16 `F.linear`。
  真 kernel 是 int32 累加（**更准**）⇒ 本门在累加轴上偏保守；
  但**未**建模 epilogue 的 scale 应用顺序与 M=41 的 tile 边界效应。
- **范围只到 DiT**：`state_encoder`/`action_encoder`/`action_decoder`/`vlln`/
  `vl_self_attention` 未量化，与杠杆 #3 的定义一致。
- 实验脚本 `/tmp/dit_int8_fakequant.py` 是一次性件，**未进仓**（复现方法已写进 §6.8）。

### 5. Phase 2: INT8 DiT kernel 接入（杠杆 #3 已交付）

**目标**: 把 §6.8 解锁的 INT8 档真正接进 DiT，并按原则 #16 做 in-pipeline A/B
（fake-quant 门不产生速度主张）。additive：bf16 档语义一字未改，两档共存可 A/B。

**设计**: `dit_forward` 用 `"q_w8" in weights` 选档；分发在**循环外**解析成三个闭包
（`qz` 量化并返回该读的指针 / `mm` GEMM / `W` 解析权重族），所以 32 层 × 6 站点
没有一个 `if`。前端加 `use_int8_dit=False` 构造开关（在 `super().__init__()`
**之后**量化，因为要先有 bf16 权重）。

**关键发现**:
| 发现 | 影响 |
|---|---|
| **K/V 权重形状随层奇偶变**：cross 层 (2048,1536)、self 层 (1536,1536) | 第一版写死 `nk = {"k": (D,D)}` 立刻炸在构造期。**改为逐层从权重形状推导**（权重是 ground truth，不是一张表）。另加断言：奇数层 k/v 必须是 (D,D)，因为 `dit_forward` 对它硬编码 `(Sa,D,D)` |
| **spec 的 `T()` 让布局与 INT8 kernel 相反**：`_dit_<F>_w` 是 (K,N)，kernel 要 nn.Linear 原生 (N,K) | 且 q/k/v/o 在实际用到的层上是**方阵** ⇒ 转置错误**维度断言抓不到**。加了 `_smoke_int8_dit_layout()`：每族一次真 INT8 launch，与 `x @ w_kn`（bf16 档算的东西）比 cos，<0.99 拒绝构造。**与融合期 `_merger_fc1_w` (4096,4096) 是同一个陷阱** |
| adaLN 后的 `xn` 同时喂 Q/K/V | 量化一次复用 ⇒ **每层 4 次 quantize 而非 6 次**（32 层 128 quantize + 160 GEMM）。**被无 GPU 契约测试钉住**，退化立刻抓到 |
| `gate_residual_ada_norm_int8` **不适用** | 读源码：它算 `rsqrt(mean(r²)+eps)` 是 **RMS**、残差是带门的 `residual+x*gate`；DiT 用**减均值** AdaLayerNorm + 无门残差。§8 原待办**撤回**（红线 #7：每个引用的 API 都要对源码核过） |
| SM87 **没有** int8 输出的 `ada_layer_norm` / `layer_norm_no_affine` / `gelu_tanh` | ⇒ 4 个量化 pass 必须独立存在。已计入 41.50 ms；额外流量 ~42 MB/步 对 2182 MB/步 是 ~2%。三个缺口报给 kernel owner |
| **INT8 静默回落 bf16 在数值上更准** | 任何"比输出"的验证都抓不到它，只会让 INT8 的精度/延迟数字悄悄变成 bf16 的。所以加了无 GPU 契约测试：缺 scratch 必须 `KeyError`，**且报错前一个 kernel 都没发**（AGENTS.md §3.6） |

**验证方法**: 两帧真机 SO101，与 bf16 档**共用同一套门与同一个 `THR_CONSUMED=0.999`**
（**未为 INT8 放宽任何门**）；延迟用**两个独立口径互证** ——
隔离量测（进程里只驻留一个 frontend，median of 10）与配对交替 A/B（median of 11）。

**累计 E2E（INT8 档，opt-in）**: backbone ~66.3 ms + DiT(graph) **41.5 ms** =
**107.8–108.8 ms @ 边界 → 3.31–3.34×**（bf16 档 124.6–125.3 ms → 2.88–2.89×）。
DiT **1.411×**（隔离，3 次）与 **1.391–1.422×**（配对交替，3 次独立跑）一致，差 ≤2.2%。
G4 cos **0.999999**（两帧）；max&#124;d&#124; frame 0 **与 bf16 完全相同**（0.008613 rad，
说明该帧最坏元素由 backbone 决定）、frame 300 由 0.003260 → 0.006345 rad（0.19°→0.36°）。
G5 graph≡eager / replay≡replay **bit-identical**。**30 passed**（precision 17 + dispatch 13）。

**决策记录**:
- **measured vs predicted 的 −9.6% 偏差照实记**：§6.3 µbench 预测 1.56×，实测 1.411×。
  机理是 µbench 为合成（随机权重，且没跑真实循环里的 attention / adaLN /
  no-affine LN / GELU / bias / residual，**这些不随 INT8 变快**）。
  ⇒ 记一条通用教训：**合成 µbench 能定"值不值得接"，不能当交付数字。**
- **默认档保持 bf16**：INT8 全门通过，但改默认等于改已验证通路的出厂数值契约，
  属部署决策；登记为 §8 第一条待办，交使用方拍板。
- **bf16 DiT 权重故意不释放**（+2.18 GB）：配对交替 A/B 要求两档同进程可比
  （AGENTS.md §3.8）。定档后再与 §6.6.6 第 3 条的冗余权重一起清理。
- **cross 层 k/v 也量化了**（int8 副本 1.19 GB 而非 §6.8 估的 1.09 GB）：
  它们在 kernel 循环里用不到（K/V 由 `_precompute_dit_cross_kv` 在 torch fp32 里预算），
  但逐层统一推导形状比"按奇偶跳过"更少特例、更不易错。

### 6. Phase 2 收尾: INT8 定为出厂默认档（使用方拍板）

**目标**: 把 INT8 从 opt-in 翻成默认，并保证测试与文档不会因为这次翻转而失同步。

**改动**: `use_int8_dit: bool = True`；`_DIT_QUANT` **两个方向都在实例上设**
（类属性现在广告 int8，若只在 INT8 分支里赋值，bf16 实例会**继承到错的**
`_DIT_QUANT` —— 这是翻默认时最容易漏的一处）。

**关键设计**: 测试**不硬编码**默认档。`DEFAULT_INT8` 由
`inspect.signature(...).parameters["use_int8_dit"].default` 读出，
`prompted` fixture 与 `test_latency_at_the_hf_boundary` 都跟随它。
⇒ 头条延迟数字永远描述**真正出厂的那一档**；一个硬编码默认值的测试会
安静地一直验证没人跑的档。另加无 GPU 的
`test_int8_dit_is_the_shipped_default` 钉住签名与 `_DIT_QUANT`，
误翻回去是**测试失败**而非静默降速。

**验证**: **31 passed**（precision 17 + dispatch 14）。默认档下的头条量测：
`backbone 66.26 + DiT(graph) 41.72 = 107.99 ms → 3.34×`；
同次配对交替 A/B 复证 **1.398×**（bf16 58.33 / INT8 41.73 ms）。
至此 DiT 比值共 **4 次独立量测：1.398 / 1.404 / 1.411 / 1.422×**，极差 1.7%；
§6.10 文档收尾后的纯复跑（代码未变）再给一次 **1.391×**（bf16 58.77 / INT8 42.23 ms，
边界 108.76 ms → 3.31×）⇒ **5 次：1.391 / 1.398 / 1.404 / 1.411 / 1.422×，极差 2.2%**。
这是 session 噪声带，不是趋势（§6.9.3）。

**决策记录**:
- 使用方指示"**若 outlier 太大则考虑动态量化，尽量不用 QuaRot**"。
  ⇒ **DiT 本来就是动态量化**：`quantize_int8_rowwise` 每次 forward 在设备上
  重算 per-token amax，无标定、无静态 scale，正是原则 #12 说的那一类。
  这条指示对 DiT 无需额外动作，对 **LLM/ViT 成为下一步的首选路径**（§8）。
- 杠杆 #6（QuaRot）**降为备选**：只在 LLM 的动态量化门实测不过时才启用。
  §6.5 那句"per-row 也救不了 LLM"**从未做过 DiT 那样的真激活门** ——
  而 DiT 当年同样被判"142× 危险"，实测却过门且余量很大 ⇒ 该判定必须先验再用。

### 7. Phase 2（续）: LLM / ViT 的动态量化门（§6.10，纯精度侧，无 kernel 改动）

**目标**: 兑现上一条的使用方指示（"outlier 太大就用动态量化，尽量不用 QuaRot"）。
DiT 本来就是动态量化 ⇒ 无需动作；要重测的是 §6.5 里**从未过真激活门**的
LLM（2147–3547×）与 ViT（120–205×）两条"per-row 也救不了"判定。

**方法**: 与 §6.8 同一套（原则 #11 的可重复回路）：给厂商模型打
quant→dequant 补丁 → 跑厂商自己的 `get_action` → 真机 fixture 两帧。
**两个脚本都先断言未量化基线复现 fixture 到 max&#124;d&#124;=0，否则实验作废**
（这一步是 fixture 有效性的唯一证据，跳过它后面所有数字都不可信）。
全部动态：scale 每次从当次 forward 自己的 amax 算，无标定集、无静态 scale、无旋转。

**结论（两条相反，都写进 §6.10）**:
- **LLM 过门 ⇒ §6.5 判定 2 推翻。** `all_row`（全 112 Lin 含 QKV/O）G4
  cos **0.999999/0.999997**、最坏 **0.573°**；`ffn_row`（48 Lin，杠杆 #4 本体）
  更好：**0.247°/0.242°** —— 已优于上线的 DiT INT8 档（0.493°/0.364°）。
  per-tensor 差 **2.9–9.0×** ⇒ 原则 #12 的"scale 问题"分类**第二次坐实**
  ⇒ **LLM 也不需要 QuaRot**。
- **ViT 不过门 ⇒ §6.5 判定 3 确认，杠杆 #5 关闭。** 被消费的 tap
  `vit_block_17` per-row 只有 **0.971083**（f0）/ 0.990567（f300），
  门是 0.998，连宽松的 0.995 都不到；per-tensor 崩到 **0.512692**。
  按"尽量不用 QuaRot"的指示 ⇒ **ViT 保持 bf16**。

**这一 Phase 最该记的两件事**:

1. **只看 G4 会误判（原则 #2 的第三次现形）。** ViT `blk_row` 的 G4 是
   **cos 0.999999 / 0.740°** —— 单看它会把一个 tap cos 0.971 的档放上线。
   抓住它的是**逐级门**，不是 E2E 指标。原因与 §6.7.2 同一条：
   DiT 只经 cross-attention 读 backbone 特征，逐 token 误差被平均掉。
2. **§6.5 那条推理错在哪**（值得单独记，因为它"看起来对"）：
   "离群是逐通道的 ⇒ 每行都被同一通道主导 ⇒ per-row 也救不了"。
   前半句对，**结论不跟着成立** —— 决定成败的是量化后**非离群通道的相对误差**，
   不是离群比本身。**2147–3547× 描述的是分布，不是误差。**
   §6.5 自己记过反证：LLM 通道 1793（幅度 15296.0）在 bf16 下 ULP=64、
   `llm_layer_2..15` 的 max&#124;a&#124; 完全相同 ⇒ 它在后 14 层**已被 bf16 冻结**，
   粗量化它损失不了什么。ViT 反过来（§6.10.3 的两条假说）。

**产出**: 无代码改动、无 kernel 改动 ⇒ **仍是 31 passed**。
杠杆 #4（LLM INT8）**精度侧解锁**，接入留作 §8 的下一步（ROI 最高项）。
⚠️ 接入时**不得照搬 DiT 的 1.42×**：LLM 是 prefill（M=Se=141），
不是 DiT 那种 M=41 的权重带宽极端场景（原则 #13/#16，必须自己量）。

**实验件**: `/tmp/llm_int8_fakequant.py`、`/tmp/vit_int8_fakequant.py`，未进仓。
LLM 脚本自己踩过一个会让关键对比失效的坑：`fams` 算了却没用，
导致 ffn-only 与 all-tower 两条补丁**实际打的是同一批层** —— 若没打印真实
补丁数（48 vs 112）就发现不了，"QKV/O 也能救"的结论会变成无意义的自比。
ViT 脚本踩过 `o[0]` 坑（vision block 返回裸张量、decoder layer 返回 tuple，
一律 `o[0]` 会静默取到**第一行**，cos 照算不误）⇒ 已改成
`isinstance(o, tuple)` **并在比较处断言元素数**：这类错误不抛异常，
只给出一个看着合理的假 cos。

### 8. Phase 2（续）: 接 LLM INT8 FFN → 实测否决 → 回退，并量到 backbone 的真实瓶颈

**目标**: 兑现 §6.10.1 解锁的杠杆 #4（LLM FFN-only INT8，动态 per-row）。
按原则 #13 **先微基准再写代码**。

**过程与结果（详见 §6.11）**:

1. **微基准**（真实形状 M=141/D=2048/FF=6144，配对交替，no-op 臂量出
   **事件地板 18.4 µs** 并从单算子行扣除）：合成权重预测省 3.56 ms；
   换**真实权重 + 真实 xn** 后是 **704.4 → 467.2 µs/层，省 3.80 ms（1.508×）**，
   与权重流量 roofline 的 **3.82 ms** 吻合。
   ⇒ **修正了 §6.9.4 那条教训的表述**：合成 µbench 不能当**交付数字**，
   但确实能定**值不值得做**（这次两边都对）。
   另测出两条意外：**k/v 在 INT8 下更慢（0.851×）**；
   **`quantize_int8_rowwise` 是 ~37 µs 固定开销（cols 2048 与 6144 一样）⇒ 延迟受限**，
   这给"量化融进 norm epilogue"的 kernel 缺口定了价。
2. **接入**（additive，与 DiT 同一套模式）：`use_int8_llm_ffn`、
   `_quantize_llm_ffn_weights`（(K,N)→(N,K) 逐层从张量推 + 形状互校验）、
   `_smoke_int8_llm_layout`（bind-time 真发射）、4 个 device-side scratch、
   档位分发（缺 scratch 必须 `KeyError` 且**一个 kernel 都不发**）。
   **红线 #5 的证据**：kernel 计数 **32 quantize + 48 INT8 GEMM / backbone**，bf16 档均为 0。
3. **精度：不过门。** `backbone_features`（= `vlsa_block_3`，DiT 真正消费的张量）
   cos **0.992077 < THR_FUSED_CONSUMED 0.995**。而 **G4 是 cos 0.999999 /
   max&#124;d&#124; 0.329°，比 bf16 档自己的 0.493° 还小** ⇒
   **只看 G4 会放过它（同一教训第三次现形，前两次 §6.7.2 / §6.10.2）**。
   机理量到了：vlln + 4 层 VLSA 把 LLM 偏差按 1−cos 放大 **40–120×**。
4. **延迟：GPU 省了，墙钟没省。** 逐 kernel GPU 中位数说 FFN 省 **3.93 ms**，
   LLM stage **捕获后**省 **4.77 ms（20.31→15.54）**，
   但 **eager 墙钟只省 0.44 / 0.85 / 0.97 ms**（3 个进程的配对交替）。
5. **诊断出真正的原因（本 Phase 最大的产出）**：
   一次 backbone 的**纯 CPU 提交时间 60.00 ms，墙钟 65.77 ms（91.2%）**
   ⇒ **backbone 是 launch 受限的**。单算子 CPU 成本
   `bf16_nn` **21.14 µs** / `cutlass_int8_rowwise_bf16out` 10.88 /
   `quantize_int8_rowwise` 9.76 / `rms_norm` 9.41 ⇒
   **"INT8 发射更多所以更慢"被排除**（INT8 臂 CPU 反而便宜 0.13 ms）。
   旁证一直在文档里：杠杆 #1 的 **DiT eager 115.0 → 捕获 58.4 ms（1.97×）**；
   **backbone 从来没被捕获过**。
   ⇒ 同时修正 §6.6.3：那 30 ms 的 roofline 差**量级来自 CPU dispatch，不是 GPU 侧 torch shim**。
6. **回退**：pipeline / frontend / 两个测试文件全部还原，**31 passed** 复证。
   §6.11 保留全部量测、机理与否决理由（原则 #11(b) 的记录义务）。

**决策记录**:
- **不放宽门。** 放宽 `THR_FUSED_CONSUMED` 去换 0.6% 的 eager 收益，
  正是 AGENTS.md §6"smoke floors are load-bearing / no receipt is ever written
  from a floor-relaxation experiment"要拦的事，而 §6.10.2 刚写下"只看 G4 会误判"。
- **比原则 #11(b) 更严一档**：不是"修复成本 > 收益 ⇒ 默认关"，
  而是**"连门都没过 ⇒ 不落地"**。两条理由都记，精度门是主因。
- **捕获也救不回杠杆 #4**：4.77 ms 的 GPU 收益会显现，但
  `backbone_features` 0.992077 与是否捕获**无关** ⇒ 两件事独立。
- **新杠杆 #11（backbone CUDA-graph 捕获）升为 ROI 最高项**：
  **零精度让步**，且是杠杆 #4 那 4.77 ms 的唯一兑现途径。
  下界已量到（LLM stage 省 2.06 ms）；⚠️ **不得照搬 DiT 的 1.97×**。
  → **已在下一条（#9）实现并实测：只值 4.54 ms（1.078×），不落地。**
- **杠杆 #9（RoPE / gate·up kernel 化）因 §6.11.5 升值**：在 CPU 提交受限的
  backbone 里，省下的**主要是 CPU dispatch**（`_rope_rotate_half` 每次 ~7 个 torch 算子，
  LLM+ViT 共 80 次调用），而不只是 GPU 时间。
  → **该升值已在 #9 撤回**：backbone 的 launch 受限份额只有 7.3%，省的仍是 GPU 时间。

**三个量测陷阱（都踩过，写进 §6.11.5）**:
`_run_kernel_backbone` 末尾自带 `torch.cuda.synchronize()`（不屏蔽就量到
CPU/wall=100% 的同义反复）；n>1 连发会打满 launch 队列（量到 `max(CPU,GPU)`，
**必须 n=1**）；捕获必须显式传 `stream=s.cuda_stream`（用 `stream=0` 得到
**空图且不报错**，第一次量出"LLM stage 捕获后 2.72 ms"，差点当成 8× 收益写进文档）。

### 9. Phase 3/4: 做杠杆 #11 → 前置重构落地（#12），捕获本身实测后不落地

**目标**: 按 §6.12.3 的清单把 Thor 早就有、Orin 因继承链漏掉的 backbone 全图捕获补上。

**做了什么（两步，第一步留下、第二步不留）**:

- **Step 1（已交付，杠杆 #12）**：`_run_kernel_backbone` 拆成
  `_build_backbone_runtime()` / `_kbb_load_inputs(aux)` / `_kbb_forward(stream, ...)`；
  ~30 个 buffer、`OrinGrootN17BackboneAttn`、5 个 kernel 参数字典全部只建一次；
  DeepStack inject 的布尔掩码赋值换成 `zero_()` + `index_copy_`
  （**逐位相同**，且去掉 3 次内含 `nonzero()` 的 host 同步）；
  `set_prompt` 的 `.to(_BF16)`（bf16 上是 no-op，会让 `_backbone_features`
  alias 常驻 buffer）换成 `.clone()`。
  **实测**：backbone 65.71 → **63.03–63.36 ms**，边界 107.40 → **104.79–105.26 ms**
  （3.31–3.34× → **3.42–3.44×**），CPU 提交 −7.2 ms，融合余量 4.21 → 2.16 ms。
  **精度逐位不变**：四级 cos 与 §6.6.1/§6.7.2 早先记下的数字逐位对上
  （`vit_block_17` 0.998156 / `deepstack_out_*` 0.999960·0.999687·0.999224 /
  `llm_h` 0.999975 / `backbone_features` 0.996951）。测试 **31 → 36 passed**。
- **Step 2（探针完成，判定不落地）**：整条 backbone 捕获**成功且
  replay≡eager 逐位相同（max&#124;d&#124; = 0）**，`F.conv3d` patch embed 也确认可捕获。
  但配对交替 A/B（中位数 of 5×7，噪声底 ±0.06 ms）只给出
  **62.47 → 57.93 ms = 4.54 ms（1.078×）**，一次性成本 **262.1 ms**
  ⇒ **回本需 57.7 帧**，而 `set_prompt` 一实例只跑一帧 ⇒ 净亏 ⇒ **不留代码**
  （不留死方法、不留开关）。复活条件写进 §6.13.2。

**过程中真踩到的 bug（31 个 GPU 测试全绿也没拦住）**:
unfused 模式把观测直接载进常驻 `llm_h`，而它是 LLM 的残差流、被 16 层原地覆写
⇒ 第二次观测从第一次的 layer-15 输出起跑。不报错、cos 也"看着合理"。
根因：**GPU 侧没有任何测试构造 `fuse_image_embeds=False` 的前端**。
已修（独立 `llm_in` + 每轮重新播种），并用 **5 个 CPU 契约测试**钉住
（`object.__new__` + 假 runtime，无需 GPU/checkpoint）。

**最重要的产出（方法论）**: §6.11.5 从"CPU 提交 60.00 ms / 墙钟 65.77 ms（91.2%）"
推出"backbone 是 CPU 提交受限"，**这个推论是错的**。CPU 提交与 GPU 执行重叠，
该比值量的是"CPU 有多忙"。捕获后 CPU 提交 51.97 → **0.46 ms**（−99%）而墙钟只降 7.3%
⇒ **launch 受限份额只能用 `(eager − replay)/eager` 量**。
DiT 的 1.97× 是因为它 ~1900 个 kernel 每个只有 ~9.4 µs（**小于** 21 µs 的发射成本）；
backbone 的 GEMM 是 27–400 µs，CPU 完全藏在 GPU 后面。
连带撤回杠杆 #9 的升值，并重算了杠杆 #4 的旧账（§6.13.3）。

**原则对号**: #13（先探针再决定写不写）、#15（报 measured vs predicted：
原判"数毫秒到十几毫秒"，实测 4.54 ms，偏大 ~3×）、
#16（瓶颈要按 kernel 粒度分类，不能跨阶段外推 DiT 的比值）、
#11(b)（量出 ROI ⇒ 记档 ⇒ 默认关 ⇒ 往下走）。

**实验件**：`/tmp/bb_graph_probe.py`、`/tmp/conv3d_capturable.py`、
`/tmp/verify_refactor_identical.py`，均未进仓。

---

### 10. Phase 2（续）: DiT 的 K/V 到底有没有被量化 → k/v 豁免成为出厂默认档（§6.14）

**目标**: 使用方给出一条经验教训——"DiT 的 KV Cache 尽量别量化"。先**审计**这条
在本通路上是否已被违反，再决定要不要动档位；不凭印象回答。

**设计**: 把"KV cache"拆成两个不同的东西分别审（这是本节最容易混的地方）：
(a) cross-attention 的 **K/V 缓存**（`_dit_cross_K/_V`，喂给 16 个 cross block）；
(b) self-attention 每步现算的 **K/V 投影**（`_dit_k_w/_dit_v_w` 那两族 GEMM）。
审计 (a) 走代码路径，(b) 走三臂 A/B/C 实测。

**关键发现**:

| 发现 | 影响 |
|---|---|
| **(a) 从未被量化**：`_precompute_dit_cross_kv` 是 **fp32 数学 + bf16 存储**，且 cross block 里**没有 K/V GEMM**（K/V 是缓存，不是每步算的） | 使用方的经验教训在本通路上**本来就没被违反**；不需要为此改任何东西 |
| **(b) 曾被量化**：32 个每步 self-attn K/V 投影跑在 INT8 上 | 这才是可以动的地方 ⇒ 新增 `dit_bf16_families`，默认 `("k","v")` |
| **frame 0 的 0.4935° 与 DiT 档位无关**：三臂（六族全量化 / k/v 豁免 / 全 bf16）**逐位同值** 0.008613 rad | 🔴 **此前 §0.1/§6.9 把 0.493° 记成"INT8 的代价"，归因错了**。它来自 bf16 backbone（`backbone_features` 0.996951 那一级）。INT8 的真实代价只在 f300 与 velocity cos |
| 豁免 k/v 把 f300 从 **0.3636° → 0.1940°**，贴上 bf16 自己的 0.1868°（差距 1.95× → **1.04×**），decoded cos 还略好 | 精度**等或更好** ⇒ 可以按使用方的经验教训定档，不必在精度与偏好之间取舍 |
| 成本很小：只把 **160 个 GEMM 里的 32 个**挪回 bf16，且 **quantize 一个都不省**（adaLN 之后的激活是 q/k/v **共用**的，q 仍在 INT8） | DiT ×4 **+1.00 ms**（配对交替 33.23 → 34.23），边界 **104.79 → 106.91 ms（3.44× → 3.37×）**，DiT 档比值 **1.398× → 1.329×**（门是 1.15×） |

**验证方法**: 三臂**配对交替**、各 5×11 取中位数、**最多两个前端常驻**；
精度用同一套 17 个门 + decoded/velocity 双口径。
计数由 CPU 契约测试钉住：`test_int8_dit_exempt_families_really_run_bf16`
断言 **128 个 int8 GEMM / 32 个 `bf16_nn` / 128 个 quantize**（对照六族全量化的 160/0/128），
`test_int8_dit_exemptions_validate_in_both_directions` 断言**双向**校验
（被豁免的族不得带 `_w8`，未豁免的族必须带）。

**累计 E2E**: 边界 **106.91 ms → 3.37×**（口径 A/B 重合）；每观测口径见 §11。

**决策记录**:
- **k/v 豁免 = 出厂默认**。类常量 `_DIT_BF16_FAMILIES = ("k","v")` 是唯一事实来源，
  由 `test_int8_dit_is_the_shipped_default` 从 `inspect.signature` 钉住。
- 构造参数默认写 **`None` 哨兵**而非字面量 `("k","v")`：否则 `use_int8_dit=False`
  会撞上"bf16 档还传豁免表"的校验而**构造失败**（第一次就这么写坏了 6 个测试）。
- `dit_forward` 的按族分派 `mmq()` **只定义一次**、按 `i8[fam]` 查表。
  ⚠️ 最初按整档分支定义（`if use_int8:` 里一份、`else:` 里一份），
  于是 INT8 档下**所有**站点都走 int8、`bf16_families` 被静默忽略，
  到第一个 self 层才因 `k_w8` 不存在而 KeyError —— **在 A/B 脚本里才暴露**，
  这就是那个计数测试不是装饰的原因。
- 三档可同进程共存（`dit_bf16_families=()` / `=("k","v")` / `use_int8_dit=False`），
  便于配对 A/B。

**原则对号**: #2（decoded 误差与 velocity cos 两把尺都报，不只看外层）、
#11（精度驱动的迭代闭环：审计 → 实测 → 定档 → 记档）、
#13（先微基准/先计数再改）、#16（配对交替 + 锁频 + 中位数，否则 1.00 ms 的差量不出来）、
红线 #4（双向校验，绝不猜某个 GEMM 跑在哪档）。

**实验件**：`/tmp/dit_kv_bf16_ab.py`（未进仓）。
⚠️ 第一版一次性建 6 个前端（3 臂 × 2 帧）把 CUDA caching allocator 打到
`NVML_SUCCESS == r INTERNAL ASSERT FAILED`；改成最多两个常驻才跑通。

---

### 11. Phase 5/6: 第二个数据集复验 + 全链路 E2E + 连续推理（杠杆 #13 已交付）

**目标**: 使用方指定的三项验收——(1) 完整链路、(2) 端到端精度对 Eager、
(3) 连续推理正确性；外加把提供的仿真数据集接成独立验证源。

**设计**:
- **数据集**：`green_to_blue_block_sim`（仿真 SO101，50 ep / 31166 帧 / h264）。
  元数据适配沿用 §5.2 先例（只改 `meta/`，`data`+`videos` 符号链接并逐字节校验）。
  加载器加**一个**通用的 `_resolve_column`（`.joint` 后缀列名），不做模糊匹配。
- **门禁复用而非复制**：`test_orin_groot_n17_precision.py` 的 fixture tag 与帧列表
  改成环境变量驱动，**同一套 17 个门**跑第二个数据集，一个新测试都不加。
- **连续推理**：沿用 RTX FP8 mixin 的 `infer(aux=...)` +
  `_snapshot/_validate_*_contract` 先例，但 Orin 有三处不同（§6.15.1 表），
  其中 DiT cross-KV **必须原地写**——四张图捕获的是槽的 `data_ptr()`。

**关键发现**:

| 发现 | 影响 |
|---|---|
| **通路对 Se 与数据集都不敏感**：Se=148 上同一边界 **106.89 ms → 3.36×**、配对 DiT **1.336×**，与 Se=141 的 106.91 / 1.329× 差 **<0.5%** | 常驻 runtime 的 shape keying 正确重建；出厂门的两帧结论可以外推 |
| 🔴 **decoded action 的 cos 门不住 stranded KV**（§6.15.3）：打断 cross-KV 刷新后，连续帧只退化 **1.45×**（cos 0.9999988）、远端帧 **5.8×**（cos **0.9999917**）——**两种都仍过 0.999 的门** | 第一版负控制**无效**。分三层重测才定位：问题不在 bug，在**量错了张量** |
| **层 1（槽内容 vs 该帧应有的 K）完全判别**：好 **1.000000000**，打断 **0.809373** 且**精确等于**被滞留那帧的 K；层 2（velocity）0.999591 → 0.994225 | ⇒ **stale-value 门必须打在内部张量或 velocity**。层 0 还量到两帧**自己的** HF 参考动作只差 cos 0.9001 ⇒ 外层本可以动，是本 checkpoint 的 decoded action **被 state 主导**、视觉 context 只值 ~1° |
| cross-KV 刷新每帧重做 **403 MB 的 `.float()` 权重提升**，占该步 **5.10/12.68 = 40%** | 缓存后 **13.10 → 8.52 ms**，且 `torch.equal` 全 32 项 **True**（bf16→fp32 精确）；每观测 **122.29 → 117.56 ms（3.16× → 3.29×）** |
| `gen_reference.py` 的 `meta.provenance` 是**硬编码**的"real robot frames … AV1" | 🔴 加 `--tag _sim` 复用后，10 个**仿真** fixture 全都自称真机 AV1 —— provenance 说谎（AGENTS.md §3.7 是验收项）。改成 `--provenance` 参数 + `DEFAULT_PROVENANCE`（既有捕获不受影响），已落盘的 10 个就地改写并**逐张量 sha256 校验 payload 不变** |

**验证方法**: 七道门 C1–C7（§6.15.2），全部在一个前端上跑完 8 个连续帧 + 1 个远端帧。
其中三条是"证据链"而非"看着对"：
**C4** 连续 == 一次性**逐位相同**（max&#124;d&#124; = 0）；
**C5** 捕获计数 **1 次 / 复用 8 次**，且 `_dit_attn`、`_dit_graphs`、槽 `data_ptr`
集合各**只有 1 个**（红线 #5：identical output 不是证据，必须配调用计数）；
**C6** 负控制**分层**，层 1 判别、层 3 明确标注"不可用"。
契约部分**已进仓**为 CPU 测试（dispatch **21 → 51**），无需 GPU/checkpoint。
解码后端另做 PyAV vs ffmpeg **10 次逐位相同**的复验，作为 provenance 的依据。

**累计 E2E**: 每观测 **117.56 ms**（min 117.35 / max 117.84，帧 101–107 中位数）
⇒ **3.29×** vs HF 完整 `get_action` 386.27 ms / **3.05×** vs 同一边界 358.84 ms。
分段：契约校验 **0.031** / backbone **63.51** / cross-KV 刷新 **8.52** /
其余（state+action encode、4 次 DiT replay、decode、D2H）**43.65** / 合计 **115.70**。
逐帧 decoded cos **0.999999232–0.999999628**，max&#124;d&#124; **0.194–0.387°**。

**决策记录**:
- cross-KV **原地刷新**而非重建：`_refresh_dit_cross_kv`（保图）与
  `_precompute_dit_cross_kv`（**必须**弃图）**不能互换**，两者共用
  `_project_dit_cross_kv` 所以投影数学只有一份。
- 形状变化（`Skv_text/Skv_image` 经 `_dit_dims` 烘进了图）**拒绝而非服务**，
  且拒绝时**不留半失效状态**（`_dit_graphs` 与 `_dit_attn` 都还在）。
- 契约钉 `grid_thw` / `visual_pos_masks` / `rope_cos` / `rope_sin`，
  融合档另钉 `input_ids`；**非融合档不钉** `input_ids`（那条路不消费它）。
- 稳态用 identity + `_version` 跳过比较：rope 表每帧 ~2.4 MB，
  在 Orin 的 ARM CPU 上逐帧比会吃掉这条路径本来要省的延迟（实测 **0.031 ms**）。
- **杠杆 #11 改判**：其"净亏"结论的唯一前提（一实例一帧）已被本节推翻
  ⇒ 复活条件两条均满足，改为"未落地、ROI 最高"（§6.15.6）。
- cross-KV 剩下的 8.52 ms 是 fp32 GEMM 本身；改走 bf16 tensor core **不是量化**
  （两臂输入逐位相同，只是乘加路径不同），实测差 **≈1 个 bf16 ULP**、cos 0.999999997，
  收益每观测 **5.22 ms** —— 详见 §6.18，此前"会动到 KV 数值所以不做"的措辞**已纠正**。

**原则对号**: #2（外层指标 ≠ 内部精度，这次是**反方向**：动作对 context 不敏感）、
#13（先诊断 13.10 ms 的构成再优化，不猜）、#15（两把尺并列报，口径写清）、
#16（锁频 + 中位数 + 分段归因；噪声底 ±0.06 ms 远小于 4.58 ms 的差）、
红线 #1（`_kv_slot_copy` 由 `_build_dit_attn` 与 `_refresh` **共用**，不各写一份）、
红线 #4（形状变化与契约篡改一律 raise）、
红线 #5（C5 的捕获计数与身份集合）、
AGENTS.md §1.1(6)（host cache 契约：原地写槽，不 repoint）、
§3.6（负控制必须可见退化——**这条正是第一版失败的地方**）、
§3.7（数据溯源，含那个 provenance 缺陷）。

**实验件**：`/tmp/continuous_infer_check.py`、`/tmp/kv_refresh_probe.py`，均未进仓
（其 GPU 门已在 §12 进仓为 `tests/test_orin_groot_n17_continuous.py`，
契约与计数部分更早已进仓为 CPU 测试）。

---

### 12. Phase 7: 连续推理的七道门进仓，并量到一个测试顺序缺陷

**输入**: §11 的七道门当时坐在 `/tmp/continuous_infer_check.py` 里。
AGENTS.md §3.6 要求负控制属于验收集，一个 `/tmp` 脚本不算。

**做了什么**: 把 C1/C3/C4/C5/C6/C7 提升为
`tests/test_orin_groot_n17_continuous.py`（**7 passed**，22.6 s）。
C2（换帧动作必须变）在 C1 的参数化里隐含覆盖。文件默认
`FLASHRT_GROOT_N17_FIXTURE_TAG=_sim`（与精度套件默认档**故意不同**：
连续门需要 8 个连续帧 + 1 个远端帧，只有仿真集捕了）。

**量到的东西**: 提升过程中 C4 报了 max&#124;d&#124; **3.1e-02**，
而 §11 记录的是 **0**。定位：`fe` 是 module-scoped 且携带可变观测状态，
C3 把前端留在 NEIGHBOUR，C4 的一次性臂于是拿 RUNUP 的 state 比 NEIGHBOUR 的
context —— 3.1e-02 正是 C2 量到的普通帧间差。脚本里 C4 跑在最前面所以从没暴露。
修法：C4 自己把 RUNUP 装回去，并加强成两臂（重载 vs `set_prompt` 自己那份
backbone **逐位相同**；同一观测下 `infer()` vs `infer(aux=)` **逐位相同**）。
**这不是前端缺陷**，前端一行未改。

**测试计数**: 真机档 **68 passed**（precision 17 + dispatch 51，56.2 s）；
仿真档 **75 passed**（+ continuous 7，66.6 s，需
`FLASHRT_GROOT_N17_FRAMES=100,107`，否则 precision 模块整体 skip —— 
skip 不是绿，§6.5 记过这条）。两次均锁频 1300.5 MHz，均在干净进程里跑
（§11 记过一次 NVML 断言：与 3.3B 参数的 N1.6 探针同进程会 OOM）。

**原则对号**: #1（先把基线可复现，再谈别的）、
红线 #5（C4 加强后的第一臂就是"相同输出要配得上证据"）、
AGENTS.md §3.6（负控制进验收集）。

---

### 13. Phase 3/4: 杠杆 #11 落地 —— backbone CUDA graph（§6.17）

**输入**: §6.13.2 判"不落地"、§6.15.6 改判"ROI 最高的未落地项"，
两条复活条件都已在 §6.15 满足。使用方点名要做。

**做了什么**: `_capture_backbone_graph` + 公开的 `run_backbone_graph(aux)` +
`infer(aux=...)` 分流 + `use_backbone_graph=True` 构造参数。
**`_kbb_forward` 一行未改**（红线 #1）—— 杠杆 #12 早就把它拆成
"载入常驻缓冲 + 纯 kernel forward"，那正是捕获的硬前置，所以这次几乎是装配工作。

**量到的**: 配对交替 A/B（同一前端同一次捕获，n=56/臂，中位数，锁频，仿真档）
backbone **63.52 → 59.52 ms（−4.00，1.0673×）**、
每观测 **115.44 → 111.40 ms（−4.03，1.0362×）**、
一次性 **259.7 ms ⇒ 64.9 帧回本**；
`backbone_features` 与解码动作**逐位相同**（max|d| 均 0）。
口径 C 因此从 3.29× → **3.47×**（vs 完整 `get_action`）。

**实测 vs 预测（#15）**: §6.13.2 探针说 4.54 ms / 57.7 帧，落地实测 4.00 ms / 64.9 帧。
差 12%，**方向是变小且原因明确**：探针量裸 `_kbb_forward`，落地量每观测真实边界，
而契约校验 + 输入载入这两步**图吃不掉**（一个是 host 逻辑，一个是 H2D copy），
它们同时抬高两臂、压低差值。⇒ 4.54 是"kernel 层可省量"，4.00 是"部署能拿到的量"。

**⚠️ 纠正了本节原计划里的一处错误**: 原写"`_kbb_load_inputs` 的 `copy_` 要在捕获区内"。
**不行** —— 载入的源是每观测新的 aux 张量，捕进图就把第一帧的源指针烘死，
此后 replay 永远读那一帧，正是 §6.15.3 的 stranded 失效模式。
正解是载入在捕获区外、replay 之前，只让目标槽常驻（RTX FP8 mixin 就是这么做的）。

**⚠️ 量测脚本自己也踩了一次同类坑**: 等价性检查**没传 `initial_noise`**，
两臂各抽一份新噪声 ⇒ 报 max|d| **4.96**，看着像 graph 破了数值。
与 §6.15.3、N1.6 §2.5 是同一类：**比对前先确认两臂输入是同一个**。

**门**: 逐位相同**不**当"graph 跑了"的证据（它本来就是设计目标，静默绕回 eager 也一样过），
另立 **C8**：`_kbb_graph` 存在、捕获**恰好 1 次**（此时已服务过 C1/C3/C4/C5 多次观测）、
再观测后图对象身份与 `vlsa_h` 的 `data_ptr` 均不变；
另加 eager 臂可达性与"校验不重复且在 replay 之前"两条源码级钉子。
C1–C7 **全部重跑通过**。dispatch **56 → 58**、continuous **7 → 9**；
真机档 **75 passed**、仿真档 **100 passed**。
那条反向绊线 `test_the_backbone_graph_entry_point_is_a_known_gap` **按设计触发并被替换**。

**原则对号**: #13（先拆缝再装配）、#14（可捕获性就是这条路的天花板）、
#15（实测 vs 预测都报并解释 12% 的差）、
#16（配对交替 + 中位数 + 锁频；臂间差 4.00 ms 远大于 ±0.3 ms 散布）、
红线 #1（全 additive）、红线 #5（逐位相同不算证据，另立 C8）。

**实验件**: `/tmp/backbone_graph_ab.py`（未进仓；结论已进 §6.17，门已进两个测试文件）。

---

### 14. Phase 3/4: cross-KV 投影改走 bf16 tensor core（§6.19）

**输入**: 使用方澄清「KV 别量化」指的是**不做 INT8**。据此复核发现
§6.15.5/§8 把"改走 bf16 tensor core"错误归类成"量化 KV"（§6.18.4 已纠正）：
两臂输入**逐位相同**，`.float()` 是精确提升，改的只是乘加路径。

**做了什么**: `_project_dit_cross_kv` 改用 `gemm.bf16_nn` + `fvk.add_bias_bf16`，
M/N/K 从权重形状读；旧 fp32 臂保留为 `_project_dit_cross_kv_fp32`（**只作参考**，
是三档门的对照臂）；删除 403 MB 的 `_dit_cross_kv_weights` 缓存。

**量到的**:
- 三档门（6 帧 / 两数据集 / 同一前端同一噪声）全过：存量 cross-K cos **0.999995672**、
  velocity **0.9994165**、decoded **0.999999073**。
- **诚实收益 每观测 −2.62 ms（111.17 → 108.55，1.0241×）**，refresh 8.50 → 4.75 ms，
  口径 C **3.47× → 3.56×**，另释放 **403 MB**。
- **代价**：4/6 帧 decoded 误差 +0.034…+0.137°，跨帧最坏值不变（0.4935°）。

**🔴 本节最大的教训：隔离微基准把收益高估了 2.3×。**
先量到 5.22 ms（合成张量、裸 matmul、无 bias/cast），再量到 6.15 ms
（真张量、32 个 GEMM 循环），**两个都错且同向**。真值是 2.62 ms，因为：
(a) refresh 里还有 **~3 ms 两臂相同的非 GEMM 开销**（掩码索引内含 `nonzero()` ⇒ host sync、
`.contiguous()`、32 次 `torch.empty`、32 次槽 copy）；
(b) 进流水线后 fp32 臂的一部分开销**与其他工作重叠**（refresh 省 3.75 而每观测只省 2.62）。
⇒ **三臂 A/B 里必须有"改动前的出厂实现"那一臂**：本次若拿参考臂（每次提升权重，13.36 ms）
当基线，会把收益报成 7.59 ms，**高估 2.9×**。
✅ 量测健康检查：fp32+缓存臂 8.50 ms 对上 §6.15.4 的 8.52，每观测 111.17 对上 §6.17 的 111.40。

**另一处实测纠正**: `bf16_nn_bias` 的 cuBLASLt epilogue 在 SM87 返回 `code=15`，
**M=20 与 M=128 都失败** ⇒ 不是 `pipeline_orin.py:520` 归因的"M 未 16 对齐"，
而是该 arch 上没有这个 epilogue。按红线 #1/#8 只报不改（§8）。

**决策记录**: **保留 bf16 臂为服务路径**，但明确标注为**使用方可否决的取舍**
（动的是使用方点名关注的 KV 通路），并把回退方式写进 §6.19.3 —— 
参考臂与其文档都还在仓里，回退不需要重新实现。

**测试**: dispatch **58 → 63**（新增 5 条 CPU 契约钉子，其中最重要的是
`bf16_nn` 的 **(M,N,K)** 参数顺序 —— 注意 **N 在 K 前**，换位在裸指针上
不会报错、只会算错）。CPU 桩改为影子绑定 fp32 参考臂（服务臂需 CUDA + `GemmRunner`，
而这些钉子测的是槽与图的记账）。仿真档 **105 passed**、真机档 **80 passed**，
G1–G5 与 C1–C8 全部重跑通过。

**原则对号**: #6、#13（探针先行，也正是探针给出了偏大的数字后被流水线 A/B 纠正）、
**#16（本节主教训）**、红线 #2/#4/#7。

**实验件**: `/tmp/kv_bf16_native_check.py`、`/tmp/kv_bf16_kernel_probe.py`、
`/tmp/kv_bf16_three_tier.py`、`/tmp/kv_bf16_latency_ab.py`（均未进仓）。

---

### 15. Phase 5: 长任务精度对比（全 episode 593 连续观测），并修掉两个真实缺陷（§6.20、§6.21）

**触发**: 使用方拍板"保留 bf16 臂"，并要求用其提供的仿真数据集做一次**长任务**精度对比。

**为什么 6 帧不够**: §6.19.1 的三档门只跑 6 帧——够判定"过门"，不够判定"代价的分布"，
更抓不到**状态漂移**（槽位陈旧、刷新漏刷），而后者正是连续推理通路的特征性故障。

**做法**: 新增 `tests/_helpers/groot_orin/long_horizon_ab.py`（**已进仓**），
`set_prompt` 一次 + **593 次 `infer(aux=...)`**（episode 0 = "Pick up green block and put it
on the blue block"，19.7 s 操作，距 prompt 0..592），三臂同一 frontend、同一 pinned
`initial_noise`。流式，不落盘 6.8 GB fixture；hooks 复用 `capture_aux.py`（红线 #2）。
HF eager 401.2 ms/帧，FlashRT 两臂 264.6 ms/帧，时钟全程锁定。

**harness 先自校验**: frame 100/101/102 上与 fixture 版三档门逐项吻合
（t1 cos 0.999995876 vs 0.99999588、bf16 decoded 0.2467° 同值、t2 0.9994513 同值）
⇒ 活体集成与磁盘 fixture 同口径（原则 #1）。

**结论**: **撤回 §6.19.3 的"使用方可否决的取舍"定性**。n=593 时两臂均值差 **+0.0012°**、
**36.1% 的帧逐值相等**、全程最坏 bf16 **0.4935°** 优于 fp32 **0.5808°**、三档门 **0/593 违反**、
decoded cos 最低 **0.999998845**。
**决定性的一条**: A−B 对 prompt 距离的相关 **+0.0076**、斜率 **+3.3e-06 °/帧**（592 帧累计 +0.002°）
⇒ **排除状态漂移**。而 fp32 参考臂自己的距离趋势（r=+0.2641）比 bf16 臂（+0.2107）**更强**，
说明那点趋势是流水线/内容固有的（相对误差与距离 **负**相关 −0.3198；最后一桶两臂同时抬高且 signal 最大）。

**教训**: 原则 #16 的"perf delta 要 hygiene + triage"**同样适用于精度 delta，且样本量要求更高**
——延迟可靠"中位数 of N 次重复"自带降噪，精度每帧只有一次采样，n=6 的置信区间宽到能改变结论**符号**。
**规则：任何"取舍"定性都要有长时程或大样本支撑；否则只写"过门"，不写"代价"。**

**顺带修掉两个真实缺陷（都有负控，§6.21）**：

1. **inference tensor 打在每观测通路上**（第一帧就炸）。根因是 `getattr(t, "_version", None)`
   **不会**返回默认值——`_version` 是个**会抛异常的 property**。`inference_mode` 正是官方
   `get_action` 的推荐跑法 ⇒ 活体集成不可用。此前全部 A/B 走磁盘 fixture，是**共同盲区**。
   更深的一个洞：备忘会缓存 `None` 版本，而 `None == None` 成立 ⇒ **原地改写被静默放行**。
   修法：版本号不可读 ⇒ 绝不走 identity 快路径（退化到慢而正确的一边）+ 一次性 warning。
   dispatch **63 → 67**。**同款写法在 `groot_n17_rtx_fp8.py:211/:244`**（RTX FP8 / SM89 / Thor FP8），
   按红线 #1 只报不改（§8）。
2. **视频读取器丢掉整段 episode 的尾巴**：593 帧读到 **581** 抛 `EOFError`。数据没问题
   （视频/parquet/episodes.jsonl 四者都是 593）。根因：`_decode` 每次重建
   `container.decode(stream)` 并在循环内 `return`，**弃置活生成器** ⇒ PyAV finalize 时 flush 编解码器
   ⇒ 队列里的尾部帧不可达。20 行单变量探针定位（弃置式 581 失败 / 持有式 593 通过；
   合成 600 帧复现在 587）。修法：跨调用持有**挂起**的迭代器，seek 前置 `None`。
   既有调用都是散点帧号，所以只有"整段 episode"这种读法会踩。
   新增 `tests/test_lerobot_video.py`（**5 条，CPU-only，0.78 s**，自带合成视频，不依赖仓外数据集）。

**一次测量卫生事故**: 后台长时程进程处于 `Z`（未被回收）时立刻跑真机档闸门 → **4 failed**（全在 bf16 档）。
逐项隔离（单函数 4 passed → 模块 17 passed → dispatch+precision 84 passed → **同一条命令重跑 89 passed**）
⇒ 与代码改动**无因果**，是 CUDA 上下文未释放的瞬态；bf16 档要额外构造 frontend，最先顶到内存压力。
**规则：后台 GPU 任务报 zombie 后，先确认进程被回收再跑闸门。**

**测试**: dispatch **63 → 67**、新增 `test_lerobot_video.py` **5**。
仿真档 **105 → 114 passed**（67.4 s），真机档 **80 → 89 passed**（56.5 s），时钟全程 1300500000。

**原则对号**: #1、#13（单变量探针）、#16（瞬态 ≠ 回归；本节主教训是精度 delta 的样本量）、
红线 #1/#2/#4/#7、AGENTS.md §3.6（两个修复都有负控）、§3.7（593 帧全部真实采集）。

**产物**: `tests/_helpers/groot_orin/long_horizon_ab.py`（进仓）、`tests/test_lerobot_video.py`（进仓）、
`/tmp/n17_long_ep0_full.jsonl`（593 条逐帧记录，未进仓；结论已进 §6.20）。

---

### 16. Phase 3/4: 融合 bf16 rotate-half RoPE —— 撤销一个空判的 `blocked_on_kernel`（杠杆 #9，§6.22）

**触发**: 使用方要求"继续完善 N1.7，例如检查是否还有可做算子融合的，或者引入 FA2 等"。

**第一件事是核实 FA2 这条**，结果是**它早就花掉了**：bf16 FA2 已在全部四个注意力位点
（ViT self-attn、LLM self-attn、VLLN 的 4 个 `vl_self_attention`、DiT 的奇偶交替 self/cross），
出厂配置的 GPU 普查里注意力合计 **2.335 ms = 2.28%**（172 次 launch，逐项对得上位点数）
⇒ 清零也只值 2.3 ms，没有可动的空间。
⚠️ 这个数**当轮先记成了 0.312 ms = 0.3%，错了 7.5×**，是收尾自查时发现的（见下面教训 4）。
**规则：使用方点名的杠杆也要先量它还剩多少，再决定做不做**（原则 #15 的"锚点"侧）。

**重排杠杆树时翻出一个空判**。§6.6.4 记的"SM87 缺 bf16 rope kernel ⇒ `blocked_on_kernel`"
是**错的**：`rope_neox_qk_bf16` **早就编好了**，只是它在
`flash_rt_qwen3_vl_kernels`（`csrc/qwen3_vl_bindings.cpp` 单独绑定），
而审计只枚举了 `flash_rt_kernels`。本仓 SM87 构建有**三个**扩展模块。
更要紧的是**代价被记小了 4.4×**：杠杆树写 ~2.7 ms，归因探针（把 `_rope_qk` 换成 no-op，
同进程同流水线量差）实测 **11.902 ms GPU = backbone 的 19.5%**。

**做法**: `_rope_neox_qk_kernel()` 惰性解析 + 缺失时**响亮告警**（红线 #4，不静默降级）；
`_rope_qk(Q,K,tbufs,rows,q_heads,k_heads,head_dim,stream)` 一个入口两处调用
（ViT `q_heads==k_heads==16`，LLM 是 GQA `16/8`）；`_rope_half_table()` 把 `cat(emb,emb)`
的表切半并**校验两半确实相同**（不相同就退回 shim 并播报，因为做错不报错只掉 cos）；
`set_prompt` 建齐 4 张半表（`.contiguous()`，kernel 按 `row*(hd/2)+d` 平铺寻址，跨步视图会静默读错行）。
`_rope_rotate_half` 保留为**回落 + 数值参考**。

**结论**: **逐位相同，端到端**。`backbone_features` / `infer` 输出 / decoded action
三者 `torch.equal` 全 True，两个数据集各验一次；vs HF 的 cos 与 max° **两臂同值**
（真机 0.999999398 / 0.4935°，仿真 0.999999496 / 0.2467°）⇒ **精度门数值一个没动**。
每观测 **107.752 → 97.281 ms（−10.471，1.1076×，仿真）** / 106.970 → 98.137（真机）；
边界 **106.47 → 94.35 ms（3.38× → 3.82×）**；口径 C **3.56× → 3.97×**；
完整 `get_action` 等价 **~134 → ~122 ms（2.89× → 3.18×）**。
backbone eager **63.03 → 51.08**（ViT −9.15 / LLM −2.63），CPU 提交 **54.54 → 20.82**（800 次 launch 没了）。
**measured vs predicted**：预测 11.902、普查实测 **11.478（96.4%）**；
差的 3.6% 正是融合 kernel 自己的 1.07 ms —— no-op 归因法的**已知系统性偏高**，
比 §6.19.2 那个 2.3× 的隔离微基准偏差小两个量级，因为这次是同进程同流水线。

**同轮否决的两条**（原则 #13：先微基准再写）：
`silu_mul_qwen36_bf16` **逐位相同**但只省 **0.092 ms/backbone（0.08%）** ⇒ 低于噪声地板，
不值得动一条已出厂的数值路径；`bf16_nn_bias` 符号存在但 SM87 运行时 `code=15`，
**重验 M=20/128/141/512 全失败** ⇒ 与 16 对齐无关，`pipeline_orin.py:520` 的归因注释是错的
（按红线 #1 未改，已进 §8 报给 owner；那 8.880 ms / 848 次调用是当前**最大的非 GEMM 单项**）。

**教训**: 四条。
**(1) `blocked_on_kernel` 判定必须写明"在哪些模块里查过"**——本仓三个扩展，漏一个就把
已建好的 kernel 记成缺口，还顺带把代价记小 4.4×（缺口一旦入账就没人再回去量）。
**(2) 归因探针要在同进程同流水线里做**——§6.19.2 的隔离微基准偏 2.3×，本次偏 3.6%。
**(3) 逐位相同的改动，门要打在契约上而不是数值上**——两臂一路相同，任何 cos 门都抓不到
接线错误（接错了也相同，直到某天突然不同）。13 条钉子全是契约：半表是实体拷贝、
GQA 的 `q_heads/k_heads` 顺序、原地语义、缺失时回落且 kernel 不得执行、解析器只导入一次、
两处 `tbufs` 都带半表、forward 里不得再直调 shim。**负控**：把 ViT 位点还原成两行
`_rope_rotate_half` ⇒ 2 条钉子立刻红。
**(4) 关键字分桶对 C++ mangled 模板名不可靠，且会同时朝两个方向错**——收尾自查才发现
FA2 被记成 0.312 ms / 0.3%（实为 **2.335 ms / 2.28%**，偏低 **7.5×**）：
`fa2_vendor::flash_fwd_kernel<…cutlass::arch::Sm80…>` 含 `cutlass`/`sm80` 而被**排在前面**的
GEMM 规则吃掉，同时 attention 规则里的 `"flash"` 又匹配命名空间 `flash_rt::`，
于是那 0.312 ms 根本是漏网的 FlashRT 自有 kernel。连带"GEMM 73.3%"也虚高（真值 **64.1%**）。
这是**同一份普查第二次**栽在分桶上（第一次是 `DefaultGemmWi…` 被归进 copy/cast）。
更正的自证：**GEMM 在两臂之间只差 0.12 ms**，而 rope 本来就不该碰 GEMM；
两臂 9.818 ms 的总差可由 elementwise −8.948 / cat −1.581 / rope +1.053 逐项加出。
**规则：分桶按精确名字前缀、attention 排在 GEMM 之前，并打印每个桶的成员名单而不只是总量**
——两个错都是"总量看着合理"才活下来的。⚠️ 结论这次没变（FA2 仍是 2.3% 的小头），
**但画像数字错了不会立刻现形，它只在未来的排序决策里发作**：
若按 0.3% 去否决一个值 1 ms 的注意力改动就会错杀。详见 §6.22.1 / §6.22.8。

**测试**: dispatch **67 → 80**（+13）。闸门：仿真档 **114 → 127 passed**（66.4 s）、
真机档 **89 → 102 passed**（55.9 s），时钟全程 1300500000。

**原则对号**: #13（先微基准；两条否决）、#14（cublaslt 的图可捕获性**未验就不落地**）、
#15（FA2 先量剩余空间；每个门报 measured vs predicted；给出排序后的剩余杠杆菜单）、
#16（配对交替 + 中位数 + 锁频；健康检查 shim 臂 107.752 vs §6.19 的 108.55，差 0.7% ⇒ 基线没漂）、
红线 #1/#4/#5（40 vs 80 次调用计数 + 图冻结检查）/#7/#8、AGENTS.md §2.9（能力探针式接入，缺失是回落不是拒绝）、§3.6。

**产物**: `pipeline_orin.py`（`_rope_qk` / `_rope_neox_qk_kernel` / `_rope_half_table`）、
`groot_n17_orin.py`（4 张半表 + 两处 `tbufs`）、`tests/test_orin_groot_n17_dispatch.py`（+13）、
§6.22（6.22.1–6.22.8）。未进仓的探针：
`/tmp/n17_{rope_attribution,rope_kernel_microbench,rope_kernel_ab,remaining_time}.py`，
以及普查两件 —— `/tmp/n17_kernel_census.py`（⚠️ **分桶有缺陷，仅存档**）与
`/tmp/n17_census_corrected.py`（**更正版，两臂同进程**）+ `/tmp/n17_attn_bucket_probe.py`（定位用）。

---

### 17. Phase 3/4: 每观测图像通路改走纯 torch GPU 链（杠杆 #14，§6.23）

**触发**: 使用方要求"处理 HF pre/post，这部分 Thor/RTX 是如何处理"，并给了两条边界——
入口形态是"**裸输入**，即输入就是图和 prompt"；实现上"**既然当前是 ORIN，就不应该看
NPU/AMD，而尽量参考 N1.6/Pi0.5 的方案，走纯 torch 后端，依靠 GPU 加速处理**"。

**第一件事仍是核实而不是照做**：逐符号查了本仓 **7 个** N1.7 CUDA 前端
（Thor、Thor FP8、RTX、RTX FP16/FP8、SM89、AMD、Orin），**没有一个跑图像通路**——
全都要求调用方交已处理好的 `aux` 束，即厂商 `Gr00tN1d7Processor` **每观测跑一遍**。
唯一替换掉它的是 Ascend NPU 通路（按要求不看）。⇒ 这不是"Orin 漏了一项"，
而是**整条 CUDA 家族共同的空白**，而 Orin 是它代价最高的地方（纯标量 host 代码跑在弱 ARM CPU 上）。
可复用的先例是 N1.6 `groot_rtx.py:1054/1191` 与 Pi0.5 `pi05_rtx.py:1498/2060` 的裸输入房型。

**做法**: 新模块 `_groot_n17_preprocess.py`（`_groot_n17_fusion.py` 的兄弟，照抄其房型），
把厂商 eval 链整条搬到设备上：letterbox → **稠密 fp32 matmul 的连续覆盖 INTER_AREA**
→ 中心裁剪 → 放大 → rescale/normalize → Qwen2-VL merge-block patchify。
前端**只加不改**：`infer(state, *, frames=None, aux=None, …)` 多一个关键字-only 参数，
几何从 **checkpoint 自己的 `processor_config.json`** 读（读不到就拒绝，不猜），
plan 按 `(h,w,shortest,crop_fraction)` 缓存，H2D 走**常驻 pinned buffer**，
图像通路**在任何 CUDA graph 之外**（§6.17 的纠正：每观测的**源**不得烘进捕获）。
六处**响亮拒绝**：`frames=` 与 `aux["pixel_values"]` 同时给、dict-of-views、
非 uint8、非 4-D、未开融合、行数与契约不符。

**结论**: 三臂配对交替（锁频，中位数 of 11，**4 帧真机数据 × 两个数据集**），
**vendor 臂自证**（其 `pixel_values` 与 fixture **bf16 逐位相同，4/4**）：
预处理 **12.916–14.027 → 1.448–1.485 ms**；
每观测 **112.374–115.499 → 99.579–100.428 ms（省 11.945–15.410，1.1189–1.1540×）**。
**measured vs predicted**：预测 ~8.6 ms ⇒ **偏低 1.4–1.8×**。
**精度（12 帧 × 3 臂）**：cv2 臂与 HF **九位小数全同** ⇒ T1 的移动全部归因于 GPU 链；
T3/T4 **未放宽门**（`THR_CONSUMED = 0.999`），只有 T1 新增 `THR_IMAGE_GPU_BACKBONE = 0.985`
（取在最坏实测帧再往下一整个观测跨度）；T3 变化 **−0.000087…+0.000153**（**有时更好**）、
T4 max 误差 **4 好 / 4 差 / 4 等** ⇒ 对称、**无系统性退化**（§6.20 的 n=4 教训在此执行为 n=12）。
负控（跳过裁剪+放大）T1 **0.9918 → 0.8989**，掉穿门 0.086。
**新增口径 D**（每观测含图像预处理）：真机 **3.439/3.444 → 3.896/3.883×**、
仿真 **3.344/3.437 → 3.859/3.846×**；⚠️ 同时**下修口径 C**（方法论纠正 5）。
顺带把每观测的 host 比较从 **5 次 / 77156 字节**降到 **0 次 / 0 字节**
（契约项按**身份**重新呈现，命中校验器快路径）。

**精确性**（本节最硬的部分）：**缩小步逐位精确且可证** —— 把尺度写成最简分数
`s = src/dst = p/q`，连续覆盖的权重是 `m_i/p`（求和为 `p` 的非负整数），
精确输出是 `N/p`；平局需要 `2N = p(2k+1)`，**`p` 奇数时不可能** ⇒ 任何输入都无平局，
fp32 误差就翻不掉舍入。出厂的 640→256 是 **5/2**：权重 {0.2, 0.4}、3 抽头、
`out = (a + 2b + 2c)/5`。实测 vs `cv2.INTER_AREA` **max = 0**（4 帧 × 2 数据集 × 393216 像素，
numpy fp64 与 torch fp32/CUDA 各一遍），并**穷举全部 256³ 输入**（2 个抽头模式 ×
16,777,216 = **3350 万次**）**0 分歧**，负控扰动一个权重 1e-3 ⇒ 分歧 > 0。
条件**被记录成 `shrink_exact` 并在使用时强制**（512 宽相机给 `p = 2`，
`out = (a+b)/2` 在每个奇数和上都平局 ⇒ **拒绝执行**）；fp16/bf16 权重与 TF32 **都被拒**。
⚠️ 顺带量到 **torch 自己的 `interpolate(mode="area")` 偏 18 LSB**（整数源边界 + 普通均值），不能用。
**放大步（243→256）不精确**：真机帧 **max = 1 LSB、mean 0.051–0.069、5.1–6.9% 像素**，
**fp64 下相同 ⇒ 算法性而非精度性**；4 种表述（连续覆盖 / OpenCV 的 1/2048 两抽头定点，
各带与不带中间移位，配 round/trunc/round-half-even，fp32/fp64/int32）**全部收敛到 max = 1**，
出厂的浮点覆盖形式**差异占比最低** ⇒ **声明未知**，不编说法。
**patchify 在 bf16 下 `torch.equal`**（0/786432）。

**教训**: 五条（前两条是两个探针 harness bug）。
**(1) 精确性探针必须自带一个"答案已知"的用例** —— 放大步探针第一版写成
`A @ b.transpose(0,2,1)`，而测试图是**方图** ⇒ 转置被静默吸收，
在**已知逐位精确**的 640→256 上报出 **max = 155**；只因那个已知用例被当 harness 自检跑了一遍才当场抓住。
**(2) 探针必须调用出厂的那个函数，不能重抄一份算术** —— 第二版在两次 matmul 后**各舍入一次**
（双重舍入），于是在 320→256 上报 **max = 1**，而出厂的单次舍入给 **max = 0**；
一个**不存在的残差**差点被写进文档。
**(3) 分段中位数相加不是量测** —— 计划里的 "~10.3 ms" 来自分段之和，整链实测
**12.916–14.027 ms**（偏低 1.25–1.36×）；分段之和 11.16–14.44 **确实包住**整链值 ⇒
分段没错，错在拿它当总量。本文档那个 "~27.55 ms pre/post" 是同一毛病的另一面
（两个独立总量**相减**）。
**(4) "为什么有残差"这种归因要探针，不要合理的故事** —— "残差在 OpenCV 放大分支的
权重推导里"这句**已经写进模块 docstring**，读起来完全合理，而且**是错的**：
one-hot float32 输入能单独隔离权重，一跑就显示 OpenCV 的有效权重**就是**连续覆盖
（四个尺度吻合到 **7.2e-8**）⇒ 残差在 **uint8 累加/舍入路径**上。
**(5) 跑门禁套件时不许编辑被测源文件**（方法论纠正 6）—— 仿真档首轮报
`test_the_eager_backbone_arm_stays_reachable` 失败，单独跑通过、三条断言手工验证也成立；
根因是套件运行途中改了 `groot_n17_orin.py` 的 docstring，而那条测试用
`inspect.getsource(CLS.infer)` 做源码钉子，`inspect` 走 `linecache` **会按 mtime 重读盘**，
进程里的 code object 仍带**改动前**的 `co_firstlineno` ⇒ 抽出的源码块整体错位 3 行。
**只在做源码钉子的测试上出现，数值测试完全不受影响** ⇒ 这是 §6.22 教训 3
（大量使用源码钉子）带来的一个**新失效面**。重跑后仿真档 **204 passed**。
📌 另记三条 torch 2.3 的坑：`Tensor._version` 是**会抛异常的 property**（inference tensor 上）、
`np.abs` 作用在 uint8 差值上**模 256 回绕**（一次 "max=255" 其实是比对 bug）、
`np.arange(n, dtype=np.uint8) * 255` **溢出**（"渐变图"其实是回绕噪声——
这一条是被负控自己揭发的：扰动只让输出动了 0.12 而不是 2.5）。

**Stage 2（HF-free `set_prompt(str)`）已定位、被阻塞、且没有延迟收益**：
prompt 侧一切**都是 prompt-scoped，`set_prompt` 已经提走** ⇒ Stage 2 买的是
**部署独立性**（原则 #10），不是速度。🔴 阻塞项是
`flash_rt/models/groot_n17/mrope_table.py` **既休眠又错**（对 4/4 fixture 的
`rope_cos`/`rope_sin` maxdiff **2.2e-02…3.1e-02**，bf16 约 5–10 ULP），
而 `tests/_helpers/groot_n17/mrope_ref.py` 那句"对 HF 验证过逐位精确"
是在**旋转后的 Q/K** 上做的，**表本身从来没被验证过**。
诊断路径已写进 §8（`mrope_section=[24,20,20]` 铺 64 槽只覆盖 **62** 个，
槽 60/61/62 留在 `clone()` 的 T 轴上；**逐列**比对即可判定）。按红线 #1 **未改**它。

**测试**: precision **17 → 29**（G6.1–G6.4 各 ×2 帧、G6.5–G6.6 各 ×1 帧、G6.7 ×2 帧）；
新增 `tests/test_orin_groot_n17_preprocess.py` **65 条，CPU-only，2.96 s，不需 checkpoint**
（P1 算子结构 / P2 精确性含 3350 万穷举 / P3 vs cv2 含负控 / P4 plan 与 6 种拒绝 /
P5 patchify 对 6 层显式索引循环 / P6 输入拒绝 / P7 前端接线含校验器快路径计数）；
dispatch **不变 80**。闸门：真机档 **102 → 179 passed（71.6 s）**、
仿真档 **127 → 204 passed（82.1 s）**，时钟全程 1300500000。

**原则对号**: #10（加速路径必须能在厂商训练代码不在场时跑——本模块**零 HF/albumentations/PIL 依赖**）、
#13（先微基准；**故意不做** gather/稀疏化，因为精确性是对稠密 fp32 matmul 验的）、
#14（天花板检查：`fvk.patch_im2col_uint8` 存在但几何被 SigLIP 锁死 ⇒ 只登记不采用）、
#15（每个门报 measured vs predicted，并解释 1.4–1.8× 的偏差来自哪）、
#16（配对交替 + 中位数 + 锁频；**隔离 1.448–1.485 vs 配对交替 1.803–1.854**，
门里断言较保守的那个）、#6（配置以文件为准，不硬编码 0.9）、
红线 #1（只加不改）/#2（复用 N1.6·Pi0.5 的裸输入房型与 `_groot_n17_fusion` 的模块房型）/
#4（六处响亮拒绝 + 放大步声明未知）/#5（调用计数 + "GPU 臂与 fixture **不同**"）/
#7（每个引用的 API 都对过源码，包括推翻自己 docstring 里那句归因）/#8（未写任何新 kernel）、
AGENTS.md §3.6（负控）/§3.7（**全部 12 帧来自真实数据集**，两个数据集，无合成输入）/§3.8（配对交替）。

**产物**: `flash_rt/frontends/torch/_groot_n17_preprocess.py`（新）、
`groot_n17_orin.py`（`_read_processor_geometry` / `_image_plan` / `_frames_to_device` /
`_pixel_values_from_frames` / `_observation_aux` + `infer(frames=)`）、
`tests/test_orin_groot_n17_preprocess.py`（新，65）、
`tests/test_orin_groot_n17_precision.py`（G6 段，+12）、§6.23（6.23.1–6.23.7）。
未进仓的探针：`/tmp/n17_{vendor_arm_ab,enlarge_exact_probe,tier_table,frames_arm_smoke,slowpath_count}.py`、
`/tmp/verify_preprocess_module.py`。

### 18. Phase 5: 三方审阅的六项发现——**先复验为真，再修**（§6.24）

**触发**: 使用方要求审阅已实现的 Orin × N1.7 代码（原仓风格 / 测试验证 / 推理准确性）。
审阅（风格 / 测试覆盖 / 数值正确性三路）产出 7 项，使用方给了两条边界：
"**1~6 需要进行实现/补充，但还需要再次确认 1～6 内的问题是真实存在的，即再次验证后再开始修复**"，
以及"**对于 7，咱不该动**"。

**做法**: 严格按边界分成两段，**Phase V 期间一行代码都没改**。
每项配一个**可falsify的探针**和一句**预期观察**：探针不复现就当假阳性上报、不修。
V1/V2 上 GPU（真 fixture、锁频、一次一个前端——连续套件的模块 docstring 记过两个前端同驻会 OOM 这块板子），
V3–V6 是源码核实加小的 CPU 探针。

**结论**: **六项全部复现，0 假阳性**；但审阅支撑第 2 项的那个数字**本身错了 4×**——
它写"每次推理 ~600 MB cast 流量"，而 `_compute_dit_adaln_modulators` 是**每去噪步**跑一次，
按真实权重形状（32 × `(1536,3072)` bf16）重算是 **576.38 MiB/步 ⇒ 2305.50 MiB/次推理**。
第 1 项是六项里唯一影响数值的：DiT 图把 `Sa = action_horizon + 1` 与 bucket 数按值烘死，
而修复前唯一的护栏只查步数 ⇒ 覆写 `action_horizon` 会**静默重放旧图并返回一个形状完好的动作**，
实测偏 **21.4859°**（捕获@20→请求40）/ **3.3572°**（出厂顺序 40→20）/ **3.5810°**（buckets→200），
**三者都不报错**；`action_horizon` 在三个测试文件里出现 **0 次**，而 §6.16 的对齐表把它标成无条件 ✅。

**决定修法的那一条探针是 V1c**：eager 臂自己有没有 horizon 状态？
`_run_dit` 是 `if not hasattr(self, "_dit_attn"): self._build_dit_attn(Sa)`，所以看着像有。
实测**没有**——Sa=41 的前端跑 eager@20 与 Sa=21 的前端跑 eager@20，**max&#124;d&#124; = 0**。
⇒ 回落是安全的，修法取 **warn 一次 + `graphs = None`**（`_rope_qk` 先例），而不是 raise。
如果 V1c 反过来，同一个缺陷就必须修成 raise：回落只会把"错得静默"换成"错得响亮"。

**红线 #5 在本轮的具体形态**：图臂与 eager 臂**在参数相符时逐位相同**，
所以"输出对了"完全不能证明图臂跑过——一个静默回落的回归能过掉这里**每一道数值门**。
⇒ 除了 F1 的参数护栏门，另加 **C9** 用**重放计数**钉住图臂每观测真的重放了 4 次。
（`torch.cuda.CUDAGraph` 是 pybind 对象、**没有 `__dict__`**，
`monkeypatch.setattr(graph, "replay", …)` 挂不上去，只能用代理 list 计数。）

**量到的**: 回落代价**配对交替实测** **98.068 → 165.290 ms**，即 **+67.222 ms（1.6855×）**；
第二次跑 **+64.120 ms（1.6564×）** ⇒ **告警文案写区间 `+64…+67 ms（1.66-1.69x）`，不写单次值**
（第一版硬写了 1.6855×，属于把一次测量当常量）。归因到 AdaLN 调制器重算
**48.021 ms/次推理（12.005 ms/步）**。修后 V1 的三条陈旧路径全部 **max&#124;d&#124; 0.000000e+00**。

**精确性**: F6 的融合参考是**从双线性定义独立转写**的（显式 `(row, col)` gather +
显式四重 merge-block 循环，正是两处可能藏转置的地方），并在**真表**上自证：
`visual.pos_embed.weight` 实测 **`(2304,1024)`**（side 48，checkpoint 存 F32，
Orin 的 weight spec 把每个 `ToFp16` 改写成 `ToBf16`）、**absmax 32.75 / std 0.6019**；
参考 vs 实现 **max&#124;d&#124; = 0.0**（`torch.equal = True`），
fp32-then-round 变体 vs 实现 **max&#124;d&#124; = 0.125**（该量级下 1 bf16 ULP，**36.4% 元素不同**）
——与 `_groot_n17_fusion.py` 注释里记的那个数**独立复现一致**。
合成替身表**按真表量级分布造**（std 0.6 主体 + 1% 行 ×12 重尾），因为纯 `randn`
会让这个负控小到不足以当证据（1 ULP 在 &#124;x&#124;≈0.6 只有 ~0.004）。

**负控（AGENTS.md §3.6，逐个跑红）**: F1 换回旧护栏 ⇒ **2 failed**
（`4 graph replays … the stale graphs were served`）；F5 四个断点 ⇒ **3/4/1/2 red**；
F6 七个断点 ⇒ **1/1/2/1/3/4/1 red**；C6 的灾难界从 10.0° 收紧到 1.0° ⇒ red 且**打印出 1.2337°**，
与 §6.15.3 记录值精确一致。所有断点跑完**源码逐字节还原**并复跑至绿。

**原则对号**: #7（**先复验再修**；并因此推翻审阅的 4× 数字、推翻自己告警文案里的单次值）、
#15/#16（代价**实测**不继承估计；配对交替、锁频）、
红线 #1（改动全在 Orin 自己的 override 里，Thor/RTX-FP16 基类一行未动；F5/F6 只加测试，
`lerobot_video.py` 400 行、`attn_backend_groot_n17_orin.py` 98 行、`_groot_n17_fusion.py` 的代码都没动）、
#2（新测试全部进**已有**文件，不开新文件；复用 `_obs_stub`/`_FakeAttn`/`_runtime_stub` 的既有 stub 房型）、
#4（静默回落改成一次性响亮告警，文案带实测代价与补救办法）、
#5（重放计数 + 调用计数 + 负控，不接受"输出一样"当证据）、
AGENTS.md §3.7（F5 的夹具照**真实 LeRobot v2.1 录制**的布局造：两种列命名约定都取自真数据集，
`chunks_size=1` 是为了让 `ep // chunks_size` 这段算术**可观察**）。

**⚠️ 第 7 项按要求完全没动**：`groot_n17_thor.py` / `groot_n17_rtx_fp16.py` /
`groot_n17_rtx_fp8.py` / `groot_n17_rtx_sm89.py` 与 AMD 前端一行未改，
§8 **也没有**为那三个继承来的同名缺陷加条目。因此 §6.16 对齐表本轮**只改了 Orin 那一格**，
Thor / RTX 两格保持原样（未在本轮复核）。

**测试计数**: dispatch **80 → 102**、precision **29 → 31**、`test_lerobot_video` **5 → 20**、
continuous **9 → 10**、preprocess **65** 不变、N1.6 后端门 **16** 不变；
**真机档 179 → 218 passed（74.44 s）**、**仿真档 204 → 244 passed, 0 skipped（86.13 s）**，
两档全程锁频 **1300500000**、干净进程。dispatch 另验过**随机序**跑同样 102 passed
（新增测试不得依赖别的测试先导入过什么，见 §6.24.7 教训 2）。

**产物**: `groot_n17_orin.py`（`_warned_dit_graph_params` / `_note_dit_graph_bypass` /
`_capture_dit_graphs` 签名与 `(steps, buckets)` 键控与 `_dit_graph_params` / `infer` 三元组护栏 /
模块 docstring 与 `:1824` 两处纠正）、`_groot_n17_fusion.py`（**仅 Args 纠正**，代码未动）、
`tests/test_orin_groot_n17_{precision,dispatch,continuous}.py`、`tests/test_lerobot_video.py`、
§6.24（6.24.1–6.24.7）+ §0.1/§0.2 四行 + §6.16 对齐表一格。
未进仓的探针：`/tmp/n17_{horizon_probe,fallthrough_cost,f5_negctl,f6_negctl,f6_ref,obs_split}.py`。

---

## 8. 待办

**已闭环**

- [x] HF eager 真机数据基线（N1.7），落 fixture + aux 束
- [x] task #9：INT8 在真实 DiT 形状上的 A/B（决定精度档）
- [x] `("groot_n17","torch","rtx_sm87")` 注册进 `_PIPELINE_MAP` **与** `_SM87_ALLOWED`
- [x] Orin frontend + `models/groot_n17/pipeline_orin.py`
- [x] 精度门（被消费张量 cos ≥0.995 逐级，`THR_FUSED_CONSUMED` 含理由 / E2E）+ 图安全门（capture + stale-value）
- [x] **image→embeds 融合**（杠杆 #10，§6.7）：`aux["llm_input_embeds"]` 依赖已摘掉；
      真实独立部署成本 **174.86 → 124.56 ms（1.40×）**，两个口径重合于 2.89×；23 passed
- [x] **INT8 DiT 的真激活 fake-quant 门**（§6.8）：228 个 DiT Linear 打 quant→dequant，
      跑厂商自己的 `get_action`，两帧真机数据 —— per-row W8A8 **cos ≥0.999999**、
      最坏 **0.255°**（小于 bf16 融合档已有的 0.49°）⇒ **杠杆 #3 精度侧解锁**，
      且 **DiT 不需要 QuaRot**；per-tensor 差 2.5× ⇒ per-row 是硬要求
- [x] **接 INT8 DiT kernel**（杠杆 #3，§6.9）：`cutlass_int8_rowwise_bf16out` +
      `quantize_int8_rowwise`（device-side scale）接到 DiT 的 160 个 GEMM，
      每层 4 次量化（`xn1` 被 Q/K/V 共享）。**实测 DiT 58.5 → 41.5 ms（1.39–1.42×）**，
      边界 **124.9 → 108.0–108.8 ms（3.31–3.34× vs HF）**。G1–G5 全过、未放宽门、
      graph≡eager bit-identical
- [x] **默认档决策**：使用方拍板 **INT8 为出厂默认**（§6.9.6）。
      `use_int8_dit` 默认翻成 `True`；测试用 `inspect.signature` 自动跟随默认档，
      另加 `test_int8_dit_is_the_shipped_default` 无 GPU 钉住 ⇒ **31 passed**
      （precision 17 + dispatch 14）
- [x] **LLM / ViT 的动态量化门**（§6.10，按使用方"优先动态量化、尽量不用 QuaRot"执行）：
      **LLM 过门**（per-row 动态，全 112 Lin 含 QKV/O，最坏 0.573°；FFN-only 0.247°）
      ⇒ **§6.5 判定 2 推翻，LLM 不需要 QuaRot**；
      **ViT 不过门**（tap `vit_block_17` 0.971083 vs 门 0.998，per-tensor 崩到 0.5127）
      ⇒ §6.5 判定 3 确认，**杠杆 #5 关闭、ViT 保持 bf16**
- [x] **接 LLM INT8 FFN kernel**（杠杆 #4）—— **做完、量完、判定不落地、代码已回退**（§6.11）：
      微基准（真实权重+真实激活）预测省 **3.80 ms**，与权重流量 roofline 的 3.82 ms 吻合；
      接入后 kernel 计数实测 **32 quantize + 48 INT8 GEMM / backbone**（红线 #5 的证据）。
      **两条独立的否决理由**：
      **(a) 精度不过门** —— `backbone_features`（DiT 真正消费的张量）cos **0.992077 < 0.995**，
      而 G4 是 cos 0.999999 / **0.329°（比 bf16 档自己的 0.493° 还小）**
      ⇒ **只看 G4 会放过它（第三次现形）**，且暴露出 §6.10.1 的 fake-quant 门
      **漏量了 VLSA 之后那一级**；
      **(b) 收益拿不到** —— GPU 层面确实省 3.93 ms（捕获后 **4.77 ms**），
      但 eager 墙钟只省 **0.44 / 0.85 / 0.97 ms**（3 个进程），因为 backbone 是
      **CPU 提交受限**。⇒ 放宽门换 0.6% 正是 AGENTS.md §6 要拦的事
- [x] **量出 backbone 的真实瓶颈**（§6.11.5，本次最大的产出）：
      **CPU 提交 60.00 ms / 墙钟 65.77 ms = 91.2%**；单算子 CPU 成本
      `bf16_nn` 21.14 µs、`cutlass_int8_rowwise_bf16out` 10.88、
      `quantize_int8_rowwise` 9.76、`rms_norm` 9.41
      ⇒ **推翻了"INT8 发射更多所以更慢"的猜测**（INT8 臂 CPU 反而便宜 0.13 ms），
      也修正了 §6.6.3 把 30 ms 差归给"GPU 侧 torch shim"的说法（**量级来自 CPU dispatch**）。
      附带记了**三个量测陷阱**：`_run_kernel_backbone` 末尾自带 sync、
      n>1 连发会打满 launch 队列、捕获必须显式传 capture stream

- [x] **DiT 的 K/V 审计 + k/v 豁免成为出厂默认档**（§6.14，使用方指定的经验教训）：
      cross-attention 的 **KV 缓存从未被量化**（fp32 数学 + bf16 存储，
      且 cross block 里没有 K/V GEMM）；被量化的是 **32 个每步 self-attn K/V 投影**。
      新增 `dit_bf16_families`（类常量 `_DIT_BF16_FAMILIES=("k","v")`，`None` 哨兵）。
      三臂实测：f300 decoded **0.3636° → 0.1940°**，贴上 bf16 自己的 0.1868°
      （差距 1.95× → **1.04×**）；代价 DiT ×4 **+1.00 ms**、边界 **3.44× → 3.37×**、
      DiT 档比值 **1.398× → 1.329×**（门 1.15×）。
      🔴 顺带纠正一处归因：**frame 0 的 0.4935° 三臂逐位同值 ⇒ 是 bf16 backbone 的地板，
      不是 INT8 的代价**（此前 §0.1/§6.9 记错了）
- [x] **第二个独立数据集复验**（§5.6）：`green_to_blue_block_sim`（仿真 SO101，
      h264，**Se=148**）。加载器加通用的 `_resolve_column`（`.joint` 列名，不做模糊匹配）；
      10 帧参考 + 10 aux；PyAV vs ffmpeg **10 次解码逐位相同**；
      **同一套 17 个门全过**，同一边界 **106.89 ms → 3.36×**、配对 DiT **1.336×**
      （vs Se=141 的 106.91 / 1.329×，差 **<0.5%**）⇒ 通路对 Se 与数据集都不敏感。
      顺带修掉 `gen_reference.py` 硬编码 provenance 导致仿真 fixture 自称真机 AV1 的缺陷
- [x] **每观测入口 `infer(aux=...)`（连续推理，杠杆 #13）**（§6.15）：
      一个前端服务整条观测流。七道门 C1–C7 全过 —— 8 个连续帧逐帧 decoded cos
      **0.999999232–0.999999628** / max&#124;d&#124; **0.194–0.387°**；repeat 与
      "连续 == 一次性"均**逐位相同**；DiT 图**捕获 1 次 / 复用 8 次**（对象与槽
      `data_ptr` 身份集合各 1 个）；契约篡改 **4/4 被拒**。
      每观测 **117.56 ms → 3.29×**（vs 完整 `get_action`）/ **3.05×**（vs 同一边界）。
      顺带诊断并落地 fp32 权重缓存：cross-KV 刷新 **13.10 → 8.52 ms，逐位相同**。
      契约部分进仓为 CPU 测试（dispatch **21 → 51**）。
      🔴 **方法论产出**：decoded action 的 cos **门不住** stranded KV
      （打断刷新后仍 0.9999917 > 0.999），必须门在槽内容或 velocity 上（§6.15.3）
- [x] **连续推理的七道门进仓**（Phase 日志 §12）：从 `/tmp/continuous_infer_check.py`
      提升为 `tests/test_orin_groot_n17_continuous.py`（**7 passed**，22.6 s），
      负控制 C6 因此坐进验收集（AGENTS.md §3.6）。计数：真机档 **68 passed**、
      仿真档 **75 passed**（§6.16 的对齐审计后各 +5 ⇒ **73 / 80**）。提升时量到并修掉一个**测试自身**的顺序缺陷
      （module-scoped fixture 携带可变观测状态 ⇒ C4 拿 RUNUP 的 state 比
      NEIGHBOUR 的 context，报 3.1e-02 = C2 的普通帧间差）；C4 改为自装前置条件
      并加强成两臂，**前端一行未改**
- [x] **与 Thor / RTX 的功能对齐审计**（§6.16，逐符号核实）：继承链
      Orin→RtxFP16→Rtx→Thor，公共 API 全部可见。落地一处修复 ——
      继承来的 `calibrate` 无条件读 `_vit_alpha_*`（本前端 grep 计数 **0**，
      实测抛裸 `AttributeError`），现覆写为 `NotImplementedError` 并写清理由
      （`api.py` 的统一门明文要求这个异常类型）；`precision_spec` 继续返回 `None`，
      现在是诚实的。dispatch **51 → 56**（含一条反向绊线：哪天有了 FP8 alpha
      就要求撤掉这个拒绝）。核实到行的跨平台缺陷与纠正的错误记录见 §6.16

**下一步（按 ROI）**

- [x] **🔝 backbone CUDA-graph 捕获**（**杠杆 #11，已交付，§6.17**）：
      `run_backbone_graph` 与 Thor FP8 / RTX FP8 mixin **同名同契约**。
      配对交替 A/B（同一前端同一次捕获，n=**56**/臂，中位数，锁频，仿真档 Se=148）：
      backbone **63.52 → 59.52 ms（−4.00，1.0673×）**、
      每观测 `infer(aux=)` **115.44 → 111.40 ms（−4.03，1.0362×）**；
      `backbone_features` 与解码动作**逐位相同**（max|d| 均为 0）。
      一次性 **259.7 ms ⇒ 64.9 帧回本**。默认开，只作用于 `infer(aux=...)`，
      关掉用 `use_backbone_graph=False`；一次性路径永不捕获 ⇒ 口径 A/B **不受影响**。
      ⚠️ **本节原计划有一处是错的，落地时纠正**：原写"`_kbb_load_inputs` 的 `copy_`
      要在捕获区内"——**不行**。载入的**源**是每观测新的 aux 张量，
      把 `copy_` 捕进图会把**第一帧的源指针**烘进去，此后 replay 永远读那一帧
      （正是 §6.15.3 那个 stranded 失效模式）。正解是载入在**捕获区外**、
      replay 之前，只让**目标槽**常驻——这也正是 RTX FP8 mixin 的做法。
      其余三条原计划都对并照做了：显式传 `stream=s.cuda_stream`（陷阱 3）、
      `capture` 快照与 `synchronize()` 留在捕获区外、侧流 warmup 3 次。
      门：C8（`_kbb_graph` 存在 + 捕获**恰好 1 次** + 图对象与 `vlsa_h` 指针身份不变）、
      eager 臂可达性、校验不重复且发生在 `replay()` 之前；
      C1–C7 **全部重跑通过**（continuous 7 → **9**，dispatch 56 → **58**）。
      那条"反向绊线"测试 `test_the_backbone_graph_entry_point_is_a_known_gap`
      **按设计触发并被替换**成正向契约测试。

- [x] **cross-KV 投影改走 bf16 tensor core**（§6.19，三档门通过）：
      `gemm.bf16_nn` + `fvk.add_bias_bf16`，直接吃已是 bf16 的 `_dit_k_w`，
      **403 MB 的 fp32 权重缓存随之删除**。三档门（6 帧 / 两个数据集）：
      存量 cross-K cos **0.999995672**、velocity cos **0.9994165**、
      decoded cos **0.999999073** —— 全过，余量 4 个数量级以上。
      **诚实收益：每观测 111.17 → 108.55 ms（−2.62 ms，1.0241×）**、
      refresh 8.50 → 4.75 ms、口径 C **3.47× → 3.56×**。
      ⚠️ **隔离微基准把这个收益高估了 2.3×**（先报 5.22、再报 6.15 ms），
      因为 refresh 里还有 ~3 ms 两臂相同的非 GEMM 开销，且进流水线后部分重叠
      —— 原则 #16 的又一次实例，详见 §6.19.2。
      ⚠️ **代价（使用方可否决）**：6 帧里 4 帧 decoded 误差 +0.034…+0.137°，
      跨帧最坏值不变（0.4935°，是 bf16 backbone 的地板）。
      **KV 存储精度未变**（仍 bf16，与 HF eager 一致），无 INT8、无 scale/校准。
      回退方式：绑回 `_project_dit_cross_kv_fp32` 并恢复权重缓存（参考臂仍在仓里）。
      dispatch **58 → 63**（含 `bf16_nn` 的 **(M,N,K)** 参数顺序钉子）
      ⚠️ **上一条"代价"是 n=6 的观测，§6.20 用 593 连续观测重测后已撤回该定性**
- [x] **长任务精度对比 + 两个真实缺陷修复**（§6.20、§6.21，使用方要求）：
      `tests/_helpers/groot_orin/long_horizon_ab.py`（**进仓**），episode 0 全 **593 帧连续观测**
      （`set_prompt` 一次 + 593 次 `infer(aux=...)`，距 prompt 0..592，19.7 s 操作），
      流式不落盘 6.8 GB fixture，hooks 复用 `capture_aux.py`。
      **harness 先与 fixture 版三档门对表吻合**才用于长时程（原则 #1）。
      **结论：bf16 臂与 fp32 臂在门内不可区分** —— 三档门 **0/593 违反**、
      decoded cos 最低 **0.999998845**、两臂均值差 **+0.0012°**、**36.1% 的帧逐值相等**、
      全程最坏 bf16 **0.4935°** 优于 fp32 **0.5808°**、相对误差 mean **0.186%** / max 0.413%。
      **排除状态漂移**：A−B 对 prompt 距离 r=**+0.0076**、斜率 +3.3e-06 °/帧；
      而 fp32 参考臂自己的距离趋势（r=+0.2641）**更强**，且相对误差与距离**负**相关（−0.3198）
      ⇒ 那点趋势是内容固有的。⇒ **§6.19.3 的"取舍"定性撤回**。
      **教训**：精度 delta 同样受原则 #16 约束，且样本量要求比延迟更高（延迟可重复取中位数降噪，
      精度每帧只有一次采样，n=6 的置信区间宽到能改变结论符号）。
      顺带修掉两个缺陷（**都有负控**）：
      **(a) inference tensor 打在每观测通路上** —— `getattr(t,"_version",None)` 不会兜底，
      因为 `_version` 是**会抛异常的 property**；`inference_mode` 正是官方 `get_action` 的推荐跑法
      ⇒ 活体集成第一帧就炸，而此前全部 A/B 走磁盘 fixture 所以是**共同盲区**。
      更深的洞是备忘会缓存 `None` 版本而 `None == None` 成立 ⇒ **原地改写被静默放行**。
      修法：版本不可读 ⇒ 绝不走 identity 快路径 + 一次性 warning。dispatch **63 → 67**。
      **(b) 视频读取器丢掉 episode 尾巴** —— 593 帧读到 **581** 抛 `EOFError`；
      数据无问题（视频/parquet/episodes.jsonl 都是 593）。根因是 `_decode` 每次重建
      `container.decode(stream)` 并在循环内 `return`，**弃置活生成器** ⇒ PyAV finalize 时 flush 编解码器。
      20 行单变量探针定位（弃置式 581 失败 / 持有式 593 通过）。修法：跨调用持有**挂起**的迭代器，
      seek 前置 `None`。新增 `tests/test_lerobot_video.py`（**5 条，CPU-only，0.78 s**，自带合成视频）。
      闸门：仿真档 **105 → 114 passed**、真机档 **80 → 89 passed**。
- [x] **🔝 每观测图像通路 → 纯 torch GPU 链**（**杠杆 #14，已交付，§6.23**）：
      本仓 **7 个** N1.7 CUDA 前端（Thor、Thor FP8、RTX、RTX FP16/FP8、SM89、AMD、Orin）
      **没有一个**跑图像通路，全都要求调用方交已处理好的 `aux` 束 ⇒ 厂商
      `Gr00tN1d7Processor` 每观测跑一遍**纯标量 host 代码**。唯一替换掉它的是
      Ascend NPU 通路（按使用方要求不看）⇒ 这是**整条 CUDA 家族共同的空白**，
      而 Orin 是它代价最高的地方。按使用方指定走 **N1.6/Pi0.5 的纯 torch 房型**。
      **实测（三臂配对交替，中位数 of 11，4 帧真机数据 × 两个数据集，锁频）**：
      预处理 **12.916–14.027 → 1.448–1.485 ms**；
      每观测 **112.374–115.499 → 99.579–100.428 ms（省 11.945–15.410，1.1189–1.1540×）**。
      **vendor 臂自证**：其 `pixel_values` 与 fixture **bf16 逐位相同（4/4 帧）**。
      **measured vs predicted**：预测 ~8.6 ms ⇒ **偏低 1.4–1.8×**，
      因为那个账面值是**分段中位数相加**（教训 3）。
      **新增口径 D**（每观测含图像预处理）：真机 **3.439/3.444 → 3.896/3.883×**、
      仿真 **3.344/3.437 → 3.859/3.846×**；⚠️ 同时**下修口径 C**（3.97× 偏乐观，
      分子不含预处理而分母含 ⇒ 方法论纠正 5）。
      **精度**：12 帧 × 3 臂，cv2 臂与 HF **九位小数全同**，T3/T4 **未放宽门**，
      只有 T1 新增 `THR_IMAGE_GPU_BACKBONE = 0.985`；T3 变化 **−0.000087…+0.000153**
      （有时更好）、T4 max 误差 **4 好 / 4 差 / 4 等** ⇒ 对称、无系统性退化
      （§6.20 的 n=4 教训在此**执行**为 n=12）。负控 T1 **0.9918 → 0.8989**。
      **精确性**：缩小步**逐位精确且可证**（`s = p/q` 最简分数、`p` 奇数 ⇒ 无平局；
      640→256 是 5/2，`out = (a+2b+2c)/5`），**3350 万次穷举 0 分歧**；
      放大步 **max = 1 LSB**（fp64 下相同 ⇒ **算法性**），4 种表述全部收敛 ⇒ **声明未知**；
      patchify 在 bf16 下 `torch.equal`（0/786432）。
      🔴 **推翻了自己先前的一个归因**：残差**不在** OpenCV 的权重推导里
      （one-hot 探针证明其有效权重就是连续覆盖，吻合到 7.2e-8），
      而在 **uint8 累加/舍入路径**上。
      **未写任何新 kernel**（红线 #8）；`fvk.patch_im2col_uint8` 存在但**几何被锁死**（见下）。
      测试 precision **17 → 29**、新增 `test_orin_groot_n17_preprocess.py` **65 条 CPU-only**，
      真机档 **102 → 179**、仿真档 **127 → 204**。
- [ ] 🔴 **RTX FP8 / SM89 / Thor FP8 的 backbone-graph 契约有同一处 inference-tensor 洞**（§6.21.1）：
      `groot_n17_rtx_fp8.py:211` 与 `:244` 是同一个 `getattr(..., "_version", None)` 写法，
      而 `_version` 是个**会抛异常的 property** ⇒ 活体集成（HF 在 `inference_mode()` 下产出 aux）
      第一帧就 `RuntimeError`；并且备忘会把 `None` 版本缓存下来，`None == None` 成立
      ⇒ **原地改写的元数据被静默放行**。Orin 已修（`_mutation_version` + `version is not None` 守卫 + 4 条钉子）。
      改 mixin 会同时动 RTX FP8 / SM89 / Thor FP8 / AMD 四个通路，按红线 #1 只报不改，
      需要 owner 判定。属既有代码，不是本次改动引入。
- [ ] 🔴 **RTX FP8 通路的 `infer(aux=...)` 至今不失效 cross-KV**（§6.13.2 发现，未修）：
      它刷新了 `_backbone_features`，却没有任何地方失效 `_dit_cross_K/_V`
      （`infer` 用 `hasattr` 缓存），而 `_precompute_dit_cross_kv` 是从
      `_backbone_features` 算出来的 ⇒ **action head 会继续用上一帧的 cross-KV**。
      Orin 已在 §6.15 用原地刷新修好并加了负控制；
      `groot_n17_rtx_fp8.py` / `groot_n17_rtx_sm89.py` / `amd/frontends/torch/groot_n17.py`
      三处共用那个 mixin，**都还带着这个 bug**。
      ⚠️ 按 §6.15.3，验收它的门必须打在**槽内容或 velocity**上——
      decoded action 的 cos 测不出来。属本仓库既有代码，不是本次改动引入的。
- [x] **kernel 化 ViT/LLM 的 torch shim（杠杆 #9，已交付，§6.22）**：
      原判 `blocked_on_kernel` 是**空判** —— `rope_neox_qk_bf16` 早已编好，只是在
      **另一个扩展** `flash_rt_qwen3_vl_kernels`（`csrc/qwen3_vl_bindings.cpp`）里，
      只枚举 `flash_rt_kernels` 必然漏（§6.22.3）。归因探针重测得 RoPE shim 实为
      **11.902 ms GPU**（backbone 的 19.5%），比记录的 2.7 ms **偏低 4.4×**。
      接入后与 shim **逐位相同**（3 个真实形状、两臂到 fp64 等距）⇒ 精度门数值一个没动；
      80 次 shim → **40 次 launch**；每观测 **107.752 → 97.281 ms（1.1076×）**。
      ❌ **gate·up 的 `mul_` 一并否决**：`silu_mul_qwen36_bf16` 逐位相同但只省
      **0.092 ms/backbone（0.08%）**，低于噪声地板，不值得动一条已出厂的数值路径（原则 #13）。
      ⇒ **torch elementwise shim 现在只剩 `mul` 这一处，且已判定不动**。
      💡 原记的那条线索（`qkv_split_norm_rope_bf16` / `qwen3_k_norm_rope_kvwrite_bf16`）
      **没有被采用**：那两个是 q/k-norm+rope+kvwrite 的三合一，会改变 Qwen3 的 norm 语义；
      实际用的是纯 rotate-half 的 `rope_neox_qk_bf16`，与 shim 数学**逐位等价**，
      因此不需要核实 `freqs_re/freqs_im` 排布（那是"做错不报错只掉 cos"的那类风险，本路径没有）
- [ ] **`bf16_matmul_cublaslt_bf16` 替换 backbone 的 `bf16_nn`**（§6.22.7，已量未落地）：
      在 256 个真实形状 backbone GEMM 上实测 **1.1778×** ⇒ **5.766 ms/backbone（~5.9%）**。
      **两条未清的风险，落地前必须各自过门**：
      **(a) 不逐位相同** —— 7 个形状全部有差异（cuBLASLt 的 split-K / 算法选择不同），
      所以这是**数值改动**而非纯 launch 优化，必须走完整三档精度门（§6.19.1 那套），
      不能像杠杆 #9 那样只用 `torch.equal`；
      **(b) 图可捕获性未验证** —— 原则 #14 的前车之鉴是 cuBLASLt 的 **FP8** GEMM
      在 stream capture 下返回 `code=13` 而**静默**封顶整条路径；
      bf16 路径大概率没这个问题，但**必须实测 capture + replay**，不能推定。
      ⚠️ **ROI 排序已因杠杆 #14 翻转**（§6.23）：本项原先排在 HF pre/post 之后，
      理由是后者"~27.55 ms 是真实部署口径里最大的单块"。图像通路替换之后，
      每观测残留的 pre/post 只剩 **GPU 图像链 1.448–1.485 ms + `decode_action` 0.24 ms
      ≈ 1.7–1.8 ms**（`_apply_vlm_processing`+tokenizer 那 2.74 ms 是 prompt-scoped，
      §6.15 已提进 `set_prompt`）⇒ **本项的 5.766 ms 现在是更大的那块**，
      排序上应高于剩余 pre/post，但仍低于 `fvk.add_bias_bf16` 的 **8.880 ms**
      （那条要等 kernel owner 的 epilogue，见下）
- [ ] **QuaRot/Hadamard 旋转**（杠杆 #6）：**挂起**。三次真激活门后适用范围只剩 ViT，
      而 ViT 已按"尽量不用旋转"关闭。仅当将来确实要吃 ViT 那 41.2% 的 backbone
      占比时才重启，且按原则 #13 先微基准（旋转要融进 norm kernel，
      radix-16 寄存器 FHT，才近乎免费）
- [ ] **`use_fp16=True` 的报错信息没提 sm87**（⚠️ 本项此前记错了，已按源码核实纠正）：
      原记"公共 API 路由不到 Orin N1.7"—— **错**。`api.py:875` 的
      `pipe_cls = resolve_pipeline_class(config, framework, arch)` 就是走
      `_PIPELINE_MAP` 的，实测 `resolve_pipeline_class("groot_n17","torch",
      "rtx_sm87")` → **`GrootN17TorchFrontendOrin`**，默认路径
      （`use_fp8=False, use_fp16=False`）**能路由到**。
      真正成立的只有一条窄得多的事实：`api.py:831` 那个
      **`use_fp16=True` 的实验性白名单**里没有本三元组（只有
      thor/rtx_sm120/rtx_sm89/amd_cdna3/amd_cdna4），所以在 Orin 上传
      `use_fp16=True` 会拿到一条**没提到 sm87** 的 `ValueError`。
      另核实：`use_fp8=True` + sm87 会正常路由到 Orin，并在 `__init__` 里
      **任何 CUDA 工作之前**被 `RuntimeError` 拒绝（`_require_arch` 与
      `super().__init__` 都在拒绝之后），行为正确。
      ⇒ 剩下的只是**报错信息质量**（是否把 sm87 写进那条提示，或干脆说明
      Orin 的 bf16+INT8 档不接受 `use_fp16`），不是功能缺口
- [ ] **摘掉融合残留的 HF 依赖**（§6.7 末尾；⚠️ **三项里最重的一项已由杠杆 #14 摘掉，§6.23**）：
      ~~`pixel_values`（HF processor 的 resize/normalize/im2col，最重）~~
      ✅ **已替换**：`infer(frames=...)` 走 `_groot_n17_preprocess.py` 的纯 torch GPU 链，
      **12.916–14.027 → 1.448–1.485 ms**，无 HF/albumentations/PIL。
      **仍未摘的两项**（都是 prompt-scoped，`set_prompt` 已提走 ⇒ **没有延迟收益**，
      只有部署独立性，原则 #10）：
      `visual_pos_masks`（≡ `input_ids == image_token_index`，**已验证** 4/4 fixture，最容易）、
      `rope_cos`/`rope_sin`（**被阻塞**，见下一条）
- [ ] 🔴 **Stage 2（HF-free `set_prompt(str)`）被 `mrope_table.py` 阻塞**（§6.23.7，本次实测发现）：
      `flash_rt/models/groot_n17/mrope_table.py` **既是休眠的、又是错的** ——
      对着 4/4 fixture 捕获的 `rope_cos`/`rope_sin`，maxdiff **2.2e-02 … 3.1e-02**
      （bf16，约 **5–10 ULP**）。而 `tests/_helpers/groot_n17/mrope_ref.py` 实现的是
      **同一个** `apply_interleaved_mrope`（`slice(offset, mrope_section[axis]*3, 3)`），
      所以它那句"对 HF 验证过逐位精确"是在**旋转后的 Q/K** 上做的，
      **表本身从来没被验证过**。
      ⚠️ 按红线 #1 本次**未改**它（它是既有代码、且被别的通路共用与否需先 grep 确认）。
      **已定位的诊断路径**：`mrope_section=[24,20,20]` 铺在 `head_dim/2 = 64` 个槽上时，
      T 写 `slice(0,72,3)`（**22** 槽）、H 写 `slice(1,60,3)`（20）、W 写 `slice(2,60,3)`（20）
      ⇒ **64 个里只覆盖 62 个**，槽 60/61/62 留在初始 `clone()` 的 T 轴上。
      **逐列**比对捕获到的表即可判定差异是不是这三个槽（或轴分配）。
      另需 `grid_for(H,W)` 产 `image_grid_thw`，以及每 prompt 一次 tokenizer 调用
      （模板 + 确定性的 `<|fim_prefix|>` → 64 份展开）
- [ ] **N1.6 通路**：侦察已闭环（fixture + aux + `docs/groot_n16_orin_sm87.md`，
      含 post-norm 反向差异、无 ViT 截断杠杆、324 token/视角、三个 fp16-only 冷路径
      kernel 等实测结论）；**待做**：`attn_backend_groot_orin.py` +
      `models/groot/pipeline_orin.py` + `frontends/torch/groot_orin.py` +
      `("groot","torch","rtx_sm87")` 双注册 + 两个测试文件
- [ ] 清理冗余权重内存（§6.6.6 第 3 条 + §6.9.6）：spec 里的融合 `_llm_qkv_w`/
      `_vit_qkv_w`（~420 MB）本通路不用；INT8 档另留 2.18 GB bf16 DiT 权重
      （为配对 A/B 故意保留，若定档 INT8 可释放，**但 cross 层 k/v 必须留** ——
      `_precompute_dit_cross_kv` 在 torch fp32 里用）
- [ ] 1 相机口径复测，才能与厂商 216.5 ms 直接比（§3.1）
- [ ] 把 `_groot_n17_fusion.fast_pos_embed_interpolate` 折回 Thor FP8 通路
      （消除重复实现；针对已上线通路的独立重构，需另过 Thor 的门）
- [ ] 向 kernel owner 报缺口（红线 #8，只报不做）：
      🔴 **`bf16_nn_bias` / `bf16_nn_bias_gelu` 的 cuBLASLt epilogue 在 SM87 上
      返回 `code=15`（NOT_SUPPORTED）**。§6.22.6 **重验并扩大了 M 的取值**：
      M=**20 / 128 / 141 / 512** 四个都失败 ⇒ 与 M 的 16 对齐**无关**，
      不是 `pipeline_orin.py:520` 归因的"M 未 16 对齐"，而是**该 arch 上没有这个 epilogue**。
      请 owner 判定是 build 未实例化还是 cuBLASLt 本身不支持，并**顺手纠正
      `pipeline_orin.py` 那条归因注释**（本次按红线 #1 未改它）。
      📈 **缺口现已量化**：`fvk.add_bias_bf16` 是 GPU 普查里**最大的非 GEMM 项**，
      **8.880 ms / 848 次调用**（§6.22.1）—— 有 epilogue 就能整块省掉。
      - `gelu_erf_fp16`（用于修 RTX FP16 通路的 merger GELU）
      - M=41 的 INT4 GEMM（DiT 再减半字节；仓库只有 M=1 decode 专用 int4 gemv）
      - **SM87 int8 输出的 ada-`LayerNorm`（减均值 + scale/shift 调制）**、
        int8 输出的 `layer_norm_no_affine`、int8 输出的 `gelu_tanh`
        ⇒ 三者各能省掉 DiT 的一个独立量化 pass（§6.9.5）
      - ✅ **已撤销的两条**（不要按旧版报）：bf16 `rope_rotate_half` **已由
        `flash_rt_qwen3_vl_kernels.rope_neox_qk_bf16` 交付**（§6.22）；
        bf16 elementwise `mul` **已量化为 0.092 ms 并主动否决**（§6.22.6），不是缺口
      - ⚙️ **审计规约（本次空判的教训，§6.22.3）**：本仓 SM87 构建有**三个**扩展模块
        （`flash_rt_kernels` / `flash_rt_fa2` / `flash_rt_qwen3_vl_kernels`），
        最后一个由 `csrc/qwen3_vl_bindings.cpp` **单独绑定**。
        任何"kernel 不存在"的判定都必须枚举全部三个，并写明查过哪些模块
      - ⚪ **`fvk.patch_im2col_uint8` 的几何被 SigLIP 锁死**（§6.23.4 天花板检查，
        **不是本通路的阻塞项**）：`csrc/kernels/patch_embed.cu:64` 硬编码
        `total = nv * 256 * 588`（SigLIP 14×14×3）并经 LUT 输出 **fp16**。
        Qwen2-VL 家族要的是 **1536 = 3×2×16×16** 列、merge-block 行序、**bf16**
        ⇒ 三处都不合。torch 版 patchify **0.44 ms** 已经够快，所以**没有 ROI**；
        登记在此只是提醒：若将来要做通用的 uint8→patch-rows，**行序与列宽必须由调用方传**，
        不能再烘进 kernel。同类：任何"im2col"命名的 kernel 都要先核实它的
        patch/merge/temporal 三个因子
- [ ] ⚙️ **kernel 普查的分桶规约**（§6.22.1 / §6.22.8，本轮自查出的第二次同类错误）：
      按关键字给 C++ mangled 模板名分桶会**同时朝两个方向错**。已实测的两个撞车：
      **(a)** `fa2_vendor::flash_fwd_kernel<Flash_fwd_kernel_traits<…, cutlass::arch::Sm80, …>>`
      含 `cutlass`/`sm80`，被**排在 attention 之前**的 GEMM 规则吃掉 ⇒ FA2 记成
      0.312 ms / 0.3%，实为 **2.335 ms / 2.28%**（偏低 7.5×），GEMM 同时虚高到 73.3%（真值 64.1%）；
      **(b)** attention 规则里的 `"flash" in name` **匹配命名空间 `flash_rt::kernels::`**
      ⇒ 那 0.312 ms 根本不是注意力，是漏网的 FlashRT 自有 kernel。
      此前已因 `DefaultGemmWi…` 被归进 copy/cast 纠正过一次。
      **规则：分桶按精确名字前缀、attention 排在 GEMM 之前、并打印每个桶的成员名单而不只是总量**
      （两个错都是"总量看着合理"才活下来的）；
      更正后要能**自证**——本例的自证是"GEMM 在两臂之间只差 0.12 ms，而 rope 改动不该碰 GEMM"。
      用 `/tmp/n17_census_corrected.py`，不要用 `/tmp/n17_kernel_census.py`
- [x] ~~`gate_residual_ada_norm_int8` 把量化融进 ada_norm~~ —— **撤回，该 kernel 不适用**：
      读 `csrc/kernels/fusion.cu:157-225`，它算 `rsqrt(mean(r*r)+eps)` 是 **RMS**，
      且残差是带门的 `residual + x*gate`；GR00T DiT 用的是**减均值** `AdaLayerNorm`
      (eps=1e-5) + 无门 `h += o_out`。接上去会**静默改变数学**（§6.9.5）
