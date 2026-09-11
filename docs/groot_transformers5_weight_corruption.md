# GR00T N1.6/N1.7: transformers>=5 静默权重损坏与修复

## 状态

已修复（2026-08-08）。影响所有通过 `AutoModel.from_pretrained` 加载 Gr00tN1d6 / Gr00tN1d7
的 transformers **5.x** 环境；transformers 4.51.3（训练同版本）不受影响。

## 症状

- 服务侧（`serving/groot_n16/eager_server.py`，HF eager 服务端）表现为"不使用实时图像"：
  换图 / 换 prompt 时动作输出只变化 ~0.05，策略几乎只跟随 state。
- 加载日志完全正常：`Loading weights: 100% 1106/1106`，missing / unexpected keys 均为 0。
- checkpoint 本身完好（其他设备、transformers 4.51.3 下推理正常）。

## 根因

**加载本身是正确的，损坏发生在 `from_pretrained` 的收尾阶段。**

1. `from_pretrained` 把全部 1106 个张量正确写入模型（加载结束瞬间逐张量与磁盘
   safetensors 比对，maxΔ = 0）。
2. 随后 `_finalize_model_loading` → `_initialize_missing_keys` → `initialize_weights()`
   遍历所有模块调用各自的 `_init_weights`。跳过已加载参数的唯一依据是参数上的
   `_is_hf_initialized` 标记，而 `PreTrainedModel._initialize_weights` 只在
   `is_remote_code() == True` 时才检查该标记：

   ```python
   # transformers/modeling_utils.py (5.10)
   if getattr(module, "_is_hf_initialized", False):
       return
   if is_remote_code and all(getattr(p, "_is_hf_initialized", False) for p in module.parameters(recurse=False)) ...:
       return
   self._init_weights(module)   # <-- 重新随机初始化
   ```

3. Gr00tN1d6 / Gr00tN1d7 虽然经 `trust_remote_code` 动态加载，但类的 `_auto_class`
   属性为空 → `is_remote_code()` 返回 `False` → 参数级标记被无视 →
   Siglip2 视觉塔的 `_init_weights` 对已加载的 `nn.Linear` / `nn.Embedding`
   执行重新随机初始化。

4. 受损范围：整个 Siglip2 视觉塔 282 个张量（`vision_model.vision_model.*`，
   含 `position_embedding`、全部 encoder layers、head）+ 投影层 `mlp1.1/3`。
   Qwen3 LLM 与 action head 因各自 `_init_weights` 的行为未被波及。

5. 后果：视觉特征是随机噪声 → DiT 交叉注意拿不到有效图像信息 → 策略退化为
   纯 state 条件模型。

实测特征（可用于快速诊断）：受损张量的 live 值 std ≈ 0.0294（随机初始化分布），
与磁盘值的余弦相似度 ≈ 0；例如
`backbone.model.vision_model.vision_model.encoder.layers.0.self_attn.q_proj.weight`
磁盘 std 0.02049，加载后变为 0.02936。

## 修复

一行核心改动，在 `Gr00tN1d6.__init__` / `Gr00tN1d7.__init__` 的 `post_init()` 之前：

```python
# transformers>=5: 必须标记为 auto/remote-code 类，否则加载收尾阶段的
# `_initialize_missing_keys` 会无视 `_is_hf_initialized` 标记，
# 用子模型的 `_init_weights` 重新随机化已加载的权重。
type(self)._auto_class = "AutoModel"
```

涉及文件：

- `<checkpoint>/gr00t/model/gr00t_n1d6/gr00t_n1d6.py`
- `<checkpoint>/gr00t/model/gr00t_n1d7/gr00t_n1d7.py`

## 预防措施

权重完整性自检收敛为共享函数
`serving/groot_n17/aux_builder.py:verify_weight_integrity()`:对 checkpoint
safetensors 均匀抽样 ~12 个张量与 live 参数数值比对(容差取 bf16 舍入量级),
不一致则抛错、拒绝启动。两条服务链路都在构建 `Gr00tPolicy` 后立即执行:

- FlashRT 服务:`Gr00tN17AuxBuilder.__init__`(HF 仅作预处理器,权重同样经
  transformers 5 加载,必须检查);
- HF 基线服务:`serving/groot_n16/eager_server.py` /
  `serving/groot_n17/eager_server.py`(后者已复用共享函数)。

日志标记 `[weight-check]`。

## 验证数据（2026-08-08）

| 指标（HF eager 服务端） | 修复前 | 修复后 |
| --- | --- | --- |
| 权重与磁盘不一致张量 | 282（整个视觉塔） | 0 / 1106 |
| 相同请求重复噪声底 | 0.65 ~ 0.71 | 0.011 ~ 0.016 |
| 黑图 vs 真实图 Δ（mean） | 被噪声淹没 | 0.135 |
| 随机图 vs 真实图 Δ | — | 0.137 |
| prompt 切换 Δ | — | 0.036 |
| state +0.3 Δ | — | 0.307 |

离线 `Gr00tPolicy` 同条件复现一致（seeded 噪声底 0，黑图 Δ ≈ 0.13）。

N1.7（5558）权重 1030/1030 核对一致；其图像敏感度偏弱（反色图 Δ ≈ 0.03 <
噪声底 0.07，state Δ 0.22 / prompt Δ 0.07 正常），属该 epoch5 checkpoint 自身
特性，非加载问题。

## 教训

- 跨 transformers 大版本迁移 `trust_remote_code` 模型时，
  `from_pretrained` 成功（0 missing / 0 unexpected）**不等于**权重正确，
  必须做数值级核对（抽样 live vs safetensors）。
- 复合模型（PreTrainedModel 嵌套 PreTrainedModel）在 5.x 下的收尾初始化
  依赖 `is_remote_code()`；自定义顶层模型类务必显式设置 `_auto_class`。
- 诊断此类问题的快捷手段：比较 live 权重 std 与随机初始化分布；
  对 `_load_pretrained_model` / `_initialize_missing_keys` 打桩二分定位。
