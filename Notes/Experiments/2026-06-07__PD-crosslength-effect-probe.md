# PD 跨长度(cross-length) effect 收敛 probe

**日期** 2026-06-07
**脚本** `Code/scripts/analyze_pd_crosslength_effect_probe.py`
**产物** `Code/runs/analyze_pd_crosslength_effect_probe/summary.json`
**问题** PD 对的两个药是否经常在**不同 hop 深度**($\ell_A\neq\ell_B$)收敛到同一 effect 节点？若是，则"跨长度/RoPE-相对深度"注意力有结构依据；否则不值得建。

所有数字来自本次运行，未凭记忆。

## 设置
- merged KG，PD/PK/NEG 各 100 对（PD/PK 采自 `ddi_edges.csv` 关键词分桶；NEG 采自 seed42 `test_s2` 负样本），seed 42，L=3，undirected BFS。
- 共享 mediator 限定为 effect 类；两版定义：`effect`(Side Effect/effect-phenotype/Symptom) 与 `effect_plus`(+biological_process+disease)。
- on-diagonal=$\ell_A=\ell_B$；off-diagonal=$\ell_A\neq\ell_B$。
- **OFF-ONLY gap** = 有共享 effect 但**只在 off-diagonal**（同深度一个都没有）的对占比 = 跨长度的**净增益**。

## 结果（verified）

| effect 定义 | bucket | cov_any | cov_same_depth | OFF-ONLY gap | off_mass% |
|---|---|---|---|---|---|
| effect | PD | 97% | 97% | **0.0%** | 10.9% |
| effect | PK | 94% | 94% | 0.0% | 10.0% |
| effect | NEG | 93% | 93% | 0.0% | 11.1% |
| effect_plus | PD | 97% | 97% | **0.0%** | 15.8% |
| effect_plus | PK | 94% | 94% | 0.0% | 15.5% |
| effect_plus | NEG | 93% | 93% | 0.0% | 17.6% |

(dA,dB) 网格被 (3,3) 对角 hub 质量主导（PD effect_plus 的 (3,3)=17417/对）。

## 结论（负结果）
1. **OFF-ONLY gap = 0%（全 bucket / 全定义）**：没有任何一对"只能靠跨深度"才碰到同一 effect；每个有共享 effect 的对在同深度也一定有。`cov_any == cov_same_depth`。
2. 跨深度 mediator 占质量 11–16%，但**冗余**（所属对已被同深度覆盖）。
3. **PD/PK/NEG 几乎一致**，跨长度**非 PD 特异**。

→ **"跨长度 / RoPE 相对深度"注意力在 KG 结构上不成立，不值得作为架构重头戏。** PD 的瓶颈不是"effect 够不着不同 hop"（同深度 97% 覆盖），而是**共享 effect 是弥散 hub、不判别**。方向回到 **effect 身份/特异性加权（node-semantics 线）**。

## 诚实边界
- 测的是"跨深度收敛是否**存在/独有**"的结构事实；**不能**回答"RoPE 相对深度项比 depth-blind 注意力多买多少"（那是模型消融）。
- 最短路深度、单 seed、n=100、undirected。
- off-diagonal mediator 是否比 on-diagonal 更**特异**未测；但 (3,3) hub 主导，估计也是 hub-ish。

## 复现
```bash
python Code/scripts/analyze_pd_crosslength_effect_probe.py --n 100 --seed 42 --L 3
```
