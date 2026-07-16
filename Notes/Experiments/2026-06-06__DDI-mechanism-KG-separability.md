# DDI 机制类别在合并 KG 中的可区分度分析

**日期** 2026-06-06
**脚本** `Code/scripts/analyze_ddi_mechanism_kg_separability.py`
**产物** `Code/runs/analyze_ddi_mechanism_kg_separability/`（`summary.json`、`mechanism_fingerprint.parquet`、`per_mechanism_auc.parquet`）
**动机** C-MPNN 框架的"方式1"：每种机制 = 一个 query relation，不同机制应涌现不同传播路径。问题：DrugBank 给的机制类别（`ddi_type`，215 种）在 KG 结构里到底分不分得开？

所有数字均从本次运行输出读取，未凭记忆。

---

## 0. 机制清单（verified）

- 565,731 条正样本，**215 种机制**（模板化文本，编码药代/药效机制 + 方向）
- 重尾：top-30 覆盖 **92.3%** 样本；95 种机制各 <100 条，合计仅 0.6%
- 机制文本天然分两轴：**PK**（metabolism / excretion / serum concentration，52.2%）vs **PD**（risk/severity / activities / efficacy / QTc / CNS，47.7%），other 0.1%

---

## 1. 每机制的结构指纹（mean shared-neighbour per channel）

按关系类型分通道统计共享邻居均值（节选 top 机制）：

| n | 共享靶点 | 共享酶 | 共享转运体 | 共享副作用 | 共享适应症 | 机制 |
|---|---|---|---|---|---|---|
| 99273 | 0.07 | **1.94** | 0.36 | 16.9 | 0.03 | metabolism decreased |
| 96650 | 0.02 | 0.19 | 0.19 | 14.4 | 0.03 | excretion rate decrease |
| 40216 | 0.07 | **1.38** | 0.87 | 11.7 | 0.04 | serum conc increased |
| 37967 | 0.21 | 0.32 | 0.03 | 12.0 | 0.05 | CNS depression risk |
| 26975 | 0.05 | **1.77** | 0.35 | 12.1 | 0.09 | metabolism increased |
| 11906 | 0.03 | **1.83** | 0.47 | 7.2 | 0.04 | serum conc decreased |
| 7166 | 0.03 | 0.32 | **1.71** | 15.3 | 0.11 | excretion decreased |

**读法：** 不同机制确实在**不同关系通道**上加载——代谢类机制共享酶显著偏高（1.8–1.9），排泄类机制共享转运体偏高（1.71）。这正是"不同机制走不同 path"在数据上的体现。

**PK vs PD 通道对比（sample-weighted，2026-06-06 补查更正）：**

| 通道 | PK (295k 对) | PD (270k 对) | 谁高 |
|---|---|---|---|
| 共享酶 | 1.165 | 0.362 | PK ~3.2× |
| 共享转运体 | 0.401 | 0.098 | PK ~4× |
| 共享靶点 | 0.046 | **0.186** | **PD ~4×** |
| 共享载体 | 0.100 | 0.056 | PK |

> ⚠ 更正先前措辞："PD 酶/转运体几乎为 0" 不准确。PD 酶均值 0.362（非 0），且部分 PD 机制酶共享很高：serotonin syndrome 0.81、CNS depressant(increase) 0.82、arrhythmogenic 0.72、hyperglycemia 0.72、hypoglycemia 0.65、myopathy/rhabdo trn 0.33。正确说法是 **PD 的酶/转运体比 PK 低约 1/3–1/4 但非零**；而 **PD 真正的特异通道是共享靶点（0.186，PK 的 ~4 倍）**——药效相互作用本应由"共享靶点"而非"共享代谢酶"驱动，生物学上自洽。

---

## 2. 结构 → 机制 可预测性（区分度 headline）

多项 logistic 回归，特征 = 14 个 typed shared 通道 + 共享邻居数 + 距离，top-30 机制（30 类）：

| 指标 | 值 |
|---|---|
| top-1 accuracy | **0.359**（majority 0.190，random 0.033）|
| top-5 accuracy | **0.720** |
| macro-F1 | **0.064** |

**读法：**
- top-1 0.36 是 majority 的 ~1.9 倍，top-5 0.72，说明结构**确实携带机制信息**，但只够把样本压进"几个候选机制"里。
- **macro-F1 仅 0.064 是关键**：宏平均被打到地板，意味着结构只分得开少数头部（PK）机制，绝大多数 PD/尾部机制几乎零召回。区分度高度不均衡。

---

## 3. 逐机制 one-vs-rest AUC（哪些机制分得开）

| 最可分（PK 主导）| AUC | 最难分（PD 主导）| AUC |
|---|---|---|---|
| excretion decreased | **0.933** | bleeding risk | 0.680 |
| metabolism decreased | 0.866 | hyperglycemia risk | 0.676 |
| serum conc increased | 0.804 | CNS depressant | 0.663 |
| metabolism increased | 0.800 | hypoglycemia risk | 0.636 |
| serum conc decreased | 0.797 | bradycardic | 0.633 |
| myopathy/rhabdo | 0.791 | QTc prolongation | **0.606** |

**核心结论：** 区分度沿 **PK / PD 轴系统性分裂**。
- **PK 机制（代谢/排泄/血药浓度）结构可分，AUC 0.80–0.93**，因为它们由共享酶/转运体这种稀疏但高特异的 drugbank 通道唯一决定。
- **PD 机制（出血/QTc/CNS/低血糖）结构难分，AUC 0.60–0.68**。原因（2026-06-06 更正）：不是"PD 无酶/转运体信号"（PD 酶均值 0.362、部分机制达 0.6–0.8），而是判别力最强的稀疏通道（酶/转运体）被 PK 主导；PD 主要靠共享靶点，但该信号绝对量低（均值 0.186，多数 PD 对共享 0 靶点）且在不同 PD 机制间高度雷同，故结构能识别"是 PD 交互"却分不出"是哪种 PD"。（末句为与 AUC 一致的推断，待"PD 子类间结构指纹类间距离"验证。）

---

## 4. Relational-WL 机制碰撞（WL 上界处的机制分辨力）

全合并 KG 上跑 relational 1-WL（117 关系含逆，收敛于 round 3）：

- 542,203 个不同 pair 签名 / 565,731 对 → 签名近乎单射
- 落在碰撞签名（≥2 对同签名）上的样本 = **4.89%**
- 签名机制歧义（同签名对应 >1 机制）= **3.95%**
- 碰撞签名上的主导机制纯度均值 = **0.803**

**读法：** WL 几乎把每个 pair 当成独立指纹，机制歧义只有 ~4%；但这是因为结构近乎单射，不是因为结构对齐了机制。碰撞核内主导机制纯度 0.80，说明即便结构相同，机制也有 ~20% 的剩余不确定。

---

## 5. 粗粒度 PK / PD 可路由性

二分类（PK vs PD），同一结构特征向量：

- **PK-vs-PD 结构 AUC = 0.756**（0.5 = 结构完全无法路由）

**读法：** 结构能**部分**路由 PK/PD 轴（0.76，远高于 0.5），但不干净。这直接关联 memory 里"learned routing over hard PK/PD split"——结构里**存在**软路由信号，但不足以支撑硬 PK/PD 切分；用一个学习到的归纳偏置去 exploit 这个 0.76 的软信号，比手写 PK/PD 规则更合理。

---

## 6. 对方法线的结论

1. **机制区分度极度不均衡，沿 PK/PD 轴分裂**：PK 类由共享酶/转运体唯一决定（AUC 0.80–0.93），C-MPNN 方式1 在 PK 上天然 work；PD 类缺判别性结构签名（AUC 0.60–0.68），是机制分类的真正瓶颈。
2. **PD 机制是 molecular/semantic 注入的主战场**：PD 的特异通道是共享靶点（0.186），但绝对量低、且不同 PD 机制间雷同，判别力不足；酶/转运体又被 PK 主导。所以 PD 子类之间的区分缺乏可靠结构签名。这与上一份分析的 ~4% WL 碰撞核、以及 ColdDDI 的 Tanimoto 诊断一脉相承——要打破的正是 PD 这批结构上难分的机制。
3. **PK/PD 路由信号是软的（0.76）而非硬的**：支持 memory 里"learned routing 优于 hard PK/PD split"的判断——结构提供软先验，路由应当学习而非手写。
4. **macro-F1 0.064 是诚实的天花板信号**：纯结构对绝大多数机制（尤其尾部 + PD）几乎无召回，机制级多分类不能只靠 KG 结构。

### Caveat
- `ddi_type` 是模板化文本（去掉药名），PK/PD 桶用关键词规则（99.9% 覆盖，388 条 other）。
- 可预测性用 LR + log1p 特征；非线性模型可能略高，但 PK/PD 分裂的定性结论稳健（来自逐机制 one-vs-rest AUC，与分类器无关的通道指纹一致）。

## 复现
```bash
python Code/scripts/analyze_ddi_mechanism_kg_separability.py --top-k 30 --wl-rounds 3
```
