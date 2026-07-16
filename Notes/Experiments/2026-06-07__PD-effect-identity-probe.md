# PD effect-identity probe (one-hot, go/no-go for KG semantic signal)

**日期** 2026-06-07
**脚本** `Code/scripts/analyze_pd_effect_identity_probe.py`
**产物** `Code/runs/2026-06-07__pd_effect_identity_probe/` (metrics.json, effect_node_weights.csv) — **artifacts 反映最后一次 L=1 run**。
**问题** 共享 effect 节点的 **identity**(one-hot 节点 id)比单纯 **count**,对 PD 的 pos-vs-neg 多带多少 test-AUROC? = KG 的 effect 身份到底有没有判别信号(one-hot = 上界、无 embedding 质量混淆)。

所有数字来自实跑。

## 设置
- 1900-drug 规范 split seed42,S2 冷启动(train=G1 池、test=G2 池,药不相交;effect 节点词表共享 → identity 冷启动合法)。
- train: PD 正(cap 8000)+ train_negatives 负(8000);test: test_s2 PD 正(cap 8000)+ test_s2 负(8000)。
- 特征:undirected BFS ≤L,共享 effect 类节点(Side Effect/effect-phenotype/Symptom/disease)。
  - **F0 (count)**:[共享总数、1hop both、min-depth≤2、total]。
  - **F1 = F0 + identity multi-hot**,词表 V = 训练对共享 ≥20 次的 effect 节点,**cap top-3000 by freq**(只从 train 建)。
- 模型:LogisticRegression(L2, balanced) + HistGradientBoosting。train 上训、**test 上报 AUROC**。

## 结果(verified)

| 深度 | 分类器 | F0 AUROC | F1 AUROC | **gain F1−F0** |
|---|---|---|---|---|
| **L=1**(直接共享 effect)| LR | 0.6054 | 0.6070 | +0.0016 |
| **L=1** | **GBDT** | 0.6056 | **0.6287** | **+0.0231** |
| L=3(深层)| LR | 0.6373 | 0.6138 | −0.0236 |
| L=3 | GBDT | 0.6323 | 0.6428 | +0.0105 |

**L=1 top 正权重 effect(可解释,真实 PD 轴)**:Torsade de pointes、Myelosuppression、Febrile neutropenia、Nephropathy toxic、Apnoea、Aplastic anemia、hypertension…(train_pair_freq 24–110,具体效应)。
**L=3 top 正权重 effect(垃圾)**:polycystic kidney disease、Noonan syndrome、Duchenne…(train_pair_freq ~13600 ≈ 85% 训练对,全局 hub)。

## 结论:borderline,弱真信号
1. **L=3 被 hub 饱和淹死**:深度 2-3 的"共享 effect"是被 ~85% 对共享的全局 hub,identity 无真信号(LR 掉点,top 是无关疾病)。→ **core effect 虽常在深层,但深层=hub 饱和,identity 在那儿不可用。**
2. **L=1 干净、有真但弱的信号**:GBDT **+0.0231**(< 0.03 clean-go,> 0.01 clean-no = borderline),且 top 是 Torsade de pointes/骨髓抑制/肾毒性这些**真机制** → KG 的直接 effect identity **确带判别信号**。
3. **LR≈0 vs GBDT+0.023** → 信号是 effect 的**非线性组合**,非单 effect 线性权重。

## 对方向的判断
- **effect 语义编码值得有限尝试**(1-hop 真 +0.023、可解释),但**单靠它补不平 13 点 PD gap** → 大头需 **分子/文本**。
- **depth-spanning 够深层 core effect 撞 hub 饱和**(L=3 identity 不可用),给"深层 core effect"设想泼冷水。
- 绝对 AUROC 低(0.60–0.63 < NBFNet 0.727):手工结构特征弱,模型能更好,但 identity 的**增量**就这量级。

## 诚实边界
- 单 seed、样本各 cap 8000、V cap top-3000、effect+disease kinds、最短路深度、undirected。
- one-hot=上界:其 borderline 即"语义编码上界也就这么点"。
- 测的是结构里的 effect 身份;**不测**分子/文本。
- L=3 artifacts 被 L=1 覆盖(L=3 数字见本 note 表格)。

## 复现
```bash
# L=1 (clean); 改脚本顶部 L=3 复现深层 hub-saturated 版
python Code/scripts/analyze_pd_effect_identity_probe.py
```
