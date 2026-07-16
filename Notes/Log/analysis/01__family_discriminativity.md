# Exp 1(premise, model-free)— 相遇家族的内生判别性

2026-07-01。脚本:`Code/scripts/analyze_spmn_v2_family_discriminativity.py`(**无模型/无 GPU**,
纯 counts + AUROC)。AND support cache(and, l_max=3, kpt=64, nmax=400, cp0),3-seed(42/43/44)池化,
**pairs=11282, pos=neg=5641**。这是发现故事的 **F1 前提层**:先证家族本身有 DDI 信号,之后模型
注意力(Exp2)+因果删除(Exp3)再证"模型用了它、GNN 没用好"。

复跑:`CUDA_VISIBLE_DEVICES= python Code/scripts/analyze_spmn_v2_family_discriminativity.py`

## 结果
总支撑计数 baseline("中介多→更像 DDI")AUROC = **0.6525**。

| 相遇家族 | count-AUROC | **matched-AUROC**(支撑量分层,去混淆) |
|---|---|---|
| molecular_function | 0.686 | **0.613** |
| pathway | 0.661 | **0.572** |
| biological_process | 0.659 | **0.561** |
| cellular_component | 0.588 | 0.510 |
| disease | 0.622 | 0.510(≈随机) |
| exposure | 0.587 | 0.507 |
| anatomy | 0.536 | 0.496 |
| protein_gene(HIN) | 0.598 | **0.483** |
| compound | 0.598 | 0.470 |
| side_effect(HIN) | 0.586 | **0.459** |

## 结论(诚实)
1. **GO/通路家族(MF 0.613 / pathway 0.572 / BP 0.561)带 DDI 信号,超出"中介多"baseline** —— 正是
   HIN-DDI 手工目录没用的中介类型 → **前提成立,novelty 有据**。
2. **HIN-DDI 自家家族(protein_gene 0.483 / side_effect 0.459)matched <0.5** —— 表观信号几乎全是
   "中介多",非蛋白/副作用特异。**强对照点**(经典 meta-path 的判别力其实来自支撑量)。
3. **disease matched ≈0.510 随机** → 坐实 codex"disease 易被驳成共病",避开正确。
4. **加锐**:`molecular_function` matched 0.613 > biological_process 0.561 → headline 星例 MF 数字最硬;
   MF/BP/pathway 都是 GO/通路类、都超 HIN 目录。case-study 星例可在 MF / BP 间选。

## caveat
- matched-AUROC 用支撑量三分层平均,是"去混淆"的粗近似(非严格偏相关);够作前提,精细版可后补。
- 这是**单变量家族计数**的判别性,不是模型的。"模型是否真重视该家族" = Exp2(需模型,待 GPU 重训)。

相关:[[README]] · [[00__schema_candidates]] · [[metapath_discovery_analysis_design]]
