# Exp 2+3 — 走廊内部"非顺序 chord"假设:证伪(FULL NO-GO)

2026-07-02。codex-designed(threads 019f20f1 / 019f20fb)。kill-test-first:先验证"drug A→B
走廊里非顺序的 mediator-mediator chord(如 N3→N1)是否携带 DDI 信号、是否被序列方法漏掉",
GO 再建语义门组件。**结论:证伪,不建。**

脚本(只读、无 GPU、CPU):
- `Code/scripts/analyze_spmn_v2_corridor_chords.py`(broad strict_chord)→ `02__corridor_chords.json`
- `Code/scripts/analyze_spmn_v2_corridor_chords_crossarm.py`(精炼:cross-arm-only + 关系黑名单)→ `03__corridor_chords_crossarm.json`
数据:AND support cache(and,l_max3,kpt64,nmax400),3-seed 池化,balanced 采样。

## Exp2 broad(strict_chord = 位置 tuple 不可比的内部边)
- **存在性 GO 但被污染**:strict_chord 覆盖 95.5%,**98.75% 是 same-shell(同位置 laterals)**,cross-arm 仅 1.25%;被 `(anatomy,protein_gene)` 460 万条(`anatomy_protein_present` 通用共注释)灌爆。
- **判别性 NO-GO**:分层 AUROC(支撑量×度四分位)strict_any/density/wdensity 全 ≈ **0.50**(perm_p 0.57–0.69)。唯一预测量 = n_support 0.625(未分层)= **密度混淆**。
- under-use:strict_density FN-vs-TP 0.615 但 CI[0.46,0.75] 跨 0.5(不显著)。

## Exp3 精炼(cross-arm-only,d 位置严格不可比;滤掉 21 个通用共注释/本体关系)
- **cross-arm chord 确实存在**:覆盖 **37%** of |S|≥2,中位数 4 条;保留的关系是机制性的
  (disease_protein / het:CcSE / drug_protein / protein_protein / het:CbG / het:Gr>G / DaG …)。
- **判别性 NULL**:crossarm_any AUROC **0.506**(perm_p 0.183);crossarm_wdensity **0.511**
  (perm_p 0.060,CI[0.497,0.524] 跨 0.5)。**均未过预登记阈值(>0.53 且 p<0.05)。**

## VERDICT(预登记)
**FULL NO-GO**:控制走廊支撑量 + 度之后,**内部 cross-arm mediator chord 对 DDI 无增量信号**
(此 KG/测试设定下)。→ **不建 chord / 语义门组件;不再做 chord-family 测试。**

## 含义(对 paper)
- "非顺序走廊语义"这条线**经验上不成立**,meta-path 分析故事**不能**押在 chord 上。
- 真正有信号的是 **Exp1 的中介-KIND 判别性**(molecular_function 0.613 / pathway 0.572 /
  biological_process 0.561,超支撑量 baseline;HIN 自家 protein/side_effect matched<0.5)——
  发现故事继续用这个,而非 chord。
- 方法侧:当前 adapter(逐 typed 共邻池化,无 mediator-mediator 边)没漏掉可用信号,维持。

相关:[[README]] · [[00__schema_candidates]] · [[01__family_discriminativity]] · [[metapath_discovery_analysis_design]]
