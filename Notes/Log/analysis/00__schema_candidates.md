# Exp 0 — 候选 schema 提名(相遇家族覆盖率)

2026-07-01。脚本:`Code/scripts/analyze_spmn_v2_schema_candidates.py`(只读,无模型,无重训)。
数据:锁定 adapter 的 AND support cache(support_mode=and, l_max=3, k_per_type=64, n_max=400, cp0),
3-seed(42/43/44)池化,**test_s2 pairs=11282,总中介=2,678,474**。

复跑:`python Code/scripts/analyze_spmn_v2_schema_candidates.py`

## 关系桶(adapter REL_BUCKET,retrieval.py:72-82)
0={db:target,prime:drug_protein} · 1=db:enzyme · 2=db:transporter · 3=db:carrier · 4=db:pathway ·
5={het:CbG,het:CdG,het:CuG} · 6={het:CcSE,prime:drug_effect} · 7={prime:indication,off-label} ·
8=prime:contraindication · 9=other · 10=2hop

## Top schema by coverage(全部)
| coverage | mass | asym | schema (rel_a, type, rel_b) |
|---|---|---|---|
| 0.853 | 0.168 | . | (2hop, disease, 2hop) |
| 0.843 | 0.152 | . | (2hop, protein_gene, 2hop) |
| 0.829 | 0.128 | . | (2hop, compound, 2hop) |
| 0.745 | 0.142 | . | (2hop, anatomy, 2hop) |
| 0.740 | 0.084 | . | (2hop, side_effect, 2hop) |
| 0.722 | 0.055 | . | (2hop, molecular_function, 2hop) |
| 0.713 | 0.029 | . | (2hop, cellular_component, 2hop) |
| **0.705** | **0.104** | . | **(2hop, biological_process, 2hop)** ← headline |
| 0.583 | 0.035 | . | (2hop, pathway, 2hop) |
| 0.252 | 0.002 | . | (target, protein_gene, target) ← ≈HIN-DDI |
| 0.238 | 0.002 | . | (enzyme, protein_gene, enzyme) ← ≈HIN-DDI |
| 0.216 | 0.007 | . | (side_effect, side_effect, side_effect) ← ≈HIN-DDI |
| 0.099 | 0.001 | . | (gene-reg/bind, protein_gene, gene-reg/bind) ← fallback |

## 结论
- support **被 2-hop 中介主导**;干净 1-hop 对称 schema ≈ HIN-DDI 手工那些(非新)。
- 1-hop×1-hop **非对称 schema 极稀**(最高 (indication,disease,contraindication) 0.018)。
- **桶合并了源**(bucket0/bucket5)→ 跨源关系精化不可表达 → novelty 落在**中介-KIND + 高阶**。
- headline 候选 = **(2hop, biological_process, 2hop) cov 0.705**;backup (2hop, pathway) 0.583;
  fallback 直接 schema (gene-reg/bind, gene, 同) 0.099。
- ⚠ 这是 **support 覆盖率**(量),不是**注意力重要度**——BP 是否"模型真重视"待 Exp 2 确认。

相关:[[README]] · [[metapath_discovery_analysis_design]]
