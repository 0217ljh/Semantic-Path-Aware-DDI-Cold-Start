# Analysis section — meta-path discovery (index)

本文件夹收集 paper **Analysis section** 的所有实验记录 + 结果(meta-path 自动发现线)。
设计总纲见 [[metapath_discovery_analysis_design]](../metapath_discovery_analysis_design.md)。

**约定**:脚本一律在 `Code/scripts/`(项目铁律:`.py` 不进 Notes/);本文件夹放
**实验记录 + 结果 .md**,每个实验一份,不覆盖历史。下表把脚本 ↔ 结果串起来。

## 故事(用户锁定 2026-07-01)
两段式:
1. **性能侧**:高阶相遇处理(l_max=3 纳入 2-hop 中介)→ 涨点(**需消融证明来源**)。
2. **分析侧**:还原具体 meta-path(NBFNet 抽 `Drug-…-BiologicalProcess-…-Drug`)→
   证明"我们发现了传统 GNN 做不好的 path"。

Headline schema(codex round 5)= **共享 biological-process 模块的多跳高阶相遇**;
组合 = **PMP 发现相遇家族,NBFNet 还原路径实现**。不改锁定 adapter(Option A)。

## 实验清单(脚本 → 结果)
| # | 目的 | 脚本(Code/scripts/) | 结果(本目录) | 状态 |
|---|---|---|---|---|
| 0 | 候选 schema 提名(相遇家族覆盖率) | `analyze_spmn_v2_schema_candidates.py` | `00__schema_candidates.md` | ✅ 已跑 |
| 1 | **前提(模型无关)**:家族内生判别性(GO/通路超 baseline;HIN 家族 matched<0.5) | `analyze_spmn_v2_family_discriminativity.py` | `01__family_discriminativity.md` | ✅ 已跑 |
| 2+3 | **走廊内部非顺序 chord**(存在/判别/NBFNet under-use)—— codex kill-test | `analyze_spmn_v2_corridor_chords.py` + `..._crossarm.py` | `02_03__corridor_chords_verdict.md` | ✅ 已跑 → **证伪 NO-GO** |
| — | 高阶相遇**消融**(1-hop vs +2-hop → 证明涨点/不损来源) | (待建) | — | ⏳ 需 GPU |
| 3 | PMP 相遇家族**注意力加权**重要度(GO/BP 是否真靠前) | (待建) | — | ⏳ 需模型(0.7765 无 ckpt→重训) |
| 4 | **因果删除** vs matched control(家族必要性) | (待建) | — | ⏳ 需模型 |
| 5 | plain GNN 对照(归因保真更差) | (待建) | — | ⏳ 需 GPU |

> ⚠ **gating**:0.7765 run **没存 checkpoint**(`run_spmn_v2_aware.py` 只落 predictions npz)→
> Exp3/4 需载模型,得等 GPU 空重训一次(在同进程内抽注意力/做删除,或补存 ckpt)。
> Exp1(前提)已用模型无关方式做完。

## 仍待
- HIN-DDI 确切 meta-path 目录(定 novelty 基准 S_HIN;确认它没用 BP/disease/GO)。
- 外部机制验证集(若最终要生物锚定 case study)。
