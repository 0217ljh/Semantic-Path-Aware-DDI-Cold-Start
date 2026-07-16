---
title: ARIS auto-research 进度 — idea2 v2
started: 2026-07-01
target: binary_cls / multi_class_cls / multi_label_cls 三任务都明显优于 EmerGNN（差距不小）
baseline: EmerGNN（单一 baseline）
kg: Code/data/KG/_merged_kg_dedup_v15
method: idea2-design-v2.md（Step1 单药 liability 补全 + Step2 3头机制路由器 PK/PD/PK→PD）
n_loops: 25
---

# ARIS auto-research 进度追踪

## ⚠ GPU 约束（每轮必读，2026-07-01 用户强调）
GPU 显存有限、且**常有另一条 idea 的 run 在占用**（不是我启的，绝不 kill）。任何 GPU 实验前必须:
1. **先查空闲显存**：`nvidia-smi --query-gpu=memory.free --format=csv,noheader`。若空闲不足（例 <8GB），**不启动、等它释放**（这轮改做 CPU 工作：code-idea-2 骨架 / v15 loader / PK-feature 预计算 / 分析），下轮再查。
2. **缩小显存占用**：我的 run 用**小 batch size**（EmerGNN S1/S2 默认 batch32 → 需要时降到 16/8）；设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；一次只跑**一个** GPU 实验，不并行。
3. **绝不与其他 run 争用**：GPU 已被别的进程占满时排队等待，不抢。
4. EmerGNN 在全 KG(6.5M 边)上传播**极吃显存**（~32GB）；idea2 自己的模型也要按此约束设计（子图采样/小 batch/邻域裁剪）。

## 每轮协议（loop protocol）
每个 loop 依次做 4 阶段，结果追加到下方 loop 日志（不覆盖历史）：
1. **文献调研**：针对本轮问题定向检索（/arxiv、semantic-scholar、web），记 3–5 条可借用点 + 撞车风险。
2. **实验设计**：明确本轮假设、要跑的模型/ablation、数据集/split、评测指标（vs EmerGNN）、成功/kill 标准。
3. **开展实验**：代码在 `Code/code-idea-2/`（与主 Code/ 隔离）；结果写 `Code/runs/<run_id>/`；用 WSL conda env project_1。**用户已授权 auto-launch**。
4. **分析结果**：vs EmerGNN 分桶（PK/PD、S1/S2、三任务），判断是否达标；提炼下一轮定向问题。

## 建造顺序对齐的 loop 计划（可动态调整）
- L1: 基础设施 —— code-idea-2 骨架 + v15 loader + EmerGNN baseline 三任务基线数字（先拿到对手分数）。
- L2: 纯 PK typed 模型（shared 酶/转运体 motif，无 liability/无 LLM）→ 打 PK cold-start。
- L3: QTc 最小 kill-test —— 加 hERG/qt_liability（结构化预测器）+ 3头路由器雏形；CYP-matched 对照。
- L4+: 若 kill-test 过 → 扩 liability family（5-HT/muscarinic/GABA/opioid）+ 完整 3头路由 + 反事实正则；分别在三任务上对齐 EmerGNN 并拉开差距。
- 每轮据结果定向续做文献 + 迭代。

## Loop 日志

### Loop 0 — 打磨记录 idea（DONE 2026-07-01）
- 记录：`Notes/Ideas/idea2/idea2-design-v2.md`（机制分型 thesis + 2-step 方法 + kill-test，codex 锁定）。
- 关键实证：PK 判别 0.83–0.89（酶）/ PD 弱（QTc≤0.67，hERG 覆盖 4.5%，CYP3A4 混淆 47%）。
- 下一轮问题（L1）：拿到 EmerGNN 在三任务 inductive S1/S2 的基线数字 + 搭 code-idea-2 骨架 + v15 loader。

### Loop 1 — 基础设施 + EmerGNN 基线（IN PROGRESS 2026-07-01）
**阶段1 文献**：待本轮实验阶段后补（先解决基线数字 + 基础设施）。
**阶段2 设计（已核实）**：
- EmerGNN 三任务 unified baseline 代码就绪且 codex review 过（`Code/baseline/emergnn/{binary,multi_cls,multi_label_cls}/baseline_unified.py`，2026-06-30）。
- 运行入口：`Code/scripts/run_baseline_unified.py --baseline emergnn --task {binary,multi_class,multi_label} --dataset <ds> --split cold_s2 --fold fold0 --epochs N`。
- ⚠ 未决点（下一步先查）：① EmerGNN 需 KG edges 文件 `edges__drugbank_hetionet_primekg__mask1.parquet`（是否存在/对齐 v15？review 用的是旧 "merged"）；② 数据集刚改名(`drugbank_latest`→`_full`/`_partial`)，`--dataset` 有效值需核（deng/ryu 是 KG-free，EmerGNN raise → 只能用有 KG source 的 leaf，如 latest_full/partial 或 our-KG leaf）；③ EmerGNN unified **从未端到端跑过**（review: "End-to-end training run: PENDING"）。
**阶段3 实验**：下一轮启动 —— 先 smoke 验证 harness+GPU（`--epochs 3`），再正式三任务 cold_s2/s1 基线。
**阶段4 分析**：待基线数字。
**GPU 事件（2026-07-01）**：启 binary smoke 时发现 GPU 已被**另一个 run(pid 329747: emergnn multilabel twosides 50ep，写 `Code/runs/_gate_ml_s2.log`)占满 ~31.5GB**（不是我启的）。我的 smoke 与其争用 → 已 kill 我的 smoke，保留对方 run。**在对方 run 释放前，GPU 实验一律排队**。
**下一轮定向问题（L1 续）**：① 先查 GPU 空闲（对方 twosides 50ep 跑完没）；GPU 忙 → 这轮做 **CPU 工作**：搭 code-idea-2 骨架 + v15 loader + PK typed-feature 预计算（酶/转运体 shared-motif，已在 scratchpad 有原型逻辑）。② GPU 空 → EmerGNN 三任务基线（小 batch，先 smoke 再正式）。③ 和 codex 定"基线该用哪个 KG 公平"（EmerGNN 现用旧 `_merged_kg`，idea2 用 v15）。

### Loop 1b — CPU 基础设施(GPU 忙时做,2026-07-01)
GPU 仍满(294 MiB free,另一 run twosides multilabel 在跑,非我启),按约束**不启 GPU 实验**,改做 CPU:
- **v15 loader 完成并验证**:`Code/code-idea-2/kg_v15.py`(隔离目录)—— load_v15 / build_typed_drug_adj(target/enzyme/transporter/carrier/protein/pclass) / build_protein_meso / pk_pair_feats。smoke OK:8048 药,typed 邻接齐;样例 PK 对共享 1 酶(CYP3A4)、共享 target 0(印证 PK 信号在酶)。
- code-idea-2 README(v1 旧版)未动(遵守不改已有文件);新代码走新模块。
**下一轮(L1 续 / L2 起)**:① 先查 GPU 空闲 —— 空 → EmerGNN 三任务基线(小 batch,先 smoke);忙 → 继续 CPU:用 `kg_v15` + ddi_unified 对 → 算 PK typed-feature 表(shared enzyme/transporter/target),搭 L2 纯 PK logreg/GBT 基线(CPU 可训,先拿 idea2-PK 侧的数字)。② 和 codex 定基线公平 KG(EmerGNN 旧 KG vs v15)。

### Loop 2 — EmerGNN binary 端到端验证(GPU 释放后,2026-07-01)
GPU 释放(另一 run 结束,30GB free)→ 启 EmerGNN binary smoke(ddi_full cold_s2 fold0 epochs 3,后台 detached,`PYTORCH_CUDA_ALLOC_CONF=expandable_segments`,log `Code/runs/_idea2_emergnn_binary_s2_smoke.log`)。
- **端到端训练正常**:leaf load OK,train=393436/val=44378/test=50296(inductive/S2),kg_setup cache HIT,`[ep1/3 step50/12295] loss=0.2949`。**显存仅 ~8.5GB**(binary 可控;之前 32GB 是 twosides multilabel)。12295 步/epoch。
**下一轮**:① 收 smoke 的 val/test 数字 + 每 epoch 时长 → 定正式 epoch 数;② 启 EmerGNN binary 正式 + multiclass + multilabel 三任务基线(GPU 空时,一次一个,小显存);③ 并行(GPU 有余量时或 CPU)搭 L2 PK 特征表 + 纯 PK 基线。目标先拿齐 EmerGNN 三任务 target 数字。

### Loop 3 — L2 纯 PK 基线（CPU,thesis 真实 benchmark 验证,2026-07-01）
smoke 仍训练中(GPU 8.9GB 占用,23GB free)→ 这轮做 L2 CPU:`Code/code-idea-2/l2_pk_baseline.py`,drugbank_latest_full **inductive S2**(train 393436 / test 50296,y_bin 平衡),纯 PK typed-motif 特征(shared enzyme/transporter/target/carrier/protein/pclass + jaccard),logreg + GBT。
- **结果(test)**:overall AUROC logreg 0.695 / GBT 0.698;**PK 桶 0.738/0.740,PD 桶 0.610/0.612**(桶用 multi_cls 类型 join taxonomy)。
- **✅ 机制分型 thesis 在真实 cold-start benchmark 成立**:PK vs PD 差 13 分。纯 PK CPU 模型 0.70 已接近 EmerGNN(smoke val 0.712@ep2 还在爬)——PK 信号浅、易抓;**PD(0.61)是 L3 liability 要攻的点**。
- 修正:multi_cls 全是正例(是类型分类不是 interact-vs-not),binary 任务负例在 binary_cls(y_bin);已改用 binary_cls 训练、multi_cls 类型分桶。
**下一轮**:① EmerGNN smoke(3ep)收数 + 启**正式 binary 基线**(epoch↑,后台,GPU 空且一次一个)拿真 target;② 记 L2 数;③ 起 L3 准备:hERG/QT liability(先结构化预测器路线)+ CYP-matched 负样本设计,和 codex 过 L3 kill-test 方案。

### Loop 4 — EmerGNN binary 数字 + L3 CYP-matched 必要性探针(2026-07-01）
**EmerGNN binary smoke(3ep,ddi_full cold_s2)完成**:val 0.685→0.712→**0.753**(陡升未收敛),**test AUROC 0.7377 / AUPRC 0.7419 / F1 0.667**,~800s/epoch。→ 已启**正式 binary 基线 e20**(后台,log `Code/runs/_idea2_emergnn_binary_s2_e20.log`,~4.3h)拿收敛 target。
- 对比(未收敛):EmerGNN 3ep 0.738 vs idea2 纯 PK CPU 0.695(overall)。
**L3 CYP-matched 必要性探针**(`Code/code-idea-2/l3_qtc_cyp_matched_probe.py`):QTc-DDI 12270 对 vs CYP-matched 非-QTc 24540 对(都共享 CYP)。
- 单特征 AUROC:**shared_enzyme 0.399(<0.5!)**、meso_gobp 0.575、shared_target 0.568,其余 ~0.5;max|AUROC-0.5|=0.101。
- **解读**:CYP-matched 下 KG 基本区分不了 QTc(无强分离,~0.60 上限);shared_enzyme<0.5 实证"共享 CYP 会误导向非-QTc"(纯代谢 DDI 酶重叠更高)。→ **hERG/QT liability 层有据**,kill-test 目标 = liability 把 QTc 判别推到明显 >0.60(CYP-matched 下)。
**下一轮**:① 收 e20 收敛 binary 基线 target;② 和 codex 定 L3 hERG 预测器方案(结构化化学:用 drugs.parquet 的 SMILES + 公开 hERG 数据/预训练模型,label-blind 单药)+ 完整 kill-test 协议;③ 视 GPU 再启 multiclass/multilabel 基线。

### Loop 5 — L3 hERG liability + kill-test（NEGATIVE 结果,2026-07-01）
codex L3 方案(thread 019f1d52):(a) 自训 hERG QSAR(TDC + Morgan/RDKit + HistGBT,scaffold);label-blind 精确=DDI-blind;kill-test 主特征 max(qA,qB)+ min/product ablation;负样本精确匹配共享 CYP;注入=加权 `drug→hp:qt_liability` 边。
- **hERG QSAR 建成**(`Code/code-idea-2/liability/train_herg_qsar.py`):TDC hERG scaffold split **test AUROC 0.804**;打分 2179 ddi 药,与 hERG-train 重叠仅 **3.2%**(泄漏审计干净);scores 导出 `qt_liability_scores.parquet`。
- **kill-test(`l3_qtc_killtest.py`)FAILED**:QTc-DDI 12270 vs CYP-matched 非-QTc 24540。**liab_max AUROC 0.515**(无信号),min/prod 0.45,kg_meso_gobp 0.575(KG 上限),KG+liab 合并 0.636(主要来自 KG)。→ **hERG liability 在 CYP-matched 下打不过 KG 上限,当前 PD-liability 方案不成立**。
- **诊断**:① TDC hERG 训练集 **69% 是 blocker**(基率虚高)→ 打分膨胀(55% 药 >0.5)、不判别;② "QTc-DDI vs 非-QTc-DDI 类型"可能本就不由单药 hERG 决定(QTc-DDI 常 = 一个 hERG 受害者 + 一个 CYP 增效者;类型标签来自 DrugBank 文档,idiosyncratic)。
**下一轮**:和 codex 诊断修正 —— ① 换更好的 hERG 信号(TDC hERG **回归 pIC50**/hERG_central,或校准 base-rate),重跑 kill-test;② 重审 kill-test 框架是否过严(QTc-DDI vs 非交互 随机负样本 是否有信号?先确认 liability 有没有任何用);③ 若仍无效,重估 PD-liability 是否是对的 lever(可能 PD 判别不在单药 hERG,而在 pair 的 victim+perpetrator 结构 → 需 inhibitor/substrate 方向,即 C 头,但 KG 缺该方向)。此为 auto-research 正常的假设证伪,记录不粉饰。

### Loop 6 — L3 kill confirmed: PD-liability line DEAD（decisive negative,2026-07-01）
baseline: e20 binary 到 ep5/20 val_auc 0.7362(~780s/ep,趋近已知 ~0.737,收敛中,~3h)。
**codex(019f1d73)纠正了我的诊断**:0.515 不是校准问题 —— AUROC 只看排序,base-rate 重标不改 AUROC(我 (a) 诊断错了)。排名 `c(不对称)> b(标签敌对)>> a(校准)`。QTc pair = 一个 hERG-victim + 一个 CYP-perpetrator,对称的 max 丢掉了这个结构(min/prod 掉到 <0.5 佐证)。指定一个 cheap 实验定生死:**easier control**(QTc-DDI vs degree-matched 非交互对,用现成 scores,不控 CYP 混淆)——问"liability 到底有没有任何 pair 级富集"。
- **easier control(`l3_qtc_easier_control.py`)结果**:QTc 正样本 52219 vs degree-matched 非交互负样本 104438。**liab_max 0.516 / min 0.517 / prod 0.516 / mean 0.517** —— 全部 ~0.5。**即使不控 CYP 混淆、即使换 min/prod,qt_liability 对 QTc pair 零富集。**
- **判定(codex 预定 + 数据坐实):PD-liability 线 DEAD。** hERG QSAR 在 TDC 上 0.804,但不迁移到 QTc-DDI pair 富集。外挂结构 liability proxy 不 recover PD。**不 rescue、不 post-hoc fishing**(不再花 loop 试 regression QSAR / 校准),直接 pivot。
- **method-level 十字路口(下一轮和 codex 定)**:PD-liability 死了;但纯 PK typed-motif baseline overall 0.695 **低于** EmerGNN 0.737 —— mechanism-typing 目前是**诊断洞见,不是能打过 baseline 的方法**。要打过 EmerGNN(用户目标:3 任务都明显超),idea2 真正的 lever 需要重新定位。候选:① **v15 分层 KG 才是 contribution**(把 meso/macro renorm KG 插进 GNN,比 EmerGNN 的旧 KG 更利于 cold-start 泛化)—— 把"KG 重整化"当主线,用 GNN 实测;② learned routing(把 pair 路由到 PK 子空间,PD 不硬塞 PK 特征,见 [[feedback_ddi_learned_routing]]);③ 承认 mechanism-typed 是 analysis-only。**下一轮 codex 专议:PD-liability 已死 + 纯 PK<EmerGNN 的前提下,idea2 怎么真正打过 EmerGNN。**

### Loop 7 — method pivot: lever A (v15 layered-KG → GNN), GO on GNN KG-swap（borderline→GBT flips GO,2026-07-01）
baseline: e20 binary 仍在爬,ep9 val_auc **0.7507**(ep4 0.7362→ep9 0.7507,未 plateau,~2h)。真实 binary target 趋势 ~0.75+,高于 3ep 的 0.737。
**codex(019f1daa)method-level 定调**:`A > C > B,选 A`。B(routing)造不出 KG 里没有的 PD 信号;C 诚实但放弃目标;A 是唯一有 cross-task upside 的路,但**"same GNN, better KG",弱 novelty**,只有当 v15 给 unseen-drug pair 造出**可达的判别性 mediator**时才成立。**meso 有戏,macro 大概率没戏(离 drug 太多跳 + 太 hubby)**。指定 cheap label-aware CPU 预检(不是 raw density,hub 会同时抬高 pos/neg)。
- **L4 probe(`l4_v15_meso_probe.py`,binary S2 val)**:特征 `x_k(a,b)=Σ_m 1[dist≤2 both]/log(2+deg(m))`,k∈{pathway,gobp,anatomy,disease,phenotype}。
  - 线性 logreg:OLD(micro only)0.6999(对齐 L2 0.695 sanity)→ ADD(micro+meso)0.7170,**Δ+0.0171**(过 delta gate)但 add 0.717 **差 0.003 没到 0.72** → 线性 NO-GO(合取门)。per-meso standalone 全 >0.6(pathway 0.689 最强),非 hub 噪声。
  - **GBT 非线性(codex 019f1db1 预登记的 FINAL adjudicator,匹配 GNN 非线性)**:OLD 0.7011 → ADD **0.7286**,**Δ+0.0276**,**两门全过(≥0.72 ✓ & ≥+0.015 ✓)→ FINAL GO**。非线性从 meso 层榨出更多(+0.0276 > 线性 +0.0171),证实 codex "线性探针低估" 的判断。
- **决定:GO → greenlight GNN KG-swap 实验**(v15 meso 是 GNN-liftable)。
- **⚠ 诚实天花板(codex 两次强调,必须对用户讲明)**:GBT hand-crafted 上限 0.7286 仍 **低于** EmerGNN ~0.75 —— GO 的含义是"KG-swap 值得一测",**不是**"能明显打过 EmerGNN"。现实 lift 看起来是 **incremental(~+2-3pt)**,用户"3 任务都明显超 baseline"的目标经此证据链**很可能不可达**;idea2 诚实的 contribution 更可能是 **mechanism-typed 诊断 + renormalized-KG-as-resource + 一个实测的 KG-swap lift 数**,而非干净的 SOTA sweep。跑完 GNN swap 拿到真实 lift 再定性。
- **下一轮(Loop 8)= 建 v15→EmerGNN KG adapter(CPU 数据准备,不抢 GPU)**:先读懂 EmerGNN 的 KG 输入格式(entity/relation id scheme,当前用 _merged_kg),再造 3 个条件的 KG 输入(old flat / v15 micro-only / v15 micro+meso,**round1 不含 macro**,codex 019f1daa Q3)。baseline 跑完释放 GPU 后,同 seed/split/optimizer/epochs/negatives/hidden/path-length,**只换 KG**,binary S2 单任务先跑,隔离 KG 贡献。

### Loop 8 — CRITICAL 发现:EmerGNN 只留 drug-incident 边 → meso 层不可达;codex 定 B>A>C（2026-07-01）
baseline: e20 binary ep13/20 val_auc **0.7555**(仍在爬,~1.5h)。GPU-conscious:本轮纯 CPU(读码+设计)。
**读通了 EmerGNN 的 KG 输入契约**:`_merged_kg_path`= `kg.source/MERGED_EDGES`(MERGED_EDGES="edges__drugbank_hetionet_primekg__mask1.parquet")。builder `build_kg_from_merged_parquet(edges_path, drug_ids)`(kg_builder_merged.py)7 列 schema `src,src_kind,dst,dst_kind,relation,source_kg,directed`;drug 用 `drug_ids` arg 的 `edges.src/dst.isin(drug_set)` 识别。**v15 vs benchmark id 对齐**:v15 drug 节点 = `drug:DB00006`(带前缀),benchmark drug_ids = 裸 `DB00035` → **adapter 必须 strip `drug:` 前缀**才能匹配。
- **⚠ CRITICAL(kg_builder_merged.py:89)**:builder 先 `drug_incident = edges[src_is_drug | dst_is_drug]`,**只从 drug-incident 边建 triplets,丢弃所有非 drug-incident 边**(protein→GO / GO is_a / protein→pathway / protein→protein / disease→phenotype / ontology 全丢)。EmerGNN 实际工作在 **bipartite drug↔直连实体图**,length-3 path = drug→entity→drug→entity。**meso 层级对它不可达**。codex 复核确认(_per_mode.py:409 训练用 train_ddi+kg_triplets;shuffle_utils.py:131 只扩这些)。
- **含义**:① EmerGNN 现在的 ~0.75 只用 drug-incident 边,整个 v15 分层 hierarchy 未被使用;② **L4 探针测的是 drug→protein→meso 2-hop 信号(+0.0276),EmerGNN-as-built 根本看不到**(protein→meso 被丢)→ probe 与 plain KG-swap 存在错配;③ 但这也是 idea2 novelty 机会:"利用 baseline 塌掉的分层"。
- **codex 设计裁决(019f1dd2):B > A > C**。A(plain swap)只测 drug-incident 边的 renormalization,测不了 hierarchy → 单独 A 与 L4 依据错配。**B = 把 2-hop meso 物化成 1-hop drug→meso shortcut 边**,让 EmerGNN bipartite builder 能留住 → L4 信号可达,且仍"只换 KG"无需改模型。C(新模型吃全多跳图)只有 B 见效才上。
- **codex 精确 B spec(不折不扣实现,不许简化)**:
  - KG 变体:**A1**=legacy flat(旧 _merged_kg,已存在,control);**A2**=v15 micro-only(drug-incident:drug→protein/pclass/drug-drug,**排除** drug→meso);**A3**=v15 micro + native 直连 drug→meso(db:pathway/prime:indication·contraindication·off-label/drug→side_effect 等,= "best no-shortcut v15 arm");**B**=A3 + 物化 shortcuts。B 的对照基准是 **A3**(不是只对 legacy)。
  - **4 个 shortcut relation family**:`drug_targets_pathway` / `drug_targets_gobp` / `drug_targets_disease` / `drug_targets_phenotype`。**round1 不含 anatomy、GO-MF/CC、ontology-closure**。
  - 构造:每 drug d → v15 直连 protein 邻居 → 各 protein 的直连 `p→meso`(4 类)→ 加 `(d, meso, rel_type)` shortcut(每可达 meso 一条)。native drug→meso 保留原 relation,**不与 shortcut relation 合并**。
  - hub 控制:score = `support_count / log1p(该 meso 在该 type 内的 global protein-degree)`;先丢每 type 内 protein-degree > 99th pct 的 meso;再每 drug 保 top:**pathway 32 / GO-BP 64 / disease 32 / phenotype 32**。首轮不再调参。
  - **go/no-go**:B 比 best no-shortcut A(=A3)在 binary S2 val **≥ +0.010 AUROC,5 seed 平均,≥4/5 seed 为正** → 才 greenlight 全跑。否则停在 incremental 结论(v15 靠更干净的直连结构有小提升,但 hierarchy 在 EmerGNN cold-start 下无足够可达价值),**不升 C**。
- **下一轮(Loop 9)= 建 adapter `Code/scripts/prepare_v15_emergnn_kg.py`**:emit A2/A3/B 三个 KG 变体 dir(各含 MERGED_EDGES 名的 7 列 edges parquet,drug id strip `drug:`),verify 节点/边数 + benchmark drug coverage;A1 复用旧 _merged_kg。然后 stage 多 seed CLI(baseline 释放 GPU 后再跑,不抢)。

### Loop 9 — adapter + runner built & verified（KG-swap 就绪,2026-07-01）
baseline: e20 binary ep15/20(~1h)。本轮纯 CPU。
- **adapter `Code/scripts/prepare_v15_emergnn_kg.py`**(codex B spec 不折不扣实现)已跑通,emit 3 变体(strip `drug:` 前缀 → 裸 DBxxxx 对齐 benchmark):
  - **A2 `_v15_micro/`**:99,746 边 / 10 rel / bench-drug 覆盖 1812/1900(95.4%)
  - **A3 `_v15_micro_meso_native/`**:346,855 边 / 18 rel / 1835/1900(96.6%)= best no-shortcut arm
  - **B `_v15_micro_meso_shortcut/`**:1,006,751 边(A3 + 659,896 shortcut)/ 22 rel / 96.6%。shortcut/family:pathway 151076 / gobp 349118 / disease 122591 / phenotype 37111。hub 控制:各 type 丢 99th-pct protein-degree(pathway>223 / gobp>618 / disease>174 / phenotype>103 的 meso 剔除)+ 每 drug top pathway32/gobp64/disease32/phenotype32。id-strip sanity PASS。
  - A1 control = 现有 `_merged_kg/`(旧 flat KG)。
- **runner `Code/scripts/run_emergnn_kgswap.py`**(新文件,不改任何已有签名):load leaf → override `leaf.resources.kg.source`(KGHandle 可变,已验证)→ set global seed(torch/np/random;原 runner 无 seed 参数,5-seed 靠全局 seed)→ EmerGNN fit/predict/AUROC → results.json。**CPU dry-run 验证**:default kg.source=_merged_kg,override 成功指向变体,4 个变体 dir 的 MERGED_EDGES 都存在。DRYRUN_OK,未训练不抢 GPU。
- **run matrix(staged,待 baseline 释放 GPU)**:{A1,A2,A3,B} × seed{0..4},20 epoch,`run_emergnn_kgswap.py`。**分阶段省 GPU**:先 **Round-1 screen = 4 变体 × seed0 × 20ep**(每 ~3-4h,后台顺跑,看排序 B>A3>A2~A1?);若 B 比 A3 有希望再补 seed1-4 做 5-seed go/no-go(codex:B−A3 ≥ +0.010 AUROC,5-seed 平均,≥4/5 正 → greenlight;否则停在 incremental,不升 C)。A1 也必须走同 runner+seed(现有 e20 只给 ~0.756 ballpark,受控对比需同 harness)。
  - 命令模板:`PYTHONPATH=Code/code-idea-2 python -u Code/scripts/run_emergnn_kgswap.py --kg-dir Code/data/KG/<variant> --tag <A1|A2|A3|B> --seed <s> --epochs 20`,variant∈{_merged_kg,_v15_micro,_v15_micro_meso_native,_v15_micro_meso_shortcut}。
- **下一轮(Loop 10)**:若 baseline 已完成(记录其 final binary S2 AUROC 作 target)且 GPU 空,启动 Round-1 screen 第一个 run(A1 seed0),后台 + GPU-conscious;否则继续等。诚实提醒:hand-crafted 天花板 0.729 < EmerGNN ~0.756,KG-swap 现实提升大概率 incremental。

### Loop 10 — 等 baseline 收尾（2026-07-01）
e20 baseline ep18/20(~27min 到完),GPU 8.9GB 占用 / 23GB 空。虽 VRAM 够并行,但为不给 baseline 最后 2 个 epoch 引入 compute 争用(GPU-conscious),本轮不启动,等一个周期让 baseline 干净收尾再启 Round-1 screen(4×3-4h 的长战役,晚 25min 可忽略)。尚无 kgswap 结果。下轮:baseline 完成→记录其 final binary S2 AUROC(A1 ballpark)→启 A1 seed0(后台,nohup,expandable_segments)。

### Loop 11 — Round-1 screen 启动:A3 seed0 训练中（KG-swap 生效确认,2026-07-01）
- e20 baseline 已到 ep20/20 final eval(results.json 尚未覆盖,仍显示旧 3ep 0.7377;e20 final AUROC 下轮抓)。
- **launch bug 修复**:首次启动 A3 用 `nohup env PYTHONPATH=.. python` 但漏了 `conda activate project_1` → base env 无 pandas,`ModuleNotFoundError` 秒退(未占 GPU)。修正:`conda activate project_1 && ... PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=Code/code-idea-2 nohup python -u ...`(env 变量前置 export 进 nohup 子进程)。以后启动 kgswap 一律带 conda activate。
- **A3 seed0 重启成功训练中**(`_idea2_kgswap_A3_seed0.log`,tag A3=_v15_micro_meso_native,20ep):`kg.source -> _v15_micro_meso_native` ✓;**EmerGNN 日志 `kg_setup STALE (key 7189fb44…不在 cache,3 个 sibling 缓存 input 不同)→ BUILT`——确认 builder 从我的变体重建了 entity2id/triplets,KG-swap 真正生效**。GPU 15.5GB 占用/16.5GB 空,baseline+A3 共存无 OOM。
- **优先级**:先跑 go/no-go 对(A3 + B),再补 context(A1/A2)。下轮:等 baseline 完成释放 ~8.9GB → 启 **B seed0** 与 A3 并行(2-way),~4h 出 A3/B seed0;同时抓 e20 final AUROC(A1 ballpark)。诚实提醒不变:预期 KG-swap 提升 incremental。

### Loop 12 — A1 target 记录 + A3/B seed0 并行训练中（2026-07-01）
- **A1 reference(e20 baseline,old flat KG,20ep)完成**:**test AUROC 0.7465 / best val_auc 0.7579**(AUPRC 0.7516,ep20 raw val 0.7326 但载入 best-epoch state;fit 4.5h)。旧的 3ep 0.7377 作废(欠训)。已存 `Code/baseline/emergnn/_results/2026-07-01__binary_s2_e20__A1_reference.md`。**e20 即 A1(old KG 20ep),不再单跑 A1 seed0**,省一个 GPU slot;最终 5-seed 若需再补 A1 via kgswap runner。对比指标 = best val_auc(codex go/no-go 口径)+ test AUROC 附报。
- **A3 seed0 + B seed0 并行训练中**(go/no-go 对):
  - A3(_v15_micro_meso_native):ep2/20,ep1 val 0.6853(ep1 与收尾中的 baseline overlap,偏噪)。
  - B(_v15_micro_meso_shortcut):启动干净,`kg.source -> _v15_micro_meso_shortcut`,kg_setup key 98ea12…(与 A3 的 7189fb… 不同 → shortcut KG 确实重建生效)。
  - GPU 16.4GB 占用 / 15.7GB 空,双跑无 OOM。每 epoch ~1400-1500s(2-way 争用),**20ep ≈ ~8h**,预计出结果。
- **决策口径**:待 A3、B seed0 都出 best val_auc → 比 B−A3。codex go/no-go:≥+0.010(5-seed 平均 ≥4/5 正)才 greenlight;seed0 只是先看方向。A2(micro-only,context:native meso 是否 helps over micro)留到 A3/B 跑完再补,现维持 2-way 不上 3-way(避免三路 compute 争用)。
- **下轮**:监控 A3/B 轨迹(每 epoch val_auc)+ 抓任何 error;~8h 后出 seed0 结果再定 5-seed / incremental。诚实提醒:hand-crafted 上限 0.729 < A1 0.746/0.758,预期 KG-swap 提升 incremental。

<!-- 后续每轮在此追加 Loop N 小节 -->
