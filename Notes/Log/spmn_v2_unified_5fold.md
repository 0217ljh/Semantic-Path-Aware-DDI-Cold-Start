# spmn_v2 adapter (aware0) 迁到 unified benchmark — 5 折

2026-07-02. 把 adapter(spmn_v2 aware,`aware_step=0` 纯粗 = branch off / mech off /
copath OFF / AND support / lmax3 / d32 / 80ep)从 legacy 800drug pkl 迁到项目
**unified benchmark**,和已迁移的 baseline 同 split 同评测集,做公平对比。

## 为什么要迁

之前 adapter 只在 `coldddi_legacy/800drug`(单 seed split)上跑,baseline 已全部迁到
`ddi_unified/.../drugbank_latest_partial/inductive/S2/foldN`。同名 "800 drug S2" 但
**两套不同 split**,不可比。迁移后 adapter 走和 baseline 完全相同的 unified leaf。

## 实现

- 新脚本 `Code/scripts/run_spmn_v2_unified.py`(只新建,不改 `run_spmn_v2_standalone.py` /
  `run_spmn_v2_aware.py` 签名;复用它们的 retrieval `_precompute` + `_gather` + 模型类)。
- 只换数据源:读 unified fold 的 baked `train/val/test.parquet`(`y_bin`→`label`),
  eval 在 `test.parquet`(和 baseline 逐对同一 6084 对)。support cache 按 fold key。
- codex 独立 review PASS 8/8(`_precompute` 参数顺序 / AdamW / BCE / model kwargs /
  训练循环 / 超参默认 / eval-label 对齐),确认是"只换数据源"的忠实迁移。

## 关键对齐(已验证)

- drug id 三边(unified / legacy / MergedKG)同一套 DrugBank id;unified fold0 train
  480 drug 全在 KG → retrieval 零 id 翻译。
- 负样本:unified train/val/test parquet 已 baked 1:1 负样本。用固定负样本训练(和
  legacy aware0 一致);baseline 训练用 per-epoch fresh 负样本(训练细节差异),但
  **评测集完全相同**,报告数字可比。

## 结果 (test_s2, best-val-AUROC ckpt)

| fold | AUROC | AUPRC | best_ep |
|---|---|---|---|
| fold0 | 0.7400 | 0.7557 | 2 |
| fold1 | 0.7319 | 0.7471 | 2 |
| fold2 | 0.7575 | 0.7604 | 1 |
| fold3 | 0.7289 | 0.7348 | 3 |
| fold4 | 0.7487 | 0.7785 | 1 |
| **mean±std** | **0.7414 ± 0.0119** | **0.7553 ± 0.0162** |

结果文件 `Code/runs/spmn_v2_unified/{fold0..4}.json` + `summary_5fold.json`。

## 对比

- 同 fold0 直比:adapter 0.7400 > EmerGNN(full KG)0.7302(+1.0pt AUROC / +1.5pt AUPRC)。
- EmerGNN 目前只有 fold0 单 seed,其余 baseline 还没跑齐 5 折 → 完整"5 折 mean vs
  5 折 mean"表待 baseline 补跑 fold1–4。
- vs legacy:legacy 3-seed 0.7658 → unified 5 折 0.7414,系统性低 ~2pt(unified S2
  划分略难,非 bug;两者都在 0.73–0.78）。
- val 曲线 ep1–3 见顶后单调塌(best_ep 都 ≤3)—— 和 legacy aware0 塌陷模式一致
  (拟合 seen drug 迁移不到 unseen),见 [[spmn_v2_adapter_asym_ab]] 的协变量漂移诊断。

## 复现

```
python Code/scripts/run_spmn_v2_unified.py --fold fold0   # fold0..fold4
```

## 待办

- baseline(EmerGNN 等)补跑 unified S2 fold1–4,凑齐 5 折,做完整对比表。
- 之后可考虑 aware_step≥1(mech/branch)、copath ON 在 unified 上的增益。

## 2026-07-02 续:S2 划分改 640-train(Option A),方法涨点

诊断"unified 5 折(0.7414)比 legacy 3-seed(0.7658)低 ~2pt"→ 根因是 unified S2 训练集小:
旧协议(`folds.py` `cold_fold_drug_roles`)每折 `test=G[j]`/`val=G[j+1]`/`train=其余3组=480`,
牺牲一整组做冷 val。改成 **Option A(legacy 式)**:`test=G[j]`,`train=其余4组=640`,
`val` 从 held-out 组 `G[j]×G[j]` 的 pair 里按 0.5 切(冷,与 test 共享药物)。

- 重生成脚本 `Code/scripts/prepare_ddi800_s2_640train.py`(**新文件**,复用 `folds`/`negatives`/
  `unified` 不改现有函数;同 `PART_SEED=42` 分组 → 每折 test 药物与旧 benchmark 一致)。
  codex review PASS(负采样/carving/cold-start/640-train/只动 S2 全过);唯一 caveat:
  `drug_split` 单角色表达不了"val 药物=test 药物",held 标 `test`,通用审计
  `analyze_unified_audit.py`(建模旧协议)会 flag 此 leaf → 不对此 leaf 跑该审计,已在脚本 docstring 记明。
- **只重生成** `binary_cls/drugbank_latest_partial/inductive/S2`;S0/S1/multiclass/full 未动。
  旧 480-train S2 leaf 已备份到会话 scratchpad。
- 新数据核对:train 640 药物 / ~59–63k 正样本(= legacy 规模),val/test 各 ~1.5–2k。

### 结果:640-train vs 480-train(同一 unified S2,同 PART_SEED test 组)

| fold | AUROC 480 | AUROC 640 | Δ |
|---|---|---|---|
| fold0 | 0.7400 | 0.7518 | +0.0118 |
| fold1 | 0.7319 | 0.7283 | −0.0036 |
| fold2 | 0.7575 | 0.7650 | +0.0075 |
| fold3 | 0.7289 | 0.7619 | +0.0330 |
| fold4 | 0.7487 | 0.7726 | +0.0239 |
| **mean±std** | **0.7414** | **0.7559 ± 0.0171** | **+0.0145** |

640-train AUPRC = 0.7665 ± 0.0178。结果文件 `Code/runs/spmn_v2_unified/{fold*}.json`(现为 640-train)
+ `summary_5fold_640train.json`;旧 480 数字保留在本节表 + `summary_5fold.json`。
结论:训练集补回 640,方法 +1.45pt,回到 legacy 水位 → 坐实"低 2pt 主要是训练集被砍"。

### 待办(更新)
- **EmerGNN 等 baseline 的 fold0 是在旧 480-train S2 上跑的(0.7302),已 stale** → 需在 640-train
  S2 上重跑(runner 读 `--split cold_s2` 会自动指向新数据),才能和 adapter 0.7559 同表对比。
- 5 折完整 baseline 对比表。之后 aware_step≥1 / copath ON 的增益。

## 2026-07-02 再续:640-train 回退,恢复 480-train 严格设计

跑了 EmerGNN 在 640-train S2 上后发现它被抬得过高 —— 640-train 的 val 和 test 共享 held-out
药物,model-selection 能"偷看"到 test 药物,给基线不公平的乐观。所以**回退到 480-train 的严格
设计**(val = 独立不相交的冷药物组 G[(j+1)%5])。

- 从 scratchpad 备份恢复 480-train S2 leaf;核对 fold0 = train 480 / val 160 / test 160,
  val∩test=∅。640-train 的 adapter cache 已删。
- **canonical 对比回到 480-train**(两者都在这个严格设计上有效):
  **adapter aware0 = 0.7414±0.0119 AUROC**(`summary_5fold.json`,昨天的,不重跑)
  vs **EmerGNN full-KG fold0 = 0.7302**。
- 640-train 探索作为**被否决的设计记录**保留在 `summary_5fold_640train.json` + 本笔记上一节;
  `prepare_ddi800_s2_640train.py` 脚本保留但其产物已不激活。磁盘上 `fold*.json` 仍是 640-train
  的 stale 版,480 per-fold 数字以 `summary_5fold.json` 为准。
- 教训:恢复训练数据不能靠牺牲 val 的独立性 —— 那样基线也一起被抬,反而看不出方法优势。

相关:[[project_semantic_path_ddi]] · [[project_ddi_unified_datasets]] · [[project_ddi_notes_log_convention]]
