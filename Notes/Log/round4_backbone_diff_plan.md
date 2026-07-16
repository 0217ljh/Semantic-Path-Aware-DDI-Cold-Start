# Round 4 — 从 readout-side aux 推回 backbone，让架构跟 EmerGNN 有"足够区别"

**写这份的目的**。我把当前 0.78+ 模型跟 EmerGNN 的真实差异盘点清楚，然后给出 4 个候选改造方向。另一个窗口接手时，挑一个方向直接进入 codex review → 实现 → 跑 verified baseline → 控制实验 的流程即可。

**两个硬目标**（同时满足，缺一不可）。
1. **架构上跟 EmerGNN 出现可见的、非 readout-only 的差异**，让 paper 的 novelty claim 能撑得住 AAAI 2027 reviewer。
2. **不能掉点**。当前 0.7804 (v2i4_main, seed42, single-rep) 是地板，新方法必须 ≥ 0.78 同 setting；理想 ≥ 0.79。

---

## 1. 当前 0.78+ 模型快照（verified）

**Run**. `Code/runs/2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42/`
**Combined AUROC**. **0.7804**，AUPRC 0.7922，NLL 1.472（S2 test，1919 pos / 1919 neg）。
**分支级 AUROC**（来自 `auc_emergnn=0.7405 / auc_count_only=0.6688 / auc_i4_only=0.6003`）。EmerGNN 分支 0.7405，counts head 0.6688，i4 head 0.6003，combined 0.7804。

**架构**（`Code/my_code/models/screen_s2_v3_multimodal/v2i4_trainer.py:103` 起）。

```
combined_logit = emergnn_logit                       # EmerGNN 原样
               + β_count × MLP_22→32→1(c_a,b)       # i2, MNAH Stage 1
               + β_i4    × MLP_13→32→1(p_a,b)       # i4 LLM 机制特征
β_* = softplus(raw_β),  β_count init = 1.0, β_i4 init = 0.5
```

- `c_a,b`. 22 维 KG meeting-node 按 type 计数（`Code/data/_cache/meet_counts_drugbank_*.parquet`，由 `precompute_meet_features.py` 产出）。
- `p_a,b`. 13 维 LLM 蒸馏机制重叠特征（共享 CYP substrate / inhibitor / inducer，CYP inhib→subst cross 等），来自 `Code/data/_cache/llm_pharma/i4_typed_sets.json`。
- backbone 完全继承自 `_PerModeEmerGNN_MNAH`，**没改一行 propagation 代码**。
- training 仍走 EmerGNN 的 `shuffle_train(mode="S2")`，drug-level 20% emerging cold simulation。

**入口命令**（已 verified 可跑）。
```
python -u Code/my_code/models/screen_s2_v3_multimodal/run_v2i4.py \
    --epochs 100 --tag v2i4_main_seed42 --seed 42
```

---

## 2. 跟 EmerGNN 的逐项 diff（盘清楚再说）

| 维度 | EmerGNN | v2i4 (0.7804) | 改了？ |
|---|---|---|---|
| Path-flow GNN backbone, length=3 | 是 | 是，原样继承 | ❌ |
| Edge type embedding（KG relation） | 是 | 是 | ❌ |
| `shuffle_train(mode="S2")` drug-level cold sim | 是 | 是 | ❌ |
| Negative sampling | 是 | 是 | ❌ |
| Readout 22-d count aux head（i2） | 否 | 是 | ✅ |
| Readout 13-d LLM 机制 aux head（i4） | 否 | 是 | ✅ |
| Backbone 用到 meeting-node identity | 否 | **否**（只在 readout） | ❌ |
| Backbone 用到 LLM 字段 | 否 | **否**（只在 readout） | ❌ |
| 多模态对齐 m_u↔k_u InfoNCE 落地到主路径 | 否 | **未生效**（v2_main 0.7719 比 v2i4 还低） | ⚠ |
| Learned routing 进 path-flow attention | 否 | **未做** | ⚠ |

**结论**。架构差异 = readout 上 + 2 个加性 MLP head。reviewer 会判 incremental。

---

## 3. 必须解决的两个问题

**P1（novelty）**. 至少把当前 work 的两个信号源（meeting-node, LLM 字段）中**至少一个**推回 backbone 里，而不是停留在 readout 加 head。

**P2（保点）**. 推回 backbone 不能让 combined 跌破 0.78。如果直接替换 readout 会掉点，**采取并存策略**（backbone 改造 + 旧 readout head 同时保留，让旧 head 当 fallback），最差也是 ≥ 0.78。

---

## 4. 候选方向（4 个，按可执行性 + 期望 lift + 故事完整度排序）

### D1（强推第一优先级）. **LLM 机制字段 → 新 edge type，注入 KG，让 path-flow 自己学（i4 → backbone）**

**思路**. 当前 13 维 LLM 特征是"算完才喂到 readout MLP"。改成在 KG 里**新增节点类型 + 新增 edge type**，让 EmerGNN 的 path-flow 在 backbone 自己看到 LLM 字段。

**具体设计**.
- 从 `Code/data/_cache/llm_pharma/i4_typed_sets.json` 抽 10 类字段（CYP substrate/inhibitor/inducer, transporter substrate/inhibitor, therapeutic_class, primary_targets, pd_effects, toxicity_mechanisms, clearance）。
- 对每类字段，把每个 token 当成一个新 KG 节点（类型 `cyp_substrate`, `pd_effect`, ...），从 drug 节点连一条新 edge type（10 个新 relation type）。
- KG 从原 ~180k 节点扩展到 ~200k，relation type 从 K 扩展到 K+10。
- EmerGNN 的 propagation 改动 = **零**，只是 KG 输入更大、relation embedding 多 10 行。
- 训练时 i4 信号通过 length≤3 的 path-flow 自然进入 drug pair logit。
- **readout 侧的 13-d i4 head 保留**（fallback，β_i4 仍可学）。

**控制实验**.
- shuf-token control. 把 LLM tokens 在 drug 间随机重排（保留 cardinality），新 edge 接错对象。期望 backbone gain 显著回落。
- random-token control. tokens 替换成随机字符串（保留 type 维持 vocab 大小）。
- ablation drop i4-head. 关掉 readout 那 13-d head，纯看 backbone 是否单独 carry。

**期望**.
- combined ≥ 0.785，理想 0.79+。
- `auc_emergnn` 分支从 0.7405 涨到 ≥ 0.755（这是 backbone 真吸收 LLM 信号的证据）。
- shuf-token control 至少回落 1.5 pt 以上。

**风险**.
- KG 扩展后 path budget 是否仍 length=3 够用。
- vocabulary 维度爆炸（如果 token 数太多），可能需要 token clustering。先跑全 vocab 看 OOM 风险。

**关键路径**.
- 复用 `precompute_kg_typed_target.py` 类似的 KG builder pattern（已存在）。
- 写新 builder `precompute_kg_with_llm_edges.py` 落盘新 KG triple parquet 到 `Code/data/_cache/`。
- 写新 trainer `v3_llm_edge_trainer.py`（继承 `_PerModeEmerGNN_V2I4`，只在 KG 输入处接新 triple 集合）。

---

### D2. **Meeting-node-aware propagation（i2 → backbone）**

**思路**. 当前 22-d counts 是 readout 旁路。改成在 EmerGNN propagation 的第 t 层 message function 里，给"既属于 a 的 path 又属于 b 的 path"的中介节点加 attention bonus。

**具体设计**.
- 对 batch 内每对 (a, b)，预计算 meeting-node mask（在 `precompute_meet_features.py` 已有的中介集合上）。
- EmerGNN 现有 propagation 在每层 t 聚合 message。新加一项 `α_meet × m_v × is_meeting(v, a, b)`，其中 α_meet 是可学参数，`is_meeting` 是 mask（1 表示 v 是 a/b 的共享中介）。
- **readout 的 22-d count head 保留**作 fallback。

**控制实验**.
- mask shuffle. 把 meeting mask 随机重排到别的 pair，期望 backbone gain 回落。
- α_meet=0 ablation. 关闭新加项，期望退化到 v2i4 0.7804。

**期望**.
- combined ≥ 0.785。
- `auc_emergnn` 分支从 0.7405 涨到 ≥ 0.75。

**风险**.
- 每个 batch pair 都要算 meeting mask，可能慢。可以 cache。
- attention bias 加在哪一层最优需要 sweep（最后一层 / 全层）。

**关键路径**.
- 拷贝 EmerGNN propagation 模块到 `screen_s2_v3_multimodal/`（保 file 独立性），改 message function。
- 不要原地改 `Code/baseline/emergnn/`（违反 "禁止修改已有文件"）。

---

### D3. **Alignment (m_u↔k_u InfoNCE) 进 backbone init embedding**

**思路**. 你之前 lock 的多模态对齐核心点目前**没贡献到 0.78+**（v2_main 0.7719 < v2i4 0.7804）。原因是 alignment loss 只跑了，没把对齐产物用进主路径。改成 alignment 的 m_u（分子）vector 作为 KG drug node embedding 的 init，而不是单独算个 head。

**具体设计**.
- 先跑 `train_alignment_infonce.py`（已存在），存下对齐后 m_u 向量（每药 1 个 64d）。
- 改 EmerGNN drug node embedding 的 init = `concat(原 embed, projection(m_u))` 或 `原 embed + projection(m_u)`，再走 path-flow。
- alignment 当 pretraining 阶段，main 训练时 m_u 可 freeze 或 fine-tune。

**控制实验**.
- shuffled-mol control. m_u 在 drug 之间打乱，期望 backbone gain 回落（类似 E3 init mirror）。
- random m_u. m_u 替换为高斯随机向量。
- no-init ablation. 关掉 m_u init，验证退化到 baseline。

**期望**.
- combined ≥ 0.785。
- 把"多模态对齐"这条 paper 卖点真正落地（之前没落地是大缺口）。

**风险**.
- v2_main (xattn) 之前没 carry 数字，alignment 信号本身可能比想象弱。要先 sanity check alignment 训出来的 m_u 在 drug similarity retrieval 上是否合理。
- 跟 D1 / D2 不冲突，可以最后并到一起。

---

### D4. **Pair-conditional relation routing 进 path-flow attention**

**思路**. EmerGNN 在每一步对每个 relation 算 attention 权重，但是 pair-agnostic。改成 pair-conditional——对每对 (a, b)，根据 a/b 的 LLM PK/PD 标签或 KG context，gate relation attention，让 PK pair 偏走 transporter/CYP relation，PD pair 偏走 target/pathway relation。

**具体设计**.
- pair feature `f(a, b)`（可以直接复用 v2i4 那 13 维 + drug-level features）。
- gate_t = sigmoid(MLP(f(a, b))) × |R|，乘到第 t 层的 relation attention logits 上。
- 不强制 PK/PD label 监督，让它自学。

**控制实验**.
- shuf-pair-feature. f(a, b) 在 batch 内重排，期望 backbone gain 回落。
- 看 gate 是否真的在 PK vs PD pair 上区分（用 ddi_pk_pd_labels.csv 做 post-hoc 验证，跟 stage3 一样）。

**期望**.
- combined ≥ 0.785。
- 把"routing"卖点落地。

**风险**.
- 已经做过 stage3 的 PK/PD dual-channel 在 readout 层，combined 0.7696 没超过 MNAH。直接搬到 backbone 不保证更好。先做小规模 probe。

---

## 5. 推荐执行顺序

1. **D1 优先**（LLM 字段当新 edge type）。预计 lift 最大，控制实验最干净，故事最 paper-friendly（"我们识别了 LLM-distilled mechanistic edges 作为 KG 上 missing 的 evidence layer，把它当成新 relation type 直接进 path-flow"）。
2. D1 落地后跑 **D2**（meeting-node-aware propagation），D1 + D2 并存可加性叠加。
3. D3 + D4 看时间，作为 paper section 4 的"完成 multimodal + routing 卖点"补完。

---

## 6. 跑实验的规范约束（必须遵守）

**Anti-fabrication**. 任何 paper claim 之前必须 verify。引用数字附 file:line 或 run_id。不许凭印象说"我估计有 X"。

**WSL conda env**. 训练 / 评测必须用 `wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python ..."`。Windows native Python 是 CPU torch，会 silent fallback 到 ~10x 慢。

**Log 规范**. 每次 run 写 `Code/runs/<run_id>/results.json` + `train.log`，同步镜像到 `Code/runs/_logs/<run_id>.log`（已有的 `RunLogger` 工具）。

**保点 hard-stop rule**. 每个方向的 first main run（seed42, 100 epoch）出来后：
- combined < 0.775. **立刻停**，不要继续 sweep，回来讨论是否设计有问题。
- 0.775 ≤ combined < 0.785. 允许跑 shuffle control + 1 个 hyperparam sweep，再判断。
- combined ≥ 0.785. 进 multi-seed (seed 43, 44) 验证 robustness，进 control 全套。

**Codex review gate**. 每个方向的实现先过 codex 1 轮 review（用 `mcp__codex__codex`），critical issue 修完再跑训练。这是 reproductions/baseline 规范里规定的。

**禁止改的文件**.
- `Code/baseline/emergnn/`（baseline 独立性）。
- `Code/my_code/models/screen_s2_v2_meetnode/mnah_trainer.py`（Stage 1 已固化）。
- `Code/my_code/models/screen_s2_v3_multimodal/v2i4_trainer.py`（0.7804 baseline 不动）。
- 新方向**全部新建文件**（`v3_*_trainer.py`，`run_*.py`）。

---

## 7. 数据来源（必须先读完再写代码）

### 7.1 DDI 数据（drug pair + cold-start split）

**Canonical split**. `Code/data/coldddi_legacy/800drug/seed42.pkl`，verified 存在，121.7 MB。

- **构造来源**. 上游路径 `Code/data/private/outputs_full/splits_legacy/800drug/latest_drugbank_ddi-Binary_cls-{42..46}+cold_start_split_fair_step-and-fair_negatives_step.pkl`（来自 `Notes/Experiments/first_step_plan.md:49`）。
- **scope**. DrugBank binary DDI，800 drug = **640 G1 (SEEN) + 160 G2 (UNSEEN)**。S2 = both unseen。test_s2 = 1919 正样本 + 1919 负样本（已 verified from `results.json` 中 `n_pos`/`n_neg`）。
- **seeds 可用**. 42, 43, 44, 45, 46。Multi-seed 默认用 42 + 43 + 44。
- **SMILES**. `Code/data/coldddi_legacy/800drug/drug_smiles__seed42.csv`（149 KB），1994 行（注：跟 800 drug split 的 universe 不是 1:1，1994 是分子 SMILES 全表）。
- **negative sampling**. fair-step（split 文件内 baked in，不要在 trainer 里重抽）。

**Why 800-drug not 1900-drug**. 800-drug 是 EmerGNN 原 paper 的 setting，所有 baseline 在这上面 comparable。1900-drug split 在 `Code/data/private/outputs_full/splits_legacy/1900drug/P0/` 也存在但**不是**当前 0.7804 的 setting，**不要切**，会让 0.78+ 对照失效。

### 7.2 Knowledge graph（KG）

**Canonical merged KG**. `Code/data/KG/_merged_kg/`（verified 存在）。

- `nodes__drugbank_hetionet_primekg.parquet`. **178,029 节点**，columns `[id, kind, name, source_kg]`。merged from DrugBank + Hetionet + PrimeKG。
- `edges__drugbank_hetionet_primekg__mask1.parquet`. 19.9 MB edge list。mask1 = 默认 release mask。
- **Per-source KG**（备用，做 ablation）. `Code/data/KG/drugbank/`, `Code/data/KG/hetionet/`, `Code/data/KG/primekg/`, `Code/data/KG/ddinter/`, `Code/data/KG/twosides/`。
- **DrugBank-only filtered**. `Code/data/KG/drugbank/filtered/` 含 `ddi_edges.csv` + `id2name.json`。
- **enriched DDI**. `Code/data/KG/drugbank/enriched/ddi_edges_enriched.csv`（含 DDI type 信息）。

**当前 0.78+ 用的是哪一份**. v2i4 trainer 的 KG 输入是 merged KG（178,029 nodes）。**D1 新增 LLM edge 时直接在 merged KG 上扩展**，不要回退到 drugbank-only。

### 7.3 Drug text（PubMedBERT 用，Stage 2 已用）

**节点名 PubMedBERT [CLS] embeddings**.
- `Code/data/KG/_merged_kg/_cache/screen1_tag_init/d_name_only__pubmedbert.pt` — `dict{node_ids: list[178029], embeddings: Tensor[178029, 768]}`，按 KG 节点 name 字段编码（**所有节点 name**，不只 drug）。
- `Code/data/KG/_merged_kg/_cache/screen1_tag_init/e_shuffled_text__pubmedbert.pt` — 同 shape，shuffled-name 对照（E3-style 容量-vs-语义控制）。
- Drug text 原始结构化字段在 `Code/data/KG/drug_text/drug_profiles.json` 和 `drug_profiles_flat.csv`（已扁平化，1 行 1 药）。

### 7.4 LLM 蒸馏机制字段（i4 用，D1 也会用）

**Source**. `Code/my_code/models/screen_s2_v3_multimodal/distill_llm_pharmacology.py`（verified 8180 bytes，2026-05-26 22:15 跑）。

- **LLM**. `claude-haiku-4-5-20251001`（hardcoded `MODEL` 常量 line 50）。
- **Prompt 设计**（line 52–60）. **硬约束 cold-start leakage**。Prompt forbids (a) mentioning any other drug / drug class as interacting partner; (b) DDI phrasing ("when combined with", "coadministered with", "interacts with", "increases levels of [drug]", "avoid with", "contraindicated with")。**only intrinsic** properties = primary targets, MoA, therapeutic class, CYP/transporter roles, clearance, PD effects, toxicity mechanisms。
- **Sanitizer**. 跑完后扫每条 output 找 (a) 任何 1900-drug vocab 里的别的药名；(b) DDI phrase。命中处 redact 并 leakage-flag for audit。codex thread `019e6734` 的 leakage protocol。
- **二次 sanitizer**（v2i4_trainer.py:128–141）. 加载 i4_typed_sets.json 时再扫一次 free-text 字段（`therapeutic_class`, `primary_targets`, `pd_effects`, `toxicity_mechanisms`, `clearance`），drop 含 "interact", "coadminist", "co-administ", "combined with", "concomitant", "avoid with", "contraindicated", "with inhibitor", "with inducer", "increase levels", "decrease levels", "co-medic", "co-prescri" 的 tokens。

**Cache 产物**（verified 存在）.
- `Code/data/_cache/llm_pharma/llm_pharma.jsonl` (4.9 MB). 每行一药 `{drug_id, name, raw, sanitized_text, structured{...}, leakage_flag, partner_hits, phrase_hits}`。
- `Code/data/_cache/llm_pharma/i4_typed_sets.json` (980 KB). 结构化 per-drug typed sets，给 v2i4_trainer.py 直接读。**D1 必须基于这个文件，不要重跑 LLM**。
- `Code/data/_cache/llm_pharma/llm_text_pubmedbert.npz` (4.7 MB). LLM 输出文本的 PubMedBERT embedding（v2llm 用过，i4 没用上）。
- **API key**. `API-KEY/API-KEY.txt`（不入 git，**不要 print / log**）。

**D1 用这份 LLM 字段时必须保留两个 sanitizer**。Trainer 里 v2i4_trainer.py:128–141 那段 free-text 二次扫描代码**复制过去**，不要 silently drop。

### 7.5 Pair-level cache（MNAH / i4 都用）

- **22-d meeting-node count cache**. `Code/data/_cache/meet_counts_drugbank_*.parquet`，由 `Code/my_code/models/screen_s2_v2_meetnode/precompute_meet_features.py` 产出。columns = `[drug_a_id, drug_b_id, f_0..f_21]`。
- **64-d PubMedBERT shared-mediator text cache**. `Code/data/_cache/meet_text_real_drugbank_seed42_kgonly_v1.parquet`（490 MB），由 `precompute_meet_text_features.py` 产出，PCA 64d fit on train-pair pooled vectors only。
- **shuf 对照**. `meet_text_shuffled_drugbank_seed42_kgonly_v1.parquet`（490 MB）。
- **degree cache**. `precompute_degree_features.py`。

**Canonical pair convention**（重要，别搞错）. 所有 cache 用 `canonical_pair(a, b) = (a, b) if a <= b else (b, a)` 排序。trainer 的 `_canonical(a, b)` 和 cache 的 row 顺序按这个对齐。merge 时用 `validate="one_to_one"`，错位会 raise。

### 7.6 Annotation（分析用，不训练）

`Code/data/private/outputs_full/annotations/`.
- `pkpd.parquet`. PK / PD class per DDI pair。**只用于 post-hoc 分析**（Stage 3 的 gate 验证），never trained on。
- `ab.parquet`. A/B asymmetry。
- `action_pairs.parquet`. action type。
- `mediating_entities.parquet`. mediator label。

### 7.7 已 verified 的可复用代码资产

| 资产 | 路径 | 用途 |
|---|---|---|
| EmerGNN baseline impl | `Code/baseline/emergnn/`（**不许改**，复制后改） | path-flow propagation 起点 |
| EmerGNN reproduction | `Code/reproductions/EmerGNN/` | paper-faithful 参考 |
| MNAH Stage 1 trainer | `Code/my_code/models/screen_s2_v2_meetnode/mnah_trainer.py`（**不许改**） | 22-d count head + logit fusion 模板 |
| v2i4 trainer (0.7804) | `Code/my_code/models/screen_s2_v3_multimodal/v2i4_trainer.py`（**不许改**） | 当前 SOTA，继承基类 |
| Run logger | `Code/my_code/utils/run_logger.py` | 所有新 run 必须用 |
| Training progress | `Code/my_code/utils/train_progress.py` | per-step + per-epoch loss log |

---

## 8. Codex review 使用规范（必须遵守，不是可选项）

**为什么必须有**. 项目历史上每个 work 的 MNAH/v2i4 都经过 codex 多轮（round 11/12/13/15）审过，每轮都揪出过 critical bug（normalizer fit bias、shuffle control 失效、sigmoid-MSE 饱和、PCA train-only leakage 等）。新 backbone 改造比加 readout head 更容易出 silent bug（KG triple 漂移 / propagation 数值不稳定 / cold-start leakage），**绝对不能跳过 review**。

### 8.1 调用接口

**MCP tool**. `mcp__codex__codex`（同窗口可用），或 `mcp__codex__codex-reply`（继续上一轮）。
- `model`. 用 `gpt-5-codex`（项目历史默认）。
- 单次最长 prompt ~50K chars，过长拆多次。

### 8.2 哪几个 checkpoint 必须过 review

**Round 4 任一方向（D1–D4）的标准流水线**.

| Checkpoint | 触发 | review 内容 | PASS gate 后允许的下一步 |
|---|---|---|---|
| **CP-1 设计 review** | 设计草案 `.md` 写完，**还没写代码** | 看设计有没有 cold-start leakage、控制实验是否设计正确、是否复用错文件、是否违反"禁止修改已有文件" | 才能开始写代码 |
| **CP-2 实现 review** | 新 trainer / builder 写完，**还没跑训练** | 看代码是否 paper-faithful、normalizer fit 是否正确、shuffle/random control 是否真的破坏信号、API 调用是否正确 | 才能跑 seed42 main run |
| **CP-3 结果 review** | seed42 main 跑完，combined ≥ 0.775 | 看 control 跌幅是否符合预期、是否有 capacity confound、AUROC lift 归因是否成立 | 才能跑 multi-seed + 写 paper section |

**任何 critical issue 必须修完再前进**，不能挂着继续。

### 8.3 review prompt 模板

写 prompt 时按 `CLAUDE.md §"Code review 报告归档规范"` 的 contribution-first 原则，**先列要 review 的 contribution / mechanism**，然后让 codex 逐项验证。

**CP-1 设计 review prompt 模板**.

```
请 review 以下 DDI cold-start backbone 改造设计。背景：当前 baseline v2i4 = 0.7804 AUROC，
backbone 是 EmerGNN（path-flow GNN, length=3），改造目标见 Notes/Log/round4_backbone_diff_plan.md。

本次设计的核心 claim（请逐条验证）.
1. <一句话 claim>。具体做什么改动。
2. <第 2 个 claim>。
3. <第 3 个 claim>。

要 review 的设计文档. <粘贴 design .md 内容或路径>。

请重点检查.
- Cold-start leakage. S2 setting 下 unseen drug 的特征是否真的不依赖 train DDI pair 信息。
- Control 设计. shuffle / random / shuf-text 的 control 是否真的破坏目标信号、保留容量。
- 跟现有文件的耦合. 是否违反 "禁止修改 mnah_trainer.py / v2i4_trainer.py / baseline/emergnn"。
- 数据 path 是否对得上 Code/data/_cache/、Code/data/KG/_merged_kg/ 的现有 cache。
- 期望 lift 是否有理论支撑。

输出 verdict ∈ {PASS, PASS_WITH_NITS, NOT_PASS, FAIL}，每条 issue 标 critical/major/minor。
```

**CP-2 实现 review prompt 模板**.

```
请 review 以下 trainer 实现。设计文档已在 CP-1 PASS（粘 CP-1 review 报告链接）。

代码文件.
- <path/to/new_trainer.py>
- <path/to/build_*.py>
- <path/to/run_*.py>

请逐项验证以下 contribution 真的体现在代码里.
1. <Claim 1>。check file:line 是否真的在做这件事，数据流是否 trace 得通。
2. <Claim 2>。
3. <Claim 3>。

特别检查.
- normalizer / PCA 是否 fit on train-only（防 leakage）。
- shuffle control 是否真的破坏 pair-specific 绑定（不是只换 row 顺序）。
- 训练 loss / β / softplus / sigmoid 是否数值稳定。
- forward 在 batch=1 / 全负 batch / 全正 batch 是否会 nan。
- emergnn backbone 部分是否 byte-equivalent 跟 baseline/emergnn（如不是，列 diff）。

verdict ∈ {PASS, PASS_WITH_NITS, NOT_PASS, FAIL}。
```

**CP-3 结果 review prompt 模板**.

```
请 review 以下 backbone 改造的实验结果。设计 CP-1 PASS、实现 CP-2 PASS。

Main run.
- run_id. <id>，combined AUROC <X>。
- 分支级 AUROC. emergnn=<>, count=<>, i4=<>。

Control runs.
- shuffle. combined=<>, expected to drop by ≥1.5pt（actual drop=<>）。
- random. combined=<>。
- ablation drop new module. combined=<>。

请判断.
1. lift 归因是否成立。real – shuffle 的差距是否真的来自目标 mechanism。
2. 是否有 capacity confound（参数量增长是否能解释 lift）。
3. backbone vs readout 的分支级 AUROC 变化是否符合 "信号进 backbone" 的预期（emergnn 分支应该涨）。
4. NLL 跟 AUROC 是否同向（dissociation 会触发 calibration vs ranking 讨论）。

verdict ∈ {PASS, PASS_WITH_NITS, NOT_PASS, FAIL}，并建议是否进 multi-seed。
```

### 8.4 PASS / 不 PASS 的判定标准

按 `Code/baseline/emergnn/_reviews/2026-05-18__baseline_round5_PASS.md` 既有判例.

- **PASS**. 0 critical + 0 major issue。可以前进。
- **PASS_WITH_NITS**. 0 critical + 0 major，只有 minor（如 log 不规范、文档缺）。前进，但同时开 follow-up 修 minor。
- **NOT_PASS**. 1+ critical。**立刻停**，修完重 review，round 数 +1。
- **FAIL**. multiple critical + 设计有根本问题。回到 CP-1 重新做设计。

### 8.5 review 报告归档（强制）

按 `CLAUDE.md §"Code review 报告归档规范"`.

- 路径. **co-located**，跟代码同目录。新 trainer 在 `Code/my_code/models/screen_s2_v3_multimodal/`，对应 review 报告放 `Code/my_code/models/screen_s2_v3_multimodal/_reviews/`（**新建**，目前没有）。
- 文件名. `<YYYY-MM-DD>__<scope>__round<N>.md`，N=1 起，**不覆盖历史**。同一天多轮加 round suffix。
- 必含字段（已规范化）.
  1. 元信息. 日期 / Primary reviewer / Independent reviewer (= codex gpt-5-codex) / 触发方式。
  2. Contribution 列表（CP-1）或 code 验证（CP-2/3）。
  3. Codex 独立 verdict 章节. **原样复制 codex 输出，不要总结改写**。
  4. Critical / Major / Minor 三类 issue 清单 + file:line + 修复 commit。
  5. 未决 issue / future work。

**Independent reviewer 字段不许留空**。如果某轮跳过 codex（不允许，但如果有特殊原因），必须 explicit 写 "Independent reviewer. 未调用，原因 = <>"，不准 silent。

### 8.6 同窗口 codex 循环用法

复用 `mcp__codex__codex` + `mcp__codex__codex-reply` 做迭代.
- 第 1 轮 call `codex` with prompt，拿 verdict。
- 如果 NOT_PASS，修代码后 call `codex-reply`，把"我已修复 X / Y / Z，请重 audit"作为 reply。
- 直到 PASS。每轮的 verdict 都落进同一份 review .md 的 timeline 表（见 round5_PASS.md line 12–21 的格式）。

### 8.7 反例（项目历史上踩过的坑）

- ❌ "我估计 shuffle control 会跌"，没真跑就声明 control 设计 OK。**必须实际跑出 shuffle run，看 combined AUROC 真的回落。**
- ❌ codex round-3 PASS_WITH_NITS 之后立刻进 main run，没修 nits。结果 nits 后来成了 round-4 NOT_PASS 的根因（STALE filter bug）。**Nits 不修不准进 multi-seed。**
- ❌ "review 只在对话里说一遍，不落盘"。信息会丢，下次接手的 agent 没法追溯。**强制落到 `_reviews/<date>__<scope>__round<N>.md`。**
- ❌ Independent reviewer 写 "Claude (sonnet)" 单字段。看不出第二意见来源。**必须拆 Primary / Independent 两字段。**

---

## 9. paper 故事候选（写完 D1+D2 后能讲的版本）

> 现有 KG-based DDI cold-start backbone（EmerGNN）在 S2 上停在 ~0.746 AUROC。我们识别出两类 evidence 在 backbone 中缺失：(i) 两药共享的 KG meeting-node 的 type-aware identity（i2）；(ii) 从大语言模型蒸馏出的 PK/PD-flavored 机制字段（i4）。我们把这两类 evidence 注入 path-flow backbone——meeting-node 通过 propagation 时的 attention bias 直接进入 message function；LLM 字段被建模为新增的 KG edge type，与 path-flow 共享同一传播机制。S2 AUROC 从 0.746 提升到 0.79（+4.4pt），shuffle / random / shuf-text / shuf-pair 四种 control 全部回落到 baseline，证明 lift 来自 evidence semantics 而非 capacity。同时我们补完了多模态对齐与 pair-conditional routing 两条 architectural prior，给后续 cold-start 工作提供 evidence-form analysis 的方法学框架。

要点 = (a) backbone 真的改了；(b) 信号源来自 LLM substrate + KG meeting-node，是新的；(c) 控制实验严密；(d) 卖点（alignment + routing）落地。

---

## 10. 接手 checklist

新窗口接手时按这个顺序走。

- [ ] Read 这份文档完整
- [ ] Read `Notes/Log/paper_writeup.md`（已写的 paper material）
- [ ] Read `Notes/Settings/Insights/i2.md` 和 `i4.md`
- [ ] Pick 一个方向（建议 D1）
- [ ] 写设计草案 `Notes/Log/d1_llm_edge_design.md`（或对应 dN 文件）
- [ ] Codex review 1 轮，critical 修掉
- [ ] 新建 `Code/my_code/models/screen_s2_v3_multimodal/v3_llm_edge_trainer.py` 等新文件，**不改任何已有 trainer**
- [ ] 跑 seed42 main，verify combined ≥ 0.775 才继续
- [ ] 跑 shuffle / random control
- [ ] 跑 seed43 / seed44 multi-seed
- [ ] 把 verified 数字追加进 `Notes/Log/paper_writeup.md` 的 Section 3
- [ ] 在 `Notes/Log/round4_<direction>_results.md` 落盘控制实验 + ablation 完整结果

**任何 deviate from 这份 plan 之前，先在 `Notes/Log/` 新开一个 .md 记录原因，不要 silent 改方向。**
