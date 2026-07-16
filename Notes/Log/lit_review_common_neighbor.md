# Lit Review — Common-Neighbor GNN Link Prediction (PMP 文献定位 2026-05-30)

**目的**. 系统搜近 5 年 common-neighbor + GNN 链路预测的文献，verify PMP 跟最近 SOTA 的真实 architectural diff，避免重复造轮子。所有引用都附 verified source URL，**没 verify 的部分明确标 "印象 / 需 verify"**。

---

## 1. 直接对手 paper（PMP 必须 differentiate 的）

### 1.1 NCN / NCNC (ICLR 2024, GraphPKU)

- **Source**. [arxiv 2302.00890](https://arxiv.org/abs/2302.00890), [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/3efb4bdc6bfe13e1ff95b4407c37961d-Paper-Conference.pdf), [OpenReview](https://openreview.net/forum?id=sNFLN3itAd), [GitHub GraphPKU/NeuralCommonNeighbor](https://github.com/GraphPKU/NeuralCommonNeighbor)
- **Scoring function (verified from arxiv html v4)**.
  ```
  NCN(i, j, A, X) = MPNN(i, A, X) ⊙ MPNN(j, A, X)  ||  Σ_{u ∈ N(i) ∩ N(j)} MPNN(u, A, X)
  ```
  其中 `⊙` 是 Hadamard product，`||` 是 concatenation。**端点 embedding 通过 Hadamard 进入 score，共邻 sum pool concat 在后面**。
- **Framework**. MPNN-then-SF（MPNN 跑一次，structural feature 指导 pooling）。
- **Datasets**. Cora / Citeseer / Pubmed / Collab / PPA / **DDI** / Citation2（OGB datasets）。
- **Setting**. **Transductive only**（random split），不是 cold-start。
- **Performance claim**. NCN/NCNC outperform BUDDY 平均 5%。
- **PMP 的差异**. PMP 完全剥离端点 embedding（不要 Hadamard 项），cold-start S2 setting 评估。**这是 paper 必须 head-to-head 的最直接对手**。

### 1.2 NCNC 完成机制

- **完成 motivation**. KG incompleteness 减少 common neighbors 并 induce distribution shift。
- **机制**. 先用 link prediction 模型补全 common neighbor structure，再用 NCN 在补全图上。
- **PMP 的差异**. C3 mol bridge 做类似的事但用 **molecular similarity** 而不是 learned link prediction 来补共邻。这是不同的补全 source（domain knowledge vs learned）。

### 1.3 TNCN — Temporal NCN (NeurIPS 2024)

- **Source**. [arxiv 2406.07926](https://arxiv.org/abs/2406.07926), [OpenReview](https://openreview.net/forum?id=XLt0eudh8t)
- **Innovation**. multi-hop common neighbors（不只 1-hop）+ temporal dictionary。
- **Inductive evaluation**. TNCN 跑 transductive + inductive setting，6/7 transductive SOTA, 3 inductive SOTA。**注意**. inductive 是动态图 future node 概念，不是 cold-start drug。
- **PMP 的差异**. multi-hop CN 这块我们的 PMP（MNAH 已经做 1-hop + 2-hop）跟它对应。Cold-start drug 这块 TNCN 不 cover。

### 1.4 MPLP — Pure Message Passing (NeurIPS 2024)

- **Source**. [arxiv 2309.00976](https://arxiv.org/abs/2309.00976), [OpenReview](https://openreview.net/forum?id=QGR5IeMNDF), [GitHub Barcavin/efficient-node-labelling](https://github.com/Barcavin/efficient-node-labelling)
- **核心 Theorem 3.2**. 用 quasi-orthogonal (QO) input vectors + sum aggregation，1-layer MPNN 是 CN 的 unbiased estimator. `E(h_u · h_v) = CN(u, v)`。
- **Method**. QO vectors 从 hypercube 采样，hub nodes 用 one-hot，距离编码加多 hop，norm 用 learned MLP 实现 AA / RA 变体。
- **Performance**. MPLP/MPLP+ 在 OGB Collab MRR 6.79、Citation2 MRR 23.11 SOTA，对 SEAL / ELPH / NCNC 2–10% Hits@50 提升。
- **对 PMP 的挑战**. **MPLP 证明 pure MPNN with QO init 已经能隐式估计 CN，不需要显式架构**。这削弱了 PMP 的 "explicit common neighbor pooling" 作为 architectural novelty。
- **但 PMP 还有的差异**. MPLP 用 endpoint vector dot product 算 score（CN estimate 跟 endpoint vec 绑定），PMP 把端点完全剥离；MPLP 是 transductive，PMP 是 S2 cold-start。

### 1.5 Neo-GNN (NeurIPS 2022)

- **Source**. [arxiv 2206.04216](https://arxiv.org/abs/2206.04216)
- **核心**. learnable structural feature generator 从邻接矩阵学，可以 generate AA-equivalent feature（reciprocal of log）或其他 heuristic equivalent。
- **PMP 的差异**. Neo-GNN 是 SF-and-MPNN 架构（SF 跟 MPNN 平行独立），PMP 把 SF 提升为 score function 唯一主体。

### 1.6 BUDDY / ELPH (ICLR 2023 oral)

- **Source**. [arxiv 2209.15486](https://arxiv.org/abs/2209.15486), [ICLR proceedings](https://iclr.cc/virtual/2023/oral/12595), [GitHub melifluos/subgraph-sketching](https://github.com/melifluos/subgraph-sketching)
- **核心**. 用 subgraph sketching（hash-based summary）让 MPNN 隐式带上 subgraph info，避免 SEAL 每条边 re-extract subgraph 的 cost。BUDDY 是 ELPH 的 scalable 版。
- **PMP 的差异**. BUDDY/ELPH 是 efficient SEAL，仍然 subgraph-encode 的 mindset。PMP 是显式 common neighbor pool，跟 BUDDY 是不同 design philosophy。

### 1.7 HL-GNN (KDD 2024)

- **Source**. [arxiv 2406.07979](https://arxiv.org/abs/2406.07979)
- **核心**. local 和 global heuristics 用 adjacency matrix multiplications 统一表达，intra-layer + inter-layer propagation 形成 unified framework。
- **Status**. abstract 没给具体 scoring function 是否端点进 score（**需要 verify full PDF**）。
- **数据集**. Planetoid / Amazon / OGB datasets。
- **PMP 的差异**. HL-GNN 是统一各 heuristic，PMP 是 cold-start specific architectural primitive。两者关系待 verify。

### 1.8 HGNN-CNA (arxiv 2025-04)

- **Source**. [arxiv 2504.18758](https://arxiv.org/pdf/2504.18758)
- **核心（从 search description）**. 多 hop common neighbor correlation score，fuse 进 message-passing。
- **Status**. PDF fetch 失败，**精确机制待 verify**。
- **PMP 的差异**. HGNN-CNA 把 CN 信号 fuse 进 message passing（modify backbone），PMP 把 score function 替换成 mediator pool（modify scoring）。两者 attack point 不同。

### 1.9 PROXI (arxiv 2410.01802, 2024)

- **Source**. [arxiv 2410.01802](https://arxiv.org/abs/2410.01802)
- **核心**. 20 个手工 structural + domain indices + XGBoost 做 binary classification，在 12 个 benchmark 上 outperform 或 competitive 跟 GNN SOTA。
- **核心 finding**. **"Using individual node embeddings for pair prediction is not practically viable"** —— 这恰好支持 PMP 的"端点 embedding 不进 score"的设计原则。
- **PMP 的差异**. PROXI 是 hand-crafted feature + XGBoost（不是 end-to-end GNN），PMP 是 end-to-end neural mediator pooling。PMP 可以 cite PROXI 作为"端点 embedding 不必要"的独立 evidence。

### 1.10 "Can GNNs Learn Link Heuristics?" (arxiv 2411.14711, 2024)

- **Source**. [arxiv 2411.14711](https://arxiv.org/abs/2411.14711)
- **核心 finding**. **"GNNs cannot effectively learn structural information related to the number of common neighbors between two nodes, primarily due to the nature of set-based pooling of the neighborhood aggregation scheme."**
- **PMP 的用法**. paper 的 motivation 段可以 cite 这一篇作为"standard GNN 学不到 CN，所以需要显式架构"的 evidence。
- **注意**. 这跟 MPLP 的结论 partially conflict（MPLP 说 QO init + sum aggregation 能学），用 PMP cite 时要 nuanced 处理。

---

## 2. Pair-based representation theory（PMP 的 theoretical anchor）

### 2.1 SEAL (NeurIPS 2018)

- **Source**. [arxiv 1802.09691](https://arxiv.org/pdf/1802.09691), [Muhan Zhang's homepage](https://muhanzhang.github.io/)
- **核心**. 每对 (a, b) extract enclosing k-hop subgraph + DRNL labeling + GNN encode。
- **PMP 的差异**. SEAL 仍然 GNN-encode 整个 subgraph 包含端点，PMP 只 pool common neighbor。

### 2.2 Labeling Trick (NeurIPS 2021)

- **Source**. [arxiv 2010.16103](https://arxiv.org/pdf/2010.16103), [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2021/hash/4be49c79f233b4f4070794825c323733-Abstract.html)
- **作者**. Zhang, Li, Salha, Wang, Jin
- **核心 theorem**. labeling trick + GNN 严格强于无 labeling 的 GNN。给 pair (i, j) 把 i, j 标 special color，让 GNN message passing pair-aware。
- **PMP 的 theoretical anchor**. **PMP paper 必须 cite 这一篇作为 theoretical foundation**，论证 pair-induced representation learning 是 link prediction 的必要操作。

### 2.3 Distance Encoding (NeurIPS 2020)

- **Source**. Li, Wang, Wang, Leskovec — 需要 verify 精确 arxiv id（印象是 2009.00142 或类似）
- **核心**. 给每个节点 v 加 feature `(d(v, a), d(v, b))`，pair-conditional 距离。
- **PMP 的关系**. "共邻 = d(v, a) = 1 AND d(v, b) = 1" 是 distance encoding 的特殊形式。PMP 是 DE 的 typed + attentioned + endpoint-stripped 版本。

### 2.4 GLASS — GNN with Labeling Trick for Subgraph

- **Source**. [OpenReview](https://openreview.net/pdf?id=XLxhEjKNbXj)
- **核心**. labeling trick + subgraph representation learning。
- **Status**. 没深读，**待 verify 跟 PMP 的关系**。

---

## 3. KG inductive link prediction（DDI cold-start 直系祖宗）

### 3.1 EmerGNN (Nature Computational Science 2023)

- **Source**. [Nature paper](https://www.nature.com/articles/s43588-023-00558-4), [GitHub LARS-research/EmerGNN](https://github.com/LARS-research/EmerGNN), [arxiv 2311.09261](https://arxiv.org/pdf/2311.09261)
- **核心**. flow-based GNN，path-flow length=3，shuffle_train(mode="S2") cold simulation。
- **S0/S1/S2 setting**. S0 = existing-existing, S1 = emerging-existing, S2 = emerging-emerging。
- **PMP 的关系**. PMP 直接对标，0.7458 是 anchor。

### 3.2 SumGNN (2020)

- **Source**. [arxiv 2010.01450](https://arxiv.org/pdf/2010.01450), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10060701/)
- **核心**. KG summarization → local subgraph around drug pair → GNN encode。
- **Performance**. 比 KGNN 高 4% (DrugBank) / 2% (TWOSIDES)。
- **PMP 的关系**. SumGNN subgraph-based 但是仍然端点 readout。

### 3.3 GraIL (ICML 2020)

- **Source**. Teru, Denis, Hamilton — 需 verify 精确 venue 和 arxiv id
- **核心**. KG inductive link prediction，enclosing subgraph + GNN。
- **PMP 的关系**. GraIL 是 inductive KG link prediction 的代表，PMP 是 cold-start DDI 的代表。

### 3.4 NBFNet (NeurIPS 2021)

- **Source**. [arxiv 2312.10293 提到](https://arxiv.org/pdf/2312.10293) — Zhu, Zhang, Xhonneux, Tang
- **核心**. Neural Bellman-Ford，query-conditional path GNN。
- **PMP 的关系**. NBFNet 是 path-based query-conditional（跟 EmerGNN 类似 family），PMP 是 subgraph-based pair-conditional。

### 3.5 RED-GNN (WWW 2022)

- **Source**. Zhang & Yao — 需 verify
- **核心**. relation embedding for path reasoning on KG。
- **PMP 的关系**. path-based 路线，跟 PMP 正交。

### 3.6 ULTRA (ICLR 2024)

- **Source**. [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/85dd09d356ca561169b2c03e43cf305e-Paper-Conference.pdf)
- **核心**. labeling trick + zero-shot generalization to any KG without graph-specific entity/relation embeddings。
- **PMP 的关系**. ULTRA 是 fully zero-shot，PMP 是 cold-start drug 但 KG 是固定的。Different problem。

### 3.7 GAINET (2025)

- **Source**. [Nature Scientific Reports 2025](https://www.nature.com/articles/s41598-025-12936-1)（fetch 时遇到 redirect，需 retry）
- **核心**. GAT + co-attention + KG for DDI prediction。
- **Status**. 精确机制 + 数字 **待 verify**，但应作为 2025 latest baseline 考虑。

### 3.8 DDI cold-start systematic review

- **Source**. [Computers in Biology and Medicine 2025 (S0010482525004731)](https://www.sciencedirect.com/science/article/abs/pii/S0010482525004731)
- **核心**. systematic review of molecular structures, knowledge graphs, and cold-start scenario in DDI prediction.
- **Status**. abstract fetch 失败（403），**必须想办法拿全文 PDF**，对 paper related work section 是 reference 钥匙。

### 3.9 Other DDI cold-start

- DTKGIN — drug-target interaction with intent graph，[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1046202324001002)。
- DDI cold-start DTI multilevel — [arxiv 2510.04126](https://arxiv.org/pdf/2510.04126)（暂未读）。

---

## 4. PMP 真实 novelty 重评估（诚实版）

经过这次 lit review，PMP 的 novelty edge **必须明确**。

### 4.1 已经被 prior work 覆盖（不是 PMP 独有）

| Claim | 已被谁覆盖 |
|---|---|
| "GNN-化 common neighbor" | NCN / NCNC / Neo-GNN / MPLP / HGNN-CNA |
| "Pair-based representation theory" | Labeling Trick / SEAL / Distance Encoding |
| "Multi-hop common neighbor" | HGNN-CNA / TNCN |
| "Hand-crafted heuristic + ML" | PROXI |
| "Common neighbor 在 link prediction 重要" | 所有 above |
| "MPNN 学不动 CN" 这个 motivation | "Can GNNs Learn Link Heuristics?" / PROXI |

### 4.2 PMP unique（必须在 paper 里 explicit defend）

| Claim | 为什么 unique |
|---|---|
| **端点 embedding 架构性剥离** | NCN 仍保 Hadamard endpoint，MPLP 用 endpoint dot product，没人 fully strip endpoint。PROXI 做了但不是 end-to-end GNN |
| **Cold-start S2 specific evaluation** | NCN/MPLP/Neo-GNN/HL-GNN/HGNN-CNA 全部是 transductive。Cold-start drug-drug interaction 这个 setting 上 common-neighbor pooling 还没系统评估 |
| **Typed mediator pooling** | DDI KG 是 heterogeneous (Drug, Protein, Pathway, ...), 现有 common neighbor work 主要 homogeneous graph |
| **LLM-enriched mediator universe (C2)** | DDI 文献里没人把 LLM 蒸馏字段当作新 KG edge type，PMP 在 enriched KG 上 pool |
| **Molecular-bridged neighbor expansion (C3)** | 通过分子相似度扩 mediator candidate set，cold pair fallback —— 这是新的 mechanism |
| **MNAH 22-d typed common-neighbor count → +3pt verified evidence** | PMP 直接 cite 这个 verified evidence 当 motivation，其他 paper 没在 DDI cold-start 上做过这种 controlled ablation |

### 4.3 必须做的 head-to-head（决定 paper 成败）

| Baseline | 为什么必须比 | 当前 status |
|---|---|---|
| **NCN / NCNC (ICLR 2024)** | 直接对手，端点剥离 vs Hadamard endpoint 的核心差异在这里证 | 必须新建 baseline |
| **MPLP (NeurIPS 2024)** | 证明 PMP 在 cold-start 比 QO-MPNN 强 | 必须新建 baseline |
| **Neo-GNN (NeurIPS 2022)** | learnable AA-equivalent，比 PMP 弱版 | 必须新建 baseline |
| **BUDDY (ICLR 2023)** | subgraph sketching 高效版 SEAL | 应该建 baseline |
| **SEAL** | pair-induced subgraph 古典 | 已有 reproduction infra 可建 |
| **EmerGNN** | DDI cold-start anchor | 已 verified 0.7458 |
| **SumGNN / KGNN** | DDI KG 系列 baseline | 已 verified or partial |
| **TIGER (AAAI 2024)** | dual-channel DDI 对标 paper | 已有 reproduction |
| **HL-GNN (KDD 2024)** | unified heuristic learning | 可选 |
| **GAINET (2025)** | latest DDI baseline | 可选，看时间 |

### 4.4 调整后的 paper story（基于 lit review）

> Cold-start drug-drug interaction prediction on knowledge graphs requires evidence that survives the absence of endpoint information at test time. Recent work has shown that explicit common-neighbor signal helps link prediction (NCN [ICLR 2024], MPLP [NeurIPS 2024], Neo-GNN [NeurIPS 2022], HGNN-CNA [2025], "Can GNNs Learn Link Heuristics?" [2024], PROXI [2024]), yet these methods uniformly retain endpoint embeddings as a primary signal in the scoring function and evaluate only under transductive settings. We propose **Pair-Mediator Pooling (PMP)**, an architectural primitive that fully removes endpoint embeddings from the scoring function and computes pair representations exclusively from typed common-neighbor mediators. PMP is specialized for the cold-start regime where endpoint embeddings are under-trained and thus unreliable.
>
> Three orthogonal contributions sharpen PMP for DDI cold-start.
>
> (C1) Typed common-neighbor attention pooling. PMP subsumes Adamic-Adar [2003], Common Neighbors [Newman 2001], and the MNAH typed common-neighbor count (+3.0 pt verified lift over EmerGNN [Nature Comput Sci 2023]) as degenerate cases of a unified attention-pooled formulation over typed mediator embeddings.
>
> (C2) LLM-enriched mediator universe. We distill intrinsic pharmacology via an LLM under strict leakage sanitization and materialize the typed fields as new node types and edge types in the KG. PMP applies on the enriched KG with no further changes.
>
> (C3) Molecular-bridged mediator expansion. When the KG common-neighbor set is sparse, molecular nearest neighbors contribute their KG neighbors as virtual mediators. Molecular embeddings never enter the scoring function directly, ensuring the expansion is "no harm" on KG-rich pairs.
>
> Each component is independently ablated against capacity-controlled controls. The full method achieves [target ≥ 0.80] S2 AUROC on DrugBank cold-start, outperforming NCN, MPLP, Neo-GNN, BUDDY, SEAL, EmerGNN, SumGNN, KGNN, and TIGER under the cold-start setting where they were not originally designed.

---

## 5. 必须补的 lit verify（接手时优先完成）

| 项 | 工具 | 优先级 |
|---|---|---|
| Distance Encoding 精确 arxiv id 和作者 | WebSearch | 中 |
| HGNN-CNA (2504.18758) PDF 全文 | WebFetch retry / 下载 PDF 本地 parse | 高（最接近的 multi-hop CN 对手） |
| DDI cold-start systematic review S0010482525004731 全文 | 想办法绕过 paywall（学校 IP / institutional access） | 高（paper related work 的 key reference） |
| GAINET (Nature Scientific Reports 2025) 全文 | WebFetch redirect retry | 中（latest DDI baseline） |
| HL-GNN (KDD 2024) 完整 scoring function | 直接读 arxiv PDF | 中 |
| GraIL / NBFNet / RED-GNN / GLASS 精确出处 | WebSearch | 中 |
| TIGER (AAAI 2024) Su et al. paper 本身 | 项目已有 reproduction，复习 paper PDF | 已 cached at `Code/reproductions/TIGER/_paper-and-GitHub/` |

---

## 6. 接手 checklist（lit review 这一步）

新窗口接手时按顺序.

- [ ] Read 本文档完整
- [ ] 优先 verify HGNN-CNA / DDI systematic review / GAINET 三个最新 paper
- [ ] 把 NCN / MPLP / Neo-GNN / BUDDY / SEAL 加入 baseline TODO list（每个建 `Code/baseline/<name>/` 文件夹按规范）
- [ ] 在 `Code/baseline/<name>/_reviews/` 建 review checklist，先确认这些 baseline 的 paper-faithful reproduction
- [ ] 在 `paper-writeup.md` 添加 related work draft section based on 本文档 section 1+2+3
- [ ] 任何超出本文档的 lit claim 必须先 WebSearch verify，不许凭印象写 paper
- [ ] 跑 baselines 时优先在 DrugBank 800-drug S0/S1/S2 全部 setting 上对比

---

## 7. 一句话总结

> PMP 在 2026 年的 common-neighbor + GNN link prediction 文献版图里，**核心 architectural novelty 是"端点 embedding 完全从 score function 剥离"**，这一点是 NCN / MPLP / Neo-GNN / BUDDY 都没明确做的 design choice。但**单独这一点不足以撑 AAAI paper**，PMP 必须靠 (a) **cold-start S2 specific evaluation 优势**、(b) **C2 LLM enrichment 的 DDI unique signal**、(c) **C3 mol bridge 的 no-harm 设计**三者叠加才有完整的 paper story。所有共邻 family baseline（NCN, MPLP, Neo-GNN, BUDDY）必须 head-to-head 跑在我们的 S2 cold-start setting 上，**这是 paper 能否成立的硬底线**。
