# NBFNet v3.0 Design Spec — Joint Binary + Multi-Class DDI

**Date**. 2026-06-05
**Predecessor**. v1.7 (paper-faithful single-query NBFNet, binary-only, codex Round 5 GO)
**Series**. v3.x = "3-series" joint dual-task models. v3.0 is MVP (no ULTRA), v3.1+ adds mechanism meta-graph.
**Theoretical basis**. User's "Proposition B + soft-OR" framework (see top of this file).

---

## 0. Core thesis

> **Binary cls 不是 multi-cls 的前置 gate, 是所有 mechanism evidence 的析取 (soft-OR) 汇总.**
> $P(\text{DDI exists} \mid u, v) = P\left(\bigcup_m \text{path of mechanism } m \text{ exists}\right)$

这是 **multi → binary** 的信息流方向 (反对 binary → multi 的 gate 方向). 理由是 gate 方向有 "假阴性一票否决" 的单向误差传播漏洞, 不是真正的相辅相成.

两个任务通过**共享 path 传播**实现真正的相辅相成.
- multi-cls 的梯度反传给共享 NBFNet 参数 → 帮助 binary 学得更好
- binary 的梯度通过 ⊕_m soft-OR 流回每个 mechanism 的 BF → 帮助 multi-cls 学得更好

---

## 1. Architecture

### 1.1 Multi-query NBFNet propagation (shared params)

对每个 mechanism $m \in \{1, ..., K\}$, 学一个独立 query embedding $q_m \in \mathbb{R}^d$.

**Vocabulary $K = 80$ (locked 2026-06-05)**. 见 §1.4 + 数据 verify 报告:
A_strict (present in TRAIN of all 3 seeds 42/43/44) $\cap$ B (top-86 by global freq across all
splits and seeds) $= 80$ classes, 覆盖 97.3-98.3% 正例. 长尾 135 类丢弃 (CE 梯度太噪),
top-86 中 6 类因为某个 seed train 缺失而被踢出 (CE 在该 seed 上 undefined).

向量保存在 `Code/data/_cache/ddi_type_map_v3_0.json`, 通过 `MultiClsVocab.load_default()` 读取.



NBFNet 传播参数 $(W_r^{(t)}, b_r^{(t)}, \text{PNA}^{(t)})$ **跨所有 query 共享** (来自 v1.7 NBFLayer / PNAAggregator, 不修改).

对每个 pair $(a, b)$, 对每个 mechanism $m$, 做一次 BF.
$$
h_v^{(m), (t)} = \text{AGGREGATE}^{(t)}\!\left(\bigcup_{(x,r,v) \in E(v)} \text{DistMult}(h_x^{(m), (t-1)}, W_r^{(t)} q_m + b_r^{(t)}) \cup \{h_v^{(m), (0)}\}\right)
$$
$$
h_v^{(m), (0)} = \mathbf{1}[v = a] \cdot q_m
$$

L 层后, 在 $b$ 处 readout 得到 $h_q^{(m)}(a, b) := h_b^{(m), (L)}$.

**对称化**. DDI 对称 → 反向再做一次 (source=b), readout at a → $h_q^{(m)}(b, a)$.
$$
h_q^{(m), \text{sym}} = h_q^{(m)}(a, b) + h_q^{(m)}(b, a) \in \mathbb{R}^d
$$

得到 stacked tensor $H \in \mathbb{R}^{N_m \times d}$, 每行是一个 mechanism 的 symmetric evidence.

**为什么 query embedding $q_m$ 独立但传播参数共享**.
- $q_m$ 独立 → 每个 mechanism 有自己 "想 propagate 什么信号" 的语义
- 传播参数共享 → 跨机制学到 KG path 上的通用归纳偏置, 不同 mechanism 互相 regularize

### 1.2 Dual readout head

**Multi-cls head** (per-mechanism scoring + softmax).
$$
s_m^{\text{multi}} = \text{MLP}_{\text{multi}}([h_q^{(m), \text{sym}}; q_m]) \in \mathbb{R}
$$
$$
p(m \mid u, v) = \text{softmax}_m(s_m^{\text{multi}})
$$

注意 MLP_multi **跨 mechanism 共享**, 输入 concat $q_m$ 让 head 知道 "当前评分的是哪个 mechanism" — 这种 hyper-network 式设计也方便后续 v3.1 把 $q_m$ 换成 ULTRA 算出的 relation-conditional 表示.

**Binary head** (soft-OR over mechanisms).
$$
z^{\text{binary}} = \underbrace{\text{LogSumExp}_m(H)}_{\text{soft-max along mechanism axis} \in \mathbb{R}^d}
$$
$$
\text{logit}^{\text{binary}} = \text{MLP}_{\text{binary}}([z^{\text{binary}}; \bar{q}]) \in \mathbb{R}
$$

其中 $\bar{q} = \frac{1}{N_m} \sum_m q_m$ 是全局 query summary (类比 v1.7 concat query, 保持 head 结构 sane).

**LogSumExp 作为软 OR 的数学根基**.
- $\text{LSE}(\{x_m\}) = \log \sum_m e^{x_m}$, 是 $\max$ 的光滑近似
- 对应 "存在某 mechanism 的 evidence 强 → binary 高分"
- 梯度光滑, 训练稳定 (vs hard max 的 sparse gradient)
- 比 mean 强 (mean 会被弱信号稀释), 比 max 软 (避免单点 explode)

**可选 ablation aggregator** (v3.0 默认 LogSumExp, 后续可测).
| Aggregator | 公式 | 直觉 |
|---|---|---|
| LogSumExp (default) | $\log \sum_m \exp(H_m)$ | soft OR, paper-justified |
| Max | $\max_m H_m$ | hard OR, sparse |
| Mean | $\frac{1}{N_m} \sum_m H_m$ | average, 弱信号会稀释 |
| Attention | $\sum_m \alpha_m H_m, \alpha = \text{softmax}(w^\top H)$ | learnable, 引入参数 |

### 1.4 Multi-cls vocabulary loader (locked 2026-06-05)

**File**. `Code/data/_cache/ddi_type_map_v3_0.json`
**API**. `from my_code.models.nbfnet_v3_0 import MultiClsVocab`

```python
v = MultiClsVocab.load_default()           # auto-locates project root
assert v.K == 80
# row-level lookup, returns -1 for out-of-vocab OR negatives (no ddi_type)
multi_label = v.lookup_type_or_negative(row["ddi_type"])
# pair-level lookup (when reading via canonical "a|b" pair table)
multi_label = v.lookup_pair_via_table(a_id, b_id, pair_table)
```

**Out-of-vocab semantics**. 长尾正例 (~2.4% of positives) 在 v3.0 训练中:
- Multi-cls CE 上 mask out (NBFNetJointDDI.joint_loss 已支持 multi_label = -1)
- Binary BCE 上仍然算 label=1 (它们 IS a DDI, only mechanism unknown to v3.0 vocab)
- 评测时 macro-F1 在 K=80 vocab 上算; 超出 vocab 的也单独切出来报告

### 1.5 Per-seed train coverage (verified)

| Seed | Vocab classes in train | OOV positive rows in train |
|---|---|---|
| seed42 | 80 / 80 ✓ | 4,670 / 204,469 = 2.28% |
| seed43 | 80 / 80 ✓ | 4,107 / 202,773 = 2.03% |
| seed44 | 80 / 80 ✓ | 3,282 / 197,196 = 1.66% |

每个 seed 的 train, val_s2, test_s2 中 vocab 覆盖率均 $\geq 97.3\%$, multi-cls CE 信号充足.

### 1.3 Forward shape summary

```
Input. (drug_a, drug_b, KG edges)
       │
       ▼
For m in 1..N_m:
    BF(source=a, query=q_m) → h^(m)_from_a ∈ (n_nodes, d)
    BF(source=b, query=q_m) → h^(m)_from_b ∈ (n_nodes, d)
    h_q^(m)_ab = h^(m)_from_a[b]
    h_q^(m)_ba = h^(m)_from_b[a]
    h_q^(m)_sym = h_q^(m)_ab + h_q^(m)_ba           ∈ (d,)
       │
       ▼
Stack. H ∈ (N_m, d)
       │
       ├─ Multi-cls: MLP_multi([H; Q]) → (N_m,) logits → softmax → p(m|u,v)
       │
       └─ Binary: LogSumExp(H) → (d,) → MLP_binary([z; q̄]) → scalar logit
```

---

## 2. Loss formulation

**Per pair $(a, b)$ 的 loss**.

- 如果是 positive pair (label $y_{\text{DDI}} = 1$, 有真实 mechanism $m^*$):
$$
\mathcal{L}_{\text{pair}}^{+} = \underbrace{\text{CE}(p(m \mid u, v), m^*)}_{\text{multi-cls}} + \lambda \cdot \underbrace{\text{BCE}(\sigma(\text{logit}^{\text{binary}}), 1)}_{\text{binary}}
$$

- 如果是 negative pair (label $y_{\text{DDI}} = 0$, 无 mechanism label):
$$
\mathcal{L}_{\text{pair}}^{-} = \lambda \cdot \text{BCE}(\sigma(\text{logit}^{\text{binary}}), 0)
$$

**Batch loss** 是上面 per-pair loss 在 batch 内 mean.

**Hyperparameter**. $\lambda \in \{0.1, 0.3, 1.0, 3.0\}$ sweep, 默认 $\lambda = 1.0$.

**为什么 multi-cls loss 只在 positive 上**. negative pair 不存在 "正确 mechanism", multi-cls label 无定义. CE 在 negative 上没意义. 但 binary loss 在 positive + negative 都有, 这才保证 binary 是真正的 OR 判断.

**梯度流可视化**.
```
CE loss (multi-cls, on positives only)
  ↓ ∂/∂H, ∂/∂q_m, ∂/∂(W_r, b_r, PNA)
  ↓
shared NBFNet params  ◀──── 多任务共享 here
  ↑
  ↑ ∂/∂H, ∂/∂q_m, ∂/∂(W_r, b_r, PNA)
BCE loss (binary, on all)
  ↑ ∂/∂z_binary
  ↑ ∂/∂LogSumExp(H) — 每个 H_m 都收到 softmax-weighted 梯度
```

**关键**. shared params 同时受两个 loss 梯度更新 → 这是 "相辅相成" 的数学根基.

---

## 3. MVP scope (v3.0 = Step 1-3 of user's plan)

### Step 1 ✓ MVP includes
- 固定 mechanism 集合 $M$ (用 DrugBank DDI type vocabulary, $K = 80$, **locked 2026-06-05**, 见 §1.4)
- 每个 mechanism 一个独立 learnable $q_m$
- 共享 NBFNet 传播

### Step 2 ✓ MVP includes
- Dual readout (multi-cls softmax + binary LogSumExp)
- 联合 loss with $\lambda$

### Step 3 ✓ MVP focus = 实验验证
**三组对照**.
- **(a) binary-only**: 设 K = 1 (单 query, 等同 v1.7), 只用 binary loss
- **(b) multi-only**: 多 query, 只用 multi-cls CE loss (设 $\lambda = 0$)
- **(c) joint**: 多 query + 双任务 loss (设 $\lambda = 1.0$)

**成功判定**.
- (c) binary AUROC > (a) binary AUROC, AND
- (c) macro-F1 > (b) macro-F1

如果只有一个变好, 是 trade-off 不是 synergy, 重新设计 (调 $\lambda$ / 换 aggregator / etc.).

### Step 4 ✗ MVP NOT include (留给 v3.1+)
- ULTRA-style relation-conditional $q_m = \text{GNN}_r(G_r \mid m)$
- Bio-semantic meta-graph edges (CYP450 / pathway overlap)
- Zero-shot to held-out mechanisms

---

## 4. Implementation plan

### 4.1 File layout

```
Code/my_code/models/nbfnet_v3_0/
├── __init__.py                              # exports NBFNetJointDDI
├── nbfnet_v3_model.py                       # multi-query model + dual head
└── nbfnet_v3_trainer.py                     # (待写, in Task #6)

Code/scripts/
└── run_nbfnet_v3.py                         # (待写, in Task #6)

Notes/Log/
└── nbfnet_v3_0_design.md                    # this file
```

### 4.2 Reuse from v1.7 (no modification)

```python
from my_code.models.nbfnet_v1_7.nbfnet_model import NBFLayer, PNAAggregator
```

- `NBFLayer`. single-query layer with per-(layer, relation) $W_r^{(t)}, b_r^{(t)}$
- `PNAAggregator`. PNA with 4 aggs × 3 scalers

v3.0 在外层做 multi-query loop (initial), 不动 NBFLayer 内部 contract.

### 4.3 New class signature

```python
class NBFNetJointDDI(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        n_base_rel: int,
        n_mechanisms: int,               # N_m, e.g. 86 for DrugBank
        d: int = 32,                     # paper default
        n_layers: int = 6,               # paper default L
        mlp_hidden: int = 64,            # paper default head hidden
        ddi_rel_id: int | None = None,   # for query-edge masking
        binary_aggregator: str = "logsumexp",  # logsumexp/max/mean/attention
    ): ...
    
    def encode_all_mechanisms_from_source(
        self,
        source: int,
        aug_src: torch.Tensor, aug_dst: torch.Tensor, aug_rel: torch.Tensor,
        edge_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns. (N_m, n_nodes, d) — h^(m)_v for all (m, v).
        Initial impl: Python loop over mechanisms, calling NBFLayer per-(m, layer).
        Future v3.0.1+: vectorized via stacked queries + batched scatter.
        """
        ...
    
    def score_pair_dual(
        self,
        drug_a: int, drug_b: int,
        edge_src, edge_dst, edge_rel,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns. (multi_logits, binary_logit)
            multi_logits: (N_m,) per-mechanism raw logits (apply softmax+CE externally)
            binary_logit: scalar, raw logit (apply sigmoid+BCE externally)
        """
        ...
```

### 4.4 Computational cost note

**Per pair, per training step**.
- v1.7 baseline. 2 BF passes (forward + reverse) × L=6 layers × E edges
- v3.0 joint. $2 \times N_m$ BF passes (each mechanism × 2 directions) × L × E

With $N_m = 86$: **v3.0 是 v1.7 的 86× 计算量**. 这是 MVP 第一版的代价, 后续优化方向.

**优化路径** (v3.0.1+):
1. **Vectorize over queries**. stacked $Q \in (N_m, d)$, batched scatter (~10x speedup, complex impl)
2. **Per-source amortization with shared cache**. batch 内同 source 复用 N_m × 1 次 BF (而不是 N_m × n_pairs 次)
3. **Mechanism subsampling**. 训练时只对每个 batch sample $K < N_m$ 个 mechanism (mini-batch over mechanisms)

第一版用 Python loop, 跑通后再优化.

---

## 5. Risks and open questions

### 5.1 Risks
1. **$K = 80$ 过大 → 计算爆炸**. 若 $N_m = 86$, single-pair forward 已经是 v1.7 的 86×, batch 训练可能不可行
   - **Mitigation**. MVP 先用 mechanism subsampling, sample K=8 mechanism per batch (binary loss aggregate over sampled K)
2. **Multi-cls 在 cold-start 上信号弱**. 已知 macro-F1 baseline 在 0.02 量级, 梯度信号差
   - **Mitigation**. 把 multi-cls 当 auxiliary regularization (per MEMORY 的 multi-cls 定位)
3. **LogSumExp 数值稳定性**. high-dim soft OR 可能 explode
   - **Mitigation**. PyTorch `torch.logsumexp` 内部已做 subtract-max trick

### 5.2 Open questions (TBD before Task #6 trainer)
1. ~~**Mechanism vocabulary**~~. ✅ **Resolved 2026-06-05**. K=80 locked via top-86 ∩ train-in-all-3-seeds (见 §1.4). Vocab file `Code/data/_cache/ddi_type_map_v3_0.json`. 长尾 135 类直接丢弃, top-86 中 6 类因为某 seed train 缺失踢出.
2. **Query-edge masking under multi-cls**. 当前 v1.7 mask 是 "remove DDI edges between (a,b)". multi-query 下是否 mask 所有 mechanism 的 (a,b) edge? 还是只 mask 当前 query mechanism 对应的 edge?
   - **Default**. mask all DDI-relation edges between (a,b) (relation-aware on DDI bundle, mechanism-agnostic) — 保持跟 v1.7 一致
3. **Negative sampling**. multi-cls 在 negative 上没 label, 但 binary 需要 negative. 是否每个 batch 都 sample 1:1 neg:pos? 还是 only positives for multi-cls portion of loss?
   - **Default**. 1:1 neg:pos per batch, multi-cls loss 只算在 pos 上 (per §2 公式)
4. **MLP_multi 是否真该共享跨 mechanism**. 共享 + concat $q_m$ 让 head 是 hyper-network style; 不共享每个 mechanism 一个 head 参数膨胀 K=80×
   - **Default**. 共享, concat $q_m$ (per §1.2)
5. **OOV positives (~2.4%) 怎么处理 binary 信号**. ✅ **Decided 2026-06-05**. 长尾正例 multi_label = -1 (CE skip), 仍然进 binary BCE label=1. 它们 IS a DDI, only mechanism unknown to vocab. trainer 用 `MultiClsVocab.lookup_type_or_negative(row["ddi_type"])` 取标签.
6. **Eval split: OOV macro-F1 怎么报**. ✅ **Decided**. macro-F1 在 K=80 vocab 上算 (denominator=K=80, 缺类按 0 处理). OOV row 在 multi-cls eval 中跳过 (不算 TP/FP/FN). Binary AUROC/AP 在所有 row 上算, 不区分 OOV.

---

## 6. Acceptance criteria for v3.0 MVP

**Code-level**. 
- ✅ `NBFNetJointDDI.score_pair_dual` returns (multi_logits, binary_logit) shape正确
- ✅ 在 toy 10-node KG 上 forward + backward 无 NaN, gradient 流到 shared params + 每个 $q_m$
- ✅ Multi-cls logits 在 toy 上可以 fit (CE loss 下降)
- ✅ Binary logits 在 toy 上可以 fit (BCE loss 下降)
- ✅ Joint loss 下两个 loss 都下降 (or 至少不 catastrophic 互相干扰)
- ✅ codex 多轮 review pass

**Experimental-level (Task #7)**.
- ✅ (c) binary AUROC > (a) binary AUROC ON DrugBank S2
- ✅ (c) macro-F1 > (b) macro-F1 ON DrugBank S2
- ⚠ Bonus. (c) binary AUROC > v1.7 single-query baseline

---

## 7. Comparison with related v-series

| Version | Task | Architecture | Status |
|---|---|---|---|
| v1.5A | binary | analytical PMP (cluster + within-cluster pool) | done, AUC 0.7764 |
| v1.6 | binary | PyG GNN-form of v1.5A | done, mathematically equivalent |
| v1.7 | binary | paper-faithful single-query NBFNet | model GO, trainer待修 (Task #1) |
| **v3.0** | **binary + multi-cls joint** | **multi-query NBFNet + dual head + soft-OR** | **MVP, this doc** |
| v3.1+ | binary + multi-cls + zero-shot | + ULTRA mechanism meta-graph | future, Task #8 |

v3.0 build on v1.7 NBFLayer/PNAAggregator (reuse, no modification). 如果 v1.7 trainer 还没修完, 不阻塞 v3.0 model 开发 (model 独立).

---

## 8. References

- **NBFNet**. Zhu et al., NeurIPS 2021, arxiv 2106.06935
- **PNA**. Corso et al., NeurIPS 2020, "Principal Neighbourhood Aggregation for Graph Nets"
- **ULTRA**. Galkin et al., ICLR 2024, "Towards Foundation Models for Knowledge Graph Reasoning"
- **MEMORY 笔记**. project_pmp_v1_5_official, project_pmp_v1_6_gnn_form, project_semantic_path_ddi
- **Codex thread**. 019e95e4 (v1.7 review history, v3 will continue at new thread)
