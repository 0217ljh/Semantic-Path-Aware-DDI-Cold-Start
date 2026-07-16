# exp3 — Non-trivial Phase Transitions

> **Goal**: 发现 cold-start DDI 上 *paradoxical* 的现象——"看似 supported 但失败"的 regime。判断标准: 不是单调 scaling, 不是 trivial locality, 必须是 *某种意义上 well-conditioned 的 input 仍然系统性塌*。

## Codex thread
`019e2607-ef2e-7f21-b4af-1c428bdf20e8` (gpt-5.4, fresh thread, 不带 v1/v2 理论)

## Round 1 — 候选 brainstorm (codex 给 5 个, commit 2 个)

### Codex 5 个 candidates

| # | Phenomenon | Surprising point | 假阳性风险 |
|---|---|---|---|
| C1 | Hubness inversion under neighborhood heterogeneity | High-degree drug 在异质邻域里反而坏 | 中等——degree 容易 confound |
| **C2** | **Mediator conflict transition** | mediator-rich 但 mechanism 冲突时突然塌 | **低-中** |
| C3 | Twin-drug ambiguity U-shape | 极高相似度反而难 | 高——label/chemistry confound |
| **C4** | **Cross-modal disagreement boundary** | structure/text/KG 各模态 support 但 disagree 时塌 | **低** |
| C5 | Local spectral anisotropy boundary | embedding cloud 谱形状决定边界 | 信号弱 |

### Codex commit (ruthless ranking)

- **第一优先: C2 Mediator conflict transition**
  - 最贴现有资产 (E7 meeting-node + E8 oracle 已暗示)
  - 最难被读成 trivial monotone (固定 mediator abundance, 只看 conflict)
  - 主叙事: **evidence superposition causes non-identifiability**

- **第二优先: C4 Cross-modal disagreement boundary**
  - 实现干净, 2D heatmap 直接出
  - paradox 强: 每模态 individually supported, jointly fail
  - 后续推进容易 (agreement-aware gating / uncertainty routing)

### Combined narrative

> **Cold-start DDI 的 boundary 不是 support scarcity, 而是 support inconsistency causing identifiability collapse.**

C2 说"同一模态内部, support 过多且冲突会塌"；C4 说"跨模态之间, support 不一致也会塌"。

### 不优先 (with reasons)

- C1: degree confound 风险大
- C3: label/chemistry family confound 风险大
- C5: 信号可能弱, 适合 follow-up mechanistic 不适合首打

## Next steps

- Round 2: push codex 给精确 analysis plan
  - C2 三个 conflict measure (template entropy / mechanism split entropy / community disagreement) 选哪个 / 怎么混
  - C4 modality 三选哪几对 (structure × text / structure × KG / text × KG)
  - 必做 negative controls
  - Clean figure 设计
- Round 3+: implement
