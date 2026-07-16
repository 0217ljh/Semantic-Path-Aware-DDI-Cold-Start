---
type: proj-insight
project: Semantic-Path-Aware-DDI-Cold-Start
status: open
tldr: 实现基于语义图谱感知的cold start预测，根据我们之前在benchmark上的分析，目前的cold start的预测瓶颈来自PD-B
tags: ["#proj-insight", "#project/Semantic-Path-Aware-DDI-Cold-Start"]
created: 2026-05-11
---
# PD-B

## Trigger
实现基于语义图谱感知的cold start预测，根据我们之前在benchmark上的分析，目前的cold start的预测瓶颈来自PD-B

## Direction
*What to change starting from this insight. What kind of result we hope to see. Rough — formalized later in [[Setting]].*

根据我们的分析，PD-B这个ddi预测，往往基于drug effect的临床效果，在器官或者系统级别上起到作用，然后这个作用同样是在这个层面上pair的两个drug出现冲突，才导致的ddi发生。

因此，这个预测需要明确的特征指标变化-》宏观身体side effect，即相对粗糙的临床症状的这个path才能进行比较好的预测。

而根据我们目前对多个Biomedical KG的考察，这个path在KG里面往往是显式缺少的，因为相对生物医学领域来说比较basic（可能是这个原因）。

对于这种比较general的推理（特征指标变化-〉side effect），让llm推理可以做的比较好，因为它有更general的知识。

但是根据我们对于llm的ddi预测分析表明，llm具有一定的医学知识，但它不知道怎么用，ft仅能在一定程度上改善了这个现象。

因此，我们希望：
* 用KG的结构偏置来guide llm的prediction，即路径感知或者路径分布感知，来代替ft使得llm进行更robust以及可解释的ddi预测，在维持pk准确率的同时可以识别PD，
* 并在另一方面，通过结合llm的知识来补全kg内缺失的路径，以实现pd上的准确率突破



## Why Worth Pursuing
*What makes this non-trivial or non-obvious?*
* 更好的cold start预测准确率
* 更可解释，能够超越KG本身的局限性，即不受具体的path约束
* **KG↔LLM 双向耦合**
* **互补性而非替代性 — Patch 两侧的弱点**
* **训练成本优势 — Structural guidance 替代 FT**


## Open Questions
*What does this insight not yet answer?*

-
