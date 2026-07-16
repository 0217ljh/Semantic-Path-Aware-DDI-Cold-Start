# Language Encoding & Node Aggregation — Follow-up Improvement Area

**Status**: not yet tested (placeholder for future iteration)
**Logged**: 2026-05-30
**Context**: the `kg_neighbor_target_pubmedbert.npz` cache (`k_u`, 7754 × 768) is the alignment target for the InfoNCE multimodal alignment and the embedding source for the v2 effect cross-attention. It's used (directly or via `k_typed`) by every multimodal / effect-channel experiment so far. **All current experiments are bottlenecked by a single fixed encoder + a single fixed input format + a single fixed aggregation choice.** None of these three has been compared against alternatives.

---
后续这个改进可能有效的理由：codex 在 strategic verdict 里指出:S2 是 DDI-edge cold-start,unseen 药仍保留 KG 邻居,所以 EmerGNN path-flow 已经直接传播过这些邻居信号。k_u(无论用哪个 encoder)只是把这个邻域信息重新表达了一遍,不是新增信息。
验证证据:E-frag(分子→k_typed 对齐)mol_head_alone 只有 0.577,在加上 KG 之后没增量。这说明问题不在"对齐目标编码不够好",而在"分子表示在结构上跟 KG 邻域已表达的内容高度冗余"。
但这条论据有个例外:如果 encoder 升级后能让 k_u 编码出 KG 拓扑路径之外的"语义先验"(比如 PubMed 文献频繁共现关系、临床描述里的关联),理论上能给模型新信息。这个还没人测过 —— 是个开放问题。

## What's currently encoded

Three locked-in choices that none of the experiments to date have varied:

```
[step 1: input text]          [step 2: encoder]               [step 3: aggregation]
       ↓                              ↓                                ↓
  KG node NAME            PubMedBERT-base                 mean-pool over drug's
  (bare string)           (microsoft/BiomedNLP-           1-hop non-drug neighbors
                          PubMedBERT-base-uncased-
                          abstract-fulltext) [CLS]
```

- **Input text**: each KG node feeds just its bare name string — e.g. `"CYP3A4"`, `"thrombin (Factor IIa)"`, `"headache"`, `"G2/M DNA damage checkpoint"`. No node kind, no UMLS/Mesh definition, no synonyms, no multi-sentence description.
- **Encoder**: PubMedBERT-base (110M params, 2020-era, trained on PubMed abstracts + fulltext), [CLS] token, 768d. Cached in `d_name_only__pubmedbert.pt`.
- **Aggregation**: per drug, mean of the [CLS] embeddings over all 1-hop non-drug KG neighbors. `k_typed` (`kg_typed_target_pubmedbert.npz`) is a bucketed version (8 node-kind buckets, mean within each) but still mean-pool within bucket.

---

## Why this is likely an improvement area (the case for trying)

### Axis 1 — Input text is information-poor

The encoder is given only a short token like "PMS2" or "G2/M DNA damage checkpoint". It must rely entirely on what PubMedBERT saw in pretraining about that exact token. There's no context, no node-type signal, no definition, no synonyms.

Concrete cheap upgrades:

- **`name + node_kind`**: `"CYP3A4 (gene/protein)"` — 2-token gain, free
- **`name + brief definition`**: pull UMLS / Mesh / GO term description (1-2 sentences) → input becomes `"CYP3A4 — cytochrome P450 3A4; primary hepatic drug-metabolizing CYP isoform; substrates include statins, immunosuppressants"`. UMLS/MeSH lookups are zero-cost for established biomedical ontology terms.
- **`name + synonyms`**: include common aliases (e.g. `"PMS2 | postmeiotic segregation increased 2 | mismatch repair protein"`).

This is arguably the highest-EV change on this axis because it doesn't change anything downstream — same encoder, same cache shape — only the input is richer.

### Axis 2 — Encoder choice is locked to PubMedBERT-name-CLS

We never tested any alternative. Candidates worth comparing (all open-source unless noted):

- **MedCPT** (NCBI 2023): explicitly trained for biomedical *retrieval*, not just MLM. Usually beats PubMedBERT on similarity / retrieval tasks. Strongest a-priori bet for "embed biomedical names well".
- **BioLinkBERT-large** / **PubMedBERT-large**: bigger versions of the same family.
- **E5-large / BGE-large / GTE-large**: general-purpose dense retrieval encoders on top of MTEB. Out-of-domain but often strong because the contrastive pre-training is much heavier.
- **NV-Embed-v2** / **SFR-Embedding-Mistral**: MTEB top-tier; heavier to deploy, but worth one comparison.
- **OpenAI `text-embedding-3-large`**: API-based, cheap (~$0.13/M tokens), generic but strong; useful as an apples-to-apples comparator.
- **Sentence-Transformers all-MiniLM-L12** as a fast tiny-model floor for the comparison.

Note PubMedBERT[CLS] is a **non-pooled token**; modern retrieval-tuned encoders usually pool differently (mean-of-tokens, attention-pool). If we switch encoder we should adopt the encoder's *recommended* pooling, not force CLS.

### Axis 3 — Aggregation is unweighted mean-pool

Even with the same input + encoder, the mean-pool over a drug's neighbors **dilutes informative rare neighbors with generic ones**. A drug's KG neighborhood may include one mechanistically specific node (e.g. its primary CYP target) plus dozens of generic GO biological-process nodes; mean-pool flattens that signal.

`k_typed` partially addresses this (per-kind bucket means), but within-bucket dilution remains.

Concrete alternatives:

- **Inverse-document-frequency weighting**: weight each neighbor by `1/log(1 + drug_degree(neighbor))` — rare specific neighbors (a unique target) outweigh generic ones (a common pathway). We already cache `nbr_drugdeg` for effect-neighbors (#5), this would just generalize.
- **Attention pooling** (learnable): a small attention module over the neighbor set, trained jointly with the alignment loss. Could let the model learn which neighbor types/identities matter per drug.
- **Multiple statistics concat**: `[mean, std, max]` over neighbor embeddings instead of mean alone (4× dimensionality but cheap).
- **Per-relation-type bucketing** (extension of `k_typed`): if the merged KG has relation types beyond just node kinds, bucket by **edge type** (e.g. "metabolized-by", "indicated-for", "side-effect") and pool within each. Probably the most mechanistically meaningful split.

---

## The case against (why it might not move the needle)

codex's strategic verdict (thread `019e67af`, 2026-05-27) argued that S2 cold-start in this dataset is **DDI-edge cold-start, not biomedical cold-start** — unseen drugs retain their biomedical KG neighbors at test time, so EmerGNN's path-flow already propagates over those same edges. `k_u` (any encoder) is therefore a **redundant re-expression** of information the backbone already exploits.

This was supported by E-frag (shared/residual fragment-alignment to typed-KG, mol-head alone 0.577, no lift on top of KG): even with a relation-typed target and a richer molecular source, the alignment added nothing on top of KG.

So the honest expectation under codex's thesis: encoder swap **on its own** won't break the ~0.78 ceiling, because the alignment target is the wrong lever.

**Counter-argument worth testing**: if a stronger encoder + richer input lets `k_u` encode **semantic priors beyond the KG's edge graph** (PubMed co-occurrence, clinical-narrative associations, GO/Mesh definitional knowledge), it might inject genuinely new information that EmerGNN couldn't reach via topology alone. This is an open empirical question — codex's thesis says probably not; standard contrastive-retrieval intuition says possibly yes. Untested.

---

## Concrete experiment plan (when this gets queued)

Recommended priority order (cheapest to most expensive, all clean orthogonal axes):

1. **Input enrichment, same encoder**: PubMedBERT-base + `name + kind + UMLS/MeSH definition (1 sentence)`. Rebuild `k_u`. Re-run InfoNCE alignment → measure (a) holdout retrieval AUC vs current 0.68 / top1 0.16 / MRR 0.26, (b) downstream effect on v2 multimodal-alignment lift.
2. **Encoder swap, same simple input**: MedCPT → rebuild `k_u`. Same metrics.
3. **Encoder × input matrix (4 cells)**: PubMedBERT vs MedCPT × name-only vs name+definition. Establishes which lever matters more.
4. **Aggregation upgrade**: pick best (encoder, input) from above; replace mean with IDF-weighted, then attention-pool.
5. **Relation-typed bucketing** (if merged KG relation labels are easily accessible): bucket by edge type, not just node kind.

Success bar (per codex's noise calibration ±0.3pt):
- Holdout retrieval AUC > 0.80 → encoder/input was the bottleneck
- Downstream multimodal-alignment lift > +1pt over current best → axis is real
- If both fail → codex's redundancy thesis confirmed, write it explicitly as a finding

---

## Convention going forward

Per user instruction (2026-05-30): **experiment-related log notes go under `Notes/Log/` with short topic-named markdown files** (this file = `language_encoding.md`). Cache modifications / replacement experiments should each get such a log so the design history is traceable.

---

## Affected downstream artifacts (what would need rebuilding if `k_u` changes)

If `k_u` is regenerated under a new encoder/input/aggregation:

- `kg_neighbor_target_pubmedbert.npz` → new file (suggest path-encode the encoder, e.g. `kg_neighbor_target_medcpt_namedef.npz`)
- `kg_typed_target_pubmedbert.npz` → same logic, rebuild bucketed version
- `effect_neighbors_pubmedbert.npz` indexes into the SAME PubMedBERT node embeddings. If we switch the per-node encoder, `effect_neighbors_*` should be rebuilt with the new node embeddings cached separately.
- `molecular_aligned_infonce.npz` → InfoNCE retrain (~6 min, fast)
- All v2 / v3 / E-frag experiments that consume these caches would need to be re-run for apples-to-apples comparison (≈ 35 min each on GPU)

Total cost for the full encoder×input×aggregation sweep is on the order of one GPU-day plus a few hours for the precompute runs. Achievable in one focused session.
