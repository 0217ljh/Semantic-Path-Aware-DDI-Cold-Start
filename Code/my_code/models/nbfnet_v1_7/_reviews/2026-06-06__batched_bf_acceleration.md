# NBFNet v1.7 — batched multi-source BF (parallel acceleration) review

## Meta

- **Date**. 2026-06-06
- **Primary reviewer**. Claude (claude-opus-4-8) — implemented the additive batched path, wrote + ran the equivalence test.
- **Independent reviewer**. codex (gpt-5.4, `model_reasoning_effort=xhigh`, MCP) — thread `019e9b2f-8167-7661-baea-cee77f94162b` (prior threads `019e95e4`/`019e990a` expired). Codex read the files directly.
- **Trigger**. User. "处理一下并行加速的代码，开始 codex review" — replace the per-source Python-loop BF with a batched multi-source BF, keep results unchanged.

## What changed (purely additive — no Round-7-GO function modified)

`nbfnet_model.py` (additive methods):
- `PNAAggregator.forward_batched(messages (S,M,d), target (M,) shared, n_nodes)` — scatter along dim=1, shared degree/log_deg broadcast `(1,n,1)`.
- `NBFLayer.forward_batched(h,h0 (S,n,d), edges, q, n_nodes)` — `msg = h[:,edge_src]*w_e`, boundary reinjection via `cat([msg,h0],dim=1)`, relu.
- `NBFNetDDI.bellman_ford_batched(sources (S,), edges)` — `h0[arange(S),sources]=q`, L layers.
- `NBFNetDDI.encode_from_sources(...)` — batched encoder; supports only a SINGLE shared edge_keep_mask (no per-source masking by design).

`nbfnet_trainer.py`:
- `_score_pairs` dispatcher; `_query_edges_present` no-op-mask guard; `_score_pairs_batched` (vectorized unique-source encode + gather compose); `use_batched_bf` flag (default True). `_train_epoch`/`_validate`/`predict_proba` call the dispatcher. Existing `_score_pairs_amortized` kept as exact fallback.

`run_nbfnet.py`: `--disable-batched-bf` flag.

## Equivalence test (toy 14-node KG, d=8 L=3, GPU)

1. `encode_from_sources` rows vs per-source `encode_from_source`: max|diff| **5.96e-08**.
2. `_score_pairs_batched` vs `_score_pairs_amortized(training=False)`: logits max|diff| **7.45e-09**.
3. `_query_edges_present` on an S2 epoch KG = **False** → batched path taken in training.
4. grad parity: amortized total-param-grad == batched total-param-grad == **0.2258** (identical loss 0.6949).

(Open item: grad parity is an aggregate scalar; a per-parameter max-abs-diff would be stronger — Codex flagged this.)

## Codex verdict (verbatim, key parts)

> **Verdict** — GO for architectural correctness. I do not see a leakage or math-equivalence blocker in the additive batched path. The only important caveat is terminology: this is mathematically equivalent, not guaranteed bitwise-identical across GPUs / torch_scatter builds.
>
> - No critical correctness findings in the added batched BF path. The approved single-source path is still intact, and the fast path is purely additive.
> - The mask guard is safe but conservative. `_query_edges_present()` can force fallback when there is some DDI edge among batch nodes even if none of the queried pairs needs masking, so S0/S1 speedup may collapse in some batches. That is a performance loss, not a leakage bug.
> - The main real-run risk is memory, not correctness. Batched BF materializes `O(S*E*d)` tensors and training builds two graphs per batch (`hf`, `hr`).
> - Your "grad parity" evidence is weaker than it looks if `total_param_grad` is one aggregate scalar. I would want per-parameter `max_abs_diff`.
>
> 1. Yes — unmasked shared-edge batched BF is the same recurrence stacked over S sources; dim-1 scatter, shared 1D target, boundary concat on dim 1, log_deg broadcast all consistent.
> 2. No-op-mask guard is sound for leakage. A false negative would require a maskable query edge that is not a DDI edge with both endpoints in batch_a ∪ batch_b, which cannot happen. Fallback routes to the exact amortized masked path. For S0/S1 it may over-fallback, not under-fallback.
> 3. All 8 invariants preserved. Nuance: invariant 6 is preserved at the trainer dispatch level (batched encoder used only when masking is provably a no-op).
> 4. Correctness risk on merged KG low if it fits memory; practical risk medium/high from activation RAM / throughput if unique-source count per batch is large. Merge for S2 correctness; keep batch size conservative; do not promise exact long-run training-trajectory identity.

## Practical guidance

- Batched BF is a **speed / GPU-utilization** win (one big propagation vs S sequential passes), **not an OOM fix** — peak memory is still `O(S·E·d)` with two graphs per batch. For OOM on merged KG, still tune `--batch-size` / `--n-layers` / `--n-dim`.
- Merged-KG run confirmed on the correct graph (`n_nodes=22049`, `n_eval_edges=373046 = 53743 train_ddi + 319303 base KG`).

## Open / future

- Strengthen grad equivalence to per-parameter max-abs-diff (currently aggregate scalar + 7.45e-9 logit parity).
- Optional: support true per-source masking inside the batched encoder (would let S0/S1 use the fast path too) — not needed for S2.
