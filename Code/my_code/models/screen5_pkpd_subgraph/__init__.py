"""Screen 5: PK/PD-aware subgraph + Virtual Mediator Embedding (VME).

Splits merged KG into PK and PD subgraphs (per E1b layer mapping); flow
runs in parallel on each; at structural gaps (distance ≥2 pairs with no
intermediate node), inject LLM-derived virtual mediator embeddings.

See: Notes/Experiments/first_step_plan.md §4.7.
"""
