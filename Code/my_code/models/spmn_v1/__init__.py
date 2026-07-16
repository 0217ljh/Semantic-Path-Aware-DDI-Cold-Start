"""SPMN v1 — Semantic-Path Mechanism Network for cold-start DDI prediction.

Design frozen 2026-06-22 after a 14-round codex debate
(see ``Notes/Log/algorithm_design_debate.md``). Core thesis: preserve the
pairwise mechanism support ``A_tau^(l)(a,b)`` EXPLICITLY by exact typed
set-intersection over a symmetrized biomedical KG, BEFORE any message
passing. Coarse-to-fine: explicit per-type reachability + explicit cross-type
co-path features carry structure discovery; a shallow (H<=2) propagation only
refines node representations; molecular BRICS fragments align to mechanism
types/entities/channels (mechanism-level, never drug-identity).

Build is phased and gated (Phase 1 = explicit structural features only). This
package grows one module per frozen forward-pass stage:

    retrieval.py       symmetrized merged-KG + bounded-distance corridor support
    struct_features.py exact zero-param features s_tau, s_{tau,tau'}
    (later)            frag_encoder / node_init / refinement / prototypes /
                       fragalign / gate / fusion_head / model / trainer

Only the merged KG (Code/data/KG/_merged_kg) is used; the EmerGNN-style
propagation backbone is intentionally NOT included.
"""
from __future__ import annotations
