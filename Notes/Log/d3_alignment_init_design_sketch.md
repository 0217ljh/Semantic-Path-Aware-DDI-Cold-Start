# D3 Design Sketch — Alignment InfoNCE → EmerGNN Backbone Init Embedding

**Status**. Pre-staged sketch (2026-06-01) while D2 runs. If D2 CP-3 PASSes, this stays archived. If D2 NOT_PASS, this becomes the next CP-1 input.

## Mechanism

Round4 plan §4 D3: "alignment loss only produced `m_u` (molecular vector) but didn't reach the main path; D3 changes EmerGNN drug node embedding init = concat(原 embed, projection(m_u)) or 原 embed + projection(m_u), training m_u as pretraining + freeze or fine-tune".

The infrastructure already exists at `Code/my_code/models/screen_s2_v3_multimodal/train_alignment_infonce.py` (verified at file lines 1-60). Output `Code/data/_cache/molecular_aligned_infonce.npz` schema verified 2026-06-01:
- `drug_ids`: (1994,) str — full DrugBank universe
- `z_m`: (1994, **128**) float32 — aligned molecular embedding per drug (mean≈0, std≈0.088)
- `is_seen`: (1994,) bool — 640 True (matches G1 seen drugs in 800-pool from LEAKAGE_AUDIT)
- `residual`: (1994,) float32 — self-gating proxy (R6 stop-grad use)
- `align_dim` / `temp` / `holdout_auc_best` / `holdout_top5_best` / `holdout_top10_best` / `holdout_mrr_best`: scalars (training metadata)

## D3 architecture (subclass plan)

New file `Code/my_code/models/screen_s2_v3_multimodal/v3_mol_init_trainer.py`:

```python
class _PerModeEmerGNN_V3MolInit(_PerModeEmerGNN_V2I4):
    def __init__(self, *, d3_mol_npz: Path = None, d3_init_mode: str = "add",
                 d3_proj_dim: int = 64, d3_disable: bool = False,
                 d3_shuf_mol: bool = False, d3_rand_mol: bool = False,
                 ...):
        super().__init__(...)
        ...
```

Hook point: override `_setup_graph` to load `z_m`, build a per-drug-id mapping, then apply at model construction:
- `d3_init_mode="add"`: morgan_mat[drug_idx] += proj(z_m[drug])
- `d3_init_mode="concat"`: requires extending the EmerGNN morgan_feat_dim, which is invasive

Simpler: only `d3_init_mode="add"` for round 1.

## Controls (baked in from CP-1, learning from D2's K1/K2)

- **K1 shuf-mol**: shuffle drug→z_m binding (permute z_m rows across drugs). If D3 lift survives, molecular semantic identity doesn't drive it.
- **K2 rand-mol**: replace each drug's z_m with random Gaussian of matching dim. Tests "any vector init helps" hypothesis.
- **K3 no-init**: `d3_disable=True`, no init modification. v2i4 byte-equivalent.

## R1-equivalent risk

`z_m` is dense per-drug (no zero-mediator pairs like D2). D3 should give signal on ALL pairs, not just 30% like D2's Path B reality. This is a structural advantage over D2.

## Hard-stop / expected lift

Same as D2:
- combined ≥ 0.785 (target), 0.79 ideal
- emergnn-branch ≥ 0.75

## Will only spend time on this if D2 fails

CP-1 codex review prerequisite. Build steps:
1. Verify `train_alignment_infonce.py` outputs (re-run if cache stale)
2. Write D3 design doc (this sketch expanded)
3. CP-1 codex review
4. Implement
5. CP-2
6. Run + controls
7. CP-3

Total: ~6-8 hours follow-up work if D2 NOT_PASS.
