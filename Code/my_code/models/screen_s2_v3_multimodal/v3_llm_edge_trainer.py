"""D1 trainer — EmerGNN + LLM-typed edges injected into the path-flow backbone.

Round 4 D1 (per Notes/Log/d1_llm_edge_design.md + CP-1 PASS_WITH_NITS at
Code/my_code/models/screen_s2_v3_multimodal/_reviews/2026-05-30__d1_design__round1.md).

Inherits from `_PerModeEmerGNN_V2I4` (v2i4 = the 0.7804 baseline). Overrides
`_setup_graph` to extend the drugbank 5-bucket KG with new drug→LLM-token edges,
producing 10 new relation types (`llm:cyp_substrate`, ..., `llm:clearance`) and
~7700 new entity nodes (one per unique LLM-distilled token). The EmerGNN path-flow
backbone is unchanged — only the KG input grows.

Controls (per design §3):
  K1: --d1-shuf-token   edge-level permute drug_id column (drug->own-token binding
                        destroyed; per-drug cardinality NOT preserved by design;
                        per-relation count + per-token degree preserved before dedup)
  K2: --d1-rand-token   replace each edge's token with a globally unique rand_<i>
                        (per-edge cardinality preserved, zero cross-drug bridges)
  K3: --d1-drop-i4-head freeze β_i4 at 0 (force backbone to carry the signal alone)
  K4: --d1-disable      skip injection entirely (v2i4 parity smoke)

File-independence: this file does NOT modify mnah_trainer.py / v2i4_trainer.py /
baseline/emergnn/*. It only inherits + overrides.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen_s2_v3_multimodal.v2i4_trainer import _PerModeEmerGNN_V2I4
from baseline.emergnn.kg_builder import build_sparse_adj, edges_as_dense_lists


DEFAULT_EDGES_PARQUET = (
    PROJECT_ROOT
    / "Code" / "data" / "_cache" / "llm_edges"
    / "llm_drug_edges__seed42_drugbank.parquet"
)


class _PerModeEmerGNN_V3LLMEdge(_PerModeEmerGNN_V2I4):
    """EmerGNN + LLM-typed edges (D1 backbone integration)."""

    def __init__(
        self,
        *,
        d1_edges_parquet: str | Path | None = None,
        d1_disable: bool = False,
        d1_shuf_token: bool = False,
        d1_rand_token: bool = False,
        d1_drop_i4_head: bool = False,
        d1_shuffle_seed: int = 12345,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d1_edges_parquet = (
            Path(d1_edges_parquet) if d1_edges_parquet else DEFAULT_EDGES_PARQUET
        )
        self.d1_disable = bool(d1_disable)
        self.d1_shuf_token = bool(d1_shuf_token)
        self.d1_rand_token = bool(d1_rand_token)
        self.d1_drop_i4_head = bool(d1_drop_i4_head)
        self.d1_shuffle_seed = int(d1_shuffle_seed)
        # Internal: populated by _inject_llm_edges for diagnostics.
        self._d1_summary: dict | None = None

    # ------------------------------------------------------------------
    # _setup_graph override (the single hook point for D1)
    # ------------------------------------------------------------------

    def _setup_graph(self, train, kg):
        morgan_mat, drug_id_list = super()._setup_graph(train, kg)
        if self.d1_disable:
            print(
                "[d1] --d1-disable set; skipping LLM-edge injection (K4 parity smoke).",
                flush=True,
            )
            return morgan_mat, drug_id_list

        if self.d1_shuf_token and self.d1_rand_token:
            raise ValueError("--d1-shuf-token and --d1-rand-token are mutually exclusive.")

        self._inject_llm_edges(drug_id_list)

        # Zero-pad morgan_mat to the new (larger) n_ent. LLM token nodes have no
        # SMILES, so their feat row is the zero vector (consistent with how
        # _per_mode.py:305-309 already handles drugs without SMILES).
        new_n_ent = self._n_ent
        n_old_rows = morgan_mat.shape[0]
        if new_n_ent > n_old_rows:
            padded = np.zeros((new_n_ent, morgan_mat.shape[1]), dtype=np.float32)
            padded[:n_old_rows] = morgan_mat
            morgan_mat = padded
            print(
                f"[d1] morgan_mat padded from {n_old_rows} to {new_n_ent} rows "
                f"(zeros for LLM token nodes)",
                flush=True,
            )
        return morgan_mat, drug_id_list

    # ------------------------------------------------------------------
    # LLM-edge injection
    # ------------------------------------------------------------------

    def _inject_llm_edges(self, drug_id_list: list[str]) -> None:
        """Extend self._entity2id / self._n_base_rel / self._kg_triplets / edges."""
        if not self.d1_edges_parquet.is_file():
            raise FileNotFoundError(
                f"D1 LLM-edge parquet not found at {self.d1_edges_parquet}. Run "
                f"`precompute_llm_edges.py` first."
            )
        edges_df = pd.read_parquet(self.d1_edges_parquet)
        n_input = len(edges_df)
        required_cols = {"drug_id", "token_node_id", "relation"}
        if not required_cols.issubset(edges_df.columns):
            raise ValueError(
                f"D1 parquet missing required columns; have {list(edges_df.columns)}"
            )
        edges_df = edges_df[["drug_id", "token_node_id", "relation"]].copy()

        # Drop rows whose drug_id is not in the trained drug pool. The 800-drug
        # split is a strict subset of i4_typed_sets.json's 1530 drugs, so we
        # naturally have edges for drugs outside the trained pool — those edges
        # would attach to nothing useful and we skip them with a count log.
        drug_in_pool = edges_df["drug_id"].astype(str).isin(self._entity2id)
        n_dropped_out_of_pool = int((~drug_in_pool).sum())
        edges_df = edges_df.loc[drug_in_pool].reset_index(drop=True)

        # Control mutations (K1 shuf-token / K2 rand-token).
        if self.d1_shuf_token:
            edges_df = self._mutate_shuf_token(edges_df)
        elif self.d1_rand_token:
            edges_df = self._mutate_rand_token(edges_df)

        # Build LLM relation vocab + new entity ids.
        llm_relations = sorted(edges_df["relation"].unique().tolist())
        n_old_rel = int(self._n_base_rel)
        rel2id = {r: n_old_rel + i for i, r in enumerate(llm_relations)}

        new_token_nodes = sorted(edges_df["token_node_id"].unique().tolist())
        # Hard guard: no llm:<...> ever collides with an existing entity id. The
        # builder prefixes with "llm:" which is disjoint from drugbank ID prefixes
        # (no DrugBank entity uses "llm:" — DrugBank ids start with "DB",
        # protein/enzyme/etc ids carry their bucket-specific schemas).
        existing_with_llm = {e for e in self._entity2id if e.startswith("llm:")}
        if existing_with_llm:
            raise RuntimeError(
                f"unexpected 'llm:'-prefixed nodes already in vocab: "
                f"{sorted(existing_with_llm)[:5]}"
            )
        new_collisions = [n for n in new_token_nodes if n in self._entity2id]
        if new_collisions:
            raise RuntimeError(
                f"token_node_id collisions with existing vocab: "
                f"{new_collisions[:5]} (expected zero by prefix-disjointness)"
            )

        n_old_ent = int(self._n_ent)
        for i, tok in enumerate(new_token_nodes):
            self._entity2id[tok] = n_old_ent + i
        n_new_ent = n_old_ent + len(new_token_nodes)

        # Build new triplets (drug as head, token as tail, llm:<field> as rel).
        h_arr = edges_df["drug_id"].astype(str).map(self._entity2id).to_numpy(dtype=np.int64)
        t_arr = edges_df["token_node_id"].astype(str).map(self._entity2id).to_numpy(dtype=np.int64)
        r_arr = edges_df["relation"].astype(str).map(rel2id).to_numpy(dtype=np.int64)
        if (np.isnan(h_arr.astype(np.float64)).any() or
                np.isnan(t_arr.astype(np.float64)).any() or
                np.isnan(r_arr.astype(np.float64)).any()):
            raise RuntimeError("D1 edge index mapping produced NaN — vocab gap")
        new_triplets = np.stack([h_arr, t_arr, r_arr], axis=1)

        # Merge into self._kg_triplets, dedupe (defensive — builder already
        # dedupes, but a drug→token edge could in theory clash with an existing
        # drugbank 5-bucket entry if a drug had been linked to a token-named
        # entity via a non-LLM relation; we keep both with distinct rel ids,
        # so no actual clash, but np.unique guards against accidental dup).
        kg_all = np.concatenate([self._kg_triplets, new_triplets], axis=0)
        before = len(kg_all)
        kg_all = np.unique(kg_all, axis=0)
        if before != len(kg_all):
            print(
                f"[d1] merged-triplet dedup removed {before - len(kg_all)} "
                f"(safety net; expected 0)",
                flush=True,
            )

        # Commit the expansion.
        self._kg_triplets = kg_all
        self._n_ent = n_new_ent
        self._n_base_rel = n_old_rel + len(llm_relations)
        self._kg_entity_set = set(
            int(x) for x in np.unique(self._kg_triplets[:, :2]).astype(np.int64).tolist()
        )

        # Rebuild edge tensors (parent expects them in self._edge_src/_dst/_rel
        # for save/load round-trip via _per_mode.py:644-649).
        adj = build_sparse_adj(self._kg_triplets, self._n_ent, self._n_base_rel)
        self._edge_src, self._edge_dst, self._edge_rel = edges_as_dense_lists(adj)

        self._d1_summary = {
            "n_edges_input": n_input,
            "n_edges_dropped_out_of_pool": n_dropped_out_of_pool,
            "n_llm_edges_injected": int(len(new_triplets)),
            "n_new_relations": len(llm_relations),
            "n_old_relations": n_old_rel,
            "n_new_token_nodes": len(new_token_nodes),
            "n_ent_before": n_old_ent,
            "n_ent_after": n_new_ent,
            "n_base_rel_before": n_old_rel,
            "n_base_rel_after": self._n_base_rel,
            "control_shuf_token": self.d1_shuf_token,
            "control_rand_token": self.d1_rand_token,
            "control_drop_i4_head": self.d1_drop_i4_head,
        }
        print(f"[d1] injection summary: {self._d1_summary}", flush=True)

    # ------------------------------------------------------------------
    # Control mutations
    # ------------------------------------------------------------------

    def _mutate_shuf_token(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """K1 — edge-level permutation of the drug_id column.

        Implementation (CP-2 round 1 fix; the previous list-level permutation gave
        drug A the full token list of drug B, exploding A's cardinality):

          For each row i in edges_df we reassign `drug_id[i] = drug_id_pool[perm[i]]`
          where `perm` is a global random permutation of the drug-id column. Every
          row keeps its (token_node_id, relation); only the head-drug attaches to a
          different drug. After permutation we dedup at (drug, token, relation) to
          drop the (small) number of rows where the permuted drug already had this
          edge.

        What this PRESERVES (so capacity / structural density of the new edges is
        not the explanation for any AUC change vs canonical D1). All "exactly"
        claims hold BEFORE the dedup at line ~273; after dedup they hold modulo
        the small number of dropped duplicate rows (logged):
          - total edge count (modulo dedup; expected loss << 1%),
          - per-relation edge count exactly (before dedup),
          - per-token-node in-degree exactly (before dedup),
          - per-(token, relation) edge count exactly (before dedup).

        What this DESTROYS (the targeted control variable):
          - the drug → its own LLM-derived token-set binding. A drug u that had
            CYP3A4-substrate / specific PD effects now points to an unrelated drug's
            tokens. Drugs u and v that genuinely share a mediator no longer do.

        What this DOES NOT preserve (acknowledged trade-off, less aggressive than the
        old "preserve cardinality" claim that was broken):
          - per-drug edge count distribution. After permutation a drug's cardinality
            is approximately Poisson around the global mean rather than its own
            empirical value. This is the unavoidable cost of an edge-level breaker;
            list-level / cardinality-preserving alternatives reintroduce binding
            (see K2 rand-token for the orthogonal control that destroys bridges
            instead).
        """
        rng = np.random.default_rng(self.d1_shuffle_seed)
        out = edges_df.copy().reset_index(drop=True)
        n_in = len(out)
        if n_in == 0:
            return out
        perm = rng.permutation(n_in)
        original_drugs = out["drug_id"].to_numpy().copy()
        out["drug_id"] = original_drugs[perm]
        # Drug-id has many repeats, so perm[i]==i undercounts how many rows
        # ended up with the SAME drug after permutation (a row could land on a
        # different perm index whose drug happens to be the original drug).
        # Compare the actual reassigned drug-id to the original.
        n_same_drug_after_perm = int(
            (original_drugs[perm] == original_drugs).sum()
        )
        before = len(out)
        out = out.drop_duplicates(
            subset=["drug_id", "token_node_id", "relation"]
        ).reset_index(drop=True)
        n_dropped = before - len(out)
        # Diagnostic: how far per-drug cardinality drifted.
        before_card = pd.Series(original_drugs).value_counts()
        after_card = out["drug_id"].value_counts()
        common = before_card.index.intersection(after_card.index)
        if len(common) > 0:
            mean_abs_card_drift = float(
                (before_card.loc[common] - after_card.loc[common]).abs().mean()
            )
        else:
            mean_abs_card_drift = float("nan")
        print(
            f"[d1] *** SHUF-TOKEN CONTROL *** edge-level drug-column permutation: "
            f"n_edges {n_in} -> {len(out)} after dedup (dropped {n_dropped} dup rows); "
            f"rows landing back on same drug = {n_same_drug_after_perm}/{n_in}; "
            f"mean abs per-drug cardinality drift vs canonical = "
            f"{mean_abs_card_drift:.2f}. Preserves (before dedup): per-relation count, "
            f"per-token degree. Destroys: drug->own-token binding.",
            flush=True,
        )
        return out

    def _mutate_rand_token(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """K2 — replace each edge's token with a globally unique rand_<idx>.

        Zero cross-drug bridges by construction (each synthetic token used by 1 drug).
        Edge count preserved 1:1; relation ids preserved.
        """
        out = edges_df.copy().reset_index(drop=True)
        n = len(out)
        unique_ids = [f"llm:rand:rand_{i:08d}" for i in range(n)]
        out["token_node_id"] = unique_ids
        print(
            f"[d1] *** RAND-TOKEN CONTROL *** replaced {n} tokens with globally unique "
            f"synthetic ids (zero cross-drug sharing by construction).",
            flush=True,
        )
        return out

    # ------------------------------------------------------------------
    # K3: drop-i4-head — applied at fit() time
    # ------------------------------------------------------------------

    def fit(self, train, val=None, *, kg=None):
        super().fit(train, val=val, kg=kg)
        if self.d1_drop_i4_head:
            # Force β_i4 = 0 and freeze (post-training; but we want at training,
            # so this must happen in _build_aux_head). Override below.
            pass
        return

    def _build_aux_head(self, in_dim: int) -> nn.Module:
        """Override v2i4's head to optionally freeze β_i4 = 0 (K3 drop-i4-head)."""
        head = super()._build_aux_head(in_dim)
        if self.d1_drop_i4_head:
            # head.raw_beta_i4 is a Parameter from CountPlusI4Head.__init__.
            with torch.no_grad():
                # softplus(-50) ≈ 0; freeze.
                head.raw_beta_i4.fill_(-50.0)
            head.raw_beta_i4.requires_grad_(False)
            print(
                "[d1] *** DROP-I4-HEAD CONTROL *** β_i4 frozen at ~0 via raw_beta_i4=-50; "
                "i4 readout head disabled (backbone must carry LLM signal alone).",
                flush=True,
            )
        return head


__all__ = ["_PerModeEmerGNN_V3LLMEdge", "DEFAULT_EDGES_PARQUET"]
