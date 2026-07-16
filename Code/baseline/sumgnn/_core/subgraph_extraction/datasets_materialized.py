"""RAM-resident materialized SubgraphDataset for SumGNN (perf, faithful).

Motivation (codex thread 019f1fb3, step 1): SumGNN's ``SubgraphDataset.__getitem__``
reconstructs each enclosing subgraph EVERY epoch — deserialize from LMDB, then
``dgl.node_subgraph`` induction over the full KG + feature build + root-edge removal
(datasets.py:161-210). That per-epoch reconstruction (not the GPU message passing)
dominates wall-clock. This wrapper calls the underlying dataset's ``__getitem__``
EXACTLY ONCE per index, caches the resulting ``(subgraph, g_label, r_label)`` in host
RAM, and serves the cached tuples for every subsequent epoch.

Faithfulness: byte-identical to the un-materialized path by construction — it calls the
SAME ``_prepare_subgraphs``. There is no per-epoch randomness in the dataset to freeze:
``edge_dropout`` is applied inside the MODEL (rgcn_model.py:27,89,109), and the only
dataset-side edge op is the DETERMINISTIC root-root edge removal (datasets.py:209). The
extraction randomness (``random.sample`` when ``max_nodes_per_hop`` is hit) already
happened once at extraction time and is fixed in the LMDB.

Usage: wrap an existing SubgraphDataset and use ``num_workers=0`` in the DataLoader
(the cache is already in-process; forked workers would each duplicate it). This does not
modify SubgraphDataset and re-exposes the attributes the trainer/evaluator read.
"""
from __future__ import annotations

from torch.utils.data import Dataset


class MaterializedSubgraphDataset(Dataset):
    """Wrap a SubgraphDataset; precompute every item once into host RAM.

    Any attribute the trainer/evaluator/wrappers read off the dataset that is NOT
    defined on this wrapper (``num_rels``, ``aug_num_rels``, ``n_feat_dim``,
    ``max_n_label``, ``ssp_graph``, ``graph``, ``id2entity``, ``id2relation``,
    ``file_name`` ...) forwards to ``base`` via ``__getattr__`` (codex 019f1fdc:
    more robust than a hand-maintained whitelist).
    """

    def __init__(self, base: Dataset, *, verbose: bool = True) -> None:
        # NB: set via object.__setattr__-free plain assignment is fine; __getattr__
        # only fires for MISSING attributes, so `base`/`_cache` never recurse.
        self.base = base
        n = len(base)
        self._cache = [base[i] for i in range(n)]   # single pass: one _prepare_subgraphs/idx
        if verbose:
            print(f"[sumgnn] materialized {n} subgraphs into RAM (per-epoch reconstruction eliminated)",
                  flush=True)

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, index: int):
        return self._cache[index]

    def __getattr__(self, name: str):
        # Only called when `name` is not found on the instance/class. Forward to base.
        return getattr(self.__dict__["base"], name)


__all__ = ["MaterializedSubgraphDataset"]
