"""Stage 4 - the H_base seam. Fills AdapterBatch.h_base per pair's mediators so the
adapter (M_A.M_B + read-out) stays UNCHANGED while the base representation swaps:

  * StructuralHBaseProvider  - adapter-alone: a lightweight MLP over [type_onehot |
    norm_log_degree]; DDI-fact-independent (leak-free trivially).
  * BackboneHBaseProvider    - +KG backbone (RGCN/EmerGNN/KnowDDI): the backbone's
    per-node states at the mediators (added in the next increment).

Contract (both):
  d                      : the H_base dimension (the adapter's cfg.d follows it)
  encode(fact_context)   : refresh any per-epoch KG encoding (structural = no-op)
  h_base(med_idx) -> [M, d] : gather the base rows for the batch's mediators (KG idx)

The harness calls encode(fact_context) once per train epoch / eval pass, then the
model builds batches whose h_base comes from h_base(med_idx). fact_context carries the
leak-free DDI facts (the scored pair's own edge is asserted absent by the protocol).
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn as nn


class HBaseProvider(nn.Module):
    """Base contract. Subclasses set self._d and implement encode/h_base."""

    @property
    def d(self) -> int:
        return self._d

    def encode(self, fact_context) -> None:
        """Refresh any per-epoch KG encoding from the leak-free DDI facts. No-op by default."""

    def h_base(self, med_idx: torch.Tensor) -> torch.Tensor:   # [M] KG idx -> [M, d]
        raise NotImplementedError


class StructuralHBaseProvider(HBaseProvider):
    """Lightweight structural H_base: MLP over [type_onehot | norm_log_degree].
    Does not use the DDI fact context (structural mediator profile only)."""

    def __init__(self, node_feat: torch.Tensor, d: int):
        super().__init__()
        self.register_buffer("node_feat", node_feat)           # [n_nodes, n_types+1]
        self._d = int(d)
        self.mlp = nn.Sequential(nn.Linear(node_feat.shape[1], d), nn.ReLU(), nn.Linear(d, d))

    def h_base(self, med_idx: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.node_feat[med_idx])               # [M, d]


class ZeroHBaseProvider(HBaseProvider):
    """h_base = 0: the adapter contributes PURE M_A.M_B (no structural base). Used in
    the +backbone composer (design R), where the backbone supplies the base pair
    representation and the adapter is only the semantic correction z_uv."""

    def __init__(self, d: int):
        super().__init__()
        self._d = int(d)

    def h_base(self, med_idx: torch.Tensor) -> torch.Tensor:
        return torch.zeros(med_idx.shape[0], self._d, device=med_idx.device)


class BackboneHBaseProvider(HBaseProvider):
    """+KG backbone H_base: the backbone's per-node states at the mediators. Increment 2
    = RGCN (a whole-KG node encoder), FROZEN. encode(fact_context) runs the backbone's
    KG forward once per call and caches the node matrix; h_base gathers mediator rows via
    a KGStore-idx -> backbone-idx map (missing -> zero, logged; hard-fail if coverage low).

    JOINT mode is NOT supported here yet: reusing one encoded graph across minibatch
    backwards double-backprops (codex). JOINT needs a re-encode-per-minibatch schedule.
    Only backbones that expose whole-KG per-node states fit this seam (RGCN); pair/subgraph
    backbones (EmerGNN/KnowDDI) need a different, pair-conditioned provider."""

    def __init__(self, backbone_key: str, kg_store, task, hp: dict, freeze: bool = True):
        super().__init__()
        from model.wrappers import WRAPPERS                    # local import: avoid cycle
        self.freeze = bool(freeze)
        self.wrapper = WRAPPERS[backbone_key]()
        ckpt = hp.get("backbone_ckpt")
        if ckpt:                                               # rebuild backbone from the ckpt
            from model.checkpoints import load_checkpoint      # bundle's own hp/task, NOT the
            from specs import TaskSpec                          # adapter run hp (codex: avoid drift)
            b = load_checkpoint(ckpt)
            bb_task = (TaskSpec.binary() if b.get("task_kind") == "binary"
                       else TaskSpec.multiclass(int(b["n_classes"])))
            bb_hp = {**(b.get("hp") or {}),
                     **{k: hp[k] for k in ("seed", "dataset", "fold") if k in hp}}
            self.wrapper.setup(bb_task, bb_hp)                 # backbone architecture from bundle
            self.wrapper.model.load_state_dict(b["state"])
            print(f"[hbase] {backbone_key}: rebuilt from ckpt bundle "
                  f"(task={b.get('task_kind')}, n_classes={b.get('n_classes')}) <- {ckpt}")
        else:                                                  # no ckpt: adapter hp (plumbing only)
            self.wrapper.setup(task, hp)
            if self.freeze:
                print(f"[hbase] WARN {backbone_key}: FROZEN with no backbone_ckpt "
                      f"(random-init; plumbing only, numbers not meaningful)")
        self.device = self.wrapper.device
        if self.freeze:
            for p in self.wrapper.model.parameters():
                p.requires_grad_(False)
        with torch.no_grad():                                  # d_bb from a dummy bio-graph encode
            self.wrapper.model.eval()
            h0 = self.wrapper.model.encode(self.wrapper.x, self.wrapper.bio_ei, self.wrapper.bio_et)
        self._d = int(h0.shape[1])
        self._H: torch.Tensor | None = None
        # coverage contract: KGStore idx -> backbone idx (-1 = absent), never drop mediators
        id2i = self.wrapper.id2i
        nodes = kg_store.idx2node
        mp = np.fromiter((id2i.get(str(n), -1) for n in nodes.tolist()), dtype=np.int64,
                         count=len(nodes))
        cov = int((mp >= 0).sum()); frac = cov / max(len(nodes), 1)
        print(f"[hbase] {backbone_key} node coverage {cov}/{len(nodes)} ({frac:.1%}); "
              f"missing mediators -> zero h_base (d_bb={self._d})")
        if frac < 0.5:
            raise ValueError(f"backbone covers only {frac:.1%} of KG nodes; wrong KG pairing?")
        self._map = torch.from_numpy(mp).to(self.device)

    def encode(self, fact_context) -> None:
        aug_ei, aug_et = self.wrapper._aug_graph(fact_context)   # add leak-free DDI facts
        ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
        self.wrapper.model.eval() if self.freeze else self.wrapper.model.train()
        with ctx:
            self._H = self.wrapper.model.encode(self.wrapper.x, aug_ei, aug_et)  # [n_bb, d_bb]

    def h_base(self, med_idx: torch.Tensor) -> torch.Tensor:
        if self._H is None:
            raise RuntimeError("call encode(fact_context) before h_base()")
        bb = self._map[med_idx]                                # [M], -1 = absent in backbone KG
        out = self._H.new_zeros(med_idx.shape[0], self._d)
        ok = bb >= 0
        if bool(ok.any()):
            out[ok] = self._H[bb[ok]]
        return out


__all__ = ["HBaseProvider", "StructuralHBaseProvider", "ZeroHBaseProvider",
           "BackboneHBaseProvider"]
