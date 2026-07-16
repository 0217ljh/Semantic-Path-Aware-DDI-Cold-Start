"""NBFNet v1.72 trainer — dual-source signed-bilinear interference (subclass of v1.71).

Inherits v1.71 (AMP/TF32/eval-cadence/batched dual-source BF). Adds:
  - SIGNED model (NBFNetV172) with action-signed edges.
  - Interference readout: shared endpoint RMSNorm on the two endpoint fields, then
    combine ∈ {additive (u+v) , hadamard (u⊙v) = interference}.
  - Action edge-sign threading (epoch / eval), inverse edges inherit same sign.
  - Signedness diagnostic (neg-coord fraction of u,v on eval).

2×2 attribution: activation ∈ {relu, signed} × combine ∈ {additive, hadamard};
D = signed+hadamard is the interference core. Action kept ON in all cells (use it
as a constant input; --use-action off is a separate ablation).

Only the S2 batched path is supported (query mask is a no-op there); a maskable
batch raises, by design.
"""
from __future__ import annotations

import time  # noqa: F401 (parity with siblings)
from pathlib import Path  # noqa: F401

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from my_code.models.nbfnet_v1_71.nbfnet_trainer import NBFNetTrainerV171
from my_code.models.nbfnet_v1_72.nbfnet_model import NBFNetV172


class NBFNetTrainerV172(NBFNetTrainerV171):
    """v1.71 trainer + signed-field action-aware interference readout."""

    def __init__(
        self,
        *args,
        activation: str = "signed",      # "signed" (RMSNorm) or "relu" (=v1.71 field)
        combine: str = "hadamard",       # "hadamard" (interference) or "additive"
        use_action: bool = True,         # inject action edge signs
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.activation = str(activation)
        self.combine = str(combine)
        self.use_action = bool(use_action)
        # sign state
        self._base_kg_sign_t = None      # (n_base_edges,) torch
        self._eval_edge_sign = None      # (n_eval_edges,) torch (pre-augment)
        self._cur_epoch_edge_sign = None # (n_epoch_edges,) torch (pre-augment)
        self._cur_training = False
        # signedness diagnostic buffers
        self._diag_u_neg: list[float] = []
        self._diag_v_neg: list[float] = []

    # ------------------------------------------------------------------
    # model build (V172 with signed activation) + TF32
    # ------------------------------------------------------------------
    def init_model(self) -> None:
        if self.n_nodes is None or self.n_base_rel is None:
            raise RuntimeError("setup_graph must be called before init_model")
        if self.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model = NBFNetV172(
            n_nodes=self.n_nodes,
            n_base_rel=self.n_base_rel,
            d=self.d,
            n_layers=self.n_layers,
            mlp_hidden=self.mlp_hidden,
            ddi_rel_id=self.ddi_rel_id,
            signed_act=(self.activation == "signed"),
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(
            f"[nbfnet-v1.72] model init. d={self.d} L={self.n_layers} "
            f"n_params={n_params:,} | activation={self.activation} combine={self.combine} "
            f"use_action={self.use_action} | tf32={self.use_tf32} amp={self.use_amp}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # KG setup: store action edge signs (aligned with edge order)
    # ------------------------------------------------------------------
    def setup_graph(
        self,
        base_kg_triplets,
        train_ddi_triplets,
        n_nodes,
        n_base_rel,
        ddi_rel_id,
        base_kg_signs=None,              # (len(base_kg_triplets),) +1/-1, aligned
    ) -> None:
        super().setup_graph(
            base_kg_triplets, train_ddi_triplets, n_nodes, n_base_rel, ddi_rel_id
        )
        n_base = len(self.base_kg_triplets)
        n_ddi = len(self.train_ddi_triplets)
        if self.use_action and base_kg_signs is not None:
            bs = np.asarray(base_kg_signs, dtype=np.float32)
            if len(bs) != n_base:
                raise ValueError(f"base_kg_signs len {len(bs)} != n_base {n_base}")
            self._base_kg_sign_t = torch.from_numpy(bs).to(self.device)
            # eval KG order = [train_ddi (neutral +1), base_kg]
            eval_sign = np.concatenate([np.ones(n_ddi, np.float32), bs])
        else:
            self._base_kg_sign_t = torch.ones(n_base, device=self.device)
            eval_sign = np.ones(n_ddi + n_base, np.float32)
        self._eval_edge_sign = torch.from_numpy(np.asarray(eval_sign, np.float32)).to(self.device)
        n_neg = int((self._base_kg_sign_t < 0).sum().item())
        print(f"[nbfnet-v1.72] action signs. base_neg_edges={n_neg}/{n_base} "
              f"use_action={self.use_action}", flush=True)

    # ------------------------------------------------------------------
    # per-epoch sign (epoch KG = [fact (neutral +1), base_kg] per shuffle_train)
    # ------------------------------------------------------------------
    def _build_epoch_kg(self, epoch):
        (edge_src, edge_dst, edge_rel), epoch_targets = super()._build_epoch_kg(epoch)
        n_total = int(edge_src.numel())
        n_base = len(self.base_kg_triplets)
        n_fact = n_total - n_base
        if n_fact < 0:
            raise RuntimeError(f"epoch edges {n_total} < base {n_base}; order assumption broken")
        if self.use_action:
            self._cur_epoch_edge_sign = torch.cat(
                [torch.ones(n_fact, device=self.device), self._base_kg_sign_t], dim=0
            )
        else:
            self._cur_epoch_edge_sign = torch.ones(n_total, device=self.device)
        return (edge_src, edge_dst, edge_rel), epoch_targets

    # ------------------------------------------------------------------
    # dispatch: force batched (signed interference only), AMP, S2 guard
    # ------------------------------------------------------------------
    def _score_pairs(self, batch_a, batch_b, aug_edges, training):
        self._cur_training = bool(training)
        if training and self._query_edges_present(aug_edges, batch_a, batch_b):
            raise RuntimeError(
                "v1.72 batched interference requires a no-op query mask (S2). "
                "Got a batch with maskable DDI edges among its drugs."
            )
        if self.use_amp and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                out = self._score_pairs_batched(batch_a, batch_b, aug_edges)
            return out.float()
        return self._score_pairs_batched(batch_a, batch_b, aug_edges)

    # ------------------------------------------------------------------
    # interference readout
    # ------------------------------------------------------------------
    def _score_pairs_batched(self, batch_a, batch_b, aug_edges):
        aug_src, aug_dst, aug_rel = aug_edges
        pre = self._cur_epoch_edge_sign if self._cur_training else self._eval_edge_sign
        aug_sign = torch.cat([pre, pre], dim=0)   # inverse inherits SAME sign (#1)
        if aug_sign.numel() != aug_src.numel():
            raise RuntimeError(
                f"edge-sign misalignment: sign {aug_sign.numel()} vs edges {aug_src.numel()}"
            )

        ua, inv_a = np.unique(batch_a, return_inverse=True)
        ub, inv_b = np.unique(batch_b, return_inverse=True)
        ua_t = torch.as_tensor(ua, device=self.device, dtype=torch.long)
        ub_t = torch.as_tensor(ub, device=self.device, dtype=torch.long)

        hf = self.model.encode_from_sources_signed(
            ua_t, aug_src, aug_dst, aug_rel, aug_sign, already_augmented=True)
        hr = self.model.encode_from_sources_signed(
            ub_t, aug_src, aug_dst, aug_rel, aug_sign, already_augmented=True)

        rows_a = torch.as_tensor(inv_a, device=self.device, dtype=torch.long)
        rows_b = torch.as_tensor(inv_b, device=self.device, dtype=torch.long)
        bt_a = torch.as_tensor(batch_a, device=self.device, dtype=torch.long)
        bt_b = torch.as_tensor(batch_b, device=self.device, dtype=torch.long)

        fa = hf[rows_a, bt_b]                         # (B, d) = h_q(a, b)
        rb = hr[rows_b, bt_a]                         # (B, d) = h_q(b, a)
        u = self.model.endpoint_norm(fa)             # shared norm (#2)
        v = self.model.endpoint_norm(rb)

        if self.combine == "hadamard":
            h_comb = u * v                            # signed bilinear cross-term = interference
        elif self.combine == "additive":
            h_comb = u + v
        else:
            raise ValueError(f"unknown combine={self.combine!r}")

        if not self._cur_training:                   # signedness diagnostic (#3)
            self._diag_u_neg.append(float((u < 0).float().mean().item()))
            self._diag_v_neg.append(float((v < 0).float().mean().item()))

        q = self.model.query.unsqueeze(0).expand(h_comb.size(0), -1)
        z = torch.cat([h_comb, q], dim=-1)
        return self.model.mlp_head(z).squeeze(-1)

    # ------------------------------------------------------------------
    # validation + signedness diagnostic print
    # ------------------------------------------------------------------
    def _validate(self, val_a, val_b, val_y):
        self._diag_u_neg = []
        self._diag_v_neg = []
        auc, ap = super()._validate(val_a, val_b, val_y)
        if self._diag_u_neg:
            print(
                f"[nbfnet-v1.72] signedness: u_neg_frac={np.mean(self._diag_u_neg):.3f} "
                f"v_neg_frac={np.mean(self._diag_v_neg):.3f} "
                f"(activation={self.activation})",
                flush=True,
            )
        return auc, ap
