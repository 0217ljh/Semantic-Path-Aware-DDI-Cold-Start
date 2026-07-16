"""Stage 4 - AdapterBackboneComposer (design R): the M_A.M_B adapter as an additive
low-rank CORRECTION to a backbone's pair representation. A single module wrapping the
backbone (PEFT-LoRA-style engineering), but the correction is at the READOUT with an
EXTERNAL KG-semantic signal, not a weight-level BA on backbone activations.

    p_bb'^(s) = LayerNorm( p_bb^(s) + gamma_s * W_s(z_uv) )        score = Head([p_bb'...])
    z_uv = the adapter's pooled PURE M_A.M_B pair representation (ZeroHBaseProvider).

Phase 1a: frozen backbone + `last` (one stream = the backbone's native pre-scorer pair
representation, PairEncoding.pair_repr). JOINT (grad through the backbone) and `default`
(multi-stream) come next. The adapter side reuses AdapterRankModel as a z_uv factory
(its _build_batch + adapter.pair_repr); the composer adds W/gamma/LN + a fusion head.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ADAPTER_ROOT = Path(__file__).resolve().parents[1]
if str(_ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_ROOT))

from .losses import AdapterLoss                               # noqa: E402
from .geom_loss import neighborhood_preservation_loss         # noqa: E402
from .rank_model import AdapterRankModel                      # noqa: E402
from model.contracts import (EpochData, PairEncoding, RankModel,  # noqa: E402
                             ScoringContext, TrainEpochOutput)

#: composer-specific tunables (adapter-side hp pass through to AdapterRankModel)
_C_DEFAULTS = dict(backbone_freeze=True, adapter_layers="last",   # last | default (phase 2)
                   fusion_hidden=256, fusion_gamma_init=1e-2, fusion_dropout=0.2,
                   #  fusion_gamma_init: R residual scale (small -> start ~backbone-alone).
                   #  DISTINCT from the adapter's own gamma_init (its gamma_t); do NOT alias.
                   core_proj_rank=0,   # >0 (last+frozen only): project the correction into the
                   #                     ORTHOGONAL COMPLEMENT of the frozen backbone's top-r_core
                   #                     variance core (V_core from TRAIN reps). Unsupervised geometric
                   #                     inductive bias so c writes outside the topology core.
                   nbr_loss_weight=0.0, nbr_k=10, nbr_temp=0.5,   # >0 (last only): unsupervised
                   #                     z_uv neighborhood-preservation loss (geom_loss), makes p_bb'
                   #                     preserve z_uv's local geometry. NO shared-MBE labels.
                   lr=1e-3, weight_decay=5e-4, batch_size=64, log_step_every=50, seed=42)


class AdapterBackboneComposer(RankModel):
    # -- setup ---------------------------------------------------------------
    def setup(self, task, hp: dict) -> None:
        from model.wrappers import WRAPPERS                    # local import: avoid cycle
        self.task = task
        self.hp = {**_C_DEFAULTS, **(hp or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(int(self.hp["seed"])); np.random.seed(int(self.hp["seed"]))
        self.frozen = bool(self.hp["backbone_freeze"])
        self.multi = str(self.hp.get("adapter_layers", "last")) == "default"   # last | default
        if self.hp.get("adapter_layers", "last") not in ("last", "default"):
            raise ValueError(f"adapter_layers must be 'last'|'default', got {self.hp['adapter_layers']!r}")

        # -- backbone channel (provides p_bb via encode_pairs.pair_repr) -----
        key = str(self.hp["backbone"])
        self.backbone = WRAPPERS[key]()
        ckpt = self.hp.get("backbone_ckpt")
        if ckpt:                                              # rebuild from the ckpt bundle's meta
            from model.checkpoints import load_checkpoint
            from specs import TaskSpec
            b = load_checkpoint(ckpt)
            bb_task = (TaskSpec.binary() if b.get("task_kind") == "binary"
                       else TaskSpec.multiclass(int(b["n_classes"])))
            bb_hp = {**(b.get("hp") or {}),
                     **{k: self.hp[k] for k in ("seed", "dataset", "fold") if k in self.hp}}
            self.backbone.setup(bb_task, bb_hp)
            self.backbone.model.load_state_dict(b["state"])
            print(f"[composer] backbone {key}: rebuilt from ckpt bundle <- {ckpt}")
        else:
            self.backbone.setup(task, self.hp)
            if self.frozen:
                print(f"[composer] WARN {key}: FROZEN with no backbone_ckpt (random-init; plumbing)")
        if self.frozen:
            for p in self.backbone.model.parameters():
                p.requires_grad_(False)
        if self.multi and not hasattr(self.backbone, "pair_streams"):     # default: per-stream
            raise NotImplementedError(
                f"adapter_layers='default' needs {key}.pair_streams(pairs, ctx)->list[[N,d_s]]; "
                f"the {key} wrapper does not expose it yet")
        if (not self.frozen) and (not self.multi) and not hasattr(self.backbone, "pair_forward"):
            raise NotImplementedError(                          # joint 'last' needs a grad forward
                f"joint (last) needs {key}.pair_forward(pairs, ctx)->[N,d] (grad-enabled); "
                f"the {key} wrapper does not expose it yet")

        # -- adapter channel (z_uv factory: pure M_A.M_B via ZeroHBaseProvider)
        self.adapter_model = AdapterRankModel()
        self.adapter_model.setup(task, {**self.hp, "hbase": "zero"})
        self.adapter = self.adapter_model.adapter             # the adapter module
        if not (hasattr(self.adapter, "pair_repr") and hasattr(self.adapter, "z_dim")):
            raise NotImplementedError(                        # formal seam requirement (codex)
                f"composer needs the adapter to expose pair_repr(batch)->(z_uv,beta) + z_dim; "
                f"{type(self.adapter).__name__} (arm_mode={self.hp.get('arm_mode')!r}) does not. "
                f"Add pair_repr/z_dim to that adapter variant (e.g. FactorizedAdapter for pd1).")
        z_dim = self.adapter.z_dim

        # -- fusion (design R). 'last': correct the single final backbone stream.
        # 'default': correct EVERY stream s (its own W_s/gamma_s/LN_s), concat -> head.
        g0 = float(self.hp["fusion_gamma_init"])
        if not self.multi:
            d_pbb = self._probe_pbb_dim()
            self.W = nn.Linear(z_dim, d_pbb).to(self.device)
            self.gamma = nn.Parameter(torch.tensor(g0, device=self.device))
            self.ln = nn.LayerNorm(d_pbb).to(self.device)
            head_in = d_pbb
            print(f"[composer] R last: p_bb({d_pbb}) + gamma*W(z_uv[{z_dim}]) -> LN -> head "
                  f"(backbone={key}, frozen={self.frozen})")
        else:
            dims = self._probe_stream_dims()
            self.stream_dims = dims
            self.W_list = nn.ModuleList([nn.Linear(z_dim, d) for d in dims]).to(self.device)
            self.gamma_list = nn.ParameterList(
                [nn.Parameter(torch.tensor(g0, device=self.device)) for _ in dims])
            self.ln_list = nn.ModuleList([nn.LayerNorm(d) for d in dims]).to(self.device)
            head_in = int(sum(dims))
            print(f"[composer] R default: {len(dims)} streams dims={dims} + gamma_s*W_s(z_uv"
                  f"[{z_dim}]) -> LN_s -> concat({head_in}) -> head "
                  f"(backbone={key}, frozen={self.frozen})")
        n_out = 1 if task.is_binary else task.n_classes
        self.head = nn.Sequential(
            nn.Linear(head_in, int(self.hp["fusion_hidden"])), nn.ReLU(),
            nn.Dropout(float(self.hp["fusion_dropout"])),
            nn.Linear(int(self.hp["fusion_hidden"]), n_out)).to(self.device)
        self.crit = AdapterLoss(self.adapter_model.cfg,
                                "binary" if task.is_binary else "multiclass")

        params = self._adapter_params() + list(self.head.parameters()) + self._fusion_params()
        if not self.frozen:
            params += list(self.backbone.model.parameters())  # JOINT: backbone trains too
        self.opt = torch.optim.AdamW(params, lr=float(self.hp["lr"]),
                                     weight_decay=float(self.hp["weight_decay"]))

        # core-complement projection (unsupervised geometric bias): V_core = frozen backbone
        # top-r_core variance basis from TRAIN reps; computed lazily on the first train_epoch.
        self.core_rank = int(self.hp.get("core_proj_rank", 0))
        self.V_core = None                                    # [d_pbb, r_core] on device, or None
        if self.core_rank > 0 and (self.multi or not self.frozen):
            raise NotImplementedError("core_proj_rank>0 is supported only for frozen + last "
                                      "(V_core needs a fixed frozen backbone and a single stream)")
        # unsupervised z_uv neighborhood-preservation loss (part 2)
        self.nbr_w = float(self.hp.get("nbr_loss_weight", 0.0))
        self.nbr_k = int(self.hp.get("nbr_k", 10))
        self.nbr_temp = float(self.hp.get("nbr_temp", 0.5))
        if self.nbr_w > 0 and self.multi:
            raise NotImplementedError("nbr_loss_weight>0 is supported only for adapter_layers='last'")

    def _ensure_core_basis(self, p_bb_all: torch.Tensor) -> None:
        """Compute V_core once from the frozen backbone's TRAIN pair reps (top-r_core right
        singular vectors of the centered reps). No-op if core_proj is off or already computed."""
        if self.core_rank <= 0 or self.V_core is not None:
            return
        with torch.no_grad():
            X = p_bb_all.detach().to(self.device).float()
            Xc = X - X.mean(0, keepdim=True)
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)   # Vh [d, d]
            self.V_core = Vh[:self.core_rank].t().contiguous()     # [d, r_core]
        print(f"[composer] core-complement projection ON: r_core={self.core_rank} "
              f"(V_core from {len(p_bb_all)} train reps)")

    def _apply_core_proj(self, c: torch.Tensor) -> torch.Tensor:
        """Project the correction c into the orthogonal complement of V_core (c - c V_core V_core^T)."""
        if self.V_core is None:
            if self.core_rank > 0:                            # fail loud: never eval with proj silently off
                raise RuntimeError("core_proj_rank>0 but V_core is unset (compute it via a train "
                                   "epoch, or load a ckpt that persisted V_core)")
            return c
        return c - (c @ self.V_core) @ self.V_core.t()

    def _fusion_params(self) -> list:
        """The design-R correction params (per mode): W/gamma/LN (last) or the S-stream lists."""
        if not self.multi:
            return [self.gamma] + list(self.W.parameters()) + list(self.ln.parameters())
        return (list(self.gamma_list) + list(self.W_list.parameters())
                + list(self.ln_list.parameters()))

    def _adapter_params(self) -> list:
        p = list(self.adapter.parameters())
        if self.adapter_model.pathway is not None:
            p += list(self.adapter_model.pathway.parameters())
        return p

    def _probe_pbb_dim(self) -> int:
        """Backbone pair-repr width, from a dummy 1-pair encode on 2 known drugs. Falls back to
        pair_forward when encode_pairs exposes no pair_repr (e.g. MKG-FENN, whose fused readout
        includes the molecular channel we drop -> the KG-only rep comes from pair_forward)."""
        dks = list(self.backbone.known_drugs())[:2]
        pr = np.array([[dks[0], dks[1]]], dtype=object)
        ctx = ScoringContext("fact_kg", np.empty((0, 3), object))
        enc = self.backbone.encode_pairs(pr, ctx)
        if enc.pair_repr is not None:
            return int(enc.pair_repr.shape[1])
        if hasattr(self.backbone, "pair_forward"):
            with torch.no_grad():
                return int(self.backbone.pair_forward(pr, ctx).shape[1])
        raise ValueError(f"backbone {self.hp['backbone']} exposes neither pair_repr nor "
                         f"pair_forward (needed for design R); expose one in the wrapper first")

    def _probe_stream_dims(self) -> list:
        """Per-stream widths [d_1,..,d_S], from a dummy 1-pair pair_streams on 2 known drugs."""
        dks = list(self.backbone.known_drugs())[:2]
        pr = np.array([[dks[0], dks[1]]], dtype=object)
        with torch.no_grad():
            streams = self.backbone.pair_streams(pr, ScoringContext("fact_kg", np.empty((0, 3), object)))
        return [int(s.shape[1]) for s in streams]

    # -- fusion forward ------------------------------------------------------
    def _encode_backbone(self, pairs: np.ndarray, ctx: ScoringContext) -> torch.Tensor:
        """ONE backbone forward over ALL pairs (frozen: no_grad, node states are constant
        within a run) -> p_bb [N, d_pbb]. Avoids re-encoding the whole KG per minibatch.
        (JOINT would need per-minibatch re-encode for grad; phase 1b.)"""
        with (torch.no_grad() if self.frozen else contextlib.nullcontext()):
            enc = self.backbone.encode_pairs(np.asarray(pairs), ctx)
            if enc.pair_repr is not None:
                p_bb = np.asarray(enc.pair_repr)
                return torch.as_tensor(p_bb, dtype=torch.float32, device=self.device)
            if hasattr(self.backbone, "pair_forward"):     # KG-only rep (e.g. MKG-FENN)
                return self.backbone.pair_forward(np.asarray(pairs), ctx).detach().to(self.device).float()
        raise ValueError(f"backbone {self.hp['backbone']} returned no pair_repr and has no pair_forward")

    def _encode_streams(self, pairs: np.ndarray, ctx: ScoringContext) -> list:
        """ONE backbone forward over ALL pairs -> per-stream reps [ [N,d_s], .. ] ('default').
        frozen: no_grad (constant); joint would re-encode per minibatch instead."""
        with (torch.no_grad() if self.frozen else contextlib.nullcontext()):
            streams = self.backbone.pair_streams(np.asarray(pairs), ctx)   # list of [N, d_s]
        return list(streams)

    def _z_uv(self, pairs_sel: np.ndarray):
        """The adapter's pure M_A.M_B pooled pair rep z_uv [B, z_dim] (ZeroHBaseProvider)."""
        batch = self.adapter_model._build_batch(pairs_sel)    # mediators + z_m + pathway, h_base=0
        return self.adapter.pair_repr(batch)                  # (z_uv [B, z_dim], beta: flat [M,lam]
        #                                                        or pd1 [M,K,H]; only train_epoch reads it)

    def _fuse_last(self, pairs_sel: np.ndarray, p_bb_sel: torch.Tensor):
        """Design-R on the single final stream: LN(p_bb + gamma*W(z_uv)) -> head. When core_proj
        is on, the correction is projected into the backbone core's orthogonal complement first."""
        z_uv, beta = self._z_uv(pairs_sel)
        c = self._apply_core_proj(self.gamma * self.W(z_uv))
        p_corr = self.ln(p_bb_sel + c)
        return self.head(p_corr), beta                        # [B, n_out], beta (flat[M,lam]|pd1[M,K,H])

    def _fuse_last_geom(self, pairs_sel: np.ndarray, p_bb_sel: torch.Tensor):
        """_fuse_last + tensors for the nbr-preservation loss. z_uv is the FIXED teacher geometry
        (detached), so the nbr loss shapes only the fusion (W/gamma/LN) to imprint z_uv's local
        neighbor structure onto p_bb' -- NOT the adapter that produces z_uv (codex teacher-student).
        The task logits still use the attached p_corr (adapter trains on the task as usual). The
        detached-teacher p_corr has the SAME numerical value, only a different grad graph."""
        z_uv, beta = self._z_uv(pairs_sel)
        p_corr = self.ln(p_bb_sel + self._apply_core_proj(self.gamma * self.W(z_uv)))   # task path
        p_corr_nbr = self.ln(p_bb_sel + self._apply_core_proj(self.gamma * self.W(z_uv.detach())))
        return self.head(p_corr), beta, z_uv.detach(), p_corr_nbr

    def _fuse_default(self, pairs_sel: np.ndarray, streams_sel: list):
        """Design-R on EVERY stream: LN_s(p_s + gamma_s*W_s(z_uv)) then concat -> head."""
        z_uv, beta = self._z_uv(pairs_sel)
        corr = [self.ln_list[s](streams_sel[s] + self.gamma_list[s] * self.W_list[s](z_uv))
                for s in range(len(streams_sel))]
        return self.head(torch.cat(corr, dim=-1)), beta       # [B, n_out], beta (flat[M,lam]|pd1[M,K,H])

    def _minibatches(self, n: int, rng, shuffle: bool):
        order = rng.permutation(n) if shuffle else np.arange(n)
        bs = int(self.hp["batch_size"])
        for s in range(0, n, bs):
            yield order[s:s + bs]

    # -- RankModel contract --------------------------------------------------
    def train_epoch(self, epoch: EpochData, rng) -> TrainEpochOutput:
        pairs, labels = epoch.target_pairs, epoch.target_labels
        self._train_mode(True)
        # frozen: encode the whole KG ONCE (no_grad, constant) then slice; joint: re-encode
        # per minibatch WITH grad (fresh graph each backward -> no double-backward, codex).
        if self.multi:
            streams_all = self._encode_streams(pairs, epoch.fact_context) if self.frozen else None
        else:
            p_bb_all = self._encode_backbone(pairs, epoch.fact_context) if self.frozen else None
            self._ensure_core_basis(p_bb_all)                 # core-complement basis (once, TRAIN reps)
        tot, seen, step = 0.0, 0, 0
        n_steps = (len(pairs) + int(self.hp["batch_size"]) - 1) // int(self.hp["batch_size"])
        for sel in self._minibatches(len(pairs), rng, shuffle=True):
            if self.multi:
                streams = [s[sel] for s in streams_all] if self.frozen \
                    else self.backbone.pair_streams(pairs[sel], epoch.fact_context)  # grad
                logits, beta = self._fuse_default(pairs[sel], streams)
            else:
                p_bb = p_bb_all[sel] if self.frozen \
                    else self.backbone.pair_forward(pairs[sel], epoch.fact_context)  # [b,d] grad
                if self.nbr_w > 0:                            # part-2: need z_uv + p_corr for nbr loss
                    logits, beta, z_uv_b, p_corr_b = self._fuse_last_geom(pairs[sel], p_bb)
                else:
                    logits, beta = self._fuse_last(pairs[sel], p_bb)
            y = torch.as_tensor(labels[sel], device=self.device)
            if hasattr(self.adapter, "aux_reg"):              # pd1: task loss + factorized reg,
                loss, comps = self.crit(logits, y, None)      # skipping the flat beta-entropy
                loss = loss + float(self.hp.get("reg_weight", 0.01)) * self.adapter.aux_reg(beta)
            else:                                             # flat adapter: beta [M,lam] ->
                loss, comps = self.crit(logits, y, beta)      # beta-entropy inside AdapterLoss
            if self.nbr_w > 0 and not self.multi:             # unsupervised z_uv geometry preservation
                loss = loss + self.nbr_w * neighborhood_preservation_loss(
                    z_uv_b, p_corr_b, self.nbr_k, self.nbr_temp)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            tot += float(loss.item()) * len(sel); seen += len(sel); step += 1
            if step % int(self.hp["log_step_every"]) == 0:
                print(f"  [composer][step {step}/{n_steps}] loss={tot/seen:.4f} "
                      f"gamma={self._gamma_repr()}")
        return TrainEpochOutput(mean_loss=(tot / max(seen, 1)), n_targets=len(labels))

    def _gamma_repr(self) -> str:
        """Human-readable fusion-gamma for logging (single 'last' vs per-stream 'default')."""
        if not self.multi:
            return f"{float(self.gamma):.3f}"
        return "[" + ",".join(f"{float(g):.3f}" for g in self.gamma_list) + "]"

    @torch.no_grad()
    def encode_pairs(self, pairs, context: ScoringContext) -> PairEncoding:
        pairs = np.asarray(pairs)
        self._train_mode(False)
        if self.multi:
            streams_all = self._encode_streams(pairs, context)   # ONE backbone encode (per stream)
        else:
            p_bb_all = self._encode_backbone(pairs, context)     # ONE backbone encode
        out = []
        rng = np.random.default_rng(0)
        for sel in self._minibatches(len(pairs), rng, shuffle=False):
            if self.multi:
                logits, _ = self._fuse_default(pairs[sel], [s[sel] for s in streams_all])
            else:
                logits, _ = self._fuse_last(pairs[sel], p_bb_all[sel])
            out.append(logits.cpu().numpy().astype(np.float32))
        n_out = 1 if self.task.is_binary else self.task.n_classes
        logits = np.concatenate(out, axis=0) if out else np.zeros((0, n_out), np.float32)
        return PairEncoding(pair_ids=pairs, logits=logits,
                            repr_kind="adapter_backbone_R", repr_stage="fused_head")

    def _train_mode(self, train: bool) -> None:
        mods = [self.adapter, self.head]
        mods += [self.W_list, self.ln_list] if self.multi else [self.W, self.ln]
        for m in mods:
            m.train(train)
        if self.adapter_model.pathway is not None:
            self.adapter_model.pathway.train(train)
        self.backbone.model.train(train and not self.frozen)

    def known_drugs(self) -> set:
        return self.adapter_model.known_drugs() & self.backbone.known_drugs()

    def effective_hp(self) -> dict:
        return dict(self.hp)

    def state_dict(self) -> dict:
        def sd(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        st = {"adapter": sd(self.adapter), "head": sd(self.head), "multi": self.multi}
        if self.multi:                                        # per-stream W_s/gamma_s/LN_s
            st["W_list"] = sd(self.W_list); st["ln_list"] = sd(self.ln_list)
            st["gamma_list"] = [g.detach().cpu().clone() for g in self.gamma_list]
        else:
            st["W"] = sd(self.W); st["ln"] = sd(self.ln)
            st["gamma"] = self.gamma.detach().cpu().clone()
        if self.adapter_model.pathway is not None:
            st["pathway"] = sd(self.adapter_model.pathway)
        if not self.frozen:
            st["backbone"] = sd(self.backbone.model)
        if self.V_core is not None:                           # core-complement basis (derived, but
            st["V_core"] = self.V_core.detach().cpu().clone()  # persist so eval-after-load matches)
        return st

    def load_state_dict(self, state: dict) -> None:
        if bool(state.get("multi", False)) != self.multi:
            raise ValueError(f"state multi={state.get('multi')} != composer multi={self.multi}")
        self.adapter.load_state_dict(state["adapter"])
        self.head.load_state_dict(state["head"])
        if self.multi:
            self.W_list.load_state_dict(state["W_list"]); self.ln_list.load_state_dict(state["ln_list"])
            if len(state["gamma_list"]) != len(self.gamma_list):     # fail loud like W_list/ln_list
                raise ValueError(f"gamma_list len {len(state['gamma_list'])} != {len(self.gamma_list)} "
                                 f"(stream-count mismatch)")
            with torch.no_grad():
                for g, gs in zip(self.gamma_list, state["gamma_list"]):
                    g.copy_(gs.to(g))
        else:
            self.W.load_state_dict(state["W"]); self.ln.load_state_dict(state["ln"])
            with torch.no_grad():
                self.gamma.copy_(state["gamma"].to(self.gamma))
        if self.adapter_model.pathway is not None and "pathway" in state:
            self.adapter_model.pathway.load_state_dict(state["pathway"])
        if not self.frozen and "backbone" in state:
            self.backbone.model.load_state_dict(state["backbone"])
        if "V_core" in state:                                 # restore core-complement basis
            self.V_core = state["V_core"].to(self.device)

    def adapter_state(self):
        st = self.adapter.adapter_state()
        if self.multi:
            st["gamma_fusion"] = torch.stack([g.detach().cpu() for g in self.gamma_list])
        else:
            st["gamma_fusion"] = self.gamma.detach().cpu()
        return st


__all__ = ["AdapterBackboneComposer"]
