"""Stage 4 / DDI-LoRA adapter - adapter-ALONE RankModel (step 3 milestone).

Wraps the M_A.M_B adapter as a standalone cold-start predictor in the harness's
RankModel contract, with a lightweight STRUCTURAL H_base (no backbone):

    h^base_m = MLP([type_onehot(m), norm_log_degree(m)])        (deterministic feats)

Per pair (u, v) the shared mediators M_uv come from the static bio-KG neighborhoods
(kg.store / kg.neighborhood / kg.mediators, Part A). The bio-KG carries NO DDI edge
(verified: 59 relations, all drug-protein / het / prime), so mediators are leak-free
by construction and this model ignores the harness fact_context. Frozen z_m loads
from the Part-B cache; the arm relation feature r_tilde is STUBBED (zeros) until the
step-4 path-relation aggregation lands (then W_A/W_B activate).

Mediators are memoized per (u_idx, v_idx) (bio-KG static -> unchanged across epochs);
per-drug neighborhoods are memoized lazily (only dataset drugs, not all KG drugs).

This is our METHOD (adapter), not a baseline wrapper. FROZEN/JOINT backbone modes
(steps 6-7) compose a real backbone's mediator-node states as H_base instead.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ADAPTER_ROOT = Path(__file__).resolve().parents[1]         # code-adapter/
if str(_ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_ROOT))

from kg.store import build_kg_store                          # noqa: E402
from kg.neighborhood import neighborhood_of_idx             # noqa: E402
from kg.mediators import shared_mediators_idx               # noqa: E402
from kg.path_relations import build_traversal_csr, PathRelationCoeffs  # noqa: E402
from kg import relations as _relmod                          # noqa: E402

from .batch import AdapterBatch                              # noqa: E402
from .config import AdapterConfig                            # noqa: E402
from .losses import AdapterLoss                              # noqa: E402
from .modules import DDILoRAAdapter, PathwayFeature          # noqa: E402
from .path_design1 import FactorizedAdapter                   # noqa: E402
from .hbase import (StructuralHBaseProvider, ZeroHBaseProvider,  # noqa: E402
                    BackboneHBaseProvider)
from model.contracts import (EpochData, PairEncoding, RankModel,  # noqa: E402
                             ScoringContext, TrainEpochOutput)

#: tunable hyperparameters (exposed via hp / --set)
_DEFAULTS = dict(
    d=256, lam=4, mlp_hidden=256, dropout=0.2,               # adapter capacity
    rho=0.5, tau=2, med_mode="and", nbhd_L=2, max_med=0,     # mediator / path
    #                     max_med=0 -> ALL mediators (paper-faithful); >0 -> keep the
    #                     top-max_med by (d_a+d_b) asc then degree asc (tractability cap)
    arm_mode="pathway",                                      # pathway | ztext | off | pd1
    #     pathway = typed meta-path atoms (r1, t(x)); ztext = frozen-z_r r_tilde; off = zeros;
    #     pd1 = Path Design 1 factorized node-family x route-variant (the complementarity design).
    #     Ablate against use_node (flat) or node_on/path_on (pd1) to prove both help.
    K=3, H=2, q_rank=8, d_r=32, eta_route=1.0, reg_weight=0.01,  # pd1: family/variant/route sizes
    node_on=True, path_on=True,                              # pd1 ablation: family (node) / variant (route)
    shuffle_route=False, shuffle_node=False,                # pd1 signal-destruction controls (codex):
    #     keep K3H2 structure fixed but misalign the route tokens / z_m across mediators, to prove
    #     the route/node SIGNAL matters (not just the factorized structure). full > shuffle-* => signal real.
    use_node=True,                                           # node term U z_m in beta (ablatable, flat)
    use_mb=True,                                             # M_B typed-prototype residual (ablatable)
    warmstart_u=True, km_tau=1.0,                            # step5: k-means U_t warm-start
    use_aa=True,                                             # step5: Adamic-Adar pooling prior
    gamma_init=1.0,                                          # gamma_t init. adapter-ALONE: the
    #     semantic residual IS the model, so start ~1 (NOT eps=near-identity, which is for a
    #     strong-backbone correction). Small gamma gates the whole M_A.M_B semantic pathway off.
    eta_entropy=0.0,                                          # loss
    lr=1e-3, weight_decay=5e-4, batch_size=64, log_step_every=50,
    encoder="sapbert", seed=42)

#: node_embed / relation_embed cache filename prefix per encoder key
_ZM_PREFIX = {"sapbert": "cambridgeltl-SapBERT",
              "pubmedbert": "microsoft-BiomedNLP-PubMedBERT"}


class _LazyNeighborhoods:
    """nbhd interface (of_idx / of_dist / L) backed by lazy per-drug BFS + memo,
    so kg.mediators.shared_mediators_idx is reused unchanged over only the drugs
    that actually appear (not all 8048 KG drugs)."""

    def __init__(self, kg, L: int):
        self.kg = kg
        self.L = L
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _row(self, i: int):
        r = self._cache.get(i)
        if r is None:
            r = neighborhood_of_idx(self.kg, int(i), self.L)
            self._cache[i] = r
        return r

    def of_idx(self, i: int) -> np.ndarray:
        return self._row(i)[0]

    def of_dist(self, i: int) -> np.ndarray:
        return self._row(i)[1]


class StructuralHBase(nn.Module):
    """Lightweight structural H_base: MLP over [type_onehot | norm_log_degree]."""

    def __init__(self, node_feat: torch.Tensor, d: int):
        super().__init__()
        self.register_buffer("node_feat", node_feat)         # [n_nodes, n_types+1]
        in_dim = node_feat.shape[1]
        self.mlp = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, med_idx: torch.Tensor) -> torch.Tensor:  # [M] -> [M, d]
        return self.mlp(self.node_feat[med_idx])


class AdapterRankModel(RankModel):
    # -- setup ---------------------------------------------------------------
    def setup(self, task, hp: dict) -> None:
        self.task = task
        self.hp = {**_DEFAULTS, **(hp or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.hp["seed"]); np.random.seed(self.hp["seed"])

        self.kg = build_kg_store()
        self.n_types = len(self.kg.type_names)
        self.nbhd = _LazyNeighborhoods(self.kg, int(self.hp["nbhd_L"]))

        # structural node features (type one-hot + normalized log-degree)
        n = self.kg.n_nodes
        deg = np.diff(self.kg.u_indptr).astype(np.float64)
        self._deg = deg                                       # for the max_med cap
        logd = np.log1p(deg)
        feat = np.zeros((n, self.n_types + 1), dtype=np.float32)
        feat[np.arange(n), self.kg.type_id.astype(np.int64)] = 1.0
        feat[:, self.n_types] = ((logd - logd.mean()) / (logd.std() + 1e-8)).astype(np.float32)
        node_feat = torch.from_numpy(feat).to(self.device)

        # frozen z_m (Part-B cache), aligned to KG idx; missing -> zero (logged)
        z, zids = self._load_zm(self.hp["encoder"])
        self.d_z = int(z.shape[1])
        zm_by_idx = np.zeros((n, self.d_z), dtype=np.float32)
        covered = 0
        for row, nid in zip(z, zids):
            i = self.kg.get_idx(str(nid))
            if i is not None:
                zm_by_idx[i] = row; covered += 1
        frac = covered / max(len(zids), 1)                   # provenance check
        if frac < 0.99:
            raise ValueError(f"z_m cache provenance mismatch: only {covered}/{len(zids)} "
                             f"({frac:.1%}) of cached nodes map to the current KG; "
                             f"wrong/stale cache? pin the right one via hp['zm_npz']")
        self._zm = torch.from_numpy(zm_by_idx)               # CPU; gathered per batch
        print(f"[adapter] z_m coverage: {covered}/{len(zids)} cached nodes mapped "
              f"to KG idx ({frac:.1%}, d_z={self.d_z}); non-union KG nodes -> zero z_m")

        # arm-feature machinery: directed traversal CSR (+ typed atoms or frozen z_r)
        self.arm_mode = str(self.hp["arm_mode"])
        self.use_path = self.arm_mode != "off"
        if self.use_path:
            self.pathrel = PathRelationCoeffs(build_traversal_csr(self.kg),
                                              self.kg.type_id, self.n_types)
            if self.arm_mode == "ztext":                     # old frozen-z_r r_tilde (ablation)
                zr = self._load_zr(self.hp["encoder"])
                if zr.shape != (self.pathrel.n_rel, self.d_z):
                    raise ValueError(f"z_r {zr.shape} != ({self.pathrel.n_rel},{self.d_z})")
                self._zr = torch.from_numpy(zr).to(self.device)
                print(f"[adapter] arm=ztext: z_r {tuple(zr.shape)}, rho={self.hp['rho']}")
            else:
                print(f"[adapter] arm=pathway: typed meta-path (r1,t(x)) atoms, rho={self.hp['rho']}")
        else:
            print("[adapter] arm=off (zeros)")

        n_out = 1 if task.is_binary else task.n_classes
        # H_base provider (the seam; its .d becomes the adapter base dim). structural =
        # adapter-alone; backbone providers (RGCN/...) come next increment.
        hbase_mode = str(self.hp.get("hbase", "structural"))
        if hbase_mode == "structural":
            self.hbase_provider = StructuralHBaseProvider(node_feat, int(self.hp["d"])).to(self.device)
        elif hbase_mode == "zero":                            # +backbone (design R): pure M_A.M_B
            self.hbase_provider = ZeroHBaseProvider(int(self.hp["d"])).to(self.device)
        else:                                                 # +KG backbone node-state (superseded)
            self.hbase_provider = BackboneHBaseProvider(
                hbase_mode, self.kg, task, self.hp,
                freeze=bool(self.hp.get("backbone_freeze", True)))
        self.cfg = AdapterConfig(
            n_types=self.n_types, lam=int(self.hp["lam"]), d=self.hbase_provider.d,
            d_z=self.d_z, n_out=n_out, mlp_hidden=int(self.hp["mlp_hidden"]),
            rho=float(self.hp["rho"]), eta_entropy=float(self.hp["eta_entropy"]),
            dropout=float(self.hp["dropout"]), tau=float(self.hp["km_tau"]),
            eps=float(self.hp["gamma_init"]),
            rel_init_scale=float(self.hp.get("rel_init_scale", 0.1)),
            use_node=bool(self.hp["use_node"]), use_mb=bool(self.hp["use_mb"]))
        self.pathway = None                                   # typed meta-path feature module
        if self.arm_mode == "pd1":                            # Path Design 1 (factorized)
            n_tok = self.pathrel.n_rel * (self.n_types + 1)
            K, H = int(self.hp["K"]), int(self.hp["H"])
            self.adapter = FactorizedAdapter(
                self.n_types, self.d_z, self.cfg.d, n_out, n_tok, K=K, H=H,
                q=int(self.hp["q_rank"]), d_r=int(self.hp["d_r"]), tau=float(self.hp["km_tau"]),
                eta=float(self.hp["eta_route"]), mlp_hidden=int(self.hp["mlp_hidden"]),
                dropout=float(self.hp["dropout"]), gamma_init=float(self.hp["gamma_init"]),
                node_on=bool(self.hp["node_on"]), path_on=bool(self.hp["path_on"])).to(self.device)
            if bool(self.hp["warmstart_u"]):
                self.adapter.warmstart_U(self._kmeans_centroids(zm_by_idx, K).to(self.device),
                                         float(self.hp["km_tau"]))
            print(f"[adapter] pd1 factorized: K={K} H={H} node_on={self.hp['node_on']} "
                  f"path_on={self.hp['path_on']}")
        else:
            self.adapter = DDILoRAAdapter(self.cfg).to(self.device)
            if self.arm_mode == "pathway":
                self.pathway = PathwayFeature(self.pathrel.n_rel, self.n_types, self.d_z,
                                              float(self.cfg.rho)).to(self.device)
            if bool(self.hp["warmstart_u"]):                  # k-means U_t warm-start (flat)
                cents = self._kmeans_centroids(zm_by_idx, int(self.hp["lam"]))
                self.adapter.assign.warmstart_from_centroids(cents.to(self.device), self.cfg.tau)
                print(f"[adapter] U warm-started from per-type k-means (tau={self.cfg.tau})")
        # step5: Adamic-Adar pooling prior (inverse-log-degree), fed as aa_weight
        self.use_aa = bool(self.hp["use_aa"])
        self._aa = torch.from_numpy((1.0 / np.log1p(deg + 1.0)).astype(np.float32)) if self.use_aa else None

        self.crit = AdapterLoss(self.cfg, "binary" if task.is_binary else "multiclass")
        params = list(self.adapter.parameters()) + list(self.hbase_provider.parameters())
        if self.pathway is not None:
            params += list(self.pathway.parameters())
        self.opt = torch.optim.AdamW(params, lr=float(self.hp["lr"]),
                                     weight_decay=float(self.hp["weight_decay"]))

    def _load_zm(self, encoder: str):
        explicit = self.hp.get("zm_npz")                     # pin the exact cache file
        if explicit:
            p = Path(explicit)
            if not p.is_file():
                raise FileNotFoundError(f"hp['zm_npz'] not found: {p}")
            d = np.load(p, allow_pickle=True)
            print(f"[adapter] z_m loaded (explicit zm_npz): {p.name} ({len(d['node_ids'])} nodes)")
            return d["z"].astype(np.float32), list(d["node_ids"])
        prefix = _ZM_PREFIX.get(encoder)
        if prefix is None:
            raise ValueError(f"unknown encoder {encoder!r}; known: {sorted(_ZM_PREFIX)}")
        cache = _ADAPTER_ROOT / "kg" / "_cache" / "node_embed"
        cands = sorted(glob.glob(str(cache / f"zm__{_slug(prefix)}*.npz")))
        if not cands:
            raise FileNotFoundError(f"no z_m cache for encoder {encoder!r} under {cache}")
        info = sorted(((p, len(np.load(p, allow_pickle=True)["node_ids"])) for p in cands),
                      key=lambda t: t[1], reverse=True)      # max coverage first
        if len(info) > 1:                                    # never silent: list all + chosen
            print(f"[adapter] WARN {len(info)} z_m caches match encoder {encoder!r}; picking "
                  f"max-coverage {Path(info[0][0]).name}; pin via hp['zm_npz']. all="
                  f"{[(Path(p).name, k) for p, k in info]}")
        best, best_n = info[0]
        d = np.load(best, allow_pickle=True)
        print(f"[adapter] z_m loaded: {Path(best).name} ({best_n} nodes)")
        return d["z"].astype(np.float32), list(d["node_ids"])

    def _load_zr(self, encoder: str) -> np.ndarray:
        explicit = self.hp.get("zr_npz")
        if explicit:
            d = np.load(Path(explicit), allow_pickle=True)
        else:
            prefix = _ZM_PREFIX.get(encoder)
            if prefix is None:
                raise ValueError(f"unknown encoder {encoder!r}; known: {sorted(_ZM_PREFIX)}")
            cache = _ADAPTER_ROOT / "kg" / "_cache" / "relation_embed"
            cands = sorted(glob.glob(str(cache / f"zr__{_slug(prefix)}*.npz")))
            if not cands:
                raise FileNotFoundError(f"no z_r cache for encoder {encoder!r} under {cache}")
            best = max(cands, key=lambda p: Path(p).stat().st_mtime)  # newest
            d = np.load(best, allow_pickle=True)
            print(f"[adapter] z_r loaded: {Path(best).name}")
        keys = list(d["keys"])
        if keys != _relmod.canonical_relations():            # row order must match zr_map
            raise ValueError("z_r keys do not match canonical_relations() ordering; "
                             "path-relation coefficients would be misaligned")
        return d["z"].astype(np.float32)

    def _rtilde(self, c1, c2, n1, n2) -> torch.Tensor:
        """(c1,c2 [M,47], n1,n2 [M]) -> r_tilde [M, d_z] = LayerNorm(num/den), den=0 -> 0."""
        dev, rho = self.device, float(self.cfg.rho)
        C1 = torch.as_tensor(c1, dtype=torch.float32, device=dev)
        C2 = torch.as_tensor(c2, dtype=torch.float32, device=dev)
        num = rho * (C1 @ self._zr) + (rho * rho / 2.0) * (C2 @ self._zr)   # [M, d_z]
        den = (rho * torch.as_tensor(n1, dtype=torch.float32, device=dev)
               + rho * rho * torch.as_tensor(n2, dtype=torch.float32, device=dev)).unsqueeze(-1)
        r = F.layer_norm(num / den.clamp_min(1e-9), (self.d_z,))            # non-affine LN
        return torch.where(den > 0, r, torch.zeros_like(r))                # den=0 fallback

    def _kmeans_centroids(self, zm_by_idx: np.ndarray, n_clusters: int) -> torch.Tensor:
        """Per-type k-means on z_m -> [T, n_clusters, d_z] centroids for U warm-start.
        Types with no z_m coverage keep a small random init; types with fewer than
        n_clusters covered nodes repeat centers to fill the slots."""
        from sklearn.cluster import KMeans
        T, lam = self.n_types, int(n_clusters)
        seed = int(self.hp["seed"])
        cents = np.random.default_rng(seed).normal(0, 0.02, (T, lam, self.d_z)).astype(np.float32)
        has = np.linalg.norm(zm_by_idx, axis=1) > 0
        for t in range(T):
            idx = np.where((self.kg.type_id == t) & has)[0]
            if len(idx) == 0:
                continue
            k = min(lam, len(idx))
            centers = KMeans(n_clusters=k, random_state=seed, n_init=4).fit(zm_by_idx[idx]).cluster_centers_
            for j in range(lam):
                cents[t, j] = centers[j % k]                    # repeat to fill slots
        return torch.from_numpy(cents)

    # -- mediators / batch assembly ------------------------------------------
    def _drug_idx(self, ids) -> list:
        out = []
        for x in ids:
            i = self.kg.get_idx(str(x))
            if i is None:
                raise ValueError(f"drug {x} not in KG")
            out.append(i)
        return out

    def _mediators(self, u_idx: int, v_idx: int) -> dict:
        # computed on the fly (per-drug neighborhoods are memoized in self.nbhd);
        # NOT cached per pair - training negatives change every epoch, so a pair
        # cache would grow unbounded across epochs.
        m = shared_mediators_idx(self.kg, self.nbhd, u_idx, v_idx,
                                 mode=self.hp["med_mode"], tau=int(self.hp["tau"]))
        cap = int(self.hp["max_med"])
        k = len(m["med_idx"])
        if cap and k > cap:                                   # keep top-cap: (d_a+d_b) asc, deg asc
            order = np.lexsort((self._deg[m["med_idx"]], m["d_a"] + m["d_b"]))[:cap]
            m = {kk: vv[order] for kk, vv in m.items()}
        return m

    def _build_batch(self, pairs: np.ndarray) -> AdapterBatch:
        ia = self._drug_idx(pairs[:, 0]); ib = self._drug_idx(pairs[:, 1])
        pidx, mtype, midx = [], [], []
        aa_, ab_ = [], []                                     # per-pair arm features (atoms/coeffs)
        for b, (u, v) in enumerate(zip(ia, ib)):
            med = self._mediators(u, v)
            mi = med["med_idx"]; k = len(mi)
            if not k:
                continue
            midx.append(mi); mtype.append(med["type_id"]); pidx.append(np.full(k, b, np.int64))
            if self.arm_mode in ("pathway", "pd1"):
                aa_.append(self.pathrel.arm_atoms(u, mi)); ab_.append(self.pathrel.arm_atoms(v, mi))
            elif self.arm_mode == "ztext":
                aa_.append(self.pathrel.arm_coeffs(u, mi)); ab_.append(self.pathrel.arm_coeffs(v, mi))
        dev = self.device
        if not midx:                                          # no pair has mediators
            zc = torch.zeros(0, self.d_z, device=dev); hc = torch.zeros(0, self.cfg.d, device=dev)
            empty_l = torch.zeros(0, dtype=torch.long, device=dev)
            return AdapterBatch(len(pairs), empty_l, empty_l, zc, zc, zc, hc)
        med_idx = torch.as_tensor(np.concatenate(midx), dtype=torch.long, device=dev)
        med_type = torch.as_tensor(np.concatenate(mtype).astype(np.int64), device=dev)
        pair_idx = torch.as_tensor(np.concatenate(pidx), dtype=torch.long, device=dev)
        z_m = self._zm[med_idx.cpu()].to(dev)                 # frozen gather
        if self.hp.get("shuffle_node"):                       # control: destroy node signal (keep dist)
            z_m = z_m[torch.randperm(z_m.shape[0], device=dev)]
        h_base = self.hbase_provider.h_base(med_idx)          # [M, d] from the H_base seam
        aa = self._aa[med_idx.cpu()].to(dev) if self.use_aa else None   # AA pooling prior
        r_a = r_b = torch.zeros(med_idx.shape[0], self.d_z, device=dev)  # default (off / pd1)
        tok = {}
        if self.arm_mode == "pd1":                            # top-2 route tokens for both arms
            ta = self._top2_tokens(np.concatenate(aa_, 0)); tb = self._top2_tokens(np.concatenate(ab_, 0))
            if self.hp.get("shuffle_route"):                  # control: misalign route signal (keep dist)
                p = torch.randperm(ta[0].shape[0], device=dev)
                ta = (ta[0][p], ta[1][p]); tb = (tb[0][p], tb[1][p])
            tok = dict(tok_a_idx=ta[0], tok_a_wt=ta[1], tok_b_idx=tb[0], tok_b_wt=tb[1])
        elif self.arm_mode == "pathway":
            A_a = torch.from_numpy(np.concatenate(aa_, 0)).to(dev)      # [M, R, T+1]
            A_b = torch.from_numpy(np.concatenate(ab_, 0)).to(dev)
            r_a = self.pathway(A_a); r_b = self.pathway(A_b)
        elif self.arm_mode == "ztext":
            def cat(col, arm):
                return np.concatenate([t[col] for t in arm])
            r_a = self._rtilde(cat(0, aa_), cat(1, aa_), cat(2, aa_), cat(3, aa_))
            r_b = self._rtilde(cat(0, ab_), cat(1, ab_), cat(2, ab_), cat(3, ab_))
        return AdapterBatch(len(pairs), pair_idx, med_type, z_m, r_a, r_b, h_base,
                            aa_weight=aa, med_idx=med_idx, **tok)

    def _top2_tokens(self, A_np: np.ndarray):
        """A [M, R, T+1] atom counts -> (idx [M,2] long, wt [M,2]) top-2 route-token mixture."""
        dev = self.device
        A = torch.from_numpy(A_np.reshape(A_np.shape[0], -1)).to(dev)   # [M, R*(T+1)]
        val, idx = torch.log1p(A).topk(2, dim=1)                        # [M, 2]
        wt = val / val.sum(1, keepdim=True).clamp_min(1e-9)            # normalized (0 if empty)
        return idx.long(), wt.float()

    def _minibatches(self, n: int, rng, shuffle: bool):
        order = rng.permutation(n) if shuffle else np.arange(n)
        bs = int(self.hp["batch_size"])
        for s in range(0, n, bs):
            yield order[s:s + bs]

    # -- RankModel contract --------------------------------------------------
    def train_epoch(self, epoch: EpochData, rng) -> TrainEpochOutput:
        pairs, labels = epoch.target_pairs, epoch.target_labels
        self.adapter.train(); self.hbase_provider.train()
        self.hbase_provider.encode(epoch.fact_context)        # refresh KG encoding (structural=no-op)
        tot, seen, step = 0.0, 0, 0
        n_steps = (len(pairs) + int(self.hp["batch_size"]) - 1) // int(self.hp["batch_size"])
        for sel in self._minibatches(len(pairs), rng, shuffle=True):
            batch = self._build_batch(pairs[sel])
            y = torch.as_tensor(labels[sel], device=self.device)
            logits, extra = self.adapter(batch)
            if self.arm_mode == "pd1":                        # factorized: task loss + reg (no beta entropy)
                loss, comps = self.crit(logits, y, None)
                reg = extra.get("reg") if isinstance(extra, dict) else None
                if reg is not None:
                    loss = loss + float(self.hp["reg_weight"]) * reg
            else:
                loss, comps = self.crit(logits, y, extra)     # extra = beta
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            tot += float(loss.item()) * len(sel); seen += len(sel); step += 1
            if step % int(self.hp["log_step_every"]) == 0:
                print(f"  [adapter][step {step}/{n_steps}] loss={tot/seen:.4f} "
                      f"ent={float(comps.get('entropy', 0.0)):.3f}")
        if self.use_path:                                     # watch memo growth
            memo = len(self.pathrel._amemo if self.arm_mode in ("pathway", "pd1") else self.pathrel._memo)
            print(f"  [adapter] memo={memo} aa_gate={float(self.adapter.aa_gate):.3f}")
        return TrainEpochOutput(mean_loss=(tot / max(seen, 1)), n_targets=len(labels))

    @torch.no_grad()
    def encode_pairs(self, pairs, context: ScoringContext) -> PairEncoding:
        pairs = np.asarray(pairs)
        self.adapter.eval(); self.hbase_provider.eval()
        self.hbase_provider.encode(context)                   # refresh KG encoding for this context
        out = []
        rng = np.random.default_rng(0)
        for sel in self._minibatches(len(pairs), rng, shuffle=False):
            batch = self._build_batch(pairs[sel])
            logits, _ = self.adapter(batch)                   # [b, n_out]
            out.append(logits.cpu().numpy().astype(np.float32))
        logits = np.concatenate(out, axis=0) if out else np.zeros((0, self.cfg.n_out), np.float32)
        return PairEncoding(pair_ids=pairs, logits=logits,
                            repr_kind="adapter_readout", repr_stage="head_out")

    def known_drugs(self) -> set:
        return {str(x) for x in self.kg.drug_ids()}

    def state_dict(self) -> dict:
        st = {"adapter": {k: v.detach().cpu().clone() for k, v in self.adapter.state_dict().items()},
              "hbase": {k: v.detach().cpu().clone() for k, v in self.hbase_provider.state_dict().items()}}
        if self.pathway is not None:
            st["pathway"] = {k: v.detach().cpu().clone() for k, v in self.pathway.state_dict().items()}
        return st

    def load_state_dict(self, state: dict) -> None:
        self.adapter.load_state_dict(state["adapter"])
        self.hbase_provider.load_state_dict(state["hbase"])
        if self.pathway is not None and "pathway" in state:
            self.pathway.load_state_dict(state["pathway"])

    def effective_hp(self) -> dict:
        return dict(self.hp)

    def adapter_state(self):
        if self.arm_mode == "pd1":                           # factorized: emergent-alignment hooks
            return self.adapter.mechanism_state()
        st = self.adapter.adapter_state()
        if self.pathway is not None:                         # PK/PD mechanism map = E routing:
            st["pathway_E"] = self.pathway.E.detach().cpu()  # [R, T+1, d_z] atom table (W_A also
            #                                                  in st -> analyze W_A E[r,t] routing)
        return st


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-")[:40]


__all__ = ["AdapterRankModel", "StructuralHBase"]
