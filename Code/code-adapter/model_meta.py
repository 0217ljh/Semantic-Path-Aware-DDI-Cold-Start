"""Stage 3 / atom P0 - method/baseline METADATA registry, protocol-centric.

Each backbone/baseline carries its source, idea and NATIVE cold-start protocol
(from the original paper); our method carries its variant modifications and its
own protocol. The cold-start protocol has two categories (verbatim from the paper,
Section "Baselines"): protocol 1 partitions drugs in advance into disjoint seen/
unseen sets; protocol 2 is the EmerGNN-style dynamic partition where a subset of
seen drugs is randomly treated as unseen at each training epoch and their DDI
edges removed, training the model directly under cold-start.

Protocol resolution for the adapter (user-locked rule):
  - adapter-alone (no backbone) -> the METHOD's own protocol
  - with a backbone (frozen OR jointly trained) -> the BACKBONE's protocol
so switching a run's protocol only depends on (mode, backbone) and plugs into the
trainer without the loop knowing which protocol it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Protocol(Enum):
    #: drugs partitioned in advance into disjoint seen/unseen; train on seen only.
    P1_FIXED = "protocol_1"
    #: EmerGNN-style: each epoch randomly treat a subset of seen drugs as unseen
    #: and drop their DDI edges (dynamic per-epoch emerging-drug simulation).
    P2_EMERGING = "protocol_2"


class AdapterMode(Enum):
    ALONE = "adapter_alone"        # H_base = 0, adapter is the whole model
    FROZEN = "backbone_frozen"     # frozen backbone + trained adapter
    JOINT = "backbone_joint"       # backbone + adapter trained together


@dataclass
class BaselineMeta:
    """Metadata for a baseline backbone (source paper, idea, native protocol)."""
    key: str                       # registry key
    name: str                      # display name
    cite_key: str                  # bib key in the paper
    reasoning: str                 # node / molecular-graph / path / subgraph based
    idea: str                      # one-line core idea
    native_protocol: Protocol      # protocol of the ORIGINAL paper
    set_protocol: Protocol | None = None  # protocol WE use; default = native
    note: str = ""

    def __post_init__(self) -> None:
        if self.set_protocol is None:
            self.set_protocol = self.native_protocol


@dataclass
class MethodMeta:
    """Metadata for our method: the variant modifications + our own protocol."""
    name: str
    protocol: Protocol             # protocol used in adapter-alone mode
    variants: dict[str, str] = field(default_factory=dict)  # variant name -> what it changes
    note: str = ""


# ---------------------------------------------------------------------------
# registry (baselines: paper Section "Baselines", line 393; protocol per the
# rule "native cold-start baselines keep their protocol, warm-start-only use
# protocol 1" + the recorded original-repo protocol audit: only EmerGNN is P2).
# ---------------------------------------------------------------------------
BASELINES: dict[str, BaselineMeta] = {
    "rgcn": BaselineMeta(
        key="rgcn", name="R-GCN", cite_key="rgcn", reasoning="node-based",
        idea="relational graph convolution over the KG (per-relation message passing).",
        native_protocol=Protocol.P1_FIXED),
    "ssiddi": BaselineMeta(
        key="ssiddi", name="SSI-DDI", cite_key="ssiddi", reasoning="molecular-graph",
        idea="substructure-substructure interactions from molecular graphs; KG-free "
             "(adapter enters as an added KG channel concatenated with the mol representation).",
        native_protocol=Protocol.P1_FIXED),
    "emergnn": BaselineMeta(
        key="emergnn", name="EmerGNN", cite_key="emergnn", reasoning="path-based",
        idea="path-based flow-style GNN designed for emerging (unseen) drugs.",
        native_protocol=Protocol.P2_EMERGING,
        note="native cold-start: per-epoch dynamic seen->unseen partition (protocol 2)."),
    "mkgfenn": BaselineMeta(
        key="mkgfenn", name="MKG-FENN", cite_key="mkgfenn", reasoning="multimodal-KG",
        idea="multimodal KG fused end-to-end neural network for DDI event prediction.",
        native_protocol=Protocol.P1_FIXED),
    "knowddi": BaselineMeta(
        key="knowddi", name="KnowDDI", cite_key="knowddi", reasoning="subgraph-based",
        idea="enclosing-subgraph reasoning over the KG for DDI.",
        native_protocol=Protocol.P1_FIXED),
    "tiger": BaselineMeta(
        key="tiger", name="TIGER", cite_key="tiger", reasoning="graph-based",
        idea="graph-based DDI prediction backbone.",
        native_protocol=Protocol.P1_FIXED,
        note="reasoning family not individually pinned in tex; confirm from repo if needed."),
}

METHOD = MethodMeta(
    name="DDI-LoRA",
    protocol=Protocol.P2_EMERGING,  # adapter-alone default; trains directly under cold-start
    variants={
        "full": "block-sparse M_A + typed prototypes M_B + frozen semantic z_m/z_r.",
        "untyped_mb": "RQ2 module ablation: M_B de-typed (single shared prototype block).",
        "dense_ma": "RQ2 module ablation: M_A no longer block-sparse per type.",
        "random_semantic": "RQ2 semantic ablation (capacity-matched): z_m/z_r -> random same-rank vectors.",
        "shuffled_semantic": "RQ2 semantic ablation (capacity-matched): z_m/z_r shuffled across nodes/relations.",
    },
    note="variants extended as ablations are implemented; see experiment design RQ2/RQ3.",
)


def resolve_protocol(mode: AdapterMode, backbone: BaselineMeta | None,
                     method: MethodMeta = METHOD) -> Protocol:
    """The protocol a training run should use (user-locked rule)."""
    if mode is AdapterMode.ALONE:
        return method.protocol
    if backbone is None:
        raise ValueError(f"mode {mode} needs a backbone")
    return backbone.set_protocol


__all__ = ["Protocol", "AdapterMode", "BaselineMeta", "MethodMeta",
           "BASELINES", "METHOD", "resolve_protocol"]
