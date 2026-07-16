"""Generate Section 1 motivation figure: Why multi-modal (Mol + KG) is necessary.

Renders the user's hand-drawn diagram (2026-06-08) as a clean matplotlib figure.

Logic:
  - Mol-side gives us: drug A's structure → which proteins it binds (drug A → P1, P2, P3)
  - KG-side gives us: how those proteins connect via pathways / other proteins
    / shared physiology to drug B's proteins → drug B
  - System-level DDI = closed loop of (mol binding at drug A) + (KG propagation
    through host biology) + (mol binding at drug B)
  - In cold-start, both drugs are unseen → the closure CANNOT be inferred from
    drug A's molecular info alone → multi-modal is necessary by construction

Output: Notes/Log/figures/figure_motivation_multimodal_necessity.{png, pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

C_DRUG_A = "#3b82f6"          # blue for drug A
C_DRUG_B = "#ef4444"          # red for drug B
C_PROTEIN_A = "#60a5fa"        # lighter blue for drug A's proteins
C_PROTEIN_B = "#fb7185"        # lighter red for drug B's proteins
C_KG_NODE = "#a78bfa"          # purple for KG intermediate nodes
C_KG_BOX = "#ede9fe"           # light purple for KG bridge area
C_MOL_ANN = "#2563eb"          # mol annotation color
C_KG_ANN = "#7c3aed"           # KG annotation color


def _arrow(ax, start, end, color="black", lw=1.5, style="->",
           rad=0.0, alpha=1.0):
    """Draw a curved arrow."""
    arr = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        lw=lw,
        alpha=alpha,
        mutation_scale=15,
    )
    ax.add_patch(arr)


def _drug_node(ax, xy, label, color, size=0.55):
    """Draw a drug node (drug A or drug B) — large filled circle."""
    ax.add_patch(Circle(xy, size, facecolor=color, edgecolor="black",
                        linewidth=1.8, zorder=5))
    ax.text(*xy, label, ha="center", va="center", fontsize=13,
            fontweight="bold", color="white", zorder=6)


def _protein_node(ax, xy, label, color, size=0.35):
    """Draw a protein node."""
    ax.add_patch(Circle(xy, size, facecolor=color, edgecolor="black",
                        linewidth=1.2, zorder=4))
    ax.text(*xy, label, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="white", zorder=5)


def _kg_node(ax, xy, label, size=0.25, fontsize=7):
    """Draw a KG intermediate node (pathway / other protein)."""
    ax.add_patch(Circle(xy, size, facecolor=C_KG_NODE, edgecolor="indigo",
                        linewidth=1.0, zorder=4, alpha=0.85))
    ax.text(*xy, label, ha="center", va="center", fontsize=fontsize,
            color="white", zorder=5)


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # Top row: Drug A (left) and Drug B (right)
    # ------------------------------------------------------------------
    drug_a_pos = (1.5, 8.5)
    drug_b_pos = (14.5, 8.5)
    _drug_node(ax, drug_a_pos, "Drug A", C_DRUG_A, size=0.55)
    _drug_node(ax, drug_b_pos, "Drug B", C_DRUG_B, size=0.55)

    ax.text(drug_a_pos[0], drug_a_pos[1] + 0.95,
            "(complete molecular structure)",
            ha="center", fontsize=8.5, style="italic", color="gray")
    ax.text(drug_b_pos[0], drug_b_pos[1] + 0.95,
            "(complete molecular structure)",
            ha="center", fontsize=8.5, style="italic", color="gray")

    # ------------------------------------------------------------------
    # Mol-side: Drug A → its binding proteins (left column)
    # ------------------------------------------------------------------
    protein_a_pos = [(1.5, 6.5), (1.5, 5.5), (1.5, 4.5)]
    protein_a_labels = ["P₁", "P₂", "P₃"]
    for pos, lab in zip(protein_a_pos, protein_a_labels):
        _protein_node(ax, pos, lab, C_PROTEIN_A, size=0.32)

    # Arrows from drug A to its proteins (bundled)
    for pos in protein_a_pos:
        _arrow(ax, (drug_a_pos[0], drug_a_pos[1] - 0.55), (pos[0], pos[1] + 0.32),
               color=C_MOL_ANN, lw=1.4, rad=0.15)

    # Mol-side annotation for drug A
    ax.text(0.2, 7.5,
            "Mol determines:\nwhich proteins\nDrug A binds",
            ha="left", va="center", fontsize=9.5, color=C_MOL_ANN,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_MOL_ANN, linewidth=1.0))

    # ------------------------------------------------------------------
    # Mol-side: Drug B → its binding proteins (right column)
    # ------------------------------------------------------------------
    protein_b_pos = [(14.5, 6.5), (14.5, 5.5), (14.5, 4.5)]
    protein_b_labels = ["Q₁", "Q₂", "Q₃"]
    for pos, lab in zip(protein_b_pos, protein_b_labels):
        _protein_node(ax, pos, lab, C_PROTEIN_B, size=0.32)

    for pos in protein_b_pos:
        _arrow(ax, (drug_b_pos[0], drug_b_pos[1] - 0.55), (pos[0], pos[1] + 0.32),
               color=C_MOL_ANN, lw=1.4, rad=-0.15)

    # Mol-side annotation for drug B
    ax.text(15.8, 7.5,
            "Mol determines:\nwhich proteins\nDrug B binds",
            ha="right", va="center", fontsize=9.5, color=C_MOL_ANN,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_MOL_ANN, linewidth=1.0))

    # ------------------------------------------------------------------
    # KG-side: big box in the middle showing the bridge
    # ------------------------------------------------------------------
    kg_box = FancyBboxPatch(
        (3.0, 1.7), 10.0, 4.0,
        boxstyle="round,pad=0.15",
        facecolor=C_KG_BOX,
        edgecolor="indigo",
        linewidth=1.8,
        zorder=1,
    )
    ax.add_patch(kg_box)

    # KG label inside the box
    ax.text(8, 5.4, "KG bridge (system-level biology)",
            ha="center", fontsize=10, fontweight="bold", color="indigo",
            zorder=3)
    ax.text(8, 5.0, "pathways  •  shared regulators  •  PPI network  •  enzyme/transporter sharing",
            ha="center", fontsize=8, color="indigo", style="italic",
            zorder=3)

    # KG intermediate nodes (a small network in the box)
    kg_nodes_pos = [
        (5.0, 4.0),    # pathway A
        (5.0, 3.0),    # other protein
        (8.0, 3.5),    # central hub (e.g. CYP3A4)
        (8.0, 2.5),    # pathway
        (11.0, 4.0),   # other protein
        (11.0, 3.0),   # side effect
    ]
    kg_nodes_labels = [
        "pathway\nα",
        "other\nprotein",
        "shared\nenzyme",
        "PPI\nhub",
        "other\nprotein",
        "shared\npathway",
    ]
    for pos, lab in zip(kg_nodes_pos, kg_nodes_labels):
        _kg_node(ax, pos, lab, size=0.38, fontsize=6.5)

    # Connect drug A's proteins → KG nodes
    for src in protein_a_pos:
        for dst in kg_nodes_pos[:3]:  # left/middle KG nodes
            _arrow(ax, (src[0] + 0.32, src[1]), (dst[0] - 0.38, dst[1]),
                   color="gray", lw=0.6, alpha=0.45, rad=0.05)

    # Connect KG nodes among themselves (intermediate propagation)
    inner_edges = [
        (kg_nodes_pos[0], kg_nodes_pos[2]),
        (kg_nodes_pos[1], kg_nodes_pos[2]),
        (kg_nodes_pos[2], kg_nodes_pos[3]),
        (kg_nodes_pos[2], kg_nodes_pos[4]),
        (kg_nodes_pos[3], kg_nodes_pos[5]),
        (kg_nodes_pos[4], kg_nodes_pos[5]),
    ]
    for s, t in inner_edges:
        _arrow(ax, s, t, color="indigo", lw=1.0, alpha=0.65, rad=0.08,
               style="-")

    # Connect KG nodes → drug B's proteins
    for dst in protein_b_pos:
        for src in kg_nodes_pos[3:]:  # right/middle KG nodes
            _arrow(ax, (src[0] + 0.38, src[1]), (dst[0] - 0.32, dst[1]),
                   color="gray", lw=0.6, alpha=0.45, rad=-0.05)

    # KG-side annotation (top of box)
    ax.text(8, 6.05,
            "KG must provide: how Drug A's perturbations PROPAGATE through host biology to reach Drug B",
            ha="center", fontsize=9.5, fontweight="bold", color=C_KG_ANN)

    # ------------------------------------------------------------------
    # Bottom bracket and conclusion
    # ------------------------------------------------------------------
    # Bracket below the whole figure showing the closed loop
    bracket_y = 1.0
    ax.annotate("", xy=(1.5, bracket_y), xytext=(14.5, bracket_y),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
    ax.annotate("", xy=(1.5, bracket_y), xytext=(1.5, bracket_y + 0.3),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
    ax.annotate("", xy=(14.5, bracket_y), xytext=(14.5, bracket_y + 0.3),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.2))

    ax.text(8, 0.55,
            "DDI = closed causal chain.  "
            r"$\bf{Multi-modal\ fusion\ is\ necessary}$"
            "  because neither modality alone closes the loop.",
            ha="center", fontsize=10.5, color="black")

    # ------------------------------------------------------------------
    # Title and final layout
    # ------------------------------------------------------------------
    fig.suptitle(
        "Figure 1.  Multi-modal is necessary for DDI prediction.\n"
        "Mol determines each drug's binding profile (endpoints). KG determines "
        "the system-level bridge that links the two drugs into a single causal "
        "chain.",
        fontsize=11.5, y=0.985,
    )

    fig.text(
        0.5, 0.02,
        "Drug A's molecular structure determines which proteins it binds (left). Drug B's molecular structure determines its own binding profile (right).\n"
        "But whether the two drugs INTERACT depends on how their perturbations compose in host biology — a property of the protein/pathway network (KG), not of either drug alone.\n"
        "In cold-start, both drugs are unseen at training; the bridge cannot be inferred from Drug A's molecular structure alone, making multi-modal fusion necessary by construction.",
        ha="center", fontsize=8.5, style="italic",
    )

    plt.tight_layout(rect=(0.0, 0.07, 1.0, 0.94))

    out_dir = Path(__file__).resolve().parents[2] / "Notes" / "Log" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / "figure_motivation_multimodal_necessity.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "figure_motivation_multimodal_necessity.pdf",
                bbox_inches="tight")
    print(f"saved -> {out_dir / 'figure_motivation_multimodal_necessity.png'}")
    print(f"saved -> {out_dir / 'figure_motivation_multimodal_necessity.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
