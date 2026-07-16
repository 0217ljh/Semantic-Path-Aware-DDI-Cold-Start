"""Generate Paper Figure 2 — Multi-Modal Alignment Failure on Cold-Start.

Four panels in 2x2:
  (a) Top-left:  Endpoint alignment works on known drugs (KG ↔ Mol manifolds aligned)
  (b) Top-right: Endpoint alignment collapses on cold drug (off-manifold)
  (c) Bottom-left:  Mechanism alignment works on shared units (fragment ↔ hyperedge)
  (d) Bottom-right: Mechanism alignment preserved on cold drug

Output: Notes/Log/figures/figure_2_multimodal_alignment.png (+ .pdf)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

C_KG = "#3b82f6"               # blue for KG side
C_MOL = "#ef4444"              # red for molecular side
C_COLD = "#94a3b8"             # gray for cold drug (no embedding)
C_FRAG = "#f59e0b"             # amber for fragment
C_HYP = "#8b5cf6"              # purple for hyperedge
C_OK = "#10b981"               # green check
C_BAD = "#dc2626"              # dark red X


# ----------------------------------------------------------------------------
# Synthetic embeddings (for clarity, not real data)
# ----------------------------------------------------------------------------
np.random.seed(42)


def make_known_drug_clouds() -> tuple[np.ndarray, np.ndarray]:
    """Two well-aligned drug-identity embedding clouds (KG + mol).

    Returns (kg_pts, mol_pts) each (N, 2).
    """
    n = 20
    # KG cloud in upper-left
    kg = np.column_stack([
        np.random.normal(2.0, 0.7, n),
        np.random.normal(3.0, 0.6, n),
    ])
    # Mol cloud in upper-right
    mol = np.column_stack([
        np.random.normal(7.0, 0.7, n),
        np.random.normal(3.0, 0.6, n),
    ])
    return kg, mol


# ----------------------------------------------------------------------------
# Panel (a) Top-Left: Endpoint alignment on known drugs (works)
# ----------------------------------------------------------------------------
def draw_panel_a(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("(a) MKG-FENN/TIGER on known drugs:\n"
                 r"$e_{\mathrm{KG}}(d) \;\leftrightarrow\; e_{\mathrm{mol}}(d)$  aligned",
                 fontsize=10, pad=5)

    # Background separator
    ax.axvline(5, color="lightgray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.text(2.5, 5.6, "KG embedding space", ha="center", fontsize=8,
            color=C_KG, style="italic")
    ax.text(7.5, 5.6, "Mol embedding space", ha="center", fontsize=8,
            color=C_MOL, style="italic")

    kg_pts, mol_pts = make_known_drug_clouds()
    # Draw points
    ax.scatter(kg_pts[:, 0], kg_pts[:, 1], s=35, c=C_KG, edgecolor="black",
                linewidth=0.5, zorder=3)
    ax.scatter(mol_pts[:, 0], mol_pts[:, 1], s=35, c=C_MOL, edgecolor="black",
                linewidth=0.5, zorder=3)

    # Connect same-drug across modalities
    for k_pt, m_pt in zip(kg_pts[:8], mol_pts[:8]):
        ax.plot([k_pt[0], m_pt[0]], [k_pt[1], m_pt[1]], color="gray",
                linewidth=0.4, alpha=0.4, zorder=1)

    # Caption
    ax.text(5, 0.7, r"$f_\mathrm{align}$ fitted on $d \in V_{\mathrm{drug}}^{\mathrm{seen}}$",
            ha="center", fontsize=9, style="italic")
    # Green check
    ax.text(9.3, 0.5, "✓", fontsize=20, color=C_OK, fontweight="bold")


# ----------------------------------------------------------------------------
# Panel (b) Top-Right: Endpoint alignment collapses on cold drug
# ----------------------------------------------------------------------------
def draw_panel_b(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("(b) MKG-FENN/TIGER on cold drug:\n"
                 "alignment collapses (off-manifold)",
                 fontsize=10, pad=5)

    ax.axvline(5, color="lightgray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.text(2.5, 5.6, "KG embedding space", ha="center", fontsize=8,
            color=C_KG, style="italic")
    ax.text(7.5, 5.6, "Mol embedding space", ha="center", fontsize=8,
            color=C_MOL, style="italic")

    # Known drugs in gray (background)
    kg_pts, mol_pts = make_known_drug_clouds()
    ax.scatter(kg_pts[:, 0], kg_pts[:, 1], s=20, c="lightgray",
                edgecolor="gray", linewidth=0.3, alpha=0.5, zorder=2)
    ax.scatter(mol_pts[:, 0], mol_pts[:, 1], s=20, c="lightgray",
                edgecolor="gray", linewidth=0.3, alpha=0.5, zorder=2)

    # Cold drug d_new at OOD position
    d_kg = (4.0, 1.2)         # well off the known KG cloud
    d_mol = (8.0, 4.8)        # well off the known mol cloud (different region)
    ax.scatter(*d_kg, s=140, marker="*", c=C_KG, edgecolor="black",
                linewidth=1.0, zorder=5)
    ax.text(d_kg[0] - 0.5, d_kg[1] - 0.5, r"$d_{\mathrm{new}}$ KG",
            ha="center", fontsize=8, color=C_KG, fontweight="bold")
    ax.scatter(*d_mol, s=140, marker="*", c=C_MOL, edgecolor="black",
                linewidth=1.0, zorder=5)
    ax.text(d_mol[0] + 0.5, d_mol[1] + 0.5, r"$d_{\mathrm{new}}$ mol",
            ha="center", fontsize=8, color=C_MOL, fontweight="bold")

    # Dashed broken arrow with red X
    ax.annotate("", xy=d_mol, xytext=d_kg,
                arrowprops=dict(arrowstyle="->", linestyle="--", color=C_BAD,
                                lw=1.4))
    # Red X over the arrow midpoint
    mid_x = (d_kg[0] + d_mol[0]) / 2
    mid_y = (d_kg[1] + d_mol[1]) / 2
    ax.text(mid_x, mid_y, "✗", fontsize=22, color=C_BAD,
            fontweight="bold", ha="center", va="center")

    # Caption
    ax.text(5, 0.4,
            r"$d_{\mathrm{new}}$ off the known manifold;",
            ha="center", fontsize=9, style="italic", color=C_BAD)
    ax.text(5, 0.0,
            r"$f_\mathrm{align}$ has no valid extrapolation.  AUC $\approx 0.5$",
            ha="center", fontsize=9, style="italic", color=C_BAD)


# ----------------------------------------------------------------------------
# Panel (c) Bottom-Left: Mechanism alignment (fragments ↔ hyperedges)
# ----------------------------------------------------------------------------
def draw_panel_c(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("(c) Our framework on shared units:\n"
                 r"fragment $\leftrightarrow$ hyperedge  alignment",
                 fontsize=10, pad=5)

    # Left column: fragments
    frag_labels = [r"$\beta$-lactam ring", "halogen group", "amide bond",
                   "carboxylic acid"]
    frag_ys = [4.8, 3.8, 2.8, 1.8]
    for label, y in zip(frag_labels, frag_ys):
        ax.add_patch(FancyBboxPatch((0.5, y - 0.25), 2.5, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=C_FRAG, alpha=0.5,
                                     edgecolor="darkorange", linewidth=1))
        ax.text(1.75, y, label, ha="center", va="center", fontsize=8.5,
                fontweight="bold")
    ax.text(1.75, 5.5, "Molecular fragments", ha="center", fontsize=9,
            fontweight="bold", color="darkorange")

    # Right column: hyperedges
    hyp_labels = [r"$h_{\tau=\mathrm{enzyme}}$",
                  r"$h_{\tau=\mathrm{target}}$",
                  r"$h_{\tau=\mathrm{pathway}}$",
                  r"$h_{\tau=\mathrm{transporter}}$"]
    for label, y in zip(hyp_labels, frag_ys):
        ax.add_patch(FancyBboxPatch((7.0, y - 0.25), 2.5, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=C_HYP, alpha=0.4,
                                     edgecolor="indigo", linewidth=1))
        ax.text(8.25, y, label, ha="center", va="center", fontsize=8.5,
                fontweight="bold")
    ax.text(8.25, 5.5, "KG hyperedges", ha="center", fontsize=9,
            fontweight="bold", color="indigo")

    # Many-to-many connections
    connections = [
        (0, 0), (0, 1),         # β-lactam ↔ enzyme, target
        (1, 1), (1, 3),         # halogen ↔ target, transporter
        (2, 2), (2, 0),         # amide ↔ pathway, enzyme
        (3, 3), (3, 2),         # carboxylic ↔ transporter, pathway
    ]
    for (i, j) in connections:
        ax.plot([3.0, 7.0], [frag_ys[i], frag_ys[j]], color="gray",
                linewidth=0.7, alpha=0.5, zorder=1)

    ax.text(5, 0.6,
            "Alignment unit = mechanism, not drug identity",
            ha="center", fontsize=9, style="italic", fontweight="bold")
    ax.text(9.3, 0.5, "✓", fontsize=20, color=C_OK, fontweight="bold")


# ----------------------------------------------------------------------------
# Panel (d) Bottom-Right: Mechanism alignment preserved on cold drug
# ----------------------------------------------------------------------------
def draw_panel_d(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("(d) Our framework on cold drug:\n"
                 "alignment preserved (fragments transfer)",
                 fontsize=10, pad=5)

    # Cold drug node at top
    d_new_pos = (5, 5.3)
    ax.add_patch(Circle(d_new_pos, 0.35, facecolor="white", edgecolor=C_COLD,
                        linewidth=1.5, linestyle="--"))
    ax.text(*d_new_pos, r"$d_{\mathrm{new}}$", ha="center", va="center",
            fontsize=9, color=C_COLD, fontweight="bold")
    ax.text(d_new_pos[0] + 1.4, d_new_pos[1], "(cold drug)",
            ha="left", va="center", fontsize=8, color=C_COLD, style="italic")

    # SMILES → fragments arrow
    ax.annotate("", xy=(5, 4.3), xytext=(5, 4.95),
                arrowprops=dict(arrowstyle="->", color=C_FRAG, lw=1.5))
    ax.text(5.5, 4.6, "extract fragments from SMILES",
            ha="left", fontsize=7.5, style="italic", color=C_FRAG)

    # Extracted fragments (3)
    frag_labels = [r"$\beta$-lactam", "amide bond", "halogen"]
    frag_xs = [2.0, 5.0, 8.0]
    for label, x in zip(frag_labels, frag_xs):
        ax.add_patch(FancyBboxPatch((x - 0.9, 3.5), 1.8, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=C_FRAG, alpha=0.5,
                                     edgecolor="darkorange", linewidth=1))
        ax.text(x, 3.75, label, ha="center", va="center", fontsize=8,
                fontweight="bold")

    # Hyperedges (3 anchors, well-trained on other drugs)
    hyp_labels = [r"$h_{\tau=\mathrm{enzyme}}$", r"$h_{\tau=\mathrm{pathway}}$",
                  r"$h_{\tau=\mathrm{target}}$"]
    hyp_xs = [2.0, 5.0, 8.0]
    for label, x in zip(hyp_labels, hyp_xs):
        ax.add_patch(FancyBboxPatch((x - 1.0, 1.5), 2.0, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=C_HYP, alpha=0.4,
                                     edgecolor="indigo", linewidth=1))
        ax.text(x, 1.75, label, ha="center", va="center", fontsize=8,
                fontweight="bold")
    ax.text(5, 1.05, "(well-trained on many other drugs)",
            ha="center", fontsize=7.5, style="italic", color="indigo")

    # Map fragments to hyperedges with green arrows
    for fx, hx in zip(frag_xs, hyp_xs):
        ax.annotate("", xy=(hx, 2.0), xytext=(fx, 3.5),
                    arrowprops=dict(arrowstyle="->", color=C_OK, lw=1.3))

    # Caption
    ax.text(5, 0.5,
            "Fragments and hyperedges transfer beyond seen drugs",
            ha="center", fontsize=9, style="italic", fontweight="bold",
            color=C_OK)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    draw_panel_a(axes[0, 0])
    draw_panel_b(axes[0, 1])
    draw_panel_c(axes[1, 0])
    draw_panel_d(axes[1, 1])

    fig.suptitle(
        "Figure 2.  Multi-modal alignment object determines cold-start transferability.\n"
        "Top: drug-identity alignment (MKG-FENN, TIGER) collapses on cold drug.\n"
        "Bottom: mechanism alignment (ours) is preserved.",
        fontsize=12, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.02,
        "Multi-modal DDI methods like MKG-FENN and TIGER align KG and molecular views at the drug-identity level.\n"
        "Under cold-start, the new drug's identity is OOD; the learned alignment manifold cannot extrapolate.\n"
        "Our framework aligns at the mechanism level — molecular fragments map to hyperedge mechanism units —\n"
        "and because fragments and mechanism units transfer beyond seen drugs, alignment survives cold-start.",
        ha="center", fontsize=8.5, style="italic",
    )

    plt.tight_layout(rect=(0.0, 0.07, 1.0, 0.94))

    out_dir = Path(__file__).resolve().parents[2] / "Notes" / "Log" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / "figure_2_multimodal_alignment.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "figure_2_multimodal_alignment.pdf",
                bbox_inches="tight")
    print(f"saved -> {out_dir / 'figure_2_multimodal_alignment.png'}")
    print(f"saved -> {out_dir / 'figure_2_multimodal_alignment.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
