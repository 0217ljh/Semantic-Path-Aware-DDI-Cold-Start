"""Generate Paper Figure 1 — Common Neighbor Structural Variable Preservation.

Three panels showing:
  (a) The structural variable A_τ(a, b) — a KG-intrinsic anchor node set
  (b) Endpoint GNN over-smoothing — A_τ dissolves into mixed embeddings
  (c) Our framework — A_τ stays explicit through the forward pass

Output: Notes/Log/figures/figure_1_common_neighbor_preservation.png (+ .pdf)
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

C_DRUG_KNOWN = "#3b82f6"     # blue for known drug
C_DRUG_COLD = "#94a3b8"       # gray for cold drug
C_ANCHOR = "#ef4444"          # red for anchor m
C_OTHER = "#cbd5e1"           # light gray for other neighbors
C_EMBED_KNOWN = "#10b981"     # green for known embedding
C_HIGHLIGHT = "#fbbf24"       # yellow for highlight set
C_PATH = "#8b5cf6"            # purple for NBFNet path


# ----------------------------------------------------------------------------
# Panel (a): The Structural Variable
# ----------------------------------------------------------------------------
def draw_panel_a(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(5, 5.6, "(a) The Structural Variable", ha="center", fontsize=11,
            fontweight="bold")

    # Drug a (left)
    a_pos = (1.5, 3)
    ax.add_patch(Circle(a_pos, 0.4, facecolor=C_DRUG_KNOWN, edgecolor="black",
                        linewidth=1.5))
    ax.text(*a_pos, "a", ha="center", va="center", fontsize=12, fontweight="bold",
            color="white")

    # Drug b (right)
    b_pos = (8.5, 3)
    ax.add_patch(Circle(b_pos, 0.4, facecolor=C_DRUG_KNOWN, edgecolor="black",
                        linewidth=1.5))
    ax.text(*b_pos, "b", ha="center", va="center", fontsize=12, fontweight="bold",
            color="white")

    # Common anchor m (center)
    m_pos = (5, 3)
    ax.add_patch(Circle(m_pos, 0.4, facecolor=C_ANCHOR, edgecolor="black",
                        linewidth=1.5))
    ax.text(*m_pos, "m", ha="center", va="center", fontsize=12, fontweight="bold",
            color="white")
    ax.text(5, 2.2, r"$\kappa(m)=\tau$", ha="center", fontsize=9, style="italic")

    # Other neighbors (one per drug, not common)
    a_neighbor = (2, 4.5)
    ax.add_patch(Circle(a_neighbor, 0.25, facecolor=C_OTHER, edgecolor="gray",
                        linewidth=1))
    b_neighbor = (8, 4.5)
    ax.add_patch(Circle(b_neighbor, 0.25, facecolor=C_OTHER, edgecolor="gray",
                        linewidth=1))

    # Connect: a → m, b → m, a → other, b → other
    edges = [(a_pos, m_pos), (b_pos, m_pos), (a_pos, a_neighbor),
             (b_pos, b_neighbor)]
    for s, t in edges:
        ax.annotate("", xy=t, xytext=s,
                    arrowprops=dict(arrowstyle="-", color="black", lw=1.0))

    # The structural variable box
    ax.add_patch(FancyBboxPatch((2.5, 0.6), 5, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=C_HIGHLIGHT, alpha=0.3,
                                 edgecolor="black", linewidth=1.2))
    ax.text(5, 1.4,
            r"$A_\tau(a, b) = N_\tau(a) \cap N_\tau(b) = \{m\}$",
            ha="center", fontsize=10)
    ax.text(5, 0.9,
            "KG-intrinsic. No learning. Computable for any (a, b).",
            ha="center", fontsize=8, style="italic")


# ----------------------------------------------------------------------------
# Panel (b): Endpoint GNN Over-Smoothing
# ----------------------------------------------------------------------------
def draw_panel_b(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(5, 5.6, "(b) Endpoint GNN — over-smoothing loss",
            ha="center", fontsize=11, fontweight="bold")

    # Two drugs with multi-layer aggregation
    a_pos = (1.5, 4)
    b_pos = (8.5, 4)
    ax.add_patch(Circle(a_pos, 0.35, facecolor=C_DRUG_KNOWN, edgecolor="black",
                        linewidth=1.2))
    ax.text(*a_pos, "a", ha="center", va="center", fontsize=10, color="white",
            fontweight="bold")
    ax.add_patch(Circle(b_pos, 0.35, facecolor=C_DRUG_KNOWN, edgecolor="black",
                        linewidth=1.2))
    ax.text(*b_pos, "b", ha="center", va="center", fontsize=10, color="white",
            fontweight="bold")

    # Show multi-hop neighbors as concentric circles
    for r, alpha in [(0.9, 0.20), (1.4, 0.12), (1.9, 0.06)]:
        ax.add_patch(Circle(a_pos, r, facecolor=C_DRUG_KNOWN, alpha=alpha,
                            edgecolor="none"))
        ax.add_patch(Circle(b_pos, r, facecolor=C_DRUG_KNOWN, alpha=alpha,
                            edgecolor="none"))

    # m is tiny here
    m_pos = (5, 4)
    ax.add_patch(Circle(m_pos, 0.18, facecolor=C_ANCHOR, edgecolor="darkred"))

    # Embedding bars below (showing dilution)
    bar_y = 1.5
    bar_h = 0.35

    # For e_a^(L)
    bar_left_x = 0.3
    bar_width = 4.0
    # 5 colored slices: m (red) tiny + 4 others large
    slice_widths = [0.15, 1.0, 1.2, 0.9, 0.75]  # m tiny, others big
    slice_colors = [C_ANCHOR, "#60a5fa", "#34d399", "#facc15", "#f97316"]
    x_cursor = bar_left_x
    for w, c in zip(slice_widths, slice_colors):
        ax.add_patch(Rectangle((x_cursor, bar_y), w, bar_h, facecolor=c,
                                edgecolor="black", linewidth=0.6))
        x_cursor += w
    ax.text(bar_left_x + bar_width / 2, bar_y - 0.3,
            r"$e_a^{(L)}$", ha="center", fontsize=10)

    # For e_b^(L)
    bar_left_x_b = 5.7
    x_cursor = bar_left_x_b
    slice_widths_b = [0.15, 0.85, 1.15, 1.0, 0.85]
    slice_colors_b = [C_ANCHOR, "#a78bfa", "#22d3ee", "#fb7185", "#84cc16"]
    for w, c in zip(slice_widths_b, slice_colors_b):
        ax.add_patch(Rectangle((x_cursor, bar_y), w, bar_h, facecolor=c,
                                edgecolor="black", linewidth=0.6))
        x_cursor += w
    ax.text(bar_left_x_b + bar_width / 2, bar_y - 0.3,
            r"$e_b^{(L)}$", ha="center", fontsize=10)

    # Annotate m's tiny red contribution
    ax.annotate(r"$m$'s signal $\sim 5\%$", xy=(bar_left_x + 0.1, bar_y + bar_h + 0.05),
                xytext=(2.0, 3.0), fontsize=8, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))

    # Arrow from drugs to embedding bars
    ax.annotate("", xy=(bar_left_x + bar_width / 2, bar_y + bar_h + 0.05),
                xytext=(a_pos[0], a_pos[1] - 0.4),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8,
                                connectionstyle="arc3,rad=0.0"))
    ax.annotate("", xy=(bar_left_x_b + bar_width / 2, bar_y + bar_h + 0.05),
                xytext=(b_pos[0], b_pos[1] - 0.4),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8,
                                connectionstyle="arc3,rad=0.0"))

    # Bottom box: MLP can't recover
    ax.text(5, 0.6,
            r"MLP$([e_a^{(L)}; e_b^{(L)}])$ cannot tell that $a, b$ share $m$.",
            ha="center", fontsize=8.5, style="italic", color="darkred")
    ax.text(5, 0.2,
            "L-layer message passing dilutes the structural signal.",
            ha="center", fontsize=8, style="italic")


# ----------------------------------------------------------------------------
# Panel (c): Hyper-Edge + NBFNet Preservation
# ----------------------------------------------------------------------------
def draw_panel_c(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(5, 5.6, "(c) Hyper-Edge + NBFNet — preservation",
            ha="center", fontsize=11, fontweight="bold")

    # Drugs: NO learnable embedding (dashed, gray)
    a_pos = (1.5, 3)
    b_pos = (8.5, 3)
    ax.add_patch(Circle(a_pos, 0.4, facecolor="white", edgecolor=C_DRUG_COLD,
                        linewidth=1.5, linestyle="--"))
    ax.text(*a_pos, "a", ha="center", va="center", fontsize=12, fontweight="bold",
            color=C_DRUG_COLD)
    ax.add_patch(Circle(b_pos, 0.4, facecolor="white", edgecolor=C_DRUG_COLD,
                        linewidth=1.5, linestyle="--"))
    ax.text(*b_pos, "b", ha="center", va="center", fontsize=12, fontweight="bold",
            color=C_DRUG_COLD)

    ax.text(1.5, 2.3, "no embed.", ha="center", fontsize=7,
            style="italic", color=C_DRUG_COLD)
    ax.text(8.5, 2.3, "no embed.", ha="center", fontsize=7,
            style="italic", color=C_DRUG_COLD)

    # A_τ set highlighted as yellow box around m
    ax.add_patch(FancyBboxPatch((4.2, 2.55), 1.6, 1.1,
                                 boxstyle="round,pad=0.05",
                                 facecolor=C_HIGHLIGHT, alpha=0.35,
                                 edgecolor="darkorange", linewidth=1.4))
    m_pos = (5, 3)
    ax.add_patch(Circle(m_pos, 0.32, facecolor=C_ANCHOR, edgecolor="black",
                        linewidth=1.2))
    ax.text(*m_pos, "m", ha="center", va="center", fontsize=11, fontweight="bold",
            color="white")
    ax.text(5, 3.85, r"$A_\tau(a, b) = \{m\}$", ha="center", fontsize=9,
            fontweight="bold", color="darkorange")

    # NBFNet path from a → m → b shown as colored arrows
    ax.annotate("", xy=(m_pos[0] - 0.4, m_pos[1]),
                xytext=(a_pos[0] + 0.4, a_pos[1]),
                arrowprops=dict(arrowstyle="->", color=C_PATH, lw=2.2))
    ax.annotate("", xy=(b_pos[0] - 0.4, b_pos[1]),
                xytext=(m_pos[0] + 0.4, m_pos[1]),
                arrowprops=dict(arrowstyle="->", color=C_PATH, lw=2.2))
    ax.text(3.3, 3.3, r"$h(a \to m)$", ha="center", fontsize=8,
            color=C_PATH, fontweight="bold")
    ax.text(6.7, 3.3, r"$h(m \to b)$", ha="center", fontsize=8,
            color=C_PATH, fontweight="bold")

    # h_τ box
    ax.add_patch(FancyBboxPatch((2.5, 0.6), 5, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor="#ddd6fe", alpha=0.7,
                                 edgecolor=C_PATH, linewidth=1.4))
    ax.text(5, 1.4,
            r"$h_\tau(a, b) = \mathrm{AttnPool}_{m \in A_\tau} \;\phi(\cdot)$",
            ha="center", fontsize=10)
    ax.text(5, 0.9,
            "Structural variable stays explicit. No smoothing.",
            ha="center", fontsize=8, style="italic")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 12.5))
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c(axes[2])

    fig.suptitle(
        "Figure 1.  Cold-start DDI requires preserving the KG structural variable\n"
        r"$A_\tau(a, b)$ — the typed middle common neighbors of the drug pair.",
        fontsize=12, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.01,
        "Endpoint GNNs smooth this variable into learned drug embeddings, where it becomes inaccessible to the pair classifier.\n"
        "Our hyper-edge + NBFNet framework reifies $A_\\tau$ as an explicit structural input, preserving it through the entire forward pass.",
        ha="center", fontsize=8.5, style="italic",
    )

    plt.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))

    out_dir = Path(__file__).resolve().parents[2] / "Notes" / "Log" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / "figure_1_common_neighbor_preservation.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "figure_1_common_neighbor_preservation.pdf",
                bbox_inches="tight")
    print(f"saved -> {out_dir / 'figure_1_common_neighbor_preservation.png'}")
    print(f"saved -> {out_dir / 'figure_1_common_neighbor_preservation.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
