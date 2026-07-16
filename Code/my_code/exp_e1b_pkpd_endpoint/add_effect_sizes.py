"""Augment E1b report with effect-size measures (Cramer's V).

Codex round 5 methodology critique: with n=6000, chi-square is significant
even at small effect sizes. Report Cramer's V to characterize magnitude.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

# Hardcoded contingency from E1b v2 result
LAYERS = ["PK_layer", "PD_layer", "BioProcess", "Other"]
TABLE = np.array([
    [1611, 1353, 0, 112],  # PK
    [1071, 1823, 0, 383],  # PD
], dtype=np.float64)


def cramers_v(table: np.ndarray) -> tuple[float, float]:
    """Return (chi2, Cramer's V) for the given contingency table."""
    from scipy.stats import chi2_contingency
    # Drop zero columns (chi2 requires nonzero expected)
    keep = table.sum(axis=0) > 0
    t = table[:, keep]
    chi2, p, dof, exp = chi2_contingency(t)
    n = t.sum()
    r, c = t.shape
    cv = np.sqrt(chi2 / (n * (min(r, c) - 1)))
    return chi2, p, cv


def main():
    chi2, p, cv = cramers_v(TABLE)
    print("E1b effect-size augmentation")
    print(f"  Chi-square: {chi2:.2f}")
    print(f"  p-value:    {p:.2e}")
    print(f"  Cramer's V: {cv:.4f}  (rule of thumb: 0.1=small, 0.3=medium, 0.5=large)")

    # Per-class layer share with bootstrap CIs
    from sklearn.utils import resample
    rng = np.random.default_rng(42)

    # Reconstruct per-pair labels from contingency for bootstrap
    # For each row (class), expand into per-pair tokens (layer choice)
    samples = {"PK": [], "PD": []}
    for i, cls in enumerate(["PK", "PD"]):
        for j, layer in enumerate(LAYERS):
            count = int(TABLE[i, j])
            samples[cls].extend([layer] * count)

    def share_with_ci(sample, layer_of_interest, n_boot=1000):
        sample = np.array(sample)
        shares = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(sample), size=len(sample))
            shares.append((sample[idx] == layer_of_interest).mean())
        lo, hi = np.percentile(shares, [2.5, 97.5])
        return sample.tolist().count(layer_of_interest) / len(sample), lo, hi

    print("\nPer-class layer shares with 95% bootstrap CI:")
    for cls in ["PK", "PD"]:
        s = samples[cls]
        if not s:
            continue
        pk_share, pk_lo, pk_hi = share_with_ci(s, "PK_layer")
        pd_share, pd_lo, pd_hi = share_with_ci(s, "PD_layer")
        print(f"  {cls} (n={len(s)}):")
        print(f"    PK_layer: {pk_share:.3f} [{pk_lo:.3f}, {pk_hi:.3f}]")
        print(f"    PD_layer: {pd_share:.3f} [{pd_lo:.3f}, {pd_hi:.3f}]")
        # Margin (PK - PD) with CI
        diffs = []
        for _ in range(1000):
            idx = rng.integers(0, len(s), size=len(s))
            sub = np.array(s)[idx]
            d = (sub == "PK_layer").mean() - (sub == "PD_layer").mean()
            diffs.append(d)
        d_mean = np.mean(diffs)
        d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
        print(f"    PK_share - PD_share margin: {d_mean:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]")


if __name__ == "__main__":
    main()
