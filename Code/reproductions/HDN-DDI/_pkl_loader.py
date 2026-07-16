"""Reproduction-side pkl loader with detect → load (NO auto-build).

Per CLAUDE.md §"复现代码 (Reproduction) 规范" step 6: reproduction-side
pipeline does NOT do auto-detect-and-generate. It only does detect → load
(prefer official, fall back to mine) → **error** when both are missing,
with a clear message pointing the user at the manual builder CLI.

(auto-build belongs to baseline-side detection; the reproduction side's
job is to faithfully consume upstream's __official.pkl, with __mine.pkl
as a structural fallback for ad-hoc reruns where the user already built
it once via the builder CLI.)

Two candidate files:
* ``id_data_dict__official.pkl`` — upstream-shipped pkl, byte-exact to
  what HDN-DDI's paper used. Preferred when present.
* ``id_data_dict__mine.pkl`` — produced by our reproduction-side
  ``_Original-Dataset/necessary/build_hierarchical_pkl.py``
  reverse-engineering the official format. Structurally validated
  against the official on 5 invariants
  (``baseline/hdn_ddi/_data/necessary/validate_hierarchical_pkl.py``).

Detection priority:
  1. ``__official.pkl`` exists → load + log "USING OFFICIAL"
  2. else ``__mine.pkl`` exists → load + log "USING MINE (official missing)"
  3. else neither exists → **raise FileNotFoundError** with builder CLI hint

The log line is printed to stderr at module-load time so the user always
knows which pkl is feeding ``run_faithful.py``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

# This module lives next to data_preprocessing.py inside reproductions/HDN-DDI/.
_REPRO_DIR = Path(__file__).resolve().parent

# Builder lives inside _Original-Dataset/necessary/ per the new spec
# (CLAUDE.md §"复现代码 (Reproduction) 规范" §1 folder layout).
_BUILDER_SCRIPT = (
    _REPRO_DIR / "_Original-Dataset" / "necessary" / "build_hierarchical_pkl.py"
)

OFFICIAL_FILENAME = "id_data_dict__official.pkl"
MINE_FILENAME = "id_data_dict__mine.pkl"


def ensure_id_data_dict_pkl(
    necessary_dir: Path,
    *,
    upstream_smiles_csv: Path | None = None,
) -> dict:
    """Detect → load (prefer official). Raises on missing — does NOT auto-build.

    Args:
        necessary_dir: ``_Original-Dataset/necessary/`` — where the two
            candidate pkl files live.
        upstream_smiles_csv: ``_Original-Dataset/drug_smiles.csv`` —
            only used to render a helpful error message pointing at the
            builder CLI when both pkl files are missing. Defaults to
            ``<necessary_dir>.parent / "drug_smiles.csv"``.

    Returns:
        The loaded ``id_data_dict`` (upstream format:
        ``{drug_id: (smiles, Data)}`` or ``{drug_id: Data}``).

    Raises:
        FileNotFoundError: when both ``__official.pkl`` and ``__mine.pkl``
            are missing. Message contains the exact CLI to run the
            reproduction-side builder.

    Side effects:
        Prints one line to stderr identifying which file was used:
            "[hdn-repro] USING OFFICIAL pkl: <path>"
            "[hdn-repro] USING MINE pkl (official missing): <path>"
    """
    necessary_dir = Path(necessary_dir).resolve()
    official_path = necessary_dir / OFFICIAL_FILENAME
    mine_path = necessary_dir / MINE_FILENAME

    # 1. Prefer official.
    if official_path.is_file():
        print(
            f"[hdn-repro] USING OFFICIAL pkl: {official_path}",
            file=sys.stderr,
            flush=True,
        )
        with official_path.open("rb") as f:
            return pickle.load(f)

    # 2. Fallback to mine if it exists.
    if mine_path.is_file():
        print(
            f"[hdn-repro] USING MINE pkl (official missing): {mine_path}",
            file=sys.stderr,
            flush=True,
        )
        with mine_path.open("rb") as f:
            return pickle.load(f)

    # 3. Neither exists — REFUSE TO AUTO-BUILD on the reproduction side.
    #    Print actionable manual CLI for the user, then raise.
    smiles_csv = (
        Path(upstream_smiles_csv).resolve()
        if upstream_smiles_csv is not None
        else (necessary_dir.parent / "drug_smiles.csv").resolve()
    )
    hint = (
        f"To build __mine.pkl manually, run:\n"
        f"    python {_BUILDER_SCRIPT} \\\n"
        f"        --smiles-csv {smiles_csv} \\\n"
        f"        --id-col drug_id --smiles-col smiles \\\n"
        f"        --out {mine_path}\n"
        f"Or download the upstream __official.pkl into {necessary_dir}/."
    )
    raise FileNotFoundError(
        f"Neither {official_path.name} nor {mine_path.name} found in "
        f"{necessary_dir}. Reproduction pipeline does NOT auto-build "
        f"(per CLAUDE.md §复现代码规范 step 6).\n\n{hint}"
    )
