"""Common ABC and registry for the eight ColdDDI baselines.

Design summary (locked in state log)
-----------------------------------
* :meth:`BaselineModel.fit` consumes a :class:`PairDataset` directly.
  Implementations are responsible for their own iteration / batching;
  the ABC does not impose a DataLoader. ``kg`` is passed as a keyword
  so KG-free baselines (e.g. DeepDDI) can ignore it.

* :meth:`BaselineModel.predict_proba` takes a plain
  ``DataFrame[drug_a_id, drug_b_id]`` and returns a 1-D
  ``np.ndarray`` of P(positive). This is the single contract baselines
  plug into the evaluator.

* :meth:`BaselineModel.save` is **free-form** — each baseline picks the
  serialization that suits it (a `.pt` state dict, a HuggingFace
  directory, a pickle, …). The only convention is that ``save(path)``
  must also drop a small ``manifest.json`` in the same directory listing
  ``{"baseline_name": cls.name, ...}`` so the top-level
  :func:`load_baseline` dispatcher can route to the right subclass's
  :meth:`load` classmethod.

* Subclasses register with :func:`register` (decorator) so the
  evaluator and ``load_baseline`` can find them by name.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


#: Filename written by every baseline's :meth:`save` next to its
#: checkpoint files. Read by :func:`load_baseline` to pick the subclass.
BASELINE_MANIFEST_FILENAME: str = "manifest.json"

#: Module-level registry mapping baseline name → subclass.
_REGISTRY: dict[str, type["BaselineModel"]] = {}

#: Declared baseline name → submodule import path. Used by
#: :func:`ensure_imported` so :func:`load_baseline` and the evaluator
#: can route to baselines that haven't been imported yet (a fresh
#: ``from baseline import load_baseline`` does not pull every
#: baseline's dependencies).
NAME_TO_MODULE: dict[str, str] = {
    "deepddi": "baseline.deepddi",
    "emergnn": "baseline.emergnn",
    "ssi_ddi": "baseline.ssi_ddi",
    "dsn_ddi": "baseline.dsn_ddi",
    "hdn_ddi": "baseline.hdn_ddi",
    "tiger": "baseline.tiger",
    "mkg_fenn": "baseline.mkg_fenn",
    "textddi": "baseline.textddi",
}


def ensure_imported(name: str) -> None:
    """Lazy-import the submodule that registers ``name`` if not already done."""
    if name in _REGISTRY:
        return
    module_name = NAME_TO_MODULE.get(name)
    if module_name is None:
        return  # unknown name — caller will surface a clear error
    try:
        __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Baseline {name!r} is mapped to {module_name!r} but importing "
            f"it failed: {exc}. Install the missing dependencies."
        ) from exc


def register(name: str):
    """Class decorator: register a :class:`BaselineModel` subclass under ``name``.

    Usage::

        @register("deepddi")
        class DeepDDI(BaselineModel):
            ...

    The decorated class also gets its ``name`` class attribute set to
    the registered string.
    """
    def decorator(cls: type["BaselineModel"]) -> type["BaselineModel"]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"Baseline name {name!r} is already registered to {_REGISTRY[name]!r}"
            )
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


class BaselineModel(ABC):
    """Common interface for all ColdDDI baseline models."""

    #: Registered name (set by :func:`register`).
    name: ClassVar[str] = "abstract"

    @abstractmethod
    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        """Train on ``train`` (positives + negatives via the dataset)."""

    @abstractmethod
    def predict_proba(
        self,
        pairs: "pd.DataFrame",
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> "np.ndarray":
        """Score ``pairs[["drug_a_id", "drug_b_id"]]`` → 1-D P(positive)."""

    @abstractmethod
    def save(self, path: "Path | str") -> None:
        """Serialize the trained model to ``path``.

        Implementations are free to pick any on-disk layout, but
        **must** drop a ``manifest.json`` (see
        :data:`BASELINE_MANIFEST_FILENAME`) in the same directory with
        at least the keys::

            {"baseline_name": cls.name, "version": "<your-version>"}

        so :func:`load_baseline` can dispatch back to this subclass.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: "Path | str") -> "BaselineModel":
        """Reload a baseline saved by :meth:`save`. Subclass-specific."""


def write_manifest(
    out_dir: Path,
    *,
    baseline_name: str,
    extra: dict | None = None,
) -> Path:
    """Helper for :meth:`BaselineModel.save` implementations.

    Writes a ``manifest.json`` under ``out_dir`` recording the baseline
    name and any subclass-specific extras (hyperparameters, train seed,
    etc.) so :func:`load_baseline` can route correctly later.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"baseline_name": baseline_name}
    if extra:
        payload.update(extra)
    manifest_path = out_dir / BASELINE_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return manifest_path


def load_baseline(path: Path | str) -> BaselineModel:
    """Auto-detect a baseline checkpoint's type and dispatch to its loader.

    Reads ``<path>/manifest.json`` to pick the registered subclass, then
    calls that subclass's :meth:`BaselineModel.load`.
    """
    p = Path(path)
    manifest_path = p / BASELINE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Baseline manifest not found at {manifest_path}. "
            "Did the baseline's save() write a manifest.json?"
        )
    payload = json.loads(manifest_path.read_text())
    name = payload.get("baseline_name")
    if not name:
        raise ValueError(
            f"{manifest_path} is missing the required `baseline_name` field."
        )
    ensure_imported(name)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown baseline {name!r}; registered baselines are "
            f"{sorted(_REGISTRY)} (declared: {sorted(NAME_TO_MODULE)})."
        )
    return _REGISTRY[name].load(p)


def list_baselines() -> list[str]:
    """Return every registered baseline name."""
    return sorted(_REGISTRY.keys())
