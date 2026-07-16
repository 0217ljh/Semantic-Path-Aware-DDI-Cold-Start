"""KnowDDI baseline ``model`` package (independent copy; DGL-2.x migrated).

Re-exports the classifier + its two sub-modules (GraphSAGE-once embedding + the
GSL block) so callers can ``from baseline.knowddi.model import Classifier_model``.
"""
from __future__ import annotations

from .Classifier_model import Classifier_model
from .GraphSAGE import GraphSAGE
from .gsl_model import gsl_model

__all__ = ["Classifier_model", "GraphSAGE", "gsl_model"]
