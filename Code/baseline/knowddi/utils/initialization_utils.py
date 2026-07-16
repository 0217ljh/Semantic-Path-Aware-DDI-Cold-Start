"""KnowDDI's ``utils/initialization_utils.py`` — experiment dir + model init.

Faithful port of KnowDDI's official
``Paper/Reference/Original-Code/KnowDDI/pytorch/utils/initialization_utils.py``
(``initialize_experiment`` :7, ``initialize_model`` :35). No DGL involved.

In the unified pipeline the wrapper builds the classifier directly (like the
SumGNN wrapper) to avoid cwd-relative path assumptions, but this function is kept
for parity with the official source and to support the ``load_model`` path.
INDEPENDENT copy under ``baseline.knowddi`` (CLAUDE.md file-independence).
"""
from __future__ import annotations

import json
import logging
import os

import torch


def initialize_experiment(params, file_name: str) -> None:
    """Make the experiment directory, set standard paths, initialize the logger."""
    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(file_name))), '..')
    global_exps_dir = os.path.join(params.main_dir, 'experiments')
    if not os.path.exists(global_exps_dir):
        os.makedirs(global_exps_dir)

    params.exp_dir = os.path.join(global_exps_dir, params.experiment_name)

    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)

    file_handler = logging.FileHandler(os.path.join(params.exp_dir, "log_train.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\t '.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
        json.dump({k: str(v) for k, v in vars(params).items()}, fout)


def initialize_model(params, model):
    """Initialize a fresh model or load a saved one (KnowDDI initialize_model:35)."""
    if getattr(params, "load_model", False) and os.path.exists(
            os.path.join(params.exp_dir, 'best_graph_classifier.pth')):
        logging.info('Loading existing model from %s'
                     % os.path.join(params.exp_dir, 'best_graph_classifier.pth'))
        classifier = torch.load(
            os.path.join(params.exp_dir, 'best_graph_classifier.pth'),
            weights_only=False).to(device=params.device)
    else:
        logging.info('No existing model found. Initializing new model..')
        classifier = model(params).to(device=params.device)

    return classifier


__all__ = ["initialize_experiment", "initialize_model"]
