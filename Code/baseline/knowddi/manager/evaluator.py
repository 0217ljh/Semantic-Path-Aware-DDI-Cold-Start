"""KnowDDI's ``manager/evaluator.py`` — multiclass (DrugBank) evaluator.

Faithful port of KnowDDI's official
``Paper/Reference/Original-Code/KnowDDI/pytorch/manager/evaluator.py``
(``Evaluator_multiclass`` :30). The DrugBank main metric is macro-F1, reported
under the key ``'auc'`` (orig :64 — the name is historical; the VALUE is
``metrics.f1_score(..., average='macro')``). Best-ckpt selection in the trainer
keys on this ``'auc'`` field, i.e. VAL macro-F1.

``Evaluator_multilabel`` (BioSNAP) is ported for parity but unused by the Phase-1
multiclass pipeline. INDEPENDENT copy (CLAUDE.md file-independence).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import (accuracy_score, average_precision_score, roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.graph_utils import collate_dgl, move_batch_to_device_dgl


class Evaluator_multiclass:
    """Drugbank multiclass evaluator (macro-F1 under key 'auc').

    OOV-in-train sentinel handling: the builder writes a SENTINEL gold id == K_train
    (one beyond the fc head's 0..K_train-1 range) for valid/test gold classes unseen in
    train (build_knowddi_data._write_split). ``argmax`` over K_train logits can never
    equal K_train, so such rows are always mispredicted. ``metrics.f1_score`` here runs
    on plain Python lists (no fixed-size array indexed by label), so sklearn simply
    treats K_train as a gold class with recall 0 (counted wrong, dragging macro-F1
    down) rather than raising — no explicit guard needed. This keeps the internal VAL
    best-ckpt selection from being corrupted by OOV rows.
    """

    def __init__(self, params, classifier, data, is_test: bool = False) -> None:
        self.params = params
        self.graph_classifier = classifier
        self.data = data
        self.global_graph = data.global_graph
        self.move_batch_to_device = move_batch_to_device_dgl
        self.collate_fn = collate_dgl
        self.num_workers = params.num_workers
        self.is_test = is_test
        self.eval_times = 0
        self.current_epoch = 0

    def eval(self):
        self.eval_times += 1
        scores = []
        labels = []
        self.current_epoch += 1
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False,
                                num_workers=self.num_workers, collate_fn=self.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in tqdm(enumerate(dataloader)):
                data, r_labels, polarity = self.move_batch_to_device(batch, self.params.device, multi_type=1)
                score = self.graph_classifier(data)

                label_ids = r_labels.to('cpu').numpy()
                labels += label_ids.flatten().tolist()
                scores += torch.argmax(score, dim=1).cpu().flatten().tolist()

        auc = metrics.f1_score(labels, scores, average='macro')
        auc_pr = metrics.f1_score(labels, scores, average='micro')
        f1 = metrics.f1_score(labels, scores, average=None)
        kappa = metrics.cohen_kappa_score(labels, scores)
        return {'auc': auc, 'auc_pr': auc_pr, 'k': kappa}, {'f1': f1}


class Evaluator_multilabel:
    """BioSNAP multilabel evaluator (ported for parity; unused in Phase-1)."""

    def __init__(self, params, classifier, data) -> None:
        self.params = params
        self.graph_classifier = classifier
        self.data = data
        self.global_graph = data.global_graph
        self.move_batch_to_device = move_batch_to_device_dgl
        self.collate_fn = collate_dgl
        self.num_workers = params.num_workers

    def eval(self):
        pred_class = {}
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False,
                                num_workers=self.num_workers, collate_fn=self.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                data, r_labels, polarity = self.move_batch_to_device(batch, self.params.device, multi_type=2)
                score_pos = self.graph_classifier(data)

                m = nn.Sigmoid()
                pred = m(score_pos)
                labels = r_labels.detach().to('cpu').numpy()
                preds = pred.detach().to('cpu').numpy()
                polarity = polarity.detach().to('cpu').numpy()
                for (label, pred, pol) in zip(labels, preds, polarity):
                    for i, (l, p) in enumerate(zip(label, pred)):
                        if l == 1:
                            if i in pred_class:
                                pred_class[i]['pred'] += [p]
                                pred_class[i]['pol'] += [pol]
                                pred_class[i]['pred_label'] += [1 if p > 0.5 else 0]
                            else:
                                pred_class[i] = {'pred': [p], 'pol': [pol], 'pred_label': [1 if p > 0.5 else 0]}

        roc_auc = [roc_auc_score(pred_class[l]['pol'], pred_class[l]['pred']) for l in pred_class]
        prc_auc = [average_precision_score(pred_class[l]['pol'], pred_class[l]['pred']) for l in pred_class]
        ap = [accuracy_score(pred_class[l]['pol'], pred_class[l]['pred_label']) for l in pred_class]
        return ({'auc': np.mean(roc_auc), 'auc_pr': np.mean(prc_auc), 'f1': np.mean(ap)},
                {"auc_all": roc_auc, "aupr_all": prc_auc, "f1_all": ap})


__all__ = ["Evaluator_multiclass", "Evaluator_multilabel"]
