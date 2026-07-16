"""KnowDDI's ``manager/trainer.py`` — training loop (+ mandatory TrainProgress log).

Ported from KnowDDI's official
``Paper/Reference/Original-Code/KnowDDI/pytorch/manager/trainer.py``. Preserves
KnowDDI's training protocol exactly:

  * optimizer = Adam, lr = ``params.lr`` (0.005), weight_decay = ``params.weight_decay_rate`` (orig :40)
  * scheduler = ExponentialLR(gamma = ``params.lr_decay_rate`` = 0.93) (orig :41)
  * criterion = CrossEntropyLoss for drugbank (orig :44)
  * clip_grad_norm_ max_norm=10 (orig :91)
  * best checkpoint on VAL macro-F1 (evaluator key ``'auc'``; orig :115-119)
  * early stop after ``early_stop_epoch`` non-improving evals (orig :129-131)

EVAL + LR-SCHEDULER CADENCE (restored to official, 2026-07-03): the official loop
evaluates the VALID set mid-epoch every ``params.eval_every_iter`` parameter UPDATES
(orig :103 ``if self.updates_counter % self.params.eval_every_iter == 0``) and steps
``ExponentialLR`` on the SAME cadence, at the END of each eval block (orig :133,
i.e. per-eval, not per-batch and not per-epoch). Early stop ``break``s inside the
batch loop once VAL macro-F1 has not improved for ``early_stop_epoch`` evals (orig
:127-131). This trainer replicates that EXACTLY: eval + best-ckpt + scheduler.step()
every ``eval_every_iter`` updates. The DrugBank official value is
``eval_every_iter=526`` (experiments/Drugbank/log_train.txt:2; argparse default
train.py:129 is also 526). Per-eval logging via ``TrainProgress.log_eval`` +
``eval_strategy="steps"`` (CLAUDE.md 训练进度日志规范).

INDEPENDENT copy (CLAUDE.md file-independence); no ``baseline.sumgnn`` import.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from ..utils.graph_utils import collate_dgl, move_batch_to_device_dgl

_ROOT = Path(__file__).resolve().parents[4]  # .../Code
import sys  # noqa: E402

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from my_code.utils.train_progress import TrainProgress  # noqa: E402


class Trainer(object):
    def __init__(self, params, model, train_data, valid_evaluator, test_evaluator) -> None:
        self.params = params
        self.graph_classifier = model

        self.train_data = train_data
        self.valid_evaluator = valid_evaluator
        self.test_evaluator = test_evaluator

        self.batch_size = params.batch_size
        self.collate_fn = collate_dgl
        self.num_workers = params.num_workers
        self.updates_counter = 0
        self.early_stop = 0
        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        self.optimizer = Adam(self.graph_classifier.parameters(), lr=params.lr,
                              weight_decay=params.weight_decay_rate)
        self.scheduler = ExponentialLR(self.optimizer, params.lr_decay_rate)

        if params.dataset == 'drugbank':
            self.criterion = nn.CrossEntropyLoss()
        elif params.dataset == 'BioSNAP':
            self.criterion = nn.BCELoss(reduce=False)
        self.move_batch_to_device = move_batch_to_device_dgl
        self.reset_training_state()

        # official eval + scheduler cadence: every ``eval_every_iter`` UPDATES
        # (orig trainer.py:103 / :133). DrugBank official value 526
        # (experiments/Drugbank/log_train.txt:2, argparse default train.py:129).
        self.eval_every_iter = int(getattr(params, "eval_every_iter", 526))

        self.prog = TrainProgress(
            total_epochs=params.num_epochs,
            log_step_every=getattr(params, "log_step_every", 50),
            prefix="[knowddi] ",
            eval_strategy="steps",
            eval_steps=self.eval_every_iter,
            save_strategy="no",
        )

    def train_epoch(self):
        total_loss = 0
        all_labels = []
        all_scores = []
        b_idx = 0

        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers, collate_fn=self.collate_fn)
        self.graph_classifier.train()

        for b_idx, batch in enumerate(train_dataloader):
            if self.params.dataset == 'drugbank':
                subgraph_data, relation_labels, polarity = self.move_batch_to_device(
                    batch, self.params.device, multi_type=1)
            elif self.params.dataset == 'BioSNAP':
                subgraph_data, relation_labels, polarity = self.move_batch_to_device(
                    batch, self.params.device, multi_type=2)
            self.optimizer.zero_grad()
            scores = self.graph_classifier(subgraph_data)

            if self.params.dataset == 'drugbank':
                loss = self.criterion(scores, relation_labels)
            elif self.params.dataset == 'BioSNAP':
                m = nn.Sigmoid()
                scores = m(scores)
                polarity = polarity.unsqueeze(1)
                loss_train = self.criterion(scores, relation_labels * polarity)
                loss = torch.sum(loss_train * relation_labels)
            loss.backward()
            clip_grad_norm_(self.graph_classifier.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            self.updates_counter += 1
            self.prog.step(loss.item())
            with torch.no_grad():
                total_loss += loss.item()
                if self.params.dataset != 'BioSNAP':
                    label_ids = relation_labels.to('cpu').numpy()
                    all_labels += label_ids.flatten().tolist()
                    all_scores += torch.argmax(scores, dim=1).cpu().flatten().tolist()

            # Mid-epoch VALID eval + best-ckpt + LR step, every ``eval_every_iter``
            # UPDATES (official trainer.py:103). Scheduler steps at the END of the eval
            # block, i.e. per-eval (official trainer.py:133), NOT per-batch/per-epoch.
            # Early stop breaks out of the batch loop (official trainer.py:127-131).
            if self.updates_counter % self.eval_every_iter == 0:
                result, _ = self.valid_evaluator.eval()
                self.prog.log_eval({"val_macro_f1": result['auc'],
                                    "val_micro_f1": result['auc_pr']}, scope="step")
                if result['auc'] >= self.best_metric:
                    self.save_classifier()
                    self.early_stop = 0
                    self.best_metric = result['auc']
                    self.not_improved_count = 1
                else:
                    self.not_improved_count += 1
                    if self.not_improved_count >= self.params.early_stop_epoch:
                        self.early_stop = 1
                        break  # official: break BEFORE last_metric/scheduler.step()
                self.last_metric = result['auc']
                self.scheduler.step()

        if self.params.dataset != 'BioSNAP':
            auc = metrics.f1_score(all_labels, all_scores, average='macro')
            auc_pr = metrics.f1_score(all_labels, all_scores, average='micro')
            f1 = metrics.f1_score(all_labels, all_scores, average=None)
            return total_loss / max(b_idx, 1), auc, auc_pr, f1
        else:
            return total_loss / max(b_idx, 1), 0, 0, 0

    def save_classifier(self) -> None:
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))
        logging.info('Better models found w.r.t accuracy. Saved it!')

    def reset_training_state(self) -> None:
        self.best_metric = 0
        self.test_best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 1

    def train(self) -> None:
        self.reset_training_state()
        for epoch in range(1, self.params.num_epochs + 1):
            self.prog.epoch_start(epoch - 1)
            time_start = time.time()
            # eval / best-ckpt / scheduler.step / early-stop all happen mid-epoch inside
            # train_epoch(), every ``eval_every_iter`` updates (official trainer.py:102-133).
            loss, auc, auc_pr, f1 = self.train_epoch()
            time_elapsed = time.time() - time_start

            self.prog.epoch_end(extra={"mean_loss": loss, "train_macro_f1": auc,
                                       "best_val_macro_f1": self.best_metric})
            logging.info(f'Epoch {epoch} with loss: {loss}, training macro-F1: {auc}, '
                         f'best validation macro-F1: {self.best_metric} in {time_elapsed}')
            # Early stop is set inside train_epoch when VAL macro-F1 has not improved for
            # ``early_stop_epoch`` evals; halt the epoch loop (landed behavior retained).
            if self.early_stop == 1:
                logging.info(f"Validation performance didn't improve for "
                             f"{self.params.early_stop_epoch} evals. Training stops.")
                break


__all__ = ["Trainer"]
