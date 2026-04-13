"""Simple CNN for MNIST 28x28 -> 10 classes. Same interface as GCN: setup, train_step, eval_step, state_dict, load_state_dict."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict


class CNN(nn.Module):
    """Small CNN: (B, 1, 28, 28) -> (B, 10)."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout = dropout
        # 28 -> 26 -> 13 -> 11 -> 5
        self.fc = nn.Linear(64 * 5 * 5, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 13
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 5
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)


class CNNMethod:
    """Method wrapper: setup, train_step, eval_step, state_dict, load_state_dict. Uses batch['x'], batch['y'], batch['train_mask'] (all True for MNIST)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.device = None

    def setup(self, cfg: dict, runtime: dict) -> None:
        self.device = runtime.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        num_classes = cfg.get("model", {}).get("num_classes", 10)
        dropout = cfg.get("model", {}).get("dropout", 0.25)
        self.model = CNN(num_classes=num_classes, dropout=dropout).to(self.device)
        lr = cfg.get("training", {}).get("learning_rate", 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, batch: dict, runtime: dict) -> Dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad()
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)
        train_mask = batch["train_mask"].to(self.device)
        out = self.model(x)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        self.optimizer.step()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            acc = (pred[train_mask] == y[train_mask]).float().mean().item()
        return {"loss": loss.item(), "grad_norm": grad_norm, "metrics": {"train_acc": acc}, "preds": pred, "targets": y}

    def eval_step(self, batch: dict, runtime: dict) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            out = self.model(x)
            split = batch.get("split", "val")
            if split == "train":
                mask = batch["train_mask"].to(self.device)
            elif split == "val":
                mask = batch["val_mask"].to(self.device)
            else:
                mask = batch["test_mask"].to(self.device)
            loss = F.cross_entropy(out[mask], y[mask]).item()
            pred = out.argmax(dim=1)
        return {
            "loss": loss,
            "metrics": {"accuracy": (pred[mask] == y[mask]).float().mean().item()},
            "preds": pred.cpu(),
            "targets": y.cpu(),
            "mask": mask.cpu(),
        }

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict() if self.model else {}}

    def load_state_dict(self, state: dict) -> None:
        if self.model and "model" in state:
            self.model.load_state_dict(state["model"])
