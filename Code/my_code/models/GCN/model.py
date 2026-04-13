"""Simple 2-layer GCN for node classification. Cora: in_dim=1433, num_classes=7."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

try:
    from torch_geometric.nn import GCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


if HAS_PYG:
    class GCN(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            self.dropout = dropout

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return x
else:
    GCN = None


class GCNMethod:
    """Method wrapper: setup, train_step, eval_step, state_dict, load_state_dict."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.device = None

    def setup(self, cfg: dict, runtime: dict) -> None:
        if not HAS_PYG:
            raise ImportError("torch_geometric required for GCN")
        self.device = runtime.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        in_channels = cfg.get("model", {}).get("in_channels", 1433)
        hidden = cfg.get("model", {}).get("hidden_channels", 16)
        out_channels = cfg.get("model", {}).get("out_channels", 7)
        dropout = cfg.get("model", {}).get("dropout", 0.5)
        self.model = GCN(in_channels, hidden, out_channels, dropout).to(self.device)
        lr = cfg.get("training", {}).get("learning_rate", 0.01)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, batch: dict, runtime: dict) -> Dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad()
        x = batch["x"].to(self.device)
        edge_index = batch["edge_index"].to(self.device)
        y = batch["y"].to(self.device)
        train_mask = batch["train_mask"].to(self.device)
        out = self.model(x, edge_index)
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
            edge_index = batch["edge_index"].to(self.device)
            y = batch["y"].to(self.device)
            out = self.model(x, edge_index)
            split = batch.get("split", "val")
            if split == "train":
                mask = batch["train_mask"].to(self.device)
            elif split == "val":
                mask = batch["val_mask"].to(self.device)
            else:
                mask = batch["test_mask"].to(self.device)
            loss = F.cross_entropy(out[mask], y[mask]).item()
            pred = out.argmax(dim=1)
            acc = (pred[mask] == y[mask]).float().mean().item()
        return {
            "loss": loss,
            "metrics": {"accuracy": acc},
            "preds": pred.cpu(),
            "targets": y.cpu(),
            "mask": mask.cpu(),
        }

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict() if self.model else {}}

    def load_state_dict(self, state: dict) -> None:
        if self.model and "model" in state:
            self.model.load_state_dict(state["model"])
