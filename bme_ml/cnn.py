"""1D CNN baseline for raw-signal BP classification.

Consumes 8-second `(2, 1000)` PPG+ECG segments and outputs a binary BP
class. Trained on the same subject-level splits as the RF, so RF / SVM /
CNN are directly comparable on the same held-out test subjects.

Architecture is a small VGG-style 1D ConvNet — four conv blocks with
increasing channel counts, average pooling, and a 2-layer head. About
~250k parameters; trains in a few minutes per epoch on a 4070 with the
full UCI dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import WINDOW_SAMPLES


class SegmentDataset(Dataset):
    """Lazy reader over `signals.h5`, restricted to a subject ID list.

    HDF5 reads release the GIL so multi-worker DataLoader is fine. We do
    per-channel z-score normalization on the fly — fast on the 4070 and
    avoids storing a separate normalized cache.
    """

    def __init__(self, h5_path, indices: np.ndarray, labels: np.ndarray):
        self.h5_path = str(h5_path)
        self.indices = indices.astype(np.int64)
        self.labels = labels.astype(np.int64)
        self._h5 = None  # opened lazily per-worker for fork safety

    def _h5f(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        sig = self._h5f()["signals"][idx]  # (2, 1000)
        # Per-channel z-score (robust enough; signals already bandpassed).
        sig = sig - sig.mean(axis=1, keepdims=True)
        std = sig.std(axis=1, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        sig = (sig / std).astype(np.float32)
        return torch.from_numpy(sig), int(self.labels[i])


def build_indices_for_subjects(h5_path, subject_ids: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (row_indices, labels) for all H5 rows whose subject_id is in `subject_ids`."""
    wanted = set(subject_ids)
    with h5py.File(h5_path, "r") as f:
        subj = f["subject_id"][:]  # bytes
        labels = f["label_binary"][:]
    subj_str = np.array([s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in subj])
    mask = np.fromiter((s in wanted for s in subj_str), dtype=bool, count=len(subj_str))
    idx = np.nonzero(mask)[0]
    return idx, labels[idx]


class CNN1D(nn.Module):
    """Small 1D ConvNet for `(2, 1000)` segments → binary logits."""

    def __init__(self, in_channels: int = 2, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        def block(c_in, c_out, kernel=7):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv1d(c_out, c_out, kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )

        self.body = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, 128, kernel=5),
        )
        # After 4 pools on 1000 samples => 1000 / 16 = 62 timesteps.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x = self.pool(x)
        return self.head(x)


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: TrainConfig,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """Standard supervised training loop. Returns the model with the best
    val macro-F1 weights loaded (or the final epoch's if no val_loader)."""
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    best_state = None
    best_f1 = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        train_loss = running / max(n, 1)

        msg = f"epoch {epoch+1:02d}/{cfg.epochs}  train_loss={train_loss:.4f}"
        if val_loader is not None:
            f1 = _val_f1(model, val_loader, device)
            msg += f"  val_macro_f1={f1:.4f}"
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _val_f1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    from sklearn.metrics import f1_score

    model.eval()
    all_y, all_p = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        all_p.append(logits.argmax(dim=1).cpu().numpy())
        all_y.append(y.numpy())
    return float(f1_score(np.concatenate(all_y), np.concatenate(all_p), average="macro"))


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred, y_proba_class1)."""
    model.eval()
    ys, preds, probas = [], [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds.append(logits.argmax(dim=1).cpu().numpy())
        probas.append(prob)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probas)
