"""Metrics, plots, and side-by-side comparison helpers.

Two metric flavors live here:

  - `BinaryMetrics` / `evaluate_binary` — Normal vs Abnormal.
  - `MultiClassMetrics` / `evaluate_multiclass` — AHA 3-class (Normal /
    Elevated / Hypertensive). Includes a "hypertensive AUROC" that
    collapses the multi-class output to a binary detector for class 2,
    so the writeup's "AUROC for detecting hypertensive cases" line has
    a sensible source even though the model itself is 3-class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .labels import MULTICLASS_HYPERTENSIVE


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_macro: float
    roc_auc: float
    pr_auc: float
    confusion: list[list[int]]  # [[tn, fp], [fn, tp]]


@dataclass(frozen=True)
class MultiClassMetrics:
    accuracy: float
    f1_macro: float
    confusion: list[list[int]]                 # 3x3
    hypertensive_auroc: float                  # P(class=Hyper) vs y==Hyper
    hypertensive_pr_auc: float
    hypertensive_recall: float                 # tp / (tp + fn) for class 2
    hypertensive_false_negative_rate: float    # 1 - recall_on_class_2


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> BinaryMetrics:
    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        confusion=confusion_matrix(y_true, y_pred).tolist(),
    )


def evaluate_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_full: np.ndarray,
    *,
    n_classes: int = 3,
    positive_class: int = MULTICLASS_HYPERTENSIVE,
) -> MultiClassMetrics:
    """Compute multi-class metrics + a binary "hypertensive detector" view.

    `y_proba_full` is `(n_samples, n_classes)` from `predict_proba` (or
    softmax output for a torch model). We extract `y_proba_full[:, positive_class]`
    as the score for the binary AUROC / PR-AUC.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()

    y_pos = (y_true == positive_class).astype(np.int64)
    proba_pos = y_proba_full[:, positive_class]

    # Per-class recall for the hypertensive class.
    pos_pred = (y_pred == positive_class)
    pos_true = (y_true == positive_class)
    tp = int((pos_pred & pos_true).sum())
    fn = int((~pos_pred & pos_true).sum())
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return MultiClassMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        confusion=cm,
        hypertensive_auroc=float(roc_auc_score(y_pos, proba_pos)) if y_pos.sum() and y_pos.sum() < len(y_pos) else float("nan"),
        hypertensive_pr_auc=float(average_precision_score(y_pos, proba_pos)) if y_pos.sum() else float("nan"),
        hypertensive_recall=float(recall_pos),
        hypertensive_false_negative_rate=float(1.0 - recall_pos),
    )


def compare(rows: Sequence[tuple[str, BinaryMetrics]]) -> pd.DataFrame:
    """Stack {model_name: metrics} into a tidy DataFrame for the results
    notebook. Confusion matrix is excluded — render that separately."""
    return pd.DataFrame(
        [
            {
                "model": name,
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1_macro": m.f1_macro,
                "roc_auc": m.roc_auc,
                "pr_auc": m.pr_auc,
            }
            for name, m in rows
        ]
    )


def compare_multiclass(rows: Sequence[tuple[str, MultiClassMetrics]]) -> pd.DataFrame:
    """3-class comparison table. Confusion is excluded — render separately."""
    return pd.DataFrame(
        [
            {
                "model": name,
                "accuracy": m.accuracy,
                "f1_macro": m.f1_macro,
                "hypertensive_auroc": m.hypertensive_auroc,
                "hypertensive_recall": m.hypertensive_recall,
                "false_negative_rate": m.hypertensive_false_negative_rate,
            }
            for name, m in rows
        ]
    )


def feature_importance(
    model, X_test: np.ndarray, y_test: np.ndarray, feature_names: list[str], n_repeats: int = 10
) -> pd.DataFrame:
    """Permutation importance — more reliable than Gini when features are
    correlated (e.g., HR and HRV here)."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, n_jobs=-1, random_state=42
    )
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
