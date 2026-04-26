"""Metrics, plots, and side-by-side comparison helpers."""

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


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_macro: float
    roc_auc: float
    pr_auc: float
    confusion: list[list[int]]  # [[tn, fp], [fn, tp]]


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
