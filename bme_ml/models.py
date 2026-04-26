"""Tabular classifiers for BP classification.

Three models live here, all trained on subject-level GroupKFold splits:

  - `build_rf` / `tune_rf` — Random Forest baseline.
  - `build_xgb` / `tune_xgb` — XGBoost; usually the strongest tabular model.
  - `build_ptt_logreg` — single-feature logistic regression on PTT alone,
    matching the writeup's "PTT regression" baseline.

`Xy` drops rows with NaN in any engineered feature; `Xy_ptt_only` drops
just on `ptt_ms`. All three return `(X, y)` arrays with `y` typed `int64`,
so they work for binary or multi-class labels interchangeably.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import FEATURE_NAMES

DEFAULT_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", "log2"],
}

# XGBoost defaults. `tree_method="hist"` is the modern fast histogram
# splitter; `multi:softprob` returns per-class probabilities so we can
# derive a hypertensive AUROC after the fact.
DEFAULT_PARAMS_XGB = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42,
)

PARAM_GRID_XGB = {
    "n_estimators": [200, 500],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def build_rf(**overrides) -> RandomForestClassifier:
    params = {**DEFAULT_PARAMS, **overrides}
    return RandomForestClassifier(**params)


def build_xgb(**overrides) -> XGBClassifier:
    params = {**DEFAULT_PARAMS_XGB, **overrides}
    return XGBClassifier(**params)


def build_ptt_logreg() -> Pipeline:
    """Logistic regression on `ptt_ms` only — the writeup's PTT baseline.

    `multi_class` is left at sklearn's default ('auto'), which selects
    multinomial when `solver='lbfgs'` and >2 classes are present.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def Xy(features: pd.DataFrame, label_col: str = "label_binary") -> tuple[np.ndarray, np.ndarray]:
    """Extract the full engineered-feature matrix and labels, dropping any
    rows where any feature or the label is NaN.
    """
    cols = [c for c in FEATURE_NAMES if c in features.columns]
    sub = features.dropna(subset=cols + [label_col])
    return sub[cols].to_numpy(dtype=np.float32), sub[label_col].to_numpy(dtype=np.int64)


def Xy_ptt_only(
    features: pd.DataFrame, label_col: str = "label_binary"
) -> tuple[np.ndarray, np.ndarray]:
    """Single-feature variant of `Xy` — keeps only `ptt_ms`."""
    sub = features.dropna(subset=["ptt_ms", label_col])
    return sub[["ptt_ms"]].to_numpy(dtype=np.float32), sub[label_col].to_numpy(dtype=np.int64)


def _grid_search(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
    param_grid: dict,
    *,
    scoring: str = "f1_macro",
    verbose: int = 1,
) -> GridSearchCV:
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=list(cv_splits),
        scoring=scoring,
        n_jobs=-1,
        verbose=verbose,
        refit=True,
    )
    grid.fit(X, y)
    return grid


def tune_rf(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    param_grid: dict | None = None,
    scoring: str = "f1_macro",
    verbose: int = 1,
) -> GridSearchCV:
    """Run GridSearchCV for RF with the provided CV iterator.

    Pass `cv_splits` as a list of (train_idx, val_idx) tuples — typically
    the output of `GroupKFold.split()` applied to your training subjects.
    """
    return _grid_search(
        build_rf(), X, y, cv_splits, param_grid or PARAM_GRID, scoring=scoring, verbose=verbose
    )


def tune_xgb(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    param_grid: dict | None = None,
    scoring: str = "f1_macro",
    verbose: int = 1,
) -> GridSearchCV:
    """GridSearchCV for XGBoost. Same calling convention as `tune_rf`."""
    return _grid_search(
        build_xgb(),
        X,
        y,
        cv_splits,
        param_grid or PARAM_GRID_XGB,
        scoring=scoring,
        verbose=verbose,
    )
