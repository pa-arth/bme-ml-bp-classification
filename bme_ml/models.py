"""Random Forest baseline for binary BP classification.

`build_rf` returns the bare estimator. `tune_rf` runs a small grid search
inside the user-provided cross-validation iterator (so the CV uses
subject-level group folds, not sklearn's default random KFold).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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


def build_rf(**overrides) -> RandomForestClassifier:
    params = {**DEFAULT_PARAMS, **overrides}
    return RandomForestClassifier(**params)


def Xy(features: pd.DataFrame, label_col: str = "label_binary") -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and labels, dropping any rows with NaN features.
    Returns `(X, y)` arrays plus prints how many rows were dropped.
    """
    cols = [c for c in FEATURE_NAMES if c in features.columns]
    sub = features.dropna(subset=cols + [label_col])
    return sub[cols].to_numpy(dtype=np.float32), sub[label_col].to_numpy(dtype=np.int64)


def tune_rf(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    param_grid: dict | None = None,
    scoring: str = "f1_macro",
    verbose: int = 1,
) -> GridSearchCV:
    """Run GridSearchCV over `param_grid` with the provided CV iterator.

    Pass `cv_splits` as a list of (train_idx, val_idx) tuples — typically the
    output of GroupKFold.split() applied to your training subjects.
    """
    grid = GridSearchCV(
        estimator=build_rf(),
        param_grid=param_grid or PARAM_GRID,
        cv=list(cv_splits),
        scoring=scoring,
        n_jobs=-1,
        verbose=verbose,
        refit=True,
    )
    grid.fit(X, y)
    return grid
