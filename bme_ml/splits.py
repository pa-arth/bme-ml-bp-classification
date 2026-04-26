"""Subject-level train/test splits.

The UCI Cuff-Less BP dataset doesn't ship explicit subject IDs, so we treat
the *record* index (per-part) as the subject identifier. The dataset paper
notes that records correspond to subjects, with some subjects contributing
multiple records — for the calibration-free claim we conservatively treat
every record as its own subject. If you later get true subject IDs (e.g.
from MIMIC mapping), swap them in here without touching downstream code.

Two key invariants enforced below:
  1. No subject_id appears in both train and test.
  2. The persisted split is reproducible — same seed → same fold assignment.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


@dataclass(frozen=True)
class Splits:
    test_subjects: list[str]
    cv_folds: list[dict]  # each: {"train": [...], "val": [...]}

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "Splits":
        data = json.loads(Path(path).read_text())
        return cls(test_subjects=data["test_subjects"], cv_folds=data["cv_folds"])


def _subject_id(part: int, record: int) -> str:
    return f"p{part}_r{record}"


def add_subject_id(features: pd.DataFrame) -> pd.DataFrame:
    """Add a stable `subject_id` column derived from `part` and `record`."""
    if "subject_id" in features.columns:
        return features
    subject_id = features.apply(lambda r: _subject_id(int(r["part"]), int(r["record"])), axis=1)
    return features.assign(subject_id=subject_id)


def make_splits(
    features: pd.DataFrame,
    *,
    label_col: str = "label_binary",
    test_frac: float = 0.20,
    n_folds: int = 5,
    random_state: int = 42,
) -> Splits:
    """Carve out a held-out test set of subjects, then GroupKFold the rest.

    Stratifies the test split approximately by per-subject majority class
    so the test set isn't accidentally missing a class. Works for binary
    and multi-class labels — it iterates over whatever classes appear in
    `label_col`.
    """
    features = add_subject_id(features)
    # Per-subject majority class (mode). For ties, pandas returns the
    # smallest, which is fine for our purposes.
    subjects = features.groupby("subject_id")[label_col].agg(
        lambda s: int(s.mode().iloc[0])
    )
    subjects = subjects.reset_index()

    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
    # Stratify-ish: split once per class and union, to preserve class
    # balance in the test fold without needing StratifiedGroupKFold (which
    # exists but has subtle behavior for severe imbalance).
    test_ids: list[str] = []
    for cls in sorted(subjects[label_col].unique()):
        cls_subjects = subjects[subjects[label_col] == cls]["subject_id"].to_numpy()
        if len(cls_subjects) < 2:
            continue
        dummy_X = np.zeros((len(cls_subjects), 1))
        dummy_y = np.zeros(len(cls_subjects))
        for _, te in gss.split(dummy_X, dummy_y, groups=cls_subjects):
            test_ids.extend(cls_subjects[te].tolist())
            break

    train_pool = features[~features["subject_id"].isin(test_ids)]
    train_subjects = train_pool["subject_id"].to_numpy()

    gkf = GroupKFold(n_splits=n_folds)
    cv_folds: list[dict] = []
    for tr_idx, va_idx in gkf.split(train_pool, groups=train_subjects):
        cv_folds.append(
            {
                "train": sorted(set(train_pool.iloc[tr_idx]["subject_id"])),
                "val": sorted(set(train_pool.iloc[va_idx]["subject_id"])),
            }
        )

    splits = Splits(test_subjects=sorted(set(test_ids)), cv_folds=cv_folds)
    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits: Splits) -> None:
    """Raise if any subject appears in multiple partitions."""
    test = set(splits.test_subjects)
    for i, fold in enumerate(splits.cv_folds):
        tr = set(fold["train"])
        va = set(fold["val"])
        assert tr.isdisjoint(va), f"fold {i}: train/val overlap"
        assert tr.isdisjoint(test), f"fold {i}: train/test overlap"
        assert va.isdisjoint(test), f"fold {i}: val/test overlap"


def select_rows(features: pd.DataFrame, subject_ids: list[str]) -> pd.DataFrame:
    features = add_subject_id(features)
    return features[features["subject_id"].isin(set(subject_ids))]
