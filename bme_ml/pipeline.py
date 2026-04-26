"""Per-record processing for parallel feature extraction.

`process_record` is the unit of work joblib hands to a worker. It returns
both the engineered-feature rows and the raw 2-channel (PPG, ECG) segments,
so the same parallel sweep populates both `features.parquet` (for RF/SVM)
and `signals.h5` (for the CNN).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .features import extract_features
from .labels import label_segment
from .preprocessing import is_valid_segment, preprocess_record, window_record


def process_record(
    part: int,
    record_idx: int,
    rec: np.ndarray,
    max_segments: int | None = None,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Run preprocessing → windowing → labels → features for one record.

    `max_segments` caps the number of 8-second segments contributed by this
    record. When set, we pick that many *evenly spaced* segment indices
    across the recording before any feature extraction — this decorrelates
    the training set temporally (a 5–10 minute record naturally produces
    35–70 nearly-identical 8s windows; keeping 15 covers the same time
    span without inflating effective sample size).

    Returns:
        rows: list of feature/label dicts, one per kept segment.
        signals: list of `(2, N)` arrays (PPG, ECG only — ABP is for labels).
                 Same length and order as `rows`, so global row index in the
                 parquet equals row index in the HDF5 signal cache.
    """
    rec_p = preprocess_record(rec)
    segs = list(window_record(rec_p))  # cheap — these are views, not copies

    if max_segments is not None and len(segs) > max_segments:
        keep_idx = np.linspace(0, len(segs) - 1, max_segments).round().astype(int)
        keep_idx = sorted({int(i) for i in keep_idx})
    else:
        keep_idx = list(range(len(segs)))

    rows: list[dict[str, Any]] = []
    signals: list[np.ndarray] = []
    for seg_idx in keep_idx:
        seg = segs[seg_idx]
        if not is_valid_segment(seg):
            continue
        lbl = label_segment(seg[1])
        if lbl is None:
            continue
        feats = extract_features(seg)
        rows.append(
            {
                "part": int(part),
                "record": int(record_idx),
                "segment": int(seg_idx),
                "sbp": float(lbl.sbp),
                "dbp": float(lbl.dbp),
                "label_binary": int(lbl.binary),
                "label_3class": int(lbl.multiclass),
                "n_beats": int(lbl.n_beats),
                **feats,
            }
        )
        # Stack PPG (row 0) + ECG (row 2) → (2, N) for the CNN.
        signals.append(np.stack([seg[0], seg[2]]).astype(np.float32, copy=False))
    return rows, signals
