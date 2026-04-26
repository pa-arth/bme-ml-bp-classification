"""Derive ground-truth BP labels from the ABP channel.

Per segment we extract median SBP (peak amplitudes) and median DBP (trough
amplitudes), then map to two clinical label schemes:

    binary:    Normal = SBP < 130 ∧ DBP < 80; Abnormal otherwise.
    3-class:   AHA categories — Normal / Elevated / Hypertensive.

Both come for free off the same `(sbp, dbp)` pair, and the raw medians are
stored in the feature table, so adding new thresholds (4-class JNC, etc.)
later is a load-time derivation — no need to re-run extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from . import SAMPLE_RATE_HZ

# AHA 3-class scheme.
MULTICLASS_NORMAL = 0
MULTICLASS_ELEVATED = 1
MULTICLASS_HYPERTENSIVE = 2
MULTICLASS_NAMES = ["Normal", "Elevated", "Hypertensive"]


def multiclass_label(sbp: float, dbp: float) -> int:
    """AHA 3-class hypertension category from SBP and DBP medians (mmHg).

    - Normal:        SBP < 120 ∧ DBP < 80
    - Elevated:      120 ≤ SBP ≤ 129 ∧ DBP < 80
    - Hypertensive:  SBP ≥ 130 ∨ DBP ≥ 80
    """
    if sbp >= 130.0 or dbp >= 80.0:
        return MULTICLASS_HYPERTENSIVE
    if sbp >= 120.0:
        return MULTICLASS_ELEVATED
    return MULTICLASS_NORMAL


def add_multiclass_column(features: pd.DataFrame, col: str = "label_3class") -> pd.DataFrame:
    """Return `features` with a `label_3class` column derived from `sbp`/`dbp`.

    Idempotent — if the column already exists, returns the frame unchanged.
    """
    if col in features.columns:
        return features
    labels = np.fromiter(
        (multiclass_label(s, d) for s, d in zip(features["sbp"], features["dbp"])),
        dtype=np.int8,
        count=len(features),
    )
    return features.assign(**{col: labels})

# Heart rate floor of ~30 bpm => 2 s between beats. We use a slightly tighter
# minimum (300 ms) to allow brief tachycardia while still rejecting noise
# peaks from the same cardiac cycle.
MIN_BEAT_DISTANCE_SAMPLES = int(0.3 * SAMPLE_RATE_HZ)
MIN_PEAK_PROMINENCE_MMHG = 10.0  # SBP-DBP < 10 mmHg is unphysiological for an
# adult ABP trace; using prominence instead of absolute height handles
# baseline drift.


@dataclass(frozen=True)
class BPLabel:
    sbp: float       # median systolic, mmHg
    dbp: float       # median diastolic, mmHg
    binary: int      # 0 = Normal, 1 = Abnormal
    multiclass: int  # AHA: 0 = Normal, 1 = Elevated, 2 = Hypertensive
    n_beats: int     # peaks detected in the segment (sanity feature)


def label_segment(abp: np.ndarray) -> BPLabel | None:
    """Return a `BPLabel` for a single segment's ABP trace, or None if no
    usable peaks were detected.
    """
    peaks, _ = find_peaks(
        abp,
        distance=MIN_BEAT_DISTANCE_SAMPLES,
        prominence=MIN_PEAK_PROMINENCE_MMHG,
    )
    troughs, _ = find_peaks(
        -abp,
        distance=MIN_BEAT_DISTANCE_SAMPLES,
        prominence=MIN_PEAK_PROMINENCE_MMHG,
    )
    if len(peaks) < 2 or len(troughs) < 2:
        return None

    sbp = float(np.median(abp[peaks]))
    dbp = float(np.median(abp[troughs]))
    return BPLabel(
        sbp=sbp,
        dbp=dbp,
        binary=int(not (sbp < 130.0 and dbp < 80.0)),
        multiclass=multiclass_label(sbp, dbp),
        n_beats=len(peaks),
    )
