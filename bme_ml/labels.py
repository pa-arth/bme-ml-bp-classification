"""Derive ground-truth BP labels from the ABP channel.

Per segment we extract median SBP (peak amplitudes) and median DBP (trough
amplitudes), then map to the binary clinical label:

    Normal   = SBP < 130 and DBP < 80
    Abnormal = otherwise

We store the raw SBP/DBP medians in the feature table so callers can
re-derive other thresholds (e.g. 4-class JNC) without re-running extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from . import SAMPLE_RATE_HZ

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
        n_beats=len(peaks),
    )
