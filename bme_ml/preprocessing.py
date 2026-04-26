"""Filtering, windowing, and segment-level quality control.

We keep ECG/PPG bandpasses tight (0.5–40 Hz / 0.5–8 Hz) and only median-
center the ABP — its absolute amplitude is the BP signal we're trying to
extract from peaks/troughs, so don't filter it.

`is_valid_segment` is a single predicate that gates everything downstream.
Tweak its thresholds here when you discover edge cases in the data; don't
scatter QC logic across feature extraction.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from scipy.signal import butter, filtfilt

from . import SAMPLE_RATE_HZ, WINDOW_SAMPLES

ECG_BAND_HZ = (0.5, 40.0)
PPG_BAND_HZ = (0.5, 8.0)


def _butter_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    return butter(order, [low_hz / nyq, high_hz / nyq], btype="band")


_ECG_BA = _butter_bandpass(*ECG_BAND_HZ, fs=SAMPLE_RATE_HZ)
_PPG_BA = _butter_bandpass(*PPG_BAND_HZ, fs=SAMPLE_RATE_HZ)


def filter_ecg(ecg: np.ndarray) -> np.ndarray:
    return filtfilt(_ECG_BA[0], _ECG_BA[1], ecg).astype(np.float32, copy=False)


def filter_ppg(ppg: np.ndarray) -> np.ndarray:
    return filtfilt(_PPG_BA[0], _PPG_BA[1], ppg).astype(np.float32, copy=False)


def center_abp(abp: np.ndarray) -> np.ndarray:
    """Median-subtract for plotting only. Feature/label code uses raw ABP."""
    return (abp - np.median(abp)).astype(np.float32, copy=False)


def preprocess_record(record: np.ndarray) -> np.ndarray:
    """Apply per-channel filtering to a `(3, N)` record.

    Returns a new `(3, N)` array — `[filtered PPG, raw ABP, filtered ECG]`.
    The ABP channel stays unfiltered so SBP/DBP peak amplitudes are preserved.
    """
    ppg, abp, ecg = record[0], record[1], record[2]
    return np.stack([filter_ppg(ppg), abp.astype(np.float32, copy=False), filter_ecg(ecg)])


def window_record(
    record: np.ndarray, window_samples: int = WINDOW_SAMPLES
) -> Iterator[np.ndarray]:
    """Yield non-overlapping `(3, window_samples)` segments from a record."""
    n = record.shape[1]
    for start in range(0, n - window_samples + 1, window_samples):
        yield record[:, start : start + window_samples]


# --- Quality control --------------------------------------------------------

ABP_MIN_MMHG = 30.0
ABP_MAX_MMHG = 250.0
ECG_FLAT_MAX_SAMPLES = SAMPLE_RATE_HZ  # 1 second of flat-line is too long
PPG_VARIANCE_MIN = 1e-3  # PPG is reported in arbitrary units; this catches
# essentially-flat traces while keeping legitimate low-amplitude pulses.


def is_valid_segment(seg: np.ndarray) -> bool:
    """Return True if a `(3, window_samples)` segment is usable.

    Rejection reasons (in order, cheapest first):
      1. NaN/inf anywhere.
      2. ABP outside physiological bounds (clipped or sensor failure).
      3. ECG flat-line stretch > 1 s (lead detached).
      4. PPG variance below noise floor.
    """
    if not np.all(np.isfinite(seg)):
        return False

    ppg, abp, ecg = seg[0], seg[1], seg[2]

    if abp.min() < ABP_MIN_MMHG or abp.max() > ABP_MAX_MMHG:
        return False

    # Detect runs of identical adjacent ECG samples — proxy for flat-line.
    diffs = np.diff(ecg)
    flat_mask = diffs == 0
    if flat_mask.any():
        # Longest run of consecutive Trues:
        # cheap calc — flip on transitions, take max run length.
        run = 0
        max_run = 0
        for v in flat_mask:
            run = run + 1 if v else 0
            if run > max_run:
                max_run = run
        if max_run > ECG_FLAT_MAX_SAMPLES:
            return False

    if float(np.var(ppg)) < PPG_VARIANCE_MIN:
        return False

    return True
