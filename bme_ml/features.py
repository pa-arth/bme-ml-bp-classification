"""Feature extraction from ECG + PPG segments.

Per 8-second segment we extract two overlapping feature sets — the **Kachuee
2015** set (10 features used by the SVM regression baseline) and an
**extended** set adding HRV, pulse widths, and QRS duration:

  Kachuee (10):
    ptt_ms          PTTp — Pulse Transit Time, R-peak → PPG systolic peak
    pttf_ms         PTTf — R-peak → PPG foot (next trough after R)
    pttd_ms         PTTd — R-peak → PPG max-slope point on the upstroke
    hr_bpm          Mean heart rate from R-R intervals
    ai_kachuee      Augmentation Index — diastolic peak / systolic peak
                    (both relative to the per-cycle foot baseline)
    lasi_ms         Large Artery Stiffness Index — time(sys peak → dia peak)
    s1_area         Area under PPG: foot → systolic peak
    s2_area         Area under PPG: systolic peak → dicrotic notch
    s3_area         Area under PPG: dicrotic notch → diastolic peak
    s4_area         Area under PPG: diastolic peak → next foot
    (Areas are integrated above the per-cycle foot baseline, so they are
    independent of absolute PPG offset.)

  Extended (in addition to the 10 above):
    rr_sd_ms        Standard deviation of R-R intervals
    hrv_rmssd_ms    Time-domain HRV (RMSSD)
    hrv_sdnn_ms     Time-domain HRV (SDNN)
    ppg_pw50_ms     PPG pulse width at 50% peak amplitude
    ppg_pw25_ms     PPG pulse width at 25% peak amplitude
    ppg_rise_ms     Foot-to-peak rise time
    ppg_decay_ms    Peak-to-foot decay time
    ecg_qrs_ms      Median QRS duration
    ppg_amp_ratio   Notch / systolic amplitude ratio (legacy, *not* AI —
                    uses the dicrotic notch, kept for back-compat)
    ppg_aug_index   Same as ppg_amp_ratio (legacy, kept for back-compat)

`extract_features` returns NaN for any feature it can't compute (e.g. no
R-peaks). Downstream training drops rows containing NaN for the chosen
feature subset — see `KACHUEE_FEATURES` and `OUR_FEATURES`.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from . import SAMPLE_RATE_HZ

# The Kachuee 2015 feature set (10 features) — used by the SVM regression
# baseline and as one arm of the apples-to-apples comparison in the writeup.
KACHUEE_FEATURES = [
    "ptt_ms",      # PTTp
    "pttf_ms",
    "pttd_ms",
    "hr_bpm",
    "ai_kachuee",
    "lasi_ms",
    "s1_area",
    "s2_area",
    "s3_area",
    "s4_area",
]

# Extended feature set — Kachuee + HRV + pulse widths + QRS duration. This
# is what the RF / XGBoost / CNN comparison-table models train on by
# default; `notebooks/03_train_tabular.ipynb` accepts a `feature_set` arg
# to switch.
OUR_FEATURES = KACHUEE_FEATURES + [
    "rr_sd_ms",
    "hrv_rmssd_ms",
    "hrv_sdnn_ms",
    "ppg_pw50_ms",
    "ppg_pw25_ms",
    "ppg_rise_ms",
    "ppg_decay_ms",
    "ecg_qrs_ms",
    "ppg_amp_ratio",
    "ppg_aug_index",
]

# Back-compat: callers that just want "all features" import this name.
FEATURE_NAMES = OUR_FEATURES


def _empty_features() -> dict[str, float]:
    return {name: np.nan for name in FEATURE_NAMES}


def _samples_to_ms(n: float) -> float:
    return float(n) * 1000.0 / SAMPLE_RATE_HZ


def _find_pulse_landmarks(
    pulse: np.ndarray, peak_in_pulse: int
) -> tuple[int | None, int | None]:
    """Locate the dicrotic notch and diastolic peak within a single PPG
    cycle (foot to next foot). Both indices are relative to the start of
    `pulse`. Returns `(None, None)` if either can't be reliably detected.

    Method:
      1. Smooth the pulse aggressively (Savitzky-Golay, ~11-sample window)
         so small noise spikes don't register as peaks.
      2. Diastolic peak = first local maximum of the smoothed pulse after
         the systolic peak. On low-amplitude / elderly cycles the
         diastolic peak is a shoulder, not a true peak — those cycles
         return None and are skipped, which is the right behavior (we'd
         rather have NaN than a wrong landmark).
      3. Dicrotic notch = local minimum of the pulse between the systolic
         peak and the diastolic peak.
    """
    n = len(pulse)
    if peak_in_pulse >= n - 5:
        return None, None

    # Window must be odd and ≤ n. 11 samples ≈ 88 ms at 125 Hz — wide
    # enough to suppress noise without smearing the notch / dia peak.
    win = min(11, n)
    if win % 2 == 0:
        win -= 1
    if win < 5:
        return None, None
    try:
        smoothed = savgol_filter(pulse, win, 3)
    except ValueError:
        return None, None

    descending = smoothed[peak_in_pulse:]
    if len(descending) < 4:
        return None, None
    dia_peaks, _ = find_peaks(descending)
    if len(dia_peaks) == 0:
        return None, None
    dia_idx = peak_in_pulse + int(dia_peaks[0])

    # Notch = local min of the (unsmoothed) pulse between sys and dia.
    between = pulse[peak_in_pulse + 1 : dia_idx]
    if len(between) < 1:
        return None, None
    notch_idx = peak_in_pulse + 1 + int(np.argmin(between))

    if not (peak_in_pulse < notch_idx < dia_idx < n):
        return None, None
    return notch_idx, dia_idx


def extract_features(seg: np.ndarray) -> Mapping[str, float]:
    """Return a {name: float} dict of features for a `(3, N)` segment.

    Channels: row 0 = PPG (filtered), row 1 = ABP (unused here, label-only),
    row 2 = ECG (filtered).
    """
    import neurokit2 as nk

    ppg, _abp, ecg = seg[0], seg[1], seg[2]
    out = _empty_features()

    # --- ECG: R-peaks, HR, HRV, QRS --------------------------------------
    try:
        _, ecg_info = nk.ecg_peaks(ecg, sampling_rate=SAMPLE_RATE_HZ, correct_artifacts=True)
        r_peaks = np.asarray(ecg_info["ECG_R_Peaks"], dtype=int)
    except Exception:
        r_peaks = np.array([], dtype=int)

    if len(r_peaks) >= 2:
        rr = np.diff(r_peaks) / SAMPLE_RATE_HZ  # seconds
        out["hr_bpm"] = float(60.0 / np.mean(rr))
        out["rr_sd_ms"] = float(np.std(rr) * 1000.0)
        out["hrv_sdnn_ms"] = float(np.std(rr) * 1000.0)
        if len(rr) >= 2:
            rr_diff = np.diff(rr) * 1000.0
            out["hrv_rmssd_ms"] = float(np.sqrt(np.mean(rr_diff**2)))

    if len(r_peaks) >= 1:
        try:
            delineate = nk.ecg_delineate(
                ecg, rpeaks=r_peaks, sampling_rate=SAMPLE_RATE_HZ, method="dwt"
            )[1]
            q_onsets = np.asarray(delineate.get("ECG_Q_Peaks", []), dtype=float)
            s_offsets = np.asarray(delineate.get("ECG_S_Peaks", []), dtype=float)
            valid = ~(np.isnan(q_onsets) | np.isnan(s_offsets))
            if valid.any():
                qrs_lengths = (s_offsets[valid] - q_onsets[valid])
                qrs_lengths = qrs_lengths[qrs_lengths > 0]
                if len(qrs_lengths):
                    out["ecg_qrs_ms"] = _samples_to_ms(np.median(qrs_lengths))
        except Exception:
            pass

    # --- PPG: peaks, troughs, width, rise/decay, AI, areas --------------
    try:
        ppg_signals, ppg_info = nk.ppg_process(ppg, sampling_rate=SAMPLE_RATE_HZ)
        ppg_peaks = np.asarray(ppg_info["PPG_Peaks"], dtype=int)
    except Exception:
        ppg_peaks = np.array([], dtype=int)
        ppg_signals = None

    troughs = np.array([], dtype=int)
    if len(ppg_peaks) >= 2 and ppg_signals is not None:
        # Troughs = local minima between consecutive peaks.
        troughs_list = []
        for a, b in zip(ppg_peaks[:-1], ppg_peaks[1:]):
            if b - a > 2:
                troughs_list.append(a + int(np.argmin(ppg[a:b])))
        troughs = np.asarray(troughs_list, dtype=int)

        if len(troughs):
            # Rise time: trough_i → peak_{i+1}; decay: peak_i → trough_i.
            rises = []
            decays = []
            for t in troughs:
                later_peaks = ppg_peaks[ppg_peaks > t]
                earlier_peaks = ppg_peaks[ppg_peaks < t]
                if len(later_peaks):
                    rises.append(later_peaks[0] - t)
                if len(earlier_peaks):
                    decays.append(t - earlier_peaks[-1])
            if rises:
                out["ppg_rise_ms"] = _samples_to_ms(np.median(rises))
            if decays:
                out["ppg_decay_ms"] = _samples_to_ms(np.median(decays))

            # Per-cycle features: pulse widths, legacy notch ratios, AI/LASI/S1-S4.
            pw50, pw25 = [], []
            amp_ratios, aug_indices = [], []
            ai_vals, lasi_vals = [], []
            s1_vals, s2_vals, s3_vals, s4_vals = [], [], [], []

            dt = 1.0 / SAMPLE_RATE_HZ

            for peak in ppg_peaks:
                before = troughs[troughs < peak]
                after = troughs[troughs > peak]
                if len(before) == 0 or len(after) == 0:
                    continue
                t0, t1 = before[-1], after[0]
                pulse = ppg[t0 : t1 + 1].astype(np.float64, copy=False)
                if len(pulse) < 6:
                    continue
                base = float(pulse.min())
                top = float(pulse.max())
                sys_amp = top - base
                if sys_amp <= 0:
                    continue
                peak_in_pulse = int(peak - t0)

                # Pulse widths at 50% / 25% above the foot baseline.
                lvl50 = base + 0.5 * sys_amp
                lvl25 = base + 0.25 * sys_amp
                above50 = np.where(pulse >= lvl50)[0]
                above25 = np.where(pulse >= lvl25)[0]
                if len(above50) >= 2:
                    pw50.append(above50[-1] - above50[0])
                if len(above25) >= 2:
                    pw25.append(above25[-1] - above25[0])

                # Legacy notch / systolic ratio (kept for back-compat).
                tail = pulse[peak_in_pulse:]
                if len(tail) >= 4:
                    d2_tail = np.diff(np.diff(tail))
                    if len(d2_tail):
                        legacy_notch_offset = int(np.argmax(d2_tail))
                        notch_amp_legacy = tail[legacy_notch_offset + 1] - base
                        amp_ratios.append(notch_amp_legacy / sys_amp)
                        aug_indices.append(notch_amp_legacy / sys_amp)

                # Kachuee landmarks: dicrotic notch + diastolic peak.
                # NOTE: many MIMIC-derived 125 Hz cycles lack a clearly
                # visible notch (subject-dependent; low-amplitude or
                # heavily filtered notches don't survive). The detector
                # returns None for those cycles, so AI/LASI/S1-S4 are
                # NaN for any segment whose cycles lack landmarks. Empirically
                # ~16% of segments yield non-NaN landmark features.
                notch_idx, dia_idx = _find_pulse_landmarks(pulse, peak_in_pulse)
                if dia_idx is not None and notch_idx is not None:
                    dia_amp = pulse[dia_idx] - base
                    # AI: amplitude of secondary peak relative to systolic.
                    ai_vals.append(dia_amp / sys_amp)
                    # LASI: time from systolic peak to diastolic peak.
                    lasi_vals.append(dia_idx - peak_in_pulse)
                    # S1-S4: trapezoidal areas above foot baseline.
                    pulse_above = pulse - base
                    s1_vals.append(float(np.trapezoid(pulse_above[: peak_in_pulse + 1], dx=dt)))
                    s2_vals.append(float(np.trapezoid(pulse_above[peak_in_pulse : notch_idx + 1], dx=dt)))
                    s3_vals.append(float(np.trapezoid(pulse_above[notch_idx : dia_idx + 1], dx=dt)))
                    s4_vals.append(float(np.trapezoid(pulse_above[dia_idx:], dx=dt)))

            if pw50:
                out["ppg_pw50_ms"] = _samples_to_ms(np.median(pw50))
            if pw25:
                out["ppg_pw25_ms"] = _samples_to_ms(np.median(pw25))
            if amp_ratios:
                out["ppg_amp_ratio"] = float(np.median(amp_ratios))
            if aug_indices:
                out["ppg_aug_index"] = float(np.median(aug_indices))
            if ai_vals:
                out["ai_kachuee"] = float(np.median(ai_vals))
            if lasi_vals:
                out["lasi_ms"] = _samples_to_ms(np.median(lasi_vals))
            if s1_vals:
                out["s1_area"] = float(np.median(s1_vals))
            if s2_vals:
                out["s2_area"] = float(np.median(s2_vals))
            if s3_vals:
                out["s3_area"] = float(np.median(s3_vals))
            if s4_vals:
                out["s4_area"] = float(np.median(s4_vals))

    # --- PTT family: PTTp / PTTf / PTTd, all on the same matched cycle ---
    # For each R-peak, match the next PPG systolic peak within 0.5 s
    # (= PTTp). The matched cycle's foot is the trough immediately
    # preceding that peak; PTTd is the max-slope point on the upstroke
    # from foot → peak. This guarantees PTTf < PTTd < PTTp in time and
    # all three landmarks correspond to the same cardiac cycle.
    if len(r_peaks) and len(ppg_peaks):
        max_lag = int(0.5 * SAMPLE_RATE_HZ)
        # Allow slightly negative deltas: in some recordings PTT is short
        # enough that the foot/upstroke can begin before R is detected.
        min_lag_foot = -int(0.2 * SAMPLE_RATE_HZ)
        min_lag_d = -int(0.05 * SAMPLE_RATE_HZ)
        dppg = np.gradient(ppg)

        ptt_samples: list[int] = []
        pttf_samples: list[int] = []
        pttd_samples: list[int] = []
        for r_raw in r_peaks:
            r = int(r_raw)
            mask = (ppg_peaks > r) & (ppg_peaks < r + max_lag)
            candidates = ppg_peaks[mask]
            if len(candidates) == 0:
                continue
            peak_idx = int(candidates[0])
            ptt_samples.append(peak_idx - r)

            # Foot of this cycle = trough immediately before peak_idx.
            prior_troughs = troughs[troughs < peak_idx] if len(troughs) else np.array([], dtype=int)
            if len(prior_troughs) == 0:
                continue
            foot_idx = int(prior_troughs[-1])
            delta_f = foot_idx - r
            if min_lag_foot < delta_f < max_lag:
                pttf_samples.append(delta_f)

            # Max-slope point on the upstroke from foot to peak.
            if peak_idx - foot_idx >= 3:
                upstroke_offset = int(np.argmax(dppg[foot_idx : peak_idx + 1]))
                max_slope_idx = foot_idx + upstroke_offset
                delta_d = max_slope_idx - r
                if min_lag_d < delta_d < max_lag:
                    pttd_samples.append(delta_d)

        if ptt_samples:
            out["ptt_ms"] = _samples_to_ms(np.median(ptt_samples))
        if pttf_samples:
            out["pttf_ms"] = _samples_to_ms(np.median(pttf_samples))
        if pttd_samples:
            out["pttd_ms"] = _samples_to_ms(np.median(pttd_samples))

    return out
