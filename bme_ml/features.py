"""Feature extraction from ECG + PPG segments.

Per 8-second segment we extract:
  - ptt_ms                 Pulse Transit Time (R-peak → next PPG peak)
  - hr_bpm                 Mean heart rate from R-R intervals
  - rr_sd_ms               Standard deviation of R-R intervals
  - hrv_rmssd_ms           Time-domain HRV (RMSSD)
  - hrv_sdnn_ms            Time-domain HRV (SDNN)
  - ppg_pw50_ms            PPG pulse width at 50% peak amplitude
  - ppg_pw25_ms            PPG pulse width at 25% peak amplitude
  - ppg_amp_ratio          Systolic/diastolic amplitude ratio (notch-based)
  - ppg_rise_ms            Foot-to-peak rise time
  - ppg_decay_ms           Peak-to-foot decay time
  - ecg_qrs_ms             Median QRS duration
  - ppg_aug_index          Augmentation index proxy (inflection / systolic)

`extract_features` returns NaN for any feature it can't compute (e.g.
no R-peaks). Downstream training drops rows containing NaN for the chosen
feature subset.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from . import SAMPLE_RATE_HZ

FEATURE_NAMES = [
    "ptt_ms",
    "hr_bpm",
    "rr_sd_ms",
    "hrv_rmssd_ms",
    "hrv_sdnn_ms",
    "ppg_pw50_ms",
    "ppg_pw25_ms",
    "ppg_amp_ratio",
    "ppg_rise_ms",
    "ppg_decay_ms",
    "ecg_qrs_ms",
    "ppg_aug_index",
]


def _empty_features() -> dict[str, float]:
    return {name: np.nan for name in FEATURE_NAMES}


def _samples_to_ms(n: float) -> float:
    return float(n) * 1000.0 / SAMPLE_RATE_HZ


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

    # --- PPG: peaks, troughs, width, rise/decay, AI --------------------
    try:
        ppg_signals, ppg_info = nk.ppg_process(ppg, sampling_rate=SAMPLE_RATE_HZ)
        ppg_peaks = np.asarray(ppg_info["PPG_Peaks"], dtype=int)
    except Exception:
        ppg_peaks = np.array([], dtype=int)
        ppg_signals = None

    if len(ppg_peaks) >= 2 and ppg_signals is not None:
        # Troughs = local minima between consecutive peaks.
        troughs = []
        for a, b in zip(ppg_peaks[:-1], ppg_peaks[1:]):
            if b - a > 2:
                troughs.append(a + int(np.argmin(ppg[a:b])))
        troughs = np.asarray(troughs, dtype=int)

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

            # Pulse widths at 50% / 25% peak amplitude (per-pulse, then median).
            pw50, pw25 = [], []
            amp_ratios = []
            aug_indices = []
            for i, peak in enumerate(ppg_peaks):
                # Find the trough immediately before and after this peak.
                before = troughs[troughs < peak]
                after = troughs[troughs > peak]
                if len(before) == 0 or len(after) == 0:
                    continue
                t0, t1 = before[-1], after[0]
                pulse = ppg[t0 : t1 + 1]
                if len(pulse) < 4:
                    continue
                base = pulse.min()
                top = pulse.max()
                if top - base <= 0:
                    continue
                lvl50 = base + 0.5 * (top - base)
                lvl25 = base + 0.25 * (top - base)
                above50 = np.where(pulse >= lvl50)[0]
                above25 = np.where(pulse >= lvl25)[0]
                if len(above50) >= 2:
                    pw50.append(above50[-1] - above50[0])
                if len(above25) >= 2:
                    pw25.append(above25[-1] - above25[0])
                # Amplitude ratio + AI proxy: look for second-derivative
                # inflection between systolic peak and end of pulse.
                tail = pulse[peak - t0 :]
                if len(tail) >= 4:
                    d2 = np.diff(np.diff(tail))
                    if len(d2):
                        notch_offset = int(np.argmax(d2))
                        notch_amp = tail[notch_offset + 1] - base
                        sys_amp = top - base
                        if sys_amp > 0:
                            amp_ratios.append(notch_amp / sys_amp)
                            aug_indices.append(notch_amp / sys_amp)
            if pw50:
                out["ppg_pw50_ms"] = _samples_to_ms(np.median(pw50))
            if pw25:
                out["ppg_pw25_ms"] = _samples_to_ms(np.median(pw25))
            if amp_ratios:
                out["ppg_amp_ratio"] = float(np.median(amp_ratios))
            if aug_indices:
                out["ppg_aug_index"] = float(np.median(aug_indices))

    # --- PTT: R-peak → next PPG peak --------------------------------------
    if len(r_peaks) and len(ppg_peaks):
        ptt_samples = []
        j = 0
        for r in r_peaks:
            while j < len(ppg_peaks) and ppg_peaks[j] <= r:
                j += 1
            if j >= len(ppg_peaks):
                break
            delta = ppg_peaks[j] - r
            # PTT > 500 ms is suspicious — likely a missed PPG peak — skip.
            if 0 < delta < 0.5 * SAMPLE_RATE_HZ:
                ptt_samples.append(delta)
        if ptt_samples:
            out["ptt_ms"] = _samples_to_ms(np.median(ptt_samples))

    return out
