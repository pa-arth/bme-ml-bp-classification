# Cuff-less BP classification

Hypertension classification from ECG + PPG on the [UCI Cuff-Less Blood Pressure Estimation dataset](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation) (~12k records derived from MIMIC-II, 125 Hz). Trains five models on the same subject-level splits and produces a single auto-filled writeup paragraph.

The methodological question: **does direct multi-class classification beat the canonical regression-then-threshold approach (Kachuee et al. 2015) on the same data?**

## Models compared

| Model | Type | Trained on | Notebook |
|---|---|---|---|
| Random Forest | Direct 3-class classifier | 20 engineered features | `03_train_tabular.ipynb` |
| XGBoost | Direct 3-class classifier | 20 engineered features | `03_train_tabular.ipynb` |
| 1D CNN | Direct 3-class classifier | Raw `(2, 1000)` PPG+ECG | `05_train_cnn.ipynb` |
| SVM (regression + threshold) | Indirect: SVR for SBP/DBP, then AHA threshold | 10 Kachuee features | `03b_train_svm.ipynb` |
| PTT-only LR | Single-feature reference baseline | Just `ptt_ms` | `03_train_tabular.ipynb` |

The label is **AHA 3-class hypertension** (`Normal` if SBP<120 ∧ DBP<80; `Elevated` if 120≤SBP≤129 ∧ DBP<80; `Hypertensive` if SBP≥130 ∨ DBP≥80). Even though the model is 3-class, the writeup reports a binary "AUROC for detecting hypertensive cases" by collapsing class-2 probability vs the rest — see `bme_ml/evaluation.py:evaluate_multiclass`.

## Pipeline

Run the notebooks in this order from `notebooks/`:

1. **`00_setup.ipynb`** — installs deps, downloads the UCI zip (~3.4 GB), verifies CUDA.
2. **`01_explore_signals.ipynb`** — eyeball one record (raw vs filtered ECG/PPG/ABP, label distribution).
3. **`02_build_features.ipynb`** — joblib-parallel sweep over all 4 parts → `features.parquet` (28 cols: 20 features + sbp/dbp + labels + metadata) and `signals.h5` (raw `(N, 2, 1000)` segments for the CNN). Default `MAX_RECORDS = None` runs the full ~12k records (~30–60 min CPU-bound).
4. **`03_train_tabular.ipynb`** — RF + XGBoost grid-searched in subject-level 5-fold GroupKFold, plus the PTT-only LR baseline. Writes `splits.json` (reused by 03b/05), three metrics JSONs, two importance CSVs.
5. **`03b_train_svm.ipynb`** — SVR(SBP) + SVR(DBP) on `KACHUEE_FEATURES`, median-imputed inside the pipeline so the 16% landmark-detection sparsity doesn't drop training data to zero. Outputs are thresholded into 3-class via `multiclass_label`. Writes `svm_metrics.json`, `svm_regression_metrics.json`, `svm_predictions.csv`.
6. **`05_train_cnn.ipynb`** — 1D ConvNet on z-scored PPG+ECG segments. Auto-uses CUDA on a 4070; falls back to MPS / CPU. Saves model weights, multi-class metrics JSON, and a `(2, 1000)` saliency map (gradient × input averaged over the test loader).
7. **`04_results.ipynb`** — comparison table, 3×3 confusion matrix for the best model, top-feature plots for RF + XGBoost, saliency overlay for the CNN, and an **auto-filled writeup paragraph** that interpolates the loaded metrics.

## Quick start

Repo expects Python 3.13 and (for the CNN) a CUDA-capable GPU.

```bash
git clone https://github.com/pa-arth/bme-ml-bp-classification.git
cd bme-ml-bp-classification
python -m venv .venv && .venv/Scripts/activate     # Windows; use bin/activate on macOS/Linux
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu124 torch
jupyter lab notebooks/
```

Then run notebooks 00 → 01 → 02 → 03 → 03b → 05 → 04 (in that order). A full sweep takes roughly 1.5–2 hours of wall clock on a modern laptop + RTX 4070, dominated by feature extraction in nb 02 and CNN training in nb 05.

For a smoke test (~5 min), set `MAX_RECORDS = 50` in nb 02's first code cell. The numbers won't be meaningful but the pipeline runs end-to-end.

## Methodology highlights

- **Subject-level splits.** `bme_ml/splits.py` enforces no overlap between train, validation, and test subjects. Stratification is per-subject majority class. All five models use the same `splits.json` so test-set comparisons are apples-to-apples.
- **AHA 3-class labels are derived at load time** from the `sbp`/`dbp` medians in `features.parquet` (see `bme_ml/labels.py:add_multiclass_column`). Re-defining thresholds (e.g. JNC 4-class) is a one-line change — no re-extraction.
- **Saliency for the CNN.** `bme_ml/saliency.py:compute_saliency` returns the mean `|d logit / d input|` across the test loader, broken down per channel (PPG, ECG). The result is a `(2, 1000)` heatmap that's plotted overlaid on the mean test-set waveform in nb 04, so saliency peaks line up with actual signal landmarks.
- **Soft probabilities for the SVM AUROC.** Regression-then-threshold doesn't give native class probabilities, so 03b constructs them as logistic functions of the AHA-threshold distances (`sigmoid((sbp_pred - 130)/10)` etc.). Documented in 03b.
- **Median imputation for the SVM.** ~84% of segments lack the diastolic-peak landmarks (AI/LASI/S1–S4) at 125 Hz — the SVM pipeline imputes the missing values inside the sklearn `Pipeline` rather than dropna'ing, retaining ~80% of training data instead of ~16%. See `SVM_BASELINE.md` for the rationale.

## Two real data quirks worth knowing

1. **Diastolic-peak detection rate is ~16%.** Many MIMIC-derived 125 Hz cycles simply don't have a visible secondary peak — the dicrotic notch is often shallower than the noise floor at this sampling rate. AI/LASI/S1–S4 are NaN for those cycles. Documented in `bme_ml/features.py:_find_pulse_landmarks`.
2. **Fixed ECG/PPG channel offset.** The PPG appears to be ~80 ms ahead of physiological alignment in this dataset, so `pttf_ms` is consistently negative (~-100 ms) and `pttd_ms` ≈ 0 ms. The PTTp > PTTd > PTTf inequality still holds in 100% of rows. Don't try to "correct" individual PTT features — the relative scaling is what carries the signal, and the SVM (and tree models) learn around the absolute offset.

## Repository layout

```
bme_ml/                     # importable library
  __init__.py               # global constants (SAMPLE_RATE_HZ, WINDOW_SAMPLES)
  paths.py                  # project paths + UCI dataset download
  data_loader.py            # v7.3 .mat reader (HDF5-backed, handles UCI's (N,1) layout)
  preprocessing.py          # ECG/PPG bandpass, windowing, segment QC
  labels.py                 # binary + 3-class AHA labels from ABP
  features.py               # 20 engineered features incl. KACHUEE_FEATURES subset
  pipeline.py               # process_record() — joblib worker
  splits.py                 # subject-level GroupKFold + GroupShuffleSplit
  models.py                 # RF, XGBoost, PTT-LR builders + grid searches
  cnn.py                    # 1D ConvNet, dataset, training loop
  saliency.py               # gradient × input attribution for the CNN
  evaluation.py             # BinaryMetrics + MultiClassMetrics + comparison
notebooks/                  # numbered run order (00 → 04)
SVM_BASELINE.md             # partner brief for the SVM-regression baseline
data/                       # [gitignored] raw + processed
models/                     # [gitignored] persisted models
```

## Reference numbers (Kachuee et al. 2015 SVM regression, full UCI, no subject-level splits)

| | DBP MAE | DBP STD | MAP MAE | MAP STD | SBP MAE | SBP STD |
|---|---|---|---|---|---|---|
| Their SVM | 6.34 | 8.45 | 7.52 | 9.54 | 12.38 | 16.17 |

Their BHS grade: DBP grade B, SBP fails grade C. Use these as a regression-side sanity check; the writeup's main comparison is on classification metrics.

## References

- **Dataset**: Kachuee, Kiani, Mohammadzade, Shabany. _Cuff-Less Blood Pressure Estimation Dataset_. UCI ML Repository #340. https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation
- **Method baseline**: Kachuee, Kiani, Mohammadzade, Shabany (2015). _Cuff-less high-accuracy calibration-free blood pressure estimation using pulse transit time_. IEEE ISCAS.
- **AHA hypertension classification thresholds**: 2017 ACC/AHA guideline.
