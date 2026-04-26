# SVM baseline reproduction — partner brief

## What we're building together

We're submitting a writeup that compares **direct multi-class hypertension classification** (RF / XGBoost / 1D CNN) against the canonical **regression-then-threshold approach** of Kachuee et al. (2015 ISCAS, "Cuff-less high-accuracy calibration-free blood pressure estimation using pulse transit time"). The methodological contribution is showing whether (and by how much) the classification framing wins over regression-then-threshold on the same data, with subject-level splits enforced.

Your job is to produce the **regression-then-threshold baseline** so the comparison is real. The pipeline below already does the direct-classification side, so all you need to plug in is your SVM with outputs in our schema.

## The deliverable

When you're done, the repo's `data/processed/` directory should contain:

| File | Schema | Used by |
|---|---|---|
| `svm_metrics.json` | `MultiClassMetrics` JSON (matches `bme_ml/evaluation.py:MultiClassMetrics`) | `notebooks/04_results.ipynb` comparison table |
| `svm_regression_metrics.json` | `{"sbp_mae": float, "sbp_std": float, "dbp_mae": ..., "dbp_std": ..., "map_mae": ..., "map_std": ...}` | Reference comparison vs Kachuee Table I |
| `svm_predictions.csv` | columns: `row_index, sbp_pred, dbp_pred, label_3class_pred` (one row per held-out test segment) | Allows me to recompute anything if needed |

And `models/svm_regression.joblib` (your trained pipeline).

If those four files exist, I uncomment one block in `04_results.ipynb` and the comparison table picks you up automatically.

## What's already in place (don't redo)

- **Data download + parsing** — `bme_ml/paths.py:download_dataset` + `bme_ml/data_loader.py`. All four UCI parts load uniformly (fixed v7.3 detection earlier).
- **Preprocessing + windowing** — `bme_ml/preprocessing.py`. ECG/PPG bandpass, 8-second windows @ 125 Hz, segment QC.
- **Per-record processing pipeline** — `bme_ml/pipeline.py:process_record` (joblib-parallelized in `notebooks/02_build_features.ipynb`).
- **Subject-level splits** — `bme_ml/splits.py`. Persisted as `data/processed/splits.json`. **You must reuse these splits** so test sets match.
- **Labels** — both `label_binary` and `label_3class` (AHA: Normal SBP<120∧DBP<80 / Elevated 120-129 SBP∧DBP<80 / Hypertensive SBP≥130∨DBP≥80) live in `features.parquet`.
- **Multi-class metrics** — `bme_ml/evaluation.py:evaluate_multiclass`. **Use this** so your numbers are comparable.

## What you need to implement

### 1. Add the missing Kachuee features to `bme_ml/features.py`

The paper's feature set is **10 features**, of which we currently extract only 2 cleanly (`ptt_ms` ≈ PTTp, `hr_bpm`). The other 8 are missing or wrong. Add these:

| Kachuee | Variable name to add | What it is |
|---|---|---|
| **PTTp** | `pttp_ms` | R-peak → next PPG **systolic peak** time. (We already have this as `ptt_ms` — rename for clarity.) |
| **PTTf** | `pttf_ms` | R-peak → next PPG **foot/minimum** (start of upstroke). |
| **PTTd** | `pttd_ms` | R-peak → next PPG **maximum-slope point** (peak of dPPG/dt during upstroke). |
| **HR** | `hr_bpm` | Heart rate from peak-to-peak (already have it). |
| **AI** | `ai_kachuee` | Diastolic peak amplitude / systolic peak amplitude — the **secondary post-dicrotic-notch peak**, NOT the dicrotic notch itself. Our current `ppg_aug_index` is wrong on this and uses the notch. Replace it. |
| **LASI** | `lasi_ms` | Time interval between **systolic peak** and **diastolic peak** (within one cardiac cycle). |
| **S1** | `s1_area` | Area under PPG curve from **foot → systolic peak**. |
| **S2** | `s2_area` | Area under PPG curve from **systolic peak → dicrotic notch**. |
| **S3** | `s3_area` | Area under PPG curve from **dicrotic notch → diastolic peak**. |
| **S4** | `s4_area` | Area under PPG curve from **diastolic peak → next foot**. |

The hard part is **diastolic-peak detection**. NeuroKit2's `ppg_process` returns systolic peaks but not diastolic. Approach:

1. For each cardiac cycle (between two consecutive PPG troughs):
2. Find the systolic peak (already detected by NeuroKit).
3. On the descending limb (peak → next foot), compute the second derivative of the smoothed PPG.
4. The **dicrotic notch** is the first local maximum of d²PPG/dt² after the systolic peak (a.k.a. the inflection where curvature reverses from concave-down to concave-up).
5. The **diastolic peak** is the next local maximum of the PPG signal itself after the notch (or, if no clear peak exists for that pulse, the next local maximum of d²PPG/dt² — and skip the cycle for AI/LASI/S2-S4 if neither resolves).

Per-cycle, return median across cycles in the segment (consistent with how the existing features are aggregated).

**Sanity checks**:
- `pttp_ms > pttd_ms > pttf_ms` should hold per cycle (foot first, then max-slope, then peak).
- `s1 + s2 + s3 + s4` should approximately equal the total area under the PPG cycle (above its foot baseline).
- `ai_kachuee` typically lies in [0.3, 0.9] for healthy adults.

Once added, update `FEATURE_NAMES` in `features.py` and add two named subsets:

```python
KACHUEE_FEATURES = [
    "pttp_ms", "pttf_ms", "pttd_ms", "hr_bpm",
    "ai_kachuee", "lasi_ms",
    "s1_area", "s2_area", "s3_area", "s4_area",
]
OUR_FEATURES = KACHUEE_FEATURES + [
    "rr_sd_ms", "hrv_rmssd_ms", "hrv_sdnn_ms",
    "ppg_pw50_ms", "ppg_pw25_ms", "ppg_rise_ms",
    "ppg_decay_ms", "ecg_qrs_ms",
]
```

### 2. Re-run feature extraction

```bash
# In a fresh terminal, with the project's venv active:
jupyter nbconvert --to notebook --execute --inplace notebooks/02_build_features.ipynb
```

This regenerates `features.parquet` with the new columns. **Set `MAX_RECORDS = None`** for the full dataset (~30–60 min on a decent CPU).

### 3. Train the SVM regression

The paper trains two SVMs — one for SBP, one for DBP — on the 10-feature set. Hyperparameters from Kachuee et al. (2015):

- Kernel: RBF
- Cost (C) and gamma: tuned per the paper. They don't publish exact values; you'll need to grid-search inside our 5-fold CV. Reasonable starting grid:
  - `C ∈ {0.1, 1, 10, 100}`
  - `gamma ∈ {1e-3, 1e-2, 1e-1, 1, 'scale'}`
- Standardize features (`StandardScaler` → `SVR`); SVMs are not scale-invariant.

Reuse the existing CV iterator pattern:

```python
import joblib, json, numpy as np, pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from bme_ml.paths import setup_paths
from bme_ml.splits import Splits, select_rows, add_subject_id
from bme_ml.labels import add_multiclass_column, multiclass_label
from bme_ml.features import KACHUEE_FEATURES
from bme_ml.evaluation import evaluate_multiclass

paths = setup_paths()
features = add_subject_id(pd.read_parquet(paths.features_parquet))
features = add_multiclass_column(features)
splits = Splits.from_json(paths.splits_json)

# Use the SAME splits as RF/XGB/CNN — do not re-derive.
train_df = select_rows(features, [s for f in splits.cv_folds for s in f['train']] +
                                  [s for f in splits.cv_folds for s in f['val']])
test_df = select_rows(features, splits.test_subjects)

# Drop NaN rows on the Kachuee subset.
sub_train = train_df.dropna(subset=KACHUEE_FEATURES + ['sbp', 'dbp'])
sub_test  = test_df.dropna(subset=KACHUEE_FEATURES + ['sbp', 'dbp'])

X_train = sub_train[KACHUEE_FEATURES].to_numpy(np.float32)
X_test  = sub_test[KACHUEE_FEATURES].to_numpy(np.float32)

# CV groups for the train pool.
gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(sub_train, groups=sub_train['subject_id']))

# Train two SVRs.
svm_pipelines = {}
for target in ('sbp', 'dbp'):
    y_train = sub_train[target].to_numpy(np.float32)
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('svr', SVR(kernel='rbf'))])
    grid = GridSearchCV(pipe,
                        param_grid={'svr__C': [0.1, 1, 10, 100],
                                    'svr__gamma': [1e-3, 1e-2, 1e-1, 1, 'scale']},
                        cv=cv_splits, scoring='neg_mean_absolute_error',
                        n_jobs=-1, verbose=1, refit=True)
    grid.fit(X_train, y_train)
    svm_pipelines[target] = grid.best_estimator_
    print(f'{target} best params:', grid.best_params_)
```

### 4. Threshold to 3-class + save outputs

```python
# Predict on test set.
sbp_pred = svm_pipelines['sbp'].predict(X_test)
dbp_pred = svm_pipelines['dbp'].predict(X_test)

# Apply AHA thresholds via our shared helper.
y_pred = np.fromiter(
    (multiclass_label(s, d) for s, d in zip(sbp_pred, dbp_pred)),
    dtype=np.int64, count=len(sbp_pred),
)
y_true = sub_test['label_3class'].to_numpy(np.int64)

# Build a "soft" probability matrix from the regression output. Simplest
# option: assign 1.0 to the predicted class. Cleaner option that gives
# a real AUROC: probability via distance from threshold (e.g. logistic
# of (sbp_pred - 130) for the hypertensive class). Pick one and document.
y_proba = np.zeros((len(y_pred), 3), dtype=np.float32)
y_proba[np.arange(len(y_pred)), y_pred] = 1.0   # placeholder hard probabilities

metrics = evaluate_multiclass(y_true, y_pred, y_proba)

# Save in the schema the comparison table expects.
import dataclasses
(paths.processed / 'svm_metrics.json').write_text(json.dumps(dataclasses.asdict(metrics), indent=2))

reg_metrics = {
    'sbp_mae': float(np.abs(sbp_pred - sub_test['sbp']).mean()),
    'sbp_std': float((sbp_pred - sub_test['sbp']).std()),
    'dbp_mae': float(np.abs(dbp_pred - sub_test['dbp']).mean()),
    'dbp_std': float((dbp_pred - sub_test['dbp']).std()),
}
(paths.processed / 'svm_regression_metrics.json').write_text(json.dumps(reg_metrics, indent=2))

pd.DataFrame({
    'row_index': sub_test.index,
    'sbp_pred': sbp_pred, 'dbp_pred': dbp_pred,
    'label_3class_pred': y_pred,
}).to_csv(paths.processed / 'svm_predictions.csv', index=False)

joblib.dump(svm_pipelines, paths.models / 'svm_regression.joblib')
```

About the AUROC probabilities: a hard one-hot gives a degenerate AUROC (0.5 for any wrong class). A better approach: treat the SBP-prediction-distance from the 130 mmHg threshold as a hypertensive score, e.g.

```python
y_proba_hyper = 1 / (1 + np.exp(-(sbp_pred - 130) / 10))  # logistic, scale=10 mmHg
```

then renormalize across classes. Pick whichever you prefer and **document the choice** in your commit message — I'll match it in the writeup.

## Methodology constraints (non-negotiable)

These are the things that make our comparison apples-to-apples. **Don't change any of them**:

1. **Use the same `splits.json`** the RF/XGB/CNN already use. Do not re-derive splits with different seeds or stratification.
2. **Use the same AHA thresholds** for the 3-class label (use `bme_ml.labels.multiclass_label` directly). Do not invent new thresholds.
3. **Subject-level splits are required.** No segment-level random splits. (`splits.json` already enforces this.)
4. **Train and test on the same 8-second windowed segments** that go into `features.parquet`. Don't re-window the data.
5. **Standardize features** before fitting the SVM (already in the example pipeline).

If you find yourself wanting to deviate from any of the above, ping me first — most likely your concern is real and the comparison framing needs adjusting jointly.

## Setup

Clone the repo and set up a venv. On a CUDA-12.4 machine:

```bash
git clone <REPO_URL>
cd bme-ml-bp-classification
python -m venv .venv && .venv/Scripts/activate     # Windows; use bin/activate on Linux/Mac
pip install -r requirements.txt
# torch is only needed for the CNN side; you don't need it for the SVM work
# but installing it lets you run notebook 05 if you want to verify the CNN side
pip install --index-url https://download.pytorch.org/whl/cu124 torch

# Download the UCI dataset (~3.4 GB, ~5–10 min depending on UCI's server)
python -c "from bme_ml.paths import setup_paths, download_dataset; download_dataset(setup_paths())"

# Run the existing pipeline to get features + splits + my baseline numbers
jupyter nbconvert --to notebook --execute --inplace notebooks/02_build_features.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_train_tabular.ipynb
```

After step 3 you'll have `features.parquet`, `splits.json`, and the RF/XGBoost/PTT-LR metrics JSONs. Then add your features and run your SVM training script.

## Verification

Before considering yourself done, sanity-check:

- `data/processed/svm_metrics.json` exists, parses as `MultiClassMetrics`, and `confusion` is 3×3.
- `svm_regression_metrics.json`'s SBP MAE is within ~3 mmHg of Kachuee's reported 12.38 mmHg (Table I, full UCI). If it's wildly different (e.g. > 25 mmHg), something is wrong — most likely the splits, the feature set, or the standardization.
- `svm_predictions.csv` has one row per test segment (`len(svm_predictions) == len(test_df.dropna(subset=KACHUEE_FEATURES))`).
- Run `notebooks/04_results.ipynb` after uncommenting the SVM block in cell 1 — your model should appear in the comparison table without errors.

## Reference numbers

**From Kachuee et al. (2015), Table I, full UCI dataset, regression task, no subject-level split**:

| | DBP MAE | DBP STD | MAP MAE | MAP STD | SBP MAE | SBP STD |
|---|---|---|---|---|---|---|
| SVM | 6.34 | 8.45 | 7.52 | 9.54 | 12.38 | 16.17 |

Their BHS grade (Table II): DBP grade B, SBP fails grade C.

**Our smoke-test numbers (50-record subset, ~423 segments, multi-class)**:

| Model | Accuracy | F1-macro | Hyper. AUROC | Hyper. Recall |
|---|---|---|---|---|
| 1D CNN | 0.527 | 0.368 | 0.607 | 0.877 |
| XGBoost | 0.473 | 0.442 | 0.580 | 0.625 |
| Random Forest | 0.402 | 0.384 | 0.563 | 0.500 |
| PTT-only LR | 0.286 | 0.281 | 0.419 | 0.146 |

These are not meaningful numbers (50 records is far too few — we expect strong improvement on the full dataset). They're included so you have something to verify your pipeline ran end-to-end.

After your full sweep, expected ballpark for **subject-level evaluation** on the AHA 3-class task:

- Random / PTT-only: 30–40% accuracy, hyper. AUROC ~0.45–0.55
- RF / XGBoost / CNN: 55–70% accuracy, hyper. AUROC 0.70–0.85
- Your SVM-then-threshold: probably between the PTT baseline and the engineered-feature classifiers. If it lands above the engineered-feature classifiers, that's interesting and we should dig into why.

## Coordination

- **Branch off main** for your work; open a PR when ready. I'll review the feature-extraction code carefully (diastolic-peak detection is finicky and we don't want bugs to muddy the comparison).
- **Document the AUROC probability choice** in the SVM commit message — see Section 4 above.
- **If you change `features.py`**, re-run notebook 03 too so the RF/XGBoost numbers reflect the same feature columns. Don't ship a state where my models trained on `OUR_FEATURES` and yours on `KACHUEE_FEATURES` if a comparable comparison can be made.
- **Reach out** if anything's unclear, especially:
  - Diastolic-peak detection edge cases (some pulses don't have a clear secondary peak)
  - Whether `S1–S4` should be raw areas (paper's recommendation) or the IPA ratio (paper's original formulation)
  - Whether to use `OUR_FEATURES` (12+ features) for an additional SVM run, since "richer features + classification" is part of the story

That's it. Ping me when you've added the Kachuee features and we'll review the implementation before you grind through the full sweep.
