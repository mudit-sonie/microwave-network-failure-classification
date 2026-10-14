# How to run this project

Step-by-step guide to clone the repo, run the full experiment and read the results.
Project: four-class microwave network failure classification (IDU / ODU / Cable / Power) from 164 alarm-duration features.

## 1. Clone the repo
```bash
git clone <your-repo-url>
cd <repo-folder>
```

## 2. Create an environment and install dependencies
Python 3.10+ is recommended (the results below were produced with Python 3.14 and scikit-learn 1.9).
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
`xgboost` is listed in `requirements.txt`; if it fails to install, the notebook still runs and simply skips XGBoost.

## 3. Download the dataset
The CSV is not stored in git (`data/*.csv` is in `.gitignore`), so download it yourself:

1. Get `dataset_TNSM_06995_v2.csv` from https://github.com/bonsai-lab-polimi/tnsm2024-data-centric
2. Put it in the `data/` folder, so the path is `data/dataset_TNSM_06995_v2.csv`.

Expected shape: 1,669 rows × 165 columns (164 `alarm_*` features + `label`).

## 4. Run the full experiment
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_experiments.ipynb
```
- Takes roughly 1.5–5 minutes depending on your CPU.
- Runs all 17 code cells: data checks, grouped train/test split, grouped 5-fold CV grid search for 11 models × 3 feature representations (raw / binary / binned), leakage check, imbalance check, alarm importance, final evaluation.
- It overwrites the notebook's saved outputs and regenerates everything in `models/` and `reports/`.
- Prefer to click through it? Run `jupyter lab notebooks/02_model_experiments.ipynb` and use **Run All**.

## 5. Try the saved model
```bash
python src/predict.py data/dataset_TNSM_06995_v2.csv --rows 5
```
Prints true label, predicted label and class probabilities for the first 5 rows. Use your own CSV (columns `alarm_0` … `alarm_163`, `label` optional) in place of the dataset path.

Expected output:
```
    true predicted  P(IDU)  P(ODU)  P(Cable)  P(Power)
0    ODU       ODU   0.014   0.984     0.001     0.001
1  Cable     Cable   0.012   0.016     0.962     0.010
2  Power     Power   0.020   0.036     0.008     0.937
3    ODU       ODU   0.022   0.966     0.004     0.009
4    ODU       ODU   0.021   0.966     0.003     0.010
```

## 6. Where the results are
| File | Contents |
|---|---|
| `reports/model_results.md` | Full summary: data checks, selected model, all 30 model/representation results, leakage and imbalance tables |
| `reports/model_comparison.csv` | Same model table as CSV |
| `reports/test_predictions.csv` | Per-row hold-out predictions and probabilities |
| `reports/leakage_check.csv` | Plain vs. grouped CV scores |
| `reports/imbalance_experiments.csv` | None vs. class weights vs. SMOTE |
| `reports/l1_logreg_coefficients.csv`, `reports/rf_permutation_importance.csv` | Alarm importance |
| `reports/figures/*.png` | Confusion matrix, model comparison, per-class metrics, representation comparison, leakage, importances |
| `models/final_model.joblib` | Fitted preprocessing + classifier |

## 7. What you should get
Seed is fixed at 42, so a run should reproduce these numbers (small differences are possible with other library versions).

**Selected model:** SVM-RBF on binned features (`C=10`, `gamma="scale"`), chosen by grouped-CV macro-F1 (0.947 ± 0.022).

**Hold-out test (334 rows):** accuracy 0.961, balanced accuracy 0.950, macro-F1 0.957.

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| IDU | 0.953 | 0.981 | 0.967 | 103 |
| ODU | 0.960 | 0.975 | 0.967 | 122 |
| Cable | 0.974 | 0.905 | 0.938 | 42 |
| Power | 0.969 | 0.940 | 0.955 | 67 |

**Test confusion matrix** (rows = true, columns = predicted):

| | IDU | ODU | Cable | Power |
|---|---|---|---|---|
| IDU | 101 | 1 | 0 | 1 |
| ODU | 1 | 119 | 1 | 1 |
| Cable | 1 | 3 | 38 | 0 |
| Power | 3 | 1 | 0 | 63 |

**Top models by CV macro-F1:** SVM-RBF binned 0.947, LogReg-L2 binned 0.935, LogReg-L1 binned 0.930, SVM-linear binned 0.928. The majority-class dummy baseline scores 0.134.

## 8. Notes on the data and protocol
- **Class balance:** ODU 611 (36.6%), IDU 515 (30.9%), Power 336 (20.1%), Cable 207 (12.4%). This is moderately imbalanced. Class weights and SMOTE changed macro-F1 by about 0.005 or less, so no rebalancing is applied; metrics are macro-averaged and splits are stratified. Cable has the lowest recall.
- **Duplicates:** 1,135 unique alarm vectors across 1,669 rows (no vector has conflicting labels). Rows with identical vectors are grouped so none appears in both train and test. Plain random splits overstate scores by 0.01–0.05 macro-F1 (`reports/leakage_check.csv`).
- **Selection:** by CV only. Test columns in the model table are for transparency and were not used to choose the model.
- **Cable test support is only 42 rows**, so test-score differences of a few points between models are within noise.

## Troubleshooting
- `FileNotFoundError` for the CSV: the dataset is missing from `data/` (step 3).
- `jupyter: command not found`: activate the virtual environment (step 2) or run `python -m jupyter nbconvert ...`.
- Kernel not found: `python -m ipykernel install --user --name python3`.
- Unpickling error in `predict.py`: use the same scikit-learn version as in `requirements.txt`, or re-run step 4 to regenerate `models/final_model.joblib`.
