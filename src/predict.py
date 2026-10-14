"""Diagnose failure type for alarm windows with the saved pipeline.

Usage:
    python src/predict.py path/to/windows.csv            # CSV with alarm_0 ... alarm_163 (label optional)
    python src/predict.py data/dataset_TNSM_06995_v2.csv --rows 5
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import CLASS_NAMES, TARGET  # noqa: E402  (also needed to unpickle the pipeline)

FEATURES = [f"alarm_{i}" for i in range(164)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", default=ROOT / "models" / "final_model.joblib")
    ap.add_argument("--rows", type=int, default=None, help="only score the first N rows")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        sys.exit(f"missing {len(missing)} alarm columns, e.g. {missing[:5]}")
    if a.rows:
        df = df.head(a.rows)

    model = joblib.load(a.model)
    X = df[FEATURES]
    out = pd.DataFrame({"predicted": pd.Series(model.predict(X), index=df.index).map(CLASS_NAMES)})
    if hasattr(model, "predict_proba"):
        P = pd.DataFrame(model.predict_proba(X), index=df.index, columns=[f"P({n})" for n in CLASS_NAMES.values()])
        out = out.join(P.round(3))
    if TARGET in df.columns:
        out.insert(0, "true", df[TARGET].map(CLASS_NAMES))
    print(out.to_string())


if __name__ == "__main__":
    main()
