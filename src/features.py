"""Shared data loading, feature representations and grouping helpers.

Kept in a module (not the notebook) so the saved joblib pipeline can be
re-loaded from any script or the demo without redefining functions.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "label"
CLASS_NAMES = {0: "IDU", 1: "ODU", 2: "Cable", 3: "Power"}

# Duration bins from Di Cicco et al. (2024), Sec. III-C "Mode-2: Categorical"
#   0 -> inactive, 1-45 s -> low, 46-450 s -> medium, 451-900 s -> high severity
BIN_EDGES = [0, 45, 450]


def load_data(path):
    df = pd.read_csv(Path(path))
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return df, X, y


def vector_groups(X):
    """One integer group id per unique 164-alarm input vector (label not used)."""
    keys = pd.util.hash_pandas_object(X, index=False)
    return pd.factorize(keys)[0]


# ---- feature representations (stateless, so safe inside CV pipelines) ----
def to_binary(X):
    """Mode-1: alarm present (>0 s) in the 15-min window or not."""
    return (np.asarray(X) > 0).astype(np.int8)


def to_bins(X):
    """Mode-2: ordinal severity code 0-3 using BIN_EDGES (right-inclusive)."""
    X = np.asarray(X)
    return np.digitize(X, BIN_EDGES, right=True).astype(np.int8)
