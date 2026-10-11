"""Chargement et préparation des données (phases 2 et 3 CRISP-DM)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (ID_COLUMN, OPENML_DATA_ID, RANDOM_STATE, TARGET,
                     TEST_SIZE)


def load_raw(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Charge le dataset depuis un CSV local, sinon depuis OpenML."""
    if csv_path is not None:
        return pd.read_csv(csv_path)

    from sklearn.datasets import fetch_openml

    raw = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True)
    return raw.frame.copy()


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """TotalCharges -> numérique, cible -> 0/1, retrait de customerID."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    X = df.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    """Découpage stratifié train/test 80/20."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
