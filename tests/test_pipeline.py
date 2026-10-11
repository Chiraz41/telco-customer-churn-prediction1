import json

import pytest

from churn import config as C
from churn.data import prepare_xy, split
from churn.features import build_preprocessor
from conftest import make_fake_telco


def test_prepare_xy():
    X, y = prepare_xy(make_fake_telco())
    assert "customerID" not in X and "Churn" not in X
    assert set(y.unique()) <= {0, 1}
    assert X["TotalCharges"].dtype.kind == "f"      # " " -> NaN


def test_preprocessor_no_nan_and_no_leak():
    X, y = prepare_xy(make_fake_telco())
    X_tr, X_te, y_tr, y_te = split(X, y)
    pre = build_preprocessor(X_tr)
    import numpy as np
    assert not np.isnan(pre.fit_transform(X_tr)).any()
    assert not np.isnan(pre.transform(X_te)).any()
    assert abs(y_tr.mean() - y_te.mean()) < 0.02     # stratification


def test_train_then_api(tmp_path, monkeypatch):
    """Entraîne sur données synthétiques et appelle l'API."""
    monkeypatch.setattr(C, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(C, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(C, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(C, "N_ITER_SEARCH", 2)
    monkeypatch.setattr(C, "CV_FOLDS", 3)
    import churn.evaluate as ev
    monkeypatch.setattr(ev, "FIGURES_DIR", tmp_path / "reports" / "figures")

    csv = tmp_path / "telco.csv"
    make_fake_telco().to_csv(csv, index=False)

    from churn import train
    train.main(str(csv))
    for f in ("preprocess.joblib", "churn_model.joblib",
              "metadata.json", "ui_schema.json"):
        assert (tmp_path / "artifacts" / f).exists()

    # API chargée avec ces artefacts
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    import api.main as api_main
    api_main.get_artifacts.cache_clear()

    from fastapi.testclient import TestClient
    client = TestClient(api_main.app)
    assert client.get("/health").json() == {"status": "ok"}

    X, _ = prepare_xy(make_fake_telco())
    r = client.post("/predict", json={"data": X.iloc[0].to_dict()})
    assert r.status_code == 200, r.text
    assert 0 <= r.json()["probabilite_churn"] <= 1

    r = client.post("/predict", json={"data": {"tenure": 3}})
    assert r.status_code == 422


def test_api_without_artifacts_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "vide"))
    import api.main as api_main
    api_main.get_artifacts.cache_clear()
    from fastapi.testclient import TestClient
    client = TestClient(api_main.app)
    assert client.get("/health").status_code == 200      # health sans modèle
    assert client.post("/predict", json={"data": {}}).status_code == 503
