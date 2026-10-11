"""API FastAPI de prédiction : uvicorn api.main:app --reload"""
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@lru_cache(maxsize=1)
def get_artifacts() -> dict:
    """Charge modèle, prétraitement et métadonnées (au premier appel)."""
    art_dir = Path(os.getenv("ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR))
    if not (art_dir / "churn_model.joblib").exists():
        raise HTTPException(
            status_code=503,
            detail="Artefacts introuvables : lancez d'abord `python -m churn.train`.",
        )
    with open(art_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return {
        "preprocess": joblib.load(art_dir / "preprocess.joblib"),
        "model": joblib.load(art_dir / "churn_model.joblib"),
        "metadata": metadata,
        "threshold": metadata["decision_threshold"],
    }


app = FastAPI(
    title="Telco Customer Churn API",
    description="API de prédiction du risque de résiliation des clients",
    version="1.0.0",
)


class CustomerData(BaseModel):
    data: dict[str, Any]


@app.get("/")
def home():
    art = get_artifacts()
    return {"message": "API Churn opérationnelle",
            "model": art["metadata"]["model_name"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    art = get_artifacts()
    preprocess, model, threshold = art["preprocess"], art["model"], art["threshold"]
    try:
        input_df = pd.DataFrame([customer.data])
        expected = list(preprocess.feature_names_in_)
        missing = [c for c in expected if c not in input_df.columns]
        if missing:
            raise HTTPException(status_code=422,
                                detail={"colonnes_manquantes": missing})

        processed = preprocess.transform(input_df[expected])
        probability = float(model.predict_proba(processed)[0, 1])
        prediction = int(probability >= threshold)

        return {
            "prediction": prediction,
            "resultat": ("Risque de résiliation élevé" if prediction == 1
                         else "Risque de résiliation faible"),
            "probabilite_churn": round(probability, 4),
            "seuil_decision": threshold,
            "modele": art["metadata"]["model_name"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
