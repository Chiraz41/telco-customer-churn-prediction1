"""Entraînement complet : python -m churn.train [--data-path fichier.csv].

Reproduit les phases 3 à 6 du notebook : préparation, optimisation des 5
modèles, évaluation, choix du seuil (coût métier) et sauvegarde des artefacts.
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import config as C
from .data import load_raw, prepare_xy, split
from .evaluate import evaluate_models, save_figures, threshold_analysis
from .features import build_preprocessor, build_ui_schema
from .models import PARAM_DISTRIBUTIONS, get_models


def main(data_path: str | None = None) -> None:
    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw(data_path)
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Train : {X_train.shape} | Test : {X_test.shape}")

    preprocess = build_preprocessor(X_train)
    Xtr = preprocess.fit_transform(X_train)   # ajusté sur train uniquement
    Xte = preprocess.transform(X_test)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = get_models(scale_pos_weight)
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True,
                         random_state=C.RANDOM_STATE)

    best_models, cv_rows = {}, []
    for name, model in models.items():
        print(f"\nOptimisation de : {name}")
        t0 = time.time()
        search = RandomizedSearchCV(
            model, PARAM_DISTRIBUTIONS[name], n_iter=C.N_ITER_SEARCH,
            scoring="roc_auc", cv=cv, random_state=C.RANDOM_STATE,
            n_jobs=-1, refit=True,
        )
        search.fit(Xtr, y_train)
        best_models[name] = search.best_estimator_
        cv_rows.append({
            "Modèle": name,
            "Meilleure AUC-ROC CV": search.best_score_,
            "Meilleurs paramètres": json.dumps(search.best_params_, default=str),
            "Temps (s)": round(time.time() - t0, 2),
        })
        print(f"  AUC-ROC CV : {search.best_score_:.4f}")

    cv_df = (pd.DataFrame(cv_rows)
             .sort_values("Meilleure AUC-ROC CV", ascending=False)
             .reset_index(drop=True))
    print("\nClassement (validation croisée) :\n", cv_df.to_string())

    # --- Évaluation sur le jeu de test
    eval_df = evaluate_models(best_models, Xte, y_test)
    print("\nÉvaluation sur le test (seuil 0.5) :\n", eval_df.round(4).to_string())

    final_model = best_models[C.FINAL_MODEL_NAME]
    proba = final_model.predict_proba(Xte)[:, 1]
    cost_df = threshold_analysis(y_test, proba)
    best_row = cost_df.loc[cost_df["Coût total"].idxmin()]
    best_threshold = float(best_row["Seuil"])
    print(f"\nSeuil minimisant le coût : {best_threshold} "
          f"(coût = {best_row['Coût total']:.0f})")

    # --- Sauvegardes
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(C.REPORTS_DIR / "cv_results.csv", index=False)
    eval_df.to_csv(C.REPORTS_DIR / "test_metrics.csv", index=False)
    cost_df.to_csv(C.REPORTS_DIR / "threshold_cost.csv", index=False)
    save_figures(best_models, Xte, y_test, cost_df)

    joblib.dump(preprocess, C.ARTIFACT_DIR / "preprocess.joblib")
    joblib.dump(final_model, C.ARTIFACT_DIR / "churn_model.joblib")

    metadata = {
        "project": "Telco Customer Churn Prediction",
        "model_name": C.FINAL_MODEL_NAME,
        "target": C.TARGET,
        "positive_class": 1,
        "decision_threshold": best_threshold,
        "threshold_status": "provisional",
        "threshold_reason": ("Lowest estimated business cost among tested "
                             "thresholds under assumed FN and FP costs"),
        "false_negative_cost": C.COST_FN,
        "false_positive_cost": C.COST_FP,
        "random_state": C.RANDOM_STATE,
    }
    with open(C.ARTIFACT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    with open(C.ARTIFACT_DIR / "ui_schema.json", "w", encoding="utf-8") as f:
        json.dump(build_ui_schema(X_train), f, indent=2, ensure_ascii=False)

    print(f"\nArtefacts sauvegardés dans {C.ARTIFACT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", default=None,
                    help="CSV local (sinon téléchargement depuis OpenML)")
    main(ap.parse_args().data_path)
