"""Évaluation (phase 5) : métriques, seuils, coût métier, figures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from .config import COST_FN, COST_FP, FIGURES_DIR, THRESHOLDS


def evaluate_models(best_models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in best_models.items():
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
        rows.append({
            "Modèle": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "AUC-ROC": auc,
            "PR-AUC": average_precision_score(y_test, proba),
            "Gini": 2 * auc - 1,
        })
    return (pd.DataFrame(rows)
            .sort_values("AUC-ROC", ascending=False)
            .reset_index(drop=True))


def threshold_analysis(y_test, proba) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({
            "Seuil": t,
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "Coût total": fn * COST_FN + fp * COST_FP,
        })
    return pd.DataFrame(rows)


def save_figures(best_models: dict, X_test, y_test, cost_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    probas = {n: m.predict_proba(X_test)[:, 1] for n, m in best_models.items()}

    # Courbes ROC
    plt.figure(figsize=(9, 6))
    for n, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, label=f"{n} (AUC = {roc_auc_score(y_test, p):.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Classification aléatoire")
    plt.xlabel("Taux de faux positifs"); plt.ylabel("Taux de vrais positifs")
    plt.title("Comparaison des courbes ROC"); plt.legend(loc="lower right")
    plt.grid(alpha=0.3); plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150,
                                     bbox_inches="tight"); plt.close()

    # Courbes Precision-Recall
    plt.figure(figsize=(9, 6))
    for n, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        plt.plot(rec, prec, label=f"{n} (AP = {average_precision_score(y_test, p):.3f})")
    plt.xlabel("Rappel"); plt.ylabel("Précision")
    plt.title("Comparaison des courbes Precision-Recall"); plt.legend()
    plt.grid(alpha=0.3); plt.savefig(FIGURES_DIR / "precision_recall_curves.png",
                                     dpi=150, bbox_inches="tight"); plt.close()

    # Matrices de confusion (seuil 0.5)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, (n, p) in enumerate(probas.items()):
        cm = confusion_matrix(y_test, (p >= 0.5).astype(int), labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["Pas de churn", "Churn"]).plot(
            ax=axes[i], colorbar=False, values_format="d")
        axes[i].set_title(n)
    for j in range(len(probas), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "confusion_matrices.png",
                                    dpi=150, bbox_inches="tight"); plt.close()

    # Coût métier selon le seuil
    plt.figure(figsize=(9, 5))
    plt.plot(cost_df["Seuil"], cost_df["Coût total"], marker="o")
    plt.xlabel("Seuil de décision"); plt.ylabel("Coût métier estimé")
    plt.title("Évolution du coût métier selon le seuil"); plt.grid(alpha=0.3)
    plt.savefig(FIGURES_DIR / "business_cost.png", dpi=150, bbox_inches="tight")
    plt.close()
