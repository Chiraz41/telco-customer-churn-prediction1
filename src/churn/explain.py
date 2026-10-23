"""Module 5 – Explainable AI : explications globales et locales (SHAP, LIME)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.inspection import permutation_importance
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline

from .config import FIGURES_DIR, RANDOM_STATE, REPORTS_DIR

# Variables sensibles présentes dans Telco (pas d'origine ethnique dans ce jeu)
SENSITIVE = ["gender", "SeniorCitizen", "Partner", "Dependents"]

LABELS_FR = {
    "Contract": "type de contrat", "tenure": "ancienneté",
    "MonthlyCharges": "facture mensuelle", "TotalCharges": "total facturé",
    "InternetService": "type d'accès internet", "OnlineSecurity": "sécurité en ligne",
    "OnlineBackup": "sauvegarde en ligne", "DeviceProtection": "protection d'appareil",
    "TechSupport": "support technique", "StreamingTV": "streaming TV",
    "StreamingMovies": "streaming films", "PaymentMethod": "mode de paiement",
    "PaperlessBilling": "facturation dématérialisée", "MultipleLines": "lignes multiples",
    "PhoneService": "service téléphonique", "SeniorCitizen": "statut senior",
    "gender": "sexe", "Partner": "vie en couple", "Dependents": "personnes à charge",
}


# ----------------------------------------------------------------- utilitaires
def to_dense(X) -> np.ndarray:
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def get_feature_names(preprocess) -> list[str]:
    """Noms des colonnes après prétraitement (sans le préfixe num__/cat__)."""
    return [n.split("__", 1)[1] for n in preprocess.get_feature_names_out()]


def original_column(feature: str, columns: list[str]) -> str:
    """'Contract_Month-to-month' -> 'Contract' ; 'tenure' -> 'tenure'."""
    for col in sorted(columns, key=len, reverse=True):
        if feature == col or feature.startswith(col + "_"):
            return col
    return feature


def group_by_original(values, feature_names, original_columns) -> pd.DataFrame:
    """Somme des contributions des colonnes one-hot par variable d'origine."""
    cols = [original_column(f, list(original_columns)) for f in feature_names]
    df = pd.DataFrame(np.atleast_2d(values), columns=cols)
    return df.T.groupby(level=0).sum().T


# --------------------------------------------------------- explications globales
def native_importance(model, names) -> pd.DataFrame | None:
    """Coefficients (régression logistique) ou importance par impureté (arbres)."""
    if hasattr(model, "coef_"):
        v = np.ravel(model.coef_)
    elif hasattr(model, "feature_importances_"):
        v = np.asarray(model.feature_importances_)
    else:                                    # KNN : pas d'importance native
        return None
    df = pd.DataFrame({"feature": names, "importance": v})
    return df.reindex(df["importance"].abs().sort_values(ascending=False).index)


def permutation_table(best_models: dict, preprocess, X_test, y_test,
                      n_repeats: int = 10) -> pd.DataFrame:
    """Importance par permutation (baisse d'AUC), sur les variables d'ORIGINE.

    Même métrique et mêmes données pour les 5 modèles -> comparable.
    """
    rows = []
    for name, model in best_models.items():
        pipe = Pipeline([("prep", preprocess), ("model", model)])
        r = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                   n_repeats=n_repeats, random_state=RANDOM_STATE,
                                   n_jobs=1)
        for col, m, s in zip(X_test.columns, r.importances_mean, r.importances_std):
            rows.append({"Modèle": name, "Variable": col, "Baisse AUC": m, "Ecart-type": s})
    return pd.DataFrame(rows)


def compute_shap(model, X_dense, names):
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer(pd.DataFrame(X_dense, columns=names))
    return explainer, sv            # valeurs en log-odds (échelle du modèle XGBoost)


def plot_native_importance(models: dict, names, path, top=12):
    items = {n: native_importance(m, names) for n, m in models.items()}
    items = {n: d for n, d in items.items() if d is not None}
    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 6))
    axes = np.atleast_1d(axes)
    for ax, (n, d) in zip(axes, items.items()):
        d = d.head(top).iloc[::-1]
        ax.barh(d["feature"], d["importance"])
        ax.set_title(n + (" (coefficients)" if n == "Logistic Regression" else " (impureté)"))
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_permutation(perm_df: pd.DataFrame, path, top=10):
    pivot = perm_df.pivot(index="Variable", columns="Modèle", values="Baisse AUC")
    order = pivot.max(axis=1).sort_values(ascending=False).head(top).index
    ax = pivot.loc[order].iloc[::-1].plot.barh(figsize=(9, 7))
    ax.set_xlabel("Baisse d'AUC-ROC après permutation")
    ax.set_title("Importance par permutation (jeu de test)")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_shap_global(sv, out_dir, n_dependence=4):
    import shap
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP summary (beeswarm) – XGBoost")
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

    top = np.abs(sv.values).mean(0).argsort()[::-1][:n_dependence]
    for j in top:
        shap.plots.scatter(sv[:, int(j)], color=sv, show=False)
        plt.savefig(out_dir / f"shap_dependence_{sv.feature_names[int(j)]}.png",
                    dpi=150, bbox_inches="tight"); plt.close()


def run_global_explanations(best_models, final_model, preprocess, X_test, y_test, threshold):
    """Génère tous les graphiques/CSV globaux (appelé par train.py)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    names = get_feature_names(preprocess)
    Xte = to_dense(preprocess.transform(X_test))

    plot_native_importance(best_models, names, FIGURES_DIR / "native_importance.png")

    perm = permutation_table(best_models, preprocess, X_test, y_test)
    perm.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    plot_permutation(perm, FIGURES_DIR / "permutation_importance.png")

    _, sv = compute_shap(final_model, Xte, names)
    plot_shap_global(sv, FIGURES_DIR)
    g = group_by_original(np.abs(sv.values), names, X_test.columns).mean()
    (g.sort_values(ascending=False).rename("mean|SHAP|")
      .to_csv(REPORTS_DIR / "shap_importance_by_variable.csv"))

    fairness_table(final_model, preprocess, X_test, y_test, threshold).to_csv(
        REPORTS_DIR / "fairness_by_group.csv", index=False)
    pd.DataFrame(counterfactual_shift(final_model, preprocess, X_test)).to_csv(
        REPORTS_DIR / "counterfactual_sensitive.csv", index=False)


# ----------------------------------------------------------- explications locales
def explain_one(model, preprocess, row: pd.DataFrame, background_dense, threshold):
    """SHAP + LIME + synthèse pour UN individu (DataFrame d'une ligne, colonnes brutes)."""
    names = get_feature_names(preprocess)
    x = to_dense(preprocess.transform(row))
    proba = float(model.predict_proba(x)[0, 1])
    _, sv = compute_shap(model, x, names)
    lime_exp = lime_explain(model, background_dense, x[0], names, preprocess)
    cmp_df = compare_shap_lime(sv[0], lime_exp, names, row.columns)
    shap_g = group_by_original(sv.values, names, row.columns).iloc[0]
    return {"proba": proba, "decision": int(proba >= threshold), "shap": sv[0],
            "lime": lime_exp, "comparison": cmp_df,
            "sentence": summary_sentence(shap_g, row.iloc[0], proba, threshold)}


def lime_explain(model, background_dense, x_row, names, preprocess, num_features=15):
    from lime.lime_tabular import LimeTabularExplainer
    n_num = len(preprocess.transformers_[0][2])        # colonnes numériques d'abord
    explainer = LimeTabularExplainer(
        background_dense, feature_names=names, class_names=["Reste", "Part"],
        categorical_features=list(range(n_num, len(names))),
        mode="classification", discretize_continuous=False,
        random_state=RANDOM_STATE)
    return explainer.explain_instance(x_row, model.predict_proba,
                                      num_features=num_features, num_samples=5000)


def compare_shap_lime(sv_row, lime_exp, names, original_columns, top=8) -> pd.DataFrame:
    lime_vals = np.zeros(len(names))
    for idx, w in lime_exp.as_map()[1]:
        lime_vals[idx] = w
    df = pd.DataFrame({
        "SHAP (log-odds)": group_by_original(sv_row.values, names, original_columns).iloc[0],
        "LIME (poids)": group_by_original(lime_vals, names, original_columns).iloc[0],
    })
    both = df.iloc[:, 1] != 0                      # LIME n'a pas retenu la variable -> n/a
    df["Même sens ?"] = np.where(both, np.sign(df.iloc[:, 0]) == np.sign(df.iloc[:, 1]), None)
    df = df.reindex(df.iloc[:, 0].abs().sort_values(ascending=False).index).head(top)
    return df.round(3)


def summary_sentence(shap_grouped: pd.Series, row: pd.Series, proba, threshold, k=2) -> str:
    high = proba >= threshold
    s = shap_grouped.sort_values(ascending=not high)       # moteurs de la décision
    drivers = [c for c in s.index if (shap_grouped[c] > 0) == high][:k]
    def fmt(c):
        return f"{LABELS_FR.get(c, c)} ({row[c]})"
    level = "élevé" if high else "faible"
    why = "à cause de" if high else "grâce à"
    txt = (f"Le risque de départ est {level} ({proba:.0%}, seuil {threshold:.0%}) "
           f"principalement {why} : " + " ; ".join(fmt(c) for c in drivers) + ".")
    opp = shap_grouped.sort_values(ascending=high)
    opp = [c for c in opp.index if (shap_grouped[c] > 0) != high][:1]
    if opp:
        txt += f" À l'inverse, le facteur « {fmt(opp[0])} » joue en sens contraire."
    return txt


# ------------------------------------------------------------------ équité
def fairness_table(model, preprocess, X_test, y_test, threshold, cols=SENSITIVE):
    proba = model.predict_proba(preprocess.transform(X_test))[:, 1]
    pred = (proba >= threshold).astype(int)
    y = np.asarray(y_test)
    rows = []
    for col in cols:
        for g in X_test[col].unique():
            m = (X_test[col] == g).to_numpy()
            rows.append({"Variable": col, "Groupe": g, "N": int(m.sum()),
                         "Churn réel": y[m].mean(), "Taux d'alerte": pred[m].mean(),
                         "Recall": recall_score(y[m], pred[m], zero_division=0),
                         "Faux positifs (taux)": ((pred[m] == 1) & (y[m] == 0)).sum()
                                                 / max((y[m] == 0).sum(), 1)})
    return pd.DataFrame(rows)


def counterfactual_shift(model, preprocess, X_test, cols=SENSITIVE):
    """Variation moyenne de probabilité si on ne change QUE la variable sensible."""
    base = model.predict_proba(preprocess.transform(X_test))[:, 1]
    out = []
    for col in cols:
        for val in X_test[col].unique():
            Xc = X_test.copy(); Xc[col] = val
            p = model.predict_proba(preprocess.transform(Xc))[:, 1]
            out.append({"Variable": col, "Valeur imposée": val,
                        "Variation moyenne |Δproba|": float(np.abs(p - base).mean())})
    return out