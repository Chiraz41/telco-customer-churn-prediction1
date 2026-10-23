"""Page Streamlit – Module 5 : Explainable AI (apparaît dans le menu latéral)."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from churn import explain as E

ROOT = Path(__file__).resolve().parents[2]
ART, FIG = ROOT / "artifacts", ROOT / "reports" / "figures"

st.set_page_config(page_title="Explicabilité", page_icon="🔍", layout="wide")
st.title("🔍 Explicabilité du modèle (XAI)")

if not (ART / "test_set.csv").exists():
    st.error("Artefacts XAI introuvables. Relancez : python -m churn.train")
    st.stop()


@st.cache_resource
def load():
    meta = json.loads((ART / "metadata.json").read_text(encoding="utf-8"))
    pre = joblib.load(ART / "preprocess.joblib")
    model = joblib.load(ART / "churn_model.joblib")
    bg = pd.read_csv(ART / "background.csv")
    test = pd.read_csv(ART / "test_set.csv")
    schema = json.loads((ART / "ui_schema.json").read_text(encoding="utf-8"))
    return meta, pre, model, E.to_dense(pre.transform(bg)), test, schema


meta, pre, model, bg_dense, test, schema = load()
threshold = meta["decision_threshold"]
tab_g, tab_l = st.tabs(["🌍 Explications globales", "👤 Explication d'un client"])

with tab_g:
    for title, f in [("Importance propre au modèle", "native_importance.png"),
                     ("Importance par permutation (comparable entre modèles)", "permutation_importance.png"),
                     ("SHAP summary (beeswarm)", "shap_beeswarm.png")]:
        st.subheader(title)
        st.image(str(FIG / f))
    st.subheader("Dépendance SHAP (variables principales)")
    cols = st.columns(2)
    for i, p in enumerate(sorted(FIG.glob("shap_dependence_*.png"))):
        cols[i % 2].image(str(p))
    st.subheader("Équité : variables sensibles")
    st.dataframe(pd.read_csv(ROOT / "reports" / "fairness_by_group.csv").round(3))
    st.dataframe(pd.read_csv(ROOT / "reports" / "counterfactual_sensitive.csv").round(4))

with tab_l:
    mode = st.radio("Source", ["Individu du jeu de test", "Formulaire (individu fictif)"],
                    horizontal=True)
    if mode.startswith("Individu"):
        i = st.number_input("Index dans le jeu de test", 0, len(test) - 1, 0)
        row = test.drop(columns="Churn").iloc[[int(i)]]
        st.caption(f"Churn réel : {'Oui' if test.iloc[int(i)]['Churn'] == 1 else 'Non'}")
    else:
        data = {}
        for col, info in schema.items():
            data[col] = (st.number_input(col, value=float(info["default"]))
                         if info["type"] == "numeric"
                         else st.selectbox(col, info["options"]))
        row = pd.DataFrame([data])

    if st.button("Expliquer"):
        with st.spinner("Calcul SHAP + LIME…"):
            r = E.explain_one(model, pre, row, bg_dense, threshold)
        c1, c2 = st.columns(2)
        c1.metric("Probabilité de départ", f"{r['proba']:.1%}")
        c2.metric("Décision", "⚠️ Risque élevé" if r["decision"] else "✅ Risque faible",
                  help=f"Seuil = {threshold:.0%}")
        st.info(r["sentence"])
        a, b = st.columns(2)
        a.subheader("SHAP – waterfall")
        shap.plots.waterfall(r["shap"], max_display=12, show=False)
        a.pyplot(plt.gcf()); plt.close()
        b.subheader("LIME")
        b.pyplot(r["lime"].as_pyplot_figure(label=1)); plt.close()
        st.subheader("Comparaison SHAP / LIME (par variable d'origine)")
        st.dataframe(r["comparison"])