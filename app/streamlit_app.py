"""Interface Streamlit : streamlit run app/streamlit_app.py"""
import json
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="Telco Churn Prediction", page_icon="📊",
                   layout="wide")
st.title("📊 Telco Customer Churn Prediction")
st.write("Prédiction du risque de résiliation des clients.")

api_url = st.text_input("URL de l'API FastAPI",
                        value="http://127.0.0.1:8000").rstrip("/")

schema_path = Path(__file__).resolve().parents[1] / "artifacts" / "ui_schema.json"
if not schema_path.exists():
    st.error("artifacts/ui_schema.json introuvable. "
             "Lancez d'abord : python -m churn.train")
    st.stop()

schema = json.loads(schema_path.read_text(encoding="utf-8"))

st.subheader("Informations du client")
customer_data = {}
with st.form("customer_form"):
    for col, info in schema.items():
        if info["type"] == "numeric":
            customer_data[col] = st.number_input(col, value=float(info["default"]))
        else:
            options = info.get("options", [])
            customer_data[col] = (st.selectbox(col, options) if options
                                  else st.text_input(col))
    submitted = st.form_submit_button("Prédire le churn")

if submitted:
    try:
        r = requests.post(f"{api_url}/predict",
                          json={"data": customer_data}, timeout=30)
        if r.status_code == 200:
            res = r.json()
            st.subheader("Résultat de la prédiction")
            st.metric("Probabilité de résiliation",
                      f"{res['probabilite_churn']:.2%}")
            if res["prediction"] == 1:
                st.error("⚠️ Risque de résiliation élevé")
            else:
                st.success("✅ Risque de résiliation faible")
            st.caption(f"Modèle : {res['modele']} | "
                       f"Seuil : {res['seuil_decision']:.0%}")
        else:
            st.error(f"Erreur API ({r.status_code}) : {r.text}")
    except requests.RequestException as e:
        st.error(f"Impossible de joindre l'API : {e}")
