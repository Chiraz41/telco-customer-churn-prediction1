# 📊 Telco Customer Churn — Prédiction du départ des clients

Projet de classification binaire suivant la méthodologie **CRISP-DM** : identifier les clients d'un
opérateur télécom susceptibles de résilier leur abonnement, choisir un seuil de décision en fonction
du **coût métier** des erreurs, et exposer le modèle via une **API FastAPI** et une interface **Streamlit**.

- **Données** : [Telco Customer Churn](https://www.openml.org/d/42178) (OpenML, 7 043 clients, 19 variables explicatives), téléchargées automatiquement.
- **Cible** : `Churn` (1 = le client part, 0 = il reste) — ≈ 26,5 % de churn (classes déséquilibrées).
- **Modèles comparés** : Régression logistique, Arbre de décision, KNN, Random Forest, XGBoost
  (`RandomizedSearchCV`, validation croisée stratifiée à 5 plis, critère AUC-ROC).

## Résultats (jeu de test, 20 %)

| Modèle | AUC-ROC | Recall (seuil 0,5) | F1 |
|---|---|---|---|
| **XGBoost** (retenu) | **0,848** | 0,802 | 0,635 |
| Logistic Regression | 0,841 | 0,783 | 0,615 |
| Random Forest | 0,839 | 0,719 | 0,627 |
| Decision Tree | 0,806 | 0,735 | 0,603 |
| KNN | 0,805 | 0,572 | 0,569 |

Avec les coûts supposés **FN = 200** et **FP = 20** (hypothèses de travail, pas des coûts réels),
le seuil minimisant le coût total parmi ceux testés est **0,20** (≈ 14 300 unités) :
363 des 374 départs réels détectés, au prix de 605 fausses alertes. Le seuil est marqué
`provisional` dans `artifacts/metadata.json` et doit être revu si les coûts changent.

> Les chiffres ci-dessus proviennent de l'exécution du notebook d'origine ; ils peuvent varier légèrement
> selon les versions des bibliothèques.

## Structure du projet

```
telco-churn/
├── notebooks/Telco_Customer_Churn.ipynb   # Analyse CRISP-DM complète (phases 1 à 5)
├── src/churn/                             # Code réutilisable
│   ├── config.py                          #   constantes (seed, coûts, seuils…)
│   ├── data.py                            #   chargement + préparation + split
│   ├── features.py                        #   pipeline de prétraitement, schéma UI
│   ├── models.py                          #   5 modèles + espaces d'hyperparamètres
│   ├── evaluate.py                        #   métriques, seuils, coût, figures
│   └── train.py                           #   entraînement + sauvegarde des artefacts
├── api/main.py                            # API FastAPI (/health, /predict)
├── app/streamlit_app.py                   # Interface web Streamlit
├── tests/                                 # Tests pytest (données synthétiques)
├── artifacts/                             # Modèle, prétraitement, métadonnées (générés)
├── reports/                               # Métriques CSV et figures (générés)
├── requirements.txt / requirements-dev.txt
└── .vscode/                               # Réglages, extensions, configs de lancement
```

## Installation

Prérequis : Python ≥ 3.10 et Git.

```bash
git clone <url-du-depot>
cd telco-churn

python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

pip install -r requirements-dev.txt   # ou requirements.txt si vous n'utilisez pas le notebook
pip install -e .                      # rend le package `churn` importable
```

Dans VS Code : *Ctrl+Shift+P → Python: Select Interpreter → `.venv`*, puis ouvrez le notebook
dans `notebooks/` (extension Jupyter recommandée).

## Utilisation

**1. Entraîner et sauvegarder le modèle** (nécessite un accès Internet pour OpenML)

```bash
python -m churn.train
# ou avec un CSV local :
python -m churn.train --data-path data/telco.csv
```

Génère `artifacts/` (`churn_model.joblib`, `preprocess.joblib`, `metadata.json`, `ui_schema.json`)
et `reports/` (métriques CSV + courbes ROC, Precision-Recall, matrices de confusion, coût métier).

**2. Lancer l'API**

```bash
uvicorn api.main:app --reload
```

Documentation interactive : <http://127.0.0.1:8000/docs>

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"data": {
  "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
  "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
  "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35, "TotalCharges": 140.7}}'
```

**3. Lancer l'interface** (l'API doit tourner)

```bash
streamlit run app/streamlit_app.py
```

**4. Tests**

```bash
pytest
```

Les tests utilisent un jeu de données synthétique : aucun accès à OpenML n'est nécessaire.

Dans VS Code, le menu *Run and Debug* propose aussi les trois configurations
« Entraîner les modèles », « API FastAPI » et « Interface Streamlit ».

## Choix méthodologiques

- Prétraitement ajusté **uniquement sur le jeu d'entraînement** (pas de fuite de données) :
  médiane + standardisation pour le numérique, valeur la plus fréquente + One-Hot pour le catégoriel.
- Déséquilibre des classes géré par `class_weight="balanced"` / `scale_pos_weight`.
- Découpage stratifié 80/20, `random_state=42`.

## Limites et pistes d'amélioration

- Les coûts FN/FP sont des **hypothèses** ; le seuil optimal en dépend directement.
- Le seuil est choisi sur le jeu de test ; pour un usage réel, utiliser un jeu de validation distinct.
- Le notebook annonce des analyses **SHAP**, **LIME** et d'**équité** (`gender`, `SeniorCitizen`)
  qui ne sont pas encore implémentées : ce sont les prochaines étapes naturelles.
- Ajouter le suivi d'expériences (MLflow) et une CI de déploiement.
