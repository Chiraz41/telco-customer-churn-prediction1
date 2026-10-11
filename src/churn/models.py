"""Définition des 5 modèles et de leurs espaces d'hyperparamètres."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def get_models(scale_pos_weight: float) -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


PARAM_DISTRIBUTIONS = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear"]},
    "Decision Tree": {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 5, 10],
    },
    "KNN": {
        "n_neighbors": [5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
}
