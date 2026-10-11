"""Constantes du projet (reprises du notebook)."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER_SEARCH = 5

OPENML_DATA_ID = 42178  # Telco Customer Churn
TARGET = "Churn"
ID_COLUMN = "customerID"

# Hypothèses de coût métier (section 1.6 du notebook)
COST_FN = 200
COST_FP = 20
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

FINAL_MODEL_NAME = "XGBoost"
