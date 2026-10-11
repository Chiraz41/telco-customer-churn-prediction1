import numpy as np
import pandas as pd


def make_fake_telco(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Jeu de données synthétique avec les mêmes colonnes que Telco."""
    rng = np.random.default_rng(seed)
    yn = ["Yes", "No"]
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(18, 118, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2])
    logit = -1 + 1.5 * (contract == "Month-to-month") - 0.03 * tenure + 0.01 * monthly
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    total = (tenure * monthly).astype(object)
    total[rng.random(n) < 0.01] = " "      # espaces comme dans le vrai dataset
    return pd.DataFrame({
        "customerID": [f"C{i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(yn, n), "Dependents": rng.choice(yn, n),
        "tenure": tenure, "PhoneService": rng.choice(yn, n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": contract, "PaperlessBilling": rng.choice(yn, n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)",
                                     "Credit card (automatic)"], n),
        "MonthlyCharges": monthly, "TotalCharges": total,
        "Churn": np.where(churn, "Yes", "No"),
    })
