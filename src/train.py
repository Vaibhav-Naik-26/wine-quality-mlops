import json
import os
import sys

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

TARGET = "target"

MARGIN = 0.05
RANDOM_STATE = 42


def load_dataset():
    df = pd.read_csv(DATA_URL, sep=";")

    if "quality" not in df.columns:
        raise ValueError("Dataset is missing the required quality column.")

    df = df.copy()
    df[TARGET] = (df["quality"] >= 6).astype(int)

    return df


def validate_dataset(df):
    required_columns = FEATURES + [TARGET]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset validation failed. Missing columns: {missing}"
        )

    if df[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Dataset validation failed: missing values found.")

    if len(df) < 100:
        raise ValueError("Dataset validation failed: too few rows.")


def main():
    df = load_dataset()

    print(f"Dataset loaded successfully: {df.shape}")

    validate_dataset(df)

    print("Dataset validation passed.")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Baseline
    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_val)

    baseline_score = f1_score(
        y_val,
        baseline_predictions,
        zero_division=0,
    )

    print(f"Baseline F1 score: {baseline_score:.4f}")

    # Candidate model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    model_score = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    print(f"Candidate F1 score: {model_score:.4f}")

    required_score = baseline_score + MARGIN

    gate_passed = model_score >= required_score

    print(f"Required F1 score: {required_score:.4f}")
    print(f"Quality gate passed: {gate_passed}")

    os.makedirs("artifacts", exist_ok=True)

    metrics = {
        "metric": "f1",
        "baseline_score": round(float(baseline_score), 6),
        "model_score": round(float(model_score), 6),
        "margin": float(MARGIN),
        "required_score": round(float(required_score), 6),
        "gate_passed": bool(gate_passed),
        "validation_size": 0.20,
        "random_state": RANDOM_STATE,
    }

    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if not gate_passed:
        print(
            "QUALITY GATE FAILED: "
            "candidate model did not improve enough over baseline."
        )
        sys.exit(1)

    # Save trained model only after quality gate passes.
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        "artifacts/model.joblib",
    )

    print("Model saved successfully.")


if __name__ == "__main__":
    main()