import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(csv_path)
    if list(data.columns) != FEATURES + [TARGET]:
        raise ValueError(f"CSV headers must be exactly {FEATURES + [TARGET]}.")
    for column in FEATURES + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data.isna().any().any():
        raise ValueError("The CSV contains non-numeric or missing values.")
    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Outcome must contain only 0 and 1.")
    return data[FEATURES].astype("float64"), data[TARGET].astype("int32")


def replace_invalid_zeros(features: pd.DataFrame) -> pd.DataFrame:
    cleaned = features.copy()
    cleaned[ZERO_AS_MISSING] = cleaned[ZERO_AS_MISSING].replace(0, np.nan)
    return cleaned


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "invalid_zeros",
                FunctionTransformer(replace_invalid_zeros, validate=False),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(C=10, max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def train(data_path: Path, output_dir: Path) -> None:
    features, target = load_dataset(data_path)
    cross_validation = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    results = cross_validate(
        build_model(),
        features,
        target,
        cv=cross_validation,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )

    model = build_model().fit(features, target)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "logistic_model.joblib")

    summary = {
        "rows": int(len(target)),
        "validation": "stratified 5-fold cross-validation",
        "decision_threshold": 0.5,
        "model": "unweighted logistic regression (C=10)",
        "preprocessing": (
            "Invalid zeros become missing; median imputation and scaling are "
            "fitted separately within each training fold."
        ),
        "metrics": {
            metric: {
                "mean": float(np.mean(results["test_" + metric])),
                "std": float(np.std(results["test_" + metric], ddof=1)),
            }
            for metric in scoring
        },
        "note": (
            "This small public dataset is for experimentation, not clinical use. "
            "Higher precision here comes with lower recall than class-weighted training."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
