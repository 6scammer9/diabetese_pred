import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


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
LOG_FEATURES = ["Insulin", "DiabetesPedigreeFunction", "Age"]
RANDOM_STATE = 42
IMPUTATION_STRATEGIES = ("median", "knn")
BATCH_SIZES = (16, 32, 64)
FEATURE_NAMES = FEATURES + [
    "Insulin_missing",
    "Skin_missing",
    "Glucose_BMI",
    "Glucose_Age",
    "insulin_glucose_interaction",
    "BMI_underweight",
    "BMI_normal",
    "BMI_overweight",
    "BMI_obese",
    "Glucose_below_100",
    "Glucose_100_to_125",
    "Glucose_126_plus",
]


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


class DiabetesPreprocessor:
    def __init__(self, imputation_strategy: str = "median"):
        if imputation_strategy not in IMPUTATION_STRATEGIES:
            raise ValueError(f"Unsupported imputation strategy: {imputation_strategy}")
        self.imputation_strategy = imputation_strategy

    def _prepare(self, features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        frame = features[FEATURES].astype("float64").copy()
        missing_flags = np.column_stack(
            [
                (frame["Insulin"].isna() | frame["Insulin"].eq(0)).to_numpy(),
                (frame["SkinThickness"].isna() | frame["SkinThickness"].eq(0)).to_numpy(),
            ]
        ).astype("float64")
        frame[ZERO_AS_MISSING] = frame[ZERO_AS_MISSING].replace(0, np.nan)

        for column in FEATURES:
            frame[column] = frame[column].clip(
                lower=self.clip_limits_[column][0],
                upper=self.clip_limits_[column][1],
            )
        frame[LOG_FEATURES] = np.log1p(frame[LOG_FEATURES])
        return frame, missing_flags

    def _make_features(self, prepared: np.ndarray, missing_flags: np.ndarray) -> np.ndarray:
        values = pd.DataFrame(prepared, columns=FEATURES)
        glucose = values["Glucose"].to_numpy()
        bmi = values["BMI"].to_numpy()
        insulin = np.expm1(values["Insulin"].to_numpy())
        age = np.expm1(values["Age"].to_numpy())

        bmi_bins = np.column_stack(
            [
                bmi < 18.5,
                (bmi >= 18.5) & (bmi < 25),
                (bmi >= 25) & (bmi < 30),
                bmi >= 30,
            ]
        )
        glucose_bins = np.column_stack(
            [
                glucose < 100,
                (glucose >= 100) & (glucose < 126),
                glucose >= 126,
            ]
        )
        engineered = np.column_stack(
            [
                prepared,
                missing_flags,
                glucose * bmi,
                glucose * age,
                glucose * insulin,
                bmi_bins,
                glucose_bins,
            ]
        ).astype("float64")
        return engineered

    def fit(self, features: pd.DataFrame) -> "DiabetesPreprocessor":
        frame = features[FEATURES].astype("float64").copy()
        frame[ZERO_AS_MISSING] = frame[ZERO_AS_MISSING].replace(0, np.nan)
        self.clip_limits_ = {
            column: (
                float(np.nanpercentile(frame[column], 1)),
                float(np.nanpercentile(frame[column], 99)),
            )
            for column in FEATURES
        }
        prepared, _ = self._prepare(features)

        if self.imputation_strategy == "knn":
            self.distance_scaler_ = StandardScaler().fit(prepared)
            scaled = self.distance_scaler_.transform(prepared)
            self.imputer_ = KNNImputer(n_neighbors=5)
            imputed = self.imputer_.fit_transform(scaled)
            imputed = self.distance_scaler_.inverse_transform(imputed)
        else:
            self.imputer_ = SimpleImputer(strategy="median")
            imputed = self.imputer_.fit_transform(prepared)

        missing_flags = self._missing_flags(features)
        engineered = self._make_features(imputed, missing_flags)
        self.feature_scaler_ = StandardScaler().fit(engineered)
        return self

    @staticmethod
    def _missing_flags(features: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                (features["Insulin"].isna() | features["Insulin"].eq(0)).to_numpy(),
                (
                    features["SkinThickness"].isna()
                    | features["SkinThickness"].eq(0)
                ).to_numpy(),
            ]
        ).astype("float64")

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        prepared, missing_flags = self._prepare(features)
        if self.imputation_strategy == "knn":
            scaled = self.distance_scaler_.transform(prepared)
            imputed = self.imputer_.transform(scaled)
            imputed = self.distance_scaler_.inverse_transform(imputed)
        else:
            imputed = self.imputer_.transform(prepared)
        engineered = self._make_features(imputed, missing_flags)
        return self.feature_scaler_.transform(engineered)

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        return self.fit(features).transform(features)


def make_model(input_size: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def class_weights(labels: pd.Series) -> dict[int, float]:
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def score_model(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype("int32")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def fit_neural_model(
    x_train: np.ndarray,
    y_train: pd.Series,
    x_validation: np.ndarray,
    y_validation: pd.Series,
    batch_size: int,
    max_epochs: int = 150,
) -> tuple[tf.keras.Model, int]:
    tf.keras.backend.clear_session()
    model = make_model(x_train.shape[1])
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=max_epochs,
        batch_size=batch_size,
        class_weight=class_weights(y_train),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    return model, best_epoch


def cross_validate(
    features: pd.DataFrame, target: pd.Series
) -> tuple[list[dict[str, object]], dict[str, object]]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: list[dict[str, object]] = []

    for strategy in IMPUTATION_STRATEGIES:
        logistic_scores = []
        for train_indices, validation_indices in folds.split(features, target):
            x_fold_train = features.iloc[train_indices]
            x_fold_validation = features.iloc[validation_indices]
            y_fold_train = target.iloc[train_indices]
            y_fold_validation = target.iloc[validation_indices]
            preprocessor = DiabetesPreprocessor(strategy).fit(x_fold_train)
            x_fold_train_scaled = preprocessor.transform(x_fold_train)
            x_fold_validation_scaled = preprocessor.transform(x_fold_validation)
            baseline = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            )
            baseline.fit(x_fold_train_scaled, y_fold_train)
            logistic_scores.append(
                score_model(
                    y_fold_validation,
                    baseline.predict_proba(x_fold_validation_scaled)[:, 1],
                )
            )
        results.append(
            {
                "model": "logistic_regression",
                "imputation": strategy,
                "batch_size": None,
                "fold_metrics": logistic_scores,
            }
        )

        for batch_size in BATCH_SIZES:
            fold_scores = []
            for fold_number, (train_indices, validation_indices) in enumerate(
                folds.split(features, target), start=1
            ):
                outer_train_x = features.iloc[train_indices]
                outer_train_y = target.iloc[train_indices]
                x_fit, x_early_stop, y_fit, y_early_stop = train_test_split(
                    outer_train_x,
                    outer_train_y,
                    test_size=0.15,
                    random_state=RANDOM_STATE,
                    stratify=outer_train_y,
                )
                preprocessor = DiabetesPreprocessor(strategy).fit(x_fit)
                x_fit_scaled = preprocessor.transform(x_fit)
                x_early_stop_scaled = preprocessor.transform(x_early_stop)
                x_score_scaled = preprocessor.transform(features.iloc[validation_indices])
                tf.keras.utils.set_random_seed(RANDOM_STATE + fold_number)
                model, _ = fit_neural_model(
                    x_fit_scaled,
                    y_fit,
                    x_early_stop_scaled,
                    y_early_stop,
                    batch_size=batch_size,
                )
                fold_scores.append(
                    score_model(
                        target.iloc[validation_indices],
                        model.predict(x_score_scaled, verbose=0).ravel(),
                    )
                )
                del model
            results.append(
                {
                    "model": "dropout_neural_network",
                    "imputation": strategy,
                    "batch_size": batch_size,
                    "fold_metrics": fold_scores,
                }
            )

    neural_results = [row for row in results if row["model"] == "dropout_neural_network"]
    selected = max(
        neural_results,
        key=lambda row: (
            float(np.mean([metric["f1"] for metric in row["fold_metrics"]])),
            float(np.mean([metric["roc_auc"] for metric in row["fold_metrics"]])),
        ),
    )
    return results, selected


def summarize_cross_validation(results: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc")
    for result in results:
        folds = result["fold_metrics"]
        row = {
            "model": result["model"],
            "imputation": result["imputation"],
            "batch_size": result["batch_size"],
        }
        for name in metric_names:
            values = [fold[name] for fold in folds]
            row[name + "_mean"] = float(np.mean(values))
            row[name + "_std"] = float(np.std(values, ddof=1))
        summary.append(row)
    return summary


def train(data_path: Path, output_dir: Path) -> None:
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    features, target = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    cross_validation, selected = cross_validate(x_train, y_train)
    strategy = str(selected["imputation"])
    batch_size = int(selected["batch_size"])

    x_fit, x_early_stop, y_fit, y_early_stop = train_test_split(
        x_train,
        y_train,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    early_preprocessor = DiabetesPreprocessor(strategy).fit(x_fit)
    model, best_epoch = fit_neural_model(
        early_preprocessor.transform(x_fit),
        y_fit,
        early_preprocessor.transform(x_early_stop),
        y_early_stop,
        batch_size=batch_size,
    )

    final_preprocessor = DiabetesPreprocessor(strategy).fit(x_train)
    x_train_scaled = final_preprocessor.transform(x_train)
    x_test_scaled = final_preprocessor.transform(x_test)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    final_model = make_model(x_train_scaled.shape[1])
    final_model.fit(
        x_train_scaled,
        y_train,
        epochs=best_epoch,
        batch_size=batch_size,
        class_weight=class_weights(y_train),
        verbose=0,
    )

    baseline = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    baseline.fit(x_train_scaled, y_train)
    test_metrics = {
        "logistic_regression": score_model(
            y_test, baseline.predict_proba(x_test_scaled)[:, 1]
        ),
        "dropout_neural_network": score_model(
            y_test, final_model.predict(x_test_scaled, verbose=0).ravel()
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    final_model.save(output_dir / "tabular_nn.keras")
    joblib.dump(final_preprocessor, output_dir / "preprocessor.joblib")
    summary = {
        "rows": int(len(target)),
        "test_rows": int(len(y_test)),
        "cv_folds": 5,
        "cross_validation_threshold": 0.5,
        "selected_neural_network": {
            "imputation": strategy,
            "batch_size": batch_size,
            "epochs_after_early_stopping": best_epoch,
        },
        "cross_validation": summarize_cross_validation(cross_validation),
        "held_out_test_metrics": test_metrics,
        "note": (
            "Metrics are experimental results on a small public dataset, not "
            "evidence of clinical performance. Glucose bins are exploratory; "
            "fasting status is not established by this dataset."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
