import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_ROOT = PROJECT_ROOT / "data" / "alfa" / "curated"

WINDOW_DATASET_PATH = CURATED_ROOT / "telemetry_windows.parquet"
WINDOW_MANIFEST_PATH = CURATED_ROOT / "telemetry_windows_manifest.json"
TRACKING_DATABASE_PATH = PROJECT_ROOT / "mlflow.db"

METADATA_COLUMNS = {
    "flight_name",
    "window_start_ns",
    "window_end_ns",
    "window_duration_seconds",
    "window_label",
    "fault_family_label"
}

RANDOM_STATE = 42


# Parsing command-line arguments for experiment configuration
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-name",
        default = "telemetry_fault_state_classification"
    )

    parser.add_argument(
        "--run-name",
        default = None
    )

    parser.add_argument(
        "--fold-count",
        type = int,
        default = 5
    )

    return parser.parse_args()


# Verifying that all required input files exist before proceeding
def ensure_inputs_exist() -> None:
    required_paths = [
        WINDOW_DATASET_PATH,
        WINDOW_MANIFEST_PATH
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required input paths not found : \n{missing_text}")


# Computing SHA256 hash of a file for data integrity tracking
def calculate_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# Loading and preparing training data from parquet and manifest files
def load_training_data() -> tuple[pd.DataFrame, dict, list[str]]:
    window_dataset = pd.read_parquet(WINDOW_DATASET_PATH)
    manifest = json.loads(WINDOW_MANIFEST_PATH.read_text())

    # Identifying feature columns by excluding metadata columns
    feature_columns = [
        column_name
        for column_name in window_dataset.columns
        if column_name not in METADATA_COLUMNS
    ]

    # Filtering only labeled windows for supervised learning
    labeled_windows = window_dataset.loc[
        window_dataset["window_label"].isin(["normal", "fault_state"])
    ].copy()

    if labeled_windows.empty:
        raise RuntimeError("No labeled telemetry windows are available.")

    # Creating binary target column from window labels
    labeled_windows["target"] = labeled_windows["window_label"].eq(
        "fault_state"
    ).astype(int)

    # Validating that features contain no missing values
    if labeled_windows[feature_columns].isna().any().any():
        raise RuntimeError("Labeled telemetry windows contain missing feature values.")

    feature_values = labeled_windows[feature_columns].to_numpy(dtype = "float64")

    # Validating that all feature values are finite numbers
    if not np.isfinite(feature_values).all():
        raise RuntimeError("Labeled telemetry windows contain non-finite feature values.")

    # Ensuring both classes are present for binary classification
    if labeled_windows["target"].nunique() != 2:
        raise RuntimeError("Both normal and fault-state labels are required.")

    return labeled_windows, manifest, feature_columns


# Building the classification pipeline with variance filter and LightGBM
def build_classifier() -> Pipeline:
    return Pipeline(
        steps = [
            ("variance_filter", VarianceThreshold()),
            (
                "fault_state_classifier",
                LGBMClassifier(
                    objective = "binary",
                    random_state = RANDOM_STATE,
                    n_jobs = -1,
                    verbosity = -1
                )
            )
        ]
    )


# Evaluating model performance using grouped cross-validation by flight
def evaluate_grouped_folds(labeled_windows: pd.DataFrame, feature_columns: list[str], fold_count: int) -> list[dict]:
    # Extracting unique flight-level labels with their target classifications
    flight_labels = (
        labeled_windows[["flight_name", "target"]]
        .drop_duplicates()
        .sort_values("flight_name")
        .reset_index(drop = True)
    )

    # Verifying that no flight has conflicting classification labels across windows
    if flight_labels["flight_name"].duplicated().any():
        raise RuntimeError("A flight has conflicting classification labels.")

    # Ensuring the fold count is sufficient for meaningful cross-validation
    if fold_count < 2:
        raise ValueError("Fold count must be at least 2.")

    # Counting how many flights belong to each target class
    class_flight_counts = flight_labels["target"].value_counts()

    # Validating that each class has enough flights to support the requested fold count
    if class_flight_counts.min() < fold_count:
        raise RuntimeError(
            "Each class must contain at least as many flights as the fold count."
        )

    # Initializing stratified k-fold splitter to preserve class distribution across folds
    splitter = StratifiedKFold(
        n_splits = fold_count,
        shuffle = True,
        random_state = RANDOM_STATE
    )

    fold_records = []

    # Iterating through each fold to train and evaluate the model
    for fold_number, (training_flight_indices, test_flight_indices) in enumerate(
        splitter.split(
            flight_labels["flight_name"],
            flight_labels["target"]
        ),
        start = 1
    ):
        # Extracting flight names for training and testing partitions
        training_flights = set(
            flight_labels.iloc[training_flight_indices]["flight_name"]
        )

        test_flights = set(
            flight_labels.iloc[test_flight_indices]["flight_name"]
        )

        # Ensuring no flight leaks between training and test sets
        if training_flights.intersection(test_flights):
            raise RuntimeError(
                "A flight is present in both training and test partitions."
            )

        # Filtering windows belonging to training flights
        training_windows = labeled_windows.loc[
            labeled_windows["flight_name"].isin(training_flights)
        ]

        # Filtering windows belonging to test flights
        test_windows = labeled_windows.loc[
            labeled_windows["flight_name"].isin(test_flights)
        ]

        # Building and training a fresh classifier for this fold
        classifier = build_classifier()

        classifier.fit(
            training_windows[feature_columns],
            training_windows["target"]
        )

        # Generating probability predictions for test windows
        test_probabilities = classifier.predict_proba(
            test_windows[feature_columns]
        )[:, 1]

        # Computing window-level ROC-AUC score
        window_roc_auc = roc_auc_score(
            test_windows["target"],
            test_probabilities
        )

        # Aggregating predictions to flight level using median probability per flight
        flight_predictions = (
            pd.DataFrame(
                {
                    "flight_name": test_windows["flight_name"].to_numpy(),
                    "target": test_windows["target"].to_numpy(),
                    "fault_probability": test_probabilities
                }
            )
            .groupby(
                ["flight_name", "target"],
                as_index = False
            )
            .agg(
                fault_probability = ("fault_probability", "median")
            )
        )

        # Computing flight-level ROC-AUC score
        flight_roc_auc = roc_auc_score(
            flight_predictions["target"],
            flight_predictions["fault_probability"]
        )

        # Storing comprehensive fold metrics for later analysis
        fold_records.append(
            {
                "fold_number": fold_number,
                "training_flight_count": int(len(training_flights)),
                "test_flight_count": int(len(test_flights)),
                "test_normal_flight_count": int(
                    flight_predictions.loc[
                        flight_predictions["target"].eq(0),
                        "flight_name"
                    ].nunique()
                ),
                "test_fault_flight_count": int(
                    flight_predictions.loc[
                        flight_predictions["target"].eq(1),
                        "flight_name"
                    ].nunique()
                ),
                "window_roc_auc": float(window_roc_auc),
                "flight_roc_auc": float(flight_roc_auc)
            }
        )

    return fold_records


def main() -> None:
    args = parse_args()

    ensure_inputs_exist()

    labeled_windows, manifest, feature_columns = load_training_data()

    fold_records = evaluate_grouped_folds(
        labeled_windows = labeled_windows,
        feature_columns = feature_columns,
        fold_count = args.fold_count
    )

    fold_metrics = pd.DataFrame(fold_records)

    # Training final model on all labeled data
    final_classifier = build_classifier()

    final_classifier.fit(
        labeled_windows[feature_columns],
        labeled_windows["target"]
    )

    # Counting features retained after variance threshold filtering
    retained_feature_count = int(
        final_classifier.named_steps["variance_filter"].get_support().sum()
    )

    # Configuring MLflow tracking to use SQLite database
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DATABASE_PATH}")
    mlflow.set_experiment(args.experiment_name)

    # Logging all parameters, metrics and model artifacts to MLflow
    with mlflow.start_run(run_name = args.run_name) as run:
        mlflow.log_params(
            {
                "model_type": "LightGBMClassifier",
                "random_state": RANDOM_STATE,
                "fold_count": args.fold_count,
                "source_topic_count": manifest["source_topic_count"],
                "source_column_count": manifest["source_column_count"],
                "window_feature_count": len(feature_columns),
                "retained_feature_count": retained_feature_count,
                "window_duration_seconds": manifest["window_duration_seconds"],
                "window_step_seconds": manifest["window_step_seconds"],
                "normal_flight_count": int(
                    labeled_windows.loc[
                        labeled_windows["target"].eq(0),
                        "flight_name"
                    ].nunique()
                ),
                "fault_flight_count": int(
                    labeled_windows.loc[
                        labeled_windows["target"].eq(1),
                        "flight_name"
                    ].nunique()
                )
            }
        )

        # Logging aggregated evaluation metrics
        mlflow.log_metrics(
            {
                "window_roc_auc_mean": float(
                    fold_metrics["window_roc_auc"].mean()
                ),
                "window_roc_auc_standard_deviation": float(
                    fold_metrics["window_roc_auc"].std(ddof = 0)
                ),
                "flight_roc_auc_mean": float(
                    fold_metrics["flight_roc_auc"].mean()
                ),
                "flight_roc_auc_standard_deviation": float(
                    fold_metrics["flight_roc_auc"].std(ddof = 0)
                )
            }
        )

        # Logging manifest and evaluation metrics as JSON artifacts
        mlflow.log_dict(manifest, "telemetry_windows_manifest.json")

        mlflow.log_dict(
            {
                "dataset_sha256": calculate_file_sha256(WINDOW_DATASET_PATH),
                "fold_metrics": fold_records
            },
            "evaluation_metrics.json"
        )

        # Saving the trained model with skops serialization format
        mlflow.sklearn.log_model(
            sk_model = final_classifier,
            name = "model",
            serialization_format = mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types = [
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier"
            ]
        )

        print(f"MLflow run ID : {run.info.run_id}")
        print(f"Retained feature columns : {retained_feature_count}")
        print(
            "Flight-level ROC-AUC mean : "
            f"{fold_metrics['flight_roc_auc'].mean():.4f}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fault-state classifier training failed : {exc}")
        sys.exit(1)