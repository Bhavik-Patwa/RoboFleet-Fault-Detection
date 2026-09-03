import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_ROOT = PROJECT_ROOT / "data" / "alfa" / "curated"
METADATA_ROOT = PROJECT_ROOT / "data" / "alfa" / "metadata"

WINDOW_DATASET_PATH = CURATED_ROOT / "telemetry_windows.parquet"
WINDOW_MANIFEST_PATH = CURATED_ROOT / "telemetry_windows_manifest.json"
TRAINING_FLIGHT_REFERENCE_PATH = METADATA_ROOT / "training_flight_reference.csv"
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

# Defining multiple hyperparameter configurations for model selection
MODEL_CONFIGURATIONS = [
    {
        "name": "lgbm_n200_lr0p03_leaves7_minchild20_l2_1_colsample0p8",
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.8
    },
    {
        "name": "lgbm_n300_lr0p03_leaves7_minchild30_l2_5_colsample0p8",    # Stronger regularization
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_child_samples": 30,
        "reg_lambda": 5.0,
        "colsample_bytree": 0.8
    },
    {
        "name": "lgbm_n200_lr0p05_leaves15_minchild20_l2_5_colsample0p8",   # Medium tree
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 20,
        "reg_lambda": 5.0,
        "colsample_bytree": 0.8
    },
    {
        "name": "lgbm_n400_lr0p02_leaves15_minchild30_l2_10_colsample0p8",  # Stronger medium
        "n_estimators": 400,
        "learning_rate": 0.02,
        "num_leaves": 15,
        "min_child_samples": 30,
        "reg_lambda": 10.0,
        "colsample_bytree": 0.8
    }
]


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

    # Using separate inner folds for model selection and threshold calibration
    parser.add_argument(
        "--inner-fold-count",
        type = int,
        default = 4
    )

    return parser.parse_args()


# Verifying that all required input files exist before proceeding
def ensure_inputs_exist() -> None:
    required_paths = [
        WINDOW_DATASET_PATH,
        WINDOW_MANIFEST_PATH,
        TRAINING_FLIGHT_REFERENCE_PATH
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


# Extracting unique flight labels from labeled windows for stratified splitting
def build_flight_labels(labeled_windows: pd.DataFrame) -> pd.DataFrame:
    flight_labels = (
        labeled_windows[["flight_name", "target"]]
        .drop_duplicates()
        .sort_values("flight_name")
        .reset_index(drop = True)
    )

    # Verifying that no flight has conflicting classification labels across windows
    if flight_labels["flight_name"].duplicated().any():
        raise RuntimeError("A flight has conflicting fault-state labels.")

    return flight_labels


# Validating that each class has sufficient flights for the requested fold count
def validate_split_count(flight_labels: pd.DataFrame, fold_count: int) -> None:
    if fold_count < 2:
        raise ValueError("Fold count must be at least 2.")

    # Counting how many flights belong to each target class
    class_flight_counts = flight_labels["target"].value_counts()

    # Validating that each class has enough flights to support the requested fold count
    if class_flight_counts.min() < fold_count:
        raise RuntimeError("Each class must contain at least as many flights as the fold count.")


# Loading and preparing training data from parquet, manifest and flight reference
def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[str]]:
    window_dataset = pd.read_parquet(WINDOW_DATASET_PATH)
    manifest = json.loads(WINDOW_MANIFEST_PATH.read_text())
    flight_reference = pd.read_csv(TRAINING_FLIGHT_REFERENCE_PATH)

    # Identifying feature columns by excluding metadata columns
    feature_columns = [
        column_name
        for column_name in window_dataset.columns
        if column_name not in METADATA_COLUMNS
    ]

    # Validating that the dataset contains feature columns
    if not feature_columns:
        raise RuntimeError("Telemetry window dataset does not contain feature columns.")

    # Validating dataset integrity against manifest counts
    if len(window_dataset) != manifest["window_count"]:
        raise RuntimeError("Telemetry window count does not match the manifest.")

    if len(feature_columns) != manifest["feature_column_count"]:
        raise RuntimeError("Feature column count does not match the manifest.")

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

    return (
        window_dataset,
        labeled_windows,
        flight_reference,
        manifest,
        feature_columns
    )


# Comprehensive validation of training data integrity and suitability
def validate_training_data(window_dataset: pd.DataFrame, labeled_windows: pd.DataFrame,
                           flight_reference: pd.DataFrame, feature_columns: list[str],
                           fold_count: int, inner_fold_count: int
) -> None:
    # Ensuring all expected columns are present in the dataset
    required_columns = METADATA_COLUMNS.union(feature_columns)

    missing_columns = required_columns.difference(window_dataset.columns)

    if missing_columns:
        raise RuntimeError(
            f"Telemetry window dataset is missing columns : {sorted(missing_columns)}"
        )

    # Verifying flight reference contains required columns
    required_flight_columns = {
        "flight_name",
        "first_fault_signal_time"
    }

    missing_flight_columns = required_flight_columns.difference(
        flight_reference.columns
    )

    if missing_flight_columns:
        raise RuntimeError(
            "Training flight reference is missing columns : "
            f"{sorted(missing_flight_columns)}"
        )

    # Validating that features contain no missing values
    if window_dataset[feature_columns].isna().any().any():
        raise RuntimeError("Telemetry windows contain missing feature values.")

    feature_values = window_dataset[feature_columns].to_numpy(dtype = "float64")

    # Validating that all feature values are finite numbers
    if not np.isfinite(feature_values).all():
        raise RuntimeError("Telemetry windows contain non-finite feature values.")

    # Ensuring both classes are present for binary classification
    if labeled_windows["target"].nunique() != 2:
        raise RuntimeError("Both normal and fault-state labels are required.")

    # Building flight labels and validating split counts
    flight_labels = build_flight_labels(labeled_windows)

    fault_flights = flight_labels.loc[
        flight_labels["target"].eq(1),
        "flight_name"
    ]

    flight_reference_by_name = flight_reference.set_index("flight_name")

    missing_failure_times = [
        flight_name
        for flight_name in fault_flights
        if (
            flight_name not in flight_reference_by_name.index
            or pd.isna(
                flight_reference_by_name.loc[
                    flight_name,
                    "first_fault_signal_time"
                ]
            )
        )
    ]

    if missing_failure_times:
        raise RuntimeError(
            "Fault flights are missing failure-status timestamps: "
            f"{sorted(missing_failure_times)}"
        )

    validate_split_count(
        flight_labels = flight_labels,
        fold_count = fold_count
    )

    # Estimating normal flight count for calibration sufficiency
    normal_flight_count = int(
        flight_labels.loc[
            flight_labels["target"].eq(0),
            "flight_name"
        ].nunique()
    )

    # Verifying sufficient normal flights for calibration
    maximum_test_normal_flight_count = math.ceil(
        normal_flight_count / fold_count
    )

    minimum_training_normal_flight_count = (
        normal_flight_count - maximum_test_normal_flight_count
    )

    if minimum_training_normal_flight_count < inner_fold_count:
        raise RuntimeError(
            "Outer training partitions do not contain enough normal flights for the requested calibration fold count."
        )


# Building the classification pipeline with specified model configuration
def build_classifier(model_configuration: dict) -> Pipeline:
    return Pipeline(
        steps = [
            ("variance_filter", VarianceThreshold()),
            (
                "fault_state_classifier",
                LGBMClassifier(
                    objective = "binary",
                    random_state = RANDOM_STATE,
                    n_jobs = -1,
                    verbosity = -1,
                    n_estimators = model_configuration["n_estimators"],
                    learning_rate = model_configuration["learning_rate"],
                    num_leaves = model_configuration["num_leaves"],
                    min_child_samples = model_configuration["min_child_samples"],
                    reg_lambda = model_configuration["reg_lambda"],
                    colsample_bytree = model_configuration["colsample_bytree"]
                )
            )
        ]
    )


# Calculating flight-level ROC-AUC from window predictions
def calculate_flight_roc_auc(windows: pd.DataFrame, probabilities: np.ndarray) -> float:
    # Aggregating predictions to flight level using median probability per flight
    flight_predictions = pd.DataFrame(
        {
            "flight_name": windows["flight_name"].to_numpy(),
            "target": windows["target"].to_numpy(),
            "fault_probability": probabilities
        }
    ).groupby(
        ["flight_name", "target"],
        as_index = False
    ).agg(
        fault_probability = ("fault_probability", "median")
    )

    return float(
        roc_auc_score(
            flight_predictions["target"],
            flight_predictions["fault_probability"]
        )
    )


# Selecting the best model configuration using inner cross-validation
def select_model_configuration(training_windows: pd.DataFrame, feature_columns: list[str],
                               fold_count: int, outer_fold_number: int
) -> tuple[dict, list[dict]]:
    flight_labels = build_flight_labels(training_windows)

    validate_split_count(
        flight_labels = flight_labels,
        fold_count = fold_count
    )

    # Creating inner stratified folds for configuration selection
    splitter = StratifiedKFold(
        n_splits = fold_count,
        shuffle = True,
        random_state = RANDOM_STATE + outer_fold_number
    )

    configuration_records = []

    # Evaluating each model configuration using inner cross-validation
    for model_configuration in MODEL_CONFIGURATIONS:
        fold_scores = []

        for training_indices, validation_indices in splitter.split(
            flight_labels["flight_name"],
            flight_labels["target"]
        ):
            inner_training_flights = set(
                flight_labels.iloc[training_indices]["flight_name"]
            )

            inner_validation_flights = set(
                flight_labels.iloc[validation_indices]["flight_name"]
            )

            inner_training_windows = training_windows.loc[
                training_windows["flight_name"].isin(inner_training_flights)
            ]

            inner_validation_windows = training_windows.loc[
                training_windows["flight_name"].isin(inner_validation_flights)
            ]

            classifier = build_classifier(model_configuration)

            classifier.fit(
                inner_training_windows[feature_columns],
                inner_training_windows["target"]
            )

            validation_probabilities = classifier.predict_proba(
                inner_validation_windows[feature_columns]
            )[:, 1]

            fold_scores.append(
                calculate_flight_roc_auc(
                    windows = inner_validation_windows,
                    probabilities = validation_probabilities
                )
            )

        # Recording performance metrics for this configuration
        configuration_records.append(
            {
                "model_configuration_name": model_configuration["name"],
                "inner_flight_roc_auc_mean": float(np.mean(fold_scores)),
                "inner_flight_roc_auc_standard_deviation": float(
                    np.std(fold_scores, ddof = 0)
                ),
                "parameters": {
                    key: value
                    for key, value in model_configuration.items()
                    if key != "name"
                }
            }
        )

    # Selecting by mean ROC-AUC, stability and then lower model complexity
    selected_record = max(
        configuration_records,
        key = lambda record: (
            record["inner_flight_roc_auc_mean"],
            -record["inner_flight_roc_auc_standard_deviation"],
            -record["parameters"]["num_leaves"],
            -record["parameters"]["n_estimators"]
        )
    )

    selected_configuration = next(
        configuration
        for configuration in MODEL_CONFIGURATIONS
        if configuration["name"] == selected_record["model_configuration_name"]
    )

    return selected_configuration, configuration_records


# Generating calibration predictions from normal windows using inner cross-validation
def collect_calibration_predictions(training_windows: pd.DataFrame, feature_columns: list[str],
                                    model_configuration: dict, fold_count: int,
                                    random_state_offset: int
) -> pd.DataFrame:
    flight_labels = build_flight_labels(training_windows)

    validate_split_count(
        flight_labels = flight_labels,
        fold_count = fold_count
    )

    # Creating inner stratified folds for calibration
    splitter = StratifiedKFold(
        n_splits = fold_count,
        shuffle = True,
        random_state = RANDOM_STATE + random_state_offset
    )

    calibration_frames = []

    # Iterating through inner folds to collect normal window predictions
    for training_indices, validation_indices in splitter.split(
        flight_labels["flight_name"],
        flight_labels["target"]
    ):
        inner_training_flights = set(
            flight_labels.iloc[training_indices]["flight_name"]
        )

        inner_validation_flights = set(
            flight_labels.iloc[validation_indices]["flight_name"]
        )

        inner_training_windows = training_windows.loc[
            training_windows["flight_name"].isin(inner_training_flights)
        ]

        inner_validation_normal_windows = training_windows.loc[
            training_windows["flight_name"].isin(inner_validation_flights)
            & training_windows["target"].eq(0)
        ]

        classifier = build_classifier(model_configuration)

        classifier.fit(
            inner_training_windows[feature_columns],
            inner_training_windows["target"]
        )

        # Storing flight name and probability for each normal validation window
        calibration_frames.append(
            pd.DataFrame(
                {
                    "flight_name": (
                        inner_validation_normal_windows["flight_name"]
                        .to_numpy()
                    ),
                    "fault_probability": classifier.predict_proba(
                        inner_validation_normal_windows[feature_columns]
                    )[:, 1]
                }
            )
        )

    calibration_predictions = pd.concat(
        calibration_frames,
        ignore_index = True
    )

    expected_normal_flight_count = int(
        flight_labels.loc[
            flight_labels["target"].eq(0),
            "flight_name"
        ].nunique()
    )

    # Verifying that calibration covers every normal training flight
    if (
        calibration_predictions["flight_name"].nunique()
        != expected_normal_flight_count
    ):
        raise RuntimeError(
            "Calibration predictions do not cover every normal training flight."
        )

    return calibration_predictions


# Creating unique thresholds from per-flight maximum probabilities
def build_thresholds(calibration_predictions: pd.DataFrame) -> np.ndarray:
    # Computing maximum probability per normal flight for threshold calibration
    normal_flight_scores = (
        calibration_predictions
        .groupby("flight_name", as_index = False)
        .agg(
            maximum_fault_probability = (
                "fault_probability",
                "max"
            )
        )
    )

    thresholds = np.unique(
        normal_flight_scores["maximum_fault_probability"].to_numpy()
    )

    # Adding a threshold slightly above the maximum for complete coverage
    return np.append(
        thresholds,
        np.nextafter(thresholds[-1], np.inf)
    )


# Evaluating online detection performance across all thresholds
def evaluate_online_detection(scored_windows: pd.DataFrame, flight_labels: pd.DataFrame,
                              failure_times: pd.Series, thresholds: np.ndarray,
                              calibration_predictions: pd.DataFrame, fold_number: int,
                              model_configuration_name: str
) -> list[dict]:
    normal_flights = set(
        flight_labels.loc[
            flight_labels["target"].eq(0),
            "flight_name"
        ]
    )

    fault_flights = set(
        flight_labels.loc[
            flight_labels["target"].eq(1),
            "flight_name"
        ]
    )

    # Computing per-flight maximum probabilities from calibration data
    normal_flight_scores = (
        calibration_predictions
        .groupby("flight_name", as_index = False)
        .agg(
            maximum_fault_probability = (
                "fault_probability",
                "max"
            )
        )
    )

    threshold_records = []

    # Evaluating each threshold independently
    for threshold in thresholds:
        normal_false_alert_count = 0
        pre_failure_false_alert_count = 0
        pre_failure_evaluation_count = 0
        detected_fault_flight_count = 0
        detection_delays = []

        # Counting normal flights with any alert above threshold
        for flight_name in normal_flights:
            flight_windows = scored_windows.loc[
                scored_windows["flight_name"].eq(flight_name)
            ]

            if flight_windows["fault_probability"].ge(threshold).any():
                normal_false_alert_count += 1

        # Evaluating fault flights for detection and pre-failure alerts
        for flight_name in fault_flights:
            flight_windows = scored_windows.loc[
                scored_windows["flight_name"].eq(flight_name)
            ].sort_values("window_end_ns")

            failure_time_ns = int(round(float(failure_times.loc[flight_name])))

            # Windows ending before the recorded failure-status time
            pre_failure_windows = flight_windows.loc[
                flight_windows["window_end_ns"].lt(failure_time_ns)
            ]

            if not pre_failure_windows.empty:
                pre_failure_evaluation_count += 1

                if pre_failure_windows["fault_probability"].ge(threshold).any():
                    pre_failure_false_alert_count += 1

            # Windows ending at or after the recorded failure-status time
            post_failure_windows = flight_windows.loc[
                flight_windows["window_end_ns"].ge(failure_time_ns)
            ]

            detection_windows = post_failure_windows.loc[
                post_failure_windows["fault_probability"].ge(threshold)
            ]

            if not detection_windows.empty:
                first_detection_time_ns = int(
                    detection_windows["window_end_ns"].iloc[0]
                )

                # Calculating detection delay in seconds
                detection_delays.append(
                    (first_detection_time_ns - failure_time_ns) / 1e9
                )

                detected_fault_flight_count += 1

        threshold_records.append(
            {
                "fold_number": fold_number,
                "model_configuration_name": model_configuration_name,
                "threshold": float(threshold),
                "calibration_normal_flight_alert_rate": float(
                    np.mean(
                        normal_flight_scores[
                            "maximum_fault_probability"
                        ].ge(threshold)
                    )
                ),
                "normal_flight_count": len(normal_flights),
                "normal_flight_false_alert_count": normal_false_alert_count,
                "normal_flight_false_alert_rate": float(
                    normal_false_alert_count / len(normal_flights)
                ),
                "fault_flight_count": len(fault_flights),
                "fault_flight_detection_count": detected_fault_flight_count,
                "fault_flight_detection_rate": float(
                    detected_fault_flight_count / len(fault_flights)
                ),
                "fault_pre_failure_evaluation_count": (
                    pre_failure_evaluation_count
                ),
                "fault_pre_failure_false_alert_count": (
                    pre_failure_false_alert_count
                ),
                "fault_pre_failure_false_alert_rate": (
                    float(
                        pre_failure_false_alert_count /
                        pre_failure_evaluation_count
                    )
                    if pre_failure_evaluation_count > 0
                    else None
                ),
                "mean_detection_delay_seconds": (
                    float(np.mean(detection_delays))
                    if detection_delays
                    else None
                ),
                "maximum_detection_delay_seconds": (
                    float(np.max(detection_delays))
                    if detection_delays
                    else None
                )
            }
        )

    return threshold_records


# Evaluating model performance using grouped cross-validation by flight
def evaluate_grouped_folds(window_dataset: pd.DataFrame, labeled_windows: pd.DataFrame,
                           flight_reference: pd.DataFrame, feature_columns: list[str],
                           fold_count: int, inner_fold_count: int
) -> tuple[list[dict], list[dict], list[dict]]:
    flight_labels = build_flight_labels(labeled_windows)

    # Initializing stratified k-fold splitter to preserve class distribution across folds
    splitter = StratifiedKFold(
        n_splits = fold_count,
        shuffle = True,
        random_state = RANDOM_STATE
    )

    failure_times = flight_reference.set_index("flight_name")[
        "first_fault_signal_time"
    ]

    fold_records = []
    threshold_records = []
    selection_records = []

    # Iterating through each fold to train and evaluate the model
    for fold_number, (training_indices, test_indices) in enumerate(
        splitter.split(
            flight_labels["flight_name"],
            flight_labels["target"]
        ),
        start = 1
    ):
        # Extracting flight names for training and testing partitions
        training_flights = set(
            flight_labels.iloc[training_indices]["flight_name"]
        )

        test_flights = set(
            flight_labels.iloc[test_indices]["flight_name"]
        )

        # Filtering windows belonging to training flights
        training_windows = labeled_windows.loc[
            labeled_windows["flight_name"].isin(training_flights)
        ]

        # Filtering windows belonging to test flights
        test_labeled_windows = labeled_windows.loc[
            labeled_windows["flight_name"].isin(test_flights)
        ]

        # Getting all windows for test flights including unlabeled ones
        test_windows = window_dataset.loc[
            window_dataset["flight_name"].isin(test_flights)
        ].copy()

        # Selecting the best model configuration using inner cross-validation
        selected_configuration, inner_selection_records = (
            select_model_configuration(
                training_windows = training_windows,
                feature_columns = feature_columns,
                fold_count = inner_fold_count,
                outer_fold_number = fold_number
            )
        )

        # Recording inner selection metrics for this outer fold
        for record in inner_selection_records:
            selection_records.append(
                {
                    "outer_fold_number": fold_number,
                    **record
                }
            )

        # Building and training classifier with selected configuration
        classifier = build_classifier(selected_configuration)

        classifier.fit(
            training_windows[feature_columns],
            training_windows["target"]
        )

        # Generating probability predictions for test windows
        test_probabilities = classifier.predict_proba(
            test_labeled_windows[feature_columns]
        )[:, 1]

        # Computing window-level ROC-AUC score
        window_roc_auc = roc_auc_score(
            test_labeled_windows["target"],
            test_probabilities
        )

        # Computing flight-level ROC-AUC score
        flight_roc_auc = calculate_flight_roc_auc(
            windows = test_labeled_windows,
            probabilities = test_probabilities
        )

        # Collecting calibration predictions from training data
        calibration_predictions = collect_calibration_predictions(
            training_windows = training_windows,
            feature_columns = feature_columns,
            model_configuration = selected_configuration,
            fold_count = inner_fold_count,
            random_state_offset = fold_number
        )

        # Scoring all windows for test flights for online detection evaluation
        test_windows["fault_probability"] = classifier.predict_proba(
            test_windows[feature_columns]
        )[:, 1]

        test_flight_labels = flight_labels.loc[
            flight_labels["flight_name"].isin(test_flights)
        ]

        # Evaluating online detection performance across all thresholds
        threshold_records.extend(
            evaluate_online_detection(
                scored_windows = test_windows,
                flight_labels = test_flight_labels,
                failure_times = failure_times,
                thresholds = build_thresholds(calibration_predictions),
                calibration_predictions = calibration_predictions,
                fold_number = fold_number,
                model_configuration_name = selected_configuration["name"]
            )
        )

        # Storing comprehensive fold metrics for later analysis
        fold_records.append(
            {
                "fold_number": fold_number,
                "model_configuration_name": selected_configuration["name"],
                "training_flight_count": int(len(training_flights)),
                "test_flight_count": int(len(test_flights)),
                "test_normal_flight_count": int(
                    test_flight_labels.loc[
                        test_flight_labels["target"].eq(0),
                        "flight_name"
                    ].nunique()
                ),
                "test_fault_flight_count": int(
                    test_flight_labels.loc[
                        test_flight_labels["target"].eq(1),
                        "flight_name"
                    ].nunique()
                ),
                "window_roc_auc": float(window_roc_auc),
                "flight_roc_auc": float(flight_roc_auc),
                "calibration_normal_flight_count": int(
                    calibration_predictions["flight_name"].nunique()
                ),
                "calibration_normal_window_count": int(
                    len(calibration_predictions)
                )
            }
        )

    return fold_records, threshold_records, selection_records


def main() -> None:
    args = parse_args()

    ensure_inputs_exist()

    (
        window_dataset,
        labeled_windows,
        flight_reference,
        manifest,
        feature_columns
    ) = load_training_data()

    validate_training_data(
        window_dataset = window_dataset,
        labeled_windows = labeled_windows,
        flight_reference = flight_reference,
        feature_columns = feature_columns,
        fold_count = args.fold_count,
        inner_fold_count = args.inner_fold_count
    )

    (
        fold_records,
        threshold_records,
        selection_records
    ) = evaluate_grouped_folds(
        window_dataset = window_dataset,
        labeled_windows = labeled_windows,
        flight_reference = flight_reference,
        feature_columns = feature_columns,
        fold_count = args.fold_count,
        inner_fold_count = args.inner_fold_count
    )

    fold_metrics = pd.DataFrame(fold_records)

    # Counting how often each configuration was selected across outer folds
    outer_configuration_selection_counts = (
        fold_metrics["model_configuration_name"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # Performing final model selection using full dataset and inner cross-validation
    (
        final_configuration,
        final_configuration_selection_metrics
    ) = select_model_configuration(
        training_windows = labeled_windows,
        feature_columns = feature_columns,
        fold_count = args.inner_fold_count,
        outer_fold_number = 0
    )

    # Fitting the final model on all labeled windows
    final_classifier = build_classifier(final_configuration)

    final_classifier.fit(
        labeled_windows[feature_columns],
        labeled_windows["target"]
    )

    # Counting features retained after variance threshold filtering
    retained_feature_count = int(
        final_classifier.named_steps["variance_filter"].get_support().sum()
    )

    # Creating input example and inferring model signature for MLflow
    input_example = labeled_windows[feature_columns].head(5)
    input_example_probabilities = final_classifier.predict_proba(
        input_example
    )

    model_signature = infer_signature(
        input_example,
        input_example_probabilities
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
                "inner_fold_count": args.inner_fold_count,
                "final_model_configuration": final_configuration["name"],
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
                "training_flight_reference_sha256": calculate_file_sha256(
                    TRAINING_FLIGHT_REFERENCE_PATH
                ),
                "model_configurations": MODEL_CONFIGURATIONS,
                "outer_configuration_selection_counts": (
                    outer_configuration_selection_counts
                ),
                "final_configuration_selection_metrics": (
                    final_configuration_selection_metrics
                ),
                "fold_metrics": fold_records,
                "inner_configuration_metrics": selection_records,
                "online_detection_threshold_metrics": threshold_records
            },
            "evaluation_metrics.json"
        )

        # Saving the trained model with skops serialization format and signature
        mlflow.sklearn.log_model(
            sk_model = final_classifier,
            name = "model",
            signature = model_signature,
            input_example = input_example,
            serialization_format = mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types = [
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier"
            ],
            pyfunc_predict_fn = "predict_proba"
        )

        print(f"MLflow run ID : {run.info.run_id}")
        print(f"Retained feature columns : {retained_feature_count}")
        print(
            "Nested flight-level ROC-AUC mean : "
            f"{fold_metrics['flight_roc_auc'].mean():.4f}"
        )
        print(
            "Final model configuration : "
            f"{final_configuration['name']}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fault-state classifier training failed : {exc}")
        sys.exit(1)