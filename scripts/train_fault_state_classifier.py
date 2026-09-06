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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
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
RECORDING_DATE_PATTERN = r"^carbonZ_(\d{4}-\d{2}-\d{2})-"

# Defining multiple hyperparameter configurations for model selection
MODEL_CONFIGURATIONS = [
    {
        "name": "logistic_regression_l2_c0p01",
        "model_type": "logistic_regression",
        "C": 0.01,
        "model_complexity": 0
    },
    {
        "name": "logistic_regression_l2_c0p1",
        "model_type": "logistic_regression",
        "C": 0.1,
        "model_complexity": 0
    },
    {
        "name": "logistic_regression_l2_c1",
        "model_type": "logistic_regression",
        "C": 1.0,
        "model_complexity": 0
    },
    {
        "name": "lgbm_n200_lr0p03_leaves7_minchild20_l2_1_colsample0p8",
        "model_type": "lightgbm",
        "model_complexity": 1,
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.8
    },
    {
        "name": "lgbm_n300_lr0p03_leaves7_minchild30_l2_5_colsample0p8",    # Stronger regularization
        "model_type": "lightgbm",
        "model_complexity": 1,
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_child_samples": 30,
        "reg_lambda": 5.0,
        "colsample_bytree": 0.8
    },
    {
        "name": "lgbm_n200_lr0p05_leaves15_minchild20_l2_5_colsample0p8",   # Medium tree
        "model_type": "lightgbm",
        "model_complexity": 1,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 20,
        "reg_lambda": 5.0,
        "colsample_bytree": 0.8
    },
    {
        "name": "lgbm_n400_lr0p02_leaves15_minchild30_l2_10_colsample0p8",  # Stronger medium
        "model_type": "lightgbm",
        "model_complexity": 1,
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


# Extracting the recording date used as a conservative evaluation group
def extract_recording_dates(flight_names: pd.Series) -> pd.Series:
    recording_dates = flight_names.astype(str).str.extract(
        RECORDING_DATE_PATTERN,
        expand = False
    )

    if recording_dates.isna().any():
        invalid_flights = sorted(
            flight_names.loc[recording_dates.isna()]
            .astype(str)
            .unique()
        )

        raise RuntimeError(
            "Recording dates could not be extracted from flights: "
            f"{invalid_flights}"
        )

    return recording_dates


# Extracting one label and one evaluation group for each flight
def build_flight_labels(
    labeled_windows: pd.DataFrame
) -> pd.DataFrame:
    flight_labels = (
        labeled_windows[["flight_name", "target"]]
        .drop_duplicates()
        .sort_values("flight_name")
        .reset_index(drop = True)
    )

    if flight_labels["flight_name"].duplicated().any():
        raise RuntimeError(
            "A flight has conflicting fault-state labels."
        )

    flight_labels["recording_date"] = extract_recording_dates(
        flight_labels["flight_name"]
    )

    return flight_labels


# Building leave-one-recording-date-out splits and validating every partition
def build_recording_date_splits(
    flight_labels: pd.DataFrame
) -> list[tuple[np.ndarray, np.ndarray]]:
    recording_date_count = int(
        flight_labels["recording_date"].nunique()
    )

    if recording_date_count < 2:
        raise RuntimeError(
            "At least two recording dates are required."
        )

    splitter = LeaveOneGroupOut()

    grouped_splits = list(
        splitter.split(
            flight_labels["flight_name"],
            flight_labels["target"],
            groups = flight_labels["recording_date"]
        )
    )

    for training_indices, test_indices in grouped_splits:
        training_labels = flight_labels.iloc[training_indices]
        test_labels = flight_labels.iloc[test_indices]

        held_out_recording_dates = (
            test_labels["recording_date"].unique()
        )

        if len(held_out_recording_dates) != 1:
            raise RuntimeError(
                "A test fold contains more than one recording date."
            )

        if training_labels["target"].nunique() != 2:
            raise RuntimeError(
                "A grouped training fold does not contain both classes."
            )

        if test_labels["target"].nunique() != 2:
            raise RuntimeError(
                "A grouped test fold does not contain both classes."
            )

    return grouped_splits


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
                           flight_reference: pd.DataFrame, feature_columns: list[str]
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

    # Building flight labels and validating recording-date partitions
    flight_labels = build_flight_labels(labeled_windows)

    fault_flights = flight_labels.loc[
        flight_labels["target"].eq(1),
        "flight_name"
    ]

    if flight_reference["flight_name"].duplicated().any():
        duplicated_flights = sorted(
            flight_reference.loc[
                flight_reference["flight_name"].duplicated(
                    keep = False
                ),
                "flight_name"
            ]
            .astype(str)
            .unique()
        )

        raise RuntimeError(
            "Training flight reference contains duplicate flights: "
            f"{duplicated_flights}"
        )

    flight_reference_by_name = flight_reference.set_index(
        "flight_name"
    )

    missing_failure_times = [
        str(flight_name)
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

    # Confirming that every recording-date fold contains both classes
    build_recording_date_splits(flight_labels)


# Building the classification pipeline with specified model configuration
def build_classifier(model_configuration: dict) -> Pipeline:
    steps = [
        ("variance_filter", VarianceThreshold())
    ]

    if model_configuration["model_type"] == "logistic_regression":
        steps.extend(
            [
                ("feature_scaler", StandardScaler()),
                (
                    "fault_state_classifier",
                    LogisticRegression(
                        C = model_configuration["C"],
                        solver = "lbfgs",
                        max_iter = 5000,
                        random_state = RANDOM_STATE
                    )
                )
            ]
        )

    elif model_configuration["model_type"] == "lightgbm":
        steps.append(
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
        )

    else:
        raise ValueError(
            f"Unsupported model type : {model_configuration['model_type']}"
        )

    return Pipeline(steps = steps)


# Giving each class equal total weight and each flight within a class equal influence
def calculate_flight_balanced_sample_weights(training_windows: pd.DataFrame) -> np.ndarray:
    flight_window_counts = (
        training_windows
        .groupby(
            ["target", "flight_name"],
            as_index = False
        )
        .size()
        .rename(columns = {"size": "labeled_window_count"})
    )

    # Each class receives total weight of one, preventing fault flights from
    # outweighing normal flights simply because there are more of them.
    flight_window_counts["class_flight_count"] = (
        flight_window_counts
        .groupby("target")["flight_name"]
        .transform("size")
    )

    flight_window_counts["flight_total_weight"] = (
        1.0 / flight_window_counts["class_flight_count"]
    )

    # A flight's total weight is shared across its overlapping windows.
    flight_window_counts["sample_weight"] = (
        flight_window_counts["flight_total_weight"] /
        flight_window_counts["labeled_window_count"]
    )

    weight_lookup = {
        (int(row.target), str(row.flight_name)): float(row.sample_weight)
        for row in flight_window_counts.itertuples(index = False)
    }

    sample_weights = np.asarray(
        [
            weight_lookup[(int(row.target), str(row.flight_name))]
            for row in training_windows[
                ["target", "flight_name"]
            ].itertuples(index = False)
        ],
        dtype = "float64"
    )

    if not np.isfinite(sample_weights).all():
        raise RuntimeError("Training sample weights contain non-finite values.")

    total_weight = float(sample_weights.sum())

    if total_weight <= 0.0:
        raise RuntimeError("Training sample weights must have a positive total.")

    # Preserving relative class/flight balancing while keeping mean weight at one
    sample_weights *= len(sample_weights) / total_weight

    return sample_weights


# Fitting either logistic regression or LightGBM with identical flight-balanced weights
def fit_classifier(classifier: Pipeline, training_windows: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    sample_weights = calculate_flight_balanced_sample_weights(
        training_windows
    )

    classifier.fit(
        training_windows[feature_columns],
        training_windows["target"],
        fault_state_classifier__sample_weight = sample_weights
    )

    return classifier


# Aggregating window predictions into one score for each flight
def build_flight_prediction_frame(windows: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    flight_predictions = pd.DataFrame(
        {
            "flight_name": windows["flight_name"].to_numpy(),
            "fault_family_label": (
                windows["fault_family_label"].to_numpy()
            ),
            "target": windows["target"].to_numpy(),
            "fault_state_score": probabilities
        }
    ).groupby(
        ["flight_name", "target"],
        as_index = False
    ).agg(
        fault_family_label = ("fault_family_label", "first"),
        fault_state_score = ("fault_state_score", "median"),
        labeled_window_count = ("fault_state_score", "size")
    )

    return flight_predictions


# Calculating flight-level ROC-AUC from aggregated window predictions
def calculate_flight_roc_auc(
    windows: pd.DataFrame,
    probabilities: np.ndarray
) -> float:
    flight_predictions = build_flight_prediction_frame(
        windows = windows,
        probabilities = probabilities
    )

    return float(
        roc_auc_score(
            flight_predictions["target"],
            flight_predictions["fault_state_score"]
        )
    )


# Benchmarking every fixed candidate against the same untouched flight set
def benchmark_candidates_on_outer_fold(training_windows: pd.DataFrame, test_windows: pd.DataFrame,
                                       feature_columns: list[str], fold_number: int,
                                       held_out_recording_date: str

) -> list[dict]:
    benchmark_records = []

    for model_configuration in MODEL_CONFIGURATIONS:
        classifier = build_classifier(model_configuration)

        classifier = fit_classifier(
            classifier = classifier,
            training_windows = training_windows,
            feature_columns = feature_columns
        )

        test_probabilities = classifier.predict_proba(
            test_windows[feature_columns]
        )[:, 1]

        benchmark_records.append(
            {
                "fold_number": fold_number,
                "held_out_recording_date": held_out_recording_date,
                "model_configuration_name": model_configuration["name"],
                "model_type": model_configuration["model_type"],
                "window_roc_auc": float(
                    roc_auc_score(
                        test_windows["target"],
                        test_probabilities
                    )
                ),
                "flight_roc_auc": calculate_flight_roc_auc(
                    windows = test_windows,
                    probabilities = test_probabilities
                )
            }
        )

    return benchmark_records


# Creating JSON-safe held-out window prediction records
def build_window_prediction_records(scored_windows: pd.DataFrame, fold_number: int,
                                    held_out_recording_date: str, model_configuration_name: str
) -> list[dict]:
    return [
        {
            "fold_number": fold_number,
            "held_out_recording_date": held_out_recording_date,
            "model_configuration_name": model_configuration_name,
            "flight_name": str(row.flight_name),
            "fault_family_label": (
                None
                if pd.isna(row.fault_family_label)
                else str(row.fault_family_label)
            ),
            "window_start_ns": int(row.window_start_ns),
            "window_end_ns": int(row.window_end_ns),
            "window_label": str(row.window_label),
            "fault_state_score": float(row.fault_probability)
        }
        for row in scored_windows[
            [
                "flight_name",
                "fault_family_label",
                "window_start_ns",
                "window_end_ns",
                "window_label",
                "fault_probability"
            ]
        ].itertuples(index = False)
    ]


# Selecting the best model configuration using inner cross-validation
def select_model_configuration(training_windows: pd.DataFrame, feature_columns: list[str]
) -> tuple[dict, list[dict]]:
    flight_labels = build_flight_labels(training_windows)

    grouped_splits = build_recording_date_splits(
        flight_labels
    )

    configuration_records = []

    for model_configuration in MODEL_CONFIGURATIONS:
        fold_scores = []

        for training_indices, validation_indices in grouped_splits:
            inner_training_flights = set(
                flight_labels.iloc[training_indices]["flight_name"]
            )

            inner_validation_flights = set(
                flight_labels.iloc[validation_indices]["flight_name"]
            )

            inner_training_windows = training_windows.loc[
                training_windows["flight_name"].isin(
                    inner_training_flights
                )
            ]

            inner_validation_windows = training_windows.loc[
                training_windows["flight_name"].isin(
                    inner_validation_flights
                )
            ]

            classifier = build_classifier(model_configuration)

            classifier = fit_classifier(
                classifier = classifier,
                training_windows = inner_training_windows,
                feature_columns = feature_columns
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

        # Recording grouped validation performance for this configuration
        configuration_records.append(
            {
                "model_configuration_name": model_configuration["name"],
                "model_type": model_configuration["model_type"],
                "model_complexity": model_configuration["model_complexity"],
                "inner_flight_roc_auc_mean": float(
                    np.mean(fold_scores)
                ),
                "inner_flight_roc_auc_standard_deviation": float(
                    np.std(fold_scores, ddof = 0)
                ),
                "parameters": {
                    key: value
                    for key, value in model_configuration.items()
                    if key not in {"name", "model_complexity"}
                }
            }
        )

    # Selecting by mean ROC-AUC, stability and then lower model complexity
    selected_record = min(
        configuration_records,
        key = lambda record: (
            -record["inner_flight_roc_auc_mean"],
            record["inner_flight_roc_auc_standard_deviation"],
            record["model_complexity"],
            record["parameters"].get("num_leaves", 0),
            record["parameters"].get("n_estimators", 0),
            record["model_configuration_name"]
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
                                    model_configuration: dict
) -> pd.DataFrame:
    flight_labels = build_flight_labels(training_windows)

    grouped_splits = build_recording_date_splits(
        flight_labels
    )

    calibration_frames = []

    # Iterating through grouped splits to collect normal window predictions
    for training_indices, validation_indices in grouped_splits:
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

        classifier = fit_classifier(
            classifier = classifier,
            training_windows = inner_training_windows,
            feature_columns = feature_columns
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
                              held_out_recording_date: str, model_configuration_name: str
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
        pre_observed_signal_alert_count = 0
        pre_observed_signal_evaluation_count = 0
        detected_fault_flight_count = 0
        detection_delays = []

        # Counting normal flights with any alert above threshold
        for flight_name in normal_flights:
            flight_windows = scored_windows.loc[
                scored_windows["flight_name"].eq(flight_name)
            ]

            if flight_windows["fault_probability"].ge(threshold).any():
                normal_false_alert_count += 1

        # Evaluating fault flights for detection and pre-observed-signal alerts
        for flight_name in fault_flights:
            flight_windows = scored_windows.loc[
                scored_windows["flight_name"].eq(flight_name)
            ].sort_values("window_end_ns")

            failure_time_ns = int(
                round(float(failure_times.loc[flight_name]))
            )

            # Windows ending before the first recorded failure-status signal
            pre_observed_signal_windows = flight_windows.loc[
                flight_windows["window_end_ns"].lt(
                    failure_time_ns
                )
            ]

            if not pre_observed_signal_windows.empty:
                pre_observed_signal_evaluation_count += 1

                if pre_observed_signal_windows[
                    "fault_probability"
                ].ge(threshold).any():
                    pre_observed_signal_alert_count += 1

            # Windows ending at or after the first recorded failure-status signal
            post_observed_signal_windows = flight_windows.loc[
                flight_windows["window_end_ns"].ge(
                    failure_time_ns
                )
            ]

            detection_windows = post_observed_signal_windows.loc[
                post_observed_signal_windows[
                    "fault_probability"
                ].ge(threshold)
            ]

            if not detection_windows.empty:
                first_detection_time_ns = int(
                    detection_windows["window_end_ns"].iloc[0]
                )

                detection_delays.append(
                    (
                        first_detection_time_ns
                        - failure_time_ns
                    ) / 1e9
                )

                detected_fault_flight_count += 1

        threshold_records.append(
            {
                "fold_number": fold_number,
                "held_out_recording_date": held_out_recording_date,
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
                "pre_observed_signal_evaluation_count": (
                    pre_observed_signal_evaluation_count
                ),
                "pre_observed_signal_alert_count": (
                    pre_observed_signal_alert_count
                ),
                "pre_observed_signal_alert_rate": (
                    float(
                        pre_observed_signal_alert_count /
                        pre_observed_signal_evaluation_count
                    )
                    if pre_observed_signal_evaluation_count > 0
                    else None
                ),
                "mean_post_observed_signal_detection_delay_seconds": (
                    float(np.mean(detection_delays))
                    if detection_delays
                    else None
                ),
                "maximum_post_observed_signal_detection_delay_seconds": (
                    float(np.max(detection_delays))
                    if detection_delays
                    else None
                )
            }
        )

    return threshold_records


# Evaluating model selection, all candidate benchmarks and held-out predictions
def evaluate_grouped_folds(window_dataset: pd.DataFrame, labeled_windows: pd.DataFrame,
                           flight_reference: pd.DataFrame, feature_columns: list[str]
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:

    flight_labels = build_flight_labels(labeled_windows)

    grouped_splits = build_recording_date_splits(
        flight_labels
    )

    failure_times = flight_reference.set_index("flight_name")[
        "first_fault_signal_time"
    ]

    fold_records = []
    threshold_records = []
    selection_records = []
    candidate_benchmark_records = []
    out_of_fold_flight_prediction_records = []
    out_of_fold_window_prediction_records = []

    for fold_number, (training_indices, test_indices) in enumerate(
        grouped_splits,
        start = 1
    ):
        held_out_recording_date = str(
            flight_labels.iloc[test_indices][
                "recording_date"
            ].iloc[0]
        )
        
        training_flights = set(
            flight_labels.iloc[training_indices]["flight_name"]
        )

        test_flights = set(
            flight_labels.iloc[test_indices]["flight_name"]
        )

        training_windows = labeled_windows.loc[
            labeled_windows["flight_name"].isin(training_flights)
        ]

        test_labeled_windows = labeled_windows.loc[
            labeled_windows["flight_name"].isin(test_flights)
        ]

        # Including unlabeled windows for online alert evaluation only
        test_windows = window_dataset.loc[
            window_dataset["flight_name"].isin(test_flights)
        ].copy()

        # Comparing every candidate on this untouched outer test fold
        candidate_benchmark_records.extend(
            benchmark_candidates_on_outer_fold(
                training_windows = training_windows,
                test_windows = test_labeled_windows,
                feature_columns = feature_columns,
                fold_number = fold_number,
                held_out_recording_date = held_out_recording_date
            )
        )

        # Selecting the candidate using only the outer training flights
        selected_configuration, inner_selection_records = (
            select_model_configuration(
                training_windows = training_windows,
                feature_columns = feature_columns
            )
        )

        for record in inner_selection_records:
            selection_records.append(
                {
                    "outer_fold_number": fold_number,
                    "held_out_recording_date": held_out_recording_date,
                    **record
                }
            )

        classifier = build_classifier(selected_configuration)

        classifier = fit_classifier(
            classifier = classifier,
            training_windows = training_windows,
            feature_columns = feature_columns
        )

        test_probabilities = classifier.predict_proba(
            test_labeled_windows[feature_columns]
        )[:, 1]

        window_roc_auc = roc_auc_score(
            test_labeled_windows["target"],
            test_probabilities
        )

        flight_prediction_frame = build_flight_prediction_frame(
            windows = test_labeled_windows,
            probabilities = test_probabilities
        )

        flight_roc_auc = roc_auc_score(
            flight_prediction_frame["target"],
            flight_prediction_frame["fault_state_score"]
        )

        # Recording one truly held-out prediction for every labeled flight
        for record in flight_prediction_frame.to_dict(orient = "records"):
            out_of_fold_flight_prediction_records.append(
                {
                    "fold_number": fold_number,
                    "held_out_recording_date": held_out_recording_date,
                    "model_configuration_name": selected_configuration["name"],
                    **record
                }
            )

        calibration_predictions = collect_calibration_predictions(
            training_windows = training_windows,
            feature_columns = feature_columns,
            model_configuration = selected_configuration
        )

        # Scoring every window from held-out flights without retraining
        test_windows["fault_probability"] = classifier.predict_proba(
            test_windows[feature_columns]
        )[:, 1]

        out_of_fold_window_prediction_records.extend(
            build_window_prediction_records(
                scored_windows = test_windows,
                fold_number = fold_number,
                held_out_recording_date = held_out_recording_date,
                model_configuration_name = selected_configuration["name"]
            )
        )

        test_flight_labels = flight_labels.loc[
            flight_labels["flight_name"].isin(test_flights)
        ]

        threshold_records.extend(
            evaluate_online_detection(
                scored_windows = test_windows,
                flight_labels = test_flight_labels,
                failure_times = failure_times,
                thresholds = build_thresholds(calibration_predictions),
                calibration_predictions = calibration_predictions,
                fold_number = fold_number,
                held_out_recording_date = held_out_recording_date,
                model_configuration_name = selected_configuration["name"]
            )
        )

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
                ),
                "held_out_recording_date": held_out_recording_date
            }
        )

    return (
        fold_records,
        threshold_records,
        selection_records,
        candidate_benchmark_records,
        out_of_fold_flight_prediction_records,
        out_of_fold_window_prediction_records
    )


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
        feature_columns = feature_columns
    )

    (
        fold_records,
        threshold_records,
        selection_records,
        candidate_benchmark_records,
        out_of_fold_flight_prediction_records,
        out_of_fold_window_prediction_records
    ) = evaluate_grouped_folds(
        window_dataset = window_dataset,
        labeled_windows = labeled_windows,
        flight_reference = flight_reference,
        feature_columns = feature_columns
    )

    fold_metrics = pd.DataFrame(fold_records)

    candidate_benchmark_metrics = pd.DataFrame(
        candidate_benchmark_records
    )

    candidate_benchmark_summary = (
        candidate_benchmark_metrics
        .groupby(
            ["model_configuration_name", "model_type"],
            as_index = False
        )
        .agg(
            outer_window_roc_auc_mean = (
                "window_roc_auc",
                "mean"
            ),
            outer_window_roc_auc_standard_deviation = (
                "window_roc_auc",
                lambda values: float(np.std(values, ddof = 0))
            ),
            outer_flight_roc_auc_mean = (
                "flight_roc_auc",
                "mean"
            ),
            outer_flight_roc_auc_standard_deviation = (
                "flight_roc_auc",
                lambda values: float(np.std(values, ddof = 0))
            ),
            outer_flight_roc_auc_minimum = (
                "flight_roc_auc",
                "min"
            ),
            outer_flight_roc_auc_maximum = (
                "flight_roc_auc",
                "max"
            )
        )
        .sort_values(
            [
                "outer_flight_roc_auc_mean",
                "outer_flight_roc_auc_standard_deviation"
            ],
            ascending = [False, True]
        )
        .reset_index(drop = True)
    )

    out_of_fold_flight_predictions = pd.DataFrame(
        out_of_fold_flight_prediction_records
    )

    if out_of_fold_flight_predictions["flight_name"].duplicated().any():
        raise RuntimeError(
            "Out-of-fold flight predictions contain duplicate flights."
        )

    if (
        out_of_fold_flight_predictions["flight_name"].nunique()
        != labeled_windows["flight_name"].nunique()
    ):
        raise RuntimeError(
            "Out-of-fold flight predictions do not cover every labeled flight."
        )

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
        feature_columns = feature_columns
    )

    # Fitting the final model on all labeled windows
    final_classifier = build_classifier(final_configuration)

    final_classifier = fit_classifier(
        classifier = final_classifier,
        training_windows = labeled_windows,
        feature_columns = feature_columns
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
                "model_type": final_configuration["model_type"],
                "random_state": RANDOM_STATE,
                "validation_strategy": (
                    "nested_leave_one_recording_date_out"
                ),
                "recording_date_count": int(
                    build_flight_labels(
                        labeled_windows
                    )["recording_date"].nunique()
                ),
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
                ),
                "training_sample_weighting": (
                    "class_balanced_flight_balanced_window_weights"
                ),
                "fault_family_reweighting": "not_applied"
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

        mlflow.log_dict(
            {
                "outer_candidate_benchmark_summary": (
                    candidate_benchmark_summary.to_dict(orient = "records")
                ),
                "outer_candidate_benchmark_records": (
                    candidate_benchmark_records
                )
            },
            "outer_candidate_benchmarks.json"
        )

        mlflow.log_dict(
            {
                "out_of_fold_flight_predictions": (
                    out_of_fold_flight_prediction_records
                ),
                "out_of_fold_window_predictions": (
                    out_of_fold_window_prediction_records
                )
            },
            "out_of_fold_predictions.json"
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