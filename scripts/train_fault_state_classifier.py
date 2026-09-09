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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_ROOT = PROJECT_ROOT / "data" / "alfa" / "curated"
METADATA_ROOT = PROJECT_ROOT / "data" / "alfa" / "metadata"

WINDOW_DATASET_PATH = (
    CURATED_ROOT / "telemetry_windows_v2.parquet"
)

WINDOW_MANIFEST_PATH = (
    CURATED_ROOT / "telemetry_windows_v2_manifest.json"
)

TRAINING_FLIGHT_REFERENCE_PATH = (
    METADATA_ROOT / "training_flight_reference.csv"
)

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
RECORDING_DATE_PATTERN = (
    r"^carbonZ_(\d{4}-\d{2}-\d{2})-"
)


# Defining multiple hyperparameter configurations for model comparison
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
        "name": (
            "lgbm_n200_lr0p03_leaves7_"
            "minchild20_l2_1_colsample0p8"
        ),
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
        "name": (
            "lgbm_n300_lr0p03_leaves7_"
            "minchild30_l2_5_colsample0p8"
        ),
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
        "name": (
            "lgbm_n200_lr0p05_leaves15_"
            "minchild20_l2_5_colsample0p8"
        ),
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
        "name": (
            "lgbm_n400_lr0p02_leaves15_"
            "minchild30_l2_10_colsample0p8"
        ),
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


# Fixing the primary model so feature-set comparisons measure representation changes
PRIMARY_MODEL_CONFIGURATION_NAME = (
    "lgbm_n200_lr0p03_leaves7_"
    "minchild20_l2_1_colsample0p8"
)


# Retrieving one declared model configuration by its unique name
def get_model_configuration(configuration_name: str) -> dict:
    matching_configurations = [
        configuration
        for configuration in MODEL_CONFIGURATIONS
        if configuration["name"] == configuration_name
    ]

    if len(matching_configurations) != 1:
        raise RuntimeError(f"Expected exactly one model configuration named : {configuration_name}")

    return matching_configurations[0]


# Parsing command-line arguments for experiment configuration
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-name",
        default = (
            "telemetry_fault_state_classification"
        )
    )

    parser.add_argument(
        "--run-name",
        default = None
    )

    parser.add_argument(
        "--feature-set",
        choices = [
            "raw",
            "dynamic_only",
            "command_response_only",
            "initial_baseline_only",
            "trailing_history_only",
            "dynamic_plus_initial_baseline",
            "dynamic_plus_command_response",
            "relative_dynamic",
            "combined"
        ],
        required = True
    )

    parser.add_argument(
        "--target-normal-flight-alert-rate",
        type = float,
        required = True
    )

    return parser.parse_args()


# Verifying that all required input files exist before proceeding
def ensure_inputs_exist() -> None:
    required_paths = [
        WINDOW_DATASET_PATH,
        WINDOW_MANIFEST_PATH,
        TRAINING_FLIGHT_REFERENCE_PATH
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        missing_text = "\n".join(
            str(path)
            for path in missing_paths
        )

        raise FileNotFoundError(f"Required input paths not found : \n{missing_text}")


# Computing the SHA256 digest of one input file
def calculate_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


# Extracting the recording date used as a conservative evaluation group
def extract_recording_dates(flight_names: pd.Series) -> pd.Series:
    recording_dates = (
        flight_names
        .astype(str)
        .str.extract(
            RECORDING_DATE_PATTERN,
            expand = False
        )
    )

    if recording_dates.isna().any():
        invalid_flights = sorted(
            flight_names.loc[
                recording_dates.isna()
            ]
            .astype(str)
            .unique()
        )

        raise RuntimeError(f"Recording dates could not be extracted from flights : {invalid_flights}")

    return recording_dates


# Extracting one label and one evaluation group for each flight
def build_flight_labels(labeled_windows: pd.DataFrame) -> pd.DataFrame:
    flight_labels = (
        labeled_windows[
            [
                "flight_name",
                "target"
            ]
        ]
        .drop_duplicates()
        .sort_values("flight_name")
        .reset_index(drop = True)
    )

    if flight_labels[
        "flight_name"
    ].duplicated().any():
        raise RuntimeError("A flight has conflicting fault-state labels.")

    flight_labels["recording_date"] = (
        extract_recording_dates(
            flight_labels["flight_name"]
        )
    )

    return flight_labels


# Building leave-one-recording-date-out splits and validating every partition
def build_recording_date_splits(flight_labels: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    recording_date_count = int(
        flight_labels[
            "recording_date"
        ].nunique()
    )

    if recording_date_count < 2:
        raise RuntimeError("At least two recording dates are required.")

    splitter = LeaveOneGroupOut()

    grouped_splits = list(
        splitter.split(
            flight_labels["flight_name"],
            flight_labels["target"],
            groups = flight_labels[
                "recording_date"
            ]
        )
    )

    for (
        training_indices,
        test_indices
    ) in grouped_splits:
        training_labels = flight_labels.iloc[
            training_indices
        ]

        test_labels = flight_labels.iloc[
            test_indices
        ]

        held_out_recording_dates = (
            test_labels[
                "recording_date"
            ].unique()
        )

        if len(held_out_recording_dates) != 1:
            raise RuntimeError("A test fold contains more than one recording date.")

        if training_labels[
            "target"
        ].nunique() != 2:
            raise RuntimeError("A grouped training fold does not contain both classes.")

        if test_labels[
            "target"
        ].nunique() != 2:
            raise RuntimeError("A grouped test fold does not contain both classes.")

    return grouped_splits


# Selecting non-overlapping feature groups for controlled representation comparisons
def select_feature_set(all_feature_columns: list[str], feature_set: str) -> list[str]:
    command_response_marker = (
        "_command_response_error__"
    )

    initial_baseline_marker = (
        "__initial_baseline_delta"
    )

    trailing_history_marker = (
        "__trailing_median_delta"
    )

    engineered_markers = (
        command_response_marker,
        initial_baseline_marker,
        trailing_history_marker
    )

    # Keeping original telemetry statistics without engineered derivatives
    raw_feature_columns = [
        column_name
        for column_name in all_feature_columns
        if not any(
            marker in column_name
            for marker in engineered_markers
        )
    ]

    # Keeping short-window variation while excluding command-response features
    dynamic_feature_columns = [
        column_name
        for column_name in raw_feature_columns
        if (
            column_name.endswith(
                "__standard_deviation"
            )
            or column_name.endswith(
                "__change"
            )
        )
    ]

    # Keeping every direct and relative command-response error feature
    command_response_feature_columns = [
        column_name
        for column_name in all_feature_columns
        if command_response_marker
        in column_name
    ]

    # Excluding command-response derivatives so ablation groups remain separate
    initial_baseline_feature_columns = [
        column_name
        for column_name in all_feature_columns
        if (
            column_name.endswith(
                initial_baseline_marker
            )
            and command_response_marker
            not in column_name
        )
    ]

    trailing_history_feature_columns = [
        column_name
        for column_name in all_feature_columns
        if (
            column_name.endswith(
                trailing_history_marker
            )
            and command_response_marker
            not in column_name
        )
    ]

    relative_dynamic_names = set(
        dynamic_feature_columns
        + command_response_feature_columns
        + initial_baseline_feature_columns
        + trailing_history_feature_columns
    )

    # Combining only feature groups supported by the ablation evidence
    dynamic_plus_initial_baseline_names = set(
        dynamic_feature_columns
        + initial_baseline_feature_columns
    )

    dynamic_plus_command_response_names = set(
        dynamic_feature_columns
        + command_response_feature_columns
    )

    feature_sets = {
        "raw": raw_feature_columns,
        "dynamic_only": dynamic_feature_columns,
        "command_response_only": (
            command_response_feature_columns
        ),
        "initial_baseline_only": (
            initial_baseline_feature_columns
        ),
        "trailing_history_only": (
            trailing_history_feature_columns
        ),
        "dynamic_plus_initial_baseline": [
            column_name
            for column_name in all_feature_columns
            if column_name
            in dynamic_plus_initial_baseline_names
        ],
        "dynamic_plus_command_response": [
            column_name
            for column_name in all_feature_columns
            if column_name
            in dynamic_plus_command_response_names
        ],
        "relative_dynamic": [
            column_name
            for column_name in all_feature_columns
            if column_name
            in relative_dynamic_names
        ],
        "combined": all_feature_columns
    }

    selected_columns = feature_sets.get(
        feature_set
    )

    if selected_columns is None:
        raise ValueError(f"Unsupported feature set : {feature_set}")

    if not selected_columns:
        raise RuntimeError(f"Feature set '{feature_set}' does not contain columns.")

    return selected_columns


# Loading and preparing training data from parquet, manifest and flight reference
def load_training_data(feature_set: str
) -> tuple[
    pd.DataFrame, 
    pd.DataFrame, 
    pd.DataFrame, 
    dict, 
    list[str]
]:
    window_dataset = pd.read_parquet(
        WINDOW_DATASET_PATH
    )

    manifest = json.loads(
        WINDOW_MANIFEST_PATH.read_text()
    )

    flight_reference = pd.read_csv(
        TRAINING_FLIGHT_REFERENCE_PATH
    )

    # Identifying feature columns by excluding metadata columns
    all_feature_columns = [
        column_name
        for column_name
        in window_dataset.columns
        if column_name
        not in METADATA_COLUMNS
    ]

    feature_columns = select_feature_set(
        all_feature_columns = (
            all_feature_columns
        ),
        feature_set = feature_set
    )

    if not feature_columns:
        raise RuntimeError("Telemetry window dataset does not contain feature columns.")

    if len(window_dataset) != manifest[
        "window_count"
    ]:
        raise RuntimeError("Telemetry window count does not match the manifest.")

    if len(all_feature_columns) != manifest[
        "feature_column_count"
    ]:
        raise RuntimeError("Total feature column count does not match the manifest.")

    # Filtering only labeled windows for supervised learning
    labeled_windows = window_dataset.loc[
        window_dataset[
            "window_label"
        ].isin(
            [
                "normal",
                "fault_state"
            ]
        )
    ].copy()

    if labeled_windows.empty:
        raise RuntimeError("No labeled telemetry windows are available.")

    # Creating the binary fault-state target from window labels
    labeled_windows["target"] = (
        labeled_windows[
            "window_label"
        ]
        .eq("fault_state")
        .astype(int)
    )

    return (
        window_dataset,
        labeled_windows,
        flight_reference,
        manifest,
        feature_columns
    )


# Validating training data integrity and grouped-evaluation suitability
def validate_training_data(window_dataset: pd.DataFrame, labeled_windows: pd.DataFrame, 
                           flight_reference: pd.DataFrame, feature_columns: list[str]
) -> None:
    required_columns = (
        METADATA_COLUMNS.union(
            feature_columns
        )
    )

    missing_columns = (
        required_columns.difference(
            window_dataset.columns
        )
    )

    if missing_columns:
        raise RuntimeError(f"Telemetry window dataset is missing columns : {sorted(missing_columns)}"
        )

    required_flight_columns = {
        "flight_name",
        "first_fault_signal_time"
    }

    missing_flight_columns = (
        required_flight_columns.difference(
            flight_reference.columns
        )
    )

    if missing_flight_columns:
        raise RuntimeError(f"Training flight reference is missing columns : {sorted(missing_flight_columns)}")

    if window_dataset[
        feature_columns
    ].isna().any().any():
        raise RuntimeError("Telemetry windows contain missing feature values.")

    feature_values = window_dataset[
        feature_columns
    ].to_numpy(
        dtype = "float64"
    )

    if not np.isfinite(
        feature_values
    ).all():
        raise RuntimeError("Telemetry windows contain non-finite feature values.")

    if labeled_windows[
        "target"
    ].nunique() != 2:
        raise RuntimeError("Both normal and fault-state labels are required.")

    flight_labels = build_flight_labels(
        labeled_windows
    )

    fault_flights = flight_labels.loc[
        flight_labels["target"].eq(1),
        "flight_name"
    ]

    if flight_reference[
        "flight_name"
    ].duplicated().any():
        duplicated_flights = sorted(
            flight_reference.loc[
                flight_reference[
                    "flight_name"
                ].duplicated(
                    keep = False
                ),
                "flight_name"
            ]
            .astype(str)
            .unique()
        )

        raise RuntimeError(f"Training flight reference contains duplicate flights : {duplicated_flights}")

    flight_reference_by_name = (
        flight_reference.set_index(
            "flight_name"
        )
    )

    missing_failure_times = [
        str(flight_name)
        for flight_name in fault_flights
        if (
            flight_name
            not in flight_reference_by_name.index
            or pd.isna(
                flight_reference_by_name.loc[
                    flight_name,
                    "first_fault_signal_time"
                ]
            )
        )
    ]

    if missing_failure_times:
        raise RuntimeError(f"Fault flights are missing failure-status timestamps : {sorted(missing_failure_times)}")

    # Confirming that every recording-date fold contains both classes
    build_recording_date_splits(
        flight_labels
    )


# Building the classification pipeline for one declared configuration
def build_classifier(model_configuration: dict) -> Pipeline:
    steps = [
        (
            "variance_filter",
            VarianceThreshold()
        )
    ]

    if (
        model_configuration["model_type"]
        == "logistic_regression"
    ):
        steps.extend(
            [
                (
                    "feature_scaler",
                    StandardScaler()
                ),
                (
                    "fault_state_classifier",
                    LogisticRegression(
                        C = model_configuration[
                            "C"
                        ],
                        solver = "lbfgs",
                        max_iter = 5000,
                        random_state = (
                            RANDOM_STATE
                        )
                    )
                )
            ]
        )

    elif (
        model_configuration["model_type"]
        == "lightgbm"
    ):
        steps.append(
            (
                "fault_state_classifier",
                LGBMClassifier(
                    objective = "binary",
                    random_state = RANDOM_STATE,
                    n_jobs = -1,
                    verbosity = -1,
                    n_estimators = (
                        model_configuration[
                            "n_estimators"
                        ]
                    ),
                    learning_rate = (
                        model_configuration[
                            "learning_rate"
                        ]
                    ),
                    num_leaves = (
                        model_configuration[
                            "num_leaves"
                        ]
                    ),
                    min_child_samples = (
                        model_configuration[
                            "min_child_samples"
                        ]
                    ),
                    reg_lambda = (
                        model_configuration[
                            "reg_lambda"
                        ]
                    ),
                    colsample_bytree = (
                        model_configuration[
                            "colsample_bytree"
                        ]
                    )
                )
            )
        )

    else:
        raise ValueError(f"Unsupported model type : {model_configuration['model_type']}")

    return Pipeline(
        steps = steps
    )


# Giving each class equal total weight and each flight within a class equal influence
def calculate_flight_balanced_sample_weights(training_windows: pd.DataFrame) -> np.ndarray:
    flight_window_counts = (
        training_windows
        .groupby(
            [
                "target",
                "flight_name"
            ],
            as_index = False
        )
        .size()
        .rename(
            columns = {
                "size": "labeled_window_count"
            }
        )
    )

    # Giving each target class equal total training influence
    flight_window_counts[
        "class_flight_count"
    ] = (
        flight_window_counts
        .groupby("target")[
            "flight_name"
        ]
        .transform("size")
    )

    flight_window_counts[
        "flight_total_weight"
    ] = (
        1.0
        / flight_window_counts[
            "class_flight_count"
        ]
    )

    # Sharing each flight's total weight across its overlapping windows
    flight_window_counts[
        "sample_weight"
    ] = (
        flight_window_counts[
            "flight_total_weight"
        ]
        / flight_window_counts[
            "labeled_window_count"
        ]
    )

    weight_lookup = {
        (
            int(row.target),
            str(row.flight_name)
        ): float(row.sample_weight)
        for row
        in flight_window_counts.itertuples(
            index = False
        )
    }

    sample_weights = np.asarray(
        [
            weight_lookup[
                (
                    int(row.target),
                    str(row.flight_name)
                )
            ]
            for row in training_windows[
                [
                    "target",
                    "flight_name"
                ]
            ].itertuples(
                index = False
            )
        ],
        dtype = "float64"
    )

    if not np.isfinite(
        sample_weights
    ).all():
        raise RuntimeError("Training sample weights contain non-finite values.")

    total_weight = float(
        sample_weights.sum()
    )

    if total_weight <= 0.0:
        raise RuntimeError("Training sample weights must have a positive total.")

    # Preserving relative balancing while keeping the mean weight at one
    sample_weights *= (
        len(sample_weights)
        / total_weight
    )

    return sample_weights


# Fitting either logistic regression or LightGBM with identical weights
def fit_classifier(classifier: Pipeline, training_windows: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    sample_weights = (
        calculate_flight_balanced_sample_weights(
            training_windows
        )
    )

    classifier.fit(
        training_windows[
            feature_columns
        ],
        training_windows["target"],
        fault_state_classifier__sample_weight = (
            sample_weights
        )
    )

    return classifier


# Aggregating window predictions into one score for each flight
def build_flight_prediction_frame(windows: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    flight_predictions = (
        pd.DataFrame(
            {
                "flight_name": windows[
                    "flight_name"
                ].to_numpy(),
                "fault_family_label": windows[
                    "fault_family_label"
                ].to_numpy(),
                "target": windows[
                    "target"
                ].to_numpy(),
                "fault_state_score": (
                    probabilities
                )
            }
        )
        .groupby(
            [
                "flight_name",
                "target"
            ],
            as_index = False
        )
        .agg(
            fault_family_label = (
                "fault_family_label",
                "first"
            ),
            fault_state_score = (
                "fault_state_score",
                "median"
            ),
            labeled_window_count = (
                "fault_state_score",
                "size"
            )
        )
    )

    return flight_predictions


# Calculating flight-level ROC-AUC from aggregated window predictions
def calculate_flight_roc_auc(windows: pd.DataFrame, probabilities: np.ndarray) -> float:
    flight_predictions = (
        build_flight_prediction_frame(
            windows = windows,
            probabilities = probabilities
        )
    )

    return float(
        roc_auc_score(
            flight_predictions["target"],
            flight_predictions[
                "fault_state_score"
            ]
        )
    )


# Comparing declared candidates on the same held-out recording-date fold
def benchmark_candidates_on_outer_fold(training_windows: pd.DataFrame, test_windows: pd.DataFrame,
                                       feature_columns: list[str], fold_number: int,
                                       held_out_recording_date: str
) -> list[dict]:
    benchmark_records = []

    for (
        model_configuration
    ) in MODEL_CONFIGURATIONS:
        classifier = build_classifier(
            model_configuration
        )

        classifier = fit_classifier(
            classifier = classifier,
            training_windows = (
                training_windows
            ),
            feature_columns = (
                feature_columns
            )
        )

        test_probabilities = (
            classifier.predict_proba(
                test_windows[
                    feature_columns
                ]
            )[:, 1]
        )

        benchmark_records.append(
            {
                "fold_number": fold_number,
                "held_out_recording_date": (
                    held_out_recording_date
                ),
                "model_configuration_name": (
                    model_configuration[
                        "name"
                    ]
                ),
                "model_type": (
                    model_configuration[
                        "model_type"
                    ]
                ),
                "window_roc_auc": float(
                    roc_auc_score(
                        test_windows[
                            "target"
                        ],
                        test_probabilities
                    )
                ),
                "flight_roc_auc": (
                    calculate_flight_roc_auc(
                        windows = test_windows,
                        probabilities = (
                            test_probabilities
                        )
                    )
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
            "held_out_recording_date": (
                held_out_recording_date
            ),
            "model_configuration_name": (
                model_configuration_name
            ),
            "flight_name": str(
                row.flight_name
            ),
            "fault_family_label": (
                None
                if pd.isna(
                    row.fault_family_label
                )
                else str(
                    row.fault_family_label
                )
            ),
            "window_start_ns": int(
                row.window_start_ns
            ),
            "window_end_ns": int(
                row.window_end_ns
            ),
            "window_label": str(
                row.window_label
            ),
            "fault_state_score": float(
                row.fault_probability
            )
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
        ].itertuples(
            index = False
        )
    ]


# Returning the pre-declared model so experiments compare feature representations
def get_primary_model_configuration() -> dict:
    return get_model_configuration(
        PRIMARY_MODEL_CONFIGURATION_NAME
    )


# Generating calibration predictions from normal windows using inner validation
def collect_calibration_predictions(training_windows: pd.DataFrame, feature_columns: list[str],
                                    model_configuration: dict
) -> pd.DataFrame:
    flight_labels = build_flight_labels(
        training_windows
    )

    grouped_splits = (
        build_recording_date_splits(
            flight_labels
        )
    )

    calibration_frames = []

    # Producing held-out normal-window predictions for each recording date
    for (
        training_indices,
        validation_indices
    ) in grouped_splits:
        inner_training_flights = set(
            flight_labels.iloc[
                training_indices
            ]["flight_name"]
        )

        inner_validation_flights = set(
            flight_labels.iloc[
                validation_indices
            ]["flight_name"]
        )

        inner_training_windows = (
            training_windows.loc[
                training_windows[
                    "flight_name"
                ].isin(
                    inner_training_flights
                )
            ]
        )

        inner_validation_normal_windows = (
            training_windows.loc[
                training_windows[
                    "flight_name"
                ].isin(
                    inner_validation_flights
                )
                & training_windows[
                    "target"
                ].eq(0)
            ]
        )

        classifier = build_classifier(
            model_configuration
        )

        classifier = fit_classifier(
            classifier = classifier,
            training_windows = (
                inner_training_windows
            ),
            feature_columns = (
                feature_columns
            )
        )

        calibration_frames.append(
            pd.DataFrame(
                {
                    "flight_name": (
                        inner_validation_normal_windows[
                            "flight_name"
                        ].to_numpy()
                    ),
                    "fault_probability": (
                        classifier.predict_proba(
                            inner_validation_normal_windows[
                                feature_columns
                            ]
                        )[:, 1]
                    )
                }
            )
        )

    calibration_predictions = pd.concat(
        calibration_frames,
        ignore_index = True
    )

    expected_normal_flight_count = int(
        flight_labels.loc[
            flight_labels[
                "target"
            ].eq(0),
            "flight_name"
        ].nunique()
    )

    if (
        calibration_predictions[
            "flight_name"
        ].nunique()
        != expected_normal_flight_count
    ):
        raise RuntimeError("Calibration predictions do not cover every normal training flight.")

    return calibration_predictions


# Creating unique thresholds from per-flight maximum probabilities
def build_thresholds(calibration_predictions: pd.DataFrame) -> np.ndarray:
    normal_flight_scores = (
        calibration_predictions
        .groupby(
            "flight_name",
            as_index = False
        )
        .agg(
            maximum_fault_probability = (
                "fault_probability",
                "max"
            )
        )
    )

    thresholds = np.unique(
        normal_flight_scores[
            "maximum_fault_probability"
        ].to_numpy()
    )

    return np.append(
        thresholds,
        np.nextafter(
            thresholds[-1],
            np.inf
        )
    )


# Selecting the most sensitive threshold within the calibration alert budget
def select_operating_threshold(calibration_predictions: pd.DataFrame, target_alert_rate: float) -> float:
    if not 0.0 <= target_alert_rate < 1.0:
        raise ValueError("Target normal-flight alert rate must be between zero and one.")

    normal_flight_scores = (
        calibration_predictions
        .groupby("flight_name")[
            "fault_probability"
        ]
        .max()
        .to_numpy()
    )

    thresholds = build_thresholds(
        calibration_predictions
    )

    eligible_thresholds = [
        threshold
        for threshold in thresholds
        if np.mean(
            normal_flight_scores
            >= threshold
        ) <= target_alert_rate
    ]

    return float(
        eligible_thresholds[0]
    )


# Evaluating online detection performance across all thresholds
def evaluate_online_detection(scored_windows: pd.DataFrame, flight_labels: pd.DataFrame,
                              failure_times: pd.Series, thresholds: np.ndarray,
                              calibration_predictions: pd.DataFrame, fold_number: int,
                              held_out_recording_date: str, model_configuration_name: str
) -> list[dict]:
    normal_flights = set(
        flight_labels.loc[
            flight_labels[
                "target"
            ].eq(0),
            "flight_name"
        ]
    )

    fault_flights = set(
        flight_labels.loc[
            flight_labels[
                "target"
            ].eq(1),
            "flight_name"
        ]
    )

    normal_flight_scores = (
        calibration_predictions
        .groupby(
            "flight_name",
            as_index = False
        )
        .agg(
            maximum_fault_probability = (
                "fault_probability",
                "max"
            )
        )
    )

    threshold_records = []

    for threshold in thresholds:
        normal_false_alert_count = 0
        pre_observed_signal_alert_count = 0
        pre_observed_signal_evaluation_count = 0
        detected_fault_flight_count = 0
        detection_delays = []
        fault_flight_any_alert_count = 0
        first_alert_signed_delays = []

        # Counting normal flights with any alert above the threshold
        for flight_name in normal_flights:
            flight_windows = (
                scored_windows.loc[
                    scored_windows[
                        "flight_name"
                    ].eq(flight_name)
                ]
            )

            if flight_windows[
                "fault_probability"
            ].ge(threshold).any():
                normal_false_alert_count += 1

        # Evaluating fault-flight detection and premature alerts
        for flight_name in fault_flights:
            flight_windows = (
                scored_windows.loc[
                    scored_windows[
                        "flight_name"
                    ].eq(flight_name)
                ]
                .sort_values(
                    "window_end_ns"
                )
            )

            failure_time_ns = int(
                round(
                    float(
                        failure_times.loc[
                            flight_name
                        ]
                    )
                )
            )

            # Finding the first alert anywhere in the fault flight
            all_detection_windows = (
                flight_windows.loc[
                    flight_windows[
                        "fault_probability"
                    ].ge(threshold)
                ]
            )

            if not all_detection_windows.empty:
                first_alert_time_ns = int(
                    all_detection_windows[
                        "window_end_ns"
                    ].iloc[0]
                )

                # Negative values indicate an alert before the observed signal
                first_alert_signed_delays.append(
                    (
                        first_alert_time_ns
                        - failure_time_ns
                    ) / 1e9
                )

                fault_flight_any_alert_count += 1

            # Evaluating alerts before the recorded failure-status signal
            pre_observed_signal_windows = (
                flight_windows.loc[
                    flight_windows[
                        "window_end_ns"
                    ].lt(
                        failure_time_ns
                    )
                ]
            )

            if not pre_observed_signal_windows.empty:
                pre_observed_signal_evaluation_count += 1

                if pre_observed_signal_windows[
                    "fault_probability"
                ].ge(threshold).any():
                    pre_observed_signal_alert_count += 1

            # Evaluating detections at or after the failure-status signal
            post_observed_signal_windows = (
                flight_windows.loc[
                    flight_windows[
                        "window_end_ns"
                    ].ge(
                        failure_time_ns
                    )
                ]
            )

            detection_windows = (
                post_observed_signal_windows.loc[
                    post_observed_signal_windows[
                        "fault_probability"
                    ].ge(threshold)
                ]
            )

            if not detection_windows.empty:
                first_detection_time_ns = int(
                    detection_windows[
                        "window_end_ns"
                    ].iloc[0]
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
                "held_out_recording_date": (
                    held_out_recording_date
                ),
                "model_configuration_name": (
                    model_configuration_name
                ),
                "threshold": float(
                    threshold
                ),
                "calibration_normal_flight_alert_rate": float(
                    np.mean(
                        normal_flight_scores[
                            "maximum_fault_probability"
                        ].ge(threshold)
                    )
                ),
                "normal_flight_count": len(
                    normal_flights
                ),
                "normal_flight_false_alert_count": (
                    normal_false_alert_count
                ),
                "normal_flight_false_alert_rate": float(
                    normal_false_alert_count
                    / len(normal_flights)
                ),
                "fault_flight_count": len(
                    fault_flights
                ),
                "fault_flight_detection_count": (
                    detected_fault_flight_count
                ),
                "fault_flight_detection_rate": float(
                    detected_fault_flight_count
                    / len(fault_flights)
                ),
                "fault_flight_any_alert_count": (
                    fault_flight_any_alert_count
                ),
                "fault_flight_any_alert_rate": float(
                    fault_flight_any_alert_count
                    / len(fault_flights)
                ),
                "mean_first_alert_signed_delay_seconds": (
                    float(
                        np.mean(
                            first_alert_signed_delays
                        )
                    )
                    if first_alert_signed_delays
                    else None
                ),
                "median_first_alert_signed_delay_seconds": (
                    float(
                        np.median(
                            first_alert_signed_delays
                        )
                    )
                    if first_alert_signed_delays
                    else None
                ),
                "minimum_first_alert_signed_delay_seconds": (
                    float(
                        np.min(
                            first_alert_signed_delays
                        )
                    )
                    if first_alert_signed_delays
                    else None
                ),
                "maximum_first_alert_signed_delay_seconds": (
                    float(
                        np.max(
                            first_alert_signed_delays
                        )
                    )
                    if first_alert_signed_delays
                    else None
                ),
                "pre_observed_signal_evaluation_count": (
                    pre_observed_signal_evaluation_count
                ),
                "pre_observed_signal_alert_count": (
                    pre_observed_signal_alert_count
                ),
                "pre_observed_signal_alert_rate": (
                    float(
                        pre_observed_signal_alert_count
                        / pre_observed_signal_evaluation_count
                    )
                    if (
                        pre_observed_signal_evaluation_count
                        > 0
                    )
                    else None
                ),
                "mean_post_observed_signal_detection_delay_seconds": (
                    float(
                        np.mean(
                            detection_delays
                        )
                    )
                    if detection_delays
                    else None
                ),
                "maximum_post_observed_signal_detection_delay_seconds": (
                    float(
                        np.max(
                            detection_delays
                        )
                    )
                    if detection_delays
                    else None
                )
            }
        )

    return threshold_records


# Evaluating fixed models, thresholds and predictions across recording dates
def evaluate_grouped_folds(window_dataset: pd.DataFrame, labeled_windows: pd.DataFrame,
                           flight_reference: pd.DataFrame, feature_columns: list[str],
                           target_normal_flight_alert_rate: float
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict]
]:
    flight_labels = build_flight_labels(
        labeled_windows
    )

    grouped_splits = (
        build_recording_date_splits(
            flight_labels
        )
    )

    failure_times = (
        flight_reference
        .set_index("flight_name")[
            "first_fault_signal_time"
        ]
    )

    fold_records = []
    threshold_records = []
    operating_point_records = []
    candidate_benchmark_records = []
    out_of_fold_flight_prediction_records = []
    out_of_fold_window_prediction_records = []

    for fold_number, (
        training_indices,
        test_indices
    ) in enumerate(
        grouped_splits,
        start = 1
    ):
        held_out_recording_date = str(
            flight_labels.iloc[
                test_indices
            ]["recording_date"].iloc[0]
        )

        training_flights = set(
            flight_labels.iloc[
                training_indices
            ]["flight_name"]
        )

        test_flights = set(
            flight_labels.iloc[
                test_indices
            ]["flight_name"]
        )

        training_windows = (
            labeled_windows.loc[
                labeled_windows[
                    "flight_name"
                ].isin(
                    training_flights
                )
            ]
        )

        test_labeled_windows = (
            labeled_windows.loc[
                labeled_windows[
                    "flight_name"
                ].isin(
                    test_flights
                )
            ]
        )

        # Including unlabeled windows only for online alert evaluation
        test_windows = window_dataset.loc[
            window_dataset[
                "flight_name"
            ].isin(
                test_flights
            )
        ].copy()

        # Recording exploratory candidate comparisons on the held-out date
        candidate_benchmark_records.extend(
            benchmark_candidates_on_outer_fold(
                training_windows = (
                    training_windows
                ),
                test_windows = (
                    test_labeled_windows
                ),
                feature_columns = (
                    feature_columns
                ),
                fold_number = fold_number,
                held_out_recording_date = (
                    held_out_recording_date
                )
            )
        )

        selected_configuration = (
            get_primary_model_configuration()
        )

        classifier = build_classifier(
            selected_configuration
        )

        classifier = fit_classifier(
            classifier = classifier,
            training_windows = (
                training_windows
            ),
            feature_columns = (
                feature_columns
            )
        )

        test_probabilities = (
            classifier.predict_proba(
                test_labeled_windows[
                    feature_columns
                ]
            )[:, 1]
        )

        window_roc_auc = roc_auc_score(
            test_labeled_windows["target"],
            test_probabilities
        )

        flight_prediction_frame = (
            build_flight_prediction_frame(
                windows = test_labeled_windows,
                probabilities = (
                    test_probabilities
                )
            )
        )

        flight_roc_auc = roc_auc_score(
            flight_prediction_frame["target"],
            flight_prediction_frame[
                "fault_state_score"
            ]
        )

        # Recording one held-out prediction for every labeled flight
        for record in (
            flight_prediction_frame.to_dict(
                orient = "records"
            )
        ):
            out_of_fold_flight_prediction_records.append(
                {
                    "fold_number": fold_number,
                    "held_out_recording_date": (
                        held_out_recording_date
                    ),
                    "model_configuration_name": (
                        selected_configuration[
                            "name"
                        ]
                    ),
                    **record
                }
            )

        calibration_predictions = (
            collect_calibration_predictions(
                training_windows = (
                    training_windows
                ),
                feature_columns = (
                    feature_columns
                ),
                model_configuration = (
                    selected_configuration
                )
            )
        )

        # Selecting this fold's threshold using only training-flight calibration
        operating_threshold = (
            select_operating_threshold(
                calibration_predictions = (
                    calibration_predictions
                ),
                target_alert_rate = (
                    target_normal_flight_alert_rate
                )
            )
        )

        # Scoring every window from held-out flights without retraining
        test_windows[
            "fault_probability"
        ] = classifier.predict_proba(
            test_windows[
                feature_columns
            ]
        )[:, 1]

        out_of_fold_window_prediction_records.extend(
            build_window_prediction_records(
                scored_windows = (
                    test_windows
                ),
                fold_number = fold_number,
                held_out_recording_date = (
                    held_out_recording_date
                ),
                model_configuration_name = (
                    selected_configuration[
                        "name"
                    ]
                )
            )
        )

        test_flight_labels = (
            flight_labels.loc[
                flight_labels[
                    "flight_name"
                ].isin(
                    test_flights
                )
            ]
        )

        fold_threshold_records = (
            evaluate_online_detection(
                scored_windows = (
                    test_windows
                ),
                flight_labels = (
                    test_flight_labels
                ),
                failure_times = failure_times,
                thresholds = build_thresholds(
                    calibration_predictions
                ),
                calibration_predictions = (
                    calibration_predictions
                ),
                fold_number = fold_number,
                held_out_recording_date = (
                    held_out_recording_date
                ),
                model_configuration_name = (
                    selected_configuration[
                        "name"
                    ]
                )
            )
        )

        threshold_records.extend(
            fold_threshold_records
        )

        matching_operating_records = [
            record
            for record in fold_threshold_records
            if np.isclose(
                record["threshold"],
                operating_threshold,
                rtol = 0.0,
                atol = 0.0
            )
        ]

        if len(
            matching_operating_records
        ) != 1:
            raise RuntimeError("Exactly one calibrated operating threshold was expected.")

        operating_point_records.append(
            matching_operating_records[0]
        )

        fold_records.append(
            {
                "fold_number": fold_number,
                "model_configuration_name": (
                    selected_configuration[
                        "name"
                    ]
                ),
                "training_flight_count": int(
                    len(training_flights)
                ),
                "test_flight_count": int(
                    len(test_flights)
                ),
                "test_normal_flight_count": int(
                    test_flight_labels.loc[
                        test_flight_labels[
                            "target"
                        ].eq(0),
                        "flight_name"
                    ].nunique()
                ),
                "test_fault_flight_count": int(
                    test_flight_labels.loc[
                        test_flight_labels[
                            "target"
                        ].eq(1),
                        "flight_name"
                    ].nunique()
                ),
                "window_roc_auc": float(
                    window_roc_auc
                ),
                "flight_roc_auc": float(
                    flight_roc_auc
                ),
                "calibration_normal_flight_count": int(
                    calibration_predictions[
                        "flight_name"
                    ].nunique()
                ),
                "calibration_normal_window_count": int(
                    len(
                        calibration_predictions
                    )
                ),
                "held_out_recording_date": (
                    held_out_recording_date
                )
            }
        )

    return (
        fold_records,
        threshold_records,
        operating_point_records,
        candidate_benchmark_records,
        out_of_fold_flight_prediction_records,
        out_of_fold_window_prediction_records
    )


# Training, evaluating and logging one fault-state feature representation
def main() -> None:
    args = parse_args()

    ensure_inputs_exist()

    (
        window_dataset,
        labeled_windows,
        flight_reference,
        manifest,
        feature_columns
    ) = load_training_data(
        feature_set = args.feature_set
    )

    validate_training_data(
        window_dataset = window_dataset,
        labeled_windows = labeled_windows,
        flight_reference = flight_reference,
        feature_columns = feature_columns
    )

    (
        fold_records,
        threshold_records,
        operating_point_records,
        candidate_benchmark_records,
        out_of_fold_flight_prediction_records,
        out_of_fold_window_prediction_records
    ) = evaluate_grouped_folds(
        window_dataset = window_dataset,
        labeled_windows = labeled_windows,
        flight_reference = flight_reference,
        feature_columns = feature_columns,
        target_normal_flight_alert_rate = (
            args.target_normal_flight_alert_rate
        )
    )

    fold_metrics = pd.DataFrame(
        fold_records
    )

    candidate_benchmark_metrics = pd.DataFrame(
        candidate_benchmark_records
    )

    candidate_benchmark_summary = (
        candidate_benchmark_metrics
        .groupby(
            [
                "model_configuration_name",
                "model_type"
            ],
            as_index = False
        )
        .agg(
            outer_window_roc_auc_mean = (
                "window_roc_auc",
                "mean"
            ),
            outer_window_roc_auc_standard_deviation = (
                "window_roc_auc",
                lambda values: float(
                    np.std(
                        values,
                        ddof = 0
                    )
                )
            ),
            outer_flight_roc_auc_mean = (
                "flight_roc_auc",
                "mean"
            ),
            outer_flight_roc_auc_standard_deviation = (
                "flight_roc_auc",
                lambda values: float(
                    np.std(
                        values,
                        ddof = 0
                    )
                )
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
            ascending = [
                False,
                True
            ]
        )
        .reset_index(
            drop = True
        )
    )

    out_of_fold_flight_predictions = pd.DataFrame(
        out_of_fold_flight_prediction_records
    )

    if out_of_fold_flight_predictions[
        "flight_name"
    ].duplicated().any():
        raise RuntimeError("Out-of-fold flight predictions contain duplicate flights.")

    if (
        out_of_fold_flight_predictions[
            "flight_name"
        ].nunique()
        != labeled_windows[
            "flight_name"
        ].nunique()
    ):
        raise RuntimeError("Out-of-fold flight predictions do not cover every labeled flight.")

    pooled_flight_roc_auc = float(
        roc_auc_score(
            out_of_fold_flight_predictions[
                "target"
            ],
            out_of_fold_flight_predictions[
                "fault_state_score"
            ]
        )
    )

    worst_recording_date_flight_roc_auc = float(
        fold_metrics[
            "flight_roc_auc"
        ].min()
    )

    fault_family_score_summary = (
        out_of_fold_flight_predictions.loc[
            out_of_fold_flight_predictions[
                "target"
            ].eq(1)
        ]
        .groupby(
            "fault_family_label",
            as_index = False
        )
        .agg(
            fault_flight_count = (
                "flight_name",
                "nunique"
            ),
            mean_fault_state_score = (
                "fault_state_score",
                "mean"
            ),
            minimum_fault_state_score = (
                "fault_state_score",
                "min"
            ),
            median_fault_state_score = (
                "fault_state_score",
                "median"
            ),
            maximum_fault_state_score = (
                "fault_state_score",
                "max"
            )
        )
        .sort_values(
            "mean_fault_state_score"
        )
        .reset_index(
            drop = True
        )
    )

    # Confirming that the fixed primary model was evaluated in every outer fold
    outer_primary_model_evaluation_counts = (
        fold_metrics[
            "model_configuration_name"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    final_configuration = (
        get_primary_model_configuration()
    )

    final_classifier = build_classifier(
        final_configuration
    )

    final_classifier = fit_classifier(
        classifier = final_classifier,
        training_windows = labeled_windows,
        feature_columns = feature_columns
    )

    # Deriving the full-training operating threshold from grouped predictions
    final_calibration_predictions = (
        collect_calibration_predictions(
            training_windows = (
                labeled_windows
            ),
            feature_columns = (
                feature_columns
            ),
            model_configuration = (
                final_configuration
            )
        )
    )

    final_operating_threshold = (
        select_operating_threshold(
            calibration_predictions = (
                final_calibration_predictions
            ),
            target_alert_rate = (
                args.target_normal_flight_alert_rate
            )
        )
    )

    retained_feature_count = int(
        final_classifier.named_steps[
            "variance_filter"
        ].get_support().sum()
    )

    input_example = labeled_windows[
        feature_columns
    ].head(5)

    input_example_probabilities = (
        final_classifier.predict_proba(
            input_example
        )
    )

    model_signature = infer_signature(
        input_example,
        input_example_probabilities
    )

    mlflow.set_tracking_uri(
        f"sqlite:///{TRACKING_DATABASE_PATH}"
    )

    mlflow.set_experiment(
        args.experiment_name
    )

    with mlflow.start_run(
        run_name = args.run_name
    ) as run:
        mlflow.log_params(
            {
                "model_type": (
                    final_configuration[
                        "model_type"
                    ]
                ),
                "random_state": RANDOM_STATE,
                "validation_strategy": (
                    "leave_one_recording_date_out_"
                    "fixed_primary_model"
                ),
                "recording_date_count": int(
                    build_flight_labels(
                        labeled_windows
                    )[
                        "recording_date"
                    ].nunique()
                ),
                "final_model_configuration": (
                    final_configuration[
                        "name"
                    ]
                ),
                "source_topic_count": manifest[
                    "source_topic_count"
                ],
                "source_column_count": manifest[
                    "source_column_count"
                ],
                "window_feature_count": len(
                    feature_columns
                ),
                "retained_feature_count": (
                    retained_feature_count
                ),
                "window_duration_seconds": manifest[
                    "window_duration_seconds"
                ],
                "window_step_seconds": manifest[
                    "window_step_seconds"
                ],
                "normal_flight_count": int(
                    labeled_windows.loc[
                        labeled_windows[
                            "target"
                        ].eq(0),
                        "flight_name"
                    ].nunique()
                ),
                "fault_flight_count": int(
                    labeled_windows.loc[
                        labeled_windows[
                            "target"
                        ].eq(1),
                        "flight_name"
                    ].nunique()
                ),
                "training_sample_weighting": (
                    "class_balanced_flight_"
                    "balanced_window_weights"
                ),
                "final_operating_threshold": (
                    final_operating_threshold
                ),
                "final_threshold_calibration_normal_flight_count": int(
                    final_calibration_predictions[
                        "flight_name"
                    ].nunique()
                ),
                "fault_family_reweighting": (
                    "not_applied"
                ),
                "feature_set": (
                    args.feature_set
                ),
                "feature_representation": (
                    args.feature_set
                ),
                "target_normal_flight_alert_rate": (
                    args.target_normal_flight_alert_rate
                )
            }
        )

        mlflow.log_metrics(
            {
                "window_roc_auc_mean": float(
                    fold_metrics[
                        "window_roc_auc"
                    ].mean()
                ),
                "window_roc_auc_standard_deviation": float(
                    fold_metrics[
                        "window_roc_auc"
                    ].std(
                        ddof = 0
                    )
                ),
                "flight_roc_auc_mean": float(
                    fold_metrics[
                        "flight_roc_auc"
                    ].mean()
                ),
                "flight_roc_auc_standard_deviation": float(
                    fold_metrics[
                        "flight_roc_auc"
                    ].std(
                        ddof = 0
                    )
                ),
                "pooled_flight_roc_auc": (
                    pooled_flight_roc_auc
                ),
                "worst_recording_date_flight_roc_auc": (
                    worst_recording_date_flight_roc_auc
                )
            }
        )

        mlflow.log_dict(
            manifest,
            "telemetry_windows_v2_manifest.json"
        )

        mlflow.log_dict(
            {
                "dataset_sha256": (
                    calculate_file_sha256(
                        WINDOW_DATASET_PATH
                    )
                ),
                "training_flight_reference_sha256": (
                    calculate_file_sha256(
                        TRAINING_FLIGHT_REFERENCE_PATH
                    )
                ),
                "primary_model_configuration": (
                    final_configuration
                ),
                "operating_point_metrics": (
                    operating_point_records
                ),
                "model_configurations": (
                    MODEL_CONFIGURATIONS
                ),
                "outer_primary_model_evaluation_counts": (
                    outer_primary_model_evaluation_counts
                ),
                "fault_family_score_summary": (
                    fault_family_score_summary.to_dict(
                        orient = "records"
                    )
                ),
                "final_operating_threshold": (
                    final_operating_threshold
                ),
                "final_threshold_calibration_normal_flight_count": int(
                    final_calibration_predictions[
                        "flight_name"
                    ].nunique()
                ),
                "fold_metrics": fold_records,
                "online_detection_threshold_metrics": (
                    threshold_records
                )
            },
            "evaluation_metrics.json"
        )

        mlflow.log_dict(
            {
                "outer_candidate_benchmark_summary": (
                    candidate_benchmark_summary.to_dict(
                        orient = "records"
                    )
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

        mlflow.sklearn.log_model(
            sk_model = final_classifier,
            name = "model",
            signature = model_signature,
            input_example = input_example,
            serialization_format = (
                mlflow.sklearn
                .SERIALIZATION_FORMAT_SKOPS
            ),
            skops_trusted_types = [
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier"
            ],
            pyfunc_predict_fn = "predict_proba"
        )

        print(
            f"MLflow run ID : {run.info.run_id}"
        )

        print(
            "Retained feature columns : "
            f"{retained_feature_count}"
        )

        print(
            "Leave-one-recording-date-out "
            "flight ROC-AUC mean : "
            f"{fold_metrics['flight_roc_auc'].mean():.4f}"
        )

        print(
            "Final operating threshold : "
            f"{final_operating_threshold:.6f}"
        )

        print(
            "Final model configuration : "
            f"{final_configuration['name']}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "Fault-state classifier training failed : "
            f"{exc}"
        )
        sys.exit(1)