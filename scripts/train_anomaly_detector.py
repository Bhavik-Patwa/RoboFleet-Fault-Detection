import argparse
import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_ROOT = PROJECT_ROOT / "data" / "alfa" / "curated"

WINDOW_DATASET_PATH = CURATED_ROOT / "telemetry_windows_v3.parquet"
WINDOW_MANIFEST_PATH = CURATED_ROOT / "telemetry_windows_v3_manifest.json"
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

# Comparing two normal-behavior models under the same training and evaluation setup
MODEL_CONFIGURATIONS = [
    {
        "name": "isolation_forest",
        "model_type": "isolation_forest"
    },
    {
        "name": "pca_reconstruction",
        "model_type": "pca_reconstruction",
        "explained_variance": 0.95
    }
]


# Returning anomaly scores where larger values indicate more unusual behavior
class AnomalyScoreEstimator(BaseEstimator):
    def __init__(self, model_type: str, explained_variance: float = 0.95) -> None:
        self.model_type = model_type
        self.explained_variance = explained_variance

    def fit(self, features: pd.DataFrame, target = None) -> "AnomalyScoreEstimator":
        if self.model_type == "isolation_forest":
            self.detector_ = IsolationForest(
                random_state = RANDOM_STATE,
                n_jobs = -1
            )

        elif self.model_type == "pca_reconstruction":
            self.detector_ = PCA(
                n_components = self.explained_variance,
                svd_solver = "full"
            )

        else:
            raise ValueError(f"Unsupported model type : {self.model_type}")

        self.detector_.fit(features)

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        feature_values = np.asarray(features, dtype = "float64")

        if self.model_type == "isolation_forest":
            return -self.detector_.score_samples(feature_values)

        transformed_features = self.detector_.transform(feature_values)
        reconstructed_features = self.detector_.inverse_transform(
            transformed_features
        )

        return np.mean(
            (feature_values - reconstructed_features) ** 2,
            axis = 1
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-name",
        default = "telemetry_anomaly_detection"
    )

    parser.add_argument(
        "--run-name",
        default = "anomaly_detector_benchmark"
    )

    return parser.parse_args()


def ensure_inputs_exist() -> None:
    required_paths = [
        WINDOW_DATASET_PATH,
        WINDOW_MANIFEST_PATH
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required input paths not found : \n{missing_text}")


def load_window_dataset() -> tuple[pd.DataFrame, dict, list[str]]:
    window_dataset = pd.read_parquet(WINDOW_DATASET_PATH)
    manifest = json.loads(WINDOW_MANIFEST_PATH.read_text())

    feature_columns = [
        column_name
        for column_name in window_dataset.columns
        if column_name not in METADATA_COLUMNS
    ]

    if not feature_columns:
        raise RuntimeError("Telemetry window dataset does not contain feature columns.")

    if len(window_dataset) != manifest["window_count"]:
        raise RuntimeError("Telemetry window count does not match the manifest.")

    if len(feature_columns) != manifest["feature_column_count"]:
        raise RuntimeError("Feature column count does not match the manifest.")

    return window_dataset, manifest, feature_columns


def validate_window_dataset(window_dataset: pd.DataFrame, feature_columns: list[str]) -> None:
    required_columns = METADATA_COLUMNS.union(feature_columns)

    missing_columns = required_columns.difference(window_dataset.columns)

    if missing_columns:
        raise RuntimeError(
            f"Telemetry window dataset is missing columns : {sorted(missing_columns)}"
        )

    # Accepting all labels produced by the V3 window dataset
    allowed_labels = {
        "normal",
        "pre_fault_state",
        "fault_state",
        "unlabeled"
    }

    unexpected_labels = set(window_dataset["window_label"]) - allowed_labels

    if unexpected_labels:
        raise RuntimeError(
            f"Unexpected window labels found : {sorted(unexpected_labels)}"
        )
    
    # Including confirmed pre-fault periods when validating labeled features
    labelled_windows = window_dataset.loc[
        window_dataset["window_label"].isin(
            ["normal", "pre_fault_state", "fault_state"]
        )
    ]

    if labelled_windows.empty:
        raise RuntimeError("No labelled telemetry windows are available.")

    if labelled_windows[feature_columns].isna().any().any():
        raise RuntimeError(
            "Labelled telemetry windows contain missing feature values."
        )

    feature_values = labelled_windows[feature_columns].to_numpy(
        dtype = "float64"
    )

    if not np.isfinite(feature_values).all():
        raise RuntimeError(
            "Labelled telemetry windows contain non-finite feature values."
        )

    normal_flight_count = int(
        labelled_windows.loc[
            labelled_windows["window_label"].eq("normal"),
            "flight_name"
        ].nunique()
    )

    fault_flight_count = int(
        labelled_windows.loc[
            labelled_windows["window_label"].eq("fault_state"),
            "flight_name"
        ].nunique()
    )

    if normal_flight_count < 2:
        raise RuntimeError(
            "At least two normal flights are required for evaluation."
        )

    if fault_flight_count == 0:
        raise RuntimeError("No fault-state flights are available for evaluation.")


# Fitting feature filtering, scaling and the anomaly model on normal-state training data
def build_anomaly_pipeline(model_configuration: dict) -> Pipeline:
    return Pipeline(
        steps = [
            ("variance_filter", VarianceThreshold()),
            ("feature_scaler", RobustScaler()),
            (
                "anomaly_detector",
                AnomalyScoreEstimator(
                    model_type = model_configuration["model_type"],
                    explained_variance = model_configuration.get(
                        "explained_variance",
                        0.95
                    )
                )
            )
        ]
    )


# Evaluating normal and fault windows from a completely held-out recording date
def evaluate_leave_one_normal_flight_out(normal_windows: pd.DataFrame, fault_state_windows: pd.DataFrame,
                                         feature_columns: list[str], model_configuration: dict
) -> list[dict]:
    date_pattern = r"^carbonZ_(\d{4}-\d{2}-\d{2})-"

    normal_dates = normal_windows["flight_name"].str.extract(
        date_pattern,
        expand = False
    )

    fault_dates = fault_state_windows["flight_name"].str.extract(
        date_pattern,
        expand = False
    )

    if normal_dates.isna().any() or fault_dates.isna().any():
        raise RuntimeError("Recording dates could not be extracted.")

    if normal_dates.nunique() < 2:
        raise RuntimeError("At least two normal-state recording dates are required.")

    if not set(fault_dates).issubset(set(normal_dates)):
        raise RuntimeError("A fault recording date has no normal-state evaluation data.")

    evaluation_records = []

    for recording_date in sorted(normal_dates.unique()):
        # Excluding the entire evaluation date from model and preprocessing fits
        training_windows = normal_windows.loc[
            normal_dates.ne(recording_date)
        ]

        validation_windows = normal_windows.loc[
            normal_dates.eq(recording_date)
        ]

        test_fault_windows = fault_state_windows.loc[
            fault_dates.eq(recording_date)
        ]

        if test_fault_windows.empty:
            raise RuntimeError("A recording date has no fault-state evaluation windows.")

        pipeline = build_anomaly_pipeline(model_configuration)
        pipeline.fit(training_windows[feature_columns])

        # Each pair contains only flights from the held-out date
        for normal_flight, normal_frame in validation_windows.groupby("flight_name"):
            normal_scores = pipeline.predict(normal_frame[feature_columns])

            for fault_flight, fault_frame in test_fault_windows.groupby("flight_name"):
                fault_scores = pipeline.predict(fault_frame[feature_columns])

                evaluation_records.append(
                    {
                        "held_out_recording_date": str(recording_date),
                        "held_out_normal_flight": str(normal_flight),
                        "fault_state_flight": str(fault_flight),
                        "normal_window_count": int(len(normal_scores)),
                        "fault_state_window_count": int(len(fault_scores)),
                        "roc_auc": float(
                            roc_auc_score(
                                np.concatenate(
                                    [np.zeros(len(normal_scores)), np.ones(len(fault_scores))]
                                ),
                                np.concatenate([normal_scores, fault_scores])
                            )
                        ),
                        "normal_score_median": float(np.median(normal_scores)),
                        "fault_state_score_median": float(np.median(fault_scores))
                    }
                )

    return evaluation_records


def train_and_log_model(normal_windows: pd.DataFrame, fault_state_windows: pd.DataFrame,
                        feature_columns: list[str], manifest: dict, model_configuration: dict
) -> dict:
    evaluation_records = evaluate_leave_one_normal_flight_out(
        normal_windows = normal_windows,
        fault_state_windows = fault_state_windows,
        feature_columns = feature_columns,
        model_configuration = model_configuration
    )

    evaluation_metrics = pd.DataFrame(evaluation_records)

    final_pipeline = build_anomaly_pipeline(model_configuration)

    final_pipeline.fit(normal_windows[feature_columns])

    retained_feature_count = int(
        final_pipeline.named_steps["variance_filter"].get_support().sum()
    )

    input_example = normal_windows[feature_columns].head(5)
    input_example_scores = final_pipeline.predict(input_example)

    model_signature = infer_signature(
        input_example,
        input_example_scores
    )

    with mlflow.start_run(run_name = model_configuration["name"],
                          nested = True
    ) as run:
        mlflow.log_params(
            {
                "model_type": model_configuration["model_type"],
                "explained_variance": model_configuration.get(
                    "explained_variance",
                    "not_applicable"
                ),
                "random_state": RANDOM_STATE,
                "source_topic_count": manifest["source_topic_count"],
                "source_column_count": manifest["source_column_count"],
                "window_feature_count": len(feature_columns),
                "retained_feature_count": retained_feature_count,
                "window_duration_seconds": manifest["window_duration_seconds"],
                "window_step_seconds": manifest["window_step_seconds"],
                "normal_flight_count": int(
                    normal_windows["flight_name"].nunique()
                ),
                "fault_state_flight_count": int(
                    fault_state_windows["flight_name"].nunique()
                )
            }
        )

        mlflow.log_metrics(
            {
                "flight_pair_roc_auc_mean": float(
                    evaluation_metrics["roc_auc"].mean()
                ),
                "flight_pair_roc_auc_standard_deviation": float(
                    evaluation_metrics["roc_auc"].std(ddof = 0)
                ),
                "flight_pair_roc_auc_median": float(
                    evaluation_metrics["roc_auc"].median()
                ),
                "flight_pair_count": len(evaluation_metrics)
            }
        )

        mlflow.log_dict(model_configuration,
                        "model_configuration.json"
        )

        mlflow.log_dict(
            {
                "evaluation_records": evaluation_records
            },
            "evaluation_metrics.json"
        )

        mlflow.sklearn.log_model(
            sk_model = final_pipeline,
            name = "model",
            signature = model_signature,
            input_example = input_example,
            serialization_format = (
                mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            )
        )

        return {
            "model_name": model_configuration["name"],
            "run_id": run.info.run_id,
            "mean_roc_auc": float(evaluation_metrics["roc_auc"].mean()),
            "standard_deviation": float(
                evaluation_metrics["roc_auc"].std(ddof = 0)
            )
        }


def main() -> None:
    args = parse_args()

    ensure_inputs_exist()

    window_dataset, manifest, feature_columns = load_window_dataset()

    validate_window_dataset(
        window_dataset = window_dataset,
        feature_columns = feature_columns
    )

    # Learning normal behavior from healthy flights and confirmed pre-fault
    # periods; fault-state and transition windows are excluded from fitting.
    normal_windows = window_dataset.loc[
        window_dataset["window_label"].isin(
            ["normal", "pre_fault_state"]
        )
    ].copy()

    fault_state_windows = window_dataset.loc[
        window_dataset["window_label"].eq("fault_state")
    ].copy()

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DATABASE_PATH}")
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name = args.run_name) as parent_run:
        mlflow.log_params(
            {
                "benchmark_type": "normal_only_anomaly_detection",
                "model_count": len(MODEL_CONFIGURATIONS)
            }
        )

        mlflow.log_dict(manifest, "telemetry_windows_v3_manifest.json")

        model_results = []

        for model_configuration in MODEL_CONFIGURATIONS:
            model_results.append(
                train_and_log_model(
                    normal_windows = normal_windows,
                    fault_state_windows = fault_state_windows,
                    feature_columns = feature_columns,
                    manifest = manifest,
                    model_configuration = model_configuration
                )
            )

        mlflow.log_dict(
            {
                "model_results": model_results
            },
            "benchmark_summary.json"
        )

    print(f"MLflow parent run ID : {parent_run.info.run_id}")

    for result in model_results:
        print(
            f"{result['model_name']} | "
            f"run ID : {result['run_id']} | "
            f"mean flight-pair ROC-AUC : {result['mean_roc_auc']:.4f} | "
            f"standard deviation : {result['standard_deviation']:.4f}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Anomaly benchmark failed : {exc}")
        sys.exit(1)