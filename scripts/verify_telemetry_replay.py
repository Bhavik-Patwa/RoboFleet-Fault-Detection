import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import hashlib
import json
import mlflow

from build_telemetry_window_dataset import (
    build_feature_columns_by_topic,
    build_feature_name_index,
    build_source_file_index,
    build_window_dataset,
    load_reference_data,
    validate_processed_data_acquisition,
    validate_window_dataset
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "alfa"
    / "curated"
    / "telemetry_windows_v3.parquet"
)
TRACKING_DATABASE_PATH = (
    PROJECT_ROOT / "mlflow.db"
)

# Parsing one flight and one running prediction service
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--flight-name",
        required = True
    )

    parser.add_argument(
        "--api-url",
        default = "http://127.0.0.1:8000"
    )

    parser.add_argument(
        "--request-window-count",
        type = int,
        default = 5
    )

    return parser.parse_args()


# Requiring one successful JSON response from the API
def request_json(method: str, url: str, payload: object | None = None) -> dict:
    response = requests.request(
        method = method,
        url = url,
        json = payload,
        timeout = 30
    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):
        raise RuntimeError("API response is not a JSON object.")

    return result


# Confirming that the serving model was trained from this exact dataset
def validate_serving_dataset_identity(contract: dict) -> str:
    evaluation_uri = contract.get(
        "evaluation_uri"
    )

    if (
        not isinstance(evaluation_uri, str)
        or not evaluation_uri.startswith("runs:/")
    ):
        raise RuntimeError("API contract does not identify its evaluation artifact.")

    evaluation_uri = contract.get(
        "evaluation_uri"
    )

    source_run_id = contract.get(
        "source_run_id"
    )

    expected_evaluation_prefix = (
        f"runs:/{source_run_id}/"
    )

    if (
        not isinstance(source_run_id, str)
        or not source_run_id
        or not isinstance(evaluation_uri, str)
        or not evaluation_uri.startswith(
            expected_evaluation_prefix
        )
    ):
        raise RuntimeError("API contract does not identify an evaluation artifact from its source model run.")

    if not TRACKING_DATABASE_PATH.is_file():
        raise FileNotFoundError(
            f"Tracking database was not found : "
            f"{TRACKING_DATABASE_PATH}"
        )

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DATABASE_PATH}")

    evaluation_path = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri = evaluation_uri
        )
    )

    evaluation = json.loads(
        evaluation_path.read_text()
    )

    expected_sha256 = evaluation.get(
        "dataset_sha256"
    )

    digest = hashlib.sha256()

    with CURATED_DATASET_PATH.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    actual_sha256 = digest.hexdigest()

    if expected_sha256 != actual_sha256:
        raise RuntimeError("Curated dataset differs from the dataset used by the serving model.")

    return actual_sha256


# Rebuilding one flight from telemetry and verifying API scoring
def main() -> None:
    args = parse_args()

    if args.request_window_count <= 0:
        raise ValueError("Request window count must be greater than zero.")

    validate_processed_data_acquisition()

    if not CURATED_DATASET_PATH.is_file():
        raise FileNotFoundError(f"Curated telemetry window dataset was not found : {CURATED_DATASET_PATH}")

    (
        file_profile,
        flight_reference,
        column_reference
    ) = load_reference_data()

    selected_flight_reference = (
        flight_reference.loc[
            flight_reference["flight_name"].eq(
                args.flight_name
            )
        ]
    )

    if len(selected_flight_reference) != 1:
        raise ValueError("Flight name must identify exactly one training-flight reference record.")

    if not selected_flight_reference[
        "supervision_status"
    ].eq("usable").all():
        raise ValueError("Selected flight is not marked as usable.")

    columns_by_topic = (
        build_feature_columns_by_topic(
            column_reference
        )
    )

    source_file_index = (
        build_source_file_index(
            file_profile = file_profile,
            columns_by_topic = columns_by_topic
        )
    )

    feature_name_index = (
        build_feature_name_index(
            columns_by_topic
        )
    )

    reconstructed_windows = (
        build_window_dataset(
            flight_reference = (
                selected_flight_reference
            ),
            columns_by_topic = columns_by_topic,
            source_file_index = source_file_index,
            feature_name_index = feature_name_index
        )
    )

    validate_window_dataset(
        window_dataset = reconstructed_windows,
        flight_reference = (
            selected_flight_reference
        ),
        columns_by_topic = columns_by_topic
    )

    stored_windows = pd.read_parquet(
        CURATED_DATASET_PATH
    )

    stored_windows = (
        stored_windows.loc[
            stored_windows["flight_name"].eq(
                args.flight_name
            )
        ]
        .sort_values("window_start_ns")
        .reset_index(drop = True)
    )

    reconstructed_windows = (
        reconstructed_windows
        .sort_values("window_start_ns")
        .reset_index(drop = True)
    )

    if stored_windows.empty:
        raise RuntimeError("Selected flight has no stored telemetry windows.")

    if (
        reconstructed_windows.columns.tolist()
        != stored_windows.columns.tolist()
    ):
        raise RuntimeError("Reconstructed and stored feature schemas differ.")

    pd.testing.assert_frame_equal(
        reconstructed_windows,
        stored_windows,
        check_dtype = False,
        check_exact = True
    )

    api_root = args.api_url.rstrip("/")

    contract = request_json(
        method = "GET",
        url = f"{api_root}/model"
    )

    dataset_sha256 = (
        validate_serving_dataset_identity(
            contract = contract
        )
    )

    feature_names = contract.get(
        "feature_names"
    )

    if (
        not isinstance(feature_names, list)
        or not feature_names
        or len(feature_names)
        != len(set(feature_names))
    ):
        raise RuntimeError("API contract contains invalid feature names.")

    missing_features = set(
        feature_names
    ).difference(
        reconstructed_windows.columns
    )

    if missing_features:
        raise RuntimeError(f"Reconstructed telemetry is missing serving features : {sorted(missing_features)}")

    request_windows = reconstructed_windows.head(
        args.request_window_count
    )

    records = request_windows[
        feature_names
    ].to_dict(
        orient = "records"
    )

    result = request_json(
        method = "POST",
        url = f"{api_root}/predict",
        payload = records
    )

    if (
        result.get("model_uri")
        != contract["registered_model_uri"]
    ):
        raise RuntimeError("API used an unexpected model version.")

    if (
        result.get("threshold")
        != contract["alert_threshold"]
    ):
        raise RuntimeError("API used an unexpected alert threshold.")

    predictions = result.get(
        "predictions"
    )

    if (
        not isinstance(predictions, list)
        or len(predictions) != len(records)
    ):
        raise RuntimeError("API returned an unexpected prediction count.")

    threshold = float(
        result["threshold"]
    )

    for prediction in predictions:
        score = prediction.get("fault_score")
        alert = prediction.get("alert")

        if (
            type(score) not in (int, float)
            or not np.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise RuntimeError("API returned an invalid fault score.")

        if (
            type(alert) is not bool
            or alert != (score >= threshold)
        ):
            raise RuntimeError("API returned an inconsistent alert decision.")

    print(
        "Serving dataset SHA-256 : "
        f"{dataset_sha256}"
    )
    print(
        "Telemetry feature parity verified : "
        f"{args.flight_name}"
    )
    print(
        "Reconstructed windows : "
        f"{len(reconstructed_windows)}"
    )
    print(
        "API predictions verified : "
        f"{len(predictions)}"
    )
    print(
        "Registered model : "
        f"{result['model_uri']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "Telemetry replay verification failed : "
            f"{exc}"
        )
        sys.exit(1)