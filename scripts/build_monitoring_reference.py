import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_ROOT = PROJECT_ROOT / "data" / "alfa" / "curated"

DEFAULT_WINDOW_DATASET_PATH = (
    CURATED_ROOT / "telemetry_windows_v3.parquet"
)

DEFAULT_SERVING_BUNDLE_DIRECTORY = (
    PROJECT_ROOT / "deployment" / "serving_bundle"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "monitoring" / "reference"
)

REFERENCE_WINDOW_LABELS = (
    "normal",
    "pre_fault_state"
)


# Parsing the immutable model bundle and monitoring reference destination
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--window-dataset",
        type = Path,
        default = DEFAULT_WINDOW_DATASET_PATH
    )

    parser.add_argument(
        "--serving-bundle-directory",
        type = Path,
        default = DEFAULT_SERVING_BUNDLE_DIRECTORY
    )

    parser.add_argument(
        "--output-directory",
        type = Path,
        default = DEFAULT_OUTPUT_DIRECTORY
    )

    return parser.parse_args()


# Computing a content hash for reference-data traceability
def calculate_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


# Building the fixed healthy-operation reference used by drift reports
def main() -> None:
    args = parse_args()

    window_dataset_path = (
        args.window_dataset.resolve()
    )

    serving_bundle_directory = (
        args.serving_bundle_directory.resolve()
    )

    output_directory = (
        args.output_directory.resolve()
    )

    model_directory = (
        serving_bundle_directory / "model"
    )

    contract_path = (
        serving_bundle_directory
        / "prediction_contract.json"
    )

    required_paths = [
        window_dataset_path,
        model_directory,
        contract_path
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

        raise FileNotFoundError(f"Required monitoring inputs not found : \n{missing_text}")

    reference_path = (
        output_directory
        / "serving_reference.parquet"
    )

    manifest_path = (
        output_directory
        / "serving_reference_manifest.json"
    )

    if reference_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Monitoring reference already exists. Remove it only when "
            "intentionally creating a reference for a new model release."
        )

    contract = json.loads(
        contract_path.read_text()
    )

    feature_names = contract.get(
        "feature_names"
    )

    if (
        not isinstance(feature_names, list)
        or not feature_names
        or len(feature_names) != len(set(feature_names))
    ):
        raise RuntimeError("Prediction contract does not contain valid feature names.")

    window_dataset = pd.read_parquet(
        window_dataset_path
    )

    required_columns = {
        "window_label",
        *feature_names
    }

    missing_columns = required_columns.difference(
        window_dataset.columns
    )

    if missing_columns:
        raise RuntimeError(f"Telemetry dataset is missing monitoring columns : {sorted(missing_columns)}")

    reference_windows = window_dataset.loc[
        window_dataset["window_label"].isin(
            REFERENCE_WINDOW_LABELS
        )
    ].copy()

    if reference_windows.empty:
        raise RuntimeError("No confirmed normal-state reference windows were found.")

    reference_features = reference_windows[
        feature_names
    ]

    feature_values = reference_features.to_numpy(
        dtype = "float64"
    )

    if not np.isfinite(feature_values).all():
        raise RuntimeError("Monitoring reference contains non-finite feature values.")

    model = mlflow.sklearn.load_model(
        str(model_directory)
    )

    if not hasattr(model, "named_steps"):
        raise RuntimeError("Serving model does not expose its preprocessing pipeline.")

    variance_filter = model.named_steps.get(
        "variance_filter"
    )

    if variance_filter is None:
        raise RuntimeError("Serving model does not contain the expected variance filter.")

    retained_mask = variance_filter.get_support()

    if len(retained_mask) != len(feature_names):
        raise RuntimeError("Variance-filter inputs do not match the prediction contract.")

    retained_feature_indices = np.flatnonzero(
        retained_mask
    ).tolist()

    retained_feature_names = [
        feature_names[index]
        for index in retained_feature_indices
    ]

    if not retained_feature_names:
        raise RuntimeError("Serving model does not retain any monitoring features.")

    probabilities = np.asarray(
        model.predict_proba(reference_features),
        dtype = "float64"
    )

    if (
        probabilities.shape
        != (len(reference_features), 2)
        or not np.isfinite(probabilities).all()
    ):
        raise RuntimeError("Serving model returned invalid reference probabilities.")

    fault_score_index = int(
        contract["fault_score_column_index"]
    )

    alert_threshold = float(
        contract["alert_threshold"]
    )

    fault_scores = probabilities[
        :,
        fault_score_index
    ]

    reference = reference_features[
        retained_feature_names
    ].copy()

    reference["fault_score"] = fault_scores
    reference["alert"] = (
        fault_scores >= alert_threshold
    ).astype("int64")

    manifest = {
        "registered_model_uri": contract[
            "registered_model_uri"
        ],
        "source_run_id": contract[
            "source_run_id"
        ],
        "source_dataset_sha256": calculate_file_sha256(
            window_dataset_path
        ),
        "reference_window_labels": list(
            REFERENCE_WINDOW_LABELS
        ),
        "reference_row_count": len(reference),
        "input_feature_count": len(feature_names),
        "input_feature_names": feature_names,
        "retained_feature_count": len(
            retained_feature_names
        ),
        "retained_feature_indices": (
            retained_feature_indices
        ),
        "retained_feature_names": (
            retained_feature_names
        ),
        "monitoring_columns": (
            reference.columns.tolist()
        ),
        "fault_score_column": "fault_score",
        "alert_column": "alert",
        "alert_threshold": alert_threshold
    }

    output_directory.mkdir(
        parents = True,
        exist_ok = True
    )

    temporary_reference_path = (
        output_directory
        / "serving_reference.tmp.parquet"
    )

    temporary_manifest_path = (
        output_directory
        / "serving_reference_manifest.tmp.json"
    )

    reference.to_parquet(
        temporary_reference_path,
        index = False
    )

    temporary_manifest_path.write_text(
        json.dumps(
            manifest,
            indent = 2,
            sort_keys = True
        ) + "\n"
    )

    temporary_reference_path.replace(
        reference_path
    )

    temporary_manifest_path.replace(
        manifest_path
    )

    print(f"Monitoring reference saved : {reference_path}")
    print(f"Monitoring manifest saved : {manifest_path}")
    print(f"Reference windows : {len(reference)}")
    print(
        "Retained monitoring features : "
        f"{len(retained_feature_names)}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "Monitoring reference build failed : "
            f"{exc}"
        )
        sys.exit(1)