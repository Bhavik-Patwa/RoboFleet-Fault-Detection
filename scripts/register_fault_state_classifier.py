import json
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_DATABASE_PATH = PROJECT_ROOT / "mlflow.db"

SOURCE_RUN_ID = "6505151bcb8b4e658dea78f0da5ae712"
SOURCE_MODEL_URI = "models:/m-fb5b83165063410292bdd684b3bb78e6"

REGISTERED_MODEL_NAME = "telemetry-fault-state-classifier"
MODEL_ALIAS = "serving"


# Registering the selected model and verifying its prediction contract
def main() -> None:
    if not TRACKING_DATABASE_PATH.exists():
        raise FileNotFoundError(f"Tracking database not found : {TRACKING_DATABASE_PATH}")

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DATABASE_PATH}")

    client = MlflowClient()
    run = client.get_run(SOURCE_RUN_ID)

    model_info = mlflow.models.get_model_info(
        SOURCE_MODEL_URI
    )

    # Confirming that the model and threshold share the same source run
    if model_info.run_id != SOURCE_RUN_ID:
        raise RuntimeError("Selected model does not belong to the source run.")

    if run.info.status != "FINISHED":
        raise RuntimeError("The source training run has not finished successfully.")

    if model_info.signature is None:
        raise RuntimeError("Selected model does not contain a prediction signature.")

    threshold = float(
        run.data.params["provisional_full_training_threshold"]
    )

    if not np.isfinite(threshold) or threshold < 0.0:
        raise RuntimeError("The saved alert threshold must be finite and non-negative.")

    # Loading the saved example without changing its feature order
    model_path = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri = SOURCE_MODEL_URI
        )
    )

    example = json.loads(
        (model_path / "input_example.json").read_text()
    )

    features = pd.DataFrame(
        data = example["data"],
        columns = example["columns"]
    )

    original_model = mlflow.sklearn.load_model(
        SOURCE_MODEL_URI
    )

    # Class order determines which probability represents a fault
    np.testing.assert_array_equal(
        original_model.classes_,
        [0, 1]
    )

    original_scores = original_model.predict_proba(
        features
    )

    version = mlflow.register_model(
        SOURCE_MODEL_URI,
        REGISTERED_MODEL_NAME
    )

    registered_uri = (
        f"models:/{REGISTERED_MODEL_NAME}/{version.version}"
    )

    # Verifying the prediction method that the API will use
    registered_model = mlflow.pyfunc.load_model(
        registered_uri
    )

    registered_scores = np.asarray(
        registered_model.predict(features)
    )

    np.testing.assert_array_equal(
        original_scores,
        registered_scores
    )

    np.testing.assert_array_equal(
        original_scores[:, 1] >= threshold,
        registered_scores[:, 1] >= threshold
    )

    # Keeping a separate contract artifact for each registered version
    contract_path = (
        f"registered_models/{REGISTERED_MODEL_NAME}/"
        f"{version.version}/prediction_contract.json"
    )

    contract_uri = (
        f"runs:/{SOURCE_RUN_ID}/{contract_path}"
    )

    contract = {
        "source_run_id": SOURCE_RUN_ID,
        "source_model_uri": SOURCE_MODEL_URI,
        "registered_model_uri": registered_uri,
        "model_signature": model_info.signature.to_dict(),
        "feature_names": features.columns.tolist(),
        "feature_set": run.data.params["feature_set"],
        "output_classes": [0, 1],
        "output_class_labels": {
            "0": "normal_state",
            "1": "fault_state"
        },
        "fault_score_column_index": 1,
        "alert_threshold": threshold,
        "alert_rule": "fault_score >= alert_threshold",
        "threshold_status": (
            "cross_validated_calibration_without_external_validation"
        ),
        "target_normal_state_alert_rate": float(
            run.data.params["target_normal_flight_alert_rate"]
        ),
        "feature_manifest_uri": (
            f"runs:/{SOURCE_RUN_ID}/"
            "telemetry_windows_v3_manifest.json"
        ),
        "evaluation_uri": (
            f"runs:/{SOURCE_RUN_ID}/evaluation_metrics.json"
        )
    }

    client.log_dict(
        run_id = SOURCE_RUN_ID,
        dictionary = contract,
        artifact_file = contract_path
    )

    client.set_model_version_tag(
        name = REGISTERED_MODEL_NAME,
        version = version.version,
        key = "prediction_contract_uri",
        value = contract_uri
    )

    client.set_model_version_tag(
        name = REGISTERED_MODEL_NAME,
        version = version.version,
        key = "prediction_roundtrip",
        value = "passed"
    )

    # Publishing the lookup alias only after verification and contract logging
    client.set_registered_model_alias(
        name = REGISTERED_MODEL_NAME,
        alias = MODEL_ALIAS,
        version = version.version
    )

    selected_version = client.get_model_version_by_alias(
        name = REGISTERED_MODEL_NAME,
        alias = MODEL_ALIAS
    )

    if str(selected_version.version) != str(version.version):
        raise RuntimeError("Serving alias does not resolve to the verified version.")

    print(f"Registered and verified : {registered_uri}")
    print(
        "Serving model : "
        f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    )
    print(f"Prediction contract : {contract_uri}")
    print(f"Input feature count : {len(features.columns)}")
    print(f"Alert threshold : {threshold:.17g}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Model registration failed : {exc}")
        sys.exit(1)