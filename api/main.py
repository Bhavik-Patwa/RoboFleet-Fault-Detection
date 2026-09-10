import os
import json
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_DATABASE_PATH = PROJECT_ROOT / "mlflow.db"

MODEL_NAME = "telemetry-fault-state-classifier"
MODEL_ALIAS = "serving"
SERVING_BUNDLE_ROOT = os.getenv(
    "SERVING_BUNDLE_ROOT"
)


# Checking that the contract matches the loaded model and alert behavior
def validate_prediction_contract(contract: dict, model, model_uri: str) -> None:
    if contract["registered_model_uri"] != model_uri:
        raise RuntimeError("Prediction contract identifies a different model version.")

    if model.metadata.run_id != contract["source_run_id"]:
        raise RuntimeError("Prediction contract and model have different source runs.")

    expected_source_model_uri = (f"models:/{model.metadata.model_id}")

    if contract["source_model_uri"] != expected_source_model_uri:
        raise RuntimeError("Prediction contract identifies a different source model.")

    names = contract["feature_names"]

    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise RuntimeError("Prediction contract must contain unique feature names.")

    signature = model.metadata.signature

    if signature is None or signature.inputs is None:
        raise RuntimeError("Loaded model does not contain an input schema.")

    if names != signature.inputs.input_names():
        raise RuntimeError("Contract feature names or order differ from the model schema.")

    if (
        contract["output_classes"] != [0, 1]
        or contract["fault_score_column_index"] != 1
        or contract["alert_rule"] != "fault_score >= alert_threshold"
    ):
        raise RuntimeError("Prediction contract has unsupported output rules.")

    threshold = contract["alert_threshold"]

    if (
        type(threshold) not in (int, float)
        or not np.isfinite(threshold)
        or threshold < 0.0
    ):
        raise RuntimeError("Alert threshold must be finite and non-negative.")


# Loading either the local registry version or an immutable serving bundle
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False

    if SERVING_BUNDLE_ROOT:
        bundle_root = Path(
            SERVING_BUNDLE_ROOT
        ).resolve()

        model_load_uri = str(bundle_root / "model")
        contract_path = (
            bundle_root / "prediction_contract.json"
        )

        if not Path(model_load_uri).is_dir():
            raise RuntimeError("Serving bundle does not contain a model.")

        if not contract_path.is_file():
            raise RuntimeError("Serving bundle does not contain a prediction contract.")

        contract = json.loads(contract_path.read_text())
        registered_model_uri = contract[
            "registered_model_uri"
        ]

    else:
        if not TRACKING_DATABASE_PATH.exists():
            raise RuntimeError("Tracking database is missing.")

        mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DATABASE_PATH}")

        client = MlflowClient()

        version = client.get_model_version_by_alias(
            name = MODEL_NAME,
            alias = MODEL_ALIAS
        )

        registered_model_uri = (f"models:/{MODEL_NAME}/{version.version}")

        contract_uri = version.tags.get(
            "prediction_contract_uri"
        )

        if not contract_uri:
            raise RuntimeError("Registered model version has no prediction contract.")

        contract_path = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri = contract_uri
            )
        )

        contract = json.loads(contract_path.read_text())
        model_load_uri = registered_model_uri

    model = mlflow.pyfunc.load_model(model_load_uri)

    validate_prediction_contract(
        contract = contract,
        model = model,
        model_uri = registered_model_uri
    )

    app.state.model = model
    app.state.contract = contract
    app.state.ready = True

    try:
        yield
    finally:
        app.state.ready = False


app = FastAPI(
    title = "Telemetry Fault Detection API",
    description = ("Scores precomputed telemetry window features and returns binary fault-state alerts."),
    lifespan = lifespan
)


# Reporting whether the application process can answer requests
@app.get("/health")
def health() -> dict:
    return {"status": "alive"}


# Reporting whether model initialization completed successfully
@app.get("/ready")
def readiness() -> dict:
    if not getattr(app.state, "ready", False):
        raise HTTPException(
            status_code = 503,
            detail = "Model is not ready."
        )

    return {"status": "ready"}


# Exposing the input schema, model version and alert interpretation
@app.get("/model")
def model_metadata() -> dict:
    readiness()
    return app.state.contract


# Validating input records before restoring the model's feature order
@app.post("/predict")
def predict(records: list[dict[str, object]] = Body(...)) -> dict:
    readiness()

    contract = app.state.contract
    names = contract["feature_names"]
    expected_names = set(names)

    if not records:
        raise HTTPException(
            status_code = 422,
            detail = "Provide at least one feature record."
        )

    for record_index, record in enumerate(records):
        if set(record) != expected_names:
            raise HTTPException(
                status_code = 422,
                detail = {
                    "record_index": record_index,
                    "missing_features": sorted(expected_names - set(record)),
                    "unexpected_features": sorted(set(record) - expected_names)
                }
            )

        if any(type(value) not in (int, float) for value in record.values()):
            raise HTTPException(
                status_code = 422,
                detail = (
                    f"Record {record_index} : feature values must be numbers; strings, nulls and booleans are invalid."
                )
            )

    try:
        features = pd.DataFrame(
            records,
            columns = names,
            dtype = "float64"
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(
            status_code = 422,
            detail = "Feature values cannot be represented as numeric inputs."
        ) from exc

    if not np.isfinite(features.to_numpy()).all():
        raise HTTPException(
            status_code = 422,
            detail = "Feature values must be finite."
        )

    scores = np.asarray(
        app.state.model.predict(features),
        dtype = "float64"
    )

    if (
        scores.shape != (len(records), 2)
        or not np.isfinite(scores).all()
        or (scores < 0.0).any()
        or (scores > 1.0).any()
    ):
        raise HTTPException(
            status_code = 500,
            detail = "Model returned invalid probabilities."
        )

    threshold = contract["alert_threshold"]
    fault_scores = scores[:, contract["fault_score_column_index"]]

    return {
        "model_uri": contract["registered_model_uri"],
        "threshold": threshold,
        "predictions": [
            {
                "fault_score": float(score),
                "alert": bool(score >= threshold)
            }
            for score in fault_scores
        ]
    }