import argparse
import json
import sys
from pathlib import Path

import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "deployment" / "serving_bundle"
)


# Parsing the temporary serving-bundle destination
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-directory",
        type = Path,
        default = DEFAULT_OUTPUT_DIRECTORY
    )

    return parser.parse_args()


# Building a deterministic model artifact for software verification
def main() -> None:
    args = parse_args()
    output_directory = args.output_directory.resolve()

    if output_directory.exists():
        raise FileExistsError(f"Serving test bundle already exists : {output_directory}")

    output_directory.mkdir(
        parents = True,
        exist_ok = False
    )

    features = pd.DataFrame(
        {
            "telemetry_change": [
                -2.0,
                -1.0,
                1.0,
                2.0
            ],
            "initial_baseline_delta": [
                -1.0,
                -0.5,
                0.5,
                1.0
            ]
        }
    )

    targets = [
        0,
        0,
        1,
        1
    ]

    model = Pipeline(
        steps = [
            (
                "feature_scaler",
                StandardScaler()
            ),
            (
                "fault_state_classifier",
                LogisticRegression(
                    random_state = 42
                )
            )
        ]
    )

    model.fit(
        features,
        targets
    )

    probabilities = model.predict_proba(
        features
    )

    signature = infer_signature(
        features,
        probabilities
    )

    model_directory = (
        output_directory / "model"
    )

    mlflow.sklearn.save_model(
        sk_model = model,
        path = model_directory,
        signature = signature,
        input_example = features,
        serialization_format = (
            mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS
        ),
        pyfunc_predict_fn = "predict_proba"
    )

    loaded_model = mlflow.pyfunc.load_model(
        str(model_directory)
    )

    contract = {
        "source_run_id": loaded_model.metadata.run_id,
        "source_model_uri": (
            f"models:/{loaded_model.metadata.model_id}"
        ),
        "registered_model_uri": (
            "models:/telemetry-fault-state-classifier/verification"
        ),
        "feature_names": features.columns.tolist(),
        "output_classes": [0, 1],
        "output_class_labels": {
            "0": "normal_state",
            "1": "fault_state"
        },
        "fault_score_column_index": 1,
        "alert_threshold": 0.5,
        "alert_rule": "fault_score >= alert_threshold",
        "threshold_status": "software_verification",
        "target_normal_state_alert_rate": 0.0
    }

    contract_path = (
        output_directory / "prediction_contract.json"
    )

    contract_path.write_text(
        json.dumps(
            contract,
            indent = 2,
            sort_keys = True
        ) + "\n"
    )

    print(f"Serving test bundle created : {output_directory}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Serving test bundle build failed : {exc}")
        sys.exit(1)