import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_DATABASE_PATH = PROJECT_ROOT / "mlflow.db"

DEFAULT_MODEL_NAME = "telemetry-fault-state-classifier"
DEFAULT_MODEL_ALIAS = "serving"
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "deployment" / "serving_bundle"
)


# Parsing the registered model and bundle destination
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        default = DEFAULT_MODEL_NAME
    )

    parser.add_argument(
        "--model-alias",
        default = DEFAULT_MODEL_ALIAS
    )

    parser.add_argument(
        "--output-directory",
        type = Path,
        default = DEFAULT_OUTPUT_DIRECTORY
    )

    return parser.parse_args()


# Exporting one verified registry version into a portable serving bundle
def main() -> None:
    args = parse_args()

    if not TRACKING_DATABASE_PATH.exists():
        raise FileNotFoundError(f"Tracking database not found : {TRACKING_DATABASE_PATH}")

    output_directory = args.output_directory.resolve()

    if output_directory.exists():
        raise FileExistsError(f"Serving bundle already exists : {output_directory}")

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DATABASE_PATH}")

    client = MlflowClient()

    version = client.get_model_version_by_alias(
        name = args.model_name,
        alias = args.model_alias
    )

    registered_model_uri = (f"models:/{args.model_name}/{version.version}")

    contract_uri = version.tags.get(
        "prediction_contract_uri"
    )

    if not contract_uri:
        raise RuntimeError("Registered model version has no prediction contract.")

    model_source = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri = registered_model_uri
        )
    )

    contract_source = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri = contract_uri
        )
    )

    contract = json.loads(contract_source.read_text())

    if contract["registered_model_uri"] != registered_model_uri:
        raise RuntimeError("Prediction contract identifies a different model version.")

    output_directory.parent.mkdir(
        parents = True,
        exist_ok = True
    )

    # Constructing the bundle separately so partial exports are never published
    with tempfile.TemporaryDirectory(dir = output_directory.parent) as temporary_directory:
        staged_bundle = (
            Path(temporary_directory) / "serving_bundle"
        )

        staged_bundle.mkdir()

        shutil.copytree(
            model_source,
            staged_bundle / "model"
        )

        shutil.copy2(
            contract_source,
            staged_bundle / "prediction_contract.json"
        )

        staged_bundle.replace(output_directory)

    print(f"Serving bundle exported : {output_directory}")
    print(f"Registered model : {registered_model_uri}")
    print(f"Source run ID : {contract['source_run_id']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Serving bundle export failed : {exc}")
        sys.exit(1)