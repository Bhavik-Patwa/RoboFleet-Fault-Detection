import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_DIRECTORY = (
    PROJECT_ROOT / "deployment" / "serving_bundle"
)


# Parsing the running API and exported serving bundle locations
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--api-url",
        default = "http://127.0.0.1:8000"
    )

    parser.add_argument(
        "--bundle-directory",
        type = Path,
        default = DEFAULT_BUNDLE_DIRECTORY
    )

    return parser.parse_args()


# Sending one JSON request and requiring a successful JSON response
def request_json(url: str, method: str = "GET", payload: object | None = None) -> dict:
    request_data = None

    if payload is not None:
        request_data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url = url,
        data = request_data,
        method = method,
        headers = {
            "Content-Type": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout = 30
        ) as response:
            return json.loads(
                response.read().decode("utf-8")
            )
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API request failed : {url}") from exc


# Verifying that the container can execute its bundled model
def main() -> None:
    args = parse_args()

    bundle_directory = args.bundle_directory.resolve()
    contract_path = (
        bundle_directory / "prediction_contract.json"
    )
    example_path = (
        bundle_directory / "model" / "input_example.json"
    )

    if not contract_path.is_file():
        raise FileNotFoundError(f"Prediction contract not found : {contract_path}")

    if not example_path.is_file():
        raise FileNotFoundError(f"Model input example not found : {example_path}")

    contract = json.loads(contract_path.read_text())
    example = json.loads(example_path.read_text())

    records = [
        dict(zip(example["columns"], values))
        for values in example["data"]
    ]

    readiness = request_json(f"{args.api_url.rstrip('/')}/ready")

    if readiness != {"status": "ready"}:
        raise RuntimeError("API did not report a ready state.")

    result = request_json(
        url = f"{args.api_url.rstrip('/')}/predict",
        method = "POST",
        payload = records
    )

    if result.get("model_uri") != contract["registered_model_uri"]:
        raise RuntimeError("API prediction used an unexpected model version.")

    if result.get("threshold") != contract["alert_threshold"]:
        raise RuntimeError("API prediction used an unexpected alert threshold.")

    predictions = result.get("predictions")

    if (
        not isinstance(predictions, list)
        or len(predictions) != len(records)
    ):
        raise RuntimeError("API returned an unexpected prediction count.")

    for prediction in predictions:
        score = prediction.get("fault_score")
        alert = prediction.get("alert")

        if (
            type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise RuntimeError("API returned an invalid fault score.")

        if alert is not (score >= result["threshold"]):
            raise RuntimeError("API returned an inconsistent alert decision.")

    print(f"Container prediction verified : {result['model_uri']}")
    print(f"Prediction count : {len(predictions)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Serving API smoke test failed : {exc}")
        sys.exit(1)