import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PREDICTION_DATABASE_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "prediction_events.db"
)

DEFAULT_REFERENCE_DIRECTORY = (
    PROJECT_ROOT
    / "monitoring"
    / "reference"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "monitoring"
    / "reports"
)

# Computing a content hash for monitoring-report traceability
def calculate_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()

# Parsing the current prediction events and fixed reference locations
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prediction-database",
        type = Path,
        default = DEFAULT_PREDICTION_DATABASE_PATH
    )

    parser.add_argument(
        "--reference-directory",
        type = Path,
        default = DEFAULT_REFERENCE_DIRECTORY
    )

    parser.add_argument(
        "--output-directory",
        type = Path,
        default = DEFAULT_OUTPUT_DIRECTORY
    )

    parser.add_argument(
        "--minimum-event-count",
        type = int,
        required = True
    )

    parser.add_argument(
        "--start-event-id-exclusive",
        type = int,
        required = True
    )

    parser.add_argument(
        "--end-event-id-inclusive",
        type = int,
        required = True
    )

    return parser.parse_args()


# Loading one explicit prediction-event range for the registered model version
def load_prediction_events(database_path: Path, model_uri: str,
                           start_event_id_exclusive: int, end_event_id_inclusive: int
) -> pd.DataFrame:
    try:
        with closing(
            sqlite3.connect(
                database_path,
                timeout = 30
            )
        ) as connection:
            return pd.read_sql_query(
                """
                SELECT
                    event_id,
                    batch_id,
                    record_index,
                    observed_at_utc,
                    fault_score,
                    alert,
                    features_json
                FROM prediction_events
                WHERE model_uri = ?
                AND event_id > ?
                AND event_id <= ?
                ORDER BY event_id
                """,
                connection,
                params = (
                    model_uri,
                    start_event_id_exclusive,
                    end_event_id_inclusive
                )
            )

    except sqlite3.Error as exc:
        raise RuntimeError("Prediction events could not be loaded.") from exc


# Restoring stored feature arrays into the retained model feature schema
def build_current_monitoring_data(events: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    input_feature_count = int(
        manifest["input_feature_count"]
    )

    retained_feature_indices = [
        int(index)
        for index in manifest[
            "retained_feature_indices"
        ]
    ]

    retained_feature_names = manifest[
        "retained_feature_names"
    ]

    feature_records = []

    for event_index, serialized_features in enumerate(
        events["features_json"]
    ):
        try:
            feature_values = json.loads(
                serialized_features
            )
        except (
            TypeError,
            json.JSONDecodeError
        ) as exc:
            raise RuntimeError(f"Prediction event {event_index} has invalid features.") from exc

        if (
            not isinstance(feature_values, list)
            or len(feature_values)
            != input_feature_count
        ):
            raise RuntimeError(f"Prediction event {event_index} has an unexpected feature count.")

        feature_records.append(
            feature_values
        )

    feature_matrix = np.asarray(
        feature_records,
        dtype = "float64"
    )

    if (
        feature_matrix.shape
        != (len(events), input_feature_count)
        or not np.isfinite(feature_matrix).all()
    ):
        raise RuntimeError("Prediction events contain invalid feature values.")

    retained_values = feature_matrix[
        :,
        retained_feature_indices
    ]

    current_data = pd.DataFrame(
        retained_values,
        columns = retained_feature_names
    )

    current_data["fault_score"] = pd.to_numeric(
        events["fault_score"],
        errors = "coerce"
    )

    current_data["alert"] = pd.to_numeric(
        events["alert"],
        errors = "coerce"
    ).astype("int64")

    current_values = current_data.to_numpy(
        dtype = "float64"
    )

    if not np.isfinite(current_values).all():
        raise RuntimeError("Current monitoring data contains non-finite values.")

    if (
        current_data["fault_score"].lt(0.0).any()
        or current_data["fault_score"].gt(1.0).any()
        or not set(current_data["alert"]).issubset(
            {0, 1}
        )
    ):
        raise RuntimeError("Prediction events contain invalid model outputs.")

    return current_data


# Comparing recent prediction behavior with the fixed normal-state reference
def main() -> None:
    args = parse_args()

    if args.minimum_event_count <= 0:
        raise ValueError("Minimum event count must be greater than zero.")

    if args.start_event_id_exclusive < 0:
        raise ValueError("Start event ID must be zero or greater.")

    if (
        args.end_event_id_inclusive
        <= args.start_event_id_exclusive
    ):
        raise ValueError("End event ID must be greater than the start event ID.")

    prediction_database_path = (
        args.prediction_database.resolve()
    )

    reference_directory = (
        args.reference_directory.resolve()
    )

    output_directory = (
        args.output_directory.resolve()
    )

    reference_path = (
        reference_directory
        / "serving_reference.parquet"
    )

    manifest_path = (
        reference_directory
        / "serving_reference_manifest.json"
    )

    required_paths = [
        prediction_database_path,
        reference_path,
        manifest_path
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

    manifest = json.loads(
        manifest_path.read_text()
    )

    reference_data = pd.read_parquet(
        reference_path
    )

    expected_columns = manifest[
        "monitoring_columns"
    ]

    if reference_data.columns.tolist() != expected_columns:
        raise RuntimeError("Monitoring reference schema does not match its manifest.")

    events = load_prediction_events(
        database_path = prediction_database_path,
        model_uri = manifest[
            "registered_model_uri"
        ],
        start_event_id_exclusive = (
            args.start_event_id_exclusive
        ),
        end_event_id_inclusive = (
            args.end_event_id_inclusive
        )
    )

    if len(events) < args.minimum_event_count:
        raise RuntimeError(
            "Not enough prediction events are available for the requested report. "
            f"Required : {args.minimum_event_count}; "
            f"available : {len(events)}"
        )

    current_data = build_current_monitoring_data(
        events = events,
        manifest = manifest
    )

    if current_data.columns.tolist() != expected_columns:
        raise RuntimeError("Current prediction schema does not match the reference.")

    report = Report(
        [
            DataDriftPreset()
        ],
        metadata = {
            "registered_model_uri": manifest[
                "registered_model_uri"
            ],
            "reference_row_count": len(
                reference_data
            ),
            "current_event_count": len(
                current_data
            ),
            "prediction_batch_count": int(
                events["batch_id"].nunique()
            ),
            "start_event_id": int(
                events["event_id"].min()
            ),
            "end_event_id": int(
                events["event_id"].max()
            )
        }
    )

    snapshot = report.run(
        current_data = current_data,
        reference_data = reference_data,
        name = "telemetry-fault-state-monitoring"
    )

    output_directory.mkdir(
        parents = True,
        exist_ok = True
    )

    generated_at_utc = datetime.now(
        timezone.utc
    )

    report_timestamp = generated_at_utc.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    report_stem = (
        f"prediction_drift_report_{report_timestamp}"
    )

    html_path = (
        output_directory
        / f"{report_stem}.html"
    )

    json_path = (
        output_directory
        / f"{report_stem}.json"
    )

    report_manifest_path = (
        output_directory
        / f"{report_stem}_manifest.json"
    )

    snapshot.save_html(
        str(html_path)
    )

    snapshot.save_json(
        str(json_path)
    )

    # Saving an audit record because Evidently output does not preserve the model identity
    # and selected prediction-event range.
    report_manifest = {
        "report_name": "telemetry-fault-state-monitoring",
        "generated_at_utc": generated_at_utc.isoformat(),
        "registered_model_uri": manifest[
            "registered_model_uri"
        ],
        "source_run_id": manifest[
            "source_run_id"
        ],
        "reference_file": reference_path.name,
        "reference_file_sha256": calculate_file_sha256(
            reference_path
        ),
        "reference_row_count": len(reference_data),
        "requested_start_event_id_exclusive": (
            args.start_event_id_exclusive
        ),
        "requested_end_event_id_inclusive": (
            args.end_event_id_inclusive
        ),
        "analyzed_start_event_id": int(
            events["event_id"].min()
        ),
        "analyzed_end_event_id": int(
            events["event_id"].max()
        ),
        "current_event_count": len(current_data),
        "prediction_batch_count": int(
            events["batch_id"].nunique()
        ),
        "html_report_file": html_path.name,
        "html_report_sha256": calculate_file_sha256(
            html_path
        ),
        "json_results_file": json_path.name,
        "json_results_sha256": calculate_file_sha256(
            json_path
        )
    }

    temporary_report_manifest_path = (
        output_directory
        / f"{report_stem}_manifest.tmp.json"
    )

    temporary_report_manifest_path.write_text(
        json.dumps(
            report_manifest,
            indent = 2,
            sort_keys = True
        ) + "\n"
    )

    temporary_report_manifest_path.replace(
        report_manifest_path
    )

    print(f"Drift report saved : {html_path}")
    print(f"Drift results saved : {json_path}")
    print(f"Prediction events analyzed : {len(events)}")
    print(
        "Prediction batches analyzed : "
        f"{events['batch_id'].nunique()}"
    )
    print(
        "Prediction event range : "
        f"{events['event_id'].min()}-"
        f"{events['event_id'].max()}"
    )
    print(
        "Reference alert rate : "
        f"{reference_data['alert'].mean():.6f}"
    )
    print(
        "Current alert rate : "
        f"{current_data['alert'].mean():.6f}"
    )

    print(
        "Report manifest saved : "
        f"{report_manifest_path}"
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "Prediction drift report failed : "
            f"{exc}"
        )
        sys.exit(1)