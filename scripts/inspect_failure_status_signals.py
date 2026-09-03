import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "alfa" / "processed"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "alfa" / "metadata"

EXPECTED_FLIGHT_COUNT = 47


def ensure_paths_exist() -> None:
    if not PROCESSED_ROOT.exists():
        raise FileNotFoundError(f"Processed dataset directory not found : {PROCESSED_ROOT}")

    OUTPUT_ROOT.mkdir(parents = True, exist_ok = True)


def list_flight_directories() -> list[Path]:
    flight_directories = sorted([path for path in PROCESSED_ROOT.iterdir() if path.is_dir()])

    if len(flight_directories) != EXPECTED_FLIGHT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FLIGHT_COUNT} flight directories, found {len(flight_directories)}."
        )

    return flight_directories


def topic_name_from_file(file_name: str, flight_name: str) -> str:
    prefix = f"{flight_name}-"
    if file_name.startswith(prefix):
        return file_name[len(prefix):]
    return file_name


def normalize_fault_label(flight_name: str) -> str:
    if flight_name.endswith("_no_ground_truth"):
        return "no_ground_truth"

    if flight_name.endswith("_no_failure"):
        return "no_failure"

    known_fault_labels = [
        "rudder_zero__left_aileron_failure",
        "left_aileron__right_aileron__failure",
        "both_ailerons_failure",
        "right_aileron_failure_with_emr_traj",
        "left_aileron_failure",
        "right_aileron_failure",
        "rudder_right_failure",
        "rudder_left_failure",
        "elevator_failure",
        "engine_failure_with_emr_traj",
        "engine_failure"
    ]

    for label in known_fault_labels:
        suffix = f"_{label}"
        if flight_name.endswith(suffix):
            return label

    raise ValueError(f"Could not normalize fault label from flight name : {flight_name}")


def find_timestamp_column(columns: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}

    priority_candidates = [
        "%time",
        "timestamp",
        "time",
        "header.stamp.secs",
        "header.stamp.nsecs"
    ]

    for candidate in priority_candidates:
        if candidate in lowered:
            return lowered[candidate]

    for column in columns:
        lowered_column = column.lower()
        if "time" in lowered_column or "stamp" in lowered_column:
            return column

    return None


def summarize_signal_column(frame: pd.DataFrame, timestamp_column: str | None, column_name: str) -> dict:
    series = pd.to_numeric(frame[column_name], errors = "coerce")

    non_null_series = series.dropna()
    non_zero_mask = series.fillna(0).ne(0)

    non_zero_count = int(non_zero_mask.sum())
    unique_non_null_count = int(non_null_series.nunique())

    min_value = float(non_null_series.min()) if not non_null_series.empty else None
    max_value = float(non_null_series.max()) if not non_null_series.empty else None

    first_non_zero_index = None
    first_non_zero_time = None

    if non_zero_count > 0:
        first_non_zero_index = int(non_zero_mask[non_zero_mask].index[0])

        if timestamp_column is not None:
            timestamp_series = pd.to_numeric(frame[timestamp_column], errors = "coerce")
            timestamp_value = timestamp_series.iloc[first_non_zero_index]

            if pd.notna(timestamp_value):
                first_non_zero_time = float(timestamp_value)

    return {
        "column_name": column_name,
        "non_null_count": int(non_null_series.shape[0]),
        "null_fraction": float(series.isna().mean()),
        "unique_non_null_count": unique_non_null_count,
        "min_value": min_value,
        "max_value": max_value,
        "non_zero_count": non_zero_count,
        "first_non_zero_row_index": first_non_zero_index,
        "first_non_zero_time": first_non_zero_time
    }


def main() -> None:
    ensure_paths_exist()
    flight_directories = list_flight_directories()

    file_records = []
    signal_records = []

    for flight_directory in flight_directories:
        flight_name = flight_directory.name
        fault_label_raw = normalize_fault_label(flight_name)

        failure_status_files = sorted(flight_directory.glob("*failure_status-*.csv"))

        for csv_path in failure_status_files:
            topic_name = topic_name_from_file(csv_path.name, flight_name)
            frame = pd.read_csv(csv_path)

            timestamp_column = find_timestamp_column(frame.columns.tolist())

            file_records.append(
                {
                    "flight_name": flight_name,
                    "fault_label_raw": fault_label_raw,
                    "topic_name": topic_name,
                    "row_count": int(len(frame)),
                    "column_count": int(len(frame.columns)),
                    "timestamp_column": timestamp_column,
                    "columns": "|".join(frame.columns.tolist())
                }
            )

            value_columns = [column for column in frame.columns if column != timestamp_column]

            for column_name in value_columns:
                signal_summary = summarize_signal_column(frame, timestamp_column, column_name)

                signal_records.append(
                    {
                        "flight_name": flight_name,
                        "fault_label_raw": fault_label_raw,
                        "topic_name": topic_name,
                        **signal_summary
                    }
                )

    file_frame = pd.DataFrame(file_records).sort_values(
        ["flight_name", "topic_name"]
    ).reset_index(drop = True)

    signal_frame = pd.DataFrame(signal_records).sort_values(
        ["topic_name", "flight_name", "column_name"]
    ).reset_index(drop = True)

    summary_frame = (
        signal_frame
        .groupby(["topic_name", "column_name"], as_index = False)
        .agg(
            flight_count = ("flight_name", "nunique"),
            non_zero_flight_count = ("non_zero_count", lambda s: int((s > 0).sum())),
            min_first_non_zero_time = ("first_non_zero_time", "min"),
            max_first_non_zero_time = ("first_non_zero_time", "max"),
            min_value = ("min_value", "min"),
            max_value = ("max_value", "max")
        )
        .sort_values(["topic_name", "column_name"])
        .reset_index(drop = True)
    )

    file_output_path = OUTPUT_ROOT / "processed_failure_status_file_profile.csv"
    signal_output_path = OUTPUT_ROOT / "processed_failure_status_signal_profile.csv"
    summary_output_path = OUTPUT_ROOT / "processed_failure_status_signal_summary.csv"
    json_output_path = OUTPUT_ROOT / "processed_failure_status_signal_summary.json"

    file_frame.to_csv(file_output_path, index = False)
    signal_frame.to_csv(signal_output_path, index = False)
    summary_frame.to_csv(summary_output_path, index = False)

    json_output_path.write_text(
        json.dumps(
            {
                "failure_status_topics": summary_frame["topic_name"].drop_duplicates().tolist(),
                "signal_summary": summary_frame.to_dict(orient = "records")
            },
            indent = 2
        )
    )

    print(f"Failure status file profile created : {file_output_path}")
    print(f"Failure status signal profile created : {signal_output_path}")
    print(f"Failure status signal summary created : {summary_output_path}")
    print(f"Failure status topics found : {file_frame['topic_name'].nunique()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failure status signal inspection failed : {exc}")
        sys.exit(1)