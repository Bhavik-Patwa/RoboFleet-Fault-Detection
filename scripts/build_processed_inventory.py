import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "alfa" / "processed"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "alfa" / "metadata"

EXPECTED_FLIGHT_DIRECTORIES = 47

FAILURE_STATUS_MARKER = "failure_status-"


def ensure_paths_exist() -> None:
    if not PROCESSED_ROOT.exists():
        raise FileNotFoundError(f"Processed dataset directory not found : {PROCESSED_ROOT}")

    OUTPUT_ROOT.mkdir(parents = True, exist_ok = True)


def list_flight_directories() -> list[Path]:
    flight_directories = sorted([path for path in PROCESSED_ROOT.iterdir() if path.is_dir()])

    if len(flight_directories) != EXPECTED_FLIGHT_DIRECTORIES:
        raise RuntimeError(
            f"Expected {EXPECTED_FLIGHT_DIRECTORIES} flight directories, found {len(flight_directories)}."
        )

    return flight_directories


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


def topic_name_from_file(file_name: str, flight_name: str) -> str:
    prefix = f"{flight_name}-"
    if file_name.startswith(prefix):
        return file_name[len(prefix):]
    return file_name


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", newline = "") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def read_csv_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", newline = "") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


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


def summarize_flight_directory(flight_directory: Path) -> dict:
    flight_name = flight_directory.name
    files = sorted([path for path in flight_directory.iterdir() if path.is_file()])

    csv_files = sorted([path for path in files if path.suffix == ".csv"])
    mat_files = sorted([path for path in files if path.suffix == ".mat"])
    bag_files = sorted([path for path in files if path.suffix == ".bag"])

    topic_names = sorted([topic_name_from_file(path.name, flight_name) for path in csv_files])
    failure_status_topics = sorted(
        [topic_name for topic_name in topic_names if FAILURE_STATUS_MARKER in topic_name]
    )

    timestamp_topics = []
    missing_timestamp_topics = []

    for csv_path in csv_files:
        topic_name = topic_name_from_file(csv_path.name, flight_name)
        header = read_csv_header(csv_path)
        timestamp_column = find_timestamp_column(header)

        if timestamp_column is None:
            missing_timestamp_topics.append(topic_name)
        else:
            timestamp_topics.append(topic_name)

    sample_topic_row_counts = {}
    sample_topic_timestamp_columns = {}

    for csv_path in csv_files[:10]:
        topic_name = topic_name_from_file(csv_path.name, flight_name)
        sample_topic_row_counts[topic_name] = count_csv_rows(csv_path)
        sample_topic_timestamp_columns[topic_name] = find_timestamp_column(read_csv_header(csv_path))

    return {
        "flight_name": flight_name,
        "fault_label_raw": normalize_fault_label(flight_name),
        "csv_file_count": len(csv_files),
        "mat_file_count": len(mat_files),
        "bag_file_count": len(bag_files),
        "topic_count": len(topic_names),
        "topics": "|".join(topic_names),
        "failure_status_topic_count": len(failure_status_topics),
        "failure_status_topics": "|".join(failure_status_topics),
        "timestamp_topic_count": len(timestamp_topics),
        "missing_timestamp_topic_count": len(missing_timestamp_topics),
        "missing_timestamp_topics": "|".join(missing_timestamp_topics),
        "sample_topic_row_counts_json": json.dumps(sample_topic_row_counts, sort_keys = True),
        "sample_topic_timestamp_columns_json": json.dumps(sample_topic_timestamp_columns, sort_keys = True)
    }


def write_json_summary(inventory_frame: pd.DataFrame) -> None:
    fault_distribution = Counter(inventory_frame["fault_label_raw"])
    topic_count_distribution = Counter(inventory_frame["topic_count"])

    summary = {
        "flight_count": int(len(inventory_frame)),
        "fault_distribution_raw": dict(sorted(fault_distribution.items())),
        "topic_count_distribution": dict(sorted(topic_count_distribution.items())),
        "flights_with_failure_status_topics": inventory_frame.loc[
            inventory_frame["failure_status_topic_count"] > 0,
            "flight_name"
        ].sort_values().tolist(),
        "flights_with_missing_timestamp_topics": inventory_frame.loc[
            inventory_frame["missing_timestamp_topic_count"] > 0,
            "flight_name"
        ].sort_values().tolist()
    }

    summary_path = OUTPUT_ROOT / "processed_flight_inventory_summary.json"
    summary_path.write_text(json.dumps(summary, indent = 2))


def main() -> None:
    ensure_paths_exist()
    flight_directories = list_flight_directories()

    records = [summarize_flight_directory(flight_directory) for flight_directory in flight_directories]
    inventory_frame = pd.DataFrame(records).sort_values("flight_name").reset_index(drop = True)

    inventory_csv_path = OUTPUT_ROOT / "processed_flight_inventory.csv"
    inventory_parquet_path = OUTPUT_ROOT / "processed_flight_inventory.parquet"

    inventory_frame.to_csv(inventory_csv_path, index = False)
    inventory_frame.to_parquet(inventory_parquet_path, index = False)

    write_json_summary(inventory_frame)

    print(f"Flight inventory created : {inventory_csv_path}")
    print(f"Parquet copy created : {inventory_parquet_path}")
    print(f"Flights profiled : {len(inventory_frame)}")

    print("Raw fault label distribution :")
    print(inventory_frame["fault_label_raw"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Processed inventory build failed : {exc}")
        sys.exit(1)