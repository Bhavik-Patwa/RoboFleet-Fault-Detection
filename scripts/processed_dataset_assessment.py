import csv
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "alfa" / "processed"
METADATA_ROOT = PROJECT_ROOT / "data" / "alfa" / "metadata"

EXPECTED_FLIGHT_COUNT = 47


def ensure_paths_exist() -> None:
    if not PROCESSED_ROOT.exists():
        raise FileNotFoundError(f"Processed dataset directory not found : {PROCESSED_ROOT}")

    METADATA_ROOT.mkdir(parents = True, exist_ok = True)


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


def inspect_csv_structure(csv_path: Path) -> dict:
    with csv_path.open("r", newline = "") as handle:
        reader = csv.reader(handle)

        header = next(reader, [])
        expected_field_count = len(header)

        row_count = 0
        malformed_row_count = 0
        first_malformed_line_number = None

        for line_number, row in enumerate(reader, start = 2):
            row_count += 1

            if len(row) != expected_field_count:
                malformed_row_count += 1

                if first_malformed_line_number is None:
                    first_malformed_line_number = line_number

    return {
        "header": header,
        "expected_field_count": expected_field_count,
        "row_count": row_count,
        "malformed_row_count": malformed_row_count,
        "first_malformed_line_number": first_malformed_line_number
    }


def profile_well_formed_csv(csv_path: Path, columns: list[str]) -> dict:
    if not columns:
        return {
            "timestamp_column": None,
            "timestamp_null_fraction": None,
            "timestamp_is_monotonic": None,
            "max_null_fraction": None,
            "duplicate_row_count": None
        }

    frame = pd.read_csv(csv_path)

    timestamp_column = find_timestamp_column(columns)

    timestamp_null_fraction = None
    timestamp_is_monotonic = None

    if timestamp_column is not None:
        timestamp_series = pd.to_numeric(frame[timestamp_column], errors = "coerce")
        timestamp_null_fraction = float(timestamp_series.isna().mean())

        if timestamp_series.notna().any():
            timestamp_is_monotonic = bool(timestamp_series.dropna().is_monotonic_increasing)

    return {
        "timestamp_column": timestamp_column,
        "timestamp_null_fraction": timestamp_null_fraction,
        "timestamp_is_monotonic": timestamp_is_monotonic,
        "max_null_fraction": float(frame.isna().mean().max()) if len(frame.columns) > 0 else 0.0,
        "duplicate_row_count": int(frame.duplicated().sum())
    }


def build_topic_level_profile() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flight_directories = list_flight_directories()

    file_records = []
    failure_status_records = []

    for flight_directory in flight_directories:
        flight_name = flight_directory.name
        fault_label_raw = normalize_fault_label(flight_name)

        csv_files = sorted(flight_directory.glob("*.csv"))

        for csv_path in csv_files:
            topic_name = topic_name_from_file(csv_path.name, flight_name)
            structure_profile = inspect_csv_structure(csv_path)

            columns = structure_profile["header"]
            timestamp_column = find_timestamp_column(columns)

            parse_status = "well_formed"
            timestamp_null_fraction = None
            timestamp_is_monotonic = None
            max_null_fraction = None
            duplicate_row_count = None

            if structure_profile["malformed_row_count"] > 0:
                parse_status = "malformed_rows_detected"
            else:
                detailed_profile = profile_well_formed_csv(csv_path, columns)
                timestamp_null_fraction = detailed_profile["timestamp_null_fraction"]
                timestamp_is_monotonic = detailed_profile["timestamp_is_monotonic"]
                max_null_fraction = detailed_profile["max_null_fraction"]
                duplicate_row_count = detailed_profile["duplicate_row_count"]

            record = {
                "flight_name": flight_name,
                "fault_label_raw": fault_label_raw,
                "file_name": csv_path.name,
                "topic_name": topic_name,
                "row_count": structure_profile["row_count"],
                "column_count": structure_profile["expected_field_count"],
                "timestamp_column": timestamp_column,
                "timestamp_null_fraction": timestamp_null_fraction,
                "timestamp_is_monotonic": timestamp_is_monotonic,
                "max_null_fraction": max_null_fraction,
                "duplicate_row_count": duplicate_row_count,
                "malformed_row_count": structure_profile["malformed_row_count"],
                "first_malformed_line_number": structure_profile["first_malformed_line_number"],
                "parse_status": parse_status,
                "columns": "|".join(columns)
            }

            file_records.append(record)

            if "failure_status-" in topic_name:
                failure_status_records.append(
                    {
                        "flight_name": flight_name,
                        "fault_label_raw": fault_label_raw,
                        "topic_name": topic_name,
                        "row_count": structure_profile["row_count"],
                        "timestamp_column": timestamp_column,
                        "timestamp_is_monotonic": timestamp_is_monotonic,
                        "malformed_row_count": structure_profile["malformed_row_count"],
                        "parse_status": parse_status,
                        "columns": "|".join(columns)
                    }
                )

    file_profile_frame = pd.DataFrame(file_records).sort_values(
        ["topic_name", "flight_name"]
    ).reset_index(drop = True)

    topic_summary_frame = (
        file_profile_frame
        .groupby("topic_name", as_index = False)
        .agg(
            flight_count = ("flight_name", "nunique"),
            min_row_count = ("row_count", "min"),
            max_row_count = ("row_count", "max"),
            min_column_count = ("column_count", "min"),
            max_column_count = ("column_count", "max"),
            malformed_file_count = ("malformed_row_count", lambda s: int((s > 0).sum())),
            malformed_row_total = ("malformed_row_count", "sum"),
            timestamp_column_variant_count = ("timestamp_column", "nunique"),
            non_monotonic_timestamp_flight_count = ("timestamp_is_monotonic", lambda s: int(s.fillna(False).eq(False).sum())),
            max_null_fraction_seen = ("max_null_fraction", "max"),
            duplicate_rows_total = ("duplicate_row_count", "sum")
        )
        .sort_values(["flight_count", "topic_name"], ascending = [False, True])
        .reset_index(drop = True)
    )

    topic_summary_frame["is_present_in_all_flights"] = (
        topic_summary_frame["flight_count"] == EXPECTED_FLIGHT_COUNT
    )

    topic_summary_frame["has_stable_column_count"] = (
        topic_summary_frame["min_column_count"] == topic_summary_frame["max_column_count"]
    )

    topic_summary_frame["is_structurally_clean"] = (
        topic_summary_frame["malformed_file_count"] == 0
    )

    topic_summary_frame["is_structurally_eligible"] = (
        topic_summary_frame["is_present_in_all_flights"] &
        topic_summary_frame["has_stable_column_count"] &
        topic_summary_frame["is_structurally_clean"]
    )

    failure_status_frame = pd.DataFrame(failure_status_records).sort_values(
        ["flight_name", "topic_name"]
    ).reset_index(drop = True)

    return file_profile_frame, topic_summary_frame, failure_status_frame


def write_decision_summary(topic_summary_frame: pd.DataFrame, failure_status_frame: pd.DataFrame) -> None:
    summary_payload = {
        "flight_count_expected": EXPECTED_FLIGHT_COUNT,
        "distinct_topics": int(topic_summary_frame["topic_name"].nunique()),
        "topics_present_in_all_flights": topic_summary_frame.loc[
            topic_summary_frame["is_present_in_all_flights"] == True,
            "topic_name"
        ].sort_values().tolist(),
        "topics_structurally_clean": topic_summary_frame.loc[
            topic_summary_frame["is_structurally_clean"] == True,
            "topic_name"
        ].sort_values().tolist(),
        "is_structurally_eligible": topic_summary_frame.loc[
            topic_summary_frame["is_structurally_eligible"] == True,
            "topic_name"
        ].sort_values().tolist(),
        "failure_status_topics_found": sorted(failure_status_frame["topic_name"].unique().tolist()) if not failure_status_frame.empty else [],
        "flights_with_failure_status_files": sorted(failure_status_frame["flight_name"].unique().tolist()) if not failure_status_frame.empty else []
    }

    output_path = METADATA_ROOT / "processed_dataset_assessment_summary.json"
    output_path.write_text(json.dumps(summary_payload, indent = 2))


def main() -> None:
    ensure_paths_exist()

    file_profile_frame, topic_summary_frame, failure_status_frame = build_topic_level_profile()

    file_profile_path = METADATA_ROOT / "processed_dataset_file_profile.csv"
    topic_summary_path = METADATA_ROOT / "processed_dataset_topic_assessment.csv"
    failure_status_path = METADATA_ROOT / "processed_failure_status_inventory.csv"

    file_profile_frame.to_csv(file_profile_path, index = False)
    topic_summary_frame.to_csv(topic_summary_path, index = False)
    failure_status_frame.to_csv(failure_status_path, index = False)

    write_decision_summary(topic_summary_frame, failure_status_frame)

    print(f"Processed dataset file profile created : {file_profile_path}")
    print(f"Processed dataset topic assessment created : {topic_summary_path}")
    print(f"Failure status inventory created : {failure_status_path}")
    print(f"Distinct topics assessed : {topic_summary_frame['topic_name'].nunique()}")
    print(
        "Topics recommended for EDA review : "
        f"{int(topic_summary_frame['is_structurally_eligible'].sum())}"
    )

    malformed_topics = topic_summary_frame.loc[
        topic_summary_frame["malformed_file_count"] > 0,
        ["topic_name", "malformed_file_count", "malformed_row_total"]
    ]

    if not malformed_topics.empty:
        print("\nTopics with malformed CSV rows detected :")
        print(malformed_topics.to_string(index = False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Processed dataset assessment failed : {exc}")
        sys.exit(1)