import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "alfa" / "processed"
METADATA_ROOT = PROJECT_ROOT / "data" / "alfa" / "metadata"

EXPECTED_FLIGHT_COUNT = 47
NON_FEATURE_COLUMN_NAMES = {"field.header.seq", "field.header.stamp", "field.time_ref"}


def ensure_inputs_exist() -> None:
    required_paths = [
        PROCESSED_ROOT,
        METADATA_ROOT / "processed_flight_inventory.csv",
        METADATA_ROOT / "processed_dataset_file_profile.csv",
        METADATA_ROOT / "processed_dataset_topic_assessment.csv",
        METADATA_ROOT / "processed_failure_status_inventory.csv",
        METADATA_ROOT / "processed_failure_status_signal_profile.csv"
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required input paths not found:\n{missing_text}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flight_inventory = pd.read_csv(METADATA_ROOT / "processed_flight_inventory.csv")
    file_profile = pd.read_csv(METADATA_ROOT / "processed_dataset_file_profile.csv")
    topic_assessment = pd.read_csv(METADATA_ROOT / "processed_dataset_topic_assessment.csv")
    failure_status_inventory = pd.read_csv(METADATA_ROOT / "processed_failure_status_inventory.csv")
    failure_status_signal_profile = pd.read_csv(METADATA_ROOT / "processed_failure_status_signal_profile.csv")

    flight_count = flight_inventory["flight_name"].nunique()

    if flight_count != EXPECTED_FLIGHT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FLIGHT_COUNT} flights, found {flight_count}."
        )

    return (
        flight_inventory,
        file_profile,
        topic_assessment,
        failure_status_inventory,
        failure_status_signal_profile
    )


def get_fault_attributes(raw_fault_label: str) -> dict:
    if raw_fault_label == "no_failure":
        return {
            "fault_name": "no_failure",
            "fault_family": "normal",
            "recovery_profile": "not_applicable",
            "supervision_status": "usable"
        }

    if raw_fault_label == "no_ground_truth":
        return {
            "fault_name": "no_ground_truth",
            "fault_family": "unknown",
            "recovery_profile": "unknown",
            "supervision_status": "excluded"
        }

    fault_name = raw_fault_label
    recovery_profile = "standard"

    if raw_fault_label.endswith("_with_emr_traj"):
        fault_name = raw_fault_label.removesuffix("_with_emr_traj")
        recovery_profile = "emergency_response"

    fault_family_by_name = {
        "both_ailerons_failure": "aileron",
        "elevator_failure": "elevator",
        "engine_failure": "engine",
        "left_aileron__right_aileron__failure": "aileron",
        "left_aileron_failure": "aileron",
        "right_aileron_failure": "aileron",
        "rudder_left_failure": "rudder",
        "rudder_right_failure": "rudder",
        "rudder_zero__left_aileron_failure": "multi_surface"
    }

    if fault_name not in fault_family_by_name:
        raise ValueError(f"Unsupported fault label : {raw_fault_label}")

    return {
        "fault_name": fault_name,
        "fault_family": fault_family_by_name[fault_name],
        "recovery_profile": recovery_profile,
        "supervision_status": "usable"
    }


def build_flight_reference(
    flight_inventory: pd.DataFrame,
    failure_status_inventory: pd.DataFrame,
    failure_status_signal_profile: pd.DataFrame
) -> pd.DataFrame:
    flight_reference = flight_inventory[["flight_name", "fault_label_raw"]].drop_duplicates().copy()

    fault_attributes = flight_reference["fault_label_raw"].apply(get_fault_attributes).apply(pd.Series)
    flight_reference = pd.concat([flight_reference, fault_attributes], axis = 1)

    flight_reference["is_fault_flight"] = ~flight_reference["fault_label_raw"].isin(
        ["no_failure", "no_ground_truth"]
    )

    fault_signal_inventory = (
        failure_status_inventory
        .groupby("flight_name", as_index = False)
        .agg(
            fault_signal_topic_count = ("topic_name", "nunique"),
            fault_signal_topics = ("topic_name", lambda values: "|".join(sorted(set(values))))
        )
    )

    first_fault_signal = (
        failure_status_signal_profile
        .dropna(subset = ["first_non_zero_time"])
        .groupby("flight_name", as_index = False)
        .agg(
            first_fault_signal_time = ("first_non_zero_time", "min")
        )
    )

    flight_reference = flight_reference.merge(
        fault_signal_inventory,
        on = "flight_name",
        how = "left",
        validate = "one_to_one"
    )

    flight_reference = flight_reference.merge(
        first_fault_signal,
        on = "flight_name",
        how = "left",
        validate = "one_to_one"
    )

    flight_reference["fault_signal_topic_count"] = (
        flight_reference["fault_signal_topic_count"].fillna(0).astype(int)
    )

    flight_reference["fault_signal_topics"] = (
        flight_reference["fault_signal_topics"].fillna("")
    )

    flight_reference["has_fault_signal_file"] = flight_reference["fault_signal_topic_count"] > 0

    flight_reference["fault_signal_source"] = "not_applicable"
    flight_reference.loc[
        flight_reference["is_fault_flight"] & flight_reference["has_fault_signal_file"],
        "fault_signal_source"
    ] = "failure_status_file"
    flight_reference.loc[
        flight_reference["is_fault_flight"] & (~flight_reference["has_fault_signal_file"]),
        "fault_signal_source"
    ] = "not_available"

    flight_reference["fault_signal_time_kind"] = "not_applicable"
    flight_reference.loc[
        flight_reference["is_fault_flight"] & flight_reference["has_fault_signal_file"],
        "fault_signal_time_kind"
    ] = "observed_non_zero_signal"
    flight_reference.loc[
        flight_reference["is_fault_flight"] & (~flight_reference["has_fault_signal_file"]),
        "fault_signal_time_kind"
    ] = "not_available"

    return flight_reference.sort_values("flight_name").reset_index(drop = True)


def build_topic_reference(
    topic_assessment: pd.DataFrame,
    file_profile: pd.DataFrame
) -> pd.DataFrame:
    column_signatures = (
        file_profile[["topic_name", "columns"]]
        .drop_duplicates()
        .groupby("topic_name", as_index = False)
        .agg(
            column_signature_count = ("columns", "nunique")
        )
    )

    topic_reference = topic_assessment.merge(
        column_signatures,
        on = "topic_name",
        how = "left",
        validate = "one_to_one"
    )

    topic_reference["column_signature_count"] = (
        topic_reference["column_signature_count"].fillna(0).astype(int)
    )

    topic_reference["has_complete_flight_coverage"] = (
        topic_reference["flight_count"] == EXPECTED_FLIGHT_COUNT
    )

    topic_reference["has_consistent_column_count"] = (
        topic_reference["min_column_count"] == topic_reference["max_column_count"]
    )

    topic_reference["has_consistent_column_signature"] = (
        topic_reference["column_signature_count"] == 1
    )

    topic_reference["has_parse_errors"] = (
        topic_reference["malformed_file_count"] > 0
    )

    topic_reference["is_label_source"] = (
        topic_reference["topic_name"].str.startswith("failure_status-")
    )

    topic_reference["is_recovery_artifact"] = (
        topic_reference["topic_name"] == "emergency_responder-traj_file.csv"
    )

    topic_reference["is_feature_eligible"] = (
        topic_reference["has_complete_flight_coverage"] &
        topic_reference["has_consistent_column_count"] &
        topic_reference["has_consistent_column_signature"] &
        (~topic_reference["has_parse_errors"]) &
        (~topic_reference["is_label_source"]) &
        (~topic_reference["is_recovery_artifact"])
    )

    topic_reference["feature_exclusion_reason"] = ""

    topic_reference.loc[
        topic_reference["is_label_source"],
        "feature_exclusion_reason"
    ] = "label_source"

    topic_reference.loc[
        topic_reference["feature_exclusion_reason"].eq("") &
        topic_reference["is_recovery_artifact"],
        "feature_exclusion_reason"
    ] = "recovery_artifact"

    topic_reference.loc[
        topic_reference["feature_exclusion_reason"].eq("") &
        (~topic_reference["has_complete_flight_coverage"]),
        "feature_exclusion_reason"
    ] = "incomplete_flight_coverage"

    topic_reference.loc[
        topic_reference["feature_exclusion_reason"].eq("") &
        (~topic_reference["has_consistent_column_count"]),
        "feature_exclusion_reason"
    ] = "inconsistent_column_count"

    topic_reference.loc[
        topic_reference["feature_exclusion_reason"].eq("") &
        (~topic_reference["has_consistent_column_signature"]),
        "feature_exclusion_reason"
    ] = "inconsistent_column_signature"

    topic_reference.loc[
        topic_reference["feature_exclusion_reason"].eq("") &
        topic_reference["has_parse_errors"],
        "feature_exclusion_reason"
    ] = "parse_errors"

    return topic_reference.sort_values("topic_name").reset_index(drop = True)


def build_feature_column_reference(
    topic_reference: pd.DataFrame,
    file_profile: pd.DataFrame
) -> pd.DataFrame:
    eligible_files = file_profile.loc[
        file_profile["topic_name"].isin(
            topic_reference.loc[
                topic_reference["is_feature_eligible"],
                "topic_name"
            ]
        ),
        ["flight_name", "file_name", "topic_name", "timestamp_column"]
    ].drop_duplicates()

    column_records = []

    for row in eligible_files.to_dict(orient = "records"):
        csv_path = PROCESSED_ROOT / row["flight_name"] / row["file_name"]
        frame = pd.read_csv(csv_path)

        for column_name in frame.columns:
            series = frame[column_name]

            column_records.append(
                {
                    "topic_name": row["topic_name"],
                    "flight_name": row["flight_name"],
                    "column_name": column_name,
                    "is_timestamp_column": column_name == row["timestamp_column"],
                    "null_fraction": float(series.isna().mean()),
                    "is_all_null": bool(series.notna().sum() == 0),
                    "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
                    "is_metadata_or_time_column": column_name in NON_FEATURE_COLUMN_NAMES
                }
            )

    feature_column_reference = (
        pd.DataFrame(column_records)
        .groupby(["topic_name", "column_name", "is_timestamp_column"], as_index = False)
        .agg(
            flight_count = ("flight_name", "nunique"),
            all_null_flight_count = ("is_all_null", "sum"),
            max_null_fraction = ("null_fraction", "max"),
            mean_null_fraction = ("null_fraction", "mean"),
            numeric_flight_count = ("is_numeric", "sum"),
            is_metadata_or_time_column = ("is_metadata_or_time_column", "all")
        )
        .sort_values(["topic_name", "column_name"])
        .reset_index(drop = True)
    )

    feature_column_reference["is_feature_column"] = (
        (~feature_column_reference["is_timestamp_column"]) &
        (feature_column_reference["all_null_flight_count"] == 0) &
        (feature_column_reference["max_null_fraction"] < 0.95) &
        (feature_column_reference["numeric_flight_count"] == feature_column_reference["flight_count"]) &
        (~feature_column_reference["is_metadata_or_time_column"])
    )

    feature_column_reference["feature_exclusion_reason"] = ""

    feature_column_reference.loc[
        feature_column_reference["is_timestamp_column"],
        "feature_exclusion_reason"
    ] = "timestamp_column"

    feature_column_reference.loc[
        feature_column_reference["feature_exclusion_reason"].eq("") &
        feature_column_reference["is_metadata_or_time_column"],
        "feature_exclusion_reason"
    ] = "metadata_or_time_column"

    feature_column_reference.loc[
        feature_column_reference["feature_exclusion_reason"].eq("") &
        (feature_column_reference["all_null_flight_count"] > 0),
        "feature_exclusion_reason"
    ] = "all_null_values"

    feature_column_reference.loc[
        feature_column_reference["feature_exclusion_reason"].eq("") &
        (feature_column_reference["max_null_fraction"] >= 0.95),
        "feature_exclusion_reason"
    ] = "high_null_fraction"

    feature_column_reference.loc[
        feature_column_reference["feature_exclusion_reason"].eq("") &
        (feature_column_reference["numeric_flight_count"] < feature_column_reference["flight_count"]),
        "feature_exclusion_reason"
    ] = "non_numeric_values"

    return feature_column_reference


def write_outputs(
    flight_reference: pd.DataFrame,
    topic_reference: pd.DataFrame,
    feature_column_reference: pd.DataFrame
) -> None:
    flight_reference_path = METADATA_ROOT / "flight_reference.csv"
    topic_reference_path = METADATA_ROOT / "topic_reference.csv"
    feature_column_reference_path = METADATA_ROOT / "feature_column_reference.csv"

    flight_reference.to_csv(flight_reference_path, index = False)
    topic_reference.to_csv(topic_reference_path, index = False)
    feature_column_reference.to_csv(feature_column_reference_path, index = False)

    print(f"Saved : {flight_reference_path}")
    print(f"Saved : {topic_reference_path}")
    print(f"Saved : {feature_column_reference_path}")
    print(f"Flights referenced : {len(flight_reference)}")
    print(f"Feature-eligible topics : {int(topic_reference['is_feature_eligible'].sum())}")
    print(f"Feature-eligible columns : {int(feature_column_reference['is_feature_column'].sum())}")


def main() -> None:
    ensure_inputs_exist()

    (
        flight_inventory,
        file_profile,
        topic_assessment,
        failure_status_inventory,
        failure_status_signal_profile
    ) = load_inputs()

    flight_reference = build_flight_reference(
        flight_inventory = flight_inventory,
        failure_status_inventory = failure_status_inventory,
        failure_status_signal_profile = failure_status_signal_profile
    )

    topic_reference = build_topic_reference(
        topic_assessment = topic_assessment,
        file_profile = file_profile
    )

    feature_column_reference = build_feature_column_reference(
        topic_reference = topic_reference,
        file_profile = file_profile
    )

    write_outputs(
        flight_reference = flight_reference,
        topic_reference = topic_reference,
        feature_column_reference = feature_column_reference
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Dataset reference build failed : {exc}")
        sys.exit(1)