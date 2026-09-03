import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_ROOT = PROJECT_ROOT / "data" / "alfa" / "metadata"


def ensure_inputs_exist() -> None:
    required_paths = [
        METADATA_ROOT / "flight_reference.csv",
        METADATA_ROOT / "topic_reference.csv",
        METADATA_ROOT / "feature_column_reference.csv"
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required input paths not found:\n{missing_text}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flight_reference = pd.read_csv(METADATA_ROOT / "flight_reference.csv")
    topic_reference = pd.read_csv(METADATA_ROOT / "topic_reference.csv")
    feature_column_reference = pd.read_csv(METADATA_ROOT / "feature_column_reference.csv")

    return flight_reference, topic_reference, feature_column_reference


def build_training_flight_reference(flight_reference: pd.DataFrame) -> pd.DataFrame:
    training_flight_reference = flight_reference.copy()

    training_flight_reference["anomaly_model_fit_candidate"] = (
        training_flight_reference["supervision_status"].eq("usable") &
        training_flight_reference["fault_name"].eq("no_failure")
    )

    training_flight_reference["anomaly_evaluation_candidate"] = (
        training_flight_reference["supervision_status"].eq("usable")
    )

    training_flight_reference["anomaly_detection_label"] = pd.NA
    training_flight_reference.loc[
        training_flight_reference["anomaly_evaluation_candidate"] &
        training_flight_reference["fault_name"].eq("no_failure"),
        "anomaly_detection_label"
    ] = "normal"
    training_flight_reference.loc[
        training_flight_reference["anomaly_evaluation_candidate"] &
        training_flight_reference["is_fault_flight"].eq(True),
        "anomaly_detection_label"
    ] = "fault"

    training_flight_reference["fault_signal_time_available"] = (
        training_flight_reference["first_fault_signal_time"].notna()
    )

    training_flight_reference["fault_classification_candidate"] = (
        training_flight_reference["supervision_status"].eq("usable") &
        training_flight_reference["is_fault_flight"].eq(True) &
        training_flight_reference["has_fault_signal_file"].eq(True) &
        training_flight_reference["fault_signal_time_available"]
    )

    training_flight_reference["fault_classification_label"] = pd.NA
    training_flight_reference.loc[
        training_flight_reference["fault_classification_candidate"],
        "fault_classification_label"
    ] = training_flight_reference["fault_family"]

    return training_flight_reference.sort_values("flight_name").reset_index(drop = True)


def build_training_topic_reference(
    topic_reference: pd.DataFrame,
    feature_column_reference: pd.DataFrame
) -> pd.DataFrame:
    usable_feature_counts = (
        feature_column_reference.loc[
            feature_column_reference["is_feature_column"].eq(True)
        ]
        .groupby("topic_name", as_index = False)
        .agg(
            feature_column_count = ("column_name", "nunique")
        )
    )

    training_topic_reference = topic_reference.merge(
        usable_feature_counts,
        on = "topic_name",
        how = "left",
        validate = "one_to_one"
    )

    training_topic_reference["feature_column_count"] = (
        training_topic_reference["feature_column_count"].fillna(0).astype(int)
    )

    training_topic_reference["included_for_training"] = (
        training_topic_reference["is_feature_eligible"].eq(True) &
        training_topic_reference["feature_column_count"].gt(0)
    )

    return training_topic_reference.sort_values("topic_name").reset_index(drop = True)


def build_training_column_reference(feature_column_reference: pd.DataFrame) -> pd.DataFrame:
    training_column_reference = feature_column_reference.loc[
        feature_column_reference["is_feature_column"].eq(True)
    ].copy()

    return training_column_reference.sort_values(
        ["topic_name", "column_name"]
    ).reset_index(drop = True)


def write_outputs(
    training_flight_reference: pd.DataFrame,
    training_topic_reference: pd.DataFrame,
    training_column_reference: pd.DataFrame
) -> None:
    training_flight_reference_path = METADATA_ROOT / "training_flight_reference.csv"
    training_topic_reference_path = METADATA_ROOT / "training_topic_reference.csv"
    training_column_reference_path = METADATA_ROOT / "training_column_reference.csv"

    training_flight_reference.to_csv(training_flight_reference_path, index = False)
    training_topic_reference.to_csv(training_topic_reference_path, index = False)
    training_column_reference.to_csv(training_column_reference_path, index = False)

    print(f"Saved : {training_flight_reference_path}")
    print(f"Saved : {training_topic_reference_path}")
    print(f"Saved : {training_column_reference_path}")
    print(
        "Flights available for anomaly model fitting : "
        f"{int(training_flight_reference['anomaly_model_fit_candidate'].sum())}"
    )
    print(
        "Flights available for anomaly evaluation : "
        f"{int(training_flight_reference['anomaly_evaluation_candidate'].sum())}"
    )
    print(
        "Flights available for fault classification : "
        f"{int(training_flight_reference['fault_classification_candidate'].sum())}"
    )
    print(f"Columns included for training : {len(training_column_reference)}")


def main() -> None:
    ensure_inputs_exist()

    flight_reference, topic_reference, feature_column_reference = load_inputs()

    training_flight_reference = build_training_flight_reference(flight_reference)
    training_topic_reference = build_training_topic_reference(
        topic_reference = topic_reference,
        feature_column_reference = feature_column_reference
    )
    training_column_reference = build_training_column_reference(feature_column_reference)

    write_outputs(
        training_flight_reference = training_flight_reference,
        training_topic_reference = training_topic_reference,
        training_column_reference = training_column_reference
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Training reference build failed : {exc}")
        sys.exit(1)