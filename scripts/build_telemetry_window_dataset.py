import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "alfa" / "processed"
METADATA_ROOT = PROJECT_ROOT / "data" / "alfa" / "metadata"
CURATED_ROOT = PROJECT_ROOT / "data" / "alfa" / "curated"

WINDOW_DURATION_SECONDS = 7
WINDOW_STEP_SECONDS = 1

WINDOW_STATISTICS = (
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
    "first",
    "last",
    "change"
)

# Reserving initial windows as a causal flight-specific reference period
INITIAL_BASELINE_WINDOW_COUNT = 10
TRAILING_WINDOW_COUNT = 20

# Shifting history so its source interval ends before the current window begins
NON_OVERLAPPING_WINDOW_LAG = (
    WINDOW_DURATION_SECONDS // WINDOW_STEP_SECONDS
)

COMMAND_RESPONSE_TOPICS = {
    "mavros-nav_info-airspeed.csv": (
        "field.commanded",
        "field.measured"
    ),
    "mavros-nav_info-pitch.csv": (
        "field.commanded",
        "field.measured"
    ),
    "mavros-nav_info-roll.csv": (
        "field.commanded",
        "field.measured"
    ),
    "mavros-nav_info-yaw.csv": (
        "field.commanded",
        "field.measured"
    )
}

COMMAND_RESPONSE_STATISTICS = (
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
    "first",
    "last",
    "change",
    "root_mean_square",
    "mean_absolute",
    "maximum_absolute"
)


# Verifying that all required source and reference paths exist
def ensure_inputs_exist() -> None:
    required_paths = [
        PROCESSED_ROOT,
        METADATA_ROOT / "processed_dataset_file_profile.csv",
        METADATA_ROOT / "training_flight_reference.csv",
        METADATA_ROOT / "training_column_reference.csv"
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

        raise FileNotFoundError(
            f"Required input paths not found : \n{missing_text}"
        )


# Loading canonical file, flight and feature references
def load_reference_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:
    file_profile = pd.read_csv(
        METADATA_ROOT / "processed_dataset_file_profile.csv"
    )

    flight_reference = pd.read_csv(
        METADATA_ROOT / "training_flight_reference.csv"
    )

    column_reference = pd.read_csv(
        METADATA_ROOT / "training_column_reference.csv"
    )

    if column_reference.empty:
        raise RuntimeError("Training column reference does not contain feature columns.")

    return file_profile, flight_reference, column_reference


# Grouping selected numeric feature columns by telemetry topic
def build_feature_columns_by_topic(column_reference: pd.DataFrame) -> dict[str, list[str]]:
    return (
        column_reference
        .groupby("topic_name")["column_name"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )


# Indexing the unique source file and timestamp column for each flight topic
def build_source_file_index(file_profile: pd.DataFrame, 
                            columns_by_topic: dict[str, list[str]]
) -> dict[tuple[str, str], tuple[str, str]]:
    source_files = file_profile.loc[
        file_profile["topic_name"].isin(columns_by_topic),
        [
            "flight_name",
            "topic_name",
            "file_name",
            "timestamp_column"
        ]
    ].drop_duplicates()

    duplicate_sources = source_files.duplicated(
        ["flight_name", "topic_name"],
        keep = False
    )

    if duplicate_sources.any():
        duplicate_rows = source_files.loc[
            duplicate_sources,
            ["flight_name", "topic_name"]
        ]

        raise RuntimeError(
            "Multiple source files found for the same flight and topic : \n"
            f"{duplicate_rows.to_string(index = False)}"
        )

    return {
        (row["flight_name"], row["topic_name"]): (
            row["file_name"],
            row["timestamp_column"]
        )
        for row in source_files.to_dict(orient = "records")
    }


# Converting source identifiers into deterministic feature names
def normalize_feature_name(value: str) -> str:
    normalized_value = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        value
    ).strip("_").lower()
    if not normalized_value:
        raise ValueError(f"Unable to create a feature name from : {value}")

    return normalized_value


# Mapping every telemetry topic and column to one unique feature prefix
def build_feature_name_index(columns_by_topic: dict[str, list[str]]) -> dict[tuple[str, str], str]:
    feature_name_index = {}
    source_by_feature_name = {}

    for topic_name in sorted(columns_by_topic):
        topic_stem = topic_name.removesuffix(".csv")

        for column_name in columns_by_topic[topic_name]:
            feature_name = normalize_feature_name(
                f"{topic_stem}__{column_name}"
            )

            source = (
                topic_name,
                column_name
            )

            existing_source = source_by_feature_name.get(
                feature_name
            )

            if (
                existing_source is not None
                and existing_source != source
            ):
                raise RuntimeError(f"Feature-name collision detected between {existing_source} and {source}."
                )

            source_by_feature_name[feature_name] = source
            feature_name_index[source] = feature_name

    return feature_name_index


# Loading and validating all selected telemetry sources for one flight
def load_flight_sources(flight_name: str, columns_by_topic: dict[str, list[str]],
                        source_file_index: dict[tuple[str, str], tuple[str, str]]
) -> dict[str, pd.DataFrame]:
    source_frames = {}

    for topic_name, column_names in columns_by_topic.items():
        source_details = source_file_index.get(
            (flight_name, topic_name)
        )

        if source_details is None:
            raise FileNotFoundError(f"Missing source file for flight '{flight_name}' and topic '{topic_name}'.")

        file_name, timestamp_column = source_details
        source_path = (
            PROCESSED_ROOT
            / flight_name
            / file_name
        )

        source_frame = pd.read_csv(
            source_path,
            usecols = [
                timestamp_column,
                *column_names
            ]
        )

        timestamp_series = pd.to_numeric(
            source_frame[timestamp_column],
            errors = "coerce"
        )

        if timestamp_series.isna().any():
            raise RuntimeError(f"Invalid timestamp values found in : {source_path}")

        source_frame = source_frame.drop(
            columns = timestamp_column
        )

        for column_name in column_names:
            numeric_series = pd.to_numeric(
                source_frame[column_name],
                errors = "coerce"
            )

            invalid_value_count = int(
                source_frame[column_name].notna().sum()
                - numeric_series.notna().sum()
            )

            if invalid_value_count > 0:
                raise RuntimeError(f"Non-numeric values found in '{column_name}' from : {source_path}")

            source_frame[column_name] = numeric_series

        source_frame.insert(
            0,
            "timestamp_ns",
            timestamp_series.astype("int64")
        )

        source_frame = source_frame.sort_values(
            "timestamp_ns"
        ).reset_index(drop = True)

        if not source_frame[
            "timestamp_ns"
        ].is_monotonic_increasing:
            raise RuntimeError(f"Timestamps are not monotonic in : {source_path}")

        source_frames[topic_name] = source_frame

    return source_frames


# Finding the telemetry interval shared by every selected topic
def get_common_time_bounds(source_frames: dict[str, pd.DataFrame]) -> tuple[int, int]:
    source_start_times = [
        int(source_frame["timestamp_ns"].iloc[0])
        for source_frame in source_frames.values()
    ]

    source_end_times = [
        int(source_frame["timestamp_ns"].iloc[-1])
        for source_frame in source_frames.values()
    ]

    flight_start_ns = max(source_start_times)
    flight_end_ns = min(source_end_times)

    if flight_end_ns <= flight_start_ns:
        raise RuntimeError("No common telemetry interval found.")

    return flight_start_ns, flight_end_ns


# Adding standard summary statistics for one numeric telemetry series
def add_series_statistics(record: dict, feature_name: str, values: pd.Series) -> None:
    if values.empty:
        return

    record[f"{feature_name}__mean"] = float(
        values.mean()
    )

    record[f"{feature_name}__minimum"] = float(
        values.min()
    )

    record[f"{feature_name}__maximum"] = float(
        values.max()
    )

    record[f"{feature_name}__first"] = float(
        values.iloc[0]
    )

    record[f"{feature_name}__last"] = float(
        values.iloc[-1]
    )

    record[f"{feature_name}__change"] = float(
        values.iloc[-1] - values.iloc[0]
    )

    if len(values) > 1:
        record[
            f"{feature_name}__standard_deviation"
        ] = float(
            values.std(ddof = 1)
        )


# Measuring disagreement between commanded and measured aircraft behavior
def add_command_response_features(record: dict, topic_name: str,
                                  window_frame: pd.DataFrame
) -> None:
    command_response_columns = (
        COMMAND_RESPONSE_TOPICS.get(topic_name)
    )

    if command_response_columns is None:
        return

    commanded_column, measured_column = (
        command_response_columns
    )

    paired_values = window_frame[
        [
            commanded_column,
            measured_column
        ]
    ].dropna()

    if paired_values.empty:
        return

    errors = (
        paired_values[measured_column]
        - paired_values[commanded_column]
    )

    # Computing the shortest signed yaw difference across the -180/180 boundary
    if topic_name == "mavros-nav_info-yaw.csv":
        errors = (
            (errors + 180.0) % 360.0
        ) - 180.0

    feature_name = normalize_feature_name(
        f"{topic_name.removesuffix('.csv')}"
        "__command_response_error"
    )

    add_series_statistics(
        record = record,
        feature_name = feature_name,
        values = errors
    )

    record[
        f"{feature_name}__root_mean_square"
    ] = float(
        errors.pow(2).mean() ** 0.5
    )

    record[
        f"{feature_name}__mean_absolute"
    ] = float(
        errors.abs().mean()
    )

    record[
        f"{feature_name}__maximum_absolute"
    ] = float(
        errors.abs().max()
    )


# Extracting telemetry and command-response features for one window
def add_window_features(record: dict, source_frames: dict[str, pd.DataFrame],
                        columns_by_topic: dict[str, list[str]], feature_name_index: dict[tuple[str, str], str],
                        window_start_ns: int, window_end_ns: int
) -> None:
    for topic_name, source_frame in source_frames.items():
        timestamps = source_frame["timestamp_ns"]

        start_index = timestamps.searchsorted(
            window_start_ns,
            side = "left"
        )

        end_index = timestamps.searchsorted(
            window_end_ns,
            side = "left"
        )

        window_frame = source_frame.iloc[
            start_index:end_index
        ][columns_by_topic[topic_name]]

        for column_name in columns_by_topic[topic_name]:
            feature_name = feature_name_index[
                (
                    topic_name,
                    column_name
                )
            ]

            values = window_frame[
                column_name
            ].dropna()

            add_series_statistics(
                record = record,
                feature_name = feature_name,
                values = values
            )

        add_command_response_features(
            record = record,
            topic_name = topic_name,
            window_frame = window_frame
        )


# Assigning labels using the beginning of each complete telemetry window
def get_window_labels(flight: dict, window_start_ns: int) -> tuple[str, str | None]:
    if flight["fault_name"] == "no_failure":
        return "normal", None

    first_fault_signal_time = (
        flight["first_fault_signal_time"]
    )

    if (
        pd.notna(first_fault_signal_time)
        and window_start_ns
        >= int(round(first_fault_signal_time))
    ):
        return (
            "fault_state",
            flight["fault_family"]
        )

    return "unlabeled", None


# Building deterministic names for raw and command-response features
def build_feature_column_names(feature_name_index: dict[tuple[str, str], str]) -> list[str]:
    feature_columns = []

    for feature_name in sorted(
        feature_name_index.values()
    ):
        for statistic_name in WINDOW_STATISTICS:
            feature_columns.append(
                f"{feature_name}__{statistic_name}"
            )

    for topic_name in sorted(
        COMMAND_RESPONSE_TOPICS
    ):
        feature_name = normalize_feature_name(
            f"{topic_name.removesuffix('.csv')}"
            "__command_response_error"
        )

        for statistic_name in (
            COMMAND_RESPONSE_STATISTICS
        ):
            feature_columns.append(
                f"{feature_name}__{statistic_name}"
            )

    return feature_columns


# Verifying that the causal baseline ends before any observed fault signal
def validate_initial_baseline_period(window_dataset: pd.DataFrame, flight_reference: pd.DataFrame) -> None:
    ordered_windows = window_dataset.sort_values(
        [
            "flight_name",
            "window_start_ns"
        ]
    )

    baseline_windows = (
        ordered_windows
        .groupby(
            "flight_name",
            sort = False
        )
        .head(
            INITIAL_BASELINE_WINDOW_COUNT
        )
    )

    baseline_summary = (
        baseline_windows
        .groupby(
            "flight_name",
            as_index = False
        )
        .agg(
            baseline_window_count = (
                "window_start_ns",
                "size"
            ),
            baseline_end_ns = (
                "window_end_ns",
                "max"
            )
        )
    )

    incomplete_baselines = baseline_summary.loc[
        baseline_summary[
            "baseline_window_count"
        ].lt(
            INITIAL_BASELINE_WINDOW_COUNT
        )
    ]

    if not incomplete_baselines.empty:
        raise RuntimeError("Flights do not contain enough initial baseline windows.")

    fault_reference = flight_reference.loc[
        flight_reference["fault_name"].ne(
            "no_failure"
        ),
        [
            "flight_name",
            "first_fault_signal_time"
        ]
    ]

    baseline_summary = baseline_summary.merge(
        fault_reference,
        on = "flight_name",
        how = "inner",
        validate = "one_to_one"
    )

    invalid_baselines = baseline_summary.loc[
        baseline_summary[
            "baseline_end_ns"
        ].gt(
            baseline_summary[
                "first_fault_signal_time"
            ]
        )
    ]

    if not invalid_baselines.empty:
        invalid_flights = sorted(
            invalid_baselines[
                "flight_name"
            ].astype(str)
        )

        raise RuntimeError(f"Initial baseline overlaps an observed fault period : {invalid_flights}")


# Adding features relative to causal history from the same flight
def add_causal_relative_features(window_dataset: pd.DataFrame, source_feature_columns: list[str]) -> pd.DataFrame:
    window_dataset = window_dataset.sort_values(
        [
            "flight_name",
            "window_start_ns"
        ]
    ).reset_index(drop = True)

    minimum_required_window_count = (
        INITIAL_BASELINE_WINDOW_COUNT
        + NON_OVERLAPPING_WINDOW_LAG
    )

    flight_window_counts = (
        window_dataset[
            "flight_name"
        ].value_counts()
    )

    insufficient_flights = sorted(
        flight_window_counts.loc[
            flight_window_counts.lt(
                minimum_required_window_count
            )
        ].index.astype(str)
    )

    if insufficient_flights:
        raise RuntimeError(f"Flights do not contain enough windows for causal features : {insufficient_flights}")

    level_feature_columns = [
        column_name
        for column_name in source_feature_columns
        if column_name.endswith("__mean")
    ]

    grouped_windows = window_dataset.groupby(
        "flight_name",
        sort = False
    )

    relative_feature_data = {}

    for column_name in level_feature_columns:
        baseline_values = grouped_windows[
            column_name
        ].transform(
            lambda values: values.iloc[
                :INITIAL_BASELINE_WINDOW_COUNT
            ].median()
        )

        trailing_values = grouped_windows[
            column_name
        ].transform(
            lambda values: (
                values
                .shift(
                    NON_OVERLAPPING_WINDOW_LAG
                )
                .rolling(
                    window = TRAILING_WINDOW_COUNT,
                    min_periods = (
                        INITIAL_BASELINE_WINDOW_COUNT
                    )
                )
                .median()
            )
        )

        baseline_feature_name = (
            f"{column_name}"
            "__initial_baseline_delta"
        )

        trailing_feature_name = (
            f"{column_name}"
            "__trailing_median_delta"
        )

        relative_feature_data[
            baseline_feature_name
        ] = (
            window_dataset[column_name]
            - baseline_values
        )

        relative_feature_data[
            trailing_feature_name
        ] = (
            window_dataset[column_name]
            - trailing_values
        )

    relative_features = pd.DataFrame(
        relative_feature_data,
        index = window_dataset.index
    )

    window_dataset = pd.concat(
        [
            window_dataset,
            relative_features
        ],
        axis = 1
    )

    # Removing warm-up windows that do not yet have causal history
    flight_window_numbers = (
        grouped_windows.cumcount()
    )

    first_usable_window_number = (
        INITIAL_BASELINE_WINDOW_COUNT
        + NON_OVERLAPPING_WINDOW_LAG
        - 1
    )

    window_dataset = window_dataset.loc[
        flight_window_numbers.ge(
            first_usable_window_number
        )
    ].reset_index(drop = True)

    relative_feature_columns = list(
        relative_feature_data
    )

    if window_dataset[
        relative_feature_columns
    ].isna().any().any():
        raise RuntimeError("Causal relative features contain missing values.")

    return window_dataset


# Building complete telemetry windows for every usable flight
def build_window_dataset(flight_reference: pd.DataFrame, columns_by_topic: dict[str, list[str]],
                         source_file_index: dict[tuple[str, str],tuple[str, str]], 
                         feature_name_index: dict[tuple[str, str], str]
) -> pd.DataFrame:
    usable_flights = (
        flight_reference.loc[
            flight_reference[
                "supervision_status"
            ].eq("usable")
        ]
        .sort_values("flight_name")
    )

    window_duration_ns = int(
        WINDOW_DURATION_SECONDS * 1e9
    )

    window_step_ns = int(
        WINDOW_STEP_SECONDS * 1e9
    )

    window_records = []

    for flight in usable_flights.to_dict(
        orient = "records"
    ):
        source_frames = load_flight_sources(
            flight_name = flight[
                "flight_name"
            ],
            columns_by_topic = columns_by_topic,
            source_file_index = (
                source_file_index
            )
        )

        (
            flight_start_ns,
            flight_end_ns
        ) = get_common_time_bounds(
            source_frames = source_frames
        )

        final_window_start_ns = (
            flight_end_ns
            - window_duration_ns
        )

        window_start_ns = flight_start_ns

        while (
            window_start_ns
            <= final_window_start_ns
        ):
            window_end_ns = (
                window_start_ns
                + window_duration_ns
            )

            (
                window_label,
                fault_family_label
            ) = get_window_labels(
                flight = flight,
                window_start_ns = (
                    window_start_ns
                )
            )

            record = {
                "flight_name": flight[
                    "flight_name"
                ],
                "window_start_ns": (
                    window_start_ns
                ),
                "window_end_ns": (
                    window_end_ns
                ),
                "window_duration_seconds": (
                    WINDOW_DURATION_SECONDS
                ),
                "window_label": window_label,
                "fault_family_label": (
                    fault_family_label
                )
            }

            add_window_features(
                record = record,
                source_frames = source_frames,
                columns_by_topic = (
                    columns_by_topic
                ),
                feature_name_index = (
                    feature_name_index
                ),
                window_start_ns = (
                    window_start_ns
                ),
                window_end_ns = window_end_ns
            )

            window_records.append(record)

            window_start_ns += (
                window_step_ns
            )

    if not window_records:
        raise RuntimeError("No telemetry windows were created.")

    metadata_columns = [
        "flight_name",
        "window_start_ns",
        "window_end_ns",
        "window_duration_seconds",
        "window_label",
        "fault_family_label"
    ]

    source_feature_columns = (
        build_feature_column_names(
            feature_name_index
        )
    )

    window_dataset = pd.DataFrame(
        window_records
    ).reindex(
        columns = [
            *metadata_columns,
            *source_feature_columns
        ]
    )

    validate_initial_baseline_period(
        window_dataset = window_dataset,
        flight_reference = flight_reference
    )

    window_dataset = (
        add_causal_relative_features(
            window_dataset = window_dataset,
            source_feature_columns = (
                source_feature_columns
            )
        )
    )

    relative_feature_columns = [
        column_name
        for column_name
        in window_dataset.columns
        if (
            column_name.endswith(
                "__initial_baseline_delta"
            )
            or column_name.endswith(
                "__trailing_median_delta"
            )
        )
    ]

    return window_dataset.reindex(
        columns = [
            *metadata_columns,
            *source_feature_columns,
            *relative_feature_columns
        ]
    )


# Validating the completed telemetry window dataset
def validate_window_dataset(window_dataset: pd.DataFrame, flight_reference: pd.DataFrame,
                            columns_by_topic: dict[str, list[str]]
) -> None:
    if window_dataset.duplicated(
        [
            "flight_name",
            "window_start_ns"
        ]
    ).any():
        raise RuntimeError("Duplicate telemetry windows were created.")

    expected_flights = set(
        flight_reference.loc[
            flight_reference[
                "supervision_status"
            ].eq("usable"),
            "flight_name"
        ]
    )

    actual_flights = set(
        window_dataset["flight_name"]
    )

    if actual_flights != expected_flights:
        raise RuntimeError("Telemetry windows do not cover every usable flight.")

    if any(
        topic_name.startswith(
            "failure_status-"
        )
        for topic_name in columns_by_topic
    ):
        raise RuntimeError("Label-source topics are present in the feature dataset.")

    if (
        "emergency_responder-traj_file.csv"
        in columns_by_topic
    ):
        raise RuntimeError("Recovery artifacts are present in the feature dataset.")

    allowed_labels = {
        "normal",
        "fault_state",
        "unlabeled"
    }

    unexpected_labels = (
        set(window_dataset["window_label"])
        - allowed_labels
    )

    if unexpected_labels:
        raise RuntimeError(f"Unexpected window labels found : {sorted(unexpected_labels)}")

    normal_flights = set(
        flight_reference.loc[
            flight_reference[
                "fault_name"
            ].eq("no_failure"),
            "flight_name"
        ]
    )

    normal_window_flights = set(
        window_dataset.loc[
            window_dataset[
                "window_label"
            ].eq("normal"),
            "flight_name"
        ]
    )

    if normal_window_flights != normal_flights:
        raise RuntimeError("Normal windows do not match the no-failure flight cohort.")

    expected_fault_state_flights = set(
        flight_reference.loc[
            flight_reference[
                "fault_classification_candidate"
            ].eq(True),
            "flight_name"
        ]
    )

    actual_fault_state_flights = set(
        window_dataset.loc[
            window_dataset[
                "window_label"
            ].eq("fault_state"),
            "flight_name"
        ]
    )

    if (
        actual_fault_state_flights
        != expected_fault_state_flights
    ):
        raise RuntimeError("At least one fault flight has no complete observed fault-state window.")

    invalid_fault_family_labels = (
        window_dataset.loc[
            window_dataset[
                "window_label"
            ].ne("fault_state")
            & window_dataset[
                "fault_family_label"
            ].notna()
        ]
    )

    if not invalid_fault_family_labels.empty:
        raise RuntimeError("Fault-family labels are present outside observed fault-state windows.")


# Calculating the SHA256 digest for one reference file
def calculate_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


# Writing the versioned dataset and reproducibility manifest atomically
def write_outputs(window_dataset: pd.DataFrame, columns_by_topic: dict[str, list[str]],
                  feature_name_index: dict[tuple[str, str], str]
) -> None:
    CURATED_ROOT.mkdir(
        parents = True,
        exist_ok = True
    )

    dataset_path = (
        CURATED_ROOT
        / "telemetry_windows_v2.parquet"
    )

    manifest_path = (
        CURATED_ROOT
        / "telemetry_windows_v2_manifest.json"
    )

    temporary_dataset_path = (
        CURATED_ROOT
        / "telemetry_windows_v2.tmp.parquet"
    )

    temporary_manifest_path = (
        CURATED_ROOT
        / "telemetry_windows_v2.tmp.json"
    )

    window_label_counts = {
        label: int(count)
        for label, count in (
            window_dataset[
                "window_label"
            ]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    metadata_columns = {
        "flight_name",
        "window_start_ns",
        "window_end_ns",
        "window_duration_seconds",
        "window_label",
        "fault_family_label"
    }

    feature_column_count = len(
        [
            column_name
            for column_name
            in window_dataset.columns
            if column_name
            not in metadata_columns
        ]
    )

    manifest = {
        "window_duration_seconds": (
            WINDOW_DURATION_SECONDS
        ),
        "window_step_seconds": (
            WINDOW_STEP_SECONDS
        ),
        "window_statistics": list(
            WINDOW_STATISTICS
        ),
        "source_topic_count": len(
            columns_by_topic
        ),
        "source_column_count": sum(
            len(column_names)
            for column_names
            in columns_by_topic.values()
        ),
        "feature_column_count": (
            feature_column_count
        ),
        "initial_baseline_window_count": (
            INITIAL_BASELINE_WINDOW_COUNT
        ),
        "trailing_window_count": (
            TRAILING_WINDOW_COUNT
        ),
        "non_overlapping_window_lag": (
            NON_OVERLAPPING_WINDOW_LAG
        ),
        "command_response_topics": sorted(
            COMMAND_RESPONSE_TOPICS
        ),
        "window_count": len(
            window_dataset
        ),
        "window_label_counts": (
            window_label_counts
        ),
        "reference_file_sha256": {
            "processed_dataset_file_profile.csv": (
                calculate_file_sha256(
                    METADATA_ROOT
                    / "processed_dataset_file_profile.csv"
                )
            ),
            "training_column_reference.csv": (
                calculate_file_sha256(
                    METADATA_ROOT
                    / "training_column_reference.csv"
                )
            ),
            "training_flight_reference.csv": (
                calculate_file_sha256(
                    METADATA_ROOT
                    / "training_flight_reference.csv"
                )
            )
        }
    }

    window_dataset.to_parquet(
        temporary_dataset_path,
        index = False
    )

    temporary_manifest_path.write_text(
        json.dumps(
            manifest,
            indent = 2,
            sort_keys = True
        ) + "\n"
    )

    temporary_dataset_path.replace(
        dataset_path
    )

    temporary_manifest_path.replace(
        manifest_path
    )

    print(f"Saved : {dataset_path}")
    print(f"Saved : {manifest_path}")
    print(f"Telemetry windows created : {len(window_dataset)}")


# Building, validating and saving the telemetry window dataset
def main() -> None:
    ensure_inputs_exist()

    (
        file_profile,
        flight_reference,
        column_reference
    ) = load_reference_data()

    columns_by_topic = (
        build_feature_columns_by_topic(
            column_reference = (
                column_reference
            )
        )
    )

    source_file_index = (
        build_source_file_index(
            file_profile = file_profile,
            columns_by_topic = (
                columns_by_topic
            )
        )
    )

    feature_name_index = (
        build_feature_name_index(
            columns_by_topic = (
                columns_by_topic
            )
        )
    )

    window_dataset = build_window_dataset(
        flight_reference = flight_reference,
        columns_by_topic = columns_by_topic,
        source_file_index = source_file_index,
        feature_name_index = feature_name_index
    )

    validate_window_dataset(
        window_dataset = window_dataset,
        flight_reference = flight_reference,
        columns_by_topic = columns_by_topic
    )

    write_outputs(
        window_dataset = window_dataset,
        columns_by_topic = columns_by_topic,
        feature_name_index = feature_name_index
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Telemetry window dataset build failed : {exc}")
        sys.exit(1)