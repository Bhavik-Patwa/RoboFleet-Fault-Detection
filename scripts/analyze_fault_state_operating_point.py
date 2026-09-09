import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_ROOT = PROJECT_ROOT / "data" / "alfa" / "metadata"
TRACKING_DATABASE_PATH = PROJECT_ROOT / "mlflow.db"

TRAINING_FLIGHT_REFERENCE_PATH = (
    METADATA_ROOT / "training_flight_reference.csv"
)


# Parsing one MLflow run and one calibration alert budget
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        required = True
    )

    parser.add_argument(
        "--target-normal-flight-alert-rate",
        type = float,
        required = True
    )

    parser.add_argument(
        "--confidence-level",
        type = float,
        default = 0.95
    )

    return parser.parse_args()


# Validating the requested operating-point analysis
def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.target_normal_flight_alert_rate < 1.0:
        raise ValueError("Target normal-flight alert rate must be between zero and one.")

    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("Confidence level must be between zero and one.")


# Reading calibration alert rates from current and legacy artifacts
def get_calibration_alert_rate(record: dict) -> float:
    current_key = (
        "calibration_normal_state_flight_alert_rate"
    )

    legacy_key = (
        "calibration_normal_flight_alert_rate"
    )

    if current_key in record:
        return float(record[current_key])

    if legacy_key in record:
        return float(record[legacy_key])

    raise RuntimeError("Threshold record does not contain a recognized calibration alert-rate field.")


# Selecting the most sensitive threshold satisfying one calibration budget
def select_fold_operating_record(
    fold_records: list[dict],
    target_alert_rate: float
) -> dict:
    eligible_records = [
        record
        for record in fold_records
        if (
            get_calibration_alert_rate(record)
            <= target_alert_rate + 1e-12
        )
    ]

    if not eligible_records:
        raise RuntimeError("No threshold satisfies the requested calibration alert budget.")

    return min(
        eligible_records,
        key = lambda record: float(record["threshold"])
    )


# Calculating a Wilson confidence interval for a binomial rate
def calculate_wilson_interval(success_count: int, observation_count: int,
                              confidence_level: float
) -> tuple[float, float]:
    if observation_count <= 0:
        raise ValueError("Wilson interval requires at least one observation.")

    if not 0 <= success_count <= observation_count:
        raise ValueError("Success count must be between zero and the observation count.")

    z_score = NormalDist().inv_cdf(
        0.5 + confidence_level / 2.0
    )

    observed_rate = success_count / observation_count
    z_squared = z_score ** 2

    denominator = (
        1.0 + z_squared / observation_count
    )

    center = (
        observed_rate
        + z_squared / (2.0 * observation_count)
    ) / denominator

    margin = (
        z_score
        * math.sqrt(
            (
                observed_rate
                * (1.0 - observed_rate)
                / observation_count
            )
            + (
                z_squared
                / (4.0 * observation_count ** 2)
            )
        )
        / denominator
    )

    return (
        max(0.0, center - margin),
        min(1.0, center + margin)
    )


# Loading the evaluation and held-out prediction artifacts from MLflow
def load_run_artifacts(client: MlflowClient, run_id: str) -> tuple[str, dict, dict]:
    run = client.get_run(run_id)

    feature_set = run.data.params.get(
        "feature_set",
        "unknown"
    )

    evaluation_path = client.download_artifacts(
        run_id = run_id,
        path = "evaluation_metrics.json"
    )

    prediction_path = client.download_artifacts(
        run_id = run_id,
        path = "out_of_fold_predictions.json"
    )

    evaluation_metrics = json.loads(
        Path(evaluation_path).read_text()
    )

    predictions = json.loads(
        Path(prediction_path).read_text()
    )

    required_evaluation_keys = {
        "online_detection_threshold_metrics"
    }

    missing_evaluation_keys = (
        required_evaluation_keys.difference(
            evaluation_metrics
        )
    )

    if missing_evaluation_keys:
        raise RuntimeError(f"Evaluation artifact is missing fields : {sorted(missing_evaluation_keys)}")

    if "out_of_fold_window_predictions" not in predictions:
        raise RuntimeError("Prediction artifact does not contain held-out window predictions.")

    return feature_set, evaluation_metrics, predictions


# Selecting one independently calibrated operating threshold per outer fold
def build_fold_operating_records(evaluation_metrics: dict, target_alert_rate: float
) -> dict[int, dict]:
    threshold_records = evaluation_metrics[
        "online_detection_threshold_metrics"
    ]

    fold_numbers = sorted(
        {
            int(record["fold_number"])
            for record in threshold_records
        }
    )

    operating_records = {}

    for fold_number in fold_numbers:
        fold_records = [
            record
            for record in threshold_records
            if int(record["fold_number"]) == fold_number
        ]

        operating_records[fold_number] = (
            select_fold_operating_record(
                fold_records = fold_records,
                target_alert_rate = target_alert_rate
            )
        )

    return operating_records


# Retrieving the single non-missing fault family for one fault flight
def get_fault_family(flight_windows: pd.DataFrame) -> str:
    fault_families = sorted(
        flight_windows["fault_family_label"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(fault_families) != 1:
        raise RuntimeError("A fault flight does not contain exactly one fault family.")

    return fault_families[0]


# Building one held-out operating-point result for every flight
def build_flight_results(window_predictions: pd.DataFrame, flight_reference: pd.DataFrame,
                         operating_records: dict[int, dict]
) -> pd.DataFrame:
    if flight_reference["flight_name"].duplicated().any():
        raise RuntimeError("Training flight reference contains duplicate flights.")

    failure_times = flight_reference.set_index(
        "flight_name"
    )["first_fault_signal_time"]

    flight_results = []

    for flight_name, flight_windows in window_predictions.groupby(
        "flight_name",
        sort = True
    ):
        fold_numbers = flight_windows[
            "fold_number"
        ].unique()

        recording_dates = flight_windows[
            "held_out_recording_date"
        ].unique()

        if len(fold_numbers) != 1:
            raise RuntimeError(f"Flight '{flight_name}' appears in multiple folds.")

        if len(recording_dates) != 1:
            raise RuntimeError(f"Flight '{flight_name}' appears on multiple recording dates.")

        fold_number = int(fold_numbers[0])
        held_out_recording_date = str(recording_dates[0])

        if fold_number not in operating_records:
            raise RuntimeError(f"No operating threshold exists for fold {fold_number}.")

        threshold = float(
            operating_records[fold_number]["threshold"]
        )

        flight_windows = flight_windows.sort_values(
            "window_end_ns"
        )

        is_fault_flight = bool(
            flight_windows["window_label"].eq(
                "fault_state"
            ).any()
        )

        if not is_fault_flight:
            false_alert = bool(
                flight_windows["fault_state_score"]
                .ge(threshold)
                .any()
            )

            flight_results.append(
                {
                    "flight_name": str(flight_name),
                    "held_out_recording_date": (
                        held_out_recording_date
                    ),
                    "fold_number": fold_number,
                    "threshold": threshold,
                    "flight_type": "no_failure",
                    "fault_family_label": None,
                    "normal_state_false_alert": false_alert,
                    "pre_fault_false_alert": None,
                    "post_fault_detected": None,
                    "post_fault_detection_delay_seconds": None
                }
            )

            continue

        if flight_name not in failure_times.index:
            raise RuntimeError(f"Fault flight '{flight_name}' is missing from the training flight reference.")

        failure_time = failure_times.loc[flight_name]

        if pd.isna(failure_time):
            raise RuntimeError(f"Fault flight '{flight_name}' does not have a failure time.")

        failure_time_ns = int(
            round(float(failure_time))
        )

        pre_fault_windows = flight_windows.loc[
            flight_windows["window_label"].eq(
                "pre_fault_state"
            )
        ]

        if pre_fault_windows.empty:
            raise RuntimeError(
                f"Fault flight '{flight_name}' does not contain "
                "confirmed pre-fault windows."
            )

        pre_fault_false_alert = bool(
            pre_fault_windows["fault_state_score"]
            .ge(threshold)
            .any()
        )

        # Matching the online evaluator by allowing detection at the end of
        # the first window that finishes after the ground-truth fault signal.
        post_fault_windows = flight_windows.loc[
            flight_windows["window_end_ns"].gt(
                failure_time_ns
            )
        ]

        detection_windows = post_fault_windows.loc[
            post_fault_windows["fault_state_score"].ge(
                threshold
            )
        ]

        post_fault_detected = not detection_windows.empty
        detection_delay_seconds = None

        if post_fault_detected:
            first_detection_time_ns = int(
                detection_windows["window_end_ns"].iloc[0]
            )

            detection_delay_seconds = (
                first_detection_time_ns
                - failure_time_ns
            ) / 1e9

        flight_results.append(
            {
                "flight_name": str(flight_name),
                "held_out_recording_date": (
                    held_out_recording_date
                ),
                "fold_number": fold_number,
                "threshold": threshold,
                "flight_type": "fault",
                "fault_family_label": get_fault_family(
                    flight_windows
                ),
                "normal_state_false_alert": (
                    pre_fault_false_alert
                ),
                "pre_fault_false_alert": (
                    pre_fault_false_alert
                ),
                "post_fault_detected": (
                    post_fault_detected
                ),
                "post_fault_detection_delay_seconds": (
                    detection_delay_seconds
                )
            }
        )

    return pd.DataFrame(flight_results)


# Confirming that reconstructed results match the stored fold evaluation
def validate_reconstructed_results(flight_results: pd.DataFrame, operating_records: dict[int, dict]
) -> None:
    for fold_number, operating_record in operating_records.items():
        fold_results = flight_results.loc[
            flight_results["fold_number"].eq(
                fold_number
            )
        ]

        normal_results = fold_results.loc[
            fold_results["flight_type"].eq(
                "no_failure"
            )
        ]

        fault_results = fold_results.loc[
            fold_results["flight_type"].eq(
                "fault"
            )
        ]

        reconstructed_values = {
            "normal_flight_count": int(
                len(normal_results)
            ),
            "normal_flight_false_alert_count": int(
                normal_results[
                    "normal_state_false_alert"
                ].sum()
            ),
            "fault_flight_count": int(
                len(fault_results)
            ),
            "fault_flight_detection_count": int(
                fault_results[
                    "post_fault_detected"
                ].sum()
            ),
            "pre_observed_signal_evaluation_count": int(
                fault_results[
                    "pre_fault_false_alert"
                ].notna().sum()
            ),
            "pre_observed_signal_alert_count": int(
                fault_results[
                    "pre_fault_false_alert"
                ].sum()
            )
        }

        mismatches = {
            field_name: {
                "reconstructed": reconstructed_value,
                "stored": int(
                    operating_record[field_name]
                )
            }
            for field_name, reconstructed_value
            in reconstructed_values.items()
            if (
                reconstructed_value
                != int(operating_record[field_name])
            )
        }

        if mismatches:
            raise RuntimeError(f"Fold {fold_number} reconstruction does not match stored evaluation metrics : {mismatches}")


# Summarizing one success rate with its confidence interval
def build_rate_summary(metric_name: str, success_count: int,
                       observation_count: int, confidence_level: float
) -> dict:
    interval_lower, interval_upper = (
        calculate_wilson_interval(
            success_count = success_count,
            observation_count = observation_count,
            confidence_level = confidence_level
        )
    )

    return {
        "metric": metric_name,
        "count": f"{success_count}/{observation_count}",
        "rate": success_count / observation_count,
        "confidence_interval_lower": interval_lower,
        "confidence_interval_upper": interval_upper
    }


# Building overall detection and false-alert results
def build_overall_summary(flight_results: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    normal_results = flight_results.loc[
        flight_results["flight_type"].eq(
            "no_failure"
        )
    ]

    fault_results = flight_results.loc[
        flight_results["flight_type"].eq(
            "fault"
        )
    ]

    no_failure_false_alert_count = int(
        normal_results[
            "normal_state_false_alert"
        ].sum()
    )

    pre_fault_false_alert_count = int(
        fault_results[
            "pre_fault_false_alert"
        ].sum()
    )

    detected_fault_count = int(
        fault_results[
            "post_fault_detected"
        ].sum()
    )

    combined_normal_state_false_alert_count = (
        no_failure_false_alert_count
        + pre_fault_false_alert_count
    )

    combined_normal_state_period_count = (
        len(normal_results)
        + len(fault_results)
    )

    summary_records = [
        build_rate_summary(
            metric_name = "post_fault_detection_rate",
            success_count = detected_fault_count,
            observation_count = len(fault_results),
            confidence_level = confidence_level
        ),
        build_rate_summary(
            metric_name = "normal_state_false_alert_rate",
            success_count = (
                combined_normal_state_false_alert_count
            ),
            observation_count = (
                combined_normal_state_period_count
            ),
            confidence_level = confidence_level
        ),
        build_rate_summary(
            metric_name = (
                "no_failure_flight_false_alert_rate"
            ),
            success_count = (
                no_failure_false_alert_count
            ),
            observation_count = len(normal_results),
            confidence_level = confidence_level
        ),
        build_rate_summary(
            metric_name = (
                "pre_fault_period_false_alert_rate"
            ),
            success_count = (
                pre_fault_false_alert_count
            ),
            observation_count = len(fault_results),
            confidence_level = confidence_level
        )
    ]

    return pd.DataFrame(summary_records)


# Summarizing held-out results independently for every recording date
def build_recording_date_summary(flight_results: pd.DataFrame) -> pd.DataFrame:
    summary_records = []

    for recording_date, date_results in flight_results.groupby(
        "held_out_recording_date",
        sort = True
    ):
        fault_results = date_results.loc[
            date_results["flight_type"].eq(
                "fault"
            )
        ]

        summary_records.append(
            {
                "held_out_recording_date": recording_date,
                "normal_state_period_count": int(
                    len(date_results)
                ),
                "normal_state_false_alert_count": int(
                    date_results[
                        "normal_state_false_alert"
                    ].sum()
                ),
                "normal_state_false_alert_rate": float(
                    date_results[
                        "normal_state_false_alert"
                    ].mean()
                ),
                "fault_flight_count": int(
                    len(fault_results)
                ),
                "detected_fault_flight_count": int(
                    fault_results[
                        "post_fault_detected"
                    ].sum()
                ),
                "post_fault_detection_rate": float(
                    fault_results[
                        "post_fault_detected"
                    ].mean()
                ),
                "mean_detection_delay_seconds": (
                    float(
                        fault_results[
                            "post_fault_detection_delay_seconds"
                        ].mean()
                    )
                    if fault_results[
                        "post_fault_detection_delay_seconds"
                    ].notna().any()
                    else None
                )
            }
        )

    return pd.DataFrame(summary_records)


# Summarizing detection results independently for every fault family
def build_fault_family_summary(flight_results: pd.DataFrame) -> pd.DataFrame:
    fault_results = flight_results.loc[
        flight_results["flight_type"].eq(
            "fault"
        )
    ]

    summary = (
        fault_results
        .groupby(
            "fault_family_label",
            as_index = False
        )
        .agg(
            fault_flight_count = (
                "flight_name",
                "size"
            ),
            detected_fault_flight_count = (
                "post_fault_detected",
                "sum"
            ),
            post_fault_detection_rate = (
                "post_fault_detected",
                "mean"
            ),
            mean_detection_delay_seconds = (
                "post_fault_detection_delay_seconds",
                "mean"
            ),
            maximum_detection_delay_seconds = (
                "post_fault_detection_delay_seconds",
                "max"
            )
        )
        .sort_values(
            [
                "post_fault_detection_rate",
                "fault_family_label"
            ]
        )
        .reset_index(drop = True)
    )

    # Normalizing aggregated count columns for consistent integer reporting
    summary["fault_flight_count"] = (
        summary["fault_flight_count"].astype("int64")
    )

    summary["detected_fault_flight_count"] = (
        summary["detected_fault_flight_count"].astype("int64")
    )

    return summary


def main() -> None:
    args = parse_args()

    validate_args(args)

    if not TRAINING_FLIGHT_REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Required input path not found : {TRAINING_FLIGHT_REFERENCE_PATH}")

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DATABASE_PATH}")

    client = MlflowClient()

    (
        feature_set,
        evaluation_metrics,
        predictions
    ) = load_run_artifacts(
        client = client,
        run_id = args.run_id
    )

    operating_records = build_fold_operating_records(
        evaluation_metrics = evaluation_metrics,
        target_alert_rate = (
            args.target_normal_flight_alert_rate
        )
    )

    window_predictions = pd.DataFrame(
        predictions["out_of_fold_window_predictions"]
    )

    if window_predictions.empty:
        raise RuntimeError(
            "Held-out window predictions are empty."
        )

    flight_reference = pd.read_csv(
        TRAINING_FLIGHT_REFERENCE_PATH
    )

    flight_results = build_flight_results(
        window_predictions = window_predictions,
        flight_reference = flight_reference,
        operating_records = operating_records
    )

    validate_reconstructed_results(
        flight_results = flight_results,
        operating_records = operating_records
    )

    overall_summary = build_overall_summary(
        flight_results = flight_results,
        confidence_level = args.confidence_level
    )

    recording_date_summary = build_recording_date_summary(
        flight_results = flight_results
    )

    fault_family_summary = build_fault_family_summary(
        flight_results = flight_results
    )

    # Selecting fault flights that were not detected after activation
    missed_fault_flights = (
        flight_results.loc[
            flight_results["flight_type"].eq("fault")
            & flight_results["post_fault_detected"].eq(False),
            [
                "flight_name",
                "held_out_recording_date",
                "fault_family_label",
                "threshold"
            ]
        ]
        .sort_values(
            [
                "held_out_recording_date",
                "fault_family_label",
                "flight_name"
            ]
        )
    )

    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        240
    )

    float_formatter = (
        lambda value: f"{value:.4f}"
    )

    print(f"Run ID : {args.run_id}")
    print(f"Feature set : {feature_set}")

    print(
        "Target calibration normal-state alert rate : "
        f"{args.target_normal_flight_alert_rate:.4f}"
    )

    print("\nOverall held-out performance :")

    print(
        overall_summary.to_string(
            index = False,
            float_format = float_formatter
        )
    )

    print("\nHeld-out performance by recording date :")

    print(
        recording_date_summary.to_string(
            index = False,
            float_format = float_formatter
        )
    )

    print("\nHeld-out performance by fault family :")

    print(
        fault_family_summary.to_string(
            index = False,
            float_format = float_formatter
        )
    )

    print("\nMissed fault flights :")

    if missed_fault_flights.empty:
        print("None")

    else:
        print(
            missed_fault_flights.to_string(
                index = False,
                float_format = float_formatter
            )
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            "Fault-state operating-point analysis failed : "
            f"{exc}"
        )

        sys.exit(1)