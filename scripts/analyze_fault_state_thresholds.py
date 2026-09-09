import argparse
import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_DATABASE_PATH = PROJECT_ROOT / "mlflow.db"


# Parsing MLflow runs and calibration alert budgets for comparison
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        action = "append",
        required = True
    )

    parser.add_argument(
        "--target-normal-flight-alert-rate",
        action = "append",
        type = float,
        default = None
    )

    parser.add_argument(
        "--include-all-achievable-alert-rates",
        action = "store_true"
    )

    return parser.parse_args()


# Validating explicit alert budgets and requiring one analysis mode
def validate_alert_rate_arguments(target_alert_rates: list[float], include_all_achievable_alert_rates: bool) -> None:
    if (
        not target_alert_rates
        and not include_all_achievable_alert_rates
    ):
        raise ValueError("Provide at least one target normal-flight alert rate or request all achievable alert rates.")

    invalid_alert_rates = [
        target_alert_rate
        for target_alert_rate in target_alert_rates
        if not 0.0 <= target_alert_rate < 1.0
    ]

    if invalid_alert_rates:
        raise ValueError(f"Target normal-flight alert rates must be between zero and one : {invalid_alert_rates}")


# Collecting every distinct calibration alert rate represented by stored thresholds
def build_achievable_alert_rates(run_evaluations: list[tuple[str, str, dict]]) -> list[float]:
    achievable_alert_rates = {
        float(
            record[
                "calibration_normal_flight_alert_rate"
            ]
        )
        for _, _, evaluation_metrics in run_evaluations
        for record in evaluation_metrics[
            "online_detection_threshold_metrics"
        ]
        if float(
            record[
                "calibration_normal_flight_alert_rate"
            ]
        ) < 1.0
    }

    if not achievable_alert_rates:
        raise RuntimeError("No achievable calibration alert rates were found.")

    return sorted(achievable_alert_rates)


# Loading evaluation records already stored by one MLflow run
def load_run_evaluation(client: MlflowClient, run_id: str) -> tuple[str, dict]:
    run = client.get_run(run_id)

    feature_set = run.data.params.get(
        "feature_set",
        "unknown"
    )

    artifact_path = client.download_artifacts(
        run_id = run_id,
        path = "evaluation_metrics.json"
    )

    evaluation_metrics = json.loads(
        Path(artifact_path).read_text()
    )

    required_keys = {
        "fold_metrics",
        "online_detection_threshold_metrics"
    }

    missing_keys = required_keys.difference(
        evaluation_metrics
    )

    if missing_keys:
        raise RuntimeError(f"Run '{run_id}' is missing evaluation records : {sorted(missing_keys)}")

    return feature_set, evaluation_metrics


# Selecting the most sensitive threshold within one calibration alert budget
def select_fold_operating_record(fold_records: list[dict], target_alert_rate: float) -> dict:
    eligible_records = [
        record
        for record in fold_records
        if (
            record[
                "calibration_normal_flight_alert_rate"
            ]
            <= target_alert_rate + 1e-12
        )
    ]

    if not eligible_records:
        raise RuntimeError("No threshold satisfies the requested calibration alert budget.")

    return min(
        eligible_records,
        key = lambda record: record["threshold"]
    )


# Calculating an average weighted by the number of represented flights
def calculate_weighted_average(records: list[dict], value_name: str, weight_name: str) -> float | None:
    usable_records = [
        record
        for record in records
        if (
            record.get(value_name) is not None
            and record.get(weight_name, 0) > 0
        )
    ]

    if not usable_records:
        return None

    weighted_total = sum(
        float(record[value_name])
        * int(record[weight_name])
        for record in usable_records
    )

    total_weight = sum(
        int(record[weight_name])
        for record in usable_records
    )

    return float(
        weighted_total / total_weight
    )


# Summarizing held-out behavior for one model and one alert budget
def build_operating_point_summary(run_id: str, feature_set: str,
                                  evaluation_metrics: dict, target_alert_rate: float
) -> dict:
    threshold_records = evaluation_metrics[
        "online_detection_threshold_metrics"
    ]

    fold_numbers = sorted(
        {
            int(record["fold_number"])
            for record in threshold_records
        }
    )

    selected_records = []

    for fold_number in fold_numbers:
        fold_records = [
            record
            for record in threshold_records
            if int(record["fold_number"]) == fold_number
        ]

        selected_records.append(
            select_fold_operating_record(
                fold_records = fold_records,
                target_alert_rate = target_alert_rate
            )
        )

    normal_flight_count = sum(
        int(record["normal_flight_count"])
        for record in selected_records
    )

    normal_false_alert_count = sum(
        int(
            record[
                "normal_flight_false_alert_count"
            ]
        )
        for record in selected_records
    )

    fault_flight_count = sum(
        int(record["fault_flight_count"])
        for record in selected_records
    )

    fault_any_alert_count = sum(
        int(record["fault_flight_any_alert_count"])
        for record in selected_records
    )

    fault_post_signal_detection_count = sum(
        int(record["fault_flight_detection_count"])
        for record in selected_records
    )

    pre_signal_evaluation_count = sum(
        int(
            record[
                "pre_observed_signal_evaluation_count"
            ]
        )
        for record in selected_records
    )

    pre_signal_alert_count = sum(
        int(
            record[
                "pre_observed_signal_alert_count"
            ]
        )
        for record in selected_records
    )

    return {
        "run_id": run_id,
        "feature_set": feature_set,
        "target_calibration_alert_rate": (
            target_alert_rate
        ),
        "selected_calibration_alert_rate_minimum": min(
            float(
                record[
                    "calibration_normal_flight_alert_rate"
                ]
            )
            for record in selected_records
        ),
        "selected_calibration_alert_rate_maximum": max(
            float(
                record[
                    "calibration_normal_flight_alert_rate"
                ]
            )
            for record in selected_records
        ),
        "threshold_minimum": min(
            float(record["threshold"])
            for record in selected_records
        ),
        "threshold_maximum": max(
            float(record["threshold"])
            for record in selected_records
        ),
        "normal_false_alerts": (
            f"{normal_false_alert_count}/"
            f"{normal_flight_count}"
        ),
        "held_out_normal_false_alert_rate": (
            normal_false_alert_count
            / normal_flight_count
        ),
        "fault_flights_alerted": (
            f"{fault_any_alert_count}/"
            f"{fault_flight_count}"
        ),
        "held_out_fault_any_alert_rate": (
            fault_any_alert_count
            / fault_flight_count
        ),
        "fault_flights_detected_after_signal": (
            f"{fault_post_signal_detection_count}/"
            f"{fault_flight_count}"
        ),
        "held_out_post_signal_detection_rate": (
            fault_post_signal_detection_count
            / fault_flight_count
        ),
        "fault_flights_alerted_before_signal": (
            f"{pre_signal_alert_count}/"
            f"{pre_signal_evaluation_count}"
        ),
        "held_out_pre_signal_alert_rate": (
            pre_signal_alert_count
            / pre_signal_evaluation_count
            if pre_signal_evaluation_count > 0
            else None
        ),
        "mean_first_alert_signed_delay_seconds": (
            calculate_weighted_average(
                records = selected_records,
                value_name = (
                    "mean_first_alert_signed_delay_seconds"
                ),
                weight_name = (
                    "fault_flight_any_alert_count"
                )
            )
        )
    }


# Comparing stored threshold behavior across runs and alert budgets
def main() -> None:
    args = parse_args()

    requested_alert_rates = (
        args.target_normal_flight_alert_rate
        or []
    )

    validate_alert_rate_arguments(
        target_alert_rates = requested_alert_rates,
        include_all_achievable_alert_rates = (
            args.include_all_achievable_alert_rates
        )
    )

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DATABASE_PATH}")

    client = MlflowClient()
    run_evaluations = []

    # Loading every run once before constructing the comparison budgets
    for run_id in args.run_id:
        (
            feature_set,
            evaluation_metrics
        ) = load_run_evaluation(
            client = client,
            run_id = run_id
        )

        run_evaluations.append(
            (
                run_id,
                feature_set,
                evaluation_metrics
            )
        )

    target_alert_rates = set(
        requested_alert_rates
    )

    if args.include_all_achievable_alert_rates:
        target_alert_rates.update(
            build_achievable_alert_rates(
                run_evaluations = run_evaluations
            )
        )

    summary_records = []

    for (
        run_id,
        feature_set,
        evaluation_metrics
    ) in run_evaluations:
        for target_alert_rate in sorted(
            target_alert_rates
        ):
            summary_records.append(
                build_operating_point_summary(
                    run_id = run_id,
                    feature_set = feature_set,
                    evaluation_metrics = (
                        evaluation_metrics
                    ),
                    target_alert_rate = (
                        target_alert_rate
                    )
                )
            )

    summary = pd.DataFrame(
        summary_records
    ).sort_values(
        [
            "target_calibration_alert_rate",
            "held_out_normal_false_alert_rate",
            "held_out_fault_any_alert_rate"
        ],
        ascending = [
            True,
            True,
            False
        ]
    )

    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        280
    )

    print(
        summary.to_string(
            index = False,
            float_format = (
                lambda value: f"{value:.4f}"
            )
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "Fault-state threshold analysis failed : "
            f"{exc}"
        )
        sys.exit(1)