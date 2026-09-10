import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

# Restricting orchestration to final project scripts
ALLOWED_PIPELINE_SCRIPTS = {
    "download_ALFA_from_S3.py",
    "build_processed_inventory.py",
    "processed_dataset_assessment.py",
    "inspect_failure_status_signals.py",
    "build_dataset_reference.py",
    "build_modeling_reference.py",
    "build_telemetry_window_dataset.py",
    "train_fault_state_classifier.py"
}

SUPPORTED_FEATURE_SETS = [
    "raw",
    "dynamic_only",
    "command_response_only",
    "initial_baseline_only",
    "trailing_history_only",
    "dynamic_plus_initial_baseline",
    "dynamic_plus_command_response",
    "relative_dynamic",
    "combined"
]


# Running one reviewed project script with the active Python environment
@task(
    name = "Run project script",
    task_run_name = "{script_name}"
)
def run_project_script(script_name: str, arguments: list[str] | None = None) -> str:
    if script_name not in ALLOWED_PIPELINE_SCRIPTS:
        raise ValueError(f"Script is not allowed in an orchestrated workflow : {script_name}")

    script_path = SCRIPTS_ROOT / script_name

    if not script_path.is_file():
        raise FileNotFoundError(f"Project script not found : {script_path}")

    command = [
        sys.executable,
        str(script_path),
        *(arguments or [])
    ]

    result = subprocess.run(
        command,
        cwd = PROJECT_ROOT,
        capture_output = True,
        text = True,
        check = False
    )

    logger = get_run_logger()

    if result.stdout.strip():
        logger.info(result.stdout.strip())

    if result.stderr.strip():
        logger.warning(result.stderr.strip())

    if result.returncode != 0:
        raise RuntimeError(f"Project script failed with exit code {result.returncode} : {script_name}")

    return result.stdout


# Rebuilding validated metadata and the versioned telemetry window dataset
@flow(
    name = "Build telemetry training dataset",
    log_prints = True
)
def build_telemetry_dataset(download_processed_data: bool = False) -> str:
    if download_processed_data:
        run_project_script(
            script_name = "download_ALFA_from_S3.py",
            arguments = [
                "--sections",
                "processed"
            ]
        )

    run_project_script(
        script_name = "build_processed_inventory.py"
    )

    run_project_script(
        script_name = "processed_dataset_assessment.py"
    )

    run_project_script(
        script_name = "inspect_failure_status_signals.py"
    )

    run_project_script(
        script_name = "build_dataset_reference.py"
    )

    run_project_script(
        script_name = "build_modeling_reference.py"
    )

    return run_project_script(
        script_name = "build_telemetry_window_dataset.py"
    )


# Training one explicitly configured classifier without promoting it
@flow(
    name = "Train fault-state classifier",
    log_prints = True
)
def train_fault_state_classifier(feature_set: str = "dynamic_plus_initial_baseline", 
                                 target_normal_state_alert_rate: float = 0.10,
                                 run_name: str = "fault-state-classifier-dynamic-baseline", 
                                 rebuild_dataset: bool = False,
                                 download_processed_data: bool = False
) -> str:
    if feature_set not in SUPPORTED_FEATURE_SETS:
        raise ValueError(f"Unsupported feature set : {feature_set}")

    if not 0.0 <= target_normal_state_alert_rate < 1.0:
        raise ValueError("Target normal-state alert rate must be between zero and one.")

    if download_processed_data and not rebuild_dataset:
        raise ValueError("Downloading processed data requires rebuilding the dataset.")

    if rebuild_dataset:
        build_telemetry_dataset(
            download_processed_data = download_processed_data
        )

    return run_project_script(
        script_name = "train_fault_state_classifier.py",
        arguments = [
            "--feature-set",
            feature_set,
            "--target-normal-flight-alert-rate",
            str(target_normal_state_alert_rate),
            "--run-name",
            run_name
        ]
    )


# Parsing one explicitly selected local workflow
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workflow",
        choices = [
            "dataset",
            "training"
        ],
        required = True
    )

    parser.add_argument(
        "--download-processed-data",
        action = "store_true"
    )

    parser.add_argument(
        "--rebuild-dataset",
        action = "store_true"
    )

    parser.add_argument(
        "--feature-set",
        choices = SUPPORTED_FEATURE_SETS,
        default = "dynamic_plus_initial_baseline"
    )

    parser.add_argument(
        "--target-normal-state-alert-rate",
        type = float,
        default = 0.10
    )

    parser.add_argument(
        "--run-name",
        default = "fault-state-classifier-dynamic-baseline"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.workflow == "dataset":
        if args.rebuild_dataset:
            raise ValueError("--rebuild-dataset applies only to the training workflow.")

        build_telemetry_dataset(
            download_processed_data = (
                args.download_processed_data
            )
        )

        return

    train_fault_state_classifier(
        feature_set = args.feature_set,
        target_normal_state_alert_rate = (
            args.target_normal_state_alert_rate
        ),
        run_name = args.run_name,
        rebuild_dataset = args.rebuild_dataset,
        download_processed_data = (
            args.download_processed_data
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Pipeline orchestration failed : {exc}")
        sys.exit(1)