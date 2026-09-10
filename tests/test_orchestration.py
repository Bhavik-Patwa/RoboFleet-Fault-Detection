import subprocess
import unittest
from unittest.mock import Mock, call, patch

from orchestration.pipelines import (
    build_telemetry_dataset,
    run_project_script,
    train_fault_state_classifier
)


# Verifying workflow order, parameters and failure propagation
class PipelineOrchestrationTests(unittest.TestCase):
    @patch(
        "orchestration.pipelines.run_project_script"
    )
    def test_dataset_tasks_run_in_dependency_order(self, script_task) -> None:
        build_telemetry_dataset.fn(
            download_processed_data = False
        )

        self.assertEqual(
            script_task.call_args_list,
            [
                call(
                    script_name = ("build_processed_inventory.py")
                ),
                call(
                    script_name = ("processed_dataset_assessment.py")
                ),
                call(
                    script_name = ("inspect_failure_status_signals.py")
                ),
                call(
                    script_name = ("build_dataset_reference.py")
                ),
                call(
                    script_name = ("build_modeling_reference.py")
                ),
                call(
                    script_name = ("build_telemetry_window_dataset.py")
                )
            ]
        )

    @patch(
        "orchestration.pipelines.run_project_script"
    )
    def test_download_precedes_dataset_processing(self, script_task) -> None:
        build_telemetry_dataset.fn(
            download_processed_data = True
        )

        self.assertEqual(
            script_task.call_args_list[0],
            call(
                script_name = "download_ALFA_from_S3.py",
                arguments = [
                    "--sections",
                    "processed"
                ]
            )
        )

    @patch(
        "orchestration.pipelines.run_project_script"
    )
    def test_training_uses_explicit_model_parameters(self,script_task) -> None:
        train_fault_state_classifier.fn(
            feature_set = "dynamic_plus_initial_baseline",
            target_normal_state_alert_rate = 0.10,
            run_name = "fault-state-classifier-dynamic-baseline"
        )

        script_task.assert_called_once_with(
            script_name = "train_fault_state_classifier.py",
            arguments = [
                "--feature-set",
                "dynamic_plus_initial_baseline",
                "--target-normal-flight-alert-rate",
                "0.1",
                "--run-name",
                "fault-state-classifier-dynamic-baseline"
            ]
        )

    def test_download_requires_dataset_rebuild(self) -> None:
        with self.assertRaises(ValueError):
            train_fault_state_classifier.fn(
                rebuild_dataset = False,
                download_processed_data = True
            )

    def test_unreviewed_script_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_project_script.fn(
                script_name = (
                    "register_fault_state_classifier.py"
                )
            )

    @patch(
        "orchestration.pipelines.get_run_logger"
    )
    @patch(
        "orchestration.pipelines.subprocess.run"
    )
    def test_script_failure_stops_the_pipeline(self, run_command, get_logger) -> None:
        get_logger.return_value = Mock()

        run_command.return_value = subprocess.CompletedProcess(
            args = [],
            returncode = 1,
            stdout = "Script validation failed.",
            stderr = ""
        )

        with self.assertRaises(RuntimeError):
            run_project_script.fn(
                script_name = "build_processed_inventory.py"
            )


if __name__ == "__main__":
    unittest.main()