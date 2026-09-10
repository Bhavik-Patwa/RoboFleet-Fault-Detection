import json
import unittest
from pathlib import Path

import mlflow.artifacts
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app


# Checking the real registered model through the API request interface
class FaultDetectionAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Entering the context runs the same startup checks as the server.
        cls.client = TestClient(app)
        cls.client.__enter__()
        cls.addClassCleanup(cls.client.__exit__, None, None, None)

        response = cls.client.get("/model")
        response.raise_for_status()
        cls.contract = response.json()

        model_path = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri = cls.contract["registered_model_uri"]
            )
        )

        example = json.loads(
            (model_path / "input_example.json").read_text()
        )

        cls.features = pd.DataFrame(
            example["data"],
            columns = example["columns"]
        )

        cls.records = cls.features.to_dict(orient = "records")

    def test_health_and_readiness(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/ready").status_code, 200)

    def test_predictions_match_native_model(self) -> None:
        native_model = mlflow.sklearn.load_model(
            self.contract["registered_model_uri"]
        )

        expected_scores = native_model.predict_proba(
            self.features
        )[:, 1]

        # JSON object key order must not change prediction behavior.
        reordered_records = [
            dict(reversed(list(record.items())))
            for record in self.records
        ]

        response = self.client.post(
            "/predict",
            json = reordered_records
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()

        self.assertEqual(
            result["model_uri"],
            self.contract["registered_model_uri"]
        )
        self.assertEqual(
            result["threshold"],
            self.contract["alert_threshold"]
        )

        np.testing.assert_array_equal(
            [item["fault_score"] for item in result["predictions"]],
            expected_scores
        )

        self.assertEqual(
            [item["alert"] for item in result["predictions"]],
            (expected_scores >= result["threshold"]).tolist()
        )

    def test_invalid_records_are_rejected(self) -> None:
        record = self.records[0]
        feature_name = next(iter(record))

        missing_feature = dict(record)
        missing_feature.pop(feature_name)

        extra_feature = dict(record)
        extra_feature["unexpected_feature"] = 0.0

        invalid_payloads = [[], [missing_feature], [extra_feature]]

        for value in [None, True, "1.0", [], {}, 10 ** 400]:
            invalid_record = dict(record)
            invalid_record[feature_name] = value
            invalid_payloads.append([invalid_record])

        for payload in invalid_payloads:
            with self.subTest(payload_type = str(type(payload))):
                response = self.client.post("/predict", json = payload)
                self.assertEqual(response.status_code, 422)

    def test_non_finite_input_is_rejected(self) -> None:
        record = dict(self.records[0])
        record[next(iter(record))] = float("inf")

        response = self.client.post(
            "/predict",
            content = json.dumps([record]),
            headers = {"Content-Type": "application/json"}
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()