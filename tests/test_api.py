"""HTTP boundary tests: validation, model readiness and reference isolation.

The successful inference test uses the saved trained artifact and an actually
discovered student-testing trip. It is skipped only while training is pending.
No successful prediction response is mocked.
"""
from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.main as api


def assert_no_reference_payload(test: unittest.TestCase, value, path="response"):
    """Allow aggregate methodology/counts, never reference labels or geometry."""
    banned = {
        "reference", "reference_route", "reference_routes", "reference_labels",
        "reference_arc_ids", "reference_coordinates", "ground_truth", "groundtruth",
        "ground_truth_labels", "ground_truth_route", "ground_truth_arc_ids",
        "labels", "y_true", "true_arcs", "true_arc_ids", "route_labels",
        "reference_sequence", "reference_route_positions", "alignment_labels",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            test.assertNotIn(str(key).lower(), banned, f"Reference payload leaked at {path}.{key}")
            # reference_label_note and reference_sequence_length are deliberately
            # allowed: a methodology sentence and aggregate count reveal no label.
            assert_no_reference_payload(test, child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_reference_payload(test, child, f"{path}[{index}]")


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(api.app)
        cls.client = cls.client_context.__enter__()
        response = cls.client.get("/trips")
        cls.discovered = response.json().get("trips", []) if response.status_code == 200 else []

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def trip_id(self):
        if not self.discovered:
            self.skipTest("No permitted runtime trip files are installed")
        return self.discovered[0]["id"]

    def test_health_reports_readiness_without_requiring_trained_artifact(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIsInstance(data["model_ready"], bool)
        self.assertIsInstance(data["trip_count"], int)
        self.assertIn(data["partition"], ("student_test", "evaluator_test", "evaluator_testing"))
        self.assertIn("model_error", data)
        assert_no_reference_payload(self, data)

    def test_trip_discovery_exposes_identifiers_and_counts_only(self):
        response = self.client.get("/trips")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data["trips"], list)
        for trip in data["trips"]:
            self.assertEqual(set(trip), {"id", "observations"})
            self.assertRegex(trip["id"], r"^[A-Za-z0-9_-]+$")
            self.assertGreater(trip["observations"], 0)
        assert_no_reference_payload(self, data)

    def test_unknown_trip_is_not_found_even_if_model_missing(self):
        response = self.client.post("/predict", json={"trip_id": "unknown-trip-not-present"})
        self.assertEqual(response.status_code, 404)
        self.assertIn("not available", response.json()["detail"])

    def test_missing_or_invalid_trip_ids_are_rejected_before_inference(self):
        bodies = [
            {}, {"trip_id": ""}, {"trip_id": "../train/00000001"},
            {"trip_id": "..\\train\\00000001"}, {"trip_id": "/etc/passwd"},
            {"trip_id": "C:\\Windows\\win.ini"}, {"trip_id": "a" * 65},
            {"trip_id": "%2e%2e%2ftrain%2f00000001"}, {"trip_id": None},
        ]
        with patch.object(api, "predict_trip") as predict:
            for body in bodies:
                with self.subTest(body=body):
                    response = self.client.post("/predict", json=body)
                    self.assertEqual(response.status_code, 422)
            predict.assert_not_called()

    def test_unavailable_model_is_explicit_503_and_health_is_informative(self):
        trip = self.trip_id()
        with patch.object(api, "MODEL", None), patch.object(api, "MODEL_ERROR", None), \
                patch.object(api, "load_artifact", side_effect=FileNotFoundError("Model artifact has not been trained")), \
                patch.object(api, "predict_trip") as predict:
            response = self.client.post("/predict", json={"trip_id": trip})
            self.assertEqual(response.status_code, 503)
            self.assertIn("Trained model not available", response.json()["detail"])
            health = self.client.get("/health").json()
            self.assertFalse(health["model_ready"])
            self.assertIn("not been trained", health["model_error"])
            predict.assert_not_called()

    def test_prediction_failure_does_not_invoke_reference_evaluation(self):
        trip = self.trip_id()
        # A simulated failure exercises only control flow, not a fabricated
        # successful run. Reference evaluation must wait for real prediction.
        with patch.object(api, "MODEL", {"model_name": "failure-path-test"}), \
                patch.object(api, "predict_trip", side_effect=RuntimeError("Deliberate test failure")), \
                patch.object(api, "evaluate_prediction") as evaluate:
            response = self.client.post("/predict", json={"trip_id": trip})
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json()["detail"], "Prediction failed; inspect the server request log")
            evaluate.assert_not_called()

    def test_separate_laptop_origin_can_send_prediction_json(self):
        response = self.client.options("/predict", headers={
            "Origin": "http://192.168.77.2:8800",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.headers.get("access-control-allow-origin"), ("*", "http://192.168.77.2:8800"))
        self.assertIn("POST", response.headers.get("access-control-allow-methods", ""))

    def test_actual_trained_inference_has_one_prediction_per_observation_and_no_reference_labels(self):
        if api.MODEL is None:
            self.skipTest("Training pending: no saved model was loaded; no inference result fabricated")
        trip = self.trip_id()
        response = self.client.post("/predict", json={"trip_id": trip})
        self.assertEqual(response.status_code, 200, response.text[:500])
        data = response.json()
        self.assertEqual(data["trip_id"], trip)
        self.assertTrue(data["run_id"])
        self.assertTrue(data["model_name"])
        n = len(data["raw_gps"])
        self.assertGreater(n, 0)
        self.assertEqual(len(data["predictions"]), n)
        for point in data["raw_gps"]:
            self.assertEqual(len(point), 2)
            self.assertTrue(all(math.isfinite(number) for number in point))
        self.assertTrue(all(arc is None or isinstance(arc, int) for arc in data["predictions"]))
        self.assertEqual(data["road_network"]["type"], "FeatureCollection")
        self.assertEqual(data["predicted_route"]["type"], "FeatureCollection")
        for key in ("top1_accuracy", "prediction_coverage", "candidate_miss_rate", "route_continuity", "sequence_lcs_f1"):
            self.assertGreaterEqual(data["metrics"][key], 0)
            self.assertLessEqual(data["metrics"][key], 1)
        self.assertGreater(data["timing"]["inference_seconds"], 0)
        for key in ("candidate_count", "heading_reliability", "score_margin"):
            self.assertEqual(len(data["diagnostics"][key]), n)
        assert_no_reference_payload(self, data)


if __name__ == "__main__":
    unittest.main()
