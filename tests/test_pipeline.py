"""Behavioral geometry, label, metric and fold-isolation checks."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from sklearn.model_selection import GroupKFold

from pathsnap.alignment import align_reference
from pathsnap.features import build_features, motion_features, FEATURE_NAMES
from pathsnap.geometry import RoadNetwork, project_segments
from pathsnap.learning import PackedData, score_rows, training_sample
from pathsnap.metrics import lcs_f1, observation_metrics, select_top1


def small_network():
    # Approximately metre-scaled equatorial coordinates; projection is still
    # exercised by the real RoadNetwork implementation.
    xy = np.asarray([[0, 0], [100, 0], [100, 100], [0, 100], [200, 0]], float)
    return RoadNetwork(xy / 111319.49, [[0, 1], [1, 2], [2, 3], [3, 0], [1, 4]], "pipeline-test")


def classical_lcs(a, b):
    row = [0] * (len(b) + 1)
    for x in a:
        previous = row[:]
        for i, y in enumerate(b, 1):
            row[i] = previous[i - 1] + 1 if x == y else max(previous[i], row[i - 1])
    return row[-1]


class GeometryTests(unittest.TestCase):
    def test_exact_projection_clamps_endpoints_and_handles_degenerate_segment(self):
        starts = np.array([[0, 0], [0, 0], [3, 4]], float)
        ends = np.array([[10, 0], [1, 0], [3, 4]], float)
        distance, fraction, projected = project_segments(np.array([5, 4]), starts, ends)
        np.testing.assert_allclose(distance, [4, np.sqrt(32), 2])
        np.testing.assert_allclose(fraction, [0.5, 1, 0])
        np.testing.assert_allclose(projected, [[5, 0], [1, 0], [3, 4]])

    def test_projection_round_trip_and_reference_free_candidates(self):
        net = small_network()
        points = np.array([[0.0002, 0.0001], [0.0008, 0.0004]])
        np.testing.assert_allclose(net.unproject(net.project(points)), points, atol=1e-10)
        c = net.candidates(net.project(points), k=3, radius=120)
        self.assertEqual(c.arc_ids.shape, (2, 3))
        self.assertTrue(np.all(np.diff(c.distances, axis=1) >= 0))
        self.assertTrue(np.all((c.fractions >= 0) & (c.fractions <= 1)))

    def test_far_observation_stays_missing_and_features_are_finite(self):
        net = small_network()
        xy = np.array([[10000, 10000]], float)
        c = net.candidates(xy, k=4, radius=20, max_radius=100)
        self.assertFalse(c.valid.any())
        x, _ = build_features(net, c, xy, np.array([0.0]))
        self.assertEqual(x.shape, (1, 4, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.all(x == 0))


class ReferenceAlignmentTests(unittest.TestCase):
    def test_repeated_arc_occurrences_keep_ordered_position_identity(self):
        net = small_network()
        raw_xy = np.array([[10, 0], [90, 0], [100, 50], [50, 100],
                           [0, 50], [10, 0], [90, 0], [150, 0]], float)
        xy = net.project(raw_xy / 111319.49)
        route = np.array([0, 1, 2, 3, 0, 4], np.int32)
        labels, positions, info = align_reference(net, xy, np.arange(len(xy)) * 2, route)
        self.assertEqual(labels[0], 0)
        self.assertEqual(labels[-1], 4)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[6], 4)
        self.assertTrue(np.all(np.diff(positions) >= 0))
        np.testing.assert_array_equal(labels, route[positions])
        self.assertEqual(info["observations"], len(xy))

    def test_alignment_handles_single_observation_and_long_gap(self):
        net = small_network()
        xy = net.project(np.array([[50, 0]], float) / 111319.49)
        labels, positions, _ = align_reference(net, xy, np.array([0]), np.array([0, 4]))
        self.assertEqual(labels.shape, (1,))
        self.assertEqual(labels[0], 0)
        xy = net.project(np.array([[50, 0], [150, 0]], float) / 111319.49)
        labels, positions, info = align_reference(net, xy, np.array([0, 100]), np.array([0, 4]))
        self.assertEqual(labels.tolist(), [0, 4])
        self.assertIn(1, info["restarts"])

    def test_heading_does_not_use_observations_across_large_gap(self):
        xy = np.array([[0, 0], [10, 0], [0, 0], [0, 10]], float)
        heading, reliability, _, _ = motion_features(xy, np.array([0, 1, 100, 101]))
        np.testing.assert_allclose(heading[:2], [0, 0])
        np.testing.assert_allclose(heading[2:], [np.pi / 2, np.pi / 2])
        self.assertTrue(np.all((reliability >= 0) & (reliability <= 1)))


class MetricTests(unittest.TestCase):
    def test_missing_candidate_remains_top1_failure(self):
        labels = np.array([5, 6, 7])
        candidates = np.array([[5, 2], [1, 2], [-1, -1]])
        prediction, indices = select_top1(candidates, np.array([[0.8, 0.2], [0.6, 0.4], [0, 0]]))
        self.assertEqual(prediction.tolist(), [5, 1, -1])
        m = observation_metrics(labels, prediction, candidates)
        self.assertEqual(m["observations"], 3)
        self.assertEqual(m["correct"], 1)
        self.assertAlmostEqual(m["top1_accuracy"], 1 / 3)
        self.assertAlmostEqual(m["candidate_miss_rate"], 2 / 3)

    def test_exact_lcs_matches_classical_dynamic_programming(self):
        rng = np.random.default_rng(352)
        for _ in range(40):
            a = rng.integers(0, 5, size=int(rng.integers(0, 35))).tolist()
            b = rng.integers(0, 5, size=int(rng.integers(0, 35))).tolist()
            self.assertEqual(lcs_f1(a, b)["sequence_lcs"], classical_lcs(a, b))
        self.assertEqual(lcs_f1([0, 1, 0], [0, 0, 1])["sequence_lcs"], 2)

    def test_packed_scoring_keeps_observation_without_any_candidate(self):
        d = PackedData(X=np.zeros((3, 2)), y=np.array([1, 0, 0]), groups=np.array([0, 0, 0]),
                       row_obs=np.array([0, 0, 1]), arcs=np.array([5, 2, 1]),
                       labels=np.array([5, 6, 7]), obs_groups=np.array([0, 0, 0]),
                       ranks=np.array([0, 1, 0]), trip_ids=["trip0"], files=[])
        metrics, predictions = score_rows(d, np.arange(3), np.array([0.9, 0.1, 0.8]))
        self.assertEqual(predictions.tolist(), [5, 1, -1])
        self.assertEqual(metrics["observations"], 3)
        self.assertAlmostEqual(metrics["top1_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["candidate_miss_rate"], 2 / 3)


class GroupIsolationTests(unittest.TestCase):
    def test_training_sampling_is_subset_and_retains_positives_and_hard_negatives(self):
        rows = np.arange(50, 100)
        d = type("Packed", (), {})()
        d.y = np.zeros(100, int)
        d.y[[55, 99]] = 1
        d.ranks = np.arange(100) % 10
        sample = training_sample(d, rows, seed=352)
        self.assertTrue(set(sample).issubset(set(rows)))
        self.assertTrue({55, 99}.issubset(set(sample)))
        self.assertTrue(set(rows[d.ranks[rows] < 2]).issubset(set(sample)))

    def test_grouped_folds_keep_all_trip_candidates_together(self):
        groups = np.repeat(np.arange(8), [3, 8, 4, 12, 5, 8, 7, 4])
        x = np.zeros((len(groups), 2))
        for train, validation in GroupKFold(3).split(x, groups=groups):
            self.assertFalse(set(groups[train]) & set(groups[validation]))
            self.assertEqual(len(train) + len(validation), len(groups))


if __name__ == "__main__":
    unittest.main()
