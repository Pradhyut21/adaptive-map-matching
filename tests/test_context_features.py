"""Behavior checks for the trajectory feature schema."""

import unittest
from types import SimpleNamespace

import numpy as np

from pathsnap.context_features import FEATURE_NAMES, build_context_features
from pathsnap.geometry import CandidateSet, project_segments


def bidirectional_road():
    nodes = np.array([[0., 0.], [200., 0.]])
    arcs = np.array([[0, 1], [1, 0]])
    starts, ends = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    vectors = ends - starts
    return SimpleNamespace(arcs=arcs, starts=starts, ends=ends,
                           lengths=np.linalg.norm(vectors, axis=1),
                           bearings=np.arctan2(vectors[:, 1], vectors[:, 0]))


def candidates_for(road, xy):
    ids = np.broadcast_to(np.arange(len(road.arcs)), (len(xy), len(road.arcs)))
    distance, fraction, projection = project_segments(xy[:, None, :], road.starts[ids], road.ends[ids])
    return CandidateSet(ids, distance, fraction, projection)


class ContextFeaturesTests(unittest.TestCase):
    def test_directed_motion_and_connection_favor_the_travel_direction(self):
        xy = np.column_stack((np.arange(40, 161, 20), np.zeros(7)))
        road = bidirectional_road()
        result, movement = build_context_features(road, candidates_for(road, xy), xy, np.arange(len(xy)))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (len(xy), 2, len(FEATURE_NAMES)))
        self.assertGreater(movement["heading_reliability"][3], .9)
        heading = FEATURE_NAMES.index("heading_agreement_4")
        connection = FEATURE_NAMES.index("incoming_direct_connection_cost")
        self.assertGreater(result[3, 0, heading], result[3, 1, heading])
        self.assertLess(result[3, 0, connection], result[3, 1, connection])

    def test_context_does_not_use_observations_across_large_time_gap(self):
        road = bidirectional_road()
        first = np.array([[20., 2.], [25., 2.], [30., 2.]])
        second = np.array([[90., 20.], [85., 20.], [80., 20.]])
        xy = np.concatenate((first, second))
        together, _ = build_context_features(road, candidates_for(road, xy), xy, [0, 1, 2, 500, 501, 502])
        isolated, _ = build_context_features(road, candidates_for(road, second), second, [500, 501, 502])
        context = FEATURE_NAMES.index("heading_agreement_4")
        np.testing.assert_array_equal(together[3:, :, context:], isolated[:, :, context:])

    def test_padding_kept_zero(self):
        road = bidirectional_road()
        xy = np.array([[10., 2.], [25., 2.], [45., 2.]])
        original = candidates_for(road, xy)
        candidate_ids = original.arc_ids.copy()
        distances = original.distances.copy()
        candidate_ids[1, 1], distances[1, 1] = -1, np.inf
        padded = CandidateSet(candidate_ids, distances, original.fractions, original.xy)
        actual, _ = build_context_features(road, padded, xy, [0, 1, 2])
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(actual[1, 1], np.zeros(len(FEATURE_NAMES)))


if __name__ == "__main__":
    unittest.main()
