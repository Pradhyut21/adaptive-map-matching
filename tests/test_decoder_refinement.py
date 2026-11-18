"""Regression tests for noise tolerance and predicted-route alignment."""

from types import SimpleNamespace
import unittest

import numpy as np

from pathsnap.decoder import DecoderConfig
from pathsnap.decoder_refinement import (
    JitterAwareRouter,
    RefinementConfig,
    _align_run,
    decode_refined,
)
from pathsnap.features import motion_features
from pathsnap.geometry import CandidateSet


def graph(nodes, arcs):
    nodes_xy = np.asarray(nodes, dtype=float)
    arcs = np.asarray(arcs, dtype=int)
    vectors = nodes_xy[arcs[:, 1]] - nodes_xy[arcs[:, 0]]
    return SimpleNamespace(nodes_xy=nodes_xy, arcs=arcs, lengths=np.linalg.norm(vectors, axis=1),
                           bearings=np.arctan2(vectors[:, 1], vectors[:, 0]))


def candidates(network, arcs, fractions):
    ids = np.asarray(arcs, dtype=int).reshape(-1, 1)
    projected_fractions = np.asarray(fractions, dtype=float).reshape(-1, 1)
    starts, ends = network.nodes_xy[network.arcs[ids, 0]], network.nodes_xy[network.arcs[ids, 1]]
    positions = starts + projected_fractions[..., None] * (ends - starts)
    return CandidateSet(ids, np.zeros(ids.shape), projected_fractions, positions)


class DecoderRefinementTests(unittest.TestCase):
    def test_slow_gps_jitter_keeps_one_way_route_without_inventing_loops(self):
        network = graph([[0, 0], [1000, 0]], [[0, 1]])
        fraction = [.50, .52, .505, .515, .50, .52]
        candidate = candidates(network, [0] * len(fraction), fraction)
        timestamps = np.arange(len(fraction), dtype=float)
        options = RefinementConfig(same_arc_jitter_m=30.)
        result = decode_refined(network, candidate, np.ones(candidate.arc_ids.shape),
                                candidate.xy[:, 0], timestamps, refinement=options)
        self.assertEqual(result.route_segments, [[0]])
        self.assertEqual(result.break_indices, [0])
        self.assertEqual(result.warnings, [])
        router = JitterAwareRouter(network, DecoderConfig(), same_arc_jitter_m=30.)
        self.assertTrue(np.isinf(router.distance_matrix([0], [.5], [0], [.469], 200.)[0, 0]))
        self.assertIsNone(router.arc_path(0, .5, 0, .469, 200.))

    def test_small_same_arc_reverse_jitter_dwells_and_real_reverse_is_unreachable(self):
        network = graph([[0, 0], [100, 0]], [[0, 1]])
        router = JitterAwareRouter(network, DecoderConfig(), same_arc_jitter_m=5.)
        self.assertEqual(router.distance_matrix([0], [.5], [0], [.47], 200.)[0, 0], 0.)
        self.assertEqual(router.arc_path(0, .5, 0, .47, 200.), [0])
        self.assertTrue(np.isinf(router.distance_matrix([0], [.5], [0], [.41], 200.)[0, 0]))
        self.assertIsNone(router.arc_path(0, .5, 0, .41, 200.))

    def test_route_realignment_respects_duplicate_timestamps_and_large_gaps(self):
        network = graph([[0, 0], [100, 0], [200, 0], [300, 0]], [[0, 1], [1, 2], [2, 3]])
        candidate = candidates(network, [0, 0, 1, 1, 2, 2], [.2, .3, .3, .4, .3, .5])
        timestamps = np.array([0, 3, 3, 4, 200, 202], dtype=float)
        options = RefinementConfig(same_arc_jitter_m=5., smoothing_half_window=2,
                                   smooth_projections=True, realign_predicted_route=True)
        with np.errstate(divide="raise", invalid="raise"):
            result = decode_refined(network, candidate, np.ones(candidate.arc_ids.shape),
                                    candidate.xy[:, 0], timestamps, refinement=options)
        self.assertEqual(result.break_indices, [0, 2, 4])
        self.assertEqual(result.route_segments, [[0], [1], [2]])
        self.assertEqual([warning["reason"] for warning in result.warnings],
                         ["invalid_timestamp_interval", "large_gps_gap"])
        np.testing.assert_array_equal(result.arc_ids, candidate.arc_ids[:, 0])

    def test_repeated_arc_occurrence_can_be_aligned_after_its_first_visit(self):
        network = graph([[0, 0], [75, 0], [75, 55]], [[0, 1], [1, 2], [2, 0]])
        arcs = [0, 1, 2, 0]
        candidate = candidates(network, arcs, [.85, .8, .8, .35])
        positions = candidate.xy[:, 0]
        timestamps = np.array([0, 4, 10, 13], dtype=float)
        probabilities = np.ones(candidate.arc_ids.shape)
        config = DecoderConfig()
        refinement = RefinementConfig(same_arc_jitter_m=5., realign_predicted_route=True)
        result = decode_refined(network, candidate, probabilities, positions, timestamps,
                                config, refinement)
        self.assertEqual(result.route_segments, [arcs])
        np.testing.assert_array_equal(result.arc_ids, candidate.arc_ids[:, 0])
        heading, reliability, _, _ = motion_features(positions, timestamps)
        # Assert that the second occurrence was accepted, rather than the
        # public result silently falling back to the original Viterbi path.
        aligned = _align_run(network, candidate, probabilities, positions, timestamps, heading, reliability,
                             result, 0, len(arcs), arcs, config, refinement)
        np.testing.assert_array_equal(aligned, np.zeros(len(arcs), dtype=int))


if __name__ == "__main__":
    unittest.main()
