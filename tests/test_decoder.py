"""Behavioral tests for directed routing, trellis gaps and failure denominators."""
from types import SimpleNamespace
import unittest

import numpy as np

from pathsnap.decoder import DecoderConfig, DirectedRouter, decode, evaluate_continuity


def network(nodes, arcs):
    nodes = np.asarray(nodes, dtype=float)
    arcs = np.asarray(arcs, dtype=int).reshape(-1, 2)
    lengths = np.linalg.norm(nodes[arcs[:, 1]] - nodes[arcs[:, 0]], axis=1)
    return SimpleNamespace(nodes_xy=nodes, arcs=arcs, lengths=lengths, graph_id="test")


def candidates(arc_ids, fractions=None):
    ids = np.asarray(arc_ids, dtype=int)
    fractions = np.full(ids.shape, 0.5) if fractions is None else np.asarray(fractions, dtype=float)
    return SimpleNamespace(arc_ids=ids, fractions=fractions)


class DirectedRouterTests(unittest.TestCase):
    def setUp(self):
        self.net = network([[0, 0], [100, 0], [200, 0], [300, 0]],
                           [[0, 1], [1, 2], [2, 3]])
        self.router = DirectedRouter(self.net)

    def test_intermediate_arcs_and_projected_distance(self):
        distance = self.router.distance_matrix([0], [0.75], [2], [0.25], 500)[0, 0]
        self.assertAlmostEqual(distance, 150)
        self.assertEqual(self.router.arc_path(0, 0.75, 2, 0.25, 500), [0, 1, 2])

    def test_forward_same_arc_and_unreachable_reverse(self):
        self.assertAlmostEqual(self.router.distance_matrix([0], [0.2], [0], [0.8], 100)[0, 0], 60)
        self.assertTrue(np.isinf(self.router.distance_matrix([0], [0.8], [0], [0.2], 500)[0, 0]))
        self.assertIsNone(self.router.arc_path(2, 0.5, 0, 0.5, 500))

    def test_same_arc_backwards_requires_real_directed_loop(self):
        net = network([[0, 0], [100, 0], [100, 100]], [[0, 1], [1, 2], [2, 0]])
        router = DirectedRouter(net)
        expected = 20 + 100 + np.sqrt(20000) + 20
        self.assertAlmostEqual(router.distance_matrix([0], [0.8], [0], [0.2], 500)[0, 0], expected)
        self.assertEqual(router.arc_path(0, 0.8, 0, 0.2, 500), [0, 1, 2, 0])

    def test_cache_shared_for_multiple_destination_candidates(self):
        self.router.distance_matrix([0, 0], [0.2, 0.8], [1, 2], [0.5, 0.5], 500)
        self.assertEqual(self.router.searches, 1)
        self.assertGreaterEqual(self.router.cache_hits, 1)

    def test_distance_bucket_does_not_relax_requested_bound(self):
        self.assertTrue(np.isinf(self.router.node_distance(0, 2, 199)))
        self.assertIsNone(self.router.node_path(0, 2, 199))
        self.assertEqual(self.router.node_distance(0, 2, 200), 200)

    def test_cache_stays_bounded(self):
        router = DirectedRouter(self.net, cache_max=2)
        for source in range(4):
            router.node_distance(source, source, 300)
        self.assertEqual(len(router.cache), 2)

    def test_zero_length_arc(self):
        net = network([[0, 0], [0, 0], [100, 0]], [[0, 1], [1, 2]])
        router = DirectedRouter(net)
        self.assertEqual(router.distance_matrix([0], [0.8], [0], [0.2], 200)[0, 0], 0)
        self.assertEqual(router.distance_matrix([0], [0.5], [1], [0.5], 200)[0, 0], 50)


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.net = network([[0, 0], [100, 0], [200, 0], [300, 0], [0, 100], [100, 100]],
                           [[0, 1], [1, 2], [2, 3], [4, 5]])

    def test_viterbi_uses_reachable_alternative_over_local_winner(self):
        c = candidates([[0, 3], [1, 3], [2, 3]])
        p = np.asarray([[0.99, 0.01], [0.6, 0.4], [0.6, 0.4]])
        r = decode(self.net, c, p, [[50, 0], [150, 0], [250, 0]], [0, 2, 4])
        np.testing.assert_array_equal(r.arc_ids, [0, 1, 2])
        self.assertEqual(r.route_segments, [[0, 1, 2]])
        self.assertEqual(r.warnings, [])

    def test_reconstructs_unobserved_intermediate_arc(self):
        r = decode(self.net, candidates([[0], [2]]), [[1], [1]], [[50, 0], [250, 0]], [0, 5])
        self.assertEqual(r.route_arc_ids, [0, 1, 2])

    def test_gap_starts_new_segment(self):
        r = decode(self.net, candidates([[0], [2]]), [[1], [1]], [[50, 0], [250, 0]], [0, 100])
        self.assertEqual(r.route_segments, [[0], [2]])
        self.assertEqual(r.warnings[0]["reason"], "large_gps_gap")
        self.assertEqual(r.stats["restart_count"], 1)

    def test_disconnected_transition_is_explicit(self):
        r = decode(self.net, candidates([[0], [3]]), [[1], [1]], [[50, 0], [50, 100]], [0, 2])
        self.assertEqual(r.route_segments, [[0], [3]])
        self.assertEqual(r.warnings[0]["reason"], "unreachable_directed_transition")

    def test_missing_row_retains_observation_alignment(self):
        r = decode(self.net, candidates([[0], [-1], [2]]), [[1], [0], [1]],
                   [[50, 0], [150, 0], [250, 0]], [0, 2, 4])
        np.testing.assert_array_equal(r.arc_ids, [0, -1, 2])
        self.assertEqual(r.route_segments, [[0], [2]])
        self.assertAlmostEqual(r.stats["coverage"], 2 / 3)

    def test_single_observation_and_all_missing(self):
        r = decode(self.net, candidates([[0, 3]]), [[0.1, 0.9]], [[50, 100]], [0])
        self.assertEqual(r.route_segments, [[3]])
        r = decode(self.net, candidates([[-1]]), [[np.nan]], [[0, 0]], [0])
        self.assertEqual(r.arc_ids.tolist(), [-1])
        self.assertEqual(r.route_segments, [])

    def test_duplicate_timestamp_breaks_without_dividing_by_zero(self):
        r = decode(self.net, candidates([[0], [1]]), [[1], [1]], [[50, 0], [150, 0]], [3, 3])
        self.assertEqual(r.route_segments, [[0], [1]])
        self.assertEqual(r.warnings[0]["reason"], "invalid_timestamp_interval")

    def test_invalid_gps_retains_missing_marker(self):
        r = decode(self.net, candidates([[0], [1]]), [[1], [1]], [[50, 0], [np.nan, 0]], [0, 2])
        self.assertEqual(r.arc_ids.tolist(), [0, -1])

    def test_probability_nan_zero_and_infinity_are_safe(self):
        r = decode(self.net, candidates([[0, 3], [1, 3]]), [[0, np.nan], [np.inf, 0]],
                   [[50, 0], [150, 0]], [0, 2])
        self.assertEqual(len(r.arc_ids), 2)
        self.assertTrue(np.all(r.arc_ids >= 0))

    def test_graph_id_collision_does_not_share_router(self):
        c = candidates([[0], [1]])
        decode(self.net, c, [[1], [1]], [[50, 0], [150, 0]], [0, 2])
        other = network([[0, 0], [100, 0], [200, 0]], [[0, 1], [2, 1]])
        r = decode(other, c, [[1], [1]], [[50, 0], [150, 0]], [0, 2])
        self.assertEqual(r.warnings[0]["reason"], "unreachable_directed_transition")

    def test_continuity_counts_missing_and_unreachable(self):
        m = evaluate_continuity(self.net, [0, 1, -1, 3], [0.5] * 4,
                                [[50, 0], [150, 0], [200, 0], [50, 100]], [0, 2, 4, 6])
        self.assertEqual(m["eligible_transitions"], 3)
        self.assertEqual(m["feasible_transitions"], 1)
        self.assertEqual(m["missing_transitions"], 2)
        self.assertAlmostEqual(m["route_continuity"], 1 / 3)
        m = evaluate_continuity(self.net, [0, 3], [0.5, 0.5], [[50, 0], [50, 100]], [0, 2])
        self.assertEqual(m["route_continuity"], 0)
        self.assertEqual(m["unreachable_transitions"], 1)

    def test_continuity_excludes_only_exogenous_time_gap(self):
        m = evaluate_continuity(self.net, [0, 3], [0.5, 0.5], [[50, 0], [50, 100]], [0, 100])
        self.assertEqual(m["eligible_transitions"], 0)
        self.assertIsNone(m["route_continuity"])
        self.assertEqual(m["excluded_time_gaps"], 1)


if __name__ == "__main__":
    unittest.main()
