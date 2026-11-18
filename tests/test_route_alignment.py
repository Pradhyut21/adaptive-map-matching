"""Tests for global reference alignment and candidate-index preservation."""

from types import SimpleNamespace
import unittest

import numpy as np

from pathsnap.decoder_refinement import RefinementConfig, decode_refined
from pathsnap.geometry import CandidateSet
from pathsnap.route_alignment import align_ordered_route


def line_network():
    nodes = np.array([[0., 0.], [30., 0.], [60., 0.]])
    arcs = np.array([[0, 1], [1, 2], [2, 1], [1, 0]])
    starts, ends = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    vectors = ends - starts
    return SimpleNamespace(nodes_xy=nodes, arcs=arcs, starts=starts, ends=ends, vectors=vectors,
                           lengths=np.linalg.norm(vectors, axis=1))


class RouteAlignmentTests(unittest.TestCase):
    def test_jitter_stays_on_ordered_arc_and_far_fix_keeps_label(self):
        net = line_network()
        xy = np.array([[8., 2.], [20., 2.], [15., 2.], [39., 2.], [50., 90.]])
        labels, positions, distances = align_ordered_route(net, xy, np.arange(len(xy)), [0, 1])
        np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1])
        self.assertTrue(np.all(np.diff(positions) >= 0))
        self.assertGreater(distances[-1], 60.)

    def test_emission_shortlist_preserves_original_candidate_index(self):
        net = line_network()
        ids = np.array([[3, 0, 1], [-1, -1, -1]])
        distances = np.array([[22., 5., 90.], [np.inf, np.inf, np.inf]])
        points = np.array([[10., 0.], [17., 0.]])
        candidates = CandidateSet(ids, distances, np.full(ids.shape, .5), np.zeros((2, 3, 2)))
        with np.errstate(invalid='raise'):
            result = decode_refined(net, candidates, np.array([[.5, .4, .1], [0., 0., 0.]]),
                                    points, np.array([0., 1.]),
                                    refinement=RefinementConfig(distance_prior_sigma_m=10., max_states=1))
        np.testing.assert_array_equal(result.arc_ids, [0, -1])
        np.testing.assert_array_equal(result.candidate_indices, [1, -1])


if __name__ == '__main__':
    unittest.main()
