"""Shared reference-free inputs for the contextual boosting artifact."""

from .candidate_refinement import nearest_candidates
from .context_features import FEATURE_NAMES, build_context_features

K = 20
RADIUS = 250.0
VERSION = 'context-boost-v1'


def context_inputs(network, xy, timestamps):
    candidates = nearest_candidates(network, xy, k=K, radius=RADIUS)
    features, motion = build_context_features(network, candidates, xy, timestamps)
    return candidates, features, motion
