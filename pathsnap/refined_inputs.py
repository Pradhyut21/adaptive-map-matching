"""Shared reference-free inputs for the contextual boosting artifact."""

from .candidate_refinement import nearest_candidates
from .context_features import FEATURE_NAMES, build_context_features
from .route_context import FEATURE_NAMES as ROUTE_FEATURE_NAMES, build_route_context

K = 20
RADIUS = 250.0
VERSION = 'context-boost-v1'
ROUTE_VERSION = 'route-context-v1'
SCHEMAS = {VERSION: FEATURE_NAMES, ROUTE_VERSION: ROUTE_FEATURE_NAMES}


def context_inputs(network, xy, timestamps, version=VERSION):
    candidates = nearest_candidates(network, xy, k=K, radius=RADIUS)
    builder = build_route_context if version == ROUTE_VERSION else build_context_features
    features, motion = builder(network, candidates, xy, timestamps)
    return candidates, features, motion
