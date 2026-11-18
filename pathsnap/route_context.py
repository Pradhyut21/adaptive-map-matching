"""Features based on a road's fit to nearby fixes and its local topology."""

import numpy as np

from .context_features import FEATURE_NAMES as CONTEXT_NAMES, _row_bounds, build_context_features
from .geometry import project_segments


FEATURE_NAMES = CONTEXT_NAMES + [
    "neighbor_distance_mean", "neighbor_distance_max", "distance_ratio",
    "endpoint_distance_m", "incoming_degree", "outgoing_degree", "has_reverse_arc",
]


def build_route_context(network, candidates, xy, timestamps):
    context, motion = build_context_features(network, candidates, xy, timestamps)
    extra = np.zeros((*candidates.arc_ids.shape, len(FEATURE_NAMES) - len(CONTEXT_NAMES)), dtype=np.float32)
    rows, columns = np.nonzero(candidates.valid)
    if len(rows):
        arcs = candidates.arc_ids[rows, columns]
        first, last = _row_bounds(timestamps)
        source, target = network.arcs.T
        incoming = np.bincount(target, minlength=len(network.nodes_xy))
        outgoing = np.bincount(source, minlength=len(network.nodes_xy))
        order = np.sort(source.astype(np.int64) * len(incoming) + target)
        reversed_keys = target[arcs].astype(np.int64) * len(incoming) + source[arcs]
        indexes = np.minimum(np.searchsorted(order, reversed_keys), len(order) - 1)
        for lo in range(0, len(rows), 32768):
            take = slice(lo, lo + 32768)
            r, c, a = rows[take], columns[take], arcs[take]
            window = np.clip(r[:, None] + np.arange(-3, 4), first[r, None], last[r, None])
            distance, _, _ = project_segments(xy[window], network.starts[a, None], network.ends[a, None])
            fraction = candidates.fractions[r, c]
            extra[r, c] = np.column_stack((
                distance.mean(axis=1), distance.max(axis=1),
                (1.0 + candidates.distances[r, c]) / (1.0 + candidates.distances[r, 0]),
                np.minimum(fraction, 1.0 - fraction) * network.lengths[a],
                incoming[source[a]], outgoing[target[a]], order[indexes[take]] == reversed_keys[take],
            ))
    return np.concatenate((context, extra), axis=2), motion
