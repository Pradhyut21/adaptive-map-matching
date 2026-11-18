"""Exact, reference-only route alignment for labelling and evaluation."""

import numpy as np

from .geometry import project_segments
from .context_features import _row_bounds


def align_ordered_route(network, xy, timestamps, route):
    """Assign fixes to nondecreasing route positions, allowing GPS jitter.

    Capped distance lets isolated outliers have limited influence. The motion
    term distinguishes opposite directions on geometrically identical arcs.
    Return a label for every observation plus the geometric distance to it.
    """
    route = np.asarray(route, dtype=int)
    if not len(route):
        raise ValueError("Reference route must be nonempty")
    n = len(xy)
    if n == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty, np.empty(0, dtype=float)

    first, last = _row_bounds(timestamps)
    rows = np.arange(n)
    direction = xy[np.minimum(rows + 2, last)] - xy[np.maximum(rows - 2, first)]
    movement = np.linalg.norm(direction, axis=1)
    direction /= np.maximum(movement, 3.0)[:, None]
    unit_arcs = network.vectors[route] / np.maximum(network.lengths[route], 1e-12)[:, None]
    starts, ends = network.starts[route], network.ends[route]
    along = np.arange(len(route))
    back_dtype = np.uint16 if len(route) <= 65536 else np.uint32
    ancestors = np.empty((n, len(route)), dtype=back_dtype)
    block_size = max(1, min(256, 200000 // len(route)))
    cost = np.zeros(len(route))
    for first_row in range(0, n, block_size):
        stop = min(first_row + block_size, n)
        distances, _, _ = project_segments(xy[first_row:stop, None, :], starts, ends)
        cosine = np.clip(direction[first_row:stop] @ unit_arcs.T, -1, 1)
        moving = movement[first_row:stop, None] >= 3.0
        emissions = np.minimum(distances, 100.0) + np.where(moving, 7.5 * (1.0 - cosine), 0.0)
        for row, emission in enumerate(emissions, start=first_row):
            prefix = np.minimum.accumulate(cost)
            improvement = np.r_[True, cost[1:] < prefix[:-1]]
            ancestors[row] = np.maximum.accumulate(np.where(improvement, along, 0))
            cost = prefix + emission

    positions = np.empty(n, dtype=int)
    positions[-1] = np.argmin(cost)
    for row in range(n - 1, 0, -1):
        positions[row - 1] = ancestors[row, positions[row]]
    arcs = route[positions]
    distances, _, _ = project_segments(xy, network.starts[arcs], network.ends[arcs])
    return arcs, positions, distances
