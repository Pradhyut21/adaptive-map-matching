"""Vectorized exact segment candidates, using only the GPS track and graph."""
import numpy as np
import shapely

from .geometry import CandidateSet, project_segments


def nearest_candidates(network, xy, k=20, radius=250.0, max_radius=300.0):
    """Return the nearest directed arcs in distance and arc-ID order.

    The initial radius applies to each point. The retry radius applies only to
    points with no initial hits. Candidate rows remain aligned with GPS points.
    """
    if k < 1 or radius < 0 or max_radius < radius:
        raise ValueError("Expected at least one candidate and ordered nonnegative radii")
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = len(xy)
    candidate_ids = np.full((n, k), -1, dtype=np.int32)
    distances = np.full((n, k), np.inf, dtype=np.float32)
    fractions = np.zeros((n, k), dtype=np.float32)
    projected = np.zeros((n, k, 2), dtype=np.float64)

    for offset in range(0, n, 256):
        block = xy[offset : offset + 256]
        points = shapely.points(block)
        row, arc = network.tree.query(points, predicate="dwithin", distance=radius)
        if max_radius > radius:
            found = np.zeros(len(block), dtype=bool)
            found[row] = True
            empty = np.flatnonzero(~found)
            if len(empty):
                fallback_row, fallback_arc = network.tree.query(
                    points[empty], predicate="dwithin", distance=max_radius
                )
                row = np.concatenate((row, empty[fallback_row]))
                arc = np.concatenate((arc, fallback_arc))
        if not len(row):
            continue

        distance, fraction, projection = project_segments(
            block[row], network.starts[arc], network.ends[arc]
        )
        order = np.lexsort((arc, distance, row))
        sorted_rows = row[order]
        sequence = np.arange(len(order))
        starts_row = np.r_[True, sorted_rows[1:] != sorted_rows[:-1]]
        ranks = sequence - np.maximum.accumulate(np.where(starts_row, sequence, 0))
        within_budget = ranks < k
        source = order[within_budget]
        target = (sorted_rows[within_budget] + offset, ranks[within_budget])
        candidate_ids[target] = arc[source]
        distances[target] = distance[source]
        fractions[target] = fraction[source]
        projected[target] = projection[source]

    return CandidateSet(candidate_ids, distances, fractions, projected)
