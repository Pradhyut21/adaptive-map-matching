"""Reference-free features for distinguishing nearby directed road segments."""

import numpy as np

from .features import FEATURE_NAMES as BASE_NAMES


BASE_COLUMNS = tuple(i for i, name in enumerate(BASE_NAMES) if name not in {"heading_cos", "heading_sin"})
FEATURE_NAMES = [BASE_NAMES[i] for i in BASE_COLUMNS] + [
    "heading_agreement_4", "heading_agreement_10",
    "incoming_heading_agreement", "outgoing_heading_agreement",
    "incoming_lateral_m", "outgoing_lateral_m",
    "smoothed_distance_m", "along_arc_m", "remaining_arc_m",
    "incoming_endpoint_distance_m", "outgoing_endpoint_distance_m",
    "incoming_direct_connection_cost", "outgoing_direct_connection_cost",
    "earlier_arc_presence", "later_arc_presence", "local_motion_speed_mps",
]


def _row_bounds(timestamps, gap_seconds=60.0):
    """Return row-wise limits for symmetric and one-sided GPS windows."""
    n = len(timestamps)
    dt = np.diff(timestamps)
    breaks = np.r_[0, np.flatnonzero(~np.isfinite(dt) | (dt <= 0) | (dt > gap_seconds)) + 1, n]
    lengths = np.diff(breaks)
    return np.repeat(breaks[:-1], lengths), np.repeat(breaks[1:] - 1, lengths)


def _agreement(unit, displacement):
    # Normalize on a minimum 15 m displacement. A nearly stationary reading
    # therefore contributes less, even when its angle looks persuasive.
    scale = np.maximum(np.linalg.norm(displacement, axis=1), 15.0)
    direction = displacement / scale[:, None]
    return np.einsum("nkd,nd->nk", unit, direction)


def _lateral_distance(unit, points, starts):
    offset = points[:, None, :] - starts
    cross = offset[..., 0] * unit[..., 1] - offset[..., 1] * unit[..., 0]
    return np.abs(cross)


def _connection_cost(network, candidates, xy, neighbor_rows, incoming):
    """Minimum direct (same arc or shared intersection) neighbor cost.

    This feature supplies local evidence, and a missing direct connection
    can still be bridged by a multi-hop route in the sequence decoder.
    """
    arc_ids = candidates.arc_ids
    n, k = arc_ids.shape
    cost = np.empty((n, k), dtype=np.float32)
    for lo in range(0, n, 768):
        current = np.arange(lo, min(lo + 768, n))
        nearby = neighbor_rows[current]
        first_rows, last_rows = (nearby, current) if incoming else (current, nearby)
        first_ids = arc_ids[first_rows]
        last_ids = arc_ids[last_rows]
        safe_first, safe_last = np.maximum(first_ids, 0), np.maximum(last_ids, 0)
        first_lengths = network.lengths[safe_first]
        last_lengths = network.lengths[safe_last]
        first_tail = first_lengths * (1 - candidates.fractions[first_rows])
        last_prefix = last_lengths * candidates.fractions[last_rows]
        appends = network.arcs[safe_first, 1, None] == network.arcs[safe_last, 0][:, None, :]
        same = first_ids[:, :, None] == last_ids[:, None, :]
        forward = last_prefix[:, None, :] - (first_lengths - first_tail)[:, :, None]
        distance = np.where(same, forward, first_tail[:, :, None] + last_prefix[:, None, :])
        usable = (same | appends) & (first_ids[:, :, None] >= 0) & (last_ids[:, None, :] >= 0)
        gps_step = np.linalg.norm(xy[last_rows] - xy[first_rows], axis=1)
        neighbor_distance = candidates.distances[nearby]
        neighbor_penalty = neighbor_distance[:, :, None] if incoming else neighbor_distance[:, None, :]
        mismatch = np.abs(distance - gps_step[:, None, None]) + 0.6 * neighbor_penalty
        mismatch = np.where(usable, mismatch, 300.)
        best = mismatch.min(axis=1 if incoming else 2)
        cost[current] = np.where((current != nearby)[:, None], np.log1p(np.minimum(best, 300.)), 0.)
    return cost


def _presence(candidate_ids, neighbor_rows):
    """Similarity of candidate membership across three preceding/following rows."""
    n, k = candidate_ids.shape
    membership = np.zeros((n, k), dtype=np.float32)
    for row_offset, rows in enumerate(neighbor_rows):
        for lo in range(0, n, 768):
            stop = min(lo + 768, n)
            current_ids = candidate_ids[lo:stop]
            selected = rows[lo:stop]
            other_ids = candidate_ids[selected]
            match = (current_ids[:, :, None] == other_ids[:, None, :]).any(axis=2)
            usable = (selected != np.arange(lo, stop))[:, None]
            membership[lo:stop] += np.where(usable & (current_ids >= 0), match / (row_offset + 1), 0.)
    return membership


def build_context_features(network, candidates, xy, timestamps):
    """Return bounded trajectory context for each valid directed candidate.

    Neighborhoods are restricted to their timestamp segment and suitable for
    whole-track inference. Arc identifiers are only used to look up geometry
    and adjacency. The feature columns never include an identifier.
    """
    xy = np.asarray(xy, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    valid = candidates.valid
    n, k = valid.shape
    feature_matrix = np.zeros((n, k, len(FEATURE_NAMES)), dtype=np.float32)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return feature_matrix, dict.fromkeys(("heading", "heading_reliability", "speed", "time_delta"), empty)

    row = np.arange(n)
    left, right = _row_bounds(timestamps)
    prior = np.maximum(row - 2, left)
    following = np.minimum(row + 2, right)
    motion = xy[following] - xy[prior]
    heading = np.arctan2(motion[:, 1], motion[:, 0])
    reliable = np.minimum(np.linalg.norm(motion, axis=1) / 15., 1.)
    dt = np.r_[1., np.diff(timestamps)]
    speed = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(dt[1:], 1e-6)]
    if n > 1:
        speed[0] = speed[1]
    speed[(dt <= 0) | (dt > 60)] = 0.
    speed = np.minimum(speed, 200.)
    metadata = {"heading": heading, "heading_reliability": reliable, "speed": speed, "time_delta": dt}
    if not valid.any():
        return feature_matrix, metadata

    safe = np.maximum(candidates.arc_ids, 0)
    lengths = network.lengths[safe]
    starts, ends = network.starts[safe], network.ends[safe]
    unit = (ends - starts) / np.maximum(lengths[..., None], 1e-9)
    distances = np.where(valid, candidates.distances, 0.)
    bearing_difference = network.bearings[safe] - heading[:, None]
    angle = np.abs(np.arctan2(np.sin(bearing_difference), np.cos(bearing_difference)))
    base = (
        distances, np.log1p(distances), angle, angle * reliable[:, None], reliable[:, None],
        candidates.fractions, np.log1p(lengths), np.log1p(speed[:, None]),
        np.log1p(np.maximum(dt[:, None], 0)), np.arange(k), distances - distances[:, :1],
        valid.sum(axis=1)[:, None],
    )
    for column, values in enumerate(base):
        feature_matrix[..., column] = values

    earlier = [np.maximum(row - delta, left) for delta in (1, 2, 3)]
    later = [np.minimum(row + delta, right) for delta in (1, 2, 3)]
    before4, after4 = np.maximum(row - 4, left), np.minimum(row + 4, right)
    before10, after10 = np.maximum(row - 10, left), np.minimum(row + 10, right)
    segment_motion = xy[after4] - xy[before4]
    local_speed = np.linalg.norm(segment_motion, axis=1) / np.maximum(timestamps[after4] - timestamps[before4], 1e-6)

    # The five-point median limits the influence of a single noisy reading.
    median_rows = np.stack((earlier[1], earlier[0], row, later[0], later[1]))
    smoothed = np.median(xy[median_rows], axis=0)
    smooth_offset = smoothed[:, None, :] - starts
    smooth_along = np.sum(smooth_offset * unit, axis=-1)
    smooth_projection = starts + unit * np.clip(smooth_along, 0., lengths)[..., None]
    smooth_distance = np.linalg.norm(smoothed[:, None, :] - smooth_projection, axis=-1)
    along = np.sum((xy[:, None, :] - starts) * unit, axis=-1)
    earlier_points, later_points = xy[earlier[2]], xy[later[2]]
    context = (
        _agreement(unit, segment_motion),
        _agreement(unit, xy[after10] - xy[before10]),
        _agreement(unit, xy - earlier_points),
        _agreement(unit, later_points - xy),
        np.minimum(_lateral_distance(unit, earlier_points, starts), 300.),
        np.minimum(_lateral_distance(unit, later_points, starts), 300.),
        np.minimum(smooth_distance, 300.),
        np.clip(along, -300., 1000.),
        np.clip(lengths - along, -300., 1000.),
        np.minimum(np.linalg.norm(earlier_points[:, None, :] - starts, axis=-1), 300.),
        np.minimum(np.linalg.norm(later_points[:, None, :] - ends, axis=-1), 300.),
        _connection_cost(network, candidates, xy, earlier[0], incoming=True),
        _connection_cost(network, candidates, xy, later[0], incoming=False),
        _presence(candidates.arc_ids, earlier),
        _presence(candidates.arc_ids, later),
        np.minimum(local_speed, 200.)[:, None],
    )
    for column, values in enumerate(context, start=len(BASE_COLUMNS)):
        feature_matrix[..., column] = values
    feature_matrix[~valid] = 0.
    return feature_matrix, metadata
