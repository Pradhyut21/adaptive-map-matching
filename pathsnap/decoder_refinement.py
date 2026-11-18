"""Reference-free decoder adjustments for noisy GPS progress and motion."""

from dataclasses import asdict, dataclass, replace
import math
from time import perf_counter

import numpy as np

from .decoder import DecoderConfig, DirectedRouter, _bound, decode
from .features import motion_features


@dataclass(frozen=True)
class RefinementConfig:
    """Controls applied before the existing directed Viterbi decoder."""

    same_arc_jitter_m: float = 0.0
    smoothing_half_window: int = 0
    smoothing_fraction: float = 0.65
    smooth_projections: bool = False
    emission_temperature: float = 1.0
    realign_predicted_route: bool = False
    distance_prior_sigma_m: float = 0.0
    max_states: int = 0
    route_assignment_classifier_weight: float = 0.15

    def __post_init__(self):
        if not math.isfinite(self.same_arc_jitter_m) or not 0 <= self.same_arc_jitter_m <= 15:
            raise ValueError("same_arc_jitter_m must be between 0 and 15 metres")
        if not isinstance(self.smoothing_half_window, int) or not 0 <= self.smoothing_half_window <= 6:
            raise ValueError("smoothing_half_window must be an integer between 0 and 6")
        if not 0 <= self.smoothing_fraction <= 1:
            raise ValueError("smoothing_fraction must be between 0 and 1")
        if not math.isfinite(self.emission_temperature) or self.emission_temperature <= 0:
            raise ValueError("emission_temperature must be finite and positive")
        if not math.isfinite(self.distance_prior_sigma_m) or self.distance_prior_sigma_m < 0:
            raise ValueError("distance prior sigma must be finite and nonnegative")
        if not isinstance(self.max_states, int) or self.max_states < 0:
            raise ValueError("max_states must be a nonnegative integer")
        if not math.isfinite(self.route_assignment_classifier_weight) or self.route_assignment_classifier_weight < 0:
            raise ValueError("route assignment classifier weight must be finite and nonnegative")


def smooth_trajectory(xy, timestamps, gap_seconds, half_window, fraction):
    """Fit short weighted local lines without joining separate GPS sessions."""
    xy = np.asarray(xy, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if not half_window or not len(xy) or not fraction:
        return xy.copy()
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(timestamps)
    delta_t = np.diff(timestamps)
    gaps = (~valid[:-1] | ~valid[1:] | ~np.isfinite(delta_t)
            | (delta_t <= 0) | (delta_t > gap_seconds))
    session = np.r_[0, np.cumsum(gaps)]
    n = len(xy)
    weight_sum = np.ones(n)
    time_sum = np.zeros(n)
    time_square = np.zeros(n)
    position_sum = np.zeros_like(xy)
    time_position = np.zeros_like(xy)
    rows = np.arange(n)
    for offset in range(-half_window, half_window + 1):
        if not offset:
            continue
        at = rows + offset
        here = rows[(at >= 0) & (at < n)]
        neighbors = here + offset
        keep = (session[here] == session[neighbors]) & valid[here] & valid[neighbors]
        here, neighbors = here[keep], neighbors[keep]
        elapsed = timestamps[neighbors] - timestamps[here]
        keep = np.abs(elapsed) <= 2.5 * half_window
        here, neighbors, elapsed = here[keep], neighbors[keep], elapsed[keep]
        weight = (half_window + 1 - abs(offset)) / (half_window + 1)
        displacement = xy[neighbors] - xy[here]
        weight_sum[here] += weight
        time_sum[here] += weight * elapsed
        time_square[here] += weight * elapsed * elapsed
        position_sum[here] += weight * displacement
        time_position[here] += (weight * elapsed)[:, None] * displacement

    denom = weight_sum * time_square - time_sum * time_sum
    usable = denom > 1e-8
    adjustment = np.zeros_like(xy)
    adjustment[usable] = (
        time_square[usable, None] * position_sum[usable]
        - time_sum[usable, None] * time_position[usable]
    ) / denom[usable, None]
    return xy + fraction * adjustment


class JitterAwareRouter(DirectedRouter):
    """Represent small reverse GPS projection changes as a same-arc dwell."""

    def __init__(self, network, cfg, same_arc_jitter_m):
        super().__init__(network, cfg.cache_max, cfg.distance_bucket_m)
        self.same_arc_jitter_m = same_arc_jitter_m

    def distance_matrix(self, previous_arcs, previous_fractions, next_arcs,
                        next_fractions, bound):
        previous_arcs = np.asarray(previous_arcs, dtype=int)
        next_arcs = np.asarray(next_arcs, dtype=int)
        result = np.full((len(previous_arcs), len(next_arcs)), np.inf)
        previous_indexes = np.flatnonzero((previous_arcs >= 0) & (previous_arcs < len(self.arcs)))
        next_indexes = np.flatnonzero((next_arcs >= 0) & (next_arcs < len(self.arcs)))
        if not len(previous_indexes) or not len(next_indexes):
            return result

        first = previous_arcs[previous_indexes]
        last = next_arcs[next_indexes]
        sources = self.arcs[first, 1]
        destinations = self.arcs[last, 0]
        next_fraction = np.clip(np.asarray(next_fractions)[next_indexes], 0, 1)
        previous_fraction = np.clip(np.asarray(previous_fractions)[previous_indexes], 0, 1)
        remaining = self.lengths[first] * (1 - previous_fraction)
        prefixes = self.lengths[last] * next_fraction
        routed = np.empty((len(first), len(last)))
        for source in np.unique(sources):
            tree = self._tree(int(source), bound)
            routed[sources == source] = [tree.distances.get(int(end), math.inf) for end in destinations]
        distance = remaining[:, None] + routed + prefixes[None, :]
        progress = (next_fraction[None, :] - previous_fraction[:, None]) * self.lengths[first, None]
        same_arc = first[:, None] == last[None, :]
        direct = same_arc & (progress >= -self.same_arc_jitter_m)
        distance[direct] = np.minimum(distance[direct], np.maximum(0, progress[direct]))
        result[np.ix_(previous_indexes, next_indexes)] = np.where(distance <= bound + 1e-9, distance, np.inf)
        return result

    def arc_path(self, first, first_fraction, last, last_fraction, bound):
        if first == last and first >= 0:
            progress = (last_fraction - first_fraction) * self.lengths[first]
            if -self.same_arc_jitter_m <= progress < 0:
                return [int(first)]
        return super().arc_path(first, first_fraction, last, last_fraction, bound)


@dataclass
class _RoutingView:
    arcs: np.ndarray
    lengths: np.ndarray
    nodes_xy: np.ndarray
    _pathsnap_directed_router: DirectedRouter


def refinement_inputs(network, candidates, track_xy, timestamps, config, refinement):
    """Apply the same projection and bounded routing assumptions in evaluation."""
    xy = smooth_trajectory(track_xy, timestamps, config.gap_seconds,
                           refinement.smoothing_half_window, refinement.smoothing_fraction)
    if refinement.smoothing_half_window and refinement.smooth_projections and len(network.arcs):
        valid = (candidates.arc_ids >= 0) & (candidates.arc_ids < len(network.arcs))
        arcs_safe = np.where(valid, candidates.arc_ids, 0)
        start = network.nodes_xy[network.arcs[arcs_safe, 0]]
        vector = network.nodes_xy[network.arcs[arcs_safe, 1]] - start
        magnitude = np.maximum(np.sum(vector * vector, axis=-1), 1e-12)
        fraction = np.clip(np.sum((xy[:, None, :] - start) * vector, axis=-1) / magnitude, 0, 1)
        xy_projection = start + fraction[..., None] * vector
        candidates = replace(candidates, fractions=fraction, xy=xy_projection)
    router = JitterAwareRouter(network, config, refinement.same_arc_jitter_m)
    view = _RoutingView(network.arcs, network.lengths, network.nodes_xy, router)
    return view, candidates, xy


def _continuous_runs(result):
    if not len(result.arc_ids):
        return []
    boundaries = sorted({0, len(result.arc_ids), *result.break_indices,
                         *(int(i) for i in np.flatnonzero(np.diff(result.arc_ids >= 0)) + 1)})
    return [(start, stop) for start, stop in zip(boundaries[:-1], boundaries[1:])
            if result.arc_ids[start] >= 0]


def _align_run(network, candidates, probabilities, xy, timestamps, heading, reliability,
               result, first_row, stop_row, route, config, refinement):
    """Find a monotone observation assignment on an already decoded route."""
    occurrences = {}
    for position, arc in enumerate(route):
        occurrences.setdefault(arc, []).append(position)
    route = np.asarray(route, dtype=int)
    cumulative = np.r_[0.0, np.cumsum(network.lengths[route])]
    backpointers, choices = [], []
    previous_positions = previous_offsets = previous_score = None
    expected = 0
    # The expected position tracks a matching occurrence of each original
    # Viterbi state and limits duplicate occurrences along long looped routes.
    for row in range(first_row, stop_row):
        original = int(result.arc_ids[row])
        if original not in occurrences:
            return None
        original_positions = np.asarray(occurrences[original])
        next_index = np.searchsorted(original_positions, expected)
        if next_index < len(original_positions):
            expected = int(original_positions[next_index])
        pos, col = [], []
        for candidate_col in np.flatnonzero(candidates.valid[row]):
            for position in occurrences.get(int(candidates.arc_ids[row, candidate_col]), ()):
                if abs(cumulative[position] - cumulative[expected]) <= 300:
                    pos.append(position)
                    col.append(candidate_col)
        if not pos:
            return None
        pos, col = np.asarray(pos), np.asarray(col)
        arcs = route[pos]
        offsets = cumulative[pos] + candidates.fractions[row, col] * network.lengths[arcs]
        if hasattr(network, "bearings"):
            angle = network.bearings[arcs] - heading[row]
        else:
            vector = network.nodes_xy[network.arcs[arcs, 1]] - network.nodes_xy[network.arcs[arcs, 0]]
            angle = np.arctan2(vector[:, 1], vector[:, 0]) - heading[row]
        angular_error = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))
        emission = -(candidates.distances[row, col] / 20) ** 2 - 1.5 * reliability[row] * angular_error
        emission += refinement.route_assignment_classifier_weight * np.log(
            np.clip(probabilities[row, col], config.probability_floor, 1))
        emission -= np.abs(cumulative[expected] - cumulative[pos]) / 200
        if row == first_row:
            score = emission
            back = np.full(len(pos), -1, dtype=int)
        else:
            elapsed = timestamps[row] - timestamps[row - 1]
            displacement = float(np.linalg.norm(xy[row] - xy[row - 1]))
            travel = offsets[None, :] - previous_offsets[:, None]
            forward = pos[None, :] >= previous_positions[:, None]
            minimum = -max(0.001, refinement.same_arc_jitter_m)
            within_bounds = (travel >= minimum) & (travel <= _bound(displacement, elapsed, config))
            transition = -np.abs(np.maximum(travel, 0) - displacement) / 50
            transition -= np.maximum(travel / elapsed - config.max_speed, 0) / 10
            score_table = np.where(forward & within_bounds, previous_score[:, None] + transition, -np.inf)
            back = np.argmax(score_table, axis=0)
            score = emission + score_table[back, np.arange(len(pos))]
        maximum = np.max(score)
        if not np.isfinite(maximum):
            return None
        previous_positions, previous_offsets, previous_score = pos, offsets, score - maximum
        backpointers.append(back)
        choices.append(col)
    selected = np.empty(stop_row - first_row, dtype=int)
    state = np.argmax(previous_score)
    for row in range(len(selected) - 1, -1, -1):
        selected[row] = choices[row][state]
        state = backpointers[row][state]
    return selected


def realign_route_points(network, candidates, probabilities, xy, timestamps,
                         result, config, refinement):
    """Reassign points along the predicted path using only original candidates."""
    runs = _continuous_runs(result)
    if len(runs) != len(result.route_segments):
        return result
    heading, reliable, _, _ = motion_features(xy, timestamps, gap_seconds=config.gap_seconds)
    selected = result.candidate_indices.copy()
    for (start, stop), route in zip(runs, result.route_segments):
        updated = _align_run(network, candidates, probabilities, xy, timestamps, heading, reliable,
                             result, start, stop, route, config, refinement)
        if updated is not None:
            selected[start:stop] = updated
    arc_ids = result.arc_ids.copy()
    valid = selected >= 0
    arc_ids[valid] = candidates.arc_ids[np.flatnonzero(valid), selected[valid]]
    stats = dict(result.stats)
    stats["realigned_observations"] = int(np.sum(arc_ids != result.arc_ids))
    return replace(result, arc_ids=arc_ids, candidate_indices=selected, stats=stats)


def decode_refined(network, candidates, probabilities, track_xy, timestamps,
                   config=None, refinement=None, realignment_probabilities=None):
    """Decode using bounded projection-noise tolerance and optional smoothing."""
    started = perf_counter()
    cfg = DecoderConfig(**config) if isinstance(config, dict) else config or DecoderConfig()
    tweak = RefinementConfig(**refinement) if isinstance(refinement, dict) else refinement or RefinementConfig()
    view, adjusted, xy = refinement_inputs(network, candidates, track_xy, timestamps, cfg, tweak)
    classifier_probabilities = probabilities if realignment_probabilities is None else realignment_probabilities
    if tweak.emission_temperature != 1.0:
        potentials = np.clip(np.nan_to_num(probabilities, nan=0, posinf=1, neginf=0), cfg.probability_floor, 1)
        probabilities = potentials ** (1 / tweak.emission_temperature)
    if tweak.distance_prior_sigma_m:
        log_scores = np.log(np.clip(probabilities, cfg.probability_floor, 1.0))
        log_scores -= 0.5 * np.square(adjusted.distances / tweak.distance_prior_sigma_m)
        row_max = np.max(log_scores, axis=1, keepdims=True, initial=-np.inf)
        probabilities = np.exp(log_scores - np.where(np.isfinite(row_max), row_max, 0.0))
    columns = None
    trellis = adjusted
    if 0 < tweak.max_states < adjusted.arc_ids.shape[1]:
        ranked = np.where(adjusted.valid, probabilities, -1.0)
        columns = np.argsort(-ranked, axis=1, kind="stable")[:, :tweak.max_states]
        rows = np.arange(len(columns))[:, None]
        trellis = replace(adjusted, arc_ids=adjusted.arc_ids[rows, columns],
                          distances=adjusted.distances[rows, columns], fractions=adjusted.fractions[rows, columns],
                          xy=adjusted.xy[rows, columns])
        probabilities = probabilities[rows, columns]
    result = decode(view, trellis, probabilities, xy, timestamps, config=cfg)
    if columns is not None:
        selected = result.candidate_indices.copy()
        valid = selected >= 0
        selected[valid] = columns[np.flatnonzero(valid), selected[valid]]
        result = replace(result, candidate_indices=selected)
    if tweak.realign_predicted_route:
        result = realign_route_points(network, adjusted, classifier_probabilities,
                                      xy, timestamps, result, cfg, tweak)
    result.stats["refinement"] = asdict(tweak)
    result.stats["viterbi_seconds"] = result.stats["inference_seconds"]
    result.stats["inference_seconds"] = perf_counter() - started
    return result


def combine_scores(current, previous, previous_weight):
    """Geometric blend that preserves relative candidate emission scores."""
    return np.maximum(current, 1e-8) ** (1 - previous_weight) * np.maximum(previous, 1e-8) ** previous_weight


def decode_with_previous(network, candidates, current, previous, xy, times,
                         config, refinement, previous_weight, preserve_route):
    """Use both boosters for routing, or use the new scores on the previous path."""
    if preserve_route:
        return decode_refined(network, candidates, previous, xy, times, config, refinement,
                              realignment_probabilities=current)
    scores = combine_scores(current, previous, previous_weight) if previous_weight else current
    return decode_refined(network, candidates, scores, xy, times, config, refinement)
