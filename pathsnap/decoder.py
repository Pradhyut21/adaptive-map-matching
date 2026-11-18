"""Independent directed map-matching decoder.

The classifier supplies discriminative emission potentials, not calibrated route
probabilities. Routing uses only the supplied graph and observed positions.
No reference-route data is read by this module.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import heapq
import math
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DecoderConfig:
    beta: float = 30.0
    max_speed: float = 55.0
    gap_seconds: float = 60.0
    min_route_m: float = 200.0
    max_route_m: float = 5000.0
    route_slack_m: float = 100.0
    displacement_multiplier: float = 3.0
    speed_penalty_weight: float = 2.0
    cache_max: int = 512
    distance_bucket_m: float = 250.0
    probability_floor: float = 1e-8
    uturn_penalty: float = 0.0
    hard_speed_limit: bool = False

    def __post_init__(self):
        if self.beta <= 0 or self.max_speed <= 0 or self.gap_seconds <= 0:
            raise ValueError("beta, max_speed and gap_seconds must be positive")
        if self.cache_max < 1 or self.cache_max > 512:
            raise ValueError("cache_max must be in [1, 512]")
        if self.distance_bucket_m <= 0 or self.min_route_m <= 0:
            raise ValueError("routing bounds must be positive")
        if self.max_route_m < self.min_route_m:
            raise ValueError("max_route_m must be >= min_route_m")
        if not 0 < self.probability_floor < 0.5:
            raise ValueError("probability_floor must lie between 0 and 0.5")
        if not math.isfinite(self.uturn_penalty) or self.uturn_penalty < 0:
            raise ValueError("U-turn penalty must be finite and nonnegative")


@dataclass
class DecodeResult:
    arc_ids: np.ndarray
    candidate_indices: np.ndarray
    route_arc_ids: list[int]
    route_segments: list[list[int]]
    warnings: list[dict[str, Any]]
    stats: dict[str, Any]
    break_indices: list[int] = field(default_factory=list)


@dataclass
class _Tree:
    distances: dict[int, float]
    parent: dict[int, tuple[int, int]]


class DirectedRouter:
    """Bounded sparse Dijkstra trees, reused across all candidate pairs.

    A tree is shared by destinations with the same origin and distance bucket.
    Adjacency arrays are compact CSR; cached distances are sparse dictionaries.
    The cache belongs to one concrete graph instance and cannot cross graph IDs.
    """

    def __init__(self, network, cache_max: int = 512, bucket_m: float = 250.0):
        if not 1 <= cache_max <= 512 or bucket_m <= 0:
            raise ValueError("invalid router cache settings")
        self.graph_id = str(getattr(network, "graph_id", "anonymous"))
        self.arcs = np.asarray(network.arcs, dtype=np.int64)
        self.lengths = np.asarray(network.lengths, dtype=float)
        n_nodes = len(network.nodes_xy)
        if self.arcs.ndim != 2 or self.arcs.shape[1] != 2:
            raise ValueError("arcs must have shape (n_arcs, 2)")
        if len(self.lengths) != len(self.arcs):
            raise ValueError("one length required for every arc")
        if np.any(self.arcs < 0) or np.any(self.arcs >= n_nodes):
            raise ValueError("arc endpoints outside graph")
        if np.any(~np.isfinite(self.lengths)) or np.any(self.lengths < 0):
            raise ValueError("arc lengths must be finite and nonnegative")
        order = np.argsort(self.arcs[:, 0], kind="stable")
        counts = np.bincount(self.arcs[:, 0], minlength=n_nodes)
        self.indptr = np.concatenate(([0], np.cumsum(counts)))
        self.destinations = self.arcs[order, 1]
        self.edge_ids = order
        self.edge_lengths = self.lengths[order]
        self.cache_max = cache_max
        self.bucket_m = bucket_m
        self.cache: OrderedDict[tuple[int, float], _Tree] = OrderedDict()
        self.cache_hits = 0
        self.searches = 0
        self.nodes_settled = 0

    def _tree(self, source: int, bound: float) -> _Tree:
        bound = max(0.0, float(bound))
        bucket_bound = math.ceil(bound / self.bucket_m) * self.bucket_m
        key = (int(source), bucket_bound)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return cached
        self.searches += 1
        distances = {int(source): 0.0}
        parent: dict[int, tuple[int, int]] = {}
        heap = [(0.0, int(source))]
        while heap:
            distance, node = heapq.heappop(heap)
            if distance != distances.get(node):
                continue
            self.nodes_settled += 1
            for offset in range(self.indptr[node], self.indptr[node + 1]):
                next_distance = distance + float(self.edge_lengths[offset])
                if next_distance > bucket_bound:
                    continue
                destination = int(self.destinations[offset])
                if next_distance < distances.get(destination, math.inf):
                    distances[destination] = next_distance
                    parent[destination] = (node, int(self.edge_ids[offset]))
                    heapq.heappush(heap, (next_distance, destination))
        tree = _Tree(distances, parent)
        self.cache[key] = tree
        if len(self.cache) > self.cache_max:
            self.cache.popitem(last=False)
        return tree

    def node_distance(self, source: int, destination: int, bound: float) -> float:
        distance = self._tree(source, bound).distances.get(int(destination), math.inf)
        return float(distance) if distance <= bound + 1e-9 else math.inf

    def node_path(self, source: int, destination: int, bound: float) -> list[int] | None:
        tree = self._tree(source, bound)
        if tree.distances.get(int(destination), math.inf) > bound + 1e-9:
            return None
        path: list[int] = []
        current = int(destination)
        while current != int(source):
            if current not in tree.parent:
                return None
            current, edge = tree.parent[current]
            path.append(edge)
        return path[::-1]

    def distance_matrix(self, previous_arcs, previous_fractions, next_arcs,
                        next_fractions, bound: float) -> np.ndarray:
        previous_arcs = np.asarray(previous_arcs, dtype=int)
        next_arcs = np.asarray(next_arcs, dtype=int)
        previous_fractions = np.clip(np.asarray(previous_fractions, dtype=float), 0, 1)
        next_fractions = np.clip(np.asarray(next_fractions, dtype=float), 0, 1)
        result = np.full((len(previous_arcs), len(next_arcs)), np.inf)
        if not len(previous_arcs) or not len(next_arcs):
            return result
        valid_p = (previous_arcs >= 0) & (previous_arcs < len(self.arcs))
        valid_n = (next_arcs >= 0) & (next_arcs < len(self.arcs))
        for i in np.flatnonzero(valid_p):
            first = int(previous_arcs[i])
            remaining = self.lengths[first] * (1.0 - previous_fractions[i])
            source = int(self.arcs[first, 1])
            # One bounded tree serves every destination road for this origin.
            tree = self._tree(source, bound)
            for j in np.flatnonzero(valid_n):
                last = int(next_arcs[j])
                distance = remaining + tree.distances.get(int(self.arcs[last, 0]), math.inf)
                distance += self.lengths[last] * next_fractions[j]
                if first == last and (next_fractions[j] >= previous_fractions[i] or self.lengths[first] <= 1e-12):
                    direct = self.lengths[first] * (next_fractions[j] - previous_fractions[i])
                    distance = min(distance, direct)
                if distance <= bound + 1e-9:
                    result[i, j] = max(0.0, distance)
        return result

    def arc_path(self, first: int, first_fraction: float, last: int,
                 last_fraction: float, bound: float) -> list[int] | None:
        if first < 0 or last < 0:
            return None
        if first == last and (last_fraction >= first_fraction or self.lengths[first] <= 1e-12):
            distance = self.lengths[first] * (last_fraction - first_fraction)
            return [int(first)] if distance <= bound + 1e-9 else None
        remaining = self.lengths[first] * (1 - first_fraction)
        prefix = self.lengths[last] * last_fraction
        if remaining + prefix > bound + 1e-9:
            return None
        middle = self.node_path(int(self.arcs[first, 1]), int(self.arcs[last, 0]),
                                bound - remaining - prefix)
        return None if middle is None else [int(first), *middle, int(last)]

    def statistics(self) -> dict[str, int]:
        return {"routing_searches": self.searches, "routing_cache_hits": self.cache_hits,
                "routing_cached_trees": len(self.cache), "routing_nodes_settled": self.nodes_settled}


def _config(config) -> DecoderConfig:
    if config is None:
        return DecoderConfig()
    return DecoderConfig(**config) if isinstance(config, dict) else config


def _router(network, config: DecoderConfig) -> DirectedRouter:
    router = getattr(network, "_pathsnap_directed_router", None)
    if (router is None or router.cache_max != config.cache_max or
            router.bucket_m != config.distance_bucket_m):
        router = DirectedRouter(network, config.cache_max, config.distance_bucket_m)
        try:
            network._pathsnap_directed_router = router
        except (AttributeError, TypeError):
            pass
    return router


def _bound(displacement: float, elapsed: float, cfg: DecoderConfig) -> float:
    return float(min(cfg.max_route_m, max(cfg.min_route_m,
                     cfg.displacement_multiplier * displacement + cfg.route_slack_m,
                     cfg.max_speed * elapsed * 2.0)))


def _time_break(t: int, track_xy, timestamps, cfg: DecoderConfig) -> str | None:
    elapsed = float(timestamps[t] - timestamps[t - 1])
    if not math.isfinite(elapsed) or elapsed <= 0:
        return "invalid_timestamp_interval"
    if elapsed > cfg.gap_seconds:
        return "large_gps_gap"
    if not np.all(np.isfinite(track_xy[t - 1:t + 1])):
        return "invalid_gps_position"
    return None


def decode(network, candidates, probabilities, track_xy, timestamps,
           config=None) -> DecodeResult:
    """Decode one candidate state per observation, retaining explicit restarts.

    Missing candidate rows yield -1. All-unreachable trellis boundaries start a
    fresh segment; no connecting arc is fabricated. Large/invalid time intervals
    also break the trellis. Classifier probabilities are normalised per row.
    """
    started = perf_counter()
    cfg = _config(config)
    arcs = np.asarray(candidates.arc_ids, dtype=np.int64)
    fractions = np.clip(np.asarray(candidates.fractions, dtype=float), 0.0, 1.0)
    probabilities = np.asarray(probabilities, dtype=float)
    track_xy = np.asarray(track_xy, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if arcs.ndim != 2 or probabilities.shape != arcs.shape or fractions.shape != arcs.shape:
        raise ValueError("candidate arcs, fractions and probabilities must have matching (N,K) shapes")
    n, k = arcs.shape
    if track_xy.shape != (n, 2) or timestamps.shape != (n,):
        raise ValueError("GPS positions and timestamps must align with candidate rows")
    predictions = np.full(n, -1, dtype=np.int64)
    selected = np.full(n, -1, dtype=np.int32)
    if not n:
        return DecodeResult(predictions, selected, [], [], [], {"inference_seconds": 0.0}, [])
    valid = (arcs >= 0) & (arcs < len(network.arcs)) & np.isfinite(fractions)
    valid &= np.all(np.isfinite(track_xy), axis=1)[:, None]
    finite_probability = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
    potentials = np.where(valid, np.clip(finite_probability, cfg.probability_floor, 1.0), 0.0)
    totals = potentials.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        emission = np.where(valid, np.log(potentials / np.maximum(totals, cfg.probability_floor)), -np.inf)
    router = _router(network, cfg)
    before = router.statistics()
    pointers = np.full((n, k), -1, dtype=np.int32)
    warnings: list[dict[str, Any]] = []
    breaks: list[int] = []
    segment_start: int | None = None
    previous_scores: np.ndarray | None = None

    def finish(end: int):
        if segment_start is None or previous_scores is None or end < segment_start:
            return
        state = int(np.argmax(previous_scores))
        for row in range(end, segment_start - 1, -1):
            selected[row] = state
            predictions[row] = arcs[row, state]
            if row > segment_start:
                state = int(pointers[row, state])
                if state < 0:
                    raise RuntimeError("Broken Viterbi backpointer inside a supported segment")

    for t in range(n):
        if not np.any(valid[t]):
            finish(t - 1)
            segment_start, previous_scores = None, None
            warnings.append({"point_index": t, "reason": "missing_candidates_or_invalid_gps"})
            continue
        reason = _time_break(t, track_xy, timestamps, cfg) if t > 0 else None
        if segment_start is not None and reason is not None:
            finish(t - 1)
            segment_start, previous_scores = None, None
            warnings.append({"point_index": t, "reason": reason})
        if segment_start is None:
            segment_start = t
            breaks.append(t)
            previous_scores = emission[t].copy()
            continue
        elapsed = float(timestamps[t] - timestamps[t - 1])
        displacement = float(np.linalg.norm(track_xy[t] - track_xy[t - 1]))
        bound = _bound(displacement, elapsed, cfg)
        distances = router.distance_matrix(arcs[t - 1], fractions[t - 1], arcs[t], fractions[t], bound)
        with np.errstate(invalid="ignore"):
            transition = -np.abs(distances - displacement) / cfg.beta
            speeds = distances / elapsed
            transition -= cfg.speed_penalty_weight * np.maximum(speeds / cfg.max_speed - 1.0, 0.0) ** 2
        transition[~np.isfinite(distances)] = -np.inf
        if cfg.hard_speed_limit:
            transition[speeds > cfg.max_speed] = -np.inf
        if cfg.uturn_penalty:
            prev = network.arcs[np.where(valid[t - 1], arcs[t - 1], 0)]
            curr = network.arcs[np.where(valid[t], arcs[t], 0)]
            reverse = (prev[:, 0, None] == curr[None, :, 1]) & (prev[:, 1, None] == curr[None, :, 0])
            reverse &= arcs[t - 1, :, None] != arcs[t, None, :]
            transition[reverse] -= cfg.uturn_penalty
        scores = previous_scores[:, None] + transition
        best = np.argmax(scores, axis=0)
        new_scores = emission[t] + scores[best, np.arange(k)]
        if not np.any(np.isfinite(new_scores)):
            finish(t - 1)
            warnings.append({"point_index": t, "reason": "unreachable_directed_transition"})
            segment_start = t
            breaks.append(t)
            previous_scores = emission[t].copy()
        else:
            pointers[t] = best
            new_scores -= np.max(new_scores[np.isfinite(new_scores)])
            previous_scores = new_scores
    finish(n - 1)

    # Reconstruct intermediate graph arcs after backtracking the candidate trellis.
    route_segments: list[list[int]] = []
    current: list[int] = []
    break_set = set(breaks)
    for t in range(n):
        if predictions[t] < 0:
            if current:
                route_segments.append(current)
                current = []
            continue
        if t in break_set or not current or t == 0 or predictions[t - 1] < 0:
            if current:
                route_segments.append(current)
            current = [int(predictions[t])]
            continue
        displacement = float(np.linalg.norm(track_xy[t] - track_xy[t - 1]))
        bound = _bound(displacement, float(timestamps[t] - timestamps[t - 1]), cfg)
        path = router.arc_path(int(predictions[t - 1]), float(fractions[t - 1, selected[t - 1]]),
                               int(predictions[t]), float(fractions[t, selected[t]]), bound)
        if path is None:
            warnings.append({"point_index": t, "reason": "route_reconstruction_unreachable"})
            route_segments.append(current)
            current = [int(predictions[t])]
            breaks.append(t)
        else:
            current.extend(path[1:])
    if current:
        route_segments.append(current)
    after = router.statistics()
    stats: dict[str, Any] = {name: after[name] - before[name] for name in before if name != "routing_cached_trees"}
    stats.update({"routing_cached_trees": after["routing_cached_trees"],
                  "inference_seconds": perf_counter() - started,
                  "observations": n, "predicted_observations": int(np.sum(predictions >= 0)),
                  "coverage": float(np.mean(predictions >= 0)),
                  "segments": len(route_segments), "restart_count": max(0, len(breaks) - 1),
                  "config": asdict(cfg)})
    return DecodeResult(predictions, selected, [a for segment in route_segments for a in segment],
                        route_segments, warnings, stats, sorted(set(breaks)))


def evaluate_continuity(network, arc_ids, fractions, track_xy, timestamps,
                        config=None) -> dict[str, Any]:
    """Directed reachability rate over every time-eligible adjacent point pair.

    Exclude exogenous invalid/nonpositive/large time gaps. Missing predictions
    and unreachable transitions remain failures in the eligible denominator.
    This is a project diagnostic, not an official challenge scoring formula.
    """
    cfg = _config(config)
    arc_ids = np.asarray(arc_ids, dtype=int)
    fractions = np.asarray(fractions, dtype=float)
    track_xy = np.asarray(track_xy, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    n = len(arc_ids)
    if fractions.shape != (n,) or track_xy.shape != (n, 2) or timestamps.shape != (n,):
        raise ValueError("continuity arrays must be observation-aligned")
    router = _router(network, cfg)
    feasible = eligible = missing = unreachable = excluded = 0
    reasons: dict[str, int] = {}
    for t in range(1, n):
        reason = _time_break(t, track_xy, timestamps, cfg)
        if reason is not None:
            excluded += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        eligible += 1
        a, b = int(arc_ids[t - 1]), int(arc_ids[t])
        if (a < 0 or b < 0 or a >= len(network.arcs) or b >= len(network.arcs) or
                not np.isfinite(fractions[t - 1:t + 1]).all()):
            missing += 1
            continue
        displacement = float(np.linalg.norm(track_xy[t] - track_xy[t - 1]))
        distance = router.distance_matrix([a], [fractions[t - 1]], [b], [fractions[t]],
                                          _bound(displacement, float(timestamps[t] - timestamps[t - 1]), cfg))[0, 0]
        if np.isfinite(distance):
            feasible += 1
        else:
            unreachable += 1
    continuity = float(feasible / eligible) if eligible else None
    return {"route_continuity": continuity, "continuity": continuity,
            "feasible_transitions": feasible, "eligible_transitions": eligible,
            "missing_transitions": missing, "unreachable_transitions": unreachable,
            "excluded_time_gaps": excluded, "excluded_reasons": reasons,
            "restart_count": unreachable + excluded,
            "coverage": float(np.mean(arc_ids >= 0)) if n else 0.0}
