"""Reference-free candidate features with motion reliability near stops."""
import numpy as np

FEATURE_NAMES = ["distance_m", "log_distance", "heading_difference", "reliable_heading_difference", "heading_reliability", "projection_fraction", "log_segment_length", "log_speed", "log_time_delta", "distance_rank", "distance_above_best", "heading_cos", "heading_sin", "candidate_count"]

def motion_features(xy, timestamps, gap_seconds=60):
    n = len(xy)
    delta_t = np.r_[1., np.diff(timestamps)]
    speed = np.zeros(n)
    if n > 1:
        speed[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(delta_t[1:], 1e-6)
        speed[0] = speed[1]
    heading = np.zeros(n)
    reliability = np.zeros(n)
    # Symmetric four-sample motion uses future GPS observations: this is an
    # offline reconstruction system, not a claim of causal online tracking.
    breaks = np.r_[0, np.flatnonzero((delta_t[1:] <= 0) | (delta_t[1:] > gap_seconds))+1, n]
    for start, stop in zip(breaks[:-1], breaks[1:]):
        for t in range(start, stop):
            a, b = max(start, t-2), min(stop-1, t+2)
            d = xy[b]-xy[a]
            elapsed = timestamps[b]-timestamps[a]
            heading[t] = np.arctan2(d[1], d[0])
            reliability[t] = np.clip(np.linalg.norm(d)/15.0, 0, 1) if elapsed > 0 else 0.
    return heading, reliability, np.clip(speed, 0, 200), delta_t

def build_features(network, candidates, xy, timestamps):
    heading, reliable, speed, dt = motion_features(xy, timestamps)
    valid = candidates.valid
    safe = np.maximum(candidates.arc_ids, 0)
    distances = np.where(valid, candidates.distances, 0.)
    diff = np.abs(np.arctan2(np.sin(network.bearings[safe]-heading[:, None]), np.cos(network.bearings[safe]-heading[:, None])))
    n, k = valid.shape
    cols = [distances, np.log1p(distances), diff, diff*reliable[:, None], np.broadcast_to(reliable[:, None], (n,k)), candidates.fractions, np.log1p(network.lengths[safe]), np.broadcast_to(np.log1p(speed[:, None]),(n,k)), np.broadcast_to(np.log1p(np.maximum(dt[:,None],0)),(n,k)), np.broadcast_to(np.arange(k),(n,k)), distances-distances[:, :1], np.broadcast_to(np.cos(heading[:,None]),(n,k)), np.broadcast_to(np.sin(heading[:,None]),(n,k)), np.broadcast_to(valid.sum(axis=1)[:,None],(n,k))]
    X = np.stack(cols, axis=-1).astype(np.float32)
    X[~valid] = 0
    return X, {"heading":heading,"heading_reliability":reliable,"speed":speed,"time_delta":dt}
