"""Reference-only derived labels. Never imported by prediction feature code."""
from pathlib import Path
import numpy as np
import shapely
from shapely.strtree import STRtree
from .geometry import project_segments
from .features import motion_features

def load_reference(trip_dir, network):
    p = Path(trip_dir)
    route = np.loadtxt(p / f"{p.name}.route", dtype=np.int32, ndmin=1)
    if len(route) == 0 or route.min() < 0 or route.max() >= len(network.arcs):
        raise ValueError("Empty reference or out-of-bounds reference arc")
    return route

def align_reference(network, track_xy, timestamps, route, width=24):
    """Sparse monotone route-position DP; repeated arc occurrences stay distinct.

    Reference positions (not arc IDs) are states. This is approximate labeling,
    not additional physical ground truth. Fallbacks/restarts are returned.
    """
    route = np.asarray(route, dtype=np.int32)
    tree = STRtree(network.lines[route])
    starts, ends = network.starts[route], network.ends[route]
    cum = np.r_[0., np.cumsum(network.lengths[route])]
    heading, reliability, _, dt = motion_features(track_xy, timestamps)
    n = len(track_xy)
    states, costs, offsets = [], [], []
    distance_best = []
    ambiguous = 0
    for t, point in enumerate(track_xy):
        query = shapely.Point(point)
        ids = tree.query(query, predicate="dwithin", distance=150.).astype(np.int32)
        if len(ids) == 0:
            ids = np.asarray([tree.nearest(query)], np.int32)
        d, f, _ = project_segments(point, starts[ids], ends[ids])
        delta = np.abs(np.arctan2(np.sin(network.bearings[route[ids]]-heading[t]), np.cos(network.bearings[route[ids]]-heading[t])))
        emission = np.minimum(d/20., 100.)**2 + 1.5*delta*reliability[t]
        best = np.argsort(emission, kind="stable")[:width]
        ids, d, f, emission = ids[best], d[best], f[best], emission[best]
        states.append(ids)
        costs.append(emission)
        offsets.append(cum[ids]+f*network.lengths[route[ids]])
        distance_best.append(float(d[0]))
        ambiguous += int(len(emission)>1 and emission[1]-emission[0]<0.1)
    labels = np.full(n, -1, np.int32)
    positions = np.full(n, -1, np.int32)
    restarts, fallbacks = [], 0
    backs = []
    score = None
    segment_start = 0
    def finish(stop, last):
        for t in range(stop-1, segment_start-1, -1):
            positions[t] = states[t][last]
            labels[t] = route[positions[t]]
            if t > segment_start:
                last = int(backs[t][last])
    for t in range(n):
        restart = t==0 or dt[t]<=0 or dt[t]>60.
        if not restart:
            forward = states[t][None,:] >= states[t-1][:,None]
            travel = offsets[t][None,:]-offsets[t-1][:,None]
            forward &= travel >= -2.
            gps = np.linalg.norm(track_xy[t]-track_xy[t-1])
            transition = np.abs(travel-gps)/50.
            transition += np.maximum(travel/max(dt[t],1e-6)-55.,0.)/10.
            candidates = score[:,None]+transition
            candidates[~forward] = np.inf
            previous = np.argmin(candidates,axis=0)
            nextscore = candidates[previous,np.arange(len(states[t]))]+costs[t]
            if not np.isfinite(nextscore).any():
                restart = True
                fallbacks += 1
        if restart:
            if t:
                finish(t,int(np.argmin(score)))
                restarts.append(t)
            segment_start = t
            previous = np.full(len(states[t]),-1,np.int32)
            # First position is slightly preferred only when distance scores tie.
            nextscore = costs[t]+states[t]*1e-5
        score = nextscore-np.min(nextscore)
        backs.append(previous)
    if n:
        finish(n,int(np.argmin(score)))
    selected_distance = np.linalg.norm(track_xy - np.array([network.starts[a]+np.clip(np.dot(track_xy[t]-network.starts[a],network.vectors[a])/max(network.lengths[a]**2,1e-12),0,1)*network.vectors[a] for t,a in enumerate(labels)]),axis=1) if n else np.array([])
    return labels, positions, {"label_type":"derived_monotone_reference_alignment", "observations":n,"ambiguous_observations":ambiguous,"ambiguous_fraction":ambiguous/max(n,1),"fallback_restarts":fallbacks,"restarts":restarts,"nearest_fallback_fraction":fallbacks/max(n,1),"median_label_distance_m":float(np.median(selected_distance)) if n else None,"max_label_distance_m":float(selected_distance.max()) if n else None}
