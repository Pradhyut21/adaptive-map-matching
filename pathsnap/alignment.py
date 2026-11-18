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
    not additional physical ground truth. When the monotone trellis has no
    feasible transition, the boundary observation is explicitly fixed to its
    nearest reference segment and a new alignment segment begins. Ordinary
    large-time-gap restarts are logged separately from this nearest fallback.
    Poor fits are flagged rather than silently changed after alignment.
    """
    route = np.asarray(route, dtype=np.int32)
    if width < 1 or route.ndim != 1 or not len(route):
        raise ValueError("Alignment needs a nonempty one-dimensional route and positive width")
    if route.min() < 0 or route.max() >= len(network.arcs):
        raise ValueError("Reference arc outside graph")
    tree = STRtree(network.lines[route])
    starts, ends = network.starts[route], network.ends[route]
    cum = np.r_[0., np.cumsum(network.lengths[route])]
    heading, reliability, _, dt = motion_features(track_xy, timestamps)
    n = len(track_xy)
    states, costs, offsets = [], [], []
    nearest_distances, nearest_local = [], []
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
        # Keep an exact geometrically nearest reference occurrence even when
        # heading costs would prune it. It is used only if alignment fails.
        nearest = int(np.lexsort((ids, d))[0])
        if nearest not in best:
            best[-1] = nearest
        ids, d, f, emission = ids[best], d[best], f[best], emission[best]
        states.append(ids)
        costs.append(emission)
        offsets.append(cum[ids]+f*network.lengths[route[ids]])
        nearest_local.append(int(np.lexsort((ids, d))[0]))
        nearest_distances.append(float(d[nearest_local[-1]]))
        ordered_emission = np.sort(emission)
        ambiguous += int(len(emission)>1 and ordered_emission[1]-ordered_emission[0]<0.1)
    labels = np.full(n, -1, np.int32)
    positions = np.full(n, -1, np.int32)
    restarts, fallback_events = [], []
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
        nearest_fallback = False
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
                nearest_fallback = True
        if restart:
            if t:
                finish(t,int(np.argmin(score)))
                restarts.append(t)
            segment_start = t
            previous = np.full(len(states[t]),-1,np.int32)
            # First position is slightly preferred only when distance scores tie.
            nextscore = costs[t]+states[t]*1e-5
            if nearest_fallback:
                chosen = nearest_local[t]
                forced = np.full(len(states[t]), np.inf)
                forced[chosen] = nextscore[chosen]
                nextscore = forced
                fallback_events.append({"point_index":t,
                    "reference_position":int(states[t][chosen]),
                    "arc_id":int(route[states[t][chosen]]),
                    "distance_m":nearest_distances[t],
                    "reason":"no_feasible_monotone_transition"})
        score = nextscore-np.min(nextscore)
        backs.append(previous)
    if n:
        finish(n,int(np.argmin(score)))
    selected_distance = np.linalg.norm(track_xy - np.array([network.starts[a]+np.clip(np.dot(track_xy[t]-network.starts[a],network.vectors[a])/max(network.lengths[a]**2,1e-12),0,1)*network.vectors[a] for t,a in enumerate(labels)]),axis=1) if n else np.array([])
    poor_fit_indices = np.flatnonzero(selected_distance > np.asarray(nearest_distances)+50.).tolist()
    fallbacks = len(fallback_events)
    return labels, positions, {"label_type":"derived_monotone_reference_alignment", "observations":n,"ambiguous_observations":ambiguous,"ambiguous_fraction":ambiguous/max(n,1),"fallback_restarts":fallbacks,"restarts":restarts,"nearest_fallback_count":fallbacks,"nearest_fallback_fraction":fallbacks/max(n,1),"nearest_fallbacks":fallback_events,"poor_fit_threshold_above_nearest_m":50.,"poor_fit_count":len(poor_fit_indices),"poor_fit_fraction":len(poor_fit_indices)/max(n,1),"poor_fit_indices":poor_fit_indices,"median_nearest_reference_distance_m":float(np.median(nearest_distances)) if n else None,"median_label_distance_m":float(np.median(selected_distance)) if n else None,"max_label_distance_m":float(selected_distance.max()) if n else None}
