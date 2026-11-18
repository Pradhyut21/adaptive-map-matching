# Directed HMM-style route decoder

`pathsnap.decoder` is an independent implementation. It imports no map-matching
engine and reads no reference routes or labels. All distances are in metres and
times in seconds. The supplied road graph remains directed.

The current artifact uses short-window GPS smoothing before decoder projection.
The half-window is three observations, and a same-arc backward projection shift
up to 15 m can represent a dwell caused by measurement jitter. Larger shifts
still require a feasible directed path. These settings were selected on
out-of-fold predictions from sixteen training trips, including slow motion.
The original strict continuity diagnostic continues to use raw GPS projections.
It can therefore reject transitions accepted by the decoder's explicit noise
model. Both this diagnostic and ordered route LCS F1 are retained.
See [training selection](../results/gps-noise-v1/training_validation.json) and
[student-trip evaluation](../results/gps-noise-v1/student_test.json).

## State, scores and recurrence

Each candidate state contains an arc ID and a projected fraction along that arc.
The two boosting components supply positive scores for each valid candidate.
Their equal geometric blend is temperature-scaled, combined with a Gaussian
GPS-distance prior, and reduced to eight candidate states per observation.
Scores are clipped to `[1e-8, 1]`, normalised within each observation, and logged.
These are discriminative emission potentials; normalisation does not establish
calibrated probability that a route is physically correct.

For elapsed time `dt > 0`, observed GPS displacement `g`, and directed network
distance `d`, the implemented transition score is

```text
-abs(d - g) / beta - speed_penalty_weight * max(d / (dt * max_speed) - 1, 0)^2
    - uturn_penalty * is_direct_reverse_arc_transition
```

The log-space recurrence takes the largest predecessor score plus the transition,
then adds the current log emission. Backpointers retain the selected predecessors.
Subtracting the row's finite maximum maintains numerical stability without
changing the selected sequence.

For different arcs the network distance is the remaining length on the first arc,
a directed shortest path from its end node to the next arc's start node, and the
travelled fraction on the next arc. Forward movement on the same arc uses its
fraction difference directly. Backward projection shifts within the current
15 m tolerance represent GPS jitter. Larger shifts require a directed loop. Zero-length arcs
have zero same-arc displacement.

## Routing cost and cache

Adjacency is stored as compact CSR arrays. One bounded sparse Dijkstra tree serves
all destination candidates for a source node. Cache entries are sparse distance
and predecessor dictionaries, not dense all-node distance arrays. A cache is
attached to a concrete graph object and retains at most 512 source/bound entries;
trip-local integer IDs can therefore never retrieve another graph's routes.

The search distance bound is

```text
min(max_route_m,
    max(min_route_m, 3 * GPS_displacement + route_slack_m, 2 * max_speed * dt))
```

Routing bounds are 200 to 5,000 metres, speed scale `55 m/s`, and gap threshold
`60 s`. The current setting uses `beta=10 m` and a U-turn penalty of 4. The original
decoder default was `beta=30 m`. Sparse searches are grouped into
250-metre cache buckets, but each returned transition still respects its exact
requested distance bound.

## Failures and returned route

An empty candidate row or nonfinite GPS point gets an observation-aligned `-1`
prediction. Missing observations, elapsed time over the configured gap threshold,
invalid/nonpositive elapsed time, and a boundary with no finite directed transition
all break the trellis. Each supported portion is backtracked independently.
Warnings state the observation index and reason. No route connects these gaps.

The result contains one predicted arc and candidate index per GPS observation,
the reconstructed intermediate road arcs, separate route segments, break indices,
warnings, coverage, elapsed time, and routing/cache counts. `route_arc_ids` is a
flattened convenience representation; use `route_segments` when drawing routes
so unsupported gaps cannot become connecting lines.

## Continuity denominator

`evaluate_continuity` measures the fraction of time-eligible adjacent observation
pairs whose predicted arc positions have a directed network path within the same
routing bound. Nonpositive/nonfinite time intervals, large configured time gaps,
and invalid GPS positions are excluded and their reasons counted. Missing
predictions and unreachable transitions remain failures in the eligible
denominator. With no eligible pairs the rate is `None`, not a fabricated perfect
score. This is a project diagnostic, not an official aggregate evaluator formula.

The measure is directed reachability, not a claim that every transition has a
plausible speed or follows physical ground truth. Report Top-1 accuracy, sequence
performance, coverage, speed diagnostics and decoding restarts alongside it.

## Verification

Run `python -m unittest discover -s tests -p test_decoder.py -v` from the project
directory. Tests cover direction, intermediate arcs, same-arc reverse movement,
real loops, zero-length roads, exact routing bounds, bounded cache reuse,
disconnected states, missing candidates, nonfinite GPS, invalid timestamps,
single observations, safe probability clipping, graph isolation, and continuity
denominators. Synthetic unit cases are behavioral checks, not dataset accuracy
experiments.
