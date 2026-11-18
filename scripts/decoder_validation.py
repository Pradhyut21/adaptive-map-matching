"""Compare decoder variants on a fixed holdout drawn only from training trips."""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from pathsnap.decoder import DecoderConfig, decode
from pathsnap.decoder_refinement import RefinementConfig, decode_refined
from pathsnap.geometry import CandidateSet, load_network
from pathsnap.metrics import lcs_f1, observation_metrics
from pathsnap.scoring import positive_scores
from pathsnap.settings import CACHE_ROOT, ROOT, SEED, trip_dirs


def choose_holdout(paths, count):
    """Select trips before examining their metrics, lengths, or coordinates."""
    if not 0 < count < len(paths):
        raise ValueError("Holdout must leave at least one fitting trip")
    ordered = sorted(paths, key=lambda path: hashlib.sha256(f"decoder-{SEED}:{path.stem}".encode()).digest())
    return ordered[count:], ordered[:count]


def fit_classifier(files, threads):
    rng = np.random.default_rng(SEED)
    feature_rows, label_rows = [], []
    for path in files:
        with np.load(path, allow_pickle=False) as cache:
            labels = cache["y"]
            valid = cache["arc_ids"] >= 0
            keep = valid & ((labels == 1) | (np.arange(valid.shape[1]) < 2)
                            | (rng.random(valid.shape) < 0.12))
            feature_rows.append(cache["X"][keep])
            label_rows.append(labels[keep])
    features, labels = np.concatenate(feature_rows), np.concatenate(label_rows)
    del feature_rows, label_rows
    model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=31, learning_rate=0.1,
                                           l2_regularization=1.0, min_samples_leaf=30,
                                           early_stopping=False, random_state=SEED)
    with threadpool_limits(limits=threads):
        model.fit(features, labels)
    return model, len(labels)


def configurations():
    base = DecoderConfig()
    standard = RefinementConfig()
    dwell = replace(standard, same_arc_jitter_m=5.0)
    smooth = replace(dwell, smoothing_half_window=2, smooth_projections=True)
    return {
        "current": (base, standard),
        "route_beta_15": (replace(base, beta=15), standard),
        "route_beta_7": (replace(base, beta=7), standard),
        "dwell_5m": (base, dwell),
        "dwell_5m_beta_15": (replace(base, beta=15), dwell),
        "dwell_5m_beta_7": (replace(base, beta=7), dwell),
        "local_smoothing_beta_15": (replace(base, beta=15), smooth),
        "local_smoothing_beta_7": (replace(base, beta=7), smooth),
        "strong_transitions_dwell": (replace(base, beta=11), replace(dwell, emission_temperature=2.5)),
    }


def summarize(per_trip):
    n = sum(row["observations"] for row in per_trip)
    return {
        "observations": n,
        "top1_accuracy": sum(row["correct"] for row in per_trip) / n,
        "mean_trip_top1_accuracy": float(np.mean([row["top1_accuracy"] for row in per_trip])),
        "candidate_miss_rate": sum(row["candidate_miss_count"] for row in per_trip) / n,
        "sequence_lcs_f1": 2 * sum(row["sequence_lcs"] for row in per_trip) / max(1, sum(
            row["reference_sequence_length"] + row["predicted_sequence_length"] for row in per_trip)),
        "route_restarts": sum(row["route_restarts"] for row in per_trip),
        "decoder_seconds": sum(row["decoder_seconds"] for row in per_trip),
        "trips": per_trip,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-trips", type=int, default=12)
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--cache", type=Path, default=CACHE_ROOT / "trace-lite-v2" / "train")
    parser.add_argument("--output", type=Path, default=ROOT.parent.parent / "work" / "decoder_validation.json")
    parser.add_argument("--route-realignment", action="store_true",
                        help="Compare the best first-stage settings with predicted-route realignment")
    args = parser.parse_args()
    training = {trip.name: trip for trip in trip_dirs("train")}
    files = sorted(args.cache.glob("*.npz"))
    if (len(files) != len(training) or {f.stem for f in files} != training.keys()
            or args.cache.resolve() != (CACHE_ROOT / "trace-lite-v2" / "train").resolve()):
        raise ValueError("Validation requires the v2 training cache and only training trips")
    fit, holdout = choose_holdout(files, args.holdout_trips)
    variants = configurations()
    if args.route_realignment:
        shortlist = ("strong_transitions_dwell", "local_smoothing_beta_15", "local_smoothing_beta_7")
        variants = {name: variants[name] for name in ("current", *shortlist)}
        for name in shortlist:
            config, tweak = variants[name]
            variants[name + "_realign"] = (config, replace(tweak, realign_predicted_route=True))
    start = perf_counter()
    classifier, trained_rows = fit_classifier(fit, args.threads)
    print(f"Fit HGB on {len(fit)} training trips ({trained_rows:,} rows) in {perf_counter()-start:.1f}s", flush=True)
    print("Fixed held-out training trips:", ", ".join(path.stem for path in holdout), flush=True)
    stats = defaultdict(list)
    for path in holdout:
        network = load_network(training[path.stem])
        with np.load(path, allow_pickle=False) as cache:
            candidate = CandidateSet(cache["arc_ids"], cache["distances"], cache["fractions"], cache["projected_xy"])
            xy, timestamps = cache["track_xy"], cache["timestamps"]
            probabilities = np.zeros(candidate.arc_ids.shape, dtype=np.float32)
            with threadpool_limits(limits=args.threads):
                probabilities[candidate.valid] = positive_scores(classifier, cache["X"][candidate.valid])
            scored = {}
            for name, (config, tweak) in variants.items():
                began = perf_counter()
                if name == "current":
                    result = decode(network, candidate, probabilities, xy, timestamps, config=config)
                else:
                    result = decode_refined(network, candidate, probabilities, xy, timestamps,
                                            config=config, refinement=tweak)
                scored[name] = result, perf_counter() - began

            # The predictions for this trip are fixed before reading its labels.
            labels, reference = cache["labels"], cache["reference"]
            for name, (result, seconds) in scored.items():
                row = observation_metrics(labels, result.arc_ids, candidate.arc_ids)
                row.update(lcs_f1(result.route_arc_ids, reference))
                row.update(trip_id=path.stem, route_restarts=result.stats["restart_count"],
                           decoder_seconds=seconds)
                stats[name].append(row)
            print(path.stem, len(xy), " ".join(f"{key}={rows[-1]['top1_accuracy']:.3f}"
                                              for key, rows in stats.items()), flush=True)

        # Persist each completed trip to make the benchmark resumable by analysis.
        result = {
            "partition": "train_only_trip_holdout",
            "selection_objective": "pooled exact directed arc top-1, ties by LCS F1",
            "fit_trips": [path.stem for path in fit],
            "holdout_trips": [path.stem for path in holdout],
            "completed_holdout_trips": len(stats["current"]),
            "training_rows": trained_rows,
            "variants": {name: {**summarize(rows), "decoder": asdict(variants[name][0]),
                                 "refinement": asdict(variants[name][1])} for name, rows in stats.items()},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    ranking = sorted(result["variants"].items(), key=lambda item: (
        item[1]["top1_accuracy"], item[1]["sequence_lcs_f1"]), reverse=True)
    for name, metrics in ranking:
        print(f"{name}: Top-1 {metrics['top1_accuracy']:.2%}, LCS {metrics['sequence_lcs_f1']:.3f}, "
              f"{metrics['route_restarts']} restarts, {metrics['decoder_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
