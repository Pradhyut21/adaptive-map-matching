"""Validate decoder noise tolerance exclusively on training out-of-fold scores."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from time import perf_counter

import numpy as np

from pathsnap.decoder import DecoderConfig, evaluate_continuity
from pathsnap.decoder_refinement import RefinementConfig, decode_with_previous
from pathsnap.geometry import CandidateSet, load_network
from pathsnap.metrics import lcs_f1, observation_metrics
from pathsnap.settings import ROOT, SEED, trip_dirs
from scripts.compare_route_alignment import CACHE, OLD_CACHE
from scripts.refine_accuracy import summarize, write_json

OUT = ROOT / 'results' / 'gps-noise-v1'


def choices():
    with (ROOT / 'artifacts/ordered_route_boost.config.json').open(encoding='utf-8') as f:
        original = json.load(f)
    decoder = DecoderConfig(**original['decoder'])
    refinement = RefinementConfig(**original['refinement'])
    return {
        'frozen_baseline': (decoder, refinement, original['previous_weight']),
        'dwell_15': (decoder, replace(refinement, same_arc_jitter_m=15.), .5),
        'dwell_30': (decoder, replace(refinement, same_arc_jitter_m=30.), .5),
        'smooth_3': (decoder, replace(refinement, smoothing_half_window=3, smoothing_fraction=1.,
                                       smooth_projections=True, same_arc_jitter_m=15.), .5),
        'smooth_5': (decoder, replace(refinement, smoothing_half_window=5, smoothing_fraction=1.,
                                       smooth_projections=True, same_arc_jitter_m=30.), .5),
        'single_booster_dwell': (decoder, replace(refinement, same_arc_jitter_m=30.), 0.),
    }


def choose_trips(count):
    paths = list((OLD_CACHE / 'train').glob('*.npz'))
    slow = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            xy, times = data['xy'], data['timestamps']
        rows = np.arange(len(xy))
        low, high = np.maximum(0, rows - 2), np.minimum(len(xy) - 1, rows + 2)
        speed = np.linalg.norm(xy[high] - xy[low], axis=1) / np.maximum(times[high] - times[low], 1e-6)
        slow.append((float(np.mean(speed < 3.)), path.stem))
    slow_ids = [tid for _, tid in sorted(slow, key=lambda v: (-v[0], v[1]))[:count // 2]]
    broad = sorted([path.stem for path in paths if path.stem not in slow_ids],
                   key=lambda tid: hashlib.sha256(f'noise-validation:{SEED}:{tid}'.encode()).digest())
    return sorted(slow_ids + broad[:count - len(slow_ids)]), slow_ids


def score(count):
    directories = {path.name: path for path in trip_dirs('train')}
    trip_ids, slow_ids = choose_trips(count)
    if not set(trip_ids) <= set(directories):
        raise ValueError('Validation inputs must be training trips')
    cases = choices()
    records = {name: [] for name in cases}
    old_recipe = json.loads((ROOT / 'results/context-boost-v1/selection.json').read_text(encoding='utf-8'))['selected_recipe']
    started = perf_counter()
    for trip_id in trip_ids:
        with np.load(OLD_CACHE / 'train' / f'{trip_id}.npz', allow_pickle=False) as old:
            candidates = CandidateSet(old['arc_ids'], old['distances'], old['fractions'], old['projected_xy'])
            xy, times, route = old['xy'], old['timestamps'], old['reference']
        with np.load(CACHE / 'train' / f'{trip_id}.npz', allow_pickle=False) as training:
            labels = training['labels']
        current = np.load(CACHE / 'oof' / f'{trip_id}.npy', allow_pickle=False)
        previous = np.load(OLD_CACHE / 'oof' / old_recipe / f'{trip_id}.npy', allow_pickle=False)
        graph = load_network(directories[trip_id])
        for name, (decoder, tweak, blend) in cases.items():
            prediction = decode_with_previous(graph, candidates, current, previous, xy, times,
                                              decoder, tweak, blend, False)
            row = observation_metrics(labels, prediction.arc_ids, candidates.arc_ids)
            row.update(lcs_f1(prediction.route_arc_ids, route))
            idx = np.maximum(prediction.candidate_indices, 0)
            fractions = candidates.fractions[np.arange(len(labels)), idx]
            row.update(evaluate_continuity(graph, prediction.arc_ids, fractions, xy, times, decoder))
            row.update(trip_id=trip_id, slow_training_stratum=trip_id in slow_ids,
                       decoder_seconds=prediction.stats['inference_seconds'])
            records[name].append(row)
        print(trip_id, {key: round(rows[-1]['top1_accuracy'], 4) for key, rows in records.items()}, flush=True)
    scores = {key: summarize(rows) for key, rows in records.items()}
    winner = max(scores, key=lambda key: (scores[key]['top1_accuracy'], scores[key]['sequence_lcs_f1']))
    decoder, tweak, blend = cases[winner]
    result = {
        'selected': winner, 'selection_metric': 'pooled all-point exact arc Top-1 on training OOF scores',
        'trip_ids': trip_ids, 'slow_training_ids': slow_ids,
        'sampling': 'Half slowest training trips by observed GPS motion, half deterministic hash sample',
        'scores': scores, 'elapsed_seconds': perf_counter() - started,
        'settings': {'decoder': asdict(decoder), 'refinement': asdict(tweak), 'previous_weight': blend, 'preserve_route': False},
        'scope': 'All classifier scores are from models excluding each trip in the original grouped fold. Evaluator files are excluded.',
    }
    write_json(OUT / 'training_validation.json', result)
    print('Training-selected', winner, scores[winner]['top1_accuracy'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trips', type=int, default=16)
    args = parser.parse_args()
    if not 2 <= args.trips <= 80:
        raise ValueError('Choose 2 to 80 training trips')
    score(args.trips)


if __name__ == '__main__':
    main()
