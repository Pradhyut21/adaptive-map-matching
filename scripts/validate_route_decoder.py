"""Choose a route decoder on trip-grouped OOF predictions, then benchmark it."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
from threadpoolctl import threadpool_limits

from pathsnap.decoder import DecoderConfig, evaluate_continuity
from pathsnap.decoder_refinement import RefinementConfig, decode_with_previous
from pathsnap.geometry import CandidateSet, load_network
from pathsnap.metrics import observation_metrics, lcs_f1
from pathsnap.predict import load_artifact, predict_trip
from pathsnap.route_alignment import align_ordered_route
from pathsnap.settings import ROOT, SEED, trip_dirs
from scripts.compare_route_alignment import CACHE, LEGACY, OLD_CACHE, OUT
from scripts.refine_accuracy import summarize, write_json


ARTIFACT = ROOT / 'artifacts' / 'ordered_route_boost.joblib'
BASELINE = ROOT / 'artifacts' / 'context_boost.joblib'


def score_candidates(labels, legacy_labels, candidates, decoded, ref, selected_distances):
    ordered = observation_metrics(labels, decoded.arc_ids, candidates.arc_ids)
    legacy = observation_metrics(legacy_labels, decoded.arc_ids, candidates.arc_ids)
    return {**ordered, **lcs_f1(decoded.route_arc_ids, ref),
            'legacy_correct': legacy['correct'], 'legacy_top1_accuracy': legacy['top1_accuracy'],
            'legacy_candidate_miss_count': legacy['candidate_miss_count'],
            'label_changes': int(np.sum(labels != legacy_labels)),
            'label_distance_over_60_m': int(np.sum(selected_distances > 60.0)),
            'restarts': max(0, len(decoded.route_segments) - 1),
            'decoder_seconds': decoded.stats['inference_seconds']}


def summarize_dual(rows):
    scores = summarize(rows)
    scores['legacy_correct'] = sum(row['legacy_correct'] for row in rows)
    scores['legacy_top1_accuracy'] = scores['legacy_correct'] / scores['observations']
    scores['label_changes'] = sum(row['label_changes'] for row in rows)
    scores['label_distance_over_60_m'] = sum(row['label_distance_over_60_m'] for row in rows)
    scores['decoder_seconds'] = sum(row['decoder_seconds'] for row in rows)
    return scores


def select(n_trips, reuse_decoding=False):
    baseline = load_artifact(BASELINE)
    old_decoder = DecoderConfig(**baseline['decoder_config'])
    old_refinement = RefinementConfig(**baseline['decoder_refinement'])
    new_decoder = replace(old_decoder, beta=10., uturn_penalty=4.)
    geometric = RefinementConfig(same_arc_jitter_m=5., emission_temperature=2.,
                                 distance_prior_sigma_m=12., max_states=8)
    choices = {
        'previous_model': (old_decoder, old_refinement, 1., False),
        'new_model_previous_decoder': (old_decoder, old_refinement, 0., False),
        'distance_and_turn': (new_decoder, geometric, 0., False),
        'distance_and_turn_realign': (new_decoder, replace(geometric, realign_predicted_route=True), 0., False),
        'distance_turn_speed': (replace(new_decoder, hard_speed_limit=True), geometric, 0., False),
        'blend_equal': (old_decoder, old_refinement, .5, False),
        'blend_route_focused': (old_decoder, old_refinement, .75, False),
        'blend_with_geometry': (new_decoder, geometric, .5, False),
        'preserve_path': (old_decoder, replace(old_refinement, route_assignment_classifier_weight=1.0), 0., True),
    }
    if not (CACHE / 'oof').exists():
        raise FileNotFoundError('Run scripts.compare_route_alignment first')
    ids = {p.name: p for p in trip_dirs('train')}
    sample = sorted(ids, key=lambda tid: hashlib.sha256(f'ordered-decoder:{SEED}:{tid}'.encode()).digest())[:n_trips]
    old_record = OUT / 'decoder_selection.json'
    cached = {}
    if reuse_decoding and old_record.is_file():
        data = json.loads(old_record.read_text(encoding='utf-8'))
        if data['training_trips'] == sample:
            cached = data['choices']
    records = {name: [] for name in choices if name not in cached}
    for trip_id in sample:
        with np.load(CACHE / 'train' / f'{trip_id}.npz', allow_pickle=False) as new:
            labels, legacy_labels, distances = new['labels'], new['legacy_labels'], new['label_distances']
        with np.load(OLD_CACHE / 'train' / f'{trip_id}.npz', allow_pickle=False) as old:
            candidates = CandidateSet(old['arc_ids'], old['distances'], old['fractions'], old['projected_xy'])
            xy, times, ref = old['xy'], old['timestamps'], old['reference']
        old_scores = np.load(OLD_CACHE / 'oof' / baseline['recipe'] / f'{trip_id}.npy', allow_pickle=False)
        new_scores = np.load(CACHE / 'oof' / f'{trip_id}.npy', allow_pickle=False)
        net = load_network(ids[trip_id])
        for name in records:
            config, refinement, weight, preserve = choices[name]
            decoded = decode_with_previous(net, candidates, new_scores, old_scores, xy, times,
                                           config, refinement, weight, preserve)
            row = score_candidates(labels, legacy_labels, candidates, decoded, ref, distances)
            records[name].append({'trip_id': trip_id, **row})
        print(trip_id, {name: round(rows[-1]['top1_accuracy'], 4) for name, rows in records.items()}, flush=True)
    summaries = {**cached, **{name: summarize_dual(rows) for name, rows in records.items()}}
    original = summaries['previous_model']
    accepted = [name for name, value in summaries.items() if name != 'previous_model'
                and value['top1_accuracy'] > original['top1_accuracy']
                and value['sequence_lcs_f1'] >= original['sequence_lcs_f1']]
    winner = max(accepted, key=lambda name: summaries[name]['top1_accuracy'], default='previous_model')
    config, refinement, weight, preserve = choices[winner]
    write_json(OUT / 'decoder_selection.json',
               {'selected': winner, 'training_trips': sample, 'choices': summaries,
                'rule': 'Improve all-point ordered-label Top-1 without reducing ordered-route LCS F1 on fixed training OOF trips',
                'decoder': asdict(config), 'refinement': asdict(refinement),
                'previous_weight': weight, 'preserve_route': preserve})
    if winner != 'previous_model':
        write_json(ARTIFACT.with_suffix('.config.json'), {'decoder': asdict(config), 'refinement': asdict(refinement),
                                                        'previous_weight': weight, 'preserve_route': preserve})
    print(f'{winner}: original {original["top1_accuracy"]:.2%}, selected {summaries[winner]["top1_accuracy"]:.2%}', flush=True)


def benchmark(threads):
    selection = json.loads((OUT / 'decoder_selection.json').read_text(encoding='utf-8'))
    models = {'previous_model': load_artifact(BASELINE)}
    if selection['selected'] != 'previous_model':
        models['ordered_route_boost'] = load_artifact(ARTIFACT)
    results = {name: [] for name in models}
    for trip in trip_dirs('student_test'):
        with threadpool_limits(limits=threads):
            predictions = {name: predict_trip(trip, model) for name, model in models.items()}
        p = predictions['previous_model']
        with np.load(LEGACY / 'student_test' / f'{trip.name}.npz', allow_pickle=False) as frozen:
            ref, legacy_labels = frozen['reference'], frozen['labels']
            if not np.array_equal(frozen['timestamps'], p.track[:, 2]):
                raise ValueError('Frozen labels do not match trip timeline')
        labels, _, distances = align_ordered_route(p.network, p.track_xy, p.track[:, 2], ref)
        for name, prediction in predictions.items():
            decoded = prediction.decoded
            row = score_candidates(labels, legacy_labels, prediction.candidates, decoded, ref, distances)
            fractions = prediction.candidates.fractions[np.arange(len(labels)), np.maximum(decoded.candidate_indices, 0)]
            row.update(evaluate_continuity(prediction.network, decoded.arc_ids, fractions,
                                          prediction.track_xy, prediction.track[:, 2], decoded.stats['config']))
            row.update(prediction.timing)
            results[name].append({'trip_id': trip.name, **row})
            destination = OUT / 'predictions' / name
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / f'{trip.name}.npz', predictions=decoded.arc_ids,
                                 candidate_indices=decoded.candidate_indices, route_arc_ids=decoded.route_arc_ids)
        print(trip.name, {name: (round(rows[-1]['top1_accuracy'], 4), round(rows[-1]['legacy_top1_accuracy'], 4))
                         for name, rows in results.items()}, flush=True)
    if any(len(rows) != 10 for rows in results.values()):
        raise ValueError('Expected all ten student trips')
    active = 'ordered_route_boost' if 'ordered_route_boost' in models else 'previous_model'
    summaries = {name: summarize_dual(rows) for name, rows in results.items()}
    write_json(OUT / 'student_test.json',
               {'partition': 'student_test', 'selected': active, 'comparisons': summaries,
                'hmm': summaries[active], 'decoder_selection': selection['selected'],
                'denominator': 'Every GPS observation under each independently derived label set'})
    print({name: (scores['top1_accuracy'], scores['legacy_top1_accuracy'], scores['sequence_lcs_f1'])
           for name, scores in summaries.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('select', 'benchmark'))
    parser.add_argument('--training-trips', type=int, default=16)
    parser.add_argument('--reuse-decoding', action='store_true',
                        help='Reuse previously scored choices for the same frozen OOF scores')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    if args.task == 'select':
        select(args.training_trips, args.reuse_decoding)
    else:
        benchmark(args.threads)


if __name__ == '__main__':
    main()
