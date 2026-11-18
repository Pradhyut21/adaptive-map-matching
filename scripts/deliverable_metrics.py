"""Report matched Bayesian tuning metrics and the deployed decoder's tradeoffs."""

import argparse
import csv
import json
from time import perf_counter

import numpy as np

from pathsnap.decoder import evaluate_continuity
from pathsnap.decoder_refinement import combine_scores
from pathsnap.geometry import load_network, load_observations
from pathsnap.metrics import candidate_metrics, collapse_arcs, lcs_f1, observation_metrics, select_top1
from pathsnap.predict import load_artifact
from pathsnap.refined_inputs import FEATURE_NAMES, ROUTE_VERSION, VERSION, context_inputs
from pathsnap.route_alignment import align_ordered_route
from pathsnap.settings import ROOT, trip_dirs
from scripts.compare_route_alignment import probabilities, training_folds
from scripts.refine_accuracy import FROZEN_LABELS, RECIPES, WORK, classify, fit, summarize, write_json

ROOT_RESULTS = ROOT / 'results'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def candidate_rows(labels, arcs, scores):
    valid = arcs >= 0
    return (arcs == labels[:, None])[valid], scores[valid]


def cv_scores(recipe, folds):
    metrics = []
    for fold in folds:
        actual, predicted, correct, total = [], [], 0, 0
        for trip_id in fold['validation_trips']:
            with np.load(WORK / 'train' / f'{trip_id}.npz', allow_pickle=False) as cache:
                labels, arcs = cache['labels'], cache['arc_ids']
            scores = np.load(WORK / 'oof' / recipe / f'{trip_id}.npy', allow_pickle=False)
            prediction, _ = select_top1(arcs, scores)
            target, score = candidate_rows(labels, arcs, scores)
            actual.append(target)
            predicted.append(score)
            correct += int((prediction == labels).sum())
            total += len(labels)
        metrics.append({'top1_accuracy': correct / total,
                        **candidate_metrics(np.concatenate(actual), np.concatenate(predicted))})
    return {
        'mean_top1': float(np.mean([m['top1_accuracy'] for m in metrics])),
        'mean_macro_f1': float(np.mean([m['macro_f1'] for m in metrics])),
        'folds': metrics,
    }


def boost_tuning_metrics(threads):
    path = ROOT_RESULTS / VERSION
    tuning = read_json(path / 'boost_tuning.json')
    fold_ids = training_folds()
    train_ids = sorted({tid for fold in fold_ids for tid in fold['validation_trips']})
    if len(train_ids) != 80:
        raise ValueError('Expected all 80 training trips')
    before, after = tuning['original_recipe'], f"tpe_{tuning['best_trial']:02d}"
    tuned = load_artifact(ROOT / 'artifacts/context_boost.joblib')
    if tuned['recipe'] != after:
        raise ValueError('Saved booster differs from winning trial')
    cv = [cv_scores(recipe, fold_ids) for recipe in (before, after)]
    started = perf_counter()
    untuned = fit(train_ids, RECIPES[before], threads)
    untuned_fit_seconds = perf_counter() - started
    models = (untuned, tuned['model'])
    stats = [dict(train_count=0, train_correct=0, inference_seconds=0., actual=[], scores=[], rows=[]) for _ in models]
    for trip_id in train_ids:
        with np.load(WORK / 'train' / f'{trip_id}.npz', allow_pickle=False) as cache:
            features, arcs, labels = cache['features'], cache['arc_ids'], cache['labels']
        for model, result in zip(models, stats):
            pred, _ = classify(model, features, arcs, threads)
            result['train_count'] += len(labels)
            result['train_correct'] += int((pred == labels).sum())
    for trip in trip_dirs('student_test'):
        net, track = load_network(trip), load_observations(trip)
        candidates, features, _ = context_inputs(net, net.project(track[:, :2]), track[:, 2])
        with np.load(FROZEN_LABELS / 'student_test' / f'{trip.name}.npz', allow_pickle=False) as f:
            labels = f['labels']
            if not np.array_equal(track[:, 2], f['timestamps']):
                raise ValueError('Frozen labels do not match the student timeline')
        for model, result in zip(models, stats):
            start = perf_counter()
            pred, scores = classify(model, features, candidates.arc_ids, threads)
            result['inference_seconds'] += perf_counter() - start
            actual, prob = candidate_rows(labels, candidates.arc_ids, scores)
            result['actual'].append(actual)
            result['scores'].append(prob)
            result['rows'].append({'trip_id': trip.name, **observation_metrics(labels, pred, candidates.arc_ids)})
    rows = []
    for name, model, fold_metrics, result in zip(('untuned', 'tuned'), models, cv, stats):
        student = summarize(result['rows'])
        binary = candidate_metrics(np.concatenate(result['actual']), np.concatenate(result['scores']))
        rows.append({
            'variant': name,
            'train_top1': result['train_correct'] / result['train_count'],
            'training_observations': result['train_count'],
            'cv_top1': fold_metrics['mean_top1'],
            'cv_macro_f1': fold_metrics['mean_macro_f1'],
            'student_top1': student['top1_accuracy'],
            'student_macro_f1': binary['macro_f1'],
            'student_log_loss': binary['log_loss'],
            'student_inference_seconds': result['inference_seconds'],
            'student_observations': student['observations'],
        })
    expected = read_json(path / 'student_test.json')['classifier']['top1_accuracy']
    if rows[1]['student_top1'] != expected or any(row['student_observations'] != 19160 for row in rows):
        raise ValueError('Student benchmark differs from the recorded evaluation')
    write_json(path / 'boost_tuning_comparison.json', {
        'rows': rows, 'parameters': [model.get_params() for model in models], 'cv': cv,
        'untuned_refit_seconds': untuned_fit_seconds, 'threads': threads,
        'timing_scope': 'candidate scoring and Top-1 selection across all ten student trips',
        'reference': 'original frozen label method for each metric',
        'interpretation': 'This compares the boosted recipe before and after TPE. The active blend reuses its hyperparameters.',
    })
    write_csv(path / 'boost_tuning_comparison.csv', rows)
    print('Bayesian tuning comparison:', rows, flush=True)


def decoder_metrics(threads):
    path = ROOT_RESULTS / ROUTE_VERSION
    artifact = load_artifact(ROOT / 'artifacts/ordered_route_boost.joblib')
    frozen = read_json(path / 'student_test.json')['hmm']
    before = {'new_booster': [], 'boosting_blend': []}
    for trip in trip_dirs('student_test'):
        net, track = load_network(trip), load_observations(trip)
        xy, times = net.project(track[:, :2]), track[:, 2]
        candidates, features, _ = context_inputs(net, xy, times, ROUTE_VERSION)
        with np.load(FROZEN_LABELS / 'student_test' / f'{trip.name}.npz', allow_pickle=False) as data:
            route = data['reference']
        labels, _, _ = align_ordered_route(net, xy, times, route)
        started = perf_counter()
        new = probabilities(artifact['model'], features, candidates.arc_ids, threads)
        old = probabilities(artifact['previous_model'], features[..., :len(FEATURE_NAMES)], candidates.arc_ids, threads)
        blend = combine_scores(new, old, artifact['previous_weight'])
        shared_seconds = perf_counter() - started
        for name, scores in zip(before, (new, blend)):
            arcs, idx = select_top1(candidates.arc_ids, scores)
            fractions = candidates.fractions[np.arange(len(arcs)), idx]
            row = observation_metrics(labels, arcs, candidates.arc_ids)
            row.update(lcs_f1(collapse_arcs(arcs), route))
            row.update(evaluate_continuity(net, arcs, fractions, xy, times, artifact['decoder_config']))
            before[name].append({'trip_id': trip.name, 'shared_scoring_seconds': shared_seconds, **row})
    before = {name: summarize(trips) for name, trips in before.items()}
    per_trip = {row['trip_id']: row for row in before['boosting_blend']['trips']}
    comparison = []
    for row in frozen['trips']:
        old = per_trip[row['trip_id']]
        comparison.append({
            'trip_id': row['trip_id'], 'observations': row['observations'],
            'blend_top1': old['top1_accuracy'], 'decoded_top1': row['top1_accuracy'],
            'top1_difference': row['top1_accuracy'] - old['top1_accuracy'],
            'blend_lcs_f1': old['sequence_lcs_f1'], 'decoded_lcs_f1': row['sequence_lcs_f1'],
            'blend_continuity': old['route_continuity'], 'decoded_continuity': row['route_continuity'],
        })
    if len(comparison) != 10 or sum(row['observations'] for row in comparison) != 19160:
        raise ValueError('Expected all ten student-test trips and 19,160 GPS points')
    methods = {'new_booster': before['new_booster'], 'boosting_blend': before['boosting_blend'], 'blend_hmm': frozen}
    print('Final decoder comparison:', {key: (row['top1_accuracy'], row['sequence_lcs_f1'],
                                                row['route_continuity']) for key, row in methods.items()}, flush=True)
    write_json(path / 'classifier_decoder_comparison.json', {
        'methods': methods, 'trips': comparison,
        'reference': 'All 19,160 full-route labels and the supplied ordered reference routes',
    })
    write_csv(path / 'classifier_decoder_comparison.csv', comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('tuning', 'decoder'))
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    task = boost_tuning_metrics if args.task == 'tuning' else decoder_metrics
    task(args.threads)


if __name__ == '__main__':
    main()
