"""Build the public model summary from recorded experiments and frozen scores."""

import argparse
import csv
import json
from time import perf_counter

import numpy as np

from pathsnap.decoder_refinement import combine_scores
from pathsnap.metrics import candidate_metrics, select_top1
from pathsnap.predict import load_artifact
from pathsnap.refined_inputs import FEATURE_NAMES, ROUTE_VERSION
from pathsnap.settings import ROOT, trip_dirs
from pathsnap.geometry import load_network, load_observations
from pathsnap.refined_inputs import context_inputs
from pathsnap.route_alignment import align_ordered_route
from scripts.compare_route_alignment import CACHE, OLD_CACHE, probabilities
from scripts.refine_accuracy import FROZEN_LABELS, write_json

OUT = ROOT / 'results'
SUMMARY = OUT / 'final_model_summary.json'
REQUIRED_COLUMNS = [
    'Model', 'Algorithm Family', 'Key Configuration', 'CV Macro F1-score',
    'Top-1 Accuracy', 'Route continuity', 'Student-Test Macro F1-score',
    'Log Loss', 'Inference Time',
]
CONTEXT_COLUMNS = [
    'Status', 'Evaluator Top-1 Accuracy', 'Train Top-1 Accuracy', 'CV Top-1 Accuracy', 'Ordered Route LCS F1',
    'Partition', 'Reference Labels', 'Inference Time Scope', 'Source',
]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def model_metrics(threads):
    """Score fitted training rows, OOF validation rows and student candidates."""
    artifact = load_artifact()
    source = read(OUT / 'context-boost-v1/selection.json')['selected_recipe']
    names = ('full_route_booster', 'boosting_blend')
    accum = {name: {part: {'labels': [], 'scores': [], 'correct': 0, 'observations': 0,
                          'inference_seconds': 0.}
                    for part in ('train', 'cv', 'student_test')} for name in names}

    def add(partition, labels, arcs, new, old, elapsed=0.):
        combined = combine_scores(new, old, artifact['previous_weight'])
        valid = arcs >= 0
        for name, scores in zip(names, (new, combined)):
            record = accum[name][partition]
            pred, _ = select_top1(arcs, scores)
            record['correct'] += int((pred == labels).sum())
            record['observations'] += len(labels)
            record['inference_seconds'] += elapsed
            if partition != 'train':
                record['labels'].append((arcs == labels[:, None])[valid])
                record['scores'].append(scores[valid])

    for path in sorted((CACHE / 'train').glob('*.npz')):
        with np.load(path, allow_pickle=False) as cached:
            labels, arcs, features = cached['labels'], cached['candidates'], cached['features']
        oof_new = np.load(CACHE / 'oof' / f'{path.stem}.npy', allow_pickle=False)
        oof_old = np.load(OLD_CACHE / 'oof' / source / f'{path.stem}.npy', allow_pickle=False)
        add('cv', labels, arcs, oof_new, oof_old)
        start = perf_counter()
        new = probabilities(artifact['model'], features, arcs, threads)
        old = probabilities(artifact['previous_model'], features[..., :len(FEATURE_NAMES)], arcs, threads)
        add('train', labels, arcs, new, old, perf_counter() - start)
    for trip in trip_dirs('student_test'):
        net, track = load_network(trip), load_observations(trip)
        candidates, features, _ = context_inputs(net, net.project(track[:, :2]), track[:, 2], ROUTE_VERSION)
        # Use the already fixed full-route labels from the training experiment's
        # evaluation, deriving them only after the candidate scores are ready.
        start = perf_counter()
        new = probabilities(artifact['model'], features, candidates.arc_ids, threads)
        old = probabilities(artifact['previous_model'], features[..., :len(FEATURE_NAMES)], candidates.arc_ids, threads)
        elapsed = perf_counter() - start
        with np.load(FROZEN_LABELS / 'student_test' / f'{trip.name}.npz', allow_pickle=False) as frozen:
            route = frozen['reference']
        labels, _, _ = align_ordered_route(net, net.project(track[:, :2]), track[:, 2], route)
        add('student_test', labels, candidates.arc_ids, new, old, elapsed)
    summary = {}
    for name, partitions in accum.items():
        summary[name] = {}
        for partition, record in partitions.items():
            expected = 19160 if partition == 'student_test' else 208417
            if record['observations'] != expected:
                raise ValueError(f'Incomplete {partition} observations in the component comparison')
            result = {'top1_accuracy': record['correct'] / record['observations'],
                      'correct': record['correct'], 'observations': record['observations']}
            if partition != 'train':
                result.update(candidate_metrics(np.concatenate(record['labels']), np.concatenate(record['scores'])))
            if partition == 'student_test':
                result['shared_pair_scoring_seconds'] = record['inference_seconds']
            summary[name][partition] = result
        print(name, {key: value['top1_accuracy'] for key, value in summary[name].items()}, flush=True)
    reference = read(OUT / 'route-context-v1/classifier_decoder_comparison.json')['methods']
    for key, old_key in (('full_route_booster', 'new_booster'), ('boosting_blend', 'boosting_blend')):
        if summary[key]['student_test']['correct'] != reference[old_key]['correct']:
            raise ValueError('Recomputed classifier results differ from the frozen results')
    write_json(SUMMARY, {
        'components': summary, 'deployed_artifact': 'artifacts/gps_noise_boost.joblib',
        'selection': read(OUT / 'gps-noise-v1/training_validation.json')['selected'],
        'metric_scope': 'Candidate Macro F1 and log loss use all valid candidate rows; Top-1 counts every GPS observation.',
        'cv_scope': 'Pooled scores for all 80 training trips, each excluded from its fitting fold.',
        'label_method': 'full-route monotone alignment', 'threads': threads,
    })


def write_tables():
    baseline = read(OUT / 'model_comparison.json')['models']
    if len(baseline) != 5:
        raise ValueError('The original comparison must include all five completed models')
    rows = []
    assessment_path = OUT / 'final_evaluation_summary.json'
    local = OUT / 'runtime/evaluator-noise-v1/api_integration.json'
    if local.is_file():
        assessed = read(local)
        requests = assessed['requests']
        if not assessed['all_passed'] or len(requests) != 10 or assessed['health'].get('deployment_version') != 'gps-noise-v1':
            raise ValueError('Final evaluation evidence is incomplete or uses a different deployment')
        observations = sum(item['observations'] for item in requests)
        correct = sum(item['metrics']['correct'] for item in requests)
        write_json(assessment_path, {
            'partition': 'evaluator_test', 'deployment_version': 'gps-noise-v1',
            'trips': len(requests), 'observations': observations, 'correct': correct,
            'top1_accuracy': correct / observations, 'all_http_200': all(item['http_status'] == 200 for item in requests),
            'label_method': 'full-route monotone alignment, every GPS point included',
            'assessment_started_utc': assessed['started_utc'],
            'interpretation': 'Reassessment after an observed decoder failure. Settings for the revision were selected on training OOF predictions.',
        })
    assessment = read(assessment_path) if assessment_path.is_file() else None
    current = read(OUT / 'gps-noise-v1/student_test.json')['hmm']
    validation = read(OUT / 'gps-noise-v1/training_validation.json')
    artifact = load_artifact()
    settings = read(ROOT / 'artifacts/gps_noise_boost.config.json')
    classifier_only = read(OUT / 'route-context-v1/classifier_decoder_comparison.json')['methods']
    summary = read(SUMMARY)

    def row(values, **context):
        return {**dict(zip(REQUIRED_COLUMNS, values)), **context}

    na = 'N/A - decoder emits a road path; candidate metrics are in the classifier rows'
    rows.append(row([
        'Histogram Gradient Boosting blend + GPS-noise HMM (deployed)', 'Boosting + directed sequence decoding',
        json.dumps({'boosters': 2, 'blend_weight': settings['previous_weight'], 'leaves': 63, 'iterations': 220,
                    'smoothing_half_window': 3, 'same_arc_jitter_m': 15}, sort_keys=True),
        na, current['top1_accuracy'], current['route_continuity'], na, na,
        sum(item['inference_seconds'] for item in current['trips']),
    ], **{'Status': 'DEPLOYED',
           'Evaluator Top-1 Accuracy': assessment['top1_accuracy'] if assessment else 'N/A - final evaluation evidence unavailable',
           'Train Top-1 Accuracy': 'N/A - fitted full-pipeline training score was not measured',
           'CV Top-1 Accuracy': f"{validation['scores'][validation['selected']]['top1_accuracy']:.12f} - 16-trip OOF decoder selection subset",
           'Ordered Route LCS F1': current['sequence_lcs_f1'], 'Partition': 'student_test (19,160 GPS observations)',
           'Reference Labels': 'full-route monotone alignment',
           'Inference Time Scope': 'seconds summed across all 10 trips; graph + features + scoring + decoding',
           'Source': 'results/gps-noise-v1/student_test.json; results/final_evaluation_summary.json; artifacts/gps_noise_boost.config.json'}))
    for name, key, frozen_key in (
        ('Histogram Gradient Boosting (full-route component)', 'full_route_booster', 'new_booster'),
        ('Histogram Gradient Boosting blend (classifier only)', 'boosting_blend', 'boosting_blend'),
    ):
        values = summary['components'][key]
        frozen = classifier_only[frozen_key]
        rows.append(row([
            name, 'Gradient boosting', json.dumps({k: artifact['model'].get_params()[k] for k in
                ('max_iter', 'max_leaf_nodes', 'min_samples_leaf', 'learning_rate', 'l2_regularization')}, sort_keys=True),
            values['cv']['macro_f1'], values['student_test']['top1_accuracy'], frozen['route_continuity'],
            values['student_test']['macro_f1'], values['student_test']['log_loss'],
            values['student_test']['shared_pair_scoring_seconds'],
        ], **{'Status': 'DEPLOYED COMPONENT / CLASSIFIER DIAGNOSTIC',
               'Evaluator Top-1 Accuracy': 'N/A - component was not scored separately in the final assessment',
               'Train Top-1 Accuracy': values['train']['top1_accuracy'], 'CV Top-1 Accuracy': values['cv']['top1_accuracy'],
               'Ordered Route LCS F1': frozen['sequence_lcs_f1'], 'Partition': 'student_test (19,160 GPS observations)',
               'Reference Labels': 'full-route monotone alignment',
               'Inference Time Scope': 'seconds for the shared pair of booster predictions across all 10 cached-feature trips',
               'Source': 'results/final_model_summary.json; results/route-context-v1/classifier_decoder_comparison.json'}))
    for model in baseline:
        rows.append(row([
            model['name'], model['algorithm_family'], json.dumps(model['params'], sort_keys=True),
            model['cv']['metrics']['macro_f1'], model['student_test']['top1_accuracy'],
            model['student_test']['route_continuity'], model['student_test']['macro_f1'],
            model['student_test']['log_loss'], model['inference_seconds'],
        ], **{'Status': 'ORIGINAL BASELINE', 'Evaluator Top-1 Accuracy': 'N/A - baseline was not run in the final assessment',
               'Train Top-1 Accuracy': model['training_metrics']['top1_accuracy'],
               'CV Top-1 Accuracy': model['cv']['metrics']['fold_mean_top1'],
               'Ordered Route LCS F1': '', 'Partition': 'student_test (19,160 GPS observations)',
               'Reference Labels': 'original sparse route alignment',
               'Inference Time Scope': 'seconds for classifier scoring across all 10 cached-feature trips',
               'Source': 'results/model_comparison.json'}))
    with (OUT / 'model_comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS + CONTEXT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / 'baseline_model_comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows({key: value for key, value in record.items() if key in REQUIRED_COLUMNS}
                         for record in rows if record['Status'] == 'ORIGINAL BASELINE')
    methods = {**classifier_only, 'blend_hmm': current}
    before = {item['trip_id']: item for item in methods['boosting_blend']['trips']}
    trip_rows = []
    for item in current['trips']:
        raw = before[item['trip_id']]
        trip_rows.append({
            'trip_id': item['trip_id'], 'observations': item['observations'],
            'blend_top1': raw['top1_accuracy'], 'decoded_top1': item['top1_accuracy'],
            'top1_difference': item['top1_accuracy'] - raw['top1_accuracy'],
            'blend_lcs_f1': raw['sequence_lcs_f1'], 'decoded_lcs_f1': item['sequence_lcs_f1'],
            'blend_continuity': raw['route_continuity'], 'decoded_continuity': item['route_continuity'],
        })
    write_json(OUT / 'gps-noise-v1/classifier_decoder_comparison.json', {
        'methods': methods, 'trips': trip_rows,
        'reference': 'All 19,160 full-route labels and the supplied ordered reference routes',
        'source': 'Frozen classifier predictions and gps-noise-v1 decoded student results',
    })
    with (OUT / 'gps-noise-v1/classifier_decoder_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trip_rows[0]))
        writer.writeheader()
        writer.writerows(trip_rows)
    print('Updated current and baseline comparison tables', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score', action='store_true', help='Recompute component metrics using existing models and caches')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    if args.score:
        model_metrics(args.threads)
    write_tables()


if __name__ == '__main__':
    main()
