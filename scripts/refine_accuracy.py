"""Improve exact directed-arc matching with the existing boosting algorithm.

Keep the original candidate labels, partition, and all-point denominator.
Training and recipe selection use only grouped folds of the 80 training trips.
The student-test partition is opened by the explicit test command after fitting.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from pathsnap.geometry import CandidateSet, load_network
from pathsnap.decoder import DecoderConfig, evaluate_continuity
from pathsnap.decoder_refinement import RefinementConfig, decode_refined
from pathsnap.learning import make_model
from pathsnap.metrics import lcs_f1, select_top1, observation_metrics
from pathsnap.refined_inputs import FEATURE_NAMES, K, RADIUS, VERSION, context_inputs
from pathsnap.scoring import positive_scores
from pathsnap.settings import CACHE_ROOT, ROOT, SEED, trip_dirs


FROZEN_LABELS = CACHE_ROOT / 'trace-lite-v2'
WORK = CACHE_ROOT / VERSION
OUT = ROOT / 'results' / VERSION

RECIPES = {
    'context_standard': {'quality_weighted': False, 'max_leaf_nodes': 31, 'max_iter': 140},
    'context_weighted': {'quality_weighted': True, 'max_leaf_nodes': 31, 'max_iter': 140},
    'context_weighted_larger': {'quality_weighted': True, 'max_leaf_nodes': 63, 'max_iter': 160},
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare_train():
    destination = WORK / 'train'
    destination.mkdir(parents=True, exist_ok=True)
    training_trips = sorted((FROZEN_LABELS / 'train').glob('*.npz'))
    directories = {path.name: path for path in trip_dirs('train')}
    start = perf_counter()
    for index, source in enumerate(training_trips, 1):
        target = destination / source.name
        if target.is_file():
            continue
        with np.load(source, allow_pickle=False) as frozen:
            xy = frozen['track_xy']
            timestamps = frozen['timestamps']
            labels = frozen['labels']
            reference = frozen['reference']
        network = load_network(directories[source.stem])
        candidates, features, _ = context_inputs(network, xy, timestamps)
        correct = candidates.arc_ids == labels[:, None]
        covered = correct.any(axis=1)
        selected_distance = np.min(np.where(correct, candidates.distances, np.inf), axis=1)
        extra = np.maximum(selected_distance - candidates.distances[:, 0], 0)
        confidence = np.clip(1 / (1 + (extra / 25)**2 + (selected_distance / 60)**2), 0.15, 1)
        diagnostics = json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
        confidence[diagnostics['alignment']['poor_fit_indices']] *= 0.3
        rng = np.random.default_rng(SEED + int(source.stem))
        # Retain labeled groups and preserve all positives, nearby hard negatives,
        # and a deterministic sample of the remaining negatives during fitting.
        rank = np.arange(K)[None, :]
        sample = covered[:, None] & candidates.valid & (correct | (rank < 3) | (rng.random(correct.shape) < 0.12))
        weights = np.broadcast_to(confidence[:, None], sample.shape)[sample].astype(np.float32)
        temporary = target.with_suffix('.tmp.npz')
        np.savez(temporary, features=features, arc_ids=candidates.arc_ids,
                 distances=candidates.distances, fractions=candidates.fractions,
                 projected_xy=candidates.xy.astype(np.float32), labels=labels,
                 reference=reference, xy=xy, timestamps=timestamps,
                 sample=sample, weights=weights)
        temporary.replace(target)
        print(f'{index:02}/{len(training_trips)} {source.stem}: {covered.mean():.2%} coverage, '
              f'{sample.sum():,} fit rows ({perf_counter()-start:.1f}s elapsed)', flush=True)


def load_fit_rows(trip_ids):
    features = []
    targets = []
    weights = []
    for trip_id in trip_ids:
        with np.load(WORK / 'train' / f'{trip_id}.npz', allow_pickle=False) as data:
            sample = data['sample']
            features.append(data['features'][sample])
            targets.append((data['arc_ids'] == data['labels'][:, None])[sample])
            weights.append(data['weights'])
    return np.concatenate(features), np.concatenate(targets), np.concatenate(weights)


def fit(trip_ids, recipe, threads):
    features, targets, weights = load_fit_rows(trip_ids)
    params = {key: value for key, value in recipe.items() if key != 'quality_weighted'}
    model = make_model('HistGradientBoosting', params)
    with threadpool_limits(limits=threads):
        model.fit(features, targets, sample_weight=weights if recipe['quality_weighted'] else None)
    return model


def classify(model, features, arc_ids, threads):
    probabilities = np.zeros(arc_ids.shape, dtype=np.float32)
    valid = arc_ids >= 0
    with threadpool_limits(limits=threads):
        probabilities[valid] = positive_scores(model, features[valid])
    prediction, _ = select_top1(arc_ids, probabilities)
    return prediction, probabilities


def validate(threads):
    folds = [json.loads((ROOT / 'results' / 'cv' / f'HistGradientBoosting_fold{i}.json')
                        .read_text(encoding='utf-8')) for i in range(3)]
    validation = []
    for name, recipe in RECIPES.items():
        rows = []
        fold_scores = []
        for index, fold in enumerate(folds):
            start = perf_counter()
            model = fit(fold['training_trips'], recipe, threads)
            folder = WORK / 'oof' / name
            folder.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, folder / f'fold{index}.joblib', compress=3)
            records = []
            for trip_id in fold['validation_trips']:
                with np.load(WORK / 'train' / f'{trip_id}.npz', allow_pickle=False) as data:
                    arc_ids, labels = data['arc_ids'], data['labels']
                    prediction, probabilities = classify(model, data['features'], arc_ids, threads)
                np.save(folder / f'{trip_id}.npy', probabilities)
                records.append({'trip_id': trip_id, 'fold': index, **observation_metrics(labels, prediction, arc_ids)})
            score = sum(row['correct'] for row in records) / sum(row['observations'] for row in records)
            fold_scores.append(score)
            rows.extend(records)
            print(f'{name} fold {index}: exact Top-1 {score:.2%} in {perf_counter()-start:.1f}s', flush=True)
        validation.append({'recipe': name, 'params': recipe, 'fold_accuracy': fold_scores,
                           'pooled_accuracy': sum(row['correct'] for row in rows) / sum(row['observations'] for row in rows),
                           'mean_trip_accuracy': float(np.mean([row['top1_accuracy'] for row in rows])),
                           'mean_fold_accuracy': float(np.mean(fold_scores)), 'trips': rows})
        write_json(OUT / 'validation.json', {'recipes': validation})
    best = max(validation, key=lambda record: record['mean_fold_accuracy'])
    frozen = {'selected_recipe': best['recipe'], 'selection_metric': 'mean grouped-fold exact arc Top-1',
              'validation_accuracy': best['mean_fold_accuracy'], 'recipes': validation,
              'candidate_k': K, 'candidate_radius_m': RADIUS, 'feature_names': FEATURE_NAMES}
    write_json(OUT / 'selection.json', frozen)


def refit(threads):
    selection = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))
    recipe = selection['selected_recipe']
    ids = [path.stem for path in sorted((WORK / 'train').glob('*.npz'))]
    if len(ids) != 80:
        raise ValueError('Refit requires all 80 training trips')
    model = fit(ids, selection.get('selected_parameters', RECIPES.get(recipe)), threads)
    artifact = {'model': model, 'model_name': 'HistGradientBoosting', 'feature_names': FEATURE_NAMES,
                'feature_version': VERSION, 'candidate_k': K, 'candidate_radius_m': RADIUS,
                'recipe': recipe, 'training_trip_count': len(ids)}
    joblib.dump(artifact, ROOT / 'artifacts' / 'context_boost.joblib', compress=3)
    print(f'Fitted {recipe} on {len(ids)} training trips', flush=True)


def decoder_choices():
    raw = DecoderConfig()
    standard = RefinementConfig()
    dwell = replace(standard, same_arc_jitter_m=5.0)
    smooth = replace(dwell, smoothing_half_window=2, smooth_projections=True)
    strong = replace(dwell, emission_temperature=2.5)
    return {
        'original_geometry': (raw, standard),
        'allow_gps_dwell': (raw, dwell),
        'dwell_smooth': (replace(raw, beta=15), smooth),
        'stronger_transitions': (replace(raw, beta=11), strong),
        'favor_classifier': (replace(raw, beta=50), dwell),
        'stronger_transitions_realign': (replace(raw, beta=11), replace(strong, realign_predicted_route=True)),
        'smoothed_realign': (replace(raw, beta=7), replace(smooth, realign_predicted_route=True)),
    }


def tune(threads, trials):
    import optuna

    selection_path = OUT / 'selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    # Resume against the original untuned comparison, even if a previous
    # invocation already selected the winning TPE trial.
    untuned = max(selection['recipes'], key=lambda row: row['mean_fold_accuracy'])
    original = untuned['recipe']
    before = untuned['mean_fold_accuracy']
    folds = [json.loads((ROOT / 'results/cv' / f'HistGradientBoosting_fold{i}.json').read_text(encoding='utf-8'))
             for i in range(3)]
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=3)
    database = OUT / 'boost_tpe.sqlite3'
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name='context-boost-v1',
                                storage='sqlite:///' + database.as_posix(), load_if_exists=True)

    def objective(trial):
        recipe = {
            'max_leaf_nodes': trial.suggest_categorical('max_leaf_nodes', [31, 63, 95]),
            'max_iter': trial.suggest_categorical('max_iter', [120, 160, 220]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 120, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 0.06, 0.17, log=True),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.2, 12, log=True),
            'quality_weighted': True,
        }
        name = f'tpe_{trial.number:02d}'
        target = WORK / 'oof' / name
        target.mkdir(parents=True, exist_ok=True)
        scores = []
        for index, fold in enumerate(folds):
            start = perf_counter()
            model = fit(fold['training_trips'], recipe, threads)
            total, correct = 0, 0
            for trip_id in fold['validation_trips']:
                with np.load(WORK / 'train' / f'{trip_id}.npz', allow_pickle=False) as data:
                    arcs, labels = data['arc_ids'], data['labels']
                    point_prediction, probabilities = classify(model, data['features'], arcs, threads)
                np.save(target / f'{trip_id}.npy', probabilities)
                total += len(labels)
                correct += int((point_prediction == labels).sum())
            scores.append(correct / total)
            print(f'{name} fold {index}: {scores[-1]:.2%} in {perf_counter()-start:.1f}s', flush=True)
        trial.set_user_attr('fold_accuracy', scores)
        trial.set_user_attr('recipe', recipe)
        return float(np.mean(scores))

    remaining = max(0, trials - len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))))
    study.optimize(objective, n_trials=remaining)
    winner = study.best_trial
    improved = winner.value > before
    evidence = {'original_recipe': original, 'original_accuracy': before,
                'best_trial': winner.number, 'best_accuracy': winner.value, 'use_tuned': improved,
                'trials': [{'number': row.number, 'value': row.value, 'state': row.state.name,
                            'phase': 'startup' if row.number < 3 else 'guided_TPE',
                            'parameters': row.params, 'fold_accuracy': row.user_attrs.get('fold_accuracy')}
                           for row in study.trials]}
    write_json(OUT / 'boost_tuning.json', evidence)
    if improved:
        selection.update(selected_recipe=f'tpe_{winner.number:02d}', selected_parameters=winner.user_attrs['recipe'],
                         validation_accuracy=winner.value)
        write_json(selection_path, selection)
    print(f'TPE best {winner.value:.2%}, untuned {before:.2%}, selected {selection["selected_recipe"]}', flush=True)


def restore_missing_oof(threads):
    """Recreate untracked OOF caches when resuming a published Optuna study."""
    selection = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))
    name = selection['selected_recipe']
    recipe = selection.get('selected_parameters', RECIPES.get(name))
    destination = WORK / 'oof' / name
    destination.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        fold = json.loads((ROOT / 'results/cv' / f'HistGradientBoosting_fold{index}.json').read_text(encoding='utf-8'))
        missing = [trip_id for trip_id in fold['validation_trips']
                   if not (destination / f'{trip_id}.npy').is_file()]
        if not missing:
            continue
        if set(fold['training_trips']) & set(fold['validation_trips']):
            raise ValueError('OOF recovery requires disjoint trip groups')
        model = fit(fold['training_trips'], recipe, threads)
        for trip_id in missing:
            with np.load(WORK / 'train' / f'{trip_id}.npz', allow_pickle=False) as data:
                _, probabilities = classify(model, data['features'], data['arc_ids'], threads)
            np.save(destination / f'{trip_id}.npy', probabilities)
        print(f'Restored {len(missing)} training OOF score files for fold {index}', flush=True)


def calibrate(holdout_trips, threads=2):
    restore_missing_oof(threads)
    selected = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))['selected_recipe']
    folders = {path.name: path for path in trip_dirs('train')}
    train_paths = list((WORK / 'train').glob('*.npz'))
    order = sorted(train_paths, key=lambda path: hashlib.sha256(f'refined-decoder:{SEED}:{path.stem}'.encode()).digest())
    selected_trips = order[:holdout_trips]
    settings = decoder_choices()
    rows = {name: [] for name in settings}
    for path in selected_trips:
        with np.load(path, allow_pickle=False) as data:
            candidates = CandidateSet(data['arc_ids'], data['distances'], data['fractions'], data['projected_xy'])
            xy, timestamps = data['xy'], data['timestamps']
            labels, route = data['labels'], data['reference']
        prob = np.load(WORK / 'oof' / selected / f'{path.stem}.npy', allow_pickle=False)
        graph = load_network(folders[path.stem])
        for name, (config, refinement) in settings.items():
            result = decode_refined(graph, candidates, prob, xy, timestamps, config=config, refinement=refinement)
            rows[name].append({'trip_id': path.stem, **observation_metrics(labels, result.arc_ids, candidates.arc_ids),
                               **lcs_f1(result.route_arc_ids, route)})
        print(path.stem, {name: round(trips[-1]['top1_accuracy'], 4) for name, trips in rows.items()}, flush=True)
    summaries = {name: summarize(trips) for name, trips in rows.items()}
    winner = max(summaries, key=lambda name: summaries[name]['top1_accuracy'])
    config, refinement = settings[winner]
    result = {'selected': winner, 'decoder': asdict(config), 'refinement': asdict(refinement),
              'selection': 'pooled exact Top-1 on fixed training trips with out-of-fold boosting probabilities',
              'trip_ids': [path.stem for path in selected_trips], 'scores': summaries}
    write_json(OUT / 'decoder_selection.json', result)
    write_json(ROOT / 'artifacts' / 'context_boost.config.json', {'decoder': asdict(config), 'refinement': asdict(refinement)})
    print(f'Selected {winner}: {summaries[winner]["top1_accuracy"]:.2%} held-out training Top-1', flush=True)


def summarize(trips):
    count = sum(trip['observations'] for trip in trips)
    result = {
        'observations': count,
        'correct': sum(trip['correct'] for trip in trips),
        'top1_accuracy': sum(trip['correct'] for trip in trips) / count,
        'mean_trip_top1_accuracy': float(np.mean([trip['top1_accuracy'] for trip in trips])),
        'candidate_miss_rate': sum(trip['candidate_miss_count'] for trip in trips) / count,
        'trips': trips,
    }
    if 'sequence_lcs' in trips[0]:
        sequence_length = sum(trip['predicted_sequence_length'] + trip['reference_sequence_length'] for trip in trips)
        result['sequence_lcs_f1'] = 2 * sum(trip['sequence_lcs'] for trip in trips) / max(1, sequence_length)
    if 'eligible_transitions' in trips[0]:
        eligible = sum(trip['eligible_transitions'] for trip in trips)
        result['route_continuity'] = sum(trip['feasible_transitions'] for trip in trips) / max(1, eligible)
    return result


def test_frozen(threads):
    from pathsnap.predict import load_artifact, predict_trip

    settings = json.loads((OUT / 'decoder_selection.json').read_text(encoding='utf-8'))
    artifact = load_artifact(ROOT / 'artifacts' / 'context_boost.joblib')
    independent, decoded_rows = [], []
    destination = OUT / 'predictions'
    destination.mkdir(parents=True, exist_ok=True)
    for trip in trip_dirs('student_test'):
        with threadpool_limits(limits=threads):
            prediction = predict_trip(trip, artifact=artifact)
        # Labels are opened only after the full prediction is returned.
        with np.load(FROZEN_LABELS / 'student_test' / f'{trip.name}.npz', allow_pickle=False) as frozen:
            labels, reference = frozen['labels'], frozen['reference']
            if not np.array_equal(frozen['timestamps'], prediction.track[:, 2]):
                raise ValueError('Frozen labels do not match the current trip timeline')
        arcs = prediction.candidates.arc_ids
        decoded = prediction.decoded
        independent.append({'trip_id': trip.name, **observation_metrics(labels, prediction.classifier_arcs, arcs)})
        fractions = prediction.candidates.fractions[np.arange(len(labels)), np.maximum(decoded.candidate_indices, 0)]
        continuity = evaluate_continuity(prediction.network, decoded.arc_ids, fractions,
                                         prediction.track_xy, prediction.track[:, 2], config=settings['decoder'])
        row = {'trip_id': trip.name, **observation_metrics(labels, decoded.arc_ids, arcs),
               **lcs_f1(decoded.route_arc_ids, reference), **continuity, **prediction.timing}
        decoded_rows.append(row)
        np.savez_compressed(destination / f'{trip.name}.npz', predictions=decoded.arc_ids,
                            candidate_indices=decoded.candidate_indices, route_arc_ids=decoded.route_arc_ids)
        write_json(destination / f'{trip.name}.json', {'trip_id': trip.name, 'metrics': row,
                   'route_segments': decoded.route_segments, 'warnings': decoded.warnings})
        print(f'{trip.name}: boost {independent[-1]["top1_accuracy"]:.2%}, decoder {row["top1_accuracy"]:.2%}', flush=True)
    if len(decoded_rows) != 10:
        raise ValueError('Expect exactly ten student-test trips')
    output = {'partition': 'student_test', 'reference': 'frozen original derived labels', 'selected_recipe': artifact['recipe'],
              'configuration': {'decoder': settings['decoder'], 'refinement': settings['refinement']},
              'classifier': summarize(independent), 'hmm': summarize(decoded_rows)}
    write_json(OUT / 'student_test.json', output)
    print(f'Frozen student-test Top-1: {output["hmm"]["top1_accuracy"]:.2%}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate', 'tune', 'refit', 'calibrate', 'test'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--holdout-trips', type=int, default=16)
    parser.add_argument('--trials', type=int, default=6)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_train()
    elif args.command == 'validate':
        validate(args.threads)
    elif args.command == 'tune':
        tune(args.threads, args.trials)
    elif args.command == 'refit':
        refit(args.threads)
    elif args.command == 'calibrate':
        calibrate(args.holdout_trips, args.threads)
    else:
        test_frozen(args.threads)


if __name__ == '__main__':
    main()
