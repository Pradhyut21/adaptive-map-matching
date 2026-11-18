"""Compare route-derived supervision on the same whole-trip training folds."""

import argparse
import json

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from pathsnap.geometry import load_network
from pathsnap.learning import make_model
from pathsnap.metrics import select_top1
from pathsnap.refined_inputs import K, RADIUS, ROUTE_VERSION, SCHEMAS, VERSION, context_inputs
from pathsnap.route_alignment import align_ordered_route
from pathsnap.settings import CACHE_ROOT, ROOT, trip_dirs
from scripts.refine_accuracy import write_json


OUT = ROOT / 'results' / ROUTE_VERSION
CACHE = CACHE_ROOT / ROUTE_VERSION
OLD_CACHE = CACHE_ROOT / VERSION
LEGACY = CACHE_ROOT / 'trace-lite-v2'
ALIGNMENT_FIT_LIMIT_M = 60.0


def training_folds():
    return [
        json.loads((ROOT / 'results' / 'cv' / f'HistGradientBoosting_fold{i}.json').read_text(encoding='utf-8'))
        for i in range(3)
    ]


def prepare():
    target = CACHE / 'train'
    target.mkdir(parents=True, exist_ok=True)
    for trip in trip_dirs('train'):
        path = target / f'{trip.name}.npz'
        if path.is_file():
            continue
        with np.load(LEGACY / 'train' / path.name, allow_pickle=False) as old:
            xy, timeline = old['track_xy'], old['timestamps']
            route, legacy_labels = old['reference'], old['labels']
        graph = load_network(trip)
        labels, _, distances = align_ordered_route(graph, xy, timeline, route)
        candidates, features, _ = context_inputs(graph, xy, timeline, ROUTE_VERSION)
        positive = labels[:, None] == candidates.arc_ids
        fit_rows = (np.arange(len(labels)) % 2 == 0) & (distances <= ALIGNMENT_FIT_LIMIT_M) & positive.any(axis=1)
        # Keep whole candidate groups for retained fixes. Remove distant negatives
        # outside the noise envelope used for supervision.
        fit_mask = fit_rows[:, None] & candidates.valid & ((candidates.distances <= 65) | positive)
        temporary = path.with_suffix('.tmp.npz')
        np.savez(temporary, labels=labels, legacy_labels=legacy_labels, label_distances=distances,
                 candidates=candidates.arc_ids, features=features, fit_mask=fit_mask)
        temporary.replace(path)
        print(f'{trip.name}: {np.mean(labels != legacy_labels):.2%} labels differ; {fit_mask.sum():,} fit rows', flush=True)


def fit(ids, threads):
    x, y = [], []
    for trip_id in ids:
        with np.load(CACHE / 'train' / f'{trip_id}.npz', allow_pickle=False) as trip:
            mask = trip['fit_mask']
            x.append(trip['features'][mask])
            y.append((trip['candidates'] == trip['labels'][:, None])[mask])
    selection = json.loads((ROOT / 'results' / VERSION / 'selection.json').read_text(encoding='utf-8'))
    params = {key: value for key, value in selection['selected_parameters'].items() if key != 'quality_weighted'}
    model = make_model('HistGradientBoosting', params)
    with threadpool_limits(limits=threads):
        return model.fit(np.concatenate(x), np.concatenate(y))


def probabilities(model, features, candidates, threads):
    valid = candidates >= 0
    scores = np.zeros(candidates.shape, dtype=np.float32)
    with threadpool_limits(limits=threads):
        scores[valid] = model.predict_proba(features[valid])[:, 1]
    return scores


def compare():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    prepare()
    oof = CACHE / 'oof'
    oof.mkdir(parents=True, exist_ok=True)
    previous = json.loads((ROOT / 'results' / VERSION / 'selection.json').read_text(encoding='utf-8'))['selected_recipe']
    results = []
    all_training = []
    for index, fold in enumerate(training_folds()):
        train_ids, validation_ids = fold['training_trips'], fold['validation_trips']
        if set(train_ids) & set(validation_ids):
            raise ValueError('Training and validation trips overlap')
        all_training.extend(validation_ids)
        model = fit(train_ids, args.threads)
        counts = dict(observations=0, label_changes=0, old_legacy=0, new_legacy=0,
                      old_ordered=0, new_ordered=0, legacy_misses=0, ordered_misses=0)
        trips = []
        for trip_id in validation_ids:
            with np.load(CACHE / 'train' / f'{trip_id}.npz', allow_pickle=False) as data:
                candidates, labels, old_labels = data['candidates'], data['labels'], data['legacy_labels']
                score = probabilities(model, data['features'], candidates, args.threads)
            with np.load(OLD_CACHE / 'train' / f'{trip_id}.npz', allow_pickle=False) as old_data:
                if not np.array_equal(candidates, old_data['arc_ids']):
                    raise ValueError('Candidate sets differ from prior run')
            old_scores = np.load(OLD_CACHE / 'oof' / previous / f'{trip_id}.npy', allow_pickle=False)
            old_pred, _ = select_top1(candidates, old_scores)
            new_pred, _ = select_top1(candidates, score)
            row = dict(observations=len(labels), label_changes=int((labels != old_labels).sum()),
                       old_legacy=int((old_pred == old_labels).sum()), new_legacy=int((new_pred == old_labels).sum()),
                       old_ordered=int((old_pred == labels).sum()), new_ordered=int((new_pred == labels).sum()),
                       legacy_misses=int((~(candidates == old_labels[:, None]).any(axis=1)).sum()),
                       ordered_misses=int((~(candidates == labels[:, None]).any(axis=1)).sum()))
            for key in counts:
                counts[key] += row[key]
            trips.append({'trip_id': trip_id, **row})
            np.save(oof / f'{trip_id}.npy', score)
        results.append({'fold': index, **counts, 'trips': trips})
        print(f"Fold {index}: {counts}", flush=True)
        write_json(OUT / 'grouped_comparison.json', {'reference_fit_limit_m': ALIGNMENT_FIT_LIMIT_M, 'folds': results})
    if len(all_training) != 80 or len(set(all_training)) != 80:
        raise ValueError('Expected 80 distinct training trips in validation')
    model = fit(sorted(all_training), args.threads)
    artifact = dict(model=model, model_name='HistGradientBoosting', feature_version=ROUTE_VERSION,
                    feature_names=SCHEMAS[ROUTE_VERSION], candidate_k=K, candidate_radius_m=RADIUS,
                    training_trip_count=80, recipe='ordered-labels-context')
    joblib.dump(artifact, ROOT / 'artifacts' / 'ordered_route_boost.joblib', compress=3)
    print('Saved model fitted on 80 training trips', flush=True)


if __name__ == '__main__':
    compare()
