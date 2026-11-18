"""Freeze a training-selected decoder, then score the permitted student set."""

import hashlib
import json

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from pathsnap.decoder import evaluate_continuity
from pathsnap.predict import load_artifact, predict_trip
from pathsnap.route_alignment import align_ordered_route
from pathsnap.settings import ROOT, trip_dirs
from scripts.compare_route_alignment import LEGACY
from scripts.refine_accuracy import write_json
from scripts.validate_route_decoder import score_candidates, summarize_dual
from scripts.validate_gps_noise import OUT


def main():
    training = json.loads((OUT / 'training_validation.json').read_text(encoding='utf-8'))
    if training['selected'] == 'frozen_baseline':
        raise RuntimeError('Training validation did not select a noise-handling change')
    original_model = ROOT / 'artifacts/ordered_route_boost.joblib'
    model = joblib.load(original_model)
    model['deployment_version'] = 'gps-noise-v1'
    model['decoder_selection'] = training['selected']
    artifact_path = ROOT / 'artifacts/gps_noise_boost.joblib'
    joblib.dump(model, artifact_path, compress=3)
    write_json(artifact_path.with_suffix('.config.json'), training['settings'])
    frozen = {'selected': training['selected'], 'feature_version': model['feature_version'],
              'source_model_sha256': hashlib.sha256(original_model.read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
              'config_sha256': hashlib.sha256(artifact_path.with_suffix('.config.json').read_bytes()).hexdigest(),
              'training_validation_sha256': hashlib.sha256((OUT / 'training_validation.json').read_bytes()).hexdigest(),
              'selection_source': 'training out-of-fold predictions', 'student_test_opened': False}
    write_json(OUT / 'freeze.json', frozen)
    artifact = load_artifact(artifact_path)
    rows = []
    for trip in trip_dirs('student_test'):
        with threadpool_limits(limits=3):
            prediction = predict_trip(trip, artifact=artifact)
        with np.load(LEGACY / 'student_test' / f'{trip.name}.npz', allow_pickle=False) as reference:
            route, old_labels = reference['reference'], reference['labels']
            if not np.array_equal(reference['timestamps'], prediction.track[:, 2]):
                raise ValueError('Frozen label timeline differs from current trip')
        labels, _, distances = align_ordered_route(prediction.network, prediction.track_xy,
                                                   prediction.track[:, 2], route)
        result = prediction.decoded
        row = score_candidates(labels, old_labels, prediction.candidates, result, route, distances)
        fractions = prediction.candidates.fractions[np.arange(len(labels)), np.maximum(result.candidate_indices, 0)]
        row.update(evaluate_continuity(prediction.network, result.arc_ids, fractions, prediction.track_xy,
                                      prediction.track[:, 2], result.stats['config']))
        row.update(prediction.timing)
        rows.append({'trip_id': trip.name, **row})
        output = OUT / 'predictions'
        output.mkdir(exist_ok=True)
        np.savez_compressed(output / f'{trip.name}.npz', predictions=result.arc_ids,
                            candidate_indices=result.candidate_indices, route_arc_ids=result.route_arc_ids)
        write_json(output / f'{trip.name}.json', {'route_segments': result.route_segments,
                                                'warnings': result.warnings, 'metrics': rows[-1]})
        print(trip.name, f"{row['top1_accuracy']:.2%}", flush=True)
    if len(rows) != 10:
        raise ValueError('Expected all ten permitted student trips')
    write_json(OUT / 'student_test.json', {'partition': 'student_test', 'hmm': summarize_dual(rows),
                                          'selection_source': 'training only', 'recipe': training['selected']})
    print(f"Frozen student-test Top-1: {summarize_dual(rows)['top1_accuracy']:.2%}", flush=True)


if __name__ == '__main__':
    main()
