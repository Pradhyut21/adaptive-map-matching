"""Verify the frozen refinement artifacts and scores without running inference or fitting."""

import hashlib
import json
import math

import numpy as np

from pathsnap.predict import load_artifact
from pathsnap.refined_inputs import VERSION
from pathsnap.settings import CACHE_ROOT, ROOT, SEED, trip_dirs


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit():
    results = ROOT / "results" / VERSION
    cache = CACHE_ROOT / VERSION
    train = {trip.name for trip in trip_dirs("train")}
    student = {trip.name for trip in trip_dirs("student_test")}
    require(len(train) == 80 and len(student) == 10 and train.isdisjoint(student),
            "Expected 80 training trips and 10 disjoint student-test trips")

    fold_paths = [ROOT / "results" / "cv" / f"HistGradientBoosting_fold{i}.json" for i in range(3)]
    held_out = set()
    fold_summary = []
    for path in fold_paths:
        fold = read_json(path)
        fit, valid = set(fold["training_trips"]), set(fold["validation_trips"])
        require(len(fit) == len(fold["training_trips"]) and len(valid) == len(fold["validation_trips"]),
                f"Repeated ID within {path.name}")
        require(fit.isdisjoint(valid) and fit | valid == train and held_out.isdisjoint(valid),
                f"Unexpected group overlap or omission in {path.name}")
        held_out.update(valid)
        fold_summary.append({"fit_trips": len(fit), "validation_trips": len(valid)})
    require(held_out == train, "Every training trip must be held out exactly once")

    selection_path = results / "selection.json"
    decoder_path = results / "decoder_selection.json"
    test_path = results / "student_test.json"
    selection, decoder, result = map(read_json, (selection_path, decoder_path, test_path))
    recipe = selection["selected_recipe"]
    require(result["partition"] == "student_test" and result["selected_recipe"] == recipe,
            "Reported evaluation partition or selected recipe differs")
    subset = decoder["trip_ids"]
    order = sorted(train, key=lambda name: hashlib.sha256(f"refined-decoder:{SEED}:{name}".encode()).digest())
    require(len(subset) == 16 and subset == order[:16] and set(subset) <= held_out,
            "Decoder calibration does not match the fixed training subset")
    score_paths = sorted((cache / "oof" / recipe).glob("*.npy"))
    require({path.stem for path in score_paths} == train, "Selected recipe must have 80 training OOF score files")
    for path in score_paths:
        with np.load(cache / "train" / f"{path.stem}.npz", allow_pickle=False) as trip:
            require(np.load(path, mmap_mode="r", allow_pickle=False).shape == trip["arc_ids"].shape,
                    f"OOF scores do not match saved candidate dimensions for {path.stem}")

    model_path = ROOT / "artifacts" / "context_boost.joblib"
    config_path = model_path.with_suffix(".config.json")
    active_path = ROOT / "artifacts" / "active_model.json"
    chosen_settings = {key: decoder[key] for key in ("decoder", "refinement")}
    require(read_json(config_path) == chosen_settings == result["configuration"],
            "Decoder artifact, selection and student-test settings must match")
    require(read_json(active_path)["file"] == model_path.name, "Active artifact differs from audited boosted model")
    artifact = load_artifact(model_path)
    require(artifact["training_trip_count"] == 80 and artifact["recipe"] == recipe
            and artifact["feature_names"] == selection["feature_names"]
            and artifact["candidate_k"] == selection["candidate_k"]
            and artifact["candidate_radius_m"] == selection["candidate_radius_m"],
            "Model metadata does not match the chosen recipe and schema")
    settings = artifact["model"].get_params()
    for key, value in selection.get("selected_parameters", {}).items():
        if key != "quality_weighted":
            require(settings[key] == value, f"Fitted estimator parameter {key} differs from selection")

    correct = observations = misses = 0
    seen = set()
    hashes = [model_path, config_path, active_path, selection_path, decoder_path, test_path, *fold_paths]
    classifier_by_id = {r["trip_id"]: r for r in result["classifier"]["trips"]}
    require(len(result["classifier"]["trips"]) == 10 and set(classifier_by_id) == student,
            "Classifier comparison and decoded result must use the same student trips")
    for record in result["hmm"]["trips"]:
        trip_id = record["trip_id"]
        require(trip_id in student and trip_id not in seen, f"Unexpected or repeated evaluated trip {trip_id}")
        seen.add(trip_id)
        prediction_path = results / "predictions" / f"{trip_id}.npz"
        frozen_path = CACHE_ROOT / "trace-lite-v2" / "student_test" / f"{trip_id}.npz"
        with np.load(prediction_path, allow_pickle=False) as saved, np.load(frozen_path, allow_pickle=False) as frozen:
            predictions, labels = saved["predictions"], frozen["labels"]
            require(predictions.shape == labels.shape == (record["observations"],) and np.all(labels >= 0),
                    f"Expected a prediction and frozen original label per observation in {trip_id}")
            matches = int(np.count_nonzero(predictions == labels))
        require(matches == record["correct"] and math.isclose(matches / len(labels), record["top1_accuracy"]),
                f"Frozen exact-arc matches differ from recorded result for {trip_id}")
        missing = record["candidate_miss_count"]
        classifier = classifier_by_id[trip_id]
        require(0 <= missing <= len(labels) - matches and missing == classifier["candidate_miss_count"]
                and len(labels) == classifier["observations"]
                and math.isclose(missing / len(labels), record["candidate_miss_rate"]),
                f"Persisted candidate-miss records disagree for {trip_id}")
        correct += matches
        observations += len(labels)
        misses += missing
        hashes.append(prediction_path)

    aggregate = result["hmm"]
    require(seen == student and observations == 19160 and correct == 15436,
            "Audit did not reproduce all 19,160 observations and 15,436 exact arc matches")
    require(aggregate["observations"] == observations and aggregate["correct"] == correct
            and math.isclose(aggregate["top1_accuracy"], correct / observations)
            and math.isclose(aggregate["candidate_miss_rate"], misses / observations),
            "Aggregate accuracy or persisted candidate-miss denominator disagrees")
    return {
        "status": "passed",
        "training_trips": len(train),
        "student_test_trips": len(student),
        "folds": fold_summary,
        "selected_recipe": recipe,
        "out_of_fold_score_files": len(score_paths),
        "decoder_calibration_training_trips": subset,
        "student_test": {"observations": observations, "correct": correct, "exact_arc_top1": correct / observations,
                         "persisted_candidate_misses": misses, "candidate_miss_rate": misses / observations},
        "sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in hashes},
        "interpretation": [
            "Student test is a reused benchmark across project versions. The final configuration was frozen using training data before this round of student-test evaluation.",
            "Grouped validation was used for recipe selection and decoder calibration, so those scores are selection evidence rather than an independent generalization estimate.",
            "The audit compares saved predictions against frozen original point labels and sums persisted candidate-miss counts. It does not refit or select a model.",
        ],
    }


def main():
    result = audit()
    output = ROOT / "results" / VERSION / "integrity_audit.json"
    temporary = output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Integrity audit passed. Reproduced {result['student_test']['correct']:,} / "
          f"{result['student_test']['observations']:,} exact matches. Wrote {output.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
