# PathSnap execution status

The deployed system loads `artifacts/gps_noise_boost.joblib` and its frozen decoder settings. It combines two Histogram Gradient Boosting scorers with directed Viterbi decoding, three-sample GPS smoothing, and a 15 m same-arc jitter tolerance. It scored **94.03%** on all 19,160 student-test observations and **95.51%** on all 19,674 evaluator observations. [Current comparison](results/model_comparison.csv), [student results](results/gps-noise-v1/student_test.json), [final assessment](results/final_evaluation_summary.json)

## Current measurements

| Measurement | Result | Evidence |
| --- | ---: | --- |
| Student exact road-arc Top-1 | 18,016 / 19,160 = 94.03% | `results/gps-noise-v1/student_test.json` |
| Student ordered route LCS F1 | 0.9795 | Same student result |
| Student strict raw-projection continuity | 96.23% | Same student result |
| Evaluator exact road-arc Top-1 | 18,790 / 19,674 = 95.51% | `results/final_evaluation_summary.json` |
| Behavior and API tests | 52 passed | `results/gps-noise-v1/unit_tests.txt` |
| Student HTTP checks | All 10 exactly matched frozen metrics | `results/gps-noise-v1/api_integration.json` |

Student and evaluator Top-1 use full-route labels. The earlier sparse alignment remains available as a diagnostic. Strict continuity uses raw GPS projections and can reject transitions accepted by the decoder's noise model.

## Training and selection

- All five original algorithms were fitted and evaluated using 80 training trips and three trip-grouped folds. Their fixed results are in `results/baseline_model_comparison.csv` and `results/model_comparison.json`.
- Extra Trees was selected during the original comparison. Its eight-trial TPE run raised mean grouped-CV Top-1 from 58.84% to 59.53% and student Top-1 from 70.61% to 71.66%. This was an earlier release.
- The contextual boosting model completed six TPE trials. Mean grouped-CV Top-1 rose from 62.24% to 62.61%, with student Top-1 rising from 73.40% to 73.71% under the earlier labels. See `results/context-boost-v1/boost_tuning_comparison.csv` for F1, loss and timing.
- The full-route scorer uses 35 GPS and graph features. The current decoder was selected on OOF scores from sixteen training trips, combining slow motion and a deterministic sample. Full-route Top-1 on that subset rose from 48.49% to 67.51%. See `results/gps-noise-v1/training_validation.json`.
- The decoder revision followed an observed evaluator failure. Settings were selected from training scores, and the revised frozen model was reassessed on the evaluator partition. This is a repeated assessment after development feedback.

## Deliverables and deployment

The [public repository](https://github.com/Pradhyut21/adaptive-map-matching) contains code, fitted models, configurations, experiments, a current Folium map, UI captures and the [consolidated PDF](results/PathSnap_Technical_Report.pdf). The [checklist](docs/deliverables.md) maps requirements to files. [The README](README.md) gives reproduction commands.

The running API serves all ten `evaluator_test` trips on port 8000. Set `PATHSNAP_PARTITION=student_test` for development scoring. Raw evaluator files and per-request assessment records remain in local data and ignored runtime folders. The public assessment record contains aggregate metrics.

LAN requests from a separate client reached the backend. The receiving CMD screenshot and team confirmation of the physical laptop setup remain pending. The team must review the implementation and results, add its interpretation, and provide the required AI-assistance disclosure.
