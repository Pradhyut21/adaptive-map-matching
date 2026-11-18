# PathSnap execution status

The implementation and local verification are complete. The physical two-laptop demonstration and receiving-terminal screenshot are pending. The team will run those checks later.

## Data and features

- Validated all 80 training trips (208,417 observations) and 10 student-test trips (19,160 observations). The EDA inventory includes file hashes, missing-value checks, graph connectivity, timestamp gaps and reference-route checks. See `results/eda/`.
- Derived approximate pointwise labels from the reference route using monotone alignment. Logged ambiguous, poor-fit and nearest-reference fallback cases in `results/feature_inventory.json`.
- Used up to ten candidates with a 250 m search radius. A training-trip comparison supported the change from 100 m. The resulting candidate miss rates are 12.03% on training and 4.65% on student testing. Misses remain in the Top-1 denominator. See `results/candidate_radius_experiment.json`.

## Training and Bayesian optimization

- Completed three-fold trip-grouped cross-validation and full student-test scoring for all five classifiers. Saved the models, fold membership, confusion matrices, timings and 50 student-trip prediction files. See `results/model_comparison.csv`, `results/cv/` and `results/execution_manifest.json`.
- Selected Extra Trees by mean grouped-CV Top-1. Ran all eight Optuna trials with three startup proposals followed by five guided TPE proposals. The best objective came from the first startup trial. Refit that configuration on all 80 training trips. See `results/optuna.sqlite3` and `results/tuning_summary.json`.
- Tuning raised mean grouped-CV Top-1 from 58.84% to 59.53% and student-test Top-1 from 70.61% to 71.66%. Candidate log loss increased from 0.2081 to 0.2097. The tuned model is the deployed artifact.
- The same-algorithm feature ablation measured 59.53% versus 57.19% grouped-CV Top-1 and 71.66% versus 68.25% student-test Top-1 with and without the two heading reliability features. Reference alignment also uses heading. The comparison measures agreement with derived labels. See `results/ablation.json`.

## Route reconstruction

| Method | Student-test Top-1 | Directed continuity | Ordered-sequence LCS F1 |
|---|---:|---:|---:|
| Nearest road | 42.96% | 75.58% | 0.2984 |
| Tuned Extra Trees | 71.66% | 92.57% | 0.4765 |
| Tuned Extra Trees + HMM | 76.13% | 99.87% | 0.7946 |

The HMM improved Top-1 on eight trips. It reduced Top-1 on trips `00000000` and `00000007`. All 19,160 observations were retained. Continuity includes 25 unreachable transitions among 19,149 eligible pairs and reports one excluded time gap separately. See `results/hmm_comparison.json` and `results/predictions/HMM/`.

## Local verification and deliverables

- All 39 behavior and API tests passed in `results/unit_tests.txt`.
- All ten live HTTP predictions returned 200. Each trip's Top-1 matched its recorded offline result exactly. Full inference took 0.32–7.82 seconds per trip on the test machine. Reference routes and per-point labels remained on the server. See `results/api_integration.json` and `results/execution_logs/actual_api_server.txt`.
- Verified desktop and mobile UI captures from a live request to a separate API service on the same machine. Retained the browser log viewer capture and the original text request logs in `results/`.
- The interactive Folium map includes GPS observations, road graph, supplied student reference and HMM prediction. The consolidated PDF and its source manifest are under `results/`. The project is published at [Pradhyut21/adaptive-map-matching](https://github.com/Pradhyut21/adaptive-map-matching).

## Pending team verification

- Run a trip from the second laptop over the LAN and capture the receiving terminal. Follow [the demo instructions](docs/deployment.md).
- Run the reserved evaluation when the organizers supply their files. Development and verification used the permitted training and student-test trips.
- Review the implementation and measurements, add the team's interpretation and disclose AI assistance according to the event policy.
