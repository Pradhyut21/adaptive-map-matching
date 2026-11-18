# PathSnap execution status

The full-route boost blend is active. On the same student-test points and full-route reference labels, it improved from 78.41% to 86.41% exact Top-1. It reached 74.97% with the earlier labels versus 80.56% for the previous model. Ordered arc LCS F1 rose from 0.8790 to 0.8867 and strict continuity fell from 98.80% to 98.27%. See [paired results](results/route-context-v1/student_test.json) and [peer review](docs/peer_approach.md). The physical two-laptop demonstration and receiving-terminal screenshot are pending.

## Data and features

- Validated all 80 training trips (208,417 observations) and 10 student-test trips (19,160 observations). The EDA inventory includes file hashes, missing-value checks, graph connectivity, timestamp gaps and reference-route checks. See `results/eda/`.
- Derived approximate pointwise labels from the reference route using monotone alignment. Logged ambiguous, poor-fit and nearest-reference fallback cases in `results/feature_inventory.json`.
- The original model used up to ten candidates with a 250 m search radius. A training-trip comparison supported the change from 100 m. The resulting original candidate miss rates were 12.03% on training and 4.65% on student testing. The refinement uses 20 candidates, 28 GPS and geometry inputs and the same frozen reference labels. Misses remain in the Top-1 denominator. See `results/candidate_radius_experiment.json` and `results/context-boost-v1/selection.json`.

## Training and Bayesian optimization

- Completed three-fold trip-grouped cross-validation and full student-test scoring for all five classifiers. Saved the models, fold membership, confusion matrices, timings and 50 student-trip prediction files. See `results/model_comparison.csv`, `results/cv/` and `results/execution_manifest.json`.
- Selected Extra Trees by mean grouped-CV Top-1. Ran all eight Optuna trials with three startup proposals followed by five guided TPE proposals. The best objective came from the first startup trial. Refit that configuration on all 80 training trips. See `results/optuna.sqlite3` and `results/tuning_summary.json`.
- Original tuning raised mean grouped-CV Top-1 from 58.84% to 59.53% and student-test Top-1 from 70.61% to 71.66%. Candidate log loss increased from 0.2081 to 0.2097. Tuned Extra Trees is preserved as `artifacts/final_model.joblib`.
- The same-algorithm feature ablation measured 59.53% versus 57.19% grouped-CV Top-1 and 71.66% versus 68.25% student-test Top-1 with and without the two heading reliability features. Reference alignment also uses heading. The comparison measures agreement with derived labels. See `results/ablation.json`.
- The refinement compared three Histogram Gradient Boosting configurations on the original grouped folds. The best initial weighted variant reached 62.24% mean fold Top-1. TPE trial 5 of six completed trials reached 62.61% and was selected. It was refit on all 80 training trips. Decoder settings were selected using out-of-fold scores on 16 fixed training trips, reaching 79.07% pooled Top-1 over 30,651 observations. See `results/context-boost-v1/{selection,boost_tuning,decoder_selection}.json`.

## Route reconstruction

| Method | Student-test Top-1 | Directed continuity | Ordered-sequence LCS F1 |
|---|---:|---:|---:|
| Nearest road | 42.96% | 75.58% | 0.2984 |
| Tuned Extra Trees | 71.66% | 92.57% | 0.4765 |
| Tuned Extra Trees + HMM | 76.13% | 99.87% | 0.7946 |
| Refined boosting classifier | 73.71% | — | — |
| **Previous refined boosting + decoder** | **80.56%** | **98.80%** | **0.8790** |

The original HMM improved Top-1 over the Extra Trees classifier on eight trips and reduced it on `00000000` and `00000007`. The refined model and decoder improved over the original HMM on all ten trips. Exact matches increased from 14,587 to 15,436 of 19,160 observations (+4.43 percentage points). Candidate misses decreased from 890 to 300. Feasible transitions decreased from 19,124 to 18,920 of 19,149 eligible pairs, with one time gap excluded in both runs. The ordered-sequence LCS F1 improved to 0.8790. These metrics describe agreement with the frozen approximate labels on the student-test trips. See `results/hmm_comparison.json`, `results/predictions/HMM/` and `results/context-boost-v1/student_test.json`.

## Local verification and deliverables

- The active matcher passed 51 unit and API tests. All ten genuine predictions from the host's Wi-Fi address returned 200 and exactly reproduced both offline Top-1 measures and the route LCS F1. The live desktop and mobile run had no page errors or horizontal overflow. See `results/route-context-v1/`. The separate-device test needs a client on Laptop 2.

- All 49 behavior and API tests passed in `results/context-boost-v1/unit_tests.txt`, including the new contextual features, frozen artifact selection and decoder edge cases.
- All ten refined-model live HTTP predictions returned 200. Each trip's Top-1 and ordered-sequence LCS F1 exactly matched its frozen offline result. Full inference took 1.00–22.05 seconds per trip on the test machine. Reference routes and per-point labels remained on the server. See `results/context-boost-v1/api_integration.json` and `results/context-boost-v1/execution_logs/actual_api_server.txt`. Original Extra Trees evidence remains separately in `results/`.
- Verified updated desktop and mobile UI captures from a live request to the refined API on a separate same-machine service. No JavaScript errors or horizontal overflow. Evidence is in `results/context-boost-v1/screenshots/`.
- The interactive Folium map includes GPS observations, road graph, supplied student reference and HMM prediction. The consolidated PDF and its source manifest are under `results/`. The project is published at [Pradhyut21/adaptive-map-matching](https://github.com/Pradhyut21/adaptive-map-matching).

## Pending team verification

- Run a trip from the second laptop over the LAN and capture the receiving terminal. Follow [the demo instructions](docs/deployment.md).
- Run the reserved evaluation when the organizers supply their files. Development and verification used the permitted training and student-test trips.
- Review the implementation and measurements, add the team's interpretation and disclose AI assistance according to the event policy.
