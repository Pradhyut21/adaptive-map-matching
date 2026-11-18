# PathSnap Trace-Lite

Independent, CPU-based noisy GPS map matching: learned candidate scoring, directed network transitions, and a custom log-space Viterbi decoder. The live evaluator uses FastAPI on Laptop 1 and a plain HTML/CSS/JavaScript UI on Laptop 2. The UI draws actual supplied road geometry without external tiles or CDN dependencies.

This repository was implemented with AI assistance. Every team member must review the code, reproduce the measurements, interpret the recorded results independently, and make the required disclosure. This README is implementation documentation, not evidence that the team has completed its viva or two-device demonstration.

## Executed results

The active matcher combines two fitted Histogram Gradient Boosting variants with a distance prior, a U-turn penalty, and a decoder that tolerates GPS jitter. Training-trip validation selected a three-sample smoothing half-window and a 15 m same-arc projection tolerance. The frozen run matched **18,016 of 19,160 student-test observations (94.03%)** under full-route labels, with ordered route LCS F1 of **0.9795** and strict continuity of **96.23%**. Its earlier-label Top-1 was **72.83%**. See [training selection and student evidence](results/gps-noise-v1/).

The preceding comparison introduced the full-route labels and the boosting blend. Each row below uses the same 19,160 student-test observations.

| Pipeline | Full-route label Top-1 | Earlier label Top-1 | Ordered-sequence LCS F1 | Directed continuity |
|---|---:|---:|---:|---:|
| Previous contextual boosting + decoder | 78.41% | 80.56% | 0.8790 | 98.80% |
| Full-route boosting blend + earlier decoder | 86.41% | 74.97% | 0.8867 | 98.27% |
| Active blend + GPS-noise decoder | **94.03%** | 72.83% | **0.9795** | 96.23% |

The full-route method derives reference arcs with a global monotone alignment. The two methods assign different labels to 5,011 observations. All 19,160 points, including 25 whose full-route labels are more than 60 m from the GPS fix, stay in both denominators. The API displays both scores. See the [peer review and paired results](docs/peer_approach.md).

Measured across all ten student-testing trips (19,160 observations), using approximate reference-aligned point labels:

| Method | Exact arc Top-1 | Directed continuity | Ordered-sequence LCS F1 | Candidate misses |
|---|---:|---:|---:|---:|
| Nearest road | 42.96% | 75.58% | 0.2984 | 4.65% |
| Tuned Extra Trees | 71.66% | 92.57% | 0.4765 | 4.65% |
| Tuned Extra Trees + HMM | 76.13% | 99.87% | 0.7946 | 4.65% |
| Refined boosting classifier | 73.71% | — | — | 1.57% |
| **Previous refined boosting + decoder** | **80.56%** | **98.80%** | **0.8790** | **1.57%** |

The previous refinement correctly matched **15,436 of 19,160** observations using the earlier label method, up from 14,587 with the original Extra Trees HMM (**+4.43 percentage points**). Each of the ten trips improved. Candidate misses fell from 890 to 300 out of 19,160 observations. Directed continuity fell from 19,124 to 18,920 successes among 19,149 eligible adjacent pairs. Both runs excluded the same one reported time gap from continuity. Ordered-sequence LCS F1 rose from 0.7946 to 0.8790. See [`student_test.json`](results/context-boost-v1/student_test.json) and [`hmm_comparison.json`](results/hmm_comparison.json).

The original comparison used all 80 training trips and three grouped folds across five algorithms. Its eight TPE trials (three startup, five guided) improved Extra Trees mean CV Top-1 from 58.84% to 59.53%. Student-test Top-1 improved from 70.61% to 71.66%, while binary log loss increased from 0.2081 to 0.2097. The original model, predictions and comparisons are retained.

The refinement reused Histogram Gradient Boosting with 20 candidates and 28 GPS and road geometry features. It compared standard and confidence-weighted variants on the same three training folds. Six TPE trials (three startup, three guided) raised the best mean grouped-fold Top-1 from 62.24% to 62.61%. These scores guided model and hyperparameter selection on reused folds. The selected classifier was refit on all 80 training trips. Decoder settings were selected at 79.07% pooled exact Top-1 on 30,651 observations from 16 training trips using out-of-fold classifier predictions. The refinements were frozen before this run scored the ten student-test trips against the original labels. The earlier boosting model's 90.78% binary candidate-row accuracy includes true negatives. Its comparable exact arc Top-1 was 69.35%.

With the two heading reliability features, the selected classifier scored 59.53% mean grouped-CV Top-1 and 71.66% student-test Top-1. Removing them gave 57.19% and 68.25%, respectively. The reference-label alignment also uses heading. Run `python -m scripts.ablation` after tuning to reproduce this comparison.

All 52 behavior and API tests passed with the updated artifact, including a regression case for slow GPS jitter on a one-way road. The earlier release's HTTP evidence is under `results/route-context-v1/`; current student verification and screenshots are under `results/gps-noise-v1/`. The physical two-laptop demonstration and receiving-terminal screenshot require team evidence.

## Setup

Use Python 3.12 or 3.13 for the pinned dependencies. The saved models and results were produced with Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-executed.txt
python -m scripts.import_data --zip "C:\path\to\Student folder.zip" --destination data
$env:PATHSNAP_DATA = (Resolve-Path data).Path
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
```

The input directory must contain `train/<trip>/<trip>.{track,nodes,arcs,route}` and `student_test/...`. The importer only extracts these permitted development partitions. Raw data, feature caches, secrets and evaluator files are excluded from Git.

`requirements-executed.txt` pins the versions used for the saved model artifacts. `requirements.txt` gives broader development ranges; if you change the scikit-learn version, retrain rather than assuming serialized estimators remain compatible.

## Reproduce the experiments

```powershell
python -m scripts.eda
python -m scripts.prepare --smoke
python -m scripts.prepare
python -m scripts.train --smoke
python -m scripts.train
python -m scripts.tune --trials 8
python -m scripts.evaluate
python -m scripts.eda --map-only
python -m unittest discover -s tests -v
```

Preparation caches features once per trip, verifies source/configuration/code fingerprints and keeps labels separate from prediction features. The exactly five algorithms requested are Gaussian Naive Bayes, Logistic Regression, Decision Tree, Extra Trees and HistGradientBoosting. Three-fold GroupKFold keeps whole trips together. Preprocessing is fit within each training fold. Training-only negative subsampling preserves all positives, the two closest candidates and a seeded 12% sample of other negatives; validation and student-testing candidate sets remain complete.

Model selection uses mean grouped-CV observation Top-1 accuracy. TPE tuning uses three startup trials followed by guided trials, with a persistent SQLite study. Tuned and untuned models are both retained; the final artifact is selected using grouped CV, not evaluator outcomes. `results/execution_manifest.json` records runtime versions, seed, source commit and trip IDs.

After the original features and folds are prepared, reproduce the accuracy refinement in order:

```powershell
python -m scripts.refine_accuracy prepare
python -m scripts.refine_accuracy validate --threads 2
python -m scripts.refine_accuracy tune --threads 2 --trials 6
python -m scripts.refine_accuracy refit --threads 2
python -m scripts.refine_accuracy calibrate --holdout-trips 16 --threads 2
python -m scripts.refine_accuracy test --threads 2
```

The tuning stage reuses the recorded SQLite study in `results/context-boost-v1/`. Calibration selects decoder settings on the same 16 deterministically sampled training trips and restores any missing out-of-fold predictions by refitting the corresponding training folds. The `--threads` option controls model fitting and scoring. Run `python -m scripts.api_check` with the active service running to compare live and offline results.

After the earlier refinement commands above are complete, run the paired alignment experiment and select the final decoder using only training trips.

```powershell
python -m scripts.compare_route_alignment
python -m scripts.validate_route_decoder select
python -m scripts.validate_route_decoder benchmark
```

To reproduce training-only decoder noise selection and freeze the selected settings before student evaluation:

```powershell
python -m scripts.validate_gps_noise
python -m scripts.freeze_gps_noise
```

The selection includes all points from sixteen training trips, split between slow GPS motion and a deterministic sample of other trips. It uses out-of-fold booster scores. The existing classifiers remain fitted on the original 80 training trips. Reserved evaluator files are used by the runtime evaluation service.

## Run Laptop 1 (backend)

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --log-level info
```

Open `http://127.0.0.1:8000` for a same-machine check. The UI and API may run from this one process, but the required final deployment uses a second laptop. `GET /health`, `GET /trips` and `POST /predict` perform real inference. A request body is `{"trip_id":"00000000"}` using an ID actually returned by `/trips`.

`artifacts/active_model.json` selects `artifacts/gps_noise_boost.joblib` and its frozen companion settings, which use `artifacts/context_boost.joblib` for the second set of candidate scores. To reproduce the preceding decoder, set `$env:PATHSNAP_MODEL = (Resolve-Path artifacts/ordered_route_boost.joblib).Path` before starting Uvicorn. Use `artifacts/context_boost.joblib` for the earlier contextual model and `artifacts/final_model.joblib` for the original Extra Trees service.

## Run Laptop 2 (UI)

Copy the `frontend` folder to Laptop 2, then:

```powershell
python -m http.server 8800 --directory frontend
```

Open `http://localhost:8800`, enter `http://<Laptop-1-LAN-IP>:8000` in the server field and connect. Select a trip and reconstruct it. Both laptops must share a reachable network; permit the backend port if the operating system prompts. See [deployment details](docs/deployment.md). No cloud service, authentication or database is needed for inference.

## Reference isolation and evaluator trips

`pathsnap/predict.py`, geometry, features, scoring and decoder do not read `.route` files. Only after inference completes does `backend/evaluator.py` load server-side reference information to compute aggregate metrics. The HTTP response never contains reference routes or per-observation reference labels.

When the organizers supply the reserved runtime data, set `PATHSNAP_DATA` to its parent and `PATHSNAP_PARTITION=evaluator_test` (or `evaluator_testing`). Trip IDs are discovered from files; none are guessed. If references are stored separately, use `PATHSNAP_REFERENCE_ROOT` with the same partition/trip layout. Do not use reserved files for training, feature choices, tuning or development evaluation. Restart the server after changing environment variables or final artifacts.

## Measurements and limitations

- `.route` is an ordered arc sequence, not a point-label file. Training/evaluation point labels are derived with monotone sparse alignment preserving repeated route occurrences. Alignment uses distance and a small reliable-heading term, so pointwise evaluation shares modeling assumptions with the features. Failed alignment uses a logged nearest-reference fallback; ambiguous and poor fits are recorded. Sequence comparison is complementary evidence.
- Candidate generation keeps up to 20 arcs for the active model and ten for the original model, with a 250 m initial search and a 300 m retry for empty queries. Refined features use GPS and road geometry. Training confidence weights use the known training labels and alignment diagnostics only during fitting.
- Most observations are 1 Hz, but large gaps and disconnected roads exist. The decoder uses explicit restarts and reconstructs intermediate arcs only when directed routing supports them.
- Motion heading uses a symmetric four-sample span, so this is offline route reconstruction. A score margin is an ambiguity indicator, not calibrated probability that the route is correct.
- Candidate F1 and log loss score candidate rows. Observation Top-1 divides correct exact directed-arc predictions by all GPS points, including candidate misses and missing predictions. Candidate misses count observations whose aligned reference arc is absent. Continuity divides feasible transitions by all eligible adjacent pairs, including missing or unreachable predictions, and reports excluded exogenous time gaps. Sequence LCS F1 divides twice the total matched subsequence length by the total predicted and reference arc counts. See `results/metric_definitions.json`.
- The refined comparison uses frozen, approximate student-test labels. The student-test partition had already been examined for the original run. The measured gain describes these ten trips, and reserved evaluator performance awaits the organizer-provided data.
- Classifier timings in model comparison exclude feature preparation and graph loading. The API separately measures the full inference pipeline and reference evaluation.
- Same-machine HTTP checks and browser screenshots do not establish a second-laptop LAN demonstration. The latter and receiving-terminal screenshots remain pending until performed on the team's devices.

## Artifacts

The course submission consists of this public repository and the [consolidated PDF](results/PathSnap_Technical_Report.pdf). The [deliverables checklist](docs/deliverables.md) maps each required section to its evidence.

To regenerate the paired boosting metrics, active-model decoder comparison, interactive map and PDF after reproducing the experiments:

```powershell
python -m scripts.deliverable_metrics tuning
python -m scripts.deliverable_metrics decoder
python -m scripts.eda --active-map 00000052 --screenshot
python -m scripts.report --render
```

`artifacts/` holds all five original fitted models, tuned Extra Trees, and the active refined boosting model with its decoder configuration. `results/` holds grouped-fold records, predictions for all ten permitted student trips, optimization trials, comparison tables, plots and the consolidated PDF. Refinement evidence is under `results/context-boost-v1/`. `results/eda/representative_map.html` shows the original HMM example. `STATUS.md` records completed work and pending evidence.

Optional screenshot verification uses `python -m pip install playwright`, `python -m playwright install chromium`, then `python -m scripts.capture_ui` against the running server. `python -m scripts.report` creates the consolidated PDF from completed result files. Never manufacture missing screenshots, losses or results.

For screenshot capture and PDF rendering together, install `requirements-dev.txt`, install Chromium as above, and use `python -m scripts.report --render`. Rendered review pages are local QA files and are excluded from the submission ZIP.

## Research concepts

The [current repository review](docs/current_repo_review.md) records the 18 November 2026 check of map-matching and boosting projects, ideas already implemented, and the requirements for a future ranking experiment.

The custom implementation is informed by [Newson and Krumm (2009)](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/), [FMM](https://github.com/cyang-kth/fmm), [LeuvenMapMatching](https://github.com/wannesm/LeuvenMapMatching), and [heading feature-engineering research](https://arxiv.org/abs/1409.0797). No existing matcher or another team's implementation was copied. See [decoder details](docs/decoder.md), [model notes](docs/model_notes.md) and [viva preparation](docs/VIVA.md).
