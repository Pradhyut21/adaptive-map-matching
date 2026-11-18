# PathSnap Trace-Lite

Independent, CPU-based noisy GPS map matching: learned candidate scoring, directed network transitions, and a custom log-space Viterbi decoder. The live evaluator uses FastAPI on Laptop 1 and a plain HTML/CSS/JavaScript UI on Laptop 2. The UI draws actual supplied road geometry without external tiles or CDN dependencies.

This repository was implemented with AI assistance. Every team member must review the code, reproduce the measurements, interpret the recorded results independently, and make the required disclosure. This README is implementation documentation, not evidence that the team has completed its viva or two-device demonstration.

## Setup (Python 3.11+; executed here on Python 3.13)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.import_data --zip "C:\path\to\Student folder.zip" --destination data
$env:PATHSNAP_DATA = (Resolve-Path data).Path
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
```

The input directory must contain `train/<trip>/<trip>.{track,nodes,arcs,route}` and `student_test/...`. The importer only extracts these permitted development partitions. Raw data, feature caches, secrets and evaluator files are excluded from Git.

## Reproduce the experiments

```powershell
python -m scripts.eda
python -m scripts.prepare --smoke
python -m scripts.prepare
python -m scripts.train --smoke
python -m scripts.train
python -m scripts.tune --trials 8
python -m scripts.evaluate
python -m unittest discover -s tests -v
```

Preparation caches features once per trip, verifies source/configuration/code fingerprints and keeps labels separate from prediction features. The exactly five algorithms requested are Gaussian Naive Bayes, Logistic Regression, Decision Tree, Extra Trees and HistGradientBoosting. Three-fold GroupKFold keeps whole trips together. Preprocessing is fit within each training fold. Training-only negative subsampling preserves all positives, the two closest candidates and a seeded 12% sample of other negatives; validation and student-testing candidate sets remain complete.

Model selection uses mean grouped-CV observation Top-1 accuracy. TPE tuning uses three startup trials followed by guided trials, with a persistent SQLite study. Tuned and untuned models are both retained; the final artifact is selected using grouped CV, not evaluator outcomes. `results/execution_manifest.json` records runtime versions, seed, source commit and trip IDs.

## Run Laptop 1 (backend)

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --log-level info
```

Open `http://127.0.0.1:8000` for a same-machine check. The UI and API may run from this one process, but the required final deployment uses a second laptop. `GET /health`, `GET /trips` and `POST /predict` perform real inference. A request body is `{"trip_id":"00000000"}` using an ID actually returned by `/trips`.

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

- `.route` is an ordered arc sequence, not a point-label file. Training/evaluation point labels are derived with monotone sparse alignment preserving repeated route occurrences. Failed alignment uses a logged nearest-reference fallback; ambiguous and poor fits are recorded.
- Candidate generation uses at most ten arcs within 250 m (empty queries may expand to 300 m). The radius was increased after a training-only coverage experiment; remaining candidate misses stay Top-1 failures.
- Most observations are 1 Hz, but large gaps and disconnected roads exist. The decoder uses explicit restarts and reconstructs intermediate arcs only when directed routing supports them.
- Motion heading uses a symmetric four-sample span, so this is offline route reconstruction. A score margin is an ambiguity indicator, not calibrated probability that the route is correct.
- Candidate F1 and log loss score candidate rows; observation Top-1 scores all GPS points. Route continuity counts unreachable/missing transitions as failures and reports excluded exogenous time gaps. Exact ordered LCS F1 compares reconstructed arc sequences. See `results/metric_definitions.json`; these diagnostics are not an invented official scoring formula.
- Classifier timings in model comparison exclude feature preparation and graph loading. The API separately measures the full inference pipeline and reference evaluation.
- Same-machine HTTP checks and browser screenshots do not establish a second-laptop LAN demonstration. The latter and receiving-terminal screenshots remain pending until performed on the team's devices.

## Artifacts

`artifacts/` holds all five fitted models, the tuned candidate, final model and configuration. `results/` holds grouped-fold records, predictions for all ten permitted student trips, optimization trials, comparison tables, plots and the consolidated PDF. `results/eda/representative_map.html` is the actual Folium map. `STATUS.md` records real progress and outstanding evidence.

Optional screenshot verification uses `python -m pip install playwright`, `python -m playwright install chromium`, then `python -m scripts.capture_ui` against the running server. `python -m scripts.report` creates the consolidated PDF from completed result files. Never manufacture missing screenshots, losses or results.

## Research concepts

The custom implementation is informed by [Newson and Krumm (2009)](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/), [FMM](https://github.com/cyang-kth/fmm), [LeuvenMapMatching](https://github.com/wannesm/LeuvenMapMatching), and [heading feature-engineering research](https://arxiv.org/abs/1409.0797). No existing matcher or another team's implementation was copied. See [decoder details](docs/decoder.md), [model notes](docs/model_notes.md) and [viva preparation](docs/VIVA.md).
