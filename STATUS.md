# PathSnap execution status

Started 18 November 2026, approximately 10:30 IST. Target: 13:30 IST.

## Completed
- Read the five official guideline PDFs and inspected the problem video.
- Downloaded permitted Student folder.zip; verified 80 training and 10 student-testing trips.
- Created isolated Git repository on codex/pathsnap-trace-lite. Authorized remote: Pradhyut21/adaptive-map-matching (empty public repository verified).

## Active
- Stage 1: extraction, geometry, spatial candidates, EDA and nearest-road baseline.
- Stage 2: derived reference label alignment, feature cache, five-model grouped comparison.

## Remaining
- Bayesian optimization; custom decoder; actual ten-trip comparisons.
- FastAPI, UI, integration logs, Folium and measured graphs, consolidated PDF.
- Local stage commits and publish to authorized repository.

## Constraints and pending evidence
- No evaluator data is available or used.
- Pointwise labels will be derived from ordered routes and must be described as approximate.
- Actual second-laptop LAN demonstration requires the team's second device and remains pending.
- AI implementation assistance must be disclosed; team review and independent interpretation remain required.
- No experiments have been claimed before execution.

## Stage 1 execution evidence
- `results/eda/inventory.csv`: all90 permitted trips validated; EDA plots and actual Folium map saved.
- Initial candidate smoke: train00000001 1315observations/2.33s/miss0.152%; student00000000 2503observations/2.36s/miss3.755% (100m radius).
- Training-only candidate radius experiment on noisy trip00000054: K10 miss84.658% at100m,54.404% at250m. Radius250m selected for v2; remaining omissions reported, never reference-injected.
- Custom directed decoder:19behavioral tests passed; actual2503point distance-potential smoke1.32s. This was a decoder benchmark, not trained-model accuracy.
- EDA: finite rows/valid IDs, disconnected graph components,13train/1student nonadjacent reference steps; derived labels and explicit restarts required.
