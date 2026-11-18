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

## Stage 2 active execution
- Final v2 cache built for all80training/10student-testing trips.208417/19160GPSobservations retained.
- Candidate miss rate12.0345%train,4.6451%student-test; completevalidation/test candidate sets; no referenceinsertion.
- Realtraining smoke2.26s, firsttrainingtrip only, excluded from selection.
- GaussianNB,LogisticRegression,DecisionTree completed all3groupedfolds and all10studenttrip predictions; ExtraTrees/HGB in progress.
- 31geometry/alignment/decoder/evaluation behavior tests passed;7APIreadiness/security tests passed, genuine finalmodeltestpending.
- First source checkpoints published to authorized publicrepository main.

## Stage 2 completed
- Allfive algorithms fitted/evaluated with3GroupKFold folds acrossall80trainingtrips; all10student-testingpredictions saved permodel.
- ExtraTrees selected solely by mean groupedCVTop1:0.5883772223. Complete comparison inresults/model_comparison.json/csv.
- Actual fullcomparison elapsed513.20seconds. Fullvalidation/test candidates retained; modelartifacts andfoldconfigs saved.
- Stage3A genuineOptunaTPEstarted:8plannedtrials,3startupthen5guided; persistentSQLite study.

## Bayesian optimization progress
- Trial0completed in187.3seconds: mean groupedCVTop1=0.5952758852 versusuntuned0.5883772223. Trial0isstartup sampling,notguidedTPE.
- Eighttrials planned:3startupthen5guided. Estimatedfirst-trialtotal25minutes.
- Allfive fittedmodelartifacts and50studenttrip predictionfiles pushed to main(2582811).
- Lightweight UIservice runningonport8800; APIwillstartwhenfinalmodel/configarecomplete.

## Stage 3A search completed; refit active
- All8 optimization trials executed:3startup and5guidedTPE.
- Best executed meanCVTop1=0.5952758852.
- Final refit on all80trainingtrips in progress. Full ten-trip classifier/HMM comparison andliveHTTPchecks follow.
