# Model interpretation guide

This is an explanation of the implemented model families, not a claim of measured superiority. Actual configurations, metrics and selection live in `results/models/`, `results/cv/`, and `results/selection.json`.

| Model | Why compare it? | Objective or fit criterion | Main limitation to check in measured evidence |
|---|---|---|---|
| Gaussian Naive Bayes | Cheap probabilistic baseline | Estimate Gaussian feature densities and class priors | Correlated/non-Gaussian inputs can make probabilities overconfident |
| Logistic Regression | Compact linear candidate scorer | Regularized binary logistic log loss after training-fold scaling | Can miss nonlinear feature interactions |
| Decision Tree | Nonlinear conditional candidate rules | Greedy impurity reduction using recorded criterion | Single-tree overfitting and coarse probabilities |
| Extra Trees | Average randomized nonlinear trees | Recorded impurity criterion per tree; average class probabilities | Artifact size, scoring cost and probability quality |
| HistGradientBoosting | Efficient nonlinear tabular boosting | Binary log loss with additive histogram trees | Leaf/iteration settings can fit noise in derived labels |

Compare training versus trip-held-out validation results, fold variation, candidate Macro F1 and log loss, observation Top-1, and inference time. A positive train-minus-validation gap is evidence of weaker held-out performance, not proof of its cause. Do not select from training accuracy alone.

The candidate label is binary: whether that generated arc equals the approximate reference-aligned arc at the observation. Pointwise labels are derived from the ordered route; a route row is not a GPS row. Preserve repeated route occurrences during alignment.

Every candidate from a trip stays with the same validation group. Negative subsampling occurs only in fitting folds. Complete validation and test candidate sets remain intact, and missing reference candidates remain failures in observation-level accuracy.

The final decoder consumes the selected classifier's positive scores as emission potentials. These discriminative scores are not calibrated final-route confidence. Normalizing over candidates does not establish calibration.

The report generator reads actual artifacts and does not fabricate epoch-loss curves. Team members must review measured outcomes and add their own interpretation.
