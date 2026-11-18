# Model interpretation guide

The deployed system combines two Histogram Gradient Boosting scorers and a GPS-noise-aware directed Viterbi decoder. It matched 18,016 of 19,160 student points (94.03%) and 18,790 of 19,674 evaluator points (95.51%). Current and baseline rows are identified in the [comparison CSV](../results/model_comparison.csv). The original `results/selection.json` records the earlier Extra Trees selection.

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

The final decoder consumes the equal geometric blend of the two boosters' positive scores as emission potentials, with a GPS-distance prior and directed transition costs. Its three-sample smoothing half-window and 15 m same-arc jitter tolerance were selected on OOF scores from sixteen training trips. The selected subset scored 67.51% exact Top-1, up from 48.49%. This subset is intentionally rich in slow GPS motion.

The full-route booster alone scored 91.88% on the student set. The classifier blend scored 89.52%, and the current decoded pipeline scored 94.03%. Current ordered route LCS F1 is 0.9795. Earlier-label Top-1 is 72.83%, and strict raw-projection continuity is 96.23%. [Current pipeline results](../results/gps-noise-v1/student_test.json), [classifier diagnostics](../results/gps-noise-v1/classifier_decoder_comparison.json)

| Scorer with full-route labels | Fitted training Top-1 | Pooled trip-grouped CV Top-1 | Student Top-1 |
| --- | ---: | ---: | ---: |
| Full-route boosting component | 86.01% | 80.84% | 91.88% |
| Equal boosting blend, before decoding | 85.75% | 80.09% | 89.52% |

The fitted training scores include all 208,417 training observations. Each CV score uses predictions from a model excluding that trip. The current pipeline's sixteen-trip decoder selection score is a separate measurement. [Component measurements](../results/final_model_summary.json)

The six-trial contextual boosting TPE study raised mean CV Top-1 from 62.24% to 62.61% and student Top-1 from 73.40% to 73.71%, using the earlier labels. Its fitted training accuracy rose from 71.34% to 73.50%. These tuning figures describe the contextual component. The deployed pipeline adds the full-route component and decoding. [Matched tuning comparison](../results/context-boost-v1/boost_tuning_comparison.csv)

The report generator reads actual artifacts and does not fabricate epoch-loss curves. Team members must review measured outcomes and add their own interpretation.
