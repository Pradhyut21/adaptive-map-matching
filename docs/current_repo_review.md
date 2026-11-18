# Map matching and boosting repository review

Checked the public GitHub repositories and their project documentation on 18 November 2026. The check covers the current default branches and published documentation.

## Ideas from map matching projects

| Project | Approach in the source | Application to PathSnap |
| --- | --- | --- |
| FMM | Bounded routing lookup and configurable candidate count, search radius and GPS error. | PathSnap searches at most 20 nearby arcs and caches directed shortest-path searches within a distance bound. A future experiment can measure recall and latency when expanding the shortlist only at uncertain observations. [FMM repository](https://github.com/cyang-kth/fmm) |
| LeuvenMapMatching | HMM states can traverse an arc between GPS readings without requiring a measurement on it. The emissions and transitions can be customized. | PathSnap reconstructs intermediate directed arcs and marks gaps. A future outlier state could handle isolated corrupt fixes. The skipped fixes would still count in the observation-level denominator. [LeuvenMapMatching repository](https://github.com/wannesm/LeuvenMapMatching) |
| Valhalla Meili | Candidate road states are linked using routed transitions. Its lazy path search reduces the number of expensive route calculations. | PathSnap already combines learned candidate scores with directed route feasibility. It also reassigns observations along the predicted route while retaining valid candidate IDs. A more selective route search is principally a latency experiment. [Valhalla repository](https://github.com/valhalla/valhalla) and [Meili algorithms](https://valhalla.github.io/valhalla/contributing/architecture/meili/algorithms/) |

## Learning to rank as a future experiment

XGBoost `rank:ndcg` implements LambdaMART over candidates grouped by query. LightGBM provides `LGBMRanker`, and CatBoost supports grouped losses including `PairLogit` and `QuerySoftMax`. This is a plausible way to learn which of a GPS fix's candidates belongs at the top. The current submission retains the five required algorithms and uses the validated histogram gradient boosting recipe. A ranking comparison is deferred to a future benchmark. [XGBoost learning to rank](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html), [LightGBM ranker](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html), [CatBoost ranking objectives](https://catboost.ai/docs/en/concepts/loss-functions-ranking)

- A ranking query contains the candidate arcs for **one GPS observation**. A cross-validation group contains **one entire trip**, including every observation and candidate row. Keep the IDs out of model features.
- Use whole training trips for inner validation and early stopping. A random split of GPS rows can place neighboring observations from the same trip on both sides of the split. Use a route or geographic holdout to test transfer if trips share much of the road network. The existing trip split leaves geographic overlap and new-region performance unverified.
- Keep candidate misses in the exact directed-arc Top-1 score, including queries with no positive candidate. A ranker returns relevance scores. Use those as relative emission scores and choose their scale on inner training folds before decoding. Score the frozen pipeline with Top-1, route sequence LCS F1, per-trip spread and latency.

## Recorded experiments and current deployment

The contextual boosting component averaged 62.61% exact Top-1 over the original three grouped folds using the earlier labels. That component's decoder reached 80.56% on the student set during an earlier release. The current full-route classifier and GPS-noise decoder reached **94.03%** (18,016/19,160) student Top-1, with ordered LCS F1 **0.9795**. Final evaluator reassessment reached **95.51%** (18,790/19,674). [Current comparison](../results/model_comparison.csv), [current student results](../results/gps-noise-v1/student_test.json), [assessment summary](../results/final_evaluation_summary.json)

The six-trial TPE study and later decoder settings used training OOF predictions for selection. The noise revision followed an observed evaluator failure, so its final evaluator result is a repeated assessment after feedback. Comparisons should keep the metric, label construction and all-point denominator explicit. The current CSV records each row's scope.
