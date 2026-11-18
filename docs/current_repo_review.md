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

## Measured result and next check

The tuned contextual boosting classifier averaged **62.61% exact Top-1** over the three grouped folds of the 80 training trips. Its validation folds scored 48.34%, 69.42% and 70.06%. The frozen full pipeline with directed decoding reached **80.56% exact Top-1**, or **15,436 of 19,160 GPS observations across all ten student-test trips**. These counts use the original derived arc labels. Candidate misses account for 300 of those observations and remain in the denominator. The grouped classifier evaluation and the end-to-end student test measure different stages on different trips. [Boosting tuning record](../results/context-boost-v1/boost_tuning.json) and [Student-test results](../results/context-boost-v1/student_test.json)

Keep the deployed recipe frozen. The next low-risk experiment is a one-change ablation on training trips using the existing features, candidate sets and decoder. Set an accuracy and latency acceptance rule before running it. Keep a fresh whole-trip or whole-region holdout for transfer if available. A reported 90% score is comparable only when it uses exact directed-arc Top-1 with the same data split, label construction and all-observation denominator. The repository review supplies no measured 90% result for PathSnap.
