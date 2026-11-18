# Viva preparation

Read the actual source and experiment records before using these prompts. This is a technical explanation guide; it is not evidence that a team member independently performed an experiment.

The current artifact is `gps_noise_boost.joblib`. It uses two Histogram Gradient Boosting scorers with directed Viterbi decoding, three-sample GPS smoothing and a 15 m projection-jitter tolerance. Student Top-1 is **94.03%** (18,016/19,160), with ordered LCS F1 **0.9795** and strict continuity **96.23%**. Final evaluator reassessment reached **95.51%** (18,790/19,674). The reassessment followed decoder development prompted by an observed failure. [Current comparison](../results/model_comparison.csv), [assessment summary](../results/final_evaluation_summary.json)

1. **What is being predicted?** A directed road arc for each GPS observation, followed by a road sequence with supported intermediate arcs.
2. **Why is nearest-road insufficient?** A nearby arc can have the wrong direction or be disconnected from neighboring choices. Compare the actual nearest-road, classifier-only and HMM results; do not assume the HMM always improves accuracy.
3. **Why project coordinates?** Longitude and latitude are angular. Local azimuthal-equidistant coordinates make point-to-segment geometry meaningful in metres over the trip's extent.
4. **How are labels obtained?** Reference `.route` files list traversed arcs, not observation labels. A separate monotone route-position alignment derives approximate point labels, preserving repeated occurrences and recording ambiguity/fallbacks.
5. **What prevents label leakage?** Candidate generation, features and decoder read only GPS and graph information. Training/evaluation alignment is separate. Evaluator references are read server-side only after prediction and are not sent to the frontend.
6. **Why group by trip?** Adjacent GPS points are highly related. Randomly splitting candidate rows would place observations from the same trajectory in fitting and validation, inflating the estimate.
7. **Why exactly these five algorithms?** They give fast probabilistic, linear, tree, randomized-ensemble and boosting comparisons. Their selection is supported by measured grouped validation, not a fixed assumption that one model wins.
8. **Why motion reliability?** Very small displacements make GPS-derived heading unstable. Heading features and a motion-reliability signal allow the classifier to condition on this uncertainty. Any claimed benefit requires an actual ablation.
9. **What does Viterbi optimize?** The sum of log emission potentials and transition scores over a candidate-state sequence. Dynamic programming keeps the best predecessor for each current state; backtracking recovers the sequence.
10. **What is network distance?** Remaining source-arc length plus a directed shortest path plus destination-arc prefix. Forward same-arc movement is handled directly. The current decoder treats bounded backward projection jitter as a dwell. Larger backward shifts require a feasible directed path.
11. **Why bounded Dijkstra and caching?** Repeated independent shortest-path searches across every candidate pair are expensive. Reuse sparse bounded source searches, limit cache size, and keep caches tied to one graph identity.
12. **How are gaps handled?** Missing candidates, invalid times, long gaps or unreachable transitions trigger explicit warnings/restarts. The UI must not draw a confident connection that inference did not support.
13. **What does Top-1 mean?** Correct nonmissing predicted directed arcs divided by every observation. Candidate misses and decoder failures remain in the denominator.
14. **Why report candidate miss rate?** A classifier cannot choose a reference road absent from its candidate list. This separates candidate generation failures from scoring errors.
15. **Why Macro F1 and log loss?** Candidate negatives usually outnumber positives. Macro F1 shows both classes; log loss evaluates score quality. Neither replaces observation accuracy or route-sequence evaluation.
16. **Is confidence calibrated?** No final-route confidence claim is made. Candidate margins are relative ambiguity indicators.
17. **What is Bayesian tuning evidence?** Inspect executed Optuna trials, startup and guided proposals, grouped objective, best configuration and tuned-versus-untuned comparisons.
18. **What deployment evidence exists?** All ten student HTTP results matched the frozen metrics, all ten evaluator requests returned 200, and a separate LAN client reached the backend. The receiving CMD screenshot and team confirmation of the physical setup remain pending.
19. **Where do numerical report claims come from?** Stored JSON/CSV metrics and prediction arrays, not handwritten values. Report manifests hash the source records.
20. **What assistance was used?** Disclose AI implementation/debugging/report-compilation assistance according to event policy. The team remains responsible for understanding, reproduction and independent interpretation.
