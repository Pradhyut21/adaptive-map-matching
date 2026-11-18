# Official submission checklist

The course requires two submissions: the public GitHub repository and one consolidated PDF. This checklist maps the course requirements to evidence and records the items that need the team's devices or organizer data.

| GitHub requirement | Repository evidence |
| --- | --- |
| Public source and saved model artifacts | [Public repository](https://github.com/Pradhyut21/adaptive-map-matching), [trained artifacts](../artifacts/), [installation and run instructions](../README.md) |
| Data processing and feature engineering | [Data import](../scripts/import_data.py), [EDA](../scripts/eda.py), [candidate construction](../pathsnap/candidate_refinement.py), [route context](../pathsnap/route_context.py), [label alignment](../pathsnap/route_alignment.py) |
| Training and evaluation for five models | [Training entry point](../scripts/train.py), [results table](../results/model_comparison.csv), [metrics and confusion matrices](../results/model_comparison.json), [fold records](../results/cv/) |
| Bayesian optimization | [Extra Trees study](../results/tuning_summary.json), [boosting study](../results/context-boost-v1/boost_tuning.json), [paired boosting metrics](../results/context-boost-v1/boost_tuning_comparison.csv) |
| Route reconstruction | [Viterbi and routing](../pathsnap/decoder.py), [refined decoding](../pathsnap/decoder_refinement.py), [final model comparison](../results/route-context-v1/classifier_decoder_comparison.csv) |
| API and UI | [FastAPI backend](../backend/main.py), [server-side evaluator](../backend/evaluator.py), [browser UI](../frontend/), [ten live requests](../results/route-context-v1/api_integration.json), [test log](../results/route-context-v1/unit_tests.txt) |
| Deployment | [Pinned dependencies](../requirements-executed.txt), [active artifact configuration](../artifacts/active_model.json), [two-device runbook](deployment.md) |
| Visualizations and evidence | [Folium generator](../scripts/eda.py), [interactive map](../results/route-context-v1/map/representative_map.html), [plots](../results/figures/), [experiment logs](../results/route-context-v1/execution_logs/) |

| Required PDF section | Included evidence |
| --- | --- |
| 2.2 Dataset and EDA | Partition sizes, data checks, preprocessing, features, missing-value analysis, GPS trajectories and roads |
| 3.1 Overall comparison | Official table columns with measured results for all five algorithms |
| 3.2 Questions for every model | Individual subsections with configuration, loss/objective, measured student results and held-out performance |
| 3.3 Visualizations | Grouped CV Macro F1, student Top-1 and Macro F1, log loss and diagnostic confusion matrices. The recorded fits did not generate epoch histories. |
| 4 Bayesian optimization | Search spaces, trials, winning parameters, CV Macro F1, student Top-1 and Macro F1, log loss and inference time for paired untuned and tuned runs |
| 5 HMM and Viterbi | States, emissions, transitions, gaps, and classifier-only versus decoded results on all ten student trips |
| 6 Folium | Actual GPS, reference, road graph and predicted route, with a legend, layer controls, and rendered view from the active model |
| 7 Deployment | Working UI screenshots and successful HTTP and receiving-log evidence. **Receiving CMD screenshot and the physical run remain pending.** |

The [consolidated PDF](../results/PathSnap_Technical_Report.pdf) is generated from source records. The [report manifest](../results/report_manifest.json) identifies them. The team should add its own interpretation and provide the required disclosure. The ten reserved evaluator trips must come from the organizers. [Demo and screenshot steps](deployment.md#actual-demonstration-evidence)
