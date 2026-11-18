"""Generate the consolidated technical PDF from recorded experiment artifacts.

Run after training, tuning, decoding, and API verification:
    python -m scripts.report
This script never trains a model and never fills missing results with numbers.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import hashlib
import json
from html import escape
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from pathsnap.settings import ROOT
from pathsnap.metrics import METRIC_DEFINITIONS

RESULTS=ROOT/"results"
FIGURES=RESULTS/"figures"
MODEL_NOTES={
"GaussianNB":{
"family":"Probabilistic",
"suitability":"Fast Gaussian class-conditional density estimates provide a low-cost baseline for distance and motion features.",
"strengths":"Very small model and fast fitting/prediction; useful for testing whether simple probabilistic assumptions suffice.",
"limitations":"Candidate geometry features are correlated and not Gaussian. Conditional independence can produce overconfident scores.",
"objective":"Estimate per-class Gaussian feature means and variances plus class priors; select using posterior class scores. No epoch-wise gradient loss history."},
"LogisticRegression":{
"family":"Linear probabilistic",
"suitability":"Tests whether a linear decision boundary in the engineered candidate features generalizes across trips.",
"strengths":"Compact, interpretable coefficients and direct positive-class probabilities; scaling is fitted inside training folds.",
"limitations":"A linear boundary can miss interactions between heading reliability, distance, speed and segment geometry.",
"objective":"Regularized binary logistic log loss; the pipeline scales features using training-fold statistics."},
"DecisionTree":{
"family":"Tree",
"suitability":"Tests nonlinear threshold interactions in candidate distance, heading disagreement and motion reliability.",
"strengths":"Readable conditional rules and little need for feature scaling.",
"limitations":"An individual tree can overfit route-specific patterns and give coarse or extreme probability estimates.",
"objective":"Greedy impurity reduction (the recorded criterion controls Gini or entropy); constrained tree size controls complexity."},
"ExtraTrees":{
"family":"Randomized tree ensemble",
"suitability":"Captures nonlinear candidate-feature interactions while averaging randomized trees.",
"strengths":"Handles varied feature scales and interactions; averaging can stabilize a single tree's decisions.",
"limitations":"Larger artifacts and more inference work than linear models; probability quality still needs measurement.",
"objective":"Each randomized decision tree reduces recorded impurity; predictions average tree class probabilities."},
"HistGradientBoosting":{
"family":"Boosting",
"suitability":"Tests sequential nonlinear corrections with histogram-binned tabular features.",
"strengths":"Efficient nonlinear fitting on many candidate rows and explicit regularization controls.",
"limitations":"Depth, leaf size and iteration count can overfit derived labels; raw scores are not calibrated route probabilities.",
"objective":"Binary log loss optimized with additive histogram gradient-boosted trees; no fabricated epoch-history plots are used."}}

def load_json(path, required=True):
    path=Path(path)
    if not path.exists():
        if required:raise FileNotFoundError(f"Required experiment artifact missing: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def number(value, digits=4):
    if value is None:return "N/A"
    if isinstance(value,(float,np.floating)):
        return f"{value:.{digits}f}" if np.isfinite(value) else "N/A"
    return str(value)

def metric(m, *keys):
    for key in keys:
        if key in m:return m[key]
    return None

def model_records():
    data=load_json(RESULTS/"model_comparison.json")
    models=data["models"] if isinstance(data,dict) else data
    if len(models)!=5 or not data.get("complete",len(models)==5):
        raise ValueError("Exactly five completed model records required; incomplete records cannot become a final report")
    return data,models

def make_comparison_plots(models,tuning,hmm):
    FIGURES.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold"})
    names=[m["name"] for m in models]
    short=[s.replace("LogisticRegression","Logistic\nRegression").replace("HistGradientBoosting","Histogram\nBoosting").replace("GaussianNB","Gaussian\nNB").replace("DecisionTree","Decision\nTree").replace("ExtraTrees","Extra\nTrees") for s in names]
    fig,ax=plt.subplots(figsize=(10,4))
    values=[m["cv"]["metrics"]["macro_f1"] for m in models]
    bars=ax.bar(short,values,color="#087f8c",width=.6)
    for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+.006,f"{v:.3f}",ha="center",fontsize=9)
    ax.set(ylim=(0,1.05),ylabel="Candidate Macro F1",title="Three-fold trip-grouped cross-validation")
    ax.grid(axis="y",alpha=.15)
    fig.savefig(FIGURES/"cv_macro_f1.png",dpi=170,bbox_inches="tight");plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,key,title in [(axs[0],"top1_accuracy","Student-test observation Top-1"),(axs[1],"macro_f1","Student-test candidate Macro F1")]:
        values=[m["student_test"][key] for m in models]
        bars=ax.bar(short,values,color="#087f8c",width=.6)
        for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+.006,f"{v:.3f}",ha="center",fontsize=8)
        ax.set(ylim=(0,1.05),title=title);ax.tick_params(axis="x",labelsize=8);ax.grid(axis="y",alpha=.15)
    fig.savefig(FIGURES/"student_model_comparison.png",dpi=170,bbox_inches="tight");plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    axs[0].bar(short,[m["student_test"]["log_loss"] for m in models],color="#cb6c36")
    axs[0].set(ylabel="Candidate binary log loss (lower is better)",title="Probability quality on full student candidate sets")
    axs[1].bar(short,[m["inference_seconds"] for m in models],color="#087f8c")
    axs[1].set(ylabel="Seconds",title="Classifier scoring time from cached features")
    for ax in axs:ax.tick_params(axis="x",labelsize=8);ax.grid(axis="y",alpha=.15)
    fig.savefig(FIGURES/"log_loss_and_scoring_time.png",dpi=170,bbox_inches="tight");plt.close(fig)
    # Confusion matrices come only from recorded candidate predictions.
    fig,axs=plt.subplots(1,5,figsize=(14,3))
    for ax,m in zip(axs,models):
        matrix=np.asarray(m["student_test"]["confusion_matrix"])
        ax.imshow(np.log1p(matrix),cmap="Blues")
        for (r,c),v in np.ndenumerate(matrix):ax.text(c,r,str(v),ha="center",va="center",fontsize=8,color="black")
        ax.set(xticks=[0,1],yticks=[0,1],xlabel="Predicted",title=m["name"].replace("Gradient","\nGradient"));ax.tick_params(labelsize=8)
    axs[0].set_ylabel("True candidate class")
    fig.suptitle("Student-test candidate confusion matrices (0 = nonmatch, 1 = match)")
    fig.savefig(FIGURES/"confusion_matrices.png",dpi=170,bbox_inches="tight");plt.close(fig)

class Report:
    def __init__(self):
        self.styles=getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name="BodyCompact",fontName="Helvetica",fontSize=10.2,leading=14,spaceAfter=7,textColor=colors.HexColor("#23343e")))
        self.styles.add(ParagraphStyle(name="SmallText",fontName="Helvetica",fontSize=8,leading=10,spaceAfter=5))
        self.styles.add(ParagraphStyle(name="Cell",fontName="Helvetica",fontSize=7.3,leading=9))
        self.styles.add(ParagraphStyle(name="CodeSmall",fontName="Courier",fontSize=7.2,leading=10,spaceAfter=5))
        self.styles["Heading1"].textColor=colors.HexColor("#075663")
        self.styles["Heading2"].textColor=colors.HexColor("#075663")
        self.story=[]

    def p(self,text,style="BodyCompact"):
        self.story.append(Paragraph(str(text),self.styles[style]))

    def heading(self,text):self.p(escape(text),"Heading1")
    def subheading(self,text):self.p(escape(text),"Heading2")
    def page(self):self.story.append(PageBreak())
    def table(self,rows,widths=None):
        cells=[[Paragraph(escape(str(v)).replace("\n","<br/>"),self.styles["Cell"]) for v in row] for row in rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dceef0")),("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#c2d1d5")),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f8f9")])]))
        self.story.extend([t,Spacer(1,10)])
    def image(self,path,max_width=700,max_height=320):
        p=Path(path)
        if not p.exists():self.p(f"Evidence image pending: {escape(str(p.relative_to(ROOT)))}.");return
        from PIL import Image as PILImage
        with PILImage.open(p) as im:w,h=im.size
        scale=min(max_width/w,max_height/h)
        self.story.extend([Image(str(p),width=w*scale,height=h*scale,hAlign="LEFT"),Spacer(1,7)])

def _footer(canvas,doc):
    canvas.saveState();w,h=landscape(A4)
    canvas.setStrokeColor(colors.HexColor("#d0dde0"));canvas.line(35,30,w-35,30)
    canvas.setFont("Helvetica",7);canvas.setFillColor(colors.HexColor("#536a73"))
    canvas.drawString(35,19,"PathSnap Trace-Lite | Recorded execution evidence | Team review required")
    canvas.drawRightString(w-35,19,f"Page {doc.page}")
    canvas.restoreState()

def build_report():
    comparison,models=model_records()
    eda=load_json(RESULTS/"eda"/"summary.json")
    selection=load_json(RESULTS/"selection.json")
    tuning=load_json(RESULTS/"tuning_summary.json")
    hmm=load_json(RESULTS/"hmm_comparison.json")
    api=load_json(RESULTS/"api_verification.json",False)
    prep=load_json(RESULTS/"prepare_summary.json",False)
    make_comparison_plots(models,tuning,hmm)
    r=Report()
    r.p("PATHSNAP TRACE-LITE","Title")
    r.p("Distributed map matching and route reconstruction","Heading2")
    now=datetime.now(timezone(timedelta(hours=5,minutes=30)))
    r.p(f"Generated from recorded artifacts: {now:%d %B %Y, %H:%M IST}")
    r.p("<b>Review status:</b> AI-assisted implementation and evidence compilation. This report summarizes executed code and stored metrics; the team must review the implementation, verify results, add its own interpretation, and disclose assistance according to the event policy. No second-laptop demonstration is claimed without recorded evidence.")
    r.p("The system generates candidate directed road arcs from GPS and road geometry, learns candidate scores, and reconstructs routes using an independent log-space Viterbi decoder. Reference-route data is confined to permitted label derivation and server-side evaluation after inference.")
    r.p("<b>Evidence index:</b> results/eda/inventory.csv; results/model_comparison.json and .csv; results/cv/; results/models/; results/tuning_summary.json; results/hmm_comparison.json; results/predictions/; artifacts/; API execution logs and screenshots when recorded.")
    r.p("<b>Important:</b> pointwise labels are derived from supplied chronological route sequences. They are approximate reference labels, not independently surveyed physical ground truth. No exact official aggregate scoring formula is asserted.")
    r.page();r.heading("2.2 Dataset Understanding and EDA")
    r.p("All 80 training and 10 student-testing trips were inspected. No evaluator-testing files were used. Raw files are whitespace-delimited without headers: .track = longitude, latitude, seconds; .nodes = longitude, latitude; .arcs = directed node pairs; .route = chronological reference arc IDs. Node and arc row indexes are trip-local identifiers.")
    rows=[["Partition","Trips","GPS observations","Node rows summed","Arc rows summed","Largest gap (s)","1-second intervals"]]
    for name in ("train","student_test"):
        a=eda["partitions"][name]
        rows.append([name,a["trips"],a["observations"],a["nodes_summed"],a["arcs_summed"],number(a["max_gap_seconds"],0),f"{a['one_second_fraction']:.3%}"])
    r.table(rows,[85,55,110,110,110,105,110])
    a=eda["partitions"]["train"];b=eda["partitions"]["student_test"]
    r.p(f"Measured checks found {a['track_nonfinite_values']+b['track_nonfinite_values']} nonfinite track values, {a['invalid_arc_rows']+b['invalid_arc_rows']} invalid arc-endpoint rows, and {a['invalid_route_rows']+b['invalid_route_rows']} invalid reference IDs. Training graphs contain up to {int(a['arcs_rows']['maximum']):,} arcs. Maximum weak-component counts are {int(a['weak_components']['maximum'])} (train) and {int(b['weak_components']['maximum'])} (student-test). No isolated nodes were found.")
    r.p(f"The reference lists contain {a['nonadjacent_reference_steps']} train and {b['nonadjacent_reference_steps']} student-test nonadjacent consecutive arc steps. Repeated reference arcs occur, reaching {int(b['repeated_reference_arcs']['maximum'])} repeated occurrences in one student trip. These diagnostics motivate preserving reference-sequence positions and disclosing approximate alignment.")
    r.p("Preprocessing uses a trip-centered WGS84 azimuthal-equidistant projection in metres, exact point-to-segment distance, a Shapely STRtree and bounded candidate lists. Empty local queries expand within the documented maximum radius; observations outside this limit remain unmatched. Networks are processed one trip at a time. The source inventory includes SHA-256 file checksums.")
    r.image(RESULTS/"eda"/"trip_and_graph_counts.png",max_height=235)
    r.page();r.subheading("EDA observations and feature design")
    r.image(RESULTS/"eda"/"gaps_and_connectivity.png",max_height=230)
    r.image(RESULTS/"eda"/"missing_values.png",max_height=150)
    r.p("Candidate features include distance, log distance, heading disagreement, reliability-weighted heading disagreement, motion reliability, projection fraction, segment length, observed speed, time interval, distance rank, distance above best, heading sine/cosine and candidate count. IDs are excluded. Symmetric motion smoothing uses neighboring observations, so the implementation performs offline reconstruction rather than claiming causal tracking. Learned preprocessing is fitted within training folds.")
    r.page();r.heading("3.1 Overall Model Comparison")
    r.p("Exactly five distinct algorithms were trained and compared on the same cached candidate features. Three-fold GroupKFold keeps every trip and all of its candidate rows in one fold. Complete validation and student-test candidate sets are retained. Training-only negative sampling retains positives, hard negatives and a reproducible sample of remaining negatives; it changes score calibration and is disclosed.")
    csv= pd.read_csv(RESULTS/"model_comparison.csv",keep_default_na=False)
    columns=["Model","Algorithm Family","Key Configuration","CV Macro F1-score","Top-1 Accuracy","Route continuity","Student-Test Macro F1-score","Log Loss","Inference Time"]
    rows=[columns]
    for _,row in csv.iterrows():
        rows.append([number(row[c]) if isinstance(row[c],(float,np.floating)) else row[c] for c in columns])
    r.table(rows,[80,65,170,65,60,65,70,55,55])
    r.p("Top-1 Accuracy, Route continuity, Student-Test Macro F1-score and Log Loss in this table refer to the student-testing partition; CV Macro F1-score refers to held-out training trips. Inference Time is classifier scoring from cached features; complete loading, candidates and decoding costs are reported separately.")
    r.p(f"<b>Model selection:</b> {escape(selection['selected_model'])}. {escape(selection['reason'])} Grouped validation mean Top-1 = {number(selection.get('cv_top1'))}. This is a measured selection, not a model chosen from training accuracy alone.")
    r.p("Metric definitions: observation Top-1 includes every GPS observation, counting missing predictions and candidate misses as failures. Candidate Macro F1 is the mean across classes 0 and 1 at threshold 0.5. Candidate precision/recall and binary log loss use all valid generated candidate rows. Candidate-miss rate is reported separately because an omitted reference candidate cannot be recovered by a scorer.")
    r.page();r.heading("3.2 Separate Model Explanations")
    for i,m in enumerate(models):
        if i and i%2==0:r.page()
        name=m["name"];notes=MODEL_NOTES.get(name,{})
        r.subheading(name)
        for key in ("suitability","strengths","limitations","objective"):
            r.p(f"<b>{key.capitalize()}:</b> {escape(notes.get(key,'See recorded estimator configuration.'))}")
        r.p(f"<b>Configuration:</b> {escape(json.dumps(m['params'],sort_keys=True))}","SmallText")
        cv=m["cv"]["metrics"];test=m["student_test"];train=m["training_metrics"]
        r.table([["Train Top-1","CV Top-1 (pooled)","CV fold mean +/- SD","Student Top-1","Student Macro F1","Precision / Recall","Log loss","Fit / scoring seconds"],[number(train.get("top1_accuracy")),number(cv.get("top1_accuracy")),f"{number(cv.get('fold_mean_top1'))} +/- {number(cv.get('fold_std_top1'))}",number(test.get("top1_accuracy")),number(test.get("macro_f1")),f"{number(test.get('precision'))} / {number(test.get('recall'))}",number(test.get("log_loss")),f"{number(m.get('training_seconds'),2)} / {number(m.get('inference_seconds'),2)}"]],[75,90,110,85,85,95,70,75])
        gap=float(train.get("top1_accuracy",0))-float(cv.get("top1_accuracy",0))
        r.p(f"<b>Generalisation evidence:</b> training minus pooled grouped-CV Top-1 = {gap:+.4f}. A positive gap indicates lower performance on held-out trips; the magnitude and fold variation should be interpreted together. This measured gap alone does not prove a specific cause. Full fold assignments, confusion matrices and per-trip results are retained.","SmallText")
    r.page();r.heading("3.3 Model Visualisations")
    r.image(FIGURES/"cv_macro_f1.png",max_height=260)
    r.image(FIGURES/"student_model_comparison.png",max_height=205)
    r.page();r.subheading("Probability quality, computational cost and confusion matrices")
    r.image(FIGURES/"log_loss_and_scoring_time.png",max_height=240)
    r.image(FIGURES/"confusion_matrices.png",max_height=180)
    r.p("All charts above are generated from the stored experiment records. No artificial training curves or epoch histories have been created for estimators that did not produce them.")
    r.page();r.heading("4. Bayesian Hyperparameter Optimisation")
    # Print complete measured tuning metadata without guessing key names; the
    # summary producer determines the fields and every value stays auditable.
    r.p("The selected classifier is tuned with Optuna TPE, using three-fold trip-grouped cross-validation and Top-1 accuracy as the objective. Startup random proposals are followed by model-guided proposals; the persistent study and executed trial table preserve the search history. The evaluator partition is excluded.")
    for key,value in tuning.items():
        if key in ("trials","trial_history"):continue
        if isinstance(value,dict):
            r.subheading(key.replace("_"," ").capitalize())
            r.table([["Field","Recorded value"]]+[[k,json.dumps(v) if isinstance(v,(dict,list)) else number(v)] for k,v in value.items()],[190,495])
        elif not isinstance(value,list):r.p(f"<b>{escape(key.replace('_',' '))}:</b> {escape(number(value))}")
    r.p("Tuned and untuned results use the same metric definitions and data partitions. A tuned configuration is not automatically treated as better; the final-artifact choice is based on recorded grouped-validation results.")
    for candidate in ("tuned_vs_untuned.png","tuning_comparison.png","optuna_trials.png"):
        if (FIGURES/candidate).exists():r.image(FIGURES/candidate,max_height=230)
    r.page();r.heading("5. HMM and Viterbi Route Reconstruction")
    r.p("Hidden states are candidate directed arcs and projected positions along them. Emissions are clamped positive-class classifier scores normalized within each observation and used as log potentials. They are discriminative scores, not independently calibrated probabilities of final-route correctness.")
    r.p("The transition score penalizes disagreement between directed network distance and GPS displacement, with an additional implausible-speed penalty. For different arcs, distance includes the remaining source arc, a directed node path, and the destination prefix. Forward movement on the same arc uses direct fractional distance; backward movement requires a feasible directed loop. Impossible transitions receive negative infinity.")
    r.p("V[t,j] = log_emission[t,j] + max_i(V[t-1,i] + log_transition[i,j]). Backpointers recover the candidate sequence. Bounded Dijkstra trees are reused across destinations with an LRU limit; caches belong to one graph. Intermediate roads are reconstructed for supported transitions. Missing candidates, invalid times, large gaps and unreachable transitions produce explicit warnings and decoding restarts rather than invented connecting routes.")
    config=load_json(ROOT/"configs"/"decoder.json",False)
    if config:r.p(f"<b>Decoder configuration:</b> {escape(json.dumps(config,sort_keys=True))}","SmallText")
    if isinstance(hmm,dict):
        for key,value in hmm.items():
            if isinstance(value,dict) and key not in ("trips","per_trip"):
                r.subheading(key.replace("_"," ").capitalize())
                r.table([["Metric","Recorded value"]]+[[k,number(v) if not isinstance(v,(dict,list)) else json.dumps(v)] for k,v in value.items() if not isinstance(v,list)],[220,465])
            elif not isinstance(value,(dict,list)):r.p(f"<b>{escape(key.replace('_',' '))}:</b> {escape(number(value))}")
    for candidate in ("classifier_vs_hmm.png","hmm_comparison.png"):
        if (FIGURES/candidate).exists():r.image(FIGURES/candidate,max_height=260)
    r.p("Continuity counts feasible directed transitions within the recorded routing/time bounds. Eligible missing or unreachable predictions fail; exogenous invalid/large timestamp gaps are excluded and counted separately. Coverage and restarts are disclosed. Ordered-sequence performance is exact bitset LCS F1 = 2*LCS/(predicted sequence length + reference sequence length), a diagnostic rather than an official scoring formula.")
    r.page();r.heading("6. Geographical Visualisation Using Folium")
    meta=load_json(RESULTS/"eda"/"representative_map_metadata.json",False) or {}
    r.p(f"The committed interactive map is results/eda/representative_map.html, generated from actual permitted student-test trip {escape(str(meta.get('trip_id','recorded in metadata')))}. Blue markers show GPS, grey lines show road geometry, orange shows the supplied reference, and teal shows predictions when available. Layer controls and labels allow comparison. No reference layer is returned to the evaluator UI.")
    folium_shot=RESULTS/"screenshots"/"folium_map.png"
    r.image(folium_shot if folium_shot.exists() else RESULTS/"eda"/"representative_geographical_view.png",max_height=365)
    r.p("The image above is an actual rendered Folium view when a screenshot exists; otherwise it is a generated geographical export from the same measured geometry. Prediction-layer availability is recorded in representative_map_metadata.json. The HTML and map-generation code are included.","SmallText")
    r.page();r.heading("7. API, UI and Deployment")
    r.p("Laptop 1 runs FastAPI with the final artifact, reference-free feature extraction and the decoder. GET /health checks readiness; GET /trips dynamically discovers trips; POST /predict accepts a trip ID and executes inference. Only after inference does the separate server-side evaluator read reference information and produce aggregate metrics. Reference routes and pointwise labels are excluded from the HTTP response.")
    r.p("Laptop 2 serves the plain HTML/CSS/JavaScript evaluator UI. It connects to Laptop 1 over its LAN address, displays raw observations, predicted road geometry, metrics, inference time and warnings. The local canvas map remains functional without external basemap tiles.")
    r.p("Backend command: python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000","CodeSmall")
    r.p("Laptop 2 command: python -m http.server 8800 --directory frontend","CodeSmall")
    if api:
        for key,value in api.items():
            if not isinstance(value,(dict,list)):r.p(f"<b>{escape(key.replace('_',' '))}:</b> {escape(str(value))}")
        r.p("See results/api_verification.json and the retained actual server request logs for complete integration evidence.")
    else:r.p("<b>API verification record pending.</b> No successful API execution is inferred from source code alone.")
    r.p("<b>Second-laptop LAN demonstration:</b> pending unless the team adds real evidence from the second physical device. Local browser/API checks do not establish a successful two-device demonstration.")
    screenshot=RESULTS/"screenshots"/"ui.png"
    if screenshot.exists():r.image(screenshot,max_height=330)
    else:r.p("Working-UI screenshot pending at generation time.")
    server=RESULTS/"screenshots"/"api_terminal.png"
    if server.exists():r.page();r.subheading("Actual API receiving-terminal evidence");r.image(server,max_height=380)
    else:r.p("Receiving-laptop terminal screenshot pending; retained text logs do not substitute for the required physical demonstration screenshot.")
    r.page();r.heading("Metric definitions, limitations and review")
    for name,definition in METRIC_DEFINITIONS.items():r.p(f"<b>{escape(name)}:</b> {escape(definition)}","SmallText")
    r.p("The ten student-testing trips are permitted development data and should not be described as an untouched final generalisation estimate. Pointwise alignment ambiguity, candidate omissions, graph discontinuities, routing bounds and training-negative sampling limit interpretation. No claim is made about accuracy on reserved evaluator trips before their official evaluation.")
    r.p("The team should verify the recorded results, discuss cases where HMM helps or hurts, review all source code and configurations, conduct the physical LAN demonstration, and add its own conclusions. AI assistance supported implementation, debugging and evidence compilation; this document must not be misrepresented as independent student analysis.")
    r.subheading("Research concepts credited")
    r.p("Newson and Krumm (2009), Hidden Markov Map Matching Through Noise and Sparseness; Yang and Gidofalvi (2018), Fast Map Matching; Yang and Meng (2014), Feature Engineering for Map Matching of Low-Sampling-Rate GPS Trajectories. Existing engines were conceptual references; no complete external matcher is used as this decoder.","SmallText")
    target=RESULTS/"PathSnap_Technical_Report.pdf"
    doc=SimpleDocTemplate(str(target),pagesize=landscape(A4),rightMargin=35,leftMargin=35,topMargin=34,bottomMargin=42,title="PathSnap Trace-Lite Technical Report",author="PathSnap team - AI-assisted execution evidence")
    doc.build(r.story,onFirstPage=_footer,onLaterPages=_footer)
    source_files=[RESULTS/"model_comparison.json",RESULTS/"tuning_summary.json",RESULTS/"hmm_comparison.json",RESULTS/"eda"/"summary.json"]
    manifest={"report":str(target.relative_to(ROOT)),"generated_ist":now.isoformat(),"sources":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},"team_review_required":True,"second_laptop_demo":"pending_without_team_evidence"}
    (RESULTS/"report_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Generated {target}",flush=True)
    return target

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--render",action="store_true",help="Render every PDF page to results/report_review for visual QA");args=parser.parse_args()
    target=build_report()
    if args.render:
        import fitz
        dest=RESULTS/"report_review";dest.mkdir(exist_ok=True)
        doc=fitz.open(target)
        for i,page in enumerate(doc):page.get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(dest/f"page_{i+1:02d}.png")
        print(f"Rendered {len(doc)} pages for review",flush=True)

if __name__=="__main__":main()
