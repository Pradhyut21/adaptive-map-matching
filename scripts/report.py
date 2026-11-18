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
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, PageBreak
from pathsnap.settings import ROOT
from pathsnap.metrics import METRIC_DEFINITIONS

RESULTS=ROOT/"results"
FIGURES=RESULTS/"figures"
REFINEMENT_RESULTS=RESULTS/"context-boost-v1"

REFINEMENT_RECIPE_LABELS={
    "context_standard":"Unweighted",
    "context_weighted":"Confidence weighted",
    "context_weighted_larger":"Confidence weighted, larger trees",
}
REFINEMENT_DECODER_LABELS={
    "original_geometry":"Original transition geometry",
    "allow_gps_dwell":"Same-arc GPS jitter tolerance",
    "dwell_smooth":"GPS jitter tolerance and smoothing",
    "stronger_transitions":"Stronger transitions",
    "favor_classifier":"Higher classifier influence",
    "stronger_transitions_realign":"Stronger transitions and route realignment",
    "smoothed_realign":"Smoothed geometry and route realignment",
}
MODEL_LABELS={
    "GaussianNB":"Gaussian Naive Bayes",
    "LogisticRegression":"Logistic Regression",
    "DecisionTree":"Decision Tree",
    "ExtraTrees":"Extra Trees",
    "HistGradientBoosting":"Histogram Gradient Boosting",
}
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

def evidence_sha256(path):
    """Give text evidence the same digest in LF and CRLF checkouts."""
    data=path.read_bytes()
    if path.suffix in {".csv", ".json"}:
        data=data.replace(b"\r\n",b"\n")
    return hashlib.sha256(data).hexdigest()

def number(value, digits=4):
    if value is None:return "N/A"
    if isinstance(value,(float,np.floating)):
        return f"{value:.{digits}f}" if np.isfinite(value) else "N/A"
    return str(value)

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
    heat_max = max(np.log1p(np.asarray(m["student_test"]["confusion_matrix"])).max() for m in models)
    for ax,m in zip(axs,models):
        matrix=np.asarray(m["student_test"]["confusion_matrix"])
        ax.imshow(np.log1p(matrix),cmap="Blues",vmin=0,vmax=heat_max)
        for (row,col),count in np.ndenumerate(matrix):
            text_color = "white" if np.log1p(count) > 0.6 * heat_max else "#132c39"
            ax.text(col,row,f"{count:,}",ha="center",va="center",fontsize=8,color=text_color)
        ax.set(xticks=[0,1],yticks=[0,1],xlabel="Predicted",title=m["name"].replace("Gradient","\nGradient"));ax.tick_params(labelsize=8)
    axs[0].set_ylabel("True candidate class")
    fig.suptitle("Student-test candidate confusion matrices (0 = nonmatch, 1 = match)")
    fig.savefig(FIGURES/"confusion_matrices.png",dpi=170,bbox_inches="tight");plt.close(fig)
    a,b=tuning["untuned"],tuning["tuned"]
    labels=["CV Macro F1","CV Top-1","Student Top-1","Student Macro F1"]
    av=[a["cv"]["metrics"]["macro_f1"],a["cv"]["metrics"]["fold_mean_top1"],a["student_test"]["top1_accuracy"],a["student_test"]["macro_f1"]]
    bv=[b["cv"]["metrics"]["macro_f1"],b["cv"]["metrics"]["fold_mean_top1"],b["student_test"]["top1_accuracy"],b["student_test"]["macro_f1"]]
    x=np.arange(len(labels));fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(x-.18,av,width=.36,label="Untuned",color="#9eafb6");ax.bar(x+.18,bv,width=.36,label="TPE tuned",color="#087f8c")
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1.04),title=f"Tuned versus untuned: {tuning['selected_model']}");ax.legend();ax.grid(axis="y",alpha=.15)
    fig.savefig(FIGURES/"tuned_vs_untuned.png",dpi=170,bbox_inches="tight");plt.close(fig)
    trials=tuning.get("trials",[])
    if trials:
        fig,ax=plt.subplots(figsize=(10,3.6))
        for phase,color in (("startup_random","#9eafb6"),("guided_TPE","#087f8c")):
            group=[t for t in trials if t.get("phase")==phase and t.get("value") is not None]
            if group:ax.scatter([t["number"] for t in group],[t["value"] for t in group],c=color,s=45,label=phase)
        ax.set(xlabel="Executed trial number",ylabel="Mean grouped-CV Top-1",title="Actual Optuna trial objectives");ax.legend();ax.grid(alpha=.15)
        fig.savefig(FIGURES/"optuna_trials.png",dpi=170,bbox_inches="tight");plt.close(fig)
    categories=[("baseline","Nearest road"),("classifier","Classifier only"),("hmm","Classifier + HMM")]
    fig,axs=plt.subplots(1,3,figsize=(12,3.6))
    for ax,key,title in ((axs[0],"top1_accuracy","Observation Top-1"),(axs[1],"route_continuity","Directed continuity"),(axs[2],"sequence_lcs_f1","Ordered sequence LCS F1")):
        values=[hmm[c]["aggregate"][key] for c,_ in categories]
        bars=ax.bar([label for _,label in categories],values,color=["#9eafb6","#cb6c36","#087f8c"])
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+.01,f"{v:.3f}",ha="center",fontsize=8)
        ax.set(ylim=(0,1.06),title=title);ax.tick_params(axis="x",labelrotation=12,labelsize=8);ax.grid(axis="y",alpha=.15)
    fig.savefig(FIGURES/"classifier_vs_hmm.png",dpi=170,bbox_inches="tight");plt.close(fig)

class Report:
    def __init__(self):
        self.styles=getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name="BodyCompact",fontName="Helvetica",fontSize=10.2,leading=14,spaceAfter=7,textColor=colors.HexColor("#23343e")))
        self.styles.add(ParagraphStyle(name="SmallText",fontName="Helvetica",fontSize=8,leading=10,spaceAfter=5))
        self.styles.add(ParagraphStyle(name="Cell",fontName="Helvetica",fontSize=7.3,leading=9))
        self.styles.add(ParagraphStyle(name="CellHeader",parent=self.styles["Cell"],fontName="Helvetica-Bold",textColor=colors.HexColor("#174954")))
        self.styles.add(ParagraphStyle(name="CodeSmall",fontName="Courier",fontSize=7.2,leading=10,spaceAfter=5))
        self.styles.add(ParagraphStyle(name="CoverTitle",fontName="Helvetica-Bold",fontSize=31,leading=36,spaceAfter=6,textColor=colors.HexColor("#103742")))
        self.styles["Heading1"].textColor=colors.HexColor("#075663")
        self.styles["Heading2"].textColor=colors.HexColor("#075663")
        self.story=[]

    def p(self,text,style="BodyCompact"):
        self.story.append(Paragraph(str(text),self.styles[style]))

    def heading(self,text):self.p(escape(text),"Heading1")
    def subheading(self,text):self.p(escape(text),"Heading2")
    def page(self):self.story.append(PageBreak())
    def table(self,rows,widths=None,padding=6):
        cells=[[Paragraph(escape(str(v)).replace("\n","<br/>"),self.styles["CellHeader" if i==0 else "Cell"]) for v in row] for i,row in enumerate(rows)]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dceef0")),("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#c2d1d5")),("TOPPADDING",(0,0),(-1,-1),padding),("BOTTOMPADDING",(0,0),(-1,-1),padding),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f8f9")])]))
        t.spaceAfter = 10
        self.story.append(t)
    def image(self,path,max_width=720,max_height=320,align="CENTER"):
        p=Path(path)
        if not p.exists():self.p(f"Evidence image pending: {escape(str(p.relative_to(ROOT)))}.");return
        from PIL import Image as PILImage
        with PILImage.open(p) as im:w,h=im.size
        scale=min(max_width/w,max_height/h)
        visual=Image(str(p),width=w*scale,height=h*scale,hAlign=align)
        visual.spaceAfter=7
        self.story.append(visual)

def _footer(canvas,doc):
    canvas.saveState();w,h=landscape(A4)
    canvas.setStrokeColor(colors.HexColor("#d0dde0"));canvas.line(35,30,w-35,30)
    canvas.setFont("Helvetica",7);canvas.setFillColor(colors.HexColor("#536a73"))
    canvas.drawString(35,19,"PathSnap Trace-Lite | Recorded execution evidence | Team review required")
    canvas.drawRightString(w-35,19,f"Page {doc.page}")
    canvas.restoreState()

def add_reliability_ablation(report, evidence):
    """Optional same-algorithm feature comparison, only from executed records."""
    full=evidence["full_features"]
    reduced=evidence["without_reliability"]
    report.page();report.heading("3.4 Heading Reliability Feature Ablation")
    report.p(f"Both variants use the selected {escape(evidence['algorithm'])} algorithm, final hyperparameters, trip folds and negative sampling. The ablation removes {escape(', '.join(evidence['removed_features']))} while retaining the original smoothed heading.")
    report.p(f"<b>Fixed parameters:</b> {escape(json.dumps(evidence['fixed_params'],sort_keys=True))}","SmallText")
    report.p(f"<b>Hypothesis:</b> {escape(str(evidence['hypothesis']))}")
    rows=[["Feature variant","CV fold-mean Top-1","CV Macro F1","Student Top-1","Student Macro F1","Student log loss"]]
    values=[]
    for label,record in (("Full features",full),("Without reliability",reduced)):
        cv=record["cv"]["metrics"];test=record["student_test"]
        v=[cv["fold_mean_top1"],cv["macro_f1"],test["top1_accuracy"],test["macro_f1"],test["log_loss"]]
        values.append(v)
        rows.append([label]+[number(x,5) for x in v])
    delta=np.asarray(values[0])-np.asarray(values[1])
    rows.append(["Full minus removed"]+[f"{x:+.5f}" for x in delta])
    report.table(rows,[135,120,100,110,115,105])
    def outcome(value):
        if value>0:return "improvement"
        if value<0:return "regression"
        return "no change"
    report.p(f"Using full features yields a grouped-CV Top-1 {outcome(delta[0])} of {abs(delta[0])*100:.2f} percentage points and a student-test Top-1 {outcome(delta[2])} of {abs(delta[2])*100:.2f} points. Higher Top-1 and F1 and lower log loss indicate better performance against the derived reference labels.")
    fig,ax=plt.subplots(figsize=(10,3.3));x=np.arange(4)
    ax.bar(x-.18,values[0][:4],width=.36,color="#087f8c",label="Full features")
    ax.bar(x+.18,values[1][:4],width=.36,color="#9eafb6",label="Without reliability")
    for shift,v in ((-.18,values[0]),(.18,values[1])):
        for j,y in enumerate(v[:4]):ax.text(j+shift,y+.008,f"{y:.4f}",ha="center",fontsize=7)
    ax.set(xticks=x,xticklabels=["CV Top-1","CV Macro F1","Student Top-1","Student Macro F1"],ylim=(0,1.06),title="Executed same-algorithm feature ablation")
    ax.legend(loc="lower right");ax.grid(axis="y",alpha=.15)
    target=FIGURES/"motion_reliability_ablation.png"
    fig.savefig(target,dpi=170,bbox_inches="tight");plt.close(fig)
    report.image(target,max_height=205)
    report.p(f"The final artifact was frozen before this experiment, which took {number(evidence['elapsed_seconds'],2)} seconds. The label alignment also uses heading, so the comparison measures agreement with that alignment. The two variants retain the same smoothed heading inputs and sequence decoding is outside the ablation.","SmallText")

def percentage(value):
    return "N/A" if value is None else f"{value:.2%}"


def refinement_label(key, labels):
    if key.startswith("tpe_"):
        return f"TPE trial {int(key.removeprefix('tpe_'))}"
    return labels.get(key, key.replace("_", " "))


def load_refinement_evidence():
    required_names=("selection", "decoder_selection", "student_test")
    records={name: load_json(REFINEMENT_RESULTS/f"{name}.json", False) for name in required_names}
    if any(record is None for record in records.values()):
        return None
    for name in ("boost_tuning", "api_integration"):
        records[name]=load_json(REFINEMENT_RESULTS/f"{name}.json", False)
    return records


def add_accuracy_refinement(report, records, original, models):
    selection=records["selection"]
    decoder=records["decoder_selection"]
    student=records["student_test"]
    report.page()
    report.heading("8. Accuracy Refinement")
    report.p(f"The refined run reused the Histogram Gradient Boosting algorithm and increased the candidate limit to {selection['candidate_k']} arcs within the {selection['candidate_radius_m']:g} m search radius. Its {len(selection['feature_names'])} inputs use only GPS observations, candidate geometry and short observation windows. The original trip partition, three cross-validation folds and derived reference labels were held fixed.")
    report.p("The confidence-weighted variants downweight training observations whose derived label is far from the GPS point or farther than the nearest candidate, with an extra reduction for flagged poor fits. Pointwise evaluation still counts every observation. A reference arc absent from the candidate list counts as a miss.")
    report.subheading("Model and decoder selection on training trips")
    recipe_rows=[["Supervised variant", "Mean fold Top-1", "Pooled Top-1", "Mean trip Top-1"]]
    for recipe in selection["recipes"]:
        recipe_rows.append([refinement_label(recipe["recipe"], REFINEMENT_RECIPE_LABELS), percentage(recipe["mean_fold_accuracy"]), percentage(recipe["pooled_accuracy"]), percentage(recipe["mean_trip_accuracy"])])
    report.table(recipe_rows, [278, 150, 150, 147], padding=4)
    selected_recipe=escape(refinement_label(selection["selected_recipe"], REFINEMENT_RECIPE_LABELS))
    report.p(f"The selected variant was <b>{selected_recipe}</b> at {percentage(selection['validation_accuracy'])} mean fold exact Top-1. The selected model was refit on the 80 training trips.")
    if (tuning:=records["boost_tuning"]) is not None:
        complete=[trial for trial in tuning["trials"] if trial["state"]=="COMPLETE"]
        startup=sum(trial["phase"]=="startup" for trial in complete)
        guided=sum(trial["phase"]=="guided_TPE" for trial in complete)
        decision="The tuned parameters were used for refitting." if tuning["use_tuned"] else "Refitting used the original selected parameters."
        report.p(f"The search completed {startup} startup and {guided} TPE-guided trials. Best trial {tuning['best_trial']} reached {percentage(tuning['best_accuracy'])} mean fold Top-1. The original selected variant reached {percentage(tuning['original_accuracy'])}. {decision}", "SmallText")
    decoder_rows=[["Decoder variant", "Training OOF Top-1", "Ordered-sequence LCS F1"]]
    for key, score in decoder["scores"].items():
        label=refinement_label(key, REFINEMENT_DECODER_LABELS)
        if key==decoder["selected"]:
            label += " (selected)"
        decoder_rows.append([label, percentage(score["top1_accuracy"]), number(score.get("sequence_lcs_f1"))])
    report.table(decoder_rows, [365, 180, 180], padding=3)
    report.p(f"Decoder selection used a fixed subset of {len(decoder['trip_ids'])} training trips. Each received out-of-fold scores from a model fitted on other trips. Selection maximized pooled exact Top-1. The student-test partition was evaluated after the classifier and decoder were frozen. Reused training folds provide model-selection evidence; the student-test set is a reused benchmark across project versions. Neither is a new independent generalisation estimate.", "SmallText")

    report.page()
    report.subheading("Student-test accuracy with frozen reference labels")
    baseline=original["hmm"]["aggregate"]
    classifier=student["classifier"]
    refined=student["hmm"]
    if baseline["observations"]!=refined["observations"]:
        raise ValueError("Original and refined evaluation observation totals differ")
    score_rows=[["Student-test method", "Exact arc Top-1", "Directed continuity", "Ordered-sequence LCS F1", "Candidate miss rate"]]
    for label, score in (("Original selected classifier + HMM",baseline),("Refined classifier only",classifier),("Refined classifier + decoder",refined)):
        score_rows.append([label, percentage(score["top1_accuracy"]), percentage(score.get("route_continuity")), number(score.get("sequence_lcs_f1")), percentage(score["candidate_miss_rate"])])
    report.table(score_rows, [230, 105, 120, 155, 115], padding=4)
    delta=(refined["top1_accuracy"]-baseline["top1_accuracy"])*100
    report.p(f"The refinement changed exact-arc Top-1 by <b>{delta:+.2f} percentage points</b> over {refined['observations']:,} student-test observations with the same frozen labels. These scores measure agreement with the derived labels.")
    original_boost=next(model for model in models if model["name"]=="HistGradientBoosting")["student_test"]
    confusion=np.asarray(original_boost["confusion_matrix"])
    candidate_accuracy=np.trace(confusion)/confusion.sum()
    report.p(f"The earlier Histogram Gradient Boosting run correctly classified {percentage(candidate_accuracy)} of binary candidate rows, including true negatives. It selected the exact directed arc for {percentage(original_boost['top1_accuracy'])} of GPS observations. Exact arc Top-1 is the comparable measure in the table above.", "SmallText")
    original_trips={trip["trip_id"]: trip for trip in original["hmm"]["trips"]}
    refined_trips={trip["trip_id"]: trip for trip in refined["trips"]}
    if original_trips.keys()!=refined_trips.keys():
        raise ValueError("Original and refined student-test trip IDs differ")
    trip_rows=[["Trip", "Original HMM Top-1", "Refined Top-1", "Change (percentage points)", "Refined sequence LCS F1"]]
    for trip_id, trip in sorted(refined_trips.items()):
        previous=original_trips[trip_id]["top1_accuracy"]
        current=trip["top1_accuracy"]
        trip_rows.append([trip_id, percentage(previous), percentage(current), f"{(current-previous)*100:+.2f}", number(trip.get("sequence_lcs_f1"))])
    report.table(trip_rows, [100, 145, 145, 170, 165], padding=3)
    if (api:=records["api_integration"]) is not None and api.get("requests"):
        requests=api["requests"]
        passed=sum(request.get("http_status")==200 for request in requests)
        report.p(f"The refined artifact produced {passed} HTTP 200 responses in {len(requests)} recorded local requests. The physical two-laptop demonstration is pending.", "SmallText")
    screenshot=REFINEMENT_RESULTS/"screenshots/ui_desktop.png"
    if screenshot.is_file():
        report.page()
        report.subheading("Refined model - actual working interface")
        report.image(screenshot,max_height=402)
        report.p("Actual same-machine browser inference using the refined Histogram Gradient Boosting model. Ten HTTP checks exactly reproduced the frozen Top-1 and sequence scores. This does not establish the pending physical two-laptop demonstration.", "SmallText")


def build_report():
    _,models=model_records()
    eda=load_json(RESULTS/"eda"/"summary.json")
    selection=load_json(RESULTS/"selection.json")
    tuning=load_json(RESULTS/"tuning_summary.json")
    hmm=load_json(RESULTS/"hmm_comparison.json")
    api=load_json(RESULTS/"api_integration.json",False)
    ablation=load_json(RESULTS/"ablation.json",False)
    capture=load_json(RESULTS/"screenshots"/"capture_metadata.json",False)
    refinement=load_refinement_evidence()
    make_comparison_plots(models,tuning,hmm)
    r=Report()
    r.p("PathSnap Trace-Lite","CoverTitle")
    r.p("Distributed map matching and route reconstruction","Heading2")
    now=datetime.now(timezone(timedelta(hours=5,minutes=30)))
    r.p(f"Technical report generated from execution records on {now:%d %B %Y at %H:%M IST}.")
    r.p("The system generates candidate directed road arcs from GPS and road geometry, learns candidate scores, and reconstructs routes using an independent log-space Viterbi decoder. Reference-route data is confined to permitted label derivation and server-side evaluation after inference.")
    aggregate=hmm["hmm"]["aggregate"]
    best_model=tuning["selected_model"]
    selected_name=escape(MODEL_LABELS.get(best_model,best_model))
    version="tuned " if tuning["used_tuned_model"] else ""
    r.p(f"The original model comparison selected {version}{selected_name} by three-fold trip-grouped validation on 80 training trips. Route comparisons use {aggregate['observations']:,} observations across {aggregate['trips']} student-test trips.")
    summary_rows=[["Student-test method","Observation Top-1","Directed continuity","Ordered-sequence LCS F1"]]
    for key,label in (("baseline","Nearest road"),("classifier","Selected classifier"),("hmm","Selected classifier + HMM")):
        item=hmm[key]["aggregate"]
        summary_rows.append([label,f"{item['top1_accuracy']:.2%}",f"{item['route_continuity']:.2%}",number(item['sequence_lcs_f1'])])
    if refinement is not None:
        item=refinement["student_test"]["hmm"]
        summary_rows.append(["Accuracy refinement + decoder",percentage(item["top1_accuracy"]),percentage(item.get("route_continuity")),number(item.get("sequence_lcs_f1"))])
    r.table(summary_rows,[220,160,155,190])
    if api and api.get("requests"):
        passed=sum(request.get("http_status")==200 for request in api["requests"])
        r.p(f"The original artifact's local HTTP check returned {passed} successful predictions from {len(api['requests'])} requests. The physical two-laptop demonstration and receiving-terminal screenshot are pending.")
    r.p("Pointwise accuracy uses approximate labels derived from the supplied chronological route sequences. The sequence and continuity metrics are project diagnostics.")
    r.p("AI assisted implementation, debugging and evidence compilation. The team must verify the implementation, interpret results independently and disclose the assistance according to the event policy.")
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
    r.p("Preprocessing uses a trip-centered WGS84 azimuthal-equidistant projection in metres, exact point-to-segment distance, a Shapely STRtree and ten-candidate bounded lists. The final feature cache uses a 250 m radius selected from training-only candidate coverage checks; the earlier 100 m smoke setting is not mixed into final training. Empty local queries may expand within the documented maximum radius; observations outside this limit remain unmatched. Networks are processed one trip at a time. The source inventory includes SHA-256 file checksums.")
    r.image(RESULTS/"eda"/"trip_and_graph_counts.png",max_height=200)
    r.page();r.subheading("EDA observations and feature design")
    r.image(RESULTS/"eda"/"gaps_and_connectivity.png",max_height=173)
    r.image(RESULTS/"eda"/"missing_values.png",max_height=113)
    r.p("Candidate features include distance, log distance, heading disagreement, reliability-weighted heading disagreement, motion reliability, projection fraction, segment length, observed speed, time interval, distance rank, distance above best, heading sine/cosine and candidate count. IDs are excluded. Symmetric motion smoothing uses neighboring observations, so the implementation performs offline reconstruction rather than claiming causal tracking. Learned preprocessing is fitted within training folds.")
    r.p("Sparse dynamic programming derives labels from ordered reference-route positions and preserves repeated arcs. Its emission cost combines squared segment distance at a 20 m scale with a small reliability-weighted heading term. Transitions favor monotone progress consistent with observed displacement. Impossible monotone steps trigger a logged exact nearest-reference fallback. Pointwise evaluation shares these geometry and motion assumptions with the features. Ordered-sequence agreement provides a complementary check.")
    inventory=load_json(RESULTS/'feature_inventory.json',False)
    if inventory:
        trips=inventory['trips'];total=sum(t['observations'] for t in trips)
        fallbacks=sum(t['alignment']['fallback_restarts'] for t in trips)
        poor=sum(t['alignment'].get('poor_fit_count',0) for t in trips)
        ambiguous=sum(t['alignment']['ambiguous_observations'] for t in trips)
        r.p(f"Across the 90 permitted trips, alignment logged {fallbacks:,} nearest-fallback boundaries ({fallbacks/max(total,1):.3%} of observations), {ambiguous:,} ambiguous observations ({ambiguous/max(total,1):.3%}), and {poor:,} poor-fit observations whose selected reference distance exceeds the nearest-reference distance by over 50 m. Poor fits are flagged rather than silently relabeled. Source: results/feature_inventory.json.",'SmallText')
    r.page();r.heading("3.1 Overall Model Comparison")
    r.p("Exactly five distinct algorithms were trained and compared on the same cached candidate features. Three-fold GroupKFold keeps every trip and all of its candidate rows in one fold. Complete validation and student-test candidate sets are retained. Training-only negative sampling retains positives, hard negatives and a reproducible sample of remaining negatives; it changes score calibration and is disclosed.")
    csv= pd.read_csv(RESULTS/"model_comparison.csv",keep_default_na=False)
    columns=["Model","Algorithm Family","Key Configuration","CV Macro F1-score","Top-1 Accuracy","Route continuity","Student-Test Macro F1-score","Log Loss","Inference Time"]
    rows=[columns]
    for _,row in csv.iterrows():
        values=[number(row[c]) if isinstance(row[c],(float,np.floating)) else row[c] for c in columns]
        values[0]=MODEL_LABELS.get(values[0],values[0])
        rows.append(values)
    r.table(rows,[80,65,170,65,60,65,70,55,55])
    r.p("Top-1 Accuracy, Route continuity, Student-Test Macro F1-score and Log Loss in this table refer to the student-testing partition; CV Macro F1-score refers to held-out training trips. Inference Time is classifier scoring from cached features; complete loading, candidates and decoding costs are reported separately.")
    r.p(f"<b>Model selection:</b> {escape(selection['selected_model'])}. {escape(selection['reason'])} Grouped validation mean Top-1 = {number(selection.get('cv_top1'))}. This is a measured selection, not a model chosen from training accuracy alone.")
    r.p("Metric definitions: observation Top-1 includes every GPS observation, counting missing predictions and candidate misses as failures. Candidate Macro F1 is the mean across classes 0 and 1 at threshold 0.5. Candidate precision/recall and binary log loss use all valid generated candidate rows. Candidate-miss rate is reported separately because an omitted reference candidate cannot be recovered by a scorer.")
    r.page();r.heading("3.2 Separate Model Explanations")
    for i,m in enumerate(models):
        if i and i%2==0:r.page()
        name=m["name"];notes=MODEL_NOTES.get(name,{})
        r.subheading(MODEL_LABELS.get(name,name))
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
    if ablation is not None:add_reliability_ablation(r,ablation)
    r.page();r.heading("4. Bayesian Hyperparameter Optimisation")
    r.p("The selected classifier is tuned with Optuna TPE, using three-fold trip-grouped cross-validation and Top-1 accuracy as the objective. Startup random proposals are followed by model-guided proposals; the persistent study and executed trial table preserve the search history. The evaluator partition is excluded.")
    r.p(f"<b>Selected model:</b> {escape(tuning['selected_model'])}; sampler: {escape(tuning['sampler'])}; fixed seed: {tuning['seed']}; startup trials: {tuning['startup_trials']}; executed trials: {len(tuning['trials'])}; elapsed tuning/refit time: {number(tuning['elapsed_seconds'],2)} seconds.")
    spaces={"GaussianNB":"var_smoothing: log-uniform [1e-12, 1e-4]", "LogisticRegression":"C: log-uniform [0.005, 20]", "DecisionTree":"max_depth: integer [7,20]; min_samples_leaf: log-integer [5,80]; ccp_alpha: log-uniform [1e-7,1e-3]", "ExtraTrees":"n_estimators: [40,96] step 8; max_depth: integer [10,22]; min_samples_leaf: log-integer [3,35]; max_features: uniform [0.5,1]", "HistGradientBoosting":"max_iter: [60,140] step 20; max_leaf_nodes: categorical {7,15,31}; learning_rate: log-uniform [0.04,0.2]; l2_regularization: log-uniform [0.1,20]; min_samples_leaf: [20,80] step 10"}
    r.p(f"<b>Search space:</b> {escape(spaces.get(tuning['selected_model'],'See scripts/tune.py'))}. The small model-specific space targets complexity and regularization under the measured CPU budget.")
    r.p(f"<b>Best executed parameters:</b> {escape(json.dumps(tuning['best_params'],sort_keys=True))}; best search objective = {number(tuning['best_objective'])}.")
    rows=[["Version","CV Macro F1","CV fold-mean Top-1","Student Top-1","Student Macro F1","Student log loss","Scoring seconds"]]
    for key in ("untuned","tuned"):
        m=tuning[key];cv=m["cv"]["metrics"];test=m["student_test"]
        rows.append([key,number(cv["macro_f1"]),number(cv["fold_mean_top1"]),number(test["top1_accuracy"]),number(test["macro_f1"]),number(test["log_loss"]),number(m["inference_seconds"],3)])
    r.table(rows,[75,95,120,100,100,100,95])
    r.p(f"<b>Final artifact tag:</b> {escape(tuning['final_tag'])}; tuned configuration selected: {str(tuning['used_tuned_model']).lower()}. {escape(tuning['rationale'])} Tuned and untuned scores use identical data partitions and metric definitions.")
    r.image(FIGURES/"tuned_vs_untuned.png",max_height=220)
    r.page();r.subheading("Executed Bayesian search history")
    rows=[["Trial","Phase","State","Grouped objective","Seconds","Parameters"]]
    for t in tuning["trials"]:
        rows.append([t["number"],t.get("phase","recorded in trial"),t.get("state",""),number(t.get("value")),number(t.get("seconds"),2),json.dumps(t["params"],sort_keys=True)])
    r.table(rows,[40,90,60,85,65,345])
    r.image(FIGURES/"optuna_trials.png",max_height=220)
    r.p("Source records: results/optuna.sqlite3, results/optuna_trials.csv and results/tuning_trials/. Startup proposals are random; later guided_TPE proposals are supplied by TPE. No executed search result has been replaced by a fabricated curve.","SmallText")
    r.page();r.heading("5. HMM and Viterbi Route Reconstruction")
    r.p("Hidden states are candidate directed arcs and projected positions along them. Emissions are clamped positive-class classifier scores normalized within each observation and used as log potentials. They are discriminative scores, not independently calibrated probabilities of final-route correctness.")
    r.p("The transition score penalizes disagreement between directed network distance and GPS displacement, with an additional implausible-speed penalty. For different arcs, distance includes the remaining source arc, a directed node path, and the destination prefix. Forward movement on the same arc uses direct fractional distance; backward movement requires a feasible directed loop. Impossible transitions receive negative infinity.")
    r.p("V[t,j] = log_emission[t,j] + max_i(V[t-1,i] + log_transition[i,j]). Backpointers recover the candidate sequence. Bounded Dijkstra trees are reused across destinations with an LRU limit; caches belong to one graph. Intermediate roads are reconstructed for supported transitions. Missing candidates, invalid times, large gaps and unreachable transitions produce explicit warnings and decoding restarts rather than invented connecting routes.")
    r.p(f"<b>Decoder configuration:</b> {escape(json.dumps(hmm['config'],sort_keys=True))}","SmallText")
    rows=[["Method","Top-1","Mean trip Top-1","Continuity","Sequence LCS F1","Coverage","Eligible transitions","Excluded gaps"]]
    for key,label in (("baseline","Nearest road"),("classifier","Classifier only"),("hmm","Classifier + HMM")):
        a=hmm[key]["aggregate"]
        rows.append([label,number(a["top1_accuracy"]),number(a["mean_trip_top1_accuracy"]),number(a["route_continuity"]),number(a["sequence_lcs_f1"]),number(a["prediction_coverage"]),a["eligible_transitions"],a["excluded_time_gaps"]])
    r.table(rows,[110,65,95,75,95,70,100,75])
    r.p(f"HMM decoder-only time across all ten cached student trips: {number(hmm['hmm']['aggregate']['inference_seconds'],2)} seconds. This excludes feature preparation and model scoring; live API timing measures the complete prediction pipeline. Candidate-miss rate on these shared candidates is {number(hmm['hmm']['aggregate']['candidate_miss_rate'])}. {escape(hmm['methodology'])}","SmallText")
    r.image(FIGURES/"classifier_vs_hmm.png",max_height=142)
    r.p("Continuity measures directed reachability over eligible adjacent observations within the recorded routing bounds. Missing or unreachable predictions fail. The excluded invalid or large time gaps are counted separately. Ordered-sequence LCS F1 is twice the longest common subsequence length divided by the sum of the predicted and reference sequence lengths.","SmallText")
    r.page();r.subheading("Per-trip reconstruction evidence and failure cases")
    rows=[["Trip","Nearest Top-1","Classifier Top-1","HMM Top-1","HMM continuity","HMM LCS F1","Restarts","Decoder seconds"]]
    for b,c,h in zip(hmm["baseline"]["trips"],hmm["classifier"]["trips"],hmm["hmm"]["trips"]):
        if not (b["trip_id"]==c["trip_id"]==h["trip_id"]):raise ValueError("Trip order mismatch in HMM evidence")
        rows.append([h["trip_id"],number(b["top1_accuracy"]),number(c["top1_accuracy"]),number(h["top1_accuracy"]),number(h["route_continuity"]),number(h["sequence_lcs_f1"]),h.get("restart_count","See decoder_stats"),number(h.get("inference_seconds"),2)])
    r.table(rows,[75,85,90,85,90,85,85,90])
    improvements=[h['trip_id'] for c,h in zip(hmm['classifier']['trips'],hmm['hmm']['trips']) if h['top1_accuracy']>c['top1_accuracy']]
    regressions=[h['trip_id'] for c,h in zip(hmm['classifier']['trips'],hmm['hmm']['trips']) if h['top1_accuracy']<c['top1_accuracy']]
    r.p(f"Measured pointwise improvement over classifier-only occurs on trips: {', '.join(improvements) if improvements else 'none'}. Measured pointwise regression occurs on trips: {', '.join(regressions) if regressions else 'none'}. Inspect the saved route overlays and warning records before assigning a cause; better continuity does not necessarily imply a more accurate reference match.")
    reasons={}
    for h in hmm['hmm']['trips']:
        for w in h.get('warnings',[]):
            name=w.get('reason','unspecified');reasons[name]=reasons.get(name,0)+1
    r.table([["Recorded warning reason","Count"]]+[[k,v] for k,v in sorted(reasons.items())],[500,185])
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
        requests=api.get("requests",[])
        if requests:
            rows=[["Trip","HTTP status","Observations","Inference (s)","HTTP elapsed (s)","Run ID"]]
            for request in requests:
                rows.append([request["trip_id"],request.get("http_status","N/A"),request.get("observations","N/A"),number(request.get("timing",{}).get("inference_seconds"),2),number(request.get("http_seconds"),2),request.get("run_id","N/A")])
            r.table(rows,[105,95,105,105,115,160],padding=4)
            passed=sum(request.get("http_status")==200 for request in requests)
            r.p(f"Same-machine integration returned HTTP 200 for {passed} of {len(requests)} requests in {escape(api['partition'])}. See results/api_integration.json and results/execution_logs/actual_api_server.txt.","SmallText")
    else:r.p("<b>API verification record pending.</b> No successful API execution is inferred from source code alone.")
    r.p("The physical two-laptop LAN demonstration and receiving-terminal screenshot are pending. See docs/deployment.md for the demonstration procedure.","SmallText")
    r.page();r.subheading("Actual working interface")
    screenshot=RESULTS/"screenshots"/"ui_desktop.png"
    if not screenshot.exists():screenshot=RESULTS/"screenshots"/"ui.png"
    if screenshot.exists():
        r.image(screenshot,max_height=418)
        if capture:
            r.p(f"Local browser run {escape(capture['run_id'])} reconstructed student-test trip {escape(capture['trip_id'])} from {capture['observation_count']:,} GPS observations. The separate frontend and API services ran on the same machine.","SmallText")
    else:r.p("Working-UI screenshot pending at generation time.")
    server=RESULTS/"screenshots"/"api_terminal.png"
    if server.exists():r.page();r.subheading("Actual API receiving-terminal evidence");r.image(server,max_height=380)
    else:
        logs=RESULTS/"screenshots"/"server_logs_browser.png"
        if logs.exists():
            r.page();r.subheading("Actual server logs displayed in the browser")
            r.image(logs,max_height=408)
            r.p("The server log viewer displays the stored request record from the browser run. The report archive also contains the server's original text log.","SmallText")
        r.p("The receiving-terminal screenshot remains pending until the physical demonstration.","SmallText")
    if refinement is not None:add_accuracy_refinement(r,refinement,hmm,models)
    r.page();r.heading("Metric definitions, limitations and review")
    for name,definition in METRIC_DEFINITIONS.items():r.p(f"<b>{escape(name)}:</b> {escape(definition)}","SmallText")
    r.p("The ten student-testing trips are permitted development data and should not be described as an untouched final generalisation estimate. Pointwise alignment ambiguity, candidate omissions, graph discontinuities, routing bounds and training-negative sampling limit interpretation. No claim is made about accuracy on reserved evaluator trips before their official evaluation.")
    r.p("The team should verify the recorded results, discuss cases where HMM helps or hurts, review all source code and configurations, conduct the physical LAN demonstration, and add its own conclusions. AI assistance supported implementation, debugging and evidence compilation; this document must not be misrepresented as independent student analysis.")
    r.subheading("Research concepts credited")
    r.p('Newson and Krumm (2009), <link href="https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/" color="#075663">Hidden Markov Map Matching Through Noise and Sparseness</link>.',"SmallText")
    r.p('Yang and Gidofalvi (2018), <link href="https://doi.org/10.1080/13658816.2017.1400548" color="#075663">Fast map matching, an algorithm integrating hidden Markov model with precomputation</link>.',"SmallText")
    r.p('Yang and Meng (2014), <link href="https://arxiv.org/abs/1409.0797" color="#075663">Feature Engineering for Map Matching of Low-Sampling-Rate GPS Trajectories</link>.',"SmallText")
    target=RESULTS/"PathSnap_Technical_Report.pdf"
    doc=SimpleDocTemplate(str(target),pagesize=landscape(A4),rightMargin=35,leftMargin=35,topMargin=34,bottomMargin=42,title="PathSnap Trace-Lite Technical Report",author="PathSnap team - AI-assisted execution evidence")
    doc.build(r.story,onFirstPage=_footer,onLaterPages=_footer)
    source_files=[RESULTS/"model_comparison.json",RESULTS/"model_comparison.csv",RESULTS/"selection.json",RESULTS/"tuning_summary.json",RESULTS/"hmm_comparison.json",RESULTS/"eda"/"summary.json",RESULTS/"eda"/"inventory.csv",RESULTS/"eda"/"representative_map_metadata.json"]
    source_files += [path for path in (RESULTS/"feature_inventory.json",RESULTS/"screenshots"/"capture_metadata.json",RESULTS/"screenshots"/"folium_map.png",screenshot,RESULTS/"screenshots"/"server_logs_browser.png",server) if path.is_file()]
    if (RESULTS/"api_integration.json").exists():source_files.append(RESULTS/"api_integration.json")
    if ablation is not None:source_files.append(RESULTS/"ablation.json")
    if refinement is not None:
        source_files.extend(REFINEMENT_RESULTS/f"{name}.json" for name,record in refinement.items() if record is not None)
        source_files.extend(path for path in (REFINEMENT_RESULTS/"screenshots/ui_desktop.png", REFINEMENT_RESULTS/"screenshots/capture_metadata.json", REFINEMENT_RESULTS/"integrity_audit.json") if path.is_file())
    manifest={
        "report":target.relative_to(ROOT).as_posix(),
        "generated_ist":now.isoformat(),
        "hashing":"SHA-256 with LF-normalized JSON and CSV, and original image bytes.",
        "sources":{p.relative_to(ROOT).as_posix():evidence_sha256(p) for p in source_files},
        "team_review_required":True,
        "second_laptop_demo":"pending_without_team_evidence",
    }
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
