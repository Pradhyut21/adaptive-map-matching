"""Metrics have explicit populations; candidate misses stay Top-1 failures."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, log_loss, confusion_matrix

def candidate_metrics(y, scores):
    y = np.asarray(y,dtype=np.int32).ravel()
    scores = np.clip(np.asarray(scores,dtype=float).ravel(),1e-7,1-1e-7)
    if not len(y):
        return {"candidate_count":0,"macro_f1":0.,"precision":0.,"recall":0.,"log_loss":None,"confusion_matrix":[[0,0],[0,0]]}
    pred = scores>=0.5
    p,r,f,_ = precision_recall_fscore_support(y,pred,labels=[0,1],average="macro",zero_division=0)
    return {"candidate_count":len(y),"macro_f1":float(f),"precision":float(p),"recall":float(r),"log_loss":float(log_loss(y,scores,labels=[0,1])),"confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}

def select_top1(arc_ids,scores):
    masked = np.where(arc_ids>=0,scores,-np.inf)
    indices = np.argmax(masked,axis=1)
    pred = arc_ids[np.arange(len(arc_ids)),indices].copy()
    pred[~np.isfinite(masked.max(axis=1))] = -1
    return pred, indices

def collapse_arcs(arcs):
    result=[]
    previous=None
    for a in arcs:
        a=int(a)
        if a>=0 and a!=previous:result.append(a)
        previous=a
    return result

def lcs_f1(predicted,reference):
    """Exact bitset LCS, F1 = 2*LCS / (predicted length + reference length)."""
    predicted=[int(a) for a in predicted if a>=0]
    reference=[int(a) for a in reference if a>=0]
    masks={}
    for i,a in enumerate(reference):masks[a]=masks.get(a,0)|(1<<i)
    state=0
    for a in predicted:
        x=state|masks.get(a,0)
        state=x & ~(x-((state<<1)|1))
    lcs=state.bit_count()
    return {"sequence_lcs":lcs,"sequence_lcs_f1":2*lcs/max(len(predicted)+len(reference),1),"predicted_sequence_length":len(predicted),"reference_sequence_length":len(reference)}

def observation_metrics(labels,predictions,arc_ids=None):
    labels=np.asarray(labels)
    predictions=np.asarray(predictions)
    if labels.shape!=predictions.shape:raise ValueError("One prediction per observation is required")
    n=len(labels)
    result={"observations":n,"correct":int(((labels==predictions)&(predictions>=0)).sum()),"top1_accuracy":float(((labels==predictions)&(predictions>=0)).sum()/max(n,1)),"prediction_coverage":float((predictions>=0).sum()/max(n,1))}
    if arc_ids is not None:
        present=(arc_ids==labels[:,None]).any(axis=1)
        result.update(candidate_miss_count=int((~present).sum()),candidate_miss_rate=float((~present).sum()/max(n,1)))
    return result

METRIC_DEFINITIONS={
"top1_accuracy":"Correct nonmissing predicted directed arc / all GPS observations, including candidate misses and decoder failures; labels are derived reference labels.",
"candidate_macro_f1":"Unweighted mean of negative/positive candidate-class F1 at probability threshold 0.5 over all generated valid candidate rows.",
"precision_recall":"Macro average over candidate classes 0 and 1 with zero_division=0; all generated candidate rows retained at validation/test.",
"log_loss":"Binary candidate log loss on all valid generated rows, probabilities clamped to [1e-7,1-1e-7]. Missing reference candidates are reported separately.",
"candidate_miss_rate":"GPS observations whose derived reference arc is absent from generated candidates / all observations.",
"prediction_coverage":"Observations with nonmissing predictions / all observations.",
"sequence_lcs_f1":"2*exact ordered arc LCS length / (predicted reconstructed arc sequence length + reference route length). A diagnostic, not an official evaluator formula.",
"route_continuity":"Directed reachable transitions within documented time/speed/search bounds / eligible consecutive observation pairs; exogenous invalid/large time gaps excluded, missing or unreachable predictions fail. Always report denominator.",
"labels":"Pointwise labels are approximate monotone alignment to supplied reference arc sequences; no assertion of independently verified physical truth."
}
