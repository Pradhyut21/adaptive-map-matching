"""Server-only evaluator, invoked AFTER production inference has returned.

No derived labels or reference geometry are included in HTTP responses.
"""
from pathsnap.alignment import load_reference,align_reference
from pathsnap.metrics import observation_metrics,lcs_f1
from pathsnap.decoder import evaluate_continuity
from pathsnap.route_alignment import align_ordered_route
import numpy as np

def evaluate_prediction(prediction,reference_trip_dir,ordered_alignment=False):
    reference=load_reference(reference_trip_dir,prediction.network)
    labels,_,alignment=align_reference(prediction.network,prediction.track_xy,prediction.track[:,2],reference)
    result=observation_metrics(labels,prediction.decoded.arc_ids,prediction.candidates.arc_ids)
    note='Aggregate metrics use approximate reference-route alignment.'
    if ordered_alignment:
        legacy_top1=result['top1_accuracy']
        labels,_,distances=align_ordered_route(prediction.network,prediction.track_xy,prediction.track[:,2],reference)
        result=observation_metrics(labels,prediction.decoded.arc_ids,prediction.candidates.arc_ids)
        result['legacy_top1_accuracy']=legacy_top1
        result['reference_alignment_over_60_m']=int((distances>60).sum())
        note='Top-1 uses monotone full-route alignment and includes every GPS observation.'
    result.update(lcs_f1(prediction.decoded.route_arc_ids,reference))
    choices=np.maximum(prediction.decoded.candidate_indices,0)
    fractions=prediction.candidates.fractions[np.arange(len(choices)),choices]
    continuity=evaluate_continuity(prediction.network,prediction.decoded.arc_ids,fractions,prediction.track_xy,prediction.track[:,2],config=prediction.decoded.stats['config'])
    result.update(continuity)
    result['reference_label_note']=note
    result['alignment_ambiguous_fraction']=alignment['ambiguous_fraction']
    return result
