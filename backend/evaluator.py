"""Server-only evaluator, invoked AFTER production inference has returned.

No derived labels or reference geometry are included in HTTP responses.
"""
from pathsnap.alignment import load_reference,align_reference
from pathsnap.metrics import observation_metrics,lcs_f1
from pathsnap.decoder import evaluate_continuity
import numpy as np

def evaluate_prediction(prediction,reference_trip_dir):
    reference=load_reference(reference_trip_dir,prediction.network)
    labels,_,alignment=align_reference(prediction.network,prediction.track_xy,prediction.track[:,2],reference)
    result=observation_metrics(labels,prediction.decoded.arc_ids,prediction.candidates.arc_ids)
    result.update(lcs_f1(prediction.decoded.route_arc_ids,reference))
    choices=np.maximum(prediction.decoded.candidate_indices,0)
    fractions=prediction.candidates.fractions[np.arange(len(choices)),choices]
    continuity=evaluate_continuity(prediction.network,prediction.decoded.arc_ids,fractions,prediction.track_xy,prediction.track[:,2])
    result.update(continuity)
    result['reference_label_note']='Aggregate metrics use approximate reference-route alignment, not independently verified pointwise truth.'
    result['alignment_ambiguous_fraction']=alignment['ambiguous_fraction']
    return result
