"""Production inference: no reference loading or label-generation imports."""
from dataclasses import dataclass
from time import perf_counter
import joblib
import numpy as np
from .settings import ROOT,CANDIDATE_K,CANDIDATE_RADIUS,FEATURE_VERSION
from .geometry import load_network,load_observations
from .features import build_features,FEATURE_NAMES
from .scoring import positive_scores
from .decoder import decode,DecoderConfig,evaluate_continuity
from .metrics import select_top1

@dataclass
class Prediction:
    network:object
    candidates:object
    track:np.ndarray
    track_xy:np.ndarray
    probabilities:np.ndarray
    classifier_arcs:np.ndarray
    decoded:object
    motion:dict
    timing:dict
    model_name:str

def load_artifact(path=None):
    artifact=joblib.load(path or ROOT/'artifacts/final_model.joblib')
    if artifact['feature_names']!=FEATURE_NAMES or artifact['feature_version']!=FEATURE_VERSION:
        raise ValueError('Model and feature schema differ; retrain or use matching code')
    if artifact.get('candidate_k')!=CANDIDATE_K or artifact.get('candidate_radius_m')!=CANDIDATE_RADIUS:
        raise ValueError('Model and candidate configuration differ')
    return artifact

def predict_trip(trip_dir,artifact=None,decoder_config=None):
    start=perf_counter()
    artifact=artifact or load_artifact()
    network=load_network(trip_dir)
    track=load_observations(trip_dir)
    xy=network.project(track[:,:2])
    loaded=perf_counter()
    candidates=network.candidates(xy,k=CANDIDATE_K,radius=CANDIDATE_RADIUS)
    X,motion=build_features(network,candidates,xy,track[:,2])
    features_done=perf_counter()
    probabilities=np.zeros_like(candidates.distances,dtype=np.float32)
    if candidates.valid.any():probabilities[candidates.valid]=positive_scores(artifact['model'],X[candidates.valid])
    scored=perf_counter()
    classifier,_=select_top1(candidates.arc_ids,probabilities)
    result=decode(network,candidates,probabilities,xy,track[:,2],config=decoder_config)
    end=perf_counter()
    return Prediction(network,candidates,track,xy,probabilities,classifier,result,motion,{'graph_and_track_seconds':loaded-start,'feature_seconds':features_done-loaded,'classifier_seconds':scored-features_done,'decoder_seconds':end-scored,'inference_seconds':end-start},artifact['model_name'])
