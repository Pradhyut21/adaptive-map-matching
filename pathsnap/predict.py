"""Production inference: no reference loading or label-generation imports."""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import os
import joblib
import json
import numpy as np
from .settings import ROOT,CANDIDATE_K,CANDIDATE_RADIUS,FEATURE_VERSION
from .geometry import load_network,load_observations
from .features import build_features,FEATURE_NAMES
from .scoring import positive_scores
from .decoder import decode,DecoderConfig,evaluate_continuity
from .metrics import select_top1
from . import refined_inputs
from .decoder_refinement import decode_refined

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
    base_dir = ROOT / 'artifacts'
    if path is not None:
        artifact_path = Path(path)
    elif os.environ.get('PATHSNAP_MODEL'):
        artifact_path = Path(os.environ['PATHSNAP_MODEL'])
    elif (base_dir / 'active_model.json').exists():
        filename = json.loads((base_dir / 'active_model.json').read_text(encoding='utf-8'))['file']
        if Path(filename).name != filename:
            raise ValueError('Active model must be a file directly under artifacts')
        artifact_path = base_dir / filename
    else:
        artifact_path = base_dir / 'final_model.joblib'
    artifact = joblib.load(artifact_path)
    if artifact['feature_version'] == refined_inputs.VERSION:
        names, k, radius = refined_inputs.FEATURE_NAMES, refined_inputs.K, refined_inputs.RADIUS
        config_path = artifact_path.with_suffix('.config.json')
        # Model selection freezes the decoder too. Missing settings must not
        # silently change the deployed predictions to default parameters.
        settings = json.loads(config_path.read_text(encoding='utf-8'))
        if not isinstance(settings.get('decoder'), dict) or not isinstance(settings.get('refinement'), dict):
            raise ValueError('Context model requires its frozen decoder and refinement settings')
        artifact['decoder_config'] = settings['decoder']
        artifact['decoder_refinement'] = settings['refinement']
    elif artifact['feature_version'] == FEATURE_VERSION:
        names, k, radius = FEATURE_NAMES, CANDIDATE_K, CANDIDATE_RADIUS
        config_path = base_dir / 'final_config.json'
        if config_path.exists():
            artifact['decoder_config'] = json.loads(config_path.read_text(encoding='utf-8')).get('decoder')
    else:
        raise ValueError('Unknown feature version in model artifact')
    if artifact['feature_names'] != names:
        raise ValueError('Model and feature schema differ; retrain or use matching code')
    if artifact.get('candidate_k') != k or artifact.get('candidate_radius_m') != radius:
        raise ValueError('Model and candidate configuration differ')
    return artifact

def predict_trip(trip_dir,artifact=None,decoder_config=None):
    start=perf_counter()
    artifact=artifact or load_artifact()
    network=load_network(trip_dir)
    track=load_observations(trip_dir)
    xy=network.project(track[:,:2])
    loaded=perf_counter()
    contextual = artifact.get('feature_version') == refined_inputs.VERSION
    if contextual:
        candidates, X, motion = refined_inputs.context_inputs(network, xy, track[:,2])
    else:
        candidates = network.candidates(xy, k=CANDIDATE_K, radius=CANDIDATE_RADIUS)
        X, motion = build_features(network, candidates, xy, track[:,2])
    if artifact.get('zeroed_feature_indices'):X[...,artifact['zeroed_feature_indices']]=0.
    features_done=perf_counter()
    probabilities=np.zeros_like(candidates.distances,dtype=np.float32)
    if candidates.valid.any():probabilities[candidates.valid]=positive_scores(artifact['model'],X[candidates.valid])
    scored=perf_counter()
    classifier,_=select_top1(candidates.arc_ids,probabilities)
    config = decoder_config or artifact.get('decoder_config')
    if contextual:
        result = decode_refined(network, candidates, probabilities, xy, track[:,2],
                                config=config, refinement=artifact.get('decoder_refinement'))
    else:
        result = decode(network, candidates, probabilities, xy, track[:,2], config=config)
    end=perf_counter()
    return Prediction(network,candidates,track,xy,probabilities,classifier,result,motion,{'graph_and_track_seconds':loaded-start,'feature_seconds':features_done-loaded,'classifier_seconds':scored-features_done,'decoder_seconds':end-scored,'inference_seconds':end-start},artifact['model_name'])
