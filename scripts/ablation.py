"""One controlled feature ablation using the already selected algorithm.

This is not a sixth algorithm: only the two motion-reliability feature columns
are removed, keeping trips, sampling, hyperparameters and scoring fixed.
"""
from dataclasses import replace
import json
from time import perf_counter
import numpy as np
import joblib
from pathsnap.settings import ROOT
from pathsnap.features import FEATURE_NAMES
from pathsnap.learning import load_partition
from scripts.train import cross_validate,fit_evaluate,dump

def main():
    start=perf_counter()
    summary=json.loads((ROOT/'results/tuning_summary.json').read_text())
    full=summary['tuned'] if summary['used_tuned_model'] else summary['untuned']
    name=summary['selected_model'];params=full['params']
    train=load_partition('train');test=load_partition('student_test')
    masked=[FEATURE_NAMES.index(c) for c in ['reliable_heading_difference','heading_reliability']]
    trainX=train.X.copy();testX=test.X.copy()
    trainX[:,masked]=0.;testX[:,masked]=0.
    train=replace(train,X=trainX);test=replace(test,X=testX)
    cv,_=cross_validate(name,train,params)
    result=fit_evaluate(name,train,test,params,tag=name+'_without_motion_reliability')
    result['cv']=cv
    path=ROOT/'artifacts'/f'{name}_without_motion_reliability.joblib'
    artifact=joblib.load(path)
    artifact['zeroed_feature_indices']=masked
    artifact['feature_variant']='without explicit motion-reliability columns'
    joblib.dump(artifact,path,compress=3)
    dump(ROOT/'results/ablation.json',{'algorithm':name,'fixed_params':params,'removed_features':[FEATURE_NAMES[i] for i in masked],'full_features':full,'without_reliability':result,'elapsed_seconds':perf_counter()-start,'selection_effect':'None; final model selection frozen before this diagnostic. This is a feature variant of the selected algorithm, not a sixth algorithm.','hypothesis':'Do explicit motion reliability inputs improve trip-held-out local candidate selection?','limitations':'Labels remain approximate; this isolates the two explicit reliability columns, not heading smoothing or sequence decoding.'})
    print('ABLATION completed',flush=True)

if __name__=='__main__':main()
