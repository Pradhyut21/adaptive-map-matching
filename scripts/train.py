import argparse
import json
import csv
import platform
import subprocess
from datetime import datetime,timezone
import importlib.metadata
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from pathsnap.settings import ROOT,SEED,FEATURE_VERSION,CANDIDATE_K,CANDIDATE_RADIUS
from pathsnap.cache import read_cache
from pathsnap.features import FEATURE_NAMES
from pathsnap.learning import *
from pathsnap.metrics import METRIC_DEFINITIONS,select_top1,lcs_f1,collapse_arcs

RESULTS=ROOT/'results'

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=lambda a:a.item() if isinstance(a,np.generic) else str(a)),encoding='utf-8')

def cross_validate(name,data,params=None,record=False):
    folds=[]
    oof=np.zeros(len(data.y),np.float32)
    for fold,(train,val) in enumerate(GroupKFold(3).split(data.X,data.y,data.groups)):
        train_groups=np.unique(data.groups[train]);val_groups=np.unique(data.groups[val])
        if np.intersect1d(train_groups,val_groups).size:raise AssertionError('Trip leakage')
        sampled=training_sample(data,train,SEED+fold)
        model=make_model(name,params)
        start=time.perf_counter()
        with threadpool_limits(limits=4):model.fit(data.X[sampled],data.y[sampled])
        elapsed=time.perf_counter()-start
        start=time.perf_counter()
        with threadpool_limits(limits=4):scores=positive_scores(model,data.X[val])
        inference=time.perf_counter()-start
        oof[val]=scores
        metrics,_=score_rows(data,val,scores,val_groups)
        item={'fold':fold,'training_trips':[data.trip_ids[g] for g in train_groups],'validation_trips':[data.trip_ids[g] for g in val_groups],'training_rows_before_sampling':len(train),'training_rows_after_sampling':len(sampled),'validation_rows':len(val),'training_seconds':elapsed,'inference_seconds':inference,**metrics}
        folds.append(item)
        if record:dump(RESULTS/'cv'/f'{name}_fold{fold}.json',item)
        print(f'{name} fold{fold}: Top1={metrics["top1_accuracy"]:.5f}, F1={metrics["macro_f1"]:.5f}, fit={elapsed:.1f}s, predict={inference:.1f}s',flush=True)
    combined,_=score_rows(data,np.arange(len(data.y)),oof)
    combined['fold_mean_top1']=float(np.mean([f['top1_accuracy'] for f in folds]))
    combined['fold_std_top1']=float(np.std([f['top1_accuracy'] for f in folds]))
    combined['fold_mean_macro_f1']=float(np.mean([f['macro_f1'] for f in folds]))
    return {'metrics':combined,'folds':folds},oof

def fit_evaluate(name,train,test,params=None,tag=None):
    tag=tag or name
    rows=np.arange(len(train.y))
    sampled=training_sample(train,rows)
    model=make_model(name,params)
    start=time.perf_counter()
    with threadpool_limits(limits=4):model.fit(train.X[sampled],train.y[sampled])
    fit_seconds=time.perf_counter()-start
    with threadpool_limits(limits=4):train_scores=positive_scores(model,train.X)
    train_metrics,_=score_rows(train,rows,train_scores)
    start=time.perf_counter()
    with threadpool_limits(limits=4):scores=positive_scores(model,test.X)
    inference=time.perf_counter()-start
    metrics,predictions=score_rows(test,np.arange(len(test.y)),scores)
    artifact={'model':model,'model_name':name,'params':DEFAULTS[name]|(params or {}),'feature_names':FEATURE_NAMES,'feature_version':FEATURE_VERSION,'candidate_k':CANDIDATE_K,'candidate_radius_m':CANDIDATE_RADIUS,'seed':SEED,'negative_sampling':'all positives, rank<2, seeded12percent other negatives','training_trip_ids':train.trip_ids}
    joblib.dump(artifact,ROOT/'artifacts'/f'{tag}.joblib',compress=3)
    target=RESULTS/'predictions'/tag;target.mkdir(parents=True,exist_ok=True)
    per_trip=[]
    for group,path in enumerate(test.files):
        d=read_cache(path)
        valid=d['arc_ids']>=0
        p=np.zeros_like(d['distances'],dtype=np.float32)
        p[valid]=scores[test.groups==group]
        pred,indices=select_top1(d['arc_ids'],p)
        np.savez_compressed(target/f'{path.stem}.npz',probabilities=p,predictions=pred,candidate_indices=indices)
        mm=observation_metrics(d['labels'],pred,d['arc_ids'])
        mm.update(lcs_f1(collapse_arcs(pred),d['reference']))
        per_trip.append({'trip_id':path.stem,**mm})
    result={'name':name,'tag':tag,'algorithm_family':FAMILIES[name],'params':DEFAULTS[name]|(params or {}),'training_metrics':train_metrics,'student_test':metrics,'student_trips':per_trip,'training_seconds':fit_seconds,'inference_seconds':inference,'training_rows_before_sampling':len(rows),'training_rows_after_sampling':len(sampled),'inference_timing_scope':'classifier scoring only from cached features; full pipeline timed separately'}
    dump(RESULTS/'models'/f'{tag}.json',result)
    return result

def comparison_csv(records):
    columns=['Model','Algorithm Family','Key Configuration','CV Macro F1-score','Top-1 Accuracy','Route continuity','Student-Test Macro F1-score','Log Loss','Inference Time']
    with (RESULTS/'model_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in records:
            writer.writerow(dict(zip(columns,[r['name'],r['algorithm_family'],json.dumps(r['params']),r['cv']['metrics']['macro_f1'],r['student_test']['top1_accuracy'],r['student_test'].get('route_continuity','Pending routed evaluation'),r['student_test']['macro_f1'],r['student_test']['log_loss'],r['inference_seconds']])))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    start=time.perf_counter()
    train=load_partition('train');test=load_partition('student_test')
    if not args.smoke and (len(train.trip_ids)!=80 or len(test.trip_ids)!=10):
        raise ValueError('Full run requires cached80train/10student trips; finish scripts.prepare first')
    dump(RESULTS/'metric_definitions.json',METRIC_DEFINITIONS)
    if args.smoke:
        rows=np.flatnonzero(train.groups==0)
        model=make_model('ExtraTrees',{'n_estimators':16})
        sampled=training_sample(train,rows)
        with threadpool_limits(limits=4):
            model.fit(train.X[sampled],train.y[sampled]);p=positive_scores(model,test.X)
        metrics,_=score_rows(test,np.arange(len(test.y)),p)
        dump(RESULTS/'smoke_training.json',{'seconds':time.perf_counter()-start,'training_trips':[train.trip_ids[0]],'student_metrics':metrics,'purpose':'runtime and API smoke only, excluded from official model comparison'})
        print('Smoke:',metrics,'seconds',time.perf_counter()-start,flush=True);return
    dump(RESULTS/'execution_manifest.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'platform':platform.platform(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'seed':SEED,'feature_version':FEATURE_VERSION,'candidate_k':CANDIDATE_K,'radius_m':CANDIDATE_RADIUS,'training_trips':train.trip_ids,'student_test_trips':test.trip_ids,'versions':{m:importlib.metadata.version(m) for m in ['numpy','scipy','scikit-learn','shapely','pyproj','optuna','joblib']}})
    records=[]
    for name in MODEL_NAMES:
        cv,_=cross_validate(name,train,record=True)
        r=fit_evaluate(name,train,test);r['cv']=cv
        dump(RESULTS/'models'/f'{name}.json',r)
        records.append(r)
        dump(RESULTS/'model_comparison.json',{'models':records,'seed':SEED,'selection_metric':'mean grouped-CV observation-level Top-1 accuracy','complete':len(records)==5})
        comparison_csv(records)
        print('SAVED',name,flush=True)
    chosen=max(records,key=lambda r:(r['cv']['metrics']['fold_mean_top1'],r['cv']['metrics']['macro_f1'],-r['cv']['metrics']['log_loss']))
    dump(RESULTS/'selection.json',{'selected_model':chosen['name'],'reason':'Highest mean three-fold trip-grouped validation Top-1 accuracy; candidate F1 and log loss as tie-breakers. Student-test labels did not choose classifier.','cv_top1':chosen['cv']['metrics']['fold_mean_top1'],'elapsed_seconds':time.perf_counter()-start})
    print('SELECTED',chosen['name'],flush=True)

if __name__=='__main__':main()
