"""Exactly five supervised algorithm families; trip-grouped experiments."""
from dataclasses import dataclass
import time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from .settings import SEED
from .cache import cache_files,read_cache
from .metrics import candidate_metrics, observation_metrics

MODEL_NAMES=['GaussianNB','LogisticRegression','DecisionTree','ExtraTrees','HistGradientBoosting']
FAMILIES={'GaussianNB':'Probabilistic','LogisticRegression':'Linear','DecisionTree':'Decision tree','ExtraTrees':'Randomized tree ensemble','HistGradientBoosting':'Gradient boosting'}
DEFAULTS={
 'GaussianNB':{'var_smoothing':1e-8},
 'LogisticRegression':{'C':1.,'max_iter':250,'solver':'lbfgs','random_state':SEED},
 'DecisionTree':{'max_depth':14,'min_samples_leaf':20,'random_state':SEED},
 'ExtraTrees':{'n_estimators':64,'max_depth':16,'min_samples_leaf':8,'max_features':1.,'n_jobs':4,'random_state':SEED},
 'HistGradientBoosting':{'max_iter':90,'max_leaf_nodes':15,'learning_rate':0.1,'l2_regularization':1.,'min_samples_leaf':30,'early_stopping':False,'random_state':SEED},
}

def make_model(name,params=None):
    configuration=DEFAULTS[name]|(params or {})
    cls={'GaussianNB':GaussianNB,'LogisticRegression':LogisticRegression,'DecisionTree':DecisionTreeClassifier,'ExtraTrees':ExtraTreesClassifier,'HistGradientBoosting':HistGradientBoostingClassifier}[name]
    model=cls(**configuration)
    return make_pipeline(StandardScaler(),model) if name in ['GaussianNB','LogisticRegression'] else model

@dataclass
class PackedData:
    X:np.ndarray
    y:np.ndarray
    groups:np.ndarray
    row_obs:np.ndarray
    arcs:np.ndarray
    labels:np.ndarray
    obs_groups:np.ndarray
    ranks:np.ndarray
    trip_ids:list
    files:list

def load_partition(partition):
    xs=[];ys=[];groups=[];observations=[];arcs=[];labels=[];obs_groups=[];ranks=[]
    files=cache_files(partition)
    offset=0
    for group,path in enumerate(files):
        d=read_cache(path)
        valid=d['arc_ids']>=0
        n,k=valid.shape
        xs.append(d['X'][valid]);ys.append(d['y'][valid]);groups.append(np.full(valid.sum(),group,np.int16))
        observations.append(np.broadcast_to(np.arange(n)[:,None]+offset,(n,k))[valid])
        arcs.append(d['arc_ids'][valid]);labels.append(d['labels']);obs_groups.append(np.full(n,group,np.int16))
        ranks.append(np.broadcast_to(np.arange(k),(n,k))[valid].astype(np.int8))
        offset+=n
    if not files:raise ValueError(f'No cached {partition} trips; run scripts.prepare first')
    return PackedData(np.concatenate(xs),np.concatenate(ys),np.concatenate(groups),np.concatenate(observations).astype(np.int32),np.concatenate(arcs),np.concatenate(labels),np.concatenate(obs_groups),np.concatenate(ranks),[p.stem for p in files],files)

def training_sample(data,rows,seed=SEED):
    """Preserve all positives, closest two candidates, plus seeded 12% negatives.

    Only called on training-fold row indices; validation/test remain complete.
    """
    rng=np.random.default_rng(seed)
    keep=(data.y[rows]==1)|(data.ranks[rows]<2)|(rng.random(len(rows))<0.12)
    return rows[keep]

def positive_scores(model,X,batch_size=100000):
    result=np.empty(len(X),np.float32)
    index=int(np.flatnonzero(model.classes_==1)[0])
    for lo in range(0,len(X),batch_size):
        result[lo:lo+batch_size]=model.predict_proba(X[lo:lo+batch_size])[:,index]
    return result

def score_rows(data,rows,scores,observation_groups=None):
    obs=data.row_obs[rows]
    maximum=np.full(len(data.labels),-np.inf)
    np.maximum.at(maximum,obs,scores)
    winner=np.full(len(data.labels),len(rows),np.int64)
    local=np.arange(len(rows))
    possible=scores==maximum[obs]
    np.minimum.at(winner,obs[possible],local[possible])
    predictions=np.full(len(data.labels),-1,np.int32)
    valid=winner<len(rows)
    predictions[valid]=data.arcs[rows[winner[valid]]]
    obs_mask=np.ones(len(data.labels),bool) if observation_groups is None else np.isin(data.obs_groups,observation_groups)
    result=candidate_metrics(data.y[rows],scores)
    result.update(observation_metrics(data.labels[obs_mask],predictions[obs_mask]))
    found=np.zeros(len(data.labels),bool)
    np.logical_or.at(found,obs,data.y[rows].astype(bool))
    result['candidate_miss_rate']=float((~found[obs_mask]).mean())
    result['per_trip_top1']={data.trip_ids[g]:float(((predictions==data.labels)&(predictions>=0))[data.obs_groups==g].mean()) for g in np.unique(data.obs_groups[obs_mask])}
    result['mean_trip_top1_accuracy']=float(np.mean(list(result['per_trip_top1'].values())))
    return result,predictions
