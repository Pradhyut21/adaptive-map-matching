"""Persistent genuine TPE search after measured algorithm selection."""
import argparse
import json
import shutil
import time
from dataclasses import asdict
import optuna
from pathsnap.settings import ROOT,SEED
from pathsnap.learning import load_partition,DEFAULTS
from pathsnap.decoder import DecoderConfig
from scripts.train import cross_validate,fit_evaluate,dump

def suggest(trial,name):
    if name=='GaussianNB':return {'var_smoothing':trial.suggest_float('var_smoothing',1e-12,1e-4,log=True)}
    if name=='LogisticRegression':return {'C':trial.suggest_float('C',0.005,20.,log=True)}
    if name=='DecisionTree':return {'max_depth':trial.suggest_int('max_depth',7,20),'min_samples_leaf':trial.suggest_int('min_samples_leaf',5,80,log=True),'ccp_alpha':trial.suggest_float('ccp_alpha',1e-7,1e-3,log=True)}
    if name=='ExtraTrees':return {'n_estimators':trial.suggest_int('n_estimators',40,96,step=8),'max_depth':trial.suggest_int('max_depth',10,22),'min_samples_leaf':trial.suggest_int('min_samples_leaf',3,35,log=True),'max_features':trial.suggest_float('max_features',0.5,1.)}
    if name=='HistGradientBoosting':return {'max_iter':trial.suggest_int('max_iter',60,140,step=20),'max_leaf_nodes':trial.suggest_categorical('max_leaf_nodes',[7,15,31]),'learning_rate':trial.suggest_float('learning_rate',0.04,0.2,log=True),'l2_regularization':trial.suggest_float('l2_regularization',0.1,20.,log=True),'min_samples_leaf':trial.suggest_int('min_samples_leaf',20,80,step=10)}
    raise ValueError(name)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trials',type=int,default=8);args=parser.parse_args()
    selected=json.loads((ROOT/'results/selection.json').read_text())['selected_model']
    train=load_partition('train');test=load_partition('student_test')
    study=optuna.create_study(study_name=f'pathsnap-{selected}-grouped-top1',direction='maximize',storage='sqlite:///'+str(ROOT/'results/optuna.sqlite3').replace('\\','/'),load_if_exists=True,sampler=optuna.samplers.TPESampler(seed=SEED,n_startup_trials=3))
    start=time.perf_counter()
    def objective(trial):
        params=suggest(trial,selected)
        begin=time.perf_counter()
        cv,_=cross_validate(selected,train,params=params)
        trial.set_user_attr('elapsed_seconds',time.perf_counter()-begin)
        trial.set_user_attr('macro_f1',cv['metrics']['macro_f1'])
        trial.set_user_attr('cv',cv)
        dump(ROOT/'results/tuning_trials'/f'trial_{trial.number:03}.json',{'number':trial.number,'params':params,'cv':cv,'elapsed_seconds':time.perf_counter()-begin,'sampler_phase':'startup_random' if trial.number<3 else 'guided_TPE'})
        if trial.number==0:print(f'First trial {time.perf_counter()-begin:.1f}s; estimated {args.trials} trials {(time.perf_counter()-begin)*args.trials/60:.1f} minutes',flush=True)
        return cv['metrics']['fold_mean_top1']
    complete=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective,n_trials=max(0,args.trials-complete),gc_after_trial=True)
    study.trials_dataframe().to_csv(ROOT/'results/optuna_trials.csv',index=False)
    tuned=fit_evaluate(selected,train,test,study.best_params,tag=selected+'_tuned')
    tuned['cv']=study.best_trial.user_attrs['cv']
    dump(ROOT/'results/models'/f'{selected}_tuned.json',tuned)
    untuned=json.loads((ROOT/'results/models'/f'{selected}.json').read_text())
    # Select using grouped CV only; retain untuned if tuning actually regresses.
    use_tuned=tuned['cv']['metrics']['fold_mean_top1']>=untuned['cv']['metrics']['fold_mean_top1']
    tag=selected+'_tuned' if use_tuned else selected
    shutil.copy2(ROOT/'artifacts'/f'{tag}.joblib',ROOT/'artifacts/final_model.joblib')
    trials=[{'number':t.number,'value':t.value,'params':t.params,'state':t.state.name,'seconds':t.user_attrs.get('elapsed_seconds'),'phase':'startup_random' if t.number<3 else 'guided_TPE'} for t in study.trials]
    summary={'selected_model':selected,'objective':'mean three-fold GroupKFold observation-level Top-1 accuracy','sampler':'Optuna TPESampler','startup_trials':3,'seed':SEED,'trials':trials,'best_params':study.best_params,'best_objective':study.best_value,'untuned':untuned,'tuned':tuned,'final_tag':tag,'used_tuned_model':use_tuned,'elapsed_seconds':time.perf_counter()-start,'rationale':'Choose tuned only if grouped-CV Top-1 >= untuned; never select using evaluator data.'}
    dump(ROOT/'results/tuning_summary.json',summary)
    dump(ROOT/'artifacts/final_config.json',{'model_name':selected,'tag':tag,'feature_version':tuned.get('feature_version','trace-lite-v2'),'decoder':asdict(DecoderConfig()),'selected_by':'grouped-CV Top-1'})
    print('FINAL MODEL',tag,flush=True)

if __name__=='__main__':main()
