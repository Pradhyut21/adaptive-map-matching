"""All-ten-trip baseline, classifier and directed-HMM route evaluation."""
from dataclasses import asdict
import json
import time
import numpy as np
from pathsnap.settings import ROOT,trip_dirs
from pathsnap.cache import cache_files,read_cache
from pathsnap.geometry import load_network,CandidateSet
from pathsnap.metrics import observation_metrics,candidate_metrics,lcs_f1,select_top1
from pathsnap.decoder import decode,evaluate_continuity,DecoderConfig,DirectedRouter,_bound,_time_break,_router
from scripts.train import dump,comparison_csv

def reconstruct(network,arcs,fractions,xy,timestamps,cfg=None):
    cfg=cfg or DecoderConfig()
    router=_router(network,cfg)
    result=[];current=[]
    for t,a in enumerate(arcs):
        a=int(a)
        if a<0:
            if current:result.append(current);current=[]
            continue
        if t==0 or not current or arcs[t-1]<0 or _time_break(t,xy,timestamps,cfg):
            if current:result.append(current)
            current=[a];continue
        bound=_bound(float(np.linalg.norm(xy[t]-xy[t-1])),float(timestamps[t]-timestamps[t-1]),cfg)
        path=router.arc_path(int(arcs[t-1]),float(fractions[t-1]),a,float(fractions[t]),bound)
        if path is None:
            result.append(current);current=[a]
        else:current.extend(path[1:])
    if current:result.append(current)
    return result

def route_metrics(network,d,arcs,indices,route=None):
    fractions=d['fractions'][np.arange(len(arcs)),np.maximum(indices,0)]
    result=observation_metrics(d['labels'],arcs,d['arc_ids'])
    result.update(evaluate_continuity(network,arcs,fractions,d['track_xy'],d['timestamps']))
    if route is None:
        segments=reconstruct(network,arcs,fractions,d['track_xy'],d['timestamps'])
        route=[a for segment in segments for a in segment]
    result.update(lcs_f1(route,d['reference']))
    return result

def aggregate(trips):
    n=sum(t['observations'] for t in trips)
    eligible=sum(t['eligible_transitions'] for t in trips)
    predicted_len=sum(t['predicted_sequence_length'] for t in trips)
    reference_len=sum(t['reference_sequence_length'] for t in trips)
    return {'trips':len(trips),'observations':n,'top1_accuracy':sum(t['correct'] for t in trips)/n,'mean_trip_top1_accuracy':float(np.mean([t['top1_accuracy'] for t in trips])),'prediction_coverage':sum(t['prediction_coverage']*t['observations'] for t in trips)/n,'candidate_miss_rate':sum(t['candidate_miss_count'] for t in trips)/n,'route_continuity':sum(t['feasible_transitions'] for t in trips)/max(eligible,1),'eligible_transitions':eligible,'feasible_transitions':sum(t['feasible_transitions'] for t in trips),'unreachable_transitions':sum(t['unreachable_transitions'] for t in trips),'excluded_time_gaps':sum(t['excluded_time_gaps'] for t in trips),'sequence_lcs_f1':2*sum(t['sequence_lcs'] for t in trips)/max(predicted_len+reference_len,1),'mean_trip_sequence_lcs_f1':float(np.mean([t['sequence_lcs_f1'] for t in trips])),'inference_seconds':sum(t.get('inference_seconds',0) for t in trips)}

def main():
    comparison=json.loads((ROOT/'results/model_comparison.json').read_text())
    tuning=json.loads((ROOT/'results/tuning_summary.json').read_text())
    tag=tuning['final_tag']
    paths={p.name:p for p in trip_dirs('student_test')}
    methods={r['name']:[] for r in comparison['models']}
    if tag not in methods:methods[tag]=[]
    baseline=[];hmm=[]
    out=ROOT/'results/predictions/HMM';out.mkdir(parents=True,exist_ok=True)
    for path in cache_files('student_test'):
        start=time.perf_counter()
        d=read_cache(path)
        network=load_network(paths[path.stem])
        candidates=CandidateSet(d['arc_ids'],d['distances'],d['fractions'],d['projected_xy'])
        nearest=d['arc_ids'][:,0]
        metrics=route_metrics(network,d,nearest,np.zeros(len(nearest),dtype=int))
        baseline.append({'trip_id':path.stem,**metrics})
        finalp=None
        for name in methods:
            with np.load(ROOT/'results/predictions'/name/f'{path.stem}.npz') as z:
                arcs=z['predictions'];indices=z['candidate_indices'];p=z['probabilities']
            mm=route_metrics(network,d,arcs,indices)
            mm.update(candidate_metrics(d['y'][candidates.valid],p[candidates.valid]))
            methods[name].append({'trip_id':path.stem,**mm})
            if name==tag:finalp=p
        # Baseline/classifier continuity checks warmed routing caches. Reset
        # before the timed decoder so its wall time reflects a cold trip router.
        if hasattr(network,'_pathsnap_directed_router'):del network._pathsnap_directed_router
        begin=time.perf_counter()
        result=decode(network,candidates,finalp,d['track_xy'],d['timestamps'])
        inference=time.perf_counter()-begin
        mm=route_metrics(network,d,result.arc_ids,result.candidate_indices,result.route_arc_ids)
        mm.update(candidate_metrics(d['y'][candidates.valid],finalp[candidates.valid]))
        hmm.append({'trip_id':path.stem,**mm,'inference_seconds':inference,'restart_count':result.stats['restart_count'],'warnings':result.warnings,'decoder_stats':result.stats})
        np.savez_compressed(out/f'{path.stem}.npz',predictions=result.arc_ids,candidate_indices=result.candidate_indices,route_arc_ids=np.asarray(result.route_arc_ids,dtype=np.int32),probabilities=finalp)
        dump(out/f'{path.stem}.json',{'trip_id':path.stem,'route_segments':result.route_segments,'warnings':result.warnings,'metrics':mm,'decoder_stats':result.stats})
        print(f'{path.stem}: nearest={metrics["top1_accuracy"]:.4f} classifier={methods[tag][-1]["top1_accuracy"]:.4f} HMM={mm["top1_accuracy"]:.4f} continuity={mm["route_continuity"]:.4f} decoder={inference:.2f}s total={time.perf_counter()-start:.2f}s',flush=True)
    for r in comparison['models']:
        route=aggregate(methods[r['name']])
        r['student_test'].update({k:v for k,v in route.items() if k not in ['inference_seconds']})
        r['student_trips']=methods[r['name']]
        dump(ROOT/'results/models'/f'{r["name"]}.json',r)
    dump(ROOT/'results/model_comparison.json',comparison);comparison_csv(comparison['models'])
    output={'partition':'student_test','model_tag':tag,'config':asdict(DecoderConfig()),'baseline':{'aggregate':aggregate(baseline),'trips':baseline},'classifier':{'aggregate':aggregate(methods[tag]),'trips':methods[tag]},'hmm':{'aggregate':aggregate(hmm),'trips':hmm},'methodology':'Same candidates/reference-derived labels and directed shortest-path reconstruction for classifier and HMM. Candidate failures remain in Top-1; gaps excluded only from continuity as documented. HMM times are decoder-only from cached features; live API records full pipeline timing.'}
    dump(ROOT/'results/hmm_comparison.json',output)
    config=json.loads((ROOT/'artifacts/final_config.json').read_text())
    config['decoder']=asdict(DecoderConfig())
    dump(ROOT/'artifacts/final_config.json',config)
    print('HMM aggregate',output['hmm']['aggregate'],flush=True)

if __name__=='__main__':main()
