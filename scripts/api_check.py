"""Exercise every actual runtime trip over HTTP; retain compact evidence."""
import argparse
from datetime import datetime,timezone
import json
import time
from urllib.request import Request,urlopen
from pathlib import Path
from pathsnap.settings import ROOT

FORBIDDEN={'labels','reference','reference_route','ground_truth','true_labels','reference_positions'}
def check_no_reference(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in FORBIDDEN:raise AssertionError(f'Reference disclosure: {key}')
            check_no_reference(item)
    elif isinstance(value,list):
        for item in value:check_no_reference(item)

def request(url,payload=None):
    data=json.dumps(payload).encode() if payload is not None else None
    with urlopen(Request(url,data=data,headers={'Content-Type':'application/json'}),timeout=300) as response:
        return response.status,json.load(response)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8000');args=parser.parse_args()
    url=args.url.rstrip('/')
    _,health=request(url+'/health')
    if not health['model_ready']:raise RuntimeError('Final model is not ready')
    _,inventory=request(url+'/trips')
    if len(inventory['trips'])!=10:raise AssertionError('Expected all10 runtime trips')
    records=[]
    offline_path=ROOT/'results/hmm_comparison.json'
    offline=json.loads(offline_path.read_text()) if offline_path.exists() and inventory['partition']=='student_test' else None
    expected={r['trip_id']:r for r in offline['hmm']['trips']} if offline else {}
    for trip in inventory['trips']:
        start=time.perf_counter();status,result=request(url+'/predict',{'trip_id':trip['id']})
        check_no_reference(result)
        if len(result['raw_gps'])!=len(result['predictions']) or len(result['predictions'])!=trip['observations']:raise AssertionError('Lost observation alignment')
        if result['predicted_route']['type']!='FeatureCollection':raise AssertionError('Missing route geometry')
        record={'trip_id':trip['id'],'run_id':result['run_id'],'http_status':status,'observations':len(result['predictions']),'metrics':result['metrics'],'timing':result['timing'],'http_seconds':time.perf_counter()-start,'warning_count':len(result['warnings']),'reference_disclosure_check':'passed'}
        if trip['id'] in expected:
            saved=expected[trip['id']]
            delta=abs(result['metrics']['top1_accuracy']-saved['top1_accuracy'])
            record['offline_top1_delta']=delta
            record['offline_sequence_lcs_f1_delta']=abs(result['metrics']['sequence_lcs_f1']-saved['sequence_lcs_f1'])
            if delta>1/max(trip['observations'],1):raise AssertionError('Live inference diverged from saved offline evaluation by more than one observation')
        records.append(record)
        print(f"HTTP{status} {trip['id']} n={record['observations']} Top1={result['metrics'].get('top1_accuracy')} inference={result['timing']['inference_seconds']:.2f}s run={result['run_id']}",flush=True)
    evidence={'started_utc':datetime.now(timezone.utc).isoformat(),'url':url,'partition':inventory['partition'],'health':health,'requests':records,'scope':'Genuine same-machine HTTP requests for all10student-testing trips. No claim of second-laptop LAN demonstration.','second_laptop_demonstrated':False,'all_passed':len(records)==10}
    (ROOT/'results/api_integration.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')

if __name__=='__main__':main()
