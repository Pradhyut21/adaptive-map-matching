import argparse
import json
import time
from pathsnap.settings import ROOT,trip_dirs
from pathsnap.cache import build_trip

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    records=[]
    started=time.perf_counter()
    for partition in ['train','student_test']:
        trips=trip_dirs(partition)
        if args.smoke:trips=trips[:1]
        for i,trip in enumerate(trips):
            path,info=build_trip(trip,partition,force=args.force)
            records.append(info)
            print(f"{partition} {i+1}/{len(trips)} {trip.name}: {info['observations']} points, miss={info['candidate_miss_rate']:.3%}, {info['seconds']:.2f}s",flush=True)
    filename='smoke_prepare.json' if args.smoke else 'feature_inventory.json'
    (ROOT/'results'/filename).write_text(json.dumps({'trips':records,'elapsed_seconds':time.perf_counter()-started},indent=2),encoding='utf-8')

if __name__=='__main__':main()
