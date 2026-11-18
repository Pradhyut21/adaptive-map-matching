"""Build features once; references are used solely in derived labels."""
import json
import time
from pathlib import Path
import hashlib
import numpy as np
from .settings import CACHE_ROOT, FEATURE_VERSION, CANDIDATE_K, CANDIDATE_RADIUS
from .geometry import load_observations, load_network
from .features import build_features, FEATURE_NAMES
from .alignment import load_reference, align_reference

def build_trip(trip_dir, partition, force=False):
    p=Path(trip_dir)
    directory=CACHE_ROOT/FEATURE_VERSION/partition
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/f"{p.name}.npz"
    metadata=target.with_suffix('.json')
    signature=hashlib.sha256(b''.join((p/f'{p.name}.{ext}').read_bytes() for ext in ['track','nodes','arcs','route'])).hexdigest()
    configuration={'k':CANDIDATE_K,'radius':CANDIDATE_RADIUS,'feature_version':FEATURE_VERSION}
    code_hash=hashlib.sha256(b''.join((Path(__file__).parent/name).read_bytes() for name in ['geometry.py','features.py','alignment.py','settings.py'])).hexdigest()
    if target.exists() and metadata.exists() and not force:
        old=json.loads(metadata.read_text())
        if old.get('source_sha256')==signature and old.get('configuration')==configuration and old.get('code_sha256')==code_hash:return target,old
    start=time.perf_counter()
    track=load_observations(p)
    network=load_network(p)
    xy=network.project(track[:,:2])
    candidates=network.candidates(xy,k=CANDIDATE_K,radius=CANDIDATE_RADIUS)
    X,motion=build_features(network,candidates,xy,track[:,2])
    # Deliberately only now open the reference, after inference inputs exist.
    reference=load_reference(p,network)
    labels,positions,alignment=align_reference(network,xy,track[:,2],reference)
    y=(candidates.arc_ids==labels[:,None]).astype(np.int8)
    valid=candidates.valid
    np.savez(target,X=X,y=y,labels=labels,reference_positions=positions,reference=reference,arc_ids=candidates.arc_ids,distances=candidates.distances,fractions=candidates.fractions,projected_xy=candidates.xy,track_xy=xy,timestamps=track[:,2],lonlat=track[:,:2],heading_reliability=motion['heading_reliability'])
    info={'trip_id':p.name,'partition':partition,'observations':len(track),'candidate_rows':int(valid.sum()),'candidate_miss_rate':float((~y.any(axis=1)).mean()),'no_candidate_count':int((~valid.any(axis=1)).sum()),'seconds':time.perf_counter()-start,'source_sha256':signature,'graph_id':network.graph_id,'feature_version':FEATURE_VERSION,'feature_names':FEATURE_NAMES,'candidate_k':CANDIDATE_K,'radius_m':CANDIDATE_RADIUS,'alignment':alignment}
    info.update(configuration=configuration,code_sha256=code_hash)
    metadata.write_text(json.dumps(info,indent=2),encoding='utf-8')
    return target,info

def cache_files(partition):
    return sorted((CACHE_ROOT/FEATURE_VERSION/partition).glob('*.npz'))

def read_cache(path):
    with np.load(path) as z:return {key:z[key] for key in z.files}
