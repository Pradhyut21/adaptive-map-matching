"""Laptop 1 FastAPI service. Logs real requests; references stay server-side."""
from contextlib import asynccontextmanager
from pathlib import Path
import html
import json
import logging
import os
import re
import threading
import time
from uuid import uuid4
import numpy as np
import shapely
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from pathsnap.settings import ROOT,DATA_ROOT,trip_dirs
from pathsnap.predict import load_artifact,predict_trip
from .evaluator import evaluate_prediction

logger=logging.getLogger('pathsnap.api')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler=logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logger.addHandler(handler)
logger.propagate=False
MODEL=None
MODEL_ERROR=None
LOCK=threading.Lock()
RUNTIME=ROOT/'results/runtime'
RUNTIME.mkdir(parents=True,exist_ok=True)

def partition_name():
    explicit=os.environ.get('PATHSNAP_PARTITION')
    if explicit:
        if explicit not in ('student_test','evaluator_test','evaluator_testing'):raise ValueError('Unsupported runtime partition')
        return explicit
    for p in ('evaluator_test','evaluator_testing'):
        if (DATA_ROOT/p).is_dir():return p
    return 'student_test'

@asynccontextmanager
async def lifespan(app):
    global MODEL,MODEL_ERROR
    try:MODEL=load_artifact();MODEL_ERROR=None
    except Exception as exc:MODEL_ERROR=str(exc);logger.warning('Model not ready: %s',exc)
    yield

app=FastAPI(title='PathSnap Trace-Lite',lifespan=lifespan)
# No authentication/credentials: allow the separate Laptop2 static origin.
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST'],allow_headers=['Content-Type'],allow_credentials=False)

class PredictionRequest(BaseModel):
    trip_id:str=Field(min_length=1,max_length=64,pattern=r'^[A-Za-z0-9_-]+$')

@app.get('/health')
def health():
    return {'status':'ok','model_ready':MODEL is not None,'model_name':MODEL['model_name'] if MODEL else None,'model_error':MODEL_ERROR,'partition':partition_name(),'trip_count':len(trip_dirs(partition_name()))}

@app.get('/trips')
def trips():
    partition=partition_name()
    records=[]
    for p in trip_dirs(partition):
        with (p/f'{p.name}.track').open() as stream:
            records.append({'id':p.name,'observations':sum(1 for line in stream if line.strip())})
    return {'partition':partition,'trips':records}

def geometry_payload(prediction):
    network=prediction.network
    xy=prediction.track_xy
    lo=xy.min(axis=0)-150.;hi=xy.max(axis=0)+150.
    ids=network.tree.query(shapely.box(lo[0],lo[1],hi[0],hi[1]))
    total=len(ids)
    if len(ids)>15000:ids=ids[np.linspace(0,len(ids)-1,15000,dtype=int)]
    roads={'type':'FeatureCollection','features':[{'type':'Feature','properties':{'arc_id':int(a)},'geometry':{'type':'LineString','coordinates':network.nodes_lonlat[network.arcs[a]].tolist()}} for a in ids]}
    routes={'type':'FeatureCollection','features':[{'type':'Feature','properties':{'segment':i},'geometry':{'type':'MultiLineString','coordinates':[network.nodes_lonlat[network.arcs[a]].tolist() for a in segment]}} for i,segment in enumerate(prediction.decoded.route_segments) if segment]}
    return roads,routes,{'road_arcs_in_extent':total,'road_arcs_displayed':len(ids),'road_layer_display_limit':15000}

@app.post('/predict')
def predict(request:PredictionRequest):
    global MODEL,MODEL_ERROR
    partition=partition_name()
    available={p.name:p for p in trip_dirs(partition)}
    if request.trip_id not in available:raise HTTPException(404,'Trip is not available in the runtime partition')
    with LOCK:
        if MODEL is None:
            try:MODEL=load_artifact();MODEL_ERROR=None
            except Exception as exc:MODEL_ERROR=str(exc);raise HTTPException(503,'Trained model not available; complete training first') from exc
        run_id=uuid4().hex[:12]
        begin=time.perf_counter()
        logger.info('run=%s POST /predict trip=%s partition=%s BEGIN',run_id,request.trip_id,partition)
        try:
            # This function has no reference reader. Evaluation starts only after
            # a complete production prediction has been computed.
            prediction=predict_trip(available[request.trip_id],MODEL)
            evaluated=time.perf_counter()
            reference_root=Path(os.environ.get('PATHSNAP_REFERENCE_ROOT',DATA_ROOT))
            reference_trip=reference_root/partition/request.trip_id
            if (reference_trip/f'{request.trip_id}.route').exists():
                metrics=evaluate_prediction(prediction,reference_trip)
            else:
                metrics={'prediction_coverage':float((prediction.decoded.arc_ids>=0).mean()),'reference_label_note':'Reference unavailable; accuracy metrics not computed.'}
            evaluation_seconds=time.perf_counter()-evaluated
            roads,routes,road_info=geometry_payload(prediction)
            sorted_p=np.sort(prediction.probabilities,axis=1)
            margins=sorted_p[:,-1]-sorted_p[:,-2] if sorted_p.shape[1]>1 else sorted_p[:,-1]
            warnings=prediction.decoded.warnings
            if road_info['road_arcs_in_extent']>road_info['road_arcs_displayed']:warnings=warnings+[{'point_index':None,'reason':'Road display capped; inference used the complete graph.'}]
            response={'run_id':run_id,'trip_id':request.trip_id,'partition':partition,'model_name':prediction.model_name,'raw_gps':prediction.track[:,:2].tolist(),'road_network':roads,'predicted_route':routes,'predictions':[int(a) if a>=0 else None for a in prediction.decoded.arc_ids],'metrics':metrics,'timing':prediction.timing|{'evaluation_seconds':evaluation_seconds,'request_seconds':time.perf_counter()-begin},'warnings':warnings,'diagnostics':{'candidate_count':prediction.candidates.valid.sum(axis=1).tolist(),'heading_reliability':prediction.motion['heading_reliability'].tolist(),'score_margin':margins.tolist(),**road_info}}
            evidence={'run_id':run_id,'trip_id':request.trip_id,'partition':partition,'model_name':prediction.model_name,'timing':response['timing'],'metrics':metrics,'warning_count':len(warnings),'prediction_observations':len(response['predictions']),'http_status':200}
            (RUNTIME/f'{run_id}.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
            logger.info('run=%s POST /predict trip=%s status=200 points=%d inference=%.3fs accuracy=%s warnings=%d',run_id,request.trip_id,len(response['predictions']),prediction.timing['inference_seconds'],metrics.get('top1_accuracy'),len(warnings))
            return response
        except HTTPException:raise
        except Exception as exc:
            logger.exception('run=%s trip=%s failed',run_id,request.trip_id)
            raise HTTPException(500,'Prediction failed; inspect the server request log') from exc

@app.get('/diagnostics/logs',response_class=HTMLResponse)
def logs():
    records=[]
    for p in sorted(RUNTIME.glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:20]:
        records.append(json.loads(p.read_text()))
    body=html.escape(json.dumps(records,indent=2))
    return '<html><head><title>PathSnap actual request records</title><style>body{background:#101b2c;color:#cceee8;font:15px monospace;padding:30px}pre{white-space:pre-wrap}</style></head><body><h1>Actual server request records</h1><p>Generated after live inference. Local integration evidence; second-laptop LAN demonstration pending.</p><pre>'+body+'</pre></body></html>'

app.mount('/',StaticFiles(directory=ROOT/'frontend',html=True),name='frontend')
