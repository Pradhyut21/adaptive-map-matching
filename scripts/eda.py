"""Inspect every permitted trip and retain measured tables, plots and maps.

Run: python -m scripts.eda
No evaluator partition is inspected by this script.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathsnap.settings import ROOT, DATA_ROOT, trip_dirs

OUT = ROOT / "results" / "eda"
COLORS = {"train":"#087f8c", "student_test":"#cb6c36"}

def _save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def inspect_trip(directory, partition):
    p = Path(directory)
    files = {s:p/f"{p.name}.{s}" for s in ("track","nodes","arcs","route")}
    missing = [s for s,f in files.items() if not f.exists()]
    if missing:
        return {"partition":partition,"trip_id":p.name,"missing_files":";".join(missing)}
    arrays = {s:np.loadtxt(f, ndmin=2 if s!="route" else 1) for s,f in files.items()}
    track,nodes,arcs,route = (arrays[s] for s in ("track","nodes","arcs","route"))
    row = {"partition":partition,"trip_id":p.name,"missing_files":""}
    for s,a in arrays.items():
        row[f"{s}_rows"] = len(a)
        row[f"{s}_bytes"] = files[s].stat().st_size
        row[f"{s}_nonfinite_values"] = int((~np.isfinite(a)).sum())
        row[f"{s}_sha256"] = hashlib.sha256(files[s].read_bytes()).hexdigest()
    row["track_columns"] = track.shape[1]
    row["nodes_columns"] = nodes.shape[1]
    row["arcs_columns"] = arcs.shape[1]
    row["invalid_track_coordinates"] = int(((np.abs(track[:,0])>180)|(np.abs(track[:,1])>90)).sum())
    row["invalid_node_coordinates"] = int(((np.abs(nodes[:,0])>180)|(np.abs(nodes[:,1])>90)).sum())
    invalid_arcs = (~np.isfinite(arcs)).any(axis=1) | (arcs<0).any(axis=1) | (arcs>=len(nodes)).any(axis=1) | (arcs!=np.floor(arcs)).any(axis=1)
    row["invalid_arc_rows"] = int(invalid_arcs.sum())
    invalid_route = (~np.isfinite(route))|(route<0)|(route>=len(arcs))|(route!=np.floor(route))
    row["invalid_route_rows"] = int(invalid_route.sum())
    d = np.diff(track[:,2])
    row.update(time_steps=len(d),nonpositive_time_steps=int((d<=0).sum()),one_second_steps=int((d==1).sum()),gaps_over_60s=int((d>60).sum()),max_gap_seconds=float(d.max()) if len(d) else 0.,median_gap_seconds=float(np.median(d)) if len(d) else 0.)
    for col, axis in (("longitude",0),("latitude",1)):
        row[f"gps_min_{col}"]=float(track[:,axis].min())
        row[f"gps_max_{col}"]=float(track[:,axis].max())
    a = arcs[~invalid_arcs].astype(np.int32)
    graph = csr_matrix((np.ones(len(a),dtype=np.int8),(a[:,0],a[:,1])),shape=(len(nodes),len(nodes)))
    row["weak_components"], weak_labels = connected_components(graph,directed=True,connection="weak")
    row["strong_components"], _ = connected_components(graph,directed=True,connection="strong")
    degree = np.bincount(a.ravel(),minlength=len(nodes))
    row["isolated_nodes"] = int((degree==0).sum())
    row["largest_weak_component_fraction"] = float(np.bincount(weak_labels).max()/max(len(nodes),1))
    row["self_loop_arcs"] = int((a[:,0]==a[:,1]).sum())
    if not invalid_route.any() and len(route)>1:
        r=arcs[route.astype(np.int32)].astype(np.int32)
        row["nonadjacent_reference_steps"] = int((r[:-1,1]!=r[1:,0]).sum())
        row["reference_steps"] = len(route)-1
        row["repeated_reference_arcs"] = int(len(route)-len(np.unique(route)))
    return row

def summary_inventory(df):
    summary={"scope":"All 80 train and 10 student_test trips; no evaluator files inspected.","partitions":{},"metric_notes":{"node_arc_totals":"Summed per-trip rows, not unique global map objects.","nonadjacent_reference_steps":"Successive listed reference arcs with unequal end/start nodes; this is a file-level diagnostic, not a claim that reference routes are physically incorrect.","connectivity":"Directed road graph: weak/strong connected-component counts include isolated nodes."}}
    for partition,g in df.groupby("partition",sort=False):
        p={"trips":len(g),"observations":int(g.track_rows.sum()),"nodes_summed":int(g.nodes_rows.sum()),"arcs_summed":int(g.arcs_rows.sum()),"route_rows_summed":int(g.route_rows.sum()),"max_gap_seconds":float(g.max_gap_seconds.max()),"time_steps":int(g.time_steps.sum()),"one_second_fraction":float(g.one_second_steps.sum()/max(g.time_steps.sum(),1)),"gaps_over_60s":int(g.gaps_over_60s.sum())}
        for field in ["track_rows","nodes_rows","arcs_rows","route_rows","weak_components","strong_components","isolated_nodes","largest_weak_component_fraction","repeated_reference_arcs"]:
            p[field]={"minimum":float(g[field].min()),"median":float(g[field].median()),"maximum":float(g[field].max())}
        for field in ["nonpositive_time_steps","invalid_track_coordinates","invalid_node_coordinates","invalid_arc_rows","invalid_route_rows","nonadjacent_reference_steps","reference_steps"]+[f"{s}_nonfinite_values" for s in ("track","nodes","arcs","route")]:
            p[field]=int(g[field].sum())
        p["trips_with_missing_files"]=int((g.missing_files.fillna("")!="").sum())
        summary["partitions"][partition]=p
    return summary

def plot_inventory(df):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold"})
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for p,g in df.groupby("partition",sort=False):
        axs[0].hist(g.track_rows,bins=18,alpha=.65,color=COLORS[p],label=p)
        axs[1].scatter(g.nodes_rows,g.arcs_rows,s=25,alpha=.7,color=COLORS[p],label=p)
    axs[0].set(xlabel="GPS observations per trip",ylabel="Trips",title="Trajectory sizes")
    axs[1].set(xlabel="Nodes per trip",ylabel="Directed arcs per trip",title="Road graph sizes",xscale="log",yscale="log")
    for ax in axs:ax.legend();ax.grid(alpha=.15)
    _save(fig,"trip_and_graph_counts.png")
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for p,g in df.groupby("partition",sort=False):
        axs[0].scatter(np.arange(len(g)),g.max_gap_seconds,s=22,label=p,color=COLORS[p])
        axs[1].scatter(g.weak_components,g.largest_weak_component_fraction,s=25,alpha=.7,label=p,color=COLORS[p])
    axs[0].set(xlabel="Trip index within partition",ylabel="Largest timestamp gap (seconds)",yscale="log",title="Irregular observation gaps")
    axs[1].set(xlabel="Weak connected components",ylabel="Fraction of nodes in largest component",title="Graph connectivity")
    for ax in axs:ax.legend();ax.grid(alpha=.15)
    _save(fig,"gaps_and_connectivity.png")
    checks=["track_nonfinite_values","nodes_nonfinite_values","arcs_nonfinite_values","route_nonfinite_values","invalid_arc_rows","invalid_route_rows"]
    fig,ax=plt.subplots(figsize=(9,3))
    values=[int(df[c].sum()) for c in checks]
    ax.barh([c.replace("_"," ") for c in checks],values,color="#087f8c")
    for i,v in enumerate(values):ax.text(v+.02,i,str(v),va="center")
    ax.set(xlabel="Measured count over permitted partitions",title="Missing / invalid numeric values",xlim=(0,max(values+[1])*1.15))
    _save(fig,"missing_values.png")

def representative_map(trip_dir, predictions=None, route_segments=None):
    """Permitted student-test reference overlay; never served as evaluator UI."""
    from pathsnap.geometry import load_network,load_observations
    import folium
    network=load_network(trip_dir)
    track=load_observations(trip_dir)
    xy=network.project(track[:,:2])
    arcs=network.arcs
    reference=np.loadtxt(Path(trip_dir)/f"{Path(trip_dir).name}.route",dtype=np.int32,ndmin=1)
    center=[float(np.mean(track[:,1])),float(np.mean(track[:,0]))]
    m=folium.Map(location=center,zoom_start=13,tiles=None,control_scale=True)
    # Deliberately self-contained geometry; generated Folium HTML still requires
    # Leaflet JS/CSS unless vendored by the packaging step.
    road=folium.FeatureGroup(name="Road network",show=True)
    pad=.002
    lo=track[:,:2].min(axis=0)-pad;hi=track[:,:2].max(axis=0)+pad
    ends=network.nodes_lonlat[arcs]
    near=((ends.max(axis=1)>=lo)&(ends.min(axis=1)<=hi)).all(axis=1)
    road_ids=np.flatnonzero(near)
    road_coords=ends[road_ids][:,:,::-1].tolist()
    folium.PolyLine(road_coords,color="#a4aeb5",weight=1,opacity=.5).add_to(road)
    road.add_to(m)
    gps=folium.FeatureGroup(name="Raw GPS observations",show=True)
    for i,(lon,lat,_) in enumerate(track):
        folium.CircleMarker([lat,lon],radius=2,color="#3182ce",fill=True,fill_opacity=.7,weight=0,tooltip=f"GPS {i}").add_to(gps)
    ref=folium.FeatureGroup(name="Supplied reference route (student-test only)",show=True)
    folium.PolyLine(ends[reference][:,:,::-1].tolist(),color="#e09124",weight=5,opacity=.7).add_to(ref)
    ref.add_to(m)
    if predictions is not None:
        pred=folium.FeatureGroup(name="Predicted route",show=True)
        segs=route_segments if route_segments is not None else [[int(a)] for a in predictions if a>=0]
        for segment in segs:
            if segment:
                folium.PolyLine(ends[np.asarray(segment,dtype=np.int32)][:,:,::-1].tolist(),color="#087f8c",weight=3,opacity=.95).add_to(pred)
        pred.add_to(m)
    gps.add_to(m)
    m.get_root().header.add_child(folium.Element("<style>.leaflet-container{background:#f3f7f9!important}</style>"))
    folium.LayerControl(collapsed=False).add_to(m)
    legend='<div style="position:fixed;bottom:25px;left:20px;z-index:9999;background:white;border:1px solid #ccc;padding:12px;font:13px Arial"><b>PathSnap - actual student-test data</b><br><span style="color:#3182ce">Blue</span>: GPS observations<br><span style="color:#e09124">Orange</span>: supplied reference<br><span style="color:#087f8c">Teal</span>: prediction when supplied<br>Grey: road graph<br>Reference is excluded from inference and evaluator UI.</div>'
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[float(lo[1]),float(lo[0])],[float(hi[1]),float(hi[0])]])
    m.save(str(OUT/"representative_map.html"))
    fig,ax=plt.subplots(figsize=(10,6))
    segments=np.stack([network.starts[road_ids],network.ends[road_ids]],axis=1)/1000
    ax.add_collection(LineCollection(segments,colors="#cbd2d6",linewidths=.6))
    ax.add_collection(LineCollection(np.stack([network.starts[reference],network.ends[reference]],axis=1)/1000,colors="#e09124",linewidths=2.5,label="Supplied reference"))
    ax.scatter(xy[:,0]/1000,xy[:,1]/1000,s=5,c="#3182ce",alpha=.6,label="Raw GPS")
    if predictions is not None:
        ids=np.asarray([a for a in predictions if a>=0],dtype=np.int32)
        ax.add_collection(LineCollection(np.stack([network.starts[ids],network.ends[ids]],axis=1)/1000,colors="#087f8c",linewidths=1.5,label="Predicted roads"))
    ax.autoscale();ax.set_aspect("equal");ax.legend(loc="best")
    ax.set(xlabel="Local easting (km)",ylabel="Local northing (km)",title=f"Student-test trip {Path(trip_dir).name}: road geometry and GPS")
    ax.grid(alpha=.12)
    _save(fig,"representative_geographical_view.png")
    (OUT/"representative_map_metadata.json").write_text(json.dumps({"trip_id":Path(trip_dir).name,"partition":"student_test","observations":len(track),"road_arcs_displayed":len(road_ids),"reference_arcs":len(reference),"prediction_layer_included":predictions is not None,"coordinate_system":network.crs.to_string()},indent=2),encoding="utf-8")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--data-root",type=Path,default=DATA_ROOT);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();rows=[]
    for partition in ("train","student_test"):
        for p in trip_dirs(partition,args.data_root):
            rows.append(inspect_trip(p,partition))
        print(f"EDA {partition}: {sum(r['partition']==partition for r in rows)} trips",flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(OUT/"inventory.csv",index=False)
    summary=summary_inventory(df);summary["execution_seconds"]=time.perf_counter()-start
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    plot_inventory(df)
    representative_map(trip_dirs("student_test",args.data_root)[0])
    print(json.dumps(summary,indent=2),flush=True)
    print(f"EDA outputs saved to {OUT}; elapsed {time.perf_counter()-start:.2f}s",flush=True)

if __name__=="__main__":main()
