"""Trip-local metric road geometry; contains no reference-route reader."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import shapely
from shapely.strtree import STRtree
from pyproj import Transformer, CRS

@dataclass
class CandidateSet:
    arc_ids: np.ndarray
    distances: np.ndarray
    fractions: np.ndarray
    xy: np.ndarray

    @property
    def valid(self):
        return self.arc_ids >= 0

class RoadNetwork:
    def __init__(self, nodes_lonlat, arcs, graph_id=""):
        self.nodes_lonlat = np.asarray(nodes_lonlat, dtype=np.float64).reshape(-1, 2)
        self.arcs = np.asarray(arcs, dtype=np.int32).reshape(-1, 2)
        if not np.isfinite(self.nodes_lonlat).all():
            raise ValueError("Nonfinite road coordinates")
        if len(self.arcs) and (self.arcs.min() < 0 or self.arcs.max() >= len(self.nodes_lonlat)):
            raise ValueError("Arc endpoint outside node bounds")
        center = np.median(self.nodes_lonlat, axis=0)
        self.crs = CRS.from_proj4(f"+proj=aeqd +lat_0={center[1]} +lon_0={center[0]} +datum=WGS84 +units=m +no_defs")
        self.projector = Transformer.from_crs(4326, self.crs, always_xy=True)
        self.inverse = Transformer.from_crs(self.crs, 4326, always_xy=True)
        self.nodes_xy = self.project(self.nodes_lonlat)
        self.starts = self.nodes_xy[self.arcs[:, 0]]
        self.ends = self.nodes_xy[self.arcs[:, 1]]
        self.vectors = self.ends - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.bearings = np.arctan2(self.vectors[:, 1], self.vectors[:, 0])
        self.lines = shapely.linestrings(np.stack([self.starts, self.ends], axis=1))
        self.tree = STRtree(self.lines)
        self.graph_id = graph_id or hashlib.sha256(self.arcs.tobytes() + self.nodes_lonlat.tobytes()).hexdigest()[:20]

    def project(self, lonlat):
        a = np.asarray(lonlat, dtype=float).reshape(-1, 2)
        return np.column_stack(self.projector.transform(a[:, 0], a[:, 1]))

    def unproject(self, xy):
        a = np.asarray(xy, dtype=float).reshape(-1, 2)
        return np.column_stack(self.inverse.transform(a[:, 0], a[:, 1]))

    def candidates(self, xy, k=10, radius=100.0, max_radius=300.0):
        """Exact segment distances after STRtree dwithin; no labels enter here.

        Expand only an empty query to max_radius. Points farther from every road
        remain explicitly unmatched. Ties are broken by arc row ID.
        """
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        n = len(xy)
        ids = np.full((n, k), -1, np.int32)
        distances = np.full((n, k), np.inf, np.float32)
        fractions = np.zeros((n, k), np.float32)
        projected = np.zeros((n, k, 2), np.float64)
        for lo in range(0, n, 512):
            pts = shapely.points(xy[lo:lo+512])
            pairs = self.tree.query(pts, predicate="dwithin", distance=radius)
            grouped = [[] for _ in range(len(pts))]
            for point, arc in pairs.T:
                grouped[point].append(arc)
            for j, choices in enumerate(grouped):
                if not choices:
                    choices = self.tree.query(pts[j], predicate="dwithin", distance=max_radius).tolist()
                if not choices:
                    continue
                choices = np.asarray(choices, np.int32)
                dist, frac, proj = project_segments(xy[lo+j], self.starts[choices], self.ends[choices])
                order = np.lexsort((choices, dist))[:k]
                m = len(order)
                ids[lo+j, :m] = choices[order]
                distances[lo+j, :m] = dist[order]
                fractions[lo+j, :m] = frac[order]
                projected[lo+j, :m] = proj[order]
        return CandidateSet(ids, distances, fractions, projected)

def project_segments(point, starts, ends):
    v = ends - starts
    denom = (v*v).sum(axis=-1)
    f = np.clip(((point - starts)*v).sum(axis=-1) / np.maximum(denom, 1e-12), 0, 1)
    projected = starts + f[..., None]*v
    return np.linalg.norm(point-projected, axis=-1), f, projected

def load_observations(trip_dir):
    p = Path(trip_dir)
    a = np.loadtxt(p / f"{p.name}.track", ndmin=2)
    if a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError("Track must contain three finite numeric columns")
    if np.any(np.abs(a[:, 0]) > 180) or np.any(np.abs(a[:, 1]) > 90):
        raise ValueError("Implausible longitude/latitude")
    return a

def load_network(trip_dir):
    p = Path(trip_dir)
    nodes = np.loadtxt(p / f"{p.name}.nodes", ndmin=2)
    arcs = np.loadtxt(p / f"{p.name}.arcs", dtype=np.int32, ndmin=2)
    fingerprint = hashlib.sha256(nodes.tobytes() + arcs.tobytes()).hexdigest()[:20]
    return RoadNetwork(nodes, arcs, fingerprint)
