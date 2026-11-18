from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", ROOT.parent.parent / "work" / "data" / "Student folder"))
CACHE_ROOT = Path(os.environ.get("PATHSNAP_CACHE", ROOT / "cache"))
SEED = 352
CANDIDATE_K = 10
CANDIDATE_RADIUS = 250.0
FEATURE_VERSION = "trace-lite-v2"

def trip_dirs(partition="student_test", data_root=None):
    base = Path(data_root or DATA_ROOT) / partition
    return sorted(p for p in base.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists()) if base.exists() else []
