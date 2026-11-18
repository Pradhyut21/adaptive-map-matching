"""Package source, measured results and artifacts without data or caches."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime,timezone
from zipfile import ZipFile,ZIP_DEFLATED
from pathsnap.settings import ROOT

def include(path):
    rel=path.relative_to(ROOT)
    if any(part in {'.git','cache','data','__pycache__','.pytest_cache','.venv','runtime','report_review'} for part in rel.parts):return False
    if path.suffix in {'.pyc','.log'} or path.name=='.env':return False
    return path.is_file()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT.parent/'PathSnap_submission.zip');args=parser.parse_args()
    manifest_path=ROOT/'results/artifact_manifest.json'
    records={str(p.relative_to(ROOT)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(ROOT.rglob('*')) if include(p) and p!=manifest_path}
    manifest_path.write_text(json.dumps({'generated_utc':datetime.now(timezone.utc).isoformat(),'files':records,'excludes':'Raw datasets, caches, evaluator data, local runtime records and Git internals.','ai_assistance':'Implementation and evidence compilation assisted by AI; team review and independent interpretation required.'},indent=2),encoding='utf-8')
    with ZipFile(args.output,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(ROOT.rglob('*')):
            if include(p):z.write(p,Path('PathSnap')/p.relative_to(ROOT))
    print('Submission package',args.output.resolve(),'bytes',args.output.stat().st_size)

if __name__=='__main__':main()
