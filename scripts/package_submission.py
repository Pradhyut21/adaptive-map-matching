"""Package source, measured results and fitted artifacts."""
import argparse
import hashlib
import json
import os
from datetime import datetime,timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile,ZIP_DEFLATED
from pathsnap.settings import ROOT


EXCLUDED_DIRS={'.git','cache','data','__pycache__','.pytest_cache','.venv','runtime','report_review'}
EXCLUDED_SUFFIXES={'.pyc','.log','.tmp','.track','.nodes','.arcs','.route','.zip'}


def package_files(target):
    """Walk the deliverable tree without entering data and scratch folders."""
    files=[]
    for folder, directories, names in os.walk(ROOT):
        directory=Path(folder)
        directories[:]=[name for name in directories if name not in EXCLUDED_DIRS and not (directory/name).is_symlink()]
        for name in names:
            path=directory/name
            if path.is_symlink() or name.startswith('.env') or path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            if path.is_file() and path.resolve()!=target:
                files.append(path)
    return sorted(files)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT.parent/'PathSnap_submission.zip')
    target=parser.parse_args().output.resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    manifest_path=ROOT/'results/artifact_manifest.json'
    files=[path for path in package_files(target) if path!=manifest_path]
    records={}
    for path in files:
        with path.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        records[path.relative_to(ROOT).as_posix()]={'bytes':path.stat().st_size,'sha256':digest}
    manifest={
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'hashing':'SHA-256 of each file as stored in the submission ZIP.',
        'files':records,
        'excludes':'Raw datasets, caches, evaluator data, local runtime records and Git internals.',
        'ai_assistance':'AI assisted implementation and evidence compilation. Team review and independent interpretation remain required.',
    }
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    with NamedTemporaryFile(dir=target.parent,prefix=f'.{target.stem}-',suffix='.tmp',delete=False) as temp:
        temporary=Path(temp.name)
    try:
        with ZipFile(temporary,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
            for path in [*files,manifest_path]:
                archive.write(path,(Path('PathSnap')/path.relative_to(ROOT)).as_posix())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Submission ZIP {target} ({target.stat().st_size:,} bytes, {len(files)+1} files)')


if __name__=='__main__':
    main()
