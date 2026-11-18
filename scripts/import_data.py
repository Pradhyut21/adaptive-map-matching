"""Extract ONLY permitted training/student partitions from the supplied ZIP."""
import argparse
from pathlib import Path,PurePosixPath
from zipfile import ZipFile
from pathsnap.settings import ROOT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--zip',required=True,type=Path);parser.add_argument('--destination',default=ROOT/'data',type=Path);args=parser.parse_args()
    destination=args.destination.resolve();destination.mkdir(parents=True,exist_ok=True)
    extracted=0
    with ZipFile(args.zip) as z:
        for item in z.infolist():
            parts=PurePosixPath(item.filename).parts
            if 'train' in parts:at=parts.index('train')
            elif 'student_test' in parts:at=parts.index('student_test')
            else:continue
            relative=Path(*parts[at:])
            target=(destination/relative).resolve()
            if not target.is_relative_to(destination):raise ValueError('Unsafe archive path')
            if item.is_dir():target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item));extracted+=1
    print(f'Extracted {extracted} permitted files to {destination}')
    print('Set PATHSNAP_DATA to that directory before running preparation or the API.')

if __name__=='__main__':main()
