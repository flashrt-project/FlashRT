"""Verify and unpack the source-only runtime used by the Thor regressions."""
import argparse, hashlib, json, tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
root=Path(__file__).resolve().parent;target=Path(a.out).resolve()
lock=json.loads((root/'versions.json').read_text());archive=root/'runtime-source.tar.gz'
if hashlib.sha256(archive.read_bytes()).hexdigest()!=lock['runtime_archive_sha256']:
    raise RuntimeError('Runtime archive checksum mismatch')
if target.exists():raise FileExistsError('Choose a new runtime output directory')
target.mkdir(parents=True)
with tarfile.open(archive,'r:gz') as source:source.extractall(target,filter='data')
print(target)
