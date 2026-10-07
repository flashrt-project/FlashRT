"""Download an immutable ModelScope snapshot and verify its published SHA256."""
import argparse, hashlib, json
from pathlib import Path
import requests
from modelscope import snapshot_download

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    response=requests.get(f'https://modelscope.cn/api/v1/models/{a.repo}/repo/files',params={'Revision':a.revision,'Recursive':'true'},timeout=60)
    response.raise_for_status(); metadata=response.json()
    if metadata.get('Code')!=200:raise RuntimeError(metadata)
    root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True)
    files=[f for f in metadata['Data']['Files'] if f['Type']=='blob']
    missing=[]
    for f in files:
        path=root/f['Path']
        if path.is_file() and path.stat().st_size==f['Size']:
            with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
            if sha==f['Sha256']:continue
        missing.append(f['Path'])
    # Reuse locally verified files instead of trusting SDK cache metadata.
    if missing:
        snapshot_download(a.repo,revision=a.revision,local_dir=str(root),allow_file_pattern=missing,max_workers=2)
    manifest=[]
    for f in metadata['Data']['Files']:
        if f['Type']!='blob':continue
        path=root/f['Path']
        if not path.exists():
            # Some SDK versions omit dotfiles even with an explicit allow-list.
            path.parent.mkdir(parents=True,exist_ok=True)
            response=requests.get(f'https://modelscope.cn/models/{a.repo}/resolve/{a.revision}/{f["Path"]}',stream=True,timeout=120)
            response.raise_for_status()
            temporary=path.with_name(path.name+'.part')
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(4*1024*1024):stream.write(chunk)
            temporary.replace(path)
        if path.stat().st_size!=f['Size']:raise RuntimeError(f'Size mismatch: {path}')
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if sha!=f['Sha256']:raise RuntimeError(f'SHA256 mismatch: {path}')
        manifest.append({'path':f['Path'],'size':f['Size'],'sha256':sha})
    (root/'download-manifest.json').write_text(json.dumps({'repo':a.repo,'revision':a.revision,'files':manifest},indent=2))
    print(f'Verified {len(manifest)} files against ModelScope metadata. This does not establish equivalence to another hub.')
if __name__=='__main__':main()
