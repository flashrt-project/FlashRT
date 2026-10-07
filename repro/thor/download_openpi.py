import google_crc32c
import json, pathlib, urllib.request, urllib.parse, concurrent.futures, hashlib, base64, time, traceback
import argparse
p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
metadata_root=pathlib.Path(__file__).resolve().parent
root=pathlib.Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True)
items=json.load(open(metadata_root/'official-files.json'))['items']
mirror={i['Path']:i for i in json.load(open(metadata_root/'mirror-files.json'))['Data']['Files']}
def fetch(i):
 rel=i['name'].removeprefix('checkpoints/'); m=mirror[rel]; assert m['Size']==int(i['size'])
 dest=root/'source'/rel; dest.parent.mkdir(parents=True,exist_ok=True); tmp=dest.with_name(dest.name+'.part')
 url='https://modelscope.cn/models/zebinyang/Jetson-PI-pi05/resolve/'+m['Revision']+'/'+rel
 for attempt in range(6):
  try:
   if not dest.exists():
    start=tmp.stat().st_size if tmp.exists() else 0
    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-'} if start else {})
    with urllib.request.urlopen(req,timeout=90) as r:
     mode='ab' if start and r.status==206 else 'wb'
     with open(tmp,mode) as f:
      while b:=r.read(4*1024*1024): f.write(b)
    assert tmp.stat().st_size==int(i['size'])
    tmp.rename(dest)
   h=hashlib.md5() if "md5Hash" in i else google_crc32c.Checksum()
   with open(dest,'rb') as f:
    while b:=f.read(8*1024*1024):h.update(b)
   assert base64.b64encode(h.digest()).decode()==i.get('md5Hash',i.get('crc32c')), 'OFFICIAL CHECKSUM MISMATCH '+rel
   print(time.strftime('%F %T'),'VERIFIED',rel,dest.stat().st_size,flush=True);return
  except Exception as e:
   print('RETRY',attempt,rel,repr(e),flush=True)
   if 'MISMATCH' in str(e):raise
   time.sleep(min(30,5*(attempt+1)))
 raise RuntimeError(rel)
try:
 print('START',time.time(),'TOTAL',sum(int(i['size']) for i in items),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:list(e.map(fetch,items))
 (root/'DOWNLOAD_VERIFIED').write_text(time.strftime('%F %T'))
 print('DOWNLOAD_COMPLETE_OFFICIAL_MD5_VERIFIED',flush=True)
except Exception:
 traceback.print_exc();raise
