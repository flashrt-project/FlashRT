import argparse,json
from pathlib import Path
import numpy as np

def metrics(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError(f'nonfinite or mismatched outputs: {a.shape} / {b.shape}')
 aa=a.reshape(len(a),-1);bb=b.reshape(len(b),-1)
 norm=np.linalg.norm(aa,axis=1)*np.linalg.norm(bb,axis=1)
 if (norm==0).any():raise ValueError('zero-norm sample: cosine undefined')
 c=(aa*bb).sum(1)/norm;d=a-b
 return {'per_sample_cosine':c.tolist(),'mean_sample_cosine':float(c.mean()),'worst_sample_cosine':float(c.min()),'rmse':float(np.sqrt((d*d).mean())),'max_abs':float(np.abs(d).max())}

def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 r=np.load(a.reference)['actions'];c=np.load(a.candidate)['actions']
 # LIBERO policy outputs 7 physical action dimensions. Reject unequal horizons.
 if r.ndim!=3 or c.ndim!=3 or r.shape[-1]!=7 or c.shape[-1]!=7:raise ValueError('Expected [samples, horizon, 7] decoded LIBERO actions')
 if r.shape[1]!=c.shape[1]:raise ValueError('action horizons differ; align the model configs first')
 d=metrics(c,r);d['gripper_sign_disagreement']=float((np.sign(c[...,6])!=np.sign(r[...,6])).mean())
 d['passed']=d['mean_sample_cosine']>=0.999 and d['worst_sample_cosine']>=0.995 and d['gripper_sign_disagreement']==0
 Path(a.out).write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2));return 0 if d['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
