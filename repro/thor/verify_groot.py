"""Real-state, fresh-image GR00T comparison against captured official actions."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import torch
from compare_actions import metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--records',required=True);p.add_argument('--trajectory',type=int,default=1);p.add_argument('--tier',choices=['fp8','fp4'],required=True);p.add_argument('--out',required=True);p.add_argument('--warmup',type=int,default=20);p.add_argument('--iters',type=int,default=100);p.add_argument('--backbone-input',choices=['postpatch','patches'],default='postpatch');a=p.parse_args()
 records=torch.load(a.records,map_location='cpu',weights_only=False)
 if a.backbone_input=='postpatch':
  for r in records:
   r['aux'].pop('pixel_values',None);r['aux'].pop('input_ids',None)
 # Different frames from one trajectory retain the same prompt and captured shape.
 cal=[r for r in records if r['meta']['traj']==a.trajectory and r['meta']['step'] in (0,50)]
 ev=[r for r in records if r['meta']['traj']==a.trajectory and r['meta']['step'] in (100,150)]
 if len(cal)!=2 or len(ev)!=2:raise ValueError('need calibration frames 0/50 and held-out frames 100/150')
 if a.tier=='fp4':
  from flash_rt.frontends.torch.groot_n17_thor_fp4 import GrootN17TorchFrontendThorFP4 as Frontend
 else:
  from flash_rt.frontends.torch.groot_n17_thor_fp8 import GrootN17TorchFrontendThorFP8 as Frontend
 fe=Frontend(a.checkpoint,num_views=2,embodiment_tag='oxe_droid_relative_eef_relative_joint')
 fe.set_prompt(aux=cal[0]['aux'],prompt='calibration')
 fe.calibrate([r['aux'] for r in cal],percentile=99.9)
 rows=[];outputs=[];timings=[]
 for r in ev:
  aux=r['aux']
  for key in ('grid_thw','rope_cos','rope_sin','visual_pos_masks'):
   if not torch.equal(aux[key],cal[0]['aux'][key]):raise ValueError(f'new prompt/shape requires another frontend: {key}')
  state={('state.'+k if not k.startswith('state.') else k):np.asarray(v) for k,v in r['state'].items()}
  sn=fe.normalize_state(state);noise=aux['initial_noise'].to('cuda').bfloat16().contiguous()
  state_error=float((sn.cpu().float()-aux['official_normalized_state'].float()).abs().max()) if 'official_normalized_state' in aux else None
  def run():
   fe._backbone_features=fe.run_backbone_graph(aux)
   return fe.infer(sn,initial_noise=noise.clone(),num_inference_timesteps=4,action_horizon=40)
  for _ in range(a.warmup):run()
  first=run().detach().cpu().clone();repeat=run().detach().cpu().clone()
  if not torch.equal(first,repeat):raise RuntimeError('repeat-identical check failed')
  den=fe.denormalize_action(first,state_dict=state)
  mods={}
  for k,ref in r['actions'].items():
   cand=np.asarray(den[k]);ref=np.asarray(ref)
   # preserve the entire 40-step horizon and every native modality dimension
   mods[k]=metrics(cand.reshape(1,-1),ref.reshape(1,-1))
  ms=[]
  for _ in range(a.iters):
   torch.cuda.synchronize();t=time.perf_counter();run();torch.cuda.synchronize();ms.append((time.perf_counter()-t)*1000)
  normalized_metrics=metrics(first.numpy().reshape(1,-1),aux['official_normalized_output']['action_pred'].float().numpy().reshape(1,-1)) if 'official_normalized_output' in aux else None
  timings.extend(ms);rows.append({'meta':r['meta'],'modalities':mods,'normalized_state_max_abs_vs_official':state_error,'normalized_action_vs_official':normalized_metrics,'p50_ms':float(np.median(ms)),'p95_ms':float(np.percentile(ms,95)),'repeat_identical':True})
  outputs.append({k:torch.as_tensor(v).cpu() for k,v in den.items()});print(r['meta']['traj'],r['meta']['step'],mods,flush=True)
 d={'tier':a.tier,'boundary':a.backbone_input+' input + normalized state + fixed noise -> normalized action; decode checked separately','trajectory':a.trajectory,'calibration_steps':[0,50],'evaluation_steps':[100,150],'denoise_steps':4,'action_horizon':40,'rows':rows,'p50_ms':float(np.median(timings)),'p95_ms':float(np.percentile(timings,95)),'passed':all(m['worst_sample_cosine']>=0.995 for row in rows for m in row['modalities'].values())}
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(d,indent=2));torch.save(outputs,out.with_suffix('.pt'));print(json.dumps(d,indent=2));return 0 if d['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
