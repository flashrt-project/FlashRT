"""Same-checkpoint, same-images/noise OpenPI vs FlashRT comparison; subprocess arms."""
import argparse,json,time,os,subprocess,sys
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--checkpoint',required=True);p.add_argument('--fixture',required=True)
 p.add_argument('--output',required=True);p.add_argument('--mode',choices=['openpi','fp8','fp4'],required=True)
 p.add_argument('--seed',type=int,default=20261007);p.add_argument('--warmup',type=int,default=20);p.add_argument('--iters',type=int,default=100)
 p.add_argument('--entrypoint',choices=['infer','predict'],default='infer')
 p.add_argument('--awq-alpha',type=float,default=None)
 a=p.parse_args()
 if a.mode=='openpi':os.environ['TORCH_COMPILE_DISABLE']='1'
 import torch
 if a.mode=='openpi':torch.backends.cuda.enable_mem_efficient_sdp(False)
 z=np.load(a.fixture); n=int(z['n']); ck=Path(a.checkpoint)
 prompt='pick up the black bowl and place it on the plate'
 if a.mode=='openpi':
  from openpi.training import config
  from openpi.policies import policy_config
  cfg=config.get_config('pi05_libero')
  policy=policy_config.create_trained_policy(cfg,str(ck),pytorch_device='cuda')
  def run(i):
   noise=np.random.randn(cfg.model.action_horizon,32).astype(np.float16).astype(np.float32)
   return policy.infer({'observation/image':z[f'img_{i}'],'observation/wrist_image':z[f'wrist_{i}'],'observation/state':z[f'state_{i}'].astype(np.float32),'prompt':prompt},noise=noise)['actions']
  extra={'openpi_config':str(cfg.model),'reference_execution':'eager; TORCH_COMPILE_DISABLE=1; memory-efficient SDPA disabled (NGC torch dtype compatibility)'}
 else:
  import flash_rt
  options={'use_fp4':True,'use_fp4_decoder':True} if a.mode=='fp4' else {}
  if a.awq_alpha is not None:options['awq_alpha']=a.awq_alpha
  policy=flash_rt.load_model(checkpoint=str(ck),config='pi05',hardware='thor',framework='torch',num_views=2,autotune=3,use_fa4=True,**options)
  obs=[{'image':z[f'img_{i}'],'wrist_image':z[f'wrist_{i}'],'state':z[f'state_{i}']} for i in range(n)]
  policy.set_prompt(prompt)
  policy.calibrate(obs,percentile=99.9,verbose=False)
  def run(i):
   if a.entrypoint=='predict':return policy.predict(images=[obs[i]['image'],obs[i]['wrist_image']],prompt=prompt)
   return policy.infer(obs[i])['actions']
  knobs=['num_views','autotune','use_fa4','use_fp4_encoder_ffn','use_fp4_decoder','use_awq','awq_alpha','use_p1_split_gu','use_fp4_encoder_attn','use_fp4_siglip_ffn','encoder_p1_combiner','encoder_down_variant','decoder_gate_up_variant']
  extra={'flash_rt':flash_rt.__file__,'entrypoint':a.entrypoint,'resolved_options':{k:getattr(policy._pipe,k,None) for k in knobs},'fp4_layers':sorted(getattr(policy._pipe,'_fp4_layers',[]))}
 for j in range(a.warmup):run(j%n)
 actions=[]
 for i in range(n):
  np.random.seed(a.seed+i);actions.append(np.asarray(run(i)))
 ms=[]
 for i in range(a.iters):
  torch.cuda.synchronize();t=time.perf_counter();run(i%n);torch.cuda.synchronize();ms.append((time.perf_counter()-t)*1000)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 np.savez(out,actions=np.stack(actions),ms=ms)
 out.with_suffix('.json').write_text(json.dumps({'mode':a.mode,'checkpoint':str(ck),'fixture':a.fixture,'seed':a.seed,'warmup':a.warmup,'iters':a.iters,'p50_ms':float(np.median(ms)),'p95_ms':float(np.percentile(ms,95)),'torch':torch.__version__,**extra},indent=2))
 print(a.mode, np.stack(actions).shape, 'p50',np.median(ms),flush=True)
if __name__=='__main__':main()
