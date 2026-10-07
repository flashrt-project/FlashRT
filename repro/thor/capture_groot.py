"""Capture official GR00T inputs/actions with the repository's existing hooks."""
import argparse,sys,json,time,inspect
from pathlib import Path
import numpy as np
import torch

def cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [cpu(v) for v in x]
 return x

def main():
 p=argparse.ArgumentParser();p.add_argument('--embodiment',default='OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT');p.add_argument('--flashrt',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--dataset');p.add_argument('--input-fixture');p.add_argument('--out',required=True);p.add_argument('--pairs',default='1:0,1:50,1:100,1:150,2:0,2:50,2:100,2:150');a=p.parse_args()
 sys.path.insert(0,str(Path(a.flashrt)/'tests/_helpers/groot_n17'))
 from capture_aux_multi import _install_hooks,_restore_hooks,_build_parsed
 import gr00t.model
 from gr00t.policy.gr00t_policy import Gr00tPolicy
 from gr00t.data.embodiment_tags import EmbodimentTag
 from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
 tag=a.embodiment
 policy=Gr00tPolicy(embodiment_tag=EmbodimentTag.resolve(tag),model_path=a.checkpoint,device='cuda:0')
 if a.input_fixture:
  fixed=torch.load(a.input_fixture,map_location='cpu',weights_only=False)['inputs'];a.pairs='1:0';loader=None
 else:
  if not a.dataset:p.error('--dataset or --input-fixture is required')
  extra={'video_backend':'ffmpeg'} if 'video_backend' in inspect.signature(LeRobotEpisodeLoader).parameters else {}
  loader=LeRobotEpisodeLoader(dataset_path=a.dataset,modality_configs=policy.modality_configs,**extra)
 records=[]
 for pair in a.pairs.split(','):
  traj,step=map(int,pair.split(':'));parsed=fixed if a.input_fixture else _build_parsed(loader,policy,traj,step);captured={};hooks=_install_hooks(policy,captured)
  def capture_ids(module,args,kwargs):
   captured['input_ids']=kwargs['input_ids'].detach().cpu()
  ids_hook=policy.model.backbone.model.register_forward_pre_hook(capture_ids,with_kwargs=True)
  ah=policy.model.action_head;gawf=ah.get_action_with_features
  def capture_action_inputs(*args,**kwargs):
   bound=inspect.signature(hooks['ah_gawf_orig']).bind(*args,**kwargs).arguments
   captured['official_normalized_state']=cpu(bound['action_input']['state'])
   captured['official_embodiment_id']=cpu(bound['embodiment_id'])
   captured['official_processed_backbone']=cpu(bound['backbone_features'])
   result=gawf(*args,**kwargs)
   captured['official_normalized_output']=cpu(dict(result))
   return result
  ah.get_action_with_features=capture_action_inputs
  visual=policy.model.backbone.model.model.visual
  hooked_visual=visual.forward
  def visual_input(hidden_states,grid_thw,**kw):
   captured['pixel_values']=hidden_states.detach().cpu()
   return hooked_visual(hidden_states,grid_thw,**kw)
  visual.forward=visual_input
  try:
   torch.manual_seed(0);np.random.seed(0);torch.cuda.synchronize();t=time.perf_counter()
   with torch.inference_mode():result=policy.get_action(parsed)
   torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000
  finally:
   ids_hook.remove()
   _restore_hooks(hooks)
  actions=result[0] if isinstance(result,tuple) else result
  record={'inputs':cpu(parsed),'aux':cpu(captured),'state':cpu(parsed['state']),'actions':cpu(actions),'meta':{'traj':traj,'step':step,'seed':0,'tag':tag,'instrumented_capture_ms':ms,'views':list(policy.modality_configs['video'].modality_keys)}}
  records.append(record);print('CAPTURED',pair,ms, list(actions),flush=True)
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);torch.save(records,out);print('SAVED',out,len(records),flush=True)
if __name__=='__main__':main()
