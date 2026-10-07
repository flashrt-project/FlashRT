"""Reproduce the documented, official-reference DROID fixture check."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from compare_actions import metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--fixture',required=True);p.add_argument('--reference-records');p.add_argument('--boundary',choices=['raw-full','feature'],default='raw-full');p.add_argument('--tier',choices=['fp8','fp4'],required=True);p.add_argument('--out',required=True);p.add_argument('--warmup',type=int,default=20);p.add_argument('--iters',type=int,default=100);a=p.parse_args()
    f=torch.load(a.fixture,map_location='cpu',weights_only=False);aux=f['aux']
    reference_provenance='historical official policy fixture';golden_comparison=None
    if a.reference_records:
        records=torch.load(a.reference_records,map_location='cpu',weights_only=False)
        if len(records)!=1:raise ValueError('Expected one capture of the fixed input')
        record=records[0];keys=sorted(f['actions'])
        golden_comparison=metrics(np.concatenate([np.asarray(record['actions'][k]) for k in keys],axis=-1),np.concatenate([np.asarray(f['actions'][k]) for k in keys],axis=-1))
        for k,v in f['state'].items():
            if not np.array_equal(np.asarray(v),np.asarray(record['state'][k])):raise ValueError('Reference state differs from fixture')
        f={**f,'actions':record['actions']};aux=dict(record['aux'])
        if a.boundary=='feature':
            aux.pop('pixel_values',None);aux.pop('input_ids',None)
        reference_provenance='fresh official policy capture of the included raw input'
    if a.tier=='fp4':
        from flash_rt.frontends.torch.groot_n17_thor_fp4 import GrootN17TorchFrontendThorFP4 as Frontend
    else:
        from flash_rt.frontends.torch.groot_n17_thor_fp8 import GrootN17TorchFrontendThorFP8 as Frontend
    fe=Frontend(a.checkpoint,num_views=2,embodiment_tag='oxe_droid_relative_eef_relative_joint')
    fe.set_prompt(aux=aux,prompt='fixture uses captured embeddings')
    state={'state.'+k:np.asarray(v) for k,v in f['state'].items()}
    normalized=fe.normalize_state(state)
    noise=aux['initial_noise'].to('cuda').bfloat16().contiguous()
    if a.boundary=='raw-full':
        import gr00t.model
        from transformers import AutoProcessor
        from gr00t.policy.gr00t_policy import Gr00tPolicy,_rec_to_dtype
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import MessageType
        prep=Gr00tPolicy.__new__(Gr00tPolicy)
        prep.processor=AutoProcessor.from_pretrained(a.checkpoint);prep.processor.eval()
        prep.embodiment_tag=EmbodimentTag.resolve('OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT')
        prep.modality_configs=prep.processor.get_modality_configs()[prep.embodiment_tag.value]
        prep.language_key=prep.modality_configs['language'].modality_keys[0]
        def prepare():
            processed=[]
            for obs in prep._unbatch_observation(f['inputs']):
                step=prep._to_vla_step_data(obs)
                processed.append(prep.processor([{'type':MessageType.EPISODE_STEP.value,'content':step}]))
            return _rec_to_dtype(prep.processor.collator(processed),torch.bfloat16)['inputs']
        check=prepare()
        for key,other in (('pixel_values','pixel_values'),('input_ids','input_ids'),('image_grid_thw','grid_thw')):
            if not torch.equal(check[key].cpu(),aux[other].to(check[key].dtype).cpu()):
                raise ValueError('Processor input mismatch: '+key)
        def run():
            fresh=prepare()
            if not torch.equal(fresh['input_ids'].cpu(),check['input_ids'].cpu()) or not torch.equal(fresh['image_grid_thw'].cpu(),check['image_grid_thw'].cpu()):
                raise ValueError('Changed prompt/grid requires recapture')
            current={'pixel_values':fresh['pixel_values']}
            fe._backbone_features=fe.run_backbone_graph(current)
            output=fe.infer(fresh['state'],initial_noise=noise,num_inference_timesteps=4,action_horizon=40)
            fe.denormalize_action(output,state_dict=state)
            return output
    else:
        def run():
            fe._backbone_features=fe.run_backbone_graph(aux)
            return fe.infer(normalized,initial_noise=noise,num_inference_timesteps=4,action_horizon=40)
    for _ in range(a.warmup):run()
    y=run().detach().cpu().clone();repeat=run().detach().cpu().clone()
    repeat_metrics=metrics(y.float().numpy(),repeat.float().numpy())
    decoded=fe.denormalize_action(y,state_dict=state)
    keys=sorted(f['actions']);actual=np.concatenate([np.asarray(decoded[k]) for k in keys],axis=-1);reference=np.concatenate([np.asarray(f['actions'][k]) for k in keys],axis=-1)
    if actual.shape!=(1,40,17):raise ValueError(f'Unexpected physical action shape: {actual.shape}')
    report=metrics(actual,reference)
    report['reference']=reference_provenance;report['official_vs_historical']=golden_comparison
    report['modalities']={k:metrics(np.asarray(decoded[k]),np.asarray(f['actions'][k])) for k in keys}
    timings=[]
    for _ in range(a.iters):
        torch.cuda.synchronize();t=time.perf_counter();run();torch.cuda.synchronize();timings.append((time.perf_counter()-t)*1000)
    report.update(tier=a.tier,shape=list(actual.shape),boundary=('raw RGB/state/language -> fresh official processor patches -> FlashRT kernel backbone/action head -> physical action decode; static prompt/grid captured once' if a.boundary=='raw-full' else 'captured post-patch features and image/text embeddings -> normalized action; physical decode checked separately'),calibration='documented set_prompt single fixture; no extra multi-frame calibration',repeat_identical=bool(torch.equal(y,repeat)),repeat_normalized=repeat_metrics,p50_ms=float(np.median(timings)),p95_ms=float(np.percentile(timings,95)),samples_ms=timings,passed=report['mean_sample_cosine']>=0.999 and report['worst_sample_cosine']>=0.995)
    report['input_boundary']=a.boundary
    report['warmup']=a.warmup;report['iters']=a.iters
    report['per_call_reference_model_execution']=False
    report['setup']='one-time fixed prompt/grid graph setup and calibration; captured embeddings used only during calibration'
    report['thresholds']={'combined_mean_cosine':0.999,'combined_worst_cosine':0.995,'eef_and_joint_cosine':0.995,'gripper_max_abs':0.05,'repeat_normalized_cosine':0.9999,'repeat_normalized_max_abs':0.05}
    report['passed']=report['passed'] and repeat_metrics['worst_sample_cosine']>=0.9999 and repeat_metrics['max_abs']<=0.05 and all(report['modalities'][k]['worst_sample_cosine']>=0.995 for k in ('eef_9d','joint_position')) and report['modalities']['gripper_position']['max_abs']<=0.05
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));np.savez(out.with_suffix('.npz'),actual=actual,reference=reference,normalized_actual=y.numpy())
    if golden_comparison and golden_comparison['worst_sample_cosine']<0.995:
        report['passed']=False;out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
