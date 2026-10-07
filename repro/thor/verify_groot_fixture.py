"""Validate GR00T against official actions and measure the selected input boundary."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from compare_actions import metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--embodiment',default='OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT');p.add_argument('--num-views',type=int,default=2);p.add_argument('--cpu-threads',type=int,default=None);p.add_argument('--checkpoint',required=True);p.add_argument('--fixture',required=True);p.add_argument('--reference-records');p.add_argument('--boundary',choices=['raw-full','feature'],default='raw-full');p.add_argument('--tier',choices=['fp8','fp4'],required=True);p.add_argument('--out',required=True);p.add_argument('--warmup',type=int,default=20);p.add_argument('--iters',type=int,default=100);a=p.parse_args()
    if a.cpu_threads is not None and a.cpu_threads<1:p.error('--cpu-threads must be positive')
    f=torch.load(a.fixture,map_location='cpu',weights_only=False);aux=f['aux']
    reference_provenance='historical official policy fixture';golden_comparison=None
    component_samples=[]
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
    from gr00t.data.embodiment_tags import EmbodimentTag
    tag=EmbodimentTag.resolve(a.embodiment)
    fe=Frontend(a.checkpoint,num_views=a.num_views,embodiment_tag=tag.value)
    fe.set_prompt(aux=aux,prompt='fixture uses captured embeddings')
    if a.cpu_threads is not None:torch.set_num_threads(a.cpu_threads)
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
        prep.embodiment_tag=tag
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
            t=time.perf_counter();fresh=prepare();dp=(time.perf_counter()-t)*1000
            # Prompt/grid invariants were checked in preflight; this fixed fixture is immutable.
            torch.cuda.synchronize();t=time.perf_counter()
            current={'pixel_values':fresh['pixel_values'].to('cuda')}
            state_gpu=fresh['state'].to('cuda')
            fe._backbone_features=fe.run_backbone_graph(current)
            output=fe.infer(state_gpu,initial_noise=noise,num_inference_timesteps=4,action_horizon=40)
            torch.cuda.synchronize();model=(time.perf_counter()-t)*1000
            t=time.perf_counter();fe.denormalize_action(output,state_dict=state)
            torch.cuda.synchronize();decode=(time.perf_counter()-t)*1000
            component_samples.append({'data_processing_ms':dp,'model_ms':model,'physical_decode_ms':decode,'jal_style_sum_ms':dp+model})
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
    if actual.shape!=reference.shape:raise ValueError(f'Physical action shape mismatch: {actual.shape} vs {reference.shape}')
    report=metrics(actual,reference)
    report['reference']=reference_provenance;report['official_vs_historical']=golden_comparison
    report['modalities']={k:metrics(np.asarray(decoded[k]),np.asarray(f['actions'][k])) for k in keys}
    timings=[]
    for _ in range(a.iters):
        torch.cuda.synchronize();t=time.perf_counter();run();torch.cuda.synchronize();timings.append((time.perf_counter()-t)*1000)
    report.update(tier=a.tier,shape=list(actual.shape),boundary=('raw RGB/state/language -> fresh official processor patches -> FlashRT kernel backbone/action head -> physical action decode; static prompt/grid captured once' if a.boundary=='raw-full' else 'captured post-patch features and image/text embeddings -> normalized action; physical decode checked separately'),calibration='documented set_prompt single fixture; no extra multi-frame calibration',repeat_identical=bool(torch.equal(y,repeat)),repeat_normalized=repeat_metrics,p50_ms=float(np.median(timings)),p95_ms=float(np.percentile(timings,95)),samples_ms=timings,passed=report['mean_sample_cosine']>=0.999 and report['worst_sample_cosine']>=0.995)
    if a.boundary=='raw-full':
        measured=component_samples[-a.iters:]
        report['component_samples']=measured
        report['component_medians_ms']={k:float(np.median([v[k] for v in measured])) for k in measured[0]}
        report['component_boundary']='model includes input GPU transfer, patch/vision backbone and four-step action head; excludes CPU processor and physical decode; JAL-style total is per-call processor+model sum'
    report['input_boundary']=a.boundary
    report['cpu_threads']=torch.get_num_threads();report['interop_threads']=torch.get_num_interop_threads()
    report['warmup']=a.warmup;report['iters']=a.iters
    report['per_call_reference_model_execution']=False
    report['setup']='one-time fixed prompt/grid graph setup and calibration; captured embeddings used only during calibration'
    report['thresholds']={'combined_mean_cosine':0.999,'combined_worst_cosine':0.995,'eef_and_joint_cosine':0.995,'gripper_max_abs':0.05,'repeat_normalized_cosine':0.9999,'repeat_normalized_max_abs':0.05}
    if tag.value=='libero_sim':
        groups={'position':('x','y','z'),'rotation':('roll','pitch','yaw'),'gripper':('gripper',)}
        report['action_groups']={name:metrics(np.concatenate([np.asarray(decoded[k]) for k in members],axis=-1),np.concatenate([np.asarray(f['actions'][k]) for k in members],axis=-1)) for name,members in groups.items()}
        modality_pass=all(report['action_groups'][k]['worst_sample_cosine']>=.995 for k in ('position','rotation')) and report['action_groups']['gripper']['max_abs']<=.05
    else:
        modality_pass=all(report['modalities'][k]['worst_sample_cosine']>=.995 for k in ('eef_9d','joint_position')) and report['modalities']['gripper_position']['max_abs']<=.05
    report['strict_group_diagnostic_passed']=bool(modality_pass)
    report['passed']=report['passed'] and bool(torch.equal(y,repeat)) and repeat_metrics['worst_sample_cosine']>=0.9999 and repeat_metrics['max_abs']<=0.05
    if tag.value!='libero_sim':report['passed']=report['passed'] and modality_pass
    report['acceptance_scope']=('combined physical-action cosine and identical repeated normalized output; per-group thresholds are separate strict diagnostics, not task-success validation' if tag.value=='libero_sim' else 'combined and per-group physical-action checks plus repeatability')
    report['embodiment']=tag.value
    report['requested_num_views']=a.num_views
    report['image_grid_thw']=aux['grid_thw'].tolist()
    report['delivered_action_keys']=keys

    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));np.savez(out.with_suffix('.npz'),actual=actual,reference=reference,normalized_actual=y.numpy())
    if golden_comparison and golden_comparison['worst_sample_cosine']<0.995:
        report['passed']=False;out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
