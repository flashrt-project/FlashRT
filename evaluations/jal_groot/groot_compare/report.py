"""Compare completed backend runs only when their observations and noise match."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import write_json
from .metrics import error_metrics


def load_run(path):
    p=Path(path)
    if p.is_dir():p=p/'run.json'
    run=json.loads(p.read_text())
    if not run.get('completed'):
        raise ValueError(f'Run incomplete: {p}')
    return p.parent,run


def validate_pair(reference,candidate):
    for key in ('manifest_sha256','protocol_sha256','checkpoint_config_sha256','processor_config_sha256','weight_manifest_sha256','noise_seed','cpu_threads','input_contract'):
        if reference[key]!=candidate[key]:raise ValueError(f'Unmatched run setting: {key}')
    rr={r['id']:r for r in reference['records']}
    cc={r['id']:r for r in candidate['records']}
    if not rr or rr.keys()!=cc.keys():raise ValueError('Evaluation record IDs differ or are empty')
    for key in rr:
        for field in ('observation_sha256','noise_sha256','processed_inputs'):
            if rr[key][field]!=cc[key][field]:raise ValueError(f'Unmatched {field} at {key}')
    return rr,cc


def score_runs(reference_path,candidate_path):
    rd,reference=load_run(reference_path)
    cd,candidate=load_run(candidate_path)
    if reference['backend'] != 'torch':raise ValueError('Numerical reference must be official PyTorch')
    if any(r.get('phase') not in ('accuracy','both') for r in (reference,candidate)):raise ValueError('Accuracy scoring requires saved action outputs')
    rr,cc=validate_pair(reference,candidate)
    arrays,expected,truth,executed=[],[],[],[]
    per_trajectory={}
    for key in rr:
        with np.load(rd/rr[key]['file'],allow_pickle=False) as r,np.load(cd/cc[key]['file'],allow_pickle=False) as c:
            if not np.array_equal(r['noise'],c['noise']):raise ValueError('Noise payload differs')
            if not np.array_equal(r['ground_truth'],c['ground_truth']):raise ValueError('Ground truth differs')
            arrays.append(c['actions']);expected.append(r['actions'])
            truth.append(c['ground_truth']);executed.append(c['executed_actions'])
            tid=str(cc[key]['trajectory_id'])
            per_trajectory.setdefault(tid,{'a':[],'b':[]})
            per_trajectory[tid]['a'].append(c['executed_actions']);per_trajectory[tid]['b'].append(c['ground_truth'])
    traj={tid:error_metrics(np.concatenate(v['a']),np.concatenate(v['b'])) for tid,v in per_trajectory.items()}
    return {'backend':candidate['backend'],'observation_count':len(rr),
            'against_official_pytorch':error_metrics(np.concatenate(arrays),np.concatenate(expected)),
            'against_recorded_actions':error_metrics(np.concatenate(executed),np.concatenate(truth)),
            'per_trajectory_ground_truth':traj,
            'jal_style_mean_trajectory_mae':float(np.mean([v['mae'] for v in traj.values()])),
            'jal_style_mean_trajectory_mse':float(np.mean([v['mse'] for v in traj.values()])),
            'processed_inputs_match_reference':True,'task_success_measured':False}


def timing_summary(paths, accuracy_paths=None):
    results={}
    order=None
    contract=None
    first_run=None
    accuracy={}
    for path in accuracy_paths or []:
        _,run=load_run(path)
        if run['backend'] in accuracy:raise ValueError('Duplicate accuracy backend')
        accuracy[run['backend']]=run
    for path in paths:
        _,run=load_run(path)
        if first_run is None:first_run=run
        else:validate_pair(first_run,run)
        if accuracy_paths is not None:
            if run['backend'] not in accuracy:raise ValueError('Timing backend has no matched accuracy evidence')
            validate_pair(accuracy[run['backend']],run)
        if run['backend'] in results:raise ValueError('Duplicate timing backend')
        if not run['exclusive_gpu_asserted']:raise ValueError('Timing was not exclusive')
        settings={key:run[key] for key in ('manifest_sha256','protocol_sha256','checkpoint_config_sha256','processor_config_sha256','weight_manifest_sha256','noise_seed','cpu_threads','warmup','iters_per_prompt')}
        if contract is None:contract=settings
        elif contract!=settings:raise ValueError('Timing settings differ')
        samples=run['timing_samples']
        if not samples:raise ValueError('Missing latency samples')
        sequence=[(x['prompt'],x['record_id'],x['iteration']) for x in samples]
        if order is None:order=sequence
        elif sequence!=order:raise ValueError('Timing observations/order differ')
        values=[x['full_call_ms'] for x in samples]
        results[run['backend']]={'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),'calls':len(values),'per_prompt':{g['prompt']:g['latency_ms'] for g in run['groups']}}
    if 'trt' in results and 'flashrt' in results:
        results['flashrt_speedup_over_jal_trt']=results['trt']['median_ms']/results['flashrt']['median_ms']
    return results



def repeated_timing_summary(round_paths, accuracy_paths):
    rounds=[]
    samples={}
    contract=None
    expected_backends=None
    for paths in round_paths:
        summary=timing_summary(paths,accuracy_paths)
        rounds.append(summary)
        backends=set()
        for path in paths:
            _,run=load_run(path)
            backends.add(run['backend'])
            current={key:run[key] for key in ('warmup','iters_per_prompt','manifest_sha256','protocol_sha256','noise_seed','cpu_threads')}
            if contract is None:contract=current
            elif current!=contract:raise ValueError('Settings differ across timing rounds')
            samples.setdefault(run['backend'],[]).extend(x['full_call_ms'] for x in run['timing_samples'])
        if expected_backends is None:expected_backends=backends
        elif backends!=expected_backends:raise ValueError('Timing rounds have different backends')
    aggregate={backend:{'median_ms':float(np.median(values)),
                        'p95_ms':float(np.percentile(values,95)),
                        'calls':len(values),
                        'round_medians_ms':[r[backend]['median_ms'] for r in rounds]}
               for backend,values in samples.items()}
    if 'trt' in aggregate and 'flashrt' in aggregate:
        aggregate['flashrt_speedup_over_jal_trt']=aggregate['trt']['median_ms']/aggregate['flashrt']['median_ms']
    return {'round_count':len(rounds),'aggregate':aggregate,'rounds':rounds}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True)
    p.add_argument('--candidates',nargs='+',required=True)
    p.add_argument('--timing',nargs='*',default=[])
    p.add_argument('--timing-round',nargs=2,action='append',default=[])
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.timing and a.timing_round:p.error('Use --timing or --timing-round')
    result={'scope':'offline paired evaluation; not a robot task-success benchmark',
            'acceptance':'raw measured errors; no equivalence margin has been agreed',
            'reference_accuracy':score_runs(a.reference,a.reference),
            'accuracy':[score_runs(a.reference,x) for x in a.candidates]}
    if a.timing:result['latency']=timing_summary(a.timing,[a.reference]+a.candidates)
    if a.timing_round:result['repeated_latency']=repeated_timing_summary(a.timing_round,[a.reference]+a.candidates)
    write_json(a.output,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
