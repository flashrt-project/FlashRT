"""Reject mismatched LIBERO pilot cases before comparing descriptive successes."""
import argparse
import json
from pathlib import Path
from .common import write_json

SETTINGS = ('scope','suite','isaac_revision','libero_revision','wrapper_sha256','manifest_sha256',
            'protocol_sha256','weight_manifest_sha256','checkpoint_config_sha256',
            'processor_config_sha256','initialization','settling_steps_excluded_from_budget',
            'execution_horizon','max_episode_steps','seed','initial_state_start','episodes_per_task',
            'batch_size','render_backend','selected_tasks')
CASE_FIELDS = ('task_name','prompt','reset_seed','initial_state','settled_state','initial_processed_inputs')


def compare(paths):
    runs=[]
    for path in paths:
        p=Path(path)
        if p.is_dir():p=p/'closed_loop.json'
        run=json.loads(p.read_text())
        if not run.get('completed'):raise ValueError(f'Incomplete rollout: {p}')
        runs.append(run)
    if len(runs)<2 or len({r['backend'] for r in runs})!=len(runs):
        raise ValueError('At least two unique completed backends required')
    reference=runs[0]
    expected_count=len(reference['selected_tasks'])*reference['episodes_per_task']
    reference_cases={(e['task_id'],e['state_index']):e for e in reference['episodes']}
    if len(reference_cases)!=expected_count:raise ValueError('Missing or duplicate reference episodes')
    first_groups={g['task_id']:(g['state_file_sha256'],g['bddl_sha256']) for g in reference['groups']}
    results={}
    for run in runs:
        for key in SETTINGS:
            if run[key]!=reference[key]:raise ValueError(f'Rollout protocol differs: {key}')
        for key in ('camera_keys', 'camera_ablation'):
            if run.get(key)!=reference.get(key):raise ValueError(f'Rollout protocol differs: {key}')
        groups={g['task_id']:(g['state_file_sha256'],g['bddl_sha256']) for g in run['groups']}
        if groups!=first_groups:raise ValueError('Task assets or initial-state files differ')
        cases={(e['task_id'],e['state_index']):e for e in run['episodes']}
        if len(cases)!=len(run['episodes']) or cases.keys()!=reference_cases.keys():
            raise ValueError('Rollout cases missing, duplicated or different')
        for key,episode in cases.items():
            other=reference_cases[key]
            for field in CASE_FIELDS:
                if episode[field]!=other[field]:raise ValueError(f'Initial case differs: {key} {field}')
            for left,right in zip(episode['decisions'],other['decisions']):
                for field in ('decision_index','noise_record_id','noise_sha256'):
                    if left[field]!=right[field]:raise ValueError(f'Paired noise differs: {key} {field}')
            if episode['policy_calls']!=len(episode['decisions']):raise ValueError('Missing decision records')
            if not 1<=episode['environment_steps']<=run['max_episode_steps']:
                raise ValueError('Invalid environment-step count')
        wins=sum(int(e['success']) for e in cases.values())
        results[run['backend']]={'successes':wins,'episodes':len(cases),'success_rate':wins/len(cases),
                                'cases':[{'task_id':key[0],'state_index':key[1],'success':e['success'],
                                          'environment_steps':e['environment_steps']} for key,e in cases.items()]}
    protocol={k:reference[k] for k in SETTINGS}
    protocol.update({k:reference[k] for k in ('camera_keys','camera_ablation') if k in reference})
    return {'scope':reference['scope'],'note':'Descriptive matched pilot; no full-suite or accuracy-equivalence claim',
            'matched_initial_states_and_processed_observations':True,
            'protocol':protocol,'results':results}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    result=compare(args.runs)
    write_json(args.output,result)
    print(json.dumps(result['results'],indent=2))

if __name__=='__main__':main()
