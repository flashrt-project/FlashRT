import copy
import json
import pytest
from groot_compare.closed_loop_report import compare, SETTINGS


def sample(backend):
    data={key:'same' for key in SETTINGS}
    data.update(completed=True,backend=backend,selected_tasks=[{'task_id':4}],episodes_per_task=1,
                scope='task_subset',max_episode_steps=720,groups=[{'task_id':4,'state_file_sha256':'state-file','bddl_sha256':'task-file'}])
    data['episodes']=[{'task_id':4,'state_index':0,'task_name':'task','prompt':'do task','reset_seed':123,
                       'initial_state':'state','settled_state':'settled','initial_processed_inputs':'pixels',
                       'decisions':[{'decision_index':0,'noise_record_id':'key','noise_sha256':'noise'}],
                       'policy_calls':1,'environment_steps':8,'success':True}]
    return data


def files(tmp_path,a,b):
    paths=[tmp_path/'a.json',tmp_path/'b.json']
    for p,v in zip(paths,[a,b]):p.write_text(json.dumps(v))
    return paths


def test_pilot_allows_different_later_actions_and_outcomes(tmp_path):
    a=sample('trt');b=sample('flashrt');b['episodes'][0]['success']=False
    result=compare(files(tmp_path,a,b))
    assert result['results']['trt']['successes']==1
    assert result['results']['flashrt']['successes']==0
    assert result['scope']=='task_subset'


@pytest.mark.parametrize('field',['initial_state','settled_state','initial_processed_inputs'])
def test_rejects_unmatched_start(tmp_path,field):
    a=sample('trt');b=sample('flashrt');b['episodes'][0][field]='changed'
    with pytest.raises(ValueError,match='Initial case differs'):compare(files(tmp_path,a,b))


def test_rejects_unmatched_noise(tmp_path):
    a=sample('trt');b=sample('flashrt');b['episodes'][0]['decisions'][0]['noise_sha256']='different'
    with pytest.raises(ValueError,match='noise'):compare(files(tmp_path,a,b))


def test_rejects_missing_case(tmp_path):
    a=sample('trt');b=sample('flashrt');b['episodes']=[]
    with pytest.raises(ValueError,match='cases'):compare(files(tmp_path,a,b))


@pytest.mark.parametrize('field,value', [('camera_keys',['image']), ('camera_ablation',True)])
def test_rejects_mixed_camera_protocols(tmp_path,field,value):
    a=sample('trt');b=sample('flashrt')
    for run in (a,b):run.update(camera_keys=['image','wrist_image'],camera_ablation=False)
    b[field]=value
    with pytest.raises(ValueError,match=field):compare(files(tmp_path,a,b))
