import copy
import pytest
from groot_compare.report import validate_pair


def fixture():
    data={k:1 for k in ('manifest_sha256','protocol_sha256','checkpoint_config_sha256','processor_config_sha256','weight_manifest_sha256','noise_seed','cpu_threads','input_contract')}
    data['records']=[{'id':'episode1/frame0','observation_sha256':'raw','noise_sha256':'noise','processed_inputs':{'pixels':'same'}}]
    return data


def test_rejects_same_raw_input_with_different_preprocessing():
    a=fixture();b=copy.deepcopy(a)
    b['records'][0]['processed_inputs']={'pixels':'different'}
    with pytest.raises(ValueError,match='processed_inputs'):validate_pair(a,b)


def test_rejects_incomplete_case_set():
    a=fixture();b=copy.deepcopy(a);b['records']=[]
    with pytest.raises(ValueError,match='IDs'):validate_pair(a,b)


def test_rejects_changed_noise_even_with_same_seed():
    a=fixture();b=copy.deepcopy(a);b['records'][0]['noise_sha256']='changed'
    with pytest.raises(ValueError,match='noise'):validate_pair(a,b)


def test_rejects_timing_unmatched_to_accuracy(tmp_path):
    import json
    from groot_compare.report import timing_summary
    a=fixture()
    a.update(completed=True,backend='flashrt',exclusive_gpu_asserted=True,warmup=20,iters_per_prompt=1,
             timing_samples=[{'prompt':'task','record_id':'episode1/frame0','iteration':0,'full_call_ms':10.}],
             groups=[{'prompt':'task','latency_ms':{'median':10.}}])
    b=copy.deepcopy(a)
    b['records'][0]['processed_inputs']={'pixels':'different'}
    accuracy=tmp_path/'accuracy.json';accuracy.write_text(json.dumps(a))
    timing=tmp_path/'timing.json';timing.write_text(json.dumps(b))
    with pytest.raises(ValueError,match='processed_inputs'):
        timing_summary([timing],[accuracy])
