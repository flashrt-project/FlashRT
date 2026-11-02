"""Checkpoint-free control-flow regressions; no native extension or GPU needed."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    # Replace only the native boundary; execute the real Python frontend and pipeline.
    import flash_rt
    for name, attrs in {
        "flash_rt.flash_rt_kernels": {"GemmRunner": lambda: object()},
        "flash_rt.flash_rt_fa2": {},
        "flash_rt.core.cuda_buffer": {"CudaBuffer": object},
        "flash_rt.core.cuda_graph": {"CUDAGraph": object},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        if name.count(".") == 1:
            monkeypatch.setitem(flash_rt.__dict__, name.rsplit(".",1)[1], module)
    pipe = _load("_molmo_pipe_test", "flash_rt/models/molmoact2/pipeline_rtx.py")
    monkeypatch.setitem(sys.modules, "flash_rt.models.molmoact2.pipeline_rtx", pipe)
    frontend = _load("_molmo_frontend_test", "flash_rt/frontends/torch/molmoact2_rtx.py")
    action = _load("_molmo_action_test", "flash_rt/models/molmoact2/action_mk.py")
    return frontend, action


def staging(mod):
    fe = mod.MolmoAct2TorchFrontendRtx.__new__(mod.MolmoAct2TorchFrontendRtx)
    fe._state_dim = 14
    fe.pipeline = SimpleNamespace(horizon=30, act_dim=14)
    fe._host_images = torch.empty(12, dtype=torch.uint8)
    fe._host_images_np = fe._host_images.numpy()
    fe._host_state_np = np.empty(14,np.float32)
    fe._host_noise_np = np.full((30,32),0x7fc0,np.uint16)  # BF16 NaN sentinel
    return fe


def test_default_noise_overwrites_uninitialized_memory(modules):
    mod, _ = modules
    fe = staging(mod)
    fe.upload_observations(np.zeros(12,np.uint8),np.zeros(14,np.float32))
    assert not fe._host_noise_np.any()


def test_explicit_noise_is_masked_and_not_reused_by_default(modules):
    pytest.importorskip("ml_dtypes")
    mod, _ = modules
    fe = staging(mod)
    inputs = (np.zeros(12,np.uint8),np.zeros(14,np.float32))
    noise = np.ones((30,32),np.float32)
    fe.upload_observations(*inputs,noise=noise)
    assert (fe._host_noise_np[:,:14] == 0x3f80).all()
    assert not fe._host_noise_np[:,14:].any()
    assert (noise == 1).all()
    fe.upload_observations(*inputs)
    assert not fe._host_noise_np.any()


@pytest.mark.parametrize("available,expected", [(36,36),(64,64)])
def test_action_grid_uses_native_residency_query(modules, available, expected):
    _, action = modules
    assert action._select_grid(SimpleNamespace(molmo_action_mk_max_grid=lambda:available),1) == expected


@pytest.mark.parametrize("available", [0,32,65,-1])
def test_action_grid_rejects_invalid_or_insufficient_residency(modules, available):
    _, action = modules
    with pytest.raises(RuntimeError,match="grid|resident"):
        action._select_grid(SimpleNamespace(molmo_action_mk_max_grid=lambda:available),1)


@pytest.mark.parametrize("precision", ["bf16","fp8","nvfp4"])
def test_prompt_replacement_invalidates_and_recalibrates(modules, monkeypatch, tmp_path, precision):
    mod, _ = modules
    fe = staging(mod)
    fe.precision, fe.norm_tag, fe.checkpoint_dir = precision, "robot", tmp_path
    fe._calibrated, fe.graph_recorded, fe.actions = True, True, np.ones(1)
    fe._stats = {"action_q01":np.zeros(14),"action_horizon":30}
    fe._host_state = torch.zeros(14)
    fe._host_noise = torch.zeros(30,32,dtype=torch.int16)
    fe._host_actions = torch.zeros(30,14)
    fe._images_buf = SimpleNamespace(ptr=SimpleNamespace(value=0))
    fe._embedding_ptr, fe.num_steps, fe._vis_fused_layers = 0, 10, []
    for name in ("_bf16","_fp8","_bias","_nvfp4","_skinny","_smallm", "_swiglu_fold","_fp8_nk","_act_fused","_vis_w2_pad"):
        setattr(fe,name,{})
    (tmp_path/"config.json").write_text(json.dumps({"state_token_start_id":1,"image_patch_id":2}))
    fe._prompt_ids = lambda *args: (np.array([1,2]),0,[0])
    fe._precompute = lambda *args: {}
    calls = []
    def factory(*args,**kwargs):
        return SimpleNamespace(bufs={"ids":SimpleNamespace(upload=lambda _:None)},horizon=30,act_dim=14,
            fp8_calibrated=False,fp8_scales={},run_pipeline=lambda **kw:calls.append("run"),
            prepare_scale_host=lambda:None,prepare_skinny_scales=lambda:None,
            prepare_vis_gelu_inv_scales=lambda:None,prepare_action_kernel=lambda:None)
    monkeypatch.setattr(mod,"MolmoAct2Pipeline",factory)
    original_empty = torch.empty
    monkeypatch.setattr(torch,"empty",lambda *a,**kw:original_empty(*a,**{k:v for k,v in kw.items() if k!="pin_memory"}))
    fe._cudart = SimpleNamespace(cudaStreamSynchronize=lambda _:0)
    fe.set_prompt("new task")
    assert not fe._calibrated and not fe.graph_recorded and fe.actions is None
    assert not fe.pipeline.fp8_calibrated
    with pytest.raises(RuntimeError,match="record_infer_graph"):
        fe.infer(np.zeros(12,np.uint8),np.zeros(14,np.float32))
    if precision != "bf16":
        with pytest.raises(RuntimeError,match="calibrate"):
            fe.record_infer_graph()
    fe.calibrate([(np.zeros(12,np.uint8),np.zeros(14,np.float32))])
    assert fe._calibrated
    assert calls == ([] if precision=="bf16" else ["run"])
    assert fe.pipeline.fp8_calibrated == (precision!="bf16")


def test_model_extra_contains_required_runtime_dependencies():
    tomllib = pytest.importorskip("tomllib")
    config = tomllib.loads((ROOT/"pyproject.toml").read_text())
    deps = config["project"]["optional-dependencies"]["molmoact2"]
    assert "flash-rt[torch]" in deps and "ml_dtypes" in deps
    assert any(dep.startswith("tokenizers") for dep in deps)


def test_infer_default_stages_zero_noise_before_replay(modules):
    mod, _ = modules
    fe = staging(mod)
    fe.graph_recorded = True
    fe._stream = SimpleNamespace(cuda_stream=0)
    fe._cudart = SimpleNamespace(cudaStreamSynchronize=lambda _:0)
    fe._host_actions_np = np.zeros((30,14),np.float32)
    staged = []
    fe.pipeline.replay = lambda _: staged.append(fe._host_noise_np.copy())
    result = fe.infer(np.zeros(12,np.uint8),np.zeros(14,np.float32))
    assert len(staged) == 1 and not staged[0].any()
    assert np.isfinite(result["actions"]).all()
