"""Default infer path without checkpoint/native kernels, including warmup reads."""
from types import SimpleNamespace
import numpy as np
import pytest

torch=pytest.importorskip("torch")


@pytest.mark.parametrize("graph",[False,True])
def test_default_noise_is_initialized_before_first_read_and_reused(monkeypatch,graph):
    from flash_rt.frontends.torch import vla_jepa_rtx as mod
    fe=mod.VlaJepaTorchFrontendRtx.__new__(mod.VlaJepaTorchFrontendRtx)
    fe.chunk_size,fe.action_dim,fe.use_cuda_graph=7,7,graph
    empty=torch.empty
    monkeypatch.setattr(torch,"empty",lambda *a,**kw:empty(*a,**{k:v for k,v in kw.items() if k!="pin_memory"}))
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    monkeypatch.setattr(mod,"_require_kernels",lambda _:None)
    fe._pin_frames=torch.zeros(2,3,2,2,dtype=torch.uint8)
    fe._pin_state=torch.zeros(8)
    fe._pin_out=torch.empty(7,7)
    actions=torch.full((7,7),float("nan"),dtype=torch.bfloat16)
    out=torch.empty(7,7)
    seen=[]
    def forward():
        seen.append(actions.clone())
        out.copy_(actions)
        fe._pin_out.copy_(out)
        actions.fill_(float("nan"))  # model consumes/mutates its start state
    fe.pipe=SimpleNamespace(actions=actions,raw_frames=torch.zeros_like(fe._pin_frames),
        raw_state=torch.zeros_like(fe._pin_state),out_actions=out,forward=forward,_kernels=lambda:None)
    old=torch.are_deterministic_algorithms_enabled()
    fill=torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory=True
        fe._initialize_noise()
    finally:
        torch.use_deterministic_algorithms(old)
        torch.utils.deterministic.fill_uninitialized_memory=fill
    expected=torch.from_numpy(np.random.default_rng(0).standard_normal((7,7)).astype(np.float32)).bfloat16()
    assert torch.equal(fe._pin_noise,expected) and torch.isfinite(fe._pin_noise).all()
    captures=[]
    def record():
        assert torch.isfinite(fe._pin_noise).all()
        if not captures:
            fe._upload_pinned();forward();captures.append(True)
    def replay():fe._upload_pinned();forward()
    fe.record_graph=record
    fe._graph=SimpleNamespace(replay=replay)
    inputs=(np.zeros((2,2,2,3),np.uint8),np.zeros(8,np.float32))
    first=fe.infer(*inputs)
    second=fe.infer()  # no observation setter; Euler output must not become new noise
    assert np.isfinite(first).all() and np.array_equal(first,second)
    assert all(torch.equal(item,expected) for item in seen)
    fe.set_noise(np.ones((7,7),np.float32))
    assert (fe.infer()==1).all()
    fe.set_noise()
    assert np.array_equal(fe.infer(),first)
