"""CPU contract checks for the shared Pi0.5 tensor pipeline."""
import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

from flash_rt.models.pi05 import torch_pipeline as model
from flash_rt.hardware.torch_reference import TorchTensorOps, TorchGemmBackend, TorchAttentionBackend


@pytest.fixture
def small_model(monkeypatch):
    # Reduced widths keep the complete stage traversal test suitable for CPU CI.
    dims = dict(VIS_L=2, VIS_D=8, VIS_H=12, VIS_NH=2, VIS_HD=4,
                ENC_L=2, ENC_D=8, ENC_H=16, ENC_NH=2, ENC_HD=4,
                DEC_L=2, DEC_D=8, DEC_H=12, DEC_NH=2, DEC_HD=4, ACTION_DIM=4)
    for key, value in dims.items():
        monkeypatch.setattr(model, key, value)
    generator = torch.Generator().manual_seed(213)
    def rand(*shape):
        return (torch.randn(*shape, generator=generator) * .05).bfloat16()
    weights = {
        'vision_patch_embedding_w': rand(8, 3, 14, 14),
        'vision_patch_embedding_b': rand(8),
        'vision_position_embedding': rand(256, 8),
        'vision_pre_attn_norm_w': torch.ones(2, 8).bfloat16(),
        'vision_pre_attn_norm_b': rand(2, 8),
        'vision_pre_ffn_norm_w': torch.ones(2, 8).bfloat16(),
        'vision_pre_ffn_norm_b': rand(2, 8),
        'vision_attn_qkv_w': rand(2, 8, 24), 'vision_attn_qkv_b': rand(2, 24),
        'vision_attn_o_w': rand(2, 8, 8), 'vision_attn_o_b': rand(2, 8),
        'vision_ffn_up_w': rand(2, 8, 12), 'vision_ffn_up_b': rand(2, 12),
        'vision_ffn_down_w': rand(2, 12, 8), 'vision_ffn_down_b': rand(2, 8),
        'vision_final_norm_w': torch.ones(8).bfloat16(), 'vision_final_norm_b': rand(8),
        'encoder_multi_modal_projector_w': rand(8, 8), 'encoder_multi_modal_projector_b': rand(8),
        'encoder_attn_qkv_w': rand(2, 8, 16), 'encoder_attn_o_w': rand(2, 8, 8),
        'encoder_ffn_gate_up_w': rand(2, 8, 32), 'encoder_ffn_down_w': rand(2, 16, 8),
        'decoder_time_embeds': rand(2, 8),
        'decoder_time_mlp_in_w': rand(8, 8), 'decoder_time_mlp_in_b': rand(8),
        'decoder_time_mlp_out_w': rand(8, 8), 'decoder_time_mlp_out_b': rand(8),
        'decoder_action_in_proj_w': rand(4, 8), 'decoder_action_in_proj_b': rand(8),
        'decoder_action_out_proj_w': rand(8, 4), 'decoder_action_out_proj_b': rand(4),
        'decoder_attn_qkv_w': rand(2, 8, 16), 'decoder_attn_o_w': rand(2, 8, 8),
        'decoder_ffn_gate_up_w': rand(2, 8, 24), 'decoder_ffn_down_w': rand(2, 12, 8),
        'decoder_pre_attn_norm_mod_w': rand(2, 8, 24), 'decoder_pre_attn_norm_mod_b': rand(2, 24),
        'decoder_pre_ffn_norm_mod_w': rand(2, 8, 24), 'decoder_pre_ffn_norm_mod_b': rand(2, 24),
        'decoder_final_norm_mod_w': rand(8, 24), 'decoder_final_norm_mod_b': rand(24),
    }
    return SimpleNamespace(weights=weights, rand=rand, dims=dims)


@pytest.mark.parametrize('views', [2, 3])
@pytest.mark.parametrize('compact', [False, True])
def test_shared_pipeline_precompute_matches_runtime(small_model, views, compact):
    fixture = small_model
    arguments = dict(num_views=views, max_prompt_len=5, chunk_size=3, num_steps=2,
                     device='cpu', compact_encoder=compact)
    make = lambda precompute: model.Pi05TorchPipeline(
        fixture.weights, TorchTensorOps(), TorchGemmBackend(), TorchAttentionBackend(2),
        precompute_modulation=precompute, **arguments)
    images, prompt, noise = fixture.rand(views, 224, 224, 3), fixture.rand(3, 8), fixture.rand(3, 4)
    a, b = make(False), make(True)
    first = a.forward_with_inputs(images, prompt, 3, noise, capture_probes=True).clone()
    second = b.forward_with_inputs(images, prompt, 3, noise, capture_probes=True).clone()
    assert first.shape == (1, 3, 4)
    assert torch.isfinite(first).all()
    assert not torch.equal(first[0], noise)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(a.forward_with_inputs(images, prompt, 3, noise), first, rtol=0, atol=0)
    assert 'noise_s0' in a.probes and 'noise_s1' in a.probes
    for key in a.probes:
        torch.testing.assert_close(a.probes[key], b.probes[key], rtol=0, atol=0)


@pytest.mark.parametrize('compact', [False, True])
def test_shared_pipeline_decoder_only_reuses_last_encoder_cache(small_model, compact):
    fixture = small_model
    pipeline = model.Pi05TorchPipeline(
        fixture.weights, TorchTensorOps(), TorchGemmBackend(),
        TorchAttentionBackend(2), num_views=2, max_prompt_len=5,
        chunk_size=3, num_steps=2, device='cpu', compact_encoder=compact,
    )
    images = fixture.rand(2, 224, 224, 3)
    prompt = fixture.rand(3, 8)
    noise = fixture.rand(3, 4)

    with pytest.raises(RuntimeError, match='preceding full forward'):
        pipeline.forward_decode_only(noise)

    full = pipeline.forward_with_inputs(images, prompt, 3, noise).clone()
    cached = pipeline.forward_decode_only(noise).clone()
    torch.testing.assert_close(cached, full, rtol=0, atol=0)


def test_rdna_binding_has_no_model_traversal():
    from flash_rt.amd.models.pi05_rdna35.pipeline import Pi05PipelineRdna35
    assert Pi05PipelineRdna35._vision is model.Pi05TorchPipeline._vision
    assert Pi05PipelineRdna35._encoder is model.Pi05TorchPipeline._encoder
    assert Pi05PipelineRdna35._decoder is model.Pi05TorchPipeline._decoder
    tree = ast.parse(inspect.getsource(Pi05PipelineRdna35))
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    source = inspect.getsource(model)
    assert 'flash_rt.amd' not in source
    assert 'FLASHRT_RDNA' not in source
    assert '_rdna(' not in source


def test_model_package_is_lazy_in_fresh_process():
    import subprocess
    import sys
    code = "import sys; import flash_rt.models.pi05.torch_pipeline; assert 'flash_rt.models.pi05.pipeline_rtx' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_legacy_model_exports_keep_their_provider(monkeypatch):
    import sys
    import types
    import flash_rt.models.pi05 as package
    original = types.ModuleType('flash_rt.models.pi05.pipeline_rtx')
    for name in package.__all__:
        setattr(original, name, object())
        monkeypatch.delitem(package.__dict__, name, raising=False)
    monkeypatch.setitem(sys.modules, original.__name__, original)
    try:
        for name in package.__all__:
            assert getattr(package, name) is getattr(original, name)
    finally:
        for name in package.__all__:
            package.__dict__.pop(name, None)
