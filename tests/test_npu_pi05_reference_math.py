"""CPU checks against independent model operations, not a self-generated golden."""
import torch
import torch.nn.functional as F

from flash_rt.npu.models.pi05 import pipeline


def test_vision_attention_preserves_token_head_and_view_axes():
    generator = torch.Generator().manual_seed(27)
    q, k, v = [torch.randn(2, 7, 12, generator=generator) for _ in range(3)]
    expected = []
    for view in range(2):
        qh, kh, vh = [x[view].reshape(7, 3, 4).transpose(0, 1)
                      for x in (q, k, v)]
        expected.append(F.scaled_dot_product_attention(qh, kh, vh)
                        .transpose(0, 1).reshape(7, 12))
    torch.testing.assert_close(pipeline.vision_attention(q, k, v, 3),
                               torch.stack(expected), atol=1e-6, rtol=1e-6)
    changed = v.clone()
    changed[1].add_(100)
    torch.testing.assert_close(pipeline.vision_attention(q, k, changed, 3)[0],
                               expected[0], atol=1e-6, rtol=1e-6)


def test_wrapped_safetensors_keys_are_read_before_prefix_removal(tmp_path):
    from safetensors.torch import save_file
    path = tmp_path / "weights.safetensors"
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    save_file({"model.proj.weight": value}, path)
    loaded = pipeline.load_weights_fp32(path)
    assert list(loaded) == ["proj.weight"]
    torch.testing.assert_close(loaded["proj.weight"], value)


def test_encoder_output_projection_affects_next_layer_cache(monkeypatch):
    for key, value in {"ENC_L": 2, "ENC_D": 8, "ENC_HD": 4,
                       "ENC_NH": 2, "ENC_NKV": 1}.items():
        monkeypatch.setattr(pipeline, key, value)
    generator = torch.Generator().manual_seed(19)
    weights = {}
    for layer in range(2):
        prefix = f"{pipeline._EP}.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            weights[f"{prefix}.{norm}.weight"] = torch.zeros(8)
        for proj, rows in (("q_proj", 8), ("k_proj", 4), ("v_proj", 4), ("o_proj", 8)):
            weights[f"{prefix}.self_attn.{proj}.weight"] = torch.randn(rows, 8, generator=generator)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            weights[f"{prefix}.mlp.{proj}.weight"] = torch.zeros(8, 8)
    x = torch.randn(3, 8, generator=generator)
    key = f"{pipeline._EP}.0.self_attn.o_proj.weight"
    weights[key].zero_()
    zero_cache = pipeline.encoder_pass(x, weights)
    weights[key] = torch.eye(8)
    projected_cache = pipeline.encoder_pass(x, weights)
    torch.testing.assert_close(zero_cache[0][0], projected_cache[0][0])
    assert not torch.allclose(zero_cache[1][0], projected_cache[1][0])


def test_setup_head_padding_preserves_full_attention_projection(monkeypatch):
    from flash_rt.npu.models.pi05 import fast
    for key, value in {"VIS_L": 1, "VIS_D": 6, "VIS_NH": 2, "VIS_HD": 3}.items():
        monkeypatch.setattr(fast, key, value)
    generator = torch.Generator().manual_seed(43)
    prefix = f"{fast._VP}.encoder.layers.0.self_attn"
    weights = {}
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        weights[f"{prefix}.{name}.weight"] = torch.randn(6, 6, generator=generator, dtype=torch.float64)
        weights[f"{prefix}.{name}.bias"] = torch.randn(6, generator=generator, dtype=torch.float64)
    padded = fast.make_vision_padded_weights(weights, padded_head_dim=4)
    x = torch.randn(2, 5, 6, generator=generator, dtype=torch.float64)

    def full_attention(w, width):
        q, k, v = [F.linear(x, w[f"{prefix}.{name}.weight"], w[f"{prefix}.{name}.bias"])
                   .reshape(2, 5, 2, width).transpose(1, 2)
                   for name in ("q_proj", "k_proj", "v_proj")]
        out = F.scaled_dot_product_attention(q, k, v, scale=3 ** -0.5)
        return F.linear(out.transpose(1, 2).reshape(2, 5, 2 * width),
                        w[f"{prefix}.out_proj.weight"], w[f"{prefix}.out_proj.bias"])

    torch.testing.assert_close(full_attention(padded, 4), full_attention(weights, 3),
                               atol=1e-12, rtol=1e-12)
