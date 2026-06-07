import numpy as np
import pytest


_FP8_HIPBLASLT_LINEAR_AVAILABLE = None


def _requires_fp8_hipblaslt_linear_algo():
    global _FP8_HIPBLASLT_LINEAR_AVAILABLE
    if _FP8_HIPBLASLT_LINEAR_AVAILABLE is None:
        import torch

        from flash_rt import flash_rt_rocm_kernels as rocm

        x = torch.randn(4, 32, device="cuda", dtype=torch.float32) * 0.5
        weight = torch.randn(16, 32, device="cuda", dtype=torch.float32) * 0.5
        x_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
        weight_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
        x8 = (x / x_scale).to(torch.float8_e4m3fnuz).contiguous()
        weight8 = (weight / weight_scale).to(torch.float8_e4m3fnuz).contiguous()
        try:
            rocm.hipblaslt_linear_fp8_e4m3fnuz_bf16(
                x8, weight8, x_scale, weight_scale
            )
            rocm.hip_sync()
        except RuntimeError as exc:
            if "hipBLASLt did not return a usable" not in str(exc):
                raise
            _FP8_HIPBLASLT_LINEAR_AVAILABLE = False
        else:
            _FP8_HIPBLASLT_LINEAR_AVAILABLE = True

    if not _FP8_HIPBLASLT_LINEAR_AVAILABLE:
        pytest.skip("hipBLASLt FP8 Linear algorithm is unavailable on this ROCm runtime")


def test_hip_buffer_roundtrip_float32():
    from flash_rt.core.hip_buffer import HipBuffer, sync

    arr = np.arange(1024, dtype=np.float32)
    buf = HipBuffer.from_numpy(arr)
    sync()

    out = buf.download_new(arr.shape, arr.dtype)
    np.testing.assert_array_equal(out, arr)


def test_hip_buffer_zero_float32():
    from flash_rt.core.hip_buffer import HipBuffer, sync

    buf = HipBuffer.device_zeros(257, np.float32)
    sync()

    out = buf.download_new((257,), np.float32)
    np.testing.assert_array_equal(out, np.zeros((257,), dtype=np.float32))


def test_rocm_raw_pointer_kernel_uses_hip_buffer_addresses():
    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.core.hip_buffer import HipBuffer, sync

    a = np.arange(1024, dtype=np.float32)
    b = np.full((1024,), 2.0, dtype=np.float32)
    a_buf = HipBuffer.from_numpy(a)
    b_buf = HipBuffer.from_numpy(b)
    out_buf = HipBuffer.device_empty(1024, np.float32)

    rocm.vector_add_f32_ptr(
        a_buf.ptr.value,
        b_buf.ptr.value,
        out_buf.ptr.value,
        a.size,
    )
    sync()

    out = out_buf.download_new(a.shape, a.dtype)
    np.testing.assert_array_equal(out, a + b)


def _patch_im2col_ref_u16(images: np.ndarray) -> np.ndarray:
    nv = images.shape[0]
    return (
        images.reshape(nv, 16, 14, 16, 14, 3)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(nv * 256, 588)
        .copy()
    )


def test_rocm_patch_im2col_raw_pointer_matches_reference():
    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.core.hip_buffer import HipBuffer, sync

    images = np.arange(2 * 224 * 224 * 3, dtype=np.uint16).reshape(
        2, 224, 224, 3
    )
    out = HipBuffer.device_empty(2 * 256 * 588, np.uint16)
    inp = HipBuffer.from_numpy(images)

    rocm.patch_im2col_ptr(inp.ptr.value, out.ptr.value, 2)
    sync()

    got = out.download_new((2 * 256, 588), np.uint16)
    np.testing.assert_array_equal(got, _patch_im2col_ref_u16(images))


def test_pi05_rocm_pipeline_patch_im2col_uses_owned_buffers():
    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm(num_views=1, max_prompt_len=48, num_steps=10)
    images = np.arange(224 * 224 * 3, dtype=np.uint16).reshape(1, 224, 224, 3)
    pipe.input_images_buf.upload(images)

    pipe.vision_patch_im2col(rocm)
    rocm.hip_sync()

    got = pipe.bufs["vision_patches"].download_new((256, 588), np.uint16)
    np.testing.assert_array_equal(got, _patch_im2col_ref_u16(images))


def test_pi05_rocm_pipeline_patch_embed_matches_torch_reference():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm(num_views=1, max_prompt_len=48, num_steps=10)
    images_u16 = np.arange(224 * 224 * 3, dtype=np.uint16).reshape(1, 224, 224, 3)
    images_bf16 = (images_u16.astype(np.float32) / 2048.0).astype(ml_dtypes.bfloat16)
    pipe.input_images_buf.upload(images_bf16)

    weight = (
        torch.arange(1152 * 588, device="cuda", dtype=torch.float32)
        .reshape(1152, 588)
        .remainder(17)
        .mul_(0.001)
        .to(torch.bfloat16)
        .contiguous()
    )
    bias = torch.zeros(1152, device="cuda", dtype=torch.bfloat16)

    pipe.vision_patch_embed(rocm, weight.data_ptr(), bias.data_ptr())
    rocm.hip_sync()

    got_u16 = pipe.bufs["vision_x"].download_new((256, 1152), np.uint16)
    got = torch.from_numpy(got_u16).view(torch.bfloat16).to(torch.float32)

    patches = torch.from_numpy(_patch_im2col_ref_u16(images_bf16.view(np.uint16))).view(
        torch.bfloat16
    )
    ref = torch.nn.functional.linear(
        patches.to(device="cuda", dtype=torch.bfloat16),
        weight,
        bias,
    ).cpu().float()
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-1)


def test_rocm_patch_embed_bias_pos_raw_pointer_matches_torch_reference():
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm

    s = 512
    d = 1152
    s_per_view = 256
    out = (
        torch.arange(s * d, device="cuda", dtype=torch.float32)
        .reshape(s, d)
        .remainder(31)
        .mul_(0.003)
        .to(torch.bfloat16)
        .contiguous()
    )
    bias = (
        torch.arange(d, device="cuda", dtype=torch.float32)
        .remainder(13)
        .mul_(0.002)
        .to(torch.bfloat16)
        .contiguous()
    )
    pos = (
        torch.arange(s_per_view * d, device="cuda", dtype=torch.float32)
        .reshape(s_per_view, d)
        .remainder(19)
        .mul_(0.001)
        .to(torch.bfloat16)
        .contiguous()
    )
    before = out.clone()

    rocm.patch_embed_bias_pos_bf16_ptr(
        out.data_ptr(),
        bias.data_ptr(),
        pos.data_ptr(),
        s,
        d,
        s_per_view,
    )
    rocm.hip_sync()

    ref = (before.float() + bias.float() + pos.repeat(2, 1).float()).to(
        torch.bfloat16
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=8e-3)


def test_pi05_rocm_pipeline_patch_embed_with_bias_pos_matches_torch_reference():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm(num_views=1, max_prompt_len=48, num_steps=10)
    images_u16 = np.arange(224 * 224 * 3, dtype=np.uint16).reshape(1, 224, 224, 3)
    images_bf16 = (images_u16.astype(np.float32) / 4096.0).astype(ml_dtypes.bfloat16)
    pipe.input_images_buf.upload(images_bf16)

    weight = (
        torch.arange(1152 * 588, device="cuda", dtype=torch.float32)
        .reshape(1152, 588)
        .remainder(23)
        .mul_(0.0007)
        .to(torch.bfloat16)
        .contiguous()
    )
    bias = (
        torch.arange(1152, device="cuda", dtype=torch.float32)
        .remainder(29)
        .mul_(0.0005)
        .to(torch.bfloat16)
        .contiguous()
    )
    pos = (
        torch.arange(256 * 1152, device="cuda", dtype=torch.float32)
        .reshape(256, 1152)
        .remainder(37)
        .mul_(0.0003)
        .to(torch.bfloat16)
        .contiguous()
    )

    pipe.vision_patch_embed_with_bias_pos(
        rocm,
        weight.data_ptr(),
        bias.data_ptr(),
        pos.data_ptr(),
    )
    rocm.hip_sync()

    got_u16 = pipe.bufs["vision_x"].download_new((256, 1152), np.uint16)
    got = torch.from_numpy(got_u16).view(torch.bfloat16).to(torch.float32)
    patches = torch.from_numpy(_patch_im2col_ref_u16(images_bf16.view(np.uint16))).view(
        torch.bfloat16
    )
    projected = torch.nn.functional.linear(
        patches.to(device="cuda", dtype=torch.bfloat16),
        weight,
        None,
    )
    ref = (projected.float() + bias.float() + pos.float()).to(torch.bfloat16).cpu()
    torch.testing.assert_close(got, ref.float(), rtol=2e-2, atol=2e-1)


def test_pi05_rocm_pipeline_vision_pre_attn_layer_norm_matches_torch_reference():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm(num_views=1, max_prompt_len=48, num_steps=10)
    x = (
        np.arange(256 * 1152, dtype=np.float32)
        .reshape(256, 1152)
        % 97
    ) * 0.01
    x_bf16 = x.astype(ml_dtypes.bfloat16)
    pipe.bufs["vision_x"].upload(x_bf16)

    weight = (
        torch.arange(1152, device="cuda", dtype=torch.float32)
        .remainder(31)
        .mul_(0.002)
        .add_(0.95)
        .to(torch.bfloat16)
        .contiguous()
    )
    bias = (
        torch.arange(1152, device="cuda", dtype=torch.float32)
        .remainder(17)
        .mul_(0.001)
        .to(torch.bfloat16)
        .contiguous()
    )

    pipe.vision_pre_attn_layer_norm(rocm, weight.data_ptr(), bias.data_ptr())
    rocm.hip_sync()

    got_u16 = pipe.bufs["vision_x_norm"].download_new((256, 1152), np.uint16)
    got = torch.from_numpy(got_u16).view(torch.bfloat16).to(torch.float32)
    x_t = torch.from_numpy(x_bf16.view(np.uint16)).view(torch.bfloat16)
    ref = torch.nn.functional.layer_norm(
        x_t.to(device="cuda").float(),
        (1152,),
        weight.float(),
        bias.float(),
        1e-5,
    ).to(torch.bfloat16)
    torch.testing.assert_close(got, ref.cpu().float(), rtol=0, atol=0.02)


def test_pi05_rocm_pipeline_vision_qkv_bf16_matches_torch_reference():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm(num_views=1, max_prompt_len=48, num_steps=10)
    x = (
        np.arange(256 * 1152, dtype=np.float32)
        .reshape(256, 1152)
        % 53
    ) * 0.004
    x_bf16 = x.astype(ml_dtypes.bfloat16)
    pipe.bufs["vision_x_norm"].upload(x_bf16)

    weight = (
        torch.arange(3 * 1152 * 1152, device="cuda", dtype=torch.float32)
        .reshape(3 * 1152, 1152)
        .remainder(11)
        .mul_(0.0008)
        .to(torch.bfloat16)
        .contiguous()
    )
    bias = (
        torch.arange(3 * 1152, device="cuda", dtype=torch.float32)
        .remainder(23)
        .mul_(0.0004)
        .to(torch.bfloat16)
        .contiguous()
    )

    pipe.vision_qkv_bf16(rocm, weight.data_ptr(), bias.data_ptr())
    rocm.hip_sync()

    got_u16 = pipe.bufs["vision_QKV"].download_new((256, 3 * 1152), np.uint16)
    got = torch.from_numpy(got_u16).view(torch.bfloat16).to(torch.float32)
    x_t = torch.from_numpy(x_bf16.view(np.uint16)).view(torch.bfloat16)
    ref = torch.nn.functional.linear(
        x_t.to(device="cuda", dtype=torch.bfloat16),
        weight,
        bias,
    ).cpu()
    torch.testing.assert_close(got, ref.float(), rtol=2e-2, atol=2e-1)


def test_pi05_rocm_pipeline_vision_qkv_split_matches_torch_reference():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm(num_views=2, max_prompt_len=48, num_steps=10)
    qkv = (
        np.arange(512 * 3 * 1152, dtype=np.float32)
        .reshape(512, 3 * 1152)
        % 251
    ).astype(ml_dtypes.bfloat16)
    pipe.bufs["vision_QKV"].upload(qkv)

    q = torch.empty((2, 256, 16, 72), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((2, 256, 16, 72), device="cuda", dtype=torch.bfloat16)
    v = torch.empty((2, 256, 16, 72), device="cuda", dtype=torch.bfloat16)

    pipe.vision_qkv_split(rocm, q.data_ptr(), k.data_ptr(), v.data_ptr())
    rocm.hip_sync()

    qkv_t = torch.from_numpy(qkv.view(np.uint16)).view(torch.bfloat16).reshape(
        512, 3 * 1152
    )
    torch.testing.assert_close(
        q.cpu().reshape(512, 1152), qkv_t[:, :1152], rtol=0, atol=0
    )
    torch.testing.assert_close(
        k.cpu().reshape(512, 1152), qkv_t[:, 1152 : 2 * 1152], rtol=0, atol=0
    )
    torch.testing.assert_close(
        v.cpu().reshape(512, 1152), qkv_t[:, 2 * 1152 :], rtol=0, atol=0
    )


def test_pi05_rocm_pipeline_vision_attn_output_residual_norm_matches_reference():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm(num_views=1, max_prompt_len=48, num_steps=10)
    residual = (
        np.arange(256 * 1152, dtype=np.float32).reshape(256, 1152) % 79
    ) * 0.003
    residual_bf16 = residual.astype(ml_dtypes.bfloat16)
    pipe.bufs["vision_x"].upload(residual_bf16)

    attn_out = (
        torch.arange(256 * 1152, device="cuda", dtype=torch.float32)
        .reshape(256, 1152)
        .remainder(47)
        .mul_(0.002)
        .to(torch.bfloat16)
        .contiguous()
    )
    weight = (
        torch.arange(1152 * 1152, device="cuda", dtype=torch.float32)
        .reshape(1152, 1152)
        .remainder(13)
        .mul_(0.0005)
        .to(torch.bfloat16)
        .contiguous()
    )
    out_bias = (
        torch.arange(1152, device="cuda", dtype=torch.float32)
        .remainder(17)
        .mul_(0.0007)
        .to(torch.bfloat16)
        .contiguous()
    )
    norm_weight = (
        torch.arange(1152, device="cuda", dtype=torch.float32)
        .remainder(19)
        .mul_(0.001)
        .add_(0.98)
        .to(torch.bfloat16)
        .contiguous()
    )
    norm_bias = (
        torch.arange(1152, device="cuda", dtype=torch.float32)
        .remainder(23)
        .mul_(0.0003)
        .to(torch.bfloat16)
        .contiguous()
    )

    pipe.vision_attn_output_residual_norm(
        rocm,
        attn_out.data_ptr(),
        weight.data_ptr(),
        out_bias.data_ptr(),
        norm_weight.data_ptr(),
        norm_bias.data_ptr(),
    )
    rocm.hip_sync()

    got_residual_u16 = pipe.bufs["vision_x"].download_new((256, 1152), np.uint16)
    got_norm_u16 = pipe.bufs["vision_x_norm"].download_new((256, 1152), np.uint16)
    got_residual = torch.from_numpy(got_residual_u16).view(torch.bfloat16).float()
    got_norm = torch.from_numpy(got_norm_u16).view(torch.bfloat16).float()

    residual_t = torch.from_numpy(residual_bf16.view(np.uint16)).view(torch.bfloat16)
    projected = torch.nn.functional.linear(attn_out, weight, None)
    updated = (residual_t.to("cuda").float() + projected.float() + out_bias.float()).to(
        torch.bfloat16
    )
    ref_norm = torch.nn.functional.layer_norm(
        updated.float(), (1152,), norm_weight.float(), norm_bias.float(), 1e-5
    ).to(torch.bfloat16)

    torch.testing.assert_close(got_residual, updated.cpu().float(), rtol=2e-2, atol=2e-1)
    torch.testing.assert_close(got_norm, ref_norm.cpu().float(), rtol=0, atol=0.02)


def test_pi05_rocm_pipeline_vision_ffn_up_gelu_matches_reference():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm(num_views=1, max_prompt_len=48, num_steps=10)
    x = (
        np.arange(256 * 1152, dtype=np.float32).reshape(256, 1152) % 43
    ) * 0.002
    x_bf16 = x.astype(ml_dtypes.bfloat16)
    pipe.bufs["vision_x_norm"].upload(x_bf16)

    weight = (
        torch.arange(4304 * 1152, device="cuda", dtype=torch.float32)
        .reshape(4304, 1152)
        .remainder(7)
        .mul_(0.0006)
        .to(torch.bfloat16)
        .contiguous()
    )
    bias = (
        torch.arange(4304, device="cuda", dtype=torch.float32)
        .remainder(11)
        .mul_(0.0004)
        .to(torch.bfloat16)
        .contiguous()
    )

    pipe.vision_ffn_up_gelu(rocm, weight.data_ptr(), bias.data_ptr())
    rocm.hip_sync()

    got_u16 = pipe.bufs["vision_hidden"].download_new((256, 4304), np.uint16)
    got = torch.from_numpy(got_u16).view(torch.bfloat16).float()
    x_t = torch.from_numpy(x_bf16.view(np.uint16)).view(torch.bfloat16)
    projected = torch.nn.functional.linear(
        x_t.to(device="cuda", dtype=torch.bfloat16),
        weight,
        None,
    )
    ref = torch.nn.functional.gelu(
        (projected.float() + bias.float()).to(torch.bfloat16).float(),
        approximate="tanh",
    ).to(torch.bfloat16)
    torch.testing.assert_close(got, ref.cpu().float(), rtol=2e-2, atol=2e-1)


def test_pi05_rocm_pipeline_vision_ffn_down_residual_norm_matches_reference():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm(num_views=1, max_prompt_len=48, num_steps=10)
    residual = (
        np.arange(256 * 1152, dtype=np.float32).reshape(256, 1152) % 61
    ) * 0.002
    hidden = (
        np.arange(256 * 4304, dtype=np.float32).reshape(256, 4304) % 37
    ) * 0.001
    residual_bf16 = residual.astype(ml_dtypes.bfloat16)
    hidden_bf16 = hidden.astype(ml_dtypes.bfloat16)
    pipe.bufs["vision_x"].upload(residual_bf16)
    pipe.bufs["vision_hidden"].upload(hidden_bf16)

    weight = (
        torch.arange(1152 * 4304, device="cuda", dtype=torch.float32)
        .reshape(1152, 4304)
        .remainder(5)
        .mul_(0.0005)
        .to(torch.bfloat16)
        .contiguous()
    )
    down_bias = (
        torch.arange(1152, device="cuda", dtype=torch.float32)
        .remainder(13)
        .mul_(0.0003)
        .to(torch.bfloat16)
        .contiguous()
    )
    norm_weight = (
        torch.arange(1152, device="cuda", dtype=torch.float32)
        .remainder(17)
        .mul_(0.001)
        .add_(0.99)
        .to(torch.bfloat16)
        .contiguous()
    )
    norm_bias = (
        torch.arange(1152, device="cuda", dtype=torch.float32)
        .remainder(19)
        .mul_(0.0002)
        .to(torch.bfloat16)
        .contiguous()
    )

    pipe.vision_ffn_down_residual_norm(
        rocm,
        weight.data_ptr(),
        down_bias.data_ptr(),
        norm_weight.data_ptr(),
        norm_bias.data_ptr(),
    )
    rocm.hip_sync()

    got_residual_u16 = pipe.bufs["vision_x"].download_new((256, 1152), np.uint16)
    got_norm_u16 = pipe.bufs["vision_x_norm"].download_new((256, 1152), np.uint16)
    got_residual = torch.from_numpy(got_residual_u16).view(torch.bfloat16).float()
    got_norm = torch.from_numpy(got_norm_u16).view(torch.bfloat16).float()

    residual_t = torch.from_numpy(residual_bf16.view(np.uint16)).view(torch.bfloat16)
    hidden_t = torch.from_numpy(hidden_bf16.view(np.uint16)).view(torch.bfloat16)
    projected = torch.nn.functional.linear(
        hidden_t.to(device="cuda", dtype=torch.bfloat16),
        weight,
        None,
    )
    updated = (
        residual_t.to("cuda").float() + projected.float() + down_bias.float()
    ).to(torch.bfloat16)
    ref_norm = torch.nn.functional.layer_norm(
        updated.float(), (1152,), norm_weight.float(), norm_bias.float(), 1e-5
    ).to(torch.bfloat16)

    torch.testing.assert_close(got_residual, updated.cpu().float(), rtol=2e-2, atol=2e-1)
    torch.testing.assert_close(got_norm, ref_norm.cpu().float(), rtol=0, atol=0.02)


def test_pi05_rocm_pipeline_vision_layer_bf16_zero_weights_structural_smoke():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm.with_sdpa_attention(
        num_views=1, max_prompt_len=48, num_steps=10
    )
    x = (
        np.arange(256 * 1152, dtype=np.float32).reshape(256, 1152) % 29
    ) * 0.001
    x_bf16 = x.astype(ml_dtypes.bfloat16)
    pipe.bufs["vision_x"].upload(x_bf16)
    pipe.bufs["vision_x_norm"].upload(x_bf16)

    qkv_w = torch.zeros((3 * 1152, 1152), device="cuda", dtype=torch.bfloat16)
    qkv_b = torch.zeros((3 * 1152,), device="cuda", dtype=torch.bfloat16)
    attn_o_w = torch.zeros((1152, 1152), device="cuda", dtype=torch.bfloat16)
    attn_o_b = torch.zeros((1152,), device="cuda", dtype=torch.bfloat16)
    pre_ffn_w = torch.ones((1152,), device="cuda", dtype=torch.bfloat16)
    pre_ffn_b = torch.zeros((1152,), device="cuda", dtype=torch.bfloat16)
    up_w = torch.zeros((4304, 1152), device="cuda", dtype=torch.bfloat16)
    up_b = torch.zeros((4304,), device="cuda", dtype=torch.bfloat16)
    down_w = torch.zeros((1152, 4304), device="cuda", dtype=torch.bfloat16)
    down_b = torch.zeros((1152,), device="cuda", dtype=torch.bfloat16)
    next_w = torch.ones((1152,), device="cuda", dtype=torch.bfloat16)
    next_b = torch.zeros((1152,), device="cuda", dtype=torch.bfloat16)

    pipe.vision_layer_bf16(
        rocm,
        qkv_weight_ptr=qkv_w.data_ptr(),
        qkv_bias_ptr=qkv_b.data_ptr(),
        attn_o_weight_ptr=attn_o_w.data_ptr(),
        attn_o_bias_ptr=attn_o_b.data_ptr(),
        pre_ffn_norm_weight_ptr=pre_ffn_w.data_ptr(),
        pre_ffn_norm_bias_ptr=pre_ffn_b.data_ptr(),
        ffn_up_weight_ptr=up_w.data_ptr(),
        ffn_up_bias_ptr=up_b.data_ptr(),
        ffn_down_weight_ptr=down_w.data_ptr(),
        ffn_down_bias_ptr=down_b.data_ptr(),
        next_pre_attn_norm_weight_ptr=next_w.data_ptr(),
        next_pre_attn_norm_bias_ptr=next_b.data_ptr(),
    )
    rocm.hip_sync()

    got_residual_u16 = pipe.bufs["vision_x"].download_new((256, 1152), np.uint16)
    got_norm_u16 = pipe.bufs["vision_x_norm"].download_new((256, 1152), np.uint16)
    got_residual = torch.from_numpy(got_residual_u16).view(torch.bfloat16).float()
    got_norm = torch.from_numpy(got_norm_u16).view(torch.bfloat16).float()

    x_t = torch.from_numpy(x_bf16.view(np.uint16)).view(torch.bfloat16).to("cuda")
    ref_norm = torch.nn.functional.layer_norm(
        x_t.float(), (1152,), next_w.float(), next_b.float(), 1e-5
    ).to(torch.bfloat16)
    torch.testing.assert_close(got_residual, x_t.cpu().float(), rtol=0, atol=0)
    torch.testing.assert_close(got_norm, ref_norm.cpu().float(), rtol=0, atol=0.02)


def test_pi05_rocm_pipeline_vision_encoder_bf16_one_layer_structural_smoke():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm.with_sdpa_attention(
        num_views=1, max_prompt_len=48, num_steps=10
    )
    images = np.zeros((1, 224, 224, 3), dtype=ml_dtypes.bfloat16)
    pipe.input_images_buf.upload(images)

    weights = {
        "vision_patch_embedding_w": torch.zeros(
            (1152, 588), device="cuda", dtype=torch.bfloat16
        ),
        "vision_patch_embedding_b": torch.zeros(
            (1152,), device="cuda", dtype=torch.bfloat16
        ),
        "vision_position_embedding": torch.zeros(
            (256, 1152), device="cuda", dtype=torch.bfloat16
        ),
        "vision_pre_attn_norm_w": [
            torch.ones((1152,), device="cuda", dtype=torch.bfloat16)
        ],
        "vision_pre_attn_norm_b": [
            torch.zeros((1152,), device="cuda", dtype=torch.bfloat16)
        ],
        "vision_pre_ffn_norm_w": [
            torch.ones((1152,), device="cuda", dtype=torch.bfloat16)
        ],
        "vision_pre_ffn_norm_b": [
            torch.zeros((1152,), device="cuda", dtype=torch.bfloat16)
        ],
        "vision_attn_qkv_w": [
            torch.zeros((3 * 1152, 1152), device="cuda", dtype=torch.bfloat16)
        ],
        "vision_attn_qkv_b": [
            torch.zeros((3 * 1152,), device="cuda", dtype=torch.bfloat16)
        ],
        "vision_attn_o_w": [
            torch.zeros((1152, 1152), device="cuda", dtype=torch.bfloat16)
        ],
        "vision_attn_o_b": [
            torch.zeros((1152,), device="cuda", dtype=torch.bfloat16)
        ],
        "vision_ffn_up_w": [
            torch.zeros((4304, 1152), device="cuda", dtype=torch.bfloat16)
        ],
        "vision_ffn_up_b": [
            torch.zeros((4304,), device="cuda", dtype=torch.bfloat16)
        ],
        "vision_ffn_down_w": [
            torch.zeros((1152, 4304), device="cuda", dtype=torch.bfloat16)
        ],
        "vision_ffn_down_b": [
            torch.zeros((1152,), device="cuda", dtype=torch.bfloat16)
        ],
    }

    pipe.vision_encoder_bf16(rocm, weights, vision_num_layers=1)
    rocm.hip_sync()

    got_u16 = pipe.bufs["vision_x"].download_new((256, 1152), np.uint16)
    got = torch.from_numpy(got_u16).view(torch.bfloat16).float()
    torch.testing.assert_close(got, torch.zeros_like(got), rtol=0, atol=0)


def test_pi05_rocm_pipeline_vision_project_to_encoder_matches_reference():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm(num_views=1, max_prompt_len=48, num_steps=10)
    x = (
        np.arange(256 * 1152, dtype=np.float32).reshape(256, 1152) % 73
    ) * 0.002
    x_bf16 = x.astype(ml_dtypes.bfloat16)
    pipe.bufs["vision_x"].upload(x_bf16)

    weights = {
        "vision_final_norm_w": (
            torch.arange(1152, device="cuda", dtype=torch.float32)
            .remainder(19)
            .mul_(0.001)
            .add_(0.98)
            .to(torch.bfloat16)
            .contiguous()
        ),
        "vision_final_norm_b": (
            torch.arange(1152, device="cuda", dtype=torch.float32)
            .remainder(23)
            .mul_(0.0003)
            .to(torch.bfloat16)
            .contiguous()
        ),
        "encoder_multi_modal_projector_w": (
            torch.arange(2048 * 1152, device="cuda", dtype=torch.float32)
            .reshape(2048, 1152)
            .remainder(7)
            .mul_(0.0004)
            .to(torch.bfloat16)
            .contiguous()
        ),
        "encoder_multi_modal_projector_b": (
            torch.arange(2048, device="cuda", dtype=torch.float32)
            .remainder(17)
            .mul_(0.0002)
            .to(torch.bfloat16)
            .contiguous()
        ),
    }

    pipe.vision_project_to_encoder_bf16(rocm, weights)
    rocm.hip_sync()

    got_u16 = pipe.bufs["encoder_x"].download_new((pipe.encoder_seq_len, 2048), np.uint16)
    got = torch.from_numpy(got_u16[:256]).view(torch.bfloat16).float()
    x_t = torch.from_numpy(x_bf16.view(np.uint16)).view(torch.bfloat16).to("cuda")
    normed = torch.nn.functional.layer_norm(
        x_t.float(),
        (1152,),
        weights["vision_final_norm_w"].float(),
        weights["vision_final_norm_b"].float(),
        1e-5,
    ).to(torch.bfloat16)
    ref = torch.nn.functional.linear(
        normed,
        weights["encoder_multi_modal_projector_w"],
        weights["encoder_multi_modal_projector_b"],
    )
    torch.testing.assert_close(got, ref.cpu().float(), rtol=2e-2, atol=2e-1)


def test_pi05_rocm_pipeline_encoder_qkv_rope_bf16_structural_smoke():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm.with_sdpa_attention(
        num_views=1, max_prompt_len=16, chunk_size=4, num_steps=4
    )
    enc = (
        np.arange(pipe.encoder_seq_len * 2048, dtype=np.float32)
        .reshape(pipe.encoder_seq_len, 2048)
        % 17
    ).astype(ml_dtypes.bfloat16)
    pipe.bufs["encoder_x"].upload(enc)

    weights = {
        "encoder_input_norm_w": [
            torch.zeros((2048,), device="cuda", dtype=torch.bfloat16)
        ],
        "encoder_attn_qkv_w": [
            torch.zeros((2560, 2048), device="cuda", dtype=torch.bfloat16)
        ],
    }

    pipe.encoder_qkv_rope_bf16(rocm, weights, 0)
    rocm.hip_sync()

    assert torch.isfinite(pipe.attn.enc_Q[: pipe.encoder_seq_len].float()).all()
    assert torch.count_nonzero(pipe.attn.enc_Q[: pipe.encoder_seq_len]).item() == 0
    assert torch.count_nonzero(pipe.attn.enc_K[0, : pipe.encoder_seq_len]).item() == 0
    assert torch.count_nonzero(pipe.attn.enc_V[0, : pipe.encoder_seq_len]).item() == 0


def test_pi05_rocm_pipeline_encoder_layer_bf16_zero_weight_smoke():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm.with_sdpa_attention(
        num_views=1, max_prompt_len=16, chunk_size=4, num_steps=4
    )
    enc = (
        np.arange(pipe.encoder_seq_len * 2048, dtype=np.float32)
        .reshape(pipe.encoder_seq_len, 2048)
        % 17
    ).astype(ml_dtypes.bfloat16)
    pipe.bufs["encoder_x"].upload(enc)
    pipe.encoder_seq_len = 2

    weights = {
        "encoder_input_norm_w": [
            torch.zeros((2048,), device="cuda", dtype=torch.bfloat16),
            torch.zeros((2048,), device="cuda", dtype=torch.bfloat16),
        ],
        "encoder_attn_qkv_w": [
            torch.zeros((2560, 2048), device="cuda", dtype=torch.bfloat16),
            torch.zeros((2560, 2048), device="cuda", dtype=torch.bfloat16),
        ],
        "encoder_attn_o_w": [
            torch.zeros((2048, 2048), device="cuda", dtype=torch.bfloat16),
        ],
        "encoder_post_attn_norm_w": [
            torch.zeros((2048,), device="cuda", dtype=torch.bfloat16),
        ],
        "encoder_ffn_gate_w": [
            torch.zeros((16384, 2048), device="cuda", dtype=torch.bfloat16),
        ],
        "encoder_ffn_up_w": [
            torch.zeros((16384, 2048), device="cuda", dtype=torch.bfloat16),
        ],
        "encoder_ffn_down_w": [
            torch.zeros((2048, 16384), device="cuda", dtype=torch.bfloat16),
        ],
    }

    pipe.encoder_bf16(rocm, weights, encoder_num_layers=2)
    rocm.hip_sync()

    got_u16 = pipe.bufs["encoder_x"].download_new((272, 2048), np.uint16)
    got = torch.from_numpy(got_u16[:2]).view(torch.bfloat16).float()
    ref = torch.from_numpy(enc[:2].view(np.uint16)).view(torch.bfloat16).float()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    assert torch.count_nonzero(pipe.attn.enc_Q[:2]).item() == 0
    assert torch.count_nonzero(pipe.attn.enc_K[1, :2]).item() == 0


def test_pi05_rocm_pipeline_decoder_step_bf16_zero_weight_smoke():
    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm.with_sdpa_attention(
        num_views=1, max_prompt_len=16, chunk_size=2, num_steps=1
    )
    pipe.encoder_seq_len = 2
    pipe.attn.enc_K.zero_()
    pipe.attn.enc_V.zero_()
    noise = (
        np.arange(pipe.chunk_size * 32, dtype=np.float32).reshape(pipe.chunk_size, 32)
        * 0.01
    ).astype(ml_dtypes.bfloat16)
    pipe.input_noise_buf.upload(noise)

    weights = {
        "decoder_action_in_proj_w": torch.zeros(
            (1024, 32), device="cuda", dtype=torch.bfloat16
        ),
        "decoder_action_in_proj_b": torch.zeros(
            (1024,), device="cuda", dtype=torch.bfloat16
        ),
        "decoder_attn_qkv_w": [
            torch.zeros((2560, 1024), device="cuda", dtype=torch.bfloat16)
        ],
        "decoder_attn_o_w": [
            torch.zeros((1024, 2048), device="cuda", dtype=torch.bfloat16)
        ],
        "decoder_ffn_gate_w": [
            torch.zeros((4096, 1024), device="cuda", dtype=torch.bfloat16)
        ],
        "decoder_ffn_up_w": [
            torch.zeros((4096, 1024), device="cuda", dtype=torch.bfloat16)
        ],
        "decoder_ffn_down_w": [
            torch.zeros((1024, 4096), device="cuda", dtype=torch.bfloat16)
        ],
        "decoder_action_out_proj_w": torch.zeros(
            (32, 1024), device="cuda", dtype=torch.bfloat16
        ),
        "decoder_action_out_proj_b": torch.zeros(
            (32,), device="cuda", dtype=torch.bfloat16
        ),
        "precomputed": {
            "time_emb": np.zeros((1, 2, 1024), dtype=np.uint16),
            "style_attn": np.zeros((1, 18, 2, 3 * 1024), dtype=np.uint16),
            "style_ffn": np.zeros((1, 18, 2, 3 * 1024), dtype=np.uint16),
            "style_final": np.zeros((1, 2, 3 * 1024), dtype=np.uint16),
        },
    }

    pipe.decoder_bf16(rocm, weights, decoder_num_layers=1)
    rocm.hip_sync()

    got_u16 = pipe.input_noise_buf.download_new((pipe.chunk_size, 32), np.uint16)
    got = torch.from_numpy(got_u16).view(torch.bfloat16).float()
    ref = torch.from_numpy(noise.view(np.uint16)).view(torch.bfloat16).float()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    assert torch.isfinite(pipe.attn.dec_O[: pipe.chunk_size].float()).all()


def test_pi05_rocm_pipeline_decoder_step_fp8_zero_weight_smoke():
    _requires_fp8_hipblaslt_linear_algo()

    import ml_dtypes
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm.with_sdpa_attention(
        num_views=1, max_prompt_len=16, chunk_size=2, num_steps=1
    )
    pipe.encoder_seq_len = 2
    pipe.attn.enc_K.zero_()
    pipe.attn.enc_V.zero_()
    noise = (
        np.arange(pipe.chunk_size * 32, dtype=np.float32).reshape(pipe.chunk_size, 32)
        * 0.01
    ).astype(ml_dtypes.bfloat16)

    def fp8_zero(shape):
        w = torch.zeros(shape, device="cuda", dtype=torch.float32)
        scale = torch.tensor([1.0e-8], device="cuda", dtype=torch.float32)
        return w.to(torch.float8_e4m3fnuz).contiguous(), scale

    weights = {
        "decoder_action_in_proj_w": torch.zeros(
            (1024, 32), device="cuda", dtype=torch.bfloat16
        ),
        "decoder_action_in_proj_b": torch.zeros(
            (1024,), device="cuda", dtype=torch.bfloat16
        ),
        "decoder_action_out_proj_w": torch.zeros(
            (32, 1024), device="cuda", dtype=torch.bfloat16
        ),
        "decoder_action_out_proj_b": torch.zeros(
            (32,), device="cuda", dtype=torch.bfloat16
        ),
        "precomputed": {
            "time_emb": np.zeros((1, 2, 1024), dtype=np.uint16),
            "style_attn": np.zeros((1, 18, 2, 3 * 1024), dtype=np.uint16),
            "style_ffn": np.zeros((1, 18, 2, 3 * 1024), dtype=np.uint16),
            "style_final": np.zeros((1, 2, 3 * 1024), dtype=np.uint16),
        },
        "fp8": {
            "decoder_attn_qkv_w_0": fp8_zero((2560, 1024)),
            "decoder_attn_o_w_0": fp8_zero((1024, 2048)),
            "decoder_ffn_gate_up_w_0": fp8_zero((8192, 1024)),
            "decoder_ffn_down_w_0": fp8_zero((1024, 4096)),
        },
    }

    pipe.input_noise_buf.upload(noise)
    pipe.decoder_fp8(rocm, weights, decoder_num_layers=1)
    rocm.hip_sync()
    assert "decoder_attn_qkv_w_0" in pipe.fp8_act_scales
    assert "decoder_ffn_down_w_0" in pipe.fp8_act_scales

    pipe.fp8_calibrated = True
    pipe.input_noise_buf.upload(noise)
    pipe.decoder_fp8(rocm, weights, decoder_num_layers=1)
    rocm.hip_sync()

    got_u16 = pipe.input_noise_buf.download_new((pipe.chunk_size, 32), np.uint16)
    got = torch.from_numpy(got_u16).view(torch.bfloat16).float()
    ref = torch.from_numpy(noise.view(np.uint16)).view(torch.bfloat16).float()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    assert torch.isfinite(pipe.attn.dec_O[: pipe.chunk_size].float()).all()


def test_pi05_rocm_fp8_quant_site_contract_full_model():
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    names = Pi05PipelineRocm.expected_fp8_scale_names()

    assert len(names) == 250
    assert len(set(names)) == 250
    assert names[0] == "vision_attn_qkv_w_0"
    assert "vision_ffn_down_w_26" in names
    assert "encoder_multi_modal_projector_w" in names
    assert "encoder_attn_qkv_w_17" in names
    assert "encoder_attn_o_w_17" not in names
    assert "encoder_ffn_gate_up_w_17" not in names
    assert "encoder_ffn_down_w_17" not in names
    assert "decoder_attn_qkv_w_17" in names
    assert "decoder_ffn_down_w_17" in names


def test_pi05_rocm_fp8_quant_site_contract_reduced_layers_and_coverage():
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    names = Pi05PipelineRocm.expected_fp8_scale_names(
        vision_num_layers=1,
        encoder_num_layers=1,
        decoder_num_layers=1,
    )
    assert names == (
        "vision_attn_qkv_w_0",
        "vision_attn_o_w_0",
        "vision_ffn_up_w_0",
        "vision_ffn_down_w_0",
        "encoder_multi_modal_projector_w",
        "encoder_attn_qkv_w_0",
        "decoder_attn_qkv_w_0",
        "decoder_attn_o_w_0",
        "decoder_ffn_gate_up_w_0",
        "decoder_ffn_down_w_0",
    )

    coverage = Pi05PipelineRocm.fp8_scale_coverage_for(
        [*names, "stale_site"],
        vision_num_layers=1,
        encoder_num_layers=1,
        decoder_num_layers=1,
    )
    assert coverage["expected_scale_count"] == 10
    assert coverage["actual_scale_count"] == 11
    assert coverage["missing_scales"] == ()
    assert coverage["unexpected_scales"] == ("stale_site",)

    coverage = Pi05PipelineRocm.fp8_scale_coverage_for(
        names[:-1],
        vision_num_layers=1,
        encoder_num_layers=1,
        decoder_num_layers=1,
    )
    assert coverage["missing_scales"] == ("decoder_ffn_down_w_0",)
    assert coverage["unexpected_scales"] == ()


def test_pi05_rocm_pipeline_bake_bf16_gemms_populates_plan_cache():
    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm(num_views=1, max_prompt_len=16, chunk_size=2, num_steps=1)
    rocm.hipblaslt_linear_plan_cache_clear()

    result = pipe.bake_bf16_gemms(rocm)
    rocm.hip_sync()

    assert result["plan_cache_size_before"] == 0
    assert result["plan_cache_size_after"] >= 12
    keys = rocm.hipblaslt_linear_plan_cache_keys()
    assert "linear_bf16:256x1152x588:bias=0" in keys
    assert "linear_bf16:272x2560x2048:bias=0" in keys
    assert "linear_bf16:2x1024x32:bias=1" in keys


def test_pi05_rocm_pipeline_bake_fp8_gemms_populates_algo_cache():
    _requires_fp8_hipblaslt_linear_algo()

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm(num_views=1, max_prompt_len=16, chunk_size=2, num_steps=1)
    rocm.hipblaslt_algo_cache_clear()

    result = pipe.bake_fp8_gemms(rocm)
    rocm.hip_sync()

    assert result["algo_cache_size_before"] == 0
    assert result["algo_cache_size_after"] >= 10
    keys = rocm.hipblaslt_algo_cache_keys()
    assert "linear_fp8_e4m3fnuz_bf16:256x3456x1152:bias=1" in keys
    assert "linear_fp8_e4m3fnuz_bf16:272x2560x2048:bias=0" in keys
    assert "linear_fp8_e4m3fnuz_bf16:2x1024x2048:bias=0" in keys


def test_pi05_rocm_decoder_static_fp8_mlp_smoke():
    _requires_fp8_hipblaslt_linear_algo()

    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import DEC_D, DEC_H, Pi05PipelineRocm

    pipe = Pi05PipelineRocm(num_views=1, max_prompt_len=16, chunk_size=2, num_steps=1)
    x = torch.randn(pipe.chunk_size, DEC_D, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    pipe.copy_device_to_buffer("x_normed_buf", x.data_ptr(), x.numel() * 2)

    def fp8_weight(shape):
        w = torch.randn(shape, device="cuda", dtype=torch.float32) * 0.05
        scale = torch.clamp(w.abs().max() / 240.0, min=1.0e-8).reshape(1)
        return (w / scale).to(torch.float8_e4m3fnuz).contiguous(), scale.contiguous()

    weights = {
        "fp8": {
            "decoder_ffn_gate_up_w_0": fp8_weight((2 * DEC_H, DEC_D)),
            "decoder_ffn_down_w_0": fp8_weight((DEC_D, DEC_H)),
        }
    }
    pipe.set_fp8_act_scale("decoder_ffn_gate_up_w_0", 0.05)
    pipe.set_fp8_act_scale("decoder_ffn_down_w_0", 0.05)

    pipe.decoder_mlp_fp8_static(rocm, weights, 0)
    rocm.hip_sync()

    got_u16 = pipe.bufs["x_normed_buf"].download_new((pipe.chunk_size, DEC_D), np.uint16)
    got = torch.from_numpy(got_u16).view(torch.bfloat16).float()
    assert torch.isfinite(got).all()
    assert got.abs().sum().item() > 0


def test_pi05_rocm_pipeline_allocates_fixed_buffers():
    from flash_rt.models.pi05.pipeline_rocm import (
        DEC_D,
        DEC_H,
        ENC_D,
        VIS_D,
        Pi05PipelineRocm,
    )

    pipe = Pi05PipelineRocm(num_views=3, max_prompt_len=48, num_steps=10)

    assert pipe.vision_seq == 3 * 256
    assert pipe.encoder_seq_len == 3 * 256 + 48
    assert pipe.input_images_buf.nbytes == 3 * 224 * 224 * 3 * 2
    assert pipe.input_encoder_x_buf.nbytes == pipe.encoder_seq_len * ENC_D * 2
    assert pipe.bufs["vision_x"].nbytes == pipe.vision_seq * VIS_D * 2
    assert pipe.bufs["decoder_x"].nbytes == pipe.chunk_size * DEC_D * 2
    assert pipe.bufs["encoder_act_fp8"].nbytes == pipe.encoder_seq_len * ENC_D
    assert pipe.bufs["decoder_act_fp8_large"].nbytes == pipe.chunk_size * DEC_H
    assert pipe.bufs["encoder_rope_weights"].ptr.value
    assert pipe.bufs["decoder_rope_weights"].ptr.value
