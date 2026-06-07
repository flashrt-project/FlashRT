import numpy as np
import torch


def test_rocm_backend_info_reports_runtime_capabilities():
    from flash_rt.hardware.rocm.backend import get_backend_info

    info = get_backend_info(probe_fp8_gemm=True)
    assert info.name == "rocm"
    assert info.extension_name == "flash_rt_rocm_kernels"
    assert info.available
    assert info.hip_version
    assert isinstance(info.supports_graph, bool)
    assert isinstance(info.supports_fp8_dtype, bool)
    assert isinstance(info.supports_fp8_gemm, bool)


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


def test_rocm_sdpa_attention_backend_protocol_slots():
    from flash_rt.hardware.rocm.attn_backend import RocmSdpaAttnBackend

    backend = RocmSdpaAttnBackend(num_views=1, encoder_seq_max=64, chunk_size=5)
    assert set(backend.sites()) == {"siglip", "encoder", "decoder"}
    assert backend.head_dim("encoder") == 256
    assert backend.num_q_heads("encoder") == 8
    assert backend.num_kv_heads("encoder") == 1

    siglip = backend.get_slot_ptrs("siglip", 0)
    assert siglip["Q"] == backend.vis_Q.data_ptr()
    assert siglip["O"] == backend.vis_O.data_ptr()

    encoder0 = backend.get_slot_ptrs("encoder", 0)
    encoder1 = backend.get_slot_ptrs("encoder", 1)
    assert encoder0["Q"] == backend.enc_Q.data_ptr()
    assert encoder1["K"] - encoder0["K"] == (
        backend.enc_K.stride(0) * backend.enc_K.element_size()
    )

    decoder = backend.get_slot_ptrs("decoder", 0)
    assert decoder["Q"] == backend.dec_Q.data_ptr()
    assert decoder["K"] == encoder0["K"]


def test_rocm_sdpa_attention_backend_matches_torch_reference():
    from flash_rt.hardware.rocm.attn_backend import RocmSdpaAttnBackend

    backend = RocmSdpaAttnBackend(num_views=2, encoder_seq_max=64, chunk_size=5)
    assert backend.active_backend_name in {"FLASH_ATTENTION", "auto"}
    backend.vis_Q.copy_(torch.randn_like(backend.vis_Q.float()).to(torch.bfloat16))
    backend.vis_K.copy_(torch.randn_like(backend.vis_K.float()).to(torch.bfloat16))
    backend.vis_V.copy_(torch.randn_like(backend.vis_V.float()).to(torch.bfloat16))

    out_ptr = backend.run("siglip", 0, 256)
    torch.cuda.synchronize()

    assert out_ptr == backend.vis_O.data_ptr()
    ref = torch.nn.functional.scaled_dot_product_attention(
        backend.vis_Q.transpose(1, 2),
        backend.vis_K.transpose(1, 2),
        backend.vis_V.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)
    torch.testing.assert_close(backend.vis_O.float(), ref.float(), rtol=2e-2, atol=3e-2)


def test_rocm_sdpa_attention_backend_encoder_and_decoder_shapes():
    from flash_rt.hardware.rocm.attn_backend import RocmSdpaAttnBackend

    backend = RocmSdpaAttnBackend(num_views=1, encoder_seq_max=64, chunk_size=5)
    backend.enc_Q[:32].copy_(
        torch.randn_like(backend.enc_Q[:32].float()).to(torch.bfloat16)
    )
    backend.enc_K[0, :32].copy_(
        torch.randn_like(backend.enc_K[0, :32].float()).to(torch.bfloat16)
    )
    backend.enc_V[0, :32].copy_(
        torch.randn_like(backend.enc_V[0, :32].float()).to(torch.bfloat16)
    )
    enc_ptr = backend.run("encoder", 0, 32)
    torch.cuda.synchronize()
    assert enc_ptr == backend.enc_O.data_ptr()
    assert backend.enc_O[:32].shape == (32, 8, 256)

    backend.dec_Q[:5].copy_(
        torch.randn_like(backend.dec_Q[:5].float()).to(torch.bfloat16)
    )
    backend.enc_K[0, 32:37].copy_(
        torch.randn_like(backend.enc_K[0, 32:37].float()).to(torch.bfloat16)
    )
    backend.enc_V[0, 32:37].copy_(
        torch.randn_like(backend.enc_V[0, 32:37].float()).to(torch.bfloat16)
    )
    dec_ptr = backend.run("decoder", 0, 5, kv_seq=37)
    torch.cuda.synchronize()
    assert dec_ptr == backend.dec_O.data_ptr()
    assert backend.dec_O[:5].shape == (5, 8, 256)
    assert backend.decoder_backend_name in {"FLASH_ATTENTION", "auto"}
