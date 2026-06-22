from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest


class _Buf:
    def __init__(self, ptr, nbytes):
        self.ptr = SimpleNamespace(value=ptr)
        self.nbytes = nbytes


class _Gemm:
    def __init__(self):
        self.autotune_bf16_nn = Mock(return_value=None)


class _CudaRt:
    def __init__(self):
        self.cudaDeviceSynchronize = Mock(return_value=0)


def _load_pi05_rtx_symbols():
    try:
        from flash_rt.models.pi05.pipeline_rtx_batched import Pi05BatchedPipeline
        from flash_rt.models.pi05.pipeline_rtx import (
            DEC_D,
            DEC_H,
            DEC_HD,
            DEC_NH,
            ENC_D,
            ENC_H,
            Pi05Pipeline,
            VIS_D,
            VIS_H,
            _fp8_nt_autotune_enabled,
        )
    except (ImportError, OSError) as exc:
        pytest.skip(f"Pi05 RTX imports unavailable: {exc}")
    return SimpleNamespace(
        DEC_D=DEC_D,
        DEC_H=DEC_H,
        DEC_HD=DEC_HD,
        DEC_NH=DEC_NH,
        ENC_D=ENC_D,
        ENC_H=ENC_H,
        Pi05BatchedPipeline=Pi05BatchedPipeline,
        Pi05Pipeline=Pi05Pipeline,
        VIS_D=VIS_D,
        VIS_H=VIS_H,
        _fp8_nt_autotune_enabled=_fp8_nt_autotune_enabled,
    )


def _setup_pi05_pipe(pipe_cls):
    pipe = pipe_cls.__new__(pipe_cls)
    pipe._gemms_autotuned = False
    pipe.use_fp8 = True
    pipe.fp8_calibrated = True
    pipe.use_int8_vision = False
    pipe.use_int8_encoder = False
    pipe.use_fp8_decoder = True
    pipe.use_int8_decoder = False
    pipe.num_views = 2
    pipe.vision_seq = 512
    pipe.encoder_seq_len = 692
    pipe.chunk_size = 10
    pipe.gemm = _Gemm()
    pipe._cudart = _CudaRt()
    pipe.weights = {
        "vision_patch_embedding_w": 11,
        "fp8": {
            "vision_attn_qkv_w_0": True,
            "vision_attn_o_w_0": True,
            "vision_ffn_up_w_0": True,
            "vision_ffn_down_w_0": True,
            "vision_projector_w": True,
            "encoder_attn_qkv_w_0": True,
            "encoder_attn_o_w_0": True,
            "encoder_ffn_gate_up_w_0": True,
            "encoder_ffn_down_w_0": True,
            "decoder_attn_qkv_w_0": True,
            "decoder_attn_o_w_0": True,
            "decoder_ffn_gate_up_w_0": True,
            "decoder_ffn_down_w_0": True,
        },
    }
    scale_names = [
        "vision_attn_qkv_w_0",
        "vision_attn_o_w_0",
        "vision_ffn_up_w_0",
        "vision_ffn_down_w_0",
        "vision_projector_w",
        "encoder_attn_qkv_w_0",
        "encoder_attn_o_w_0",
        "encoder_ffn_gate_up_w_0",
        "encoder_ffn_down_w_0",
        "decoder_attn_qkv_w_0",
        "decoder_attn_o_w_0",
        "decoder_ffn_gate_up_w_0",
        "decoder_ffn_down_w_0",
    ]
    pipe.fp8_act_scales = {
        name: _Buf(500 + idx, 4) for idx, name in enumerate(scale_names)
    }

    def _weight_fp8(self, name):
        return (1000 + hash(name) % 1000, 2000 + hash(name) % 1000)

    pipe._weight_fp8 = MethodType(_weight_fp8, pipe)
    return pipe


def _set_base_test_bufs(pipe, symbols):
    pipe.bufs = {
        "vision_patches": _Buf(101, 1),
        "vision_x": _Buf(102, 1),
        "vision_QKV": _Buf(103, 1),
        "vision_x_norm": _Buf(104, 1),
        "vision_hidden": _Buf(105, 1),
        "encoder_x": _Buf(106, 1),
        "encoder_QKV": _Buf(107, 1),
        "encoder_gate_merged": _Buf(108, 1),
        "encoder_x_norm": _Buf(109, 1),
        "decoder_QKV": _Buf(110, 1),
        "decoder_gate_merged": _Buf(111, 1),
        "x_normed_buf": _Buf(112, 1),
        "vis_act_fp8": _Buf(201, pipe.vision_seq * symbols.VIS_D),
        "vis_act_fp8_large": _Buf(202, pipe.vision_seq * symbols.VIS_H),
        "vis_act_scale": _Buf(203, 4),
        "enc_act_fp8": _Buf(301, pipe.encoder_seq_len * symbols.ENC_D),
        "enc_act_fp8_large": _Buf(302, pipe.encoder_seq_len * 2 * symbols.ENC_H),
        "enc_act_scale": _Buf(303, 4),
        "dec_act_fp8": _Buf(401, pipe.chunk_size * symbols.DEC_D),
        "dec_act_fp8_large": _Buf(402, pipe.chunk_size * 2 * symbols.DEC_H),
        "dec_act_scale": _Buf(403, 4),
    }


def _set_batched_test_bufs(pipe, symbols):
    pipe.B = 2
    batched_slack = pipe.B + 1
    pipe.bufs = {
        "vision_patches": _Buf(1, 1),
        "vision_x": _Buf(2, 1),
        "vision_QKV": _Buf(3, 1),
        "vision_x_norm": _Buf(4, 1),
        "vision_hidden": _Buf(5, 1),
        "encoder_x": _Buf(6, 1),
        "encoder_QKV": _Buf(7, 1),
        "encoder_gate_merged": _Buf(8, 1),
        "encoder_x_norm": _Buf(9, 1),
        "decoder_QKV": _Buf(10, 1),
        "decoder_gate_merged": _Buf(11, 1),
        "x_normed_buf": _Buf(12, 1),
        "vision_patches_b2": _Buf(101, 1),
        "vision_x_b2": _Buf(102, 1),
        "vision_QKV_b2": _Buf(103, 1),
        "vision_x_norm_b2": _Buf(104, 1),
        "vision_hidden_b2": _Buf(105, 1),
        "encoder_x_b2": _Buf(106, 1),
        "encoder_QKV_b2": _Buf(107, 1),
        "encoder_gate_merged_b2": _Buf(108, 1),
        "encoder_x_norm_b2": _Buf(109, 1),
        "decoder_QKV_b2": _Buf(110, 1),
        "decoder_gate_merged_b2": _Buf(111, 1),
        "x_normed_buf_b2": _Buf(112, 1),
        "vis_act_fp8": _Buf(201, pipe.vision_seq * symbols.VIS_D),
        "vis_act_fp8_large": _Buf(202, pipe.vision_seq * symbols.VIS_H),
        "vis_act_fp8_b2": _Buf(
            211, batched_slack * pipe.vision_seq * symbols.VIS_D),
        "vis_act_fp8_large_b2": _Buf(
            212, batched_slack * pipe.vision_seq * symbols.VIS_H),
        "vis_act_scale": _Buf(203, 4),
        "enc_act_fp8": _Buf(301, pipe.encoder_seq_len * symbols.ENC_D),
        "enc_act_fp8_large": _Buf(302, pipe.encoder_seq_len * 2 * symbols.ENC_H),
        "enc_act_fp8_b2": _Buf(
            311, batched_slack * pipe.encoder_seq_len * symbols.ENC_D),
        "enc_act_fp8_large_b2": _Buf(
            312, batched_slack * pipe.encoder_seq_len * 2 * symbols.ENC_H),
        "enc_act_scale": _Buf(303, 4),
        "dec_act_fp8": _Buf(401, pipe.chunk_size * symbols.DEC_D),
        "dec_act_fp8_large": _Buf(402, pipe.chunk_size * 2 * symbols.DEC_H),
        "dec_act_fp8_b2": _Buf(
            411, batched_slack * pipe.chunk_size * symbols.DEC_D),
        "dec_act_fp8_large_b2": _Buf(
            412, batched_slack * pipe.chunk_size * 2 * symbols.DEC_H),
        "dec_act_scale": _Buf(403, 4),
    }


def test_autotune_uses_large_decoder_fp8_scratch_for_attn_o():
    symbols = _load_pi05_rtx_symbols()
    pipe = _setup_pi05_pipe(symbols.Pi05Pipeline)
    _set_base_test_bufs(pipe, symbols)
    pipe._autotune_fp8_matmul = Mock()

    symbols.Pi05Pipeline.autotune_gemms(pipe)

    decoder_attn_o = [
        call.args for call in pipe._autotune_fp8_matmul.call_args_list
        if call.args[3:6] == (
            pipe.chunk_size, symbols.DEC_D, symbols.DEC_NH * symbols.DEC_HD)
    ]
    assert decoder_attn_o, "decoder attn_o FP8 autotune shape was not visited"
    assert decoder_attn_o[0][0] == pipe.bufs["dec_act_fp8_large"].ptr.value


def test_batched_autotune_uses_size_based_decoder_fp8_scratch_for_attn_o(
        monkeypatch):
    symbols = _load_pi05_rtx_symbols()
    pipe = _setup_pi05_pipe(symbols.Pi05BatchedPipeline)
    _set_batched_test_bufs(pipe, symbols)
    pipe._autotune_fp8_matmul = Mock()

    monkeypatch.setattr(
        symbols.Pi05Pipeline, "autotune_gemms", lambda self: None)
    symbols.Pi05BatchedPipeline.autotune_gemms(pipe)

    decoder_attn_o = [
        call.args for call in pipe._autotune_fp8_matmul.call_args_list
        if call.args[3:6] == (
            pipe.B * pipe.chunk_size,
            symbols.DEC_D,
            symbols.DEC_NH * symbols.DEC_HD,
        )
    ]
    assert decoder_attn_o, (
        "batched decoder attn_o FP8 autotune shape was not visited")
    assert decoder_attn_o[0][0] == pipe.bufs["dec_act_fp8_large_b2"].ptr.value


def test_batched_autotune_runs_b2_shapes_only_once(monkeypatch):
    symbols = _load_pi05_rtx_symbols()
    pipe = _setup_pi05_pipe(symbols.Pi05BatchedPipeline)
    _set_batched_test_bufs(pipe, symbols)
    pipe._autotune_fp8_matmul = Mock()

    monkeypatch.setattr(
        symbols.Pi05Pipeline, "autotune_gemms", lambda self: None)

    symbols.Pi05BatchedPipeline.autotune_gemms(pipe)
    first_count = pipe._autotune_fp8_matmul.call_count
    symbols.Pi05BatchedPipeline.autotune_gemms(pipe)

    assert first_count > 0
    assert pipe._autotune_fp8_matmul.call_count == first_count
    assert pipe._gemms_autotuned_b2 is True
    assert pipe._cudart.cudaDeviceSynchronize.call_count == 1


def test_fp8_nk_layout_dispatches_to_nt_entrypoints():
    symbols = _load_pi05_rtx_symbols()
    pipe = symbols.Pi05Pipeline.__new__(symbols.Pi05Pipeline)
    pipe.fp8_layout = "nk"
    pipe.gemm = SimpleNamespace(
        fp8_nt_dev=Mock(),
        autotune_fp8_nt_dev=Mock(),
        fp8_nn_dev=Mock(),
        autotune_fp8_nn_dev=Mock(),
    )

    pipe._fp8_matmul(11, 12, 13, 14, 15, 16, 17, 18, 19)
    pipe._autotune_fp8_matmul(21, 22, 23, 24, 25, 26, 27, 28)

    pipe.gemm.fp8_nt_dev.assert_called_once_with(
        11, 12, 13, 14, 15, 16, 17, 18, stream=19)
    pipe.gemm.autotune_fp8_nt_dev.assert_called_once_with(
        21, 22, 23, 24, 25, 26, 27, 28)
    pipe.gemm.fp8_nn_dev.assert_not_called()
    pipe.gemm.autotune_fp8_nn_dev.assert_not_called()


def test_fp8_nt_autotune_auto_skips_sm89(monkeypatch):
    symbols = _load_pi05_rtx_symbols()
    monkeypatch.delenv("FLASHRT_FP8_NT_AUTOTUNE", raising=False)
    assert not symbols._fp8_nt_autotune_enabled("rtx_sm89", "nk")
    assert symbols._fp8_nt_autotune_enabled("rtx_sm120", "nk")
    assert symbols._fp8_nt_autotune_enabled("rtx_sm89", "kn")


def test_fp8_nt_autotune_env_override(monkeypatch):
    symbols = _load_pi05_rtx_symbols()
    monkeypatch.setenv("FLASHRT_FP8_NT_AUTOTUNE", "force")
    assert symbols._fp8_nt_autotune_enabled("rtx_sm89", "nk")
    monkeypatch.setenv("FLASHRT_FP8_NT_AUTOTUNE", "safe")
    assert not symbols._fp8_nt_autotune_enabled("rtx_sm120", "nk")


def test_fp8_nk_layout_skips_nt_autotune_when_policy_disabled(monkeypatch):
    symbols = _load_pi05_rtx_symbols()
    monkeypatch.setenv("FLASHRT_FP8_NT_AUTOTUNE", "safe")
    pipe = symbols.Pi05Pipeline.__new__(symbols.Pi05Pipeline)
    pipe.fp8_layout = "nk"
    pipe._autotune_fp8_nt = symbols._fp8_nt_autotune_enabled("rtx_sm89", "nk")
    pipe.gemm = SimpleNamespace(
        fp8_nt_dev=Mock(),
        autotune_fp8_nt_dev=Mock(),
        fp8_nn_dev=Mock(),
        autotune_fp8_nn_dev=Mock(),
    )

    pipe._autotune_fp8_matmul(21, 22, 23, 24, 25, 26, 27, 28)

    pipe.gemm.autotune_fp8_nt_dev.assert_not_called()
    pipe.gemm.autotune_fp8_nn_dev.assert_not_called()


def test_gemmrunner_sm89_fp8_nt_surface_exists():
    try:
        from flash_rt import flash_rt_kernels as fvk
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"flash_rt_kernels is not built: {exc}")
    assert hasattr(fvk, "GemmRunner")
    missing = [
        name for name in ("fp8_nt_dev", "autotune_fp8_nt_dev")
        if not hasattr(fvk.GemmRunner, name)
    ]
    assert not missing, f"GemmRunner missing FP8 NT symbols: {missing}"
