from examples.qwen36_openai_server import (
    Qwen36Engine,
    _dedupe_shapes,
    _long_warmup_flags,
    _parse_warmup_shapes,
    _warmup_preset_shapes,
)


def test_warmup_preset_auto_respects_max_seq():
    shapes = _warmup_preset_shapes('auto', 32768)
    assert (8, 64) in shapes
    assert (128, 64) in shapes
    assert (512, 64) in shapes
    assert (2048, 64) in shapes
    assert (4096, 64) in shapes
    assert (8192, 64) in shapes
    assert (16384, 64) in shapes
    assert (32768, 64) not in shapes


def test_warmup_preset_all_covers_long_buckets_when_they_fit():
    shapes = _warmup_preset_shapes('all', 262208)
    assert (2048, 64) in shapes
    assert (32768, 64) in shapes
    assert (65536, 64) in shapes
    assert (131072, 64) in shapes
    assert (204800, 64) in shapes
    assert (262144, 16) in shapes


def test_parse_and_dedupe_custom_shapes():
    shapes = _dedupe_shapes(
        _parse_warmup_shapes('4096:64,8192:64,4096:64'))
    assert shapes == [(4096, 64), (8192, 64)]


def test_long_server_warmup_modes(monkeypatch):
    monkeypatch.delenv('FLASHRT_QWEN36_SERVER_LONG_WARMUP', raising=False)
    assert _long_warmup_flags() == (True, False)

    monkeypatch.setenv(
        'FLASHRT_QWEN36_SERVER_LONG_WARMUP', 'prefill_graphs')
    assert _long_warmup_flags() == (False, True)

    monkeypatch.setenv(
        'FLASHRT_QWEN36_SERVER_LONG_WARMUP', 'all_graphs')
    assert _long_warmup_flags() == (True, True)

    monkeypatch.setenv('FLASHRT_QWEN36_SERVER_LONG_WARMUP', 'off')
    assert _long_warmup_flags() == (False, False)


def test_qwen36_prefill_gdn_backend_defaults_to_native_wy(monkeypatch):
    from flash_rt.frontends.torch.qwen36_rtx import (
        _qwen36_tq_prefill_gdn_backend,
    )

    monkeypatch.delenv('FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND',
                       raising=False)
    assert _qwen36_tq_prefill_gdn_backend() == 'wy_lt'

    monkeypatch.setenv(
        'FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND', 'native')
    assert _qwen36_tq_prefill_gdn_backend() == 'native'

    monkeypatch.setenv(
        'FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND', 'fla_chunk')
    import pytest
    with pytest.raises(ValueError, match='no longer supports'):
        _qwen36_tq_prefill_gdn_backend()


def test_server_chat_template_disables_thinking_by_default():
    class FakeTokenizer:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, normalized, **kwargs):
            self.kwargs = kwargs
            return 'rendered'

    class FakeFrontend:
        def __init__(self):
            self._tokenizer = FakeTokenizer()

    engine = Qwen36Engine.__new__(Qwen36Engine)
    engine.fe = FakeFrontend()

    assert engine._render_chat([{'role': 'user', 'content': 'hi'}], None) == (
        'rendered')
    assert engine.fe._tokenizer.kwargs['enable_thinking'] is False

    engine._render_chat(
        [{'role': 'user', 'content': 'hi'}], None, enable_thinking=True)
    assert engine.fe._tokenizer.kwargs['enable_thinking'] is True


def test_long_mtp_tail_auto_policy(monkeypatch):
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    monkeypatch.delenv('FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL',
                       raising=False)
    fe = Qwen36TorchFrontendRtx.__new__(Qwen36TorchFrontendRtx)
    fe._weights = type('Weights', (), {
        'ptrs': {'mtp': {'k_proj_w_bf16': 1}},
    })()

    assert fe._long_mtp_prefill_tail_for_prompt(128) == 0
    assert fe._long_mtp_prefill_tail_for_prompt(512) == 512
    assert fe._long_mtp_prefill_tail_for_prompt(1024) == 2048
    assert fe._long_mtp_prefill_tail_for_prompt(4096) == 512
    assert fe._long_mtp_prefill_tail_for_prompt(8192) == 2048
    assert fe._long_mtp_prefill_tail_for_prompt(204800) == 2048

    monkeypatch.setenv('FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL', '512')
    assert fe._long_mtp_prefill_tail_for_prompt(204800) == 512


def test_long_mtp_tail_auto_disables_without_bf16_kv(monkeypatch):
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    monkeypatch.delenv('FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL',
                       raising=False)
    fe = Qwen36TorchFrontendRtx.__new__(Qwen36TorchFrontendRtx)

    fe._weights = type('Weights', (), {'ptrs': {'mtp': None}})()
    assert fe._long_mtp_prefill_tail_for_prompt(204800) == 0

    fe._weights = type('Weights', (), {'ptrs': {'mtp': {}}})()
    assert fe._long_mtp_prefill_tail_for_prompt(204800) == 0

    monkeypatch.setenv('FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL', '512')
    assert fe._long_mtp_prefill_tail_for_prompt(204800) == 512


def test_long_tq_effective_k_uses_measured_context_buckets(monkeypatch):
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    monkeypatch.delenv('FLASHRT_QWEN36_TQ_SPEC_K', raising=False)
    fe = Qwen36TorchFrontendRtx.__new__(Qwen36TorchFrontendRtx)

    assert fe._long_tq_effective_k(512, 6) == 4
    assert fe._long_tq_effective_k(1024, 6) == 5
    assert fe._long_tq_effective_k(2048, 6) == 6
    assert fe._long_tq_effective_k(4096, 6) == 3
    assert fe._long_tq_effective_k(8192, 6) == 5
    assert fe._long_tq_effective_k(16384, 7) == 7
    assert fe._long_tq_effective_k(32768, 6) == 6
    assert fe._long_tq_effective_k(65536, 6) == 7
    assert fe._long_tq_effective_k(131072, 6) == 7
    assert fe._long_tq_effective_k(204800, 6) == 6

    assert fe._long_tq_effective_k(65536, 5) == 5
    monkeypatch.setenv('FLASHRT_QWEN36_TQ_SPEC_K', '4')
    assert fe._long_tq_effective_k(65536, 6) == 4


def test_long_ctx_route_uses_prompt_bucket_before_total_length():
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    fe = Qwen36TorchFrontendRtx.__new__(Qwen36TorchFrontendRtx)
    fe._long_ctx_mode = True
    fe._long_ctx_route_min_seq = 512
    fe._short_ctx_spec_max_seq = 2048

    assert fe._should_use_long_ctx_route(128, 512) is False
    assert fe._should_use_long_ctx_route(511, 64) is False
    assert fe._should_use_long_ctx_route(512, 64) is True
    assert fe._should_use_long_ctx_route(128, 2048) is True


def test_fp8_xqa_auto_bucket_policy():
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    fe = Qwen36TorchFrontendRtx.__new__(Qwen36TorchFrontendRtx)

    assert not fe._fp8_xqa_auto_bucket_enabled(4, 4096)
    assert fe._fp8_xqa_auto_bucket_enabled(5, 8192)
    assert not fe._fp8_xqa_auto_bucket_enabled(6, 16384)
    assert fe._fp8_xqa_auto_bucket_enabled(5, 32768)
    assert fe._fp8_xqa_auto_bucket_enabled(8, 65536)


def test_long_mtp_cache_capacity_is_compact(monkeypatch):
    from types import SimpleNamespace

    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    monkeypatch.delenv('FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL',
                       raising=False)
    fe = Qwen36TorchFrontendRtx.__new__(Qwen36TorchFrontendRtx)
    fe._weights = SimpleNamespace(ptrs={'mtp': {}})
    fe._short_ctx_spec_max_seq = 2048
    fe._user_max_seq = 262208
    calls = []

    def record_extend(target):
        calls.append(int(target))

    fe._extend_mtp_cache_to = record_extend

    fe._ensure_long_mtp_cache_capacity(
        prompt_len=204800, max_new_tokens=64, K=6)
    assert calls[-1] == 2048

    fe._ensure_long_mtp_cache_capacity(
        prompt_len=204800, max_new_tokens=4096, K=6)
    assert calls[-1] == 4110

    fe._weights = SimpleNamespace(ptrs={'mtp': {'k_proj_w_bf16': 1}})
    fe._ensure_long_mtp_cache_capacity(
        prompt_len=204800, max_new_tokens=64, K=6)
    assert calls[-1] == 2126


def test_long_graph_capture_waterline_can_be_disabled(monkeypatch):
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    fe = Qwen36TorchFrontendRtx.__new__(Qwen36TorchFrontendRtx)
    monkeypatch.setenv('FLASHRT_QWEN36_LONG_GRAPH_MIN_FREE_MB', '0')

    assert fe._long_tq_graph_capture_allowed() is True


def test_long_prefill_graph_capture_has_ctx_ceiling(monkeypatch):
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    fe = Qwen36TorchFrontendRtx.__new__(Qwen36TorchFrontendRtx)
    monkeypatch.setenv('FLASHRT_QWEN36_LONG_GRAPH_MIN_FREE_MB', '0')
    monkeypatch.delenv('FLASHRT_QWEN36_LONG_PREFILL_GRAPH_MAX_CTX',
                       raising=False)

    assert fe._long_prefill_graph_capture_allowed(131072) is True
    assert fe._long_prefill_graph_capture_allowed(204800) is False

    monkeypatch.setenv('FLASHRT_QWEN36_LONG_PREFILL_GRAPH_MAX_CTX', '0')
    assert fe._long_prefill_graph_capture_allowed(262144) is True


def test_clear_graphs_drops_long_prefill_caches():
    import collections

    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    fe = Qwen36TorchFrontendRtx.__new__(Qwen36TorchFrontendRtx)
    fe._captured_prefill_graphs_tq = collections.OrderedDict(
        {(0, 8, 'none'): object()})
    fe._captured_prefill_graphs_fp8kv = collections.OrderedDict(
        {(0, 8, 'none'): object()})

    fe.clear_graphs()

    assert fe._captured_prefill_graphs_tq == collections.OrderedDict()
    assert fe._captured_prefill_graphs_fp8kv == collections.OrderedDict()
