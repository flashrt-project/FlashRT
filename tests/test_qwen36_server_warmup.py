from examples.qwen36_openai_server import (
    Qwen36Engine,
    _dedupe_shapes,
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

    assert fe._long_mtp_prefill_tail_for_prompt(128) == 0
    assert fe._long_mtp_prefill_tail_for_prompt(512) == 512
    assert fe._long_mtp_prefill_tail_for_prompt(1024) == 2048
    assert fe._long_mtp_prefill_tail_for_prompt(4096) == 512
    assert fe._long_mtp_prefill_tail_for_prompt(8192) == 2048
    assert fe._long_mtp_prefill_tail_for_prompt(204800) == 2048

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
    fe._weights = SimpleNamespace(ptrs={'mtp': object()})
    fe._short_ctx_spec_max_seq = 2048
    fe._user_max_seq = 262208
    calls = []

    def record_extend(target):
        calls.append(int(target))

    fe._extend_mtp_cache_to = record_extend

    fe._ensure_long_mtp_cache_capacity(
        prompt_len=204800, max_new_tokens=64, K=6)
    assert calls[-1] == 2126

    fe._ensure_long_mtp_cache_capacity(
        prompt_len=204800, max_new_tokens=4096, K=6)
    assert calls[-1] == 6158


def test_long_graph_capture_waterline_can_be_disabled(monkeypatch):
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    fe = Qwen36TorchFrontendRtx.__new__(Qwen36TorchFrontendRtx)
    monkeypatch.setenv('FLASHRT_QWEN36_LONG_GRAPH_MIN_FREE_MB', '0')

    assert fe._long_tq_graph_capture_allowed() is True
