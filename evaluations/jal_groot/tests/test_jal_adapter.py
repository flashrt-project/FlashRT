"""CPU-only tests for explicit noise injection and captured-buffer refresh."""

from types import SimpleNamespace
import unittest

from groot_compare.backends.jal import _install_noise_sampler, _refresh_trt_noise


class Tensor:
    def __init__(self, value, shape=(1, 40, 132), dtype="bf16", device="cuda:0"):
        self.value, self.shape, self.dtype, self.device = value, shape, dtype, device
        self.copies = 0

    def clone(self):
        return Tensor(self.value, self.shape, self.dtype, self.device)

    def copy_(self, other):
        self.value = other.value
        self.copies += 1
        return self


class FakeTorch:
    def __init__(self):
        self.random_calls = 0

    def randn(self, **kwargs):
        self.random_calls += 1
        return Tensor(-999, kwargs["size"], kwargs["dtype"], kwargs["device"])

    def no_grad(self):
        return lambda fn: fn

    def unchanged_operation(self, value):
        return value


torch = FakeTorch()


class Head:
    def get_action_with_features(self, batch=1):
        actions = torch.randn(size=(batch, 40, 132), dtype="bf16", device="cuda:0")
        return torch.unchanged_operation(actions)


class UnsupportedHead:
    def get_action_with_features(self):
        return 0


class NoiseInjectionTests(unittest.TestCase):
    def setUp(self):
        torch.random_calls = 0

    def test_private_proxy_preserves_code_and_global_torch(self):
        head = Head()
        original_code = head.get_action_with_features.__func__.__code__
        proxy, original, digest = _install_noise_sampler(head, torch)
        proxy.noise = Tensor(17)
        result = head.get_action_with_features()
        self.assertEqual(result.value, 17)
        self.assertIsNot(result, proxy.noise)
        self.assertEqual(proxy.calls, 1)
        self.assertEqual(torch.random_calls, 0)
        self.assertEqual(len(digest), 64)
        self.assertIs(head.get_action_with_features.__func__.__code__, original_code)
        self.assertEqual(original().value, -999)
        self.assertEqual(torch.random_calls, 1)
        self.assertEqual(Head().get_action_with_features().value, -999)

    def test_changed_noise_is_used_on_next_call(self):
        head = Head()
        proxy, _, _ = _install_noise_sampler(head, torch)
        for value in (2, 19):
            proxy.noise, proxy.calls = Tensor(value), 0
            self.assertEqual(head.get_action_with_features().value, value)
            self.assertEqual(proxy.calls, 1)

    def test_missing_noise_and_wrong_batch_fail(self):
        head = Head()
        proxy, _, _ = _install_noise_sampler(head, torch)
        with self.assertRaisesRegex(RuntimeError, "outside get_action"):
            head.get_action_with_features()
        proxy.noise = Tensor(1)
        with self.assertRaisesRegex(RuntimeError, "shape differs"):
            head.get_action_with_features(batch=2)

    def test_second_draw_and_unexpected_sampler_fail(self):
        head = Head()
        proxy, _, _ = _install_noise_sampler(head, torch)
        proxy.noise = Tensor(1)
        head.get_action_with_features()
        with self.assertRaisesRegex(RuntimeError, "more than one"):
            head.get_action_with_features()
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            _install_noise_sampler(UnsupportedHead(), torch)


class TensorRTNoiseTests(unittest.TestCase):
    def test_initial_capture_and_replay_use_current_noise(self):
        head = SimpleNamespace(init_actions=Tensor(-1))
        runner = SimpleNamespace(_graph_state=None)
        _refresh_trt_noise(head, runner, Tensor(3))
        self.assertEqual(head.init_actions.value, 3)
        graph_buffer = head.init_actions.clone()
        runner._graph_state = SimpleNamespace(initial_actions=graph_buffer)
        for value in (5, 11):
            _refresh_trt_noise(head, runner, Tensor(value))
            self.assertIs(runner._graph_state.initial_actions, graph_buffer)
            self.assertEqual(head.init_actions.value, value)
            self.assertEqual(graph_buffer.value, value)
        self.assertEqual(graph_buffer.copies, 2)

    def test_changed_graph_shape_fails(self):
        head = SimpleNamespace(init_actions=Tensor(-1))
        runner = SimpleNamespace(
            _graph_state=SimpleNamespace(initial_actions=Tensor(0, shape=(2, 40, 132)))
        )
        with self.assertRaisesRegex(RuntimeError, "graph noise shape differs"):
            _refresh_trt_noise(head, runner, Tensor(1))


if __name__ == "__main__":
    unittest.main()
