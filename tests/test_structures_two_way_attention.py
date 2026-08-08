from __future__ import annotations

import torch

from flash_rt.structures.adapters.factored_two_way_attention import (
    FactoredTwoWayAttentionAdapter,
)
from flash_rt.structures.impls.attention_core import two_way_fa2


class _FakeFa2:
    @staticmethod
    def allocate_outputs(q):
        return torch.empty_like(q), torch.empty(
            q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32)

    @staticmethod
    def allocate_workspace(q, k):
        del q, k
        return None

    @staticmethod
    def forward_static(
        q,
        k,
        v,
        *,
        out,
        softmax_lse,
        workspace,
        softmax_scale,
        causal,
    ):
        del softmax_lse, workspace
        groups = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(groups, dim=2)
        v = v.repeat_interleave(groups, dim=2)
        scores = torch.einsum(
            "bthd,bshd->bhts", q.float(), k.float()) * softmax_scale
        if causal:
            mask = torch.ones(
                q.shape[1], k.shape[1], dtype=torch.bool).triu(1)
            scores.masked_fill_(mask, -torch.inf)
        probs = scores.softmax(dim=-1)
        result = torch.einsum(
            "bhts,bshd->bthd", probs, v.float()).to(q.dtype)
        out.copy_(result)
        return out


def _pack(causal, full):
    return {
        "causal_seq": causal,
        "full_only_seq": full,
        "sample_offsets": torch.tensor([0, 5], dtype=torch.int32),
        "_causal_indices": torch.tensor([0, 2, 4], dtype=torch.int32),
        "_full_indices": torch.tensor([1, 3], dtype=torch.int32),
        "_causal_seq_offsets": torch.tensor([0, 3], dtype=torch.int32),
        "_full_only_seq_offsets": torch.tensor([0, 2], dtype=torch.int32),
        "_num_causal_tokens": 3,
        "_num_full_tokens": 2,
        "max_sample_len": 5,
        "max_causal_len": 3,
        "max_full_len": 2,
        "max_num_tokens": 5,
        "is_sharded": False,
    }


def test_two_way_attention_matches_causal_and_joint_gqa(monkeypatch):
    monkeypatch.setattr(two_way_fa2, "hub_kernel", lambda *args: _FakeFa2())
    torch.manual_seed(7)
    q = _pack(
        torch.randn(3, 4, 8, dtype=torch.bfloat16),
        torch.randn(2, 4, 8, dtype=torch.bfloat16),
    )
    k = _pack(
        torch.randn(3, 2, 8, dtype=torch.bfloat16),
        torch.randn(2, 2, 8, dtype=torch.bfloat16),
    )
    v = _pack(
        torch.randn(3, 2, 8, dtype=torch.bfloat16),
        torch.randn(2, 2, 8, dtype=torch.bfloat16),
    )
    bound = two_way_fa2.bind_two_way_attention(
        {"query": q, "key": k, "value": v})
    got = bound(q, k, v)

    fake = _FakeFa2()
    causal_out = torch.empty(1, 3, 4, 8, dtype=torch.bfloat16)
    fake.forward_static(
        q["causal_seq"].unsqueeze(0),
        k["causal_seq"].unsqueeze(0),
        v["causal_seq"].unsqueeze(0),
        out=causal_out,
        softmax_lse=torch.empty(1, 4, 3),
        workspace=None,
        softmax_scale=8 ** -0.5,
        causal=True,
    )
    joint_k = torch.empty(5, 2, 8, dtype=torch.bfloat16)
    joint_v = torch.empty_like(joint_k)
    joint_k[q["_causal_indices"].long()] = k["causal_seq"]
    joint_k[q["_full_indices"].long()] = k["full_only_seq"]
    joint_v[q["_causal_indices"].long()] = v["causal_seq"]
    joint_v[q["_full_indices"].long()] = v["full_only_seq"]
    full_out = torch.empty(1, 2, 4, 8, dtype=torch.bfloat16)
    fake.forward_static(
        q["full_only_seq"].unsqueeze(0),
        joint_k.unsqueeze(0),
        joint_v.unsqueeze(0),
        out=full_out,
        softmax_lse=torch.empty(1, 4, 2),
        workspace=None,
        softmax_scale=8 ** -0.5,
        causal=False,
    )
    torch.testing.assert_close(
        got["causal_seq"], causal_out.squeeze(0).flatten(-2, -1))
    torch.testing.assert_close(
        got["full_only_seq"], full_out.squeeze(0).flatten(-2, -1))
    assert bound._frt_guard.calls == 1


def test_two_way_adapter_routes_and_restores(monkeypatch):
    class HostProcessor:
        def __call__(self, query, key, value):
            del key, value
            return query

    class HostAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dispatch_attention_fn = HostProcessor()

        def forward(self, query, key, value):
            return self.dispatch_attention_fn(query, key, value)

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = HostAttention()
            self.pack = {
                "causal_seq": torch.tensor([[[2.0]]]),
                "full_only_seq": torch.tensor([[[4.0]]]),
            }

        def forward(self):
            return self.attention(self.pack, self.pack, self.pack)

    class Core(torch.nn.Module):
        def forward(self, query, key, value):
            del key, value
            return {
                **query,
                "causal_seq": query["causal_seq"] + 3,
                "full_only_seq": query["full_only_seq"] + 3,
            }

    # resolve module and class at run time from sys.modules: a
    # neighbouring test (test_install_smoke) purges and re-imports the
    # flash_rt tree between collection and this test, so the names bound
    # at this file's import can be a different module object than the
    # one a dotted-path monkeypatch would patch
    import importlib

    mod = importlib.import_module(
        "flash_rt.structures.adapters.factored_two_way_attention")
    monkeypatch.setattr(mod, "bind_two_way_attention",
                        lambda capture: Core())
    root = Root()
    original = root.attention.dispatch_attention_fn
    result = mod.FactoredTwoWayAttentionAdapter()(root, root.forward)
    assert result is not None
    _, _, extras = result
    assert root.forward()["causal_seq"].item() == 5
    extras["toggle"][1]()
    assert root.forward()["causal_seq"].item() == 2
    extras["toggle"][0]()
    assert root.forward()["causal_seq"].item() == 5
    extras["revert"][0]()
    assert root.attention.dispatch_attention_fn is original
