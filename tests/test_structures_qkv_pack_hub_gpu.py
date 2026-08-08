"""Opt-in qualification against the published QKV-pack Hub artifact."""

from __future__ import annotations

import os
import unittest

import torch

from flash_rt.structures.impls.qkv_pack.fp8_static import bind_qkv_pack


@unittest.skipUnless(
    os.environ.get("FLASHRT_RUN_HUB_TESTS") == "1"
    and torch.cuda.is_available(),
    "set FLASHRT_RUN_HUB_TESTS=1 on a target CUDA host",
)
class QkvPackHubQualification(unittest.TestCase):
    def test_bf16_joint_partial_capacity_compiles_and_captures(self):
        """Cover BF16 fused entry + joint output + logical M below capacity."""
        torch.manual_seed(123)
        input_width = 1536
        splits = (1536, 512, 512)
        capacity, logical_rows = 41, 7
        modules = [
            torch.nn.Linear(
                input_width,
                width,
                bias=True,
                device="cuda",
                dtype=torch.bfloat16,
            ).eval()
            for width in splits
        ]
        head = bind_qkv_pack(
            modules,
            torch.tensor([0.02], device="cuda"),
            rows=capacity,
            in_dtype="bf16_fused_quant",
        )[0]
        head.enable_joint(3)
        value = torch.randn(
            logical_rows,
            input_width,
            device="cuda",
            dtype=torch.bfloat16,
        )

        eager = head.joint(value).clone()
        host = torch.cat([module(value) for module in modules], dim=-1)
        cosine = torch.nn.functional.cosine_similarity(
            eager.float().flatten(),
            host.float().flatten(),
            dim=0,
        )

        compiled = torch.compile(
            lambda tensor: head.joint(tensor),
            fullgraph=True,
        )
        for _ in range(3):
            compiled(value)
        compiled_out = compiled(value).clone()

        static_value = value.clone()
        for _ in range(3):
            head.joint(static_value)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = head.joint(static_value)
        graph.replay()
        torch.cuda.synchronize()
        captured_out = graph_out.clone()

        self.assertEqual(eager.shape, (logical_rows, sum(splits)))
        self.assertGreater(float(cosine.detach()), 0.99)
        self.assertTrue(torch.equal(compiled_out, eager))
        self.assertTrue(torch.equal(captured_out, eager))
        self.assertEqual(head._frt_guard.fallbacks, 0)


if __name__ == "__main__":
    unittest.main()
