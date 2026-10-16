#!/usr/bin/env python3
"""Prefix embedding with a multimodal merge.

Row r of the output is either an image feature row or a scaled token row, picked
per row by a host-computed index table:

    img_index[r] >= 0  ->  out[r, :] = img[img_index[r], :]
    otherwise          ->  out[r, :] = bf16(embed[token_ids[r], :] * scale)

The plain `embedding_lookup_bf16` covers neither the substitution nor the scale.

Usage:
  PYTHONPATH=. pytest -q tests/test_embedding_lookup_merge.py
"""
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

BF16 = torch.bfloat16


def _fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    return fvk


def _case(rows, dim, n_img, n_vocab, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(0, n_vocab, (rows,), device="cuda", dtype=torch.int64)
    idx = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
    if n_img:
        k = min(n_img, rows)
        where = torch.randperm(rows, device="cuda")[:k]
        idx[where] = torch.arange(k, device="cuda", dtype=torch.int32)
    table = (torch.randn(n_vocab, dim, device="cuda") * 0.5).to(BF16)
    img = (torch.randn(max(n_img, 1), dim, device="cuda")).to(BF16)
    return ids, idx, table, img


def _reference(ids, idx, table, img, rows, dim, scale):
    out = torch.empty(rows, dim, device="cuda", dtype=BF16)
    for r in range(rows):
        if int(idx[r]) >= 0:
            out[r] = img[int(idx[r])]
        else:
            out[r] = (table[int(ids[r])].float() * scale).to(BF16)
    return out


def _run(fvk, rows, dim, n_img, n_vocab, scale, seed=0):
    ids, idx, table, img = _case(rows, dim, n_img, n_vocab, seed)
    out = torch.zeros(rows, dim, device="cuda", dtype=BF16)
    fvk.embedding_lookup_merge_bf16(ids.data_ptr(), idx.data_ptr(), table.data_ptr(),
                                    img.data_ptr(), out.data_ptr(), rows, dim, scale,
                                    torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return out, _reference(ids, idx, table, img, rows, dim, scale)


class TestEmbeddingLookupMerge:
    @pytest.mark.parametrize("rows,dim,n_img,n_vocab", [
        (568, 2560, 256, 262144),   # a prefix with 256 image tokens among text
        (40, 2048, 0, 1000),        # text only: every row takes the token path
        (1, 2560, 1, 500),          # a single image row
        (300, 1152, 300, 700),      # every row is an image row
    ])
    def test_matches_the_reference(self, rows, dim, n_img, n_vocab):
        fvk = _fvk()
        out, ref = _run(fvk, rows, dim, n_img, n_vocab, 44.0)
        assert torch.equal(out, ref), \
            f"{int((out != ref).sum().item())} of {ref.numel()} elements differ"

    def test_the_scale_is_applied(self):
        """A kernel that skipped the scale would still look plausible."""
        fvk = _fvk()
        a, _ = _run(fvk, 32, 512, 0, 100, 1.0)
        b, _ = _run(fvk, 32, 512, 0, 100, 44.0)
        assert not torch.equal(a, b)

    def test_the_scale_does_not_touch_image_rows(self):
        """The image path copies the feature row; only the token path is scaled."""
        fvk = _fvk()
        ids, idx, table, img = _case(16, 256, 16, 50)
        scale = 44.0
        out = torch.zeros(16, 256, device="cuda", dtype=BF16)
        fvk.embedding_lookup_merge_bf16(ids.data_ptr(), idx.data_ptr(), table.data_ptr(),
                                        img.data_ptr(), out.data_ptr(), 16, 256, scale,
                                        torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        # every row is an image row, so each row must be that row's feature row
        # copied verbatim -- idx is a permutation, so row r does not take img[r]
        for r in range(16):
            assert int(idx[r]) >= 0
            assert torch.equal(out[r], img[int(idx[r])]), f"row {r} was modified"

    def test_a_null_pointer_fails_on_the_host(self):
        """A null table is a dereference, not a no-op; it must not be launched.

        Asserting on the message matters here: an unguarded launch takes an
        illegal memory access, which pytest.raises would also "pass" while
        leaving the CUDA context poisoned for every test that follows.
        """
        fvk = _fvk()
        ids, _, table, img = _case(4, 128, 1, 50)
        out = torch.zeros(4, 128, device="cuda", dtype=BF16)
        with pytest.raises(RuntimeError, match="null pointer"):
            fvk.embedding_lookup_merge_bf16(ids.data_ptr(), 0, table.data_ptr(),
                                            img.data_ptr(), out.data_ptr(), 4, 128, 1.0,
                                            torch.cuda.current_stream().cuda_stream)
        # the context is still usable
        torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
