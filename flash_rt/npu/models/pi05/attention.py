"""Setup-owned shared pages for the action decoder's bidirectional attention."""
import os
from dataclasses import dataclass

import torch

from .pipeline import DEC_HD, DEC_NH, DEC_NKV

# The attention kernel's key tile. One L0B slot has to be exactly half the
# buffer for a K split to accumulate, which at BF16 and HD 256 fixes it at 64.
_KEY_TILE = 64


def transposed_attention_enabled() -> bool:
    """The transposed cache is the default; the environment turns it off so a
    control arm can run the vendor operator through the same harness."""
    return os.environ.get("FLASHRT_NPU_TRANSPOSED_ATTENTION", "1") != "0"


@dataclass(frozen=True)
class PagedDecoderAttention:
    table: torch.Tensor
    total: int
    queries: int
    block_size: int
    capacity: int

    @classmethod
    def create(cls, total, queries, device="npu", block_size=128):
        if total <= 0 or queries <= 0 or block_size <= 0 or block_size % 16:
            raise ValueError("invalid decoder page geometry")
        pages = (total + block_size - 1) // block_size
        # Each action query sees the same current-frame prefix and suffix.
        table = torch.arange(pages, device=device, dtype=torch.int32).expand(
            queries, -1).contiguous()
        # CANN requires storage capacity for the sum of per-batch page counts,
        # even though every query's table points to the same first pages.
        return cls(table, total, queries, block_size, queries * pages * block_size)

    def __call__(self, q, k, v):
        import torch_npu
        # Every action query attends to the same bidirectional KV rows. Pair
        # queries as additional heads without moving data or changing softmax's
        # reduction domain. This geometry is tuned for the ten-action replay;
        # other chunk sizes retain their original scheduling.
        batch = self.queries // 2 if self.queries == 10 else self.queries
        heads = self.queries * DEC_NH // batch
        return torch_npu.npu_incre_flash_attention(
            q.reshape(batch, heads, 1, DEC_HD),
            k.reshape(-1, DEC_NKV, self.block_size, DEC_HD),
            v.reshape(-1, DEC_NKV, self.block_size, DEC_HD),
            num_heads=heads, num_key_value_heads=DEC_NKV,
            input_layout="BNSD", scale_value=DEC_HD ** -0.5,
            block_table=self.table[:batch], block_size=self.block_size,
            actual_seq_lengths=[self.total] * batch,
            inner_precise=0).reshape(self.queries, DEC_NH * DEC_HD)


@dataclass(frozen=True)
class TransposedDecoderAttention:
    """Decode attention over a cache that keeps the values transposed.

    A raw ``Mmad`` B operand wants its GM source in ``(N, K)`` form -- measured,
    not assumed. For ``S = Q * K^T`` the cache already holds K that way; for
    ``O = P * V`` it means V as ``(HD, KV)``, which the row-major cache does
    not hold and which no transposing load was made to produce. Keeping the
    transpose in the cache instead makes both attention GEMMs the same code
    path as the decoder GEMM, with nothing to convert on either operand.

    The transpose is stored in fractal NZ, so a sixteen-position block is 8 KB
    of contiguous bytes and the value tile for a key block is a plain copy.
    That is also why the action suffix leads this cache and the encoder prefix
    starts at ``column``: a denoise step rewrites the suffix on every layer,
    and writing part of a fractal block while other cores hold the rest of it
    is not safe. The prefix is not a multiple of sixteen rows, so a suffix
    placed after it would straddle two blocks; placed first it takes block zero
    whole. Attention does not care in what order the keys arrive.

    The gap between the suffix and ``column``, and the tail past ``end``, are
    padding. Both cache halves are zero there, so those columns score zero,
    take a finite share of the softmax and multiply a zero value column. The
    row max and the row sum are taken over the two live ranges only.
    """

    prefix: int
    chunk: int
    column: int
    end: int
    kvp: int
    mq: int
    scores: torch.Tensor
    probs: torch.Tensor
    ctx: torch.Tensor
    out: torch.Tensor
    launch: object
    transposed: bool = True

    @classmethod
    def create(cls, prefix_len, chunk, device="npu"):
        if prefix_len <= 0 or chunk <= 0 or chunk > 16:
            raise ValueError("invalid decode attention geometry")
        column = (chunk + 15) // 16 * 16
        end = column + prefix_len
        kvp = (end + _KEY_TILE - 1) // _KEY_TILE * _KEY_TILE
        mq = chunk * DEC_NH
        if mq % 16:
            raise ValueError("the query rows must fill whole 16-row fractal blocks")
        from flash_rt.npu.core.decode_attention import DecodeAttentionLibrary
        # Scratch is setup-owned: a replay only launches.
        return cls(prefix_len, chunk, column, end, kvp, mq,
                   torch.zeros(mq, kvp, dtype=torch.float32, device=device),
                   torch.zeros(mq, kvp, dtype=torch.bfloat16, device=device),
                   torch.zeros(mq, DEC_HD, dtype=torch.float32, device=device),
                   torch.zeros(mq, DEC_HD, dtype=torch.bfloat16, device=device),
                   DecodeAttentionLibrary().launch)

    def __call__(self, q, k, v):
        # The stream handle has to be read inside the capture: one taken before
        # it belongs to a stream the replay does not run on.
        code = self.launch(
            torch.npu.current_stream(q.device).npu_stream,
            q.data_ptr(), k.data_ptr(), v.data_ptr(), self.out.data_ptr(),
            self.scores.data_ptr(), self.probs.data_ptr(), self.ctx.data_ptr(),
            self.mq, self.chunk, self.column, self.end, self.kvp,
            float(DEC_HD ** -0.5))
        if code:
            raise RuntimeError(f"native decode attention rejected arguments: {code}")
        return self.out.reshape(self.chunk, DEC_NH * DEC_HD)
