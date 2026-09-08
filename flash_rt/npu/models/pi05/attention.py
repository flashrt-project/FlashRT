"""Setup-owned shared pages for the action decoder's bidirectional attention."""
from dataclasses import dataclass

import torch

from .pipeline import DEC_HD, DEC_NH, DEC_NKV


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
