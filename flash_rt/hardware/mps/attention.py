"""Apple MPS SDPA provider with explicit grouped-query head expansion."""

import torch
import torch.nn.functional as F

from flash_rt.hardware.torch_reference import TorchAttentionBackend


class MpsAttentionBackend(TorchAttentionBackend):
    def gqa(self, q, k, v, *, valid_prefix, prefix_capacity, out=None):
        # SDPA's enable_gqa route is not portable to MPS. Expanding the
        # single KV head preserves Gemma's grouped-query attention semantics.
        repeats = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        mask = None
        if k.shape[0] not in (valid_prefix, valid_prefix + q.shape[0]):
            mask = torch.zeros(q.shape[0], k.shape[0], dtype=torch.bool, device=q.device)
            mask[:, :valid_prefix] = True
            mask[:, prefix_capacity:] = True
        value = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0), attn_mask=mask,
        )
        value = value.squeeze(0).transpose(0, 1).reshape(q.shape[0], -1)
        if out is None:
            return value
        out.copy_(value)
        return out
