"""FP8-KV decode band for the whole-step loop.

Attention reads dominate a deep-context decode step, and the XQA entry
reads its KV as FP8 pages — half the bytes of the BF16 cache the host's
SDPA walks. The loop owns both ends of the attention data path (its
static cache writes the KV, its registered attention interface consumes
it), so the band slots in without touching the host layer: ``update``
keeps returning the BF16 tensors the prefill path needs and *also*
quantises the same rows into the paged FP8 store; the interface routes
short query batches (the decode row, a spec verify) through XQA and
everything else back to SDPA on the BF16 arm.

v1 is dual-store: the BF16 cache stays for prefill and fallback, so
the win is attention read bandwidth at depth, not memory — dropping
the BF16 store rides on an FP8 prefill attention, recorded follow-up.
The page shape is the kernel's v1 contract (24 query heads, 4 KV
heads, head dim 256); other profiles refuse cleanly at bind.

Graph discipline: seq_lens is a device buffer written in-graph, the
per-shape spec masks and the workspace are allocated at first use
(warmup) and never repointed, and the FP8 row writes are index_copy_
into stable pages — everything a captured replay requires.
"""

from __future__ import annotations

from functools import lru_cache

import torch

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "flashrt/fp8-kv-attention",
    "version": ">=1",
}

#: the kernel's v1 fixed profile
_QH, _KVH, _HD, _PAGE = 24, 4, 256, 128

#: query batches at or under this route through XQA; longer batches
#: (prompts) keep the BF16 SDPA arm
_XQA_MAX_Q = 32

_INTERFACE_NAME = "frt_fp8kv"


@lru_cache(maxsize=1)
def _kernel():
    from flash_rt.structures.impls import hub_kernel

    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


class Fp8KvBand:
    """Paged FP8 KV store plus the XQA call state for one loop."""

    def __init__(self, attn_layers, max_len, device):
        kern = _kernel()
        self._kern = kern
        self._max = int(max_len)
        pages = (self._max + _PAGE - 1) // _PAGE
        # the kernel's max_seq_len speaks in whole pages
        self._max_paged = pages * _PAGE
        f8 = torch.float8_e4m3fn
        self.k_pages = {i: torch.zeros(pages, _PAGE, _KVH, _HD,
                                       device=device, dtype=f8)
                        for i in attn_layers}
        self.v_pages = {i: torch.zeros_like(self.k_pages[i])
                        for i in attn_layers}
        self._seq = torch.zeros(1, 1, device=device, dtype=torch.int32)
        s_mb = 64   # the default 256MB scratch tips a full band
        # form over the card rim; 64MB serves the v1 shapes
        self._sem, self._scratch = kern.allocate_workspace(
            q_seq=_XQA_MAX_Q, device=device, scratch_mb=s_mb)
        self._masks: dict[int, torch.Tensor] = {}
        self._table = kern.default_page_table(pages, device=device)

    def write(self, layer_idx, k, v, pos):
        """Quantise post-rope rows into the pages at ``pos``."""
        kp = self.k_pages.get(layer_idx)
        if kp is None:
            return
        # eager ATen has no fp8 index_copy kernel (the compiled path
        # codegens around it); the byte view is the same write
        rows_k = k[0].transpose(0, 1).to(kp.dtype).view(torch.uint8)
        rows_v = v[0].transpose(0, 1).to(kp.dtype).view(torch.uint8)
        kp.view(torch.uint8).view(-1, _KVH, _HD).index_copy_(
            0, pos, rows_k)
        self.v_pages[layer_idx].view(torch.uint8).view(
            -1, _KVH, _HD).index_copy_(0, pos, rows_v)

    def reset(self):
        """Zero the pages: a fresh prompt must not see the previous
        stream's rows. The read path is page-granular in the kernel,
        so rows beyond seq_len are reachable garbage unless cleared —
        the repeat gate caught exactly that leak."""
        for kp in self.k_pages.values():
            kp.view(torch.uint8).zero_()
        for vp in self.v_pages.values():
            vp.view(torch.uint8).zero_()

    def prewarm(self, shapes):
        """Materialise the per-shape masks ahead of compiled use: a
        lazy build inside a compiled region flips a dynamo guard
        between warmup and capture, and recompiling while a stream is
        capturing is illegal."""
        dev = self._table.device
        for s in shapes:
            if s not in self._masks:
                self._masks[s] = self._kern.causal_spec_mask(
                    int(s), device=dev)

    def clear_rows(self, pos):
        """Zero the page rows at ``pos`` across every layer.

        The warmup steps before capture write real rows past the
        prompt; the read path is page-granular, so a later replay
        from the rolled-back position can still reach them. Clearing
        restores the exact page state a fresh call would see."""
        for kp in self.k_pages.values():
            kp.view(torch.uint8).view(-1, _KVH, _HD).index_fill_(
                0, pos, 0)
        for vp in self.v_pages.values():
            vp.view(torch.uint8).view(-1, _KVH, _HD).index_fill_(
                0, pos, 0)

    def set_len(self, total):
        """Total sequence length (device tensor or int), in-graph safe."""
        if torch.is_tensor(total):
            self._seq.view(-1).copy_(total.view(-1).to(torch.int32))
        else:
            self._seq.fill_(int(total))

    def attend(self, layer_idx, q):
        """``q`` is [1, qh, S, hd] post-rope; returns [1, S, qh, hd]."""
        s = q.shape[2]
        # the kernel wrapper self-builds a host-side mask when none is
        # passed - illegal inside a capture - so every shape's mask is
        # built once here (warmup) and replayed as a device constant
        mask = self._masks.get(s)
        if mask is None:
            mask = self._kern.causal_spec_mask(s, device=q.device)
            self._masks[s] = mask
        out = self._kern.xqa_bf16_fp8kv(
            q[0].transpose(0, 1).contiguous(),
            self.k_pages[layer_idx], self.v_pages[layer_idx],
            page_table=self._table, seq_lens=self._seq, mask=mask,
            semaphores=self._sem, scratch=self._scratch,
            max_seq_len=self._max_paged)
        return out.view(1, s, _QH, _HD)


def _interface(module, q, k, v, attention_mask, scaling=None, **kwargs):
    band = getattr(module, "_frt_fp8_band", None)
    # the band serves exactly the loop whose cache filled its pages;
    # the loop's static cache hands attention a full-window K (its
    # second-to-last dim is the window), a host-side DynamicCache hands
    # a growing one — that shape is the ownership signature, and host
    # forwards fall through to plain SDPA untouched
    if band is not None and q.shape[0] == 1 \
            and q.shape[2] <= _XQA_MAX_Q \
            and k.shape[2] == band._max:
        return band.attend(module.layer_idx, q), None
    if attention_mask is None and q.shape[2] > 1 \
            and k.shape[2] > q.shape[2]:
        # maskless prompt rows over the full static window: causal
        # rows 0..S-1 never see columns past S, and the square slice
        # keeps SDPA on its fused causal path — the rectangular case
        # falls to the math backend, which materialises an [S, window]
        # score matrix (gigabytes at deep windows)
        k = k[:, :, :q.shape[2]]
        v = v[:, :, :q.shape[2]]
    # the fall-through must be the host's own sdpa interface, bit for
    # bit — a lookalike SDPA call differs in repeat/contiguity details
    # and a detached model would stop reproducing its own baseline
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    return ALL_ATTENTION_FUNCTIONS["sdpa"](
        module, q, k, v, attention_mask, scaling=scaling, **kwargs)


def install(model, lm, cache, max_len):
    """Attach the band to a loop: pages on the cache, interface on the
    host's attention dispatch, handles on the attention modules.

    Refuses (returns ``None``) when the host profile is not the
    kernel's v1 contract — the loop simply keeps its BF16 attention.
    """
    cfg = getattr(model.config, "text_config", model.config)
    if (int(cfg.num_attention_heads), int(cfg.num_key_value_heads),
            int(getattr(cfg, "head_dim", 0))) != (_QH, _KVH, _HD):
        return None
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if _INTERFACE_NAME not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(_INTERFACE_NAME, _interface)
    attn_layers = [i for i, lyr in enumerate(lm.layers)
                   if hasattr(lyr, "self_attn")]
    dev = lm.embed_tokens.weight.device
    band = Fp8KvBand(attn_layers, max_len, dev)
    for i in attn_layers:
        lm.layers[i].self_attn._frt_fp8_band = band
        lm.layers[i].self_attn.config._attn_implementation = \
            _INTERFACE_NAME
    cache.frt_fp8_band = band
    return band
