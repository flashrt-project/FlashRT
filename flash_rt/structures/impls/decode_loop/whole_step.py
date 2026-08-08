"""decode_loop — the whole-loop serving form for cached LLM decode.

Seam swaps accelerate the pieces; this family owns the loop. The host's
``generate`` walks Python glue between every token, repoints its KV
cache with ``torch.cat``, and rebuilds masks per step — none of which
can replay inside a CUDA graph. The whole-step form drives the decoder
layers directly with static buffers:

- a duck-typed static hybrid cache (preallocated KV written by
  ``index_copy_`` at an in-graph position buffer; recurrent/conv slots
  as the layers expect them) — no host cache class is imported, only
  the surface the layers actually touch is implemented;
- the causal row itself masks the padded tail of the static KV, so the
  decode mask is one ``index_select`` on the position — no mask
  bookkeeping;
- argmax and the position increment run in-graph, and the step is
  optionally wrapped in ``torch.compile`` so the elementwise regions
  between custom kernels fuse before capture.

The loop is greedy and exact: its tokens are gated against the host's
own generation by the caller's probe. Structure swaps (fused layers,
precision bands) compose underneath — build the loop after
``auto_swaps`` and it captures whatever is attached.
"""

from __future__ import annotations

import torch


class _StaticHybridCache:
    """The surface a hybrid (attention + gated-delta) stack touches.

    ``update`` serves attention layers from preallocated buffers;
    ``conv_states``/``recurrent_states`` are plain per-layer slots the
    gated-delta layers read and write; ``has_previous_state`` mirrors
    the host convention (a filled conv slot means decode).
    """

    def __init__(self, n_layers, attn_layers, kv_heads, head_dim,
                 max_len, device, dtype=torch.bfloat16):
        self.conv_states = [None] * n_layers
        self.recurrent_states = [None] * n_layers
        self.key_cache = [None] * n_layers
        self.value_cache = [None] * n_layers
        self._max = int(max_len)
        self._cp = None
        # true prompt-progress counter for HOST-side forwards: host glue
        # branches on "is there history" (rope deltas, prefill vs
        # continue); the loop's own fwd never consults it
        self._seen = 0
        for i in attn_layers:
            self.key_cache[i] = torch.zeros(
                1, kv_heads, self._max, head_dim, device=device,
                dtype=dtype)
            self.value_cache[i] = torch.zeros_like(self.key_cache[i])

    def update(self, k, v, layer_idx, cache_kwargs=None):
        self.key_cache[layer_idx].index_copy_(2, self._cp, k)
        self.value_cache[layer_idx].index_copy_(2, self._cp, v)
        band = getattr(self, "frt_fp8_band", None)
        if band is not None:
            # dual-store: the same post-rope rows land in the FP8 pages
            # the XQA read path consumes; BF16 stays the prefill arm
            band.write(layer_idx, k, v, self._cp)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self._seen

    def get_mask_sizes(self, query_length, layer_idx):
        return self._max, 0

    @property
    def has_previous_state(self):
        return any(s is not None for s in self.conv_states)


def _find_stack(model):
    """Locate the decoder stack by the slots it must carry, not names."""
    for mod in (getattr(model, "model", model),):
        for cand in (getattr(mod, "language_model", None), mod):
            if cand is None:
                continue
            if (hasattr(cand, "layers") and hasattr(cand, "embed_tokens")
                    and hasattr(cand, "norm")
                    and hasattr(cand, "rotary_emb")):
                return cand
    raise ValueError(
        "refused: no decoder stack with (layers, embed_tokens, norm, "
        "rotary_emb) slots found on this host")


class WholeStepDecodeLoop:
    """Compiled, graph-captured greedy decode over the attached model."""

    def __init__(self, model, *, max_len, compile_step=True,
                 compile_prefill=True, kv_band=None):
        lm = _find_stack(model)
        self._model = model
        self._lm = lm
        self._compile_prefill = bool(compile_prefill)
        head = getattr(model, "lm_head", None)
        if head is None:
            raise ValueError("refused: host carries no lm_head")
        self._layers = lm.layers
        self._rotary = lm.rotary_emb
        self._norm = lm.norm
        self._embed = lm.embed_tokens
        self._head = head
        self._full = [i for i, lyr in enumerate(self._layers)
                      if hasattr(lyr, "self_attn")]
        cfg = getattr(model.config, "text_config", model.config)
        kvh = int(cfg.num_key_value_heads)
        hd = int(getattr(cfg, "head_dim",
                         cfg.hidden_size // cfg.num_attention_heads))
        dev = head.weight.device if hasattr(head, "weight") else "cuda"
        # the static window rounds up to a 16-byte row: the fused
        # attention kernels vector-read the mask row in aligned chunks,
        # and an unaligned tail overreads into neighbouring allocations
        # — outputs then drift with allocator contents (the 2K repeat
        # gate caught it; compute-sanitizer named the aligned kernel).
        # Rounding the window keeps mask and KV rows contiguous and
        # aligned; the padded tail rows stay -inf-masked and zero.
        self._max = -(-int(max_len) // 8) * 8
        self.cache = _StaticHybridCache(
            len(self._layers), self._full, kvh, hd, self._max, dev)
        # the loop routes attention through its own interface even
        # without a band: the maskless prompt pass needs the square
        # causal slice to stay off SDPA's materialising math path
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from .fp8_kv import _INTERFACE_NAME, _interface
        if _INTERFACE_NAME not in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS.register(_INTERFACE_NAME, _interface)
        for i in self._full:
            self._layers[i].self_attn.config._attn_implementation = \
                _INTERFACE_NAME
        self._kv_band = None
        if kv_band == "fp8":
            from .fp8_kv import install

            self._kv_band = install(model, lm, self.cache, self._max)
            if self._kv_band is None:
                raise ValueError(
                    "refused: fp8 kv band serves the kernel's v1 head "
                    "profile (24/4/256); this host keeps BF16 KV")
        elif kv_band is not None:
            raise ValueError(f"refused: unknown kv band {kv_band!r}")
        # no quadratic causal table: the decode mask is one static row
        # rebuilt in-graph from the position buffer, the prompt pass
        # rides SDPA's own causal flag (top-left alignment matches the
        # window layout), and offset multi-row passes build their rows
        # on demand — at deep windows the [max, max] table and the
        # prefill's materialised [S, max] mask are gigabytes
        self._ar = torch.arange(self._max, device=dev)
        self._mrow = torch.zeros(1, 1, 1, self._max, device=dev,
                                 dtype=torch.bfloat16)
        self._cur = torch.empty(1, 1, dtype=torch.long, device=dev)
        self._pos = torch.empty(1, dtype=torch.long, device=dev)
        self._rope_delta = torch.zeros(1, dtype=torch.long, device=dev)
        self._use_compile = bool(compile_step)
        if compile_step:
            # the layer loop specialises per layer index; the default
            # recompile budget is smaller than a deep stack
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit,
                4 * len(self._layers))
            self._step = torch.compile(self._fwd, dynamic=False)
        else:
            self._step = self._fwd
        self._graph = None
        self._aot_pf = None

    def _fwd(self, tok_ids, pos_t):
        h = self._embed(tok_ids)
        # KV slot index and rotary position are different things on
        # multimodal hosts: the host's rope delta shifts the latter
        pe = self._rotary(h, (pos_t + self._rope_delta).view(1, -1))
        self.cache._cp = pos_t
        if self._kv_band is not None:
            self._kv_band.set_len(pos_t[-1:] + 1)
        if tok_ids.shape[1] == 1:
            self._mrow.zero_()
            self._mrow.masked_fill_(
                (self._ar > pos_t[-1]).view(1, 1, 1, -1), float("-inf"))
            m4 = self._mrow
        else:
            # prompt rows start at zero: SDPA's causal flag is this
            # exact mask, with nothing materialised
            m4 = None
        for i, lyr in enumerate(self._layers):
            h = lyr(h, position_embeddings=pe,
                    attention_mask=(m4 if i in self._full_set else None),
                    past_key_values=self.cache, use_cache=True,
                    cache_position=pos_t)
        return self._head(self._norm(h)[:, -1:])

    @property
    def _full_set(self):
        s = getattr(self, "_full_cache", None)
        if s is None:
            s = set(self._full)
            self._full_cache = s
        return s

    def _prefill_callable(self):
        """The prompt pass gets its own compile, pinned deterministic.

        The fast step's inductor artifacts use non-deterministic
        reductions that are harmless inside a captured replay but flip
        long-prompt logits between runs — the 2K-context receipts
        caught exactly that. Prefill therefore compiles separately with
        the backend's deterministic mode, keeping the eager arm's
        repeatability at compiled speed. Shapes specialise per prompt
        length, same as the step.
        """
        if not self._compile_prefill:
            return self._fwd
        if getattr(self, "_pf", None) is None:
            self._pf = torch.compile(self._fwd, dynamic=False,
                                     options={"deterministic": True})
        return self._pf

    def _gstep(self):
        logits = self._step(self._cur, self._pos)
        # hosts sample from FP32 logits; argmax there too, or BF16
        # ties break differently and free runs diverge
        self._cur.copy_(logits.float().argmax(-1))
        self._pos.add_(1)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens):
        """Greedy generation; the first call warms, compiles, captures."""
        L = int(input_ids.shape[1])
        if L + max_new_tokens > self._max:
            raise ValueError(
                f"refused: {L}+{max_new_tokens} exceeds the static "
                f"window {self._max}")
        if self._graph is None and not getattr(self, "_warming", False):
            # the loop's true first pass is a throwaway: it compiles,
            # captures, and settles the library workspaces whose
            # content the earliest replays see differently from every
            # later call. Returned generations all come from the
            # settled regime, which repeats bitwise
            self._warming = True
            try:
                self.generate(input_ids, max_new_tokens)
            finally:
                self._warming = False
        self.cache.frt_continue = False
        self._rope_delta.zero_()
        if self._kv_band is not None:
            self._kv_band.reset()
        # canonical window: rows past the prompt zero before any
        # produced token. Masked tail rows are not numerically inert
        # in every attention backend, and the repeat gate compares
        # calls whose tails would otherwise carry different residue
        for i in self._full:
            self.cache.key_cache[i][:, :, L:].zero_()
            self.cache.value_cache[i][:, :, L:].zero_()
        pf = self._aot_pf or self._prefill_callable()
        logits = pf(input_ids,
                    torch.arange(L, device=input_ids.device))
        self._cur.copy_(logits.float().argmax(-1))
        self._pos.fill_(L)
        toks = [self._cur.clone()]
        self._decode_tail(max_new_tokens, toks)
        return torch.cat([input_ids] + toks, dim=1)

    def _decode_tail(self, max_new_tokens, toks):
        warm = min(3, max_new_tokens - 1)
        if self._graph is None:
            # first capture takes the uniform path: an eager warmup
            # step and a replayed step need not be bit-identical, so
            # no produced token may come off the eager arm — the first
            # call would disagree with every steady call at the first
            # post-warmup token. Warm without producing tokens,
            # capture, roll the state back, and produce every token
            # from the replay. KV rows (BF16 and FP8 pages alike)
            # rewrite the same slots on replay, so only the
            # gated-delta slots and cur/pos need snapshots; those
            # stage to host — this runs once, and the device is
            # already packed to the rim.
            snaps = [(i, self.cache.conv_states[i].to("cpu"),
                      self.cache.recurrent_states[i].to("cpu"))
                     for i in self._gdn_slots()]
            cur0, pos0 = self._cur.clone(), self._pos.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warm):
                    self._gstep()
            torch.cuda.current_stream().wait_stream(side)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._gstep()
            # prime the pool: capture records without executing, so
            # the first replay ever reads workspace buffers no replay
            # has written; its output is discarded and the state
            # rolled back, so every produced token comes from a replay
            # whose workspace a previous replay populated
            self._graph.replay()
            for i, cs, rs in snaps:
                self.cache.conv_states[i].copy_(cs)
                self.cache.recurrent_states[i].copy_(rs)
            self._cur.copy_(cur0)
            self._pos.copy_(pos0)
            # the rollback extends to the written rows: warmup steps
            # and the priming replay committed KV past the prompt, and
            # the canonical window holds that region zero before
            # produced tokens
            p0 = int(pos0[0])
            span = warm + 1
            for i in self._full:
                self.cache.key_cache[i][:, :, p0:p0 + span].zero_()
                self.cache.value_cache[i][:, :, p0:p0 + span].zero_()
            if self._kv_band is not None:
                self._kv_band.clear_rows(
                    torch.arange(span, device=pos0.device) + pos0)
            for _ in range(max_new_tokens - 1):
                self._graph.replay()
                toks.append(self._cur.clone())
        else:
            # steady calls run every decode step as a replay
            for _ in range(max_new_tokens - 1):
                self._graph.replay()
                toks.append(self._cur.clone())

    @torch.no_grad()
    def aot_prefill(self, example_ids, package_path,
                    inductor_configs=None):
        """Package the prompt pass whole and route ``generate`` through it.

        S-specialised like the compiled prefill. The package is built
        weights-external: the graph binary carries no constants, and at
        load it borrows the live parameters and the loop's own KV rows
        in place — a second copy of the weights never exists on the
        card, and the packaged KV writes are writes into the cache the
        captured decode step reads. The same-token gate downstream
        checks exactly that hand-off.
        """
        if self._kv_band is not None:
            raise ValueError(
                "refused: aot_prefill serves the BF16-KV loop; the FP8 "
                "band's prefill dual-store is not packaged yet")
        from ...aot import aot_load, aot_package_external

        dev = example_ids.device
        vehicle = _PrefillVehicle(self)
        pos = torch.arange(int(example_ids.shape[1]), device=dev)
        self.cache.frt_continue = False
        self._rope_delta.zero_()
        try:
            path, weights = aot_package_external(
                vehicle, args=(example_ids, pos),
                package_path=package_path,
                inductor_configs=inductor_configs)
        finally:
            # tracing rebinds the cache lists through the vehicle's
            # buffer attributes and leaves fake proxies behind; put the
            # real tensors back whatever happened
            for i in self._full:
                self.cache.key_cache[i] = getattr(vehicle, f"frt_k{i}")
                self.cache.value_cache[i] = getattr(vehicle, f"frt_v{i}")
        self._aot_pf = aot_load(path, weights=weights)
        return path

    @torch.no_grad()
    def generate_from(self, inputs, max_new_tokens):
        """Host-side prefill, loop-side decode.

        The host forward runs the whole multimodal front (vision tower,
        embed merge, mrope) and writes its KV into the loop's cache;
        the captured step takes over from the first generated token.
        ``inputs`` is the processor's dict; batch of one, no padding.
        """
        inputs = {k: v for k, v in inputs.items()
                  if k != "attention_mask"}
        ids = inputs["input_ids"]
        L = int(ids.shape[1])
        if L + max_new_tokens > self._max:
            raise ValueError(
                f"refused: {L}+{max_new_tokens} exceeds the static "
                f"window {self._max}")
        self.cache.frt_continue = False
        self.cache._seen = 0        # host must take its prefill branch
        for i in self._full:
            self.cache.key_cache[i][:, :, L:].zero_()
            self.cache.value_cache[i][:, :, L:].zero_()
        cp = torch.arange(L, device=ids.device)
        self.cache._cp = cp
        out = self._model(**inputs, past_key_values=self.cache,
                          use_cache=True, cache_position=cp)
        self.cache._seen = L
        delta = getattr(
            getattr(self._model, "model", self._model),
            "rope_deltas", None)
        self._rope_delta.fill_(
            int(delta.reshape(-1)[0]) if torch.is_tensor(delta) else 0)
        self._cur.copy_(out.logits[:, -1:].float().argmax(-1))
        self._pos.fill_(L)
        toks = [self._cur.clone()]
        self._decode_tail(max_new_tokens, toks)
        return torch.cat([ids] + toks, dim=1)


    @torch.no_grad()
    def enable_mtp(self, ckpt_dir=None, head=None,
                   projection_format=None, default_k=6,
                   verify_capture=True):
        """Attach the checkpoint's draft head.

        The draft's precision axes are explicit and answer to acceptance
        length alone (the verify pass anchors the output either way):
        ``projection_format`` picks the draft expert-bank arm —
        ``"bf16"`` (default, conservative) or ``"nvfp4_dynamic"``
        (measured AL-equal, smaller) — and the draft always carries a
        private W8 view of the shared head, because the model's own
        head must stay on the step/verify numeric family. Pass a
        prebuilt ``head`` to load it early, while the device still has
        assembly headroom; a prebuilt head already carries its formats
        and a conflicting ``projection_format`` here is refused, not
        silently ignored.

        ``verify_capture=False`` keeps the M=K+1 verify and rewrite
        passes eager — the diagnostic arm; the captured passes are the
        production form."""
        from .mtp_speculative import MtpDraftHead, check_draft_formats

        if projection_format is not None:
            check_draft_formats("w8a16_static", projection_format)
            if head is not None and getattr(head, "formats", {}).get(
                    "experts") not in (None, projection_format):
                raise ValueError(
                    f"refused: prebuilt draft head carries experts "
                    f"format {head.formats['experts']!r}, caller asked "
                    f"for {projection_format!r}")
        slot = len(self._layers)
        ref = self.cache.key_cache[self._full[0]]
        self.cache.key_cache.append(torch.zeros_like(ref))
        self.cache.value_cache.append(torch.zeros_like(ref))
        self.cache.conv_states.append(None)
        self.cache.recurrent_states.append(None)
        if head is None:
            head = MtpDraftHead(
                self._model, self._lm, ckpt_dir, slot,
                experts_format=(projection_format or "bf16"))
        if head.slot != slot:
            raise ValueError(
                f"refused: draft head was built for slot {head.slot}, "
                f"this loop's slot is {slot}")
        self._mtp = head
        self._mtp_formats = dict(getattr(head, "formats", {}))
        self._default_k = int(default_k)
        self._verify_capture = bool(verify_capture)
        return self._mtp

    def _fwd_full(self, tok_ids, pos_t):
        h = self._embed(tok_ids)
        pe = self._rotary(h, pos_t.view(1, -1))
        self.cache._cp = pos_t
        if self._kv_band is not None:
            self._kv_band.set_len(pos_t[-1:] + 1)
        m4 = self._row(pos_t)
        for i, lyr in enumerate(self._layers):
            h = lyr(h, position_embeddings=pe,
                    attention_mask=(m4 if i in self._full_set else None),
                    past_key_values=self.cache, use_cache=True,
                    cache_position=pos_t)
        hn = self._norm(h)
        return self._head(hn), hn

    def _row(self, pos_t):
        # offset multi-row mask (spec verify/rewrite): built on demand
        # from the positions, a few rows at a time
        return torch.zeros(
            1, 1, pos_t.shape[0], self._max, device=pos_t.device,
            dtype=torch.bfloat16).masked_fill_(
                (self._ar.view(1, -1) > pos_t.view(-1, 1)).view(
                    1, 1, -1, self._max), float("-inf"))

    def _gdn_slots(self):
        return [i for i, sl in enumerate(self.cache.conv_states)
                if torch.is_tensor(sl)]

    def _ensure_spec_graphs(self, K, dev, snaps, h_last, tok, pos):
        """Capture the K-step draft chain and the M=K+1 verify pass.

        The warmup executions commit real state; the caller's snapshot
        is restored before capture so the first replay starts clean.
        """
        if self._kv_band is not None:
            # every spec shape's mask exists before the compiled passes
            # trace: a lazy dict fill flips a guard between warmup and
            # capture, and recompiling mid-capture is illegal
            self._kv_band.prewarm(range(1, K + 2))
        self._dh_buf = torch.empty(1, 1, self._embed.weight.shape[1],
                                   device=dev, dtype=torch.bfloat16)
        self._dtok_buf = torch.empty(1, 1, dtype=torch.long, device=dev)
        self._dpos_buf = torch.empty(1, dtype=torch.long, device=dev)
        self._dtoks_out = torch.empty(K, dtype=torch.long, device=dev)
        self._vseq_buf = torch.empty(1, K + 1, dtype=torch.long,
                                     device=dev)
        self._vpos_buf = torch.empty(K + 1, dtype=torch.long, device=dev)
        # warmup must run on live values — an uninitialised token
        # buffer is an out-of-bounds embedding lookup
        self._dh_buf.copy_(h_last)
        self._dtok_buf.copy_(tok)
        self._dpos_buf.fill_(pos)
        self._vpos_buf.copy_(torch.arange(pos, pos + K + 1, device=dev))

        def draft_chain():
            for k in range(K):
                lg, dh = self._mtp(self._dh_buf, self._dtok_buf,
                                   self._dpos_buf, self.cache,
                                   self._row(self._dpos_buf))
                self._dh_buf.copy_(dh)
                nxt = lg[:, -1].float().argmax(-1)
                self._dtoks_out[k].copy_(nxt[0])
                self._dtok_buf.copy_(nxt.view(1, 1))
                self._dpos_buf.add_(1)

        # the verify pass runs every round: when the step is compiled,
        # verify takes the same compile-then-capture recipe — inductor's
        # elementwise fusion is the same third it buys the plain step
        if self._use_compile and getattr(self, "_fwd_full_c", None) is None:
            self._fwd_full_c = torch.compile(self._fwd_full,
                                             dynamic=False)
        fwd_v = (self._fwd_full_c if self._use_compile
                 else self._fwd_full)

        def verify():
            lg, hn = fwd_v(self._vseq_buf, self._vpos_buf)
            return lg, hn

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            draft_chain()
            self._vseq_buf[:, 0].copy_(tok[0])
            self._vseq_buf[:, 1:].copy_(self._dtoks_out)
            verify()
        torch.cuda.current_stream().wait_stream(side)
        for i, cs, rs in snaps:
            self.cache.conv_states[i].copy_(cs)
            self.cache.recurrent_states[i].copy_(rs)
        def snap_states():
            # the pre-round state snapshot rides inside the draft graph:
            # sixty host-launched little copies a round otherwise sit
            # squarely inside the per-round sync window
            for i, (cb, rb) in self._snap_bufs.items():
                cb.copy_(self.cache.conv_states[i])
                rb.copy_(self.cache.recurrent_states[i])

        self._dgraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._dgraph):
            snap_states()
            draft_chain()
        # rejected rounds re-advance the accepted prefix; each of the K
        # possible prefix lengths is a fixed shape, so each gets its own
        # captured pass — the reject path replays instead of paying an
        # eager whole-model forward
        self._ra_seq, self._ra_pos = {}, {}
        self._ra_lg, self._ra_hn = {}, {}
        self._ra_graphs = {}
        for m in range(1, K + 1):
            self._ra_seq[m] = torch.empty(1, m, dtype=torch.long,
                                          device=dev)
            self._ra_pos[m] = torch.empty(m, dtype=torch.long, device=dev)
            self._ra_seq[m].copy_(self._vseq_buf[:, :m])
            self._ra_pos[m].copy_(self._vpos_buf[:m])
        side_ra = torch.cuda.Stream()
        side_ra.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_ra):
            for m in range(1, K + 1):
                self._fwd_full(self._ra_seq[m], self._ra_pos[m])
        torch.cuda.current_stream().wait_stream(side_ra)
        for i, cs, rs in snaps:
            self.cache.conv_states[i].copy_(cs)
            self.cache.recurrent_states[i].copy_(rs)
        for m in range(1, K + 1):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                # the rollback restore rides in-graph too: the draft
                # graph snapshotted these buffers before the round
                for i, (cb, rb) in self._snap_bufs.items():
                    self.cache.conv_states[i].copy_(cb)
                    self.cache.recurrent_states[i].copy_(rb)
                self._ra_lg[m], self._ra_hn[m] = self._fwd_full(
                    self._ra_seq[m], self._ra_pos[m])
            self._ra_graphs[m] = g
        # the draft-KV rewrite over a cut-short accepted region is the
        # same story: K possible shapes, each captured once. The draft
        # layer carries no gated-delta state, so no snapshot dance.
        hdim = self._dh_buf.shape[-1]
        self._rwm_h, self._rwm_graphs = {}, {}
        side_rw = torch.cuda.Stream()
        side_rw.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_rw):
            for m in range(1, K + 1):
                self._rwm_h[m] = torch.zeros(1, m, hdim, device=dev,
                                             dtype=torch.bfloat16)
                self._mtp(self._rwm_h[m], self._ra_seq[m],
                          self._ra_pos[m], self.cache,
                          self._row(self._ra_pos[m]))
        torch.cuda.current_stream().wait_stream(side_rw)
        for m in range(1, K + 1):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._mtp(self._rwm_h[m], self._ra_seq[m],
                          self._ra_pos[m], self.cache,
                          self._row(self._ra_pos[m]))
            self._rwm_graphs[m] = g
        if not getattr(self, "_verify_capture", True):
            # multi-token passes stay eager (MoE hosts: the T>1 expert
            # path routes on the host); the draft chain above is the
            # captured piece either way
            self._vgraph = None
            self._rwgraph = None
            return
        self._vgraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._vgraph):
            self._vlg, self._vhn = verify()
        # full-accept rewrite pass is also fixed-shape (K+1 rows)
        self._rwh_buf = torch.empty(1, K + 1, self._dh_buf.shape[-1],
                                    device=dev, dtype=torch.bfloat16)
        side2 = torch.cuda.Stream()
        side2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side2):
            self._mtp(self._rwh_buf.zero_(), self._vseq_buf,
                      self._vpos_buf, self.cache,
                      self._row(self._vpos_buf))
        torch.cuda.current_stream().wait_stream(side2)
        self._rwgraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._rwgraph):
            self._mtp(self._rwh_buf, self._vseq_buf, self._vpos_buf,
                      self.cache, self._row(self._vpos_buf))

    def _seed_spec(self, h_last, tok, pos, K, dev, snaps):
        self._ensure_spec_graphs(K, dev, snaps, h_last, tok, pos)
        self._dh_buf.copy_(h_last)
        self._dtok_buf.copy_(tok)
        self._dpos_buf.fill_(pos)
        self._dgraph.replay()

    @torch.no_grad()
    def generate_speculative(self, input_ids, max_new_tokens, K=None):
        """Greedy MTP spec decode; exact vs plain greedy by verify.

        ``K`` (draft chain length) defaults to the value set at
        ``enable_mtp``; passing a different K rebuilds the captured
        passes for the new shape."""
        if getattr(self, "_mtp", None) is None:
            raise ValueError("refused: enable_mtp first")
        K = int(K) if K is not None else self._default_k
        if getattr(self, "_spec_k", None) not in (None, K):
            # the captured passes are shaped by K — rebuild, never
            # replay a mismatched shape
            self._dgraph = None
            self._vgraph = None
            self._rwgraph = None
            self._ra_graphs = None
            self._rwm_graphs = None
        self._spec_k = K
        dev = input_ids.device
        L = int(input_ids.shape[1])
        if L + max_new_tokens + K + 1 > self._max:
            raise ValueError("refused: window too small for spec tail")
        self.cache.frt_continue = False
        logits, hn = self._fwd_full(
            input_ids, torch.arange(L, device=dev))
        self.cache.frt_continue = True
        tok = logits[:, -1:].float().argmax(-1)
        # draft KV over the prompt: position p keys on (h_{p-1}, y_p) —
        # one multi-position pass over the whole prompt
        if L > 1:
            pr = torch.arange(1, L, device=dev)
            self._mtp(hn[:, :L - 1], input_ids[:, 1:L], pr,
                      self.cache, self._row(pr))
        produced = [tok]
        pos = L
        h_last = hn[:, -1:]
        accepted_hist = []
        self._dgraph = getattr(self, "_dgraph", None)
        # snapshot buffers live across rounds: per-round clones would
        # allocate sixty tensors a round for nothing
        if getattr(self, "_snap_bufs", None) is None:
            self._snap_bufs = {
                i: (torch.empty_like(self.cache.conv_states[i]),
                    torch.empty_like(self.cache.recurrent_states[i]))
                for i in self._gdn_slots()}
        while len(produced) < max_new_tokens:
            k_eff = min(K, max_new_tokens - len(produced))
            if k_eff == K and getattr(self, "_dgraph", None) is not None:
                # the draft graph snapshots in-graph on replay
                snaps = [(i, cb, rb)
                         for i, (cb, rb) in self._snap_bufs.items()]
            else:
                snaps = []
                for i, (cb, rb) in self._snap_bufs.items():
                    cb.copy_(self.cache.conv_states[i])
                    rb.copy_(self.cache.recurrent_states[i])
                    snaps.append((i, cb, rb))
            if k_eff == K:
                # captured fast path: draft chain replay + verify replay
                if self._dgraph is None:
                    self._dh_buf_init = True
                    self._seed_spec(h_last, produced[-1], pos, K, dev,
                                    snaps)
                else:
                    self._dh_buf.copy_(h_last)
                    self._dtok_buf.copy_(produced[-1])
                    self._dpos_buf.fill_(pos)
                    self._dgraph.replay()
                self._vseq_buf[:, 0].copy_(produced[-1][0])
                self._vseq_buf[:, 1:].copy_(self._dtoks_out)
                self._vpos_buf.copy_(torch.arange(
                    pos, pos + K + 1, device=dev))
                if self._vgraph is not None:
                    self._vgraph.replay()
                    lg_v, hn_v = self._vlg, self._vhn
                else:
                    lg_v, hn_v = self._fwd_full(self._vseq_buf,
                                                self._vpos_buf)
                dtoks = [self._dtoks_out[k].view(1, 1)
                         for k in range(K)]
                seq = self._vseq_buf
            else:
                dtoks, dh, dtok = [], h_last, produced[-1]
                for k in range(k_eff):
                    pt = torch.tensor([pos + k], device=dev)
                    lg, dh = self._mtp(dh, dtok, pt, self.cache,
                                       self._row(pt))
                    dtok = lg[:, -1:].float().argmax(-1)
                    dtoks.append(dtok)
                seq = torch.cat([produced[-1]] + dtoks,
                                dim=1)[:, :k_eff + 1]
                pos_v = torch.arange(pos, pos + k_eff + 1, device=dev)
                lg_v, hn_v = self._fwd_full(seq[:, :k_eff + 1], pos_v)
            targets = lg_v.float().argmax(-1)          # (1, k_eff+1)
            # device-side prefix match: one sync per round, not one
            # per accepted token
            dstack = torch.cat([d.view(1) for d in dtoks])
            j = int((dstack == targets[0, :k_eff]).cumprod(0)
                    .sum().item())
            if j == k_eff:
                # full acceptance: the verify pass committed exactly
                # the accepted stream — no rollback, no second pass
                bonus = targets[:, -1:].contiguous()
                hn_x = hn_v
            else:
                # roll the gated-delta states back and re-advance the
                # accepted prefix — linear states cannot unwind
                m = j + 1
                ra = getattr(self, "_ra_graphs", None)
                if ra and m in ra:
                    # the rollback restore is captured at the head of
                    # the re-advance graph
                    self._ra_seq[m].copy_(seq[:, :m])
                    self._ra_pos[m].copy_(torch.arange(
                        pos, pos + m, device=dev))
                    ra[m].replay()
                    lg_a, hn_x = self._ra_lg[m], self._ra_hn[m]
                else:
                    for i, cs, rs in snaps:
                        self.cache.conv_states[i].copy_(cs)
                        self.cache.recurrent_states[i].copy_(rs)
                    pos_a = torch.arange(pos, pos + m, device=dev)
                    lg_a, hn_x = self._fwd_full(seq[:, :m], pos_a)
                bonus = lg_a[:, -1:].float().argmax(-1)
            # draft KV for the accepted region keys on main hiddens —
            # one multi-position pass; captured when the shape is the
            # full-accept one
            if (j == k_eff and k_eff == K
                    and getattr(self, "_rwgraph", None) is not None):
                self._rwh_buf[:, 0].copy_(h_last[:, 0])
                self._rwh_buf[:, 1:].copy_(hn_x[:, :K])
                self._rwgraph.replay()
            else:
                m = j + 1
                rw = getattr(self, "_rwm_graphs", None)
                if rw and m in rw:
                    self._rwm_h[m][:, :1].copy_(h_last)
                    if j > 0:
                        self._rwm_h[m][:, 1:].copy_(hn_x[:, :j])
                    self._ra_seq[m].copy_(seq[:, :m])
                    self._ra_pos[m].copy_(torch.arange(
                        pos, pos + m, device=dev))
                    rw[m].replay()
                else:
                    prev_rows = (torch.cat([h_last, hn_x[:, :j]], dim=1)
                                 if j > 0 else h_last)
                    pos_rows = torch.arange(pos, pos + m, device=dev)
                    self._mtp(prev_rows, seq[:, :m], pos_rows,
                              self.cache, self._row(pos_rows))
            for r in range(j):
                produced.append(dtoks[r].clone())
            produced.append(bonus.clone())
            accepted_hist.append(j + 1)
            pos += j + 1
            h_last = hn_x[:, j:j + 1]
        self.cache.frt_continue = False
        out = torch.cat([input_ids] + produced, dim=1)
        self.last_acceptance = (sum(accepted_hist) / len(accepted_hist)
                                if accepted_hist else 0.0)
        return out[:, :L + max_new_tokens]


class _PrefillVehicle(torch.nn.Module):
    """Export vehicle for the prompt pass.

    The state the pass mutates — the static KV rows, the rope delta —
    rides as registered buffers so functionalization records the writes
    as buffer mutations instead of refusing them. The host model hangs
    in the module hierarchy so every parameter exports under its own
    FQN. Loaded with the package borrowing these very tensors, the
    packaged writes land in the loop's own cache.
    """

    def __init__(self, loop):
        super().__init__()
        self.model = loop._model
        self._frt_loop = loop
        for i in loop._full:
            self.register_buffer(f"frt_k{i}", loop.cache.key_cache[i])
            self.register_buffer(f"frt_v{i}", loop.cache.value_cache[i])
        self.register_buffer("frt_rope_delta", loop._rope_delta)

    def forward(self, tok_ids, pos_t):
        # the cache reaches its KV through a plain python object, which
        # the exporter can only lift as an immutable constant; rerouted
        # through this module's own attributes the same tensors carry a
        # buffer source, and the prompt pass's index_copy_ becomes a
        # legal buffer mutation. Outside tracing the getattr returns
        # the very same tensors, so this is an identity rebind — but
        # under fake tracing it leaves proxies in the cache lists,
        # which ``aot_prefill`` restores right after export.
        loop = self._frt_loop
        for i in loop._full:
            loop.cache.key_cache[i] = getattr(self, f"frt_k{i}")
            loop.cache.value_cache[i] = getattr(self, f"frt_v{i}")
        return loop._fwd(tok_ids, pos_t)


def build_decode_loop(model, *, max_len, compile_step=True,
                      compile_prefill=True, kv_band=None):
    """Build the whole-loop form over whatever is attached to ``model``."""
    return WholeStepDecodeLoop(model, max_len=max_len,
                               compile_step=compile_step,
                               compile_prefill=compile_prefill,
                               kv_band=kv_band)
