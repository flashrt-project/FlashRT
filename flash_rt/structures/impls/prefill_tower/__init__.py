"""The plain-norm decoder tower region: the prefix-building pass.

The tower that fills the cache another stack later attends over: plain
RMS norms (no conditioning), bias-free attention and gated FFN, one
forward per observation with ``use_cache``. Its chain candidate
re-expresses the per-layer loop in static-FP8 hub primitives while
writing host-layout keys back into the host's own cache, so every
downstream consumer — the sibling chain or the host fallback — reads
what it always read.
"""
