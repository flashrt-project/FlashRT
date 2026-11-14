"""Higgs delay-mask, categorical sampling and fused codebook embedding.

The SplitMix64 request-seed/step/codebook mapping matches the CUDA helper.
Sampling and host EOC decisions remain outside the captured backbone step.
"""
import triton
import triton.language as tl


@triton.jit
def _uniform(seed, step, cb):
    key = tl.cast(seed, tl.uint64) ^ (tl.cast(step, tl.uint64) * 0xD2B74407B1CE6E93)
    key ^= tl.cast(cb, tl.uint64) * 0xCA5A826395121157
    key += 0x9E3779B97F4A7C15
    key = (key ^ (key >> 30)) * 0xBF58476D1CE4E5B9
    key = (key ^ (key >> 27)) * 0x94D049BB133111EB
    key ^= key >> 31
    return ((key >> 40).to(tl.uint32).to(tl.float32) + 0.5) * (1.0 / 16777216.0)


@triton.jit
def _sample(LOGITS, CODES, V: tl.constexpr, NC: tl.constexpr, DELAY, BOC: tl.constexpr,
            TEMPERATURE, SEED, STEP, SAMPLED: tl.constexpr, BLOCK: tl.constexpr):
    cb = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    values = tl.load(LOGITS + cb * V + index, index < V, -float("inf")).to(tl.float32)
    maximum = tl.max(values, 0)
    if SAMPLED:
        weights = tl.exp((values - maximum) / TEMPERATURE)
        target = _uniform(SEED, STEP, cb) * tl.sum(weights, 0)
        cumulative = tl.cumsum(weights, 0)
        fallback = V - 1
        chosen = tl.min(tl.where((cumulative > target) & (index < V), index, fallback), 0)
    else:
        chosen = tl.min(tl.where((values == maximum) & (index < V), index, V - 1), 0)
    chosen = tl.where((DELAY < NC) & (cb > DELAY), BOC, chosen)
    tl.store(CODES + cb, chosen.to(tl.int64))


@triton.jit
def _embed(CODES, TABLE, OUT, NC: tl.constexpr, V: tl.constexpr,
           H: tl.constexpr, BLOCK: tl.constexpr):
    dim = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0.0, tl.float32)
    for cb in tl.static_range(NC):
        code = tl.load(CODES + cb)
        acc += tl.load(TABLE + (cb * V + code) * H + dim, dim < H, 0).to(tl.float32)
    tl.store(OUT + dim, acc, dim < H)


def delayed_codebook_sample_embed_bf16(logits, codebook, codes, embed, *, delay, boc,
                                       temperature, seed, step):
    nc, vocab = logits.shape
    _sample[(nc,)](logits, codes, vocab, nc, delay, boc, temperature, seed, step,
                   temperature > 1e-5, triton.next_power_of_2(vocab), num_warps=4)
    _embed[(triton.cdiv(embed.numel(), 256),)](
        codes, codebook, embed, nc, vocab, embed.numel(), 256, num_warps=4)


def delayed_codebook_argmax_embed_bf16(logits, codebook, codes, embed, *, delay, boc):
    delayed_codebook_sample_embed_bf16(
        logits, codebook, codes, embed, delay=delay, boc=boc,
        temperature=0.0, seed=0, step=0)
