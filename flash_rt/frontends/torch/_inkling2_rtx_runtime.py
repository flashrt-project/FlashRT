"""Inkling 2 Mini runtime on RTX SM120: buffers, weights and the kernel sequences of ``flash_rt_inkling2``.

Every forward is a recorded program: a fixed list of kernel launches over raw device pointers that reads the
position and the input tokens from device memory, so it is captured once into a CUDA graph and replayed.

* ``decode``: one token (greedy next token on the device, position advanced, token appended to ``history``).
* ``verify<W>``: the anchor and W - 1 drafts in W rows, greedy targets, acceptance and commit of the accepted rows'
  KV / convolution states (speculative decoding). Each row runs the decode step's arithmetic, so the targets equal
  the tokens W decode steps would produce, bit for bit.
* ``exact<R>`` / ``exact<R>_last``: R prompt rows on the same per-row arithmetic (bit-equal to R decode steps).
* ``prefill<R>`` / ``prefill<R>_last``: R prompt rows on tensor cores (BF16 projections and the router's FP32
  logits on the module's cuBLAS GEMM, grouped block-scaled MMA experts, tensor-core attention); not bit-equal to
  decode steps (different accumulation order), see ``docs/inkling2_rtx_usage.md``.

``*_last`` programs also run the LM head of the last row (the first generated token). Every tensor the kernels read
is checked against the checkpoint geometry (dtype and shape, from the safetensors headers) before any is loaded.
PyTorch is the allocator and captures the graphs; no PyTorch matmul setting is read or changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:  # the extension is optional at import time (see _require_module)
    from flash_rt import flash_rt_inkling2 as _ik
    _IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - exercised on machines without the build
    _ik, _IMPORT_ERROR = None, e

BF16, F32, I32, U8 = torch.bfloat16, torch.float32, torch.int32, torch.uint8
PREFIX = "model.llm."

# Text-decoder geometry the kernels are written for (config.json text_config).
GEOMETRY = {
    "hidden_size": 2304, "num_hidden_layers": 24, "num_attention_heads": 24, "num_key_value_heads": 8,
    "head_dim": 128, "swa_head_dim": 128, "swa_num_attention_heads": 24, "swa_num_key_value_heads": 8,
    "d_rel": 16, "rel_extent": 1024, "sliding_window_size": 1024, "sconv_kernel_size": 4,
    "n_layers_per_attn_res_block": 6, "n_routed_experts": 128, "num_experts_per_tok": 8, "n_shared_experts": 0,
    "intermediate_size": 1152, "q_bias": False, "o_bias": False, "use_gate_bias": True,
}
LOOP = {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": False}
D, H, HKV, HD, E, I, TOPK = 2304, 24, 8, 128, 128, 1152, 8
WINDOW, BLOCK_LAYERS, GATE_RANK, REL = 1024, 6, 64, 16
TARGET_ROWS = 8          # rows of the LM-head / target buffers (verification of at most 8 rows)
LM_BLOCKS = 680          # vocabulary blocks of the LM head's argmax


class Inkling2ConfigError(ValueError):
    """The checkpoint is not the Inkling 2 Mini text decoder the kernels are written for."""


def _require_module():
    if _ik is None:
        raise RuntimeError(f"flash_rt.flash_rt_inkling2 is not available ({_IMPORT_ERROR}): build it with "
                           "-DGPU_ARCH=120 -DFLASHRT_ENABLE_INKLING2=ON (see docs/inkling2_rtx_usage.md)")
    return _ik


def read_config(checkpoint_dir) -> Tuple[dict, dict]:
    """(config.json, text_config) of the checkpoint, checked against the geometry the kernels are written for."""
    path = Path(checkpoint_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    raw = json.loads(path.read_text())
    check_config(raw)
    return raw, raw.get("text_config", raw)


def check_config(raw: dict) -> None:
    c = raw.get("text_config", raw)
    if c.get("model_type") not in ("inkling2_loop_text", "inkling2_loop"):
        raise Inkling2ConfigError(f"model_type {c.get('model_type')!r}, expected the Inkling 2 text decoder")
    for key, value in GEOMETRY.items():
        if c.get(key) != value:
            raise Inkling2ConfigError(f"{key}={c.get(key)!r}; the SM120 Inkling 2 kernels are built for {value!r}")
    loop = c.get("loop_config") or {}
    for key, value in LOOP.items():
        if loop.get(key) != value:
            raise Inkling2ConfigError(f"loop_config {key}={loop.get(key)!r}; the kernels are built for {value!r}")
    local = set(c.get("local_layer_ids") or ())
    glob = [l for l in range(c["num_hidden_layers"]) if l not in local]
    # global layers are the last layer of each depth-attention block
    if glob != [b - 1 for b in range(BLOCK_LAYERS, c["num_hidden_layers"] + 1, BLOCK_LAYERS)]:
        raise Inkling2ConfigError(f"global layers {glob}: expected the last layer of each block of {BLOCK_LAYERS}")


def visit_count(c: dict) -> int:
    """Layer visits per token: the stored layers, the looped ones once more per extra pass."""
    lc = c["loop_config"]
    return c["num_hidden_layers"] + (lc["num_passes"] - 1) * (lc["core_end"] - lc["core_start"])


def checkpoint_shards(checkpoint_dir) -> List[Path]:
    """The text-decoder safetensors files (the index's weight map, else ``model*.safetensors``)."""
    d = Path(checkpoint_dir)
    index = d / "model.safetensors.index.json"
    if index.is_file():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        shards = [d / f for f in files]
    else:
        shards = sorted(d.glob("model*.safetensors"))
    if not shards or not all(s.is_file() for s in shards):
        raise FileNotFoundError(f"no model safetensors in {d}")
    return shards


def plan_chunks(n: int, chunk_rows: Sequence[int], exact_rows: int, exact: bool) -> List[Tuple[str, int]]:
    """Prompt of n tokens as (kind, rows): ``("chunk", R)`` tensor-core chunks (largest first), then
    ``("exact", exact_rows)`` chunks, then single decode ``("step", 1)``. ``exact`` uses no tensor-core chunks."""
    if n < 1:
        raise ValueError("empty prompt")
    out, left = [], n
    if not exact:
        for r in sorted(chunk_rows, reverse=True):
            while left >= r:
                out.append(("chunk", r))
                left -= r
    while left >= exact_rows:
        out.append(("exact", exact_rows))
        left -= exact_rows
    out += [("step", 1)] * left
    return out


def safetensors_header(path) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    """{name: (dtype, shape)} of a safetensors file, read from its header (no tensor data)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if not 2 <= n <= 1 << 28:
            raise Inkling2ConfigError(f"{path}: not a safetensors file")
        header = json.loads(f.read(n))
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in header.items() if k != "__metadata__"}


def check_tensors(found: Dict[str, Tuple[str, Tuple[int, ...]]], expected: Dict[str, Tuple[str, Tuple[int, ...]]],
                  what: str) -> None:
    """Every expected tensor present with its dtype and shape (the kernels index them by the geometry)."""
    missing = [k for k in expected if k not in found]
    wrong = [f"{k}: {found[k][0]} {list(found[k][1])}, expected {v[0]} {list(v[1])}"
             for k, v in expected.items() if k in found and tuple(found[k]) != (v[0], tuple(v[1]))]
    if missing or wrong:
        parts = ([f"{len(wrong)} tensors of another dtype or shape ({'; '.join(wrong[:3])})"] if wrong else []) + \
                ([f"{len(missing)} missing ({', '.join(missing[:3])})"] if missing else [])
        raise Inkling2ConfigError(f"{what}: " + ", ".join(parts))


def expert_formats(found: Dict[str, Tuple[str, Tuple[int, ...]]], layers: int) -> List[str]:
    """Per layer ``"nvfp4"`` (E2M1 codes in U8) or ``"mxfp8"`` (F8_E4M3 codes), from the stored gate/up codes."""
    out = []
    for l in range(layers):
        dt = found.get(f"layers.{l}.mlp.experts.w13_weight", (None,))[0]
        if dt is None:          # reported as missing by check_tensors
            out.append("nvfp4")
            continue
        if dt not in ("U8", "F8_E4M3"):
            raise Inkling2ConfigError(f"layer {l}: experts stored as {dt}, expected NVFP4 (U8) or MXFP8 (F8_E4M3)")
        out.append("nvfp4" if dt == "U8" else "mxfp8")
    return out


def expected_tensors(c: dict, formats: Sequence[str]) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    """The text-decoder tensors the kernels read (names without ``model.llm.``): (safetensors dtype, shape)."""
    bf, f32 = "BF16", "F32"
    V, L = c["vocab_size"], c["num_hidden_layers"]
    lc = c["loop_config"]
    core, local = range(lc["core_start"], lc["core_end"]), set(c["local_layer_ids"])
    t = {"embed.weight": (bf, (V, D)), "unembed.weight": (bf, (V, D)), "embed_norm.weight": (bf, (D,)),
         "norm.weight": (bf, (D,)), "attn_res.output_query": (f32, (D,)),
         "middle_cycle_cross_pass_queries": (f32, (2 * len(core), D)),
         "middle_cycle_cross_pass_kv_queries": (f32, (2 * len(core), D))}
    for l in range(L):
        p, a, ex = f"layers.{l}.", f"layers.{l}.attn.", f"layers.{l}.mlp.experts."
        sliding = l in local
        t.update({a + "wq_du.weight": (bf, (H * HD, D)), a + "wr_du.weight": (bf, (H * REL, D)),
                  a + "gate_attn.weight": (bf, (H * HD, D)), a + "attn_sink_proj.weight": (bf, (H, D)),
                  a + "wk_dv.weight": (bf, (HKV * HD, D)), a + "wv_dv.weight": (bf, (HKV * HD, D)),
                  a + "wo_ud.weight": (bf, (D, H * HD)), a + "q_l2_clip.scale": (bf, (H, HD)),
                  a + "rel_l2_clip.scale": (bf, (H, REL)), a + "rel_logits_proj.proj": (bf, (REL, WINDOW)),
                  p + "attn_sconv.weight": (bf, (D, 1, 4)), p + "mlp_sconv.weight": (bf, (D, 1, 4)),
                  p + "mlp_norm.weight": (bf, (D,)), p + "mlp.gate.weight": (bf, (E, D)),
                  p + "mlp.gate.bias": (f32, (E,)), f"attn_res.mlp_queries.{l}": (f32, (D,))})
        if l > 0:   # the depth-attention queries of the layer's input: one, or q / k / v for global layers
            t[f"attn_res.attn_queries.{l - 1}"] = (f32, (D,) if sliding else (3, D))
        if sliding:
            t.update({a + "k_sconv.weight": (bf, (HKV * HD, 1, 4)), a + "v_sconv.weight": (bf, (HKV * HD, 1, 4)),
                      p + "attn_norm.weight": (bf, (D,))})
        else:
            for n in "kv":
                t.update({a + f"{n}_sconv.static_w": (bf, (4, HKV * HD)), a + f"{n}_sconv.weight_proj": (bf, (32, D))})
            t.update({p + f"{n}_input_norm.weight": (bf, (D,)) for n in "qkv"})
        if l in core:
            for n in ("attn", "k", "v", "mlp"):
                cp = p + f"controllers.{n}."
                t.update({cp + "down_weight": (bf, (GATE_RANK, 2 * D)), cp + "forget_weight": (bf, (D, GATE_RANK)),
                          cp + "gain_weight": (bf, (D, GATE_RANK))})
        if formats[l] == "nvfp4":   # E2M1 codes (two per byte), E4M3 scales per 16, per-expert amax, denominator
            t.update({ex + "w13_weight": ("U8", (E, 2 * I, D // 2)), ex + "w13_weight_scale": ("F8_E4M3", (E, 2 * I, D // 16)),
                      ex + "w2_weight": ("U8", (E, D, I // 2)), ex + "w2_weight_scale": ("F8_E4M3", (E, D, I // 16))})
            for w in ("w13", "w2"):
                t.update({ex + f"{w}_weight_amax": (f32, (E,)), ex + f"{w}_weight_outer_scale_denominator": (f32, (1,))})
        else:                       # E4M3 codes, E8M0 scales per 32
            t.update({ex + "w13_weight": ("F8_E4M3", (E, 2 * I, D)), ex + "w13_weight_scale": ("U8", (E, 2 * I, D // 32)),
                      ex + "w2_weight": ("F8_E4M3", (E, D, I)), ex + "w2_weight_scale": ("U8", (E, D, I // 32))})
    return t


def gemm_call(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, workspace: torch.Tensor):
    """``c = a b^T`` on the module's cuBLAS GEMM (FP32 accumulation, precision fixed by the module): a [m][k] and
    b [n][k] BF16, c [m][n] BF16 or FP32, rows contiguous."""
    m, k = a.shape
    n = b.shape[0]
    if a.dtype != BF16 or b.dtype != BF16 or c.dtype not in (BF16, F32) or b.shape[1] != k or \
            tuple(c.shape) != (m, n) or a.stride(1) != 1 or b.stride(1) != 1 or c.stride(1) != 1:
        raise ValueError("gemm: a [m][k], b [n][k] BF16 and c [m][n] BF16 / FP32 with contiguous rows")
    return (_ik.gemm, dict(m=m, n=n, k=k, a=P(a), lda=a.stride(0), b=P(b), ldb=b.stride(0), c=P(c), ldc=c.stride(0),
                           out_f32=int(c.dtype == F32), workspace=P(workspace), workspace_bytes=nbytes(workspace)))


def P(t: torch.Tensor) -> int:
    return t.data_ptr()


def nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class Inkling2RtxRuntime:
    """One sequence on the native ``flash_rt_inkling2`` kernels.

    ``max_seq``: KV capacity of the global-attention layers (sliding-window layers keep a ring of 1024 positions).
    ``prefill_rows``: rows of the largest tensor-core prefill chunk (``chunk_rows`` lists the chunk sizes).
    ``exact_rows``: rows of the bit-exact prefill chunks and of the largest verification.
    ``mma_experts`` (grouped MMA experts in prompt chunks, else flat passes) applies to programs recorded after it is
    set; ``flash_prefill`` (tensor-core attention in prompt chunks) also sizes buffers and is fixed at construction.
    The runtime makes ``device`` the current CUDA device and launches on its current stream.
    """

    def __init__(self, checkpoint_dir: str, max_seq: int = 8192, device: int = 0, prefill_rows: int = 512,
                 chunk_rows: Sequence[int] = (512, 256, 128, 64, 32, 16), exact_rows: int = 8, prefetch: bool = True,
                 mma_experts: bool = True, flash_prefill: bool = True):
        ik = _require_module()
        if not torch.cuda.is_available():
            raise RuntimeError("Inkling 2 on RTX needs a CUDA device")
        if torch.cuda.get_device_capability(device) != (12, 0):
            raise RuntimeError(f"flash_rt_inkling2 is built for SM120; device {device} is "
                               f"SM{''.join(map(str, torch.cuda.get_device_capability(device)))}")
        if not 1 <= exact_rows <= ik.MAX_ROWS:
            raise ValueError(f"exact_rows must be in 1..{ik.MAX_ROWS}")
        if not ik.MAX_ROWS < prefill_rows <= ik.MAX_CHUNK:
            raise ValueError(f"prefill_rows must be in {ik.MAX_ROWS + 1}..{ik.MAX_CHUNK}")
        chunk_rows = tuple(sorted({int(r) for r in chunk_rows if ik.MAX_ROWS < r <= prefill_rows}, reverse=True))
        if not chunk_rows:
            raise ValueError("chunk_rows: at least one size in (MAX_ROWS, prefill_rows]")
        if max_seq < 2 * WINDOW:
            raise ValueError(f"max_seq must be at least {2 * WINDOW}")
        self.dir = Path(checkpoint_dir)
        self.raw, self.c = read_config(self.dir)
        torch.cuda.set_device(device)
        self.dev = torch.device("cuda", device)
        self.max_seq = int(max_seq)
        self.prefill_rows, self.chunk_rows, self.exact_rows = int(prefill_rows), chunk_rows, int(exact_rows)
        self.prefetch, self.mma_experts, self._flash_prefill = prefetch, mma_experts, bool(flash_prefill)
        c = self.c
        self.vocab = int(c["unpadded_vocab_size"])
        self.local = set(c["local_layer_ids"])
        lc = c["loop_config"]
        self.core = range(lc["core_start"], lc["core_end"])
        self.passes = lc["num_passes"]
        layers = list(range(c["num_hidden_layers"]))
        self.visits = layers[: self.core.start] + layers[self.core.start: self.core.stop] * self.passes + \
            layers[self.core.stop:]
        self.w = self._load()
        w = self.w
        q = w["middle_cycle_cross_pass_queries"].float().reshape(-1, 2, D)
        kv = w["middle_cycle_cross_pass_kv_queries"].float().reshape(-1, 2, D)
        self.message_query = torch.cat([q[:, :1], kv, q[:, 1:]], 1).contiguous()      # [12][4][D]
        self.query_f32 = {k: v.float().contiguous() for k, v in w.items() if k.startswith("attn_res.")}
        # tensor-core chunks: the relative projection transposed (the MMA bias tables)
        self.rel_t = {k: v.t().contiguous() for k, v in w.items() if k.endswith("rel_logits_proj.proj")}
        self.RC = max(exact_rows, prefill_rows)
        self._buffers()
        self.programs: Dict[str, list] = {}
        self.graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self.use_graphs = True
        self._gemm_ready = False
        self.drafter = None                 # the DSpark drafter recorded into this runtime (at most one)
        self.programs["decode"] = self._record(1, verify=False)

    @property
    def flash_prefill(self) -> bool:
        return self._flash_prefill

    # ------------------------------------------------------------------------------------------- weights
    def _load(self) -> Dict[str, torch.Tensor]:
        """The tensors of ``expected_tensors``, checked in the safetensors headers before anything is loaded."""
        from safetensors import safe_open
        found, where = {}, {}
        for shard in checkpoint_shards(self.dir):
            for key, spec in safetensors_header(shard).items():
                if key.startswith(PREFIX):
                    found[key[len(PREFIX):]], where[key[len(PREFIX):]] = spec, shard
        expected = expected_tensors(self.c, expert_formats(found, self.c["num_hidden_layers"]))
        check_tensors(found, expected, "text decoder weights")
        w: Dict[str, torch.Tensor] = {}
        for shard in sorted({where[k] for k in expected}):
            with safe_open(str(shard), "pt", device=str(self.dev)) as f:
                for key in expected:
                    if where[key] == shard:
                        w[key] = f.get_tensor(PREFIX + key)
        return w

    # ------------------------------------------------------------------------------------------- buffers
    def _buffers(self):
        ik = _ik
        dev, nv, RC = self.dev, len(self.visits), self.RC
        z = lambda *s, dt=F32: torch.zeros(*s, dtype=dt, device=dev)
        self.pos = z(1, dt=I32)
        self.token = z(1, dt=I32)                       # decode: the token fed / the greedy next token
        self.history = z(self.max_seq + 1, dt=I32)      # decode: history[p] = the token fed at position p
        self.vtokens = z(RC, dt=I32)                    # prompt / verification rows (anchor + drafts)
        self.targets = z(TARGET_ROWS, dt=I32)
        self.accepted = z(1, dt=I32); self.next_pos = z(1, dt=I32); self.step_tokens = z(TARGET_ROWS, dt=I32)
        self.hid_embed = z(RC, D, dt=BF16)
        self.hid = z(nv, RC, D, dt=BF16)
        self.normed = z(nv, RC, D)
        self.bank_state = z(self.passes, RC, D)
        self.att_in = z(RC, 3, D)
        self.msgs = z(RC, 4, D)
        self.gate_in_f = z(3 * RC * 2 * D, dt=BF16)    # [3][R][2D] per program
        self.states_f = z(3 * RC * GATE_RANK, dt=BF16)  # [3][R][64]
        self.mixed_f = z(3 * RC * D)                    # [3][R][D]
        self.q_lin = z(RC, H * HD, dt=BF16); self.r_lin = z(RC, H * REL, dt=BF16); self.g_lin = z(RC, H * HD, dt=BF16)
        self.s_lin = z(RC, H, dt=BF16); self.k_lin = z(RC, HKV * HD, dt=BF16); self.v_lin = z(RC, HKV * HD, dt=BF16)
        self.k_dyn = z(RC, 32, dt=BF16); self.v_dyn = z(RC, 32, dt=BF16)
        self.max_splits = (self.max_seq + ik.ATTN_SPLIT_KEYS - 1) // ik.ATTN_SPLIT_KEYS
        # split-attention partials: verification rows (tensor-core chunks use the flash kernel), or every chunk row
        split_rows = max(self.exact_rows, TARGET_ROWS) if self.flash_prefill else RC
        self.partial = z(HKV * self.max_splits * split_rows * 3 * (HD + 2))
        self.attn_counters = z(HKV * RC, dt=I32)
        self.q_pre = z(RC * HKV * (3 * HD + 3 * REL))  # prepared queries per row / KV head
        self.attn_out = z(RC, H * HD, dt=BF16)
        # the normalized projection inputs (q / k / v, the router), computed once per row
        self.nrm = z(3, RC, D, dt=BF16); self.nrm_mlp = z(RC, D, dt=BF16)
        self.o_out = z(RC, D, dt=BF16)
        self.residual = z(RC, D)
        self.mlp_in = z(RC, D)
        self.gate_in_mlp = z(RC, 2 * D, dt=BF16)
        self.states_mlp = z(RC, GATE_RANK, dt=BF16)
        self.mixed_mlp = z(RC, D)
        self.logits_r = z(RC, E)
        self.ids = z(RC, TOPK, dt=I32); self.route_w = z(RC, TOPK)
        self.act = z(RC * TOPK, I, dt=BF16)
        self.expert_out = z(RC * TOPK, D, dt=BF16)
        # flat expert passes: staged quantized inputs per row (w13) and per slot (w2)
        self.xq_rows = z(RC, D * 2, dt=U8); self.xq_rows_s = z(RC, D // 16); self.xq_rows_a = z(RC)
        self.xq_act = z(RC * TOPK, I * 2, dt=U8); self.xq_act_s = z(RC * TOPK, I // 16)
        self.xq_act_a = z(RC * TOPK)
        # grouped MMA experts (tensor-core chunks): raw quantized rows / slots, the slots sorted by expert
        self.xq_rows_raw = z(RC, D, dt=U8); self.xq_rows_sc8 = z(RC, D // 16, dt=U8)
        self.xq_act_raw = z(RC * TOPK, I, dt=U8); self.xq_act_sc8 = z(RC * TOPK, I // 16, dt=U8)
        self.slot_order = z(RC * TOPK, dt=I32); self.slot_offset = z(E + 1, dt=I32); self.tile_off = z(E + 1, dt=I32)
        self.att_final = z(RC, D)
        self.feat_final = z(RC, D, dt=BF16)              # the final normalized rows (drafter feature)
        self.lm_blocks = LM_BLOCKS
        self.lm_best = z(LM_BLOCKS * TARGET_ROWS); self.lm_index = z(LM_BLOCKS * TARGET_ROWS, dt=I32)
        self.lm_counter = z(1, dt=I32)
        self.logits = z(TARGET_ROWS, self.vocab)
        self.kv, self.conv = [], []
        for layer in self.visits:
            S = WINDOW if layer in self.local else self.max_seq
            self.kv.append((z(S, HKV, HD, dt=BF16), z(S, HKV, HD, dt=BF16)))
            self.conv.append([z(3, HKV * HD, dt=BF16), z(3, HKV * HD, dt=BF16), z(3, D, dt=BF16), z(3, D, dt=BF16)])
        # staged rows per visit (verification and prompt chunks), committed after acceptance
        self.st_kvin = z(nv, RC, 2, HKV * HD, dt=BF16)
        self.st_attn = z(nv, RC, D, dt=BF16)
        self.st_mlp = z(nv, RC, D, dt=BF16)
        self.st_kv = z(nv, RC, 2, HKV * HD, dt=BF16)
        rows = []
        for v, layer in enumerate(self.visits):
            k, vv, a, m = self.conv[v]
            sl = layer in self.local
            rows.append([P(k), P(vv), P(a), P(m), P(self.st_kvin[v]), P(self.st_attn[v]), P(self.st_mlp[v]),
                         P(self.kv[v][0]) if sl else 0, P(self.kv[v][1]) if sl else 0, P(self.st_kv[v]) if sl else 0,
                         1 if sl else 0])
        self.commit_table = torch.tensor(rows, dtype=torch.int64, device=dev)
        self.gemm_ws = z(ik.GEMM_WORKSPACE_BYTES, dt=U8)    # this runtime's cuBLAS workspace (captured graphs)

    def reset(self):
        """Start a new sequence: position 0 and empty convolution states. KV rows are always written before they
        are read (attention reads positions below the current one only), so they are not cleared."""
        self.pos.zero_()
        for cs in self.conv:
            for s in cs:
                s.zero_()

    def state(self):
        """Copies of the persistent state (KV caches, convolution states, position)."""
        return ([(k.clone(), v.clone()) for k, v in self.kv], [[s.clone() for s in cs] for cs in self.conv],
                self.pos.clone())

    def restore(self, st):
        kv, conv, pos = st
        for (k, v), (k0, v0) in zip(self.kv, kv):
            k.copy_(k0); v.copy_(v0)
        for cs, cs0 in zip(self.conv, conv):
            for s, s0 in zip(cs, cs0):
                s.copy_(s0)
        self.pos.copy_(pos)

    # -------------------------------------------------------------------------------------------- record
    def _pass_of(self, v):
        n0, n1 = self.core.start, len(self.core)
        if v < n0:
            return 0
        if v < n0 + self.passes * n1:
            return 1 + (v - n0) // n1
        return self.passes + 1

    def _input_of(self, v):
        n0, n1 = self.core.start, len(self.core)
        if v == 0:
            return self.hid_embed
        if n0 <= v < n0 + self.passes * n1 and (v - n0) % n1 == 0:
            return self.hid[n0 - 1]
        return self.hid[v - 1]

    def _blocks_before(self, v):
        """Depth-attention value blocks visible at visit v: the embedding block and the block boundaries of the
        current pass (the coda continues the last pass)."""
        n0, n1 = self.core.start, len(self.core)
        p = self._pass_of(v)
        blocks = [self.normed[0]] if v > 0 else []
        if p == 0:
            return blocks
        start = n0 + (min(p, self.passes) - 1) * n1
        return blocks + [self.normed[u] for u in range(start, v) if self.visits[u] % BLOCK_LAYERS == 0]

    def _proj_ranges(self, v):
        layer = self.visits[v]
        a = f"layers.{layer}.attn."
        ts = [self.w[a + n + ".weight"] for n in ("wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv")]
        if layer not in self.local:
            ts += [self.w[a + "k_sconv.weight_proj"], self.w[a + "v_sconv.weight_proj"]]
        return [(P(t), nbytes(t)) for t in ts]

    def _finish(self, v, verify):
        p = f"layers.{self.visits[v]}."
        d = dict(finish=1, expert_out=P(self.expert_out), route_w=P(self.route_w), residual=P(self.residual),
                 mlp_conv_state=P(self.conv[v][3]), mlp_conv_w=P(self.w[p + "mlp_sconv.weight"]),
                 hidden_out=P(self.hid[v]))
        if verify:
            d["stage"] = P(self.st_mlp[v])
        return d

    def _record(self, R, verify, prefill=None):
        """The kernel sequence of one forward over R rows.

        ``verify=False, prefill=None``: decode (R = 1). ``verify=True``: verification (rows staged, greedy targets,
        accept + commit). ``prefill="rows"``: R prompt rows staged and all committed, the final-norm rows for the
        drafter, no LM head; ``prefill="last"``: also the LM head of the last row (-> ``token``). R > MAX_ROWS
        runs the tensor-core path (cuBLAS projections, grouped MMA experts, flash attention)."""
        ik = _ik
        c, w = self.c, self.w
        staged = verify or prefill is not None
        calls = []
        nv = len(self.visits)
        n0, n1 = self.core.start, len(self.core)
        pass_ends = [n0 + k * n1 - 1 for k in range(1, self.passes)]
        gate_in = self.gate_in_f[: 3 * R * 2 * D].view(3, R, 2 * D)
        states = self.states_f[: 3 * R * GATE_RANK].view(3, R, GATE_RANK)
        mixed = self.mixed_f[: 3 * R * D].view(3, R, D)
        tokens = self.vtokens if staged else self.token
        big = R > ik.MAX_ROWS
        if big and (verify or prefill is None):
            raise ValueError(f"verification and decode programs take at most {ik.MAX_ROWS} rows")
        if R > self.RC:
            raise ValueError(f"{R} rows exceed the buffers ({self.RC})")
        softcap = float(c["final_logit_softcapping"])
        for v, layer in enumerate(self.visits):
            p = f"layers.{layer}."
            a = p + "attn."
            sliding = layer in self.local
            ps = self._pass_of(v)
            loop_msg = 2 <= ps <= self.passes
            ci = layer - self.core.start
            mx = w[p + "mlp.experts.w13_weight"].dtype != U8
            prev = v - 1
            bd = dict(rows=R, next_hidden=P(self._input_of(v)), normed_res=P(self.normed[v]))
            if v == 0:
                bd.update(embed=1, token=P(tokens), embed_table=P(w["embed.weight"]),
                          embed_norm=P(w["embed_norm.weight"]), hidden_out=P(self.hid_embed))
            else:
                bd.update(self._finish(prev, staged))
            if prev in pass_ends:
                bd["bank_partial"] = P(self.bank_state[self._pass_of(prev) - 1])
            if layer > 0:
                q = self.query_f32[f"attn_res.attn_queries.{layer - 1}"]
                bd.update(blocks=[P(b) for b in self._blocks_before(v)], att_query=P(q),
                          nq=1 if q.dim() == 1 else q.shape[0])
            bd["att_in"] = P(self.att_in)
            if loop_msg:
                bd.update(messages=1, bank_block=P(self.normed[n0 + (ps - 2) * n1 + BLOCK_LAYERS]),
                          bank_state=P(self.bank_state[ps - 2]), message_query=P(self.message_query[ci]),
                          messages_out=P(self.msgs), gate_in=P(gate_in))
            if self.prefetch:
                bd["prefetch"] = self._proj_ranges(v)
            calls.append((ik.visit_boundary, bd))
            nq = 0 if layer == 0 else (1 if sliding else 3)
            if loop_msg:
                names = ("attn", "k", "v")
                if big:
                    calls += [gemm_call(gate_in[g], w[p + f"controllers.{n}.down_weight"], states[g], self.gemm_ws)
                              for g, n in enumerate(names)]
                else:
                    calls.append(("gemv", [(P(w[p + f"controllers.{n}.down_weight"]), P(gate_in[g]), P(states[g]),
                                            GATE_RANK, 2 * D, 0, 0, ik.GEMV_PLAIN, 0, R)
                                           for g, n in enumerate(names)], []))
                hptr = [P(self.att_in)] * 3 if nq == 1 else [P(self.att_in) + g * D * 4 for g in range(3)]
                calls.append((ik.gate_mix, dict(rows=R, h_stride=3 * D, m_stride=4 * D, out_stride=D,
                                                forget=[P(w[p + f"controllers.{n}.forget_weight"]) for n in names],
                                                gain=[P(w[p + f"controllers.{n}.gain_weight"]) for n in names],
                                                states=[P(states[g]) for g in range(3)], h=hptr,
                                                m=[P(self.msgs) + g * D * 4 for g in range(3)],
                                                out=[P(mixed[g]) for g in range(3)])))
                xin = [(P(mixed[g]), 1) for g in range(3)]
            elif layer == 0:
                xin = [(P(self.hid_embed), 0)] * 3
            else:
                # attention inputs: att_in rows [R][3][D] (row stride 3 D), query 0 or the q / k / v queries
                xin = [(P(self.att_in) + (g * D * 4 if nq == 3 else 0), 1) for g in range(3)]
            norms = [w[p + "attn_norm.weight"]] * 3 if sliding else [w[p + f"{n}_input_norm.weight"] for n in "qkv"]
            groups = ([("wq_du", self.q_lin, H * HD), ("wr_du", self.r_lin, H * REL),
                       ("gate_attn", self.g_lin, H * HD), ("attn_sink_proj", self.s_lin, H)],
                      [("wk_dv", self.k_lin, HKV * HD)] +
                      ([] if sliding else [("k_sconv.weight_proj", self.k_dyn, 32)]),
                      [("wv_dv", self.v_lin, HKV * HD)] +
                      ([] if sliding else [("v_sconv.weight_proj", self.v_dyn, 32)]))
            xs = 3 * D if (layer > 0 and not loop_msg) else 0
            src = [0, 1, 2]
            if R > 1:
                # the norm once per (input, weight) and row, the projections plain on the normalized rows
                uniq, src = [], []
                for g in range(3):
                    key = (xin[g][0], P(norms[g]))
                    if key not in uniq:
                        uniq.append(key)
                    src.append(uniq.index(key))
                calls.append(("norm", [(xin[g][0], xin[g][1], xs, P(norms[g]), P(self.nrm[src[g]]), R, D)
                                       for g in [src.index(u) for u in range(len(uniq))]]))
                xin = [(P(self.nrm[src[g]]), 0) for g in range(3)]
            tasks = []
            for g, items in enumerate(groups):
                for name, y, n in items:
                    wt = w[a + name] if name.endswith("weight_proj") else w[a + name + ".weight"]
                    if big:
                        calls.append(gemm_call(self.nrm[src[g]][:R], wt, y[:R], self.gemm_ws))
                    elif R > 1:
                        tasks.append((P(wt), xin[g][0], P(y), n, D, 0, 0, ik.GEMV_PLAIN, 0, R, 0))
                    else:
                        tasks.append((P(wt), xin[g][0], P(y), n, D, xin[g][1], 0, ik.GEMV_NORM, P(norms[g]), R, xs))
            if tasks:
                calls.append(("gemv", tasks, []))
            kc, vc = self.kv[v]
            kcs, vcs, acs, mcs = self.conv[v]
            att = dict(rows=R, sliding=int(sliding), pos=P(self.pos), q_lin=P(self.q_lin), r_lin=P(self.r_lin),
                       k_lin=P(self.k_lin), v_lin=P(self.v_lin), k_conv_state=P(kcs), v_conv_state=P(vcs),
                       q_clip=P(w[a + "q_l2_clip.scale"]), rel_clip=P(w[a + "rel_l2_clip.scale"]),
                       rel_proj=P(w[a + "rel_logits_proj.proj"]), qk_threshold=c["qk_l2_clip_threshold"],
                       rel_threshold=c["rel_l2_clip_threshold"], log_alpha=c["log_scaling_alpha"],
                       log_floor=float(c["log_scaling_n_floor"]), k_cache=P(kc), v_cache=P(vc), sinks=P(self.s_lin),
                       partial=P(self.partial), counters=P(self.attn_counters),
                       max_splits=WINDOW // ik.ATTN_SPLIT_KEYS if sliding else self.max_splits,
                       out=P(self.attn_out), gate=P(self.g_lin))     # the output gate is applied by the merge
            if sliding:
                att.update(k_conv_w=P(w[a + "k_sconv.weight"]), v_conv_w=P(w[a + "v_sconv.weight"]))
            else:
                att.update(k_conv_w=P(w[a + "k_sconv.static_w"]), v_conv_w=P(w[a + "v_sconv.static_w"]),
                           k_dyn=P(self.k_dyn), v_dyn=P(self.v_dyn))
            if staged:
                att.update(stage_in=P(self.st_kvin[v]), stage_kv=P(self.st_kv[v]), q_pre=P(self.q_pre))
                calls.append((ik.kv_rows, dict(att)))
            if self.prefetch:
                wo = w[a + "wo_ud.weight"]
                att["prefetch"] = [(P(wo), nbytes(wo))]
            if big and self.flash_prefill:
                calls.append((ik.attn_flash, dict(att, rel_t=P(self.rel_t[a + "rel_logits_proj.proj"]))))
            else:
                calls.append((ik.attn_decode, att))
            if big:
                calls.append(gemm_call(self.attn_out[:R], w[a + "wo_ud.weight"], self.o_out[:R], self.gemm_ws))
            else:
                calls.append(("gemv", [(P(w[a + "wo_ud.weight"]), P(self.attn_out), P(self.o_out), D, H * HD, 0, 0,
                                        ik.GEMV_PLAIN, 0, R)], []))
            boundary = layer % BLOCK_LAYERS == 0
            blocks = self._blocks_before(v) + ([self.normed[v]] if boundary else [])
            post = dict(rows=R, attn_out=P(self.o_out), block_residual=P(self._input_of(v)), attn_conv_state=P(acs),
                        attn_conv_w=P(w[p + "attn_sconv.weight"]), boundary=int(boundary), residual=P(self.residual),
                        blocks=[P(b) for b in blocks], mlp_query=P(self.query_f32[f"attn_res.mlp_queries.{layer}"]),
                        mlp_in=P(self.mlp_in))
            if staged:
                post["stage"] = P(self.st_attn[v])
            if loop_msg:
                post.update(message=P(self.msgs) + 3 * D * 4, gate_in=P(self.gate_in_mlp))
            if self.prefetch:
                post["prefetch"] = [(P(t), nbytes(t)) for t in [w[p + "mlp.gate.weight"]] + (
                    [w[p + "controllers.mlp.down_weight"], w[p + "controllers.mlp.forget_weight"],
                     w[p + "controllers.mlp.gain_weight"]] if loop_msg else [])]
            calls.append((ik.post_attention, post))
            router_in = self.mlp_in
            if loop_msg:
                if big:
                    calls.append(gemm_call(self.gate_in_mlp[:R], w[p + "controllers.mlp.down_weight"],
                                           self.states_mlp[:R], self.gemm_ws))
                else:
                    calls.append(("gemv", [(P(w[p + "controllers.mlp.down_weight"]), P(self.gate_in_mlp),
                                            P(self.states_mlp), GATE_RANK, 2 * D, 0, 0, ik.GEMV_PLAIN, 0, R)], []))
                calls.append((ik.gate_mix, dict(rows=R, h_stride=D, m_stride=4 * D, out_stride=D,
                                                forget=[P(w[p + "controllers.mlp.forget_weight"])],
                                                gain=[P(w[p + "controllers.mlp.gain_weight"])],
                                                states=[P(self.states_mlp)], h=[P(self.mlp_in)],
                                                m=[P(self.msgs) + 3 * D * 4], out=[P(self.mixed_mlp)])))
                router_in = self.mixed_mlp
            m = p + "mlp."
            if R > 1:
                calls.append(("norm", [(P(router_in), 1, 0, P(w[p + "mlp_norm.weight"]), P(self.nrm_mlp), R, D)]))
                if big:
                    # router logits: BF16 products summed and written in FP32
                    calls.append(gemm_call(self.nrm_mlp[:R], w[m + "gate.weight"], self.logits_r[:R], self.gemm_ws))
                else:
                    calls.append(("gemv", [(P(w[m + "gate.weight"]), P(self.nrm_mlp), P(self.logits_r), E, D, 0, 1,
                                            ik.GEMV_PLAIN, 0, R)], []))
            else:
                calls.append(("gemv", [(P(w[m + "gate.weight"]), P(router_in), P(self.logits_r), E, D, 1, 1,
                                        ik.GEMV_NORM, P(w[p + "mlp_norm.weight"]), R)], []))
            calls += self._experts(p, R, mx, router_in, big)
        last = nv - 1
        fin = dict(self._finish(last, staged), rows=R, next_hidden=P(self.hid[last]), normed_res=P(self.att_final),
                   blocks=[P(b) for b in self._blocks_before(last)],
                   att_query=P(self.query_f32["attn_res.output_query"]), nq=1, att_in=P(self.att_in))
        calls.append((ik.visit_boundary, fin))
        head = dict(norm_w=P(w["norm.weight"]), weight=P(w["unembed.weight"]), vocab=self.vocab, softcap=softcap,
                    blocks=self.lm_blocks, logits=P(self.logits), block_best=P(self.lm_best),
                    block_index=P(self.lm_index), counter=P(self.lm_counter), in_stride=3 * D)
        if prefill is not None:
            # the final-norm rows (the LM head's normalization, the drafter's last feature), all rows committed
            calls.append(("norm", [(P(self.att_in), 1, 3 * D, P(w["norm.weight"]), P(self.feat_final), R, D)]))
            if prefill == "last":
                calls.append((ik.lm_head, dict(head, rows=1, token=P(self.token),
                                               **{"in": P(self.att_in) + (R - 1) * 3 * D * 4})))
            calls.append(("all", R))
            calls.append((ik.commit, dict(visits=nv, table=P(self.commit_table), accepted=P(self.accepted),
                                          pos=P(self.pos), rows=R)))
            calls.append(("copy", (self.pos, self.next_pos)))
            return calls
        calls.append((ik.lm_head, dict(head, rows=R, token=P(self.targets if verify else self.token),
                                       normed_out=P(self.feat_final), **{"in": P(self.att_in)})))
        if verify:
            calls.append((ik.accept, dict(rows=R, drafts=P(self.vtokens) + 4, target=P(self.targets), pos=P(self.pos),
                                          accepted=P(self.accepted), tokens=P(self.step_tokens),
                                          next_token=P(self.vtokens), next_pos=P(self.next_pos))))
            calls.append((ik.commit, dict(visits=nv, table=P(self.commit_table), accepted=P(self.accepted),
                                          pos=P(self.pos), rows=R)))
            calls.append(("copy", (self.pos, self.next_pos)))
        else:
            calls.append(("advance", None))
        return calls

    def _experts(self, p, R, mx, router_in, big):
        """Routing and the routed experts of one visit: fused per-row kernels (one row), flat passes over slices of
        at most MAX_ROWS rows, or grouped block-scaled MMAs over the slots sorted by expert (tensor-core chunks)."""
        ik, w, c = _ik, self.w, self.c
        m, ex = p + "mlp.", p + "mlp.experts."
        caps = dict(gate_cap=float(c["mlp_gate_softcap"]), up_cap=float(c["mlp_up_softcap"]))
        route = dict(bias=P(w[m + "gate.bias"]), route_scale=float(c["route_scale"]),
                     norm_w=P(w[p + "mlp_norm.weight"]))
        wq = {}
        for which in ("w13", "w2"):
            wq[which] = dict(mxfp8=int(mx), codes=P(w[ex + f"{which}_weight"]),
                             scales=P(w[ex + f"{which}_weight_scale"]))
            if not mx:
                wq[which].update(amax=P(w[ex + f"{which}_weight_amax"]), denominator=float(
                    w[ex + f"{which}_weight_outer_scale_denominator"].float().reshape(-1)[0]))
        calls = []
        if R == 1:
            calls.append((ik.expert_w13, dict(wq["w13"], rows=R, ids=P(self.ids), act=P(self.act),
                                              out=P(self.expert_out), logits=P(self.logits_r), x_in=P(router_in),
                                              ids_out=P(self.ids), weights_out=P(self.route_w), **caps, **route)))
            calls.append((ik.expert_w2, dict(wq["w2"], rows=R, ids=P(self.ids), act=P(self.act),
                                             out=P(self.expert_out), **caps)))
            return calls
        if big and self.mma_experts:
            srt = dict(rows=R, ids=P(self.ids), order=P(self.slot_order), offset=P(self.slot_offset),
                       tile_off=P(self.tile_off), max_tiles=R * TOPK // ik.MMA_TILE_SLOTS + E)
            rows_q = dict(xq=P(self.xq_rows), xq_scale=P(self.xq_rows_s), xq_amax=P(self.xq_rows_a),
                          xq_codes=P(self.xq_rows_raw), xq_sc8=P(self.xq_rows_sc8))
            act_q = dict(xq=P(self.xq_act), xq_scale=P(self.xq_act_s), xq_amax=P(self.xq_act_a),
                         xq_codes=P(self.xq_act_raw), xq_sc8=P(self.xq_act_sc8))
            calls.append((ik.expert_route, dict(route, rows=R, mxfp8=int(mx), logits=P(self.logits_r),
                                                x_in=P(router_in), ids_out=P(self.ids), weights_out=P(self.route_w),
                                                **rows_q)))
            calls.append((ik.expert_sort, srt))
            calls.append((ik.expert_mma_w13, dict(wq["w13"], **srt, **caps, act=P(self.act), **rows_q)))
            calls.append((ik.expert_act_quant, dict(rows=R, mxfp8=int(mx), act=P(self.act), **act_q)))
            calls.append((ik.expert_mma_w2, dict(wq["w2"], **srt, out=P(self.expert_out), **act_q)))
            return calls
        # flat passes over slices of at most MAX_ROWS rows (pointers offset by the slice's rows)
        for s0 in range(0, R, ik.MAX_ROWS):
            r = min(ik.MAX_ROWS, R - s0)
            o_ids, o_slot = s0 * TOPK * 4, s0 * TOPK
            rows_q = dict(xq=P(self.xq_rows) + s0 * D * 2, xq_scale=P(self.xq_rows_s) + s0 * (D // 16) * 4,
                          xq_amax=P(self.xq_rows_a) + s0 * 4)
            act_q = dict(xq=P(self.xq_act) + o_slot * I * 2, xq_scale=P(self.xq_act_s) + o_slot * (I // 16) * 4,
                         xq_amax=P(self.xq_act_a) + o_slot * 4)
            ids, act = P(self.ids) + o_ids, P(self.act) + o_slot * I * 2
            calls.append((ik.expert_route, dict(route, rows=r, mxfp8=int(mx), logits=P(self.logits_r) + s0 * E * 4,
                                                x_in=P(router_in) + s0 * D * 4, ids_out=ids,
                                                weights_out=P(self.route_w) + o_ids, **rows_q)))
            calls.append((ik.expert_w13_flat, dict(wq["w13"], rows=r, ids=ids, act=act, **caps, **rows_q)))
            calls.append((ik.expert_act_quant, dict(rows=r, mxfp8=int(mx), act=act, **act_q)))
            calls.append((ik.expert_w2_flat, dict(wq["w2"], rows=r, ids=ids, out=P(self.expert_out) + o_slot * D * 2,
                                                  **act_q)))
        return calls

    # --------------------------------------------------------------------------------------------- run
    def _run(self, calls):
        ik = _ik
        s = torch.cuda.current_stream().cuda_stream
        for call in calls:
            fn = call[0]
            if fn == "gemv":
                ik.gemv_multi(call[1], call[2], s)
            elif fn == "norm":
                ik.norm_rows(call[1], s)
            elif fn == "advance":
                self.pos.add_(1)
                self.history.index_copy_(0, self.pos.long(), self.token)
            elif fn == "copy":
                call[1][0].copy_(call[1][1])
            elif fn == "zero":
                call[1].zero_()
            elif fn == "all":                       # a prompt chunk: every row accepted
                self.accepted.fill_(call[1] - 1)
                self.next_pos.copy_(self.pos)
                self.next_pos.add_(call[1])
            else:
                fn(call[1], s)

    def _warm_gemm(self):
        """The module's cuBLAS handle is created on its first product, which cannot happen during capture."""
        a = torch.zeros(16, 16, device=self.dev, dtype=BF16)
        self._run([gemm_call(a, a, torch.empty(16, 16, device=self.dev, dtype=F32), self.gemm_ws)])
        torch.cuda.synchronize()
        self._gemm_ready = True

    def capture(self, name: str):
        if not self._gemm_ready and any(c[0] is _ik.gemm for c in self.programs[name]):
            self._warm_gemm()
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            self._run(self.programs[name])
        self.graphs[name] = g

    def run(self, name: str):
        """Run a program: its CUDA graph (captured on first use), or eager launches with ``use_graphs=False``."""
        if not self.use_graphs:
            self._run(self.programs[name])
            return
        if name not in self.graphs:
            self.capture(name)
        self.graphs[name].replay()

    def program(self, kind: str, rows: int) -> str:
        """Name of a program, recorded on first use: ``chunk`` / ``chunk_last`` (tensor-core prompt chunks),
        ``exact`` / ``exact_last`` (per-row prompt chunks), ``verify`` (verification of ``rows`` rows)."""
        base = {"chunk": "prefill", "chunk_last": "prefill", "exact": "exact", "exact_last": "exact",
                "verify": "verify"}[kind]
        name = f"{base}{rows}" + ("_last" if kind.endswith("_last") else "")
        if name not in self.programs:
            if kind == "verify":
                self.programs[name] = self._record(rows, verify=True)
            else:
                self.programs[name] = self._record(rows, verify=False,
                                                   prefill="last" if kind.endswith("_last") else "rows")
        return name

    def step(self, token: Optional[int] = None):
        """One decode step (``token``: the input token, else the device token left by the previous step)."""
        if token is not None:
            self.token.fill_(int(token))
        self.run("decode")

    def prefill(self, ids: Sequence[int], exact: bool = False, after=None) -> int:
        """Run the prompt from the current position (see ``plan_chunks``); returns the first generated token.
        ``exact``: per-row chunks only (bit-equal to decode steps). ``after(kind, rows)``: called after each chunk or
        step (the drafter writes its context there)."""
        plan = plan_chunks(len(ids), self.chunk_rows, self.exact_rows, exact)
        i = 0
        for j, (kind, rows) in enumerate(plan):
            last = j == len(plan) - 1
            if kind == "step":
                self.step(ids[i])
            else:
                self.vtokens[:rows].copy_(torch.tensor(ids[i:i + rows], dtype=I32))
                self.run(self.program(kind + ("_last" if last else ""), rows))
            if after is not None:
                after(kind, rows)
            i += rows
        return int(self.token.item())
