"""End-to-end native DM05 predict: vision -> prefix cache -> 10-step action loop.

The frontend owns IO, calibration and graph capture; this module composes the
vision, Gemma3 prefix and action-expert computations through one `_run` path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .action import NativeActionExpert
from .text import NativePrefix
from .vision import NativeVision


def segments_from_token_types(token_type_ids) -> list[tuple[int, int, int, bool]]:
    """Attention segments of the Gemma3 prefix mask, from token types.

    Gemma3 is causal over text and bidirectional within each image block, so every
    text row i sees keys [0, i] and every row of an image block [a, b) sees keys
    [0, b). Returns (q_start, q_end, k_len, causal) runs: causal runs use FA2's
    bottom-right-aligned causal mask with k_len = q_end.
    """
    tt = (token_type_ids.reshape(-1).numpy() if not token_type_ids.is_cuda
          else token_type_ids.reshape(-1).cpu().numpy()).tolist()
    segs, i, n = [], 0, len(tt)
    while i < n:
        j = i
        while j < n and tt[j] == tt[i]:
            j += 1
        segs.append((i, j, j, tt[i] == 0))
        i = j
    return segs


@dataclass
class Dm05Config:
    """Precision boundary of the native pipeline (layer ranges are inclusive).

    The default is the configuration measured on the 444-observation LIBERO corpus:
    static NVFP4 for the prefix FFN and late attention projections, static per-tensor FP8 for the remaining prefix
    projections and the SigLIP tower, weight-only int8 for the action expert.
    """
    prefix_nvfp4: dict = field(default_factory=lambda: {"gate": (0, 33), "up": (0, 33), "down": (0, 33),
                                                        "q": (21, 33), "k": (21, 33), "v": (21, 33),
                                                        "o": (25, 33)})
    prefix_fp8: dict = field(default_factory=lambda: {"q": (0, 20), "k": (0, 20), "v": (0, 20), "o": (0, 24)})
    vision_fp8: tuple = ("qkv", "o", "fc1", "fc2")
    action_int8: tuple = ("q", "k", "v", "o", "gate", "up", "down")
    fuse_action_qkv: bool = True
    fuse_action_glu: bool = True
    # FP8 GEMM shapes left on cuBLASLt's heuristic algorithm instead of FlashRT's autotuned
    # one, as ("vt" | "lm", kind) names, e.g. ("vt.fc2",).
    fp8_autotune_skip: tuple = ()
    # SigLIP bias folding, exact in real arithmetic: "v" folds the value bias into o_proj's,
    # "kv" also drops the key bias; "" keeps the checkpoint's biases.
    vision_fold_bias: str = "v"
    fa2_splits: int = 16
    # key splits of the native GQA decode attention in the action expert (0: FlashRT FA2)
    ae_decode_splits: int = 8
    # SigLIP attention on the native head_dim-72 kernel instead of FA2's hdim-96 instantiation
    vision_attn_hd72: bool = True
    # SigLIP attention epilogue writes o_proj's static FP8 input (no standalone quantize)
    vision_attn_quant_epilogue: bool = True
    # prefix q|k|v as one FP8 GEMM writing q and the K/V cache in place (layers with FP8 q, k, v)
    prefix_fuse_qkv: bool = True
    # prefix attention writes o_proj's static FP8 input from its own epilogue
    prefix_attn_fp8_epilogue: bool = True
    # one native prefix-attention launch per layer (segment mask as a per-row key limit)
    prefix_attn_single_call: bool = False
    nvfp4_clip: float = 0.95
    # per-site NVFP4 clip overrides, keyed by (layer, kind)
    nvfp4_clip_sites: dict = field(default_factory=lambda: {**{(i, "down"): 0.85 for i in range(0, 10)},
                                                             **{(i, "down"): 0.9 for i in (10, 11, 12)}})


class Dm05PipelineRtx:
    def __init__(self, dm05, num_views: int = 2, max_prefix: int = 640, steps: int = 10,
                 chunk: int = 10, invalidate_graphs=None):
        vlm = dm05.model.vlm.model
        self.dm05 = dm05
        self.image_token_id = int(dm05.config.vlm_config.image_token_index)
        self.vision = NativeVision(vlm.vision_tower, vlm.multi_modal_projector, num_views)
        self.prefix = NativePrefix(vlm.language_model, max_seq=max_prefix)
        self.action = NativeActionExpert(dm05, self.prefix, steps=steps, chunk=chunk)
        self.A = self.action.A
        self.config: Dm05Config | None = None
        self._invalidate_graphs = invalidate_graphs or (lambda: None)
        # raw uint8 views (native preprocessing): resized + normalized inside the graph
        self.raw_u8 = None
        self.raw_mode = False
        # host-staged predict (make_predict): pinned input/output buffers copied by the
        # graph itself, so a replay has no eager transfer around it
        self.host_in = None
        self.host_out = None  # frontend-owned pinned output staging
        self._host_out_np = None
        self.host_staged = False
        self._prompt_key = None
        self._noise_seed = None

    # ------------------------------------------------------------------ configuration
    def configure(self, cfg: Dm05Config) -> None:
        """Build the quantized sites. Weights of quantized sites are released afterwards."""
        from .attention import set_num_splits
        rng = lambda ab: range(ab[0], ab[1] + 1)
        self.prefix.enable_nvfp4({k: rng(v) for k, v in cfg.prefix_nvfp4.items()})
        self.prefix.enable_fp8({k: rng(v) for k, v in cfg.prefix_fp8.items()})
        if cfg.vision_fold_bias:
            self.vision.fold_kv_bias(drop_k="k" in cfg.vision_fold_bias)
        if cfg.vision_fp8:
            self.vision.enable_fp8(list(cfg.vision_fp8))
        from .fp8 import Fp8Linear
        skip = {tuple(x.split(".")) for x in cfg.fp8_autotune_skip}     # ("vt" | "lm", kind)
        for where, sites in (("vt", self.vision.fp8), ("lm", self.prefix.fp4)):
            for (_, kind), site in sites.items():
                if isinstance(site, Fp8Linear):
                    site.autotune = (where, kind) not in skip
        if cfg.action_int8:
            self.action.enable_int8w(list(cfg.action_int8))
            if cfg.fuse_action_qkv:
                self.action.fuse_qkv_int8()
            if cfg.fuse_action_glu:
                self.action.fuse_glu_int8()
            for s in self.action.int8w.values():
                s.drop_rows()
        if cfg.prefix_fuse_qkv:
            self.prefix.fuse_qkv_fp8()
        set_num_splits(cfg.fa2_splits)
        self.action.decode_splits = cfg.ae_decode_splits
        if cfg.prefix_attn_fp8_epilogue and self.prefix.HD == 256:
            from ._fa2_fp8 import load as _load_fa2_fp8
            self.prefix.attn_fp8_epilogue = _load_fa2_fp8()
        else:
            self.prefix.attn_fp8_epilogue = None
        self.prefix.single_call_attn = cfg.prefix_attn_single_call and self.prefix.HD == 256
        self.vision.attn_hd72 = cfg.vision_attn_hd72 and self.vision.HD == 72 and self.vision.seq_per_view % 128 == 0
        self.vision.attn_quant_epilogue = cfg.vision_attn_quant_epilogue and self.vision.attn_hd72
        self.config = cfg
        self._invalidate_graphs()

    def _static_sites(self):
        yield from (("prefix", k, s) for k, s in self.prefix.fp4.items())
        yield from (("vision", k, s) for k, s in self.vision.fp8.items())

    def begin_calibration(self) -> None:
        for _, _, site in self._static_sites():
            site.reset_calibration()

    def finish_calibration(self) -> dict:
        from .fp8 import Fp8Linear
        sites = list(self._static_sites())
        fp4 = self.prefix.fp4
        # consumers read a producer's quantized activation and never quantize themselves:
        # up <- gate; k, v <- the first of q/k/v present in that layer (same format)
        consumers = set()
        for (i, k), s in fp4.items():
            if k == "up" and (i, "gate") in fp4 and type(fp4[(i, "gate")]) is type(s):
                consumers.add((i, k))
            if k in ("k", "v"):
                first = next(x for x in ("q", "k", "v") if (i, x) in fp4)
                if first != k and type(fp4[(i, first)]) is type(s):
                    consumers.add((i, k))
        for where, key, s in sites:
            consumer = where == "prefix" and key in consumers
            if isinstance(s, Fp8Linear):
                s.finish_calibration(consumer=consumer)
            else:
                clip = self.config.nvfp4_clip if self.config else 1.0
                if self.config and where == "prefix":
                    clip = self.config.nvfp4_clip_sites.get(key, clip)
                s.finish_calibration(clip, consumer=consumer)
        self._invalidate_graphs()
        return {"static_sites": len(sites)}

    def calibration_state(self) -> dict:
        """The frozen static parameters, to be saved and reloaded without recalibrating."""
        from .fp8 import Fp8Linear
        st = {}
        for where, (i, k), s in self._static_sites():
            t = s.a_scale if isinstance(s, Fp8Linear) else s.group_amax
            st[f"{where}.{i}.{k}"] = t.detach().cpu().clone()
        return st

    def load_calibration_state(self, state: dict) -> None:
        from .fp8 import Fp8Linear
        for where, (i, k), s in self._static_sites():
            t = state[f"{where}.{i}.{k}"]
            (s.a_scale if isinstance(s, Fp8Linear) else s.group_amax).copy_(t)
            s.calibrating, s.calibrated = False, True
        self._invalidate_graphs()

    # ------------------------------------------------------------------ execution
    def _vision(self, stream: int) -> None:
        if self.raw_mode:
            v = self.vision
            v.ext.resize_norm(self.raw_u8.data_ptr(), v.pixels.data_ptr(), v.nv,
                              self.raw_u8.shape[1], self.raw_u8.shape[2], v.img, v.img, stream)
        self.vision.forward(stream)

    def _run(self, stream: int) -> None:
        if self.host_staged:
            self.raw_u8.copy_(self.host_in, non_blocking=True)      # H2D nodes in the graph
            self.prefix.upload_staged()
        self._vision(stream)
        self.prefix.forward(self.vision.out, stream)
        self.action.forward(stream)
        if self.host_staged:
            self.host_out.copy_(self.action.x_t, non_blocking=True)  # D2H node in the graph
