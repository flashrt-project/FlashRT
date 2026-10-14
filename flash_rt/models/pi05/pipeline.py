"""End-to-end native DM05 predict: vision -> prefix cache -> 10-step action loop.

``NativeDM05.inference_action`` takes the same arguments as
``DM05ForConditionalGeneration.inference_action`` and returns actions of the same
shape and dtype, so it can stand in for it behind ``DM05Exp._predict``'s pre/post
processing. The whole predict (image resize/normalize when raw views are given, the
SigLIP tower, the Gemma3 prefix and all diffusion steps) replays as one CUDA graph per
prefix layout; every quantization parameter is static and fitted by ``calibrate``.
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
class NativeConfig:
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


class NativeDM05:
    def __init__(self, dm05, num_views: int = 2, max_prefix: int = 640, steps: int = 10,
                 chunk: int = 10, use_graph: bool = True):
        vlm = dm05.model.vlm.model
        self.dm05 = dm05
        self.image_token_id = int(dm05.config.vlm_config.image_token_index)
        self.vision = NativeVision(vlm.vision_tower, vlm.multi_modal_projector, num_views)
        self.prefix = NativePrefix(vlm.language_model, max_seq=max_prefix)
        self.action = NativeActionExpert(dm05, self.prefix, steps=steps, chunk=chunk)
        self.A = self.action.A
        self.config: NativeConfig | None = None
        # One CUDA graph per distinct prefix layout (length + segment table). Every
        # buffer the graph touches is preallocated, so a layout replays with fixed
        # shapes and addresses; the served prompt template yields a handful of lengths.
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.use_graph = use_graph
        self._graph_stream = torch.cuda.Stream()
        # raw uint8 views (native preprocessing): resized + normalized inside the graph
        self.raw_u8 = None
        self.raw_mode = False
        # host-staged predict (make_predict): pinned input/output buffers copied by the
        # graph itself, so a replay has no eager transfer around it
        self.host_in = None
        self.host_out = torch.empty(self.action.T, self.A, dtype=torch.float32, pin_memory=True)
        self._host_out_np = self.host_out.numpy()          # one view, reused: no per-request op
        self.host_staged = False
        self._prompt_key = None
        self._noise_seed = None

    # ------------------------------------------------------------------ configuration
    def configure(self, cfg: NativeConfig) -> None:
        """Build the quantized sites. Weights of quantized sites are released afterwards."""
        from ..flash_rt_attention import set_num_splits
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
        self.graphs.clear()

    def _static_sites(self):
        yield from (("prefix", k, s) for k, s in self.prefix.fp4.items())
        yield from (("vision", k, s) for k, s in self.vision.fp8.items())

    def calibrate(self, run_one, observations) -> dict:
        """Fit every static quantization parameter from representative real observations.

        run_one(obs) must drive one full predict through this pipeline. The pass runs
        eagerly with the activations quantized dynamically, so the statistics describe
        the distribution the quantized pipeline itself produces; afterwards the
        parameters are frozen and previously captured graphs are discarded.
        """
        if not observations:
            raise RuntimeError("calibration needs representative real observations")
        from .fp8 import Fp8Linear
        sites = list(self._static_sites())
        for _, _, s in sites:
            s.reset_calibration()
        use_graph, self.use_graph = self.use_graph, False
        try:
            for obs in observations:
                run_one(obs)
        finally:
            self.use_graph = use_graph
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
        self.graphs.clear()
        return {"static_sites": len(sites), "observations": len(observations)}

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
        self.graphs.clear()

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

    def _replay(self) -> None:
        key = (self.prefix.seq, tuple(self.prefix.segments), self.action.use_mask, self.raw_mode,
               self.host_staged)
        g = self.graphs.get(key)
        if g is None:
            s = self._graph_stream
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                # Warm-up outside capture: builds every cuBLASLt plan and the FA2 scratch
                # buffers so nothing is created or allocated while capturing.
                from . import fp8
                fp8.set_autotune(True)
                try:
                    self._run(s.cuda_stream)
                finally:
                    fp8.set_autotune(False)
                s.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    self._run(s.cuda_stream)
            torch.cuda.current_stream().wait_stream(s)
            self.graphs[key] = g
        g.replay()

    @torch.no_grad()
    def inference_action(self, input_ids=None, attention_mask=None, pixel_values=None,
                         token_type_ids=None, action_mask=None, history_pixel_values=None,
                         history_mask=None, **_):
        if history_mask is not None and bool(history_mask.any()):
            raise NotImplementedError("native DM05 path does not implement history images")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise NotImplementedError("native DM05 path expects unpadded single requests")
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.from_numpy(pixel_values)   # cached-tokenization hit: raw numpy views
        raw = pixel_values.dtype == torch.uint8
        if raw:
            pv = pixel_values.reshape(self.vision.nv, *pixel_values.shape[-3:])
            if self.raw_u8 is None or self.raw_u8.shape != pv.shape:
                self.raw_u8 = torch.empty(pv.shape, dtype=torch.uint8, device="cuda")
                self.graphs.clear()
            self.raw_u8.copy_(pv, non_blocking=True)
        else:
            self.vision.pixels.copy_(pixel_values.reshape(self.vision.pixels.shape))
        if raw != self.raw_mode or self.host_staged:
            self.raw_mode, self.host_staged = raw, False
            self.graphs.clear()
        self._prompt_key = None          # prefix state is rebound below; the staged path must re-upload
        self.prefix.staged = False
        self.prefix.set_prompt(input_ids, self.image_token_id,
                               segments_from_token_types(token_type_ids))
        # Same draw as the reference: after seeding, the first RNG use is this randn.
        noise = torch.randn(1, self.action.T, self.A, device="cuda", dtype=torch.bfloat16)
        self.action.set_noise(noise[0])
        am = None if action_mask is None else action_mask.reshape(-1)[: self.A].float()
        self.action.set_action_mask(am)
        if self.use_graph:
            self._replay()
        else:
            self._run(torch.cuda.current_stream().cuda_stream)
        return self.action.x_t[None].clone()

    # ------------------------------------------------------------------ host-staged predict
    def make_predict(self, ic, fallback):
        """A drop-in for ``DM05Exp._predict`` whose whole GPU side is one graph replay.

        Same host transforms (minus ``ToDevice``), seeding, action mask and output
        transform as the reference ``_predict``. The raw uint8 views go through a pinned
        buffer that the graph copies to the device; the prompt tensors are uploaded only
        when the prompt changes; under ``DM05_INFERENCE_SEED`` the per-request noise --
        the first RNG draw after reseeding, hence the same tensor on every request -- is
        drawn once per seed. Requests the native path does not cover (non-square views,
        history images) go to ``fallback``.
        """
        import os
        import time
        import numpy as np
        from .preprocess import RawAwareTokenization
        tr = ic.input_transform.transforms
        host_tr = [t for t in tr if type(t).__name__ != "ToDevice"]
        tok = next(t for t in tr if isinstance(t, RawAwareTokenization))
        action_dim = self.A
        mask = torch.ones(action_dim) if ic.full_action_mask else torch.zeros(action_dim)
        if not ic.full_action_mask:
            mask[: ic.output_action_dim] = 1.0
        mask = mask.to(torch.bfloat16).float().cuda()
        prompt_info: dict = {}

        def predict(data):
            ic.last_model_latency_sec = None
            original_state = data["state"]
            key = tok._key(data)
            x = data
            for t in host_tr:
                x = t(x)
            pv = x["pixel_values"]
            is_np = isinstance(pv, np.ndarray)
            if (not is_np and pv.dtype != torch.uint8) or x.get("history_mask") is not None:
                data["state"] = original_state
                return fallback(data)
            state = data["state"] if ic.use_transformed_state else original_state
            seed_env = os.environ.get("DM05_INFERENCE_SEED")
            if seed_env is not None:
                seed = int(seed_env)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                if self._noise_seed != seed:
                    self.action.set_noise(torch.randn(1, self.action.T, action_dim, device="cuda",
                                                      dtype=torch.bfloat16)[0])
                    self._noise_seed = seed
            else:
                self.action.set_noise(torch.randn(1, self.action.T, action_dim, device="cuda",
                                                  dtype=torch.bfloat16)[0])
                self._noise_seed = None
            self.prefix.staged = True
            if key != self._prompt_key:
                # per-prompt host data (ids as numpy, attention segments) is derived once and
                # reused, so even a prompt switch does no tensor op on the host
                info = prompt_info.get(key)
                if info is None:
                    info = (x["input_ids"].reshape(-1).numpy().copy(),
                            segments_from_token_types(x["token_type_ids"]))
                    prompt_info[key] = info
                self.prefix.set_prompt_host(info[0], self.image_token_id, info[1])
                self._prompt_key = key
            # (inference_action / fallback paths handle a torch pixel tensor as before)
            self.action.set_action_mask(mask)
            pv = (pv if is_np else pv.numpy()).reshape(self.vision.nv, *pv.shape[-3:])
            if self.host_in is None or tuple(self.host_in.shape) != pv.shape:
                self.host_in = torch.empty(pv.shape, dtype=torch.uint8, pin_memory=True)
                self.raw_u8 = torch.empty(pv.shape, dtype=torch.uint8, device="cuda")
                self._host_in_np = self.host_in.numpy()
                self.graphs.clear()
            np.copyto(self._host_in_np, pv)                  # host memcpy into the pinned buffer
            if not (self.raw_mode and self.host_staged):
                self.raw_mode = self.host_staged = True
                self.graphs.clear()
            t0 = time.perf_counter()
            self._replay()
            torch.cuda.current_stream().synchronize()
            ic.last_model_latency_sec = time.perf_counter() - t0
            raw_actions = self._host_out_np[:, : ic.output_action_dim].copy()
            out = ic.output_transform({"action": raw_actions, "state": state, "meta_data": data["meta_data"]})
            return out["action"]

        return predict
