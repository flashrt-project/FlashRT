"""FlashRT -- LaWAM torch frontend for RTX SM120.

Serves the LeRobot LaWAM checkpoint (``jialei02/lawam-pretrain-lerobot``):
two 256x256 RGB views + a task string -> a normalized-then-unnormalized
action chunk (``action_horizon`` x action dims), with the full policy step
(preprocessing, Qwen3-VL, QFormer, DINOv3, LAM decoder, flow conditioning,
all denoising steps, unnormalization and both host copies) replayed as one
CUDA graph. See :mod:`flash_rt.models.lawam.pipeline_rtx` for the kernel
sequence.

Precision (default): Qwen3-VL ViT / LLM, DINOv3, the LAM decoder and the
flow conditioning on FP16 GEMMs with FP16 accumulation; the flow DiT on
skinny weight-only INT8 GEMMs (per-channel scales, BF16 activations).
Optional tiers per stage: static per-tensor FP8 with a real-frame
calibration pass (cached on disk) and NVFP4 / BF16 DiT. Residual streams are
FP16 (vision / language / LAM) and BF16 (DiT).

Usage::

    fe = LawamTorchFrontendRtx("/path/to/lawam-pretrain-lerobot")
    fe.set_prompt("put the bowl on the plate")
    fe.calibrate([(frames, task), ...])     # FP8 tiers only: real frames, once; cached
    actions = fe.infer(frames)["actions"]   # frames uint8 [2, 256, 256, 3]
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib

import numpy as np
import torch

from flash_rt.models.lawam import pipeline_rtx as P

_FP16, _BF16, _F32, _U8 = torch.float16, torch.bfloat16, torch.float32, torch.uint8
_PREFIX = "model.policy_backend."
_IMAGE_TOKEN = 151655
_DEFAULT_PROMPTS = (
    "Your task is {instruction}.\n\nFrom main-view observations, infer the robot arm's motion.",
    "Then use observations and the inferred motion to produce the robot policy actions.",
)
# Skinny NVFP4 tile configuration per (N, K) at the action-row count.
# Coordinate-descent sweep of the captured 10-step loop on RTX 5060 Ti.
DIT_SKINNY_CFG = {
    (3 * 1024, 1024): 19,
    (1024, 1024): 11,
    (4096, 1024): 5,
    (1024, 4096): 3,
    (32, 1024): 9,
}


# Skinny W8A16 tile configuration (lawam.w8a16_skinny_gemm cfg) per (N, K).
DIT_W8A16_CFG = {
    (3 * 1024, 1024): 0,
    (1024, 1024): 0,
    (4096, 1024): 0,
    (1024, 4096): 2,
    (32, 1024): 0,
}


# Skinny W4A16 tile configuration (lawam.w4a16_skinny_gemm cfg) per (N, K).
DIT_W4A16_CFG = {
    (3 * 1024, 1024): 2,
    (1024, 1024): 2,
    (4096, 1024): 2,
    (1024, 4096): 2,
    (32, 1024): 0,
}

# Key slices (CTAs per head and query tile) of the cross-attention over the
# 512 image tokens (lawam.dit_cross_attention); 0 runs FA2's small-query tile.
DIT_IMG_KV_SLICES = 4


def _nvfp4_weight_only(w: torch.Tensor):
    """Round-to-nearest NVFP4 of w [N, K] (K % 16 == 0) for lawam.gemv_nvfp4w:
    E2M1 pairs [N, K / 2] (element 2j in the low nibble), UE4M3 block scales
    [N, K / 16] under one FP32 global scale (amax / (6 * 448))."""
    N, K = w.shape
    gs = (w.abs().amax() / (6.0 * 448.0)).clamp_min(1e-12)
    blocks = w.view(N, K // 16, 16)
    sf = (blocks.abs().amax(-1) / 6.0 / gs).clamp(max=448.0).to(torch.float8_e4m3fn)
    scale = sf.float().clamp_min(2.0 ** -9) * gs
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device)
    idx = torch.empty(N, K // 16, 16, dtype=torch.uint8, device=w.device)
    for r in range(0, N, 8192):   # bounded temporaries
        x = blocks[r:r + 8192] / scale[r:r + 8192, :, None]
        code = (x.abs()[..., None] - grid).abs().argmin(-1).to(torch.uint8)
        idx[r:r + 8192] = code | ((x < 0) & (code > 0)).to(torch.uint8) << 3
    idx = idx.view(N, K // 2, 2)
    packed = idx[..., 0] | (idx[..., 1] << 4)
    return packed.contiguous(), sf.view(torch.uint8).contiguous(), gs.reshape(1).float().contiguous()


def _ck(rc, what):
    if rc:
        raise RuntimeError(f"LaWAM {what} failed (status {rc})")


def _expand_sites(sites) -> set:
    """FP8 site tokens: "name" (every layer), "name@i" or "name@a-b" (those layers)."""
    out = set()
    for tok in sites:
        name, _, rng = str(tok).partition("@")
        if not rng:
            out.add(name)
            continue
        lo, _, hi = rng.partition("-")
        out.update(f"{name}@{i}" for i in range(int(lo), int(hi or lo) + 1))
    return out


def _site_on(sites, name, layer) -> bool:
    return name in sites or (layer is not None and f"{name}@{layer}" in sites)


#: One entry from every kernel group the pipeline calls. A stale
#: flash_rt_lawam (or one built from an older tree) fails here with a build
#: hint instead of an AttributeError midway through CUDA-graph capture.
_REQUIRED_LAWAM = (
    "graph_step", "memcpy_async", "memset_async",
    "preprocess_u8", "gated_residual_norm", "qformer_latent", "scatter_rows",
    "gemm_f16acc", "gemm_f16acc_act", "gemm_f16acc_persistent",
    "gemv_nvfp4w", "w8a16_skinny_gemm", "w4a16_skinny_gemm",
    "dit_flow_persistent", "dit_cross_attention",
    "dit_sum_bias_residual_norm", "qkv_rope_split", "qk_rope_inplace",
)


def _check_lawam_symbols(lk) -> None:
    missing = [name for name in _REQUIRED_LAWAM if not hasattr(lk, name)]
    if missing:
        raise RuntimeError(
            f"flash_rt_lawam is missing {missing}; rebuild it with "
            "-DGPU_ARCH=120 -DFLASHRT_ENABLE_LAWAM=ON")


class LawamTorchFrontendRtx:
    """LaWAM inference on RTX SM120 (one CUDA graph per prompt)."""

    def __init__(self, checkpoint_path: str, *, processor_path: str | None = None, device: str = "cuda:0",
                 num_views: int = 2, dit_cfg: dict | None = None, calibration_dir: str | None = None,
                 precision: dict | None = None, fp8_sites: dict | None = None):
        if int(num_views) != 2:
            raise ValueError("the LaWAM checkpoint is trained on one primary + one wrist view")
        # Check the model extension first: it is the one a default build is
        # most likely to be missing, and the message names the build flag.
        try:
            from flash_rt import flash_rt_lawam as lk
        except ImportError as e:
            raise RuntimeError("LaWAM needs flash_rt_lawam: build with -DGPU_ARCH=120 "
                               "-DFLASHRT_ENABLE_LAWAM=ON") from e
        _check_lawam_symbols(lk)
        import flash_rt.flash_rt_kernels as fvk
        from flash_rt import flash_rt_fa2 as fa2
        required_fa2 = ("lawam_fwd_fp16", "lawam_fwd_bf16", "lawam_fwd_bf16_fp8out",
                        "lawam_fwd_fp16_causal", "lawam_fwd_bf16_causal")
        missing = [name for name in required_fa2 if not hasattr(fa2, name)]
        if missing:
            raise RuntimeError(f"flash_rt_fa2 is missing {missing}; rebuild with "
                               "-DGPU_ARCH=120 -DFLASHRT_ENABLE_LAWAM=ON")
        self.fvk, self.fa2, self.lk = fvk, fa2, lk
        self.gemm = fvk.GemmRunner()
        self.device = torch.device(device)
        # DINOv3 depends on the camera frame only: it runs on a side branch of
        # the graph, concurrently with the Qwen3-VL towers (own GEMM runner,
        # cuBLASLt workspace and scratch).
        self.gemm_side = fvk.GemmRunner()
        self._side = torch.cuda.Stream(self.device)
        self._ev_frames = torch.cuda.Event()
        self._ev_dino = torch.cuda.Event()
        self._ev_embeds = torch.cuda.Event()
        # after the VLM: the LAM modulation GEMV (DRAM bound) of blocks 1..11 and the text
        # conditioning of the flow run on side branches, next to the compute-bound LAM blocks
        self.gemm_side2 = fvk.GemmRunner()
        self._side2 = torch.cuda.Stream(self.device)
        self._ev_cond = torch.cuda.Event()
        self._ev_q = torch.cuda.Event()
        # modulation launches of the side branch (block ranges; block li waits for its range)
        self._mod_groups = [(1, 2), (2, 12)]
        self._mod_ctas = 36
        self._ev_mod = [torch.cuda.Event() for _ in self._mod_groups]
        # the GEMMs of the DeepStack mergers (DRAM bound) on a side branch next to the ViT layers
        self._ev_tap = [torch.cuda.Event() for _ in range(3)]
        self._ev_ds = torch.cuda.Event()
        self._ds_side = True
        self.checkpoint_path = str(checkpoint_path)
        self.num_views = int(num_views)
        self.dit_cfg = dict(DIT_SKINNY_CFG if dit_cfg is None else dit_cfg)
        # Per-stage GEMM precision. Towers: "f16acc" (FP16 with FP16
        # accumulation), "fp16" (FP32 accumulation) or "fp8" (static per-tensor,
        # calibrated). DiT: "w8a16" (INT8 weight-only), "w4a16" (NVFP4
        # weight-only), "bf16" or "nvfp4"; "dit_ffn" overrides the FFN GEMMs:
        # "w4a16" (NVFP4 weight-only, BF16 activations) or "nvfp4" (W4A4),
        # "dit_qkv" the self-attention QKV projection ("w4a16"). LAM AdaLN
        # modulation GEMV: "nvfp4" (weight-only, per-16 block scales), "int8"
        # (per-channel weight-only) or "fp16". LLM
        # attention operands: "fp16" (no cast before the o projection) or "bf16"
        # (the reference's rounding; closest to the LeRobot outputs). Flow
        # loop: "persistent" (one cooperative kernel for all denoising steps,
        # the W8A16 / W4A16 tier with FP16 accumulation per 128 / 256-wide k
        # unit) or "chain" (the per-kernel chain of pipeline_rtx.dit_step_forward).
        # The flow's action chunks bifurcate under small numeric changes on a
        # few real frames; per-tensor FP8 anywhere upstream and NVFP4 DiT weights
        # push measurably past the reference's own precision noise floor, the
        # default stays at it.
        self.precision = {"vit": "f16acc", "llm": "f16acc", "dino": "f16acc", "lam": "f16acc", "enc": "f16acc",
                          "dit": "w8a16", "dit_ffn": "w4a16", "dit_qkv": "w4a16", "lam_mod": "nvfp4",
                          "llm_attn": "fp16", "dit_impl": "persistent"}
        self.precision.update(precision or {})
        # GEMM sites of a 16-bit tower that run static FP8 instead
        # (vit: qkv / o / fc1 / fc2, llm: qkv / o / gateup / down)
        # (a site may be limited to layers: "site@3" or "site@0-7"; expanded to "site@<layer>")
        self.fp8_sites = {k: frozenset(_expand_sites(v)) for k, v in (fp8_sites or {}).items()}
        if self.precision.get("llm_attn", "fp16") == "fp16" and not hasattr(self.fa2, "lawam_fwd_fp16_causal"):
            raise RuntimeError(
                "LaWAM needs flash_rt_fa2.lawam_fwd_fp16_causal (head_dim 128, fp16): "
                "rebuild flash_rt_fa2 with 128 in -DFA2_HDIMS and fp16 in "
                "-DFA2_DTYPES, or pass precision={'llm_attn': 'bf16'}")
        self._layer = None
        self.num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.dit_num_ctas = (P.select_dit_num_ctas(self.num_sms)
                             if self.precision["dit_impl"] == "persistent" else self.num_sms)
        ckpt = pathlib.Path(self.checkpoint_path)
        self.config = json.loads((ckpt / "config.json").read_text())
        cfg = self.config
        self.chunk = int(cfg["chunk_size"])
        self.horizon = int(cfg.get("action_horizon", cfg.get("n_action_steps", 24)))
        self.num_steps = int(cfg["flow_num_inference_steps"])
        self.embodiment = int(cfg["embodiment_id"])
        self.action_hz = 20.0
        self.n_act = int(cfg["num_action_queries"])
        self.n_flow = int(cfg["flow_action_num_queries"])
        self.action_dim = int(cfg["output_features"]["action"]["shape"][0])
        self.processor_path = processor_path or cfg["base_vlm"]
        self.placeholder = cfg.get("latent_action_placeholder_token", "<ACT_PH>")
        self.default_task = cfg.get("default_task", "Execute the robot action.")
        self.prompts = _DEFAULT_PROMPTS
        pre = ckpt / "policy_preprocessor.json"
        if pre.exists():
            for step in json.loads(pre.read_text()).get("steps", []):
                c = step.get("config", {})
                if "cot_prompt_before_wrist" in c:
                    self.prompts = (c["cot_prompt_before_wrist"], c["cot_prompt_after_wrist"])
                if "action_hz" in c:
                    self.action_hz = float(c["action_hz"])
        self.calibration_dir = pathlib.Path(
            calibration_dir or os.path.expanduser("~/.flash_rt/calibration"))
        self._keep: list = []
        self._scales = torch.zeros(1024, dtype=_F32, device=self.device)   # activation scale slots
        self._one_ptr = self._k(torch.ones(1, dtype=_F32, device=self.device))
        self._scale_names: list = []
        self._load_weights()
        self._build_step_tables()
        self._dit_persistent = self.precision["dit_impl"] == "persistent"
        if self._dit_persistent:
            if (self.horizon != 24 or self.precision["dit"] != "w8a16" or self.precision.get("dit_ffn") != "w4a16"
                    or self.precision.get("dit_qkv") != "w4a16"):
                raise ValueError("the persistent flow kernel runs the 24-row W8A16 / W4A16 DiT tier")
            stream, tab, units = P.pack_dit_persistent(torch, self.dit_w, self.dit_num_ctas, self.device)
            self._dit_stream = stream
            self._dit_tab = torch.from_numpy(tab).to(self.device)
            self._dit_units = torch.from_numpy(units).to(self.device)
        self.graph = None
        self.Se = None
        self.task = None
        # Static FP8 scales are only meaningful for an "fp8" stage or an
        # explicit fp8_sites selection; the default tier is calibration-free.
        self._needs_calibration = (any(v == "fp8" for v in self.precision.values())
                                   or any(self.fp8_sites.values()))
        self.calibrated = not self._needs_calibration

    # ── small helpers ───────────────────────────────────────────────
    def _k(self, t: torch.Tensor) -> int:
        t = t.contiguous()
        self._keep.append(t)
        return t.data_ptr()

    def _buf(self, *shape, dtype=_FP16, zero=False) -> torch.Tensor:
        t = (torch.zeros if zero else torch.empty)(*shape, dtype=dtype, device=self.device)
        self._keep.append(t)
        return t

    def _slot(self, name: str) -> int:
        """Device FP32 activation-scale slot (amax / 448) for one FP8 site."""
        i = len(self._scale_names)
        if i >= self._scales.numel():
            raise RuntimeError("out of activation scale slots")
        self._scale_names.append(name)
        self._scales[i] = 1.0
        return self._scales.data_ptr() + 4 * i

    def _get(self, key: str) -> torch.Tensor:
        return self._st.get_tensor(_PREFIX + key)

    def _fp8(self, w_nk: torch.Tensor) -> tuple:
        """nn.Linear weight [N, K] -> e4m3 [K, N] + device weight scale."""
        w = w_nk.to(self.device, _F32)
        s = (w.abs().amax() / 448.0).clamp_min(1e-12)
        q = (w / s).clamp(-448.0, 448.0).t().contiguous().to(torch.float8_e4m3fn).view(_U8)
        return self._k(q), self._k(s.reshape(1))

    def _lin(self, L: dict, name: str, w_nk: torch.Tensor, *, block: bool = True) -> None:
        """Store a GEMM weight in the precision of the stage being loaded:
        FP8 [K, N] + weight scale, or FP16 [K, N] ("fp16" stages). ``block``
        sites may be listed in ``fp8_sites``."""
        sites = self.fp8_sites.get(self._stage, ()) if block else ()
        fp8_site = any(_site_on(sites, n, self._layer) for n in ((name, "gateup") if name in ("gate", "up") else (name,)))
        if self.precision[self._stage] in ("fp16", "f16acc") and not fp8_site:
            L[name + "_w16"] = self._h(w_nk.to(self.device, _F32).t())
        else:
            L[name + "_w"], L[name + "_ws"] = self._fp8(w_nk)

    def _h(self, t: torch.Tensor) -> int:
        return self._k(t.to(self.device, _FP16))

    def _nvfp4(self, w_nk: torch.Tensor, bias: torch.Tensor | None, mse_steps: int = 4, *,
               mode: str | None = None) -> dict:
        """[N, K] DiT weight -> GEMM site in the DiT precision: packed skinny
        NVFP4 (w, sf, gs) or BF16 (w16, gs = 1), plus the bias."""
        b = torch.zeros(w_nk.shape[0]) if bias is None else bias
        mode = mode or self.precision["dit"]
        if mode == "w4a16":
            w4, sf, gs = _nvfp4_weight_only(w_nk.to(self.device, _F32))
            return {"w4": self._k(w4), "sf4": self._k(sf), "gs": self._k(gs), "w4_t": w4, "sf4_t": sf,
                    "N": w_nk.shape[0], "K": w_nk.shape[1],
                    "b": self._k(b.to(self.device, _BF16)), "b32": self._k(b.to(self.device, _F32))}
        if mode == "w8a16":
            w = w_nk.to(self.device, _F32)
            ws = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
            q = (w / ws[:, None]).round().clamp(-127, 127).to(torch.int8)
            return {"w8": self._k(q), "ws": self._k(ws), "gs": self._one_ptr, "w8_t": q,
                    "N": w_nk.shape[0], "K": w_nk.shape[1],
                    "b": self._k(b.to(self.device, _BF16)), "b32": self._k(b.to(self.device, _F32))}
        if mode == "bf16":
            return {"w16": self._k(w_nk.to(self.device, _BF16)), "gs": self._one_ptr,
                    "b": self._k(b.to(self.device, _BF16)), "b32": self._k(b.to(self.device, _F32))}
        fvk = self.fvk
        w = w_nk.to(self.device, _BF16).contiguous()
        N, K = w.shape
        gs = (w.float().abs().amax() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1).contiguous()
        q = torch.empty(N, K // 2, dtype=_U8, device=self.device)
        sf = torch.empty(N, K // 16, dtype=_U8, device=self.device)
        packed, sf_packed = torch.empty_like(q), torch.empty_like(sf)
        _ck(fvk.nvfp4_skinny_quantize_weight(w.data_ptr(), gs.data_ptr(), q.data_ptr(), sf.data_ptr(), N, K, 0,
                                             mse_steps), f"nvfp4 quantize [{N}, {K}]")
        _ck(fvk.nvfp4_skinny_pack_weight(q.data_ptr(), sf.data_ptr(), packed.data_ptr(), sf_packed.data_ptr(), N,
                                         K, 0), f"nvfp4 pack [{N}, {K}]")
        return {"w": self._k(packed), "sf": self._k(sf_packed), "gs": self._k(gs),
                "b": self._k(b.to(self.device, _BF16)), "b32": self._k(b.to(self.device, _F32))}

    # ── weights ─────────────────────────────────────────────────────
    def _load_weights(self) -> None:
        from safetensors import safe_open
        path = os.path.join(self.checkpoint_path, "model.safetensors")
        if self._uses_fp8():
            from flash_rt.models.lawam.cache_identity import file_sha256
            self._weight_fingerprint = file_sha256(path)
        with torch.no_grad(), safe_open(path, framework="pt", device=str(self.device)) as st:
            self._st = st
            self._load_qwen()
            self._load_dino()
            self._load_lam()
            self._load_flow()
            self._embed_tokens = st.get_tensor(
                _PREFIX + "vlm.model.language_model.embed_tokens.weight").to("cpu")
            self._act_query = self._get("act_query").float().cpu()
            self._flow_query = self._get("flow_action_query").float().cpu()
            del self._st
        from safetensors.torch import load_file
        stats = load_file(os.path.join(self.checkpoint_path,
                                       "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
        self._act_lo = self._k(stats["action.min"].float().to(self.device))
        self._act_hi = self._k(stats["action.max"].float().to(self.device))
        torch.cuda.synchronize()

    def _load_qwen(self) -> None:
        g = self._get
        self._stage = "vit"
        V = "vlm.model.visual."
        pe = g(V + "patch_embed.proj.weight").float()            # [1024, 3, 2, 16, 16]
        pe = pe.sum(dim=2).reshape(1024, 768)                     # still frame: fold the temporal pair
        self.vit = {"patch_w": self._h(pe.t()), "patch_b": self._h(g(V + "patch_embed.proj.bias")), "layers": []}
        self._vit_pos_table = g(V + "pos_embed.weight").float()
        for li in range(24):
            self._layer = li
            b = V + f"blocks.{li}."
            L = {"n1_w": self._h(g(b + "norm1.weight")), "n1_b": self._h(g(b + "norm1.bias")),
                 "n2_w": self._h(g(b + "norm2.weight")), "n2_b": self._h(g(b + "norm2.bias")),
                 "qkv_b": self._h(g(b + "attn.qkv.bias")), "o_b": self._h(g(b + "attn.proj.bias")),
                 "fc1_b": self._h(g(b + "mlp.linear_fc1.bias")), "fc2_b": self._h(g(b + "mlp.linear_fc2.bias"))}
            for n, key in (("qkv", "attn.qkv"), ("o", "attn.proj"), ("fc1", "mlp.linear_fc1"),
                           ("fc2", "mlp.linear_fc2")):
                self._lin(L, n, g(b + key + ".weight"))
            for n in ("qkv", "o", "fc1", "fc2"):
                L["a_" + n] = self._slot(f"vit{li}.{n}")
            self.vit["layers"].append(L)

        def merger(prefix, tag):
            M = {"norm_w": self._h(g(prefix + "norm.weight")), "norm_b": self._h(g(prefix + "norm.bias")),
                 "fc1_b": self._h(g(prefix + "linear_fc1.bias")),
                 "a_fc1": self._slot(tag + ".fc1"), "a_fc2": self._slot(tag + ".fc2")}
            self._lin(M, "fc1", g(prefix + "linear_fc1.weight"), block=False)
            self._lin(M, "fc2", g(prefix + "linear_fc2.weight"), block=False)
            return M, self._h(g(prefix + "linear_fc2.bias"))

        self.vit["ds"], self.ds_b = [], []
        for j in range(3):
            M, b = merger(V + f"deepstack_merger_list.{j}.", f"ds{j}")
            self.vit["ds"].append(M)
            self.ds_b.append(b)
        self.vit["merger"], self.merger_b = merger(V + "merger.", "merger")

        self._stage = "llm"
        T = "vlm.model.language_model."
        self.llm = {"layers": [], "norm_w": self._h(g(T + "norm.weight")), "a_enc": self._slot("enc_vlm")}
        for li in range(16):
            self._layer = li
            b = T + f"layers.{li}."
            L = {"in_ln_w": self._h(g(b + "input_layernorm.weight")),
                 "post_ln_w": self._h(g(b + "post_attention_layernorm.weight")),
                 "q_norm_w": self._h(g(b + "self_attn.q_norm.weight")),
                 "k_norm_w": self._h(g(b + "self_attn.k_norm.weight"))}
            qkv = torch.cat([g(b + f"self_attn.{n}_proj.weight") for n in ("q", "k", "v")], dim=0)
            self._lin(L, "qkv", qkv)
            self._lin(L, "o", g(b + "self_attn.o_proj.weight"))
            # gate and up share their input; FP8 keeps separate weight scales,
            # the 16-bit path runs them as one [K, 2 FF] GEMM
            if self.precision["llm"] in ("fp16", "f16acc") and not _site_on(self.fp8_sites.get("llm", ()), "gateup", li):
                gu = torch.cat([g(b + "mlp.gate_proj.weight"), g(b + "mlp.up_proj.weight")], 0)
                if self.precision["llm"] == "f16acc" and P.swiglu_interleaved(gu.shape[0] // 2, gu.shape[1]):
                    # the persistent GEMM's SwiGLU epilogue reads 8-column gate / up blocks
                    L["gateup_w16"] = self._h(P.interleave_gate_up(gu.to(self.device, _F32).t()))
                    L["gateup_il"] = True
                else:
                    self._lin(L, "gateup", gu)
            else:
                self._lin(L, "gate", g(b + "mlp.gate_proj.weight"))
                self._lin(L, "up", g(b + "mlp.up_proj.weight"))
            self._lin(L, "down", g(b + "mlp.down_proj.weight"))
            for n in ("qkv", "o", "gate", "down"):
                L["a_" + n] = self._slot(f"llm{li}.{n}")
            L["a_up"] = L["a_gate"]
            self.llm["layers"].append(L)

    def _load_dino(self) -> None:
        g = self._get
        self._stage = "dino"
        E = "lam.vision_encoder.model.embeddings."
        pre = torch.cat([g(E + "cls_token").reshape(1, 768), g(E + "register_tokens").reshape(4, 768)], dim=0)
        pw = g(E + "patch_embeddings.weight").float().reshape(768, 768)
        self.dino = {"prefix": self._h(pre), "patch_w": self._h(pw.t()),
                     "patch_b": self._h(g(E + "patch_embeddings.bias")), "layers": []}
        for li in range(P.DINO_LAYERS):
            b = f"lam.vision_encoder.model.model.layer.{li}."
            ls1, ls2 = g(b + "layer_scale1.lambda1").float(), g(b + "layer_scale2.lambda1").float()
            A = b + "attention."
            qkv = torch.cat([g(A + "q_proj.weight"), g(A + "k_proj.weight"), g(A + "v_proj.weight")], dim=0)
            qkv_b = torch.cat([g(A + "q_proj.bias"), torch.zeros_like(g(A + "q_proj.bias")), g(A + "v_proj.bias")])
            L = {"n1_w": self._h(g(b + "norm1.weight")), "n1_b": self._h(g(b + "norm1.bias")),
                 "n2_w": self._h(g(b + "norm2.weight")), "n2_b": self._h(g(b + "norm2.bias")),
                 "qkv_b": self._h(qkv_b),
                 # LayerScale folded into the projections that feed the residual
                 "o_b": self._h(g(A + "o_proj.bias").float() * ls1),
                 "fc1_b": self._h(g(b + "mlp.up_proj.bias")),
                 "fc2_b": self._h(g(b + "mlp.down_proj.bias").float() * ls2)}
            self._lin(L, "qkv", qkv)
            self._lin(L, "o", g(A + "o_proj.weight").float() * ls1[:, None])
            self._lin(L, "fc1", g(b + "mlp.up_proj.weight"))
            self._lin(L, "fc2", g(b + "mlp.down_proj.weight").float() * ls2[:, None])
            for n in ("qkv", "o", "fc1", "fc2"):
                L["a_" + n] = self._slot(f"dino{li}.{n}")
            self.dino["layers"].append(L)
        self.dino["a_pin"] = self._slot("lam.pin")
        # DINOv3 2D RoPE (patch rows; cls / register rows unrotated)
        n = 16
        c = (torch.arange(0.5, n, dtype=_F32) / n) * 2.0 - 1.0
        coords = torch.stack(torch.meshgrid(c, c, indexing="ij"), dim=-1).flatten(0, 1)    # [256, 2] (y, x)
        inv = 1.0 / (100.0 ** torch.arange(0, 1, 4 / 64, dtype=_F32))                      # [16]
        ang = (2 * math.pi * coords[:, :, None] * inv[None, None, :]).flatten(1, 2)        # [256, 32]
        cos = torch.ones(261, 64)
        sin = torch.zeros(261, 64)
        cos[5:] = torch.cat([ang.cos(), ang.cos()], dim=1)
        sin[5:] = torch.cat([ang.sin(), ang.sin()], dim=1)
        self.dino["cos"], self.dino["sin"] = self._h(cos), self._h(sin)

    def _load_lam(self) -> None:
        g = self._get
        self._stage = "lam"
        Dd = "lam.decoder."
        self.lam = {"layers": []}
        self._lin(self.lam, "pin", g(Dd + "project_input.weight"))
        self.lam["a_pin"] = self.dino["a_pin"]
        # 2D sin-cos positional table + project_input bias, added once to the block input
        H = W = 16
        dim = 1024
        hd, wd = dim // 2, dim - dim // 2

        def pe1d(d, length):
            pos = torch.arange(length, dtype=_F32).unsqueeze(1)
            div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
            pe = torch.zeros(length, d)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe

        ph = torch.nn.functional.pad(pe1d(hd, H)[:, None, :], (0, dim - hd))
        pw = torch.nn.functional.pad(pe1d(wd, W)[None, :, :], (0, dim - wd))
        pos = (ph + pw).reshape(H * W, dim) + g(Dd + "project_input.bias").float().cpu()
        self.lam["pos_b"] = self._h(pos)
        mod_w, mod_b = [], []
        for li in range(12):
            b = Dd + f"dec_layers.{li}."
            mod_w.append(g(b + "adaLN_modulation.1.weight"))
            mod_b.append(g(b + "adaLN_modulation.1.bias"))
            L = {"qkv_b": self._h(g(b + "attn.in_proj_bias")), "o_b": self._h(g(b + "attn.out_proj.bias")),
                 "fc1_b": self._h(g(b + "ffn.0.bias")), "fc2_b": self._h(g(b + "ffn.3.bias"))}
            self._lin(L, "qkv", g(b + "attn.in_proj_weight"))
            self._lin(L, "o", g(b + "attn.out_proj.weight"))
            self._lin(L, "fc1", g(b + "ffn.0.weight"))
            self._lin(L, "fc2", g(b + "ffn.3.weight"))
            for n in ("qkv", "o", "fc1", "fc2"):
                L["a_" + n] = self._slot(f"lam{li}.{n}")
            self.lam["layers"].append(L)
        # AdaLN modulation of all 12 blocks as one GEMV [12 * 6144, 1024]: NVFP4
        # weights ("lam_mod": "nvfp4", per-16 block scales), INT8 weights with
        # per-channel scales ("int8"), or FP16
        mw = torch.cat(mod_w, dim=0).to(self.device, _F32)
        if self.precision.get("lam_mod", "fp16") == "nvfp4":
            w4, sf, gs = _nvfp4_weight_only(mw)
            self.lam["mod_w4"], self.lam["mod_sf"], self.lam["mod_gs"] = self._k(w4), self._k(sf), self._k(gs)
        elif self.precision.get("lam_mod", "fp16") == "int8":
            ms = (mw.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
            self.lam["mod_w8"] = self._k((mw / ms[:, None]).round().clamp(-127, 127).to(torch.int8))
            self.lam["mod_ws"] = self._k(ms)
        else:
            self.lam["mod_w"] = self._h(mw.t())
        self.lam["mod_b"] = self._h(torch.cat(mod_b, dim=0))
        self._lin(self.lam, "pout", g(Dd + "project_output.weight"))
        self.lam["pout_b"] = self._h(g(Dd + "project_output.bias"))
        self.lam["a_pout"] = self._slot("lam.pout")
        self.lam["last_w"] = self._h(g(Dd + "last_ln.weight"))
        self.lam["last_b"] = self._h(g(Dd + "last_ln.bias"))
        # QFormer (FP32, tiny) + the decoder conditioning projection
        Q = "vlm_to_lam."
        f = lambda t: self._k(t.to(self.device, _F32))   # noqa: E731
        ft = lambda t: f(t.t().contiguous())                # noqa: E731  [in, out]: coalesced head projections
        self.qformer = {
            "query": f(g(Q + "query").reshape(-1)),
            "norm_q_w": f(g(Q + "norm_qs.0.weight")), "norm_q_b": f(g(Q + "norm_qs.0.bias")),
            "norm_kv_w": f(g(Q + "norm_kvs.0.weight")), "norm_kv_b": f(g(Q + "norm_kvs.0.bias")),
            "wq_t": ft(g(Q + "cross_attns.0.q_proj_weight")), "wk": f(g(Q + "cross_attns.0.k_proj_weight")),
            "wv": f(g(Q + "cross_attns.0.v_proj_weight")), "in_bias": f(g(Q + "cross_attns.0.in_proj_bias")),
            "wo_t": ft(g(Q + "cross_attns.0.out_proj.weight")), "bo": f(g(Q + "cross_attns.0.out_proj.bias")),
            "ffn_ln_w": f(g(Q + "ffns.0.0.weight")), "ffn_ln_b": f(g(Q + "ffns.0.0.bias")),
            "w1_t": ft(g(Q + "ffns.0.1.weight")), "b1": f(g(Q + "ffns.0.1.bias")),
            "w2_t": ft(g(Q + "ffns.0.4.weight")), "b2": f(g(Q + "ffns.0.4.bias")),
            "final_w": f(g(Q + "final_norm.weight")), "final_b": f(g(Q + "final_norm.bias")),
            "act_w_t": ft(g(Dd + "action_in_proj.weight")), "act_b": f(g(Dd + "action_in_proj.bias")),
        }

    def _load_flow(self) -> None:
        g = self._get
        self._stage = "enc"
        F = "flow."
        e = self.embodiment
        self.flow = {"enc_b": self._h(g(F + "enc_vlm.net.0.bias")),
                     "enc_ln_w": self._h(g(F + "enc_vlm.net.1.weight")),
                     "enc_ln_b": self._h(g(F + "enc_vlm.net.1.bias")), "a_enc": self.llm["a_enc"]}
        self._lin(self.flow, "enc", g(F + "enc_vlm.net.0.weight"))
        blk = lambda li, n: g(F + f"DiT.transformer_blocks.{li}." + n)   # noqa: E731
        for kind, off in (("text", 0), ("img", 2)):
            ws, bs = [], []
            for gi in range(4):
                li = 4 * gi + off
                ws += [blk(li, "attn1.to_k.weight"), blk(li, "attn1.to_v.weight")]
                bs += [blk(li, "attn1.to_k.bias"), blk(li, "attn1.to_v.bias")]
            self.flow[f"kv_{kind}_w"] = self._k(torch.cat(ws, dim=0).to(_BF16))      # [8192, 768]
            self.flow[f"kv_{kind}_b"] = self._k(torch.cat(bs, dim=0).to(_BF16))
            if self.precision["enc"] == "f16acc" and self.precision["lam"] != "fp8":
                # FP16 [768, 8192] for the FP16-accumulating projection (BF16 output); its DINOv3
                # rows are the LAM's FP16 input operand (not written by an FP8 LAM)
                self.flow[f"kv_{kind}_w16"] = self._h(torch.cat(ws, dim=0).to(self.device, _F32).t())
        W = {"qkv": [], "q": [], "o": [], "ff_up": [], "ff_down": []}
        for li in range(P.LAYERS_DIT):
            if li % 2 == 1:
                w = torch.cat([blk(li, f"attn1.to_{n}.weight") for n in ("q", "k", "v")], dim=0)
                b = torch.cat([blk(li, f"attn1.to_{n}.bias") for n in ("q", "k", "v")], dim=0)
                W["qkv"].append(self._nvfp4(w, b, mode=self.precision.get("dit_qkv")))
            else:
                W["q"].append(self._nvfp4(blk(li, "attn1.to_q.weight"), blk(li, "attn1.to_q.bias")))
            W["o"].append(self._nvfp4(blk(li, "attn1.to_out.0.weight"), blk(li, "attn1.to_out.0.bias")))
            ffm = self.precision.get("dit_ffn")
            W["ff_up"].append(self._nvfp4(blk(li, "ff.net.0.proj.weight"), blk(li, "ff.net.0.proj.bias"), mode=ffm))
            W["ff_down"].append(self._nvfp4(blk(li, "ff.net.2.weight"), blk(li, "ff.net.2.bias"), mode=ffm))
        W["ae_w1"] = self._k(g(F + "action_encoder.W1.W")[e].to(self.device, _F32))     # [32, 1024]
        W["ae_b1"] = self._k(g(F + "action_encoder.W1.b")[e].to(self.device, _F32))
        W["ae2"] = self._nvfp4(g(F + "action_encoder.W2.W")[e].t(), g(F + "action_encoder.W2.b")[e])
        # proj_out_2 and decoder layer 1 are consecutive linear maps: one GEMM
        wp2 = g(F + "DiT.proj_out_2.weight").double()
        bp2 = g(F + "DiT.proj_out_2.bias").double()
        wd1 = g(F + "action_decoder.layer1.W")[e].double()       # [in, out]
        bd1 = g(F + "action_decoder.layer1.b")[e].double()
        W["head1"] = self._nvfp4((wd1.t() @ wp2).float(), (bp2 @ wd1 + bd1).float())
        W["head2"] = self._nvfp4(g(F + "action_decoder.layer2.W")[e].t(), g(F + "action_decoder.layer2.b")[e])
        W["mode"] = self.precision["dit"]
        W["ffn_mode"] = self.precision.get("dit_ffn")
        W["w8_cfg"] = dict(DIT_W8A16_CFG)
        W["w4_cfg"] = dict(DIT_W4A16_CFG)
        W["img_slices"] = DIT_IMG_KV_SLICES
        self.dit_w = W
        # step-invariant conditioning precompute inputs
        self._t_lin1_w = g(F + "DiT.timestep_encoder.timestep_embedder.linear_1.weight").double()
        self._t_lin1_b = g(F + "DiT.timestep_encoder.timestep_embedder.linear_1.bias").double()
        self._t_lin2_w = g(F + "DiT.timestep_encoder.timestep_embedder.linear_2.weight").double()
        self._t_lin2_b = g(F + "DiT.timestep_encoder.timestep_embedder.linear_2.bias").double()
        self._ada_w = [blk(li, "norm1.linear.weight").double() for li in range(P.LAYERS_DIT)]
        self._ada_b = [blk(li, "norm1.linear.bias").double() for li in range(P.LAYERS_DIT)]
        self._out_w = g(F + "DiT.proj_out_1.weight").double()
        self._out_b = g(F + "DiT.proj_out_1.bias").double()

    def _build_step_tables(self) -> None:
        """AdaLN modulation of every DiT block and of the output norm for each
        denoising step (the timestep schedule is fixed), and the continuous-time
        positional embedding of the action rows."""
        D = P.D_DIT
        steps = self.num_steps
        T = {"scale": [], "shift": [], "out_scale": [], "out_shift": []}
        silu = torch.nn.functional.silu
        half = 128
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64,
                                                             device=self.device) / (half - 1))
        for s in range(steps):
            t_disc = int(s / float(steps) * 1000)
            t_disc = min(999, max(0, t_disc))
            arg = t_disc * freqs
            proj = torch.cat([torch.cos(arg), torch.sin(arg)])          # flip_sin_to_cos
            temb = self._t_lin2_w @ silu(self._t_lin1_w @ proj.float().double() + self._t_lin1_b) + self._t_lin2_b
            st = silu(temb)
            sc, sh = [], []
            for li in range(P.LAYERS_DIT):
                m = self._ada_w[li] @ st + self._ada_b[li]
                sc.append(self._k(m[:D].to(_BF16)))
                sh.append(self._k(m[D:].to(_BF16)))
            T["scale"].append(sc)
            T["shift"].append(sh)
            m = self._out_w @ st + self._out_b
            T["out_shift"].append(self._k(m[:D].to(_BF16)))
            T["out_scale"].append(self._k(m[D:].to(_BF16)))
        hz = self.action_hz
        t = torch.arange(self.horizon, dtype=torch.float64) / hz
        half = D // 2
        f = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64) / max(half - 1, 1))
        ang = t[:, None] * f[None, :]
        T["pos"] = self._k(torch.cat([torch.sin(ang), torch.cos(ang)], dim=1).to(self.device, _BF16))
        self.tables = T
        for n in ("_t_lin1_w", "_t_lin1_b", "_t_lin2_w", "_t_lin2_b", "_ada_w", "_ada_b", "_out_w", "_out_b"):
            delattr(self, n)

    # ── prompt ──────────────────────────────────────────────────────
    def _tokenize(self, task: str):
        from PIL import Image
        from transformers import AutoProcessor
        if not hasattr(self, "_processor"):
            proc = AutoProcessor.from_pretrained(self.processor_path)
            proc.tokenizer.add_special_tokens({"additional_special_tokens": [self.placeholder]})
            self._processor = proc
            self._ph_id = int(proc.tokenizer.convert_tokens_to_ids(self.placeholder))
        before, after = (p.replace("{instruction}", task) for p in self.prompts)
        act = " ".join([self.placeholder] * self.n_act)
        flow = " ".join([self.placeholder] * self.n_flow)
        img = Image.new("RGB", (256, 256))
        content = [{"type": "image", "image": img}, {"type": "text", "text": f"{before}\n{act}"},
                   {"type": "image", "image": img}, {"type": "text", "text": f"{after}\n{flow}"}]
        out = self._processor.apply_chat_template([[{"role": "user", "content": content}]], tokenize=True,
                                                  add_generation_prompt=True, return_dict=True,
                                                  processor_kwargs={"return_tensors": "pt"})
        return out["input_ids"][0], out["image_grid_thw"]

    def set_prompt(self, task: str | None = None, *, capture: bool = True) -> None:
        """Tokenize the instruction, build every prompt-scoped table and
        buffer, load (or keep) the calibrated scales and capture the graph."""
        from flash_rt.frontends.torch import _qwen3_vl_geometry as geo
        task = task or self.default_task
        ids, grid = self._tokenize(task)
        S = int(ids.numel())
        ph = (ids == self._ph_id).nonzero().flatten()
        if ph.numel() < self.n_act + self.n_flow:
            raise ValueError("prompt lost its latent placeholder tokens")
        act_rows, flow_rows = ph[:self.n_act], ph[self.n_act:self.n_act + self.n_flow]
        vis = (ids == _IMAGE_TOKEN).nonzero().flatten()
        if tuple(int(x) for x in grid.flatten()) != (1, 16, 16) * self.num_views:
            raise ValueError(f"unexpected image grid {grid.tolist()}")
        emb = self._embed_tokens[ids].float()
        emb[act_rows] = self._act_query
        emb[flow_rows] = self._flow_query
        emb[vis] = 0.0
        self.task, self.Se = task, S
        self._prompt_keep = []
        K = lambda t: (self._prompt_keep.append(t.contiguous()), self._prompt_keep[-1].data_ptr())[1]   # noqa
        self.llm["embeds"] = K(emb.to(self.device, _FP16))
        self.llm["vis_idx"] = K(vis.to(self.device, torch.int32))
        self.llm["merger_b"] = self.merger_b
        self.llm["ds_b"] = self.ds_b
        self.n_vis = int(vis.numel())
        self._act_idx = K(act_rows.to(self.device, torch.int32))
        # LaWAM runs the VLM on inputs_embeds alone (the query rows replace
        # placeholder embeddings), so Qwen3-VL cannot derive multimodal M-RoPE
        # ids and uses sequential positions on all three axes.
        pos = torch.arange(S).view(1, -1).expand(3, -1)
        cos, sin = geo.mrope_cos_sin(pos, head_dim=128, rope_theta=5000000.0, mrope_section=[24, 20, 20],
                                     device=self.device, dtype=_F32)
        self.llm["cos"] = K(torch.cat([cos, cos], dim=1).to(_FP16))
        self.llm["sin"] = K(torch.cat([sin, sin], dim=1).to(_FP16))
        vcos, vsin = geo.vision_rope_cos_sin(grid, head_dim=64, spatial_merge_size=2, device=self.device,
                                             dtype=_F32)
        self.vit["cos"] = K(torch.cat([vcos, vcos], dim=1).to(_FP16))
        self.vit["sin"] = K(torch.cat([vsin, vsin], dim=1).to(_FP16))
        self.vit["pos"] = K(geo.vision_pos_embeds(grid, self._vit_pos_table.to(self.device),
                                                  num_grid_per_side=48, spatial_merge_size=2, device=self.device,
                                                  dtype=_FP16))
        self._alloc(S)
        self.graph = None
        self._load_calibration()
        if capture and self.calibrated:
            self._capture()

    # ── buffers ─────────────────────────────────────────────────────
    def _alloc(self, S: int) -> None:
        b = lambda *s, dtype=_FP16, zero=False: (  # noqa: E731
            self._prompt_keep.append((torch.zeros if zero else torch.empty)(*s, dtype=dtype, device=self.device)),
            self._prompt_keep[-1].data_ptr())[1]
        nv = self.num_views
        Sv = 256 * nv
        f16_rows = max(Sv * 4096, S * 6144, 261 * 3072, 256 * 4096)
        f16 = b(f16_rows)
        self.frames_host = torch.empty(nv, 256, 256, 3, dtype=_U8).pin_memory()
        self.noise_host = torch.empty(self.horizon, 32, dtype=_F32).pin_memory()
        self.actions_host = torch.empty(self.horizon, self.action_dim, dtype=_F32).pin_memory()
        self._prompt_keep += [self.frames_host, self.noise_host, self.actions_host]
        # numpy views of the pinned staging buffers: the per-call host work is
        # plain memcpy into / out of them, no framework dispatch
        self._frames_np = self.frames_host.numpy()
        self._noise_np = self.noise_host.numpy()
        self._actions_np = self.actions_host.numpy()
        self.B = B = {}
        B["frames"] = b(nv, 256, 256, 3, dtype=_U8)
        B["vit"] = {"patch": b(Sv, 768), "h": b(Sv, 1024), "x8": b(Sv, 4096, dtype=_U8), "qkv": b(Sv, 3072),
                    "Q": b(Sv, 1024, dtype=_BF16), "K": b(Sv, 1024, dtype=_BF16), "V": b(Sv, 1024, dtype=_BF16),
                    "O": b(Sv, 1024, dtype=_BF16), "lse": b(nv, 16, 256, dtype=_F32),
                    "o_out": b(Sv, 1024), "fc1": b(Sv, 4096), "f16": f16,
                    "m_fp8": b(Sv, 4096, dtype=_U8), "m_fc1": b(Sv // 4, 4096), "m_f16": f16,
                    "ds_out": [b(Sv // 4, 2048) for _ in range(3)], "merger_out": b(Sv // 4, 2048),
                    # DeepStack mergers on the side branch: per-tap normalized input, shared fc1 output
                    "ds_in16": [b(Sv // 4, 4096) for _ in range(3)], "ds_fc1": b(Sv // 4, 4096)}
        nsplit = (S + 127) // 128
        B["llm"] = {"h": b(S, 2048), "x8": b(S, 6144, dtype=_U8), "qkv": b(S, 4096),
                    "Q": b(S, 2048, dtype=_BF16), "K": b(S, 1024, dtype=_BF16), "V": b(S, 1024, dtype=_BF16),
                    "O": b(S, 2048, dtype=_BF16), "lse": b(1, 16, nsplit * 128, dtype=_F32),
                    "lse_acc": b(nsplit, 1, 16, S, dtype=_F32), "o_acc": b(nsplit, 1, 16, S, 128, dtype=_F32),
                    "o_out": b(S, 2048), "gate": b(S, 2 * 6144), "up": b(S, 6144), "f16": f16,
                    "inject": [b(S, 2048, zero=True) for _ in range(3)],
                    "merger_out": B["vit"]["merger_out"], "ds_out": B["vit"]["ds_out"],
                    "vlm": b(S, 2048), "vlm8": b(S, 2048, dtype=_U8)}
        B["q"] = {"scratch": b(8 * 64, dtype=_F32), "latent": b(32, dtype=_F32), "cond_silu": b(1, 1024)}
        img_kv_in = b(512, 768, dtype=_BF16)
        img16 = b(512, 768)   # FP16 image tokens: DINOv3 latents (the LAM input) | LAM prediction
        B["dino"] = {"patch": b(256, 768), "h": b(261, 768), "x8": b(261, 3072, dtype=_U8), "qkv": b(261, 2304),
                     "Q": b(261, 768, dtype=_BF16), "K": b(261, 768, dtype=_BF16), "V": b(261, 768, dtype=_BF16),
                     "O": b(261, 768, dtype=_BF16), "lse": b(1, 12, 384, dtype=_F32), "o_out": b(261, 768),
                     "fc1": b(261, 3072), "f16": b(261 * 3072), "ht8": b(256, 768, dtype=_U8), "ht16": img16,
                     "img_kv_in": img_kv_in}
        B["lam"] = {"x": b(256, 1024), "x8": b(256, 4096, dtype=_U8), "qkv": b(256, 3072),
                    "Q": b(256, 1024, dtype=_BF16), "K": b(256, 1024, dtype=_BF16), "V": b(256, 1024, dtype=_BF16),
                    "O": b(256, 1024, dtype=_BF16), "lse": b(1, 16, 256, dtype=_F32), "o_out": b(256, 1024),
                    "fc1": b(256, 4096), "f16": f16, "mods": b(12 * 6144), "ht8": B["dino"]["ht8"],
                    "ht16": B["dino"]["ht16"],
                    "cond_silu": B["q"]["cond_silu"], "pout": b(256, 768), "img_kv_in": img_kv_in, "img16": img16}
        B["flow"] = {"vlm8": B["llm"]["vlm8"], "vlm": B["llm"]["vlm"], "enc_out": b(S, 768), "cond_vlm": b(S, 768, dtype=_BF16),
                     "cond16": b(S, 768), "img_kv_in": img_kv_in, "img16": img16, "kv_text": b(S, P.KV_LD, dtype=_BF16),
                     "kv_img": b(512, P.KV_LD, dtype=_BF16)}
        rows = self.horizon
        part_floats = max(k // self.fvk.nvfp4_skinny_config_k_chunk(c) * rows * n
                          for (n, k), c in self.dit_cfg.items())
        if self.precision["dit"] == "bf16":
            part_floats = rows * 4096
        elif self.precision["dit"] in ("w8a16", "w4a16"):
            part_floats = max(max(k // self.lk.w8a16_skinny_config_k_chunk(c) * rows * n
                                  for (n, k), c in DIT_W8A16_CFG.items()),
                              max(k // self.lk.w4a16_skinny_config_k_chunk(c) * rows * n
                                  for (n, k), c in DIT_W4A16_CFG.items()))
            if self.precision.get("dit_ffn") == "nvfp4":
                part_floats = max(part_floats, max(k // self.fvk.nvfp4_skinny_config_k_chunk(c) * rows * n
                                                   for (n, k), c in self.dit_cfg.items()))
        one = torch.ones(1, dtype=_F32, device=self.device)
        self._prompt_keep.append(one)
        B["dit"] = {"h": b(rows, 1024, dtype=_BF16), "xn": b(rows, 512, dtype=_U8), "xn_sf": b(rows, 64, dtype=_U8),
                    "hid": b(rows, 2048, dtype=_U8), "hid_sf": b(rows, 256, dtype=_U8),
                    "part": b(part_floats, dtype=_F32), "one": one.data_ptr(), "O": b(rows, 1024, dtype=_BF16),
                    "hmod": b(rows, 1024, dtype=_BF16), "ae": b(rows, 1024, dtype=_BF16),
                    "x": b(rows, 32, dtype=_F32), "q_bf16": b(rows, 1024, dtype=_BF16),
                    "xb": b(rows, 1024, dtype=_BF16), "hb": b(rows, 4096, dtype=_BF16),
                    "lse": b(1, 16, 128, dtype=_F32), "kv_text": B["flow"]["kv_text"], "kv_img": B["flow"]["kv_img"],
                    "xattn_state": b(self.lk.dit_cross_attention_state_floats(rows, 16, max(1, self.dit_w["img_slices"])),
                                     dtype=_F32),
                    "xattn_cnt": b(16 * ((rows + 15) // 16), dtype=torch.int32, zero=True)}
        B["actions"] = b(rows, self.action_dim, dtype=_F32)
        if self._dit_persistent:
            D = B["dit"]
            spad = (S + 31) // 32 * 32
            D.update({"Qs": b(32, 1024, dtype=_BF16, zero=True), "Ks": b(32, 1024, dtype=_BF16, zero=True),
                      "Vts": b(1024, 32, dtype=_BF16, zero=True), "vt_text": b(4, 1024, spad, dtype=_BF16, zero=True),
                      "vt_img": b(4, 1024, 512, dtype=_BF16, zero=True), "actions": B["actions"],
                      "counter": b(4, dtype=torch.int32, zero=True),
                      "part": b(P.DIT_KSPLIT_FC2, rows, 1024, dtype=_F32),
                      "rowstat": b(self.dit_num_ctas, rows, 2, dtype=_F32), "num_ctas": self.dit_num_ctas,
                      "h2": b(rows, 1024, dtype=_BF16),
                      "xa": b(rows, 1024, dtype=_FP16)})
            ops = P.dit_persistent_ops(self.dit_w, self.tables, D, {"S": S, "spad": spad}, num_steps=self.num_steps,
                                       lo=self._act_lo, hi=self._act_hi, action_dim=self.action_dim)
            self._dit_ops = torch.from_numpy(ops).to(self.device)
            self._prompt_keep.append(self._dit_ops)
        self.seed = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._prompt_keep.append(self.seed)

    # ── the policy step ─────────────────────────────────────────────
    def _forward(self, stream: int, *, cal: bool = False, host_io: bool = True, noise_from_host: bool = True):
        prec = {k: ("cal" if cal and v == "fp8" else v) for k, v in self.precision.items()
                if k not in ("dit", "dit_ffn", "dit_qkv", "lam_mod", "llm_attn", "dit_impl")}
        fvk, lk, fa2, gemm, B = self.fvk, self.lk, self.fa2, self.gemm, self.B
        st = int(stream)
        nv = self.num_views
        if host_io:
            _ck(lk.memcpy_async(B["frames"], self.frames_host.data_ptr(), self.frames_host.numel(), st), "frames h2d")
        _ck(lk.preprocess_u8(B["frames"], nv, B["vit"]["patch"], B["dino"]["patch"], st), "preprocess")
        main = torch.cuda.ExternalStream(st)
        self._ev_frames.record(main)
        self._side.wait_event(self._ev_frames)
        # off the critical path: the flow's initial sample (and the persistent kernel's barrier
        # counter) on the DINOv3 branch, which the LAM waits for; the flow runs after the LAM
        D, side = B["dit"], self._side.cuda_stream
        if noise_from_host:
            _ck(lk.memcpy_async(D["x"], self.noise_host.data_ptr(), self.noise_host.numel() * 4, side), "noise h2d")
        else:
            _ck(lk.gaussian_noise(D["x"], self.seed.data_ptr(), self.horizon * 32, side), "noise")
        if self._dit_persistent:
            _ck(lk.memset_async(D["counter"], 0, 4, side), "flow counter")
        # the LLM's prompt-template embeddings (static) into its residual stream
        fvk.gpu_copy(B["llm"]["h"], self.llm["embeds"], self.Se * 2048 * 2, side)
        self._ev_embeds.record(self._side)
        P.dino_forward(self.gemm_side, fvk, lk, fa2, B["dino"], self.dino, stream=self._side.cuda_stream,
                       prec=prec["dino"], out_prec=prec["lam"])
        # frame-only consumers of the DINOv3 latents: the LAM input projection and the
        # flow image K / V of those 256 tokens
        P.lam_input_forward(self.gemm_side, B["lam"], self.lam, stream=side, prec=prec["lam"])
        split_kv = "kv_text_w16" in self.flow
        if split_kv:
            P.flow_condition_img(self.gemm_side, lk, B["flow"], self.flow, 0, 256, stream=side)
        self._ev_dino.record(self._side)
        ds_side = None
        if self._ds_side and prec["vit"] in P._F16 and not self.fp8_sites.get("vit"):
            def fork(j):
                self._ev_tap[j].record(main)
                self._side2.wait_event(self._ev_tap[j])

            def join():
                self._ev_ds.record(self._side2)
                main.wait_event(self._ev_ds)

            ds_side = (self._side2.cuda_stream, fork, join)
        P.qwen_vit_forward(gemm, fvk, lk, fa2, B["vit"], self.vit, {"S": 256 * nv, "views": nv}, stream=st,
                           prec=prec["vit"], fp8_sites=self.fp8_sites.get("vit", ()), cal=cal, ds_side=ds_side)
        main.wait_event(self._ev_embeds)
        P.qwen_llm_forward(gemm, fvk, lk, fa2, B["llm"], self.llm, {"S": self.Se, "n_vis": self.n_vis}, stream=st,
                           prec=prec["llm"], num_sms=self.num_sms, out_prec=prec["enc"],
                           fp8_sites=self.fp8_sites.get("llm", ()), cal=cal,
                           attn_dtype=self.precision.get("llm_attn", "bf16"), embeds_ready=True)
        _ck(lk.qformer_latent(B["llm"]["vlm"], self._act_idx, self.n_act, 2048, 32, 8, 1024, self.qformer,
                              B["q"]["scratch"], B["q"]["latent"], B["q"]["cond_silu"], st), "qformer")
        # AdaLN modulation (DRAM bound): block 0 on the critical path, blocks 1..11 on a side branch
        # with a capped grid next to the compute-bound LAM blocks (each block waits for its rows only),
        # then the flow's text conditioning (VLM only) on the same branch
        self._ev_q.record(main)
        side2 = self._side2.cuda_stream
        self._side2.wait_event(self._ev_q)
        for i, (b0, b1) in enumerate(self._mod_groups):
            P.lam_modulation(self.gemm_side2, lk, B["lam"], self.lam, b0, b1, stream=side2, max_ctas=self._mod_ctas)
            self._ev_mod[i].record(self._side2)
        P.flow_condition_text(self.gemm_side2, lk, B["flow"], self.flow, {"S": self.Se}, stream=side2,
                              prec=prec["enc"])
        self._ev_cond.record(self._side2)
        P.lam_modulation(gemm, lk, B["lam"], self.lam, 0, self._mod_groups[0][0], stream=st)
        main.wait_event(self._ev_dino)
        mod_wait = {b0: i for i, (b0, _) in enumerate(self._mod_groups)}

        def before_block(li):
            if li in mod_wait:
                main.wait_event(self._ev_mod[mod_wait[li]])

        P.lam_decoder_forward(gemm, fvk, lk, fa2, B["lam"], self.lam, stream=st, prec=prec["lam"], inputs=False,
                              before_block=before_block)
        P.flow_condition_img(gemm, lk, B["flow"], self.flow, 256 if split_kv else 0, 512, stream=st)
        main.wait_event(self._ev_cond)
        if self._dit_persistent:
            _ck(lk.dit_flow_persistent(self._dit_ops.data_ptr(), self._dit_ops.shape[0], self._dit_tab.data_ptr(),
                                       self._dit_tab.shape[0], self._dit_units.data_ptr(), self.num_steps,
                                       self._dit_stream.data_ptr(), self._dit_stream.stride(0), D["counter"],
                                       self.dit_num_ctas, st, False), "flow (persistent)")
        else:
            dims = {"rows": self.horizon, "S": self.Se, "dt": 1.0 / self.num_steps}
            for s in range(self.num_steps):
                P.dit_step_forward(fvk, lk, fa2, D, self.dit_w, self.tables, dims, step=s, stream=st,
                                   cfg=self.dit_cfg)
            _ck(lk.unnormalize_actions(D["x"], 32, self._act_lo, self._act_hi, B["actions"], self.horizon,
                                       self.action_dim, st), "unnormalize")
        if host_io:
            _ck(lk.memcpy_async(self.actions_host.data_ptr(), B["actions"], self.actions_host.numel() * 4, st),
                "actions d2h")

    def _capture(self) -> None:
        """Capture the policy step twice: with the initial flow sample copied
        from the host (``infer(noise=...)``) and drawn on device (default)."""
        stream = torch.cuda.Stream(self.device)
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._forward(stream.cuda_stream)
        stream.synchronize()
        graphs = {}
        for host_noise in (True, False):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                g.capture_begin()
                self._forward(stream.cuda_stream, noise_from_host=host_noise)
                g.capture_end()
            stream.synchronize()
            graphs[host_noise] = g
        self.graph, self._graph_rng, self._graph_stream = graphs[True], graphs[False], stream

    # ── calibration ─────────────────────────────────────────────────
    def _uses_fp8(self):
        return any(v == "fp8" for v in self.precision.values()) or any(self.fp8_sites.values())

    def _calibration_path(self) -> pathlib.Path:
        """Versioned content identity of the actual checkpoint and processor."""
        from flash_rt.models.lawam.cache_identity import file_sha256, processor_sha256
        ckpt = pathlib.Path(self.checkpoint_path)
        weights = getattr(self, "_weight_fingerprint", None)
        if weights is None:
            weights = file_sha256(ckpt / "model.safetensors")
        if not hasattr(self, "_processor"):
            raise RuntimeError("LaWAM processor must be loaded before calibration cache lookup")
        identity = {
            "schema": 2, "weights": weights,
            "processor": processor_sha256(self._processor),
            "configs": {name: file_sha256(ckpt/name) if (ckpt/name).is_file() else None
                        for name in ("config.json", "policy_preprocessor.json")},
            "precision": self.precision,
            "sites": {k: sorted(v) for k, v in self.fp8_sites.items()},
            "capability": torch.cuda.get_device_capability(self.device),
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return self.calibration_dir / f"{digest}_lawam_v2.json"

    def _load_calibration(self) -> None:
        if self.calibrated:
            return
        path = self._calibration_path()
        if not path.exists():
            return
        data = json.loads(path.read_text())
        if data.get("sites") != self._scale_names:
            return
        n = len(self._scale_names)
        self._scales[:n] = torch.tensor(data["scales"], dtype=_F32, device=self.device)
        self.calibrated = True

    def calibrate(self, samples, *, percentile: float = 99.9, save: bool = True) -> dict:
        """Static FP8 activation scales from real observations.

        ``samples`` is a list of ``(frames, task)`` (frames uint8 [views, 256,
        256, 3]). One dynamic-scale pass per sample records amax / 448 at every
        FP8 site; each site's static scale is the ``percentile`` of its amax
        across samples (every denoising step shares the DiT-free sites, and the
        NVFP4 DiT uses dynamic block scales with no calibration). The scales do
        not depend on the prompt length, so one calibration serves every task.
        The current prompt's graph is (re)captured afterwards."""
        if not self._needs_calibration:
            return {}   # no FP8 site: the default tier needs no calibration pass
        from flash_rt.core.calibration import accumulate_amax
        n = len(self._scale_names)
        current = self.task
        per = []
        stream = torch.cuda.Stream(self.device)
        for frames, task in samples:
            if task != self.task:
                self.set_prompt(task, capture=False)
            self.frames_host.copy_(torch.as_tensor(np.asarray(frames, dtype=np.uint8)))
            self.noise_host.normal_()
            with torch.cuda.stream(stream):
                self._forward(stream.cuda_stream, cal=True)
            stream.synchronize()
            per.append(self._scales[:n].double().cpu().numpy() * 448.0)
        amax = accumulate_amax(per, percentile=percentile)
        scales = np.maximum(amax / 448.0, 1e-12).astype(np.float32)
        self._scales[:n] = torch.from_numpy(scales).to(self.device)
        self.calibrated = True
        if save:
            path = self._calibration_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"sites": self._scale_names, "scales": scales.tolist(),
                                        "samples": len(samples), "percentile": percentile,
                                        "precision": self.precision}))
        if current is not None and current != self.task:
            self.set_prompt(current, capture=False)
        self._capture()
        return {name: float(s) for name, s in zip(self._scale_names, scales)}

    # ── inference ───────────────────────────────────────────────────
    def _frames(self, obs) -> np.ndarray:
        """uint8 [views, 256, 256, 3] (primary first) from an observation dict
        (``images`` list or ``image`` / ``wrist_image``) or an array."""
        if isinstance(obs, dict):
            views = obs.get("images") or [obs["image"], obs["wrist_image"]]
            frames = np.stack([np.asarray(v) for v in views[:self.num_views]])
        else:
            frames = np.asarray(obs)
        if frames.shape != (self.num_views, 256, 256, 3) or frames.dtype != np.uint8:
            raise ValueError(f"LaWAM expects uint8 frames [{self.num_views}, 256, 256, 3] (primary, wrist), "
                             f"got {frames.dtype} {tuple(frames.shape)}")
        return frames

    def calibrate_with_real_data(self, obs_list, *, percentile: float = 99.9) -> dict:
        """calibrate() on observations under the current prompt."""
        return self.calibrate([(self._frames(o), self.task) for o in obs_list], percentile=percentile)

    def infer(self, obs, *, noise=None) -> dict:
        """One policy step. ``obs``: observation dict or uint8 frames [views,
        256, 256, 3]; ``noise``: optional FP32 [horizon (or chunk), 32]
        initial flow sample (rows past the horizon are ignored, as in the
        reference), drawn from N(0, 1) on device otherwise. Returns
        ``{"actions": [horizon, action_dim]}`` (unnormalized)."""
        if self.graph is None:
            raise RuntimeError("set_prompt() and calibrate() before infer()")
        frames = np.ascontiguousarray(self._frames(obs))
        out = np.empty_like(self._actions_np)
        if noise is None:
            graph, nz, nz_bytes = self._graph_rng, None, 0
        else:
            nz = np.ascontiguousarray(np.asarray(noise, dtype=np.float32).reshape(-1, 32)[:self.horizon])
            if nz.shape != self._noise_np.shape:
                raise ValueError(f"noise must have at least {self.horizon} rows of 32")
            graph, nz_bytes = self.graph, nz.nbytes
        # one native call: stage the host inputs, replay the graph, wait, copy the actions out
        _ck(self.lk.graph_step(graph.raw_cuda_graph_exec(), self._graph_stream.cuda_stream, frames.ctypes.data,
                               self.frames_host.data_ptr(), frames.nbytes, 0 if nz is None else nz.ctypes.data,
                               self.noise_host.data_ptr(), nz_bytes, self.actions_host.data_ptr(), out.ctypes.data,
                               out.nbytes), "policy step")
        return {"actions": out}
