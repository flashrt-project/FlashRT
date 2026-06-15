"""Cosmos3-Nano text2video FP8 denoise — RTX SM120 torch frontend.

class Cosmos3VideoTorchFrontendRtx (docs/adding_new_model.md §0 rule 2).

Cosmos is a novel architecture (two-tower MoT denoise), so it self-loads +
quantizes its weights (the guide permits a hand-written loader for novel
backbones). The compute path is models/cosmos3_video/pipeline_rtx.py, a
fully self-contained two-tower MoT denoise with its own model-local kernels.

Lifecycle (mirrors wan22_rtx / motus_rtx — typed parameters, no env knobs):
  __init__(checkpoint, use_fp8=...)  select precision, defer load to set_prompt
  set_prompt(ref=...)                load conditioning from the official ref
                                     dump, build + calibrate the towers, cache
                                     the static text K/V, capture the gen graph
  infer(teacache_skip=, shift=)      run the UniPC denoise; return the latent

Conditioning (text / VAE encode, rope tables, initial latent, timestep embeds)
comes from the official reference dump passed to set_prompt(); VAE decode to
frames is the downstream step.
"""
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ...models.cosmos3_video.pipeline_rtx import (
    CosmosVideo, patchify, unpatchify, BF, DEV)
from ...models.cosmos3_video.fm_solvers_unipc import (
    FlowUniPCMultistepScheduler)


class Cosmos3VideoTorchFrontendRtx:
    DEFAULT_SHIFT = 10.0
    DEFAULT_TEACACHE_SKIP = ""

    def __init__(self, checkpoint, num_views=2, autotune=3, use_fp8=True, **_):
        self.checkpoint = checkpoint
        self.num_views = num_views
        self.autotune = autotune
        self.quant = "fp8" if use_fp8 else "bf16"
        self.m = None
        self._rf = None
        self._last_latency_ms = None

    def set_prompt(self, ref=None):
        """Load conditioning from the official reference dump, build + calibrate the
        towers, and capture the per-step gen graph.

        ``ref`` is the official reference dump (tensors.safetensors): und/gen inputs,
        rope tables, per-step timestep embeds, and the initial vision latent.
        """
        if not ref:
            raise ValueError(
                "Cosmos3-video needs the official reference dump: "
                "set_prompt(ref=<.../tensors.safetensors>).")
        self._rf = safe_open(ref, "pt", device=DEV)
        r = self._rf.get_tensor

        und = r("once/und_in")
        nu, ng = und.shape[0], r("s00/gen_in").shape[0]
        self.m = CosmosVideo(self.checkpoint, nu, ng, quant=self.quant)
        self.m.set_rope(r("once/rope_und_cos"), r("once/rope_gen_cos"),
                        r("once/rope_und_sin"), r("once/rope_gen_sin"))
        self.m.precompute_und(und)

        self._n_steps = sum(1 for k in self._rf.keys()
                            if k.endswith("/timestep_emb"))
        fl = r("once/final_vision_latent__0")
        _, self._C, self._T, Hh, Ww = fl.shape
        self._p = 2
        self._h, self._w = Hh // self._p, Ww // self._p
        self._final_ref = fl

        # capture the per-step gen graph (text K/V already cached)
        self.m.embed_gen(r("s00/vae2llm_in"), r("s00/timestep_emb"))
        self.m.capture()
        torch.cuda.synchronize()

    def _parse_skip(self, teacache_skip):
        return {int(t) for t in str(teacache_skip).split(",")
                if t.strip().isdigit() and 0 < int(t) < self._n_steps - 1}

    def _denoise(self, *, teacache_skip, shift):
        r = self._rf.get_tensor
        C, T, h, w, p = self._C, self._T, self._h, self._w, self._p
        skip = self._parse_skip(teacache_skip)
        sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0,
                                            use_dynamic_shifting=False)
        sched.set_timesteps(self._n_steps, device=DEV, shift=shift)
        lat = unpatchify(r("s00/vae2llm_in"), C, T, h, w, p).float()
        cached = None
        t0 = time.perf_counter()
        for i, t in enumerate(sched.timesteps):
            if i in skip and cached is not None:
                vel_lat = cached
            else:
                pe = patchify(lat.to(BF), C, p)
                self.m.embed_gen(pe, r(f"s{i:02d}/timestep_emb"))
                vel = self.m.replay().clone()
                vel_lat = unpatchify(vel, C, T, h, w, p).float()
                cached = vel_lat
            lat = sched.step(vel_lat, t, lat, return_dict=True).prev_sample
        torch.cuda.synchronize()
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return lat

    def infer(self, *, teacache_skip=DEFAULT_TEACACHE_SKIP, shift=DEFAULT_SHIFT,
              compare_ref=False, return_metadata=False):
        """Run the UniPC denoise on the reference conditioning.

        Returns the denoised ``[1,C,T,H,W]`` vision latent tensor. With
        ``return_metadata=True`` returns a dict with the tensor, denoise latency,
        and (if ``compare_ref``) cos / rel_l2 vs the official reference latent.
        """
        if self.m is None:
            raise ValueError("set_prompt(ref=...) must be called before infer()")
        lat = self._denoise(teacache_skip=teacache_skip, shift=shift)
        if not return_metadata:
            return lat
        out = {"latent": lat, "latency_ms": self._last_latency_ms}
        if compare_ref:
            a = lat.flatten()
            b = self._final_ref.flatten().float().to(lat.device)
            out["rel_l2"] = ((a - b).norm() / b.norm()).item()
            out["cos"] = F.cosine_similarity(a, b, 0).item()
        return out

    def get_latency_stats(self):
        return {"denoise_loop_ms": self._last_latency_ms}
