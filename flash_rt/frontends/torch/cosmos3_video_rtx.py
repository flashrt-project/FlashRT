"""Cosmos3-Nano text2video FP8 denoise — RTX SM120 torch frontend.

class Cosmos3VideoTorchFrontendRtx (per docs/adding_new_model.md §0 rule 2).

Cosmos is a novel architecture (two-tower MoT denoise), so it self-loads + quantizes
its weights in the model __init__ (the guide permits a hand-written loader for novel
backbones). The proven compute path lives in ..models.cosmos3_av._impl.cosmos_video;
model-local kernels (fp4-direct FFN, fused qk-norm+rope, GQA sage) come from the
precompiled extension.

Pipeline: 10-step UniPC denoise of the all-noisy vision latent through the gen tower
(text tower computed once and cached), FP8 GEMMs by default (near-lossless for the
video latent; fp4 is lossy here), optional TeaCache step caching. infer() returns the
denoised vision latent; VAE decode to frames is the downstream step (the Wan VAE +
its fp4/fp8 conv acceleration are upstream/downstream of this denoise policy).

Inputs (text/VAE-encode conditioning, rope tables, initial latent, timestep embeds)
come from the official reference dump. Paths are env/arg-driven — no host paths baked
in.
"""
import os
import time


class Cosmos3VideoTorchFrontendRtx:
    def __init__(self, checkpoint, ref=None, quant="fp8", shift=10.0,
                 teacache_skip="", **_ignored):
        ref = ref or os.environ.get("COSMOS_R")
        if not ref:
            raise ValueError(
                "Cosmos3-video needs the official reference dump: pass "
                "ref=<.../tensors.safetensors> or set COSMOS_R.")
        os.environ["COSMOS_W"] = checkpoint
        os.environ["COSMOS_R"] = ref
        os.environ.setdefault("COSMOS_VIDEO_QUANT", quant)

        import torch
        from safetensors import safe_open
        from ...models.cosmos3_av._impl import cosmos_video as cv
        from ...models.cosmos3_av._impl.fm_solvers_unipc import (
            FlowUniPCMultistepScheduler)
        self._torch = torch
        self._cv = cv
        self._sched_cls = FlowUniPCMultistepScheduler
        self.shift = float(os.environ.get("COSMOS_SHIFT", shift))

        self._rf = safe_open(ref, "pt", device=cv.DEV)
        r = self._rf.get_tensor
        und = r("once/und_in")
        nu, ng = und.shape[0], r("s00/gen_in").shape[0]
        self.m = cv.CosmosVideo(nu, ng, quant=os.environ["COSMOS_VIDEO_QUANT"])
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

        skip = teacache_skip or os.environ.get("COSMOS_VIDEO_TEACACHE_SKIP", "")
        self._skip = {int(t) for t in str(skip).split(",")
                      if t.strip().isdigit() and 0 < int(t) < self._n_steps - 1}

        # capture the per-step gen graph (text K/V already cached)
        self.m.embed_gen(r("s00/vae2llm_in"), r("s00/timestep_emb"))
        self.m.capture()
        torch.cuda.synchronize()
        self._last_latency_ms = None

    def set_prompt(self, prompt=None, **_):
        """Conditioning is fixed by the reference dump; no text prompt here."""
        return None

    def _denoise(self):
        torch = self._torch
        cv = self._cv
        r = self._rf.get_tensor
        BF = cv.BF
        C, T, h, w, p = self._C, self._T, self._h, self._w, self._p
        sched = self._sched_cls(num_train_timesteps=1000, shift=1.0,
                                use_dynamic_shifting=False)
        sched.set_timesteps(self._n_steps, device=cv.DEV, shift=self.shift)
        lat = cv.unpatchify(r("s00/vae2llm_in"), C, T, h, w, p).float()
        cached = None
        t0 = time.perf_counter()
        for i, t in enumerate(sched.timesteps):
            if i in self._skip and cached is not None:
                vel_lat = cached
            else:
                pe = cv.patchify(lat.to(BF), C, p)
                self.m.embed_gen(pe, r(f"s{i:02d}/timestep_emb"))
                vel = self.m.replay().clone()
                vel_lat = cv.unpatchify(vel, C, T, h, w, p).float()
                cached = vel_lat
            lat = sched.step(vel_lat, t, lat, return_dict=True).prev_sample
        torch.cuda.synchronize()
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return lat

    def infer(self, obs=None):
        """Run the UniPC denoise on the reference conditioning.

        Returns {'latent': [1,C,T,H,W] vision latent, 'latency_ms': float}. With
        obs={'compare_ref': True} also returns rel_l2/cos vs the official
        once/final_vision_latent.
        """
        lat = self._denoise()
        out = {"latent": lat, "latency_ms": self._last_latency_ms}
        if isinstance(obs, dict) and obs.get("compare_ref"):
            import torch.nn.functional as F
            a = lat.flatten()
            b = self._final_ref.flatten().float().to(lat.device)
            out["rel_l2"] = ((a - b).norm() / b.norm()).item()
            out["cos"] = F.cosine_similarity(a, b, 0).item()
        return out

    def get_latency_stats(self):
        return {"denoise_loop_ms": self._last_latency_ms}
