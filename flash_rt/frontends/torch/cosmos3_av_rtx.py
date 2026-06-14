"""Cosmos3-Nano AV FP4 inverse-dynamics denoise — RTX SM120 torch frontend.

class Cosmos3AvTorchFrontendRtx (per docs/adding_new_model.md §0 rule 2).

Cosmos is a novel architecture (two-tower MoT denoise, not a Paligemma VLA), so it
self-loads + NVFP4-quantizes its weights in the model __init__ (the guide permits a
hand-written loader for novel backbones) rather than a declarative WEIGHT_SPEC. The
proven V13 compute path lives in ..models.cosmos3_av._impl; the model-local kernels
(fp4-direct FFN, GQA sage, fused qk-norm-rope) come from the precompiled extension.

Deliverable baseline (RTX 5090): E2E 10-step denoise ~1784 ms, action rel_l2 2.458%
(<3% gate), cos 0.99971 vs the official once/final_action.

Inputs come from the official AV reference dump (VAE-encoded vision tokens, text, rope
tables, initial action latent). Paths are env/arg-driven — no host paths baked in.
"""
import os
import time


class Cosmos3AvTorchFrontendRtx:
    def __init__(self, checkpoint, ref=None, num_views=2, autotune=3,
                 shift=5.0, wquant="mse", fp4_norm=16.0, sage_lo=11,
                 active_dims=9, **_ignored):
        # checkpoint = Cosmos3 weights safetensors; ref = official AV ref dump
        # (tensors.safetensors). The _impl model reads these via env at import,
        # so set them BEFORE importing the model chain.
        ref = ref or os.environ.get("COSMOS_R")
        if not ref:
            raise ValueError(
                "Cosmos3-AV needs the official reference dump: pass ref=<.../"
                "tensors.safetensors> or set COSMOS_R.")
        os.environ["COSMOS_W"] = checkpoint
        os.environ["COSMOS_R"] = ref
        os.environ.setdefault("COSMOS_WQUANT", wquant)
        os.environ["COSMOS_FP4_NORM"] = str(fp4_norm)
        os.environ["COSMOS_SAGE_LO"] = str(sage_lo)
        os.environ.setdefault("COSMOS_SHIFT", str(shift))

        import torch
        from safetensors import safe_open
        # Import the proven model chain AFTER env is set (the _impl modules read
        # COSMOS_W/COSMOS_R at import time for self-load + shape inference).
        from ...models.cosmos3_av._impl import cosmos_v2 as v2
        from ...models.cosmos3_av._impl.cosmos_v13 import CosmosV13
        from ...models.cosmos3_av._impl.fm_solvers_unipc import (
            FlowUniPCMultistepScheduler)
        self._torch = torch
        self.shift = float(os.environ.get("COSMOS_SHIFT", shift))
        self.active_dims = active_dims

        self._v2 = v2
        self.m = CosmosV13(bf16_projs=())
        self._rf = safe_open(v2.R, "pt", device=v2.DEV)
        self._sched_cls = FlowUniPCMultistepScheduler
        self._te_table = v2.te_table
        self._n_steps = sum(1 for k in self._rf.keys()
                            if k.endswith("/timesteps_in"))
        # precompute timestep embeds (outside graph) + capture the per-step graph
        r = lambda k: self._rf.get_tensor(k)
        self._tes = self._te_table(
            [r(f"s{st:02d}/timesteps_in") for st in range(self._n_steps)], self.m)
        AC, AL, BF = self._v2.AC, self._v2.AL, self._v2.BF
        self.m.set_input(r("s00/action2llm_in")[AC:AL], self._tes[0].to(BF))
        self.m.capture()
        torch.cuda.synchronize()
        self._last_latency_ms = None

    def set_prompt(self, prompt=None, **_):
        """Cosmos conditioning is fixed by the reference dump; no text prompt."""
        return None

    def _denoise(self):
        torch = self._torch
        v2 = self._v2
        r = lambda k: self._rf.get_tensor(k)
        AC, AL, BF, DEV = v2.AC, v2.AL, v2.BF, v2.DEV
        sched = self._sched_cls(num_train_timesteps=1000, shift=1.0,
                                use_dynamic_shifting=False)
        sched.set_timesteps(self._n_steps, device=DEV, shift=self.shift)
        lat = r("s00/action2llm_in")[AC:AL].float()
        t0 = time.perf_counter()
        for i, t in enumerate(sched.timesteps):
            self.m.set_input(lat.to(BF), self._tes[i].to(BF))
            v = self.m.replay().clone().float()
            v[:, self.active_dims:] = 0
            lat = sched.step(v, t, lat, return_dict=True).prev_sample
        torch.cuda.synchronize()
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return lat

    def infer(self, obs=None):
        """Run the 10-step UniPC denoise on the reference conditioning.

        Returns {'actions': [T, action_dim] tensor, 'latency_ms': float}. With
        obs={'compare_ref': True} also returns rel_l2/cos vs once/final_action.
        """
        lat = self._denoise()
        out = {"actions": lat, "latency_ms": self._last_latency_ms}
        if isinstance(obs, dict) and obs.get("compare_ref"):
            import torch.nn.functional as F
            ref = self._rf.get_tensor("once/final_action").float()
            a, b = lat[:, :self.active_dims].flatten(), ref[:, :self.active_dims].flatten()
            out["rel_l2"] = ((a - b).norm() / b.norm()).item()
            out["cos"] = F.cosine_similarity(a, b, 0).item()
        return out

    def get_latency_stats(self):
        return {"denoise_loop_ms": self._last_latency_ms}
