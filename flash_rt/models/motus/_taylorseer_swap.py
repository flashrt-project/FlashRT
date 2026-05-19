"""TaylorSeer (final-velocity Taylor forecast) swap for Motus.

Unlike TeaCache (zeroth-order reuse: v(t_skip) = v(t_last_compute)),
TaylorSeer order-1 linearly extrapolates from the last two computed
velocities:

    v_pred(t) = v_curr + ((t - t_curr) / (t_curr - t_prev)) * (v_curr - v_prev)

For the flow-matching velocity in Motus the time-derivative is smooth
within the middle of the schedule, so order-1 is the right place to
start. The BAGEL ablation (flash_wm/REALITY_CHECK.md §3.11) shows
higher orders amplify finite-difference noise on this kind of velocity.

The schedule (compute vs skip steps) is identical in shape to TeaCache
so the two methods can be ablated apples-to-apples. The only difference
is what tensor gets passed to ``euler_step`` on a skip.

Env gates:
    FLASH_RT_MOTUS_USE_TAYLORSEER=1
    FLASH_RT_MOTUS_TAYLORSEER_SKIP_STEPS=2,3,4,5,6,7,8   (default)
    FLASH_RT_MOTUS_TAYLORSEER_ORDER=1                    (0|1)
"""

from __future__ import annotations

import os
import types
from typing import Dict, List, Set

import torch


def _parse_skip_steps(env_val: str, num_steps: int) -> Set[int]:
    if not env_val:
        return set()
    out = set()
    for tok in env_val.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            continue
        if 0 <= v < num_steps:
            out.add(v)
    out.discard(0)
    out.discard(num_steps - 1)
    return out


def _build_extrap_coeffs(
    num_steps: int,
    skip_steps: Set[int],
) -> Dict[int, float]:
    """Pre-compute the order-1 extrapolation coefficient for every
    skip step against the two most-recent prior compute steps, using
    the canonical linspace(1.0, 0.0, num_steps+1) schedule.

    coeff = (t_skip - t_recent) / (t_recent - t_prev)
    """
    timesteps_host: List[float] = [
        1.0 - i / float(num_steps) for i in range(num_steps + 1)
    ]
    compute_steps = sorted(set(range(num_steps)) - skip_steps)
    out: Dict[int, float] = {}
    for i in range(num_steps):
        if i not in skip_steps:
            continue
        prior_computes = [c for c in compute_steps if c < i]
        if len(prior_computes) < 2:
            continue
        recent = prior_computes[-1]
        prev = prior_computes[-2]
        denom = timesteps_host[recent] - timesteps_host[prev]
        if abs(denom) < 1e-12:
            continue
        out[i] = (timesteps_host[i] - timesteps_host[recent]) / denom
    return out


def _make_taylorseer_run(
    skip_steps: Set[int],
    order: int,
    extrap_coeffs: Dict[int, float],
):
    def run(self, first_frame, state, t5_embeds, vlm_inputs):
        cond_latent = self.encode_first_frame(first_frame)
        t5_ctx = self.preprocess_t5_context(t5_embeds)
        B = cond_latent.shape[0]

        video_latent = self.init_video_latent(cond_latent)
        action_latent = self.init_action_latent(B)

        use_cached_t = (
            getattr(self, "_cached_v_t_emb", None) is not None
            and getattr(self, "_cached_a_t_emb", None) is not None
            and getattr(self, "_cached_time_steps", None) is not None
        )
        use_cached_mod = (
            use_cached_t
            and getattr(self, "_cached_v_mod", None) is not None
            and getattr(self, "_cached_a_mod", None) is not None
        )
        if use_cached_t:
            timesteps = self._cached_time_steps
        else:
            timesteps = torch.linspace(
                1.0, 0.0, self.dims.num_inference_steps + 1,
                device=self.device, dtype=self.dtype)

        v_curr = getattr(self, "_taylor_v_curr", None)
        v_prev = getattr(self, "_taylor_v_prev", None)
        a_curr = getattr(self, "_taylor_a_curr", None)
        a_prev = getattr(self, "_taylor_a_prev", None)
        compute_count = 0

        num_steps = self.dims.num_inference_steps

        for i in range(num_steps):
            self._current_step_idx = i
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt_host = None
            if (use_cached_t
                    and getattr(self, "_cached_dt_host", None) is not None):
                dt_host = self._cached_dt_host[i]
                dt = dt_host
            else:
                dt = t_next - t

            if i in skip_steps and v_curr is not None:
                coeff = extrap_coeffs.get(i)
                if order >= 1 and v_prev is not None and coeff is not None and compute_count >= 2:
                    v_pred = v_curr + coeff * (v_curr - v_prev)
                    a_pred = a_curr + coeff * (a_curr - a_prev)
                else:
                    v_pred = v_curr
                    a_pred = a_curr

                video_latent = self.euler_step(
                    video_latent, v_pred, dt, dt_host)
                action_latent = self.euler_step(
                    action_latent, a_pred, dt, dt_host)
                video_latent = self.teacher_force_first_frame(
                    video_latent, cond_latent)
                continue

            video_tokens = self.prepare_video_tokens(video_latent)
            action_tokens = self.prepare_action_tokens(action_latent, state)
            und_tokens = self.extract_und_tokens(vlm_inputs)

            with torch.autocast(
                device_type="cuda",
                dtype=self.model.video_model.precision,
            ):
                if use_cached_t:
                    v_t_emb = self._cached_v_t_emb[i]
                    a_t_emb = self._cached_a_t_emb[i]
                    if use_cached_mod:
                        if getattr(self, "_cached_mod_partial", False):
                            v_adaln = self._cached_v_adaln[i]
                            a_adaln = self._cached_a_adaln[i]
                        else:
                            v_adaln = None
                            a_adaln = None
                    else:
                        v_adaln = self._cached_v_adaln[i]
                        a_adaln = self._cached_a_adaln[i]
                else:
                    video_t_scaled = (t * 1000).expand(B).to(self.dtype)
                    action_t_scaled = (t * 1000).expand(B).to(self.dtype)
                    v_t_emb, v_adaln = self.get_video_time_embeddings(
                        video_t_scaled, video_tokens.shape[1])
                    a_t_emb, a_adaln = self.get_action_time_embeddings(
                        action_t_scaled, action_tokens.shape[1])

                for L in range(self.dims.num_layers):
                    if use_cached_mod:
                        v_mod = self._cached_v_mod[i][L]
                        a_mod = self._cached_a_mod[i][L]
                        if v_mod is None:
                            v_mod = self.compute_video_adaln_modulation(
                                v_adaln, L)
                        if a_mod is None:
                            a_mod = self.compute_action_adaln_modulation(
                                a_adaln, L)
                        v_mod = self._maybe_materialize_static_mod(
                            v_mod, "video")
                        a_mod = self._maybe_materialize_static_mod(
                            a_mod, "action")
                    else:
                        v_mod = self.compute_video_adaln_modulation(v_adaln, L)
                        a_mod = self.compute_action_adaln_modulation(a_adaln, L)

                    video_tokens, action_tokens, und_tokens = (
                        self.denoise_layer_joint_attention(
                            video_tokens, action_tokens, und_tokens,
                            v_mod, a_mod, L))
                    video_tokens = self.denoise_layer_cross_attn_to_t5(
                        video_tokens, v_adaln, L, t5_ctx)
                    video_tokens = self.denoise_layer_ffn_video(
                        video_tokens, v_mod, L)
                    action_tokens = self.denoise_layer_ffn_action(
                        action_tokens, a_mod, L)
                    und_tokens = self.denoise_layer_ffn_und(und_tokens, L)

                video_velocity = self.video_output_head(video_tokens, v_t_emb)
                action_velocity = self.action_output_head(action_tokens, a_t_emb)

                # Maintain order-1 ring: prev := curr, curr := new velocity.
                if v_curr is None:
                    v_curr = torch.empty_like(video_velocity)
                    v_prev = torch.empty_like(video_velocity)
                    a_curr = torch.empty_like(action_velocity)
                    a_prev = torch.empty_like(action_velocity)
                    self._taylor_v_curr = v_curr
                    self._taylor_v_prev = v_prev
                    self._taylor_a_curr = a_curr
                    self._taylor_a_prev = a_prev
                v_prev.copy_(v_curr)
                a_prev.copy_(a_curr)
                v_curr.copy_(video_velocity)
                a_curr.copy_(action_velocity)
                compute_count += 1

                video_latent = self.euler_step(
                    video_latent, video_velocity, dt, dt_host)
                action_latent = self.euler_step(
                    action_latent, action_velocity, dt, dt_host)
                video_latent = self.teacher_force_first_frame(
                    video_latent, cond_latent)
        self._current_step_idx = None

        predicted_frames = self.decode_video(video_latent)
        predicted_actions = self.cast_bf16_to_float(action_latent)
        return predicted_frames, predicted_actions

    return run


def install_motus_taylorseer(pipeline) -> dict:
    if os.environ.get("FLASH_RT_MOTUS_USE_TAYLORSEER", "0") != "1":
        return {"enabled": False, "reason": "env_disabled"}
    if getattr(pipeline, "_taylorseer_installed", False):
        return {"enabled": True, "reason": "already_installed"}
    if getattr(pipeline, "_teacache_installed", False):
        return {"enabled": False, "reason": "teacache_already_installed"}
    if getattr(pipeline, "_easycache_installed", False):
        return {"enabled": False, "reason": "easycache_already_installed"}

    num_steps = pipeline.dims.num_inference_steps
    default = "2,3,4,5,6,7,8" if num_steps == 10 else ""
    skip_raw = os.environ.get(
        "FLASH_RT_MOTUS_TAYLORSEER_SKIP_STEPS", default)
    skip_steps = _parse_skip_steps(skip_raw, num_steps)
    if not skip_steps:
        return {"enabled": False, "reason": "no_valid_skip_steps",
                "raw": skip_raw}

    order = int(os.environ.get("FLASH_RT_MOTUS_TAYLORSEER_ORDER", "1"))
    order = max(0, min(order, 1))

    extrap_coeffs = _build_extrap_coeffs(num_steps, skip_steps)
    new_run = _make_taylorseer_run(skip_steps, order, extrap_coeffs)
    pipeline.run = types.MethodType(new_run, pipeline)
    pipeline._taylorseer_installed = True
    return {
        "enabled": True,
        "skip_steps": sorted(skip_steps),
        "compute_steps": sorted(set(range(num_steps)) - skip_steps),
        "order": order,
        "num_steps": num_steps,
        "extrap_coeffs": {k: round(v, 4) for k, v in extrap_coeffs.items()},
    }
