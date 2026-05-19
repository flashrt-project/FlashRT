"""TeaCache (training-free step-level caching) swap for Motus.

PoC implementation. Caches video/action velocities at compute steps,
re-uses them at skip steps to bypass the 30-layer transformer + heads.

Schedule is BAKED IN at graph capture time via env
``FLASH_RT_MOTUS_TEACACHE_SKIP_STEPS`` (default 3,5,7 for 10-step run).
This is required because the captured CUDA graph cannot evaluate Python
branches per-replay.

Default OFF. Activate via:
    FLASH_RT_MOTUS_USE_TEACACHE=1

Cos red line: action >= 0.999, frames >= 0.99.
If violated, the method is dead — do not relax floor to mask quality loss.
"""

from __future__ import annotations

import os
import types
from typing import Set

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


def _make_teacache_run(orig_run, skip_steps: Set[int]):
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

        cached_v_vel = getattr(self, "_teacache_v_vel", None)
        cached_a_vel = getattr(self, "_teacache_a_vel", None)

        for i in range(self.dims.num_inference_steps):
            self._current_step_idx = i
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt_host = None
            if use_cached_t and getattr(self, "_cached_dt_host", None) is not None:
                dt_host = self._cached_dt_host[i]
                dt = dt_host
            else:
                dt = t_next - t

            if i in skip_steps and cached_v_vel is not None:
                video_latent = self.euler_step(
                    video_latent, cached_v_vel, dt, dt_host)
                action_latent = self.euler_step(
                    action_latent, cached_a_vel, dt, dt_host)
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

                if cached_v_vel is None:
                    cached_v_vel = torch.empty_like(video_velocity)
                    cached_a_vel = torch.empty_like(action_velocity)
                    self._teacache_v_vel = cached_v_vel
                    self._teacache_a_vel = cached_a_vel
                cached_v_vel.copy_(video_velocity)
                cached_a_vel.copy_(action_velocity)

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


def install_motus_teacache(pipeline) -> dict:
    """Monkey-patch pipeline.run to insert TeaCache schedule.

    Must be called BEFORE CUDA graph capture (i.e. before the first
    replayable infer). Returns stats dict {'enabled', 'skip_steps',
    'compute_steps'}.
    """
    if os.environ.get("FLASH_RT_MOTUS_USE_TEACACHE", "0") != "1":
        return {"enabled": False, "reason": "env_disabled"}
    if getattr(pipeline, "_teacache_installed", False):
        return {"enabled": True, "reason": "already_installed"}

    num_steps = pipeline.dims.num_inference_steps
    default = "2,3,4,5,6,7,8" if num_steps == 10 else ""
    skip_raw = os.environ.get("FLASH_RT_MOTUS_TEACACHE_SKIP_STEPS", default)
    skip_steps = _parse_skip_steps(skip_raw, num_steps)
    if not skip_steps:
        return {"enabled": False, "reason": "no_valid_skip_steps",
                "raw": skip_raw}

    orig_run = pipeline.run
    new_run = _make_teacache_run(orig_run.__func__, skip_steps)
    pipeline.run = types.MethodType(new_run, pipeline)
    pipeline._teacache_installed = True
    compute_steps = sorted(set(range(num_steps)) - skip_steps)
    return {
        "enabled": True,
        "skip_steps": sorted(skip_steps),
        "compute_steps": compute_steps,
        "num_steps": num_steps,
    }
