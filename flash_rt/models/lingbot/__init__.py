"""FlashRT — LingBot-VLA model package.

Pipeline / weight-loading / kernel-orchestration code for LingBot-VLA
(Qwen2.5-VL-3B + Action Expert Qwen2-768, Mixed-Head joint attention,
50-step Euler flow matching). Hardware-specific dispatch lives in
``flash_rt.hardware.{thor,rtx}.attn_backend_lingbot``; frontends in
``flash_rt.frontends.{torch,jax}.lingbot_{thor,rtx}``.

status: pipeline + frontend stubs raise NotImplementedError;
``flash_rt.load_model(config="lingbot_vla")`` resolves dispatch and
reaches the stub boundary. Subsequent gates fill in the forward path.
"""
