"""FlashRT — LingBot-VLA model package (Thor / sm_110).

Pipeline / weight-loading / kernel-orchestration code for LingBot-VLA
(Qwen2.5-VL-3B + Action Expert Qwen2-768, Mixed-Head joint attention,
flow-matching denoise). FP8 backbone + FP4 gate_up; FA4 (optional) for the
prefix / denoise attention via ``flash_rt.hardware.thor.fa4_backend``.

Status: LingBot runs through the **low-level graph_runner path** —
``graph_runner.sample_actions_graph`` (weight spec + CUDA-graph capture). See
``examples/lingbot_quickstart.py`` and ``docs/lingbot_usage.md``.

It is **not** registered in ``_PIPELINE_MAP`` and ``flash_rt.load_model`` does
**not** dispatch a ``lingbot`` config yet; ``LingbotTorchFrontendThor``
(``flash_rt.frontends.torch.lingbot_thor``) is a scaffold whose methods
raise ``NotImplementedError``. Use the graph_runner entry point above, not
``flash_rt.load_model``.
"""
