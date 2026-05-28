from pathlib import Path

import numpy as np


def test_pi05_rtx_caches_recurring_state_prompt_lengths(monkeypatch):
    import torch
    from flash_rt.frontends.torch import pi05_rtx

    class FakePipeline:
        instances = []

        def __init__(self, *args, max_prompt_len, **kwargs):
            self.max_prompt_len = max_prompt_len
            self.use_int8_vision_static = False
            self.fp8_calibrated = False
            self.uploads = []
            type(self).instances.append(self)

        def set_language_embeds(self, embeds_np):
            self.uploads.append(embeds_np.shape)

    lengths = iter([17, 23, 17])

    def fake_embed_prompt(prompt_text, embedding_weight, max_len=48, state=None):
        prompt_len = next(lengths)
        return torch.zeros(prompt_len, 8, dtype=torch.bfloat16), prompt_len

    pipe = object.__new__(pi05_rtx.Pi05TorchFrontendRtx)
    pipe._rl_config = None
    pipe.embedding_weight = None
    pipe.max_prompt_len = 200
    pipe.num_views = 2
    pipe.chunk_size = 10
    pipe._num_steps = 10
    pipe._vision_pool_factor = 1
    pipe._vision_num_layers = 27
    pipe.pipeline = None
    pipe.current_prompt_len = 0
    pipe._prompt_pipeline_cache = {}
    pipe.graph_recorded = False
    pipe.calibrated = False
    pipe.gemm = None
    pipe.fvk = None
    pipe.attn_backend = None
    pipe._build_pipeline_weights = lambda: {}
    pipe._pipeline_precision_kwargs = lambda: {}

    monkeypatch.setattr(pi05_rtx, "_embed_prompt", fake_embed_prompt)
    monkeypatch.setattr(pi05_rtx, "Pi05Pipeline", FakePipeline)

    state = np.zeros(8, dtype=np.float32)
    pipe.set_prompt("pick up", state=state)
    first = pipe.pipeline
    pipe.set_prompt("pick up", state=state)
    second = pipe.pipeline
    pipe.set_prompt("pick up", state=state)

    assert [p.max_prompt_len for p in FakePipeline.instances] == [17, 23]
    assert pipe.pipeline is first
    assert pipe._prompt_pipeline_cache == {17: first, 23: second}
    assert first.uploads
    assert second.uploads


def test_groot_thor_rejects_prompt_changes_after_graph_build():
    source = Path("flash_rt/frontends/torch/groot_thor.py").read_text()
    set_prompt_pos = source.index("    def set_prompt(self, prompt):")
    guard_pos = source.index("getattr(self, '_graphs_built', False)",
                             set_prompt_pos)
    tokenizer_pos = source.index("from transformers import AutoTokenizer",
                                 set_prompt_pos)

    assert guard_pos < tokenizer_pos


def test_groot_thor_infer_refreshes_state_before_dit_graph_replay():
    source = Path("flash_rt/frontends/torch/groot_thor.py").read_text()
    infer_pos = source.index("    def infer(self, obs):")
    replay_pos = source.index("self._dit_graph.replay()", infer_pos)
    state_pos = source.index("_copy_state_feature_to_dit", infer_pos)

    assert state_pos < replay_pos


def test_groot_n17_rejects_second_prompt_runtime():
    source = Path("flash_rt/frontends/torch/groot_n17_thor.py").read_text()
    set_prompt_pos = source.index("    def set_prompt(\n")
    guard_pos = source.index('hasattr(self, "_backbone_features")',
                             set_prompt_pos)
    calibration_pos = source.index("from flash_rt.models.groot_n17",
                                   set_prompt_pos)

    assert guard_pos < calibration_pos
