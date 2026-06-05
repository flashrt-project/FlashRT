import numpy as np


def test_pi05_state_prompt_matches_openpi_format():
    from flash_rt.core.utils.pi05_prompt import (
        discretize_pi05_state,
        format_pi05_prompt,
    )

    state = np.array([-1.0, 0.0, 1.0, 2.0, -2.0], dtype=np.float32)

    assert discretize_pi05_state(state).tolist() == [0, 128, 255, 255, -1]
    assert format_pi05_prompt("pick_up\nred", state) == (
        "Task: pick up red, State: 0 128 255 255 -1;\nAction: "
    )


def test_pi05_state_prompt_without_state_keeps_text_only_format():
    from flash_rt.core.utils.pi05_prompt import format_pi05_prompt

    assert format_pi05_prompt(" pick_up\nred ") == "pick up red"


def test_jax_prompt_embedding_formats_state(monkeypatch):
    from flash_rt.core import thor_frontend_utils

    seen = {}

    class FakeSentencePiece:
        def Encode(self, text, add_bos=False):
            seen["text"] = text
            seen["add_bos"] = add_bos
            return [0, 1, 2]

    monkeypatch.setattr(
        "flash_rt.utils.paligemma_tokenizer.load_paligemma_sentencepiece",
        lambda: FakeSentencePiece())
    embedding = np.ones((3, 4), dtype=np.float16)

    embeds, prompt_len = thor_frontend_utils.embed_prompt_numpy(
        "pick_up\nred", embedding, state=np.array([-1.0, 0.0, 1.0]))

    assert prompt_len == 3
    assert embeds.shape == (3, 4)
    assert seen["text"] == (
        "Task: pick up red, State: 0 128 255;\nAction: "
    )
    assert seen["add_bos"] is True
