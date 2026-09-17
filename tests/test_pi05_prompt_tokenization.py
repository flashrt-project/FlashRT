"""The Pi0.5 frontend's SentencePiece fallback must tokenize prompts the
way openpi's PaligemmaTokenizer does, including for RL prompts that
carry a newline before the advantage tag.

CPU only; skipped unless a PaliGemma ``tokenizer.model`` is reachable
(``FLASH_RT_PALIGEMMA_TOKENIZER`` or the usual cache locations).
"""

import pytest


def _sp():
    try:
        from flash_rt.utils.paligemma_tokenizer import load_paligemma_sentencepiece

        return load_paligemma_sentencepiece()
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        pytest.skip(f"paligemma tokenizer unavailable: {exc}")


def _openpi_format(sp, prompt):
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    return sp.encode(cleaned, add_bos=True) + sp.encode("\n")


def _fallback_format(sp, prompt):
    from flash_rt.utils.paligemma_tokenizer import encode_pi05_prompt
    return encode_pi05_prompt(sp, prompt)


@pytest.mark.parametrize("prompt", [
    "pick up the cup",
    "put_the bowl on the plate ",
    "pick up the cup\nAdvantage: positive",
    "pick up the cup\nAdvantage: negative",
])
def test_fallback_tokens_match_openpi_format(prompt):
    sp = _sp()
    assert _fallback_format(sp, prompt) == _openpi_format(sp, prompt)


def test_newline_token_is_108():
    sp = _sp()
    assert sp.encode("\n") == [108]
