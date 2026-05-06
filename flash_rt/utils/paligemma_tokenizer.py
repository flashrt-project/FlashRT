"""Locate the PaliGemma SentencePiece tokenizer.

Pi0 and Pi0.5 both prefix prompts via the PaliGemma SentencePiece
tokenizer (`paligemma_tokenizer.model`, ~4.1 MiB). The file is NOT
bundled inside the openpi pi0 / pi05 checkpoints — it lives in
Google's public big_vision storage bucket and the user must obtain
it once.

This module centralizes the lookup so failures surface a clear,
actionable error instead of the cryptic SentencePiece segfault that
occurs when `bos_id()` is called on an unloaded processor.

Resolution order (first hit wins):
    1. `$FLASH_RT_PALIGEMMA_TOKENIZER`            — explicit override
    2. `~/.cache/flash_rt/paligemma_tokenizer.model`
    3. `~/.cache/openpi/big_vision/paligemma_tokenizer.model`
       (compatible with `openpi.shared.download.maybe_download`)
    4. `/workspace/paligemma_tokenizer.model`
       (legacy dev container path)
    5. `openpi.models.tokenizer.PaligemmaTokenizer` — only used when
       `gcsfs` is installed and openpi can fetch from
       `gs://big_vision/paligemma_tokenizer.model`. The downloaded
       file is cached under `~/.cache/openpi/...` per openpi.

If none of the above succeeds, raise FileNotFoundError with the
download command spelled out — no hidden failure modes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_DOWNLOAD_URL = (
    "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
)

_DEFAULT_CACHE = Path.home() / ".cache" / "flash_rt"
_DEFAULT_PATH = _DEFAULT_CACHE / "paligemma_tokenizer.model"


def _candidate_paths() -> list[Path]:
    """Ordered list of paths to probe for an already-downloaded model."""
    out: list[Path] = []
    env = os.environ.get("FLASH_RT_PALIGEMMA_TOKENIZER")
    if env:
        out.append(Path(env).expanduser())
    out.append(_DEFAULT_PATH)
    out.append(
        Path.home() / ".cache" / "openpi" / "big_vision"
        / "paligemma_tokenizer.model",
    )
    out.append(Path("/workspace/paligemma_tokenizer.model"))
    return out


def _try_openpi_download() -> Optional[Path]:
    """Try openpi's auto-download path. Returns the cached file path on
    success, None on any failure (missing openpi, missing gcsfs,
    network error, etc.)."""
    try:
        from openpi.shared import download as _openpi_download
    except ImportError:
        return None
    try:
        local = _openpi_download.maybe_download(
            "gs://big_vision/paligemma_tokenizer.model",
            gs={"token": "anon"},
        )
    except Exception:
        # openpi reaches the GCS layer through fsspec / gcsfs. If
        # gcsfs isn't installed, fsspec raises ImportError; on
        # network failure it's some IOError. Either way, fall
        # through to the structured error message below — never
        # let the silent-tokenizer-segfault path win.
        return None
    return Path(str(local))


def _format_help_message(probed: list[Path]) -> str:
    probed_str = "\n".join(f"    {p}" for p in probed)
    return (
        "paligemma_tokenizer.model not found.\n\n"
        "Pi0 / Pi0.5 prefix prompts via the PaliGemma SentencePiece\n"
        "tokenizer; this file is NOT shipped with openpi checkpoints.\n"
        "Download it once with:\n\n"
        f"    mkdir -p {_DEFAULT_CACHE}\n"
        f"    curl -L {_DOWNLOAD_URL} \\\n"
        f"         -o {_DEFAULT_PATH}\n\n"
        "Or set $FLASH_RT_PALIGEMMA_TOKENIZER to a local copy.\n\n"
        "Searched:\n"
        f"{probed_str}\n\n"
        "See docs USAGE.md → 'PaliGemma tokenizer setup' for details."
    )


def resolve_paligemma_tokenizer_path() -> str:
    """Return the absolute path to a usable paligemma_tokenizer.model.

    Raises:
      FileNotFoundError with a clear download command if no candidate
      path resolves and openpi auto-download is not available.
    """
    probed = _candidate_paths()
    for p in probed:
        if p.is_file():
            return str(p.resolve())

    # None of the on-disk candidates exist. Try openpi auto-download
    # (which itself caches under ~/.cache/openpi). If openpi or
    # gcsfs is missing, this returns None — fall through to the
    # structured error message rather than letting the loader
    # segfault on bos_id().
    downloaded = _try_openpi_download()
    if downloaded is not None and downloaded.is_file():
        return str(downloaded.resolve())

    raise FileNotFoundError(_format_help_message(probed))


def load_paligemma_sentencepiece():
    """Return a loaded `sentencepiece.SentencePieceProcessor` for the
    PaliGemma model. Raises FileNotFoundError with a helpful download
    command if the model file can't be located.
    """
    import sentencepiece as spm
    path = resolve_paligemma_tokenizer_path()
    sp = spm.SentencePieceProcessor()
    if not sp.Load(path):
        raise RuntimeError(
            f"sentencepiece failed to load {path!r} — file is corrupted "
            "or not a valid SentencePiece model. Re-download via:\n"
            f"  curl -L {_DOWNLOAD_URL} -o {path}"
        )
    return sp
