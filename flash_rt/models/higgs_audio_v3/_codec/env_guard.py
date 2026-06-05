"""Import-time guards so the vendored Higgs codec loads on a stock install.

Two upstream pitfalls, both unrelated to the codec's decode path:

* ``kernels`` (an optional transformers acceleration package) constructs a
  ``huggingface_hub`` strict dataclass with a ``str | None`` field that some
  ``huggingface_hub`` versions reject at import, raising — not an ImportError —
  when transformers touches ``PreTrainedAudioTokenizerBase``. Neutralising the
  module makes transformers treat it as absent and degrade gracefully. The
  codec decoder is pure conv/transpose and never needs it.
* ``torchaudio`` is imported at the top of the tokenizer module but is only
  exercised by the *encode* path (resampling). Decode never calls it, so a stub
  with a valid ``__spec__`` (so transformers' ``find_spec`` probe succeeds)
  lets the module import without the dependency.

Call :func:`apply` before importing anything from ``transformers``.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types

_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return
    # Neutralise the optional `kernels` package (broken-or-absent both fine).
    if sys.modules.get("kernels") is None or "kernels" not in sys.modules:
        sys.modules["kernels"] = None  # type: ignore[assignment]
    # Stub torchaudio only if it is genuinely not installed.
    if "torchaudio" not in sys.modules and importlib.util.find_spec("torchaudio") is None:
        ta = types.ModuleType("torchaudio")
        ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
        ta.functional = types.ModuleType("torchaudio.functional")
        sys.modules["torchaudio"] = ta
        sys.modules["torchaudio.functional"] = ta.functional
    _applied = True
