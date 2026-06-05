"""Pi0.5 prompt helpers.

Pi0.5 follows the openpi convention where proprioceptive state is
discretized and represented in the language prefix:

    Task: <task>, State: <bins>;\nAction:

Keep this logic in one small module so RTX/Thor frontends and tests use the
same state formatting.
"""

from __future__ import annotations

import numpy as np


PI05_STATE_PROMPT_MAX_LEN = 200


def discretize_pi05_state(state) -> np.ndarray:
    """Discretize normalized Pi0.5 state to openpi's 256 language bins."""
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    bins = np.linspace(-1, 1, 256 + 1, dtype=np.float32)[:-1]
    tokens = np.digitize(arr, bins=bins) - 1
    return tokens.astype(np.int64)


def format_pi05_prompt(prompt: str, state=None) -> str:
    """Format a text prompt, optionally with Pi0.5 discrete state tokens."""
    cleaned = str(prompt).strip().replace("_", " ").replace("\n", " ")
    if state is None:
        return cleaned
    state_tokens = discretize_pi05_state(state)
    state_str = " ".join(map(str, state_tokens.tolist()))
    return f"Task: {cleaned}, State: {state_str};\nAction: "
