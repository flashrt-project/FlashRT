"""Pi0.5 prompt helpers.

Pi0.5 follows the openpi convention where proprioceptive state is
discretized and represented in the language prefix:

    Task: <task>, State: <bins>;\nAction:

When state-as-text is enabled, OpenPI (and the LeRobot port) zero-pad
the state to ``max_state_dim`` (32) before discretizing. A shorter state
is padded with zeros, which map to bin 128. Passing only the raw dimensions
changes the language prefix.

Keep this logic in one small module so RTX/Thor frontends and tests use the
same state formatting.
"""

from __future__ import annotations

import numpy as np


PI05_STATE_PROMPT_MAX_LEN = 200
PI05_STATE_DIM = 32  # openpi max_state_dim: zero-pad before discretizing


def discretize_pi05_state(state, state_dim: int = PI05_STATE_DIM) -> np.ndarray:
    """Discretize normalized Pi0.5 state to openpi's 256 language bins.

    The state is zero-padded to ``state_dim`` first (openpi's
    ``max_state_dim``); pass ``state_dim=None`` to skip the padding.
    """
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_dim is not None and arr.shape[0] < int(state_dim):
        arr = np.concatenate(
            [arr, np.zeros(int(state_dim) - arr.shape[0], dtype=np.float32)])
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
