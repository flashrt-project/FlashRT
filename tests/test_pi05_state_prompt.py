import numpy as np


def test_pi05_state_prompt_matches_openpi_format():
    from flash_rt.core.utils.pi05_prompt import (
        discretize_pi05_state,
        format_pi05_prompt,
    )

    state = np.array([-1.0, 0.0, 1.0, 2.0, -2.0], dtype=np.float32)

    assert discretize_pi05_state(state).tolist() == [0, 128, 255, 255, 0]
    assert format_pi05_prompt("pick_up\nred", state) == (
        "Task: pick up red, State: 0 128 255 255 0;\nAction: "
    )


def test_pi05_state_prompt_without_state_keeps_text_only_format():
    from flash_rt.core.utils.pi05_prompt import format_pi05_prompt

    assert format_pi05_prompt(" pick_up\nred ") == "pick up red"
