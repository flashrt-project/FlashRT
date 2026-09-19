"""FlashRT — Action post-processing utilities."""

import numpy as np

LIBERO_ACTION_DIM = 7


def unnormalize_actions(actions, norm_stats):
    """Unnormalize actions using q01/q99 statistics (pure numpy).

    Matches openpi.transforms.Unnormalize._unnormalize_quantile exactly:
    unnorm = (x + 1) / 2 * (q99 - q01) + q01, with NO clipping of the raw
    model output to [-1, 1] beforehand. Clipping first (as an earlier
    version of this function did) silently saturates any action dimension
    whose model output exceeds the training-data quantile range, which is
    common for pi0.5 policies and produces materially wrong actions.
    """
    q01 = np.array(norm_stats["actions"]["q01"], dtype=np.float32)
    q99 = np.array(norm_stats["actions"]["q99"], dtype=np.float32)
    dim = min(actions.shape[-1], len(q01))
    unnorm = actions.copy()
    unnorm[..., :dim] = (
        (actions[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim] + 1e-6)
        + q01[:dim]
    )
    return unnorm
