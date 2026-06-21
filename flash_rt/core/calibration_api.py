"""Shared ``calibrate()`` shim for single-sample implicit recalibration.

Some legacy frontend paths still use an implicit "first infer triggers
recalibration" mechanism for ``N == 1``. This helper turns that path into
the explicit public ``calibrate([obs])`` API shape. Frontends with native
multi-sample dataset calibration define their own ``calibrate`` methods and
call this helper only for their single-sample compatibility path.

Behaviour:
    N == 1 : run one forward via ``frontend.infer(obs)`` and discard the
             output. This fires whatever implicit calibration hook the
             frontend has (Thor's ``_recalibrate_with_real_data``
             auto-runs in the first infer).
    N >= 2 : raise NotImplementedError with a clear pointer that this
             compatibility shim is single-sample only.

New frontend code should implement a native ``calibrate`` method for any
``N >= 2`` dataset path and call this helper only for the legacy ``N == 1``
branch.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


_IMPLICIT_CALIBRATE_MULTI_SAMPLE_MESSAGE = (
    "Multi-sample (N>=2) dataset calibration is not supported by the "
    "implicit_calibrate compatibility shim. Frontends with native "
    "multi-sample calibration must route N>=2 to their own calibrate "
    "implementation. Pass a single observation for the implicit "
    "recalibration path, or use a frontend whose public calibrate() "
    "documents N>=2 support."
)


def implicit_calibrate(
    frontend: Any,
    observations: Iterable[Any],
    *,
    percentile: float = 99.9,
    max_samples: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Shim: force a single implicit recalibration via ``infer(obs_list[0])``.

    Raises NotImplementedError for N >= 2. Native multi-sample frontends
    should handle that path before calling this helper.
    """
    if isinstance(observations, dict):
        obs_list = [observations]
    elif isinstance(observations, list):
        obs_list = observations
    else:
        obs_list = list(observations)
    if max_samples is not None:
        obs_list = obs_list[:max_samples]
    n = len(obs_list)
    if n == 0:
        raise ValueError("observations must contain at least 1 sample")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")

    if n > 1:
        raise NotImplementedError(_IMPLICIT_CALIBRATE_MULTI_SAMPLE_MESSAGE)

    # Trigger implicit recalibration by running one inference and
    # discarding the result. The frontend's first-infer path will set
    # its ``_real_data_calibrated`` flag (or equivalent).
    frontend.infer(obs_list[0])
