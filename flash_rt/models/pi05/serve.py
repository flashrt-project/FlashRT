"""Switch a DM05 inference runtime over to this package's pipeline.

    from flash_rt.models.pi05 import serve
    native = serve.enable(inference_config, calibration_state=state)

``enable`` builds :class:`NativeDM05` on the runtime's model with the measured
default precision boundary, installs this package's preprocessing (cached prompt
tokenization, on-device image resize/normalize), routes ``model.inference_action``
through it, and replaces ``inference_config._predict`` with the host-staged predict
whose whole GPU side is one graph replay. Either fit the static quantization
parameters from ``calibration`` (an iterable of already-prepared model inputs, i.e.
``inference_config._prepare_model_input`` results) or load a state saved with
``torch.save(native.calibration_state(), path)``.

Why this function exists rather than callers wiring the pieces up themselves
----------------------------------------------------------------------------

``preprocess.install`` and ``NativeDM05.make_predict`` have to come from the *same*
module tree, and getting that wrong fails silently.

``RawAwareTokenization`` is defined once in ``preprocess.py``, so it is two distinct
class objects if two copies of that module are importable -- one here and one in the
repository this was ported from. ``install`` inserts an instance of whichever copy it
was imported from; ``make_predict`` does ``from .preprocess import ...`` and tests
with ``isinstance``. Mix them and the search for the tokenizer finds nothing, raising
a bare ``StopIteration`` three lines later from inside the predict path, with the
transform stack silently missing its tokenizer rather than reporting a mismatch.

``install`` makes that worse by returning ``False`` instead of raising when the
transform stack has no ``PixelTransform``/``ChatTokenization``, so a caller that
ignores the return value gets no signal at all. This function checks it.

Model handling
--------------

The ported-from runtime reached for its own LoRA-unwrapping helper here. That helper
belongs to the other repository and is deliberately not imported: pass the model via
``model``, or pass an ``unwrap`` callable to apply to ``inference_config.model``.
Without either, ``inference_config.model`` is used as-is, which is correct when the
runtime already holds an unwrapped model.
"""
from __future__ import annotations

import torch

from .pipeline import NativeConfig, NativeDM05
from . import preprocess

__all__ = ["enable"]


def enable(inference_config, calibration=None, calibration_state=None,
           config: NativeConfig | None = None, num_views: int = 2,
           model=None, unwrap=None) -> NativeDM05:
    """Install this package's DM05 pipeline into a live inference runtime.

    Args:
        inference_config: the runtime's inference config; ``_predict`` is replaced
            and ``input_transform`` is rewritten in place.
        calibration: iterable of prepared model inputs to fit the static scales from.
        calibration_state: a state dict, or a path to one saved by
            ``torch.save(native.calibration_state(), path)``. Takes precedence over
            ``calibration``; one of the two is required.
        config: precision boundary; ``NativeConfig()`` defaults are used if omitted.
        num_views: camera views the runtime feeds.
        model: the model to attach to. Defaults to ``inference_config.model``.
        unwrap: optional callable applied to ``inference_config.model`` to obtain the
            underlying model, for runtimes that wrap it.

    Returns:
        The installed :class:`NativeDM05`.
    """
    if model is None:
        model = inference_config.model
        if unwrap is not None:
            model = unwrap(model)

    native = NativeDM05(model, num_views=num_views)
    native.configure(config or NativeConfig())

    # Same tree as `make_predict` below, and checked: install() returns False rather
    # than raising when it does not recognise the transform stack.
    if not preprocess.install(inference_config):
        raise RuntimeError(
            "preprocess.install did not recognise inference_config.input_transform: "
            "expected PixelTransform and ChatTokenization among its transforms. "
            "Passing an uninstalled stack through is what produces a bare "
            "StopIteration from inside the predict path."
        )

    model.inference_action = native.inference_action
    reference_predict = inference_config._predict

    if calibration_state is not None:
        if not isinstance(calibration_state, dict):
            calibration_state = torch.load(calibration_state)
        native.load_calibration_state(calibration_state)
    elif calibration is not None:
        with torch.no_grad():
            native.calibrate(inference_config._predict, list(calibration))
    else:
        raise ValueError(
            "the pipeline needs calibration observations or a saved calibration state"
        )

    # The host-staged predict: whole GPU side in one graph replay. The calibration
    # above, if it ran, went through the reference _predict eagerly.
    inference_config._predict = native.make_predict(inference_config, reference_predict)
    return native
