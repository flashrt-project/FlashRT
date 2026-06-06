from unittest.mock import patch
import ast
from pathlib import Path

import pytest


def test_predict_forwards_state_to_prompt_and_observation():
    from flash_rt.api import VLAModel

    image0 = object()
    image1 = object()
    state = object()
    actions = object()

    class StateFrontend:
        prompt_state = None
        seen_obs = None

        def set_prompt(self, prompt, state=None):
            type(self).prompt_state = state

        def infer(self, obs):
            type(self).seen_obs = obs
            return {"actions": actions}

    model = VLAModel(StateFrontend(), framework="torch")
    result = model.predict(
        images=[image0, image1],
        prompt="pick up the red block",
        state=state,
    )

    assert result is actions
    assert StateFrontend.prompt_state is state
    assert StateFrontend.seen_obs["state"] is state
    assert StateFrontend.seen_obs["image"] is image0
    assert StateFrontend.seen_obs["wrist_image"] is image1


def test_predict_refreshes_prompt_when_prompt_state_changes():
    from flash_rt.api import VLAModel

    image = object()
    state0 = [0.0, 1.0]
    state1 = [1.0, 2.0]

    class TokenStateFrontend:
        prompt_states = []

        def set_prompt(self, prompt, state=None):
            type(self).prompt_states.append(list(state))

        def infer(self, obs):
            return {"actions": None}

    TokenStateFrontend.prompt_states = []
    model = VLAModel(TokenStateFrontend(), framework="torch")
    model.predict(images=[image], prompt="pick", state=state0)
    model.predict(images=[image], state=state0)
    model.predict(images=[image], state=state1)

    assert TokenStateFrontend.prompt_states == [state0, state1]


def test_predict_refreshes_prompt_when_prompt_state_is_removed():
    from flash_rt.api import VLAModel

    image = object()
    state0 = [0.0, 1.0]

    class TokenStateFrontend:
        prompt_states = []

        def set_prompt(self, prompt, state=None):
            type(self).prompt_states.append(
                None if state is None else list(state))

        def infer(self, obs):
            return {"actions": None}

    TokenStateFrontend.prompt_states = []
    model = VLAModel(TokenStateFrontend(), framework="torch")
    model.predict(images=[image], prompt="pick", state=state0)
    model.predict(images=[image], state=None)

    assert TokenStateFrontend.prompt_states == [state0, None]


def test_manual_set_prompt_tracks_prompt_state():
    from flash_rt.api import VLAModel

    image = object()
    state0 = [0.0, 1.0]

    class TokenStateFrontend:
        prompt_states = []

        def set_prompt(self, prompt, state=None):
            type(self).prompt_states.append(
                None if state is None else list(state))

        def infer(self, obs):
            return {"actions": None}

    TokenStateFrontend.prompt_states = []
    model = VLAModel(TokenStateFrontend(), framework="torch")
    model.set_prompt("pick", state=state0)
    model.predict(images=[image], state=None)

    assert TokenStateFrontend.prompt_states == [state0, None]


def test_predict_preserves_state_from_observation_dict():
    from flash_rt.api import VLAModel

    image = object()
    dict_state = object()
    kwarg_state = object()

    class ObservationFrontend:
        seen_obs = None

        def set_prompt(self, prompt):
            return None

        def infer(self, obs):
            type(self).seen_obs = obs
            return {"actions": None}

    model = VLAModel(ObservationFrontend(), framework="torch")
    model.predict(
        images={"image": image, "state": dict_state},
        prompt="pick up the red block",
        state=kwarg_state,
    )

    assert ObservationFrontend.seen_obs["state"] is dict_state
    assert ObservationFrontend.seen_obs["image"] is image


def test_load_model_only_passes_use_fp8_when_frontend_accepts_it():
    from flash_rt.api import load_model

    class NoUseFp8Frontend:
        def __init__(self, checkpoint, num_views=2):
            self.checkpoint = checkpoint
            self.num_views = num_views

        def infer(self, obs):
            return {"actions": None}

    with patch("flash_rt.hardware.detect_arch", return_value="rtx_sm120"), \
            patch("flash_rt.hardware.resolve_pipeline_class",
                  return_value=NoUseFp8Frontend):
        model = load_model(
            "/tmp/nonexistent", config="pi05", framework="torch",
            use_fp8=False)

    assert isinstance(model._pipe, NoUseFp8Frontend)


def test_load_model_propagates_use_fp8_when_frontend_accepts_it():
    from flash_rt.api import load_model

    class UseFp8Frontend:
        seen_use_fp8 = None

        def __init__(self, checkpoint, num_views=2, use_fp8=True):
            type(self).seen_use_fp8 = use_fp8

        def infer(self, obs):
            return {"actions": None}

    with patch("flash_rt.hardware.detect_arch", return_value="rtx_sm120"), \
            patch("flash_rt.hardware.resolve_pipeline_class",
                  return_value=UseFp8Frontend):
        model = load_model(
            "/tmp/nonexistent", config="pi05", framework="torch",
            use_fp8=False)

    assert isinstance(model._pipe, UseFp8Frontend)
    assert UseFp8Frontend.seen_use_fp8 is False


def test_load_model_propagates_hardware_when_frontend_accepts_it():
    from flash_rt.api import load_model

    class HardwareFrontend:
        seen_hardware = None

        def __init__(self, checkpoint, num_views=2, hardware=None):
            type(self).seen_hardware = hardware

        def infer(self, obs):
            return {"actions": None}

    with patch("flash_rt.hardware.resolve_pipeline_class",
              return_value=HardwareFrontend):
        model = load_model(
            "/tmp/nonexistent", config="pi05", framework="torch",
            hardware="rtx_sm89")

    assert isinstance(model._pipe, HardwareFrontend)
    assert HardwareFrontend.seen_hardware == "rtx_sm89"


def test_load_model_propagates_pi05_orin_tuning_kwargs_when_supported():
    from flash_rt.api import load_model

    class OrinTuningFrontend:
        seen = None

        def __init__(self, checkpoint, num_views=2, num_steps=10,
                     vision_pool_factor=1, vision_num_layers=27,
                     cache_frames=1):
            type(self).seen = {
                "num_steps": num_steps,
                "vision_pool_factor": vision_pool_factor,
                "vision_num_layers": vision_num_layers,
                "cache_frames": cache_frames,
            }

        def infer(self, obs):
            return {"actions": None}

    with patch("flash_rt.hardware.resolve_pipeline_class",
              return_value=OrinTuningFrontend):
        model = load_model(
            "/tmp/nonexistent", config="pi05", framework="torch",
            hardware="rtx_sm87", num_steps=5, vision_pool_factor=2,
            vision_num_layers=18, cache_frames=2)

    assert isinstance(model._pipe, OrinTuningFrontend)
    assert OrinTuningFrontend.seen == {
        "num_steps": 5,
        "vision_pool_factor": 2,
        "vision_num_layers": 18,
        "cache_frames": 2,
    }


def test_sm87_rejects_unvalidated_pi0_and_jax_backends():
    from flash_rt.hardware import resolve_pipeline_class

    for config, framework in [
        ("pi05", "jax"),
        ("pi0", "torch"),
        ("pi0", "jax"),
    ]:
        with pytest.raises(RuntimeError, match="Jetson Orin SM87"):
            resolve_pipeline_class(config, framework, "rtx_sm87")


def test_groot_n17_rtx_sm120_is_registered():
    from flash_rt.hardware import resolve_pipeline_class

    cls = resolve_pipeline_class("groot_n17", "torch", "rtx_sm120")
    assert cls.__name__ == "GrootN17TorchFrontendRtx"


def test_groot_n17_sm89_is_not_registered_without_validation():
    from flash_rt.hardware import resolve_pipeline_class

    with pytest.raises(RuntimeError, match="rtx_sm120"):
        resolve_pipeline_class("groot_n17", "torch", "rtx_sm89")


def test_wan22_ti2v_5b_rtx_sm120_is_registered():
    from flash_rt.hardware import resolve_pipeline_class

    cls = resolve_pipeline_class("wan22_ti2v_5b", "torch", "rtx_sm120")
    assert cls.__name__ == "Wan22TorchFrontendRtx"


def test_wan22_ti2v_5b_sm89_is_not_registered_without_validation():
    from flash_rt.hardware import resolve_pipeline_class

    with pytest.raises(RuntimeError, match="rtx_sm120"):
        resolve_pipeline_class("wan22_ti2v_5b", "torch", "rtx_sm89")


def test_load_model_accepts_wan22_ti2v_5b_config():
    from flash_rt.api import load_model

    class Wan22Frontend:
        seen = None

        def __init__(self, checkpoint, num_views=1, autotune=3):
            type(self).seen = {
                "checkpoint": checkpoint,
                "num_views": num_views,
                "autotune": autotune,
            }

        def set_prompt(self, *args, **kwargs):
            return None

        def infer(self, *args, **kwargs):
            return None

    with patch("flash_rt.hardware.resolve_pipeline_class",
              return_value=Wan22Frontend):
        model = load_model(
            "unused-checkpoint",
            config="wan22_ti2v_5b",
            framework="torch",
            hardware="rtx_sm120",
            num_views=1,
            autotune=0,
        )

    assert isinstance(model._pipe, Wan22Frontend)
    assert Wan22Frontend.seen == {
        "checkpoint": "unused-checkpoint",
        "num_views": 1,
        "autotune": 0,
    }


def test_wan22_infer_exposes_teacache_parameters():
    import inspect
    from flash_rt.frontends.torch.wan22_rtx import Wan22TorchFrontendRtx

    sig = inspect.signature(Wan22TorchFrontendRtx.infer)
    for name in (
        "teacache",
        "teacache_threshold",
        "teacache_start_step",
        "teacache_end_step",
        "teacache_cache_device",
    ):
        assert name in sig.parameters


def test_load_model_accepts_groot_n17_config():
    from flash_rt.api import load_model

    class GrootN17Frontend:
        seen = None

        def __init__(self, checkpoint, num_views=2, embodiment_tag=None):
            type(self).seen = {
                "checkpoint": checkpoint,
                "num_views": num_views,
                "embodiment_tag": embodiment_tag,
            }

        def set_prompt(self, *args, **kwargs):
            return None

        def infer(self, *args, **kwargs):
            return None

    with patch("flash_rt.hardware.resolve_pipeline_class",
              return_value=GrootN17Frontend):
        model = load_model(
            "unused-checkpoint",
            config="groot_n17",
            framework="torch",
            hardware="rtx_sm120",
            num_views=2,
            embodiment_tag="oxe_droid_relative_eef_relative_joint",
        )

    assert isinstance(model._pipe, GrootN17Frontend)
    assert GrootN17Frontend.seen == {
        "checkpoint": "unused-checkpoint",
        "num_views": 2,
        "embodiment_tag": "oxe_droid_relative_eef_relative_joint",
    }


def test_pi05_rtx_fp8_layout_selection():
    from flash_rt.frontends.torch.pi05_rtx import _select_fp8_layout

    assert _select_fp8_layout("rtx_sm89", None) == "nk"
    assert _select_fp8_layout("rtx_sm120", None) == "kn"
    assert _select_fp8_layout("rtx_sm120", "nk") == "nk"


def test_vla_frontend_constructors_accept_use_fp8():
    frontend_classes = {
        "flash_rt/frontends/torch/pi05_rtx.py": "Pi05TorchFrontendRtx",
        "flash_rt/frontends/jax/pi05_rtx.py": "Pi05JaxFrontendRtx",
        "flash_rt/frontends/torch/pi05_thor.py": "Pi05TorchFrontendThor",
        "flash_rt/frontends/jax/pi05_thor.py": "Pi05JaxFrontendThor",
        "flash_rt/frontends/torch/pi05_thor_fp4.py": "Pi05TorchFrontendThorFP4",
        "flash_rt/frontends/jax/pi05_thor_fp4.py": "Pi05JaxFrontendThorFP4",
        "flash_rt/frontends/torch/pi0_rtx.py": "Pi0TorchFrontendRtx",
        "flash_rt/frontends/jax/pi0_rtx.py": "Pi0JaxFrontendRtx",
        "flash_rt/frontends/torch/pi0_thor.py": "Pi0TorchFrontendThor",
        "flash_rt/frontends/jax/pi0_thor.py": "Pi0JaxFrontendThor",
        "flash_rt/frontends/torch/pi0fast.py": "Pi0FastTorchFrontend",
        "flash_rt/frontends/jax/pi0fast.py": "Pi0FastJaxFrontend",
        "flash_rt/frontends/torch/groot_rtx.py": "GrootTorchFrontendRtx",
        "flash_rt/frontends/torch/groot_thor.py": "GrootTorchFrontendThor",
    }

    repo_root = Path(__file__).resolve().parents[1]
    for rel_path, class_name in frontend_classes.items():
        tree = ast.parse((repo_root / rel_path).read_text())
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name)
        init = next(
            node for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        args = [arg.arg for arg in init.args.args]
        args += [arg.arg for arg in init.args.kwonlyargs]
        assert "use_fp8" in args, f"{class_name} must accept use_fp8"


def test_pi05_jax_rtx_frontend_mirrors_runtime_knobs():
    repo_root = Path(__file__).resolve().parents[1]
    tree = ast.parse(
        (repo_root / "flash_rt/frontends/jax/pi05_rtx.py").read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Pi05JaxFrontendRtx")
    init = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    args = [arg.arg for arg in init.args.args]
    assigned = set()
    for node in ast.walk(init):
        targets = list(getattr(node, "targets", []))
        if isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        for target in targets:
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                assigned.add(target.attr)

    for arg in (
        "num_steps",
        "vision_pool_factor",
        "vision_num_layers",
        "cache_frames",
    ):
        assert arg in args
    for attr in (
        "_num_steps",
        "_vision_pool_factor",
        "_vision_num_layers",
        "_cache_frames",
        "_frame_count",
        "_int8_weights",
        "_int8_weight_scales",
    ):
        assert attr in assigned
