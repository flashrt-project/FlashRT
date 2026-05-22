from unittest.mock import patch
import ast
from pathlib import Path

import pytest


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
