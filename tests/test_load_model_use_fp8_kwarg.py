from unittest.mock import patch
import ast
from pathlib import Path


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
